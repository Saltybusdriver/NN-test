#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <cstring>

// Project Headers
#include "cpu_california_regression.h"
#include "DataLoader.h"
#include "California_Dataset.h"
#include "NeuralNetwork.h"
#include "optimizer.h"
#include "utils.h"
#include "layer_proxy.h"

namespace cpu_california_regression
{

// --- Adam Defaults ---
const float ADAM_BETA1 = 0.9f;
const float ADAM_BETA2 = 0.999f;
const float ADAM_EPSILON = 1e-8f;

// --- Main Training Function (CPU version) ---
void train_california_regressor(
    int batch_size, int num_epochs,
    const std::string& dataset_path,
    float learning_rate ,
    const std::string& loss_type_str )
{
    // --- Setup CPU Mode ---
    bool use_cuda = false;
    SetUseCuda(use_cuda);
    std::cout << "\n=== California Housing Regression Training (CPU) ===\n";
    std::cout << "Using CUDA: No" << std::endl;
    std::cout << "Epochs: " << num_epochs << ", Batch Size: " << batch_size
              << ", Learning Rate: " << learning_rate
              << ", Loss: " << loss_type_str << std::endl;
    std::cout << "Dataset Path: " << dataset_path << std::endl;

    // --- Timing Setup (CPU) ---
    auto program_start = std::chrono::high_resolution_clock::now();
    auto train_start_overall = std::chrono::high_resolution_clock::now();
    auto test_start_overall = std::chrono::high_resolution_clock::now();
    std::vector<double> epoch_times_s;
    size_t total_train_batches_processed = 0;
    double total_data_load_s = 0.0;
    double total_forward_s = 0.0;
    double total_loss_compute_s = 0.0;
    double total_loss_sum_s = 0.0;
    double total_backward_s = 0.0;
    double total_update_s = 0.0;

    // --- Create Datasets ---
    std::cout << "Loading full dataset..." << std::endl;
    auto dataset_load_start = std::chrono::high_resolution_clock::now();
    std::shared_ptr<CaliforniaDataset> full_dataset_unnormalized;
    try
    {
        full_dataset_unnormalized = std::make_shared<CaliforniaDataset>(dataset_path, false);
        auto dataset_load_end = std::chrono::high_resolution_clock::now();
        auto dataset_load_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(dataset_load_end - dataset_load_start);
        std::cout << "Full dataset loaded successfully in " << dataset_load_duration_ms.count() << " ms." << std::endl;
        if (full_dataset_unnormalized->size() == 0) throw std::runtime_error("Loaded dataset is empty.");
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error loading full California dataset: " << e.what() << std::endl;
        return;
    }

    // --- Perform Train/Test Split ---
    size_t total_samples = full_dataset_unnormalized->size();
    size_t test_size = static_cast<size_t>(total_samples * 0.2);
    size_t train_size = total_samples - test_size;
    if (train_size == 0 || test_size == 0)
    {
        std::cerr << "Error: Train/test split resulted in zero samples." << std::endl;
        return;
    }
    std::cout << "Splitting data: Train=" << train_size << ", Test=" << test_size << std::endl;
    std::vector<std::vector<float>> all_features = full_dataset_unnormalized->get_all_features();
    std::vector<float> all_targets = full_dataset_unnormalized->get_all_targets();
    std::vector<size_t> indices(total_samples);
    std::iota(indices.begin(), indices.end(), 0);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine(seed));
    std::vector<std::vector<float>> train_features_vec, test_features_vec;
    std::vector<float> train_targets_vec, test_targets_vec;
    train_features_vec.reserve(train_size); train_targets_vec.reserve(train_size);
    test_features_vec.reserve(test_size); test_targets_vec.reserve(test_size);
    for (size_t i = 0; i < train_size; ++i) { train_features_vec.push_back(all_features[indices[i]]); train_targets_vec.push_back(all_targets[indices[i]]); }
    for (size_t i = train_size; i < total_samples; ++i) { test_features_vec.push_back(all_features[indices[i]]); test_targets_vec.push_back(all_targets[indices[i]]); }

    // --- Create Train Dataset (Calculate & Apply Normalization) ---
    std::shared_ptr<CaliforniaDataset> train_dataset;
    try
    {
        train_dataset = std::make_shared<CaliforniaDataset>(train_features_vec, train_targets_vec, true);
    }
    catch (const std::exception& e) { std::cerr << "Error creating training dataset: " << e.what() << std::endl; return; }
    std::vector<float> train_mean_f = train_dataset->get_feature_mean();
    std::vector<float> train_std_f = train_dataset->get_feature_std();
    float train_mean_t = train_dataset->get_target_mean();
    float train_std_t = train_dataset->get_target_std();

    // --- Create Test Dataset (Apply Training Normalization) ---
    std::shared_ptr<CaliforniaDataset> test_dataset;
     try
     {
        test_dataset = std::make_shared<CaliforniaDataset>(test_features_vec, test_targets_vec, true, train_mean_f, train_std_f, train_mean_t, train_std_t);
    }
    catch (const std::exception& e) { std::cerr << "Error creating test dataset: " << e.what() << std::endl; return; }

    // --- Create DataLoaders (using the split datasets) ---
    DataLoader train_loader(train_dataset, batch_size, true, 4);
    DataLoader test_loader(test_dataset, batch_size, false, 2);
    size_t total_train_batches_per_epoch = train_loader.get_num_batches();
    size_t total_test_batches = test_loader.get_num_batches();
    std::cout << "Train batches/epoch: " << total_train_batches_per_epoch
              << ", Test batches: " << total_test_batches << std::endl;

    // --- Create Network, Optimizer, Loss ---
    size_t num_features = train_dataset->feature_size();
    const size_t output_size = train_dataset->target_size();

    NeuralNetwork network(batch_size);

    try
    {
        network.add_layer(Linear(num_features, 128, batch_size, "relu"));
        network.add_layer(Linear(128, 64, batch_size, "relu"));
        network.add_layer(Linear(64, output_size, batch_size, "none"));

        if (network.layers.empty())
        {
             throw std::runtime_error("Network definition is empty.");
        }
    }
    catch (const std::exception& e)
    {
         std::cerr << "Error building network: " << e.what() << std::endl;
         return;
    }

    // --- Optimizer ---
    std::unique_ptr<OptimizerBase> optimizer = create_optimizer("adam", learning_rate, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON);
    std::cout << "Using Adam Optimizer (LR=" << learning_rate << ")" << std::endl;

    // --- Loss ---
    const char* loss_type = loss_type_str.c_str();
    std::cout << "Using Loss: " << loss_type << std::endl;

    // --- Allocate Loss Buffers (Host only) ---
    size_t max_loss_elements = batch_size;
    std::vector<float> h_element_loss(max_loss_elements);
    float* loss_buffer_ptr = h_element_loss.data();

    // --- Training Loop ---
    std::cout << "Starting training..." << std::endl;
    std::cout << std::fixed << std::setprecision(6);

    train_start_overall = std::chrono::high_resolution_clock::now();

    std::shared_ptr<Batch> batch;
    for (int epoch = 0; epoch < num_epochs; ++epoch)
    {
        auto epoch_start_time = std::chrono::high_resolution_clock::now();
        std::cout << "\n--- Epoch " << (epoch + 1) << "/" << num_epochs << " ---" << std::endl;

        float epoch_loss_sum = 0.0f;
        size_t batches_processed_this_epoch = 0;
        train_loader.reset();

        double epoch_data_load_s = 0.0;
        double epoch_forward_s = 0.0;
        double epoch_loss_compute_s = 0.0;
        double epoch_loss_sum_s = 0.0;
        double epoch_backward_s = 0.0;
        double epoch_update_s = 0.0;

        while ((batch = train_loader.next_batch()) != nullptr)
        {
            auto dload_start = std::chrono::high_resolution_clock::now();
            if (!batch || batch->batch_size == 0) continue;
            auto dload_end = std::chrono::high_resolution_clock::now();
            epoch_data_load_s += std::chrono::duration<double>(dload_end - dload_start).count();

            float* inputs = batch->inputs_flattened.data();
            float* targets = batch->targets_flattened.data();
            size_t current_batch_size = batch->batch_size;
            size_t current_output_elements = current_batch_size * output_size;

            if (!inputs || !targets)
            {
                 std::cerr << "Warning: Skipping batch with null data pointers." << std::endl;
                 continue;
            }

            auto fwd_chrono_start = std::chrono::high_resolution_clock::now();
            network.forward(inputs, current_batch_size, false);
            auto fwd_chrono_end = std::chrono::high_resolution_clock::now();
            epoch_forward_s += std::chrono::duration<double>(fwd_chrono_end - fwd_chrono_start).count();

            float* outputs = network.layers.back()->output;
            if (!outputs)
            {
                 std::cerr << "Warning: Skipping batch due to null network output." << std::endl;
                 continue;
            }

            auto loss_comp_chrono_start = std::chrono::high_resolution_clock::now();
            if (h_element_loss.size() < current_batch_size)
            {
                h_element_loss.resize(current_batch_size);
                loss_buffer_ptr = h_element_loss.data();
            }
            network.compute_loss(outputs, targets, loss_buffer_ptr, current_batch_size, loss_type);
            auto loss_comp_chrono_end = std::chrono::high_resolution_clock::now();
            epoch_loss_compute_s += std::chrono::duration<double>(loss_comp_chrono_end - loss_comp_chrono_start).count();

            float batch_loss_sum = 0.0f;
            auto loss_sum_start = std::chrono::high_resolution_clock::now();
            batch_loss_sum = std::accumulate(h_element_loss.begin(), h_element_loss.begin() + current_batch_size, 0.0f);
            auto loss_sum_end = std::chrono::high_resolution_clock::now();
            epoch_loss_sum_s += std::chrono::duration<double>(loss_sum_end - loss_sum_start).count();

            float batch_avg_loss = (current_batch_size > 0) ? (batch_loss_sum / current_batch_size) : 0.0f;

            auto bwd_chrono_start = std::chrono::high_resolution_clock::now();
            network.backward(targets, 0.0f, loss_type);
            auto bwd_chrono_end = std::chrono::high_resolution_clock::now();
            epoch_backward_s += std::chrono::duration<double>(bwd_chrono_end - bwd_chrono_start).count();

            auto update_chrono_start = std::chrono::high_resolution_clock::now();
            for (auto& layer : network.layers)
            {
                layer->update_params(*optimizer);
            }
            auto update_chrono_end = std::chrono::high_resolution_clock::now();
            epoch_update_s += std::chrono::duration<double>(update_chrono_end - update_chrono_start).count();

            epoch_loss_sum += batch_avg_loss;
            batches_processed_this_epoch++;
            total_train_batches_processed++;

            if (batches_processed_this_epoch % 50 == 0 || batches_processed_this_epoch == 1)
            {
                 std::cout << "  Batch " << std::setw(4) << batches_processed_this_epoch << "/" << train_loader.get_num_batches()
                           << " AvgLoss: " << std::fixed << std::setprecision(6) << batch_avg_loss << "\r" << std::flush;
            }
        }
        std::cout << std::endl;

        float avg_epoch_loss = 0.0f;
        if (batches_processed_this_epoch > 0)
        {
            avg_epoch_loss = epoch_loss_sum / batches_processed_this_epoch;
            std::cout << "  Train Loss (" << loss_type << "): " << std::fixed << std::setprecision(6) << avg_epoch_loss << std::endl;
        }
        else
        {
            std::cout << "  No training batches processed this epoch." << std::endl;
        }

        auto epoch_end_time = std::chrono::high_resolution_clock::now();
        double epoch_duration_s_cpu = std::chrono::duration<double>(epoch_end_time - epoch_start_time).count();
        epoch_times_s.push_back(epoch_duration_s_cpu);
        std::cout << "  Epoch Time: " << std::fixed << std::setprecision(6) << epoch_times_s.back() << " s" << std::endl;

        if (batches_processed_this_epoch > 0)
        {
            std::cout << "  Avg Batch Timings (ms): "
                      << "Load: " << (epoch_data_load_s * 1000.0 / batches_processed_this_epoch)
                      << " | Fwd: "<< (epoch_forward_s * 1000.0 / batches_processed_this_epoch)
                      << " | LossComp: " << (epoch_loss_compute_s * 1000.0 / batches_processed_this_epoch)
                      << " | LossSum: "  << (epoch_loss_sum_s * 1000.0 / batches_processed_this_epoch)
                      << " | Bwd: " << (epoch_backward_s * 1000.0 / batches_processed_this_epoch)
                      << " | Update: " << (epoch_update_s * 1000.0 / batches_processed_this_epoch)
                      << std::endl;
        }

        total_data_load_s += epoch_data_load_s;
        total_forward_s += epoch_forward_s;
        total_loss_compute_s += epoch_loss_compute_s;
        total_loss_sum_s += epoch_loss_sum_s;
        total_backward_s += epoch_backward_s;
        total_update_s += epoch_update_s;

        float current_cpu_mem_epoch = get_peak_cpu_memory_mb();
        if (current_cpu_mem_epoch >= 0)
        {
             std::cout << "  CPU Mem (Peak MB): " << current_cpu_mem_epoch << std::endl;
        }
        else
        {
             std::cout << "  CPU Mem (Peak MB): N/A" << std::endl;
        }

    }

    auto train_end_overall = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> train_duration_overall_ms = train_end_overall - train_start_overall;
    double cpu_train_time_ms = train_duration_overall_ms.count();
    std::cout << "\nTraining complete! Total CPU Training Time: " << cpu_train_time_ms << " ms\n";

    // --- Testing Phase ---
    std::cout << "\n--- Testing (CPU) ---" << std::endl;
    test_start_overall = std::chrono::high_resolution_clock::now();

    float test_loss_sum_total = 0.0f;
    size_t test_samples = 0;
    size_t test_batches_processed = 0;
    test_loader.reset();

    std::vector<float> all_test_preds_denorm;
    std::vector<float> all_test_targets_denorm;

    std::shared_ptr<Batch> test_batch;
    while ((test_batch = test_loader.next_batch()) != nullptr)
    {
         if (!test_batch || test_batch->batch_size == 0) continue;

         float* inputs = test_batch->inputs_flattened.data();
         float* targets = test_batch->targets_flattened.data();
         size_t current_batch_size = test_batch->batch_size;
         size_t current_output_elements = current_batch_size * output_size;

         if (!inputs || !targets) continue;

         network.forward(inputs, current_batch_size, false);
         float* outputs = network.layers.back()->output;
         if (!outputs) continue;

         if (h_element_loss.size() < current_batch_size)
         {
             h_element_loss.resize(current_batch_size);
             loss_buffer_ptr = h_element_loss.data();
         }
         network.compute_loss(outputs, targets, loss_buffer_ptr, current_batch_size, loss_type);

         float batch_loss_sum = std::accumulate(h_element_loss.begin(), h_element_loss.begin() + current_batch_size, 0.0f);
         test_loss_sum_total += batch_loss_sum;
         test_samples += current_batch_size;
         test_batches_processed++;

         std::vector<float> h_outputs(outputs, outputs + current_output_elements);
         std::vector<float> h_targets(targets, targets + current_output_elements);

         std::vector<float> batch_preds_denorm = test_dataset->denormalize_targets(h_outputs);
         std::vector<float> batch_targets_denorm = test_dataset->denormalize_targets(h_targets);
         all_test_preds_denorm.insert(all_test_preds_denorm.end(), batch_preds_denorm.begin(), batch_preds_denorm.end());
         all_test_targets_denorm.insert(all_test_targets_denorm.end(), batch_targets_denorm.begin(), batch_targets_denorm.end());

    }

    auto test_end_overall = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> test_duration_overall_ms = test_end_overall - test_start_overall;
    double cpu_test_time_ms = test_duration_overall_ms.count();

    float final_test_loss = 0.0f;
    float final_mae = 0.0f;
    if (test_samples > 0)
    {
        final_test_loss = test_loss_sum_total / test_samples;
        std::cout << "Final Test Loss (" << loss_type << "): " << final_test_loss << std::endl;

        double mae_sum = 0.0;
        for (size_t i = 0; i < all_test_targets_denorm.size(); ++i)
        {
            mae_sum += std::abs(all_test_preds_denorm[i] - all_test_targets_denorm[i]);
        }
        final_mae = mae_sum / all_test_targets_denorm.size();
        std::cout << "Final Test MAE (denormalized): " << final_mae << std::endl;

    }
    else
    {
        std::cout << "No test samples were processed." << std::endl;
    }
    std::cout << "CPU Testing Time: " << cpu_test_time_ms << " ms\n";

    double avg_latency_ms = (total_train_batches_processed > 0 && cpu_train_time_ms > 0) ? (cpu_train_time_ms / total_train_batches_processed) : 0.0;
    double throughput_samples_s = (cpu_train_time_ms > 0) ? (static_cast<double>(test_dataset->size() * num_epochs) / (cpu_train_time_ms / 1000.0)) : 0.0;
    double avg_epoch_time_s = !epoch_times_s.empty() ? (std::accumulate(epoch_times_s.begin(), epoch_times_s.end(), 0.0) / epoch_times_s.size()) : 0.0;

    float peak_cpu_mem = get_peak_cpu_memory_mb();

    std::cout << "\n--- Final Summary (CPU) ---" << std::endl;
    if (test_samples > 0)
    {
        std::cout << "Test Loss (" << loss_type << "):      " << final_test_loss << std::endl;
        std::cout << "Test MAE (denormalized): " << final_mae << std::endl;
    }
    else
    {
        std::cout << "Test Loss:               N/A" << std::endl;
        std::cout << "Test MAE:                N/A" << std::endl;
    }
    std::cout << "--- Performance Metrics (CPU) ---" << std::endl;
    std::cout << "Total CPU Train Time (ms): "  << cpu_train_time_ms << std::endl;
    std::cout << "Total CPU Test Time (ms):  "  << cpu_test_time_ms << std::endl;
    std::cout << "Avg. Latency/Batch (ms): "  << avg_latency_ms << std::endl;
    std::cout << "Throughput (Samples/sec):" << throughput_samples_s << std::endl;
    std::cout << "Avg. Time per Epoch (s): "  << avg_epoch_time_s << std::endl;
    if (peak_cpu_mem >= 0)
    {
        std::cout << "Peak CPU Memory (MB):    "<< peak_cpu_mem << " (Linux VmHWM)" << std::endl;
    }
    else
    {
        std::cout << "Peak CPU Memory (MB):    N/A" << std::endl;
    }
    if (total_train_batches_processed > 0)
    {
        std::cout << "--- Avg Batch Breakdown (ms) ---" << std::endl;
        std::cout << "  Data Load: " << (total_data_load_s * 1000.0 / total_train_batches_processed) << std::endl;
        std::cout << "  Forward:   "  << (total_forward_s * 1000.0 / total_train_batches_processed) << std::endl;
        std::cout << "  Loss Comp: "  << (total_loss_compute_s * 1000.0 / total_train_batches_processed) << std::endl;
        std::cout << "  Loss Sum:  "  << (total_loss_sum_s * 1000.0 / total_train_batches_processed) << std::endl;
        std::cout << "  Backward:  "  << (total_backward_s * 1000.0 / total_train_batches_processed) << std::endl;
        std::cout << "  Update:    "  << (total_update_s * 1000.0 / total_train_batches_processed) << std::endl;
    }

    std::cout << "\n--- Sample Test Predictions (Denormalized - CPU) ---" << std::endl;
    if (!all_test_targets_denorm.empty())
    {
        std::cout << "Sample | Predicted   | Actual      | Difference" << std::endl;
        std::cout << "------------------------------------------------" << std::endl;
        int samples_to_show = std::min((int)all_test_targets_denorm.size(), 10);
        std::cout << std::fixed << std::setprecision(2);
        for (int i = 0; i < samples_to_show; ++i)
        {
            float pred = all_test_preds_denorm[i];
            float actual = all_test_targets_denorm[i];
            std::cout << std::setw(6) << i << " | "
                      << std::setw(11) << pred << " | "
                      << std::setw(11) << actual << " | "
                      << std::setw(11) << (pred - actual) << std::endl;
        }
        std::cout << "------------------------------------------------" << std::endl;
        std::cout << std::fixed << std::setprecision(6);
    }
    else
    {
        std::cout << "No test samples available for prediction output." << std::endl;
    }

    std::cout << "\nCleaning up California Regression CPU resources...\n";
    for (auto& layer : network.layers)
    {
        delete layer;
    }
    network.layers.clear();

    auto program_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_duration_ms = program_end - program_start;
    std::cout << "Total California Regression CPU function time: " << total_duration_ms.count() << " ms\n";
    std::cout << "============================================\n";
}

} // namespace cpu_california_regression