#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <cstring>

// Project Headers
#include "cpu_mnist_train.h"
#include "DataLoader.h"
#include "MnistDataset.h"
#include "NeuralNetwork.h"
#include "optimizer.h"
#include "debug.h"
#include "utils.h"
#include "layer_proxy.h"

namespace cpu_mnist_train
{

// --- Constants specific to MNIST ---
const int INPUT_WIDTH = 28;
const int INPUT_HEIGHT = 28;
const int INPUT_CHANNELS = 1;
const int NUM_CLASSES = 10;
const size_t INPUT_SIZE = INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS;

// --- Adam Defaults ---
const float ADAM_BETA1 = 0.9f;
const float ADAM_BETA2 = 0.999f;
const float ADAM_EPSILON = 1e-8f;

// --- Helper Function (CPU version) ---
float calculate_accuracy(const float* predictions, const float* targets, size_t batch_size, int num_classes)
{
    int correct = 0;
    for (size_t i = 0; i < batch_size; ++i)
    {
        const float* current_pred = predictions + i * num_classes;
        const float* current_target = targets + i * num_classes;
        int pred_idx = std::distance(current_pred, std::max_element(current_pred, current_pred + num_classes));
        int target_idx = std::distance(current_target, std::max_element(current_target, current_target + num_classes));
        if (pred_idx == target_idx)
        {
            correct++;
        }
    }
    return (batch_size > 0) ? static_cast<float>(correct) / batch_size : 0.0f;
}


void print_mnist_predictions(
    const float* targets_host,
    const float* outputs_host,
    int num_samples_to_show,
    int batch_size,
    int num_classes)
{
    if (!targets_host || !outputs_host)
    {
        std::cerr << "Warning: Cannot print predictions due to null host data pointers." << std::endl;
        return;
    }

    num_samples_to_show = std::min(num_samples_to_show, batch_size);
    std::cout << "\n--- Sample MNIST Predictions (CPU) ---" << std::endl;
    std::cout << "Sample | True Label | Predicted Label | Output Scores (Optional)" << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    for (int i = 0; i < num_samples_to_show; ++i)
    {
        const float* current_target = targets_host + i * num_classes;
        const float* current_output = outputs_host + i * num_classes;

        int true_label = std::distance(current_target, std::max_element(current_target, current_target + num_classes));
        int predicted_label = std::distance(current_output, std::max_element(current_output, current_output + num_classes));

        std::cout << std::setw(6) << i << " | "
                  << std::setw(10) << true_label << " | "
                  << std::setw(15) << predicted_label << " | ";

        std::cout << "[";
        for(int j=0; j<num_classes; ++j)
        {
            std::cout << std::fixed << std::setprecision(2) << current_output[j] << (j == num_classes - 1 ? "" : ", ");
        }
        std::cout << "]" << std::endl;

    }
    std::cout << "----------------------------------------------------------------" << std::endl;
}


// --- Main Training Function (CPU version) ---
void train_mnist_classifier(
    int batch_size, int num_epochs,
        const std::string& dataset_path,
        int image_height, int image_width,
        bool use_grayscale)
{
    // --- Parameter Validation ---
    if (!use_grayscale)
    {
    }
    if (image_height != INPUT_HEIGHT || image_width != INPUT_WIDTH)
    {
         std::cerr << "Warning: Image dimensions should be " << INPUT_HEIGHT << "x" << INPUT_WIDTH
                   << " for MNIST. Using constants." << std::endl;
    }
    use_grayscale = true;

    // --- Setup CPU Mode ---
    bool use_cuda = false;
    SetUseCuda(use_cuda);
    int epochs=num_epochs;
    float learning_rate=0.001f;

    std::cout << "\n=== MNIST Classifier Training (CPU) ===\n";
    std::cout << "Using CUDA: No" << std::endl;
    std::cout << "Epochs: " << epochs << ", Batch Size: " << batch_size
              << ", Learning Rate: " << learning_rate << std::endl;
    std::cout << "Dataset Path: " << dataset_path<< std::endl;

    // --- Timing Setup (CPU) ---
    auto program_start = std::chrono::high_resolution_clock::now();
    auto train_start_overall = std::chrono::high_resolution_clock::now();
    auto test_start_overall = std::chrono::high_resolution_clock::now();

    // --- Create Datasets ---
    std::cout << "Loading datasets..." << std::endl;
    std::shared_ptr<MnistDataset> train_dataset;
    std::shared_ptr<MnistDataset> val_dataset;
    std::shared_ptr<MnistDataset> test_dataset;
    try
    {
        train_dataset = std::make_shared<MnistDataset>(dataset_path, "training", INPUT_WIDTH, INPUT_HEIGHT);
        val_dataset = std::make_shared<MnistDataset>(dataset_path, "testing", INPUT_WIDTH, INPUT_HEIGHT);
        test_dataset = std::make_shared<MnistDataset>(dataset_path, "testing", INPUT_WIDTH, INPUT_HEIGHT);

        std::cout << "Training set size: " << train_dataset->size() << std::endl;
        std::cout << "Validation set size: " << val_dataset->size() << std::endl;
        std::cout << "Test set size: " << test_dataset->size() << std::endl;

        if (train_dataset->size() == 0 || test_dataset->size() == 0)
        {
             throw std::runtime_error("One or more datasets are empty. Check paths and data.");
        }

    }
    catch (const std::exception& e)
    {
        std::cerr << "Error loading datasets: " << e.what() << std::endl;
        return;
    }

    // --- Create DataLoaders ---
    DataLoader train_loader(train_dataset, batch_size, true, 12, 10 * batch_size);
    DataLoader val_loader(val_dataset, batch_size, false, 4, batch_size);
    DataLoader test_loader(test_dataset, batch_size, true, 4, batch_size);

    // --- Create Network, Optimizer, Loss ---
    NeuralNetwork network(batch_size);

    // --- Define MNIST Network Architecture ---
    try
    {
        network.add_layer(Conv2d(INPUT_CHANNELS, 32, INPUT_HEIGHT, INPUT_WIDTH, 3, 1, 1, batch_size, "relu"));
        network.add_layer(Conv2d(32, 64, INPUT_HEIGHT, INPUT_WIDTH, 3, 1, 1, batch_size, "relu"));
        network.add_layer(Flatten(batch_size, 64, INPUT_HEIGHT, INPUT_WIDTH));
        int flattened_size = 64 * INPUT_HEIGHT * INPUT_WIDTH;
        network.add_layer(Linear(flattened_size, 128, batch_size, "relu"));
        network.add_layer(Linear(128, NUM_CLASSES, batch_size, "none"));

        if (network.layers.empty())
        {
             throw std::runtime_error("Network definition is empty. Please add layers.");
        }
    }
    catch (const std::exception& e)
    {
         std::cerr << "Error building network: " << e.what() << std::endl;
         return;
    }


    std::unique_ptr<OptimizerBase> optimizer = create_optimizer("adam", learning_rate, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON);
    std::cout << "Using Adam Optimizer (LR=" << learning_rate << ")" << std::endl;

    // --- Loss Setup ---
    const char* loss_type = "cross_entropy";
    std::cout << "Using Loss: " << loss_type << std::endl;

    // --- Allocate Loss Buffers (Host only) ---
    size_t max_output_elements = batch_size * NUM_CLASSES;
    std::vector<float> h_element_loss(max_output_elements);
    float* loss_buffer_ptr = h_element_loss.data();


    std::cout << "Starting training..." << std::endl;
    std::cout << std::fixed << std::setprecision(4);

    train_start_overall = std::chrono::high_resolution_clock::now();

    // --- Variables for Metrics ---
    size_t total_train_batches_processed = 0;
    std::vector<double> epoch_times_s;
    // --- Add accumulators for detailed timings (total) ---
    double total_data_load_s = 0.0;
    double total_forward_s = 0.0;
    double total_loss_compute_s = 0.0;
    double total_loss_sum_s = 0.0;
    double total_backward_s = 0.0;
    double total_update_s = 0.0;
    double total_acc_calc_s = 0.0;

    // --- Training Loop ---
    std::shared_ptr<Batch> batch;
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        auto epoch_start_chrono = std::chrono::high_resolution_clock::now();

        float epoch_loss = 0.0f;
        float epoch_accuracy = 0.0f;
        size_t batches_processed_this_epoch = 0;
        train_loader.reset();

        // --- Reset epoch timing accumulators ---
        double epoch_data_load_s = 0.0;
        double epoch_forward_s = 0.0;
        double epoch_loss_compute_s = 0.0;
        double epoch_loss_sum_s = 0.0;
        double epoch_backward_s = 0.0;
        double epoch_update_s = 0.0;
        double epoch_acc_calc_s = 0.0;

        std::cout << "\n--- Epoch " << (epoch + 1) << "/" << epochs << " ---" << std::endl;

        while ((batch = train_loader.next_batch()) != nullptr)
        {
            // --- Time Data Loading ---
            auto dload_start = std::chrono::high_resolution_clock::now();
            if (!batch || batch->batch_size == 0) continue;
            auto dload_end = std::chrono::high_resolution_clock::now();
            epoch_data_load_s += std::chrono::duration<double>(dload_end - dload_start).count();
            // --- End Data Loading Time ---

            float* inputs = batch->inputs_flattened.data();
            float* targets = batch->targets_flattened.data();
            size_t current_batch_size = batch->batch_size;

            if (!inputs || !targets)
            {
                 std::cerr << "Warning: Skipping batch with null data pointers." << std::endl;
                 continue;
            }

            // --- Time Forward Pass ---
            auto fwd_chrono_start = std::chrono::high_resolution_clock::now();
            network.forward(inputs, current_batch_size, false);
            auto fwd_chrono_end = std::chrono::high_resolution_clock::now();
            epoch_forward_s += std::chrono::duration<double>(fwd_chrono_end - fwd_chrono_start).count();
            // --- End Forward Pass Time ---

            float* outputs = network.layers.back()->output;
            if (!outputs)
            {
                 std::cerr << "Warning: Skipping batch due to null network output." << std::endl;
                 continue;
            }

            size_t current_output_elements = current_batch_size * NUM_CLASSES;

            // --- Time Loss Computation ---
            auto loss_comp_chrono_start = std::chrono::high_resolution_clock::now();
            if (h_element_loss.size() < current_output_elements)
            {
                h_element_loss.resize(current_output_elements);
                loss_buffer_ptr = h_element_loss.data();
            }
            network.compute_loss(outputs, targets, loss_buffer_ptr, current_output_elements, loss_type);
            auto loss_comp_chrono_end = std::chrono::high_resolution_clock::now();
            epoch_loss_compute_s += std::chrono::duration<double>(loss_comp_chrono_end - loss_comp_chrono_start).count();
            // --- End Loss Computation Time ---

            // --- Time Loss Summation ---
            float batch_loss_sum = 0.0f;
            auto loss_sum_start = std::chrono::high_resolution_clock::now();
            batch_loss_sum = std::accumulate(h_element_loss.begin(), h_element_loss.begin() + current_output_elements, 0.0f);
            auto loss_sum_end = std::chrono::high_resolution_clock::now();
            epoch_loss_sum_s += std::chrono::duration<double>(loss_sum_end - loss_sum_start).count();
            // --- End Loss Summation Time ---

            float batch_avg_loss = (current_batch_size > 0) ? (batch_loss_sum / current_batch_size) : 0.0f;

            // --- Time Backward Pass ---
            auto bwd_chrono_start = std::chrono::high_resolution_clock::now();
            network.backward(targets, 0.0f, loss_type);
            auto bwd_chrono_end = std::chrono::high_resolution_clock::now();
            epoch_backward_s += std::chrono::duration<double>(bwd_chrono_end - bwd_chrono_start).count();
            // --- End Backward Pass Time ---

            // --- Time Optimizer Update ---
            auto update_chrono_start = std::chrono::high_resolution_clock::now();
            for (auto& layer : network.layers)
            {
                layer->update_params(*optimizer);
            }
            auto update_chrono_end = std::chrono::high_resolution_clock::now();
            epoch_update_s += std::chrono::duration<double>(update_chrono_end - update_chrono_start).count();
            // --- End Optimizer Update Time ---

            // --- Time Accuracy Calculation ---
            auto acc_calc_start = std::chrono::high_resolution_clock::now();
            float accuracy = calculate_accuracy(outputs, targets, current_batch_size, NUM_CLASSES);
            auto acc_calc_end = std::chrono::high_resolution_clock::now();
            epoch_acc_calc_s += std::chrono::duration<double>(acc_calc_end - acc_calc_start).count();
            // --- End Accuracy Calculation Time ---

            epoch_loss += batch_avg_loss;
            epoch_accuracy += accuracy;
            batches_processed_this_epoch++;
            total_train_batches_processed++;

            if (batches_processed_this_epoch % 10 == 0 || batches_processed_this_epoch == 1)
            {
                 std::cout << "  Batch " << std::setw(4) << batches_processed_this_epoch << "/" << train_loader.get_num_batches()
                           << " Loss: " << std::fixed << std::setprecision(4) << batch_avg_loss
                           << " Acc: " << std::fixed << std::setprecision(4) << accuracy << "\r" << std::flush;
            }
        } // End Batch Loop
        std::cout << std::endl;

        // --- Epoch Summary ---
        float avg_epoch_loss = 0.0f;
        float avg_epoch_acc = 0.0f;
        if (batches_processed_this_epoch > 0)
        {
            avg_epoch_loss = epoch_loss / batches_processed_this_epoch;
            avg_epoch_acc = epoch_accuracy / batches_processed_this_epoch;
            std::cout << "  Train Loss: " << std::fixed << std::setprecision(4) << avg_epoch_loss
                      << ", Train Accuracy: " << std::fixed << std::setprecision(4) << avg_epoch_acc << std::endl;
        }
        else
        {
            std::cout << "  No training batches processed this epoch." << std::endl;
        }

        // --- Epoch Timing ---
        auto epoch_end_chrono = std::chrono::high_resolution_clock::now();
        double epoch_duration_s_cpu = std::chrono::duration<double>(epoch_end_chrono - epoch_start_chrono).count();
        epoch_times_s.push_back(epoch_duration_s_cpu);
        std::cout << "  Epoch Time: " << std::fixed << std::setprecision(2) << epoch_times_s.back() << " s" << std::endl;

        // --- Report Detailed Epoch Timings ---
        if (batches_processed_this_epoch > 0)
        {
            std::cout << "  Avg Batch Timings (ms): "
                      << "Load: " << std::fixed << std::setprecision(3) << (epoch_data_load_s * 1000.0 / batches_processed_this_epoch)
                      << " | Fwd: " << std::fixed << std::setprecision(3) << (epoch_forward_s * 1000.0 / batches_processed_this_epoch)
                      << " | LossComp: " << std::fixed << std::setprecision(3) << (epoch_loss_compute_s * 1000.0 / batches_processed_this_epoch)
                      << " | LossSum: " << std::fixed << std::setprecision(3) << (epoch_loss_sum_s * 1000.0 / batches_processed_this_epoch)
                      << " | Bwd: " << std::fixed << std::setprecision(3) << (epoch_backward_s * 1000.0 / batches_processed_this_epoch)
                      << " | Update: " << std::fixed << std::setprecision(3) << (epoch_update_s * 1000.0 / batches_processed_this_epoch)
                      << " | AccCalc: " << std::fixed << std::setprecision(3) << (epoch_acc_calc_s * 1000.0 / batches_processed_this_epoch)
                      << std::endl;
        }

        // --- Accumulate total times ---
        total_data_load_s += epoch_data_load_s;
        total_forward_s += epoch_forward_s;
        total_loss_compute_s += epoch_loss_compute_s;
        total_loss_sum_s += epoch_loss_sum_s;
        total_backward_s += epoch_backward_s;
        total_update_s += epoch_update_s;
        total_acc_calc_s += epoch_acc_calc_s;

        // --- Validation Loop ---
        // ... (validation loop remains mostly the same) ...

    } // End Epoch Loop

    auto train_end_overall = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> train_duration_overall_ms = train_end_overall - train_start_overall;
    double cpu_train_time_ms = train_duration_overall_ms.count();
    std::cout << "\nTraining complete! Total CPU Training Time: " << std::fixed << std::setprecision(2) << cpu_train_time_ms << " ms\n";


    // --- Testing Phase ---
    std::cout << "\n--- Testing (CPU) ---" << std::endl;
    test_start_overall = std::chrono::high_resolution_clock::now();

    float test_loss = 0.0f;
    float test_accuracy = 0.0f;
    size_t test_batches = 0;
    float final_test_loss = 0.0f;
    float final_test_acc = 0.0f;
    test_loader.reset();

    while ((batch = test_loader.next_batch()) != nullptr)
    {
         if (!batch || batch->batch_size == 0) continue;
         float* inputs = batch->inputs_flattened.data();
         float* targets = batch->targets_flattened.data();
         size_t current_batch_size = batch->batch_size;
         if (!inputs || !targets) continue;

         network.forward(inputs, current_batch_size, false);
         float* outputs = network.layers.back()->output;
         if (!outputs) continue;

         size_t current_output_elements = current_batch_size * NUM_CLASSES;
         if (h_element_loss.size() < current_output_elements)
         {
             h_element_loss.resize(current_output_elements);
             loss_buffer_ptr = h_element_loss.data();
         }
         network.compute_loss(outputs, targets, loss_buffer_ptr, current_output_elements, loss_type);

         float batch_loss_sum = std::accumulate(h_element_loss.begin(), h_element_loss.begin() + current_output_elements, 0.0f);
         float batch_avg_loss = (current_batch_size > 0) ? (batch_loss_sum / current_batch_size) : 0.0f;
         test_loss += batch_avg_loss;

         float accuracy = calculate_accuracy(outputs, targets, current_batch_size, NUM_CLASSES);
         test_accuracy += accuracy;
         test_batches++;
    }

    auto test_end_overall = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> test_duration_overall_ms = test_end_overall - test_start_overall;
    double cpu_test_time_ms = test_duration_overall_ms.count();

    if (test_batches > 0)
    {
        final_test_loss = test_loss / test_batches;
        final_test_acc = test_accuracy / test_batches;
    }
    else
    {
        std::cout << "No test batches were processed." << std::endl;
    }
    std::cout << "CPU Testing Time: " << std::fixed << std::setprecision(2) << cpu_test_time_ms << " ms\n";


    // --- Print Sample Predictions ---
    std::cout << "\n--- Generating Samples for Prediction Output (CPU) ---" << std::endl;
    test_loader.reset();
    std::shared_ptr<Batch> sample_batch = test_loader.next_batch();

    if (sample_batch && sample_batch->batch_size > 0)
    {
        size_t sample_batch_size = sample_batch->batch_size;
        size_t sample_output_elements = sample_batch_size * NUM_CLASSES;

        std::vector<float> h_sample_targets(sample_output_elements);
        std::vector<float> h_sample_outputs(sample_output_elements);

        network.forward(sample_batch->inputs_flattened.data(), sample_batch_size, false);
        const float* host_sample_outputs_ptr = network.layers.back()->output;

        if (host_sample_outputs_ptr)
        {
            memcpy(h_sample_targets.data(), sample_batch->targets_flattened.data(), sample_output_elements * sizeof(float));
            memcpy(h_sample_outputs.data(), host_sample_outputs_ptr, sample_output_elements * sizeof(float));

            print_mnist_predictions(
                h_sample_targets.data(),
                h_sample_outputs.data(),
                std::min((int)sample_batch_size, 5),
                sample_batch_size,
                NUM_CLASSES
            );
        }
        else
        {
            std::cerr << "Could not generate sample outputs (forward pass failed?)." << std::endl;
        }
    }
    else
    {
        std::cout << "Could not get a sample batch for prediction output." << std::endl;
    }

    // --- Calculate Final Metrics ---
    double avg_latency_ms = (total_train_batches_processed > 0 && cpu_train_time_ms > 0) ? (cpu_train_time_ms / total_train_batches_processed) : 0.0;
    double throughput_samples_s = (cpu_train_time_ms > 0) ? (static_cast<double>(train_dataset->size() * num_epochs) / (cpu_train_time_ms / 1000.0)) : 0.0;
    double avg_epoch_time_s = !epoch_times_s.empty() ? (std::accumulate(epoch_times_s.begin(), epoch_times_s.end(), 0.0) / epoch_times_s.size()) : 0.0;

    // --- Get Memory Usage at End ---
    float peak_cpu_mem = get_peak_cpu_memory_mb();
  
    std::cout << "\n--- Final Summary (CPU) ---" << std::endl;
    if (test_batches > 0)
    {
        std::cout << "Test Loss:               " << std::fixed << std::setprecision(4) << final_test_loss << std::endl;
        std::cout << "Test Accuracy:           " << std::fixed << std::setprecision(4) << final_test_acc << std::endl;
    }
    else
    {
        std::cout << "Test Loss:               N/A" << std::endl;
        std::cout << "Test Accuracy:           N/A" << std::endl;
    }
    std::cout << "--- Performance Metrics (CPU) ---" << std::endl;
    std::cout << "Total CPU Train Time (ms): " << std::fixed << std::setprecision(2) << cpu_train_time_ms << std::endl;
    std::cout << "Total CPU Test Time (ms):  " << std::fixed << std::setprecision(2) << cpu_test_time_ms << std::endl;
    std::cout << "Avg. Latency/Batch (ms): " << std::fixed << std::setprecision(2) << avg_latency_ms << std::endl;
    std::cout << "Throughput (Samples/sec):" << std::fixed << std::setprecision(0) << throughput_samples_s << std::endl;
    std::cout << "Avg. Time per Epoch (s): " << std::fixed << std::setprecision(2) << avg_epoch_time_s << std::endl;
    if (peak_cpu_mem >= 0)
    {
        std::cout << "Peak CPU Memory (MB):    " << std::fixed << std::setprecision(1) << peak_cpu_mem << " (Linux VmHWM)" << std::endl;
    }
    else
    {
        std::cout << "Peak CPU Memory (MB):    N/A" << std::endl;
    }

    if (total_train_batches_processed > 0)
    {
        std::cout << "--- Avg Batch Breakdown (ms) ---" << std::endl;
        std::cout << "  Data Load: " << std::fixed << std::setprecision(3) << (total_data_load_s * 1000.0 / total_train_batches_processed) << std::endl;
        std::cout << "  Forward:   " << std::fixed << std::setprecision(3) << (total_forward_s * 1000.0 / total_train_batches_processed) << std::endl;
        std::cout << "  Loss Comp: " << std::fixed << std::setprecision(3) << (total_loss_compute_s * 1000.0 / total_train_batches_processed) << std::endl;
        std::cout << "  Loss Sum:  " << std::fixed << std::setprecision(3) << (total_loss_sum_s * 1000.0 / total_train_batches_processed) << std::endl;
        std::cout << "  Backward:  " << std::fixed << std::setprecision(3) << (total_backward_s * 1000.0 / total_train_batches_processed) << std::endl;
        std::cout << "  Update:    " << std::fixed << std::setprecision(3) << (total_update_s * 1000.0 / total_train_batches_processed) << std::endl;
        std::cout << "  Acc Calc:  " << std::fixed << std::setprecision(3) << (total_acc_calc_s * 1000.0 / total_train_batches_processed) << std::endl;
    }

    // --- Cleanup ---
    std::cout << "\nCleaning up MNIST CPU resources...\n";
    for (auto& layer : network.layers)
    {
        delete layer;
    }
    network.layers.clear();

    auto program_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(program_end - program_start);
    std::cout << "Total MNIST CPU function time: " << total_duration.count() << " ms\n";
    std::cout << "MNIST CPU training finished." << std::endl;
}

} // namespace cpu_mnist_train