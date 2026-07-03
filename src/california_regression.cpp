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

#include "california_regression.h"
#include "DataLoader.h"
#include "California_Dataset.h" 
#include "NeuralNetwork.h"
#include "optimizer.h"
#include "utils.h"
#include "layer_proxy.h"
#include "cuda_functions.h"

namespace california_regression {

const float ADAM_BETA1 = 0.9f;
const float ADAM_BETA2 = 0.999f;
const float ADAM_EPSILON = 1e-8f;

void train_california_regressor(
    int batch_size, int num_epochs,
    const std::string& dataset_path,
    float learning_rate ,
    const std::string& loss_type_str )
{   
    SetUseCuda(true);
    bool use_cuda = GetUseCuda();
    std::cout << "\n=== California Housing Regression Training ===\n";
    std::cout << "Using CUDA: " << (use_cuda ? "Yes" : "No") << std::endl;
    std::cout << "Epochs: " << num_epochs << ", Batch Size: " << batch_size
              << ", Learning Rate: " << learning_rate
              << ", Loss: " << loss_type_str << std::endl;
    std::cout << "Dataset Path: " << dataset_path << std::endl;

    cudaEvent_t cuda_start, cuda_stop;

    cudaEvent_t batch_fwd_start, batch_fwd_stop;
    cudaEvent_t batch_loss_start, batch_loss_stop;
    cudaEvent_t batch_bwd_start, batch_bwd_stop;
    cudaEvent_t batch_update_start, batch_update_stop;
    cudaEvent_t epoch_start_event, epoch_stop_event; 

    if (use_cuda) 
    {
        cudaEventCreate(&cuda_start);
        cudaEventCreate(&cuda_stop);
        cudaEventCreate(&batch_fwd_start);
        cudaEventCreate(&batch_fwd_stop);
        cudaEventCreate(&batch_loss_start);
        cudaEventCreate(&batch_loss_stop);
        cudaEventCreate(&batch_bwd_start);
        cudaEventCreate(&batch_bwd_stop);
        cudaEventCreate(&batch_update_start);
        cudaEventCreate(&batch_update_stop);
        cudaEventCreate(&epoch_start_event);
        cudaEventCreate(&epoch_stop_event);
    }
    auto program_start = std::chrono::high_resolution_clock::now();
    std::cout << "Loading full dataset..." << std::endl;
    auto dataset_load_start = std::chrono::high_resolution_clock::now();
    std::shared_ptr<CaliforniaDataset> full_dataset_unnormalized;
    try 
    {
        full_dataset_unnormalized = std::make_shared<CaliforniaDataset>(dataset_path, false);
        auto dataset_load_end = std::chrono::high_resolution_clock::now();
        auto dataset_load_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(dataset_load_end - dataset_load_start);
        std::cout << "Full dataset loaded successfully in " << dataset_load_duration_ms.count() << " ms." << std::endl;

        if (full_dataset_unnormalized->size() == 0) 
        {
             throw std::runtime_error("Loaded dataset is empty.");
        }
    } 
    catch (const std::exception& e) 
    {
        std::cerr << "Error loading full California dataset: " << e.what() << std::endl;
        return;
    }

    size_t total_samples = full_dataset_unnormalized->size();
    size_t test_size = static_cast<size_t>(total_samples * 0.2);
    size_t train_size = total_samples - test_size;

    if (train_size == 0 || test_size == 0) 
    {
        std::cerr << "Error: Train or test split resulted in zero samples. Adjust split ratio or check dataset size." << std::endl;
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
    train_features_vec.reserve(train_size);
    train_targets_vec.reserve(train_size);
    test_features_vec.reserve(test_size);
    test_targets_vec.reserve(test_size);

    for (size_t i = 0; i < train_size; ++i) 
    {
        train_features_vec.push_back(all_features[indices[i]]);
        train_targets_vec.push_back(all_targets[indices[i]]);
    }
    for (size_t i = train_size; i < total_samples; ++i) 
    {
        test_features_vec.push_back(all_features[indices[i]]);
        test_targets_vec.push_back(all_targets[indices[i]]);
    }

     std::shared_ptr<CaliforniaDataset> train_dataset;
    try 
    {
        train_dataset = std::make_shared<CaliforniaDataset>(train_features_vec, train_targets_vec, true);
    } 
    catch (const std::exception& e) 
    {
        std::cerr << "Error creating training dataset: " << e.what() << std::endl;
        return;
    }

    std::vector<float> train_mean_f = train_dataset->get_feature_mean();
    std::vector<float> train_std_f = train_dataset->get_feature_std();
    float train_mean_t = train_dataset->get_target_mean();
    float train_std_t = train_dataset->get_target_std();

    std::shared_ptr<CaliforniaDataset> test_dataset;
    try 
    {    
        test_dataset = std::make_shared<CaliforniaDataset>(
            test_features_vec, test_targets_vec, true, // Apply norm = true
            train_mean_f, train_std_f, train_mean_t, train_std_t // Use training stats
        );
    } 
    catch (const std::exception& e) 
    {
        std::cerr << "Error creating test dataset: " << e.what() << std::endl;
        return;
    }

    DataLoader train_loader(train_dataset, batch_size, true, 12, 2*batch_size);
    DataLoader test_loader(test_dataset, batch_size, false, 4, 2*batch_size);
    size_t total_train_batches_per_epoch = train_loader.get_num_batches();
    size_t total_test_batches = test_loader.get_num_batches();
    std::cout << "Train batches/epoch: " << total_train_batches_per_epoch
              << ", Test batches: " << total_test_batches << std::endl;

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

    std::unique_ptr<OptimizerBase> optimizer = create_optimizer("adam", learning_rate, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON);
    std::cout << "Using Adam Optimizer (LR=" << learning_rate << ")" << std::endl;
    const char* loss_type = loss_type_str.c_str();
    std::cout << "Using Loss: " << loss_type << std::endl;

    float* d_element_loss = nullptr;
    size_t max_loss_elements = batch_size;
    std::vector<float> h_element_loss(max_loss_elements);

    if (use_cuda) 
    {
        safeCudaMalloc(&d_element_loss, max_loss_elements * sizeof(float), "d_element_loss (California)");
    } 
    else 
    {
        d_element_loss = h_element_loss.data();
    }

    std::cout << "Starting training..." << std::endl;
    std::cout << std::fixed << std::setprecision(6);

    float total_gpu_train_time_ms = 0;
    if (use_cuda) cudaEventRecord(cuda_start);
    auto train_start_overall_chrono = std::chrono::high_resolution_clock::now();

    size_t total_train_batches_processed = 0;
    std::vector<double> epoch_times_s;

    double total_data_load_s = 0.0;
    double total_forward_s = 0.0;
    double total_loss_gpu_s = 0.0; 
    double total_loss_host_s = 0.0;
    double total_backward_s = 0.0;
    double total_update_s = 0.0;

    for (int epoch = 0; epoch < num_epochs; ++epoch) 
    {
        auto epoch_start_chrono = std::chrono::high_resolution_clock::now();
        if (use_cuda) cudaEventRecord(epoch_start_event);

        float epoch_loss_sum = 0.0f;
        size_t batches_processed_this_epoch = 0;
        train_loader.reset();

        double epoch_data_load_s = 0.0;
        double epoch_forward_s = 0.0;
        double epoch_loss_gpu_s = 0.0;
        double epoch_loss_host_s = 0.0;
        double epoch_backward_s = 0.0;
        double epoch_update_s = 0.0;
        
        float batch_fwd_time_ms = 0.0f;
        float batch_loss_gpu_time_ms = 0.0f;
        float batch_bwd_time_ms = 0.0f;
        float batch_update_time_ms = 0.0f;

        std::cout << "\n--- Epoch " << (epoch + 1) << "/" << num_epochs << " ---" << std::endl;

        std::shared_ptr<Batch> batch;
        while ((batch = train_loader.next_batch()) != nullptr) 
        {

            auto dload_start = std::chrono::high_resolution_clock::now();
            if (!batch || batch->batch_size == 0) continue;
            auto dload_end = std::chrono::high_resolution_clock::now();
            epoch_data_load_s += std::chrono::duration<double>(dload_end - dload_start).count();

            float* inputs = use_cuda ? batch->d_inputs : batch->inputs_flattened.data();
            float* targets = use_cuda ? batch->d_targets : batch->targets_flattened.data();
            size_t current_batch_size = batch->batch_size;
            size_t current_output_elements = current_batch_size * output_size;

            if (!inputs || !targets) 
            {
                 std::cerr << "Warning: Skipping batch with null data pointers." << std::endl;
                 continue;
            }

            if (use_cuda) cudaEventRecord(batch_fwd_start);
            auto fwd_chrono_start = std::chrono::high_resolution_clock::now();
            network.forward(inputs, current_batch_size, use_cuda);
            auto fwd_chrono_end = std::chrono::high_resolution_clock::now();
            if (use_cuda) cudaEventRecord(batch_fwd_stop);
            else epoch_forward_s += std::chrono::duration<double>(fwd_chrono_end - fwd_chrono_start).count();
     
            float* outputs = network.layers.back()->output;
            if (!outputs) 
            {
                 std::cerr << "Warning: Skipping batch due to null network output." << std::endl;
                 continue;
            }

            float* current_loss_buffer_ptr = use_cuda ? d_element_loss : h_element_loss.data();

            if (use_cuda) cudaEventRecord(batch_loss_start);
            auto loss_comp_chrono_start = std::chrono::high_resolution_clock::now();

            network.compute_loss(outputs, targets, current_loss_buffer_ptr, current_batch_size, loss_type);
            auto loss_comp_chrono_end = std::chrono::high_resolution_clock::now();
            if (use_cuda) cudaEventRecord(batch_loss_stop);
            else epoch_loss_gpu_s += std::chrono::duration<double>(loss_comp_chrono_end - loss_comp_chrono_start).count();
  
            float batch_loss_sum = 0.0f;
            auto loss_sum_start = std::chrono::high_resolution_clock::now();
            if (use_cuda) 
            {
                if (h_element_loss.size() < current_batch_size) 
                {
                    h_element_loss.resize(current_batch_size);
                }
                CUDA_CHECK_ERROR(cudaMemcpy(h_element_loss.data(), d_element_loss, current_batch_size * sizeof(float), cudaMemcpyDeviceToHost));
            }
            batch_loss_sum = std::accumulate(h_element_loss.begin(), h_element_loss.begin() + current_batch_size, 0.0f);
            auto loss_sum_end = std::chrono::high_resolution_clock::now();
            epoch_loss_host_s += std::chrono::duration<double>(loss_sum_end - loss_sum_start).count();
 
            float batch_avg_loss = (current_batch_size > 0) ? (batch_loss_sum / current_batch_size) : 0.0f;

            if (use_cuda) cudaEventRecord(batch_bwd_start);
            auto bwd_chrono_start = std::chrono::high_resolution_clock::now();
            network.backward(targets, 0.0f, loss_type); 
            auto bwd_chrono_end = std::chrono::high_resolution_clock::now();
            if (use_cuda) cudaEventRecord(batch_bwd_stop);
            else epoch_backward_s += std::chrono::duration<double>(bwd_chrono_end - bwd_chrono_start).count();
   
            if (use_cuda) cudaEventRecord(batch_update_start);
            auto update_chrono_start = std::chrono::high_resolution_clock::now();
            for (auto& layer : network.layers) 
            {
                layer->update_params(*optimizer);
            }
            auto update_chrono_end = std::chrono::high_resolution_clock::now();
            if (use_cuda) cudaEventRecord(batch_update_stop);
            else epoch_update_s += std::chrono::duration<double>(update_chrono_end - update_chrono_start).count();

            if (use_cuda) 
            {
                cudaEventSynchronize(batch_fwd_stop);
                cudaEventElapsedTime(&batch_fwd_time_ms, batch_fwd_start, batch_fwd_stop);
                epoch_forward_s += batch_fwd_time_ms / 1000.0;

                cudaEventSynchronize(batch_loss_stop);
                cudaEventElapsedTime(&batch_loss_gpu_time_ms, batch_loss_start, batch_loss_stop);
                epoch_loss_gpu_s += batch_loss_gpu_time_ms / 1000.0;

                cudaEventSynchronize(batch_bwd_stop);
                cudaEventElapsedTime(&batch_bwd_time_ms, batch_bwd_start, batch_bwd_stop);
                epoch_backward_s += batch_bwd_time_ms / 1000.0;

                cudaEventSynchronize(batch_update_stop);
                cudaEventElapsedTime(&batch_update_time_ms, batch_update_start, batch_update_stop);
                epoch_update_s += batch_update_time_ms / 1000.0;
            }

            epoch_loss_sum += batch_avg_loss;
            batches_processed_this_epoch++;
            total_train_batches_processed++;

            if (batches_processed_this_epoch % 50 == 0 || batches_processed_this_epoch == 1) 
            {
                 std::cout << "  Batch " << std::setw(4) << batches_processed_this_epoch << "/" << total_train_batches_per_epoch 
                           << " AvgLoss: " << batch_avg_loss << "\r" << std::flush;
            }
        }
        std::cout << std::endl; 

        float avg_epoch_loss = 0.0f;
        if (batches_processed_this_epoch > 0) 
        {
            avg_epoch_loss = epoch_loss_sum / batches_processed_this_epoch;
            std::cout << "  Train Loss (" << loss_type << "): " << avg_epoch_loss << std::endl;
        } 
        else 
        {
            std::cout << "  No training batches processed this epoch." << std::endl;
        }

        auto epoch_end_chrono = std::chrono::high_resolution_clock::now();
        double epoch_duration_s_cpu = std::chrono::duration<double>(epoch_end_chrono - epoch_start_chrono).count();
        float gpu_epoch_time_ms = 0;
        if (use_cuda) 
        {
            cudaEventRecord(epoch_stop_event);
            cudaEventSynchronize(epoch_stop_event);
            cudaEventElapsedTime(&gpu_epoch_time_ms, epoch_start_event, epoch_stop_event);
            epoch_times_s.push_back(gpu_epoch_time_ms / 1000.0);
        } 
        else 
        {
            epoch_times_s.push_back(epoch_duration_s_cpu);
        }
        std::cout << "  Epoch Time: "  << epoch_times_s.back() << " s" << std::endl;

        if (batches_processed_this_epoch > 0) 
        {
            std::cout << "  Avg Batch Timings (ms): "
                      << "Load: "  << (epoch_data_load_s * 1000.0 / batches_processed_this_epoch)
                      << " | Fwd: "  << (epoch_forward_s * 1000.0 / batches_processed_this_epoch)
                      << " | LossGPU: " << (epoch_loss_gpu_s * 1000.0 / batches_processed_this_epoch)
                      << " | LossHost: "  << (epoch_loss_host_s * 1000.0 / batches_processed_this_epoch)
                      << " | Bwd: "  << (epoch_backward_s * 1000.0 / batches_processed_this_epoch)
                      << " | Update: " << (epoch_update_s * 1000.0 / batches_processed_this_epoch)
                      << std::endl;
        }

        total_data_load_s += epoch_data_load_s;
        total_forward_s += epoch_forward_s;
        total_loss_gpu_s += epoch_loss_gpu_s;
        total_loss_host_s += epoch_loss_host_s;
        total_backward_s += epoch_backward_s;
        total_update_s += epoch_update_s;

    }

    if (use_cuda) 
    {
        cudaEventRecord(cuda_stop);
        cudaEventSynchronize(cuda_stop);
        cudaEventElapsedTime(&total_gpu_train_time_ms, cuda_start, cuda_stop);
        std::cout << "\nTraining complete! GPU Training Time: " << total_gpu_train_time_ms << " ms\n";
    } 
    else 
    {
         auto cpu_train_end = std::chrono::high_resolution_clock::now();
         total_gpu_train_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_train_end - train_start_overall_chrono).count();
         std::cout << "\nTraining complete! CPU Training Time: " << total_gpu_train_time_ms << " ms\n";
    }

    std::cout << "\n--- Testing ---" << std::endl;
    float gpu_test_time_ms = 0;
    if (use_cuda) cudaEventRecord(cuda_start);
    auto test_start_chrono = std::chrono::high_resolution_clock::now();

    float test_loss_sum_total = 0.0f;
    size_t test_samples = 0;
    test_loader.reset();

    std::vector<float> all_test_preds_denorm;
    std::vector<float> all_test_targets_denorm;

    std::shared_ptr<Batch> test_batch;
    while ((test_batch = test_loader.next_batch()) != nullptr) 
    {
         if (!test_batch || test_batch->batch_size == 0) continue;

         float* inputs = use_cuda ? test_batch->d_inputs : test_batch->inputs_flattened.data();
         float* targets = use_cuda ? test_batch->d_targets : test_batch->targets_flattened.data();
         size_t current_batch_size = test_batch->batch_size;
         size_t current_output_elements = current_batch_size * output_size;

         if (!inputs || !targets) continue;

         network.forward(inputs, current_batch_size, use_cuda);
         float* outputs = network.layers.back()->output;
         if (!outputs) continue;

         float* current_loss_buffer_ptr = use_cuda ? d_element_loss : h_element_loss.data();
         network.compute_loss(outputs, targets, current_loss_buffer_ptr, current_batch_size, loss_type); 

         float batch_loss_sum = 0.0f;
         std::vector<float> h_outputs(current_output_elements);
         std::vector<float> h_targets(current_output_elements);

         if (use_cuda) 
         {
             if (h_element_loss.size() < current_batch_size) 
             {
                 h_element_loss.resize(current_batch_size);
             }
             CUDA_CHECK_ERROR(cudaMemcpy(h_element_loss.data(), d_element_loss, current_batch_size * sizeof(float), cudaMemcpyDeviceToHost));
             CUDA_CHECK_ERROR(cudaMemcpy(h_outputs.data(), outputs, current_output_elements * sizeof(float), cudaMemcpyDeviceToHost));
             CUDA_CHECK_ERROR(cudaMemcpy(h_targets.data(), targets, current_output_elements * sizeof(float), cudaMemcpyDeviceToHost));
         } 
         else 
         {
             memcpy(h_outputs.data(), outputs, current_output_elements * sizeof(float));
             memcpy(h_targets.data(), targets, current_output_elements * sizeof(float));
         }
         batch_loss_sum = std::accumulate(h_element_loss.begin(), h_element_loss.begin() + current_batch_size, 0.0f);
         test_loss_sum_total += batch_loss_sum;
         test_samples += current_batch_size;

         std::vector<float> batch_preds_denorm = test_dataset->denormalize_targets(h_outputs);
         std::vector<float> batch_targets_denorm = test_dataset->denormalize_targets(h_targets);
         all_test_preds_denorm.insert(all_test_preds_denorm.end(), batch_preds_denorm.begin(), batch_preds_denorm.end());
         all_test_targets_denorm.insert(all_test_targets_denorm.end(), batch_targets_denorm.begin(), batch_targets_denorm.end());

    } 

    if (use_cuda) 
    {
        cudaEventRecord(cuda_stop);
        cudaEventSynchronize(cuda_stop);
        cudaEventElapsedTime(&gpu_test_time_ms, cuda_start, cuda_stop);
    } 
    else 
    {
        auto test_end_chrono = std::chrono::high_resolution_clock::now();
        gpu_test_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(test_end_chrono - test_start_chrono).count(); 
    }

    float final_test_loss = 0.0f;
    float final_mae = 0.0f;
    if (test_samples > 0) {
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

    if (use_cuda) 
    {
         std::cout << "GPU Testing Time: " << gpu_test_time_ms << " ms\n";
    }
    else 
    {
         std::cout << "CPU Testing Time: " << gpu_test_time_ms << " ms\n";
    }

    float avg_latency_ms = (total_train_batches_processed > 0) ? (total_gpu_train_time_ms / total_train_batches_processed) : 0.0f;
    float throughput_samples_s = (total_gpu_train_time_ms > 0) ? (static_cast<float>(test_dataset->size() * num_epochs) / (total_gpu_train_time_ms / 1000.0f)) : 0.0f;
  
    double avg_epoch_time_s = !epoch_times_s.empty() ? (std::accumulate(epoch_times_s.begin(), epoch_times_s.end(), 0.0) / epoch_times_s.size()) : 0.0;


    float peak_cpu_mem = get_peak_cpu_memory_mb();
    float current_gpu_mem = get_current_gpu_memory_usage_mb(GetUseCuda());

    std::cout << "\n--- Final Summary ---" << std::endl;

    if (test_samples > 0) 
    {
        std::cout << "Test Loss (" << loss_type << "):      " << final_test_loss << std::endl;
        std::cout << "Test MAE (denormalized): "<< final_mae << std::endl;
    } 
    else 
    {
        std::cout << "Test Loss:               N/A" << std::endl;
        std::cout << "Test MAE:                N/A" << std::endl;
    }
    std::cout << "--- Performance Metrics ---" << std::endl;
    std::cout << "Total Train Time (ms):   "  << total_gpu_train_time_ms << std::endl;
    std::cout << "Total Test Time (ms):    "  << gpu_test_time_ms << std::endl;
    std::cout << "Avg. Latency/Batch (ms): "  << avg_latency_ms << std::endl;
    std::cout << "Throughput (Samples/sec):"  << throughput_samples_s << std::endl;
    std::cout << "Avg. Time per Epoch (s): "  << avg_epoch_time_s << std::endl;
    if (peak_cpu_mem >= 0) 
    {
        std::cout << "Peak CPU Memory (MB):    "  << peak_cpu_mem << " (Linux VmHWM)" << std::endl;
    } 
    else 
    {
        std::cout << "Peak CPU Memory (MB):    N/A" << std::endl;
    }
    if (current_gpu_mem >= 0) 
    {
         std::cout << "Current GPU Memory (MB): " << current_gpu_mem << " (Used at end)" << std::endl;
    } 
    else 
    {
         std::cout << "Current GPU Memory (MB): N/A" << std::endl;
    }

    if (total_train_batches_processed > 0) 
    {
        std::cout << "--- Avg Batch Breakdown (ms) ---" << std::endl;
        std::cout << "  Data Load: " << std::fixed << std::setprecision(3) << (total_data_load_s * 1000.0 / total_train_batches_processed) << std::endl;
        std::cout << "  Forward:   " << std::fixed << std::setprecision(3) << (total_forward_s * 1000.0 / total_train_batches_processed) << std::endl;
        std::cout << "  Loss GPU:  " << std::fixed << std::setprecision(3) << (total_loss_gpu_s * 1000.0 / total_train_batches_processed) << std::endl;
        std::cout << "  Loss Host: " << std::fixed << std::setprecision(3) << (total_loss_host_s * 1000.0 / total_train_batches_processed) << std::endl;
        std::cout << "  Backward:  " << std::fixed << std::setprecision(3) << (total_backward_s * 1000.0 / total_train_batches_processed) << std::endl;
        std::cout << "  Update:    " << std::fixed << std::setprecision(3) << (total_update_s * 1000.0 / total_train_batches_processed) << std::endl;
    }
    std::cout << "============================================\n";

    std::cout << "\nCleaning up California Regression resources...\n";
    if (use_cuda) {
        safeCudaFree(&d_element_loss, "d_element_loss (California)");
        cudaEventDestroy(batch_fwd_start);
        cudaEventDestroy(batch_fwd_stop);
        cudaEventDestroy(batch_loss_start);
        cudaEventDestroy(batch_loss_stop);
        cudaEventDestroy(batch_bwd_start);
        cudaEventDestroy(batch_bwd_stop);
        cudaEventDestroy(batch_update_start);
        cudaEventDestroy(batch_update_stop);
        cudaEventDestroy(epoch_start_event);
        cudaEventDestroy(epoch_stop_event);
        cudaEventDestroy(cuda_start);
        cudaEventDestroy(cuda_stop);
    }

    auto program_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(program_end - program_start);
    std::cout << "Total California Regression function time: " << total_duration.count() << " ms\n"; 

}

}