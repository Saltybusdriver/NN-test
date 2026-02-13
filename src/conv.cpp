#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <iostream>
#include <vector>
#include <chrono>
#include <cublas_v2.h>
#include <curand.h>
#include <opencv4/opencv2/opencv.hpp>
#include <numeric>
#include <iomanip> 

#include "NeuralNetwork.h"
#include "debug.h"
#include "DataLoader.h"
#include "utils.h"
#include "ImageDataset.h"
#include "autodiff.h"
#include "cuda_functions.h"
#include "Butterfly_dataset.h"
#include "optimizer.h"
#include "layer_proxy.h" 


DEFINE_CUSTOM_LOSS({
    auto diff = o - t;
    auto squared_error = square(diff);
    auto weight = Constant(1.0f) + squared_error * Constant(0.5f);
    return weight * squared_error;
})

namespace conv{


// Calculate loss for a single sample (used in display_ae_samples)
float calculate_sample_loss(const std::vector<float>& output, const std::vector<float>& target, const char* loss_type) {
    if (output.size() != target.size() || output.empty()) 
    {
        return 0.0f;
    }

    float total_loss = 0.0f;
    if (strcmp(loss_type, "mse") == 0) 
    {
        for (size_t i = 0; i < output.size(); i++) 
        {
            float diff = output[i] - target[i];
            total_loss += diff * diff;
        }
        return total_loss / output.size();
    } 
    else if (strcmp(loss_type, "custom") == 0)
    { 
        for (size_t i = 0; i < output.size(); i++) 
        {
            auto loss_expr = autodiff::loss::CustomLoss::expression();
            total_loss += loss_expr.eval(output[i], target[i]);
        }
        return total_loss / output.size(); 
    }
    else 
    {
        std::cerr << "Warning: Unsupported loss type '" << loss_type << "' in calculate_sample_loss. Returning 0." << std::endl;
        return 0.0f;
    }
}


void display_ae_samples(
    const std::vector<float>& inputs,
    const std::vector<float>& outputs, 
    int batch_size,
    int channels,
    int height,
    int width,
    bool use_grayscale,
    const char* loss_type,
    int num_samples = 5)
{
    std::cout << "\n=== Validation Samples (Autoencoder - Showing up to " << num_samples << ") ===\n";
    int samples_to_show = std::min(num_samples, batch_size);
    size_t single_image_elements = (size_t)channels * height * width;

    int output_channels = channels;
    size_t single_output_elements = (size_t)output_channels * height * width;

    std::vector<std::vector<std::string>> input_term_reps;
    std::vector<std::vector<std::string>> output_term_reps;
    std::vector<float> sample_losses;

    for (int i = 0; i < samples_to_show; ++i) 
    {
        std::vector<float> input_sample(
            inputs.begin() + i * single_image_elements,
            inputs.begin() + (i + 1) * single_image_elements
        );

         std::vector<float> output_sample(
            outputs.begin() + i * single_output_elements,
            outputs.begin() + (i + 1) * single_output_elements
        );

        input_term_reps.push_back(display_image_opencv(input_sample, height, width, use_grayscale, "AE Input Sample " + std::to_string(i), true));
        output_term_reps.push_back(display_image_opencv(output_sample, height, width, use_grayscale, "AE Output Sample " + std::to_string(i), true)); 

        sample_losses.push_back(calculate_sample_loss(output_sample, input_sample, loss_type));
    }

    for (int i = 0; i < samples_to_show; ++i) 
    {
        std::cout << "\n--- Sample " << i << " --- Loss (" << loss_type << " per element): " << std::fixed << std::setprecision(6) << sample_losses[i] << " ---\n";

        const auto& input_lines = input_term_reps[i];
        const auto& output_lines = output_term_reps[i];

        size_t max_height = std::max(input_lines.size(), output_lines.size());
        size_t input_width = input_lines.empty() ? 0 : input_lines[0].length();
        size_t output_width = output_lines.empty() ? 0 : output_lines[0].length();

        std::string header1 = "Input/Target:";
        std::string header2 = "Reconstruction:";
        std::cout << std::left << std::setw(input_width) << header1 << " | " << std::left << header2 << std::endl;

        std::cout << std::string(input_width, '-') << "-+-" << std::string(output_width, '-') << std::endl;


        for (size_t line_idx = 0; line_idx < max_height; ++line_idx) 
        {
            std::string in_line = (line_idx < input_lines.size()) ? input_lines[line_idx] : std::string(input_width, ' ');
            std::string out_line = (line_idx < output_lines.size()) ? output_lines[line_idx] : std::string(output_width, ' '); // Pad output line too

            std::cout << std::left << std::setw(input_width) << in_line << " | " << out_line << std::endl;
        }
    }
    std::cout << "---------------------------------------------------\n";

    std::cout << "Press any key in an image window to continue...\n";
    cv::waitKey(0);
    cv::destroyAllWindows();
}




void train_conv_model(int batch_size = 64, int num_epochs = 5,
                     const std::string& dataset_path = "",
                     int image_height = 32, int image_width = 32,
                     bool use_grayscale = true) {
    bool use_cuda = true;
    SetUseCuda(use_cuda);
    set_debug_level(LEVEL_ERROR);

    std::cout << "\n=== Convolutional Autoencoder Training ===\n";
    std::cout << "Using CUDA: Yes\n";
    std::cout << "Image Size: " << image_height << "x" << image_width << ", Grayscale: " << (use_grayscale ? "Yes" : "No") << "\n";
    std::cout << "Batch Size: " << batch_size << ", Epochs: " << num_epochs << "\n";

    cudaEvent_t cuda_start, cuda_stop, cuda_epoch_start, cuda_epoch_stop, cuda_val_start, cuda_val_stop;

    cudaEvent_t batch_fwd_start, batch_fwd_stop;
    cudaEvent_t batch_loss_gpu_start, batch_loss_gpu_stop;
    cudaEvent_t batch_bwd_start, batch_bwd_stop;
    cudaEvent_t batch_update_start, batch_update_stop;

    cudaEventCreate(&cuda_start);
    cudaEventCreate(&cuda_stop);
    cudaEventCreate(&cuda_epoch_start); 
    cudaEventCreate(&cuda_epoch_stop);  
    cudaEventCreate(&cuda_val_start);  
    cudaEventCreate(&cuda_val_stop);   


    cudaEventCreate(&batch_fwd_start);
    cudaEventCreate(&batch_fwd_stop);
    cudaEventCreate(&batch_loss_gpu_start);
    cudaEventCreate(&batch_loss_gpu_stop);
    cudaEventCreate(&batch_bwd_start);
    cudaEventCreate(&batch_bwd_stop);
    cudaEventCreate(&batch_update_start);
    cudaEventCreate(&batch_update_stop);


    auto program_start = std::chrono::high_resolution_clock::now();

    std::cout << "Building CNN Autoencoder architecture...\n";
    NeuralNetwork nn(batch_size);
    int channels = use_grayscale ? 1 : 3;

    // Encoder
    nn.add_layer(Conv2d(channels, 16, image_height, image_width, 3, 1, 1, batch_size, "relu"));
    nn.add_layer(Conv2d(16, 8, image_height, image_width, 3, 1, 1, batch_size, "relu"));

    // Decoder
    nn.add_layer(Conv2d(8, 16, image_height, image_width, 3, 1, 1, batch_size, "relu"));
    nn.add_layer(Conv2d(16, channels, image_height, image_width, 3, 1, 1, batch_size, "sigmoid"));

    std::cout << "Neural network initialized with " << nn.layers.size() << " layers\n";

    std::string root_dir = dataset_path;
    std::string default_train_csv = root_dir + "Training_set.csv";
    std::string default_test_csv = root_dir + "Testing_set.csv";
    std::string train_csv = default_train_csv;
    std::string test_csv = default_test_csv;
    bool use_rgb = !use_grayscale;
    
    auto dataset_load_start = std::chrono::high_resolution_clock::now(); 

    auto train_dataset = std::make_shared<ButterflyDataset>(train_csv, root_dir, image_width, image_height, use_rgb);
    std::cout << "Training dataset loaded with " << train_dataset->size() << " images\n";

    auto test_dataset = std::make_shared<ButterflyDataset>(test_csv, root_dir, image_width, image_height, use_rgb);
    std::cout << "Test dataset loaded with " << test_dataset->size() << " images\n";
    
    auto dataset_load_end = std::chrono::high_resolution_clock::now();
    auto dataset_load_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(dataset_load_end - dataset_load_start);
    std::cout << "Datasets loaded successfully in " << dataset_load_duration_ms.count() << " ms." << std::endl;
                
    DataLoader loader(train_dataset, batch_size, false, 12, 6000); 
    DataLoader test_loader(test_dataset, batch_size, true, 12, 2000);

    float learning_rate = 0.001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    std::unique_ptr<OptimizerBase> optimizer = create_optimizer("adam", learning_rate, beta1, beta2, epsilon);
    std::cout << "Using Adam Optimizer (LR=" << learning_rate << ")" << std::endl;

    const char* loss_type = "mse";
    std::cout << "Using loss function: " << loss_type << std::endl;

    float* d_element_loss = nullptr;

    size_t max_output_elements = (size_t)batch_size * channels * image_height * image_width;
    safeCudaMalloc(&d_element_loss, max_output_elements * sizeof(float), "d_element_loss (ConvAE)");
    std::vector<float> h_element_loss(max_output_elements); 

    size_t total_train_batches_processed = 0;
    std::vector<float> epoch_times_s;

    // Start recording overall GPU training time
    cudaEventRecord(cuda_start);

    std::cout << "Starting training..." << std::endl;
    std::cout << std::fixed << std::setprecision(6); 

    cudaEventRecord(cuda_start);
    auto train_start_overall_chrono = std::chrono::high_resolution_clock::now();

    
    double total_data_load_s = 0.0;
    double total_forward_s = 0.0;
    double total_loss_gpu_s = 0.0;
    double total_loss_host_s = 0.0;
    double total_backward_s = 0.0;
    double total_update_s = 0.0;
    double total_acc_calc_s = 0.0; // Placeholder for AE, will remain 0

    std::shared_ptr<Batch> batch;
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        auto epoch_start_chrono = std::chrono::high_resolution_clock::now();
        cudaEventRecord(cuda_epoch_start);

        float epoch_loss = 0.0f;

        size_t batches_processed_this_epoch = 0;
        loader.reset();

        double epoch_data_load_s = 0.0;
        double epoch_forward_s = 0.0;
        double epoch_loss_gpu_s = 0.0;
        double epoch_loss_host_s = 0.0;
        double epoch_backward_s = 0.0;
        double epoch_update_s = 0.0;
        double epoch_acc_calc_s = 0.0; 
  
        float batch_fwd_time_ms = 0.0f;
        float batch_loss_gpu_time_ms = 0.0f;
        float batch_bwd_time_ms = 0.0f;
        float batch_update_time_ms = 0.0f;

        std::cout << "\n--- Epoch " << (epoch + 1) << "/" << num_epochs << " ---" << std::endl;

        while ((batch = loader.next_batch()) != nullptr) 
        {
            // --- Time Data Loading ---
            auto dload_start = std::chrono::high_resolution_clock::now();
            if (!batch || batch->batch_size == 0) continue;
            int current_batch_size = batch->batch_size;
            if(batch->d_inputs == nullptr) 
            {
                std::cerr << "Warning: Skipping batch with null input data pointer." << std::endl;
                continue;
            }
            auto dload_end = std::chrono::high_resolution_clock::now();
            epoch_data_load_s += std::chrono::duration<double>(dload_end - dload_start).count();
            // --- End Data Loading Time ---

            float* inputs = batch->d_inputs;
            float* targets = batch->d_inputs; // Target is input for AE

            // --- Time Forward Pass ---
            cudaEventRecord(batch_fwd_start);
            nn.forward(inputs, current_batch_size, use_cuda);
            cudaEventRecord(batch_fwd_stop);
            // --- End Forward Pass Time ---

            float* outputs = nn.layers.back()->output;
            if (!outputs) 
            {
                 std::cerr << "Warning: Skipping batch due to null network output." << std::endl;
                 continue;
            }

            size_t current_output_elements = (size_t)current_batch_size * channels * image_height * image_width;

            // --- Time Loss Computation (GPU) ---
            cudaEventRecord(batch_loss_gpu_start);
            nn.compute_loss(outputs, targets, d_element_loss, current_output_elements, loss_type);
            cudaEventRecord(batch_loss_gpu_stop);
            // --- End Loss Computation Time ---

            // --- Time Loss Summation (Host) ---
            float batch_loss_sum = 0.0f;
            auto loss_sum_start = std::chrono::high_resolution_clock::now();
            if (current_output_elements > 0) 
            {
                if (h_element_loss.size() < current_output_elements) 
                {
                    h_element_loss.resize(current_output_elements);
                }
                CUDA_CHECK_ERROR(cudaMemcpy(h_element_loss.data(), d_element_loss, current_output_elements * sizeof(float), cudaMemcpyDeviceToHost));
                batch_loss_sum = std::accumulate(h_element_loss.begin(), h_element_loss.begin() + current_output_elements, 0.0f);
            }
            auto loss_sum_end = std::chrono::high_resolution_clock::now();
            epoch_loss_host_s += std::chrono::duration<double>(loss_sum_end - loss_sum_start).count();
            // --- End Loss Summation Time ---

            float batch_avg_loss = (current_output_elements > 0) ? (batch_loss_sum / current_output_elements) : 0.0f;

            // --- Time Backward Pass ---
            cudaEventRecord(batch_bwd_start);
            nn.backward(targets, 0.0f, loss_type); // Target is input for AE
            cudaEventRecord(batch_bwd_stop);
            // --- End Backward Pass Time ---

            // --- Time Optimizer Update ---
            cudaEventRecord(batch_update_start);
            for (auto& layer : nn.layers) 
            {
                layer->update_params(*optimizer);
            }
            cudaEventRecord(batch_update_stop);
            // --- End Optimizer Update Time ---

            // --- Time Accuracy Calculation (Placeholder) ---
            auto acc_calc_start = std::chrono::high_resolution_clock::now();

            auto acc_calc_end = std::chrono::high_resolution_clock::now();
            epoch_acc_calc_s += std::chrono::duration<double>(acc_calc_end - acc_calc_start).count();
            // --- End Accuracy Calculation Time ---

            // --- Synchronize and Accumulate GPU Timings ---
            cudaEventSynchronize(batch_fwd_stop);
            cudaEventElapsedTime(&batch_fwd_time_ms, batch_fwd_start, batch_fwd_stop);
            epoch_forward_s += batch_fwd_time_ms / 1000.0;

            cudaEventSynchronize(batch_loss_gpu_stop);
            cudaEventElapsedTime(&batch_loss_gpu_time_ms, batch_loss_gpu_start, batch_loss_gpu_stop);
            epoch_loss_gpu_s += batch_loss_gpu_time_ms / 1000.0;

            cudaEventSynchronize(batch_bwd_stop);
            cudaEventElapsedTime(&batch_bwd_time_ms, batch_bwd_start, batch_bwd_stop);
            epoch_backward_s += batch_bwd_time_ms / 1000.0;

            cudaEventSynchronize(batch_update_stop);
            cudaEventElapsedTime(&batch_update_time_ms, batch_update_start, batch_update_stop);
            epoch_update_s += batch_update_time_ms / 1000.0;
            // --- End Accumulate GPU Timings ---

            epoch_loss += batch_avg_loss; 
 
            batches_processed_this_epoch++;
            total_train_batches_processed++;

            if (batches_processed_this_epoch % 10 == 0 || batches_processed_this_epoch == 1) {
                 std::cout << "  Batch " << std::setw(4) << batches_processed_this_epoch << "/" << loader.get_num_batches()
                           << " Loss: " << std::fixed << std::setprecision(6) << batch_avg_loss 
                           << "\r" << std::flush;
            }
        } 
        std::cout << std::endl;

        float avg_epoch_loss = 0.0f;
        if (batches_processed_this_epoch > 0) {
            avg_epoch_loss = epoch_loss / batches_processed_this_epoch;
            std::cout << "  Train Loss: " << std::fixed << std::setprecision(6) << avg_epoch_loss 
                      << std::endl;
        } 
        else 
        {
            std::cout << "  No training batches processed this epoch." << std::endl;
        }

        auto epoch_end_chrono = std::chrono::high_resolution_clock::now();
        
        float gpu_epoch_time_ms = 0;
        cudaEventRecord(cuda_epoch_stop);
        cudaEventSynchronize(cuda_epoch_stop);
        cudaEventElapsedTime(&gpu_epoch_time_ms, cuda_epoch_start, cuda_epoch_stop);
        epoch_times_s.push_back(gpu_epoch_time_ms / 1000.0);

        std::cout << "  Epoch Time: "  << epoch_times_s.back() << " s" << std::endl;

        if (batches_processed_this_epoch > 0) 
        {
            std::cout << "  Avg Batch Timings (ms): "
                      << "Load: "  << (epoch_data_load_s * 1000.0 / batches_processed_this_epoch)
                      << " | Fwd: "  << (epoch_forward_s * 1000.0 / batches_processed_this_epoch)
                      << " | LossGPU: " << (epoch_loss_gpu_s * 1000.0 / batches_processed_this_epoch)
                      << " | LossHost: "  << (epoch_loss_host_s * 1000.0 / batches_processed_this_epoch)
                      << " | Bwd: " << (epoch_backward_s * 1000.0 / batches_processed_this_epoch)
                      << " | Update: " << (epoch_update_s * 1000.0 / batches_processed_this_epoch)
                      << " | AccCalc: "  << (epoch_acc_calc_s * 1000.0 / batches_processed_this_epoch)
                      << std::endl;
        }

        total_data_load_s += epoch_data_load_s;
        total_forward_s += epoch_forward_s;
        total_loss_gpu_s += epoch_loss_gpu_s;
        total_loss_host_s += epoch_loss_host_s;
        total_backward_s += epoch_backward_s;
        total_update_s += epoch_update_s;
        total_acc_calc_s += epoch_acc_calc_s;

        float current_gpu_mem_epoch = get_current_gpu_memory_usage_mb(true);
        if (current_gpu_mem_epoch >= 0) 
        {
            std::cout << "  GPU Mem Used (MB): "<< current_gpu_mem_epoch << std::endl;
        }

    } 

    cudaEventRecord(cuda_stop);
    cudaEventSynchronize(cuda_stop);
    float gpu_train_time = 0;
    cudaEventElapsedTime(&gpu_train_time, cuda_start, cuda_stop);
    std::cout << "\nTraining complete! Total GPU Training Time: " << gpu_train_time << " ms\n";


    std::cout << "\n===== Final Validation (Autoencoder) =====" << std::endl;
    cudaEventRecord(cuda_val_start);

    float total_val_loss = 0.0f;
    int val_batches_processed = 0;
    size_t total_val_samples_processed = 0; 
    
    std::vector<float> first_batch_inputs_h;
    std::vector<float> first_batch_outputs_h;
    int first_batch_actual_size = 0;

    std::cout<< "Validating on test dataset...\n";

    if (!test_dataset || test_dataset->size() == 0) 
    {
        std::cerr << "ERROR: Test dataset is null or empty before validation loop!" << std::endl;
    } 
    else 
    {
        std::cout << "Test dataset size before loop: " << test_dataset->size() << std::endl;
        size_t expected_batches = (test_dataset->size() + batch_size - 1) / batch_size;
        std::cout << "Expected validation batches: " << expected_batches << std::endl;
    }
    std::cout << "Test loader pointer before loop: " << &test_loader << std::endl;
    
    std::shared_ptr<Batch> val_batch;
    while ((val_batch = test_loader.next_batch()) != nullptr) 
    {
        if (val_batch->batch_size == 0) 
        {
             continue;
        }
        if (!val_batch->d_inputs) 
        {
             std::cerr << "  ERROR: Validation batch data pointer d_inputs is null! Skipping batch." << std::endl;
             continue;
        }
        
        int current_val_batch_size = val_batch->batch_size;
        total_val_samples_processed += current_val_batch_size;

        nn.forward(val_batch->d_inputs, current_val_batch_size, true);

        size_t val_output_elements = (size_t)current_val_batch_size * channels * image_height * image_width;
        Layer* last_layer = nn.layers.back();
        float* d_output = last_layer->output;
        float* d_target = val_batch->d_inputs;

        nn.compute_loss(d_output, d_target, d_element_loss, val_output_elements, loss_type);

        if (h_element_loss.size() < val_output_elements) 
        {
             h_element_loss.resize(val_output_elements);
        }
        CUDA_CHECK_ERROR(cudaMemcpy(h_element_loss.data(), d_element_loss, val_output_elements * sizeof(float), cudaMemcpyDeviceToHost));

        float current_batch_loss_sum = std::accumulate(h_element_loss.begin(), h_element_loss.begin() + val_output_elements, 0.0f);
        float batch_avg_val_loss = (val_output_elements > 0) ? (current_batch_loss_sum / val_output_elements) : 0.0f;

        total_val_loss += batch_avg_val_loss;

        
        val_batches_processed++;
    } 

    float final_avg_val_loss = (val_batches_processed > 0) ? (total_val_loss / val_batches_processed) : 0.0f;

    cudaEventRecord(cuda_val_stop);
    cudaEventSynchronize(cuda_val_stop);
    float gpu_val_time = 0;
    cudaEventElapsedTime(&gpu_val_time, cuda_val_start, cuda_val_stop);

    std::cout << "Validation complete. GPU Validation Time: " << gpu_val_time << " ms\n";
    std::cout << "---------------------------------------\n";
    std::cout << "Final Average Validation Loss (Avg/Sample - " << loss_type << "): " << std::fixed << std::setprecision(6) << final_avg_val_loss << "\n";
    std::cout << "---------------------------------------\n";

    float avg_latency_ms = (total_train_batches_processed > 0 && gpu_train_time > 0) ? (gpu_train_time / total_train_batches_processed) : 0.0f;
    float throughput_samples_s = (gpu_train_time > 0) ? (static_cast<float>(train_dataset->size() * num_epochs) / (gpu_train_time / 1000.0f)) : 0.0f;
    float avg_epoch_time_s = !epoch_times_s.empty() ? (std::accumulate(epoch_times_s.begin(), epoch_times_s.end(), 0.0f) / epoch_times_s.size()) : 0.0f;

    float peak_cpu_mem = get_peak_cpu_memory_mb();
    float current_gpu_mem = get_current_gpu_memory_usage_mb(use_cuda);

    std::cout << "\n--- Performance Metrics ---" << std::endl;
    std::cout << "Total GPU Train Time (ms): "  << gpu_train_time << std::endl;
    std::cout << "Total GPU Val Time (ms):   "  << gpu_val_time << std::endl;
    std::cout << "Avg. Latency/Batch (ms): "  << avg_latency_ms << std::endl;
    std::cout << "Throughput (Samples/sec):"  << throughput_samples_s << std::endl;
    std::cout << "Avg. Time per Epoch (s): "  << avg_epoch_time_s << std::endl;
    if (peak_cpu_mem >= 0) 
    {
        std::cout << "Peak CPU Memory (MB):    "  << peak_cpu_mem << " (Linux VmHWM)" << std::endl;
    } 
    else 
    {
        std::cout << "Peak CPU Memory (MB):    N/A (Couldn't read /proc/self/status or not Linux)" << std::endl;
    }
    if (current_gpu_mem >= 0) 
    {
         std::cout << "Current GPU Memory (MB): " << current_gpu_mem << " (Used at end)" << std::endl;
    } 
    else 
    {
         std::cout << "Current GPU Memory (MB): N/A (CUDA error)" << std::endl;
    }
    if (total_train_batches_processed > 0) 
    {
        std::cout << "--- Avg Batch Breakdown (ms) ---" << std::endl;
        std::cout << "  Data Load: "  << (total_data_load_s * 1000.0 / total_train_batches_processed) << std::endl;
        std::cout << "  Forward:   "  << (total_forward_s * 1000.0 / total_train_batches_processed) << std::endl;
        std::cout << "  Loss GPU:  "  << (total_loss_gpu_s * 1000.0 / total_train_batches_processed) << std::endl;
        std::cout << "  Loss Host: "  << (total_loss_host_s * 1000.0 / total_train_batches_processed) << std::endl;
        std::cout << "  Backward:  "  << (total_backward_s * 1000.0 / total_train_batches_processed) << std::endl;
        std::cout << "  Update:    "  << (total_update_s * 1000.0 / total_train_batches_processed) << std::endl;
    }
    
    if (!first_batch_inputs_h.empty() && !first_batch_outputs_h.empty()) 
    {
        display_ae_samples(first_batch_inputs_h,
                           first_batch_outputs_h,
                           first_batch_actual_size,
                           channels,
                           image_height,
                           image_width,
                           use_grayscale,
                           loss_type,
                           5);
    } 
    else 
    {
        std::cout << "No validation samples to display (validation set might be empty or failed to load)." << std::endl;
    }

    std::cout << "\nCleaning up ConvAE resources...\n";
    safeCudaFree(&d_element_loss, "d_element_loss (ConvAE)");

    auto program_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(program_end - program_start);

    std::cout << "\nExecution Summary:\n"
              << "----------------\n"
              << "Total program time:   " << total_duration.count() << " ms\n";

    cudaEventDestroy(cuda_start);
    cudaEventDestroy(cuda_stop);
    cudaEventDestroy(cuda_epoch_start);
    cudaEventDestroy(cuda_epoch_stop);
    cudaEventDestroy(cuda_val_start);
    cudaEventDestroy(cuda_val_stop);
    cudaEventDestroy(batch_fwd_start);
    cudaEventDestroy(batch_fwd_stop);
    cudaEventDestroy(batch_loss_gpu_start); 
    cudaEventDestroy(batch_loss_gpu_stop);
    cudaEventDestroy(batch_bwd_start);
    cudaEventDestroy(batch_bwd_stop);
    cudaEventDestroy(batch_update_start);
    cudaEventDestroy(batch_update_stop);

    for (auto& layer : nn.layers) 
    {
        layer->clearBuffers();
        delete layer;
    }
    nn.layers.clear();

    std::cout << "ConvAE training finished." << std::endl;
}
}