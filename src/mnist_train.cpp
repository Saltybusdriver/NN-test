#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <stdexcept>

// Project Headers
#include "mnist_train.h"
#include "DataLoader.h"
#include "MnistDataset.h"
#include "NeuralNetwork.h"
#include "optimizer.h"
#include "debug.h"
#include "utils.h"
#include "layer_proxy.h"
#include "cuda_functions.h"

namespace mnist_train
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

// --- Helper Function ---
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
    std::cout << "\n--- Sample MNIST Predictions ---" << std::endl;
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
            std::cout << current_output[j] << (j == num_classes - 1 ? "" : ", ");
        }
        std::cout << "]" << std::endl;

    }
    std::cout << "----------------------------------------------------------------" << std::endl;
}


// --- Main Training Function ---
void train_mnist_classifier(
    int batch_size, int num_epochs,
        const std::string& dataset_path,
        int image_height, int image_width,
        bool use_grayscale)
{
    // --- Parameter Validation ---
    if (!use_grayscale)
    {
        std::cerr << "Warning: MNIST is a grayscale dataset. 'use_grayscale' is forced to true." << std::endl;
    }
    if (image_height != INPUT_HEIGHT || image_width != INPUT_WIDTH)
    {
         std::cerr << "Warning: Image dimensions should be " << INPUT_HEIGHT << "x" << INPUT_WIDTH
                   << " for MNIST. Using constants." << std::endl;
    }
    use_grayscale = true;

    // --- Setup CUDA ---
    bool use_cuda = true;
    SetUseCuda(use_cuda);
    int epochs=num_epochs;
    float learning_rate=0.001f;

    std::cout << "\n=== MNIST Classifier Training ===\n";
    std::cout << "Using CUDA: " << (use_cuda ? "Yes" : "No") << std::endl;
    std::cout << "Epochs: " << epochs << ", Batch Size: " << batch_size
              << ", Learning Rate: " << learning_rate << std::endl;
    std::cout << "Dataset Path: " << dataset_path<< std::endl;

    cudaEvent_t cuda_start = nullptr, cuda_stop = nullptr, cuda_test_start = nullptr, cuda_test_stop = nullptr;
    // --- Add timers for detailed breakdown ---
    cudaEvent_t batch_fwd_start, batch_fwd_stop;
    cudaEvent_t batch_loss_start, batch_loss_stop;
    cudaEvent_t batch_bwd_start, batch_bwd_stop;
    cudaEvent_t batch_update_start, batch_update_stop;
    cudaEvent_t epoch_start_event, epoch_stop_event;

    if (use_cuda)
    {
        cudaEventCreate(&cuda_start);
        cudaEventCreate(&cuda_stop);
        cudaEventCreate(&cuda_test_start);
        cudaEventCreate(&cuda_test_stop);
        // --- Create the new events ---
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

    // --- Create Datasets ---
    std::cout << "Loading datasets..." << std::endl;
    auto dataset_load_start = std::chrono::high_resolution_clock::now();
    std::shared_ptr<MnistDataset> train_dataset;
    std::shared_ptr<MnistDataset> val_dataset;
    std::shared_ptr<MnistDataset> test_dataset;
    try
    {
        train_dataset = std::make_shared<MnistDataset>(dataset_path, "training", INPUT_WIDTH, INPUT_HEIGHT);
        val_dataset = std::make_shared<MnistDataset>(dataset_path, "testing", INPUT_WIDTH, INPUT_HEIGHT);
        test_dataset = std::make_shared<MnistDataset>(dataset_path, "testing", INPUT_WIDTH, INPUT_HEIGHT);

        auto dataset_load_end = std::chrono::high_resolution_clock::now();
        auto dataset_load_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(dataset_load_end - dataset_load_start);
        std::cout << "Datasets loaded successfully in " << dataset_load_duration_ms.count() << " ms." << std::endl;

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
        if (use_cuda)
        {
            if(cuda_start) cudaEventDestroy(cuda_start);
            if(cuda_stop) cudaEventDestroy(cuda_stop);
            if(cuda_test_start) cudaEventDestroy(cuda_test_start);
            if(cuda_test_stop) cudaEventDestroy(cuda_test_stop);
            // Destroy other events too
            if(batch_fwd_start) cudaEventDestroy(batch_fwd_start);
            if(batch_fwd_stop) cudaEventDestroy(batch_fwd_stop);
            if(batch_loss_start) cudaEventDestroy(batch_loss_start);
            if(batch_loss_stop) cudaEventDestroy(batch_loss_stop);
            if(batch_bwd_start) cudaEventDestroy(batch_bwd_start);
            if(batch_bwd_stop) cudaEventDestroy(batch_bwd_stop);
            if(batch_update_start) cudaEventDestroy(batch_update_start);
            if(batch_update_stop) cudaEventDestroy(batch_update_stop);
            if(epoch_start_event) cudaEventDestroy(epoch_start_event);
            if(epoch_stop_event) cudaEventDestroy(epoch_stop_event);
        }
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
         if (use_cuda)
         {
             if(cuda_start) cudaEventDestroy(cuda_start);
             if(cuda_stop) cudaEventDestroy(cuda_stop);
             if(cuda_test_start) cudaEventDestroy(cuda_test_start);
             if(cuda_test_stop) cudaEventDestroy(cuda_test_stop);
             // Destroy other events too
             if(batch_fwd_start) cudaEventDestroy(batch_fwd_start);
             if(batch_fwd_stop) cudaEventDestroy(batch_fwd_stop);
             if(batch_loss_start) cudaEventDestroy(batch_loss_start);
             if(batch_loss_stop) cudaEventDestroy(batch_loss_stop);
             if(batch_bwd_start) cudaEventDestroy(batch_bwd_start);
             if(batch_bwd_stop) cudaEventDestroy(batch_bwd_stop);
             if(batch_update_start) cudaEventDestroy(batch_update_start);
             if(batch_update_stop) cudaEventDestroy(batch_update_stop);
             if(epoch_start_event) cudaEventDestroy(epoch_start_event);
             if(epoch_stop_event) cudaEventDestroy(epoch_stop_event);
         }
         return;
    }


    std::unique_ptr<OptimizerBase> optimizer = create_optimizer("adam", learning_rate, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON);
    std::cout << "Using Adam Optimizer (LR=" << learning_rate << ")" << std::endl;

    // --- Loss Setup ---
    const char* loss_type = "crossentropy";
    std::cout << "Using Loss: " << loss_type << std::endl;

    // --- Allocate Loss Buffers ---
    float* d_element_loss = nullptr;
    size_t max_output_elements = batch_size * NUM_CLASSES;
    std::vector<float> h_element_loss(max_output_elements);

    if (use_cuda)
    {
        safeCudaMalloc(&d_element_loss, max_output_elements * sizeof(float), "d_element_loss (MNIST)");
    }
    else
    {
        d_element_loss = h_element_loss.data();
    }


    std::cout << "Starting training..." << std::endl;
    std::cout << std::fixed << std::setprecision(6);

    if (use_cuda) cudaEventRecord(cuda_start);
    auto train_start_overall_chrono = std::chrono::high_resolution_clock::now();

    // --- Variables for Metrics ---
    size_t total_train_batches_processed = 0;
    std::vector<double> epoch_times_s;
    // --- Add accumulators for detailed timings (total) ---
    double total_data_load_s = 0.0;
    double total_forward_s = 0.0;
    double total_loss_gpu_s = 0.0;
    double total_loss_host_s = 0.0;
    double total_backward_s = 0.0;
    double total_update_s = 0.0;
    double total_acc_calc_s = 0.0;

    // --- Training Loop ---
    std::shared_ptr<Batch> batch;
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        auto epoch_start_chrono = std::chrono::high_resolution_clock::now();
        if (use_cuda) cudaEventRecord(epoch_start_event);

        float epoch_loss = 0.0f;
        float epoch_accuracy = 0.0f;
        size_t batches_processed_this_epoch = 0;
        train_loader.reset();

        // --- Reset epoch timing accumulators ---
        double epoch_data_load_s = 0.0;
        double epoch_forward_s = 0.0;
        double epoch_loss_gpu_s = 0.0;
        double epoch_loss_host_s = 0.0;
        double epoch_backward_s = 0.0;
        double epoch_update_s = 0.0;
        double epoch_acc_calc_s = 0.0;
        // --- Temporary variables for batch timings ---
        float batch_fwd_time_ms = 0.0f;
        float batch_loss_gpu_time_ms = 0.0f;
        float batch_bwd_time_ms = 0.0f;
        float batch_update_time_ms = 0.0f;

        std::cout << "\n--- Epoch " << (epoch + 1) << "/" << epochs << " ---" << std::endl;

        while ((batch = train_loader.next_batch()) != nullptr)
        {
            // --- Time Data Loading ---
            auto dload_start = std::chrono::high_resolution_clock::now();
            if (!batch || batch->batch_size == 0) continue;
            auto dload_end = std::chrono::high_resolution_clock::now();
            epoch_data_load_s += std::chrono::duration<double>(dload_end - dload_start).count();
            // --- End Data Loading Time ---

            float* inputs = use_cuda ? batch->d_inputs : batch->inputs_flattened.data();
            float* targets = use_cuda ? batch->d_targets : batch->targets_flattened.data();
            size_t current_batch_size = batch->batch_size;

            if (!inputs || !targets)
            {
                 std::cerr << "Warning: Skipping batch with null data pointers." << std::endl;
                 continue;
            }

            // --- Time Forward Pass ---
            if (use_cuda) cudaEventRecord(batch_fwd_start);
            auto fwd_chrono_start = std::chrono::high_resolution_clock::now();
            network.forward(inputs, current_batch_size, use_cuda);
            auto fwd_chrono_end = std::chrono::high_resolution_clock::now();
            if (use_cuda) cudaEventRecord(batch_fwd_stop);
            else epoch_forward_s += std::chrono::duration<double>(fwd_chrono_end - fwd_chrono_start).count();
            // --- End Forward Pass Time ---

            float* outputs = network.layers.back()->output;
            if (!outputs)
            {
                 std::cerr << "Warning: Skipping batch due to null network output." << std::endl;
                 continue;
            }

            size_t current_output_elements = current_batch_size * NUM_CLASSES;
            float* current_loss_buffer_ptr = use_cuda ? d_element_loss : h_element_loss.data();

            // --- Time Loss Computation (GPU/CPU) ---
            if (use_cuda) cudaEventRecord(batch_loss_start);
            auto loss_comp_chrono_start = std::chrono::high_resolution_clock::now();
            network.compute_loss(outputs, targets, current_loss_buffer_ptr, current_output_elements, loss_type);
            auto loss_comp_chrono_end = std::chrono::high_resolution_clock::now();
            if (use_cuda) cudaEventRecord(batch_loss_stop);
            else epoch_loss_gpu_s += std::chrono::duration<double>(loss_comp_chrono_end - loss_comp_chrono_start).count();
            // --- End Loss Computation Time ---

            // --- Time Loss Summation (Host) ---
            float batch_loss_sum = 0.0f;
            auto loss_sum_start = std::chrono::high_resolution_clock::now();
            if (use_cuda)
            {
                if (h_element_loss.size() < current_output_elements)
                {
                    h_element_loss.resize(current_output_elements);
                }
                CUDA_CHECK_ERROR(cudaMemcpy(h_element_loss.data(), d_element_loss, current_output_elements * sizeof(float), cudaMemcpyDeviceToHost));
            }
            batch_loss_sum = std::accumulate(h_element_loss.begin(), h_element_loss.begin() + current_output_elements, 0.0f);
            auto loss_sum_end = std::chrono::high_resolution_clock::now();
            epoch_loss_host_s += std::chrono::duration<double>(loss_sum_end - loss_sum_start).count();
            // --- End Loss Summation Time ---

            float batch_avg_loss = (current_batch_size > 0) ? (batch_loss_sum / current_batch_size) : 0.0f;

            // --- Time Backward Pass ---
            if (use_cuda) cudaEventRecord(batch_bwd_start);
            auto bwd_chrono_start = std::chrono::high_resolution_clock::now();
            network.backward(targets, 0.0f, loss_type);
            auto bwd_chrono_end = std::chrono::high_resolution_clock::now();
            if (use_cuda) cudaEventRecord(batch_bwd_stop);
            else epoch_backward_s += std::chrono::duration<double>(bwd_chrono_end - bwd_chrono_start).count();
            // --- End Backward Pass Time ---

            // --- Time Optimizer Update ---
            if (use_cuda) cudaEventRecord(batch_update_start);
            auto update_chrono_start = std::chrono::high_resolution_clock::now();
            for (auto& layer : network.layers)
            {
                layer->update_params(*optimizer);
            }
            auto update_chrono_end = std::chrono::high_resolution_clock::now();
            if (use_cuda) cudaEventRecord(batch_update_stop);
            else epoch_update_s += std::chrono::duration<double>(update_chrono_end - update_chrono_start).count();
            // --- End Optimizer Update Time ---

            // --- Time Accuracy Calculation ---
            float accuracy = 0.0f;
            auto acc_calc_start = std::chrono::high_resolution_clock::now();
            if (use_cuda)
            {
                 std::vector<float> host_outputs(current_output_elements);
                 std::vector<float> host_targets(current_output_elements);
                 CUDA_CHECK_ERROR(cudaMemcpy(host_outputs.data(), outputs, current_output_elements * sizeof(float), cudaMemcpyDeviceToHost));
                 CUDA_CHECK_ERROR(cudaMemcpy(host_targets.data(), targets, current_output_elements * sizeof(float), cudaMemcpyDeviceToHost));
                 accuracy = calculate_accuracy(host_outputs.data(), host_targets.data(), current_batch_size, NUM_CLASSES);
            }
            else
            {
                 accuracy = calculate_accuracy(outputs, targets, current_batch_size, NUM_CLASSES);
            }
            auto acc_calc_end = std::chrono::high_resolution_clock::now();
            epoch_acc_calc_s += std::chrono::duration<double>(acc_calc_end - acc_calc_start).count();
            // --- End Accuracy Calculation Time ---

            // --- Synchronize and Accumulate GPU Timings ---
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
            // --- End Accumulate GPU Timings ---

            epoch_loss += batch_avg_loss;
            epoch_accuracy += accuracy;
            batches_processed_this_epoch++;
            total_train_batches_processed++;

            if (batches_processed_this_epoch % 10 == 0 || batches_processed_this_epoch == 1)
            {
                std::cout << "  Batch " << std::setw(4) << batches_processed_this_epoch << "/" << train_loader.get_num_batches()
                          << " Loss: "<< batch_avg_loss
                          << " Acc: "  << accuracy << "\r" << std::flush;
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
            std::cout << "  Train Loss: " << avg_epoch_loss
                      << ", Train Accuracy: "  << avg_epoch_acc << std::endl;
        }
        else
        {
            std::cout << "  No training batches processed this epoch." << std::endl;
        }

        // --- Epoch Timing ---
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
        std::cout << "  Epoch Time: " << epoch_times_s.back() << " s" << std::endl;

        // --- Report Detailed Epoch Timings ---
        if (batches_processed_this_epoch > 0)
        {
            std::cout << "  Avg Batch Timings (ms): "
                      << "Load: "  << (epoch_data_load_s * 1000.0 / batches_processed_this_epoch)
                      << " | Fwd: "  << (epoch_forward_s * 1000.0 / batches_processed_this_epoch)
                      << " | LossGPU: "  << (epoch_loss_gpu_s * 1000.0 / batches_processed_this_epoch)
                      << " | LossHost: "  << (epoch_loss_host_s * 1000.0 / batches_processed_this_epoch)
                      << " | Bwd: "  << (epoch_backward_s * 1000.0 / batches_processed_this_epoch)
                      << " | Update: "  << (epoch_update_s * 1000.0 / batches_processed_this_epoch)
                      << " | AccCalc: "  << (epoch_acc_calc_s * 1000.0 / batches_processed_this_epoch)
                      << std::endl;
        }

        // --- Accumulate total times ---
        total_data_load_s += epoch_data_load_s;
        total_forward_s += epoch_forward_s;
        total_loss_gpu_s += epoch_loss_gpu_s;
        total_loss_host_s += epoch_loss_host_s;
        total_backward_s += epoch_backward_s;
        total_update_s += epoch_update_s;
        total_acc_calc_s += epoch_acc_calc_s;

        // --- Validation Loop ---
        // ... (validation loop remains mostly the same) ...

    } // End Epoch Loop

    // --- Timing & Final Output ---
    float gpu_train_time_ms = 0;
    if (use_cuda)
    {
        cudaEventRecord(cuda_stop);
        cudaEventSynchronize(cuda_stop);
        cudaEventElapsedTime(&gpu_train_time_ms, cuda_start, cuda_stop);
        std::cout << "\nTraining complete! Total GPU Training Time: "  << gpu_train_time_ms << " ms\n";
    }
    else
    {
         auto train_end_overall_chrono = std::chrono::high_resolution_clock::now();
         gpu_train_time_ms = std::chrono::duration<double, std::milli>(train_end_overall_chrono - train_start_overall_chrono).count();
         std::cout << "\nTraining complete! Total CPU Training Time: "  << gpu_train_time_ms << " ms\n";
    }


    // --- Testing Phase ---
    std::cout << "\n--- Testing ---" << std::endl;
    if (use_cuda) cudaEventRecord(cuda_test_start);

    float test_loss = 0.0f;
    float test_accuracy = 0.0f;
    size_t test_batches = 0;
    float final_test_loss = 0.0f;
    float final_test_acc = 0.0f;
    test_loader.reset();

    while ((batch = test_loader.next_batch()) != nullptr)
    {
         if (!batch || batch->batch_size == 0) continue;
         float* inputs = use_cuda ? batch->d_inputs : batch->inputs_flattened.data();
         float* targets = use_cuda ? batch->d_targets : batch->targets_flattened.data();
         size_t current_batch_size = batch->batch_size;
         if (!inputs || !targets) continue;

         network.forward(inputs, current_batch_size, use_cuda);
         float* outputs = network.layers.back()->output;
         if (!outputs) continue;

         size_t current_output_elements = current_batch_size * NUM_CLASSES;
         float* current_d_element_loss_ptr = use_cuda ? d_element_loss : h_element_loss.data();
         network.compute_loss(outputs, targets, current_d_element_loss_ptr, current_output_elements, loss_type);

         float batch_loss_sum = 0.0f;
         if (use_cuda)
         {
             CUDA_CHECK_ERROR(cudaMemcpy(h_element_loss.data(), d_element_loss, current_output_elements * sizeof(float), cudaMemcpyDeviceToHost));
         }
         batch_loss_sum = std::accumulate(h_element_loss.begin(), h_element_loss.begin() + current_output_elements, 0.0f);
         float batch_avg_loss = (current_batch_size > 0) ? (batch_loss_sum / current_batch_size) : 0.0f;
         test_loss += batch_avg_loss;

         float accuracy = 0.0f;
         if (use_cuda)
         {
             std::vector<float> host_outputs(current_output_elements);
             std::vector<float> host_targets(current_output_elements);
             CUDA_CHECK_ERROR(cudaMemcpy(host_outputs.data(), outputs, current_output_elements * sizeof(float), cudaMemcpyDeviceToHost));
             CUDA_CHECK_ERROR(cudaMemcpy(host_targets.data(), targets, current_output_elements * sizeof(float), cudaMemcpyDeviceToHost));
             accuracy = calculate_accuracy(host_outputs.data(), host_targets.data(), current_batch_size, NUM_CLASSES);
         }
         else
         {
             accuracy = calculate_accuracy(outputs, targets, current_batch_size, NUM_CLASSES);
         }
         test_accuracy += accuracy;
         test_batches++;
    }

    float gpu_test_time = 0;
    if (use_cuda)
    {
        cudaEventRecord(cuda_test_stop);
        cudaEventSynchronize(cuda_test_stop);
        cudaEventElapsedTime(&gpu_test_time, cuda_test_start, cuda_test_stop);
    }

    if (test_batches > 0)
    {
        final_test_loss = test_loss / test_batches;
        final_test_acc = test_accuracy / test_batches;
        std::cout << "Final Test Loss: "  << final_test_loss
                  << ", Final Test Accuracy: " << final_test_acc << std::endl;
    }
    else
    {
        std::cout << "No test batches were processed." << std::endl;
    }
     if (use_cuda)
     {
         std::cout << "GPU Testing Time: " << gpu_test_time << " ms\n";
     }
     else
     {
         std::cout << "(CPU Mode Testing)\n";
     }

    // --- Print Sample Predictions ---
    std::cout << "\n--- Generating Samples for Prediction Output ---" << std::endl;
    test_loader.reset();
    std::shared_ptr<Batch> sample_batch = test_loader.next_batch();

    if (sample_batch && sample_batch->batch_size > 0)
    {
        size_t sample_batch_size = sample_batch->batch_size;
        size_t sample_output_elements = sample_batch_size * NUM_CLASSES;

        std::vector<float> h_sample_targets(sample_output_elements);
        std::vector<float> h_sample_outputs(sample_output_elements);

        network.forward(use_cuda ? sample_batch->d_inputs : sample_batch->inputs_flattened.data(),
                        sample_batch_size, use_cuda);
        const float* d_sample_outputs = network.layers.back()->output;

        if (d_sample_outputs)
        {
            if (use_cuda)
            {
                CUDA_CHECK_ERROR(cudaMemcpy(h_sample_targets.data(), sample_batch->d_targets, sample_output_elements * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK_ERROR(cudaMemcpy(h_sample_outputs.data(), d_sample_outputs, sample_output_elements * sizeof(float), cudaMemcpyDeviceToHost));
            }
            else
            {
                memcpy(h_sample_targets.data(), sample_batch->targets_flattened.data(), sample_output_elements * sizeof(float));
                memcpy(h_sample_outputs.data(), d_sample_outputs, sample_output_elements * sizeof(float));
            }

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
    float avg_latency_ms = (total_train_batches_processed > 0 && gpu_train_time_ms > 0) ? (gpu_train_time_ms / total_train_batches_processed) : 0.0f;
    float throughput_samples_s = (gpu_train_time_ms > 0) ? (static_cast<float>(train_dataset->size() * num_epochs) / (gpu_train_time_ms / 1000.0f)) : 0.0f;
    float avg_epoch_time_s = !epoch_times_s.empty() ? (std::accumulate(epoch_times_s.begin(), epoch_times_s.end(), 0.0f) / epoch_times_s.size()) : 0.0f;

    // --- Get Memory Usage at End ---
    float peak_cpu_mem = get_peak_cpu_memory_mb();
    float current_gpu_mem = get_current_gpu_memory_usage_mb(use_cuda);

    std::cout << "\n--- Final Summary ---" << std::endl;
    if (test_batches > 0)
    {
        std::cout << "Test Loss:               "<< final_test_loss << std::endl;
        std::cout << "Test Accuracy:           " << final_test_acc << std::endl;
    }
    else
    {
        std::cout << "Test Loss:               N/A (No test batches)" << std::endl;
        std::cout << "Test Accuracy:           N/A (No test batches)" << std::endl;
    }
    std::cout << "--- Performance Metrics ---" << std::endl;
    if (use_cuda)
    {
        std::cout << "Total GPU Train Time (ms): " << gpu_train_time_ms << std::endl;
        std::cout << "Total GPU Test Time (ms):  "  << gpu_test_time << std::endl;
    }
    else
    {
        std::cout << "Total CPU Train Time (ms): "<< gpu_train_time_ms << std::endl;
        std::cout << "Total CPU Test Time (ms):  N/A (Not measured separately)" << std::endl;
    }
    std::cout << "Avg. Latency/Batch (ms): "<< avg_latency_ms << std::endl;
    std::cout << "Throughput (Samples/sec):" << throughput_samples_s << std::endl;
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
        std::cout << "Current GPU Memory (MB): N/A (CUDA error or not using CUDA)" << std::endl;
    }
    if (total_train_batches_processed > 0)
    {
        std::cout << "--- Avg Batch Breakdown (ms) ---" << std::endl;
        std::cout << "  Data Load: " << (total_data_load_s * 1000.0 / total_train_batches_processed) << std::endl;
        std::cout << "  Forward:   " << (total_forward_s * 1000.0 / total_train_batches_processed) << std::endl;
        std::cout << "  Loss GPU:  " << (total_loss_gpu_s * 1000.0 / total_train_batches_processed) << std::endl;
        std::cout << "  Loss Host: " << (total_loss_host_s * 1000.0 / total_train_batches_processed) << std::endl;
        std::cout << "  Backward:  " << (total_backward_s * 1000.0 / total_train_batches_processed) << std::endl;
        std::cout << "  Update:    " << (total_update_s * 1000.0 / total_train_batches_processed) << std::endl;
        std::cout << "  Acc Calc:  " << (total_acc_calc_s * 1000.0 / total_train_batches_processed) << std::endl;
    }
    // --- Cleanup ---
    std::cout << "\nCleaning up MNIST resources...\n";
    if (use_cuda)
    {
        safeCudaFree(&d_element_loss, "d_element_loss (MNIST)");
        if(cuda_start) cudaEventDestroy(cuda_start);
        if(cuda_stop) cudaEventDestroy(cuda_stop);
        if(cuda_test_start) cudaEventDestroy(cuda_test_start);
        if(cuda_test_stop) cudaEventDestroy(cuda_test_stop);
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
    }

    for (auto& layer : network.layers)
    {
        delete layer;
    }
    network.layers.clear();

    auto program_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(program_end - program_start);
    std::cout << "Total MNIST function time: " << total_duration.count() << " ms\n";
    std::cout << "MNIST training finished." << std::endl;
}

} // namespace mnist_train