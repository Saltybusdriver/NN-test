#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <iostream>
#include <vector>
#include <chrono>
#include <cublas_v2.h>
#include <curand.h>
#include <opencv4/opencv2/opencv.hpp>
#include <algorithm> // For std::max, std::min, std::sort
#include <limits>    // For std::numeric_limits
#include <cmath>     // For std::exp, std::sqrt

#include "NeuralNetwork.h"
#include "debug.h"
#include "DataLoader.h"
#include "utils.h"
#include "ImageDataset.h"
#include "autodiff.h"
#include "cuda_functions.h"
#include "Butterfly_dataset.h"
#include "optimizer.h" // Include optimizer
#include "layers.h"    // Include layer definitions

// Function to display images using OpenCV
DEFINE_CUSTOM_LOSS({
    auto diff = o - t;
    auto squared_error = square(diff);
    auto weight = Constant(1.0f) + squared_error * Constant(0.5f);
    return weight * squared_error;
})


namespace convlin{
void display_image_opencv(const std::vector<float>& image_data, int height, int width,
    bool is_grayscale = true, const std::string& label = "Image") {
    // Create OpenCV Mat for processing
    cv::Mat image;
    if (is_grayscale) {
        image = cv::Mat(height, width, CV_32FC1, const_cast<float*>(image_data.data()));
    } else {
        // Assuming image_data is [H, W, C] - need to reshape or ensure correct format
        // If data is [C, H, W], need conversion
        // For now, assume [H, W, C] which OpenCV handles directly for CV_32FC3
        image = cv::Mat(height, width, CV_32FC3, const_cast<float*>(image_data.data()));
    }

    // Normalize to 0-1 range if needed
    cv::Mat normalized;
    double min_val, max_val;
    cv::minMaxLoc(image, &min_val, &max_val);
    if (min_val < 0 || max_val > 1.0) {
        cv::normalize(image, normalized, 0, 1, cv::NORM_MINMAX);
    } else {
        normalized = image;
    }

    // Convert to 8-bit for display
    cv::Mat display_image;
    normalized.convertTo(display_image, CV_8U, 255);

    // Display image info in terminal
    std::cout << "\n=== " << label << " (" << width << "x" << height << ") ===\n";
    std::cout << "Value range: " << min_val << " to " << max_val << "\n";

    // Create a resizable window
    cv::namedWindow(label, cv::WINDOW_NORMAL);

    // For small images, resize for better visibility
    if (height < 100 || width < 100) {
        // Scale up small images (e.g., 64x64 -> 320x320)
        cv::resizeWindow(label, width * 5, height * 5);
    }

    // Display in OpenCV window
    cv::imshow(label, display_image);

    // Keep the terminal display for context but make it smaller
    const char* shades[] = {" ", "░", "▒", "▓", "█"};

    // Scale down for terminal display
    int display_height = std::min(height, 20);
    int display_width = std::min(width, 40);

    cv::Mat resized;
    // Handle grayscale vs color for resize
    cv::Mat temp_display_for_resize;
    if (display_image.channels() == 3) {
        cv::cvtColor(display_image, temp_display_for_resize, cv::COLOR_BGR2GRAY);
    } else {
        temp_display_for_resize = display_image;
    }
    cv::resize(temp_display_for_resize, resized, cv::Size(display_width, display_height));

    for (int y = 0; y < resized.rows; y++) {
        for (int x = 0; x < resized.cols; x++) {
            uchar value = resized.at<uchar>(y, x);
            int shade_idx = value * 5 / 256;
            std::cout << shades[shade_idx];
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// Custom loss function using autodiff


float calculate_custom_loss(const std::vector<float>& output, const std::vector<float>& target) {
    float total_loss = 0.0f;
    for (size_t i = 0; i < output.size(); i++) {
        auto loss_expr = autodiff::loss::CustomLoss::expression();
        total_loss += loss_expr.eval(output[i], target[i]);
    }
    return total_loss / output.size();
}



// Train a ConvNet model for image processing
void train_conv_model(int batch_size = 64, int num_epochs = 5,
                     const std::string& dataset_path = "", // dataset_path is currently unused, using hardcoded paths
                     int image_height = 32, int image_width = 32,
                     bool use_grayscale = true) {
    set_debug_level(LEVEL_WARN);
    cudaEvent_t cuda_start, cuda_stop;
    cudaEventCreate(&cuda_start);
    cudaEventCreate(&cuda_stop);

    auto program_start = std::chrono::high_resolution_clock::now();
    std::cout << "Building CNN architecture for butterfly classification...\n";
    NeuralNetwork nn(batch_size);

    // Define layer dimensions
    int in_channels = use_grayscale ? 1 : 3;

    // --- Define a deeper CNN Architecture ---
    // Layer 1: Conv + ReLU
    nn.add_layer( Conv2d(in_channels, 32, image_height, image_width, 3, 1, 1, batch_size, "relu"));
    // Assuming no pooling, dimensions remain image_height x image_width

    // Layer 2: Conv + ReLU
    nn.add_layer( Conv2d(32, 64, image_height, image_width, 3, 1, 1, batch_size, "relu"));
    // Dimensions still image_height x image_width

    // Layer 3: Conv + ReLU
    nn.add_layer( Conv2d(64, 128, image_height, image_width, 3, 1, 1, batch_size, "relu"));
    // Dimensions still image_height x image_width

    // Flatten layer to connect convolutional output to fully connected layers
    int final_conv_channels = 128; // Channels from the last Conv layer
    int conv_output_height = image_height; // Height after last Conv layer (no pooling)
    int conv_output_width = image_width;   // Width after last Conv layer (no pooling)
    nn.add_layer( Flatten(batch_size, final_conv_channels, conv_output_height, conv_output_width));

    // Calculate flattened dimension
    int flattened_size = final_conv_channels * conv_output_height * conv_output_width;
    std::cout << "Flattened size: " << flattened_size << std::endl;

    // Initialize dataset first to get number of classes
    // Using hardcoded paths as dataset_path parameter is not used inside ButterflyDataset
    auto dataset = std::make_shared<ButterflyDataset>(
        "Dataset/Training_set.csv",
        "Dataset", // Base path for images relative to CSV
        image_height, image_width,
        !use_grayscale  // use_rgb = !use_grayscale
    );

    // Get number of classes from the dataset
    int num_classes = dataset->get_num_classes();
    if (num_classes <= 0) {
        throw std::runtime_error("Could not determine the number of classes from the dataset.");
    }
    std::cout << "Dataset contains " << num_classes << " butterfly species\n";

    // Fully connected layers for multi-class classification
    nn.add_layer( Linear(flattened_size, 256, batch_size, "relu")); // Increased size
    // Optional: Add Dropout here if implemented
    nn.add_layer( Linear(256, num_classes, batch_size, "none")); // Output layer (logits)

    std::cout << "Neural network initialized with " << nn.layers.size() << " layers\n";

    // Select loss function - use cross entropy for multi-class classification
    const char* loss_type = "cross_entropy";

    std::cout << "Dataset loaded with " << dataset->size() << " images\n";
    std::cout << "Image size: " << image_height << "x" << image_width << "\n";
    std::cout << "Using grayscale: " << (use_grayscale ? "Yes" : "No") << "\n";

    // Create data loader with shuffling and multiple worker threads
    DataLoader loader(dataset, batch_size, true, 8, 6000);

    // Setup optimizer
    float learning_rate = 0.001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    std::unique_ptr<OptimizerBase> optimizer = create_optimizer("adam", learning_rate, beta1, beta2, epsilon);

    // Display dataset diagnostic
    std::cout << "\n==== DATASET DIAGNOSTIC ====\n";
    // Optionally show a sample image here
    std::cout << "==========================\n\n";

    // Start recording GPU time
    cudaEventRecord(cuda_start);

    // Training loop
    std::cout << "Starting training for " << num_epochs << " epochs...\n";
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        loader.reset();
        float epoch_loss = 0.0f;
        int batch_count = 0;

        std::cout << "Epoch " << epoch + 1 << "/" << num_epochs << "\n";

        while (loader.has_next()) {
            auto batch = loader.next_batch();
            if (!batch) break;
            
            // --- Use actual batch size from the loader ---
            int actual_batch_size = batch->batch_size;

            // Forward pass - Pass actual batch size
            nn.forward(batch->d_inputs,actual_batch_size, true);

            // Calculate and report loss
            float batch_loss = 0.0f;

            // Calculate per-element losses
            float* d_element_loss = nullptr;
            // Use actual batch size for output size calculation
            size_t output_size = actual_batch_size * num_classes;
            // Allocate buffer based on MAX batch size to avoid reallocations
            size_t max_output_size = batch_size * num_classes;
            safeCudaMalloc(&d_element_loss, max_output_size * sizeof(float), "d_el_loss");

            cudafunc::calculate_loss_values(
                nn.layers.back()->pre_activation_values, // Use logits before activation
                batch->d_targets,
                d_element_loss,
                output_size, // Pass actual total elements
                loss_type,
                actual_batch_size // Pass actual batch size for cross-entropy kernel
            );

            std::vector<float> host_losses(output_size); // Use actual size for host vector
            CUDA_CHECK_ERROR(cudaMemcpy(host_losses.data(), d_element_loss,
                                  output_size * sizeof(float), // Copy actual size
                                  cudaMemcpyDeviceToHost));

            // Average the losses
            batch_loss = 0.0f;
            if (!host_losses.empty()) {
                for (const float& loss_val : host_losses) {
                    batch_loss += loss_val;
                }
                // Average over batch items for cross-entropy (loss is already per-item)
                 batch_loss /= actual_batch_size;
            }


            // Backward pass - use the actual targets and actual batch size
            nn.backward(batch->d_targets, learning_rate, loss_type);

            // Clean up
            safeCudaFree(&d_element_loss, "d_el_loss");

            // Update parameters using optimizer
            for (auto& layer : nn.layers) {
                layer->update_params(*optimizer);
            }
            epoch_loss += batch_loss;
            batch_count++;

            if (batch_count % 10 == 0 || batch_count == 1) {
                std::cout << "  Batch " << batch_count << " (Size: " << actual_batch_size << ") Loss: " << batch_loss << std::endl;
            }
        }

        // Print epoch results
        if (batch_count > 0) {
            std::cout << "Epoch " << epoch + 1 << " completed. Average loss: "
                      << (epoch_loss / batch_count) << std::endl;
        }
    }

    std::cout << "Training complete!\n";

    // Evaluate model on a test sample
    loader.reset();
    auto final_batch = loader.next_batch();
    if (final_batch) {
        int actual_eval_batch_size = final_batch->batch_size;
        // Run forward pass through the network - Pass actual batch size
        nn.forward(final_batch->d_inputs,actual_eval_batch_size, false);

        // Get predictions (logits) for the first image in the batch
        std::vector<float> predictions(num_classes);
        CUDA_CHECK_ERROR(cudaMemcpy(predictions.data(), nn.layers.back()->output, // Use final output (logits)
                              num_classes * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // Get the actual image for display (first image in batch)
        size_t single_image_elements = image_height * image_width * in_channels;
        std::vector<float> final_input(single_image_elements);
        CUDA_CHECK_ERROR(cudaMemcpy(final_input.data(), final_batch->d_inputs,
                              single_image_elements * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // Get the ground truth target (first image in batch)
        std::vector<float> ground_truth(num_classes);
        CUDA_CHECK_ERROR(cudaMemcpy(ground_truth.data(), final_batch->d_targets,
                              num_classes * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // Display the input image
        std::cout << "Sample input image:" << std::endl;
        display_image_opencv(final_input, image_height, image_width, use_grayscale, "Input Image");

        // --- Apply Softmax manually to logits for probabilities ---
        std::vector<float> probabilities(num_classes);
        float max_logit = -std::numeric_limits<float>::infinity();
        for (float logit : predictions) {
            max_logit = std::max(max_logit, logit);
        }
        float sum_exp = 0.0f;
        for (int i = 0; i < num_classes; ++i) {
            probabilities[i] = std::exp(predictions[i] - max_logit); // Subtract max for stability
            sum_exp += probabilities[i];
        }
        // Normalize
        if (sum_exp > 1e-9) { // Avoid division by zero
             for (int i = 0; i < num_classes; ++i) {
                 probabilities[i] /= sum_exp;
             }
        }
        // --- End Softmax ---


        // Find the predicted class (max probability)
        int predicted_class = 0;
        float max_prob = probabilities[0];
        for (int i = 1; i < num_classes; i++) {
            if (probabilities[i] > max_prob) {
                max_prob = probabilities[i];
                predicted_class = i;
            }
        }

        // Find the actual/expected class from one-hot encoding
        int expected_class = 0;
        for (int i = 0; i < num_classes; i++) {
            if (ground_truth[i] > 0.5f) { // Check if the target is 1 for this class
                expected_class = i;
                break;
            }
        }

        // Get the class names
        std::string predicted_species = dataset->get_class_name(predicted_class);
        std::string expected_species = dataset->get_class_name(expected_class);

        // Show prediction results with comparison to ground truth
        std::cout << "===== PREDICTION RESULTS =====\n";
        std::cout << "EXPECTED: " << expected_species << " (Class " << expected_class << ")\n";
        std::cout << "PREDICTED: " << predicted_species << " (Class " << predicted_class << ") with confidence " << max_prob * 100.0f << "%\n";

        // Highlight if prediction was correct
        if (predicted_class == expected_class) {
            std::cout << "✓ CORRECT PREDICTION ✓\n";
        } else {
            std::cout << "✗ INCORRECT PREDICTION ✗\n";
        }

        // Print top 3 predictions
        std::cout << "\nTop 3 predictions:\n";
        std::vector<std::pair<float, int>> prob_idx;
        for (int i = 0; i < num_classes; i++) {
            prob_idx.push_back({probabilities[i], i});
        }
        std::sort(prob_idx.begin(), prob_idx.end(),
                 [](const auto& a, const auto& b) { return a.first > b.first; });

        for (int i = 0; i < std::min(3, num_classes); i++) {
            std::string class_name = dataset->get_class_name(prob_idx[i].second);
            bool is_correct = (prob_idx[i].second == expected_class);
            std::cout << i+1 << ". " << class_name << " (" << prob_idx[i].first * 100.0f << "%)";
            if (is_correct) std::cout << " ✓";
            std::cout << "\n";
        }
    }


    // Stop GPU timing and report
    cudaEventRecord(cuda_stop);
    cudaEventSynchronize(cuda_stop);

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, cuda_start, cuda_stop);

    auto program_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(program_end - program_start);

    std::cout << "\nExecution Summary:\n"
              << "----------------\n"
              << "Total program time: " << total_duration.count() << " ms\n"
              << "GPU computation time: " << gpu_time << " ms\n";

    // Cleanup
    cudaEventDestroy(cuda_start);
    cudaEventDestroy(cuda_stop);

    // Free memory in layers
    for (auto& layer : nn.layers) {
        layer->clearBuffers(); // Call existing clearBuffers
        delete layer; // Delete the layer object itself
    }
    nn.layers.clear(); // Clear the vector pointers


    std::cout << "Press any key in image window to exit...\n";
    cv::waitKey(0); // Wait for key press in the OpenCV window

    cudaDeviceSynchronize(); // Ensure all CUDA tasks are finished
    // cudaDeviceReset(); // Optional: Reset CUDA context if needed
}
} // namespace convlin