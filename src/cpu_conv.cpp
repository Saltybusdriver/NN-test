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
#include <cstring>

#include "NeuralNetwork.h"
#include "debug.h"
#include "DataLoader.h"
#include "utils.h"
#include "ImageDataset.h"
#include "autodiff.h"
#include "cuda_functions.h"
#include "Butterfly_dataset.h"
#include "cpu_utils.h"
#include "optimizer.h"
#include "layer_proxy.h"

DEFINE_CUSTOM_LOSS({
    auto diff = o - t;
    auto squared_error = square(diff);
    auto weight = Constant(1.0f) + squared_error * Constant(0.5f);
    return weight * squared_error;
})


namespace conv_cpu
{
std::vector<std::string> display_image_opencv(
    const std::vector<float>& image_data, int height, int width,
    bool is_grayscale = true, const std::string& label = "Image", bool show_window = true)
{
    std::vector<std::string> terminal_lines;

    cv::Mat image;
    int channels = is_grayscale ? 1 : 3;
    size_t expected_elements = (size_t)height * width * channels;

    if (image_data.size() != expected_elements)
    {
        std::cerr << "Error in display_image_opencv: image_data size (" << image_data.size()
                  << ") does not match expected size (" << expected_elements << ") for "
                  << height << "x" << width << "x" << channels << std::endl;
        image = cv::Mat::zeros(height, width, is_grayscale ? CV_32FC1 : CV_32FC3);
    }
    else if (is_grayscale)
    {
        image = cv::Mat(height, width, CV_32FC1, const_cast<float*>(image_data.data()));
    }
    else
    {
        cv::Mat temp_image(height, width, CV_32FC3);
        size_t plane_size = (size_t)height * width;
        std::vector<cv::Mat> channel_mats;
        for(int c=0; c<channels; ++c)
        {
            channel_mats.push_back(cv::Mat(height, width, CV_32F, const_cast<float*>(image_data.data() + c * plane_size)));
        }
        cv::merge(channel_mats, temp_image);
        image = temp_image;
    }

    cv::Mat normalized;
    double min_val, max_val;
    cv::minMaxLoc(image, &min_val, &max_val);
    if (min_val < 0.0 || max_val > 1.0)
    {
        if (min_val == max_val)
        {
             normalized = cv::Mat::zeros(image.size(), image.type());
             if (min_val > 0.5)
             {
                 normalized.setTo(cv::Scalar::all(1.0));
             }
        }
        else
        {
            cv::normalize(image, normalized, 0.0, 1.0, cv::NORM_MINMAX);
        }
    }
    else
    {
        normalized = image.clone();
    }

    cv::Mat display_image_8u;
    normalized.convertTo(display_image_8u, CV_8U, 255.0);

    if (show_window)
    {
        cv::namedWindow(label, cv::WINDOW_NORMAL);
        int display_window_width = 512;
        int display_window_height = 512;
        cv::resizeWindow(label, display_window_width, display_window_height);
        cv::Mat resized_for_display;
        cv::resize(display_image_8u, resized_for_display, cv::Size(display_window_width, display_window_height), 0, 0, cv::INTER_NEAREST);
        cv::imshow(label, resized_for_display);
    }

    const char* shades[] = {" ", "░", "▒", "▓", "█"};
    const int MAX_TERM_HEIGHT = 15;
    const int MAX_TERM_WIDTH = 30;

    int terminal_display_height = std::min(height, MAX_TERM_HEIGHT);
    int terminal_display_width = std::min(width, MAX_TERM_WIDTH);

    cv::Mat resized_for_terminal;

    cv::Mat temp_gray_for_terminal;
    if (display_image_8u.channels() == 3)
    {
        cv::cvtColor(display_image_8u, temp_gray_for_terminal, cv::COLOR_BGR2GRAY);
    }
    else
    {
        temp_gray_for_terminal = display_image_8u;
    }
    cv::resize(temp_gray_for_terminal, resized_for_terminal, cv::Size(terminal_display_width, terminal_display_height), 0, 0, cv::INTER_NEAREST);

    for (int y = 0; y < resized_for_terminal.rows; y++)
    {
        std::string line = "";
        for (int x = 0; x < resized_for_terminal.cols; x++)
        {
            uchar value = resized_for_terminal.at<uchar>(y, x);
            int shade_idx = value * 5 / 256;
            shade_idx = std::max(0, std::min(4, shade_idx));
            line += shades[shade_idx];
        }
        terminal_lines.push_back(line);
    }

    return terminal_lines;
}

float calculate_sample_loss(const std::vector<float>& output, const std::vector<float>& target, const char* loss_type)
{
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
            std::string out_line = (line_idx < output_lines.size()) ? output_lines[line_idx] : std::string(output_width, ' ');
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
                     bool use_grayscale = true)
{

    set_debug_level(LEVEL_ERROR);
    SetUseCuda(false);
    std::cout << "\n=== Convolutional Autoencoder Training (CPU) ===\n";
    std::cout << "Using CUDA: No\n";
    std::cout << "Image Size: " << image_height << "x" << image_width << ", Grayscale: " << (use_grayscale ? "Yes" : "No") << "\n";
    std::cout << "Batch Size: " << batch_size << ", Epochs: " << num_epochs << "\n";

    auto program_start = std::chrono::high_resolution_clock::now();
    auto train_start = std::chrono::high_resolution_clock::now();
    auto val_start = std::chrono::high_resolution_clock::now();
    size_t total_train_batches_processed = 0;
    std::vector<double> epoch_times_s;
    double total_batch_load_time_s = 0.0;
    double total_forward_time_s = 0.0;
    double total_loss_compute_time_s = 0.0;
    double total_loss_sum_time_s = 0.0;
    double total_backward_time_s = 0.0;
    double total_update_time_s = 0.0;
    double total_acc_calc_s = 0.0;

    std::cout << "Building CNN Autoencoder architecture...\n";
    NeuralNetwork nn(batch_size);
    int channels = use_grayscale ? 1 : 3;

    nn.add_layer(Conv2d(channels, 16, image_height, image_width, 3, 1, 1, batch_size, "relu"));
    nn.add_layer(Conv2d(16, 8, image_height, image_width, 3, 1, 1, batch_size, "relu"));

    nn.add_layer(Conv2d(8, 16, image_height, image_width, 3, 1, 1, batch_size, "relu"));
    nn.add_layer(Conv2d(16, channels, image_height, image_width, 3, 1, 1, batch_size, "sigmoid"));

    std::cout << "Neural network initialized with " << nn.layers.size() << " layers\n";

    std::string root_dir = "Dataset/";
    std::string default_train_csv = root_dir + "Training_set.csv";
    std::string default_test_csv = root_dir + "Testing_set.csv";
    std::string train_csv = default_train_csv;
    std::string test_csv = default_test_csv;
    bool use_rgb = !use_grayscale;

    auto train_dataset = std::make_shared<ButterflyDataset>(train_csv, root_dir, image_width, image_height, use_rgb);
    std::cout << "Training dataset loaded with " << train_dataset->size() << " images\n";

    auto test_dataset = std::make_shared<ButterflyDataset>(test_csv, root_dir, image_width, image_height, use_rgb);
    std::cout << "Test dataset loaded with " << test_dataset->size() << " images\n";

    DataLoader loader(train_dataset, batch_size, false, 12, 6000);
    DataLoader test_loader(test_dataset, batch_size, true, 12, 6000);

    float learning_rate = 0.001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    std::unique_ptr<OptimizerBase> optimizer = create_optimizer("adam", learning_rate, beta1, beta2, epsilon);
    std::cout << "Using Adam Optimizer (LR=" << learning_rate << ")" << std::endl;

    const char* loss_type = "mse";
    std::cout << "Using loss function: " << loss_type << std::endl;

    size_t max_output_elements = (size_t)batch_size * channels * image_height * image_width;
    std::vector<float> h_element_loss_buffer(max_output_elements);

    float* d_element_loss = nullptr;

    std::cout << "Starting training for " << num_epochs << " epochs...\n";
    std::cout << std::fixed << std::setprecision(6);

    train_start = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < num_epochs; ++epoch)
    {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        auto epoch_train_start = std::chrono::high_resolution_clock::now();

        loader.reset();
        float epoch_loss = 0.0f;
        size_t batches_processed_this_epoch = 0;
        double epoch_batch_load_time_s = 0.0;
        double epoch_forward_time_s = 0.0;
        double epoch_loss_compute_time_s = 0.0;
        double epoch_loss_sum_time_s = 0.0;
        double epoch_backward_time_s = 0.0;
        double epoch_update_time_s = 0.0;
        double epoch_acc_calc_s = 0.0;

        std::cout << "\n--- Epoch " << epoch + 1 << "/" << num_epochs << " ---" << std::endl;

        while (loader.has_next())
        {
            auto dload_start = std::chrono::high_resolution_clock::now();
            auto batch = loader.next_batch();
            auto dload_end = std::chrono::high_resolution_clock::now();
            epoch_batch_load_time_s += std::chrono::duration<double>(dload_end - dload_start).count();

            if (!batch) break;
            int current_batch_size = batch->batch_size;
            if (current_batch_size <= 0) continue;
            if(batch->d_inputs == nullptr)
            {
                 std::cerr << "Error: Training batch data pointer is null for size " << current_batch_size << std::endl;
                 continue;
            }

            auto fwd_start = std::chrono::high_resolution_clock::now();
            nn.forward(batch->d_inputs, current_batch_size, false);
            auto fwd_end = std::chrono::high_resolution_clock::now();
            epoch_forward_time_s += std::chrono::duration<double>(fwd_end - fwd_start).count();

            Layer* last_layer = nn.layers.back();
            float* h_output = last_layer->output;
            float* h_target = batch->d_inputs;
            size_t output_elements = (size_t)current_batch_size * channels * image_height * image_width;

            auto loss_comp_start = std::chrono::high_resolution_clock::now();
            nn.compute_loss(h_output, h_target, h_element_loss_buffer.data(), output_elements, loss_type);
            auto loss_comp_end = std::chrono::high_resolution_clock::now();
            epoch_loss_compute_time_s += std::chrono::duration<double>(loss_comp_end - loss_comp_start).count();

            float batch_loss_sum = 0.0f;
            auto loss_sum_start = std::chrono::high_resolution_clock::now();
            if (output_elements > 0)
            {
                if (h_element_loss_buffer.size() < output_elements)
                {
                    h_element_loss_buffer.resize(output_elements);
                }
                batch_loss_sum = std::accumulate(h_element_loss_buffer.begin(), h_element_loss_buffer.begin() + output_elements, 0.0f);
            }
            auto loss_sum_end = std::chrono::high_resolution_clock::now();
            epoch_loss_sum_time_s += std::chrono::duration<double>(loss_sum_end - loss_sum_start).count();

            float batch_avg_loss_per_sample = (output_elements > 0) ? (batch_loss_sum / output_elements) : 0.0f;

            auto bwd_start = std::chrono::high_resolution_clock::now();
            nn.backward(h_target, 0.0f, loss_type);
            auto bwd_end = std::chrono::high_resolution_clock::now();
            epoch_backward_time_s += std::chrono::duration<double>(bwd_end - bwd_start).count();

            auto update_start = std::chrono::high_resolution_clock::now();
            for (auto& layer : nn.layers)
            {
                layer->update_params(*optimizer);
            }
            auto update_end = std::chrono::high_resolution_clock::now();
            epoch_update_time_s += std::chrono::duration<double>(update_end - update_start).count();

            epoch_loss += batch_avg_loss_per_sample;
            batches_processed_this_epoch++;
            total_train_batches_processed++;

            if (batches_processed_this_epoch % 50 == 0 || batches_processed_this_epoch == 1)
            {
                std::cout << "  Batch " << std::setw(4) << batches_processed_this_epoch << "/" << loader.get_num_batches()
                          << " AvgLoss/Sample: " << std::fixed << std::setprecision(6) << batch_avg_loss_per_sample << "\r" << std::flush;
            }
        }
        std::cout << std::endl;
        auto epoch_train_end = std::chrono::high_resolution_clock::now();

        total_batch_load_time_s += epoch_batch_load_time_s;
        total_forward_time_s += epoch_forward_time_s;
        total_loss_compute_time_s += epoch_loss_compute_time_s;
        total_loss_sum_time_s += epoch_loss_sum_time_s;
        total_backward_time_s += epoch_backward_time_s;
        total_update_time_s += epoch_update_time_s;

        float epoch_train_loss = 0.0f;
        if (batches_processed_this_epoch > 0)
        {
            epoch_train_loss = epoch_loss / batches_processed_this_epoch;
            std::cout << "  Train Loss (Avg/Sample): " << std::setw(10) << std::fixed << std::setprecision(6) << epoch_train_loss << std::endl;
        }
        else
        {
             std::cout << "  No training batches were processed in this epoch." << std::endl;
        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        double epoch_duration_s = std::chrono::duration<double>(epoch_end - epoch_start).count();
        epoch_times_s.push_back(epoch_duration_s);

        double train_duration_s = std::chrono::duration<double>(epoch_train_end - epoch_train_start).count();

        size_t samples_this_epoch = batches_processed_this_epoch * batch_size;
        double epoch_throughput = (train_duration_s > 0) ? (static_cast<double>(samples_this_epoch) / train_duration_s) : 0.0;

        float current_cpu_mem_epoch = get_peak_cpu_memory_mb();

        std::cout << "  Epoch Time: "<< epoch_duration_s << " s"
                  << " | Train Throughput: " << epoch_throughput << " samples/s" << std::endl;

        if (batches_processed_this_epoch > 0)
        {
            std::cout << "  Avg Batch Timings (ms): "
                      << "Load: "  << (epoch_batch_load_time_s * 1000.0 / batches_processed_this_epoch)
                      << " | Fwd: "  << (epoch_forward_time_s * 1000.0 / batches_processed_this_epoch)
                      << " | LossComp: "  << (epoch_loss_compute_time_s * 1000.0 / batches_processed_this_epoch)
                      << " | LossSum: "  << (epoch_loss_sum_time_s * 1000.0 / batches_processed_this_epoch)
                      << " | Bwd: " << (epoch_backward_time_s * 1000.0 / batches_processed_this_epoch)
                      << " | Update: " << (epoch_update_time_s * 1000.0 / batches_processed_this_epoch)
                      << std::endl;
        }

        if (current_cpu_mem_epoch >= 0)
        {
             std::cout << "  CPU Mem (Peak MB): "<< current_cpu_mem_epoch << std::endl;
        }
        else
        {
             std::cout << "  CPU Mem (Peak MB): N/A" << std::endl;
        }

    }

    auto train_end = std::chrono::high_resolution_clock::now();
    double train_time_ms = std::chrono::duration<double, std::milli>(train_end - train_start).count();
    std::cout << "\nTraining complete! Total CPU Training Time: " << train_time_ms << " ms\n";

    std::cout << "\n===== Final Validation (Autoencoder - CPU) =====" << std::endl;
    val_start = std::chrono::high_resolution_clock::now();

    float total_val_loss = 0.0f;
    int val_batches_processed = 0;
    std::vector<float> first_batch_inputs_h;
    std::vector<float> first_batch_outputs_h;
    int first_batch_actual_size = 0;

    std::cout<< "Validating on test dataset...\n";
    test_loader.reset();
    while (test_loader.has_next())
    {
        auto val_batch = test_loader.next_batch();
        if (!val_batch) break;
        int current_val_batch_size = val_batch->batch_size;
        if (current_val_batch_size <= 0) continue;
        if (val_batch->d_inputs == nullptr)
        {
             std::cerr << "Error: Validation batch data pointer is null for size " << current_val_batch_size << std::endl;
             continue;
        }

        nn.forward(val_batch->d_inputs, current_val_batch_size, false);

        size_t val_output_elements = (size_t)current_val_batch_size * channels * image_height * image_width;
        Layer* last_layer = nn.layers.back();
        float* h_output = last_layer->output;
        float* h_target = val_batch->d_inputs;

        nn.compute_loss(h_output, h_target, h_element_loss_buffer.data(), val_output_elements, loss_type);

        float current_batch_loss_sum = 0.0f;
        if (val_output_elements > 0)
        {
             if (h_element_loss_buffer.size() < val_output_elements)
             {
                 h_element_loss_buffer.resize(val_output_elements);
             }
             current_batch_loss_sum = std::accumulate(h_element_loss_buffer.begin(), h_element_loss_buffer.begin() + val_output_elements, 0.0f);
        }

        float current_batch_avg_loss_per_sample = (current_val_batch_size > 0) ? (current_batch_loss_sum / current_val_batch_size) : 0.0f;

        total_val_loss += current_batch_avg_loss_per_sample;

        if (val_batches_processed == 0)
        {
             first_batch_inputs_h.assign(h_target, h_target + val_output_elements);
             first_batch_outputs_h.assign(h_output, h_output + val_output_elements);
             first_batch_actual_size = current_val_batch_size;
        }

        val_batches_processed++;
    }

    float final_avg_val_loss = (val_batches_processed > 0) ? (total_val_loss / val_batches_processed) : 0.0f;

    auto val_end = std::chrono::high_resolution_clock::now();
    double val_time_ms = std::chrono::duration<double, std::milli>(val_end - val_start).count();

    std::cout << "Validation complete. CPU Validation Time: " << val_time_ms << " ms\n";
    std::cout << "---------------------------------------\n";
    std::cout << "Final Average Validation Loss (Avg/Sample - " << loss_type << "): " << std::fixed << std::setprecision(6) << final_avg_val_loss << "\n";
    std::cout << "---------------------------------------\n";

    double avg_latency_ms = (total_train_batches_processed > 0 && train_time_ms > 0) ? (train_time_ms / total_train_batches_processed) : 0.0;
    double throughput_samples_s = (train_time_ms > 0) ? (static_cast<double>(train_dataset->size() * num_epochs) / (train_time_ms / 1000.0)) : 0.0;
    double avg_epoch_time_s = !epoch_times_s.empty() ? (std::accumulate(epoch_times_s.begin(), epoch_times_s.end(), 0.0) / epoch_times_s.size()) : 0.0;

    float peak_cpu_mem = get_peak_cpu_memory_mb();

    std::cout << "\n--- Performance Metrics (CPU) ---" << std::endl;
    std::cout << "Total CPU Train Time (ms): "  << val_time_ms << std::endl;
    std::cout << "Avg. Latency/Batch (ms): "  << avg_latency_ms << std::endl;
    std::cout << "Throughput (Samples/sec):" << throughput_samples_s << std::endl;
    std::cout << "Avg. Time per Epoch (s): " << avg_epoch_time_s << std::endl;
    if (peak_cpu_mem >= 0)
    {
        std::cout << "Peak CPU Memory (MB):    "  << peak_cpu_mem << " (Linux VmHWM)" << std::endl;
    }
    else
    {
        std::cout << "Peak CPU Memory (MB):    N/A (Couldn't read /proc/self/status or not Linux)" << std::endl;
    }
    if (total_train_batches_processed > 0)
    {
        std::cout << "--- Avg Batch Breakdown (ms) ---" << std::endl;
        std::cout << "  Data Load: "  << (total_batch_load_time_s * 1000.0 / total_train_batches_processed) << std::endl;
        std::cout << "  Forward:   "  << (total_forward_time_s * 1000.0 / total_train_batches_processed) << std::endl;
        std::cout << "  Loss Comp: "  << (total_loss_compute_time_s * 1000.0 / total_train_batches_processed) << std::endl;
        std::cout << "  Loss Host: "  << (total_loss_sum_time_s * 1000.0 / total_train_batches_processed) << std::endl;
        std::cout << "  Backward:  "  << (total_backward_time_s* 1000.0 / total_train_batches_processed) << std::endl;
        std::cout << "  Update:    "  << (total_update_time_s * 1000.0 / total_train_batches_processed) << std::endl;
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

    std::cout << "\nCleaning up CPU ConvAE resources...\n";

    auto program_end = std::chrono::high_resolution_clock::now();
    double total_duration_ms = std::chrono::duration<double, std::milli>(program_end - program_start).count();

    std::cout << "\nExecution Summary:\n"
              << "----------------\n"
              << "Total program time:   " << std::fixed << std::setprecision(2) << total_duration_ms << " ms\n";

    for (auto& layer : nn.layers)
    {
        layer->clearBuffers();
        delete layer;
    }
    nn.layers.clear();

    std::cout << "CPU ConvAE training finished." << std::endl;
}
} // namespace conv_cpu