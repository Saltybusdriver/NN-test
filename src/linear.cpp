#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cublas_v2.h>
#include <curand.h>
#include <random>

#include "NeuralNetwork.h"
#include "debug.h"
#include "DataLoader.h"
#include "utils.h"
#include "autodiff.h"
#include "cuda_functions.h"
#include "linear.h"

DEFINE_CUSTOM_LOSS(
{
    auto diff = o - t;
    auto squared_error = square(diff);
    auto weight = Constant(1.0f) + squared_error * Constant(0.5f);
    return weight * squared_error;
})

namespace linear
{

float calculate_custom_loss(const std::vector<float>& output, const std::vector<float>& target)
{
    float total_loss = 0.0f;
    for (size_t i = 0; i < output.size(); i++)
    {
        auto loss_expr = autodiff::loss::CustomLoss::expression();
        total_loss += loss_expr.eval(output[i], target[i]);
    }
    return total_loss / output.size();
}

void generate_synthetic_data(std::vector<float>& inputs, std::vector<float>& targets,
    int batch_size, int input_size, int output_size,
    bool binary_classification, bool is_classification)
{
    inputs.resize(batch_size * input_size);
    targets.resize(batch_size * output_size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < batch_size; i++)
    {
        float sum = 0.0f;
        for (int j = 0; j < input_size; j++)
        {
            float value;
            if (j % 2 == 0)
            {
                value = 0.8f * dist(gen);
            }
            else
            {
                value = 0.3f * dist(gen);
            }
            inputs[i * input_size + j] = value;
            sum += value;
        }

        if (binary_classification && output_size >= 2)
        {
            for (int k = 0; k < output_size; k++)
            {
                targets[i * output_size + k] = 0.0f;
            }

            if (sum > 0)
            {
                targets[i * output_size] = 1.0f;
            }
            else
            {
                targets[i * output_size + 1] = 1.0f;
            }
        }
        else if (is_classification)
        {
            for (int k = 0; k < output_size; k++)
            {
                targets[i * output_size + k] = 0.0f;
            }

            int target_class = std::abs(static_cast<int>(sum * 5.0f)) % output_size;
            targets[i * output_size + target_class] = 1.0f;
        }
        else
        {
            for (int k = 0; k < output_size; k++)
            {
                targets[i * output_size + k] = std::tanh(sum) * 0.5f + 0.5f;
            }
        }
    }
}

void visualize_linear_output(const std::vector<float>& outputs, const std::vector<float>& targets, int output_size)
{
    std::cout << "\n=== Model Output Visualization ===\n";

    for (int i = 0; i < std::min(5, (int)(outputs.size() / output_size)); i++)
    {
        std::cout << "\nSample " << i << ":\n";

        for (int j = 0; j < output_size; j++)
        {
            float output = outputs[i * output_size + j];
            float target = targets[i * output_size + j];

            int out_bars = static_cast<int>(output * 40);
            int target_bars = static_cast<int>(target * 40);

            std::cout << "Class " << j << " Pred: [" << std::fixed << std::setprecision(4) << output << "] ";
            std::cout << std::string(out_bars, '|') << "\n";

            std::cout << "Class " << j << " True: [" << std::fixed << std::setprecision(4) << target << "] ";
            std::cout << std::string(target_bars, '*') << "\n";
        }
    }
}

float calculate_accuracy(const std::vector<float>& outputs, const std::vector<float>& targets, int output_size)
{
    int correct = 0;
    int total = outputs.size() / output_size;

    for (int i = 0; i < total; i++)
    {
        int predicted_class = 0;
        float max_val = outputs[i * output_size];

        for (int j = 1; j < output_size; j++)
        {
            if (outputs[i * output_size + j] > max_val)
            {
                max_val = outputs[i * output_size + j];
                predicted_class = j;
            }
        }

        int target_class = 0;
        max_val = targets[i * output_size];

        for (int j = 1; j < output_size; j++)
        {
            if (targets[i * output_size + j] > max_val)
            {
                max_val = targets[i * output_size + j];
                target_class = j;
            }
        }

        if (predicted_class == target_class)
        {
            correct++;
        }
    }

    return static_cast<float>(correct) / total;
}

void train_linear_model(int input_size = 32, int hidden_size = 64, int output_size = 10,
                       int batch_size = 128, int num_epochs = 100, bool is_classification = true)
{
    set_debug_level(LEVEL_ERROR);
    SetUseCuda(true);
    std::cout << "\n=== Linear Neural Network Training ===\n";

    cudaEvent_t cuda_start, cuda_stop;
    cudaEventCreate(&cuda_start);
    cudaEventCreate(&cuda_stop);

    auto program_start = std::chrono::high_resolution_clock::now();

    std::cout << "Building network architecture...\n";
    NeuralNetwork nn(batch_size);

    nn.add_layer( Linear(input_size, hidden_size, batch_size, "relu"));
    nn.add_layer( Linear(hidden_size, hidden_size, batch_size, "relu"));
    if (is_classification)
    {
        nn.add_layer( Linear(hidden_size, output_size, batch_size, "none"));
    }
    else
    {
        nn.add_layer( Linear(hidden_size, output_size, batch_size, "sigmoid"));
    }

    std::cout << "Neural network initialized with " << nn.layers.size() << " layers\n";

    std::vector<float> x_train, y_train;
    std::vector<float> x_val, y_val;

    std::cout << "Generating synthetic data...\n";
    generate_synthetic_data(x_train, y_train, batch_size * 10, input_size, output_size, is_classification);
    generate_synthetic_data(x_val, y_val, batch_size, input_size, output_size, is_classification);

    float* d_inputs = nullptr;
    float* d_targets = nullptr;
    CUDA_CHECK_ERROR(cudaMalloc(&d_inputs, x_train.size() * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_targets, y_train.size() * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_inputs, x_train.data(), x_train.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_targets, y_train.data(), y_train.size() * sizeof(float), cudaMemcpyHostToDevice));

    float* d_val_inputs = nullptr;
    float* d_val_targets = nullptr;
    CUDA_CHECK_ERROR(cudaMalloc(&d_val_inputs, x_val.size() * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_val_targets, y_val.size() * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_val_inputs, x_val.data(), x_val.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_val_targets, y_val.data(), y_val.size() * sizeof(float), cudaMemcpyHostToDevice));

    float learning_rate = 0.001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    std::unique_ptr<OptimizerBase> optimizer = create_optimizer("adam", learning_rate, beta1, beta2, epsilon);

    const char* loss_type = is_classification ? "cross_entropy" : "mse";

    cudaEventRecord(cuda_start);

     int steps_per_epoch = x_train.size() / (batch_size * input_size);

    for (int epoch = 0; epoch < num_epochs; ++epoch)
    {
        float epoch_loss = 0.0f;

        for (int batch = 0; batch < steps_per_epoch; ++batch)
        {
            int offset = batch * batch_size * input_size;
            float* batch_inputs = d_inputs + offset;
            float* batch_targets = d_targets + (batch * batch_size * output_size);

            nn.forward(batch_inputs,offset, true);

            float batch_loss = 0.0f;
            float* d_loss = nullptr;
            CUDA_CHECK_ERROR(cudaMalloc(&d_loss, sizeof(float)));
            CUDA_CHECK_ERROR(cudaMemset(d_loss, 0, sizeof(float)));

            float* d_element_loss = nullptr;
            size_t output_elements = batch_size * output_size;
            CUDA_CHECK_ERROR(cudaMalloc(&d_element_loss, output_elements * sizeof(float)));

            cudafunc::calculate_loss_values(
                nn.layers.back()->pre_activation_values,
                batch_targets,
                d_element_loss,
                output_elements,
                loss_type
            );

            std::vector<float> host_losses(output_elements);
            CUDA_CHECK_ERROR(cudaMemcpy(host_losses.data(), d_element_loss,
                           output_elements * sizeof(float),
                           cudaMemcpyDeviceToHost));

            batch_loss = 0.0f;
            for (const float& loss_val : host_losses)
            {
                batch_loss += loss_val;
            }
            batch_loss /= output_elements;

            CUDA_CHECK_ERROR(cudaFree(d_element_loss));
            CUDA_CHECK_ERROR(cudaFree(d_loss));

            nn.backward(batch_targets, 0.0f, loss_type);

            for (auto& layer : nn.layers)
            {
                layer->update_params(*optimizer);
            }

            epoch_loss += batch_loss;
        }

        epoch_loss /= steps_per_epoch;

        if (epoch % 10 == 0 || epoch == num_epochs - 1)
        {
            nn.forward(d_val_inputs,1, true);

            float* d_val_element_loss = nullptr;
            size_t val_output_elements = batch_size * output_size;
            CUDA_CHECK_ERROR(cudaMalloc(&d_val_element_loss, val_output_elements * sizeof(float)));

            cudafunc::calculate_loss_values(
                nn.layers.back()->pre_activation_values,
                d_val_targets,
                d_val_element_loss,
                val_output_elements,
                loss_type
            );

            std::vector<float> val_losses(val_output_elements);
            CUDA_CHECK_ERROR(cudaMemcpy(val_losses.data(), d_val_element_loss,
                           val_output_elements * sizeof(float),
                           cudaMemcpyDeviceToHost));

            float val_loss = 0.0f;
            for (const float& loss_val : val_losses)
            {
                val_loss += loss_val;
            }
            val_loss /= val_output_elements;

            CUDA_CHECK_ERROR(cudaFree(d_val_element_loss));

            std::vector<float> val_outputs(batch_size * output_size);
            CUDA_CHECK_ERROR(cudaMemcpy(val_outputs.data(), nn.layers.back()->output,
                           batch_size * output_size * sizeof(float),
                           cudaMemcpyDeviceToHost));

            float accuracy = 0.0f;
            if (is_classification)
            {
                accuracy = calculate_accuracy(val_outputs, y_val, output_size);
            }

            std::cout << "Epoch " << epoch + 1 << "/" << num_epochs
                      << ", Loss: " << epoch_loss
                      << ", Val Loss: " << val_loss;

            if (is_classification)
            {
                std::cout << ", Accuracy: " << accuracy * 100.0f << "%";
            }

            std::cout << std::endl;
        }
        else
        {
            std::cout << "Epoch " << epoch + 1 << "/" << num_epochs
                      << ", Loss: " << epoch_loss << std::endl;
        }
    }

    std::cout << "Training complete!\n";

    nn.forward(d_val_inputs,1, true);

    std::vector<float> final_logits(batch_size * output_size);
    CUDA_CHECK_ERROR(cudaMemcpy(final_logits.data(), nn.layers.back()->pre_activation_values, // <<< FIX: Get pre_activation_values (logits)
                           batch_size * output_size * sizeof(float),
                           cudaMemcpyDeviceToHost));

    std::vector<float> final_probabilities(batch_size * output_size);
    for (int b = 0; b < batch_size; ++b)
    {
        float max_logit = -std::numeric_limits<float>::infinity();
        int offset = b * output_size;
        for (int i = 0; i < output_size; ++i)
        {
            max_logit = std::max(max_logit, final_logits[offset + i]);
        }
        float sum_exp = 0.0f;
        for (int i = 0; i < output_size; ++i)
        {
            float exp_val = std::exp(final_logits[offset + i] - max_logit);
            final_probabilities[offset + i] = exp_val;
            sum_exp += exp_val;
        }
        float inv_sum = (sum_exp > 1e-9f) ? (1.0f / sum_exp) : 1.0f;
        for (int i = 0; i < output_size; ++i)
        {
            final_probabilities[offset + i] *= inv_sum;
        }
    }

    float final_custom_loss = calculate_custom_loss(final_probabilities, y_val);
    float final_accuracy = is_classification ? calculate_accuracy(final_probabilities, y_val, output_size) : 0.0f;

    std::cout << "\nFinal Results:\n";
    std::cout << "Custom Loss: " << final_custom_loss << "\n";
    if (is_classification)
    {
        std::cout << "Accuracy: " << final_accuracy * 100.0f << "%\n";
    }

    visualize_linear_output(final_probabilities, y_val, output_size);

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

    cudaEventDestroy(cuda_start);
    cudaEventDestroy(cuda_stop);

    CUDA_CHECK_ERROR(cudaFree(d_inputs));
    CUDA_CHECK_ERROR(cudaFree(d_targets));
    CUDA_CHECK_ERROR(cudaFree(d_val_inputs));
    CUDA_CHECK_ERROR(cudaFree(d_val_targets));

    for (auto& layer : nn.layers)
    {
        layer->clearBuffers();
    }
}
} // namespace linear