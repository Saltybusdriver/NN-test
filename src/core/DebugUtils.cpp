#include "DebugUtils.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include "utils.h"
#include <chrono>
// Initialize static members
std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> DebugUtils::timers;

// ====== Tensor Inspection Methods ======

void DebugUtils::inspectTensor(const char* name, float* d_tensor, int size, bool force_print)
{
    if (force_print || current_debug_level >= LEVEL_DEBUG)
    {
        float min_val = 0.0f, max_val = 0.0f;
        find_array_range(d_tensor, size, min_val, max_val);
        DEBUG_PRINT("%s tensor stats: min=%.6f, max=%.6f, size=%d\n",
                   name, min_val, max_val, size);
    }
}
void DebugUtils::inspectGradients(const char* name, float* gradient_data, int size, bool show_full)
{
    if (current_debug_level >= LEVEL_INFO)
    {
        std::vector<float> host_grad(size);
        cudaMemcpy(host_grad.data(), gradient_data, size * sizeof(float), cudaMemcpyDeviceToHost);

        float grad_min = *std::min_element(host_grad.begin(), host_grad.end());
        float grad_max = *std::max_element(host_grad.begin(), host_grad.end());
        float grad_abs_max = 0.0f;
        float grad_avg = 0.0f;
        int zero_count = 0;

        for (int i = 0; i < size; i++)
        {
            float val = host_grad[i];
            grad_avg += val;
            grad_abs_max = std::max(grad_abs_max, std::abs(val));
            if (std::abs(val) < 1e-10) zero_count++;
        }
        grad_avg /= size;

        INFO_PRINT("--- Gradients: %s ---\n", name);
        INFO_PRINT("  Size: %d, Min: %.6e, Max: %.6e, Abs Max: %.6e, Avg: %.6e\n",
                  size, grad_min, grad_max, grad_abs_max, grad_avg);
        INFO_PRINT("  Zero/near-zero values: %d (%.1f%%)\n", zero_count, (100.0f * zero_count) / size);

        if (show_full && current_debug_level >= LEVEL_DEBUG)
        {
            int display_count = std::min(size, 20);
            DEBUG_COUT("  Sample values: ");
            for (int i = 0; i < display_count; i++)
            {
                DEBUG_PRINT("%.6e ", host_grad[i]);
                if ((i+1) % 5 == 0) DEBUG_COUT("\n               ");
            }
            if (size > display_count) DEBUG_COUT("...");
            DEBUG_COUT("\n");
        }
    }
}

void DebugUtils::compareValues(const char* name, float* before, float* after, int size, bool show_full)
{
    if (current_debug_level >= LEVEL_DEBUG)
    {
        std::vector<float> host_before(size);
        std::vector<float> host_after(size);

        cudaMemcpy(host_before.data(), before, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_after.data(), after, size * sizeof(float), cudaMemcpyDeviceToHost);

        float max_diff = 0.0f;
        float avg_diff = 0.0f;
        int changed_count = 0;

        for (int i = 0; i < size; i++)
        {
            float diff = std::abs(host_after[i] - host_before[i]);
            max_diff = std::max(max_diff, diff);
            avg_diff += diff;
            if (diff > 1e-10) changed_count++;
        }
        avg_diff /= size;

        DEBUG_PRINT("--- Parameter Update: %s ---\n", name);
        DEBUG_PRINT("  Max change: %.6e, Avg change: %.6e, Changed elements: %d/%d (%.1f%%)\n",
                   max_diff, avg_diff, changed_count, size, (100.0f * changed_count) / size);

        if (changed_count == 0)
        {
            DEBUG_COUT("  WARNING: No values changed during update!\n");
        }

        if (show_full && changed_count > 0)
        {
            int display_count = std::min(5, changed_count);
            DEBUG_PRINT("  Sample changes (first %d changed values):\n", display_count);
            int shown = 0;
            for (int i = 0; i < size && shown < display_count; i++)
            {
                float diff = std::abs(host_after[i] - host_before[i]);
                if (diff > 1e-10)
                {
                    DEBUG_PRINT("    [%d] %.6e -> %.6e (diff: %.6e)\n",
                              i, host_before[i], host_after[i], host_after[i] - host_before[i]);
                    shown++;
                }
            }
        }
    }
}
void DebugUtils::ValidateTensor(float* tensor, size_t size, const char* name)
{
    float min_val, max_val;
    find_array_range(tensor, size, min_val, max_val);
    if (std::isnan(min_val) || std::isnan(max_val) ||
        std::isinf(min_val) || std::isinf(max_val))
    {
        ERROR_PRINT("WARNING: Invalid values in %s: range [%f, %f]\n",
                   name, min_val, max_val);
    }
}

void DebugUtils::inspectTensorFull(const char* name, float* d_tensor, int size, int height, int width)
{
    if (current_debug_level >= LEVEL_TRACE)
    {
        std::vector<float> host_data(size);
        ToHost(host_data, d_tensor);
        printTensor2D(name, host_data, height, width);
    }
}

void DebugUtils::compareTensors(const char* name1, float* d_tensor1,
                               const char* name2, float* d_tensor2, int size)
{
    if (current_debug_level >= LEVEL_DEBUG)
    {
        std::vector<float> host_data1(size);
        std::vector<float> host_data2(size);

        cudaMemcpy(host_data1.data(), d_tensor1, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_data2.data(), d_tensor2, size * sizeof(float), cudaMemcpyDeviceToHost);

        float max_diff = 0.0f;
        float sum_diff = 0.0f;
        int diff_count = 0;

        for (int i = 0; i < size; i++)
        {
            float diff = std::abs(host_data1[i] - host_data2[i]);
            max_diff = std::max(max_diff, diff);
            sum_diff += diff;
            if (diff > 1e-6) diff_count++;
        }

        DEBUG_PRINT("Tensor comparison %s vs %s: max_diff=%.6f, avg_diff=%.6f, diff_count=%d/%d\n",
                   name1, name2, max_diff, sum_diff/size, diff_count, size);
    }
}

void DebugUtils::checkForNaN(const char* name, float* d_tensor, int size)
{
    if (current_debug_level >= LEVEL_ERROR)
    {
        std::vector<float> host_data(size);
        cudaMemcpy(host_data.data(), d_tensor, size * sizeof(float), cudaMemcpyDeviceToHost);

        int nan_count = 0;
        int inf_count = 0;

        for (int i = 0; i < size; i++)
        {
            if (std::isnan(host_data[i])) nan_count++;
            if (std::isinf(host_data[i])) inf_count++;
        }

        if (nan_count > 0 || inf_count > 0)
        {
            ERROR_PRINT("%s tensor contains %d NaN values and %d Inf values out of %d elements\n",
                       name, nan_count, inf_count, size);

            int shown = 0;
            for (int i = 0; i < size && shown < 10; i++)
            {
                if (std::isnan(host_data[i]) || std::isinf(host_data[i]))
                {
                    ERROR_PRINT("  Index %d: %f\n", i, host_data[i]);
                    shown++;
                }
            }
        }
    }
}

void DebugUtils::printTensorShape(const char* name, int batch_size, int channels, int height, int width)
{
    if (current_debug_level >= LEVEL_DEBUG)
    {
        DEBUG_PRINT("%s shape: [%d, %d, %d, %d]\n", name, batch_size, channels, height, width);
    }
}

void DebugUtils::printTensorShape(const char* name, int batch_size, int features)
{
    if (current_debug_level >= LEVEL_DEBUG)
    {
        DEBUG_PRINT("%s shape: [%d, %d]\n", name, batch_size, features);
    }
}

// ====== Layer Operations ======

void DebugUtils::logLayerForward(const char* layer_type, int batch_size, int channels,
                                int height, int width)
{
    if (current_debug_level >= LEVEL_INFO)
    {
        INFO_PRINT("%s forward: [%d, %d, %d, %d]\n",
                   layer_type, batch_size, channels, height, width);
    }
}

void DebugUtils::logActivation(const char* layer_type, const char* activation,
                              float* input, float* output, int size)
{
    if (current_debug_level >= LEVEL_DEBUG)
    {
        float in_min = 0.0f, in_max = 0.0f;
        float out_min = 0.0f, out_max = 0.0f;

        findArrayRange(input, size, in_min, in_max);
        findArrayRange(output, size, out_min, out_max);

        DEBUG_PRINT("%s %s activation: input range [%.6f, %.6f], output range [%.6f, %.6f]\n",
                   layer_type, activation, in_min, in_max, out_min, out_max);
    }
}

void DebugUtils::logLayerBackward(const char* layer_type, float learning_rate)
{
    if (current_debug_level >= LEVEL_INFO)
    {
        INFO_PRINT("%s backward: learning_rate=%.6f\n", layer_type, learning_rate);
    }
}

void DebugUtils::logGradients(const char* layer_type, float* gradients, int size)
{
    if (current_debug_level >= LEVEL_DEBUG)
    {
        float min_val = 0.0f, max_val = 0.0f;
        findArrayRange(gradients, size, min_val, max_val);

        std::vector<float> host_grads(size);
        cudaMemcpy(host_grads.data(), gradients, size * sizeof(float), cudaMemcpyDeviceToHost);

        int zero_count = 0;
        float sum = 0.0f;
        float sum_abs = 0.0f;

        for (int i = 0; i < size; i++)
        {
            if (host_grads[i] == 0.0f) zero_count++;
            sum += host_grads[i];
            sum_abs += std::abs(host_grads[i]);
        }

        DEBUG_PRINT("%s gradients: min=%.6g, max=%.6g, mean=%.6g, mean_abs=%.6g, zeros=%d/%d (%.1f%%)\n",
                   layer_type, min_val, max_val, sum/size, sum_abs/size,
                   zero_count, size, (100.0f * zero_count / size));
    }
}

// ====== Layer-Specific Debug ======

void DebugUtils::VerifyLayerAligment(int total_output_size, int next_input_size,int layer_num)
{
    if (current_debug_level >= LEVEL_ERROR)
    {
        if (total_output_size != next_input_size)
        {
            throw std::runtime_error(
                "Dimension mismatch: Layer " + std::to_string(layer_num) +
                " output(" + std::to_string(total_output_size) +
                ") != Layer " + std::to_string(layer_num+1) +
                " input(" + std::to_string(next_input_size) + ")"
            );
        }
    }
}

void DebugUtils::logConv2DOperation(int batch_size, int in_channels, int out_channels,
                                   int input_height, int input_width,
                                   int kernel_size, int stride, int padding)
{
    if (current_debug_level >= LEVEL_DEBUG)
    {
        int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
        int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

        DEBUG_PRINT("Conv2D: in=[%d,%d,%d,%d], out=[%d,%d,%d,%d], k=%d, s=%d, p=%d\n",
                   batch_size, in_channels, input_height, input_width,
                   batch_size, out_channels, output_height, output_width,
                   kernel_size, stride, padding);
    }
}

void DebugUtils::debugBottleneckLayer(int in_channels, int out_channels,
                                     int input_height, int input_width)
{
    if (current_debug_level >= LEVEL_DEBUG)
    {
        DEBUG_PRINT("Bottleneck layer: %d -> %d channels, spatial: %dx%d\n",
                   in_channels, out_channels, input_height, input_width);
    }
}

void DebugUtils::debugBottleneckBackward(float* output_error, float* prev_output_error,
                                        int batch_size, int in_channels, int out_channels,
                                        int input_height, int input_width,
                                        int output_height, int output_width)
{
    if (current_debug_level >= LEVEL_DEBUG)
    {
        size_t output_size = batch_size * out_channels * output_height * output_width;
        size_t input_size = batch_size * in_channels * input_height * input_width;

        float out_err_min = 0.0f, out_err_max = 0.0f;
        float prev_err_min = 0.0f, prev_err_max = 0.0f;

        findArrayRange(output_error, output_size, out_err_min, out_err_max);

        DEBUG_COUT("Bottleneck backward:\n");
        DEBUG_PRINT("  Output error [%d,%d,%d,%d]: min=%.6g, max=%.6g\n",
                   batch_size, out_channels, output_height, output_width,
                   out_err_min, out_err_max);

        if (prev_output_error != nullptr)
        {
            findArrayRange(prev_output_error, input_size, prev_err_min, prev_err_max);
            DEBUG_PRINT("  Prev error [%d,%d,%d,%d]: min=%.6g, max=%.6g\n",
                      batch_size, in_channels, input_height, input_width,
                      prev_err_min, prev_err_max);
        }
    }
}

void DebugUtils::logFCOperation(int batch_size, int in_features, int out_features)
{
    if (current_debug_level >= LEVEL_INFO)
    {
        INFO_PRINT("FullyConnected: in=[%d,%d], out=[%d,%d]\n",
                   batch_size, in_features, batch_size, out_features);
    }
}

// ====== Memory Management ======

void DebugUtils::logMemoryAllocation(const char* tensor_name, size_t size_bytes)
{
    if (current_debug_level >= LEVEL_DEBUG)
    {
        float size_mb = size_bytes / (1024.0f * 1024.0f);
        DEBUG_PRINT("Allocated %s: %.2f MB (%.0f bytes)\n",
                   tensor_name, size_mb, (float)size_bytes);
    }
}

void DebugUtils::logMemoryFree(const char* tensor_name)
{
    if (current_debug_level >= LEVEL_DEBUG)
    {
        DEBUG_PRINT("Freed %s\n", tensor_name);
    }
}

void DebugUtils::logCudaMemoryInfo()
{
    if (current_debug_level >= LEVEL_INFO)
    {
        size_t free_bytes, total_bytes;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        float free_mb = free_bytes / (1024.0f * 1024.0f);
        float total_mb = total_bytes / (1024.0f * 1024.0f);
        float used_mb = (total_bytes - free_bytes) / (1024.0f * 1024.0f);
        float percent_used = 100.0f * used_mb / total_mb;

        INFO_PRINT("CUDA Memory: %.1f/%.1f MB used (%.1f%%), %.1f MB free\n",
                  used_mb, total_mb, percent_used, free_mb);
    }
}

// ====== Performance Monitoring ======

void DebugUtils::startTimer(const char* operation_name)
{
    if (current_debug_level >= LEVEL_DEBUG)
    {
        timers[operation_name] = std::chrono::high_resolution_clock::now();
    }
}

void DebugUtils::stopTimer(const char* operation_name)
{
    if (current_debug_level >= LEVEL_DEBUG)
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto start_time = timers[operation_name];

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        float ms = duration / 1000.0f;

        DEBUG_PRINT("%s: %.3f ms\n", operation_name, ms);
    }
}

void DebugUtils::logLayerPerformance(const char* layer_type, float forward_ms, float backward_ms)
{
    if (current_debug_level >= LEVEL_INFO)
    {
        INFO_PRINT("%s performance: forward=%.3f ms, backward=%.3f ms, total=%.3f ms\n",
                  layer_type, forward_ms, backward_ms, forward_ms + backward_ms);
    }
}

// ====== Training Progress ======

void DebugUtils::logEpochStart(int epoch, int total_epochs)
{
    if (current_debug_level >= LEVEL_INFO)
    {
        horizontalLine();
        INFO_PRINT("Starting Epoch %d/%d\n", epoch, total_epochs);
    }
}

void DebugUtils::logEpochEnd(int epoch, float loss, float accuracy)
{
    if (current_debug_level >= LEVEL_INFO)
    {
        INFO_PRINT("Epoch %d complete: loss=%.6f, accuracy=%.2f%%\n",
                  epoch, loss, accuracy * 100.0f);
        horizontalLine();
    }
}

void DebugUtils::logBatchProgress(int batch, int total_batches, float batch_loss)
{
    if (current_debug_level >= LEVEL_INFO)
    {
        if (batch % std::max(1, total_batches / 10) == 0 || batch == total_batches - 1)
        {
            INFO_PRINT("Batch %d/%d: loss=%.6f\n", batch + 1, total_batches, batch_loss);
        }
    }
}

void DebugUtils::logWeightUpdate(const char* layer_name, float* weights, int size, float learning_rate)
{
    if (current_debug_level >= LEVEL_DEBUG)
    {
        float min_val = 0.0f, max_val = 0.0f;
        findArrayRange(weights, size, min_val, max_val);

        DEBUG_PRINT("%s weights updated (lr=%.6f): min=%.6f, max=%.6f\n",
                   layer_name, learning_rate, min_val, max_val);
    }
}

// ====== Error Handling ======

void DebugUtils::logCudaError(cudaError_t error, const char* operation)
{
    if (error != cudaSuccess)
    {
        ERROR_PRINT("CUDA error in %s: %s\n", operation, cudaGetErrorString(error));
    }
}

void DebugUtils::logTrainingError(const char* message)
{
    ERROR_PRINT("Training error: %s\n", message);
}

// ====== Helper Methods ======

void DebugUtils::horizontalLine(char symbol, int length)
{
    if (current_debug_level > LEVEL_TRACE)
    {
        for (int i = 0; i < length; i++)
        {
            fprintf(stderr, "%c", symbol);
        }
        fprintf(stderr, "\n");
    }
}

void DebugUtils::sectionHeader(const char* title)
{
    if (current_debug_level > LEVEL_NONE)
    {
        horizontalLine();
        fprintf(stderr, "== %s ==\n", title);
        horizontalLine();
    }
}

// ====== Private Helper Methods ======

void DebugUtils::findArrayRange(float* d_array, int size, float& min_val, float& max_val)
{
    std::vector<float> host_array(size);
    cudaMemcpy(host_array.data(), d_array, size * sizeof(float), cudaMemcpyDeviceToHost);

    min_val = std::numeric_limits<float>::max();
    max_val = std::numeric_limits<float>::lowest();

    for (int i = 0; i < size; i++)
    {
        if (!std::isnan(host_array[i]) && !std::isinf(host_array[i]))
        {
            min_val = std::min(min_val, host_array[i]);
            max_val = std::max(max_val, host_array[i]);
        }
    }

    if (min_val == std::numeric_limits<float>::max())
    {
        min_val = 0.0f;
        max_val = 0.0f;
    }
}

double DebugUtils::getElapsedTime(const std::string& timer_name)
{
    auto end_time = std::chrono::high_resolution_clock::now();
    auto start_time = timers[timer_name];

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    return duration / 1000.0;
}

void DebugUtils::calculateStats(const std::vector<float>& data, float& min_val, float& max_val, float& mean_val, float& std_dev)
{
    if (data.empty())
    {
        min_val = max_val = mean_val = std_dev = std::numeric_limits<float>::quiet_NaN();
        return;
    }
    min_val = std::numeric_limits<float>::quiet_NaN();
    max_val = std::numeric_limits<float>::quiet_NaN();
    double sum = 0.0;
    size_t valid_count = 0;
    bool first_valid = true;

    for (float val : data)
    {
        if (std::isnan(val) || std::isinf(val)) continue;

        if (first_valid)
        {
            min_val = val;
            max_val = val;
            first_valid = false;
        }
        else
        {
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }
        sum += val;
        valid_count++;
    }

    if (valid_count == 0)
    {
         mean_val = std_dev = std::numeric_limits<float>::quiet_NaN();
         return;
    }

    mean_val = static_cast<float>(sum / valid_count);

    double sq_sum_diff = 0.0;
    for (float val : data)
    {
         if (std::isnan(val) || std::isinf(val)) continue;
        sq_sum_diff += (val - mean_val) * (val - mean_val);
    }
    std_dev = (valid_count > 1) ? static_cast<float>(std::sqrt(sq_sum_diff / (valid_count - 1))) : 0.0f;
}

// --- Public Static Method Implementations ---

void DebugUtils::inspectVector(const char* label, float* data_d, size_t size, bool print_sample /*= true*/, int sample_count /*= 10*/)
{
    if (!data_d)
    {
        ERROR_PRINT("%s: Pointer is NULL\n", label);
        return;
    }
    if (size == 0)
    {
         WARN_PRINT("%s: Size is 0\n", label);
         return;
    }

    std::vector<float> data_h(size);
    cudaError_t err = cudaMemcpy(data_h.data(), data_d, size * sizeof(float), cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        ERROR_PRINT("%s: Failed to copy data from device: %s\n", label, cudaGetErrorString(err));
        return;
    }

    float min_val, max_val, mean_val, std_dev;
    calculateStats(data_h, min_val, max_val, mean_val, std_dev);

    INFO_PRINT("%s (Size: %zu): Min=%.6e, Max=%.6e, Mean=%.6e, StdDev=%.6e\n",
               label, size, min_val, max_val, mean_val, std_dev);

    int nan_count = 0;
    int inf_count = 0;
    for(float val : data_h)
    {
        if (std::isnan(val)) nan_count++;
        if (std::isinf(val)) inf_count++;
    }
    if (nan_count > 0 || inf_count > 0)
    {
         WARN_PRINT("  WARNING: Contains %d NaNs, %d Infs\n", nan_count, inf_count);
    }

    if (print_sample && size > 0)
    {
        std::cout << "  Sample values: [";
        int count = std::min((int)size, sample_count);
        for (int i = 0; i < count; ++i)
        {
            printf("%.4e%s", data_h[i], (i == count - 1) ? "" : ", ");
        }
        if (size > (size_t)count) std::cout << " ...";
        std::cout << "]" << std::endl;
    }
}

void DebugUtils::inspectMatrix(const char* label, float* data_d, size_t rows, size_t cols, bool print_sample /*= true*/, int sample_rows /*= 5*/, int sample_cols /*= 5*/)
{
     size_t num_elements = rows * cols;
     if (!data_d)
     {
        ERROR_PRINT("%s: Pointer is NULL\n", label);
        return;
     }
     if (num_elements == 0)
     {
         WARN_PRINT("%s: Size is 0 (rows=%zu, cols=%zu)\n", label, rows, cols);
         return;
     }

     std::vector<float> data_h(num_elements);
     cudaError_t err = cudaMemcpy(data_h.data(), data_d, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

     if (err != cudaSuccess)
     {
        ERROR_PRINT("%s: Failed to copy data from device: %s\n", label, cudaGetErrorString(err));
        return;
     }

     float min_val, max_val, mean_val, std_dev;
     calculateStats(data_h, min_val, max_val, mean_val, std_dev);

     INFO_PRINT("%s (Shape: %zux%zu, Size: %zu): Min=%.6e, Max=%.6e, Mean=%.6e, StdDev=%.6e\n",
                label, rows, cols, num_elements, min_val, max_val, mean_val, std_dev);

     int nan_count = 0;
     int inf_count = 0;
     for(float val : data_h)
     {
         if (std::isnan(val)) nan_count++;
         if (std::isinf(val)) inf_count++;
     }
     if (nan_count > 0 || inf_count > 0)
     {
          WARN_PRINT("  WARNING: Contains %d NaNs, %d Infs\n", nan_count, inf_count);
     }

     if (print_sample && rows > 0 && cols > 0)
     {
         int r_count = std::min((int)rows, sample_rows);
         int c_count = std::min((int)cols, sample_cols);
         std::cout << "  Sample matrix (" << r_count << "x" << c_count << " from " << rows << "x" << cols << "):" << std::endl;
         for (int r = 0; r < r_count; ++r)
         {
             std::cout << "    [";
             for (int c = 0; c < c_count; ++c)
             {
                 printf("%.3e%s", data_h[r * cols + c], (c == c_count - 1) ? "" : ", ");
             }
              if (cols > (size_t)c_count) std::cout << " ...";
             std::cout << "]" << std::endl;
         }
          if (rows > (size_t)r_count) std::cout << "    ..." << std::endl;
     }
}

void DebugUtils::inspectBatchedVector(const char* label, float* data_d, size_t batch_size, size_t vector_size, bool print_sample /*= true*/, int sample_batches /*= 3*/, int sample_elements /*= 5*/)
{
    size_t num_elements = batch_size * vector_size;
    if (!data_d)
    {
        ERROR_PRINT("%s: Pointer is NULL\n", label);
        return;
    }
    if (num_elements == 0)
    {
        WARN_PRINT("%s: Size is 0 (batch=%zu, vector=%zu)\n", label, batch_size, vector_size);
        return;
    }

    std::vector<float> data_h(num_elements);
    cudaError_t err = cudaMemcpy(data_h.data(), data_d, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        ERROR_PRINT("%s: Failed to copy data from device: %s\n", label, cudaGetErrorString(err));
        return;
    }

    float min_val, max_val, mean_val, std_dev;
    calculateStats(data_h, min_val, max_val, mean_val, std_dev);

    INFO_PRINT("%s (Shape: %zux%zu, Size: %zu): Min=%.6e, Max=%.6e, Mean=%.6e, StdDev=%.6e\n",
               label, batch_size, vector_size, num_elements, min_val, max_val, mean_val, std_dev);

    int nan_count = 0;
    int inf_count = 0;
    for(float val : data_h)
    {
        if (std::isnan(val)) nan_count++;
        if (std::isinf(val)) inf_count++;
    }
    if (nan_count > 0 || inf_count > 0)
    {
         WARN_PRINT("  WARNING: Contains %d NaNs, %d Infs\n", nan_count, inf_count);
    }

    if (print_sample && batch_size > 0 && vector_size > 0)
    {
        int b_count = std::min((int)batch_size, sample_batches);
        int e_count = std::min((int)vector_size, sample_elements);
        std::cout << "  Sample batches (" << b_count << " from " << batch_size << "):" << std::endl;
        for (int b = 0; b < b_count; ++b)
        {
            printf("    Batch %d: [", b);
            for (int i = 0; i < e_count; ++i)
            {
                 printf("%.4e%s", data_h[b * vector_size + i], (i == e_count - 1) ? "" : ", ");
            }
            if (vector_size > (size_t)e_count) std::cout << " ...";
            std::cout << "]" << std::endl;
        }
        if (batch_size > (size_t)b_count) std::cout << "    ..." << std::endl;
    }
}