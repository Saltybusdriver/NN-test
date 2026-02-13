#include "cpu_DebugUtils.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <limits>
#include <stdexcept>
#include "utils.h"

namespace cpu_debug
{

std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> timers;

void findHostArrayRange(const float* h_array, int size, float& min_val, float& max_val)
{
    if (!h_array || size <= 0)
    {
        min_val = 0.0f;
        max_val = 0.0f;
        return;
    }

    min_val = std::numeric_limits<float>::max();
    max_val = std::numeric_limits<float>::lowest();
    bool valid_found = false;

    for (int i = 0; i < size; i++)
    {
        if (!std::isnan(h_array[i]) && !std::isinf(h_array[i]))
        {
            min_val = std::min(min_val, h_array[i]);
            max_val = std::max(max_val, h_array[i]);
            valid_found = true;
        }
    }

    if (!valid_found)
    {
        min_val = 0.0f;
        max_val = 0.0f;
    }
}

void calculateHostStats(const float* data, size_t size, float& min_val, float& max_val, float& mean_val, float& std_dev)
{
    if (!data || size == 0)
    {
        min_val = max_val = mean_val = std_dev = std::numeric_limits<float>::quiet_NaN();
        return;
    }
    min_val = std::numeric_limits<float>::quiet_NaN();
    max_val = std::numeric_limits<float>::quiet_NaN();
    double sum = 0.0;
    size_t valid_count = 0;
    bool first_valid = true;

    for (size_t i = 0; i < size; ++i)
    {
        float val = data[i];
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
    for (size_t i = 0; i < size; ++i)
    {
        float val = data[i];
         if (std::isnan(val) || std::isinf(val)) continue;
        sq_sum_diff += (val - mean_val) * (val - mean_val);
    }
      std_dev = (valid_count > 1) ? static_cast<float>(std::sqrt(sq_sum_diff / (valid_count - 1))) : 0.0f;
}

void inspectHostTensor(const char* name, const float* h_tensor, int size, bool force_print)
{
    {
        if (!h_tensor)
        {
             ERROR_PRINT("%s: Host tensor pointer is NULL\n", name);
             return;
        }
        float min_val = 0.0f, max_val = 0.0f;
        findHostArrayRange(h_tensor, size, min_val, max_val);
        DEBUG_PRINT("%s host tensor stats: min=%.6f, max=%.6f, size=%d\n",
                   name, min_val, max_val, size);
    }
}

void inspectHostGradients(const char* name, const float* gradient_data, int size, bool show_full)
{
    {
        if (!gradient_data)
        {
             ERROR_PRINT("%s: Host gradient pointer is NULL\n", name);
             return;
        }
        if (size == 0)
        {
             WARN_PRINT("%s: Host gradient size is 0\n", name);
             return;
        }

        float grad_min = std::numeric_limits<float>::max();
        float grad_max = std::numeric_limits<float>::lowest();
        float grad_abs_max = 0.0f;
        double grad_sum = 0.0;
        int zero_count = 0;
        size_t valid_count = 0;
        bool first_valid = true;

        for (int i = 0; i < size; i++)
        {
            float val = gradient_data[i];
            if (std::isnan(val) || std::isinf(val)) continue;

            if (first_valid)
            {
                grad_min = val;
                grad_max = val;
                first_valid = false;
            }
            else
            {
                grad_min = std::min(grad_min, val);
                grad_max = std::max(grad_max, val);
            }
            grad_sum += val;
            grad_abs_max = std::max(grad_abs_max, std::abs(val));
            if (std::abs(val) < 1e-10) zero_count++;
            valid_count++;
        }

        float grad_avg = (valid_count > 0) ? static_cast<float>(grad_sum / valid_count) : 0.0f;
        if (!first_valid)
        {
             INFO_PRINT("--- Host Gradients: %s ---\n", name);
             INFO_PRINT("  Size: %d (Valid: %zu), Min: %.6e, Max: %.6e, Abs Max: %.6e, Avg: %.6e\n",
                       size, valid_count, grad_min, grad_max, grad_abs_max, grad_avg);
             INFO_PRINT("  Zero/near-zero values: %d (%.1f%% of valid)\n", zero_count, (valid_count > 0) ? (100.0f * zero_count) / valid_count : 0.0f);
        }
        else
        {
             WARN_PRINT("--- Host Gradients: %s --- No valid values found in %d elements.\n", name, size);
        }

        if (show_full && valid_count > 0)
        {
            int display_count = std::min((int)valid_count, 20);
            DEBUG_COUT("  Sample values: ");
            int shown = 0;
            for (int i = 0; i < size && shown < display_count; i++)
            {
                float val = gradient_data[i];
                if (!std::isnan(val) && !std::isinf(val))
                {
                    DEBUG_PRINT("%.6e ", val);
                    shown++;
                    if (shown % 5 == 0 && shown < display_count) DEBUG_COUT("\n               ");
                }
            }
            if (valid_count > (size_t)display_count) DEBUG_COUT("...");
            DEBUG_COUT("\n");
        }
    }
}

void compareHostValues(const char* name, const float* before, const float* after, int size, bool show_full)
{
    {
        if (!before || !after)
        {
             ERROR_PRINT("%s: Host before/after pointer is NULL\n", name);
             return;
        }

        float max_diff = 0.0f;
        double sum_diff = 0.0;
        int changed_count = 0;
        size_t valid_pairs = 0;

        for (int i = 0; i < size; i++)
        {
            if (!std::isnan(before[i]) && !std::isinf(before[i]) &&
                !std::isnan(after[i]) && !std::isinf(after[i]))
            {
                float diff = std::abs(after[i] - before[i]);
                max_diff = std::max(max_diff, diff);
                sum_diff += diff;
                if (diff > 1e-10) changed_count++;
                valid_pairs++;
            }
        }
        float avg_diff = (valid_pairs > 0) ? static_cast<float>(sum_diff / valid_pairs) : 0.0f;

        DEBUG_PRINT("--- Host Parameter Update: %s ---\n", name);
        DEBUG_PRINT("  Max change: %.6e, Avg change: %.6e, Changed elements: %d/%zu (%.1f%% of valid pairs)\n",
                   max_diff, avg_diff, changed_count, valid_pairs, (valid_pairs > 0) ? (100.0f * changed_count) / valid_pairs : 0.0f);

        if (changed_count == 0 && valid_pairs > 0)
        {
            DEBUG_COUT("  WARNING: No values changed during update!\n");
        }
        else if (valid_pairs == 0 && size > 0)
        {
            WARN_PRINT("  WARNING: No valid pairs found to compare in %d elements.\n", size);
        }

        if (show_full && changed_count > 0)
        {
            int display_count = std::min(5, changed_count);
            DEBUG_PRINT("  Sample changes (first %d changed values):\n", display_count);
            int shown = 0;
            for (int i = 0; i < size && shown < display_count; i++)
            {
                 if (!std::isnan(before[i]) && !std::isinf(before[i]) &&
                     !std::isnan(after[i]) && !std::isinf(after[i]))
                 {
                    float diff = std::abs(after[i] - before[i]);
                    if (diff > 1e-10)
                    {
                        DEBUG_PRINT("    [%d] %.6e -> %.6e (diff: %.6e)\n",
                                  i, before[i], after[i], after[i] - before[i]);
                        shown++;
                    }
                 }
            }
        }
    }
}

void validateHostTensor(const float* tensor, size_t size, const char* name)
{
     if (!tensor)
     {
         ERROR_PRINT("WARNING: Invalid values check skipped for %s: Pointer is NULL\n", name);
         return;
     }
    float min_val, max_val;
    findHostArrayRange(tensor, size, min_val, max_val);
    if (std::isnan(min_val) || std::isnan(max_val) ||
        std::isinf(min_val) || std::isinf(max_val))
    {
        bool has_nan = false;
        bool has_inf = false;
        for(size_t i = 0; i < size; ++i)
        {
            if (std::isnan(tensor[i])) has_nan = true;
            if (std::isinf(tensor[i])) has_inf = true;
            if (has_nan && has_inf) break;
        }
        ERROR_PRINT("WARNING: Invalid values in host %s: HasNaN=%s, HasInf=%s (Reported range [%f, %f])\n",
                   name, has_nan ? "Yes" : "No", has_inf ? "Yes" : "No", min_val, max_val);
    }
}

void inspectHost4DTensor(const char* name, const float* h_tensor,
                         size_t dim1, size_t dim2, size_t dim3, size_t dim4,
                         bool print_sample, int sample_count)
{
    if (!h_tensor)
    {
        WARN_PRINT("CPU Inspect 4D '%s': Tensor pointer is NULL.\n", name);
        return;
    }

    size_t total_size = dim1 * dim2 * dim3 * dim4;
    if (total_size == 0)
    {
        INFO_PRINT("CPU Inspect 4D '%s': Tensor is empty (size 0).\n", name);
        return;
    }

    float min_val = std::numeric_limits<float>::quiet_NaN();
    float max_val = std::numeric_limits<float>::quiet_NaN();
    float mean_val = std::numeric_limits<float>::quiet_NaN();
    float std_dev = std::numeric_limits<float>::quiet_NaN();
    size_t nan_count = 0;
    size_t inf_count = 0;

    calculateHostStats(h_tensor, total_size, min_val, max_val, mean_val, std_dev);

    for (size_t i = 0; i < total_size; ++i)
    {
        if (std::isnan(h_tensor[i])) nan_count++;
        if (std::isinf(h_tensor[i])) inf_count++;
    }

    INFO_PRINT("CPU Inspect 4D '%s' (%zu x %zu x %zu x %zu = %zu elements):\n",
               name, dim1, dim2, dim3, dim4, total_size);
    INFO_PRINT("  Range: [%.6e, %.6e], Mean: %.6e, StdDev: %.6e\n",
               min_val, max_val, mean_val, std_dev);

    if (nan_count > 0 || inf_count > 0)
    {
        WARN_PRINT("  WARNING: Contains %zu NaN(s) and %zu Inf(s)!\n", nan_count, inf_count);
    }

    if (print_sample && total_size > 0)
    {
        int count = std::min((int)total_size, sample_count);
        DEBUG_PRINT("  Sample Data (first %d values):\n    [", count);
        for (int i = 0; i < count; ++i)
        {
            DEBUG_PRINT("%.6e%s", h_tensor[i], (i == count - 1) ? "" : ", ");
        }
        DEBUG_COUT("]\n");
    }
}

void inspectHostTensorFull(const char* name, const float* h_tensor, int size, int height, int width)
{
    {
        if (!h_tensor)
        {
             ERROR_PRINT("%s: Host tensor pointer is NULL for full inspection\n", name);
             return;
        }
        printTensor2D(name, h_tensor, height, width);
    }
}

void compareHostTensors(const char* name1, const float* h_tensor1,
                        const char* name2, const float* h_tensor2, int size)
{
    {
        if (!h_tensor1 || !h_tensor2)
        {
             ERROR_PRINT("Tensor comparison %s vs %s: One or both host pointers are NULL\n", name1, name2);
             return;
        }

        float max_diff = 0.0f;
        double sum_diff = 0.0;
        int diff_count = 0;
        size_t valid_pairs = 0;

        for (int i = 0; i < size; i++)
        {
             if (!std::isnan(h_tensor1[i]) && !std::isinf(h_tensor1[i]) &&
                 !std::isnan(h_tensor2[i]) && !std::isinf(h_tensor2[i]))
             {
                float diff = std::abs(h_tensor1[i] - h_tensor2[i]);
                max_diff = std::max(max_diff, diff);
                sum_diff += diff;
                if (diff > 1e-6) diff_count++;
                valid_pairs++;
             }
        }

        float avg_diff = (valid_pairs > 0) ? static_cast<float>(sum_diff / valid_pairs) : 0.0f;

        DEBUG_PRINT("Host Tensor comparison %s vs %s: max_diff=%.6e, avg_diff=%.6e, diff_count=%d/%zu (%.1f%% of valid pairs)\n",
                   name1, name2, max_diff, avg_diff, diff_count, valid_pairs, (valid_pairs > 0) ? (100.0f * diff_count) / valid_pairs : 0.0f);
        if (valid_pairs == 0 && size > 0)
        {
             WARN_PRINT("  WARNING: No valid pairs found to compare in %d elements.\n", size);
        }
    }
}

void checkForNaNHost(const char* name, const float* h_tensor, int size)
{
    {
        if (!h_tensor)
        {
             ERROR_PRINT("%s: Host tensor pointer is NULL for NaN check\n", name);
             return;
        }

        int nan_count = 0;
        int inf_count = 0;

        for (int i = 0; i < size; i++)
        {
            if (std::isnan(h_tensor[i])) nan_count++;
            if (std::isinf(h_tensor[i])) inf_count++;
        }

        if (nan_count > 0 || inf_count > 0)
        {
            ERROR_PRINT("%s host tensor contains %d NaN values and %d Inf values out of %d elements\n",
                       name, nan_count, inf_count, size);

            int shown = 0;
            for (int i = 0; i < size && shown < 10; i++)
            {
                if (std::isnan(h_tensor[i]) || std::isinf(h_tensor[i]))
                {
                    ERROR_PRINT("  Index %d: %f\n", i, h_tensor[i]);
                    shown++;
                }
            }
        }
    }
}

void inspectHostVector(const char* label, const float* data_h, size_t size, bool print_sample /*= true*/, int sample_count /*= 10*/)
{
    if (!data_h)
    {
        ERROR_PRINT("%s: Pointer is NULL\n", label);
        return;
    }
    if (size == 0)
    {
         WARN_PRINT("%s: Size is 0\n", label);
         return;
    }

    float min_val, max_val, mean_val, std_dev;
    calculateHostStats(data_h, size, min_val, max_val, mean_val, std_dev);

    INFO_PRINT("%s (Size: %zu): Min=%.6e, Max=%.6e, Mean=%.6e, StdDev=%.6e\n",
               label, size, min_val, max_val, mean_val, std_dev);

    int nan_count = 0;
    int inf_count = 0;
    for(size_t i = 0; i < size; ++i)
    {
        if (std::isnan(data_h[i])) nan_count++;
        if (std::isinf(data_h[i])) inf_count++;
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

void inspectHostMatrix(const char* label, const float* data_h, size_t rows, size_t cols, bool print_sample /*= true*/, int sample_rows /*= 5*/, int sample_cols /*= 5*/)
{
     size_t num_elements = rows * cols;
     if (!data_h)
     {
        ERROR_PRINT("%s: Pointer is NULL\n", label);
        return;
     }
     if (num_elements == 0)
     {
         WARN_PRINT("%s: Size is 0 (rows=%zu, cols=%zu)\n", label, rows, cols);
         return;
     }

     float min_val, max_val, mean_val, std_dev;
     calculateHostStats(data_h, num_elements, min_val, max_val, mean_val, std_dev);

     INFO_PRINT("%s (Shape: %zux%zu, Size: %zu): Min=%.6e, Max=%.6e, Mean=%.6e, StdDev=%.6e\n",
                label, rows, cols, num_elements, min_val, max_val, mean_val, std_dev);

     int nan_count = 0;
     int inf_count = 0;
     for(size_t i = 0; i < num_elements; ++i)
     {
         if (std::isnan(data_h[i])) nan_count++;
         if (std::isinf(data_h[i])) inf_count++;
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

void inspectHostBatchedVector(const char* label, const float* data_h, size_t batch_size, size_t vector_size, bool print_sample /*= true*/, int sample_batches /*= 3*/, int sample_elements /*= 5*/)
{
    size_t num_elements = batch_size * vector_size;
    if (!data_h)
    {
        ERROR_PRINT("%s: Pointer is NULL\n", label);
        return;
    }
    if (num_elements == 0)
    {
        WARN_PRINT("%s: Size is 0 (batch=%zu, vector=%zu)\n", label, batch_size, vector_size);
        return;
    }

    float min_val, max_val, mean_val, std_dev;
    calculateHostStats(data_h, num_elements, min_val, max_val, mean_val, std_dev);

    INFO_PRINT("%s (Shape: %zux%zu, Size: %zu): Min=%.6e, Max=%.6e, Mean=%.6e, StdDev=%.6e\n",
               label, batch_size, vector_size, num_elements, min_val, max_val, mean_val, std_dev);

    int nan_count = 0;
    int inf_count = 0;
    for(size_t i = 0; i < num_elements; ++i)
    {
        if (std::isnan(data_h[i])) nan_count++;
        if (std::isinf(data_h[i])) inf_count++;
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

void logActivationHost(const char* layer_type, const char* activation,
                       const float* input, const float* output, int size)
{
    {
        if (!input || !output)
        {
             ERROR_PRINT("%s %s activation: Input or output host pointer is NULL\n", layer_type, activation);
             return;
        }
        float in_min = 0.0f, in_max = 0.0f;
        float out_min = 0.0f, out_max = 0.0f;

        findHostArrayRange(input, size, in_min, in_max);
        findHostArrayRange(output, size, out_min, out_max);

        DEBUG_PRINT("%s %s activation (host): input range [%.6f, %.6f], output range [%.6f, %.6f]\n",
                   layer_type, activation, in_min, in_max, out_min, out_max);
    }
}

void logGradientsHost(const char* layer_type, const float* gradients, int size)
{
    {
        if (!gradients)
        {
             ERROR_PRINT("%s gradients (host): Pointer is NULL\n", layer_type);
             return;
        }
        float min_val = 0.0f, max_val = 0.0f;
        findHostArrayRange(gradients, size, min_val, max_val);

        int zero_count = 0;
        double sum = 0.0;
        double sum_abs = 0.0;
        size_t valid_count = 0;

        for (int i = 0; i < size; i++)
        {
            float val = gradients[i];
            if (std::isnan(val) || std::isinf(val)) continue;
            if (std::abs(val) < 1e-10) zero_count++;
            sum += val;
            sum_abs += std::abs(val);
            valid_count++;
        }

        float mean_val = (valid_count > 0) ? static_cast<float>(sum / valid_count) : 0.0f;
        float mean_abs_val = (valid_count > 0) ? static_cast<float>(sum_abs / valid_count) : 0.0f;

        DEBUG_PRINT("%s gradients (host): min=%.6g, max=%.6g, mean=%.6g, mean_abs=%.6g, zeros=%d/%zu (%.1f%% of valid)\n",
                   layer_type, min_val, max_val, mean_val, mean_abs_val,
                   zero_count, valid_count, (valid_count > 0) ? (100.0f * zero_count / valid_count) : 0.0f);
         if (valid_count != (size_t)size)
         {
             WARN_PRINT("  (%zu invalid values skipped)\n", size - valid_count);
         }
    }
}

void startTimer(const char* operation_name)
{
    {
        timers[operation_name] = std::chrono::high_resolution_clock::now();
    }
}

void stopTimer(const char* operation_name)
{
    {
        if (timers.find(operation_name) == timers.end())
        {
            WARN_PRINT("Timer '%s' stopped without being started.\n", operation_name);
            return;
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto start_time = timers[operation_name];

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        float ms = duration / 1000.0f;

        DEBUG_PRINT("%s: %.3f ms\n", operation_name, ms);
        timers.erase(operation_name);
    }
}

double getElapsedTime(const std::string& timer_name)
{
     if (timers.find(timer_name) == timers.end())
     {
         WARN_PRINT("Timer '%s' not found for getting elapsed time.\n", timer_name.c_str());
         return 0.0;
     }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto start_time = timers[timer_name];

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    return duration / 1000.0;
}

void logLayerPerformance(const char* layer_type, float forward_ms, float backward_ms)
{
    {
        INFO_PRINT("%s performance: forward=%.3f ms, backward=%.3f ms, total=%.3f ms\n",
                  layer_type, forward_ms, backward_ms, forward_ms + backward_ms);
    }
}

void logWeightUpdateHost(const char* layer_name, const float* weights, int size, float learning_rate)
{
    {
        if (!weights)
        {
             ERROR_PRINT("%s weights updated (host): Pointer is NULL\n", layer_name);
             return;
        }
        float min_val = 0.0f, max_val = 0.0f;
        findHostArrayRange(weights, size, min_val, max_val);

        DEBUG_PRINT("%s weights updated (lr=%.6f, host): min=%.6f, max=%.6f\n",
                   layer_name, learning_rate, min_val, max_val);
    }
}

void horizontalLine(char symbol, int length)
{
    {
        for (int i = 0; i < length; i++)
        {
            std::cerr << symbol;
        }
        std::cerr << std::endl;
    }
}

void sectionHeader(const char* title)
{
    {
        horizontalLine();
        std::cerr << "== " << title << " ==" << std::endl;
        horizontalLine();
    }
}

} // namespace cpu_debug