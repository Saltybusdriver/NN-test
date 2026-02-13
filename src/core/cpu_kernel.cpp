// src/cpu_kernels.cpp
#include "cpu_functions.h"
#include "autodiff.h" 
#include "debug.h"     
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstring> 
#include <stdexcept> 
#include <limits>
#include <thread>  
#include <functional>
#include <immintrin.h>
#include "thread_pool.h"
namespace autodiff { namespace loss 
{
    __attribute__((weak)) auto CustomLoss::expression() 
    {
        Output o;
        Target t;
        // Default is MSE
        return square(o - t);
    }
}}


namespace {
    ThreadPool g_thread_pool;
void parallel_for(int start, int end, const std::function<void(int)>& func) 
{
    int range = end - start;
    if (range <= 0) return;

    size_t num_workers = g_thread_pool.size();
    int min_chunk_size = 1;
    int desired_chunk_size = (range + num_workers - 1) / num_workers;
    int chunk_size = std::max(min_chunk_size, desired_chunk_size);
    size_t num_tasks = (range + chunk_size - 1) / chunk_size;

     std::vector<std::future<void>> futures;
     futures.reserve(num_tasks);

     for (size_t t = 0; t < num_tasks; ++t) 
     {
        long long current_chunk_start_ll = static_cast<long long>(start) + static_cast<long long>(t) * chunk_size;
        long long next_chunk_start_ll    = static_cast<long long>(start) + static_cast<long long>(t + 1) * chunk_size;

        long long actual_chunk_end_ll = std::min(next_chunk_start_ll, static_cast<long long>(end));

        int chunk_start = static_cast<int>(std::max(static_cast<long long>(start), current_chunk_start_ll));
        int chunk_end   = static_cast<int>(actual_chunk_end_ll);

         if (chunk_start < chunk_end) 
         {
             futures.emplace_back(
                 g_thread_pool.enqueue([=, &func]() 
                 {
                     for (int i = chunk_start; i < chunk_end; ++i) 
                     {
                         func(i);
                     }
                 })
             );
         }
     }
     for (auto& fut : futures) 
     {
         fut.get();
     }
}


void parallel_for_2d(int start1, int end1, int start2, int end2,
                     const std::function<void(int, int)>& func) {
    long long range1 = static_cast<long long>(end1) - start1;
    long long range2 = static_cast<long long>(end2) - start2;
    if (range1 <= 0 || range2 <= 0) return;

    long long total_range = range1 * range2;
    if (total_range <= 0) return;

    size_t num_workers = g_thread_pool.size();
    long long min_chunk_size_ll = 1;
    long long desired_chunk_size_ll = (total_range + num_workers - 1) / num_workers;
    long long chunk_size_ll = std::max(min_chunk_size_ll, desired_chunk_size_ll);
    size_t num_tasks = (total_range + chunk_size_ll - 1) / chunk_size_ll;

    std::vector<std::future<void>> futures;
    futures.reserve(num_tasks);

    for (size_t t = 0; t < num_tasks; ++t) 
    {
        long long chunk_start_flat = t * chunk_size_ll;
        long long chunk_end_flat = std::min(static_cast<long long>(t + 1) * chunk_size_ll, total_range);

        if (chunk_start_flat < chunk_end_flat) 
        {
            futures.emplace_back(
                g_thread_pool.enqueue([=, &func]() 
                {
                    for (long long flat_idx = chunk_start_flat; flat_idx < chunk_end_flat; ++flat_idx) {
                        int idx1 = start1 + static_cast<int>(flat_idx / range2);
                        int idx2 = start2 + static_cast<int>(flat_idx % range2);
                        func(idx1, idx2);
                    }
                })
            );
        }
    }

    for (auto& fut : futures) 
    {
        fut.get();
    }
}
}


namespace cpu_func {

// Activation Functions (Direct translation from __device__ functions) - Kept as is
inline float Sigmoid(float x) 
{
    return 1.0f / (1.0f + std::exp(-x));
}

inline float ReLU(float x) 
{
    return std::max(0.0f, x);
}

inline float LeakyReLU(float x, float negative_slope) 
{
    return std::max(x * negative_slope, x);
}

inline float SigmoidDerivative(float x) 
{
    float sigmoid = Sigmoid(x);
    return sigmoid * (1.0f - sigmoid);
}

inline float ReLUDerivative(float x) 
{
    return (x > 0.0f) ? 1.0f : 0.0f;
}

inline float LeakyReLUDerivative(float x, float negative_slope) 
{
    return (x > 0.0f) ? 1.0f : negative_slope;
}

inline float activationFunction(float input, int type) 
{
    switch (type) 
    {
        case 0: return Sigmoid(input);        
        case 1: return ReLU(input);          
        case 2: return LeakyReLU(input, 0.01f);
        // case 3: // SOFTMAX handled separately
        default: return input;
    }
}

inline float activationFunctionDerivative(float input, int type) 
{
    switch (type) 
    {
        case 0: return SigmoidDerivative(input);        
        case 1: return ReLUDerivative(input);           
        case 2: return LeakyReLUDerivative(input, 0.01f); 
        // case 3: // SOFTMAX handled separately
        default: return 1.0f;                       
    }
}

// --- Activations ---
void apply_activation_inplace(float* data, int size, int activation_type) {
    if (activation_type == 3) 
    { 
        float max_val = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < size; ++i) 
        {
            max_val = std::max(max_val, data[i]);
        }
        float sum_exp = 0.0f;
        for (int i = 0; i < size; ++i) 
        {
            data[i] = std::exp(data[i] - max_val);
            sum_exp += data[i];
        }
        float inv_sum = 1.0f / (sum_exp + 1e-9f);
        for (int i = 0; i < size; ++i) 
        {
            data[i] *= inv_sum;
        }
    } 
    else 
    {
        parallel_for(0, size, [&](int i) 
        {
            data[i] = activationFunction(data[i], activation_type);
        });
    }
}


// --- Linear Layer ---
void add_vectors(float* out, const float* bias, int size) 
{
    parallel_for(0, size, [&](int i) 
    {
        out[i] += bias[i];
    });
}

// Corresponds to the matrix multiplication part of cuda_forward_propagation and forward_Propagate
void forward_Propagate(
    const float* input, float* weight, float* biases,
    float* pre_activation, float* output,
    int batch_size, int input_size, int output_size, int activation_type)
{
    if (!input || !weight || !biases || !pre_activation || !output) 
    {
         throw std::runtime_error("Null pointer passed to cpu_func::forward_Propagate");
    }
     if (input_size <= 0 || output_size <= 0 || batch_size <= 0) 
    {
        throw std::runtime_error("Invalid dimensions in cpu_func::forward_Propagate");
    }

    // Step 1 & 2: Matrix Multiplication (Input * Weights) + Bias -> output (temporarily stores Z+b)
    parallel_for(0, batch_size * output_size, [&](int global_idx) 
    {
        int b = global_idx / output_size;
        int j = global_idx % output_size;
        float sum = 0.0f;
        for (int i = 0; i < input_size; ++i) 
        {
             int weight_index = j * input_size + i;
             int input_index = b * input_size + i;
             sum += input[input_index] * weight[weight_index];
        }
        sum += biases[j];

        // Store pre-activation (Z+b)
        pre_activation[global_idx] = sum;
        // Store result temporarily in output before activation
        output[global_idx] = sum;
    });


    // Step 3: Apply Activation (output -> output)
    // Softmax needs special handling per batch item
    if (activation_type == 3) 
    { // SOFTMAX
        parallel_for(0, batch_size, [&](int b) 
        {
            float* current_output = output + b * output_size;
        
            float max_val = -std::numeric_limits<float>::infinity();
            for (int j = 0; j < output_size; ++j) 
            {
                max_val = std::max(max_val, current_output[j]);
            }

            float sum_exp = 0.0f;
            for (int j = 0; j < output_size; ++j) 
            {
                current_output[j] = std::exp(current_output[j] - max_val);
                sum_exp += current_output[j];
            }
            float inv_sum = 1.0f / (sum_exp + 1e-9f);
            for (int j = 0; j < output_size; ++j) 
            {
                current_output[j] *= inv_sum;
            }
        });
    } 
    else 
    {
        parallel_for(0, batch_size * output_size, [&](int i) 
        {
            output[i] = activationFunction(output[i], activation_type);
        });
    }
}


// Corresponds to compute_weight_gradients_kernel
void compute_weight_gradients(
    const float* input, const float* output_error, float* weight_grad,
    int batch_size, int input_size, int output_size)
{
    memset(weight_grad, 0, (size_t)input_size * output_size * sizeof(float));
    parallel_for(0, output_size * input_size, [&](int global_idx) 
    {
        int j = global_idx / input_size; 
        int i = global_idx % input_size;
        float gradient_sum = 0.0f; 

        for (int b = 0; b < batch_size; ++b) 
        {
            gradient_sum += input[b * input_size + i] * output_error[b * output_size + j];
        }

        weight_grad[global_idx] = gradient_sum;
    });
}

// Corresponds to compute_bias_gradients_kernel
void compute_bias_gradients(
    const float* output_error, float* bias_grad,
    int batch_size, int output_size)
{
     memset(bias_grad, 0, (size_t)output_size * sizeof(float));

     parallel_for(0, output_size, [&](int j) 
     {
        float gradient_sum = 0.0f; 

        for (int b = 0; b < batch_size; ++b) 
        {
            gradient_sum += output_error[b * output_size + j];
        }
        bias_grad[j] = gradient_sum;
    });
}

// Combines the gradient calculations
void backward_propagate(
    float* input, float* weights, float* output_error,
    float* weight_grad, float* bias_grad,
    int batch_size, int input_size, int output_size)
{
    compute_weight_gradients(input, output_error, weight_grad, batch_size, input_size, output_size);
    compute_bias_gradients(output_error, bias_grad, batch_size, output_size);
}


// Corresponds to compute_hidden_error_kernel
void compute_hidden_error(
     float* weights,  float* output_error, float* hidden_error,
    int batch_size, int input_size, int output_size)
{
    memset(hidden_error, 0, (size_t)batch_size * input_size * sizeof(float));

    parallel_for(0, batch_size * input_size, [&](int global_idx) 
    {
        int b = global_idx / input_size;
        int i = global_idx % input_size;
        float error = 0.0f;

        for (int j = 0; j < output_size; ++j) 
        { 
             error += weights[j * input_size + i] * output_error[b * output_size + j];
        }
        hidden_error[global_idx] = error;
    });
}

// Corresponds to update_params_kernel
void update_params(
    float* weights, float* biases, const float* weight_grad, const float* bias_grad,
    int batch_size, int input_size, int output_size, float learning_rate)
{
    if (batch_size <= 0) return;
    float norm_factor = 1.0f;
    parallel_for(0, input_size * output_size, [&](int i) 
    {
        weights[i] -= learning_rate * (weight_grad[i] * norm_factor);
    });
    parallel_for(0, output_size, [&](int i) 
    {
        biases[i] -= learning_rate * (bias_grad[i] * norm_factor);
    });
}


// --- Convolutional Layer ---
// Corresponds to conv2d_forward_kernel
void conv2d_forward(
    const float* input, const float* kernels, const float* bias,
    float* pre_activation_output, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width,
    int kernel_size, int stride, int padding, int activation_type)
{
    int out_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (input_width + 2 * padding - kernel_size) / stride + 1;

    if (out_height <= 0 || out_width <= 0) 
    {
        throw std::runtime_error("Invalid output dimensions in conv2d_forward");
    }

    memset(pre_activation_output, 0, (size_t)batch_size * out_channels * out_height * out_width * sizeof(float));
    memset(output, 0, (size_t)batch_size * out_channels * out_height * out_width * sizeof(float));

    parallel_for_2d(0, batch_size * out_channels, 0, out_height * out_width, [&](int nc_out, int hw_out) {
        int n = nc_out / out_channels;         
        int c_out = nc_out % out_channels;     
        int h_out = hw_out / out_width;   
        int w_out = hw_out % out_width;

        float sum = bias[c_out];
        int h_in_start = h_out * stride - padding;
        int w_in_start = w_out * stride - padding;

        // Inner loops for convolution remain sequential within the parallel task, maybe not a good idea
        // but this is how the original code was structured.
        for (int c_in = 0; c_in < in_channels; ++c_in) 
        {
            for (int kh = 0; kh < kernel_size; ++kh) 
            {
                int h_in = h_in_start + kh;
                if (h_in < 0 || h_in >= input_height) continue;

                for (int kw = 0; kw < kernel_size; ++kw) 
                {
                    int w_in = w_in_start + kw;
                    if (w_in < 0 || w_in >= input_width) continue;

                    int in_idx = ((n * in_channels + c_in) * input_height + h_in) * input_width + w_in;
 
                    int k_idx = ((c_in * out_channels + c_out) * kernel_size + kh) * kernel_size + kw;
                    sum += input[in_idx] * kernels[k_idx];
                }
            }
        }
        int out_idx = ((n * out_channels + c_out) * out_height + h_out) * out_width + w_out;
        pre_activation_output[out_idx] = sum;
        if (activation_type != 3) 
        {
            output[out_idx] = activationFunction(sum, activation_type);
        } 
        else 
        {
            output[out_idx] = sum;
        }
    });
}

// Corresponds to conv2d_backward_kernel (calculates kernel and bias gradients)
void conv2d_backward(
    const float* input,                 // 1. Input X
    const float* kernels,               // 2. Kernels W (Not used)
    const float* pre_activation_output, // 3. Pre-activation Z
    const float* output_error,          // 4. Incoming error dL/dY
    float* kernel_grad,                 // 5. Output: Kernel gradient dL/dK
    float* bias_grad,                   // 6. Output: Bias gradient dL/dB
    float* input_grad,                  // 7. REMOVED
    int batch_size,                     // 8. Batch Size
    int in_channels,                    // 9. Input Channels (ic)
    int out_channels,                   // 10. Output Channels (c / oc)
    int input_height,                   // 11. Input Height
    int input_width,                    // 12. Input Width
    int kernel_size,                    // 13. Kernel Size (ky, kx)
    int stride,                         // 14. Stride
    int padding,                        // 15. Padding
    int activation_type                 // 16. Activation Type
)
{
    // --- Calculate Output Dimensions ---
    int out_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (input_width + 2 * padding - kernel_size) / stride + 1;

    // --- Zero Gradients ---
    size_t kernel_grad_size_bytes = (size_t)in_channels * out_channels * kernel_size * kernel_size * sizeof(float);
    size_t bias_grad_size_bytes = (size_t)out_channels * sizeof(float);
    memset(kernel_grad, 0, kernel_grad_size_bytes);
    memset(bias_grad, 0, bias_grad_size_bytes);

    // --- Calculate Bias Gradients (dL/dB)
    parallel_for(0, out_channels, [&](int c) 
    {
        float bias_grad_sum = 0.0f;
        size_t base_offset_c = (size_t)c * out_height * out_width; 

        for (int n = 0; n < batch_size; ++n) 
        {
            size_t base_offset_nc = (size_t)n * out_channels * out_height * out_width + base_offset_c;

            for (int h = 0; h < out_height; ++h) 
            {
                size_t base_offset_nch = base_offset_nc + (size_t)h * out_width;

                for (int w = 0; w < out_width; ++w) 
                {
                    size_t out_idx = base_offset_nch + w;

                    float delta_val;
                    if (activation_type == 3) 
                    {
                        delta_val = output_error[out_idx];
                    } 
                    else 
                    {
                        float activation_grad = activationFunctionDerivative(pre_activation_output[out_idx], activation_type);
                        delta_val = output_error[out_idx] * activation_grad;
                    }

                    bias_grad_sum += delta_val;
                }
            }
        }
        bias_grad[c] = bias_grad_sum;
    });

    // --- Calculate Kernel Gradients (dL/dK)
    size_t kernel_elements = (size_t)in_channels * out_channels * kernel_size * kernel_size;
    const int UNROLL_FACTOR_W = 4;

    parallel_for(0, kernel_elements, [&](int global_kernel_idx) 
    {
        int kx = global_kernel_idx % kernel_size;
        int temp_idx = global_kernel_idx / kernel_size;
        int ky = temp_idx % kernel_size;
        temp_idx /= kernel_size;
        int c = temp_idx % out_channels;
        int ic = temp_idx / out_channels;

        float kernel_grad_sum_total = 0.0f;

        for (int n = 0; n < batch_size; ++n) 
        {
            const float* input_nic = input + (n * in_channels + ic) * input_height * input_width;

            size_t base_offset_nc_out = (size_t)n * out_channels * out_height * out_width + (size_t)c * out_height * out_width;

            for (int h = 0; h < out_height; ++h) 
            {
                int input_y_base = h * stride - padding;
                int iy = input_y_base + ky;

                // --- Prefetching might need adjustment if delta calculation is costly ---
                int next_h = h + 1;
                if (next_h < out_height) 
                {
                    int next_iy = next_h * stride - padding + ky;
                    size_t next_out_idx_start = base_offset_nc_out + (size_t)next_h * out_width;
                     __builtin_prefetch(output_error + next_out_idx_start, 0, 0);
                     __builtin_prefetch(pre_activation_output + next_out_idx_start, 0, 0);
                    if (next_iy >= 0 && next_iy < input_height) 
                    {
                        __builtin_prefetch(input_nic + next_iy * input_width, 0, 0);
                    }
                }

                if (iy >= 0 && iy < input_height) 
                {
                    const float* input_nicy = input_nic + iy * input_width;
                    int input_x_base = 0 * stride - padding;
                    int ix_base = input_x_base + kx;

                    size_t base_offset_nch_out = base_offset_nc_out + (size_t)h * out_width;

                    // Unrolled loop over output width (w)
                    float kernel_grad_sum0 = 0.0f;
                    float kernel_grad_sum1 = 0.0f;
                    float kernel_grad_sum2 = 0.0f;
                    float kernel_grad_sum3 = 0.0f;
                    int w = 0;
                    for (; w <= out_width - UNROLL_FACTOR_W; w += UNROLL_FACTOR_W) 
                    {
                        int ix0 = ix_base + w * stride;
                        int ix1 = ix_base + (w + 1) * stride;
                        int ix2 = ix_base + (w + 2) * stride;
                        int ix3 = ix_base + (w + 3) * stride;
                        bool valid_ix0 = (ix0 >= 0 && ix0 < input_width);
                        bool valid_ix1 = (ix1 >= 0 && ix1 < input_width);
                        bool valid_ix2 = (ix2 >= 0 && ix2 < input_width);
                        bool valid_ix3 = (ix3 >= 0 && ix3 < input_width);

                        // --- Calculate delta locally for each w ---
                        size_t out_idx0 = base_offset_nch_out + w;
                        size_t out_idx1 = base_offset_nch_out + w + 1;
                        size_t out_idx2 = base_offset_nch_out + w + 2;
                        size_t out_idx3 = base_offset_nch_out + w + 3;

                        float delta_val0=0.f, delta_val1=0.f, delta_val2=0.f, delta_val3=0.f;

                        if (activation_type == 3) 
                        { 
                            delta_val0 = output_error[out_idx0];
                            delta_val1 = output_error[out_idx1];
                            delta_val2 = output_error[out_idx2];
                            delta_val3 = output_error[out_idx3];
                        } 
                        else 
                        {
                            float act_grad0 = activationFunctionDerivative(pre_activation_output[out_idx0], activation_type);
                            float act_grad1 = activationFunctionDerivative(pre_activation_output[out_idx1], activation_type);
                            float act_grad2 = activationFunctionDerivative(pre_activation_output[out_idx2], activation_type);
                            float act_grad3 = activationFunctionDerivative(pre_activation_output[out_idx3], activation_type);
                            delta_val0 = output_error[out_idx0] * act_grad0;
                            delta_val1 = output_error[out_idx1] * act_grad1;
                            delta_val2 = output_error[out_idx2] * act_grad2;
                            delta_val3 = output_error[out_idx3] * act_grad3;
                        }

                        if (valid_ix0) { kernel_grad_sum0 += input_nicy[ix0] * delta_val0; }
                        if (valid_ix1) { kernel_grad_sum1 += input_nicy[ix1] * delta_val1; }
                        if (valid_ix2) { kernel_grad_sum2 += input_nicy[ix2] * delta_val2; }
                        if (valid_ix3) { kernel_grad_sum3 += input_nicy[ix3] * delta_val3; }
                    }
                    float unrolled_sum = kernel_grad_sum0 + kernel_grad_sum1 + kernel_grad_sum2 + kernel_grad_sum3;

                    float remainder_sum = 0.0f;
                    for (; w < out_width; ++w) 
                    {
                        int ix = ix_base + w * stride;
                        if (ix >= 0 && ix < input_width) 
                        {
                            size_t out_idx = base_offset_nch_out + w;

                            float delta_val;
                            if (activation_type == 3) 
                            { 
                                delta_val = output_error[out_idx];
                            } 
                            else 
                            {
                                float activation_grad = activationFunctionDerivative(pre_activation_output[out_idx], activation_type);
                                delta_val = output_error[out_idx] * activation_grad;
                            }
                            remainder_sum += input_nicy[ix] * delta_val;
                        }
                    }
                    kernel_grad_sum_total += unrolled_sum + remainder_sum;
                }
            }
        } 
        kernel_grad[global_kernel_idx] = kernel_grad_sum_total;
    });
}


// Corresponds to conv2d_compute_hidden_error_kernel (calculates dL/dX)
void conv2d_compute_hidden_error(
    const float* kernels,               
    const float* output_error,          
    const float* pre_activation_output, 
    float* prev_output_error,        
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int out_height,                    
    int out_width,                     
    int kernel_size,
    int stride,
    int padding,
    int activation_type
)
{

    size_t hidden_error_size_bytes = (size_t)batch_size * in_channels * input_height * input_width * sizeof(float);
    memset(prev_output_error, 0, hidden_error_size_bytes);

    // --- Define Tile Sizes (Tunable Parameters, would be nice to calculate them instead of hardcoding but hella hard) ---
    const int TILE_OC = 16;
    const int TILE_OH = 8;
    const int TILE_OW = 16;
    const int UNROLL_FACTOR_OW = 4;

    parallel_for(0, batch_size * in_channels, [&](int bc_idx) {
        int b = bc_idx / in_channels;  
        int c_in = bc_idx % in_channels;

        // Kernel layout: [in_channels][out_channels][kh][kw]
        const float* kernels_cin = kernels + c_in * (out_channels * kernel_size * kernel_size);
        const float* output_error_b = output_error + b * out_channels * out_height * out_width;

        const float* pre_activation_output_b = pre_activation_output + b * out_channels * out_height * out_width;
        float* prev_output_error_bc = prev_output_error + bc_idx * input_height * input_width;
        
        // Implement unrolling and tiling
        for (int h_in = 0; h_in < input_height; ++h_in) 
        {
            for (int w_in = 0; w_in < input_width; ++w_in) 
            {
                float error_sum_total = 0.0f; 
                
                int oh_min_full = std::max(0, (h_in + padding - kernel_size + stride) / stride);
                int oh_max_full = std::min(out_height, (h_in + padding) / stride + 1);
                int ow_min_full = std::max(0, (w_in + padding - kernel_size + stride) / stride);
                int ow_max_full = std::min(out_width, (w_in + padding) / stride + 1);

                for (int oc_tile_start = 0; oc_tile_start < out_channels; oc_tile_start += TILE_OC) 
                {
                    int oc_tile_end = std::min(oc_tile_start + TILE_OC, out_channels);

                    for (int oh_tile_start = oh_min_full; oh_tile_start < oh_max_full; oh_tile_start += TILE_OH) {
                        int oh_tile_end = std::min(oh_tile_start + TILE_OH, oh_max_full);

                        for (int ow_tile_start = ow_min_full; ow_tile_start < ow_max_full; ow_tile_start += TILE_OW) {
                            int ow_tile_end = std::min(ow_tile_start + TILE_OW, ow_max_full);

                            for (int oc = oc_tile_start; oc < oc_tile_end; ++oc) 
                            {
                                const float* kernels_cin_oc = kernels_cin + oc * (kernel_size * kernel_size);
                                const float* output_error_boc = output_error_b + oc * out_height * out_width;
                                const float* pre_activation_output_boc = pre_activation_output_b + oc * out_height * out_width;

                                for (int oh = oh_tile_start; oh < oh_tile_end; ++oh) 
                                {
                                    int kh = h_in + padding - oh * stride;

                                    if (kh >= 0 && kh < kernel_size) 
                                    {
                                        const float* output_error_boch = output_error_boc + oh * out_width;
                                        const float* pre_activation_output_boch = pre_activation_output_boc + oh * out_width;
                                        const float* kernels_cin_oc_kh = kernels_cin_oc + kh * kernel_size;

                                        float error_sum0 = 0.0f;
                                        float error_sum1 = 0.0f;
                                        float error_sum2 = 0.0f;
                                        float error_sum3 = 0.0f;

                                        auto calculate_local_delta = [&](int current_ow) -> float 
                                        {
                                            if (current_ow < 0 || current_ow >= out_width) return 0.0f; 
                                            size_t out_idx = current_ow; 
                                            if (activation_type == 3) 
                                            {
                                                return output_error_boch[out_idx];
                                            } 
                                            else 
                                            {
                                                float activation_grad = activationFunctionDerivative(pre_activation_output_boch[out_idx], activation_type);
                                                return output_error_boch[out_idx] * activation_grad;
                                            }
                                        };
                                        int ow = ow_tile_start;
                                        for (; ow <= ow_tile_end - UNROLL_FACTOR_OW; ow += UNROLL_FACTOR_OW) {
                                            int kw0 = w_in + padding - ow * stride;
                                            int kw1 = w_in + padding - (ow + 1) * stride;
                                            int kw2 = w_in + padding - (ow + 2) * stride;
                                            int kw3 = w_in + padding - (ow + 3) * stride;

                                            bool valid_kw0 = (kw0 >= 0 && kw0 < kernel_size);
                                            bool valid_kw1 = (kw1 >= 0 && kw1 < kernel_size);
                                            bool valid_kw2 = (kw2 >= 0 && kw2 < kernel_size);
                                            bool valid_kw3 = (kw3 >= 0 && kw3 < kernel_size);

                                            float delta_val0 = calculate_local_delta(ow);
                                            float delta_val1 = calculate_local_delta(ow + 1);
                                            float delta_val2 = calculate_local_delta(ow + 2);
                                            float delta_val3 = calculate_local_delta(ow + 3);

                                            if (valid_kw0) { error_sum0 += delta_val0 * kernels_cin_oc_kh[kw0]; }
                                            if (valid_kw1) { error_sum1 += delta_val1 * kernels_cin_oc_kh[kw1]; }
                                            if (valid_kw2) { error_sum2 += delta_val2 * kernels_cin_oc_kh[kw2]; }
                                            if (valid_kw3) { error_sum3 += delta_val3 * kernels_cin_oc_kh[kw3]; }
                                        }
                                        float unrolled_sum = error_sum0 + error_sum1 + error_sum2 + error_sum3;

                                        float remainder_sum = 0.0f;
                                        for (; ow < ow_tile_end; ++ow) 
                                        {
                                            int kw = w_in + padding - ow * stride;
                                            if (kw >= 0 && kw < kernel_size) 
                                            {
                                                float delta_val = calculate_local_delta(ow); 
                                                remainder_sum += delta_val * kernels_cin_oc_kh[kw]; 
                                            }
                                        }
                                        error_sum_total += unrolled_sum + remainder_sum;
                                    } 
                                }
                            }
                        }
                    } 
                } 
                prev_output_error_bc[h_in * input_width + w_in] = error_sum_total;

            }
        }
    });
}


// Corresponds to conv2d_update_params_kernel
void conv2d_update_params(
    float* weights, float* bias, const float* weight_grad, const float* bias_grad,
    int batch_size, int in_channels, int out_channels, int kernel_size,
    int stride, int padding, float learning_rate)
{
    if (batch_size <= 0) return;
    float norm_factor = 1.0f / static_cast<float>(batch_size);

    size_t total_weights = (size_t)out_channels * in_channels * kernel_size * kernel_size;

    parallel_for(0, total_weights, [&](int i) 
    {
        weights[i] -= learning_rate * (weight_grad[i] * norm_factor);
    });

    parallel_for(0, out_channels, [&](int i) 
    {
        bias[i] -= learning_rate * (bias_grad[i] * norm_factor);
    });
}


// --- Loss Functions ---
// Corresponds to evaluate_loss_kernel template
template <typename LossFunction>
void evaluate_loss_cpu(
    const float* output,
    const float* target,
    float* element_loss,
    int size)
{
    auto loss_expr = LossFunction::expression();
    parallel_for(0, size, [&](int i) 
    {
        float o = output[i];
        if constexpr (std::is_same_v<LossFunction, autodiff::loss::BCELoss>) {
             o = std::max(1e-9f, std::min(1.0f - 1e-9f, o));
        }
        element_loss[i] = loss_expr.eval(o, target[i]);
    });
}

// Corresponds to compute_loss_error_kernel template
template <typename LossFunction>
void compute_loss_error_cpu(
    const float* output,
    const float* target,
    float* error,
    int batch_size,
    int output_size)
{
    int total_size = batch_size * output_size;
    if (total_size == 0) return;
    auto loss_expr = LossFunction::expression();

    float norm_factor = 1.0f / static_cast<float>(total_size);
    parallel_for(0, total_size, [&](int i) 
    {
        float o = output[i];
        error[i] = loss_expr.grad(o, target[i]) / total_size;
    });
}

// Special CPU implementation for Softmax + Cross Entropy Gradient
// Corresponds to softmax_cross_entropy_gradient_kernel
void softmax_cross_entropy_gradient_cpu(
    const float* logits,    
    const float* target,    
    float* error,          
    int batch_size,
    int output_size  
) {
    if (batch_size == 0 || output_size == 0) return;
    float norm_factor = 1.0f / static_cast<float>(batch_size);

    parallel_for(0, batch_size, [&](int b) 
    {
        const float* current_logits = logits + b * output_size;
        const float* current_target = target + b * output_size;
        float* current_error = error + b * output_size;
        std::vector<float> softmax_output(output_size);

        // --- Step 1: Find max logit ---
        float max_logit = -std::numeric_limits<float>::infinity();
        for (int k = 0; k < output_size; ++k) 
        {
            max_logit = std::max(max_logit, current_logits[k]);
        }

        // --- Step 2: Calculate sum of exponentials ---
        float sum_exp = 0.0f;
        for (int k = 0; k < output_size; ++k) 
        {
            sum_exp += std::exp(current_logits[k] - max_logit);
        }
        float inv_sum = 1.0f / (sum_exp + 1e-9f);

        // --- Step 3: Calculate softmax probability and gradient ---
        for (int i = 0; i < output_size; ++i) 
        {
            softmax_output[i] = std::exp(current_logits[i] - max_logit) * inv_sum;
            current_error[i] = (softmax_output[i] - current_target[i]) * norm_factor;
        }
    });
}


// Corresponds to extern "C" void compute_output_error
void compute_output_error(
    const float* output_or_logits,
    const float* target,
    float* error,
    int batch_size,
    int output_size,
    const char* loss_type)
{
    if (strcmp(loss_type, "cross_entropy") == 0) 
    {
        softmax_cross_entropy_gradient_cpu(output_or_logits, target, error, batch_size, output_size);
    } 
    else if (strcmp(loss_type, "mse") == 0) 
    {
        compute_loss_error_cpu<autodiff::loss::MSELoss>(output_or_logits, target, error, batch_size, output_size);
    } 
    else if (strcmp(loss_type, "bce") == 0) 
    {
        compute_loss_error_cpu<autodiff::loss::BCELoss>(output_or_logits, target, error, batch_size, output_size);
    } 
    else if (strcmp(loss_type, "l1") == 0) 
    {
        compute_loss_error_cpu<autodiff::loss::L1Loss>(output_or_logits, target, error, batch_size, output_size);
    } 
    else if (strcmp(loss_type, "custom") == 0) 
    {
        compute_loss_error_cpu<autodiff::loss::CustomLoss>(output_or_logits, target, error, batch_size, output_size);
    } 
    else if (strcmp(loss_type, "huber") == 0) 
    {
        compute_loss_error_cpu<autodiff::loss::HuberLoss>(output_or_logits, target, error, batch_size, output_size);
    } 
    else 
    {
        // Default to MSE
        WARN_PRINT("Unsupported loss type '%s' in compute_output_error, defaulting to MSE.\n", loss_type);
        compute_loss_error_cpu<autodiff::loss::MSELoss>(output_or_logits, target, error, batch_size, output_size);
    }
}

// Special CPU implementation for Softmax + Cross Entropy Evaluation
// Corresponds to softmax_cross_entropy_eval_kernel
void softmax_cross_entropy_eval_cpu(
    const float* logits,  
    const float* target,   
    float* element_loss,
    int batch_size,
    int output_size  
) {
     if (batch_size == 0 || output_size == 0) return;

    parallel_for(0, batch_size, [&](int b) 
    {
        const float* current_logits = logits + b * output_size;
        const float* current_target = target + b * output_size;
        float* current_loss = element_loss + b * output_size; 
        std::vector<float> softmax_prob(output_size);

        // --- Step 1: Find max logit ---
        float max_logit = -std::numeric_limits<float>::infinity();
        for (int k = 0; k < output_size; ++k) 
        {
            max_logit = std::max(max_logit, current_logits[k]);
        }

        // --- Step 2: Calculate sum of exponentials ---
        float sum_exp = 0.0f;
        for (int k = 0; k < output_size; ++k) 
        {
            sum_exp += std::exp(current_logits[k] - max_logit);
        }
        float inv_sum = 1.0f / (sum_exp + 1e-9f);

        // --- Step 3: Calculate softmax probability and loss ---
        float batch_item_loss = 0.0f;
        for (int i = 0; i < output_size; ++i) 
        {
            softmax_prob[i] = std::exp(current_logits[i] - max_logit) * inv_sum;

            batch_item_loss += -current_target[i] * std::log(std::max(softmax_prob[i], 1e-9f));
        }
        element_loss[b] = batch_item_loss;
    });
}


// Corresponds to extern "C" void calculate_loss_values
void calculate_loss_values(
    const float* output_or_logits,
    const float* target,
    float* element_loss,
    int size,
    const char* loss_type,
    int batch_size)
{
    if (size <= 0 || batch_size <= 0) return;
    int output_size = size / batch_size;
    if (output_size <= 0) return;

    if (strcmp(loss_type, "cross_entropy") == 0) 
    {
        softmax_cross_entropy_eval_cpu(output_or_logits, target, element_loss, batch_size, output_size);
    } 
    else 
    {
        if (strcmp(loss_type, "mse") == 0) 
        {
            evaluate_loss_cpu<autodiff::loss::MSELoss>(output_or_logits, target, element_loss, size);
        } 
        else if (strcmp(loss_type, "bce") == 0) 
        {
            evaluate_loss_cpu<autodiff::loss::BCELoss>(output_or_logits, target, element_loss, size);
        } 
        else if (strcmp(loss_type, "l1") == 0) 
        {
            evaluate_loss_cpu<autodiff::loss::L1Loss>(output_or_logits, target, element_loss, size);
        } 
        else if (strcmp(loss_type, "custom") == 0) 
        {
            evaluate_loss_cpu<autodiff::loss::CustomLoss>(output_or_logits, target, element_loss, size);
        } 
        else if (strcmp(loss_type, "huber") == 0) 
        {
            evaluate_loss_cpu<autodiff::loss::HuberLoss>(output_or_logits, target, element_loss, size);
        } 
        else 
        {
            // Default to MSE
            WARN_PRINT("Unsupported loss type '%s' in calculate_loss_values, defaulting to MSE.\n", loss_type);
            evaluate_loss_cpu<autodiff::loss::MSELoss>(output_or_logits, target, element_loss, size);
        }
    }
}


// --- Utility ---
// Corresponds to unflatten_gradient_kernel
void unflatten_gradient(
    const float* flattened_grad, float* unflattened_grad,
    int batch_size, int channels, int height, int width)
{
    size_t total_elements = (size_t)batch_size * channels * height * width;
    // Simple memcpy assumes the internal layout is already correct (NCHW)
    memcpy(unflattened_grad, flattened_grad, total_elements * sizeof(float));
}

}