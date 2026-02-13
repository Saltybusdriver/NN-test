#include <cmath>
#include <cuda_runtime.h>
#include "cuda_functions.h"
#include <cstdio>
#include <cstdlib>
#include "debug.h"
#include "autodiff.h"

#include <cooperative_groups.h>

#include <cuda_runtime.h>      
#include <cmath>             
#include <algorithm>     

__device__ float Sigmoid(float x)
{
    return 1.0 / (1.0 + exp(-x));
}

__device__ float ReLU(float x)
{
    return fmaxf(0.0f, x);
}
__device__ float LeakyReLU(float x,float negative_slope)
{
    return fmaxf(x * negative_slope, x); 
}

__device__ float SigmoidDerivative(float x) {
    float sigmoid = Sigmoid(x);
    return sigmoid * (1.0f - sigmoid);
}

__device__ float ReLUDerivative(float x) 
{
    float cmp = x > 0.0f;
    return fmaxf(cmp, 0.0f);  // Returns 1.0f if x > 0, 0.0f otherwise
}

__device__ float LeakyReLUDerivative(float x, float negative_slope) 
{
    float cmp = x > 0.0f;
    return fmaxf(cmp * 1.0f, (1.0f - cmp) * negative_slope);
}

__device__ float SoftmaxKernel(float *input, float* output, int size) 
{
    __shared__ float s_max;
    __shared__ float s_sum;

    float thread_max = -INFINITY;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        thread_max = fmaxf(thread_max, input[i]);
    }

    if (threadIdx.x == 0) {
        s_max = thread_max;
    }
    __syncthreads();
    atomicMax((int*)&s_max, __float_as_int(thread_max));
    __syncthreads();

    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        float exp_val = expf(input[i] - s_max);
        output[i] = exp_val;
        thread_sum += exp_val;
    }
    
    if (threadIdx.x == 0) {
        s_sum = 0.0f;
    }
    __syncthreads();
    atomicAdd(&s_sum, thread_sum);
    __syncthreads();

    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        output[i] /= s_sum;
    }
}

__device__ float activationFunction(float input,int type)
{
    switch(type)
        {
            case 0:
                return input=Sigmoid(input);
                break;
            case 1:
                return input=ReLU(input);
                break;
            case 2:
                return input=LeakyReLU(input,0.01);
                break;
            default:
                return input;
        }
}

__device__ float activationFunctionDerivative(float input,int type)
{
    switch(type)
        {
            case 0:
                return input=SigmoidDerivative(input);
                break;
            case 1:
                return input=ReLUDerivative(input);
                break;
            case 2:
                return input=LeakyReLUDerivative(input,0.01);
                break;
            default:
                return 1.0f;
        }
}

__global__ void add_vectors_kernel(float* out, const float* bias, int size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) 
    {
        out[idx] += bias[idx];
    }
}

__global__ void activationKernel(float *input, float *output, int size, int activation_type)
{
    int idx=blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<size)
    {
        switch(activation_type)
        {
            case 0:
                output[idx]=Sigmoid(input[idx]);
                break;
            case 1:
                output[idx]=ReLU(input[idx]);
                break;
            case 2:
                output[idx]=LeakyReLU(input[idx],0.01);
                break;
            default:
                output[idx]=input[idx];
                break;
        }
    }
}

__global__ void softmax_cross_entropy_eval_kernel(
    const float* logits,    // Input: Raw scores from the final layer (pre-activation)
    const float* target,    // Input: One-hot encoded target labels
    float* element_loss,    // Output: Per-element loss -t * log(softmax(p))
    int batch_size,
    int output_size 
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * output_size;

    if (idx >= total_elements) return;

    int b = idx / output_size;
    int i = idx % output_size;

    const float* current_logits = logits + b * output_size;
    const float* current_target = target + b * output_size;

    // --- Step 1: Find max logit for this batch item ---
    float max_logit = -INFINITY;
    for (int k = 0; k < output_size; ++k) 
    {
        max_logit = fmaxf(max_logit, current_logits[k]);
    }

    // --- Step 2: Calculate sum of exponentials for this batch item ---
    float sum_exp = 0.0f;
    for (int k = 0; k < output_size; ++k) 
    {
        sum_exp += expf(current_logits[k] - max_logit);
    }
    float inv_sum = 1.0f / (sum_exp + 1e-9f);

    // --- Step 3: Calculate softmax probability and loss for the current element 'i' ---
    float softmax_prob = expf(current_logits[i] - max_logit) * inv_sum;
    element_loss[idx] = -current_target[i] * logf(fmaxf(softmax_prob, 1e-9f));
}

__global__ void softmax_cross_entropy_gradient_kernel(
    const float* logits,    // Input: Raw scores from the final layer (pre-activation)
    const float* target,    // Input: One-hot encoded target labels
    float* error,           // Output: Gradient dL/dZ = (softmax(logits) - target) / batch_size
    int batch_size,
    int output_size        
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * output_size;

    if (idx >= total_elements) return;

    // Determine batch item and element within batch
    int b = idx / output_size;
    int i = idx % output_size;

    const float* current_logits = logits + b * output_size;
    const float* current_target = target + b * output_size;

    // --- Step 1: Find max logit for this batch item ---
    float max_logit = -INFINITY;
    for (int k = 0; k < output_size; ++k) 
    {
        max_logit = fmaxf(max_logit, current_logits[k]);
    }

    // --- Step 2: Calculate sum of exponentials for this batch item ---
    float sum_exp = 0.0f;
    for (int k = 0; k < output_size; ++k) 
    {
        sum_exp += expf(current_logits[k] - max_logit);
    }
    float inv_sum = 1.0f / (sum_exp + 1e-9f);

    // --- Step 3: Calculate softmax probability and gradient for the current element 'i' ---
    float softmax_output = expf(current_logits[i] - max_logit) * inv_sum; 
    
    // if (b == 0) {
    //     printf("Kernel Debug (idx=%d): b=%d, i=%d\n", idx, b, i);
    //     printf("  logits[i]=%.6e, target[i]=%.6e\n", current_logits[i], current_target[i]);
    //     printf("  max_logit=%.6e, sum_exp=%.6e, inv_sum=%.6e\n", max_logit, sum_exp, inv_sum);
    //     printf("  softmax_output=%.6e\n", softmax_output);
    //     printf("  batch_size=%d\n", batch_size);
    //     printf("  Calculated error[idx=%d] = %.6e\n", idx, (softmax_output - current_target[i]) / batch_size);
    // }

    error[idx] = (softmax_output - current_target[i]) / batch_size;
}



__global__ void softmax_kernel(const float* input, float* output, int size) {
    __shared__ float shared_max;
    __shared__ float shared_sum;
    
    // Step 1: Find maximum value for numerical stability
    float thread_max = -INFINITY;
    for (int i = threadIdx.x; i < size; i += blockDim.x) 
    {
        thread_max = fmaxf(thread_max, input[i]);
    }

    if (threadIdx.x == 0) {
        shared_max = -INFINITY;
    }
    __syncthreads();
    atomicMax((int*)&shared_max, __float_as_int(thread_max));
    __syncthreads();

    // Step 2: Compute exponentials and their sum
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < size; i += blockDim.x) 
    {
        float exp_val = expf(input[i] - shared_max);
        output[i] = exp_val;  
        thread_sum += exp_val;
    }
    
    if (threadIdx.x == 0) {
        shared_sum = 0.0f;
    }
    __syncthreads();
    atomicAdd(&shared_sum, thread_sum);
    __syncthreads();

    // Step 3: Normalize with the sum
    float inv_sum = 1.0f / (shared_sum + 1e-9f);  
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        output[i] *= inv_sum;
    }
}

__global__ void cuda_forward_propagation(float* input, float* weight, float* biases, float* output,int batch_size, int input_size, int output_size,int activation_type)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size)
    {
        for(int out_idx=0;out_idx<output_size;++out_idx)
        {
            float sum = 0.0f;
            for (int i = 0; i < input_size; ++i) 
            {
                int weight_index=i*output_size+out_idx;
                int input_index=idx*input_size+i;
                //printf("current information: i=%d, idx=%d, input_size=%d, output_size=%d, weight index=%d,weight value:%f\n", i, idx, input_size, output_size,i * input_size + idx,weight[weight_index]);
                //printf("thread: %d, pointer: %p, value: %d\n", idx, &(input[idx]), input[idx]);
                sum += input[input_index]*weight[weight_index]; 
            }
            sum += biases[out_idx];
            //ULTRA GIGA DIRTY DIRTY HACK REFACTOR LATER
            if(activation_type==3)
            {
                SoftmaxKernel(input,output,output_size);    
            }
            else
            {        
                output[out_idx]=activationFunction(sum,activation_type);
            }
        }
    }
}

__global__ void conv2d_forward_kernel(const float* input, const float* kernels, const float* bias, float* pre_activation_output,
                                    float* output, int batch_size, int in_channels, int out_channels,
                                    int input_height, int input_width, int kernel_size, int stride, int padding, int activation_type) 
{

    
    //standard formula, should do it differently than recalculate every kernel call
    int out_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (input_width + 2 * padding - kernel_size) / stride + 1;

    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int b_c = blockIdx.z;
    
    if (w >= out_width || h >= out_height) return;
    
    int c = b_c % out_channels;
    int n = b_c / out_channels;

    if (w >= out_width || h >= out_height || n >= batch_size || c >= out_channels) {
        return;
    }

    float sum = bias[c];

    //Account for stride and padding otherwise bruh moment
    int start_h = h * stride - padding;
    int start_w = w * stride - padding;


    int i_start = max(0, -start_h);
    int i_end   = min(kernel_size, input_height - start_h);
    int j_start = max(0, -start_w);
    int j_end   = min(kernel_size, input_width - start_w);

    // Convolution, start with input channel loop, kernel height loop
    for (int ic = 0; ic < in_channels; ++ic) 
    {
        for (int i = 0; i < kernel_size; ++i) 
        {
            int cur_h = start_h + i;
            if(cur_h < 0 || cur_h >= input_height)
                continue;
            for (int j = 0; j < kernel_size; ++j) 
            {
                int cur_w = start_w + j;
                if(cur_w < 0 || cur_w >= input_width)
                    continue;
                int in_idx = ((n * in_channels + ic) * input_height + cur_h) * input_width + cur_w;
                int k_idx = ic * (out_channels * kernel_size * kernel_size) +
                            c  * (kernel_size * kernel_size) +
                            i  * kernel_size +
                            j;
                sum += input[in_idx] * kernels[k_idx];
                if (isnan(sum) || isinf(sum)) 
                {
                    printf("Invalid sum at (n=%d,c=%d,h=%d,w=%d): %f\n", n, c, h, w, sum);
                }
            }
        }
    }

    //same shit as above for in_idx except output.
    int out_idx = ((n * out_channels + c) * out_height + h) * out_width + w;
    pre_activation_output[out_idx] = sum;

    output[out_idx] = activationFunction(sum, activation_type);
}


__global__ void mse_derivative_kernel(const float* output, const float* target, float* out_error, int size)
{
    int idx=blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<size)
    {
        out_error[idx]=(2.0f*(output[idx]-target[idx]))/size;
    }
}

__global__ void unflatten_gradient_kernel(
    const float* flattened_grad, // Input: Gradient from next layer [B, C*H*W]
    float* unflattened_grad,     // Output: Gradient for previous layer [B, C, H, W]
    int batch_size,
    int channels,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height * width;

    if (idx >= total_elements) return;

    int w = idx % width;
    int h = (idx / width) % height;
    int c = (idx / (width * height)) % channels;
    int n = idx / (width * height * channels);

    int flattened_idx = n * (channels * height * width) + c * (height * width) + h * width + w;

    unflattened_grad[idx] = flattened_grad[flattened_idx];
}

__global__ void conv2d_backward_kernel(
    const float* input,
    const float* kernels,
    const float* pre_activation_output,
    float* output_error,
    float* kernel_grad,
    float* bias_grad,
    float* input_error,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_size,
    int stride,
    int padding,
    int activation_type
) {
    extern __shared__ float s_data[];

    float* s_bias_grad = s_data;
    
    float* s_kernel_grad = s_data + out_channels; 


    int out_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    int out_height = (input_height + 2 * padding - kernel_size) / stride + 1;

    int w = blockIdx.x * blockDim.x + threadIdx.x; 
    int h = blockIdx.y * blockDim.y + threadIdx.y; 
    int b_c = blockIdx.z;                          

    int c = b_c % out_channels; 
    int n = b_c / out_channels; 

    if (threadIdx.x == 0 && threadIdx.y == 0) 
    {
         s_bias_grad[c] = 0.0f;
    }

    int kernel_grad_size_per_channel = in_channels * kernel_size * kernel_size;
    for (int i = threadIdx.x + blockDim.x * threadIdx.y; i < kernel_grad_size_per_channel; i += blockDim.x * blockDim.y) 
    {
        s_kernel_grad[i] = 0.0f;
    }
    __syncthreads();

    if (w >= out_width || h >= out_height) return;

    // --- Calculate Delta ---
    int out_idx = ((n * out_channels + c) * out_height + h) * out_width + w;
    float activation_grad = activationFunctionDerivative(pre_activation_output[out_idx], activation_type);
    float delta = output_error[out_idx] * activation_grad;

    // --- Accumulate Bias Gradient in Shared Memory ---
    atomicAdd(&s_bias_grad[c], delta);

    int input_x_base = w * stride - padding;
    int input_y_base = h * stride - padding;

    // --- Accumulate Kernel Gradients in Shared Memory ---
    for (int ic = 0; ic < in_channels; ++ic) 
    {
        for (int ky = 0; ky < kernel_size; ++ky) 
        {
            int iy = input_y_base + ky; 

            if (iy < 0 || iy >= input_height) 
            {
                continue;
            }

            for (int kx = 0; kx < kernel_size; ++kx) 
            {
                int ix = input_x_base + kx;
                if (ix < 0 || ix >= input_width) 
                {
                    continue;
                }
                int input_idx = ((n * in_channels + ic) * input_height + iy) * input_width + ix;

                // Calculate linear index for the shared kernel gradient element
                // This index is local to the output channel 'c' being processed by this block
                int shared_k_idx = ic * (kernel_size * kernel_size) + ky * kernel_size + kx;
                float input_value = input[input_idx];
                float grad_val = input_value * delta;

                atomicAdd(&s_kernel_grad[shared_k_idx], grad_val);
            }
        }
    }

    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) 
    {
        atomicAdd(&bias_grad[c], s_bias_grad[c]);
    }

    for (int i = threadIdx.x + blockDim.x * threadIdx.y; i < kernel_grad_size_per_channel; i += blockDim.x * blockDim.y) 
    {

        int ic = i / (kernel_size * kernel_size);
        int k_spatial = i % (kernel_size * kernel_size);
        int ky = k_spatial / kernel_size;
        int kx = k_spatial % kernel_size;

        int global_kernel_idx = ic * (out_channels * kernel_size * kernel_size) +
                                c * (kernel_size * kernel_size) +
                                ky * kernel_size +
                                kx;

        atomicAdd(&kernel_grad[global_kernel_idx], s_kernel_grad[i]);
    }
}



__global__ void conv2d_compute_hidden_error_kernel(
    const float* kernels,
    const float* output_error,          // Input: dL/dY
    const float* pre_activation_output, // Input: Z <--- ADDED
    float* prev_output_error,           // Output: dL/dX
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,

    int output_height,                 
    int output_width,             
    int kernel_size,
    int stride,
    int padding,
    int activation_type     
)
{

    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y; 
    int b_c = blockIdx.z;                 

    if (w >= input_width || h >= input_height) return;

    int c = b_c % in_channels;  
    int b = b_c / in_channels;  

    int idx = ((b * in_channels + c) * input_height + h) * input_width + w;

    float error_sum = 0.0f;

    // Calculate valid output region that this input pixel contributes to
    // These are indices in the output tensor (dL/dY, Z)
    int oh_min = max(0, (h + padding - kernel_size + stride) / stride);
    int oh_max = min(output_height, (h + padding) / stride + 1);
    int ow_min = max(0, (w + padding - kernel_size + stride) / stride);
    int ow_max = min(output_width, (w + padding) / stride + 1);

    for (int oc = 0; oc < out_channels; ++oc) 
    {
        size_t base_offset_boc_out = (size_t)b * out_channels * output_height * output_width + (size_t)oc * output_height * output_width;

        // Iterate over the relevant output locations
        for (int oh = oh_min; oh < oh_max; ++oh) 
        {
            size_t base_offset_boch_out = base_offset_boc_out + (size_t)oh * output_width;

            for (int ow = ow_min; ow < ow_max; ++ow) 
            {
                // Calculate the kernel indices (kh, kw) that connect (h, w) to (oh, ow)
                int kh = h + padding - oh * stride;
                int kw = w + padding - ow * stride;

                if (kh >= 0 && kh < kernel_size && kw >= 0 && kw < kernel_size) 
                {
             
                    int out_idx = base_offset_boch_out + ow;
                    // --- Calculate local delta (dL/dZ) ---
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

                    // Kernel layout: [in_channels][out_channels][kh][kw]
                    int k_idx = c * (out_channels * kernel_size * kernel_size) +
                              oc * (kernel_size * kernel_size) +
                              kh * kernel_size +
                              kw;
                    error_sum += delta_val * kernels[k_idx];
                }
            } 
        } 
    } 
    atomicAdd(&prev_output_error[idx], error_sum);
}

__global__ void conv2d_update_params_kernel(
    float* weights,         
    float* biases,    
    const float* weight_grad, 
    const float* bias_grad,   
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    float learning_rate
) {
    int total_weights = out_channels * in_channels * kernel_size * kernel_size;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_weights) 
    {
        weights[idx] -= learning_rate * weight_grad[idx] / batch_size;
    }
    if (idx < out_channels) 
    {
        biases[idx] -= learning_rate * bias_grad[idx] / batch_size;
    }
}


__global__ void mse_loss_kernel(float* output, float* target, float* loss, int size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_size = size;
        
        __shared__ float shared_loss[256];
        float local_loss = 0.0f;
        
        if (idx < total_size) 
        {
            float diff = output[idx] - target[idx];
            local_loss = diff * diff / total_size;
        }

        shared_loss[threadIdx.x] = local_loss;
        __syncthreads();
        
        for (int stride = blockDim.x/2; stride > 0; stride >>= 1) 
        {
            if (threadIdx.x < stride) 
            {
                shared_loss[threadIdx.x] += shared_loss[threadIdx.x + stride];
            }
            __syncthreads();
        }
        
        if (threadIdx.x == 0) 
        {
            atomicAdd(loss, shared_loss[0]);
        }
    }
}



__global__ void compute_weight_gradients_kernel(
    const float* input,
    const float* output_error,
    float* weight_grad,
    int batch_size,
    int input_size,
    int output_size
) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total_weights = input_size * output_size;
    if (idx >= total_weights) return;
    
    int input_idx = idx / output_size;
    int output_idx = idx % output_size;
    
    float gradient = 0.0f;
    
    for (int b = 0; b < batch_size; b++) 
    {
        int batch_input_offset = b * input_size;
        int batch_output_offset = b * output_size;
        

        if (input_idx < input_size && output_idx < output_size) 
        {
            gradient += input[batch_input_offset + input_idx] * 
                       output_error[batch_output_offset + output_idx];
        }
    }
    
    weight_grad[idx] = gradient / batch_size;
}

__global__ void compute_bias_gradients_kernel(
    const float* output_error,
    float* bias_grad,
    int batch_size,
    int output_size
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= output_size) return;

    float gradient = 0.0f;

    for (int b = 0; b < batch_size; b++) 
    {
        int batch_output_offset = b * output_size;
        if (idx < output_size) 
        {
            gradient += output_error[batch_output_offset + idx];
        }
    }

    bias_grad[idx] = gradient / batch_size;
}



// __global__ void compute_output_error_kernel(
//     float* output,     // network output 
//     const float* target,     // target values
//     float* error,      // output error storage
//     int batch_size,    // batch size (usually 1)
//     int output_size    // number of outputs
// )
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int total_size = batch_size * output_size;
//     if(idx < total_size) 
//     {
//         float gradient = 2.0f * (output[idx] - target[idx]) / total_size;
//         error[idx] = gradient;
//     }
// }

template <typename LossFunction>
__global__ void evaluate_loss_kernel(
    float* output,
    const float* target,
    float* temp_loss,
    int size
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        auto loss_expr = LossFunction::expression();
        temp_loss[idx] = loss_expr.eval(output[idx], target[idx]);
    }
}

template <typename LossFunction>
__global__ void compute_loss_error_kernel(
    float* output,
    const float* target,
    float* error,
    int batch_size,
    int output_size
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * output_size;
    if(idx < total_size) 
    {
        auto loss_expr = LossFunction::expression();
        float gradient = loss_expr.grad(output[idx], target[idx]) / total_size;
        error[idx] = gradient;
    }
}


__global__ void compute_hidden_error_kernel(
    float* weights,        
    float* output,         
    float* hidden_error,  
    int batch_size,       
    int input_size,        
    int output_size        
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * input_size) return;

    int b = idx / input_size; 
    int i = idx % input_size;   

    float error = 0.0f;
    for(int j = 0; j < output_size; j++) 
    {
        error += weights[i * output_size + j] * output[b * output_size + j];
    }
    hidden_error[idx] = error;
}

__global__ void update_params_kernel(
    float* weights,       
    float* biases,        
    float* weight_grad,   
    float* bias_grad,      
    int batch_size,       
    int input_size,        
    int output_size,      
    float learning_rate
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < input_size * output_size) 
    {
        int i = idx / output_size;    
        int j = idx % output_size;    
        weights[i * output_size + j] -= learning_rate * weight_grad[idx] / batch_size;
    }

    if (idx < output_size) 
    {
        biases[idx] -= learning_rate * bias_grad[idx] / batch_size;
    }
}

extern "C" void add_vectors(float* out, const float* bias, int size) 
{
    add_vectors_kernel<<<(size + 255) / 256, 256>>>(out, bias, size);

}



extern "C" void calculate_loss_values(
    float* output,
    const float* target,
    float* element_loss,  
    int size,
    const char* loss_type = "mse",
    int batch_size = 1
)
{
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    if (strcmp(loss_type, "cross_entropy") == 0) 
    {
        // Special handling for cross-entropy loss value calculation from logits
        int output_dim = size / batch_size;
        int blocks_softmax = batch_size;
        size_t shared_mem_size = (2 + output_dim) * sizeof(float);

        softmax_cross_entropy_eval_kernel<<<blocks_softmax, threads>>>(
            output, // Pass LOGITS here
            target,
            element_loss,
            batch_size,       // Pass batch_size
            output_dim        // Pass output_dim
        );
    }
    else if (strcmp(loss_type, "mse") == 0) 
    {
        evaluate_loss_kernel<autodiff::loss::MSELoss><<<blocks, threads>>>(
            output, target, element_loss, size);
    }
    else if (strcmp(loss_type, "bce") == 0) 
    {
        evaluate_loss_kernel<autodiff::loss::BCELoss><<<blocks, threads>>>(
            output, target, element_loss, size);
    }
    else if (strcmp(loss_type, "l1") == 0) 
    {
        evaluate_loss_kernel<autodiff::loss::L1Loss><<<blocks, threads>>>(
            output, target, element_loss, size);
    }
    else if (strcmp(loss_type, "custom") == 0) 
    {
        evaluate_loss_kernel<autodiff::loss::CustomLoss><<<blocks, threads>>>(
            output, target, element_loss, size);
    }
    else if (strcmp(loss_type, "huber") == 0) 
    {
        evaluate_loss_kernel<autodiff::loss::HuberLoss><<<blocks, threads>>>(
            output, target, element_loss, size);
    }
    else 
    {
        evaluate_loss_kernel<autodiff::loss::MSELoss><<<blocks, threads>>>(
            output, target, element_loss, size);
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in loss calculation: %s\n", cudaGetErrorString(err));
    }
}


extern "C" void compute_output_error(
    float* output,
    const float* target,
    float* error,
    int batch_size,
    int output_size,
    const char* loss_type = "mse"
)
{
    int threads = 256;
    int blocks = (batch_size * output_size + threads - 1) / threads;
    
    if (strcmp(loss_type, "cross_entropy") == 0)
    {
        int threads_softmax = 256;
        int blocks_softmax = batch_size; 
        size_t shared_mem_size = (2 + output_size) * sizeof(float);

        softmax_cross_entropy_gradient_kernel<<<blocks_softmax, threads_softmax>>>(
            output,
            target,
            error,
            batch_size,
            output_size
        );
    }
    if (strcmp(loss_type, "mse") == 0) 
    {
        compute_loss_error_kernel<autodiff::loss::MSELoss><<<blocks, threads>>>(
            output, target, error, batch_size, output_size);
    }
    else if (strcmp(loss_type, "bce") == 0) 
    {
        compute_loss_error_kernel<autodiff::loss::BCELoss><<<blocks, threads>>>(
            output, target, error, batch_size, output_size);
    }
    else if (strcmp(loss_type, "l1") == 0) 
    {
        compute_loss_error_kernel<autodiff::loss::L1Loss><<<blocks, threads>>>(
            output, target, error, batch_size, output_size);
    }
    else if (strcmp(loss_type, "custom") == 0) 
    {
        compute_loss_error_kernel<autodiff::loss::CustomLoss><<<blocks, threads>>>(
            output, target, error, batch_size, output_size);
    }
    else if (strcmp(loss_type,"cross_entropy")==0)
    {
        compute_loss_error_kernel<autodiff::loss::CrossEntropyLoss><<<blocks, threads>>>(
            output, target, error, batch_size,output_size);
    }
    else 
    {
        // Default to MSE
        compute_loss_error_kernel<autodiff::loss::MSELoss><<<blocks, threads>>>(
            output, target, error, batch_size, output_size);
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
        fprintf(stderr, "CUDA error in gradient calc: %s\n", cudaGetErrorString(err));
    }
}




extern "C" void compute_hidden_error(float* weights, float* output, float* hidden_error,
                         int batch_size, int input_size, int output_size) 
{
    int threads = 256;
    int blocks = (input_size * batch_size + threads - 1) / threads;

    compute_hidden_error_kernel<<<blocks, threads>>>(
        weights, output, hidden_error, batch_size, input_size, output_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error in hidden gradient calc: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


extern "C" void update_params(float* weights, float* biases, float* weight_grad,
                  float* bias_grad, int batch_size, int input_size,
                  int output_size, float learning_rate) {
    int threads = 256;
    int blocks = (input_size * output_size + threads - 1) / threads;

    update_params_kernel<<<blocks, threads>>>(
        weights, biases, weight_grad, bias_grad,
        batch_size, input_size, output_size, learning_rate);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error in param updating: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

extern "C" void calc_mse_loss_kernel(float* output, float* target, float* loss, int size)
{
    int blockSize=256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    mse_loss_kernel<<<numBlocks, blockSize>>>(output, target, loss, size);

    CUDA_CHECK_ERROR(cudaGetLastError());
}
extern "C" void backward_propagate(
    float* input,
    float* weights,
    float* output_error,
    float* weight_grad,
    float* bias_grad, 
    int batch_size,
    int input_size,
    int output_size
) {
    
    int total_weights = input_size * output_size;
    int threadsPerBlock = 256;
    int blocksNeeded = (total_weights + threadsPerBlock - 1) / threadsPerBlock;
    int biasBlocks = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    cudaMemset(weight_grad, 0, input_size * output_size * sizeof(float));
    cudaMemset(bias_grad, 0, output_size * sizeof(float));

    compute_weight_gradients_kernel<<<blocksNeeded, threadsPerBlock>>>(
        input,
        output_error,
        weight_grad,
        batch_size, 
        input_size,
        output_size
    );

    compute_bias_gradients_kernel<<<biasBlocks, threadsPerBlock>>>(
        output_error,
        bias_grad,
        batch_size,
        output_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
        fprintf(stderr, "error during backward: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

extern "C" void forward_Propagate(
    const float* input,         
    float* weight,               
    float* biases,             
    float* pre_activation,       
    float* output,          
    int batch_size,
    int input_size,
    int output_size,
    int activation_type
)
{
    if (input_size <= 0 || output_size <= 0) 
    {
        fprintf(stderr, "Invalid input_size (%d) or output_size (%d)\n", input_size, output_size);
        return;
    }

    if (input == nullptr || weight == nullptr || biases == nullptr || output == nullptr) 
    {
        printf("this pointer is: %p\n", input);
        printf("this pointer is: %p\n", weight);
        printf("this pointer is: %p\n", biases);
        printf("this pointer is: %p\n", output);
        throw("Invalid memory pointers");
        
        return;
    }
    size_t total_elements = (size_t)batch_size * output_size;
    size_t total_bytes = total_elements * sizeof(float);
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

    // --- Step 1: Copy Z from pre_activation to output buffer ---
    CUDA_CHECK_ERROR(cudaMemcpyAsync(output, pre_activation, total_bytes, cudaMemcpyDeviceToDevice));

    for (int b = 0; b < batch_size; b++) 
    {
        add_vectors_kernel<<<(output_size + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
            output + b * output_size,
            biases, 
            output_size
        );
    }
    
    activationKernel<<<blocksPerGrid, threadsPerBlock>>>(
        output,  
        output,       
        batch_size * output_size, 
        activation_type
    );
    
    CUDA_CHECK_ERROR(cudaGetLastError());
}

extern "C" void conv2d_forward(const float* input, const float* kernels, const float* bias,
                       float* pre_activation_output,
                       float* output, int batch_size, int in_channels, int out_channels,
                       int input_height, int input_width, int kernel_size, int stride, int padding,int activation_type) 
    {
        int out_height = (input_height + 2 * padding - kernel_size) / stride + 1;
        int out_width = (input_width + 2 * padding - kernel_size) / stride + 1;

        dim3 blockDim(16, 16);

        int grid_x = (out_width + blockDim.x - 1) / blockDim.x;
        int grid_y = (out_height + blockDim.y - 1) / blockDim.y;

        grid_x = grid_x > 0 ? grid_x : 0;
        grid_y = grid_y > 0 ? grid_y : 0;
        dim3 gridDim(grid_x,grid_y,batch_size * out_channels);
        conv2d_forward_kernel<<<gridDim, blockDim>>>(input, kernels, bias, pre_activation_output,output, batch_size, in_channels, out_channels, input_height, input_width, kernel_size, stride, padding,activation_type);

    }

extern "C" void conv2d_backward(
    const float* input,
    const float* kernels,
    float* pre_activation_output,
    float* output_grad,
    float* kernel_grad,
    float* bias_grad,
    float* input_grad,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_size,
    int stride,
    int padding,
    int activation_type
)
{
    int out_w = (input_width + 2 * padding - kernel_size) / stride + 1;
    int out_h = (input_height + 2 * padding - kernel_size) / stride + 1;

    
    dim3 blockDim(16, 16, 1);
    int grid_x = (out_w + blockDim.x - 1) / blockDim.x;
    int grid_y = (out_h + blockDim.y - 1) / blockDim.y;
    int grid_z = batch_size * out_channels;
    
    // Ensure grid dimensions are at least 1
    grid_x = max(1, grid_x);
    grid_y = max(1, grid_y);
    grid_z = max(1, grid_z);

    dim3 gridDim(grid_x, grid_y, grid_z);

    // --- Calculate Shared Memory Size ---
    size_t kernel_grad_size_per_channel = (size_t)in_channels * kernel_size * kernel_size;
 
    size_t shared_mem_size = (out_channels + kernel_grad_size_per_channel) * sizeof(float);

    cudaMemset(kernel_grad, 0, (size_t)out_channels * in_channels * kernel_size * kernel_size * sizeof(float));
    cudaMemset(bias_grad, 0, (size_t)out_channels * sizeof(float));


    conv2d_backward_kernel<<<gridDim, blockDim, shared_mem_size>>>(
        input,
        kernels,
        pre_activation_output,
        output_grad,
        kernel_grad,
        bias_grad,
        nullptr, // Pass nullptr for input_grad as it's not calculated here
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding,
        activation_type
    );
    CUDA_CHECK_ERROR(cudaGetLastError());

}

extern "C" void launchActivation(float *d_input, float *d_output, int size, const char* activation_type) 
{
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    if (strcmp(activation_type, "sigmoid") == 0)
    {
        activationKernel<<<numBlocks, blockSize>>>(d_input, d_output, size, 0);
    } 
    else if (strcmp(activation_type, "relu") == 0)
    {
        activationKernel<<<numBlocks, blockSize>>>(d_input, d_output, size, 1);
    }
    else if (strcmp(activation_type, "leakyrelu") == 0)
    {
        activationKernel<<<numBlocks, blockSize>>>(d_input, d_output, size, 2);
    } 
    else if (strcmp(activation_type, "softmax") == 0)
    {
        softmax_kernel<<<1, blockSize>>>(d_input, d_output, size);
    }
    else if (strcmp(activation_type, "none") == 0)
    {
        // No activation function, just copy input to output
        cudaMemcpy(d_output, d_input, size * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    
    else
    {
        throw std::invalid_argument("Unsupported activation type");
        return;
    }

    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s activation: %s\n", activation_type, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

extern "C" void conv2d_compute_hidden_error(
    const float* weights,
    const float* output_grad,           // Input: dL/dY
    const float* pre_activation_output, // Input: Z <--- ADDED
    float* hidden_error,                // Output: dL/dX
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int output_height,     
    int output_width,         
    int kernel_size,
    int stride,
    int padding,
    int activation_type          
) {

    dim3 blockDim(16, 16, 1); 
 
    int grid_x = (input_width + blockDim.x - 1) / blockDim.x;
    int grid_y = (input_height + blockDim.y - 1) / blockDim.y;
    int grid_z = batch_size * in_channels;

    grid_x = max(1, grid_x);
    grid_y = max(1, grid_y);
    grid_z = max(1, grid_z);

    dim3 gridDim(grid_x, grid_y, grid_z);

    size_t hidden_error_size_bytes = (size_t)batch_size * in_channels * input_height * input_width * sizeof(float);
    cudaMemset(hidden_error, 0, hidden_error_size_bytes);

    conv2d_compute_hidden_error_kernel<<<gridDim, blockDim>>>(
        weights,
        output_grad,
        pre_activation_output,
        hidden_error,
        batch_size,
        in_channels, out_channels,
        input_height, input_width,
        output_height, output_width,
        kernel_size, stride, padding,
        activation_type      
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
        fprintf(stderr, "Failed to launch conv2d_compute_hidden_error_kernel (error code %s)!\n", cudaGetErrorString(err));
    }
}
extern "C" void mse_derivative(const float* output, const float* target, float* out_error, int size)
{
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    mse_derivative_kernel<<<numBlocks, blockSize>>>(output, target, out_error, size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
        fprintf(stderr, "Failed to launch mse_derivative_kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

extern "C" void conv2d_update_params(
    float* weights,
    float* biases,
    const float* weight_grad,
    const float* bias_grad,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    float learning_rate
) {
    int total_weights = out_channels * in_channels * kernel_size * kernel_size;

    dim3 blockDim(16, 16); 
    int blockSize = blockDim.x * blockDim.y;

    int weightGridX = (total_weights + blockSize - 1) / blockSize;
    dim3 gridDim_weight(weightGridX, 1, 1);

    conv2d_update_params_kernel<<<gridDim_weight, blockDim>>>(
        weights, biases, weight_grad, bias_grad,
        batch_size, in_channels, out_channels, kernel_size, stride, padding,
        learning_rate
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
        fprintf(stderr, "Failed to launch conv2d_update_params_kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

}

extern "C" void unflatten_gradient(const float* flattened_grad,
    float* unflattened_grad,   
    int batch_size,
    int channels,
    int height,
    int width
)
{
    size_t total_elements = batch_size * channels * height * width;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    unflatten_gradient_kernel<<<blocks, threads>>>(
        flattened_grad,
        unflattened_grad,
        batch_size,
        channels,
        height,
        width
    );
}


//necessary because templates are cringe amd skill issue on my part.
namespace autodiff { namespace loss {
    // Add __attribute__((weak)) to make this a weak symbol 
    __attribute__((weak)) __device__ __host__ auto CustomLoss::expression() {
        Output o;
        Target t;
        // Default is MSE
        return square(o - t);
    }
}}