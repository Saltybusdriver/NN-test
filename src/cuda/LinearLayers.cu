#include <cmath>
#include <cuda_runtime.h>
#include "cuda_functions.h"
#include <cstdio>
#include <cstdlib>

#include <cooperative_groups.h>
#include <algorithm>
#include "helpers.cu"
#include "Activations.cu"
#include "debug.h"

extern "C" void compute_hidden_error(
    float* weights, float* output_error, float* hidden_error,
    int batch_size, int input_size, int output_size) 
{
    // dX = dZ * W^T
    // output_error: (batch_size x output_size)
    // weights: transposed (output_size x input_size) to match shapes
    custom_gemm(batch_size, input_size, output_size, 
                output_error, weights, hidden_error, 
                false, true);
}

static __global__ void compute_bias_gradients_kernel(
    const float* __restrict__ output_error, float* __restrict__ bias_grad,
    int batch_size, int output_size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_size) return;

    float grad = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
        grad += output_error[(size_t)b * output_size + idx];
    }
    bias_grad[idx] = grad / batch_size;
}

extern "C" void backward_propagate(
    float* input,
    float* weights,
    float* output_error,
    float* weight_grad,
    float* bias_grad,
    int batch_size,
    int input_size,
    int output_size) 
{
    // Weight gradients: dW = X^T * dZ
    // X is transposed to (input_size x batch_size)
    custom_gemm(input_size, output_size, batch_size,
                input, output_error, weight_grad,
                true, false);
    
    // Bias gradients
    int bias_blocks = (output_size + 255) / 256;
    compute_bias_gradients_kernel<<<bias_blocks, 256>>>(output_error, bias_grad, batch_size, output_size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
        printf("CUDA Error in Linear backward: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void forward_Propagate(
    const float* input,
    float* weight,
    float* biases,
    float* pre_activation_output,
    float* output,
    
    int batch_size,
    int input_size,
    int output_size,
    int activation_type) 
{
    // Forward: Z = X * W
    // X: (batch_size x input_size)
    // W: (input_size x output_size)
    custom_gemm(batch_size, output_size, input_size,
                input, weight, pre_activation_output,
                false, false);

    // Add bias
    int total_tasks = batch_size * output_size;
    int blocks = (total_tasks + 255) / 256;
    batched_add_bias_kernel<<<blocks, 256>>>(pre_activation_output, biases, batch_size, output_size);

    applyActivationForward(pre_activation_output, output, batch_size, output_size, activation_type);
}

extern "C" void apply_linear_activation_derivative(
    float* output_error, const float* pre_activation_output, int batch_size, int out_features, int activation_type) 
{
    applyActivationBackward(output_error, pre_activation_output, batch_size, out_features, activation_type);
}
