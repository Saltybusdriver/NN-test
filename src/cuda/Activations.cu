#ifndef ACTIVATIONS_CU_INCLUDED
#define ACTIVATIONS_CU_INCLUDED


#include <cmath>
#include <cuda_runtime.h>
#include "cuda_functions.h"
#include <cstdio>
#include <cstdlib>

#include <cooperative_groups.h>

#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include "Reduction_Helpers.cu"


// ------------------------------
// Activations
// ------------------------------
static __device__ __forceinline__ float Sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
}
static __device__ __forceinline__ float ReLU(float x) {
    return fmaxf(0.0f, x);
}
static __device__ __forceinline__ float LeakyReLU(float x, float negative_slope) {
    return fmaxf(x * negative_slope, x);
}
static __device__ __forceinline__ float SigmoidDerivative(float x) {
    float s = Sigmoid(x);
    return s * (1.0f - s);
}
static __device__ __forceinline__ float ReLUDerivative(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}
static __device__ __forceinline__ float LeakyReLUDerivative(float x, float negative_slope) {
    return x > 0.0f ? 1.0f : negative_slope;
}

static __device__ __forceinline__ float activationFunction(float input, int type) {
    switch (type) {
        case 0: return Sigmoid(input);
        case 1: return ReLU(input);
        case 2: return LeakyReLU(input, 0.01f);
        default: return input;
    }
}
static __device__ __forceinline__ float activationFunctionDerivative(float input, int type) {
    switch (type) {
        case 0: return SigmoidDerivative(input);
        case 1: return ReLUDerivative(input);
        case 2: return LeakyReLUDerivative(input, 0.01f);
        default: return 1.0f;
    }
}

static __global__ void activationKernel(float* input, float* output, int size, int activation_type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        switch (activation_type) {
            case 0: output[idx] = Sigmoid(input[idx]); break;
            case 1: output[idx] = ReLU(input[idx]); break;
            case 2: output[idx] = LeakyReLU(input[idx], 0.01f); break;
            default: output[idx] = input[idx]; break;
        }
    }
}

static __device__ void SoftmaxKernel(float* input, float* output, int size) {
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        local_max = fmaxf(local_max, input[i]);
    }
    float maxv = blockReduceMax<256>(local_max);
    __syncthreads();
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        float e = __expf(input[i] - maxv);
        output[i] = e;
        local_sum += e;
    }
    float sumv = blockReduceSum<256>(local_sum);
    __syncthreads();
    float inv = 1.0f / (sumv + 1e-9f);
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        output[i] *= inv;
    }
}



static __global__ void softmax_kernel(const float* input, float* output, int size) {
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        local_max = fmaxf(local_max, input[i]);
    }
    float maxv = blockReduceMax<256>(local_max);
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        float e = __expf(input[i] - maxv);
        output[i] = e;
        local_sum += e;
    }
    float sumv = blockReduceSum<256>(local_sum);
    __syncthreads();

    float inv = 1.0f / (sumv + 1e-9f);
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        output[i] *= inv;
    }
}

static __global__ void softmax_rows_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int output_size)
{
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const float* row_input = input + (size_t)row * output_size;
    float* row_output = output + (size_t)row * output_size;

    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < output_size; i += blockDim.x) {
        local_max = fmaxf(local_max, row_input[i]);
    }
    float max_value = blockReduceMax<256>(local_max);
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < output_size; i += blockDim.x) {
        float value = __expf(row_input[i] - max_value);
        row_output[i] = value;
        local_sum += value;
    }
    float sum_value = blockReduceSum<256>(local_sum);
    __syncthreads();

    float inv_sum = 1.0f / (sum_value + 1e-9f);
    for (int i = threadIdx.x; i < output_size; i += blockDim.x) {
        row_output[i] *= inv_sum;
    }
}

static __global__ void activation_derivative_kernel(
    float* __restrict__ output_error,
    const float* __restrict__ pre_activation_output,
    int total,
    int activation_type)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        float deriv = activationFunctionDerivative(pre_activation_output[idx], activation_type);
        output_error[idx] *= deriv;
    }
}

static __global__ void softmax_rows_backward_kernel(
    float* __restrict__ output_error,
    const float* __restrict__ pre_activation_output,
    int batch_size,
    int output_size)
{
    int row = blockIdx.x;
    if (row >= batch_size) return;

    const float* row_logits = pre_activation_output + (size_t)row * output_size;
    float* row_error = output_error + (size_t)row * output_size;

    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < output_size; i += blockDim.x) {
        local_max = fmaxf(local_max, row_logits[i]);
    }
    float max_value = blockReduceMax<256>(local_max);
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < output_size; i += blockDim.x) {
        local_sum += __expf(row_logits[i] - max_value);
    }
    float sum_value = blockReduceSum<256>(local_sum);
    __syncthreads();

    float inv_sum = 1.0f / (sum_value + 1e-9f);
    float local_dot = 0.0f;
    for (int i = threadIdx.x; i < output_size; i += blockDim.x) {
        float p = __expf(row_logits[i] - max_value) * inv_sum;
        local_dot += row_error[i] * p;
    }
    float dot = blockReduceSum<256>(local_dot);
    __syncthreads();

    for (int i = threadIdx.x; i < output_size; i += blockDim.x) {
        float p = __expf(row_logits[i] - max_value) * inv_sum;
        row_error[i] = p * (row_error[i] - dot);
    }
}

static inline void applyActivationForward(
    float* pre_activation_output,
    float* output,
    int batch_size,
    int output_size,
    int activation_type)
{
    int total = batch_size * output_size;
    int blocks = (total + 255) / 256;
    if (activation_type == 3) {
        softmax_rows_kernel<<<batch_size, 256>>>(pre_activation_output, output, batch_size, output_size);
    } else {
        activationKernel<<<blocks, 256>>>(pre_activation_output, output, total, activation_type);
    }
}

static inline void applyActivationBackward(
    float* output_error,
    const float* pre_activation_output,
    int batch_size,
    int output_size,
    int activation_type)
{
    int total = batch_size * output_size;
    int blocks = (total + 255) / 256;
    if (activation_type == 3) {
        softmax_rows_backward_kernel<<<batch_size, 256>>>(output_error, pre_activation_output, batch_size, output_size);
    } else {
        activation_derivative_kernel<<<blocks, 256>>>(output_error, pre_activation_output, total, activation_type);
    }
}

#endif 
