
#include <cmath>
#include <cuda_runtime.h>
#include "cuda_functions.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "debug.h"
#include <cooperative_groups.h>
#include <autodiff.h>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include "Activations.cu"
#include "Reduction_Helpers.cu"

static bool is_cross_entropy_loss(const char* loss_type) {
    return loss_type != nullptr &&
           (strcmp(loss_type, "cross_entropy") == 0 ||
            strcmp(loss_type, "crossentropy") == 0 ||
            strcmp(loss_type, "cross-entropy") == 0);
}

static __global__ void mse_derivative_kernel(const float* output, const float* target, float* out_error, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out_error[idx] = (2.0f * (output[idx] - target[idx])) / size;
    }
}

static __global__ void mse_loss_kernel(float* output, float* target, float* loss, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        __shared__ float shared_loss[256];
        float local_loss = 0.0f;

        float diff = output[idx] - target[idx];
        local_loss = (diff * diff) / size;

        shared_loss[threadIdx.x] = local_loss;
        __syncthreads();

        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) shared_loss[threadIdx.x] += shared_loss[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) atomicAdd(loss, shared_loss[0]);
    }
}

template <typename LossFunction>
static __global__ void evaluate_loss_kernel(
    float* output, const float* target, float* temp_loss, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        auto loss_expr = LossFunction::expression();
        temp_loss[idx] = loss_expr.eval(output[idx], target[idx]);
    }
}

template <typename LossFunction>
static __global__ void compute_loss_error_kernel(
    float* output, const float* target, float* error, int batch_size, int output_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * output_size;
    if (idx < total) {
        auto loss_expr = LossFunction::expression();
        float divisor = autodiff::loss::GradientDivisor<LossFunction>::value(batch_size, output_size);
        float grad = loss_expr.grad(output[idx], target[idx]) / divisor;
        error[idx] = grad;
    }
}

static __global__ void sum_reduce_kernel(const float* in, float* out, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    
    float local = 0.0f;
    for (int i = tid; i < n; i += blockDim.x)
        local += in[i];
    
    smem[tid] = local;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    
    if (tid == 0) atomicAdd(out, smem[0]);
}

extern "C" void calculate_loss_values(
    float* output,
    const float* target,
    float* element_loss,
    int size,
    const char* loss_type = "mse",
    int batch_size = 1
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    if (is_cross_entropy_loss(loss_type)) {
        evaluate_loss_kernel<autodiff::loss::CrossEntropyLoss><<<blocks, threads>>>(output, target, element_loss, size);
    } else if (strcmp(loss_type, "mse") == 0) {
        evaluate_loss_kernel<autodiff::loss::MSELoss><<<blocks, threads>>>(output, target, element_loss, size);
    } else if (strcmp(loss_type, "bce") == 0) {
        evaluate_loss_kernel<autodiff::loss::BCELoss><<<blocks, threads>>>(output, target, element_loss, size);
    } else if (strcmp(loss_type, "l1") == 0) {
        evaluate_loss_kernel<autodiff::loss::L1Loss><<<blocks, threads>>>(output, target, element_loss, size);
    } else if (strcmp(loss_type, "custom") == 0) {
        evaluate_loss_kernel<autodiff::loss::CustomLoss><<<blocks, threads>>>(output, target, element_loss, size);
    } else if (strcmp(loss_type, "huber") == 0) {
        evaluate_loss_kernel<autodiff::loss::HuberLoss><<<blocks, threads>>>(output, target, element_loss, size);
    } else {
        evaluate_loss_kernel<autodiff::loss::MSELoss><<<blocks, threads>>>(output, target, element_loss, size);
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
) {
    int threads = 256;
    int blocks = (batch_size * output_size + threads - 1) / threads;

    if (is_cross_entropy_loss(loss_type)) {
        compute_loss_error_kernel<autodiff::loss::CrossEntropyLoss><<<blocks, threads>>>(output, target, error, batch_size, output_size);
    }
    else if (strcmp(loss_type, "mse") == 0) {
        compute_loss_error_kernel<autodiff::loss::MSELoss><<<blocks, threads>>>(output, target, error, batch_size, output_size);
    } else if (strcmp(loss_type, "bce") == 0) {
        compute_loss_error_kernel<autodiff::loss::BCELoss><<<blocks, threads>>>(output, target, error, batch_size, output_size);
    } else if (strcmp(loss_type, "l1") == 0) {
        compute_loss_error_kernel<autodiff::loss::L1Loss><<<blocks, threads>>>(output, target, error, batch_size, output_size);
    } else if (strcmp(loss_type, "custom") == 0) {
        compute_loss_error_kernel<autodiff::loss::CustomLoss><<<blocks, threads>>>(output, target, error, batch_size, output_size);
    } else {
        compute_loss_error_kernel<autodiff::loss::MSELoss><<<blocks, threads>>>(output, target, error, batch_size, output_size);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in gradient calc: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void calc_mse_loss_kernel(float* output, float* target, float* loss, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    mse_loss_kernel<<<numBlocks, blockSize>>>(output, target, loss, size);
    CUDA_CHECK_ERROR(cudaGetLastError());
}

extern "C" void mse_derivative(const float* output, const float* target, float* out_error, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    mse_derivative_kernel<<<numBlocks, blockSize>>>(output, target, out_error, size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch mse_derivative_kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

namespace autodiff { namespace loss {
    __attribute__((weak)) __device__ __host__ auto CustomLoss::expression() {
        Output o;
        Target t;
        return square(o - t);
    }
}}


static __global__ void sum_loss_kernel(const float* data, float* out_sum, int num_elements) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    smem[tid] = (gid < num_elements) ? data[gid] : 0.0f;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(out_sum, smem[0]);
    }
}

static __global__ void accumulate_loss_sum_kernel(
    const float* data,
    float* primary_sum,
    float* secondary_sum,
    int num_elements
) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    smem[tid] = (gid < num_elements) ? data[gid] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (primary_sum) {
            atomicAdd(primary_sum, smem[0]);
        }
        if (secondary_sum) {
            atomicAdd(secondary_sum, smem[0]);
        }
    }
}

extern "C" float sum_loss_cuda(const float* loss_buffer, int num_elements) {
    static float* d_out_sum = nullptr;
    if (num_elements <= 0) return 0.0f;

    if (d_out_sum == nullptr) {
        cudaMalloc(&d_out_sum, sizeof(float));
    }
    cudaMemset(d_out_sum, 0, sizeof(float));
    
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    sum_loss_kernel<<<blocks, threads>>>(loss_buffer, d_out_sum, num_elements);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in loss sum: %s\n", cudaGetErrorString(err));
        return 0.0f;
    }
    
    float h_sum = 0.0f;
    cudaMemcpy(&h_sum, d_out_sum, sizeof(float), cudaMemcpyDeviceToHost);
    return h_sum;
}

static __global__ void calculate_accuracy_kernel(const float* predictions, const float* targets, int* correct_count, int batch_size, int num_classes) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size) {
        const float* pred_row = predictions + b * num_classes;
        const float* target_row = targets + b * num_classes;
        
        float max_pred = -1e9f;
        int max_pred_idx = -1;
        float max_target = -1e9f;
        int max_target_idx = -1;
        
        for (int i = 0; i < num_classes; ++i) {
            if (pred_row[i] > max_pred) {
                max_pred = pred_row[i];
                max_pred_idx = i;
            }
            if (target_row[i] > max_target) {
                max_target = target_row[i];
                max_target_idx = i;
            }
        }
        
        if (max_pred_idx == max_target_idx) {
            atomicAdd(correct_count, 1);
        }
    }
}

static __global__ void accumulate_correct_count_kernel(
    const float* predictions,
    const float* targets,
    int* primary_count,
    int* secondary_count,
    int batch_size,
    int num_classes
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size) {
        const float* pred_row = predictions + b * num_classes;
        const float* target_row = targets + b * num_classes;

        float max_pred = -1e9f;
        int max_pred_idx = -1;
        float max_target = -1e9f;
        int max_target_idx = -1;

        for (int i = 0; i < num_classes; ++i) {
            if (pred_row[i] > max_pred) {
                max_pred = pred_row[i];
                max_pred_idx = i;
            }
            if (target_row[i] > max_target) {
                max_target = target_row[i];
                max_target_idx = i;
            }
        }

        if (max_pred_idx == max_target_idx) {
            if (primary_count) {
                atomicAdd(primary_count, 1);
            }
            if (secondary_count) {
                atomicAdd(secondary_count, 1);
            }
        }
    }
}

extern "C" float calculate_accuracy_cuda(const float* predictions, const float* targets, int batch_size, int num_classes) {
    static int* d_correct_count = nullptr;
    if (batch_size <= 0 || num_classes <= 0) return 0.0f;

    if (d_correct_count == nullptr) {
        cudaMalloc(&d_correct_count, sizeof(int));
    }
    cudaMemset(d_correct_count, 0, sizeof(int));
    
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    calculate_accuracy_kernel<<<blocks, threads>>>(predictions, targets, d_correct_count, batch_size, num_classes);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in accuracy calculation: %s\n", cudaGetErrorString(err));
        return 0.0f;
    }
    
    int h_correct_count = 0;
    cudaMemcpy(&h_correct_count, d_correct_count, sizeof(int), cudaMemcpyDeviceToHost);
    
    return (float)h_correct_count / (float)batch_size;
}

extern "C" void accumulate_loss_sum_cuda(
    const float* loss_buffer,
    float* primary_sum,
    float* secondary_sum,
    int num_elements
) {
    if (num_elements <= 0 || (!primary_sum && !secondary_sum)) return;

    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    accumulate_loss_sum_kernel<<<blocks, threads>>>(loss_buffer, primary_sum, secondary_sum, num_elements);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in loss accumulation: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void accumulate_correct_count_cuda(
    const float* predictions,
    const float* targets,
    int* primary_count,
    int* secondary_count,
    int batch_size,
    int num_classes
) {
    if (batch_size <= 0 || num_classes <= 0 || (!primary_count && !secondary_count)) return;

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    accumulate_correct_count_kernel<<<blocks, threads>>>(
        predictions, targets, primary_count, secondary_count, batch_size, num_classes
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in accuracy accumulation: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void sum_loss(const float* d_element_loss, float* d_out, int num_elements) {
    cudaMemset(d_out, 0, sizeof(float));
    int blocks = (num_elements + 255) / 256;
    sum_reduce_kernel<<<blocks, 256>>>(d_element_loss, d_out, num_elements);
}
