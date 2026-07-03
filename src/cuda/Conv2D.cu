#include <cmath>
#include <cuda_runtime.h>
#include "cuda_functions.h"
#include <cstdio>
#include <cstdlib>
#include "debug.h"
#include <cooperative_groups.h>
#include <algorithm>
#include "Activations.cu"   
#include "helpers.cu"

#define BM 64
#define BN 64
#define BK 8
#define TM 8
#define TN 8

constexpr int DW_BM = 64;
constexpr int DW_BN = 64;
constexpr int DW_BK = 8;
constexpr int DW_TM = 4;
constexpr int DW_TN = 2;

// Computes Convolution Forward implicitly (No im2col), 4 floats at a time
static __global__ void implicit_gemm_conv2d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernels,
    const float* __restrict__ bias,
    float* __restrict__ pre_activation_output,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int H, int W, int out_H, int out_W,
    int K_H, int K_W, int stride, int pad, int activation_type)
{
    int b = blockIdx.z;
    if (b >= batch_size) return;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tid = ty * blockDim.x + tx;
    int num_threads = blockDim.x * blockDim.y;

    int M = out_channels;
    int N = out_H * out_W;
    int K = in_channels * K_H * K_W;

    int row = by * BM + ty * TM;
    int col = bx * BN + tx * TN;

    float accum[TM][TN] = {0.0f};

    const float* batch_input = input + b * (in_channels * H * W);
    float* batch_preact = pre_activation_output + b * (out_channels * N);
    float* batch_output = output + b * (out_channels * N);

    int A_total = BM * BK;
    int B_total = BK * BN;

    int A_per_thread = (A_total + num_threads - 1) / num_threads;
    int B_per_thread = (B_total + num_threads - 1) / num_threads;

    for (int k_tile = 0; k_tile < (K + BK - 1) / BK; ++k_tile)
    {
        // =========================
        // LOAD KERNELS (A MATRIX)
        // =========================
        for (int step = 0; step < A_per_thread; ++step)
        {
            int idx = step * num_threads + tid;
            if (idx < A_total)
            {
                int r = idx / BK;
                int c = idx % BK;

                int global_r = by * BM + r;
                int global_c = k_tile * BK + c;

                As[r * BK + c] =
                    (global_r < M && global_c < K)
                    ? kernels[global_r * K + global_c]
                    : 0.0f;
            }
        }

        // =========================
        // LOAD INPUT IM2COL (B MATRIX)
        // =========================
        for (int step = 0; step < B_per_thread; ++step)
        {
            int idx = step * num_threads + tid;
            if (idx < B_total)
            {
                int r = idx / BN;
                int c = idx % BN;

                int k_row = k_tile * BK + r;
                int n_col = bx * BN + c;

                float val = 0.0f;

                if (k_row < K && n_col < N)
                {
                    int c_in  = k_row / (K_H * K_W);
                    int k_idx = k_row % (K_H * K_W);
                    int k_h   = k_idx / K_W;
                    int k_w   = k_idx % K_W;

                    int oh = n_col / out_W;
                    int ow = n_col % out_W;

                    int ih = oh * stride - pad + k_h;
                    int iw = ow * stride - pad + k_w;

                    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                    {
                        val = batch_input[(c_in * H + ih) * W + iw];
                    }
                }

                Bs[r * BN + c] = val;
            }
        }

        __syncthreads();

        // =========================
        // COMPUTE
        // =========================
        for (int k = 0; k < BK; ++k)
        {
            #pragma unroll
            for (int i = 0; i < TM; ++i)
            {
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                {
                    accum[i][j] +=
                        As[(ty * TM + i) * BK + k] *
                        Bs[k * BN + (tx * TN + j)];
                }
            }
        }

        __syncthreads();
    }

    // =========================
    // WRITE OUTPUT (SAFE)
    // =========================
    for (int i = 0; i < TM; ++i)
    {
        int global_row = row + i;
        if (global_row >= M) continue;

        for (int j = 0; j < TN; j += 4)
        {
            int global_col = col + j;
            if (global_col >= N) continue;

            size_t out_idx = (size_t)global_row * N + global_col;

            float bias_val = bias[global_row];

            if (global_col + 3 < N && ((out_idx & 3) == 0))
            {
                float s0 = accum[i][j] + bias_val;
                float s1 = accum[i][j + 1] + bias_val;
                float s2 = accum[i][j + 2] + bias_val;
                float s3 = accum[i][j + 3] + bias_val;

                float4 pre = make_float4(s0, s1, s2, s3);
                float4 activated = make_float4(
                    activationFunction(s0, activation_type),
                    activationFunction(s1, activation_type),
                    activationFunction(s2, activation_type),
                    activationFunction(s3, activation_type));

                reinterpret_cast<float4*>(&batch_preact[out_idx])[0] = pre;
                reinterpret_cast<float4*>(&batch_output[out_idx])[0] = activated;
            }
            else
            {
                for (int v = 0; v < 4; ++v)
                {
                    if (global_col + v < N)
                    {
                        float pre = accum[i][j + v] + bias_val;
                        batch_preact[global_row * N + global_col + v] = pre;
                        batch_output[global_row * N + global_col + v] =
                            activationFunction(pre, activation_type);
                    }
                }
            }
        }
    }
}

static __global__ void implicit_gemm_conv2d_backward_weights_kernel(
    const float* __restrict__ input,       // [batch_size, in_channels, H, W]
    const float* __restrict__ delta,       // [batch_size, out_channels, out_H, out_W]
    float* __restrict__ kernel_grad,       // [out_channels, in_channels * KH * KW]
    int batch_size, int in_channels, int out_channels,
    int H, int W, int out_H, int out_W,
    int K_H, int K_W, int stride, int pad) 
{
    // Accumulate the gradients per kernel weight across the entire batch
    int m_idx = blockIdx.y * blockDim.y + threadIdx.y; // out_channels (M)
    int k_idx = blockIdx.x * blockDim.x + threadIdx.x; // in_channels * KH * KW (K)

    if (m_idx < out_channels && k_idx < (in_channels * K_H * K_W)) {
        
        int in_c = k_idx / (K_H * K_W);
        int rem  = k_idx % (K_H * K_W);
        int kh   = rem / K_W;
        int kw   = rem % K_W;

        float grad_sum = 0.0f;

        // Loop over the batch and spatial dimensions
        for (int b = 0; b < batch_size; ++b) {
            for (int oh = 0; oh < out_H; ++oh) {
                for (int ow = 0; ow < out_W; ++ow) {
                    int ih = oh * stride - pad + kh;
                    int iw = ow * stride - pad + kw;

                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        float d_val = delta[b * (out_channels * out_H * out_W) + m_idx * (out_H * out_W) + oh * out_W + ow];
                        float i_val = input[b * (in_channels * H * W) + in_c * (H * W) + ih * W + iw];
                        grad_sum += d_val * i_val;
                    }
                }
            }
        }
        // Write out the calculated gradient
        atomicAdd(&kernel_grad[m_idx * (in_channels * K_H * K_W) + k_idx], grad_sum);
    }
}

// Tiled implicit GEMM for convolution weight gradients.
// The old path did:
//   im2col(input) -> custom_gemm(delta, im2col^T) -> reduce across batch
// That is correct, but report48 showed it burns time writing a huge im2col buffer
// and then immediately reading it back. This kernel keeps the same math and the
// same batch-reduce structure, but generates each im2col value directly while
// loading the B tile. Same handheld CUDA style, less memory traffic, fewer
// "why did we write the whole damn matrix just to read it again" moments.
static __global__ void implicit_conv2d_weight_grad_tiled_kernel(
    const float* __restrict__ input,
    const float* __restrict__ delta,
    float* __restrict__ workspace,
    int batch_size, int in_channels, int out_channels,
    int H, int W, int out_H, int out_W,
    int K_H, int K_W, int stride, int pad)
{
    __shared__ float As[DW_BM * DW_BK];
    __shared__ float Bs[DW_BK * DW_BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int b = blockIdx.z;
    if (b >= batch_size) return;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    int M = out_channels;
    int N = out_H * out_W;
    int K = in_channels * K_H * K_W;

    int row = by * DW_BM + ty * DW_TM;
    int col = bx * DW_BN + tx * DW_TN;

    float acc[DW_TM][DW_TN] = {0.0f};

    const float* batch_delta = delta + (size_t)b * M * N;
    const float* batch_input = input + (size_t)b * in_channels * H * W;

    for (int n0 = 0; n0 < N; n0 += DW_BK) {
        int a_row_local = tid / DW_BK;
        int a_col_local = tid % DW_BK;
        int a_row = by * DW_BM + a_row_local;
        int a_col = n0 + a_col_local;

        if (a_row < M && a_col < N) {
            As[a_row_local * DW_BK + a_col_local] = batch_delta[(size_t)a_row * N + a_col];
        } else {
            As[a_row_local * DW_BK + a_col_local] = 0.0f;
        }

        int b_row_local = tid / DW_BN;
        int b_col_local = tid % DW_BN;
        int n_col = n0 + b_row_local;
        int k_col = bx * DW_BN + b_col_local;

        float input_val = 0.0f;
        if (n_col < N && k_col < K) {
            int out_y = n_col / out_W;
            int out_x = n_col - out_y * out_W;

            int c_in = k_col / (K_H * K_W);
            int kernel_idx = k_col - c_in * K_H * K_W;
            int k_y = kernel_idx / K_W;
            int k_x = kernel_idx - k_y * K_W;

            int in_y = out_y * stride - pad + k_y;
            int in_x = out_x * stride - pad + k_x;

            if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                input_val = batch_input[((size_t)c_in * H + in_y) * W + in_x];
            }
        }
        Bs[b_row_local * DW_BN + b_col_local] = input_val;

        __syncthreads();

        #pragma unroll
        for (int n = 0; n < DW_BK; ++n) {
            #pragma unroll
            for (int i = 0; i < DW_TM; ++i) {
                float a = As[(ty * DW_TM + i) * DW_BK + n];
                #pragma unroll
                for (int j = 0; j < DW_TN; ++j) {
                    acc[i][j] += a * Bs[n * DW_BN + (tx * DW_TN + j)];
                }
            }
        }

        __syncthreads();
    }

    size_t partial_stride = (size_t)M * K;
    float* batch_workspace = workspace + (size_t)b * partial_stride;

    #pragma unroll
    for (int i = 0; i < DW_TM; ++i) {
        int out_m = row + i;
        if (out_m >= M) continue;
        #pragma unroll
        for (int j = 0; j < DW_TN; ++j) {
            int out_k = col + j;
            if (out_k < K) {
                batch_workspace[(size_t)out_m * K + out_k] = acc[i][j];
            }
        }
    }
}

static __global__ void conv2d_update_params_kernel(
    float* __restrict__ weights, float* __restrict__ biases,
    const float* __restrict__ weight_grad, const float* __restrict__ bias_grad,
    int batch_size, int in_channels, int out_channels,
    int kernel_size, int stride, int padding, float learning_rate) 
{
    int lin = blockIdx.x * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
    int total_w = out_channels * in_channels * kernel_size * kernel_size;

    if (lin < total_w) {
        float g = weight_grad[lin] / batch_size;
        weights[lin] -= learning_rate * g;
    }
    if (lin < out_channels) {
        float gb = bias_grad[lin] / batch_size;
        biases[lin] -= learning_rate * gb;
    }
}

static __global__ void conv2d_compute_delta_kernel(
    const float* __restrict__ output_error,
    const float* __restrict__ pre_activation_output,
    float* __restrict__ delta, int total, int activation_type) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        float deriv = activationFunctionDerivative(pre_activation_output[idx], activation_type);
        delta[idx] = output_error[idx] * deriv;
    }
}

static __global__ void compute_conv_bias_grads_kernel(const float* delta, float* bias_grad,
                                 int B, int C, int H, int W) {
    int oc = blockIdx.x; // Current channel (C)
    float sum = 0.0f;

    // Correctly stride across batches and spatial dimensions
    for (int b = 0; b < B; ++b) {
        // Offset straight to [batch, channel, 0]
        size_t channel_offset = (size_t)b * (C * H * W) + (size_t)oc * (H * W);
        
        for (int s = threadIdx.x; s < H * W; s += blockDim.x) {
            sum += delta[channel_offset + s];
        }
    }

    __shared__ float smem[256];
    smem[threadIdx.x] = sum;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        bias_grad[oc] = smem[0];
}

// Bias addition for NCHW layout accurately
static __global__ void add_bias_nchw_kernel(
    float* output, const float* biases, int batch_size, int channels, int spatial_size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * spatial_size;
    if (idx < total) {
        int c = (idx / spatial_size) % channels;
        output[idx] += biases[c];
    }
}

extern "C" void conv2d_forward(
    const float* input, const float* kernels, const float* bias,
    float* pre_activation_output, float* output, float* d_im2col_cache, // Note: not used anymore!
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width, int kernel_size, int stride, int padding, int activation_type) 
{
    int out_h = (input_height + 2 * padding - kernel_size) / stride + 1;
    int out_w = (input_width + 2 * padding - kernel_size) / stride + 1;

    int M = out_channels;
    int N = out_h * out_w;

    dim3 blockDim(BN / TN, BM / TM); // 8x8 = 64 threads
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM, batch_size);

    // Call fused convolution directly. The kernel writes both the raw pre-activation
    // tensor needed by backward and the activated output needed by the next layer,
    // so forward no longer pays for separate bias-add and activation tensor passes.
    implicit_gemm_conv2d_forward_kernel<<<gridDim, blockDim>>>(
        input, kernels, bias, pre_activation_output, output,
        batch_size, in_channels, out_channels,
        input_height, input_width, out_h, out_w,
        kernel_size, kernel_size, stride, padding, activation_type);
}

extern "C" void conv2d_backward(
    const float* input, const float* kernels, float* pre_activation_output,
    float* output_grad, float* kernel_grad, size_t kernel_size_bytes,
    float* bias_grad, size_t bias_size, float* input_grad, float* d_im2col_cache,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width, int kernel_size, int stride, int padding,
    int activation_type) 
{
    int out_h = (input_height + 2 * padding - kernel_size) / stride + 1;
    int out_w = (input_width + 2 * padding - kernel_size) / stride + 1;
    int total_output = batch_size * out_channels * out_h * out_w;

    float* delta = output_grad; 
    
    int blocks_delta = (total_output + 255) / 256;
    conv2d_compute_delta_kernel<<<blocks_delta, 256>>>(
        output_grad, pre_activation_output, delta, total_output, activation_type);
    
    int M = out_channels;
    int K = in_channels * kernel_size * kernel_size;

    cudaMemset(kernel_grad, 0, kernel_size_bytes);
    cudaMemset(bias_grad, 0, bias_size);

    size_t required_workspace_size = (size_t)batch_size * M * K * sizeof(float);
    if (required_workspace_size > current_gemm_workspace_size) {
        if (d_global_gemm_workspace) safeCudaFree(&d_global_gemm_workspace, "conv2d weight-grad workspace");
        safeCudaMalloc(&d_global_gemm_workspace, required_workspace_size, "conv2d weight-grad workspace");
        current_gemm_workspace_size = required_workspace_size;
    }

    dim3 weight_grad_block(DW_BN / DW_TN, DW_BM / DW_TM);
    dim3 weight_grad_grid((K + DW_BN - 1) / DW_BN, (M + DW_BM - 1) / DW_BM, batch_size);
    implicit_conv2d_weight_grad_tiled_kernel<<<weight_grad_grid, weight_grad_block>>>(
        input, delta, d_global_gemm_workspace,
        batch_size, in_channels, out_channels,
        input_height, input_width, out_h, out_w,
        kernel_size, kernel_size, stride, padding);

    int reduce_threads = 256;
    int reduce_blocks = ((size_t)M * K + reduce_threads - 1) / reduce_threads;
    reduce_batch_kernel<<<reduce_blocks, reduce_threads>>>(
        d_global_gemm_workspace, kernel_grad, M, K, batch_size, false);

    int blocks_bias = (out_channels + 255) / 256;
    compute_conv_bias_grads_kernel<<<blocks_bias, 256>>>(delta, bias_grad, batch_size, out_channels, out_h, out_w);
}

extern "C" void conv2d_compute_hidden_error(
    const float* weights, const float* delta, const float* pre_activation_output,
    float* hidden_error, size_t input_tensor_size_bytes, float* d_im2col_error_cache,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width, int output_height, int output_width,
    int kernel_size, int stride, int padding, int activation_type,
    float** delta_cache_ptr, size_t* cache_size_ptr) 
{
    int M = out_channels;
    int N = output_height * output_width;
    int K = in_channels * kernel_size * kernel_size;
    cudaMemset(hidden_error, 0, input_tensor_size_bytes);
    custom_gemm(K, N, M, weights, delta, d_im2col_error_cache, 
                true, false, false, batch_size, 0, M * N, K * N);

    col2im(d_im2col_error_cache, batch_size, in_channels, input_height, input_width,
           kernel_size, stride, padding, output_height, output_width, hidden_error);
}

extern "C" void conv2d_update_params(
    float* weights, float* biases, const float* weight_grad, const float* bias_grad,
    int batch_size, int in_channels, int out_channels,
    int kernel_size, int stride, int padding, float learning_rate) 
{
    int total_weights = out_channels * in_channels * kernel_size * kernel_size;
    dim3 block(256);
    dim3 grid((total_weights + 255) / 256);

    conv2d_update_params_kernel<<<grid, block>>>(
        weights, biases, weight_grad, bias_grad,
        batch_size, in_channels, out_channels, kernel_size, stride, padding, learning_rate
    );
}
