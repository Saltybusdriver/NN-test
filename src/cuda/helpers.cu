#ifndef HELPERS_CU_INCLUDED
#define HELPERS_CU_INCLUDED

#include <cmath>
#include <cuda_runtime.h>
#include "cuda_functions.h"
#include <cstdio>
#include <cstdlib>
#include "utils.h"
#include <cooperative_groups.h>
#include <algorithm>
#include "Activations.cu"

// Batch-reduce the temporary per-sample GEMM outputs.
// workspace is laid out as [batch_size][M * N]. Each thread owns one C element
// and loops across the batch dimension to sum that element over all samples.
// If accumulate is true, it adds into out; otherwise it overwrites out. This is
// used by batched gradient GEMMs where each sample produces a tiny matrix and
// the final gradient is the sum. Very 2016 "combine all the things" energy.
static __global__ void reduce_batch_kernel(const float* workspace, float* out, int M, int N, int batch_size, bool accumulate) {
    size_t total_elements = (size_t)M * N;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        float sum = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            sum += workspace[b * total_elements + idx];
        }
        
        if (accumulate) {
            out[idx] += sum;
        } else {
            out[idx] = sum;
        }
    }
}
constexpr int GEMM_BM = 64;
constexpr int GEMM_BN = 64;
constexpr int GEMM_BK = 8;
constexpr int GEMM_TM = 4;
constexpr int GEMM_TN = 2;
constexpr int GEMM_SPLITK_TARGET_TILES = 256;
constexpr int GEMM_SPLITK_MIN_K_PER_SPLIT = 1024;
constexpr int GEMM_SPLITK_MAX_SPLITS = 32;

// Main tiled GEMM workhorse.
// Computes C = A * B for logical shapes:
//   A: M x K, or K x M when TRANS_A is true
//   B: K x N, or N x K when TRANS_B is true
//   C: M x N
// Each block computes a 64x64 output tile. Each thread computes a 4x2 micro-tile,
// so one 32x16 block covers the full 64x64 tile. A and B are loaded through
// shared memory in K chunks of 8. The template flags delete runtime branches
// from the hot loop, because branchy GEMM is the "damn Daniel" of bad ideas.
// FULL_TILE means M, N, and K line up with the tile sizes, so bounds checks can
// be skipped safely. Otherwise the kernel zeros out-of-range shared-memory loads.
template <bool TRANS_A, bool TRANS_B, bool ACCUMULATE, bool FULL_TILE>
static __global__ void gemm_kernel_2d_tiled_fast(
    int M, int N, int K,
    const float* __restrict__ A, int lda, size_t strideA,
    const float* __restrict__ B, int ldb, size_t strideB,
    float* __restrict__ C, int ldc, size_t strideC) 
{
    __shared__ float As[GEMM_BM * GEMM_BK];
    __shared__ float Bs[GEMM_BK * GEMM_BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tid = ty * blockDim.x + tx;


    A += bz * strideA;
    B += bz * strideB;
    C += bz * strideC;

    int row = by * GEMM_BM + ty * GEMM_TM;
    int col = bx * GEMM_BN + tx * GEMM_TN;

    float acc[GEMM_TM][GEMM_TN] = {0.0f};

    for (int k0 = 0; k0 < K; k0 += GEMM_BK) {

        int a_row_local = tid / GEMM_BK;
        int a_col_local = tid % GEMM_BK; 
        
        int a_row = by * GEMM_BM + a_row_local;
        int a_col = k0 + a_col_local;

        if (FULL_TILE || (a_row < M && a_col < K)) {
            As[a_row_local * GEMM_BK + a_col_local] = TRANS_A
                ? A[(size_t)a_col * lda + a_row]
                : A[(size_t)a_row * lda + a_col];
        } else {
            As[a_row_local * GEMM_BK + a_col_local] = 0.0f;
        }

        int b_row_local = tid / GEMM_BN;
        int b_col_local = tid % GEMM_BN; 
        
        int b_row = k0 + b_row_local;
        int b_col = bx * GEMM_BN + b_col_local;

        if (FULL_TILE || (b_row < K && b_col < N)) {
            Bs[b_row_local * GEMM_BN + b_col_local] = TRANS_B
                ? B[(size_t)b_col * ldb + b_row]
                : B[(size_t)b_row * ldb + b_col];
        } else {
            Bs[b_row_local * GEMM_BN + b_col_local] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < GEMM_BK; k++) {
            #pragma unroll
            for (int i = 0; i < GEMM_TM; i++) {
                #pragma unroll
                for (int j = 0; j < GEMM_TN; j++) {
                    acc[i][j] += As[(ty * GEMM_TM + i) * GEMM_BK + k] *
                                 Bs[k * GEMM_BN + (tx * GEMM_TN + j)];
                }
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < GEMM_TM; i++) {
        for (int j = 0; j < GEMM_TN; j++) {

            int r = row + i;
            int c = col + j;

            if (FULL_TILE || (r < M && c < N)) {
                size_t idx = (size_t)r * ldc + c;

                if (ACCUMULATE)
                    C[idx] += acc[i][j];
                else
                    C[idx] = acc[i][j];
            }
        }
    }
}

// Launch one concrete GEMM variant.
// TRANS_A, TRANS_B, and ACCUMULATE are already compile-time decisions here.
// This function only decides whether the shape is perfectly tile-aligned.
// Full-tile path removes boundary checks. Edge-safe path keeps them, because
// illegal memory access is not a personality trait.
template <bool TRANS_A, bool TRANS_B, bool ACCUMULATE>
static inline void launch_gemm_variant(
    int M, int N, int K,
    const float* A, int lda, size_t strideA,
    const float* B, int ldb, size_t strideB,
    float* C, int ldc, size_t strideC,
    dim3 dimGrid, dim3 dimBlock)
{
    bool full_tile = (M % GEMM_BM == 0) && (N % GEMM_BN == 0) && (K % GEMM_BK == 0);

    if (full_tile) {
        gemm_kernel_2d_tiled_fast<TRANS_A, TRANS_B, ACCUMULATE, true><<<dimGrid, dimBlock>>>(
            M, N, K, A, lda, strideA, B, ldb, strideB, C, ldc, strideC);
    } else {
        gemm_kernel_2d_tiled_fast<TRANS_A, TRANS_B, ACCUMULATE, false><<<dimGrid, dimBlock>>>(
            M, N, K, A, lda, strideA, B, ldb, strideB, C, ldc, strideC);
    }
}

// Dispatch the transpose combination.
// The public API gives transA/transB as runtime bools. This turns those bools
// into template parameters before launch, so the GPU kernel does not keep asking
// "am I transposed?" inside every multiply. That question gets answered once on
// the host. Big brain, not galaxy brain.
template <bool ACCUMULATE>
static inline void dispatch_gemm_variant(
    int M, int N, int K,
    const float* A, int lda, size_t strideA,
    const float* B, int ldb, size_t strideB,
    float* C, int ldc, size_t strideC,
    bool transA, bool transB,
    dim3 dimGrid, dim3 dimBlock)
{
    if (!transA && !transB) {
        launch_gemm_variant<false, false, ACCUMULATE>(
            M, N, K, A, lda, strideA, B, ldb, strideB, C, ldc, strideC, dimGrid, dimBlock);
    } else if (transA && !transB) {
        launch_gemm_variant<true, false, ACCUMULATE>(
            M, N, K, A, lda, strideA, B, ldb, strideB, C, ldc, strideC, dimGrid, dimBlock);
    } else if (!transA && transB) {
        launch_gemm_variant<false, true, ACCUMULATE>(
            M, N, K, A, lda, strideA, B, ldb, strideB, C, ldc, strideC, dimGrid, dimBlock);
    } else {
        launch_gemm_variant<true, true, ACCUMULATE>(
            M, N, K, A, lda, strideA, B, ldb, strideB, C, ldc, strideC, dimGrid, dimBlock);
    }
}

// Final normal-GEMM launcher.
// This is the bridge from the library's runtime flags into the templated GEMM
// variants. accumulate chooses whether the output C is overwritten or added to.
// transA/transB choose the memory addressing scheme. dimGrid/dimBlock are passed
// in by the caller so both custom_gemm entrypoints can reuse the same dispatch.
static inline void launch_custom_gemm_kernel(
    int M, int N, int K,
    const float* A, int lda, size_t strideA,
    const float* B, int ldb, size_t strideB,
    float* C, int ldc, size_t strideC,
    bool transA, bool transB, bool accumulate,
    dim3 dimGrid, dim3 dimBlock)
{
    if (accumulate) {
        dispatch_gemm_variant<true>(
            M, N, K, A, lda, strideA, B, ldb, strideB, C, ldc, strideC,
            transA, transB, dimGrid, dimBlock);
    } else {
        dispatch_gemm_variant<false>(
            M, N, K, A, lda, strideA, B, ldb, strideB, C, ldc, strideC,
            transA, transB, dimGrid, dimBlock);
    }
}

static float* d_global_gemm_workspace = nullptr;
static size_t current_gemm_workspace_size = 0;
static float* d_global_splitk_workspace = nullptr;
static size_t current_splitk_workspace_size = 0;

// Split-K partial GEMM.
// Normal GEMM parallelizes over output tiles. That sucks when M*N is small but K
// is huge, because only a few blocks get launched and the GPU sits there like
// "bro, is that it?" Split-K adds another parallel dimension: blockIdx.z packs
// both batch_id and split_id. Each split computes only a slice of K and writes a
// partial C tile into the partials workspace.
//
// partials layout:
//   [batch_id][split_id][M * N]
//
// No accumulation into C happens here. This kernel only writes partial sums.
// reduce_splitk_kernel combines them afterward.
template <bool TRANS_A, bool TRANS_B, bool FULL_MN_TILE>
static __global__ void gemm_splitk_kernel(
    int M, int N, int K,
    const float* __restrict__ A, int lda, size_t strideA,
    const float* __restrict__ B, int ldb, size_t strideB,
    float* __restrict__ partials, size_t partial_stride,
    int split_count, int split_k)
{
    __shared__ float As[GEMM_BM * GEMM_BK];
    __shared__ float Bs[GEMM_BK * GEMM_BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int packed_z = blockIdx.z;
    int split_id = packed_z % split_count;
    int batch_id = packed_z / split_count;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    int k_begin = split_id * split_k;
    if (k_begin >= K) return;
    int k_limit = min(K, k_begin + split_k);

    A += (size_t)batch_id * strideA;
    B += (size_t)batch_id * strideB;
    partials += ((size_t)batch_id * split_count + split_id) * partial_stride;

    int row = by * GEMM_BM + ty * GEMM_TM;
    int col = bx * GEMM_BN + tx * GEMM_TN;

    float acc[GEMM_TM][GEMM_TN] = {0.0f};

    for (int k0 = k_begin; k0 < k_limit; k0 += GEMM_BK) {
        int a_row_local = tid / GEMM_BK;
        int a_col_local = tid % GEMM_BK;
        int a_row = by * GEMM_BM + a_row_local;
        int a_col = k0 + a_col_local;

        if ((FULL_MN_TILE || a_row < M) && a_col < k_limit) {
            As[a_row_local * GEMM_BK + a_col_local] = TRANS_A
                ? A[(size_t)a_col * lda + a_row]
                : A[(size_t)a_row * lda + a_col];
        } else {
            As[a_row_local * GEMM_BK + a_col_local] = 0.0f;
        }

        int b_row_local = tid / GEMM_BN;
        int b_col_local = tid % GEMM_BN;
        int b_row = k0 + b_row_local;
        int b_col = bx * GEMM_BN + b_col_local;

        if (b_row < k_limit && (FULL_MN_TILE || b_col < N)) {
            Bs[b_row_local * GEMM_BN + b_col_local] = TRANS_B
                ? B[(size_t)b_col * ldb + b_row]
                : B[(size_t)b_row * ldb + b_col];
        } else {
            Bs[b_row_local * GEMM_BN + b_col_local] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < GEMM_BK; ++k) {
            #pragma unroll
            for (int i = 0; i < GEMM_TM; ++i) {
                #pragma unroll
                for (int j = 0; j < GEMM_TN; ++j) {
                    acc[i][j] += As[(ty * GEMM_TM + i) * GEMM_BK + k] *
                                 Bs[k * GEMM_BN + (tx * GEMM_TN + j)];
                }
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < GEMM_TM; ++i) {
        for (int j = 0; j < GEMM_TN; ++j) {
            int r = row + i;
            int c = col + j;
            if (FULL_MN_TILE || (r < M && c < N)) {
                partials[(size_t)r * N + c] = acc[i][j];
            }
        }
    }
}

// Reduce split-K partials back into the final C matrix.
// Each thread owns one output element for one batch item. It walks all K-splits,
// sums the matching partial values, then writes or accumulates into C depending
// on ACCUMULATE. This second kernel is the price we pay for more parallelism.
// Worth it only when the original GEMM had tiny-grid sadness.
template <bool ACCUMULATE>
static __global__ void reduce_splitk_kernel(
    const float* __restrict__ partials,
    float* __restrict__ C,
    int M, int N, int split_count, int batch_size,
    size_t partial_stride, size_t strideC)
{
    size_t elems_per_batch = (size_t)M * N;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = elems_per_batch * batch_size;
    if (idx >= total) return;

    int batch_id = idx / elems_per_batch;
    size_t elem = idx - (size_t)batch_id * elems_per_batch;

    float sum = 0.0f;
    const float* batch_partials = partials + (size_t)batch_id * split_count * partial_stride;
    for (int split = 0; split < split_count; ++split) {
        sum += batch_partials[(size_t)split * partial_stride + elem];
    }

    float* out = C + (size_t)batch_id * strideC;
    if (ACCUMULATE) {
        out[elem] += sum;
    } else {
        out[elem] = sum;
    }
}

// Allocate split-K workspace and launch split-K.
// This handles the whole split-K sequence:
//   1. Ensure workspace is big enough for batch_size * split_count * M * N.
//   2. Launch gemm_splitk_kernel over normal output tiles plus K splits.
//   3. Launch reduce_splitk_kernel to combine partial sums into C.
// The workspace is cached globally so we do not cudaMalloc every batch like a
// cursed tutorial from 2016.
template <bool TRANS_A, bool TRANS_B>
static inline void launch_splitk_variant(
    int M, int N, int K,
    const float* A, int lda, size_t strideA,
    const float* B, int ldb, size_t strideB,
    float* C, size_t strideC,
    bool accumulate, int batch_size,
    dim3 dimGrid, dim3 dimBlock,
    int split_count, int split_k)
{
    size_t partial_stride = (size_t)M * N;
    size_t required_workspace_size = (size_t)batch_size * split_count * partial_stride * sizeof(float);
    if (required_workspace_size > current_splitk_workspace_size) {
        if (d_global_splitk_workspace) {
            safeCudaFree(&d_global_splitk_workspace, "split-k gemm workspace");
        }
        safeCudaMalloc(&d_global_splitk_workspace, required_workspace_size, "split-k gemm workspace");
        current_splitk_workspace_size = required_workspace_size;
    }

    bool full_mn_tile = (M % GEMM_BM == 0) && (N % GEMM_BN == 0);
    dim3 splitGrid(dimGrid.x, dimGrid.y, batch_size * split_count);

    if (full_mn_tile) {
        gemm_splitk_kernel<TRANS_A, TRANS_B, true><<<splitGrid, dimBlock>>>(
            M, N, K, A, lda, strideA, B, ldb, strideB,
            d_global_splitk_workspace, partial_stride, split_count, split_k);
    } else {
        gemm_splitk_kernel<TRANS_A, TRANS_B, false><<<splitGrid, dimBlock>>>(
            M, N, K, A, lda, strideA, B, ldb, strideB,
            d_global_splitk_workspace, partial_stride, split_count, split_k);
    }

    int threads = 256;
    int blocks = ((size_t)batch_size * M * N + threads - 1) / threads;
    if (accumulate) {
        reduce_splitk_kernel<true><<<blocks, threads>>>(
            d_global_splitk_workspace, C, M, N, split_count, batch_size, partial_stride, strideC);
    } else {
        reduce_splitk_kernel<false><<<blocks, threads>>>(
            d_global_splitk_workspace, C, M, N, split_count, batch_size, partial_stride, strideC);
    }
}

// Decide whether split-K is worth using for this shape.
// Split-K is not free: it writes partials and launches a reduction kernel. So we
// only use it when:
//   - K is large enough that splitting has real work per split.
//   - normal output tile count is too small to feed the GPU.
//   - split_count is at least 2 and capped to avoid workspace exploding.
// This is shape-based, not CIFAR-specific. The vibe is "help any skinny-output,
// huge-K GEMM", not "hardcode the dataset and pray".
static inline bool choose_splitk_gemm(
    int M, int N, int K, int batch_size,
    int tile_count, int& split_count, int& split_k)
{
    if (batch_size <= 0 || tile_count <= 0) return false;
    if (K < GEMM_SPLITK_MIN_K_PER_SPLIT * 2) return false;
    if (tile_count * batch_size >= GEMM_SPLITK_TARGET_TILES) return false;

    int needed_splits = (GEMM_SPLITK_TARGET_TILES + tile_count * batch_size - 1) / (tile_count * batch_size);
    int max_splits_by_k = K / GEMM_SPLITK_MIN_K_PER_SPLIT;
    split_count = min(GEMM_SPLITK_MAX_SPLITS, min(needed_splits, max_splits_by_k));
    if (split_count < 2) return false;

    int raw_split_k = (K + split_count - 1) / split_count;
    split_k = ((raw_split_k + GEMM_BK - 1) / GEMM_BK) * GEMM_BK;
    return split_k > 0;
}

// Dispatch split-K for all transpose combinations.
// Same idea as dispatch_gemm_variant, but for the split-K path. We convert the
// runtime transpose flags into template parameters so the split-K kernel gets
// clean addressing logic without hot-loop if statements. The API stays boring;
// the compiler does the spicy part.
static inline void launch_splitk_gemm_kernel(
    int M, int N, int K,
    const float* A, int lda, size_t strideA,
    const float* B, int ldb, size_t strideB,
    float* C, size_t strideC,
    bool transA, bool transB, bool accumulate,
    int batch_size, dim3 dimGrid, dim3 dimBlock,
    int split_count, int split_k)
{
    if (!transA && !transB) {
        launch_splitk_variant<false, false>(
            M, N, K, A, lda, strideA, B, ldb, strideB, C, strideC,
            accumulate, batch_size, dimGrid, dimBlock, split_count, split_k);
    } else if (transA && !transB) {
        launch_splitk_variant<true, false>(
            M, N, K, A, lda, strideA, B, ldb, strideB, C, strideC,
            accumulate, batch_size, dimGrid, dimBlock, split_count, split_k);
    } else if (!transA && transB) {
        launch_splitk_variant<false, true>(
            M, N, K, A, lda, strideA, B, ldb, strideB, C, strideC,
            accumulate, batch_size, dimGrid, dimBlock, split_count, split_k);
    } else {
        launch_splitk_variant<true, true>(
            M, N, K, A, lda, strideA, B, ldb, strideB, C, strideC,
            accumulate, batch_size, dimGrid, dimBlock, split_count, split_k);
    }
}

// Main handheld GEMM entrypoint used by the CUDA layers.
// This function owns the policy:
//   - If batch_size > 1 and strideC == 0, each batch item writes into workspace
//     and reduce_batch_kernel sums across the batch. This is used by conv weight
//     gradients where the final output is shared across batch items.
//   - Otherwise it may choose split-K when K is huge and output tile count is
//     weak.
//   - If neither special path applies, it launches the normal tiled GEMM.
// The public implementation type stays the same: handwritten CUDA, no library
// magic, no cuBLAS, no "just trust me bro" black box.
inline void custom_gemm(
    int M, int N, int K,
    const float* A, const float* B, float* C,
    bool transA, bool transB, bool accumulate = false,
    int batch_size = 1, size_t strideA = 0, size_t strideB = 0, size_t strideC = 0) 
{
    if (M <= 0 || N <= 0 || K <= 0) return;

    int lda = transA ? M : K;
    int ldb = transB ? K : N;
    int ldc = N;

    dim3 dimBlock(32,16);
    dim3 dimGrid((N + GEMM_BN - 1) / GEMM_BN, (M + GEMM_BM - 1) / GEMM_BM, batch_size);

    if (strideC == 0 && batch_size > 1) {
        size_t required_workspace_size = (size_t)batch_size * M * N * sizeof(float);
        if (required_workspace_size > current_gemm_workspace_size) {
            if (d_global_gemm_workspace) safeCudaFree(&d_global_gemm_workspace,"custom_gemm workspace");
            safeCudaMalloc(&d_global_gemm_workspace, required_workspace_size,"custom_gemm workspace");
            current_gemm_workspace_size = required_workspace_size;
        }
        
        size_t workspace_stride = (size_t)M * N;  
        
        launch_custom_gemm_kernel(
            M, N, K, A, lda, strideA, B, ldb, strideB,
            d_global_gemm_workspace, ldc, workspace_stride,
            transA, transB, false, dimGrid, dimBlock);
        
        int threads = 256;
        int blocks = ((size_t)M * N + threads - 1) / threads;
        reduce_batch_kernel<<<blocks, threads>>>(d_global_gemm_workspace, C, M, N, batch_size, accumulate);
    } 
    else {
        size_t effective_strideC = (strideC == 0) ? (size_t)M * N : strideC;
        int split_count = 1;
        int split_k = 0;
        int tile_count = dimGrid.x * dimGrid.y;
        if (choose_splitk_gemm(M, N, K, batch_size, tile_count, split_count, split_k)) {
            launch_splitk_gemm_kernel(
                M, N, K, A, lda, strideA, B, ldb, strideB, C, effective_strideC,
                transA, transB, accumulate, batch_size, dimGrid, dimBlock,
                split_count, split_k);
        } else {
            launch_custom_gemm_kernel(
                M, N, K, A, lda, strideA, B, ldb, strideB, C, ldc, effective_strideC,
                transA, transB, accumulate, dimGrid, dimBlock);
        }
    }
}

// Secondary GEMM entrypoint kept for compatibility.
// It mirrors custom_gemm but uses the symbolic block dimensions derived from the
// GEMM constants. Existing code can call it without caring which internal path
// runs. Same choices: batched workspace reduce, split-K for under-occupied huge
// K, or normal tiled GEMM.
inline void custom_gemm2(
    int M, int N, int K,
    const float* A, const float* B, float* C,
    bool transA, bool transB, bool accumulate = false,
    int batch_size = 1, size_t strideA = 0, size_t strideB = 0, size_t strideC = 0) 
{
    if (M <= 0 || N <= 0 || K <= 0) return;

    int lda = transA ? M : K;
    int ldb = transB ? K : N;
    int ldc = N;

    dim3 dimBlock(GEMM_BN / GEMM_TN, GEMM_BM / GEMM_TM); 
    dim3 dimGrid((N + GEMM_BN - 1) / GEMM_BN, (M + GEMM_BM - 1) / GEMM_BM, batch_size);

    if (strideC == 0 && batch_size > 1) {
        size_t required_workspace_size = (size_t)batch_size * M * N * sizeof(float);
        
        if (required_workspace_size > current_gemm_workspace_size) {
            if (d_global_gemm_workspace) safeCudaFree(&d_global_gemm_workspace,"custom_gemm workspace");
            safeCudaMalloc(&d_global_gemm_workspace, required_workspace_size,"custom_gemm workspace");
            current_gemm_workspace_size = required_workspace_size;
        }
        
        size_t workspace_stride = (size_t)M * N; 
        
        launch_custom_gemm_kernel(
            M, N, K, A, lda, strideA, B, ldb, strideB,
            d_global_gemm_workspace, ldc, workspace_stride,
            transA, transB, false, dimGrid, dimBlock);
        
        int threads = 256;
        int blocks = ((size_t)M * N + threads - 1) / threads;
        reduce_batch_kernel<<<blocks, threads>>>(d_global_gemm_workspace, C, M, N, batch_size, accumulate);
    } 
    else {
        size_t effective_strideC = (strideC == 0) ? (size_t)M * N : strideC;
        int split_count = 1;
        int split_k = 0;
        int tile_count = dimGrid.x * dimGrid.y;
        if (choose_splitk_gemm(M, N, K, batch_size, tile_count, split_count, split_k)) {
            launch_splitk_gemm_kernel(
                M, N, K, A, lda, strideA, B, ldb, strideB, C, effective_strideC,
                transA, transB, accumulate, batch_size, dimGrid, dimBlock,
                split_count, split_k);
        } else {
            launch_custom_gemm_kernel(
                M, N, K, A, lda, strideA, B, ldb, strideB, C, ldc, effective_strideC,
                transA, transB, accumulate, dimGrid, dimBlock);
        }
    }
}


// ----------------------------------------------------
// Batched Im2Col
// ----------------------------------------------------
// Im2col kernel for batched convolution.
// Each thread writes one element of the column buffer. The flattened index is
// decoded into:
//   batch, input channel, kernel row/col, output row/col
// Then it maps that output/kernel position back to an input pixel. If padding
// puts the coordinate outside the image, it writes zero. This converts NCHW image
// tensors into GEMM-friendly columns. Convolution said "I am matrix multiply
// now", and honestly, respect.
static __global__ void im2col_kernel(
    const float* data_im, int channels,
    int height, int width, int ksize, int stride, int pad,
    int height_col, int width_col,
    float* data_col, size_t total_elements) 
{
    size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < total_elements) {
        int col_size = channels * ksize * ksize * height_col * width_col;
        int b = n / col_size; 
        int col_idx = n % col_size;

        int w_out = col_idx % width_col;
        int h_out = (col_idx / width_col) % height_col;
        int k_w = (col_idx / (width_col * height_col)) % ksize;
        int k_h = (col_idx / (width_col * height_col * ksize)) % ksize;
        int c_im = col_idx / (width_col * height_col * ksize * ksize);

        int h_in = h_out * stride - pad + k_h;
        int w_in = w_out * stride - pad + k_w;

        size_t im_offset = b * (channels * height * width);
        size_t col_offset = b * col_size;

        if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
            data_col[col_offset + col_idx] = data_im[im_offset + (c_im * height + h_in) * width + w_in];
        } else {
            data_col[col_offset + col_idx] = 0.0f;
        }
    }
}

// Col2im kernel for convolution backward.
// Each thread owns one input-gradient pixel and gathers every im2col position
// that could have contributed to it. This is the inverse-ish operation of
// im2col: multiple column entries may map back to the same input pixel, so this
// kernel sums them. No atomics needed because one thread owns one output pixel.
static __global__ void col2im_kernel(
    const float* data_col, int channels,
    int height, int width, int ksize, int stride, int pad,
    int height_col, int width_col,
    float* data_im, size_t total_elements) 
{
    size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < total_elements) {
        int im_size = channels * height * width;
        int b = n / im_size;
        int im_idx = n % im_size;

        float val = 0.0f;
        int w_im = im_idx % width;
        int h_im = (im_idx / width) % height;
        int c_im = im_idx / (width * height);

        int w_col_start = (w_im + pad < ksize) ? 0 : (w_im + pad - ksize) / stride + 1;
        int w_col_end = min((w_im + pad) / stride + 1, width_col);
        int h_col_start = (h_im + pad < ksize) ? 0 : (h_im + pad - ksize) / stride + 1;
        int h_col_end = min((h_im + pad) / stride + 1, height_col);

        size_t col_offset = b * (channels * ksize * ksize * height_col * width_col);

        for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
            for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
                int k_h = h_im + pad - h_col * stride;
                int k_w = w_im + pad - w_col * stride;
                int c_idx = (((c_im * ksize + k_h) * ksize + k_w) * height_col + h_col) * width_col + w_col;
                val += data_col[col_offset + c_idx];
            }
        }
        data_im[n] = val;
    }
}

// Host launcher for batched im2col.
// Computes the total number of column-buffer elements and launches one thread
// per element. The caller is responsible for passing the already-computed output
// spatial size, because this helper is just the launcher, not the conv layer
// therapist.
inline void im2col(
    const float* data_im, int batch_size, int channels, int height, int width,
    int ksize, int stride, int pad, int height_col, int width_col, float* data_col)
{
    size_t total_threads = (size_t)batch_size * channels * ksize * ksize * height_col * width_col;
    int num_blocks = (total_threads + 255) / 256;
    im2col_kernel<<<num_blocks, 256>>>(data_im, channels, height, width, ksize, stride, pad, height_col, width_col, data_col, total_threads);
}

// Host launcher for batched col2im.
// Computes one thread per input-gradient element and lets col2im_kernel gather
// the matching column entries. Used in conv backward when dX has to return to
// normal NCHW layout.
inline void col2im(
    const float* data_col, int batch_size, int channels, int height, int width,
    int ksize, int stride, int pad, int height_col, int width_col, float* data_im)
{
    size_t total_threads = (size_t)batch_size * channels * height * width;
    int num_blocks = (total_threads + 255) / 256;
    col2im_kernel<<<num_blocks, 256>>>(data_col, channels, height, width, ksize, stride, pad, height_col, width_col, data_im, total_threads);
}

// ----------------------------------------------------
// Original Helpers
// ----------------------------------------------------
// Add one vector into another.
// Thread idx maps directly to one scalar. If idx is in range, out[idx] gets
// bias[idx] added. Small helper, no drama, no main-character arc.
static __global__ void add_vectors_kernel(float* out, const float* bias, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] += bias[idx];
}

// Add a bias vector to every row in a batched 2D output.
// output is [batch_size][output_size]. The neuron index is idx % output_size,
// so every sample gets the same bias value for that output neuron. Classic dense
// layer bias add, just parallelized.
static __global__ void batched_add_bias_kernel(
    float* output, const float* biases, int batch_size, int output_size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * output_size;
    if (idx < total) {
        int neuron = idx % output_size;
        output[idx] += biases[neuron];
    }
}

// Reshape flattened gradients back to NCHW.
// The flattened input is logically [batch][channels * height * width]. The output
// is [batch][channel][height][width]. Since the current flatten layout is already
// contiguous in that same order, this mainly decodes idx into n/c/h/w and writes
// the matching flattened element. Index gymnastics, not rocket science.
static __global__ void unflatten_gradient_kernel(
    const float* flattened_grad, float* unflattened_grad, int batch_size, int channels, int height, int width) 
{
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
#endif
#if defined(COMPILE_HELPERS_API) && !defined(HELPERS_API_IMPLEMENTED)
#define HELPERS_API_IMPLEMENTED

namespace cudafunc 
{

// Public wrapper for unflattening gradients.
// Computes the total NCHW element count, launches unflatten_gradient_kernel with
// 256 threads per block, and leaves synchronization/error handling to the caller
// pattern used elsewhere in this CUDA code.
extern "C" void unflatten_gradient(
    const float* flattened_grad, float* unflattened_grad,
    int batch_size, int channels, int height, int width) 
{
    size_t total_elements = (size_t)batch_size * channels * height * width;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    unflatten_gradient_kernel<<<blocks, threads>>>(
        flattened_grad, unflattened_grad, batch_size, channels, height, width
    );
}

} // namespace cudafunc
#endif
