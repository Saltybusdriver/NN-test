#ifndef REDUCTION_HELPERS_CU_INCLUDED
#define REDUCTION_HELPERS_CU_INCLUDED


#include <cmath>
#include <cuda_runtime.h>
#include "cuda_functions.h"
#include <cstdio>
#include <cstdlib>

#include <cooperative_groups.h>

#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

static __device__ __forceinline__ float warpReduceMax(float v) {
    unsigned mask = 0xffffffffu;
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(mask, v, off));
    return v;
}
static __device__ __forceinline__ float warpReduceSum(float v) {
    unsigned mask = 0xffffffffu;
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(mask, v, off);
    return v;
}
template<int BLOCK_SIZE>
static __device__ __forceinline__ float blockReduceMax(float v) {
    __shared__ float smem[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    v = warpReduceMax(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();
    float out = -INFINITY;
    if (wid == 0) {
        out = (lane < (BLOCK_SIZE + 31) / 32) ? smem[lane] : -INFINITY;
        out = warpReduceMax(out);
        if (lane == 0) smem[0] = out;
    }
    __syncthreads();
    out = smem[0];
    return out;
}
template<int BLOCK_SIZE>
static __device__ __forceinline__ float blockReduceSum(float v) {
    __shared__ float smem[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    v = warpReduceSum(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();
    float out = 0.f;
    if (wid == 0) {
        out = (lane < (BLOCK_SIZE + 31) / 32) ? smem[lane] : 0.f;
        out = warpReduceSum(out);
        if (lane == 0) smem[0] = out;
    }
    __syncthreads();
    out = smem[0];
    return out;
}

#endif 
