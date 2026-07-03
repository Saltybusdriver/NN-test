#include "optimizer.h"
#include <cmath>
#include <cstdio>
#include "utils.h"
#define BLOCK_SIZE 256

__global__ void adam_update_kernel(
    float* params,
    float* grads,
    float* momentum,
    float* velocity,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    float beta1_t,
    float beta2_t,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Update momentum
        float gradient = grads[idx];
        float threshold = 1.0f;
        gradient = fmaxf(-threshold, fminf(threshold, gradient));
        
        momentum[idx] = beta1 * momentum[idx] + (1.0f - beta1) * grads[idx];
        
        // Update velocity
        velocity[idx] = beta2 * velocity[idx] + (1.0f - beta2) * grads[idx] * grads[idx];
        
        // Compute bias-corrected moments
        float m_hat = momentum[idx] / beta1_t;
        float v_hat = velocity[idx] / beta2_t;
        
        // Update parameters
        params[idx] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
    }
}

AdamOptimizer::AdamOptimizer(float lr, float b1, float b2, float eps)
    : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), iteration(0) {}

AdamOptimizer::~AdamOptimizer() 
{
    for (auto& pair : state) 
    {
        printf("Freeing Adam state\n");
        safeCudaFree(&pair.second.momentum, "Adam momentum");
        safeCudaFree(&pair.second.velocity, "Adam velocity");
    }
    state.clear();
}

void AdamOptimizer::init_state(float* params, size_t size) 
{
    if (state.find(params) == state.end()) {
        ParamState param_state;
        param_state.momentum = nullptr;
        param_state.velocity = nullptr;
        safeCudaMalloc(&param_state.momentum, size * sizeof(float),"momentum");
        safeCudaMalloc(&param_state.velocity, size * sizeof(float),"velocity");
        cudaMemset(param_state.momentum, 0, size * sizeof(float));
        cudaMemset(param_state.velocity, 0, size * sizeof(float));
        state[params] = param_state;
    }
}

void AdamOptimizer::update(float* params, float* grads, size_t size) {
    if (!params || !grads) return;
    
    init_state(params, size);
    iteration++;
    float beta1_t = 1.0f - pow(beta1, iteration);
    float beta2_t = 1.0f - pow(beta2, iteration);
    
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    adam_update_kernel<<<num_blocks, BLOCK_SIZE>>>(
        params,
        grads,
        state[params].momentum,
        state[params].velocity,
        learning_rate,
        beta1,
        beta2,
        epsilon,
        beta1_t,
        beta2_t,
        size
    );
}

void AdamOptimizer::step() 
{
    // done automatically so not needed, i think.
}