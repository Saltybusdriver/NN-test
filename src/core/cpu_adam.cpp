#include "optimizer.h"
#include "cpu_utils.h"
#include <cmath>
#include <algorithm>
#include <iostream>

CpuAdamOptimizer::CpuAdamOptimizer(float lr, float b1, float b2, float eps)
    : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), iteration(0) {}

void CpuAdamOptimizer::init_state(float* params, size_t size) 
{
    if (state.find(params) == state.end()) 
    {
        ParamState param_state;
        param_state.momentum.resize(size, 0.0f);
        param_state.velocity.resize(size, 0.0f);
        state[params] = std::move(param_state); 
    }
}

void CpuAdamOptimizer::update(float* params, float* grads, size_t size) 
{
    if (!params || !grads) 
    {
        std::cerr << "Warning: CpuAdamOptimizer::update received null pointers." << std::endl;
        return;
    }

    init_state(params, size);

    iteration++;
    float beta1_t = 1.0f - std::pow(beta1, iteration);
    float beta2_t = 1.0f - std::pow(beta2, iteration);

    ParamState& current_state = state[params];
 
    if (current_state.momentum.size() != size || current_state.velocity.size() != size) 
    {
         std::cerr << "Error: CPU Adam state size mismatch for params at " << params
                   << ". Expected " << size << ", got M:" << current_state.momentum.size()
                   << ", V:" << current_state.velocity.size() << std::endl;
  
         current_state.momentum.resize(size, 0.0f);
         current_state.velocity.resize(size, 0.0f);
 
    }


    for (size_t i = 0; i < size; ++i) 
    {
        float gradient = grads[i];
        float threshold = 1.0f; 
        gradient = std::max(-threshold, std::min(threshold, gradient));

        current_state.momentum[i] = beta1 * current_state.momentum[i] + (1.0f - beta1) * gradient;
        current_state.velocity[i] = beta2 * current_state.velocity[i] + (1.0f - beta2) * gradient * gradient;

        float m_hat = current_state.momentum[i] / beta1_t;
        float v_hat = current_state.velocity[i] / beta2_t;

        params[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
    }
}

void CpuAdamOptimizer::step() {
    // Currently, no additional steps are required for standard Adam on CPU
}