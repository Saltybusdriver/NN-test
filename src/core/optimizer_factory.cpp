#include "optimizer.h"
#include "utils.h"
#include <stdexcept>
#include <iostream>
#include <layer_proxy.h>

std::unique_ptr<OptimizerBase> create_optimizer(const std::string& type, float lr, float beta1, float beta2, float eps) {
    bool use_cuda = GetUseCuda();

    if (type == "adam") 
    {
        if (use_cuda) 
        {
            std::cout << "Creating CUDA Adam Optimizer (lr=" << lr << ")" << std::endl;
            return std::make_unique<AdamOptimizer>(lr, beta1, beta2, eps);
        } 
        else 
        {
            std::cout << "Creating CPU Adam Optimizer (lr=" << lr << ")" << std::endl;
            return std::make_unique<CpuAdamOptimizer>(lr, beta1, beta2, eps);
        }
    }
    // Add cases for other optimizer types here (e.g., "sgd")
    throw std::invalid_argument("Unknown optimizer type: " + type);
}