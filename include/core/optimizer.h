#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <cuda_runtime.h>
#include <unordered_map>
#include <string>
#include <memory>
#include <vector> 

class CpuAdamOptimizer;


class OptimizerBase {
public:
    virtual ~OptimizerBase() = default;
    virtual void update(float* params, float* grads, size_t size) = 0;
    virtual void step() = 0; // Keep step, might be useful later
};

class AdamOptimizer : public OptimizerBase {
public:
    AdamOptimizer(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);
    ~AdamOptimizer() override;

    void update(float* params, float* grads, size_t size) override;
    void step() override;

private:
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    int iteration;

    struct ParamState {
        float* momentum;
        float* velocity;
    };

    std::unordered_map<float*, ParamState> state;
    void init_state(float* params, size_t size);
};

class CpuAdamOptimizer : public OptimizerBase {
    public:
        CpuAdamOptimizer(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);
        ~CpuAdamOptimizer() override = default;
    
        void update(float* params, float* grads, size_t size) override;
        void step() override;
    
    private:
        float learning_rate;
        float beta1;
        float beta2;
        float epsilon;
        int iteration;
    
        struct ParamState {
            std::vector<float> momentum;
            std::vector<float> velocity;
        };
    
        std::unordered_map<float*, ParamState> state;
        void init_state(float* params, size_t size);
    };

// --- Optimizer Factory Function ---
// Creates the appropriate optimizer based on whether CUDA is used.
// Returns a unique_ptr for automatic memory management.
std::unique_ptr<OptimizerBase> create_optimizer(const std::string& type, float lr, float beta1, float beta2, float eps);


#endif // OPTIMIZER_H