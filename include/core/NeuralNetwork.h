#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "layers.h"
#include "optimizer.h"
#include <vector>
#include "layer_proxy.h"

class NeuralNetwork
{
private:
    // Remove the separate cuda_forward and cpu_forward functions
    AdamOptimizer optimizer;
    bool use_cuda_flag = true;

public:
    std::vector<Layer*> layers;
    NeuralNetwork(int batch);
    ~NeuralNetwork();
    
    float* d_element_loss = nullptr;
    float* d_total_loss = nullptr;
    size_t loss_buffer_elements = 0;
    cublasHandle_t cublas_handle = nullptr;
    
    int batch_size;
    static void set_use_cuda(bool use_cuda) { SetUseCuda(use_cuda); }
    static bool get_use_cuda() { return GetUseCuda(); }
    
    void use_cuda(bool use_cuda) { use_cuda_flag = use_cuda; }
    bool is_using_cuda() const { return use_cuda_flag; }
    void load_parameters(const std::string& filename);
    void save_parameters(const std::string& filename) const;
    void add_layer(Layer* layer);
    
    void forward(float* input,int act_batch_size, bool use_cuda = true);
    
    void backward(float* target, float learning_rate, const char* loss_type);   
    void compute_loss(float* d_output, float* d_target, float* loss, int size, const char* loss_type = "mse");
};

#endif  // NEURALNETWORK_H