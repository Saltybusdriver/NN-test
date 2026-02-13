#include "NeuralNetwork.h"
#include "cuda_functions.h"
#include <cublas_v2.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "debug.h"
#include "utils.h"
#include "DebugUtils.h"
#include "layer_proxy.h"
#include "cpu_functions.h"
#include "cpu_DebugUtils.h"

void NeuralNetwork::forward(float* input,int act_batch_size, bool use_cuda)
{
    if (!input) {
        throw std::invalid_argument("Input pointer is null");
    }

    use_cuda_flag = use_cuda;  
    SetUseCuda(use_cuda);      

    float* layer_input = input;
    float* next_input = nullptr;
    
    for (int i = 0; i < layers.size(); ++i) 
    {
        Layer* layer = layers[i];
        if (!layer) 
        {
            throw std::runtime_error("Layer pointer is null");
        }
        layer->set_batch_size(act_batch_size);
        size_t total_output_size = layer->get_total_output_size();
        size_t output_bytes = total_output_size * sizeof(float);
        size_t input_bytes = layer->get_input_size() * sizeof(float);

        layer->forward(layer_input, layer->output);
 
        if (!use_cuda && debug_flag) 
        {
            std::cout << "[DEBUG] CPU Forward Output - Layer " << i << ":" << std::endl;
            int print_elements = std::min((size_t)16, total_output_size); // Print up to 16 elements
            cpu_debug::inspectHostTensorFull("Layer Output Sample", layer->output, print_elements, 4, 4);
        }
        if (use_cuda && debug_flag) 
        {
            std::cout << "[DEBUG] CUDA Forward Output - Layer " << i << ":" << std::endl;

            int print_elements = std::min((size_t)16, total_output_size); // Print up to 16 elements
            DebugUtils::inspectTensorFull("Layer Output Sample", layer->output, print_elements, 4, 4);
        }
        if (i < layers.size() - 1) 
        {
            size_t next_input_size = layers[i+1]->get_input_size();
            

            if (use_cuda) 
            {
                safeCudaMalloc(&next_input, output_bytes, "Next layer input");
                CUDA_CHECK_ERROR(cudaMemcpy(next_input, layer->output, 
                                        output_bytes, cudaMemcpyDeviceToDevice));
            } 
            else
            {
             
                if (next_input) free(next_input);
                next_input = (float*)malloc(output_bytes);
                if (!next_input) 
                {
                    throw std::runtime_error("Failed to allocate host memory for next layer input");
                }
                memcpy(next_input, layer->output, output_bytes);
            }
            layer_input = next_input;
        }
    }
    if (next_input) 
    {
        if (use_cuda) 
        {
            cudaFree(next_input);
        } 
        else 
        {
            free(next_input);
        }
        next_input = nullptr;
    }
}

NeuralNetwork::NeuralNetwork(int batch) : optimizer(0.001f, 0.9f, 0.999f, 1e-8f), batch_size(batch)
{
    batch_size = batch;
    
}

NeuralNetwork::~NeuralNetwork()
{

}

void NeuralNetwork::add_layer(Layer* layer) 
{

    layers.push_back(layer);
    layer->batch_size = batch_size;
}

void NeuralNetwork::backward(float* target, float learning_rate, const char* loss_type)
{

    float* curr_error = nullptr;
    for (int i = layers.size() - 1; i >= 0; --i)
    {
        Layer* layer = layers[i];
        float* curr_target = (i == layers.size() - 1) ? target : nullptr;
        layer->backward(curr_target, curr_error, learning_rate, loss_type);
        curr_error = layer->get_input_grad();
    }
}

void NeuralNetwork::compute_loss(float* d_output, float* d_target, float* loss, int size, const char* loss_type)
{
    if (layers.empty()) 
    {
        throw std::runtime_error("Cannot calculate loss: no layers in network");
    }
    
    Layer* last_layer = layers.back();
    int output_features = last_layer->output_size;
    size_t output_elements = batch_size * output_features;
    
    if (use_cuda_flag) 
    {
        cudafunc::calculate_loss_values(d_output, d_target, loss, size, loss_type, batch_size);
    } 
    else 
    {
        cpu_func::calculate_loss_values(d_output, d_target, loss, size, loss_type, batch_size);
    }
}

void NeuralNetwork::save_parameters(const std::string& filename) const 
{
    std::ofstream file(filename, std::ios::binary | std::ios::trunc);
 

    for (const auto* layer : layers) 
    {
        size_t weight_size = 0;//placeholder
        size_t bias_size = 0; //placeholder


        if (weight_size > 0) 
        {
            std::vector<float> h_weights(weight_size);

            file.write(reinterpret_cast<const char*>(h_weights.data()), weight_size * sizeof(float));
        }

        if (bias_size > 0) 
        {
            std::vector<float> h_biases(bias_size);
            file.write(reinterpret_cast<const char*>(h_biases.data()), bias_size * sizeof(float));
        }
    }
    file.close();
}

void NeuralNetwork::load_parameters(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    // Assumes 'this' network has the EXACT same architecture as the saved one.

    for (auto* layer : layers) { // Iterate through existing layers
        size_t weight_size = 0; //placeholder
        size_t bias_size = 0;

        // Read weights into host buffer
        if (weight_size > 0) {
            std::vector<float> h_weights(weight_size);
            file.read(reinterpret_cast<char*>(h_weights.data()), weight_size * sizeof(float));
            if (!file) throw std::runtime_error("Failed to read weights for a layer");
            layer->set_weights(h_weights.data());
        }

        if (bias_size > 0) {
            std::vector<float> h_biases(bias_size);
            file.read(reinterpret_cast<char*>(h_biases.data()), bias_size * sizeof(float));
            if (!file) throw std::runtime_error("Failed to read biases for a layer");
            layer->set_biases(h_biases.data());
        }
    }
    file.peek();
    if (!file.eof()) {
        // Warning or error: file might be from a different network structure
    }
    file.close();
}