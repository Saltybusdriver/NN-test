#ifndef LAYERS_H
#define LAYERS_H
#include <cstddef>
#include <stdexcept>
#include <cuda_runtime.h>
#include <iostream>
#include <cublas_v2.h>
#include <curand.h>
#include <cuda_runtime.h>
#include "optimizer.h"

class Layer {
    public:
        Layer(const char* activation);
        virtual ~Layer();
    
        int activation_type_id;
        int input_size;
        int output_size;
        int batch_size;
        float* output;
        float* output_error;
    
    
        float* stored_input = nullptr;
        size_t stored_input_size = 0;
        float* pre_activation_values = nullptr;
        size_t preact_buffer_size = 0;
    
        int activation_type(const char* type);
    

        virtual void forward(const float* input, float* output) = 0;
        virtual void backward(const float* target, float* prev_output_error, float learning_rate,const char* loss_type) = 0;
        virtual void update_params(OptimizerBase& optimizer) = 0;
    
        virtual float* set_weights(float* weights) = 0;
        virtual float* set_biases(float* biases) = 0;

        virtual float* get_weights() const = 0;
        virtual float* get_biases() const = 0;
        virtual float* get_weight_grad() const = 0;
        virtual float* get_bias_grad() const = 0;
        virtual int get_input_size() const = 0;
        virtual int get_output_size() const = 0;
        virtual float* get_input_grad() const=0;
        virtual float* get_output_grad() const = 0;
        virtual float* set_pre_activation_values(float* pre_activation_values)=0;
        virtual float* get_pre_activation_values() const = 0;
        virtual void set_batch_size(int batch_size) { this->batch_size = batch_size; }   
        virtual void clearBuffers() = 0;
        virtual void init_sizes() = 0;
        virtual int get_total_output_size() const { return output_size; }
    
    protected:
        virtual void allocate_buffers();
        virtual void free_buffers();
    };
 

namespace Cuda{
class Linear : public Layer {
private:
    int in_features;
    int out_features;
    float *weights;
    float *bias;
    float *weight_grad;
    float *bias_grad;
    float* input_gradients = nullptr; 
    cublasHandle_t cublas_handle;  

public:
    Linear(int in_feat, int out_feat, int b_size,const char* activation);
    ~Linear() override;

    void forward(const float* input, float* output) override;
    void backward(const float* target, float* prev_output_error, float learning_rate,const char* loss_type) override;
    void update_params(OptimizerBase& optimizer) ;

    float* set_weights(float* weights) override;
    float* set_biases(float* biases) override;

    float* get_weights() const override { return weights; }
    float* get_biases() const override { return bias; }
    float* get_weight_grad() const override { return weight_grad; }
    float* get_bias_grad() const override { return bias_grad; }
    virtual float* get_input_grad() const override { return input_gradients; }
    virtual float* get_output_grad() const override { return output_error; }
    int get_input_size() const override { return in_features; }
    int get_output_size() const override { return out_features; }
    float* set_pre_activation_values(float* pre_activation_values) override { return this->pre_activation_values = pre_activation_values; }
    float* get_pre_activation_values() const override { return pre_activation_values; }
    virtual void clearBuffers() override;
    void init_sizes() override;

protected:
    void allocate_buffers() override;
    void init_weights();
    void init_biases();
};




class Conv2d : public Layer {
public:
    Conv2d(int in_ch, int out_ch, int img_h, int img_w, int k_size, int str = 1, int pad = 0, int b_sz = 1,const char* activation = "relu");
    ~Conv2d() override;

    void forward(const float* input, float* output) override;
    void backward(const float* target, float* prev_output_error, float learning_rate,const char* loss_type) override;
    void update_params(OptimizerBase& optimizer) ;

    float* set_weights(float* weights) override;
    float* set_biases(float* biases) override;

    float* get_weights() const override { return kernels; }
    float* get_biases() const override { return bias; }
    float* get_weight_grad() const override { return kernel_grad; }
    float* get_bias_grad() const override { return bias_grad; }

    virtual float* get_input_grad() const override { return input_gradients; }
    virtual float* get_output_grad() const override { return output_error; }
    virtual float* get_pre_activation_values() const override { return pre_activation_values; }
    virtual float* set_pre_activation_values(float* pre_activation_values) override { return this->pre_activation_values = pre_activation_values; }

    virtual void clearBuffers() override;
    int get_input_size() const override;
    int get_output_size() const override;
    void init_sizes() override;

protected:
    void allocate_buffers() override;
    void init_weights();
    void init_biases();
    void init_weights_ones();
    void init_biases_zeros();
private:
    int in_channels, out_channels, kernel_size, stride, padding;
    float *kernels, *bias;
    float *kernel_grad, *bias_grad;
    float* input_gradients = nullptr; 
    int input_height;
    int input_width;
    int kernel_tensor_size;
    size_t kernel_size_bytes;
    size_t bias_size;
    int output_height;
    int output_width;
};


class Flatten : public Layer {
    public:
        Flatten(int batch_size, int channels, int height, int width);
        ~Flatten() override;
        
        void forward(const float* input, float* output) override;
        void backward(const float* target, float* prev_output_error, float learning_rate, const char* loss_type) override;
        void update_params(OptimizerBase& optimizer) { /* No parameters to update */ }
        
        float* set_weights(float* weights) override { return nullptr; }
        float* set_biases(float* biases) override { return nullptr; }
        float* get_weights() const override { return nullptr; }
        float* get_biases() const override { return nullptr; }
        float* get_weight_grad() const override { return nullptr; }
        float* get_bias_grad() const override { return nullptr; }
        float* get_input_grad() const override { return input_gradients; }
        float* get_output_grad() const override { return output_error; }
        float* set_pre_activation_values(float* values) override { return nullptr; }
        float* get_pre_activation_values() const override { return nullptr; }
        
        int get_input_size() const override;
        int get_output_size() const override;
        
        void clearBuffers() override;
        void init_sizes() override;
        
    private:
        int channels;
        int height;
        int width;
        float* input_gradients = nullptr;
    };
}

namespace cpu
{
class Linear : public Layer {
private:
    int in_features;
    int out_features;
    float *weights;
    float *bias;
    float *weight_grad;
    float *bias_grad;
    float* input_gradients = nullptr; 
    cublasHandle_t cublas_handle;  

public:
    Linear(int in_feat, int out_feat, int b_size,const char* activation);
    ~Linear() override;

    void forward(const float* input, float* output) override;
    void backward(const float* target, float* prev_output_error, float learning_rate,const char* loss_type) override;
    void update_params(OptimizerBase& optimizer) ;

    float* set_weights(float* weights) override;
    float* set_biases(float* biases) override;

    float* get_weights() const override { return weights; }
    float* get_biases() const override { return bias; }
    float* get_weight_grad() const override { return weight_grad; }
    float* get_bias_grad() const override { return bias_grad; }
    virtual float* get_input_grad() const override { return input_gradients; }
    virtual float* get_output_grad() const override { return output_error; }
    int get_input_size() const override { return in_features; }
    int get_output_size() const override { return out_features; }
    float* set_pre_activation_values(float* pre_activation_values) override { return this->pre_activation_values = pre_activation_values; }
    float* get_pre_activation_values() const override { return pre_activation_values; }
    virtual void clearBuffers() override;
    void init_sizes() override;

protected:
    void allocate_buffers() override;
    void init_weights();
    void init_biases();
};




class Conv2d : public Layer {
public:
    Conv2d(int in_ch, int out_ch, int img_h, int img_w, int k_size, int str = 1, int pad = 0, int b_sz = 1,const char* activation = "relu");
    ~Conv2d() override;

    void forward(const float* input, float* output) override;
    void backward(const float* target, float* prev_output_error, float learning_rate,const char* loss_type) override;
    void update_params(OptimizerBase& optimizer) ;

    float* set_weights(float* weights) override;
    float* set_biases(float* biases) override;

    float* get_weights() const override { return kernels; }
    float* get_biases() const override { return bias; }
    float* get_weight_grad() const override { return kernel_grad; }
    float* get_bias_grad() const override { return bias_grad; }

    virtual float* get_input_grad() const override { return input_gradients; }
    virtual float* get_output_grad() const override { return output_error; }
    virtual float* get_pre_activation_values() const override { return pre_activation_values; }
    virtual float* set_pre_activation_values(float* pre_activation_values) override { return this->pre_activation_values = pre_activation_values; }

    virtual void clearBuffers() override;
    int get_input_size() const override;
    int get_output_size() const override;
    void init_sizes() override;

protected:
    void allocate_buffers() override;
    void init_weights();
    void init_biases();
    void init_weights_ones();
    void init_biases_zeros();
private:
    int in_channels, out_channels, kernel_size, stride, padding;
    float *kernels, *bias;
    float *kernel_grad, *bias_grad;
    float* input_gradients = nullptr; 
    int input_height;
    int input_width;
    int kernel_tensor_size;
    size_t kernel_size_bytes;
    size_t bias_size;
    int output_height;
    int output_width;
};


class Flatten : public Layer {
    public:
        Flatten(int batch_size, int channels, int height, int width);
        ~Flatten() override;
        
        void forward(const float* input, float* output) override;
        void backward(const float* target, float* prev_output_error, float learning_rate, const char* loss_type) override;
        void update_params(OptimizerBase& optimizer)  { /* No parameters to update */ }
        
        float* set_weights(float* weights) override { return nullptr; }
        float* set_biases(float* biases) override { return nullptr; }
        float* get_weights() const override { return nullptr; }
        float* get_biases() const override { return nullptr; }
        float* get_weight_grad() const override { return nullptr; }
        float* get_bias_grad() const override { return nullptr; }
        float* get_input_grad() const override { return input_gradients; }
        float* get_output_grad() const override { return output_error; }
        float* set_pre_activation_values(float* values) override { return nullptr; }
        float* get_pre_activation_values() const override { return nullptr; }
        
        int get_input_size() const override;
        int get_output_size() const override;
        
        void clearBuffers() override;
        void init_sizes() override;
        
    private:
        int channels;
        int height;
        int width;
        float* input_gradients = nullptr;
    };
}
#endif // LAYERS_H