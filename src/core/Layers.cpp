#include "layers.h"
#include <cstdio>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include "cuda_functions.h"
#include <cublas_v2.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstring>
#include "debug.h"
#include "optimizer.h"  
#include "utils.h"
#include "DebugUtils.h"
#include "autodiff.h"

using namespace Cuda;

Linear::Linear(int in_feat, int out_feat, int b_size, const char* activation) 
    : Layer(activation),
      in_features(in_feat), 
      out_features(out_feat),
      weights(nullptr),
      bias(nullptr),
      weight_grad(nullptr),
      bias_grad(nullptr),
      cublas_handle(nullptr)  {
    
    if (in_feat <= 0 || out_feat <= 0 || b_size <= 0) 
    {
        throw std::invalid_argument("Invalid dimensions");
    }

    batch_size = b_size;
    init_sizes();
    allocate_buffers();
    init_weights();
    init_biases();
    
    INFO_PRINT("Linear layer initialized with %d input features and %d output features\n", 
           in_features, out_features);
}
void Linear::init_weights() 
{
    std::vector<float> h_weights(in_features * out_features);

    std::random_device rd;
    std::mt19937 gen(rd());


    float stddev = std::sqrt(2.0f / static_cast<float>(in_features));
    std::normal_distribution<float> dist(0.0f, stddev);

    for(size_t i = 0; i < h_weights.size(); i++) 
    {
        h_weights[i] = dist(gen);
    }

    CUDA_CHECK_ERROR(cudaMemcpy(weights, h_weights.data(), in_features * out_features * sizeof(float), cudaMemcpyHostToDevice));

}
void Linear::init_biases() 
{
    std::vector<float> h_bias(out_features, 0.0f);
    CUDA_CHECK_ERROR(cudaMemcpy(bias, h_bias.data(), out_features * sizeof(float), cudaMemcpyHostToDevice));
   
}
void Linear::init_sizes() 
{
    input_size = in_features*batch_size;
    output_size = out_features*batch_size;
}
void Linear::allocate_buffers() 
{
    Layer::allocate_buffers();
    
    size_t weight_size = in_features * out_features * sizeof(float);
    size_t bias_size = out_features * sizeof(float);
    size_t input_size_bytes = batch_size * in_features * sizeof(float);
    size_t output_size_bytes = batch_size * out_features * sizeof(float);
    
    safeCudaMalloc(&weights, weight_size, "Weights");
    safeCudaMalloc(&bias, bias_size, "Bias");
    safeCudaMalloc(&weight_grad, weight_size, "Weight grad");
    safeCudaMalloc(&bias_grad, bias_size, "Bias grad");
    // safeCudaMalloc(&stored_input, input_size_bytes, "Stored input"); // Eliminated!
    safeCudaMalloc(&pre_activation_values, output_size_bytes, "Pre-activation values");
    safeCudaMalloc(&input_gradients, input_size_bytes, "Input gradients");
    
    stored_input_size = input_size_bytes;
    preact_buffer_size = output_size_bytes;

    if (debug_flag) 
    {
        DebugUtils::horizontalLine(30);
        checkPointerAlignment(weights, "Weights");
        checkPointerAlignment(bias, "Bias");
        checkPointerAlignment(weight_grad, "Weight grad");
        checkPointerAlignment(bias_grad, "Bias grad");
        checkPointerAlignment(stored_input, "Stored input");
        checkPointerAlignment(pre_activation_values, "Pre-activation values");
        DebugUtils::horizontalLine(30);
    }
}

void Linear::forward(const float* input, float* output) 
{
    // NO MEMCOPY: We just store the pointer because 'input' memory is kept alive by the previous layer or batch cache
    this->stored_input = const_cast<float*>(input);

    // forward_Propagate is declared in cuda_functions.h, implemented in LinearLayers.cu
    // which internally only calls our batched custom_gemm
    cudafunc::forward_Propagate(
        this->stored_input,
        weights,
        bias,
        pre_activation_values,
        output,
        batch_size,
        in_features,
        out_features,
        activation_type_id
    );
}

void Linear::backward(const float* target, float* prev_output_error, float learning_rate, const char* loss_type) 
{
    if (batch_size <= 0) {
        fprintf(stderr, "Error: Batch size must be greater than 0.\n");
        return;
    }

    // 1. Setup output_error (dL/dY) for this layer safely
    if (target != nullptr) {
        cudafunc::compute_output_error(output, target, output_error, batch_size, out_features, loss_type);
    } else if (prev_output_error != nullptr) {
        // prev_output_error from the next layer is ALREADY out_features size. Just copy it!
        cudaMemcpy(output_error, prev_output_error, batch_size * out_features * sizeof(float), cudaMemcpyDeviceToDevice);
    } 

    if (debug_flag) {
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error after calculating output_error/target tracking: %s\n", cudaGetErrorString(err));
        }
    }

    // 2. Apply Activation Derivative natively to output_error
    if (activation_type_id != 4 && autodiff::loss::applies_activation_derivative(loss_type)) { // 4 is 'none'
        cudafunc::apply_linear_activation_derivative(
            output_error,
            pre_activation_values,
            batch_size,
            out_features,
            activation_type_id
        );
    }

    // 3. Compute Weight & Bias Gradients
    cudafunc::backward_propagate(
        stored_input,
        weights,
        output_error,
        weight_grad,
        bias_grad,
        batch_size,
        in_features,
        out_features
    );

    // 4. Compute input_gradients (dL/dX) to pass backward to the previous layer
    if (!input_gradients) {
        //safeCudaMalloc(&input_gradients, batch_size * in_features * sizeof(float), "Linear input gradients");
    }
    
    // Natively dumps into input_gradients! No temporary arrays needed!
    cudafunc::compute_hidden_error(
        weights, 
        output_error, 
        input_gradients, 
        batch_size, 
        in_features, 
        out_features
    );
    
    if (debug_flag) {
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error after compute_hidden_error: %s\n", cudaGetErrorString(err));
        }
    }
}

void Linear::update_params(OptimizerBase& optimizer) 
{
    // Ensure params are fully tied accurately 
    optimizer.update(weights, weight_grad, in_features * out_features);
    optimizer.update(bias, bias_grad, out_features);
}

float* Linear::set_weights(float* host_weights) 
{
    if (!weights) 
    {
        size_t weight_size = in_features * out_features * sizeof(float);
        CUDA_CHECK_ERROR(cudaMalloc(&weights, weight_size));
    }
    
    size_t weight_size = in_features * out_features * sizeof(float);
    CUDA_CHECK_ERROR(cudaMemcpy(this->weights, host_weights, weight_size, cudaMemcpyHostToDevice));
    
    return this->weights;
}

float* Linear::set_biases(float* host_bias) 
{
    if(bias)
    {
        cudaFree(bias);
        bias=nullptr;
    }
    if (!bias) 
    {
        size_t bias_size = out_features * sizeof(float);
        CUDA_CHECK_ERROR(cudaMalloc(&bias, bias_size));
    }
    
    size_t bias_size = out_features * sizeof(float);
    CUDA_CHECK_ERROR(cudaMemcpy(this->bias, host_bias, bias_size, cudaMemcpyHostToDevice));
    
    return this->bias;
}

Conv2d::Conv2d(int in_ch, int out_ch, int img_h, int img_w, int k_size, int str, int pad, int batch_sz, const char* activation)
    : Layer(activation),
      in_channels(in_ch),
      out_channels(out_ch),
      kernel_size(k_size),
      stride(str),
      padding(pad),
      kernels(nullptr),
      bias(nullptr),
      kernel_grad(nullptr),
      bias_grad(nullptr),
      d_delta_cache(nullptr),
      d_im2col_cache(nullptr),
      d_im2col_error_cache(nullptr),   
      delta_cache_size(0), 
      input_height(img_h),
      input_width(img_w),     
      output_height(0),
      output_width(0)
{
    
    if (in_ch <= 0 || out_ch <= 0 || k_size <= 0 || batch_sz <= 0) 
    {
        throw std::invalid_argument("Invalid Conv2d parameters");
    }

    batch_size = batch_sz;
    init_sizes();
    allocate_buffers();
    INFO_PRINT("Allocated buffers at output=%p, output_error=%p\n", output, output_error);
   
    init_weights();
    init_biases();

    //REMOVE LATER
    // init_weights_ones();
    // init_biases_zeros();
    
    if(DebugLevel::LEVEL_TRACE >= current_debug_level) 
    {
        TRACE_COUT("Kernels after weight init.");
        DebugUtils::inspectTensorFull("Kernels", kernels, kernel_tensor_size,kernel_size,kernel_size);
        
        TRACE_COUT("Kernel grads after weight init..");
        DebugUtils::inspectTensorFull("Kernel grads", kernel_grad, kernel_tensor_size,kernel_size,kernel_size);
    }
    
    INFO_PRINT("Conv2d initialized: kernels=%p (%zu bytes), bias=%p (%zu bytes)\n", 
           kernels, kernel_size_bytes, bias, bias_size);
}
void Conv2d::init_weights()
{

    float fan_in = static_cast<float>(in_channels * kernel_size * kernel_size);
    float fan_out = static_cast<float>(out_channels * kernel_size * kernel_size);
    float stddev = std::sqrt(2.0f / (fan_in + fan_out));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, stddev);

    std::vector<float> h_weights(in_channels * out_channels * kernel_size * kernel_size);
    for (int oc = 0; oc < out_channels; ++oc) 
    {
        for (int ic = 0; ic < in_channels; ++ic) 
        {
            for (int i = 0; i < kernel_size; ++i) 
            {
                for (int j = 0; j < kernel_size; ++j) 
                {
                    int idx = oc * (in_channels * kernel_size * kernel_size)
                            + ic * (kernel_size * kernel_size)
                            + i * kernel_size + j;
                    h_weights[idx] = dist(gen); 
                }
            }
        }
    }
    CUDA_CHECK_ERROR(cudaMemcpy(kernels, h_weights.data(), kernel_size_bytes, cudaMemcpyHostToDevice));
}
void Conv2d::init_weights_ones()
{
    std::vector<float> h_weights(in_channels * out_channels * kernel_size * kernel_size);
    for (int oc = 0; oc < out_channels; ++oc) 
    {
        for (int ic = 0; ic < in_channels; ++ic) 
        {
            for (int i = 0; i < kernel_size; ++i) 
            {
                for (int j = 0; j < kernel_size; ++j) 
                {
                    int idx = oc * (in_channels * kernel_size * kernel_size)
                            + ic * (kernel_size * kernel_size)
                            + i * kernel_size + j;
                    h_weights[idx] = 0.1f;
                }
            }
        }
    }
    CUDA_CHECK_ERROR(cudaMemcpy(kernels, h_weights.data(), kernel_size_bytes, cudaMemcpyHostToDevice));
}
void Conv2d::init_biases()
{
float fan_in = in_channels * kernel_size * kernel_size;
    float fan_out = out_channels * kernel_size * kernel_size;
    float xavier_scale = sqrt(2.0f / (fan_in + fan_out));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, xavier_scale);

    std::vector<float> h_bias(out_channels);
    std::uniform_real_distribution<float> bias_dist(-0.01f, 0.01f);
    for(size_t i = 0; i < out_channels; ++i) 
    {
        h_bias[i] = bias_dist(gen);
    }
    CUDA_CHECK_ERROR(cudaMemcpy(bias, h_bias.data(), bias_size, cudaMemcpyHostToDevice));
}
void Conv2d::init_biases_zeros()
{
    std::vector<float> h_bias(out_channels, 0.0f);
    CUDA_CHECK_ERROR(cudaMemcpy(bias, h_bias.data(), bias_size, cudaMemcpyHostToDevice));
}
void Conv2d::init_sizes() 
{
    int out_h = (input_height + 2 * padding - kernel_size) / stride + 1;
    int out_w = (input_width + 2 * padding - kernel_size) / stride + 1;

    input_size = in_channels * input_height * input_width*batch_size;
    output_size = out_channels * out_h * out_w*batch_size;
    kernel_tensor_size = in_channels * out_channels * kernel_size * kernel_size;

    kernel_size_bytes = out_channels * in_channels * kernel_size * kernel_size * sizeof(float);
    bias_size = out_channels * sizeof(float);

    INFO_PRINT("Conv2d: input_size=%d, output_size=%d\n", input_size, output_size);
}

void Conv2d::allocate_buffers() 
{
    Layer::allocate_buffers();
    output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    size_t input_size_bytes = batch_size * in_channels * input_height * input_width * sizeof(float);
    
    size_t kernel_size_bytes = in_channels * out_channels * kernel_size * kernel_size * sizeof(float);
    size_t bias_size = out_channels * sizeof(float);
    size_t output_size_bytes = batch_size * out_channels * output_height * output_width * sizeof(float);
    size_t im2col_bytes = (size_t)in_channels * kernel_size * kernel_size * 
                          batch_size * output_height * output_width * sizeof(float);
    size_t total_elements = get_total_output_size();
    size_t total_bytes = total_elements * sizeof(float);
    
    size_t weight_elements = out_channels * in_channels * kernel_size * kernel_size;
    size_t bias_elements = out_channels;

    delta_cache_size = output_size_bytes;

    // safeCudaMalloc(&stored_input, input_size_bytes, "Stored input"); // Eliminated!
    safeCudaMalloc(&kernels, kernel_size_bytes,"Kernels");
    safeCudaMalloc(&bias, bias_size,"Bias");
    safeCudaMalloc(&kernel_grad, kernel_size_bytes,"Kernel grad");
    safeCudaMalloc(&bias_grad, bias_size,"Bias grad");
    safeCudaMalloc(&pre_activation_values, output_size_bytes,"Pre-activation values");
    safeCudaMalloc(&input_gradients, input_size * sizeof(float), "Input gradients");
    safeCudaMalloc(&d_delta_cache, delta_cache_size, "Delta cache");
    safeCudaMalloc(&kernels, weight_elements * sizeof(float), "Conv2D Weights (CUDA)");
    safeCudaMalloc(&bias, bias_elements * sizeof(float), "Conv2D Biases (CUDA)");
    safeCudaMalloc(&kernel_grad, weight_elements * sizeof(float), "Conv2D W Grad (CUDA)");
    safeCudaMalloc(&bias_grad, bias_elements * sizeof(float), "Conv2D B Grad (CUDA)");
    safeCudaMalloc(&d_delta_cache, total_bytes, "Conv2D Delta Cache (CUDA)");
        
        // --> NEW: Safe Allocation of Cache
    safeCudaMalloc(&d_im2col_cache, im2col_bytes, "Im2Col Fwd Cache (CUDA)"); 
    safeCudaMalloc(&d_im2col_error_cache, im2col_bytes, "Im2Col Bwd Error Cache (CUDA)"); 
        

    if(debug_flag)
    {
        //printf("\n");
        DebugUtils::horizontalLine(30);
        checkPointerAlignment(kernels, "Kernels");
        checkPointerAlignment(bias, "Bias");
        checkPointerAlignment(kernel_grad, "Kernel grad");
        checkPointerAlignment(bias_grad, "Bias grad");
        INFO_PRINT("kernel_grad_size: %zu\n", kernel_size_bytes);
        INFO_PRINT("bias_grad_size: %zu\n", bias_size);
        DebugUtils::horizontalLine(30);
        //printf("\n");
    }

    if(DebugLevel::LEVEL_TRACE >= current_debug_level) 
    {
        TRACE_COUT("Kernels during alloc_buffer.");
        DebugUtils::inspectTensorFull("Kernels", kernels, kernel_tensor_size, kernel_size, kernel_size);
        DebugUtils::inspectTensorFull("Kernel grads", kernel_grad, kernel_tensor_size, kernel_size, kernel_size);
    }


    stored_input_size = input_size_bytes;
    preact_buffer_size = output_size_bytes;
}
void Conv2d::forward(const float* input, float* output) 
{
    size_t input_size_bytes = batch_size * in_channels * input_height * input_width * sizeof(float);
    size_t preact_bytes    = batch_size * out_channels * output_height * output_width * sizeof(float);
    size_t kernel_sizes = in_channels * out_channels * kernel_size * kernel_size;

    int expected_output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int expected_output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    
    std::vector<float> kernel_grad_t(kernel_tensor_size);


    if (output_height <= 0 || output_width <= 0) 
    {
        throw std::runtime_error("Conv2d: computed output dimensions are invalid");
    }
    if (output_height != expected_output_height || output_width != expected_output_width) 
    {
        throw std::runtime_error("Conv2d: Output dimensions mismatch");
    }
    size_t expected_input_size = batch_size * in_channels * input_height * input_width * sizeof(float);
    
    if(DebugLevel::LEVEL_TRACE >= current_debug_level)
    {
        TRACE_COUT("Kernels forward before.");
        DebugUtils::inspectTensorFull("Kernels", kernels, kernel_tensor_size, kernel_size, kernel_size);
        
        TRACE_COUT("kernel grad forward before.");
        DebugUtils::inspectTensorFull("Kernel grads", kernel_grad, kernel_tensor_size, kernel_size, kernel_size);
    }
       
    if (preact_buffer_size < preact_bytes) 
    {
        // Reallocate buffers gracefully
        if (pre_activation_values) cudaFree(pre_activation_values);
        // Free and reallocate any other batch-dependent caches here...
        
        cudaMalloc(&pre_activation_values, preact_bytes);
        
        preact_buffer_size = preact_bytes;
    }
    
    // NO MEMCOPY: We just store the pointer because 'input' memory is kept alive by the previous layer or batch cache
    this->stored_input = const_cast<float*>(input);
    stored_input_size = expected_input_size;
    
    if (debug_flag)
    {
        checkPointerAlignment(stored_input, "Stored input");
    }

    cudafunc::conv2d_forward(input, kernels, bias,pre_activation_values, output,d_im2col_cache,
                            batch_size, in_channels, out_channels,
                            input_height, input_width,
                            kernel_size, stride, padding,activation_type_id);
    if(DebugLevel::LEVEL_TRACE >= current_debug_level)
    {
        TRACE_COUT("Kernels forward after.");
        DebugUtils::inspectTensorFull("Kernels", kernels, kernel_tensor_size, kernel_size, kernel_size);
        
        TRACE_COUT("kernel grad forward after.");
        DebugUtils::inspectTensorFull("Kernel grads", kernel_grad, kernel_tensor_size, kernel_size, kernel_size);
    }
    
    DEBUG_PRINT("Conv2d: Forward pass completed (output dims: %d x %d)\n", output_height, output_width);
}

void Conv2d::backward(const float* target, float* prev_output_error, float learning_rate,const char* loss_type) {
    if (!pre_activation_values || !stored_input) throw std::runtime_error("Missing buffers");

    if (!output_error && target != nullptr) {
        printf("Allocating output_error buffer for Conv2d backward...\n");
         size_t output_tensor_size_bytes = batch_size * out_channels * output_height * output_width * sizeof(float);
         safeCudaMalloc(&output_error, output_tensor_size_bytes, "Output error buffer");
    }
    if (!input_gradients) {
        printf("Allocating input_gradients buffer for Conv2d backward...\n");
        size_t input_tensor_size_bytes = batch_size * in_channels * input_height * input_width * sizeof(float);
        safeCudaMalloc(&input_gradients, input_tensor_size_bytes, "Input gradients buffer");
    }

    size_t input_tensor_size_bytes = batch_size * in_channels * input_height * input_width * sizeof(float);
    //MOVE ALL MEMORY OPERATIONS TO KERNELS, PREFERABLY DONT USE MEMSET. KERNELS TO ZERO OUT IN PLACE
    cudaMemsetAsync(kernel_grad, 0, kernel_size_bytes);
    cudaMemsetAsync(bias_grad, 0, bias_size);

    // PART 1: Calculate output error (dL/dY) for the final layer
    if (target != nullptr) {
        cudafunc::compute_output_error(
            output, target, output_error, batch_size,
            out_channels * output_height * output_width, loss_type
        );
    }

    // This buffer holds the error entering the layer
    float* error_term_for_grads = (target != nullptr) ? output_error : prev_output_error;
    int effective_activation_type = (target != nullptr && !autodiff::loss::applies_activation_derivative(loss_type))
                                  ? 4
                                  : activation_type_id;

    // PART 2: CALCLUS FIX - Apply Activation Derivative AND Compute Weight Gradients
    // *Critical*: conv2d_backward securely mutates error_term_for_grads IN-PLACE to become dL/dZ !
    cudafunc::conv2d_backward(
        stored_input,         
        kernels,               
        pre_activation_values,
        error_term_for_grads,
        kernel_grad,
        kernel_size_bytes,          
        bias_grad,
        bias_size,
        nullptr, d_im2col_cache, 
        batch_size, in_channels, out_channels,
        input_height, input_width, kernel_size, stride, padding,
        effective_activation_type
    );

    // PART 3: Calculate input gradients (dL/dX) to pass to the previous layer
    // We now pass the MUTATED error_term_for_grads which is correctly dL/dZ!
    if (target == nullptr && prev_output_error == nullptr) throw std::runtime_error("Missing prev error");
    
    cudaMemsetAsync(input_gradients, 0, input_tensor_size_bytes);
    
    cudafunc::conv2d_compute_hidden_error(
        kernels,                   
        error_term_for_grads,      // NOW CORRECTLY dL/dZ!
        pre_activation_values,  
        input_gradients,input_tensor_size_bytes, d_im2col_cache,        
        batch_size, in_channels, out_channels,
        input_height, input_width,
        output_height, output_width, 
        kernel_size, stride, padding,
        static_cast<int>(activation_type_id),
        &d_delta_cache,
        &delta_cache_size 
    );
}

float* Conv2d::set_weights(float* host_weights) 
{
    size_t kernels_size = in_channels * out_channels * kernel_size * kernel_size * sizeof(float);
    safeCudaMalloc(&kernels, kernels_size, "Kernels");
    CUDA_CHECK_ERROR(cudaMemcpy(this->kernels, host_weights, kernels_size, cudaMemcpyHostToDevice));

    return this->kernels;
}

float* Conv2d::set_biases(float* host_bias) 
{
    size_t bias_size = out_channels * sizeof(float);
    safeCudaMalloc(&bias, bias_size, "Bias");
    CUDA_CHECK_ERROR(cudaMemcpy(this->bias, host_bias, bias_size, cudaMemcpyHostToDevice));

    return this->bias;
}

int Conv2d::get_input_size() const 
{
    return input_size;
}

int Conv2d::get_output_size() const 
{
    return output_size;
}



void Conv2d::update_params(OptimizerBase& optimizer) 
{
    size_t kernel_size = in_channels * out_channels * this->kernel_size * this->kernel_size;
    size_t bias_size = out_channels;
    
    optimizer.update(kernels, kernel_grad, kernel_size);
    optimizer.update(bias, bias_grad, bias_size);
}

Flatten::Flatten(int batch_size, int channels, int height, int width)
    : Layer("none"), channels(channels), height(height), width(width) 
{
    
    this->batch_size = batch_size;
    init_sizes();
    allocate_buffers();
    
    INFO_PRINT("Flatten constructed: batch_size=%d, channels=%d, height=%d, width=%d\n", 
                batch_size, channels, height, width);
    INFO_PRINT("Flattened dimensions: input_size=%d, output_size=%d\n", 
                input_size, output_size);
                
    size_t input_size_bytes = batch_size * channels * height * width * sizeof(float);
    safeCudaMalloc(&input_gradients, input_size_bytes, "Flatten input gradients");
}

void Flatten::init_sizes() 
{
    input_size = batch_size * channels * height * width;
    output_size = input_size;
}

void Flatten::forward(const float* input, float* output) 
{
    size_t total_bytes = batch_size * channels * height * width * sizeof(float);
    CUDA_CHECK_ERROR(cudaMemcpy(output, input, total_bytes, cudaMemcpyDeviceToDevice));
}

void Flatten::backward(const float* target, float* prev_output_error, float learning_rate, const char* loss_type) {
    if (!prev_output_error) 
    {
        ERROR_COUT("Flatten::backward received null prev_output_error\n");
        return;
   }

   if (!input_gradients) 
   {
        ERROR_COUT("Flatten::backward: input_gradients buffer is NULL\n");
        size_t input_size_bytes = batch_size * channels * height * width * sizeof(float);
        safeCudaMalloc(&input_gradients, input_size_bytes, "Flatten input gradients (recovery)");
        if (!input_gradients) throw std::runtime_error("Failed to allocate input_gradients in Flatten::backward");
   }

   cudafunc::unflatten_gradient(
    prev_output_error,
       input_gradients,
       batch_size,
       channels,
       height,
       width
   );
}

int Flatten::get_input_size() const 
{
    return input_size;
}

int Flatten::get_output_size() const 
{
    return output_size;
}

void Flatten::clearBuffers() 
{
    safeCudaFree(&input_gradients, "Flatten input gradients");
}

Flatten::~Flatten() 
{
    clearBuffers();
}


void Conv2d::clearBuffers() 
{
    safeCudaFree(&kernels, "Conv2d kernels");
    safeCudaFree(&bias, "Conv2d bias");
    safeCudaFree(&kernel_grad, "Conv2d kernel gradients");
    safeCudaFree(&bias_grad, "Conv2d bias gradients");
    safeCudaFree(&pre_activation_values, "Conv2d pre-activation");
    safeCudaFree(&input_gradients, "Conv2d input gradients");
    this->stored_input = nullptr;
    safeCudaFree(&d_delta_cache," Conv2d delta cache");
    safeCudaFree(&d_im2col_cache, "Im2Col Cache (CUDA)");             // <-- NEW
    safeCudaFree(&d_im2col_error_cache, "Im2Col Error Cache (CUDA)"); // <-- NEW
   
    
}

void Linear::clearBuffers() 
{
    safeCudaFree(&weights, "Linear weights");
    safeCudaFree(&bias, "Linear bias");
    safeCudaFree(&weight_grad, "Linear weight gradients");
    safeCudaFree(&bias_grad, "Linear bias gradients");
    safeCudaFree(&pre_activation_values, "Linear pre-activation");
    safeCudaFree(&input_gradients, "Linear input gradients");
    this->stored_input = nullptr;
}


Linear::~Linear() 
{
    this->stored_input = nullptr;
    if (weights) cudaFree(weights);
    if (bias) cudaFree(bias);
    if (weight_grad) cudaFree(weight_grad);
    if (bias_grad) cudaFree(bias_grad);
    if (cublas_handle) 
    {
        cublasDestroy(cublas_handle);
    }
}

Conv2d::~Conv2d() 
{
    clearBuffers();
    
}
