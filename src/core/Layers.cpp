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
    safeCudaMalloc(&stored_input, input_size_bytes, "Stored input");
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

    size_t input_size_bytes = batch_size * in_features * sizeof(float);
    
    if (stored_input_size < input_size_bytes) 
    {
        if (stored_input) 
        {
            safeCudaFree(&stored_input, "Stored input");
        }
        safeCudaMalloc(&stored_input, input_size_bytes, "Stored input");
        stored_input_size = input_size_bytes;
    }
    CUDA_CHECK_ERROR(cudaMemcpyAsync(stored_input, input, input_size_bytes, cudaMemcpyDeviceToDevice));
    
    if (cublas_handle == nullptr) 
    {
        cudaError_t cudaErr = cudaGetLastError();
        if (cudaErr != cudaSuccess) 
        {
            fprintf(stderr, "CUDA error before cuBLAS init: %s\n", cudaGetErrorString(cudaErr));
        }
        cudaFree(0);
        //REMOVE LATER?
        cublasStatus_t status = cublasCreate(&cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS) 
        {
            const char* errStr = "Unknown error";
            switch(status) 
            {
                case CUBLAS_STATUS_NOT_INITIALIZED: errStr = "CUBLAS_STATUS_NOT_INITIALIZED"; break;
                case CUBLAS_STATUS_ALLOC_FAILED: errStr = "CUBLAS_STATUS_ALLOC_FAILED"; break;
                default: errStr = "Other cuBLAS error";
            }
            fprintf(stderr, "cuBLAS handle creation failed: %s\n", errStr);
            throw std::runtime_error(std::string("Failed to create cuBLAS handle: ") + errStr);
        }
    }
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    if (debug_flag) 
    {
        INFO_PRINT("Matrix multiplication: [%d x %d] * [batch=%d, %d] -> [batch=%d, %d]\n",
                  out_features, in_features, batch_size, in_features, batch_size, out_features);
    }
    
    if (!weights || !input || !pre_activation_values) 
    {
        throw std::runtime_error("Null pointer in matrix multiplication");
    }

    cublasStatus_t status = cublasSgemm(
        cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        out_features, batch_size, in_features,
        &alpha, weights, out_features,
        input, in_features,
        &beta,
        pre_activation_values,
        out_features
    );

    if (status != CUBLAS_STATUS_SUCCESS) 
    {
        const char* error_msg;
        switch(status) {
            case CUBLAS_STATUS_NOT_INITIALIZED: error_msg = "CUBLAS_STATUS_NOT_INITIALIZED"; break;
            case CUBLAS_STATUS_INVALID_VALUE: error_msg = "CUBLAS_STATUS_INVALID_VALUE"; break;
            case CUBLAS_STATUS_ARCH_MISMATCH: error_msg = "CUBLAS_STATUS_ARCH_MISMATCH"; break;
            case CUBLAS_STATUS_EXECUTION_FAILED: error_msg = "CUBLAS_STATUS_EXECUTION_FAILED"; break;
            default: error_msg = "Unknown error";
        }
        throw std::runtime_error(std::string("CUBLAS error in batched forward pass: ") + error_msg);
    }

    cudafunc::forward_Propagate(
        input,
        weights,
        bias,
        pre_activation_values, 
        output, 
        batch_size,
        in_features,
        out_features,
        activation_type_id
    );
    if (this->output != output) 
    {
        cudaMemcpy(this->output, output, batch_size * out_features * sizeof(float), 
                  cudaMemcpyDeviceToDevice);
    }
}

void Linear::backward(const float* target, float* prev_output_error, float learning_rate,const char* loss_type) 
{
    if (!stored_input) 
    {
        throw std::runtime_error("Linear::backward: No stored input for backward pass");
    }
    if (!pre_activation_values) 
    {
        throw std::runtime_error("Linear::backward: No pre-activation values for backward pass");
    }

    if (target == nullptr && !prev_output_error) 
    {
         throw std::runtime_error("Linear::backward: prev_output_error is NULL for a hidden layer");
    }

    int in_size = in_features;
    int out_size = out_features; 
    size_t output_tensor_elements = batch_size * out_features;
    size_t input_tensor_elements = batch_size * in_features;
    size_t weight_elements = in_features * out_features;
    size_t bias_elements = out_features;
    size_t input_grad_size_bytes = input_tensor_elements * sizeof(float); 

    if (debug_flag) {
        DebugUtils::sectionHeader("LINEAR BACKWARD START");
        DebugUtils::inspectBatchedVector("Stored Input (X)", stored_input, batch_size, in_features);
        DebugUtils::inspectBatchedVector("Pre-activation (Z)", pre_activation_values, batch_size, out_features);
    }

    if (!weight_grad) safeCudaMalloc(&weight_grad, weight_elements * sizeof(float), "Weight grad buffer");
    if (!bias_grad) safeCudaMalloc(&bias_grad, bias_elements * sizeof(float), "Bias grad buffer");
    cudaMemset(weight_grad, 0, weight_elements * sizeof(float));
    cudaMemset(bias_grad, 0, bias_elements * sizeof(float));

    // --- PART 1: Calculate output error (dL/dY for output layer) ---
    if (target != nullptr) 
    {
        if (!output_error) 
        {
            safeCudaMalloc(&output_error, output_tensor_elements * sizeof(float), "Output error buffer");
        }

        cudafunc::compute_output_error(
            pre_activation_values,  
            target,     
            output_error, 
            batch_size,
            out_features,
            loss_type
        );
        cudaDeviceSynchronize(); 

        if (debug_flag) 
        {
            DebugUtils::inspectBatchedVector("Output Error (dL/dY) - Output Layer", output_error, batch_size, out_features);
        }
    }

    // --- PART 2: Calculate input gradients (dL/dX for previous layer) ---
    if (!input_gradients) 
    {
        safeCudaMalloc(&input_gradients, input_grad_size_bytes, "Input gradients buffer");
    }
    // Case 1: Hidden Layer (target is null
    if (target == nullptr) 
    {

        if (debug_flag) 
        {
             DebugUtils::inspectBatchedVector("Incoming Error (dL/dY from next)", prev_output_error, batch_size, out_features);
        }

        // Calculate dL/dX = W^T * (dL/dY_next)
        // Note: compute_hidden_error calculates W^T * error_input
        cudafunc::compute_hidden_error(
            weights,           
            prev_output_error, 
            input_gradients,    
            batch_size,
            in_features,  
            out_features 
        );
        cudaDeviceSynchronize();

        if (debug_flag) 
        {
            DebugUtils::inspectBatchedVector("Input Gradients (dL/dX) - Hidden Layer", input_gradients, batch_size, in_features);
        }
    }
    // Case 2: Output Layer (target is not null)
    else 
    {
        // Calculate dL/dX = W^T * (dL/dY_output)
        cudafunc::compute_hidden_error(
            weights,            
            output_error,       
            input_gradients,    
            batch_size,
            in_features,
            out_features
        );
        cudaDeviceSynchronize();

        if (debug_flag) 
        {
            DebugUtils::inspectBatchedVector("Input Gradients (dL/dX) - Output Layer", input_gradients, batch_size, in_features);
        }
    }

    // --- PART 3: Calculate weight/bias gradients (dL/dW, dL/dB) ---
    float* error_term_for_grads = (target != nullptr) ? output_error : prev_output_error;

    if (debug_flag) 
    {
        DebugUtils::inspectBatchedVector("Error Term for Weight/Bias Grads (dL/dY)", error_term_for_grads, batch_size, out_features);
    }
    // Calculate dL/dW and dL/dB
    cudafunc::backward_propagate(
        stored_input,        
        weights,            
        error_term_for_grads, 
        weight_grad,         
        bias_grad,    
        batch_size,
        in_features,
        out_features
    );
    cudaDeviceSynchronize(); 

    if (debug_flag) 
    {
        DebugUtils::sectionHeader("LINEAR FINAL GRADIENTS (Post Calculation)");
        DebugUtils::inspectMatrix("Weight Gradients (dL/dW)", weight_grad, in_features, out_features);
        DebugUtils::inspectVector("Bias Gradients (dL/dB)", bias_grad, bias_elements);

    }
     if (debug_flag) 
     {
        DebugUtils::sectionHeader("LINEAR BACKWARD END");
     }
}

void Linear::update_params(OptimizerBase& optimizer) 
{
    size_t weight_size = in_features * out_features;
    size_t bias_size = out_features;
    optimizer.update(weights, weight_grad, weight_size);
    optimizer.update(bias, bias_grad, bias_size);
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
    for (int ic = 0; ic < in_channels; ++ic) 
    {
        for (int oc = 0; oc < out_channels; ++oc) 
        {
            for (int i = 0; i < kernel_size; ++i) 
            {
                for (int j = 0; j < kernel_size; ++j) 
                {
                    int idx = ic * (out_channels * kernel_size * kernel_size)
                            + oc * (kernel_size * kernel_size)
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
    for (int ic = 0; ic < in_channels; ++ic) 
    {
        for (int oc = 0; oc < out_channels; ++oc) 
        {
            for (int i = 0; i < kernel_size; ++i) 
            {
                for (int j = 0; j < kernel_size; ++j) 
                {
                    int idx = ic * (out_channels * kernel_size * kernel_size)
                            + oc * (kernel_size * kernel_size)
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
  
    safeCudaMalloc(&stored_input, input_size_bytes, "Stored input");
    safeCudaMalloc(&kernels, kernel_size_bytes,"Kernels");
    safeCudaMalloc(&bias, bias_size,"Bias");
    safeCudaMalloc(&kernel_grad, kernel_size_bytes,"Kernel grad");
    safeCudaMalloc(&bias_grad, bias_size,"Bias grad");
    safeCudaMalloc(&pre_activation_values, output_size_bytes,"Pre-activation values");
    safeCudaMalloc(&input_gradients, input_size * sizeof(float), "Input gradients");
 

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
    if (stored_input_size < expected_input_size) 
    {
        throw std::runtime_error("Conv2d: Input buffer too small");
    }

    if(DebugLevel::LEVEL_TRACE >= current_debug_level)
    {
        TRACE_COUT("Kernels forward before.");
        DebugUtils::inspectTensorFull("Kernels", kernels, kernel_tensor_size, kernel_size, kernel_size);
        
        TRACE_COUT("kernel grad forward before.");
        DebugUtils::inspectTensorFull("Kernel grads", kernel_grad, kernel_tensor_size, kernel_size, kernel_size);
    }
       

    CUDA_CHECK_ERROR(cudaMemcpy(stored_input, input, input_size_bytes, cudaMemcpyDeviceToDevice));
    if (debug_flag)
    {
        checkPointerAlignment(stored_input, "Stored input");
    }
    cudaDeviceSynchronize();


    cudafunc::conv2d_forward(input, kernels, bias,pre_activation_values, output,
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
    
    if (!pre_activation_values) 
    {
        throw std::runtime_error("No pre-activation buffer available for backward pass");
    }
    if (!stored_input) 
    {
        throw std::runtime_error("No stored input available for backward pass");
    }
    if (!output_error && target != nullptr)
    {
         size_t output_tensor_size_bytes = batch_size * out_channels * output_height * output_width * sizeof(float);
         safeCudaMalloc(&output_error, output_tensor_size_bytes, "Output error buffer");
    }
    if (!input_gradients)
    {
        size_t input_tensor_size_bytes = batch_size * in_channels * input_height * input_width * sizeof(float);
        safeCudaMalloc(&input_gradients, input_tensor_size_bytes, "Input gradients buffer");
    }


    int out_h = output_height;
    int out_w = output_width;
    size_t output_tensor_size = batch_size * out_channels * output_height * output_width;
    size_t input_tensor_size = batch_size * in_channels * input_height * input_width;
    size_t input_tensor_size_bytes = input_tensor_size * sizeof(float); 

    cudaMemset(kernel_grad, 0, kernel_size_bytes);
    cudaMemset(bias_grad, 0, bias_size);

    // PART 1: Calculate output error (dL/dY for the current layer) - only for the final output layer
    if (target != nullptr) 
    {
        cudaDeviceSynchronize();
        cudafunc::compute_output_error(
            output, target, output_error, batch_size,
            out_channels * out_h * out_w, loss_type
        );
        
        if (debug_flag) 
        {
            DebugUtils::ValidateTensor(output_error, output_tensor_size, "Output Error (Post Calculation)");
            // DebugUtils::inspectTensorFull("Output error", output_error, output_tensor_size, 4, 4); // Optional detailed view
        }
    }
    
    // PART 2: Calculate input gradients (dL/dX for the current layer)
    // Case 1: Hidden Layer (target is null)
    if (target == nullptr) 
    {
        if (prev_output_error == nullptr) 
        {
             throw std::runtime_error("Conv2d::backward received null prev_output_error for a hidden layer");
        }
        

        if (debug_flag) 
        {
            INFO_COUT("Computing hidden error (dL/dX)");
            DebugUtils::inspectTensor("INCOMING_GRADIENTS (dL/dY from next layer)", prev_output_error, output_tensor_size, false);
        }
        
        // Calculate dL/dX using dL/dY from the next layer (prev_output_error) and store in temp_input_grad
        cudafunc::conv2d_compute_hidden_error(
            kernels,                   
            prev_output_error,          
            pre_activation_values,  
            input_gradients,          
            batch_size, in_channels, out_channels,
            input_height, input_width,
            output_height, output_width, 
            kernel_size, stride, padding,
            activation_type_id       
        );
        cudaDeviceSynchronize(); 
        if (debug_flag) 
        {
             DebugUtils::ValidateTensor(input_gradients, input_tensor_size, "Input Gradients (dL/dX) - Hidden Layer");
        }
    }
    // Case 2: Output Layer (target is not null)
    else 
    {
        cudaMemset(input_gradients, 0, input_tensor_size_bytes); 
        cudafunc::conv2d_compute_hidden_error(
            kernels,                    
            output_error,               
            pre_activation_values,    
            input_gradients,   
            batch_size, in_channels, out_channels,
            input_height, input_width,
            output_height, output_width, 
            kernel_size, stride, padding,
            activation_type_id     
        );
        cudaDeviceSynchronize();

        if (debug_flag) 
        {
             DebugUtils::ValidateTensor(input_gradients, input_tensor_size, "Input Gradients (dL/dX) - Output Layer");
        }
    }
    
    // PART 3: Calculate weight/bias gradients (dL/dK, dL/dB) for the current layer
    cudaDeviceSynchronize();

    float* error_term_for_grads = (target != nullptr) ? output_error : prev_output_error;

    cudafunc::conv2d_backward(
        stored_input,         
        kernels,               
        pre_activation_values,
        error_term_for_grads,
        kernel_grad,          
        bias_grad,      
        nullptr,  
        batch_size, in_channels, out_channels,
        input_height, input_width, kernel_size, stride, padding,
        activation_type_id
    );
    cudaDeviceSynchronize(); 

    if (debug_flag) 
    {
        DebugUtils::sectionHeader("CONV2D FINAL GRADIENTS (Post Calculation)");
        DebugUtils::inspectTensor("KERNEL_GRADIENTS (dL/dK)", kernel_grad, kernel_tensor_size, false);
        DebugUtils::inspectTensor("BIAS_GRADIENTS (dL/dB)", bias_grad, out_channels, false);
        DebugUtils::inspectTensor("INPUT_GRADIENTS (dL/dX)", input_gradients, input_tensor_size, false); // Re-inspect dL/dX
        // DebugUtils::logGradients("Conv2d final dL/dX", input_gradients, input_tensor_size); // Optional logging
        // DebugUtils::printTensorShape("Final dL/dX shape", batch_size, in_channels, input_height, input_width); // Optional shape check
    }
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
    safeCudaFree(&stored_input, "Conv2d stored input");
}

void Linear::clearBuffers() 
{
    safeCudaFree(&weights, "Linear weights");
    safeCudaFree(&bias, "Linear bias");
    safeCudaFree(&weight_grad, "Linear weight gradients");
    safeCudaFree(&bias_grad, "Linear bias gradients");
    safeCudaFree(&pre_activation_values, "Linear pre-activation");
    safeCudaFree(&input_gradients, "Linear input gradients");
    safeCudaFree(&stored_input, "Linear stored input");
}


Linear::~Linear() 
{
    if (stored_input) 
    {
        cudaFree(stored_input);
        stored_input=nullptr;
    }
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
