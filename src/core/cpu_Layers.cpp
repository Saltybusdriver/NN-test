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
#include "cpu_utils.h"
#include "cpu_functions.h"



#include "cpu_DebugUtils.h"

using namespace cpu;

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
void Linear::init_weights() {
    std::vector<float> h_weights(in_features * out_features);

    std::random_device rd;
    std::mt19937 gen(rd());


    float stddev = std::sqrt(2.0f / static_cast<float>(in_features));
    std::normal_distribution<float> dist(0.0f, stddev);

    for(size_t i = 0; i < h_weights.size(); i++) 
    {
        h_weights[i] = dist(gen);
    }

    memcpy(weights, h_weights.data(), in_features * out_features * sizeof(float));
}
void Linear::init_biases() 
{
    std::vector<float> h_bias(out_features, 0.0f);
    memcpy(bias, h_bias.data(), out_features * sizeof(float));

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
    
    safeMalloc(&weights, weight_size, "Weights");
    safeMalloc(&bias, bias_size, "Bias");
    safeMalloc(&weight_grad, weight_size, "Weight grad");
    safeMalloc(&bias_grad, bias_size, "Bias grad");
    safeMalloc(&stored_input, input_size_bytes, "Stored input");
    safeMalloc(&pre_activation_values, output_size_bytes, "Pre-activation values");
    safeMalloc(&input_gradients, input_size_bytes, "Input gradients");
    
    stored_input_size = input_size_bytes;
    preact_buffer_size = output_size_bytes;

    if (debug_flag) 
    {
        // DebugUtils::horizontalLine(30);
        // checkPointerAlignment(weights, "Weights");
        // checkPointerAlignment(bias, "Bias");
        // checkPointerAlignment(weight_grad, "Weight grad");
        // checkPointerAlignment(bias_grad, "Bias grad");
        // checkPointerAlignment(stored_input, "Stored input");
        // checkPointerAlignment(pre_activation_values, "Pre-activation values");
        // DebugUtils::horizontalLine(30);
    }
}

void Linear::forward(const float* input, float* output) 
{
    size_t input_size_bytes = batch_size * in_features * sizeof(float);
    
    if (stored_input_size < input_size_bytes) 
    {
        if (stored_input) 
        {
            safeFree(&stored_input, "Stored input");
        }
        safeMalloc(&stored_input, input_size_bytes, "Stored input");
        stored_input_size = input_size_bytes;
    }
    
    
    memcpy(stored_input, input, input_size_bytes);
    
    
    
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

    for (int b = 0; b < batch_size; b++) 
    {
        for (int out_idx = 0; out_idx < out_features; out_idx++) 
        {
            float sum = 0.0f;
            for (int in_idx = 0; in_idx < in_features; in_idx++) 
            {
                sum += weights[out_idx * in_features + in_idx] * input[b * in_features + in_idx];
            }
            pre_activation_values[b * out_features + out_idx] = sum + bias[out_idx];
        }
    }
    cpu_func::forward_Propagate(
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
        memcpy(this->output, output, batch_size * out_features * sizeof(float));
    }
}

void Linear::backward(const float* target, float* prev_output_error, float learning_rate,const char* loss_type) {

    if (!stored_input) {
        throw std::runtime_error("Linear::backward: No stored input for backward pass");
    }
    if (!pre_activation_values) 
    {
        throw std::runtime_error("Linear::backward: No pre-activation values for backward pass");
    }
    // For hidden layers, prev_output_error must exist
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
 

    // --- Reset Gradients ---
    if (!weight_grad) safeMalloc(&weight_grad, weight_elements * sizeof(float), "Weight grad buffer");
    if (!bias_grad) safeMalloc(&bias_grad, bias_elements * sizeof(float), "Bias grad buffer");
    memset(weight_grad, 0, weight_elements * sizeof(float));
    memset(bias_grad, 0, bias_elements * sizeof(float));

    // --- PART 1: Calculate output error (dL/dY for output layer) ---
    if (target != nullptr) 
    {
        if (!output_error) 
        {
            safeMalloc(&output_error, output_tensor_elements * sizeof(float), "Output error buffer");
        }

        cpu_func::compute_output_error
        (
            pre_activation_values,
            target,        
            output_error,
            batch_size,
            out_features,
            loss_type
        );
    }

    // --- PART 2: Calculate input gradients (dL/dX for previous layer) ---
    if (!input_gradients) 
    {
        safeMalloc(&input_gradients, input_grad_size_bytes, "Input gradients buffer");
    }

    // Case 1: Hidden Layer (target is null)
    if (target == nullptr) 
    {
        // Calculate dL/dX = W^T * (dL/dY_next
        // compute_hidden_error calculates W^T * error_input
        cpu_func::compute_hidden_error(
            weights,           
            prev_output_error,  // dL/dY from next layer
            input_gradients,    // Output buffer for dL/dX (this layer's input grad)
            batch_size,
            in_features,        
            out_features 
        );
    }
    // Case 2: Output Layer (target is not null)
    else 
    {
        // Calculate dL/dX = W^T * (dL/dY_output)
        cpu_func::compute_hidden_error(
            weights,            // W
            output_error,       // dL/dY calculated in Part 1
            input_gradients,    // Output buffer for dL/dX (this layer's input grad)
            batch_size,
            in_features,
            out_features
        );
    }

    // --- PART 3: Calculate weight/bias gradients (dL/dW, dL/dB) ---
    float* error_term_for_grads = (target != nullptr) ? output_error : prev_output_error;
    // Calculate dL/dW and dL/dB
    cpu_func::backward_propagate(
        stored_input,
        weights,
        error_term_for_grads,
        weight_grad,       
        bias_grad,
        //nullptr,             
        batch_size,
        in_features,
        out_features
    );
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
        safeMalloc(&weights, weight_size, "Weights");
    }
    
    size_t weight_size = in_features * out_features * sizeof(float);
    memcpy(this->weights, host_weights, weight_size);
    
    return this->weights;
}

float* Linear::set_biases(float* host_bias) 
{
    if(bias)
    {
        safeFree(&bias, "Bias");
        bias = nullptr;
    }
    if (!bias) 
    {
        size_t bias_size = out_features * sizeof(float);
        safeMalloc(&bias, bias_size, "Bias");
    }
    
    size_t bias_size = out_features * sizeof(float);
    memcpy(this->bias, host_bias, bias_size);
    
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
        cpu_debug::inspectHostTensorFull("Kernels", kernels, kernel_tensor_size,kernel_size,kernel_size);

        TRACE_COUT("Kernel grads after weight init..");
        cpu_debug::inspectHostTensorFull("Kernel grads", kernel_grad, kernel_tensor_size,kernel_size,kernel_size);
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
    memcpy(kernels, h_weights.data(), kernel_size_bytes);
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
                    h_weights[idx] = 0.01f;
                }
            }
        }
    }
    memcpy(kernels, h_weights.data(), kernel_size_bytes);
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
    memcpy(bias, h_bias.data(), bias_size);
}
void Conv2d::init_biases_zeros()
{
    std::vector<float> h_bias(out_channels, 0.0f);
    memcpy(bias, h_bias.data(), bias_size);
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
  
    safeMalloc(&stored_input, input_size_bytes, "Stored input");
    safeMalloc(&kernels, kernel_size_bytes,"Kernels");
    safeMalloc(&bias, bias_size,"Bias");
    safeMalloc(&kernel_grad, kernel_size_bytes,"Kernel grad");
    safeMalloc(&bias_grad, bias_size,"Bias grad");
    safeMalloc(&pre_activation_values, output_size_bytes,"Pre-activation values");
    safeMalloc(&input_gradients, input_size * sizeof(float), "Input gradients");
 
    stored_input_size = input_size_bytes;
    preact_buffer_size = output_size_bytes;
}
void Conv2d::forward(const float* input, float* output) 
{
    size_t input_size_bytes = batch_size * in_channels * input_height * input_width * sizeof(float);
    size_t preact_bytes    = batch_size * out_channels * output_height * output_width * sizeof(float);
    size_t kernel_elements = out_channels * in_channels * kernel_size * kernel_size; 

    int expected_output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int expected_output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

    if (output_height <= 0 || output_width <= 0)
    {
        throw std::runtime_error("Conv2d: computed output dimensions are invalid");
    }
    if (output_height != expected_output_height || output_width != expected_output_width)
    {
        throw std::runtime_error("Conv2d: Output dimensions mismatch");
    }
    size_t expected_input_size_bytes = batch_size * in_channels * input_height * input_width * sizeof(float);
    if (stored_input_size < expected_input_size_bytes) 
    { 
        throw std::runtime_error("Conv2d: Input buffer too small");
    }

    if(debug_flag)
    {
        TRACE_COUT("Kernels forward before.");
        cpu_debug::inspectHostTensorFull("Kernels", kernels, kernel_tensor_size, kernel_size, kernel_size); // Needs 4D adaptation

        TRACE_COUT("kernel grad forward before.");
        cpu_debug::inspectHostTensorFull("Kernel grads", kernel_grad, kernel_tensor_size, kernel_size, kernel_size); // Needs 4D adaptation
    }




    memcpy(stored_input, input, input_size_bytes);

    // Perform convolution and activation
    cpu_func::conv2d_forward(input, kernels, bias, pre_activation_values, output,
        batch_size, in_channels, out_channels,
        input_height, input_width,
        kernel_size, stride, padding, activation_type_id);

    if (debug_flag) 
    {

        cpu_debug::inspectHost4DTensor("Forward Input (X)", input, batch_size, in_channels, input_height, input_width);
        cpu_debug::inspectHost4DTensor("Forward Kernels (W)", kernels, out_channels, in_channels, kernel_size, kernel_size);
        cpu_debug::inspectHostVector("Forward Bias (B)", bias, out_channels);
        cpu_debug::inspectHost4DTensor("Forward Pre-activation (Z)", pre_activation_values, batch_size, out_channels, output_height, output_width);
        cpu_debug::inspectHost4DTensor("Forward Output (Y)", output, batch_size, out_channels, output_height, output_width);

    }
    if(debug_flag)
    {
        TRACE_COUT("Kernels forward after.");
        cpu_debug::inspectHostTensorFull("Kernels", kernels, kernel_tensor_size, kernel_size, kernel_size); // Needs 4D adaptation

        TRACE_COUT("kernel grad forward after.");
        cpu_debug::inspectHostTensorFull("Kernel grads", kernel_grad, kernel_tensor_size, kernel_size, kernel_size); // Needs 4D adaptation
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
         safeMalloc(&output_error, output_tensor_size_bytes, "Output error buffer");

    }
    if (!input_gradients)
    {
        size_t input_tensor_size_bytes = batch_size * in_channels * input_height * input_width * sizeof(float);
        safeMalloc(&input_gradients, input_tensor_size_bytes, "Input gradients buffer");
    }


    int out_h = output_height;
    int out_w = output_width;
    size_t output_tensor_size = batch_size * out_channels * output_height * output_width;
    size_t input_tensor_size = batch_size * in_channels * input_height * input_width;
    size_t input_tensor_size_bytes = input_tensor_size * sizeof(float);

    // Reset weight/bias gradients
    // memset(kernel_grad, 0, kernel_size_bytes);
    // memset(bias_grad, 0, bias_size);
    if (debug_flag) 
    {
        cpu_debug::sectionHeader("CONV2D BACKWARD START");
        cpu_debug::inspectHost4DTensor("Stored Input (X)", stored_input, batch_size, in_channels, input_height, input_width);
        cpu_debug::inspectHost4DTensor("Pre-activation (Z)", pre_activation_values, batch_size, out_channels, output_height, output_width);
    }
    // PART 1: Calculate output error (dL/dY for the current layer) - only for the final output layer
    if (target != nullptr) 
    {
        cpu_func::compute_output_error(
            output,
            target,
            output_error,
            batch_size,
            out_channels * out_h * out_w,
            loss_type
        );

        if (debug_flag) 
        {
            cpu_debug::inspectHost4DTensor("Conv2d Backward: Calculated output_error (dL/dY)", output_error, batch_size, out_channels, output_height, output_width);
            cpu_debug::checkForNaNHost("Output Error (dL/dY) - Output Layer", output_error, output_tensor_size);
        }
    }

    // PART 2: Calculate input gradients (dL/dX for the current layer) - MATCHING GPU ORDER
    // Case 1: Hidden Layer (target is null)
    if (target == nullptr) 
    {
        if (prev_output_error == nullptr) {
             throw std::runtime_error("Conv2d::backward received null prev_output_error for a hidden layer");
        }

        if (debug_flag) {
            // INFO_COUT("Computing hidden error (dL/dX)");
            cpu_debug::checkForNaNHost("Incoming Error (dL/dY from next)", prev_output_error, output_size);
        }

        // Calculate dL/dX using dL/dY from the next layer (prev_output_error)
        // MATCHING GPU: Passing only kernels and error term
        cpu_func::conv2d_compute_hidden_error(
            kernels,                    // Layer's kernels (W)
            prev_output_error,          // Gradient from next layer (dL/dY)
            pre_activation_values,      // Pre-activation values (Z)
            input_gradients,            // Output buffer for dL/dX
            batch_size, in_channels, out_channels,
            input_height, input_width,
            output_height, output_width, 
            kernel_size, stride, padding,
            activation_type_id 
        );
        if (debug_flag) 
        {
            cpu_debug::checkForNaNHost("Input Gradients (dL/dX)", input_gradients, input_tensor_size);
        }
    }
    // Case 2: Output Layer (target is not null)
    else
    {
        // Calculate dL/dX using the layer's own output_error (dL/dY)
        // MATCHING GPU: Passing only kernels and error term

        // memset(input_gradients, 0, input_tensor_size_bytes); // Initialize before kernel
        cpu_func::conv2d_compute_hidden_error(
            kernels,                    // Layer's kernels (W)
            output_error,               // Gradient calculated in Part 1 (dL/dY)
            pre_activation_values,      // Pre-activation values (Z)
            input_gradients,            // Output buffer for dL/dX
            batch_size, in_channels, out_channels,
            input_height, input_width,
            output_height, output_width,
            kernel_size, stride, padding,
            activation_type_id       
        );
    }


    // PART 3: Calculate weight/bias gradients (dL/dK, dL/dB) for the current layer - MATCHING GPU ORDER
    float* error_term_for_grads = (target != nullptr) ? output_error : prev_output_error;

    cpu_func::conv2d_backward(
        stored_input,          // Input from forward pass (X)
        kernels,               // Layer's kernels (W)
        pre_activation_values, // Pre-activation values (Z)
        error_term_for_grads,  // Gradient w.r.t output (dL/dY)
        kernel_grad,           // Output buffer for dL/dK
        bias_grad,             // Output buffer for dL/dB
        nullptr,               // calculated input_grad (dL/dX) separately in Part 2
        batch_size, in_channels, out_channels,
        input_height, input_width, kernel_size, stride, padding,
        activation_type_id
    );

    if (debug_flag) 
    {
        cpu_debug::sectionHeader("CONV2D FINAL GRADIENTS (Post Calculation)");
        cpu_debug::inspectHostVector("Bias Gradients (dL/dB)", bias_grad, out_channels);
        cpu_debug::checkForNaNHost("Kernel Gradients (dL/dK)", kernel_grad, kernel_tensor_size);
        cpu_debug::checkForNaNHost("Bias Gradients (dL/dB)", bias_grad, out_channels);
        cpu_debug::logGradientsHost("Conv2d Kernels", kernel_grad, kernel_tensor_size);
        cpu_debug::logGradientsHost("Conv2d Bias", bias_grad, out_channels);
    }
     if (debug_flag) 
     {
        cpu_debug::sectionHeader("CONV2D BACKWARD END");
     }
}

float* Conv2d::set_weights(float* host_weights) 
{
    size_t kernels_size = in_channels * out_channels * kernel_size * kernel_size * sizeof(float);
    safeMalloc(&kernels, kernels_size, "Kernels");
    memcpy(this->kernels, host_weights, kernels_size);

    return this->kernels;
}

float* Conv2d::set_biases(float* host_bias) 
{
    size_t bias_size = out_channels * sizeof(float);
    safeMalloc(&bias, bias_size, "Bias");
    memcpy(this->bias, host_bias, bias_size);

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
    safeMalloc(&input_gradients, input_size_bytes, "Flatten input gradients");
}

void Flatten::init_sizes() 
{
    input_size = batch_size * channels * height * width;
    output_size = input_size;
}

void Flatten::forward(const float* input, float* output) 
{
    size_t total_bytes = batch_size * channels * height * width * sizeof(float);
    memcpy(output, input, total_bytes);
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
        safeMalloc(&input_gradients, input_size_bytes, "Flatten input gradients (recovery)");
        if (!input_gradients) throw std::runtime_error("Failed to allocate input_gradients in Flatten::backward");
   }

   cpu_func::unflatten_gradient(
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
    safeFree(&input_gradients, "Flatten input gradients");
}

Flatten::~Flatten() 
{
    clearBuffers();
}


void Conv2d::clearBuffers() 
{
    safeFree(&kernels, "Conv2d kernels");
    safeFree(&bias, "Conv2d bias");
    safeFree(&kernel_grad, "Conv2d kernel gradients");
    safeFree(&bias_grad, "Conv2d bias gradients");
    safeFree(&pre_activation_values, "Conv2d pre-activation");
    safeFree(&input_gradients, "Conv2d input gradients");
    safeFree(&stored_input, "Conv2d stored input");
}

void Linear::clearBuffers() 
{
    safeFree(&weights, "Linear weights");
    safeFree(&bias, "Linear bias");
    safeFree(&weight_grad, "Linear weight gradients");
    safeFree(&bias_grad, "Linear bias gradients");
    safeFree(&pre_activation_values, "Linear pre-activation");
    safeFree(&input_gradients, "Linear input gradients");
    safeFree(&stored_input, "Linear stored input");
}


Linear::~Linear() 
{
    if (stored_input) 
    {
        safeFree(&stored_input,"stored input");
        stored_input=nullptr;
    }
    if (weights) safeFree(&weights,"linear weights");
    if (bias) safeFree(&bias, "linear bias");
    if (weight_grad) safeFree(&weight_grad,"Weight gradinets");
    if (bias_grad) safeFree(&bias_grad, "Bias gradients");
}

Conv2d::~Conv2d() {
    clearBuffers();
    
}
