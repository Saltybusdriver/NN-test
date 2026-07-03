#ifndef CUDA_FUNCTIONS_H_
#define CUDA_FUNCTIONS_H_

#ifdef __cplusplus
extern "C" {
#endif
namespace cudafunc
{
    void launchActivation(float *device_z_values, float *device_activations, int arraySize, const char* activation_type);
    void forward_Propagate(
        const float* input,          
        float* weight,               
        float* biases,               
        float* pre_activation,       
        float* output,               // Final output after activation
        int batch_size,
        int input_size,
        int output_size,
        int activation_type
    );
    void backward_propagate(float* input, float* weights, float* output_error,
                       float* weight_grad, float* bias_grad, 
                       int batch_size, int input_size, int output_size);
    extern "C" void compute_output_error(
                        float* output,
                        const float* target,
                        float* error,
                        int batch_size,
                        int output_size,
                        const char* loss_type = "mse"
                        );
    extern "C" void calculate_loss_values(
            float* output,
            const float* target,
            float* element_loss,
            int size,
            const char* loss_type,
            int batch_size = 1
        );
    extern "C" float sum_loss_cuda(const float* loss_buffer, int num_elements);
    extern "C" float calculate_accuracy_cuda(
        const float* predictions,
        const float* targets,
        int batch_size,
        int num_classes
    );
    extern "C" void accumulate_loss_sum_cuda(
        const float* loss_buffer,
        float* primary_sum,
        float* secondary_sum,
        int num_elements
    );
    extern "C" void accumulate_correct_count_cuda(
        const float* predictions,
        const float* targets,
        int* primary_count,
        int* secondary_count,
        int batch_size,
        int num_classes
    );
    extern "C" void unflatten_gradient(
        const float* flattened_grad, float* unflattened_grad,
        int batch_size, int channels, int height, int width);
    
    // extern "C" void unflatten_gradient(
    // const float* flattened_grad, // Input: Gradient from next layer [B, C*H*W]
    // float* unflattened_grad,     // Output: Gradient for previous layer [B, C, H, W]
    // int batch_size,
    // int channels,
    // int height,
    // int width
    // );
    void compute_hidden_error(float* weights, float* output, float* hidden_error,
                         int batch_size, int input_size, int output_size);
    void update_params(float* weights, float* biases, float* weight_grad,
                  float* bias_grad, int batch_size, int input_size,
                  int output_size, float learning_rate);
    void calc_mse_loss_kernel(float* output, float* target, float* loss, int size);
    void add_vectors(float* out, const float* bias, int size);
    extern "C"  void batched_add_bias_kernel(
        float* output,
        const float* biases,
        int batch_size,
        int output_size
    );
    extern "C" void apply_linear_activation_derivative(
    float* output_error, 
    const float* pre_activation_output, 
    int batch_size, 
    int out_features, 
    int activation_type
);
    void mse_derivative(const float* output, const float* target, float* out_error, int size);

    void conv2d_forward(const float* input, const float* kernels, const float* bias,
                       float* pre_activation_output,float* output, float* d_im2col_cache,int batch_size, int in_channels, int out_channels,
                       int input_height, int input_width, int kernel_size, int stride, int padding,int activation_type);

    void conv2d_backward(
        const float* input,
        const float* kernels,
        float* pre_activation_output,
        float* output_grad,
        float* kernel_grad,
        size_t kernel_size_bytes,
        float* bias_grad,
        size_t bias_size,
        float* input_grad,
        float* d_im2col_cache, 
        int batch_size,
        int in_channels,
        int out_channels,
        int input_height,
        int input_width,
        int kernel_size,
        int stride,
        int padding,
        int activation_type
    );

    void sum_loss(const float* d_element_loss, float* d_out, int num_elements);
    void conv2d_compute_hidden_error(
    const float* weights,
    const float* output_grad,           // Input: dL/dY
    const float* pre_activation_output, // Input: Z
    float* hidden_error,
    size_t input_tensor_size_bytes,               // Output: dL/dX
    float* d_im2col_error_cache,   
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int output_height,               
    int output_width,                 
    int kernel_size,
    int stride,
    int padding,
    int activation_type,
    float** delta_cache_ptr,  // ADD THIS PARAMETER
    size_t* cache_size_ptr 
);

    void conv2d_update_params(
        float* weights,              // Convolutional kernels to update
        float* bias,                 // Biases to update
        const float* weight_grad,    // Gradient of loss w.r.t. weights
        const float* bias_grad,      // Gradient of loss w.r.t. biases
        int batch_size,
        int in_channels,
        int out_channels,
        int kernel_size,
        int stride,
        int padding,
        float learning_rate
    );

    // Add these to include/core/cuda_functions.h
extern "C" void custom_gemm(
    int M, int N, int K,
    const float* A, const float* B, float* C,
    bool transA, bool transB);

extern "C" void im2col(
    const float* data_im, int batch_size, int channels, int height, int width,
    int ksize, int stride, int pad, int height_col, int width_col, float* data_col);

extern "C" void col2im(
    const float* data_col, int batch_size, int channels, int height, int width,
    int ksize, int stride, int pad, int height_col, int width_col, float* data_im);
}
#ifdef __cplusplus
}
#endif

#endif /* CUDA_FUNCTIONS_H_ */
