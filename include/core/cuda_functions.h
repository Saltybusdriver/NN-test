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
    extern "C" void unflatten_gradient(
    const float* flattened_grad, // Input: Gradient from next layer [B, C*H*W]
    float* unflattened_grad,     // Output: Gradient for previous layer [B, C, H, W]
    int batch_size,
    int channels,
    int height,
    int width
    );
    void compute_hidden_error(float* weights, float* output, float* hidden_error,
                         int batch_size, int input_size, int output_size);
    void update_params(float* weights, float* biases, float* weight_grad,
                  float* bias_grad, int batch_size, int input_size,
                  int output_size, float learning_rate);
    void calc_mse_loss_kernel(float* output, float* target, float* loss, int size);
    void add_vectors(float* out, const float* bias, int size);
    void mse_derivative(const float* output, const float* target, float* out_error, int size);

    void conv2d_forward(const float* input, const float* kernels, const float* bias,
                       float* pre_activation_output,float* output, int batch_size, int in_channels, int out_channels,
                       int input_height, int input_width, int kernel_size, int stride, int padding,int activation_type);

    void conv2d_backward(
        const float* input,
        const float* kernels,
        float* pre_activation_output,
        float* output_grad,
        float* kernel_grad,
        float* bias_grad,
        float* input_grad,
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


    void conv2d_compute_hidden_error(
    const float* weights,
    const float* output_grad,           // Input: dL/dY
    const float* pre_activation_output, // Input: Z
    float* hidden_error,                // Output: dL/dX
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
    int activation_type     
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
}
#ifdef __cplusplus
}
#endif

#endif /* CUDA_FUNCTIONS_H_ */
