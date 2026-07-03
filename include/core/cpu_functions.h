#ifndef CPU_FUNCTIONS_H_
#define CPU_FUNCTIONS_H_

#include <cstddef> 

namespace cpu_func {

    // --- Activations ---
    void apply_activation_inplace(float* data, int size, int activation_type);

    // --- Linear Layer ---
    void add_vectors(float* out, const float* bias, int size);
    void forward_Propagate(
        const float* input, float* weight, float* biases,
        float* pre_activation, float* output,
        int batch_size, int input_size, int output_size, int activation_type);
    void compute_weight_gradients( 
        const float* input, const float* output_error, float* weight_grad,
        int batch_size, int input_size, int output_size);
    void compute_bias_gradients( 
        const float* output_error, float* bias_grad,
        int batch_size, int output_size);
    void backward_propagate(
        float* input, float* weights, float* output_error,
        float* weight_grad, float* bias_grad,
        int batch_size, int input_size, int output_size);
    void compute_hidden_error(
        float* weights, float* next_layer_error, float* hidden_error,
        int batch_size, int input_size, int output_size);
    void update_params(
        float* weights, float* biases, float* weight_grad, float* bias_grad,
        int batch_size, int input_size, int output_size, float learning_rate);

    // --- Convolutional Layer ---
    void conv2d_forward(
        const float* input, const float* kernels, const float* bias,
        float* pre_activation_output, float* output,
        int batch_size, int in_channels, int out_channels,
        int input_height, int input_width,
        int kernel_size, int stride, int padding, int activation_type);
    void conv2d_backward( // Calculates dL/dK and dL/dB
        const float* input,                 // 1. Input X
        const float* kernels,               // 2. Kernels W
        const float* pre_activation_output, // 3. Pre-activation Z
        const float* output_error,          // 4. Error term dL/dY (or dL/dY_next)
        float* kernel_grad,                 // 5. Output: dL/dK
        float* bias_grad,                   // 6. Output: dL/dB
        float* input_grad,                  // 7. Output: dL/dX (pass nullptr if not calculated here)
        int batch_size,                     // 8. Batch Size
        int in_channels,                    // 9. Input Channels
        int out_channels,                   // 10. Output Channels
        int input_height,                   // 11. Input Height
        int input_width,                    // 12. Input Width
        int kernel_size,                    // 13. Kernel Size
        int stride,                         // 14. Stride
        int padding,                        // 15. Padding
        int activation_type                 // 16. Activation Type ID
    );
    void conv2d_compute_hidden_error(
        const float* kernels,               // Kernels W (Layout: [in_channels, out_channels, kh, kw])
        const float* output_error,          // Input: dL/dY (Gradient from next layer or loss)
        const float* pre_activation_output, // Input: Z (Pre-activation values from forward pass)
        float* prev_output_error,           // Output: dL/dX 
        int batch_size,
        int in_channels,
        int out_channels,
        int input_height,
        int input_width,
        int out_height,                     
        int out_width,                  
        int kernel_size,
        int stride,
        int padding,
        int activation_type               
    );
    void conv2d_update_params(
        float* weights, float* bias, const float* weight_grad, const float* bias_grad,
        int batch_size, int in_channels, int out_channels, int kernel_size,
        int stride, int padding, float learning_rate);

    // --- Loss Functions ---
    void compute_output_error(
        const float* output, const float* target, float* error,
        int batch_size, int output_size, const char* loss_type);
    void calculate_loss_values(
        const float* output, const float* target, float* element_loss,
        int size, const char* loss_type, int batch_size);

    // --- Utility ---
    void unflatten_gradient(
        const float* flattened_grad, float* unflattened_grad,
        int batch_size, int channels, int height, int width);

} // namespace cpu_func

#endif // CPU_FUNCTIONS_H_