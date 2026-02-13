#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>    // For printf
#include <cstdlib>   // For EXIT_FAILURE and exit
#include "cuda_functions.h"
#include <ostream>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cassert>
#include <cublas_v2.h>
#include <curand.h>
#include <random>
#include "NeuralNetwork.h"
#include "debug.h"
#include <chrono>
#include "utils.h"
#include "test_nn.h"
void test_network()
{

    printf("Starting test_network...\n");
    int batch_size = 1;
    int image_height = 4;
    int image_width = 4;
    
    NeuralNetwork nn(batch_size);

    int in_channels = 1; 
    int out_channels = 1; 
    int kernel_size = 3;
    int stride = 1;
    int padding = 1;

    nn.add_layer( Conv2d(in_channels, out_channels, image_height, image_width, kernel_size, stride, padding, batch_size, "relu"));
    
    std::vector<float> host_weights = 
    {
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f
    };
     
  
    nn.layers[0]->set_weights(host_weights.data());
    std::vector<float> host_biases = {0.0f};
    nn.layers[0]->set_biases(host_biases.data());

    std::vector<float> input = {
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f
    };
    size_t input_size = image_height * image_width;
    size_t target_size = image_height * image_width;

    std::vector<float> target = {
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f
    };
    printf("Input size: %zu, Target size: %zu\n", input.size(), target.size());

    std::vector<float> expected_output = {
        4.0f, 6.0f, 6.0f, 4.0f,
        6.0f, 9.0f, 9.0f, 6.0f,
        6.0f, 9.0f, 9.0f, 6.0f,
        4.0f, 6.0f, 6.0f, 4.0f
    };

    std::vector<float> expected_output_err = {
        0.375f, 0.625f, 0.625f, 0.375f,
        0.625f, 1.000f, 1.000f, 0.625f,
        0.625f, 1.000f, 1.000f, 0.625f,
        0.375f, 0.625f, 0.625f, 0.375f
    };
    float* d_input;
    ToDevice(&d_input, input,"input");
    DEBUG_PRINT("Input allocated at: %p, size: %zu bytes\n", d_input, input.size() * sizeof(float));

    float* d_target;
    ToDevice(&d_target, target,"target");
    DEBUG_PRINT("Target allocated at: %p, size: %zu bytes\n", d_target, target.size() * sizeof(float));
    

    nn.forward(d_input,1, true);

    std::vector<float> output(target_size);
    ToHost(output, nn.layers.back()->output);

    printTensor2D("forward pass output",output, image_height, image_width);
    printTensor2D("Expected forward pass output",expected_output, image_height, image_width);

    bool all_passed = true;
    for (size_t i = 0; i < nn.layers.back()->output_size; ++i) 
    {
        if (fabs(output[i] - expected_output[i]) >= 1e-5) 
        {
            all_passed = false;
            std::cerr << "Assertion failed at index " << i << ": "
                    << "output = " << output[i] 
                    << ", expected = " << expected_output[i] << std::endl;
        }
    }

    
    nn.backward(d_target, 0.001f,"mse");


    std::vector<float> expected_kernel_grad = //calculated via sliding 3x3 kernel over out_error tensor
    {
        6.8750, 8.5000, 6.8750, 
        8.5000, 10.5000, 8.5000,
        6.8750, 8.5000, 6.8750
    };

    std::vector<float> expected_bias_grad = { 10.5f }; //sum of out_error tensor

    std::vector<float> kernel_grad(expected_kernel_grad.size());

    std::vector<float> output_error(expected_output.size());
    std::vector<float> bias_grad(expected_bias_grad.size());
    
    ToHost(output_error, nn.layers[0]->get_output_grad());
    ToHost(kernel_grad, nn.layers[0]->get_weight_grad());
    ToHost(bias_grad, nn.layers[0]->get_bias_grad());
    
   /*  std::vector<float> output_grad(output.size());
    ToHost(output_grad, nn.layers[0]->get_output_grad());

    printTensor2D("Output gradient", output_grad, image_height, image_width);
 */

    printTensor2D("Output error", output_error, image_height, image_width);
    printTensor2D("Kernel gradient",kernel_grad, 3, 3);
    printTensor2D("Expected kernel gradient",expected_kernel_grad, 3, 3);
    printf("Bias gradient: %f\n", bias_grad[0]);
    printf("Expected bias gradient: %f\n", expected_bias_grad[0]);
    bool gradients_passed = true;
    
    for (size_t i = 0; i < nn.layers.back()->output_size; ++i) 
    {
        if (fabs(output_error[i] - expected_output_err[i]) >= 1e-5) 
        {
            all_passed = false;
            std::cerr << "Assertion failed at index " << i << ": "
                    << "output = " << output[i] 
                    << ", expected = " << expected_output[i] << std::endl;
        }
    }
    for (size_t i = 0; i < expected_kernel_grad.size(); ++i) {
        if (fabs(kernel_grad[i] - expected_kernel_grad[i]) >= 1e-5) {
            gradients_passed = false;
            std::cerr << "Kernel gradient assertion failed at index " << i << ": "
                      << "gradient = " << kernel_grad[i] 
                      << ", expected = " << expected_kernel_grad[i] << std::endl;
        }
    }

    for (size_t i = 0; i < expected_bias_grad.size(); ++i) {
        if (fabs(bias_grad[i] - expected_bias_grad[i]) >= 1e-5) {
            gradients_passed = false;
            std::cerr << "Bias gradient assertion failed at index " << i << ": "
                      << "gradient = " << bias_grad[i] 
                      << ", expected = " << expected_bias_grad[i] << std::endl;
        }
    }
  
    if (all_passed) 
    {
        std::cout << "All assertions passed! Success!" << std::endl;
    } 
    else 
    {
        std::cerr << "Some assertions failed! Fail!" << std::endl;
    }
}