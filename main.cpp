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
#include <chrono>
#include "NeuralNetwork.h"
#include "debug.h"
#include "DataLoader.h"
#include "utils.h"
#include "test_nn.h"
#include "ImageDataset.h"
#include <opencv4/opencv2/opencv.hpp>
#include "autodiff.h"
#include "linear.h"
#include "conv.h"
#include "convlin.h"
#include "cpu_conv.h"
#include "cpu_linear.h"
#include "mnist_train.h"
#include "cifar_train.h"
#include "boston_regression.h"

#include "california_regression.h"
#include "cpu_mnist_train.h"
#include "cpu_cifar_train.h"
#include "cpu_boston_regression.h"
#include "cpu_california_regression.h"

DebugLevel current_debug_level = LEVEL_ERROR;
bool debug_flag=true;
#define USE_CUDA 1



enum ActivationFunction { NONE, SIGMOID, RELU, LEAKY_RELU, SOFTMAX };

//IMPLEMENT LATER 
/* bool validate_dimensions(const float* tensor, size_t expected_size) {
    if (!tensor) return false;
    size_t allocated_size;
    CUDA_CHECK_ERROR(cudaMemGetSize(&allocated_size, tensor));
    return allocated_size >= expected_size * sizeof(float);
} */


DEFINE_CUSTOM_LOSS({
    auto diff = o - t;
    auto squared_error = square(diff);
    auto weight = Constant(1.0f) + squared_error * Constant(0.5f);
    return weight * squared_error;
})

    
float calculate_custom_loss(const std::vector<float>& output, const std::vector<float>& target) {
    float total_loss = 0.0f;
    for (size_t i = 0; i < output.size(); i++) {
        auto loss_expr = autodiff::loss::CustomLoss::expression();
        total_loss += loss_expr.eval(output[i], target[i]);
    }
    return total_loss / output.size();
}

int main() 
{
    set_debug_level(LEVEL_WARN); // use cli args later.


    int batch_size = 256;
    int image_height = 28;
    int image_width = 28;
    int epochs = 5;
    bool use_grayscale = true;
    
    std::string dataset_path = "Dataset/";
    std::string mnist_dataset_path = "Dataset/";
    std::string cifar_dataset_path = "Dataset/CIFAR-10/";
    std::string boston_csv_path = "Dataset/Boston.csv";
    std::string california_csv_path = "Dataset/housing.csv";

    std::cout << "\n--- Starting Boston Housing Regression ---\n";
    boston_regression::train_boston_regressor(
        batch_size,
        epochs,
        boston_csv_path,
        0.001f,
        "mse" // Loss function: "mse" or "mae" or "huber"
    );
    cpu_boston_regression::train_boston_regressor(
        batch_size,
        epochs,
        boston_csv_path,
        0.001f,
        "mse" // Loss function: "mse" or "mae" or "huber"
    );
    std::cout << "\n--- Finished Boston Housing Regression ---\n";


    //--- Run California Housing Regression ---
    std::cout << "\n--- Starting California Housing Regression ---\n";
    california_regression::train_california_regressor(
        batch_size, // Can use a different batch size if desired
        epochs,     // Can use different epochs
        california_csv_path,
        0.001f,    // Learning rate
        "mse" // Loss function: "mse" or "mae" or "huber"
    );

    cpu_california_regression::train_california_regressor(
        batch_size, // Can use a different batch size if desired
        epochs,     // Can use different epochs
        california_csv_path,
        0.001f,    // Learning rate
        "mse" // Loss function: "mse" or "mae" or "huber"
    );
    std::cout << "\n--- Finished California Housing Regression ---\n";

    // // // Train the model
    cifar_train::train_cifar_classifier(batch_size, epochs, cifar_dataset_path, 32, 32, false);
    // // //cpu_cifar_train::train_cifar_classifier(batch_size, epochs, cifar_dataset_path, 32, 32, false);

    mnist_train::train_mnist_classifier(batch_size, epochs, mnist_dataset_path, image_height, image_width, use_grayscale);
    // //cpu_mnist_train::train_mnist_classifier(batch_size, epochs, mnist_dataset_path, image_height, image_width, use_grayscale);
    
    conv::train_conv_model(batch_size, epochs, dataset_path, image_height, image_width, use_grayscale);
    conv_cpu::train_conv_model(batch_size, epochs, dataset_path, image_height, image_width, use_grayscale);
 
    return 0;
}
