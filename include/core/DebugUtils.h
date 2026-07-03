#ifndef DEBUG_UTILS_H
#define DEBUG_UTILS_H

#include "debug.h"
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <algorithm>

class DebugUtils {
public:
    // ====== Tensor Inspection Methods ======
    // Basic tensor statistics
    static void inspectTensor(const char* name, float* d_tensor, int size, bool force_print = false);
    static void inspectTensorFull(const char* name, float* d_tensor, int size, int height, int width);
    static void compareTensors(const char* name1, float* d_tensor1, const char* name2, float* d_tensor2, int size);
    static void checkForNaN(const char* name, float* d_tensor, int size);
    
    // Tensor shape printing
    static void printTensorShape(const char* name, int batch_size, int channels, int height, int width);
    static void printTensorShape(const char* name, int batch_size, int features);
    
    static void inspectVector(const char* label, float* data_d, size_t size, bool print_sample = true, int sample_count = 10);
    static void inspectMatrix(const char* label, float* data_d, size_t rows, size_t cols, bool print_sample = true, int sample_rows = 5, int sample_cols = 5);
    static void inspectBatchedVector(const char* label, float* data_d, size_t batch_size, size_t vector_size, bool print_sample = true, int sample_batches = 3, int sample_elements = 5);

    static void calculateStats(const std::vector<float>& data, float& min_val, float& max_val, float& mean_val, float& std_dev);


    // ====== Layer Operations ======
    // Forward pass debugging
    static void logLayerForward(const char* layer_type, int batch_size, int channels, int height, int width);
    static void logActivation(const char* layer_type, const char* activation, float* input, float* output, int size);

    // Backward pass debugging
    static void logLayerBackward(const char* layer_type, float learning_rate);
    static void logGradients(const char* layer_type, float* gradients, int size);
    
    static void VerifyLayerAligment(int total_output_size, int next_input_size,int layer_num);

    // Gradient inspection
    static void inspectGradients(const char* name, float* gradient_data, int size, bool show_full = false);
    static void compareValues(const char* name, float* before, float* after, int size, bool show_full = false);
    // Conv2D specific
    static void logConv2DOperation(int batch_size, int in_channels, int out_channels, 
                                  int input_height, int input_width, 
                                  int kernel_size, int stride, int padding);
    static void debugBottleneckLayer(int in_channels, int out_channels, int input_height, int input_width);
    static void debugBottleneckBackward(float* output_error, float* prev_output_error, int batch_size,
                                       int in_channels, int out_channels, 
                                       int input_height, int input_width,
                                       int output_height, int output_width);
    
    // FC layer specific
    static void logFCOperation(int batch_size, int in_features, int out_features);
    
    // ====== Memory Management ======
    static void logMemoryAllocation(const char* tensor_name, size_t size_bytes);
    static void logMemoryFree(const char* tensor_name);
    static void logCudaMemoryInfo();
    
    // ====== Performance Monitoring ======
    static void startTimer(const char* operation_name);
    static void stopTimer(const char* operation_name);
    static void logLayerPerformance(const char* layer_type, float forward_ms, float backward_ms);
    
    // ====== Training Progress ======
    static void logEpochStart(int epoch, int total_epochs);
    static void logEpochEnd(int epoch, float loss, float accuracy);
    static void logBatchProgress(int batch, int total_batches, float batch_loss);
    static void logWeightUpdate(const char* layer_name, float* weights, int size, float learning_rate);
    
    // ====== Error Handling ======
    static void logCudaError(cudaError_t error, const char* operation);
    static void logTrainingError(const char* message);
    static void ValidateTensor(float* tensor, size_t size, const char* name);
    
    // ====== Helper Methods ======
    static void horizontalLine(char symbol = '-', int length = 80);
    static void sectionHeader(const char* title);
    
private:
    // Helper methods
    static void findArrayRange(float* d_array, int size, float& min_val, float& max_val);
    static double getElapsedTime(const std::string& timer_name);
    
    static std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> timers;
};

#endif // DEBUG_UTILS_H