#ifndef CPU_DEBUG_UTILS_H_
#define CPU_DEBUG_UTILS_H_

#include <vector>
#include <string>
#include <chrono>
#include <unordered_map>
#include "debug.h"

namespace cpu_debug {
// ====== Tensor Inspection Methods ======
void findHostArrayRange(const float* h_array, int size, float& min_val, float& max_val);
void calculateHostStats(const float* data, size_t size, float& min_val, float& max_val, float& mean_val, float& std_dev);

void inspectHostTensor(const char* name, const float* h_tensor, int size, bool force_print = false);
void inspectHostGradients(const char* name, const float* gradient_data, int size, bool show_full = false);
void compareHostValues(const char* name, const float* before, const float* after, int size, bool show_full = false);
void validateHostTensor(const float* tensor, size_t size, const char* name);
void inspectHostTensorFull(const char* name, const float* h_tensor, int size, int height, int width);
void compareHostTensors(const char* name1, const float* h_tensor1, const char* name2, const float* h_tensor2, int size);
void checkForNaNHost(const char* name, const float* h_tensor, int size);

void inspectHostVector(const char* label, const float* data_h, size_t size, bool print_sample = true, int sample_count = 10);
void inspectHostMatrix(const char* label, const float* data_h, size_t rows, size_t cols, bool print_sample = true, int sample_rows = 5, int sample_cols = 5);
void inspectHostBatchedVector(const char* label, const float* data_h, size_t batch_size, size_t vector_size, bool print_sample = true, int sample_batches = 3, int sample_elements = 5);
void inspectHost4DTensor(const char* name, const float* h_tensor,
    size_t dim1, size_t dim2, size_t dim3, size_t dim4,
    bool print_sample = true, int sample_count = 10);


// ====== Layer Operations ======
void logActivationHost(const char* layer_type, const char* activation,
                       const float* input, const float* output, int size);
void logGradientsHost(const char* layer_type, const float* gradients, int size);

// ====== Performance Monitoring (Generic) ======
extern std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> timers;
void startTimer(const char* operation_name);
void stopTimer(const char* operation_name);
double getElapsedTime(const std::string& timer_name);
void logLayerPerformance(const char* layer_type, float forward_ms, float backward_ms);


// ====== Training Progress (Generic) ======
void logWeightUpdateHost(const char* layer_name, const float* weights, int size, float learning_rate);

// ====== Helper Methods (Generic) ======
void horizontalLine(char symbol = '-', int length = 80);
void sectionHeader(const char* title);

} // namespace cpu_debug

#endif // CPU_DEBUG_UTILS_H_