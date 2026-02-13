#ifndef DEBUG_H
#define DEBUG_H

#include <iostream>
#include <cstdio>
#include <fstream>     
#include <sstream> 
#include <string>       
#include <cuda_runtime.h> 


// Debug levels
enum DebugLevel {
    LEVEL_NONE = 0,    // No debug output
    LEVEL_ERROR = 1,   // Critical errors only
    LEVEL_WARN = 2,    // Warnings and errors
    LEVEL_INFO = 3,    // General information
    LEVEL_DEBUG = 4,   // Detailed debug information
    LEVEL_TRACE = 5    // Most verbose, detailed execution tracing
};

extern DebugLevel current_debug_level;
extern bool debug_flag;

inline void set_debug_level(DebugLevel level) 
{
    current_debug_level = level;
    if(level>LEVEL_DEBUG)
    {
        debug_flag=true;
    }
    else
    {
        debug_flag=false;
    }
}

inline float get_peak_cpu_memory_mb() 
{
#ifdef __linux__
    std::ifstream status_file("/proc/self/status");
    if (!status_file.is_open()) {
        return -1.0f;
    }
    std::string line;
    while (std::getline(status_file, line)) 
    {
        if (line.rfind("VmHWM:", 0) == 0) 
        { 
            std::stringstream ss(line);
            std::string key;
            long value;
            std::string unit;
            ss >> key >> value >> unit;
            if (unit == "kB") 
            {
                return static_cast<float>(value) / 1024.0f;
            } 
            else 
            {
                return static_cast<float>(value); 
            }
        }
    }
    return -1.0f; // VmHWM not found
#else
    return -1.0f; // Not implemented
#endif
}

// --- Helper Function: Get Current GPU Memory Usage ---
// Returns currently used GPU memory in MB, or -1.0 if CUDA error or not using CUDA.
inline float get_current_gpu_memory_usage_mb(bool usec) 
{
    if (!usec) 
    {
        return 0.0f; 
    }
    size_t free_mem, total_mem;
    cudaError_t status = cudaMemGetInfo(&free_mem, &total_mem);
    if (status != cudaSuccess) 
    {
        std::cerr << "Warning: cudaMemGetInfo failed: " << cudaGetErrorString(status) << std::endl;
        return -1.0f;
    }
    size_t used_mem = total_mem - free_mem;
    return static_cast<float>(used_mem) / (1024.0f * 1024.0f);
}

// Debug print macros for different levels
#define ERROR_PRINT(fmt, ...) \
    do { if (current_debug_level >= LEVEL_ERROR) fprintf(stderr, "[ERROR] " fmt, __VA_ARGS__); } while (0)

#define WARN_PRINT(fmt, ...) \
    do { if (current_debug_level >= LEVEL_WARN) fprintf(stderr, "[WARN] " fmt, __VA_ARGS__); } while (0)

#define INFO_PRINT(fmt, ...) \
    do { if (current_debug_level >= LEVEL_INFO) fprintf(stderr, "[INFO] " fmt, __VA_ARGS__); } while (0)

#define DEBUG_PRINT(fmt, ...) \
    do { if (current_debug_level >= LEVEL_DEBUG) fprintf(stderr, "[DEBUG] " fmt, __VA_ARGS__); } while (0)

#define TRACE_PRINT(fmt, ...) \
    do { if (current_debug_level >= LEVEL_TRACE) fprintf(stderr, "[TRACE] " fmt, __VA_ARGS__); } while (0)

#define ERROR_COUT(msg) \
    do { if (current_debug_level >= LEVEL_ERROR) std::cout << std::endl << "[ERROR] " << msg; } while (0)

#define WARN_COUT(msg) \
    do { if (current_debug_level >= LEVEL_WARN) std::cout << std::endl << "[WARN] " << msg; } while (0)

#define INFO_COUT(msg) \
    do { if (current_debug_level >= LEVEL_INFO) std::cout << std::endl << "[INFO] " << msg; } while (0)

#define DEBUG_COUT(msg) \
    do { if (current_debug_level >= LEVEL_DEBUG) std::cout << std::endl << "[DEBUG] " << msg; } while (0)

#define TRACE_COUT(msg) \
    do { if (current_debug_level >= LEVEL_TRACE) std::cout << std::endl << "[TRACE] " << msg; } while (0)

#define CUDA_CHECK_ERROR(call) \
    do { \
        cudaError_t err = call; \
        cudaError_t sync_err = cudaDeviceSynchronize(); \
        if (err != cudaSuccess || sync_err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in file '%s' in line %i:\n", __FILE__, __LINE__); \
            fprintf(stderr, "Error for call: %s\n", cudaGetErrorString(err)); \
            fprintf(stderr, "Error after sync: %s\n", cudaGetErrorString(sync_err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#endif // DEBUG_H