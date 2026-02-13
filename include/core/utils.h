#ifndef UTILS_H_
#define UTILS_H_

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include "debug.h"
#include <algorithm>
#include <opencv4/opencv2/opencv.hpp>

inline void printTensor2D(std::string name,const std::vector<float>& data, int height, int width) 
{
    if (data.size() != height * width) 
    {
        //throw std::invalid_argument("WARNING: Tensor dimensions don't match data size!\n");
    }
    std::cout << "\n "<<name<<" Tensor [" << height << "x" << width << "]:\n";
    for (int h = 0; h < height; h++) 
    {
        std::cout << "[ ";
        for (int w = 0; w < width; w++) 
        {
            std::cout << std::fixed << std::setprecision(4) << data[h * width + w] << " ";
        }
        std::cout << "]\n";
    }
}
inline void printTensor2D(std::string name, const float* data, int height, int width) 
{
    if (!data) 
    {
        std::cerr << "ERROR: Cannot print tensor '" << name << "'. Pointer is NULL.\n";
        return;
    }
    if (height <= 0 || width <= 0) 
    {
         std::cerr << "WARNING: Cannot print tensor '" << name << "'. Invalid dimensions [" << height << "x" << width << "].\n";
         return;
    }
    std::cout << "\n "<<name<<" Tensor (from pointer) [" << height << "x" << width << "]:\n";
    for (int h = 0; h < height; h++) 
    {
        std::cout << "[ ";
        for (int w = 0; w < width; w++) 
        {
            std::cout << std::fixed << std::setprecision(4) << data[(size_t)h * width + w] << " ";
        }
        std::cout << "]\n";
    }
}

inline void find_array_range(const float* d_array, int size, float& min_val, float& max_val) 
{
    std::vector<float> h_array(size);
    CUDA_CHECK_ERROR(cudaMemcpy(h_array.data(), d_array, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    auto minmax = std::minmax_element(h_array.begin(), h_array.end());
    min_val = *minmax.first;
    max_val = *minmax.second;
}

inline bool CheckAvailableCudaDevices() 
{
    int deviceCount = 0;

    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) 
    {
        std::cerr << "Error in cudaGetDeviceCount: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    if (deviceCount != 0) 
    {
        std::cout << "Number of CUDA-capable devices: " << deviceCount << std::endl;
        return true;
    }
    std::cout << "No CUDA-capable devices found." << std::endl;
    return false;
}

inline void fillWithRandomValues(std::vector<float>& vec) 
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0); 

    for (float& value : vec) 
    {
        value = dis(gen);
    }
}



inline void safeCudaMalloc(float** ptr, size_t bytes, const char* name) 
{
     //This function makes sure that no double allocation happens in case of programmer error by first freeing allocated
    //memory before allocating new memory. It also sets the pointer to nullptr after freeing to avoid dangling pointers.
    //After init it memsets to 0, therefore no garbage values are present and no need to do it manually.
    if (*ptr != nullptr) 
    {
        CUDA_CHECK_ERROR(cudaFree(*ptr));
        *ptr = nullptr;
        if(debug_flag) 
        {
           DEBUG_PRINT("\033[1;31m[%s] Freed previous allocation.\033[0m\n", name);
        }
    }
    CUDA_CHECK_ERROR(cudaMalloc(ptr, bytes));
    CUDA_CHECK_ERROR(cudaMemset(*ptr, 0, bytes));
    cudaDeviceSynchronize();
    if(debug_flag) 
    {
        DEBUG_PRINT("\033[1;32m%s allocated at %p (%zu bytes)\033[0m\n", name, *ptr, bytes);
    }
}
inline void safeCudaFree(float** ptr, const char* name) 
{
    if (*ptr != nullptr) 
    {
        cudaError_t err = cudaFree(*ptr);
        if (err != cudaSuccess) 
        {
            fprintf(stderr, "Warning: cudaFree failed for %s at %p: %s\n", 
                    name, (void*)*ptr, cudaGetErrorString(err));
        } 
        else if (debug_flag) 
        {
            fprintf(stderr, "Successfully freed %s at %p\n", name, (void*)*ptr);
        }
        *ptr = nullptr;
    }
}


inline void ToDevice(float** d_ptr, const std::vector<float>& h_vec,const char* name) 
{
    cudaDeviceSynchronize();
    safeCudaMalloc(d_ptr, h_vec.size() * sizeof(float), "Input");
    CUDA_CHECK_ERROR(cudaMemcpy(*d_ptr, h_vec.data(), h_vec.size() * sizeof(float), cudaMemcpyHostToDevice));
    DEBUG_PRINT("ToDev function successfully allocated %s at: %p\n", name,*d_ptr);
}

inline void ToHost(std::vector<float>& h_vec, const float* d_ptr) 
{
    cudaDeviceSynchronize();
    if(d_ptr==nullptr)
    {
        throw std::invalid_argument("Device pointer is null");
    }
    DEBUG_PRINT("Copying data To Host with size: %zu\n",h_vec.size());
    CUDA_CHECK_ERROR(cudaMemcpy(h_vec.data(), d_ptr, h_vec.size() * sizeof(float), cudaMemcpyDeviceToHost));
    DEBUG_COUT("ToHost function successfully copied data from device to host\n");
}

inline void ToDevice(float** d_ptr, const float* h_ptr, size_t num_elements, const char* name) {
    cudaDeviceSynchronize();
    safeCudaMalloc(d_ptr, num_elements * sizeof(float), name);
    printf("starting copy of %zu elements\n", num_elements);
    CUDA_CHECK_ERROR(cudaMemcpy(*d_ptr, h_ptr, num_elements * sizeof(float), cudaMemcpyHostToDevice));
    printf("finished copy of %zu elements\n", num_elements);
    DEBUG_PRINT("ToDev function successfully allocated %s at: %p (%zu elements)\n", 
                name, *d_ptr, num_elements);
}



// Overload for direct pointer usage (no std::vector)
inline void ToHost(float* h_ptr, const float* d_ptr, size_t num_elements) {
    cudaDeviceSynchronize();
    if (d_ptr == nullptr) {
        throw std::invalid_argument("Device pointer is null");
    }
    DEBUG_PRINT("Copying data To Host with size: %zu elements\n", num_elements);
    CUDA_CHECK_ERROR(cudaMemcpy(h_ptr, d_ptr, num_elements * sizeof(float), cudaMemcpyDeviceToHost));
    DEBUG_COUT("ToHost function successfully copied data from device to host\n");
}

inline void checkPointerAlignment(const float* ptr, const char* name) {
    size_t addr = reinterpret_cast<size_t>(ptr);
    if (addr % 4 != 0) {
        std::cerr << name << " is not 4-byte aligned. Address: " << ptr << std::endl;
    } else {
        DEBUG_COUT(name << " is 4-byte aligned. Address: " << ptr);
        //std::cout << name << " is 4-byte aligned. Address: " << ptr << std::endl;
    }
}

inline float calculate_accuracy(const float* predictions, const float* targets, size_t batch_size, int num_classes) 
{
    int correct = 0;
    for (size_t i = 0; i < batch_size; ++i) 
    {
        const float* current_pred = predictions + i * num_classes;
        const float* current_target = targets + i * num_classes;
        int pred_idx = std::distance(current_pred, std::max_element(current_pred, current_pred + num_classes));
        int target_idx = std::distance(current_target, std::max_element(current_target, current_target + num_classes));
        if (pred_idx == target_idx) 
        {
            correct++;
        }
    }
    return (batch_size > 0) ? static_cast<float>(correct) / batch_size : 0.0f;
}

inline std::vector<std::string> display_image_opencv(
    const std::vector<float>& image_data, int height, int width,
    bool is_grayscale = true, const std::string& label = "Image", bool show_window = true)
{
    std::vector<std::string> terminal_lines; 

    cv::Mat image;
    int channels = is_grayscale ? 1 : 3;
    size_t expected_elements = (size_t)height * width * channels;

    if (image_data.size() != expected_elements) 
    {
        std::cerr << "Error in display_image_opencv: image_data size (" << image_data.size()
                  << ") does not match expected size (" << expected_elements << ") for "
                  << height << "x" << width << "x" << channels << std::endl;
        image = cv::Mat::zeros(height, width, is_grayscale ? CV_32FC1 : CV_32FC3);
    } 
    else if (is_grayscale) 
    {
        image = cv::Mat(height, width, CV_32FC1, const_cast<float*>(image_data.data()));
    } 
    else 
    {
      
        cv::Mat temp_image(height, width, CV_32FC3);
        size_t plane_size = (size_t)height * width;
        std::vector<cv::Mat> channel_mats;
        for(int c=0; c<channels; ++c) 
        {
            channel_mats.push_back(cv::Mat(height, width, CV_32F, const_cast<float*>(image_data.data() + c * plane_size)));
        }
        cv::merge(channel_mats, temp_image);
        image = temp_image;
    }

    cv::Mat normalized;
    double min_val, max_val;
    cv::minMaxLoc(image, &min_val, &max_val);
    if (min_val < 0.0 || max_val > 1.0) 
    {
        if (min_val == max_val) 
        {
             normalized = cv::Mat::zeros(image.size(), image.type());
             if (min_val > 0.5) 
             {
                 normalized.setTo(cv::Scalar::all(1.0));
             }
        } 
        else 
        {
            cv::normalize(image, normalized, 0.0, 1.0, cv::NORM_MINMAX);
        }
    } 
    else 
    {
        normalized = image.clone();
    }

    cv::Mat display_image_8u;
    normalized.convertTo(display_image_8u, CV_8U, 255.0);

    if (show_window) 
    {
        cv::namedWindow(label, cv::WINDOW_NORMAL);
        int display_window_width = 512;
        int display_window_height = 512;
        cv::resizeWindow(label, display_window_width, display_window_height);
        cv::Mat resized_for_display;
        cv::resize(display_image_8u, resized_for_display, cv::Size(display_window_width, display_window_height), 0, 0, cv::INTER_NEAREST);
        cv::imshow(label, resized_for_display);
    }

    // --- Generate Terminal Representation ---
    const char* shades[] = {" ", "░", "▒", "▓", "█"};
 
    const int MAX_TERM_HEIGHT = 15;
    const int MAX_TERM_WIDTH = 30;

    int terminal_display_height = std::min(height, MAX_TERM_HEIGHT);
    int terminal_display_width = std::min(width, MAX_TERM_WIDTH);

    cv::Mat resized_for_terminal;

    cv::Mat temp_gray_for_terminal;
    if (display_image_8u.channels() == 3) 
    {
        cv::cvtColor(display_image_8u, temp_gray_for_terminal, cv::COLOR_BGR2GRAY);
    } 
    else 
    {
        temp_gray_for_terminal = display_image_8u;
    }
    cv::resize(temp_gray_for_terminal, resized_for_terminal, cv::Size(terminal_display_width, terminal_display_height), 0, 0, cv::INTER_NEAREST);

    for (int y = 0; y < resized_for_terminal.rows; y++) 
    {
        std::string line = "";
        for (int x = 0; x < resized_for_terminal.cols; x++) 
        {
            uchar value = resized_for_terminal.at<uchar>(y, x);
            int shade_idx = value * 5 / 256;
            shade_idx = std::max(0, std::min(4, shade_idx)); 
            line += shades[shade_idx];
        }
        terminal_lines.push_back(line);
    }

    return terminal_lines;
}
#endif // UTILS_H_