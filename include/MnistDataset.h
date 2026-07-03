#ifndef MNIST_DATASET_H
#define MNIST_DATASET_H

#include <vector>
#include <string>
#include <filesystem> 
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "DataLoader.h" 

namespace fs = std::filesystem;

class MnistDataset : public Dataset {
private:
    std::vector<std::string> image_paths;
    std::vector<int> labels;
    int input_width;
    int input_height;
    const int num_classes = 10; 
    size_t feature_size_ = 0;
    size_t target_size_ = num_classes;

    void load_mnist_data(const std::string& base_path, const std::string& mode) 
    {
        std::string data_dir = base_path + "/MNIST - JPG - " + mode;
        if (!fs::exists(data_dir) || !fs::is_directory(data_dir)) 
        {
            throw std::runtime_error("MNIST data directory not found: " + data_dir);
        }
        std::cout << "Loading MNIST data from: " << data_dir << std::endl;

        for (int i = 0; i < num_classes; ++i) 
        {
            std::string class_dir = data_dir + "/" + std::to_string(i);
            if (!fs::exists(class_dir) || !fs::is_directory(class_dir)) 
            {
                 std::cerr << "Warning: Class directory not found: " << class_dir << std::endl;
                 continue;
            }
            for (const auto& entry : fs::directory_iterator(class_dir)) 
            {
                if (entry.is_regular_file()) 
                {
                    std::string ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") 
                    {
                        image_paths.push_back(entry.path().string());
                        labels.push_back(i);
                    }
                }
            }
        }
        std::cout << "Loaded " << image_paths.size() << " images for MNIST " << mode << " set." << std::endl;
        if (image_paths.empty()) 
        {
             throw std::runtime_error("No images found in " + data_dir);
        }
    }

public:

    MnistDataset(const std::string& base_dataset_path,
                 const std::string& mode,
                 int width = 28, int height = 28)
        : input_width(width), input_height(height) {
        load_mnist_data(base_dataset_path, mode);
        feature_size_ = input_width * input_height;
    }


    size_t size() const override 
    {
        return image_paths.size();
    }

    void get_sample(size_t index, std::vector<float>& out_features, std::vector<float>& out_targets) const override 
    {
        if (index >= image_paths.size()) 
        {
            throw std::out_of_range("Index out of range in MnistDataset");
        }

        cv::Mat img = cv::imread(image_paths[index], cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "Warning: Failed to load image: " << image_paths[index] << ". Returning blank." << std::endl;
            out_features.assign(feature_size_, 0.0f);
            out_targets.assign(target_size_, 0.0f);
            return;
        }


        cv::Mat resized_img;
        if (img.cols != input_width || img.rows != input_height) 
        {
             cv::resize(img, resized_img, cv::Size(input_width, input_height));
        } 
        else 
        {
            resized_img = img;
        }


        cv::Mat float_img;
        resized_img.convertTo(float_img, CV_32F, 1.0 / 255.0);

        out_features.resize(feature_size_);
        if (float_img.isContinuous()) 
        {
            std::memcpy(out_features.data(), float_img.data, feature_size_ * sizeof(float));
        }
        else 
        {

            float* out_ptr = out_features.data();
            for (int r = 0; r < float_img.rows; ++r) 
            {
                std::memcpy(out_ptr, float_img.ptr<float>(r), float_img.cols * sizeof(float));
                out_ptr += float_img.cols;
            }
        }

        out_targets.assign(target_size_, 0.0f);
        int label = labels[index];
        if (label >= 0 && label < num_classes)
        {
             out_targets[label] = 1.0f;
        } 
        else 
        {
             std::cerr << "Warning: Invalid label encountered: " << label << " for image " << image_paths[index] << std::endl;
        }
    }

    size_t feature_size() const override { return feature_size_; }
    size_t target_size() const override { return target_size_; }

    std::tuple<int, int, int> get_dimensions() const 
    {
        return std::make_tuple(1, input_height, input_width);
    }

    int get_num_classes() const 
    {
        return num_classes;
    }
};

#endif // MNIST_DATASET_H