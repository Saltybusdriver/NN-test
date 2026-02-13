#ifndef IMAGE_DATASET_H
#define IMAGE_DATASET_H

#include "DataLoader.h" 
#include <opencv2/opencv.hpp> 
#include <string>
#include <vector>
#include <filesystem> 
#include <iostream>
#include <random>
#include <functional>
#include <utility>
#include <unordered_map>
#include <algorithm> 
#include <mutex>
#include <set>    

namespace fs = std::filesystem;
static std::mutex opencv_mutex;

using Transform = std::function<std::vector<float>(const cv::Mat&)>;
using TargetTransform = std::function<std::vector<float>(int)>;

class ImageDataset : public Dataset {
private:
    std::vector<std::string> image_paths;
    std::vector<int> labels;
    std::unordered_map<std::string, int> class_to_idx; 
    int image_height;
    int image_width;
    int in_channels; 
    bool grayscale;
    Transform transform; 
    TargetTransform target_transform;

    size_t feature_size_ = 0;
    size_t target_size_ = 0; 
    int num_classes_ = 0; 

  
    void update_class_info() {
        if (labels.empty()) {
            num_classes_ = 0;
            target_size_ = 0; 
            return;
        }
    
        num_classes_ = *std::max_element(labels.begin(), labels.end()) + 1;

        if (!target_transform) 
        {
            target_size_ = num_classes_;
        } 
        else 
        {
            target_size_ = 0;
        }
    }

public:

    ImageDataset(const std::string& directory_path,
                int height,
                int width,
                bool use_grayscale = true,
                Transform img_transform = nullptr,
                TargetTransform lbl_transform = nullptr)
        : image_height(height),
          image_width(width),
          grayscale(use_grayscale),
          transform(img_transform),
          target_transform(lbl_transform) {

        in_channels = grayscale ? 1 : 3;
        feature_size_ = image_height * image_width * in_channels;

        std::cout << "Scanning directory: " << directory_path << std::endl;

        for (const auto& entry : fs::directory_iterator(directory_path)) 
        {
            if (entry.is_regular_file()) 
            {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tiff") 
                {
                    image_paths.push_back(entry.path().string());
                    labels.push_back(0);
                }
            }
        }

        if (image_paths.empty()) 
        {
            std::cerr << "Warning: No images found in " << directory_path << std::endl;
        } 
        else
        {
            std::cout << "Found " << image_paths.size() << " images in " << directory_path << std::endl;
            update_class_info();
        }
    }

    void set_labels(const std::vector<int>& new_labels) 
    {
        if (new_labels.size() != image_paths.size()) 
        {
            throw std::invalid_argument(
                "Number of labels (" + std::to_string(new_labels.size()) +
                ") doesn't match number of images (" + std::to_string(image_paths.size()) + ")"
            );
        }

        labels = new_labels;
        update_class_info(); 
        std::cout << "Set " << labels.size() << " custom labels. Detected " << num_classes_ << " classes." << std::endl;
    }

    void set_label(size_t index, int label) 
    {
        if (index >= image_paths.size()) 
        {
            throw std::out_of_range("Image index out of range");
        }
        labels[index] = label;
    }
    std::string get_image_path(size_t index) const 
    {
        if (index >= image_paths.size()) 
        {
            throw std::out_of_range("Image index out of range");
        }
        return image_paths[index];
    }

    std::string get_filename(size_t index) const 
    {
        if (index >= image_paths.size()) 
        {
            throw std::out_of_range("Image index out of range");
        }
        fs::path path(image_paths[index]);
        return path.filename().string();
    }

    std::vector<std::string> get_filenames() const 
    {
        std::vector<std::string> filenames;
        filenames.reserve(image_paths.size());
        for (const auto& path : image_paths)
        {
            fs::path fs_path(path);
            filenames.push_back(fs_path.filename().string());
        }
        return filenames;
    }

    std::vector<int> get_labels() const 
    {
        return labels;
    }

    size_t size() const override 
    {
        return image_paths.size();
    }
    void get_sample(size_t index, std::vector<float>& out_features, std::vector<float>& out_targets) const override 
    {
        if (index >= image_paths.size()) 
        {
            throw std::out_of_range("Image index out of range");
        }

        const std::string& path = image_paths[index];
        cv::Mat image;
        {
            image = cv::imread(path, grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
        }

        if (image.empty()) 
        {
            std::cerr << "Warning: Failed to load image: " << path << ". Returning blank." << std::endl;

            out_features.assign(feature_size_, 0.0f);
    
            if (target_size_ > 0) out_targets.assign(target_size_, 0.0f);
            else out_targets.clear();
            return;
        }


        cv::Mat resized_image;
        cv::resize(image, resized_image, cv::Size(image_width, image_height));

        
        if (transform) 
        {
            out_features = transform(resized_image);
            if (out_features.size() != feature_size_) {
                 std::cerr << "Warning: Custom transform returned vector of incorrect size ("
                           << out_features.size() << " vs expected " << feature_size_
                           << ") for image " << path << std::endl;
              
                 out_features.assign(feature_size_, 0.0f);
            }
        } 
        else 
        {

            cv::Mat float_image;
            resized_image.convertTo(float_image, CV_32F, 1.0 / 255.0);

            out_features.resize(feature_size_);
            if (float_image.isContinuous()) 
            {
                std::memcpy(out_features.data(), float_image.data, feature_size_ * sizeof(float));
            } 
            else {
      
                float* out_ptr = out_features.data();
                for (int r = 0; r < float_image.rows; ++r) {
                    std::memcpy(out_ptr, float_image.ptr<float>(r), float_image.cols * in_channels * sizeof(float));
                    out_ptr += float_image.cols * in_channels;
                }
            }
        }

        int current_label = labels[index];
        if (target_transform) 
        {

            out_targets = target_transform(current_label);

        } 
        else 
        {
           
            if (num_classes_ > 0) 
            {
                out_targets.assign(num_classes_, 0.0f); 
                if (current_label >= 0 && current_label < num_classes_) 
                {
                    out_targets[current_label] = 1.0f;
                } 
                else 
                {
                    std::cerr << "Warning: Label " << current_label << " is out of range [0, "
                              << num_classes_ - 1 << "] for image " << path << std::endl;

                }
            } 
            else 
            {
                
                 out_targets.clear(); 
            }
        }
    }

    size_t feature_size() const override { return feature_size_; }
    size_t target_size() const override { return target_size_; } 

    std::tuple<int, int, int> get_dimensions() const 
    {
        return std::make_tuple(in_channels, image_height, image_width);
    }

    int get_num_classes() const 
    {
        return num_classes_;
    }
};

#endif // IMAGE_DATASET_H