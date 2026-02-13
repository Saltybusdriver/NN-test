#ifndef BUTTERFLY_DATASET_H
#define BUTTERFLY_DATASET_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream> 
#include <unordered_map>
#include <algorithm>
#include <set>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "DataLoader.h"

namespace fs = std::filesystem;

class ButterflyDataset : public Dataset {
private:
    std::string dataset_dir;
    std::vector<std::string> image_paths;
    std::vector<std::string> labels; 
    std::unordered_map<std::string, int> label_to_index;
    int input_width;
    int input_height;
    int num_classes = 0; 
    bool use_rgb;
    std::string data_subfolder;
    size_t feature_size_ = 0;
    size_t target_size_ = 0;

    void load_from_csv(const std::string& csv_path) 
    {
        std::ifstream file(csv_path);
        if (!file.is_open()) 
        {
            throw std::runtime_error("Failed to open file: " + csv_path);
        }
        fs::path csv_filepath(csv_path);
        std::string csv_filename = csv_filepath.filename().string();
        if (csv_filename == "Training_set.csv") 
        {
            data_subfolder = "/train/";
        } 
        else if (csv_filename == "Testing_set.csv") 
        {
            data_subfolder = "/test/";
        } 
        else 
        {
            data_subfolder = "/";
            std::cerr << "Warning: Could not determine data subfolder from CSV name '" << csv_filename << "'. Using root folder." << std::endl;
        }
        std::cout << "Using data subfolder: " << data_subfolder << std::endl;


        std::string line;
        std::getline(file, line); // Skip header

        std::set<std::string> unique_labels_found;

        while (std::getline(file, line)) 
        {
            line.erase(line.find_last_not_of(" \n\r\t")+1);
            if (line.empty()) continue;

            std::stringstream ss(line);
            std::string filename;
            std::string label_str;
            bool is_test_set_file = (data_subfolder == "/test/");

            if (is_test_set_file) 
            {

                filename = line;
                label_str = "";
            } 
            else 
            {
    
                std::stringstream ss(line);
                if (!std::getline(ss, filename, ',')) 
                {
                    
                    std::cerr << "Warning: Skipping training line with unexpected format (no comma?): " << line << std::endl;
                    continue;
                }

                std::getline(ss, label_str);
                label_str.erase(0, label_str.find_first_not_of(" \t"));
                label_str.erase(label_str.find_last_not_of(" \t") + 1); 
                if (label_str.empty()) 
                {
                     std::cerr << "Warning: Skipping training line with empty label: " << line << std::endl;
                     continue; 
                }
                unique_labels_found.insert(label_str);
            }


            std::string full_path = dataset_dir + data_subfolder + filename;

            if (fs::exists(full_path) && fs::is_regular_file(full_path)) 
            {
                image_paths.push_back(full_path);
                labels.push_back(label_str);
            } 
            else 
            {
                std::cerr << "Warning: File not found or not a regular file: " << full_path << std::endl;
            }
        }
        file.close();

        // Create mapping from label to index based ONLY on actual labels found
        int idx = 0;
        label_to_index.clear();
        for (const auto& unique_label : unique_labels_found) 
        {
            label_to_index[unique_label] = idx++;
        }
        num_classes = unique_labels_found.size();
        target_size_ = num_classes;

        std::cout << "Loaded " << image_paths.size() << " images. Found "
                  << num_classes << " unique classes requiring labels." << std::endl;
    }

public:
    ButterflyDataset(const std::string& csv_path, const std::string& data_dir,
                     int width = 224, int height = 224, bool rgb = false)
        : dataset_dir(data_dir), input_width(width), input_height(height), use_rgb(rgb) {
        load_from_csv(csv_path);
        feature_size_ = input_width * input_height * (use_rgb ? 3 : 1);
        std::cout << "Using " << (use_rgb ? "RGB" : "grayscale") << " images" << std::endl;
    }

    size_t size() const override {
        return image_paths.size();
    }

    void get_sample(size_t index, std::vector<float>& out_features, std::vector<float>& out_targets) const override {
        if (index >= image_paths.size()) {
            throw std::out_of_range("Index out of range");
        }

        try {

            cv::Mat img = cv::imread(image_paths[index], cv::IMREAD_COLOR);
            if (img.empty()) {
                std::cerr << "Warning: Failed to load image: " << image_paths[index] << ". Returning blank." << std::endl;
 
                out_features.assign(feature_size_, 0.0f);
                // Leave out_targets empty or assign zeros if num_classes > 0
                if (num_classes > 0) out_targets.assign(target_size_, 0.0f);
                else out_targets.clear();
                return;
            }

            // Convert to grayscale if needed
            cv::Mat processed_img;
            if (!use_rgb) 
            {
                cv::cvtColor(img, processed_img, cv::COLOR_BGR2GRAY);
            } 
            else 
            {
                processed_img = img;
            }

            // Resize
            cv::Mat resized_img;
            if (processed_img.cols != input_width || processed_img.rows != input_height) 
            {
                cv::resize(processed_img, resized_img, cv::Size(input_width, input_height));
            } 
            else 
            {
                resized_img = processed_img;
            }

            cv::Mat float_img;
            resized_img.convertTo(float_img, CV_32F, 1.0/255.0);


            int channels = use_rgb ? 3 : 1;
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
                     std::memcpy(out_ptr, float_img.ptr<float>(r), float_img.cols * channels * sizeof(float));
                     out_ptr += float_img.cols * channels;
                 }
            }

            const std::string& current_label = labels[index];
            if (!current_label.empty() && num_classes > 0) 
            {
                // If the label string is NOT empty AND we have classes defined
                out_targets.assign(target_size_, 0.0f); 
                auto it = label_to_index.find(current_label);
                if (it != label_to_index.end()) 
                {
                    out_targets[it->second] = 1.0f; 
                } 
                else 
                {
                    std::cerr << "Warning: Label '" << current_label << "' not found in label_to_index map for image: " << image_paths[index] << std::endl;
                   
                }
            } 
            else 
            {
                out_targets.clear(); 
            }

        } 
        catch (const std::exception& e) 
        {
            std::cerr << "Error processing image at index " << index << " (" << image_paths[index] << "): " << e.what() << std::endl;
            out_features.assign(feature_size_, 0.0f);
            if (num_classes > 0) out_targets.assign(target_size_, 0.0f);
            else out_targets.clear();
        }
    }

    size_t feature_size() const override { return feature_size_; }
    size_t target_size() const override { return target_size_; } 

    std::string get_class_name(int index) const 
    {
        for (const auto& pair : label_to_index) 
        {
            if (pair.second == index) 
            {
                return pair.first;
            }
        }
        return "Unknown";
    }

    int get_num_classes() const {
        return num_classes;
    }
};

#endif // BUTTERFLY_DATASET_H