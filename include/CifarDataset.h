#ifndef CIFAR_DATASET_H
#define CIFAR_DATASET_H

#include <string>
#include <vector>
#include <stdexcept>
#include <fstream>   
#include <iostream>  
#include <numeric>  
#include <cstring>

class CifarDataset : public Dataset {
private:
    std::vector<float> images_host; 
    std::vector<float> labels_host;
    std::string base_path;
    std::string split;

    const int IMAGE_HEIGHT = 32;
    const int IMAGE_WIDTH = 32;
    const int IMAGE_CHANNELS = 3;
    const int NUM_CLASSES = 10;
    const size_t IMAGE_BYTES = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS; 
    const size_t RECORD_BYTES = 1 + IMAGE_BYTES; // 1 label byte + image bytes

    size_t feature_size_ = IMAGE_BYTES; 
    size_t target_size_ = NUM_CLASSES;

    // Helper function to load a single CIFAR batch file
    void load_cifar_batch(const std::string& filename) 
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) 
        {
            throw std::runtime_error("Failed to open CIFAR batch file: " + filename);
        }

        file.seekg(0, std::ios::end);
        size_t file_size = file.tellg();
        file.seekg(0, std::ios::beg);

        if (file_size == 0 || file_size % RECORD_BYTES != 0) 
        {
            throw std::runtime_error("Invalid file size for CIFAR batch: " + filename);
        }
        size_t num_images_in_batch = file_size / RECORD_BYTES;

        // Reserve space to reduce reallocations
        images_host.reserve(images_host.size() + num_images_in_batch * IMAGE_BYTES);
        labels_host.reserve(labels_host.size() + num_images_in_batch * NUM_CLASSES);

        std::vector<unsigned char> buffer(RECORD_BYTES);

        for (size_t i = 0; i < num_images_in_batch; ++i) 
        {
            if (!file.read(reinterpret_cast<char*>(buffer.data()), RECORD_BYTES)) 
            {
                if (file.eof()) 
                {
                     std::cerr << "Warning: Unexpected end of file reached while reading record " << i << " from " << filename << std::endl;
                } 
                else 
                {
                     throw std::runtime_error("Failed to read record " + std::to_string(i) + " from CIFAR batch: " + filename);
                }
                break;
            }

            unsigned char label_byte = buffer[0];
            if (label_byte >= NUM_CLASSES) 
            {
                 std::cerr << "Warning: Invalid label " << (int)label_byte << " encountered in " << filename << std::endl;
                 continue;
            }

            size_t label_offset = labels_host.size();
            labels_host.resize(label_offset + NUM_CLASSES, 0.0f);
            labels_host[label_offset + label_byte] = 1.0f;

            size_t image_offset = images_host.size();
            images_host.resize(image_offset + IMAGE_BYTES);
            float* image = images_host.data() + image_offset;

            const size_t channel_size = IMAGE_HEIGHT * IMAGE_WIDTH; 
            for (size_t row = 0; row < IMAGE_HEIGHT; ++row) 
            {
                for (size_t col = 0; col < IMAGE_WIDTH; ++col) 
                {
                    size_t plane_idx = row * IMAGE_WIDTH + col;
                    image[plane_idx] = static_cast<float>(buffer[1 + plane_idx]) / 255.0f;
                    image[channel_size + plane_idx] = static_cast<float>(buffer[1 + channel_size + plane_idx]) / 255.0f;
                    image[2 * channel_size + plane_idx] = static_cast<float>(buffer[1 + 2 * channel_size + plane_idx]) / 255.0f;
                }
            }
        }

        file.close();
        std::cout << "  Loaded " << num_images_in_batch << " images from " << filename << std::endl;
    }

public:
    CifarDataset(const std::string& dataset_base_path, const std::string& split)
        : base_path(dataset_base_path), split(split)
    {
        std::cout << "Loading CIFAR-10 data from: " << base_path << " (" << split << ")" << std::endl;

        if (split == "training") 
        {
            for (int i = 1; i <= 5; ++i)
            {
                load_cifar_batch(base_path + "/data_batch_" + std::to_string(i) + ".bin");
            }
        } 
        else if (split == "testing") 
        {
            load_cifar_batch(base_path + "/test_batch.bin");
        } 
        else 
        {
            throw std::invalid_argument("Invalid split type for CifarDataset: " + split + ". Use 'training' or 'testing'.");
        }

        if (images_host.empty() || labels_host.empty() || size() != labels_host.size() / NUM_CLASSES) 
        {
             throw std::runtime_error("CIFAR-10 dataset loading failed or resulted in inconsistent data.");
        }

        std::cout << "Finished loading CIFAR-10 " << split << " set. Total images: " << size() << std::endl;
    }

    size_t size() const override {
        return images_host.size() / IMAGE_BYTES;
    }

    void get_sample(size_t index, std::vector<float>& out_features, std::vector<float>& out_targets) const override 
    {
        if (index >= size()) 
        {
            fprintf(stderr, "Error: CifarDataset index %zu out of range (size %zu).\n", index, size());
            throw std::out_of_range("Dataset index out of range");
        }
        size_t image_offset = index * IMAGE_BYTES;
        size_t label_offset = index * NUM_CLASSES;
        out_features.assign(images_host.begin() + image_offset, images_host.begin() + image_offset + IMAGE_BYTES);
        out_targets.assign(labels_host.begin() + label_offset, labels_host.begin() + label_offset + NUM_CLASSES);
    }

    size_t feature_size() const override { return feature_size_; }
    size_t target_size() const override { return target_size_; }

    void copy_sample_to(size_t index, float* out_features, float* out_targets) const override
    {
        if (index >= size()) 
        {
            fprintf(stderr, "Error: CifarDataset index %zu out of range (size %zu).\n", index, size());
            throw std::out_of_range("Dataset index out of range");
        }

        if (out_features) {
            std::memcpy(out_features, images_host.data() + index * IMAGE_BYTES, IMAGE_BYTES * sizeof(float));
        }
        if (out_targets) {
            std::memcpy(out_targets, labels_host.data() + index * NUM_CLASSES, NUM_CLASSES * sizeof(float));
        }
    }
};

#endif // CIFAR_DATASET_H
