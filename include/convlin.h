#pragma once

#include <string>
#include <vector>

namespace convlin {
    void display_image_opencv(const std::vector<float>& image_data, int height, int width, 
                            bool is_grayscale = true, const std::string& label = "Image");


    float calculate_custom_loss(const std::vector<float>& output, const std::vector<float>& target);

    void train_conv_model(int batch_size = 64, int num_epochs = 5, 
                        const std::string& dataset_path = "", 
                        int image_height = 32, int image_width = 32,
                        bool use_grayscale = true);
}