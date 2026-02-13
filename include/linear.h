#pragma once

#include <vector>
namespace linear
{
    float calculate_custom_loss(const std::vector<float>& output, const std::vector<float>& target);


    void generate_synthetic_data(std::vector<float>& inputs, std::vector<float>& targets, 
                            int batch_size, int input_size, int output_size,
                            bool binary_classification = true,bool is_classification=false);

    void visualize_linear_output(const std::vector<float>& outputs, const std::vector<float>& targets, int output_size);


    float calculate_accuracy(const std::vector<float>& outputs, const std::vector<float>& targets, int output_size);


    void train_linear_model(int input_size, int hidden_size, int output_size,
        int batch_size, int num_epochs, bool is_classification);
    float calculate_custom_loss(const std::vector<float>& output, const std::vector<float>& target);
}