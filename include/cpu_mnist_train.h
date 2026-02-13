#ifndef CPU_MNIST_TRAIN_H
#define CPU_MNIST_TRAIN_H

#include <string>

namespace cpu_mnist_train {

void train_mnist_classifier(
    int batch_size = 64, int num_epochs = 5, 
        const std::string& dataset_path = "", 
        int image_height = 32, int image_width = 32,
        bool use_grayscale = true
);

} // namespace mnist_train

#endif // MNIST_TRAIN_H