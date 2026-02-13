#ifndef CPU_CIFAR_TRAIN_H
#define CPU_CIFAR_TRAIN_H

#include <string>

namespace cpu_cifar_train {

void train_cifar_classifier(
    int batch_size,
    int num_epochs,
    const std::string& dataset_path,
    int image_height, // Should be 32
    int image_width,  // Should be 32
    bool use_grayscale // Should be false
);

}

#endif // CIFAR_TRAIN_H