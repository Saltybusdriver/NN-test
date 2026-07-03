#ifndef CIFAR_TRAIN_H
#define CIFAR_TRAIN_H

#include <string>

namespace cifar_train {

void train_cifar_classifier(
    int batch_size,
    int num_epochs,
    const std::string& dataset_path,
    int image_height, // Should be 32
    int image_width,  // Should be 32
    bool use_grayscale // Should be false
);

} // namespace cifar_train

#endif // CIFAR_TRAIN_H