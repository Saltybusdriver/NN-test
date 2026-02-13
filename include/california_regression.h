#ifndef CALIFORNIA_REGRESSION_H
#define CALIFORNIA_REGRESSION_H

#include <string>

namespace california_regression {

void train_california_regressor(
    int batch_size, int num_epochs,
    const std::string& dataset_path,
    float learning_rate = 0.001f,
    const std::string& loss_type_str = "mse");

} // namespace california_regression

#endif // CALIFORNIA_REGRESSION_H