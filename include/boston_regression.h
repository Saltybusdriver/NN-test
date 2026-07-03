#ifndef BOSTON_REGRESSION_H
#define BOSTON_REGRESSION_H

#include <string>

namespace boston_regression {

void train_boston_regressor(
    int batch_size, int num_epochs,
    const std::string& dataset_path,
    float learning_rate = 0.001f,
    const std::string& loss_type_str = "mse");

} // namespace boston_regression

#endif // BOSTON_REGRESSION_H