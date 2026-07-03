# Neural Network lib

This repository provides a modular C++/CUDA neural network framework for research and experimentation. It supports both CPU and GPU backends, custom layers, and flexible data loading. The codebase is designed for extensibility, educational use, and benchmarking.

## Features

*   Modular neural network architecture (CPU & CUDA support)
*   Layer abstraction (Linear, Conv2D, etc.)
*   Customizable optimizers (Adam, SGD, etc.)
*   DataLoader utilities for datasets (MNIST, CIFAR-10, Boston, California Housing, etc.)
*   Debugging and inspection utilities for tensors and gradients
*   Model serialization (save/load parameters)
*   Utilities for memory management and performance monitoring

## Core Components

*   `NeuralNetwork` class: Build, train, and evaluate neural networks.
*   `Layer` classes: Implement various neural network layers.
*   `DataLoader`: Efficient batch loading for datasets.
*   `optimizer`: Optimizer implementations (Adam, SGD, etc.)
*   `debug` and `DebugUtils`: Debugging, logging, and inspection tools.

## How to Use
## IMPORTANT:
Datasets must be placed inside a /Dataset folder in root dir.

### Dataset Directory Structure

The framework expects datasets to be organized within a `Dataset` directory located at the root of the project. Here's the expected structure for common datasets:

```
.
├── Dataset/
│   ├── MNIST/
│   │   ├── train-images-idx3-ubyte
│   │   ├── train-labels-idx1-ubyte
│   │   ├── t10k-images-idx3-ubyte
│   │   └── t10k-labels-idx1-ubyte
│   ├── CIFAR-10/
│   │   ├── cifar-10-batches-bin/
│   │   │   ├── data_batch_1.bin
│   │   │   ├── data_batch_2.bin
│   │   │   ├── data_batch_3.bin
│   │   │   ├── data_batch_4.bin
│   │   │   ├── data_batch_5.bin
│   │   │   ├── test_batch.bin
│   │   │   └── batches.meta.txt
│   │   └── cifar-10-python.tar.gz # Optional, if using Python version loader
│   ├── BostonHousing/
│   │   └── housing.csv
│   ├── CaliforniaHousing/
│   │   └── california_housing.csv
│   └── ... # Other datasets
├── src/
├── include/
├── Makefile
└── README.md
```

Ensure your dataset files are placed correctly within the `Dataset` folder before running any training or evaluation scripts.

### 1. Build the Project

```sh
# Build everything
make

# Build only the library and necessary components
make library

