#ifndef NN_CORE_H
#define NN_CORE_H

// Core components
#include "NeuralNetwork.h"
#include "layers.h"         // Includes base Layer class
#include "layer_proxy.h"    // Includes factory functions (Linear, Conv2d, Flatten)
#include "optimizer.h"
#include "autodiff.h"
#include "DataLoader.h"     // Data handling classes
#include "thread_pool.h"    // Thread pool utility

// CPU specific
#include "cpu_functions.h"
#include "cpu_utils.h"
#include "cpu_DebugUtils.h"

// CUDA specific
#include "cuda_functions.h"

#include "utils.h"          // General utility functions (CPU/GPU agnostic or switching)
#include "debug.h"          // Core debug macros and level settings
#include "DebugUtils.h"     // General debug utility class

#endif // NN_CORE_H