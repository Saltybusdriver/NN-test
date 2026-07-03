#include "layer_proxy.h"

bool g_use_cuda = true;

void SetUseCuda(bool use_cuda) 
{
    g_use_cuda = use_cuda;
}

bool GetUseCuda() {
    return g_use_cuda;
}