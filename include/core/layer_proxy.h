#ifndef LAYER_PROXY_H
#define LAYER_PROXY_H

#include "layers.h" 

extern bool g_use_cuda;

void SetUseCuda(bool use_cuda);
bool GetUseCuda();

// Factory functions that create the correct implementation based on g_use_cuda
// Each returns a pointer to the base Layer class
inline Layer* createLinear(int in_feat, int out_feat, int b_size, const char* activation) 
{
    if (g_use_cuda) 
    {
        return static_cast<Layer*>(new Cuda::Linear(in_feat, out_feat, b_size, activation));
    } 
    else 
    {
        return static_cast<Layer*>(new cpu::Linear(in_feat, out_feat, b_size, activation));
    }
}

inline Layer* createConv2d(int in_ch, int out_ch, int img_h, int img_w, 
                          int k_size, int str = 1, int pad = 0, int b_sz = 1, 
                          const char* activation = "relu") {
    if (g_use_cuda) 
    {
        return static_cast<Layer*>(new Cuda::Conv2d(in_ch, out_ch, img_h, img_w, k_size, str, pad, b_sz, activation));
    } 
    else 
    {
        return static_cast<Layer*>(new cpu::Conv2d(in_ch, out_ch, img_h, img_w, k_size, str, pad, b_sz, activation));
    }
}

inline Layer* createFlatten(int batch_size, int channels, int height, int width) 
{
    if (g_use_cuda) 
    {
        return static_cast<Layer*>(new Cuda::Flatten(batch_size, channels, height, width));
    } 
    else 
    {
        return static_cast<Layer*>(new cpu::Flatten(batch_size, channels, height, width));
    }
}

#define Linear(args...) createLinear(args)
#define Conv2d(args...) createConv2d(args)
#define Flatten(args...) createFlatten(args)

#endif // LAYER_PROXY_H