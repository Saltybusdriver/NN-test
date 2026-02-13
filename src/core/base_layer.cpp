
#include "layers.h"
#include "utils.h"
#include "DebugUtils.h"
#include "layer_proxy.h"

Layer::Layer(const char* activation) : input_size(0), output_size(0), batch_size(0),
                 output(nullptr), output_error(nullptr), stored_input(nullptr)
{
    activation_type_id = activation_type(activation);
    if (activation_type_id == -1) {
        throw std::invalid_argument("Invalid activation function");
    }
    INFO_COUT("INITIALIZING LAYER....\n");
}

Layer::~Layer() {
    if (GetUseCuda()) 
    {
        safeCudaFree(&output, "Layer output (CUDA)");
        safeCudaFree(&output_error, "Layer error (CUDA)");
    } else 
    {

        delete[] output;
        delete[] output_error;
        output = nullptr;
        output_error = nullptr;
    }
}
void Layer::free_buffers() 
{
    if (GetUseCuda()) {
        safeCudaFree(&output, "Output (CUDA)");
        safeCudaFree(&output_error, "Output error (CUDA)");
    }
    else 
    {
        delete[] output;
        delete[] output_error;
        output = nullptr;
        output_error = nullptr;
    }
}

int Layer::activation_type(const char* type) 
{
    if (strcmp(type, "sigmoid") == 0)
    {
        return 0;
    } 
    else if (strcmp(type, "relu") == 0)
    {
        return 1;
    }
    else if (strcmp(type, "leakyrelu") == 0)
    {
        return 2;
    } 
    else if (strcmp(type, "softmax") == 0)
    {
        return 3;
    }
    else if(strcmp(type,"none")==0)
    {
        return 4;
    } 
    else
    {
        fprintf(stderr, "Unknown/missing activation function: %s\n", type);
        return 4;
    }
    return -1;
}

void Layer::allocate_buffers() {
    size_t total_elements = get_total_output_size();
    size_t total_bytes = total_elements * sizeof(float);

    if (GetUseCuda()) 
    {

        safeCudaMalloc(&output, total_bytes, "Output (CUDA)");
        safeCudaMalloc(&output_error, total_bytes, "Output error (CUDA)");
    } 
    else 
    {
        try 
        {
            output = new float[total_elements];
            output_error = new float[total_elements];

            std::fill_n(output, total_elements, 0.0f);
            std::fill_n(output_error, total_elements, 0.0f);
        } 
        catch (const std::bad_alloc& e) 
        {
            fprintf(stderr, "Failed to allocate CPU memory for Layer buffers: %s\n", e.what());
            output = nullptr; 
            output_error = nullptr;
            throw;
        }
}
}