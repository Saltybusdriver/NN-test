#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include "debug.h"
#include <algorithm>
#include <cstring> 

inline void safeMalloc(float** ptr, size_t bytes, const char* name) {
    //This function makes sure that no double allocation happens in case of programmer error by first freeing allocated
    //memory before allocating new memory. It also sets the pointer to nullptr after freeing to avoid dangling pointers.
    //After init it memsets to 0, therefore no garbage values are present and no need to do it manually.
    if (*ptr != nullptr) 
    {
        free(*ptr);
        *ptr = nullptr;
        if(debug_flag) 
        {
           DEBUG_PRINT("\033[1;31m[%s] Freed previous allocation.\033[0m\n", name);
        }
    }
    
    *ptr = (float*)malloc(bytes);
    
    if (*ptr == nullptr) 
    {
        fprintf(stderr, "Error: Failed to allocate memory for %s (%zu bytes)\n", name, bytes);
        exit(EXIT_FAILURE);
    }
    
    memset(*ptr, 0, bytes);
    
    if(debug_flag) 
    {
        DEBUG_PRINT("\033[1;32m%s allocated at %p (%zu bytes)\033[0m\n", name, *ptr, bytes);
    }
}

inline void safeFree(float** ptr, const char* name) 
{
    if (*ptr != nullptr) 
    {
        free(*ptr);
        
        if (debug_flag) 
        {
            fprintf(stderr, "Successfully freed %s at %p\n", name, (void*)*ptr);
        }
        *ptr = nullptr;
    }
}

// Simple CPU memory operations that might be useful
inline void copyMemory(float* dst, const float* src, size_t size) 
{
    std::memcpy(dst, src, size);
}

inline void setMemory(float* ptr, int value, size_t size) 
{
    std::memset(ptr, value, size);
}