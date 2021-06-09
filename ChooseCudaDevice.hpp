
#ifndef CHOOSE_CUDA_DEVICE
#define CHOOSE_CUDA_DEVICE

#include <cuda.h>
#include "CudaError.hpp"

int chooseCudaDevice()
{
    int dev;

    cudaDeviceProp prop;
    memset( &prop, 0, sizeof(cudaDeviceProp));
    prop.major = 2;
    prop.minor = 0;

    CUDA_ERROR( cudaChooseDevice( &dev, &prop ) );

    return dev;
}


#endif
