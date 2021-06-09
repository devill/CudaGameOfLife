
#ifndef SYNCED_SWAP_BUFFER_HPP
#define SYNCED_SWAP_BUFFER_HPP

#include <iostream>

#include "CudaError.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

class SyncedSwapBuffer
{
    public:
        SyncedSwapBuffer(int argWidth, int argHeight)
            : width(argWidth)
            , height(argHeight)
        {
            instanceCount = new int(1);
            
            hostData = new int[getSize()];

            CUDA_ERROR( cudaMalloc( (void**)&deviceOldData, getSize() ) );
            CUDA_ERROR( cudaMalloc( (void**)&deviceNewData, getSize() ) );

            clear();
        }

        SyncedSwapBuffer(SyncedSwapBuffer& rhs)
            : width(rhs.width)
            , height(rhs.height)
            , instanceCount(rhs.instanceCount)
            , hostData(rhs.hostData)
            , deviceOldData(rhs.deviceOldData)
            , deviceNewData(rhs.deviceNewData)
        {
            ++(*instanceCount);
        }

        ~SyncedSwapBuffer()
        {
            if(--(*instanceCount) == 0)
            {
                delete[] hostData;
                delete instanceCount;

                try
                {
                    CUDA_ERROR( cudaFree( deviceOldData ) );
                    CUDA_ERROR( cudaFree( deviceNewData ) );
                }
                catch(CudaException e)
                {
                    std::cerr << e.what() << std::endl;
                }

            }
        }

        void clear()
        {
            memset(hostData, 0, getSize());
            CUDA_ERROR( cudaMemset(deviceOldData, 0, getSize()));
            CUDA_ERROR( cudaMemset(deviceNewData, 0, getSize()));
        }

        void clearDeviceNewData()
        {
            CUDA_ERROR( cudaMemset(deviceNewData, 0, getSize()));
        }

        void swapDeviceBuffers(){ std::swap(deviceNewData, deviceOldData); }

        __host__ __device__ int getWidth() const { return width; }
        __host__ __device__ int getHeight() const { return height; }
        int getSize() const { return width*height*sizeof(int); }
        
        __host__ __device__ bool withinBounds(int x, int y) const { return ( x >= 0 && y >= 0 && x < width && y < height ); }
        __host__ __device__ int findIndex(int x, int y) const { return x + y * width; }

        int& host(int x, int y) { return hostData[findIndex(x,y)]; }
        __device__ int& deviceOld(int x, int y) { return deviceOldData[findIndex(x,y)]; }
        __device__ int& deviceNew(int x, int y) { return deviceNewData[findIndex(x,y)]; }

        const int& host(int x, int y) const { return hostData[findIndex(x,y)]; }
        __device__ const int& deviceOld(int x, int y) const { return deviceOldData[findIndex(x,y)]; }
        __device__ const int& deviceNew(int x, int y) const { return deviceNewData[findIndex(x,y)]; }

        void copyHostToDevice()
        {
            if(!dataIsOnDevice)
            {
                CUDA_ERROR( cudaMemcpy( deviceNewData,
                                        hostData,
                                        getSize(),
                                        cudaMemcpyHostToDevice) );
                dataIsOnDevice = true;
            }
        }
        void copyDeviceToHost() const
        {
            if(dataIsOnDevice)
            {
                CUDA_ERROR( cudaMemcpy( hostData,
                                        deviceNewData,
                                        getSize(),
                                        cudaMemcpyDeviceToHost) );
                dataIsOnDevice = false;
            }
        }

    private:
        SyncedSwapBuffer& operator=(SyncedSwapBuffer& rhs);

    private:
        int width;
        int height;

        mutable int* hostData;
        int* deviceOldData;
        int* deviceNewData;

        int* instanceCount;
        mutable bool dataIsOnDevice;
};

#endif
