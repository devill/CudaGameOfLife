
#define GL_GLEXT_PROTOTYPES

#include "CudaSharedGLBuffer.h"

#include "GL/glext.h"
#include "GL/glx.h"

#include "CudaError.hpp"

#include <iostream>

CudaSharedGLBuffer::CudaSharedGLBuffer(size_t size)
{
    // Generate an OpenGL buffer handle
    glGenBuffers(1, &bufferObj);
    // Bind it to a pixel buffer
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj );
    // Allocate memory for the pixel buffer
    glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, size, NULL, GL_DYNAMIC_DRAW_ARB );

    // Set up the GPU for CUDA/OpenGL interoperability, by sharing the buffer through the resurce pointer
    CUDA_ERROR( cudaGraphicsGLRegisterBuffer( &resource, bufferObj, cudaGraphicsMapFlagsNone ) );
    isResourceRegistered = true;

    BufferCudaLock bufferLock(*this);
    CUDA_ERROR( cudaMemset( bufferLock.getDeviceBuffer(), 0, size ) );
}

CudaSharedGLBuffer::~CudaSharedGLBuffer()
{
    try
    {
        unregisterResource();
    }
    catch(CudaException e)
    {
        std::cerr << e.what() << std::endl;
    }
}

void CudaSharedGLBuffer::unregisterResource()
{
    if(isResourceRegistered)
    {
        CUDA_ERROR( cudaGraphicsUnregisterResource( resource ) );
        glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );
        glDeleteBuffers( 1, &bufferObj );
        isResourceRegistered = false;
    }
}

BufferCudaLock::BufferCudaLock(CudaSharedGLBuffer& sharedBuffer) : resource(sharedBuffer.resource)
{
    size_t size;

    CUDA_ERROR( cudaGraphicsMapResources( 1, &(resource), NULL ) );
    CUDA_ERROR( cudaGraphicsResourceGetMappedPointer( (void**)&deviceBuffer, &size, resource) );
}

BufferCudaLock::~BufferCudaLock()
{
    CUDA_ERROR( cudaGraphicsUnmapResources( 1, &(resource), NULL ) );
}

