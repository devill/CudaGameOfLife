
#ifndef CUDA_SHARED_GL_BUFFER_H
#define CUDA_SHARED_GL_BUFFER_H

#include <boost/noncopyable.hpp>

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <GL/glut.h>

class CudaSharedGLBuffer : private boost::noncopyable
{
    friend class BufferCudaLock;

    public:
        CudaSharedGLBuffer(size_t size);
        ~CudaSharedGLBuffer();

    private:
        void unregisterResource();

    private:
        bool isResourceRegistered; 

        GLuint bufferObj;
        cudaGraphicsResource* resource;
};

class BufferCudaLock : private boost::noncopyable
{
    public:
        BufferCudaLock(CudaSharedGLBuffer& sharedBuffer);
        ~BufferCudaLock();
        uchar4* getDeviceBuffer() { return deviceBuffer; };
    private:
        cudaGraphicsResource* resource;
        uchar4* deviceBuffer;
};

#endif
