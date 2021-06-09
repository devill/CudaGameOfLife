
#ifndef GLUT_DISPLAY_H
#define GLUT_DISPLAY_H

#include "CudaSharedGLBuffer.h"

#include <boost/scoped_ptr.hpp>
#include <boost/tr1/functional.hpp>
#include <boost/noncopyable.hpp>

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <GL/glut.h>

class GlutDisplay : private boost::noncopyable
{
    public:
        typedef std::tr1::function<void ( uchar4* , int )> redrawCallback_t;

    public:
        ~GlutDisplay();

        int getWidth() { return width; }
        int getHeight() { return height; }

        static GlutDisplay& createInstance(int& argc, char* argv[]);
        static GlutDisplay& getInstance();
        static void resetInstance();

        static void startAnimation(redrawCallback_t argRedrawCallback);

    private:
        GlutDisplay(int& argc, char* argv[]);

        static void draw();
        static void animateWhenIdle();

    private:
        int width;
        int height;
        
        boost::scoped_ptr<CudaSharedGLBuffer> sharedBufferPtr;

        redrawCallback_t redrawCallback;

        // For some reason simply defining a static variable doesn't work :(
        static boost::scoped_ptr<GlutDisplay>& getSingletonInstancePointer()
        {
            static boost::scoped_ptr<GlutDisplay> singletonInstance;
            return singletonInstance;
        }
};

#endif
