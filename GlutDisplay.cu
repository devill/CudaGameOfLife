
#define GL_GLEXT_PROTOTYPES

#include "GlutDisplay.h"
#include "GL/glext.h"
#include "GL/glx.h"
#include "NoInstance.hpp"

#include "CudaError.hpp"
#include "ChooseCudaDevice.hpp"

GlutDisplay::GlutDisplay(int& argc, char* argv[]) : width(0), height(0)
{
    CUDA_ERROR( cudaGLSetGLDevice( chooseCudaDevice() ) );

    glutInit(&argc,argv);

    width = glutGet(GLUT_SCREEN_WIDTH);
    height = glutGet(GLUT_SCREEN_HEIGHT);
    
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
    glutInitWindowSize( width, height );
    glutCreateWindow( "" );
    glutFullScreen();

    sharedBufferPtr.reset(new CudaSharedGLBuffer(width * height * 4));
}

GlutDisplay::~GlutDisplay()
{
}

void GlutDisplay::draw()
{
    GlutDisplay& display = getInstance();
    glClearColor( 0.0, 0.0, 0.0, 1.0 );
    glClear( GL_COLOR_BUFFER_BIT );
    glDrawPixels( display.width, display.height, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
    glutSwapBuffers();
}

void GlutDisplay::animateWhenIdle()
{
    static int ticks = 0;
    {
        GlutDisplay& display = getInstance();
        BufferCudaLock bufferLock(*(display.sharedBufferPtr));
        display.redrawCallback( bufferLock.getDeviceBuffer(), ++ticks );
    }

    glutPostRedisplay();
}

void GlutDisplay::startAnimation(redrawCallback_t argRedrawCallback)
{
    getInstance().redrawCallback = argRedrawCallback;

    glutDisplayFunc(draw);
    glutIdleFunc(animateWhenIdle);

    glutMainLoop();
}

GlutDisplay& GlutDisplay::createInstance(int& argc, char* argv[])
{
    if(getSingletonInstancePointer().get() != NULL) throw ExsistingInstance("Display singleton instance already exists");
    getSingletonInstancePointer().reset(new GlutDisplay(argc, argv));

    return getInstance(); 
}

void GlutDisplay::resetInstance()
{
    getSingletonInstancePointer().reset();
}


GlutDisplay& GlutDisplay::getInstance()
{
    if(getSingletonInstancePointer().get() == NULL) throw NoInstance("Display singleton instance was not created!");
    return *(getSingletonInstancePointer());
}
