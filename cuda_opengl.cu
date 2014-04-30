#define GL_GLEXT_PROTOTYPES

#include "GL/glut.h"
#include "cuda.h"
#include "cuda_gl_interop.h"

#define DIM 512

GLuint bufferObj;
cudaGraphicsResource *resource;

static void draw_func(void)
{
    glDrawPixels(DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glutSwapBuffers();
}

static void key_func( unsigned char key, int x, int y ) 
{
    switch (key) 
    {
        case 27:
            // clean up OpenGL and CUDA
            cudaGraphicsUnregisterResource( resource );
            glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );
            glDeleteBuffers( 1, &bufferObj );
        exit(0);
    }
}

__global__ void kernel(uchar4 *ptr)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    // now calculate the value at that position
    float fx = x/(float)DIM - 0.5f;
    float fy = y/(float)DIM - 0.5f;
    unsigned char green = 128 + 127 * sin( abs(fx*100) - abs(fy*100) );

    // accessing uchar4 vs. unsigned char*
    ptr[offset].x = 0;
    ptr[offset].y = green;
    ptr[offset].z = 0;
    ptr[offset].w = 255;
}

int main(int argc, char **argv)
{
    cudaDeviceProp prop;
    int dev;

    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 1;
    prop.minor = 0;
    cudaChooseDevice(&dev, &prop);

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA); 
    glutInitWindowSize(DIM, DIM);
    glutCreateWindow("Bitmap");

    glGenBuffers(1, &bufferObj);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, DIM * DIM * 4, NULL, GL_DYNAMIC_DRAW_ARB);

    //cuda and openGL interoperability
    cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone); 

    uchar4* devPtr;
    size_t size;
    
    cudaGraphicsMapResources(1, &resource, NULL);
    
    cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource);

    dim3 grids(DIM/16, DIM/16);
    dim3 threads(16,16);

    kernel<<<grids, threads>>>(devPtr);

    cudaGraphicsUnmapResources(1, &resource, NULL);

    glutKeyboardFunc(key_func);
    glutDisplayFunc(draw_func);

    glutMainLoop();
}


