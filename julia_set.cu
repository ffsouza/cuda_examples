#include <stdio.h>
#include <SOIL/SOIL.h>

#define DIM 500

struct cuComplex {
    float   r;
    float   i;
    __device__ cuComplex( float a, float b ) : r(a), i(b)  {}
    __device__ float magnitude2( void ) {
        return r * r + i * i;
    }
    __device__ cuComplex operator*(const cuComplex& a) {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    __device__ cuComplex operator+(const cuComplex& a) {
        return cuComplex(r+a.r, i+a.i);
    }
};

__device__ int julia(int x, int y)
{
    const float scale = 1;
    float jx = scale * (float) (DIM/2 - x)/(DIM/2);
    float jy = scale * (float) (DIM/2 - y)/(DIM/2);

    cuComplex c(-0.8,0.156);
    cuComplex a(jx,jy);

    for(int i=0; i<200;i++)
    {
        a = a * a + c;
    
        if(a.magnitude2() > 1000)
            return 0;
    }

    return 1;
}

__global__ void kernel(unsigned char *ptr)
{
    int x = blockIdx.x;
    int y = blockIdx.y;

    int offset = x + y * gridDim.x;

    int julia_value = julia(x,y);

    ptr[offset*4+0] = 255 * julia_value;
    ptr[offset*4+1] = 0;
    ptr[offset*4+2] = 0;
    ptr[offset*4+3] = 255;
}

int main(void)
{

    unsigned char *pixels = new unsigned char[DIM * DIM * 4];
    unsigned char *dev_pixels;

    cudaMalloc((void**)&dev_pixels, DIM*DIM*4);
   
    dim3 grid(DIM,DIM);

    kernel<<<grid, 1>>>(dev_pixels);

    cudaMemcpy(pixels, dev_pixels, DIM * DIM * 4, cudaMemcpyDeviceToHost);

    int err = SOIL_save_image("img.bmp", SOIL_SAVE_TYPE_BMP, DIM, DIM, 4,&pixels[0]);

    cudaFree(dev_pixels);

    return 0;
}
