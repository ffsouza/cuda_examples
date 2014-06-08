#include <stdio.h>

__global__ void add(int a, int b, int *c)
{
    *c = a + b;
}

int main()
{
    //----------- cuda devices info ---------------
    int cuda_count;
    cudaDeviceProp prop;

    cudaGetDeviceCount(&cuda_count);

    printf("Exist %d device with cuda support\n",cuda_count);
    
    for(int device=0; device<cuda_count;device++)
    {
        cudaGetDeviceProperties(&prop,device);

        printf("--- General Information for device: %d ---\n", device);
        printf("\tDevice name: %s\n", prop.name);
        printf("\tComputer capability: %d.%d\n", prop.major,prop.minor);
        printf("\tClock rate: %d\n",prop.clockRate);
        printf("\tDevice copy overlap: ");
        if(prop.deviceOverlap)
            printf("Enabled\n");
        else
            printf("Disabled\n");

        printf("\tKernel execition timeout: "); 
        if(prop.kernelExecTimeoutEnabled)
            printf("Enabled\n");
        else
            printf("Disabled\n");
    
        printf("\tZero copy supported (cudaHostAllocMapped): ");
        if(prop.canMapHostMemory)
            printf("Yes\n");
        else
            printf("Not\n"); 
    
        printf("\t---Memory Information---\n");
        printf("\tTotal global memory: %ld bytes\n",prop.totalGlobalMem);
        printf("\tTotal const memory: %ld bytes\n",prop.totalConstMem);
        printf("\tMax memory pitch: %ld bytes\n",prop.memPitch);
        printf("\tTexture alignment: %ld bytes\n",prop.textureAlignment);
        
        printf("\t---Multiprocessor Information---\n");
        printf("\tMultiprocessor count: %d\n",prop.multiProcessorCount);
        printf("\tShared memory per multiprocessor: %ld bytes\n",prop.sharedMemPerBlock);
        printf("\tRegisters per multiprocessor: %d\n",prop.regsPerBlock);
        printf("\tThreads in warp:: %d\n",prop.warpSize);
        printf("\tMax thread dimensions: %d, %d, %d\n",prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("\tMax grid dimensions: %d, %d, %d\n",prop.maxGridSize[0], prop.maxGridSize[1],prop.maxGridSize[2] );
        printf("\tMax texture 1D dimensions: %d\n",prop.maxTexture1D);
        printf("\tMax texture 2D dimensions: %d,%d\n",prop.maxTexture2D[0], prop.maxTexture2D[1]);
        printf("\tMax texture 3D dimensions: %d,%d,%d\n",prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]);
        
        printf("\tConcurrent kernels: "); 
        if(prop.concurrentKernels)
            printf("Enabled\n");
        else
            printf("Disabled\n");

        printf("\n");
    }

    int c;
    int *dev_c;

    cudaMalloc((void**)&dev_c,sizeof(int));

    add<<<1,1>>>(1,2,dev_c);
   
    cudaMemcpy(&c,dev_c,sizeof(int),cudaMemcpyDeviceToHost);
    printf("Resul: %d\n",c);
    cudaFree(dev_c);

    return 0;
}
