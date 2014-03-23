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

    printf("\t---Memory Information---\n");
    printf("\tTotal global memory: %ld bytes\n",prop.totalGlobalMem);
    printf("\tTotal const memory: %ld bytes\n",prop.totalConstMem);


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
