#include <cstdio>
#include <cuda_runtime.h>

int main(void)
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) {
        fprintf(stderr, "ERROR: no CUDA device\n");
        return 1;
    }
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaGetDeviceProperties failed\n");
        return 1;
    }
    printf("Name                    : %s\n", p.name);
    printf("Compute capability      : %d.%d\n", p.major, p.minor);
    printf("Multiprocessors count   : %d\n", p.multiProcessorCount);
    printf("Total global memory     : %zu\n", p.totalGlobalMem);
    printf("Shared memory per block : %zu\n", p.sharedMemPerBlock);
    printf("Total constant memory   : %zu\n", p.totalConstMem);
    printf("Registers per block     : %d\n", p.regsPerBlock);
    printf("Warp size               : %d\n", p.warpSize);
    printf("Max threads per block   : (%d, %d, %d)\n", p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
    printf("Max block               : (%d, %d, %d)\n", p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
    return 0;
}
