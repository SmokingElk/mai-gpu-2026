#include <cstdio>
#include <cstdlib>
#include <climits>

#define STAGE_BLOCK 256
#define STAGE_GRID 512
#define MERGE_BLOCK 1024
#define MERGE_GRID 512
#define FILL_BLOCK 256
#define FILL_GRID 512
#define SMEM_ELEMS 8192
#define SMEM_JMAX 4096
#define ELEMS_PER_THREAD (SMEM_ELEMS / MERGE_BLOCK)
#define MAX_N 256000000

/* Заполняет хвост [from, to) значением INT_MAX (дополнение до степени двойки). */
__global__ void fillTailKernel(int *data, int from, int to)
{
    int stride = gridDim.x * blockDim.x;
    for (int i = from + blockIdx.x * blockDim.x + threadIdx.x; i < to; i += stride)
        data[i] = INT_MAX;
}

/* Одна стадия (k, j) битонической сортировки через глобальную память.
 * Используется, когда рабочий чанк 2*j не помещается в разделяемую память. */
__global__ void bitonicStageGlobal(int *data, int m, int k, int j)
{
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m; i += stride) {
        int ixj = i ^ j;
        if (ixj > i) {
            int a = data[i];
            int b = data[ixj];
            int asc = ((i & k) == 0);
            if ((asc && a > b) || (!asc && a < b)) {
                data[i] = b;
                data[ixj] = a;
            }
        }
    }
}

/* Адаптивное битоническое слияние через разделяемую память:
 * каждый блок обрабатывает чанк из SMEM_ELEMS элементов и выполняет
 * все стадии слияния от jstart до 1 внутри чанка. */
__global__ void bitonicMergeSmem(int *data, int m, int k, int jstart)
{
    __shared__ int smem[SMEM_ELEMS];
    int tid = threadIdx.x;
    int cstride = gridDim.x * SMEM_ELEMS;
    for (int base = blockIdx.x * SMEM_ELEMS; base < m; base += cstride) {
        for (int q = 0; q < ELEMS_PER_THREAD; q++)
            smem[tid + q * MERGE_BLOCK] = data[base + tid + q * MERGE_BLOCK];
        __syncthreads();
        for (int j = jstart; j > 0; j >>= 1) {
            for (int q = 0; q < ELEMS_PER_THREAD; q++) {
                int i = tid + q * MERGE_BLOCK;
                int ixj = i ^ j;
                if (ixj > i) {
                    int a = smem[i];
                    int b = smem[ixj];
                    int asc = ((base + i) & k) == 0;
                    if ((asc && a > b) || (!asc && a < b)) {
                        smem[i] = b;
                        smem[ixj] = a;
                    }
                }
            }
            __syncthreads();
        }
        for (int q = 0; q < ELEMS_PER_THREAD; q++)
            data[base + tid + q * MERGE_BLOCK] = smem[tid + q * MERGE_BLOCK];
        __syncthreads();
    }
}

/* Полная битоническая сортировка одним блоком в разделяемой памяти —
 * случай, когда весь массив помещается в разделяемую память. */
__global__ void bitonicSortSmem(int *data, int m)
{
    __shared__ int smem[SMEM_ELEMS];
    int tid = threadIdx.x;
    for (int q = 0; q < ELEMS_PER_THREAD; q++) {
        int i = tid + q * MERGE_BLOCK;
        if (i < m)
            smem[i] = data[i];
    }
    __syncthreads();
    for (int k = 2; k <= m; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int q = 0; q < ELEMS_PER_THREAD; q++) {
                int i = tid + q * MERGE_BLOCK;
                if (i < m) {
                    int ixj = i ^ j;
                    if (ixj > i) {
                        int a = smem[i];
                        int b = smem[ixj];
                        int asc = (i & k) == 0;
                        if ((asc && a > b) || (!asc && a < b)) {
                            smem[i] = b;
                            smem[ixj] = a;
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
    for (int q = 0; q < ELEMS_PER_THREAD; q++) {
        int i = tid + q * MERGE_BLOCK;
        if (i < m)
            data[i] = smem[i];
    }
}

static void fail(const char *msg)
{
    fprintf(stderr, "ERROR: %s\n", msg);
    exit(0);
}

static void checkCuda(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
        fail(msg);
}

int main(void)
{
    int n = 0;
    if (fread(&n, 4, 1, stdin) != 1)
        fail("failed to read array size");
    if (n < 0)
        fail("invalid array size");
    if (n == 0)
        return 0;
    if (n > MAX_N)
        fail("array is too large");

    size_t inBytes = (size_t)n * 4;

    int pinned = 1;
    int *hBuf = NULL;
    cudaError_t err = cudaHostAlloc((void **)&hBuf, inBytes, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        pinned = 0;
        hBuf = (int *)malloc(inBytes);
        if (!hBuf)
            fail("out of memory");
    }
    if (fread(hBuf, 1, inBytes, stdin) != inBytes)
        fail("input is truncated");

    int m = 1;
    while (m < n)
        m <<= 1;

    int devCount = 0;
    if ((err = cudaGetDeviceCount(&devCount)) != cudaSuccess || devCount < 1)
        fail("no CUDA device available");

    int *dData = NULL;
    if ((err = cudaMalloc((void **)&dData, (size_t)m * 4)) != cudaSuccess)
        fail("cudaMalloc failed");
    if ((err = cudaMemcpy(dData, hBuf, inBytes, cudaMemcpyHostToDevice)) != cudaSuccess)
        fail("cudaMemcpy host to device failed");

    if (m > n) {
        fillTailKernel<<<FILL_GRID, FILL_BLOCK>>>(dData, n, m);
        checkCuda(cudaGetLastError(), "fill kernel launch failed");
    }

    if (m > 1) {
        if (m <= SMEM_ELEMS) {
            bitonicSortSmem<<<1, MERGE_BLOCK>>>(dData, m);
            checkCuda(cudaGetLastError(), "sort kernel launch failed");
        } else {
            for (int k = 2; k <= m; k <<= 1) {
                for (int j = k >> 1; j > SMEM_JMAX; j >>= 1) {
                    bitonicStageGlobal<<<STAGE_GRID, STAGE_BLOCK>>>(dData, m, k, j);
                    checkCuda(cudaGetLastError(), "stage kernel launch failed");
                }
                int jstart = (k >> 1) < SMEM_JMAX ? (k >> 1) : SMEM_JMAX;
                bitonicMergeSmem<<<MERGE_GRID, MERGE_BLOCK>>>(dData, m, k, jstart);
                checkCuda(cudaGetLastError(), "merge kernel launch failed");
            }
        }
    }

    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

    if ((err = cudaMemcpy(hBuf, dData, inBytes, cudaMemcpyDeviceToHost)) != cudaSuccess)
        fail("cudaMemcpy device to host failed");

    if (fwrite(hBuf, 1, inBytes, stdout) != inBytes)
        fail("failed to write output data");

    cudaFree(dData);
    if (pinned)
        cudaFreeHost(hBuf);
    else
        free(hBuf);
    return 0;
}
