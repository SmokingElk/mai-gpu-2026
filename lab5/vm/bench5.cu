/* Бенчмарк для ЛР5: битоническая сортировка int.
 * Ядра и логика сортировки идентичны lab5.cu, конфигурации запуска параметризованы.
 * Вывод (parseable):
 *   CPU <n> <ms> <threads>          - эталон std::sort на CPU
 *   BENCH MAIN <sg> <sb> <mg> <n> <evMs> <wallMs>  - рабочая конфигурация
 *   BENCH SWEEP <sg> <sb> <mg> <n> <evMs> <wallMs> - свип конфигураций
 *   COPY <n> <h2dMs> <d2hMs>       - время копирования
 *   VERIFY <n> <ok>                - сверка с CPU-эталоном (memcmp)
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>

#define SMEM_ELEMS 8192
#define SMEM_JMAX 4096
#define MERGE_BLOCK 1024
#define ELEMS_PER_THREAD (SMEM_ELEMS / MERGE_BLOCK)
#define FILL_GRID 512
#define FILL_BLOCK 256
#define WORK_STAGE_GRID 512
#define WORK_STAGE_BLOCK 256
#define WORK_MERGE_GRID 512

#define RNG_MUL 6364136223846793005ULL
#define RNG_ADD 1442695040888963407ULL
#define RNG_SEED 42ULL

__global__ void fillTailKernel(int *data, int from, int to)
{
    int stride = gridDim.x * blockDim.x;
    for (int i = from + blockIdx.x * blockDim.x + threadIdx.x; i < to; i += stride)
        data[i] = INT_MAX;
}

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
    exit(1);
}

static void checkCuda(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
        fail(msg);
}

static unsigned long long rngState;

static void genArray(int n, int *h)
{
    rngState = RNG_SEED;
    for (int i = 0; i < n; i++) {
        rngState = rngState * RNG_MUL + RNG_ADD;
        h[i] = (int)(rngState >> 32);
    }
}

static cudaEvent_t evStart, evStop;

static void gpuSortOnce(int *dData, int n, int m, int stageGrid, int stageBlock, int mergeGrid)
{
    if (m > n) {
        fillTailKernel<<<FILL_GRID, FILL_BLOCK>>>(dData, n, m);
    }
    if (m > 1) {
        if (m <= SMEM_ELEMS) {
            bitonicSortSmem<<<1, MERGE_BLOCK>>>(dData, m);
        } else {
            for (int k = 2; k <= m; k <<= 1) {
                for (int j = k >> 1; j > SMEM_JMAX; j >>= 1)
                    bitonicStageGlobal<<<stageGrid, stageBlock>>>(dData, m, k, j);
                int jstart = (k >> 1) < SMEM_JMAX ? (k >> 1) : SMEM_JMAX;
                bitonicMergeSmem<<<mergeGrid, MERGE_BLOCK>>>(dData, m, k, jstart);
            }
        }
    }
    checkCuda(cudaGetLastError(), "kernel launch failed");
}

static void runConfig(int *dData, const int *hSrc, int n, int m,
                      int stageGrid, int stageBlock, int mergeGrid,
                      int runs, int warmup, double *evAvg, double *wallAvg)
{
    size_t inBytes = (size_t)n * 4;
    checkCuda(cudaMemcpy(dData, hSrc, inBytes, cudaMemcpyHostToDevice),
              "cudaMemcpy host to device failed");
    for (int w = 0; w < warmup; w++)
        gpuSortOnce(dData, n, m, stageGrid, stageBlock, mergeGrid);
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    double evSum = 0.0, wallSum = 0.0;
    for (int r = 0; r < runs; r++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        checkCuda(cudaEventRecord(evStart), "cudaEventRecord failed");
        gpuSortOnce(dData, n, m, stageGrid, stageBlock, mergeGrid);
        checkCuda(cudaEventRecord(evStop), "cudaEventRecord failed");
        checkCuda(cudaEventSynchronize(evStop), "cudaEventSynchronize failed");
        float ms = 0.0f;
        checkCuda(cudaEventElapsedTime(&ms, evStart, evStop), "cudaEventElapsedTime failed");
        auto t1 = std::chrono::high_resolution_clock::now();
        evSum += (double)ms;
        wallSum += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    *evAvg = evSum / runs;
    *wallAvg = wallSum / runs;
}

struct SizeSpec { int n; int runs; int warmup; int cpuRuns; };
struct StageCfg { int grid; int block; };

int main(void)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    int devCount = 0;
    checkCuda(cudaGetDeviceCount(&devCount), "cudaGetDeviceCount failed");
    if (devCount < 1)
        fail("no CUDA device");
    checkCuda(cudaEventCreate(&evStart), "cudaEventCreate failed");
    checkCuda(cudaEventCreate(&evStop), "cudaEventCreate failed");

    SizeSpec sizes[] = {
        {1 << 10, 20, 3, 5},
        {8192, 20, 3, 5},
        {1 << 16, 10, 3, 3},
        {1 << 18, 10, 2, 3},
        {1 << 20, 5, 2, 2},
        {1 << 22, 5, 2, 2},
        {1 << 24, 3, 1, 1},
        {1 << 26, 3, 1, 1},
    };
    const int numSizes = (int)(sizeof(sizes) / sizeof(sizes[0]));

    for (int s = 0; s < numSizes; s++) {
        int n = sizes[s].n;
        int m = 1;
        while (m < n)
            m <<= 1;
        size_t inBytes = (size_t)n * 4;
        int *hSrc = (int *)malloc(inBytes);
        int *hRef = (int *)malloc(inBytes);
        int *hOut = (int *)malloc(inBytes);
        if (!hSrc || !hRef || !hOut)
            fail("host malloc failed");
        genArray(n, hSrc);

        printf("# n=%d: CPU-эталон std::sort, %d прогонов\n", n, sizes[s].cpuRuns);
        double cpuSum = 0.0;
        for (int r = 0; r < sizes[s].cpuRuns; r++) {
            memcpy(hRef, hSrc, inBytes);
            auto t0 = std::chrono::high_resolution_clock::now();
            std::sort(hRef, hRef + n);
            auto t1 = std::chrono::high_resolution_clock::now();
            cpuSum += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        printf("CPU %d %.6f 1\n", n, cpuSum / sizes[s].cpuRuns);

        int *dData = NULL;
        checkCuda(cudaMalloc((void **)&dData, (size_t)m * 4), "cudaMalloc failed");

        auto c0 = std::chrono::high_resolution_clock::now();
        checkCuda(cudaMemcpy(dData, hSrc, inBytes, cudaMemcpyHostToDevice),
                  "cudaMemcpy host to device failed");
        auto c1 = std::chrono::high_resolution_clock::now();
        double h2dMs = std::chrono::duration<double, std::milli>(c1 - c0).count();

        printf("# n=%d: рабочая конфигурация stage<<<%d, %d>>>, merge<<<%d, %d>>>, %d прогонов\n",
               n, WORK_STAGE_GRID, WORK_STAGE_BLOCK, WORK_MERGE_GRID, MERGE_BLOCK, sizes[s].runs);
        double evAvg, wallAvg;
        runConfig(dData, hSrc, n, m, WORK_STAGE_GRID, WORK_STAGE_BLOCK, WORK_MERGE_GRID,
                  sizes[s].runs, sizes[s].warmup, &evAvg, &wallAvg);
        printf("BENCH MAIN %d %d %d %d %.6f %.6f\n",
               WORK_STAGE_GRID, WORK_STAGE_BLOCK, WORK_MERGE_GRID, n, evAvg, wallAvg);

        auto c2 = std::chrono::high_resolution_clock::now();
        checkCuda(cudaMemcpy(hOut, dData, inBytes, cudaMemcpyDeviceToHost),
                  "cudaMemcpy device to host failed");
        auto c3 = std::chrono::high_resolution_clock::now();
        double d2hMs = std::chrono::duration<double, std::milli>(c3 - c2).count();
        printf("COPY %d %.6f %.6f\n", n, h2dMs, d2hMs);

        int ok = memcmp(hOut, hRef, inBytes) == 0;
        printf("VERIFY %d %d\n", n, ok);

        cudaFree(dData);
        free(hSrc);
        free(hRef);
        free(hOut);
    }

    {
        int n = 1 << 22;
        int m = n;
        size_t inBytes = (size_t)n * 4;
        int *hSrc = (int *)malloc(inBytes);
        int *hRef = (int *)malloc(inBytes);
        int *hOut = (int *)malloc(inBytes);
        if (!hSrc || !hRef || !hOut)
            fail("host malloc failed");
        genArray(n, hSrc);
        memcpy(hRef, hSrc, inBytes);
        std::sort(hRef, hRef + n);

        int *dData = NULL;
        checkCuda(cudaMalloc((void **)&dData, (size_t)m * 4), "cudaMalloc failed");

        StageCfg stageCfgs[] = {
            {64, 128}, {128, 128}, {256, 128}, {512, 128},
            {128, 256}, {256, 256}, {512, 256}, {1024, 256},
        };
        int mergeGrids[] = {56, 112, 224, 448, 896};

        double bestEv = -1.0;
        int bestSg = WORK_STAGE_GRID, bestSb = WORK_STAGE_BLOCK, bestMg = WORK_MERGE_GRID;

        for (int c = 0; c < (int)(sizeof(stageCfgs) / sizeof(stageCfgs[0])); c++) {
            int sg = stageCfgs[c].grid, sb = stageCfgs[c].block;
            printf("# n=%d: свип stage<<<%d, %d>>> (merge<<<%d, %d>>>)\n",
                   n, sg, sb, WORK_MERGE_GRID, MERGE_BLOCK);
            double evAvg, wallAvg;
            runConfig(dData, hSrc, n, m, sg, sb, WORK_MERGE_GRID, 3, 1, &evAvg, &wallAvg);
            printf("BENCH SWEEP %d %d %d %d %.6f %.6f\n",
                   sg, sb, WORK_MERGE_GRID, n, evAvg, wallAvg);
            if (bestEv < 0.0 || evAvg < bestEv) {
                bestEv = evAvg;
                bestSg = sg;
                bestSb = sb;
                bestMg = WORK_MERGE_GRID;
            }
        }
        for (int c = 0; c < (int)(sizeof(mergeGrids) / sizeof(mergeGrids[0])); c++) {
            int mg = mergeGrids[c];
            printf("# n=%d: свип merge<<<%d, %d>>> (stage<<<%d, %d>>>)\n",
                   n, mg, MERGE_BLOCK, WORK_STAGE_GRID, WORK_STAGE_BLOCK);
            double evAvg, wallAvg;
            runConfig(dData, hSrc, n, m, WORK_STAGE_GRID, WORK_STAGE_BLOCK, mg, 3, 1,
                      &evAvg, &wallAvg);
            printf("BENCH SWEEP %d %d %d %d %.6f %.6f\n",
                   WORK_STAGE_GRID, WORK_STAGE_BLOCK, mg, n, evAvg, wallAvg);
            if (bestEv < 0.0 || evAvg < bestEv) {
                bestEv = evAvg;
                bestSg = WORK_STAGE_GRID;
                bestSb = WORK_STAGE_BLOCK;
                bestMg = mg;
            }
        }

        printf("# n=%d: проверка лучшей конфигурации stage<<<%d, %d>>, merge<<<%d, %d>>>\n",
               n, bestSg, bestSb, bestMg, MERGE_BLOCK);
        double evAvg, wallAvg;
        runConfig(dData, hSrc, n, m, bestSg, bestSb, bestMg, 1, 0, &evAvg, &wallAvg);
        checkCuda(cudaMemcpy(hOut, dData, inBytes, cudaMemcpyDeviceToHost),
                  "cudaMemcpy device to host failed");
        int ok = memcmp(hOut, hRef, inBytes) == 0;
        printf("VERIFY %d %d\n", n, ok);

        cudaFree(dData);
        free(hSrc);
        free(hRef);
        free(hOut);
    }

    cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);
    return 0;
}
