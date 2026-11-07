#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define AUX_GRID 128
#define AUX_BLOCK 256
#define ZERO_EPS 1e-7
#define RNG_MUL 6364136223846793005ULL
#define RNG_ADD 1442695040888963407ULL
#define RNG_SEED 42ULL

struct AbsLess {
    __host__ __device__ bool operator()(double a, double b) const {
        return fabs(a) < fabs(b);
    }
};

__global__ void swapRowsKernel(double *a, int n, int k, int pr)
{
    int stride = gridDim.x * blockDim.x;
    for (int j = k + (int)(blockIdx.x * blockDim.x + threadIdx.x); j < n; j += stride) {
        double t = a[j * n + k];
        a[j * n + k] = a[j * n + pr];
        a[j * n + pr] = t;
    }
}

__global__ void factorKernel(double *a, double *f, int n, int k, double pivot)
{
    int stride = gridDim.x * blockDim.x;
    for (int i = k + 1 + (int)(blockIdx.x * blockDim.x + threadIdx.x); i < n; i += stride)
        f[i] = a[k * n + i] / pivot;
}

__global__ void eliminateKernel(double *a, const double *f, int n, int k)
{
    int sx = gridDim.x * blockDim.x;
    int sy = gridDim.y * blockDim.y;
    for (int j = k + 1 + (int)(blockIdx.y * blockDim.y + threadIdx.y); j < n; j += sy)
        for (int i = k + 1 + (int)(blockIdx.x * blockDim.x + threadIdx.x); i < n; i += sx) {
            double fi = f[i];
            if (fi != 0.0)
                a[j * n + i] -= fi * a[j * n + k];
        }
}

__global__ void eliminateKernelUncoalesced(double *a, const double *f, int n, int k)
{
    int sx = gridDim.x * blockDim.x;
    int sy = gridDim.y * blockDim.y;
    for (int i = k + 1 + (int)(blockIdx.y * blockDim.y + threadIdx.y); i < n; i += sy)
        for (int j = k + 1 + (int)(blockIdx.x * blockDim.x + threadIdx.x); j < n; j += sx) {
            double fi = f[i];
            if (fi != 0.0)
                a[j * n + i] -= fi * a[j * n + k];
        }
}

static void fail(const char *msg)
{
    fprintf(stderr, "ERROR: %s\n", msg);
    exit(1);
}

static unsigned long long rngState;

static double rngUniform(void)
{
    rngState = rngState * RNG_MUL + RNG_ADD;
    return (double)(rngState >> 11) * (1.0 / 9007199254740992.0);
}

static void genMatrix(int n, double *hRow, double *hCol)
{
    rngState = RNG_SEED;
    double S = sqrt(12.0 * 2.718281828459045 / (double)n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double v = (rngUniform() - 0.5) * S;
            hRow[(size_t)i * n + j] = v;
            hCol[(size_t)j * n + i] = v;
        }
}

static cudaEvent_t evStart, evStop;

static double gpuDet(double *dA, const double *hCol, size_t matBytes, double *dF, int n,
                     int gx, int gy, int bx, int by, int uncoalesced,
                     double *eventMs, double *wallMs)
{
    cudaError_t err;
    if ((err = cudaMemcpy(dA, hCol, matBytes, cudaMemcpyHostToDevice)) != cudaSuccess)
        fail("cudaMemcpy host to device failed");
    dim3 grid(gx, gy);
    dim3 block(bx, by);
    auto t0 = std::chrono::high_resolution_clock::now();
    if ((err = cudaEventRecord(evStart)) != cudaSuccess)
        fail("cudaEventRecord failed");
    double det = 1.0;
    int sign = 1;
    for (int k = 0; k < n; k++) {
        double *col = dA + (size_t)k * n + k;
        double *mx = thrust::max_element(thrust::device, col, col + (n - k), AbsLess());
        int pr = k + (int)(mx - col);
        double pivot;
        if ((err = cudaMemcpy(&pivot, dA + (size_t)k * n + pr, sizeof(double), cudaMemcpyDeviceToHost)) != cudaSuccess)
            fail("cudaMemcpy device to host failed");
        if (fabs(pivot) < ZERO_EPS) {
            det = 0.0;
            break;
        }
        det *= pivot;
        if (pr != k) {
            sign = -sign;
            swapRowsKernel<<<AUX_GRID, AUX_BLOCK>>>(dA, n, k, pr);
        }
        if (k + 1 < n) {
            factorKernel<<<AUX_GRID, AUX_BLOCK>>>(dA, dF, n, k, pivot);
            if (uncoalesced)
                eliminateKernelUncoalesced<<<grid, block>>>(dA, dF, n, k);
            else
                eliminateKernel<<<grid, block>>>(dA, dF, n, k);
        }
    }
    if ((err = cudaGetLastError()) != cudaSuccess)
        fail("kernel error");
    if ((err = cudaEventRecord(evStop)) != cudaSuccess)
        fail("cudaEventRecord failed");
    if ((err = cudaEventSynchronize(evStop)) != cudaSuccess)
        fail("cudaEventSynchronize failed");
    float ms = 0.0f;
    if ((err = cudaEventElapsedTime(&ms, evStart, evStop)) != cudaSuccess)
        fail("cudaEventElapsedTime failed");
    auto t1 = std::chrono::high_resolution_clock::now();
    if (eventMs)
        *eventMs = (double)ms;
    if (wallMs)
        *wallMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return (det == 0.0) ? 0.0 : det * (double)sign;
}

static double cpuDet(const double *A0, double *A, int n)
{
    memcpy(A, A0, (size_t)n * n * sizeof(double));
    double det = 1.0;
    int sign = 1;
    for (int k = 0; k < n; k++) {
        int pr = k;
        for (int i = k + 1; i < n; i++)
            if (fabs(A[(size_t)i * n + k]) > fabs(A[(size_t)pr * n + k]))
                pr = i;
        if (fabs(A[(size_t)pr * n + k]) < ZERO_EPS) {
            det = 0.0;
            break;
        }
        if (pr != k) {
            sign = -sign;
            for (int j = 0; j < n; j++) {
                double t = A[(size_t)k * n + j];
                A[(size_t)k * n + j] = A[(size_t)pr * n + j];
                A[(size_t)pr * n + j] = t;
            }
        }
        det *= A[(size_t)k * n + k];
        double piv = A[(size_t)k * n + k];
        #pragma omp parallel for
        for (int i = k + 1; i < n; i++) {
            double fi = A[(size_t)i * n + k] / piv;
            if (fi != 0.0) {
                const double *rowk = A + (size_t)k * n;
                double *rowi = A + (size_t)i * n;
                for (int j = k + 1; j < n; j++)
                    rowi[j] -= fi * rowk[j];
            }
        }
    }
    return (det == 0.0) ? 0.0 : det * (double)sign;
}

static double relDiff(double a, double b)
{
    double d = a - b;
    if (d < 0.0)
        d = -d;
    double m = fabs(a) > fabs(b) ? fabs(a) : fabs(b);
    if (m == 0.0)
        return 0.0;
    return d / m;
}

struct Config { int gx, gy, bx, by; };
struct SizeSpec { int n, runs, warmup, cpuRuns; };

static double runConfig(double *dA, const double *hCol, size_t matBytes, double *dF,
                        int n, const Config &cfg, int runs, int warmup, int uncoalesced,
                        double *evAvg, double *wallAvg)
{
    for (int w = 0; w < warmup; w++)
        gpuDet(dA, hCol, matBytes, dF, n, cfg.gx, cfg.gy, cfg.bx, cfg.by, uncoalesced, NULL, NULL);
    double evSum = 0.0, wallSum = 0.0, det = 0.0;
    for (int r = 0; r < runs; r++) {
        double ev, wall;
        det = gpuDet(dA, hCol, matBytes, dF, n, cfg.gx, cfg.gy, cfg.bx, cfg.by, uncoalesced, &ev, &wall);
        evSum += ev;
        wallSum += wall;
    }
    *evAvg = evSum / runs;
    *wallAvg = wallSum / runs;
    return det;
}

static void printBench(const Config &cfg, int n, double evAvg, double wallAvg)
{
    printf("BENCH %d %d %d %d %d %d %d %.6f %.6f\n",
           cfg.gx * cfg.gy, cfg.bx * cfg.by, cfg.gx, cfg.gy, cfg.bx, cfg.by, n,
           evAvg, wallAvg);
}

int main(void)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    cudaError_t err;
    int devCount = 0;
    if ((err = cudaGetDeviceCount(&devCount)) != cudaSuccess || devCount < 1)
        fail("no CUDA device");
    if ((err = cudaEventCreate(&evStart)) != cudaSuccess)
        fail("cudaEventCreate failed");
    if ((err = cudaEventCreate(&evStop)) != cudaSuccess)
        fail("cudaEventCreate failed");
#ifdef _OPENMP
    int cpuThreads = omp_get_max_threads();
#else
    int cpuThreads = 1;
#endif

    Config configs[] = {
        {1, 1, 32, 1}, {1, 1, 64, 1}, {1, 1, 128, 1}, {1, 1, 256, 1}, {1, 1, 512, 1},
        {2, 2, 16, 16}, {4, 4, 16, 16}, {8, 8, 16, 16}, {16, 16, 16, 16},
        {32, 32, 16, 16}, {16, 16, 32, 8}, {32, 32, 32, 8}, {32, 32, 32, 32},
    };
    const int numCfg = (int)(sizeof(configs) / sizeof(configs[0]));
    const int workCfg = 10;

    SizeSpec sizes[] = {
        {256, 10, 3, 3},
        {1024, 5, 2, 3},
        {3000, 3, 1, 2},
    };
    const int numSizes = (int)(sizeof(sizes) / sizeof(sizes[0]));
    int bestCfg = workCfg;
    double bestEv = -1.0;

    for (int s = 0; s < numSizes; s++) {
        int n = sizes[s].n;
        size_t matBytes = (size_t)n * n * sizeof(double);
        double *hRow = (double *)malloc(matBytes);
        double *hCol = (double *)malloc(matBytes);
        double *hWork = (double *)malloc(matBytes);
        if (!hRow || !hCol || !hWork)
            fail("host malloc failed");
        genMatrix(n, hRow, hCol);

        double *dA = NULL, *dF = NULL;
        if ((err = cudaMalloc((void **)&dA, matBytes)) != cudaSuccess)
            fail("cudaMalloc failed");
        if ((err = cudaMalloc((void **)&dF, (size_t)n * sizeof(double))) != cudaSuccess)
            fail("cudaMalloc failed");

        double cpuTotal = 0.0;
        double detCpu = 0.0;
        printf("# n=%d: CPU-эталон, %d прогонов\n", n, sizes[s].cpuRuns);
        for (int r = 0; r < sizes[s].cpuRuns; r++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            detCpu = cpuDet(hRow, hWork, n);
            auto t1 = std::chrono::high_resolution_clock::now();
            cpuTotal += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        printf("CPU %d %.6f %d\n", n, cpuTotal / sizes[s].cpuRuns, cpuThreads);

        double detWork = 0.0;
        for (int c = 0; c < numCfg; c++) {
            if (n > 1024 && configs[c].gx == 1 && configs[c].gy == 1) {
                printf("# n=%d: пропуск 1D-конфигурации <<<%d, %d>>> (%d×%d; %d×%d)\n", n,
                       configs[c].gx * configs[c].gy, configs[c].bx * configs[c].by,
                       configs[c].gx, configs[c].gy, configs[c].bx, configs[c].by);
                continue;
            }
            printf("# n=%d: конфигурация <<<%d, %d>>> (%d×%d; %d×%d)\n", n,
                   configs[c].gx * configs[c].gy, configs[c].bx * configs[c].by,
                   configs[c].gx, configs[c].gy, configs[c].bx, configs[c].by);
            double evAvg, wallAvg;
            double det = runConfig(dA, hCol, matBytes, dF, n, configs[c],
                                   sizes[s].runs, sizes[s].warmup, 0, &evAvg, &wallAvg);
            printBench(configs[c], n, evAvg, wallAvg);
            printf("VERIFY %d %.3e\n", n, relDiff(det, detCpu));
            if (c == workCfg)
                detWork = det;
            if (n == 3000 && (bestEv < 0.0 || evAvg < bestEv)) {
                bestEv = evAvg;
                bestCfg = c;
            }
        }
        printf("DET %d %.15e %.15e\n", n, detCpu, detWork);

        int uRuns = (n > 1024) ? 1 : sizes[s].runs;
        int uWarm = (n > 1024) ? 1 : sizes[s].warmup;
        printf("# n=%d: uncoalesced-вариант, %d прогонов\n", n, uRuns);
        double evAvg, wallAvg;
        double detU = runConfig(dA, hCol, matBytes, dF, n, configs[workCfg],
                                uRuns, uWarm, 1, &evAvg, &wallAvg);
        printf("UNCOAL %d %.6f %.6f\n", n, evAvg, wallAvg);
        printf("VERIFYU %d %.3e\n", n, relDiff(detU, detCpu));

        cudaFree(dA);
        cudaFree(dF);
        free(hRow);
        free(hCol);
        free(hWork);
    }

    {
        int n = 4096;
        size_t matBytes = (size_t)n * n * sizeof(double);
        double *hRow = (double *)malloc(matBytes);
        double *hCol = (double *)malloc(matBytes);
        double *hWork = (double *)malloc(matBytes);
        if (!hRow || !hCol || !hWork)
            fail("host malloc failed");
        genMatrix(n, hRow, hCol);
        double *dA = NULL, *dF = NULL;
        if ((err = cudaMalloc((void **)&dA, matBytes)) != cudaSuccess)
            fail("cudaMalloc failed");
        if ((err = cudaMalloc((void **)&dF, (size_t)n * sizeof(double))) != cudaSuccess)
            fail("cudaMalloc failed");

        printf("# n=%d: CPU-эталон, 1 прогон\n", n);
        auto t0 = std::chrono::high_resolution_clock::now();
        double detCpu = cpuDet(hRow, hWork, n);
        auto t1 = std::chrono::high_resolution_clock::now();
        printf("CPU %d %.6f %d\n", n,
               std::chrono::duration<double, std::milli>(t1 - t0).count(), cpuThreads);

        double detWork = 0.0;
        for (int q = 0; q < 2; q++) {
            int c = (q == 0) ? workCfg : bestCfg;
            if (q == 1 && bestCfg == workCfg)
                break;
            printf("# n=%d: конфигурация <<<%d, %d>>> (%d×%d; %d×%d)\n", n,
                   configs[c].gx * configs[c].gy, configs[c].bx * configs[c].by,
                   configs[c].gx, configs[c].gy, configs[c].bx, configs[c].by);
            double evAvg, wallAvg;
            double det = runConfig(dA, hCol, matBytes, dF, n, configs[c], 2, 1, 0, &evAvg, &wallAvg);
            printBench(configs[c], n, evAvg, wallAvg);
            printf("VERIFY %d %.3e\n", n, relDiff(det, detCpu));
            if (c == workCfg)
                detWork = det;
        }
        printf("DET %d %.15e %.15e\n", n, detCpu, detWork);

        cudaFree(dA);
        cudaFree(dF);
        free(hRow);
        free(hCol);
        free(hWork);
    }

    cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);
    return 0;
}
