#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#define RUNS 3
#define WARMUP 1
#define CPU_RUNS 3
#define MAX_NC 32

__constant__ double c_avg[MAX_NC * 3];
__constant__ double c_inv[MAX_NC * 9];
__constant__ double c_logdet[MAX_NC];
__constant__ int c_nc;

__device__ int classifyPixel(uchar4 p)
{
    int best = 0;
    double bestScore = 0.0;
    for (int j = 0; j < c_nc; j++) {
        const double *avg = c_avg + j * 3;
        const double *inv = c_inv + j * 9;
        double d0 = (double)p.x - avg[0];
        double d1 = (double)p.y - avg[1];
        double d2 = (double)p.z - avg[2];
        double t0 = inv[0] * d0 + inv[1] * d1 + inv[2] * d2;
        double t1 = inv[3] * d0 + inv[4] * d1 + inv[5] * d2;
        double t2 = inv[6] * d0 + inv[7] * d1 + inv[8] * d2;
        double q = d0 * t0 + d1 * t1 + d2 * t2;
        double score = -q - c_logdet[j];
        if (j == 0 || score > bestScore) {
            bestScore = score;
            best = j;
        }
    }
    return best;
}

__global__ void classifyKernel(uchar4 *img, long long n)
{
    long long stride = (long long)gridDim.x * (long long)blockDim.x;
    for (long long i = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
         i < n; i += stride) {
        uchar4 p = img[i];
        p.w = (unsigned char)classifyPixel(p);
        img[i] = p;
    }
}

static void fail(const char *msg)
{
    fprintf(stderr, "ERROR: %s\n", msg);
    exit(1);
}

static void classifyHost(uchar4 *img, long long n, const double *avg, const double *inv,
                         const double *logdet, int nc)
{
    for (long long i = 0; i < n; i++) {
        uchar4 p = img[i];
        int best = 0;
        double bestScore = 0.0;
        for (int j = 0; j < nc; j++) {
            const double *a = avg + j * 3;
            const double *iv = inv + j * 9;
            double d0 = (double)p.x - a[0];
            double d1 = (double)p.y - a[1];
            double d2 = (double)p.z - a[2];
            double t0 = iv[0] * d0 + iv[1] * d1 + iv[2] * d2;
            double t1 = iv[3] * d0 + iv[4] * d1 + iv[5] * d2;
            double t2 = iv[6] * d0 + iv[7] * d1 + iv[8] * d2;
            double q = d0 * t0 + d1 * t1 + d2 * t2;
            double score = -q - logdet[j];
            if (j == 0 || score > bestScore) {
                bestScore = score;
                best = j;
            }
        }
        p.w = (unsigned char)best;
        img[i] = p;
    }
}

static void computeStats(const uchar4 *img, long long w, long long h, int nc, int np,
                         double *avg, double *inv, double *logdet)
{
    for (int j = 0; j < nc; j++) {
        unsigned int st = 123456789u + (unsigned int)j * 2654435761u;
        double sum[3] = {0.0, 0.0, 0.0};
        double s2[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (int i = 0; i < np; i++) {
            st = st * 1103515245u + 12345u;
            int x = (int)((st >> 16) % (unsigned int)w);
            st = st * 1103515245u + 12345u;
            int y = (int)((st >> 16) % (unsigned int)h);
            uchar4 p = img[(long long)y * w + x];
            double r = p.x;
            double g = p.y;
            double b = p.z;
            sum[0] += r;
            sum[1] += g;
            sum[2] += b;
            s2[0] += r * r;
            s2[1] += r * g;
            s2[2] += r * b;
            s2[3] += g * g;
            s2[4] += g * b;
            s2[5] += b * b;
        }

        double n = (double)np;
        double dn = (double)(np - 1);
        double a0 = sum[0] / n;
        double a1 = sum[1] / n;
        double a2 = sum[2] / n;
        double m00 = (s2[0] - n * a0 * a0) / dn;
        double m01 = (s2[1] - n * a0 * a1) / dn;
        double m02 = (s2[2] - n * a0 * a2) / dn;
        double m11 = (s2[3] - n * a1 * a1) / dn;
        double m12 = (s2[4] - n * a1 * a2) / dn;
        double m22 = (s2[5] - n * a2 * a2) / dn;

        double a00 = m11 * m22 - m12 * m12;
        double a01 = m12 * m02 - m01 * m22;
        double a02 = m01 * m12 - m11 * m02;
        double a11 = m00 * m22 - m02 * m02;
        double a12 = m01 * m02 - m00 * m12;
        double a22 = m00 * m11 - m01 * m01;

        double det = m00 * a00 + m01 * a01 + m02 * a02;

        avg[j * 3 + 0] = a0;
        avg[j * 3 + 1] = a1;
        avg[j * 3 + 2] = a2;
        double *iv = inv + j * 9;
        iv[0] = a00 / det;
        iv[1] = a01 / det;
        iv[2] = a02 / det;
        iv[3] = a01 / det;
        iv[4] = a11 / det;
        iv[5] = a12 / det;
        iv[6] = a02 / det;
        iv[7] = a12 / det;
        iv[8] = a22 / det;
        logdet[j] = log(fabs(det));
    }
}

struct Config {
    int blocks, threads;
};

int main(void)
{
    const long long sizes[3] = {512LL, 4096LL, 10000LL};
    Config configs[] = {
        {1, 32}, {1, 64}, {1, 128}, {1, 256}, {1, 512}, {1, 1024},
        {4, 256}, {16, 256}, {64, 256}, {224, 256}, {256, 256}, {1024, 256},
        {1024, 1024},
    };
    const int numConfigs = (int)(sizeof(configs) / sizeof(configs[0]));
    const int nc = 5;
    const int np = 1000;

    cudaError_t err;
    int devCount = 0;
    if ((err = cudaGetDeviceCount(&devCount)) != cudaSuccess || devCount < 1)
        fail("no CUDA device");

    cudaEvent_t start, stop;
    if ((err = cudaEventCreate(&start)) != cudaSuccess)
        fail("cudaEventCreate failed");
    if ((err = cudaEventCreate(&stop)) != cudaSuccess)
        fail("cudaEventCreate failed");

    printf("PARAMS %d %d\n", nc, np);
    fflush(stdout);

    for (int s = 0; s < 3; s++) {
        long long w = sizes[s], h = sizes[s];
        long long n = w * h;
        size_t bytes = (size_t)n * 4;
        fprintf(stderr, "[bench] набор %lldx%lld (%lld пикселей)\n", w, h, n);

        uchar4 *src = (uchar4 *)malloc(bytes);
        uchar4 *dst = (uchar4 *)malloc(bytes);
        if (!src || !dst)
            fail("host malloc failed");
        for (long long y = 0; y < h; y++) {
            for (long long x = 0; x < w; x++) {
                uchar4 p;
                p.x = (unsigned char)((x * 11 + y * 7) & 0xFF);
                p.y = (unsigned char)((x * 3 + y * 13) & 0xFF);
                p.z = (unsigned char)((x * 5 + y * 5 + 37) & 0xFF);
                p.w = 0;
                src[y * w + x] = p;
            }
        }

        double h_avg[MAX_NC * 3];
        double h_inv[MAX_NC * 9];
        double h_logdet[MAX_NC];
        computeStats(src, w, h, nc, np, h_avg, h_inv, h_logdet);

        fprintf(stderr, "[bench] %lldx%lld: CPU-эталон, %d запусков...\n", w, h, CPU_RUNS + 1);
        classifyHost(src, n, h_avg, h_inv, h_logdet, nc);
        double cpuTotal = 0.0;
        for (int r = 0; r < CPU_RUNS; r++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            classifyHost(src, n, h_avg, h_inv, h_logdet, nc);
            auto t1 = std::chrono::high_resolution_clock::now();
            cpuTotal += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        printf("CPU %lld %.6f\n", n, cpuTotal / CPU_RUNS);
        fflush(stdout);

        uchar4 *dImg = NULL;
        if ((err = cudaMalloc((void **)&dImg, bytes)) != cudaSuccess)
            fail("cudaMalloc failed");
        if ((err = cudaMemcpy(dImg, src, bytes, cudaMemcpyHostToDevice)) != cudaSuccess)
            fail("cudaMemcpy failed");
        if ((err = cudaMemcpyToSymbol(c_nc, &nc, sizeof(int))) != cudaSuccess)
            fail("cudaMemcpyToSymbol failed");
        if ((err = cudaMemcpyToSymbol(c_avg, h_avg, sizeof(double) * (size_t)nc * 3)) != cudaSuccess)
            fail("cudaMemcpyToSymbol failed");
        if ((err = cudaMemcpyToSymbol(c_inv, h_inv, sizeof(double) * (size_t)nc * 9)) != cudaSuccess)
            fail("cudaMemcpyToSymbol failed");
        if ((err = cudaMemcpyToSymbol(c_logdet, h_logdet, sizeof(double) * (size_t)nc)) != cudaSuccess)
            fail("cudaMemcpyToSymbol failed");

        for (int k = 0; k < numConfigs; k++) {
            dim3 grid(configs[k].blocks);
            dim3 block(configs[k].threads);
            fprintf(stderr, "[bench] %lldx%lld: конфигурация <<<%d, %d>>>...\n",
                    w, h, configs[k].blocks, configs[k].threads);
            for (int wu = 0; wu < WARMUP; wu++)
                classifyKernel<<<grid, block>>>(dImg, n);
            if ((err = cudaGetLastError()) != cudaSuccess)
                fail("kernel launch failed");
            double total = 0.0;
            for (int r = 0; r < RUNS; r++) {
                if ((err = cudaEventRecord(start)) != cudaSuccess)
                    fail("cudaEventRecord failed");
                classifyKernel<<<grid, block>>>(dImg, n);
                if ((err = cudaEventRecord(stop)) != cudaSuccess)
                    fail("cudaEventRecord failed");
                if ((err = cudaEventSynchronize(stop)) != cudaSuccess)
                    fail("cudaEventSynchronize failed");
                float ms = 0.0f;
                if ((err = cudaEventElapsedTime(&ms, start, stop)) != cudaSuccess)
                    fail("cudaEventElapsedTime failed");
                total += (double)ms;
            }
            if ((err = cudaMemcpy(dst, dImg, bytes, cudaMemcpyDeviceToHost)) != cudaSuccess)
                fail("cudaMemcpy failed");
            long long mism = 0;
            long long maxd = 0;
            for (long long i = 0; i < n; i++) {
                if (dst[i].x != src[i].x || dst[i].y != src[i].y ||
                    dst[i].z != src[i].z || dst[i].w != src[i].w) {
                    mism++;
                    long long d = (long long)dst[i].w - (long long)src[i].w;
                    if (d < 0)
                        d = -d;
                    if (d > maxd)
                        maxd = d;
                }
            }
            printf("BENCH %d %d %lld %.6f\n",
                   configs[k].blocks, configs[k].threads, n, total / RUNS);
            printf("VERIFY %d %d %lld %lld %lld\n",
                   configs[k].blocks, configs[k].threads, n, mism, maxd);
            fflush(stdout);
        }

        cudaFree(dImg);
        free(src);
        free(dst);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
