#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#define RUNS 10
#define WARMUP 3

__global__ void robertsKernel(const uchar4 *src, uchar4 *dst, int w, int h)
{
    int strideX = gridDim.x * blockDim.x;
    int strideY = gridDim.y * blockDim.y;
    for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < h; y += strideY) {
        for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < w; x += strideX) {
            int xc = x + 1 < w ? x + 1 : w - 1;
            int yc = y + 1 < h ? y + 1 : h - 1;
            uchar4 p00 = src[(long long)y * w + x];
            uchar4 p11 = src[(long long)yc * w + xc];
            uchar4 p10 = src[(long long)yc * w + x];
            uchar4 p01 = src[(long long)y * w + xc];
            double l00 = (299.0 * p00.x + 587.0 * p00.y + 114.0 * p00.z) / 1000.0;
            double l11 = (299.0 * p11.x + 587.0 * p11.y + 114.0 * p11.z) / 1000.0;
            double l10 = (299.0 * p10.x + 587.0 * p10.y + 114.0 * p10.z) / 1000.0;
            double l01 = (299.0 * p01.x + 587.0 * p01.y + 114.0 * p01.z) / 1000.0;
            double gx = l00 - l11;
            double gy = l10 - l01;
            int v = (int)sqrt(gx * gx + gy * gy);
            if (v > 255)
                v = 255;
            dst[(long long)y * w + x] = make_uchar4((unsigned char)v, (unsigned char)v,
                                                    (unsigned char)v, 0);
        }
    }
}

static void fail(const char *msg)
{
    fprintf(stderr, "ERROR: %s\n", msg);
    exit(1);
}

static void cpuRoberts(const uchar4 *src, uchar4 *dst, long long w, long long h)
{
    for (long long y = 0; y < h; y++) {
        for (long long x = 0; x < w; x++) {
            long long xc = x + 1 < w ? x + 1 : w - 1;
            long long yc = y + 1 < h ? y + 1 : h - 1;
            uchar4 p00 = src[y * w + x];
            uchar4 p11 = src[yc * w + xc];
            uchar4 p10 = src[yc * w + x];
            uchar4 p01 = src[y * w + xc];
            double l00 = (299.0 * p00.x + 587.0 * p00.y + 114.0 * p00.z) / 1000.0;
            double l11 = (299.0 * p11.x + 587.0 * p11.y + 114.0 * p11.z) / 1000.0;
            double l10 = (299.0 * p10.x + 587.0 * p10.y + 114.0 * p10.z) / 1000.0;
            double l01 = (299.0 * p01.x + 587.0 * p01.y + 114.0 * p01.z) / 1000.0;
            double gx = l00 - l11;
            double gy = l10 - l01;
            int v = (int)sqrt(gx * gx + gy * gy);
            if (v > 255)
                v = 255;
            dst[y * w + x] = make_uchar4((unsigned char)v, (unsigned char)v,
                                         (unsigned char)v, 0);
        }
    }
}

struct Config {
    int gx, gy, bx, by;
};

int main(void)
{
    const long long sizes[3] = {512LL, 4096LL, 10000LL};
    Config configs[] = {
        {1, 1, 32, 1}, {1, 1, 64, 1}, {1, 1, 128, 1}, {1, 1, 256, 1}, {1, 1, 512, 1},
        {1, 1, 32, 32}, {2, 2, 16, 16}, {4, 4, 16, 16}, {8, 8, 16, 16},
        {16, 16, 16, 16}, {32, 32, 16, 16}, {32, 32, 32, 32},
    };
    const int numConfigs = (int)(sizeof(configs) / sizeof(configs[0]));

    cudaError_t err;
    int devCount = 0;
    if ((err = cudaGetDeviceCount(&devCount)) != cudaSuccess || devCount < 1)
        fail("no CUDA device");

    cudaEvent_t start, stop;
    if ((err = cudaEventCreate(&start)) != cudaSuccess)
        fail("cudaEventCreate failed");
    if ((err = cudaEventCreate(&stop)) != cudaSuccess)
        fail("cudaEventCreate failed");

    for (int s = 0; s < 3; s++) {
        long long w = sizes[s], h = sizes[s];
        long long n = w * h;
        size_t bytes = (size_t)n * 4;

        uchar4 *src = (uchar4 *)malloc(bytes);
        uchar4 *dst = (uchar4 *)malloc(bytes);
        uchar4 *ref = (uchar4 *)malloc(bytes);
        if (!src || !dst || !ref)
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

        uchar4 *dsrc = NULL, *ddst = NULL;
        if ((err = cudaMalloc((void **)&dsrc, bytes)) != cudaSuccess)
            fail("cudaMalloc failed");
        if ((err = cudaMalloc((void **)&ddst, bytes)) != cudaSuccess)
            fail("cudaMalloc failed");
        if ((err = cudaMemcpy(dsrc, src, bytes, cudaMemcpyHostToDevice)) != cudaSuccess)
            fail("cudaMemcpy failed");

        cpuRoberts(src, ref, w, h);

        double cpuTotal = 0.0;
        for (int r = 0; r < RUNS; r++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            cpuRoberts(src, dst, w, h);
            auto t1 = std::chrono::high_resolution_clock::now();
            cpuTotal += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        printf("CPU %lld %.6f\n", n, cpuTotal / RUNS);

        for (int k = 0; k < numConfigs; k++) {
            dim3 grid(configs[k].gx, configs[k].gy);
            dim3 block(configs[k].bx, configs[k].by);
            for (int wu = 0; wu < WARMUP; wu++)
                robertsKernel<<<grid, block>>>(dsrc, ddst, (int)w, (int)h);
            if ((err = cudaGetLastError()) != cudaSuccess)
                fail("kernel launch failed");
            double total = 0.0;
            for (int r = 0; r < RUNS; r++) {
                if ((err = cudaEventRecord(start)) != cudaSuccess)
                    fail("cudaEventRecord failed");
                robertsKernel<<<grid, block>>>(dsrc, ddst, (int)w, (int)h);
                if ((err = cudaEventRecord(stop)) != cudaSuccess)
                    fail("cudaEventRecord failed");
                if ((err = cudaEventSynchronize(stop)) != cudaSuccess)
                    fail("cudaEventSynchronize failed");
                float ms = 0.0f;
                if ((err = cudaEventElapsedTime(&ms, start, stop)) != cudaSuccess)
                    fail("cudaEventElapsedTime failed");
                total += (double)ms;
            }
            if ((err = cudaMemcpy(dst, ddst, bytes, cudaMemcpyDeviceToHost)) != cudaSuccess)
                fail("cudaMemcpy failed");
            long long mism = 0;
            int maxd = 0;
            for (long long i = 0; i < n; i++) {
                if (dst[i].x != ref[i].x || dst[i].y != ref[i].y ||
                    dst[i].z != ref[i].z || dst[i].w != ref[i].w) {
                    mism++;
                    int d = (int)dst[i].x - (int)ref[i].x;
                    if (d < 0)
                        d = -d;
                    if (d > maxd)
                        maxd = d;
                }
            }
            if (maxd > 1)
                fail("verify failed");
            int blocks = configs[k].gx * configs[k].gy;
            int threads = configs[k].bx * configs[k].by;
            printf("BENCH %d %d %d %d %d %d %lld %.6f\n",
                   blocks, threads, configs[k].gx, configs[k].gy,
                   configs[k].bx, configs[k].by, n, total / RUNS);
            printf("VERIFY %d %d %lld %lld %d\n", blocks, threads, n, mism, maxd);
        }

        cudaFree(dsrc);
        cudaFree(ddst);
        free(src);
        free(dst);
        free(ref);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
