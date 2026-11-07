#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#define BLOCK_X 32
#define BLOCK_Y 8
#define GRID_X 16
#define GRID_Y 16
#define AUX_BLOCK 256
#define AUX_GRID 128
#ifndef IN_BUF_SIZE
#define IN_BUF_SIZE (1 << 20)
#endif
#define MAX_N 10000
#define ZERO_EPS 1e-7

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

static void fail(const char *msg)
{
    fprintf(stderr, "ERROR: %s\n", msg);
    exit(0);
}

static char inBuf[IN_BUF_SIZE];
static size_t inPos = 0, inLen = 0;

static const double POW10[23] = {
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11,
    1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22
};

static int readDouble(double *out)
{
    for (;;) {
        while (inPos < inLen) {
            char c = inBuf[inPos];
            if (c != ' ' && c != '\n' && c != '\r' && c != '\t' && c != '\v' && c != '\f')
                break;
            inPos++;
        }
        if (inPos < inLen)
            break;
        inPos = 0;
        inLen = fread(inBuf, 1, IN_BUF_SIZE, stdin);
        if (inLen == 0)
            return 0;
    }

    int neg = 0;
    {
        char c = inBuf[inPos];
        if (c == '+' || c == '-') {
            neg = (c == '-');
            inPos++;
        }
    }

    double val = 0.0;
    int any = 0;
    for (;;) {
        while (inPos < inLen && inBuf[inPos] >= '0' && inBuf[inPos] <= '9') {
            val = val * 10.0 + (double)(inBuf[inPos] - '0');
            any = 1;
            inPos++;
        }
        if (inPos < inLen)
            break;
        inPos = 0;
        inLen = fread(inBuf, 1, IN_BUF_SIZE, stdin);
        if (inLen == 0)
            break;
    }

    unsigned long long fval = 0;
    int fcnt = 0;
    if (inPos < inLen && inBuf[inPos] == '.') {
        inPos++;
        for (;;) {
            while (inPos < inLen && inBuf[inPos] >= '0' && inBuf[inPos] <= '9') {
                if (fcnt < 17) {
                    fval = fval * 10ULL + (unsigned long long)(inBuf[inPos] - '0');
                    fcnt++;
                }
                any = 1;
                inPos++;
            }
            if (inPos < inLen)
                break;
            inPos = 0;
            inLen = fread(inBuf, 1, IN_BUF_SIZE, stdin);
            if (inLen == 0)
                break;
        }
    }
    if (!any)
        return 0;
    if (fcnt > 0)
        val += (double)fval / POW10[fcnt];

    if (inPos < inLen && (inBuf[inPos] == 'e' || inBuf[inPos] == 'E')) {
        inPos++;
        int eneg = 0;
        if (inPos >= inLen) {
            inPos = 0;
            inLen = fread(inBuf, 1, IN_BUF_SIZE, stdin);
        }
        if (inPos < inLen && (inBuf[inPos] == '+' || inBuf[inPos] == '-')) {
            eneg = (inBuf[inPos] == '-');
            inPos++;
        }
        int eany = 0;
        int e = 0;
        for (;;) {
            while (inPos < inLen && inBuf[inPos] >= '0' && inBuf[inPos] <= '9') {
                if (e < 100000)
                    e = e * 10 + (inBuf[inPos] - '0');
                eany = 1;
                inPos++;
            }
            if (inPos < inLen)
                break;
            inPos = 0;
            inLen = fread(inBuf, 1, IN_BUF_SIZE, stdin);
            if (inLen == 0)
                break;
        }
        if (!eany)
            return 0;
        double m = (e <= 22) ? POW10[e] : pow(10.0, (double)e);
        if (eneg)
            val /= m;
        else
            val *= m;
    }

    *out = neg ? -val : val;
    return 1;
}

int main(void)
{
    double dn;
    if (!readDouble(&dn))
        fail("failed to read matrix size");
    if (dn < 0.0 || dn > (double)MAX_N || dn != floor(dn))
        fail("invalid matrix size");
    int n = (int)dn;
    if (n == 0) {
        if (printf("1.0000000000e+00\n") < 0)
            fail("failed to write result");
        return 0;
    }

    size_t matBytes = (size_t)n * (size_t)n * sizeof(double);
    double *hA = (double *)malloc(matBytes);
    if (!hA)
        fail("out of memory");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double v;
            if (!readDouble(&v))
                fail("failed to parse matrix element");
            // храним по столбцам
            hA[(size_t)j * n + i] = v;
        }
    }

    cudaError_t err;
    int devCount = 0;
    if ((err = cudaGetDeviceCount(&devCount)) != cudaSuccess || devCount < 1)
        fail("no CUDA device available");

    double *dA = NULL, *dF = NULL;
    if ((err = cudaMalloc((void **)&dA, matBytes)) != cudaSuccess)
        fail("cudaMalloc failed");
    if ((err = cudaMalloc((void **)&dF, (size_t)n * sizeof(double))) != cudaSuccess)
        fail("cudaMalloc failed");
    if ((err = cudaMemcpy(dA, hA, matBytes, cudaMemcpyHostToDevice)) != cudaSuccess)
        fail("cudaMemcpy host to device failed");

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid(GRID_X, GRID_Y);

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
            if ((err = cudaGetLastError()) != cudaSuccess)
                fail("swap kernel launch failed");
        }
        if (k + 1 < n) {
            factorKernel<<<AUX_GRID, AUX_BLOCK>>>(dA, dF, n, k, pivot);
            if ((err = cudaGetLastError()) != cudaSuccess)
                fail("factor kernel launch failed");
            eliminateKernel<<<grid, block>>>(dA, dF, n, k);
            if ((err = cudaGetLastError()) != cudaSuccess)
                fail("eliminate kernel launch failed");
        }
    }
    if ((err = cudaDeviceSynchronize()) != cudaSuccess)
        fail("cudaDeviceSynchronize failed");

    double res = (det == 0.0) ? 0.0 : det * (double)sign;
    if (printf("%.10e\n", res) < 0)
        fail("failed to write result");

    cudaFree(dA);
    cudaFree(dF);
    free(hA);
    return 0;
}
