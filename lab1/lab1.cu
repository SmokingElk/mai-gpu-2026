#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <cctype>

#define MAX_N (1LL << 25)
#define BLOCKS 256
#define THREADS 256

__global__ void vecMulKernel(const double *a, const double *b, double *c, long long n)
{
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        c[i] = a[i] * b[i];
}

static void fail(const char *msg)
{
    fprintf(stderr, "ERROR: %s\n", msg);
    exit(0);
}

static char *readAllInput(void)
{
    size_t cap = 1 << 20, size = 0;
    char *buf = (char *)malloc(cap);
    if (!buf)
        return NULL;
    for (;;) {
        if (size == cap) {
            char *nb = (char *)realloc(buf, cap << 1);
            if (!nb) {
                free(buf);
                return NULL;
            }
            buf = nb;
            cap <<= 1;
        }
        size_t r = fread(buf + size, 1, cap - size, stdin);
        if (r == 0)
            break;
        size += r;
    }
    if (ferror(stdin)) {
        free(buf);
        return NULL;
    }
    buf[size] = '\0';
    return buf;
}

static long long parseN(char **pp)
{
    char *p = *pp;
    while (*p != '\0' && isspace((unsigned char)*p))
        p++;
    if (*p == '\0')
        return -1;
    errno = 0;
    char *end = NULL;
    long long n = strtoll(p, &end, 10);
    if (end == p || errno == ERANGE)
        return -1;
    if (*end != '\0' && !isspace((unsigned char)*end))
        return -1;
    if (n < 0 || n >= MAX_N)
        return -1;
    *pp = end;
    return n;
}

static int parseVec(char **pp, double *v, long long n)
{
    char *p = *pp;
    for (long long i = 0; i < n; i++) {
        char *end = NULL;
        double x = strtod(p, &end);
        if (end == p)
            return -1;
        v[i] = x;
        p = end;
    }
    *pp = p;
    return 0;
}

int main(void)
{
    char *buf = readAllInput();
    if (!buf)
        fail("cannot read input");

    char *p = buf;
    long long n = parseN(&p);
    if (n < 0)
        fail("invalid vector size");

    if (n == 0) {
        free(buf);
        return 0;
    }

    double *a = (double *)malloc((size_t)n * sizeof(double));
    double *b = (double *)malloc((size_t)n * sizeof(double));
    double *c = (double *)malloc((size_t)n * sizeof(double));
    if (!a || !b || !c)
        fail("out of memory");

    if (parseVec(&p, a, n) != 0 || parseVec(&p, b, n) != 0)
        fail("invalid vector data");

    free(buf);

    cudaError_t err;
    int devCount = 0;
    if ((err = cudaGetDeviceCount(&devCount)) != cudaSuccess || devCount < 1)
        fail("no CUDA device available");

    double *da = NULL, *db = NULL, *dc = NULL;
    if ((err = cudaMalloc((void **)&da, (size_t)n * sizeof(double))) != cudaSuccess)
        fail("cudaMalloc failed");
    if ((err = cudaMalloc((void **)&db, (size_t)n * sizeof(double))) != cudaSuccess)
        fail("cudaMalloc failed");
    if ((err = cudaMalloc((void **)&dc, (size_t)n * sizeof(double))) != cudaSuccess)
        fail("cudaMalloc failed");

    if ((err = cudaMemcpy(da, a, (size_t)n * sizeof(double), cudaMemcpyHostToDevice)) != cudaSuccess)
        fail("cudaMemcpy host to device failed");
    if ((err = cudaMemcpy(db, b, (size_t)n * sizeof(double), cudaMemcpyHostToDevice)) != cudaSuccess)
        fail("cudaMemcpy host to device failed");

    vecMulKernel<<<BLOCKS, THREADS>>>(da, db, dc, n);
    if ((err = cudaGetLastError()) != cudaSuccess)
        fail("kernel launch failed");

    if ((err = cudaMemcpy(c, dc, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost)) != cudaSuccess)
        fail("cudaMemcpy device to host failed");
    if ((err = cudaDeviceSynchronize()) != cudaSuccess)
        fail("cudaDeviceSynchronize failed");

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    size_t obufCap = 1 << 20;
    char *obuf = (char *)malloc(obufCap);
    if (!obuf)
        fail("out of memory");
    size_t pos = 0;
    for (long long i = 0; i < n; i++) {
        if (pos + 64 > obufCap) {
            if (fwrite(obuf, 1, pos, stdout) != pos)
                fail("write to stdout failed");
            pos = 0;
        }
        if (i > 0)
            obuf[pos++] = ' ';
        int w = snprintf(obuf + pos, 48, "%.10e", c[i]);
        if (w < 0 || w >= 48)
            fail("output formatting failed");
        pos += (size_t)w;
    }
    obuf[pos++] = '\n';
    if (fwrite(obuf, 1, pos, stdout) != pos)
        fail("write to stdout failed");

    free(obuf);
    free(a);
    free(b);
    free(c);
    return 0;
}
