#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#ifdef __CUDACC__
#define HD __device__
#else
#define HD
typedef struct { unsigned char x, y, z, w; } uchar4;
#endif

#define BLOCK 256
#define GRID_BLOCKS 224
#define MAX_NC 32
#define MAX_PIXELS 400000000LL
#define PATH_LEN 4096

#ifdef __CUDACC__
__constant__ double c_avg[MAX_NC * 3];
__constant__ double c_inv[MAX_NC * 9];
__constant__ double c_logdet[MAX_NC];
__constant__ int c_nc;
#else
static double c_avg[MAX_NC * 3];
static double c_inv[MAX_NC * 9];
static double c_logdet[MAX_NC];
static int c_nc;
#endif

HD int classifyPixel(uchar4 p)
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

#ifdef __CUDACC__
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
#endif

static void fail(const char *msg)
{
    fprintf(stderr, "ERROR: %s\n", msg);
    exit(0);
}

static void readPath(char *path)
{
    if (fgets(path, PATH_LEN, stdin) == NULL)
        fail("failed to read image path");
    size_t len = strlen(path);
    while (len > 0 && (path[len - 1] == '\n' || path[len - 1] == '\r'))
        path[--len] = '\0';
    if (len == 0)
        fail("empty image path");
}

static void writeLE32(FILE *f, long long v)
{
    unsigned char b[4];
    b[0] = (unsigned char)(v & 0xFF);
    b[1] = (unsigned char)((v >> 8) & 0xFF);
    b[2] = (unsigned char)((v >> 16) & 0xFF);
    b[3] = (unsigned char)((v >> 24) & 0xFF);
    if (fwrite(b, 1, 4, f) != 4)
        fail("failed to write image header");
}

int main(void)
{
    char srcPath[PATH_LEN];
    char dstPath[PATH_LEN];
    readPath(srcPath);
    readPath(dstPath);

    FILE *fin = fopen(srcPath, "rb");
    if (!fin)
        fail("cannot open source image");

    unsigned char hdr[8];
    if (fread(hdr, 1, 8, fin) != 8)
        fail("failed to read image header");
    long long w = (long long)hdr[0] | ((long long)hdr[1] << 8) |
                  ((long long)hdr[2] << 16) | ((long long)hdr[3] << 24);
    long long h = (long long)hdr[4] | ((long long)hdr[5] << 8) |
                  ((long long)hdr[6] << 16) | ((long long)hdr[7] << 24);

    if (w < 1 || h < 1)
        fail("invalid image size");
    if (w > MAX_PIXELS || h > MAX_PIXELS || w * h > MAX_PIXELS)
        fail("image is too large");

    size_t pxBytes = (size_t)(w * h) * 4;
    unsigned char *img = (unsigned char *)malloc(pxBytes);
    if (!img)
        fail("out of memory");
    if (fread(img, 1, pxBytes, fin) != pxBytes)
        fail("source image file is truncated");
    fclose(fin);

    int nc;
    if (fscanf(stdin, "%d", &nc) != 1)
        fail("failed to read number of classes");
    if (nc < 1 || nc > MAX_NC)
        fail("invalid number of classes");

    double h_avg[MAX_NC * 3];
    double h_inv[MAX_NC * 9];
    double h_logdet[MAX_NC];

    for (int j = 0; j < nc; j++) {
        int np;
        if (fscanf(stdin, "%d", &np) != 1)
            fail("failed to read sample size");
        if (np < 1)
            fail("invalid sample size");

        double sum[3] = {0.0, 0.0, 0.0};
        double s2[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (int i = 0; i < np; i++) {
            int x, y;
            if (fscanf(stdin, "%d %d", &x, &y) != 2)
                fail("failed to read sample pixel coordinates");
            if (x < 0 || (long long)x >= w || y < 0 || (long long)y >= h)
                fail("sample pixel coordinates out of range");
            const unsigned char *px = img + ((long long)y * w + (long long)x) * 4;
            double r = px[0];
            double g = px[1];
            double b = px[2];
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

        h_avg[j * 3 + 0] = a0;
        h_avg[j * 3 + 1] = a1;
        h_avg[j * 3 + 2] = a2;
        double *inv = h_inv + j * 9;
        inv[0] = a00 / det;
        inv[1] = a01 / det;
        inv[2] = a02 / det;
        inv[3] = a01 / det;
        inv[4] = a11 / det;
        inv[5] = a12 / det;
        inv[6] = a02 / det;
        inv[7] = a12 / det;
        inv[8] = a22 / det;
        h_logdet[j] = log(fabs(det));
    }

#ifdef __CUDACC__
    cudaError_t err;
    int devCount = 0;
    if ((err = cudaGetDeviceCount(&devCount)) != cudaSuccess || devCount < 1)
        fail("no CUDA device available");

    if ((err = cudaMemcpyToSymbol(c_nc, &nc, sizeof(int))) != cudaSuccess)
        fail("cudaMemcpyToSymbol failed");
    if ((err = cudaMemcpyToSymbol(c_avg, h_avg, sizeof(double) * (size_t)nc * 3)) != cudaSuccess)
        fail("cudaMemcpyToSymbol failed");
    if ((err = cudaMemcpyToSymbol(c_inv, h_inv, sizeof(double) * (size_t)nc * 9)) != cudaSuccess)
        fail("cudaMemcpyToSymbol failed");
    if ((err = cudaMemcpyToSymbol(c_logdet, h_logdet, sizeof(double) * (size_t)nc)) != cudaSuccess)
        fail("cudaMemcpyToSymbol failed");

    uchar4 *dImg = NULL;
    if ((err = cudaMalloc((void **)&dImg, pxBytes)) != cudaSuccess)
        fail("cudaMalloc failed");
    if ((err = cudaMemcpy(dImg, img, pxBytes, cudaMemcpyHostToDevice)) != cudaSuccess)
        fail("cudaMemcpy host to device failed");

    classifyKernel<<<GRID_BLOCKS, BLOCK>>>(dImg, w * h);
    if ((err = cudaGetLastError()) != cudaSuccess)
        fail("kernel launch failed");
    if ((err = cudaDeviceSynchronize()) != cudaSuccess)
        fail("cudaDeviceSynchronize failed");
    if ((err = cudaMemcpy(img, dImg, pxBytes, cudaMemcpyDeviceToHost)) != cudaSuccess)
        fail("cudaMemcpy device to host failed");
    cudaFree(dImg);
#else
    c_nc = nc;
    memcpy(c_avg, h_avg, sizeof(double) * (size_t)nc * 3);
    memcpy(c_inv, h_inv, sizeof(double) * (size_t)nc * 9);
    memcpy(c_logdet, h_logdet, sizeof(double) * (size_t)nc);
    long long total = w * h;
    for (long long i = 0; i < total; i++) {
        uchar4 p;
        p.x = img[i * 4 + 0];
        p.y = img[i * 4 + 1];
        p.z = img[i * 4 + 2];
        p.w = img[i * 4 + 3];
        p.w = (unsigned char)classifyPixel(p);
        img[i * 4 + 3] = p.w;
    }
#endif

    FILE *fout = fopen(dstPath, "wb");
    if (!fout)
        fail("cannot open destination image");
    writeLE32(fout, w);
    writeLE32(fout, h);
    if (fwrite(img, 1, pxBytes, fout) != pxBytes)
        fail("failed to write image data");
    fclose(fout);

    free(img);
    return 0;
}
