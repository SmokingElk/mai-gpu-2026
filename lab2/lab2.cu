#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define BLOCK_X 16
#define BLOCK_Y 16
#define GRID_X 32
#define GRID_Y 32
#define MAX_DIM 65536LL
#define MAX_PIXELS 100000000LL
#define PATH_LEN 4096

__global__ void robertsKernel(cudaTextureObject_t tex, uchar4 *dst, int w, int h)
{
    int strideX = gridDim.x * blockDim.x;
    int strideY = gridDim.y * blockDim.y;
    for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < h; y += strideY) {
        for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < w; x += strideX) {
            uchar4 p00 = tex2D<uchar4>(tex, (float)x, (float)y);
            uchar4 p11 = tex2D<uchar4>(tex, (float)(x + 1), (float)(y + 1));
            uchar4 p10 = tex2D<uchar4>(tex, (float)x, (float)(y + 1));
            uchar4 p01 = tex2D<uchar4>(tex, (float)(x + 1), (float)y);
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

    if (w < 1 || h < 1 || w >= MAX_DIM || h >= MAX_DIM)
        fail("invalid image size");
    if (w * h > MAX_PIXELS)
        fail("image is too large");

    size_t pxBytes = (size_t)(w * h) * 4;
    unsigned char *hSrc = (unsigned char *)malloc(pxBytes);
    if (!hSrc)
        fail("out of memory");
    if (fread(hSrc, 1, pxBytes, fin) != pxBytes)
        fail("source image file is truncated");
    fclose(fin);

    unsigned char *hDst = (unsigned char *)malloc(pxBytes);
    if (!hDst)
        fail("out of memory");

    cudaError_t err;
    int devCount = 0;
    if ((err = cudaGetDeviceCount(&devCount)) != cudaSuccess || devCount < 1)
        fail("no CUDA device available");

    cudaArray_t srcArr = NULL;
    uchar4 *dDst = NULL;
    if ((err = cudaMalloc((void **)&dDst, pxBytes)) != cudaSuccess)
        fail("cudaMalloc failed");

    cudaChannelFormatDesc chDesc = cudaCreateChannelDesc<uchar4>();
    if ((err = cudaMallocArray(&srcArr, &chDesc, (size_t)w, (size_t)h)) != cudaSuccess)
        fail("cudaMallocArray failed");
    if ((err = cudaMemcpy2DToArray(srcArr, 0, 0, hSrc, (size_t)w * 4, (size_t)w * 4,
                                   (size_t)h, cudaMemcpyHostToDevice)) != cudaSuccess)
        fail("cudaMemcpy2DToArray failed");

    cudaResourceDesc res;
    memset(&res, 0, sizeof(res));
    res.resType = cudaResourceTypeArray;
    res.res.array.array = srcArr;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t tex = 0;
    if ((err = cudaCreateTextureObject(&tex, &res, &texDesc, NULL)) != cudaSuccess)
        fail("cudaCreateTextureObject failed");

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid(GRID_X, GRID_Y);
    robertsKernel<<<grid, block>>>(tex, dDst, (int)w, (int)h);
    if ((err = cudaGetLastError()) != cudaSuccess)
        fail("kernel launch failed");
    if ((err = cudaDeviceSynchronize()) != cudaSuccess)
        fail("cudaDeviceSynchronize failed");
    if ((err = cudaMemcpy(hDst, dDst, pxBytes, cudaMemcpyDeviceToHost)) != cudaSuccess)
        fail("cudaMemcpy device to host failed");

    cudaDestroyTextureObject(tex);
    cudaFreeArray(srcArr);
    cudaFree(dDst);

    FILE *fout = fopen(dstPath, "wb");
    if (!fout)
        fail("cannot open destination image");
    writeLE32(fout, w);
    writeLE32(fout, h);
    if (fwrite(hDst, 1, pxBytes, fout) != pxBytes)
        fail("failed to write image data");
    fclose(fout);

    free(hSrc);
    free(hDst);
    return 0;
}
