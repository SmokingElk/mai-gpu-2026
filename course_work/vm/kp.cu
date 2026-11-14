/*
 * Курсовая работа (вариант 9).
 * Обратная трассировка лучей (Ray Tracing) на GPU с использованием CUDA
 * и сравнение производительности GPU и CPU.
 *
 * Сцена: текстурированный пол (квадратный, параллельный плоскости XY —
 * упрощённое наложение текстур), над которым расположены три платоновых
 * тела (вариант 9: гексаэдр, додекаэдр, икосаэдр). Грани тел обладают
 * зеркальным и прозрачным эффектом; на рёбрах тел располагаются точечные
 * источники света. Отражения обрабатываются итеративно (без рекурсии) до
 * заданной глубины; прозрачность упрощённая: преломлённый луч далее не
 * отражается (кроме полного внутреннего отражения внутри тела, за счёт
 * многократных переотражений внутри тела возникает эффект бесконечности).
 * Сглаживание — SSAA (n*n лучей на пиксель).
 *
 * Ключи запуска:
 *   --gpu        расчёт на видеокарте (по умолчанию, можно без ключей)
 *   --cpu        расчёт только на центральном процессоре (без CUDA)
 *   --default    напечатать в stdout конфигурацию входных данных, при
 *                которой получается наиболее красочный результат, и
 *                завершить работу
 *   --wireframe  каркасный рендер сцены (рёбра тел, рамка и сетка пола,
 *                источники света) — для материалов отчёта
 *
 * Ввод (stdin), формат по условию:
 *   количество кадров
 *   путь к выходным изображениям со спецификатором %d
 *   ширина высота угол_обзора_в_градусах_по_горизонтали
 *   параметры камеры (10): r0 z0 phi0 Ar Az om_r om_z om_phi pr pz
 *   параметры точки наблюдения (10): те же
 *   параметры тел (3 строки по 10 чисел): центр(3) цвет(3) радиус
 *     коэффициент_отражения коэффициент_прозрачности источников_на_ребре;
 *     тело 1 — гексаэдр, тело 2 — додекаэдр, тело 3 — икосаэдр
 *   параметры пола: 4 точки (12 чисел), путь к текстуре, оттенок(3),
 *     коэффициент отражения
 *   количество источников света (не более 4), далее по строке на источник:
 *     положение(3) цвет(3)
 *   максимальная глубина трассировки, квадратный корень из количества
 *     лучей на пиксель (SSAA)
 *
 * Камера и точка наблюдения движутся в цилиндрических координатах:
 *   r(t)   = r0   + Ar   * sin(om_r * t + pr)
 *   z(t)   = z0   + Az   * sin(om_z * t + pz)
 *   phi(t) = phi0 + om_phi * t
 * Кадр i соответствует t = 2*pi*i/N (кадры нумеруются с нуля, анимация
 * зацикливается: последний кадр стыкуется с первым).
 *
 * Вывод: кадры в файлах (формат ЛР2: 8 байт — ширина и высота
 * little-endian, затем пиксели RGBA); в stdout построчно
 *   {номер кадра}\t{время на обработку кадра в мс}\t{общее количество лучей}
 * Отладочная информация — в stderr. При любой ошибке печатается "ERROR: ..."
 * в stderr и программа завершается с кодом возврата 0.
 *
 * Компиляция (чекер):
 *   nvcc -Werror cross-execution-space-call -lm
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdarg>
#include <ctime>
#include <algorithm>
#include <thread>
#include <vector>
#include <cuda_runtime.h>

/* ---------- ограничения и конфигурации ---------- */

#define MAX_BODIES      3
#define MAX_PLANES      38      /* 6 (гексаэдр) + 12 (додекаэдр) + 20 (икосаэдр) */
#define MAX_WIRE        128     /* 72 ребра + рамка и сетка пола + источники */
#define MAX_MAIN_LIGHTS 4
#define MAX_EDGE_LIGHTS 256     /* точечных источников на одном ребре */
#define MAX_LIGHTS      (MAX_MAIN_LIGHTS + 72 * MAX_EDGE_LIGHTS)

#define MAX_FRAMES 100000LL
#define MAX_DIM    16384
#define MAX_PIXELS 100000000LL
#define PATH_LEN   4096
#define MAX_DEPTH  64
#define MAX_SSAA   64
#define TEX_MAX_DIM 16384

/* конфигурации нитей — фиксированные, не зависят от входных данных */
#define BLOCK_X 16
#define BLOCK_Y 16
#define GRID_X  32
#define GRID_Y  32

static const double kPi = 3.14159265358979323846;
static const float kIor = 1.5f;         /* коэффициент преломления тел */
static const float kRayEps = 1e-3f;     /* сдвиг начала луча от поверхности */
static const float kMinWeight = 4e-3f;  /* порог обрыва трассировки по весу */
static const int kMainBodyTag = 255;    /* пометка основных источников света */
static const float kEdgeGain = 0.10f;   /* яркость одного реберного источника
                                         * (их сотни — без этого суммарный
                                         * свет сильно пересвечивает сцену) */
static const float kExposure = 1.2f;    /* экспозиция перед тон-маппингом */

/* ---------- векторная арифметика (общая для host и device) ---------- */

struct vec3 {
    float x, y, z;
};

static __host__ __device__ inline vec3 v3(float x, float y, float z)
{
    vec3 r;
    r.x = x;
    r.y = y;
    r.z = z;
    return r;
}

static __host__ __device__ inline vec3 operator+(vec3 a, vec3 b)
{
    return v3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static __host__ __device__ inline vec3 operator-(vec3 a, vec3 b)
{
    return v3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static __host__ __device__ inline vec3 operator-(vec3 a)
{
    return v3(-a.x, -a.y, -a.z);
}

static __host__ __device__ inline vec3 operator*(vec3 a, vec3 b)
{
    return v3(a.x * b.x, a.y * b.y, a.z * b.z);
}

static __host__ __device__ inline vec3 operator*(vec3 a, float k)
{
    return v3(a.x * k, a.y * k, a.z * k);
}

static __host__ __device__ inline float dot(vec3 a, vec3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static __host__ __device__ inline vec3 cross(vec3 a, vec3 b)
{
    return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
              a.x * b.y - a.y * b.x);
}

static __host__ __device__ inline float vlen(vec3 v)
{
    return sqrtf(dot(v, v));
}

static __host__ __device__ inline vec3 vnorm(vec3 v)
{
    return v * (1.0f / sqrtf(dot(v, v)));
}

static __host__ __device__ inline vec3 reflectDir(vec3 d, vec3 n)
{
    return d - n * (2.0f * dot(d, n));
}

/* преломление по Снеллиусу; n направлена навстречу d, eta = n1/n2;
 * возвращает false при полном внутреннем отражении */
static __host__ __device__ inline bool refractDir(vec3 d, vec3 n, float eta,
                                                  vec3 *out)
{
    float ci = -dot(d, n);
    float s2 = eta * eta * (1.0f - ci * ci);
    if (s2 >= 1.0f)
        return false;
    float ct = sqrtf(1.0f - s2);
    *out = vnorm(d * eta + n * (eta * ci - ct));
    return true;
}

static __host__ __device__ inline unsigned char toByte(float v)
{
    int b = (int)(v * 255.0f + 0.5f);
    if (b < 0)
        b = 0;
    if (b > 255)
        b = 255;
    return (unsigned char)b;
}

/* ---------- структура сцены ---------- */

struct Light {
    vec3 pos;
    vec3 color;
    int body;       /* номер тела, на ребре которого стоит источник; kMainBodyTag — основной */
};

struct WireSeg {    /* отрезок для каркасного рендера (вырожденный — точка) */
    vec3 a, b;
    vec3 color;
};

struct Plane {      /* dot(n, p) = d, n наружу от тела */
    vec3 n;
    float d;
};

struct BodyG {
    vec3 center;
    vec3 color;
    float radius;
    float refl;
    float transp;
    int planeOff;
    int planeCnt;
};

struct Scene {
    BodyG bodies[MAX_BODIES];
    Plane planes[MAX_PLANES];
    float fx0, fx1, fy0, fy1, fz;   /* пол: квадрат в плоскости z = fz */
    float fRefl;
    vec3 fTint;
    int nLights;
    const Light *lights;
    int texW, texH;
    const unsigned char *tex;       /* RGBA */
    int maxDepth;
    int nWire;
    const WireSeg *wire;
};

struct RenderParams {
    vec3 pos, fwd, right, up;
    float tanF;     /* tan(fov/2), fov — по горизонтали */
    float aspect;   /* W/H */
    int W, H;
    int ssaa;
    int wire;
};

struct Hit {
    float t;
    vec3 n;         /* наружная нормаль */
    int obj;        /* 0..2 — тело, 3 — пол, -1 — нет пересечения */
    bool fromInside;
};

/* ---------- обработка ошибок ---------- */

static void failf(const char *fmt, ...)
{
    va_list ap;
    fputs("ERROR: ", stderr);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    exit(0);
}

static void fail(const char *msg)
{
    failf("%s", msg);
}

static void checkCuda(cudaError_t err, const char *what)
{
    if (err != cudaSuccess)
        failf("%s: %s", what, cudaGetErrorString(err));
}

static double nowMs(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

/* ---------- разбор входных данных ---------- */

struct CylPath {     /* движение точки в цилиндрических координатах */
    double r0, z0, p0, ar, az, wr, wz, wp, pr, pz;
};

struct Config {
    long long frames;
    char outPath[PATH_LEN];
    int W, H;
    double fov;
    CylPath cam, tgt;
    struct {
        double c[3], col[3], R, refl, transp;
        long long edgeL;
    } body[MAX_BODIES];
    double fp[4][3];
    char texPath[PATH_LEN];
    double tint[3], fRefl;
    int nLights;
    double lp[MAX_MAIN_LIGHTS][3], lc[MAX_MAIN_LIGHTS][3];
    int maxDepth, ssaa;
};

static void readToken(char *buf)
{
    if (fscanf(stdin, "%4095s", buf) != 1)
        fail("unexpected end of input");
}

static double readNum(void)
{
    char buf[PATH_LEN];
    char *end;
    double v;
    readToken(buf);
    v = strtod(buf, &end);
    if (end == buf || *end != '\0' || !isfinite(v))
        failf("invalid number '%s'", buf);
    return v;
}

static long long readLL(void)
{
    double v = readNum();
    if (v != floor(v))
        failf("expected integer, got %g", v);
    return (long long)v;
}

static int readIntInRange(long long lo, long long hi, const char *what)
{
    long long v = readLL();
    if (v < lo || v > hi)
        failf("%s must be between %lld and %lld", what, lo, hi);
    return (int)v;
}

static void readPath(char *dst)
{
    readToken(dst);
}

static void expandTilde(const char *src, char *dst)
{
    if (src[0] == '~' && (src[1] == '/' || src[1] == '\0')) {
        const char *home = getenv("HOME");
        if (home == NULL)
            fail("cannot expand '~': HOME is not set");
        if (snprintf(dst, PATH_LEN, "%s%s", home, src + 1) >= PATH_LEN)
            fail("path is too long");
    } else {
        if (snprintf(dst, PATH_LEN, "%s", src) >= PATH_LEN)
            fail("path is too long");
    }
}

static void readCylPath(CylPath *p)
{
    double *v = &p->r0;
    int i;
    for (i = 0; i < 10; i++)
        v[i] = readNum();
}

static void validateOutPath(const char *p)
{
    int n = 0;
    const char *q;
    for (q = p; *q != '\0'; q++) {
        if (*q != '%')
            continue;
        if (q[1] != 'd')
            failf("output path may contain only the %%d specifier: %s", p);
        n++;
        q++;
    }
    if (n != 1)
        fail("output path must contain exactly one %d specifier");
}

static void parseInput(Config *cfg)
{
    int i, j;
    long long w, h;

    cfg->frames = readLL();
    if (cfg->frames < 1 || cfg->frames > MAX_FRAMES)
        failf("frame count must be between 1 and %lld", MAX_FRAMES);

    readPath(cfg->outPath);
    validateOutPath(cfg->outPath);

    w = readLL();
    h = readLL();
    if (w < 1 || w > MAX_DIM || h < 1 || h > MAX_DIM)
        failf("image size must be between 1 and %d", MAX_DIM);
    if (w * h > MAX_PIXELS)
        fail("image is too large");
    cfg->W = (int)w;
    cfg->H = (int)h;

    cfg->fov = readNum();
    if (cfg->fov <= 0.0 || cfg->fov >= 180.0)
        fail("field of view must be in (0, 180) degrees");

    readCylPath(&cfg->cam);
    readCylPath(&cfg->tgt);

    for (i = 0; i < MAX_BODIES; i++) {
        for (j = 0; j < 3; j++)
            cfg->body[i].c[j] = readNum();
        for (j = 0; j < 3; j++)
            cfg->body[i].col[j] = readNum();
        cfg->body[i].R = readNum();
        cfg->body[i].refl = readNum();
        cfg->body[i].transp = readNum();
        cfg->body[i].edgeL = readLL();
        if (cfg->body[i].R <= 0.0)
            fail("body radius must be positive");
        if (cfg->body[i].refl < 0.0 || cfg->body[i].refl > 1.0 ||
            cfg->body[i].transp < 0.0 || cfg->body[i].transp > 1.0)
            fail("reflection and transparency must be in [0, 1]");
        if (cfg->body[i].edgeL < 0 || cfg->body[i].edgeL > MAX_EDGE_LIGHTS)
            failf("edge light count must be between 0 and %d", MAX_EDGE_LIGHTS);
    }

    for (i = 0; i < 4; i++)
        for (j = 0; j < 3; j++)
            cfg->fp[i][j] = readNum();
    readPath(cfg->texPath);
    for (j = 0; j < 3; j++)
        cfg->tint[j] = readNum();
    cfg->fRefl = readNum();
    if (cfg->fRefl < 0.0 || cfg->fRefl > 1.0)
        fail("floor reflection must be in [0, 1]");

    cfg->nLights = readIntInRange(0, MAX_MAIN_LIGHTS, "light count");
    for (i = 0; i < cfg->nLights; i++) {
        for (j = 0; j < 3; j++)
            cfg->lp[i][j] = readNum();
        for (j = 0; j < 3; j++)
            cfg->lc[i][j] = readNum();
    }

    cfg->maxDepth = readIntInRange(0, MAX_DEPTH, "max depth");
    cfg->ssaa = readIntInRange(1, MAX_SSAA, "ssaa");
}

/* ---------- загрузка текстуры пола (формат ЛР2) ---------- */

static void loadTexture(const char *path, int *w, int *h,
                        std::vector<unsigned char> *data)
{
    char full[PATH_LEN];
    unsigned char hdr[8];
    long long tw, th;
    size_t n;
    FILE *f;

    expandTilde(path, full);
    f = fopen(full, "rb");
    if (f == NULL)
        failf("cannot open texture file %s", full);
    if (fread(hdr, 1, 8, f) != 8)
        failf("failed to read texture header %s", full);
    tw = (long long)hdr[0] | ((long long)hdr[1] << 8) |
         ((long long)hdr[2] << 16) | ((long long)hdr[3] << 24);
    th = (long long)hdr[4] | ((long long)hdr[5] << 8) |
         ((long long)hdr[6] << 16) | ((long long)hdr[7] << 24);
    if (tw < 1 || th < 1 || tw > TEX_MAX_DIM || th > TEX_MAX_DIM)
        failf("invalid texture size %lldx%lld", tw, th);
    n = (size_t)tw * (size_t)th * 4;
    data->resize(n);
    if (fread(data->data(), 1, n, f) != n)
        failf("texture file %s is truncated", full);
    fclose(f);
    *w = (int)tw;
    *h = (int)th;
}

/* ---------- генерация платоновых тел ---------- */

#define KIND_HEXA     0
#define KIND_OCTA     1
#define KIND_DODECA   2
#define KIND_ICOSA    3
/* двойственные тела для построения плоскостей граней: ориентация вершин
 * двойника должна точно соответствовать телу (общая группа симметрий
 * недостаточна — у икосаэдра/додекаэдра есть два вложенных положения
 * двойника, и "не тот" даёт плоскости, касающиеся вершин, а не граней) */
#define KIND_ICOSA_D  4
#define KIND_DODECA_D 5

static const int kBodyKind[MAX_BODIES] = { KIND_HEXA, KIND_DODECA, KIND_ICOSA };
static const int kDualKind[MAX_BODIES] = { KIND_OCTA, KIND_ICOSA_D, KIND_DODECA_D };

static void addVert(double v[][3], int *n, double x, double y, double z)
{
    double l = sqrt(x * x + y * y + z * z);
    v[*n][0] = x / l;
    v[*n][1] = y / l;
    v[*n][2] = z / l;
    (*n)++;
}

/* вершины платонова тела на единичной описанной сфере */
static int platonicVerts(int kind, double v[][3])
{
    static const double kPhi = 1.6180339887498949;
    int n = 0;
    int i;
    double a, b, ip = 1.0 / kPhi;

    switch (kind) {
    case KIND_HEXA:
        for (i = 0; i < 8; i++)
            addVert(v, &n, (i & 1) ? 1.0 : -1.0, (i & 2) ? 1.0 : -1.0,
                    (i & 4) ? 1.0 : -1.0);
        break;
    case KIND_OCTA:
        addVert(v, &n, 1, 0, 0);
        addVert(v, &n, -1, 0, 0);
        addVert(v, &n, 0, 1, 0);
        addVert(v, &n, 0, -1, 0);
        addVert(v, &n, 0, 0, 1);
        addVert(v, &n, 0, 0, -1);
        break;
    case KIND_DODECA:
        for (i = 0; i < 8; i++)
            addVert(v, &n, (i & 1) ? 1.0 : -1.0, (i & 2) ? 1.0 : -1.0,
                    (i & 4) ? 1.0 : -1.0);
        for (i = 0; i < 4; i++) {
            a = (i & 1) ? 1.0 : -1.0;
            b = (i & 2) ? 1.0 : -1.0;
            addVert(v, &n, 0.0, a * ip, b * kPhi);
            addVert(v, &n, a * ip, b * kPhi, 0.0);
            addVert(v, &n, a * kPhi, 0.0, b * ip);
        }
        break;
    case KIND_ICOSA_D:
        /* двойник додекаэдра: направления на центры его граней */
        for (i = 0; i < 4; i++) {
            a = (i & 1) ? 1.0 : -1.0;
            b = (i & 2) ? 1.0 : -1.0;
            addVert(v, &n, 0.0, a * kPhi, b);
            addVert(v, &n, a * kPhi, b, 0.0);
            addVert(v, &n, a, 0.0, b * kPhi);
        }
        break;
    case KIND_DODECA_D:
        /* двойник икосаэдра: направления на центры его граней */
        for (i = 0; i < 8; i++)
            addVert(v, &n, (i & 1) ? 1.0 : -1.0, (i & 2) ? 1.0 : -1.0,
                    (i & 4) ? 1.0 : -1.0);
        for (i = 0; i < 4; i++) {
            a = (i & 1) ? 1.0 : -1.0;
            b = (i & 2) ? 1.0 : -1.0;
            addVert(v, &n, 0.0, a * kPhi, b * ip);
            addVert(v, &n, a * kPhi, b * ip, 0.0);
            addVert(v, &n, a * ip, 0.0, b * kPhi);
        }
        break;
    default: /* KIND_ICOSA */
        for (i = 0; i < 4; i++) {
            a = (i & 1) ? 1.0 : -1.0;
            b = (i & 2) ? 1.0 : -1.0;
            addVert(v, &n, 0.0, a, b * kPhi);
            addVert(v, &n, a, b * kPhi, 0.0);
            addVert(v, &n, a * kPhi, 0.0, b);
        }
        break;
    }
    return n;
}

static double ddot3(const double a[3], const double b[3])
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static double ddist2(const double a[3], const double b[3])
{
    double dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz;
}

/* ---------- сборка сцены на хосте ---------- */

static void buildScene(const Config *cfg, Scene *s, std::vector<Light> *lights,
                       std::vector<WireSeg> *wire)
{
    int i, j, k, t, a, b;
    int planeOff = 0;
    double verts[20][3], duals[20][3];
    int nv, nd;

    memset(s, 0, sizeof(*s));
    s->maxDepth = cfg->maxDepth;

    /* пол: квадрат, параллельный плоскости XY (упрощение) */
    s->fx0 = s->fx1 = cfg->fp[0][0];
    s->fy0 = s->fy1 = cfg->fp[0][1];
    for (i = 1; i < 4; i++) {
        if (cfg->fp[i][0] < s->fx0) s->fx0 = cfg->fp[i][0];
        if (cfg->fp[i][0] > s->fx1) s->fx1 = cfg->fp[i][0];
        if (cfg->fp[i][1] < s->fy0) s->fy0 = cfg->fp[i][1];
        if (cfg->fp[i][1] > s->fy1) s->fy1 = cfg->fp[i][1];
    }
    s->fz = (float)cfg->fp[0][2];
    s->fRefl = (float)cfg->fRefl;
    s->fTint = v3((float)cfg->tint[0], (float)cfg->tint[1], (float)cfg->tint[2]);
    if (s->fx1 <= s->fx0 || s->fy1 <= s->fy0)
        fail("floor points must span a non-degenerate rectangle");

    for (i = 0; i < MAX_BODIES; i++) {
        BodyG *B = &s->bodies[i];
        double best2;
        B->center = v3((float)cfg->body[i].c[0], (float)cfg->body[i].c[1],
                       (float)cfg->body[i].c[2]);
        B->color = v3((float)cfg->body[i].col[0], (float)cfg->body[i].col[1],
                      (float)cfg->body[i].col[2]);
        B->radius = (float)cfg->body[i].R;
        B->refl = (float)cfg->body[i].refl;
        B->transp = (float)cfg->body[i].transp;

        /* грани: направление на центр грани — вершина двойственного тела;
         * плоскость грани проходит через ближайшие к направлению вершины */
        nv = platonicVerts(kBodyKind[i], verts);
        nd = platonicVerts(kDualKind[i], duals);
        B->planeOff = planeOff;
        B->planeCnt = nd;
        if (planeOff + nd > MAX_PLANES)
            fail("internal error: too many planes");
        for (j = 0; j < nd; j++) {
            double dUnit = -1e30;
            double dw;
            for (t = 0; t < nv; t++) {
                double dd = ddot3(duals[j], verts[t]);
                if (dd > dUnit)
                    dUnit = dd;
            }
            dw = cfg->body[i].R * dUnit + ddot3(duals[j], cfg->body[i].c);
            s->planes[planeOff + j].n =
                v3((float)duals[j][0], (float)duals[j][1], (float)duals[j][2]);
            s->planes[planeOff + j].d = (float)dw;
        }
        planeOff += nd;

        /* рёбра: пары ближайших вершин правильного тела */
        best2 = 1e30;
        for (a = 0; a < nv; a++)
            for (b = a + 1; b < nv; b++) {
                double d2 = ddist2(verts[a], verts[b]);
                if (d2 < best2)
                    best2 = d2;
            }
        for (a = 0; a < nv; a++)
            for (b = a + 1; b < nv; b++) {
                if (ddist2(verts[a], verts[b]) > best2 * 1.05)
                    continue;
                vec3 pa = B->center + v3((float)(cfg->body[i].R * verts[a][0]),
                                         (float)(cfg->body[i].R * verts[a][1]),
                                         (float)(cfg->body[i].R * verts[a][2]));
                vec3 pb = B->center + v3((float)(cfg->body[i].R * verts[b][0]),
                                         (float)(cfg->body[i].R * verts[b][1]),
                                         (float)(cfg->body[i].R * verts[b][2]));
                /* отрезок для каркасного рендера */
                WireSeg w;
                w.a = pa;
                w.b = pb;
                w.color = B->color * 1.7f;
                wire->push_back(w);
                /* точечные источники на ребре (включая концы) */
                for (k = 0; k < (int)cfg->body[i].edgeL; k++) {
                    float tt = (cfg->body[i].edgeL == 1)
                                   ? 0.5f
                                   : (float)k / (float)(cfg->body[i].edgeL - 1);
                    Light L;
                    L.pos = pa + (pb - pa) * tt;
                    L.color = B->color * kEdgeGain;
                    L.body = i;
                    lights->push_back(L);
                }
            }
    }

    /* каркас пола: рамка и сетка */
    {
        vec3 cB = s->fTint * 0.9f, cG = s->fTint * 0.45f;
        vec3 c0 = v3(s->fx0, s->fy0, s->fz), c1 = v3(s->fx1, s->fy0, s->fz);
        vec3 c2 = v3(s->fx1, s->fy1, s->fz), c3 = v3(s->fx0, s->fy1, s->fz);
        WireSeg w;
        w.color = cB;
        w.a = c0; w.b = c1; wire->push_back(w);
        w.a = c1; w.b = c2; wire->push_back(w);
        w.a = c2; w.b = c3; wire->push_back(w);
        w.a = c3; w.b = c0; wire->push_back(w);
        w.color = cG;
        for (i = 1; i <= 7; i++) {
            float tt = (float)i / 8.0f;
            w.a = v3(s->fx0 + (s->fx1 - s->fx0) * tt, s->fy0, s->fz);
            w.b = v3(s->fx0 + (s->fx1 - s->fx0) * tt, s->fy1, s->fz);
            wire->push_back(w);
            w.a = v3(s->fx0, s->fy0 + (s->fy1 - s->fy0) * tt, s->fz);
            w.b = v3(s->fx1, s->fy0 + (s->fy1 - s->fy0) * tt, s->fz);
            wire->push_back(w);
        }
    }

    /* основные источники света (и как точки для каркасного рендера) */
    for (i = 0; i < cfg->nLights; i++) {
        Light L;
        L.pos = v3((float)cfg->lp[i][0], (float)cfg->lp[i][1],
                   (float)cfg->lp[i][2]);
        L.color = v3((float)cfg->lc[i][0], (float)cfg->lc[i][1],
                     (float)cfg->lc[i][2]);
        L.body = kMainBodyTag;
        lights->push_back(L);
        WireSeg w;
        w.a = L.pos;
        w.b = L.pos;
        w.color = L.color * 1.5f;
        wire->push_back(w);
    }

    if ((int)lights->size() > MAX_LIGHTS)
        fail("too many lights");
    if ((int)wire->size() > MAX_WIRE)
        fail("internal error: too many wire segments");
    s->nLights = (int)lights->size();
    s->lights = lights->data();
    s->nWire = (int)wire->size();
    s->wire = wire->data();
}

/* ---------- ядро трассировки (общее для host и device) ---------- */

static __host__ __device__ vec3 skyColor(vec3 d)
{
    float h = 1.0f - fabsf(d.z);
    return v3(0.02f, 0.03f, 0.06f) + v3(0.06f, 0.07f, 0.11f) * (h * h);
}

static __host__ __device__ vec3 texSample(const Scene *s, float u, float v)
{
    int x0, y0, x1, y1;
    float ax, ay;
    vec3 p00, p10, p01, p11;
    const unsigned char *q;

    if (u < 0.0f) u = 0.0f;
    if (u > 1.0f) u = 1.0f;
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    float fx = u * (float)(s->texW - 1);
    float fy = v * (float)(s->texH - 1);
    x0 = (int)fx;
    y0 = (int)fy;
    x1 = (x0 + 1 < s->texW) ? x0 + 1 : x0;
    y1 = (y0 + 1 < s->texH) ? y0 + 1 : y0;
    ax = fx - (float)x0;
    ay = fy - (float)y0;

    q = s->tex + 4 * (y0 * s->texW + x0);
    p00 = v3(q[0] / 255.0f, q[1] / 255.0f, q[2] / 255.0f);
    q = s->tex + 4 * (y0 * s->texW + x1);
    p10 = v3(q[0] / 255.0f, q[1] / 255.0f, q[2] / 255.0f);
    q = s->tex + 4 * (y1 * s->texW + x0);
    p01 = v3(q[0] / 255.0f, q[1] / 255.0f, q[2] / 255.0f);
    q = s->tex + 4 * (y1 * s->texW + x1);
    p11 = v3(q[0] / 255.0f, q[1] / 255.0f, q[2] / 255.0f);

    return (p00 + (p10 - p00) * ax) + ((p01 + (p11 - p01) * ax) -
                                       (p00 + (p10 - p00) * ax)) * ay;
}

static __host__ __device__ vec3 texSampleFloor(const Scene *s, vec3 p)
{
    float u = (p.x - s->fx0) / (s->fx1 - s->fx0);
    float v = (p.y - s->fy0) / (s->fy1 - s->fy0);
    return texSample(s, u, v);
}

/* пересечение луча с выпуклым многогранником (клиппинг по плоскостям
 * граней); возвращает попадание снаружи (fromInside=false) или выход
 * изнутри (fromInside=true) */
static __host__ __device__ bool bodyHit(const Scene *s, int bi, vec3 o, vec3 d,
                                         float *tOut, vec3 *nOut,
                                         bool *fromInside)
{
    const BodyG *B = &s->bodies[bi];
    vec3 oc = o - B->center;
    float bb = dot(oc, d);
    float cc = dot(oc, oc) - B->radius * B->radius;
    float tN = -1e30f, tF = 1e30f;
    int pN = -1, pF = -1;
    int i;

    /* быстрый тест описывающей сферой */
    if (cc > 0.0f && bb > 0.0f)
        return false;
    if (bb * bb - cc < 0.0f)
        return false;

    for (i = B->planeOff; i < B->planeOff + B->planeCnt; i++) {
        float dn = dot(d, s->planes[i].n);
        float dp = dot(o, s->planes[i].n) - s->planes[i].d;
        if (fabsf(dn) < 1e-8f) {
            if (dp > 0.0f)
                return false;   /* луч параллелен грани и снаружи */
            continue;
        }
        float t = -dp / dn;
        if (dn < 0.0f) {
            if (t > tN) { tN = t; pN = i; }
        } else {
            if (t < tF) { tF = t; pF = i; }
        }
        if (tN > tF)
            return false;
    }
    if (tN > tF)
        return false;
    if (tN > kRayEps && pN >= 0) {
        *tOut = tN;
        *nOut = s->planes[pN].n;
        *fromInside = false;
        return true;
    }
    if (tF > kRayEps && pF >= 0) {
        *tOut = tF;
        *nOut = s->planes[pF].n;
        *fromInside = true;
        return true;
    }
    return false;
}

static __host__ __device__ Hit sceneHit(const Scene *s, vec3 o, vec3 d)
{
    Hit h;
    int i;

    h.t = 1e30f;
    h.obj = -1;
    h.n = v3(0.0f, 0.0f, 0.0f);
    h.fromInside = false;

    for (i = 0; i < MAX_BODIES; i++) {
        float t;
        vec3 n;
        bool inside;
        if (bodyHit(s, i, o, d, &t, &n, &inside) && t < h.t) {
            h.t = t;
            h.n = n;
            h.obj = i;
            h.fromInside = inside;
        }
    }

    if (d.z > 1e-9f || d.z < -1e-9f) {
        float t = (s->fz - o.z) / d.z;
        if (t > kRayEps && t < h.t) {
            float x = o.x + d.x * t;
            float y = o.y + d.y * t;
            if (x >= s->fx0 && x <= s->fx1 && y >= s->fy0 && y <= s->fy1) {
                h.t = t;
                h.n = v3(0.0f, 0.0f, 1.0f);
                h.obj = 3;
                h.fromInside = false;
            }
        }
    }
    return h;
}

/* виден ли источник света из точки p (ldir и dist задают сегмент до
 * источника); тела, кроме skipBody, могут затенять; пол не затеняет,
 * так как он ниже всех объектов */
static __host__ __device__ bool shadowVisible(const Scene *s, vec3 p,
                                               vec3 ldir, float dist,
                                               int skipBody)
{
    int i, j;

    for (i = 0; i < MAX_BODIES; i++) {
        const BodyG *B;
        vec3 oc, cp;
        float proj, tN, tF;
        bool outside = false;

        if (i == skipBody)
            continue;
        B = &s->bodies[i];

        /* быстрый тест описывающей сферой на сегменте */
        oc = p - B->center;
        proj = dot(oc, ldir);
        if (proj < 0.0f)
            proj = 0.0f;
        if (proj > dist)
            proj = dist;
        cp = oc + ldir * proj;
        if (dot(cp, cp) > B->radius * B->radius)
            continue;

        /* клиппинг сегмента по плоскостям граней */
        tN = -1e30f;
        tF = 1e30f;
        for (j = B->planeOff; j < B->planeOff + B->planeCnt; j++) {
            float dn = dot(ldir, s->planes[j].n);
            float dp = dot(p, s->planes[j].n) - s->planes[j].d;
            if (fabsf(dn) < 1e-8f) {
                if (dp > 0.0f) { outside = true; break; }
                continue;
            }
            float t = -dp / dn;
            if (dn < 0.0f) {
                if (t > tN) tN = t;
            } else {
                if (t < tF) tF = t;
            }
            if (tN > tF) { outside = true; break; }
        }
        if (outside)
            continue;
        if (tN < dist - 1e-3f && tF > 1e-3f)
            return false;   /* сегмент проходит через тело */
    }
    return true;
}

/* освещённость точки: диффузная + зеркальная (Блинн-Фонг) по всем
 * источникам; hitBody — тело, на котором лежит точка (-1 — пол):
 * источники этого тела не проверяются на затенение; noShadow —
 * отключить проверку теней (внутренние попадания) */
static __host__ __device__ vec3 shade(const Scene *s, vec3 p, vec3 n,
                                      vec3 view, vec3 base, int hitBody,
                                      bool noShadow, float scale,
                                      unsigned long long *rays)
{
    vec3 c = base * 0.06f;
    int i;

    for (i = 0; i < s->nLights; i++) {
        const Light *L = &s->lights[i];
        vec3 ld = L->pos - p;
        float dist = vlen(ld);
        vec3 ldir, hv;
        float ndl, atten, k, sp;

        if (dist < 1e-6f)
            continue;
        ldir = ld * (1.0f / dist);
        ndl = dot(n, ldir);
        if (ndl <= 0.0f)
            continue;
        if (!noShadow && L->body != hitBody) {
            (*rays)++;
            if (!shadowVisible(s, p, ldir, dist, hitBody))
                continue;
        }
        /* реберные источники светят локально (спад ~1/d^2), основные —
         * без спада; реберных много, поэтому каждый взят с kEdgeGain */
        atten = (L->body != kMainBodyTag) ? 1.0f / (1.0f + 0.6f * dist * dist)
                                          : 1.0f;
        k = atten * ndl * scale;
        c = c + base * (L->color * k);
        hv = vnorm(ldir + view);
        sp = dot(n, hv);
        if (sp > 0.4f)
            c = c + L->color * (atten * scale * 0.6f * powf(sp, 48.0f));
    }
    return c;
}

/* путь преломлённого луча (упрощённая прозрачность): луч входит в тело
 * и далее только преломляется; зеркальные отражения не порождаются,
 * но внутри тела возможно полное внутреннее отражение (эффект
 * бесконечности); на полу луч гаснет */
static __host__ __device__ vec3 transmitPath(const Scene *s, vec3 p, vec3 d,
                                              vec3 nS, int body, float w0,
                                              unsigned long long *rays)
{
    vec3 col = v3(0.0f, 0.0f, 0.0f);
    vec3 o, r;
    float w = w0;
    int inside = body;
    int it;

    if (!refractDir(d, nS, 1.0f / kIor, &r))
        return col;   /* при входе в более плотную среду ТИО невозможно */
    d = r;
    o = p - nS * kRayEps;

    for (it = 0; it < s->maxDepth && w > kMinWeight; it++) {
        Hit h;
        vec3 hp, nOut, nS2, rr;
        const BodyG *B;

        (*rays)++;
        h = sceneHit(s, o, d);
        if (h.obj < 0) {
            col = col + skyColor(d) * w;
            break;
        }
        hp = o + d * h.t;
        if (h.obj == 3) {
            vec3 base = texSampleFloor(s, hp) * s->fTint;
            col = col + shade(s, hp, v3(0.0f, 0.0f, 1.0f), -d, base, -1,
                              false, 1.0f, rays) * w;
            break;   /* пол: преломлённый луч больше не отражается */
        }
        B = &s->bodies[h.obj];
        nOut = h.n;
        nS2 = h.fromInside ? -nOut : nOut;
        if (h.obj == inside) {
            /* внутреннее попадание: шейдинг без теневых лучей */
            float inScale = 0.3f + 0.7f * B->transp;
            col = col + shade(s, hp, nS2, -d, B->color, h.obj, true,
                              inScale, rays) * w;
            if (refractDir(d, nS2, kIor, &rr)) {
                d = rr;                        /* выход из тела */
                o = hp + nOut * kRayEps;
                w *= B->transp;
                inside = -1;
            } else {
                /* полное внутреннее отражение: физически без потерь,
                 * вес не гасим — многократные переотражения внутри тела
                 * дают эффект бесконечности */
                d = reflectDir(d, nS2);
                o = hp - nOut * kRayEps;
            }
        } else {
            /* внешнее попадание в другое тело: войти или погаснуть */
            col = col + shade(s, hp, nS2, -d, B->color, h.obj, false, 1.0f,
                              rays) * w;
            if (B->transp <= kMinWeight)
                break;
            if (!refractDir(d, nS2, 1.0f / kIor, &rr))
                break;
            d = rr;
            o = hp - nS2 * kRayEps;
            w *= B->transp;
            inside = h.obj;
        }
    }
    return col;
}

/* основная трассировка луча: итеративно (без рекурсии) до maxDepth;
 * на каждом попадании — шейдинг, затем под-путь прозрачности и
 * зеркальное продолжение с затуханием веса */
static __host__ __device__ vec3 traceRay(const Scene *s, vec3 o, vec3 d,
                                         unsigned long long *rays)
{
    vec3 col = v3(0.0f, 0.0f, 0.0f);
    float w = 1.0f;
    int it;

    for (it = 0; it < s->maxDepth && w > kMinWeight; it++) {
        Hit h;
        vec3 p, nOut, nS;
        const BodyG *B;

        (*rays)++;
        h = sceneHit(s, o, d);
        if (h.obj < 0) {
            col = col + skyColor(d) * w;
            break;
        }
        p = o + d * h.t;
        if (h.obj == 3) {
            vec3 base = texSampleFloor(s, p) * s->fTint;
            col = col + shade(s, p, v3(0.0f, 0.0f, 1.0f), -d, base, -1,
                              false, 1.0f, rays) * w;
            if (s->fRefl <= kMinWeight)
                break;
            w *= s->fRefl;
            d = reflectDir(d, v3(0.0f, 0.0f, 1.0f));
            o = p + v3(0.0f, 0.0f, kRayEps);
            continue;
        }
        B = &s->bodies[h.obj];
        nOut = h.n;
        nS = h.fromInside ? -nOut : nOut;
        col = col + shade(s, p, nS, -d, B->color, h.obj, false, 1.0f,
                          rays) * w;
        if (B->transp > kMinWeight)
            col = col + transmitPath(s, p, d, nS, h.obj, w * B->transp, rays);
        if (B->refl <= kMinWeight)
            break;
        w *= B->refl;
        d = reflectDir(d, nS);
        o = p + nS * kRayEps;
    }
    return col;
}

/* экспозиция + тон-маппинг (Рейнхард): суммарная освещённость от многих
 * источников — HDR-величина, заметно превышающая 1; мягкое сжатие в [0,1)
 * вместо жёсткого клиппинга в toByte сохраняет цвет и детали в светах */
static __host__ __device__ inline vec3 toneMap(vec3 c)
{
    c = c * kExposure;
    return v3(c.x / (1.0f + c.x), c.y / (1.0f + c.y), c.z / (1.0f + c.z));
}

/* каркасный рендер: минимальное расстояние от луча до отрезков сцены */
static __host__ __device__ vec3 wireColor(const Scene *s, vec3 o, vec3 d)
{
    vec3 col = v3(0.012f, 0.02f, 0.045f);
    float best = 1e30f;
    int i;

    for (i = 0; i < s->nWire; i++) {
        const WireSeg *g = &s->wire[i];
        vec3 v = g->b - g->a;
        vec3 w0 = o - g->a;
        float C = dot(v, v);
        float B = dot(d, v);
        float D = dot(d, w0);
        float E = dot(v, w0);
        float tc, sc, dist, th;
        vec3 pc;

        if (C < 1e-12f) {
            tc = 0.0f;
        } else {
            float den = C - B * B;
            if (den < 1e-9f)
                tc = E / C;
            else
                tc = (E - B * D) / den;
        }
        if (tc < 0.0f) tc = 0.0f;
        if (tc > 1.0f) tc = 1.0f;
        sc = B * tc - D;
        if (sc < 0.0f) sc = 0.0f;
        pc = (o + d * sc) - (g->a + v * tc);
        dist = vlen(pc);
        th = 0.0045f * sc + 0.0012f;
        if (dist < th && sc < best) {
            best = sc;
            col = g->color;
        }
    }
    return col;
}

static __host__ __device__ vec3 renderPixel(const Scene *s,
                                             const RenderParams *rp, int x,
                                             int y, unsigned long long *rays)
{
    vec3 col = v3(0.0f, 0.0f, 0.0f);
    int n = rp->ssaa;
    int i, j;

    for (j = 0; j < n; j++) {
        for (i = 0; i < n; i++) {
            float sx = ((float)x + (i + 0.5f) / n) / (float)rp->W;
            float sy = ((float)y + (j + 0.5f) / n) / (float)rp->H;
            /* fov задан по горизонтали: полуширина экрана в тангенцах —
             * tanF, полувысота — tanF/aspect */
            float px = (2.0f * sx - 1.0f) * rp->tanF;
            float py = (1.0f - 2.0f * sy) * rp->tanF / rp->aspect;
            vec3 d = vnorm(rp->fwd + rp->right * px + rp->up * py);
            if (rp->wire) {
                (*rays)++;
                col = col + wireColor(s, rp->pos, d);
            } else {
                col = col + traceRay(s, rp->pos, d, rays);
            }
        }
    }
    vec3 c = col * (1.0f / (float)(n * n));
    if (!rp->wire)
        c = toneMap(c);   /* каркасный рендер уже корректно яркий — не трогаем */
    return c;
}

/* ---------- ядро CUDA ---------- */

__global__ void renderKernel(const Scene *s, RenderParams rp,
                             unsigned char *out, unsigned long long *rayCounter)
{
    int strideX = gridDim.x * blockDim.x;
    int strideY = gridDim.y * blockDim.y;
    int y, x;

    for (y = blockIdx.y * blockDim.y + threadIdx.y; y < rp.H; y += strideY) {
        for (x = blockIdx.x * blockDim.x + threadIdx.x; x < rp.W; x += strideX) {
            unsigned long long rays = 0;
            vec3 c = renderPixel(s, &rp, x, y, &rays);
            unsigned char *q = out + 4 * (size_t)y * (size_t)rp.W +
                               4 * (size_t)x;
            q[0] = toByte(c.x);
            q[1] = toByte(c.y);
            q[2] = toByte(c.z);
            q[3] = 255;
            atomicAdd(rayCounter, rays);
        }
    }
}

/* ---------- рендер на CPU (без CUDA) ---------- */

static void cpuRender(const Scene *s, const RenderParams *rp,
                      unsigned char *out, unsigned long long *raysTotal,
                      int nThreads)
{
    std::vector<unsigned long long> counts(nThreads, 0);
    int tid;

    /* количество потоков фиксировано и не зависит от входных данных */
    auto worker = [&](int id) {
        int x, y;
        for (y = id; y < rp->H; y += nThreads) {
            for (x = 0; x < rp->W; x++) {
                unsigned long long rays = 0;
                vec3 c = renderPixel(s, rp, x, y, &rays);
                unsigned char *q = out + 4 * (size_t)y * (size_t)rp->W +
                                   4 * (size_t)x;
                q[0] = toByte(c.x);
                q[1] = toByte(c.y);
                q[2] = toByte(c.z);
                q[3] = 255;
                counts[id] += rays;
            }
        }
    };

    if (nThreads == 1) {
        worker(0);
    } else {
        std::vector<std::thread> pool;
        for (tid = 0; tid < nThreads; tid++)
            pool.push_back(std::thread(worker, tid));
        for (tid = 0; tid < nThreads; tid++)
            pool[tid].join();
    }
    *raysTotal = 0;
    for (tid = 0; tid < nThreads; tid++)
        *raysTotal += counts[tid];
}

/* ---------- камера ---------- */

static vec3 evalCyl(const CylPath *p, double t)
{
    double r = p->r0 + p->ar * sin(p->wr * t + p->pr);
    double z = p->z0 + p->az * sin(p->wz * t + p->pz);
    double ph = p->p0 + p->wp * t;
    return v3((float)(r * cos(ph)), (float)(r * sin(ph)), (float)z);
}

static RenderParams makeRenderParams(const Config *cfg, double t, int wire)
{
    RenderParams rp;
    vec3 pos, tgt, fwd, right, up, upRef;

    pos = evalCyl(&cfg->cam, t);
    tgt = evalCyl(&cfg->tgt, t);
    fwd = tgt - pos;
    if (vlen(fwd) < 1e-6f)
        fail("degenerate camera: position coincides with look-at point");
    fwd = vnorm(fwd);
    /* если взгляд почти вертикален, ось z нельзя использовать как "верх" */
    upRef = (fabsf(fwd.z) < 0.9f) ? v3(0.0f, 0.0f, 1.0f) : v3(0.0f, 1.0f, 0.0f);
    right = vnorm(cross(fwd, upRef));
    up = vnorm(cross(right, fwd));

    rp.pos = pos;
    rp.fwd = fwd;
    rp.right = right;
    rp.up = up;
    rp.tanF = tanf((float)(cfg->fov * 0.5 * kPi / 180.0));
    rp.aspect = (float)cfg->W / (float)cfg->H;
    rp.W = cfg->W;
    rp.H = cfg->H;
    rp.ssaa = cfg->ssaa;
    rp.wire = wire;
    return rp;
}

/* ---------- вывод изображения (формат ЛР2) ---------- */

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

static void writeImage(const char *path, int w, int h, const unsigned char *px)
{
    size_t n = (size_t)w * (size_t)h * 4;
    FILE *f = fopen(path, "wb");
    if (f == NULL)
        failf("cannot open output file %s", path);
    writeLE32(f, w);
    writeLE32(f, h);
    if (fwrite(px, 1, n, f) != n)
        failf("failed to write image %s", path);
    if (fclose(f) != 0)
        failf("failed to close output file %s", path);
}

/* ---------- конфигурация по умолчанию (--default) ---------- */

static void printDefaultConfig(void)
{
    printf("240\n");
    printf("kp_out/img_%%d.data\n");
    printf("960 540 80\n");
    printf("8.5 3.2 0.0 1.2 0.8 2.0 3.0 1.0 0.5 1.0\n");
    printf("0.6 1.0 0.0 0.4 0.3 3.0 2.0 -2.0 0.0 0.0\n");
    printf("2.3 0.3 0.3 1.0 0.25 0.15 1.3 0.40 0.55 8\n");
    printf("-2.2 2.0 0.6 0.2 1.0 0.35 1.6 0.35 0.60 4\n");
    printf("-1.8 -2.4 0.4 0.25 0.5 1.0 1.4 0.45 0.50 6\n");
    printf("-6.0 -6.0 -1.0 -6.0 6.0 -1.0 6.0 6.0 -1.0 6.0 -6.0 -1.0 "
           "floor.data 1.0 1.0 1.0 0.35\n");
    printf("3\n");
    printf("0.0 0.0 8.0 0.45 0.45 0.45\n");
    printf("-8.0 -6.0 6.0 0.45 0.34 0.24\n");
    printf("8.0 -7.0 5.0 0.22 0.32 0.45\n");
    printf("10 2\n");
}

/* ---------- main ---------- */

int main(int argc, char **argv)
{
    int cpuMode = 0, wireMode = 0, defaultMode = 0;
    int i, nThreads = 1;
    int texW = 0, texH = 0;
    Config cfg;
    Scene hScene;
    std::vector<Light> lights;
    std::vector<WireSeg> wire;
    std::vector<unsigned char> texData;
    std::vector<unsigned char> hostBuf;
    char framePath[PATH_LEN];
    long long frame;
    size_t pxBytes;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--cpu") == 0)
            cpuMode = 1;
        else if (strcmp(argv[i], "--gpu") == 0)
            cpuMode = 0;
        else if (strcmp(argv[i], "--default") == 0)
            defaultMode = 1;
        else if (strcmp(argv[i], "--wireframe") == 0)
            wireMode = 1;
        else
            failf("unknown key %s (use --cpu, --gpu, --default or --wireframe)",
                  argv[i]);
    }

    if (defaultMode) {
        printDefaultConfig();
        return 0;
    }

    parseInput(&cfg);
    loadTexture(cfg.texPath, &texW, &texH, &texData);
    buildScene(&cfg, &hScene, &lights, &wire);
    hScene.texW = texW;
    hScene.texH = texH;
    hScene.tex = texData.data();
    pxBytes = (size_t)cfg.W * (size_t)cfg.H * 4;
    hostBuf.resize(pxBytes);

    if (cpuMode) {
        nThreads = (int)std::thread::hardware_concurrency();
        if (nThreads < 1)
            nThreads = 1;
        if (nThreads > 64)
            nThreads = 64;
        fprintf(stderr, "INFO: mode=cpu threads=%d\n", nThreads);
    } else {
        int devCount = 0;
        cudaDeviceProp prop;
        checkCuda(cudaGetDeviceCount(&devCount), "cudaGetDeviceCount failed");
        if (devCount < 1)
            fail("no CUDA device available");
        checkCuda(cudaGetDeviceProperties(&prop, 0),
                  "cudaGetDeviceProperties failed");
        fprintf(stderr, "INFO: mode=gpu device=%s\n", prop.name);
    }
    fprintf(stderr,
            "INFO: %dx%d, %lld frames, depth=%d, ssaa=%d (%d rays/pixel), "
            "lights=%d, wire=%d\n",
            cfg.W, cfg.H, cfg.frames, cfg.maxDepth, cfg.ssaa, cfg.ssaa * cfg.ssaa,
            hScene.nLights, wireMode);

    /* --- данные для GPU --- */
    Scene *dScene = NULL;
    Light *dLights = NULL;
    WireSeg *dWire = NULL;
    unsigned char *dTex = NULL;
    unsigned char *dOut = NULL;
    unsigned long long *dRays = NULL;

    if (!cpuMode) {
        Scene devScene = hScene;
        checkCuda(cudaMalloc((void **)&dScene, sizeof(Scene)), "cudaMalloc scene");
        if (hScene.nLights > 0) {
            checkCuda(cudaMalloc((void **)&dLights,
                                 hScene.nLights * sizeof(Light)),
                      "cudaMalloc lights");
            checkCuda(cudaMemcpy(dLights, hScene.lights,
                                 hScene.nLights * sizeof(Light),
                                 cudaMemcpyHostToDevice),
                      "memcpy lights");
            devScene.lights = dLights;
        }
        checkCuda(cudaMalloc((void **)&dWire, hScene.nWire * sizeof(WireSeg)),
                  "cudaMalloc wire");
        checkCuda(cudaMemcpy(dWire, hScene.wire, hScene.nWire * sizeof(WireSeg),
                             cudaMemcpyHostToDevice),
                  "memcpy wire");
        devScene.wire = dWire;
        checkCuda(cudaMalloc((void **)&dTex, texData.size()), "cudaMalloc tex");
        checkCuda(cudaMemcpy(dTex, texData.data(), texData.size(),
                             cudaMemcpyHostToDevice),
                  "memcpy tex");
        devScene.tex = dTex;
        checkCuda(cudaMalloc((void **)&dOut, pxBytes), "cudaMalloc out");
        checkCuda(cudaMalloc((void **)&dRays, sizeof(unsigned long long)),
                  "cudaMalloc rays");
        checkCuda(cudaMemcpy(dScene, &devScene, sizeof(Scene),
                             cudaMemcpyHostToDevice),
                  "memcpy scene");
    }

    /* --- покадровый рендер --- */
    for (frame = 0; frame < cfg.frames; frame++) {
        double t = (double)frame / (double)cfg.frames * 2.0 * kPi;
        RenderParams rp = makeRenderParams(&cfg, t, wireMode);
        double t0, ms;
        unsigned long long totalRays = 0;

        t0 = nowMs();
        if (!cpuMode) {
#ifdef __CUDACC__
            checkCuda(cudaMemset(dRays, 0, sizeof(unsigned long long)),
                      "memset rays");
            renderKernel<<<dim3(GRID_X, GRID_Y), dim3(BLOCK_X, BLOCK_Y)>>>(
                dScene, rp, dOut, dRays);
            checkCuda(cudaGetLastError(), "kernel launch failed");
            checkCuda(cudaDeviceSynchronize(), "render failed");
            checkCuda(cudaMemcpy(hostBuf.data(), dOut, pxBytes,
                                 cudaMemcpyDeviceToHost),
                      "memcpy out failed");
            checkCuda(cudaMemcpy(&totalRays, dRays, sizeof(totalRays),
                                 cudaMemcpyDeviceToHost),
                      "memcpy rays failed");
#else
            fail("GPU mode is not available in this build");
#endif
        } else {
            cpuRender(&hScene, &rp, hostBuf.data(), &totalRays, nThreads);
        }
        ms = nowMs() - t0;

        snprintf(framePath, PATH_LEN, cfg.outPath, (int)frame);
        writeImage(framePath, cfg.W, cfg.H, hostBuf.data());
        printf("%d\t%.3f\t%llu\n", (int)frame, ms, totalRays);
        fflush(stdout);
    }

    if (!cpuMode) {
        cudaFree(dScene);
        cudaFree(dLights);
        cudaFree(dWire);
        cudaFree(dTex);
        cudaFree(dOut);
        cudaFree(dRays);
    }
    return 0;
}
