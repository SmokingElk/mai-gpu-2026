#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static void fail(const char *msg)
{
    fprintf(stderr, "ERROR: %s\n", msg);
    exit(0);
}

int main(void)
{
    float a, b, c;
    if (scanf("%f %f %f", &a, &b, &c) != 3)
        fail("failed to read coefficients");

    if (a == 0.0f) {
        if (b == 0.0f) {
            if (c == 0.0f)
                printf("any\n");
            else
                printf("incorrect\n");
        } else {
            printf("%.6f\n", -c / b);
        }
        return 0;
    }

    float d = b * b - 4.0f * a * c;
    if (d < 0.0f) {
        printf("imaginary\n");
    } else if (d == 0.0f) {
        printf("%.6f\n", -b / (2.0f * a));
    } else {
        float s = sqrtf(d);
        printf("%.6f %.6f\n", (-b + s) / (2.0f * a), (-b - s) / (2.0f * a));
    }
    return 0;
}
