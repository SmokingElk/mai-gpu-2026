#include <stdio.h>
#include <stdlib.h>

static void fail(const char *msg)
{
    fprintf(stderr, "ERROR: %s\n", msg);
    exit(0);
}

int main(void)
{
    int n;
    if (scanf("%d", &n) != 1)
        fail("failed to read array size");
    if (n < 0)
        fail("invalid array size");

    float *arr = NULL;
    if (n > 0) {
        arr = (float *)malloc((size_t)n * sizeof(float));
        if (arr == NULL)
            fail("out of memory");
    }

    for (int i = 0; i < n; i++) {
        if (scanf("%f", &arr[i]) != 1)
            fail("failed to read array element");
    }

    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - 1 - i; j++) {
            if (arr[j] > arr[j + 1]) {
                float t = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = t;
            }
        }
    }

    for (int i = 0; i < n; i++) {
        if (i > 0)
            printf(" ");
        printf("%.6e", arr[i]);
    }
    printf("\n");

    free(arr);
    return 0;
}
