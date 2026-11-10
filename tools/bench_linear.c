#define _POSIX_C_SOURCE 200809L
#include "velri_lib/linear.h"
#include "reference.h"
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(_WIN32)
#include <windows.h>
#endif

[[noreturn]] static void fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(1);
}

static double seconds(void) {
#if defined(_WIN32)
    LARGE_INTEGER counter, frequency;
    if (!QueryPerformanceCounter(&counter) || !QueryPerformanceFrequency(&frequency))
        fail("clock failed");
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t))
        fail("clock failed");
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
#endif
}

static size_t number(const char *text) {
    char *end;
    errno = 0;
    unsigned long long n = strtoull(text, &end, 10);
    if (errno || *end || end == text || text[0] == '-' || !n || n > SIZE_MAX)
        fail("dimensions and repetitions must be positive integers");
    return (size_t)n;
}

static size_t product(size_t a, size_t b) {
    if (a > SIZE_MAX / sizeof(float) / b)
        fail("matrix byte count overflows size_t");
    return a * b;
}

static float *allocate(size_t count) {
    float *p = malloc(count * sizeof(float));
    if (!p)
        fail("allocation failed");
    return p;
}

static void check(const float *actual, const float *expected, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        uint32_t a, e;
        memcpy(&a, actual + i, sizeof a);
        memcpy(&e, expected + i, sizeof e);
        if (a != e) {
            fprintf(stderr, "output %zu: actual=%08x reference=%08x\n", i, a, e);
            exit(1);
        }
    }
}

static int compare(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    if (argc != 7 && argc != 8) {
        fprintf(stderr,
                "usage: %s auto|base|avx2|avx512 rows M K repetitions samples "
                "[arithmetic-ceiling-TFLOPS]\n",
                argv[0]);
        return 2;
    }
    velri_cpu_variant variant;
    if (!strcmp(argv[1], "auto"))
        variant = VELRI_CPU_AUTO;
    else if (!strcmp(argv[1], "base"))
        variant = VELRI_CPU_BASE;
    else if (!strcmp(argv[1], "avx2"))
        variant = VELRI_CPU_AVX2;
    else if (!strcmp(argv[1], "avx512"))
        variant = VELRI_CPU_AVX512;
    else
        fail("unknown variant");
    size_t rows = number(argv[2]), m = number(argv[3]), k = number(argv[4]);
    size_t repetitions = number(argv[5]), samples = number(argv[6]);
    if (samples > 10001)
        fail("sample count exceeds 10001");
    double ceiling = 0.0;
    if (argc == 8) {
        char *end;
        errno = 0;
        ceiling = strtod(argv[7], &end);
        if (errno || *end || end == argv[7] || !(ceiling > 0.0) || ceiling > 1e9)
            fail("ceiling must be a finite positive TFLOPS value");
    }
    size_t xn = product(rows, k), wn = product(m, k), yn = product(rows, m);
    velri_linear_plan plan;
    if (velri_linear_query(m, k, variant, &plan) != VELRI_LINEAR_OK)
        fail("shape or variant unsupported");
    float *x = allocate(xn), *w = allocate(wn), *bias = allocate(m);
    float *packed = allocate(plan.packed_count), *actual = allocate(yn), *expected = allocate(yn);
    double *times = malloc(samples * sizeof *times);
    if (!times)
        fail("allocation failed");
    uint32_t state = UINT32_C(123456789);
    float *arrays[] = {x, w, bias};
    size_t counts[] = {xn, wn, m};
    for (size_t a = 0; a < 3; ++a) {
        for (size_t i = 0; i < counts[a]; ++i) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            arrays[a][i] = (float)((int32_t)(state & 65535) - 32768) / 16384.0f;
        }
    }
    double start = seconds();
    if (velri_linear_pack(&plan, w, wn, k, packed, plan.packed_count))
        fail("packing failed");
    double preparation = seconds() - start;
    reference_linear(x, w, bias, expected, rows, m, k, k, k, m);
    if (velri_linear_run(&plan, packed, plan.packed_count, x, xn, rows, k, bias, m, actual, yn, m))
        fail("execution failed");
    check(actual, expected, yn);
    start = seconds();
    do {
        if (velri_linear_run(&plan, packed, plan.packed_count, x, xn, rows, k, bias, m, actual, yn,
                             m))
            fail("warmup failed");
    } while (seconds() - start < 0.25);
    for (size_t s = 0; s < samples; ++s) {
        start = seconds();
        for (size_t r = 0; r < repetitions; ++r)
            if (velri_linear_run(&plan, packed, plan.packed_count, x, xn, rows, k, bias, m, actual,
                                 yn, m))
                fail("execution failed");
        times[s] = (seconds() - start) / (double)repetitions;
        check(actual, expected, yn);
        printf("sample=%zu seconds_per_call=%.12g\n", s, times[s]);
    }
    qsort(times, samples, sizeof *times, compare);
    double median = times[samples / 2];
    if (!(samples % 2))
        median = (median + times[samples / 2 - 1]) / 2;
    if (!(median > 0.0))
        fail("timer did not resolve the call; increase repetitions");
    double rate = 2.0 * (double)rows * (double)m * (double)k / median / 1e12;
    printf("threads=1 variant=%d columns=%zu rows=%zu M=%zu K=%zu\n", plan.variant, plan.columns,
           rows, m, k);
    printf("packed_bytes=%zu preparation_ms=%.6f bit_mismatches=0\n",
           plan.packed_count * sizeof(float), preparation * 1e3);
    printf("median_TFLOPS=%.6f median_ns=%.3f\n", rate, median * 1e9);
    if (ceiling > 0.0)
        printf("caller_supplied_arithmetic_ceiling_TFLOPS=%.6f arithmetic_fraction=%.6f\n", ceiling,
               rate / ceiling);
    else
        puts("arithmetic_ceiling_TFLOPS=unknown (provide an independently justified value)");
    puts("memory_ceiling_TFLOPS=unknown; repeated fixed weights, validation included, no "
         "worker-pool/graph timing");
    free(x);
    free(w);
    free(bias);
    free(packed);
    free(actual);
    free(expected);
    free(times);
    return 0;
}
