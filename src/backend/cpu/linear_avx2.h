#ifndef VELRI_LINEAR_AVX2_H
#define VELRI_LINEAR_AVX2_H

#include <immintrin.h>

/* SIMD is across independent outputs. Explicit four-float loads avoid the
 * load/extract/reconvert sequence emitted by GCC for the scalar C loop.
 * x_double is either null (use x_float) or points to an expanded input.
 * Call sites pass a constant null pointer for the unused representation.
 * sum already holds the FP64 initial value or preceding panel's partials. */
[[gnu::always_inline]] static inline void avx2_accumulate1(
    const float *x_float, const double *x_double, const float *weights,
    size_t count, double *sum) {
    __m256d acc[VELRI_COLUMNS / 4];
    for (size_t j = 0; j < VELRI_COLUMNS / 4; ++j)
        acc[j] = _mm256_loadu_pd(sum + 4 * j);
    for (size_t i = 0; i < count; ++i) {
        const __m256d input = x_double ? _mm256_broadcast_sd(x_double + i)
            : _mm256_cvtps_pd(_mm_broadcast_ss(x_float + i));
#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 8
#endif
        for (size_t j = 0; j < VELRI_COLUMNS / 4; ++j)
            acc[j] = _mm256_fmadd_pd(input,
                _mm256_cvtps_pd(_mm_loadu_ps(weights + i * VELRI_COLUMNS + 4 * j)), acc[j]);
    }
    for (size_t j = 0; j < VELRI_COLUMNS / 4; ++j)
        _mm256_storeu_pd(sum + 4 * j, acc[j]);
}

#if VELRI_COLUMNS == 32
/* Internal AVX2/FMA tile: three rows by sixteen columns, sequential K.
 * Include only in an AVX2 compilation of linear_kernel.c.
 * Twelve accumulators + three inputs + one converted weight use all sixteen
 * YMM registers. Keeping this assignment explicit avoids a spilled accumulator
 * in the compiler-generated 3x16 loop. No ISA/model dispatch lives here.
 * s0/s1/s2 each have 16 doubles; x0/x1/x2 each have n doubles. The prepared
 * FP32 weight rows have VELRI_COLUMNS elements and at least 16 usable slots.
 * Inputs and sums are disjoint, n may be zero, and alignment is unrestricted.
 */
static inline void avx2_accumulate3(const double *x0, const double *x1, const double *x2,
                         const float *w, size_t n, double *s0, double *s1, double *s2) {
    __asm__ volatile (
        "vmovupd 0(%[s0]), %%ymm0\n\t"
        "vmovupd 32(%[s0]), %%ymm1\n\t"
        "vmovupd 64(%[s0]), %%ymm2\n\t"
        "vmovupd 96(%[s0]), %%ymm3\n\t"
        "vmovupd 0(%[s1]), %%ymm4\n\t"
        "vmovupd 32(%[s1]), %%ymm5\n\t"
        "vmovupd 64(%[s1]), %%ymm6\n\t"
        "vmovupd 96(%[s1]), %%ymm7\n\t"
        "vmovupd 0(%[s2]), %%ymm8\n\t"
        "vmovupd 32(%[s2]), %%ymm9\n\t"
        "vmovupd 64(%[s2]), %%ymm10\n\t"
        "vmovupd 96(%[s2]), %%ymm11\n\t"
        "test %[n], %[n]\n\t"
        "jz 2f\n\t"
        ".p2align 5\n\t"
        "1:\n\t"
        "vbroadcastsd (%[x0]), %%ymm12\n\t"
        "vbroadcastsd (%[x1]), %%ymm13\n\t"
        "vbroadcastsd (%[x2]), %%ymm14\n\t"
        "vcvtps2pd 0(%[w]), %%ymm15\n\t"
        "vfmadd231pd %%ymm15, %%ymm12, %%ymm0\n\t"
        "vfmadd231pd %%ymm15, %%ymm13, %%ymm4\n\t"
        "vfmadd231pd %%ymm15, %%ymm14, %%ymm8\n\t"
        "vcvtps2pd 16(%[w]), %%ymm15\n\t"
        "vfmadd231pd %%ymm15, %%ymm12, %%ymm1\n\t"
        "vfmadd231pd %%ymm15, %%ymm13, %%ymm5\n\t"
        "vfmadd231pd %%ymm15, %%ymm14, %%ymm9\n\t"
        "vcvtps2pd 32(%[w]), %%ymm15\n\t"
        "vfmadd231pd %%ymm15, %%ymm12, %%ymm2\n\t"
        "vfmadd231pd %%ymm15, %%ymm13, %%ymm6\n\t"
        "vfmadd231pd %%ymm15, %%ymm14, %%ymm10\n\t"
        "vcvtps2pd 48(%[w]), %%ymm15\n\t"
        "vfmadd231pd %%ymm15, %%ymm12, %%ymm3\n\t"
        "vfmadd231pd %%ymm15, %%ymm13, %%ymm7\n\t"
        "vfmadd231pd %%ymm15, %%ymm14, %%ymm11\n\t"
        "add $8, %[x0]\n\t"
        "add $8, %[x1]\n\t"
        "add $8, %[x2]\n\t"
        "add $%c[ws], %[w]\n\t"
        "dec %[n]\n\t"
        "jnz 1b\n\t"
        "2:\n\t"
        "vmovupd %%ymm0, 0(%[s0])\n\t"
        "vmovupd %%ymm1, 32(%[s0])\n\t"
        "vmovupd %%ymm2, 64(%[s0])\n\t"
        "vmovupd %%ymm3, 96(%[s0])\n\t"
        "vmovupd %%ymm4, 0(%[s1])\n\t"
        "vmovupd %%ymm5, 32(%[s1])\n\t"
        "vmovupd %%ymm6, 64(%[s1])\n\t"
        "vmovupd %%ymm7, 96(%[s1])\n\t"
        "vmovupd %%ymm8, 0(%[s2])\n\t"
        "vmovupd %%ymm9, 32(%[s2])\n\t"
        "vmovupd %%ymm10, 64(%[s2])\n\t"
        "vmovupd %%ymm11, 96(%[s2])\n\t"
        : [x0] "+&r"(x0), [x1] "+&r"(x1), [x2] "+&r"(x2), [w] "+&r"(w), [n] "+&r"(n)
        : [s0] "r"(s0), [s1] "r"(s1), [s2] "r"(s2), [ws] "i"(VELRI_COLUMNS * sizeof(float))
        : "cc", "memory", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14", "ymm15");
}

#endif

#endif
