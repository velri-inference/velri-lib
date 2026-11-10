/* Compile this common C algorithm for each target. SIMD is across independent
 * outputs, never across the K reduction. Weights remain FP32 in
 * [ceil(M / C), K, C] order. Target primitives preserve this same reduction.
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <float.h>

#ifndef VELRI_KERNEL
#error "VELRI_KERNEL must name this compiled variant"
#endif
#ifndef VELRI_COLUMNS
#error "VELRI_COLUMNS must describe the column tile"
#endif
#ifndef VELRI_PAIR_COLUMNS
#define VELRI_PAIR_COLUMNS VELRI_COLUMNS
#endif
static_assert(VELRI_PAIR_COLUMNS > 0 && VELRI_COLUMNS % VELRI_PAIR_COLUMNS == 0,
               "row-pair tile must divide the stored column tile");

static_assert(sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24, "FP32 is required");
static_assert(sizeof(double) == 8 && DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024,
               "FP64 is required");

/* FP32 finite products are exact in FP64 (at most 48 significant bits).
 * Contracting this product and its sequential FP64 addition therefore keeps
 * every finite-input result identical to separate multiply/add. No fast-math,
 * no multiple K accumulators, and no rounding of partial sums to FP32.
 */
#if defined(__clang__)
#pragma STDC FP_CONTRACT ON
#endif

static inline float finish(double sum, double bias) {
    float value = (float)(sum + bias);
    /* C arithmetic does not prescribe NaN payload propagation. A single
     * canonical output representation makes the C contract deterministic. */
    uint32_t bits;
    memcpy(&bits, &value, sizeof bits);
    if ((bits & UINT32_C(0x7fffffff)) > UINT32_C(0x7f800000)) {
        bits = UINT32_C(0x7fc00000);
        memcpy(&value, &bits, sizeof value);
    }
    return value;
}

#if defined(__AVX2__) && !defined(__AVX512F__) && VELRI_COLUMNS >= 4
#define HAVE_AVX2_TILE 1
#include "linear_avx2.h"
#else
#define HAVE_AVX2_TILE 0
#endif

static void direct_linear(const float *x, const float *packed, const float *bias, float *restrict y,
                  size_t rows, size_t m, size_t k, size_t x_stride, size_t y_stride) {
    for (size_t column = 0; column < m; column += VELRI_COLUMNS) {
        size_t row = 0;
        /* Reuse each load and FP32->FP64 conversion for two independent input
         * rows. K order within each output is unchanged. Single-token GEMV skips
         * this loop; the final odd GEMM row follows the same GEMV path. */
        for (; rows - row >= 2; row += 2) {
            for (size_t part = 0; part < VELRI_COLUMNS && part < m - column;
                 part += VELRI_PAIR_COLUMNS) {
                double sum0[VELRI_PAIR_COLUMNS], sum1[VELRI_PAIR_COLUMNS];
                for (size_t j = 0; j < VELRI_PAIR_COLUMNS; ++j)
                    sum0[j] = sum1[j] = -0.0;
                for (size_t i = 0; i < k; ++i) {
                    const double x0 = (double)x[row * x_stride + i];
                    const double x1 = (double)x[(row + 1) * x_stride + i];
#if defined(__clang__)
#pragma clang loop unroll(full)
#endif
                    for (size_t j = 0; j < VELRI_PAIR_COLUMNS; ++j) {
                        const double weight =
                            (double)packed[column * k + i * VELRI_COLUMNS + part + j];
                        sum0[j] += x0 * weight;
                        sum1[j] += x1 * weight;
                    }
                }
                const size_t left = m - column - part;
                const size_t width = left < VELRI_PAIR_COLUMNS ? left : VELRI_PAIR_COLUMNS;
                if (bias) {
                    for (size_t j = 0; j < width; ++j) {
                        const double offset = (double)bias[column + part + j];
                        y[row * y_stride + column + part + j] = finish(sum0[j], offset);
                        y[(row + 1) * y_stride + column + part + j] = finish(sum1[j], offset);
                    }
                } else {
                    for (size_t j = 0; j < width; ++j) {
                        y[row * y_stride + column + part + j] = finish(sum0[j], 0.0);
                        y[(row + 1) * y_stride + column + part + j] = finish(sum1[j], 0.0);
                    }
                }
            }
        }
        for (; row < rows; ++row) {
            double sum[VELRI_COLUMNS];
            for (size_t j = 0; j < VELRI_COLUMNS; ++j)
                sum[j] = -0.0;
#if HAVE_AVX2_TILE
            avx2_accumulate1(x + row * x_stride, NULL, packed + column * k, k, sum);
#else
            for (size_t i = 0; i < k; ++i) {
                const double input = (double)x[row * x_stride + i];
#if defined(__clang__)
#pragma clang loop unroll(full)
#endif
                for (size_t j = 0; j < VELRI_COLUMNS; ++j)
                    sum[j] += input * (double)packed[column * k + i * VELRI_COLUMNS + j];
            }
#endif
            const size_t width = m - column < VELRI_COLUMNS ? m - column : VELRI_COLUMNS;
            if (bias) {
                for (size_t j = 0; j < width; ++j)
                    y[row * y_stride + column + j] = finish(sum[j], (double)bias[column + j]);
            } else {
                for (size_t j = 0; j < width; ++j)
                    y[row * y_stride + column + j] = finish(sum[j], 0.0);
            }
        }
    }
}

#if HAVE_AVX2_TILE && VELRI_COLUMNS == 32
/* A bounded temporary input copy removes a repeated conversion from every
 * output tile. It is per-call storage; prepared weights still occupy FP32.
 * The caller uses this path only when enough tiles reuse the copy. */
#define INPUT_CACHE_COUNT 2048
[[gnu::noinline]] static void cached_gemv(const float *x, const float *packed,
                                         const float *bias, float *restrict y,
                                         size_t m, size_t k) {
    double input64[INPUT_CACHE_COUNT];
    for (size_t i = 0; i < k; ++i)
        input64[i] = (double)x[i];
    for (size_t column = 0; column < m; column += VELRI_COLUMNS) {
        double sum[VELRI_COLUMNS];
        for (size_t j = 0; j < VELRI_COLUMNS; ++j)
            sum[j] = -0.0;
        avx2_accumulate1(NULL, input64, packed + column * k, k, sum);
        const size_t width = m - column < VELRI_COLUMNS ? m - column : VELRI_COLUMNS;
        if (bias) {
            for (size_t j = 0; j < width; ++j)
                y[column + j] = finish(sum[j], (double)bias[column + j]);
        } else {
            for (size_t j = 0; j < width; ++j)
                y[column + j] = finish(sum[j], 0.0);
        }
    }
}

/* A 16 KiB weight panel, 6 KiB FP64 partials and 3 KiB input conversion
 * scratch fit together in a 32 KiB L1 data cache. These bounds do not depend
 * on matrix dimensions or processor model. Store partials in FP64, carrying
 * the same accumulator through successive K panels; round only at the end. */
enum { ROW_BLOCK = 24, PANEL_K = 128 };
[[gnu::noinline]] static void blocked_gemm(const float *x, const float *packed,
                                           const float *bias, float *restrict y,
                                           size_t rows, size_t m, size_t k,
                                           size_t x_stride, size_t y_stride) {
    for (size_t column = 0; column < m; column += VELRI_COLUMNS) {
        const size_t width = m - column < VELRI_COLUMNS ? m - column : VELRI_COLUMNS;
        for (size_t rb = 0; rb < rows; rb += ROW_BLOCK) {
            const size_t nr = rows - rb < ROW_BLOCK ? rows - rb : ROW_BLOCK;
            double partial[ROW_BLOCK][VELRI_COLUMNS];
            double input[3][PANEL_K];
            for (size_t r = 0; r < nr; ++r)
                for (size_t j = 0; j < VELRI_COLUMNS; ++j)
                    partial[r][j] = -0.0;
            for (size_t kb = 0; kb < k; kb += PANEL_K) {
                const size_t count = k - kb < PANEL_K ? k - kb : PANEL_K;
                size_t r = 0;
                for (; nr - r >= 3; r += 3) {
                    for (size_t rr = 0; rr < 3; ++rr)
                        for (size_t i = 0; i < count; ++i)
                            input[rr][i] = (double)x[(rb + r + rr) * x_stride + kb + i];
                    for (size_t part = 0; part < width; part += 16)
                        avx2_accumulate3(input[0], input[1], input[2],
                                         packed + column * k + kb * VELRI_COLUMNS + part,
                                         count, partial[r] + part, partial[r + 1] + part,
                                         partial[r + 2] + part);
                }
                /* One or two remaining rows use the same ordered reduction.
                 * Keep the sums in FP64 across panel boundaries here too. */
                for (; r < nr; ++r) {
                    double sum[VELRI_COLUMNS];
                    for (size_t j = 0; j < VELRI_COLUMNS; ++j)
                        sum[j] = partial[r][j];
                    avx2_accumulate1(x + (rb + r) * x_stride + kb, NULL,
                                     packed + column * k + kb * VELRI_COLUMNS, count, sum);
                    for (size_t j = 0; j < VELRI_COLUMNS; ++j)
                        partial[r][j] = sum[j];
                }
            }
            if (bias) {
                for (size_t r = 0; r < nr; ++r)
                    for (size_t j = 0; j < width; ++j)
                        y[(rb + r) * y_stride + column + j] =
                            finish(partial[r][j], (double)bias[column + j]);
            } else {
                for (size_t r = 0; r < nr; ++r)
                    for (size_t j = 0; j < width; ++j)
                        y[(rb + r) * y_stride + column + j] = finish(partial[r][j], 0.0);
            }
        }
    }
}
#endif

void VELRI_KERNEL(const float *x, const float *packed, const float *bias, float *restrict y,
                  size_t rows, size_t m, size_t k, size_t x_stride, size_t y_stride) {
#if HAVE_AVX2_TILE && VELRI_COLUMNS == 32
    if (rows == 1 && m >= 8 * VELRI_COLUMNS && k <= INPUT_CACHE_COUNT) {
        cached_gemv(x, packed, bias, y, m, k);
        return;
    }
    if (rows >= 3 && k >= 64) {
        blocked_gemm(x, packed, bias, y, rows, m, k, x_stride, y_stride);
        return;
    }
#endif
    direct_linear(x, packed, bias, y, rows, m, k, x_stride, y_stride);
}
