#include "velri_lib/linear.h"
#include <stdint.h>
#include <string.h>

#define DECLARE(name)                                                                              \
    void name(const float *, const float *, const float *, float *, size_t, size_t, size_t,        \
              size_t, size_t)
#define VELRI_ENTRY(name, variant, columns) DECLARE(name);
#include "linear_variants.inc"
#undef VELRI_ENTRY

static const struct {
    velri_cpu_variant variant;
    size_t columns;
    velri_linear_kernel kernel;
} profiles[] = {
#define VELRI_ENTRY(name, variant, columns) {variant, columns, name},
#include "linear_variants.inc"
#undef VELRI_ENTRY
};

int velri_cpu_variant_available(velri_cpu_variant variant) {
    switch (variant) {
    case VELRI_CPU_AUTO:
    case VELRI_CPU_BASE:
        return 1;
#if VELRI_X86_VARIANTS
    /* These compiler builtins include the OS extended-state checks. The
     * dispatcher itself is compiled for baseline x86-64, without AVX. */
    case VELRI_CPU_AVX2:
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    case VELRI_CPU_AVX512:
        return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("fma");
#endif
    default:
        return 0;
    }
}

static int multiply(size_t a, size_t b, size_t *out) {
    if (b && a > SIZE_MAX / b)
        return 0;
    *out = a * b;
    return 1;
}

velri_linear_status velri_linear_query(size_t m, size_t k, velri_cpu_variant variant,
                                       velri_linear_plan *plan) {
    if (!plan || !m || !k)
        return VELRI_LINEAR_INVALID;
    if (!velri_cpu_variant_available(variant))
        return VELRI_LINEAR_UNSUPPORTED;
#if VELRI_X86_VARIANTS
    if (variant == VELRI_CPU_AUTO)
        variant = velri_cpu_variant_available(VELRI_CPU_AVX512) ? VELRI_CPU_AVX512
                  : velri_cpu_variant_available(VELRI_CPU_AVX2) ? VELRI_CPU_AVX2
                                                                : VELRI_CPU_BASE;
#endif
    if (variant == VELRI_CPU_AUTO)
        variant = VELRI_CPU_BASE;
    velri_linear_plan p = {m, k, 0, 0, variant, NULL};
    /* Pick the largest compiled tile that fits at least once. This bounds
     * padding to less than one extra actual matrix; M=1 stores exactly K
     * floats rather than padding to a wide SIMD tile. The algorithm is shared
     * by all widths and ISAs. Selection happens only during preparation. */
    for (size_t i = 0; i < sizeof profiles / sizeof profiles[0]; ++i) {
        if (profiles[i].variant == variant && profiles[i].columns <= m &&
            profiles[i].columns > p.columns) {
            p.columns = profiles[i].columns;
            p.kernel = profiles[i].kernel;
        }
    }
    if (!p.kernel)
        return VELRI_LINEAR_UNSUPPORTED;
    size_t tiles = m / p.columns + (m % p.columns != 0), padded;
    if (!multiply(tiles, p.columns, &padded) || !multiply(padded, k, &p.packed_count) ||
        p.packed_count > SIZE_MAX / sizeof(float))
        return VELRI_LINEAR_OVERFLOW;
    *plan = p;
    return VELRI_LINEAR_OK;
}

/* A span covers row gaps as well as used elements: output must not alias even
 * the padding of an input view. Integer checks avoid overflowing pointer math. */
static velri_linear_status span(const void *ptr, size_t capacity, size_t rows, size_t cols,
                                size_t stride, size_t *bytes) {
    if (stride < cols)
        return VELRI_LINEAR_INVALID;
    size_t count = 0;
    if (rows) {
        if (!multiply(rows - 1, stride, &count) || count > SIZE_MAX - cols)
            return VELRI_LINEAR_OVERFLOW;
        count += cols;
    }
    if (!multiply(count, sizeof(float), bytes))
        return VELRI_LINEAR_OVERFLOW;
    if (capacity < count || (count && !ptr))
        return VELRI_LINEAR_INVALID;
    if (count && ((uintptr_t)ptr % _Alignof(float)))
        return VELRI_LINEAR_INVALID;
    if ((uintptr_t)ptr > UINTPTR_MAX - *bytes)
        return VELRI_LINEAR_OVERFLOW;
    return VELRI_LINEAR_OK;
}

static int overlap(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    return a_bytes && b_bytes && (uintptr_t)a < (uintptr_t)b + b_bytes &&
           (uintptr_t)b < (uintptr_t)a + a_bytes;
}

static int valid_plan(const velri_linear_plan *p) {
    return p && p->m && p->k && p->columns && p->kernel && p->packed_count;
}

velri_linear_status velri_linear_pack(const velri_linear_plan *p, const float *w, size_t w_count,
                                      size_t w_stride, float *packed, size_t packed_count) {
    if (!valid_plan(p))
        return VELRI_LINEAR_INVALID;
    size_t w_bytes, packed_bytes;
    velri_linear_status s = span(w, w_count, p->m, p->k, w_stride, &w_bytes);
    if (s)
        return s;
    s = span(packed, packed_count, 1, p->packed_count, p->packed_count, &packed_bytes);
    if (s)
        return s;
    if (overlap(w, w_bytes, packed, packed_bytes) || overlap(p, sizeof *p, packed, packed_bytes))
        return VELRI_LINEAR_INVALID;
    for (size_t c = 0; c < p->m; c += p->columns) {
        const size_t width = p->m - c < p->columns ? p->m - c : p->columns;
        for (size_t i = 0; i < p->k; ++i) {
            for (size_t j = 0; j < width; ++j)
                packed[c * p->k + i * p->columns + j] = w[(c + j) * w_stride + i];
            for (size_t j = width; j < p->columns; ++j)
                packed[c * p->k + i * p->columns + j] = 0.0f;
        }
    }
    return VELRI_LINEAR_OK;
}

velri_linear_status velri_linear_run(const velri_linear_plan *p, const float *packed,
                                     size_t packed_count, const float *x, size_t x_count,
                                     size_t rows, size_t x_stride, const float *bias,
                                     size_t bias_count, float *y, size_t y_count, size_t y_stride) {
    if (!valid_plan(p) || (!bias && bias_count))
        return VELRI_LINEAR_INVALID;
    size_t p_bytes, x_bytes, b_bytes, y_bytes;
    velri_linear_status s =
        span(packed, packed_count, 1, p->packed_count, p->packed_count, &p_bytes);
    if (s)
        return s;
    s = span(x, x_count, rows, p->k, x_stride, &x_bytes);
    if (s)
        return s;
    s = span(bias, bias_count, bias ? 1 : 0, p->m, p->m, &b_bytes);
    if (s)
        return s;
    s = span(y, y_count, rows, p->m, y_stride, &y_bytes);
    if (s)
        return s;
    if (overlap(y, y_bytes, packed, p_bytes) || overlap(y, y_bytes, x, x_bytes) ||
        overlap(y, y_bytes, bias, b_bytes) || overlap(y, y_bytes, p, sizeof *p))
        return VELRI_LINEAR_INVALID;
    if (rows)
        p->kernel(x, packed, bias, y, rows, p->m, p->k, x_stride, y_stride);
    return VELRI_LINEAR_OK;
}
