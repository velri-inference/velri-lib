/* Widen two weights per channel, then reuse the block across activation rows. */
#if defined(__AVX2__) || defined(__SSE2__) || defined(_M_X64)
#include <immintrin.h>
#if defined(__AVX2__)
enum { LANES = 8 };
typedef __m256i sums_vec;
static inline sums_vec add_pairs(sums_vec sum, const int16_t *w, const int16_t *a) {
    int32_t pair;
    memcpy(&pair, a, sizeof(pair));
    return _mm256_add_epi32(
        sum, _mm256_madd_epi16(_mm256_loadu_si256((const __m256i_u *)w), _mm256_set1_epi32(pair)));
}
#else
enum { LANES = 4 };
typedef __m128i sums_vec;
static inline sums_vec add_pairs(sums_vec sum, const int16_t *w, const int16_t *a) {
    int32_t pair;
    memcpy(&pair, a, sizeof(pair));
    return _mm_add_epi32(
        sum, _mm_madd_epi16(_mm_loadu_si128((const __m128i_u *)w), _mm_set1_epi32(pair)));
}
#endif
#else
enum { LANES = 4 };
typedef struct {
    uint32_t col[LANES];
} sums_vec;
static inline sums_vec add_pairs(sums_vec sum, const int16_t *w, const int16_t *a) {
    for (size_t j = 0; j < LANES; ++j)
        sum.col[j] += (uint32_t)(a[0] * w[2 * j] + a[1] * w[2 * j + 1]);
    return sum;
}
#endif

enum { GROUPS = 4, COLS = LANES * GROUPS, ROWS = 2 };
static_assert(NR % COLS == 0);
static_assert(sizeof(sums_vec) == LANES * sizeof(uint32_t));
static inline void pairs_tile(const int8_t *x, size_t xs, const int16_t *weights, size_t depth,
                              uint32_t *y, size_t ys, size_t rows, bool first) {
    /* Convert each activation pair once so the dot loop can broadcast a single word. */
    int16_t activations[ROWS][KC];
    for (size_t r = 0; r < rows; ++r)
        for (size_t k = 0; k < depth; ++k)
            activations[r][k] = x[r * xs + k];
    sums_vec sums[ROWS][GROUPS] = {};
    if (!first) {
        for (size_t r = 0; r < rows; ++r)
            memcpy(sums[r], y + r * ys, sizeof(sums[r]));
    }
    for (size_t k = 0; k < depth; k += 2) {
        for (size_t r = 0; r < rows; ++r) {
            for (size_t j = 0; j < GROUPS; ++j)
                sums[r][j] = add_pairs(sums[r][j], weights + k * COLS + j * 2 * LANES,
                                       activations[r] + k);
        }
    }
    for (size_t r = 0; r < rows; ++r)
        memcpy(y + r * ys, sums[r], sizeof(sums[r]));
}

#if defined(__AVX2__)
/* Each input word is a pair of signed bytes at adjacent depths. Transpose eight
   channels at once and widen directly into the microkernel's paired layout. */
static_assert(COLS % 8 == 0 && KC % 16 == 0);
static void widen_raw_pairs(const int8_t *w, size_t ws, size_t depth, int16_t *dst) {
    for (size_t c = 0; c < COLS; c += 8) {
        for (size_t k = 0; k < depth; k += 16) {
            __m128i a[8], b[8];
            for (int j = 0; j < 4; ++j) {
                __m128i u = _mm_loadu_si128((const __m128i_u *)(w + (c + 2 * j) * ws + k));
                __m128i v = _mm_loadu_si128((const __m128i_u *)(w + (c + 2 * j + 1) * ws + k));
                a[2 * j] = _mm_unpacklo_epi16(u, v);
                a[2 * j + 1] = _mm_unpackhi_epi16(u, v);
            }
            for (int j = 0; j < 2; ++j) {
                b[4 * j] = _mm_unpacklo_epi32(a[4 * j], a[4 * j + 2]);
                b[4 * j + 1] = _mm_unpackhi_epi32(a[4 * j], a[4 * j + 2]);
                b[4 * j + 2] = _mm_unpacklo_epi32(a[4 * j + 1], a[4 * j + 3]);
                b[4 * j + 3] = _mm_unpackhi_epi32(a[4 * j + 1], a[4 * j + 3]);
            }
            for (int j = 0; j < 4; ++j) {
                int16_t *pair = dst + (k + 4 * j) * COLS + 2 * c;
                _mm256_storeu_si256((__m256i_u *)pair,
                                    _mm256_cvtepi8_epi16(_mm_unpacklo_epi64(b[j], b[4 + j])));
                _mm256_storeu_si256((__m256i_u *)(pair + 2 * COLS),
                                    _mm256_cvtepi8_epi16(_mm_unpackhi_epi64(b[j], b[4 + j])));
            }
        }
    }
}
#else
static void widen_raw_pairs(const int8_t *w, size_t ws, size_t depth, int16_t *dst) {
    for (size_t k = 0; k < depth; k += 2)
        for (size_t j = 0; j < COLS; ++j) {
            dst[k * COLS + 2 * j] = w[j * ws + k];
            dst[k * COLS + 2 * j + 1] = w[j * ws + k + 1];
        }
}
#endif

static void multiply_panel(const int8_t *x, size_t xs, const int8_t *panel, size_t raw_stride,
                           size_t depth, uint32_t *y, size_t ys, size_t rows, bool first) {
    alignas(64) int16_t weights[KC * COLS];
    for (size_t c = 0; c < NR; c += COLS) {
        if (raw_stride) {
            widen_raw_pairs(panel + c * raw_stride, raw_stride, depth, weights);
        } else {
            for (size_t k = 0; k < depth; k += 2)
                for (size_t j = 0; j < COLS; ++j) {
                    weights[k * COLS + 2 * j] = panel[k * NR + c + j];
                    weights[k * COLS + 2 * j + 1] = panel[(k + 1) * NR + c + j];
                }
        }
        size_t r = 0;
        for (; rows - r >= ROWS; r += ROWS)
            pairs_tile(x + r * xs, xs, weights, depth, y + r * ys + c, ys, ROWS, first);
        if (r < rows)
            pairs_tile(x + r * xs, xs, weights, depth, y + r * ys + c, ys, 1, first);
    }
}
