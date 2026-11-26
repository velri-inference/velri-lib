/* Six activation rows by 64 output channels, using unsigned weights and signed activations. */
#include <immintrin.h>

enum { MR = 6 };
static_assert(MR <= 8, "GCC unroll bound covers MR rows");
static_assert(NR == 64, "Each bit of the output mask selects one panel channel");
static_assert(KC % 64 == 0, "KC must be a multiple of 64 for 16-byte K-unrolled VNNI tiles");

static uint32_t row_correction(const int8_t *x, size_t depth) {
    const __m512i c128 = _mm512_set1_epi8((char)128);
    __m512i s0 = _mm512_setzero_si512();
    __m512i s1 = _mm512_setzero_si512();
    __m512i s2 = _mm512_setzero_si512();
    __m512i s3 = _mm512_setzero_si512();
    size_t k = 0;
    for (; k + 256 <= depth; k += 256) {
        s0 = _mm512_dpbusd_epi32(s0, c128, _mm512_loadu_si512(x + k + 0));
        s1 = _mm512_dpbusd_epi32(s1, c128, _mm512_loadu_si512(x + k + 64));
        s2 = _mm512_dpbusd_epi32(s2, c128, _mm512_loadu_si512(x + k + 128));
        s3 = _mm512_dpbusd_epi32(s3, c128, _mm512_loadu_si512(x + k + 192));
    }
    for (; k < depth; k += 64)
        s0 = _mm512_dpbusd_epi32(s0, c128, _mm512_loadu_si512(x + k));
    s0 = _mm512_add_epi32(_mm512_add_epi32(s0, s1), _mm512_add_epi32(s2, s3));
    return (uint32_t)_mm512_reduce_add_epi32(s0);
}

static inline void transpose16(__m512i rows[16], uint8_t *dst, size_t stride, __m512i flip) {
    __m512i a[16], b[16];
    for (int i = 0; i < 8; ++i) {
        a[2 * i] = _mm512_unpacklo_epi32(rows[2 * i], rows[2 * i + 1]);
        a[2 * i + 1] = _mm512_unpackhi_epi32(rows[2 * i], rows[2 * i + 1]);
    }
    for (int i = 0; i < 4; ++i) {
        b[4 * i] = _mm512_unpacklo_epi64(a[4 * i], a[4 * i + 2]);
        b[4 * i + 1] = _mm512_unpackhi_epi64(a[4 * i], a[4 * i + 2]);
        b[4 * i + 2] = _mm512_unpacklo_epi64(a[4 * i + 1], a[4 * i + 3]);
        b[4 * i + 3] = _mm512_unpackhi_epi64(a[4 * i + 1], a[4 * i + 3]);
    }
    for (int i = 0; i < 4; ++i) {
        __m512i lo = _mm512_shuffle_i32x4(b[i], b[4 + i], 0x44);
        __m512i hi = _mm512_shuffle_i32x4(b[8 + i], b[12 + i], 0x44);
        _mm512_storeu_si512(dst + (i * stride),
                            _mm512_xor_si512(_mm512_shuffle_i32x4(lo, hi, 0x88), flip));
        _mm512_storeu_si512(dst + ((4 + i) * stride),
                            _mm512_xor_si512(_mm512_shuffle_i32x4(lo, hi, 0xdd), flip));
        lo = _mm512_shuffle_i32x4(b[i], b[4 + i], 0xee);
        hi = _mm512_shuffle_i32x4(b[8 + i], b[12 + i], 0xee);
        _mm512_storeu_si512(dst + ((8 + i) * stride),
                            _mm512_xor_si512(_mm512_shuffle_i32x4(lo, hi, 0x88), flip));
        _mm512_storeu_si512(dst + ((12 + i) * stride),
                            _mm512_xor_si512(_mm512_shuffle_i32x4(lo, hi, 0xdd), flip));
    }
}

static void pack_panel(const int8_t *w, size_t ws, uint8_t *dst, size_t n) {
    const __m512i flip = _mm512_set1_epi8((char)128);
    for (size_t j = 0; j < NR; j += 16) {
        for (size_t q = 0; q < n; q += 64) {
            __m512i v[16];
            for (size_t i = 0; i < 16; ++i)
                v[i] = _mm512_loadu_si512(w + (j + i) * ws + q);
            transpose16(v, dst + q * NR + 4 * j, 4 * NR, flip);
        }
    }
}

typedef struct {
    float *dst;
    size_t stride;
    const float *row_scale;
    const float *column_scale;
    const float *bias;
    const uint32_t *correction;
    uint64_t lanes;
} panel_output;

/* Subtract the zero-point correction modulo 2^32 before converting to F32.
   Neighboring workers may own channels in the same panel. */
static inline void store_output_row(__m512i a0, __m512i a1, __m512i a2, __m512i a3,
                                    const panel_output *out, size_t row) {
    __m512i acc[4] = {a0, a1, a2, a3};
    __m512i correction = _mm512_set1_epi32((int)(int32_t)out->correction[row]);
    __m512 row_scale = _mm512_set1_ps(out->row_scale[row]);
    float *dst_row = out->dst + row * out->stride;
    const bool full = out->lanes == UINT64_MAX;
    for (size_t j = 0; j < 4; ++j) {
        const __mmask16 mask = (__mmask16)(out->lanes >> (j * 16));
        if (!full && !mask) continue;
        __m512 value = _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[j], correction));
        __m512 cscale = full ? _mm512_loadu_ps(out->column_scale + j * 16)
                             : _mm512_maskz_loadu_ps(mask, out->column_scale + j * 16);
        value = _mm512_mul_ps(value, _mm512_mul_ps(cscale, row_scale));
        if (out->bias) {
            __m512 b = full ? _mm512_loadu_ps(out->bias + j * 16)
                            : _mm512_maskz_loadu_ps(mask, out->bias + j * 16);
            value = _mm512_add_ps(value, b);
        }
        if (full) {
            _mm512_storeu_ps(dst_row + j * 16, value);
        } else {
            _mm512_mask_storeu_ps(dst_row + j * 16, mask, value);
        }
    }
}

#if defined(__clang__)
#define VELRI_VNNI_UNROLL_FULL _Pragma("clang loop unroll(full)")
#elif defined(__GNUC__)
#define VELRI_VNNI_UNROLL_FULL _Pragma("GCC unroll 8")
#else
#define VELRI_VNNI_UNROLL_FULL
#endif

/* Caller contract (`velri_w8a8_panel_shape`): `n` is a multiple of 64 (hence of 16). */
#define DEFINE_VNNI_TILE(NAME, ROWS)                                                               \
    [[gnu::noinline]] static void NAME(const int8_t *x, size_t xs, const uint8_t *panel, size_t n, \
                                       uint32_t *y, size_t ys, bool first,                         \
                                       const panel_output *out, size_t row) {                      \
        __m512i acc[ROWS][NR / 16];                                                                \
        if (first) {                                                                               \
            VELRI_VNNI_UNROLL_FULL                                                                 \
            for (size_t r = 0; r < (ROWS); ++r)                                                    \
                for (size_t j = 0; j < NR / 16; ++j) acc[r][j] = _mm512_setzero_si512();           \
        } else {                                                                                   \
            VELRI_VNNI_UNROLL_FULL                                                                 \
            for (size_t r = 0; r < (ROWS); ++r)                                                    \
                for (size_t j = 0; j < NR / 16; ++j)                                               \
                    acc[r][j] = _mm512_loadu_si512(y + r * ys + j * 16);                           \
        }                                                                                          \
        const int8_t *xp[ROWS];                                                                    \
        VELRI_VNNI_UNROLL_FULL                                                                     \
        for (size_t r = 0; r < (ROWS); ++r) xp[r] = x + r * xs;                                    \
        for (size_t k = 0; k < n; k += 16) {                                                       \
            VELRI_VNNI_UNROLL_FULL                                                                 \
            for (size_t u = 0; u < 16; u += 4) {                                                   \
                const uint8_t *p = panel + (k + u) * NR;                                           \
                __m512i w0 = _mm512_loadu_si512(p + 0);                                            \
                __m512i w1 = _mm512_loadu_si512(p + 64);                                           \
                __m512i w2 = _mm512_loadu_si512(p + 128);                                          \
                __m512i w3 = _mm512_loadu_si512(p + 192);                                          \
                VELRI_VNNI_UNROLL_FULL                                                             \
                for (size_t r = 0; r < (ROWS); ++r) {                                              \
                    int32_t word;                                                                  \
                    memcpy(&word, xp[r] + k + u, 4);                                               \
                    __m512i a = _mm512_set1_epi32(word);                                           \
                    acc[r][0] = _mm512_dpbusd_epi32(acc[r][0], w0, a);                             \
                    acc[r][1] = _mm512_dpbusd_epi32(acc[r][1], w1, a);                             \
                    acc[r][2] = _mm512_dpbusd_epi32(acc[r][2], w2, a);                             \
                    acc[r][3] = _mm512_dpbusd_epi32(acc[r][3], w3, a);                             \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
        if (out) {                                                                                 \
            VELRI_VNNI_UNROLL_FULL                                                                 \
            for (size_t r = 0; r < (ROWS); ++r)                                                    \
                store_output_row(acc[r][0], acc[r][1], acc[r][2], acc[r][3], out, row + r);        \
        } else {                                                                                   \
            VELRI_VNNI_UNROLL_FULL                                                                 \
            for (size_t r = 0; r < (ROWS); ++r)                                                    \
                for (size_t j = 0; j < NR / 16; ++j)                                               \
                    _mm512_storeu_si512(y + r * ys + j * 16, acc[r][j]);                           \
        }                                                                                          \
    }

DEFINE_VNNI_TILE(tile_6x64, 6)
DEFINE_VNNI_TILE(tile_4x64, 4)
DEFINE_VNNI_TILE(tile_2x64, 2)
DEFINE_VNNI_TILE(tile_1x64, 1)

#undef DEFINE_VNNI_TILE
#undef VELRI_VNNI_UNROLL_FULL

static inline void multiply_panel(const int8_t *x, size_t xs, const uint8_t *panel, size_t depth,
                                  uint32_t *y, size_t ys, size_t rows, bool first,
                                  const panel_output *out) {
    const size_t rem = rows % MR;
    const size_t m6 = (rem == 2U && rows >= 8U) ? (rows - 8U) : (rows - rem);
    size_t r = 0;
    for (; r < m6; r += MR)
        tile_6x64(x + r * xs, xs, panel, depth, y + r * ys, ys, first, out, r);
    for (; rows - r >= 4U; r += 4U)
        tile_4x64(x + r * xs, xs, panel, depth, y + r * ys, ys, first, out, r);
    for (; rows - r >= 2U; r += 2U)
        tile_2x64(x + r * xs, xs, panel, depth, y + r * ys, ys, first, out, r);
    if (r < rows)
        tile_1x64(x + r * xs, xs, panel, depth, y + r * ys, ys, first, out, r);
}
