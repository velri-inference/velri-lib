#include "w8a8.h"
#include "w8a8_layout.h"
#include <string.h>

enum { KC = VELRI_W8A8_KC };
static_assert(sizeof(float) == sizeof(uint32_t));
static_assert(VELRI_W8A8_NR % VELRI_W8A8_SIGNED_PANEL_WIDTH == 0);

#if defined(__GNUC__) || defined(__clang__)
#define VELRI_W8A8_NOINLINE [[gnu::noinline]]
#else
#define VELRI_W8A8_NOINLINE
#endif

#if defined(VELRI_W8A8_USE_VNNI)
enum { NR = VELRI_W8A8_NR, ROW_BLOCK = 256 };
#include "w8a8_vnni.h"

static void pack_weights(const int8_t *w, size_t ws, size_t channels, size_t depth, uint8_t *dst) {
    for (size_t c = 0; c < channels; c += NR)
        pack_panel(w + c * ws, ws, dst + c * depth, depth);
}
#else
/* The signed 32-channel panels are also consumed by the existing Slang decode kernel. */
enum { NR = VELRI_W8A8_SIGNED_PANEL_WIDTH, ROW_BLOCK = 128 };
#include "w8a8_pairs.h"
#endif

static inline float scaled_output(uint32_t sum, uint32_t correction, const float *xscale,
                                  const float *wscale, const float *bias, size_t channel) {
    float scale = wscale[channel] * *xscale;
    float value = (float)(int32_t)(sum - correction) * scale;
    return bias ? value + bias[channel] : value;
}

#if defined(VELRI_W8A8_USE_VNNI)
/* Bindings may physically overlap. Commit each direct tile before reading the next one. */
static void direct_row(const int8_t *x, const int8_t *w, size_t ws, bool packed, float *y,
                       size_t c0, size_t c1, size_t depth, uint32_t correction,
                       const float *xscale, const float *wscale, const float *bias) {
    for (size_t c = c0; c < c1;) {
        size_t count = c1 - c >= 8 ? 8 : c1 - c >= 4 ? 4 : 1;
        uint32_t sums[8] = {};
        for (size_t j = 0; j < count; ++j) {
            size_t ch = c + j;
            if (packed) {
                const uint8_t *column =
                    (const uint8_t *)w + (ch / NR) * NR * depth + 4 * (ch % NR);
                for (size_t k = 0; k < depth; k += 4) {
                    const uint8_t *b = column + k * NR;
                    sums[j] += (uint32_t)(x[k] * b[0] + x[k + 1] * b[1] + x[k + 2] * b[2] +
                                          x[k + 3] * b[3]);
                }
            } else {
                const int8_t *column = w + ch * ws;
                for (size_t k = 0; k < depth; ++k)
                    sums[j] += (uint32_t)(x[k] * column[k]);
            }
        }
        uint32_t corr = packed ? correction : 0U;
        for (size_t j = 0; j < count; ++j) {
            y[c + j] = scaled_output(sums[j], corr, xscale, wscale, bias, c + j);
        }
        c += count;
    }
}
#endif

VELRI_W8A8_NOINLINE static void run_tiled(const int8_t *x, size_t xs, const int8_t *w, size_t ws,
                                          bool packed, const float *bias, const float *xscale,
                                          const float *wscale, float *y, size_t ys, size_t r0,
                                          size_t r1, size_t c0, size_t c1, size_t depth) {
#if defined(VELRI_W8A8_USE_VNNI)
    alignas(64) uint8_t scratch[2 * KC * NR];
    uint32_t corrections[ROW_BLOCK];
#endif
    alignas(64) uint32_t sums[ROW_BLOCK * NR];
    for (size_t rb = r0; rb < r1; rb += ROW_BLOCK) {
        size_t height = r1 - rb < ROW_BLOCK ? r1 - rb : ROW_BLOCK;
        size_t block_depth = KC;
#if defined(VELRI_W8A8_USE_VNNI)
        for (size_t r = 0; r < height; ++r)
            corrections[r] = row_correction(x + (rb + r) * xs, depth);
        /* A prepared panel needs no repacking scratch. Small row blocks stream the
           full panel from L2 without spilling partial sums; large blocks reuse L1D. */
        if (packed) {
            block_depth = (height <= MR) ? depth : 2U * KC;
        } else if (height <= 128U) {
            block_depth = 2U * KC;
        }
#endif
        for (size_t c = c0 - c0 % NR; c < c1; c += NR) {
            size_t begin = c0 > c ? c0 - c : 0;
            size_t end = c1 - c < NR ? c1 - c : NR;
#if defined(VELRI_W8A8_USE_VNNI)
            panel_output output = {
                .dst = y + rb * ys + c,
                .stride = ys,
                .row_scale = xscale + rb,
                .column_scale = wscale + c,
                .bias = bias ? bias + c : nullptr,
                .correction = corrections,
                .lanes = (UINT64_MAX << begin) & (UINT64_MAX >> (NR - end)),
            };
#endif
            for (size_t k = 0; k < depth; k += block_depth) {
                size_t n = depth - k < block_depth ? depth - k : block_depth;
#if defined(VELRI_W8A8_USE_VNNI)
                const uint8_t *block;
                if (packed) {
                    block = (const uint8_t *)w + c * depth + k * NR;
                } else {
                    pack_panel(w + c * ws + k, ws, scratch, n);
                    block = scratch;
                }
                const panel_output *last = k + n == depth ? &output : nullptr;
                multiply_panel(x + rb * xs + k, xs, block, n, sums, NR, height, k == 0, last);
#else
                const int8_t *block = packed ? w + c * depth + k * NR : w + c * ws + k;
                multiply_panel(x + rb * xs + k, xs, block, packed ? 0 : ws, n, sums, NR, height,
                               k == 0);
#endif
            }
#if !defined(VELRI_W8A8_USE_VNNI)
            for (size_t r = 0; r < height; ++r) {
                float *out = y + (rb + r) * ys + c;
                for (size_t j = begin; j < end; ++j)
                    out[j] = scaled_output(sums[r * NR + j], 0U, xscale + rb + r, wscale, bias,
                                           c + j);
            }
#endif
        }
    }
}

static void run(const int8_t *x, size_t xs, const int8_t *w, size_t ws, bool packed,
                const float *bias, const float *xscale, const float *wscale, float *y, size_t ys,
                size_t r0, size_t r1, size_t c0, size_t c1, size_t depth, bool direct) {
    if (r0 >= r1 || c0 >= c1) return;
#if defined(VELRI_W8A8_USE_VNNI)
    if (direct) {
        for (size_t r = r0; r < r1; ++r) {
            uint32_t corr = packed ? row_correction(x + r * xs, depth) : 0U;
            direct_row(x + r * xs, w, ws, packed, y + r * ys, c0, c1, depth, corr, xscale + r,
                       wscale, bias);
        }
        return;
    }
#else
    (void)direct;
#endif
    run_tiled(x, xs, w, ws, packed, bias, xscale, wscale, y, ys, r0, r1, c0, c1, depth);
}

const velri_w8a8_kernel VELRI_W8A8_VARIANT = {
#if defined(VELRI_W8A8_USE_VNNI)
    .pack = pack_weights,
#endif
    .run = run,
};
