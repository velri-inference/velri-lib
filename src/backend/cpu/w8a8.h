#ifndef VELRI_CPU_W8A8_H
#define VELRI_CPU_W8A8_H
#include <stddef.h>
#include <stdint.h>

/* Temporary CPU W8A8 panel path. */
enum { VELRI_W8A8_NR = 64, VELRI_W8A8_KC = 256 };

static inline bool velri_w8a8_panel_shape(size_t channels, size_t depth) {
    return channels % VELRI_W8A8_NR == 0 && depth >= VELRI_W8A8_KC && depth % 64 == 0;
}

typedef struct {
    /* A null packer reuses the signed Slang panel format. */
    void (*pack)(const int8_t *w, size_t ws, size_t channels, size_t depth, uint8_t *dst);
    void (*run)(const int8_t *x, size_t xs, const int8_t *w, size_t ws, bool packed,
                const float *bias, const float *xscale, const float *wscale, float *y, size_t ys,
                size_t r0, size_t r1, size_t c0, size_t c1, size_t depth, bool direct);
} velri_w8a8_kernel;

extern const velri_w8a8_kernel velri_w8a8_portable;
#if VELRI_X86_VARIANTS
extern const velri_w8a8_kernel velri_w8a8_avx2;
extern const velri_w8a8_kernel velri_w8a8_vnni;
#endif
#endif
