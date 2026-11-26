#ifndef VELRI_BENCH_DECODE_H
#define VELRI_BENCH_DECODE_H

#include "velri_lib/velri_lib.h"
#include <stdio.h>

/* First two cases stream identical projections at different batch sizes.
 * The third reuses a smaller matrix to expose dispatch and compute costs. */
static const struct {
    size_t rows, m, k;
    bool stream;
} decode_shapes[] = {
    {1U, 14336U, 4096U, true},
    {64U, 14336U, 4096U, true},
    {64U, 2048U, 2048U, false},
};

static void decode_csv_header(void) {
    puts("mode,rows,m,k,weight_buffers,ms,rate,unit,ops_per_byte");
}

static size_t linear_logical_bytes(velri_linear_desc d, size_t rows, size_t m, size_t k) {
    size_t bytes = m * k * velri_dtype_bytes(d.w_dtype) + rows * k * velri_dtype_bytes(d.x_dtype)
                   + rows * m * velri_dtype_bytes(d.y_dtype);
    if (d.has_x_scale) bytes += rows * sizeof(float);
    if (d.has_w_scale) bytes += m * sizeof(float);
    return bytes;
}

static void decode_csv_row(const char *mode, velri_linear_desc d, size_t shape,
                           size_t weights, double ms) {
    size_t r = decode_shapes[shape].rows, m = decode_shapes[shape].m, k = decode_shapes[shape].k;
    double ops = 2.0 * (double)r * (double)m * (double)k;
    size_t bytes = linear_logical_bytes(d, r, m, k);
    printf("%s,%zu,%zu,%zu,%zu,%.8f,%.8f,%s,%.4f\n", mode, r, m, k, weights, ms,
           ops / (ms * 1e9), d.w_dtype == VELRI_DTYPE_I8 ? "TOPS" : "TFLOPS", ops / (double)bytes);
}

#endif
