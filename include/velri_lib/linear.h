#ifndef VELRI_CPU_LINEAR_H
#define VELRI_CPU_LINEAR_H

#include <stddef.h>

/* Experimental CPU module interface; not a stable ABI or the engine API.
 * X[rows, K] * W[M, K]^T + bias[M] -> Y[rows, M].
 * Strides and capacities are in float elements. Natural float alignment is
 * sufficient. Read-only inputs may share storage; output must be disjoint.
 * Packed storage is caller-owned, immutable after preparation and contains
 * FP32 values (including zero padding), never a second FP64 copy.
 *
 * Arithmetic: sequential FP64 sum from -0.0 in increasing K order, then
 * FP64 bias addition (+0.0 without bias), then one FP32 rounding. Requires
 * round-to-nearest/ties-to-even, gradual underflow and disabled FP traps.
 * NaN outputs are canonical 0x7fc00000; FP exception flags are not outputs. Do not compile with fast-math or reassociation.
 */
typedef enum {
    VELRI_LINEAR_OK = 0,
    VELRI_LINEAR_INVALID,
    VELRI_LINEAR_OVERFLOW,
    VELRI_LINEAR_UNSUPPORTED
} velri_linear_status;

typedef enum {
    VELRI_CPU_AUTO = 0,
    VELRI_CPU_BASE,
    VELRI_CPU_AVX2,
    VELRI_CPU_AVX512
} velri_cpu_variant;

typedef void (*velri_linear_kernel)(const float *x, const float *packed, const float *bias,
                                    float *y, size_t rows, size_t m, size_t k, size_t x_stride,
                                    size_t y_stride);

/* Obtain via query; do not modify its fields. No heap allocation or global
 * mutable state. A future worker scheduler can partition complete column
 * tiles and call kernel on validated, disjoint output ranges. */
typedef struct {
    size_t m, k, columns, packed_count;
    velri_cpu_variant variant;
    velri_linear_kernel kernel;
} velri_linear_plan;

int velri_cpu_variant_available(velri_cpu_variant variant);
velri_linear_status velri_linear_query(size_t m, size_t k, velri_cpu_variant variant,
                                       velri_linear_plan *plan);

/* All checks finish before the first write. Padding is initialized to +0.0.
 * W need not remain alive after packing. Plan and packed buffer are shareable
 * between concurrent calls with independent output storage. */
velri_linear_status velri_linear_pack(const velri_linear_plan *plan, const float *w, size_t w_count,
                                      size_t w_stride, float *packed, size_t packed_count);

velri_linear_status velri_linear_run(const velri_linear_plan *plan, const float *packed,
                                     size_t packed_count, const float *x, size_t x_count,
                                     size_t rows, size_t x_stride, const float *bias,
                                     size_t bias_count, float *y, size_t y_count, size_t y_stride);

#endif
