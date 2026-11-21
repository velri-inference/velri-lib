#ifndef VELRI_LIB_TYPES_H
#define VELRI_LIB_TYPES_H

#include <stddef.h>
#include <stdint.h>

/*
 * Portable C23 primitive types, enums, and CPU ISA queries for Velri-lib.
 *
 * Explicit 16-bit floating-point storage representations:
 *   - velri_fp16: IEEE-754 binary16 (1 sign, 5 exponent, 10 fraction bits)
 *   - velri_bf16: Brain Float 16    (1 sign, 8 exponent,  7 fraction bits)
 */
typedef uint16_t velri_fp16;
typedef uint16_t velri_bf16;

typedef enum {
    VELRI_OK = 0,
    VELRI_ERR_INVALID = 1,
    VELRI_ERR_OVERFLOW = 2,
    VELRI_ERR_UNSUPPORTED = 3,
    VELRI_ERR_NOMEM = 4,
    VELRI_ERR_CANCELLED = 5,
    VELRI_ERR_EXEC = 6,
    VELRI_ERR_BUSY = 7,
    VELRI_ERR_NOT_FOUND = 8
} velri_status;

typedef enum {
    VELRI_CPU_AUTO = 0,
    VELRI_CPU_BASE = 1,
    VELRI_CPU_AVX2 = 2,
    VELRI_CPU_AVX512 = 3
} velri_cpu_variant;

typedef enum {
    VELRI_DTYPE_F32 = 0,
    VELRI_DTYPE_F64 = 1,
    VELRI_DTYPE_F16 = 2,
    VELRI_DTYPE_BF16 = 3,
    VELRI_DTYPE_I8 = 4,
    VELRI_DTYPE_I32 = 5
} velri_dtype;

typedef enum {
    VELRI_ROPE_HALF = 0,
    VELRI_ROPE_INTERLEAVED = 1
} velri_rope_mode;

typedef enum {
    VELRI_ACT_SILU = 0,
    VELRI_ACT_GELU_TANH = 1,
    VELRI_ACT_RELU2 = 2
} velri_act_kind;

/*
 * Typed Linear operation descriptor:
 *   Y[rows, M] = X[rows, K] * W[M, K]^T + bias[M] (or scaled W8A8 form).
 */
typedef struct {
    velri_dtype x_dtype;
    velri_dtype w_dtype;
    velri_dtype y_dtype;
    bool has_x_scale; /* Per-row float scale[rows] (for W8A8 -> F32) */
    bool has_w_scale; /* Per-output float scale[M] (for W8A8 -> F32) */
} velri_linear_desc;

static inline velri_linear_desc velri_desc_f32(void) {
    return (velri_linear_desc){VELRI_DTYPE_F32, VELRI_DTYPE_F32, VELRI_DTYPE_F32, false, false};
}
static inline velri_linear_desc velri_desc_f64(void) {
    return (velri_linear_desc){VELRI_DTYPE_F64, VELRI_DTYPE_F64, VELRI_DTYPE_F64, false, false};
}
static inline velri_linear_desc velri_desc_f16(velri_dtype y_dtype) {
    return (velri_linear_desc){VELRI_DTYPE_F16, VELRI_DTYPE_F16, y_dtype, false, false};
}
static inline velri_linear_desc velri_desc_bf16(velri_dtype y_dtype) {
    return (velri_linear_desc){VELRI_DTYPE_BF16, VELRI_DTYPE_BF16, y_dtype, false, false};
}
static inline velri_linear_desc velri_desc_f32x_bf16w(void) {
    return (velri_linear_desc){VELRI_DTYPE_F32, VELRI_DTYPE_BF16, VELRI_DTYPE_F32, false, false};
}
static inline velri_linear_desc velri_desc_w8a8_f32(void) {
    return (velri_linear_desc){VELRI_DTYPE_I8, VELRI_DTYPE_I8, VELRI_DTYPE_F32, true, true};
}

typedef struct {
    int hardware_threads;
    bool has_avx2;
    bool has_fma;
    bool has_f16c;
    bool has_avx512f;
    bool has_avx512bw;
    bool has_avx512dq;
    bool has_avx512vl;
    bool has_avx512_bf16;
    bool has_avx512_vnni;
} velri_cpu_info;

/* Query element storage size in bytes (returns 0 for unknown enum values). */
size_t velri_dtype_bytes(velri_dtype dtype);

/* Query required natural alignment in bytes for a dtype. */
size_t velri_dtype_align(velri_dtype dtype);

/* Check whether BOTH the CPU and OS extended-state support `variant`. */
bool velri_cpu_variant_supported(velri_cpu_variant variant);

/* Resolve VELRI_CPU_AUTO to the widest supported variant on this host. */
velri_cpu_variant velri_cpu_resolve_variant(velri_cpu_variant variant);

/* Return immutable detected CPU ISA capability summary. */
const velri_cpu_info *velri_get_cpu_info(void);

#endif /* VELRI_LIB_TYPES_H */
