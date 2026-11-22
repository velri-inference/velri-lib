#ifndef VELRI_TEST_REFERENCE_H
#define VELRI_TEST_REFERENCE_H

#include "velri_lib/velri_lib.h"
#include <string.h>

/* Portable software IEEE-754 binary16 / bfloat16 converters (test & reference helpers). */
static inline float velri_bf16_to_f32_scalar(velri_bf16 h) {
    uint32_t u = ((uint32_t)h) << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static inline velri_bf16 velri_f32_to_bf16_scalar(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    if ((u & 0x7FFFFFFFU) > 0x7F800000U) {
        return (velri_bf16)((u >> 16) | 0x0040U);
    }
    uint32_t lsb = (u >> 16) & 1U;
    uint32_t bias = 0x7FFFU + lsb;
    return (velri_bf16)((u + bias) >> 16);
}

static inline float velri_f16_to_f32_scalar(velri_fp16 h) {
    uint32_t sign = ((uint32_t)h & 0x8000U) << 16;
    uint32_t exp = (h >> 10) & 0x1FU;
    uint32_t mant = h & 0x03FFU;
    uint32_t out;
    if (exp == 0U) {
        if (mant == 0U) {
            out = sign;
        } else {
            uint32_t e = 127U - 14U;
            while ((mant & 0x0400U) == 0U) {
                mant <<= 1;
                e--;
            }
            mant &= 0x03FFU;
            out = sign | (e << 23) | (mant << 13);
        }
    } else if (exp == 31U) {
        out = sign | 0x7F800000U | (mant ? ((mant << 13) | 0x00400000U) : 0U);
    } else {
        out = sign | ((exp + (127U - 15U)) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &out, sizeof(f));
    return f;
}

static inline velri_fp16 velri_f32_to_f16_scalar(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    uint32_t sign = (u >> 16) & 0x8000U;
    uint32_t abs_u = u & 0x7FFFFFFFU;
    if (abs_u > 0x7F800000U) {
        return (velri_fp16)(sign | 0x7E00U | ((u >> 13) & 0x03FFU));
    }
    if (abs_u >= 0x47800000U) {
        return (velri_fp16)(sign | 0x7C00U);
    }
    if (abs_u < 0x38800000U) {
        if (abs_u < 0x33000000U) {
            return (velri_fp16)sign;
        }
        uint32_t exp = abs_u >> 23;
        uint32_t mant = (abs_u & 0x007FFFFFU) | 0x00800000U;
        uint32_t shift = 113U - exp + 13U;
        uint32_t half_bit = 1U << (shift - 1U);
        uint32_t mask = (1U << shift) - 1U;
        uint32_t rem = mant & mask;
        uint32_t res = mant >> shift;
        if (rem > half_bit || (rem == half_bit && (res & 1U))) {
            res++;
        }
        return (velri_fp16)(sign | res);
    }
    uint32_t exp = (abs_u >> 23) - 112U;
    uint32_t mant = abs_u & 0x007FFFFFU;
    uint32_t rem = mant & 0x1FFFU;
    uint32_t res = (exp << 10) | (mant >> 13);
    if (rem > 0x1000U || (rem == 0x1000U && (res & 1U))) {
        res++;
    }
    return (velri_fp16)(sign | res);
}

/* Deterministic PRNG & shared tensor fill/diff utilities */
uint32_t velri_test_rng(uint64_t *state);
float velri_test_rand_f32(uint64_t *state, float lo, float hi);
void velri_test_fill_dtype(velri_dtype dt, void *buf, size_t n, uint64_t *rng, float scale);
double velri_test_load_f64(velri_dtype dt, const void *p, size_t idx);
double velri_test_max_abs_diff_2d(velri_dtype dt, const void *a, const void *b, size_t rows,
                                  size_t cols, size_t stride);

/* Pure FP64 / int64 reference implementations */
void ref_linear_run(const velri_linear_desc *desc, size_t rows, size_t m, size_t k, const void *x,
                    size_t x_stride, const float *x_scale, const void *w, size_t w_stride,
                    const float *w_scale, const void *bias, void *y, size_t y_stride);

void ref_rmsnorm(velri_dtype io_dt, velri_dtype w_dt, size_t rows, size_t dim, float eps,
                 bool unit_offset, const void *x, size_t xs, const void *w, void *y, size_t ys);

void ref_fused_add_rmsnorm_f32(size_t rows, size_t dim, float eps, bool unit_offset, float *res,
                               size_t rs, const float *x, size_t xs, const float *w, float *y,
                               size_t ys);

void ref_layernorm_f32(size_t rows, size_t dim, float eps, const float *x, size_t xs,
                       const float *w, const float *b, float *y, size_t ys);

void ref_rope(velri_dtype dt, size_t n_tokens, size_t n_heads, size_t head_dim, size_t rotary_dim,
              size_t pos_offset, const void *x, size_t tsx, size_t hsx, const float *cos_t,
              const float *sin_t, void *y, size_t tsy, size_t hsy, velri_rope_mode mode);

void ref_activation_f32(velri_act_kind kind, size_t count, const float *x, float *y);

void ref_gated_mlp_f32(velri_act_kind kind, size_t count, float clamp_limit, const float *gate,
                       const float *up, float *y);

void ref_softmax_f32(size_t rows, size_t cols, const float *x, size_t xs, float *y, size_t ys);

void ref_fwht_f32(size_t rows, size_t cols, size_t block_size, const float *x, size_t xs,
                  velri_dtype suh_dt, const void *suh, float *y, size_t ys);

void ref_vec_axpby_f32(size_t count, float alpha, const float *a, float beta, const float *b,
                       float *y);

void ref_vec_mul_f32(size_t count, const float *a, const float *b, float *y);

void ref_quantize_row_f32_i8(size_t rows, size_t k, const float *x, size_t x_stride,
                             int8_t *x_i8, size_t xi8_stride, float *scale_out);

void velri_ref_embedding_f32(size_t vocab, size_t dim, size_t active_t, size_t pos_offset,
                             const float *table, size_t table_stride, const int32_t *ids,
                             float *out, size_t out_stride);

void velri_ref_copy_last_row_f32(size_t active_t, size_t dim, const float *x, size_t x_stride,
                                 float *y);

void velri_ref_attention_ex_f32(size_t active_t, size_t n_heads, size_t n_kv_heads, size_t d_k,
                                size_t d_v, size_t kv_pos, size_t max_ctx, float scale,
                                size_t sliding_window, bool causal, const float *q, size_t q_stride,
                                const float *k_in, size_t k_stride, const float *v_in,
                                size_t v_stride, float *k_cache, size_t k_head_stride,
                                size_t k_row_stride, float *v_cache, size_t v_head_stride,
                                size_t v_row_stride, float *out, size_t o_stride);

void velri_ref_attention_f32(size_t active_t, size_t n_heads, size_t n_kv_heads, size_t head_dim,
                             size_t kv_pos, size_t max_ctx, float scale, size_t sliding_window,
                             bool causal, const float *q, size_t q_stride, const float *k_in,
                             size_t k_stride, const float *v_in, size_t v_stride, float *k_cache,
                             float *v_cache, float *out, size_t o_stride);

void velri_ref_batched_linear_f32(size_t rows, size_t n_heads, size_t n_groups, size_t in_dim,
                                  size_t out_dim, bool weight_transposed, const float *x,
                                  size_t x_stride, const float *w, size_t w_stride,
                                  const float *bias, float *y, size_t y_stride);

#endif /* VELRI_TEST_REFERENCE_H */
