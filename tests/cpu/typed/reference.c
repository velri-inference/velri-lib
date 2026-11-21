#include "reference.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>

uint32_t velri_test_rng(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return (uint32_t)(x >> 16);
}

float velri_test_rand_f32(uint64_t *state, float lo, float hi) {
    uint32_t r = velri_test_rng(state) & 0x00FFFFFFU;
    float u = (float)r / (float)0x01000000U;
    return lo + u * (hi - lo);
}

double velri_test_load_f64(velri_dtype dt, const void *p, size_t idx) {
    switch (dt) {
    case VELRI_DTYPE_F32:
        return (double)((const float *)p)[idx];
    case VELRI_DTYPE_F64:
        return ((const double *)p)[idx];
    case VELRI_DTYPE_F16:
        return (double)velri_f16_to_f32_scalar(((const velri_fp16 *)p)[idx]);
    case VELRI_DTYPE_BF16:
        return (double)velri_bf16_to_f32_scalar(((const velri_bf16 *)p)[idx]);
    case VELRI_DTYPE_I8:
        return (double)((const int8_t *)p)[idx];
    default:
        return NAN;
    }
}

static void store_from_f64(velri_dtype dt, void *p, size_t idx, double v) {
    switch (dt) {
    case VELRI_DTYPE_F32:
        ((float *)p)[idx] = (float)v;
        break;
    case VELRI_DTYPE_F64:
        ((double *)p)[idx] = v;
        break;
    case VELRI_DTYPE_F16:
        ((velri_fp16 *)p)[idx] = velri_f32_to_f16_scalar((float)v);
        break;
    case VELRI_DTYPE_BF16:
        ((velri_bf16 *)p)[idx] = velri_f32_to_bf16_scalar((float)v);
        break;
    default:
        break;
    }
}

void velri_test_fill_dtype(velri_dtype dt, void *buf, size_t n, uint64_t *rng, float scale) {
    if (dt == VELRI_DTYPE_F32) {
        float *p = (float *)buf;
        for (size_t i = 0; i < n; ++i) p[i] = velri_test_rand_f32(rng, -scale, scale);
    } else if (dt == VELRI_DTYPE_F64) {
        double *p = (double *)buf;
        for (size_t i = 0; i < n; ++i) p[i] = (double)velri_test_rand_f32(rng, -scale, scale);
    } else if (dt == VELRI_DTYPE_I8) {
        int8_t *p = (int8_t *)buf;
        for (size_t i = 0; i < n; ++i) p[i] = (int8_t)((int)(velri_test_rng(rng) % 251U) - 125);
    } else if (dt == VELRI_DTYPE_F16) {
        velri_fp16 *p = (velri_fp16 *)buf;
        for (size_t i = 0; i < n; ++i) {
            p[i] = velri_f32_to_f16_scalar(velri_test_rand_f32(rng, -scale, scale));
        }
    } else if (dt == VELRI_DTYPE_BF16) {
        velri_bf16 *p = (velri_bf16 *)buf;
        for (size_t i = 0; i < n; ++i) {
            p[i] = velri_f32_to_bf16_scalar(velri_test_rand_f32(rng, -scale, scale));
        }
    }
}

double velri_test_max_abs_diff_2d(velri_dtype dt, const void *a, const void *b, size_t rows,
                                  size_t cols, size_t stride) {
    double mx = 0.0;
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            double va = velri_test_load_f64(dt, a, r * stride + c);
            double vb = velri_test_load_f64(dt, b, r * stride + c);
            if (!isfinite(va) || !isfinite(vb)) return INFINITY;
            double d = fabs(va - vb);
            if (d > mx) mx = d;
        }
    }
    return mx;
}

void ref_linear_run(const velri_linear_desc *desc, size_t rows, size_t m, size_t k, const void *x,
                    size_t x_stride, const float *x_scale, const void *w, size_t w_stride,
                    const float *w_scale, const void *bias, void *y, size_t y_stride) {
    if (desc->x_dtype == VELRI_DTYPE_I8 && desc->w_dtype == VELRI_DTYPE_I8) {
        const int8_t *xi = (const int8_t *)x;
        const int8_t *wi = (const int8_t *)w;
        for (size_t r = 0; r < rows; ++r) {
            double xs = (desc->has_x_scale && x_scale) ? (double)x_scale[r] : 1.0;
            for (size_t c = 0; c < m; ++c) {
                int64_t sum = 0;
                for (size_t i = 0; i < k; ++i) {
                    sum += (int64_t)xi[r * x_stride + i] * (int64_t)wi[c * w_stride + i];
                }
                double ws = (desc->has_w_scale && w_scale) ? (double)w_scale[c] : 1.0;
                double b = bias ? (double)((const float *)bias)[c] : 0.0;
                ((float *)y)[r * y_stride + c] = (float)((double)sum * (xs * ws) + b);
            }
        }
        return;
    }

    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < m; ++c) {
            double sum = 0.0;
            for (size_t i = 0; i < k; ++i) {
                sum += velri_test_load_f64(desc->x_dtype, x, r * x_stride + i) *
                       velri_test_load_f64(desc->w_dtype, w, c * w_stride + i);
            }
            if (bias) {
                if (desc->y_dtype == VELRI_DTYPE_F64) {
                    sum += ((const double *)bias)[c];
                } else {
                    sum += (double)((const float *)bias)[c];
                }
            }
            store_from_f64(desc->y_dtype, y, r * y_stride + c, sum);
        }
    }
}

void ref_rmsnorm(velri_dtype io_dt, velri_dtype w_dt, size_t rows, size_t dim, float eps,
                 bool unit_offset, const void *x, size_t xs, const void *w, void *y, size_t ys) {
    for (size_t r = 0; r < rows; ++r) {
        double ss = 0.0;
        for (size_t d = 0; d < dim; ++d) {
            double v = velri_test_load_f64(io_dt, x, r * xs + d);
            ss += v * v;
        }
        double rrms = 1.0 / sqrt(ss / (double)dim + (double)eps);
        for (size_t d = 0; d < dim; ++d) {
            double v = velri_test_load_f64(io_dt, x, r * xs + d) * rrms;
            if (w) {
                double wv = velri_test_load_f64(w_dt, w, d);
                v *= unit_offset ? (1.0 + wv) : wv;
            }
            store_from_f64(io_dt, y, r * ys + d, v);
        }
    }
}

void ref_fused_add_rmsnorm_f32(size_t rows, size_t dim, float eps, bool unit_offset, float *res,
                               size_t rs, const float *x, size_t xs, const float *w, float *y,
                               size_t ys) {
    for (size_t r = 0; r < rows; ++r) {
        double ss = 0.0;
        for (size_t d = 0; d < dim; ++d) {
            float s = res[r * rs + d] + x[r * xs + d];
            res[r * rs + d] = s;
            ss += (double)s * (double)s;
        }
        double rrms = 1.0 / sqrt(ss / (double)dim + (double)eps);
        for (size_t d = 0; d < dim; ++d) {
            double v = (double)res[r * rs + d] * rrms;
            if (w) {
                v *= unit_offset ? (1.0 + (double)w[d]) : (double)w[d];
            }
            y[r * ys + d] = (float)v;
        }
    }
}

void ref_layernorm_f32(size_t rows, size_t dim, float eps, const float *x, size_t xs,
                       const float *w, const float *b, float *y, size_t ys) {
    for (size_t r = 0; r < rows; ++r) {
        double sum = 0.0;
        for (size_t d = 0; d < dim; ++d) {
            sum += (double)x[r * xs + d];
        }
        double mean = sum / (double)dim;
        double var = 0.0;
        for (size_t d = 0; d < dim; ++d) {
            double df = (double)x[r * xs + d] - mean;
            var += df * df;
        }
        double inv = 1.0 / sqrt(var / (double)dim + (double)eps);
        for (size_t d = 0; d < dim; ++d) {
            double v = ((double)x[r * xs + d] - mean) * inv;
            if (w) v *= (double)w[d];
            if (b) v += (double)b[d];
            y[r * ys + d] = (float)v;
        }
    }
}

void ref_rope(velri_dtype dt, size_t n_tokens, size_t n_heads, size_t head_dim, size_t rotary_dim,
              size_t pos_offset, const void *x, size_t tsx, size_t hsx, const float *cos_t,
              const float *sin_t, void *y, size_t tsy, size_t hsy, velri_rope_mode mode) {
    size_t half = rotary_dim / 2U;
    for (size_t t = 0; t < n_tokens; ++t) {
        const float *cr = rotary_dim ? (cos_t + (pos_offset + t) * half) : nullptr;
        const float *sr = rotary_dim ? (sin_t + (pos_offset + t) * half) : nullptr;
        for (size_t h = 0; h < n_heads; ++h) {
            size_t xo = t * tsx + h * hsx;
            size_t yo = t * tsy + h * hsy;
            if (mode == VELRI_ROPE_HALF) {
                for (size_t d = 0; d < half; ++d) {
                    double x0 = velri_test_load_f64(dt, x, xo + d);
                    double x1 = velri_test_load_f64(dt, x, xo + half + d);
                    double c = (double)cr[d], s = (double)sr[d];
                    store_from_f64(dt, y, yo + d, x0 * c - x1 * s);
                    store_from_f64(dt, y, yo + half + d, x1 * c + x0 * s);
                }
            } else {
                for (size_t d = 0; d < half; ++d) {
                    double x0 = velri_test_load_f64(dt, x, xo + 2U * d);
                    double x1 = velri_test_load_f64(dt, x, xo + 2U * d + 1U);
                    double c = (double)cr[d], s = (double)sr[d];
                    store_from_f64(dt, y, yo + 2U * d, x0 * c - x1 * s);
                    store_from_f64(dt, y, yo + 2U * d + 1U, x1 * c + x0 * s);
                }
            }
            for (size_t d = rotary_dim; d < head_dim; ++d) {
                store_from_f64(dt, y, yo + d, velri_test_load_f64(dt, x, xo + d));
            }
        }
    }
}

static double ref_act_f64(velri_act_kind kind, double x) {
    if (kind == VELRI_ACT_SILU) {
        if (x < -88.0) return 0.0;
        return x / (1.0 + exp(-x));
    }
    if (kind == VELRI_ACT_GELU_TANH) {
        if (x < -12.0) return 0.0;
        double arg = -1.5957691216057308 * x * (1.0 + 0.044715 * x * x);
        if (arg > 500.0) return 0.0;
        return x / (1.0 + exp(arg));
    }
    double r = x > 0.0 ? x : 0.0;
    return r * r;
}

void ref_activation_f32(velri_act_kind kind, size_t count, const float *x, float *y) {
    for (size_t i = 0; i < count; ++i) {
        y[i] = (float)ref_act_f64(kind, (double)x[i]);
    }
}

void ref_gated_mlp_f32(velri_act_kind kind, size_t count, float clamp_limit, const float *gate,
                       const float *up, float *y) {
    bool do_clamp = (clamp_limit > 0.0f);
    double lo = -(double)clamp_limit, hi = (double)clamp_limit;
    for (size_t i = 0; i < count; ++i) {
        double g = (double)gate[i], u = (double)up[i];
        if (do_clamp) {
            g = g < lo ? lo : (g > hi ? hi : g);
            u = u < lo ? lo : (u > hi ? hi : u);
        }
        y[i] = (float)(ref_act_f64(kind, g) * u);
    }
}

void ref_softmax_f32(size_t rows, size_t cols, const float *x, size_t xs, float *y, size_t ys) {
    for (size_t r = 0; r < rows; ++r) {
        double m = -INFINITY;
        for (size_t i = 0; i < cols; ++i) {
            double v = (double)x[r * xs + i];
            if (v > m) m = v;
        }
        if (m == -INFINITY) {
            memset(y + r * ys, 0, cols * sizeof(float));
            continue;
        }
        double sum = 0.0;
        for (size_t i = 0; i < cols; ++i) {
            double v = (double)x[r * xs + i];
            if (v > -INFINITY) sum += exp(v - m);
        }
        for (size_t i = 0; i < cols; ++i) {
            double v = (double)x[r * xs + i];
            y[r * ys + i] = (v > -INFINITY && sum > 0.0) ? (float)(exp(v - m) / sum) : 0.0f;
        }
    }
}

void ref_fwht_f32(size_t rows, size_t cols, size_t block_size, const float *x, size_t xs,
                  velri_dtype suh_dt, const void *suh, float *y, size_t ys) {
    const double norm = 1.0 / sqrt((double)block_size);
    double buf[256];
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c0 = 0; c0 < cols; c0 += block_size) {
            for (size_t j = 0; j < block_size; ++j) {
                double v = (double)x[r * xs + c0 + j];
                if (suh) v *= velri_test_load_f64(suh_dt, suh, c0 + j);
                buf[j] = v * norm;
            }
            for (size_t h = block_size >> 1; h >= 1; h >>= 1) {
                for (size_t i = 0; i < block_size; i += (h << 1)) {
                    for (size_t j = 0; j < h; ++j) {
                        double a = buf[i + j];
                        double b = buf[i + j + h];
                        buf[i + j] = a + b;
                        buf[i + j + h] = a - b;
                    }
                }
            }
            for (size_t j = 0; j < block_size; ++j) {
                y[r * ys + c0 + j] = (float)buf[j];
            }
        }
    }
}

void ref_vec_axpby_f32(size_t count, float alpha, const float *a, float beta, const float *b,
                       float *y) {
    for (size_t i = 0; i < count; ++i) {
        double v = (double)alpha * (double)a[i];
        if (beta != 0.0f) v += (double)beta * (double)b[i];
        y[i] = (float)v;
    }
}

void ref_vec_mul_f32(size_t count, const float *a, const float *b, float *y) {
    for (size_t i = 0; i < count; ++i) {
        y[i] = (float)((double)a[i] * (double)b[i]);
    }
}

void ref_quantize_row_f32_i8(size_t rows, size_t k, const float *x, size_t x_stride,
                             int8_t *x_i8, size_t xi8_stride, float *scale_out) {
    for (size_t r = 0; r < rows; ++r) {
        const float *xr = x + r * x_stride;
        int8_t *qr = x_i8 + r * xi8_stride;
        double amax = 0.0;
        for (size_t c = 0; c < k; ++c) {
            double a = fabs((double)xr[c]);
            if (a > amax) amax = a;
        }
        double scale = (amax > 0.0) ? fmax(amax / 127.0, (double)FLT_TRUE_MIN) : 1.0;
        double inv = (amax > 0.0) ? (1.0 / scale) : 0.0;
        scale_out[r] = (float)scale;
        for (size_t c = 0; c < k; ++c) {
            double v = rint((double)xr[c] * inv);
            if (v > 127.0) v = 127.0;
            if (v < -127.0) v = -127.0;
            qr[c] = (int8_t)v;
        }
    }
}
