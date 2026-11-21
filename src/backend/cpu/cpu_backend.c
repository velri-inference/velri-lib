#include "core_internal.h"
#include "pool.h"
#include <stdio.h>
#include <string.h>

/*
 * Built-in CPU Backend wrapping Slang AOT kernels (`entry_cpu.slang`) and `velri_pool`.
 *
 * Single X-macro kernel table generates declarations, vtable struct, and ISA profiles
 * (base + optional x86-64 AVX2/AVX-512 fat-binary dispatch).
 *
 * Concurrency:
 *   Multiple slots may call `submit` concurrently. A non-blocking atomic flag guards
 *   shared `velri_pool` usage; if another slot holds the pool, the second caller runs
 *   directly on its own thread without blocking or corrupting pool state.
 */

#define SLANG_CPU_KERNELS(X)                                                                       \
    X(linear_f32, (const float *, uint32_t, const float *, uint32_t, const float *, uint32_t,      \
                   float *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t))           \
    X(linear_f16_f32, (const uint16_t *, uint32_t, const uint16_t *, uint32_t, const float *,      \
                       uint32_t, float *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,        \
                       uint32_t))                                                                  \
    X(linear_bf16_f32, (const uint16_t *, uint32_t, const uint16_t *, uint32_t, const float *,     \
                        uint32_t, float *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,       \
                        uint32_t))                                                                 \
    X(linear_f32x_bf16w, (const float *, uint32_t, const uint16_t *, uint32_t, const float *,      \
                          uint32_t, float *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,     \
                          uint32_t))                                                               \
    X(linear_f64, (const double *, uint32_t, const double *, uint32_t, const double *, uint32_t,   \
                   double *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t))          \
    X(linear_w8a8_f32, (const int8_t *, uint32_t, const int8_t *, uint32_t, const float *,         \
                        const float *, const float *, uint32_t, float *, uint32_t, uint32_t,       \
                        uint32_t, uint32_t, uint32_t, uint32_t))                                   \
    X(rmsnorm_f32, (const float *, uint32_t, const float *, uint32_t, float *, uint32_t, uint32_t, \
                    uint32_t, uint32_t, float, uint32_t))                                          \
    X(rmsnorm_bf16, (const uint16_t *, uint32_t, const uint16_t *, uint32_t, uint16_t *, uint32_t, \
                     uint32_t, uint32_t, uint32_t, float, uint32_t))                               \
    X(rmsnorm_f16, (const uint16_t *, uint32_t, const uint16_t *, uint32_t, uint16_t *, uint32_t,  \
                    uint32_t, uint32_t, uint32_t, float, uint32_t))                                \
    X(fused_add_rmsnorm_f32, (float *, uint32_t, const float *, uint32_t, const float *, uint32_t, \
                              float *, uint32_t, uint32_t, uint32_t, uint32_t, float, uint32_t))   \
    X(layernorm_f32, (const float *, uint32_t, const float *, uint32_t, const float *, uint32_t,   \
                      float *, uint32_t, uint32_t, uint32_t, uint32_t, float))                     \
    X(rope_f32, (const float *, uint32_t, uint32_t, const float *, const float *, float *,         \
                 uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,   \
                 uint32_t))                                                                        \
    X(rope_bf16, (const uint16_t *, uint32_t, uint32_t, const float *, const float *, uint16_t *,  \
                  uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,  \
                  uint32_t))                                                                       \
    X(activation_silu_f32, (const float *, float *, uint32_t, uint32_t))                           \
    X(activation_gelu_f32, (const float *, float *, uint32_t, uint32_t))                           \
    X(activation_relu2_f32, (const float *, float *, uint32_t, uint32_t))                          \
    X(gated_silu_f32, (const float *, const float *, float *, uint32_t, uint32_t, float))          \
    X(gated_gelu_f32, (const float *, const float *, float *, uint32_t, uint32_t, float))          \
    X(softmax_f32, (const float *, uint32_t, float *, uint32_t, uint32_t, uint32_t, uint32_t))     \
    X(vec_axpby_f32, (float, const float *, float, const float *, float *, uint32_t, uint32_t))    \
    X(vec_mul_f32, (const float *, const float *, float *, uint32_t, uint32_t))                    \
    X(fwht_f32, (const float *, uint32_t, const uint8_t *, uint32_t, float *, uint32_t, uint32_t,  \
                 uint32_t, uint32_t, uint32_t))                                                    \
    X(quantize_row_f32_i8, (const float *, uint32_t, int8_t *, uint32_t, float *, uint32_t,        \
                            uint32_t, uint32_t))

#define DECL_BASE(NAME, SIG) extern void velri_slang_base_##NAME SIG;
SLANG_CPU_KERNELS(DECL_BASE)
#undef DECL_BASE

#if VELRI_X86_VARIANTS
#define DECL_AVX2(NAME, SIG) extern void velri_slang_avx2_##NAME SIG;
#define DECL_AVX512(NAME, SIG) extern void velri_slang_avx512_##NAME SIG;
SLANG_CPU_KERNELS(DECL_AVX2)
SLANG_CPU_KERNELS(DECL_AVX512)
#undef DECL_AVX2
#undef DECL_AVX512
#endif

typedef struct {
#define FIELD_DEF(NAME, SIG) void (*NAME) SIG;
    SLANG_CPU_KERNELS(FIELD_DEF)
#undef FIELD_DEF
} slang_cpu_vtable;

#define INIT_BASE(NAME, SIG) .NAME = velri_slang_base_##NAME,
static const slang_cpu_vtable g_vt_base = {SLANG_CPU_KERNELS(INIT_BASE)};
#undef INIT_BASE

#if VELRI_X86_VARIANTS
#define INIT_AVX2(NAME, SIG) .NAME = velri_slang_avx2_##NAME,
#define INIT_AVX512(NAME, SIG) .NAME = velri_slang_avx512_##NAME,
static const slang_cpu_vtable g_vt_avx2 = {SLANG_CPU_KERNELS(INIT_AVX2)};
static const slang_cpu_vtable g_vt_avx512 = {SLANG_CPU_KERNELS(INIT_AVX512)};
#undef INIT_AVX2
#undef INIT_AVX512
#endif

static const slang_cpu_vtable *resolve_slang_vt(velri_cpu_variant variant) {
    if (!velri_cpu_variant_supported(variant)) {
        return nullptr;
    }
    velri_cpu_variant r = velri_cpu_resolve_variant(variant);
#if VELRI_X86_VARIANTS
    if (r == VELRI_CPU_AVX512) return &g_vt_avx512;
    if (r == VELRI_CPU_AVX2) return &g_vt_avx2;
#endif
    return (r == VELRI_CPU_BASE) ? &g_vt_base : nullptr;
}

struct velri_backend_device {
    velri_pool *pool;
    size_t workers;
    velri_cpu_variant variant;
    const slang_cpu_vtable *vt;
    atomic_flag pool_busy;
};

static inline const void *binding_cptr(const velri_buffer_binding *b) {
    return (b && b->mem_handle) ? (const uint8_t *)b->mem_handle + b->byte_offset : nullptr;
}

static inline void *binding_ptr(const velri_buffer_binding *b) {
    return (b && b->mem_handle) ? (uint8_t *)b->mem_handle + b->byte_offset : nullptr;
}

static inline void split_1d(uint32_t total, size_t wid, size_t nw, uint32_t *out0, uint32_t *out1) {
    *out0 = (uint32_t)(((uint64_t)total * (uint64_t)wid) / (uint64_t)nw);
    *out1 = (uint32_t)(((uint64_t)total * (uint64_t)(wid + 1U)) / (uint64_t)nw);
}

static int linear_mode_for_desc(const velri_linear_desc *d) {
    if (d->x_dtype == VELRI_DTYPE_F32 && d->w_dtype == VELRI_DTYPE_F32 &&
        d->y_dtype == VELRI_DTYPE_F32 && !d->has_x_scale && !d->has_w_scale) {
        return 0;
    }
    if (d->x_dtype == VELRI_DTYPE_F16 && d->w_dtype == VELRI_DTYPE_F16 &&
        d->y_dtype == VELRI_DTYPE_F32 && !d->has_x_scale && !d->has_w_scale) {
        return 1;
    }
    if (d->x_dtype == VELRI_DTYPE_BF16 && d->w_dtype == VELRI_DTYPE_BF16 &&
        d->y_dtype == VELRI_DTYPE_F32 && !d->has_x_scale && !d->has_w_scale) {
        return 2;
    }
    if (d->x_dtype == VELRI_DTYPE_F32 && d->w_dtype == VELRI_DTYPE_BF16 &&
        d->y_dtype == VELRI_DTYPE_F32 && !d->has_x_scale && !d->has_w_scale) {
        return 3;
    }
    if (d->x_dtype == VELRI_DTYPE_F64 && d->w_dtype == VELRI_DTYPE_F64 &&
        d->y_dtype == VELRI_DTYPE_F64 && !d->has_x_scale && !d->has_w_scale) {
        return 4;
    }
    if (d->x_dtype == VELRI_DTYPE_I8 && d->w_dtype == VELRI_DTYPE_I8 &&
        d->y_dtype == VELRI_DTYPE_F32 && d->has_x_scale && d->has_w_scale) {
        return 5;
    }
    return -1;
}

/* ---- 1. Registration ---- */

static velri_status cpu_enumerate_devices(uint32_t *out_count) {
    if (!out_count) {
        return VELRI_ERR_INVALID;
    }
    *out_count = 1U;
    return VELRI_OK;
}

/* ---- 2. Device ---- */

static velri_status cpu_open_device(uint32_t local_index, const void *cfg,
                                    velri_backend_device **out_dev) {
    if (local_index != 0U || !out_dev) {
        return VELRI_ERR_NOT_FOUND;
    }
    const velri_cpu_open_cfg *ocfg = (const velri_cpu_open_cfg *)cfg;
    velri_cpu_variant req_var = ocfg ? ocfg->variant : VELRI_CPU_AUTO;
    const slang_cpu_vtable *vt = resolve_slang_vt(req_var);
    if (!vt) {
        return VELRI_ERR_UNSUPPORTED;
    }

    velri_backend_device *dev = (velri_backend_device *)calloc(1U, sizeof(*dev));
    if (!dev) {
        return VELRI_ERR_NOMEM;
    }

    size_t nw = ocfg ? ocfg->workers : 0U;
    if (nw == 0U) {
        const velri_cpu_info *ci = velri_get_cpu_info();
        int hw = ci->hardware_threads > 0 ? ci->hardware_threads : 1;
        nw = (size_t)(hw > 16 ? hw / 2 : hw);
        if (nw == 0U) {
            nw = 1U;
        }
    }
    dev->workers = nw;
    dev->variant = velri_cpu_resolve_variant(req_var);
    dev->vt = vt;
    atomic_flag_clear(&dev->pool_busy);

    if (nw > 1U) {
        velri_status st = velri_pool_create(nw, &dev->pool);
        if (st != VELRI_OK) {
            free(dev);
            return st;
        }
    }
    *out_dev = dev;
    return VELRI_OK;
}

static velri_status cpu_query_caps(const velri_backend_device *dev, velri_device_caps *out_caps) {
    if (!dev || !out_caps) {
        return VELRI_ERR_INVALID;
    }
    memset(out_caps, 0, sizeof(*out_caps));
    snprintf(out_caps->backend_name, sizeof(out_caps->backend_name), "%s", "cpu");
    snprintf(out_caps->device_name, sizeof(out_caps->device_name), "Slang AOT CPU (workers=%zu)",
             dev->workers);
    out_caps->local_index = 0U;
    out_caps->is_host_memory = true;
    out_caps->is_uma = true;
    out_caps->mem_total_known = false;
    out_caps->mem_total_bytes = 0U;
    out_caps->min_align_bytes = 64U;
    return VELRI_OK;
}

static void cpu_close_device(velri_backend_device *dev) {
    if (!dev) {
        return;
    }
    velri_pool_destroy(dev->pool);
    free(dev);
}

/* ---- 3. Memory ---- */

static velri_status cpu_alloc(velri_backend_device *dev, size_t bytes, size_t align,
                              void **out_mem_handle) {
    (void)dev;
    if (!out_mem_handle || (align != 0U && (align & (align - 1U)) != 0U)) {
        return VELRI_ERR_INVALID;
    }
    size_t eff_align = align < 64U ? 64U : align;
    size_t alloc_bytes = bytes ? velri_align_up(bytes, eff_align) : eff_align;
    if (alloc_bytes == SIZE_MAX) {
        return VELRI_ERR_OVERFLOW;
    }
    void *p = velri_os_aligned_alloc(eff_align, alloc_bytes);
    if (!p) {
        return VELRI_ERR_NOMEM;
    }
    memset(p, 0, alloc_bytes);
    *out_mem_handle = p;
    return VELRI_OK;
}

static void cpu_free(velri_backend_device *dev, void *mem_handle) {
    (void)dev;
    velri_os_aligned_free(mem_handle);
}

static velri_status cpu_copy_to_ram(velri_backend_device *dev, void *dst_ram,
                                    const velri_buffer_binding *src, size_t bytes) {
    (void)dev;
    if (!dst_ram || !src || !src->mem_handle || src->byte_offset > src->byte_size ||
        bytes > src->byte_size - src->byte_offset) {
        return VELRI_ERR_INVALID;
    }
    if (bytes > 0U) {
        memcpy(dst_ram, binding_cptr(src), bytes);
    }
    return VELRI_OK;
}

static velri_status cpu_copy_from_ram(velri_backend_device *dev, const velri_buffer_binding *dst,
                                      const void *src_ram, size_t bytes) {
    (void)dev;
    if (!dst || !dst->mem_handle || !src_ram || dst->byte_offset > dst->byte_size ||
        bytes > dst->byte_size - dst->byte_offset) {
        return VELRI_ERR_INVALID;
    }
    if (bytes > 0U) {
        memcpy(binding_ptr(dst), src_ram, bytes);
    }
    return VELRI_OK;
}

/* ---- 4. Preparation ---- */

static velri_status cpu_query_operation(const velri_backend_device *dev,
                                        const velri_resolved_op *op) {
    if (!dev || !op) {
        return VELRI_ERR_INVALID;
    }
    velri_status st = velri_op_validate(op);
    if (st != VELRI_OK) {
        return st;
    }
    if (op->kind == VELRI_OP_LINEAR) {
        return linear_mode_for_desc(&op->raw.attr.linear.desc) >= 0 ? VELRI_OK
                                                                    : VELRI_ERR_UNSUPPORTED;
    }
    return VELRI_OK;
}

/* ---- 5. Execution ---- */

typedef struct {
    const slang_cpu_vtable *vt;
    const velri_resolved_op *op;
    const velri_buffer_binding *in_b;
    const velri_buffer_binding *out_b;
    uint32_t pos_offset;
    uint32_t elem_count;
    int linear_mode;
} cpu_dispatch_ctx;

static void cpu_op_worker(void *vctx, size_t wid, size_t nw) {
    const cpu_dispatch_ctx *c = (const cpu_dispatch_ctx *)vctx;
    const velri_resolved_op *op = c->op;
    const slang_cpu_vtable *vt = c->vt;

    switch (op->kind) {
    case VELRI_OP_LINEAR: {
        uint32_t rows = (uint32_t)op->inputs[0].dims[0];
        uint32_t m = (uint32_t)op->inputs[1].dims[0];
        uint32_t k = (uint32_t)op->inputs[0].dims[1];
        uint32_t xs = (uint32_t)op->inputs[0].strides[0];
        uint32_t ws = (uint32_t)op->inputs[1].strides[0];
        uint32_t ys = (uint32_t)op->outputs[0].strides[0];
        uint32_t r0 = 0U, r1 = rows, c0 = 0U, c1 = m;

        if (rows >= 4U && (uint64_t)m < (uint64_t)nw * 16ULL &&
            (uint64_t)rows >= (uint64_t)nw * 6ULL) {
            uint64_t rblocks = ((uint64_t)rows + 5ULL) / 6ULL;
            uint64_t br0 = ((rblocks * (uint64_t)wid) / (uint64_t)nw) * 6ULL;
            uint64_t br1 = ((rblocks * (uint64_t)(wid + 1U)) / (uint64_t)nw) * 6ULL;
            if (br1 > rows) br1 = rows;
            if (br0 >= br1) return;
            r0 = (uint32_t)br0;
            r1 = (uint32_t)br1;
        } else {
            uint64_t align = 4ULL;
            if (rows >= 4U) {
                uint64_t want = (c->linear_mode == 4) ? 32ULL : 64ULL;
                if ((uint64_t)m >= (uint64_t)nw * want) align = want;
                else if ((uint64_t)m >= (uint64_t)nw * 16ULL) align = 16ULL;
            }
            uint64_t cblocks = ((uint64_t)m + align - 1ULL) / align;
            uint64_t bc0 = ((cblocks * (uint64_t)wid) / (uint64_t)nw) * align;
            uint64_t bc1 = ((cblocks * (uint64_t)(wid + 1U)) / (uint64_t)nw) * align;
            if (bc1 > m) bc1 = m;
            if (bc0 >= bc1) return;
            c0 = (uint32_t)bc0;
            c1 = (uint32_t)bc1;
        }

        const void *x = binding_cptr(&c->in_b[0]);
        const void *w = binding_cptr(&c->in_b[1]);
        const void *bias = op->input_present[2] ? binding_cptr(&c->in_b[2]) : nullptr;
        const float *xsc = op->input_present[3] ? (const float *)binding_cptr(&c->in_b[3]) : nullptr;
        const float *wsc = op->input_present[4] ? (const float *)binding_cptr(&c->in_b[4]) : nullptr;
        void *y = binding_ptr(&c->out_b[0]);
        uint32_t flags = (bias ? 1U : 0U) | (xsc ? 2U : 0U) | (wsc ? 4U : 0U);

        switch (c->linear_mode) {
        case 0:
            vt->linear_f32((const float *)x, xs, (const float *)w, ws, (const float *)bias, flags,
                           (float *)y, ys, r0, r1, c0, c1, k);
            break;
        case 1:
            vt->linear_f16_f32((const uint16_t *)x, xs, (const uint16_t *)w, ws,
                               (const float *)bias, flags, (float *)y, ys, r0, r1, c0, c1, k);
            break;
        case 2:
            vt->linear_bf16_f32((const uint16_t *)x, xs, (const uint16_t *)w, ws,
                                (const float *)bias, flags, (float *)y, ys, r0, r1, c0, c1, k);
            break;
        case 3:
            vt->linear_f32x_bf16w((const float *)x, xs, (const uint16_t *)w, ws,
                                  (const float *)bias, flags, (float *)y, ys, r0, r1, c0, c1, k);
            break;
        case 4:
            vt->linear_f64((const double *)x, xs, (const double *)w, ws, (const double *)bias,
                           bias ? 1U : 0U, (double *)y, ys, r0, r1, c0, c1, k);
            break;
        case 5:
            vt->linear_w8a8_f32((const int8_t *)x, xs, (const int8_t *)w, ws, (const float *)bias,
                                xsc, wsc, flags, (float *)y, ys, r0, r1, c0, c1, k);
            break;
        default:
            break;
        }
        break;
    }

    case VELRI_OP_RMSNORM: {
        uint32_t r0 = 0U, r1 = 0U;
        split_1d((uint32_t)op->inputs[0].dims[0], wid, nw, &r0, &r1);
        if (r0 >= r1) return;
        uint32_t dim = (uint32_t)op->inputs[0].dims[1];
        uint32_t xs = (uint32_t)op->inputs[0].strides[0];
        uint32_t ys = (uint32_t)op->outputs[0].strides[0];
        const void *x = binding_cptr(&c->in_b[0]);
        const void *w = op->input_present[1] ? binding_cptr(&c->in_b[1]) : nullptr;
        void *y = binding_ptr(&c->out_b[0]);
        uint32_t has_w = w ? 1U : 0U;
        uint32_t uo = op->raw.attr.rmsnorm.unit_offset ? 1U : 0U;
        float eps = op->raw.attr.rmsnorm.eps;
        if (op->inputs[0].dtype == VELRI_DTYPE_F32) {
            vt->rmsnorm_f32((const float *)x, xs, (const float *)w, has_w, (float *)y, ys, r0, r1,
                            dim, eps, uo);
        } else if (op->inputs[0].dtype == VELRI_DTYPE_BF16) {
            vt->rmsnorm_bf16((const uint16_t *)x, xs, (const uint16_t *)w, has_w, (uint16_t *)y, ys,
                             r0, r1, dim, eps, uo);
        } else {
            vt->rmsnorm_f16((const uint16_t *)x, xs, (const uint16_t *)w, has_w, (uint16_t *)y, ys,
                            r0, r1, dim, eps, uo);
        }
        break;
    }

    case VELRI_OP_FUSED_ADD_RMSNORM: {
        uint32_t r0 = 0U, r1 = 0U;
        split_1d((uint32_t)op->inputs[0].dims[0], wid, nw, &r0, &r1);
        if (r0 >= r1) return;
        const float *w = op->input_present[1] ? (const float *)binding_cptr(&c->in_b[1]) : nullptr;
        vt->fused_add_rmsnorm_f32((float *)binding_ptr(&c->out_b[0]),
                                  (uint32_t)op->outputs[0].strides[0],
                                  (const float *)binding_cptr(&c->in_b[0]),
                                  (uint32_t)op->inputs[0].strides[0], w, w ? 1U : 0U,
                                  (float *)binding_ptr(&c->out_b[1]),
                                  (uint32_t)op->outputs[1].strides[0], r0, r1,
                                  (uint32_t)op->inputs[0].dims[1], op->raw.attr.rmsnorm.eps,
                                  op->raw.attr.rmsnorm.unit_offset ? 1U : 0U);
        break;
    }

    case VELRI_OP_LAYERNORM: {
        uint32_t r0 = 0U, r1 = 0U;
        split_1d((uint32_t)op->inputs[0].dims[0], wid, nw, &r0, &r1);
        if (r0 >= r1) return;
        const float *w = op->input_present[1] ? (const float *)binding_cptr(&c->in_b[1]) : nullptr;
        const float *b = op->input_present[2] ? (const float *)binding_cptr(&c->in_b[2]) : nullptr;
        vt->layernorm_f32((const float *)binding_cptr(&c->in_b[0]),
                          (uint32_t)op->inputs[0].strides[0], w, w ? 1U : 0U, b, b ? 1U : 0U,
                          (float *)binding_ptr(&c->out_b[0]), (uint32_t)op->outputs[0].strides[0],
                          r0, r1, (uint32_t)op->inputs[0].dims[1], op->raw.attr.layernorm.eps);
        break;
    }

    case VELRI_OP_ROPE: {
        uint32_t t0 = 0U, t1 = 0U;
        split_1d((uint32_t)op->inputs[0].dims[0], wid, nw, &t0, &t1);
        if (t0 >= t1) return;
        const void *x = binding_cptr(&c->in_b[0]);
        const float *ct = (const float *)binding_cptr(&c->in_b[1]);
        const float *st = (const float *)binding_cptr(&c->in_b[2]);
        void *y = binding_ptr(&c->out_b[0]);
        uint32_t tsx = (uint32_t)op->inputs[0].strides[0];
        uint32_t hsx = (uint32_t)op->inputs[0].strides[1];
        uint32_t tsy = (uint32_t)op->outputs[0].strides[0];
        uint32_t hsy = (uint32_t)op->outputs[0].strides[1];
        uint32_t nh = (uint32_t)op->inputs[0].dims[1];
        uint32_t hd = (uint32_t)op->inputs[0].dims[2];
        uint32_t rd = (uint32_t)op->raw.attr.rope.rotary_dim;
        uint32_t mode = (uint32_t)op->raw.attr.rope.mode;
        if (op->inputs[0].dtype == VELRI_DTYPE_F32) {
            vt->rope_f32((const float *)x, tsx, hsx, ct, st, (float *)y, tsy, hsy, t0, t1, nh, hd,
                         rd, c->pos_offset, mode);
        } else {
            vt->rope_bf16((const uint16_t *)x, tsx, hsx, ct, st, (uint16_t *)y, tsy, hsy, t0, t1,
                          nh, hd, rd, c->pos_offset, mode);
        }
        break;
    }

    case VELRI_OP_ACTIVATION: {
        uint32_t i0 = 0U, i1 = 0U;
        split_1d(c->elem_count, wid, nw, &i0, &i1);
        if (i0 >= i1) return;
        const float *x = (const float *)binding_cptr(&c->in_b[0]);
        float *y = (float *)binding_ptr(&c->out_b[0]);
        velri_act_kind k = op->raw.attr.activation.kind;
        if (k == VELRI_ACT_SILU) vt->activation_silu_f32(x, y, i0, i1);
        else if (k == VELRI_ACT_GELU_TANH) vt->activation_gelu_f32(x, y, i0, i1);
        else vt->activation_relu2_f32(x, y, i0, i1);
        break;
    }

    case VELRI_OP_GATED_MLP: {
        uint32_t i0 = 0U, i1 = 0U;
        split_1d(c->elem_count, wid, nw, &i0, &i1);
        if (i0 >= i1) return;
        const float *g = (const float *)binding_cptr(&c->in_b[0]);
        const float *u = (const float *)binding_cptr(&c->in_b[1]);
        float *y = (float *)binding_ptr(&c->out_b[0]);
        if (op->raw.attr.gated_mlp.act == VELRI_ACT_SILU) {
            vt->gated_silu_f32(g, u, y, i0, i1, op->raw.attr.gated_mlp.clamp_limit);
        } else {
            vt->gated_gelu_f32(g, u, y, i0, i1, op->raw.attr.gated_mlp.clamp_limit);
        }
        break;
    }

    case VELRI_OP_SOFTMAX: {
        uint32_t r0 = 0U, r1 = 0U;
        split_1d((uint32_t)op->inputs[0].dims[0], wid, nw, &r0, &r1);
        if (r0 >= r1) return;
        vt->softmax_f32((const float *)binding_cptr(&c->in_b[0]),
                        (uint32_t)op->inputs[0].strides[0], (float *)binding_ptr(&c->out_b[0]),
                        (uint32_t)op->outputs[0].strides[0], r0, r1,
                        (uint32_t)op->inputs[0].dims[1]);
        break;
    }

    case VELRI_OP_VEC_AXPBY: {
        uint32_t i0 = 0U, i1 = 0U;
        split_1d(c->elem_count, wid, nw, &i0, &i1);
        if (i0 >= i1) return;
        const float *a = (const float *)binding_cptr(&c->in_b[0]);
        const float *b = op->input_present[1] ? (const float *)binding_cptr(&c->in_b[1]) : a;
        vt->vec_axpby_f32(op->raw.attr.axpby.alpha, a, op->raw.attr.axpby.beta, b,
                          (float *)binding_ptr(&c->out_b[0]), i0, i1);
        break;
    }

    case VELRI_OP_VEC_MUL: {
        uint32_t i0 = 0U, i1 = 0U;
        split_1d(c->elem_count, wid, nw, &i0, &i1);
        if (i0 >= i1) return;
        vt->vec_mul_f32((const float *)binding_cptr(&c->in_b[0]),
                        (const float *)binding_cptr(&c->in_b[1]),
                        (float *)binding_ptr(&c->out_b[0]), i0, i1);
        break;
    }

    case VELRI_OP_FWHT: {
        uint32_t r0 = 0U, r1 = 0U;
        split_1d((uint32_t)op->inputs[0].dims[0], wid, nw, &r0, &r1);
        if (r0 >= r1) return;
        const float *x = (const float *)binding_cptr(&c->in_b[0]);
        const void *suh = op->input_present[1] ? binding_cptr(&c->in_b[1]) : nullptr;
        uint32_t suh_mode = 0U;
        if (suh) {
            suh_mode = (op->raw.attr.fwht.suh_dtype == VELRI_DTYPE_F32) ? 1U : 2U;
        }
        vt->fwht_f32(x, (uint32_t)op->inputs[0].strides[0],
                     suh ? (const uint8_t *)suh : (const uint8_t *)x, suh_mode,
                     (float *)binding_ptr(&c->out_b[0]), (uint32_t)op->outputs[0].strides[0], r0,
                     r1, (uint32_t)op->inputs[0].dims[1], (uint32_t)op->raw.attr.fwht.block_size);
        break;
    }

    case VELRI_OP_QUANTIZE_ROW: {
        uint32_t r0 = 0U, r1 = 0U;
        split_1d((uint32_t)op->inputs[0].dims[0], wid, nw, &r0, &r1);
        if (r0 >= r1) return;
        vt->quantize_row_f32_i8((const float *)binding_cptr(&c->in_b[0]),
                                (uint32_t)op->inputs[0].strides[0],
                                (int8_t *)binding_ptr(&c->out_b[0]),
                                (uint32_t)op->outputs[0].strides[0],
                                (float *)binding_ptr(&c->out_b[1]), r0, r1,
                                (uint32_t)op->inputs[0].dims[1]);
        break;
    }

    default:
        break;
    }
}

static velri_status cpu_submit(velri_backend_device *dev, const velri_submit_item *item,
                               velri_completion **out_completion) {
    if (!out_completion) {
        return VELRI_ERR_INVALID;
    }
    *out_completion = nullptr;
    if (!dev || !item || !item->op || item->num_inputs != item->op->num_inputs ||
        item->num_outputs != item->op->num_outputs) {
        return VELRI_ERR_INVALID;
    }

    const velri_resolved_op *op = item->op;
    cpu_dispatch_ctx ctx = {
        .vt = dev->vt,
        .op = op,
        .in_b = item->inputs,
        .out_b = item->outputs,
        .pos_offset = 0U,
        .elem_count = 0U,
        .linear_mode = -1,
    };

    size_t work_units = 1U;
    switch (op->kind) {
    case VELRI_OP_LINEAR:
        ctx.linear_mode = linear_mode_for_desc(&op->raw.attr.linear.desc);
        if (ctx.linear_mode < 0) {
            return VELRI_ERR_UNSUPPORTED;
        }
        work_units = 2U; /* Always parallelize across M or rows */
        break;
    case VELRI_OP_ROPE: {
        size_t pos = (item->params && item->params->has_pos_offset)
                         ? item->params->pos_offset_override
                         : op->raw.attr.rope.pos_offset;
        if (pos > (size_t)UINT32_MAX) {
            return VELRI_ERR_OVERFLOW;
        }
        ctx.pos_offset = (uint32_t)pos;
        work_units = op->inputs[0].dims[0];
        break;
    }
    case VELRI_OP_ACTIVATION:
    case VELRI_OP_GATED_MLP:
    case VELRI_OP_VEC_AXPBY:
    case VELRI_OP_VEC_MUL: {
        size_t n = 0U;
        if (velri_tensor_view_num_elements(&op->inputs[0], &n) != VELRI_OK ||
            n > (size_t)UINT32_MAX) {
            return VELRI_ERR_OVERFLOW;
        }
        ctx.elem_count = (uint32_t)n;
        work_units = n;
        break;
    }
    case VELRI_OP_RMSNORM:
    case VELRI_OP_FUSED_ADD_RMSNORM:
    case VELRI_OP_LAYERNORM:
    case VELRI_OP_SOFTMAX:
    case VELRI_OP_FWHT:
    case VELRI_OP_QUANTIZE_ROW:
        work_units = op->inputs[0].dims[0];
        break;
    default:
        return VELRI_ERR_UNSUPPORTED;
    }

    bool owns_pool = false;
    velri_pool *use_pool = nullptr;
    size_t use_nw = 1U;
    if (work_units > 1U && dev->pool &&
        !atomic_flag_test_and_set_explicit(&dev->pool_busy, memory_order_acquire)) {
        owns_pool = true;
        use_pool = dev->pool;
        use_nw = dev->workers;
    }

    if (use_pool && use_nw > 1U) {
        velri_pool_parallel(use_pool, use_nw, cpu_op_worker, &ctx);
    } else {
        cpu_op_worker(&ctx, 0U, 1U);
    }

    if (owns_pool) {
        atomic_flag_clear_explicit(&dev->pool_busy, memory_order_release);
    }
    return VELRI_OK;
}

static const velri_backend_vtable g_cpu_vtable = {
    .name = "cpu",
    .contract_version = VELRI_BACKEND_CONTRACT_VERSION,
    .enumerate_devices = cpu_enumerate_devices,
    .open_device = cpu_open_device,
    .query_caps = cpu_query_caps,
    .close_device = cpu_close_device,
    .alloc = cpu_alloc,
    .free = cpu_free,
    .copy_to_ram = cpu_copy_to_ram,
    .copy_from_ram = cpu_copy_from_ram,
    .zero_mem = nullptr,
    .query_operation = cpu_query_operation,
    .prepare_op = nullptr,
    .destroy_prepared_op = nullptr,
    .submit = cpu_submit,
    .submit_batch = nullptr,
    .wait = nullptr,
    .destroy_completion = nullptr,
};

const velri_backend_vtable *velri_cpu_backend_vtable(void) { return &g_cpu_vtable; }
