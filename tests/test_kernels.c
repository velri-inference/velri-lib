#include "reference.h"
#include "velri_lib/velri_lib.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Unified Kernel Correctness Suite.
 *
 * Verifies ALL 12 operations and supported dtypes (including FP64 linear), aligned and odd
 * shapes, non-16B-aligned subviews, Split-K & CoopMat tiles, and numeric edge cases
 * (-inf softmax rows, -inf/-1e38 SiLU/GeLU, subnormal-adjacent quantize_row) through the
 * public Graph -> Plan -> Slot API across every available CPU ISA variant and Vulkan GPU.
 */

#define CHECK(cond, msg)                                                                          \
    do {                                                                                          \
        if (!(cond)) {                                                                            \
            fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, (msg));                     \
            exit(1);                                                                              \
        }                                                                                         \
    } while (0)

#define CHECK_OK(expr, msg)                                                                       \
    do {                                                                                          \
        velri_status _st = (expr);                                                                \
        if (_st != VELRI_OK) {                                                                    \
            fprintf(stderr, "FAIL [%s:%d]: %s (status=%d)\n", __FILE__, __LINE__, (msg),          \
                    (int)_st);                                                                    \
            exit(1);                                                                              \
        }                                                                                         \
    } while (0)

static void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n ? n : 1U, sz ? sz : 1U);
    CHECK(p != nullptr, "calloc");
    return p;
}

static velri_storage_id add_store(velri_graph *g, velri_storage_kind kind, velri_dtype dt,
                                  size_t elems, velri_device_id dev) {
    velri_storage_desc d = {
        .kind = kind,
        .dtype = dt,
        .byte_size = elems * velri_dtype_bytes(dt),
        .align = 64U,
        .device_id = dev,
    };
    velri_storage_id id = 0U;
    CHECK_OK(velri_graph_add_storage(g, &d, &id), "add_storage");
    return id;
}

static velri_view_id add_view_1d(velri_graph *g, velri_storage_id sid, velri_dtype dt, size_t n,
                                 size_t byte_off) {
    velri_tensor_view v;
    size_t dims[1] = {n};
    CHECK_OK(velri_tensor_view_contiguous(sid, dt, 1U, dims, byte_off, &v), "view1d");
    velri_view_id id = 0U;
    CHECK_OK(velri_graph_add_view(g, &v, &id), "add_view1d");
    return id;
}

static velri_view_id add_view_2d_strided(velri_graph *g, velri_storage_id sid, velri_dtype dt,
                                         size_t rows, size_t cols, size_t stride) {
    velri_tensor_view v = {
        .storage_id = sid,
        .dtype = dt,
        .rank = 2U,
        .dims = {rows, cols},
        .strides = {stride, 1U},
        .byte_offset = 0U,
    };
    velri_view_id id = 0U;
    CHECK_OK(velri_graph_add_view(g, &v, &id), "add_view2d");
    return id;
}

static velri_view_id add_view_3d_strided(velri_graph *g, velri_storage_id sid, velri_dtype dt,
                                         size_t d0, size_t d1, size_t d2, size_t s0, size_t s1) {
    velri_tensor_view v = {
        .storage_id = sid,
        .dtype = dt,
        .rank = 3U,
        .dims = {d0, d1, d2},
        .strides = {s0, s1, 1U},
        .byte_offset = 0U,
    };
    velri_view_id id = 0U;
    CHECK_OK(velri_graph_add_view(g, &v, &id), "add_view3d");
    return id;
}

typedef enum {
    LINEAR_RANDOM,
    LINEAR_F16_MATMUL,
    LINEAR_W8A8_EXTREMES,
    LINEAR_PACKED_W8A8_EXTREMES,
} linear_fixture;

/* Keep the other rows and columns random. The first two dot products have opposite signs
 * and both cross the signed 32-bit boundary at K >= 132105. */
static void prepare_w8a8_extremes(size_t rows, size_t m, size_t k, int8_t *x, size_t xs,
                                  int8_t *w, size_t ws, float *x_scale, float *w_scale,
                                  float *bias, float *expected, float *out, size_t ys) {
    memset(x, INT8_MIN, k);
    memset(w, INT8_MIN, k);
    memset(w + ws, INT8_MAX, k);
    for (size_t c = 0U; c < m; ++c) {
        w_scale[c] = 1.0f / (float)(1U << (c % 4U));
        if (bias) bias[c] = (float)((int)c - 3) * 0.25f;
    }
    for (size_t r = 0U; r < rows; ++r) {
        x_scale[r] = 1.0f / (float)(1U << (r % 3U));
        for (size_t c = 0U; c < m; ++c) {
            int64_t dot = 0;
            for (size_t i = 0U; i < k; ++i)
                dot += (int)x[r * xs + i] * (int)w[c * ws + i];
            const uint32_t low = (uint32_t)dot;
            const int64_t wrapped = (low <= INT32_MAX) ? (int64_t)low
                                                       : (int64_t)low - (INT64_C(1) << 32);
            expected[r * ys + c] = (float)wrapped * (x_scale[r] * w_scale[c]) +
                                   (bias ? bias[c] : 0.0f);
            out[r * ys + c] = NAN;
        }
        for (size_t c = m; c < ys; ++c)
            out[r * ys + c] = expected[r * ys + c] = 1234.5f;
    }
}

static void verify_linear_one(velri_context *ctx, velri_device_id dev, const char *dev_tag,
                              const char *label, velri_linear_desc desc, size_t rows, size_t m,
                              size_t k, size_t x_pad, size_t w_pad, size_t y_pad, bool with_bias,
                              double tol, linear_fixture fixture) {
    const bool allow_f16_matmul = fixture == LINEAR_F16_MATMUL;
    const bool pack = fixture == LINEAR_PACKED_W8A8_EXTREMES;
    const bool w8a8_extremes = pack || fixture == LINEAR_W8A8_EXTREMES;
    size_t xs = k + x_pad, ws = k + w_pad, ys = m + y_pad;
    size_t xb = velri_dtype_bytes(desc.x_dtype);
    size_t wb = velri_dtype_bytes(desc.w_dtype);
    size_t yb = velri_dtype_bytes(desc.y_dtype);

    velri_graph *g = nullptr;
    CHECK_OK(velri_graph_create(&g), "graph");
    velri_storage_id sx = add_store(g, VELRI_STORAGE_INPUT, desc.x_dtype, rows * xs, dev);
    velri_storage_id sw = add_store(g, VELRI_STORAGE_WEIGHT, desc.w_dtype, m * ws, dev);
    velri_storage_id sy = add_store(g, VELRI_STORAGE_OUTPUT, desc.y_dtype, rows * ys, dev);
    velri_storage_id sb = with_bias ? add_store(g, VELRI_STORAGE_WEIGHT, desc.y_dtype, m, dev) : 0U;
    velri_storage_id sxs = desc.has_x_scale ? add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, rows, dev) : 0U;
    velri_storage_id sws = desc.has_w_scale ? add_store(g, VELRI_STORAGE_WEIGHT, VELRI_DTYPE_F32, m, dev) : 0U;

    velri_op_desc op = {
        .kind = VELRI_OP_LINEAR,
        .num_inputs = 5U,
        .num_outputs = 1U,
        .inputs = {
            add_view_2d_strided(g, sx, desc.x_dtype, rows, k, xs),
            add_view_2d_strided(g, sw, desc.w_dtype, m, k, ws),
            with_bias ? add_view_1d(g, sb, desc.y_dtype, m, 0U) : VELRI_INVALID_ID,
            desc.has_x_scale ? add_view_1d(g, sxs, VELRI_DTYPE_F32, rows, 0U) : VELRI_INVALID_ID,
            desc.has_w_scale ? add_view_1d(g, sws, VELRI_DTYPE_F32, m, 0U) : VELRI_INVALID_ID,
        },
        .outputs = {add_view_2d_strided(g, sy, desc.y_dtype, rows, m, ys)},
        .attr.linear = {.desc = desc, .allow_f16_matmul = allow_f16_matmul},
    };
    CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "add linear");

    velri_plan *plan = nullptr;
    velri_status pst = pack ? velri_plan_prepare_ex(ctx, g, nullptr, VELRI_PLAN_PACK_CPU_WEIGHTS, &plan)
                            : velri_plan_prepare(ctx, g, nullptr, &plan);
    if (pst == VELRI_ERR_UNSUPPORTED && !w8a8_extremes) {
        velri_graph_destroy(g);
        return; /* Dtype combo intentionally unsupported on this device (e.g. FP64 on Vulkan) */
    }
    CHECK_OK(pst, label);

    void *hx = xcalloc(rows * xs, xb);
    void *hw = xcalloc(m * ws, wb);
    void *hy_ref = xcalloc(rows * ys, yb);
    void *hy = xcalloc(rows * ys, yb);
    void *hbias = with_bias ? xcalloc(m, yb) : nullptr;
    float *hxs = desc.has_x_scale ? (float *)xcalloc(rows, sizeof(float)) : nullptr;
    float *hws = desc.has_w_scale ? (float *)xcalloc(m, sizeof(float)) : nullptr;

    uint64_t rng = 0x55AA0001ULL ^ ((uint64_t)rows << 16) ^ ((uint64_t)m << 4) ^ (uint64_t)with_bias;
    velri_test_fill_dtype(desc.x_dtype, hx, rows * xs, &rng, 0.5f);
    velri_test_fill_dtype(desc.w_dtype, hw, m * ws, &rng, 0.5f);
    if (hbias) velri_test_fill_dtype(desc.y_dtype, hbias, m, &rng, 0.5f);
    if (hxs) velri_test_fill_dtype(VELRI_DTYPE_F32, hxs, rows, &rng, 0.05f);
    if (hws) velri_test_fill_dtype(VELRI_DTYPE_F32, hws, m, &rng, 0.05f);

    if (w8a8_extremes)
        prepare_w8a8_extremes(rows, m, k, hx, xs, hw, ws, hxs, hws, hbias, hy_ref, hy, ys);
    else
        ref_linear_run(&desc, rows, m, k, hx, xs, hxs, hw, ws, hws, hbias, hy_ref, ys);

    CHECK_OK(velri_plan_bind_weight_host(plan, sw, hw, m * ws * wb), "bind w");
    if (w8a8_extremes)
        CHECK(velri_plan_weight_is_packed(plan, sw) == pack, "W8A8 plan weight layout");
    if (with_bias) CHECK_OK(velri_plan_bind_weight_host(plan, sb, hbias, m * yb), "bind bias");
    if (desc.has_w_scale) {
        CHECK_OK(velri_plan_bind_weight_host(plan, sws, hws, m * sizeof(float)), "bind ws");
    }

    velri_slot *slot = nullptr;
    CHECK_OK(velri_slot_create(plan, &slot), "slot");
    CHECK_OK(velri_slot_bind_host(slot, sx, hx, rows * xs * xb), "bind x");
    CHECK_OK(velri_slot_bind_host(slot, sy, hy, rows * ys * yb), "bind y");
    if (desc.has_x_scale) {
        CHECK_OK(velri_slot_bind_host(slot, sxs, hxs, rows * sizeof(float)), "bind xs");
    }
    CHECK_OK(velri_slot_run(slot, nullptr, nullptr), label);
    CHECK(velri_slot_last_run_allocs(slot) == 0U, "0 hot-path allocs on linear");

    size_t checked_cols = w8a8_extremes ? ys : m;
    double err = velri_test_max_abs_diff_2d(desc.y_dtype, hy_ref, hy, rows, checked_cols, ys);
    if (allow_f16_matmul) {
        velri_fp16 *x16 = xcalloc(rows * xs, sizeof(*x16));
        velri_fp16 *w16 = xcalloc(m * ws, sizeof(*w16));
        for (size_t i = 0U; i < rows * xs; ++i) x16[i] = velri_f32_to_f16_scalar(((float *)hx)[i]);
        for (size_t i = 0U; i < m * ws; ++i) w16[i] = velri_f32_to_f16_scalar(((float *)hw)[i]);
        float *mixed_ref = xcalloc(rows * ys, sizeof(*mixed_ref));
        velri_linear_desc half = velri_desc_f16(VELRI_DTYPE_F32);
        ref_linear_run(&half, rows, m, k, x16, xs, nullptr, w16, ws, nullptr, hbias, mixed_ref, ys);
        CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, mixed_ref, hy_ref, rows, m, ys) > 2.0 * tol,
              "mixed GEMM fixture must distinguish full and half operands");
        double mixed_err = velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, mixed_ref, hy, rows, m, ys);
        if (mixed_err <= tol) {
            printf("  [PASS] %s %s: F16 operands, F32 accumulation (max abs %.3g)\n",
                   dev_tag, label, mixed_err);
            err = mixed_err;
        } else if (err <= tol) {
            printf("  [SKIP] %s %s: optional F16 GEMM unavailable; F32 fallback verified\n",
                   dev_tag, label);
        }
        free(mixed_ref);
        free(w16);
        free(x16);
    }
    if (!(err <= tol)) {
        fprintf(stderr, "FAIL [%s] %s (rows=%zu m=%zu k=%zu bias=%d): err=%.3e > %.3e\n",
                dev_tag, label, rows, m, k, (int)with_bias, err, tol);
        exit(1);
    }

    velri_slot_destroy(slot);
    velri_plan_destroy(plan);
    velri_graph_destroy(g);
    free(hx); free(hw); free(hy_ref); free(hy); free(hbias); free(hxs); free(hws);
}

static void verify_w8a8_weight_packing(velri_context *ctx, velri_device_id dev,
                                      velri_cpu_variant variant) {
    enum { rows = 29, m = 131, k = 2051, xs = k + 2, ws = k + 4, ys = m + 6 };
    velri_graph *g = nullptr;
    CHECK_OK(velri_graph_create(&g), "packed W8A8 graph");
    velri_storage_id sx = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_I8, rows * xs, dev);
    enum { w_align = 4096 };
    velri_storage_desc weight_desc = {VELRI_STORAGE_WEIGHT, VELRI_DTYPE_I8, m * ws, w_align, dev};
    velri_storage_id sw;
    CHECK_OK(velri_graph_add_storage(g, &weight_desc, &sw), "aligned W8A8 weights");
    velri_storage_id sb = add_store(g, VELRI_STORAGE_WEIGHT, VELRI_DTYPE_F32, m, dev);
    velri_storage_id sxs = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, rows, dev);
    velri_storage_id sws = add_store(g, VELRI_STORAGE_WEIGHT, VELRI_DTYPE_F32, m, dev);
    velri_storage_id sy = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, rows * ys, dev);
    velri_op_desc op = {
        .kind = VELRI_OP_LINEAR, .num_inputs = 5U, .num_outputs = 1U,
        .inputs = {add_view_2d_strided(g, sx, VELRI_DTYPE_I8, rows, k, xs),
                   add_view_2d_strided(g, sw, VELRI_DTYPE_I8, m, k, ws),
                   add_view_1d(g, sb, VELRI_DTYPE_F32, m, 0U),
                   add_view_1d(g, sxs, VELRI_DTYPE_F32, rows, 0U),
                   add_view_1d(g, sws, VELRI_DTYPE_F32, m, 0U)},
        .outputs = {add_view_2d_strided(g, sy, VELRI_DTYPE_F32, rows, m, ys)},
        .dynamic_seq = true, .attr.linear.desc = velri_desc_w8a8_f32(),
    };
    CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "packed W8A8 op");
    velri_plan *plan = nullptr;
    CHECK_OK(velri_plan_prepare_ex(ctx, g, nullptr, VELRI_PLAN_PACK_CPU_WEIGHTS, &plan),
             "packed W8A8 plan");
    const velri_plan_options share_opts = {.share_weights_from = plan};
    velri_plan *peer = nullptr;
    CHECK_OK(velri_plan_prepare(ctx, g, &share_opts, &peer), "shared W8A8 plan");
    velri_memory_report peer_memory;
    CHECK_OK(velri_plan_memory_report(peer, &peer_memory), "shared W8A8 memory report");
    CHECK(peer_memory.devices[dev].weight_bytes == 0 &&
          velri_plan_weight_is_packed(peer, sw), "shared W8A8 layout");
    velri_slot *slot = nullptr;
    CHECK_OK(velri_slot_create(peer, &slot), "packed W8A8 slot");

    int8_t *x = xcalloc(rows * xs, sizeof(*x));
    float *expected = xcalloc(rows * ys, sizeof(*expected)), *y = xcalloc(rows * ys, sizeof(*y));
    float x_scale[rows], w_scale[m], bias[m];
    uint64_t rng = 0xF8A80001U;
    velri_test_fill_dtype(VELRI_DTYPE_I8, x, rows * xs, &rng, 1.0f);
    CHECK_OK(velri_slot_bind_host(slot, sx, x, rows * xs), "packed W8A8 x");
    CHECK_OK(velri_slot_bind_host(slot, sxs, x_scale, sizeof(x_scale)), "packed W8A8 x scale");
    CHECK_OK(velri_slot_bind_host(slot, sy, y, rows * ys * sizeof(*y)), "packed W8A8 output");
    int8_t *weights[2] = {xcalloc(m * ws, sizeof(int8_t)), xcalloc(m * ws, sizeof(int8_t))};
    for (unsigned bind = 0; bind < 2U; ++bind) {
        int8_t *w = weights[bind];
        velri_test_fill_dtype(VELRI_DTYPE_I8, w, m * ws, &rng, 1.0f);
        prepare_w8a8_extremes(rows, m, k, x, xs, w, ws, x_scale, w_scale, bias, expected, y, ys);
        CHECK_OK(velri_plan_bind_weight_host(plan, sw, w, m * ws), "bind or replace W8A8 weights");
        CHECK_OK(velri_plan_bind_weight_host(plan, sws, w_scale, sizeof(w_scale)), "W8A8 w scale");
        CHECK_OK(velri_plan_bind_weight_host(plan, sb, bias, sizeof(bias)), "W8A8 bias");
        CHECK(velri_plan_weight_is_packed(plan, sw), "packed W8A8 binding");
        CHECK(!velri_plan_weight_is_packed(plan, sws), "W8A8 scales keep their original layout");
        memset(w, 0, m * ws);
        CHECK(velri_plan_bind_weight_host(plan, sw, w, m * ws - 1U) == VELRI_ERR_INVALID,
              "undersized W8A8 rebind rejected");

        velri_memory_report rep;
        CHECK_OK(velri_plan_memory_report(plan, &rep), "W8A8 memory report");
        size_t packed_m = (m + 31U) / 32U * 32U;
        size_t matrix_bytes = packed_m * k;
        size_t padding = (w_align - matrix_bytes % w_align) % w_align;
        CHECK(rep.devices[dev].weight_bytes == matrix_bytes + sizeof(w_scale) + sizeof(bias) &&
              rep.devices[dev].allocator_overhead_bytes == padding,
              "CPU memory report counts one native matrix and its allocation padding");

        const size_t active[] = {1U, 2U, 3U, rows};
        for (size_t i = 0U; i < sizeof(active) / sizeof(active[0]); ++i) {
            const float untouched = -1234.5f;
            for (size_t j = 0; j < rows * ys; ++j) y[j] = untouched;
            velri_step_params step = {.has_seq_len = true, .seq_len_override = active[i]};
            CHECK_OK(velri_slot_run(slot, &step, nullptr), "W8A8 decode or prefill");
            CHECK(velri_slot_last_run_allocs(slot) == 0U, "W8A8 hot path allocation count");
            for (size_t r = 0; r < rows; ++r)
                for (size_t c = 0; c < ys; ++c) {
                    float want = r < active[i] && c < m ? expected[r * ys + c] : untouched;
                    if (y[r * ys + c] != want) {
                        fprintf(stderr, "FAIL W8A8 variant=%d bind=%u rows=%zu at [%zu,%zu]: "
                                        "got %g expected %g\n", (int)variant, bind, active[i],
                                r, c, y[r * ys + c], want);
                        exit(1);
                    }
                }
        }
    }
    /* A second, overlapping view of the same bytes requires the original row-major layout. */
    const velri_tensor_view offset_w = {.storage_id = sw, .dtype = VELRI_DTYPE_I8, .rank = 2,
                                       .dims = {m, k}, .strides = {ws, 1}, .byte_offset = 1};
    CHECK_OK(velri_graph_add_view(g, &offset_w, &op.inputs[1]), "overlapping W8A8 view");
    CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "second W8A8 projection");
    velri_plan *mixed = nullptr;
    CHECK_OK(velri_plan_prepare_ex(ctx, g, nullptr, VELRI_PLAN_PACK_CPU_WEIGHTS, &mixed),
             "W8A8 shared across incompatible views");
    CHECK(!velri_plan_weight_is_packed(mixed, sw), "mixed views borrow original weights");
    velri_plan_destroy(mixed);
    CHECK(velri_plan_prepare(ctx, g, &share_opts, &mixed) == VELRI_ERR_INVALID && !mixed,
          "borrower cannot reinterpret packed bytes with an incompatible view");

    velri_plan_destroy(peer);
    velri_slot_destroy(slot);
    velri_plan_destroy(plan);
    velri_graph_destroy(g);
    free(weights[0]); free(weights[1]); free(x); free(y); free(expected);
}

// The mixed kernel receives original F32 projections. Round only the reference K/V
// after submitting the step so both paths independently produce their cache values.
static void round_attention_kv(float *qkv, size_t rows, size_t stride, size_t q_dim) {
    for (size_t row = 0U; row < rows; ++row) {
        for (size_t col = q_dim; col < stride; ++col) {
            float *value = qkv + row * stride + col;
            *value = velri_f16_to_f32_scalar(velri_f32_to_f16_scalar(*value));
        }
    }
}

static void verify_attention_case(velri_context *ctx, velri_device_id dev, const char *dev_tag,
                                  const char *label, uint32_t nh, uint32_t nkvh, uint32_t hd,
                                  size_t max_t, size_t max_ctx, size_t prefill_t,
                                  size_t num_decode_steps, uint32_t sliding_window, bool causal,
                                  bool same_kv_cache, size_t cache_pad, bool inplace_q, uint64_t seed,
                                  velri_dtype cache_type, bool allow_f16_matmul) {
    const size_t q_dim = (size_t)nh * hd;
    const size_t kv_dim = (size_t)nkvh * hd;
    const size_t qkv_stride = same_kv_cache ? (q_dim + kv_dim) : (q_dim + 2U * kv_dim);

    const size_t cache_stride = hd + cache_pad;
    const size_t output_stride = inplace_q ? qkv_stride : q_dim + (cache_pad ? 1U : 0U);
    uint64_t rng = seed;

    velri_graph *g = nullptr;
    CHECK_OK(velri_graph_create(&g), "attn g");
    velri_storage_id sqkv =
        add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, max_t * qkv_stride, dev);
    velri_storage_id so = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, max_t * output_stride, dev);
    velri_storage_id skc =
        add_store(g, VELRI_STORAGE_STATE, cache_type, (size_t)nkvh * max_ctx * cache_stride, dev);
    velri_storage_id svc =
        same_kv_cache
            ? skc
            : add_store(g, VELRI_STORAGE_STATE, cache_type, (size_t)nkvh * max_ctx * cache_stride, dev);

    velri_tensor_view base_qkv, qv, kv;
    size_t base_dims[2] = {max_t, qkv_stride};
    size_t q_dims[2] = {max_t, q_dim}, q_str[2] = {qkv_stride, 1U};
    size_t kv_dims[2] = {max_t, kv_dim}, kv_str[2] = {qkv_stride, 1U};

    CHECK_OK(velri_tensor_view_contiguous(sqkv, VELRI_DTYPE_F32, 2U, base_dims, 0U, &base_qkv),
             "attn base");
    CHECK_OK(velri_tensor_view_subview(&base_qkv, 2U, q_dims, q_str, 0U, &qv), "attn qv");
    CHECK_OK(velri_tensor_view_subview(&base_qkv, 2U, kv_dims, kv_str, q_dim * sizeof(float), &kv),
             "attn kv");

    velri_view_id id_q = 0U, id_k = 0U, id_v = 0U, id_kc = 0U, id_vc = 0U;
    CHECK_OK(velri_graph_add_view(g, &qv, &id_q), "add qv");
    CHECK_OK(velri_graph_add_view(g, &kv, &id_k), "add kv");
    id_kc = add_view_3d_strided(g, skc, cache_type, nkvh, max_ctx, hd,
                                max_ctx * cache_stride, cache_stride);
    if (same_kv_cache) {
        id_v = id_k;
        id_vc = id_kc;
    } else {
        velri_tensor_view vv;
        CHECK_OK(velri_tensor_view_subview(&base_qkv, 2U, kv_dims, kv_str,
                                           (q_dim + kv_dim) * sizeof(float), &vv),
                 "attn vv");
        CHECK_OK(velri_graph_add_view(g, &vv, &id_v), "add vv");
        id_vc = add_view_3d_strided(g, svc, cache_type, nkvh, max_ctx, hd,
                                    max_ctx * cache_stride, cache_stride);
    }

    velri_view_id output = add_view_2d_strided(g, so, VELRI_DTYPE_F32, max_t, q_dim, output_stride);
    if (inplace_q) {
        velri_op_desc seed_query = {
            .kind = VELRI_OP_VEC_AXPBY, .num_inputs = 1U, .num_outputs = 1U,
            .inputs = {add_view_1d(g, sqkv, VELRI_DTYPE_F32, max_t * qkv_stride, 0U)},
            .outputs = {add_view_1d(g, so, VELRI_DTYPE_F32, max_t * output_stride, 0U)},
            .attr.axpby = {.alpha = 1.0f},
        };
        CHECK_OK(velri_graph_add_op(g, &seed_query, dev, nullptr), "seed in-place Q");
        id_q = output;
    }
    velri_op_desc op = {
        .kind = VELRI_OP_ATTENTION,
        .num_inputs = 3U,
        .num_outputs = 3U,
        .inputs = {id_q, id_k, id_v},
        .outputs = {output, id_kc, id_vc},
        .attr.attention = {
            .n_heads = nh,
            .n_kv_heads = nkvh,
            .head_dim = hd,
            .scale = 0.0f,
            .sliding_window = sliding_window,
            .causal = causal,
            .allow_f16_matmul = allow_f16_matmul,
        },
    };
    CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), label);

    velri_plan *plan = nullptr;
    velri_slot *slot = nullptr;
    velri_status pst = velri_plan_prepare(ctx, g, nullptr, &plan);
    if (cache_type == VELRI_DTYPE_F16 && pst == VELRI_ERR_UNSUPPORTED) {
        velri_graph_destroy(g);
        return;
    }
    CHECK_OK(pst, label);
    float *hqkv = (float *)xcalloc(max_t * qkv_stride, sizeof(float));
    float *ho = (float *)xcalloc(max_t * output_stride, sizeof(float));
    float *ho_ref = (float *)xcalloc(max_t * output_stride, sizeof(float));
    float *hkc_ref = (float *)xcalloc((size_t)nkvh * max_ctx * cache_stride, sizeof(float));
    float *hvc_ref =
        same_kv_cache ? hkc_ref : (float *)xcalloc((size_t)nkvh * max_ctx * cache_stride, sizeof(float));
    CHECK_OK(velri_slot_create(plan, &slot), "attn slot");
    CHECK_OK(velri_slot_bind_host(slot, sqkv, hqkv, max_t * qkv_stride * sizeof(float)), "qkv");
    CHECK_OK(velri_slot_bind_host(slot, so, ho, max_t * output_stride * sizeof(float)), "o");

    const float *k_in_ptr = hqkv + q_dim;
    const float *v_in_ptr = same_kv_cache ? k_in_ptr : (hqkv + q_dim + kv_dim);

    // Decode-only fixtures upload a known KV history so long split boundaries can
    // be checked against the scalar reference without a quadratic reference prefill.
    void *initial_cache[2] = {nullptr, nullptr};
    const bool seed_history = max_t == 1U && prefill_t > 0U;
    if (seed_history) {
        size_t elements = (size_t)nkvh * max_ctx * cache_stride;
        size_t elem_bytes = velri_dtype_bytes(cache_type);
        for (unsigned i = 0U; i < (same_kv_cache ? 1U : 2U); ++i) {
            float *reference = i ? hvc_ref : hkc_ref;
            velri_storage_id store = i ? svc : skc;
            initial_cache[i] = xcalloc(elements, elem_bytes);
            velri_test_fill_dtype(cache_type, initial_cache[i], elements, &rng, 1.0f);
            for (size_t j = 0U; j < elements; ++j)
                reference[j] = (float)velri_test_load_f64(cache_type, initial_cache[i], j);
            CHECK_OK(velri_slot_bind_host(slot, store, initial_cache[i], elements * elem_bytes), label);
            CHECK_OK(velri_slot_sync_state_from_host(slot, store), label);
        }
    }
    for (size_t pos = seed_history ? prefill_t : 0U; pos < prefill_t;) {
        size_t rows = prefill_t - pos < max_t ? prefill_t - pos : max_t;
        velri_test_fill_dtype(VELRI_DTYPE_F32, hqkv, rows * qkv_stride, &rng, 1.0f);
        velri_step_params step = {
            .has_seq_len = true, .seq_len_override = rows,
            .has_kv_len = true, .kv_len_override = pos,
        };
        CHECK_OK(velri_slot_run(slot, &step, nullptr), label);
        CHECK(velri_slot_last_run_allocs(slot) == 0U, "attn prefill 0 allocs");
        if (cache_type == VELRI_DTYPE_F16) round_attention_kv(hqkv, rows, qkv_stride, q_dim);
        velri_ref_attention_ex_f32(rows, nh, nkvh, hd, hd, pos, max_ctx, 0.0f, sliding_window,
                                  causal, hqkv, qkv_stride, k_in_ptr, qkv_stride, v_in_ptr,
                                  qkv_stride, hkc_ref, max_ctx * cache_stride, cache_stride,
                                  hvc_ref, max_ctx * cache_stride, cache_stride, ho_ref, output_stride);
        double err = velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, ho_ref, ho, rows, q_dim, output_stride);
        const double tolerance = allow_f16_matmul ? 1.5e-3 : 5e-5;
        if (!(err <= tolerance)) {
            fprintf(stderr, "FAIL [%s] %s prefill(pos=%zu, rows=%zu): err=%.3e > %.3e\n",
                    dev_tag, label, pos, rows, err, tolerance);
            exit(1);
        }
        if (strcmp(dev_tag, "vulkan:0") == 0 && causal && !same_kv_cache &&
            (rows >= 128U || (rows >= 32U && pos >= 256U))) {
            // A tile may read a future K/V for another query. The mask must exclude NaNs
            // from earlier rows and retain them once the poisoned key becomes visible.
            const size_t row = rows - rows % 4U - 1U;
            float *key = hqkv + row * qkv_stride + q_dim;
            for (size_t kind = 0U; kind < 2U; ++kind) {
                float *poison = kind == 0U ? key : key + kv_dim;
                float saved = *poison;
                *poison = NAN;
                CHECK_OK(velri_slot_run(slot, &step, nullptr), "masked NaN prefill");
                for (size_t i = 0U; i < row; ++i)
                    for (size_t j = 0U; j < q_dim; ++j)
                        CHECK(fabsf(ho[i * output_stride + j] - ho_ref[i * output_stride + j]) <= tolerance,
                              "masked K/V affected an earlier row");
                CHECK(isnan(ho[row * output_stride]), "visible NaN was dropped");
                *poison = saved;
            }
            if (cache_type == VELRI_DTYPE_F32 && hd == 64U && rows >= 512U &&
                pos == 0U && sliding_window == 0U) {
                // Row 32 has one finite score after an entire tile of negative infinities.
                float *query = hqkv + 32U * qkv_stride;
                float saved_query = query[0], saved_keys[32];
                query[0] = 1.0f;
                for (size_t t = 0U; t < 32U; ++t) {
                    float *k = hqkv + t * qkv_stride + q_dim;
                    saved_keys[t] = k[0];
                    k[0] = -INFINITY;
                }
                CHECK_OK(velri_slot_run(slot, &step, nullptr), "leading negative-infinity scores");
                for (size_t c = 0U; c < hd; ++c)
                    CHECK(fabsf(ho[32U * output_stride + c] - v_in_ptr[32U * qkv_stride + c]) <= tolerance,
                          "first finite score must select its value");
                for (size_t t = 0U; t < 32U; ++t) hqkv[t * qkv_stride + q_dim] = saved_keys[t];
                query[0] = saved_query;
            }
            CHECK_OK(velri_slot_run(slot, &step, nullptr), "restore finite cache before decode");
        }
        pos += rows;
    }

    /* Subsequent 1-token decode steps reading resident device KV state (0 B shadow RAM) */
    for (size_t ds = 0U; ds < num_decode_steps; ++ds) {
        size_t kv_pos = prefill_t + ds;
        velri_test_fill_dtype(VELRI_DTYPE_F32, hqkv, qkv_stride, &rng, 1.0f);
        velri_step_params sp_dec = {
            .has_seq_len = true,
            .seq_len_override = 1U,
            .has_kv_len = true,
            .kv_len_override = kv_pos,
        };
        CHECK_OK(velri_slot_run(slot, &sp_dec, nullptr), label);
        CHECK(velri_slot_last_run_allocs(slot) == 0U, "attn decode 0 allocs");
        if (strcmp(dev_tag, "vulkan:0") == 0) {
            /* Verifies persistent graph replay works BOTH below AND above Split-KV threshold (H5)! */
            CHECK(velri_slot_last_run_graph_replays(slot) == 1U,
                  "Vulkan decode (incl. Split-KV buckets) must replay persistent captured graph");
        }
        if (cache_type == VELRI_DTYPE_F16) round_attention_kv(hqkv, 1U, qkv_stride, q_dim);
        velri_ref_attention_ex_f32(1U, nh, nkvh, hd, hd, kv_pos, max_ctx, 0.0f, sliding_window, causal,
                                hqkv, qkv_stride, k_in_ptr, qkv_stride, v_in_ptr, qkv_stride,
                                hkc_ref, max_ctx * cache_stride, cache_stride, hvc_ref,
                                max_ctx * cache_stride, cache_stride, ho_ref, output_stride);
        double err = velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, ho_ref, ho, 1U, q_dim, output_stride);
        if (!(err <= 5e-5)) {
            fprintf(stderr, "FAIL [%s] %s decode(kv_pos=%zu): err=%.3e > 5e-5\n", dev_tag, label,
                    kv_pos, err);
            exit(1);
        }
    }

    velri_slot_destroy(slot);
    free(initial_cache[0]);
    free(initial_cache[1]);
    velri_plan_destroy(plan);
    velri_graph_destroy(g);
    free(hqkv);
    free(ho);
    free(ho_ref);
    free(hkc_ref);
    if (!same_kv_cache) {
        free(hvc_ref);
    }
}

/* A residual subview can overlap the input of a preceding projection, even though each
 * individual op has legal input/output aliases. Also read the public projection output. */
static void verify_gemv_residual_outputs(velri_context *ctx, velri_device_id dev, bool overlap_x) {
    enum { dim = 2048, shift = 4 };
    velri_graph *g = nullptr;
    CHECK_OK(velri_graph_create(&g), "residual graph");
    velri_storage_id x = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, dim + shift, dev);
    velri_storage_id w = add_store(g, VELRI_STORAGE_WEIGHT, VELRI_DTYPE_F32, dim * dim, dev);
    velri_storage_id y = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, dim, dev);
    velri_storage_id r = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, dim + shift, dev);
    velri_view_id r_slice = add_view_1d(g, r, VELRI_DTYPE_F32, dim, shift * sizeof(float));
    velri_op_desc copy = {
        .kind = VELRI_OP_VEC_AXPBY, .num_inputs = 1U, .num_outputs = 1U,
        .inputs = {add_view_1d(g, x, VELRI_DTYPE_F32, dim + shift, 0)},
        .outputs = {add_view_1d(g, r, VELRI_DTYPE_F32, dim + shift, 0)},
        .attr.axpby = {.alpha = 1.0f},
    };
    velri_op_desc projection = {
        .kind = VELRI_OP_LINEAR, .num_inputs = 2U, .num_outputs = 1U,
        .inputs = {add_view_2d_strided(g, overlap_x ? r : x, VELRI_DTYPE_F32, 1, dim, dim),
                   add_view_2d_strided(g, w, VELRI_DTYPE_F32, dim, dim, dim)},
        .outputs = {add_view_2d_strided(g, y, VELRI_DTYPE_F32, 1, dim, dim)},
        .attr.linear.desc = velri_desc_f32(),
    };
    velri_op_desc add = {
        .kind = VELRI_OP_VEC_AXPBY, .num_inputs = 2U, .num_outputs = 1U,
        .inputs = {r_slice, add_view_1d(g, y, VELRI_DTYPE_F32, dim, 0)},
        .outputs = {r_slice}, .attr.axpby = {.alpha = 1.0f, .beta = 1.0f},
    };
    CHECK_OK(velri_graph_add_op(g, &copy, dev, nullptr), "residual seed");
    CHECK_OK(velri_graph_add_op(g, &projection, dev, nullptr), "residual projection");
    CHECK_OK(velri_graph_add_op(g, &add, dev, nullptr), "residual add");
    velri_plan *plan = nullptr;
    velri_slot *slot = nullptr;
    float *weights = xcalloc(dim * dim, sizeof(float));
    float input[dim + shift], projected[dim], residual[dim + shift];
    for (size_t i = 0; i < dim * dim; ++i) weights[i] = 1.0f;
    for (size_t i = 0; i < dim + shift; ++i) input[i] = 1.0f;
    CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "residual plan");
    CHECK_OK(velri_plan_bind_weight_host(plan, w, weights, dim * dim * sizeof(float)), "residual weight");
    CHECK_OK(velri_slot_create(plan, &slot), "residual slot");
    CHECK_OK(velri_slot_bind_host(slot, x, input, sizeof(input)), "residual input");
    CHECK_OK(velri_slot_bind_host(slot, y, projected, sizeof(projected)), "projection output");
    CHECK_OK(velri_slot_bind_host(slot, r, residual, sizeof(residual)), "residual output");
    CHECK_OK(velri_slot_run(slot, nullptr, nullptr), "run projection and residual");
    for (size_t i = 0; i < dim; ++i) {
        CHECK(projected[i] == (float)dim, "projection reads original activation");
        CHECK(residual[shift + i] == (float)(dim + 1), "residual uses original projection");
    }
    for (size_t i = 0; i < shift; ++i) CHECK(residual[i] == 1.0f, "residual subview boundary");
    velri_slot_destroy(slot);
    velri_plan_destroy(plan);
    velri_graph_destroy(g);
    free(weights);
}

/* Normalization, the up projection and the final gated result are all public outputs.
 * An in-place normalization has to finish reading x before any workgroup replaces x. */
static void verify_rms_gated_outputs(velri_context *ctx, velri_device_id dev, bool inplace) {
    enum { k = 260, m = 8195 };
    const float eps = 1e-5f, clamp = 1.5f;
    const velri_dtype dt = VELRI_DTYPE_F32;
    velri_graph *g = nullptr;
    CHECK_OK(velri_graph_create(&g), "RMS and gated graph");
    velri_storage_id x = add_store(g, VELRI_STORAGE_INPUT, dt, k, dev);
    velri_storage_id gamma = add_store(g, VELRI_STORAGE_WEIGHT, dt, k, dev);
    velri_storage_id norm = add_store(g, VELRI_STORAGE_OUTPUT, dt, k, dev);
    velri_view_id vx = add_view_2d_strided(g, x, dt, 1, k, k);
    velri_view_id vn = add_view_2d_strided(g, norm, dt, 1, k, k);
    if (inplace) {
        velri_op_desc seed = {.kind = VELRI_OP_VEC_AXPBY, .num_inputs = 1U, .num_outputs = 1U,
                              .inputs = {vx}, .outputs = {vn}, .attr.axpby = {.alpha = 1.0f}};
        CHECK_OK(velri_graph_add_op(g, &seed, dev, nullptr), "seed in-place norm");
    }
    velri_op_desc rms = {
        .kind = VELRI_OP_RMSNORM, .num_inputs = 2U, .num_outputs = 1U,
        .inputs = {inplace ? vn : vx, add_view_1d(g, gamma, dt, k, 0)}, .outputs = {vn},
        .attr.rmsnorm = {.eps = eps},
    };
    CHECK_OK(velri_graph_add_op(g, &rms, dev, nullptr), "add RMS");
    velri_storage_id weights[2], outputs[2];
    velri_view_id projections[2];
    velri_linear_desc desc = velri_desc_f32();
    for (int i = 0; i < 2; ++i) {
        weights[i] = add_store(g, VELRI_STORAGE_WEIGHT, dt, m * k, dev);
        outputs[i] = add_store(g, VELRI_STORAGE_OUTPUT, dt, m, dev);
        projections[i] = add_view_2d_strided(g, outputs[i], dt, 1, m, m);
        velri_op_desc projection = {
            .kind = VELRI_OP_LINEAR, .num_inputs = 2U, .num_outputs = 1U,
            .inputs = {vn, add_view_2d_strided(g, weights[i], dt, m, k, k)},
            .outputs = {projections[i]}, .attr.linear.desc = desc,
        };
        CHECK_OK(velri_graph_add_op(g, &projection, dev, nullptr), "add gated projection");
    }
    velri_op_desc gated = {
        .kind = VELRI_OP_GATED_MLP, .num_inputs = 2U, .num_outputs = 1U,
        .inputs = {projections[0], projections[1]}, .outputs = {projections[0]},
        .attr.gated_mlp = {.act = VELRI_ACT_SILU, .clamp_limit = clamp},
    };
    CHECK_OK(velri_graph_add_op(g, &gated, dev, nullptr), "add gated SiLU");
    velri_plan *plan = nullptr;
    velri_slot *slot = nullptr;
    uint64_t rng = 0xE99311U;
    float hx[k], hgamma[k], hnorm[k], expected_norm[k], hgated[m];
    float projected[2][m], expected[2][m];
    velri_test_fill_dtype(dt, hx, k, &rng, 0.25f);
    velri_test_fill_dtype(dt, hgamma, k, &rng, 2.0f);
    ref_rmsnorm(dt, dt, 1, k, eps, false, hx, k, hgamma, expected_norm, k);
    CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prepare RMS gated");
    CHECK_OK(velri_plan_bind_weight_host(plan, gamma, hgamma, sizeof(hgamma)), "RMS gamma");
    float *hw[2] = {xcalloc(m * k, sizeof(float)), xcalloc(m * k, sizeof(float))};
    for (int i = 0; i < 2; ++i) {
        velri_test_fill_dtype(dt, hw[i], m * k, &rng, 0.25f);
        ref_linear_run(&desc, 1, m, k, expected_norm, k, nullptr, hw[i], k, nullptr, nullptr,
                       expected[i], m);
        CHECK_OK(velri_plan_bind_weight_host(plan, weights[i], hw[i], m * k * sizeof(float)),
                 "bind gated weight");
    }
    ref_gated_mlp_f32(VELRI_ACT_SILU, m, clamp, expected[0], expected[1], hgated);
    CHECK_OK(velri_slot_create(plan, &slot), "RMS gated slot");
    CHECK_OK(velri_slot_bind_host(slot, x, hx, sizeof(hx)), "RMS input");
    CHECK_OK(velri_slot_bind_host(slot, norm, hnorm, sizeof(hnorm)), "normalized output");
    for (int i = 0; i < 2; ++i) {
        CHECK_OK(velri_slot_bind_host(slot, outputs[i], projected[i], sizeof(projected[i])),
                 "gated output");
    }
    CHECK_OK(velri_slot_run(slot, nullptr, nullptr), "run RMS gated");
    CHECK(velri_test_max_abs_diff_2d(dt, expected_norm, hnorm, 1, k, k) < 3e-5,
          "normalized activation remains observable");
    CHECK(velri_test_max_abs_diff_2d(dt, expected[1], projected[1], 1, m, m) < 2e-4,
          "unclamped up projection remains observable");
    CHECK(velri_test_max_abs_diff_2d(dt, hgated, projected[0], 1, m, m) < 2e-4,
          "gated projection applies clamp and SiLU");
    velri_slot_destroy(slot);
    velri_plan_destroy(plan);
    velri_graph_destroy(g);
    free(hw[0]);
    free(hw[1]);
}

/* Reuse one prepared prompt graph across all tail sizes. The padded output rows
 * exercise scalar epilogues, and 67 columns force a partial weight tile. */
static void verify_dynamic_prompt_linear(velri_context *ctx, velri_device_id dev) {
    enum { rows = 64, m = 67, k = 96, xs = 100, ws = 100, ys = 70 };
    const velri_dtype dt = VELRI_DTYPE_F32;
    const velri_linear_desc desc = velri_desc_f32();
    velri_graph *g = nullptr;
    CHECK_OK(velri_graph_create(&g), "dynamic prompt graph");
    velri_storage_id x = add_store(g, VELRI_STORAGE_INPUT, dt, rows * xs, dev);
    velri_storage_id w = add_store(g, VELRI_STORAGE_WEIGHT, dt, m * ws, dev);
    velri_storage_id bias = add_store(g, VELRI_STORAGE_WEIGHT, dt, m, dev);
    velri_storage_id y = add_store(g, VELRI_STORAGE_OUTPUT, dt, rows * ys, dev);
    velri_op_desc op = {
        .kind = VELRI_OP_LINEAR, .num_inputs = 3U, .num_outputs = 1U,
        .inputs = {add_view_2d_strided(g, x, dt, rows, k, xs), add_view_2d_strided(g, w, dt, m, k, ws),
                   add_view_1d(g, bias, dt, m, 0U)},
        .outputs = {add_view_2d_strided(g, y, dt, rows, m, ys)},
        .dynamic_seq = true, .attr.linear.desc = desc,
    };
    CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "add dynamic prompt linear");
    float hx[rows * xs], hw[m * ws], hb[m], hy[rows * ys], expected[rows * ys];
    float zero[rows * xs] = {0.0f};
    uint64_t rng = 0xFE321U;
    velri_test_fill_dtype(dt, hx, rows * xs, &rng, 0.5f);
    velri_test_fill_dtype(dt, hw, m * ws, &rng, 0.5f);
    velri_test_fill_dtype(dt, hb, m, &rng, 0.5f);
    ref_linear_run(&desc, rows, m, k, hx, xs, nullptr, hw, ws, nullptr, hb, expected, ys);
    velri_plan *plan = nullptr;
    velri_slot *slot = nullptr;
    CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "dynamic prompt plan");
    CHECK_OK(velri_plan_bind_weight_host(plan, w, hw, sizeof(hw)), "dynamic prompt weights");
    CHECK_OK(velri_plan_bind_weight_host(plan, bias, hb, sizeof(hb)), "dynamic prompt bias");
    CHECK_OK(velri_slot_create(plan, &slot), "dynamic prompt slot");
    CHECK_OK(velri_slot_bind_host(slot, x, zero, sizeof(zero)), "zero prompt input");
    CHECK_OK(velri_slot_bind_host(slot, y, hy, sizeof(hy)), "dynamic prompt output");
    CHECK_OK(velri_slot_run(slot, nullptr, nullptr), "initialize all prompt rows to the bias");
    CHECK_OK(velri_slot_bind_host(slot, x, hx, sizeof(hx)), "dynamic prompt input");
    for (size_t n = 1U; n <= rows; ++n) {
        const velri_step_params step = {.has_seq_len = true, .seq_len_override = (uint32_t)n};
        CHECK_OK(velri_slot_run(slot, &step, nullptr), "dynamic prompt step");
        CHECK(velri_test_max_abs_diff_2d(dt, expected, hy, n, m, ys) <= 2e-4,
              "dynamic prompt projection and bias match the reference");
        for (size_t r = n; r < rows; ++r) {
            CHECK(memcmp(&hy[r * ys], hb, sizeof(hb)) == 0, "inactive prompt rows stay untouched");
        }
    }
    velri_slot_destroy(slot);
    velri_plan_destroy(plan);
    velri_graph_destroy(g);
}

static void verify_all_ops_on_device(velri_context *ctx, velri_device_id dev, const char *tag,
                                     bool is_gpu) {
    if (is_gpu) {
        verify_dynamic_prompt_linear(ctx, dev);
        verify_linear_one(ctx, dev, tag, "mixed GEMM 64 rows", velri_desc_f32(),
                          64U, 81U, 37U, 3U, 4U, 2U, false, 3e-5, LINEAR_F16_MATMUL);
        verify_linear_one(ctx, dev, tag, "mixed GEMM 65 rows + bias", velri_desc_f32(),
                          65U, 97U, 49U, 2U, 5U, 3U, true, 3e-5, LINEAR_F16_MATMUL);
        verify_linear_one(ctx, dev, tag, "odd short prompt", velri_desc_f32(),
                          7, 71, 99, 1, 1, 3, true, 2e-4, LINEAR_RANDOM);
        verify_linear_one(ctx, dev, tag, "odd prompt projection", velri_desc_f32(),
                          35, 71, 99, 1, 1, 3, true, 2e-4, LINEAR_RANDOM);
        verify_rms_gated_outputs(ctx, dev, false);
        verify_rms_gated_outputs(ctx, dev, true);
        verify_gemv_residual_outputs(ctx, dev, false);
        verify_gemv_residual_outputs(ctx, dev, true);
    } else {
        verify_linear_one(ctx, dev, tag, "W8A8 GEMV signed overflow", velri_desc_w8a8_f32(),
                          1U, 9U, 132105U, 2U, 4U, 5U, false, 0.0, LINEAR_W8A8_EXTREMES);
        verify_linear_one(ctx, dev, tag, "W8A8 GEMM signed overflow", velri_desc_w8a8_f32(),
                          7U, 5U, 132106U, 1U, 5U, 7U, true, 0.0, LINEAR_W8A8_EXTREMES);
        verify_linear_one(ctx, dev, tag, "W8A8 multi-panel GEMM", velri_desc_w8a8_f32(),
                          29U, 259U, 2051U, 2U, 6U, 3U, true, 0.0, LINEAR_W8A8_EXTREMES);
        verify_linear_one(ctx, dev, tag, "W8A8 strided GEMM tails", velri_desc_w8a8_f32(),
                          127U, 131U, 516U, 3U, 5U, 7U, true, 0.0, LINEAR_W8A8_EXTREMES);
        verify_linear_one(ctx, dev, tag, "W8A8 multi-block GEMM", velri_desc_w8a8_f32(),
                          281U, 1051U, 68U, 3U, 5U, 7U, false, 0.0, LINEAR_W8A8_EXTREMES);
    }
    /* 1. Linear: all dtypes (incl. FP64 on CPU) + GEMV + GEMM + Split-K + CoopMat 64/128 */
    struct {
        size_t rows, m, k, xp, wp, yp;
    } cpu_shapes[] = {
        {1U, 67U, 131U, 5U, 7U, 3U},
        {2U, 67U, 131U, 0U, 0U, 0U},
        {7U, 67U, 131U, 5U, 7U, 3U},
        {13U, 67U, 131U, 5U, 7U, 3U},
    };
    struct {
        size_t rows, m, k, xp, wp, yp;
    } gpu_shapes[] = {
        {1U, 64U, 128U, 0U, 0U, 0U},
        {1U, 67U, 131U, 5U, 7U, 3U},
        {2U, 67U, 131U, 0U, 0U, 0U},
        {128U, 128U, 64U, 0U, 0U, 0U},
        {140U, 145U, 131U, 5U, 7U, 3U},
        {64U, 512U, 1024U, 0U, 0U, 0U},
        {67U, 259U, 1031U, 3U, 5U, 2U},
        {1280U, 1920U, 96U, 0U, 0U, 0U},
    };
    size_t nshapes = is_gpu ? (sizeof(gpu_shapes) / sizeof(gpu_shapes[0]))
                            : (sizeof(cpu_shapes) / sizeof(cpu_shapes[0]));

    for (size_t si = 0U; si < nshapes; ++si) {
        size_t rows = is_gpu ? gpu_shapes[si].rows : cpu_shapes[si].rows;
        size_t m = is_gpu ? gpu_shapes[si].m : cpu_shapes[si].m;
        size_t k = is_gpu ? gpu_shapes[si].k : cpu_shapes[si].k;
        size_t xp = is_gpu ? gpu_shapes[si].xp : cpu_shapes[si].xp;
        size_t wp = is_gpu ? gpu_shapes[si].wp : cpu_shapes[si].wp;
        size_t yp = is_gpu ? gpu_shapes[si].yp : cpu_shapes[si].yp;
        for (int b = 0; b <= 1; ++b) {
            verify_linear_one(ctx, dev, tag, "linear_f32", velri_desc_f32(), rows, m, k, xp, wp, yp,
                              b != 0, 2e-4, LINEAR_RANDOM);
            verify_linear_one(ctx, dev, tag, "linear_f64", velri_desc_f64(), rows, m, k, xp, wp, yp,
                              b != 0, 1e-11, LINEAR_RANDOM);
            verify_linear_one(ctx, dev, tag, "linear_f16", velri_desc_f16(VELRI_DTYPE_F32), rows, m,
                              k, xp, wp, yp, b != 0, 2e-3, LINEAR_RANDOM);
            verify_linear_one(ctx, dev, tag, "linear_bf16", velri_desc_bf16(VELRI_DTYPE_F32), rows,
                              m, k, xp, wp, yp, b != 0, 2e-3, LINEAR_RANDOM);
            verify_linear_one(ctx, dev, tag, "linear_f32x_bf16w", velri_desc_f32x_bf16w(), rows, m,
                              k, xp, wp, yp, b != 0, 2e-4, LINEAR_RANDOM);
            verify_linear_one(ctx, dev, tag, "linear_w8a8", velri_desc_w8a8_f32(), rows, m, k, xp,
                              wp, yp, b != 0, 2e-4, LINEAR_RANDOM);
        }
    }

    /* 2. RMSNorm (F32, BF16, F16; aligned 4096 & odd 131; unit_offset 0/1; weight on/off) */
    const velri_dtype ndts[3] = {VELRI_DTYPE_F32, VELRI_DTYPE_BF16, VELRI_DTYPE_F16};
    const double ntols[3] = {3e-5, 2e-2, 4e-3};
    const size_t ndims[2] = {131U, 4096U};
    const size_t nstrs[2] = {137U, 4096U};
    for (int di = 0; di < 3; ++di) {
        velri_dtype dt = ndts[di];
        size_t eb = velri_dtype_bytes(dt);
        for (int ci = 0; ci < 2; ++ci) {
            size_t rows = 5U, dim = ndims[ci], st = nstrs[ci];
            void *hx = xcalloc(rows * st, eb), *hw = xcalloc(dim, eb);
            void *hy_ref = xcalloc(rows * st, eb), *hy = xcalloc(rows * st, eb);
            uint64_t rng = 0x11110001ULL + (uint64_t)di * 17U + (uint64_t)ci;
            velri_test_fill_dtype(dt, hx, rows * st, &rng, 2.0f);
            velri_test_fill_dtype(dt, hw, dim, &rng, 1.5f);
            for (int uo = 0; uo <= 1; ++uo) {
                for (int has_w = 0; has_w <= 1; ++has_w) {
                    velri_graph *g = nullptr;
                    CHECK_OK(velri_graph_create(&g), "rmsnorm g");
                    velri_storage_id sx = add_store(g, VELRI_STORAGE_INPUT, dt, rows * st, dev);
                    velri_storage_id sw = has_w ? add_store(g, VELRI_STORAGE_WEIGHT, dt, dim, dev) : 0U;
                    velri_storage_id sy = add_store(g, VELRI_STORAGE_OUTPUT, dt, rows * st, dev);
                    velri_op_desc op = {
                        .kind = VELRI_OP_RMSNORM,
                        .num_inputs = has_w ? 2U : 1U,
                        .num_outputs = 1U,
                        .inputs = {add_view_2d_strided(g, sx, dt, rows, dim, st),
                                   has_w ? add_view_1d(g, sw, dt, dim, 0U) : VELRI_INVALID_ID},
                        .outputs = {add_view_2d_strided(g, sy, dt, rows, dim, st)},
                        .attr.rmsnorm = {.eps = 1e-5f, .unit_offset = (uo != 0)},
                    };
                    CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "add rmsnorm");
                    velri_plan *plan = nullptr;
                    velri_status pst = velri_plan_prepare(ctx, g, nullptr, &plan);
                    if (pst == VELRI_ERR_UNSUPPORTED) {
                        velri_graph_destroy(g);
                        continue;
                    }
                    CHECK_OK(pst, "prep rmsnorm");
                    if (has_w) CHECK_OK(velri_plan_bind_weight_host(plan, sw, hw, dim * eb), "w");
                    velri_slot *slot = nullptr;
                    CHECK_OK(velri_slot_create(plan, &slot), "slot");
                    CHECK_OK(velri_slot_bind_host(slot, sx, hx, rows * st * eb), "x");
                    CHECK_OK(velri_slot_bind_host(slot, sy, hy, rows * st * eb), "y");
                    CHECK_OK(velri_slot_run(slot, nullptr, nullptr), "run rmsnorm");
                    ref_rmsnorm(dt, dt, rows, dim, 1e-5f, uo != 0, hx, st, has_w ? hw : nullptr,
                                hy_ref, st);
                    double err = velri_test_max_abs_diff_2d(dt, hy_ref, hy, rows, dim, st);
                    CHECK(err <= ntols[di], "rmsnorm accuracy");
                    velri_slot_destroy(slot);
                    velri_plan_destroy(plan);
                    velri_graph_destroy(g);
                }
            }
            free(hx); free(hw); free(hy_ref); free(hy);
        }
    }

    /* 3. FusedAddRMSNorm (verifies both updated STATE residual + output y) */
    for (int ci = 0; ci < 2; ++ci) {
        size_t rows = 5U, dim = ndims[ci], st = nstrs[ci];
        float *hr = (float *)xcalloc(rows * st, sizeof(float));
        float *hr_ref = (float *)xcalloc(rows * st, sizeof(float));
        float *hx = (float *)xcalloc(rows * st, sizeof(float));
        float *hw = (float *)xcalloc(dim, sizeof(float));
        float *hy_ref = (float *)xcalloc(rows * st, sizeof(float));
        float *hy = (float *)xcalloc(rows * st, sizeof(float));
        uint64_t rng = 0xFA001122ULL + (uint64_t)ci;
        velri_test_fill_dtype(VELRI_DTYPE_F32, hr, rows * st, &rng, 2.0f);
        velri_test_fill_dtype(VELRI_DTYPE_F32, hx, rows * st, &rng, 2.0f);
        velri_test_fill_dtype(VELRI_DTYPE_F32, hw, dim, &rng, 1.5f);
        memcpy(hr_ref, hr, rows * st * sizeof(float));
        ref_fused_add_rmsnorm_f32(rows, dim, 1e-5f, ci == 1, hr_ref, st, hx, st, hw, hy_ref, st);

        velri_graph *g = nullptr;
        CHECK_OK(velri_graph_create(&g), "far g");
        velri_storage_id sx = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, rows * st, dev);
        velri_storage_id sw = add_store(g, VELRI_STORAGE_WEIGHT, VELRI_DTYPE_F32, dim, dev);
        velri_storage_id sr = add_store(g, VELRI_STORAGE_STATE, VELRI_DTYPE_F32, rows * st, dev);
        velri_storage_id sy = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, rows * st, dev);
        velri_op_desc op = {
            .kind = VELRI_OP_FUSED_ADD_RMSNORM,
            .num_inputs = 2U, .num_outputs = 2U,
            .inputs = {add_view_2d_strided(g, sx, VELRI_DTYPE_F32, rows, dim, st),
                       add_view_1d(g, sw, VELRI_DTYPE_F32, dim, 0U)},
            .outputs = {add_view_2d_strided(g, sr, VELRI_DTYPE_F32, rows, dim, st),
                        add_view_2d_strided(g, sy, VELRI_DTYPE_F32, rows, dim, st)},
            .attr.rmsnorm = {.eps = 1e-5f, .unit_offset = (ci == 1)},
        };
        CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "add far");
        velri_plan *plan = nullptr;
        CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prep far");
        CHECK_OK(velri_plan_bind_weight_host(plan, sw, hw, dim * sizeof(float)), "bind w");
        velri_slot *slot = nullptr;
        CHECK_OK(velri_slot_create(plan, &slot), "slot");
        CHECK_OK(velri_slot_bind_host(slot, sx, hx, rows * st * sizeof(float)), "x");
        CHECK_OK(velri_slot_bind_host(slot, sr, hr, rows * st * sizeof(float)), "r");
        CHECK_OK(velri_slot_bind_host(slot, sy, hy, rows * st * sizeof(float)), "y");
        CHECK_OK(velri_slot_run(slot, nullptr, nullptr), "run far");
        CHECK_OK(velri_slot_sync_state_to_host(slot, sr), "sync state r");
        CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, hr_ref, hr, rows, dim, st) <= 1e-5, "far res");
        CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, hy_ref, hy, rows, dim, st) <= 1e-4, "far y");
        velri_slot_destroy(slot);
        velri_plan_destroy(plan);
        velri_graph_destroy(g);
        free(hr); free(hr_ref); free(hx); free(hw); free(hy_ref); free(hy);
    }

    /* 4. LayerNorm (aligned 4096 & odd 131; weight + bias) */
    for (int ci = 0; ci < 2; ++ci) {
        size_t rows = 5U, dim = ndims[ci], st = nstrs[ci];
        float *hx = (float *)xcalloc(rows * st, sizeof(float));
        float *hw = (float *)xcalloc(dim, sizeof(float));
        float *hb = (float *)xcalloc(dim, sizeof(float));
        float *hy_ref = (float *)xcalloc(rows * st, sizeof(float));
        float *hy = (float *)xcalloc(rows * st, sizeof(float));
        uint64_t rng = 0x1A7E0001ULL + (uint64_t)ci;
        velri_test_fill_dtype(VELRI_DTYPE_F32, hx, rows * st, &rng, 3.0f);
        velri_test_fill_dtype(VELRI_DTYPE_F32, hw, dim, &rng, 1.5f);
        velri_test_fill_dtype(VELRI_DTYPE_F32, hb, dim, &rng, 0.5f);
        ref_layernorm_f32(rows, dim, 1e-5f, hx, st, hw, hb, hy_ref, st);

        velri_graph *g = nullptr;
        CHECK_OK(velri_graph_create(&g), "ln g");
        velri_storage_id sx = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, rows * st, dev);
        velri_storage_id sw = add_store(g, VELRI_STORAGE_WEIGHT, VELRI_DTYPE_F32, dim, dev);
        velri_storage_id sb = add_store(g, VELRI_STORAGE_WEIGHT, VELRI_DTYPE_F32, dim, dev);
        velri_storage_id sy = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, rows * st, dev);
        velri_op_desc op = {
            .kind = VELRI_OP_LAYERNORM,
            .num_inputs = 3U, .num_outputs = 1U,
            .inputs = {add_view_2d_strided(g, sx, VELRI_DTYPE_F32, rows, dim, st),
                       add_view_1d(g, sw, VELRI_DTYPE_F32, dim, 0U),
                       add_view_1d(g, sb, VELRI_DTYPE_F32, dim, 0U)},
            .outputs = {add_view_2d_strided(g, sy, VELRI_DTYPE_F32, rows, dim, st)},
            .attr.layernorm = {.eps = 1e-5f},
        };
        CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "add ln");
        velri_plan *plan = nullptr;
        CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prep ln");
        CHECK_OK(velri_plan_bind_weight_host(plan, sw, hw, dim * sizeof(float)), "w");
        CHECK_OK(velri_plan_bind_weight_host(plan, sb, hb, dim * sizeof(float)), "b");
        velri_slot *slot = nullptr;
        CHECK_OK(velri_slot_create(plan, &slot), "slot");
        CHECK_OK(velri_slot_bind_host(slot, sx, hx, rows * st * sizeof(float)), "x");
        CHECK_OK(velri_slot_bind_host(slot, sy, hy, rows * st * sizeof(float)), "y");
        CHECK_OK(velri_slot_run(slot, nullptr, nullptr), "run ln");
        CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, hy_ref, hy, rows, dim, st) <= 1e-4, "ln");
        velri_slot_destroy(slot);
        velri_plan_destroy(plan);
        velri_graph_destroy(g);
        free(hx); free(hw); free(hb); free(hy_ref); free(hy);
    }

    /* 5. RoPE (F32 & BF16, HALF & INTERLEAVED, partial & aligned fast-path + pos_offset_override) */
    const velri_dtype rdts[2] = {VELRI_DTYPE_F32, VELRI_DTYPE_BF16};
    const double rtols[2] = {2e-5, 2e-2};
    for (int di = 0; di < 2; ++di) {
        velri_dtype dt = rdts[di];
        size_t eb = velri_dtype_bytes(dt);
        for (int fast = 0; fast <= 1; ++fast) {
            size_t nt = fast ? 9U : 7U, nh = fast ? 8U : 5U;
            size_t hd = fast ? 128U : 72U, rd = fast ? 128U : 48U;
            size_t po_plan = 2U, po_run = 5U;
            size_t hsx = fast ? hd : (hd + 4U), tsx = fast ? (nh * hd) : (nh * hsx + 3U);
            size_t hsy = fast ? hd : (hd + 2U), tsy = fast ? (nh * hd) : (nh * hsy + 5U);
            size_t tbl_len = (po_run + nt) * (rd / 2U);

            void *hx = xcalloc(nt * tsx, eb), *hy_ref = xcalloc(nt * tsy, eb), *hy = xcalloc(nt * tsy, eb);
            float *hcos = (float *)xcalloc(tbl_len, sizeof(float));
            float *hsin = (float *)xcalloc(tbl_len, sizeof(float));
            uint64_t rng = 0x22220001ULL + (uint64_t)di * 31U + (uint64_t)fast;
            velri_test_fill_dtype(dt, hx, nt * tsx, &rng, 2.0f);
            for (size_t i = 0; i < tbl_len; ++i) {
                float ang = velri_test_rand_f32(&rng, -3.14f, 3.14f);
                hcos[i] = cosf(ang);
                hsin[i] = sinf(ang);
            }

            for (int m = 0; m <= (fast ? 0 : 1); ++m) {
                velri_rope_mode mode = m ? VELRI_ROPE_INTERLEAVED : VELRI_ROPE_HALF;
                velri_graph *g = nullptr;
                CHECK_OK(velri_graph_create(&g), "rope g");
                velri_storage_id sx = add_store(g, VELRI_STORAGE_INPUT, dt, nt * tsx, dev);
                velri_storage_id sc = add_store(g, VELRI_STORAGE_WEIGHT, VELRI_DTYPE_F32, tbl_len, dev);
                velri_storage_id ss = add_store(g, VELRI_STORAGE_WEIGHT, VELRI_DTYPE_F32, tbl_len, dev);
                velri_storage_id sy = add_store(g, VELRI_STORAGE_OUTPUT, dt, nt * tsy, dev);
                velri_op_desc op = {
                    .kind = VELRI_OP_ROPE,
                    .num_inputs = 3U, .num_outputs = 1U,
                    .inputs = {add_view_3d_strided(g, sx, dt, nt, nh, hd, tsx, hsx),
                               add_view_1d(g, sc, VELRI_DTYPE_F32, tbl_len, 0U),
                               add_view_1d(g, ss, VELRI_DTYPE_F32, tbl_len, 0U)},
                    .outputs = {add_view_3d_strided(g, sy, dt, nt, nh, hd, tsy, hsy)},
                    .attr.rope = {.rotary_dim = rd, .pos_offset = po_plan, .mode = mode},
                };
                CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "add rope");
                velri_plan *plan = nullptr;
                velri_status pst = velri_plan_prepare(ctx, g, nullptr, &plan);
                if (pst == VELRI_ERR_UNSUPPORTED) {
                    velri_graph_destroy(g);
                    continue;
                }
                CHECK_OK(pst, "prep rope");
                CHECK_OK(velri_plan_bind_weight_host(plan, sc, hcos, tbl_len * sizeof(float)), "cos");
                CHECK_OK(velri_plan_bind_weight_host(plan, ss, hsin, tbl_len * sizeof(float)), "sin");
                velri_slot *slot = nullptr;
                CHECK_OK(velri_slot_create(plan, &slot), "slot");
                CHECK_OK(velri_slot_bind_host(slot, sx, hx, nt * tsx * eb), "x");
                CHECK_OK(velri_slot_bind_host(slot, sy, hy, nt * tsy * eb), "y");
                velri_step_params sp = {.pos_offset_override = po_run, .has_pos_offset = true};
                CHECK_OK(velri_slot_run(slot, &sp, nullptr), "run rope");
                ref_rope(dt, nt, nh, hd, rd, po_run, hx, tsx, hsx, hcos, hsin, hy_ref, tsy, hsy, mode);
                for (size_t t = 0; t < nt; ++t)
                    for (size_t h = 0; h < nh; ++h)
                        for (size_t d = 0; d < hd; ++d) {
                            size_t off = t * tsy + h * hsy + d;
                            double diff = fabs(velri_test_load_f64(dt, hy_ref, off) -
                                               velri_test_load_f64(dt, hy, off));
                            CHECK(diff <= rtols[di], "rope accuracy");
                        }
                velri_slot_destroy(slot);
                velri_plan_destroy(plan);
                velri_graph_destroy(g);
            }
            free(hx); free(hy_ref); free(hy); free(hcos); free(hsin);
        }
    }

    /* 6-9. Elementwise ops (Activations + -inf edge [A17], GatedMLP, VecAxpby, VecMul)
     *      with BOTH 16B-aligned offset 0 AND 4B-aligned non-16B subview offset 4 [A5]! */
    for (int sub = 0; sub <= 1; ++sub) {
        size_t n = 1031U;
        size_t elem_off = sub ? 1U : 0U; /* byte_offset = 4 when sub == 1 (tests A5!) */
        size_t alloc_n = n + elem_off;
        float *ha = (float *)xcalloc(alloc_n, sizeof(float));
        float *hb = (float *)xcalloc(alloc_n, sizeof(float));
        float *hy_ref = (float *)xcalloc(n, sizeof(float));
        float *hy = (float *)xcalloc(alloc_n, sizeof(float));
        uint64_t rng = 0x67670001ULL + (uint64_t)sub;
        velri_test_fill_dtype(VELRI_DTYPE_F32, ha + elem_off, n, &rng, 4.0f);
        velri_test_fill_dtype(VELRI_DTYPE_F32, hb + elem_off, n, &rng, 2.0f);
        /* A17 extreme inputs: -inf, -1e38f, -100.0f, 0.0f */
        ha[elem_off + 0U] = -INFINITY;
        ha[elem_off + 1U] = -1e38f;
        ha[elem_off + 2U] = -100.0f;
        ha[elem_off + 3U] = 0.0f;

        const velri_act_kind ak[3] = {VELRI_ACT_SILU, VELRI_ACT_GELU_TANH, VELRI_ACT_RELU2};
        for (int ki = 0; ki < 3; ++ki) {
            velri_graph *g = nullptr;
            CHECK_OK(velri_graph_create(&g), "act g");
            velri_storage_id sx = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, alloc_n, dev);
            velri_storage_id sy = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, alloc_n, dev);
            velri_op_desc op = {
                .kind = VELRI_OP_ACTIVATION, .num_inputs = 1U, .num_outputs = 1U,
                .inputs = {add_view_1d(g, sx, VELRI_DTYPE_F32, n, elem_off * sizeof(float))},
                .outputs = {add_view_1d(g, sy, VELRI_DTYPE_F32, n, elem_off * sizeof(float))},
                .attr.activation = {.kind = ak[ki]},
            };
            CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "add act");
            velri_plan *plan = nullptr;
            velri_slot *slot = nullptr;
            CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prep act");
            CHECK_OK(velri_slot_create(plan, &slot), "slot");
            CHECK_OK(velri_slot_bind_host(slot, sx, ha, alloc_n * sizeof(float)), "x");
            CHECK_OK(velri_slot_bind_host(slot, sy, hy, alloc_n * sizeof(float)), "y");
            CHECK_OK(velri_slot_run(slot, nullptr, nullptr), "run act");
            ref_activation_f32(ak[ki], n, ha + elem_off, hy_ref);
            CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, hy_ref, hy + elem_off, 1U, n, n) <= 5e-4,
                  "activation + A5/A17");
            velri_slot_destroy(slot);
            velri_plan_destroy(plan);
            velri_graph_destroy(g);
        }

        /* Restore finite inputs for GatedMLP, VecAxpby, VecMul */
        velri_test_fill_dtype(VELRI_DTYPE_F32, ha + elem_off, 4U, &rng, 4.0f);
        for (int gi = 0; gi < 4; ++gi) {
            velri_act_kind gact = (gi < 2) ? VELRI_ACT_SILU : VELRI_ACT_GELU_TANH;
            float clamp = (gi & 1) ? 1.5f : 0.0f;
            velri_graph *g = nullptr;
            CHECK_OK(velri_graph_create(&g), "gated g");
            velri_storage_id s0 = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, alloc_n, dev);
            velri_storage_id s1 = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, alloc_n, dev);
            velri_storage_id sy = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, alloc_n, dev);
            velri_op_desc op = {
                .kind = VELRI_OP_GATED_MLP, .num_inputs = 2U, .num_outputs = 1U,
                .inputs = {add_view_1d(g, s0, VELRI_DTYPE_F32, n, elem_off * sizeof(float)),
                           add_view_1d(g, s1, VELRI_DTYPE_F32, n, elem_off * sizeof(float))},
                .outputs = {add_view_1d(g, sy, VELRI_DTYPE_F32, n, elem_off * sizeof(float))},
                .attr.gated_mlp = {.act = gact, .clamp_limit = clamp},
            };
            CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "add gated");
            velri_plan *plan = nullptr;
            velri_slot *slot = nullptr;
            CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prep gated");
            CHECK_OK(velri_slot_create(plan, &slot), "slot");
            CHECK_OK(velri_slot_bind_host(slot, s0, ha, alloc_n * sizeof(float)), "g");
            CHECK_OK(velri_slot_bind_host(slot, s1, hb, alloc_n * sizeof(float)), "u");
            CHECK_OK(velri_slot_bind_host(slot, sy, hy, alloc_n * sizeof(float)), "y");
            CHECK_OK(velri_slot_run(slot, nullptr, nullptr), "run gated");
            ref_gated_mlp_f32(gact, n, clamp, ha + elem_off, hb + elem_off, hy_ref);
            CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, hy_ref, hy + elem_off, 1U, n, n) <= 5e-4,
                  "gated_mlp");
            velri_slot_destroy(slot);
            velri_plan_destroy(plan);
            velri_graph_destroy(g);
        }

        /* VecAxpby (beta != 0 and beta == 0) + VecMul */
        for (int mode = 0; mode < 3; ++mode) {
            velri_graph *g = nullptr;
            CHECK_OK(velri_graph_create(&g), "vec g");
            velri_storage_id s0 = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, alloc_n, dev);
            velri_storage_id s1 = (mode != 1) ? add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, alloc_n, dev) : 0U;
            velri_storage_id sy = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, alloc_n, dev);
            velri_op_desc op = {0};
            if (mode == 0) {
                op.kind = VELRI_OP_VEC_AXPBY;
                op.num_inputs = 2U; op.num_outputs = 1U;
                op.inputs[0] = add_view_1d(g, s0, VELRI_DTYPE_F32, n, elem_off * sizeof(float));
                op.inputs[1] = add_view_1d(g, s1, VELRI_DTYPE_F32, n, elem_off * sizeof(float));
                op.outputs[0] = add_view_1d(g, sy, VELRI_DTYPE_F32, n, elem_off * sizeof(float));
                op.attr.axpby.alpha = 1.35f; op.attr.axpby.beta = -0.75f;
                ref_vec_axpby_f32(n, 1.35f, ha + elem_off, -0.75f, hb + elem_off, hy_ref);
            } else if (mode == 1) {
                op.kind = VELRI_OP_VEC_AXPBY;
                op.num_inputs = 1U; op.num_outputs = 1U;
                op.inputs[0] = add_view_1d(g, s0, VELRI_DTYPE_F32, n, elem_off * sizeof(float));
                op.outputs[0] = add_view_1d(g, sy, VELRI_DTYPE_F32, n, elem_off * sizeof(float));
                op.attr.axpby.alpha = 2.25f; op.attr.axpby.beta = 0.0f;
                ref_vec_axpby_f32(n, 2.25f, ha + elem_off, 0.0f, nullptr, hy_ref);
            } else {
                op.kind = VELRI_OP_VEC_MUL;
                op.num_inputs = 2U; op.num_outputs = 1U;
                op.inputs[0] = add_view_1d(g, s0, VELRI_DTYPE_F32, n, elem_off * sizeof(float));
                op.inputs[1] = add_view_1d(g, s1, VELRI_DTYPE_F32, n, elem_off * sizeof(float));
                op.outputs[0] = add_view_1d(g, sy, VELRI_DTYPE_F32, n, elem_off * sizeof(float));
                ref_vec_mul_f32(n, ha + elem_off, hb + elem_off, hy_ref);
            }
            CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "add vec");
            velri_plan *plan = nullptr;
            velri_slot *slot = nullptr;
            CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prep vec");
            CHECK_OK(velri_slot_create(plan, &slot), "slot");
            CHECK_OK(velri_slot_bind_host(slot, s0, ha, alloc_n * sizeof(float)), "a");
            if (mode != 1) CHECK_OK(velri_slot_bind_host(slot, s1, hb, alloc_n * sizeof(float)), "b");
            CHECK_OK(velri_slot_bind_host(slot, sy, hy, alloc_n * sizeof(float)), "y");
            CHECK_OK(velri_slot_run(slot, nullptr, nullptr), "run vec");
            CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, hy_ref, hy + elem_off, 1U, n, n) <= 1e-5,
                  "vec_axpby/mul");
            velri_slot_destroy(slot);
            velri_plan_destroy(plan);
            velri_graph_destroy(g);
        }
        free(ha); free(hb); free(hy_ref); free(hy);
    }

    /* 10. Softmax (wide range, odd & aligned 4096, plus all-(-inf) row [A17]) */
    for (int ci = 0; ci < 2; ++ci) {
        size_t rows = 5U, cols = ndims[ci], st = nstrs[ci];
        float *hx = (float *)xcalloc(rows * st, sizeof(float));
        float *hy_ref = (float *)xcalloc(rows * st, sizeof(float));
        float *hy = (float *)xcalloc(rows * st, sizeof(float));
        uint64_t rng = 0x44440001ULL + (uint64_t)ci;
        velri_test_fill_dtype(VELRI_DTYPE_F32, hx, rows * st, &rng, 20.0f);
        /* Row 2: all -INFINITY -> exact 0.0f; Row 3: causal mask half -INFINITY [A17] */
        for (size_t c = 0; c < cols; ++c) hx[2U * st + c] = -INFINITY;
        for (size_t c = cols / 2U; c < cols; ++c) hx[3U * st + c] = -INFINITY;
        ref_softmax_f32(rows, cols, hx, st, hy_ref, st);

        velri_graph *g = nullptr;
        CHECK_OK(velri_graph_create(&g), "sm g");
        velri_storage_id sx = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, rows * st, dev);
        velri_storage_id sy = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, rows * st, dev);
        velri_op_desc op = {
            .kind = VELRI_OP_SOFTMAX, .num_inputs = 1U, .num_outputs = 1U,
            .inputs = {add_view_2d_strided(g, sx, VELRI_DTYPE_F32, rows, cols, st)},
            .outputs = {add_view_2d_strided(g, sy, VELRI_DTYPE_F32, rows, cols, st)},
        };
        CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "add sm");
        velri_plan *plan = nullptr;
        velri_slot *slot = nullptr;
        CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prep sm");
        CHECK_OK(velri_slot_create(plan, &slot), "slot");
        CHECK_OK(velri_slot_bind_host(slot, sx, hx, rows * st * sizeof(float)), "x");
        CHECK_OK(velri_slot_bind_host(slot, sy, hy, rows * st * sizeof(float)), "y");
        CHECK_OK(velri_slot_run(slot, nullptr, nullptr), "run sm");
        CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, hy_ref, hy, rows, cols, st) <= 1e-4,
              "softmax + A17 -inf row");
        velri_slot_destroy(slot);
        velri_plan_destroy(plan);
        velri_graph_destroy(g);
        free(hx); free(hy_ref); free(hy);
    }

    /* 11. FWHT (B in {16,32,64,128,256} x suh in {none, F32, F16}) */
    {
        const size_t bs[5] = {16U, 32U, 64U, 128U, 256U};
        size_t rows = 5U, cols = 512U, xs = 521U, ys = 523U;
        float *hx = (float *)xcalloc(rows * xs, sizeof(float));
        float *hsf = (float *)xcalloc(cols, sizeof(float));
        velri_fp16 *hsh = (velri_fp16 *)xcalloc(cols, sizeof(velri_fp16));
        float *hy_ref = (float *)xcalloc(rows * ys, sizeof(float));
        float *hy = (float *)xcalloc(rows * ys, sizeof(float));
        uint64_t rng = 0xF1170001ULL;
        velri_test_fill_dtype(VELRI_DTYPE_F32, hx, rows * xs, &rng, 2.0f);
        velri_test_fill_dtype(VELRI_DTYPE_F32, hsf, cols, &rng, 1.5f);
        velri_test_fill_dtype(VELRI_DTYPE_F16, hsh, cols, &rng, 1.5f);
        for (int bi = 0; bi < 5; ++bi) {
            for (int sm = 0; sm < 3; ++sm) {
                velri_dtype sdt = (sm == 2) ? VELRI_DTYPE_F16 : VELRI_DTYPE_F32;
                velri_graph *g = nullptr;
                CHECK_OK(velri_graph_create(&g), "fwht g");
                velri_storage_id sx = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, rows * xs, dev);
                velri_storage_id ss = (sm > 0) ? add_store(g, VELRI_STORAGE_WEIGHT, sdt, cols, dev) : 0U;
                velri_storage_id sy = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, rows * ys, dev);
                velri_op_desc op = {
                    .kind = VELRI_OP_FWHT, .num_inputs = (sm > 0) ? 2U : 1U, .num_outputs = 1U,
                    .inputs = {add_view_2d_strided(g, sx, VELRI_DTYPE_F32, rows, cols, xs),
                               (sm > 0) ? add_view_1d(g, ss, sdt, cols, 0U) : VELRI_INVALID_ID},
                    .outputs = {add_view_2d_strided(g, sy, VELRI_DTYPE_F32, rows, cols, ys)},
                    .attr.fwht = {.block_size = bs[bi], .suh_dtype = sdt},
                };
                CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "add fwht");
                velri_plan *plan = nullptr;
                velri_status pst = velri_plan_prepare(ctx, g, nullptr, &plan);
                if (pst == VELRI_ERR_UNSUPPORTED) {
                    velri_graph_destroy(g);
                    continue;
                }
                CHECK_OK(pst, "prep fwht");
                if (sm == 1) CHECK_OK(velri_plan_bind_weight_host(plan, ss, hsf, cols * sizeof(float)), "suh32");
                if (sm == 2) CHECK_OK(velri_plan_bind_weight_host(plan, ss, hsh, cols * sizeof(velri_fp16)), "suh16");
                velri_slot *slot = nullptr;
                CHECK_OK(velri_slot_create(plan, &slot), "slot");
                CHECK_OK(velri_slot_bind_host(slot, sx, hx, rows * xs * sizeof(float)), "x");
                CHECK_OK(velri_slot_bind_host(slot, sy, hy, rows * ys * sizeof(float)), "y");
                CHECK_OK(velri_slot_run(slot, nullptr, nullptr), "run fwht");
                ref_fwht_f32(rows, cols, bs[bi], hx, xs, sdt,
                             (sm == 0) ? nullptr : ((sm == 1) ? (const void *)hsf : (const void *)hsh),
                             hy_ref, ys);
                CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, hy_ref, hy, rows, cols, ys) <= 5e-4,
                      "fwht accuracy");
                velri_slot_destroy(slot);
                velri_plan_destroy(plan);
                velri_graph_destroy(g);
            }
        }
        free(hx); free(hsf); free(hsh); free(hy_ref); free(hy);
    }

    /* 12. QuantizeRow F32 -> I8 (odd & 4096, all-zero row + A12 tiny 1e-37f row!) */
    for (int ci = 0; ci < 2; ++ci) {
        size_t rows = 6U, k = ndims[ci], st = nstrs[ci];
        float *hx = (float *)xcalloc(rows * st, sizeof(float));
        int8_t *hq_ref = (int8_t *)xcalloc(rows * st, 1U);
        int8_t *hq = (int8_t *)xcalloc(rows * st, 1U);
        float *hs_ref = (float *)xcalloc(rows, sizeof(float));
        float *hs = (float *)xcalloc(rows, sizeof(float));
        uint64_t rng = 0x89890001ULL + (uint64_t)ci;
        velri_test_fill_dtype(VELRI_DTYPE_F32, hx, rows * st, &rng, 5.0f);
        memset(hx + 2U * st, 0, k * sizeof(float)); /* all-zero row */
        for (size_t c = 0; c < k; ++c) hx[3U * st + c] = 1e-37f; /* A12 tiny subnormal-adj row */
        ref_quantize_row_f32_i8(rows, k, hx, st, hq_ref, st, hs_ref);

        velri_graph *g = nullptr;
        CHECK_OK(velri_graph_create(&g), "qrow g");
        velri_storage_id sx = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, rows * st, dev);
        velri_storage_id sq = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_I8, rows * st, dev);
        velri_storage_id ss = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, rows, dev);
        velri_op_desc op = {
            .kind = VELRI_OP_QUANTIZE_ROW, .num_inputs = 1U, .num_outputs = 2U,
            .inputs = {add_view_2d_strided(g, sx, VELRI_DTYPE_F32, rows, k, st)},
            .outputs = {add_view_2d_strided(g, sq, VELRI_DTYPE_I8, rows, k, st),
                        add_view_1d(g, ss, VELRI_DTYPE_F32, rows, 0U)},
        };
        CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "add qrow");
        velri_plan *plan = nullptr;
        velri_status pst = velri_plan_prepare(ctx, g, nullptr, &plan);
        if (pst == VELRI_ERR_UNSUPPORTED) {
            velri_graph_destroy(g);
            free(hx); free(hq_ref); free(hq); free(hs_ref); free(hs);
            continue;
        }
        CHECK_OK(pst, "prep qrow");
        velri_slot *slot = nullptr;
        CHECK_OK(velri_slot_create(plan, &slot), "slot");
        CHECK_OK(velri_slot_bind_host(slot, sx, hx, rows * st * sizeof(float)), "x");
        CHECK_OK(velri_slot_bind_host(slot, sq, hq, rows * st), "q");
        CHECK_OK(velri_slot_bind_host(slot, ss, hs, rows * sizeof(float)), "s");
        CHECK_OK(velri_slot_run(slot, nullptr, nullptr), "run qrow");

        for (size_t r = 0; r < rows; ++r) {
            CHECK(isfinite(hs[r]) && hs[r] > 0.0f, "qrow finite positive scale");
            if (r != 3U) {
                CHECK(fabsf(hs_ref[r] - hs[r]) <= 1e-6f, "qrow scale");
                for (size_t c = 0; c < k; ++c) {
                    int diff = abs((int)hq_ref[r * st + c] - (int)hq[r * st + c]);
                    CHECK(diff <= 1, "qrow int8 match");
                }
            }
        }
        velri_slot_destroy(slot);
        velri_plan_destroy(plan);
        velri_graph_destroy(g);
        free(hx); free(hq_ref); free(hq); free(hs_ref); free(hs);
    }

    /* 13. Embedding (token lookup w/ OOB & negative IDs + positional lookup w/ dynamic override) */
    {
        size_t vocab = 37U, dim = 69U, max_t = 8U, t_stride = 73U, o_stride = 71U;
        float *htbl = (float *)xcalloc(vocab * t_stride, sizeof(float));
        int32_t *hids = (int32_t *)xcalloc(max_t, sizeof(int32_t));
        float *hy = (float *)xcalloc(max_t * o_stride, sizeof(float));
        float *hy_ref = (float *)xcalloc(max_t * o_stride, sizeof(float));
        uint64_t rng = 0xE11B0001ULL;
        velri_test_fill_dtype(VELRI_DTYPE_F32, htbl, vocab * t_stride, &rng, 2.0f);
        hids[0] = 0; hids[1] = 36; hids[2] = -5; hids[3] = 999; hids[4] = 17;
        hids[5] = 3; hids[6] = 12; hids[7] = 1;

        /* 13a. Token embedding with dynamic seq_len_override = 5 */
        velri_graph *g = nullptr;
        CHECK_OK(velri_graph_create(&g), "emb g");
        velri_storage_id st = add_store(g, VELRI_STORAGE_WEIGHT, VELRI_DTYPE_F32, vocab * t_stride, dev);
        velri_storage_id si = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_I32, max_t, dev);
        velri_storage_id sy = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, max_t * o_stride, dev);
        velri_op_desc op = {
            .kind = VELRI_OP_EMBEDDING, .num_inputs = 2U, .num_outputs = 1U,
            .inputs = {add_view_2d_strided(g, st, VELRI_DTYPE_F32, vocab, dim, t_stride),
                       add_view_1d(g, si, VELRI_DTYPE_I32, max_t, 0U)},
            .outputs = {add_view_2d_strided(g, sy, VELRI_DTYPE_F32, max_t, dim, o_stride)},
        };
        CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "add emb");
        velri_plan *plan = nullptr;
        velri_slot *slot = nullptr;
        CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prep emb");
        CHECK_OK(velri_plan_bind_weight_host(plan, st, htbl, vocab * t_stride * sizeof(float)), "wt");
        CHECK_OK(velri_slot_create(plan, &slot), "slot");
        CHECK_OK(velri_slot_bind_host(slot, si, hids, max_t * sizeof(int32_t)), "ids");
        CHECK_OK(velri_slot_bind_host(slot, sy, hy, max_t * o_stride * sizeof(float)), "y");
        velri_step_params sp = {.has_seq_len = true, .seq_len_override = 5U};
        CHECK_OK(velri_slot_run(slot, &sp, nullptr), "run emb");
        CHECK(velri_slot_last_run_allocs(slot) == 0U, "emb 0 allocs");
        velri_ref_embedding_f32(vocab, dim, 5U, 0U, htbl, t_stride, hids, hy_ref, o_stride);
        CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, hy_ref, hy, 5U, dim, o_stride) <= 1e-6,
              "token embedding accuracy");
        velri_slot_destroy(slot);
        velri_plan_destroy(plan);
        velri_graph_destroy(g);

        /* 13b. Positional embedding with dynamic pos_offset_override = 11, seq_len_override = 4 */
        CHECK_OK(velri_graph_create(&g), "pos emb g");
        st = add_store(g, VELRI_STORAGE_WEIGHT, VELRI_DTYPE_F32, vocab * t_stride, dev);
        sy = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, max_t * o_stride, dev);
        velri_op_desc pop = {
            .kind = VELRI_OP_EMBEDDING, .num_inputs = 1U, .num_outputs = 1U,
            .inputs = {add_view_2d_strided(g, st, VELRI_DTYPE_F32, vocab, dim, t_stride)},
            .outputs = {add_view_2d_strided(g, sy, VELRI_DTYPE_F32, max_t, dim, o_stride)},
            .attr.embedding = {.pos_offset = 0U},
        };
        CHECK_OK(velri_graph_add_op(g, &pop, dev, nullptr), "add pos emb");
        CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prep pos emb");
        CHECK_OK(velri_plan_bind_weight_host(plan, st, htbl, vocab * t_stride * sizeof(float)), "wt");
        CHECK_OK(velri_slot_create(plan, &slot), "slot");
        CHECK_OK(velri_slot_bind_host(slot, sy, hy, max_t * o_stride * sizeof(float)), "y");
        velri_step_params psp = {
            .has_pos_offset = true, .pos_offset_override = 11U,
            .has_seq_len = true, .seq_len_override = 4U,
        };
        CHECK_OK(velri_slot_run(slot, &psp, nullptr), "run pos emb");
        velri_ref_embedding_f32(vocab, dim, 4U, 11U, htbl, t_stride, nullptr, hy_ref, o_stride);
        CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, hy_ref, hy, 4U, dim, o_stride) <= 1e-6,
              "positional embedding accuracy");
        velri_slot_destroy(slot);
        velri_plan_destroy(plan);
        velri_graph_destroy(g);
        free(htbl); free(hids); free(hy); free(hy_ref);
    }

    /* 14. CopyLastRow with dynamic seq_len_override */
    {
        size_t max_t = 9U, dim = 131U, xs = 137U;
        float *hx = (float *)xcalloc(max_t * xs, sizeof(float));
        float *hy = (float *)xcalloc(dim, sizeof(float));
        float *hy_ref = (float *)xcalloc(dim, sizeof(float));
        uint64_t rng = 0xC0910001ULL;
        velri_test_fill_dtype(VELRI_DTYPE_F32, hx, max_t * xs, &rng, 3.0f);

        velri_graph *g = nullptr;
        CHECK_OK(velri_graph_create(&g), "clr g");
        velri_storage_id sx = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, max_t * xs, dev);
        velri_storage_id sy = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, dim, dev);
        velri_op_desc op = {
            .kind = VELRI_OP_COPY_LAST_ROW, .num_inputs = 1U, .num_outputs = 1U,
            .inputs = {add_view_2d_strided(g, sx, VELRI_DTYPE_F32, max_t, dim, xs)},
            .outputs = {add_view_2d_strided(g, sy, VELRI_DTYPE_F32, 1U, dim, dim)},
        };
        CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "add clr");
        velri_plan *plan = nullptr;
        velri_slot *slot = nullptr;
        CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prep clr");
        CHECK_OK(velri_slot_create(plan, &slot), "slot");
        CHECK_OK(velri_slot_bind_host(slot, sx, hx, max_t * xs * sizeof(float)), "x");
        CHECK_OK(velri_slot_bind_host(slot, sy, hy, dim * sizeof(float)), "y");
        velri_step_params sp = {.has_seq_len = true, .seq_len_override = 6U};
        CHECK_OK(velri_slot_run(slot, &sp, nullptr), "run clr");
        velri_ref_copy_last_row_f32(6U, dim, hx, xs, hy_ref);
        CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, hy_ref, hy, 1U, dim, dim) <= 1e-6,
              "copy_last_row accuracy");
        velri_slot_destroy(slot);
        velri_plan_destroy(plan);
        velri_graph_destroy(g);
        free(hx); free(hy); free(hy_ref);
    }

    /* 15. Attention (GQA + long decode kv=25..26 + sliding window + bidir + same_kv_cache + hd=64/80/128) */
    {
        /* 15a. GQA + strided fused-QKV + long decode (kv_len=25,26 > 8 -> 4 Vulkan P=8 iters & 2 CPU B_k=16 tiles) */
        verify_attention_case(ctx, dev, tag, "attn_gqa_long_decode_hd64", 6U, 2U, 64U, 24U, 40U,
                              24U, 2U, 0U, true, false, 0U, false, 0xA77E0001ULL, VELRI_DTYPE_F32, false);

        /* 15b. Sliding window = 5 (kv_len up to 11 > 5 masks earlier keys) + head_dim = 128 */
        verify_attention_case(ctx, dev, tag, "attn_sliding_win5_hd128", 4U, 2U, 128U, 12U, 24U,
                              9U, 2U, 5U, true, false, 0U, false, 0xA77E0002ULL, VELRI_DTYPE_F32, false);

        /* 15c. Bidirectional attention (causal = false) + odd head_dim = 80 */
        verify_attention_case(ctx, dev, tag, "attn_bidir_hd80", 6U, 2U, 80U, 20U, 32U, 18U, 2U,
                              0U, false, false, 0U, false, 0xA77E0003ULL, VELRI_DTYPE_F32, false);

        /* 15d. Shared K/V cache (same_kv_cache = true, K_in == V_in, K_cache == V_cache) + head_dim = 128 */
        verify_attention_case(ctx, dev, tag, "attn_same_kv_hd128", 4U, 2U, 128U, 18U, 32U, 17U, 2U,
                              0U, true, true, 0U, false, 0xA77E0004ULL, VELRI_DTYPE_F32, false);

        /* 15e. FlashDecoding Split-KV long-context decode (kv_len=321..324 >= 256, GQA 4Q/2KV) */
        verify_attention_case(ctx, dev, tag, "attn_splitkv_long324_hd64", 4U, 2U, 64U, 320U, 384U,
                              320U, 4U, 0U, true, false, 0U, false, 0xA77E0005ULL, VELRI_DTYPE_F32, false);

        /* 15f. FlashDecoding Split-KV with sliding window = 280 (kv_len=301..303 > 280 >= 256) */
        verify_attention_case(ctx, dev, tag, "attn_splitkv_win280_hd64", 4U, 2U, 64U, 300U, 400U,
                              300U, 3U, 280U, true, false, 0U, false, 0xA77E0006ULL, VELRI_DTYPE_F32, false);

        /* Cross the short decode / Split-KV boundary while replaying the same slot. */
        verify_attention_case(ctx, dev, tag, "attn_splitkv_cross128_hd64", 4U, 2U, 64U, 125U,
                              160U, 125U, 5U, 0U, true, false, 0U, false, 0xA77E0007ULL, VELRI_DTYPE_F32, false);

        // Tiled prefill selection, partial query/key tiles, shared KV and padded
        // cache strides. Decode immediately afterwards observes the same cache.
        verify_attention_case(ctx, dev, tag, "attn_flash_boundary_below", 6U, 2U, 64U,
                              127U, 132U, 127U, 1U, 0U, true, false, 4U, false, 0xA77E0010ULL, VELRI_DTYPE_F32, false);
        verify_attention_case(ctx, dev, tag, "attn_flash_bidir", 6U, 2U, 32U,
                              128U, 136U, 128U, 1U, 0U, false, true, 4U, false, 0xA77E0011ULL, VELRI_DTYPE_F32, false);
        verify_attention_case(ctx, dev, tag, "attn_flash_ragged_window", 6U, 2U, 64U,
                              259U, 264U, 259U, 1U, 129U, true, false, 4U, false, 0xA77E0012ULL, VELRI_DTYPE_F32, false);
        verify_attention_case(ctx, dev, tag, "attn_flash_hd128", 4U, 2U, 128U,
                              257U, 272U, 257U, 1U, 0U, true, false, 0U, false, 0xA77E0013ULL, VELRI_DTYPE_F32, false);
        verify_attention_case(ctx, dev, tag, "attn_flash_unaligned_cache", 4U, 2U, 64U,
                              129U, 136U, 129U, 1U, 3U, true, false, 1U, false, 0xA77E0014ULL, VELRI_DTYPE_F32, false);

        // Several short prompt chunks accumulate a long KV prefix. Exercise an odd-sized
        // query tile, a 257-key sliding window, an output row with an unaligned stride, and Q == O.
        verify_attention_case(ctx, dev, tag, "attn_prefill_split_window", 6U, 2U, 64U,
                              33U, 521U, 519U, 2U, 257U, true, false, 4U, false, 0xA77E0015ULL, VELRI_DTYPE_F32, false);
        verify_attention_case(ctx, dev, tag, "attn_prefill_split_inplace", 3U, 1U, 64U,
                              63U, 533U, 532U, 0U, 0U, false, true, 0U, true, 0xA77E0016ULL, VELRI_DTYPE_F32, false);
        verify_attention_case(ctx, dev, tag, "attn_prefill_split_unaligned", 4U, 2U, 64U,
                              32U, 289U, 288U, 1U, 0U, true, false, 1U, false, 0xA77E0017ULL, VELRI_DTYPE_F32, false);

        // Flash prefill, eager/in-place fallback, short and split decode, ragged strides and wide heads.
        // Full-size tiles hit both direct prefill implementations, including the masked NaN checks.
        verify_attention_case(ctx, dev, tag, "attn_dense_tile_f32", 3U, 1U, 64U,
                              513U, 520U, 513U, 1U, 0U, true, false, 0U, true, 0xA77E0020ULL, VELRI_DTYPE_F32, false);
        verify_attention_case(ctx, dev, tag, "attn_cooperative_half_cache", 3U, 1U, 64U,
                              513U, 520U, 513U, 1U, 0U, true, false, 0U, true, 0xA77E0021ULL, VELRI_DTYPE_F16, false);
        verify_attention_case(ctx, dev, tag, "attn_cooperative_half_operands", 3U, 1U, 64U,
                              513U, 520U, 513U, 1U, 0U, true, false, 0U, true, 0xA77E0021ULL, VELRI_DTYPE_F16, true);
        // The third 512-row chunk selects the long-prefix NV cooperative kernel when available.
        verify_attention_case(ctx, dev, tag, "attn_cooperative_long_append", 2U, 1U, 64U,
                              512U, 1568U, 1536U, 1U, 0U, true, false, 0U, true, 0xA77E0029ULL, VELRI_DTYPE_F16, true);
        verify_attention_case(ctx, dev, tag, "attn_cooperative_compensated_short_append", 3U, 1U, 64U,
                              128U, 523U, 517U, 2U, 0U, true, false, 0U, false, 0xA77E0025ULL, VELRI_DTYPE_F16, false);
        verify_attention_case(ctx, dev, tag, "attn_cooperative_short_append", 3U, 1U, 64U,
                              64U, 523U, 517U, 2U, 0U, true, false, 0U, true, 0xA77E0025ULL, VELRI_DTYPE_F16, true);
        verify_attention_case(ctx, dev, tag, "attn_half_cache_flash", 6U, 2U, 64U,
                              257U, 312U, 257U, 3U, 0U, true, false, 0U, false, 0xA77E0020ULL, VELRI_DTYPE_F16, false);
        verify_attention_case(ctx, dev, tag, "attn_half_cache_split_window", 3U, 1U, 64U,
                              33U, 521U, 519U, 2U, 257U, true, false, 4U, true, 0xA77E0021ULL, VELRI_DTYPE_F16, false);
        verify_attention_case(ctx, dev, tag, "attn_half_cache_short_shared", 4U, 2U, 32U,
                              8U, 32U, 7U, 2U, 5U, false, true, 0U, false, 0xA77E0022ULL, VELRI_DTYPE_F16, false);
        verify_attention_case(ctx, dev, tag, "attn_half_cache_wide", 2U, 1U, 257U,
                              5U, 16U, 5U, 2U, 0U, true, false, 1U, false, 0xA77E0023ULL, VELRI_DTYPE_F16, false);

        verify_attention_case(ctx, dev, tag, "attn_half_cache_32_split_transition", 4U, 2U, 64U,
                              64U, 1040U, 1022U, 5U, 0U, true, false, 0U, false, 0xA77E0024ULL, VELRI_DTYPE_F16, false);

        for (uint32_t group = 2U; group <= 4U; ++group) {
            verify_attention_case(ctx, dev, tag, "attn_half_grouped_split_cross4096", 2U * group, 2U,
                                  64U, 1U, 4128U, 4093U, 9U, group == 3U ? 4097U : 0U,
                                  true, group == 2U, 8U, group == 4U, 0xA77E0025ULL + group,
                                  VELRI_DTYPE_F16, false);
        }

        /* 15g. C1 regression: T=1 decode with has_pos_offset=true, has_kv_len=false (fallback kv_pos=pos_offset) */
        {
            const uint32_t nh = 4U, nkvh = 2U, hd = 32U, max_t = 4U, max_ctx = 4098U;
            const size_t q_dim = (size_t)nh * hd, kv_dim = (size_t)nkvh * hd;
            float *hq = (float *)xcalloc((size_t)max_t * q_dim, sizeof(float));
            float *hk = (float *)xcalloc((size_t)max_t * kv_dim, sizeof(float));
            float *hv = (float *)xcalloc((size_t)max_t * kv_dim, sizeof(float));
            float *ho = (float *)xcalloc((size_t)max_t * q_dim, sizeof(float));
            float *ho_ref = (float *)xcalloc(q_dim, sizeof(float));
            float *hkc = (float *)xcalloc((size_t)nkvh * max_ctx * hd, sizeof(float));
            float *hvc = (float *)xcalloc((size_t)nkvh * max_ctx * hd, sizeof(float));
            uint64_t rng = 0xC1FA11B0ULL;

            velri_graph *g = nullptr;
            CHECK_OK(velri_graph_create(&g), "c1 attn g");
            velri_storage_id sq = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, (size_t)max_t * q_dim, dev);
            velri_storage_id sk = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, (size_t)max_t * kv_dim, dev);
            velri_storage_id sv = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, (size_t)max_t * kv_dim, dev);
            velri_storage_id so = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, (size_t)max_t * q_dim, dev);
            velri_storage_id skc = add_store(g, VELRI_STORAGE_STATE, VELRI_DTYPE_F32, (size_t)nkvh * max_ctx * hd, dev);
            velri_storage_id svc = add_store(g, VELRI_STORAGE_STATE, VELRI_DTYPE_F32, (size_t)nkvh * max_ctx * hd, dev);

            velri_tensor_view vkc, vvc;
            const size_t cdims[3] = {nkvh, max_ctx, hd};
            CHECK_OK(velri_tensor_view_contiguous(skc, VELRI_DTYPE_F32, 3U, cdims, 0U, &vkc), "c1 vkc");
            CHECK_OK(velri_tensor_view_contiguous(svc, VELRI_DTYPE_F32, 3U, cdims, 0U, &vvc), "c1 vvc");
            velri_view_id id_kc = VELRI_INVALID_ID, id_vc = VELRI_INVALID_ID;
            CHECK_OK(velri_graph_add_view(g, &vkc, &id_kc), "c1 add kc");
            CHECK_OK(velri_graph_add_view(g, &vvc, &id_vc), "c1 add vc");

            velri_op_desc op = {
                .kind = VELRI_OP_ATTENTION,
                .num_inputs = 3U,
                .num_outputs = 3U,
                .inputs = {
                    add_view_2d_strided(g, sq, VELRI_DTYPE_F32, max_t, q_dim, q_dim),
                    add_view_2d_strided(g, sk, VELRI_DTYPE_F32, max_t, kv_dim, kv_dim),
                    add_view_2d_strided(g, sv, VELRI_DTYPE_F32, max_t, kv_dim, kv_dim),
                },
                .outputs = {
                    add_view_2d_strided(g, so, VELRI_DTYPE_F32, max_t, q_dim, q_dim),
                    id_kc,
                    id_vc,
                },
                .dynamic_seq = true,
                .attr.attention = {.n_heads = nh, .n_kv_heads = nkvh, .head_dim = hd, .causal = true},
            };
            CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "c1 add attn");
            velri_plan *plan = nullptr;
            velri_slot *slot = nullptr;
            CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "c1 prep");
            CHECK_OK(velri_slot_create(plan, &slot), "c1 slot");
            CHECK_OK(velri_slot_bind_host(slot, sq, hq, (size_t)max_t * q_dim * sizeof(float)), "c1 q");
            CHECK_OK(velri_slot_bind_host(slot, sk, hk, (size_t)max_t * kv_dim * sizeof(float)), "c1 k");
            CHECK_OK(velri_slot_bind_host(slot, sv, hv, (size_t)max_t * kv_dim * sizeof(float)), "c1 v");
            CHECK_OK(velri_slot_bind_host(slot, so, ho, (size_t)max_t * q_dim * sizeof(float)), "c1 o");

            /* Seed the resident cache so a single decode reads successive 128-key tiles in
             * each Split-KV partition without a quadratic prefill. */
            const size_t cache_elems = (size_t)nkvh * max_ctx * hd;
            const size_t cache_bytes = cache_elems * sizeof(float);
            velri_test_fill_dtype(VELRI_DTYPE_F32, hkc, cache_elems, &rng, 4.0f);
            velri_test_fill_dtype(VELRI_DTYPE_F32, hvc, cache_elems, &rng, 1.0f);
            CHECK_OK(velri_slot_bind_host(slot, skc, hkc, cache_bytes), "c1 bind K cache");
            CHECK_OK(velri_slot_bind_host(slot, svc, hvc, cache_bytes), "c1 bind V cache");
            CHECK_OK(velri_slot_sync_state_from_host(slot, skc), "c1 upload K cache");
            CHECK_OK(velri_slot_sync_state_from_host(slot, svc), "c1 upload V cache");
            const size_t positions[] = {0U, 1U, 2U, 3U, 4U, 4095U, 4096U};
            for (size_t i = 0U; i < sizeof(positions) / sizeof(positions[0]); ++i) {
                const size_t step = positions[i];
                velri_test_fill_dtype(VELRI_DTYPE_F32, hq, q_dim, &rng, 0.5f);
                velri_test_fill_dtype(VELRI_DTYPE_F32, hk, kv_dim, &rng, 0.5f);
                velri_test_fill_dtype(VELRI_DTYPE_F32, hv, kv_dim, &rng, 0.5f);
                /* Intentionally omit has_kv_len: kv_pos MUST fall back to pos_offset_override! */
                velri_step_params sp = {
                    .has_seq_len = true,
                    .seq_len_override = 1U,
                    .has_pos_offset = true,
                    .pos_offset_override = step,
                    .has_kv_len = false,
                };
                CHECK_OK(velri_slot_run(slot, &sp, nullptr), "c1 run");
                if (is_gpu) {
                    CHECK(velri_slot_last_run_graph_replays(slot) == 1U, "c1 vk graph replay");
                }
                velri_ref_attention_f32(1U, nh, nkvh, hd, step, max_ctx, 0.0f, 0U, true,
                                        hq, q_dim, hk, kv_dim, hv, kv_dim, hkc, hvc, ho_ref, q_dim);
                CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, ho_ref, ho, 1U, q_dim, q_dim) <= 5e-5,
                      "c1 pos_offset fallback kv_pos must match FP64 ref");
            }
            velri_slot_destroy(slot);
            velri_plan_destroy(plan);
            velri_graph_destroy(g);
            free(hq); free(hk); free(hv); free(ho); free(ho_ref); free(hkc); free(hvc);
        }
    }

    /* 16. Negative contract & dynamic_seq gating tests (Findings #3 & #10) */
    {
        const size_t rows = 4U, m = 8U, k = 16U;

        /* 16a. VELRI_OP_LINEAR rejects unexpected inputs[3]/inputs[4] when has_*_scale == false */
        {
            velri_graph *g = nullptr;
            CHECK_OK(velri_graph_create(&g), "lin neg g");
            velri_storage_id sx = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, rows * k, dev);
            velri_storage_id sw = add_store(g, VELRI_STORAGE_WEIGHT, VELRI_DTYPE_F32, m * k, dev);
            velri_storage_id sy = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, rows * m, dev);
            velri_storage_id sxs = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, rows, dev);
            velri_storage_id sws = add_store(g, VELRI_STORAGE_WEIGHT, VELRI_DTYPE_F32, m, dev);

            velri_view_id vx = add_view_2d_strided(g, sx, VELRI_DTYPE_F32, rows, k, k);
            velri_view_id vw = add_view_2d_strided(g, sw, VELRI_DTYPE_F32, m, k, k);
            velri_view_id vy = add_view_2d_strided(g, sy, VELRI_DTYPE_F32, rows, m, m);
            velri_view_id vxs = add_view_1d(g, sxs, VELRI_DTYPE_F32, rows, 0U);
            velri_view_id vws = add_view_1d(g, sws, VELRI_DTYPE_F32, m, 0U);

            velri_op_desc bad_xsc = {
                .kind = VELRI_OP_LINEAR,
                .num_inputs = 4U,
                .num_outputs = 1U,
                .inputs = {vx, vw, VELRI_INVALID_ID, vxs},
                .outputs = {vy},
                .attr.linear.desc = velri_desc_f32(),
            };
            CHECK(velri_graph_add_op(g, &bad_xsc, dev, nullptr) == VELRI_ERR_INVALID,
                  "linear must reject unexpected inputs[3] when has_x_scale == false");

            velri_op_desc bad_wsc = {
                .kind = VELRI_OP_LINEAR,
                .num_inputs = 5U,
                .num_outputs = 1U,
                .inputs = {vx, vw, VELRI_INVALID_ID, VELRI_INVALID_ID, vws},
                .outputs = {vy},
                .attr.linear.desc = velri_desc_f32(),
            };
            CHECK(velri_graph_add_op(g, &bad_wsc, dev, nullptr) == VELRI_ERR_INVALID,
                  "linear must reject unexpected inputs[4] when has_w_scale == false");
            velri_graph_destroy(g);
        }

        /* 16b. dynamic_seq=true rejects seq_len_override > static_rows; dynamic_seq=false is not truncated */
        {
            float *hx = (float *)xcalloc(rows * k, sizeof(float));
            float *hw = (float *)xcalloc(m * k, sizeof(float));
            float *hy = (float *)xcalloc(rows * m, sizeof(float));
            float *hy_ref = (float *)xcalloc(rows * m, sizeof(float));
            for (size_t r = 0U; r < rows; ++r) {
                for (size_t c = 0U; c < k; ++c) {
                    hx[r * k + c] = (float)(r + 1U) * 0.5f + (float)c * 0.01f;
                }
            }
            for (size_t r = 0U; r < m; ++r) {
                for (size_t c = 0U; c < k; ++c) {
                    hw[r * k + c] = 0.25f + (float)r * 0.05f;
                }
            }
            velri_linear_desc ldesc = velri_desc_f32();
            ref_linear_run(&ldesc, rows, m, k, hx, k, nullptr, hw, k, nullptr, nullptr, hy_ref, m);

            /* Part 1: dynamic_seq = true LINEAR strictly rejects seq_len_override > static_rows */
            {
                velri_graph *g = nullptr;
                CHECK_OK(velri_graph_create(&g), "dyn lin g");
                velri_storage_id sx = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, rows * k, dev);
                velri_storage_id sw = add_store(g, VELRI_STORAGE_WEIGHT, VELRI_DTYPE_F32, m * k, dev);
                velri_storage_id sy = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, rows * m, dev);
                velri_op_desc op = {
                    .kind = VELRI_OP_LINEAR,
                    .num_inputs = 2U,
                    .num_outputs = 1U,
                    .inputs = {add_view_2d_strided(g, sx, VELRI_DTYPE_F32, rows, k, k),
                               add_view_2d_strided(g, sw, VELRI_DTYPE_F32, m, k, k)},
                    .outputs = {add_view_2d_strided(g, sy, VELRI_DTYPE_F32, rows, m, m)},
                    .dynamic_seq = true,
                    .attr.linear.desc = ldesc,
                };
                CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "add dyn lin");
                velri_plan *plan = nullptr;
                velri_slot *slot = nullptr;
                CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prep dyn lin");
                CHECK_OK(velri_plan_bind_weight_host(plan, sw, hw, m * k * sizeof(float)), "bind w");
                CHECK_OK(velri_slot_create(plan, &slot), "slot");
                CHECK_OK(velri_slot_bind_host(slot, sx, hx, rows * k * sizeof(float)), "bind x");
                CHECK_OK(velri_slot_bind_host(slot, sy, hy, rows * m * sizeof(float)), "bind y");

                velri_step_params sp_oob = {.has_seq_len = true, .seq_len_override = rows + 1U};
                CHECK(velri_slot_run(slot, &sp_oob, nullptr) == VELRI_ERR_INVALID,
                      "dynamic_seq=true must reject seq_len_override > static_rows");
                CHECK(!velri_slot_needs_reset(slot), "pre-accept rejection must not dirty slot");

                velri_step_params sp_ok = {.has_seq_len = true, .seq_len_override = 2U};
                CHECK_OK(velri_slot_run(slot, &sp_ok, nullptr), "dyn lin valid override");
                CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, hy_ref, hy, 2U, m, m) <= 2e-4,
                      "dyn lin active rows accuracy");

                velri_slot_destroy(slot);
                velri_plan_destroy(plan);
                velri_graph_destroy(g);
            }

            /* Part 2: dynamic_seq = false LINEAR is NOT truncated when has_seq_len = true */
            {
                memset(hy, 0, rows * m * sizeof(float));
                velri_graph *g = nullptr;
                CHECK_OK(velri_graph_create(&g), "static lin g");
                velri_storage_id sx = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, rows * k, dev);
                velri_storage_id sw = add_store(g, VELRI_STORAGE_WEIGHT, VELRI_DTYPE_F32, m * k, dev);
                velri_storage_id sy = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, rows * m, dev);
                velri_op_desc op = {
                    .kind = VELRI_OP_LINEAR,
                    .num_inputs = 2U,
                    .num_outputs = 1U,
                    .inputs = {add_view_2d_strided(g, sx, VELRI_DTYPE_F32, rows, k, k),
                               add_view_2d_strided(g, sw, VELRI_DTYPE_F32, m, k, k)},
                    .outputs = {add_view_2d_strided(g, sy, VELRI_DTYPE_F32, rows, m, m)},
                    .dynamic_seq = false,
                    .attr.linear.desc = ldesc,
                };
                CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "add static lin");
                velri_plan *plan = nullptr;
                velri_slot *slot = nullptr;
                CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prep static lin");
                CHECK_OK(velri_plan_bind_weight_host(plan, sw, hw, m * k * sizeof(float)), "bind w");
                CHECK_OK(velri_slot_create(plan, &slot), "slot");
                CHECK_OK(velri_slot_bind_host(slot, sx, hx, rows * k * sizeof(float)), "bind x");
                CHECK_OK(velri_slot_bind_host(slot, sy, hy, rows * m * sizeof(float)), "bind y");

                velri_step_params sp_small = {.has_seq_len = true, .seq_len_override = 1U};
                CHECK_OK(velri_slot_run(slot, &sp_small, nullptr), "static lin run");
                CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, hy_ref, hy, rows, m, m) <= 2e-4,
                      "dynamic_seq=false linear must compute all static rows");

                velri_slot_destroy(slot);
                velri_plan_destroy(plan);
                velri_graph_destroy(g);
            }

            /* Part 3: dynamic_seq = false 2D elementwise op is NOT truncated when has_seq_len = true */
            {
                float *hy_ew = (float *)xcalloc(rows * k, sizeof(float));
                float *hy_ew_ref = (float *)xcalloc(rows * k, sizeof(float));
                ref_vec_axpby_f32(rows * k, 2.0f, hx, 0.0f, hx, hy_ew_ref);

                velri_graph *g = nullptr;
                CHECK_OK(velri_graph_create(&g), "static ew g");
                velri_storage_id sx = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, rows * k, dev);
                velri_storage_id sy = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, rows * k, dev);
                velri_op_desc op = {
                    .kind = VELRI_OP_VEC_AXPBY,
                    .num_inputs = 1U,
                    .num_outputs = 1U,
                    .inputs = {add_view_2d_strided(g, sx, VELRI_DTYPE_F32, rows, k, k)},
                    .outputs = {add_view_2d_strided(g, sy, VELRI_DTYPE_F32, rows, k, k)},
                    .dynamic_seq = false,
                    .attr.axpby = {.alpha = 2.0f, .beta = 0.0f},
                };
                CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "add static ew");
                velri_plan *plan = nullptr;
                velri_slot *slot = nullptr;
                CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prep static ew");
                CHECK_OK(velri_slot_create(plan, &slot), "slot");
                CHECK_OK(velri_slot_bind_host(slot, sx, hx, rows * k * sizeof(float)), "bind x");
                CHECK_OK(velri_slot_bind_host(slot, sy, hy_ew, rows * k * sizeof(float)), "bind y");

                velri_step_params sp_small = {.has_seq_len = true, .seq_len_override = 1U};
                CHECK_OK(velri_slot_run(slot, &sp_small, nullptr), "static ew run");
                CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, hy_ew_ref, hy_ew, rows, k, k) <= 1e-6,
                      "dynamic_seq=false elemwise must compute all static rows");

                velri_slot_destroy(slot);
                velri_plan_destroy(plan);
                velri_graph_destroy(g);
                free(hy_ew);
                free(hy_ew_ref);
            }

            free(hx);
            free(hw);
            free(hy);
            free(hy_ref);
        }

        /* 16c. 1D dynamic_seq=true elementwise op (ACTIVATION): active prefix matches, tail untouched */
        {
            const size_t total_n = 37U;
            const size_t active_n = 13U;
            const float sentinel = 42.25f;
            float *hx1d = (float *)xcalloc(total_n, sizeof(float));
            float *hy1d = (float *)xcalloc(total_n, sizeof(float));
            float *hy1d_ref = (float *)xcalloc(active_n, sizeof(float));
            uint64_t rng = 0x1D5E0001ULL;
            velri_test_fill_dtype(VELRI_DTYPE_F32, hx1d, total_n, &rng, 3.0f);
            for (size_t i = 0U; i < total_n; ++i) {
                hy1d[i] = sentinel;
            }
            ref_activation_f32(VELRI_ACT_SILU, active_n, hx1d, hy1d_ref);

            velri_graph *g = nullptr;
            CHECK_OK(velri_graph_create(&g), "dyn1d g");
            velri_storage_id sx = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, total_n, dev);
            velri_storage_id sy = add_store(g, VELRI_STORAGE_STATE, VELRI_DTYPE_F32, total_n, dev);
            velri_op_desc op = {
                .kind = VELRI_OP_ACTIVATION,
                .num_inputs = 1U,
                .num_outputs = 1U,
                .inputs = {add_view_1d(g, sx, VELRI_DTYPE_F32, total_n, 0U)},
                .outputs = {add_view_1d(g, sy, VELRI_DTYPE_F32, total_n, 0U)},
                .dynamic_seq = true,
                .attr.activation = {.kind = VELRI_ACT_SILU},
            };
            CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "add dyn1d act");
            velri_plan *plan = nullptr;
            velri_slot *slot = nullptr;
            CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prep dyn1d act");
            CHECK_OK(velri_slot_create(plan, &slot), "slot dyn1d");
            CHECK_OK(velri_slot_bind_host(slot, sx, hx1d, total_n * sizeof(float)), "bind x1d");
            CHECK_OK(velri_slot_bind_host(slot, sy, hy1d, total_n * sizeof(float)), "bind y1d");

            velri_step_params sp_oob = {.has_seq_len = true, .seq_len_override = total_n + 1U};
            CHECK(velri_slot_run(slot, &sp_oob, nullptr) == VELRI_ERR_INVALID,
                  "1D dynamic_seq=true must reject seq_len_override > dims[0]");
            CHECK(!velri_slot_needs_reset(slot), "1D dyn_seq pre-accept rejection keeps slot clean");

            velri_step_params sp_act = {.has_seq_len = true, .seq_len_override = active_n};
            CHECK_OK(velri_slot_run(slot, &sp_act, nullptr), "run dyn1d act");
            CHECK_OK(velri_slot_sync_state_to_host(slot, sy), "sync dyn1d state");
            CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, hy1d_ref, hy1d, 1U, active_n,
                                             active_n) <= 5e-4,
                  "1D dynamic_seq=true active prefix matches reference");
            for (size_t i = active_n; i < total_n; ++i) {
                CHECK(hy1d[i] == sentinel, "1D dynamic_seq=true tail elements must not be overwritten");
            }

            velri_slot_destroy(slot);
            velri_plan_destroy(plan);
            velri_graph_destroy(g);
            free(hx1d);
            free(hy1d);
            free(hy1d_ref);
        }

        /* 16d. Zero-extent dimension dims={4, 0} rejected by is_contiguous and add_view */
        {
            velri_graph *g = nullptr;
            CHECK_OK(velri_graph_create(&g), "zero dim g");
            velri_storage_id s = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, 16U, dev);
            const velri_tensor_view bad_views[2] = {
                {
                    .storage_id = s,
                    .dtype = VELRI_DTYPE_F32,
                    .rank = 2U,
                    .dims = {4U, 0U},
                    .strides = {0U, 1U},
                    .byte_offset = 0U,
                },
                {
                    .storage_id = s,
                    .dtype = VELRI_DTYPE_F32,
                    .rank = 2U,
                    .dims = {4U, 0U},
                    .strides = {1U, 1U},
                    .byte_offset = 0U,
                },
            };
            for (size_t i = 0U; i < 2U; ++i) {
                CHECK(!velri_tensor_view_is_contiguous(&bad_views[i]),
                      "dims={4,0} must return false from velri_tensor_view_is_contiguous");
                velri_view_id dummy_vid = VELRI_INVALID_ID;
                CHECK(velri_graph_add_view(g, &bad_views[i], &dummy_vid) == VELRI_ERR_INVALID,
                      "dims={4,0} must be rejected by velri_graph_add_view");
            }
            velri_graph_destroy(g);
        }

        /* 16e. dynamic_seq=true elemwise rejects mismatched dims[0] across reshaped views */
        {
            velri_graph *g = nullptr;
            CHECK_OK(velri_graph_create(&g), "reshaped dyn elem g");
            velri_storage_id sx = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, 32U, dev);
            velri_storage_id sy = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, 32U, dev);
            velri_view_id vx = add_view_2d_strided(g, sx, VELRI_DTYPE_F32, 4U, 8U, 8U);
            velri_view_id vy = add_view_2d_strided(g, sy, VELRI_DTYPE_F32, 8U, 4U, 4U);
            velri_op_desc op = {
                .kind = VELRI_OP_ACTIVATION,
                .num_inputs = 1U,
                .inputs = {vx},
                .num_outputs = 1U,
                .outputs = {vy},
                .dynamic_seq = true,
                .attr.activation.kind = VELRI_ACT_SILU,
            };
            CHECK(velri_graph_add_op(g, &op, dev, nullptr) == VELRI_ERR_INVALID,
                  "dynamic_seq=true elemwise must reject mismatched dims[0]");
            velri_graph_destroy(g);
        }
    }

    /* 17. Wide & Asymmetric Attention Core (d_k=768 shared, MLA prefix d_k=576/d_v=512, odd bidir) */
    {
        struct {
            const char *name;
            uint32_t nh, nkvh, dk, dv, max_t, max_ctx, prefill_t, dec_steps, win;
            bool causal, shared_prefix;
            float scale;
            uint64_t seed;
        } cases[3] = {
            {"attn_wide_shared768", 8U, 1U, 768U, 768U, 20U, 36U, 18U, 2U, 0U, true, true, 0.125f, 0x1A7E0001ULL},
            {"attn_mla_576_512", 8U, 1U, 576U, 512U, 12U, 24U, 9U, 2U, 5U, true, true, 0.088388f, 0x1A7E0002ULL},
            {"attn_wide_bidir_odd", 4U, 2U, 513U, 321U, 14U, 28U, 12U, 2U, 0U, false, false, 0.0f, 0x1A7E0003ULL},
        };
        for (int ci = 0; ci < 3; ++ci) {
            uint32_t nh = cases[ci].nh, nkvh = cases[ci].nkvh;
            uint32_t dk = cases[ci].dk, dv = cases[ci].dv;
            uint32_t max_t = cases[ci].max_t, max_ctx = cases[ci].max_ctx;
            size_t q_dim = (size_t)nh * dk, o_dim = (size_t)nh * dv;
            size_t k_dim = (size_t)nkvh * dk, v_dim = (size_t)nkvh * dv;
            size_t q_str = q_dim + 3U, k_str = k_dim + 5U, v_str = cases[ci].shared_prefix ? k_str : (v_dim + 3U);
            size_t o_str = o_dim + 1U;

            float *hq = (float *)xcalloc((size_t)max_t * q_str, sizeof(float));
            float *hk = (float *)xcalloc((size_t)max_t * k_str, sizeof(float));
            float *hv = cases[ci].shared_prefix ? hk : (float *)xcalloc((size_t)max_t * v_str, sizeof(float));
            float *ho = (float *)xcalloc((size_t)max_t * o_str, sizeof(float));
            float *ho_ref = (float *)xcalloc((size_t)max_t * o_str, sizeof(float));
            float *hkc_ref = (float *)xcalloc((size_t)nkvh * max_ctx * dk, sizeof(float));
            float *hvc_ref = cases[ci].shared_prefix ? hkc_ref : (float *)xcalloc((size_t)nkvh * max_ctx * dv, sizeof(float));
            size_t khs = (size_t)max_ctx * dk, krs = dk;
            size_t vhs = cases[ci].shared_prefix ? khs : ((size_t)max_ctx * dv);
            size_t vrs = cases[ci].shared_prefix ? krs : dv;

            velri_graph *g = nullptr;
            CHECK_OK(velri_graph_create(&g), "wide attn g");
            velri_storage_id sq = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, (size_t)max_t * q_str, dev);
            velri_storage_id sk = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, (size_t)max_t * k_str, dev);
            velri_storage_id sv = cases[ci].shared_prefix ? sk : add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, (size_t)max_t * v_str, dev);
            velri_storage_id so = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, (size_t)max_t * o_str, dev);
            velri_storage_id skc = add_store(g, VELRI_STORAGE_STATE, VELRI_DTYPE_F32, (size_t)nkvh * max_ctx * dk, dev);
            velri_storage_id svc = cases[ci].shared_prefix ? skc : add_store(g, VELRI_STORAGE_STATE, VELRI_DTYPE_F32, (size_t)nkvh * max_ctx * dv, dev);

            velri_tensor_view vkc, vvc;
            const size_t kc_dims[3] = {nkvh, max_ctx, dk};
            CHECK_OK(velri_tensor_view_contiguous(skc, VELRI_DTYPE_F32, 3U, kc_dims, 0U, &vkc), "vkc");
            velri_view_id id_kc = VELRI_INVALID_ID, id_vc = VELRI_INVALID_ID;
            CHECK_OK(velri_graph_add_view(g, &vkc, &id_kc), "add vkc");
            if (cases[ci].shared_prefix && dv < dk) {
                const size_t vc_dims[3] = {nkvh, max_ctx, dv};
                const size_t vc_strides[3] = {khs, krs, 1U};
                CHECK_OK(velri_tensor_view_subview(&vkc, 3U, vc_dims, vc_strides, 0U, &vvc), "vvc sub");
                CHECK_OK(velri_graph_add_view(g, &vvc, &id_vc), "add vvc");
            } else if (cases[ci].shared_prefix) {
                id_vc = id_kc;
            } else {
                const size_t vc_dims[3] = {nkvh, max_ctx, dv};
                CHECK_OK(velri_tensor_view_contiguous(svc, VELRI_DTYPE_F32, 3U, vc_dims, 0U, &vvc), "vvc");
                CHECK_OK(velri_graph_add_view(g, &vvc, &id_vc), "add vvc");
            }

            velri_view_id id_ki = add_view_2d_strided(g, sk, VELRI_DTYPE_F32, max_t, k_dim, k_str);
            velri_view_id id_vi = (cases[ci].shared_prefix && dv == dk)
                                      ? id_ki
                                      : add_view_2d_strided(g, sv, VELRI_DTYPE_F32, max_t, v_dim, v_str);

            velri_op_desc op = {
                .kind = VELRI_OP_ATTENTION,
                .num_inputs = 3U,
                .num_outputs = 3U,
                .inputs = {add_view_2d_strided(g, sq, VELRI_DTYPE_F32, max_t, q_dim, q_str), id_ki, id_vi},
                .outputs = {add_view_2d_strided(g, so, VELRI_DTYPE_F32, max_t, o_dim, o_str), id_kc, id_vc},
                .dynamic_seq = true,
                .attr.attention = {
                    .n_heads = nh, .n_kv_heads = nkvh, .head_dim = dk, .v_head_dim = dv,
                    .scale = cases[ci].scale, .sliding_window = cases[ci].win, .causal = cases[ci].causal,
                },
            };
            CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "add wide attn");
            velri_plan *plan = nullptr;
            velri_slot *slot = nullptr;
            CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prep wide attn");
            CHECK_OK(velri_slot_create(plan, &slot), "slot wide attn");
            CHECK_OK(velri_slot_bind_host(slot, sq, hq, (size_t)max_t * q_str * sizeof(float)), "q");
            CHECK_OK(velri_slot_bind_host(slot, sk, hk, (size_t)max_t * k_str * sizeof(float)), "k");
            if (!cases[ci].shared_prefix) {
                CHECK_OK(velri_slot_bind_host(slot, sv, hv, (size_t)max_t * v_str * sizeof(float)), "v");
            }
            CHECK_OK(velri_slot_bind_host(slot, so, ho, (size_t)max_t * o_str * sizeof(float)), "o");

            uint64_t rng = cases[ci].seed;
            velri_test_fill_dtype(VELRI_DTYPE_F32, hq, (size_t)cases[ci].prefill_t * q_str, &rng, 0.25f);
            velri_test_fill_dtype(VELRI_DTYPE_F32, hk, (size_t)cases[ci].prefill_t * k_str, &rng, 0.25f);
            if (!cases[ci].shared_prefix) {
                velri_test_fill_dtype(VELRI_DTYPE_F32, hv, (size_t)cases[ci].prefill_t * v_str, &rng, 0.25f);
            }
            velri_step_params sp = {.has_seq_len = true, .seq_len_override = cases[ci].prefill_t,
                                    .has_kv_len = true, .kv_len_override = 0U};
            CHECK_OK(velri_slot_run(slot, &sp, nullptr), cases[ci].name);
            CHECK(velri_slot_last_run_allocs(slot) == 0U, "wide prefill 0 allocs");
            velri_ref_attention_ex_f32(cases[ci].prefill_t, nh, nkvh, dk, dv, 0U, max_ctx,
                                       cases[ci].scale, cases[ci].win, cases[ci].causal,
                                       hq, q_str, hk, k_str, hv, v_str,
                                       hkc_ref, khs, krs, hvc_ref, vhs, vrs, ho_ref, o_str);
            CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, ho_ref, ho, cases[ci].prefill_t, o_dim, o_str) <= 2e-4,
                  cases[ci].name);

            for (uint32_t ds = 0U; ds < cases[ci].dec_steps; ++ds) {
                uint32_t kv_pos = cases[ci].prefill_t + ds;
                velri_test_fill_dtype(VELRI_DTYPE_F32, hq, q_str, &rng, 0.25f);
                velri_test_fill_dtype(VELRI_DTYPE_F32, hk, k_str, &rng, 0.25f);
                if (!cases[ci].shared_prefix) {
                    velri_test_fill_dtype(VELRI_DTYPE_F32, hv, v_str, &rng, 0.25f);
                }
                velri_step_params dsp = {.has_seq_len = true, .seq_len_override = 1U,
                                         .has_kv_len = true, .kv_len_override = kv_pos};
                CHECK_OK(velri_slot_run(slot, &dsp, nullptr), cases[ci].name);
                CHECK(velri_slot_last_run_allocs(slot) == 0U, "wide decode 0 allocs");
                velri_ref_attention_ex_f32(1U, nh, nkvh, dk, dv, kv_pos, max_ctx,
                                           cases[ci].scale, cases[ci].win, cases[ci].causal,
                                           hq, q_str, hk, k_str, hv, v_str,
                                           hkc_ref, khs, krs, hvc_ref, vhs, vrs, ho_ref, o_str);
                CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, ho_ref, ho, 1U, o_dim, o_str) <= 2e-4,
                      cases[ci].name);
            }

            velri_slot_destroy(slot);
            velri_plan_destroy(plan);
            velri_graph_destroy(g);
            free(hq); free(hk);
            if (!cases[ci].shared_prefix) free(hv);
            free(ho); free(ho_ref); free(hkc_ref);
            if (!cases[ci].shared_prefix) free(hvc_ref);
        }

        /* 17b. Negative test: prefix-aliased shared KV cache with nkvh > 1 and d_v < d_k must fail */
        {
            velri_graph *g = nullptr;
            CHECK_OK(velri_graph_create(&g), "bad prefix g");
            const uint32_t nh = 4U, nkvh = 2U, dk = 576U, dv = 512U, max_t = 4U, max_ctx = 8U;
            velri_storage_id sq = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, (size_t)max_t * nh * dk, dev);
            velri_storage_id sk = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, (size_t)max_t * nkvh * dk, dev);
            velri_storage_id so = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, (size_t)max_t * nh * dv, dev);
            velri_storage_id skc = add_store(g, VELRI_STORAGE_STATE, VELRI_DTYPE_F32, (size_t)nkvh * max_ctx * dk, dev);

            velri_tensor_view vkc, vvc;
            const size_t kc_dims[3] = {nkvh, max_ctx, dk};
            const size_t vc_dims[3] = {nkvh, max_ctx, dv};
            const size_t vc_strides[3] = {(size_t)max_ctx * dk, dk, 1U};
            CHECK_OK(velri_tensor_view_contiguous(skc, VELRI_DTYPE_F32, 3U, kc_dims, 0U, &vkc), "vkc");
            CHECK_OK(velri_tensor_view_subview(&vkc, 3U, vc_dims, vc_strides, 0U, &vvc), "vvc");
            velri_view_id id_kc = VELRI_INVALID_ID, id_vc = VELRI_INVALID_ID;
            CHECK_OK(velri_graph_add_view(g, &vkc, &id_kc), "id_kc");
            CHECK_OK(velri_graph_add_view(g, &vvc, &id_vc), "id_vc");

            velri_op_desc bad_op = {
                .kind = VELRI_OP_ATTENTION,
                .num_inputs = 3U,
                .num_outputs = 3U,
                .inputs = {
                    add_view_2d_strided(g, sq, VELRI_DTYPE_F32, max_t, (size_t)nh * dk, (size_t)nh * dk),
                    add_view_2d_strided(g, sk, VELRI_DTYPE_F32, max_t, (size_t)nkvh * dk, (size_t)nkvh * dk),
                    add_view_2d_strided(g, sk, VELRI_DTYPE_F32, max_t, (size_t)nkvh * dv, (size_t)nkvh * dk),
                },
                .outputs = {
                    add_view_2d_strided(g, so, VELRI_DTYPE_F32, max_t, (size_t)nh * dv, (size_t)nh * dv),
                    id_kc,
                    id_vc,
                },
                .dynamic_seq = true,
                .attr.attention = {
                    .n_heads = nh, .n_kv_heads = nkvh, .head_dim = dk, .v_head_dim = dv,
                    .causal = true,
                },
            };
            CHECK(velri_graph_add_op(g, &bad_op, dev, nullptr) == VELRI_ERR_INVALID,
                  "shared prefix KV cache with nkvh > 1 and dv < dk must be rejected");
            velri_graph_destroy(g);
        }
    }

    /* 18. Grouped Batched Linear (X @ W^T + b and X @ W + b, GQA groups, dynamic seq) */
    {
        struct {
            const char *name;
            uint32_t max_t, active_t, nh, ng, in_d, out_d;
            bool transposed, has_bias;
            uint64_t seed;
        } cases[4] = {
            {"blin_nn_gqa_nobias", 8U, 5U, 6U, 2U, 64U, 300U, true, false, 0xB1000001ULL},
            {"blin_nt_gqa_bias", 8U, 5U, 6U, 2U, 300U, 64U, false, true, 0xB1000002ULL},
            {"blin_nn_mha_bias", 4U, 1U, 4U, 4U, 33U, 129U, true, true, 0xB1000003ULL},
            {"blin_nt_shared_nobias", 6U, 4U, 4U, 1U, 131U, 49U, false, false, 0xB1000004ULL},
        };
        for (int ci = 0; ci < 4; ++ci) {
            uint32_t max_t = cases[ci].max_t, act_t = cases[ci].active_t;
            uint32_t nh = cases[ci].nh, ng = cases[ci].ng;
            uint32_t in_d = cases[ci].in_d, out_d = cases[ci].out_d;
            size_t x_cols = (size_t)nh * in_d, y_cols = (size_t)nh * out_d;
            size_t w_rows = cases[ci].transposed ? ((size_t)ng * in_d) : ((size_t)ng * out_d);
            size_t w_cols = cases[ci].transposed ? (size_t)out_d : (size_t)in_d;
            size_t xs = x_cols + 3U, ws = w_cols + 5U, ys = y_cols + 1U;
            size_t b_len = (size_t)ng * out_d;

            float *hx = (float *)xcalloc((size_t)max_t * xs, sizeof(float));
            float *hw = (float *)xcalloc(w_rows * ws, sizeof(float));
            float *hb = cases[ci].has_bias ? (float *)xcalloc(b_len, sizeof(float)) : nullptr;
            float *hy = (float *)xcalloc((size_t)max_t * ys, sizeof(float));
            float *hy_ref = (float *)xcalloc((size_t)max_t * ys, sizeof(float));
            uint64_t rng = cases[ci].seed;
            velri_test_fill_dtype(VELRI_DTYPE_F32, hx, (size_t)max_t * xs, &rng, 0.5f);
            velri_test_fill_dtype(VELRI_DTYPE_F32, hw, w_rows * ws, &rng, 0.2f);
            if (hb) velri_test_fill_dtype(VELRI_DTYPE_F32, hb, b_len, &rng, 0.3f);

            velri_graph *g = nullptr;
            CHECK_OK(velri_graph_create(&g), "blin g");
            velri_storage_id sx = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, (size_t)max_t * xs, dev);
            velri_storage_id sw = add_store(g, VELRI_STORAGE_WEIGHT, VELRI_DTYPE_F32, w_rows * ws, dev);
            velri_storage_id sb = cases[ci].has_bias ? add_store(g, VELRI_STORAGE_WEIGHT, VELRI_DTYPE_F32, b_len, dev) : 0U;
            velri_storage_id sy = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, (size_t)max_t * ys, dev);

            velri_op_desc op = {
                .kind = VELRI_OP_BATCHED_LINEAR,
                .num_inputs = cases[ci].has_bias ? 3U : 2U,
                .num_outputs = 1U,
                .inputs = {
                    add_view_2d_strided(g, sx, VELRI_DTYPE_F32, max_t, x_cols, xs),
                    add_view_2d_strided(g, sw, VELRI_DTYPE_F32, w_rows, w_cols, ws),
                    cases[ci].has_bias ? add_view_1d(g, sb, VELRI_DTYPE_F32, b_len, 0U) : VELRI_INVALID_ID,
                },
                .outputs = {add_view_2d_strided(g, sy, VELRI_DTYPE_F32, max_t, y_cols, ys)},
                .dynamic_seq = true,
                .attr.batched_linear = {
                    .n_heads = nh, .n_groups = ng, .in_dim = in_d, .out_dim = out_d,
                    .weight_transposed = cases[ci].transposed,
                },
            };
            CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), cases[ci].name);
            velri_plan *plan = nullptr;
            velri_slot *slot = nullptr;
            CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prep blin");
            CHECK_OK(velri_plan_bind_weight_host(plan, sw, hw, w_rows * ws * sizeof(float)), "bw");
            if (hb) CHECK_OK(velri_plan_bind_weight_host(plan, sb, hb, b_len * sizeof(float)), "bb");
            CHECK_OK(velri_slot_create(plan, &slot), "slot blin");
            CHECK_OK(velri_slot_bind_host(slot, sx, hx, (size_t)max_t * xs * sizeof(float)), "bx");
            CHECK_OK(velri_slot_bind_host(slot, sy, hy, (size_t)max_t * ys * sizeof(float)), "by");

            velri_step_params sp = {.has_seq_len = true, .seq_len_override = act_t};
            CHECK_OK(velri_slot_run(slot, &sp, nullptr), cases[ci].name);
            CHECK(velri_slot_last_run_allocs(slot) == 0U, "blin 0 allocs");
            velri_ref_batched_linear_f32(act_t, nh, ng, in_d, out_d, cases[ci].transposed,
                                         hx, xs, hw, ws, hb, hy_ref, ys);
            CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, hy_ref, hy, act_t, y_cols, ys) <= 1e-4,
                  cases[ci].name);

            velri_slot_destroy(slot);
            velri_plan_destroy(plan);
            velri_graph_destroy(g);
            free(hx); free(hw); free(hb); free(hy); free(hy_ref);
        }
    }

    /* 19. Unlimited >32-op Single-Batch + DAG Barrier Elision + Persistent Graph Replay */
    {
        const size_t n_branches = 20U;
        const size_t n_elems = 64U;
        float *hx = (float *)xcalloc(n_elems, sizeof(float));
        float *hy = (float *)xcalloc(n_elems, sizeof(float));
        for (size_t i = 0U; i < n_elems; ++i) {
            hx[i] = 0.25f + (float)i * 0.01f;
        }

        velri_graph *g = nullptr;
        CHECK_OK(velri_graph_create(&g), "batch40 g");
        velri_storage_id sx = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, n_elems, dev);
        velri_view_id vx = add_view_1d(g, sx, VELRI_DTYPE_F32, n_elems, 0U);
        velri_storage_id sout = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, n_elems, dev);
        velri_view_id vout = add_view_1d(g, sout, VELRI_DTYPE_F32, n_elems, 0U);
        velri_storage_id branch_out = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, n_elems, dev);
        float *branch_host = (float *)xcalloc(n_elems, sizeof(float));

        /* The last parallel branch is also observable as a second graph output. */
        velri_view_id br_views[20];
        for (size_t b = 0U; b < n_branches; ++b) {
            velri_storage_id sb = b + 1U == n_branches ? branch_out :
                add_store(g, VELRI_STORAGE_SCRATCH, VELRI_DTYPE_F32, n_elems, dev);
            br_views[b] = add_view_1d(g, sb, VELRI_DTYPE_F32, n_elems, 0U);
            velri_op_desc op = {
                .kind = VELRI_OP_VEC_AXPBY,
                .num_inputs = 1U,
                .num_outputs = 1U,
                .inputs = {vx},
                .outputs = {br_views[b]},
                .attr.axpby = {.alpha = 1.0f + 0.05f * (float)b, .beta = 0.0f},
            };
            CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "add parallel branch");
        }
        /* Reduce all 20 branches via 19 sequential adds + 1 final scale = 40 ops (> 32 old cap!). */
        velri_view_id cur = br_views[0];
        for (size_t b = 1U; b < n_branches; ++b) {
            velri_storage_id sc = add_store(g, VELRI_STORAGE_SCRATCH, VELRI_DTYPE_F32, n_elems, dev);
            velri_view_id dst = add_view_1d(g, sc, VELRI_DTYPE_F32, n_elems, 0U);
            velri_op_desc op = {
                .kind = VELRI_OP_VEC_AXPBY,
                .num_inputs = 2U,
                .num_outputs = 1U,
                .inputs = {cur, br_views[b]},
                .outputs = {dst},
                .attr.axpby = {.alpha = 1.0f, .beta = 1.0f},
            };
            CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "add chain add");
            cur = dst;
        }
        velri_op_desc fin = {
            .kind = VELRI_OP_VEC_AXPBY,
            .num_inputs = 1U,
            .num_outputs = 1U,
            .inputs = {cur},
            .outputs = {vout},
            .attr.axpby = {.alpha = 0.5f, .beta = 0.0f},
        };
        CHECK_OK(velri_graph_add_op(g, &fin, dev, nullptr), "add final scale");

        velri_plan *plan = nullptr;
        CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prep batch40");
        uint32_t total_ops = 0U, elided = 0U;
        CHECK_OK(velri_plan_barrier_stats(plan, &total_ops, &elided), "barrier stats");
        CHECK(total_ops == 40U, "40 ops in plan");
        CHECK(elided >= 19U, "DAG analyzer must elide >=19 barriers across 20 parallel branches");

        velri_slot *slot = nullptr;
        CHECK_OK(velri_slot_create(plan, &slot), "slot batch40");
        CHECK_OK(velri_slot_bind_host(slot, sx, hx, n_elems * sizeof(float)), "bind x40");
        CHECK_OK(velri_slot_bind_host(slot, sout, hy, n_elems * sizeof(float)), "bind y40");
        CHECK_OK(velri_slot_bind_host(slot, branch_out, branch_host, n_elems * sizeof(float)), "bind branch");

        velri_step_params sp = {.has_seq_len = true, .seq_len_override = 1U};
        CHECK_OK(velri_slot_run(slot, &sp, nullptr), "run batch40");
        CHECK(velri_slot_last_run_allocs(slot) == 0U, "batch40 0 allocs");
        if (is_gpu) {
            CHECK(velri_slot_last_run_graph_replays(slot) == 1U,
                  "Vulkan must replay persistent captured graph for 40-op batch");
        }
        /* Expected scale: 0.5 * sum_{b=0..19} (1 + 0.05 * b) = 0.5 * (20 + 0.05 * 190) = 14.75 */
        for (size_t i = 0U; i < n_elems; ++i) {
            double ref = 14.75 * (double)hx[i];
            CHECK(fabs((double)hy[i] - ref) <= 1e-4, "batch40 accuracy");
        }

        CHECK_OK(velri_slot_bind_host(slot, sout, branch_host, n_elems * sizeof(float)), "rebind sum");
        CHECK_OK(velri_slot_bind_host(slot, branch_out, hy, n_elems * sizeof(float)), "rebind branch");
        for (size_t i = 0U; i < n_elems; ++i) hx[i] += 1.0f;
        CHECK_OK(velri_slot_run(slot, &sp, nullptr), "replay with rebound outputs");
        for (size_t i = 0U; i < n_elems; ++i) {
            CHECK(fabs((double)branch_host[i] - 14.75 * hx[i]) <= 1e-4, "rebound sum");
            CHECK(fabs((double)hy[i] - 1.95 * hx[i]) <= 1e-5, "rebound branch");
        }
        free(branch_host);
        velri_slot_destroy(slot);
        velri_plan_destroy(plan);
        velri_graph_destroy(g);
        free(hx);
        free(hy);
    }

    /* 20. C2 regression: static Split-K GEMM must NOT be silently captured without Split-K */
    if (is_gpu) {
        const size_t rows = 64U, m = 512U, k = 1024U;
        float *hx = (float *)xcalloc(rows * k, sizeof(float));
        float *hw = (float *)xcalloc(m * k, sizeof(float));
        float *hy = (float *)xcalloc(rows * m, sizeof(float));
        float *hy_ref = (float *)xcalloc(rows * m, sizeof(float));
        uint64_t rng = 0xC25B1701ULL;
        velri_test_fill_dtype(VELRI_DTYPE_F32, hx, rows * k, &rng, 0.25f);
        velri_test_fill_dtype(VELRI_DTYPE_F32, hw, m * k, &rng, 0.25f);
        velri_linear_desc ld = velri_desc_f32();
        ref_linear_run(&ld, rows, m, k, hx, k, nullptr, hw, k, nullptr, nullptr, hy_ref, m);

        velri_graph *g = nullptr;
        CHECK_OK(velri_graph_create(&g), "c2 splitk g");
        velri_storage_id sx = add_store(g, VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, rows * k, dev);
        velri_storage_id sw = add_store(g, VELRI_STORAGE_WEIGHT, VELRI_DTYPE_F32, m * k, dev);
        velri_storage_id sy = add_store(g, VELRI_STORAGE_OUTPUT, VELRI_DTYPE_F32, rows * m, dev);
        velri_op_desc op = {
            .kind = VELRI_OP_LINEAR,
            .num_inputs = 2U,
            .num_outputs = 1U,
            .inputs = {add_view_2d_strided(g, sx, VELRI_DTYPE_F32, rows, k, k),
                       add_view_2d_strided(g, sw, VELRI_DTYPE_F32, m, k, k)},
            .outputs = {add_view_2d_strided(g, sy, VELRI_DTYPE_F32, rows, m, m)},
            .dynamic_seq = false,
            .attr.linear.desc = ld,
        };
        CHECK_OK(velri_graph_add_op(g, &op, dev, nullptr), "c2 add splitk");
        velri_plan *plan = nullptr;
        velri_slot *slot = nullptr;
        CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "c2 prep");
        CHECK_OK(velri_plan_bind_weight_host(plan, sw, hw, m * k * sizeof(float)), "c2 bw");
        CHECK_OK(velri_slot_create(plan, &slot), "c2 slot");
        CHECK_OK(velri_slot_bind_host(slot, sx, hx, rows * k * sizeof(float)), "c2 bx");
        CHECK_OK(velri_slot_bind_host(slot, sy, hy, rows * m * sizeof(float)), "c2 by");
        CHECK_OK(velri_slot_run(slot, nullptr, nullptr), "c2 run");
        CHECK(velri_slot_last_run_graph_replays(slot) == 0U,
              "Split-K GEMM must not be bypassed by non-Split-K graph capture");
        CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, hy_ref, hy, rows, m, m) <= 5e-3,
              "c2 splitk gemm accuracy");
        velri_slot_destroy(slot);
        velri_plan_destroy(plan);
        velri_graph_destroy(g);
        free(hx); free(hw); free(hy); free(hy_ref);
    }

    printf("  [PASS] %-14s — all 16 ops + Split-KV + 40-op DAG batch verified vs FP64 reference.\n", tag);
}

int main(void) {
    printf("=== Velri-lib Unified Kernel Correctness Suite ===\n");

    const velri_cpu_variant vars[3] = {VELRI_CPU_BASE, VELRI_CPU_AVX2, VELRI_CPU_AVX512};
    const char *vnames[3] = {"cpu:base", "cpu:avx2", "cpu:avx512"};
    size_t nvars = 1U;
#if VELRI_X86_VARIANTS
    nvars = 3U;
#endif

    for (size_t vi = 0U; vi < nvars; ++vi) {
        if (!velri_cpu_variant_supported(vars[vi])) continue;
        velri_context_config cfg = {
            .cpu_workers = 4U,
            .cpu_variant = vars[vi],
            .enable_vulkan = false,
        };
        velri_context *ctx = nullptr;
        CHECK_OK(velri_context_create(&cfg, &ctx), "cpu ctx");
        velri_device_id cpu_dev = 0U;
        CHECK_OK(velri_context_find_device(ctx, "cpu", 0U, &cpu_dev), "find cpu");
        verify_all_ops_on_device(ctx, cpu_dev, vnames[vi], false);
        verify_w8a8_weight_packing(ctx, cpu_dev, vars[vi]);
        verify_linear_one(ctx, cpu_dev, vnames[vi], "packed W8A8 joined panels and tails",
                          velri_desc_w8a8_f32(), 17U, 1031U, 259U, 3U, 5U, 7U, true,
                          0.0, LINEAR_PACKED_W8A8_EXTREMES);
        verify_linear_one(ctx, cpu_dev, vnames[vi], "packed W8A8 GEMV signed overflow",
                          velri_desc_w8a8_f32(), 3U, 9U, 132105U, 2U, 4U, 5U, false,
                          0.0, LINEAR_PACKED_W8A8_EXTREMES);
        verify_linear_one(ctx, cpu_dev, vnames[vi], "packed W8A8 GEMM signed overflow",
                          velri_desc_w8a8_f32(), 14U, 5U, 132106U, 1U, 5U, 7U, true,
                          0.0, LINEAR_PACKED_W8A8_EXTREMES);
        velri_context_destroy(ctx);
    }

#ifdef VELRI_CORE_HAS_VULKAN
    velri_context_config vcfg = {
        .cpu_workers = 1U,
        .cpu_variant = VELRI_CPU_AUTO,
        .enable_vulkan = true,
    };
    velri_context *vctx = nullptr;
    CHECK_OK(velri_context_create(&vcfg, &vctx), "vk ctx");
    velri_device_id vk_dev = 0U;
    if (velri_context_find_device(vctx, "vulkan", 0U, &vk_dev) == VELRI_OK) {
        velri_device_caps caps;
        CHECK_OK(velri_context_device_caps(vctx, vk_dev, &caps), "vk caps");
        printf("  Vulkan GPU: %s (UMA=%d, VRAM=%.1f GiB, align=%zu)\n", caps.device_name,
               (int)caps.is_uma, (double)caps.mem_total_bytes / (1024.0 * 1024.0 * 1024.0),
               caps.min_align_bytes);
        verify_all_ops_on_device(vctx, vk_dev, "vulkan:0", true);
    } else {
        printf("  [SKIP] vulkan:0 (no Vulkan 1.3 GPU available at runtime).\n");
    }
    if (velri_context_find_device(vctx, "vulkan", 1U, &vk_dev) == VELRI_OK) {
        verify_linear_one(vctx, vk_dev, "vulkan:1", "first use of secondary device",
                          velri_desc_f32(), 1U, 4U, 8U, 0U, 0U, 0U, false, 1e-4, LINEAR_RANDOM);
    }
    velri_context_destroy(vctx);
#endif

    printf("=== All Unified Kernel Tests PASSED ===\n");
    return 0;
}
