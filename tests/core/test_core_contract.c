#include "mock_async_backend.h"
#include "os.h"
#include "reference.h"
#include "velri_lib/velri_lib.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Core Runtime & Backend Contract Test Suite:
 *   1. TensorView/subview bounds, aliasing/mutation rejection, host alignment validation.
 *   2. SwiGLU MLP graph with plan-level shared WEIGHTs, scratch reuse, and 2 concurrent slots.
 *   3. No silent CPU fallback when a chosen device rejects an op.
 *   4. Cross-backend RAM transfer bus + read-only dedup + RMW & partial-view coherence (A1, B8).
 *   5. Shared WEIGHT uploads once across slots (B1) & resident STATE without per-step download (B3).
 *   6. Refcounted ctx/plan lifetimes outlived by slot (B6) & graph growth > 32 storages (A8).
 *   7. Pre-accept rejection (incl. bad RoPE pos_offset_override [A4]), post/partial-accept drain,
 *      and cancellation.
 *   8. Hybrid CPU <-> Vulkan graph with batched Vulkan command buffers.
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

static velri_storage_id add_f32_storage(velri_graph *g, velri_storage_kind kind, size_t elems,
                                        velri_device_id dev) {
    velri_storage_desc d = {
        .kind = kind,
        .dtype = VELRI_DTYPE_F32,
        .byte_size = elems * sizeof(float),
        .align = 64U,
        .device_id = dev,
    };
    velri_storage_id id = 0U;
    CHECK_OK(velri_graph_add_storage(g, &d, &id), "add_storage");
    return id;
}

static velri_view_id add_2d_view(velri_graph *g, velri_storage_id sid, size_t r, size_t c) {
    size_t dims[2] = {r, c};
    velri_tensor_view v;
    CHECK_OK(velri_tensor_view_contiguous(sid, VELRI_DTYPE_F32, 2U, dims, 0U, &v), "view2d");
    velri_view_id vid = 0U;
    CHECK_OK(velri_graph_add_view(g, &v, &vid), "add_view2d");
    return vid;
}

static velri_view_id add_1d_view_off(velri_graph *g, velri_storage_id sid, size_t n,
                                     size_t byte_off) {
    size_t dims[1] = {n};
    velri_tensor_view v;
    CHECK_OK(velri_tensor_view_contiguous(sid, VELRI_DTYPE_F32, 1U, dims, byte_off, &v), "view1d");
    velri_view_id vid = 0U;
    CHECK_OK(velri_graph_add_view(g, &v, &vid), "add_view1d");
    return vid;
}

static velri_view_id add_1d_view(velri_graph *g, velri_storage_id sid, size_t n) {
    return add_1d_view_off(g, sid, n, 0U);
}

/* ---- Test 1: TensorView/subview bounds, graph immutability, aliasing, alignment ---- */
static void test_validation_and_guards(velri_context *ctx, velri_device_id cpu_dev) {
    velri_graph *g = nullptr;
    CHECK_OK(velri_graph_create(&g), "graph_create");

    velri_storage_id s_in = add_f32_storage(g, VELRI_STORAGE_INPUT, 16U, cpu_dev);
    velri_storage_id s_wt = add_f32_storage(g, VELRI_STORAGE_WEIGHT, 16U, cpu_dev);
    velri_storage_id s_out = add_f32_storage(g, VELRI_STORAGE_OUTPUT, 16U, cpu_dev);
    velri_storage_id s_st = add_f32_storage(g, VELRI_STORAGE_STATE, 16U, cpu_dev);

    /* Subview escaping base span must be rejected (B8). */
    size_t bdims[1] = {8U}, sdims[1] = {6U}, sstr[1] = {1U};
    velri_tensor_view base_v, sub_v;
    CHECK_OK(velri_tensor_view_contiguous(s_st, VELRI_DTYPE_F32, 1U, bdims, 0U, &base_v), "base");
    CHECK(velri_tensor_view_subview(&base_v, 1U, sdims, sstr, 16U, &sub_v) == VELRI_ERR_INVALID,
          "subview escaping base rejected");
    CHECK_OK(velri_tensor_view_subview(&base_v, 1U, sdims, sstr, 8U, &sub_v), "valid subview");

    /* Out-of-bounds view must be rejected cleanly. */
    size_t oob_dims[1] = {32U};
    velri_tensor_view oob_v;
    CHECK_OK(velri_tensor_view_contiguous(s_st, VELRI_DTYPE_F32, 1U, oob_dims, 0U, &oob_v), "oob");
    velri_view_id dummy = 0U;
    CHECK(velri_graph_add_view(g, &oob_v, &dummy) == VELRI_ERR_INVALID, "oob view rejected");

    /* Writing immutable WEIGHT or INPUT must be rejected (B8). */
    velri_view_id vin = add_1d_view(g, s_in, 16U);
    velri_view_id vwt = add_1d_view(g, s_wt, 16U);
    velri_view_id vout = add_1d_view(g, s_out, 16U);
    velri_op_desc bad_write_wt = {
        .kind = VELRI_OP_ACTIVATION, .num_inputs = 1U, .num_outputs = 1U,
        .inputs = {vin}, .outputs = {vwt}, .attr.activation = {.kind = VELRI_ACT_SILU},
    };
    CHECK(velri_graph_add_op(g, &bad_write_wt, cpu_dev, nullptr) == VELRI_ERR_INVALID,
          "writing WEIGHT rejected");
    velri_op_desc bad_write_in = {
        .kind = VELRI_OP_ACTIVATION, .num_inputs = 1U, .num_outputs = 1U,
        .inputs = {vwt}, .outputs = {vin}, .attr.activation = {.kind = VELRI_ACT_SILU},
    };
    CHECK(velri_graph_add_op(g, &bad_write_in, cpu_dev, nullptr) == VELRI_ERR_INVALID,
          "writing INPUT rejected");

    /* Shifted partial overlap between input and output must be rejected (A10). */
    velri_view_id vst0 = add_1d_view_off(g, s_st, 8U, 0U);
    velri_view_id vst1 = add_1d_view_off(g, s_st, 8U, 4U);
    velri_op_desc bad_overlap = {
        .kind = VELRI_OP_ACTIVATION, .num_inputs = 1U, .num_outputs = 1U,
        .inputs = {vst0}, .outputs = {vst1}, .attr.activation = {.kind = VELRI_ACT_SILU},
    };
    CHECK(velri_graph_add_op(g, &bad_overlap, cpu_dev, nullptr) == VELRI_ERR_INVALID,
          "shifted partial overlap rejected");

    /* Unaligned host pointer binding must be rejected (A15). */
    velri_op_desc ok_op = {
        .kind = VELRI_OP_VEC_AXPBY, .num_inputs = 2U, .num_outputs = 1U,
        .inputs = {vin, vwt}, .outputs = {vout}, .attr.axpby = {.alpha = 1.0f, .beta = 1.0f},
    };
    CHECK_OK(velri_graph_add_op(g, &ok_op, cpu_dev, nullptr), "ok_op");
    velri_plan *plan = nullptr;
    CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prepare");
    alignas(16) uint8_t raw[128] = {0};
    CHECK(velri_plan_bind_weight_host(plan, s_wt, raw + 1U, 64U) == VELRI_ERR_INVALID,
          "unaligned weight ptr rejected");
    velri_slot *slot = nullptr;
    CHECK_OK(velri_slot_create(plan, &slot), "slot");
    CHECK(velri_slot_bind_host(slot, s_in, raw + 2U, 64U) == VELRI_ERR_INVALID,
          "unaligned slot host ptr rejected");
    CHECK(velri_slot_bind_host(slot, s_wt, raw, 64U) == VELRI_ERR_INVALID,
          "binding WEIGHT via slot_bind_host rejected");

    velri_slot_destroy(slot);
    velri_plan_destroy(plan);
    velri_graph_destroy(g);
    printf("  [PASS] TensorView/subview bounds, immutability, overlap, and alignment guards.\n");
}

/* ---- Test 2: SwiGLU MLP graph + plan-level shared weights + 2 concurrent slots ---- */
typedef struct {
    velri_slot *slot;
    velri_status st;
} slot_thread_args;

static void slot_runner_thread(void *arg) {
    slot_thread_args *a = (slot_thread_args *)arg;
    a->st = velri_slot_run(a->slot, nullptr, nullptr);
}

static void ref_swiglu_mlp_block(size_t rows, size_t d, size_t ff, float eps, const float *x,
                                 const float *norm_w, const float *w_gate, const float *w_up,
                                 const float *w_down, float *y_out) {
    float *xn = (float *)calloc(rows * d, sizeof(float));
    float *g = (float *)calloc(rows * ff, sizeof(float));
    float *u = (float *)calloc(rows * ff, sizeof(float));
    float *h = (float *)calloc(rows * ff, sizeof(float));
    float *down = (float *)calloc(rows * d, sizeof(float));
    velri_linear_desc f32_lin = velri_desc_f32();
    ref_rmsnorm(VELRI_DTYPE_F32, VELRI_DTYPE_F32, rows, d, eps, false, x, d, norm_w, xn, d);
    ref_linear_run(&f32_lin, rows, ff, d, xn, d, nullptr, w_gate, d, nullptr, nullptr, g, ff);
    ref_linear_run(&f32_lin, rows, ff, d, xn, d, nullptr, w_up, d, nullptr, nullptr, u, ff);
    ref_gated_mlp_f32(VELRI_ACT_SILU, rows * ff, 0.0f, g, u, h);
    ref_linear_run(&f32_lin, rows, d, ff, h, ff, nullptr, w_down, ff, nullptr, nullptr, down, d);
    ref_vec_axpby_f32(rows * d, 1.0f, x, 1.0f, down, y_out);
    free(xn); free(g); free(u); free(h); free(down);
}

static void test_standalone_mlp_two_concurrent_slots(velri_context *ctx, velri_device_id dev,
                                                     bool expect_replay) {
    const size_t rows = expect_replay ? 1U : 8U, d = 64U, ff = 128U;
    const float eps = 1e-5f;

    velri_graph *g = nullptr;
    CHECK_OK(velri_graph_create(&g), "mlp graph");
    velri_storage_id s_x = add_f32_storage(g, VELRI_STORAGE_INPUT, rows * d, dev);
    velri_storage_id s_nw = add_f32_storage(g, VELRI_STORAGE_WEIGHT, d, dev);
    velri_storage_id s_wg = add_f32_storage(g, VELRI_STORAGE_WEIGHT, ff * d, dev);
    velri_storage_id s_wu = add_f32_storage(g, VELRI_STORAGE_WEIGHT, ff * d, dev);
    velri_storage_id s_wd = add_f32_storage(g, VELRI_STORAGE_WEIGHT, d * ff, dev);
    velri_storage_id s_xn = add_f32_storage(g, VELRI_STORAGE_SCRATCH, rows * d, dev);
    velri_storage_id s_g = add_f32_storage(g, VELRI_STORAGE_SCRATCH, rows * ff, dev);
    velri_storage_id s_u = add_f32_storage(g, VELRI_STORAGE_SCRATCH, rows * ff, dev);
    velri_storage_id s_h = add_f32_storage(g, VELRI_STORAGE_SCRATCH, rows * ff, dev);
    velri_storage_id s_dn = add_f32_storage(g, VELRI_STORAGE_SCRATCH, rows * d, dev);
    velri_storage_id s_y = add_f32_storage(g, VELRI_STORAGE_OUTPUT, rows * d, dev);

    velri_view_id vx = add_2d_view(g, s_x, rows, d);
    velri_view_id vnw = add_1d_view(g, s_nw, d);
    velri_view_id vwg = add_2d_view(g, s_wg, ff, d);
    velri_view_id vwu = add_2d_view(g, s_wu, ff, d);
    velri_view_id vwd = add_2d_view(g, s_wd, d, ff);
    velri_view_id vxn = add_2d_view(g, s_xn, rows, d);
    velri_view_id vg = add_2d_view(g, s_g, rows, ff);
    velri_view_id vu = add_2d_view(g, s_u, rows, ff);
    velri_view_id vh = add_2d_view(g, s_h, rows, ff);
    velri_view_id vdn = add_2d_view(g, s_dn, rows, d);
    velri_view_id vy = add_2d_view(g, s_y, rows, d);

    velri_op_desc ops[6] = {
        {.kind = VELRI_OP_RMSNORM, .num_inputs = 2U, .num_outputs = 1U,
         .inputs = {vx, vnw}, .outputs = {vxn}, .attr.rmsnorm = {.eps = eps}},
        {.kind = VELRI_OP_LINEAR, .num_inputs = 2U, .num_outputs = 1U,
         .inputs = {vxn, vwg}, .outputs = {vg}, .attr.linear.desc = velri_desc_f32()},
        {.kind = VELRI_OP_LINEAR, .num_inputs = 2U, .num_outputs = 1U,
         .inputs = {vxn, vwu}, .outputs = {vu}, .attr.linear.desc = velri_desc_f32()},
        {.kind = VELRI_OP_GATED_MLP, .num_inputs = 2U, .num_outputs = 1U,
         .inputs = {vg, vu}, .outputs = {vh}, .attr.gated_mlp = {.act = VELRI_ACT_SILU}},
        {.kind = VELRI_OP_LINEAR, .num_inputs = 2U, .num_outputs = 1U,
         .inputs = {vh, vwd}, .outputs = {vdn}, .attr.linear.desc = velri_desc_f32()},
        {.kind = VELRI_OP_VEC_AXPBY, .num_inputs = 2U, .num_outputs = 1U,
         .inputs = {vx, vdn}, .outputs = {vy}, .attr.axpby = {.alpha = 1.0f, .beta = 1.0f}},
    };
    for (int i = 0; i < 6; ++i) CHECK_OK(velri_graph_add_op(g, &ops[i], dev, nullptr), "add");

    velri_plan *plan = nullptr;
    CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prepare mlp");
    velri_memory_report rep;
    CHECK_OK(velri_plan_memory_report(plan, &rep), "report");
    CHECK(rep.devices[dev].scratch_packed_bytes < rep.devices[dev].scratch_unpacked_bytes,
          "scratch packer reuses memory");
    CHECK(rep.num_ram_transfers == 0U, "single-device 0 RAM transfers");

    uint64_t rng = 0x123456789ABCDEFULL;
    float *nw = (float *)calloc(d, sizeof(float));
    float *wg = (float *)calloc(ff * d, sizeof(float));
    float *wu = (float *)calloc(ff * d, sizeof(float));
    float *wd = (float *)calloc(d * ff, sizeof(float));
    float *x0 = (float *)calloc(rows * d, sizeof(float));
    float *x1 = (float *)calloc(rows * d, sizeof(float));
    float *y0 = (float *)calloc(rows * d, sizeof(float));
    float *y1 = (float *)calloc(rows * d, sizeof(float));
    float *ref0 = (float *)calloc(rows * d, sizeof(float));
    float *ref1 = (float *)calloc(rows * d, sizeof(float));

    for (size_t i = 0; i < d; ++i) nw[i] = velri_test_rand_f32(&rng, 0.5f, 1.5f);
    velri_test_fill_dtype(VELRI_DTYPE_F32, wg, ff * d, &rng, 0.25f);
    velri_test_fill_dtype(VELRI_DTYPE_F32, wu, ff * d, &rng, 0.25f);
    velri_test_fill_dtype(VELRI_DTYPE_F32, wd, d * ff, &rng, 0.25f);
    velri_test_fill_dtype(VELRI_DTYPE_F32, x0, rows * d, &rng, 1.0f);
    velri_test_fill_dtype(VELRI_DTYPE_F32, x1, rows * d, &rng, 2.0f);
    ref_swiglu_mlp_block(rows, d, ff, eps, x0, nw, wg, wu, wd, ref0);
    ref_swiglu_mlp_block(rows, d, ff, eps, x1, nw, wg, wu, wd, ref1);

    /* Bind immutable weights ONCE on the plan (shared read-only across slots). */
    CHECK_OK(velri_plan_bind_weight_host(plan, s_nw, nw, d * sizeof(float)), "bind nw");
    CHECK_OK(velri_plan_bind_weight_host(plan, s_wg, wg, ff * d * sizeof(float)), "bind wg");
    CHECK_OK(velri_plan_bind_weight_host(plan, s_wu, wu, ff * d * sizeof(float)), "bind wu");
    CHECK_OK(velri_plan_bind_weight_host(plan, s_wd, wd, d * ff * sizeof(float)), "bind wd");

    velri_slot *slot0 = nullptr, *slot1 = nullptr;
    CHECK_OK(velri_slot_create(plan, &slot0), "slot0");
    CHECK_OK(velri_slot_create(plan, &slot1), "slot1");
    CHECK_OK(velri_slot_bind_host(slot0, s_x, x0, rows * d * sizeof(float)), "x0");
    CHECK_OK(velri_slot_bind_host(slot1, s_x, x1, rows * d * sizeof(float)), "x1");

    for (int iter = 0; iter < 4; ++iter) {
        float *dst0 = (iter & 1) ? y1 : y0;
        float *dst1 = (iter & 1) ? y0 : y1;
        memset(dst0, 0, rows * d * sizeof(float));
        memset(dst1, 0, rows * d * sizeof(float));
        CHECK_OK(velri_slot_bind_host(slot0, s_y, dst0, rows * d * sizeof(float)), "rebind y0");
        CHECK_OK(velri_slot_bind_host(slot1, s_y, dst1, rows * d * sizeof(float)), "rebind y1");
        slot_thread_args a0 = {.slot = slot0, .st = VELRI_ERR_EXEC};
        slot_thread_args a1 = {.slot = slot1, .st = VELRI_ERR_EXEC};
        velri_os_thread *t0 = nullptr, *t1 = nullptr;
        CHECK(velri_os_thread_spawn(&t0, slot_runner_thread, &a0), "spawn t0");
        CHECK(velri_os_thread_spawn(&t1, slot_runner_thread, &a1), "spawn t1");
        velri_os_thread_join(t0);
        velri_os_thread_join(t1);
        CHECK_OK(a0.st, "slot0 run");
        CHECK_OK(a1.st, "slot1 run");
        CHECK(velri_slot_last_run_allocs(slot0) == 0U && velri_slot_last_run_allocs(slot1) == 0U,
              "0 B hot-path allocs");
        CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, ref0, dst0, rows, d, d) < 1e-4,
              "first slot writes its current host binding");
        CHECK(velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, ref1, dst1, rows, d, d) < 1e-4,
              "second slot writes its current host binding");
        if (expect_replay) {
            CHECK(velri_slot_last_run_graph_replays(slot0) == 1U &&
                  velri_slot_last_run_graph_replays(slot1) == 1U, "both slots replayed");
        }
    }

    velri_slot_destroy(slot0);
    velri_slot_destroy(slot1);
    velri_plan_destroy(plan);
    velri_graph_destroy(g);
    free(nw); free(wg); free(wu); free(wd);
    free(x0); free(x1); free(y0); free(y1); free(ref0); free(ref1);
    printf("  [PASS] SwiGLU MLP + plan-shared weights + scratch reuse + 2 concurrent slots.\n");
}

/* ---- Test 3: No silent CPU fallback ---- */
static void test_no_silent_cpu_fallback(velri_context *ctx, velri_device_id mock_dev) {
    velri_graph *g = nullptr;
    CHECK_OK(velri_graph_create(&g), "fallback g");
    velri_storage_id sx = add_f32_storage(g, VELRI_STORAGE_INPUT, 64U, mock_dev);
    velri_storage_id sy = add_f32_storage(g, VELRI_STORAGE_OUTPUT, 64U, mock_dev);
    velri_op_desc op = {
        .kind = VELRI_OP_RMSNORM, .num_inputs = 1U, .num_outputs = 1U,
        .inputs = {add_2d_view(g, sx, 4U, 16U)}, .outputs = {add_2d_view(g, sy, 4U, 16U)},
        .attr.rmsnorm = {.eps = 1e-5f},
    };
    CHECK_OK(velri_graph_add_op(g, &op, mock_dev, nullptr), "add unsupported");
    velri_plan *plan = nullptr;
    velri_plan_options strict = {.allow_cpu_fallback = false};
    CHECK(velri_plan_prepare(ctx, g, &strict, &plan) == VELRI_ERR_UNSUPPORTED && !plan,
          "strict rejects without silent CPU fallback");
    velri_plan_options fb = {.allow_cpu_fallback = true};
    CHECK_OK(velri_plan_prepare(ctx, g, &fb, &plan), "explicit fallback succeeds");
    velri_plan_destroy(plan);
    velri_graph_destroy(g);
    printf("  [PASS] No silent CPU fallback (strict reject + explicit opt-in).\n");
}

/* ---- Test 4: Cross-device coherence (A1 RMW + partial-view write) & read-only dedup (B8) ---- */
static void test_cross_device_coherence_and_dedup(velri_context *ctx, velri_device_id cpu_dev,
                                                  velri_device_id mock_dev) {
    /* 4a. Read-only dedup: reading a cpu INPUT twice across 2 ops on mock_async emits 1 transfer. */
    {
        velri_mock_async_reset_ctrl();
        const size_t n = 32U;
        velri_graph *g = nullptr;
        CHECK_OK(velri_graph_create(&g), "dedup g");
        velri_storage_id s_x = add_f32_storage(g, VELRI_STORAGE_INPUT, n, cpu_dev);
        velri_storage_id s_t = add_f32_storage(g, VELRI_STORAGE_SCRATCH, n, mock_dev);
        velri_storage_id s_y = add_f32_storage(g, VELRI_STORAGE_OUTPUT, n, cpu_dev);
        velri_view_id vx = add_1d_view(g, s_x, n);
        velri_view_id vt = add_1d_view(g, s_t, n);
        velri_view_id vy = add_1d_view(g, s_y, n);

        velri_op_desc op0 = {
            .kind = VELRI_OP_VEC_AXPBY, .num_inputs = 1U, .num_outputs = 1U,
            .inputs = {vx}, .outputs = {vt}, .attr.axpby = {.alpha = 2.0f, .beta = 0.0f},
        };
        velri_op_desc op1 = {
            .kind = VELRI_OP_VEC_AXPBY, .num_inputs = 2U, .num_outputs = 1U,
            .inputs = {vt, vx}, .outputs = {vy}, .attr.axpby = {.alpha = 1.0f, .beta = 3.0f},
        };
        CHECK_OK(velri_graph_add_op(g, &op0, mock_dev, nullptr), "op0");
        CHECK_OK(velri_graph_add_op(g, &op1, mock_dev, nullptr), "op1");
        velri_plan *plan = nullptr;
        CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prep dedup");
        velri_memory_report rep;
        CHECK_OK(velri_plan_memory_report(plan, &rep), "rep dedup");
        /* 1 transfer for s_x (cpu->mock, reused by op0 and op1!) + 1 writeback for s_y (mock->cpu). */
        CHECK(rep.num_ram_transfers == 2U, "read-only cross-device storage transferred once (B8)");
        velri_slot *slot = nullptr;
        CHECK_OK(velri_slot_create(plan, &slot), "slot");
        float hx[32], hy[32];
        for (size_t i = 0; i < n; ++i) { hx[i] = (float)(i + 1U); hy[i] = 0.0f; }
        CHECK_OK(velri_slot_bind_host(slot, s_x, hx, sizeof(hx)), "x");
        CHECK_OK(velri_slot_bind_host(slot, s_y, hy, sizeof(hy)), "y");
        CHECK_OK(velri_slot_run(slot, nullptr, nullptr), "run dedup");
        for (size_t i = 0; i < n; ++i) {
            CHECK(fabsf(hy[i] - 5.0f * hx[i]) < 1e-5f, "dedup result = 2x + 3x");
        }
        velri_slot_destroy(slot);
        velri_plan_destroy(plan);
        velri_graph_destroy(g);
    }

    /* 4b. A1 Partial-view write & in-place RMW across devices:
     *   - s_state lives on cpu_dev (size 32 floats).
     *   - Step 0 (cpu): fills full s_state[0..31] = x[0..31].
     *   - Step 1 (mock_async): updates ONLY upper half s_state[16..31] = 10 * s_state[16..31].
     *   - Step 2 (cpu): reads FULL s_state[0..31] into y[0..31].
     *   Lower half [0..15] MUST remain x[0..15] (untouched bytes preserved across RAM transfer!). */
    {
        const size_t n = 32U, half = 16U;
        velri_graph *g = nullptr;
        CHECK_OK(velri_graph_create(&g), "a1 g");
        velri_storage_id s_x = add_f32_storage(g, VELRI_STORAGE_INPUT, n, cpu_dev);
        velri_storage_id s_st = add_f32_storage(g, VELRI_STORAGE_STATE, n, cpu_dev);
        velri_storage_id s_y = add_f32_storage(g, VELRI_STORAGE_OUTPUT, n, cpu_dev);

        velri_view_id vx = add_1d_view(g, s_x, n);
        velri_view_id vst_all = add_1d_view(g, s_st, n);
        velri_view_id vst_hi = add_1d_view_off(g, s_st, half, half * sizeof(float));
        velri_view_id vy = add_1d_view(g, s_y, n);

        velri_op_desc op0 = {
            .kind = VELRI_OP_VEC_AXPBY, .num_inputs = 1U, .num_outputs = 1U,
            .inputs = {vx}, .outputs = {vst_all}, .attr.axpby = {.alpha = 1.0f, .beta = 0.0f},
        };
        velri_op_desc op1 = {
            .kind = VELRI_OP_VEC_AXPBY, .num_inputs = 1U, .num_outputs = 1U,
            .inputs = {vst_hi}, .outputs = {vst_hi}, .attr.axpby = {.alpha = 10.0f, .beta = 0.0f},
        };
        velri_op_desc op2 = {
            .kind = VELRI_OP_VEC_AXPBY, .num_inputs = 1U, .num_outputs = 1U,
            .inputs = {vst_all}, .outputs = {vy}, .attr.axpby = {.alpha = 1.0f, .beta = 0.0f},
        };
        CHECK_OK(velri_graph_add_op(g, &op0, cpu_dev, nullptr), "cpu init");
        CHECK_OK(velri_graph_add_op(g, &op1, mock_dev, nullptr), "mock partial RMW");
        CHECK_OK(velri_graph_add_op(g, &op2, cpu_dev, nullptr), "cpu readback");

        velri_plan *plan = nullptr;
        velri_slot *slot = nullptr;
        CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prep a1");
        CHECK_OK(velri_slot_create(plan, &slot), "slot a1");
        float hx[32], hst[32] = {0}, hy[32] = {0};
        for (size_t i = 0; i < n; ++i) hx[i] = (float)(i + 1U);
        CHECK_OK(velri_slot_bind_host(slot, s_x, hx, sizeof(hx)), "x");
        CHECK_OK(velri_slot_bind_host(slot, s_st, hst, sizeof(hst)), "st");
        CHECK_OK(velri_slot_bind_host(slot, s_y, hy, sizeof(hy)), "y");
        CHECK_OK(velri_slot_run(slot, nullptr, nullptr), "run a1");
        for (size_t i = 0; i < half; ++i) {
            CHECK(fabsf(hy[i] - hx[i]) < 1e-5f, "A1 lower half preserved");
            CHECK(fabsf(hy[half + i] - 10.0f * hx[half + i]) < 1e-5f, "A1 upper half updated");
        }
        velri_slot_destroy(slot);
        velri_plan_destroy(plan);
        velri_graph_destroy(g);
    }
    printf("  [PASS] Cross-device RMW + partial-view coherence (A1) & read-only dedup (B8).\n");
}

/* ---- Test 5: Shared WEIGHT uploads once (B1) & resident STATE across runs (B3) ---- */
static void test_shared_weights_and_resident_state(velri_context *ctx, velri_device_id mock_dev) {
    velri_mock_async_reset_ctrl();
    velri_mock_async_ctrl *ctrl = velri_mock_async_control();
    const size_t n = 16U;

    velri_graph *g = nullptr;
    CHECK_OK(velri_graph_create(&g), "b1b3 g");
    velri_storage_id s_w = add_f32_storage(g, VELRI_STORAGE_WEIGHT, n, mock_dev);
    velri_storage_id s_st = add_f32_storage(g, VELRI_STORAGE_STATE, n, mock_dev);
    velri_view_id vw = add_1d_view(g, s_w, n);
    velri_view_id vst = add_1d_view(g, s_st, n);

    /* Each step: state = state + w (purely resident on mock_async, NO output download!) */
    velri_op_desc op = {
        .kind = VELRI_OP_VEC_AXPBY, .num_inputs = 2U, .num_outputs = 1U,
        .inputs = {vst, vw}, .outputs = {vst}, .attr.axpby = {.alpha = 1.0f, .beta = 1.0f},
    };
    CHECK_OK(velri_graph_add_op(g, &op, mock_dev, nullptr), "add accum");

    velri_plan *plan = nullptr;
    CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prep b1b3");

    float hw[16], hs0[16] = {0}, hs1[16];
    for (size_t i = 0; i < n; ++i) {
        hw[i] = 2.0f;
        hs1[i] = 100.0f;
    }
    CHECK_OK(velri_plan_bind_weight_host(plan, s_w, hw, sizeof(hw)), "bind shared w");
    CHECK(atomic_load(&ctrl->total_uploads) == 1U, "B1 weight uploaded exactly once on plan");

    velri_slot *slot0 = nullptr, *slot1 = nullptr;
    CHECK_OK(velri_slot_create(plan, &slot0), "slot0");
    CHECK_OK(velri_slot_create(plan, &slot1), "slot1");
    CHECK_OK(velri_slot_bind_host(slot0, s_st, hs0, sizeof(hs0)), "bind st0");
    CHECK_OK(velri_slot_bind_host(slot1, s_st, hs1, sizeof(hs1)), "bind st1");

    /* Run 3 steps on slot0 and 2 steps on slot1. */
    for (int i = 0; i < 3; ++i) CHECK_OK(velri_slot_run(slot0, nullptr, nullptr), "run s0");
    for (int i = 0; i < 2; ++i) CHECK_OK(velri_slot_run(slot1, nullptr, nullptr), "run s1");

    /* Uploads = 1 (plan weight) + 1 (slot0 initial state) + 1 (slot1 initial state) = 3.
     * Downloads = 0 (STATE stays resident on device across steps!). */
    CHECK(atomic_load(&ctrl->total_uploads) == 3U, "B1/B3 no redundant weight or state uploads");
    CHECK(atomic_load(&ctrl->total_downloads) == 0U, "B3 zero per-step STATE downloads");

    /* Explicit host sync retrieves exact accumulated state for each isolated slot. */
    CHECK_OK(velri_slot_sync_state_to_host(slot0, s_st), "sync s0");
    CHECK_OK(velri_slot_sync_state_to_host(slot1, s_st), "sync s1");
    CHECK(atomic_load(&ctrl->total_downloads) == 2U, "explicit sync performed 2 downloads");
    for (size_t i = 0; i < n; ++i) {
        CHECK(fabsf(hs0[i] - 6.0f) < 1e-5f, "slot0 state = 0 + 3*2");
        CHECK(fabsf(hs1[i] - 104.0f) < 1e-5f, "slot1 state = 100 + 2*2");
    }

    velri_slot_destroy(slot0);
    velri_slot_destroy(slot1);
    velri_plan_destroy(plan);
    velri_graph_destroy(g);
    printf("  [PASS] Shared plan WEIGHT (B1) & resident STATE with explicit sync (B3).\n");
}

/* ---- Test 6: Refcounted ctx/plan lifetimes (B6) & graph growth > 32 storages (A8) ---- */
static void test_refcounted_lifetimes_and_graph_growth(void) {
    velri_context *ctx = nullptr;
    CHECK_OK(velri_context_create(nullptr, &ctx), "temp ctx");
    velri_device_id cpu_dev = 0U;
    CHECK_OK(velri_context_find_device(ctx, "cpu", 0U, &cpu_dev), "find cpu");

    velri_graph *g = nullptr;
    CHECK_OK(velri_graph_create(&g), "growth g");

    /* Add 40 storages (crosses 16 and 32 realloc thresholds) and chain 38 ops (A8). */
    const size_t n = 16U;
    velri_storage_id s_in = add_f32_storage(g, VELRI_STORAGE_INPUT, n, cpu_dev);
    velri_view_id prev = add_1d_view(g, s_in, n);
    for (int i = 0; i < 38; ++i) {
        velri_storage_id s_tmp = add_f32_storage(g, VELRI_STORAGE_SCRATCH, n, cpu_dev);
        velri_view_id v_tmp = add_1d_view(g, s_tmp, n);
        velri_op_desc op = {
            .kind = VELRI_OP_VEC_AXPBY, .num_inputs = 1U, .num_outputs = 1U,
            .inputs = {prev}, .outputs = {v_tmp}, .attr.axpby = {.alpha = 1.0f, .beta = 0.0f},
        };
        CHECK_OK(velri_graph_add_op(g, &op, cpu_dev, nullptr), "chain op");
        prev = v_tmp;
    }
    velri_storage_id s_out = add_f32_storage(g, VELRI_STORAGE_OUTPUT, n, cpu_dev);
    velri_view_id v_out = add_1d_view(g, s_out, n);
    velri_op_desc final_op = {
        .kind = VELRI_OP_VEC_AXPBY, .num_inputs = 1U, .num_outputs = 1U,
        .inputs = {prev}, .outputs = {v_out}, .attr.axpby = {.alpha = 2.0f, .beta = 0.0f},
    };
    CHECK_OK(velri_graph_add_op(g, &final_op, cpu_dev, nullptr), "final op");

    velri_plan *plan = nullptr;
    velri_slot *slot = nullptr;
    CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prep growth");
    CHECK_OK(velri_slot_create(plan, &slot), "slot growth");
    velri_graph_destroy(g);

    /* Destroy ctx and plan BEFORE using and destroying slot (B6 refcount safety!). */
    velri_plan_destroy(plan);
    velri_context_destroy(ctx);

    float hx[16], hy[16] = {0};
    for (size_t i = 0; i < n; ++i) hx[i] = (float)(i + 1U);
    CHECK_OK(velri_slot_bind_host(slot, s_in, hx, sizeof(hx)), "bind x");
    CHECK_OK(velri_slot_bind_host(slot, s_out, hy, sizeof(hy)), "bind y");
    CHECK_OK(velri_slot_run(slot, nullptr, nullptr), "run outlived slot");
    for (size_t i = 0; i < n; ++i) CHECK(fabsf(hy[i] - 2.0f * hx[i]) < 1e-5f, "outlived result");
    velri_slot_destroy(slot);
    printf("  [PASS] Refcounted ctx/plan lifetimes (B6) & graph realloc growth (A8).\n");
}

/* ---- Test 7: Errors (incl. RoPE runtime pos_offset [A4]), draining, cancellation ---- */
typedef struct {
    velri_cancel_token *tok;
    uint32_t delay_ms;
} delayed_cancel_args;

static void delayed_cancel_thread(void *arg) {
    delayed_cancel_args *a = (delayed_cancel_args *)arg;
    velri_os_sleep_ms(a->delay_ms);
    velri_cancel_request(a->tok);
}

static void test_errors_and_cancellation(velri_context *ctx, velri_device_id cpu_dev,
                                         velri_device_id mock_dev) {
    /* A4: Out-of-bounds runtime RoPE pos_offset_override rejected pre-accept (slot stays clean). */
    {
        velri_graph *rg = nullptr;
        CHECK_OK(velri_graph_create(&rg), "rope g");
        velri_storage_id sx = add_f32_storage(rg, VELRI_STORAGE_INPUT, 2U * 2U * 8U, cpu_dev);
        velri_storage_id sc = add_f32_storage(rg, VELRI_STORAGE_WEIGHT, 4U * 4U, cpu_dev);
        velri_storage_id ss = add_f32_storage(rg, VELRI_STORAGE_WEIGHT, 4U * 4U, cpu_dev);
        velri_storage_id sy = add_f32_storage(rg, VELRI_STORAGE_OUTPUT, 2U * 2U * 8U, cpu_dev);
        size_t d3[3] = {2U, 2U, 8U};
        velri_tensor_view vx_t, vy_t;
        CHECK_OK(velri_tensor_view_contiguous(sx, VELRI_DTYPE_F32, 3U, d3, 0U, &vx_t), "vx");
        CHECK_OK(velri_tensor_view_contiguous(sy, VELRI_DTYPE_F32, 3U, d3, 0U, &vy_t), "vy");
        velri_view_id vx = 0U, vy = 0U;
        CHECK_OK(velri_graph_add_view(rg, &vx_t, &vx), "add vx");
        CHECK_OK(velri_graph_add_view(rg, &vy_t, &vy), "add vy");
        velri_op_desc rop = {
            .kind = VELRI_OP_ROPE, .num_inputs = 3U, .num_outputs = 1U,
            .inputs = {vx, add_1d_view(rg, sc, 16U), add_1d_view(rg, ss, 16U)},
            .outputs = {vy},
            .attr.rope = {.rotary_dim = 8U, .pos_offset = 0U, .mode = VELRI_ROPE_HALF},
        };
        CHECK_OK(velri_graph_add_op(rg, &rop, cpu_dev, nullptr), "add rop");
        velri_plan *rplan = nullptr;
        velri_slot *rslot = nullptr;
        CHECK_OK(velri_plan_prepare(ctx, rg, nullptr, &rplan), "prep rop");
        float buf[32] = {0};
        CHECK_OK(velri_plan_bind_weight_host(rplan, sc, buf, 16U * sizeof(float)), "sc");
        CHECK_OK(velri_plan_bind_weight_host(rplan, ss, buf, 16U * sizeof(float)), "ss");
        CHECK_OK(velri_slot_create(rplan, &rslot), "rslot");
        CHECK_OK(velri_slot_bind_host(rslot, sx, buf, 32U * sizeof(float)), "sx");
        CHECK_OK(velri_slot_bind_host(rslot, sy, buf, 32U * sizeof(float)), "sy");
        /* Table only holds 4 positions; pos_offset=3 + n_tokens=2 => needs 5 -> INVALID! */
        velri_step_params bad_p = {.pos_offset_override = 3U, .has_pos_offset = true};
        CHECK(velri_slot_run(rslot, &bad_p, nullptr) == VELRI_ERR_INVALID,
              "A4 OOB pos_offset_override rejected");
        CHECK(!velri_slot_needs_reset(rslot), "A4 pre-accept rejection leaves slot clean");
        velri_slot_destroy(rslot);
        velri_plan_destroy(rplan);
        velri_graph_destroy(rg);
    }

    const size_t n = 64U;
    velri_graph *g = nullptr;
    CHECK_OK(velri_graph_create(&g), "err g");
    velri_storage_id s_a = add_f32_storage(g, VELRI_STORAGE_INPUT, n, mock_dev);
    velri_storage_id s_mid = add_f32_storage(g, VELRI_STORAGE_SCRATCH, n, mock_dev);
    velri_storage_id s_y = add_f32_storage(g, VELRI_STORAGE_OUTPUT, n, mock_dev);
    velri_view_id va = add_1d_view(g, s_a, n);
    velri_view_id vm = add_1d_view(g, s_mid, n);
    velri_view_id vy = add_1d_view(g, s_y, n);

    velri_op_desc op0 = {
        .kind = VELRI_OP_VEC_AXPBY, .num_inputs = 1U, .num_outputs = 1U,
        .inputs = {va}, .outputs = {vm}, .attr.axpby = {.alpha = 2.0f, .beta = 0.0f},
    };
    velri_op_desc op1 = {
        .kind = VELRI_OP_VEC_AXPBY, .num_inputs = 1U, .num_outputs = 1U,
        .inputs = {vm}, .outputs = {vy}, .attr.axpby = {.alpha = 3.0f, .beta = 0.0f},
    };
    CHECK_OK(velri_graph_add_op(g, &op0, mock_dev, nullptr), "op0");
    CHECK_OK(velri_graph_add_op(g, &op1, mock_dev, nullptr), "op1");

    velri_plan *plan = nullptr;
    velri_slot *slot = nullptr;
    CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prep err");
    CHECK_OK(velri_slot_create(plan, &slot), "slot err");

    float ha[64], hy[64];
    for (size_t i = 0; i < n; ++i) { ha[i] = (float)(i + 1U); hy[i] = 0.0f; }
    CHECK_OK(velri_slot_bind_host(slot, s_a, ha, sizeof(ha)), "a");
    CHECK_OK(velri_slot_bind_host(slot, s_y, hy, sizeof(hy)), "y");

    velri_mock_async_ctrl *ctrl = velri_mock_async_control();

    /* Pre-accept rejection */
    velri_mock_async_reset_ctrl();
    atomic_store(&ctrl->fail_pre_accept, true);
    CHECK(velri_slot_run(slot, nullptr, nullptr) == VELRI_ERR_EXEC, "pre-accept EXEC");
    CHECK(atomic_load(&ctrl->total_submitted) == 0U && !velri_slot_needs_reset(slot), "clean");

    /* Post-accept failure */
    velri_mock_async_reset_ctrl();
    atomic_store(&ctrl->delay_ms, 15U);
    atomic_store(&ctrl->fail_post_accept, true);
    CHECK(velri_slot_run(slot, nullptr, nullptr) == VELRI_ERR_EXEC, "post-accept EXEC");
    CHECK(atomic_load(&ctrl->in_flight) == 0 && atomic_load(&ctrl->total_completed) == 1U, "drained");
    CHECK(velri_slot_needs_reset(slot) && velri_slot_run(slot, nullptr, nullptr) == VELRI_ERR_BUSY,
          "dirty refuses run");
    CHECK_OK(velri_slot_reset(slot), "reset");

    /* Partial-accept submit error */
    velri_mock_async_reset_ctrl();
    atomic_store(&ctrl->delay_ms, 15U);
    atomic_store(&ctrl->fail_partial_submit, true);
    CHECK(velri_slot_run(slot, nullptr, nullptr) == VELRI_ERR_EXEC, "partial EXEC");
    CHECK(atomic_load(&ctrl->in_flight) == 0 && atomic_load(&ctrl->total_completed) == 1U, "drained");
    CHECK_OK(velri_slot_reset(slot), "reset 2");

    /* Pre-cancel */
    velri_mock_async_reset_ctrl();
    velri_cancel_token tok;
    velri_cancel_init(&tok);
    velri_cancel_request(&tok);
    CHECK(velri_slot_run(slot, nullptr, &tok) == VELRI_ERR_CANCELLED, "pre-cancel");
    CHECK(atomic_load(&ctrl->total_submitted) == 0U && !velri_slot_needs_reset(slot), "clean");

    /* Mid-flight cancel */
    velri_mock_async_reset_ctrl();
    atomic_store(&ctrl->delay_ms, 30U);
    velri_cancel_init(&tok);
    delayed_cancel_args ca = {.tok = &tok, .delay_ms = 10U};
    velri_os_thread *ct = nullptr;
    CHECK(velri_os_thread_spawn(&ct, delayed_cancel_thread, &ca), "spawn cancel");
    CHECK(velri_slot_run(slot, nullptr, &tok) == VELRI_ERR_CANCELLED, "mid-flight cancel");
    velri_os_thread_join(ct);
    CHECK(atomic_load(&ctrl->total_submitted) == 1U && atomic_load(&ctrl->in_flight) == 0 &&
              atomic_load(&ctrl->total_completed) == 1U && velri_slot_needs_reset(slot),
          "drained op0, skipped op1");
    CHECK_OK(velri_slot_reset(slot), "reset 3");

    /* Clean run after reset */
    velri_mock_async_reset_ctrl();
    velri_cancel_init(&tok);
    CHECK_OK(velri_slot_run(slot, nullptr, &tok), "clean run");
    for (size_t i = 0; i < n; ++i) CHECK(fabsf(hy[i] - 6.0f * ha[i]) < 1e-5f, "result");

    velri_slot_destroy(slot);
    velri_plan_destroy(plan);
    velri_graph_destroy(g);
    printf("  [PASS] RoPE override bounds (A4), pre/post/partial errors, and cancellation.\n");
}

#ifdef VELRI_CORE_HAS_VULKAN
/* ---- Test 8: Hybrid CPU <-> Vulkan graph + multi-op Vulkan command batching ---- */
static void test_hybrid_cpu_vulkan(velri_context *ctx, velri_device_id cpu_dev) {
    velri_device_id vk_dev = 0U;
    if (velri_context_find_device(ctx, "vulkan", 0U, &vk_dev) != VELRI_OK) {
        printf("  [SKIP] Hybrid CPU <-> Vulkan test (no Vulkan 1.3 GPU available).\n");
        return;
    }

    const size_t rows = 4U, k = 64U, m = 32U;
    velri_graph *g = nullptr;
    CHECK_OK(velri_graph_create(&g), "vk hybrid g");
    velri_storage_id s_x = add_f32_storage(g, VELRI_STORAGE_INPUT, rows * k, cpu_dev);
    velri_storage_id s_w = add_f32_storage(g, VELRI_STORAGE_WEIGHT, m * k, vk_dev);
    velri_storage_id s_n = add_f32_storage(g, VELRI_STORAGE_SCRATCH, rows * k, cpu_dev);
    velri_storage_id s_l = add_f32_storage(g, VELRI_STORAGE_SCRATCH, rows * m, vk_dev);
    velri_storage_id s_a = add_f32_storage(g, VELRI_STORAGE_SCRATCH, rows * m, vk_dev);
    velri_storage_id s_y = add_f32_storage(g, VELRI_STORAGE_OUTPUT, rows * m, cpu_dev);

    velri_view_id vx = add_2d_view(g, s_x, rows, k);
    velri_view_id vw = add_2d_view(g, s_w, m, k);
    velri_view_id vn = add_2d_view(g, s_n, rows, k);
    velri_view_id vl = add_2d_view(g, s_l, rows, m);
    velri_view_id va = add_2d_view(g, s_a, rows, m);
    velri_view_id vy = add_2d_view(g, s_y, rows, m);

    /* CPU RMSNorm -> [Vulkan Linear + Vulkan SiLU batched in 1 cmdbuf!] -> CPU Softmax */
    velri_op_desc op0 = {
        .kind = VELRI_OP_RMSNORM, .num_inputs = 1U, .num_outputs = 1U,
        .inputs = {vx}, .outputs = {vn}, .attr.rmsnorm = {.eps = 1e-5f},
    };
    velri_op_desc op1 = {
        .kind = VELRI_OP_LINEAR, .num_inputs = 2U, .num_outputs = 1U,
        .inputs = {vn, vw}, .outputs = {vl}, .attr.linear.desc = velri_desc_f32(),
    };
    velri_op_desc op2 = {
        .kind = VELRI_OP_ACTIVATION, .num_inputs = 1U, .num_outputs = 1U,
        .inputs = {vl}, .outputs = {va}, .attr.activation = {.kind = VELRI_ACT_SILU},
    };
    velri_op_desc op3 = {
        .kind = VELRI_OP_SOFTMAX, .num_inputs = 1U, .num_outputs = 1U,
        .inputs = {va}, .outputs = {vy},
    };
    CHECK_OK(velri_graph_add_op(g, &op0, cpu_dev, nullptr), "cpu rmsnorm");
    CHECK_OK(velri_graph_add_op(g, &op1, vk_dev, nullptr), "vk linear");
    CHECK_OK(velri_graph_add_op(g, &op2, vk_dev, nullptr), "vk silu");
    CHECK_OK(velri_graph_add_op(g, &op3, cpu_dev, nullptr), "cpu softmax");

    velri_plan *plan = nullptr;
    CHECK_OK(velri_plan_prepare(ctx, g, nullptr, &plan), "prepare cpu+vk");
    velri_memory_report rep;
    CHECK_OK(velri_plan_memory_report(plan, &rep), "report cpu+vk");
    CHECK(rep.num_ram_transfers == 2U, "expected 2 RAM transfers (cpu->vk and vk->cpu)");

    float *hx = (float *)calloc(rows * k, sizeof(float));
    float *hw = (float *)calloc(m * k, sizeof(float));
    float *hy = (float *)calloc(rows * m, sizeof(float));
    float *t0 = (float *)calloc(rows * k, sizeof(float));
    float *t1 = (float *)calloc(rows * m, sizeof(float));
    float *t2 = (float *)calloc(rows * m, sizeof(float));
    float *href = (float *)calloc(rows * m, sizeof(float));

    uint64_t rng = 0xDEADBEEF1234ULL;
    velri_test_fill_dtype(VELRI_DTYPE_F32, hx, rows * k, &rng, 1.5f);
    velri_test_fill_dtype(VELRI_DTYPE_F32, hw, m * k, &rng, 0.3f);
    velri_linear_desc ld = velri_desc_f32();
    ref_rmsnorm(VELRI_DTYPE_F32, VELRI_DTYPE_F32, rows, k, 1e-5f, false, hx, k, nullptr, t0, k);
    ref_linear_run(&ld, rows, m, k, t0, k, nullptr, hw, k, nullptr, nullptr, t1, m);
    ref_activation_f32(VELRI_ACT_SILU, rows * m, t1, t2);
    ref_softmax_f32(rows, m, t2, m, href, m);

    CHECK_OK(velri_plan_bind_weight_host(plan, s_w, hw, m * k * sizeof(float)), "bind hw");
    velri_slot *slot = nullptr;
    CHECK_OK(velri_slot_create(plan, &slot), "slot");
    CHECK_OK(velri_slot_bind_host(slot, s_x, hx, rows * k * sizeof(float)), "hx");
    CHECK_OK(velri_slot_bind_host(slot, s_y, hy, rows * m * sizeof(float)), "hy");
    CHECK_OK(velri_slot_run(slot, nullptr, nullptr), "run cpu+vk");
    CHECK(velri_slot_last_run_allocs(slot) == 0U, "0 hot-path allocs on Vulkan");

    double err = velri_test_max_abs_diff_2d(VELRI_DTYPE_F32, href, hy, rows, m, m);
    CHECK(err < 1e-4, "hybrid CPU <-> batched Vulkan matches reference");

    velri_slot_destroy(slot);
    velri_plan_destroy(plan);
    velri_graph_destroy(g);
    free(hx); free(hw); free(hy); free(t0); free(t1); free(t2); free(href);
    printf("  [PASS] Hybrid CPU <-> batched Vulkan graph (max_err=%.2e).\n", err);
}
#endif

/* ---- Test 9: Dual-Plan Shared WEIGHT Slabs & Shared STATE Arenas (0 Duplication) ---- */
static void test_shared_weights_and_shared_state_dual_plan(velri_context *ctx,
                                                           velri_device_id cpu_dev,
                                                           velri_device_id mock_dev) {
    const size_t n = 32U;
    velri_graph *g_dec = nullptr, *g_pre = nullptr, *g_bad = nullptr;
    CHECK_OK(velri_graph_create(&g_dec), "dual g_dec");
    CHECK_OK(velri_graph_create(&g_pre), "dual g_pre");
    CHECK_OK(velri_graph_create(&g_bad), "dual g_bad");

    /* Both plans store WEIGHT and STATE on non-host `mock_dev`, but execute ops on different
     * compute devices (`decode` on `cpu_dev`, `prefill` on `mock_dev`), exercising BOTH
     * shared non-host weight slabs and shared resident state arenas across heterogeneous plans! */
    velri_storage_id d_x = add_f32_storage(g_dec, VELRI_STORAGE_INPUT, n, cpu_dev);
    velri_storage_id d_w = add_f32_storage(g_dec, VELRI_STORAGE_WEIGHT, n, mock_dev);
    velri_storage_id d_s = add_f32_storage(g_dec, VELRI_STORAGE_STATE, n, mock_dev);
    velri_storage_id d_tmp = add_f32_storage(g_dec, VELRI_STORAGE_SCRATCH, n, cpu_dev);
    velri_storage_id d_y = add_f32_storage(g_dec, VELRI_STORAGE_OUTPUT, n, cpu_dev);
    velri_view_id vd_x = add_1d_view(g_dec, d_x, n);
    velri_view_id vd_w = add_1d_view(g_dec, d_w, n);
    velri_view_id vd_s = add_1d_view(g_dec, d_s, n);
    velri_view_id vd_tmp = add_1d_view(g_dec, d_tmp, n);
    velri_view_id vd_y = add_1d_view(g_dec, d_y, n);
    /* Decode: tmp = x * w; s = s + tmp; y = s */
    velri_op_desc d_op0 = {
        .kind = VELRI_OP_VEC_MUL, .num_inputs = 2U, .num_outputs = 1U,
        .inputs = {vd_x, vd_w}, .outputs = {vd_tmp},
    };
    velri_op_desc d_op1 = {
        .kind = VELRI_OP_VEC_AXPBY, .num_inputs = 2U, .num_outputs = 1U,
        .inputs = {vd_s, vd_tmp}, .outputs = {vd_s}, .attr.axpby = {.alpha = 1.0f, .beta = 1.0f},
    };
    velri_op_desc d_op2 = {
        .kind = VELRI_OP_VEC_AXPBY, .num_inputs = 1U, .num_outputs = 1U,
        .inputs = {vd_s}, .outputs = {vd_y}, .attr.axpby = {.alpha = 1.0f, .beta = 0.0f},
    };
    CHECK_OK(velri_graph_add_op(g_dec, &d_op0, cpu_dev, nullptr), "d_op0");
    CHECK_OK(velri_graph_add_op(g_dec, &d_op1, cpu_dev, nullptr), "d_op1");
    CHECK_OK(velri_graph_add_op(g_dec, &d_op2, cpu_dev, nullptr), "d_op2");

    /* Prefill has extra scratch storages (shifting raw storage IDs!) and runs on `mock_dev`. */
    velri_storage_id p_pad = add_f32_storage(g_pre, VELRI_STORAGE_SCRATCH, n * 2U, mock_dev);
    (void)p_pad;
    velri_storage_id p_x = add_f32_storage(g_pre, VELRI_STORAGE_INPUT, n, mock_dev);
    velri_storage_id p_w = add_f32_storage(g_pre, VELRI_STORAGE_WEIGHT, n, mock_dev);
    velri_storage_id p_s = add_f32_storage(g_pre, VELRI_STORAGE_STATE, n, mock_dev);
    velri_storage_id p_tmp = add_f32_storage(g_pre, VELRI_STORAGE_SCRATCH, n, mock_dev);
    velri_storage_id p_y = add_f32_storage(g_pre, VELRI_STORAGE_OUTPUT, n, mock_dev);
    velri_view_id vp_x = add_1d_view(g_pre, p_x, n);
    velri_view_id vp_w = add_1d_view(g_pre, p_w, n);
    velri_view_id vp_s = add_1d_view(g_pre, p_s, n);
    velri_view_id vp_tmp = add_1d_view(g_pre, p_tmp, n);
    velri_view_id vp_y = add_1d_view(g_pre, p_y, n);
    /* Prefill: tmp = 2*x + w; s = s + tmp; y = s */
    velri_op_desc p_op0 = {
        .kind = VELRI_OP_VEC_AXPBY, .num_inputs = 2U, .num_outputs = 1U,
        .inputs = {vp_x, vp_w}, .outputs = {vp_tmp}, .attr.axpby = {.alpha = 2.0f, .beta = 1.0f},
    };
    velri_op_desc p_op1 = {
        .kind = VELRI_OP_VEC_AXPBY, .num_inputs = 2U, .num_outputs = 1U,
        .inputs = {vp_s, vp_tmp}, .outputs = {vp_s}, .attr.axpby = {.alpha = 1.0f, .beta = 1.0f},
    };
    velri_op_desc p_op2 = {
        .kind = VELRI_OP_VEC_AXPBY, .num_inputs = 1U, .num_outputs = 1U,
        .inputs = {vp_s}, .outputs = {vp_y}, .attr.axpby = {.alpha = 1.0f, .beta = 0.0f},
    };
    CHECK_OK(velri_graph_add_op(g_pre, &p_op0, mock_dev, nullptr), "p_op0");
    CHECK_OK(velri_graph_add_op(g_pre, &p_op1, mock_dev, nullptr), "p_op1");
    CHECK_OK(velri_graph_add_op(g_pre, &p_op2, mock_dev, nullptr), "p_op2");

    /* Mismatched graph (different weight size) must fail `share_weights_from`. */
    velri_storage_id b_w = add_f32_storage(g_bad, VELRI_STORAGE_WEIGHT, n * 2U, mock_dev);
    velri_storage_id b_y = add_f32_storage(g_bad, VELRI_STORAGE_OUTPUT, n * 2U, mock_dev);
    velri_op_desc b_op = {
        .kind = VELRI_OP_VEC_AXPBY, .num_inputs = 1U, .num_outputs = 1U,
        .inputs = {add_1d_view(g_bad, b_w, n * 2U)}, .outputs = {add_1d_view(g_bad, b_y, n * 2U)},
        .attr.axpby = {.alpha = 1.0f, .beta = 0.0f},
    };
    CHECK_OK(velri_graph_add_op(g_bad, &b_op, mock_dev, nullptr), "b_op");

    velri_mock_async_reset_ctrl();
    velri_plan *dec_plan = nullptr, *pre_plan = nullptr, *bad_plan = nullptr;
    CHECK_OK(velri_plan_prepare(ctx, g_dec, nullptr, &dec_plan), "prep dec");
    uint64_t allocs_after_dec_plan = atomic_load(&velri_mock_async_control()->total_allocs);

    velri_plan_options share_opts = {.allow_cpu_fallback = false, .share_weights_from = dec_plan};
    CHECK_OK(velri_plan_prepare(ctx, g_pre, &share_opts, &pre_plan), "prep pre shared weights");
    uint64_t allocs_after_pre_plan = atomic_load(&velri_mock_async_control()->total_allocs);
    CHECK(allocs_after_pre_plan == allocs_after_dec_plan,
          "shared-weight prefill plan performs 0 duplicate weight slab allocs");

    CHECK(velri_plan_prepare(ctx, g_bad, &share_opts, &bad_plan) == VELRI_ERR_INVALID,
          "mismatched weight layout rejected");

    velri_memory_report dec_rep, pre_rep;
    CHECK_OK(velri_plan_memory_report(dec_plan, &dec_rep), "dec rep");
    CHECK_OK(velri_plan_memory_report(pre_plan, &pre_rep), "pre rep");
    CHECK(dec_rep.devices[mock_dev].weight_bytes == n * sizeof(float), "dec owns weights");
    CHECK(pre_rep.devices[mock_dev].weight_bytes == 0U, "pre reports 0 duplicate weight bytes");

    /* Bind weights ONCE on `dec_plan` -> visible in `pre_plan` immediately! */
    float hw[32], hx[32], hy_pre[32] = {0}, hy_dec[32] = {0};
    for (size_t i = 0; i < n; ++i) {
        hw[i] = 2.0f + (float)i * 0.25f;
        hx[i] = 1.0f + (float)i * 0.5f;
    }
    CHECK_OK(velri_plan_bind_weight_host(dec_plan, d_w, hw, sizeof(hw)), "bind weight once");

    velri_slot *dec_slot = nullptr, *pre_slot = nullptr;
    CHECK_OK(velri_slot_create(dec_plan, &dec_slot), "dec_slot");
    uint64_t allocs_after_dec_slot = atomic_load(&velri_mock_async_control()->total_allocs);
    CHECK_OK(velri_slot_create_shared_state(pre_plan, dec_slot, &pre_slot), "pre_slot shared state");
    uint64_t allocs_after_pre_slot = atomic_load(&velri_mock_async_control()->total_allocs);
    /* pre_slot allocates ONLY scratch + IO on mock_dev (2 arenas), NOT STATE (would be 3)! */
    CHECK(allocs_after_pre_slot - allocs_after_dec_slot == 2U,
          "shared-state prefill slot skips STATE allocation");

    /* Mismatched state layout (bad_plan without STATE) must fail shared-state slot creation. */
    CHECK_OK(velri_plan_prepare(ctx, g_bad, nullptr, &bad_plan), "prep bad standalone");
    velri_slot *bad_slot = nullptr;
    CHECK(velri_slot_create_shared_state(bad_plan, dec_slot, &bad_slot) == VELRI_ERR_INVALID,
          "mismatched STATE layout rejected");
    velri_plan_destroy(bad_plan);

    CHECK_OK(velri_slot_bind_host(pre_slot, p_x, hx, sizeof(hx)), "bind pre x");
    CHECK_OK(velri_slot_bind_host(pre_slot, p_y, hy_pre, sizeof(hy_pre)), "bind pre y");
    CHECK_OK(velri_slot_bind_host(dec_slot, d_x, hx, sizeof(hx)), "bind dec x");
    CHECK_OK(velri_slot_bind_host(dec_slot, d_y, hy_dec, sizeof(hy_dec)), "bind dec y");

    /* Run prefill -> mutates shared STATE: s = 2*x + w */
    CHECK_OK(velri_slot_run(pre_slot, nullptr, nullptr), "run pre_slot");
    CHECK(velri_slot_last_run_allocs(pre_slot) == 0U, "pre_slot 0 hot-path allocs");
    for (size_t i = 0; i < n; ++i) {
        float exp_pre = 2.0f * hx[i] + hw[i];
        CHECK(fabsf(hy_pre[i] - exp_pre) < 1e-5f, "pre_slot output exact");
    }

    /* Run decode -> reads shared STATE populated by prefill and adds x*w: s = (2*x + w) + x*w */
    CHECK_OK(velri_slot_run(dec_slot, nullptr, nullptr), "run dec_slot");
    CHECK(velri_slot_last_run_allocs(dec_slot) == 0U, "dec_slot 0 hot-path allocs");
    for (size_t i = 0; i < n; ++i) {
        float exp_dec = (2.0f * hx[i] + hw[i]) + hx[i] * hw[i];
        CHECK(fabsf(hy_dec[i] - exp_dec) < 1e-5f, "dec_slot saw shared STATE from pre_slot");
    }

    /* Arbitrary destruction order (owner destroyed BEFORE borrower!): refcounts keep memory live! */
    velri_slot_destroy(dec_slot);
    velri_plan_destroy(dec_plan);
    CHECK_OK(velri_slot_run(pre_slot, nullptr, nullptr), "pre_slot runs after owner destroyed");
    for (size_t i = 0; i < n; ++i) {
        float exp_after = 2.0f * (2.0f * hx[i] + hw[i]) + hx[i] * hw[i];
        CHECK(fabsf(hy_pre[i] - exp_after) < 1e-5f, "shared STATE & WEIGHT intact after owner drop");
    }

    /* Chained sharing (A -> B -> C), borrower bind_weight_host, shared-state needs_reset
     * propagation, dual host-bound STATE no-clobber, and alignment mismatch rejection. */
    {
        velri_plan *pA = nullptr, *pB = nullptr, *pC = nullptr;
        CHECK_OK(velri_plan_prepare(ctx, g_dec, nullptr, &pA), "chain pA");
        velri_plan_options optB = {.share_weights_from = pA};
        CHECK_OK(velri_plan_prepare(ctx, g_pre, &optB, &pB), "chain pB");
        velri_plan_options optC = {.share_weights_from = pB};
        CHECK_OK(velri_plan_prepare(ctx, g_pre, &optC, &pC), "chain pC");

        /* Bind weight directly via borrower `pC` -> forwards cleanly to root owner `pA`! */
        CHECK_OK(velri_plan_bind_weight_host(pC, p_w, hw, sizeof(hw)), "bind via borrower pC");

        velri_slot *sA = nullptr, *sB = nullptr, *sC = nullptr;
        CHECK_OK(velri_slot_create(pA, &sA), "chain sA");
        CHECK_OK(velri_slot_create_shared_state(pB, sA, &sB), "chain sB");
        CHECK_OK(velri_slot_create_shared_state(pC, sB, &sC), "chain sC");

        float stale_state[32];
        for (size_t i = 0; i < n; ++i) stale_state[i] = 999.0f;
        CHECK_OK(velri_slot_bind_host(sB, p_x, hx, sizeof(hx)), "sB x");
        CHECK_OK(velri_slot_bind_host(sB, p_y, hy_pre, sizeof(hy_pre)), "sB y");
        CHECK_OK(velri_slot_bind_host(sC, p_x, hx, sizeof(hx)), "sC x");
        CHECK_OK(velri_slot_bind_host(sC, p_y, hy_dec, sizeof(hy_dec)), "sC y");
        CHECK_OK(velri_slot_bind_host(sA, d_x, hx, sizeof(hx)), "sA x");
        CHECK_OK(velri_slot_bind_host(sA, d_y, hy_dec, sizeof(hy_dec)), "sA y");

        /* Run sB (populates shared STATE), then bind stale host buffer on sC & run sC:
         * sC MUST NOT clobber live shared STATE with `stale_state`! */
        CHECK_OK(velri_slot_run(sB, nullptr, nullptr), "sB run");
        CHECK_OK(velri_slot_bind_host(sC, p_s, stale_state, sizeof(stale_state)), "sC bind stale");
        CHECK_OK(velri_slot_run(sC, nullptr, nullptr), "sC run no-clobber");
        for (size_t i = 0; i < n; ++i) {
            float exp2 = 2.0f * (2.0f * hx[i] + hw[i]);
            CHECK(fabsf(hy_dec[i] - exp2) < 1e-5f, "shared STATE not clobbered by peer bind");
        }

        /* Post-accept failure on sB propagates `needs_reset` to sA and sC via state_group! */
        velri_mock_async_reset_ctrl();
        atomic_store(&velri_mock_async_control()->fail_post_accept, true);
        CHECK(velri_slot_run(sB, nullptr, nullptr) == VELRI_ERR_EXEC, "sB post-accept fail");
        CHECK(velri_slot_needs_reset(sB) && velri_slot_needs_reset(sA) && velri_slot_needs_reset(sC),
              "shared state_group propagates needs_reset to all peers");
        CHECK(velri_slot_run(sA, nullptr, nullptr) == VELRI_ERR_BUSY, "peer sA refuses run");

        /* Reset sA (zeroes STATE once) + reset sB & sC (skip redundant STATE zeroing). */
        velri_mock_async_reset_ctrl();
        CHECK_OK(velri_slot_reset(sA), "reset sA");
        CHECK_OK(velri_slot_reset(sB), "reset sB");
        CHECK_OK(velri_slot_reset(sC), "reset sC");
        CHECK(!velri_slot_needs_reset(sA) && !velri_slot_needs_reset(sB) && !velri_slot_needs_reset(sC),
              "all peers clean after reset");

        velri_slot_destroy(sA);
        velri_slot_destroy(sB);
        velri_slot_destroy(sC);
        velri_plan_destroy(pA);
        velri_plan_destroy(pB);
        velri_plan_destroy(pC);
    }

    /* Alignment mismatch rejection (owner align=64, borrower align=256) + host-memory weight sharing. */
    {
        velri_graph *g_hi = nullptr, *g_cpu1 = nullptr, *g_cpu2 = nullptr;
        CHECK_OK(velri_graph_create(&g_hi), "g_hi");
        velri_storage_desc sw_hi = {.kind = VELRI_STORAGE_WEIGHT, .dtype = VELRI_DTYPE_F32,
                                    .byte_size = n * sizeof(float), .align = 256U, .device_id = mock_dev};
        velri_storage_desc ss_hi = {.kind = VELRI_STORAGE_STATE, .dtype = VELRI_DTYPE_F32,
                                    .byte_size = n * sizeof(float), .align = 256U, .device_id = mock_dev};
        velri_storage_id hi_w = 0U, hi_s = 0U;
        CHECK_OK(velri_graph_add_storage(g_hi, &sw_hi, &hi_w), "hi_w");
        CHECK_OK(velri_graph_add_storage(g_hi, &ss_hi, &hi_s), "hi_s");
        velri_storage_id hi_y = add_f32_storage(g_hi, VELRI_STORAGE_OUTPUT, n, mock_dev);
        velri_op_desc hi_op = {
            .kind = VELRI_OP_VEC_AXPBY, .num_inputs = 2U, .num_outputs = 1U,
            .inputs = {add_1d_view(g_hi, hi_w, n), add_1d_view(g_hi, hi_s, n)},
            .outputs = {add_1d_view(g_hi, hi_y, n)},
            .attr.axpby = {.alpha = 1.0f, .beta = 1.0f},
        };
        CHECK_OK(velri_graph_add_op(g_hi, &hi_op, mock_dev, nullptr), "hi_op");

        velri_plan *p_lo = nullptr, *p_hi = nullptr;
        CHECK_OK(velri_plan_prepare(ctx, g_dec, nullptr, &p_lo), "p_lo");
        velri_plan_options hi_w_opts = {.share_weights_from = p_lo};
        CHECK(velri_plan_prepare(ctx, g_hi, &hi_w_opts, &p_hi) == VELRI_ERR_INVALID,
              "under-aligned owner weight slab rejected");
        CHECK_OK(velri_plan_prepare(ctx, g_hi, nullptr, &p_hi), "p_hi standalone");
        velri_slot *s_lo = nullptr, *s_hi = nullptr;
        CHECK_OK(velri_slot_create(p_lo, &s_lo), "s_lo");
        CHECK(velri_slot_create_shared_state(p_hi, s_lo, &s_hi) == VELRI_ERR_INVALID,
              "under-aligned owner STATE arena rejected");
        velri_slot_destroy(s_lo);
        velri_plan_destroy(p_hi);
        velri_plan_destroy(p_lo);
        velri_graph_destroy(g_hi);

        /* Host-memory (`cpu:0`) shared weights across 2 plans. */
        CHECK_OK(velri_graph_create(&g_cpu1), "g_cpu1");
        CHECK_OK(velri_graph_create(&g_cpu2), "g_cpu2");
        velri_storage_id c1_w = add_f32_storage(g_cpu1, VELRI_STORAGE_WEIGHT, n, cpu_dev);
        velri_storage_id c1_y = add_f32_storage(g_cpu1, VELRI_STORAGE_OUTPUT, n, cpu_dev);
        velri_op_desc c1_op = {
            .kind = VELRI_OP_VEC_AXPBY, .num_inputs = 1U, .num_outputs = 1U,
            .inputs = {add_1d_view(g_cpu1, c1_w, n)}, .outputs = {add_1d_view(g_cpu1, c1_y, n)},
            .attr.axpby = {.alpha = 3.0f, .beta = 0.0f},
        };
        CHECK_OK(velri_graph_add_op(g_cpu1, &c1_op, cpu_dev, nullptr), "c1_op");

        velri_storage_id c2_pad = add_f32_storage(g_cpu2, VELRI_STORAGE_SCRATCH, n, cpu_dev);
        (void)c2_pad;
        velri_storage_id c2_w = add_f32_storage(g_cpu2, VELRI_STORAGE_WEIGHT, n, cpu_dev);
        velri_storage_id c2_y = add_f32_storage(g_cpu2, VELRI_STORAGE_OUTPUT, n, cpu_dev);
        velri_op_desc c2_op = {
            .kind = VELRI_OP_VEC_AXPBY, .num_inputs = 1U, .num_outputs = 1U,
            .inputs = {add_1d_view(g_cpu2, c2_w, n)}, .outputs = {add_1d_view(g_cpu2, c2_y, n)},
            .attr.axpby = {.alpha = 4.0f, .beta = 0.0f},
        };
        CHECK_OK(velri_graph_add_op(g_cpu2, &c2_op, cpu_dev, nullptr), "c2_op");

        velri_plan *pc1 = nullptr, *pc2 = nullptr;
        CHECK_OK(velri_plan_prepare(ctx, g_cpu1, nullptr, &pc1), "pc1");
        velri_plan_options copt = {.share_weights_from = pc1};
        CHECK_OK(velri_plan_prepare(ctx, g_cpu2, &copt, &pc2), "pc2 shared host weight");
        CHECK_OK(velri_plan_bind_weight_host(pc1, c1_w, hw, sizeof(hw)), "bind host weight once");

        velri_slot *sc2 = nullptr;
        CHECK_OK(velri_slot_create(pc2, &sc2), "sc2");
        float hc_out[32] = {0};
        CHECK_OK(velri_slot_bind_host(sc2, c2_y, hc_out, sizeof(hc_out)), "sc2 y");
        CHECK_OK(velri_slot_run(sc2, nullptr, nullptr), "sc2 run");
        for (size_t i = 0; i < n; ++i) {
            CHECK(fabsf(hc_out[i] - 4.0f * hw[i]) < 1e-5f, "shared host weight exact");
        }
        velri_slot_destroy(sc2);
        velri_plan_destroy(pc2);
        velri_plan_destroy(pc1);
        velri_graph_destroy(g_cpu2);
        velri_graph_destroy(g_cpu1);
    }

    velri_slot_destroy(pre_slot);
    velri_plan_destroy(pre_plan);
    velri_graph_destroy(g_dec);
    velri_graph_destroy(g_pre);
    velri_graph_destroy(g_bad);
    printf("  [PASS] Dual-plan shared WEIGHT slabs & shared STATE arenas (0 duplication, refcounted, state_group).\n");
}

int main(void) {
    printf("=== Velri-lib Core & Backend Contract Suite ===\n");
    velri_context_config cfg = {
        .cpu_workers = 4U,
        .cpu_variant = VELRI_CPU_AUTO,
        .enable_vulkan = true,
    };
    velri_context *ctx = nullptr;
    CHECK_OK(velri_context_create(&cfg, &ctx), "context_create");
    CHECK_OK(velri_context_register_backend(ctx, velri_mock_async_backend_vtable()),
             "register mock_async");

    velri_device_id cpu_dev = 0U, mock_dev = 0U;
    CHECK_OK(velri_context_find_device(ctx, "cpu", 0U, &cpu_dev), "find cpu:0");
    CHECK_OK(velri_context_find_device(ctx, "mock_async", 0U, &mock_dev), "find mock_async:0");

    test_validation_and_guards(ctx, cpu_dev);
    test_standalone_mlp_two_concurrent_slots(ctx, cpu_dev, false);
    test_no_silent_cpu_fallback(ctx, mock_dev);
    test_cross_device_coherence_and_dedup(ctx, cpu_dev, mock_dev);
    test_shared_weights_and_resident_state(ctx, mock_dev);
    test_refcounted_lifetimes_and_graph_growth();
    test_errors_and_cancellation(ctx, cpu_dev, mock_dev);
    test_shared_weights_and_shared_state_dual_plan(ctx, cpu_dev, mock_dev);
#ifdef VELRI_CORE_HAS_VULKAN
    velri_device_id vk_dev;
    if (velri_context_find_device(ctx, "vulkan", 0U, &vk_dev) == VELRI_OK) {
        test_standalone_mlp_two_concurrent_slots(ctx, vk_dev, true);
    }
    test_hybrid_cpu_vulkan(ctx, cpu_dev);
#endif

    velri_context_destroy(ctx);
    printf("=== All Core Contract Tests PASSED ===\n");
    return 0;
}
