#include "bench_decode.h"
#include "os.h"
#include "pool.h"
#include "reference.h"
#include "velri_lib/velri_lib.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if VELRI_X86_VARIANTS
#include <immintrin.h>
#endif

static void check_ok(velri_status st, const char *ctx) {
    if (st != VELRI_OK) {
        fprintf(stderr, "ERROR (%s): velri_status=%d\n", ctx, (int)st);
        exit(1);
    }
}

static void *xalloc_raw(size_t bytes) {
    size_t sz = bytes ? ((bytes + 63U) & ~(size_t)63U) : 64U;
    void *p = velri_os_aligned_alloc(64U, sz);
    if (!p) {
        fprintf(stderr, "FATAL: aligned alloc failed\n");
        exit(1);
    }
    return p;
}

static void *xalloc(size_t bytes) {
    size_t sz = bytes ? ((bytes + 63U) & ~(size_t)63U) : 64U;
    void *p = xalloc_raw(sz);
    memset(p, 0, sz);
    return p;
}

static void *xcalloc(size_t count, size_t elem_size) {
    void *p = calloc(count ? count : 1U, elem_size ? elem_size : 1U);
    if (!p) {
        fprintf(stderr, "FATAL: calloc failed\n");
        exit(1);
    }
    return p;
}

static double now_sec(void) { return (double)velri_os_monotonic_ns() * 1e-9; }

static int cmp_dbl(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static double median_of(double *arr, size_t n) {
    qsort(arr, n, sizeof(double), cmp_dbl);
    return arr[n / 2U];
}

/* ---- Direct CPU Physical Silicon Ceiling Calibration ---- */
typedef struct {
    double bw_read_gbps;
    double bw_copy_gbps;
    double fma_f32_gflops;
    double fma_f64_gflops;
    double vnni_i8_gops;
} cpu_ceilings;

typedef struct {
    const float *src;
    float *dst;
    size_t n;
    volatile double sink[VELRI_POOL_MAX_WORKERS];
} cpu_bw_ctx;

static void bw_touch_worker(void *v, size_t wid, size_t nw) {
    cpu_bw_ctx *c = (cpu_bw_ctx *)v;
    size_t i0 = ((c->n / 16U) * wid / nw) * 16U;
    size_t i1 = ((c->n / 16U) * (wid + 1U) / nw) * 16U;
    float *p = c->dst;
    for (size_t i = i0; i < i1; ++i) p[i] = 1.0f;
}

static void bw_read_worker(void *v, size_t wid, size_t nw) {
    cpu_bw_ctx *c = (cpu_bw_ctx *)v;
    size_t i0 = ((c->n / 64U) * wid / nw) * 64U;
    size_t i1 = ((c->n / 64U) * (wid + 1U) / nw) * 64U;
    const float *p = c->src;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (size_t i = i0; i < i1; i += 16U) {
        s0 += p[i + 0U] + p[i + 4U] + p[i + 8U] + p[i + 12U];
        s1 += p[i + 1U] + p[i + 5U] + p[i + 9U] + p[i + 13U];
        s2 += p[i + 2U] + p[i + 6U] + p[i + 10U] + p[i + 14U];
        s3 += p[i + 3U] + p[i + 7U] + p[i + 11U] + p[i + 15U];
    }
    c->sink[wid] = (double)(s0 + s1 + s2 + s3);
}

#if VELRI_X86_VARIANTS && !defined(_MSC_VER)
__attribute__((target("avx512f"))) static void bw_read_avx512_worker(void *v, size_t wid,
                                                                     size_t nw) {
    cpu_bw_ctx *c = (cpu_bw_ctx *)v;
    size_t i0 = ((c->n / 64U) * wid / nw) * 64U;
    size_t i1 = ((c->n / 64U) * (wid + 1U) / nw) * 64U;
    const float *p = c->src;
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    for (size_t i = i0; i < i1; i += 64U) {
        s0 = _mm512_add_ps(s0, _mm512_load_ps(p + i + 0U));
        s1 = _mm512_add_ps(s1, _mm512_load_ps(p + i + 16U));
        s2 = _mm512_add_ps(s2, _mm512_load_ps(p + i + 32U));
        s3 = _mm512_add_ps(s3, _mm512_load_ps(p + i + 48U));
    }
    s0 = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    c->sink[wid] = (double)_mm512_reduce_add_ps(s0);
}
#endif

static void bw_copy_worker(void *v, size_t wid, size_t nw) {
    cpu_bw_ctx *c = (cpu_bw_ctx *)v;
    size_t i0 = ((c->n / 16U) * wid / nw) * 16U;
    size_t i1 = ((c->n / 16U) * (wid + 1U) / nw) * 16U;
    memcpy(c->dst + i0, c->src + i0, (i1 - i0) * sizeof(float));
}

#if VELRI_X86_VARIANTS && !defined(_MSC_VER)
typedef struct {
    size_t iters;
    volatile double sink[VELRI_POOL_MAX_WORKERS];
} cpu_compute_ctx;

__attribute__((target("avx512f,fma"))) static void ceil_fma_f32_worker(void *v, size_t wid,
                                                                       size_t nw) {
    (void)nw;
    cpu_compute_ctx *c = (cpu_compute_ctx *)v;
    __m512 a = _mm512_set1_ps(1.000001f + (float)wid * 1e-6f);
    __m512 b = _mm512_set1_ps(0.999999f);
    __m512 r[12];
    for (int j = 0; j < 12; ++j) r[j] = _mm512_set1_ps((float)(j + 1));
    for (size_t i = 0; i < c->iters; ++i) {
        for (int j = 0; j < 12; ++j) r[j] = _mm512_fmadd_ps(r[j], b, a);
    }
    __m512 s = r[0];
    for (int j = 1; j < 12; ++j) s = _mm512_add_ps(s, r[j]);
    c->sink[wid] = (double)_mm512_reduce_add_ps(s);
}

__attribute__((target("avx512f,fma"))) static void ceil_fma_f64_worker(void *v, size_t wid,
                                                                       size_t nw) {
    (void)nw;
    cpu_compute_ctx *c = (cpu_compute_ctx *)v;
    __m512d a = _mm512_set1_pd(1.000001 + (double)wid * 1e-6);
    __m512d b = _mm512_set1_pd(0.999999);
    __m512d r[12];
    for (int j = 0; j < 12; ++j) r[j] = _mm512_set1_pd((double)(j + 1));
    for (size_t i = 0; i < c->iters; ++i) {
        for (int j = 0; j < 12; ++j) r[j] = _mm512_fmadd_pd(r[j], b, a);
    }
    __m512d s = r[0];
    for (int j = 1; j < 12; ++j) s = _mm512_add_pd(s, r[j]);
    c->sink[wid] = _mm512_reduce_add_pd(s);
}

__attribute__((target("avx512f,avx512vnni"))) static void ceil_vnni_i8_worker(void *v, size_t wid,
                                                                              size_t nw) {
    (void)nw;
    cpu_compute_ctx *c = (cpu_compute_ctx *)v;
    __m512i a = _mm512_set1_epi8((char)(1 + (wid & 7U)));
    __m512i b = _mm512_set1_epi8((char)129);
    __m512i r[12];
    for (int j = 0; j < 12; ++j) r[j] = _mm512_set1_epi32(j + 1);
    for (size_t i = 0; i < c->iters; ++i) {
        for (int j = 0; j < 12; ++j) r[j] = _mm512_dpbusd_epi32(r[j], b, a);
    }
    __m512i s = r[0];
    for (int j = 1; j < 12; ++j) s = _mm512_add_epi32(s, r[j]);
    c->sink[wid] = (double)_mm512_reduce_add_epi32(s);
}

#endif

static cpu_ceilings measure_cpu_ceilings(size_t nw) {
    cpu_ceilings out = {0};
    velri_pool *pool = nullptr;
    check_ok(velri_pool_create(nw, &pool), "ceil pool");

    size_t n_rd = (512U * 1024U * 1024U) / sizeof(float);
    float *buf_a = (float *)xalloc_raw(n_rd * sizeof(float));
    float *buf_b = (float *)xalloc_raw((n_rd / 2U) * sizeof(float));

    cpu_bw_ctx tctx = {.dst = buf_a, .n = n_rd};
    velri_pool_parallel(pool, nw, bw_touch_worker, &tctx);
    tctx.dst = buf_b;
    tctx.n = n_rd / 2U;
    velri_pool_parallel(pool, nw, bw_touch_worker, &tctx);

    velri_parallel_fn rd_fn = bw_read_worker;
#if VELRI_X86_VARIANTS && !defined(_MSC_VER)
    if (velri_get_cpu_info()->has_avx512f) rd_fn = bw_read_avx512_worker;
#endif
    cpu_bw_ctx bctx = {.src = buf_a, .dst = buf_b, .n = n_rd};
    velri_pool_parallel(pool, nw, rd_fn, &bctx);
    for (int s = 0; s < 5; ++s) {
        double t0 = now_sec();
        velri_pool_parallel(pool, nw, rd_fn, &bctx);
        double gb = ((double)n_rd * sizeof(float)) / (now_sec() - t0) / 1e9;
        if (gb > out.bw_read_gbps) out.bw_read_gbps = gb;
    }

    bctx.n = n_rd / 2U;
    velri_pool_parallel(pool, nw, bw_copy_worker, &bctx);
    for (int s = 0; s < 5; ++s) {
        double t0 = now_sec();
        velri_pool_parallel(pool, nw, bw_copy_worker, &bctx);
        double gb = (2.0 * (double)bctx.n * sizeof(float)) / (now_sec() - t0) / 1e9;
        if (gb > out.bw_copy_gbps) out.bw_copy_gbps = gb;
    }
    velri_os_aligned_free(buf_a);
    velri_os_aligned_free(buf_b);

#if VELRI_X86_VARIANTS && !defined(_MSC_VER)
    const velri_cpu_info *ci = velri_get_cpu_info();
    if (ci->has_avx512f && ci->has_fma) {
        cpu_compute_ctx fctx = {.iters = 3000000U};
        for (int w = 0; w < 4; ++w) velri_pool_parallel(pool, nw, ceil_fma_f32_worker, &fctx);
        for (int s = 0; s < 5; ++s) {
            double t0 = now_sec();
            velri_pool_parallel(pool, nw, ceil_fma_f32_worker, &fctx);
            double gf = (double)nw * (double)fctx.iters * 12.0 * 32.0 / (now_sec() - t0) / 1e9;
            if (gf > out.fma_f32_gflops) out.fma_f32_gflops = gf;
        }
        velri_pool_parallel(pool, nw, ceil_fma_f64_worker, &fctx);
        for (int s = 0; s < 5; ++s) {
            double t0 = now_sec();
            velri_pool_parallel(pool, nw, ceil_fma_f64_worker, &fctx);
            double gf = (double)nw * (double)fctx.iters * 12.0 * 16.0 / (now_sec() - t0) / 1e9;
            if (gf > out.fma_f64_gflops) out.fma_f64_gflops = gf;
        }
    }
    if (ci->has_avx512f && ci->has_avx512_vnni) {
        cpu_compute_ctx vctx = {.iters = 3000000U};
        for (int w = 0; w < 2; ++w) velri_pool_parallel(pool, nw, ceil_vnni_i8_worker, &vctx);
        for (int s = 0; s < 5; ++s) {
            double t0 = now_sec();
            velri_pool_parallel(pool, nw, ceil_vnni_i8_worker, &vctx);
            double gops = (double)nw * (double)vctx.iters * 12.0 * 128.0 / (now_sec() - t0) / 1e9;
            if (gops > out.vnni_i8_gops) out.vnni_i8_gops = gops;
        }
    }
#endif
    velri_pool_destroy(pool);

    printf("\n=== Measured CPU Physical Ceilings (workers=%zu) ===\n", nw);
    printf("  DRAM Read-Only BW (512 MiB)  : %7.1f GB/s\n", out.bw_read_gbps);
    printf("  DRAM Copy BW (256->256 MiB)  : %7.1f GB/s\n", out.bw_copy_gbps);
    if (out.fma_f32_gflops > 0.0) {
        printf("  AVX-512 FP32 FMA Peak        : %7.3f TFLOPS (%.1f GFLOPS)\n",
               out.fma_f32_gflops * 1e-3, out.fma_f32_gflops);
        printf("  AVX-512 FP64 FMA Peak        : %7.3f TFLOPS (%.1f GFLOPS)\n",
               out.fma_f64_gflops * 1e-3, out.fma_f64_gflops);
    }
    if (out.vnni_i8_gops > 0.0) {
        printf("  AVX-512 INT8 VNNI Peak       : %7.3f TOPS   (%.1f GOPS)\n",
               out.vnni_i8_gops * 1e-3, out.vnni_i8_gops);
    }
    return out;
}

/* ---- Linear Graph Benchmark Helper (via public VelriLib Graph/Slot API) ---- */
static double bench_linear_sec(velri_context *ctx, velri_device_id dev, velri_linear_desc d,
                               size_t rows, size_t m, size_t k, size_t ring_len, size_t samples, size_t replays,
                               bool pack_weights) {
    enum { MAX_SAMPLES = 16 };
    if (samples == 0U || samples > MAX_SAMPLES) {
        fprintf(stderr, "FATAL: samples=%zu out of [1, %d]\n", samples, MAX_SAMPLES);
        exit(1);
    }
    size_t xb = velri_dtype_bytes(d.x_dtype);
    size_t wb = velri_dtype_bytes(d.w_dtype);
    size_t yb = velri_dtype_bytes(d.y_dtype);

    velri_graph *g = nullptr;
    check_ok(velri_graph_create(&g), "g");
    velri_storage_id sx = 0U, sy = 0U, sxs = 0U, sws = 0U;
    velri_storage_id *sw = (velri_storage_id *)xcalloc(ring_len, sizeof(*sw));
    velri_storage_desc dx = {VELRI_STORAGE_INPUT, d.x_dtype, rows * k * xb, 64U, dev};
    velri_storage_desc dy = {VELRI_STORAGE_OUTPUT, d.y_dtype, rows * m * yb, 64U, dev};
    check_ok(velri_graph_add_storage(g, &dx, &sx), "sx");
    check_ok(velri_graph_add_storage(g, &dy, &sy), "sy");
    if (d.has_x_scale) {
        velri_storage_desc dxs = {VELRI_STORAGE_INPUT, VELRI_DTYPE_F32, rows * sizeof(float), 64U, dev};
        check_ok(velri_graph_add_storage(g, &dxs, &sxs), "sxs");
    }
    if (d.has_w_scale) {
        velri_storage_desc dws = {VELRI_STORAGE_WEIGHT, VELRI_DTYPE_F32, m * sizeof(float), 64U, dev};
        check_ok(velri_graph_add_storage(g, &dws, &sws), "sws");
    }

    velri_tensor_view vxt, vyt, vxst, vwst;
    size_t xd[2] = {rows, k}, wd[2] = {m, k}, yd[2] = {rows, m}, xsd[1] = {rows}, wsd[1] = {m};
    velri_view_id vx = 0U, vy = 0U, vxs = VELRI_INVALID_ID, vws = VELRI_INVALID_ID;
    check_ok(velri_tensor_view_contiguous(sx, d.x_dtype, 2U, xd, 0U, &vxt), "vx");
    check_ok(velri_tensor_view_contiguous(sy, d.y_dtype, 2U, yd, 0U, &vyt), "vy");
    check_ok(velri_graph_add_view(g, &vxt, &vx), "avx");
    check_ok(velri_graph_add_view(g, &vyt, &vy), "avy");
    if (d.has_x_scale) {
        check_ok(velri_tensor_view_contiguous(sxs, VELRI_DTYPE_F32, 1U, xsd, 0U, &vxst), "vxs");
        check_ok(velri_graph_add_view(g, &vxst, &vxs), "avxs");
    }
    if (d.has_w_scale) {
        check_ok(velri_tensor_view_contiguous(sws, VELRI_DTYPE_F32, 1U, wsd, 0U, &vwst), "vws");
        check_ok(velri_graph_add_view(g, &vwst, &vws), "avws");
    }

    for (size_t r = 0; r < ring_len; ++r) {
        velri_storage_desc dw = {VELRI_STORAGE_WEIGHT, d.w_dtype, m * k * wb, 64U, dev};
        check_ok(velri_graph_add_storage(g, &dw, &sw[r]), "sw");
        velri_tensor_view vwt;
        velri_view_id vw = 0U;
        check_ok(velri_tensor_view_contiguous(sw[r], d.w_dtype, 2U, wd, 0U, &vwt), "vw");
        check_ok(velri_graph_add_view(g, &vwt, &vw), "avw");
        velri_op_desc op = {
            .kind = VELRI_OP_LINEAR,
            .num_inputs = 5U, .num_outputs = 1U,
            .inputs = {vx, vw, VELRI_INVALID_ID, vxs, vws},
            .outputs = {vy},
            .attr.linear.desc = d,
        };
        check_ok(velri_graph_add_op(g, &op, dev, nullptr), "op");
    }

    velri_plan *plan = nullptr;
    uint32_t plan_flags = pack_weights ? VELRI_PLAN_PACK_CPU_WEIGHTS : 0U;
    check_ok(velri_plan_prepare_ex(ctx, g, nullptr, plan_flags, &plan), "prep");

    uint64_t rng = 0xA0000001ULL ^ (uint64_t)rows ^ ((uint64_t)m << 12);
    void *hx = xalloc(rows * k * xb);
    void *hy = xalloc(rows * m * yb);
    float *hxs = d.has_x_scale ? (float *)xalloc(rows * sizeof(float)) : nullptr;
    float *hws = d.has_w_scale ? (float *)xalloc(m * sizeof(float)) : nullptr;
    velri_test_fill_dtype(d.x_dtype, hx, rows * k, &rng, 0.5f);
    if (hxs) velri_test_fill_dtype(VELRI_DTYPE_F32, hxs, rows, &rng, 0.02f);
    if (hws) velri_test_fill_dtype(VELRI_DTYPE_F32, hws, m, &rng, 0.02f);
    if (hws) check_ok(velri_plan_bind_weight_host(plan, sws, hws, m * sizeof(float)), "bws");

    void **hw = (void **)xcalloc(ring_len, sizeof(void *));
    for (size_t r = 0; r < ring_len; ++r) {
        hw[r] = xalloc(m * k * wb);
        velri_test_fill_dtype(d.w_dtype, hw[r], m * k, &rng, 0.5f);
        check_ok(velri_plan_bind_weight_host(plan, sw[r], hw[r], m * k * wb), "bw");
        if (velri_plan_weight_is_packed(plan, sw[r])) {
            velri_os_aligned_free(hw[r]);
            hw[r] = nullptr;
        }
    }

    velri_slot *slot = nullptr;
    check_ok(velri_slot_create(plan, &slot), "slot");
    check_ok(velri_slot_bind_host(slot, sx, hx, rows * k * xb), "bx");
    check_ok(velri_slot_bind_host(slot, sy, hy, rows * m * yb), "by");
    if (hxs) check_ok(velri_slot_bind_host(slot, sxs, hxs, rows * sizeof(float)), "bxs");

    check_ok(velri_slot_run(slot, nullptr, nullptr), "warmup");
    size_t eff_replays = replays;
    if (eff_replays == 0U) {
        /* Warm up until boost clocks stabilize (~30 ms) and scale replays to >=15 ms/sample. */
        double wt0 = now_sec();
        size_t wcalls = 0U;
        do {
            check_ok(velri_slot_run(slot, nullptr, nullptr), "warmup");
            ++wcalls;
        } while (now_sec() - wt0 < 0.03 && wcalls < 64U);
        double single = (now_sec() - wt0) / (double)wcalls;
        double target = single > 0.0 ? ceil(0.015 / single) : 1.0;
        if (target < 1.0) target = 1.0;
        if (target > 128.0) target = 128.0;
        eff_replays = (size_t)target;
    }
    double times[MAX_SAMPLES];
    for (size_t s = 0; s < samples; ++s) {
        double t0 = now_sec();
        for (size_t replay = 0; replay < eff_replays; ++replay)
            check_ok(velri_slot_run(slot, nullptr, nullptr), "run");
        times[s] = (now_sec() - t0) / (double)(ring_len * eff_replays);
    }

    velri_slot_destroy(slot);
    velri_plan_destroy(plan);
    velri_graph_destroy(g);
    for (size_t r = 0; r < ring_len; ++r) velri_os_aligned_free(hw[r]);
    free(hw); free(sw);
    velri_os_aligned_free(hx); velri_os_aligned_free(hy);
    velri_os_aligned_free(hxs); velri_os_aligned_free(hws);
    return median_of(times, samples);
}

int main(int argc, char **argv) {
    size_t nw = 0U;
    bool decode = false;
    velri_cpu_variant isa = VELRI_CPU_AUTO;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--decode") == 0) {
            decode = true;
        } else if (strcmp(argv[i], "--workers") == 0 && i + 1 < argc) {
            char *end;
            long value = strtol(argv[++i], &end, 10);
            if (end == argv[i] || *end || value < 1 || value > VELRI_POOL_MAX_WORKERS) return 2;
            nw = (size_t)value;
        } else if (strcmp(argv[i], "--isa") == 0 && i + 1 < argc) {
            const char *name = argv[++i];
            if (!strcmp(name, "base")) isa = VELRI_CPU_BASE;
            else if (!strcmp(name, "avx2")) isa = VELRI_CPU_AVX2;
            else if (!strcmp(name, "avx512")) isa = VELRI_CPU_AVX512;
            else return 2;
        } else {
            fprintf(stderr, "usage: %s [--workers N] [--decode [--isa base|avx2|avx512]]\n", argv[0]);
            return 2;
        }
    }
    if (isa != VELRI_CPU_AUTO && (!decode || !velri_cpu_variant_supported(isa))) {
        fprintf(stderr, "--isa requires --decode and a supported CPU variant\n");
        return 2;
    }
    const velri_cpu_info *ci = velri_get_cpu_info();
    if (nw == 0U) nw = (size_t)(ci->hardware_threads > 16 ? ci->hardware_threads / 2 : ci->hardware_threads);
    if (nw == 0U) nw = 1U;

    velri_context_config cfg = {.cpu_workers = nw, .cpu_variant = isa, .enable_vulkan = false};
    velri_context *ctx = nullptr;
    check_ok(velri_context_create(&cfg, &ctx), "ctx");
    velri_device_id dev = 0U;
    check_ok(velri_context_find_device(ctx, "cpu", 0U, &dev), "cpu:0");

    struct {
        const char *name;
        velri_linear_desc desc;
        bool pack_weights;
    } modes[] = {
        {"F32", velri_desc_f32(), false},
        {"F64", velri_desc_f64(), false},
        {"F16->F32", velri_desc_f16(VELRI_DTYPE_F32), false},
        {"BF16->F32", velri_desc_bf16(VELRI_DTYPE_F32), false},
        {"F32xBF16", velri_desc_f32x_bf16w(), false},
        {"W8A8->F32", velri_desc_w8a8_f32(), false},
        {"W8A8(pack)", velri_desc_w8a8_f32(), true},
    };

    if (decode) {
        decode_csv_header();
        for (size_t i = 0; i < sizeof(modes) / sizeof(modes[0]); ++i) {
            for (size_t j = 0; j < sizeof(decode_shapes) / sizeof(decode_shapes[0]); ++j) {
                size_t weights = 1U;
                if (decode_shapes[j].stream)
                    weights = modes[i].desc.w_dtype == VELRI_DTYPE_I8 ? 16U : 4U;
                double sec = bench_linear_sec(ctx, dev, modes[i].desc, decode_shapes[j].rows,
                                              decode_shapes[j].m, decode_shapes[j].k, weights, 7U,
                                              weights == 1U ? 16U : 1U, modes[i].pack_weights);
                decode_csv_row(modes[i].name, modes[i].desc, j, weights, sec * 1000.0);
            }
        }
        velri_context_destroy(ctx);
        return 0;
    }

    printf("Velri-lib Slang AOT CPU Benchmark (hw_threads=%d, workers=%zu, AVX2=%d, AVX-512=%d)\n",
           ci->hardware_threads, nw, (int)ci->has_avx2, (int)ci->has_avx512f);

    cpu_ceilings ceil = measure_cpu_ceilings(nw);

    printf("\n=== Cold-DRAM GEMV (rows=1, M=4096, K=4096, >=512 MiB weight ring) ===\n");
    printf("%-11s | %11s | %7s | %13s\n", "Mode", "Bandwidth", "%DRAM", "Compute");
    printf("------------+-------------+---------+--------------\n");
    const double gemv_ops = 2.0 * 1.0 * 4096.0 * 4096.0;
    for (size_t i = 0; i < sizeof(modes) / sizeof(modes[0]); ++i) {
        velri_linear_desc d = modes[i].desc;
        size_t w_bytes = 4096U * 4096U * velri_dtype_bytes(d.w_dtype);
        size_t ring = (512U * 1024U * 1024U + w_bytes - 1U) / w_bytes;
        if (ring < 4U) ring = 4U;
        double sec = bench_linear_sec(ctx, dev, d, 1U, 4096U, 4096U, ring, 5U, 1U, modes[i].pack_weights);
        double bytes = (double)linear_logical_bytes(d, 1U, 4096U, 4096U);
        double gbps = bytes / sec / 1e9;
        double rate_t = gemv_ops / sec / 1e12;
        bool int8 = d.x_dtype == VELRI_DTYPE_I8 && d.w_dtype == VELRI_DTYPE_I8;
        printf("%-11s | %7.1f GB/s | %6.1f%% | %6.4f %s\n", modes[i].name, gbps,
               ceil.bw_read_gbps > 0.0 ? 100.0 * gbps / ceil.bw_read_gbps : 0.0, rate_t,
               int8 ? "TOPS  " : "TFLOPS");
    }

    const size_t gemm_shapes[3][3] = {
        {128U, 2048U, 2048U},
        {256U, 4096U, 4096U},
        {1024U, 4096U, 4096U},
    };
    for (size_t si = 0; si < 3U; ++si) {
        size_t mr = gemm_shapes[si][0], mm = gemm_shapes[si][1], mk = gemm_shapes[si][2];
        printf("\n=== GEMM (rows=%zu, M=%zu, K=%zu) ===\n", mr, mm, mk);
        printf("%-11s | %14s | %7s | %s\n", "Mode", "Throughput", "%Peak", "Reference");
        printf("------------+----------------+---------+-------------\n");
        double ops = 2.0 * (double)mr * (double)mm * (double)mk;
        for (size_t i = 0; i < sizeof(modes) / sizeof(modes[0]); ++i) {
            velri_linear_desc d = modes[i].desc;
            double sec = bench_linear_sec(ctx, dev, d, mr, mm, mk, 1U, 5U, 0U, modes[i].pack_weights);
            double rate_t = ops / sec / 1e12;
            bool int8 = d.x_dtype == VELRI_DTYPE_I8 && d.w_dtype == VELRI_DTYPE_I8;
            if (int8) {
                printf("%-11s | %7.3f TOPS   | ", modes[i].name, rate_t);
                if (ceil.vnni_i8_gops > 0.0) {
                    printf("%6.1f%% | %s\n", 100.0 * (rate_t * 1e3) / ceil.vnni_i8_gops, "INT8 VNNI");
                } else {
                    printf("    n/a | %s\n", "INT8 VNNI");
                }
                continue;
            }
            bool fp64 = d.y_dtype == VELRI_DTYPE_F64;
            const char *ref = fp64 ? "FP64 FMA" : "FP32 FMA";
            double ref_peak = fp64 ? ceil.fma_f64_gflops : ceil.fma_f32_gflops;
            printf("%-11s | %7.3f TFLOPS | ", modes[i].name, rate_t);
            if (ref_peak > 0.0) {
                printf("%6.1f%% | %s\n", 100.0 * (rate_t * 1e3) / ref_peak, ref);
            } else {
                printf("    n/a | %s\n", ref);
            }
        }
    }

    velri_context_destroy(ctx);
    return 0;
}
