#include "os.h"
#include "pool.h"
#include "reference.h"
#include "velri_lib/velri_lib.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__) && VELRI_X86_VARIANTS && !defined(_MSC_VER)
#include <immintrin.h>
#endif

static void check_ok(velri_status st, const char *ctx) {
    if (st != VELRI_OK) {
        fprintf(stderr, "ERROR (%s): velri_status=%d\n", ctx, (int)st);
        exit(1);
    }
}

static void *xalloc(size_t bytes) {
    size_t sz = bytes ? ((bytes + 63U) & ~(size_t)63U) : 64U;
    void *p = velri_os_aligned_alloc(64U, sz);
    if (!p) {
        fprintf(stderr, "FATAL: aligned alloc failed\n");
        exit(1);
    }
    memset(p, 0, sz);
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
} cpu_ceilings;

typedef struct {
    const float *src;
    float *dst;
    size_t n;
    volatile double sink;
} cpu_bw_ctx;

static void bw_read_worker(void *v, size_t wid, size_t nw) {
    cpu_bw_ctx *c = (cpu_bw_ctx *)v;
    size_t i0 = ((c->n / 16U) * wid / nw) * 16U;
    size_t i1 = ((c->n / 16U) * (wid + 1U) / nw) * 16U;
    const float *p = c->src;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (size_t i = i0; i < i1; i += 16U) {
        s0 += p[i + 0U] + p[i + 4U] + p[i + 8U] + p[i + 12U];
        s1 += p[i + 1U] + p[i + 5U] + p[i + 9U] + p[i + 13U];
        s2 += p[i + 2U] + p[i + 6U] + p[i + 10U] + p[i + 14U];
        s3 += p[i + 3U] + p[i + 7U] + p[i + 11U] + p[i + 15U];
    }
    if (wid == 0U) c->sink = (double)(s0 + s1 + s2 + s3);
}

static void bw_copy_worker(void *v, size_t wid, size_t nw) {
    cpu_bw_ctx *c = (cpu_bw_ctx *)v;
    size_t i0 = ((c->n / 16U) * wid / nw) * 16U;
    size_t i1 = ((c->n / 16U) * (wid + 1U) / nw) * 16U;
    memcpy(c->dst + i0, c->src + i0, (i1 - i0) * sizeof(float));
}

#if defined(__x86_64__) && VELRI_X86_VARIANTS && !defined(_MSC_VER)
typedef struct {
    size_t iters;
    volatile double sink;
} cpu_fma_ctx;

__attribute__((target("avx512f,fma"))) static void ceil_fma_f32_worker(void *v, size_t wid,
                                                                       size_t nw) {
    (void)nw;
    cpu_fma_ctx *c = (cpu_fma_ctx *)v;
    __m512 a = _mm512_set1_ps(1.000001f + (float)wid * 1e-6f);
    __m512 b = _mm512_set1_ps(0.999999f);
    __m512 r[12];
    for (int j = 0; j < 12; ++j) r[j] = _mm512_set1_ps((float)(j + 1));
    for (size_t i = 0; i < c->iters; ++i) {
        for (int j = 0; j < 12; ++j) r[j] = _mm512_fmadd_ps(r[j], b, a);
    }
    __m512 s = r[0];
    for (int j = 1; j < 12; ++j) s = _mm512_add_ps(s, r[j]);
    if (wid == 0U) c->sink = (double)_mm512_reduce_add_ps(s);
}

__attribute__((target("avx512f,fma"))) static void ceil_fma_f64_worker(void *v, size_t wid,
                                                                       size_t nw) {
    (void)nw;
    cpu_fma_ctx *c = (cpu_fma_ctx *)v;
    __m512d a = _mm512_set1_pd(1.000001 + (double)wid * 1e-6);
    __m512d b = _mm512_set1_pd(0.999999);
    __m512d r[12];
    for (int j = 0; j < 12; ++j) r[j] = _mm512_set1_pd((double)(j + 1));
    for (size_t i = 0; i < c->iters; ++i) {
        for (int j = 0; j < 12; ++j) r[j] = _mm512_fmadd_pd(r[j], b, a);
    }
    __m512d s = r[0];
    for (int j = 1; j < 12; ++j) s = _mm512_add_pd(s, r[j]);
    if (wid == 0U) c->sink = _mm512_reduce_add_pd(s);
}
#endif

static cpu_ceilings measure_cpu_ceilings(size_t nw) {
    cpu_ceilings out = {0};
    velri_pool *pool = nullptr;
    check_ok(velri_pool_create(nw, &pool), "ceil pool");

    size_t n_rd = (512U * 1024U * 1024U) / sizeof(float);
    float *buf_a = (float *)xalloc(n_rd * sizeof(float));
    float *buf_b = (float *)xalloc((n_rd / 2U) * sizeof(float));
    for (size_t i = 0; i < n_rd; i += 1024U) buf_a[i] = 1.0f;
    for (size_t i = 0; i < n_rd / 2U; i += 1024U) buf_b[i] = 1.0f;

    cpu_bw_ctx bctx = {.src = buf_a, .dst = buf_b, .n = n_rd};
    velri_pool_parallel(pool, nw, bw_read_worker, &bctx);
    for (int s = 0; s < 5; ++s) {
        double t0 = now_sec();
        velri_pool_parallel(pool, nw, bw_read_worker, &bctx);
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

#if defined(__x86_64__) && VELRI_X86_VARIANTS && !defined(_MSC_VER)
    if (velri_get_cpu_info()->has_avx512f) {
        cpu_fma_ctx fctx = {.iters = 3000000U};
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
#endif
    velri_pool_destroy(pool);

    printf("\n=== Measured CPU Physical Ceilings (workers=%zu) ===\n", nw);
    printf("  DRAM Read-Only BW (512 MiB)  : %7.1f GB/s\n", out.bw_read_gbps);
    printf("  DRAM Copy BW (256->256 MiB)  : %7.1f GB/s\n", out.bw_copy_gbps);
    if (out.fma_f32_gflops > 0.0) {
        printf("  AVX-512 FP32 FMA Peak        : %7.1f GFLOPS\n", out.fma_f32_gflops);
        printf("  AVX-512 FP64 FMA Peak        : %7.1f GFLOPS\n", out.fma_f64_gflops);
    }
    return out;
}

/* ---- Linear Graph Benchmark Helper (via public VelriLib Graph/Slot API) ---- */
static double bench_linear_sec(velri_context *ctx, velri_device_id dev, velri_linear_desc d,
                               size_t rows, size_t m, size_t k, size_t ring_len, size_t samples) {
    size_t xb = velri_dtype_bytes(d.x_dtype);
    size_t wb = velri_dtype_bytes(d.w_dtype);
    size_t yb = velri_dtype_bytes(d.y_dtype);

    velri_graph *g = nullptr;
    check_ok(velri_graph_create(&g), "g");
    velri_storage_id sx = 0U, sy = 0U, sxs = 0U, sws = 0U;
    velri_storage_id *sw = (velri_storage_id *)calloc(ring_len, sizeof(*sw));
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
    check_ok(velri_plan_prepare(ctx, g, nullptr, &plan), "prep");

    uint64_t rng = 0xA0000001ULL ^ (uint64_t)rows ^ ((uint64_t)m << 12);
    void *hx = xalloc(rows * k * xb);
    void *hy = xalloc(rows * m * yb);
    float *hxs = d.has_x_scale ? (float *)xalloc(rows * sizeof(float)) : nullptr;
    float *hws = d.has_w_scale ? (float *)xalloc(m * sizeof(float)) : nullptr;
    velri_test_fill_dtype(d.x_dtype, hx, rows * k, &rng, 0.5f);
    if (hxs) velri_test_fill_dtype(VELRI_DTYPE_F32, hxs, rows, &rng, 0.02f);
    if (hws) velri_test_fill_dtype(VELRI_DTYPE_F32, hws, m, &rng, 0.02f);
    if (hws) check_ok(velri_plan_bind_weight_host(plan, sws, hws, m * sizeof(float)), "bws");

    void **hw = (void **)calloc(ring_len, sizeof(void *));
    for (size_t r = 0; r < ring_len; ++r) {
        hw[r] = xalloc(m * k * wb);
        velri_test_fill_dtype(d.w_dtype, hw[r], m * k, &rng, 0.5f);
        check_ok(velri_plan_bind_weight_host(plan, sw[r], hw[r], m * k * wb), "bw");
    }

    velri_slot *slot = nullptr;
    check_ok(velri_slot_create(plan, &slot), "slot");
    check_ok(velri_slot_bind_host(slot, sx, hx, rows * k * xb), "bx");
    check_ok(velri_slot_bind_host(slot, sy, hy, rows * m * yb), "by");
    if (hxs) check_ok(velri_slot_bind_host(slot, sxs, hxs, rows * sizeof(float)), "bxs");

    check_ok(velri_slot_run(slot, nullptr, nullptr), "warmup");
    double times[16];
    for (size_t s = 0; s < samples; ++s) {
        double t0 = now_sec();
        check_ok(velri_slot_run(slot, nullptr, nullptr), "run");
        times[s] = (now_sec() - t0) / (double)ring_len;
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
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--workers") == 0 && i + 1 < argc) nw = (size_t)atoi(argv[++i]);
    }
    const velri_cpu_info *ci = velri_get_cpu_info();
    if (nw == 0U) nw = (size_t)(ci->hardware_threads > 16 ? ci->hardware_threads / 2 : ci->hardware_threads);
    if (nw == 0U) nw = 1U;

    printf("Velri-lib Slang AOT CPU Benchmark (hw_threads=%d, workers=%zu, AVX2=%d, AVX-512=%d)\n",
           ci->hardware_threads, nw, (int)ci->has_avx2, (int)ci->has_avx512f);

    cpu_ceilings ceil = measure_cpu_ceilings(nw);

    velri_context_config cfg = {.cpu_workers = nw, .cpu_variant = VELRI_CPU_AUTO, .enable_vulkan = false};
    velri_context *ctx = nullptr;
    check_ok(velri_context_create(&cfg, &ctx), "ctx");
    velri_device_id dev = 0U;
    check_ok(velri_context_find_device(ctx, "cpu", 0U, &dev), "cpu:0");

    struct {
        const char *name;
        velri_linear_desc desc;
    } modes[] = {
        {"F32", velri_desc_f32()},
        {"F64", velri_desc_f64()},
        {"F16->F32", velri_desc_f16(VELRI_DTYPE_F32)},
        {"BF16->F32", velri_desc_bf16(VELRI_DTYPE_F32)},
        {"F32xBF16", velri_desc_f32x_bf16w()},
        {"W8A8->F32", velri_desc_w8a8_f32()},
    };

    printf("\n=== Cold-DRAM GEMV (rows=1, M=4096, K=4096, ring=8) ===\n");
    printf("%-11s | %11s | %7s\n", "Mode", "Throughput", "%DRAM");
    printf("------------+-------------+--------\n");
    for (size_t i = 0; i < sizeof(modes) / sizeof(modes[0]); ++i) {
        velri_linear_desc d = modes[i].desc;
        double sec = bench_linear_sec(ctx, dev, d, 1U, 4096U, 4096U, 8U, 5U);
        double bytes = (double)(4096U * 4096U * velri_dtype_bytes(d.w_dtype) +
                                4096U * velri_dtype_bytes(d.x_dtype) +
                                4096U * velri_dtype_bytes(d.y_dtype));
        double gbps = bytes / sec / 1e9;
        printf("%-11s | %7.1f GB/s | %6.1f%%\n", modes[i].name, gbps,
               ceil.bw_read_gbps > 0.0 ? 100.0 * gbps / ceil.bw_read_gbps : 0.0);
    }

    printf("\n=== GEMM (rows=128, M=2048, K=2048) ===\n");
    printf("%-11s | %13s | %7s\n", "Mode", "Throughput", "%FMA");
    printf("------------+---------------+--------\n");
    double flops = 2.0 * 128.0 * 2048.0 * 2048.0;
    for (size_t i = 0; i < sizeof(modes) / sizeof(modes[0]); ++i) {
        velri_linear_desc d = modes[i].desc;
        double sec = bench_linear_sec(ctx, dev, d, 128U, 2048U, 2048U, 1U, 5U);
        double gf = flops / sec / 1e9;
        double ref_peak = (d.y_dtype == VELRI_DTYPE_F64) ? ceil.fma_f64_gflops : ceil.fma_f32_gflops;
        if (ref_peak > 0.0) {
            printf("%-11s | %7.1f GFLOP | %6.1f%%\n", modes[i].name, gf, 100.0 * gf / ref_peak);
        } else {
            printf("%-11s | %7.1f GFLOP |     n/a\n", modes[i].name, gf);
        }
    }

    velri_context_destroy(ctx);
    return 0;
}
