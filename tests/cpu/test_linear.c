#if defined(__unix__) || defined(__APPLE__)
#define _GNU_SOURCE
#include <sys/mman.h>
#include <unistd.h>
#include <pthread.h>
#define HAVE_POSIX_TESTS 1
#endif
#include "velri_lib/linear.h"
#include "reference.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#define CHECK(c)                                                                                   \
    do {                                                                                           \
        if (!(c)) {                                                                                \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #c);                                \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)
static size_t comparisons;
static uint64_t rng = UINT64_C(0x2cd9e5397bac10f1);
static uint32_t random_u32(void) {
    rng ^= rng >> 12;
    rng ^= rng << 25;
    rng ^= rng >> 27;
    return (uint32_t)((rng * UINT64_C(2685821657736338717)) >> 32);
}
static float from_bits(uint32_t bits) {
    float x;
    memcpy(&x, &bits, sizeof x);
    return x;
}
static uint32_t bits(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof u);
    return u;
}
static float random_float(int broad) {
    uint32_t u = random_u32();
    if (broad) {
        /* Include every finite exponent, signs and subnormals. */
        if ((u & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000))
            u ^= UINT32_C(0x00800000);
        return from_bits(u);
    }
    return (float)((int32_t)(u % 65537) - 32768) / 8192.0f;
}
static float *allocate(size_t count) {
    float *p = malloc((count ? count : 1) * sizeof(float));
    CHECK(p);
    return p;
}
static void equal(const float *actual, const float *expected, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (bits(actual[i]) != bits(expected[i])) {
            fprintf(stderr, "bit mismatch at %zu: actual=%08x expected=%08x\n", i, bits(actual[i]),
                    bits(expected[i]));
            exit(1);
        }
        ++comparisons;
    }
}

static void shape(velri_cpu_variant variant, size_t rows, size_t m, size_t k, int broad,
                  int bias_on) {
    velri_linear_plan p;
    CHECK(velri_linear_query(m, k, variant, &p) == VELRI_LINEAR_OK);
    CHECK(p.columns <= m);
    CHECK(p.packed_count >= m * k && p.packed_count < 2 * m * k);
    if (m == 1)
        CHECK(p.packed_count == k);
    size_t xs = k + 3, ws = k + 5, ys = m + 7;
    size_t xn = rows ? (rows - 1) * xs + k : 0, wn = (m - 1) * ws + k,
           yn = rows ? (rows - 1) * ys + m : 0;
    /* Offset one float, so no SIMD alignment may be assumed. Exact capacities
     * exclude final row padding; prefix/suffix and row gaps remain canaries. */
    float *xb = allocate(xn + 2), *wb = allocate(wn + 2), *yb = allocate(yn + 2),
          *eb = allocate(yn + 2);
    float *pb = allocate(p.packed_count + 2), *bb = allocate(m + 2);
    float *x = xb + 1, *w = wb + 1, *y = yb + 1, *e = eb + 1, *packed = pb + 1, *b = bb + 1;
    const float guard = from_bits(UINT32_C(0x4e123456));
    for (size_t i = 0; i < xn + 2; ++i)
        xb[i] = random_float(broad);
    for (size_t i = 0; i < wn + 2; ++i)
        wb[i] = random_float(broad);
    for (size_t i = 0; i < yn + 2; ++i)
        yb[i] = eb[i] = guard;
    for (size_t i = 0; i < m + 2; ++i)
        bb[i] = random_float(broad);
    for (size_t i = 0; i < p.packed_count + 2; ++i)
        pb[i] = guard;
    CHECK(velri_linear_pack(&p, w, wn, ws, packed, p.packed_count) == VELRI_LINEAR_OK);
    CHECK(bits(pb[0]) == bits(guard) && bits(pb[p.packed_count + 1]) == bits(guard));
    for (size_t c = 0; c < m; c += p.columns)
        for (size_t i = 0; i < k; ++i)
            for (size_t j = 0; j < p.columns; ++j) {
                uint32_t expected = c + j < m ? bits(w[(c + j) * ws + i]) : 0;
                CHECK(bits(packed[c * k + i * p.columns + j]) == expected);
            }
    reference_linear(x, w, bias_on ? b : NULL, e, rows, m, k, xs, ws, ys);
    CHECK(velri_linear_run(&p, packed, p.packed_count, x, xn, rows, xs, bias_on ? b : NULL,
                           bias_on ? m : 0, y, yn, ys) == VELRI_LINEAR_OK);
    equal(yb, eb, yn + 2);
    /* Reuse an immutable packing with changed input, without preparation. */
    for (size_t i = 0; i < xn; ++i)
        x[i] = random_float(broad);
    reference_linear(x, w, NULL, e, rows, m, k, xs, ws, ys);
    CHECK(velri_linear_run(&p, packed, p.packed_count, x, xn, rows, xs, NULL, 0, y, yn, ys) ==
          VELRI_LINEAR_OK);
    equal(yb, eb, yn + 2);
    free(xb);
    free(wb);
    free(yb);
    free(eb);
    free(pb);
    free(bb);
}

static void special(velri_cpu_variant variant) {
    const uint32_t values[] = {0,          0x80000000, 1,          0x80000001,
                               0x00800000, 0x80800000, 0x3f800000, 0xbf800000,
                               0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
                               0x7fc12345, 0xffc54321, 0x7fa00123, 0xffa00321};
    const size_t ks[] = {1, 3, 17, 129};
    for (size_t z = 0; z < sizeof ks / sizeof *ks; ++z) {
        size_t k = ks[z], m = 65;
        velri_linear_plan p;
        CHECK(velri_linear_query(m, k, variant, &p) == 0);
        float x[3 * 129], w[65 * 129], b[65], y[3 * 65], e[3 * 65];
        float *packed = allocate(p.packed_count);
        for (size_t a = 0; a < 16; ++a)
            for (size_t q = 0; q < 16; ++q) {
                for (size_t i = 0; i < 3 * k; ++i)
                    x[i] = from_bits(values[a]);
                for (size_t i = 0; i < m * k; ++i)
                    w[i] = 1.0f;
                for (size_t c = 0; c < m; ++c) {
                    w[c * k + c % k] = from_bits(values[q]);
                    b[c] = from_bits(values[c % 16]);
                }
                CHECK(velri_linear_pack(&p, w, m * k, k, packed, p.packed_count) == 0);
                for (int bias = 0; bias < 2; ++bias) {
                    reference_linear(x, w, bias ? b : NULL, e, 3, m, k, k, k, m);
                    CHECK(velri_linear_run(&p, packed, p.packed_count, x, 3 * k, 3, k,
                                           bias ? b : NULL, bias ? m : 0, y, 3 * m, m) == 0);
                    equal(y, e, 3 * m);
                }
            }
        free(packed);
    }
    /* An FP32 accumulation or a reassociated K tree must fail these. */
    float x[] = {1, 1, 1}, w[] = {16777216.0f, 1, -16777216.0f, 0x1p100f, 1, -0x1p100f};
    float y[2], e[2] = {1, 0};
    velri_linear_plan p;
    CHECK(velri_linear_query(2, 3, variant, &p) == 0);
    float *packed = allocate(p.packed_count);
    CHECK(velri_linear_pack(&p, w, 6, 3, packed, p.packed_count) == 0);
    CHECK(velri_linear_run(&p, packed, p.packed_count, x, 3, 1, 3, NULL, 0, y, 2, 2) == 0);
    equal(y, e, 2);
    free(packed);
}

/* Panel boundaries must carry an FP64 accumulator, not a rounded FP32
 * output or a second independent K reduction. The 25th row also crosses the
 * blocked row group, while M=65 exercises a one-column prepared tail. */
static void panel_contract(velri_cpu_variant variant) {
    const size_t rows = 25, m = 65, k = 257;
    velri_linear_plan p;
    CHECK(velri_linear_query(m, k, variant, &p) == 0);
    float *x = allocate(rows * k), *w = allocate(m * k), *y = allocate(rows * m);
    float *expected = allocate(rows * m), *packed = allocate(p.packed_count);
    for (size_t i = 0; i < rows * k; ++i)
        x[i] = 1.0f;
    memset(w, 0, m * k * sizeof(float));
    for (size_t c = 0; c < m; ++c) {
        const size_t start = c % 2 ? 254 : 126;
        const float large = c % 3 ? 0x1p24f : 0x1p100f;
        w[c * k + start] = large;
        w[c * k + start + 1] = 1.0f;
        w[c * k + start + 2] = -large;
        for (size_t r = 0; r < rows; ++r)
            expected[r * m + c] = c % 3 ? 1.0f : 0.0f;
    }
    CHECK(velri_linear_pack(&p, w, m * k, k, packed, p.packed_count) == 0);
    CHECK(velri_linear_run(&p, packed, p.packed_count, x, rows * k, rows, k,
                           NULL, 0, y, rows * m, m) == 0);
    equal(y, expected, rows * m);
    reference_linear(x, w, NULL, y, rows, m, k, k, k, m);
    equal(y, expected, rows * m);
    free(x); free(w); free(y); free(expected); free(packed);
}

static void invalid(velri_cpu_variant variant) {
    velri_linear_plan p;
    unsigned char original[sizeof p];
    memset(&p, 0xa5, sizeof p);
    memcpy(original, &p, sizeof p);
    CHECK(velri_linear_query(0, 3, variant, &p) == VELRI_LINEAR_INVALID);
    CHECK(memcmp(&p, original, sizeof p) == 0);
    CHECK(velri_linear_query(SIZE_MAX, 2, variant, &p) == VELRI_LINEAR_OVERFLOW);
    CHECK(memcmp(&p, original, sizeof p) == 0);
    CHECK(velri_linear_query(2, 3, (velri_cpu_variant)99, &p) == VELRI_LINEAR_UNSUPPORTED);
    CHECK(velri_linear_query(2, 3, variant, &p) == 0);
    float w[6] = {1, 2, 3, 4, 5, 6}, x[3] = {1, 2, 3}, y[2] = {19, 23}, expected[2] = {19, 23};
    float *packed = allocate(p.packed_count);
    for (size_t i = 0; i < p.packed_count; ++i)
        packed[i] = 77;
    CHECK(velri_linear_pack(&p, w, 5, 3, packed, p.packed_count) == VELRI_LINEAR_INVALID);
    for (size_t i = 0; i < p.packed_count; ++i)
        CHECK(packed[i] == 77);
    CHECK(velri_linear_pack(&p, w, 6, 2, packed, p.packed_count) == VELRI_LINEAR_INVALID);
    CHECK(velri_linear_pack(&p, w, 6, SIZE_MAX, packed, p.packed_count) == VELRI_LINEAR_OVERFLOW);
    CHECK(velri_linear_pack(&p, packed, p.packed_count, 3, packed, p.packed_count) ==
          VELRI_LINEAR_INVALID);
    CHECK(velri_linear_pack(&p, w, 6, 3, packed, p.packed_count - 1) == VELRI_LINEAR_INVALID);
    CHECK(velri_linear_pack(&p, w, 6, 3, packed, p.packed_count) == 0);
    CHECK(velri_linear_run(&p, packed, p.packed_count, x, 2, 1, 3, NULL, 0, y, 2, 2) ==
          VELRI_LINEAR_INVALID);
    CHECK(velri_linear_run(&p, packed, p.packed_count, x, 3, 1, 3, NULL, 1, y, 2, 2) ==
          VELRI_LINEAR_INVALID);
    CHECK(velri_linear_run(&p, packed, p.packed_count, x, 3, 1, 3, x, 1, y, 2, 2) ==
          VELRI_LINEAR_INVALID);
    CHECK(velri_linear_run(&p, packed, p.packed_count, x, 3, 1, 3, NULL, 0, y, 1, 2) ==
          VELRI_LINEAR_INVALID);
    CHECK(velri_linear_run(&p, packed, p.packed_count, x, 3, 1, 3, NULL, 0, y, 2, 1) ==
          VELRI_LINEAR_INVALID);
    CHECK(velri_linear_run(&p, packed, p.packed_count, x, 3, SIZE_MAX, 3, NULL, 0, y, 2, 2) ==
          VELRI_LINEAR_OVERFLOW);
    CHECK(velri_linear_run(&p, packed, p.packed_count, x, 3, 1, 3, NULL, 0, (float *)x, 3, 2) ==
          VELRI_LINEAR_INVALID);
    CHECK(velri_linear_run(&p, packed, p.packed_count, x, 3, 1, 3, NULL, 0, packed, p.packed_count,
                           2) == VELRI_LINEAR_INVALID);
    CHECK(velri_linear_run(&p, packed, p.packed_count, x, 3, 1, 3, y, 2, y, 2, 2) ==
          VELRI_LINEAR_INVALID);
    equal(y, expected, 2);
    CHECK(velri_linear_run(&p, packed, p.packed_count, NULL, 0, 0, 3, NULL, 0, NULL, 0, 2) == 0);
    /* All read-only arguments are allowed to share the same allocation. */
    reference_linear(packed, w, packed, expected, 1, 2, 3, 3, 3, 2);
    CHECK(velri_linear_run(&p, packed, p.packed_count, packed, 3, 1, 3, packed, 2, y, 2, 2) == 0);
    equal(y, expected, 2);
    free(packed);
}

static uint64_t read_le(FILE *f, size_t bytes) {
    uint64_t n = 0;
    for (size_t i = 0; i < bytes; ++i) {
        int c = fgetc(f);
        CHECK(c != EOF);
        n |= (uint64_t)(unsigned)c << (8 * i);
    }
    return n;
}
static void fixture(velri_cpu_variant variant) {
    FILE *f = fopen(VELRI_FIXTURE_DIR "/rust-linear-v1.bin", "rb");
    CHECK(f);
    CHECK(read_le(f, 4) == UINT32_C(0x31465256));
    size_t n = (size_t)read_le(f, 8), m = (size_t)read_le(f, 8), k = (size_t)read_le(f, 8);
    CHECK(n == 3 && m == 33 && k == 257);
    float *x = allocate(n * k), *w = allocate(m * k), *b = allocate(m), *e = allocate(n * m),
          *y = allocate(n * m), *c = allocate(n * m);
    float *arrays[] = {x, w, b, e};
    size_t counts[] = {n * k, m * k, m, n * m};
    for (size_t a = 0; a < 4; ++a)
        for (size_t i = 0; i < counts[a]; ++i)
            arrays[a][i] = from_bits((uint32_t)read_le(f, 4));
    CHECK(fgetc(f) == EOF);
    fclose(f);
    velri_linear_plan p;
    CHECK(velri_linear_query(m, k, variant, &p) == 0);
    float *packed = allocate(p.packed_count);
    CHECK(velri_linear_pack(&p, w, m * k, k, packed, p.packed_count) == 0);
    CHECK(velri_linear_run(&p, packed, p.packed_count, x, n * k, n, k, b, m, y, n * m, m) == 0);
    reference_linear(x, w, b, c, n, m, k, k, k, m);
    equal(y, e, n * m);
    equal(c, e, n * m);
    free(x);
    free(w);
    free(b);
    free(e);
    free(y);
    free(c);
    free(packed);
}

#if HAVE_POSIX_TESTS
typedef struct {
    void *mapping;
    size_t bytes;
    float *data;
} guarded;
static guarded guard(size_t count) {
    long page = sysconf(_SC_PAGESIZE);
    CHECK(page > 0);
    size_t pages = (count * sizeof(float) + (size_t)page - 1) / (size_t)page;
    guarded g;
    g.bytes = (pages + 1) * (size_t)page;
    g.mapping = mmap(NULL, g.bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(g.mapping != MAP_FAILED);
    char *end = (char *)g.mapping + pages * (size_t)page;
    CHECK(mprotect(end, (size_t)page, PROT_NONE) == 0);
    g.data = (float *)(end - count * sizeof(float));
    return g;
}
static void bounds(velri_cpu_variant variant) {
    for (size_t m = 1; m <= 67; ++m) {
        size_t k = 17;
        velri_linear_plan p;
        CHECK(velri_linear_query(m, k, variant, &p) == 0);
        guarded x = guard(k), w = guard(m * k), b = guard(m), y = guard(m),
                packed = guard(p.packed_count);
        for (size_t i = 0; i < k; ++i)
            x.data[i] = 1;
        for (size_t i = 0; i < m * k; ++i)
            w.data[i] = 1;
        for (size_t i = 0; i < m; ++i)
            b.data[i] = 1;
        CHECK(velri_linear_pack(&p, w.data, m * k, k, packed.data, p.packed_count) == 0);
        CHECK(velri_linear_run(&p, packed.data, p.packed_count, x.data, k, 1, k, b.data, m, y.data,
                               m, m) == 0);
        for (size_t i = 0; i < m; ++i)
            CHECK(y.data[i] == 18);
        guarded gs[] = {x, w, b, y, packed};
        for (size_t i = 0; i < 5; ++i)
            CHECK(munmap(gs[i].mapping, gs[i].bytes) == 0);
    }
}
static void blocked_bounds(velri_cpu_variant variant) {
    const size_t ks[] = {63, 64, 127, 128, 129, 257, 2047, 2048, 2049, 8191, 8192, 8193};
    const size_t row_counts[] = {1, 3, 4, 5, 24, 25};
    for (size_t ki = 0; ki < sizeof ks / sizeof *ks; ++ki)
        for (size_t ri = 0; ri < sizeof row_counts / sizeof *row_counts; ++ri) {
            const size_t k = ks[ki], rows = row_counts[ri], m = rows == 1 ? 257 : 65;
            if (k > 257 && rows != 1)
                continue;
            velri_linear_plan p;
            CHECK(velri_linear_query(m, k, variant, &p) == 0);
            guarded x = guard(rows * k), w = guard(m * k), b = guard(m),
                    y = guard(rows * m), packed = guard(p.packed_count);
            for (size_t i = 0; i < rows * k; ++i) x.data[i] = 1.0f;
            for (size_t i = 0; i < m * k; ++i) w.data[i] = 1.0f;
            for (size_t i = 0; i < m; ++i) b.data[i] = 1.0f;
            CHECK(velri_linear_pack(&p, w.data, m * k, k, packed.data, p.packed_count) == 0);
            for (int bias = 0; bias < 2; ++bias) {
                CHECK(velri_linear_run(&p, packed.data, p.packed_count, x.data, rows * k,
                                       rows, k, bias ? b.data : NULL, bias ? m : 0,
                                       y.data, rows * m, m) == 0);
                for (size_t i = 0; i < rows * m; ++i)
                    CHECK(bits(y.data[i]) == bits((float)k + (float)bias));
            }
            guarded gs[] = {x, w, b, y, packed};
            for (size_t i = 0; i < sizeof gs / sizeof *gs; ++i)
                CHECK(munmap(gs[i].mapping, gs[i].bytes) == 0);
        }
}

typedef struct {
    const velri_linear_plan *p;
    const float *packed;
    float x[17];
    float y[65];
} shared_job;
static void *shared_run(void *ptr) {
    shared_job *a = ptr;
    for (size_t i = 0; i < 200; ++i)
        CHECK(velri_linear_run(a->p, a->packed, a->p->packed_count, a->x, 17, 1, 17, NULL, 0, a->y,
                               65, 65) == 0);
    return NULL;
}
static void sharing(velri_cpu_variant variant) {
    velri_linear_plan p;
    CHECK(velri_linear_query(65, 17, variant, &p) == 0);
    float w[65 * 17];
    for (size_t i = 0; i < 65 * 17; ++i)
        w[i] = random_float(0);
    float *packed = allocate(p.packed_count);
    CHECK(velri_linear_pack(&p, w, 65 * 17, 17, packed, p.packed_count) == 0);
    pthread_t threads[4];
    shared_job a[4];
    for (size_t i = 0; i < 4; ++i) {
        a[i].p = &p;
        a[i].packed = packed;
        for (size_t j = 0; j < 17; ++j)
            a[i].x[j] = random_float(0);
        CHECK(pthread_create(&threads[i], NULL, shared_run, &a[i]) == 0);
    }
    for (size_t i = 0; i < 4; ++i) {
        CHECK(pthread_join(threads[i], NULL) == 0);
        float e[65];
        reference_linear(a[i].x, w, NULL, e, 1, 65, 17, 17, 17, 65);
        equal(a[i].y, e, 65);
    }
    free(packed);
}
#endif

int main(void) {
    const size_t ms[] = {1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129};
    const size_t ks[] = {1, 2, 3, 7, 17, 63, 128, 257};
    for (velri_cpu_variant v = VELRI_CPU_BASE; v <= VELRI_CPU_AVX512; ++v) {
        if (!velri_cpu_variant_available(v)) {
            velri_linear_plan unavailable;
            CHECK(velri_linear_query(7, 11, v, &unavailable) == VELRI_LINEAR_UNSUPPORTED);
            printf("variant %d: unavailable (skipped)\n", v);
            continue;
        }
        size_t before = comparisons;
        fixture(v);
        invalid(v);
        special(v);
        panel_contract(v);
        for (size_t mi = 0; mi < sizeof ms / sizeof *ms; ++mi)
            for (size_t ki = 0; ki < sizeof ks / sizeof *ks; ++ki)
                for (int broad = 0; broad < 2; ++broad)
                    shape(v, (mi + ki) % 6, ms[mi], ks[ki], broad, (int)(mi % 2));
        shape(v, 1, 1024, 1024, 0, 1);
        shape(v, 1, 1025, 1023, 1, 0);
        shape(v, 17, 129, 127, 0, 1);
        for (size_t r = 23; r <= 25; ++r)
            for (size_t k = 127; k <= 129; ++k)
                shape(v, r, 65, k, 1, (int)(k % 2));
        for (size_t k = 2047; k <= 2049; ++k)
            shape(v, 1, 257, k, 1, (int)(k % 2));
        for (size_t k = 8191; k <= 8193; ++k)
            shape(v, 1, 129, k, 1, (int)(k % 2));
#if HAVE_POSIX_TESTS
        bounds(v);
        blocked_bounds(v);
        sharing(v);
#endif
        printf("variant %d: %zu exact output/guard comparisons passed\n", v, comparisons - before);
    }
    shape(VELRI_CPU_AUTO, 3, 33, 17, 1, 1);
    puts("PASS: independent reference, frozen Rust fixture, reuse, tails, bounds and validation");
    return 0;
}
