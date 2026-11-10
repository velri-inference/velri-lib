/* Linux diagnostic, not the engine's worker pool. Threads and packed weights
 * persist across calls. Split complete output tiles, never the K reduction. */
#define _GNU_SOURCE
#include "velri_lib/linear.h"
#include "reference.h"
#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

#define MAX_WORKERS 256
#define GUARD 16
#define LATENCIES 262144

[[noreturn]] static void fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(1);
}

static size_t number(const char *text, int zero_allowed) {
    char *end;
    errno = 0;
    unsigned long long n = strtoull(text, &end, 10);
    if (errno || *end || end == text || *text == '-' || n > SIZE_MAX ||
        (!zero_allowed && !n))
        fail("expected a nonnegative integer (dimensions must be positive)");
    return (size_t)n;
}

static void *allocate(size_t count, size_t width) {
    if (count > SIZE_MAX / width)
        fail("allocation size overflow");
    void *p = malloc(count * width);
    if (!p)
        fail("allocation failed");
    return p;
}

static double seconds(clockid_t clock) {
    struct timespec t;
    if (clock_gettime(clock, &t))
        fail("clock_gettime failed");
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static void relax(void) {
#if defined(__i386__) || defined(__x86_64__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ volatile("yield");
#else
    sched_yield();
#endif
}

static void pin(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof set, &set))
        fail("cannot pin worker to requested CPU");
}

typedef struct {
    alignas(64) atomic_uint done;
    int cpu;
    size_t first, last;
} worker;

typedef struct {
    velri_linear_plan plan;
    const float *input, *packed, *bias;
    float *output;
    size_t count;
    worker workers[MAX_WORKERS];
    pthread_t threads[MAX_WORKERS];
    alignas(64) atomic_uint generation;
    int stopping;
} pool;

/* The plan/packing are validated before starting the pool. All slices begin
 * at complete packed tiles and have disjoint outputs. The last slice may end
 * inside the final padded tile. */
static void work(pool *p, size_t id) {
    const worker *w = &p->workers[id];
    p->plan.kernel(p->input, p->packed + w->first * p->plan.k,
                   p->bias ? p->bias + w->first : NULL, p->output + w->first, 1,
                   w->last - w->first, p->plan.k, p->plan.k, p->plan.m);
}

typedef struct {
    pool *p;
    size_t id;
} worker_arg;

static void *worker_main(void *arg) {
    worker_arg a = *(const worker_arg *)arg;
    pool *p = a.p;
    pin(p->workers[a.id].cpu);
    unsigned last = 0;
    for (;;) {
        unsigned gen;
        while ((gen = atomic_load_explicit(&p->generation, memory_order_acquire)) == last)
            relax();
        if (p->stopping)
            return NULL;
        work(p, a.id);
        atomic_store_explicit(&p->workers[a.id].done, gen, memory_order_release);
        last = gen;
    }
}

static void call(pool *p) {
    if (p->count == 1) {
        work(p, 0);
        return;
    }
    unsigned gen = atomic_load_explicit(&p->generation, memory_order_relaxed) + 1;
    if (!gen)
        fail("generation counter exhausted; start a new benchmark process");
    atomic_store_explicit(&p->generation, gen, memory_order_release);
    work(p, 0);
    for (size_t id = 1; id < p->count; ++id)
        while (atomic_load_explicit(&p->workers[id].done, memory_order_acquire) != gen)
            relax();
}

static void equal(const float *a, const float *b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        uint32_t actual, expected;
        memcpy(&actual, a + i, sizeof actual);
        memcpy(&expected, b + i, sizeof expected);
        if (actual != expected) {
            fprintf(stderr, "element %zu: %08x != reference %08x\n", i, actual, expected);
            exit(1);
        }
    }
}

static void check(const pool *p, const float *reference) {
    equal(p->output, reference, p->plan.m);
    for (size_t i = 0; i < GUARD; ++i)
        if (p->output[-(ptrdiff_t)i - 1] != 12345.0f ||
            p->output[p->plan.m + i] != 12345.0f)
            fail("output guard overwritten");
}

/* Align a separate mapping to the system's huge-page size before first touch.
 * madvise is only a request: report actual AnonHugePages from this mapping. */
typedef struct {
    void *base;
    size_t length;
    float *data;
} mapping;

static mapping packed_storage(size_t count, int huge) {
    long raw_page = sysconf(_SC_PAGESIZE);
    if (raw_page <= 0)
        fail("cannot determine page size");
    size_t page = (size_t)raw_page, align = page;
    if (huge) {
        FILE *f = fopen("/sys/kernel/mm/transparent_hugepage/hpage_pmd_size", "r");
        if (!f || fscanf(f, "%zu", &align) != 1 || align < page || align % page)
            fail("cannot determine transparent huge-page size");
        fclose(f);
    }
    if (count > (SIZE_MAX - page) / sizeof(float))
        fail("packed size overflow");
    size_t bytes = count * sizeof(float), rounded = (bytes + page - 1) / page * page;
    if (align > SIZE_MAX - 2 * page || rounded > SIZE_MAX - align - 2 * page)
        fail("mapping size overflow");
    size_t total = rounded + align + 2 * page;
    void *base = mmap(NULL, total, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED)
        fail("mmap failed");
    uintptr_t address = (uintptr_t)base + page;
    address += (align - address % align) % align;
    void *data = (void *)address;
    if (mprotect(data, rounded, PROT_READ | PROT_WRITE) ||
        madvise(data, rounded, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE))
        fail("page setup failed");
    return (mapping){base, total, data};
}

static size_t huge_bytes(const float *p) {
    FILE *f = fopen("/proc/self/smaps", "r");
    if (!f)
        fail("cannot read /proc/self/smaps");
    char line[512];
    int active = 0;
    size_t bytes = 0;
    while (fgets(line, sizeof line, f)) {
        unsigned long long first, last;
        size_t kb;
        if (sscanf(line, "%llx-%llx", &first, &last) == 2)
            active = (uintptr_t)p >= first && (uintptr_t)p < last;
        else if (active && sscanf(line, "AnonHugePages: %zu kB", &kb) == 1) {
            bytes = kb * 1024;
            break;
        }
    }
    fclose(f);
    return bytes;
}

static int compare(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void measure(pool *p, const float *reference, double *latency, size_t sample,
                    double read_gbps) {
    double warm = seconds(CLOCK_MONOTONIC);
    do {
        call(p);
    } while (seconds(CLOCK_MONOTONIC) - warm < 0.03);
    check(p, reference);
    struct rusage before, after;
    if (getrusage(RUSAGE_SELF, &before))
        fail("getrusage failed");
    size_t n = 0;
    double cpu = seconds(CLOCK_PROCESS_CPUTIME_ID), start = seconds(CLOCK_MONOTONIC), end;
    do {
        double a = seconds(CLOCK_MONOTONIC);
        call(p);
        end = seconds(CLOCK_MONOTONIC);
        latency[n++] = (end - a) * 1e6;
    } while (end - start < 0.1 && n < LATENCIES);
    cpu = seconds(CLOCK_PROCESS_CPUTIME_ID) - cpu;
    if (getrusage(RUSAGE_SELF, &after))
        fail("getrusage failed");
    check(p, reference);
    qsort(latency, n, sizeof *latency, compare);
    double tflops = 2.0 * p->plan.m * p->plan.k * n / (end - start) / 1e12;
    printf("{\"kind\":\"sample\",\"sample\":%zu,\"calls\":%zu,\"wall_s\":%.9g,"
           "\"cpu_s\":%.9g,\"TFLOPS\":%.9g,\"p50_us\":%.9g,\"p99_us\":%.9g,"
           "\"max_us\":%.9g,\"involuntary_switches\":%ld,\"minor_faults\":%ld,"
           "\"bit_mismatches\":0",
           sample, n, end - start, cpu, tflops, latency[n / 2], latency[(n - 1) * 99 / 100],
           latency[n - 1], after.ru_nivcsw - before.ru_nivcsw,
           after.ru_minflt - before.ru_minflt);
    if (read_gbps > 0)
        printf(",\"read_control_TFLOPS\":%.9g,\"fraction_of_read_control\":%.9g",
               read_gbps / 2000.0, tflops / (read_gbps / 2000.0));
    puts("}");
    fflush(stdout);
}

int main(int argc, char **argv) {
    if (argc != 8 && argc != 9) {
        fprintf(stderr, "usage: %s auto|base|avx2|avx512 M K cpu-list samples normal|huge "
                        "verify-jobs [read-control-GB/s]\n"
                        "samples=0 runs correctness checks only. CPU IDs must be explicit.\n",
                argv[0]);
        return 2;
    }
    velri_cpu_variant variant;
    if (!strcmp(argv[1], "auto")) variant = VELRI_CPU_AUTO;
    else if (!strcmp(argv[1], "base")) variant = VELRI_CPU_BASE;
    else if (!strcmp(argv[1], "avx2")) variant = VELRI_CPU_AVX2;
    else if (!strcmp(argv[1], "avx512")) variant = VELRI_CPU_AVX512;
    else fail("unknown variant");
    size_t m = number(argv[2], 0), k = number(argv[3], 0);
    size_t samples = number(argv[5], 1), jobs = number(argv[7], 0);
    if (samples > 10001 || jobs > 1000000)
        fail("too many samples or verification jobs");
    int huge = !strcmp(argv[6], "huge");
    if (!huge && strcmp(argv[6], "normal"))
        fail("page mode must be normal or huge");
    double read_gbps = 0;
    if (argc == 9) {
        char *end;
        errno = 0;
        read_gbps = strtod(argv[8], &end);
        if (errno || end == argv[8] || *end || !(read_gbps > 0 && read_gbps < 1e9))
            fail("read control must be a finite positive GB/s value");
    }
    static pool p;
    if (velri_linear_query(m, k, variant, &p.plan) != VELRI_LINEAR_OK)
        fail("unsupported variant or invalid dimensions");
    cpu_set_t allowed, used;
    CPU_ZERO(&used);
    if (sched_getaffinity(0, sizeof allowed, &allowed))
        fail("cannot read process affinity");
    char *list = argv[4];
    for (;;) {
        char *comma = strchr(list, ',');
        if (comma) *comma = '\0';
        size_t cpu = number(list, 1);
        if (cpu >= CPU_SETSIZE || !CPU_ISSET(cpu, &allowed) || CPU_ISSET(cpu, &used) ||
            p.count == MAX_WORKERS)
            fail("CPU list contains unavailable/duplicate IDs or too many workers");
        CPU_SET(cpu, &used);
        p.workers[p.count++].cpu = (int)cpu;
        if (!comma) break;
        list = comma + 1;
    }
    size_t tiles = p.plan.packed_count / k / p.plan.columns;
    size_t requested = p.count;
    if (p.count > tiles) p.count = tiles;
    pin(p.workers[0].cpu);
    for (size_t id = 0; id < p.count; ++id) {
        /* Division first also avoids overflowing tiles * id. */
        size_t first = tiles / p.count * id + (tiles % p.count * id) / p.count;
        size_t next = id + 1;
        size_t last = tiles / p.count * next + (tiles % p.count * next) / p.count;
        p.workers[id].first = first * p.plan.columns;
        p.workers[id].last = last == tiles ? m : last * p.plan.columns;
    }
    if (m > SIZE_MAX - 2 * GUARD || m > SIZE_MAX / k)
        fail("matrix size overflow");
    float *input = allocate(k, sizeof(float)), *weights = allocate(m * k, sizeof(float));
    float *bias = allocate(m, sizeof(float)), *reference = allocate(m, sizeof(float));
    float *guarded = allocate(m + 2 * GUARD, sizeof(float));
    p.input = input;
    p.output = guarded + GUARD;
    for (size_t i = 0; i < m + 2 * GUARD; ++i) guarded[i] = 12345.0f;
    uint32_t seed = 123456789;
    for (size_t i = 0; i < m * k; ++i) {
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        weights[i] = ((int32_t)(seed & 65535) - 32768) / 16384.0f;
    }
    for (size_t i = 0; i < k; ++i) input[i] = ((int)(i % 257) - 128) / 127.0f;
    for (size_t i = 0; i < m; ++i) bias[i] = ((int)(i % 31) - 15) / 17.0f;
    mapping packed = packed_storage(p.plan.packed_count, huge);
    p.packed = packed.data;
    if (velri_linear_pack(&p.plan, weights, m * k, k, packed.data, p.plan.packed_count) ||
        velri_linear_run(&p.plan, p.packed, p.plan.packed_count, input, k, 1, k,
                         NULL, 0, p.output, m, m))
        fail("packing or view validation failed");
    printf("{\"kind\":\"metadata\",\"M\":%zu,\"K\":%zu,\"columns\":%zu,"
           "\"requested_workers\":%zu,\"workers\":%zu,\"packed_bytes\":%zu,"
           "\"huge_requested\":%d,\"huge_bytes\":%zu,\"cpus\":[",
           m, k, p.plan.columns, requested, p.count, p.plan.packed_count * sizeof(float),
           huge, huge_bytes(p.packed));
    for (size_t i = 0; i < p.count; ++i) printf("%s%d", i ? "," : "", p.workers[i].cpu);
    puts("]}");
    worker_arg args[MAX_WORKERS];
    for (size_t id = 1; id < p.count; ++id) {
        args[id] = (worker_arg){&p, id};
        if (pthread_create(&p.threads[id], NULL, worker_main, &args[id]))
            fail("pthread_create failed");
    }
    for (size_t j = 0; j < jobs; ++j) {
        input[j % k] += 0.125f;
        p.bias = j % 2 ? bias : NULL;
        reference_linear(input, weights, p.bias, reference, 1, m, k, k, k, m);
        for (size_t i = 0; i < m; ++i) p.output[i] = 12345.0f;
        call(&p);
        check(&p, reference);
    }
    p.bias = NULL;
    reference_linear(input, weights, NULL, reference, 1, m, k, k, k, m);
    free(weights);
    double *latency = allocate(LATENCIES, sizeof(double));
    /* Fault in the timing array before measuring; it is outside the working set. */
    for (size_t i = 0; i < LATENCIES; ++i) ((volatile double *)latency)[i] = 0;
    for (size_t sample = 0; sample < samples; ++sample)
        measure(&p, reference, latency, sample, read_gbps);
    p.stopping = 1;
    unsigned gen = atomic_load_explicit(&p.generation, memory_order_relaxed) + 1;
    atomic_store_explicit(&p.generation, gen, memory_order_release);
    for (size_t id = 1; id < p.count; ++id)
        if (pthread_join(p.threads[id], NULL)) fail("pthread_join failed");
    if (munmap(packed.base, packed.length)) fail("munmap failed");
    free(latency); free(guarded); free(reference); free(bias); free(input);
    printf("{\"kind\":\"complete\",\"verified_jobs\":%zu,\"bit_mismatches\":0}\n", jobs);
    return 0;
}
