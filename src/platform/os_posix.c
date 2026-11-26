#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 700
#endif

#include "os.h"

#include <dlfcn.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(__linux__)
#include <errno.h>
#include <sched.h>
#include <stdio.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#if __has_include(<unistd.h>)
#include <unistd.h>
#endif

/* Ensure worker threads have enough stack for L1-packed BLIS panels even on
 * small-default-stack libc implementations (e.g. musl 128 KiB). */
static const size_t kMinWorkerStackBytes = 512U * 1024U;

/* ---- Aligned Memory ---- */

void *velri_os_aligned_alloc(size_t align, size_t bytes) {
    size_t eff_align = 0U, sz = 0U;
    if (!velri_os_aligned_size(align, bytes, &eff_align, &sz)) {
        return nullptr;
    }
    void *p = nullptr;
    if (posix_memalign(&p, eff_align, sz) != 0) {
        return nullptr;
    }
    return p;
}

void velri_os_aligned_free(void *ptr) { free(ptr); }

/* ---- Monotonic Time & Sleep ---- */

uint64_t velri_os_monotonic_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

void velri_os_sleep_ms(uint32_t ms) {
    if (ms == 0U) {
        return;
    }
    struct timespec ts = {
        .tv_sec = (time_t)(ms / 1000U),
        .tv_nsec = (long)((ms % 1000U) * 1000000U),
    };
    nanosleep(&ts, nullptr);
}

/* ---- Dynamic Shared Library Loading ---- */

velri_os_lib *velri_os_dlopen(const char *const *candidates) {
    if (!candidates) {
        return nullptr;
    }
    for (size_t i = 0U; candidates[i] != nullptr; ++i) {
        void *h = dlopen(candidates[i], RTLD_NOW | RTLD_LOCAL);
        if (h) {
            return (velri_os_lib *)h;
        }
    }
    return nullptr;
}

void *velri_os_dlsym(velri_os_lib *lib, const char *symbol) {
    if (!lib || !symbol) {
        return nullptr;
    }
    return dlsym((void *)lib, symbol);
}

void velri_os_dlclose(velri_os_lib *lib) {
    if (lib) {
        (void)dlclose((void *)lib);
    }
}

/* ---- Threads, Mutex, Condition Variable ---- */

struct velri_os_thread {
    pthread_t handle;
    velri_os_thread_fn fn;
    void *arg;
};

struct velri_os_mutex {
    pthread_mutex_t handle;
};

struct velri_os_cond {
    pthread_cond_t handle;
};

static void *posix_thread_trampoline(void *raw) {
    velri_os_thread *t = (velri_os_thread *)raw;
    t->fn(t->arg);
    return nullptr;
}

bool velri_os_thread_spawn(velri_os_thread **out_thread, velri_os_thread_fn fn, void *arg) {
    if (!out_thread || !fn) {
        return false;
    }
    *out_thread = nullptr;
    velri_os_thread *t = (velri_os_thread *)calloc(1U, sizeof(*t));
    if (!t) {
        return false;
    }
    t->fn = fn;
    t->arg = arg;

    pthread_attr_t attr;
    bool has_attr = (pthread_attr_init(&attr) == 0);
    if (has_attr) {
        size_t cur_stack = 0U;
        if (pthread_attr_getstacksize(&attr, &cur_stack) == 0 && cur_stack < kMinWorkerStackBytes) {
            (void)pthread_attr_setstacksize(&attr, kMinWorkerStackBytes);
        }
    }
    int rc = pthread_create(&t->handle, has_attr ? &attr : nullptr, posix_thread_trampoline, t);
    if (has_attr) {
        pthread_attr_destroy(&attr);
    }
    if (rc != 0) {
        free(t);
        return false;
    }
    *out_thread = t;
    return true;
}

void velri_os_thread_join(velri_os_thread *thread) {
    if (!thread) {
        return;
    }
    (void)pthread_join(thread->handle, nullptr);
    free(thread);
}

bool velri_os_mutex_create(velri_os_mutex **out_mtx) {
    if (!out_mtx) {
        return false;
    }
    *out_mtx = nullptr;
    velri_os_mutex *m = (velri_os_mutex *)calloc(1U, sizeof(*m));
    if (!m) {
        return false;
    }
    if (pthread_mutex_init(&m->handle, nullptr) != 0) {
        free(m);
        return false;
    }
    *out_mtx = m;
    return true;
}

void velri_os_mutex_destroy(velri_os_mutex *mtx) {
    if (!mtx) {
        return;
    }
    pthread_mutex_destroy(&mtx->handle);
    free(mtx);
}

void velri_os_mutex_lock(velri_os_mutex *mtx) { pthread_mutex_lock(&mtx->handle); }

void velri_os_mutex_unlock(velri_os_mutex *mtx) { pthread_mutex_unlock(&mtx->handle); }

bool velri_os_cond_create(velri_os_cond **out_cv) {
    if (!out_cv) {
        return false;
    }
    *out_cv = nullptr;
    velri_os_cond *c = (velri_os_cond *)calloc(1U, sizeof(*c));
    if (!c) {
        return false;
    }
    if (pthread_cond_init(&c->handle, nullptr) != 0) {
        free(c);
        return false;
    }
    *out_cv = c;
    return true;
}

void velri_os_cond_destroy(velri_os_cond *cv) {
    if (!cv) {
        return;
    }
    pthread_cond_destroy(&cv->handle);
    free(cv);
}

void velri_os_cond_wait(velri_os_cond *cv, velri_os_mutex *mtx) {
    pthread_cond_wait(&cv->handle, &mtx->handle);
}

void velri_os_cond_signal(velri_os_cond *cv) { pthread_cond_signal(&cv->handle); }

void velri_os_cond_broadcast(velri_os_cond *cv) { pthread_cond_broadcast(&cv->handle); }

/* ---- CPU Hints ---- */

int velri_os_hardware_threads(void) {
#if defined(_SC_NPROCESSORS_ONLN)
    long nproc = sysconf(_SC_NPROCESSORS_ONLN);
    if (nproc > 0 && nproc < 65536) {
        return (int)nproc;
    }
#endif
    return 1;
}

#if defined(__linux__)
/* Query the calling thread's CPU affinity mask with dynamic sizing (>1024 CPUs supported). */
static size_t query_process_affinity(cpu_set_t **out_set, size_t *out_bytes) {
    size_t ncpus = 1024U;
#if defined(_SC_NPROCESSORS_CONF)
    long conf = sysconf(_SC_NPROCESSORS_CONF);
    if (conf > 0 && (size_t)conf > ncpus) {
        ncpus = ((size_t)conf + 1023U) & ~(size_t)1023U;
    }
#endif
    for (int attempt = 0; attempt < 8; ++attempt) {
        cpu_set_t *set = CPU_ALLOC(ncpus);
        if (!set) {
            return 0U;
        }
        size_t sz = CPU_ALLOC_SIZE(ncpus);
        CPU_ZERO_S(sz, set);
        if (sched_getaffinity(0, sz, set) == 0) {
            *out_set = set;
            *out_bytes = sz;
            return ncpus;
        }
        int err = errno;
        CPU_FREE(set);
        if (err != EINVAL || ncpus >= 65536U) {
            return 0U;
        }
        ncpus *= 2U;
    }
    return 0U;
}

/* Parse `/sys/devices/system/cpu/cpu<N>/topology/thread_siblings_list` into `out_siblings`.
 * Fails closed (returns false) if sysfs is unavailable, malformed, or does not list `cpu`. */
static bool read_thread_siblings(int cpu, size_t ncpus, size_t mask_bytes,
                                 cpu_set_t *out_siblings) {
    if (cpu < 0 || (size_t)cpu >= ncpus) {
        return false;
    }
    char path[96];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list",
             cpu);
    FILE *file = fopen(path, "r");
    if (!file) {
        return false;
    }
    CPU_ZERO_S(mask_bytes, out_siblings);
    bool found_self = false;
    bool ok = true;
    int separator = EOF;
    do {
        unsigned first = 0U, last = 0U;
        if (fscanf(file, "%u", &first) != 1) {
            ok = false;
            break;
        }
        last = first;
        separator = fgetc(file);
        if (separator == '-') {
            if (fscanf(file, "%u", &last) != 1 || last < first) {
                ok = false;
                break;
            }
            separator = fgetc(file);
        }
        if ((size_t)last >= ncpus) {
            ok = false;
            break;
        }
        for (unsigned i = first; i <= last; ++i) {
            CPU_SET_S(i, mask_bytes, out_siblings);
            if ((int)i == cpu) {
                found_self = true;
            }
        }
    } while (separator == ',');
    fclose(file);
    return ok && found_self;
}

/* Enumerate distinct physical cores accessible under `allowed` (one representative CPU per core).
 * Fails closed (returns 0) if any allowed CPU's topology file cannot be read. */
static size_t collect_physical_cores(const cpu_set_t *allowed, size_t ncpus, size_t mask_bytes,
                                     int *out_cores, size_t max_cores) {
    cpu_set_t *visited = CPU_ALLOC(ncpus);
    cpu_set_t *siblings = CPU_ALLOC(ncpus);
    if (!visited || !siblings) {
        if (visited) CPU_FREE(visited);
        if (siblings) CPU_FREE(siblings);
        return 0U;
    }
    CPU_ZERO_S(mask_bytes, visited);
    size_t n_cores = 0U;
    for (size_t cpu = 0U; cpu < ncpus; ++cpu) {
        if (!CPU_ISSET_S(cpu, mask_bytes, allowed) || CPU_ISSET_S(cpu, mask_bytes, visited)) {
            continue;
        }
        if (!read_thread_siblings((int)cpu, ncpus, mask_bytes, siblings)) {
            CPU_FREE(visited);
            CPU_FREE(siblings);
            return 0U;
        }
        CPU_OR_S(mask_bytes, visited, visited, siblings);
        if (out_cores && n_cores < max_cores) {
            out_cores[n_cores] = (int)cpu;
        }
        ++n_cores;
    }
    CPU_FREE(visited);
    CPU_FREE(siblings);
    return n_cores;
}

/* Skip static pinning on hybrid (Intel P+E) or asymmetric (ARM big.LITTLE) topologies
 * where ascending CPU indices can place workers on slower efficiency cores. */
static bool is_heterogeneous_topology(const int *cores, size_t n_cores) {
    FILE *atom = fopen("/sys/devices/cpu_atom/cpus", "r");
    if (atom) {
        fclose(atom);
        return true;
    }
    bool have_cap = false;
    unsigned ref_cap = 0U;
    for (size_t i = 0U; i < n_cores; ++i) {
        char path[96];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpu_capacity", cores[i]);
        FILE *f = fopen(path, "r");
        if (!f) {
            continue;
        }
        unsigned cap = 0U;
        if (fscanf(f, "%u", &cap) == 1) {
            if (!have_cap) {
                ref_cap = cap;
                have_cap = true;
            } else if (cap != ref_cap) {
                fclose(f);
                return true;
            }
        }
        fclose(f);
    }
    return false;
}

struct velri_os_pin_plan {
    size_t num_workers;
    size_t ncpus;
    size_t mask_bytes;
    int *worker_cores;
    cpu_set_t *worker_mask;
    cpu_set_t *reserved_mask;
    cpu_set_t *saved_mask;
    cpu_set_t *scratch_mask;
};

static pthread_mutex_t g_pin_mtx = PTHREAD_MUTEX_INITIALIZER;
static cpu_set_t *g_reserved_cpus = nullptr;
static size_t g_reserved_ncpus = 0U;
static size_t g_reserved_bytes = 0U;
static size_t g_active_pin_plans = 0U;

static bool ensure_reserved_capacity(size_t ncpus) {
    if (g_reserved_cpus && g_reserved_ncpus >= ncpus) {
        return true;
    }
    cpu_set_t *next = CPU_ALLOC(ncpus);
    if (!next) {
        return false;
    }
    size_t next_bytes = CPU_ALLOC_SIZE(ncpus);
    CPU_ZERO_S(next_bytes, next);
    if (g_reserved_cpus) {
        memcpy(next, g_reserved_cpus, g_reserved_bytes);
        CPU_FREE(g_reserved_cpus);
    }
    g_reserved_cpus = next;
    g_reserved_ncpus = ncpus;
    g_reserved_bytes = next_bytes;
    return true;
}

static void free_pin_plan_storage(velri_os_pin_plan *plan) {
    if (!plan) {
        return;
    }
    if (plan->worker_mask) CPU_FREE(plan->worker_mask);
    if (plan->reserved_mask) CPU_FREE(plan->reserved_mask);
    if (plan->saved_mask) CPU_FREE(plan->saved_mask);
    if (plan->scratch_mask) CPU_FREE(plan->scratch_mask);
    free(plan->worker_cores);
    free(plan);
}
#endif

uint32_t velri_os_available_physical_cores(void) {
#if defined(__linux__)
    cpu_set_t *allowed = nullptr;
    size_t mask_bytes = 0U;
    size_t ncpus = query_process_affinity(&allowed, &mask_bytes);
    if (ncpus == 0U) {
        return 0U;
    }
    size_t count = collect_physical_cores(allowed, ncpus, mask_bytes, nullptr, 0U);
    CPU_FREE(allowed);
    return count <= UINT32_MAX ? (uint32_t)count : UINT32_MAX;
#elif defined(__APPLE__)
    uint32_t cores = 0U;
    size_t size = sizeof(cores);
    return sysctlbyname("hw.physicalcpu", &cores, &size, nullptr, 0U) == 0 ? cores : 0U;
#else
    return 0U;
#endif
}

velri_os_pin_plan *velri_os_pin_acquire(size_t num_workers) {
#if defined(__linux__)
    if (num_workers <= 1U) {
        return nullptr;
    }
    cpu_set_t *allowed = nullptr;
    size_t mask_bytes = 0U;
    size_t ncpus = query_process_affinity(&allowed, &mask_bytes);
    if (ncpus == 0U) {
        return nullptr;
    }
    int *cores = (int *)malloc(ncpus * sizeof(*cores));
    int *free_cores = (int *)malloc(ncpus * sizeof(*free_cores));
    cpu_set_t *siblings = CPU_ALLOC(ncpus);
    if (!cores || !free_cores || !siblings) {
        free(cores);
        free(free_cores);
        if (siblings) CPU_FREE(siblings);
        CPU_FREE(allowed);
        return nullptr;
    }

    velri_os_pin_plan *plan = nullptr;
    size_t n_cores = collect_physical_cores(allowed, ncpus, mask_bytes, cores, ncpus);
    if (n_cores >= num_workers && !is_heterogeneous_topology(cores, n_cores)) {
        pthread_mutex_lock(&g_pin_mtx);
        size_t n_free = 0U;
        for (size_t i = 0U; i < n_cores; ++i) {
            int c = cores[i];
            if (!g_reserved_cpus || (size_t)c >= g_reserved_ncpus ||
                !CPU_ISSET_S((size_t)c, g_reserved_bytes, g_reserved_cpus)) {
                free_cores[n_free++] = c;
            }
        }
        if (n_free >= num_workers && ensure_reserved_capacity(ncpus)) {
            int cur = sched_getcpu();
            if (cur >= 0 && read_thread_siblings(cur, ncpus, mask_bytes, siblings)) {
                for (size_t i = 1U; i < n_free; ++i) {
                    if (CPU_ISSET_S((size_t)free_cores[i], mask_bytes, siblings)) {
                        int tmp = free_cores[0];
                        free_cores[0] = free_cores[i];
                        free_cores[i] = tmp;
                        break;
                    }
                }
            }
            velri_os_pin_plan *p = (velri_os_pin_plan *)calloc(1U, sizeof(*p));
            if (p) {
                p->num_workers = num_workers;
                p->ncpus = ncpus;
                p->mask_bytes = mask_bytes;
                p->worker_cores = (int *)malloc(num_workers * sizeof(*p->worker_cores));
                p->worker_mask = CPU_ALLOC(ncpus);
                p->reserved_mask = CPU_ALLOC(ncpus);
                p->saved_mask = CPU_ALLOC(ncpus);
                p->scratch_mask = CPU_ALLOC(ncpus);
                bool ok = p->worker_cores && p->worker_mask && p->reserved_mask && p->saved_mask &&
                          p->scratch_mask;
                if (ok) {
                    CPU_ZERO_S(mask_bytes, p->worker_mask);
                    CPU_ZERO_S(mask_bytes, p->reserved_mask);
                    for (size_t i = 0U; i < num_workers; ++i) {
                        p->worker_cores[i] = free_cores[i];
                        if (!read_thread_siblings(free_cores[i], ncpus, mask_bytes, siblings)) {
                            ok = false;
                            break;
                        }
                        CPU_OR_S(mask_bytes, p->reserved_mask, p->reserved_mask, siblings);
                        if (i > 0U) {
                            CPU_OR_S(mask_bytes, p->worker_mask, p->worker_mask, siblings);
                        }
                    }
                }
                if (ok) {
                    for (size_t cpu = 0U; cpu < ncpus; ++cpu) {
                        if (CPU_ISSET_S(cpu, mask_bytes, p->reserved_mask)) {
                            CPU_SET_S(cpu, g_reserved_bytes, g_reserved_cpus);
                        }
                    }
                    ++g_active_pin_plans;
                    plan = p;
                } else {
                    free_pin_plan_storage(p);
                }
            }
        }
        pthread_mutex_unlock(&g_pin_mtx);
    }

    CPU_FREE(siblings);
    free(free_cores);
    free(cores);
    CPU_FREE(allowed);
    return plan;
#else
    (void)num_workers;
    return nullptr;
#endif
}

void velri_os_pin_release(velri_os_pin_plan *plan) {
#if defined(__linux__)
    if (!plan) {
        return;
    }
    pthread_mutex_lock(&g_pin_mtx);
    if (g_reserved_cpus) {
        size_t limit = plan->ncpus < g_reserved_ncpus ? plan->ncpus : g_reserved_ncpus;
        for (size_t cpu = 0U; cpu < limit; ++cpu) {
            if (CPU_ISSET_S(cpu, plan->mask_bytes, plan->reserved_mask)) {
                CPU_CLR_S(cpu, g_reserved_bytes, g_reserved_cpus);
            }
        }
    }
    if (g_active_pin_plans > 0U) {
        --g_active_pin_plans;
    }
    if (g_active_pin_plans == 0U && g_reserved_cpus) {
        CPU_FREE(g_reserved_cpus);
        g_reserved_cpus = nullptr;
        g_reserved_ncpus = 0U;
        g_reserved_bytes = 0U;
    }
    pthread_mutex_unlock(&g_pin_mtx);
    free_pin_plan_storage(plan);
#else
    (void)plan;
#endif
}

void velri_os_pin_worker(const velri_os_pin_plan *plan, size_t worker_id) {
#if defined(__linux__)
    if (!plan || worker_id == 0U || worker_id >= plan->num_workers) {
        return;
    }
    int cpu = plan->worker_cores[worker_id];
    if (cpu < 0 || (size_t)cpu >= plan->ncpus) {
        return;
    }
    cpu_set_t *target = CPU_ALLOC(plan->ncpus);
    if (!target) {
        return;
    }
    CPU_ZERO_S(plan->mask_bytes, target);
    CPU_SET_S((size_t)cpu, plan->mask_bytes, target);
    (void)sched_setaffinity(0, plan->mask_bytes, target);
    CPU_FREE(target);
#else
    (void)plan;
    (void)worker_id;
#endif
}

void velri_os_caller_guard_enter(velri_os_pin_plan *plan, velri_os_affinity_guard *guard) {
    guard->plan = nullptr;
    guard->active = false;
#if defined(__linux__)
    if (!plan) {
        return;
    }
    if (sched_getaffinity(0, plan->mask_bytes, plan->saved_mask) != 0) {
        return;
    }
    /* Compute overlap = saved_mask & worker_mask; skip syscalls if caller is already disjoint. */
    CPU_AND_S(plan->mask_bytes, plan->scratch_mask, plan->saved_mask, plan->worker_mask);
    if (CPU_COUNT_S(plan->mask_bytes, plan->scratch_mask) == 0) {
        return;
    }
    /* target = saved_mask & ~worker_mask = saved_mask ^ (saved_mask & worker_mask). */
    CPU_XOR_S(plan->mask_bytes, plan->scratch_mask, plan->saved_mask, plan->scratch_mask);
    if (CPU_COUNT_S(plan->mask_bytes, plan->scratch_mask) == 0) {
        return;
    }
    if (sched_setaffinity(0, plan->mask_bytes, plan->scratch_mask) == 0) {
        guard->plan = plan;
        guard->active = true;
    }
#else
    (void)plan;
#endif
}

void velri_os_caller_guard_leave(const velri_os_affinity_guard *guard) {
#if defined(__linux__)
    if (!guard || !guard->active || !guard->plan) {
        return;
    }
    (void)sched_setaffinity(0, guard->plan->mask_bytes, guard->plan->saved_mask);
#else
    (void)guard;
#endif
}

void velri_os_cpu_relax(void) {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__) && __has_builtin(__builtin_arm_yield)
    __builtin_arm_yield();
#elif defined(__aarch64__) && __has_builtin(__builtin_aarch64_yield)
    __builtin_aarch64_yield();
#endif
}

size_t velri_os_ctz64(uint64_t x) { return x ? (size_t)__builtin_ctzll(x) : 64U; }
