#ifndef VELRI_PLATFORM_OS_H
#define VELRI_PLATFORM_OS_H

#include <stddef.h>
#include <stdint.h>

/*
 * Single OS abstraction boundary for Velri-lib.
 *
 * All OS-specific headers (<pthread.h>, <windows.h>, <dlfcn.h>, <unistd.h>,
 * <time.h>) live strictly in `os_posix.c` and `os_win32.c`.
 */

/* Shared alignment/size validation helper for `velri_os_aligned_alloc` implementations. */
static inline bool velri_os_aligned_size(size_t align, size_t bytes, size_t *out_align,
                                         size_t *out_sz) {
    size_t eff = (align < sizeof(void *)) ? sizeof(void *) : align;
    if ((eff & (eff - 1U)) != 0U) {
        return false;
    }
    size_t req = bytes ? bytes : 1U;
    if (req > SIZE_MAX - (eff - 1U)) {
        return false;
    }
    *out_align = eff;
    *out_sz = (req + (eff - 1U)) & ~(eff - 1U);
    return true;
}

/* ---- Aligned Memory ---- */
void *velri_os_aligned_alloc(size_t align, size_t bytes);
void velri_os_aligned_free(void *ptr);

/* ---- Monotonic Time & Sleep ---- */
uint64_t velri_os_monotonic_ns(void);
void velri_os_sleep_ms(uint32_t ms);

/* ---- Dynamic Shared Library Loading ---- */
typedef struct velri_os_lib velri_os_lib;
/* Try loading each library name in a nullptr-terminated candidate list in order. */
velri_os_lib *velri_os_dlopen(const char *const *candidates);
void *velri_os_dlsym(velri_os_lib *lib, const char *symbol);
void velri_os_dlclose(velri_os_lib *lib);

/* ---- Threads, Mutex, Condition Variable ---- */
typedef struct velri_os_thread velri_os_thread;
typedef struct velri_os_mutex velri_os_mutex;
typedef struct velri_os_cond velri_os_cond;

typedef void (*velri_os_thread_fn)(void *arg);

bool velri_os_thread_spawn(velri_os_thread **out_thread, velri_os_thread_fn fn, void *arg);
void velri_os_thread_join(velri_os_thread *thread);

bool velri_os_mutex_create(velri_os_mutex **out_mtx);
void velri_os_mutex_destroy(velri_os_mutex *mtx);
void velri_os_mutex_lock(velri_os_mutex *mtx);
void velri_os_mutex_unlock(velri_os_mutex *mtx);

bool velri_os_cond_create(velri_os_cond **out_cv);
void velri_os_cond_destroy(velri_os_cond *cv);
void velri_os_cond_wait(velri_os_cond *cv, velri_os_mutex *mtx);
void velri_os_cond_signal(velri_os_cond *cv);
void velri_os_cond_broadcast(velri_os_cond *cv);

/* ---- CPU Hints ---- */
int velri_os_hardware_threads(void);
/* Count accessible physical cores. Zero means the platform could not identify them. */
uint32_t velri_os_available_physical_cores(void);

/* Process-coordinated 1-to-1 physical-core pinning plan for a worker pool.
 * Returns nullptr when pinning is unsupported, topology is heterogeneous/untrusted,
 * or concurrent pools oversubscribe available physical cores. */
typedef struct velri_os_pin_plan velri_os_pin_plan;
velri_os_pin_plan *velri_os_pin_acquire(size_t num_workers);
void velri_os_pin_release(velri_os_pin_plan *plan);
/* Pin worker thread `worker_id` (1..num_workers-1) to its reserved physical core. */
void velri_os_pin_worker(const velri_os_pin_plan *plan, size_t worker_id);
/* Temporarily exclude all pinned worker cores (and their SMT siblings) from the
 * caller thread's affinity mask during a parallel section, restoring the caller's
 * exact prior mask on leave (zero heap allocations on the hot path). */
typedef struct {
    velri_os_pin_plan *plan;
    bool active;
} velri_os_affinity_guard;
void velri_os_caller_guard_enter(velri_os_pin_plan *plan, velri_os_affinity_guard *guard);
void velri_os_caller_guard_leave(const velri_os_affinity_guard *guard);
void velri_os_cpu_relax(void);
size_t velri_os_ctz64(uint64_t x);

#endif /* VELRI_PLATFORM_OS_H */
