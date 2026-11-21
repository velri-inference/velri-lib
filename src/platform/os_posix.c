#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 700
#endif

#include "os.h"

#include <dlfcn.h>
#include <pthread.h>
#include <stdlib.h>
#include <time.h>

#if __has_include(<unistd.h>)
#include <unistd.h>
#endif

/* Ensure worker threads have enough stack for L1-packed BLIS panels even on
 * small-default-stack libc implementations (e.g. musl 128 KiB). */
static const size_t kMinWorkerStackBytes = 512U * 1024U;

/* ---- Aligned Memory ---- */

void *velri_os_aligned_alloc(size_t align, size_t bytes) {
    size_t eff_align = align < sizeof(void *) ? sizeof(void *) : align;
    if ((eff_align & (eff_align - 1U)) != 0U) {
        return nullptr;
    }
    if (bytes > SIZE_MAX - (eff_align - 1U)) {
        return nullptr;
    }
    size_t sz = bytes ? ((bytes + eff_align - 1U) & ~(eff_align - 1U)) : eff_align;
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

velri_os_lib *velri_os_dlopen_vulkan(void) {
    static const char *const candidates[] = {
#if defined(__APPLE__)
        "libvulkan.1.dylib",
        "libvulkan.dylib",
        "libMoltenVK.dylib",
#else
        "libvulkan.so.1",
        "libvulkan.so",
#endif
        nullptr,
    };
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

void velri_os_cpu_relax(void) {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ volatile("yield");
#endif
}

size_t velri_os_ctz64(uint64_t x) { return x ? (size_t)__builtin_ctzll(x) : 64U; }
