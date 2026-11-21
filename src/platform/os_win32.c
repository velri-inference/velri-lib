#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0601
#endif
#define WIN32_LEAN_AND_MEAN

#include "os.h"

#include <malloc.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

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
    return _aligned_malloc(sz, eff_align);
}

void velri_os_aligned_free(void *ptr) { _aligned_free(ptr); }

/* ---- Monotonic Time & Sleep ---- */

uint64_t velri_os_monotonic_ns(void) {
    static _Atomic uint64_t g_qpf = 0;
    uint64_t freq = atomic_load_explicit(&g_qpf, memory_order_relaxed);
    if (freq == 0U) {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        freq = (f.QuadPart > 0) ? (uint64_t)f.QuadPart : UINT64_C(1);
        atomic_store_explicit(&g_qpf, freq, memory_order_relaxed);
    }
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    uint64_t q = (uint64_t)c.QuadPart;
    return (q / freq) * UINT64_C(1000000000) + ((q % freq) * UINT64_C(1000000000)) / freq;
}

void velri_os_sleep_ms(uint32_t ms) {
    if (ms > 0U) {
        Sleep(ms);
    }
}

/* ---- Dynamic Shared Library Loading ---- */

velri_os_lib *velri_os_dlopen_vulkan(void) {
    return (velri_os_lib *)LoadLibraryA("vulkan-1.dll");
}

void *velri_os_dlsym(velri_os_lib *lib, const char *symbol) {
    if (!lib || !symbol) {
        return nullptr;
    }
    FARPROC p = GetProcAddress((HMODULE)lib, symbol);
    void *out = nullptr;
    memcpy(&out, &p, sizeof(out));
    return out;
}

void velri_os_dlclose(velri_os_lib *lib) {
    if (lib) {
        FreeLibrary((HMODULE)lib);
    }
}

/* ---- Threads, Mutex, Condition Variable ---- */

struct velri_os_thread {
    HANDLE handle;
    velri_os_thread_fn fn;
    void *arg;
};

struct velri_os_mutex {
    CRITICAL_SECTION handle;
};

struct velri_os_cond {
    CONDITION_VARIABLE handle;
};

static DWORD WINAPI win32_thread_trampoline(LPVOID raw) {
    velri_os_thread *t = (velri_os_thread *)raw;
    t->fn(t->arg);
    return 0;
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
    t->handle = CreateThread(nullptr, 0, win32_thread_trampoline, t, 0, nullptr);
    if (!t->handle) {
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
    WaitForSingleObject(thread->handle, INFINITE);
    CloseHandle(thread->handle);
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
    InitializeCriticalSection(&m->handle);
    *out_mtx = m;
    return true;
}

void velri_os_mutex_destroy(velri_os_mutex *mtx) {
    if (!mtx) {
        return;
    }
    DeleteCriticalSection(&mtx->handle);
    free(mtx);
}

void velri_os_mutex_lock(velri_os_mutex *mtx) { EnterCriticalSection(&mtx->handle); }

void velri_os_mutex_unlock(velri_os_mutex *mtx) { LeaveCriticalSection(&mtx->handle); }

bool velri_os_cond_create(velri_os_cond **out_cv) {
    if (!out_cv) {
        return false;
    }
    *out_cv = nullptr;
    velri_os_cond *c = (velri_os_cond *)calloc(1U, sizeof(*c));
    if (!c) {
        return false;
    }
    InitializeConditionVariable(&c->handle);
    *out_cv = c;
    return true;
}

void velri_os_cond_destroy(velri_os_cond *cv) { free(cv); }

void velri_os_cond_wait(velri_os_cond *cv, velri_os_mutex *mtx) {
    SleepConditionVariableCS(&cv->handle, &mtx->handle, INFINITE);
}

void velri_os_cond_signal(velri_os_cond *cv) { WakeConditionVariable(&cv->handle); }

/* ---- CPU Hints ---- */

int velri_os_hardware_threads(void) {
    DWORD nproc = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    if (nproc == 0) {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        nproc = si.dwNumberOfProcessors;
    }
    return (nproc > 0 && nproc < 65536) ? (int)nproc : 1;
}

void velri_os_cpu_relax(void) { YieldProcessor(); }

size_t velri_os_ctz64(uint64_t x) {
    if (x == 0U) {
        return 64U;
    }
#if defined(_MSC_VER)
    unsigned long idx = 0;
#if defined(_WIN64)
    _BitScanForward64(&idx, x);
    return (size_t)idx;
#else
    uint32_t lo = (uint32_t)x;
    if (lo != 0U) {
        _BitScanForward(&idx, lo);
        return (size_t)idx;
    }
    _BitScanForward(&idx, (uint32_t)(x >> 32));
    return 32U + (size_t)idx;
#endif
#else
    return (size_t)__builtin_ctzll(x);
#endif
}
