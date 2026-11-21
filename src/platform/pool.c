#include "os.h"
#include "pool.h"
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

/*
 * Persistent low-latency worker pool (100% OS-agnostic C23; all OS primitives
 * live in `os.h`).
 */

typedef struct {
    alignas(64) _Atomic uint64_t done_gen;
    size_t worker_id;
    velri_pool *pool;
    velri_os_thread *thread;
    velri_os_mutex *mtx;
    velri_os_cond *cv;
} velri_worker_slot;

struct velri_pool {
    size_t num_workers;
    atomic_bool stopping;
    velri_parallel_fn fn;
    void *fn_ctx;
    size_t last_nw;
    uint64_t host_gen;
    alignas(64) _Atomic uint64_t dispatch_word; /* (gen << 16) | (nw & 0xFFFFU) */
    alignas(64) _Atomic uint64_t parked_mask[4];
    velri_worker_slot *slots;
};

static void signal_worker_slot(velri_worker_slot *slot) {
    if (slot->mtx && slot->cv) {
        velri_os_mutex_lock(slot->mtx);
        velri_os_cond_signal(slot->cv);
        velri_os_mutex_unlock(slot->mtx);
    }
}

static void worker_loop(void *arg) {
    velri_worker_slot *slot = (velri_worker_slot *)arg;
    velri_pool *pool = slot->pool;
    const size_t wid = slot->worker_id;
    const size_t mask_idx = wid >> 6;
    const uint64_t mask_bit = UINT64_C(1) << (wid & 63U);

    atomic_store_explicit(&slot->done_gen, UINT64_C(1), memory_order_release);

    uint64_t g = UINT64_C(1);
    for (;;) {
        uint64_t word = atomic_load_explicit(&pool->dispatch_word, memory_order_acquire);
        uint64_t cg = word >> 16;
        size_t cnw = (size_t)(word & UINT64_C(0xFFFF));

        if (cg == g && wid < cnw) {
            const uint64_t deadline = velri_os_monotonic_ns() + UINT64_C(250000000);
            do {
                for (int spin = 0; spin < 2048; ++spin) {
                    velri_os_cpu_relax();
                    word = atomic_load_explicit(&pool->dispatch_word, memory_order_acquire);
                    cg = word >> 16;
                    if (cg != g) {
                        break;
                    }
                }
                cnw = (size_t)(word & UINT64_C(0xFFFF));
                if (cg != g || wid >= cnw ||
                    atomic_load_explicit(&pool->stopping, memory_order_relaxed)) {
                    break;
                }
            } while (velri_os_monotonic_ns() < deadline);
        }

        if (cg == g || wid >= cnw) {
            g = cg;
            velri_os_mutex_lock(slot->mtx);
            atomic_fetch_or_explicit(&pool->parked_mask[mask_idx], mask_bit, memory_order_seq_cst);
            while (!atomic_load_explicit(&pool->stopping, memory_order_acquire)) {
                word = atomic_load_explicit(&pool->dispatch_word, memory_order_seq_cst);
                cg = word >> 16;
                cnw = (size_t)(word & UINT64_C(0xFFFF));
                if (cg != g && wid < cnw) {
                    break;
                }
                g = cg;
                velri_os_cond_wait(slot->cv, slot->mtx);
            }
            atomic_fetch_and_explicit(&pool->parked_mask[mask_idx], ~mask_bit,
                                      memory_order_relaxed);
            velri_os_mutex_unlock(slot->mtx);
        }

        if (atomic_load_explicit(&pool->stopping, memory_order_acquire)) {
            return;
        }

        pool->fn(pool->fn_ctx, wid, cnw);

        atomic_store_explicit(&slot->done_gen, cg, memory_order_release);
        g = cg;
    }
}

static void free_pool_slots(velri_pool *pool, size_t count) {
    for (size_t j = 0; j < count; ++j) {
        velri_os_cond_destroy(pool->slots[j].cv);
        velri_os_mutex_destroy(pool->slots[j].mtx);
    }
    velri_os_aligned_free(pool->slots);
    velri_os_aligned_free(pool);
}

velri_status velri_pool_create(size_t num_workers, velri_pool **out_pool) {
    if (!out_pool || num_workers == 0 || num_workers > 256) {
        return VELRI_ERR_INVALID;
    }
    *out_pool = nullptr;

    velri_pool *pool = (velri_pool *)velri_os_aligned_alloc(64, sizeof(velri_pool));
    if (!pool) {
        return VELRI_ERR_NOMEM;
    }
    memset(pool, 0, sizeof(*pool));
    pool->num_workers = num_workers;
    pool->last_nw = 1ULL;
    pool->host_gen = UINT64_C(1);
    atomic_init(&pool->stopping, false);
    atomic_init(&pool->dispatch_word, (UINT64_C(1) << 16) | UINT64_C(1));
    for (size_t m = 0; m < 4; ++m) {
        atomic_init(&pool->parked_mask[m], UINT64_C(0));
    }

    pool->slots = (velri_worker_slot *)velri_os_aligned_alloc(
        64, num_workers * sizeof(velri_worker_slot));
    if (!pool->slots) {
        velri_os_aligned_free(pool);
        return VELRI_ERR_NOMEM;
    }
    memset(pool->slots, 0, num_workers * sizeof(velri_worker_slot));

    for (size_t i = 0; i < num_workers; ++i) {
        pool->slots[i].worker_id = i;
        pool->slots[i].pool = pool;
        atomic_init(&pool->slots[i].done_gen, UINT64_C(0));
        if (i > 0) {
            if (!velri_os_mutex_create(&pool->slots[i].mtx) ||
                !velri_os_cond_create(&pool->slots[i].cv)) {
                free_pool_slots(pool, i + 1);
                return VELRI_ERR_NOMEM;
            }
        }
    }

    atomic_store_explicit(&pool->slots[0].done_gen, UINT64_C(1), memory_order_release);

    for (size_t i = 1; i < num_workers; ++i) {
        if (!velri_os_thread_spawn(&pool->slots[i].thread, worker_loop, &pool->slots[i])) {
            atomic_store_explicit(&pool->stopping, true, memory_order_seq_cst);
            uint64_t next_gen = ++pool->host_gen;
            atomic_store_explicit(&pool->dispatch_word, (next_gen << 16) | (uint64_t)num_workers,
                                  memory_order_seq_cst);
            for (size_t j = 1; j < i; ++j) {
                signal_worker_slot(&pool->slots[j]);
            }
            for (size_t j = 1; j < i; ++j) {
                velri_os_thread_join(pool->slots[j].thread);
            }
            free_pool_slots(pool, num_workers);
            return VELRI_ERR_NOMEM;
        }
    }

    for (size_t i = 1; i < num_workers; ++i) {
        while (atomic_load_explicit(&pool->slots[i].done_gen, memory_order_acquire) !=
               UINT64_C(1)) {
            velri_os_cpu_relax();
        }
    }

    *out_pool = pool;
    return VELRI_OK;
}

void velri_pool_destroy(velri_pool *pool) {
    if (!pool) {
        return;
    }
    if (pool->num_workers > 1) {
        atomic_store_explicit(&pool->stopping, true, memory_order_seq_cst);
        uint64_t next_gen = ++pool->host_gen;
        atomic_store_explicit(&pool->dispatch_word, (next_gen << 16) | (uint64_t)pool->num_workers,
                              memory_order_seq_cst);
        for (size_t i = 1; i < pool->num_workers; ++i) {
            signal_worker_slot(&pool->slots[i]);
        }
        for (size_t i = 1; i < pool->num_workers; ++i) {
            velri_os_thread_join(pool->slots[i].thread);
        }
    }
    free_pool_slots(pool, pool->num_workers);
}

void velri_pool_parallel(velri_pool *pool, size_t active_workers, velri_parallel_fn fn, void *ctx) {
    if (!fn) {
        return;
    }
    if (!pool || active_workers <= 1 || pool->num_workers == 1) {
        if (pool && pool->last_nw > 1) {
            pool->last_nw = 1ULL;
            uint64_t next_gen = ++pool->host_gen;
            atomic_store_explicit(&pool->dispatch_word, (next_gen << 16) | UINT64_C(1),
                                  memory_order_release);
        }
        fn(ctx, 0, 1);
        return;
    }

    size_t nw = active_workers < pool->num_workers ? active_workers : pool->num_workers;
    pool->last_nw = nw;
    pool->fn = fn;
    pool->fn_ctx = ctx;

    uint64_t next_gen = ++pool->host_gen;
    atomic_store_explicit(&pool->dispatch_word, (next_gen << 16) | (uint64_t)nw,
                          memory_order_release);
    atomic_thread_fence(memory_order_seq_cst);

    size_t n_words = (nw + 63ULL) >> 6;
    for (size_t m = 0; m < n_words; ++m) {
        uint64_t valid = ~UINT64_C(0);
        if (m == 0) {
            valid &= ~UINT64_C(1); /* slot 0 is caller */
        }
        if ((m + 1ULL) * 64ULL > nw) {
            size_t rem = nw - m * 64ULL;
            valid &= (UINT64_C(1) << rem) - UINT64_C(1);
        }
        uint64_t pm = atomic_load_explicit(&pool->parked_mask[m], memory_order_relaxed) & valid;
        while (pm != 0) {
            size_t bit = velri_os_ctz64(pm);
            pm &= pm - UINT64_C(1);
            signal_worker_slot(&pool->slots[m * 64ULL + bit]);
        }
    }

    fn(ctx, 0, nw);

    for (size_t i = 1; i < nw; ++i) {
        while (atomic_load_explicit(&pool->slots[i].done_gen, memory_order_acquire) != next_gen) {
            velri_os_cpu_relax();
        }
    }
}
