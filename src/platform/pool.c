#include "os.h"
#include "pool.h"
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

/*
 * Persistent low-latency worker pool (100% OS-agnostic C23; all OS primitives
 * live in `os.h`).
 */

enum {
    VELRI_POOL_MASK_WORDS = VELRI_POOL_MAX_WORKERS / 64
};

static_assert(VELRI_POOL_MAX_WORKERS > 0 && (VELRI_POOL_MAX_WORKERS % 64) == 0 &&
                  VELRI_POOL_MAX_WORKERS <= 65535,
              "VELRI_POOL_MAX_WORKERS must be a positive multiple of 64 fitting in 16 bits");

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
    velri_os_pin_plan *pin_plan;
    atomic_bool stopping;
    size_t last_nw;
    uint64_t host_gen;
    velri_worker_slot *slots;
    /* Co-locate dispatch_word + fn + fn_ctx on one 64B cache line so waking workers
     * fetch all three in a single cross-CCD cache transfer instead of two. */
    alignas(64) _Atomic uint64_t dispatch_word; /* (gen << 16) | (nw & 0xFFFFU) */
    velri_parallel_fn fn;
    void *fn_ctx;
    alignas(64) _Atomic uint64_t parked_mask[VELRI_POOL_MASK_WORDS];
};

static inline uint64_t pack_dispatch_word(uint64_t gen, size_t nw) {
    return (gen << 16) | ((uint64_t)nw & UINT64_C(0xFFFF));
}

static inline void unpack_dispatch_word(uint64_t word, uint64_t *out_gen, size_t *out_nw) {
    *out_gen = word >> 16;
    *out_nw = (size_t)(word & UINT64_C(0xFFFF));
}

static void signal_worker_slot(velri_worker_slot *slot) {
    if (slot->mtx && slot->cv) {
        velri_os_mutex_lock(slot->mtx);
        velri_os_cond_signal(slot->cv);
        velri_os_mutex_unlock(slot->mtx);
    }
}

static void stop_and_join_workers(velri_pool *pool, size_t spawned_count) {
    if (spawned_count <= 1U) {
        return;
    }
    atomic_store_explicit(&pool->stopping, true, memory_order_seq_cst);
    uint64_t next_gen = ++pool->host_gen;
    atomic_store_explicit(&pool->dispatch_word, pack_dispatch_word(next_gen, pool->num_workers),
                          memory_order_seq_cst);
    for (size_t j = 1U; j < spawned_count; ++j) {
        signal_worker_slot(&pool->slots[j]);
    }
    for (size_t j = 1U; j < spawned_count; ++j) {
        velri_os_thread_join(pool->slots[j].thread);
    }
}

static void worker_loop(void *arg) {
    velri_worker_slot *slot = (velri_worker_slot *)arg;
    velri_pool *pool = slot->pool;
    const size_t wid = slot->worker_id;
    const size_t mask_idx = wid >> 6;
    const uint64_t mask_bit = UINT64_C(1) << (wid & 63U);

    velri_os_pin_worker(pool->pin_plan, wid);
    atomic_store_explicit(&slot->done_gen, UINT64_C(1), memory_order_release);

    uint64_t g = UINT64_C(1);
    for (;;) {
        uint64_t word = atomic_load_explicit(&pool->dispatch_word, memory_order_acquire);
        uint64_t cg = 0U;
        size_t cnw = 0U;
        unpack_dispatch_word(word, &cg, &cnw);

        if (cg == g && wid < cnw) {
            /* Fast spin before calling monotonic_ns() to avoid vDSO overhead on back-to-back ops. */
            for (int spin = 0; spin < 1024; ++spin) {
                velri_os_cpu_relax();
                word = atomic_load_explicit(&pool->dispatch_word, memory_order_acquire);
                unpack_dispatch_word(word, &cg, &cnw);
                if (cg != g) {
                    break;
                }
            }
            if (cg == g && wid < cnw &&
                !atomic_load_explicit(&pool->stopping, memory_order_relaxed)) {
                const uint64_t deadline = velri_os_monotonic_ns() + UINT64_C(250000000);
                do {
                    for (int spin = 0; spin < 2048; ++spin) {
                        velri_os_cpu_relax();
                        word = atomic_load_explicit(&pool->dispatch_word, memory_order_acquire);
                        unpack_dispatch_word(word, &cg, &cnw);
                        if (cg != g) {
                            break;
                        }
                    }
                    if (cg != g || wid >= cnw ||
                        atomic_load_explicit(&pool->stopping, memory_order_relaxed)) {
                        break;
                    }
                } while (velri_os_monotonic_ns() < deadline);
            }
        }

        if (cg == g || wid >= cnw) {
            g = cg;
            velri_os_mutex_lock(slot->mtx);
            atomic_fetch_or_explicit(&pool->parked_mask[mask_idx], mask_bit, memory_order_seq_cst);
            while (!atomic_load_explicit(&pool->stopping, memory_order_acquire)) {
                word = atomic_load_explicit(&pool->dispatch_word, memory_order_seq_cst);
                unpack_dispatch_word(word, &cg, &cnw);
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
    for (size_t j = 0U; j < count; ++j) {
        velri_os_cond_destroy(pool->slots[j].cv);
        velri_os_mutex_destroy(pool->slots[j].mtx);
    }
    velri_os_pin_release(pool->pin_plan);
    velri_os_aligned_free(pool->slots);
    velri_os_aligned_free(pool);
}

velri_status velri_pool_create(size_t num_workers, velri_pool **out_pool) {
    if (!out_pool || num_workers == 0U || num_workers > VELRI_POOL_MAX_WORKERS) {
        return VELRI_ERR_INVALID;
    }
    *out_pool = nullptr;

    velri_pool *pool = (velri_pool *)velri_os_aligned_alloc(64U, sizeof(velri_pool));
    if (!pool) {
        return VELRI_ERR_NOMEM;
    }
    memset(pool, 0, sizeof(*pool));
    pool->num_workers = num_workers;
    pool->last_nw = 1U;
    pool->host_gen = UINT64_C(1);
    atomic_init(&pool->stopping, false);
    atomic_init(&pool->dispatch_word, pack_dispatch_word(UINT64_C(1), 1U));
    for (size_t m = 0U; m < VELRI_POOL_MASK_WORDS; ++m) {
        atomic_init(&pool->parked_mask[m], UINT64_C(0));
    }

    pool->slots = (velri_worker_slot *)velri_os_aligned_alloc(
        64U, num_workers * sizeof(velri_worker_slot));
    if (!pool->slots) {
        velri_os_aligned_free(pool);
        return VELRI_ERR_NOMEM;
    }
    memset(pool->slots, 0, num_workers * sizeof(velri_worker_slot));

    pool->pin_plan = (num_workers > 1U) ? velri_os_pin_acquire(num_workers) : nullptr;
    for (size_t i = 0U; i < num_workers; ++i) {
        pool->slots[i].worker_id = i;
        pool->slots[i].pool = pool;
        atomic_init(&pool->slots[i].done_gen, UINT64_C(0));
        if (i > 0U) {
            if (!velri_os_mutex_create(&pool->slots[i].mtx) ||
                !velri_os_cond_create(&pool->slots[i].cv)) {
                free_pool_slots(pool, i + 1U);
                return VELRI_ERR_NOMEM;
            }
        }
    }

    atomic_store_explicit(&pool->slots[0].done_gen, UINT64_C(1), memory_order_release);

    for (size_t i = 1U; i < num_workers; ++i) {
        if (!velri_os_thread_spawn(&pool->slots[i].thread, worker_loop, &pool->slots[i])) {
            stop_and_join_workers(pool, i);
            free_pool_slots(pool, num_workers);
            return VELRI_ERR_NOMEM;
        }
    }

    for (size_t i = 1U; i < num_workers; ++i) {
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
    stop_and_join_workers(pool, pool->num_workers);
    free_pool_slots(pool, pool->num_workers);
}

void velri_pool_parallel(velri_pool *pool, size_t active_workers, velri_parallel_fn fn, void *ctx) {
    if (!fn) {
        return;
    }
    if (!pool || active_workers <= 1U || pool->num_workers == 1U) {
        if (pool && pool->last_nw > 1U) {
            pool->last_nw = 1U;
            uint64_t next_gen = ++pool->host_gen;
            atomic_store_explicit(&pool->dispatch_word, pack_dispatch_word(next_gen, 1U),
                                  memory_order_release);
        }
        fn(ctx, 0U, 1U);
        return;
    }

    velri_os_affinity_guard pin_guard;
    velri_os_caller_guard_enter(pool->pin_plan, &pin_guard);

    size_t nw = active_workers < pool->num_workers ? active_workers : pool->num_workers;
    pool->last_nw = nw;
    pool->fn = fn;
    pool->fn_ctx = ctx;

    uint64_t next_gen = ++pool->host_gen;
    atomic_store_explicit(&pool->dispatch_word, pack_dispatch_word(next_gen, nw),
                          memory_order_release);
    atomic_thread_fence(memory_order_seq_cst);

    size_t n_words = (nw + 63U) >> 6;
    for (size_t m = 0U; m < n_words; ++m) {
        uint64_t valid = ~UINT64_C(0);
        if (m == 0U) {
            valid &= ~UINT64_C(1); /* slot 0 is caller */
        }
        size_t base = m * 64U;
        if (nw > base && nw - base < 64U) {
            size_t rem = nw - base;
            valid &= (UINT64_C(1) << rem) - UINT64_C(1);
        }
        uint64_t pm = atomic_load_explicit(&pool->parked_mask[m], memory_order_relaxed) & valid;
        while (pm != 0U) {
            size_t bit = velri_os_ctz64(pm);
            pm &= pm - UINT64_C(1);
            signal_worker_slot(&pool->slots[m * 64U + bit]);
        }
    }

    fn(ctx, 0U, nw);

    for (size_t i = 1U; i < nw; ++i) {
        while (atomic_load_explicit(&pool->slots[i].done_gen, memory_order_acquire) != next_gen) {
            velri_os_cpu_relax();
        }
    }

    velri_os_caller_guard_leave(&pin_guard);
}
