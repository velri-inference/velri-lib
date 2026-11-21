#ifndef VELRI_PLATFORM_POOL_H
#define VELRI_PLATFORM_POOL_H

#include "velri_lib/types.h"

/*
 * Internal persistent low-latency worker pool owned by the CPU backend.
 * Caller thread participates as worker 0; background workers spin briefly for
 * back-to-back dispatches and park on condition variables when idle.
 */
typedef struct velri_pool velri_pool;

typedef void (*velri_parallel_fn)(void *ctx, size_t worker_id, size_t num_workers);

velri_status velri_pool_create(size_t num_workers, velri_pool **out_pool);
void velri_pool_destroy(velri_pool *pool);

/* Dispatch `fn` synchronously across `active_workers` (clamped to [1, pool->num_workers]). */
void velri_pool_parallel(velri_pool *pool, size_t active_workers, velri_parallel_fn fn, void *ctx);

#endif /* VELRI_PLATFORM_POOL_H */
