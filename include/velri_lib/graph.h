#ifndef VELRI_LIB_GRAPH_H
#define VELRI_LIB_GRAPH_H

#include "velri_lib/backend.h"

/*
 * Device-agnostic Operation Graph, Memory/Placement Planner, Prepared Launch Plan,
 * and Independent Execution Slots (`velri-lib/docs/EN/architecture.md §4` & `execution.md`).
 *
 * Ownership & Lifecycle:
 *   1. Build a `velri_graph` (storages, zero-copy views, op nodes).
 *   2. `velri_plan_prepare(ctx, graph, &opts, &plan)` validates everything, checks backend
 *      capabilities (no silent CPU fallback), caches cross-device transfers, packs scratch
 *      arenas by liveness, allocates shared `VELRI_STORAGE_WEIGHT` device buffers, and
 *      computes an exact memory report.
 *   3. Bind immutable weights ONCE on the plan (`velri_plan_bind_weight_host`). All
 *      `velri_slot`s created from the plan share weights read-only with zero duplication.
 *   4. Create independent `velri_slot`s, bind per-slot INPUT/OUTPUT/STATE buffers via
 *      `velri_slot_bind_host`, and call `velri_slot_run` (0 B hot-path allocations).
 *      `VELRI_STORAGE_STATE` remains resident on its device across steps; use
 *      `velri_slot_sync_state_from_host` / `velri_slot_sync_state_to_host` explicitly.
 */

typedef struct velri_graph velri_graph;
typedef struct velri_plan velri_plan;
typedef struct velri_slot velri_slot;

/* ---- Operation Graph Builder ---- */

velri_status velri_graph_create(velri_graph **out_graph);
void velri_graph_destroy(velri_graph *graph);

velri_status velri_graph_add_storage(velri_graph *graph, const velri_storage_desc *desc,
                                     velri_storage_id *out_id);
velri_status velri_graph_add_view(velri_graph *graph, const velri_tensor_view *view,
                                  velri_view_id *out_id);
velri_status velri_graph_add_op(velri_graph *graph, const velri_op_desc *op,
                                velri_device_id device_id, velri_node_id *out_node_id);

uint32_t velri_graph_storage_count(const velri_graph *graph);
uint32_t velri_graph_view_count(const velri_graph *graph);
uint32_t velri_graph_node_count(const velri_graph *graph);

/* ---- Plan Preparation & Memory Budget Report ---- */

enum { VELRI_MAX_REPORT_DEVICES = 16 };

typedef struct {
    velri_device_id device_id;
    bool is_uma;
    size_t weight_bytes;           /* Shared once per plan across all slots */
    size_t input_output_bytes;     /* Per-slot external input/output storage */
    size_t state_bytes;            /* Per-slot persistent state (KV, GDN, conv) */
    size_t scratch_unpacked_bytes; /* Sum of declared scratch storages on this device */
    size_t scratch_packed_bytes;   /* Actual liveness-packed scratch arena per slot */
    size_t allocator_overhead_bytes;
} velri_device_memory_report;

typedef struct {
    uint32_t num_devices;
    velri_device_memory_report devices[VELRI_MAX_REPORT_DEVICES];
    size_t ram_staging_bytes;   /* Host RAM staging buffer budget for cross-backend transfers */
    uint32_t num_ram_transfers; /* Count of explicit cross-device RAM transfers in launch plan */
    size_t total_physical_bytes;
} velri_memory_report;

typedef struct {
    bool allow_cpu_fallback; /* Default false: unsupported op on chosen device returns UNSUPPORTED */
} velri_plan_options;

velri_status velri_plan_prepare(velri_context *ctx, const velri_graph *graph,
                                const velri_plan_options *opts, velri_plan **out_plan);
void velri_plan_destroy(velri_plan *plan);

/* Bind immutable host weight data ONCE on the plan (shared read-only across all slots). */
velri_status velri_plan_bind_weight_host(velri_plan *plan, velri_storage_id storage_id,
                                         const void *host_ptr, size_t byte_size);

velri_status velri_plan_memory_report(const velri_plan *plan, velri_memory_report *out_report);
uint32_t velri_plan_launch_step_count(const velri_plan *plan);

/* ---- Execution Slots (Independent, Reusable, 0 B Hot-Path Allocation) ---- */

velri_status velri_slot_create(velri_plan *plan, velri_slot **out_slot);
void velri_slot_destroy(velri_slot *slot);

/* Bind host buffer to per-slot INPUT, OUTPUT, or STATE storage (use plan API for WEIGHT). */
velri_status velri_slot_bind_host(velri_slot *slot, velri_storage_id storage_id, void *host_ptr,
                                  size_t byte_size);

/* Explicit host <-> device synchronization for resident VELRI_STORAGE_STATE buffers. */
velri_status velri_slot_sync_state_from_host(velri_slot *slot, velri_storage_id storage_id);
velri_status velri_slot_sync_state_to_host(velri_slot *slot, velri_storage_id storage_id);

/* Execute the prepared launch plan synchronously on this slot with 0 B device/heap allocs. */
velri_status velri_slot_run(velri_slot *slot, const velri_step_params *params,
                            velri_cancel_token *cancel);

bool velri_slot_needs_reset(const velri_slot *slot);
velri_status velri_slot_reset(velri_slot *slot);

/* Real count of backend device allocations performed during the last `velri_slot_run`. */
uint64_t velri_slot_last_run_allocs(const velri_slot *slot);

#endif /* VELRI_LIB_GRAPH_H */
