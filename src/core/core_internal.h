#ifndef VELRI_CORE_INTERNAL_H
#define VELRI_CORE_INTERNAL_H

#include "os.h"
#include "velri_lib/backend.h"
#include "velri_lib/graph.h"
#include <stdlib.h>

enum {
    VELRI_MAX_DEVICES = VELRI_MAX_REPORT_DEVICES,
    VELRI_MAX_BACKENDS = 16,
    VELRI_MAX_SUBMIT_BATCH = 32
};

static inline bool velri_mul_overflow_size(size_t a, size_t b, size_t *out) {
    if (a != 0U && b > SIZE_MAX / a) {
        return true;
    }
    *out = a * b;
    return false;
}

static inline bool velri_add_overflow_size(size_t a, size_t b, size_t *out) {
    if (a > SIZE_MAX - b) {
        return true;
    }
    *out = a + b;
    return false;
}

static inline size_t velri_align_up(size_t n, size_t align) {
    if (align <= 1U) {
        return n;
    }
    size_t mask = align - 1U;
    if (n > SIZE_MAX - mask) {
        return SIZE_MAX;
    }
    return (n + mask) & ~mask;
}

static inline bool velri_vec_grow(void **ptr, uint32_t *cap, uint32_t need, size_t elem_size) {
    if (need <= *cap) {
        return true;
    }
    uint32_t next = (*cap == 0U) ? 16U : *cap;
    while (next < need) {
        if (next > UINT32_MAX / 2U) {
            return false;
        }
        next *= 2U;
    }
    size_t bytes = 0U;
    if (velri_mul_overflow_size((size_t)next, elem_size, &bytes)) {
        return false;
    }
    void *p = realloc(*ptr, bytes);
    if (!p) {
        return false;
    }
    *ptr = p;
    *cap = next;
    return true;
}

/* ---- Context & Registered Backends ---- */

typedef struct {
    size_t workers;
    velri_cpu_variant variant;
} velri_cpu_open_cfg;

typedef struct {
    const velri_backend_vtable *vt;
    velri_backend_device *dev;
    velri_device_caps caps;
} velri_device_entry;

struct velri_context {
    _Atomic uint32_t refcount;
    _Atomic uint64_t total_allocs;
    velri_context_config cfg;
    const velri_backend_vtable *backends[VELRI_MAX_BACKENDS];
    uint32_t num_backends;
    velri_device_entry devices[VELRI_MAX_DEVICES];
    uint32_t num_devices;
};

void velri_context_retain(velri_context *ctx);
void velri_context_release(velri_context *ctx);
velri_status velri_device_alloc_tracked(velri_context *ctx, velri_device_id dev_id, size_t bytes,
                                        size_t align, void **out_handle);

/* ---- Operation Graph ---- */

typedef struct {
    velri_op_desc desc;
    velri_device_id device_id;
} velri_graph_node;

struct velri_graph {
    velri_storage_desc *storages;
    uint32_t num_storages;
    uint32_t cap_storages;

    velri_tensor_view *views;
    uint32_t num_views;
    uint32_t cap_views;

    velri_graph_node *nodes;
    uint32_t num_nodes;
    uint32_t cap_nodes;
};

void velri_graph_resolve_node(const velri_graph *graph, const velri_op_desc *op,
                              velri_resolved_op *out);

/* ---- Prepared Launch Plan ---- */

typedef enum {
    VELRI_STEP_OP = 0,
    VELRI_STEP_RAM_TRANSFER = 1
} velri_launch_step_kind;

typedef struct {
    velri_storage_kind kind;
    velri_dtype dtype;
    velri_device_id device_id;
    size_t byte_size;
    size_t align;
    bool in_scratch_arena; /* True => lives inside per-device packed scratch arena */
    size_t arena_offset;   /* Offset inside that device's scratch arena */
} velri_plan_storage;

typedef struct {
    const void *host_ptr;
    size_t bound_bytes;
    void *device_handle; /* Allocated once per plan for non-host WEIGHT storages */
    bool is_bound;
} velri_plan_weight_binding;

typedef struct {
    uint32_t plan_storage_idx;
    size_t view_byte_offset;
    bool present;
} velri_step_arg;

typedef struct {
    velri_launch_step_kind kind;
    union {
        struct {
            velri_device_id device_id;
            velri_resolved_op resolved;
            velri_prepared_op *prepared; /* Optional backend handle (may be nullptr) */
            uint32_t num_inputs;
            uint32_t num_outputs;
            velri_step_arg inputs[VELRI_OP_MAX_INPUTS];
            velri_step_arg outputs[VELRI_OP_MAX_OUTPUTS];
        } op;
        struct {
            velri_device_id src_device_id;
            velri_device_id dst_device_id;
            uint32_t src_plan_storage_idx;
            uint32_t dst_plan_storage_idx;
            size_t byte_size;
        } transfer;
    } u;
} velri_launch_step;

struct velri_plan {
    _Atomic uint32_t refcount;
    velri_context *ctx;
    uint32_t num_user_storages;
    velri_plan_storage *storages;
    uint32_t num_storages;

    velri_plan_weight_binding *weight_bindings; /* Shared immutable WEIGHT bindings [num_user_storages] */

    size_t scratch_arena_bytes[VELRI_MAX_DEVICES];
    size_t scratch_arena_align[VELRI_MAX_DEVICES];
    size_t ram_staging_bytes;

    velri_launch_step *steps;
    uint32_t num_steps;

    velri_memory_report report;
};

void velri_plan_retain(velri_plan *plan);
void velri_plan_release(velri_plan *plan);

/* Central operation registry validator (src/ops/registry.c). */
velri_status velri_op_validate(const velri_resolved_op *op);

#ifdef VELRI_CORE_HAS_VULKAN
const velri_backend_vtable *velri_vulkan_backend_vtable(void);
#endif

#endif /* VELRI_CORE_INTERNAL_H */
