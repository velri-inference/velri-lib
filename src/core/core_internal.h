#ifndef VELRI_CORE_INTERNAL_H
#define VELRI_CORE_INTERNAL_H

#include "ops_internal.h"
#include "os.h"
#include "util_internal.h"
#include "velri_lib/graph.h"

/*
 * Core Runtime, Graph, Plan, and Slot Internal Data Structures.
 * (Generic math/vector helpers live in `util_internal.h`; op metadata lives in `ops_internal.h`.)
 */

enum {
    VELRI_MAX_DEVICES = VELRI_MAX_REPORT_DEVICES,
    VELRI_MAX_BACKENDS = 16,
    VELRI_MAX_WEIGHT_SLABS = 64
};

/* Pack non-host weights into slabs up to 512 MiB each (avoids per-tensor Vulkan allocations). */
#define VELRI_WEIGHT_SLAB_LIMIT ((size_t)512U * 1024U * 1024U)

typedef enum {
    VELRI_ARENA_NONE = 0,
    VELRI_ARENA_SCRATCH = 1,
    VELRI_ARENA_WEIGHT = 2,
    VELRI_ARENA_STATE = 3,
    VELRI_ARENA_IO = 4
} velri_arena_kind;

typedef enum {
    VELRI_SLOT_ARENA_SCRATCH = 0,
    VELRI_SLOT_ARENA_STATE = 1,
    VELRI_SLOT_ARENA_IO = 2,
    VELRI_SLOT_ARENA_COUNT = 3
} velri_slot_arena_id;

/* ---- Buffer Binding Pointer Helpers (shared by executor & CPU backend) ---- */

static inline void *velri_binding_ptr(const velri_buffer_binding *b) {
    return (void *)((uint8_t *)b->mem_handle + b->byte_offset);
}

static inline const void *velri_binding_cptr(const velri_buffer_binding *b) {
    return (const void *)((const uint8_t *)b->mem_handle + b->byte_offset);
}

static inline bool velri_binding_range_valid(const velri_buffer_binding *b, size_t bytes) {
    return b && b->mem_handle && b->byte_offset <= b->byte_size &&
           bytes <= b->byte_size - b->byte_offset;
}

/* ---- Internal Tensor-View Predicates (implemented in `src/core/tensor.c`) ---- */

bool velri_tensor_view_overlaps(const velri_tensor_view *a, const velri_tensor_view *b);
bool velri_tensor_view_identical_layout(const velri_tensor_view *a, const velri_tensor_view *b);
bool velri_tensor_view_no_partial_overlap(const velri_tensor_view *in,
                                          const velri_tensor_view *out);
bool velri_tensor_view_covers_storage(const velri_tensor_view *v, size_t storage_bytes);
velri_status velri_shape_byte_size(velri_dtype dtype, uint32_t rank, const size_t *dims,
                                   size_t *out_bytes);

/* ---- Context & Registered Backends ---- */

typedef struct {
    size_t workers;
    velri_cpu_variant variant;
    bool use_physical_cores;
    uint32_t reserve_cores;
} velri_cpu_open_cfg;

typedef struct {
    bool defer_secondary_devices;
} velri_vk_open_cfg;

typedef velri_status (*velri_alloc_fn)(velri_backend_device *, size_t, size_t, void **);

typedef struct {
    size_t offset;
    size_t bytes;
} velri_io_range;

/* One backend-native representation for all uses of an immutable matrix. */
typedef struct {
    size_t bytes;
    uint32_t channels;
    uint32_t depth;
    uint32_t stride;
} velri_weight_packing;

typedef struct {
    const velri_prepared_op *packed_weight_op; /* Static backend descriptor for native weight operands. */
    bool (*describe_weight)(const velri_backend_device *dev, const velri_resolved_op *op,
                            uint32_t input, velri_weight_packing *out);
    void (*pack_weight)(const velri_backend_device *dev, const velri_weight_packing *packing,
                        const void *source, void *dst);
    velri_alloc_fn alloc_io;
    size_t max_readback_bytes;
    velri_status (*capture_outputs)(velri_backend_device *dev, const velri_submit_item *items,
                                    uint32_t count, void *io_arena, velri_io_range range,
                                    const void **out_readback, velri_captured_graph **out_graph);
} velri_device_hooks;

typedef struct {
    const velri_backend_vtable *vt;
    const velri_device_hooks *hooks;
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
                                        size_t align, velri_arena_kind kind, void **out_handle);

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
    velri_arena_kind arena_kind; /* Arena owning this storage's device memory (if any) */
    uint32_t slab_idx;           /* For VELRI_ARENA_WEIGHT: index in plan->weight_slabs[dev] */
    size_t arena_offset;         /* Byte offset inside the owning arena/slab */
    velri_weight_packing packing;
} velri_plan_storage;

typedef struct {
    const void *host_ptr;
    size_t bound_bytes;
    void *packed_mem;
    bool is_bound;
} velri_plan_weight_binding;

/* Set on the first OP step of each submission. Transfer steps stand alone. */
typedef struct {
    uint32_t count;
    uint32_t capture_index; /* Index in each slot's capture array on backends supporting replay */
    bool static_t1;
} velri_launch_batch;

typedef struct {
    velri_launch_step_kind kind;
    union {
        struct {
            velri_device_id device_id;
            velri_resolved_op resolved;  /* Authoritative num_inputs/outputs, present, byte_offset */
            velri_prepared_op *prepared; /* Optional backend handle (may be nullptr) */
            uint32_t in_storage_idx[VELRI_OP_MAX_INPUTS];
            uint32_t out_storage_idx[VELRI_OP_MAX_OUTPUTS];
            velri_launch_batch batch;
            bool uses_packed_weight;
            bool no_barrier_before; /* True => DAG analysis proved no RAW/WAR/WAW hazard in batch */
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
    velri_plan *weight_owner;                   /* Optional retained peer plan owning WEIGHT slabs/bindings */
    bool has_packed_weights;
    uint32_t *weight_peer_sid;                  /* [num_user_storages] -> owner storage_id when borrowed */

    void *weight_slabs[VELRI_MAX_DEVICES][VELRI_MAX_WEIGHT_SLABS];
    size_t weight_slab_bytes[VELRI_MAX_DEVICES][VELRI_MAX_WEIGHT_SLABS];
    size_t weight_slab_align[VELRI_MAX_DEVICES][VELRI_MAX_WEIGHT_SLABS];
    uint32_t num_weight_slabs[VELRI_MAX_DEVICES];

    size_t slot_arena_bytes[VELRI_SLOT_ARENA_COUNT][VELRI_MAX_DEVICES];
    size_t slot_arena_align[VELRI_SLOT_ARENA_COUNT][VELRI_MAX_DEVICES];
    size_t ram_staging_bytes;
    velri_io_range readback; /* Final batch output span and per-slot allocation budget. */

    velri_launch_step *steps;
    uint32_t num_steps;
    uint32_t max_batch_len;       /* Max consecutive same-device OP steps (for 0-alloc batching) */
    uint32_t num_capture_batches; /* Batches on backends supporting capture/replay */
    uint32_t dag_total_ops;       /* Total VELRI_STEP_OP steps in plan */
    uint32_t dag_elided_barriers; /* Intra-batch barriers elided by DAG hazard analysis */

    velri_memory_report report;
};

extern const velri_device_hooks velri_cpu_device_hooks;
velri_status velri_plan_prepare_host_weights(velri_plan *plan);
void velri_plan_retain(velri_plan *plan);
void velri_plan_release(velri_plan *plan);

#ifdef VELRI_CORE_HAS_VULKAN
const velri_backend_vtable *velri_vulkan_backend_vtable(void);
extern const velri_device_hooks velri_vulkan_device_hooks;
#endif

#endif /* VELRI_CORE_INTERNAL_H */
