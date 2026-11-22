#ifndef VELRI_LIB_BACKEND_H
#define VELRI_LIB_BACKEND_H

#include "velri_lib/types.h"
#include <stdatomic.h>

/*
 * Storage/TensorView descriptors, Operation descriptors, and Backend Contract.
 *
 * Backend contract (`velri-lib/docs/EN/architecture.md §5`):
 *   1. Registration (name, contract version, device enumeration)
 *   2. Device       (open with optional config, capabilities/limits, close)
 *   3. Memory       (alloc, free, copy_to_ram, copy_from_ram, optional zero_mem)
 *   4. Preparation  (query_operation, optional prepare_op / destroy_prepared_op)
 *   5. Execution    (submit, optional submit_batch for multi-op command batching)
 *   6. Completion   (optional wait / destroy_completion for asynchronous devices)
 */

#define VELRI_BACKEND_CONTRACT_VERSION 2U

enum {
    VELRI_MAX_RANK = 4,
    VELRI_INVALID_ID = 0xFFFFFFFFU,
    VELRI_OP_MAX_INPUTS = 6,
    VELRI_OP_MAX_OUTPUTS = 3,
    VELRI_ATTN_MAX_HEAD_DIM = 2048
};

typedef uint32_t velri_storage_id;
typedef uint32_t velri_view_id;
typedef uint32_t velri_node_id;
typedef uint32_t velri_device_id;

typedef enum {
    VELRI_STORAGE_WEIGHT = 0,  /* Immutable weights; shared read-only across all slots on a plan. */
    VELRI_STORAGE_INPUT = 1,   /* Step input bound per slot before execution. */
    VELRI_STORAGE_OUTPUT = 2,  /* Step output bound per slot; kept alive through step end. */
    VELRI_STORAGE_STATE = 3,   /* Mutable session state (KV, GDN, conv) resident across steps. */
    VELRI_STORAGE_SCRATCH = 4  /* Temporary slot-local buffer packed/reused by lifetime planner. */
} velri_storage_kind;

typedef struct {
    velri_storage_kind kind;
    velri_dtype dtype;
    size_t byte_size;
    size_t align;              /* 0 => natural alignment >= 64 B */
    velri_device_id device_id; /* Target device owning this storage */
} velri_storage_desc;

typedef struct {
    velri_storage_id storage_id;
    velri_dtype dtype;
    uint32_t rank;                  /* 1..VELRI_MAX_RANK */
    size_t dims[VELRI_MAX_RANK];
    size_t strides[VELRI_MAX_RANK]; /* Element strides (innermost typically 1) */
    size_t byte_offset;             /* Byte offset inside `storage_id` */
} velri_tensor_view;

/* Build a row-major contiguous tensor view at `byte_offset`. */
velri_status velri_tensor_view_contiguous(velri_storage_id storage_id, velri_dtype dtype,
                                          uint32_t rank, const size_t *dims, size_t byte_offset,
                                          velri_tensor_view *out_view);

/* Create a zero-copy subview slicing the same storage_id at `extra_byte_offset` bytes inside `base`. */
velri_status velri_tensor_view_subview(const velri_tensor_view *base, uint32_t rank,
                                       const size_t *dims, const size_t *strides,
                                       size_t extra_byte_offset, velri_tensor_view *out_view);

velri_status velri_tensor_view_num_elements(const velri_tensor_view *view, size_t *out_elems);
velri_status velri_tensor_view_byte_span(const velri_tensor_view *view, size_t *out_start,
                                         size_t *out_end);
velri_status velri_tensor_view_validate(const velri_tensor_view *view,
                                        const velri_storage_desc *storage);
bool velri_tensor_view_is_contiguous(const velri_tensor_view *view);

/* ---- Operation Descriptors ---- */

typedef enum {
    VELRI_OP_LINEAR = 0,
    VELRI_OP_RMSNORM = 1,
    VELRI_OP_FUSED_ADD_RMSNORM = 2,
    VELRI_OP_LAYERNORM = 3,
    VELRI_OP_ROPE = 4,
    VELRI_OP_ACTIVATION = 5,
    VELRI_OP_GATED_MLP = 6,
    VELRI_OP_SOFTMAX = 7,
    VELRI_OP_VEC_AXPBY = 8,
    VELRI_OP_VEC_MUL = 9,
    VELRI_OP_FWHT = 10,
    VELRI_OP_QUANTIZE_ROW = 11,
    VELRI_OP_EMBEDDING = 12,
    VELRI_OP_ATTENTION = 13,
    VELRI_OP_COPY_LAST_ROW = 14,
    VELRI_OP_BATCHED_LINEAR = 15
} velri_op_kind;

typedef struct {
    velri_op_kind kind;
    uint32_t num_inputs;
    uint32_t num_outputs;
    velri_view_id inputs[VELRI_OP_MAX_INPUTS]; /* VELRI_INVALID_ID for optional absent inputs */
    velri_view_id outputs[VELRI_OP_MAX_OUTPUTS];
    bool dynamic_seq; /* True => axis-0 scales dynamically with seq_len_override (EMBEDDING, ATTENTION, COPY_LAST_ROW are always dynamic-seq) */
    union {
        struct {
            velri_linear_desc desc;
        } linear;
        struct {
            float eps;
            bool unit_offset;
        } rmsnorm;
        struct {
            float eps;
        } layernorm;
        struct {
            size_t rotary_dim;
            size_t pos_offset;
            velri_rope_mode mode;
        } rope;
        struct {
            velri_act_kind kind;
        } activation;
        struct {
            velri_act_kind act;
            float clamp_limit;
        } gated_mlp;
        struct {
            float alpha;
            float beta;
        } axpby;
        struct {
            size_t block_size;
            velri_dtype suh_dtype;
        } fwht;
        struct {
            size_t pos_offset; /* Used when inputs[1] (`ids`) is absent (positional lookup) */
        } embedding;
        struct {
            uint32_t n_heads;
            uint32_t n_kv_heads;
            uint32_t head_dim;       /* Key/query dim d_k (1..VELRI_ATTN_MAX_HEAD_DIM) */
            uint32_t v_head_dim;     /* Value/output dim d_v (0 => head_dim; 1..VELRI_ATTN_MAX_HEAD_DIM) */
            float scale;             /* 0.0f => 1.0f / sqrtf((float)head_dim) */
            uint32_t sliding_window; /* 0 => full causal history */
            bool causal;
        } attention;
        struct {
            uint32_t n_heads;       /* Batch count per token row (B >= 1) */
            uint32_t n_groups;      /* Weight groups (G >= 1, B % G == 0) */
            uint32_t in_dim;        /* Input features per head (K >= 1) */
            uint32_t out_dim;       /* Output features per head (M >= 1) */
            bool weight_transposed; /* false: W [G*M, K] (Y = X @ W^T + b); true: W [G*K, M] (Y = X @ W + b) */
        } batched_linear;
    } attr;
} velri_op_desc;

typedef struct {
    velri_op_kind kind;
    uint32_t num_inputs;
    uint32_t num_outputs;
    bool input_present[VELRI_OP_MAX_INPUTS];
    velri_tensor_view inputs[VELRI_OP_MAX_INPUTS];
    velri_tensor_view outputs[VELRI_OP_MAX_OUTPUTS];
    velri_op_desc raw;
} velri_resolved_op;

typedef struct {
    void *mem_handle;   /* Backend allocation handle (or host ptr on host-memory device) */
    size_t byte_offset; /* Arena offset + tensor view byte_offset */
    size_t byte_size;   /* Valid bound byte size from offset 0 of allocation */
    bool is_host_ptr;   /* True if `mem_handle` is a direct host pointer */
} velri_buffer_binding;

typedef struct {
    size_t pos_offset_override;
    bool has_pos_offset;
    size_t seq_len_override; /* Dynamic active rows 1..max_seq_len (must be >= 1 when has_seq_len) */
    bool has_seq_len;
    size_t kv_len_override;  /* Valid KV cache length BEFORE appending this step's tokens */
    bool has_kv_len;
} velri_step_params;

typedef struct {
    atomic_bool cancelled;
} velri_cancel_token;

static inline void velri_cancel_init(velri_cancel_token *tok) {
    if (tok) atomic_init(&tok->cancelled, false);
}
static inline void velri_cancel_clear(velri_cancel_token *tok) {
    if (tok) atomic_store_explicit(&tok->cancelled, false, memory_order_release);
}
static inline void velri_cancel_request(velri_cancel_token *tok) {
    if (tok) atomic_store_explicit(&tok->cancelled, true, memory_order_release);
}
static inline bool velri_cancel_is_requested(const velri_cancel_token *tok) {
    return tok ? atomic_load_explicit(&tok->cancelled, memory_order_acquire) : false;
}

typedef struct {
    char backend_name[32];
    char device_name[128];
    uint32_t local_index;
    bool is_host_memory;    /* Device directly executes on host RAM pointers */
    bool is_uma;            /* Shares physical RAM pool with CPU */
    bool mem_total_known;   /* False if driver does not report reliable memory size */
    size_t mem_total_bytes; /* 0 if !mem_total_known */
    size_t min_align_bytes; /* Required base allocation alignment */
    uint32_t cpu_workers;   /* Actual pool size, including the caller; 0 on other devices */
} velri_device_caps;

typedef struct velri_backend_device velri_backend_device;
typedef struct velri_prepared_op velri_prepared_op;
typedef struct velri_completion velri_completion;
typedef struct velri_captured_graph velri_captured_graph;

typedef struct {
    const velri_resolved_op *op;
    const velri_prepared_op *prepared;
    const velri_buffer_binding *inputs;
    uint32_t num_inputs;
    const velri_buffer_binding *outputs;
    uint32_t num_outputs;
    const velri_step_params *params;
    /* DAG barrier hint (default false => conservative barrier before item; true => planner proved
     * zero RAW/WAR/WAW hazard against preceding un-barriered items in this batch). */
    bool no_barrier_before;
} velri_submit_item;

/*
 * Resource-safety contract for `submit` / `submit_batch`:
 *   - Async backends (`wait != nullptr` and `destroy_completion != nullptr`):
 *     - Pre-accept rejection: returns error AND sets `*out_completion = nullptr`.
 *     - Acceptance (incl. partial-accept error): sets `*out_completion != nullptr`;
 *       the runtime ALWAYS calls `wait` and `destroy_completion` before releasing buffers.
 *   - Sync backends (`wait == nullptr` and `destroy_completion == nullptr`):
 *     - Always leaves `*out_completion = nullptr`. Returning `VELRI_OK` or `VELRI_ERR_EXEC`
 *       means work ran; returning `VELRI_ERR_INVALID`/`UNSUPPORTED`/`OVERFLOW` is pre-accept.
 */
typedef struct velri_backend_vtable {
    /* 1. Registration */
    const char *name;
    uint32_t contract_version;
    velri_status (*enumerate_devices)(uint32_t *out_count);

    /* 2. Device */
    velri_status (*open_device)(uint32_t local_index, const void *cfg,
                                velri_backend_device **out_dev);
    velri_status (*query_caps)(const velri_backend_device *dev, velri_device_caps *out_caps);
    void (*close_device)(velri_backend_device *dev);

    /* 3. Memory */
    velri_status (*alloc)(velri_backend_device *dev, size_t bytes, size_t align,
                          void **out_mem_handle);
    void (*free)(velri_backend_device *dev, void *mem_handle);
    velri_status (*copy_to_ram)(velri_backend_device *dev, void *dst_ram,
                                const velri_buffer_binding *src, size_t bytes);
    velri_status (*copy_from_ram)(velri_backend_device *dev, const velri_buffer_binding *dst,
                                  const void *src_ram, size_t bytes);
    velri_status (*zero_mem)(velri_backend_device *dev, void *mem_handle, size_t bytes); /* opt */

    /* 4. Preparation */
    velri_status (*query_operation)(const velri_backend_device *dev, const velri_resolved_op *op);
    velri_status (*prepare_op)(velri_backend_device *dev, const velri_resolved_op *op,
                               velri_prepared_op **out_prepared);                       /* opt */
    void (*destroy_prepared_op)(velri_backend_device *dev, velri_prepared_op *prepared); /* opt */

    /* 5. Execution */
    velri_status (*submit)(velri_backend_device *dev, const velri_submit_item *item,
                           velri_completion **out_completion);
    velri_status (*submit_batch)(velri_backend_device *dev, const velri_submit_item *items,
                                 uint32_t count, velri_completion **out_completion); /* opt */

    /* 6. Completion (optional pair for asynchronous backends) */
    velri_status (*wait)(velri_backend_device *dev, velri_completion *comp);
    void (*destroy_completion)(velri_backend_device *dev, velri_completion *comp);

    /* 7. Optional Persistent Launch-Graph Capture (all 3 provided together or all nullptr) */
    velri_status (*capture_graph)(velri_backend_device *dev, const velri_submit_item *items,
                                  uint32_t count, velri_captured_graph **out_graph); /* opt */
    void (*destroy_captured_graph)(velri_backend_device *dev,
                                   velri_captured_graph *graph); /* opt */
    velri_status (*replay_graph)(velri_backend_device *dev, velri_captured_graph *graph,
                                 const velri_step_params *params,
                                 velri_completion **out_completion); /* opt */
} velri_backend_vtable;

/* Runtime context owning statically registered backends and open devices. */
typedef struct velri_context velri_context;

typedef struct {
    size_t cpu_workers;            /* 0 => auto */
    velri_cpu_variant cpu_variant; /* VELRI_CPU_AUTO by default */
    bool enable_vulkan;            /* Try opening Vulkan devices if compiled with Vulkan */
    bool use_physical_cores;       /* When cpu_workers == 0, count accessible physical cores */
    uint32_t cpu_reserve_cores;    /* Subtracted in physical-core mode; at least one worker */
} velri_context_config;

velri_status velri_context_create(const velri_context_config *cfg, velri_context **out_ctx);
void velri_context_destroy(velri_context *ctx);

velri_status velri_context_register_backend(velri_context *ctx, const velri_backend_vtable *vtable);

uint32_t velri_context_device_count(const velri_context *ctx);
velri_status velri_context_device_caps(const velri_context *ctx, velri_device_id device_id,
                                       velri_device_caps *out_caps);
velri_status velri_context_find_device(const velri_context *ctx, const char *backend_name,
                                       uint32_t local_index, velri_device_id *out_device_id);

const velri_backend_vtable *velri_cpu_backend_vtable(void);

#endif /* VELRI_LIB_BACKEND_H */
