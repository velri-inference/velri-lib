#ifndef VELRI_BACKEND_VK_INTERNAL_H
#define VELRI_BACKEND_VK_INTERNAL_H

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif

#include "core_internal.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <vulkan/vulkan.h>

/*
 * Internal shared types, ABI push-constant layouts, named tuning/hardware constants,
 * and cross-TU interfaces for the built-in Vulkan 1.3 Compute Backend.
 */

enum {
    VK_CMD_RING_SIZE          = 8U,
    VK_MAPPED_IO_MAX_BYTES    = 256U * 1024U, /* Limit use of CPU-visible device memory. */
    VK_STAGE_MAX_BYTES        = 4U * 1024U * 1024U,
    VK_MAX_PC_BYTES           = 128U,
    VK_MAX_DISPATCH_STAGES    = 3U,
    VK_MIN_BUF_ALIGN_BYTES    = 16U,
    VK_GEMM_F32_SMEM_BYTES    = 2U * 2U * 8U * 33U * 16U, /* 16896 B */
    VK_COOPMAT_128_SMEM_BYTES = 2U * 128U * 5U * 16U,     /* 20480 B */
    VK_TARGET_WGS             = 140U,
    VK_SHORT_GEMM_TARGET_WGS  = 512U,
    VK_MIXED_GEMM_MIN_ROWS    = 64U,
    VK_GEMV_WG_SIZE           = 128U,
    VK_ROW_WG_SIZE            = 256U,
    VK_SPLITKV_MIN_KV_LEN     = 256U,
    VK_SPLITKV_TILE_MIN_KV_LEN = 128U,
    VK_SPLITKV_SLICED_MIN_KV_LEN = 257U,
    VK_SPLITKV_MAX_SPLITS     = 64U,
    VK_SPLITKV_BUCKETS        = 6U,    /* Power-of-2 buckets: {2, 4, 8, 16, 32, 64} */
    VK_STEP_SLAB_SLOTS        = 1024U, /* 64 KiB shared host-BDA slab = 1024 captured batches */
    VK_ENC_SCRATCH_INIT_CAP   = 2048U
};

#define VK_SPLITK_MAX_BYTES (32ULL * 1024ULL * 1024ULL)

/* Dynamic step-parameter uniform buffer read by persistent captured T=1 decode graphs. */
typedef struct {
    uint32_t seq_len;
    uint32_t pos_offset;
    uint32_t kv_pos;
    uint32_t pad[13];
} vk_gpu_step_params;
static_assert(sizeof(vk_gpu_step_params) == 64, "vk_gpu_step_params size mismatch");

/* ---- Capability Bitmask & Pipeline IDs ---- */

enum {
    VK_CAP_NONE     = 0U,
    VK_CAP_INT16    = 1U << 0,
    VK_CAP_F16_I8   = 1U << 1,
    VK_CAP_INT_DOT  = 1U << 2,
    VK_CAP_F32_GEMM = 1U << 3,
    VK_CAP_BF16_EXT = 1U << 4,
    VK_CAP_CM_F16   = 1U << 5,
    VK_CAP_CM_BF16  = 1U << 6,
    VK_CAP_CM_I8    = 1U << 7,
    VK_CAP_F16_KV   = 1U << 8,
    VK_CAP_SHARED_32K = 1U << 9,
    VK_CAP_NV_ATTENTION = 1U << 10
};

enum {
    PIPE_RMSNORM_F32 = 0,
    PIPE_RMSNORM_BF16,
    PIPE_RMSNORM_F16,
    PIPE_FUSED_ADD_RMSNORM_F32,
    PIPE_LAYERNORM_F32,
    PIPE_ROPE_F32,
    PIPE_ROPE_BF16,
    PIPE_ACT_SILU_F32,
    PIPE_ACT_GELU_F32,
    PIPE_ACT_RELU2_F32,
    PIPE_GATED_SILU_F32,
    PIPE_GATED_GELU_F32,
    PIPE_VEC_AXPBY_F32,
    PIPE_VEC_MUL_F32,
    PIPE_SOFTMAX_F32,
    PIPE_FWHT_F32,
    PIPE_QUANTIZE_ROW_F32_I8,
    PIPE_GEMV_F32,
    PIPE_QKV_ROPE_CACHE_F32,
    PIPE_QKV_ROPE_CACHE_F16KV,
    PIPE_RMS_QKV_F32,
    PIPE_RMS_QKV_F16KV,
    PIPE_RMS_GATED_GEMV_F32,
    PIPE_GATED_GEMV_F32,
    PIPE_GEMV_RESIDUAL_F32,
    PIPE_GEMV_F16,
    PIPE_GEMV_BF16,
    PIPE_GEMV_W8A8,
    PIPE_GEMM_F32,
    PIPE_GEMM_SHORT_F32,
    PIPE_GEMV_ROWS_F32,
    PIPE_SPLITK_REDUCE,
    PIPE_GEMM_COOPMAT_F16,
    PIPE_GEMM_COOPMAT_BF16,
    PIPE_GEMM_COOPMAT_W8A8,
    PIPE_GEMM_COOPMAT_F16_64,
    PIPE_GEMM_F32_AS_F16,
    PIPE_GEMM_COOPMAT_BF16_64,
    PIPE_GEMM_COOPMAT_W8A8_64,
    PIPE_EMBEDDING_F32,
    PIPE_COPY_LAST_ROW_F32,
    PIPE_KV_APPEND_F32,
    PIPE_KV_APPEND_F16KV,
    PIPE_ATTENTION_DECODE_F32,
    PIPE_ATTENTION_DECODE_F16KV,
    PIPE_ATTENTION_PREFILL_F32,
    PIPE_ATTENTION_PREFILL_F16KV,
    PIPE_FLASH_ATTENTION_F32,
    PIPE_FLASH_ATTENTION_F16KV,
    PIPE_PREFILL_COOP_F16KV,
    PIPE_PREFILL_COOP_HALF_F16KV,
    PIPE_PREFILL_COOP_NV_F16KV,
    PIPE_PREFILL_TILE_F32,
    PIPE_PREFILL_SPLIT_F32,
    PIPE_PREFILL_SPLIT_F16KV,
    PIPE_PREFILL_MERGE_F32,
    PIPE_ATTENTION_WIDE_F32,
    PIPE_ATTENTION_WIDE_F16KV,
    PIPE_ATTENTION_SPLITKV_F32,
    PIPE_ATTENTION_SPLITKV_F16KV,
    PIPE_ATTENTION_SPLITKV_REDUCE_F32,
    PIPE_ATTENTION_GROUPED_F16KV,
    PIPE_ATTENTION_MERGE64_F32,
    PIPE_BATCHED_LINEAR_NT_F32,
    PIPE_BATCHED_LINEAR_NN_F32,
    PIPE_COUNT
};

/* ---- X-Macro Vulkan Loader Tables ---- */

#define VK_INST_FUNCS(X)                             \
    X(vkDestroyInstance)                             \
    X(vkEnumeratePhysicalDevices)                    \
    X(vkGetPhysicalDeviceProperties2)                \
    X(vkGetPhysicalDeviceFeatures2)                  \
    X(vkGetPhysicalDeviceMemoryProperties)           \
    X(vkGetPhysicalDeviceQueueFamilyProperties)      \
    X(vkEnumerateDeviceExtensionProperties)          \
    X(vkCreateDevice)                                \
    X(vkGetDeviceProcAddr)

#define VK_DEV_FUNCS(X)                              \
    X(vkDestroyDevice)                               \
    X(vkGetDeviceQueue)                              \
    X(vkCreateCommandPool)                           \
    X(vkDestroyCommandPool)                          \
    X(vkAllocateCommandBuffers)                      \
    X(vkResetCommandBuffer)                          \
    X(vkBeginCommandBuffer)                          \
    X(vkEndCommandBuffer)                            \
    X(vkQueueSubmit)                                 \
    X(vkQueueWaitIdle)                               \
    X(vkCreateFence)                                 \
    X(vkDestroyFence)                                \
    X(vkWaitForFences)                               \
    X(vkResetFences)                                 \
    X(vkCreateBuffer)                                \
    X(vkDestroyBuffer)                               \
    X(vkGetBufferMemoryRequirements)                 \
    X(vkAllocateMemory)                              \
    X(vkFreeMemory)                                  \
    X(vkBindBufferMemory)                            \
    X(vkMapMemory)                                   \
    X(vkUnmapMemory)                                 \
    X(vkGetBufferDeviceAddress)                      \
    X(vkCreateShaderModule)                          \
    X(vkDestroyShaderModule)                         \
    X(vkCreatePipelineLayout)                        \
    X(vkDestroyPipelineLayout)                       \
    X(vkCreateComputePipelines)                      \
    X(vkDestroyPipeline)                             \
    X(vkCmdBindPipeline)                             \
    X(vkCmdPushConstants)                            \
    X(vkCmdDispatch)                                 \
    X(vkCmdPipelineBarrier)                          \
    X(vkCmdCopyBuffer)                               \
    X(vkCmdFillBuffer)

typedef struct {
    velri_os_lib *dl;
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;
    PFN_vkCreateInstance vkCreateInstance;
#define DECL_INST_FIELD(NAME) PFN_##NAME NAME;
    VK_INST_FUNCS(DECL_INST_FIELD)
#undef DECL_INST_FIELD
    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR
        vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR;
    VkInstance instance;
} vk_inst_loader;

typedef struct {
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDeviceAddress bda;
    size_t size;
    void *host_map; /* Persistent host mapping for writable slot IO or cached readback. */
} vk_buf;

typedef struct vk_step_slab {
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDeviceAddress bda;
    vk_gpu_step_params *mapped;
    uint64_t used_mask[VK_STEP_SLAB_SLOTS / 64U];
    struct vk_step_slab *next;
} vk_step_slab;

struct velri_completion {
    VkCommandPool pool;
    VkCommandBuffer cmd;
    VkFence fence;
    bool in_use;
    bool uses_splitk;
    bool is_captured_graph;
};

struct velri_captured_graph {
    VkCommandPool pool;
    VkCommandBuffer cmd;
    VkCommandBuffer splitkv_cmds[VK_SPLITKV_BUCKETS]; /* Pre-recorded buckets for n_splits={2,4,8,16,32,64} */
    VkFence fence;
    struct velri_completion comp;
    vk_step_slab *step_slab;
    uint32_t step_slot;
    VkDeviceAddress step_bda;
    vk_gpu_step_params *step_map;
    vk_buf *readback; /* Private coherent staging for the final batch of this slot. */
    bool splitkv_possible;
    bool splitkv_uniform;
    uint32_t splitkv_min_len;
    uint32_t splitkv_nh;
    uint32_t splitkv_dv;
    uint32_t splitkv_max_ctx;
    uint32_t splitkv_sw;
    uint32_t splitkv_max_splits;
    uint32_t splitkv_limit;
    uint64_t splitkv_buf_gen;
    bool require_explicit_pos_offset;
    uint32_t default_pos_offset;
    bool has_pos_bound;
    uint32_t max_pos_offset;
    bool has_kv_bound;
    uint32_t max_kv_pos;
};

struct velri_prepared_op {
    int pipe_id;
    int alt_pipe_id;                /* LINEAR: GEMV pipeline for dynamic active_rows <= 2 */
    int reduce_pipe_id;             /* LINEAR Split-K: PIPE_SPLITK_REDUCE */
    int stage2_decode_pipe_id;      /* ATTENTION: stage 2 decode/wide pipeline */
    bool flash_prefill;             /* Eligible layout; final addresses checked when encoding. */
    int stage2_prefill_pipe_id;     /* ATTENTION: PIPE_ATTENTION_PREFILL_F32 */
    int stage2_splitkv_pipe_id;     /* ATTENTION: PIPE_ATTENTION_SPLITKV_F32 */
    int stage3_splitkv_reduce_pipe_id; /* ATTENTION: PIPE_ATTENTION_SPLITKV_REDUCE_F32 */
    uint32_t split_k;
    size_t splitk_bytes;
};

typedef enum {
    VK_SPLITK_NONE = 0,
    VK_SPLITK_GEMM = 1,
    VK_SPLITK_ATTN = 2
} vk_splitk_kind;

typedef struct {
    VkPipeline pipe;
    uint32_t gx, gy, gz;
    uint32_t pc_size;
    alignas(8) uint8_t pc[VK_MAX_PC_BYTES];
} vk_dispatch_stage;

static inline void set_dispatch_stage(vk_dispatch_stage *st, VkPipeline pipe, const void *pc,
                               size_t pc_size, uint32_t gx, uint32_t gy, uint32_t gz) {
    st->pipe = pipe;
    st->gx = gx;
    st->gy = gy;
    st->gz = gz;
    st->pc_size = (uint32_t)pc_size;
    memcpy(st->pc, pc, pc_size);
}

typedef struct {
    vk_dispatch_stage stages[VK_MAX_DISPATCH_STAGES];
    uint32_t num_stages;
    vk_splitk_kind splitk_kind;
    size_t splitk_bytes;
} vk_encoded_dispatch;

typedef struct {
    VkBool32 shader_int16;
    VkBool32 storage_16bit;
    VkBool32 storage_8bit;
    VkBool32 shader_float16;
    VkBool32 shader_int8;
} vk_probed_feats;

struct velri_backend_device {
    vk_inst_loader inst;
#define DECL_DEV_FIELD(NAME) PFN_##NAME NAME;
    VK_DEV_FUNCS(DECL_DEV_FIELD)
#undef DECL_DEV_FIELD

    VkPhysicalDevice phys;
    VkPhysicalDeviceMemoryProperties mem_props;
    uint32_t queue_family;
    uint32_t local_index;
    uint32_t subgroup_size;
    uint32_t max_wg_count[3];
    uint32_t caps_mask;
    velri_device_caps caps;

    VkDevice device;
    VkQueue queue;
    VkPipelineLayout pipe_layout;
    VkPipeline pipes[PIPE_COUNT];

    velri_os_mutex *mtx;
    velri_os_cond *slot_cv;
    bool splitk_busy;

    /* Dedicated transfer command buffer + bounded staging buffer (under `mtx`). */
    VkCommandPool xfer_pool;
    VkCommandBuffer xfer_cmd;
    VkFence xfer_fence;
    VkBuffer stage_buf;
    VkDeviceMemory stage_mem;
    void *stage_map;
    size_t stage_cap;

    /* Async compute submission ring (0-alloc hot path). */
    struct velri_completion ring[VK_CMD_RING_SIZE];

    /* Shared Split-K / Split-KV partial buffer (pre-allocated in `velri_vk_prepare_op`). */
    vk_buf *splitk_buf;
    uint64_t splitk_buf_gen;

    /* Shared host-BDA slab pool for captured-graph `vk_gpu_step_params` (avoids per-batch VkDeviceMemory). */
    vk_step_slab *step_slabs;

    /* Pre-allocated batch-encoding scratch under `mtx` (eliminates stack cap & hot-path allocs). */
    vk_encoded_dispatch *enc_scratch;
    uint32_t enc_scratch_cap;
    vk_probed_feats features;
};

static inline bool velri_vk_mixed_gemm_fits(const velri_backend_device *dev, uint32_t rows,
                                             uint32_t cols) {
    return rows >= VK_MIXED_GEMM_MIN_ROWS &&
           velri_div_ceil_u32(cols, 64U) <= dev->max_wg_count[0] &&
           velri_div_ceil_u32(rows, 64U) <= dev->max_wg_count[1];
}

static inline bool velri_vk_f16kv_coop_fits(const velri_backend_device *dev,
                                            const velri_resolved_op *op, uint32_t rows) {
    const velri_tensor_view *q = &op->inputs[0], *k = &op->outputs[1], *v = &op->outputs[2];
    uint32_t window = op->raw.attr.attention.sliding_window;
    return rows >= 64U && dev->subgroup_size == 32U &&
           k->dtype == VELRI_DTYPE_F16 && op->raw.attr.attention.head_dim == 64U &&
           velri_op_attn_dv(op) == 64U && k->dims[1] >= 16U && k->dims[1] <= UINT32_MAX - 256U &&
           (window == 0U || window >= 512U) && (q->strides[0] & 3U) == 0U &&
           ((k->strides[0] | k->strides[1] | v->strides[0] | v->strides[1]) & 7U) == 0U;
}

// The NV matrix loads consume complete 32-row KV and query tiles.
static inline bool velri_vk_nv_prefill_shape(const velri_resolved_op *op) {
    size_t capacity = op->outputs[1].dims[1];
    return capacity >= 1536U && (capacity & 31U) == 0U &&
           op->raw.attr.attention.causal && !op->raw.attr.attention.sliding_window;
}

/* ---- Cross-TU Vulkan Backend Functions ---- */

/* vk_device.c */
velri_status velri_vk_enumerate_devices(uint32_t *out_count);
velri_status velri_vk_open_device(uint32_t local_index, const void *cfg,
                                  velri_backend_device **out_dev);
velri_status velri_vk_query_caps(const velri_backend_device *dev, velri_device_caps *out_caps);
void velri_vk_close_device(velri_backend_device *dev);
velri_status velri_vk_activate_locked(velri_backend_device *dev);
velri_status velri_vk_init_cmd_slot(velri_backend_device *dev, VkCommandPool *out_pool,
                                    VkCommandBuffer *out_cmd, VkFence *out_fence);

/* vk_memory.c */
velri_status velri_vk_alloc_buf_locked(velri_backend_device *dev, size_t bytes, size_t align,
                                       vk_buf **out_buf);
void velri_vk_free_buf_locked(velri_backend_device *dev, vk_buf *b);
velri_status velri_vk_alloc_io(velri_backend_device *dev, size_t bytes, size_t align, void **out);
void velri_vk_record_readback(velri_backend_device *dev, VkCommandBuffer cmd, VkBuffer src,
                              velri_io_range range, VkBuffer dst);
velri_status velri_vk_alloc_readback_locked(velri_backend_device *dev, size_t bytes, vk_buf **out);
velri_status velri_vk_alloc_host_bda_locked(velri_backend_device *dev, size_t bytes,
                                            VkBuffer *out_buf, VkDeviceMemory *out_mem,
                                            VkDeviceAddress *out_bda, void **out_map);
velri_status velri_vk_alloc_step_slot_locked(velri_backend_device *dev, vk_step_slab **out_slab,
                                             uint32_t *out_slot, VkDeviceAddress *out_bda,
                                             vk_gpu_step_params **out_map);
void velri_vk_free_step_slot_locked(vk_step_slab *slab, uint32_t slot);
void velri_vk_free_all_step_slabs_locked(velri_backend_device *dev);
velri_status velri_vk_init_staging_buffer(velri_backend_device *dev);
velri_status velri_vk_alloc(velri_backend_device *dev, size_t bytes, size_t align,
                            void **out_mem_handle);
void velri_vk_free(velri_backend_device *dev, void *mem_handle);
velri_status velri_vk_copy_to_ram(velri_backend_device *dev, void *dst_ram,
                                  const velri_buffer_binding *src, size_t bytes);
velri_status velri_vk_copy_from_ram(velri_backend_device *dev, const velri_buffer_binding *dst,
                                    const void *src_ram, size_t bytes);
velri_status velri_vk_zero_mem(velri_backend_device *dev, void *mem_handle, size_t bytes);

/* vk_pipeline.c */
uint32_t velri_vk_fuse_decode(velri_backend_device *dev, const velri_submit_item *items,
                              const vk_encoded_dispatch *encoded, uint32_t count,
                              vk_encoded_dispatch *out);
VkPipeline velri_vk_get_pipe_locked(velri_backend_device *dev, int id);
bool velri_vk_splitk_layout(uint32_t sk, uint32_t rows, uint32_t m, uint32_t *out_part_stride,
                            size_t *out_bytes);
uint32_t velri_vk_choose_split_k(uint32_t rows, uint32_t m, uint32_t k, uint32_t tile_m,
                                 uint32_t tile_n, uint32_t target_wgs, uint32_t bk_elems);
bool velri_vk_splitkv_layout(uint32_t n_splits, uint32_t n_heads, uint32_t d_v,
                             uint32_t *out_part_stride, size_t *out_bytes);
uint32_t velri_vk_prefill_splits(const velri_backend_device *dev, const velri_resolved_op *op,
                                 uint32_t rows, uint32_t keys, size_t *out_bytes);
uint32_t velri_vk_splitkv_min_len(const velri_backend_device *dev, const velri_resolved_op *op);
/* Each group stays within the Q heads sharing one KV head. */
static inline uint32_t velri_vk_decode_group_size(uint32_t queries_per_kv) {
    if (queries_per_kv % 4U == 0U) return 4U;
    if (queries_per_kv % 3U == 0U) return 3U;
    return 1U;
}

uint32_t velri_vk_splitkv_limit(const velri_backend_device *dev, const velri_resolved_op *op);
uint32_t velri_vk_choose_split_kv(uint32_t n_heads, uint32_t d_v, uint32_t kv_len,
                                  uint32_t min_len, uint32_t split_limit);
velri_status velri_vk_resolve_op_pipes(const velri_backend_device *dev, const velri_resolved_op *op,
                                       struct velri_prepared_op *out_plan);
velri_status velri_vk_query_operation(const velri_backend_device *dev, const velri_resolved_op *op);
velri_status velri_vk_prepare_op(velri_backend_device *dev, const velri_resolved_op *op,
                                 velri_prepared_op **out_prepared);
void velri_vk_destroy_prepared_op(velri_backend_device *dev, velri_prepared_op *prepared);

#endif /* VELRI_BACKEND_VK_INTERNAL_H */
