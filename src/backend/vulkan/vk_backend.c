#define VK_NO_PROTOTYPES

#include "core_internal.h"
#include "velri_vk_shaders.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

/*
 * Built-in Vulkan 1.3 Compute Backend (SPIR-V 1.6 via Slang AOT).
 *
 * Architecture highlights:
 *   - Dynamically loads libvulkan via `velri_os_dlopen_vulkan()` (zero link-time dep).
 *   - Lightweight `enumerate_devices` (temporary VkInstance only, supports multi-GPU).
 *   - Lazy table-driven compute pipeline creation; pre-compiled during `prepare_op`.
 *   - Bounded pre-allocated staging buffer (<= 16 MiB) with chunked upload/download.
 *   - Pre-allocated 8-slot `(VkCommandPool, VkCommandBuffer, VkFence)` ring for 0-alloc
 *     batched (`submit_batch`) asynchronous execution with mutex-free `vkWaitForFences`.
 *   - Split-K partial buffer pre-allocated during `prepare_op` (0 B hot-path allocs).
 */

/* Push-constant layouts matching `src/kernels/slang/vk/` shaders (`-fvk-use-c-layout`). */
typedef struct {
    uint64_t x, w, y;
    uint32_t rows, dim, x_stride, y_stride;
    float eps;
    uint32_t has_weight, unit_offset, pad;
} vk_rmsnorm_args;
static_assert(sizeof(vk_rmsnorm_args) == 56, "RmsNormArgs size mismatch");

typedef struct {
    uint64_t residual_inout, x, w, y;
    uint32_t rows, dim, r_stride, x_stride, y_stride;
    float eps;
    uint32_t has_weight, unit_offset;
} vk_fused_add_rmsnorm_args;
static_assert(sizeof(vk_fused_add_rmsnorm_args) == 64, "FusedAddRmsNormArgs size mismatch");

typedef struct {
    uint64_t x, w, b, y;
    uint32_t rows, dim, x_stride, y_stride;
    float eps;
    uint32_t has_weight, has_bias, pad;
} vk_layernorm_args;
static_assert(sizeof(vk_layernorm_args) == 64, "LayerNormArgs size mismatch");

typedef struct {
    uint64_t x, cos_tbl, sin_tbl, y;
    uint32_t n_tokens, n_heads, head_dim, rotary_dim, pos_offset;
    uint32_t tok_sx, head_sx, tok_sy, head_sy, mode;
} vk_rope_args;
static_assert(sizeof(vk_rope_args) == 72, "RopeArgs size mismatch");

typedef struct {
    uint64_t x, y;
    uint32_t count, pad;
} vk_activation_args;
static_assert(sizeof(vk_activation_args) == 24, "ActivationArgs size mismatch");

typedef struct {
    uint64_t gate, up, y;
    uint32_t count;
    float clamp_limit;
} vk_gated_mlp_args;
static_assert(sizeof(vk_gated_mlp_args) == 32, "GatedMlpArgs size mismatch");

typedef struct {
    uint64_t a, b, y;
    uint32_t count;
    float alpha, beta;
    uint32_t has_b, pad;
} vk_vec_axpby_args;
static_assert(sizeof(vk_vec_axpby_args) == 48, "VecAxpbyArgs size mismatch");

typedef struct {
    uint64_t a, b, y;
    uint32_t count, pad;
} vk_vec_mul_args;
static_assert(sizeof(vk_vec_mul_args) == 32, "VecMulArgs size mismatch");

typedef struct {
    uint64_t x, y;
    uint32_t rows, cols, x_stride, y_stride;
} vk_softmax_args;
static_assert(sizeof(vk_softmax_args) == 32, "SoftmaxArgs size mismatch");

typedef struct {
    uint64_t x, suh, y;
    uint32_t rows, cols, block_size, x_stride, y_stride, suh_mode;
    float inv_sqrt_b;
    uint32_t pad;
} vk_fwht_args;
static_assert(sizeof(vk_fwht_args) == 56, "FwhtArgs size mismatch");

typedef struct {
    uint64_t x, x_i8, scale_out;
    uint32_t rows, k, x_stride, xi8_stride;
} vk_quantize_row_args;
static_assert(sizeof(vk_quantize_row_args) == 40, "QuantizeRowArgs size mismatch");

typedef struct {
    uint64_t x, w, bias, x_scale, w_scale, y;
    uint32_t rows, m, k, x_stride, w_stride, y_stride, flags, split_k;
} vk_linear_args;
static_assert(sizeof(vk_linear_args) == 80, "GemvArgs/CoopGemmArgs size mismatch");

typedef struct {
    uint64_t partials, bias, x_scale, w_scale, y;
    uint32_t rows, m, part_stride_u4, y_stride, split_k, flags, is_int32, pad;
} vk_splitk_reduce_args;
static_assert(sizeof(vk_splitk_reduce_args) == 72, "SplitKReduceArgs size mismatch");

/* ---- Capability Bitmask & Table-Driven Pipelines ---- */

enum {
    VK_CAP_NONE     = 0U,
    VK_CAP_INT16    = 1U << 0,
    VK_CAP_F16_I8   = 1U << 1,
    VK_CAP_INT_DOT  = 1U << 2,
    VK_CAP_F32_GEMM = 1U << 3,
    VK_CAP_BF16_EXT = 1U << 4,
    VK_CAP_CM_F16   = 1U << 5,
    VK_CAP_CM_BF16  = 1U << 6,
    VK_CAP_CM_I8    = 1U << 7
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
    PIPE_GEMV_F16,
    PIPE_GEMV_BF16,
    PIPE_GEMV_W8A8,
    PIPE_GEMM_F32,
    PIPE_SPLITK_REDUCE,
    PIPE_GEMM_COOPMAT_F16,
    PIPE_GEMM_COOPMAT_BF16,
    PIPE_GEMM_COOPMAT_W8A8,
    PIPE_GEMM_COOPMAT_F16_64,
    PIPE_GEMM_COOPMAT_BF16_64,
    PIPE_GEMM_COOPMAT_W8A8_64,
    PIPE_COUNT
};

typedef struct {
    const uint32_t *code;
    size_t bytes;
    uint32_t req_caps;
} vk_pipe_spec;

static const vk_pipe_spec k_pipes[PIPE_COUNT] = {
    [PIPE_RMSNORM_F32]           = {spv_vk_rmsnorm_f32, spv_vk_rmsnorm_f32_bytes, VK_CAP_NONE},
    [PIPE_RMSNORM_BF16]          = {spv_vk_rmsnorm_bf16, spv_vk_rmsnorm_bf16_bytes, VK_CAP_INT16},
    [PIPE_RMSNORM_F16]           = {spv_vk_rmsnorm_f16, spv_vk_rmsnorm_f16_bytes, VK_CAP_F16_I8},
    [PIPE_FUSED_ADD_RMSNORM_F32] = {spv_vk_fused_add_rmsnorm_f32, spv_vk_fused_add_rmsnorm_f32_bytes, VK_CAP_NONE},
    [PIPE_LAYERNORM_F32]         = {spv_vk_layernorm_f32, spv_vk_layernorm_f32_bytes, VK_CAP_NONE},
    [PIPE_ROPE_F32]              = {spv_vk_rope_f32, spv_vk_rope_f32_bytes, VK_CAP_NONE},
    [PIPE_ROPE_BF16]             = {spv_vk_rope_bf16, spv_vk_rope_bf16_bytes, VK_CAP_INT16},
    [PIPE_ACT_SILU_F32]          = {spv_vk_activation_silu_f32, spv_vk_activation_silu_f32_bytes, VK_CAP_NONE},
    [PIPE_ACT_GELU_F32]          = {spv_vk_activation_gelu_f32, spv_vk_activation_gelu_f32_bytes, VK_CAP_NONE},
    [PIPE_ACT_RELU2_F32]         = {spv_vk_activation_relu2_f32, spv_vk_activation_relu2_f32_bytes, VK_CAP_NONE},
    [PIPE_GATED_SILU_F32]        = {spv_vk_gated_silu_f32, spv_vk_gated_silu_f32_bytes, VK_CAP_NONE},
    [PIPE_GATED_GELU_F32]        = {spv_vk_gated_gelu_f32, spv_vk_gated_gelu_f32_bytes, VK_CAP_NONE},
    [PIPE_VEC_AXPBY_F32]         = {spv_vk_vec_axpby_f32, spv_vk_vec_axpby_f32_bytes, VK_CAP_NONE},
    [PIPE_VEC_MUL_F32]           = {spv_vk_vec_mul_f32, spv_vk_vec_mul_f32_bytes, VK_CAP_NONE},
    [PIPE_SOFTMAX_F32]           = {spv_vk_softmax_f32, spv_vk_softmax_f32_bytes, VK_CAP_NONE},
    [PIPE_FWHT_F32]              = {spv_vk_fwht_f32, spv_vk_fwht_f32_bytes, VK_CAP_F16_I8},
    [PIPE_QUANTIZE_ROW_F32_I8]   = {spv_vk_quantize_row_f32_i8, spv_vk_quantize_row_f32_i8_bytes, VK_CAP_F16_I8},
    [PIPE_GEMV_F32]              = {spv_vk_gemv_f32, spv_vk_gemv_f32_bytes, VK_CAP_NONE},
    [PIPE_GEMV_F16]              = {spv_vk_gemv_f16, spv_vk_gemv_f16_bytes, VK_CAP_F16_I8},
    [PIPE_GEMV_BF16]             = {spv_vk_gemv_bf16, spv_vk_gemv_bf16_bytes, VK_CAP_INT16},
    [PIPE_GEMV_W8A8]             = {spv_vk_gemv_w8a8, spv_vk_gemv_w8a8_bytes, VK_CAP_F16_I8 | VK_CAP_INT_DOT},
    [PIPE_GEMM_F32]              = {spv_vk_gemm_f32, spv_vk_gemm_f32_bytes, VK_CAP_F32_GEMM},
    [PIPE_SPLITK_REDUCE]         = {spv_vk_splitk_reduce, spv_vk_splitk_reduce_bytes, VK_CAP_NONE},
    [PIPE_GEMM_COOPMAT_F16]      = {spv_vk_gemm_coopmat_f16, spv_vk_gemm_coopmat_f16_bytes, VK_CAP_CM_F16},
    [PIPE_GEMM_COOPMAT_BF16]     = {spv_vk_gemm_coopmat_bf16, spv_vk_gemm_coopmat_bf16_bytes, VK_CAP_CM_BF16},
    [PIPE_GEMM_COOPMAT_W8A8]     = {spv_vk_gemm_coopmat_w8a8, spv_vk_gemm_coopmat_w8a8_bytes, VK_CAP_CM_I8},
    [PIPE_GEMM_COOPMAT_F16_64]   = {spv_vk_gemm_coopmat_f16_64, spv_vk_gemm_coopmat_f16_64_bytes, VK_CAP_CM_F16},
    [PIPE_GEMM_COOPMAT_BF16_64]  = {spv_vk_gemm_coopmat_bf16_64, spv_vk_gemm_coopmat_bf16_64_bytes, VK_CAP_CM_BF16},
    [PIPE_GEMM_COOPMAT_W8A8_64]  = {spv_vk_gemm_coopmat_w8a8_64, spv_vk_gemm_coopmat_w8a8_64_bytes, VK_CAP_CM_I8},
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
} vk_buf;

enum {
    VK_CMD_RING_SIZE = 8,
    VK_STAGE_MAX_BYTES = 16U * 1024U * 1024U
};

struct velri_completion {
    VkCommandPool pool;
    VkCommandBuffer cmd;
    VkFence fence;
    bool in_use;
    bool uses_splitk;
};

struct velri_prepared_op {
    int pipe_id;
    int reduce_pipe_id;
    uint32_t split_k;
    size_t splitk_bytes;
};

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

    /* Shared Split-K partial buffer (pre-allocated in `prepare_op`). */
    vk_buf *splitk_buf;
};

static inline uint32_t div_ceil_u32(uint32_t a, uint32_t b) {
    return a ? ((a - 1U) / b + 1U) : 0U;
}

static bool has_ext(const VkExtensionProperties *exts, uint32_t count, const char *name) {
    for (uint32_t i = 0U; i < count; ++i) {
        if (strcmp(exts[i].extensionName, name) == 0) {
            return true;
        }
    }
    return false;
}

static uint32_t find_mem_type(const VkPhysicalDeviceMemoryProperties *mp, uint32_t type_bits,
                              VkMemoryPropertyFlags req) {
    for (uint32_t i = 0U; i < mp->memoryTypeCount; ++i) {
        if ((type_bits & (1U << i)) && ((mp->memoryTypes[i].propertyFlags & req) == req)) {
            return i;
        }
    }
    return UINT32_MAX;
}

static void vk_inst_destroy(vk_inst_loader *L) {
    if (L->instance && L->vkDestroyInstance) {
        L->vkDestroyInstance(L->instance, nullptr);
    }
    velri_os_dlclose(L->dl);
    memset(L, 0, sizeof(*L));
}

static velri_status vk_inst_init(vk_inst_loader *L) {
    memset(L, 0, sizeof(*L));
    L->dl = velri_os_dlopen_vulkan();
    if (!L->dl) {
        return VELRI_ERR_UNSUPPORTED;
    }
    void *sym = velri_os_dlsym(L->dl, "vkGetInstanceProcAddr");
    if (!sym) {
        vk_inst_destroy(L);
        return VELRI_ERR_UNSUPPORTED;
    }
    memcpy(&L->vkGetInstanceProcAddr, &sym, sizeof(L->vkGetInstanceProcAddr));

    PFN_vkVoidFunction p_ci = L->vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance");
    memcpy(&L->vkCreateInstance, &p_ci, sizeof(L->vkCreateInstance));
    if (!L->vkCreateInstance) {
        vk_inst_destroy(L);
        return VELRI_ERR_UNSUPPORTED;
    }

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "velri-lib",
        .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
        .pEngineName = "Velri",
        .engineVersion = VK_MAKE_VERSION(0, 1, 0),
        .apiVersion = VK_API_VERSION_1_3,
    };
    VkInstanceCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
    };
    if (L->vkCreateInstance(&ci, nullptr, &L->instance) != VK_SUCCESS) {
        vk_inst_destroy(L);
        return VELRI_ERR_UNSUPPORTED;
    }

    bool ok = true;
#define LOAD_REQ_INST(NAME)                                                      \
    do {                                                                         \
        PFN_vkVoidFunction _p = L->vkGetInstanceProcAddr(L->instance, #NAME);    \
        memcpy(&L->NAME, &_p, sizeof(L->NAME));                                  \
        if (!L->NAME) ok = false;                                                \
    } while (0);
    VK_INST_FUNCS(LOAD_REQ_INST)
#undef LOAD_REQ_INST
    if (!ok) {
        vk_inst_destroy(L);
        return VELRI_ERR_UNSUPPORTED;
    }

    PFN_vkVoidFunction p_cm =
        L->vkGetInstanceProcAddr(L->instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    memcpy(&L->vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR, &p_cm,
           sizeof(L->vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR));
    return VELRI_OK;
}

typedef struct {
    VkPhysicalDevice phys;
    uint32_t queue_family;
    bool is_discrete;
} vk_eligible_phys;

static bool probe_phys_device(const vk_inst_loader *L, VkPhysicalDevice phys,
                              uint32_t *out_qf, bool *out_discrete) {
    VkPhysicalDeviceSubgroupProperties sg = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
    };
    VkPhysicalDeviceProperties2 p2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &sg,
    };
    L->vkGetPhysicalDeviceProperties2(phys, &p2);
    if (p2.properties.apiVersion < VK_API_VERSION_1_3) {
        return false;
    }
    if (p2.properties.limits.maxComputeWorkGroupInvocations < 256U) {
        return false;
    }
    VkSubgroupFeatureFlags req_sg = VK_SUBGROUP_FEATURE_BASIC_BIT |
                                    VK_SUBGROUP_FEATURE_ARITHMETIC_BIT |
                                    VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
    if (sg.subgroupSize < 4U || sg.subgroupSize > 128U ||
        (sg.subgroupSize & (sg.subgroupSize - 1U)) != 0U ||
        !(sg.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) ||
        (sg.supportedOperations & req_sg) != req_sg) {
        return false;
    }

    VkPhysicalDeviceVulkan12Features f12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
    };
    VkPhysicalDeviceFeatures2 f2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &f12,
    };
    L->vkGetPhysicalDeviceFeatures2(phys, &f2);
    if (!f12.bufferDeviceAddress || !f12.vulkanMemoryModel ||
        !f12.vulkanMemoryModelDeviceScope || !f12.scalarBlockLayout) {
        return false;
    }

    uint32_t nqf = 0U;
    L->vkGetPhysicalDeviceQueueFamilyProperties(phys, &nqf, nullptr);
    if (nqf == 0U) {
        return false;
    }
    VkQueueFamilyProperties *qfs = (VkQueueFamilyProperties *)calloc(nqf, sizeof(*qfs));
    if (!qfs) {
        return false;
    }
    L->vkGetPhysicalDeviceQueueFamilyProperties(phys, &nqf, qfs);
    uint32_t qf = UINT32_MAX;
    for (uint32_t q = 0U; q < nqf; ++q) {
        if (qfs[q].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            qf = q;
            if (!(qfs[q].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                break;
            }
        }
    }
    free(qfs);
    if (qf == UINT32_MAX) {
        return false;
    }

    *out_qf = qf;
    *out_discrete = (p2.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);
    return true;
}

static velri_status collect_eligible_devices(const vk_inst_loader *L, vk_eligible_phys **out_list,
                                             uint32_t *out_count) {
    *out_list = nullptr;
    *out_count = 0U;
    uint32_t ndev = 0U;
    if (L->vkEnumeratePhysicalDevices(L->instance, &ndev, nullptr) != VK_SUCCESS || ndev == 0U) {
        return VELRI_OK;
    }
    VkPhysicalDevice *raw = (VkPhysicalDevice *)calloc(ndev, sizeof(*raw));
    vk_eligible_phys *elig = (vk_eligible_phys *)calloc(ndev, sizeof(*elig));
    if (!raw || !elig) {
        free(raw);
        free(elig);
        return VELRI_ERR_NOMEM;
    }
    if (L->vkEnumeratePhysicalDevices(L->instance, &ndev, raw) != VK_SUCCESS) {
        free(raw);
        free(elig);
        return VELRI_OK;
    }

    uint32_t n = 0U;
    for (int pass = 0; pass < 2; ++pass) {
        bool want_discrete = (pass == 0);
        for (uint32_t i = 0U; i < ndev; ++i) {
            uint32_t qf = 0U;
            bool disc = false;
            if (probe_phys_device(L, raw[i], &qf, &disc) && disc == want_discrete) {
                elig[n++] = (vk_eligible_phys){
                    .phys = raw[i],
                    .queue_family = qf,
                    .is_discrete = disc,
                };
            }
        }
    }
    free(raw);
    if (n == 0U) {
        free(elig);
        return VELRI_OK;
    }
    *out_list = elig;
    *out_count = n;
    return VELRI_OK;
}

/* ---- Internal Buffer & Transfer Helpers (called under `dev->mtx`) ---- */

static velri_status alloc_buf_locked(velri_backend_device *dev, size_t bytes, size_t align,
                                     vk_buf **out_buf) {
    *out_buf = nullptr;
    if (align != 0U && (align & (align - 1U)) != 0U) {
        return VELRI_ERR_INVALID;
    }
    size_t eff_align = align < 16U ? 16U : align;
    size_t alloc_bytes = bytes ? velri_align_up(bytes, eff_align) : eff_align;
    if (alloc_bytes == SIZE_MAX) {
        return VELRI_ERR_OVERFLOW;
    }

    vk_buf *b = (vk_buf *)calloc(1U, sizeof(*b));
    if (!b) {
        return VELRI_ERR_NOMEM;
    }
    VkBufferCreateInfo b_ci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = (VkDeviceSize)alloc_bytes,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if (dev->vkCreateBuffer(dev->device, &b_ci, nullptr, &b->buffer) != VK_SUCCESS) {
        free(b);
        return VELRI_ERR_NOMEM;
    }

    VkMemoryRequirements mr;
    dev->vkGetBufferMemoryRequirements(dev->device, b->buffer, &mr);
    uint32_t mtype = find_mem_type(&dev->mem_props, mr.memoryTypeBits,
                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mtype == UINT32_MAX) {
        dev->vkDestroyBuffer(dev->device, b->buffer, nullptr);
        free(b);
        return VELRI_ERR_NOMEM;
    }

    VkMemoryAllocateFlagsInfo flags_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
    };
    VkMemoryAllocateInfo ma = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &flags_info,
        .allocationSize = mr.size,
        .memoryTypeIndex = mtype,
    };
    if (dev->vkAllocateMemory(dev->device, &ma, nullptr, &b->memory) != VK_SUCCESS) {
        dev->vkDestroyBuffer(dev->device, b->buffer, nullptr);
        free(b);
        return VELRI_ERR_NOMEM;
    }
    if (dev->vkBindBufferMemory(dev->device, b->buffer, b->memory, 0) != VK_SUCCESS) {
        dev->vkFreeMemory(dev->device, b->memory, nullptr);
        dev->vkDestroyBuffer(dev->device, b->buffer, nullptr);
        free(b);
        return VELRI_ERR_NOMEM;
    }

    VkBufferDeviceAddressInfo bda_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .buffer = b->buffer,
    };
    b->bda = dev->vkGetBufferDeviceAddress(dev->device, &bda_info);
    b->size = alloc_bytes;
    *out_buf = b;
    return VELRI_OK;
}

static void free_buf_locked(velri_backend_device *dev, vk_buf *b) {
    if (!b) return;
    if (b->buffer) dev->vkDestroyBuffer(dev->device, b->buffer, nullptr);
    if (b->memory) dev->vkFreeMemory(dev->device, b->memory, nullptr);
    free(b);
}

static velri_status init_staging_buffer(velri_backend_device *dev) {
    static const size_t k_sizes[] = {
        VK_STAGE_MAX_BYTES, 4U * 1024U * 1024U, 1U * 1024U * 1024U, 256U * 1024U
    };
    for (size_t i = 0U; i < sizeof(k_sizes) / sizeof(k_sizes[0]); ++i) {
        size_t sz = k_sizes[i];
        VkBufferCreateInfo b_ci = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = (VkDeviceSize)sz,
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        if (dev->vkCreateBuffer(dev->device, &b_ci, nullptr, &dev->stage_buf) != VK_SUCCESS) {
            continue;
        }
        VkMemoryRequirements mr;
        dev->vkGetBufferMemoryRequirements(dev->device, dev->stage_buf, &mr);
        uint32_t mtype = find_mem_type(&dev->mem_props, mr.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (mtype == UINT32_MAX) {
            dev->vkDestroyBuffer(dev->device, dev->stage_buf, nullptr);
            dev->stage_buf = VK_NULL_HANDLE;
            return VELRI_ERR_UNSUPPORTED;
        }
        VkMemoryAllocateInfo ma = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = mr.size,
            .memoryTypeIndex = mtype,
        };
        if (dev->vkAllocateMemory(dev->device, &ma, nullptr, &dev->stage_mem) == VK_SUCCESS &&
            dev->vkBindBufferMemory(dev->device, dev->stage_buf, dev->stage_mem, 0) == VK_SUCCESS &&
            dev->vkMapMemory(dev->device, dev->stage_mem, 0, mr.size, 0, &dev->stage_map) ==
                VK_SUCCESS) {
            dev->stage_cap = sz;
            return VELRI_OK;
        }
        if (dev->stage_mem) {
            dev->vkFreeMemory(dev->device, dev->stage_mem, nullptr);
            dev->stage_mem = VK_NULL_HANDLE;
        }
        dev->vkDestroyBuffer(dev->device, dev->stage_buf, nullptr);
        dev->stage_buf = VK_NULL_HANDLE;
    }
    return VELRI_ERR_NOMEM;
}

static velri_status xfer_submit_wait_locked(velri_backend_device *dev) {
    if (dev->vkResetFences(dev->device, 1U, &dev->xfer_fence) != VK_SUCCESS) {
        return VELRI_ERR_EXEC;
    }
    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1U,
        .pCommandBuffers = &dev->xfer_cmd,
    };
    if (dev->vkQueueSubmit(dev->queue, 1U, &si, dev->xfer_fence) != VK_SUCCESS) {
        return VELRI_ERR_EXEC;
    }
    if (dev->vkWaitForFences(dev->device, 1U, &dev->xfer_fence, VK_TRUE, UINT64_MAX) !=
        VK_SUCCESS) {
        return VELRI_ERR_EXEC;
    }
    return VELRI_OK;
}

static VkPipeline get_pipe_locked(velri_backend_device *dev, int id) {
    if (id < 0 || id >= PIPE_COUNT) {
        return VK_NULL_HANDLE;
    }
    if ((dev->caps_mask & k_pipes[id].req_caps) != k_pipes[id].req_caps) {
        return VK_NULL_HANDLE;
    }
    if (dev->pipes[id] != VK_NULL_HANDLE) {
        return dev->pipes[id];
    }
    VkShaderModuleCreateInfo sm_ci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = k_pipes[id].bytes,
        .pCode = k_pipes[id].code,
    };
    VkShaderModule sm = VK_NULL_HANDLE;
    if (dev->vkCreateShaderModule(dev->device, &sm_ci, nullptr, &sm) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    VkComputePipelineCreateInfo p_ci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = sm,
            .pName = "main",
        },
        .layout = dev->pipe_layout,
    };
    VkPipeline pipe = VK_NULL_HANDLE;
    VkResult res =
        dev->vkCreateComputePipelines(dev->device, VK_NULL_HANDLE, 1U, &p_ci, nullptr, &pipe);
    dev->vkDestroyShaderModule(dev->device, sm, nullptr);
    if (res != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    dev->pipes[id] = pipe;
    return pipe;
}

static bool has_pipe_caps(const velri_backend_device *dev, int id) {
    return id >= 0 && id < PIPE_COUNT &&
           (dev->caps_mask & k_pipes[id].req_caps) == k_pipes[id].req_caps;
}

/* ---- 1. Registration ---- */

static velri_status vk_enumerate_devices(uint32_t *out_count) {
    if (!out_count) {
        return VELRI_ERR_INVALID;
    }
    *out_count = 0U;
    vk_inst_loader L;
    if (vk_inst_init(&L) != VELRI_OK) {
        return VELRI_OK; /* No Vulkan runtime installed => 0 devices */
    }
    vk_eligible_phys *list = nullptr;
    uint32_t count = 0U;
    velri_status st = collect_eligible_devices(&L, &list, &count);
    free(list);
    vk_inst_destroy(&L);
    if (st == VELRI_OK) {
        *out_count = count;
    }
    return st;
}

/* ---- 2. Device ---- */

static void vk_close_device(velri_backend_device *dev) {
    if (!dev) {
        return;
    }
    if (dev->device) {
        if (dev->queue && dev->vkQueueWaitIdle) {
            dev->vkQueueWaitIdle(dev->queue);
        }
        free_buf_locked(dev, dev->splitk_buf);
        dev->splitk_buf = nullptr;
        if (dev->stage_map && dev->vkUnmapMemory) {
            dev->vkUnmapMemory(dev->device, dev->stage_mem);
        }
        if (dev->stage_buf && dev->vkDestroyBuffer) {
            dev->vkDestroyBuffer(dev->device, dev->stage_buf, nullptr);
        }
        if (dev->stage_mem && dev->vkFreeMemory) {
            dev->vkFreeMemory(dev->device, dev->stage_mem, nullptr);
        }
        for (int i = 0; i < PIPE_COUNT; ++i) {
            if (dev->pipes[i] && dev->vkDestroyPipeline) {
                dev->vkDestroyPipeline(dev->device, dev->pipes[i], nullptr);
            }
        }
        if (dev->pipe_layout && dev->vkDestroyPipelineLayout) {
            dev->vkDestroyPipelineLayout(dev->device, dev->pipe_layout, nullptr);
        }
        for (int i = 0; i < VK_CMD_RING_SIZE; ++i) {
            if (dev->ring[i].fence && dev->vkDestroyFence) {
                dev->vkDestroyFence(dev->device, dev->ring[i].fence, nullptr);
            }
            if (dev->ring[i].pool && dev->vkDestroyCommandPool) {
                dev->vkDestroyCommandPool(dev->device, dev->ring[i].pool, nullptr);
            }
        }
        if (dev->xfer_fence && dev->vkDestroyFence) {
            dev->vkDestroyFence(dev->device, dev->xfer_fence, nullptr);
        }
        if (dev->xfer_pool && dev->vkDestroyCommandPool) {
            dev->vkDestroyCommandPool(dev->device, dev->xfer_pool, nullptr);
        }
        if (dev->vkDestroyDevice) {
            dev->vkDestroyDevice(dev->device, nullptr);
        }
    }
    velri_os_cond_destroy(dev->slot_cv);
    velri_os_mutex_destroy(dev->mtx);
    vk_inst_destroy(&dev->inst);
    free(dev);
}

static velri_status vk_open_device(uint32_t local_index, const void *cfg,
                                   velri_backend_device **out_dev) {
    (void)cfg;
    if (!out_dev) {
        return VELRI_ERR_INVALID;
    }
    *out_dev = nullptr;

    velri_backend_device *dev = (velri_backend_device *)calloc(1U, sizeof(*dev));
    if (!dev) {
        return VELRI_ERR_NOMEM;
    }
    velri_status st = vk_inst_init(&dev->inst);
    if (st != VELRI_OK) {
        free(dev);
        return st;
    }

    vk_eligible_phys *list = nullptr;
    uint32_t count = 0U;
    st = collect_eligible_devices(&dev->inst, &list, &count);
    if (st != VELRI_OK || local_index >= count) {
        free(list);
        vk_close_device(dev);
        return (st != VELRI_OK) ? st : VELRI_ERR_NOT_FOUND;
    }
    dev->phys = list[local_index].phys;
    dev->queue_family = list[local_index].queue_family;
    dev->local_index = local_index;
    free(list);

    VkPhysicalDeviceSubgroupProperties sg_props = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
    };
    VkPhysicalDeviceProperties2 props2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &sg_props,
    };
    dev->inst.vkGetPhysicalDeviceProperties2(dev->phys, &props2);
    dev->inst.vkGetPhysicalDeviceMemoryProperties(dev->phys, &dev->mem_props);

    dev->subgroup_size = sg_props.subgroupSize;
    dev->max_wg_count[0] = props2.properties.limits.maxComputeWorkGroupCount[0];
    dev->max_wg_count[1] = props2.properties.limits.maxComputeWorkGroupCount[1];
    dev->max_wg_count[2] = props2.properties.limits.maxComputeWorkGroupCount[2];
    uint32_t max_smem = props2.properties.limits.maxComputeSharedMemorySize;

    /* Populate honest `velri_device_caps`. */
    memset(&dev->caps, 0, sizeof(dev->caps));
    snprintf(dev->caps.backend_name, sizeof(dev->caps.backend_name), "%s", "vulkan");
    snprintf(dev->caps.device_name, sizeof(dev->caps.device_name), "%.127s",
             props2.properties.deviceName);
    dev->caps.local_index = local_index;
    dev->caps.is_host_memory = false;

    bool all_dev_local_host_vis = true;
    bool has_dev_local = false;
    for (uint32_t i = 0U; i < dev->mem_props.memoryTypeCount; ++i) {
        VkMemoryPropertyFlags pf = dev->mem_props.memoryTypes[i].propertyFlags;
        if (pf & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
            has_dev_local = true;
            if (!(pf & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                all_dev_local_host_vis = false;
            }
        }
    }
    dev->caps.is_uma =
        (props2.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) ||
        (has_dev_local && all_dev_local_host_vis);

    uint64_t total_vram = 0ULL;
    for (uint32_t i = 0U; i < dev->mem_props.memoryHeapCount; ++i) {
        if (dev->mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            uint64_t hsz = (uint64_t)dev->mem_props.memoryHeaps[i].size;
            total_vram = (hsz > UINT64_MAX - total_vram) ? UINT64_MAX : (total_vram + hsz);
        }
    }
    dev->caps.mem_total_known = (total_vram > 0ULL);
    dev->caps.mem_total_bytes =
        (total_vram > (uint64_t)SIZE_MAX) ? SIZE_MAX : (size_t)total_vram;

    size_t min_align = 256U;
    size_t drv_align = (size_t)props2.properties.limits.minStorageBufferOffsetAlignment;
    while (min_align < drv_align && min_align < (1U << 20)) {
        min_align <<= 1U;
    }
    dev->caps.min_align_bytes = min_align;

    /* Query optional extensions & features. */
    uint32_t nexts = 0U;
    dev->inst.vkEnumerateDeviceExtensionProperties(dev->phys, nullptr, &nexts, nullptr);
    VkExtensionProperties *exts =
        (VkExtensionProperties *)calloc(nexts ? nexts : 1U, sizeof(*exts));
    if (!exts) {
        vk_close_device(dev);
        return VELRI_ERR_NOMEM;
    }
    if (nexts > 0U) {
        dev->inst.vkEnumerateDeviceExtensionProperties(dev->phys, nullptr, &nexts, exts);
    }
    bool ext_coopmat = has_ext(exts, nexts, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
#ifdef VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME
    bool ext_bf16 = has_ext(exts, nexts, VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME);
    VkPhysicalDeviceShaderBfloat16FeaturesKHR f_bf16 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR,
    };
#else
    bool ext_bf16 = false;
#endif
    free(exts);

    VkPhysicalDeviceCooperativeMatrixFeaturesKHR f_cm = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
    };
    VkPhysicalDeviceVulkan13Features f13 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
    };
    VkPhysicalDeviceVulkan12Features f12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &f13,
    };
    VkPhysicalDeviceVulkan11Features f11 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        .pNext = &f12,
    };
    void **tail_pnext = &f13.pNext;
    if (ext_coopmat) {
        *tail_pnext = &f_cm;
        tail_pnext = &f_cm.pNext;
    }
#ifdef VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME
    if (ext_bf16) {
        *tail_pnext = &f_bf16;
    }
#else
    (void)ext_bf16;
#endif
    VkPhysicalDeviceFeatures2 f2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &f11,
    };
    dev->inst.vkGetPhysicalDeviceFeatures2(dev->phys, &f2);

    if (f2.features.shaderInt16 && f11.storageBuffer16BitAccess) {
        dev->caps_mask |= VK_CAP_INT16;
    }
    if ((dev->caps_mask & VK_CAP_INT16) && f12.shaderFloat16 && f12.shaderInt8 &&
        f12.storageBuffer8BitAccess) {
        dev->caps_mask |= VK_CAP_F16_I8;
    }
    if (f13.shaderIntegerDotProduct) {
        dev->caps_mask |= VK_CAP_INT_DOT;
    }
    if (max_smem >= 16896U) {
        dev->caps_mask |= VK_CAP_F32_GEMM;
    }
#ifdef VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME
    if (ext_bf16 && f_bf16.shaderBFloat16Type) {
        dev->caps_mask |= VK_CAP_BF16_EXT;
    }
#endif

    bool has_coopmat = (ext_coopmat && f_cm.cooperativeMatrix == VK_TRUE);
    if (has_coopmat && dev->subgroup_size == 32U && max_smem >= 20480U &&
        (dev->caps_mask & VK_CAP_F16_I8) &&
        dev->inst.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR) {
        uint32_t ncm = 0U;
        dev->inst.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(dev->phys, &ncm, nullptr);
        VkCooperativeMatrixPropertiesKHR *cm_props =
            (VkCooperativeMatrixPropertiesKHR *)calloc(ncm ? ncm : 1U, sizeof(*cm_props));
        if (cm_props && ncm > 0U) {
            for (uint32_t i = 0U; i < ncm; ++i) {
                cm_props[i].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
            }
            dev->inst.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(dev->phys, &ncm, cm_props);
            for (uint32_t i = 0U; i < ncm; ++i) {
                const VkCooperativeMatrixPropertiesKHR *p = &cm_props[i];
                if (p->scope != VK_SCOPE_SUBGROUP_KHR || p->saturatingAccumulation) continue;
                if (p->MSize == 16U && p->NSize == 16U && p->KSize == 16U &&
                    p->AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                    p->BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                    p->CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                    p->ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR) {
                    dev->caps_mask |= VK_CAP_CM_F16;
                }
#ifdef VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME
                if (p->MSize == 16U && p->NSize == 16U && p->KSize == 16U &&
                    p->AType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                    p->BType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                    p->CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                    p->ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR && ext_bf16 &&
                    f_bf16.shaderBFloat16CooperativeMatrix == VK_TRUE) {
                    dev->caps_mask |= VK_CAP_CM_BF16;
                }
#endif
                if (p->MSize == 16U && p->NSize == 16U && p->KSize == 32U &&
                    p->AType == VK_COMPONENT_TYPE_SINT8_KHR &&
                    p->BType == VK_COMPONENT_TYPE_SINT8_KHR &&
                    p->CType == VK_COMPONENT_TYPE_SINT32_KHR &&
                    p->ResultType == VK_COMPONENT_TYPE_SINT32_KHR) {
                    dev->caps_mask |= VK_CAP_CM_I8;
                }
            }
        }
        free(cm_props);
    }

    bool enable_cm = (dev->caps_mask & (VK_CAP_CM_F16 | VK_CAP_CM_BF16 | VK_CAP_CM_I8)) != 0U;
    const char *en_exts[4];
    uint32_t n_en_exts = 0U;
    if (enable_cm) {
        en_exts[n_en_exts++] = VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME;
    }
#ifdef VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME
    bool enable_bf16 = (dev->caps_mask & (VK_CAP_BF16_EXT | VK_CAP_CM_BF16)) != 0U;
    if (enable_bf16) {
        en_exts[n_en_exts++] = VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME;
    }
    VkPhysicalDeviceShaderBfloat16FeaturesKHR en_bf16 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR,
        .shaderBFloat16Type = (dev->caps_mask & VK_CAP_BF16_EXT) ? VK_TRUE : VK_FALSE,
        .shaderBFloat16CooperativeMatrix = (dev->caps_mask & VK_CAP_CM_BF16) ? VK_TRUE : VK_FALSE,
    };
    void *bf16_next = enable_bf16 ? (void *)&en_bf16 : nullptr;
#else
    void *bf16_next = nullptr;
#endif
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR en_cm = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
        .pNext = bf16_next,
        .cooperativeMatrix = enable_cm ? VK_TRUE : VK_FALSE,
    };
    VkPhysicalDeviceVulkan13Features en13 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = enable_cm ? (void *)&en_cm : bf16_next,
        .shaderIntegerDotProduct = (dev->caps_mask & VK_CAP_INT_DOT) ? VK_TRUE : VK_FALSE,
    };
    VkPhysicalDeviceVulkan12Features en12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &en13,
        .storageBuffer8BitAccess = f12.storageBuffer8BitAccess,
        .shaderFloat16 = f12.shaderFloat16,
        .shaderInt8 = f12.shaderInt8,
        .scalarBlockLayout = VK_TRUE,
        .vulkanMemoryModel = VK_TRUE,
        .vulkanMemoryModelDeviceScope = VK_TRUE,
        .bufferDeviceAddress = VK_TRUE,
    };
    VkPhysicalDeviceVulkan11Features en11 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        .pNext = &en12,
        .storageBuffer16BitAccess = f11.storageBuffer16BitAccess,
    };
    VkPhysicalDeviceFeatures2 en2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &en11,
        .features = {
            .shaderInt16 = f2.features.shaderInt16,
        },
    };

    float qprio = 1.0f;
    VkDeviceQueueCreateInfo q_ci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = dev->queue_family,
        .queueCount = 1U,
        .pQueuePriorities = &qprio,
    };
    VkDeviceCreateInfo dev_ci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &en2,
        .queueCreateInfoCount = 1U,
        .pQueueCreateInfos = &q_ci,
        .enabledExtensionCount = n_en_exts,
        .ppEnabledExtensionNames = en_exts,
    };
    if (dev->inst.vkCreateDevice(dev->phys, &dev_ci, nullptr, &dev->device) != VK_SUCCESS) {
        vk_close_device(dev);
        return VELRI_ERR_UNSUPPORTED;
    }

    bool dev_ok = true;
#define LOAD_REQ_DEV(NAME)                                                          \
    do {                                                                            \
        PFN_vkVoidFunction _p = dev->inst.vkGetDeviceProcAddr(dev->device, #NAME);  \
        memcpy(&dev->NAME, &_p, sizeof(dev->NAME));                                 \
        if (!dev->NAME) dev_ok = false;                                             \
    } while (0);
    VK_DEV_FUNCS(LOAD_REQ_DEV)
#undef LOAD_REQ_DEV
    if (!dev_ok) {
        vk_close_device(dev);
        return VELRI_ERR_UNSUPPORTED;
    }

    dev->vkGetDeviceQueue(dev->device, dev->queue_family, 0U, &dev->queue);

    if (!velri_os_mutex_create(&dev->mtx) || !velri_os_cond_create(&dev->slot_cv)) {
        vk_close_device(dev);
        return VELRI_ERR_NOMEM;
    }

    VkCommandPoolCreateInfo cp_ci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = dev->queue_family,
    };
    VkFenceCreateInfo fn_ci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};

    /* Dedicated transfer pool + command buffer + fence. */
    if (dev->vkCreateCommandPool(dev->device, &cp_ci, nullptr, &dev->xfer_pool) != VK_SUCCESS) {
        vk_close_device(dev);
        return VELRI_ERR_NOMEM;
    }
    VkCommandBufferAllocateInfo xcb_ai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = dev->xfer_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1U,
    };
    if (dev->vkAllocateCommandBuffers(dev->device, &xcb_ai, &dev->xfer_cmd) != VK_SUCCESS ||
        dev->vkCreateFence(dev->device, &fn_ci, nullptr, &dev->xfer_fence) != VK_SUCCESS) {
        vk_close_device(dev);
        return VELRI_ERR_NOMEM;
    }

    /* Pre-allocated async compute ring (each slot owns its own VkCommandPool). */
    for (int i = 0; i < VK_CMD_RING_SIZE; ++i) {
        if (dev->vkCreateCommandPool(dev->device, &cp_ci, nullptr, &dev->ring[i].pool) !=
            VK_SUCCESS) {
            vk_close_device(dev);
            return VELRI_ERR_NOMEM;
        }
        VkCommandBufferAllocateInfo cb_ai = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = dev->ring[i].pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1U,
        };
        if (dev->vkAllocateCommandBuffers(dev->device, &cb_ai, &dev->ring[i].cmd) != VK_SUCCESS ||
            dev->vkCreateFence(dev->device, &fn_ci, nullptr, &dev->ring[i].fence) != VK_SUCCESS) {
            vk_close_device(dev);
            return VELRI_ERR_NOMEM;
        }
    }

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0U,
        .size = 128U,
    };
    VkPipelineLayoutCreateInfo pl_ci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pushConstantRangeCount = 1U,
        .pPushConstantRanges = &pcr,
    };
    if (dev->vkCreatePipelineLayout(dev->device, &pl_ci, nullptr, &dev->pipe_layout) !=
        VK_SUCCESS) {
        vk_close_device(dev);
        return VELRI_ERR_NOMEM;
    }

    st = init_staging_buffer(dev);
    if (st != VELRI_OK) {
        vk_close_device(dev);
        return st;
    }

    *out_dev = dev;
    return VELRI_OK;
}

static velri_status vk_query_caps(const velri_backend_device *dev, velri_device_caps *out_caps) {
    if (!dev || !out_caps) {
        return VELRI_ERR_INVALID;
    }
    *out_caps = dev->caps;
    return VELRI_OK;
}

/* ---- 3. Memory ---- */

static velri_status vk_alloc(velri_backend_device *dev, size_t bytes, size_t align,
                             void **out_mem_handle) {
    if (!dev || !out_mem_handle) {
        return VELRI_ERR_INVALID;
    }
    vk_buf *buf = nullptr;
    velri_os_mutex_lock(dev->mtx);
    velri_status st = alloc_buf_locked(dev, bytes, align, &buf);
    velri_os_mutex_unlock(dev->mtx);
    if (st != VELRI_OK) {
        return st;
    }
    *out_mem_handle = buf;
    return VELRI_OK;
}

static void vk_free(velri_backend_device *dev, void *mem_handle) {
    if (!dev || !mem_handle) {
        return;
    }
    velri_os_mutex_lock(dev->mtx);
    free_buf_locked(dev, (vk_buf *)mem_handle);
    velri_os_mutex_unlock(dev->mtx);
}

static velri_status vk_copy_from_ram(velri_backend_device *dev, const velri_buffer_binding *dst,
                                     const void *src_ram, size_t bytes) {
    if (!dev || !dst || !dst->mem_handle || (!src_ram && bytes > 0U)) {
        return VELRI_ERR_INVALID;
    }
    const vk_buf *vb = (const vk_buf *)dst->mem_handle;
    if (dst->byte_offset > vb->size || bytes > vb->size - dst->byte_offset) {
        return VELRI_ERR_INVALID;
    }
    if (bytes == 0U) {
        return VELRI_OK;
    }

    velri_os_mutex_lock(dev->mtx);
    velri_status st = VELRI_OK;
    const uint8_t *src = (const uint8_t *)src_ram;
    size_t done = 0U;
    while (done < bytes) {
        size_t chunk = bytes - done;
        if (chunk > dev->stage_cap) {
            chunk = dev->stage_cap;
        }
        memcpy(dev->stage_map, src + done, chunk);
        dev->vkResetCommandBuffer(dev->xfer_cmd, 0U);
        VkCommandBufferBeginInfo bi = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        if (dev->vkBeginCommandBuffer(dev->xfer_cmd, &bi) != VK_SUCCESS) {
            st = VELRI_ERR_EXEC;
            break;
        }
        VkBufferCopy region = {
            .srcOffset = 0U,
            .dstOffset = (VkDeviceSize)(dst->byte_offset + done),
            .size = (VkDeviceSize)chunk,
        };
        dev->vkCmdCopyBuffer(dev->xfer_cmd, dev->stage_buf, vb->buffer, 1U, &region);
        if (dev->vkEndCommandBuffer(dev->xfer_cmd) != VK_SUCCESS) {
            st = VELRI_ERR_EXEC;
            break;
        }
        st = xfer_submit_wait_locked(dev);
        if (st != VELRI_OK) {
            break;
        }
        done += chunk;
    }
    velri_os_mutex_unlock(dev->mtx);
    return st;
}

static velri_status vk_copy_to_ram(velri_backend_device *dev, void *dst_ram,
                                   const velri_buffer_binding *src, size_t bytes) {
    if (!dev || !src || !src->mem_handle || (!dst_ram && bytes > 0U)) {
        return VELRI_ERR_INVALID;
    }
    const vk_buf *vb = (const vk_buf *)src->mem_handle;
    if (src->byte_offset > vb->size || bytes > vb->size - src->byte_offset) {
        return VELRI_ERR_INVALID;
    }
    if (bytes == 0U) {
        return VELRI_OK;
    }

    velri_os_mutex_lock(dev->mtx);
    velri_status st = VELRI_OK;
    uint8_t *dst = (uint8_t *)dst_ram;
    size_t done = 0U;
    while (done < bytes) {
        size_t chunk = bytes - done;
        if (chunk > dev->stage_cap) {
            chunk = dev->stage_cap;
        }
        dev->vkResetCommandBuffer(dev->xfer_cmd, 0U);
        VkCommandBufferBeginInfo bi = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        if (dev->vkBeginCommandBuffer(dev->xfer_cmd, &bi) != VK_SUCCESS) {
            st = VELRI_ERR_EXEC;
            break;
        }
        VkBufferCopy region = {
            .srcOffset = (VkDeviceSize)(src->byte_offset + done),
            .dstOffset = 0U,
            .size = (VkDeviceSize)chunk,
        };
        dev->vkCmdCopyBuffer(dev->xfer_cmd, vb->buffer, dev->stage_buf, 1U, &region);
        if (dev->vkEndCommandBuffer(dev->xfer_cmd) != VK_SUCCESS) {
            st = VELRI_ERR_EXEC;
            break;
        }
        st = xfer_submit_wait_locked(dev);
        if (st != VELRI_OK) {
            break;
        }
        memcpy(dst + done, dev->stage_map, chunk);
        done += chunk;
    }
    velri_os_mutex_unlock(dev->mtx);
    return st;
}

static velri_status vk_zero_mem(velri_backend_device *dev, void *mem_handle, size_t bytes) {
    if (!dev || !mem_handle) {
        return VELRI_ERR_INVALID;
    }
    const vk_buf *vb = (const vk_buf *)mem_handle;
    if (bytes > vb->size) {
        return VELRI_ERR_INVALID;
    }
    if (bytes == 0U) {
        return VELRI_OK;
    }
    /* `vk_buf` size is always a multiple of 16 bytes; round fill size up to 4 bytes safely. */
    size_t fill_sz = (bytes + 3U) & ~(size_t)3U;
    if (fill_sz > vb->size) {
        fill_sz = vb->size;
    }

    velri_os_mutex_lock(dev->mtx);
    dev->vkResetCommandBuffer(dev->xfer_cmd, 0U);
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    velri_status st = VELRI_OK;
    if (dev->vkBeginCommandBuffer(dev->xfer_cmd, &bi) != VK_SUCCESS) {
        st = VELRI_ERR_EXEC;
    } else {
        dev->vkCmdFillBuffer(dev->xfer_cmd, vb->buffer, 0U, (VkDeviceSize)fill_sz, 0U);
        if (dev->vkEndCommandBuffer(dev->xfer_cmd) != VK_SUCCESS) {
            st = VELRI_ERR_EXEC;
        } else {
            st = xfer_submit_wait_locked(dev);
        }
    }
    velri_os_mutex_unlock(dev->mtx);
    return st;
}

/* ---- 4. Operation Selection & Preparation ---- */

static uint32_t choose_split_k(uint32_t rows, uint32_t m, uint32_t k, uint32_t tile_m,
                               uint32_t tile_n, uint32_t target_wgs, uint32_t bk_elems) {
    uint64_t wgs = (uint64_t)div_ceil_u32(m, tile_n) * (uint64_t)div_ceil_u32(rows, tile_m);
    if (rows < 16U || wgs == 0ULL || wgs >= target_wgs) {
        return 1U;
    }
    uint32_t k_tiles = div_ceil_u32(k, bk_elems);
    if (k_tiles < 16U) {
        return 1U;
    }
    uint32_t sk = (uint32_t)((uint64_t)target_wgs / wgs);
    if (sk > 8U) sk = 8U;
    if (sk > k_tiles / 4U) sk = k_tiles / 4U;
    if (sk < 2U) return 1U;
    uint32_t part_stride = (m + 3U) & ~3U;
    uint64_t part_bytes = (uint64_t)sk * (uint64_t)rows * (uint64_t)part_stride * 4ULL;
    if (part_bytes > 32ULL * 1024ULL * 1024ULL) {
        return 1U;
    }
    return sk;
}

static int select_coopmat_pipe(const velri_backend_device *dev, uint32_t rows, uint32_t m,
                               int gemv_id, int cm64_id, int cm128_id) {
    if (rows <= 2U || !has_pipe_caps(dev, cm128_id)) {
        return has_pipe_caps(dev, gemv_id) ? gemv_id : -1;
    }
    uint64_t tiles128 = (uint64_t)div_ceil_u32(m, 128U) * (uint64_t)div_ceil_u32(rows, 128U);
    return (tiles128 < 140ULL) ? cm64_id : cm128_id;
}

static velri_status resolve_op_pipes(const velri_backend_device *dev, const velri_resolved_op *op,
                                     struct velri_prepared_op *out_plan) {
    out_plan->pipe_id = -1;
    out_plan->reduce_pipe_id = -1;
    out_plan->split_k = 1U;
    out_plan->splitk_bytes = 0U;

    switch (op->kind) {
    case VELRI_OP_LINEAR: {
        const velri_linear_desc *d = &op->raw.attr.linear.desc;
        uint32_t rows = (uint32_t)op->inputs[0].dims[0];
        uint32_t m = (uint32_t)op->inputs[1].dims[0];
        uint32_t k = (uint32_t)op->inputs[0].dims[1];
        if (d->x_dtype == VELRI_DTYPE_F32 && d->w_dtype == VELRI_DTYPE_F32 &&
            d->y_dtype == VELRI_DTYPE_F32) {
            if (rows <= 2U || !has_pipe_caps(dev, PIPE_GEMM_F32)) {
                out_plan->pipe_id = PIPE_GEMV_F32;
            } else {
                out_plan->pipe_id = PIPE_GEMM_F32;
                uint32_t sk = choose_split_k(rows, m, k, 128U, 128U, 140U, 8U);
                if (sk > 1U) {
                    out_plan->split_k = sk;
                    out_plan->reduce_pipe_id = PIPE_SPLITK_REDUCE;
                    uint32_t part_stride = (m + 3U) & ~3U;
                    out_plan->splitk_bytes =
                        (size_t)sk * (size_t)rows * (size_t)part_stride * sizeof(float);
                }
            }
        } else if (d->x_dtype == VELRI_DTYPE_F16 && d->w_dtype == VELRI_DTYPE_F16 &&
                   d->y_dtype == VELRI_DTYPE_F32) {
            out_plan->pipe_id = select_coopmat_pipe(dev, rows, m, PIPE_GEMV_F16,
                                                    PIPE_GEMM_COOPMAT_F16_64,
                                                    PIPE_GEMM_COOPMAT_F16);
        } else if (d->x_dtype == VELRI_DTYPE_BF16 && d->w_dtype == VELRI_DTYPE_BF16 &&
                   d->y_dtype == VELRI_DTYPE_F32) {
            out_plan->pipe_id = select_coopmat_pipe(dev, rows, m, PIPE_GEMV_BF16,
                                                    PIPE_GEMM_COOPMAT_BF16_64,
                                                    PIPE_GEMM_COOPMAT_BF16);
        } else if (d->x_dtype == VELRI_DTYPE_I8 && d->w_dtype == VELRI_DTYPE_I8 &&
                   d->y_dtype == VELRI_DTYPE_F32) {
            out_plan->pipe_id = select_coopmat_pipe(dev, rows, m, PIPE_GEMV_W8A8,
                                                    PIPE_GEMM_COOPMAT_W8A8_64,
                                                    PIPE_GEMM_COOPMAT_W8A8);
        }
        break;
    }
    case VELRI_OP_RMSNORM:
        if (op->inputs[0].dtype == VELRI_DTYPE_F32) out_plan->pipe_id = PIPE_RMSNORM_F32;
        else if (op->inputs[0].dtype == VELRI_DTYPE_BF16) out_plan->pipe_id = PIPE_RMSNORM_BF16;
        else if (op->inputs[0].dtype == VELRI_DTYPE_F16) out_plan->pipe_id = PIPE_RMSNORM_F16;
        break;
    case VELRI_OP_FUSED_ADD_RMSNORM:
        out_plan->pipe_id = PIPE_FUSED_ADD_RMSNORM_F32;
        break;
    case VELRI_OP_LAYERNORM:
        out_plan->pipe_id = PIPE_LAYERNORM_F32;
        break;
    case VELRI_OP_ROPE:
        if (op->inputs[0].dtype == VELRI_DTYPE_F32) out_plan->pipe_id = PIPE_ROPE_F32;
        else if (op->inputs[0].dtype == VELRI_DTYPE_BF16) out_plan->pipe_id = PIPE_ROPE_BF16;
        break;
    case VELRI_OP_ACTIVATION:
        if (op->raw.attr.activation.kind == VELRI_ACT_SILU) out_plan->pipe_id = PIPE_ACT_SILU_F32;
        else if (op->raw.attr.activation.kind == VELRI_ACT_GELU_TANH)
            out_plan->pipe_id = PIPE_ACT_GELU_F32;
        else if (op->raw.attr.activation.kind == VELRI_ACT_RELU2)
            out_plan->pipe_id = PIPE_ACT_RELU2_F32;
        break;
    case VELRI_OP_GATED_MLP:
        if (op->raw.attr.gated_mlp.act == VELRI_ACT_SILU) out_plan->pipe_id = PIPE_GATED_SILU_F32;
        else if (op->raw.attr.gated_mlp.act == VELRI_ACT_GELU_TANH)
            out_plan->pipe_id = PIPE_GATED_GELU_F32;
        break;
    case VELRI_OP_SOFTMAX:
        out_plan->pipe_id = PIPE_SOFTMAX_F32;
        break;
    case VELRI_OP_VEC_AXPBY:
        out_plan->pipe_id = PIPE_VEC_AXPBY_F32;
        break;
    case VELRI_OP_VEC_MUL:
        out_plan->pipe_id = PIPE_VEC_MUL_F32;
        break;
    case VELRI_OP_FWHT:
        out_plan->pipe_id = PIPE_FWHT_F32;
        break;
    case VELRI_OP_QUANTIZE_ROW:
        out_plan->pipe_id = PIPE_QUANTIZE_ROW_F32_I8;
        break;
    default:
        break;
    }

    if (!has_pipe_caps(dev, out_plan->pipe_id)) {
        return VELRI_ERR_UNSUPPORTED;
    }
    if (out_plan->reduce_pipe_id >= 0 && !has_pipe_caps(dev, out_plan->reduce_pipe_id)) {
        return VELRI_ERR_UNSUPPORTED;
    }
    return VELRI_OK;
}

static velri_status vk_query_operation(const velri_backend_device *dev,
                                       const velri_resolved_op *op) {
    if (!dev || !op) {
        return VELRI_ERR_INVALID;
    }
    velri_status st = velri_op_validate(op);
    if (st != VELRI_OK) {
        return st;
    }
    struct velri_prepared_op plan;
    return resolve_op_pipes(dev, op, &plan);
}

static velri_status vk_prepare_op(velri_backend_device *dev, const velri_resolved_op *op,
                                  velri_prepared_op **out_prepared) {
    if (!dev || !op || !out_prepared) {
        return VELRI_ERR_INVALID;
    }
    *out_prepared = nullptr;
    velri_status st = velri_op_validate(op);
    if (st != VELRI_OK) {
        return st;
    }
    struct velri_prepared_op plan;
    st = resolve_op_pipes(dev, op, &plan);
    if (st != VELRI_OK) {
        return st;
    }

    velri_os_mutex_lock(dev->mtx);
    VkPipeline p = get_pipe_locked(dev, plan.pipe_id);
    VkPipeline rp =
        (plan.reduce_pipe_id >= 0) ? get_pipe_locked(dev, plan.reduce_pipe_id) : VK_NULL_HANDLE;
    if (!p || (plan.reduce_pipe_id >= 0 && !rp)) {
        velri_os_mutex_unlock(dev->mtx);
        return VELRI_ERR_UNSUPPORTED;
    }
    if (plan.splitk_bytes > 0U && (!dev->splitk_buf || dev->splitk_buf->size < plan.splitk_bytes)) {
        /* Wait for any in-flight split-K user before resizing shared partial buffer. */
        while (dev->splitk_busy) {
            velri_os_cond_wait(dev->slot_cv, dev->mtx);
        }
        vk_buf *nb = nullptr;
        st = alloc_buf_locked(dev, plan.splitk_bytes, 256U, &nb);
        if (st != VELRI_OK) {
            velri_os_mutex_unlock(dev->mtx);
            return st;
        }
        free_buf_locked(dev, dev->splitk_buf);
        dev->splitk_buf = nb;
    }
    velri_os_mutex_unlock(dev->mtx);

    velri_prepared_op *res = (velri_prepared_op *)calloc(1U, sizeof(*res));
    if (!res) {
        return VELRI_ERR_NOMEM;
    }
    *res = plan;
    *out_prepared = res;
    return VELRI_OK;
}

static void vk_destroy_prepared_op(velri_backend_device *dev, velri_prepared_op *prepared) {
    (void)dev;
    free(prepared);
}

/* ---- 5. Execution (Batched Async Ring) ---- */

typedef struct {
    VkPipeline pipe;
    VkPipeline red_pipe;
    uint32_t gx, gy, gz;
    uint32_t red_gx;
    uint32_t pc_size;
    uint32_t red_pc_size;
    alignas(8) uint8_t pc[128];
    vk_splitk_reduce_args red_pc;
    bool uses_splitk;
} vk_encoded_dispatch;

static bool binding_bda(const velri_buffer_binding *b, uint64_t *out_bda) {
    if (!b || !b->mem_handle) {
        return false;
    }
    const vk_buf *vb = (const vk_buf *)b->mem_handle;
    if (b->byte_offset > vb->size) {
        return false;
    }
    *out_bda = (uint64_t)(vb->bda + (VkDeviceAddress)b->byte_offset);
    return true;
}

static bool grid_fits(const velri_backend_device *dev, uint32_t gx, uint32_t gy, uint32_t gz) {
    return gx > 0U && gy > 0U && gz > 0U && gx <= dev->max_wg_count[0] &&
           gy <= dev->max_wg_count[1] && gz <= dev->max_wg_count[2];
}

static velri_status encode_item_locked(velri_backend_device *dev, const velri_submit_item *item,
                                       vk_encoded_dispatch *out) {
    memset(out, 0, sizeof(*out));
    if (!item || !item->op || item->num_inputs != item->op->num_inputs ||
        item->num_outputs != item->op->num_outputs) {
        return VELRI_ERR_INVALID;
    }
    const velri_resolved_op *op = item->op;

    uint64_t ib[VELRI_OP_MAX_INPUTS] = {0};
    uint64_t ob[VELRI_OP_MAX_OUTPUTS] = {0};
    for (uint32_t i = 0U; i < op->num_inputs; ++i) {
        if (op->input_present[i] && !binding_bda(&item->inputs[i], &ib[i])) {
            return VELRI_ERR_INVALID;
        }
    }
    for (uint32_t o = 0U; o < op->num_outputs; ++o) {
        if (!binding_bda(&item->outputs[o], &ob[o])) {
            return VELRI_ERR_INVALID;
        }
    }

    struct velri_prepared_op plan;
    if (item->prepared) {
        plan = *item->prepared;
    } else {
        velri_status st = resolve_op_pipes(dev, op, &plan);
        if (st != VELRI_OK) {
            return st;
        }
    }

    out->pipe = get_pipe_locked(dev, plan.pipe_id);
    if (!out->pipe) {
        return VELRI_ERR_UNSUPPORTED;
    }

    uint32_t warps_per_wg = 128U / dev->subgroup_size;
    switch (op->kind) {
    case VELRI_OP_LINEAR: {
        uint32_t rows = (uint32_t)op->inputs[0].dims[0];
        uint32_t m = (uint32_t)op->inputs[1].dims[0];
        uint32_t k = (uint32_t)op->inputs[0].dims[1];
        uint32_t xs = (uint32_t)op->inputs[0].strides[0];
        uint32_t ws = (uint32_t)op->inputs[1].strides[0];
        uint32_t ys = (uint32_t)op->outputs[0].strides[0];
        uint32_t flags = (op->input_present[2] ? 1U : 0U) | (op->input_present[3] ? 2U : 0U) |
                         (op->input_present[4] ? 4U : 0U);
        vk_linear_args args = {
            .x = ib[0],
            .w = ib[1],
            .bias = op->input_present[2] ? ib[2] : 0ULL,
            .x_scale = op->input_present[3] ? ib[3] : 0ULL,
            .w_scale = op->input_present[4] ? ib[4] : 0ULL,
            .y = ob[0],
            .rows = rows,
            .m = m,
            .k = k,
            .x_stride = xs,
            .w_stride = ws,
            .y_stride = ys,
            .flags = flags,
            .split_k = 1U,
        };

        if (plan.pipe_id == PIPE_GEMV_F32 || plan.pipe_id == PIPE_GEMV_F16 ||
            plan.pipe_id == PIPE_GEMV_BF16 || plan.pipe_id == PIPE_GEMV_W8A8) {
            out->gx = div_ceil_u32(m, warps_per_wg);
            out->gy = rows;
            out->gz = 1U;
        } else if (plan.pipe_id == PIPE_GEMM_COOPMAT_F16_64 ||
                   plan.pipe_id == PIPE_GEMM_COOPMAT_BF16_64 ||
                   plan.pipe_id == PIPE_GEMM_COOPMAT_W8A8_64) {
            out->gx = div_ceil_u32(m, 64U);
            out->gy = div_ceil_u32(rows, 64U);
            out->gz = 1U;
        } else {
            out->gx = div_ceil_u32(m, 128U);
            out->gy = div_ceil_u32(rows, 128U);
            out->gz = 1U;
        }

        if (plan.split_k > 1U && plan.reduce_pipe_id >= 0) {
            out->red_pipe = get_pipe_locked(dev, plan.reduce_pipe_id);
            if (!out->red_pipe || !dev->splitk_buf || dev->splitk_buf->size < plan.splitk_bytes) {
                return VELRI_ERR_EXEC;
            }
            uint32_t part_stride = (m + 3U) & ~3U;
            args.y = (uint64_t)dev->splitk_buf->bda;
            args.y_stride = part_stride;
            args.split_k = plan.split_k;
            out->gz = plan.split_k;

            out->red_pc = (vk_splitk_reduce_args){
                .partials = (uint64_t)dev->splitk_buf->bda,
                .bias = args.bias,
                .x_scale = args.x_scale,
                .w_scale = args.w_scale,
                .y = ob[0],
                .rows = rows,
                .m = m,
                .part_stride_u4 = part_stride >> 2U,
                .y_stride = ys,
                .split_k = plan.split_k,
                .flags = flags,
                .is_int32 = 0U,
                .pad = 0U,
            };
            uint64_t total_u4 = (uint64_t)rows * (uint64_t)(part_stride >> 2U);
            uint64_t rgx = (total_u4 + 255ULL) / 256ULL;
            if (rgx == 0ULL || rgx > dev->max_wg_count[0]) {
                return VELRI_ERR_OVERFLOW;
            }
            out->red_gx = (uint32_t)rgx;
            out->red_pc_size = (uint32_t)sizeof(out->red_pc);
            out->uses_splitk = true;
        }
        memcpy(out->pc, &args, sizeof(args));
        out->pc_size = (uint32_t)sizeof(args);
        break;
    }

    case VELRI_OP_RMSNORM: {
        vk_rmsnorm_args args = {
            .x = ib[0],
            .w = op->input_present[1] ? ib[1] : 0ULL,
            .y = ob[0],
            .rows = (uint32_t)op->inputs[0].dims[0],
            .dim = (uint32_t)op->inputs[0].dims[1],
            .x_stride = (uint32_t)op->inputs[0].strides[0],
            .y_stride = (uint32_t)op->outputs[0].strides[0],
            .eps = op->raw.attr.rmsnorm.eps,
            .has_weight = op->input_present[1] ? 1U : 0U,
            .unit_offset = op->raw.attr.rmsnorm.unit_offset ? 1U : 0U,
            .pad = 0U,
        };
        memcpy(out->pc, &args, sizeof(args));
        out->pc_size = (uint32_t)sizeof(args);
        out->gx = args.rows;
        out->gy = 1U;
        out->gz = 1U;
        break;
    }

    case VELRI_OP_FUSED_ADD_RMSNORM: {
        vk_fused_add_rmsnorm_args args = {
            .residual_inout = ob[0],
            .x = ib[0],
            .w = op->input_present[1] ? ib[1] : 0ULL,
            .y = ob[1],
            .rows = (uint32_t)op->inputs[0].dims[0],
            .dim = (uint32_t)op->inputs[0].dims[1],
            .r_stride = (uint32_t)op->outputs[0].strides[0],
            .x_stride = (uint32_t)op->inputs[0].strides[0],
            .y_stride = (uint32_t)op->outputs[1].strides[0],
            .eps = op->raw.attr.rmsnorm.eps,
            .has_weight = op->input_present[1] ? 1U : 0U,
            .unit_offset = op->raw.attr.rmsnorm.unit_offset ? 1U : 0U,
        };
        memcpy(out->pc, &args, sizeof(args));
        out->pc_size = (uint32_t)sizeof(args);
        out->gx = args.rows;
        out->gy = 1U;
        out->gz = 1U;
        break;
    }

    case VELRI_OP_LAYERNORM: {
        vk_layernorm_args args = {
            .x = ib[0],
            .w = op->input_present[1] ? ib[1] : 0ULL,
            .b = op->input_present[2] ? ib[2] : 0ULL,
            .y = ob[0],
            .rows = (uint32_t)op->inputs[0].dims[0],
            .dim = (uint32_t)op->inputs[0].dims[1],
            .x_stride = (uint32_t)op->inputs[0].strides[0],
            .y_stride = (uint32_t)op->outputs[0].strides[0],
            .eps = op->raw.attr.layernorm.eps,
            .has_weight = op->input_present[1] ? 1U : 0U,
            .has_bias = op->input_present[2] ? 1U : 0U,
            .pad = 0U,
        };
        memcpy(out->pc, &args, sizeof(args));
        out->pc_size = (uint32_t)sizeof(args);
        out->gx = args.rows;
        out->gy = 1U;
        out->gz = 1U;
        break;
    }

    case VELRI_OP_ROPE: {
        size_t pos = (item->params && item->params->has_pos_offset)
                         ? item->params->pos_offset_override
                         : op->raw.attr.rope.pos_offset;
        if (pos > (size_t)UINT32_MAX) {
            return VELRI_ERR_OVERFLOW;
        }
        uint32_t nt = (uint32_t)op->inputs[0].dims[0];
        uint32_t nh = (uint32_t)op->inputs[0].dims[1];
        uint32_t hd = (uint32_t)op->inputs[0].dims[2];
        uint32_t rd = (uint32_t)op->raw.attr.rope.rotary_dim;
        uint32_t tsx = (uint32_t)op->inputs[0].strides[0];
        uint32_t hsx = (uint32_t)op->inputs[0].strides[1];
        uint32_t tsy = (uint32_t)op->outputs[0].strides[0];
        uint32_t hsy = (uint32_t)op->outputs[0].strides[1];
        uint32_t v_elems = (op->inputs[0].dtype == VELRI_DTYPE_F32) ? 4U : 8U;
        uint32_t half_rot = rd >> 1U;
        bool flat_fast =
            (op->raw.attr.rope.mode == VELRI_ROPE_HALF) && (rd == hd) && (half_rot > 0U) &&
            ((half_rot & (v_elems - 1U)) == 0U) && ((tsx & (v_elems - 1U)) == 0U) &&
            ((hsx & (v_elems - 1U)) == 0U) && ((tsy & (v_elems - 1U)) == 0U) &&
            ((hsy & (v_elems - 1U)) == 0U) && ((half_rot & 3U) == 0U) &&
            ((ib[0] & 15ULL) == 0ULL) && ((ob[0] & 15ULL) == 0ULL) &&
            ((ib[1] & 15ULL) == 0ULL) && ((ib[2] & 15ULL) == 0ULL);
        uint64_t flat_gx = 0ULL;
        if (flat_fast) {
            uint64_t total_vec =
                (uint64_t)nt * (uint64_t)nh * (uint64_t)(half_rot / v_elems);
            flat_gx = (total_vec + 255ULL) / 256ULL;
            if (flat_gx == 0ULL || flat_gx > dev->max_wg_count[0]) {
                flat_fast = false;
            }
        }
        vk_rope_args args = {
            .x = ib[0],
            .cos_tbl = ib[1],
            .sin_tbl = ib[2],
            .y = ob[0],
            .n_tokens = nt,
            .n_heads = nh,
            .head_dim = hd,
            .rotary_dim = rd,
            .pos_offset = (uint32_t)pos,
            .tok_sx = tsx,
            .head_sx = hsx,
            .tok_sy = tsy,
            .head_sy = hsy,
            .mode = (uint32_t)op->raw.attr.rope.mode | (flat_fast ? 2U : 0U),
        };
        memcpy(out->pc, &args, sizeof(args));
        out->pc_size = (uint32_t)sizeof(args);
        if (flat_fast) {
            out->gx = (uint32_t)flat_gx;
            out->gy = 1U;
        } else {
            out->gx = div_ceil_u32(nh, 8U);
            out->gy = nt;
        }
        out->gz = 1U;
        break;
    }

    case VELRI_OP_ACTIVATION: {
        size_t n = 0U;
        if (velri_tensor_view_num_elements(&op->inputs[0], &n) != VELRI_OK ||
            n > (size_t)UINT32_MAX) {
            return VELRI_ERR_OVERFLOW;
        }
        vk_activation_args args = {.x = ib[0], .y = ob[0], .count = (uint32_t)n, .pad = 0U};
        memcpy(out->pc, &args, sizeof(args));
        out->pc_size = (uint32_t)sizeof(args);
        out->gx = div_ceil_u32(div_ceil_u32(args.count, 4U), 256U);
        out->gy = 1U;
        out->gz = 1U;
        break;
    }

    case VELRI_OP_GATED_MLP: {
        size_t n = 0U;
        if (velri_tensor_view_num_elements(&op->inputs[0], &n) != VELRI_OK ||
            n > (size_t)UINT32_MAX) {
            return VELRI_ERR_OVERFLOW;
        }
        vk_gated_mlp_args args = {
            .gate = ib[0],
            .up = ib[1],
            .y = ob[0],
            .count = (uint32_t)n,
            .clamp_limit = op->raw.attr.gated_mlp.clamp_limit,
        };
        memcpy(out->pc, &args, sizeof(args));
        out->pc_size = (uint32_t)sizeof(args);
        out->gx = div_ceil_u32(div_ceil_u32(args.count, 4U), 256U);
        out->gy = 1U;
        out->gz = 1U;
        break;
    }

    case VELRI_OP_VEC_AXPBY: {
        size_t n = 0U;
        if (velri_tensor_view_num_elements(&op->inputs[0], &n) != VELRI_OK ||
            n > (size_t)UINT32_MAX) {
            return VELRI_ERR_OVERFLOW;
        }
        bool has_b = op->input_present[1] && (op->raw.attr.axpby.beta != 0.0f);
        vk_vec_axpby_args args = {
            .a = ib[0],
            .b = has_b ? ib[1] : 0ULL,
            .y = ob[0],
            .count = (uint32_t)n,
            .alpha = op->raw.attr.axpby.alpha,
            .beta = op->raw.attr.axpby.beta,
            .has_b = has_b ? 1U : 0U,
            .pad = 0U,
        };
        memcpy(out->pc, &args, sizeof(args));
        out->pc_size = (uint32_t)sizeof(args);
        out->gx = div_ceil_u32(div_ceil_u32(args.count, 4U), 256U);
        out->gy = 1U;
        out->gz = 1U;
        break;
    }

    case VELRI_OP_VEC_MUL: {
        size_t n = 0U;
        if (velri_tensor_view_num_elements(&op->inputs[0], &n) != VELRI_OK ||
            n > (size_t)UINT32_MAX) {
            return VELRI_ERR_OVERFLOW;
        }
        vk_vec_mul_args args = {
            .a = ib[0], .b = ib[1], .y = ob[0], .count = (uint32_t)n, .pad = 0U
        };
        memcpy(out->pc, &args, sizeof(args));
        out->pc_size = (uint32_t)sizeof(args);
        out->gx = div_ceil_u32(div_ceil_u32(args.count, 4U), 256U);
        out->gy = 1U;
        out->gz = 1U;
        break;
    }

    case VELRI_OP_SOFTMAX: {
        vk_softmax_args args = {
            .x = ib[0],
            .y = ob[0],
            .rows = (uint32_t)op->inputs[0].dims[0],
            .cols = (uint32_t)op->inputs[0].dims[1],
            .x_stride = (uint32_t)op->inputs[0].strides[0],
            .y_stride = (uint32_t)op->outputs[0].strides[0],
        };
        memcpy(out->pc, &args, sizeof(args));
        out->pc_size = (uint32_t)sizeof(args);
        out->gx = args.rows;
        out->gy = 1U;
        out->gz = 1U;
        break;
    }

    case VELRI_OP_FWHT: {
        uint32_t suh_mode = 0U;
        if (op->input_present[1]) {
            suh_mode = (op->raw.attr.fwht.suh_dtype == VELRI_DTYPE_F32) ? 1U : 2U;
        }
        uint32_t bs = (uint32_t)op->raw.attr.fwht.block_size;
        vk_fwht_args args = {
            .x = ib[0],
            .suh = op->input_present[1] ? ib[1] : 0ULL,
            .y = ob[0],
            .rows = (uint32_t)op->inputs[0].dims[0],
            .cols = (uint32_t)op->inputs[0].dims[1],
            .block_size = bs,
            .x_stride = (uint32_t)op->inputs[0].strides[0],
            .y_stride = (uint32_t)op->outputs[0].strides[0],
            .suh_mode = suh_mode,
            .inv_sqrt_b = 1.0f / sqrtf((float)bs),
            .pad = 0U,
        };
        memcpy(out->pc, &args, sizeof(args));
        out->pc_size = (uint32_t)sizeof(args);
        out->gx = div_ceil_u32(args.cols, 1024U);
        out->gy = args.rows;
        out->gz = 1U;
        break;
    }

    case VELRI_OP_QUANTIZE_ROW: {
        vk_quantize_row_args args = {
            .x = ib[0],
            .x_i8 = ob[0],
            .scale_out = ob[1],
            .rows = (uint32_t)op->inputs[0].dims[0],
            .k = (uint32_t)op->inputs[0].dims[1],
            .x_stride = (uint32_t)op->inputs[0].strides[0],
            .xi8_stride = (uint32_t)op->outputs[0].strides[0],
        };
        memcpy(out->pc, &args, sizeof(args));
        out->pc_size = (uint32_t)sizeof(args);
        out->gx = args.rows;
        out->gy = 1U;
        out->gz = 1U;
        break;
    }

    default:
        return VELRI_ERR_UNSUPPORTED;
    }

    if (!grid_fits(dev, out->gx, out->gy, out->gz)) {
        return VELRI_ERR_OVERFLOW;
    }
    return VELRI_OK;
}

static velri_status vk_submit_batch(velri_backend_device *dev, const velri_submit_item *items,
                                    uint32_t count, velri_completion **out_completion) {
    if (!out_completion) {
        return VELRI_ERR_INVALID;
    }
    *out_completion = nullptr;
    if (!dev || !items || count == 0U || count > VELRI_MAX_SUBMIT_BATCH) {
        return VELRI_ERR_INVALID;
    }

    vk_encoded_dispatch enc[VELRI_MAX_SUBMIT_BATCH];
    bool batch_uses_splitk = false;

    velri_os_mutex_lock(dev->mtx);

    /* Validate and encode every item before touching GPU state (pre-accept safety). */
    for (uint32_t i = 0U; i < count; ++i) {
        velri_status st = encode_item_locked(dev, &items[i], &enc[i]);
        if (st != VELRI_OK) {
            velri_os_mutex_unlock(dev->mtx);
            return st;
        }
        if (enc[i].uses_splitk) {
            batch_uses_splitk = true;
        }
    }

    /* Acquire a free ring slot (and split-K exclusivity if needed). */
    struct velri_completion *slot = nullptr;
    for (;;) {
        if (!batch_uses_splitk || !dev->splitk_busy) {
            for (int i = 0; i < VK_CMD_RING_SIZE; ++i) {
                if (!dev->ring[i].in_use) {
                    slot = &dev->ring[i];
                    break;
                }
            }
            if (slot) {
                break;
            }
        }
        velri_os_cond_wait(dev->slot_cv, dev->mtx);
    }
    slot->in_use = true;
    slot->uses_splitk = batch_uses_splitk;
    if (batch_uses_splitk) {
        dev->splitk_busy = true;
    }

    VkMemoryBarrier mbar = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
    };

    dev->vkResetCommandBuffer(slot->cmd, 0U);
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    velri_status st = VELRI_OK;
    if (dev->vkBeginCommandBuffer(slot->cmd, &bi) != VK_SUCCESS) {
        st = VELRI_ERR_EXEC;
    } else {
        for (uint32_t i = 0U; i < count; ++i) {
            if (i > 0U) {
                dev->vkCmdPipelineBarrier(slot->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0U, 1U, &mbar, 0U,
                                          nullptr, 0U, nullptr);
            }
            dev->vkCmdBindPipeline(slot->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, enc[i].pipe);
            dev->vkCmdPushConstants(slot->cmd, dev->pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                                    enc[i].pc_size, enc[i].pc);
            dev->vkCmdDispatch(slot->cmd, enc[i].gx, enc[i].gy, enc[i].gz);

            if (enc[i].uses_splitk) {
                dev->vkCmdPipelineBarrier(slot->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0U, 1U, &mbar, 0U,
                                          nullptr, 0U, nullptr);
                dev->vkCmdBindPipeline(slot->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, enc[i].red_pipe);
                dev->vkCmdPushConstants(slot->cmd, dev->pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                        0U, enc[i].red_pc_size, &enc[i].red_pc);
                dev->vkCmdDispatch(slot->cmd, enc[i].red_gx, 1U, 1U);
            }
        }
        if (dev->vkEndCommandBuffer(slot->cmd) != VK_SUCCESS ||
            dev->vkResetFences(dev->device, 1U, &slot->fence) != VK_SUCCESS) {
            st = VELRI_ERR_EXEC;
        } else {
            VkSubmitInfo si = {
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .commandBufferCount = 1U,
                .pCommandBuffers = &slot->cmd,
            };
            if (dev->vkQueueSubmit(dev->queue, 1U, &si, slot->fence) != VK_SUCCESS) {
                st = VELRI_ERR_EXEC;
            }
        }
    }

    if (st != VELRI_OK) {
        if (slot->uses_splitk) {
            dev->splitk_busy = false;
            slot->uses_splitk = false;
        }
        slot->in_use = false;
        velri_os_cond_signal(dev->slot_cv);
        velri_os_mutex_unlock(dev->mtx);
        return st;
    }

    *out_completion = slot;
    velri_os_mutex_unlock(dev->mtx);
    return VELRI_OK;
}

static velri_status vk_submit(velri_backend_device *dev, const velri_submit_item *item,
                              velri_completion **out_completion) {
    return vk_submit_batch(dev, item, 1U, out_completion);
}

/* ---- 6. Completion ---- */

static velri_status vk_wait(velri_backend_device *dev, velri_completion *comp) {
    if (!dev || !comp) {
        return VELRI_ERR_INVALID;
    }
    /* Wait without holding `dev->mtx` so other threads can prepare/submit concurrently. */
    VkResult res = dev->vkWaitForFences(dev->device, 1U, &comp->fence, VK_TRUE, UINT64_MAX);
    return (res == VK_SUCCESS) ? VELRI_OK : VELRI_ERR_EXEC;
}

static void vk_destroy_completion(velri_backend_device *dev, velri_completion *comp) {
    if (!dev || !comp) {
        return;
    }
    velri_os_mutex_lock(dev->mtx);
    if (comp->uses_splitk) {
        dev->splitk_busy = false;
        comp->uses_splitk = false;
    }
    comp->in_use = false;
    velri_os_cond_signal(dev->slot_cv);
    velri_os_mutex_unlock(dev->mtx);
}

static const velri_backend_vtable g_vk_vtable = {
    .name = "vulkan",
    .contract_version = VELRI_BACKEND_CONTRACT_VERSION,
    .enumerate_devices = vk_enumerate_devices,
    .open_device = vk_open_device,
    .query_caps = vk_query_caps,
    .close_device = vk_close_device,
    .alloc = vk_alloc,
    .free = vk_free,
    .copy_to_ram = vk_copy_to_ram,
    .copy_from_ram = vk_copy_from_ram,
    .zero_mem = vk_zero_mem,
    .query_operation = vk_query_operation,
    .prepare_op = vk_prepare_op,
    .destroy_prepared_op = vk_destroy_prepared_op,
    .submit = vk_submit,
    .submit_batch = vk_submit_batch,
    .wait = vk_wait,
    .destroy_completion = vk_destroy_completion,
};

const velri_backend_vtable *velri_vulkan_backend_vtable(void) { return &g_vk_vtable; }
