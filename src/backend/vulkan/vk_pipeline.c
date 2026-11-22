#include "vk_internal.h"
#include "velri_vk_shaders.h"

#include <stdlib.h>

/*
 * Vulkan SPIR-V Compute Pipeline Table, Lazy Compilation, Split-K Planning,
 * and Operation Preparation.
 */

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
    [PIPE_EMBEDDING_F32]         = {spv_vk_embedding_f32, spv_vk_embedding_f32_bytes, VK_CAP_NONE},
    [PIPE_COPY_LAST_ROW_F32]     = {spv_vk_copy_last_row_f32, spv_vk_copy_last_row_f32_bytes, VK_CAP_NONE},
    [PIPE_KV_APPEND_F32]         = {spv_vk_kv_append_f32, spv_vk_kv_append_f32_bytes, VK_CAP_NONE},
    [PIPE_ATTENTION_DECODE_F32]  = {spv_vk_attention_decode_f32, spv_vk_attention_decode_f32_bytes, VK_CAP_NONE},
    [PIPE_ATTENTION_PREFILL_F32] = {spv_vk_attention_prefill_f32, spv_vk_attention_prefill_f32_bytes, VK_CAP_NONE},
    [PIPE_ATTENTION_WIDE_F32]    = {spv_vk_attention_wide_f32, spv_vk_attention_wide_f32_bytes, VK_CAP_NONE},
    [PIPE_ATTENTION_SPLITKV_F32] = {spv_vk_attention_splitkv_f32, spv_vk_attention_splitkv_f32_bytes, VK_CAP_NONE},
    [PIPE_ATTENTION_SPLITKV_REDUCE_F32] = {spv_vk_attention_splitkv_reduce_f32, spv_vk_attention_splitkv_reduce_f32_bytes, VK_CAP_NONE},
    [PIPE_BATCHED_LINEAR_NT_F32] = {spv_vk_batched_linear_nt_f32, spv_vk_batched_linear_nt_f32_bytes, VK_CAP_NONE},
    [PIPE_BATCHED_LINEAR_NN_F32] = {spv_vk_batched_linear_nn_f32, spv_vk_batched_linear_nn_f32_bytes, VK_CAP_NONE},
};

static bool has_pipe_caps(const velri_backend_device *dev, int id) {
    return id >= 0 && id < PIPE_COUNT &&
           (dev->caps_mask & k_pipes[id].req_caps) == k_pipes[id].req_caps;
}

VkPipeline velri_vk_get_pipe_locked(velri_backend_device *dev, int id) {
    if (!has_pipe_caps(dev, id)) {
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

bool velri_vk_splitk_layout(uint32_t sk, uint32_t rows, uint32_t m, uint32_t *out_part_stride,
                            size_t *out_bytes) {
    if (sk == 0U || rows == 0U || m == 0U || m > UINT32_MAX - 3U) {
        return false;
    }
    uint32_t part_stride = (m + 3U) & ~3U;
    size_t tmp = 0U, elems = 0U, bytes = 0U;
    if (velri_mul_overflow_size((size_t)sk, (size_t)rows, &tmp) ||
        velri_mul_overflow_size(tmp, (size_t)part_stride, &elems) ||
        velri_mul_overflow_size(elems, sizeof(float), &bytes)) {
        return false;
    }
    if (out_part_stride) {
        *out_part_stride = part_stride;
    }
    if (out_bytes) {
        *out_bytes = bytes;
    }
    return true;
}

uint32_t velri_vk_choose_split_k(uint32_t rows, uint32_t m, uint32_t k, uint32_t tile_m,
                                 uint32_t tile_n, uint32_t target_wgs, uint32_t bk_elems) {
    uint64_t wgs =
        (uint64_t)velri_div_ceil_u32(m, tile_n) * (uint64_t)velri_div_ceil_u32(rows, tile_m);
    if (rows < 16U || wgs == 0ULL || wgs >= target_wgs) {
        return 1U;
    }
    uint32_t k_tiles = velri_div_ceil_u32(k, bk_elems);
    if (k_tiles < 16U) {
        return 1U;
    }
    uint32_t sk = (uint32_t)((uint64_t)target_wgs / wgs);
    if (sk > 8U) sk = 8U;
    if (sk > k_tiles / 4U) sk = k_tiles / 4U;
    if (sk < 2U) return 1U;
    size_t part_bytes = 0U;
    if (!velri_vk_splitk_layout(sk, rows, m, nullptr, &part_bytes) ||
        (uint64_t)part_bytes > VK_SPLITK_MAX_BYTES) {
        return 1U;
    }
    return sk;
}

bool velri_vk_splitkv_layout(uint32_t n_splits, uint32_t n_heads, uint32_t d_v,
                             uint32_t *out_part_stride, size_t *out_bytes) {
    if (n_splits == 0U || n_heads == 0U || d_v == 0U || d_v > UINT32_MAX - 5U) {
        return false;
    }
    uint32_t part_stride = ((d_v + 2U) + 3U) & ~3U;
    size_t tmp = 0U, elems = 0U, bytes = 0U;
    if (velri_mul_overflow_size((size_t)n_heads, (size_t)n_splits, &tmp) ||
        velri_mul_overflow_size(tmp, (size_t)part_stride, &elems) ||
        velri_mul_overflow_size(elems, sizeof(float), &bytes)) {
        return false;
    }
    if (out_part_stride) {
        *out_part_stride = part_stride;
    }
    if (out_bytes) {
        *out_bytes = bytes;
    }
    return true;
}

uint32_t velri_vk_choose_split_kv(uint32_t n_heads, uint32_t d_v, uint32_t kv_len) {
    if (n_heads == 0U || d_v == 0U || kv_len < VK_SPLITKV_MIN_KV_LEN) {
        return 1U;
    }
    /* Target ~2x SM count at >=1024 keys to hide DRAM latency on 4-warp Split-KV WGs. */
    uint32_t target_wgs = (kv_len >= 1024U) ? (VK_TARGET_WGS * 2U) : VK_TARGET_WGS;
    uint32_t sp = velri_div_ceil_u32(target_wgs, n_heads);
    if (sp < 2U) {
        return 1U;
    }
    uint32_t by_len = kv_len / 64U;
    if (sp > by_len) {
        sp = by_len;
    }
    if (sp > VK_SPLITKV_MAX_SPLITS) {
        sp = VK_SPLITKV_MAX_SPLITS;
    }
    /* Quantize to power-of-2 buckets {2, 4, 8, 16} for O(1) persistent graph capture reuse. */
    uint32_t p2 = VK_SPLITKV_MAX_SPLITS;
    while (p2 > sp && p2 > 1U) {
        p2 >>= 1U;
    }
    sp = p2;
    while (sp >= 2U) {
        size_t bytes = 0U;
        if (velri_vk_splitkv_layout(sp, n_heads, d_v, nullptr, &bytes) &&
            (uint64_t)bytes <= VK_SPLITK_MAX_BYTES) {
            return sp;
        }
        sp >>= 1U;
    }
    return 1U;
}

static void select_coopmat_plan(const velri_backend_device *dev, uint32_t rows, uint32_t m,
                                int gemv_id, int cm64_id, int cm128_id,
                                struct velri_prepared_op *out_plan) {
    if (rows <= 2U || !has_pipe_caps(dev, cm128_id)) {
        out_plan->pipe_id = has_pipe_caps(dev, gemv_id) ? gemv_id : -1;
        return;
    }
    uint64_t tiles128 =
        (uint64_t)velri_div_ceil_u32(m, 128U) * (uint64_t)velri_div_ceil_u32(rows, 128U);
    out_plan->pipe_id = (tiles128 < (uint64_t)VK_TARGET_WGS) ? cm64_id : cm128_id;
    if (has_pipe_caps(dev, gemv_id)) {
        out_plan->alt_pipe_id = gemv_id;
    }
}

velri_status velri_vk_resolve_op_pipes(const velri_backend_device *dev, const velri_resolved_op *op,
                                       struct velri_prepared_op *out_plan) {
    out_plan->pipe_id = -1;
    out_plan->alt_pipe_id = -1;
    out_plan->reduce_pipe_id = -1;
    out_plan->stage2_decode_pipe_id = -1;
    out_plan->stage2_prefill_pipe_id = -1;
    out_plan->stage2_splitkv_pipe_id = -1;
    out_plan->stage3_splitkv_reduce_pipe_id = -1;
    out_plan->split_k = 1U;
    out_plan->splitk_bytes = 0U;

    switch (op->kind) {
    case VELRI_OP_LINEAR: {
        uint32_t rows = (uint32_t)op->inputs[0].dims[0];
        uint32_t m = (uint32_t)op->inputs[1].dims[0];
        uint32_t k = (uint32_t)op->inputs[0].dims[1];
        switch (velri_linear_classify(&op->raw.attr.linear.desc)) {
        case VELRI_LIN_F32:
            if (rows <= 2U || !has_pipe_caps(dev, PIPE_GEMM_F32)) {
                out_plan->pipe_id = PIPE_GEMV_F32;
            } else {
                out_plan->pipe_id = PIPE_GEMM_F32;
                out_plan->alt_pipe_id = PIPE_GEMV_F32;
                uint32_t sk = velri_vk_choose_split_k(rows, m, k, 128U, 128U, VK_TARGET_WGS, 8U);
                size_t sk_bytes = 0U;
                if (sk > 1U && velri_vk_splitk_layout(sk, rows, m, nullptr, &sk_bytes)) {
                    out_plan->split_k = sk;
                    out_plan->reduce_pipe_id = PIPE_SPLITK_REDUCE;
                    out_plan->splitk_bytes = sk_bytes;
                }
            }
            break;
        case VELRI_LIN_F16_F32:
            select_coopmat_plan(dev, rows, m, PIPE_GEMV_F16, PIPE_GEMM_COOPMAT_F16_64,
                                PIPE_GEMM_COOPMAT_F16, out_plan);
            break;
        case VELRI_LIN_BF16_F32:
            select_coopmat_plan(dev, rows, m, PIPE_GEMV_BF16, PIPE_GEMM_COOPMAT_BF16_64,
                                PIPE_GEMM_COOPMAT_BF16, out_plan);
            break;
        case VELRI_LIN_W8A8_F32:
            select_coopmat_plan(dev, rows, m, PIPE_GEMV_W8A8, PIPE_GEMM_COOPMAT_W8A8_64,
                                PIPE_GEMM_COOPMAT_W8A8, out_plan);
            break;
        case VELRI_LIN_F32X_BF16W:
        case VELRI_LIN_F64:
        case VELRI_LIN_UNSUPPORTED:
        default:
            break;
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
    case VELRI_OP_EMBEDDING:
        out_plan->pipe_id = PIPE_EMBEDDING_F32;
        break;
    case VELRI_OP_ATTENTION: {
        uint32_t dk = op->raw.attr.attention.head_dim;
        uint32_t dv = velri_op_attn_dv(op);
        out_plan->pipe_id = PIPE_KV_APPEND_F32;
        if (!velri_op_attn_is_wide(dk, dv)) {
            out_plan->stage2_decode_pipe_id = PIPE_ATTENTION_DECODE_F32;
            out_plan->stage2_prefill_pipe_id = PIPE_ATTENTION_PREFILL_F32;
            out_plan->stage2_splitkv_pipe_id = PIPE_ATTENTION_SPLITKV_F32;
            out_plan->stage3_splitkv_reduce_pipe_id = PIPE_ATTENTION_SPLITKV_REDUCE_F32;
            uint32_t max_ctx = (uint32_t)op->outputs[1].dims[1];
            uint32_t sw = op->raw.attr.attention.sliding_window;
            uint32_t eff_max_kv = (sw > 0U && sw < max_ctx) ? sw : max_ctx;
            uint32_t max_splits =
                velri_vk_choose_split_kv(op->raw.attr.attention.n_heads, dv, eff_max_kv);
            size_t skv_bytes = 0U;
            if (max_splits > 1U &&
                velri_vk_splitkv_layout(max_splits, op->raw.attr.attention.n_heads, dv, nullptr,
                                        &skv_bytes)) {
                out_plan->splitk_bytes = skv_bytes;
            }
        } else {
            out_plan->stage2_decode_pipe_id = PIPE_ATTENTION_WIDE_F32;
            out_plan->stage2_prefill_pipe_id = PIPE_ATTENTION_WIDE_F32;
        }
        break;
    }
    case VELRI_OP_COPY_LAST_ROW:
        out_plan->pipe_id = PIPE_COPY_LAST_ROW_F32;
        break;
    case VELRI_OP_BATCHED_LINEAR:
        out_plan->pipe_id = op->raw.attr.batched_linear.weight_transposed
                                ? PIPE_BATCHED_LINEAR_NN_F32
                                : PIPE_BATCHED_LINEAR_NT_F32;
        break;
    default:
        break;
    }

    if (!has_pipe_caps(dev, out_plan->pipe_id)) {
        return VELRI_ERR_UNSUPPORTED;
    }
    const int opt_pipes[6] = {
        out_plan->alt_pipe_id,
        out_plan->reduce_pipe_id,
        out_plan->stage2_decode_pipe_id,
        out_plan->stage2_prefill_pipe_id,
        out_plan->stage2_splitkv_pipe_id,
        out_plan->stage3_splitkv_reduce_pipe_id,
    };
    for (size_t i = 0U; i < sizeof(opt_pipes) / sizeof(opt_pipes[0]); ++i) {
        if (opt_pipes[i] >= 0 && !has_pipe_caps(dev, opt_pipes[i])) {
            return VELRI_ERR_UNSUPPORTED;
        }
    }
    return VELRI_OK;
}

/* ---- 4. Preparation Backend Contract ---- */

velri_status velri_vk_query_operation(const velri_backend_device *dev,
                                      const velri_resolved_op *op) {
    if (!dev || !op) {
        return VELRI_ERR_INVALID;
    }
    velri_status st = velri_op_validate(op);
    if (st != VELRI_OK) {
        return st;
    }
    struct velri_prepared_op plan;
    return velri_vk_resolve_op_pipes(dev, op, &plan);
}

velri_status velri_vk_prepare_op(velri_backend_device *dev, const velri_resolved_op *op,
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
    st = velri_vk_resolve_op_pipes(dev, op, &plan);
    if (st != VELRI_OK) {
        return st;
    }

    const int req_pipes[7] = {
        plan.pipe_id,
        plan.alt_pipe_id,
        plan.reduce_pipe_id,
        plan.stage2_decode_pipe_id,
        plan.stage2_prefill_pipe_id,
        plan.stage2_splitkv_pipe_id,
        plan.stage3_splitkv_reduce_pipe_id,
    };

    velri_os_mutex_lock(dev->mtx);
    for (size_t i = 0U; i < sizeof(req_pipes) / sizeof(req_pipes[0]); ++i) {
        if (req_pipes[i] >= 0 && !velri_vk_get_pipe_locked(dev, req_pipes[i])) {
            velri_os_mutex_unlock(dev->mtx);
            return VELRI_ERR_UNSUPPORTED;
        }
    }
    if (plan.splitk_bytes > 0U && (!dev->splitk_buf || dev->splitk_buf->size < plan.splitk_bytes)) {
        /* Wait for any in-flight split-K user before resizing shared partial buffer. */
        while (dev->splitk_busy) {
            velri_os_cond_wait(dev->slot_cv, dev->mtx);
        }
        if (!dev->splitk_buf || dev->splitk_buf->size < plan.splitk_bytes) {
            vk_buf *nb = nullptr;
            st = velri_vk_alloc_buf_locked(dev, plan.splitk_bytes, 256U, &nb);
            if (st != VELRI_OK) {
                velri_os_cond_broadcast(dev->slot_cv);
                velri_os_mutex_unlock(dev->mtx);
                return st;
            }
            velri_vk_free_buf_locked(dev, dev->splitk_buf);
            dev->splitk_buf = nb;
            dev->splitk_buf_gen++;
            velri_os_cond_broadcast(dev->slot_cv);
        }
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

void velri_vk_destroy_prepared_op(velri_backend_device *dev, velri_prepared_op *prepared) {
    (void)dev;
    free(prepared);
}
