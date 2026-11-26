#include "velri_vk_abi.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/*
 * Vulkan Command Encoding, Multi-Stage Batched Ring Submission, Completion Handling,
 * and Backend Vtable Registration.
 */

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

static inline uint32_t grid_x_vec4_256(uint32_t count) {
    return velri_div_ceil_u32(velri_div_ceil_u32(count, 4U), 256U);
}

static velri_status encode_item_locked(velri_backend_device *dev, const velri_submit_item *item,
                                       uint64_t step_params_bda, bool allow_splitk,
                                       uint32_t force_splitkv, vk_encoded_dispatch *out) {
    memset(out, 0, sizeof(*out));
    if (!velri_submit_item_header_valid(item)) {
        return VELRI_ERR_INVALID;
    }
    const velri_resolved_op *op = item->op;

    uint64_t ib[VELRI_OP_MAX_INPUTS] = {0};
    uint64_t ob[VELRI_OP_MAX_OUTPUTS] = {0};
    for (uint32_t i = 0U; i < op->num_inputs; ++i) {
        if (op->input_present[i] && (!item->inputs || !binding_bda(&item->inputs[i], &ib[i]))) {
            return VELRI_ERR_INVALID;
        }
    }
    for (uint32_t o = 0U; o < op->num_outputs; ++o) {
        if (!item->outputs || !binding_bda(&item->outputs[o], &ob[o])) {
            return VELRI_ERR_INVALID;
        }
    }

    struct velri_prepared_op plan;
    velri_status st = VELRI_OK;
    if (item->prepared) {
        plan = *item->prepared;
    } else {
        st = velri_op_validate(op);
        if (st != VELRI_OK) {
            return st;
        }
        st = velri_vk_resolve_op_pipes(dev, op, &plan);
        if (st != VELRI_OK) {
            return st;
        }
    }

    VkPipeline pipe0 = VK_NULL_HANDLE;
    if (op->kind != VELRI_OP_LINEAR) {
        pipe0 = velri_vk_get_pipe_locked(dev, plan.pipe_id);
        if (!pipe0) {
            return VELRI_ERR_UNSUPPORTED;
        }
    }

    uint32_t warps_per_wg = VK_GEMV_WG_SIZE / dev->subgroup_size;
    switch (op->kind) {
    case VELRI_OP_LINEAR: {
        uint32_t rows = 0U;
        st = velri_op_active_rows(op, item->params, &rows);
        if (st != VELRI_OK) {
            return st;
        }
        uint32_t m = (uint32_t)op->inputs[1].dims[0];
        uint32_t k = (uint32_t)op->inputs[0].dims[1];
        uint32_t xs = (uint32_t)op->inputs[0].strides[0];
        uint32_t ws = (uint32_t)op->inputs[1].strides[0];
        uint32_t ys = (uint32_t)op->outputs[0].strides[0];
        uint32_t flags = velri_op_linear_epilogue_flags(op);

        int active_pipe_id = plan.pipe_id;
        if (rows <= 2U && plan.alt_pipe_id >= 0) {
            active_pipe_id = plan.alt_pipe_id;
        }
        if (plan.short_w8a8 && rows == 64U && ((ib[0] | ib[1]) & 15U) == 0U &&
            dev->pipes[PIPE_GEMM_SHORT_W8A8]) {
            active_pipe_id = PIPE_GEMM_SHORT_W8A8;
        }
        if (active_pipe_id == PIPE_GEMM_F32 && op->raw.attr.linear.allow_f16_matmul &&
            dev->pipes[PIPE_GEMM_F32_AS_F16] && velri_vk_mixed_gemm_fits(dev, rows, m)) {
            active_pipe_id = PIPE_GEMM_F32_AS_F16;
        }
        if (active_pipe_id == PIPE_GEMM_F32 && rows >= 3U && rows <= 64U) {
            const bool vector_inputs = !((ib[0] | ib[1]) & 15U) && !((k | xs | ws) & 3U);
            int prompt_pipe = active_pipe_id;
            if (rows <= 16U) {
                prompt_pipe = vector_inputs ? PIPE_GEMV_ROWS_F32 : PIPE_GEMV_F32;
            } else if (vector_inputs && k % 16U == 0U) {
                prompt_pipe = PIPE_GEMM_SHORT_F32;
            }
            const uint32_t cols_per_group = rows <= 16U ? warps_per_wg : 64U;
            if (prompt_pipe != active_pipe_id &&
                velri_div_ceil_u32(m, cols_per_group) <= dev->max_wg_count[0]) {
                pipe0 = dev->pipes[prompt_pipe];
                if (pipe0) active_pipe_id = prompt_pipe;
            }
        }
        if (!pipe0) pipe0 = velri_vk_get_pipe_locked(dev, active_pipe_id);
        if (!pipe0) {
            return VELRI_ERR_UNSUPPORTED;
        }

        vk_linear_args args = {
            .x = ib[0],
            .w = ib[1],
            .bias = (op->num_inputs > 2U && op->input_present[2]) ? ib[2] : 0ULL,
            .x_scale = (op->num_inputs > 3U && op->input_present[3]) ? ib[3] : 0ULL,
            .w_scale = (op->num_inputs > 4U && op->input_present[4]) ? ib[4] : 0ULL,
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

        /* Bias is optional. Keep masked reads of the two prompt shaders inside a
         * valid F32 allocation on software Vulkan subgroup implementations. */
        if (!args.bias && (active_pipe_id == PIPE_GEMV_ROWS_F32 ||
                           active_pipe_id == PIPE_GEMM_SHORT_F32)) {
            args.bias = args.w;
        }

        uint32_t gx = 1U, gy = 1U, gz = 1U;
        if (active_pipe_id == PIPE_GEMV_ROWS_F32) {
            gx = velri_div_ceil_u32(m, warps_per_wg);
            gy = velri_div_ceil_u32(rows, 4U);
        } else if (active_pipe_id == PIPE_GEMV_F32 || active_pipe_id == PIPE_GEMV_F16 ||
                   active_pipe_id == PIPE_GEMV_BF16 || active_pipe_id == PIPE_GEMV_W8A8) {
            gx = velri_div_ceil_u32(m, warps_per_wg);
            gy = rows;
        } else if (active_pipe_id == PIPE_GEMM_SHORT_W8A8) {
            gx = m / 32U;
            gy = 1U;
        } else if (active_pipe_id == PIPE_GEMM_SHORT_F32 ||
                   active_pipe_id == PIPE_GEMM_F32_AS_F16 ||
                   active_pipe_id == PIPE_GEMM_COOPMAT_F16_64 ||
                   active_pipe_id == PIPE_GEMM_COOPMAT_BF16_64 ||
                   active_pipe_id == PIPE_GEMM_COOPMAT_W8A8_64) {
            gx = velri_div_ceil_u32(m, 64U);
            gy = velri_div_ceil_u32(rows, 64U);
        } else {
            gx = velri_div_ceil_u32(m, 128U);
            gy = velri_div_ceil_u32(rows, 128U);
        }

        out->num_stages = 1U;
        const bool short_tile = active_pipe_id == PIPE_GEMM_SHORT_F32;
        if (allow_splitk && (active_pipe_id == PIPE_GEMM_F32 || short_tile) &&
            plan.split_k > 1U && plan.reduce_pipe_id >= 0) {
            uint32_t sk = short_tile
                              ? velri_vk_choose_split_k(rows, m, k, 64U, 64U,
                                                        VK_SHORT_GEMM_TARGET_WGS, 16U)
                              : velri_vk_choose_split_k(rows, m, k, 128U, 128U, VK_TARGET_WGS, 8U);
            if (sk > plan.split_k) {
                sk = plan.split_k;
            }
            uint32_t part_stride = 0U;
            size_t need_bytes = 0U;
            if (sk > 1U && velri_vk_splitk_layout(sk, rows, m, &part_stride, &need_bytes)) {
                VkPipeline reduce_pipe =
                    (dev->splitk_buf && dev->splitk_buf->size >= need_bytes)
                        ? velri_vk_get_pipe_locked(dev, plan.reduce_pipe_id)
                        : VK_NULL_HANDLE;
                if (reduce_pipe != VK_NULL_HANDLE) {
                    uint64_t total_u4 = (uint64_t)rows * (uint64_t)(part_stride >> 2U);
                    uint64_t rgx = (total_u4 + 255ULL) / 256ULL;
                    if (rgx == 0ULL || rgx > dev->max_wg_count[0]) {
                        return VELRI_ERR_OVERFLOW;
                    }
                    args.y = (uint64_t)dev->splitk_buf->bda;
                    args.y_stride = part_stride;
                    args.split_k = sk;
                    gz = sk;

                    vk_split_k_reduce_args red = {
                        .partials = (uint64_t)dev->splitk_buf->bda,
                        .bias = args.bias,
                        .x_scale = args.x_scale,
                        .w_scale = args.w_scale,
                        .y = ob[0],
                        .rows = rows,
                        .m = m,
                        .part_stride_u4 = part_stride >> 2U,
                        .y_stride = ys,
                        .split_k = sk,
                        .flags = flags,
                    };
                    set_dispatch_stage(&out->stages[1], reduce_pipe, &red, sizeof(red),
                                       (uint32_t)rgx, 1U, 1U);
                    out->num_stages = 2U;
                    out->splitk_kind = VK_SPLITK_GEMM;
                    out->splitk_bytes = need_bytes;
                }
            }
        }
        set_dispatch_stage(&out->stages[0], pipe0, &args, sizeof(args), gx, gy, gz);
        break;
    }

    case VELRI_OP_RMSNORM: {
        uint32_t rows = 0U;
        st = velri_op_active_rows(op, item->params, &rows);
        if (st != VELRI_OK) {
            return st;
        }
        bool has_w = (op->num_inputs > 1U && op->input_present[1]);
        vk_rms_norm_args args = {
            .x = ib[0],
            .w = has_w ? ib[1] : 0ULL,
            .y = ob[0],
            .rows = rows,
            .dim = (uint32_t)op->inputs[0].dims[1],
            .x_stride = (uint32_t)op->inputs[0].strides[0],
            .y_stride = (uint32_t)op->outputs[0].strides[0],
            .eps = op->raw.attr.rmsnorm.eps,
            .has_weight = has_w ? 1U : 0U,
            .unit_offset = op->raw.attr.rmsnorm.unit_offset ? 1U : 0U,
        };
        set_dispatch_stage(&out->stages[0], pipe0, &args, sizeof(args), args.rows, 1U, 1U);
        out->num_stages = 1U;
        break;
    }

    case VELRI_OP_FUSED_ADD_RMSNORM: {
        uint32_t rows = 0U;
        st = velri_op_active_rows(op, item->params, &rows);
        if (st != VELRI_OK) {
            return st;
        }
        bool has_w = (op->num_inputs > 1U && op->input_present[1]);
        vk_fused_add_rms_norm_args args = {
            .residual_inout = ob[0],
            .x = ib[0],
            .w = has_w ? ib[1] : 0ULL,
            .y = ob[1],
            .rows = rows,
            .dim = (uint32_t)op->inputs[0].dims[1],
            .r_stride = (uint32_t)op->outputs[0].strides[0],
            .x_stride = (uint32_t)op->inputs[0].strides[0],
            .y_stride = (uint32_t)op->outputs[1].strides[0],
            .eps = op->raw.attr.rmsnorm.eps,
            .has_weight = has_w ? 1U : 0U,
            .unit_offset = op->raw.attr.rmsnorm.unit_offset ? 1U : 0U,
        };
        set_dispatch_stage(&out->stages[0], pipe0, &args, sizeof(args), args.rows, 1U, 1U);
        out->num_stages = 1U;
        break;
    }

    case VELRI_OP_LAYERNORM: {
        uint32_t rows = 0U;
        st = velri_op_active_rows(op, item->params, &rows);
        if (st != VELRI_OK) {
            return st;
        }
        bool has_w = (op->num_inputs > 1U && op->input_present[1]);
        bool has_b = (op->num_inputs > 2U && op->input_present[2]);
        vk_layer_norm_args args = {
            .x = ib[0],
            .w = has_w ? ib[1] : 0ULL,
            .b = has_b ? ib[2] : 0ULL,
            .y = ob[0],
            .rows = rows,
            .dim = (uint32_t)op->inputs[0].dims[1],
            .x_stride = (uint32_t)op->inputs[0].strides[0],
            .y_stride = (uint32_t)op->outputs[0].strides[0],
            .eps = op->raw.attr.layernorm.eps,
            .has_weight = has_w ? 1U : 0U,
            .has_bias = has_b ? 1U : 0U,
            .pad = 0U,
        };
        set_dispatch_stage(&out->stages[0], pipe0, &args, sizeof(args), args.rows, 1U, 1U);
        out->num_stages = 1U;
        break;
    }

    case VELRI_OP_ROPE: {
        uint32_t pos = 0U, nt = 0U;
        st = velri_op_resolve_rope_step(op, item->params, &nt, &pos);
        if (st != VELRI_OK) {
            return st;
        }
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
            .step_params = step_params_bda,
            .n_tokens = nt,
            .n_heads = nh,
            .head_dim = hd,
            .rotary_dim = rd,
            .pos_offset = pos,
            .tok_sx = tsx,
            .head_sx = hsx,
            .tok_sy = tsy,
            .head_sy = hsy,
            .mode = (uint32_t)op->raw.attr.rope.mode | (flat_fast ? 2U : 0U),
        };
        uint32_t gx = flat_fast ? (uint32_t)flat_gx : velri_div_ceil_u32(nh, 8U);
        uint32_t gy = flat_fast ? 1U : nt;
        set_dispatch_stage(&out->stages[0], pipe0, &args, sizeof(args), gx, gy, 1U);
        out->num_stages = 1U;
        break;
    }

    case VELRI_OP_ACTIVATION: {
        uint32_t count = 0U;
        st = velri_op_effective_elemwise_count(op, &op->inputs[0], item->params, &count);
        if (st != VELRI_OK) {
            return st;
        }
        vk_activation_args args = {.x = ib[0], .y = ob[0], .count = count, .pad = 0U};
        set_dispatch_stage(&out->stages[0], pipe0, &args, sizeof(args), grid_x_vec4_256(count), 1U,
                           1U);
        out->num_stages = 1U;
        break;
    }

    case VELRI_OP_GATED_MLP: {
        uint32_t count = 0U;
        st = velri_op_effective_elemwise_count(op, &op->inputs[0], item->params, &count);
        if (st != VELRI_OK) {
            return st;
        }
        vk_gated_mlp_args args = {
            .gate = ib[0],
            .up = ib[1],
            .y = ob[0],
            .count = count,
            .clamp_limit = op->raw.attr.gated_mlp.clamp_limit,
        };
        set_dispatch_stage(&out->stages[0], pipe0, &args, sizeof(args), grid_x_vec4_256(count), 1U,
                           1U);
        out->num_stages = 1U;
        break;
    }

    case VELRI_OP_VEC_AXPBY: {
        uint32_t count = 0U;
        st = velri_op_effective_elemwise_count(op, &op->inputs[0], item->params, &count);
        if (st != VELRI_OK) {
            return st;
        }
        bool has_b = (op->num_inputs > 1U && op->input_present[1]) &&
                     (op->raw.attr.axpby.beta != 0.0f);
        vk_vec_axpby_args args = {
            .a = ib[0],
            .b = has_b ? ib[1] : 0ULL,
            .y = ob[0],
            .count = count,
            .alpha = op->raw.attr.axpby.alpha,
            .beta = op->raw.attr.axpby.beta,
            .has_b = has_b ? 1U : 0U,
            .pad = 0U,
        };
        set_dispatch_stage(&out->stages[0], pipe0, &args, sizeof(args), grid_x_vec4_256(count), 1U,
                           1U);
        out->num_stages = 1U;
        break;
    }

    case VELRI_OP_VEC_MUL: {
        uint32_t count = 0U;
        st = velri_op_effective_elemwise_count(op, &op->inputs[0], item->params, &count);
        if (st != VELRI_OK) {
            return st;
        }
        vk_vec_mul_args args = {.a = ib[0], .b = ib[1], .y = ob[0], .count = count, .pad = 0U};
        set_dispatch_stage(&out->stages[0], pipe0, &args, sizeof(args), grid_x_vec4_256(count), 1U,
                           1U);
        out->num_stages = 1U;
        break;
    }

    case VELRI_OP_SOFTMAX: {
        uint32_t rows = 0U;
        st = velri_op_active_rows(op, item->params, &rows);
        if (st != VELRI_OK) {
            return st;
        }
        vk_softmax_args args = {
            .x = ib[0],
            .y = ob[0],
            .rows = rows,
            .cols = (uint32_t)op->inputs[0].dims[1],
            .x_stride = (uint32_t)op->inputs[0].strides[0],
            .y_stride = (uint32_t)op->outputs[0].strides[0],
        };
        set_dispatch_stage(&out->stages[0], pipe0, &args, sizeof(args), args.rows, 1U, 1U);
        out->num_stages = 1U;
        break;
    }

    case VELRI_OP_FWHT: {
        uint32_t rows = 0U;
        st = velri_op_active_rows(op, item->params, &rows);
        if (st != VELRI_OK) {
            return st;
        }
        uint32_t suh_mode = velri_op_fwht_suh_mode(op);
        uint32_t bs = (uint32_t)op->raw.attr.fwht.block_size;
        vk_fwht_args args = {
            .x = ib[0],
            .suh = (suh_mode != VELRI_FWHT_SUH_NONE) ? ib[1] : 0ULL,
            .y = ob[0],
            .rows = rows,
            .cols = (uint32_t)op->inputs[0].dims[1],
            .block_size = bs,
            .x_stride = (uint32_t)op->inputs[0].strides[0],
            .y_stride = (uint32_t)op->outputs[0].strides[0],
            .suh_mode = suh_mode,
            .inv_sqrt_b = 1.0f / sqrtf((float)bs),
            .pad = 0U,
        };
        set_dispatch_stage(&out->stages[0], pipe0, &args, sizeof(args),
                           velri_div_ceil_u32(args.cols, 1024U), args.rows, 1U);
        out->num_stages = 1U;
        break;
    }

    case VELRI_OP_QUANTIZE_ROW: {
        uint32_t rows = 0U;
        st = velri_op_active_rows(op, item->params, &rows);
        if (st != VELRI_OK) {
            return st;
        }
        vk_quantize_row_args args = {
            .x = ib[0],
            .x_i8 = ob[0],
            .scale_out = ob[1],
            .rows = rows,
            .k = (uint32_t)op->inputs[0].dims[1],
            .x_stride = (uint32_t)op->inputs[0].strides[0],
            .xi8_stride = (uint32_t)op->outputs[0].strides[0],
        };
        set_dispatch_stage(&out->stages[0], pipe0, &args, sizeof(args), args.rows, 1U, 1U);
        out->num_stages = 1U;
        break;
    }

    case VELRI_OP_EMBEDDING: {
        uint32_t rows = 0U, pos_off = 0U;
        st = velri_op_active_rows(op, item->params, &rows);
        if (st != VELRI_OK) {
            return st;
        }
        bool has_ids = velri_op_embedding_has_ids(op);
        if (!has_ids) {
            st = velri_op_resolve_pos_offset(op->raw.attr.embedding.pos_offset, item->params,
                                             &pos_off);
            if (st != VELRI_OK) {
                return st;
            }
        }
        uint32_t dim = (uint32_t)op->outputs[0].dims[1];
        vk_embedding_args args = {
            .table = ib[0],
            .ids = has_ids ? ib[1] : 0ULL,
            .y = ob[0],
            .step_params = has_ids ? 0ULL : step_params_bda,
            .rows = rows,
            .dim = dim,
            .vocab = (uint32_t)op->inputs[0].dims[0],
            .table_stride = (uint32_t)op->inputs[0].strides[0],
            .out_stride = (uint32_t)op->outputs[0].strides[0],
            .pos_offset = pos_off,
            .has_ids = has_ids ? 1U : 0U,
            .pad = 0U,
        };
        set_dispatch_stage(&out->stages[0], pipe0, &args, sizeof(args), grid_x_vec4_256(dim), rows,
                           1U);
        out->num_stages = 1U;
        break;
    }

    case VELRI_OP_ATTENTION: {
        velri_attn_step_args aa;
        st = velri_op_resolve_attention_step(op, item->params, &aa);
        if (st != VELRI_OK) {
            return st;
        }
        uint32_t nh = op->raw.attr.attention.n_heads;
        uint32_t nkvh = op->raw.attr.attention.n_kv_heads;
        uint32_t min_len = velri_vk_splitkv_min_len(dev, op);
        uint32_t same_kv = (ob[1] == ob[2]) ? 1U : 0U;

        vk_kv_append_args s1 = {
            .k_in = ib[1],
            .v_in = ib[2],
            .k_cache = ob[1],
            .v_cache = ob[2],
            .step_params = step_params_bda,
            .rows = aa.active_t,
            .n_kv_heads = nkvh,
            .d_k = aa.d_k,
            .d_v = aa.d_v,
            .max_ctx = aa.max_ctx,
            .kv_pos = aa.kv_pos,
            .k_stride = (uint32_t)op->inputs[1].strides[0],
            .v_stride = (uint32_t)op->inputs[2].strides[0],
            .k_head_stride = (uint32_t)op->outputs[1].strides[0],
            .k_row_stride = (uint32_t)op->outputs[1].strides[1],
            .v_head_stride = (uint32_t)op->outputs[2].strides[0],
            .v_row_stride = (uint32_t)op->outputs[2].strides[1],
            .same_kv_cache = same_kv,
            .pad = 0U,
        };
        uint32_t append_cols = (same_kv || aa.d_k >= aa.d_v) ? (nkvh * aa.d_k) : (nkvh * aa.d_v);
        set_dispatch_stage(&out->stages[0], pipe0, &s1, sizeof(s1),
                           velri_div_ceil_u32(append_cols, 256U), aa.active_t, 1U);

        /* FlashDecoding Split-KV fast path for long-context T=1 decode (d_k, d_v <= 256). */
        if (allow_splitk && aa.active_t == 1U && plan.stage2_splitkv_pipe_id >= 0 &&
            plan.stage3_splitkv_reduce_pipe_id >= 0) {
            uint32_t sw = op->raw.attr.attention.sliding_window;
            uint32_t eff_max_kv = (sw > 0U && sw < aa.max_ctx) ? sw : aa.max_ctx;
            uint32_t n_splits = (force_splitkv > 0U && eff_max_kv >= min_len)
                                    ? force_splitkv
                                    : 0U;
            if (n_splits == 0U && force_splitkv == 0U && step_params_bda == 0ULL) {
                uint64_t kv_end64 = (uint64_t)aa.kv_pos + 1ULL;
                uint32_t kv_end =
                    (kv_end64 < (uint64_t)aa.max_ctx) ? (uint32_t)kv_end64 : aa.max_ctx;
                uint32_t eff_kv_len = (sw > 0U && kv_end > sw) ? sw : kv_end;
                n_splits = velri_vk_choose_split_kv(nh, aa.d_v, eff_kv_len, min_len,
                                                  velri_vk_splitkv_limit(dev, op));
            }
            uint32_t group_heads = 1U;
            if (n_splits == 64U) {
                if (!((ib[0] | ob[1] | ob[2]) & 15U) && velri_vk_splitkv_limit(dev, op) == 64U &&
                    dev->pipes[PIPE_ATTENTION_GROUPED_F16KV] && dev->pipes[PIPE_ATTENTION_MERGE64_F32]) {
                    group_heads = velri_vk_decode_group_size(nh / nkvh);
                }
                if (group_heads == 1U) n_splits = 32U;
            }
            uint32_t part_stride = 0U;
            size_t need_bytes = 0U;
            if (n_splits > 1U &&
                velri_vk_splitkv_layout(n_splits, nh, aa.d_v, &part_stride, &need_bytes) &&
                dev->splitk_buf && dev->splitk_buf->size >= need_bytes) {
                VkPipeline pipe_skv = velri_vk_get_pipe_locked(dev, group_heads > 1U ? PIPE_ATTENTION_GROUPED_F16KV
                                                                                     : plan.stage2_splitkv_pipe_id);
                VkPipeline pipe_red =
                    velri_vk_get_pipe_locked(dev, group_heads > 1U ? PIPE_ATTENTION_MERGE64_F32
                                                                     : plan.stage3_splitkv_reduce_pipe_id);
                if (pipe_skv != VK_NULL_HANDLE && pipe_red != VK_NULL_HANDLE) {
                    vk_split_kv_args s2 = {
                        .q = ib[0],
                        .k_cache = ob[1],
                        .v_cache = ob[2],
                        .partials = (uint64_t)dev->splitk_buf->bda,
                        .step_params = step_params_bda,
                        .n_heads = nh,
                        .n_kv_heads = nkvh,
                        .d_k = aa.d_k,
                        .d_v = aa.d_v,
                        .max_ctx = aa.max_ctx,
                        .kv_pos = aa.kv_pos,
                        .part_stride = part_stride,
                        .k_head_stride = (uint32_t)op->outputs[1].strides[0],
                        .k_row_stride = (uint32_t)op->outputs[1].strides[1],
                        .v_head_stride = (uint32_t)op->outputs[2].strides[0],
                        .v_row_stride = (uint32_t)op->outputs[2].strides[1],
                        .sliding_window = sw,
                        .causal = op->raw.attr.attention.causal ? 1U : 0U,
                        .n_splits = n_splits,
                        .group_heads = group_heads,
                        .scale = aa.scale,
                    };
                    vk_split_kv_reduce_args s3 = {
                        .partials = (uint64_t)dev->splitk_buf->bda,
                        .out = ob[0],
                        .n_heads = nh,
                        .d_v = aa.d_v,
                        .part_stride = part_stride,
                        .n_splits = n_splits,
                    };
                    set_dispatch_stage(&out->stages[1], pipe_skv, &s2, sizeof(s2), nh / group_heads, n_splits,
                                       1U);
                    if (dev->subgroup_size == 32U && aa.d_v % 8U == 0U) {
                        s3.value_slices = aa.d_v / (n_splits >= 32U ? 4U : 8U);
                    }
                    set_dispatch_stage(&out->stages[2], pipe_red, &s3, sizeof(s3), nh,
                                       s3.value_slices ? s3.value_slices : 1U, 1U);
                    out->num_stages = 3U;
                    out->splitk_kind = VK_SPLITK_ATTN;
                    out->splitk_bytes = need_bytes;
                    break;
                }
            }
        }

        int s2_pipe_id =
            (aa.active_t == 1U) ? plan.stage2_decode_pipe_id : plan.stage2_prefill_pipe_id;
        // Several query rows reuse each key and value. Small grids favor one workgroup per row.
        bool flash = plan.flash_prefill && aa.active_t >= 128U &&
                     ((ib[0] | ob[1] | ob[2]) & 15U) == 0U;
        const bool f16 = op->outputs[1].dtype == VELRI_DTYPE_F16;
        const bool decode = s2_pipe_id == PIPE_ATTENTION_DECODE_F32 || s2_pipe_id == PIPE_ATTENTION_DECODE_F16KV;
        VkPipeline pipe2 = flash ? dev->pipes[f16 ? PIPE_FLASH_ATTENTION_F16KV : PIPE_FLASH_ATTENTION_F32] : VK_NULL_HANDLE;
        if (!pipe2) {
            flash = false;
            pipe2 = velri_vk_get_pipe_locked(dev, s2_pipe_id);
        }
        if (!pipe2) {
            return VELRI_ERR_UNSUPPORTED;
        }
        vk_attention_args s2 = {
            .q = ib[0],
            .k_cache = ob[1],
            .v_cache = ob[2],
            .y = ob[0],
            .step_params = step_params_bda,
            .rows = aa.active_t,
            .n_heads = nh,
            .n_kv_heads = nkvh,
            .d_k = aa.d_k,
            .d_v = aa.d_v,
            .max_ctx = aa.max_ctx,
            .kv_pos = aa.kv_pos,
            .q_stride = (uint32_t)op->inputs[0].strides[0],
            .o_stride = (uint32_t)op->outputs[0].strides[0],
            .k_head_stride = (uint32_t)op->outputs[1].strides[0],
            .k_row_stride = (uint32_t)op->outputs[1].strides[1],
            .v_head_stride = (uint32_t)op->outputs[2].strides[0],
            .v_row_stride = (uint32_t)op->outputs[2].strides[1],
            .sliding_window = op->raw.attr.attention.sliding_window,
            .causal = op->raw.attr.attention.causal ? 1U : 0U,
            .scale = aa.scale,
        };
        // Short cooperative grids become worthwhile once they have a resident history to scan.
        bool mixed = op->raw.attr.attention.allow_f16_matmul && dev->pipes[PIPE_PREFILL_COOP_HALF_F16KV];
        VkPipeline cooperative = dev->pipes[mixed ? PIPE_PREFILL_COOP_HALF_F16KV : PIPE_PREFILL_COOP_F16KV];
        bool use_cooperative = cooperative && velri_vk_f16kv_coop_fits(dev, op, aa.active_t) &&
                               ((ib[0] | ob[1] | ob[2]) & 15U) == 0U &&
                               (aa.active_t >= (mixed ? 128U : 512U) ||
                                (!step_params_bda && aa.kv_pos >= 256U));
        if (!use_cooperative && allow_splitk && !step_params_bda &&
            (s2_pipe_id == PIPE_ATTENTION_PREFILL_F32 || s2_pipe_id == PIPE_ATTENTION_PREFILL_F16KV) &&
            ((ib[0] | ob[1] | ob[2]) & 15U) == 0U) {
            uint64_t end = (uint64_t)aa.kv_pos + aa.active_t;
            size_t bytes = 0U;
            uint32_t splits = velri_vk_prefill_splits(dev, op, aa.active_t,
                end < aa.max_ctx ? (uint32_t)end : aa.max_ctx, &bytes);
            if (splits && dev->splitk_buf && dev->splitk_buf->size >= bytes) {
                VkPipeline partial = dev->pipes[f16 ? PIPE_PREFILL_SPLIT_F16KV : PIPE_PREFILL_SPLIT_F32];
                VkPipeline reduce = dev->pipes[PIPE_PREFILL_MERGE_F32];
                if (partial && reduce) {
                    vk_prefill_merge_args merge = {
                        .partials = dev->splitk_buf->bda, .output = s2.y, .row_heads = aa.active_t * nh,
                        .heads = nh, .output_stride = s2.o_stride, .splits = splits,
                    };
                    s2.y = merge.partials;
                    s2.value_slices = splits;
                    set_dispatch_stage(&out->stages[1], partial, &s2, sizeof(s2),
                                       velri_div_ceil_u32(aa.active_t, 32U), nh, splits);
                    set_dispatch_stage(&out->stages[2], reduce, &merge, sizeof(merge),
                                       velri_div_ceil_u32(merge.row_heads, 8U), 1U, 1U);
                    out->num_stages = 3U;
                    out->splitk_kind = VK_SPLITK_ATTN;
                    out->splitk_bytes = bytes;
                    break;
                }
            }
        }
        uint32_t gx2 = decode ? nh : aa.active_t;
        uint32_t gy2 = decode ? 1U : nh;
        if (decode && min_len == VK_SPLITKV_SLICED_MIN_KV_LEN &&
            ib[0] != ob[0] && ((ib[0] | ib[1] | ib[2] | ob[1] | ob[2]) & 15U) == 0U) {
            s2.value_slices = aa.d_v / 8U;
            gy2 = s2.value_slices;
        }
        if (use_cooperative) {
            pipe2 = cooperative;
            gx2 = velri_div_ceil_u32(aa.active_t, 16U);
            // The wider query tile reuses KV across two MMA tiles at long prefixes.
            // Its unmasked matrix loads require complete query and cache tiles.
            if (mixed && !step_params_bda && aa.kv_pos >= 1024U && aa.active_t >= 512U &&
                (aa.active_t & 31U) == 0U && velri_vk_nv_prefill_shape(op) &&
                dev->pipes[PIPE_PREFILL_COOP_NV_F16KV]) {
                pipe2 = dev->pipes[PIPE_PREFILL_COOP_NV_F16KV];
                gx2 = aa.active_t / 32U;
            }
        } else if (flash) {
            gx2 = velri_div_ceil_u32(aa.active_t, aa.d_k == 128U ? 2U : 4U);
            if (!f16 && aa.active_t >= 512U && aa.d_k == 64U &&
                (!s2.sliding_window || s2.sliding_window >= 512U) && dev->pipes[PIPE_PREFILL_TILE_F32]) {
                pipe2 = dev->pipes[PIPE_PREFILL_TILE_F32];
                gx2 = velri_div_ceil_u32(aa.active_t, 16U);
            }
        }
        set_dispatch_stage(&out->stages[1], pipe2, &s2, sizeof(s2), gx2, gy2, 1U);
        out->num_stages = 2U;
        break;
    }

    case VELRI_OP_COPY_LAST_ROW: {
        uint32_t rows = 0U;
        st = velri_op_active_rows(op, item->params, &rows);
        if (st != VELRI_OK) {
            return st;
        }
        uint32_t dim = (uint32_t)op->inputs[0].dims[1];
        vk_copy_last_row_args args = {
            .x = ib[0],
            .y = ob[0],
            .src_row = rows - 1U,
            .dim = dim,
            .x_stride = (uint32_t)op->inputs[0].strides[0],
            .pad = 0U,
        };
        set_dispatch_stage(&out->stages[0], pipe0, &args, sizeof(args), grid_x_vec4_256(dim), 1U,
                           1U);
        out->num_stages = 1U;
        break;
    }

    case VELRI_OP_BATCHED_LINEAR: {
        uint32_t rows = 0U;
        st = velri_op_active_rows(op, item->params, &rows);
        if (st != VELRI_OK) {
            return st;
        }
        uint32_t nh = op->raw.attr.batched_linear.n_heads;
        uint32_t ng = op->raw.attr.batched_linear.n_groups;
        uint32_t in_dim = op->raw.attr.batched_linear.in_dim;
        uint32_t out_dim = op->raw.attr.batched_linear.out_dim;
        bool has_bias = (op->num_inputs >= 3U && op->input_present[2]);
        vk_batched_linear_args args = {
            .x = ib[0],
            .w = ib[1],
            .bias = has_bias ? ib[2] : 0ULL,
            .y = ob[0],
            .rows = rows,
            .n_heads = nh,
            .n_groups = ng,
            .in_dim = in_dim,
            .out_dim = out_dim,
            .x_stride = (uint32_t)op->inputs[0].strides[0],
            .w_stride = (uint32_t)op->inputs[1].strides[0],
            .y_stride = (uint32_t)op->outputs[0].strides[0],
            .flags = has_bias ? VELRI_EP_FLAG_BIAS : 0U,
            .pad = 0U,
        };
        uint32_t gx = op->raw.attr.batched_linear.weight_transposed
                          ? velri_div_ceil_u32(out_dim, 256U)
                          : velri_div_ceil_u32(out_dim, 4U);
        set_dispatch_stage(&out->stages[0], pipe0, &args, sizeof(args), gx, rows, nh);
        out->num_stages = 1U;
        break;
    }

    default:
        return VELRI_ERR_UNSUPPORTED;
    }

    for (uint32_t s = 0U; s < out->num_stages; ++s) {
        const vk_dispatch_stage *ds = &out->stages[s];
        if (!grid_fits(dev, ds->gx, ds->gy, ds->gz)) {
            return VELRI_ERR_OVERFLOW;
        }
    }
    return VELRI_OK;
}

static void release_slot_locked(velri_backend_device *dev, struct velri_completion *slot) {
    if (slot->uses_splitk) {
        dev->splitk_busy = false;
        slot->uses_splitk = false;
    }
    slot->in_use = false;
    velri_os_cond_broadcast(dev->slot_cv);
}

static velri_status ensure_enc_scratch_locked(velri_backend_device *dev, uint32_t count) {
    if (count <= dev->enc_scratch_cap) {
        return VELRI_OK;
    }
    uint32_t new_cap = dev->enc_scratch_cap ? dev->enc_scratch_cap : VK_ENC_SCRATCH_INIT_CAP;
    while (new_cap < count) {
        if (new_cap > UINT32_MAX / 2U) {
            new_cap = count;
            break;
        }
        new_cap *= 2U;
    }
    vk_encoded_dispatch *nb =
        (vk_encoded_dispatch *)realloc(dev->enc_scratch, (size_t)new_cap * sizeof(*nb));
    if (!nb) {
        return VELRI_ERR_NOMEM;
    }
    dev->enc_scratch = nb;
    dev->enc_scratch_cap = new_cap;
    return VELRI_OK;
}

/* If another fused projection already appends K/V, its first stage is a different pipeline.
 * The following sliced attention then reads the cache without writing the token a second time. */
static bool combine_sliced_append(const velri_backend_device *dev, const vk_encoded_dispatch *enc,
                                  vk_dispatch_stage *stage) {
    if (enc->num_stages != 2U || enc->stages[0].pipe != dev->pipes[PIPE_KV_APPEND_F32] ||
        enc->stages[1].pipe != dev->pipes[PIPE_ATTENTION_DECODE_F32] || enc->stages[1].gy == 1U) {
        return false;
    }
    vk_kv_append_args append;
    vk_attention_args attention;
    memcpy(&append, enc->stages[0].pc, sizeof(append));
    memcpy(&attention, enc->stages[1].pc, sizeof(attention));
    attention.k_new = append.k_in;
    attention.v_new = append.v_in;
    *stage = enc->stages[1];
    memcpy(stage->pc, &attention, sizeof(attention));
    return true;
}

static void record_encoded_batch(velri_backend_device *dev, VkCommandBuffer cmd,
                                 const velri_submit_item *items, const vk_encoded_dispatch *enc,
                                 uint32_t count) {
    VkMemoryBarrier mbar = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
    };

    bool previous_fused = false;
    bool any_dispatched = false;
    bool splitk_dirty = false;
    for (uint32_t i = 0U; i < count; ++i) {
        vk_encoded_dispatch combined;
        const uint32_t consumed = velri_vk_fuse_decode(dev, &items[i], &enc[i], count - i, &combined);
        const vk_encoded_dispatch *active = consumed ? &combined : &enc[i];
        bool item_uses_sk = (active->splitk_kind != VK_SPLITK_NONE);
        vk_dispatch_stage sliced;
        bool inline_append = combine_sliced_append(dev, active, &sliced);
        uint32_t stages = inline_append ? 1U : active->num_stages;
        for (uint32_t s = 0U; s < stages; ++s) {
            const vk_dispatch_stage *ds = inline_append ? &sliced : &active->stages[s];
            bool need_bar = false;
            if (s > 0U) {
                need_bar = true;
            } else if (any_dispatched) {
                if (consumed || previous_fused || !items[i].no_barrier_before || (item_uses_sk && splitk_dirty)) {
                    need_bar = true;
                }
            }
            if (need_bar) {
                dev->vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0U, 1U, &mbar, 0U,
                                          nullptr, 0U, nullptr);
                splitk_dirty = false;
            }
            dev->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ds->pipe);
            dev->vkCmdPushConstants(cmd, dev->pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0U,
                                    ds->pc_size, ds->pc);
            dev->vkCmdDispatch(cmd, ds->gx, ds->gy, ds->gz);
            any_dispatched = true;
        }
        previous_fused = consumed != 0U;
        if (consumed) {
            i += consumed - 1U;
        }
        if (item_uses_sk) {
            splitk_dirty = true;
        }
    }
}

/* ---- 5. Execution Backend Contract ---- */

static velri_status vk_submit_batch(velri_backend_device *dev, const velri_submit_item *items,
                                    uint32_t count, velri_completion **out_completion) {
    if (!out_completion) {
        return VELRI_ERR_INVALID;
    }
    *out_completion = nullptr;
    if (!dev || !items || count == 0U) {
        return VELRI_ERR_INVALID;
    }

    velri_os_mutex_lock(dev->mtx);

    velri_status st = ensure_enc_scratch_locked(dev, count);
    if (st != VELRI_OK) {
        velri_os_mutex_unlock(dev->mtx);
        return st;
    }
    vk_encoded_dispatch *enc = dev->enc_scratch;
    bool batch_uses_splitk = false;

    /* Validate and encode every item before touching GPU state (pre-accept safety). */
    for (uint32_t i = 0U; i < count; ++i) {
        st = encode_item_locked(dev, &items[i], 0ULL, true, 0U, &enc[i]);
        if (st != VELRI_OK) {
            velri_os_mutex_unlock(dev->mtx);
            return st;
        }
        if (enc[i].splitk_kind != VK_SPLITK_NONE) {
            batch_uses_splitk = true;
        }
    }

    /* Acquire a free ring slot (and split-K/split-KV exclusivity if needed). */
    struct velri_completion *slot = nullptr;
    bool waited = false;
    for (;;) {
        if (!batch_uses_splitk || !dev->splitk_busy) {
            for (uint32_t i = 0U; i < VK_CMD_RING_SIZE; ++i) {
                if (!dev->ring[i].in_use) {
                    slot = &dev->ring[i];
                    break;
                }
            }
            if (slot) {
                break;
            }
        }
        waited = true;
        velri_os_cond_wait(dev->slot_cv, dev->mtx);
    }
    slot->in_use = true;
    slot->uses_splitk = batch_uses_splitk;
    slot->is_captured_graph = false;
    if (batch_uses_splitk) {
        dev->splitk_busy = true;
    }

    /*
     * If we slept in `cond_wait`, another thread may have overwritten `dev->enc_scratch`
     * or resized `dev->splitk_buf`. Now that we hold `dev->mtx` exclusively (plus
     * `dev->splitk_busy` if needed), re-encode cleanly before recording the command buffer.
     */
    if (waited) {
        st = ensure_enc_scratch_locked(dev, count);
        if (st == VELRI_OK) {
            enc = dev->enc_scratch;
            for (uint32_t i = 0U; i < count; ++i) {
                st = encode_item_locked(dev, &items[i], 0ULL, batch_uses_splitk, 0U, &enc[i]);
                if (st != VELRI_OK) {
                    break;
                }
            }
        }
        if (st != VELRI_OK) {
            release_slot_locked(dev, slot);
            velri_os_mutex_unlock(dev->mtx);
            return st;
        }
    } else {
        enc = dev->enc_scratch;
    }

    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    st = VELRI_OK;
    if (dev->vkResetCommandBuffer(slot->cmd, 0U) != VK_SUCCESS ||
        dev->vkBeginCommandBuffer(slot->cmd, &bi) != VK_SUCCESS) {
        st = VELRI_ERR_EXEC;
    } else {
        record_encoded_batch(dev, slot->cmd, items, enc, count);
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
        release_slot_locked(dev, slot);
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

/* ---- 5b. Persistent Launch-Graph Capture & Zero-Recording Replay ---- */

static void free_captured_graph_locked(velri_backend_device *dev, velri_captured_graph *cg) {
    if (!cg) {
        return;
    }
    if (cg->comp.in_use && cg->fence) {
        dev->vkWaitForFences(dev->device, 1U, &cg->fence, VK_TRUE, UINT64_MAX);
        cg->comp.in_use = false;
        if (cg->comp.uses_splitk) {
            dev->splitk_busy = false;
            cg->comp.uses_splitk = false;
            velri_os_cond_broadcast(dev->slot_cv);
        }
    }
    if (cg->step_slab) {
        velri_vk_free_step_slot_locked(cg->step_slab, cg->step_slot);
    }
    if (cg->fence) {
        dev->vkDestroyFence(dev->device, cg->fence, nullptr);
    }
    if (cg->pool) {
        dev->vkDestroyCommandPool(dev->device, cg->pool, nullptr);
    }
    velri_vk_free_buf_locked(dev, cg->readback);
    free(cg);
}

static uint32_t splitkv_bucket_index(uint32_t n_splits) {
    switch (n_splits) {
    case 2U:
        return 0U;
    case 4U:
        return 1U;
    case 8U:
        return 2U;
    case 16U:
        return 3U;
    case 32U:
        return 4U;
    case 64U:
        return 5U;
    default:
        return VK_SPLITKV_BUCKETS;
    }
}

static void record_output_readback(velri_backend_device *dev, VkCommandBuffer cmd,
                                   const vk_buf *source, velri_io_range range, const vk_buf *dst) {
    if (dst) velri_vk_record_readback(dev, cmd, source->buffer, range, dst->buffer);
}

static velri_status vk_capture_outputs(velri_backend_device *dev, const velri_submit_item *items,
                                       uint32_t count, void *io_arena, velri_io_range range,
                                       const void **out_readback, velri_captured_graph **out_graph) {
    if (!dev || !items || count == 0U || !out_graph) {
        return VELRI_ERR_INVALID;
    }
    *out_graph = nullptr;
    if (out_readback) {
        *out_readback = nullptr;
    }

    velri_captured_graph *cg = (velri_captured_graph *)calloc(1U, sizeof(*cg));
    if (!cg) {
        return VELRI_ERR_NOMEM;
    }

    velri_os_mutex_lock(dev->mtx);

    velri_status st = ensure_enc_scratch_locked(dev, count);
    if (st == VELRI_OK) {
        st = velri_vk_init_cmd_slot(dev, &cg->pool, &cg->cmd, &cg->fence);
    }
    if (st == VELRI_OK) {
        st = velri_vk_alloc_step_slot_locked(dev, &cg->step_slab, &cg->step_slot, &cg->step_bda,
                                             &cg->step_map);
        if (st == VELRI_OK) {
            memset(cg->step_map, 0, sizeof(*cg->step_map));
        }
    }
    if (st != VELRI_OK) {
        free_captured_graph_locked(dev, cg);
        velri_os_mutex_unlock(dev->mtx);
        return st;
    }

    bool seen_pos_op = false;
    bool seen_splitkv_attn = false;
    cg->splitkv_uniform = true;
    cg->max_pos_offset = UINT32_MAX;
    cg->max_kv_pos = UINT32_MAX;

    vk_encoded_dispatch *enc = dev->enc_scratch;
    for (uint32_t i = 0U; i < count; ++i) {
        /* allow_splitk=true lets static multi-row GEMMs select VK_SPLITK_GEMM so capture rejects
         * them cleanly (preserving Split-K GEMM execution), while Split-KV is skipped at kv_pos=0. */
        st = encode_item_locked(dev, &items[i], (uint64_t)cg->step_bda, true, 0U, &enc[i]);
        if (st != VELRI_OK) {
            free_captured_graph_locked(dev, cg);
            velri_os_mutex_unlock(dev->mtx);
            return st;
        }
        if (enc[i].splitk_kind != VK_SPLITK_NONE) {
            free_captured_graph_locked(dev, cg);
            velri_os_mutex_unlock(dev->mtx);
            return VELRI_ERR_UNSUPPORTED;
        }

        const velri_resolved_op *op = items[i].op;
        if (op->kind == VELRI_OP_ROPE ||
            (op->kind == VELRI_OP_EMBEDDING && !velri_op_embedding_has_ids(op))) {
            size_t def_pos = (op->kind == VELRI_OP_ROPE) ? op->raw.attr.rope.pos_offset
                                                         : op->raw.attr.embedding.pos_offset;
            if (def_pos > (size_t)UINT32_MAX) {
                cg->require_explicit_pos_offset = true;
            } else if (!seen_pos_op) {
                cg->default_pos_offset = (uint32_t)def_pos;
                seen_pos_op = true;
            } else if (cg->default_pos_offset != (uint32_t)def_pos) {
                cg->require_explicit_pos_offset = true;
            }
            uint32_t nt = 1U;
            (void)velri_op_active_rows(op, items[i].params, &nt);
            size_t tbl_rows = 0U;
            bool have_tbl = false;
            if (op->kind == VELRI_OP_ROPE) {
                size_t half = op->raw.attr.rope.rotary_dim / 2U;
                size_t ct_n = 0U, st_n = 0U;
                if (half > 0U && velri_tensor_view_num_elements(&op->inputs[1], &ct_n) == VELRI_OK &&
                    velri_tensor_view_num_elements(&op->inputs[2], &st_n) == VELRI_OK) {
                    tbl_rows = (ct_n < st_n ? ct_n : st_n) / half;
                    have_tbl = true;
                }
            } else {
                tbl_rows = op->inputs[0].dims[0];
                have_tbl = true;
            }
            if (have_tbl) {
                size_t rem = (tbl_rows >= (size_t)nt) ? (tbl_rows - (size_t)nt) : 0U;
                uint32_t lim = (rem > (size_t)UINT32_MAX) ? UINT32_MAX : (uint32_t)rem;
                cg->has_pos_bound = true;
                if (lim < cg->max_pos_offset) {
                    cg->max_pos_offset = lim;
                }
            }
        } else if (op->kind == VELRI_OP_ATTENTION) {
            uint32_t max_ctx = (uint32_t)op->outputs[1].dims[1];
            uint32_t kv_lim = (max_ctx > 0U) ? (max_ctx - 1U) : 0U;
            cg->has_kv_bound = true;
            if (kv_lim < cg->max_kv_pos) {
                cg->max_kv_pos = kv_lim;
            }
            uint32_t dk = op->raw.attr.attention.head_dim;
            uint32_t dv = velri_op_attn_dv(op);
            if (!velri_op_attn_is_wide(dk, dv)) {
                uint32_t nh = op->raw.attr.attention.n_heads;
                uint32_t sw = op->raw.attr.attention.sliding_window;
                uint32_t eff_max_kv = (sw > 0U && sw < max_ctx) ? sw : max_ctx;
                uint32_t min_len = velri_vk_splitkv_min_len(dev, op);
                uint32_t split_limit = velri_vk_splitkv_limit(dev, op);
                uint32_t max_sp = velri_vk_choose_split_kv(nh, dv, eff_max_kv, min_len, split_limit);
                size_t min_bytes = 0U;
                if (max_sp > 1U &&
                    velri_vk_splitkv_layout(2U, nh, dv, nullptr, &min_bytes) &&
                    dev->splitk_buf && dev->splitk_buf->size >= min_bytes) {
                    cg->splitkv_possible = true;

                    if (!seen_splitkv_attn) {
                        seen_splitkv_attn = true;
                        cg->splitkv_min_len = min_len;
                        cg->splitkv_nh = nh;
                        cg->splitkv_dv = dv;
                        cg->splitkv_max_ctx = max_ctx;
                        cg->splitkv_sw = sw;
                        cg->splitkv_max_splits = max_sp;
                        cg->splitkv_limit = split_limit;
                    } else if (cg->splitkv_nh != nh || cg->splitkv_dv != dv ||
                               cg->splitkv_max_ctx != max_ctx || cg->splitkv_sw != sw ||
                               cg->splitkv_min_len != min_len || cg->splitkv_limit != split_limit) {
                        cg->splitkv_uniform = false;
                        if (min_len < cg->splitkv_min_len) cg->splitkv_min_len = min_len;
                    }
                }
            }
        }
    }

    /* Record primary persistent command buffer (flags = 0: Executable <-> Pending). */
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = 0U,
    };
    if (dev->vkBeginCommandBuffer(cg->cmd, &bi) != VK_SUCCESS) {
        free_captured_graph_locked(dev, cg);
        velri_os_mutex_unlock(dev->mtx);
        return VELRI_ERR_EXEC;
    }
    const vk_buf *source = (const vk_buf *)io_arena;
    if (source && out_readback && range.bytes > 0U && range.offset <= source->size &&
        range.bytes <= source->size - range.offset && range.bytes <= VK_MAPPED_IO_MAX_BYTES) {
        (void)velri_vk_alloc_readback_locked(dev, range.bytes, &cg->readback);
    }
    record_encoded_batch(dev, cg->cmd, items, enc, count);
    record_output_readback(dev, cg->cmd, source, range, cg->readback);
    if (dev->vkEndCommandBuffer(cg->cmd) != VK_SUCCESS) {
        free_captured_graph_locked(dev, cg);
        velri_os_mutex_unlock(dev->mtx);
        return VELRI_ERR_EXEC;
    }

    /* Pre-record bucketed Split-KV command buffers ({2,4,8,16,32}) on the SAME pool & step slot
     * so split decode also runs without host command recording. */
    if (cg->splitkv_possible && cg->splitkv_uniform && cg->splitkv_max_splits >= 2U) {
        cg->splitkv_buf_gen = dev->splitk_buf_gen;
        for (uint32_t b = 0U; b < VK_SPLITKV_BUCKETS; ++b) {
            uint32_t b_splits = 2U << b;
            if (b_splits > cg->splitkv_max_splits) {
                break;
            }
            size_t need_bytes = 0U;
            if (!velri_vk_splitkv_layout(b_splits, cg->splitkv_nh, cg->splitkv_dv, nullptr,
                                         &need_bytes) ||
                !dev->splitk_buf || dev->splitk_buf->size < need_bytes) {
                break;
            }
            bool bucket_ok = true;
            for (uint32_t i = 0U; i < count; ++i) {
                if (encode_item_locked(dev, &items[i], (uint64_t)cg->step_bda, true, b_splits,
                                       &enc[i]) != VELRI_OK) {
                    bucket_ok = false;
                    break;
                }
            }
            if (!bucket_ok) {
                break;
            }
            VkCommandBufferAllocateInfo cb_ai = {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .commandPool = cg->pool,
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = 1U,
            };
            VkCommandBuffer bcmd = VK_NULL_HANDLE;
            if (dev->vkAllocateCommandBuffers(dev->device, &cb_ai, &bcmd) != VK_SUCCESS ||
                dev->vkBeginCommandBuffer(bcmd, &bi) != VK_SUCCESS) {
                break;
            }
            record_encoded_batch(dev, bcmd, items, enc, count);
            record_output_readback(dev, bcmd, source, range, cg->readback);
            if (dev->vkEndCommandBuffer(bcmd) != VK_SUCCESS) {
                break;
            }
            cg->splitkv_cmds[b] = bcmd;
        }
    }

    cg->comp.pool = cg->pool;
    cg->comp.cmd = cg->cmd;
    cg->comp.fence = cg->fence;
    cg->comp.in_use = false;
    cg->comp.uses_splitk = false;
    cg->comp.is_captured_graph = true;

    if (cg->readback) {
        *out_readback = cg->readback->host_map;
    }
    *out_graph = cg;
    velri_os_mutex_unlock(dev->mtx);
    return VELRI_OK;
}

static velri_status vk_capture_graph(velri_backend_device *dev, const velri_submit_item *items,
                                     uint32_t count, velri_captured_graph **out_graph) {
    return vk_capture_outputs(dev, items, count, nullptr, (velri_io_range){0}, nullptr, out_graph);
}

static void vk_destroy_captured_graph(velri_backend_device *dev, velri_captured_graph *graph) {
    if (!dev || !graph) {
        return;
    }
    velri_os_mutex_lock(dev->mtx);
    free_captured_graph_locked(dev, graph);
    velri_os_mutex_unlock(dev->mtx);
}

static velri_status vk_replay_graph(velri_backend_device *dev, velri_captured_graph *graph,
                                    const velri_step_params *params,
                                    velri_completion **out_completion) {
    if (!out_completion) {
        return VELRI_ERR_INVALID;
    }
    *out_completion = nullptr;
    if (!dev || !graph) {
        return VELRI_ERR_INVALID;
    }
    if (params && params->has_seq_len && params->seq_len_override != 1U) {
        return VELRI_ERR_UNSUPPORTED;
    }
    if (params && params->has_pos_offset && params->pos_offset_override > (size_t)UINT32_MAX) {
        return VELRI_ERR_OVERFLOW;
    }
    if (graph->require_explicit_pos_offset && (!params || !params->has_pos_offset)) {
        return VELRI_ERR_UNSUPPORTED;
    }

    uint32_t eff_pos = (params && params->has_pos_offset) ? (uint32_t)params->pos_offset_override
                                                          : graph->default_pos_offset;
    uint32_t eff_kv = 0U;
    velri_status kv_st = velri_op_resolve_kv_pos(params, &eff_kv);
    if (kv_st != VELRI_OK) {
        return kv_st;
    }
    if ((graph->has_pos_bound && eff_pos > graph->max_pos_offset) ||
        (graph->has_kv_bound && eff_kv > graph->max_kv_pos)) {
        return VELRI_ERR_INVALID;
    }

    uint32_t bucket = VK_SPLITKV_BUCKETS;
    if (graph->splitkv_possible) {
        if (!graph->splitkv_uniform) {
            if (eff_kv >= graph->splitkv_min_len - 1U) {
                return VELRI_ERR_UNSUPPORTED;
            }
        } else {
            uint64_t kv_end64 = (uint64_t)eff_kv + 1ULL;
            uint32_t kv_end = (kv_end64 < (uint64_t)graph->splitkv_max_ctx)
                                  ? (uint32_t)kv_end64
                                  : graph->splitkv_max_ctx;
            uint32_t eff_kv_len =
                (graph->splitkv_sw > 0U && kv_end > graph->splitkv_sw) ? graph->splitkv_sw : kv_end;
            uint32_t want_sp =
                velri_vk_choose_split_kv(graph->splitkv_nh, graph->splitkv_dv, eff_kv_len,
                                         graph->splitkv_min_len, graph->splitkv_limit);
            if (want_sp >= 2U) {
                bucket = splitkv_bucket_index(want_sp);
                while (bucket < VK_SPLITKV_BUCKETS && !graph->splitkv_cmds[bucket]) {
                    bucket = bucket > 0U ? bucket - 1U : VK_SPLITKV_BUCKETS;
                }
            }
        }
    }

    velri_os_mutex_lock(dev->mtx);
    if (graph->comp.in_use) {
        velri_os_mutex_unlock(dev->mtx);
        return VELRI_ERR_BUSY;
    }

    bool use_splitk = (bucket < VK_SPLITKV_BUCKETS);
    VkCommandBuffer target_cmd = use_splitk ? graph->splitkv_cmds[bucket] : graph->cmd;
    if (use_splitk) {
        graph->comp.in_use = true; /* Reserve graph before any cond_wait */
        while (dev->splitk_busy) {
            velri_os_cond_wait(dev->slot_cv, dev->mtx);
        }
        if (graph->splitkv_buf_gen != dev->splitk_buf_gen) {
            graph->comp.in_use = false;
            velri_os_mutex_unlock(dev->mtx);
            return VELRI_ERR_UNSUPPORTED;
        }
        dev->splitk_busy = true;
    } else {
        graph->comp.in_use = true;
    }

    /* Vulkan 1.3 §6.9 guarantees HOST_COHERENT writes before vkQueueSubmit are shader-visible. */
    graph->step_map->seq_len = 1U;
    graph->step_map->pos_offset = eff_pos;
    graph->step_map->kv_pos = eff_kv;

    if (dev->vkResetFences(dev->device, 1U, &graph->fence) != VK_SUCCESS) {
        graph->comp.in_use = false;
        if (use_splitk) {
            dev->splitk_busy = false;
            velri_os_cond_broadcast(dev->slot_cv);
        }
        velri_os_mutex_unlock(dev->mtx);
        return VELRI_ERR_EXEC;
    }
    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1U,
        .pCommandBuffers = &target_cmd,
    };
    if (dev->vkQueueSubmit(dev->queue, 1U, &si, graph->fence) != VK_SUCCESS) {
        graph->comp.in_use = false;
        if (use_splitk) {
            dev->splitk_busy = false;
            velri_os_cond_broadcast(dev->slot_cv);
        }
        velri_os_mutex_unlock(dev->mtx);
        return VELRI_ERR_EXEC;
    }

    graph->comp.cmd = target_cmd;
    graph->comp.uses_splitk = use_splitk;
    *out_completion = &graph->comp;
    velri_os_mutex_unlock(dev->mtx);
    return VELRI_OK;
}

/* ---- 6. Completion Backend Contract ---- */

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
    if (comp->is_captured_graph) {
        comp->in_use = false;
        if (comp->uses_splitk) {
            dev->splitk_busy = false;
            comp->uses_splitk = false;
            velri_os_cond_broadcast(dev->slot_cv);
        }
    } else {
        release_slot_locked(dev, comp);
    }
    velri_os_mutex_unlock(dev->mtx);
}

const velri_device_hooks velri_vulkan_device_hooks = {
    .alloc_io = velri_vk_alloc_io,
    .max_readback_bytes = VK_MAPPED_IO_MAX_BYTES,
    .capture_outputs = vk_capture_outputs,
};

static const velri_backend_vtable g_vk_vtable = {
    .name = "vulkan",
    .contract_version = VELRI_BACKEND_CONTRACT_VERSION,
    .enumerate_devices = velri_vk_enumerate_devices,
    .open_device = velri_vk_open_device,
    .query_caps = velri_vk_query_caps,
    .close_device = velri_vk_close_device,
    .alloc = velri_vk_alloc,
    .free = velri_vk_free,
    .copy_to_ram = velri_vk_copy_to_ram,
    .copy_from_ram = velri_vk_copy_from_ram,
    .zero_mem = velri_vk_zero_mem,
    .query_operation = velri_vk_query_operation,
    .prepare_op = velri_vk_prepare_op,
    .destroy_prepared_op = velri_vk_destroy_prepared_op,
    .submit = vk_submit,
    .submit_batch = vk_submit_batch,
    .wait = vk_wait,
    .destroy_completion = vk_destroy_completion,
    .capture_graph = vk_capture_graph,
    .destroy_captured_graph = vk_destroy_captured_graph,
    .replay_graph = vk_replay_graph,
};

const velri_backend_vtable *velri_vulkan_backend_vtable(void) { return &g_vk_vtable; }
