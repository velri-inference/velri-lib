#ifndef VELRI_OPS_INTERNAL_H
#define VELRI_OPS_INTERNAL_H

#include "util_internal.h"
#include "velri_lib/backend.h"

#include <math.h>

/*
 * Central Operation Metadata Registry, Linear Classification, ABI Flag Constants,
 * and Unified Dynamic Step-Parameter Semantics.
 *
 * Single source of truth shared by `graph/`, `plan/`, `runtime/`, and all backends.
 * Neither `planner.c` nor `executor.c` should switch on `velri_op_kind`.
 */

enum {
    VELRI_ATTN_FAST_HEAD_DIM = 256,

    /* Cross-language (C <-> Slang) epilogue bitflags for Linear & BatchedLinear. */
    VELRI_EP_FLAG_BIAS    = 1U << 0,
    VELRI_EP_FLAG_X_SCALE = 1U << 1,
    VELRI_EP_FLAG_W_SCALE = 1U << 2,

    /* Cross-language (C <-> Slang) FWHT sign-vector modes. */
    VELRI_FWHT_SUH_NONE = 0U,
    VELRI_FWHT_SUH_F32  = 1U,
    VELRI_FWHT_SUH_F16  = 2U
};

typedef enum {
    VELRI_LIN_UNSUPPORTED = -1,
    VELRI_LIN_F32         = 0,
    VELRI_LIN_F16_F32     = 1,
    VELRI_LIN_BF16_F32    = 2,
    VELRI_LIN_F32X_BF16W  = 3,
    VELRI_LIN_F64         = 4,
    VELRI_LIN_W8A8_F32    = 5
} velri_linear_mode;

typedef struct {
    uint32_t min_inputs;
    uint32_t max_inputs;
    uint32_t num_outputs;
    uint32_t req_input_mask;   /* Bit i => inputs[i] must be present */
    uint32_t rmw_output_mask;  /* Bit o => outputs[o] is read-modify-write (must preserve on offload) */
    bool always_dynamic_seq;   /* Axis-0 scales with seq_len_override even when op.dynamic_seq==false */
    bool rows_from_output0;    /* True for EMBEDDING (outputs[0].dims[0]); false => inputs[0].dims[0] */
} velri_op_meta;

typedef struct {
    uint32_t active_t;
    uint32_t kv_pos;
    uint32_t d_k;
    uint32_t d_v;
    uint32_t max_ctx;
    float scale;
} velri_attn_step_args;

/* Declarative op metadata table & validators (implemented in `src/ops/registry.c`). */
const velri_op_meta *velri_op_get_meta(velri_op_kind kind);
velri_linear_mode velri_linear_classify(const velri_linear_desc *d);
velri_status velri_op_validate(const velri_resolved_op *op);
velri_status velri_op_validate_step_params(const velri_resolved_op *op,
                                           const velri_step_params *params);

static inline bool velri_op_is_dynamic_seq(const velri_resolved_op *op) {
    if (!op) {
        return false;
    }
    const velri_op_meta *m = velri_op_get_meta(op->kind);
    return op->raw.dynamic_seq || (m && m->always_dynamic_seq);
}

static inline bool velri_op_output_is_rmw(velri_op_kind kind, uint32_t out_idx) {
    const velri_op_meta *m = velri_op_get_meta(kind);
    return m && out_idx < 32U && ((m->rmw_output_mask & (1U << out_idx)) != 0U);
}

static inline size_t velri_op_static_rows(const velri_resolved_op *op) {
    if (!op) {
        return 0U;
    }
    const velri_op_meta *m = velri_op_get_meta(op->kind);
    if (m && m->rows_from_output0) {
        return op->num_outputs > 0U ? op->outputs[0].dims[0] : 0U;
    }
    return op->num_inputs > 0U ? op->inputs[0].dims[0] : 0U;
}

static inline velri_status velri_op_active_rows(const velri_resolved_op *op,
                                                const velri_step_params *params,
                                                uint32_t *out_rows) {
    size_t static_rows = velri_op_static_rows(op);
    size_t active = static_rows;
    if (params && params->has_seq_len && velri_op_is_dynamic_seq(op)) {
        if (params->seq_len_override == 0U || params->seq_len_override > static_rows) {
            return VELRI_ERR_INVALID;
        }
        active = params->seq_len_override;
    }
    if (active == 0U) {
        return VELRI_ERR_INVALID;
    }
    if (active > (size_t)UINT32_MAX) {
        return VELRI_ERR_OVERFLOW;
    }
    *out_rows = (uint32_t)active;
    return VELRI_OK;
}

static inline velri_status velri_op_effective_elemwise_count(const velri_resolved_op *op,
                                                             const velri_tensor_view *v,
                                                             const velri_step_params *params,
                                                             uint32_t *out_count) {
    size_t n = 0U;
    velri_status st = velri_tensor_view_num_elements(v, &n);
    if (st != VELRI_OK) {
        return st;
    }
    if (params && params->has_seq_len && velri_op_is_dynamic_seq(op) && v->rank >= 1U &&
        v->dims[0] > 0U) {
        if (params->seq_len_override == 0U || params->seq_len_override > v->dims[0]) {
            return VELRI_ERR_INVALID;
        }
        n = params->seq_len_override * (n / v->dims[0]);
    }
    if (n == 0U) {
        return VELRI_ERR_INVALID;
    }
    if (n > (size_t)UINT32_MAX) {
        return VELRI_ERR_OVERFLOW;
    }
    *out_count = (uint32_t)n;
    return VELRI_OK;
}

static inline velri_status velri_op_resolve_pos_offset(size_t static_pos,
                                                       const velri_step_params *params,
                                                       uint32_t *out_pos) {
    size_t pos = (params && params->has_pos_offset) ? params->pos_offset_override : static_pos;
    if (pos > (size_t)UINT32_MAX) {
        return VELRI_ERR_OVERFLOW;
    }
    *out_pos = (uint32_t)pos;
    return VELRI_OK;
}

static inline velri_status velri_op_resolve_rope_step(const velri_resolved_op *op,
                                                      const velri_step_params *params,
                                                      uint32_t *out_rows, uint32_t *out_pos) {
    uint32_t active_t = 0U;
    velri_status st = velri_op_active_rows(op, params, &active_t);
    if (st != VELRI_OK) {
        return st;
    }
    uint32_t pos = 0U;
    st = velri_op_resolve_pos_offset(op->raw.attr.rope.pos_offset, params, &pos);
    if (st != VELRI_OK) {
        return st;
    }
    size_t end_pos = 0U, need = 0U;
    if (velri_add_overflow_size((size_t)pos, (size_t)active_t, &end_pos) ||
        velri_mul_overflow_size(end_pos, op->raw.attr.rope.rotary_dim / 2U, &need)) {
        return VELRI_ERR_OVERFLOW;
    }
    size_t ct_n = 0U, st_n = 0U;
    if (velri_tensor_view_num_elements(&op->inputs[1], &ct_n) != VELRI_OK ||
        velri_tensor_view_num_elements(&op->inputs[2], &st_n) != VELRI_OK ||
        ct_n < need || st_n < need) {
        return VELRI_ERR_INVALID;
    }
    if (out_rows) {
        *out_rows = active_t;
    }
    if (out_pos) {
        *out_pos = pos;
    }
    return VELRI_OK;
}

static inline velri_status velri_op_resolve_kv_pos(const velri_step_params *params,
                                                   uint32_t *out_kv_pos) {
    size_t kv_pos = 0U;
    if (params) {
        if (params->has_kv_len) {
            kv_pos = params->kv_len_override;
        } else if (params->has_pos_offset) {
            kv_pos = params->pos_offset_override;
        }
    }
    if (kv_pos > (size_t)UINT32_MAX) {
        return VELRI_ERR_OVERFLOW;
    }
    *out_kv_pos = (uint32_t)kv_pos;
    return VELRI_OK;
}

static inline uint32_t velri_op_attn_dv(const velri_resolved_op *op) {
    uint32_t dk = op->raw.attr.attention.head_dim;
    return op->raw.attr.attention.v_head_dim ? op->raw.attr.attention.v_head_dim : dk;
}

static inline float velri_op_attn_scale(const velri_resolved_op *op) {
    return (op->raw.attr.attention.scale > 0.0f)
               ? op->raw.attr.attention.scale
               : (1.0f / sqrtf((float)op->raw.attr.attention.head_dim));
}

static inline bool velri_op_attn_is_wide(uint32_t dk, uint32_t dv) {
    return dk > VELRI_ATTN_FAST_HEAD_DIM || dv > VELRI_ATTN_FAST_HEAD_DIM;
}

static inline velri_status velri_op_resolve_attention_step(const velri_resolved_op *op,
                                                           const velri_step_params *params,
                                                           velri_attn_step_args *out) {
    uint32_t active_t = 0U;
    velri_status st = velri_op_active_rows(op, params, &active_t);
    if (st != VELRI_OK) {
        return st;
    }
    uint32_t kv_pos = 0U;
    st = velri_op_resolve_kv_pos(params, &kv_pos);
    if (st != VELRI_OK) {
        return st;
    }
    size_t max_ctx_sz = op->outputs[1].dims[1];
    size_t kv_end = 0U;
    if (velri_add_overflow_size((size_t)kv_pos, (size_t)active_t, &kv_end) ||
        max_ctx_sz > (size_t)UINT32_MAX) {
        return VELRI_ERR_OVERFLOW;
    }
    if (kv_end > max_ctx_sz) {
        return VELRI_ERR_INVALID;
    }
    out->active_t = active_t;
    out->kv_pos = kv_pos;
    out->d_k = op->raw.attr.attention.head_dim;
    out->d_v = velri_op_attn_dv(op);
    out->max_ctx = (uint32_t)max_ctx_sz;
    out->scale = velri_op_attn_scale(op);
    return VELRI_OK;
}

static inline bool velri_op_embedding_has_ids(const velri_resolved_op *op) {
    return op && op->num_inputs >= 2U && op->input_present[1];
}

static inline uint32_t velri_op_linear_epilogue_flags(const velri_resolved_op *op) {
    return ((op->num_inputs > 2U && op->input_present[2]) ? VELRI_EP_FLAG_BIAS : 0U) |
           ((op->num_inputs > 3U && op->input_present[3]) ? VELRI_EP_FLAG_X_SCALE : 0U) |
           ((op->num_inputs > 4U && op->input_present[4]) ? VELRI_EP_FLAG_W_SCALE : 0U);
}

static inline uint32_t velri_op_fwht_suh_mode(const velri_resolved_op *op) {
    if (!(op->num_inputs > 1U && op->input_present[1])) {
        return VELRI_FWHT_SUH_NONE;
    }
    return (op->raw.attr.fwht.suh_dtype == VELRI_DTYPE_F32) ? VELRI_FWHT_SUH_F32
                                                            : VELRI_FWHT_SUH_F16;
}

static inline bool velri_submit_item_header_valid(const velri_submit_item *item) {
    return item && item->op && item->num_inputs == item->op->num_inputs &&
           item->num_outputs == item->op->num_outputs;
}

#endif /* VELRI_OPS_INTERNAL_H */
