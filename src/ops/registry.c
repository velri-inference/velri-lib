#include "core_internal.h"
#include <math.h>

/*
 * Central operation metadata registry and validator.
 * Device-agnostic single source of truth for op metadata, linear dtype classification,
 * shapes, dtypes, strides, attributes, 32-bit kernel index bounds, memory aliasing rules,
 * and dynamic step-parameter bounds across all backends.
 */

static const velri_op_meta k_op_meta[] = {
    [VELRI_OP_LINEAR]            = {2U, 5U, 1U, 0x3U, 0x0U, false, false},
    [VELRI_OP_RMSNORM]           = {1U, 2U, 1U, 0x1U, 0x0U, false, false},
    [VELRI_OP_FUSED_ADD_RMSNORM] = {1U, 2U, 2U, 0x1U, 0x1U, false, false},
    [VELRI_OP_LAYERNORM]         = {1U, 3U, 1U, 0x1U, 0x0U, false, false},
    [VELRI_OP_ROPE]              = {3U, 3U, 1U, 0x7U, 0x0U, false, false},
    [VELRI_OP_ACTIVATION]        = {1U, 1U, 1U, 0x1U, 0x0U, false, false},
    [VELRI_OP_GATED_MLP]         = {2U, 2U, 1U, 0x3U, 0x0U, false, false},
    [VELRI_OP_SOFTMAX]           = {1U, 1U, 1U, 0x1U, 0x0U, false, false},
    [VELRI_OP_VEC_AXPBY]         = {1U, 2U, 1U, 0x1U, 0x0U, false, false},
    [VELRI_OP_VEC_MUL]           = {2U, 2U, 1U, 0x3U, 0x0U, false, false},
    [VELRI_OP_FWHT]              = {1U, 2U, 1U, 0x1U, 0x0U, false, false},
    [VELRI_OP_QUANTIZE_ROW]      = {1U, 1U, 2U, 0x1U, 0x0U, false, false},
    [VELRI_OP_EMBEDDING]         = {1U, 2U, 1U, 0x1U, 0x0U, true,  true},
    [VELRI_OP_ATTENTION]         = {3U, 3U, 3U, 0x7U, 0x6U, true,  false},
    [VELRI_OP_COPY_LAST_ROW]     = {1U, 1U, 1U, 0x1U, 0x0U, true,  false},
    [VELRI_OP_BATCHED_LINEAR]    = {2U, 3U, 1U, 0x3U, 0x0U, false, false},
};

static_assert(sizeof(k_op_meta) / sizeof(k_op_meta[0]) == 16U,
              "k_op_meta must cover all 16 velri_op_kind values");

const velri_op_meta *velri_op_get_meta(velri_op_kind kind) {
    unsigned int idx = (unsigned int)kind;
    if (idx >= sizeof(k_op_meta) / sizeof(k_op_meta[0])) {
        return nullptr;
    }
    return &k_op_meta[idx];
}

velri_linear_mode velri_linear_classify(const velri_linear_desc *d) {
    if (!d) {
        return VELRI_LIN_UNSUPPORTED;
    }
    if (d->x_dtype == VELRI_DTYPE_I8 && d->w_dtype == VELRI_DTYPE_I8 &&
        d->y_dtype == VELRI_DTYPE_F32) {
        return (d->has_x_scale && d->has_w_scale) ? VELRI_LIN_W8A8_F32 : VELRI_LIN_UNSUPPORTED;
    }
    if (d->has_x_scale || d->has_w_scale) {
        return VELRI_LIN_UNSUPPORTED;
    }
    if (d->x_dtype == VELRI_DTYPE_F32 && d->w_dtype == VELRI_DTYPE_F32 &&
        d->y_dtype == VELRI_DTYPE_F32) {
        return VELRI_LIN_F32;
    }
    if (d->x_dtype == VELRI_DTYPE_F16 && d->w_dtype == VELRI_DTYPE_F16 &&
        d->y_dtype == VELRI_DTYPE_F32) {
        return VELRI_LIN_F16_F32;
    }
    if (d->x_dtype == VELRI_DTYPE_BF16 && d->w_dtype == VELRI_DTYPE_BF16 &&
        d->y_dtype == VELRI_DTYPE_F32) {
        return VELRI_LIN_BF16_F32;
    }
    if (d->x_dtype == VELRI_DTYPE_F32 && d->w_dtype == VELRI_DTYPE_BF16 &&
        d->y_dtype == VELRI_DTYPE_F32) {
        return VELRI_LIN_F32X_BF16W;
    }
    if (d->x_dtype == VELRI_DTYPE_F64 && d->w_dtype == VELRI_DTYPE_F64 &&
        d->y_dtype == VELRI_DTYPE_F64) {
        return VELRI_LIN_F64;
    }
    return VELRI_LIN_UNSUPPORTED;
}

static bool fits_u32(size_t v) { return v <= (size_t)UINT32_MAX; }

static bool is_unit_1d(const velri_tensor_view *v) {
    return v->rank == 1U && v->dims[0] > 0U && v->strides[0] == 1U && fits_u32(v->dims[0]);
}

static bool is_unit_inner_2d(const velri_tensor_view *v) {
    return v->rank == 2U && v->dims[0] > 0U && v->dims[1] > 0U &&
           v->strides[1] == 1U && v->strides[0] >= v->dims[1] &&
           fits_u32(v->dims[0]) && fits_u32(v->dims[1]) && fits_u32(v->strides[0]);
}

static bool is_nonoverlapping_unit_inner_3d(const velri_tensor_view *v) {
    if (v->rank != 3U || v->dims[0] == 0U || v->dims[1] == 0U || v->dims[2] == 0U ||
        v->strides[2] != 1U || v->strides[1] < v->dims[2] ||
        !fits_u32(v->dims[0]) || !fits_u32(v->dims[1]) || !fits_u32(v->dims[2]) ||
        !fits_u32(v->strides[0]) || !fits_u32(v->strides[1])) {
        return false;
    }
    size_t min_s0 = 0U;
    return !velri_mul_overflow_size(v->dims[1], v->strides[1], &min_s0) &&
           v->strides[0] >= min_s0;
}

static bool check_same_2d(const velri_tensor_view *a, const velri_tensor_view *b) {
    return is_unit_inner_2d(a) && is_unit_inner_2d(b) &&
           a->dtype == b->dtype && a->dims[0] == b->dims[0] && a->dims[1] == b->dims[1];
}

static bool check_opt_1d_param(const velri_resolved_op *op, uint32_t idx, velri_dtype dt,
                               size_t len, const velri_tensor_view *out0,
                               const velri_tensor_view *out1) {
    if (idx >= op->num_inputs || !op->input_present[idx]) {
        return true;
    }
    const velri_tensor_view *p = &op->inputs[idx];
    if (!is_unit_1d(p) || p->dtype != dt || p->dims[0] != len) {
        return false;
    }
    if (out0 && velri_tensor_view_overlaps(p, out0)) {
        return false;
    }
    if (out1 && velri_tensor_view_overlaps(p, out1)) {
        return false;
    }
    return true;
}

static velri_status check_elemwise_f32_view(const velri_tensor_view *v, size_t *out_n) {
    if (!velri_tensor_view_is_contiguous(v) || v->dtype != VELRI_DTYPE_F32) {
        return VELRI_ERR_INVALID;
    }
    size_t n = 0U;
    velri_status st = velri_tensor_view_num_elements(v, &n);
    if (st != VELRI_OK) {
        return st;
    }
    if (!fits_u32(n)) {
        return VELRI_ERR_OVERFLOW;
    }
    *out_n = n;
    return VELRI_OK;
}

static velri_status validate_elemwise_f32(const velri_resolved_op *op) {
    size_t ny = 0U;
    velri_status st = check_elemwise_f32_view(&op->outputs[0], &ny);
    if (st != VELRI_OK) {
        return st;
    }
    for (uint32_t i = 0U; i < op->num_inputs; ++i) {
        if (!op->input_present[i]) {
            continue;
        }
        size_t ni = 0U;
        st = check_elemwise_f32_view(&op->inputs[i], &ni);
        if (st != VELRI_OK) {
            return st;
        }
        if (ni != ny || !velri_tensor_view_no_partial_overlap(&op->inputs[i], &op->outputs[0])) {
            return VELRI_ERR_INVALID;
        }
        if (velri_op_is_dynamic_seq(op) && op->inputs[i].dims[0] != op->outputs[0].dims[0]) {
            return VELRI_ERR_INVALID;
        }
    }
    return VELRI_OK;
}

static velri_status validate_norm_op(const velri_resolved_op *op, float eps, bool allow_f16_bf16,
                                     uint32_t max_opt_param_idx) {
    if (!(eps > 0.0f) || !isfinite(eps)) {
        return VELRI_ERR_INVALID;
    }
    const velri_tensor_view *x = &op->inputs[0];
    const velri_tensor_view *out0 = &op->outputs[0];
    const velri_tensor_view *out1 = (op->num_outputs > 1U) ? &op->outputs[1] : nullptr;

    if (!check_same_2d(x, out0) || !velri_tensor_view_no_partial_overlap(x, out0)) {
        return VELRI_ERR_INVALID;
    }
    if (out1 && (!check_same_2d(x, out1) || !velri_tensor_view_no_partial_overlap(x, out1) ||
                 velri_tensor_view_overlaps(out0, out1))) {
        return VELRI_ERR_INVALID;
    }
    if (allow_f16_bf16) {
        if (x->dtype != VELRI_DTYPE_F32 && x->dtype != VELRI_DTYPE_BF16 &&
            x->dtype != VELRI_DTYPE_F16) {
            return VELRI_ERR_UNSUPPORTED;
        }
    } else if (x->dtype != VELRI_DTYPE_F32) {
        return VELRI_ERR_INVALID;
    }
    for (uint32_t idx = 1U; idx <= max_opt_param_idx; ++idx) {
        if (!check_opt_1d_param(op, idx, x->dtype, x->dims[1], out0, out1)) {
            return VELRI_ERR_INVALID;
        }
    }
    return VELRI_OK;
}

static velri_status check_embedding_pos_bounds(const velri_resolved_op *op, size_t active_t,
                                               size_t pos) {
    size_t end_pos = 0U;
    if (!fits_u32(pos) || velri_add_overflow_size(pos, active_t, &end_pos)) {
        return VELRI_ERR_OVERFLOW;
    }
    return (end_pos <= op->inputs[0].dims[0]) ? VELRI_OK : VELRI_ERR_INVALID;
}

velri_status velri_op_validate(const velri_resolved_op *op) {
    if (!op || op->num_inputs > VELRI_OP_MAX_INPUTS || op->num_outputs == 0U ||
        op->num_outputs > VELRI_OP_MAX_OUTPUTS) {
        return VELRI_ERR_INVALID;
    }
    const velri_op_meta *meta = velri_op_get_meta(op->kind);
    if (!meta) {
        return VELRI_ERR_UNSUPPORTED;
    }
    if (op->num_inputs < meta->min_inputs || op->num_inputs > meta->max_inputs ||
        op->num_outputs != meta->num_outputs) {
        return VELRI_ERR_INVALID;
    }
    for (uint32_t i = 0U; i < op->num_inputs; ++i) {
        if (((meta->req_input_mask & (1U << i)) != 0U) && !op->input_present[i]) {
            return VELRI_ERR_INVALID;
        }
    }

    switch (op->kind) {
    case VELRI_OP_LINEAR: {
        const velri_tensor_view *x = &op->inputs[0];
        const velri_tensor_view *w = &op->inputs[1];
        const velri_tensor_view *y = &op->outputs[0];
        const velri_linear_desc *d = &op->raw.attr.linear.desc;
        if (!is_unit_inner_2d(x) || !is_unit_inner_2d(w) || !is_unit_inner_2d(y)) {
            return VELRI_ERR_INVALID;
        }
        if (x->dtype != d->x_dtype || w->dtype != d->w_dtype || y->dtype != d->y_dtype) {
            return VELRI_ERR_INVALID;
        }
        if (velri_linear_classify(d) == VELRI_LIN_UNSUPPORTED) {
            return VELRI_ERR_UNSUPPORTED;
        }
        if (x->dims[1] != w->dims[1] || x->dims[0] != y->dims[0] || w->dims[0] != y->dims[1]) {
            return VELRI_ERR_INVALID;
        }
        if (!check_opt_1d_param(op, 2U, d->y_dtype, w->dims[0], y, nullptr)) {
            return VELRI_ERR_INVALID;
        }
        if (d->has_x_scale) {
            if (op->num_inputs <= 3U || !op->input_present[3] ||
                !check_opt_1d_param(op, 3U, VELRI_DTYPE_F32, x->dims[0], y, nullptr)) {
                return VELRI_ERR_INVALID;
            }
        } else if (op->num_inputs > 3U && op->input_present[3]) {
            return VELRI_ERR_INVALID;
        }
        if (d->has_w_scale) {
            if (op->num_inputs <= 4U || !op->input_present[4] ||
                !check_opt_1d_param(op, 4U, VELRI_DTYPE_F32, w->dims[0], y, nullptr)) {
                return VELRI_ERR_INVALID;
            }
        } else if (op->num_inputs > 4U && op->input_present[4]) {
            return VELRI_ERR_INVALID;
        }
        if (velri_tensor_view_overlaps(x, y) || velri_tensor_view_overlaps(w, y)) {
            return VELRI_ERR_INVALID;
        }
        return VELRI_OK;
    }

    case VELRI_OP_RMSNORM:
        return validate_norm_op(op, op->raw.attr.rmsnorm.eps, true, 1U);

    case VELRI_OP_FUSED_ADD_RMSNORM:
        return validate_norm_op(op, op->raw.attr.rmsnorm.eps, false, 1U);

    case VELRI_OP_LAYERNORM:
        return validate_norm_op(op, op->raw.attr.layernorm.eps, false, 2U);

    case VELRI_OP_ROPE: {
        const velri_tensor_view *x = &op->inputs[0];
        const velri_tensor_view *ct = &op->inputs[1];
        const velri_tensor_view *st = &op->inputs[2];
        const velri_tensor_view *y = &op->outputs[0];
        if (!is_nonoverlapping_unit_inner_3d(x) || !is_nonoverlapping_unit_inner_3d(y) ||
            x->dtype != y->dtype ||
            x->dims[0] != y->dims[0] || x->dims[1] != y->dims[1] || x->dims[2] != y->dims[2]) {
            return VELRI_ERR_INVALID;
        }
        if (x->dtype != VELRI_DTYPE_F32 && x->dtype != VELRI_DTYPE_BF16) {
            return VELRI_ERR_UNSUPPORTED;
        }
        if (op->raw.attr.rope.mode != VELRI_ROPE_HALF &&
            op->raw.attr.rope.mode != VELRI_ROPE_INTERLEAVED) {
            return VELRI_ERR_UNSUPPORTED;
        }
        size_t rd = op->raw.attr.rope.rotary_dim;
        if (rd == 0U || (rd & 1U) != 0U || rd > x->dims[2] || !fits_u32(rd)) {
            return VELRI_ERR_INVALID;
        }
        if (!velri_tensor_view_is_contiguous(ct) || !velri_tensor_view_is_contiguous(st) ||
            ct->dtype != VELRI_DTYPE_F32 || st->dtype != VELRI_DTYPE_F32) {
            return VELRI_ERR_INVALID;
        }
        velri_status tbl_st = velri_op_resolve_rope_step(op, nullptr, nullptr, nullptr);
        if (tbl_st != VELRI_OK) {
            return tbl_st;
        }
        if (!velri_tensor_view_no_partial_overlap(x, y) ||
            velri_tensor_view_overlaps(ct, y) || velri_tensor_view_overlaps(st, y)) {
            return VELRI_ERR_INVALID;
        }
        return VELRI_OK;
    }

    case VELRI_OP_ACTIVATION: {
        velri_act_kind k = op->raw.attr.activation.kind;
        if (k != VELRI_ACT_SILU && k != VELRI_ACT_GELU_TANH && k != VELRI_ACT_RELU2) {
            return VELRI_ERR_UNSUPPORTED;
        }
        return validate_elemwise_f32(op);
    }

    case VELRI_OP_GATED_MLP: {
        velri_act_kind a = op->raw.attr.gated_mlp.act;
        if (a != VELRI_ACT_SILU && a != VELRI_ACT_GELU_TANH) {
            return VELRI_ERR_UNSUPPORTED;
        }
        return validate_elemwise_f32(op);
    }

    case VELRI_OP_VEC_MUL:
        return validate_elemwise_f32(op);

    case VELRI_OP_VEC_AXPBY: {
        velri_status st = validate_elemwise_f32(op);
        if (st != VELRI_OK) {
            return st;
        }
        if (!(op->num_inputs > 1U && op->input_present[1]) && op->raw.attr.axpby.beta != 0.0f) {
            return VELRI_ERR_INVALID;
        }
        return VELRI_OK;
    }

    case VELRI_OP_SOFTMAX: {
        const velri_tensor_view *x = &op->inputs[0];
        const velri_tensor_view *y = &op->outputs[0];
        if (x->dtype != VELRI_DTYPE_F32 || !check_same_2d(x, y) ||
            !velri_tensor_view_no_partial_overlap(x, y)) {
            return VELRI_ERR_INVALID;
        }
        return VELRI_OK;
    }

    case VELRI_OP_FWHT: {
        const velri_tensor_view *x = &op->inputs[0];
        const velri_tensor_view *y = &op->outputs[0];
        size_t bs = op->raw.attr.fwht.block_size;
        if (bs != 16U && bs != 32U && bs != 64U && bs != 128U && bs != 256U) {
            return VELRI_ERR_UNSUPPORTED;
        }
        if (x->dtype != VELRI_DTYPE_F32 || !check_same_2d(x, y) ||
            !velri_tensor_view_no_partial_overlap(x, y) || (x->dims[1] % bs) != 0U) {
            return VELRI_ERR_INVALID;
        }
        if (op->num_inputs > 1U && op->input_present[1]) {
            velri_dtype sdt = op->raw.attr.fwht.suh_dtype;
            if (sdt != VELRI_DTYPE_F32 && sdt != VELRI_DTYPE_F16) {
                return VELRI_ERR_UNSUPPORTED;
            }
            if (!check_opt_1d_param(op, 1U, sdt, x->dims[1], y, nullptr)) {
                return VELRI_ERR_INVALID;
            }
        }
        return VELRI_OK;
    }

    case VELRI_OP_QUANTIZE_ROW: {
        const velri_tensor_view *x = &op->inputs[0];
        const velri_tensor_view *q = &op->outputs[0];
        const velri_tensor_view *s = &op->outputs[1];
        if (!is_unit_inner_2d(x) || !is_unit_inner_2d(q) || !is_unit_1d(s) ||
            x->dtype != VELRI_DTYPE_F32 || q->dtype != VELRI_DTYPE_I8 ||
            s->dtype != VELRI_DTYPE_F32 ||
            x->dims[0] != q->dims[0] || x->dims[1] != q->dims[1] || s->dims[0] != x->dims[0]) {
            return VELRI_ERR_INVALID;
        }
        if (velri_tensor_view_overlaps(x, q) || velri_tensor_view_overlaps(x, s) ||
            velri_tensor_view_overlaps(q, s)) {
            return VELRI_ERR_INVALID;
        }
        return VELRI_OK;
    }

    case VELRI_OP_EMBEDDING: {
        const velri_tensor_view *table = &op->inputs[0];
        const velri_tensor_view *out = &op->outputs[0];
        if (!is_unit_inner_2d(table) || !is_unit_inner_2d(out) ||
            table->dtype != VELRI_DTYPE_F32 || out->dtype != VELRI_DTYPE_F32 ||
            table->dims[1] != out->dims[1] || velri_tensor_view_overlaps(table, out)) {
            return VELRI_ERR_INVALID;
        }
        if (velri_op_embedding_has_ids(op)) {
            const velri_tensor_view *ids = &op->inputs[1];
            if (!is_unit_1d(ids) || ids->dtype != VELRI_DTYPE_I32 ||
                ids->dims[0] != out->dims[0] || velri_tensor_view_overlaps(ids, out)) {
                return VELRI_ERR_INVALID;
            }
            return VELRI_OK;
        }
        return check_embedding_pos_bounds(op, out->dims[0], op->raw.attr.embedding.pos_offset);
    }

    case VELRI_OP_ATTENTION: {
        const velri_tensor_view *q = &op->inputs[0];
        const velri_tensor_view *k_in = &op->inputs[1];
        const velri_tensor_view *v_in = &op->inputs[2];
        const velri_tensor_view *out = &op->outputs[0];
        const velri_tensor_view *k_cache = &op->outputs[1];
        const velri_tensor_view *v_cache = &op->outputs[2];

        uint32_t nh = op->raw.attr.attention.n_heads;
        uint32_t nkvh = op->raw.attr.attention.n_kv_heads;
        uint32_t dk = op->raw.attr.attention.head_dim;
        uint32_t dv = velri_op_attn_dv(op);
        float sc = op->raw.attr.attention.scale;
        if (nh == 0U || nkvh == 0U || dk == 0U || dk > VELRI_ATTN_MAX_HEAD_DIM ||
            dv == 0U || dv > VELRI_ATTN_MAX_HEAD_DIM || (nh % nkvh) != 0U) {
            return VELRI_ERR_INVALID;
        }
        if (sc != 0.0f && (!(sc > 0.0f) || !isfinite(sc))) {
            return VELRI_ERR_INVALID;
        }
        if (!is_unit_inner_2d(q) || !is_unit_inner_2d(k_in) || !is_unit_inner_2d(v_in) ||
            !is_unit_inner_2d(out)) {
            return VELRI_ERR_INVALID;
        }
        if (q->dtype != VELRI_DTYPE_F32 || k_in->dtype != VELRI_DTYPE_F32 ||
            v_in->dtype != VELRI_DTYPE_F32 || out->dtype != VELRI_DTYPE_F32) {
            return VELRI_ERR_UNSUPPORTED;
        }
        size_t max_t = q->dims[0];
        if (k_in->dims[0] != max_t || v_in->dims[0] != max_t || out->dims[0] != max_t) {
            return VELRI_ERR_INVALID;
        }
        size_t q_dim = 0U, o_dim = 0U, k_dim = 0U, v_dim = 0U;
        if (velri_mul_overflow_size((size_t)nh, (size_t)dk, &q_dim) ||
            velri_mul_overflow_size((size_t)nh, (size_t)dv, &o_dim) ||
            velri_mul_overflow_size((size_t)nkvh, (size_t)dk, &k_dim) ||
            velri_mul_overflow_size((size_t)nkvh, (size_t)dv, &v_dim) ||
            !fits_u32(q_dim) || !fits_u32(o_dim) || !fits_u32(k_dim) || !fits_u32(v_dim)) {
            return VELRI_ERR_OVERFLOW;
        }
        if (q->dims[1] != q_dim || out->dims[1] != o_dim ||
            k_in->dims[1] != k_dim || v_in->dims[1] != v_dim) {
            return VELRI_ERR_INVALID;
        }
        if (!is_nonoverlapping_unit_inner_3d(k_cache) ||
            !is_nonoverlapping_unit_inner_3d(v_cache) ||
            (k_cache->dtype != VELRI_DTYPE_F32 && k_cache->dtype != VELRI_DTYPE_F16) ||
            v_cache->dtype != k_cache->dtype) {
            return VELRI_ERR_INVALID;
        }
        size_t max_ctx = k_cache->dims[1];
        if (k_cache->dims[0] != nkvh || v_cache->dims[0] != nkvh ||
            k_cache->dims[2] != dk || v_cache->dims[2] != dv ||
            v_cache->dims[1] != max_ctx || max_ctx < max_t) {
            return VELRI_ERR_INVALID;
        }
        if ((velri_tensor_view_overlaps(out, q) &&
             !(dk == dv && velri_tensor_view_identical_layout(q, out))) ||
            velri_tensor_view_overlaps(out, k_in) || velri_tensor_view_overlaps(out, v_in) ||
            velri_tensor_view_overlaps(out, k_cache) || velri_tensor_view_overlaps(out, v_cache) ||
            velri_tensor_view_overlaps(k_cache, q) || velri_tensor_view_overlaps(k_cache, k_in) ||
            velri_tensor_view_overlaps(k_cache, v_in) || velri_tensor_view_overlaps(v_cache, q) ||
            velri_tensor_view_overlaps(v_cache, k_in) ||
            velri_tensor_view_overlaps(v_cache, v_in)) {
            return VELRI_ERR_INVALID;
        }
        /* Allow exact (K==V==H) or prefix-aliased (K=[c_kv|k_pe], V=c_kv for nkvh==1) shared cache. */
        if (velri_tensor_view_overlaps(k_cache, v_cache)) {
            bool cache_prefix_ok = (k_cache->storage_id == v_cache->storage_id &&
                                    k_cache->byte_offset == v_cache->byte_offset &&
                                    k_cache->strides[0] == v_cache->strides[0] &&
                                    k_cache->strides[1] == v_cache->strides[1] && dv <= dk);
            bool in_prefix_ok = (k_in->storage_id == v_in->storage_id &&
                                 k_in->byte_offset == v_in->byte_offset &&
                                 k_in->strides[0] == v_in->strides[0] &&
                                 (dv == dk || nkvh == 1U));
            if (!cache_prefix_ok || !in_prefix_ok) {
                return VELRI_ERR_INVALID;
            }
        }
        return VELRI_OK;
    }

    case VELRI_OP_COPY_LAST_ROW: {
        const velri_tensor_view *x = &op->inputs[0];
        const velri_tensor_view *y = &op->outputs[0];
        if (!is_unit_inner_2d(x) || !is_unit_inner_2d(y) ||
            x->dtype != VELRI_DTYPE_F32 || y->dtype != VELRI_DTYPE_F32 ||
            y->dims[0] != 1U || x->dims[1] != y->dims[1] ||
            velri_tensor_view_overlaps(x, y)) {
            return VELRI_ERR_INVALID;
        }
        return VELRI_OK;
    }

    case VELRI_OP_BATCHED_LINEAR: {
        const velri_tensor_view *x = &op->inputs[0];
        const velri_tensor_view *w = &op->inputs[1];
        const velri_tensor_view *y = &op->outputs[0];

        uint32_t b = op->raw.attr.batched_linear.n_heads;
        uint32_t g = op->raw.attr.batched_linear.n_groups;
        uint32_t k = op->raw.attr.batched_linear.in_dim;
        uint32_t m = op->raw.attr.batched_linear.out_dim;
        bool wt = op->raw.attr.batched_linear.weight_transposed;
        if (b == 0U || g == 0U || k == 0U || m == 0U || (b % g) != 0U) {
            return VELRI_ERR_INVALID;
        }
        if (!is_unit_inner_2d(x) || !is_unit_inner_2d(w) || !is_unit_inner_2d(y)) {
            return VELRI_ERR_INVALID;
        }
        if (x->dtype != VELRI_DTYPE_F32 || w->dtype != VELRI_DTYPE_F32 ||
            y->dtype != VELRI_DTYPE_F32) {
            return VELRI_ERR_UNSUPPORTED;
        }
        size_t bk = 0U, bm = 0U, gm = 0U, gk = 0U;
        if (velri_mul_overflow_size((size_t)b, (size_t)k, &bk) ||
            velri_mul_overflow_size((size_t)b, (size_t)m, &bm) ||
            velri_mul_overflow_size((size_t)g, (size_t)m, &gm) ||
            velri_mul_overflow_size((size_t)g, (size_t)k, &gk) ||
            !fits_u32(bk) || !fits_u32(bm) || !fits_u32(gm) || !fits_u32(gk)) {
            return VELRI_ERR_OVERFLOW;
        }
        if (y->dims[0] != x->dims[0] || x->dims[1] != bk || y->dims[1] != bm) {
            return VELRI_ERR_INVALID;
        }
        size_t w_rows = wt ? gk : gm;
        size_t w_cols = wt ? (size_t)m : (size_t)k;
        if (w->dims[0] != w_rows || w->dims[1] != w_cols) {
            return VELRI_ERR_INVALID;
        }
        if (!check_opt_1d_param(op, 2U, VELRI_DTYPE_F32, gm, y, nullptr)) {
            return VELRI_ERR_INVALID;
        }
        if (velri_tensor_view_overlaps(y, x) || velri_tensor_view_overlaps(y, w)) {
            return VELRI_ERR_INVALID;
        }
        return VELRI_OK;
    }

    default:
        return VELRI_ERR_UNSUPPORTED;
    }
}

velri_status velri_op_validate_step_params(const velri_resolved_op *op,
                                           const velri_step_params *params) {
    if (!op) {
        return VELRI_ERR_INVALID;
    }
    if (!params) {
        return VELRI_OK;
    }
    if (params->has_pos_offset && !fits_u32(params->pos_offset_override)) {
        return VELRI_ERR_OVERFLOW;
    }
    if (params->has_seq_len) {
        if (params->seq_len_override == 0U) {
            return VELRI_ERR_INVALID;
        }
        if (!fits_u32(params->seq_len_override)) {
            return VELRI_ERR_OVERFLOW;
        }
    }
    if (params->has_kv_len && !fits_u32(params->kv_len_override)) {
        return VELRI_ERR_OVERFLOW;
    }

    uint32_t active_rows = 0U;
    velri_status st = velri_op_active_rows(op, params, &active_rows);
    if (st != VELRI_OK) {
        return st;
    }

    switch (op->kind) {
    case VELRI_OP_ROPE:
        return velri_op_resolve_rope_step(op, params, nullptr, nullptr);

    case VELRI_OP_EMBEDDING: {
        if (velri_op_embedding_has_ids(op)) {
            return VELRI_OK;
        }
        uint32_t pos = 0U;
        st = velri_op_resolve_pos_offset(op->raw.attr.embedding.pos_offset, params, &pos);
        if (st != VELRI_OK) {
            return st;
        }
        return check_embedding_pos_bounds(op, (size_t)active_rows, (size_t)pos);
    }

    case VELRI_OP_ATTENTION: {
        velri_attn_step_args attn;
        return velri_op_resolve_attention_step(op, params, &attn);
    }

    default:
        return VELRI_OK;
    }
}
