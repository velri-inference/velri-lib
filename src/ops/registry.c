#include "core_internal.h"
#include <math.h>

/*
 * Central operation registry validator.
 * Device-agnostic single source of truth for shapes, dtypes, strides, attributes,
 * 32-bit kernel index bounds, and memory aliasing rules across all backends.
 */

static bool views_overlap(const velri_tensor_view *a, const velri_tensor_view *b) {
    if (a->storage_id != b->storage_id) {
        return false;
    }
    size_t a0 = 0U, a1 = 0U, b0 = 0U, b1 = 0U;
    if (velri_tensor_view_byte_span(a, &a0, &a1) != VELRI_OK ||
        velri_tensor_view_byte_span(b, &b0, &b1) != VELRI_OK) {
        return true;
    }
    return (a0 < b1) && (b0 < a1);
}

static bool views_identical_layout(const velri_tensor_view *a, const velri_tensor_view *b) {
    if (a->storage_id != b->storage_id || a->dtype != b->dtype ||
        a->rank != b->rank || a->byte_offset != b->byte_offset) {
        return false;
    }
    for (uint32_t d = 0U; d < a->rank; ++d) {
        if (a->dims[d] != b->dims[d] || a->strides[d] != b->strides[d]) {
            return false;
        }
    }
    return true;
}

/* Reject shifted/partial overlap (which causes parallel races); allow exact in-place or disjoint. */
static bool no_partial_overlap(const velri_tensor_view *in_v, const velri_tensor_view *out_v) {
    return !views_overlap(in_v, out_v) || views_identical_layout(in_v, out_v);
}

static bool fits_u32(size_t v) { return v <= (size_t)UINT32_MAX; }

static bool is_unit_inner_2d(const velri_tensor_view *v) {
    return v->rank == 2U && v->strides[1] == 1U && v->strides[0] >= v->dims[1] &&
           fits_u32(v->dims[0]) && fits_u32(v->dims[1]) && fits_u32(v->strides[0]);
}

static bool is_unit_1d(const velri_tensor_view *v) {
    return v->rank == 1U && v->strides[0] == 1U && fits_u32(v->dims[0]);
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
    if (out0 && views_overlap(p, out0)) {
        return false;
    }
    if (out1 && views_overlap(p, out1)) {
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

static bool valid_linear_combo(const velri_linear_desc *d) {
    if (d->x_dtype == VELRI_DTYPE_I8 && d->w_dtype == VELRI_DTYPE_I8 &&
        d->y_dtype == VELRI_DTYPE_F32) {
        return d->has_x_scale && d->has_w_scale;
    }
    if (d->has_x_scale || d->has_w_scale) {
        return false;
    }
    if (d->x_dtype == VELRI_DTYPE_F32 && d->w_dtype == VELRI_DTYPE_F32 &&
        d->y_dtype == VELRI_DTYPE_F32) {
        return true;
    }
    if (d->x_dtype == VELRI_DTYPE_F16 && d->w_dtype == VELRI_DTYPE_F16 &&
        d->y_dtype == VELRI_DTYPE_F32) {
        return true;
    }
    if (d->x_dtype == VELRI_DTYPE_BF16 && d->w_dtype == VELRI_DTYPE_BF16 &&
        d->y_dtype == VELRI_DTYPE_F32) {
        return true;
    }
    if (d->x_dtype == VELRI_DTYPE_F32 && d->w_dtype == VELRI_DTYPE_BF16 &&
        d->y_dtype == VELRI_DTYPE_F32) {
        return true;
    }
    if (d->x_dtype == VELRI_DTYPE_F64 && d->w_dtype == VELRI_DTYPE_F64 &&
        d->y_dtype == VELRI_DTYPE_F64) {
        return true;
    }
    return false;
}

velri_status velri_op_validate(const velri_resolved_op *op) {
    if (!op || op->num_inputs > VELRI_OP_MAX_INPUTS || op->num_outputs == 0U ||
        op->num_outputs > VELRI_OP_MAX_OUTPUTS) {
        return VELRI_ERR_INVALID;
    }

    switch (op->kind) {
    case VELRI_OP_LINEAR: {
        if (op->num_inputs < 2U || op->num_inputs > 5U || op->num_outputs != 1U ||
            !op->input_present[0] || !op->input_present[1]) {
            return VELRI_ERR_INVALID;
        }
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
        if (!valid_linear_combo(d)) {
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
        }
        if (d->has_w_scale) {
            if (op->num_inputs <= 4U || !op->input_present[4] ||
                !check_opt_1d_param(op, 4U, VELRI_DTYPE_F32, w->dims[0], y, nullptr)) {
                return VELRI_ERR_INVALID;
            }
        }
        if (views_overlap(x, y) || views_overlap(w, y)) {
            return VELRI_ERR_INVALID;
        }
        return VELRI_OK;
    }

    case VELRI_OP_RMSNORM: {
        if (op->num_inputs < 1U || op->num_inputs > 2U || op->num_outputs != 1U ||
            !op->input_present[0]) {
            return VELRI_ERR_INVALID;
        }
        const velri_tensor_view *x = &op->inputs[0];
        const velri_tensor_view *y = &op->outputs[0];
        if (!check_same_2d(x, y) || !no_partial_overlap(x, y) ||
            !(op->raw.attr.rmsnorm.eps > 0.0f) || !isfinite(op->raw.attr.rmsnorm.eps)) {
            return VELRI_ERR_INVALID;
        }
        if (x->dtype != VELRI_DTYPE_F32 && x->dtype != VELRI_DTYPE_BF16 &&
            x->dtype != VELRI_DTYPE_F16) {
            return VELRI_ERR_UNSUPPORTED;
        }
        if (!check_opt_1d_param(op, 1U, x->dtype, x->dims[1], y, nullptr)) {
            return VELRI_ERR_INVALID;
        }
        return VELRI_OK;
    }

    case VELRI_OP_FUSED_ADD_RMSNORM: {
        if (op->num_inputs < 1U || op->num_inputs > 2U || op->num_outputs != 2U ||
            !op->input_present[0]) {
            return VELRI_ERR_INVALID;
        }
        const velri_tensor_view *x = &op->inputs[0];
        const velri_tensor_view *r = &op->outputs[0];
        const velri_tensor_view *y = &op->outputs[1];
        if (x->dtype != VELRI_DTYPE_F32 || !check_same_2d(x, r) || !check_same_2d(x, y) ||
            !(op->raw.attr.rmsnorm.eps > 0.0f) || !isfinite(op->raw.attr.rmsnorm.eps)) {
            return VELRI_ERR_INVALID;
        }
        if (views_overlap(r, y) || !no_partial_overlap(x, r) || !no_partial_overlap(x, y) ||
            !check_opt_1d_param(op, 1U, VELRI_DTYPE_F32, x->dims[1], r, y)) {
            return VELRI_ERR_INVALID;
        }
        return VELRI_OK;
    }

    case VELRI_OP_LAYERNORM: {
        if (op->num_inputs < 1U || op->num_inputs > 3U || op->num_outputs != 1U ||
            !op->input_present[0]) {
            return VELRI_ERR_INVALID;
        }
        const velri_tensor_view *x = &op->inputs[0];
        const velri_tensor_view *y = &op->outputs[0];
        if (x->dtype != VELRI_DTYPE_F32 || !check_same_2d(x, y) || !no_partial_overlap(x, y) ||
            !(op->raw.attr.layernorm.eps > 0.0f) || !isfinite(op->raw.attr.layernorm.eps)) {
            return VELRI_ERR_INVALID;
        }
        if (!check_opt_1d_param(op, 1U, VELRI_DTYPE_F32, x->dims[1], y, nullptr) ||
            !check_opt_1d_param(op, 2U, VELRI_DTYPE_F32, x->dims[1], y, nullptr)) {
            return VELRI_ERR_INVALID;
        }
        return VELRI_OK;
    }

    case VELRI_OP_ROPE: {
        if (op->num_inputs != 3U || op->num_outputs != 1U ||
            !op->input_present[0] || !op->input_present[1] || !op->input_present[2]) {
            return VELRI_ERR_INVALID;
        }
        const velri_tensor_view *x = &op->inputs[0];
        const velri_tensor_view *ct = &op->inputs[1];
        const velri_tensor_view *st = &op->inputs[2];
        const velri_tensor_view *y = &op->outputs[0];
        if (x->rank != 3U || y->rank != 3U || x->strides[2] != 1U || y->strides[2] != 1U ||
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
        if (!fits_u32(x->dims[0]) || !fits_u32(x->dims[1]) || !fits_u32(x->dims[2]) ||
            !fits_u32(x->strides[0]) || !fits_u32(x->strides[1]) ||
            !fits_u32(y->strides[0]) || !fits_u32(y->strides[1])) {
            return VELRI_ERR_OVERFLOW;
        }
        /* Non-overlapping output head/token strides prevent parallel write races. */
        size_t min_tok_sy = 0U;
        if (y->strides[1] < y->dims[2] ||
            velri_mul_overflow_size(y->dims[1], y->strides[1], &min_tok_sy) ||
            y->strides[0] < min_tok_sy || x->strides[1] < x->dims[2]) {
            return VELRI_ERR_INVALID;
        }
        size_t rd = op->raw.attr.rope.rotary_dim;
        if (rd == 0U || (rd & 1U) != 0U || rd > x->dims[2] || !fits_u32(rd)) {
            return VELRI_ERR_INVALID;
        }
        if (!velri_tensor_view_is_contiguous(ct) || !velri_tensor_view_is_contiguous(st) ||
            ct->dtype != VELRI_DTYPE_F32 || st->dtype != VELRI_DTYPE_F32) {
            return VELRI_ERR_INVALID;
        }
        size_t end_pos = 0U, need = 0U;
        if (velri_add_overflow_size(op->raw.attr.rope.pos_offset, x->dims[0], &end_pos) ||
            velri_mul_overflow_size(end_pos, rd / 2U, &need) || !fits_u32(op->raw.attr.rope.pos_offset)) {
            return VELRI_ERR_OVERFLOW;
        }
        size_t ct_n = 0U, st_n = 0U;
        if (velri_tensor_view_num_elements(ct, &ct_n) != VELRI_OK ||
            velri_tensor_view_num_elements(st, &st_n) != VELRI_OK ||
            ct_n < need || st_n < need) {
            return VELRI_ERR_INVALID;
        }
        if (!no_partial_overlap(x, y) || views_overlap(ct, y) || views_overlap(st, y)) {
            return VELRI_ERR_INVALID;
        }
        return VELRI_OK;
    }

    case VELRI_OP_ACTIVATION: {
        if (op->num_inputs != 1U || op->num_outputs != 1U || !op->input_present[0]) {
            return VELRI_ERR_INVALID;
        }
        velri_act_kind k = op->raw.attr.activation.kind;
        if (k != VELRI_ACT_SILU && k != VELRI_ACT_GELU_TANH && k != VELRI_ACT_RELU2) {
            return VELRI_ERR_UNSUPPORTED;
        }
        size_t nx = 0U, ny = 0U;
        velri_status st = check_elemwise_f32_view(&op->inputs[0], &nx);
        if (st != VELRI_OK) return st;
        st = check_elemwise_f32_view(&op->outputs[0], &ny);
        if (st != VELRI_OK) return st;
        if (nx != ny || !no_partial_overlap(&op->inputs[0], &op->outputs[0])) {
            return VELRI_ERR_INVALID;
        }
        return VELRI_OK;
    }

    case VELRI_OP_GATED_MLP:
    case VELRI_OP_VEC_MUL: {
        if (op->num_inputs != 2U || op->num_outputs != 1U ||
            !op->input_present[0] || !op->input_present[1]) {
            return VELRI_ERR_INVALID;
        }
        if (op->kind == VELRI_OP_GATED_MLP) {
            velri_act_kind a = op->raw.attr.gated_mlp.act;
            if (a != VELRI_ACT_SILU && a != VELRI_ACT_GELU_TANH) {
                return VELRI_ERR_UNSUPPORTED;
            }
        }
        size_t na = 0U, nb = 0U, ny = 0U;
        velri_status st = check_elemwise_f32_view(&op->inputs[0], &na);
        if (st != VELRI_OK) return st;
        st = check_elemwise_f32_view(&op->inputs[1], &nb);
        if (st != VELRI_OK) return st;
        st = check_elemwise_f32_view(&op->outputs[0], &ny);
        if (st != VELRI_OK) return st;
        if (na != nb || na != ny ||
            !no_partial_overlap(&op->inputs[0], &op->outputs[0]) ||
            !no_partial_overlap(&op->inputs[1], &op->outputs[0])) {
            return VELRI_ERR_INVALID;
        }
        return VELRI_OK;
    }

    case VELRI_OP_VEC_AXPBY: {
        if (op->num_inputs < 1U || op->num_inputs > 2U || op->num_outputs != 1U ||
            !op->input_present[0]) {
            return VELRI_ERR_INVALID;
        }
        size_t na = 0U, ny = 0U;
        velri_status st = check_elemwise_f32_view(&op->inputs[0], &na);
        if (st != VELRI_OK) return st;
        st = check_elemwise_f32_view(&op->outputs[0], &ny);
        if (st != VELRI_OK) return st;
        if (na != ny || !no_partial_overlap(&op->inputs[0], &op->outputs[0])) {
            return VELRI_ERR_INVALID;
        }
        if (op->num_inputs > 1U && op->input_present[1]) {
            size_t nb = 0U;
            st = check_elemwise_f32_view(&op->inputs[1], &nb);
            if (st != VELRI_OK) return st;
            if (nb != na || !no_partial_overlap(&op->inputs[1], &op->outputs[0])) {
                return VELRI_ERR_INVALID;
            }
        } else if (op->raw.attr.axpby.beta != 0.0f) {
            return VELRI_ERR_INVALID;
        }
        return VELRI_OK;
    }

    case VELRI_OP_SOFTMAX: {
        if (op->num_inputs != 1U || op->num_outputs != 1U || !op->input_present[0]) {
            return VELRI_ERR_INVALID;
        }
        const velri_tensor_view *x = &op->inputs[0];
        const velri_tensor_view *y = &op->outputs[0];
        if (x->dtype != VELRI_DTYPE_F32 || !check_same_2d(x, y) || !no_partial_overlap(x, y)) {
            return VELRI_ERR_INVALID;
        }
        return VELRI_OK;
    }

    case VELRI_OP_FWHT: {
        if (op->num_inputs < 1U || op->num_inputs > 2U || op->num_outputs != 1U ||
            !op->input_present[0]) {
            return VELRI_ERR_INVALID;
        }
        const velri_tensor_view *x = &op->inputs[0];
        const velri_tensor_view *y = &op->outputs[0];
        size_t bs = op->raw.attr.fwht.block_size;
        if (bs != 16U && bs != 32U && bs != 64U && bs != 128U && bs != 256U) {
            return VELRI_ERR_UNSUPPORTED;
        }
        if (x->dtype != VELRI_DTYPE_F32 || !check_same_2d(x, y) || !no_partial_overlap(x, y) ||
            (x->dims[1] % bs) != 0U) {
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
        if (op->num_inputs != 1U || op->num_outputs != 2U || !op->input_present[0]) {
            return VELRI_ERR_INVALID;
        }
        const velri_tensor_view *x = &op->inputs[0];
        const velri_tensor_view *q = &op->outputs[0];
        const velri_tensor_view *s = &op->outputs[1];
        if (!is_unit_inner_2d(x) || !is_unit_inner_2d(q) || !is_unit_1d(s) ||
            x->dtype != VELRI_DTYPE_F32 || q->dtype != VELRI_DTYPE_I8 ||
            s->dtype != VELRI_DTYPE_F32 ||
            x->dims[0] != q->dims[0] || x->dims[1] != q->dims[1] || s->dims[0] != x->dims[0]) {
            return VELRI_ERR_INVALID;
        }
        if (views_overlap(x, q) || views_overlap(x, s) || views_overlap(q, s)) {
            return VELRI_ERR_INVALID;
        }
        return VELRI_OK;
    }

    default:
        return VELRI_ERR_UNSUPPORTED;
    }
}
