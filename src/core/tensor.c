#include "core_internal.h"
#include <string.h>

size_t velri_dtype_bytes(velri_dtype dtype) {
    switch (dtype) {
    case VELRI_DTYPE_F32:
    case VELRI_DTYPE_I32:
        return 4U;
    case VELRI_DTYPE_F64:
        return 8U;
    case VELRI_DTYPE_F16:
    case VELRI_DTYPE_BF16:
        return 2U;
    case VELRI_DTYPE_I8:
        return 1U;
    default:
        return 0U;
    }
}

size_t velri_dtype_align(velri_dtype dtype) {
    switch (dtype) {
    case VELRI_DTYPE_F32:
        return alignof(float);
    case VELRI_DTYPE_F64:
        return alignof(double);
    case VELRI_DTYPE_F16:
    case VELRI_DTYPE_BF16:
        return alignof(uint16_t);
    case VELRI_DTYPE_I8:
        return alignof(int8_t);
    case VELRI_DTYPE_I32:
        return alignof(int32_t);
    default:
        return 1U;
    }
}

static velri_status fill_contiguous_strides(uint32_t rank, const size_t *dims,
                                            size_t out_dims[VELRI_MAX_RANK],
                                            size_t out_strides[VELRI_MAX_RANK],
                                            size_t *out_total_elems) {
    size_t stride = 1U;
    for (uint32_t i = rank; i > 0U; --i) {
        uint32_t d = i - 1U;
        if (dims[d] == 0U) {
            return VELRI_ERR_INVALID;
        }
        out_dims[d] = dims[d];
        out_strides[d] = stride;
        if (velri_mul_overflow_size(stride, dims[d], &stride)) {
            return VELRI_ERR_OVERFLOW;
        }
    }
    if (out_total_elems) {
        *out_total_elems = stride;
    }
    return VELRI_OK;
}

velri_status velri_shape_byte_size(velri_dtype dtype, uint32_t rank, const size_t *dims,
                                   size_t *out_bytes) {
    if (!dims || !out_bytes || rank == 0U || rank > VELRI_MAX_RANK) {
        return VELRI_ERR_INVALID;
    }
    size_t eb = velri_dtype_bytes(dtype);
    if (eb == 0U) {
        return VELRI_ERR_INVALID;
    }
    size_t total_elems = 1U;
    for (uint32_t d = 0U; d < rank; ++d) {
        if (dims[d] == 0U) {
            return VELRI_ERR_INVALID;
        }
        if (velri_mul_overflow_size(total_elems, dims[d], &total_elems)) {
            return VELRI_ERR_OVERFLOW;
        }
    }
    size_t bytes = 0U;
    if (velri_mul_overflow_size(total_elems, eb, &bytes)) {
        return VELRI_ERR_OVERFLOW;
    }
    *out_bytes = bytes;
    return VELRI_OK;
}

velri_status velri_tensor_view_contiguous(velri_storage_id storage_id, velri_dtype dtype,
                                          uint32_t rank, const size_t *dims, size_t byte_offset,
                                          velri_tensor_view *out_view) {
    if (!dims || !out_view || rank == 0U || rank > VELRI_MAX_RANK) {
        return VELRI_ERR_INVALID;
    }
    size_t eb = velri_dtype_bytes(dtype);
    if (eb == 0U) {
        return VELRI_ERR_INVALID;
    }
    size_t ea = velri_dtype_align(dtype);
    if ((byte_offset % ea) != 0U) {
        return VELRI_ERR_INVALID;
    }

    velri_tensor_view v;
    memset(&v, 0, sizeof(v));
    v.storage_id = storage_id;
    v.dtype = dtype;
    v.rank = rank;
    v.byte_offset = byte_offset;

    size_t total_elems = 0U;
    velri_status st = fill_contiguous_strides(rank, dims, v.dims, v.strides, &total_elems);
    if (st != VELRI_OK) {
        return st;
    }
    size_t total_bytes = 0U;
    size_t end = 0U;
    if (velri_mul_overflow_size(total_elems, eb, &total_bytes) ||
        velri_add_overflow_size(byte_offset, total_bytes, &end)) {
        return VELRI_ERR_OVERFLOW;
    }
    *out_view = v;
    return VELRI_OK;
}

velri_status velri_tensor_view_subview(const velri_tensor_view *base, uint32_t rank,
                                       const size_t *dims, const size_t *strides,
                                       size_t extra_byte_offset, velri_tensor_view *out_view) {
    if (!base || !dims || !out_view || rank == 0U || rank > VELRI_MAX_RANK) {
        return VELRI_ERR_INVALID;
    }
    size_t eb = velri_dtype_bytes(base->dtype);
    size_t ea = velri_dtype_align(base->dtype);
    if (eb == 0U) {
        return VELRI_ERR_INVALID;
    }
    size_t base_s0 = 0U, base_s1 = 0U;
    velri_status st = velri_tensor_view_byte_span(base, &base_s0, &base_s1);
    if (st != VELRI_OK) {
        return st;
    }
    size_t new_offset = 0U;
    if (velri_add_overflow_size(base->byte_offset, extra_byte_offset, &new_offset)) {
        return VELRI_ERR_OVERFLOW;
    }
    if ((new_offset % ea) != 0U) {
        return VELRI_ERR_INVALID;
    }

    velri_tensor_view v;
    memset(&v, 0, sizeof(v));
    v.storage_id = base->storage_id;
    v.dtype = base->dtype;
    v.rank = rank;
    v.byte_offset = new_offset;

    if (strides) {
        for (uint32_t d = 0U; d < rank; ++d) {
            if (dims[d] == 0U) {
                return VELRI_ERR_INVALID;
            }
            v.dims[d] = dims[d];
            v.strides[d] = strides[d];
        }
    } else {
        st = fill_contiguous_strides(rank, dims, v.dims, v.strides, nullptr);
        if (st != VELRI_OK) {
            return st;
        }
    }

    size_t s0 = 0U, s1 = 0U;
    st = velri_tensor_view_byte_span(&v, &s0, &s1);
    if (st != VELRI_OK) {
        return st;
    }
    if (s0 < base_s0 || s1 > base_s1) {
        return VELRI_ERR_INVALID;
    }
    *out_view = v;
    return VELRI_OK;
}

velri_status velri_tensor_view_num_elements(const velri_tensor_view *view, size_t *out_elems) {
    if (!view || !out_elems || view->rank == 0U || view->rank > VELRI_MAX_RANK) {
        return VELRI_ERR_INVALID;
    }
    size_t n = 1U;
    for (uint32_t d = 0U; d < view->rank; ++d) {
        if (view->dims[d] == 0U) {
            return VELRI_ERR_INVALID;
        }
        if (velri_mul_overflow_size(n, view->dims[d], &n)) {
            return VELRI_ERR_OVERFLOW;
        }
    }
    *out_elems = n;
    return VELRI_OK;
}

velri_status velri_tensor_view_byte_span(const velri_tensor_view *view, size_t *out_start,
                                         size_t *out_end) {
    if (!view || !out_start || !out_end || view->rank == 0U || view->rank > VELRI_MAX_RANK) {
        return VELRI_ERR_INVALID;
    }
    size_t eb = velri_dtype_bytes(view->dtype);
    if (eb == 0U) {
        return VELRI_ERR_INVALID;
    }

    size_t max_elem = 0U;
    for (uint32_t d = 0U; d < view->rank; ++d) {
        if (view->dims[d] == 0U) {
            return VELRI_ERR_INVALID;
        }
        size_t term = 0U;
        if (velri_mul_overflow_size(view->dims[d] - 1U, view->strides[d], &term) ||
            velri_add_overflow_size(max_elem, term, &max_elem)) {
            return VELRI_ERR_OVERFLOW;
        }
    }
    size_t span_elems = 0U, span_bytes = 0U, end = 0U;
    if (velri_add_overflow_size(max_elem, 1U, &span_elems) ||
        velri_mul_overflow_size(span_elems, eb, &span_bytes) ||
        velri_add_overflow_size(view->byte_offset, span_bytes, &end)) {
        return VELRI_ERR_OVERFLOW;
    }
    *out_start = view->byte_offset;
    *out_end = end;
    return VELRI_OK;
}

velri_status velri_tensor_view_validate(const velri_tensor_view *view,
                                        const velri_storage_desc *storage) {
    if (!view || !storage) {
        return VELRI_ERR_INVALID;
    }
    size_t eb = velri_dtype_bytes(view->dtype);
    size_t ea = velri_dtype_align(view->dtype);
    if (eb == 0U || (view->byte_offset % ea) != 0U) {
        return VELRI_ERR_INVALID;
    }
    size_t start = 0U, end = 0U;
    velri_status st = velri_tensor_view_byte_span(view, &start, &end);
    if (st != VELRI_OK) {
        return st;
    }
    return (end <= storage->byte_size) ? VELRI_OK : VELRI_ERR_INVALID;
}

bool velri_tensor_view_is_contiguous(const velri_tensor_view *view) {
    if (!view || view->rank == 0U || view->rank > VELRI_MAX_RANK) {
        return false;
    }
    size_t expected = 1U;
    for (uint32_t i = view->rank; i > 0U; --i) {
        uint32_t d = i - 1U;
        if (view->dims[d] == 0U) {
            return false;
        }
        if (view->dims[d] > 1U && view->strides[d] != expected) {
            return false;
        }
        if (velri_mul_overflow_size(expected, view->dims[d], &expected)) {
            return false;
        }
    }
    return true;
}

bool velri_tensor_view_overlaps(const velri_tensor_view *a, const velri_tensor_view *b) {
    if (!a || !b || a->storage_id != b->storage_id) {
        return false;
    }
    size_t a0 = 0U, a1 = 0U, b0 = 0U, b1 = 0U;
    if (velri_tensor_view_byte_span(a, &a0, &a1) != VELRI_OK ||
        velri_tensor_view_byte_span(b, &b0, &b1) != VELRI_OK) {
        return true;
    }
    return (a0 < b1) && (b0 < a1);
}

bool velri_tensor_view_identical_layout(const velri_tensor_view *a, const velri_tensor_view *b) {
    if (!a || !b || a->storage_id != b->storage_id || a->dtype != b->dtype ||
        a->rank != b->rank || a->rank == 0U || a->rank > VELRI_MAX_RANK ||
        a->byte_offset != b->byte_offset) {
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
bool velri_tensor_view_no_partial_overlap(const velri_tensor_view *in,
                                          const velri_tensor_view *out) {
    return !velri_tensor_view_overlaps(in, out) || velri_tensor_view_identical_layout(in, out);
}

bool velri_tensor_view_covers_storage(const velri_tensor_view *v, size_t storage_bytes) {
    if (!v) {
        return false;
    }
    size_t s0 = 0U, s1 = 0U;
    if (velri_tensor_view_byte_span(v, &s0, &s1) != VELRI_OK) {
        return false;
    }
    return velri_tensor_view_is_contiguous(v) && s0 == 0U && s1 == storage_bytes;
}
