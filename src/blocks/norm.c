#include "blocks_internal.h"

velri_tensor velri_block_norm(velri_builder *b, velri_tensor x, const char *prefix,
                              const char *norm_name, const velri_norm_spec *spec) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!x.valid || x.rank != 2U || !norm_name || norm_name[0] == '\0') {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    if (!velri_blocks_validate_norm_spec(b, spec)) {
        return VELRI_TENSOR_NONE;
    }

    char w_name[VELRI_MANIFEST_NAME_MAX];
    if (!velri_blocks_format_name(b, w_name, prefix, norm_name, ".weight")) {
        return VELRI_TENSOR_NONE;
    }
    const size_t dim_norm[1] = {x.dims[1]};
    velri_tensor norm_w = velri_b_weight(b, w_name, x.dtype, 1U, dim_norm);
    if (!norm_w.valid) {
        return VELRI_TENSOR_NONE;
    }

    if (spec->kind == VELRI_NORM_LAYER) {
        velri_tensor norm_b = VELRI_TENSOR_NONE;
        if (spec->has_bias) {
            char b_name[VELRI_MANIFEST_NAME_MAX];
            if (!velri_blocks_format_name(b, b_name, prefix, norm_name, ".bias")) {
                return VELRI_TENSOR_NONE;
            }
            norm_b = velri_b_weight(b, b_name, x.dtype, 1U, dim_norm);
            if (!norm_b.valid) {
                return VELRI_TENSOR_NONE;
            }
        }
        return velri_b_layernorm(b, x, norm_w, norm_b, spec->eps);
    }

    return velri_b_rmsnorm(b, x, norm_w, spec->eps, spec->rms_unit_offset);
}

velri_tensor velri_block_linear(velri_builder *b, velri_tensor x, const char *prefix,
                                const char *proj_name, velri_dtype w_dtype, size_t out_dim,
                                bool has_bias) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!x.valid || x.rank != 2U || out_dim == 0U || velri_dtype_bytes(w_dtype) == 0U ||
        !proj_name || proj_name[0] == '\0') {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }

    velri_dtype act_dtype = velri_blocks_linear_out_dtype(x.dtype, w_dtype);
    velri_tensor w = VELRI_TENSOR_NONE;
    velri_tensor bias = VELRI_TENSOR_NONE;
    if (!velri_blocks_make_proj_weights(b, prefix, proj_name, w_dtype, act_dtype, out_dim,
                                        x.dims[1], has_bias, &w, &bias)) {
        return VELRI_TENSOR_NONE;
    }
    return velri_b_linear(b, x, w, bias);
}
