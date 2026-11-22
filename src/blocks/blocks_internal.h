#ifndef VELRI_BLOCKS_INTERNAL_H
#define VELRI_BLOCKS_INTERNAL_H

#include "util_internal.h"
#include "velri_lib/blocks.h"

#include <math.h>
#include <stdio.h>

/*
 * Private helpers shared across `velri-lib/src/blocks/` and `velri-lib/src/layers/`.
 * Eliminates duplicated naming, weight registration, dtype promotion, and norm validation.
 */

static inline bool velri_blocks_format_name(velri_builder *b, char out[VELRI_MANIFEST_NAME_MAX],
                                            const char *prefix, const char *sub,
                                            const char *suffix) {
    if (!sub || sub[0] == '\0') {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return false;
    }
    const char *suf = suffix ? suffix : "";
    int n = 0;
    if (prefix && prefix[0] != '\0') {
        n = snprintf(out, VELRI_MANIFEST_NAME_MAX, "%s.%s%s", prefix, sub, suf);
    } else {
        n = snprintf(out, VELRI_MANIFEST_NAME_MAX, "%s%s", sub, suf);
    }
    if (n < 0 || (size_t)n >= VELRI_MANIFEST_NAME_MAX) {
        velri_b_set_error(b, VELRI_ERR_OVERFLOW);
        return false;
    }
    return true;
}

static inline bool velri_blocks_format_layer_prefix(velri_builder *b,
                                                    char out[VELRI_MANIFEST_NAME_MAX],
                                                    const char *prefix, uint32_t layer_idx) {
    int n = 0;
    if (prefix && prefix[0] != '\0') {
        n = snprintf(out, VELRI_MANIFEST_NAME_MAX, "%s.%u", prefix, layer_idx);
    } else {
        n = snprintf(out, VELRI_MANIFEST_NAME_MAX, "%u", layer_idx);
    }
    if (n < 0 || (size_t)n >= VELRI_MANIFEST_NAME_MAX) {
        velri_b_set_error(b, VELRI_ERR_OVERFLOW);
        return false;
    }
    return true;
}

/* Central activation/output dtype promotion contract for Linear projections. */
static inline velri_dtype velri_blocks_linear_out_dtype(velri_dtype x_dtype, velri_dtype w_dtype) {
    return (x_dtype == VELRI_DTYPE_F64 && w_dtype == VELRI_DTYPE_F64) ? VELRI_DTYPE_F64
                                                                      : VELRI_DTYPE_F32;
}

static inline bool velri_blocks_validate_norm_spec(velri_builder *b, const velri_norm_spec *norm) {
    if (!norm || !(norm->eps > 0.0f) || !isfinite(norm->eps)) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return false;
    }
    if (norm->kind != VELRI_NORM_LAYER && norm->kind != VELRI_NORM_RMS) {
        velri_b_set_error(b, VELRI_ERR_UNSUPPORTED);
        return false;
    }
    if (norm->kind == VELRI_NORM_RMS && norm->has_bias) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return false;
    }
    return true;
}

/* Null-safe weight + optional bias registration for linear projections (`<prefix>.<proj>.weight/bias`). */
static inline bool velri_blocks_make_proj_weights(velri_builder *b, const char *prefix,
                                                  const char *proj_name, velri_dtype w_dtype,
                                                  velri_dtype b_dtype, size_t out_dim,
                                                  size_t in_dim, bool has_bias,
                                                  velri_tensor *out_w, velri_tensor *out_b) {
    *out_w = VELRI_TENSOR_NONE;
    if (out_b) {
        *out_b = VELRI_TENSOR_NONE;
    }

    char w_name[VELRI_MANIFEST_NAME_MAX];
    if (!velri_blocks_format_name(b, w_name, prefix, proj_name, ".weight")) {
        return false;
    }
    const size_t dim_w[2] = {out_dim, in_dim};
    *out_w = velri_b_weight(b, w_name, w_dtype, 2U, dim_w);
    if (!out_w->valid) {
        return false;
    }

    if (has_bias) {
        if (!out_b) {
            velri_b_set_error(b, VELRI_ERR_INVALID);
            return false;
        }
        char b_name[VELRI_MANIFEST_NAME_MAX];
        if (!velri_blocks_format_name(b, b_name, prefix, proj_name, ".bias")) {
            return false;
        }
        const size_t dim_b[1] = {out_dim};
        *out_b = velri_b_weight(b, b_name, b_dtype, 1U, dim_b);
        if (!out_b->valid) {
            return false;
        }
    }
    return true;
}

#endif /* VELRI_BLOCKS_INTERNAL_H */
