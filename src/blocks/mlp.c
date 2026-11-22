#include "blocks_internal.h"

velri_tensor velri_block_mlp(velri_builder *b, velri_tensor x, const char *prefix,
                             const velri_mlp_spec *spec) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!x.valid || x.rank != 2U || !spec || spec->intermediate_size == 0U ||
        velri_dtype_bytes(spec->dtype) == 0U || !spec->up_name || spec->up_name[0] == '\0' ||
        !spec->down_name || spec->down_name[0] == '\0') {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    if (spec->is_gated && (!spec->gate_name || spec->gate_name[0] == '\0' ||
                           !isfinite(spec->clamp_limit) || spec->clamp_limit < 0.0f)) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }

    bool pushed_scope = false;
    velri_tensor out = VELRI_TENSOR_NONE;

    if (spec->scope) {
        velri_b_push_scope(b, spec->scope);
        if (b->err != VELRI_OK) {
            goto cleanup;
        }
        pushed_scope = true;
    }

    size_t d_model = x.dims[1];
    size_t d_inter = (size_t)spec->intermediate_size;

    velri_tensor h = VELRI_TENSOR_NONE;
    if (!spec->is_gated) {
        h = velri_block_linear(b, x, prefix, spec->up_name, spec->dtype, d_inter, spec->has_bias);
        h = velri_b_activation_inplace(b, h, spec->act);
    } else {
        velri_tensor g =
            velri_block_linear(b, x, prefix, spec->gate_name, spec->dtype, d_inter, spec->has_bias);
        velri_tensor u =
            velri_block_linear(b, x, prefix, spec->up_name, spec->dtype, d_inter, spec->has_bias);
        h = velri_b_gated_mlp_act_inplace(b, g, u, spec->act, spec->clamp_limit);
    }

    out = velri_block_linear(b, h, prefix, spec->down_name, spec->dtype, d_model, spec->has_bias);

cleanup:
    if (pushed_scope) {
        velri_b_pop_scope(b);
    }
    return (b->err == VELRI_OK) ? out : VELRI_TENSOR_NONE;
}
