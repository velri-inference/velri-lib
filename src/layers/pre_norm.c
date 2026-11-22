#include "blocks_internal.h"

#include <limits.h>

velri_tensor velri_layer_pre_norm(velri_builder *b, velri_tensor x, const char *layer_prefix,
                                  uint32_t layer_idx, const velri_layer_spec *spec) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!x.valid || x.rank != 2U || !spec || !spec->norm1_name || spec->norm1_name[0] == '\0' ||
        !spec->norm2_name || spec->norm2_name[0] == '\0') {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    if (!velri_blocks_validate_norm_spec(b, &spec->norm)) {
        return VELRI_TENSOR_NONE;
    }
    if (layer_idx > (uint32_t)INT32_MAX) {
        velri_b_set_error(b, VELRI_ERR_OVERFLOW);
        return VELRI_TENSOR_NONE;
    }

    char layer_pfx[VELRI_MANIFEST_NAME_MAX];
    if (!velri_blocks_format_layer_prefix(b, layer_pfx, layer_prefix, layer_idx)) {
        return VELRI_TENSOR_NONE;
    }

    bool pushed_scope = false;
    bool pushed_layer = false;
    bool pushed_sub = false;

    if (spec->scope) {
        velri_b_push_scope(b, spec->scope);
        if (b->err != VELRI_OK) {
            goto cleanup;
        }
        pushed_scope = true;
    }
    velri_b_push_layer(b, (int32_t)layer_idx);
    if (b->err != VELRI_OK) {
        goto cleanup;
    }
    pushed_layer = true;

    /* Attention sub-block: optional attn_scope around norm1 + attention + residual add. */
    if (spec->attn_scope) {
        velri_b_push_scope(b, spec->attn_scope);
        if (b->err != VELRI_OK) {
            goto cleanup;
        }
        pushed_sub = true;
    }
    velri_tensor h = velri_block_norm(b, x, layer_pfx, spec->norm1_name, &spec->norm);
    h = velri_block_attention(b, h, layer_pfx, &spec->attn);
    x = velri_b_add_inplace(b, x, h);
    if (b->err != VELRI_OK) {
        goto cleanup;
    }
    if (pushed_sub) {
        velri_b_pop_scope(b);
        pushed_sub = false;
    }

    /* MLP sub-block: optional mlp_scope around norm2 + mlp + residual add. */
    if (spec->mlp_scope) {
        velri_b_push_scope(b, spec->mlp_scope);
        if (b->err != VELRI_OK) {
            goto cleanup;
        }
        pushed_sub = true;
    }
    h = velri_block_norm(b, x, layer_pfx, spec->norm2_name, &spec->norm);
    h = velri_block_mlp(b, h, layer_pfx, &spec->mlp);
    x = velri_b_add_inplace(b, x, h);
    if (b->err != VELRI_OK) {
        goto cleanup;
    }
    if (pushed_sub) {
        velri_b_pop_scope(b);
        pushed_sub = false;
    }

cleanup:
    if (pushed_sub) {
        velri_b_pop_scope(b);
    }
    if (pushed_layer) {
        velri_b_pop_layer(b);
    }
    if (pushed_scope) {
        velri_b_pop_scope(b);
    }
    return (b->err == VELRI_OK) ? x : VELRI_TENSOR_NONE;
}
