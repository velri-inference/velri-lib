#include "blocks_internal.h"

#include <limits.h>
#include <string.h>

static bool format_lm_head_bias_name(velri_builder *b, char out[VELRI_MANIFEST_NAME_MAX],
                                     const char *weight_name) {
    static const char k_suffix[] = ".weight";
    const size_t suf_len = sizeof(k_suffix) - 1U;
    size_t wlen = strlen(weight_name);
    int n = 0;
    if (wlen > suf_len && strcmp(weight_name + (wlen - suf_len), k_suffix) == 0) {
        size_t base_len = wlen - suf_len;
        if (base_len > (size_t)INT_MAX) {
            velri_b_set_error(b, VELRI_ERR_OVERFLOW);
            return false;
        }
        n = snprintf(out, VELRI_MANIFEST_NAME_MAX, "%.*s.bias", (int)base_len, weight_name);
    } else {
        n = snprintf(out, VELRI_MANIFEST_NAME_MAX, "%s.bias", weight_name);
    }
    if (n < 0 || (size_t)n >= VELRI_MANIFEST_NAME_MAX) {
        velri_b_set_error(b, VELRI_ERR_OVERFLOW);
        return false;
    }
    return true;
}

velri_tensor velri_block_lm_head(velri_builder *b, velri_tensor x, const velri_lm_head_spec *spec) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!x.valid || x.rank != 2U || !spec || spec->vocab_size == 0U || spec->dim == 0U ||
        x.dims[1] != (size_t)spec->dim || velri_dtype_bytes(spec->dtype) == 0U) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    if (spec->tie_to != nullptr) {
        if (spec->tie_to[0] == '\0' ||
            (spec->weight_name != nullptr && spec->weight_name[0] == '\0')) {
            velri_b_set_error(b, VELRI_ERR_INVALID);
            return VELRI_TENSOR_NONE;
        }
    } else if (!spec->weight_name || spec->weight_name[0] == '\0') {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    if (spec->has_bias && (!spec->weight_name || spec->weight_name[0] == '\0')) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    if (spec->norm_name != nullptr) {
        if (spec->norm_name[0] == '\0') {
            velri_b_set_error(b, VELRI_ERR_INVALID);
            return VELRI_TENSOR_NONE;
        }
        if (!velri_blocks_validate_norm_spec(b, &spec->norm)) {
            return VELRI_TENSOR_NONE;
        }
    }

    /* Slice the final active sequence row on `x.device_id` before entering `lm-head` scope
     * unless full-sequence logits were explicitly requested. */
    bool single_row = !x.is_seq_dynamic && x.dims[0] == 1U;
    velri_tensor h = (spec->full_seq_logits || single_row) ? x : velri_b_copy_last_row(b, x);
    if (b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }

    bool pushed_scope = false;
    velri_tensor logits = VELRI_TENSOR_NONE;
    velri_tensor out = VELRI_TENSOR_NONE;

    if (spec->scope) {
        velri_b_push_scope(b, spec->scope);
        if (b->err != VELRI_OK) {
            goto cleanup;
        }
        pushed_scope = true;
    }

    if (spec->norm_name != nullptr) {
        h = velri_block_norm(b, h, nullptr, spec->norm_name, &spec->norm);
        if (!h.valid) {
            goto cleanup;
        }
    }

    const size_t dim_w[2] = {spec->vocab_size, spec->dim};
    velri_tensor lm_head_w =
        velri_b_weight_tied(b, spec->weight_name, spec->tie_to, spec->dtype, 2U, dim_w);
    if (!lm_head_w.valid) {
        goto cleanup;
    }

    velri_dtype out_dtype = velri_blocks_linear_out_dtype(h.dtype, spec->dtype);
    velri_tensor lm_head_b = VELRI_TENSOR_NONE;
    if (spec->has_bias) {
        char b_name[VELRI_MANIFEST_NAME_MAX];
        if (!format_lm_head_bias_name(b, b_name, spec->weight_name)) {
            goto cleanup;
        }
        const size_t dim_b[1] = {spec->vocab_size};
        lm_head_b = velri_b_weight(b, b_name, out_dtype, 1U, dim_b);
        if (!lm_head_b.valid) {
            goto cleanup;
        }
    }

    const size_t dim_logits[2] = {h.dims[0], spec->vocab_size};
    logits = velri_b_output(b, out_dtype, 2U, dim_logits);
    if (!logits.valid) {
        goto cleanup;
    }
    logits.is_seq_dynamic = h.is_seq_dynamic;
    velri_projection_input input = velri_blocks_projection_input(b, h, spec->dtype);
    velri_tensor scale = velri_blocks_weight_scale(b, nullptr, spec->weight_name, "_scale", lm_head_w);
    out = velri_blocks_linear_into(b, input, lm_head_w, lm_head_b, scale, logits);

cleanup:
    if (pushed_scope) {
        velri_b_pop_scope(b);
    }
    if (b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (b->manifest) {
        b->manifest->logits_out = logits.storage_id;
    }
    return out;
}
