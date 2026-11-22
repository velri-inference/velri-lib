#include "velri_lib/blocks.h"

velri_tensor velri_block_embedding(velri_builder *b, const velri_embedding_spec *spec) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!spec || !spec->token_weight || spec->token_weight[0] == '\0' || spec->vocab_size == 0U ||
        spec->dim == 0U || spec->max_seq_len == 0U || velri_dtype_bytes(spec->dtype) == 0U) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    if (spec->pos_weight != nullptr &&
        (spec->pos_weight[0] == '\0' || spec->max_pos == 0U || spec->max_seq_len > spec->max_pos)) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }

    bool pushed_scope = false;
    velri_tensor token_ids = VELRI_TENSOR_NONE;
    velri_tensor x = VELRI_TENSOR_NONE;

    if (spec->scope) {
        velri_b_push_scope(b, spec->scope);
        if (b->err != VELRI_OK) {
            goto cleanup;
        }
        pushed_scope = true;
    }

    const size_t dim_ids[1] = {spec->max_seq_len};
    const size_t dim_tok[2] = {spec->vocab_size, spec->dim};

    token_ids = velri_b_input(b, VELRI_DTYPE_I32, 1U, dim_ids);
    velri_tensor token_w = velri_b_weight(b, spec->token_weight, spec->dtype, 2U, dim_tok);
    x = velri_b_embedding(b, token_w, token_ids, 0U, spec->max_seq_len);
    if (b->err != VELRI_OK) {
        goto cleanup;
    }

    if (spec->pos_weight != nullptr) {
        const size_t dim_pos[2] = {spec->max_pos, spec->dim};
        velri_tensor pos_w = velri_b_weight(b, spec->pos_weight, spec->dtype, 2U, dim_pos);
        velri_tensor pos = velri_b_embedding(b, pos_w, VELRI_TENSOR_NONE, 0U, spec->max_seq_len);
        x = velri_b_add_inplace(b, x, pos);
    }

cleanup:
    if (pushed_scope) {
        velri_b_pop_scope(b);
    }
    if (b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (b->manifest) {
        b->manifest->token_ids_in = token_ids.storage_id;
    }
    return x;
}
