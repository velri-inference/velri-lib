#include "blocks_internal.h"

static bool validate_attention_dims_scale(velri_builder *b, uint32_t n_heads, uint32_t n_kv_heads,
                                          uint32_t head_dim_k, uint32_t head_dim_v, float scale) {
    if (n_heads == 0U || n_kv_heads == 0U || head_dim_k == 0U ||
        head_dim_k > VELRI_ATTN_MAX_HEAD_DIM || head_dim_v > VELRI_ATTN_MAX_HEAD_DIM ||
        (n_heads % n_kv_heads) != 0U) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return false;
    }
    if (scale != 0.0f && (!(scale > 0.0f) || !isfinite(scale))) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return false;
    }
    return true;
}

static bool validate_attention_common(velri_builder *b, velri_tensor x,
                                      const velri_attention_spec *spec, size_t *out_q_dim,
                                      size_t *out_kv_dim, size_t *out_qkv_dim) {
    if (!x.valid || x.rank != 2U || !spec || spec->max_ctx == 0U ||
        x.dims[0] > (size_t)spec->max_ctx || velri_dtype_bytes(spec->dtype) == 0U ||
        !spec->o_name || spec->o_name[0] == '\0') {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return false;
    }
    if (!validate_attention_dims_scale(b, spec->n_heads, spec->n_kv_heads, spec->head_dim,
                                       spec->head_dim, spec->scale)) {
        return false;
    }
    if (spec->fused_qkv) {
        if (!spec->qkv_name || spec->qkv_name[0] == '\0') {
            velri_b_set_error(b, VELRI_ERR_INVALID);
            return false;
        }
    } else {
        if (!spec->q_name || spec->q_name[0] == '\0' || !spec->k_name || spec->k_name[0] == '\0' ||
            !spec->v_name || spec->v_name[0] == '\0') {
            velri_b_set_error(b, VELRI_ERR_INVALID);
            return false;
        }
    }

    size_t q_dim = 0U;
    size_t kv_dim = 0U;
    if (velri_mul_overflow_size((size_t)spec->n_heads, (size_t)spec->head_dim, &q_dim) ||
        velri_mul_overflow_size((size_t)spec->n_kv_heads, (size_t)spec->head_dim, &kv_dim) ||
        q_dim > (size_t)UINT32_MAX || kv_dim > (size_t)UINT32_MAX) {
        velri_b_set_error(b, VELRI_ERR_OVERFLOW);
        return false;
    }
    size_t qkv_dim = 0U;
    if (spec->fused_qkv) {
        size_t two_kv = 0U;
        if (velri_mul_overflow_size(2U, kv_dim, &two_kv) ||
            velri_add_overflow_size(q_dim, two_kv, &qkv_dim) || qkv_dim > (size_t)UINT32_MAX) {
            velri_b_set_error(b, VELRI_ERR_OVERFLOW);
            return false;
        }
    }

    *out_q_dim = q_dim;
    *out_kv_dim = kv_dim;
    *out_qkv_dim = qkv_dim;
    return true;
}

static velri_tensor alloc_cache_state(velri_builder *b, const char *prefix,
                                      const char *state_prefix, const char *suffix,
                                      const char *kv_cache_scope, velri_dtype dtype,
                                      const size_t dims[3]) {
    const char *spfx = (state_prefix && state_prefix[0] != '\0') ? state_prefix : "attn";
    char name[VELRI_MANIFEST_NAME_MAX];
    if (!velri_blocks_format_name(b, name, prefix, spfx, suffix)) {
        return VELRI_TENSOR_NONE;
    }

    bool pushed_kv = false;
    velri_tensor state = VELRI_TENSOR_NONE;
    if (kv_cache_scope) {
        velri_b_push_scope(b, kv_cache_scope);
        if (b->err != VELRI_OK) {
            goto cleanup;
        }
        pushed_kv = true;
    }
    state = velri_b_state(b, name, dtype, 3U, dims);

cleanup:
    if (pushed_kv) {
        velri_b_pop_scope(b);
    }
    return (b->err == VELRI_OK) ? state : VELRI_TENSOR_NONE;
}

velri_tensor velri_block_attention_core(velri_builder *b, velri_tensor q, velri_tensor k_in,
                                        velri_tensor v_in, velri_tensor k_cache,
                                        velri_tensor v_cache,
                                        const velri_attention_core_spec *spec) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!spec) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    if (!validate_attention_dims_scale(b, spec->n_heads, spec->n_kv_heads, spec->head_dim_k,
                                       spec->head_dim_v, spec->scale)) {
        return VELRI_TENSOR_NONE;
    }

    uint32_t dv = spec->head_dim_v ? spec->head_dim_v : spec->head_dim_k;
    if (spec->inplace_q && dv == spec->head_dim_k) {
        return velri_b_attention_inplace(b, q, k_in, v_in, k_cache, v_cache, spec->n_heads,
                                         spec->n_kv_heads, spec->head_dim_k, spec->scale,
                                         spec->sliding_window, spec->causal);
    }
    return velri_b_attention_ex(b, q, k_in, v_in, k_cache, v_cache, spec->n_heads, spec->n_kv_heads,
                                spec->head_dim_k, dv, spec->scale, spec->sliding_window,
                                spec->causal);
}

velri_tensor velri_block_latent_attention(velri_builder *b, velri_tensor x, velri_tensor h_lat,
                                          const char *prefix, const velri_attention_spec *spec) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }

    size_t q_dim = 0U, kv_dim = 0U, qkv_dim = 0U;
    if (!validate_attention_common(b, x, spec, &q_dim, &kv_dim, &qkv_dim)) {
        return VELRI_TENSOR_NONE;
    }
    if (spec->rope.enabled) {
        velri_b_set_error(b, VELRI_ERR_UNSUPPORTED);
        return VELRI_TENSOR_NONE;
    }
    if (!h_lat.valid || h_lat.rank != 2U || h_lat.dims[0] != x.dims[0] || h_lat.dims[1] == 0U) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    if (x.dtype != VELRI_DTYPE_F32 || h_lat.dtype != VELRI_DTYPE_F32 ||
        spec->dtype != VELRI_DTYPE_F32) {
        velri_b_set_error(b, VELRI_ERR_UNSUPPORTED);
        return VELRI_TENSOR_NONE;
    }
    if (h_lat.dims[1] > VELRI_ATTN_MAX_HEAD_DIM) {
        velri_b_set_error(b, VELRI_ERR_UNSUPPORTED);
        return VELRI_TENSOR_NONE;
    }
    /* Fused [Q|K|V] projection expects a single shared input tensor (`h_lat == x`). */
    if (spec->fused_qkv && (h_lat.storage_id != x.storage_id || h_lat.view_id != x.view_id)) {
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
    uint32_t d_lat = (uint32_t)h_lat.dims[1];

    velri_tensor q = VELRI_TENSOR_NONE;
    velri_tensor w_k = VELRI_TENSOR_NONE;
    velri_tensor w_v = VELRI_TENSOR_NONE;
    velri_tensor b_v = VELRI_TENSOR_NONE;

    if (spec->fused_qkv) {
        /* Standard fused [Q | K | V] row-concatenated projection: slice W_Q, W_K, W_V zero-copy.
         * Key bias b_K is omitted by exact softmax shift invariance (q . b_K is key-independent). */
        velri_tensor qkv_w = VELRI_TENSOR_NONE;
        velri_tensor qkv_b = VELRI_TENSOR_NONE;
        if (!velri_blocks_make_proj_weights(b, prefix, spec->qkv_name, VELRI_DTYPE_F32,
                                            VELRI_DTYPE_F32, qkv_dim, d_model, spec->qkv_bias,
                                            &qkv_w, &qkv_b)) {
            goto cleanup;
        }
        velri_tensor w_q = velri_b_subview_2d_rows(b, qkv_w, 0U, q_dim);
        w_k = velri_b_subview_2d_rows(b, qkv_w, q_dim, kv_dim);
        w_v = velri_b_subview_2d_rows(b, qkv_w, q_dim + kv_dim, kv_dim);
        velri_tensor b_q = VELRI_TENSOR_NONE;
        if (spec->qkv_bias) {
            b_q = velri_b_subview_1d(b, qkv_b, 0U, q_dim);
            b_v = velri_b_subview_1d(b, qkv_b, q_dim + kv_dim, kv_dim);
        }
        q = velri_b_linear(b, x, w_q, b_q);
    } else {
        /* Split Q / K / V (also supports compressed MLA where d_lat != d_model). */
        q = velri_block_linear(b, x, prefix, spec->q_name, VELRI_DTYPE_F32, q_dim, spec->qkv_bias);
        if (!q.valid ||
            !velri_blocks_make_proj_weights(b, prefix, spec->k_name, VELRI_DTYPE_F32,
                                            VELRI_DTYPE_F32, kv_dim, d_lat, false, &w_k, nullptr) ||
            !velri_blocks_make_proj_weights(b, prefix, spec->v_name, VELRI_DTYPE_F32,
                                            VELRI_DTYPE_F32, kv_dim, d_lat, spec->qkv_bias, &w_v,
                                            &b_v)) {
            goto cleanup;
        }
    }
    if (b->err != VELRI_OK) {
        goto cleanup;
    }

    /* 1. Weight absorption: q_abs[t, h] = q[t, h] @ W_K[kvh] ([T, n_heads * d_lat]). */
    velri_tensor q_abs = velri_b_batched_linear(b, q, w_k, VELRI_TENSOR_NONE, spec->n_heads,
                                                spec->n_kv_heads, spec->head_dim, d_lat, true);

    /* 2. Unified attention core over single shared latent cache H_cache [1, max_ctx, d_lat].
     *    Scale uses the original head_dim (not d_lat) because Q_abs . H^T == Q . (H W_K^T)^T. */
    const size_t dim_h[3] = {1U, spec->max_ctx, d_lat};
    velri_tensor h_cache = alloc_cache_state(b, prefix, spec->state_prefix, ".h_cache",
                                             spec->kv_cache_scope, VELRI_DTYPE_F32, dim_h);
    if (!h_cache.valid) {
        goto cleanup;
    }

    const velri_attention_core_spec core = {
        .n_heads = spec->n_heads,
        .n_kv_heads = 1U,
        .head_dim_k = d_lat,
        .head_dim_v = d_lat,
        .sliding_window = spec->sliding_window,
        .scale = (spec->scale > 0.0f) ? spec->scale : (1.0f / sqrtf((float)spec->head_dim)),
        .causal = spec->causal,
        .inplace_q = true,
    };
    velri_tensor h_bar =
        velri_block_attention_core(b, q_abs, h_lat, h_lat, h_cache, h_cache, &core);

    /* 3. Value projection: attn[t, h] = h_bar[t, h] @ W_V[kvh]^T + b_V[kvh]. */
    velri_tensor attn = velri_b_batched_linear(b, h_bar, w_v, b_v, spec->n_heads, spec->n_kv_heads,
                                               d_lat, spec->head_dim, false);

    out = velri_block_linear(b, attn, prefix, spec->o_name, VELRI_DTYPE_F32, d_model, spec->o_bias);

cleanup:
    if (pushed_scope) {
        velri_b_pop_scope(b);
    }
    return (b->err == VELRI_OK) ? out : VELRI_TENSOR_NONE;
}

velri_tensor velri_block_dense_attention(velri_builder *b, velri_tensor x, const char *prefix,
                                         const velri_attention_spec *spec) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }

    size_t q_dim = 0U, kv_dim = 0U, qkv_dim = 0U;
    if (!validate_attention_common(b, x, spec, &q_dim, &kv_dim, &qkv_dim)) {
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
    velri_dtype act_dtype = velri_blocks_linear_out_dtype(x.dtype, spec->dtype);

    velri_tensor q = VELRI_TENSOR_NONE;
    velri_tensor k = VELRI_TENSOR_NONE;
    velri_tensor v = VELRI_TENSOR_NONE;

    if (spec->fused_qkv) {
        velri_tensor qkv =
            velri_block_linear(b, x, prefix, spec->qkv_name, spec->dtype, qkv_dim, spec->qkv_bias);
        q = velri_b_subview_2d_cols(b, qkv, 0U, q_dim);
        k = velri_b_subview_2d_cols(b, qkv, q_dim, kv_dim);
        v = velri_b_subview_2d_cols(b, qkv, q_dim + kv_dim, kv_dim);
    } else {
        q = velri_block_linear(b, x, prefix, spec->q_name, spec->dtype, q_dim, spec->qkv_bias);
        k = velri_block_linear(b, x, prefix, spec->k_name, spec->dtype, kv_dim, spec->qkv_bias);
        v = velri_block_linear(b, x, prefix, spec->v_name, spec->dtype, kv_dim, spec->qkv_bias);
    }
    if (b->err != VELRI_OK) {
        goto cleanup;
    }

    if (spec->rope.enabled) {
        uint32_t rdim = spec->rope.rotary_dim ? spec->rope.rotary_dim : spec->head_dim;
        if (rdim == 0U || (rdim & 1U) != 0U || rdim > spec->head_dim || !spec->rope.cos_name ||
            spec->rope.cos_name[0] == '\0' || !spec->rope.sin_name ||
            spec->rope.sin_name[0] == '\0') {
            velri_b_set_error(b, VELRI_ERR_INVALID);
            goto cleanup;
        }
        const size_t tbl_dims[2] = {spec->max_ctx, (size_t)(rdim / 2U)};
        velri_tensor cos_tbl =
            velri_b_weight_shared(b, spec->rope.cos_name, VELRI_DTYPE_F32, 2U, tbl_dims);
        velri_tensor sin_tbl =
            velri_b_weight_shared(b, spec->rope.sin_name, VELRI_DTYPE_F32, 2U, tbl_dims);
        velri_tensor q3 = velri_b_reshape_2d_to_3d(b, q, spec->n_heads, spec->head_dim);
        velri_tensor k3 = velri_b_reshape_2d_to_3d(b, k, spec->n_kv_heads, spec->head_dim);
        velri_tensor q3_rot =
            velri_b_rope_inplace(b, q3, cos_tbl, sin_tbl, rdim, 0U, spec->rope.mode);
        velri_tensor k3_rot =
            velri_b_rope_inplace(b, k3, cos_tbl, sin_tbl, rdim, 0U, spec->rope.mode);
        if (b->err != VELRI_OK) {
            goto cleanup;
        }
        if (q3_rot.storage_id != q.storage_id || q3_rot.view_id != q3.view_id) {
            q = velri_b_reshape_3d_to_2d(b, q3_rot);
        }
        if (k3_rot.storage_id != k.storage_id || k3_rot.view_id != k3.view_id) {
            k = velri_b_reshape_3d_to_2d(b, k3_rot);
        }
        if (b->err != VELRI_OK) {
            goto cleanup;
        }
    }

    const size_t dim_kv[3] = {spec->n_kv_heads, spec->max_ctx, spec->head_dim};
    velri_tensor k_cache = alloc_cache_state(b, prefix, spec->state_prefix, ".k_cache",
                                             spec->kv_cache_scope, act_dtype, dim_kv);
    velri_tensor v_cache = alloc_cache_state(b, prefix, spec->state_prefix, ".v_cache",
                                             spec->kv_cache_scope, act_dtype, dim_kv);
    if (!k_cache.valid || !v_cache.valid) {
        goto cleanup;
    }

    const velri_attention_core_spec core = {
        .n_heads = spec->n_heads,
        .n_kv_heads = spec->n_kv_heads,
        .head_dim_k = spec->head_dim,
        .head_dim_v = spec->head_dim,
        .sliding_window = spec->sliding_window,
        .scale = spec->scale,
        .causal = spec->causal,
    };
    velri_tensor attn = velri_block_attention_core(b, q, k, v, k_cache, v_cache, &core);
    out = velri_block_linear(b, attn, prefix, spec->o_name, spec->dtype, d_model, spec->o_bias);

cleanup:
    if (pushed_scope) {
        velri_b_pop_scope(b);
    }
    return (b->err == VELRI_OK) ? out : VELRI_TENSOR_NONE;
}

velri_tensor velri_block_attention(velri_builder *b, velri_tensor x, const char *prefix,
                                   const velri_attention_spec *spec) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (spec && spec->latent_kv) {
        return velri_block_latent_attention(b, x, x, prefix, spec);
    }
    return velri_block_dense_attention(b, x, prefix, spec);
}
