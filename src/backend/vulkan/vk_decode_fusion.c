#include "velri_vk_abi.h"

typedef struct {
    uint64_t address;
    size_t bytes;
} device_span;

/* Input bindings already include their view offsets. Compare physical addresses since
 * distinct public storages can occupy overlapping regions of the same device arena. */
static device_span bound_span(const velri_buffer_binding *b, const velri_tensor_view *v) {
    const vk_buf *buf = b->mem_handle;
    size_t start, end;
    if (velri_tensor_view_byte_span(v, &start, &end) != VELRI_OK || b->byte_offset > buf->size ||
        end - start > buf->size - b->byte_offset) {
        return (device_span){0};
    }
    return (device_span){buf->bda + b->byte_offset, end - start};
}

static device_span input_span(const velri_submit_item *item, uint32_t i) {
    return bound_span(&item->inputs[i], &item->op->inputs[i]);
}

static device_span output_span(const velri_submit_item *item, uint32_t i) {
    return bound_span(&item->outputs[i], &item->op->outputs[i]);
}

static bool overlap(device_span a, device_span b) {
    return !a.bytes || !b.bytes ||
           (a.address <= b.address ? b.address - a.address < a.bytes
                                   : a.address - b.address < b.bytes);
}

/* Every fused write is observable. Reject writes overlapping the common inputs or
 * other writes, including partial aliases that would permit a cross-workgroup race. */
static bool independent(const device_span *reads, size_t nr, const device_span *writes, size_t nw) {
    for (size_t i = 0; i < nw; ++i) {
        for (size_t j = 0; j < nr; ++j) {
            if (overlap(writes[i], reads[j]))
                return false;
        }
        for (size_t j = 0; j < i; ++j) {
            if (overlap(writes[i], writes[j]))
                return false;
        }
    }
    return true;
}

static bool single_stage(const velri_backend_device *dev, const vk_encoded_dispatch *e, int pipe) {
    return e->num_stages == 1U && e->stages[0].pipe == dev->pipes[pipe];
}

/* The vectorized dot product reads contiguous float4s in one unbiased F32 row. */
static bool dense_gemv(const velri_backend_device *dev, const vk_encoded_dispatch *e,
                       vk_linear_args *a) {
    if (!single_stage(dev, e, PIPE_GEMV_F32))
        return false;
    memcpy(a, e->stages[0].pc, sizeof(*a));
    return a->rows == 1U && a->flags == 0U && a->k == a->w_stride && !(a->k & 3U) &&
           !((a->x | a->w) & 15U);
}

static bool half_rope(const velri_backend_device *dev, const vk_encoded_dispatch *e,
                      const vk_linear_args *projection, vk_rope_args *rope) {
    if (!single_stage(dev, e, PIPE_ROPE_F32))
        return false;
    memcpy(rope, e->stages[0].pc, sizeof(*rope));
    return (rope->mode & 1U) == VELRI_ROPE_HALF && rope->n_tokens == 1U &&
           rope->rotary_dim == rope->head_dim && !(rope->head_dim & 1U) &&
           rope->head_sx == rope->head_dim && rope->head_sy == rope->head_dim &&
           (uint64_t)rope->n_heads * rope->head_dim == projection->m && projection->y == rope->x &&
           rope->x == rope->y;
}

typedef struct {
    vk_rms_norm_args args;
    device_span input, weight, output;
} decode_norm;

static bool read_norm(const velri_backend_device *dev, const velri_submit_item *item,
                      const vk_encoded_dispatch *e, decode_norm *norm) {
    if (!single_stage(dev, e, PIPE_RMSNORM_F32))
        return false;
    memcpy(&norm->args, e->stages[0].pc, sizeof(norm->args));
    const vk_rms_norm_args *n = &norm->args;
    if (n->rows != 1U || !n->has_weight || n->unit_offset || n->dim > 2048U || (n->dim & 3U) ||
        ((n->x | n->w | n->y) & 15U))
        return false;
    norm->input = input_span(item, 0);
    norm->weight = input_span(item, 1);
    norm->output = output_span(item, 0);
    return true;
}

static bool qkv_rope_cache(velri_backend_device *dev, const velri_submit_item *items,
                           const vk_encoded_dispatch *e, const vk_linear_args *q,
                           const decode_norm *norm, vk_encoded_dispatch *out) {
    vk_linear_args k, v;
    vk_rope_args qr, kr;
    const bool f16 = e[5].num_stages >= 2U &&
                     e[5].stages[0].pipe == dev->pipes[PIPE_KV_APPEND_F16KV];
    if (!items[1].no_barrier_before || !items[2].no_barrier_before || e[5].num_stages < 2U ||
        (!f16 && e[5].stages[0].pipe != dev->pipes[PIPE_KV_APPEND_F32]) || !dense_gemv(dev, e + 1, &k) ||
        !dense_gemv(dev, e + 2, &v) || !half_rope(dev, e + 3, q, &qr) ||
        !half_rope(dev, e + 4, &k, &kr))
        return false;

    vk_kv_append_args kv;
    memcpy(&kv, e[5].stages[0].pc, sizeof(kv));
    if (q->x != k.x || q->x != v.x || q->k != k.k || q->k != v.k || k.m != v.m ||
        qr.head_dim != kr.head_dim || qr.cos_tbl != kr.cos_tbl || qr.sin_tbl != kr.sin_tbl ||
        qr.pos_offset != kr.pos_offset || qr.step_params != kr.step_params ||
        qr.step_params != kv.step_params || kv.rows != 1U || k.y != kv.k_in || v.y != kv.v_in ||
        kv.same_kv_cache || kv.n_kv_heads != kr.n_heads || kv.d_k != kr.head_dim ||
        kv.d_v != kr.head_dim || kv.k_row_stride != kr.head_dim || kv.v_row_stride != kr.head_dim ||
        kv.k_head_stride != (uint64_t)kv.max_ctx * kr.head_dim ||
        kv.v_head_stride != kv.k_head_stride)
        return false;

    const device_span reads[] = {norm ? norm->input : input_span(items, 0),
                                 input_span(items, 1),
                                 input_span(items + 1, 1),
                                 input_span(items + 2, 1),
                                 input_span(items + 3, 1),
                                 input_span(items + 3, 2),
                                 norm ? norm->weight : (device_span){0}};
    const device_span writes[] = {
        output_span(items, 0),     output_span(items + 1, 0),
        output_span(items + 2, 0), output_span(items + 5, 1),
        output_span(items + 5, 2), norm ? norm->output : (device_span){0}};
    if (!independent(reads, norm ? 7U : 6U, writes, norm ? 6U : 5U))
        return false;
    const uint64_t pairs = ((uint64_t)q->m + k.m + v.m) / 2U;
    if (pairs > UINT32_MAX)
        return false;
    const uint32_t threads = norm ? VK_ROW_WG_SIZE : VK_GEMV_WG_SIZE;
    const uint32_t gx = velri_div_ceil_u32((uint32_t)pairs, threads / dev->subgroup_size);
    if (gx > dev->max_wg_count[0])
        return false;
    if (norm && (!kv.step_params || qr.head_dim > UINT16_MAX || qr.n_heads > UINT16_MAX ||
                 kr.n_heads > UINT16_MAX))
        return false;
    const int pipe_id = norm ? (f16 ? PIPE_RMS_QKV_F16KV : PIPE_RMS_QKV_F32)
                             : (f16 ? PIPE_QKV_ROPE_CACHE_F16KV : PIPE_QKV_ROPE_CACHE_F32);
    VkPipeline pipe = dev->pipes[pipe_id];
    if (!pipe)
        return false;
    *out = e[5];
    if (norm) {
        const vk_rms_qkv_args normalized = {
            .x = norm->args.x,
            .qw = q->w,
            .kw = k.w,
            .vw = v.w,
            .q = q->y,
            .k = k.y,
            .v = v.y,
            .c = qr.cos_tbl,
            .s = qr.sin_tbl,
            .k_cache = kv.k_cache,
            .v_cache = kv.v_cache,
            .step = kv.step_params,
            .norm_weight = norm->args.w,
            .norm_y = norm->args.y,
            .dims = (qr.head_dim << 16U) | (q->k / 4U),
            .heads = (kr.n_heads << 16U) | qr.n_heads,
            .max_ctx = kv.max_ctx,
            .eps = norm->args.eps,
        };
        set_dispatch_stage(&out->stages[0], pipe, &normalized, sizeof(normalized), gx, 1U, 1U);
        return true;
    }
    const vk_qkv_rope_cache_args a = {
        .x = q->x,
        .qw = q->w,
        .kw = k.w,
        .vw = v.w,
        .q = q->y,
        .k = k.y,
        .v = v.y,
        .c = qr.cos_tbl,
        .s = qr.sin_tbl,
        .k_cache = kv.k_cache,
        .v_cache = kv.v_cache,
        .step = kv.step_params,
        .k4 = q->k / 4U,
        .q_heads = qr.n_heads,
        .kv_heads = kr.n_heads,
        .head_dim = kr.head_dim,
        .max_ctx = kv.max_ctx,
        .pos = qr.pos_offset,
        .kv_pos = kv.kv_pos,
    };
    /* Replace only KV append. Keep decode or Split-KV and its private scratch dependencies. */
    set_dispatch_stage(&out->stages[0], pipe, &a, sizeof(a), gx, 1U, 1U);
    return true;
}

static bool gated_gemv(velri_backend_device *dev, const velri_submit_item *items,
                       const vk_encoded_dispatch *e, const vk_linear_args *gate,
                       const decode_norm *norm, vk_encoded_dispatch *out) {
    vk_linear_args up;
    if (!items[1].no_barrier_before || !single_stage(dev, e + 2, PIPE_GATED_SILU_F32) ||
        !dense_gemv(dev, e + 1, &up))
        return false;
    vk_gated_mlp_args act;
    memcpy(&act, e[2].stages[0].pc, sizeof(act));
    if (gate->x != up.x || gate->m != up.m || gate->k != up.k || act.count != gate->m ||
        gate->y != act.gate || up.y != act.up || act.y != gate->y)
        return false;
    const device_span reads[] = {norm ? norm->input : input_span(items, 0), input_span(items, 1),
                                 input_span(items + 1, 1), norm ? norm->weight : (device_span){0}};
    const device_span writes[] = {output_span(items, 0), output_span(items + 1, 0),
                                  norm ? norm->output : (device_span){0}};
    if (!independent(reads, norm ? 4U : 3U, writes, norm ? 3U : 2U))
        return false;
    VkPipeline pipe = dev->pipes[norm ? PIPE_RMS_GATED_GEMV_F32 : PIPE_GATED_GEMV_F32];
    if (!pipe)
        return false;
    const vk_gated_gemv_args a = {
        .x = norm ? norm->args.x : gate->x,
        .gate_w = gate->w,
        .up_w = up.w,
        .up = up.y,
        .y = gate->y,
        .m = gate->m,
        .k4 = gate->k / 4U,
        .w_stride4 = gate->w_stride / 4U,
        .clamp_limit = act.clamp_limit,
        .norm_weight = norm ? norm->args.w : 0U,
        .norm_y = norm ? norm->args.y : 0U,
        .eps = norm ? norm->args.eps : 0.0F,
    };
    *out = e[0];
    const uint32_t threads = norm ? VK_ROW_WG_SIZE : VK_GEMV_WG_SIZE;
    const uint32_t gx = velri_div_ceil_u32(gate->m, threads / dev->subgroup_size);
    set_dispatch_stage(&out->stages[0], pipe, &a, sizeof(a), gx, 1U, 1U);
    return true;
}

static bool gemv_residual(velri_backend_device *dev, const velri_submit_item *items,
                          const vk_encoded_dispatch *e, const vk_linear_args *linear,
                          vk_encoded_dispatch *out) {
    if (!single_stage(dev, e + 1, PIPE_VEC_AXPBY_F32))
        return false;
    vk_vec_axpby_args add;
    memcpy(&add, e[1].stages[0].pc, sizeof(add));
    if (add.a != add.y || add.b != linear->y || add.alpha != 1.0F || add.beta != 1.0F ||
        add.count != linear->m)
        return false;
    const device_span reads[] = {input_span(items, 0), input_span(items, 1)};
    const device_span writes[] = {output_span(items, 0), output_span(items + 1, 0)};
    if (!independent(reads, 2, writes, 2))
        return false;
    VkPipeline pipe = dev->pipes[PIPE_GEMV_RESIDUAL_F32];
    if (!pipe)
        return false;
    const vk_gemv_residual_args a = {
        .x = linear->x,
        .w = linear->w,
        .y = linear->y,
        .residual = add.a,
        .k4 = linear->k / 4U,
        .m = linear->m,
    };
    *out = e[0];
    set_dispatch_stage(&out->stages[0], pipe, &a, sizeof(a), e->stages[0].gx, 1U, 1U);
    return true;
}

uint32_t velri_vk_fuse_decode(velri_backend_device *dev, const velri_submit_item *items,
                              const vk_encoded_dispatch *e, uint32_t count,
                              vk_encoded_dispatch *out) {
    if (dev->subgroup_size != 32U || count < 2U)
        return 0U;
    decode_norm previous;
    const decode_norm *norm = nullptr;
    if (count >= 4U && read_norm(dev, items, e, &previous)) {
        norm = &previous;
        ++items;
        ++e;
        --count;
    }
    vk_linear_args first;
    if (!dense_gemv(dev, e, &first) ||
        (norm && (norm->args.y != first.x || norm->args.dim != first.k)))
        return 0U;
    if (count >= 6U && qkv_rope_cache(dev, items, e, &first, norm, out))
        return norm ? 7U : 6U;
    if (count >= 3U && gated_gemv(dev, items, e, &first, norm, out))
        return norm ? 4U : 3U;
    return !norm && gemv_residual(dev, items, e, &first, out) ? 2U : 0U;
}
