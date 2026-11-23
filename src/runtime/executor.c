#include "core_internal.h"
#include <string.h>

/*
 * Execution Slots & Runtime Launch-Graph Executor (`velri-lib/docs/EN/execution.md`).
 *
 * Key invariants:
 *   - Immutable `VELRI_STORAGE_WEIGHT` buffers live once in per-device weight slabs on `velri_plan`
 *     (or zero-copy host pointers on CPU) and are shared read-only across all concurrent slots.
 *   - `VELRI_STORAGE_STATE` lives in one contiguous state arena per device per slot and remains
 *     resident across steps with 0 B shadow RAM required (optional host binding for explicit sync).
 *   - `VELRI_STORAGE_INPUT`/`OUTPUT` on non-host devices share one contiguous IO arena per slot.
 *   - Direct CPU <-> non-host transfers bypass `ram_staging` completely (0 B staging RAM).
 *   - Consecutive ops targeting the same device are batched via `submit_batch` (0 B hot-path allocs).
 *   - Dynamic bounds (`seq_len_override`, `pos_offset_override`, `kv_len_override`) are validated
 *     strictly before launching any device work via `velri_op_validate_step_params`.
 */

typedef struct {
    void *host_ptr;
    size_t bound_bytes;
    bool is_bound;
} slot_storage_binding;

/* Refcounted persistent STATE ownership shared across dual prefill/decode slots without pinning
 * the creator slot's SCRATCH, IO, ram_staging, or captured command buffers. */
typedef struct {
    _Atomic uint32_t refcount;
    velri_context *ctx;
    void *arenas[VELRI_MAX_DEVICES];
    size_t arena_bytes[VELRI_MAX_DEVICES];
    size_t arena_align[VELRI_MAX_DEVICES];
    uint32_t num_state_storages;
    bool *state_initialized; /* [num_state_storages] by canonical (device, order) ordinal */
    bool needs_reset;
    bool state_clean; /* True when STATE arenas are already freshly zeroed (prevents 2x zeroing) */
} velri_state_group;

typedef struct {
    velri_device_id device_id;
    velri_captured_graph *graph;
    const void *readback; /* Snapshot at plan->readback.offset after successful replay and wait. */
} slot_captured_batch;

struct velri_slot {
    velri_plan *plan;
    velri_state_group *state_group; /* Retained shared or private STATE arena group */
    uint32_t *state_ordinal;        /* [num_user_storages] -> ordinal in state_group (or UINT32_MAX) */
    void *arenas[VELRI_SLOT_ARENA_COUNT][VELRI_MAX_DEVICES];
    void *ram_staging;
    slot_storage_binding *user_bindings;

    /* Pre-allocated batch scratch (sized to plan->max_batch_len => 0 B hot-path allocs). */
    velri_submit_item *batch_items;
    velri_buffer_binding (*batch_in)[VELRI_OP_MAX_INPUTS];
    velri_buffer_binding (*batch_out)[VELRI_OP_MAX_OUTPUTS];
    uint32_t batch_cap;

    /* Optional pre-recorded persistent decode launch graphs (captured in velri_slot_create). */
    slot_captured_batch *captured;

    bool needs_reset;
    uint64_t last_run_allocs;
    uint64_t last_run_graph_replays;
};

static velri_buffer_binding resolve_storage_binding(const velri_slot *slot,
                                                    uint32_t plan_storage_idx,
                                                    size_t view_byte_offset) {
    const velri_plan *plan = slot->plan;
    const velri_plan_storage *ps = &plan->storages[plan_storage_idx];
    const velri_device_entry *de = &plan->ctx->devices[ps->device_id];
    velri_buffer_binding b;
    memset(&b, 0, sizeof(b));

    switch (ps->arena_kind) {
    case VELRI_ARENA_SCRATCH:
    case VELRI_ARENA_STATE:
    case VELRI_ARENA_IO: {
        velri_slot_arena_id aid = (ps->arena_kind == VELRI_ARENA_SCRATCH)
                                      ? VELRI_SLOT_ARENA_SCRATCH
                                      : (ps->arena_kind == VELRI_ARENA_STATE)
                                            ? VELRI_SLOT_ARENA_STATE
                                            : VELRI_SLOT_ARENA_IO;
        b.mem_handle = slot->arenas[aid][ps->device_id];
        b.byte_offset = ps->arena_offset + view_byte_offset;
        b.byte_size = plan->slot_arena_bytes[aid][ps->device_id];
        b.is_host_ptr = (ps->arena_kind == VELRI_ARENA_IO) ? false : de->caps.is_host_memory;
        return b;
    }

    case VELRI_ARENA_WEIGHT:
        b.mem_handle = plan->weight_slabs[ps->device_id][ps->slab_idx];
        b.byte_offset = ps->arena_offset + view_byte_offset;
        b.byte_size = plan->weight_slab_bytes[ps->device_id][ps->slab_idx];
        b.is_host_ptr = false;
        return b;

    case VELRI_ARENA_NONE:
    default:
        break;
    }

    if (ps->kind == VELRI_STORAGE_WEIGHT) {
        const velri_plan *wplan = plan->weight_owner ? plan->weight_owner : plan;
        uint32_t widx = plan->weight_owner ? plan->weight_peer_sid[plan_storage_idx] : plan_storage_idx;
        const velri_plan_weight_binding *wb = &wplan->weight_bindings[widx];
        b.mem_handle = wb->packed_mem ? wb->packed_mem : (void *)wb->host_ptr;
        b.byte_offset = view_byte_offset;
        b.byte_size = wb->bound_bytes;
        b.is_host_ptr = true;
        return b;
    }

    const slot_storage_binding *sb = &slot->user_bindings[plan_storage_idx];
    b.mem_handle = sb->host_ptr;
    b.byte_offset = view_byte_offset;
    b.byte_size = sb->bound_bytes;
    b.is_host_ptr = true;
    return b;
}

static velri_status validate_host_bind(const velri_plan *plan, velri_storage_id storage_id,
                                       const void *host_ptr, size_t byte_size, bool expect_weight) {
    if (!plan || !host_ptr || storage_id >= plan->num_user_storages) {
        return VELRI_ERR_INVALID;
    }
    const velri_plan_storage *ps = &plan->storages[storage_id];
    if (expect_weight) {
        if (ps->kind != VELRI_STORAGE_WEIGHT) {
            return VELRI_ERR_INVALID;
        }
    } else if (ps->kind == VELRI_STORAGE_SCRATCH || ps->kind == VELRI_STORAGE_WEIGHT) {
        return VELRI_ERR_INVALID;
    }
    if (byte_size < ps->byte_size) {
        return VELRI_ERR_INVALID;
    }
    size_t req_align = velri_dtype_align(ps->dtype);
    if (((uintptr_t)host_ptr % req_align) != 0U) {
        return VELRI_ERR_INVALID;
    }
    return VELRI_OK;
}

static velri_status copy_host_to_storage(const velri_slot *slot, uint32_t storage_id,
                                         const void *src_host) {
    const velri_plan *plan = slot->plan;
    const velri_plan_storage *ps = &plan->storages[storage_id];
    velri_device_entry *de = &plan->ctx->devices[ps->device_id];
    velri_buffer_binding dst = resolve_storage_binding(slot, storage_id, 0U);
    if (de->caps.is_host_memory) {
        if (ps->byte_size > 0U) {
            memcpy(velri_binding_ptr(&dst), src_host, ps->byte_size);
        }
        return VELRI_OK;
    }
    return de->vt->copy_from_ram(de->dev, &dst, src_host, ps->byte_size);
}

static velri_status copy_storage_to_host(const velri_slot *slot, uint32_t storage_id,
                                         void *dst_host) {
    const velri_plan *plan = slot->plan;
    const velri_plan_storage *ps = &plan->storages[storage_id];
    velri_device_entry *de = &plan->ctx->devices[ps->device_id];
    velri_buffer_binding src = resolve_storage_binding(slot, storage_id, 0U);
    if (de->caps.is_host_memory) {
        if (ps->byte_size > 0U) {
            memcpy(dst_host, velri_binding_cptr(&src), ps->byte_size);
        }
        return VELRI_OK;
    }
    return de->vt->copy_to_ram(de->dev, dst_host, &src, ps->byte_size);
}

static velri_status zero_device_arena(velri_context *ctx, velri_device_id d, void *handle,
                                      size_t sz) {
    if (!handle || sz == 0U) {
        return VELRI_OK;
    }
    velri_device_entry *de = &ctx->devices[d];
    if (de->caps.is_host_memory) {
        memset(handle, 0, sz);
        return VELRI_OK;
    }
    if (de->vt->zero_mem) {
        return de->vt->zero_mem(de->dev, handle, sz);
    }
    /* Portable 0-alloc fallback for non-host backends that omit optional `zero_mem`. */
    static const uint8_t k_zero_chunk[65536] = {0};
    size_t off = 0U;
    while (off < sz) {
        size_t chunk = sz - off;
        if (chunk > sizeof(k_zero_chunk)) {
            chunk = sizeof(k_zero_chunk);
        }
        velri_buffer_binding dst = {
            .mem_handle = handle,
            .byte_offset = off,
            .byte_size = sz,
            .is_host_ptr = false,
        };
        velri_status st = de->vt->copy_from_ram(de->dev, &dst, k_zero_chunk, chunk);
        if (st != VELRI_OK) {
            return st;
        }
        off += chunk;
    }
    return VELRI_OK;
}

velri_status velri_plan_bind_weight_host(velri_plan *plan, velri_storage_id storage_id,
                                         const void *host_ptr, size_t byte_size) {
    velri_status st = validate_host_bind(plan, storage_id, host_ptr, byte_size, true);
    if (st != VELRI_OK) {
        return st;
    }
    if (plan->weight_owner) {
        uint32_t peer_sid = plan->weight_peer_sid ? plan->weight_peer_sid[storage_id] : UINT32_MAX;
        if (peer_sid == UINT32_MAX) {
            return VELRI_ERR_INVALID;
        }
        return velri_plan_bind_weight_host(plan->weight_owner, peer_sid, host_ptr, byte_size);
    }

    const velri_plan_storage *ps = &plan->storages[storage_id];
    velri_plan_weight_binding *wb = &plan->weight_bindings[storage_id];
    velri_device_entry *de = &plan->ctx->devices[ps->device_id];
    if (wb->packed_mem) {
        de->hooks->pack_weight(de->dev, &ps->packing, host_ptr, wb->packed_mem);
        wb->bound_bytes = ps->packing.bytes;
        wb->is_bound = true;
        return VELRI_OK;
    }
    wb->host_ptr = host_ptr;
    wb->bound_bytes = byte_size;

    if (!de->caps.is_host_memory) {
        velri_buffer_binding dst = {
            .mem_handle = plan->weight_slabs[ps->device_id][ps->slab_idx],
            .byte_offset = ps->arena_offset,
            .byte_size = plan->weight_slab_bytes[ps->device_id][ps->slab_idx],
            .is_host_ptr = false,
        };
        st = de->vt->copy_from_ram(de->dev, &dst, host_ptr, ps->byte_size);
        if (st != VELRI_OK) {
            wb->is_bound = false;
            return st;
        }
    }
    wb->is_bound = true;
    return VELRI_OK;
}

static bool populate_batch_items(velri_slot *slot, uint32_t first_step, uint32_t count,
                                 const velri_step_params *params) {
    const velri_plan *plan = slot->plan;
    for (uint32_t k = 0U; k < count; ++k) {
        const velri_launch_step *step = &plan->steps[first_step + k];
        const velri_resolved_op *op = &step->u.op.resolved;
        memset(slot->batch_in[k], 0, sizeof(slot->batch_in[k]));
        memset(slot->batch_out[k], 0, sizeof(slot->batch_out[k]));
        for (uint32_t i = 0U; i < op->num_inputs; ++i) {
            if (op->input_present[i]) {
                slot->batch_in[k][i] = resolve_storage_binding(slot, step->u.op.in_storage_idx[i],
                                                               op->inputs[i].byte_offset);
                if (!slot->batch_in[k][i].mem_handle) {
                    return false;
                }
            }
        }
        for (uint32_t o = 0U; o < op->num_outputs; ++o) {
            slot->batch_out[k][o] = resolve_storage_binding(slot, step->u.op.out_storage_idx[o],
                                                            op->outputs[o].byte_offset);
            if (!slot->batch_out[k][o].mem_handle) {
                return false;
            }
        }
        slot->batch_items[k] = (velri_submit_item){
            .op = op,
            .prepared = step->u.op.uses_packed_weight
                            ? plan->ctx->devices[step->u.op.device_id].hooks->packed_weight_op
                            : step->u.op.prepared,
            .inputs = slot->batch_in[k],
            .num_inputs = op->num_inputs,
            .outputs = slot->batch_out[k],
            .num_outputs = op->num_outputs,
            .params = params,
            .no_barrier_before = (k > 0U) ? step->u.op.no_barrier_before : true,
        };
    }
    return true;
}

static velri_status validate_dynamic_params(const velri_plan *plan,
                                            const velri_step_params *params);

/* Best-effort eager capture of T=1 persistent decode graphs on backends supporting capture_graph. */
static void slot_capture_decode_graphs(velri_slot *slot) {
    const velri_plan *plan = slot->plan;
    if (plan->num_capture_batches == 0U) {
        return;
    }

    const velri_step_params cap_params = {
        .pos_offset_override = 0U,
        .has_pos_offset = true,
        .seq_len_override = 1U,
        .has_seq_len = true,
        .kv_len_override = 0U,
        .has_kv_len = true,
    };
    if (validate_dynamic_params(plan, &cap_params) != VELRI_OK) {
        return;
    }

    slot->captured = calloc(plan->num_capture_batches, sizeof(*slot->captured));
    if (!slot->captured) {
        return;
    }

    for (uint32_t s = 0U; s < plan->num_steps;) {
        const velri_launch_step *step = &plan->steps[s];
        if (step->kind == VELRI_STEP_RAM_TRANSFER) {
            s++;
            continue;
        }
        const velri_launch_batch *batch = &step->u.op.batch;
        velri_device_id dev_id = step->u.op.device_id;
        velri_device_entry *de = &plan->ctx->devices[dev_id];
        if (de->vt->capture_graph && populate_batch_items(slot, s, batch->count, &cap_params)) {
            slot_captured_batch cb = {.device_id = dev_id};
            velri_status st;
            if (plan->readback.bytes && s + batch->count == plan->num_steps) {
                void *io = slot->arenas[VELRI_SLOT_ARENA_IO][dev_id];
                st = de->hooks->capture_outputs(de->dev, slot->batch_items, batch->count, io,
                                                plan->readback, &cb.readback, &cb.graph);
            } else {
                st = de->vt->capture_graph(de->dev, slot->batch_items, batch->count, &cb.graph);
            }
            if (st == VELRI_OK) {
                slot->captured[batch->capture_index] = cb;
            }
        }
        s += batch->count;
    }
}

static void state_group_retain(velri_state_group *sg) {
    if (sg) {
        velri_ref_inc(&sg->refcount);
    }
}

static void state_group_release(velri_state_group *sg) {
    if (!sg || !velri_ref_dec(&sg->refcount)) {
        return;
    }
    if (sg->ctx) {
        for (uint32_t d = 0U; d < sg->ctx->num_devices; ++d) {
            if (sg->arenas[d]) {
                velri_device_entry *de = &sg->ctx->devices[d];
                de->vt->free(de->dev, sg->arenas[d]);
            }
        }
        velri_context_release(sg->ctx);
    }
    free(sg->state_initialized);
    free(sg);
}

static velri_status state_group_create(velri_plan *plan, velri_state_group **out_sg) {
    *out_sg = nullptr;
    velri_state_group *sg = (velri_state_group *)calloc(1U, sizeof(*sg));
    if (!sg) {
        return VELRI_ERR_NOMEM;
    }
    atomic_init(&sg->refcount, 1U);
    velri_context_retain(plan->ctx);
    sg->ctx = plan->ctx;
    sg->state_clean = true;

    uint32_t n_state = 0U;
    for (uint32_t d = 0U; d < plan->ctx->num_devices; ++d) {
        for (uint32_t i = 0U; i < plan->num_user_storages; ++i) {
            if (plan->storages[i].kind == VELRI_STORAGE_STATE &&
                plan->storages[i].device_id == d) {
                n_state++;
            }
        }
    }
    sg->num_state_storages = n_state;
    if (n_state > 0U) {
        sg->state_initialized = (bool *)calloc(n_state, sizeof(bool));
        if (!sg->state_initialized) {
            state_group_release(sg);
            return VELRI_ERR_NOMEM;
        }
    }

    for (uint32_t d = 0U; d < plan->ctx->num_devices; ++d) {
        size_t sz = plan->slot_arena_bytes[VELRI_SLOT_ARENA_STATE][d];
        size_t align = plan->slot_arena_align[VELRI_SLOT_ARENA_STATE][d];
        sg->arena_bytes[d] = sz;
        sg->arena_align[d] = align;
        if (sz == 0U) {
            continue;
        }
        velri_status st = velri_device_alloc_tracked(plan->ctx, d, sz, align, VELRI_ARENA_STATE,
                                                     &sg->arenas[d]);
        if (st != VELRI_OK) {
            state_group_release(sg);
            return st;
        }
        st = zero_device_arena(plan->ctx, d, sg->arenas[d], sz);
        if (st != VELRI_OK) {
            state_group_release(sg);
            return st;
        }
    }

    *out_sg = sg;
    return VELRI_OK;
}

static velri_status populate_state_ordinals(const velri_plan *plan, uint32_t **out_ord) {
    *out_ord = nullptr;
    if (plan->num_user_storages == 0U) {
        return VELRI_OK;
    }
    size_t bytes = 0U;
    if (velri_mul_overflow_size((size_t)plan->num_user_storages, sizeof(uint32_t), &bytes)) {
        return VELRI_ERR_OVERFLOW;
    }
    uint32_t *ord = (uint32_t *)malloc(bytes);
    if (!ord) {
        return VELRI_ERR_NOMEM;
    }
    for (uint32_t i = 0U; i < plan->num_user_storages; ++i) {
        ord[i] = UINT32_MAX;
    }
    uint32_t cur = 0U;
    for (uint32_t d = 0U; d < plan->ctx->num_devices; ++d) {
        for (uint32_t i = 0U; i < plan->num_user_storages; ++i) {
            if (plan->storages[i].kind == VELRI_STORAGE_STATE &&
                plan->storages[i].device_id == d) {
                ord[i] = cur++;
            }
        }
    }
    *out_ord = ord;
    return VELRI_OK;
}

static velri_status validate_shared_state_layout(const velri_plan *plan,
                                                 const velri_plan *owner_plan) {
    if (!plan || !owner_plan || plan->ctx != owner_plan->ctx) {
        return VELRI_ERR_INVALID;
    }
    for (uint32_t d = 0U; d < plan->ctx->num_devices; ++d) {
        size_t sz = plan->slot_arena_bytes[VELRI_SLOT_ARENA_STATE][d];
        if (sz != owner_plan->slot_arena_bytes[VELRI_SLOT_ARENA_STATE][d]) {
            return VELRI_ERR_INVALID;
        }
        if (sz > 0U && plan->slot_arena_align[VELRI_SLOT_ARENA_STATE][d] >
                           owner_plan->slot_arena_align[VELRI_SLOT_ARENA_STATE][d]) {
            return VELRI_ERR_INVALID;
        }
        uint32_t j = 0U;
        for (uint32_t i = 0U; i < plan->num_user_storages; ++i) {
            const velri_plan_storage *ps = &plan->storages[i];
            if (ps->kind != VELRI_STORAGE_STATE || ps->device_id != d) {
                continue;
            }
            while (j < owner_plan->num_user_storages &&
                   (owner_plan->storages[j].kind != VELRI_STORAGE_STATE ||
                    owner_plan->storages[j].device_id != d)) {
                j++;
            }
            if (j >= owner_plan->num_user_storages) {
                return VELRI_ERR_INVALID;
            }
            const velri_plan_storage *ops = &owner_plan->storages[j];
            if (ps->dtype != ops->dtype || ps->byte_size != ops->byte_size ||
                ps->arena_kind != ops->arena_kind || ps->arena_offset != ops->arena_offset) {
                return VELRI_ERR_INVALID;
            }
            j++;
        }
        while (j < owner_plan->num_user_storages) {
            if (owner_plan->storages[j].kind == VELRI_STORAGE_STATE &&
                owner_plan->storages[j].device_id == d) {
                return VELRI_ERR_INVALID;
            }
            j++;
        }
    }
    return VELRI_OK;
}

static velri_status slot_create_internal(velri_plan *plan, velri_slot *state_owner,
                                         velri_slot **out_slot) {
    if (!plan || !out_slot) {
        return VELRI_ERR_INVALID;
    }
    *out_slot = nullptr;

    if (state_owner) {
        velri_status vst = validate_shared_state_layout(plan, state_owner->plan);
        if (vst != VELRI_OK) {
            return vst;
        }
    }

    velri_slot *slot = (velri_slot *)calloc(1U, sizeof(*slot));
    if (!slot) {
        return VELRI_ERR_NOMEM;
    }
    velri_plan_retain(plan);
    slot->plan = plan;

    velri_status st = populate_state_ordinals(plan, &slot->state_ordinal);
    if (st != VELRI_OK) {
        velri_slot_destroy(slot);
        return st;
    }

    if (state_owner) {
        state_group_retain(state_owner->state_group);
        slot->state_group = state_owner->state_group;
        slot->needs_reset = velri_slot_needs_reset(state_owner);
    } else {
        st = state_group_create(plan, &slot->state_group);
        if (st != VELRI_OK) {
            velri_slot_destroy(slot);
            return st;
        }
    }

    if (plan->num_user_storages > 0U) {
        slot->user_bindings =
            (slot_storage_binding *)calloc(plan->num_user_storages, sizeof(*slot->user_bindings));
        if (!slot->user_bindings) {
            velri_slot_destroy(slot);
            return VELRI_ERR_NOMEM;
        }
    }

    uint32_t bcap = (plan->max_batch_len > 0U) ? plan->max_batch_len : 1U;
    slot->batch_items = (velri_submit_item *)calloc(bcap, sizeof(*slot->batch_items));
    slot->batch_in = calloc(bcap, sizeof(*slot->batch_in));
    slot->batch_out = calloc(bcap, sizeof(*slot->batch_out));
    if (!slot->batch_items || !slot->batch_in || !slot->batch_out) {
        velri_slot_destroy(slot);
        return VELRI_ERR_NOMEM;
    }
    slot->batch_cap = bcap;

    for (uint32_t d = 0U; d < plan->ctx->num_devices; ++d) {
        for (uint32_t a = 0U; a < VELRI_SLOT_ARENA_COUNT; ++a) {
            size_t sz = plan->slot_arena_bytes[a][d];
            if (sz == 0U) {
                continue;
            }
            if (a == VELRI_SLOT_ARENA_STATE) {
                slot->arenas[a][d] = slot->state_group->arenas[d];
                continue;
            }
            velri_arena_kind kind =
                (a == VELRI_SLOT_ARENA_IO) ? VELRI_ARENA_IO : VELRI_ARENA_SCRATCH;
            st = velri_device_alloc_tracked(plan->ctx, d, sz, plan->slot_arena_align[a][d],
                                            kind, &slot->arenas[a][d]);
            if (st != VELRI_OK) {
                velri_slot_destroy(slot);
                return st;
            }
        }
    }

    if (plan->ram_staging_bytes > 0U) {
        slot->ram_staging = velri_os_aligned_alloc(64U, plan->ram_staging_bytes);
        if (!slot->ram_staging) {
            velri_slot_destroy(slot);
            return VELRI_ERR_NOMEM;
        }
        memset(slot->ram_staging, 0, plan->ram_staging_bytes);
    }

    slot_capture_decode_graphs(slot);

    *out_slot = slot;
    return VELRI_OK;
}

velri_status velri_slot_create(velri_plan *plan, velri_slot **out_slot) {
    return slot_create_internal(plan, nullptr, out_slot);
}

velri_status velri_slot_create_shared_state(velri_plan *plan, velri_slot *state_owner,
                                            velri_slot **out_slot) {
    if (!state_owner) {
        return VELRI_ERR_INVALID;
    }
    return slot_create_internal(plan, state_owner, out_slot);
}

void velri_slot_destroy(velri_slot *slot) {
    if (!slot) {
        return;
    }
    velri_plan *plan = slot->plan;
    velri_state_group *sg = slot->state_group;
    if (plan && plan->ctx) {
        /* Destroy captured graphs BEFORE releasing `state_group` so shared STATE BDAs stay valid. */
        for (uint32_t i = 0U; slot->captured && i < plan->num_capture_batches; ++i) {
            const slot_captured_batch *batch = &slot->captured[i];
            if (batch->graph) {
                velri_device_entry *de = &plan->ctx->devices[batch->device_id];
                de->vt->destroy_captured_graph(de->dev, batch->graph);
            }
        }
        for (uint32_t d = 0U; d < plan->ctx->num_devices; ++d) {
            velri_device_entry *de = &plan->ctx->devices[d];
            for (uint32_t a = VELRI_SLOT_ARENA_COUNT; a > 0U; --a) {
                uint32_t aid = a - 1U;
                if (aid == VELRI_SLOT_ARENA_STATE) {
                    continue; /* Owned exclusively by `velri_state_group`. */
                }
                if (slot->arenas[aid][d]) {
                    de->vt->free(de->dev, slot->arenas[aid][d]);
                }
            }
        }
    }
    free(slot->captured);
    free(slot->batch_items);
    free(slot->batch_in);
    free(slot->batch_out);
    velri_os_aligned_free(slot->ram_staging);
    free(slot->user_bindings);
    free(slot->state_ordinal);
    free(slot);
    if (sg) {
        state_group_release(sg);
    }
    velri_plan_release(plan);
}

velri_status velri_slot_bind_host(velri_slot *slot, velri_storage_id storage_id, void *host_ptr,
                                  size_t byte_size) {
    if (!slot) {
        return VELRI_ERR_INVALID;
    }
    velri_status st = validate_host_bind(slot->plan, storage_id, host_ptr, byte_size, false);
    if (st != VELRI_OK) {
        return st;
    }

    slot_storage_binding *sb = &slot->user_bindings[storage_id];
    sb->host_ptr = host_ptr;
    sb->bound_bytes = byte_size;
    sb->is_bound = true;
    return VELRI_OK;
}

velri_status velri_slot_sync_state_from_host(velri_slot *slot, velri_storage_id storage_id) {
    if (!slot || storage_id >= slot->plan->num_user_storages) {
        return VELRI_ERR_INVALID;
    }
    const velri_plan_storage *ps = &slot->plan->storages[storage_id];
    slot_storage_binding *sb = &slot->user_bindings[storage_id];
    if (ps->kind != VELRI_STORAGE_STATE || !sb->is_bound) {
        return VELRI_ERR_INVALID;
    }
    velri_status st = copy_host_to_storage(slot, storage_id, sb->host_ptr);
    if (st == VELRI_OK && slot->state_group) {
        uint32_t ord = slot->state_ordinal ? slot->state_ordinal[storage_id] : UINT32_MAX;
        if (ord < slot->state_group->num_state_storages) {
            slot->state_group->state_initialized[ord] = true;
        }
        slot->state_group->state_clean = false;
    }
    return st;
}

velri_status velri_slot_sync_state_to_host(velri_slot *slot, velri_storage_id storage_id) {
    if (!slot || storage_id >= slot->plan->num_user_storages) {
        return VELRI_ERR_INVALID;
    }
    const velri_plan_storage *ps = &slot->plan->storages[storage_id];
    const slot_storage_binding *sb = &slot->user_bindings[storage_id];
    if (ps->kind != VELRI_STORAGE_STATE || !sb->is_bound) {
        return VELRI_ERR_INVALID;
    }
    return copy_storage_to_host(slot, storage_id, sb->host_ptr);
}

static velri_status validate_dynamic_params(const velri_plan *plan,
                                            const velri_step_params *params) {
    if (!params) {
        return VELRI_OK;
    }
    if (params->has_pos_offset && params->pos_offset_override > (size_t)UINT32_MAX) {
        return VELRI_ERR_OVERFLOW;
    }
    if (params->has_seq_len) {
        if (params->seq_len_override == 0U) {
            return VELRI_ERR_INVALID;
        }
        if (params->seq_len_override > (size_t)UINT32_MAX) {
            return VELRI_ERR_OVERFLOW;
        }
    }
    if (params->has_kv_len && params->kv_len_override > (size_t)UINT32_MAX) {
        return VELRI_ERR_OVERFLOW;
    }

    for (uint32_t s = 0U; s < plan->num_steps; ++s) {
        const velri_launch_step *step = &plan->steps[s];
        if (step->kind != VELRI_STEP_OP) {
            continue;
        }
        velri_status st = velri_op_validate_step_params(&step->u.op.resolved, params);
        if (st != VELRI_OK) {
            return st;
        }
    }
    return VELRI_OK;
}

static velri_status run_ram_transfer(velri_slot *slot, const velri_launch_step *step) {
    const velri_plan *plan = slot->plan;
    velri_device_entry *src_de = &plan->ctx->devices[step->u.transfer.src_device_id];
    velri_device_entry *dst_de = &plan->ctx->devices[step->u.transfer.dst_device_id];

    velri_buffer_binding src_b =
        resolve_storage_binding(slot, step->u.transfer.src_plan_storage_idx, 0U);
    velri_buffer_binding dst_b =
        resolve_storage_binding(slot, step->u.transfer.dst_plan_storage_idx, 0U);
    size_t n = step->u.transfer.byte_size;

    /* Direct zero-staging transfer when either side is already host memory. */
    if (src_b.is_host_ptr && dst_b.is_host_ptr) {
        if (n > 0U) {
            memcpy(velri_binding_ptr(&dst_b), velri_binding_cptr(&src_b), n);
        }
        return VELRI_OK;
    }
    if (src_b.is_host_ptr) {
        return dst_de->vt->copy_from_ram(dst_de->dev, &dst_b, velri_binding_cptr(&src_b), n);
    }
    if (dst_b.is_host_ptr) {
        return src_de->vt->copy_to_ram(src_de->dev, velri_binding_ptr(&dst_b), &src_b, n);
    }

    velri_status st = src_de->vt->copy_to_ram(src_de->dev, slot->ram_staging, &src_b, n);
    if (st != VELRI_OK) {
        return st;
    }
    return dst_de->vt->copy_from_ram(dst_de->dev, &dst_b, slot->ram_staging, n);
}

static velri_status run_compute_batch(velri_slot *slot, uint32_t first_step,
                                      const velri_step_params *params, bool *out_accepted,
                                      const slot_captured_batch **outputs) {
    *out_accepted = false;
    *outputs = nullptr;
    const velri_plan *plan = slot->plan;
    const velri_launch_batch *batch = &plan->steps[first_step].u.op.batch;
    uint32_t count = batch->count;
    velri_device_id dev_id = plan->steps[first_step].u.op.device_id;
    velri_device_entry *de = &plan->ctx->devices[dev_id];

    /* Fast path: replay pre-recorded T=1 decode launch graph if captured and accepted. */
    if (de->vt->replay_graph && slot->captured) {
        const slot_captured_batch *cb = &slot->captured[batch->capture_index];
        bool is_t1 =
            (params && params->has_seq_len) ? (params->seq_len_override == 1U) : batch->static_t1;
        if (cb->graph && is_t1) {
            velri_completion *comp = nullptr;
            velri_status r_st = de->vt->replay_graph(de->dev, cb->graph, params, &comp);
            if (r_st != VELRI_ERR_UNSUPPORTED || comp != nullptr) {
                if (de->vt->wait) {
                    if (!comp) {
                        return r_st;
                    }
                    *out_accepted = true;
                    velri_status wait_st = de->vt->wait(de->dev, comp);
                    de->vt->destroy_completion(de->dev, comp);
                    if (r_st == VELRI_OK && wait_st == VELRI_OK) {
                        slot->last_run_graph_replays++;
                        *outputs = cb;
                    }
                    return (r_st != VELRI_OK) ? r_st : wait_st;
                }
                if (r_st == VELRI_OK || r_st == VELRI_ERR_EXEC) {
                    *out_accepted = true;
                    if (r_st == VELRI_OK) {
                        slot->last_run_graph_replays++;
                    }
                }
                return r_st;
            }
            /* Pre-accept VELRI_ERR_UNSUPPORTED (e.g. long-ctx Split-KV) -> fall through to dynamic batch. */
        }
    }

    if (count > slot->batch_cap) {
        return VELRI_ERR_INVALID;
    }
    if (!populate_batch_items(slot, first_step, count, params)) {
        return VELRI_ERR_INVALID;
    }

    velri_completion *comp = nullptr;
    velri_status submit_st = (count > 1U && de->vt->submit_batch)
                                 ? de->vt->submit_batch(de->dev, slot->batch_items, count, &comp)
                                 : de->vt->submit(de->dev, &slot->batch_items[0], &comp);

    if (de->vt->wait) {
        if (!comp) {
            return submit_st;
        }
        *out_accepted = true;
        velri_status wait_st = de->vt->wait(de->dev, comp);
        de->vt->destroy_completion(de->dev, comp);
        return (submit_st != VELRI_OK) ? submit_st : wait_st;
    }

    if (submit_st == VELRI_OK || submit_st == VELRI_ERR_EXEC) {
        *out_accepted = true;
    }
    return submit_st;
}

static velri_status slot_check_bindings(const velri_slot *slot) {
    const velri_plan *plan = slot->plan;
    for (uint32_t i = 0U; i < plan->num_user_storages; ++i) {
        velri_storage_kind kind = plan->storages[i].kind;
        if (kind == VELRI_STORAGE_SCRATCH || kind == VELRI_STORAGE_STATE) {
            continue;
        }
        if (kind == VELRI_STORAGE_WEIGHT) {
            const velri_plan *wplan = plan->weight_owner ? plan->weight_owner : plan;
            uint32_t widx = plan->weight_owner ? plan->weight_peer_sid[i] : i;
            if (!wplan->weight_bindings[widx].is_bound) {
                return VELRI_ERR_INVALID;
            }
        } else if (!slot->user_bindings[i].is_bound) {
            return VELRI_ERR_INVALID;
        }
    }
    return VELRI_OK;
}

static velri_status slot_upload_inputs(velri_slot *slot, velri_cancel_token *cancel) {
    const velri_plan *plan = slot->plan;
    velri_state_group *sg = slot->state_group;
    for (uint32_t i = 0U; i < plan->num_user_storages; ++i) {
        const velri_plan_storage *ps = &plan->storages[i];
        const velri_device_entry *de = &plan->ctx->devices[ps->device_id];
        uint32_t ord = (ps->kind == VELRI_STORAGE_STATE && slot->state_ordinal)
                           ? slot->state_ordinal[i]
                           : UINT32_MAX;
        bool need_state_up = (ps->kind == VELRI_STORAGE_STATE && slot->user_bindings[i].is_bound &&
                              sg && ord < sg->num_state_storages && !sg->state_initialized[ord]);
        bool need_input_up = (!de->caps.is_host_memory && ps->kind == VELRI_STORAGE_INPUT);
        if (!need_input_up && !need_state_up) {
            continue;
        }
        if (velri_cancel_is_requested(cancel)) {
            return VELRI_ERR_CANCELLED;
        }
        velri_status st = copy_host_to_storage(slot, i, slot->user_bindings[i].host_ptr);
        if (st != VELRI_OK) {
            return st;
        }
        if (need_state_up) {
            sg->state_initialized[ord] = true;
            sg->state_clean = false;
        }
    }
    return VELRI_OK;
}

static velri_status slot_download_outputs(const velri_slot *slot,
                                          const slot_captured_batch *outputs) {
    const velri_plan *plan = slot->plan;
    for (uint32_t i = 0U; i < plan->num_user_storages; ++i) {
        const velri_plan_storage *ps = &plan->storages[i];
        const velri_device_entry *de = &plan->ctx->devices[ps->device_id];
        if (de->caps.is_host_memory || ps->kind != VELRI_STORAGE_OUTPUT) {
            continue;
        }
        if (outputs && outputs->readback && outputs->device_id == ps->device_id &&
            ps->arena_kind == VELRI_ARENA_IO) {
            if (ps->byte_size > 0U) {
                size_t offset = ps->arena_offset - plan->readback.offset;
                const uint8_t *src = (const uint8_t *)outputs->readback + offset;
                memcpy(slot->user_bindings[i].host_ptr, src, ps->byte_size);
            }
            continue;
        }
        velri_status st = copy_storage_to_host(slot, i, slot->user_bindings[i].host_ptr);
        if (st != VELRI_OK) {
            return st;
        }
    }
    return VELRI_OK;
}

velri_status velri_slot_run(velri_slot *slot, const velri_step_params *params,
                            velri_cancel_token *cancel) {
    if (!slot) {
        return VELRI_ERR_INVALID;
    }
    if (velri_slot_needs_reset(slot)) {
        return VELRI_ERR_BUSY;
    }

    const velri_plan *plan = slot->plan;
    uint64_t allocs_before = atomic_load_explicit(&plan->ctx->total_allocs, memory_order_relaxed);
    slot->last_run_allocs = 0U;
    slot->last_run_graph_replays = 0U;

    bool any_work_accepted = false;
    velri_status st = slot_check_bindings(slot);
    if (st != VELRI_OK) {
        goto finish;
    }

    velri_step_params default_params = {0};
    const velri_step_params *eff_params = params ? params : &default_params;
    st = validate_dynamic_params(plan, eff_params);
    if (st != VELRI_OK) {
        goto finish;
    }

    if (velri_cancel_is_requested(cancel)) {
        st = VELRI_ERR_CANCELLED;
        goto finish;
    }

    /* Upload step INPUTs (non-host) and initial host-bound STATE (both host & non-host). */
    st = slot_upload_inputs(slot, cancel);
    if (st != VELRI_OK) {
        goto finish;
    }

    const slot_captured_batch *outputs = nullptr;
    uint32_t s = 0U;
    while (s < plan->num_steps) {
        if (velri_cancel_is_requested(cancel)) {
            st = VELRI_ERR_CANCELLED;
            goto finish;
        }

        const velri_launch_step *step = &plan->steps[s];
        if (step->kind == VELRI_STEP_RAM_TRANSFER) {
            st = run_ram_transfer(slot, step);
            if (st != VELRI_OK) {
                goto finish;
            }
            s++;
            continue;
        }

        bool accepted = false;
        st = run_compute_batch(slot, s, eff_params, &accepted, &outputs);
        if (accepted) {
            any_work_accepted = true;
        }
        if (st != VELRI_OK) {
            goto finish;
        }
        s += step->u.op.batch.count;
    }

    /* Download ONLY non-host OUTPUTs (STATE remains resident on its home device!). */
    st = slot_download_outputs(slot, outputs);
    if (st != VELRI_OK) {
        any_work_accepted = true;
    }

finish:
    if (any_work_accepted && slot->state_group) {
        slot->state_group->state_clean = false;
        for (uint32_t i = 0U; i < slot->state_group->num_state_storages; ++i) {
            slot->state_group->state_initialized[i] = true;
        }
    }
    if (st != VELRI_OK && any_work_accepted) {
        slot->needs_reset = true;
        if (slot->state_group) {
            slot->state_group->needs_reset = true;
        }
    }
    slot->last_run_allocs =
        atomic_load_explicit(&plan->ctx->total_allocs, memory_order_relaxed) - allocs_before;
    return st;
}

bool velri_slot_needs_reset(const velri_slot *slot) {
    if (!slot) {
        return false;
    }
    return slot->needs_reset || (slot->state_group && slot->state_group->needs_reset);
}

velri_status velri_slot_reset(velri_slot *slot) {
    if (!slot) {
        return VELRI_ERR_INVALID;
    }
    const velri_plan *plan = slot->plan;
    velri_state_group *sg = slot->state_group;

    /* Always zero this slot's private SCRATCH arenas. */
    for (uint32_t d = 0U; d < plan->ctx->num_devices; ++d) {
        velri_status st = zero_device_arena(plan->ctx, d, slot->arenas[VELRI_SLOT_ARENA_SCRATCH][d],
                                            plan->slot_arena_bytes[VELRI_SLOT_ARENA_SCRATCH][d]);
        if (st != VELRI_OK) {
            return st;
        }
    }

    /* Zero shared STATE arenas at most ONCE across peer slots sharing `state_group`. */
    if (sg) {
        if (!sg->state_clean) {
            for (uint32_t d = 0U; d < plan->ctx->num_devices; ++d) {
                velri_status st = zero_device_arena(plan->ctx, d, sg->arenas[d], sg->arena_bytes[d]);
                if (st != VELRI_OK) {
                    return st;
                }
            }
            for (uint32_t i = 0U; i < sg->num_state_storages; ++i) {
                sg->state_initialized[i] = false;
            }
            sg->state_clean = true;
        }
        sg->needs_reset = false;
    }
    slot->needs_reset = false;
    return VELRI_OK;
}

uint64_t velri_slot_last_run_allocs(const velri_slot *slot) {
    return slot ? slot->last_run_allocs : 0U;
}

uint64_t velri_slot_last_run_graph_replays(const velri_slot *slot) {
    return slot ? slot->last_run_graph_replays : 0U;
}
