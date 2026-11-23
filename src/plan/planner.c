#include "core_internal.h"
#include <string.h>

/*
 * Memory/Placement Planner & Launch-Graph Preparation (`velri-lib/docs/EN/architecture.md §4`).
 *
 * Memory-efficiency guarantees:
 *   - Packs all non-host `VELRI_STORAGE_WEIGHT` storages into contiguous per-device weight slabs
 *     (up to VELRI_WEIGHT_SLAB_LIMIT each), eliminating per-tensor VkDeviceMemory allocations.
 *   - Packs all `VELRI_STORAGE_STATE` storages into one contiguous arena per device per slot.
 *   - Packs all non-host `VELRI_STORAGE_INPUT`/`OUTPUT` storages into one IO arena per device per slot.
 *   - Packs non-overlapping `VELRI_STORAGE_SCRATCH` storages (sorted descending by size, then
 *     lifetime span) into one liveness-packed arena per device per slot.
 *   - Eliminates `ram_staging` allocation whenever either transfer endpoint is host memory.
 */

typedef struct {
    velri_plan *plan;
    uint32_t storage_cap;
    uint32_t step_cap;
} plan_builder;

typedef struct {
    size_t offset;
    size_t max_align;
} arena_cursor;

typedef struct {
    uint32_t *owner_idx;
    velri_device_id *owner_dev;
    uint32_t *valid_copy; /* [num_user_storages * VELRI_MAX_DEVICES] */
} coherence_state;

/* Short-horizon offloaded WEIGHT shadow cache: deduplicates intra-op subviews and consecutive
 * weighted ops on the same device (bridging at most 1 weightless op such as ATTENTION between
 * q_abs and v_proj on fused qkv_w), without pinning non-adjacent tied weights (wte / lm_head). */
typedef struct {
    velri_device_id prev_dev;
    uint32_t prev_cnt;
    uint32_t prev_skip;
    velri_storage_id prev_sids[VELRI_OP_MAX_INPUTS];
    uint32_t prev_idxs[VELRI_OP_MAX_INPUTS];
} weight_shadow_cache;

typedef struct {
    uint32_t cnt;
    velri_storage_id sids[VELRI_OP_MAX_INPUTS];
    uint32_t idxs[VELRI_OP_MAX_INPUTS];
} op_weight_shadows;

typedef struct {
    size_t start;
    size_t end;
    uint32_t first_step;
    uint32_t last_step;
} placed_interval;

static size_t effective_align(const velri_context *ctx, velri_device_id d, size_t req_align) {
    size_t min_a = ctx->devices[d].caps.min_align_bytes;
    if (min_a < 64U) {
        min_a = 64U;
    }
    return req_align < min_a ? min_a : req_align;
}

static velri_status bump_arena(arena_cursor *cur, size_t byte_size, size_t align,
                               size_t *out_offset) {
    size_t aligned_off = 0U;
    size_t next_off = 0U;
    if (velri_align_up_overflow(cur->offset, align, &aligned_off) ||
        velri_add_overflow_size(aligned_off, byte_size, &next_off)) {
        return VELRI_ERR_OVERFLOW;
    }
    if (align > cur->max_align) {
        cur->max_align = align;
    }
    cur->offset = next_off;
    *out_offset = aligned_off;
    return VELRI_OK;
}

static velri_status finalize_arena(const arena_cursor *cur, size_t *out_bytes, size_t *out_align) {
    size_t total = 0U;
    if (cur->offset > 0U && velri_align_up_overflow(cur->offset, cur->max_align, &total)) {
        return VELRI_ERR_OVERFLOW;
    }
    *out_bytes = total;
    *out_align = cur->max_align;
    return VELRI_OK;
}

static velri_status add_shadow_storage(plan_builder *pb, velri_device_id dst_dev,
                                       velri_dtype dtype, size_t byte_size, size_t align,
                                       uint32_t *out_idx) {
    void *buf = nullptr;
    if (!velri_vec_ensure(pb->plan->storages, &buf, &pb->storage_cap, pb->plan->num_storages, 1U,
                          sizeof(velri_plan_storage))) {
        return VELRI_ERR_NOMEM;
    }
    pb->plan->storages = (velri_plan_storage *)buf;

    uint32_t idx = pb->plan->num_storages++;
    velri_plan_storage *ps = &pb->plan->storages[idx];
    memset(ps, 0, sizeof(*ps));
    ps->kind = VELRI_STORAGE_SCRATCH;
    ps->dtype = dtype;
    ps->device_id = dst_dev;
    ps->byte_size = byte_size;
    ps->align = align;
    *out_idx = idx;
    return VELRI_OK;
}

static velri_status emit_ram_transfer(plan_builder *pb, velri_device_id src_dev, uint32_t src_idx,
                                      velri_device_id dst_dev, uint32_t dst_idx, size_t byte_size) {
    void *buf = nullptr;
    if (!velri_vec_ensure(pb->plan->steps, &buf, &pb->step_cap, pb->plan->num_steps, 1U,
                          sizeof(velri_launch_step))) {
        return VELRI_ERR_NOMEM;
    }
    pb->plan->steps = (velri_launch_step *)buf;

    velri_launch_step *step = &pb->plan->steps[pb->plan->num_steps++];
    memset(step, 0, sizeof(*step));
    step->kind = VELRI_STEP_RAM_TRANSFER;
    step->u.transfer.src_device_id = src_dev;
    step->u.transfer.dst_device_id = dst_dev;
    step->u.transfer.src_plan_storage_idx = src_idx;
    step->u.transfer.dst_plan_storage_idx = dst_idx;
    step->u.transfer.byte_size = byte_size;

    const velri_context *ctx = pb->plan->ctx;
    bool src_host = ctx->devices[src_dev].caps.is_host_memory;
    bool dst_host = ctx->devices[dst_dev].caps.is_host_memory;
    if (!src_host && !dst_host && byte_size > pb->plan->ram_staging_bytes) {
        pb->plan->ram_staging_bytes = byte_size;
    }
    return VELRI_OK;
}

static velri_status select_node_device(velri_context *ctx, const velri_resolved_op *resolved,
                                       velri_device_id preferred_dev, bool allow_cpu_fallback,
                                       velri_device_id *out_dev) {
    if (preferred_dev >= ctx->num_devices) {
        return VELRI_ERR_NOT_FOUND;
    }
    const velri_device_entry *entry = &ctx->devices[preferred_dev];
    if (entry->vt->query_operation(entry->dev, resolved) == VELRI_OK) {
        *out_dev = preferred_dev;
        return VELRI_OK;
    }
    if (!allow_cpu_fallback) {
        return VELRI_ERR_UNSUPPORTED;
    }
    /* Scan in device-id order so host-memory device 0 is preferred without hardcoding backend names. */
    for (uint32_t d = 0U; d < ctx->num_devices; ++d) {
        if (d == preferred_dev || !ctx->devices[d].caps.is_host_memory) {
            continue;
        }
        const velri_device_entry *he = &ctx->devices[d];
        if (he->vt->query_operation(he->dev, resolved) == VELRI_OK) {
            *out_dev = d;
            return VELRI_OK;
        }
    }
    return VELRI_ERR_UNSUPPORTED;
}

static velri_status pack_weight_slabs(velri_plan *plan, bool allocate_slabs) {
    velri_context *ctx = plan->ctx;
    for (uint32_t d = 0U; d < ctx->num_devices; ++d) {
        if (ctx->devices[d].caps.is_host_memory) {
            continue;
        }
        uint32_t cur_slab = 0U;
        arena_cursor cur = {.offset = 0U, .max_align = effective_align(ctx, d, 0U)};

        for (uint32_t i = 0U; i < plan->num_user_storages; ++i) {
            velri_plan_storage *ps = &plan->storages[i];
            if (ps->kind != VELRI_STORAGE_WEIGHT || ps->device_id != d) {
                continue;
            }
            size_t align = effective_align(ctx, d, ps->align);
            size_t peek_off = 0U, peek_end = 0U;
            if (velri_align_up_overflow(cur.offset, align, &peek_off) ||
                velri_add_overflow_size(peek_off, ps->byte_size, &peek_end)) {
                return VELRI_ERR_OVERFLOW;
            }
            if (cur.offset > 0U && peek_end > VELRI_WEIGHT_SLAB_LIMIT) {
                velri_status st = finalize_arena(&cur, &plan->weight_slab_bytes[d][cur_slab],
                                                 &plan->weight_slab_align[d][cur_slab]);
                if (st != VELRI_OK) {
                    return st;
                }
                cur_slab++;
                if (cur_slab >= VELRI_MAX_WEIGHT_SLABS) {
                    return VELRI_ERR_NOMEM;
                }
                cur.offset = 0U;
                cur.max_align = effective_align(ctx, d, 0U);
            }

            size_t placed_off = 0U;
            velri_status st = bump_arena(&cur, ps->byte_size, align, &placed_off);
            if (st != VELRI_OK) {
                return st;
            }
            ps->arena_kind = VELRI_ARENA_WEIGHT;
            ps->slab_idx = cur_slab;
            ps->arena_offset = placed_off;
        }

        if (cur.offset > 0U) {
            velri_status st = finalize_arena(&cur, &plan->weight_slab_bytes[d][cur_slab],
                                             &plan->weight_slab_align[d][cur_slab]);
            if (st != VELRI_OK) {
                return st;
            }
            plan->num_weight_slabs[d] = cur_slab + 1U;
        }

        if (allocate_slabs) {
            for (uint32_t s = 0U; s < plan->num_weight_slabs[d]; ++s) {
                velri_status st = velri_device_alloc_tracked(
                    ctx, d, plan->weight_slab_bytes[d][s], plan->weight_slab_align[d][s],
                    VELRI_ARENA_WEIGHT, &plan->weight_slabs[d][s]);
                if (st != VELRI_OK) {
                    return st;
                }
            }
        }
    }
    return VELRI_OK;
}

static velri_status bind_shared_weights_from_owner(velri_plan *plan, velri_plan *owner) {
    if (!plan || !owner || plan->ctx != owner->ctx) {
        return VELRI_ERR_INVALID;
    }
    if (plan->num_user_storages > 0U) {
        size_t map_bytes = 0U;
        if (velri_mul_overflow_size((size_t)plan->num_user_storages, sizeof(uint32_t), &map_bytes)) {
            return VELRI_ERR_OVERFLOW;
        }
        plan->weight_peer_sid = (uint32_t *)malloc(map_bytes);
        if (!plan->weight_peer_sid) {
            return VELRI_ERR_NOMEM;
        }
        for (uint32_t i = 0U; i < plan->num_user_storages; ++i) {
            plan->weight_peer_sid[i] = UINT32_MAX;
        }
    }

    for (uint32_t d = 0U; d < plan->ctx->num_devices; ++d) {
        if (plan->num_weight_slabs[d] != owner->num_weight_slabs[d]) {
            return VELRI_ERR_INVALID;
        }
        for (uint32_t s = 0U; s < plan->num_weight_slabs[d]; ++s) {
            if (plan->weight_slab_bytes[d][s] != owner->weight_slab_bytes[d][s] ||
                plan->weight_slab_align[d][s] > owner->weight_slab_align[d][s]) {
                return VELRI_ERR_INVALID;
            }
        }

        uint32_t j = 0U;
        for (uint32_t i = 0U; i < plan->num_user_storages; ++i) {
            const velri_plan_storage *ps = &plan->storages[i];
            if (ps->kind != VELRI_STORAGE_WEIGHT || ps->device_id != d) {
                continue;
            }
            while (j < owner->num_user_storages &&
                   (owner->storages[j].kind != VELRI_STORAGE_WEIGHT ||
                    owner->storages[j].device_id != d)) {
                j++;
            }
            if (j >= owner->num_user_storages) {
                return VELRI_ERR_INVALID;
            }
            const velri_plan_storage *ops = &owner->storages[j];
            if (ps->dtype != ops->dtype || ps->byte_size != ops->byte_size ||
                ps->arena_kind != ops->arena_kind || ps->slab_idx != ops->slab_idx ||
                ps->arena_offset != ops->arena_offset) {
                return VELRI_ERR_INVALID;
            }
            plan->weight_peer_sid[i] = j;
            j++;
        }
        while (j < owner->num_user_storages) {
            if (owner->storages[j].kind == VELRI_STORAGE_WEIGHT &&
                owner->storages[j].device_id == d) {
                return VELRI_ERR_INVALID;
            }
            j++;
        }
    }

    velri_plan_retain(owner);
    plan->weight_owner = owner;
    for (uint32_t d = 0U; d < plan->ctx->num_devices; ++d) {
        for (uint32_t s = 0U; s < plan->num_weight_slabs[d]; ++s) {
            plan->weight_slabs[d][s] = owner->weight_slabs[d][s];
        }
    }
    return VELRI_OK;
}

static velri_status pack_state_and_io_arenas(velri_plan *plan) {
    velri_context *ctx = plan->ctx;
    for (uint32_t d = 0U; d < ctx->num_devices; ++d) {
        bool is_host = ctx->devices[d].caps.is_host_memory;
        arena_cursor st_cur = {.offset = 0U, .max_align = effective_align(ctx, d, 0U)};
        arena_cursor io_cur = {.offset = 0U, .max_align = effective_align(ctx, d, 0U)};

        for (uint32_t i = 0U; i < plan->num_user_storages; ++i) {
            velri_plan_storage *ps = &plan->storages[i];
            if (ps->device_id != d) {
                continue;
            }
            if (ps->kind == VELRI_STORAGE_STATE) {
                size_t align = effective_align(ctx, d, ps->align);
                velri_status st = bump_arena(&st_cur, ps->byte_size, align, &ps->arena_offset);
                if (st != VELRI_OK) {
                    return st;
                }
                ps->arena_kind = VELRI_ARENA_STATE;
            } else if (!is_host &&
                       (ps->kind == VELRI_STORAGE_INPUT || ps->kind == VELRI_STORAGE_OUTPUT)) {
                size_t align = effective_align(ctx, d, ps->align);
                velri_status st = bump_arena(&io_cur, ps->byte_size, align, &ps->arena_offset);
                if (st != VELRI_OK) {
                    return st;
                }
                ps->arena_kind = VELRI_ARENA_IO;
            }
        }

        velri_status st = finalize_arena(&st_cur,
                                         &plan->slot_arena_bytes[VELRI_SLOT_ARENA_STATE][d],
                                         &plan->slot_arena_align[VELRI_SLOT_ARENA_STATE][d]);
        if (st != VELRI_OK) {
            return st;
        }
        st = finalize_arena(&io_cur, &plan->slot_arena_bytes[VELRI_SLOT_ARENA_IO][d],
                            &plan->slot_arena_align[VELRI_SLOT_ARENA_IO][d]);
        if (st != VELRI_OK) {
            return st;
        }
    }
    return VELRI_OK;
}

static inline uint32_t *coherence_slot(coherence_state *cs, velri_storage_id sid,
                                       velri_device_id dev) {
    return &cs->valid_copy[(size_t)sid * VELRI_MAX_DEVICES + dev];
}

static velri_status coherence_state_init(coherence_state *cs, const velri_graph *graph,
                                         velri_plan *plan) {
    memset(cs, 0, sizeof(*cs));
    size_t n_user = graph->num_storages ? (size_t)graph->num_storages : 1U;
    size_t owner_idx_bytes = 0U, owner_dev_bytes = 0U, vc_count = 0U, vc_bytes = 0U;
    if (velri_mul_overflow_size(n_user, sizeof(uint32_t), &owner_idx_bytes) ||
        velri_mul_overflow_size(n_user, sizeof(velri_device_id), &owner_dev_bytes) ||
        velri_mul_overflow_size(n_user, (size_t)VELRI_MAX_DEVICES, &vc_count) ||
        velri_mul_overflow_size(vc_count, sizeof(uint32_t), &vc_bytes)) {
        return VELRI_ERR_OVERFLOW;
    }

    cs->owner_idx = (uint32_t *)malloc(owner_idx_bytes);
    cs->owner_dev = (velri_device_id *)malloc(owner_dev_bytes);
    cs->valid_copy = (uint32_t *)malloc(vc_bytes);
    if (!cs->owner_idx || !cs->owner_dev || !cs->valid_copy) {
        free(cs->owner_idx);
        free(cs->owner_dev);
        free(cs->valid_copy);
        memset(cs, 0, sizeof(*cs));
        return VELRI_ERR_NOMEM;
    }

    for (uint32_t i = 0U; i < graph->num_storages; ++i) {
        const velri_storage_desc *sd = &graph->storages[i];
        velri_plan_storage *ps = &plan->storages[i];
        memset(ps, 0, sizeof(*ps));
        ps->kind = sd->kind;
        ps->dtype = sd->dtype;
        ps->device_id = sd->device_id;
        ps->byte_size = sd->byte_size;
        ps->align = sd->align;
        cs->owner_idx[i] = i;
        cs->owner_dev[i] = sd->device_id;
        for (uint32_t d = 0U; d < VELRI_MAX_DEVICES; ++d) {
            *coherence_slot(cs, i, d) = UINT32_MAX;
        }
        *coherence_slot(cs, i, sd->device_id) = i;
    }
    plan->num_storages = graph->num_storages;
    return VELRI_OK;
}

static void coherence_state_destroy(coherence_state *cs) {
    free(cs->owner_idx);
    free(cs->owner_dev);
    free(cs->valid_copy);
    memset(cs, 0, sizeof(*cs));
}

static velri_status lower_resolve_inputs(plan_builder *pb, coherence_state *cs,
                                         const weight_shadow_cache *wcache,
                                         const velri_resolved_op *resolved,
                                         velri_device_id chosen_dev,
                                         uint32_t out_in_idx[VELRI_OP_MAX_INPUTS],
                                         op_weight_shadows *cur_w) {
    memset(cur_w, 0, sizeof(*cur_w));
    for (uint32_t i = 0U; i < VELRI_OP_MAX_INPUTS; ++i) {
        out_in_idx[i] = UINT32_MAX;
    }

    velri_plan *plan = pb->plan;
    for (uint32_t i = 0U; i < resolved->num_inputs; ++i) {
        if (!resolved->input_present[i]) {
            continue;
        }
        velri_storage_id sid = resolved->inputs[i].storage_id;
        uint32_t use_idx = *coherence_slot(cs, sid, chosen_dev);
        if (use_idx == UINT32_MAX && plan->storages[sid].kind == VELRI_STORAGE_WEIGHT) {
            for (uint32_t k = 0U; k < cur_w->cnt; ++k) {
                if (cur_w->sids[k] == sid) {
                    use_idx = cur_w->idxs[k];
                    break;
                }
            }
            if (use_idx == UINT32_MAX && wcache->prev_dev == chosen_dev) {
                for (uint32_t k = 0U; k < wcache->prev_cnt; ++k) {
                    if (wcache->prev_sids[k] == sid) {
                        use_idx = wcache->prev_idxs[k];
                        cur_w->sids[cur_w->cnt] = sid;
                        cur_w->idxs[cur_w->cnt] = use_idx;
                        cur_w->cnt++;
                        break;
                    }
                }
            }
        }
        if (use_idx == UINT32_MAX) {
            velri_status st = add_shadow_storage(pb, chosen_dev, plan->storages[sid].dtype,
                                                 plan->storages[sid].byte_size,
                                                 plan->storages[sid].align, &use_idx);
            if (st != VELRI_OK) {
                return st;
            }
            st = emit_ram_transfer(pb, cs->owner_dev[sid], cs->owner_idx[sid], chosen_dev, use_idx,
                                   plan->storages[sid].byte_size);
            if (st != VELRI_OK) {
                return st;
            }
            if (plan->storages[sid].kind != VELRI_STORAGE_WEIGHT) {
                *coherence_slot(cs, sid, chosen_dev) = use_idx;
            } else {
                cur_w->sids[cur_w->cnt] = sid;
                cur_w->idxs[cur_w->cnt] = use_idx;
                cur_w->cnt++;
            }
        }
        out_in_idx[i] = use_idx;
    }
    return VELRI_OK;
}

/*
 * Prepare output targets on `chosen_dev` (preserving untouched bytes on RMW/subviews).
 *
 * Shared-KV-cache output-dedup invariant:
 *   Multiple output views of one op may legally alias the same underlying `storage_id`
 *   (specifically MLA / shared-KV ATTENTION where `outputs[1]` (K cache) and `outputs[2]`
 *   (V cache) reference the same `VELRI_STORAGE_STATE`). All outputs targeting the same
 *   `storage_id` MUST resolve to the identical `plan_storage_idx` on `chosen_dev` so they
 *   share a single device buffer (and shadow transfer when offloaded) rather than splitting
 *   into diverging copies when coherence ownership is updated.
 */
static velri_status lower_resolve_outputs(plan_builder *pb, coherence_state *cs,
                                          const velri_resolved_op *resolved,
                                          velri_device_id chosen_dev,
                                          velri_storage_id out_sids[VELRI_OP_MAX_OUTPUTS],
                                          uint32_t out_dsts[VELRI_OP_MAX_OUTPUTS]) {
    velri_plan *plan = pb->plan;
    for (uint32_t o = 0U; o < VELRI_OP_MAX_OUTPUTS; ++o) {
        out_sids[o] = VELRI_INVALID_ID;
        out_dsts[o] = UINT32_MAX;
    }

    for (uint32_t o = 0U; o < resolved->num_outputs; ++o) {
        velri_storage_id sid = resolved->outputs[o].storage_id;
        out_sids[o] = sid;

        uint32_t dst_idx = UINT32_MAX;
        for (uint32_t prev = 0U; prev < o; ++prev) {
            if (out_sids[prev] == sid) {
                dst_idx = out_dsts[prev];
                break;
            }
        }

        if (dst_idx == UINT32_MAX) {
            bool need_preserve = false;
            for (uint32_t k = o; k < resolved->num_outputs; ++k) {
                if (resolved->outputs[k].storage_id == sid &&
                    (velri_op_output_is_rmw(resolved->kind, k) ||
                     !velri_tensor_view_covers_storage(&resolved->outputs[k],
                                                       plan->storages[sid].byte_size))) {
                    need_preserve = true;
                    break;
                }
            }
            velri_device_id home_dev = plan->storages[sid].device_id;
            uint32_t existing = *coherence_slot(cs, sid, chosen_dev);

            if (existing != UINT32_MAX) {
                dst_idx = existing;
            } else {
                if (home_dev == chosen_dev) {
                    dst_idx = sid;
                } else {
                    velri_status st = add_shadow_storage(pb, chosen_dev, plan->storages[sid].dtype,
                                                         plan->storages[sid].byte_size,
                                                         plan->storages[sid].align, &dst_idx);
                    if (st != VELRI_OK) {
                        return st;
                    }
                }
                if (need_preserve) {
                    velri_status st = emit_ram_transfer(pb, cs->owner_dev[sid], cs->owner_idx[sid],
                                                        chosen_dev, dst_idx,
                                                        plan->storages[sid].byte_size);
                    if (st != VELRI_OK) {
                        return st;
                    }
                }
            }
        }

        out_dsts[o] = dst_idx;
    }
    return VELRI_OK;
}

static void weight_shadow_cache_advance(weight_shadow_cache *wcache, velri_device_id chosen_dev,
                                        const op_weight_shadows *cur_w) {
    if (cur_w->cnt > 0U) {
        wcache->prev_dev = chosen_dev;
        wcache->prev_cnt = cur_w->cnt;
        wcache->prev_skip = 0U;
        for (uint32_t k = 0U; k < cur_w->cnt; ++k) {
            wcache->prev_sids[k] = cur_w->sids[k];
            wcache->prev_idxs[k] = cur_w->idxs[k];
        }
    } else if (wcache->prev_dev == chosen_dev && wcache->prev_cnt > 0U && wcache->prev_skip == 0U) {
        /* Bridge at most one weightless op on the same device (e.g. ATTENTION between
         * batched_linear(W_K) and batched_linear(W_V) sharing fused qkv_w). */
        wcache->prev_skip = 1U;
    } else {
        wcache->prev_dev = VELRI_INVALID_ID;
        wcache->prev_cnt = 0U;
        wcache->prev_skip = 0U;
    }
}

static velri_status maybe_flush_offloaded_storage(plan_builder *pb, coherence_state *cs,
                                                  velri_storage_id sid, uint32_t node_idx,
                                                  const uint32_t *last_node_for_storage) {
    velri_plan *plan = pb->plan;
    if (!last_node_for_storage || sid >= plan->num_user_storages ||
        last_node_for_storage[sid] != node_idx) {
        return VELRI_OK;
    }
    velri_storage_kind kind = plan->storages[sid].kind;
    if (kind != VELRI_STORAGE_STATE && kind != VELRI_STORAGE_OUTPUT) {
        return VELRI_OK;
    }
    velri_device_id home_dev = plan->storages[sid].device_id;
    if (cs->owner_dev[sid] == home_dev) {
        return VELRI_OK;
    }
    velri_status st = emit_ram_transfer(pb, cs->owner_dev[sid], cs->owner_idx[sid], home_dev, sid,
                                        plan->storages[sid].byte_size);
    if (st != VELRI_OK) {
        return st;
    }
    for (uint32_t d = 0U; d < VELRI_MAX_DEVICES; ++d) {
        *coherence_slot(cs, sid, d) = UINT32_MAX;
    }
    *coherence_slot(cs, sid, home_dev) = sid;
    cs->owner_dev[sid] = home_dev;
    cs->owner_idx[sid] = sid;
    return VELRI_OK;
}

static velri_status lower_graph_node(plan_builder *pb, const velri_graph *graph,
                                     const velri_graph_node *node, uint32_t node_idx,
                                     const uint32_t *last_node_for_storage,
                                     bool allow_cpu_fallback, coherence_state *cs,
                                     weight_shadow_cache *wcache) {
    velri_plan *plan = pb->plan;
    velri_context *ctx = plan->ctx;

    velri_resolved_op resolved;
    velri_graph_resolve_node(graph, &node->desc, &resolved);

    velri_device_id chosen_dev = 0U;
    velri_status st =
        select_node_device(ctx, &resolved, node->device_id, allow_cpu_fallback, &chosen_dev);
    if (st != VELRI_OK) {
        return st;
    }

    uint32_t in_storage_idx[VELRI_OP_MAX_INPUTS];
    velri_storage_id out_sids[VELRI_OP_MAX_OUTPUTS];
    uint32_t out_storage_idx[VELRI_OP_MAX_OUTPUTS];
    op_weight_shadows cur_w;

    st = lower_resolve_inputs(pb, cs, wcache, &resolved, chosen_dev, in_storage_idx, &cur_w);
    if (st != VELRI_OK) {
        return st;
    }

    st = lower_resolve_outputs(pb, cs, &resolved, chosen_dev, out_sids, out_storage_idx);
    if (st != VELRI_OK) {
        return st;
    }

    /* Invalidate stale copies on other devices now that all inputs/outputs are resolved. */
    for (uint32_t o = 0U; o < resolved.num_outputs; ++o) {
        velri_storage_id sid = out_sids[o];
        for (uint32_t d = 0U; d < VELRI_MAX_DEVICES; ++d) {
            *coherence_slot(cs, sid, d) = UINT32_MAX;
        }
        *coherence_slot(cs, sid, chosen_dev) = out_storage_idx[o];
        cs->owner_dev[sid] = chosen_dev;
        cs->owner_idx[sid] = out_storage_idx[o];
    }

    weight_shadow_cache_advance(wcache, chosen_dev, &cur_w);

    /* Optional backend preparation hook. */
    velri_device_entry *de = &ctx->devices[chosen_dev];
    velri_prepared_op *prepared = nullptr;
    if (de->vt->prepare_op) {
        st = de->vt->prepare_op(de->dev, &resolved, &prepared);
        if (st != VELRI_OK) {
            return st;
        }
    }

    void *buf = nullptr;
    if (!velri_vec_ensure(plan->steps, &buf, &pb->step_cap, plan->num_steps, 1U,
                          sizeof(velri_launch_step))) {
        if (prepared && de->vt->destroy_prepared_op) {
            de->vt->destroy_prepared_op(de->dev, prepared);
        }
        return VELRI_ERR_NOMEM;
    }
    plan->steps = (velri_launch_step *)buf;

    velri_launch_step *step = &plan->steps[plan->num_steps++];
    memset(step, 0, sizeof(*step));
    step->kind = VELRI_STEP_OP;
    step->u.op.device_id = chosen_dev;
    step->u.op.resolved = resolved;
    step->u.op.prepared = prepared;
    memcpy(step->u.op.in_storage_idx, in_storage_idx, sizeof(in_storage_idx));
    memcpy(step->u.op.out_storage_idx, out_storage_idx, sizeof(out_storage_idx));

    /* Flush offloaded STATE/OUTPUT immediately on their last graph use so GPU shadows
     * can be reused across subsequent layers (e.g. N-layer offloaded KV cache). */
    for (uint32_t o = 0U; o < resolved.num_outputs; ++o) {
        st = maybe_flush_offloaded_storage(pb, cs, out_sids[o], node_idx, last_node_for_storage);
        if (st != VELRI_OK) {
            return st;
        }
    }
    for (uint32_t i = 0U; i < resolved.num_inputs; ++i) {
        if (!resolved.input_present[i]) {
            continue;
        }
        st = maybe_flush_offloaded_storage(pb, cs, resolved.inputs[i].storage_id, node_idx,
                                           last_node_for_storage);
        if (st != VELRI_OK) {
            return st;
        }
    }
    return VELRI_OK;
}

static inline void touch_lifetime(uint32_t idx, uint32_t step, uint32_t *first_use,
                                  uint32_t *last_use) {
    if (step < first_use[idx]) first_use[idx] = step;
    if (step > last_use[idx]) last_use[idx] = step;
}

static void compute_storage_lifetimes(const velri_plan *plan, uint32_t *first_use,
                                      uint32_t *last_use) {
    for (uint32_t i = 0U; i < plan->num_storages; ++i) {
        first_use[i] = UINT32_MAX;
        last_use[i] = 0U;
    }
    for (uint32_t s = 0U; s < plan->num_steps; ++s) {
        const velri_launch_step *step = &plan->steps[s];
        if (step->kind == VELRI_STEP_OP) {
            const velri_resolved_op *op = &step->u.op.resolved;
            for (uint32_t i = 0U; i < op->num_inputs; ++i) {
                if (!op->input_present[i]) {
                    continue;
                }
                touch_lifetime(step->u.op.in_storage_idx[i], s, first_use, last_use);
            }
            for (uint32_t o = 0U; o < op->num_outputs; ++o) {
                touch_lifetime(step->u.op.out_storage_idx[o], s, first_use, last_use);
            }
        } else {
            touch_lifetime(step->u.transfer.src_plan_storage_idx, s, first_use, last_use);
            touch_lifetime(step->u.transfer.dst_plan_storage_idx, s, first_use, last_use);
        }
    }
}

/* Sort descending by byte_size, then descending by lifetime span for tighter packing. */
static void sort_scratch_candidates(const velri_plan *plan, const uint32_t *first_use,
                                    const uint32_t *last_use, uint32_t *order, uint32_t n_cand) {
    for (uint32_t a = 1U; a < n_cand; ++a) {
        uint32_t key = order[a];
        size_t key_sz = plan->storages[key].byte_size;
        uint32_t key_span = last_use[key] - first_use[key];
        uint32_t b = a;
        while (b > 0U) {
            uint32_t prev = order[b - 1U];
            size_t prev_sz = plan->storages[prev].byte_size;
            uint32_t prev_span = last_use[prev] - first_use[prev];
            if (prev_sz > key_sz || (prev_sz == key_sz && prev_span >= key_span)) {
                break;
            }
            order[b] = prev;
            b--;
        }
        order[b] = key;
    }
}

static velri_status pack_device_scratch_arena(velri_plan *plan, velri_device_id d,
                                              const uint32_t *first_use, const uint32_t *last_use,
                                              uint32_t *order, placed_interval *placed) {
    uint32_t n_cand = 0U;
    for (uint32_t i = 0U; i < plan->num_storages; ++i) {
        const velri_plan_storage *ps = &plan->storages[i];
        if (ps->kind == VELRI_STORAGE_SCRATCH && ps->device_id == d &&
            first_use[i] != UINT32_MAX) {
            order[n_cand++] = i;
        }
    }

    sort_scratch_candidates(plan, first_use, last_use, order, n_cand);

    uint32_t n_placed = 0U;
    size_t base_align = effective_align(plan->ctx, d, 0U);
    size_t max_align = base_align;
    size_t arena_size = 0U;

    for (uint32_t c = 0U; c < n_cand; ++c) {
        uint32_t i = order[c];
        velri_plan_storage *ps = &plan->storages[i];

        size_t align = ps->align < base_align ? base_align : ps->align;
        if (align > max_align) {
            max_align = align;
        }
        size_t sz = 0U;
        if (velri_align_up_overflow(ps->byte_size, align, &sz)) {
            return VELRI_ERR_OVERFLOW;
        }

        size_t cand = 0U;
        size_t cand_end = 0U;
        bool conflict = true;
        while (conflict) {
            if (velri_add_overflow_size(cand, sz, &cand_end)) {
                return VELRI_ERR_OVERFLOW;
            }
            conflict = false;
            for (uint32_t p = 0U; p < n_placed; ++p) {
                bool time_overlap =
                    !(last_use[i] < placed[p].first_step || first_use[i] > placed[p].last_step);
                bool addr_overlap = (cand < placed[p].end) && (cand_end > placed[p].start);
                if (time_overlap && addr_overlap) {
                    if (velri_align_up_overflow(placed[p].end, align, &cand)) {
                        return VELRI_ERR_OVERFLOW;
                    }
                    conflict = true;
                    break;
                }
            }
        }

        ps->arena_kind = VELRI_ARENA_SCRATCH;
        ps->arena_offset = cand;
        placed[n_placed++] = (placed_interval){
            .start = cand,
            .end = cand_end,
            .first_step = first_use[i],
            .last_step = last_use[i],
        };
        if (cand_end > arena_size) {
            arena_size = cand_end;
        }
    }

    if (arena_size > 0U && velri_align_up_overflow(arena_size, max_align, &arena_size)) {
        return VELRI_ERR_OVERFLOW;
    }
    plan->slot_arena_bytes[VELRI_SLOT_ARENA_SCRATCH][d] = arena_size;
    plan->slot_arena_align[VELRI_SLOT_ARENA_SCRATCH][d] = max_align;
    return VELRI_OK;
}

static velri_status pack_scratch_arenas(velri_plan *plan) {
    uint32_t ns = plan->num_storages;
    if (ns == 0U) {
        return VELRI_OK;
    }

    size_t u32_bytes = 0U, pi_bytes = 0U;
    if (velri_mul_overflow_size((size_t)ns, sizeof(uint32_t), &u32_bytes) ||
        velri_mul_overflow_size((size_t)ns, sizeof(placed_interval), &pi_bytes)) {
        return VELRI_ERR_OVERFLOW;
    }

    uint32_t *first_use = (uint32_t *)malloc(u32_bytes);
    uint32_t *last_use = (uint32_t *)malloc(u32_bytes);
    uint32_t *order = (uint32_t *)malloc(u32_bytes);
    placed_interval *placed = (placed_interval *)malloc(pi_bytes);
    if (!first_use || !last_use || !order || !placed) {
        free(first_use);
        free(last_use);
        free(order);
        free(placed);
        return VELRI_ERR_NOMEM;
    }

    compute_storage_lifetimes(plan, first_use, last_use);

    velri_status st = VELRI_OK;
    for (uint32_t d = 0U; d < plan->ctx->num_devices && st == VELRI_OK; ++d) {
        st = pack_device_scratch_arena(plan, d, first_use, last_use, order, placed);
    }

    free(first_use);
    free(last_use);
    free(order);
    free(placed);
    return st;
}

typedef struct {
    velri_device_id device_id;
    velri_arena_kind arena_kind;
    uint32_t owner_key; /* 0 for shared slot arenas; storage_idx for VELRI_ARENA_NONE */
    size_t start;
    size_t end;
} phys_span;

static bool phys_span_overlaps(const phys_span *a, const phys_span *b) {
    return a->device_id == b->device_id && a->arena_kind == b->arena_kind &&
           a->owner_key == b->owner_key && a->start < b->end && b->start < a->end;
}

static velri_status compute_operand_span(const velri_plan *plan, uint32_t storage_idx,
                                         const velri_tensor_view *view, phys_span *out,
                                         bool *out_valid) {
    *out_valid = false;
    const velri_plan_storage *ps = &plan->storages[storage_idx];
    if (ps->kind == VELRI_STORAGE_WEIGHT) {
        return VELRI_OK; /* Immutable weights never participate in RAW/WAR/WAW hazards. */
    }
    size_t vs = 0U, ve = ps->byte_size;
    if (velri_tensor_view_byte_span(view, &vs, &ve) != VELRI_OK) {
        vs = 0U;
        ve = ps->byte_size;
    }
    size_t base = (ps->arena_kind != VELRI_ARENA_NONE) ? ps->arena_offset : 0U;
    size_t s0 = 0U, s1 = 0U;
    if (velri_add_overflow_size(base, vs, &s0) || velri_add_overflow_size(base, ve, &s1)) {
        return VELRI_ERR_OVERFLOW;
    }
    out->device_id = ps->device_id;
    out->arena_kind = ps->arena_kind;
    out->owner_key = (ps->arena_kind == VELRI_ARENA_NONE) ? storage_idx : 0U;
    out->start = s0;
    out->end = s1;
    *out_valid = (s0 < s1);
    return VELRI_OK;
}

static bool span_conflicts_list(const phys_span *sp, const phys_span *list, uint32_t count) {
    for (uint32_t i = 0U; i < count; ++i) {
        if (phys_span_overlaps(sp, &list[i])) {
            return true;
        }
    }
    return false;
}

/*
 * Post-packing DAG Hazard & Batch-Length Analyzer:
 *   Runs AFTER `pack_scratch_arenas` so packed scratch reuse (WAR/WAW on shared arena bytes)
 *   is accounted for exactly. Within each contiguous same-device OP batch, tracks physical
 *   read/write spans since the last barrier and marks hazard-free ops with
 *   `no_barrier_before = true` so GPU backends can overlap independent dispatches.
 */
static velri_status analyze_dag_barriers_and_batches(velri_plan *plan) {
    plan->max_batch_len = 0U;
    plan->dag_total_ops = 0U;
    plan->dag_elided_barriers = 0U;
    if (plan->num_steps == 0U) {
        return VELRI_OK;
    }

    phys_span *reads = nullptr;
    phys_span *writes = nullptr;
    uint32_t r_cnt = 0U, r_cap = 0U;
    uint32_t w_cnt = 0U, w_cap = 0U;
    velri_status st = VELRI_OK;

    uint32_t s = 0U;
    while (s < plan->num_steps) {
        if (plan->steps[s].kind != VELRI_STEP_OP) {
            s++;
            continue;
        }
        velri_device_id dev = plan->steps[s].u.op.device_id;
        uint32_t run_end = s + 1U;
        while (run_end < plan->num_steps && plan->steps[run_end].kind == VELRI_STEP_OP &&
               plan->steps[run_end].u.op.device_id == dev) {
            run_end++;
        }
        const velri_backend_vtable *vt = plan->ctx->devices[dev].vt;
        uint32_t batch_len = vt->submit_batch ? run_end - s : 1U;
        if (batch_len > plan->max_batch_len) {
            plan->max_batch_len = batch_len;
        }

        r_cnt = 0U;
        w_cnt = 0U;
        velri_launch_batch *batch = nullptr;
        for (uint32_t idx = s; idx < run_end; ++idx) {
            velri_launch_step *step = &plan->steps[idx];
            const velri_resolved_op *op = &step->u.op.resolved;
            plan->dag_total_ops++;
            if (idx == s || batch_len == 1U) {
                batch = &step->u.op.batch;
                batch->count = batch_len;
                batch->static_t1 = true;
                if (vt->capture_graph) {
                    batch->capture_index = plan->num_capture_batches++;
                }
            }
            if (velri_op_is_dynamic_seq(op) && velri_op_static_rows(op) != 1U) {
                batch->static_t1 = false;
            }

            phys_span op_r[VELRI_OP_MAX_INPUTS];
            phys_span op_w[VELRI_OP_MAX_OUTPUTS];
            uint32_t op_rc = 0U, op_wc = 0U;
            for (uint32_t i = 0U; i < op->num_inputs; ++i) {
                if (!op->input_present[i]) {
                    continue;
                }
                bool valid_span = false;
                st = compute_operand_span(plan, step->u.op.in_storage_idx[i], &op->inputs[i],
                                          &op_r[op_rc], &valid_span);
                if (st != VELRI_OK) {
                    goto done;
                }
                if (valid_span) {
                    op_rc++;
                }
            }
            for (uint32_t o = 0U; o < op->num_outputs; ++o) {
                bool valid_span = false;
                st = compute_operand_span(plan, step->u.op.out_storage_idx[o], &op->outputs[o],
                                          &op_w[op_wc], &valid_span);
                if (st != VELRI_OK) {
                    goto done;
                }
                if (valid_span) {
                    op_wc++;
                }
            }

            if (idx == s) {
                step->u.op.no_barrier_before = true; /* First op in batch needs no leading barrier */
            } else {
                bool hazard = false;
                for (uint32_t i = 0U; i < op_rc && !hazard; ++i) {
                    if (span_conflicts_list(&op_r[i], writes, w_cnt)) {
                        hazard = true; /* RAW */
                    }
                }
                for (uint32_t o = 0U; o < op_wc && !hazard; ++o) {
                    if (span_conflicts_list(&op_w[o], writes, w_cnt) ||
                        span_conflicts_list(&op_w[o], reads, r_cnt)) {
                        hazard = true; /* WAW or WAR */
                    }
                }
                if (hazard) {
                    step->u.op.no_barrier_before = false;
                    r_cnt = 0U;
                    w_cnt = 0U;
                } else {
                    step->u.op.no_barrier_before = true;
                    plan->dag_elided_barriers++;
                }
            }

            if (op_rc > 0U) {
                void *nb = nullptr;
                if (!velri_vec_ensure(reads, &nb, &r_cap, r_cnt, op_rc, sizeof(phys_span))) {
                    st = VELRI_ERR_NOMEM;
                    goto done;
                }
                reads = (phys_span *)nb;
                memcpy(&reads[r_cnt], op_r, (size_t)op_rc * sizeof(phys_span));
                r_cnt += op_rc;
            }
            if (op_wc > 0U) {
                void *nb = nullptr;
                if (!velri_vec_ensure(writes, &nb, &w_cap, w_cnt, op_wc, sizeof(phys_span))) {
                    st = VELRI_ERR_NOMEM;
                    goto done;
                }
                writes = (phys_span *)nb;
                memcpy(&writes[w_cnt], op_w, (size_t)op_wc * sizeof(phys_span));
                w_cnt += op_wc;
            }
        }
        s = run_end;
    }

done:
    free(reads);
    free(writes);
    return st;
}

/* All OUTPUT storages of the final device share one packed IO arena.  Include only
 * their enclosing aligned byte range in the optional per-slot readback allocation. */
static void plan_output_readback(velri_plan *plan) {
    if (!plan->num_steps || plan->steps[plan->num_steps - 1U].kind != VELRI_STEP_OP) {
        return;
    }
    velri_device_id d = plan->steps[plan->num_steps - 1U].u.op.device_id;
    const velri_device_entry *de = &plan->ctx->devices[d];
    if (!de->vt->capture_graph || !de->hooks || !de->hooks->capture_outputs) {
        return;
    }
    size_t first = SIZE_MAX, end = 0U;
    for (uint32_t s = 0U; s < plan->num_user_storages; ++s) {
        const velri_plan_storage *ps = &plan->storages[s];
        if (ps->device_id == d && ps->arena_kind == VELRI_ARENA_IO &&
            ps->kind == VELRI_STORAGE_OUTPUT && ps->byte_size > 0U) {
            if (ps->arena_offset < first) first = ps->arena_offset;
            size_t last = ps->arena_offset + ps->byte_size; /* Bounded by the packed arena. */
            if (last > end) end = last;
        }
    }
    if (end > first) {
        size_t bytes = 0U;
        if (!velri_align_up_overflow(end - first, de->caps.min_align_bytes, &bytes) &&
            bytes <= de->hooks->max_readback_bytes &&
            bytes <= plan->slot_arena_bytes[VELRI_SLOT_ARENA_IO][d] - first) {
            plan->readback = (velri_io_range){.offset = first, .bytes = bytes};
        }
    }
}

static void compute_memory_report(velri_plan *plan) {
    velri_memory_report *rep = &plan->report;
    memset(rep, 0, sizeof(*rep));
    rep->num_devices = plan->ctx->num_devices;
    rep->ram_staging_bytes = plan->ram_staging_bytes;

    for (uint32_t d = 0U; d < rep->num_devices; ++d) {
        rep->devices[d].device_id = d;
        rep->devices[d].is_uma = plan->ctx->devices[d].caps.is_uma;
        rep->devices[d].scratch_packed_bytes =
            plan->slot_arena_bytes[VELRI_SLOT_ARENA_SCRATCH][d];
    }

    for (uint32_t i = 0U; i < plan->num_storages; ++i) {
        const velri_plan_storage *ps = &plan->storages[i];
        velri_device_memory_report *dr = &rep->devices[ps->device_id];
        switch (ps->kind) {
        case VELRI_STORAGE_WEIGHT:
            if (!plan->weight_owner) {
                size_t bytes = ps->packing.bytes ? ps->packing.bytes : ps->byte_size;
                dr->weight_bytes = velri_sat_add_size(dr->weight_bytes, bytes);
                if (ps->packing.bytes) {
                    size_t align = effective_align(plan->ctx, ps->device_id, ps->align);
                    size_t padding = velri_align_up(bytes, align) - bytes;
                    dr->allocator_overhead_bytes = velri_sat_add_size(dr->allocator_overhead_bytes, padding);
                }
            }
            break;
        case VELRI_STORAGE_INPUT:
        case VELRI_STORAGE_OUTPUT:
            dr->input_output_bytes = velri_sat_add_size(dr->input_output_bytes, ps->byte_size);
            break;
        case VELRI_STORAGE_STATE:
            dr->state_bytes = velri_sat_add_size(dr->state_bytes, ps->byte_size);
            break;
        case VELRI_STORAGE_SCRATCH:
            dr->scratch_unpacked_bytes =
                velri_sat_add_size(dr->scratch_unpacked_bytes, ps->byte_size);
            break;
        }
    }

    for (uint32_t d = 0U; d < rep->num_devices; ++d) {
        velri_device_memory_report *dr = &rep->devices[d];
        size_t ovh = dr->allocator_overhead_bytes;
        if (plan->readback.bytes && plan->steps[plan->num_steps - 1U].u.op.device_id == d) {
            ovh = velri_sat_add_size(ovh, plan->readback.bytes);
        }
        if (!plan->ctx->devices[d].caps.is_host_memory) {
            if (!plan->weight_owner) {
                size_t packed_w = 0U;
                for (uint32_t s = 0U; s < plan->num_weight_slabs[d]; ++s) {
                    packed_w = velri_sat_add_size(packed_w, plan->weight_slab_bytes[d][s]);
                }
                if (packed_w > dr->weight_bytes) {
                    ovh = velri_sat_add_size(ovh, packed_w - dr->weight_bytes);
                }
            }
            size_t io_bytes = plan->slot_arena_bytes[VELRI_SLOT_ARENA_IO][d];
            if (io_bytes > dr->input_output_bytes) {
                ovh = velri_sat_add_size(ovh, io_bytes - dr->input_output_bytes);
            }
        }
        size_t st_bytes = plan->slot_arena_bytes[VELRI_SLOT_ARENA_STATE][d];
        size_t st_ovh = 0U;
        if (st_bytes > dr->state_bytes) {
            st_ovh = st_bytes - dr->state_bytes;
            ovh = velri_sat_add_size(ovh, st_ovh);
        }
        dr->state_overhead_bytes = st_ovh;
        dr->allocator_overhead_bytes = ovh;
    }

    for (uint32_t s = 0U; s < plan->num_steps; ++s) {
        if (plan->steps[s].kind == VELRI_STEP_RAM_TRANSFER) {
            rep->num_ram_transfers++;
        }
    }

    size_t total = rep->ram_staging_bytes;
    for (uint32_t d = 0U; d < rep->num_devices; ++d) {
        const velri_device_memory_report *dr = &rep->devices[d];
        total = velri_sat_add_size(total, dr->weight_bytes);
        total = velri_sat_add_size(total, dr->input_output_bytes);
        total = velri_sat_add_size(total, dr->state_bytes);
        total = velri_sat_add_size(total, dr->scratch_packed_bytes);
        total = velri_sat_add_size(total, dr->allocator_overhead_bytes);
    }
    rep->total_physical_bytes = total;
}

velri_status velri_plan_prepare(velri_context *ctx, const velri_graph *graph,
                                const velri_plan_options *opts, velri_plan **out_plan) {
    return velri_plan_prepare_ex(ctx, graph, opts, 0, out_plan);
}

velri_status velri_plan_prepare_ex(velri_context *ctx, const velri_graph *graph,
                                   const velri_plan_options *opts, uint32_t flags,
                                   velri_plan **out_plan) {
    if (!ctx || !graph || !out_plan) {
        return VELRI_ERR_INVALID;
    }
    *out_plan = nullptr;
    if (flags & ~(uint32_t)VELRI_PLAN_PACK_CPU_WEIGHTS) return VELRI_ERR_INVALID;

    bool allow_cpu_fallback = opts ? opts->allow_cpu_fallback : false;
    velri_plan *share_owner = nullptr;
    if (opts && opts->share_weights_from) {
        share_owner = opts->share_weights_from->weight_owner ? opts->share_weights_from->weight_owner
                                                             : opts->share_weights_from;
        if (share_owner->ctx != ctx) {
            return VELRI_ERR_INVALID;
        }
    }

    for (uint32_t i = 0U; i < graph->num_storages; ++i) {
        if (graph->storages[i].device_id >= ctx->num_devices) {
            return VELRI_ERR_NOT_FOUND;
        }
    }

    velri_plan *plan = (velri_plan *)calloc(1U, sizeof(*plan));
    if (!plan) {
        return VELRI_ERR_NOMEM;
    }
    atomic_init(&plan->refcount, 1U);
    velri_context_retain(ctx);
    plan->ctx = ctx;
    plan->num_user_storages = graph->num_storages;

    plan_builder pb = {.plan = plan, .storage_cap = 0U, .step_cap = 0U};

    if (graph->num_storages > 0U) {
        void *buf = nullptr;
        if (!velri_vec_ensure(plan->storages, &buf, &pb.storage_cap, 0U, graph->num_storages,
                              sizeof(velri_plan_storage))) {
            velri_plan_release(plan);
            return VELRI_ERR_NOMEM;
        }
        plan->storages = (velri_plan_storage *)buf;

        if (!share_owner) {
            plan->weight_bindings = (velri_plan_weight_binding *)calloc(
                graph->num_storages, sizeof(*plan->weight_bindings));
            if (!plan->weight_bindings) {
                velri_plan_release(plan);
                return VELRI_ERR_NOMEM;
            }
        }
    }

    coherence_state cs;
    velri_status st = coherence_state_init(&cs, graph, plan);
    if (st != VELRI_OK) {
        velri_plan_release(plan);
        return st;
    }

    /* Pack & allocate (or alias) shared non-host WEIGHT slabs, plus STATE/IO arena layouts. */
    st = pack_weight_slabs(plan, share_owner == nullptr);
    if (st == VELRI_OK && share_owner != nullptr) {
        st = bind_shared_weights_from_owner(plan, share_owner);
    }
    if (st == VELRI_OK) {
        st = pack_state_and_io_arenas(plan);
    }

    weight_shadow_cache wcache;
    memset(&wcache, 0, sizeof(wcache));
    wcache.prev_dev = VELRI_INVALID_ID;

    /* Record the last graph node touching each user storage so offloaded STATE/OUTPUT
     * shadows flush immediately after their final use instead of staying pinned. */
    uint32_t *last_node_for_storage = nullptr;
    if (st == VELRI_OK && graph->num_storages > 0U) {
        size_t ln_bytes = 0U;
        if (velri_mul_overflow_size((size_t)graph->num_storages, sizeof(uint32_t), &ln_bytes)) {
            st = VELRI_ERR_OVERFLOW;
        } else {
            last_node_for_storage = (uint32_t *)malloc(ln_bytes);
            if (!last_node_for_storage) {
                st = VELRI_ERR_NOMEM;
            } else {
                for (uint32_t sid = 0U; sid < graph->num_storages; ++sid) {
                    last_node_for_storage[sid] = UINT32_MAX;
                }
                for (uint32_t n = 0U; n < graph->num_nodes; ++n) {
                    const velri_op_desc *d = &graph->nodes[n].desc;
                    for (uint32_t i = 0U; i < d->num_inputs; ++i) {
                        velri_view_id vid = d->inputs[i];
                        if (vid != VELRI_INVALID_ID) {
                            last_node_for_storage[graph->views[vid].storage_id] = n;
                        }
                    }
                    for (uint32_t o = 0U; o < d->num_outputs; ++o) {
                        velri_view_id vid = d->outputs[o];
                        if (vid != VELRI_INVALID_ID) {
                            last_node_for_storage[graph->views[vid].storage_id] = n;
                        }
                    }
                }
            }
        }
    }

    /* Lower each operation node into device-verified steps + cached explicit transfers. */
    for (uint32_t n = 0U; n < graph->num_nodes && st == VELRI_OK; ++n) {
        st = lower_graph_node(&pb, graph, &graph->nodes[n], n, last_node_for_storage,
                              allow_cpu_fallback, &cs, &wcache);
    }

    /* Safety net: transfer any remaining offloaded OUTPUT or STATE back to home device. */
    if (st == VELRI_OK) {
        for (uint32_t sid = 0U; sid < graph->num_storages; ++sid) {
            velri_storage_kind kind = plan->storages[sid].kind;
            if ((kind == VELRI_STORAGE_OUTPUT || kind == VELRI_STORAGE_STATE) &&
                cs.owner_dev[sid] != plan->storages[sid].device_id) {
                st = emit_ram_transfer(&pb, cs.owner_dev[sid], cs.owner_idx[sid],
                                       plan->storages[sid].device_id, sid,
                                       plan->storages[sid].byte_size);
                if (st != VELRI_OK) {
                    break;
                }
            }
        }
    }

    free(last_node_for_storage);
    coherence_state_destroy(&cs);

    bool inspect_weights = share_owner ? share_owner->has_packed_weights
                                       : (flags & VELRI_PLAN_PACK_CPU_WEIGHTS) != 0;
    if (st == VELRI_OK && inspect_weights) {
        st = velri_plan_prepare_host_weights(plan);
    }
    if (st == VELRI_OK) {
        st = pack_scratch_arenas(plan);
    }
    if (st == VELRI_OK) {
        st = analyze_dag_barriers_and_batches(plan);
    }
    if (st != VELRI_OK) {
        velri_plan_release(plan);
        return st;
    }

    plan_output_readback(plan);
    compute_memory_report(plan);
    *out_plan = plan;
    return VELRI_OK;
}

void velri_plan_retain(velri_plan *plan) {
    if (plan) {
        velri_ref_inc(&plan->refcount);
    }
}

void velri_plan_release(velri_plan *plan) {
    if (!plan || !velri_ref_dec(&plan->refcount)) {
        return;
    }
    if (plan->ctx) {
        if (plan->steps) {
            for (uint32_t i = 0U; i < plan->num_steps; ++i) {
                velri_launch_step *step = &plan->steps[i];
                if (step->kind == VELRI_STEP_OP && step->u.op.prepared) {
                    velri_device_entry *de = &plan->ctx->devices[step->u.op.device_id];
                    if (de->vt->destroy_prepared_op) {
                        de->vt->destroy_prepared_op(de->dev, step->u.op.prepared);
                    }
                }
            }
        }
        if (!plan->weight_owner) {
            for (uint32_t i = 0; plan->weight_bindings && i < plan->num_user_storages; ++i) {
                void *packed = plan->weight_bindings[i].packed_mem;
                if (packed) {
                    velri_device_entry *de = &plan->ctx->devices[plan->storages[i].device_id];
                    de->vt->free(de->dev, packed);
                }
            }
            for (uint32_t d = 0U; d < plan->ctx->num_devices; ++d) {
                velri_device_entry *de = &plan->ctx->devices[d];
                for (uint32_t s = 0U; s < plan->num_weight_slabs[d]; ++s) {
                    if (plan->weight_slabs[d][s]) {
                        de->vt->free(de->dev, plan->weight_slabs[d][s]);
                    }
                }
            }
        }
        if (plan->weight_owner) {
            velri_plan_release(plan->weight_owner);
        }
        velri_context_release(plan->ctx);
    }
    free(plan->weight_peer_sid);
    free(plan->weight_bindings);
    free(plan->steps);
    free(plan->storages);
    free(plan);
}

void velri_plan_destroy(velri_plan *plan) { velri_plan_release(plan); }

velri_status velri_plan_memory_report(const velri_plan *plan, velri_memory_report *out_report) {
    if (!plan || !out_report) {
        return VELRI_ERR_INVALID;
    }
    *out_report = plan->report;
    return VELRI_OK;
}

uint32_t velri_plan_launch_step_count(const velri_plan *plan) {
    return plan ? plan->num_steps : 0U;
}

velri_status velri_plan_barrier_stats(const velri_plan *plan, uint32_t *out_total_ops,
                                      uint32_t *out_elided_barriers) {
    if (!plan) {
        return VELRI_ERR_INVALID;
    }
    if (out_total_ops) {
        *out_total_ops = plan->dag_total_ops;
    }
    if (out_elided_barriers) {
        *out_elided_barriers = plan->dag_elided_barriers;
    }
    return VELRI_OK;
}
