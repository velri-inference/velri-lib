#include "core_internal.h"
#include <string.h>

/*
 * Execution Slots & Runtime Launch-Graph Executor (`velri-lib/docs/EN/execution.md`).
 *
 * Key invariants:
 *   - Immutable `VELRI_STORAGE_WEIGHT` buffers live once on `velri_plan` and are shared
 *     read-only across all concurrent slots.
 *   - `VELRI_STORAGE_STATE` remains resident on its device across `velri_slot_run` calls;
 *     explicit host sync uses `velri_slot_sync_state_{from,to}_host`.
 *   - Consecutive ops targeting the same device are batched via `submit_batch` when
 *     supported by the backend (0 B heap allocations during `velri_slot_run`).
 *   - Dynamic bounds (e.g. RoPE `pos_offset_override`) are checked before launching work.
 */

typedef struct {
    void *host_ptr;
    size_t bound_bytes;
    void *device_handle; /* Allocated once at slot creation for non-host INPUT/OUTPUT/STATE */
    bool is_bound;
    bool state_uploaded; /* Initial upload flag for non-host STATE */
} slot_storage_binding;

struct velri_slot {
    velri_plan *plan;
    void *scratch_arenas[VELRI_MAX_DEVICES];
    void *ram_staging;
    slot_storage_binding *user_bindings;
    bool needs_reset;
    uint64_t last_run_allocs;
};

static velri_buffer_binding resolve_storage_binding(const velri_slot *slot,
                                                    uint32_t plan_storage_idx,
                                                    size_t view_byte_offset) {
    const velri_plan *plan = slot->plan;
    const velri_plan_storage *ps = &plan->storages[plan_storage_idx];
    const velri_device_entry *de = &plan->ctx->devices[ps->device_id];
    velri_buffer_binding b;
    memset(&b, 0, sizeof(b));

    if (ps->in_scratch_arena) {
        b.mem_handle = slot->scratch_arenas[ps->device_id];
        b.byte_offset = ps->arena_offset + view_byte_offset;
        b.byte_size = plan->scratch_arena_bytes[ps->device_id];
        b.is_host_ptr = de->caps.is_host_memory;
        return b;
    }

    if (ps->kind == VELRI_STORAGE_WEIGHT) {
        const velri_plan_weight_binding *wb = &plan->weight_bindings[plan_storage_idx];
        if (de->caps.is_host_memory) {
            b.mem_handle = (void *)wb->host_ptr;
            b.byte_offset = view_byte_offset;
            b.byte_size = wb->bound_bytes;
            b.is_host_ptr = true;
        } else {
            b.mem_handle = wb->device_handle;
            b.byte_offset = view_byte_offset;
            b.byte_size = ps->byte_size;
            b.is_host_ptr = false;
        }
        return b;
    }

    const slot_storage_binding *sb = &slot->user_bindings[plan_storage_idx];
    if (de->caps.is_host_memory) {
        b.mem_handle = sb->host_ptr;
        b.byte_offset = view_byte_offset;
        b.byte_size = sb->bound_bytes;
        b.is_host_ptr = true;
    } else {
        b.mem_handle = sb->device_handle;
        b.byte_offset = view_byte_offset;
        b.byte_size = ps->byte_size;
        b.is_host_ptr = false;
    }
    return b;
}

velri_status velri_plan_bind_weight_host(velri_plan *plan, velri_storage_id storage_id,
                                         const void *host_ptr, size_t byte_size) {
    if (!plan || !host_ptr || storage_id >= plan->num_user_storages) {
        return VELRI_ERR_INVALID;
    }
    const velri_plan_storage *ps = &plan->storages[storage_id];
    if (ps->kind != VELRI_STORAGE_WEIGHT || byte_size < ps->byte_size) {
        return VELRI_ERR_INVALID;
    }
    size_t req_align = velri_dtype_align(ps->dtype);
    if (((uintptr_t)host_ptr % req_align) != 0U) {
        return VELRI_ERR_INVALID;
    }

    velri_plan_weight_binding *wb = &plan->weight_bindings[storage_id];
    wb->host_ptr = host_ptr;
    wb->bound_bytes = byte_size;

    velri_device_entry *de = &plan->ctx->devices[ps->device_id];
    if (!de->caps.is_host_memory) {
        velri_buffer_binding dst = {
            .mem_handle = wb->device_handle,
            .byte_offset = 0U,
            .byte_size = ps->byte_size,
            .is_host_ptr = false,
        };
        velri_status st = de->vt->copy_from_ram(de->dev, &dst, host_ptr, ps->byte_size);
        if (st != VELRI_OK) {
            wb->is_bound = false;
            return st;
        }
    }
    wb->is_bound = true;
    return VELRI_OK;
}

velri_status velri_slot_create(velri_plan *plan, velri_slot **out_slot) {
    if (!plan || !out_slot) {
        return VELRI_ERR_INVALID;
    }
    *out_slot = nullptr;

    velri_slot *slot = (velri_slot *)calloc(1U, sizeof(*slot));
    if (!slot) {
        return VELRI_ERR_NOMEM;
    }
    velri_plan_retain(plan);
    slot->plan = plan;

    if (plan->num_user_storages > 0U) {
        slot->user_bindings =
            (slot_storage_binding *)calloc(plan->num_user_storages, sizeof(*slot->user_bindings));
        if (!slot->user_bindings) {
            velri_slot_destroy(slot);
            return VELRI_ERR_NOMEM;
        }
    }

    for (uint32_t d = 0U; d < plan->ctx->num_devices; ++d) {
        size_t sz = plan->scratch_arena_bytes[d];
        if (sz == 0U) continue;
        velri_status st = velri_device_alloc_tracked(plan->ctx, d, sz, plan->scratch_arena_align[d],
                                                     &slot->scratch_arenas[d]);
        if (st != VELRI_OK) {
            velri_slot_destroy(slot);
            return st;
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

    /* Pre-allocate per-slot non-host device buffers ONLY for INPUT / OUTPUT / STATE (never WEIGHT). */
    for (uint32_t i = 0U; i < plan->num_user_storages; ++i) {
        const velri_plan_storage *ps = &plan->storages[i];
        if (ps->kind == VELRI_STORAGE_SCRATCH || ps->kind == VELRI_STORAGE_WEIGHT) {
            continue;
        }
        if (plan->ctx->devices[ps->device_id].caps.is_host_memory) {
            continue;
        }
        velri_status st = velri_device_alloc_tracked(plan->ctx, ps->device_id, ps->byte_size,
                                                     ps->align,
                                                     &slot->user_bindings[i].device_handle);
        if (st != VELRI_OK) {
            velri_slot_destroy(slot);
            return st;
        }
    }

    *out_slot = slot;
    return VELRI_OK;
}

void velri_slot_destroy(velri_slot *slot) {
    if (!slot) {
        return;
    }
    velri_plan *plan = slot->plan;
    if (plan && plan->ctx) {
        if (slot->user_bindings) {
            for (uint32_t i = 0U; i < plan->num_user_storages; ++i) {
                if (slot->user_bindings[i].device_handle) {
                    velri_device_entry *de = &plan->ctx->devices[plan->storages[i].device_id];
                    de->vt->free(de->dev, slot->user_bindings[i].device_handle);
                }
            }
        }
        for (uint32_t d = 0U; d < plan->ctx->num_devices; ++d) {
            if (slot->scratch_arenas[d]) {
                velri_device_entry *de = &plan->ctx->devices[d];
                de->vt->free(de->dev, slot->scratch_arenas[d]);
            }
        }
    }
    velri_os_aligned_free(slot->ram_staging);
    free(slot->user_bindings);
    free(slot);
    velri_plan_release(plan);
}

velri_status velri_slot_bind_host(velri_slot *slot, velri_storage_id storage_id, void *host_ptr,
                                  size_t byte_size) {
    if (!slot || !host_ptr || storage_id >= slot->plan->num_user_storages) {
        return VELRI_ERR_INVALID;
    }
    const velri_plan_storage *ps = &slot->plan->storages[storage_id];
    if (ps->kind == VELRI_STORAGE_SCRATCH || ps->kind == VELRI_STORAGE_WEIGHT ||
        byte_size < ps->byte_size) {
        return VELRI_ERR_INVALID;
    }
    size_t req_align = velri_dtype_align(ps->dtype);
    if (((uintptr_t)host_ptr % req_align) != 0U) {
        return VELRI_ERR_INVALID;
    }

    slot_storage_binding *sb = &slot->user_bindings[storage_id];
    sb->host_ptr = host_ptr;
    sb->bound_bytes = byte_size;
    sb->is_bound = true;
    sb->state_uploaded = false;
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
    velri_device_entry *de = &slot->plan->ctx->devices[ps->device_id];
    if (de->caps.is_host_memory) {
        sb->state_uploaded = true;
        return VELRI_OK;
    }
    velri_buffer_binding dst = resolve_storage_binding(slot, storage_id, 0U);
    velri_status st = de->vt->copy_from_ram(de->dev, &dst, sb->host_ptr, ps->byte_size);
    if (st == VELRI_OK) {
        sb->state_uploaded = true;
    }
    return st;
}

velri_status velri_slot_sync_state_to_host(velri_slot *slot, velri_storage_id storage_id) {
    if (!slot || storage_id >= slot->plan->num_user_storages) {
        return VELRI_ERR_INVALID;
    }
    const velri_plan_storage *ps = &slot->plan->storages[storage_id];
    slot_storage_binding *sb = &slot->user_bindings[storage_id];
    if (ps->kind != VELRI_STORAGE_STATE || !sb->is_bound) {
        return VELRI_ERR_INVALID;
    }
    velri_device_entry *de = &slot->plan->ctx->devices[ps->device_id];
    if (de->caps.is_host_memory) {
        return VELRI_OK;
    }
    velri_buffer_binding src = resolve_storage_binding(slot, storage_id, 0U);
    return de->vt->copy_to_ram(de->dev, sb->host_ptr, &src, ps->byte_size);
}

static velri_status validate_dynamic_params(const velri_plan *plan,
                                            const velri_step_params *params) {
    if (!params || !params->has_pos_offset) {
        return VELRI_OK;
    }
    if (params->pos_offset_override > (size_t)UINT32_MAX) {
        return VELRI_ERR_OVERFLOW;
    }
    for (uint32_t s = 0U; s < plan->num_steps; ++s) {
        const velri_launch_step *step = &plan->steps[s];
        if (step->kind != VELRI_STEP_OP || step->u.op.resolved.kind != VELRI_OP_ROPE) {
            continue;
        }
        const velri_resolved_op *op = &step->u.op.resolved;
        size_t end_pos = 0U, need = 0U;
        if (velri_add_overflow_size(params->pos_offset_override, op->inputs[0].dims[0], &end_pos) ||
            velri_mul_overflow_size(end_pos, op->raw.attr.rope.rotary_dim / 2U, &need)) {
            return VELRI_ERR_OVERFLOW;
        }
        size_t ct_n = 0U, st_n = 0U;
        if (velri_tensor_view_num_elements(&op->inputs[1], &ct_n) != VELRI_OK ||
            velri_tensor_view_num_elements(&op->inputs[2], &st_n) != VELRI_OK ||
            ct_n < need || st_n < need) {
            return VELRI_ERR_INVALID;
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

    velri_status st = src_de->vt->copy_to_ram(src_de->dev, slot->ram_staging, &src_b,
                                              step->u.transfer.byte_size);
    if (st != VELRI_OK) {
        return st;
    }
    return dst_de->vt->copy_from_ram(dst_de->dev, &dst_b, slot->ram_staging,
                                     step->u.transfer.byte_size);
}

static velri_status run_compute_batch(velri_slot *slot, uint32_t first_step, uint32_t count,
                                      const velri_step_params *params, bool *out_accepted) {
    *out_accepted = false;
    const velri_plan *plan = slot->plan;
    velri_device_id dev_id = plan->steps[first_step].u.op.device_id;
    velri_device_entry *de = &plan->ctx->devices[dev_id];

    velri_buffer_binding in_b[VELRI_MAX_SUBMIT_BATCH][VELRI_OP_MAX_INPUTS];
    velri_buffer_binding out_b[VELRI_MAX_SUBMIT_BATCH][VELRI_OP_MAX_OUTPUTS];
    velri_submit_item items[VELRI_MAX_SUBMIT_BATCH];

    for (uint32_t k = 0U; k < count; ++k) {
        const velri_launch_step *step = &plan->steps[first_step + k];
        memset(in_b[k], 0, sizeof(in_b[k]));
        memset(out_b[k], 0, sizeof(out_b[k]));
        for (uint32_t i = 0U; i < step->u.op.num_inputs; ++i) {
            if (step->u.op.inputs[i].present) {
                in_b[k][i] = resolve_storage_binding(slot, step->u.op.inputs[i].plan_storage_idx,
                                                     step->u.op.inputs[i].view_byte_offset);
            }
        }
        for (uint32_t o = 0U; o < step->u.op.num_outputs; ++o) {
            out_b[k][o] = resolve_storage_binding(slot, step->u.op.outputs[o].plan_storage_idx,
                                                  step->u.op.outputs[o].view_byte_offset);
        }
        items[k] = (velri_submit_item){
            .op = &step->u.op.resolved,
            .prepared = step->u.op.prepared,
            .inputs = in_b[k],
            .num_inputs = step->u.op.num_inputs,
            .outputs = out_b[k],
            .num_outputs = step->u.op.num_outputs,
            .params = params,
        };
    }

    velri_completion *comp = nullptr;
    velri_status submit_st = (count > 1U && de->vt->submit_batch)
                                 ? de->vt->submit_batch(de->dev, items, count, &comp)
                                 : de->vt->submit(de->dev, &items[0], &comp);

    if (de->vt->wait) {
        /* Asynchronous backend contract: non-null completion => work accepted, MUST wait+destroy. */
        if (!comp) {
            return submit_st;
        }
        *out_accepted = true;
        velri_status wait_st = de->vt->wait(de->dev, comp);
        de->vt->destroy_completion(de->dev, comp);
        return (submit_st != VELRI_OK) ? submit_st : wait_st;
    }

    /* Synchronous backend contract (`wait == nullptr`): OK or EXEC means work touched buffers. */
    if (submit_st == VELRI_OK || submit_st == VELRI_ERR_EXEC) {
        *out_accepted = true;
    }
    return submit_st;
}

velri_status velri_slot_run(velri_slot *slot, const velri_step_params *params,
                            velri_cancel_token *cancel) {
    if (!slot) {
        return VELRI_ERR_INVALID;
    }
    if (slot->needs_reset) {
        return VELRI_ERR_BUSY;
    }

    const velri_plan *plan = slot->plan;
    uint64_t allocs_before = atomic_load_explicit(&plan->ctx->total_allocs, memory_order_relaxed);
    slot->last_run_allocs = 0U;

    /* Validate all external storages (plan weights + per-slot bindings) and dynamic params. */
    for (uint32_t i = 0U; i < plan->num_user_storages; ++i) {
        velri_storage_kind kind = plan->storages[i].kind;
        if (kind == VELRI_STORAGE_SCRATCH) {
            continue;
        }
        if (kind == VELRI_STORAGE_WEIGHT) {
            if (!plan->weight_bindings[i].is_bound) return VELRI_ERR_INVALID;
        } else if (!slot->user_bindings[i].is_bound) {
            return VELRI_ERR_INVALID;
        }
    }

    velri_step_params default_params = {0};
    const velri_step_params *eff_params = params ? params : &default_params;
    velri_status pst = validate_dynamic_params(plan, eff_params);
    if (pst != VELRI_OK) {
        return pst;
    }

    if (velri_cancel_is_requested(cancel)) {
        return VELRI_ERR_CANCELLED;
    }

    /* Upload step INPUTs (and initial un-uploaded STATE) to non-host devices before running steps. */
    for (uint32_t i = 0U; i < plan->num_user_storages; ++i) {
        const velri_plan_storage *ps = &plan->storages[i];
        velri_device_entry *de = &plan->ctx->devices[ps->device_id];
        if (de->caps.is_host_memory) {
            continue;
        }
        bool need_up = (ps->kind == VELRI_STORAGE_INPUT) ||
                       (ps->kind == VELRI_STORAGE_STATE && !slot->user_bindings[i].state_uploaded);
        if (!need_up) {
            continue;
        }
        if (velri_cancel_is_requested(cancel)) {
            return VELRI_ERR_CANCELLED;
        }
        velri_buffer_binding dst = resolve_storage_binding(slot, i, 0U);
        velri_status st =
            de->vt->copy_from_ram(de->dev, &dst, slot->user_bindings[i].host_ptr, ps->byte_size);
        if (st != VELRI_OK) {
            return st;
        }
        if (ps->kind == VELRI_STORAGE_STATE) {
            slot->user_bindings[i].state_uploaded = true;
        }
    }

    bool any_work_accepted = false;
    uint32_t s = 0U;
    while (s < plan->num_steps) {
        if (velri_cancel_is_requested(cancel)) {
            if (any_work_accepted) slot->needs_reset = true;
            slot->last_run_allocs =
                atomic_load_explicit(&plan->ctx->total_allocs, memory_order_relaxed) - allocs_before;
            return VELRI_ERR_CANCELLED;
        }

        const velri_launch_step *step = &plan->steps[s];
        if (step->kind == VELRI_STEP_RAM_TRANSFER) {
            velri_status st = run_ram_transfer(slot, step);
            if (st != VELRI_OK) {
                if (any_work_accepted) slot->needs_reset = true;
                slot->last_run_allocs =
                    atomic_load_explicit(&plan->ctx->total_allocs, memory_order_relaxed) -
                    allocs_before;
                return st;
            }
            s++;
            continue;
        }

        /* Group consecutive ops on the same device if backend supports submit_batch. */
        velri_device_id dev_id = step->u.op.device_id;
        uint32_t batch = 1U;
        if (plan->ctx->devices[dev_id].vt->submit_batch) {
            while (s + batch < plan->num_steps && batch < VELRI_MAX_SUBMIT_BATCH &&
                   plan->steps[s + batch].kind == VELRI_STEP_OP &&
                   plan->steps[s + batch].u.op.device_id == dev_id) {
                batch++;
            }
        }

        bool accepted = false;
        velri_status st = run_compute_batch(slot, s, batch, eff_params, &accepted);
        if (accepted) {
            any_work_accepted = true;
        }
        if (st != VELRI_OK) {
            if (any_work_accepted) slot->needs_reset = true;
            slot->last_run_allocs =
                atomic_load_explicit(&plan->ctx->total_allocs, memory_order_relaxed) - allocs_before;
            return st;
        }
        s += batch;
    }

    /* Download ONLY non-host OUTPUTs (STATE stays resident on its home device!). */
    for (uint32_t i = 0U; i < plan->num_user_storages; ++i) {
        const velri_plan_storage *ps = &plan->storages[i];
        velri_device_entry *de = &plan->ctx->devices[ps->device_id];
        if (de->caps.is_host_memory || ps->kind != VELRI_STORAGE_OUTPUT) {
            continue;
        }
        velri_buffer_binding src = resolve_storage_binding(slot, i, 0U);
        velri_status st =
            de->vt->copy_to_ram(de->dev, slot->user_bindings[i].host_ptr, &src, ps->byte_size);
        if (st != VELRI_OK) {
            slot->needs_reset = true;
            slot->last_run_allocs =
                atomic_load_explicit(&plan->ctx->total_allocs, memory_order_relaxed) - allocs_before;
            return st;
        }
    }

    slot->last_run_allocs =
        atomic_load_explicit(&plan->ctx->total_allocs, memory_order_relaxed) - allocs_before;
    return VELRI_OK;
}

bool velri_slot_needs_reset(const velri_slot *slot) { return slot ? slot->needs_reset : false; }

velri_status velri_slot_reset(velri_slot *slot) {
    if (!slot) {
        return VELRI_ERR_INVALID;
    }
    const velri_plan *plan = slot->plan;
    for (uint32_t d = 0U; d < plan->ctx->num_devices; ++d) {
        size_t sz = plan->scratch_arena_bytes[d];
        if (!slot->scratch_arenas[d] || sz == 0U) {
            continue;
        }
        velri_device_entry *de = &plan->ctx->devices[d];
        if (de->caps.is_host_memory) {
            memset(slot->scratch_arenas[d], 0, sz);
        } else if (de->vt->zero_mem) {
            velri_status st = de->vt->zero_mem(de->dev, slot->scratch_arenas[d], sz);
            if (st != VELRI_OK) {
                return st;
            }
        }
    }
    for (uint32_t i = 0U; i < plan->num_user_storages; ++i) {
        slot->user_bindings[i].state_uploaded = false;
    }
    slot->needs_reset = false;
    return VELRI_OK;
}

uint64_t velri_slot_last_run_allocs(const velri_slot *slot) {
    return slot ? slot->last_run_allocs : 0U;
}
