#include "core_internal.h"

typedef enum { UNUSED, PACKABLE, NEEDS_RAW } weight_use;

static bool same_packing(const velri_weight_packing *a, const velri_weight_packing *b) {
    return a->channels == b->channels && a->depth == b->depth &&
           a->stride == b->stride && a->bytes == b->bytes;
}

/* A transfer or a consumer with a different view needs the original byte layout. */
static void inspect_uses(velri_plan *plan, weight_use *uses) {
    for (uint32_t s = 0; s < plan->num_steps; ++s) {
        const velri_launch_step *step = &plan->steps[s];
        if (step->kind == VELRI_STEP_RAM_TRANSFER) {
            uint32_t src = step->u.transfer.src_plan_storage_idx;
            if (src < plan->num_user_storages) uses[src] = NEEDS_RAW;
            continue;
        }
        const velri_resolved_op *op = &step->u.op.resolved;
        const velri_device_entry *de = &plan->ctx->devices[step->u.op.device_id];
        for (uint32_t i = 0; i < op->num_inputs; ++i) {
            if (!op->input_present[i]) continue;
            uint32_t sid = step->u.op.in_storage_idx[i];
            if (sid >= plan->num_user_storages || uses[sid] == NEEDS_RAW) continue;
            velri_plan_storage *ps = &plan->storages[sid];
            if (ps->kind != VELRI_STORAGE_WEIGHT) continue;
            velri_weight_packing candidate = {0};
            if (ps->device_id != step->u.op.device_id || !de->hooks || !de->hooks->describe_weight ||
                !de->hooks->describe_weight(op, i, &candidate) ||
                (uses[sid] == PACKABLE && !same_packing(&candidate, &ps->packing))) {
                uses[sid] = NEEDS_RAW;
            } else {
                uses[sid] = PACKABLE;
                ps->packing = candidate;
            }
        }
    }
}

bool velri_plan_weight_is_packed(const velri_plan *plan, velri_storage_id sid) {
    return plan && sid < plan->num_user_storages && plan->storages[sid].packing.bytes != 0;
}

velri_status velri_plan_prepare_host_weights(velri_plan *plan) {
    if (!plan->num_user_storages) return VELRI_OK;
    weight_use *uses = calloc(plan->num_user_storages, sizeof(*uses));
    if (!uses) return VELRI_ERR_NOMEM;
    inspect_uses(plan, uses);
    velri_status st = VELRI_OK;
    for (uint32_t i = 0; i < plan->num_user_storages && st == VELRI_OK; ++i) {
        velri_plan_storage *ps = &plan->storages[i];
        if (ps->kind != VELRI_STORAGE_WEIGHT) continue;
        if (uses[i] != PACKABLE) ps->packing = (velri_weight_packing){0};
        if (plan->weight_owner) {
            const velri_weight_packing *owner =
                &plan->weight_owner->storages[plan->weight_peer_sid[i]].packing;
            if (owner->bytes && uses[i] != UNUSED && !same_packing(owner, &ps->packing)) {
                st = VELRI_ERR_INVALID;
            }
            ps->packing = *owner;
        } else if (ps->packing.bytes) {
            st = velri_device_alloc_tracked(plan->ctx, ps->device_id, ps->packing.bytes,
                                            ps->align, VELRI_ARENA_WEIGHT,
                                            &plan->weight_bindings[i].packed_mem);
            if (st == VELRI_OK) plan->has_packed_weights = true;
        }
    }
    if (st == VELRI_OK) {
        for (uint32_t s = 0; s < plan->num_steps; ++s) {
            velri_launch_step *step = &plan->steps[s];
            if (step->kind != VELRI_STEP_OP) continue;
            const velri_resolved_op *op = &step->u.op.resolved;
            for (uint32_t i = 0; i < op->num_inputs; ++i) {
                if (op->input_present[i] && plan->storages[step->u.op.in_storage_idx[i]].packing.bytes) {
                    step->u.op.uses_packed_weight = true;
                    break;
                }
            }
        }
    }
    free(uses);
    return st;
}
