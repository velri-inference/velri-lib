#include "core_internal.h"
#include <string.h>

static bool validate_vtable(const velri_backend_vtable *vt) {
    if (!vt || !vt->name || vt->name[0] == '\0') {
        return false;
    }
    if (vt->contract_version != VELRI_BACKEND_CONTRACT_VERSION) {
        return false;
    }
    if (!vt->enumerate_devices || !vt->open_device || !vt->query_caps || !vt->close_device ||
        !vt->alloc || !vt->free || !vt->copy_to_ram || !vt->copy_from_ram ||
        !vt->query_operation || !vt->submit) {
        return false;
    }
    /* Optional paired hooks must either both be provided or both be nullptr. */
    if ((vt->prepare_op != nullptr) != (vt->destroy_prepared_op != nullptr)) {
        return false;
    }
    if ((vt->wait != nullptr) != (vt->destroy_completion != nullptr)) {
        return false;
    }
    bool has_cap = (vt->capture_graph != nullptr);
    if ((vt->destroy_captured_graph != nullptr) != has_cap ||
        (vt->replay_graph != nullptr) != has_cap) {
        return false;
    }
    return true;
}

static velri_status register_backend_with_cfg(velri_context *ctx,
                                              const velri_backend_vtable *vtable,
                                              const void *open_cfg) {
    if (!ctx || !validate_vtable(vtable)) {
        return VELRI_ERR_INVALID;
    }
    for (uint32_t i = 0U; i < ctx->num_backends; ++i) {
        if (strcmp(ctx->backends[i]->name, vtable->name) == 0) {
            return VELRI_ERR_INVALID;
        }
    }
    if (ctx->num_backends >= VELRI_MAX_BACKENDS) {
        return VELRI_ERR_NOMEM;
    }

    uint32_t count = 0U;
    velri_status st = vtable->enumerate_devices(&count);
    if (st != VELRI_OK) {
        return st;
    }

    uint32_t start_dev = ctx->num_devices;
    for (uint32_t idx = 0U; idx < count; ++idx) {
        if (ctx->num_devices >= VELRI_MAX_DEVICES) {
            break;
        }
        velri_backend_device *dev = nullptr;
        st = vtable->open_device(idx, open_cfg, &dev);
        if (st != VELRI_OK || !dev) {
            st = (st != VELRI_OK) ? st : VELRI_ERR_EXEC;
            goto rollback;
        }
        velri_device_caps caps;
        memset(&caps, 0, sizeof(caps));
        st = vtable->query_caps(dev, &caps);
        if (st != VELRI_OK) {
            vtable->close_device(dev);
            goto rollback;
        }
        ctx->devices[ctx->num_devices].vt = vtable;
        ctx->devices[ctx->num_devices].dev = dev;
        ctx->devices[ctx->num_devices].caps = caps;
        ctx->num_devices++;
    }

    ctx->backends[ctx->num_backends++] = vtable;
    return VELRI_OK;

rollback:
    for (uint32_t j = ctx->num_devices; j > start_dev; --j) {
        velri_device_entry *e = &ctx->devices[j - 1U];
        e->vt->close_device(e->dev);
        memset(e, 0, sizeof(*e));
    }
    ctx->num_devices = start_dev;
    return st;
}

velri_status velri_context_register_backend(velri_context *ctx,
                                            const velri_backend_vtable *vtable) {
    return register_backend_with_cfg(ctx, vtable, nullptr);
}

velri_status velri_context_create(const velri_context_config *cfg, velri_context **out_ctx) {
    if (!out_ctx) {
        return VELRI_ERR_INVALID;
    }
    *out_ctx = nullptr;

    velri_context *ctx = (velri_context *)calloc(1U, sizeof(*ctx));
    if (!ctx) {
        return VELRI_ERR_NOMEM;
    }
    atomic_init(&ctx->refcount, 1U);
    atomic_init(&ctx->total_allocs, 0U);
    if (cfg) {
        ctx->cfg = *cfg;
    } else {
        ctx->cfg.cpu_workers = 0U;
        ctx->cfg.cpu_variant = VELRI_CPU_AUTO;
        ctx->cfg.enable_vulkan = false;
    }

    velri_cpu_open_cfg cpu_cfg = {
        .workers = ctx->cfg.cpu_workers,
        .variant = ctx->cfg.cpu_variant,
        .use_physical_cores = ctx->cfg.use_physical_cores,
        .reserve_cores = ctx->cfg.cpu_reserve_cores,
    };
    velri_status st = register_backend_with_cfg(ctx, velri_cpu_backend_vtable(), &cpu_cfg);
    if (st != VELRI_OK) {
        velri_context_release(ctx);
        return st;
    }

#ifdef VELRI_CORE_HAS_VULKAN
    if (ctx->cfg.enable_vulkan) {
        (void)register_backend_with_cfg(ctx, velri_vulkan_backend_vtable(), nullptr);
    }
#endif

    *out_ctx = ctx;
    return VELRI_OK;
}

void velri_context_retain(velri_context *ctx) {
    if (ctx) {
        velri_ref_inc(&ctx->refcount);
    }
}

void velri_context_release(velri_context *ctx) {
    if (!ctx || !velri_ref_dec(&ctx->refcount)) {
        return;
    }
    for (uint32_t i = ctx->num_devices; i > 0U; --i) {
        velri_device_entry *e = &ctx->devices[i - 1U];
        if (e->vt && e->dev) {
            e->vt->close_device(e->dev);
        }
    }
    free(ctx);
}

void velri_context_destroy(velri_context *ctx) { velri_context_release(ctx); }

velri_status velri_device_alloc_tracked(velri_context *ctx, velri_device_id dev_id, size_t bytes,
                                        size_t align, void **out_handle) {
    velri_device_entry *de = &ctx->devices[dev_id];
    velri_status st = de->vt->alloc(de->dev, bytes, align, out_handle);
    if (st == VELRI_OK) {
        atomic_fetch_add_explicit(&ctx->total_allocs, 1U, memory_order_relaxed);
    }
    return st;
}

uint32_t velri_context_device_count(const velri_context *ctx) {
    return ctx ? ctx->num_devices : 0U;
}

velri_status velri_context_device_caps(const velri_context *ctx, velri_device_id device_id,
                                       velri_device_caps *out_caps) {
    if (!ctx || !out_caps || device_id >= ctx->num_devices) {
        return VELRI_ERR_INVALID;
    }
    *out_caps = ctx->devices[device_id].caps;
    return VELRI_OK;
}

velri_status velri_context_find_device(const velri_context *ctx, const char *backend_name,
                                       uint32_t local_index, velri_device_id *out_device_id) {
    if (!ctx || !backend_name || !out_device_id) {
        return VELRI_ERR_INVALID;
    }
    for (uint32_t i = 0U; i < ctx->num_devices; ++i) {
        const velri_device_entry *e = &ctx->devices[i];
        if (strcmp(e->vt->name, backend_name) == 0 && e->caps.local_index == local_index) {
            *out_device_id = i;
            return VELRI_OK;
        }
    }
    return VELRI_ERR_NOT_FOUND;
}
