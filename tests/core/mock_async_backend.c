#include "mock_async_backend.h"
#include "os.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static velri_mock_async_ctrl g_ctrl;

velri_mock_async_ctrl *velri_mock_async_control(void) { return &g_ctrl; }

void velri_mock_async_reset_ctrl(void) {
    atomic_store(&g_ctrl.delay_ms, 0U);
    atomic_store(&g_ctrl.fail_pre_accept, false);
    atomic_store(&g_ctrl.fail_post_accept, false);
    atomic_store(&g_ctrl.fail_partial_submit, false);
    atomic_store(&g_ctrl.in_flight, 0);
    atomic_store(&g_ctrl.total_submitted, 0U);
    atomic_store(&g_ctrl.total_completed, 0U);
    atomic_store(&g_ctrl.total_uploads, 0U);
    atomic_store(&g_ctrl.total_downloads, 0U);
}

typedef struct {
    uint8_t *bytes;
    size_t size;
} mock_buf;

struct velri_backend_device {
    uint32_t index;
};

struct velri_completion {
    velri_os_thread *thread;
    velri_status final_status;
    size_t count;
    float alpha;
    float beta;
    const float *a;
    const float *b;
    float *y;
    uint32_t delay_ms;
    bool inject_fail;
};

static void run_mock_job(void *raw) {
    struct velri_completion *c = (struct velri_completion *)raw;
    if (c->delay_ms > 0U) {
        velri_os_sleep_ms(c->delay_ms);
    }
    if (c->inject_fail) {
        if (c->y && c->count > 0U) {
            c->y[0] = -999.0f;
        }
        c->final_status = VELRI_ERR_EXEC;
        atomic_fetch_add(&g_ctrl.total_completed, 1U);
        atomic_fetch_sub(&g_ctrl.in_flight, 1);
        return;
    }

    for (size_t i = 0U; i < c->count; ++i) {
        float vb = c->b ? c->b[i] : 0.0f;
        c->y[i] = c->alpha * c->a[i] + c->beta * vb;
    }
    c->final_status = VELRI_OK;
    atomic_fetch_add(&g_ctrl.total_completed, 1U);
    atomic_fetch_sub(&g_ctrl.in_flight, 1);
}

static velri_status mock_enumerate_devices(uint32_t *out_count) {
    if (!out_count) return VELRI_ERR_INVALID;
    *out_count = 1U;
    return VELRI_OK;
}

static velri_status mock_open_device(uint32_t local_index, const void *cfg,
                                     velri_backend_device **out_dev) {
    (void)cfg;
    if (local_index != 0U || !out_dev) return VELRI_ERR_NOT_FOUND;
    velri_backend_device *dev = (velri_backend_device *)calloc(1U, sizeof(*dev));
    if (!dev) return VELRI_ERR_NOMEM;
    dev->index = local_index;
    *out_dev = dev;
    return VELRI_OK;
}

static velri_status mock_query_caps(const velri_backend_device *dev, velri_device_caps *out_caps) {
    if (!dev || !out_caps) return VELRI_ERR_INVALID;
    memset(out_caps, 0, sizeof(*out_caps));
    snprintf(out_caps->backend_name, sizeof(out_caps->backend_name), "%s", "mock_async");
    snprintf(out_caps->device_name, sizeof(out_caps->device_name), "%s", "Mock Async Device");
    out_caps->local_index = dev->index;
    out_caps->is_host_memory = false;
    out_caps->is_uma = false;
    out_caps->mem_total_known = true;
    out_caps->mem_total_bytes = 64ULL * 1024ULL * 1024ULL;
    out_caps->min_align_bytes = 64U;
    return VELRI_OK;
}

static void mock_close_device(velri_backend_device *dev) { free(dev); }

static velri_status mock_alloc(velri_backend_device *dev, size_t bytes, size_t align,
                               void **out_mem_handle) {
    (void)dev;
    if (!out_mem_handle || (align != 0U && (align & (align - 1U)) != 0U)) {
        return VELRI_ERR_INVALID;
    }
    mock_buf *mb = (mock_buf *)calloc(1U, sizeof(*mb));
    if (!mb) return VELRI_ERR_NOMEM;
    size_t eff = align < 64U ? 64U : align;
    size_t sz = bytes ? ((bytes + eff - 1U) & ~(eff - 1U)) : eff;
    mb->bytes = (uint8_t *)velri_os_aligned_alloc(eff, sz);
    if (!mb->bytes) {
        free(mb);
        return VELRI_ERR_NOMEM;
    }
    memset(mb->bytes, 0, sz);
    mb->size = sz;
    *out_mem_handle = mb;
    return VELRI_OK;
}

static void mock_free(velri_backend_device *dev, void *mem_handle) {
    (void)dev;
    mock_buf *mb = (mock_buf *)mem_handle;
    if (!mb) return;
    velri_os_aligned_free(mb->bytes);
    free(mb);
}

static velri_status mock_copy_to_ram(velri_backend_device *dev, void *dst_ram,
                                     const velri_buffer_binding *src, size_t bytes) {
    (void)dev;
    if (!dst_ram || !src || !src->mem_handle) return VELRI_ERR_INVALID;
    const mock_buf *mb = (const mock_buf *)src->mem_handle;
    if (src->byte_offset > mb->size || bytes > mb->size - src->byte_offset) {
        return VELRI_ERR_INVALID;
    }
    if (bytes > 0U) {
        memcpy(dst_ram, mb->bytes + src->byte_offset, bytes);
    }
    atomic_fetch_add(&g_ctrl.total_downloads, 1U);
    return VELRI_OK;
}

static velri_status mock_copy_from_ram(velri_backend_device *dev, const velri_buffer_binding *dst,
                                       const void *src_ram, size_t bytes) {
    (void)dev;
    if (!dst || !dst->mem_handle || !src_ram) return VELRI_ERR_INVALID;
    mock_buf *mb = (mock_buf *)dst->mem_handle;
    if (dst->byte_offset > mb->size || bytes > mb->size - dst->byte_offset) {
        return VELRI_ERR_INVALID;
    }
    if (bytes > 0U) {
        memcpy(mb->bytes + dst->byte_offset, src_ram, bytes);
    }
    atomic_fetch_add(&g_ctrl.total_uploads, 1U);
    return VELRI_OK;
}

static velri_status mock_zero_mem(velri_backend_device *dev, void *mem_handle, size_t bytes) {
    (void)dev;
    mock_buf *mb = (mock_buf *)mem_handle;
    if (!mb || bytes > mb->size) return VELRI_ERR_INVALID;
    if (bytes > 0U) memset(mb->bytes, 0, bytes);
    return VELRI_OK;
}

static velri_status mock_query_operation(const velri_backend_device *dev,
                                         const velri_resolved_op *op) {
    if (!dev || !op) return VELRI_ERR_INVALID;
    return (op->kind == VELRI_OP_VEC_AXPBY) ? VELRI_OK : VELRI_ERR_UNSUPPORTED;
}

static velri_status mock_submit(velri_backend_device *dev, const velri_submit_item *item,
                                velri_completion **out_completion) {
    if (!out_completion) return VELRI_ERR_INVALID;
    *out_completion = nullptr;
    if (!dev || !item || !item->op || item->num_inputs < 1U || item->num_outputs != 1U) {
        return VELRI_ERR_INVALID;
    }
    if (atomic_load(&g_ctrl.fail_pre_accept)) {
        return VELRI_ERR_EXEC;
    }

    struct velri_completion *c = (struct velri_completion *)calloc(1U, sizeof(*c));
    if (!c) return VELRI_ERR_NOMEM;

    const mock_buf *ba = (const mock_buf *)item->inputs[0].mem_handle;
    const mock_buf *bb = (item->num_inputs > 1U && item->op->input_present[1])
                             ? (const mock_buf *)item->inputs[1].mem_handle
                             : nullptr;
    mock_buf *by = (mock_buf *)item->outputs[0].mem_handle;

    size_t count = 0U;
    (void)velri_tensor_view_num_elements(&item->op->inputs[0], &count);
    c->count = count;
    c->alpha = item->op->raw.attr.axpby.alpha;
    c->beta = item->op->raw.attr.axpby.beta;
    c->a = (const float *)(ba->bytes + item->inputs[0].byte_offset);
    c->b = bb ? (const float *)(bb->bytes + item->inputs[1].byte_offset) : nullptr;
    c->y = (float *)(by->bytes + item->outputs[0].byte_offset);
    c->delay_ms = atomic_load(&g_ctrl.delay_ms);
    c->inject_fail = atomic_load(&g_ctrl.fail_post_accept);

    atomic_fetch_add(&g_ctrl.total_submitted, 1U);
    atomic_fetch_add(&g_ctrl.in_flight, 1);

    if (!velri_os_thread_spawn(&c->thread, run_mock_job, c)) {
        atomic_fetch_sub(&g_ctrl.in_flight, 1);
        free(c);
        return VELRI_ERR_NOMEM;
    }
    *out_completion = c;

    if (atomic_load(&g_ctrl.fail_partial_submit)) {
        return VELRI_ERR_EXEC;
    }
    return VELRI_OK;
}

static velri_status mock_wait(velri_backend_device *dev, velri_completion *comp) {
    (void)dev;
    if (!comp) return VELRI_ERR_INVALID;
    if (comp->thread) {
        velri_os_thread_join(comp->thread);
        comp->thread = nullptr;
    }
    return comp->final_status;
}

static void mock_destroy_completion(velri_backend_device *dev, velri_completion *comp) {
    (void)dev;
    free(comp);
}

static const velri_backend_vtable g_mock_vtable = {
    .name = "mock_async",
    .contract_version = VELRI_BACKEND_CONTRACT_VERSION,
    .enumerate_devices = mock_enumerate_devices,
    .open_device = mock_open_device,
    .query_caps = mock_query_caps,
    .close_device = mock_close_device,
    .alloc = mock_alloc,
    .free = mock_free,
    .copy_to_ram = mock_copy_to_ram,
    .copy_from_ram = mock_copy_from_ram,
    .zero_mem = mock_zero_mem,
    .query_operation = mock_query_operation,
    .prepare_op = nullptr,
    .destroy_prepared_op = nullptr,
    .submit = mock_submit,
    .submit_batch = nullptr,
    .wait = mock_wait,
    .destroy_completion = mock_destroy_completion,
};

const velri_backend_vtable *velri_mock_async_backend_vtable(void) { return &g_mock_vtable; }
