#include "blocks_internal.h"

#include <stdlib.h>
#include <string.h>

/* ---- Internal Helpers ---- */

static inline velri_op_desc init_op_desc(velri_op_kind kind, uint32_t num_inputs,
                                         uint32_t num_outputs) {
    velri_op_desc op;
    memset(&op, 0, sizeof(op));
    op.kind = kind;
    op.num_inputs = num_inputs;
    op.num_outputs = num_outputs;
    for (uint32_t i = 0U; i < VELRI_OP_MAX_INPUTS; ++i) {
        op.inputs[i] = VELRI_INVALID_ID;
    }
    for (uint32_t i = 0U; i < VELRI_OP_MAX_OUTPUTS; ++i) {
        op.outputs[i] = VELRI_INVALID_ID;
    }
    return op;
}

static size_t current_scope_path_len(const velri_builder *b) {
    size_t len = 0U;
    for (uint32_t i = 0U; i < b->scope_depth; ++i) {
        if (i > 0U) {
            len += 1U;
        }
        len += strlen(b->scopes[i]);
    }
    return len;
}

static void format_scope_path(const velri_builder *b, char out_path[VELRI_BUILDER_MAX_SCOPE_PATH]) {
    size_t pos = 0U;
    out_path[0] = '\0';
    for (uint32_t i = 0U; i < b->scope_depth; ++i) {
        if (i > 0U && pos + 1U < VELRI_BUILDER_MAX_SCOPE_PATH) {
            out_path[pos++] = '/';
        }
        const char *seg = b->scopes[i];
        size_t slen = strlen(seg);
        size_t avail = (VELRI_BUILDER_MAX_SCOPE_PATH - 1U) - pos;
        if (slen > avail) {
            slen = avail;
        }
        if (slen > 0U) {
            memcpy(out_path + pos, seg, slen);
            pos += slen;
        }
    }
    out_path[pos] = '\0';
}

static const velri_manifest_entry *manifest_find_entry(const velri_manifest_entry *entries,
                                                       uint32_t count, const char *name) {
    if (!entries || !name || name[0] == '\0') {
        return nullptr;
    }
    for (uint32_t i = 0U; i < count; ++i) {
        if (strcmp(entries[i].name, name) == 0) {
            return &entries[i];
        }
    }
    return nullptr;
}

static velri_tensor_view view_from_tensor(const velri_tensor *t) {
    velri_tensor_view v;
    memset(&v, 0, sizeof(v));
    v.storage_id = t->storage_id;
    v.dtype = t->dtype;
    v.rank = t->rank;
    for (uint32_t d = 0U; d < t->rank; ++d) {
        v.dims[d] = t->dims[d];
        v.strides[d] = t->strides[d];
    }
    v.byte_offset = t->byte_offset;
    return v;
}

static velri_tensor tensor_from_view(const velri_tensor_view *v, velri_view_id vid,
                                     velri_device_id device_id, bool is_seq_dynamic) {
    velri_tensor t = VELRI_TENSOR_NONE;
    t.storage_id = v->storage_id;
    t.view_id = vid;
    t.dtype = v->dtype;
    t.rank = v->rank;
    for (uint32_t d = 0U; d < v->rank; ++d) {
        t.dims[d] = v->dims[d];
        t.strides[d] = v->strides[d];
    }
    t.byte_offset = v->byte_offset;
    t.device_id = device_id;
    t.is_seq_dynamic = is_seq_dynamic;
    t.valid = true;
    return t;
}

static velri_tensor alloc_tensor(velri_builder *b, velri_storage_kind kind,
                                 velri_device_id device_id, velri_dtype dtype, uint32_t rank,
                                 const size_t *dims) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!b->graph || !dims || rank == 0U || rank > VELRI_MAX_RANK) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }

    size_t elem_bytes = velri_dtype_bytes(dtype);
    if (elem_bytes == 0U) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }

    size_t num_elems = 1U;
    for (uint32_t d = 0U; d < rank; ++d) {
        if (dims[d] == 0U) {
            velri_b_set_error(b, VELRI_ERR_INVALID);
            return VELRI_TENSOR_NONE;
        }
        if (velri_mul_overflow_size(num_elems, dims[d], &num_elems)) {
            velri_b_set_error(b, VELRI_ERR_OVERFLOW);
            return VELRI_TENSOR_NONE;
        }
    }
    size_t byte_size = 0U;
    if (velri_mul_overflow_size(num_elems, elem_bytes, &byte_size)) {
        velri_b_set_error(b, VELRI_ERR_OVERFLOW);
        return VELRI_TENSOR_NONE;
    }

    velri_storage_desc sdesc = {
        .kind = kind,
        .dtype = dtype,
        .byte_size = byte_size,
        .align = 64U,
        .device_id = device_id,
    };
    velri_storage_id sid = VELRI_INVALID_ID;
    velri_status st = velri_graph_add_storage(b->graph, &sdesc, &sid);
    if (st != VELRI_OK) {
        velri_b_set_error(b, st);
        return VELRI_TENSOR_NONE;
    }

    velri_tensor_view tv;
    st = velri_tensor_view_contiguous(sid, dtype, rank, dims, 0U, &tv);
    if (st != VELRI_OK) {
        velri_b_set_error(b, st);
        return VELRI_TENSOR_NONE;
    }

    velri_view_id vid = VELRI_INVALID_ID;
    st = velri_graph_add_view(b->graph, &tv, &vid);
    if (st != VELRI_OK) {
        velri_b_set_error(b, st);
        return VELRI_TENSOR_NONE;
    }

    return tensor_from_view(&tv, vid, device_id, false);
}

static velri_tensor alloc_compute_tensor(velri_builder *b, velri_storage_kind kind,
                                         velri_dtype dtype, uint32_t rank, const size_t *dims) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    velri_device_id compute_dev = 0U;
    if (velri_b_resolve_devices(b, nullptr, &compute_dev) != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    return alloc_tensor(b, kind, compute_dev, dtype, rank, dims);
}

static velri_tensor make_subview(velri_builder *b, velri_tensor base, uint32_t rank,
                                 const size_t *sub_dims, const size_t *sub_strides,
                                 size_t elem_offset, bool is_seq_dynamic) {
    size_t elem_bytes = velri_dtype_bytes(base.dtype);
    if (elem_bytes == 0U) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    size_t extra_byte_offset = 0U;
    if (velri_mul_overflow_size(elem_offset, elem_bytes, &extra_byte_offset)) {
        velri_b_set_error(b, VELRI_ERR_OVERFLOW);
        return VELRI_TENSOR_NONE;
    }

    velri_tensor_view base_view = view_from_tensor(&base);
    velri_tensor_view sub_view;
    velri_status st = velri_tensor_view_subview(&base_view, rank, sub_dims, sub_strides,
                                                extra_byte_offset, &sub_view);
    if (st != VELRI_OK) {
        velri_b_set_error(b, st);
        return VELRI_TENSOR_NONE;
    }

    velri_view_id vid = VELRI_INVALID_ID;
    st = velri_graph_add_view(b->graph, &sub_view, &vid);
    if (st != VELRI_OK) {
        velri_b_set_error(b, st);
        return VELRI_TENSOR_NONE;
    }

    return tensor_from_view(&sub_view, vid, base.device_id, is_seq_dynamic);
}

static bool emit_op_on(velri_builder *b, const velri_op_desc *op, velri_device_id dev) {
    velri_status st = velri_graph_add_op(b->graph, op, dev, nullptr);
    if (st != VELRI_OK) {
        velri_b_set_error(b, st);
        return false;
    }
    return true;
}

static bool emit_op(velri_builder *b, const velri_op_desc *op) {
    velri_device_id compute_dev = 0U;
    return velri_b_resolve_devices(b, nullptr, &compute_dev) == VELRI_OK &&
           emit_op_on(b, op, compute_dev);
}

static bool same_shape(const velri_tensor *a, const velri_tensor *b) {
    if (a->rank != b->rank || a->rank == 0U || a->rank > VELRI_MAX_RANK) {
        return false;
    }
    for (uint32_t d = 0U; d < a->rank; ++d) {
        if (a->dims[d] != b->dims[d]) {
            return false;
        }
    }
    return true;
}

/* ---- Named Weight & State Manifest ---- */

void velri_manifest_init(velri_manifest *m) {
    if (!m) {
        return;
    }
    memset(m, 0, sizeof(*m));
    m->token_ids_in = VELRI_INVALID_ID;
    m->logits_out = VELRI_INVALID_ID;
}

void velri_manifest_destroy(velri_manifest *m) {
    if (!m) {
        return;
    }
    free(m->weights);
    free(m->states);
    velri_manifest_init(m);
}

const velri_manifest_entry *velri_manifest_find_weight(const velri_manifest *m, const char *name) {
    return m ? manifest_find_entry(m->weights, m->num_weights, name) : nullptr;
}

const velri_manifest_entry *velri_manifest_find_state(const velri_manifest *m, const char *name) {
    return m ? manifest_find_entry(m->states, m->num_states, name) : nullptr;
}

/* ---- Lifecycle & Sticky Error Handling ---- */

void velri_b_init(velri_builder *b, velri_graph *graph, velri_manifest *manifest,
                  velri_device_id default_store_dev, velri_device_id default_compute_dev,
                  velri_placement_resolve_fn resolve_fn, void *resolve_user) {
    if (!b) {
        return;
    }
    memset(b, 0, sizeof(*b));
    b->graph = graph;
    b->manifest = manifest;
    b->err = graph ? VELRI_OK : VELRI_ERR_INVALID;
    b->default_store_dev = default_store_dev;
    b->default_compute_dev = default_compute_dev;
    b->resolve_fn = resolve_fn;
    b->resolve_user = resolve_user;
    b->current_layer = -1;
}

void velri_b_init_checked(velri_builder *b, velri_graph *graph, velri_manifest *manifest,
                          velri_device_id default_store_dev, velri_device_id default_compute_dev,
                          velri_placement_resolve_checked_fn resolve_fn, void *resolve_user) {
    velri_b_init(b, graph, manifest, default_store_dev, default_compute_dev, nullptr, resolve_user);
    if (b) {
        b->resolve_checked_fn = resolve_fn;
    }
}

velri_status velri_b_status(const velri_builder *b) {
    return b ? b->err : VELRI_ERR_INVALID;
}

void velri_b_set_error(velri_builder *b, velri_status err) {
    if (b && b->err == VELRI_OK && err != VELRI_OK) {
        b->err = err;
    }
}

/* ---- Scope Stack & Placement Resolution ---- */

void velri_b_push_scope(velri_builder *b, const char *name) {
    if (!b) {
        return;
    }
    if (b->err != VELRI_OK) {
        b->skipped_scope_pushes++;
        return;
    }
    if (!name || name[0] == '\0') {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return;
    }
    if (b->scope_depth >= VELRI_BUILDER_MAX_SCOPE_DEPTH) {
        velri_b_set_error(b, VELRI_ERR_OVERFLOW);
        return;
    }
    size_t cur_len = current_scope_path_len(b);
    size_t add_len = strlen(name) + (b->scope_depth > 0U ? 1U : 0U);
    if (add_len >= VELRI_BUILDER_MAX_SCOPE_PATH ||
        cur_len >= VELRI_BUILDER_MAX_SCOPE_PATH - add_len) {
        velri_b_set_error(b, VELRI_ERR_OVERFLOW);
        return;
    }
    b->scopes[b->scope_depth++] = name;
}

void velri_b_pop_scope(velri_builder *b) {
    if (!b) {
        return;
    }
    if (b->skipped_scope_pushes > 0U) {
        b->skipped_scope_pushes--;
        return;
    }
    if (b->scope_depth == 0U) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return;
    }
    b->scope_depth--;
    b->scopes[b->scope_depth] = nullptr;
}

void velri_b_push_layer(velri_builder *b, int32_t layer_idx) {
    if (!b) {
        return;
    }
    if (b->err != VELRI_OK) {
        b->skipped_layer_pushes++;
        return;
    }
    if (layer_idx < 0) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return;
    }
    if (b->layer_depth >= VELRI_BUILDER_MAX_SCOPE_DEPTH) {
        velri_b_set_error(b, VELRI_ERR_OVERFLOW);
        return;
    }
    b->layer_stack[b->layer_depth++] = b->current_layer;
    b->current_layer = layer_idx;
}

void velri_b_pop_layer(velri_builder *b) {
    if (!b) {
        return;
    }
    if (b->skipped_layer_pushes > 0U) {
        b->skipped_layer_pushes--;
        return;
    }
    if (b->layer_depth == 0U) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return;
    }
    b->layer_depth--;
    b->current_layer = b->layer_stack[b->layer_depth];
    b->layer_stack[b->layer_depth] = 0;
}

static velri_status current_devices(const velri_builder *b, velri_device_id *out_store,
                                    velri_device_id *out_compute) {
    velri_device_id store = b ? b->default_store_dev : 0U;
    velri_device_id compute = b ? b->default_compute_dev : 0U;
    velri_status st = velri_b_status(b);

    if (st == VELRI_OK && (b->resolve_fn || b->resolve_checked_fn)) {
        char path[VELRI_BUILDER_MAX_SCOPE_PATH];
        format_scope_path(b, path);
        if (b->resolve_checked_fn) {
            st = b->resolve_checked_fn(path, b->current_layer, b->resolve_user,
                                       out_store ? &store : nullptr, out_compute ? &compute : nullptr);
            if (st != VELRI_OK) {
                store = b->default_store_dev;
                compute = b->default_compute_dev;
            }
        } else {
            b->resolve_fn(path, b->current_layer, b->resolve_user, &store, &compute);
            st = velri_b_status(b);
        }
    }
    if (out_store) {
        *out_store = store;
    }
    if (out_compute) {
        *out_compute = compute;
    }
    return st;
}

void velri_b_devices(const velri_builder *b, velri_device_id *out_store,
                     velri_device_id *out_compute) {
    (void)current_devices(b, out_store, out_compute);
}

velri_status velri_b_resolve_devices(velri_builder *b, velri_device_id *out_store,
                                     velri_device_id *out_compute) {
    velri_status st = current_devices(b, out_store, out_compute);
    velri_b_set_error(b, st);
    return st;
}

/* ---- Tensor Constructors & Named Registration ---- */

/* Each named WEIGHT or STATE has its own namespace. Replicas may additionally name
 * their logical source. Reserve the manifest entry before adding anything to the graph. */
static velri_tensor alloc_registered_tensor(velri_builder *b, velri_storage_kind kind,
                                            velri_device_id store_dev, const char *name,
                                            const char *source_name, velri_dtype dtype,
                                            uint32_t rank, const size_t *dims) {
    if (!name) {
        return alloc_tensor(b, kind, store_dev, dtype, rank, dims);
    }
    size_t name_len = strlen(name);
    size_t source_len = source_name ? strlen(source_name) : 0U;
    if (name_len == 0U || !dims || rank == 0U || rank > VELRI_MAX_RANK) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    if (name_len >= VELRI_MANIFEST_NAME_MAX || source_len >= VELRI_MANIFEST_NAME_MAX) {
        velri_b_set_error(b, VELRI_ERR_OVERFLOW);
        return VELRI_TENSOR_NONE;
    }

    velri_manifest *m = b->manifest;
    if (!m) {
        return alloc_tensor(b, kind, store_dev, dtype, rank, dims);
    }
    bool is_weight = kind == VELRI_STORAGE_WEIGHT;
    velri_manifest_entry **entries = is_weight ? &m->weights : &m->states;
    uint32_t *count = is_weight ? &m->num_weights : &m->num_states;
    uint32_t *cap = is_weight ? &m->cap_weights : &m->cap_states;
    if (manifest_find_entry(*entries, *count, name)) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    if (*count == UINT32_MAX) {
        velri_b_set_error(b, VELRI_ERR_OVERFLOW);
        return VELRI_TENSOR_NONE;
    }

    /* Names and dimensions may have been borrowed from this manifest before realloc. */
    velri_manifest_entry entry = {0};
    memcpy(entry.name, name, name_len + 1U);
    memcpy(entry.dims, dims, rank * sizeof(dims[0]));
    if (source_name) {
        memcpy(entry.source_name, source_name, source_len + 1U);
    }
    void *new_entries = nullptr;
    if (!velri_vec_ensure(*entries, &new_entries, cap, *count, 1U, sizeof(entry))) {
        velri_b_set_error(b, VELRI_ERR_NOMEM);
        return VELRI_TENSOR_NONE;
    }
    *entries = new_entries;

    velri_tensor t = alloc_tensor(b, kind, store_dev, dtype, rank, entry.dims);
    if (t.valid) {
        entry.storage_id = t.storage_id;
        entry.view_id = t.view_id;
        entry.dtype = t.dtype;
        entry.rank = t.rank;
        entry.device_id = t.device_id;
        (*entries)[(*count)++] = entry;
    }
    return t;
}

static velri_tensor placed_storage(velri_builder *b, velri_storage_kind kind, const char *name,
                                   velri_dtype dtype, uint32_t rank, const size_t *dims) {
    velri_device_id store_dev;
    if (velri_b_resolve_devices(b, &store_dev, nullptr) != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    return alloc_registered_tensor(b, kind, store_dev, name, nullptr, dtype, rank, dims);
}

velri_tensor velri_b_weight(velri_builder *b, const char *name, velri_dtype dtype, uint32_t rank,
                            const size_t *dims) {
    return placed_storage(b, VELRI_STORAGE_WEIGHT, name, dtype, rank, dims);
}

velri_tensor velri_b_weight_shared(velri_builder *b, const char *name, velri_dtype dtype,
                                   uint32_t rank, const size_t *dims) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!b->manifest || !name || name[0] == '\0' || !dims || rank == 0U ||
        rank > VELRI_MAX_RANK) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    size_t nlen = strlen(name);
    if (nlen >= VELRI_MANIFEST_NAME_MAX) {
        velri_b_set_error(b, VELRI_ERR_OVERFLOW);
        return VELRI_TENSOR_NONE;
    }

    velri_device_id store_dev = 0U;
    if (velri_b_resolve_devices(b, &store_dev, nullptr) != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }

    const velri_manifest_entry *any_match = nullptr;
    const velri_manifest_entry *dev_match = nullptr;
    for (uint32_t i = 0U; i < b->manifest->num_weights; ++i) {
        const velri_manifest_entry *e = &b->manifest->weights[i];
        bool same_logical =
            (strcmp(e->name, name) == 0) ||
            (e->source_name[0] != '\0' && strcmp(e->source_name, name) == 0);
        if (!same_logical) {
            continue;
        }
        if (!any_match) {
            any_match = e;
        }
        if (e->device_id == store_dev) {
            dev_match = e;
            break;
        }
    }

    if (any_match) {
        if (any_match->dtype != dtype || any_match->rank != rank) {
            velri_b_set_error(b, VELRI_ERR_INVALID);
            return VELRI_TENSOR_NONE;
        }
        for (uint32_t d = 0U; d < rank; ++d) {
            if (any_match->dims[d] != dims[d]) {
                velri_b_set_error(b, VELRI_ERR_INVALID);
                return VELRI_TENSOR_NONE;
            }
        }
    }

    if (dev_match) {
        velri_tensor_view tv;
        velri_status st = velri_tensor_view_contiguous(dev_match->storage_id, dev_match->dtype,
                                                       dev_match->rank, dev_match->dims, 0U, &tv);
        if (st != VELRI_OK) {
            velri_b_set_error(b, st);
            return VELRI_TENSOR_NONE;
        }
        return tensor_from_view(&tv, dev_match->view_id, dev_match->device_id, false);
    }

    if (!any_match) {
        return alloc_registered_tensor(b, VELRI_STORAGE_WEIGHT, store_dev, name, nullptr, dtype,
                                       rank, dims);
    }

    /* Cross-device placement: allocate one per-device replica aliased to `source_name = name`. */
    char alias[VELRI_MANIFEST_NAME_MAX];
    int n = snprintf(alias, sizeof(alias), "%s@dev%u", name, (unsigned)store_dev);
    if (n <= 0 || (size_t)n >= sizeof(alias)) {
        velri_b_set_error(b, VELRI_ERR_OVERFLOW);
        return VELRI_TENSOR_NONE;
    }
    return alloc_registered_tensor(b, VELRI_STORAGE_WEIGHT, store_dev, alias, name, dtype, rank, dims);
}

velri_tensor velri_b_weight_tied(velri_builder *b, const char *name, const char *tie_to,
                                 velri_dtype dtype, uint32_t rank, const size_t *dims) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!tie_to) {
        return velri_b_weight(b, name, dtype, rank, dims);
    }
    if (!b->manifest || !dims || rank == 0U || rank > VELRI_MAX_RANK || tie_to[0] == '\0' ||
        (name && name[0] == '\0')) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    if ((name && strlen(name) >= VELRI_MANIFEST_NAME_MAX) ||
        strlen(tie_to) >= VELRI_MANIFEST_NAME_MAX) {
        velri_b_set_error(b, VELRI_ERR_OVERFLOW);
        return VELRI_TENSOR_NONE;
    }

    const velri_manifest_entry *src = velri_manifest_find_weight(b->manifest, tie_to);
    if (!src) {
        velri_b_set_error(b, VELRI_ERR_NOT_FOUND);
        return VELRI_TENSOR_NONE;
    }
    if (src->dtype != dtype || src->rank != rank) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    for (uint32_t d = 0U; d < rank; ++d) {
        if (src->dims[d] != dims[d]) {
            velri_b_set_error(b, VELRI_ERR_INVALID);
            return VELRI_TENSOR_NONE;
        }
    }

    velri_device_id store_dev = 0U;
    if (velri_b_resolve_devices(b, &store_dev, nullptr) != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (src->device_id == store_dev) {
        velri_tensor_view tv;
        velri_status st = velri_tensor_view_contiguous(src->storage_id, src->dtype, src->rank,
                                                       src->dims, 0U, &tv);
        if (st != VELRI_OK) {
            velri_b_set_error(b, st);
            return VELRI_TENSOR_NONE;
        }
        return tensor_from_view(&tv, src->view_id, src->device_id, false);
    }

    /* Cross-device placement requires a distinct logical alias name for the replicated storage. */
    if (!name) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    return alloc_registered_tensor(b, VELRI_STORAGE_WEIGHT, store_dev, name, tie_to, dtype, rank, dims);
}

velri_tensor velri_b_input(velri_builder *b, velri_dtype dtype, uint32_t rank, const size_t *dims) {
    return alloc_compute_tensor(b, VELRI_STORAGE_INPUT, dtype, rank, dims);
}

velri_tensor velri_b_output(velri_builder *b, velri_dtype dtype, uint32_t rank,
                            const size_t *dims) {
    return alloc_compute_tensor(b, VELRI_STORAGE_OUTPUT, dtype, rank, dims);
}

velri_tensor velri_b_state(velri_builder *b, const char *name, velri_dtype dtype, uint32_t rank,
                           const size_t *dims) {
    return placed_storage(b, VELRI_STORAGE_STATE, name, dtype, rank, dims);
}

velri_tensor velri_b_scratch(velri_builder *b, velri_dtype dtype, uint32_t rank,
                             const size_t *dims) {
    return alloc_compute_tensor(b, VELRI_STORAGE_SCRATCH, dtype, rank, dims);
}

velri_tensor velri_b_subview_2d_cols(velri_builder *b, velri_tensor base, size_t col_offset,
                                     size_t num_cols) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!b->graph || !base.valid || base.rank != 2U || base.strides[1] != 1U || num_cols == 0U ||
        col_offset > base.dims[1] || num_cols > base.dims[1] - col_offset) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    const size_t sub_dims[2] = {base.dims[0], num_cols};
    const size_t sub_strides[2] = {base.strides[0], 1U};
    return make_subview(b, base, 2U, sub_dims, sub_strides, col_offset, base.is_seq_dynamic);
}

velri_tensor velri_b_subview_2d_rows(velri_builder *b, velri_tensor base, size_t row_offset,
                                     size_t num_rows) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!b->graph || !base.valid || base.rank != 2U || num_rows == 0U ||
        row_offset > base.dims[0] || num_rows > base.dims[0] - row_offset) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    size_t elem_offset = 0U;
    if (velri_mul_overflow_size(row_offset, base.strides[0], &elem_offset)) {
        velri_b_set_error(b, VELRI_ERR_OVERFLOW);
        return VELRI_TENSOR_NONE;
    }
    const size_t sub_dims[2] = {num_rows, base.dims[1]};
    const size_t sub_strides[2] = {base.strides[0], base.strides[1]};
    bool dyn = (row_offset == 0U && num_rows == base.dims[0]) && base.is_seq_dynamic;
    return make_subview(b, base, 2U, sub_dims, sub_strides, elem_offset, dyn);
}

velri_tensor velri_b_subview_1d(velri_builder *b, velri_tensor base, size_t offset, size_t len) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!b->graph || !base.valid || base.rank != 1U || base.strides[0] != 1U || len == 0U ||
        offset > base.dims[0] || len > base.dims[0] - offset) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    const size_t sub_dims[1] = {len};
    const size_t sub_strides[1] = {1U};
    bool dyn = (offset == 0U && len == base.dims[0]) && base.is_seq_dynamic;
    return make_subview(b, base, 1U, sub_dims, sub_strides, offset, dyn);
}

velri_tensor velri_b_reshape_2d_to_3d(velri_builder *b, velri_tensor base, size_t d1, size_t d2) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!b->graph || !base.valid || base.rank != 2U || base.strides[1] != 1U || d1 == 0U ||
        d2 == 0U) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    size_t prod = 0U;
    if (velri_mul_overflow_size(d1, d2, &prod) || prod != base.dims[1]) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    const size_t sub_dims[3] = {base.dims[0], d1, d2};
    const size_t sub_strides[3] = {base.strides[0], d2, 1U};
    return make_subview(b, base, 3U, sub_dims, sub_strides, 0U, base.is_seq_dynamic);
}

velri_tensor velri_b_reshape_3d_to_2d(velri_builder *b, velri_tensor base) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!b->graph || !base.valid || base.rank != 3U || base.strides[2] != 1U ||
        base.strides[1] != base.dims[2]) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    size_t cols = 0U;
    if (velri_mul_overflow_size(base.dims[1], base.dims[2], &cols)) {
        velri_b_set_error(b, VELRI_ERR_OVERFLOW);
        return VELRI_TENSOR_NONE;
    }
    const size_t sub_dims[2] = {base.dims[0], cols};
    const size_t sub_strides[2] = {base.strides[0], 1U};
    return make_subview(b, base, 2U, sub_dims, sub_strides, 0U, base.is_seq_dynamic);
}

/* ---- Primitive Operation Builders ---- */

velri_tensor velri_b_embedding(velri_builder *b, velri_tensor table, velri_tensor ids_or_none,
                               size_t pos_offset, size_t max_seq_len) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!table.valid || table.rank != 2U) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }

    size_t seq_len = max_seq_len;
    if (ids_or_none.valid) {
        if (ids_or_none.rank != 1U) {
            velri_b_set_error(b, VELRI_ERR_INVALID);
            return VELRI_TENSOR_NONE;
        }
        if (seq_len == 0U) {
            seq_len = ids_or_none.dims[0];
        } else if (seq_len != ids_or_none.dims[0]) {
            velri_b_set_error(b, VELRI_ERR_INVALID);
            return VELRI_TENSOR_NONE;
        }
    } else if (seq_len == 0U) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }

    const size_t out_dims[2] = {seq_len, table.dims[1]};
    velri_tensor out = velri_b_scratch(b, table.dtype, 2U, out_dims);
    if (!out.valid) {
        return VELRI_TENSOR_NONE;
    }
    out.is_seq_dynamic = true;

    velri_op_desc op = init_op_desc(VELRI_OP_EMBEDDING, ids_or_none.valid ? 2U : 1U, 1U);
    op.dynamic_seq = true;
    op.inputs[0] = table.view_id;
    if (ids_or_none.valid) {
        op.inputs[1] = ids_or_none.view_id;
    }
    op.outputs[0] = out.view_id;
    op.attr.embedding.pos_offset = pos_offset;

    return emit_op(b, &op) ? out : VELRI_TENSOR_NONE;
}

velri_tensor velri_b_linear_into(velri_builder *b, velri_tensor x, velri_tensor w,
                                 velri_tensor bias_or_none, velri_tensor out) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!x.valid || !w.valid || !out.valid || x.rank != 2U || w.rank != 2U || out.rank != 2U ||
        x.dims[1] != w.dims[1] || out.dims[0] != x.dims[0] || out.dims[1] != w.dims[0]) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }

    velri_op_desc op = init_op_desc(VELRI_OP_LINEAR, bias_or_none.valid ? 3U : 2U, 1U);
    op.dynamic_seq = x.is_seq_dynamic;
    op.inputs[0] = x.view_id;
    op.inputs[1] = w.view_id;
    if (bias_or_none.valid) {
        op.inputs[2] = bias_or_none.view_id;
    }
    op.outputs[0] = out.view_id;
    op.attr.linear.allow_f16_matmul = b->allow_f16_matmul;
    op.attr.linear.desc = (velri_linear_desc){
        .x_dtype = x.dtype,
        .w_dtype = w.dtype,
        .y_dtype = out.dtype,
        .has_x_scale = false,
        .has_w_scale = false,
    };

    out.is_seq_dynamic = x.is_seq_dynamic;
    return emit_op(b, &op) ? out : VELRI_TENSOR_NONE;
}

velri_tensor velri_b_linear(velri_builder *b, velri_tensor x, velri_tensor w,
                            velri_tensor bias_or_none) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!x.valid || !w.valid || x.rank != 2U || w.rank != 2U || x.dims[1] != w.dims[1]) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }

    velri_dtype y_dtype = velri_blocks_linear_out_dtype(x.dtype, w.dtype);
    const size_t out_dims[2] = {x.dims[0], w.dims[0]};
    velri_tensor out = velri_b_scratch(b, y_dtype, 2U, out_dims);
    if (!out.valid) {
        return VELRI_TENSOR_NONE;
    }
    return velri_b_linear_into(b, x, w, bias_or_none, out);
}

velri_tensor velri_b_layernorm(velri_builder *b, velri_tensor x, velri_tensor w,
                               velri_tensor b_opt, float eps) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!x.valid || x.rank != 2U) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }

    velri_tensor out = velri_b_scratch(b, x.dtype, 2U, x.dims);
    if (!out.valid) {
        return VELRI_TENSOR_NONE;
    }
    out.is_seq_dynamic = x.is_seq_dynamic;

    uint32_t num_inputs = b_opt.valid ? 3U : (w.valid ? 2U : 1U);
    velri_op_desc op = init_op_desc(VELRI_OP_LAYERNORM, num_inputs, 1U);
    op.dynamic_seq = x.is_seq_dynamic;
    op.inputs[0] = x.view_id;
    if (w.valid) {
        op.inputs[1] = w.view_id;
    }
    if (b_opt.valid) {
        op.inputs[2] = b_opt.view_id;
    }
    op.outputs[0] = out.view_id;
    op.attr.layernorm.eps = eps;

    return emit_op(b, &op) ? out : VELRI_TENSOR_NONE;
}

velri_tensor velri_b_rmsnorm(velri_builder *b, velri_tensor x, velri_tensor w, float eps,
                             bool unit_offset) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!x.valid || x.rank != 2U) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }

    velri_tensor out = velri_b_scratch(b, x.dtype, 2U, x.dims);
    if (!out.valid) {
        return VELRI_TENSOR_NONE;
    }
    out.is_seq_dynamic = x.is_seq_dynamic;

    velri_op_desc op = init_op_desc(VELRI_OP_RMSNORM, w.valid ? 2U : 1U, 1U);
    op.dynamic_seq = x.is_seq_dynamic;
    op.inputs[0] = x.view_id;
    if (w.valid) {
        op.inputs[1] = w.view_id;
    }
    op.outputs[0] = out.view_id;
    op.attr.rmsnorm.eps = eps;
    op.attr.rmsnorm.unit_offset = unit_offset;

    return emit_op(b, &op) ? out : VELRI_TENSOR_NONE;
}

static velri_tensor emit_rope(velri_builder *b, velri_tensor x, velri_tensor cos_tbl,
                              velri_tensor sin_tbl, size_t rotary_dim, size_t pos_offset,
                              velri_rope_mode mode, bool inplace) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!x.valid || !cos_tbl.valid || !sin_tbl.valid || x.rank != 3U || cos_tbl.rank != 2U ||
        sin_tbl.rank != 2U) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }

    velri_device_id compute_dev = 0U;
    if (velri_b_resolve_devices(b, nullptr, &compute_dev) != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    velri_tensor out = x;
    if (!inplace || x.device_id != compute_dev) {
        out = alloc_tensor(b, VELRI_STORAGE_SCRATCH, compute_dev, x.dtype, 3U, x.dims);
        if (!out.valid) {
            return VELRI_TENSOR_NONE;
        }
    }
    out.is_seq_dynamic = x.is_seq_dynamic;

    velri_op_desc op = init_op_desc(VELRI_OP_ROPE, 3U, 1U);
    op.dynamic_seq = x.is_seq_dynamic;
    op.inputs[0] = x.view_id;
    op.inputs[1] = cos_tbl.view_id;
    op.inputs[2] = sin_tbl.view_id;
    op.outputs[0] = out.view_id;
    op.attr.rope.rotary_dim = (rotary_dim > 0U) ? rotary_dim : x.dims[2];
    op.attr.rope.pos_offset = pos_offset;
    op.attr.rope.mode = mode;

    return emit_op_on(b, &op, compute_dev) ? out : VELRI_TENSOR_NONE;
}

velri_tensor velri_b_rope(velri_builder *b, velri_tensor x, velri_tensor cos_tbl,
                          velri_tensor sin_tbl, size_t rotary_dim, size_t pos_offset,
                          velri_rope_mode mode) {
    return emit_rope(b, x, cos_tbl, sin_tbl, rotary_dim, pos_offset, mode, false);
}

velri_tensor velri_b_rope_inplace(velri_builder *b, velri_tensor x, velri_tensor cos_tbl,
                                  velri_tensor sin_tbl, size_t rotary_dim, size_t pos_offset,
                                  velri_rope_mode mode) {
    return emit_rope(b, x, cos_tbl, sin_tbl, rotary_dim, pos_offset, mode, true);
}

static velri_tensor emit_activation(velri_builder *b, velri_tensor x, velri_act_kind act_kind,
                                    bool inplace) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!x.valid || x.rank == 0U || x.rank > VELRI_MAX_RANK) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }

    velri_device_id compute_dev = 0U;
    if (velri_b_resolve_devices(b, nullptr, &compute_dev) != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    velri_tensor out = x;
    if (!inplace || x.device_id != compute_dev) {
        out = alloc_tensor(b, VELRI_STORAGE_SCRATCH, compute_dev, x.dtype, x.rank, x.dims);
        if (!out.valid) {
            return VELRI_TENSOR_NONE;
        }
    }
    out.is_seq_dynamic = x.is_seq_dynamic;

    velri_op_desc op = init_op_desc(VELRI_OP_ACTIVATION, 1U, 1U);
    op.dynamic_seq = x.is_seq_dynamic;
    op.inputs[0] = x.view_id;
    op.outputs[0] = out.view_id;
    op.attr.activation.kind = act_kind;

    return emit_op_on(b, &op, compute_dev) ? out : VELRI_TENSOR_NONE;
}

velri_tensor velri_b_activation(velri_builder *b, velri_tensor x, velri_act_kind act_kind) {
    return emit_activation(b, x, act_kind, false);
}

velri_tensor velri_b_activation_inplace(velri_builder *b, velri_tensor x, velri_act_kind act_kind) {
    return emit_activation(b, x, act_kind, true);
}

static velri_tensor emit_gated_mlp_act(velri_builder *b, velri_tensor gate, velri_tensor up,
                                       velri_act_kind act_kind, float clamp_limit, bool inplace) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!gate.valid || !up.valid || !same_shape(&gate, &up)) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }

    velri_device_id compute_dev = 0U;
    if (velri_b_resolve_devices(b, nullptr, &compute_dev) != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    velri_tensor out = gate;
    if (!inplace || gate.device_id != compute_dev) {
        out = alloc_tensor(b, VELRI_STORAGE_SCRATCH, compute_dev, gate.dtype, gate.rank, gate.dims);
        if (!out.valid) {
            return VELRI_TENSOR_NONE;
        }
    }
    bool dyn = gate.is_seq_dynamic || up.is_seq_dynamic;
    out.is_seq_dynamic = dyn;

    velri_op_desc op = init_op_desc(VELRI_OP_GATED_MLP, 2U, 1U);
    op.dynamic_seq = dyn;
    op.inputs[0] = gate.view_id;
    op.inputs[1] = up.view_id;
    op.outputs[0] = out.view_id;
    op.attr.gated_mlp.act = act_kind;
    op.attr.gated_mlp.clamp_limit = clamp_limit;

    return emit_op_on(b, &op, compute_dev) ? out : VELRI_TENSOR_NONE;
}

velri_tensor velri_b_gated_mlp_act(velri_builder *b, velri_tensor gate, velri_tensor up,
                                   velri_act_kind act_kind, float clamp_limit) {
    return emit_gated_mlp_act(b, gate, up, act_kind, clamp_limit, false);
}

velri_tensor velri_b_gated_mlp_act_inplace(velri_builder *b, velri_tensor gate, velri_tensor up,
                                           velri_act_kind act_kind, float clamp_limit) {
    return emit_gated_mlp_act(b, gate, up, act_kind, clamp_limit, true);
}

static velri_tensor emit_add(velri_builder *b, velri_tensor a, velri_tensor b_in, bool inplace) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!a.valid || !b_in.valid || !same_shape(&a, &b_in)) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }

    velri_device_id compute_dev = 0U;
    if (velri_b_resolve_devices(b, nullptr, &compute_dev) != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    velri_tensor out = a;
    if (!inplace || a.device_id != compute_dev) {
        out = alloc_tensor(b, VELRI_STORAGE_SCRATCH, compute_dev, a.dtype, a.rank, a.dims);
        if (!out.valid) {
            return VELRI_TENSOR_NONE;
        }
    }
    bool dyn = a.is_seq_dynamic || b_in.is_seq_dynamic;
    out.is_seq_dynamic = dyn;

    velri_op_desc op = init_op_desc(VELRI_OP_VEC_AXPBY, 2U, 1U);
    op.dynamic_seq = dyn;
    op.inputs[0] = a.view_id;
    op.inputs[1] = b_in.view_id;
    op.outputs[0] = out.view_id;
    op.attr.axpby.alpha = 1.0f;
    op.attr.axpby.beta = 1.0f;

    return emit_op_on(b, &op, compute_dev) ? out : VELRI_TENSOR_NONE;
}

velri_tensor velri_b_add(velri_builder *b, velri_tensor a, velri_tensor b_in) {
    return emit_add(b, a, b_in, false);
}

velri_tensor velri_b_add_inplace(velri_builder *b, velri_tensor a, velri_tensor b_in) {
    return emit_add(b, a, b_in, true);
}

velri_tensor velri_b_batched_linear(velri_builder *b, velri_tensor x, velri_tensor w,
                                    velri_tensor bias_or_none, uint32_t n_heads, uint32_t n_groups,
                                    uint32_t in_dim, uint32_t out_dim, bool weight_transposed) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!x.valid || !w.valid || x.rank != 2U || w.rank != 2U ||
        (bias_or_none.valid && bias_or_none.rank != 1U) || n_heads == 0U || n_groups == 0U ||
        in_dim == 0U || out_dim == 0U || (n_heads % n_groups) != 0U) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    size_t out_cols = 0U;
    if (velri_mul_overflow_size((size_t)n_heads, (size_t)out_dim, &out_cols)) {
        velri_b_set_error(b, VELRI_ERR_OVERFLOW);
        return VELRI_TENSOR_NONE;
    }
    const size_t out_dims[2] = {x.dims[0], out_cols};
    velri_tensor out = velri_b_scratch(b, x.dtype, 2U, out_dims);
    if (!out.valid) {
        return VELRI_TENSOR_NONE;
    }
    out.is_seq_dynamic = x.is_seq_dynamic;

    velri_op_desc op = init_op_desc(VELRI_OP_BATCHED_LINEAR, bias_or_none.valid ? 3U : 2U, 1U);
    op.dynamic_seq = x.is_seq_dynamic;
    op.inputs[0] = x.view_id;
    op.inputs[1] = w.view_id;
    if (bias_or_none.valid) {
        op.inputs[2] = bias_or_none.view_id;
    }
    op.outputs[0] = out.view_id;
    op.attr.batched_linear.n_heads = n_heads;
    op.attr.batched_linear.n_groups = n_groups;
    op.attr.batched_linear.in_dim = in_dim;
    op.attr.batched_linear.out_dim = out_dim;
    op.attr.batched_linear.weight_transposed = weight_transposed;

    return emit_op(b, &op) ? out : VELRI_TENSOR_NONE;
}

static velri_tensor emit_attention(velri_builder *b, velri_tensor q, velri_tensor k_in,
                                   velri_tensor v_in, velri_tensor k_cache, velri_tensor v_cache,
                                   uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim_k,
                                   uint32_t head_dim_v, float scale, uint32_t sliding_window,
                                   bool causal, bool inplace) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    uint32_t dv = head_dim_v ? head_dim_v : head_dim_k;
    if (!q.valid || !k_in.valid || !v_in.valid || !k_cache.valid || !v_cache.valid ||
        q.rank != 2U || n_heads == 0U || n_kv_heads == 0U || head_dim_k == 0U || dv == 0U) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }
    size_t attn_dim = 0U;
    if (velri_mul_overflow_size((size_t)n_heads, (size_t)dv, &attn_dim)) {
        velri_b_set_error(b, VELRI_ERR_OVERFLOW);
        return VELRI_TENSOR_NONE;
    }

    velri_device_id compute_dev = 0U;
    if (velri_b_resolve_devices(b, nullptr, &compute_dev) != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    velri_tensor out = q;
    bool can_inplace = inplace && dv == head_dim_k && q.device_id == compute_dev &&
                       q.dims[1] == attn_dim && q.strides[1] == 1U && q.strides[0] == q.dims[1];
    if (!can_inplace) {
        const size_t out_dims[2] = {q.dims[0], attn_dim};
        out = alloc_tensor(b, VELRI_STORAGE_SCRATCH, compute_dev, q.dtype, 2U, out_dims);
        if (!out.valid) {
            return VELRI_TENSOR_NONE;
        }
    }
    out.is_seq_dynamic = true;

    velri_op_desc op = init_op_desc(VELRI_OP_ATTENTION, 3U, 3U);
    op.dynamic_seq = true;
    op.inputs[0] = q.view_id;
    op.inputs[1] = k_in.view_id;
    op.inputs[2] = v_in.view_id;
    op.outputs[0] = out.view_id;
    op.outputs[1] = k_cache.view_id;
    op.outputs[2] = v_cache.view_id;
    op.attr.attention.n_heads = n_heads;
    op.attr.attention.n_kv_heads = n_kv_heads;
    op.attr.attention.head_dim = head_dim_k;
    op.attr.attention.v_head_dim = dv;
    op.attr.attention.scale = scale;
    op.attr.attention.sliding_window = sliding_window;
    op.attr.attention.causal = causal;
    op.attr.attention.allow_f16_matmul = b->allow_f16_attention_matmul;

    return emit_op_on(b, &op, compute_dev) ? out : VELRI_TENSOR_NONE;
}

velri_tensor velri_b_attention_ex(velri_builder *b, velri_tensor q, velri_tensor k_in,
                                  velri_tensor v_in, velri_tensor k_cache, velri_tensor v_cache,
                                  uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim_k,
                                  uint32_t head_dim_v, float scale, uint32_t sliding_window,
                                  bool causal) {
    return emit_attention(b, q, k_in, v_in, k_cache, v_cache, n_heads, n_kv_heads, head_dim_k,
                          head_dim_v, scale, sliding_window, causal, false);
}

velri_tensor velri_b_attention_inplace(velri_builder *b, velri_tensor q, velri_tensor k_in,
                                       velri_tensor v_in, velri_tensor k_cache,
                                       velri_tensor v_cache, uint32_t n_heads, uint32_t n_kv_heads,
                                       uint32_t head_dim, float scale, uint32_t sliding_window,
                                       bool causal) {
    return emit_attention(b, q, k_in, v_in, k_cache, v_cache, n_heads, n_kv_heads, head_dim,
                          head_dim, scale, sliding_window, causal, true);
}

velri_tensor velri_b_attention(velri_builder *b, velri_tensor q, velri_tensor k_in,
                               velri_tensor v_in, velri_tensor k_cache, velri_tensor v_cache,
                               uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
                               float scale, uint32_t sliding_window, bool causal) {
    return emit_attention(b, q, k_in, v_in, k_cache, v_cache, n_heads, n_kv_heads, head_dim,
                          head_dim, scale, sliding_window, causal, false);
}

velri_tensor velri_b_copy_last_row(velri_builder *b, velri_tensor x) {
    if (!b || b->err != VELRI_OK) {
        return VELRI_TENSOR_NONE;
    }
    if (!x.valid || x.rank != 2U) {
        velri_b_set_error(b, VELRI_ERR_INVALID);
        return VELRI_TENSOR_NONE;
    }

    /* Slice on `x.device_id` so cross-device lm-head only transfers 1 row instead of max_seq. */
    const size_t out_dims[2] = {1U, x.dims[1]};
    velri_tensor out = alloc_tensor(b, VELRI_STORAGE_SCRATCH, x.device_id, x.dtype, 2U, out_dims);
    if (!out.valid) {
        return VELRI_TENSOR_NONE;
    }
    out.is_seq_dynamic = false;

    velri_op_desc op = init_op_desc(VELRI_OP_COPY_LAST_ROW, 1U, 1U);
    op.dynamic_seq = true;
    op.inputs[0] = x.view_id;
    op.outputs[0] = out.view_id;

    return emit_op_on(b, &op, x.device_id) ? out : VELRI_TENSOR_NONE;
}
