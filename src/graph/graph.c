#include "core_internal.h"
#include <string.h>

velri_status velri_graph_create(velri_graph **out_graph) {
    if (!out_graph) {
        return VELRI_ERR_INVALID;
    }
    *out_graph = nullptr;
    velri_graph *g = (velri_graph *)calloc(1U, sizeof(*g));
    if (!g) {
        return VELRI_ERR_NOMEM;
    }
    *out_graph = g;
    return VELRI_OK;
}

void velri_graph_destroy(velri_graph *graph) {
    if (!graph) {
        return;
    }
    free(graph->storages);
    free(graph->views);
    free(graph->nodes);
    free(graph);
}

static bool valid_storage_kind(velri_storage_kind kind) {
    return kind == VELRI_STORAGE_WEIGHT || kind == VELRI_STORAGE_INPUT ||
           kind == VELRI_STORAGE_OUTPUT || kind == VELRI_STORAGE_STATE ||
           kind == VELRI_STORAGE_SCRATCH;
}

velri_status velri_graph_add_storage(velri_graph *graph, const velri_storage_desc *desc,
                                     velri_storage_id *out_id) {
    if (!graph || !desc || !out_id || desc->byte_size == 0U || !valid_storage_kind(desc->kind)) {
        return VELRI_ERR_INVALID;
    }
    if (velri_dtype_bytes(desc->dtype) == 0U) {
        return VELRI_ERR_INVALID;
    }
    if (desc->align != 0U && !velri_is_pow2(desc->align)) {
        return VELRI_ERR_INVALID;
    }

    void *buf = nullptr;
    if (!velri_vec_ensure(graph->storages, &buf, &graph->cap_storages, graph->num_storages, 1U,
                          sizeof(velri_storage_desc))) {
        return VELRI_ERR_NOMEM;
    }
    graph->storages = (velri_storage_desc *)buf;

    uint32_t id = graph->num_storages++;
    graph->storages[id] = *desc;
    if (graph->storages[id].align < 64U) {
        graph->storages[id].align = 64U;
    }
    *out_id = id;
    return VELRI_OK;
}

velri_status velri_graph_add_view(velri_graph *graph, const velri_tensor_view *view,
                                  velri_view_id *out_id) {
    if (!graph || !view || !out_id || view->storage_id >= graph->num_storages) {
        return VELRI_ERR_INVALID;
    }
    velri_status st = velri_tensor_view_validate(view, &graph->storages[view->storage_id]);
    if (st != VELRI_OK) {
        return st;
    }

    void *buf = nullptr;
    if (!velri_vec_ensure(graph->views, &buf, &graph->cap_views, graph->num_views, 1U,
                          sizeof(velri_tensor_view))) {
        return VELRI_ERR_NOMEM;
    }
    graph->views = (velri_tensor_view *)buf;

    uint32_t id = graph->num_views++;
    graph->views[id] = *view;
    *out_id = id;
    return VELRI_OK;
}

void velri_graph_resolve_node(const velri_graph *graph, const velri_op_desc *op,
                              velri_resolved_op *out) {
    memset(out, 0, sizeof(*out));
    out->kind = op->kind;
    out->num_inputs = op->num_inputs;
    out->num_outputs = op->num_outputs;
    out->raw = *op;
    for (uint32_t i = 0U; i < op->num_inputs; ++i) {
        velri_view_id vid = op->inputs[i];
        if (vid != VELRI_INVALID_ID) {
            out->input_present[i] = true;
            out->inputs[i] = graph->views[vid];
        }
    }
    for (uint32_t i = 0U; i < op->num_outputs; ++i) {
        out->outputs[i] = graph->views[op->outputs[i]];
    }
}

velri_status velri_graph_add_op(velri_graph *graph, const velri_op_desc *op,
                                velri_device_id device_id, velri_node_id *out_node_id) {
    if (!graph || !op || op->num_inputs > VELRI_OP_MAX_INPUTS || op->num_outputs == 0U ||
        op->num_outputs > VELRI_OP_MAX_OUTPUTS) {
        return VELRI_ERR_INVALID;
    }

    for (uint32_t i = 0U; i < op->num_inputs; ++i) {
        velri_view_id vid = op->inputs[i];
        if (vid != VELRI_INVALID_ID && vid >= graph->num_views) {
            return VELRI_ERR_INVALID;
        }
    }
    for (uint32_t i = 0U; i < op->num_outputs; ++i) {
        velri_view_id vid = op->outputs[i];
        if (vid >= graph->num_views) {
            return VELRI_ERR_INVALID;
        }
        velri_storage_kind sk = graph->storages[graph->views[vid].storage_id].kind;
        if (sk == VELRI_STORAGE_WEIGHT || sk == VELRI_STORAGE_INPUT) {
            return VELRI_ERR_INVALID;
        }
    }

    velri_resolved_op resolved;
    velri_graph_resolve_node(graph, op, &resolved);
    velri_status st = velri_op_validate(&resolved);
    if (st != VELRI_OK) {
        return st;
    }

    void *buf = nullptr;
    if (!velri_vec_ensure(graph->nodes, &buf, &graph->cap_nodes, graph->num_nodes, 1U,
                          sizeof(velri_graph_node))) {
        return VELRI_ERR_NOMEM;
    }
    graph->nodes = (velri_graph_node *)buf;

    uint32_t nid = graph->num_nodes++;
    graph->nodes[nid].desc = *op;
    graph->nodes[nid].device_id = device_id;
    if (out_node_id) {
        *out_node_id = nid;
    }
    return VELRI_OK;
}

uint32_t velri_graph_storage_count(const velri_graph *graph) {
    return graph ? graph->num_storages : 0U;
}

uint32_t velri_graph_view_count(const velri_graph *graph) {
    return graph ? graph->num_views : 0U;
}

uint32_t velri_graph_node_count(const velri_graph *graph) {
    return graph ? graph->num_nodes : 0U;
}
