// Vulkan Embedding lookup and CopyLastRow kernels (zero groupshared memory).
import vk_common;

// ============================================================================
// 1. Embedding Lookup (Token ID table lookup or positional offset lookup)
// ============================================================================
struct EmbeddingArgs {
    Ptr<float, Access.Read> table;
    Ptr<int32_t, Access.Read> ids;
    Ptr<float> y;
    Ptr<uint, Access.Read> step_params; // optional [seq_len, pos_offset, kv_pos] for captured graphs
    uint rows;
    uint dim;
    uint vocab;
    uint table_stride;
    uint out_stride;
    uint pos_offset;
    uint has_ids;
    uint pad;
};

[shader("compute")] [numthreads(256, 1, 1)]
void vk_embedding_f32(uniform EmbeddingArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint t = gid.x * kRowWgSize + gtid.x;
    uint row = gid.y;
    if (row >= a.rows || (t << 2u) >= a.dim) return;

    bool valid = false;
    uint idx = 0u;
    if (a.has_ids != 0u) {
        int32_t raw_id = a.ids[row];
        valid = (raw_id >= 0) && (uint(raw_id) < a.vocab);
        idx = uint(raw_id);
    } else {
        uint eff_pos = step_word_or(a.step_params, 1u, a.pos_offset, a.table);
        uint64_t pos64 = uint64_t(eff_pos) + uint64_t(row);
        valid = pos64 < uint64_t(a.vocab);
        idx = uint(pos64);
    }

    Ptr<float> dst = a.y + uint64_t(row) * uint64_t(a.out_stride);
    if (!valid) {
        zero_f32_chunk4(dst, t, a.dim);
        return;
    }

    Ptr<float, Access.Read> src = a.table + uint64_t(idx) * uint64_t(a.table_stride);
    copy_f32_chunk4(src, dst, t, a.dim);
}

// ============================================================================
// 2. Copy Last Active Row ([rows, dim] -> [1, dim])
// ============================================================================
struct CopyLastRowArgs {
    Ptr<float, Access.Read> x;
    Ptr<float> y;
    uint src_row;
    uint dim;
    uint x_stride;
    uint pad;
};

[shader("compute")] [numthreads(256, 1, 1)]
void vk_copy_last_row_f32(uniform CopyLastRowArgs a, uint3 tid : SV_DispatchThreadID) {
    uint t = tid.x;
    if ((t << 2u) >= a.dim) return;

    Ptr<float, Access.Read> src = a.x + uint64_t(a.src_row) * uint64_t(a.x_stride);
    copy_f32_chunk4(src, a.y, t, a.dim);
}
