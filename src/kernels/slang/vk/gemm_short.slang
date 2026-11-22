import vk_common;
import epilogue;

// A 256-thread group computes a 64x64 tile. Each thread loads four contiguous
// inner-dimension elements from each input and accumulates a 4x4 output tile.
groupshared float4 tile_x[2][16][16];
groupshared float4 tile_w[2][16][16];

void stage_inputs(uint stage, uint thread, float4 x, float4 w) {
    uint row = thread / 4u;
    uint k = (thread % 4u) * 4u;
    // XOR the four K-vector positions into the row index to spread simultaneous
    // transposed stores over shared-memory banks without padding the 16 KiB tile.
    uint slot = (row / 4u) ^ (k / 4u);
    [ForceUnroll] for (uint i = 0u; i < 4u; ++i) {
        tile_x[stage][k + i][slot][row % 4u] = x[i];
        tile_w[stage][k + i][slot][row % 4u] = w[i];
    }
}

void accumulate_tile(uint stage, uint row, uint col, inout float4 sums[4]) {
    [ForceUnroll] for (uint k = 0u; k < 16u; ++k) {
        float4 x = tile_x[stage][k][row ^ (k / 4u)];
        float4 w = tile_w[stage][k][col ^ (k / 4u)];
        [ForceUnroll] for (uint i = 0u; i < 4u; ++i) sums[i] = fma(float4(x[i]), w, sums[i]);
    }
}

void multiply_short<let Edges : bool>(LinearArgs<float, float, float> a, uint3 group, uint thread) {
    uint row0 = group.y * 64u, col0 = group.x * 64u;
    uint row = thread / 16u, col = thread % 16u;
    uint load_row = thread / 4u, load_k = thread % 4u;
    bool read_x = !Edges || row0 + load_row < a.rows;
    bool read_w = !Edges || col0 + load_row < a.m;
    // Form an in-bounds pointer even for inactive edge loads.
    Ptr<float4, Access.Read> x = Ptr<float4, Access.Read>(
        a.x + uint64_t(read_x ? row0 + load_row : 0u) * a.x_stride);
    Ptr<float4, Access.Read> w = Ptr<float4, Access.Read>(
        a.w + uint64_t(read_w ? col0 + load_row : 0u) * a.w_stride);

    uint tiles = a.k / 16u;
    uint per_split = (tiles + a.split_k - 1u) / a.split_k;
    uint first = min(tiles, group.z * per_split);
    uint end = min(tiles, (group.z + 1u) * per_split);
    float4 sums[4];
    [ForceUnroll] for (uint i = 0u; i < 4u; ++i) sums[i] = float4(0.0f);
    if (first < end) {
        uint offset = first * 4u + load_k;
        stage_inputs(0u, thread, read_x ? load_a16(x + offset) : float4(0.0f),
                     read_w ? load_a16(w + offset) : float4(0.0f));
        GroupMemoryBarrierWithGroupSync();
        uint stage = 0u;
        for (uint tile = first + 1u; tile < end; ++tile) {
            offset += 4u;
            float4 next_x = read_x ? load_a16(x + offset) : float4(0.0f);
            float4 next_w = read_w ? load_a16(w + offset) : float4(0.0f);
            accumulate_tile(stage, row, col, sums);
            stage ^= 1u;
            stage_inputs(stage, thread, next_x, next_w);
            GroupMemoryBarrierWithGroupSync();
        }
        accumulate_tile(stage, row, col, sums);
    }

    Ptr<float> dst = a.y + uint64_t(group.z) * a.rows * a.y_stride;
    uint column = col0 + col * 4u;
    if (Edges && column >= a.m) return;
    EpilogueParams ep = make_epilogue_params(a.bias, a.x_scale, a.w_scale,
                                             a.split_k > 1u ? 0u : a.flags);
    [ForceUnroll] for (uint i = 0u; i < 4u; ++i) {
        uint r = row0 + row * 4u + i;
        if (!Edges || r < a.rows) {
            Ptr<float> yr = dst + uint64_t(r) * a.y_stride;
            store_epilogue_f32x4<EpBias>(yr, r, column, a.m, is_ptr_aligned16(yr), sums[i], ep);
        }
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_gemm_short_f32(uniform LinearArgs<float, float, float> a, uint3 group : SV_GroupID,
                       uint3 thread : SV_GroupThreadID) {
    if (a.rows - group.y * 64u >= 64u && a.m - group.x * 64u >= 64u) {
        multiply_short<false>(a, group, thread.x);
    } else {
        multiply_short<true>(a, group, thread.x);
    }
}
