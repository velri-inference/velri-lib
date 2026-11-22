import attention_args;
import key_iter;
import vk_common;

// Each subgroup holds four queries in registers. The workgroup shares one 32-key tile
// across neighboring queries. Wide grids write the normalized output directly.
// Short grids distribute the KV history across additional workgroups and merge their partials.
groupshared float4 prefill_keys[32][17];
groupshared float4 prefill_values[32][16];

struct PrefillTileLoad<let Loads : uint> {
    float4 keys[Loads];
    float4 values[Loads];
};

PrefillTileLoad<16u / Waves> load_prefill_tile<let Waves : uint>(AttentionArgs a, Ptr<CacheScalar> k, Ptr<CacheScalar> v,
                                  uint base, uint limit, uint tid) {
    PrefillTileLoad<16u / Waves> tile;
    [ForceUnroll] for (uint j = 0u; j < 16u / Waves; ++j) {
        uint i = tid + j * (Waves * 32u), token = i / 16u, channel = i % 16u;
        tile.keys[j] = 0.0f;
        tile.values[j] = 0.0f;
        if (token < limit - base) {
            tile.keys[j] = load_cache4(Ptr<CacheVector, Access.Read>(k + uint64_t(base + token) * a.k_row_stride) + channel);
            tile.values[j] = load_cache4(Ptr<CacheVector, Access.Read>(v + uint64_t(base + token) * a.v_row_stride) + channel);
        }
    }
    return tile;
}

void update_prefill_split_tile<let Masked : bool, let DotLanes : uint>(float scale, uint base, uint count, uint4 low, uint4 end,
    float4 query[4][16u / DotLanes], inout float4 high, inout float4 mass, inout float4 output[4]) {
    uint lane = WaveGetLaneIndex();
    // Each score contains four queries. Neighboring lanes cooperate on their 64 channels.
    static const uint keys_per_round = 32u / DotLanes;
    float4 scores[DotLanes], next = high, local_mass = 0.0f;
    [ForceUnroll] for (uint round = 0u; round < DotLanes; ++round) {
        uint token = keys_per_round * round + lane / DotLanes;
        float4 score = 0.0f;
        [ForceUnroll] for (uint j = 0u; j < 16u / DotLanes; ++j) {
            float4 key = prefill_keys[token][j * DotLanes + lane % DotLanes];
            [ForceUnroll] for (uint r = 0u; r < 4u; ++r) score[r] += dot(key, query[r][j]);
        }
        [ForceUnroll] for (uint mask = 1u; mask < DotLanes; mask <<= 1u)
            score += WaveReadLaneAt(score, lane ^ mask);
        scores[round] = score * scale;
        if (Masked) {
            bool4 visible = (token < count) & (uint4(base + token) >= low) & (uint4(base + token) < end);
            scores[round] = select(visible, scores[round], float4(asfloat(0xff800000u)));
        }
        next = max(next, scores[round]);
    }
    next = WaveActiveMax(next);
    bool4 seen = DotLanes == 2u ? mass != 0.0f : high != float4(asfloat(0xff800000u));
    float4 carry = select(seen, exp(high - next), float4(0.0f));
    [ForceUnroll] for (uint round = 0u; round < DotLanes; ++round) {
        scores[round] = exp(scores[round] - next);
        if (Masked) {
            uint token = keys_per_round * round + lane / DotLanes;
            bool4 visible = (token < count) & (uint4(base + token) >= low) & (uint4(base + token) < end);
            scores[round] = select(visible, scores[round], float4(0.0f));
        }
        local_mass += scores[round];
    }
    if (DotLanes == 2u) {
        [ForceUnroll] for (uint mask = DotLanes; mask < 32u; mask <<= 1u)
            local_mass += WaveReadLaneAt(local_mass, lane ^ mask);
    }
    mass = mass * carry + local_mass;
    high = next;
    [ForceUnroll] for (uint r = 0u; r < 4u; ++r) output[r] *= carry[r];
    [ForceUnroll] for (uint t = 0u; t < 32u; t += 2u) {
        uint token = t + lane / 16u;
        float4 w = WaveReadLaneAt(scores[t / keys_per_round], (token % keys_per_round) * DotLanes);
        float4 value = prefill_values[token][lane % 16u];
        [ForceUnroll] for (uint r = 0u; r < 4u; ++r) {
            bool visible = !Masked || (token < count && base + token >= low[r] && base + token < end[r]);
            output[r] = fma(float4(w[r]), visible ? value : float4(0.0f), output[r]);
        }
    }
}

void prefill_tiled<let Waves : uint, let Direct : bool, let DotLanes : uint>(AttentionArgs a, uint3 group) {
    uint lane = WaveGetLaneIndex(), wave = WaveGetWaveIndex();
    uint tid = wave * 32u + lane;
    uint first = group.x * (Waves * 4u);
    uint pos = Direct ? attention_kv_pos(a) : a.kv_pos;
    uint row0 = first + wave * 4u;
    uint head = group.y, split = group.z;
    uint kv_head = head / (a.n_heads / a.n_kv_heads);
    Ptr<CacheScalar> k = a.k_cache + uint64_t(kv_head) * a.k_head_stride;
    Ptr<CacheScalar> v = a.v_cache + uint64_t(kv_head) * a.v_head_stride;

    float4 query[4][16u / DotLanes];
    uint4 low, end;
    [ForceUnroll] for (uint r = 0u; r < 4u; ++r) {
        uint row = min(row0 + r, a.rows - 1u);
        DenseRangeIter keys = make_dense_range_iter(pos, row, a.rows, a.max_ctx,
                                                    a.sliding_window, a.causal);
        low[r] = keys.start;
        end[r] = keys.start + keys.len;
        Ptr<float4, Access.Read> q = Ptr<float4, Access.Read>(
            a.q + uint64_t(row) * a.q_stride + uint64_t(head) * 64u);
        [ForceUnroll] for (uint j = 0u; j < 16u / DotLanes; ++j) query[r][j] = load_a16(q + j * DotLanes + lane % DotLanes);
    }
    DenseRangeIter first_keys = make_dense_range_iter(pos, first, a.rows, a.max_ctx,
                                                      a.sliding_window, a.causal);
    DenseRangeIter last_keys = make_dense_range_iter(pos, min(first + (Waves * 4u - 1u), a.rows - 1u),
                                                     a.rows, a.max_ctx, a.sliding_window, a.causal);
    uint span = last_keys.start + last_keys.len - first_keys.start;
    uint partitions = Direct ? 1u : a.value_slices;
    uint per_split = span / partitions + uint(span % partitions != 0u);
    uint begin = first_keys.start + uint(min(uint64_t(split) * per_split, uint64_t(span)));
    uint limit = first_keys.start + uint(min(uint64_t(split + 1u) * per_split, uint64_t(span)));
    // A leading tile of negative infinities contributes zero before the first finite score.
    float4 high = -3.402823466e38f, mass = 0.0f, output[4];
    [ForceUnroll] for (uint r = 0u; r < 4u; ++r) output[r] = 0.0f;

    PrefillTileLoad<16u / Waves> tile = load_prefill_tile<Waves>(a, k, v, begin, limit, tid);
    for (uint base = begin; base < limit;) {
        uint count = min(32u, limit - base);
        [ForceUnroll] for (uint j = 0u; j < 16u / Waves; ++j) {
            uint i = tid + j * (Waves * 32u);
            prefill_keys[i / 16u][i % 16u] = tile.keys[j];
            prefill_values[i / 16u][i % 16u] = tile.values[j];
        }
        GroupMemoryBarrierWithGroupSync();
        // Fetch the next KV tile while the subgroups compute scores and values for this one.
        tile = load_prefill_tile<Waves>(a, k, v, base + count, limit, tid);

        if (count == 32u && base >= low[3] && base + count <= end[0])
            update_prefill_split_tile<false, DotLanes>(a.scale, base, count, low, end, query, high, mass, output);
        else
            update_prefill_split_tile<true, DotLanes>(a.scale, base, count, low, end, query, high, mass, output);
        if (limit - base == count) break;
        GroupMemoryBarrierWithGroupSync();
        base += count;
    }
    if (DotLanes == 4u) {
        // The rescale is identical on all lanes. Sum their contributions once after the KV loop.
        [ForceUnroll] for (uint mask = DotLanes; mask < 32u; mask <<= 1u)
            mass += WaveReadLaneAt(mass, lane ^ mask);
    }
    [ForceUnroll] for (uint r = 0u; r < 4u; ++r) {
        float4 sum = output[r] + WaveReadLaneAt(output[r], lane ^ 16u);
        if (lane < 16u && row0 + r < a.rows) {
            if (Direct) {
                Ptr<float> y = a.y + uint64_t(row0 + r) * a.o_stride + uint64_t(head) * 64u;
                store_f32x4_maybe_a16(y + lane * 4u, mass[r] != 0.0f ? sum / mass[r] : float4(0.0f));
                continue;
            }
            uint row_head = (row0 + r) * a.n_heads + head;
            Ptr<float> partial = a.y + (uint64_t(row_head) * a.value_slices + split) * 68u;
            if (lane == 0u) { partial[0] = high[r]; partial[1] = mass[r]; }
            store_a16(Ptr<float4>(partial + 4u) + lane, sum);
        }
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_prefill_split_f32(uniform AttentionArgs a, uint3 group : SV_GroupID) {
    prefill_tiled<8u, false, 2u>(a, group);
}

[shader("compute")] [numthreads(128, 1, 1)]
void vk_prefill_tile_f32(uniform AttentionArgs a, uint3 group : SV_GroupID) {
    prefill_tiled<4u, true, 4u>(a, group);
}

struct PrefillMergeArgs {
    Ptr<float, Access.Read> partials;
    Ptr<float> output;
    uint row_heads;
    uint heads;
    uint output_stride;
    uint splits;
};

[shader("compute")] [numthreads(128, 1, 1)]
void vk_prefill_merge_f32(uniform PrefillMergeArgs a, uint3 group : SV_GroupID) {
    uint row_head = group.x * 8u + WaveGetWaveIndex() * 2u + WaveGetLaneIndex() / 16u;
    if (row_head >= a.row_heads) return;
    uint lane = WaveGetLaneIndex(), col = lane % 16u;
    Ptr<float, Access.Read> p = a.partials + uint64_t(row_head) * a.splits * 68u;
    float high = asfloat(0xff800000u), mass = 0.0f;
    if (col < a.splits) {
        Ptr<float, Access.Read> part = p + col * 68u;
        mass = part[1];
        if (mass != 0.0f) high = part[0];
    }
    float largest = high;
    [ForceUnroll] for (uint mask = 8u; mask > 0u; mask >>= 1u)
        largest = max(largest, WaveReadLaneAt(largest, lane ^ mask));
    float factor = mass != 0.0f ? exp(high - largest) : 0.0f;
    mass *= factor;
    [ForceUnroll] for (uint mask = 8u; mask > 0u; mask >>= 1u)
        mass += WaveReadLaneAt(mass, lane ^ mask);
    float4 sum = 0.0f;
    for (uint split = 0u; split < a.splits; ++split) {
        float weight = WaveReadLaneAt(factor, (lane & 16u) + split);
        sum = fma(float4(weight), load_a16(Ptr<float4, Access.Read>(p + 68u * split + 4u) + col), sum);
    }
    uint row = row_head / a.heads, head = row_head % a.heads;
    store_f32x4_maybe_a16(a.output + uint64_t(row) * a.output_stride + uint64_t(head) * 64u + col * 4u,
                          mass != 0.0f ? sum / mass : float4(0.0f));
}
