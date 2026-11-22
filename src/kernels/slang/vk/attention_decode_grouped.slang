import attention_args;
import key_iter;
import vk_common;

groupshared float group_high[4][4], group_mass[4][4];
groupshared float4 group_value[4][4][16];

// Each eight-lane team reads one complete F16 KV row. The same packed loads
// feed every Q head in the group. Each lane owns eight channels of each output.
void update_group_tile<let Heads : uint, let Full : bool>(
    SplitKvArgs a, Ptr<CacheScalar, Access.Read> k, Ptr<CacheScalar, Access.Read> v,
    uint first, uint count, float4 query[Heads][2], inout vector<float, Heads> high,
    inout vector<float, Heads> mass, inout float4 acc[Heads][2]) {
    uint lane = WaveGetLaneIndex(), wave = WaveGetWaveIndex();
    uint channel = lane % 8u, key = wave * 4u + lane / 8u;
    vector<float, Heads> scores[2], next = high;
    [ForceUnroll] for (uint i = 0u; i < 2u; ++i) {
        uint row = i * 16u + key;
        vector<float, Heads> score = 0.0f;
        if (Full || row < count) {
            Ptr<uint4, Access.Read> src = Ptr<uint4, Access.Read>(k + uint64_t(first + row) * a.k_row_stride);
            uint4 packed = load_a16(src + channel);
            float4 k0 = float4(bit_cast<half4>(packed.xy)), k1 = float4(bit_cast<half4>(packed.zw));
            [ForceUnroll] for (uint g = 0u; g < Heads; ++g)
                score[g] = dot(query[g][0], k0) + dot(query[g][1], k1);
        }
        [ForceUnroll] for (uint mask = 4u; mask > 0u; mask >>= 1u)
            score += WaveReadLaneAt(score, lane ^ mask);
        scores[i] = (Full || row < count) ? score * a.scale : vector<float, Heads>(asfloat(0xff800000u));
        next = max(next, scores[i]);
    }
    next = WaveActiveMax(next);
    vector<float, Heads> carry = select(mass > 0.0f, exp(high - next), vector<float, Heads>(0.0f));
    high = next;
    mass *= carry;
    [ForceUnroll] for (uint g = 0u; g < Heads; ++g)
        [ForceUnroll] for (uint j = 0u; j < 2u; ++j) acc[g][j] *= carry[g];
    [ForceUnroll] for (uint i = 0u; i < 2u; ++i) {
        uint row = i * 16u + key;
        if (Full || row < count) {
            vector<float, Heads> weights = exp(scores[i] - high);
            mass += weights;
            Ptr<uint4, Access.Read> src = Ptr<uint4, Access.Read>(v + uint64_t(first + row) * a.v_row_stride);
            uint4 packed = load_a16(src + channel);
            float4 v0 = float4(bit_cast<half4>(packed.xy)), v1 = float4(bit_cast<half4>(packed.zw));
            [ForceUnroll] for (uint g = 0u; g < Heads; ++g) {
                acc[g][0] = fma(float4(weights[g]), v0, acc[g][0]);
                acc[g][1] = fma(float4(weights[g]), v1, acc[g][1]);
            }
        }
    }
}

void grouped_decode_partition<let Heads : uint>(SplitKvArgs a, uint group, uint split, DenseRangeIter keys) {
    uint lane = WaveGetLaneIndex(), wave = WaveGetWaveIndex();
    uint first_head = group * Heads, kv_head = first_head / (a.n_heads / a.n_kv_heads);
    Ptr<CacheScalar, Access.Read> k = a.k_cache + uint64_t(kv_head) * a.k_head_stride;
    Ptr<CacheScalar, Access.Read> v = a.v_cache + uint64_t(kv_head) * a.v_head_stride;

    // Assign whole 16-key strips to sections. Only the final section receives
    // the suffix, so growing from 8192 to 8193 keys doesn't add a tail to every group.
    uint strips = keys.len / 16u, extra = strips % a.n_splits, per_split = strips / a.n_splits;
    uint begin = (per_split * split + min(split, extra)) * 16u;
    uint count = split + 1u == a.n_splits ? keys.len - begin : (per_split + uint(split < extra)) * 16u;
    uint first = keys.start + begin;
    float4 query[Heads][2], acc[Heads][2];
    [ForceUnroll] for (uint g = 0u; g < Heads; ++g) {
        Ptr<float4, Access.Read> q = Ptr<float4, Access.Read>(a.q + uint64_t(first_head + g) * 64u);
        [ForceUnroll] for (uint j = 0u; j < 2u; ++j) {
            query[g][j] = load_a16(q + (lane % 8u) * 2u + j);
            acc[g][j] = 0.0f;
        }
    }
    vector<float, Heads> high = asfloat(0xff800000u), mass = 0.0f;
    while (count >= 32u) {
        update_group_tile<Heads, true>(a, k, v, first, 32u, query, high, mass, acc);
        first += 32u; count -= 32u;
    }
    if (count) update_group_tile<Heads, false>(a, k, v, first, count, query, high, mass, acc);

    [ForceUnroll] for (uint mask = 8u; mask < 32u; mask *= 2u) {
        mass += WaveReadLaneAt(mass, lane ^ mask);
        [ForceUnroll] for (uint g = 0u; g < Heads; ++g)
            [ForceUnroll] for (uint j = 0u; j < 2u; ++j)
                acc[g][j] += WaveReadLaneAt(acc[g][j], lane ^ mask);
    }
    [ForceUnroll] for (uint g = 0u; g < Heads; ++g) {
        if (lane == 0u) { group_high[g][wave] = high[g]; group_mass[g][wave] = mass[g]; }
        if (lane < 8u) {
            group_value[g][wave][2u * lane] = acc[g][0];
            group_value[g][wave][2u * lane + 1u] = acc[g][1];
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (wave < Heads && lane < 16u) {
        float largest = asfloat(0xff800000u), total = 0.0f;
        float4 value = 0.0f;
        [ForceUnroll] for (uint w = 0u; w < 4u; ++w) largest = max(largest, group_high[wave][w]);
        [ForceUnroll] for (uint w = 0u; w < 4u; ++w) {
            float factor = group_mass[wave][w] > 0.0f ? exp(group_high[wave][w] - largest) : 0.0f;
            total += group_mass[wave][w] * factor;
            value += group_value[wave][w][lane] * factor;
        }
        Ptr<float> dst = a.partials + (uint64_t(first_head + wave) * a.n_splits + split) * a.part_stride;
        if (lane == 0u) { dst[0] = largest; dst[1] = total; }
        store_f32x4_maybe_a16(dst + 2u + lane * 4u, value);
    }
}

[shader("compute")] [numthreads(128, 1, 1)]
void vk_attention_grouped_f32(uniform SplitKvArgs a, uint3 gid : SV_GroupID) {
    uint pos = cache_step_pos(a.step_params, a.kv_pos, a.k_cache);
    DenseRangeIter keys = make_dense_range_iter(pos, 0u, 1u, a.max_ctx, a.sliding_window, a.causal);
    if (a.group_heads == 3u) grouped_decode_partition<3u>(a, gid.x, gid.y, keys);
    else grouped_decode_partition<4u>(a, gid.x, gid.y, keys);
}
