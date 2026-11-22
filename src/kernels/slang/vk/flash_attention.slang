// Dense FlashAttention prefill. A workgroup computes several query rows of one head
// while streaming the K/V cache. Score tiles, normalization and values stay on-chip.
import attention_args;
import key_iter;
import vk_common;

groupshared float partial_high[4][4], partial_mass[4][4];
groupshared float4 partial_values[4][4][32];

void update_flash_tile<let D : uint, let Rows : uint, let Lanes : uint, let Rounds : uint, let Masked : bool>(
    AttentionArgs a, Ptr<CacheScalar, Access.Read> k, Ptr<CacheScalar, Access.Read> v, uint base, uint remaining,
    uint start[Rows], uint end[Rows], float4 q[Rows][D/(4u*Lanes)],
    inout float high[Rows], inout float mass[Rows], inout float4 values[Rows][D/(4u*Lanes)])
{
    static const uint keys_per_wave = 32u / Lanes;
    static const uint vectors = D / (4u * Lanes);
    uint lane = WaveGetLaneIndex(), wave = WaveGetWaveIndex();
    uint channel = lane % Lanes, key = lane / Lanes;
    float neg_inf = asfloat(0xff800000u);
    float scores[Rows][Rounds], maxima[Rows];
    [ForceUnroll] for (uint r = 0u; r < Rows; ++r) maxima[r] = high[r];
    [ForceUnroll] for (uint i = 0u; i < Rounds; ++i) {
        uint index = (4u * i + wave) * keys_per_wave + key;
        uint t = base + (Masked ? min(index, remaining - 1u) : index);
        Ptr<CacheVector, Access.Read> row = Ptr<CacheVector, Access.Read>(k + uint64_t(t) * a.k_row_stride);
        float dots[Rows];
        [ForceUnroll] for (uint r = 0u; r < Rows; ++r) dots[r] = 0.0f;
        [ForceUnroll] for (uint j = 0u; j < vectors; ++j) {
            float4 key_vec = load_cache4(row + channel + j * Lanes);
            [ForceUnroll] for (uint r = 0u; r < Rows; ++r) dots[r] += dot(q[r][j], key_vec);
        }
        [ForceUnroll] for (uint r = 0u; r < Rows; ++r) {
            [ForceUnroll] for (uint mask = Lanes / 2u; mask > 0u; mask >>= 1u) {
                dots[r] += WaveReadLaneAt(dots[r], lane ^ mask);
            }
            bool valid = !Masked || (index < remaining && t >= start[r] && t < end[r]);
            scores[r][i] = valid ? dots[r] * a.scale : neg_inf;
            maxima[r] = max(maxima[r], scores[r][i]);
        }
    }
    [ForceUnroll] for (uint r = 0u; r < Rows; ++r) {
        float next_high = WaveActiveMax(maxima[r]);
        float factor = high[r] > neg_inf ? exp(high[r] - next_high) : 0.0f;
        mass[r] *= factor;
        [ForceUnroll] for (uint j = 0u; j < vectors; ++j) values[r][j] *= factor;
        high[r] = next_high;
    }
    [ForceUnroll] for (uint i = 0u; i < Rounds; ++i) {
        uint index = (4u * i + wave) * keys_per_wave + key;
        uint t = base + (Masked ? min(index, remaining - 1u) : index);
        Ptr<CacheVector, Access.Read> row = Ptr<CacheVector, Access.Read>(v + uint64_t(t) * a.v_row_stride);
        float weights[Rows];
        [ForceUnroll] for (uint r = 0u; r < Rows; ++r) {
            bool valid = !Masked || (index < remaining && t >= start[r] && t < end[r]);
            weights[r] = valid ? exp(scores[r][i] - high[r]) : 0.0f;
            mass[r] += weights[r];
        }
        [ForceUnroll] for (uint j = 0u; j < vectors; ++j) {
            float4 value = load_cache4(row + channel + j * Lanes);
            [ForceUnroll] for (uint r = 0u; r < Rows; ++r) {
                // A masked value can contain a NaN. Only the admissible keys affect this row.
                if (!Masked || (index < remaining && t >= start[r] && t < end[r])) {
                    values[r][j] = fma(float4(weights[r]), value, values[r][j]);
                }
            }
        }
    }
}

void flash_attention<let D : uint, let Rows : uint, let Lanes : uint, let Rounds : uint>(
    AttentionArgs a, uint first_row, uint head)
{
    static const uint keys_per_wave = 32u / Lanes;
    static const uint vectors = D / (4u * Lanes);
    static const uint tile_keys = 4u * keys_per_wave * Rounds;
    uint lane = WaveGetLaneIndex(), wave = WaveGetWaveIndex();
    uint channel = lane % Lanes, key = lane / Lanes;
    uint kv_head = head / (a.n_heads / a.n_kv_heads);
    Ptr<CacheScalar, Access.Read> k = Ptr<CacheScalar, Access.Read>(a.k_cache + uint64_t(kv_head) * a.k_head_stride);
    Ptr<CacheScalar, Access.Read> v = Ptr<CacheScalar, Access.Read>(a.v_cache + uint64_t(kv_head) * a.v_head_stride);
    float4 q[Rows][vectors], values[Rows][vectors];
    float high[Rows], mass[Rows];
    uint start[Rows], end[Rows];
    uint pos = attention_kv_pos(a);
    float neg_inf = asfloat(0xff800000u);
    [ForceUnroll] for (uint r = 0u; r < Rows; ++r) {
        uint row = first_row + min(r, a.rows - 1u - first_row);
        DenseRangeIter range = make_dense_range_iter(pos, row, a.rows, a.max_ctx, a.sliding_window, a.causal);
        start[r] = range.start;
        end[r] = range.start + range.len;
        Ptr<float4, Access.Read> query = Ptr<float4, Access.Read>(
            a.q + uint64_t(row) * a.q_stride + uint64_t(head) * D);
        [ForceUnroll] for (uint j = 0u; j < vectors; ++j) {
            q[r][j] = load_a16(query + channel + j * Lanes);
            values[r][j] = 0.0f;
        }
        high[r] = neg_inf;
        mass[r] = 0.0f;
    }

    uint base = start[0];
    uint remaining = end[Rows - 1u] - base;
    while (remaining != 0u) {
        // Every query can consume all keys in an interior tile.
        bool interior = base >= start[Rows - 1u] && remaining >= tile_keys && end[0] - base >= tile_keys && base < end[0];
        if (interior) {
            update_flash_tile<D, Rows, Lanes, Rounds, false>(a, k, v, base, remaining, start, end, q, high, mass, values);
        } else {
            update_flash_tile<D, Rows, Lanes, Rounds, true>(a, k, v, base, remaining, start, end, q, high, mass, values);
        }
        if (remaining <= tile_keys) break;
        remaining -= tile_keys;
        base += tile_keys;
    }

    // Each subgroup has an independently normalized subset of keys. Merge the four
    // partials once, using their running maxima to restore a common softmax scale.
    [ForceUnroll] for (uint r = 0u; r < Rows; ++r) {
        [ForceUnroll] for (uint mask = Lanes; mask < 32u; mask <<= 1u) {
            mass[r] += WaveReadLaneAt(mass[r], lane ^ mask);
            [ForceUnroll] for (uint j = 0u; j < vectors; ++j) {
                values[r][j] += WaveReadLaneAt(values[r][j], lane ^ mask);
            }
        }
        if (lane == 0u) {
            partial_high[r][wave] = high[r];
            partial_mass[r][wave] = mass[r];
        }
        if (key == 0u) {
            [ForceUnroll] for (uint j = 0u; j < vectors; ++j) {
                partial_values[r][wave][channel + j * Lanes] = values[r][j];
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    uint tid = wave * 32u + lane;
    uint r = tid / (D / 4u), vector = tid % (D / 4u);
    if (r < Rows && r < a.rows - first_row) {
        float largest = max(max(partial_high[r][0], partial_high[r][1]),
                            max(partial_high[r][2], partial_high[r][3]));
        float total = 0.0f;
        float4 value = 0.0f;
        [ForceUnroll] for (uint w = 0u; w < 4u; ++w) {
            float factor = partial_mass[r][w] == 0.0f ? 0.0f : exp(partial_high[r][w] - largest);
            total += partial_mass[r][w] * factor;
            value += partial_values[r][w][vector] * factor;
        }
        Ptr<float> out_row = a.y + uint64_t(first_row + r) * a.o_stride + uint64_t(head) * D;
        store_f32x4_maybe_a16(out_row + vector * 4u, start[r] < end[r] ? value / total : float4(0.0f));
    }
}

[shader("compute")] [numthreads(128, 1, 1)]
void vk_flash_attention_f32(uniform AttentionArgs a, uint3 gid : SV_GroupID) {
    if (a.d_k == 32u) flash_attention<32u, 4u, 4u, 8u>(a, gid.x * 4u, gid.y);
    else if (a.d_k == 64u) flash_attention<64u, 4u, 4u, 8u>(a, gid.x * 4u, gid.y);
    else flash_attention<128u, 2u, 8u, 4u>(a, gid.x * 2u, gid.y);
}
