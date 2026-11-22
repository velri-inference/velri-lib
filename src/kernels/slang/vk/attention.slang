// Vulkan KV-Cache Append and Unified Online-Softmax Attention (F32, d_k/d_v up to 2048,
// compile-time IKeyIter generic). Subgroup-size agnostic (Wave4..Wave128), zero global
// score scratch, zero local-memory spills.
import numeric;
import attention_args;
import key_iter;
import reductions_wide;
import vk_common;

// ============================================================================
// 1. Unified KV Cache Append (supports d_k != d_v, strided caches, shared K==V)
// ============================================================================
struct KvAppendArgs {
    Ptr<float, Access.Read> k_in;
    Ptr<float, Access.Read> v_in;
    Ptr<CacheScalar> k_cache;
    Ptr<CacheScalar> v_cache;
    Ptr<uint, Access.Read> step_params; // optional [seq_len, pos_offset, kv_pos] for captured graphs
    uint rows;
    uint n_kv_heads;
    uint d_k;
    uint d_v;
    uint max_ctx;
    uint kv_pos;
    uint k_stride;
    uint v_stride;
    uint k_head_stride;
    uint k_row_stride;
    uint v_head_stride;
    uint v_row_stride;
    uint same_kv_cache;
    uint pad;
};

[shader("compute")] [numthreads(256, 1, 1)]
void vk_kv_append_f32(uniform KvAppendArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint idx = gid.x * kRowWgSize + gtid.x;
    uint t = gid.y;
    if (t >= a.rows) return;

    uint eff_kv_pos = step_word_or(a.step_params, 2u, a.kv_pos, a.k_in);
    uint64_t pos64 = uint64_t(eff_kv_pos) + uint64_t(t);
    if (pos64 >= uint64_t(a.max_ctx)) return;

    uint k_dim = a.n_kv_heads * a.d_k;
    if (idx < k_dim) {
        uint kvh = idx / a.d_k;
        uint d = idx - kvh * a.d_k;
        uint64_t k_dst = uint64_t(kvh) * uint64_t(a.k_head_stride) +
                         pos64 * uint64_t(a.k_row_stride) + uint64_t(d);
        a.k_cache[k_dst] = CacheScalar(a.k_in[uint64_t(t) * uint64_t(a.k_stride) + uint64_t(idx)]);
    }

    if (a.same_kv_cache == 0u) {
        uint v_dim = a.n_kv_heads * a.d_v;
        if (idx < v_dim) {
            uint kvh = idx / a.d_v;
            uint d = idx - kvh * a.d_v;
            uint64_t v_dst = uint64_t(kvh) * uint64_t(a.v_head_stride) +
                             pos64 * uint64_t(a.v_row_stride) + uint64_t(d);
            a.v_cache[v_dst] = CacheScalar(a.v_in[uint64_t(t) * uint64_t(a.v_stride) + uint64_t(idx)]);
        }
    }
}

// ============================================================================
// 2, 3, 4. Unified Online-Softmax Attention (d_k, d_v <= 256 fast & <= 2048 wide)
// ============================================================================
groupshared float s_q[kAttnFastHeadDim];
groupshared float s_m[8];
groupshared float s_l[8];

// Store one bounded score tile per workgroup. Each lane reads four adjacent head channels.
static const uint kDecodeTileKeys = 256u;
groupshared float s_scores[kDecodeTileKeys];
groupshared float s_mass;
groupshared float4 s_value_sums[256];

void attention_score_tile<let D : uint, let WG : uint, let Partial : bool, TIter : IKeyIter>(
    Ptr<float, Access.Read> q, Ptr<CacheScalar, Access.Read> k, Ptr<CacheScalar, Access.Read> v,
    uint k_stride, uint v_stride, float scale, Ptr<float> output, TIter iter)
{
    static const uint lanes_per_key = D / 4u;
    static const uint keys_per_group = WG / lanes_per_key;
    // These SPIR-V 1.6 workgroups have full 32-lane subgroups. Number threads in subgroup order.
    uint lane = WaveGetLaneIndex();
    uint tid = WaveGetWaveIndex() * 32u + lane;
    uint channel = tid % lanes_per_key;
    uint key_in_group = tid / lanes_per_key;
    uint count = iter.count();
    float4 query = load_a16(Ptr<float4, Access.Read>(q) + channel);
    for (uint base = 0u; base < count; base += keys_per_group) {
        uint index = base + key_in_group;
        float score = 0.0f;
        if (index < count) {
            Ptr<CacheScalar, Access.Read> key = k + uint64_t(iter.key_index(index)) * k_stride;
            score = dot(query, load_cache4(Ptr<CacheVector, Access.Read>(key) + channel));
        }
        [ForceUnroll] for (uint mask = lanes_per_key / 2u; mask > 0u; mask >>= 1u) {
            score += WaveReadLaneAt(score, lane ^ mask);
        }
        if (channel == 0u && index < count) s_scores[index] = score * scale;
    }
    GroupMemoryBarrierWithGroupSync();

    // Subtract the tile maximum before exponentiation. Split-KV merges the max and mass later.
    if (tid < 32u) {
        float largest = asfloat(0xff800000u);
        for (uint i = lane; i < count; i += 32u) largest = max(largest, s_scores[i]);
        largest = WaveActiveMax(largest);
        float mass = 0.0f;
        for (uint i = lane; i < count; i += 32u) {
            float weight = exp(s_scores[i] - largest);
            s_scores[i] = weight;
            mass += weight;
        }
        mass = WaveActiveSum(mass);
        if (lane == 0u) {
            if (Partial) {
                output[0] = largest;
                output[1] = mass;
            } else {
                s_mass = mass;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    float4 acc = float4(0.0f);
    for (uint i = key_in_group; i < count; i += keys_per_group) {
        Ptr<CacheScalar, Access.Read> value = v + uint64_t(iter.key_index(i)) * v_stride;
        acc = fma(float4(s_scores[i]), load_cache4(Ptr<CacheVector, Access.Read>(value) + channel), acc);
    }
    s_value_sums[tid] = acc;
    GroupMemoryBarrierWithGroupSync();
    if (key_in_group == 0u) {
        [ForceUnroll] for (uint i = lanes_per_key; i < WG; i += lanes_per_key) {
            acc += s_value_sums[i + channel];
        }
        if (!Partial) acc /= s_mass;
        store_f32x4_maybe_a16(output + (Partial ? 2u : 0u) + channel * 4u, acc);
    }
}

// Each subgroup computes a softmax over its own keys. Only the final maxima,
// masses and value sums cross subgroup boundaries. Each subgroup merges
// successive key tiles using its running maximum for both prefill and Split-KV.
void attention_warp_partition<let D : uint, let Partial : bool, TIter : IKeyIter>(
    Ptr<float, Access.Read> query, Ptr<CacheScalar, Access.Read> k, Ptr<CacheScalar, Access.Read> v,
    uint k_stride, uint v_stride, float scale, TIter keys, Ptr<float> output)
{
    static const uint lanes_per_key = 8u;
    static const uint keys_per_wave = 32u / lanes_per_key;
    static const uint vectors_per_lane = D / (4u * lanes_per_key);
    static const uint rounds = Partial ? 8u : 4u;
    static const uint tile_keys = 4u * keys_per_wave * rounds;

    uint lane = WaveGetLaneIndex(), wave = WaveGetWaveIndex();
    uint channel = lane % lanes_per_key, key = lane / lanes_per_key;
    float4 q[vectors_per_lane], acc[vectors_per_lane];
    [ForceUnroll] for (uint j = 0u; j < vectors_per_lane; ++j) {
        q[j] = load_a16(Ptr<float4, Access.Read>(query) + channel + j * lanes_per_key);
        acc[j] = 0.0f;
    }
    float high = asfloat(0xff800000u);
    float mass = 0.0f;
    for (uint base = 0u; ; base += tile_keys) {
        uint left = keys.count() - base;
        float scores[rounds];
        float tile_high = asfloat(0xff800000u);
        [ForceUnroll] for (uint i = 0u; i < rounds; ++i) {
            uint t = (4u * i + wave) * keys_per_wave + key;
            float score = 0.0f;
            if (t < left) {
                Ptr<CacheVector, Access.Read> row = Ptr<CacheVector, Access.Read>(
                    k + uint64_t(keys.key_index(base + t)) * k_stride);
                [ForceUnroll] for (uint j = 0u; j < vectors_per_lane; ++j) {
                    score += dot(q[j], load_cache4(row + channel + j * lanes_per_key));
                }
            }
            [ForceUnroll] for (uint shift = lanes_per_key / 2u; shift > 0u; shift >>= 1u) {
                score += WaveReadLaneAt(score, lane ^ shift);
            }
            scores[i] = t < left ? score * scale : asfloat(0xff800000u);
            tile_high = max(tile_high, scores[i]);
        }
        tile_high = WaveActiveMax(tile_high);
        if (base != 0u) {
            tile_high = max(high, tile_high);
            float factor = exp(high - tile_high);
            mass *= factor;
            [ForceUnroll] for (uint j = 0u; j < vectors_per_lane; ++j) acc[j] *= factor;
        }
        high = tile_high;
        [ForceUnroll] for (uint i = 0u; i < rounds; ++i) {
            uint t = (4u * i + wave) * keys_per_wave + key;
            if (t < left) {
                float weight = exp(scores[i] - high);
                mass += weight;
                Ptr<CacheVector, Access.Read> row = Ptr<CacheVector, Access.Read>(
                    v + uint64_t(keys.key_index(base + t)) * v_stride);
                [ForceUnroll] for (uint j = 0u; j < vectors_per_lane; ++j) {
                    acc[j] = fma(float4(weight), load_cache4(row + channel + j * lanes_per_key), acc[j]);
                }
            }
        }
        if (left <= tile_keys) break;
    }
    [ForceUnroll] for (uint shift = lanes_per_key; shift < 32u; shift <<= 1u) {
        mass += WaveReadLaneAt(mass, lane ^ shift);
        [ForceUnroll] for (uint j = 0u; j < vectors_per_lane; ++j) {
            acc[j] += WaveReadLaneAt(acc[j], lane ^ shift);
        }
    }
    if (lane == 0u) { s_m[wave] = high; s_l[wave] = mass; }
    if (key == 0u) {
        [ForceUnroll] for (uint j = 0u; j < vectors_per_lane; ++j) {
            s_value_sums[wave * (D / 4u) + channel + j * lanes_per_key] = acc[j];
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (wave == 0u && lane < D / 4u) {
        high = max(max(s_m[0], s_m[1]), max(s_m[2], s_m[3]));
        float4 value = 0.0f;
        mass = 0.0f;
        [ForceUnroll] for (uint w = 0u; w < 4u; ++w) {
            float factor = (s_l[w] > 0.0f) ? exp(s_m[w] - high) : 0.0f;
            value += s_value_sums[w * (D / 4u) + lane] * factor;
            mass += s_l[w] * factor;
        }
        if (Partial && lane == 0u) { output[0] = high; output[1] = mass; }
        store_f32x4_maybe_a16(output + (Partial ? 2u : 0u) + lane * 4u, Partial ? value : value / mass);
    }
}

void attention_aligned_partition<let D : uint, let P : uint, let Partial : bool, TIter : IKeyIter>(
    Ptr<float, Access.Read> q, Ptr<CacheScalar, Access.Read> k, Ptr<CacheScalar, Access.Read> v,
    uint k_stride, uint v_stride, float scale, Ptr<float> output, TIter keys)
{
    if (!Partial && P == 4u) {
        attention_warp_partition<D, false, TIter>(q, k, v, k_stride, v_stride, scale, keys, output);
    } else {
        attention_score_tile<D, P * 32u, Partial, TIter>(q, k, v, k_stride, v_stride, scale, output, keys);
    }
}

// Fast partition-parallel path for d_k <= 256 && d_v <= 256 (0 barriers per key on Wave32).
// Split-KV stores [max, sum of weights, unnormalized values] for the reduction shader.
void attention_partition<let P : uint, let Partial : bool, TIter : IKeyIter>(
    Ptr<float, Access.Read> q_ptr, Ptr<CacheScalar, Access.Read> k_head, Ptr<CacheScalar, Access.Read> v_head,
    uint d_k, uint d_v, uint k_row_stride, uint v_row_stride, float scale,
    Ptr<float> output, uint tid, TIter iter)
{
    static const uint kWgSize = P << 5u;
    uint kv_len = iter.count();
    if (WaveGetLaneCount() == 32u && kv_len > 0u && ((!Partial && P == 4u) || kv_len <= kDecodeTileKeys) &&
        d_k == d_v && (d_k == 32u || d_k == 64u || d_k == 128u) &&
        ((k_row_stride | v_row_stride) & 3u) == 0u &&
        is_ptr_aligned16(q_ptr) && is_ptr_aligned16(k_head) && is_ptr_aligned16(v_head)) {
        if (d_k == 32u) {
            attention_aligned_partition<32u, P, Partial, TIter>(
                q_ptr, k_head, v_head, k_row_stride, v_row_stride, scale, output, iter);
        } else if (d_k == 64u) {
            attention_aligned_partition<64u, P, Partial, TIter>(
                q_ptr, k_head, v_head, k_row_stride, v_row_stride, scale, output, iter);
        } else {
            attention_aligned_partition<128u, P, Partial, TIter>(
                q_ptr, k_head, v_head, k_row_stride, v_row_stride, scale, output, iter);
        }
        return;
    }

    for (uint d = tid; d < d_k; d += kWgSize) {
        s_q[d] = q_ptr[d];
    }
    GroupMemoryBarrierWithGroupSync();

    uint part = tid >> 5u;
    uint lane = tid & 31u;

    float q_reg[8];
    float acc[8];
    [ForceUnroll] for (uint j = 0u; j < 8u; ++j) {
        uint d = lane + (j << 5u);
        q_reg[j] = (d < d_k) ? s_q[d] : 0.0f;
        acc[j] = 0.0f;
    }

    uint64_t krs64 = uint64_t(k_row_stride);
    uint64_t vrs64 = uint64_t(v_row_stride);

    float m_p = kNegInfSentinel;
    float l_p = 0.0f;
    uint n_iters = (kv_len - 1u) / P + 1u;

    for (uint it = 0u; it < n_iters; ++it) {
        uint64_t k_rel = uint64_t(it) * uint64_t(P) + uint64_t(part);
        bool valid = (k_rel < uint64_t(kv_len));
        uint k_idx = valid ? iter.key_index(uint(k_rel)) : 0u;

        float partial_dot = 0.0f;
        if (valid) {
            Ptr<CacheScalar, Access.Read> k_row = k_head + uint64_t(k_idx) * krs64;
            [ForceUnroll] for (uint j = 0u; j < 8u; ++j) {
                if ((j << 5u) < d_k) {
                    uint d = lane + (j << 5u);
                    if (d < d_k) {
                        partial_dot = fma(q_reg[j], float(k_row[d]), partial_dot);
                    }
                }
            }
        }

        float score = part32_reduce_sum(partial_dot, tid, part, lane) * scale;

        if (valid) {
            float alpha = 1.0f;
            float p_w = 1.0f;
            if (l_p == 0.0f) {
                alpha = 0.0f;
                m_p = score;
                p_w = score == asfloat(0xff800000u) ? 0.0f : 1.0f;
                l_p = p_w;
            } else if (score > m_p) {
                alpha = exp(m_p - score);
                l_p = l_p * alpha + 1.0f;
                m_p = score;
            } else {
                p_w = exp(score - m_p);
                l_p += p_w;
            }

            Ptr<CacheScalar, Access.Read> v_row = v_head + uint64_t(k_idx) * vrs64;
            [ForceUnroll] for (uint j = 0u; j < 8u; ++j) {
                if ((j << 5u) < d_v) {
                    uint d = lane + (j << 5u);
                    if (d < d_v) {
                        acc[j] = fma(acc[j], alpha, p_w * float(v_row[d]));
                    }
                }
            }
        }
    }

    if (lane == 0u) {
        s_m[part] = m_p;
        s_l[part] = l_p;
    }
    GroupMemoryBarrierWithGroupSync();

    float m_all = s_m[0];
    [ForceUnroll] for (uint p = 1u; p < P; ++p) {
        if (s_l[p] > 0.0f) {
            m_all = max(m_all, s_m[p]);
        }
    }

    float l_all = 0.0f;
    [ForceUnroll] for (uint p = 0u; p < P; ++p) {
        if (s_l[p] > 0.0f) {
            l_all += s_l[p] * exp(s_m[p] - m_all);
        }
    }

    if (Partial && tid == 0u) {
        output[0] = m_all;
        output[1] = l_all;
    }
    float w_p = (l_p > 0.0f) ? exp(m_p - m_all) : 0.0f;
    float inv_l = (l_all > 0.0f) ? (1.0f / l_all) : 0.0f;

    [ForceUnroll] for (uint j = 0u; j < 8u; ++j) {
        if ((j << 5u) < d_v) {
            uint d = lane + (j << 5u);
            float sum = part32_cross_sum<P>((d < d_v) ? (acc[j] * w_p) : 0.0f, tid, part, lane);
            if (part == 0u && d < d_v) {
                if (Partial) output[2u + d] = sum;
                else output[d] = sum * inv_l;
            }
        }
    }
}

void attention_online_impl<let P : uint, TIter : IKeyIter>(
    AttentionArgs a, uint q_row, uint h, uint tid, TIter iter)
{
    if (q_row >= a.rows || h >= a.n_heads) return;
    Ptr<float> output = a.y + uint64_t(q_row) * a.o_stride + uint64_t(h) * a.d_v;
    if (iter.count() == 0u) {
        for (uint d = tid; d < a.d_v; d += P << 5u) output[d] = 0.0f;
        return;
    }
    uint kvh = h / (a.n_heads / a.n_kv_heads);
    attention_partition<P, false, TIter>(
        a.q + uint64_t(q_row) * a.q_stride + uint64_t(h) * a.d_k,
        Ptr<CacheScalar, Access.Read>(a.k_cache + uint64_t(kvh) * a.k_head_stride), Ptr<CacheScalar, Access.Read>(a.v_cache + uint64_t(kvh) * a.v_head_stride),
        a.d_k, a.d_v, a.k_row_stride, a.v_row_stride, a.scale, output, tid, iter);
}

// Wide path for d_k > 256 || d_v > 256 (up to 2048): 4-key tiled online softmax, 0 spills.
void attention_wide_impl<TIter : IKeyIter>(
    AttentionArgs a, uint q_row, uint h, uint tid, TIter iter)
{
    if (q_row >= a.rows || h >= a.n_heads) return;

    uint kv_len = iter.count();
    Ptr<float> o_ptr =
        a.y + uint64_t(q_row) * uint64_t(a.o_stride) + uint64_t(h) * uint64_t(a.d_v);
    if (kv_len == 0u) {
        [ForceUnroll] for (uint u = 0u; u < 8u; ++u) {
            if ((u << 8u) < a.d_v) {
                uint d = tid + (u << 8u);
                if (d < a.d_v) {
                    o_ptr[d] = 0.0f;
                }
            }
        }
        return;
    }

    Ptr<float, Access.Read> q_ptr =
        a.q + uint64_t(q_row) * uint64_t(a.q_stride) + uint64_t(h) * uint64_t(a.d_k);

    float qt[8];
    float acc[8];
    [ForceUnroll] for (uint u = 0u; u < 8u; ++u) {
        uint d = tid + (u << 8u);
        qt[u] = (d < a.d_k) ? q_ptr[d] : 0.0f;
        acc[u] = 0.0f;
    }

    uint group_size = a.n_heads / a.n_kv_heads;
    uint kvh = h / group_size;
    Ptr<CacheScalar, Access.Read> k_head = Ptr<CacheScalar, Access.Read>(a.k_cache + uint64_t(kvh) * uint64_t(a.k_head_stride));
    Ptr<CacheScalar, Access.Read> v_head = Ptr<CacheScalar, Access.Read>(a.v_cache + uint64_t(kvh) * uint64_t(a.v_head_stride));
    uint64_t krs64 = uint64_t(a.k_row_stride);
    uint64_t vrs64 = uint64_t(a.v_row_stride);

    float m = kNegInfSentinel;
    float l = 0.0f;

    uint j = 0u;
    while (j < kv_len) {
        uint cnt = min(4u, kv_len - j);
        uint idx0 = iter.key_index(j);
        uint idx1 = (cnt > 1u) ? iter.key_index(j + 1u) : idx0;
        uint idx2 = (cnt > 2u) ? iter.key_index(j + 2u) : idx0;
        uint idx3 = (cnt > 3u) ? iter.key_index(j + 3u) : idx0;

        Ptr<CacheScalar, Access.Read> kj0 = k_head + uint64_t(idx0) * krs64;
        Ptr<CacheScalar, Access.Read> kj1 = k_head + uint64_t(idx1) * krs64;
        Ptr<CacheScalar, Access.Read> kj2 = k_head + uint64_t(idx2) * krs64;
        Ptr<CacheScalar, Access.Read> kj3 = k_head + uint64_t(idx3) * krs64;

        float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
        [ForceUnroll] for (uint u = 0u; u < 8u; ++u) {
            if ((u << 8u) < a.d_k) {
                uint d = tid + (u << 8u);
                if (d < a.d_k) {
                    float qtu = qt[u];
                    d0 = fma(qtu, float(kj0[d]), d0);
                    if (cnt > 1u) d1 = fma(qtu, float(kj1[d]), d1);
                    if (cnt > 2u) d2 = fma(qtu, float(kj2[d]), d2);
                    if (cnt > 3u) d3 = fma(qtu, float(kj3[d]), d3);
                }
            }
        }

        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        wg_reduce_sum4(d0, d1, d2, d3, tid, s0, s1, s2, s3);
        s0 *= a.scale;
        s1 *= a.scale;
        s2 *= a.scale;
        s3 *= a.scale;

        float m_tile = s0;
        if (cnt > 1u) m_tile = max(m_tile, s1);
        if (cnt > 2u) m_tile = max(m_tile, s2);
        if (cnt > 3u) m_tile = max(m_tile, s3);

        float m_new = (l > 0.0f) ? max(m, m_tile) : m_tile;
        float alpha = (l > 0.0f) ? exp(m - m_new) : 0.0f;
        float p0 = exp(s0 - m_new);
        float p1 = (cnt > 1u) ? exp(s1 - m_new) : 0.0f;
        float p2 = (cnt > 2u) ? exp(s2 - m_new) : 0.0f;
        float p3 = (cnt > 3u) ? exp(s3 - m_new) : 0.0f;
        l = l * alpha + (p0 + p1 + p2 + p3);
        m = m_new;

        Ptr<CacheScalar, Access.Read> vj0 = v_head + uint64_t(idx0) * vrs64;
        Ptr<CacheScalar, Access.Read> vj1 = v_head + uint64_t(idx1) * vrs64;
        Ptr<CacheScalar, Access.Read> vj2 = v_head + uint64_t(idx2) * vrs64;
        Ptr<CacheScalar, Access.Read> vj3 = v_head + uint64_t(idx3) * vrs64;

        [ForceUnroll] for (uint u = 0u; u < 8u; ++u) {
            if ((u << 8u) < a.d_v) {
                uint d = tid + (u << 8u);
                if (d < a.d_v) {
                    float v_acc = fma(acc[u], alpha, p0 * float(vj0[d]));
                    if (cnt > 1u) v_acc = fma(p1, float(vj1[d]), v_acc);
                    if (cnt > 2u) v_acc = fma(p2, float(vj2[d]), v_acc);
                    if (cnt > 3u) v_acc = fma(p3, float(vj3[d]), v_acc);
                    acc[u] = v_acc;
                }
            }
        }
        j += cnt;
    }

    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    [ForceUnroll] for (uint u = 0u; u < 8u; ++u) {
        if ((u << 8u) < a.d_v) {
            uint d = tid + (u << 8u);
            if (d < a.d_v) {
                o_ptr[d] = acc[u] * inv_l;
            }
        }
    }
}

// Each workgroup reduces eight output channels for one query head. Recomputing the
// scores in neighboring workgroups exposes parallelism without a global partials pass.
void attention_decode_slice<let D : uint>(AttentionArgs a, uint head, uint slice,
                                           DenseRangeIter keys) {
    static const uint score_lanes = D == 64u ? 4u : 8u;
    static const uint vectors_per_lane = D / (4u * score_lanes);
    uint lane = WaveGetLaneIndex();
    uint wave = WaveGetWaveIndex();
    uint tid = wave * 32u + lane;
    uint channel = tid % score_lanes;
    uint kv_head = head / (a.n_heads / a.n_kv_heads);
    Ptr<float4, Access.Read> query = Ptr<float4, Access.Read>(a.q + uint64_t(head) * D);
    Ptr<CacheScalar> k_cache = a.k_cache + uint64_t(kv_head) * a.k_head_stride;
    Ptr<CacheScalar> v_cache = a.v_cache + uint64_t(kv_head) * a.v_head_stride;
    float4 q[vectors_per_lane];
    [ForceUnroll] for (uint j = 0u; j < vectors_per_lane; ++j) {
        q[j] = load_a16(query + channel + j * score_lanes);
    }

    // With an inline append all workgroups read this token from the inputs. Exactly
    // one workgroup writes each cache head. Earlier positions are already resident.
    uint current = keys.start + keys.len - 1u;
#ifdef VELRI_F16_KV
    // The half-cache path uses a separate append or the fused QKV projection.
    bool append = false;
#else
    bool append = a.k_new != nullptr;
#endif
    Ptr<float, Access.Read> k_new = a.k_new;
    Ptr<float, Access.Read> v_new = a.v_new;
    if (append) {
        k_new = k_new + uint64_t(kv_head) * D;
        v_new = (a.k_cache == a.v_cache) ? k_new : v_new + uint64_t(kv_head) * D;
        if (slice == 0u && head % (a.n_heads / a.n_kv_heads) == 0u && tid < D) {
            k_cache[uint64_t(current) * a.k_row_stride + tid] = CacheScalar(k_new[tid]);
            if (a.k_cache != a.v_cache) {
                v_cache[uint64_t(current) * a.v_row_stride + tid] = CacheScalar(v_new[tid]);
            }
        }
    }

    // A captured primary graph can also serve long windows when split scratch is
    // unavailable. Merge bounded tiles using their individual softmax maxima.
    float max_so_far = asfloat(0xff800000u);
    float mass_so_far = 0.0f;
    float4 acc_so_far = float4(0.0f);
    for (uint base = 0u; ; base += kDecodeTileKeys) {
        uint count = min(keys.len - base, kDecodeTileKeys);
        uint start = keys.start + base;
        for (uint block = 0u; block < count; block += 256u / score_lanes) {
            uint t = block + tid / score_lanes;
            float dot_qk = 0.0f;
            if (t < count) {
                uint pos = start + t;
                Ptr<CacheScalar, Access.Read> row = (append && pos == current)
                    ? Ptr<CacheScalar, Access.Read>(k_new) : Ptr<CacheScalar, Access.Read>(k_cache + uint64_t(pos) * a.k_row_stride);
                [ForceUnroll] for (uint j = 0u; j < vectors_per_lane; ++j) {
                    dot_qk += dot(q[j], load_cache4(Ptr<CacheVector, Access.Read>(row) + channel + j * score_lanes));
                }
            }
            [ForceUnroll] for (uint step = score_lanes / 2u; step > 0u; step >>= 1u) {
                dot_qk += WaveReadLaneAt(dot_qk, lane ^ step);
            }
            if (channel == 0u && t < count) s_scores[t] = dot_qk * a.scale;
        }
        GroupMemoryBarrierWithGroupSync();

        float largest = asfloat(0xff800000u);
        for (uint t = lane; t < count; t += 32u) largest = max(largest, s_scores[t]);
        largest = WaveActiveMax(largest);
        uint half_slice = tid % 2u;
        float mass = 0.0f;
        float4 acc = float4(0.0f);
        for (uint t = tid / 2u; t < count; t += 128u) {
            uint pos = start + t;
            float weight = exp(s_scores[t] - largest);
            Ptr<CacheScalar, Access.Read> row = (append && pos == current)
                ? Ptr<CacheScalar, Access.Read>(v_new) : Ptr<CacheScalar, Access.Read>(v_cache + uint64_t(pos) * a.v_row_stride);
            acc = fma(float4(weight), load_cache4(Ptr<CacheVector, Access.Read>(row) + slice * 2u + half_slice), acc);
            mass += weight;
        }
        [ForceUnroll] for (uint step = 2u; step < 32u; step <<= 1u) {
            acc += WaveReadLaneAt(acc, lane ^ step);
            mass += WaveReadLaneAt(mass, lane ^ step);
        }
        if (lane < 2u) s_value_sums[wave * 2u + half_slice] = acc;
        if (lane == 0u) s_l[wave] = mass;
        GroupMemoryBarrierWithGroupSync();

        bool last = (keys.len - base == count);
        if (tid < 2u) {
            mass = s_l[0];
            [ForceUnroll] for (uint w = 1u; w < 8u; ++w) {
                acc += s_value_sums[w * 2u + half_slice];
                mass += s_l[w];
            }
            if (base == 0u) {
                acc_so_far = acc; mass_so_far = mass; max_so_far = largest;
            } else {
                float high = max(max_so_far, largest);
                float before = exp(max_so_far - high);
                float here = exp(largest - high);
                acc_so_far = acc_so_far * before + acc * here;
                mass_so_far = mass_so_far * before + mass * here;
                max_so_far = high;
            }
            if (last) {
                store_f32x4_maybe_a16(a.y + uint64_t(head) * D + slice * 8u + half_slice * 4u,
                                      acc_so_far / mass_so_far);
            }
        }
        if (last) break;
        GroupMemoryBarrierWithGroupSync();
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_attention_decode_f32(uniform AttentionArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint eff_kv_pos = attention_kv_pos(a);
    DenseRangeIter iter = make_dense_range_iter(eff_kv_pos, 0u, a.rows, a.max_ctx, a.sliding_window, a.causal);
    if (a.value_slices > 1u && iter.len > 0u) {
        if (a.d_k == 32u) attention_decode_slice<32u>(a, gid.x, gid.y, iter);
        else if (a.d_k == 64u) attention_decode_slice<64u>(a, gid.x, gid.y, iter);
        else attention_decode_slice<128u>(a, gid.x, gid.y, iter);
        return;
    }
    if (gid.y != 0u) return;
    attention_online_impl<8u, DenseRangeIter>(a, 0u, gid.x, gtid.x, iter);
}

[shader("compute")] [numthreads(128, 1, 1)]
void vk_attention_prefill_f32(uniform AttentionArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint eff_kv_pos = attention_kv_pos(a);
    DenseRangeIter iter = make_dense_range_iter(eff_kv_pos, gid.x, a.rows, a.max_ctx, a.sliding_window, a.causal);
    attention_online_impl<4u, DenseRangeIter>(a, gid.x, gid.y, gtid.x, iter);
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_attention_wide_f32(uniform AttentionArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint eff_kv_pos = attention_kv_pos(a);
    DenseRangeIter iter = make_dense_range_iter(eff_kv_pos, gid.x, a.rows, a.max_ctx, a.sliding_window, a.causal);
    attention_wide_impl<DenseRangeIter>(a, gid.x, gid.y, gtid.x, iter);
}

// ============================================================================
// 5. FlashDecoding Split-KV Decode (saturates high-SM GPUs on long contexts)
// ============================================================================


[shader("compute")] [numthreads(128, 1, 1)]
void vk_attention_splitkv_f32(uniform SplitKvArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint h = gid.x;
    uint sp = gid.y;
    uint tid = gtid.x;
    if (h >= a.n_heads || sp >= a.n_splits) return;

    uint eff_kv_pos = cache_step_pos(a.step_params, a.kv_pos, a.k_cache);
    DenseRangeIter full = make_dense_range_iter(eff_kv_pos, 0u, 1u, a.max_ctx, a.sliding_window, a.causal);
    uint total_kv = full.count();
    uint per_split = (total_kv > 0u) ? (((total_kv - 1u) / a.n_splits) + 1u) : 0u;
    uint sp_start = sp * per_split;
    uint sp_end = min(sp_start + per_split, total_kv);
    uint sp_len = (sp_start < sp_end) ? (sp_end - sp_start) : 0u;

    Ptr<float> slot_ptr =
        a.partials + (uint64_t(h) * uint64_t(a.n_splits) + uint64_t(sp)) * uint64_t(a.part_stride);
    if (sp_len == 0u) {
        if (tid == 0u) {
            slot_ptr[0] = kNegInfSentinel;
            slot_ptr[1] = 0.0f;
        }
        return;
    }
    DenseRangeIter iter;
    iter.start = full.start + sp_start;
    iter.len = sp_len;
    uint kvh = h / (a.n_heads / a.n_kv_heads);
    if (WaveGetLaneCount() == 32u && a.d_k == a.d_v &&
        (a.d_k == 32u || a.d_k == 64u || a.d_k == 128u) &&
        ((a.k_row_stride | a.v_row_stride) & 3u) == 0u && is_ptr_aligned16(a.q) &&
        is_ptr_aligned16(a.k_cache + uint64_t(kvh) * a.k_head_stride) &&
        is_ptr_aligned16(a.v_cache + uint64_t(kvh) * a.v_head_stride)) {
        Ptr<float, Access.Read> q = a.q + uint64_t(h) * a.d_k;
        Ptr<CacheScalar, Access.Read> k = a.k_cache + uint64_t(kvh) * a.k_head_stride;
        Ptr<CacheScalar, Access.Read> v = a.v_cache + uint64_t(kvh) * a.v_head_stride;
        if (a.d_k == 32u) {
            attention_warp_partition<32u, true, DenseRangeIter>(q, k, v, a.k_row_stride, a.v_row_stride, a.scale, iter, slot_ptr);
        } else if (a.d_k == 64u) {
            attention_warp_partition<64u, true, DenseRangeIter>(q, k, v, a.k_row_stride, a.v_row_stride, a.scale, iter, slot_ptr);
        } else {
            attention_warp_partition<128u, true, DenseRangeIter>(q, k, v, a.k_row_stride, a.v_row_stride, a.scale, iter, slot_ptr);
        }
        return;
    }
    attention_partition<4u, true, DenseRangeIter>(
        a.q + uint64_t(h) * a.d_k,
        Ptr<CacheScalar, Access.Read>(a.k_cache + uint64_t(kvh) * a.k_head_stride), Ptr<CacheScalar, Access.Read>(a.v_cache + uint64_t(kvh) * a.v_head_stride),
        a.d_k, a.d_v, a.k_row_stride, a.v_row_stride, a.scale, slot_ptr, tid, iter);
}

struct SplitKvReduceArgs {
    Ptr<float, Access.Read> partials;
    Ptr<float> out;
    uint n_heads;
    uint d_v;
    uint part_stride;
    uint n_splits;
    uint value_slices;
};

static const uint kMaxKvSplits = 32u;
groupshared float s_skv_m[kMaxKvSplits];
groupshared float s_skv_l[kMaxKvSplits];

// A subgroup merges a full set of partitions for one slice of the output head.
void attention_reduce_slice(SplitKvReduceArgs a, uint head, uint slice) {
    uint lane = WaveGetLaneIndex();
    if (a.n_splits == 32u) {
        // One lane per partition; one subgroup produces four adjacent output channels.
        Ptr<float, Access.Read> p = a.partials + (uint64_t(head) * 32u + lane) * a.part_stride;
        float mass = p[1];
        float high = mass > 0.0f ? p[0] : asfloat(0xff800000u);
        float largest = WaveActiveMax(high);
        float4 value = 0.0f;
        if (mass > 0.0f) {
            float weight = exp(high - largest);
            mass *= weight;
            value = load_f32x4_maybe_a16(p + 2u + slice * 4u) * weight;
        }
        mass = WaveActiveSum(mass);
        value = WaveActiveSum(value);
        if (lane == 0u)
            store_f32x4_maybe_a16(a.out + uint64_t(head) * a.d_v + slice * 4u,
                                  mass > 0.0f ? value / mass : float4(0.0f));
        return;
    }
    uint split = lane / 2u;
    uint channel = slice * 8u + (lane % 2u) * 4u;
    Ptr<float, Access.Read> partial = a.partials + uint64_t(head) * a.n_splits * a.part_stride;
    float high = asfloat(0xff800000u);
    float mass = 0.0f;
    if (split < a.n_splits) {
        partial = partial + uint64_t(split) * a.part_stride;
        if (partial[1] > 0.0f) {
            mass = partial[1];
            high = partial[0];
        }
    }
    float largest = WaveActiveMax(high);
    float4 acc = float4(0.0f);
    if (mass > 0.0f) {
        float weight = exp(high - largest);
        mass *= weight;
        acc = load_f32x4_maybe_a16(partial + 2u + channel) * weight;
    }
    [ForceUnroll] for (uint step = 2u; step < 32u; step <<= 1u) {
        acc += WaveReadLaneAt(acc, lane ^ step);
        mass += WaveReadLaneAt(mass, lane ^ step);
    }
    if (split == 0u) {
        store_f32x4_maybe_a16(a.out + uint64_t(head) * a.d_v + channel,
                              mass > 0.0f ? acc / mass : float4(0.0f));
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_attention_splitkv_reduce_f32(uniform SplitKvReduceArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    if (a.value_slices > 0u) {
        if (WaveGetWaveIndex() == 0u) attention_reduce_slice(a, gid.x, gid.y);
        return;
    }
    uint h = gid.x;
    uint tid = gtid.x;
    if (h >= a.n_heads) return;

    uint eff_splits = min(a.n_splits, kMaxKvSplits);
    uint64_t head_base = uint64_t(h) * uint64_t(a.n_splits) * uint64_t(a.part_stride);
    uint64_t stride64 = uint64_t(a.part_stride);
    if (tid < eff_splits) {
        Ptr<float, Access.Read> sp_ptr = a.partials + head_base + uint64_t(tid) * stride64;
        s_skv_m[tid] = sp_ptr[0];
        s_skv_l[tid] = sp_ptr[1];
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid >= a.d_v) return;

    float m_all = kNegInfSentinel;
    bool any_valid = false;
    for (uint sp = 0u; sp < eff_splits; ++sp) {
        if (s_skv_l[sp] > 0.0f) {
            m_all = any_valid ? max(m_all, s_skv_m[sp]) : s_skv_m[sp];
            any_valid = true;
        }
    }

    Ptr<float> o_ptr = a.out + uint64_t(h) * uint64_t(a.d_v);
    if (!any_valid) {
        o_ptr[tid] = 0.0f;
        return;
    }

    float l_all = 0.0f;
    float acc = 0.0f;
    for (uint sp = 0u; sp < eff_splits; ++sp) {
        float l_sp = s_skv_l[sp];
        if (l_sp > 0.0f) {
            float w = exp(s_skv_m[sp] - m_all);
            l_all += l_sp * w;
            acc = fma(a.partials[head_base + uint64_t(sp) * stride64 + 2u + uint64_t(tid)], w, acc);
        }
    }
    o_ptr[tid] = (l_all > 0.0f) ? (acc / l_all) : 0.0f;
}

[shader("compute")] [numthreads(32, 1, 1)]
void vk_attention_merge64_f32(uniform SplitKvReduceArgs a, uint3 gid : SV_GroupID) {
    uint lane = WaveGetLaneIndex(), head = gid.x, slice = gid.y;
    Ptr<float, Access.Read> p = a.partials + (uint64_t(head) * 64u + lane) * a.part_stride;
    float mass[2], high[2];
    [ForceUnroll] for (uint i = 0u; i < 2u; ++i) {
        Ptr<float, Access.Read> part = p + i * 32u * a.part_stride;
        mass[i] = part[1];
        high[i] = mass[i] > 0.0f ? part[0] : asfloat(0xff800000u);
    }
    float largest = WaveActiveMax(max(high[0], high[1])), total = 0.0f;
    float4 value = 0.0f;
    [ForceUnroll] for (uint i = 0u; i < 2u; ++i) {
        if (mass[i] > 0.0f) {
            float factor = exp(high[i] - largest);
            total += mass[i] * factor;
            value = fma(float4(factor), load_f32x4_maybe_a16(p + i * 32u * a.part_stride + 2u + slice * 4u), value);
        }
    }
    total = WaveActiveSum(total);
    value = WaveActiveSum(value);
    if (lane == 0u) store_f32x4_maybe_a16(a.out + uint64_t(head) * a.d_v + 4u * slice,
                                          total > 0.0f ? value / total : float4(0.0f));
}
