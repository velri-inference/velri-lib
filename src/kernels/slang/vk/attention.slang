// Vulkan KV-Cache Append and Unified Online-Softmax Attention (F32, d_k/d_v up to 2048,
// compile-time IKeyIter generic). Subgroup-size agnostic (Wave4..Wave128), zero global
// score scratch, zero local-memory spills.
import numeric;
import key_iter;
import reductions_wide;
import vk_common;

// ============================================================================
// 1. Unified KV Cache Append (supports d_k != d_v, strided caches, shared K==V)
// ============================================================================
struct KvAppendArgs {
    Ptr<float, Access.Read> k_in;
    Ptr<float, Access.Read> v_in;
    Ptr<float> k_cache;
    Ptr<float> v_cache;
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

    uint eff_kv_pos = (a.step_params != nullptr) ? a.step_params[2] : a.kv_pos;
    uint64_t pos64 = uint64_t(eff_kv_pos) + uint64_t(t);
    if (pos64 >= uint64_t(a.max_ctx)) return;

    uint k_dim = a.n_kv_heads * a.d_k;
    if (idx < k_dim) {
        uint kvh = idx / a.d_k;
        uint d = idx - kvh * a.d_k;
        uint64_t k_dst = uint64_t(kvh) * uint64_t(a.k_head_stride) +
                         pos64 * uint64_t(a.k_row_stride) + uint64_t(d);
        a.k_cache[k_dst] = a.k_in[uint64_t(t) * uint64_t(a.k_stride) + uint64_t(idx)];
    }

    if (a.same_kv_cache == 0u) {
        uint v_dim = a.n_kv_heads * a.d_v;
        if (idx < v_dim) {
            uint kvh = idx / a.d_v;
            uint d = idx - kvh * a.d_v;
            uint64_t v_dst = uint64_t(kvh) * uint64_t(a.v_head_stride) +
                             pos64 * uint64_t(a.v_row_stride) + uint64_t(d);
            a.v_cache[v_dst] = a.v_in[uint64_t(t) * uint64_t(a.v_stride) + uint64_t(idx)];
        }
    }
}

// ============================================================================
// 2, 3, 4. Unified Online-Softmax Attention (d_k, d_v <= 256 fast & <= 2048 wide)
// ============================================================================
struct AttentionArgs {
    Ptr<float, Access.Read> q;
    Ptr<float, Access.Read> k_cache;
    Ptr<float, Access.Read> v_cache;
    Ptr<float> y;
    Ptr<uint, Access.Read> step_params; // optional [seq_len, pos_offset, kv_pos] for captured graphs
    uint rows;
    uint n_heads;
    uint n_kv_heads;
    uint d_k;
    uint d_v;
    uint max_ctx;
    uint kv_pos;
    uint q_stride;
    uint o_stride;
    uint k_head_stride;
    uint k_row_stride;
    uint v_head_stride;
    uint v_row_stride;
    uint sliding_window;
    uint causal;
    float scale;
};

groupshared float s_q[kAttnFastHeadDim];
groupshared float s_m[8];
groupshared float s_l[8];

// Fast partition-parallel path for d_k <= 256 && d_v <= 256 (0 barriers per key on Wave32).
// Split-KV stores [max, sum of weights, unnormalized values] for the reduction shader.
void attention_partition<let P : uint, let Partial : bool, TIter : IKeyIter>(
    Ptr<float, Access.Read> q_ptr, Ptr<float, Access.Read> k_head, Ptr<float, Access.Read> v_head,
    uint d_k, uint d_v, uint k_row_stride, uint v_row_stride, float scale,
    Ptr<float> output, uint tid, TIter iter)
{
    static const uint kWgSize = P << 5u;
    uint kv_len = iter.count();
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
            Ptr<float, Access.Read> k_row = k_head + uint64_t(k_idx) * krs64;
            [ForceUnroll] for (uint j = 0u; j < 8u; ++j) {
                if ((j << 5u) < d_k) {
                    uint d = lane + (j << 5u);
                    if (d < d_k) {
                        partial_dot = fma(q_reg[j], k_row[d], partial_dot);
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
                l_p = 1.0f;
            } else if (score > m_p) {
                alpha = exp(m_p - score);
                l_p = l_p * alpha + 1.0f;
                m_p = score;
            } else {
                p_w = exp(score - m_p);
                l_p += p_w;
            }

            Ptr<float, Access.Read> v_row = v_head + uint64_t(k_idx) * vrs64;
            [ForceUnroll] for (uint j = 0u; j < 8u; ++j) {
                if ((j << 5u) < d_v) {
                    uint d = lane + (j << 5u);
                    if (d < d_v) {
                        acc[j] = fma(acc[j], alpha, p_w * v_row[d]);
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
        a.k_cache + uint64_t(kvh) * a.k_head_stride, a.v_cache + uint64_t(kvh) * a.v_head_stride,
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
    Ptr<float, Access.Read> k_head = a.k_cache + uint64_t(kvh) * uint64_t(a.k_head_stride);
    Ptr<float, Access.Read> v_head = a.v_cache + uint64_t(kvh) * uint64_t(a.v_head_stride);
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

        Ptr<float, Access.Read> kj0 = k_head + uint64_t(idx0) * krs64;
        Ptr<float, Access.Read> kj1 = k_head + uint64_t(idx1) * krs64;
        Ptr<float, Access.Read> kj2 = k_head + uint64_t(idx2) * krs64;
        Ptr<float, Access.Read> kj3 = k_head + uint64_t(idx3) * krs64;

        float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
        [ForceUnroll] for (uint u = 0u; u < 8u; ++u) {
            if ((u << 8u) < a.d_k) {
                uint d = tid + (u << 8u);
                if (d < a.d_k) {
                    float qtu = qt[u];
                    d0 = fma(qtu, kj0[d], d0);
                    if (cnt > 1u) d1 = fma(qtu, kj1[d], d1);
                    if (cnt > 2u) d2 = fma(qtu, kj2[d], d2);
                    if (cnt > 3u) d3 = fma(qtu, kj3[d], d3);
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

        Ptr<float, Access.Read> vj0 = v_head + uint64_t(idx0) * vrs64;
        Ptr<float, Access.Read> vj1 = v_head + uint64_t(idx1) * vrs64;
        Ptr<float, Access.Read> vj2 = v_head + uint64_t(idx2) * vrs64;
        Ptr<float, Access.Read> vj3 = v_head + uint64_t(idx3) * vrs64;

        [ForceUnroll] for (uint u = 0u; u < 8u; ++u) {
            if ((u << 8u) < a.d_v) {
                uint d = tid + (u << 8u);
                if (d < a.d_v) {
                    float v_acc = fma(acc[u], alpha, p0 * vj0[d]);
                    if (cnt > 1u) v_acc = fma(p1, vj1[d], v_acc);
                    if (cnt > 2u) v_acc = fma(p2, vj2[d], v_acc);
                    if (cnt > 3u) v_acc = fma(p3, vj3[d], v_acc);
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

[shader("compute")] [numthreads(256, 1, 1)]
void vk_attention_decode_f32(uniform AttentionArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint eff_kv_pos = (a.step_params != nullptr) ? a.step_params[2] : a.kv_pos;
    DenseRangeIter iter = make_dense_range_iter(eff_kv_pos, 0u, a.rows, a.max_ctx, a.sliding_window, a.causal);
    attention_online_impl<8u, DenseRangeIter>(a, 0u, gid.x, gtid.x, iter);
}

[shader("compute")] [numthreads(128, 1, 1)]
void vk_attention_prefill_f32(uniform AttentionArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint eff_kv_pos = (a.step_params != nullptr) ? a.step_params[2] : a.kv_pos;
    DenseRangeIter iter = make_dense_range_iter(eff_kv_pos, gid.x, a.rows, a.max_ctx, a.sliding_window, a.causal);
    attention_online_impl<4u, DenseRangeIter>(a, gid.x, gid.y, gtid.x, iter);
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_attention_wide_f32(uniform AttentionArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint eff_kv_pos = (a.step_params != nullptr) ? a.step_params[2] : a.kv_pos;
    DenseRangeIter iter = make_dense_range_iter(eff_kv_pos, gid.x, a.rows, a.max_ctx, a.sliding_window, a.causal);
    attention_wide_impl<DenseRangeIter>(a, gid.x, gid.y, gtid.x, iter);
}

// ============================================================================
// 5. FlashDecoding Split-KV Decode (saturates high-SM GPUs on long contexts)
// ============================================================================
struct SplitKvArgs {
    Ptr<float, Access.Read> q;
    Ptr<float, Access.Read> k_cache;
    Ptr<float, Access.Read> v_cache;
    Ptr<float> partials; // per (h, sp): [m, l, unnormalized_acc[0..d_v-1]], stride = part_stride
    Ptr<uint, Access.Read> step_params; // optional [seq_len, pos_offset, kv_pos] for captured graphs
    uint n_heads;
    uint n_kv_heads;
    uint d_k;
    uint d_v;
    uint max_ctx;
    uint kv_pos;
    uint part_stride;
    uint k_head_stride;
    uint k_row_stride;
    uint v_head_stride;
    uint v_row_stride;
    uint sliding_window;
    uint causal;
    uint n_splits;
    uint pad;
    float scale;
};

[shader("compute")] [numthreads(128, 1, 1)]
void vk_attention_splitkv_f32(uniform SplitKvArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint h = gid.x;
    uint sp = gid.y;
    uint tid = gtid.x;
    if (h >= a.n_heads || sp >= a.n_splits) return;

    uint eff_kv_pos = (a.step_params != nullptr) ? a.step_params[2] : a.kv_pos;
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
    attention_partition<4u, true, DenseRangeIter>(
        a.q + uint64_t(h) * a.d_k,
        a.k_cache + uint64_t(kvh) * a.k_head_stride, a.v_cache + uint64_t(kvh) * a.v_head_stride,
        a.d_k, a.d_v, a.k_row_stride, a.v_row_stride, a.scale, slot_ptr, tid, iter);
}

struct SplitKvReduceArgs {
    Ptr<float, Access.Read> partials;
    Ptr<float> out;
    uint n_heads;
    uint d_v;
    uint part_stride;
    uint n_splits;
};

static const uint kMaxKvSplits = 32u;
groupshared float s_skv_m[kMaxKvSplits];
groupshared float s_skv_l[kMaxKvSplits];

[shader("compute")] [numthreads(256, 1, 1)]
void vk_attention_splitkv_reduce_f32(uniform SplitKvReduceArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
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
