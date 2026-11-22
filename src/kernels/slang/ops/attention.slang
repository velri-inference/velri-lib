// Shared CPU kernels for KV-cache append and Unified Online-Softmax Scaled Dot-Product
// Attention (MHA / GQA / MQA / wide latent / MLA, d_k & d_v up to 2048, compile-time IKeyIter).
module attention;

import numeric;
import activations;
import key_iter;

public void kv_append_cpu(
    Ptr<float, Access.Read> k_in, uint k_stride,
    Ptr<float, Access.Read> v_in, uint v_stride,
    Ptr<float> k_cache, uint k_head_stride, uint k_row_stride,
    Ptr<float> v_cache, uint v_head_stride, uint v_row_stride,
    uint same_kv_cache, uint kvh0, uint kvh1,
    uint active_t, uint kv_pos, uint d_k, uint d_v)
{
    uint64_t ks64 = uint64_t(k_stride);
    uint64_t vs64 = uint64_t(v_stride);
    uint64_t khs64 = uint64_t(k_head_stride);
    uint64_t krs64 = uint64_t(k_row_stride);
    uint64_t vhs64 = uint64_t(v_head_stride);
    uint64_t vrs64 = uint64_t(v_row_stride);
    uint64_t dk64 = uint64_t(d_k);
    uint64_t dv64 = uint64_t(d_v);
    uint64_t kvp64 = uint64_t(kv_pos);

    for (uint kvh = kvh0; kvh < kvh1; ++kvh) {
        uint64_t k_head_base = uint64_t(kvh) * khs64;
        uint64_t k_in_head = uint64_t(kvh) * dk64;
        for (uint t = 0u; t < active_t; ++t) {
            Ptr<float, Access.Read> k_src = k_in + uint64_t(t) * ks64 + k_in_head;
            Ptr<float> k_dst = k_cache + k_head_base + (kvp64 + uint64_t(t)) * krs64;
            for (uint d = 0u; d < d_k; ++d) {
                k_dst[d] = k_src[d];
            }
        }
        if (same_kv_cache == 0u) {
            uint64_t v_head_base = uint64_t(kvh) * vhs64;
            uint64_t v_in_head = uint64_t(kvh) * dv64;
            for (uint t = 0u; t < active_t; ++t) {
                Ptr<float, Access.Read> v_src = v_in + uint64_t(t) * vs64 + v_in_head;
                Ptr<float> v_dst = v_cache + v_head_base + (kvp64 + uint64_t(t)) * vrs64;
                for (uint d = 0u; d < d_v; ++d) {
                    v_dst[d] = v_src[d];
                }
            }
        }
    }
}

void attention_online_core_cpu<let MAX_DK : uint, let MAX_DV : uint, TIter : IKeyIter>(
    Ptr<float, Access.Read> q_ptr,
    Ptr<float, Access.Read> k_head, uint64_t krs64,
    Ptr<float, Access.Read> v_head, uint64_t vrs64,
    Ptr<float> o_ptr,
    uint d_k, uint d_v, float scale, TIter iter)
{
    uint n_keys = iter.count();
    if (n_keys == 0u) {
        for (uint d = 0u; d < d_v; ++d) {
            o_ptr[d] = 0.0f;
        }
        return;
    }

    // Direct array indexing (never take &qv[0] as Ptr) so Slang DSE preserves stores.
    float qv[MAX_DK];
    float acc[MAX_DV];
    float scores[16];
    float probs[16];

    for (uint d = 0u; d < d_k; ++d) {
        qv[d] = q_ptr[d];
    }
    for (uint d = 0u; d < d_v; ++d) {
        acc[d] = 0.0f;
    }

    // Finite -ffast-math safe sentinel (never -INFINITY).
    float m = kNegInfSentinel;
    float l = 0.0f;

    // Block-16 tiled online softmax (B_k = 16): amortizes acc rescaling across 16 keys
    // and vectorizes fast_expf cleanly across 16 AVX-512 lanes.
    uint pos = 0u;
    while (pos < n_keys) {
        uint bk = min(16u, n_keys - pos);

        // 1. Compute up to 16 QK^T scores (4-way key-unrolled dot products reusing qv) & blk_max.
        float blk_max = kNegInfSentinel;
        uint kj = 0u;
        for (; kj + 4u <= bk; kj += 4u) {
            Ptr<float, Access.Read> k0 = k_head + uint64_t(iter.key_index(pos + kj + 0u)) * krs64;
            Ptr<float, Access.Read> k1 = k_head + uint64_t(iter.key_index(pos + kj + 1u)) * krs64;
            Ptr<float, Access.Read> k2 = k_head + uint64_t(iter.key_index(pos + kj + 2u)) * krs64;
            Ptr<float, Access.Read> k3 = k_head + uint64_t(iter.key_index(pos + kj + 3u)) * krs64;
            float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
            for (uint d = 0u; d < d_k; ++d) {
                float qd = qv[d];
                d0 += qd * k0[d];
                d1 += qd * k1[d];
                d2 += qd * k2[d];
                d3 += qd * k3[d];
            }
            float s0 = d0 * scale;
            float s1 = d1 * scale;
            float s2 = d2 * scale;
            float s3 = d3 * scale;
            scores[kj + 0u] = s0;
            scores[kj + 1u] = s1;
            scores[kj + 2u] = s2;
            scores[kj + 3u] = s3;
            float m01 = select(s0 > s1, s0, s1);
            float m23 = select(s2 > s3, s2, s3);
            float m0123 = select(m01 > m23, m01, m23);
            blk_max = select(m0123 > blk_max, m0123, blk_max);
        }
        for (; kj < bk; ++kj) {
            Ptr<float, Access.Read> k_row = k_head + uint64_t(iter.key_index(pos + kj)) * krs64;
            float dot = 0.0f;
            for (uint d = 0u; d < d_k; ++d) {
                dot += qv[d] * k_row[d];
            }
            float s = dot * scale;
            scores[kj] = s;
            blk_max = select(s > blk_max, s, blk_max);
        }

        // 2. Block-wide running maximum.
        float m_new = select(blk_max > m, blk_max, m);

        // 3. Rescale existing accumulator ONCE for the entire block of up to 16 keys.
        if (l > 0.0f && m_new > m) {
            float alpha = fast_expf(m - m_new);
            l *= alpha;
            for (uint d = 0u; d < d_v; ++d) {
                acc[d] *= alpha;
            }
        }

        // 4. Vectorizable fast_expf across 16 AVX-512 lanes + 4-way unrolled V accumulation.
        float blk_l = 0.0f;
        if (bk == 16u) {
            for (uint j = 0u; j < 16u; ++j) {
                float p = fast_expf(scores[j] - m_new);
                probs[j] = p;
                blk_l += p;
            }
        } else {
            for (uint j = 0u; j < bk; ++j) {
                float p = fast_expf(scores[j] - m_new);
                probs[j] = p;
                blk_l += p;
            }
        }
        l += blk_l;

        uint vj = 0u;
        for (; vj + 4u <= bk; vj += 4u) {
            float p0 = probs[vj + 0u];
            float p1 = probs[vj + 1u];
            float p2 = probs[vj + 2u];
            float p3 = probs[vj + 3u];
            Ptr<float, Access.Read> v0 = v_head + uint64_t(iter.key_index(pos + vj + 0u)) * vrs64;
            Ptr<float, Access.Read> v1 = v_head + uint64_t(iter.key_index(pos + vj + 1u)) * vrs64;
            Ptr<float, Access.Read> v2 = v_head + uint64_t(iter.key_index(pos + vj + 2u)) * vrs64;
            Ptr<float, Access.Read> v3 = v_head + uint64_t(iter.key_index(pos + vj + 3u)) * vrs64;
            for (uint d = 0u; d < d_v; ++d) {
                float a = acc[d];
                a += p0 * v0[d];
                a += p1 * v1[d];
                a += p2 * v2[d];
                a += p3 * v3[d];
                acc[d] = a;
            }
        }
        for (; vj < bk; ++vj) {
            float pj = probs[vj];
            Ptr<float, Access.Read> v_row = v_head + uint64_t(iter.key_index(pos + vj)) * vrs64;
            for (uint d = 0u; d < d_v; ++d) {
                acc[d] += pj * v_row[d];
            }
        }

        m = m_new;
        pos += bk;
    }

    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (uint d = 0u; d < d_v; ++d) {
        o_ptr[d] = acc[d] * inv_l;
    }
}

public void attention_rows_cpu_impl<let MAX_DK : uint, let MAX_DV : uint>(
    Ptr<float, Access.Read> q, uint q_stride,
    Ptr<float, Access.Read> k_cache, uint k_head_stride, uint k_row_stride,
    Ptr<float, Access.Read> v_cache, uint v_head_stride, uint v_row_stride,
    Ptr<float> out, uint o_stride,
    uint w0, uint w1, uint active_t, uint n_heads, uint n_kv_heads,
    uint d_k, uint d_v, uint kv_pos, float scale,
    uint sliding_window, uint causal)
{
    if (d_k == 0u || d_k > MAX_DK || d_v == 0u || d_v > MAX_DV ||
        n_heads == 0u || n_kv_heads == 0u) {
        return;
    }

    uint heads_per_kv = n_heads / n_kv_heads;
    uint64_t qs64 = uint64_t(q_stride);
    uint64_t os64 = uint64_t(o_stride);
    uint64_t khs64 = uint64_t(k_head_stride);
    uint64_t krs64 = uint64_t(k_row_stride);
    uint64_t vhs64 = uint64_t(v_head_stride);
    uint64_t vrs64 = uint64_t(v_row_stride);
    uint64_t dk64 = uint64_t(d_k);
    uint64_t dv64 = uint64_t(d_v);

    for (uint w = w0; w < w1; ++w) {
        uint t = w / n_heads;
        uint h = w - t * n_heads;
        uint kvh = h / heads_per_kv;

        DenseRangeIter iter =
            make_dense_range_iter(kv_pos, t, active_t, 0xFFFFFFFFu, sliding_window, causal);

        Ptr<float, Access.Read> q_ptr = q + uint64_t(t) * qs64 + uint64_t(h) * dk64;
        Ptr<float, Access.Read> k_head = k_cache + uint64_t(kvh) * khs64;
        Ptr<float, Access.Read> v_head = v_cache + uint64_t(kvh) * vhs64;
        Ptr<float> o_ptr = out + uint64_t(t) * os64 + uint64_t(h) * dv64;

        attention_online_core_cpu<MAX_DK, MAX_DV, DenseRangeIter>(
            q_ptr, k_head, krs64, v_head, vrs64, o_ptr, d_k, d_v, scale, iter);
    }
}
