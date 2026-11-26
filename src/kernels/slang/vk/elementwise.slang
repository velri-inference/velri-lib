// Vulkan Elementwise kernels: Activations, GatedMLP, Axpby, Mul, Softmax, FWHT, QuantizeRow.
import numeric;
import activations;
import reductions;
import vk_common;

struct ActivationArgs {
    Ptr<float, Access.Read> x;
    Ptr<float> y;
    uint count;
    uint pad;
};

void activation_gpu_impl<A : IActivation>(ActivationArgs a, uint tid) {
    uint base = tid << 2u;
    if (base + 4u <= a.count && is_ptr_aligned16(a.x) && is_ptr_aligned16(a.y)) {
        float4 v = load_a16(Ptr<float4, Access.Read>(a.x) + tid);
        store_a16(Ptr<float4>(a.y) + tid, apply_act4<A>(v));
    } else {
        uint end = min(base + 4u, a.count);
        for (uint i = base; i < end; ++i) a.y[i] = A::apply(a.x[i]);
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_activation_silu_f32(uniform ActivationArgs a, uint3 tid : SV_DispatchThreadID) {
    activation_gpu_impl<ActSiLU>(a, tid.x);
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_activation_gelu_f32(uniform ActivationArgs a, uint3 tid : SV_DispatchThreadID) {
    activation_gpu_impl<ActGeluTanh>(a, tid.x);
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_activation_relu2_f32(uniform ActivationArgs a, uint3 tid : SV_DispatchThreadID) {
    activation_gpu_impl<ActReLU2>(a, tid.x);
}

struct GatedMlpArgs {
    Ptr<float, Access.Read> gate;
    Ptr<float, Access.Read> up;
    Ptr<float> y;
    uint count;
    float clamp_limit;
};

void gated_mlp_gpu_impl<A : IActivation>(GatedMlpArgs a, uint tid) {
    uint base = tid << 2u;
    if (base + 4u <= a.count && is_ptr_aligned16(a.gate) &&
        is_ptr_aligned16(a.up) && is_ptr_aligned16(a.y)) {
        float4 g = apply_symmetric_clamp4(load_a16(Ptr<float4, Access.Read>(a.gate) + tid), a.clamp_limit);
        float4 u = apply_symmetric_clamp4(load_a16(Ptr<float4, Access.Read>(a.up) + tid), a.clamp_limit);
        store_a16(Ptr<float4>(a.y) + tid, apply_act4<A>(g) * u);
    } else {
        uint end = min(base + 4u, a.count);
        for (uint i = base; i < end; ++i) {
            float g = apply_symmetric_clamp(a.gate[i], a.clamp_limit);
            float u = apply_symmetric_clamp(a.up[i], a.clamp_limit);
            a.y[i] = A::apply(g) * u;
        }
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_gated_silu_f32(uniform GatedMlpArgs a, uint3 tid : SV_DispatchThreadID) {
    gated_mlp_gpu_impl<ActSiLU>(a, tid.x);
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_gated_gelu_f32(uniform GatedMlpArgs a, uint3 tid : SV_DispatchThreadID) {
    gated_mlp_gpu_impl<ActGeluTanh>(a, tid.x);
}

struct VecAxpbyArgs {
    Ptr<float, Access.Read> a;
    Ptr<float, Access.Read> b;
    Ptr<float> y;
    uint count;
    float alpha;
    float beta;
    uint has_b;
    uint pad;
};

[shader("compute")] [numthreads(256, 1, 1)]
void vk_vec_axpby_f32(uniform VecAxpbyArgs p, uint3 tid : SV_DispatchThreadID) {
    uint base = tid.x << 2u;
    bool aligned = is_ptr_aligned16(p.a) && is_ptr_aligned16(p.y) &&
                   ((p.has_b == 0u) || is_ptr_aligned16(p.b));
    if (base + 4u <= p.count && aligned) {
        float4 outv = float4(p.alpha) * load_a16(Ptr<float4, Access.Read>(p.a) + tid.x);
        if (p.has_b != 0u) outv = fma(float4(p.beta), load_a16(Ptr<float4, Access.Read>(p.b) + tid.x), outv);
        store_a16(Ptr<float4>(p.y) + tid.x, outv);
    } else {
        uint end = min(base + 4u, p.count);
        for (uint i = base; i < end; ++i) {
            float v = p.alpha * p.a[i];
            if (p.has_b != 0u) v = fma(p.beta, p.b[i], v);
            p.y[i] = v;
        }
    }
}

struct VecMulArgs {
    Ptr<float, Access.Read> a;
    Ptr<float, Access.Read> b;
    Ptr<float> y;
    uint count;
    uint pad;
};

[shader("compute")] [numthreads(256, 1, 1)]
void vk_vec_mul_f32(uniform VecMulArgs p, uint3 tid : SV_DispatchThreadID) {
    uint base = tid.x << 2u;
    if (base + 4u <= p.count && is_ptr_aligned16(p.a) &&
        is_ptr_aligned16(p.b) && is_ptr_aligned16(p.y)) {
        float4 av = load_a16(Ptr<float4, Access.Read>(p.a) + tid.x);
        float4 bv = load_a16(Ptr<float4, Access.Read>(p.b) + tid.x);
        store_a16(Ptr<float4>(p.y) + tid.x, av * bv);
    } else {
        uint end = min(base + 4u, p.count);
        for (uint i = base; i < end; ++i) p.y[i] = p.a[i] * p.b[i];
    }
}

struct SoftmaxArgs {
    Ptr<float, Access.Read> x;
    Ptr<float> y;
    uint rows;
    uint cols;
    uint x_stride;
    uint y_stride;
};

[shader("compute")] [numthreads(256, 1, 1)]
void vk_softmax_f32(uniform SoftmaxArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint row = gid.x;
    uint tid = gtid.x;
    if (row >= a.rows) return;
    Ptr<float, Access.Read> xr = a.x + uint64_t(row) * uint64_t(a.x_stride);
    Ptr<float> yr = a.y + uint64_t(row) * uint64_t(a.y_stride);

    bool aligned = ((a.cols & 3u) == 0u) && is_ptr_aligned16(xr) && is_ptr_aligned16(yr);
    if (aligned && a.cols <= kRowRegCacheMaxDim) {
        uint n4 = a.cols >> 2u;
        Ptr<float4, Access.Read> xr4 = Ptr<float4, Access.Read>(xr);
        Ptr<float4> yr4 = Ptr<float4>(yr);

        float4 reg[4] = { float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f) };
        float local_max = kNegInfSentinel;
        [ForceUnroll] for (uint s = 0u; s < 4u; ++s) {
            uint idx = s * kRowWgSize + tid;
            if (idx < n4) {
                float4 v = load_a16(xr4 + idx);
                reg[s] = v;
                local_max = max(local_max, max(max(v.x, v.y), max(v.z, v.w)));
            }
        }
        float row_max = wg_reduce_max(local_max, tid, kRowWgSize);
        if (!(row_max > kNegInfCutoff)) {
            [ForceUnroll] for (uint s = 0u; s < 4u; ++s) {
                uint idx = s * kRowWgSize + tid;
                if (idx < n4) store_a16(yr4 + idx, float4(0.0f));
            }
            return;
        }

        float local_sum = 0.0f;
        [ForceUnroll] for (uint s = 0u; s < 4u; ++s) {
            uint idx = s * kRowWgSize + tid;
            if (idx < n4) {
                float4 v = reg[s];
                float4 e = float4(
                    (v.x > kNegInfCutoff) ? fast_expf(v.x - row_max) : 0.0f,
                    (v.y > kNegInfCutoff) ? fast_expf(v.y - row_max) : 0.0f,
                    (v.z > kNegInfCutoff) ? fast_expf(v.z - row_max) : 0.0f,
                    (v.w > kNegInfCutoff) ? fast_expf(v.w - row_max) : 0.0f);
                reg[s] = e;
                local_sum += e.x + e.y + e.z + e.w;
            }
        }
        float total_sum = wg_reduce_sum(local_sum, tid, kRowWgSize);
        float inv_sum = (total_sum > 0.0f) ? (1.0f / total_sum) : 0.0f;

        [ForceUnroll] for (uint s = 0u; s < 4u; ++s) {
            uint idx = s * kRowWgSize + tid;
            if (idx < n4) store_a16(yr4 + idx, reg[s] * inv_sum);
        }
        return;
    }

    float local_max = kNegInfSentinel;
    for (uint c = tid; c < a.cols; c += kRowWgSize) {
        local_max = max(local_max, xr[c]);
    }
    float row_max = wg_reduce_max(local_max, tid, kRowWgSize);
    if (!(row_max > kNegInfCutoff)) {
        for (uint c = tid; c < a.cols; c += kRowWgSize) yr[c] = 0.0f;
        return;
    }

    float local_sum = 0.0f;
    for (uint c = tid; c < a.cols; c += kRowWgSize) {
        float v = xr[c];
        float e = (v > kNegInfCutoff) ? fast_expf(v - row_max) : 0.0f;
        yr[c] = e;
        local_sum += e;
    }
    float total_sum = wg_reduce_sum(local_sum, tid, kRowWgSize);
    float inv_sum = (total_sum > 0.0f) ? (1.0f / total_sum) : 0.0f;
    for (uint c = tid; c < a.cols; c += kRowWgSize) {
        yr[c] *= inv_sum;
    }
}

// ============================================================================
// Normalized FWHT (subgroup-size agnostic: works on SIMD8/16, Wave32, Wave64)
// ============================================================================
struct FwhtArgs {
    Ptr<float, Access.Read> x;
    Ptr<uint8_t, Access.Read> suh;
    Ptr<float> y;
    uint rows;
    uint cols;
    uint block_size;
    uint x_stride;
    uint y_stride;
    uint suh_mode; // 0=none, 1=f32, 2=f16
    float inv_sqrt_b;
    uint pad;
};

groupshared float4 s_fwht[kRowWgSize];

float4 fwht_stage(float4 v, uint tid, uint lane, uint lane_cnt, uint bit) {
    if (bit < lane_cnt) {
        float4 p = WaveReadLaneAt(v, lane ^ bit);
        return ((lane & bit) == 0u) ? (v + p) : (p - v);
    }
    s_fwht[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    float4 p = s_fwht[tid ^ bit];
    GroupMemoryBarrierWithGroupSync();
    return ((tid & bit) == 0u) ? (v + p) : (p - v);
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_fwht_f32(uniform FwhtArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint row = gid.y;
    uint tid = gtid.x;
    if (row >= a.rows) return;
    uint elem0 = gid.x * (kRowWgSize * 4u) + tid * 4u;
    Ptr<float, Access.Read> xr = a.x + uint64_t(row) * uint64_t(a.x_stride);
    Ptr<float> yr = a.y + uint64_t(row) * uint64_t(a.y_stride);

    float4 v = float4(0.0f);
    bool active = (elem0 < a.cols);
    if (active) {
        v = load_f32x4_maybe_a16(xr + elem0);
        if (a.suh_mode == 1u) {
            Ptr<float, Access.Read> sf = Ptr<float, Access.Read>(a.suh);
            v *= load_f32x4_maybe_a16(sf + elem0);
        } else if (a.suh_mode == 2u) {
            Ptr<half, Access.Read> sh = Ptr<half, Access.Read>(a.suh);
            v *= float4(float(sh[elem0]), float(sh[elem0 + 1u]), float(sh[elem0 + 2u]), float(sh[elem0 + 3u]));
        }
        v = hadamard4(v);
    }

    uint lane = WaveGetLaneIndex();
    uint lane_cnt = WaveGetLaneCount();
    v = fwht_stage(v, tid, lane, lane_cnt, 1u);
    v = fwht_stage(v, tid, lane, lane_cnt, 2u);
    if (a.block_size >= 32u)  v = fwht_stage(v, tid, lane, lane_cnt, 4u);
    if (a.block_size >= 64u)  v = fwht_stage(v, tid, lane, lane_cnt, 8u);
    if (a.block_size >= 128u) v = fwht_stage(v, tid, lane, lane_cnt, 16u);
    if (a.block_size >= 256u) v = fwht_stage(v, tid, lane, lane_cnt, 32u);

    if (active) {
        store_f32x4_maybe_a16(yr + elem0, v * a.inv_sqrt_b);
    }
}

// ============================================================================
// Parallel Dynamic Symmetric Row Quantization (F32 -> I8)
// ============================================================================
struct QuantizeRowArgs {
    Ptr<float, Access.Read> x;
    Ptr<int8_t> x_i8;
    Ptr<float> scale_out;
    uint rows;
    uint k;
    uint x_stride;
    uint xi8_stride;
};

uint pack_i8x4_scaled(float4 scaled) {
    float4 s = clamp(round(scaled), float4(-kSymI8MaxQ), float4(kSymI8MaxQ));
    uint b0 = uint(int32_t(s.x)) & 0xFFu;
    uint b1 = uint(int32_t(s.y)) & 0xFFu;
    uint b2 = uint(int32_t(s.z)) & 0xFFu;
    uint b3 = uint(int32_t(s.w)) & 0xFFu;
    return b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_quantize_row_f32_i8(uniform QuantizeRowArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint row = gid.x;
    uint tid = gtid.x;
    if (row >= a.rows) return;
    Ptr<float, Access.Read> xr = a.x + uint64_t(row) * uint64_t(a.x_stride);
    Ptr<int8_t> qr = a.x_i8 + uint64_t(row) * uint64_t(a.xi8_stride);

    bool aligned = ((a.k & 3u) == 0u) && is_ptr_aligned16(xr) && is_ptr_aligned4(qr);
    // Cache one row through the max reduction so the output needs no second input pass.
    static const uint chunks = 8u;
    if (aligned && a.k <= chunks * 4u * kRowWgSize) {
        uint n4 = a.k >> 2u;
        Ptr<float4, Access.Read> xr4 = Ptr<float4, Access.Read>(xr);
        Ptr<uint> qr4 = Ptr<uint>(qr);

        float4 reg[chunks];
        float local_max = 0.0f;
        [ForceUnroll] for (uint s = 0u; s < chunks; ++s) {
            uint idx = s * kRowWgSize + tid;
            if (idx < n4) {
                float4 v = load_a16(xr4 + idx);
                reg[s] = v;
                float4 av = abs(v);
                local_max = max(local_max, max(max(av.x, av.y), max(av.z, av.w)));
            }
        }
        float amax = wg_reduce_max(local_max, tid, kRowWgSize);
        float scale, inv_scale;
        compute_sym_i8_scale(amax, scale, inv_scale);
        if (tid == 0u) a.scale_out[row] = scale;
        if (inv_scale == 0.0f) {
            [ForceUnroll] for (uint s = 0u; s < chunks; ++s) {
                uint idx = s * kRowWgSize + tid;
                if (idx < n4) qr4[idx] = 0u;
            }
        } else {
            [ForceUnroll] for (uint s = 0u; s < chunks; ++s) {
                uint idx = s * kRowWgSize + tid;
                if (idx < n4) qr4[idx] = pack_i8x4_scaled(reg[s] * inv_scale);
            }
        }
        return;
    }

    float local_max = 0.0f;
    for (uint c = tid; c < a.k; c += kRowWgSize) {
        local_max = max(local_max, abs(xr[c]));
    }
    float amax = wg_reduce_max(local_max, tid, kRowWgSize);
    float scale, inv_scale;
    compute_sym_i8_scale(amax, scale, inv_scale);
    if (tid == 0u) a.scale_out[row] = scale;
    if (inv_scale == 0.0f) {
        for (uint c = tid; c < a.k; c += kRowWgSize) qr[c] = int8_t(0);
    } else {
        for (uint c = tid; c < a.k; c += kRowWgSize) qr[c] = quantize_sym_i8(xr[c], inv_scale);
    }
}

struct QuantGemvArgs {
    Ptr<float4, Access.Read> x;
    Ptr<uint> x_i8;
    Ptr<float> scale_out;
    Ptr<uint4, Access.Read> weights;
    Ptr<float, Access.Read> weight_scale;
    Ptr<float> y;
    Ptr<float> residual;
    uint channels;
    uint k4;
    uint stride16;
};

groupshared uint quant_words[2u * kRowWgSize];

[shader("compute")] [numthreads(256, 1, 1)]
void vk_quant_gemv_w8a8(uniform QuantGemvArgs a, uint3 group : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint tid = gtid.x;
    float4 reg[2] = { float4(0.0f), float4(0.0f) };
    float peak = 0.0f;
    [ForceUnroll] for (uint j = 0u; j < 2u; ++j) {
        uint i = tid + j * kRowWgSize;
        if (i < a.k4) {
            reg[j] = load_a16(a.x + i);
            float4 v = abs(reg[j]);
            peak = max(peak, max(max(v.x, v.y), max(v.z, v.w)));
        }
    }
    float scale, inv_scale;
    compute_sym_i8_scale(wg_reduce_max(peak, tid, kRowWgSize), scale, inv_scale);
    if (group.x == 0u && tid == 0u) a.scale_out[0] = scale;
    [ForceUnroll] for (uint j = 0u; j < 2u; ++j) {
        uint i = tid + j * kRowWgSize;
        if (i < a.k4) {
            uint packed = inv_scale == 0.0f ? 0u : pack_i8x4_scaled(reg[j] * inv_scale);
            quant_words[i] = packed;
            if (group.x == 0u) a.x_i8[i] = packed;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    uint lanes = WaveGetLaneCount();
    uint col = group.x * (kRowWgSize / lanes) + WaveGetWaveIndex();
    if (col >= a.channels) return;
    Ptr<uint4, Access.Read> weights = a.weights + uint64_t(col) * a.stride16;
    int32_t sum = 0;
    for (uint i = WaveGetLaneIndex(); i < a.k4 / 4u; i += lanes) {
        uint4 w = load_a16(weights + i);
        sum = dot4add_i8packed(quant_words[4u*i], w.x, sum);
        sum = dot4add_i8packed(quant_words[4u*i + 1u], w.y, sum);
        sum = dot4add_i8packed(quant_words[4u*i + 2u], w.z, sum);
        sum = dot4add_i8packed(quant_words[4u*i + 3u], w.w, sum);
    }
    sum = WaveActiveSum(sum);
    if (WaveGetLaneIndex() == 0u) {
        precise float y = float(sum) * (a.weight_scale[col] * scale);
        a.y[col] = y;
        if (a.residual != nullptr) a.residual[col] = a.residual[col] + y;
    }
}

struct RmsQuantArgs {
    Ptr<float4, Access.Read> x;
    Ptr<float4, Access.Read> weight;
    Ptr<float4> normalized;
    Ptr<uint> quantized;
    Ptr<float> scale;
    uint dim;
    float eps;
};

[shader("compute")] [numthreads(256, 1, 1)]
void vk_rms_quant(uniform RmsQuantArgs a, uint3 gtid : SV_GroupThreadID) {
    uint tid = gtid.x, count = a.dim / 4u;
    float4 values[2], weights[2];
    float energy = 0.0f;
    [ForceUnroll] for (uint j = 0u; j < 2u; ++j) {
        uint i = tid + j * kRowWgSize;
        if (i < count) {
            values[j] = load_a16(a.x + i);
            // Standalone RMS adds +0 to gamma when unit_offset is disabled, mapping -0 to +0.
            float4 w = load_a16(a.weight + i);
            weights[j] = select(w == float4(0.0f), float4(0.0f), w);
            energy += dot(values[j], values[j]);
        }
    }
    float inv_rms = rsqrt(wg_reduce_sum(energy, tid, kRowWgSize) / float(a.dim) + a.eps);
    float peak = 0.0f;
    [ForceUnroll] for (uint j = 0u; j < 2u; ++j) {
        uint i = tid + j * kRowWgSize;
        if (i < count) {
            values[j] *= inv_rms * weights[j];
            store_a16(a.normalized + i, values[j]);
            float4 v = abs(values[j]);
            peak = max(peak, max(max(v.x, v.y), max(v.z, v.w)));
        }
    }
    float scale, inv_scale;
    compute_sym_i8_scale(wg_reduce_max(peak, tid, kRowWgSize), scale, inv_scale);
    if (tid == 0u) a.scale[0] = scale;
    [ForceUnroll] for (uint j = 0u; j < 2u; ++j) {
        uint i = tid + j * kRowWgSize;
        if (i < count) a.quantized[i] = inv_scale == 0.0f ? 0u : pack_i8x4_scaled(values[j] * inv_scale);
    }
}
