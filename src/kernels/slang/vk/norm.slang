// Vulkan Normalization kernels: RMSNorm (F32/BF16/F16), FusedAddRMSNorm (F32), LayerNorm (F32).
import numeric;
import reductions;
import vk_common;

float rmsnorm_scale_w<TW : IElement>(
    Ptr<TW.Storage, Access.Read> weight, uint has_weight, uint unit_offset, uint d)
{
    if (has_weight == 0u) return 1.0f;
    float w = TW::toF32(weight[d]);
    return (unit_offset != 0u) ? (w + 1.0f) : w;
}

struct RmsNormArgs<TIO : IElement, TW : IElement> {
    Ptr<TIO.Storage, Access.Read> x;
    Ptr<TW.Storage, Access.Read> w;
    Ptr<TIO.Storage> y;
    uint rows;
    uint dim;
    uint x_stride;
    uint y_stride;
    float eps;
    uint has_weight;
    uint unit_offset;
};

struct FusedAddRmsNormArgs {
    Ptr<float> residual_inout;
    Ptr<float, Access.Read> x;
    Ptr<float, Access.Read> w;
    Ptr<float> y;
    uint rows;
    uint dim;
    uint r_stride;
    uint x_stride;
    uint y_stride;
    float eps;
    uint has_weight;
    uint unit_offset;
};

void rmsnorm_gpu_impl<TIO : IElement, TW : IElement,
                      VIO : IVecIO128<TIO>, VW : IVecIO128<TW>,
                      let FUSED_ADD : bool>(
    Ptr<TIO.Storage> rr,
    Ptr<TIO.Storage, Access.Read> xr,
    Ptr<TW.Storage, Access.Read> w,
    Ptr<TIO.Storage> yr,
    uint dim, float eps, uint has_weight, uint unit_offset,
    uint tid)
{
    uint vec_elems = VIO.kElems;
    bool aligned_vec = ((dim & (vec_elems - 1u)) == 0u) &&
                       is_ptr_aligned16(xr) && is_ptr_aligned16(yr) &&
                       ((has_weight == 0u) || is_ptr_aligned16(w)) &&
                       (!FUSED_ADD || is_ptr_aligned16(rr));

    if (aligned_vec && dim <= kRowRegCacheMaxDim) {
        static const uint kSteps = 16u / VIO.kElems;       // 4 for F32, 2 for F16/BF16
        static const uint kWide = VIO.kElems == 8u ? 1u : 0u;
        uint n_vec = dim / vec_elems;
        Ptr<uint4, Access.Read> xr4 = Ptr<uint4, Access.Read>(xr);
        Ptr<uint4, Access.Read> rr4_r = Ptr<uint4, Access.Read>(rr);
        Ptr<uint4> rr4 = Ptr<uint4>(rr);
        Ptr<uint4, Access.Read> w4 = Ptr<uint4, Access.Read>(w);
        Ptr<uint4> yr4 = Ptr<uint4>(yr);

        float4 reg[4] = { float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f) };
        float partial = 0.0f;
        [ForceUnroll] for (uint s = 0u; s < kSteps; ++s) {
            uint idx = s * kRowWgSize + tid;
            if (idx < n_vec) {
                float4 lo, hi;
                VIO::unpack(load_a16(xr4 + idx), lo, hi);
                if (FUSED_ADD) {
                    float4 rlo, rhi;
                    VIO::unpack(load_a16(rr4_r + idx), rlo, rhi);
                    lo += rlo;
                    if (kWide != 0u) hi += rhi;
                    store_a16(rr4 + idx, VIO::pack(lo, hi));
                }
                if (kWide != 0u) {
                    reg[s * 2u + 0u] = lo;
                    reg[s * 2u + 1u] = hi;
                    partial += dot(lo, lo) + dot(hi, hi);
                } else {
                    reg[s] = lo;
                    partial += dot(lo, lo);
                }
            }
        }
        float inv_rms = rsqrt(wg_reduce_sum(partial, tid, kRowWgSize) / float(dim) + eps);
        float uoff = (unit_offset != 0u) ? 1.0f : 0.0f;

        [ForceUnroll] for (uint s = 0u; s < kSteps; ++s) {
            uint idx = s * kRowWgSize + tid;
            if (idx < n_vec) {
                float4 wlo = float4(1.0f), whi = float4(1.0f);
                if (has_weight != 0u) {
                    VW::unpack(load_a16(w4 + idx), wlo, whi);
                    wlo += float4(uoff);
                    if (kWide != 0u) whi += float4(uoff);
                }
                if (kWide != 0u) {
                    store_a16(yr4 + idx, VIO::pack(
                        reg[s * 2u + 0u] * (inv_rms * wlo),
                        reg[s * 2u + 1u] * (inv_rms * whi)));
                } else {
                    store_a16(yr4 + idx, VIO::pack(reg[s] * (inv_rms * wlo), float4(0.0f)));
                }
            }
        }
        return;
    }

    float partial = 0.0f;
    for (uint d = tid; d < dim; d += kRowWgSize) {
        float v = TIO::toF32(xr[d]);
        if (FUSED_ADD) {
            v += TIO::toF32(rr[d]);
            rr[d] = TIO::fromF32(v);
        }
        partial += v * v;
    }
    float inv_rms = rsqrt(wg_reduce_sum(partial, tid, kRowWgSize) / float(dim) + eps);
    for (uint d = tid; d < dim; d += kRowWgSize) {
        float val = FUSED_ADD ? TIO::toF32(rr[d]) : TIO::toF32(xr[d]);
        float wv = rmsnorm_scale_w<TW>(w, has_weight, unit_offset, d);
        yr[d] = TIO::fromF32(val * inv_rms * wv);
    }
}

void rmsnorm_dispatch<TIO : IElement, TW : IElement, VIO : IVecIO128<TIO>, VW : IVecIO128<TW>>(
    RmsNormArgs<TIO, TW> a, uint row, uint tid)
{
    if (row >= a.rows) return;
    Ptr<TIO.Storage, Access.Read> xr = a.x + uint64_t(row) * uint64_t(a.x_stride);
    Ptr<TIO.Storage> yr = a.y + uint64_t(row) * uint64_t(a.y_stride);
    rmsnorm_gpu_impl<TIO, TW, VIO, VW, false>(
        yr, xr, a.w, yr, a.dim, a.eps, a.has_weight, a.unit_offset, tid);
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_rmsnorm_f32(uniform RmsNormArgs<ElemF32, ElemF32> a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    rmsnorm_dispatch<ElemF32, ElemF32, VecIOF32, VecIOF32>(a, gid.x, gtid.x);
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_rmsnorm_bf16(uniform RmsNormArgs<ElemBF16, ElemBF16> a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    rmsnorm_dispatch<ElemBF16, ElemBF16, VecIOBF16, VecIOBF16>(a, gid.x, gtid.x);
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_rmsnorm_f16(uniform RmsNormArgs<ElemF16, ElemF16> a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    rmsnorm_dispatch<ElemF16, ElemF16, VecIOF16, VecIOF16>(a, gid.x, gtid.x);
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_fused_add_rmsnorm_f32(uniform FusedAddRmsNormArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint row = gid.x;
    if (row >= a.rows) return;
    Ptr<float> rr = a.residual_inout + uint64_t(row) * uint64_t(a.r_stride);
    Ptr<float, Access.Read> xr = a.x + uint64_t(row) * uint64_t(a.x_stride);
    Ptr<float> yr = a.y + uint64_t(row) * uint64_t(a.y_stride);
    rmsnorm_gpu_impl<ElemF32, ElemF32, VecIOF32, VecIOF32, true>(
        rr, xr, a.w, yr, a.dim, a.eps, a.has_weight, a.unit_offset, gtid.x);
}

struct LayerNormArgs {
    Ptr<float, Access.Read> x;
    Ptr<float, Access.Read> w;
    Ptr<float, Access.Read> b;
    Ptr<float> y;
    uint rows;
    uint dim;
    uint x_stride;
    uint y_stride;
    float eps;
    uint has_weight;
    uint has_bias;
    uint pad;
};

[shader("compute")] [numthreads(256, 1, 1)]
void vk_layernorm_f32(uniform LayerNormArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint row = gid.x;
    uint tid = gtid.x;
    if (row >= a.rows) return;
    Ptr<float, Access.Read> xr = a.x + uint64_t(row) * uint64_t(a.x_stride);
    Ptr<float> yr = a.y + uint64_t(row) * uint64_t(a.y_stride);
    float inv_n = 1.0f / float(a.dim);

    bool aligned = ((a.dim & 3u) == 0u) && is_ptr_aligned16(xr) && is_ptr_aligned16(yr) &&
                   ((a.has_weight == 0u) || is_ptr_aligned16(a.w)) &&
                   ((a.has_bias == 0u) || is_ptr_aligned16(a.b));
    if (aligned && a.dim <= kRowRegCacheMaxDim) {
        uint n4 = a.dim >> 2u;
        Ptr<float4, Access.Read> xr4 = Ptr<float4, Access.Read>(xr);
        Ptr<float4, Access.Read> w4 = Ptr<float4, Access.Read>(a.w);
        Ptr<float4, Access.Read> b4 = Ptr<float4, Access.Read>(a.b);
        Ptr<float4> yr4 = Ptr<float4>(yr);

        float4 reg[4] = { float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f) };
        float psum = 0.0f;
        [ForceUnroll] for (uint s = 0u; s < 4u; ++s) {
            uint idx = s * kRowWgSize + tid;
            if (idx < n4) {
                float4 v = load_a16(xr4 + idx);
                reg[s] = v;
                psum += v.x + v.y + v.z + v.w;
            }
        }
        float mean = wg_reduce_sum(psum, tid, kRowWgSize) * inv_n;

        float pvar = 0.0f;
        [ForceUnroll] for (uint s = 0u; s < 4u; ++s) {
            uint idx = s * kRowWgSize + tid;
            if (idx < n4) {
                float4 d = reg[s] - float4(mean);
                reg[s] = d;
                pvar += dot(d, d);
            }
        }
        float inv_std = rsqrt(wg_reduce_sum(pvar, tid, kRowWgSize) * inv_n + a.eps);

        [ForceUnroll] for (uint s = 0u; s < 4u; ++s) {
            uint idx = s * kRowWgSize + tid;
            if (idx < n4) {
                float4 outv = reg[s] * inv_std;
                if (a.has_weight != 0u) outv *= load_a16(w4 + idx);
                if (a.has_bias != 0u) outv += load_a16(b4 + idx);
                store_a16(yr4 + idx, outv);
            }
        }
        return;
    }

    float psum = 0.0f;
    for (uint d = tid; d < a.dim; d += kRowWgSize) psum += xr[d];
    float mean = wg_reduce_sum(psum, tid, kRowWgSize) * inv_n;
    float pvar = 0.0f;
    for (uint d = tid; d < a.dim; d += kRowWgSize) {
        float diff = xr[d] - mean;
        pvar += diff * diff;
    }
    float inv_std = rsqrt(wg_reduce_sum(pvar, tid, kRowWgSize) * inv_n + a.eps);
    for (uint d = tid; d < a.dim; d += kRowWgSize) {
        float v = (xr[d] - mean) * inv_std;
        if (a.has_weight != 0u) v *= a.w[d];
        if (a.has_bias != 0u) v += a.b[d];
        yr[d] = v;
    }
}
