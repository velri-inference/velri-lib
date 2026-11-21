// Vulkan Double-Buffered BK=8 Tiled F32 GEMM (128x128 WG, 256T, 8x8) + Split-K Reduction.
import numeric;
import epilogue;
import vk_common;

groupshared float4 s_f32_a[2 * 8 * 33];
groupshared float4 s_f32_b[2 * 8 * 33];

float4 load_f32x4_safe(Ptr<float, Access.Read> base, uint row, uint row_lim,
                       uint k, uint k_lim, uint stride, bool aligned4)
{
    if (row >= row_lim || k >= k_lim) return float4(0.0f);
    uint64_t off = uint64_t(row) * uint64_t(stride) + uint64_t(k);
    if (aligned4 && k + 4u <= k_lim) {
        return load_a16(Ptr<float4, Access.Read>(base + off));
    }
    return float4(
        (k + 0u < k_lim) ? base[off + 0u] : 0.0f,
        (k + 1u < k_lim) ? base[off + 1u] : 0.0f,
        (k + 2u < k_lim) ? base[off + 2u] : 0.0f,
        (k + 3u < k_lim) ? base[off + 3u] : 0.0f);
}

void store_f32_transposed_a(uint sb, uint comp, float4 v) {
    s_f32_a[sb +  0u][comp] = v.x;
    s_f32_a[sb + 33u][comp] = v.y;
    s_f32_a[sb + 66u][comp] = v.z;
    s_f32_a[sb + 99u][comp] = v.w;
}

void store_f32_transposed_b(uint sb, uint comp, float4 v) {
    s_f32_b[sb +  0u][comp] = v.x;
    s_f32_b[sb + 33u][comp] = v.y;
    s_f32_b[sb + 66u][comp] = v.z;
    s_f32_b[sb + 99u][comp] = v.w;
}

void f32_outer_8x8(inout float4 acc0[8], inout float4 acc1[8],
                   float4 a0, float4 a1, float4 b0, float4 b1)
{
    acc0[0] = fma(float4(a0.x), b0, acc0[0]); acc1[0] = fma(float4(a0.x), b1, acc1[0]);
    acc1[1] = fma(float4(a0.y), b1, acc1[1]); acc0[1] = fma(float4(a0.y), b0, acc0[1]);
    acc0[2] = fma(float4(a0.z), b0, acc0[2]); acc1[2] = fma(float4(a0.z), b1, acc1[2]);
    acc1[3] = fma(float4(a0.w), b1, acc1[3]); acc0[3] = fma(float4(a0.w), b0, acc0[3]);
    acc0[4] = fma(float4(a1.x), b0, acc0[4]); acc1[4] = fma(float4(a1.x), b1, acc1[4]);
    acc1[5] = fma(float4(a1.y), b1, acc1[5]); acc0[5] = fma(float4(a1.y), b0, acc0[5]);
    acc0[6] = fma(float4(a1.z), b0, acc0[6]); acc1[6] = fma(float4(a1.z), b1, acc1[6]);
    acc1[7] = fma(float4(a1.w), b1, acc1[7]); acc0[7] = fma(float4(a1.w), b0, acc0[7]);
}

void f32_gemm_mma8(inout float4 acc0[8], inout float4 acc1[8], uint sbase, uint a_col, uint b_col) {
    [ForceUnroll] for (uint kk = 0u; kk < 8u; kk += 2u) {
        uint off0 = sbase + kk * 33u;
        uint off1 = off0 + 33u;
        float4 a0_0 = s_f32_a[off0 + a_col];
        float4 a1_0 = s_f32_a[off0 + a_col + 4u];
        float4 b0_0 = s_f32_b[off0 + b_col];
        float4 b1_0 = s_f32_b[off0 + b_col + 8u];
        float4 a0_1 = s_f32_a[off1 + a_col];
        float4 a1_1 = s_f32_a[off1 + a_col + 4u];
        float4 b0_1 = s_f32_b[off1 + b_col];
        float4 b1_1 = s_f32_b[off1 + b_col + 8u];
        f32_outer_8x8(acc0, acc1, a0_0, a1_0, b0_0, b1_0);
        f32_outer_8x8(acc0, acc1, a0_1, a1_1, b0_1, b1_1);
    }
}

void f32_gemm_store_epilogue(
    uniform GemvArgs<ElemF32, ElemF32, ElemF32> a,
    Ptr<float> y_out, uint flags_eff,
    float4 acc0[8], float4 acc1[8],
    uint row0, uint col0, uint a_col, uint b_col)
{
    EpilogueParams ep = make_vk_ep(a.bias, a.x_scale, a.w_scale, flags_eff);
    bool y_vec = is_ptr_aligned16(y_out) && ((a.y_stride & 3u) == 0u);
    bool has_bias = ep.has_bias != 0u;
    bool bias_vec = has_bias && is_ptr_aligned16(a.bias);

    uint r_base0 = row0 + a_col * 4u;
    uint r_base1 = r_base0 + 16u;
    uint c0 = col0 + b_col * 4u;
    uint c1 = c0 + 32u;

    float4 bias0 = float4(0.0f);
    float4 bias1 = float4(0.0f);
    if (has_bias) {
        if (c0 + 4u <= a.m) {
            bias0 = bias_vec ? load_a16(Ptr<float4, Access.Read>(a.bias + c0))
                             : float4(a.bias[c0], a.bias[c0 + 1u], a.bias[c0 + 2u], a.bias[c0 + 3u]);
        }
        if (c1 + 4u <= a.m) {
            bias1 = bias_vec ? load_a16(Ptr<float4, Access.Read>(a.bias + c1))
                             : float4(a.bias[c1], a.bias[c1 + 1u], a.bias[c1 + 2u], a.bias[c1 + 3u]);
        }
    }

    [ForceUnroll] for (uint i = 0u; i < 8u; ++i) {
        uint r = (i < 4u) ? (r_base0 + i) : (r_base1 + (i - 4u));
        if (r >= a.rows) continue;
        Ptr<float> yr = y_out + uint64_t(r) * uint64_t(a.y_stride);
        if (y_vec && c0 + 4u <= a.m) {
            store_a16(Ptr<float4>(yr + c0), acc0[i] + bias0);
        } else {
            float4 v0 = acc0[i];
            if (c0 + 0u < a.m) yr[c0 + 0u] = EpBias::apply(v0.x, r, c0 + 0u, ep);
            if (c0 + 1u < a.m) yr[c0 + 1u] = EpBias::apply(v0.y, r, c0 + 1u, ep);
            if (c0 + 2u < a.m) yr[c0 + 2u] = EpBias::apply(v0.z, r, c0 + 2u, ep);
            if (c0 + 3u < a.m) yr[c0 + 3u] = EpBias::apply(v0.w, r, c0 + 3u, ep);
        }
        if (y_vec && c1 + 4u <= a.m) {
            store_a16(Ptr<float4>(yr + c1), acc1[i] + bias1);
        } else {
            float4 v1 = acc1[i];
            if (c1 + 0u < a.m) yr[c1 + 0u] = EpBias::apply(v1.x, r, c1 + 0u, ep);
            if (c1 + 1u < a.m) yr[c1 + 1u] = EpBias::apply(v1.y, r, c1 + 1u, ep);
            if (c1 + 2u < a.m) yr[c1 + 2u] = EpBias::apply(v1.z, r, c1 + 2u, ep);
            if (c1 + 3u < a.m) yr[c1 + 3u] = EpBias::apply(v1.w, r, c1 + 3u, ep);
        }
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_gemm_f32(uniform GemvArgs<ElemF32, ElemF32, ElemF32> a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint tid = gtid.x;
    uint grid_x = (a.m + 127u) >> 7u;
    uint grid_y = (a.rows + 127u) >> 7u;
    uint2 tile = swizzle_tile_2d(gid, grid_x, grid_y);
    uint row0 = tile.y * 128u;
    uint col0 = tile.x * 128u;

    uint warp_id = tid >> 5u;
    uint lane = tid & 31u;
    uint wr = warp_id >> 1u;
    uint wc = warp_id & 1u;
    uint lr_w = lane >> 3u;
    uint lc_w = lane & 7u;
    uint a_col = wr * 8u + lr_w;
    uint b_col = wc * 16u + lc_w;

    uint l_lk = (tid & 1u) << 2u;
    uint lr = tid >> 1u;
    uint s_off = l_lk * 33u + (lr >> 2u);
    uint s_comp = lr & 3u;

    float4 acc0[8];
    float4 acc1[8];
    [ForceUnroll] for (uint i = 0u; i < 8u; ++i) {
        acc0[i] = float4(0.0f);
        acc1[i] = float4(0.0f);
    }

    uint k_start = 0u;
    uint k_len = a.k;
    Ptr<float> y_out = a.y;
    uint flags_eff = a.flags;
    if (a.split_k > 1u) {
        uint total_kt = (a.k + 7u) >> 3u;
        uint kt_per = (total_kt + a.split_k - 1u) / a.split_k;
        uint k_chunk = kt_per << 3u;
        k_start = min(a.k, gid.z * k_chunk);
        uint k_end = min(a.k, k_start + k_chunk);
        k_len = k_end - k_start;
        y_out = a.y + uint64_t(gid.z) * (uint64_t(a.rows) * uint64_t(a.y_stride));
        flags_eff = 0u;
    }

    Ptr<float, Access.Read> x_base = a.x + k_start;
    Ptr<float, Access.Read> w_base = a.w + k_start;
    bool aligned4 = is_ptr_aligned16(x_base) && is_ptr_aligned16(w_base) &&
                    ((a.x_stride & 3u) == 0u) && ((a.w_stride & 3u) == 0u);
    bool fast_interior = aligned4 && ((k_len & 7u) == 0u) && (row0 + 128u <= a.rows) && (col0 + 128u <= a.m);
    uint num_k_tiles = (k_len + 7u) >> 3u;

    if (fast_interior && num_k_tiles > 0u) {
        Ptr<float4, Access.Read> xp = Ptr<float4, Access.Read>(x_base + uint64_t(row0 + lr) * uint64_t(a.x_stride) + uint64_t(l_lk));
        Ptr<float4, Access.Read> wp = Ptr<float4, Access.Read>(w_base + uint64_t(col0 + lr) * uint64_t(a.w_stride) + uint64_t(l_lk));

        store_f32_transposed_a(s_off, s_comp, load_a16(xp));
        store_f32_transposed_b(s_off, s_comp, load_a16(wp));
        GroupMemoryBarrierWithGroupSync();

        uint stage = 0u;
        for (uint kt = 0u; kt + 1u < num_k_tiles; ++kt) {
            uint k_u4 = (kt + 1u) << 1u;
            float4 pA = load_a16(xp + k_u4);
            float4 pB = load_a16(wp + k_u4);
            uint sbase = stage * (8u * 33u);
            f32_gemm_mma8(acc0, acc1, sbase, a_col, b_col);
            uint nbase = (stage ^ 1u) * (8u * 33u);
            store_f32_transposed_a(nbase + s_off, s_comp, pA);
            store_f32_transposed_b(nbase + s_off, s_comp, pB);
            GroupMemoryBarrierWithGroupSync();
            stage ^= 1u;
        }
        f32_gemm_mma8(acc0, acc1, stage * (8u * 33u), a_col, b_col);
    } else {
        for (uint kt = 0u; kt < num_k_tiles; ++kt) {
            uint k0 = kt << 3u;
            float4 va = load_f32x4_safe(x_base, row0 + lr, a.rows, k0 + l_lk, k_len, a.x_stride, aligned4);
            float4 vb = load_f32x4_safe(w_base, col0 + lr, a.m,    k0 + l_lk, k_len, a.w_stride, aligned4);
            store_f32_transposed_a(s_off, s_comp, va);
            store_f32_transposed_b(s_off, s_comp, vb);
            GroupMemoryBarrierWithGroupSync();
            f32_gemm_mma8(acc0, acc1, 0u, a_col, b_col);
            GroupMemoryBarrierWithGroupSync();
        }
    }

    f32_gemm_store_epilogue(a, y_out, flags_eff, acc0, acc1, row0, col0, a_col, b_col);
}

struct SplitKReduceArgs {
    Ptr<uint4, Access.Read> partials;
    Ptr<float, Access.Read> bias;
    Ptr<float, Access.Read> x_scale;
    Ptr<float, Access.Read> w_scale;
    Ptr<float> y;
    uint rows;
    uint m;
    uint part_stride_u4;
    uint y_stride;
    uint split_k;
    uint flags;
    uint is_int32;
    uint pad;
};

[shader("compute")] [numthreads(256, 1, 1)]
void vk_splitk_reduce(uniform SplitKReduceArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint idx = gid.x * 256u + gtid.x;
    uint64_t slice_u4 = uint64_t(a.rows) * uint64_t(a.part_stride_u4);
    if (uint64_t(idx) >= slice_u4) return;

    uint gr = idx / a.part_stride_u4;
    uint c_u4 = idx - gr * a.part_stride_u4;
    uint gc = c_u4 << 2u;

    float4 fv = float4(0.0f);
    for (uint s = 0u; s < a.split_k; ++s) {
        fv += asfloat(load_a16(a.partials + uint64_t(s) * slice_u4 + uint64_t(idx)));
    }

    EpilogueParams ep = make_vk_ep(a.bias, a.x_scale, a.w_scale, a.flags);
    bool y_vec = is_ptr_aligned16(a.y) && ((a.y_stride & 3u) == 0u);
    Ptr<float> yr = a.y + uint64_t(gr) * uint64_t(a.y_stride);
    if (y_vec && gc + 4u <= a.m) {
        float4 outv = fv;
        if (ep.has_bias != 0u) {
            float4 bv = is_ptr_aligned16(ep.bias)
                        ? load_a16(Ptr<float4, Access.Read>(ep.bias + gc))
                        : float4(ep.bias[gc], ep.bias[gc + 1u], ep.bias[gc + 2u], ep.bias[gc + 3u]);
            outv += bv;
        }
        store_a16(Ptr<float4>(yr + gc), outv);
    } else {
        if (gc + 0u < a.m) yr[gc + 0u] = EpBias::apply(fv.x, gr, gc + 0u, ep);
        if (gc + 1u < a.m) yr[gc + 1u] = EpBias::apply(fv.y, gr, gc + 1u, ep);
        if (gc + 2u < a.m) yr[gc + 2u] = EpBias::apply(fv.z, gr, gc + 2u, ep);
        if (gc + 3u < a.m) yr[gc + 3u] = EpBias::apply(fv.w, gr, gc + 3u, ep);
    }
}
