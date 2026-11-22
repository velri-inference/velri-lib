// Shared Vulkan BDA 128-bit load/store helpers, IVecIO128 unpackers,
// tile swizzling, epilogue stores, and unified LinearArgs push constants.
module vk_common;

import numeric;
import activations;
import epilogue;

public static const uint kRowWgSize = 256u;
public static const uint kRowRegCacheMaxDim = 4096u;
public static const uint kGemvWgSize = 128u;

// Select a valid physical address before the load. Software subgroup implementations
// may issue uniform loads for inactive lanes, including eager kernels without step data.
public uint step_word_or(Ptr<uint, Access.Read> step, uint index, uint fallback,
                         Ptr<float, Access.Read> read_only) {
    Ptr<uint, Access.Read> word = step != nullptr ? step + index : Ptr<uint, Access.Read>(read_only);
    uint value = word[0];
    return step != nullptr ? value : fallback;
}

public bool is_ptr_aligned16<T, let A : Access>(Ptr<T, A> p) {
    uint64_t addr = spirv_asm { result : $$uint64_t = OpConvertPtrToU $p; };
    return (uint(addr) & 15u) == 0u;
}

public bool is_ptr_aligned4<T, let A : Access>(Ptr<T, A> p) {
    uint64_t addr = spirv_asm { result : $$uint64_t = OpConvertPtrToU $p; };
    return (uint(addr) & 3u) == 0u;
}

public T load_a16<T>(Ptr<T, Access.Read> p) {
    return spirv_asm { result : $$T = OpLoad $p Aligned 16; };
}

public void store_a16<T>(Ptr<T> p, T v) {
    spirv_asm { OpStore $p $v Aligned 16; };
}

public float4 load_f32x4_maybe_a16(Ptr<float, Access.Read> p) {
    if (is_ptr_aligned16(p)) {
        return load_a16(Ptr<float4, Access.Read>(p));
    }
    return float4(p[0], p[1], p[2], p[3]);
}

public void store_f32x4_maybe_a16(Ptr<float> p, float4 v) {
    if (is_ptr_aligned16(p)) {
        store_a16(Ptr<float4>(p), v);
    } else {
        p[0] = v.x;
        p[1] = v.y;
        p[2] = v.z;
        p[3] = v.w;
    }
}

public void copy_f32_chunk4(
    Ptr<float, Access.Read> src, Ptr<float> dst, uint chunk_idx, uint dim)
{
    uint base = chunk_idx << 2u;
    if (base + 4u <= dim && is_ptr_aligned16(src) && is_ptr_aligned16(dst)) {
        float4 v = load_a16(Ptr<float4, Access.Read>(src) + uint64_t(chunk_idx));
        store_a16(Ptr<float4>(dst) + uint64_t(chunk_idx), v);
    } else {
        uint end = min(base + 4u, dim);
        for (uint c = base; c < end; ++c) dst[c] = src[c];
    }
}

public void zero_f32_chunk4(Ptr<float> dst, uint chunk_idx, uint dim) {
    uint base = chunk_idx << 2u;
    if (base + 4u <= dim && is_ptr_aligned16(dst)) {
        store_a16(Ptr<float4>(dst) + uint64_t(chunk_idx), float4(0.0f));
    } else {
        uint end = min(base + 4u, dim);
        for (uint c = base; c < end; ++c) dst[c] = 0.0f;
    }
}

public void store_epilogue_f32x4<Ep : IEpilogue>(
    Ptr<float> yr, uint row, uint col, uint m, bool y_aligned4,
    float4 v, EpilogueParams ep)
{
    if (y_aligned4 && (col + 4u <= m)) {
        float4 outv = float4(
            Ep::apply(v.x, row, col + 0u, ep),
            Ep::apply(v.y, row, col + 1u, ep),
            Ep::apply(v.z, row, col + 2u, ep),
            Ep::apply(v.w, row, col + 3u, ep));
        store_a16(Ptr<float4>(yr + col), outv);
    } else {
        if (col + 0u < m) yr[col + 0u] = Ep::apply(v.x, row, col + 0u, ep);
        if (col + 1u < m) yr[col + 1u] = Ep::apply(v.y, row, col + 1u, ep);
        if (col + 2u < m) yr[col + 2u] = Ep::apply(v.z, row, col + 2u, ep);
        if (col + 3u < m) yr[col + 3u] = Ep::apply(v.w, row, col + 3u, ep);
    }
}

public interface IVecIO128<T : IElement> {
    static const uint kElems; // 4 for F32, 8 for F16/BF16
    static void unpack(uint4 raw, out float4 lo, out float4 hi);
    static uint4 pack(float4 lo, float4 hi);
}

public struct VecIOF32 : IVecIO128<ElemF32> {
    public static const uint kElems = 4u;
    public static void unpack(uint4 raw, out float4 lo, out float4 hi) {
        lo = asfloat(raw);
        hi = float4(0.0f);
    }
    public static uint4 pack(float4 lo, float4 hi) { return asuint(lo); }
}

public struct VecIOF16 : IVecIO128<ElemF16> {
    public static const uint kElems = 8u;
    public static void unpack(uint4 r, out float4 lo, out float4 hi) {
        lo = float4(
            float(bit_cast<half>(uint16_t(r.x & 0xFFFFu))),
            float(bit_cast<half>(uint16_t(r.x >> 16u))),
            float(bit_cast<half>(uint16_t(r.y & 0xFFFFu))),
            float(bit_cast<half>(uint16_t(r.y >> 16u))));
        hi = float4(
            float(bit_cast<half>(uint16_t(r.z & 0xFFFFu))),
            float(bit_cast<half>(uint16_t(r.z >> 16u))),
            float(bit_cast<half>(uint16_t(r.w & 0xFFFFu))),
            float(bit_cast<half>(uint16_t(r.w >> 16u))));
    }
    public static uint4 pack(float4 lo, float4 hi) {
        uint x = uint(bit_cast<uint16_t>(half(lo.x))) | (uint(bit_cast<uint16_t>(half(lo.y))) << 16u);
        uint y = uint(bit_cast<uint16_t>(half(lo.z))) | (uint(bit_cast<uint16_t>(half(lo.w))) << 16u);
        uint z = uint(bit_cast<uint16_t>(half(hi.x))) | (uint(bit_cast<uint16_t>(half(hi.y))) << 16u);
        uint w = uint(bit_cast<uint16_t>(half(hi.z))) | (uint(bit_cast<uint16_t>(half(hi.w))) << 16u);
        return uint4(x, y, z, w);
    }
}

public struct VecIOBF16 : IVecIO128<ElemBF16> {
    public static const uint kElems = 8u;
    public static void unpack(uint4 r, out float4 lo, out float4 hi) {
        lo = float4(
            asfloat((r.x & 0xFFFFu) << 16u), asfloat(r.x & 0xFFFF0000u),
            asfloat((r.y & 0xFFFFu) << 16u), asfloat(r.y & 0xFFFF0000u));
        hi = float4(
            asfloat((r.z & 0xFFFFu) << 16u), asfloat(r.z & 0xFFFF0000u),
            asfloat((r.w & 0xFFFFu) << 16u), asfloat(r.w & 0xFFFF0000u));
    }
    public static uint4 pack(float4 lo, float4 hi) {
        uint x = uint(ElemBF16::fromF32(lo.x)) | (uint(ElemBF16::fromF32(lo.y)) << 16u);
        uint y = uint(ElemBF16::fromF32(lo.z)) | (uint(ElemBF16::fromF32(lo.w)) << 16u);
        uint z = uint(ElemBF16::fromF32(hi.x)) | (uint(ElemBF16::fromF32(hi.y)) << 16u);
        uint w = uint(ElemBF16::fromF32(hi.z)) | (uint(ElemBF16::fromF32(hi.w)) << 16u);
        return uint4(x, y, z, w);
    }
}

public float4 apply_act4<A : IActivation>(float4 v) {
    return float4(A::apply(v.x), A::apply(v.y), A::apply(v.z), A::apply(v.w));
}

public uint2 swizzle_tile_2d(uint3 gid, uint grid_x, uint grid_y) {
    uint linear = gid.y * grid_x + gid.x;
    uint tiles_per_grp = 8u * grid_x;
    uint grp = linear / tiles_per_grp;
    uint rem = linear - grp * tiles_per_grp;
    uint first_y = grp * 8u;
    uint grp_h = min(8u, grid_y - first_y);
    return uint2(rem / grp_h, first_y + (rem % grp_h));
}

// Unified 80-byte C-layout push-constant block shared by GEMV, F32 GEMM, and CoopMat GEMM.
public struct LinearArgs<TXStorage, TWStorage, TYStorage> {
    public Ptr<TXStorage, Access.Read> x;
    public Ptr<TWStorage, Access.Read> w;
    public Ptr<float, Access.Read> bias;
    public Ptr<float, Access.Read> x_scale;
    public Ptr<float, Access.Read> w_scale;
    public Ptr<TYStorage> y;
    public uint rows;
    public uint m;
    public uint k;
    public uint x_stride;
    public uint w_stride;
    public uint y_stride;
    public uint flags;
    public uint split_k;
};
