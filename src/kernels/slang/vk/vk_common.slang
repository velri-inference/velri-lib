// Shared Vulkan BDA 128-bit load/store helpers, IVecIO128 unpackers, and tile swizzling.
module vk_common;

import numeric;
import activations;
import epilogue;

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

public EpilogueParams make_vk_ep(Ptr<float, Access.Read> bias, Ptr<float, Access.Read> xs,
                                 Ptr<float, Access.Read> ws, uint flags)
{
    EpilogueParams ep;
    ep.bias = bias;
    ep.x_scale = xs;
    ep.w_scale = ws;
    ep.has_bias = flags & 1u;
    ep.has_x_scale = flags & 2u;
    ep.has_w_scale = flags & 4u;
    return ep;
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

public float4 apply_clamp4(float4 v, float lim) {
    if (lim <= 0.0f) return v;
    return clamp(v, float4(-lim), float4(lim));
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

public struct GemvArgs<TX : IElement, TW : IElement, TY : IElement> {
    public Ptr<TX.Storage, Access.Read> x;
    public Ptr<TW.Storage, Access.Read> w;
    public Ptr<float, Access.Read> bias;
    public Ptr<float, Access.Read> x_scale;
    public Ptr<float, Access.Read> w_scale;
    public Ptr<TY.Storage> y;
    public uint rows;
    public uint m;
    public uint k;
    public uint x_stride;
    public uint w_stride;
    public uint y_stride;
    public uint flags;
    public uint split_k;
};
