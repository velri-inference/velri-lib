// Shared numeric storage/conversion abstractions for Velri-lib Slang kernels.
// One interface powers FP32, FP16, BF16, INT8, and F64 across CPU and Vulkan
// without duplicating kernel implementations per dtype.
module numeric;

public interface IElement {
    associatedtype Storage;
    static const bool kIsNativeF32;
    static float toF32(Storage v);
    static Storage fromF32(float x);
    static float4 load4F32(Ptr<Storage, Access.Read> p);
}

public struct ElemF32 : IElement {
    public typealias Storage = float;
    public static const bool kIsNativeF32 = true;
    public static float toF32(float v) { return v; }
    public static float fromF32(float x) { return x; }
    public static float4 load4F32(Ptr<float, Access.Read> p) {
        return float4(p[0], p[1], p[2], p[3]);
    }
}

#ifdef VELRI_SOFT_F16
// Software IEEE-754 binary16 conversion for baseline x86-64 without F16C,
// avoiding external __extendhfsf2/__truncsfhf2 libcalls (e.g. Windows MSVC).
static float velri_soft_f16_to_f32(uint16_t h) {
    uint sign = (uint(h) & 0x8000u) << 16u;
    uint exp = (uint(h) >> 10u) & 0x1Fu;
    uint mant = uint(h) & 0x03FFu;
    if (exp == 0u) {
        if (mant == 0u) return asfloat(sign);
        uint e = 127u - 14u;
        while ((mant & 0x0400u) == 0u) {
            mant <<= 1u;
            e -= 1u;
        }
        mant &= 0x03FFu;
        return asfloat(sign | (e << 23u) | (mant << 13u));
    }
    if (exp == 0x1Fu) {
        if (mant == 0u) return asfloat(sign | 0x7F800000u);
        return asfloat(sign | 0x7FC00000u | (mant << 13u));
    }
    return asfloat(sign | ((exp + (127u - 15u)) << 23u) | (mant << 13u));
}

static uint16_t velri_soft_f32_to_f16(float x) {
    uint u = asuint(x);
    uint sign = (u >> 16u) & 0x8000u;
    uint abs_u = u & 0x7FFFFFFFu;
    if (abs_u > 0x7F800000u) {
        return uint16_t(sign | 0x7E00u | ((abs_u >> 13u) & 0x03FFu));
    }
    if (abs_u >= 0x477FF000u) {
        if (abs_u >= 0x7F800000u) return uint16_t(sign | 0x7C00u);
        uint exp = (abs_u >> 23u) - 127u + 15u;
        uint mant = abs_u & 0x007FFFFFu;
        if (exp > 30u) return uint16_t(sign | 0x7C00u);
        uint r = mant + 0x0FFFu + ((mant >> 13u) & 1u);
        if (r >= 0x00800000u) return uint16_t(sign | 0x7C00u);
        return uint16_t(sign | (exp << 10u) | (r >> 13u));
    }
    if (abs_u < 0x33000000u) return uint16_t(sign);
    if (abs_u < 0x38800000u) {
        uint f_exp = abs_u >> 23u;
        uint sig = (abs_u & 0x007FFFFFu) | 0x00800000u;
        uint shift = 126u - f_exp;
        uint q = sig >> shift;
        uint rem = sig & ((1u << shift) - 1u);
        uint half_b = 1u << (shift - 1u);
        if (rem > half_b || (rem == half_b && (q & 1u) != 0u)) q += 1u;
        return uint16_t(sign | q);
    }
    uint exp = (abs_u >> 23u) - (127u - 15u);
    uint mant = abs_u & 0x007FFFFFu;
    uint rounded = mant + 0x0FFFu + ((mant >> 13u) & 1u);
    if (rounded >= 0x00800000u) {
        rounded -= 0x00800000u;
        exp += 1u;
        if (exp >= 31u) return uint16_t(sign | 0x7C00u);
    }
    return uint16_t(sign | (exp << 10u) | (rounded >> 13u));
}

public struct ElemF16 : IElement {
    public typealias Storage = uint16_t;
    public static const bool kIsNativeF32 = false;
    public static float toF32(uint16_t v) { return velri_soft_f16_to_f32(v); }
    public static uint16_t fromF32(float x) { return velri_soft_f32_to_f16(x); }
    public static float4 load4F32(Ptr<uint16_t, Access.Read> p) {
        return float4(toF32(p[0]), toF32(p[1]), toF32(p[2]), toF32(p[3]));
    }
}
#else
public struct ElemF16 : IElement {
    public typealias Storage = half;
    public static const bool kIsNativeF32 = false;
    public static float toF32(half v) { return float(v); }
    public static half fromF32(float x) { return half(x); }
    public static float4 load4F32(Ptr<half, Access.Read> p) {
        half4 h = half4(p[0], p[1], p[2], p[3]);
        return float4(h);
    }
}
#endif

// Stored as uint16_t so one source compiles cleanly to both CPU and Vulkan SPIR-V.
public struct ElemBF16 : IElement {
    public typealias Storage = uint16_t;
    public static const bool kIsNativeF32 = false;
    public static float toF32(uint16_t v) { return asfloat(uint(v) << 16u); }
    public static uint16_t fromF32(float x) {
        uint bits = asuint(x);
        if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
            return uint16_t(0x7FC0u);
        }
        uint rounding = 0x7FFFu + ((bits >> 16u) & 1u);
        return uint16_t((bits + rounding) >> 16u);
    }
    public static float4 load4F32(Ptr<uint16_t, Access.Read> p) {
        uint4 u = uint4(uint(p[0]), uint(p[1]), uint(p[2]), uint(p[3]));
        return asfloat(u << 16u);
    }
}

public struct ElemI8 : IElement {
    public typealias Storage = int8_t;
    public static const bool kIsNativeF32 = false;
    public static float toF32(int8_t v) { return float(int(v)); }
    public static int8_t fromF32(float x) { return int8_t(clamp(x, -127.0f, 127.0f)); }
    public static float4 load4F32(Ptr<int8_t, Access.Read> p) {
        int4 v = int4(int(p[0]), int(p[1]), int(p[2]), int(p[3]));
        return float4(v);
    }
}

// ============================================================================
// Shared Numeric Constants & Target-Agnostic Math Helpers (CPU + Vulkan)
// ============================================================================

// Finite -ffast-math safe negative-infinity sentinel and cutoff for softmax/attention.
public static const float kNegInfSentinel = -3.4028234e38f;
public static const float kNegInfCutoff = -3.4e38f;

// Supported attention head-dimension tiers (fast partition-parallel vs. wide path).
public static const uint kAttnFastHeadDim = 256u;
public static const uint kAttnMaxHeadDim = 2048u;

// Symmetric per-row INT8 quantization constants.
public static const float kSymI8Eps = 1.0e-35f;
public static const float kSymI8MinScale = 1.17549435e-38f;
public static const float kSymI8MaxQ = 127.0f;

// Compute symmetric per-row INT8 scale and safe reciprocal scale across normal,
// tiny-positive (sub-1e-35 where 127/amax would overflow), and zero rows.
public void compute_sym_i8_scale(float amax, out float scale, out float inv_scale) {
    if (amax >= kSymI8Eps) {
        scale = amax / kSymI8MaxQ;
        inv_scale = kSymI8MaxQ / amax;
    } else if (amax > 0.0f) {
        scale = max(amax / kSymI8MaxQ, kSymI8MinScale);
        inv_scale = 1.0f / scale;
    } else {
        scale = 1.0f;
        inv_scale = 0.0f;
    }
}

public int8_t quantize_sym_i8(float v, float inv_scale) {
    float r = round(v * inv_scale);
    float t = select(r < -kSymI8MaxQ, -kSymI8MaxQ, r);
    float c = select(t > kSymI8MaxQ, kSymI8MaxQ, t);
    return int8_t(int32_t(c));
}

// 4-point unnormalized Walsh-Hadamard transform (radix-4 Stockham butterfly).
public float4 hadamard4(float4 v) {
    float t0 = v.x + v.y;
    float t1 = v.x - v.y;
    float t2 = v.z + v.w;
    float t3 = v.z - v.w;
    return float4(t0 + t2, t1 + t3, t0 - t2, t1 - t3);
}

// 2D rotary complex multiplication used by both HALF and INTERLEAVED RoPE.
public void rope_rotate_pair(float x0, float x1, float c, float s, out float y0, out float y1) {
    y0 = x0 * c - x1 * s;
    y1 = x1 * c + x0 * s;
}

public void rope_rotate_pair(float4 x0, float4 x1, float4 c, float4 s, out float4 y0, out float4 y1) {
    y0 = x0 * c - x1 * s;
    y1 = x1 * c + x0 * s;
}
