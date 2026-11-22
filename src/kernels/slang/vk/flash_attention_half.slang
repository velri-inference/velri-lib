module flash_attention_half;

// Reconstruct the rounded half's F32 bits explicitly. Native float(half(x)) can
// contract inside x - float(half(x)) and discard the low part on some drivers.
public float widen_half(uint bits) {
    uint h = bits & 65535u, fraction = h & 1023u, exponent = (h >> 10u) & 31u;
    uint sign = (h & 32768u) << 16u;
    if (exponent == 0u) return asfloat(sign | 0x3f800000u) * float(fraction) * 5.960464477539063e-8f;
    uint f32_exponent = exponent == 31u ? 255u : exponent + 112u;
    return asfloat(sign | (f32_exponent << 23u) | (fraction << 13u));
}
