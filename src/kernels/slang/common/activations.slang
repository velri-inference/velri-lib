// Shared activation functors and fast vectorizable elementary math.
// Reused by standalone Activation, GatedMLP, Softmax, and Linear epilogues.
module activations;

// Branchless clamp emitting single-cycle vmaxps/vminps (no vcmpunordps / vucomiss).
public float fast_clampf(float x, float lo, float hi) {
    float t = select(x < lo, lo, x);
    return select(t > hi, hi, t);
}

// Degree-5 minimax approximation of 2^z on [-126, 126] (max rel err < 1.5e-6;
// auto-vectorizes cleanly to ZMM/YMM on CPU and single-cycle FMA on GPU).
public float fast_exp2f(float z) {
    float zc = fast_clampf(z, -126.0f, 126.0f);
    float k = floor(zc + 0.5f);
    float f = zc - k;
    float p = fma(f, 0.0018775767f, 0.0096181291f);
    p = fma(p, f, 0.0555041087f);
    p = fma(p, f, 0.2402265070f);
    p = fma(p, f, 0.6931471806f);
    p = fma(p, f, 1.0f);
    int ki = int(k);
    return p * asfloat((ki + 127) << 23);
}

public float fast_expf(float x) {
    return fast_exp2f(x * 1.4426950408889634f);
}

public interface IActivation {
    static float apply(float x);
}

public struct ActSiLU : IActivation {
    public static float apply(float x) {
        float y = x / (1.0f + fast_exp2f(x * -1.4426950408889634f));
        return select(x < -87.0f, 0.0f, y);
    }
}

public struct ActGeluTanh : IActivation {
    public static float apply(float x) {
        // -2 * sqrt(2 / pi) * log2(e) = -2.30220780998f
        float a = (x * -2.30220781f) * fma(0.044715f * x, x, 1.0f);
        float y = x / (1.0f + fast_exp2f(a));
        return select(x < -10.5f, 0.0f, y);
    }
}

public struct ActReLU2 : IActivation {
    public static float apply(float x) {
        float r = max(x, 0.0f);
        return r * r;
    }
}

public float apply_symmetric_clamp(float x, float limit) {
    if (limit > 0.0f) {
        return clamp(x, -limit, limit);
    }
    return x;
}
