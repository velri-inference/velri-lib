// Shared output epilogue components for Linear (GEMV & GEMM) across CPU and Vulkan.
module epilogue;

public static const uint kEpFlagBias = 1u;
public static const uint kEpFlagXScale = 2u;
public static const uint kEpFlagWScale = 4u;

// CPU linear execution flags (bits 3..5; mirrored in cpu_backend.c).
public static const uint kLinearFlagDirectOrder = 1u << 3u;
public static const uint kLinearFlagStreamWeights = 1u << 4u;
public static const uint kLinearFlagAlignedWeights = 1u << 5u;

// Raw W8A8 decode switches to sequential single-channel streaming above this slice size.
public static const uint kW8A8StreamMinK = 1024u;
public static const uint kW8A8StreamMinBytes = 131072u;

[ForceInline] public bool w8a8_should_stream_rows(bool direct_order, uint width, uint k) {
    return !direct_order && k >= kW8A8StreamMinK && width > (kW8A8StreamMinBytes / k);
}

public struct EpilogueParams {
    public Ptr<float, Access.Read> bias;
    public Ptr<float, Access.Read> x_scale;
    public Ptr<float, Access.Read> w_scale;
    public uint has_bias;
    public uint has_x_scale;
    public uint has_w_scale;
};

public EpilogueParams make_epilogue_params(
    Ptr<float, Access.Read> bias,
    Ptr<float, Access.Read> x_scale,
    Ptr<float, Access.Read> w_scale,
    uint flags)
{
    EpilogueParams ep;
    ep.bias = bias;
    ep.x_scale = x_scale;
    ep.w_scale = w_scale;
    ep.has_bias = flags & kEpFlagBias;
    ep.has_x_scale = flags & kEpFlagXScale;
    ep.has_w_scale = flags & kEpFlagWScale;
    return ep;
}

public interface IEpilogue {
    static bool isIdentity(EpilogueParams ep);
    static float apply(float acc, uint row, uint col, EpilogueParams ep);
}

// Standard linear epilogue: optional bias[col].
public struct EpBias : IEpilogue {
    [ForceInline] public static bool isIdentity(EpilogueParams ep) {
        return ep.has_bias == 0u;
    }

    public static float apply(float acc, uint row, uint col, EpilogueParams ep) {
        if (ep.has_bias != 0u) {
            acc += ep.bias[col];
        }
        return acc;
    }
}

// Quantized scale epilogue: supports both W8A32 (w_scale) and W8A8 (x_scale * w_scale) + bias.
public struct EpQuantScale : IEpilogue {
    [ForceInline] public static bool isIdentity(EpilogueParams ep) {
        return (ep.has_bias | ep.has_x_scale | ep.has_w_scale) == 0u;
    }

    [ForceInline] public static float apply_scales(float acc, float w_scale, float x_scale) {
        return acc * (w_scale * x_scale);
    }

    public static float apply(float acc, uint row, uint col, EpilogueParams ep) {
        float s = 1.0f;
        if (ep.has_w_scale != 0u) s *= ep.w_scale[col];
        if (ep.has_x_scale != 0u) acc = apply_scales(acc, s, ep.x_scale[row]);
        else acc *= s;
        return EpBias::apply(acc, row, col, ep);
    }
}
