// Shared output epilogue components for Linear (GEMV & GEMM) across CPU and Vulkan.
module epilogue;

public struct EpilogueParams {
    public Ptr<float, Access.Read> bias;
    public Ptr<float, Access.Read> x_scale;
    public Ptr<float, Access.Read> w_scale;
    public uint has_bias;
    public uint has_x_scale;
    public uint has_w_scale;
};

public interface IEpilogue {
    static float apply(float acc, uint row, uint col, EpilogueParams ep);
}

// Standard linear epilogue: optional bias[col].
public struct EpBias : IEpilogue {
    public static float apply(float acc, uint row, uint col, EpilogueParams ep) {
        if (ep.has_bias != 0u) {
            acc += ep.bias[col];
        }
        return acc;
    }
}

// Quantized scale epilogue: supports both W8A32 (w_scale) and W8A8 (x_scale * w_scale) + bias.
public struct EpQuantScale : IEpilogue {
    public static float apply(float acc, uint row, uint col, EpilogueParams ep) {
        float s = 1.0f;
        if (ep.has_w_scale != 0u) s *= ep.w_scale[col];
        if (ep.has_x_scale != 0u) s *= ep.x_scale[row];
        acc *= s;
        if (ep.has_bias != 0u) acc += ep.bias[col];
        return acc;
    }
}
