// Shared RMSNorm, Fused Residual+RMSNorm, and LayerNorm math for CPU.
// Reuses common/numeric.slang (IElement) across all dtypes.
module norm;

import numeric;

void rmsnorm_apply_row_cpu<TIO : IElement, TW : IElement>(
    Ptr<TIO.Storage, Access.Read> src,
    Ptr<TW.Storage, Access.Read> weight, uint has_weight,
    Ptr<TIO.Storage> dst,
    uint dim, float inv_rms, uint unit_offset)
{
    if (has_weight != 0u) {
        if (unit_offset != 0u) {
            for (uint d = 0u; d < dim; ++d) {
                float v = TIO::toF32(src[d]) * inv_rms * (1.0f + TW::toF32(weight[d]));
                dst[d] = TIO::fromF32(v);
            }
        } else {
            for (uint d = 0u; d < dim; ++d) {
                float v = TIO::toF32(src[d]) * inv_rms * TW::toF32(weight[d]);
                dst[d] = TIO::fromF32(v);
            }
        }
    } else {
        for (uint d = 0u; d < dim; ++d) {
            dst[d] = TIO::fromF32(TIO::toF32(src[d]) * inv_rms);
        }
    }
}

public void rmsnorm_rows_cpu<TIO : IElement, TW : IElement>(
    Ptr<TIO.Storage, Access.Read> x, uint x_stride,
    Ptr<TW.Storage, Access.Read> weight, uint has_weight,
    Ptr<TIO.Storage> y, uint y_stride,
    uint r0, uint r1, uint dim, float eps, uint unit_offset)
{
    float inv_dim = 1.0f / float(dim);
    for (uint r = r0; r < r1; ++r) {
        Ptr<TIO.Storage, Access.Read> xr = x + uint64_t(r) * uint64_t(x_stride);
        Ptr<TIO.Storage> yr = y + uint64_t(r) * uint64_t(y_stride);
        float sum_sq = 0.0f;
        for (uint d = 0u; d < dim; ++d) {
            float v = TIO::toF32(xr[d]);
            sum_sq += v * v;
        }
        float inv_rms = 1.0f / sqrt(sum_sq * inv_dim + eps);
        rmsnorm_apply_row_cpu<TIO, TW>(xr, weight, has_weight, yr, dim, inv_rms, unit_offset);
    }
}

public void fused_add_rmsnorm_rows_cpu(
    Ptr<float> residual_inout, uint r_stride,
    Ptr<float, Access.Read> x, uint x_stride,
    Ptr<float, Access.Read> weight, uint has_weight,
    Ptr<float> y, uint y_stride,
    uint r0, uint r1, uint dim, float eps, uint unit_offset)
{
    float inv_dim = 1.0f / float(dim);
    for (uint r = r0; r < r1; ++r) {
        Ptr<float> rr = residual_inout + uint64_t(r) * uint64_t(r_stride);
        Ptr<float, Access.Read> xr = x + uint64_t(r) * uint64_t(x_stride);
        Ptr<float> yr = y + uint64_t(r) * uint64_t(y_stride);
        float sum_sq = 0.0f;
        for (uint d = 0u; d < dim; ++d) {
            float s = rr[d] + xr[d];
            rr[d] = s;
            sum_sq += s * s;
        }
        float inv_rms = 1.0f / sqrt(sum_sq * inv_dim + eps);
        rmsnorm_apply_row_cpu<ElemF32, ElemF32>(
            Ptr<float, Access.Read>(rr), weight, has_weight, yr, dim, inv_rms, unit_offset);
    }
}

public void layernorm_rows_cpu(
    Ptr<float, Access.Read> x, uint x_stride,
    Ptr<float, Access.Read> weight, uint has_weight,
    Ptr<float, Access.Read> bias, uint has_bias,
    Ptr<float> y, uint y_stride,
    uint r0, uint r1, uint dim, float eps)
{
    float inv_dim = 1.0f / float(dim);
    for (uint r = r0; r < r1; ++r) {
        Ptr<float, Access.Read> xr = x + uint64_t(r) * uint64_t(x_stride);
        Ptr<float> yr = y + uint64_t(r) * uint64_t(y_stride);
        float sum = 0.0f;
        for (uint d = 0u; d < dim; ++d) sum += xr[d];
        float mean = sum * inv_dim;
        float var_sum = 0.0f;
        for (uint d = 0u; d < dim; ++d) {
            float df = xr[d] - mean;
            var_sum += df * df;
        }
        float inv_std = 1.0f / sqrt(var_sum * inv_dim + eps);
        if (has_weight != 0u && has_bias != 0u) {
            for (uint d = 0u; d < dim; ++d) yr[d] = fma((xr[d] - mean) * inv_std, weight[d], bias[d]);
        } else if (has_weight != 0u) {
            for (uint d = 0u; d < dim; ++d) yr[d] = (xr[d] - mean) * inv_std * weight[d];
        } else if (has_bias != 0u) {
            for (uint d = 0u; d < dim; ++d) yr[d] = fma(xr[d] - mean, inv_std, bias[d]);
        } else {
            for (uint d = 0u; d < dim; ++d) yr[d] = (xr[d] - mean) * inv_std;
        }
    }
}
