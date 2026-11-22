// Shared elementwise, gated MLP, softmax, FWHT, and row-quantization operations for CPU.
module elementwise;

import numeric;
import activations;

public void activation_cpu<A : IActivation>(
    Ptr<float, Access.Read> x, Ptr<float> y, uint i0, uint i1)
{
    for (uint i = i0; i < i1; ++i) {
        y[i] = A::apply(x[i]);
    }
}

public void gated_mlp_cpu<A : IActivation>(
    Ptr<float, Access.Read> gate,
    Ptr<float, Access.Read> up,
    Ptr<float> y,
    uint i0, uint i1, float clamp_limit)
{
    if (clamp_limit > 0.0f) {
        for (uint i = i0; i < i1; ++i) {
            float g = apply_symmetric_clamp(gate[i], clamp_limit);
            float u = apply_symmetric_clamp(up[i], clamp_limit);
            y[i] = A::apply(g) * u;
        }
    } else {
        for (uint i = i0; i < i1; ++i) {
            y[i] = A::apply(gate[i]) * up[i];
        }
    }
}

float row_max_f32_cpu(Ptr<float, Access.Read> xr, uint cols) {
    float m = kNegInfSentinel;
    for (uint c = 0u; c < cols; ++c) {
        float v = xr[c];
        m = select(v > m, v, m);
    }
    return m;
}

public void softmax_rows_cpu(
    Ptr<float, Access.Read> x, uint x_stride,
    Ptr<float> y, uint y_stride,
    uint r0, uint r1, uint cols)
{
    if (cols <= 4096u) {
        float tmp[4096];
        for (uint r = r0; r < r1; ++r) {
            Ptr<float, Access.Read> xr = x + uint64_t(r) * uint64_t(x_stride);
            Ptr<float> yr = y + uint64_t(r) * uint64_t(y_stride);
            float m = row_max_f32_cpu(xr, cols);
            if (!(m > kNegInfCutoff)) {
                for (uint c = 0u; c < cols; ++c) yr[c] = 0.0f;
                continue;
            }
            float sum = 0.0f;
            for (uint c = 0u; c < cols; ++c) {
                float v = xr[c];
                float e = select(v > kNegInfCutoff, fast_expf(v - m), 0.0f);
                tmp[c] = e;
                sum += e;
            }
            float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
            for (uint c = 0u; c < cols; ++c) {
                yr[c] = tmp[c] * inv_sum;
            }
        }
        return;
    }
    for (uint r = r0; r < r1; ++r) {
        Ptr<float, Access.Read> xr = x + uint64_t(r) * uint64_t(x_stride);
        Ptr<float> yr = y + uint64_t(r) * uint64_t(y_stride);
        float m = row_max_f32_cpu(xr, cols);
        if (!(m > kNegInfCutoff)) {
            for (uint c = 0u; c < cols; ++c) yr[c] = 0.0f;
            continue;
        }
        float sum = 0.0f;
        for (uint c = 0u; c < cols; ++c) {
            float v = xr[c];
            float e = select(v > kNegInfCutoff, fast_expf(v - m), 0.0f);
            yr[c] = e;
            sum += e;
        }
        float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        for (uint c = 0u; c < cols; ++c) {
            yr[c] *= inv_sum;
        }
    }
}

public void axpby_cpu(
    float alpha, Ptr<float, Access.Read> a,
    float beta, Ptr<float, Access.Read> b,
    Ptr<float> y, uint i0, uint i1)
{
    if (beta == 0.0f) {
        for (uint i = i0; i < i1; ++i) y[i] = alpha * a[i];
    } else {
        for (uint i = i0; i < i1; ++i) y[i] = fma(alpha, a[i], beta * b[i]);
    }
}

public void mul_cpu(
    Ptr<float, Access.Read> a,
    Ptr<float, Access.Read> b,
    Ptr<float> y, uint i0, uint i1)
{
    for (uint i = i0; i < i1; ++i) {
        y[i] = a[i] * b[i];
    }
}

void stockham_r4_cpu<let B : int>(Ptr<float> src, Ptr<float> dst) {
    static const uint kQ = uint(B) >> 2u;
    for (uint k = 0u; k < kQ; ++k) {
        float4 h = hadamard4(float4(
            src[k * 4u + 0u], src[k * 4u + 1u], src[k * 4u + 2u], src[k * 4u + 3u]));
        dst[k + 0u * kQ] = h.x;
        dst[k + 1u * kQ] = h.y;
        dst[k + 2u * kQ] = h.z;
        dst[k + 3u * kQ] = h.w;
    }
}

void stockham_r2_cpu<let B : int>(Ptr<float> src, Ptr<float> dst) {
    static const uint kH = uint(B) >> 1u;
    for (uint k = 0u; k < kH; ++k) {
        float a = src[k * 2u];
        float b = src[k * 2u + 1u];
        dst[k] = a + b;
        dst[k + kH] = a - b;
    }
}

void fwht_rows_b_cpu<let B : int, TSuh : IElement, let HAS_SUH : bool>(
    Ptr<float, Access.Read> x, uint x_stride,
    Ptr<TSuh.Storage, Access.Read> suh,
    Ptr<float> y, uint y_stride,
    uint r0, uint r1, uint cols)
{
    static const uint kB = uint(B);
    static const uint kQ = kB >> 2u;
    float norm = 1.0f / sqrt(float(B));
    float bufA[B];
    float bufB[B];
    bufA[0] = 0.0f; bufB[0] = 0.0f;
    Ptr<float> pa = &bufA[0];
    Ptr<float> pb = &bufB[0];

    for (uint r = r0; r < r1; ++r) {
        Ptr<float, Access.Read> xr = x + uint64_t(r) * uint64_t(x_stride);
        Ptr<float> yr = y + uint64_t(r) * uint64_t(y_stride);
        for (uint c0 = 0u; c0 < cols; c0 += kB) {
            for (uint k = 0u; k < kQ; ++k) {
                uint base = c0 + k * 4u;
                float4 a = float4(
                    xr[base + 0u], xr[base + 1u], xr[base + 2u], xr[base + 3u]) * norm;
                if (HAS_SUH) {
                    a *= TSuh::load4F32(suh + base);
                }
                float4 h = hadamard4(a);
                pa[k + 0u * kQ] = h.x;
                pa[k + 1u * kQ] = h.y;
                pa[k + 2u * kQ] = h.z;
                pa[k + 3u * kQ] = h.w;
            }
            if (B == 16) {
                stockham_r4_cpu<16>(pa, yr + c0);
            } else if (B == 32) {
                stockham_r4_cpu<32>(pa, pb);
                stockham_r2_cpu<32>(pb, yr + c0);
            } else if (B == 64) {
                stockham_r4_cpu<64>(pa, pb);
                stockham_r4_cpu<64>(pb, yr + c0);
            } else if (B == 128) {
                stockham_r4_cpu<128>(pa, pb);
                stockham_r4_cpu<128>(pb, pa);
                stockham_r2_cpu<128>(pa, yr + c0);
            } else {
                stockham_r4_cpu<256>(pa, pb);
                stockham_r4_cpu<256>(pb, pa);
                stockham_r4_cpu<256>(pa, yr + c0);
            }
        }
    }
}

public void fwht_rows_cpu<TSuh : IElement, let HAS_SUH : bool>(
    Ptr<float, Access.Read> x, uint x_stride,
    Ptr<TSuh.Storage, Access.Read> suh,
    Ptr<float> y, uint y_stride,
    uint r0, uint r1, uint cols, uint block_size)
{
    if (block_size == 128u)      fwht_rows_b_cpu<128, TSuh, HAS_SUH>(x, x_stride, suh, y, y_stride, r0, r1, cols);
    else if (block_size == 256u) fwht_rows_b_cpu<256, TSuh, HAS_SUH>(x, x_stride, suh, y, y_stride, r0, r1, cols);
    else if (block_size == 64u)  fwht_rows_b_cpu<64,  TSuh, HAS_SUH>(x, x_stride, suh, y, y_stride, r0, r1, cols);
    else if (block_size == 32u)  fwht_rows_b_cpu<32,  TSuh, HAS_SUH>(x, x_stride, suh, y, y_stride, r0, r1, cols);
    else                         fwht_rows_b_cpu<16,  TSuh, HAS_SUH>(x, x_stride, suh, y, y_stride, r0, r1, cols);
}

float row_max_abs_f32_cpu(Ptr<float, Access.Read> xr, uint cols) {
    float m = 0.0f;
    for (uint c = 0u; c < cols; ++c) {
        float a = abs(xr[c]);
        m = select(a > m, a, m);
    }
    return m;
}

public void quantize_rows_f32_i8_cpu(
    Ptr<float, Access.Read> x, uint x_stride,
    Ptr<int8_t> x_i8, uint xi8_stride,
    Ptr<float> scale_out,
    uint r0, uint r1, uint cols)
{
    for (uint r = r0; r < r1; ++r) {
        Ptr<float, Access.Read> xr = x + uint64_t(r) * uint64_t(x_stride);
        Ptr<int8_t> qr = x_i8 + uint64_t(r) * uint64_t(xi8_stride);
        float amax = row_max_abs_f32_cpu(xr, cols);
        float scale, inv_scale;
        compute_sym_i8_scale(amax, scale, inv_scale);
        scale_out[r] = scale;
        if (inv_scale == 0.0f) {
            for (uint c = 0u; c < cols; ++c) {
                qr[c] = int8_t(0);
            }
        } else {
            for (uint c = 0u; c < cols; ++c) {
                qr[c] = quantize_sym_i8(xr[c], inv_scale);
            }
        }
    }
}
