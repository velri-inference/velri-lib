// Shared CPU Grouped Batched Linear kernel (Y = X @ W^T + b and Y = X @ W + b).
module batched_linear;

public void batched_linear_rows_cpu(
    Ptr<float, Access.Read> x, uint x_stride,
    Ptr<float, Access.Read> w, uint w_stride,
    Ptr<float, Access.Read> bias, uint has_bias,
    Ptr<float> y, uint y_stride,
    uint w0, uint w1, uint n_heads, uint n_groups,
    uint in_dim, uint out_dim, uint weight_transposed)
{
    if (n_heads == 0u || n_groups == 0u || in_dim == 0u || out_dim == 0u) {
        return;
    }

    uint heads_per_group = n_heads / n_groups;
    uint64_t xs64 = uint64_t(x_stride);
    uint64_t ws64 = uint64_t(w_stride);
    uint64_t ys64 = uint64_t(y_stride);
    uint64_t k64 = uint64_t(in_dim);
    uint64_t m64 = uint64_t(out_dim);
    uint out_dim4 = out_dim & ~3u;
    uint in_dim4 = in_dim & ~3u;

    for (uint work = w0; work < w1; ++work) {
        uint t = work / n_heads;
        uint h = work - t * n_heads;
        uint g = h / heads_per_group;

        Ptr<float, Access.Read> x_ptr = x + uint64_t(t) * xs64 + uint64_t(h) * k64;
        Ptr<float> y_ptr = y + uint64_t(t) * ys64 + uint64_t(h) * m64;
        uint64_t b_base = uint64_t(g) * m64;

        if (weight_transposed == 0u) {
            // Y = X @ W^T + bias, W is [G * out_dim, in_dim]
            Ptr<float, Access.Read> w_grp = w + (uint64_t(g) * m64) * ws64;
            uint m = 0u;
            for (; m < out_dim4; m += 4u) {
                Ptr<float, Access.Read> w0p = w_grp + uint64_t(m + 0u) * ws64;
                Ptr<float, Access.Read> w1p = w_grp + uint64_t(m + 1u) * ws64;
                Ptr<float, Access.Read> w2p = w_grp + uint64_t(m + 2u) * ws64;
                Ptr<float, Access.Read> w3p = w_grp + uint64_t(m + 3u) * ws64;
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                for (uint k = 0u; k < in_dim; ++k) {
                    float xv = x_ptr[k];
                    a0 += xv * w0p[k];
                    a1 += xv * w1p[k];
                    a2 += xv * w2p[k];
                    a3 += xv * w3p[k];
                }
                if (has_bias != 0u) {
                    a0 += bias[b_base + uint64_t(m + 0u)];
                    a1 += bias[b_base + uint64_t(m + 1u)];
                    a2 += bias[b_base + uint64_t(m + 2u)];
                    a3 += bias[b_base + uint64_t(m + 3u)];
                }
                y_ptr[m + 0u] = a0;
                y_ptr[m + 1u] = a1;
                y_ptr[m + 2u] = a2;
                y_ptr[m + 3u] = a3;
            }
            for (; m < out_dim; ++m) {
                Ptr<float, Access.Read> wm = w_grp + uint64_t(m) * ws64;
                float acc = 0.0f;
                for (uint k = 0u; k < in_dim; ++k) {
                    acc += x_ptr[k] * wm[k];
                }
                if (has_bias != 0u) {
                    acc += bias[b_base + uint64_t(m)];
                }
                y_ptr[m] = acc;
            }
        } else {
            // Y = X @ W + bias, W is [G * in_dim, out_dim] (unit-stride AXPY across out_dim)
            if (has_bias != 0u) {
                for (uint m = 0u; m < out_dim; ++m) {
                    y_ptr[m] = bias[b_base + uint64_t(m)];
                }
            } else {
                for (uint m = 0u; m < out_dim; ++m) {
                    y_ptr[m] = 0.0f;
                }
            }
            Ptr<float, Access.Read> w_grp = w + (uint64_t(g) * k64) * ws64;
            uint k = 0u;
            for (; k < in_dim4; k += 4u) {
                float x0 = x_ptr[k + 0u];
                float x1 = x_ptr[k + 1u];
                float x2 = x_ptr[k + 2u];
                float x3 = x_ptr[k + 3u];
                Ptr<float, Access.Read> wk0 = w_grp + uint64_t(k + 0u) * ws64;
                Ptr<float, Access.Read> wk1 = w_grp + uint64_t(k + 1u) * ws64;
                Ptr<float, Access.Read> wk2 = w_grp + uint64_t(k + 2u) * ws64;
                Ptr<float, Access.Read> wk3 = w_grp + uint64_t(k + 3u) * ws64;
                for (uint m = 0u; m < out_dim; ++m) {
                    float a = y_ptr[m];
                    a += x0 * wk0[m];
                    a += x1 * wk1[m];
                    a += x2 * wk2[m];
                    a += x3 * wk3[m];
                    y_ptr[m] = a;
                }
            }
            for (; k < in_dim; ++k) {
                float xk = x_ptr[k];
                Ptr<float, Access.Read> wkp = w_grp + uint64_t(k) * ws64;
                for (uint m = 0u; m < out_dim; ++m) {
                    y_ptr[m] += xk * wkp[m];
                }
            }
        }
    }
}
