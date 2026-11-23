// Unified generic CPU Linear (GEMV + L1-packed BLIS outer-product GEMM) in Slang.
// ONE parameterized source compiles across baseline, AVX2, and AVX-512 via
// compiler flags and across F32, F16->F32, BF16->F32, F32xBF16, F64, and W8A8->F32.
module linear_cpu;

import numeric;
import epilogue;

#ifndef VELRI_KC
#define VELRI_KC 128
#endif
#define VELRI_MC 264
static const uint kKC = uint(VELRI_KC);
static const uint kMC = uint(VELRI_MC);
static const uint kGemvRowThreshold = 4u;

// ============================================================================
// 1. Streaming GEMV (rows < 4): 4-col register blocking, zero packing overhead
// ============================================================================
void gemv_4rows<TX : IElement, TW : IElement, TY : IElement, Ep : IEpilogue>(
    Ptr<TX.Storage, Access.Read> xr,
    Ptr<TW.Storage, Access.Read> w, uint w_stride,
    Ptr<TY.Storage> yr,
    uint row, uint col, uint k, EpilogueParams ep)
{
    uint64_t ws = uint64_t(w_stride);
    Ptr<TW.Storage, Access.Read> w0 = w + uint64_t(col + 0u) * ws;
    Ptr<TW.Storage, Access.Read> w1 = w + uint64_t(col + 1u) * ws;
    Ptr<TW.Storage, Access.Read> w2 = w + uint64_t(col + 2u) * ws;
    Ptr<TW.Storage, Access.Read> w3 = w + uint64_t(col + 3u) * ws;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (uint i = 0u; i < k; ++i) {
        float xv = TX::toF32(xr[i]);
        s0 += xv * TW::toF32(w0[i]);
        s1 += xv * TW::toF32(w1[i]);
        s2 += xv * TW::toF32(w2[i]);
        s3 += xv * TW::toF32(w3[i]);
    }
    yr[col + 0u] = TY::fromF32(Ep::apply(s0, row, col + 0u, ep));
    yr[col + 1u] = TY::fromF32(Ep::apply(s1, row, col + 1u, ep));
    yr[col + 2u] = TY::fromF32(Ep::apply(s2, row, col + 2u, ep));
    yr[col + 3u] = TY::fromF32(Ep::apply(s3, row, col + 3u, ep));
}

void gemv_1row<TX : IElement, TW : IElement, TY : IElement, Ep : IEpilogue>(
    Ptr<TX.Storage, Access.Read> xr,
    Ptr<TW.Storage, Access.Read> wc,
    Ptr<TY.Storage> yr,
    uint row, uint col, uint k, EpilogueParams ep)
{
    float s = 0.0f;
    for (uint i = 0u; i < k; ++i) s += TX::toF32(xr[i]) * TW::toF32(wc[i]);
    yr[col] = TY::fromF32(Ep::apply(s, row, col, ep));
}

// ============================================================================
// 2. Scatter-Free & Gather-Free Packers
// ============================================================================
void pack_b_f32<T : IElement, let NR : int>(
    Ptr<T.Storage, Access.Read> src, uint stride, uint col0, uint nr,
    uint k0, uint kc, Ptr<float4> dst4)
{
    static const uint W4 = uint(NR) >> 2u;
    uint64_t s64 = uint64_t(stride);
    uint64_t k64 = uint64_t(k0);
    Ptr<float> dst = Ptr<float>(dst4);
    uint j = 0u;
    for (; j + 4u <= nr; j += 4u) {
        Ptr<T.Storage, Access.Read> s0 = src + uint64_t(col0 + j + 0u) * s64 + k64;
        Ptr<T.Storage, Access.Read> s1 = src + uint64_t(col0 + j + 1u) * s64 + k64;
        Ptr<T.Storage, Access.Read> s2 = src + uint64_t(col0 + j + 2u) * s64 + k64;
        Ptr<T.Storage, Access.Read> s3 = src + uint64_t(col0 + j + 3u) * s64 + k64;
        uint j4 = j >> 2u;
        uint p = 0u;
        for (; p + 4u <= kc; p += 4u) {
            float4 r0 = T::load4F32(s0 + p);
            float4 r1 = T::load4F32(s1 + p);
            float4 r2 = T::load4F32(s2 + p);
            float4 r3 = T::load4F32(s3 + p);
            dst4[(p + 0u) * W4 + j4] = float4(r0.x, r1.x, r2.x, r3.x);
            dst4[(p + 1u) * W4 + j4] = float4(r0.y, r1.y, r2.y, r3.y);
            dst4[(p + 2u) * W4 + j4] = float4(r0.z, r1.z, r2.z, r3.z);
            dst4[(p + 3u) * W4 + j4] = float4(r0.w, r1.w, r2.w, r3.w);
        }
        for (; p < kc; ++p) {
            dst4[p * W4 + j4] = float4(T::toF32(s0[p]), T::toF32(s1[p]),
                                       T::toF32(s2[p]), T::toF32(s3[p]));
        }
    }
    for (; j < nr; ++j) {
        Ptr<T.Storage, Access.Read> sj = src + uint64_t(col0 + j) * s64 + k64;
        for (uint p = 0u; p < kc; ++p) dst[p * uint(NR) + j] = T::toF32(sj[p]);
    }
    if (nr < uint(NR)) {
        for (uint p = 0u; p < kc; ++p) {
            for (uint jj = nr; jj < uint(NR); ++jj) dst[p * uint(NR) + jj] = 0.0f;
        }
    }
}

void pack_a_f32<T : IElement, let MR : int>(
    Ptr<T.Storage, Access.Read> src, uint stride, uint row0, uint mr,
    uint k0, uint kc, Ptr<float> dst)
{
    uint64_t s64 = uint64_t(stride);
    uint64_t k64 = uint64_t(k0);
    for (uint r = 0u; r < mr; ++r) {
        Ptr<T.Storage, Access.Read> sr = src + uint64_t(row0 + r) * s64 + k64;
        Ptr<float> dr = dst + r * kKC;
        for (uint p = 0u; p < kc; ++p) dr[p] = T::toF32(sr[p]);
    }
    if (mr < uint(MR)) {
        for (uint r = mr; r < uint(MR); ++r) {
            Ptr<float> dr = dst + r * kKC;
            for (uint p = 0u; p < kc; ++p) dr[p] = 0.0f;
        }
    }
}

void pack_b_f64<let NR : int>(
    Ptr<double, Access.Read> src, uint stride, uint col0, uint nr,
    uint k0, uint kc, Ptr<double4> dst4)
{
    static const uint W4 = uint(NR) >> 2u;
    uint64_t s64 = uint64_t(stride);
    uint64_t k64 = uint64_t(k0);
    Ptr<double> dst = Ptr<double>(dst4);
    uint j = 0u;
    for (; j + 4u <= nr; j += 4u) {
        Ptr<double, Access.Read> s0 = src + uint64_t(col0 + j + 0u) * s64 + k64;
        Ptr<double, Access.Read> s1 = src + uint64_t(col0 + j + 1u) * s64 + k64;
        Ptr<double, Access.Read> s2 = src + uint64_t(col0 + j + 2u) * s64 + k64;
        Ptr<double, Access.Read> s3 = src + uint64_t(col0 + j + 3u) * s64 + k64;
        uint j4 = j >> 2u;
        uint p = 0u;
        for (; p + 4u <= kc; p += 4u) {
            double4 r0 = double4(s0[p], s0[p+1u], s0[p+2u], s0[p+3u]);
            double4 r1 = double4(s1[p], s1[p+1u], s1[p+2u], s1[p+3u]);
            double4 r2 = double4(s2[p], s2[p+1u], s2[p+2u], s2[p+3u]);
            double4 r3 = double4(s3[p], s3[p+1u], s3[p+2u], s3[p+3u]);
            dst4[(p + 0u) * W4 + j4] = double4(r0.x, r1.x, r2.x, r3.x);
            dst4[(p + 1u) * W4 + j4] = double4(r0.y, r1.y, r2.y, r3.y);
            dst4[(p + 2u) * W4 + j4] = double4(r0.z, r1.z, r2.z, r3.z);
            dst4[(p + 3u) * W4 + j4] = double4(r0.w, r1.w, r2.w, r3.w);
        }
        for (; p < kc; ++p) {
            dst4[p * W4 + j4] = double4(s0[p], s1[p], s2[p], s3[p]);
        }
    }
    for (; j < nr; ++j) {
        Ptr<double, Access.Read> sj = src + uint64_t(col0 + j) * s64 + k64;
        for (uint p = 0u; p < kc; ++p) dst[p * uint(NR) + j] = sj[p];
    }
    if (nr < uint(NR)) {
        for (uint p = 0u; p < kc; ++p) {
            for (uint jj = nr; jj < uint(NR); ++jj) dst[p * uint(NR) + jj] = 0.0;
        }
    }
}

void pack_a_f64<let MR : int>(
    Ptr<double, Access.Read> src, uint stride, uint row0, uint mr,
    uint k0, uint kc, Ptr<double> dst)
{
    uint64_t s64 = uint64_t(stride);
    uint64_t k64 = uint64_t(k0);
    for (uint r = 0u; r < mr; ++r) {
        Ptr<double, Access.Read> sr = src + uint64_t(row0 + r) * s64 + k64;
        Ptr<double> dr = dst + r * kKC;
        for (uint p = 0u; p < kc; ++p) dr[p] = sr[p];
    }
    if (mr < uint(MR)) {
        for (uint r = mr; r < uint(MR); ++r) {
            Ptr<double> dr = dst + r * kKC;
            for (uint p = 0u; p < kc; ++p) dr[p] = 0.0;
        }
    }
}

// ============================================================================
// 3. Unified Zero-Spill L1-Packed BLIS Outer-Product Microkernels
// ============================================================================
void gemm_mk_f32<let MR : int, let NR : int, let ZERO_INIT : bool>(
    Ptr<float, Access.Read> ap, uint lda, Ptr<float> bp,
    Ptr<float> c_tile, uint ldc, uint kc)
{
    uint64_t lda64 = uint64_t(lda);
    float c[MR * NR];
    if (ZERO_INIT) {
        for (uint i = 0u; i < uint(MR * NR); ++i) c[i] = 0.0f;
    } else {
        for (uint r = 0u; r < uint(MR); ++r) {
            Ptr<float> cr = c_tile + r * ldc;
            for (uint j = 0u; j < uint(NR); ++j) c[r * uint(NR) + j] = cr[j];
        }
    }
    for (uint p = 0u; p < kc; ++p) {
        Ptr<float> brow = bp + p * uint(NR);
        float a[MR];
        [ForceUnroll] for (uint r = 0u; r < uint(MR); ++r) a[r] = ap[uint64_t(r) * lda64 + uint64_t(p)];
        for (uint j = 0u; j < uint(NR); ++j) {
            float b = brow[j];
            [ForceUnroll] for (uint r = 0u; r < uint(MR); ++r) {
                c[r * uint(NR) + j] += a[r] * b;
            }
        }
    }
    for (uint r = 0u; r < uint(MR); ++r) {
        Ptr<float> cr = c_tile + r * ldc;
        for (uint j = 0u; j < uint(NR); ++j) cr[j] = c[r * uint(NR) + j];
    }
}

void gemm_mk_f64<let MR : int, let NR : int, let ZERO_INIT : bool>(
    Ptr<double, Access.Read> ap, uint lda, Ptr<double> bp,
    Ptr<double> c_tile, uint ldc, uint kc)
{
    uint64_t lda64 = uint64_t(lda);
    if (NR >= 32) {
        double c[MR * NR];
        if (ZERO_INIT) {
            for (uint i = 0u; i < uint(MR * NR); ++i) c[i] = 0.0;
        } else {
            for (uint r = 0u; r < uint(MR); ++r) {
                Ptr<double> cr = c_tile + r * ldc;
                for (uint j = 0u; j < uint(NR); ++j) c[r * uint(NR) + j] = cr[j];
            }
        }
        for (uint p = 0u; p < kc; ++p) {
            Ptr<double> brow = bp + p * uint(NR);
            double a[MR];
            [ForceUnroll] for (uint r = 0u; r < uint(MR); ++r) a[r] = ap[uint64_t(r) * lda64 + uint64_t(p)];
            for (uint j = 0u; j < uint(NR); ++j) {
                double b = brow[j];
                [ForceUnroll] for (uint r = 0u; r < uint(MR); ++r) {
                    c[r * uint(NR) + j] += a[r] * b;
                }
            }
        }
        for (uint r = 0u; r < uint(MR); ++r) {
            Ptr<double> cr = c_tile + r * ldc;
            for (uint j = 0u; j < uint(NR); ++j) cr[j] = c[r * uint(NR) + j];
        }
    } else {
        static const uint W4 = uint(NR) >> 2u;
        double4 c[MR * (NR / 4)];
        if (ZERO_INIT) {
            [ForceUnroll] for (uint i = 0u; i < uint(MR) * W4; ++i) c[i] = double4(0.0);
        } else {
            [ForceUnroll] for (uint r = 0u; r < uint(MR); ++r) {
                Ptr<double> cr = c_tile + r * ldc;
                [ForceUnroll] for (uint j = 0u; j < W4; ++j)
                    c[r * W4 + j] = double4(cr[j*4u], cr[j*4u+1u], cr[j*4u+2u], cr[j*4u+3u]);
            }
        }
        Ptr<double4> bp4 = Ptr<double4>(bp);
        for (uint p = 0u; p < kc; ++p) {
            Ptr<double4> brow4 = bp4 + p * W4;
            double a[MR];
            [ForceUnroll] for (uint r = 0u; r < uint(MR); ++r) a[r] = ap[uint64_t(r) * lda64 + uint64_t(p)];
            [ForceUnroll] for (uint j = 0u; j < W4; ++j) {
                double4 b = brow4[j];
                [ForceUnroll] for (uint r = 0u; r < uint(MR); ++r) {
                    c[r * W4 + j] += a[r] * b;
                }
            }
        }
        [ForceUnroll] for (uint r = 0u; r < uint(MR); ++r) {
            Ptr<double> cr = c_tile + r * ldc;
            [ForceUnroll] for (uint j = 0u; j < W4; ++j) {
                double4 v = c[r * W4 + j];
                cr[j*4u+0u] = v.x; cr[j*4u+1u] = v.y; cr[j*4u+2u] = v.z; cr[j*4u+3u] = v.w;
            }
        }
    }
}

void linear_mc_panel_f32<let MR : int, let NR : int, TX : IElement, let ZERO_INIT : bool>(
    Ptr<TX.Storage, Access.Read> x, uint x_stride,
    Ptr<float> ap, Ptr<float, Access.Read> ap_rd, Ptr<float> bp, Ptr<float> c_buf,
    uint rb, uint mc, uint k0, uint kc)
{
    for (uint rm = 0u; rm < mc; rm += uint(MR)) {
        uint mr = min(uint(MR), mc - rm);
        if (TX.kIsNativeF32 && mr == uint(MR)) {
            Ptr<float, Access.Read> xp =
                Ptr<float, Access.Read>(x) + uint64_t(rb + rm) * uint64_t(x_stride) + uint64_t(k0);
            gemm_mk_f32<MR, NR, ZERO_INIT>(xp, x_stride, bp, c_buf + rm * uint(NR), uint(NR), kc);
        } else {
            pack_a_f32<TX, MR>(x, x_stride, rb + rm, mr, k0, kc, ap);
            gemm_mk_f32<MR, NR, ZERO_INIT>(ap_rd, kKC, bp, c_buf + rm * uint(NR), uint(NR), kc);
        }
    }
}

void linear_mc_panel_f64<let MR : int, let NR : int, let ZERO_INIT : bool>(
    Ptr<double, Access.Read> x, uint x_stride,
    Ptr<double> ap, Ptr<double, Access.Read> ap_rd, Ptr<double> bp, Ptr<double> c_buf,
    uint rb, uint mc, uint k0, uint kc)
{
    for (uint rm = 0u; rm < mc; rm += uint(MR)) {
        uint mr = min(uint(MR), mc - rm);
        if (mr == uint(MR)) {
            Ptr<double, Access.Read> xp = x + uint64_t(rb + rm) * uint64_t(x_stride) + uint64_t(k0);
            gemm_mk_f64<MR, NR, ZERO_INIT>(xp, x_stride, bp, c_buf + rm * uint(NR), uint(NR), kc);
        } else {
            pack_a_f64<MR>(x, x_stride, rb + rm, mr, k0, kc, ap);
            gemm_mk_f64<MR, NR, ZERO_INIT>(ap_rd, kKC, bp, c_buf + rm * uint(NR), uint(NR), kc);
        }
    }
}

// ============================================================================
// 4. Unified Floating-Point Linear (F32, F16->F32, BF16->F32, F32xBF16)
// ============================================================================
public void linear_slice_cpu<let MR : int, let NR : int,
                             TX : IElement, TW : IElement, TY : IElement, Ep : IEpilogue>(
    Ptr<TX.Storage, Access.Read> x, uint x_stride,
    Ptr<TW.Storage, Access.Read> w, uint w_stride,
    Ptr<TY.Storage> y, uint y_stride,
    uint r0, uint r1, uint c0, uint c1, uint k,
    EpilogueParams ep)
{
    uint nrows = r1 - r0;
    if (nrows < kGemvRowThreshold) {
        for (uint r = r0; r < r1; ++r) {
            Ptr<TX.Storage, Access.Read> xr = x + uint64_t(r) * uint64_t(x_stride);
            Ptr<TY.Storage> yr = y + uint64_t(r) * uint64_t(y_stride);
            uint c = c0;
            for (; c + 4u <= c1; c += 4u) {
                gemv_4rows<TX, TW, TY, Ep>(xr, w, w_stride, yr, r, c, k, ep);
            }
            for (; c < c1; ++c) {
                gemv_1row<TX, TW, TY, Ep>(xr, w + uint64_t(c) * uint64_t(w_stride), yr, r, c, k, ep);
            }
        }
        return;
    }

    float4 bp4[VELRI_KC * (NR / 4)];
    float ap[MR * VELRI_KC];
    float c_buf[VELRI_MC * NR];
    bp4[0] = float4(0.0f); ap[0] = 0.0f; c_buf[0] = 0.0f;
    Ptr<float> bp = Ptr<float>(&bp4[0]);
    Ptr<float> ap_wr = &ap[0];
    Ptr<float, Access.Read> ap_rd = Ptr<float, Access.Read>(&ap[0]);
    Ptr<float> cb_ptr = &c_buf[0];

    for (uint cb = c0; cb < c1; cb += uint(NR)) {
        uint nr = min(uint(NR), c1 - cb);
        for (uint rb = r0; rb < r1; rb += kMC) {
            uint mc = min(kMC, r1 - rb);
            for (uint k0 = 0u; k0 < k; k0 += kKC) {
                uint kc = min(kKC, k - k0);
                pack_b_f32<TW, NR>(w, w_stride, cb, nr, k0, kc, &bp4[0]);
                if (k0 == 0u) {
                    linear_mc_panel_f32<MR, NR, TX, true>(x, x_stride, ap_wr, ap_rd, bp, cb_ptr, rb, mc, k0, kc);
                } else {
                    linear_mc_panel_f32<MR, NR, TX, false>(x, x_stride, ap_wr, ap_rd, bp, cb_ptr, rb, mc, k0, kc);
                }
            }
            for (uint rm = 0u; rm < mc; ++rm) {
                uint rr = rb + rm;
                Ptr<TY.Storage> yr = y + uint64_t(rr) * uint64_t(y_stride) + uint64_t(cb);
                for (uint j = 0u; j < nr; ++j) {
                    yr[j] = TY::fromF32(Ep::apply(c_buf[rm * uint(NR) + j], rr, cb + j, ep));
                }
            }
        }
    }
}

// ============================================================================
// 5. Exact INT8 * INT8 -> INT32 (modulo 2^32) & W8A8 -> F32
// ============================================================================
// Packed weights store signed bytes in [channel panel][depth][channel] order.
// Each panel is 32 columns wide, including the zero-padded last panel.
static const uint kW8A8PanelWidth = 32u;

public void pack_w8a8_cpu(Ptr<int8_t, Access.Read> w, uint stride,
                          uint channels, uint depth, Ptr<int8_t> packed)
{
    for (uint64_t col = 0; col < channels; col += kW8A8PanelWidth) {
        uint width = min(kW8A8PanelWidth, channels - uint(col));
        Ptr<int8_t> panel = packed + col * depth;
        for (uint k = 0; k < depth; ++k) {
            Ptr<int8_t> dst = panel + uint64_t(k) * kW8A8PanelWidth;
            for (uint j = 0; j < width; ++j) dst[j] = w[(col + j) * stride + k];
            for (uint j = width; j < kW8A8PanelWidth; ++j) dst[j] = int8_t(0);
        }
    }
}

void w8a8_gemv_columns<let COLS : int>(
    Ptr<int8_t, Access.Read> x, Ptr<int8_t, Access.Read> w, uint stride,
    Ptr<float> y, uint row, uint col, uint k, EpilogueParams ep)
{
    Ptr<int8_t, Access.Read> columns[COLS];
    uint sums[COLS];
    [ForceUnroll] for (uint j = 0; j < uint(COLS); ++j) {
        columns[j] = w + uint64_t(col + j) * stride;
        sums[j] = 0;
    }
    for (uint i = 0; i < k; ++i) {
        int a = int(x[i]);
        [ForceUnroll] for (uint j = 0; j < uint(COLS); ++j)
            sums[j] += uint(a * int(columns[j][i]));
    }
    [ForceUnroll] for (uint j = 0; j < uint(COLS); ++j)
        y[col + j] = EpQuantScale::apply(float(int32_t(sums[j])), row, col + j, ep);
}

// Signed W32 panels store the weights of 32 adjacent channels at each K.
// Worker boundaries may cut either end of a panel. Only owned channels are stored.
void packed_w8a8_gemv(
    Ptr<int8_t, Access.Read> x, uint xs, Ptr<int8_t, Access.Read> w,
    Ptr<float> y, uint ys, uint r0, uint r1, uint c0, uint c1, uint k, EpilogueParams ep)
{
    float neutral[kW8A8PanelWidth] = {};
    for (uint r = r0; r < r1; ++r) {
        Ptr<int8_t, Access.Read> xr = x + uint64_t(r) * xs;
        Ptr<float> yr = y + uint64_t(r) * ys;
        for (uint64_t start = uint64_t(c0 / kW8A8PanelWidth) * kW8A8PanelWidth; start < c1; start += kW8A8PanelWidth) {
            Ptr<int8_t, Access.Read> wp = w + start * k;
            uint sums[kW8A8PanelWidth];
            for (uint c = 0; c < kW8A8PanelWidth; ++c) sums[c] = 0;
            for (uint p = 0; p < k; ++p) {
                int a = int(xr[p]);
                for (uint c = 0; c < kW8A8PanelWidth; ++c)
                    sums[c] += uint(a * int(wp[c]));
                wp = wp + kW8A8PanelWidth;
            }
            uint lo = max(c0, uint(start)) - uint(start);
            uint hi = min(kW8A8PanelWidth, c1 - uint(start));
            EpilogueParams local = ep;
            local.bias = ep.bias != nullptr ? ep.bias + start : Ptr<float, Access.Read>(&neutral[0]);
            local.w_scale = ep.w_scale != nullptr ? ep.w_scale + start : Ptr<float, Access.Read>(&neutral[0]);
            for (uint c = lo; c < hi; ++c) {
                uint col = uint(start) + c;
                yr[col] = EpQuantScale::apply(float(int32_t(sums[c])), r, c, local);
            }
        }
    }
}

// Gather adjacent 32-channel storage panels into one NR-column compute tile.
void unpack_w8a8_panel<let NR : int>(
    Ptr<int8_t, Access.Read> weights, uint stride, uint col, uint width,
    uint k0, uint depth, Ptr<float> dst)
{
    for (uint j = 0; j < width;) {
        uint c = col + j;
        uint span = min(width - j, kW8A8PanelWidth - c % kW8A8PanelWidth);
        uint lane = c % kW8A8PanelWidth;
        Ptr<int8_t, Access.Read> src = weights + uint64_t(c - lane) * stride + lane
                                       + uint64_t(k0) * kW8A8PanelWidth;
        Ptr<float> outp = dst + j;
        for (uint p = 0; p < depth; ++p) {
            uint i = 0;
            for (; i + 4 <= span; i += 4) {
                float4 v = ElemI8::load4F32(src + i);
                [ForceUnroll] for (uint lane = 0; lane < 4; ++lane) outp[i + lane] = v[lane];
            }
            for (; i < span; ++i) outp[i] = ElemI8::toF32(src[i]);
            src = src + kW8A8PanelWidth;
            outp = outp + NR;
        }
        j += span;
    }
    if (width < uint(NR)) {
        for (uint p = 0; p < depth; ++p)
            for (uint j = width; j < uint(NR); ++j) dst[p * uint(NR) + j] = 0.0f;
    }
}

public void linear_w8a8_f32_slice_cpu<let MR : int, let NR : int, let PACKED : bool>(
    Ptr<int8_t, Access.Read> x, uint x_stride,
    Ptr<int8_t, Access.Read> w, uint w_stride,
    Ptr<float> y, uint y_stride,
    uint r0, uint r1, uint c0, uint c1, uint k,
    EpilogueParams ep)
{
    static_assert(kKC > 0 && kKC <= 1024, "Each INT8 K block must sum exactly in F32");
    if (r0 >= r1 || c0 >= c1) return;
    if (r1 - r0 < kGemvRowThreshold) {
        if (PACKED) {
            packed_w8a8_gemv(x, x_stride, w, y, y_stride, r0, r1, c0, c1, k, ep);
            return;
        }
        for (uint r = r0; r < r1; ++r) {
            Ptr<int8_t, Access.Read> xr = x + uint64_t(r) * x_stride;
            Ptr<float> yr = y + uint64_t(r) * y_stride;
            uint c = c0;
            for (; c1 - c >= 4; c += 4)
                w8a8_gemv_columns<4>(xr, w, w_stride, yr, r, c, k, ep);
            for (; c < c1; ++c)
                w8a8_gemv_columns<1>(xr, w, w_stride, yr, r, c, k, ep);
        }
        return;
    }

    float4 bp4[VELRI_KC * (NR / 4)];
    float ap[MR * VELRI_KC];
    float c_tmp[MR * NR];
    uint32_t c_i32[VELRI_MC * NR];
    bp4[0] = float4(0.0f); ap[0] = 0.0f; c_tmp[0] = 0.0f; c_i32[0] = 0u;
    Ptr<float> bp = Ptr<float>(&bp4[0]);

    for (uint cb = c0; cb < c1;) {
        uint nr = min(uint(NR), c1 - cb);
        for (uint rb = r0; rb < r1;) {
            uint mc = min(kMC, r1 - rb);
            for (uint k0 = 0; k0 < k;) {
                uint kc = min(kKC, k - k0);
                if (PACKED) {
                    unpack_w8a8_panel<NR>(w, w_stride, cb, nr, k0, kc, bp);
                } else {
                    pack_b_f32<ElemI8, NR>(w, w_stride, cb, nr, k0, kc, &bp4[0]);
                }
                for (uint rm = 0u; rm < mc; rm += uint(MR)) {
                    uint mr = min(uint(MR), mc - rm);
                    pack_a_f32<ElemI8, MR>(x, x_stride, rb + rm, mr, k0, kc, &ap[0]);
                    gemm_mk_f32<MR, NR, true>(Ptr<float, Access.Read>(&ap[0]), kKC, bp, &c_tmp[0], uint(NR), kc);
                    for (uint r = 0; r < mr; ++r) {
                        for (uint j = 0; j < nr; ++j) {
                            uint dst = (rm + r) * uint(NR) + j;
                            uint partial = uint(int32_t(c_tmp[r * uint(NR) + j]));
                            c_i32[dst] = k0 == 0 ? partial : c_i32[dst] + partial;
                        }
                    }
                }
                k0 += kc;
            }
            for (uint rm = 0u; rm < mc; ++rm) {
                uint rr = rb + rm;
                Ptr<float> yr = y + uint64_t(rr) * uint64_t(y_stride) + uint64_t(cb);
                for (uint j = 0u; j < nr; ++j) {
                    uint sum = k == 0 ? 0u : c_i32[rm * uint(NR) + j];
                    yr[j] = EpQuantScale::apply(float(int32_t(sum)), rr, cb + j, ep);
                }
            }
            rb += mc;
        }
        cb += nr;
    }
}

// ============================================================================
// 6. Native FP64 Linear (Streaming GEMV + Scatter-Free L1-Packed BLIS GEMM)
// ============================================================================
public void linear_f64_slice_cpu<let MR : int, let NR : int>(
    Ptr<double, Access.Read> x, uint x_stride,
    Ptr<double, Access.Read> w, uint w_stride,
    Ptr<double, Access.Read> bias, uint has_bias,
    Ptr<double> y, uint y_stride,
    uint r0, uint r1, uint c0, uint c1, uint k)
{
    uint nrows = r1 - r0;
    uint64_t ws64 = uint64_t(w_stride);
    if (nrows < kGemvRowThreshold) {
        for (uint r = r0; r < r1; ++r) {
            Ptr<double, Access.Read> xr = x + uint64_t(r) * uint64_t(x_stride);
            Ptr<double> yr = y + uint64_t(r) * uint64_t(y_stride);
            uint c = c0;
            for (; c + 4u <= c1; c += 4u) {
                Ptr<double, Access.Read> w0 = w + uint64_t(c + 0u) * ws64;
                Ptr<double, Access.Read> w1 = w + uint64_t(c + 1u) * ws64;
                Ptr<double, Access.Read> w2 = w + uint64_t(c + 2u) * ws64;
                Ptr<double, Access.Read> w3 = w + uint64_t(c + 3u) * ws64;
                double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                for (uint i = 0u; i < k; ++i) {
                    double xv = xr[i];
                    s0 += xv * w0[i]; s1 += xv * w1[i]; s2 += xv * w2[i]; s3 += xv * w3[i];
                }
                yr[c + 0u] = s0 + (has_bias != 0u ? bias[c + 0u] : 0.0);
                yr[c + 1u] = s1 + (has_bias != 0u ? bias[c + 1u] : 0.0);
                yr[c + 2u] = s2 + (has_bias != 0u ? bias[c + 2u] : 0.0);
                yr[c + 3u] = s3 + (has_bias != 0u ? bias[c + 3u] : 0.0);
            }
            for (; c < c1; ++c) {
                Ptr<double, Access.Read> wc = w + uint64_t(c) * ws64;
                double s = 0.0;
                for (uint i = 0u; i < k; ++i) s += xr[i] * wc[i];
                yr[c] = s + (has_bias != 0u ? bias[c] : 0.0);
            }
        }
        return;
    }

    double4 bp4[VELRI_KC * (NR / 4)];
    double ap[MR * VELRI_KC];
    double c_buf[VELRI_MC * NR];
    bp4[0] = double4(0.0); ap[0] = 0.0; c_buf[0] = 0.0;
    Ptr<double> bp = Ptr<double>(&bp4[0]);
    Ptr<double> ap_wr = &ap[0];
    Ptr<double, Access.Read> ap_rd = Ptr<double, Access.Read>(&ap[0]);
    Ptr<double> cb_ptr = &c_buf[0];

    for (uint cb = c0; cb < c1; cb += uint(NR)) {
        uint nr = min(uint(NR), c1 - cb);
        for (uint rb = r0; rb < r1; rb += kMC) {
            uint mc = min(kMC, r1 - rb);
            for (uint k0 = 0u; k0 < k; k0 += kKC) {
                uint kc = min(kKC, k - k0);
                pack_b_f64<NR>(w, w_stride, cb, nr, k0, kc, &bp4[0]);
                if (k0 == 0u) {
                    linear_mc_panel_f64<MR, NR, true>(x, x_stride, ap_wr, ap_rd, bp, cb_ptr, rb, mc, k0, kc);
                } else {
                    linear_mc_panel_f64<MR, NR, false>(x, x_stride, ap_wr, ap_rd, bp, cb_ptr, rb, mc, k0, kc);
                }
            }
            for (uint rm = 0u; rm < mc; ++rm) {
                uint rr = rb + rm;
                Ptr<double> yr = y + uint64_t(rr) * uint64_t(y_stride) + uint64_t(cb);
                for (uint j = 0u; j < nr; ++j) {
                    yr[j] = c_buf[rm * uint(NR) + j] + (has_bias != 0u ? bias[cb + j] : 0.0);
                }
            }
        }
    }
}
