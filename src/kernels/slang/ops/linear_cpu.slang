// Unified generic CPU Linear (GEMV + L1-packed BLIS outer-product GEMM) in Slang.
// ONE parameterized source compiles across baseline, AVX2, and AVX-512 via
// compiler flags and across F32, F16->F32, BF16->F32, F32xBF16, F64, and W8A8->F32.
module linear_cpu;

import numeric;
import epilogue;

#ifndef VELRI_KC
#define VELRI_KC 128
#endif
static const uint kKC = uint(VELRI_KC);

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
    if (nrows < 4u) {
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

    static const uint kMC = 264u;
    float4 bp4[VELRI_KC * (NR / 4)];
    float ap[MR * VELRI_KC];
    float c_buf[264 * NR];
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
public void linear_w8a8_f32_slice_cpu<let MR : int, let NR : int>(
    Ptr<int8_t, Access.Read> x, uint x_stride,
    Ptr<int8_t, Access.Read> w, uint w_stride,
    Ptr<float> y, uint y_stride,
    uint r0, uint r1, uint c0, uint c1, uint k,
    EpilogueParams ep)
{
    uint nrows = r1 - r0;
    uint64_t ws64 = uint64_t(w_stride);
    if (nrows < 4u) {
        for (uint r = r0; r < r1; ++r) {
            Ptr<int8_t, Access.Read> xr = x + uint64_t(r) * uint64_t(x_stride);
            Ptr<float> yr = y + uint64_t(r) * uint64_t(y_stride);
            uint c = c0;
            for (; c + 4u <= c1; c += 4u) {
                Ptr<int8_t, Access.Read> w0 = w + uint64_t(c + 0u) * ws64;
                Ptr<int8_t, Access.Read> w1 = w + uint64_t(c + 1u) * ws64;
                Ptr<int8_t, Access.Read> w2 = w + uint64_t(c + 2u) * ws64;
                Ptr<int8_t, Access.Read> w3 = w + uint64_t(c + 3u) * ws64;
                uint32_t s0 = 0u, s1 = 0u, s2 = 0u, s3 = 0u;
                for (uint i = 0u; i < k; ++i) {
                    int16_t xv = int16_t(xr[i]);
                    s0 += uint32_t(int32_t(xv * int16_t(w0[i])));
                    s1 += uint32_t(int32_t(xv * int16_t(w1[i])));
                    s2 += uint32_t(int32_t(xv * int16_t(w2[i])));
                    s3 += uint32_t(int32_t(xv * int16_t(w3[i])));
                }
                yr[c + 0u] = EpQuantScale::apply(float(int32_t(s0)), r, c + 0u, ep);
                yr[c + 1u] = EpQuantScale::apply(float(int32_t(s1)), r, c + 1u, ep);
                yr[c + 2u] = EpQuantScale::apply(float(int32_t(s2)), r, c + 2u, ep);
                yr[c + 3u] = EpQuantScale::apply(float(int32_t(s3)), r, c + 3u, ep);
            }
            for (; c < c1; ++c) {
                Ptr<int8_t, Access.Read> wc = w + uint64_t(c) * ws64;
                uint32_t s = 0u;
                for (uint i = 0u; i < k; ++i) s += uint32_t(int32_t(int16_t(xr[i]) * int16_t(wc[i])));
                yr[c] = EpQuantScale::apply(float(int32_t(s)), r, c, ep);
            }
        }
        return;
    }

    static const uint kMC = 264u;
    float4 bp4[VELRI_KC * (NR / 4)];
    float ap[MR * VELRI_KC];
    float c_tmp[MR * NR];
    uint32_t c_i32[264 * NR];
    bp4[0] = float4(0.0f); ap[0] = 0.0f; c_tmp[0] = 0.0f; c_i32[0] = 0u;
    Ptr<float> bp = Ptr<float>(&bp4[0]);

    for (uint cb = c0; cb < c1; cb += uint(NR)) {
        uint nr = min(uint(NR), c1 - cb);
        for (uint rb = r0; rb < r1; rb += kMC) {
            uint mc = min(kMC, r1 - rb);
            for (uint k0 = 0u; k0 < k; k0 += kKC) {
                uint kc = min(kKC, k - k0);
                pack_b_f32<ElemI8, NR>(w, w_stride, cb, nr, k0, kc, &bp4[0]);
                bool first = (k0 == 0u);
                for (uint rm = 0u; rm < mc; rm += uint(MR)) {
                    uint mr = min(uint(MR), mc - rm);
                    pack_a_f32<ElemI8, MR>(x, x_stride, rb + rm, mr, k0, kc, &ap[0]);
                    gemm_mk_f32<MR, NR, true>(Ptr<float, Access.Read>(&ap[0]), kKC, bp, &c_tmp[0], uint(NR), kc);
                    if (first) {
                        for (uint r = 0u; r < mr; ++r) {
                            uint dst_off = (rm + r) * uint(NR);
                            uint src_off = r * uint(NR);
                            for (uint j = 0u; j < uint(NR); ++j)
                                c_i32[dst_off + j] = uint32_t(int32_t(c_tmp[src_off + j]));
                        }
                    } else {
                        for (uint r = 0u; r < mr; ++r) {
                            uint dst_off = (rm + r) * uint(NR);
                            uint src_off = r * uint(NR);
                            for (uint j = 0u; j < uint(NR); ++j)
                                c_i32[dst_off + j] += uint32_t(int32_t(c_tmp[src_off + j]));
                        }
                    }
                }
            }
            for (uint rm = 0u; rm < mc; ++rm) {
                uint rr = rb + rm;
                Ptr<float> yr = y + uint64_t(rr) * uint64_t(y_stride) + uint64_t(cb);
                for (uint j = 0u; j < nr; ++j) {
                    yr[j] = EpQuantScale::apply(float(int32_t(c_i32[rm * uint(NR) + j])), rr, cb + j, ep);
                }
            }
        }
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
    if (nrows < 4u) {
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

    static const uint kMC = 264u;
    double4 bp4[VELRI_KC * (NR / 4)];
    double ap[MR * VELRI_KC];
    double c_buf[264 * NR];
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
