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
// Reserve +3 rows so MR==6's 4-row tail microkernel stays in-bounds at mc == VELRI_MC - 1.
#define VELRI_MC_BUF_ROWS (VELRI_MC + 3)
#define VELRI_AP_PAD_F32 16
#define VELRI_AP_PAD_F64 8
static const uint kKC = uint(VELRI_KC);
static const uint kMC = uint(VELRI_MC);
static const uint kApStrideF32 = uint(VELRI_KC + VELRI_AP_PAD_F32);
static const uint kApStrideF64 = uint(VELRI_KC + VELRI_AP_PAD_F64);
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
// 2. Scatter-Free & Gather-Free Register-Blocked Packers
// ============================================================================
[ForceInline] void transpose4x4_store<T : __BuiltinFloatingPointType>(
    vector<T, 4> r0, vector<T, 4> r1, vector<T, 4> r2, vector<T, 4> r3,
    Ptr<vector<T, 4>> dst4, uint p, uint W4, uint j4)
{
    dst4[(p + 0u) * W4 + j4] = vector<T, 4>(r0.x, r1.x, r2.x, r3.x);
    dst4[(p + 1u) * W4 + j4] = vector<T, 4>(r0.y, r1.y, r2.y, r3.y);
    dst4[(p + 2u) * W4 + j4] = vector<T, 4>(r0.z, r1.z, r2.z, r3.z);
    dst4[(p + 3u) * W4 + j4] = vector<T, 4>(r0.w, r1.w, r2.w, r3.w);
}

[ForceInline] void transpose4x8_store<T : __BuiltinFloatingPointType>(
    vector<T, 4> a0, vector<T, 4> a1, vector<T, 4> a2, vector<T, 4> a3,
    vector<T, 4> a4, vector<T, 4> a5, vector<T, 4> a6, vector<T, 4> a7,
    Ptr<vector<T, 4>> dst4, uint p, uint W4, uint j4)
{
    dst4[(p + 0u) * W4 + j4]      = vector<T, 4>(a0.x, a1.x, a2.x, a3.x);
    dst4[(p + 0u) * W4 + j4 + 1u] = vector<T, 4>(a4.x, a5.x, a6.x, a7.x);
    dst4[(p + 1u) * W4 + j4]      = vector<T, 4>(a0.y, a1.y, a2.y, a3.y);
    dst4[(p + 1u) * W4 + j4 + 1u] = vector<T, 4>(a4.y, a5.y, a6.y, a7.y);
    dst4[(p + 2u) * W4 + j4]      = vector<T, 4>(a0.z, a1.z, a2.z, a3.z);
    dst4[(p + 2u) * W4 + j4 + 1u] = vector<T, 4>(a4.z, a5.z, a6.z, a7.z);
    dst4[(p + 3u) * W4 + j4]      = vector<T, 4>(a0.w, a1.w, a2.w, a3.w);
    dst4[(p + 3u) * W4 + j4 + 1u] = vector<T, 4>(a4.w, a5.w, a6.w, a7.w);
}

void pack_b_f32<T : IElement, let NR : int, let ALLOW_ALIGNED : bool = true>(
    Ptr<T.Storage, Access.Read> src, uint stride, uint col0, uint nr,
    uint k0, uint kc, Ptr<float4> dst4, bool aligned_src = false)
{
    static const uint W4 = uint(NR) >> 2u;
    uint64_t s64 = uint64_t(stride);
    uint64_t k64 = uint64_t(k0);
    Ptr<float> dst = Ptr<float>(dst4);
    uint j = 0u;
    if (ALLOW_ALIGNED && aligned_src && (k0 & 3u) == 0u && (stride & 3u) == 0u) {
        uint kc8 = kc & ~7u;
        uint kc4 = kc & ~3u;
        for (; nr - j >= 8u; j += 8u) {
            Ptr<T.Storage, Access.Read> s0 = src + uint64_t(col0 + j + 0u) * s64 + k64;
            Ptr<T.Storage, Access.Read> s1 = src + uint64_t(col0 + j + 1u) * s64 + k64;
            Ptr<T.Storage, Access.Read> s2 = src + uint64_t(col0 + j + 2u) * s64 + k64;
            Ptr<T.Storage, Access.Read> s3 = src + uint64_t(col0 + j + 3u) * s64 + k64;
            Ptr<T.Storage, Access.Read> s4 = src + uint64_t(col0 + j + 4u) * s64 + k64;
            Ptr<T.Storage, Access.Read> s5 = src + uint64_t(col0 + j + 5u) * s64 + k64;
            Ptr<T.Storage, Access.Read> s6 = src + uint64_t(col0 + j + 6u) * s64 + k64;
            Ptr<T.Storage, Access.Read> s7 = src + uint64_t(col0 + j + 7u) * s64 + k64;
            uint j4 = j >> 2u;
            uint p = 0u;
            for (; p < kc8; p += 8u) {
                float4 a0 = T::load4F32Aligned(s0 + p), a1 = T::load4F32Aligned(s1 + p);
                float4 a2 = T::load4F32Aligned(s2 + p), a3 = T::load4F32Aligned(s3 + p);
                float4 a4 = T::load4F32Aligned(s4 + p), a5 = T::load4F32Aligned(s5 + p);
                float4 a6 = T::load4F32Aligned(s6 + p), a7 = T::load4F32Aligned(s7 + p);
                float4 b0 = T::load4F32Aligned(s0 + p + 4u), b1 = T::load4F32Aligned(s1 + p + 4u);
                float4 b2 = T::load4F32Aligned(s2 + p + 4u), b3 = T::load4F32Aligned(s3 + p + 4u);
                float4 b4 = T::load4F32Aligned(s4 + p + 4u), b5 = T::load4F32Aligned(s5 + p + 4u);
                float4 b6 = T::load4F32Aligned(s6 + p + 4u), b7 = T::load4F32Aligned(s7 + p + 4u);
                transpose4x8_store<float>(a0, a1, a2, a3, a4, a5, a6, a7, dst4, p, W4, j4);
                transpose4x8_store<float>(b0, b1, b2, b3, b4, b5, b6, b7, dst4, p + 4u, W4, j4);
            }
            for (; p < kc4; p += 4u) {
                transpose4x8_store<float>(
                    T::load4F32Aligned(s0 + p), T::load4F32Aligned(s1 + p),
                    T::load4F32Aligned(s2 + p), T::load4F32Aligned(s3 + p),
                    T::load4F32Aligned(s4 + p), T::load4F32Aligned(s5 + p),
                    T::load4F32Aligned(s6 + p), T::load4F32Aligned(s7 + p),
                    dst4, p, W4, j4);
            }
            for (; p < kc; ++p) {
                dst4[p * W4 + j4]      = float4(T::toF32(s0[p]), T::toF32(s1[p]),
                                                T::toF32(s2[p]), T::toF32(s3[p]));
                dst4[p * W4 + j4 + 1u] = float4(T::toF32(s4[p]), T::toF32(s5[p]),
                                                T::toF32(s6[p]), T::toF32(s7[p]));
            }
        }
        for (; nr - j >= 4u; j += 4u) {
            Ptr<T.Storage, Access.Read> s0 = src + uint64_t(col0 + j + 0u) * s64 + k64;
            Ptr<T.Storage, Access.Read> s1 = src + uint64_t(col0 + j + 1u) * s64 + k64;
            Ptr<T.Storage, Access.Read> s2 = src + uint64_t(col0 + j + 2u) * s64 + k64;
            Ptr<T.Storage, Access.Read> s3 = src + uint64_t(col0 + j + 3u) * s64 + k64;
            uint j4 = j >> 2u;
            uint p = 0u;
            for (; p < kc4; p += 4u) {
                transpose4x4_store<float>(T::load4F32Aligned(s0 + p), T::load4F32Aligned(s1 + p),
                                          T::load4F32Aligned(s2 + p), T::load4F32Aligned(s3 + p),
                                          dst4, p, W4, j4);
            }
            for (; p < kc; ++p) {
                dst4[p * W4 + j4] = float4(T::toF32(s0[p]), T::toF32(s1[p]),
                                           T::toF32(s2[p]), T::toF32(s3[p]));
            }
        }
    } else {
        for (; nr - j >= 4u; j += 4u) {
            Ptr<T.Storage, Access.Read> s0 = src + uint64_t(col0 + j + 0u) * s64 + k64;
            Ptr<T.Storage, Access.Read> s1 = src + uint64_t(col0 + j + 1u) * s64 + k64;
            Ptr<T.Storage, Access.Read> s2 = src + uint64_t(col0 + j + 2u) * s64 + k64;
            Ptr<T.Storage, Access.Read> s3 = src + uint64_t(col0 + j + 3u) * s64 + k64;
            uint j4 = j >> 2u;
            uint p = 0u;
            for (; kc - p >= 4u; p += 4u) {
                transpose4x4_store<float>(T::load4F32(s0 + p), T::load4F32(s1 + p),
                                          T::load4F32(s2 + p), T::load4F32(s3 + p),
                                          dst4, p, W4, j4);
            }
            for (; p < kc; ++p) {
                dst4[p * W4 + j4] = float4(T::toF32(s0[p]), T::toF32(s1[p]),
                                           T::toF32(s2[p]), T::toF32(s3[p]));
            }
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
        Ptr<float> dr = dst + r * kApStrideF32;
        for (uint p = 0u; p < kc; ++p) dr[p] = T::toF32(sr[p]);
    }
    if (mr < uint(MR)) {
        for (uint r = mr; r < uint(MR); ++r) {
            Ptr<float> dr = dst + r * kApStrideF32;
            for (uint p = 0u; p < kc; ++p) dr[p] = 0.0f;
        }
    }
}

void pack_b_f64<let NR : int>(
    Ptr<double, Access.Read> src, uint stride, uint col0, uint nr,
    uint k0, uint kc, Ptr<double4> dst4, bool aligned_src = false)
{
    static const uint W4 = uint(NR) >> 2u;
    uint64_t s64 = uint64_t(stride);
    uint64_t k64 = uint64_t(k0);
    Ptr<double> dst = Ptr<double>(dst4);
    uint j = 0u;
    if (aligned_src && (k0 & 3u) == 0u && (stride & 3u) == 0u) {
        uint kc4 = kc >> 2u;
        for (; nr - j >= 8u; j += 8u) {
            Ptr<double, Access.Read> b0 = src + uint64_t(col0 + j + 0u) * s64 + k64;
            Ptr<double, Access.Read> b1 = src + uint64_t(col0 + j + 1u) * s64 + k64;
            Ptr<double, Access.Read> b2 = src + uint64_t(col0 + j + 2u) * s64 + k64;
            Ptr<double, Access.Read> b3 = src + uint64_t(col0 + j + 3u) * s64 + k64;
            Ptr<double, Access.Read> b4 = src + uint64_t(col0 + j + 4u) * s64 + k64;
            Ptr<double, Access.Read> b5 = src + uint64_t(col0 + j + 5u) * s64 + k64;
            Ptr<double, Access.Read> b6 = src + uint64_t(col0 + j + 6u) * s64 + k64;
            Ptr<double, Access.Read> b7 = src + uint64_t(col0 + j + 7u) * s64 + k64;
            Ptr<double4, Access.Read> v0 = Ptr<double4, Access.Read>(b0);
            Ptr<double4, Access.Read> v1 = Ptr<double4, Access.Read>(b1);
            Ptr<double4, Access.Read> v2 = Ptr<double4, Access.Read>(b2);
            Ptr<double4, Access.Read> v3 = Ptr<double4, Access.Read>(b3);
            Ptr<double4, Access.Read> v4 = Ptr<double4, Access.Read>(b4);
            Ptr<double4, Access.Read> v5 = Ptr<double4, Access.Read>(b5);
            Ptr<double4, Access.Read> v6 = Ptr<double4, Access.Read>(b6);
            Ptr<double4, Access.Read> v7 = Ptr<double4, Access.Read>(b7);
            uint j4 = j >> 2u;
            uint p4 = 0u;
            for (; kc4 - p4 >= 2u; p4 += 2u) {
                double4 a0 = v0[p4], a1 = v1[p4], a2 = v2[p4], a3 = v3[p4];
                double4 a4 = v4[p4], a5 = v5[p4], a6 = v6[p4], a7 = v7[p4];
                double4 c0 = v0[p4 + 1u], c1 = v1[p4 + 1u], c2 = v2[p4 + 1u], c3 = v3[p4 + 1u];
                double4 c4 = v4[p4 + 1u], c5 = v5[p4 + 1u], c6 = v6[p4 + 1u], c7 = v7[p4 + 1u];
                uint p = p4 << 2u;
                transpose4x8_store<double>(a0, a1, a2, a3, a4, a5, a6, a7, dst4, p, W4, j4);
                transpose4x8_store<double>(c0, c1, c2, c3, c4, c5, c6, c7, dst4, p + 4u, W4, j4);
            }
            for (; p4 < kc4; ++p4) {
                transpose4x8_store<double>(v0[p4], v1[p4], v2[p4], v3[p4],
                                           v4[p4], v5[p4], v6[p4], v7[p4],
                                           dst4, p4 << 2u, W4, j4);
            }
            for (uint p = kc4 << 2u; p < kc; ++p) {
                dst4[p * W4 + j4]      = double4(b0[p], b1[p], b2[p], b3[p]);
                dst4[p * W4 + j4 + 1u] = double4(b4[p], b5[p], b6[p], b7[p]);
            }
        }
        for (; nr - j >= 4u; j += 4u) {
            Ptr<double, Access.Read> b0 = src + uint64_t(col0 + j + 0u) * s64 + k64;
            Ptr<double, Access.Read> b1 = src + uint64_t(col0 + j + 1u) * s64 + k64;
            Ptr<double, Access.Read> b2 = src + uint64_t(col0 + j + 2u) * s64 + k64;
            Ptr<double, Access.Read> b3 = src + uint64_t(col0 + j + 3u) * s64 + k64;
            Ptr<double4, Access.Read> v0 = Ptr<double4, Access.Read>(b0);
            Ptr<double4, Access.Read> v1 = Ptr<double4, Access.Read>(b1);
            Ptr<double4, Access.Read> v2 = Ptr<double4, Access.Read>(b2);
            Ptr<double4, Access.Read> v3 = Ptr<double4, Access.Read>(b3);
            uint j4 = j >> 2u;
            for (uint p4 = 0u; p4 < kc4; ++p4) {
                transpose4x4_store<double>(v0[p4], v1[p4], v2[p4], v3[p4], dst4, p4 << 2u, W4, j4);
            }
            for (uint p = kc4 << 2u; p < kc; ++p) {
                dst4[p * W4 + j4] = double4(b0[p], b1[p], b2[p], b3[p]);
            }
        }
    } else {
        for (; nr - j >= 4u; j += 4u) {
            Ptr<double, Access.Read> s0 = src + uint64_t(col0 + j + 0u) * s64 + k64;
            Ptr<double, Access.Read> s1 = src + uint64_t(col0 + j + 1u) * s64 + k64;
            Ptr<double, Access.Read> s2 = src + uint64_t(col0 + j + 2u) * s64 + k64;
            Ptr<double, Access.Read> s3 = src + uint64_t(col0 + j + 3u) * s64 + k64;
            uint j4 = j >> 2u;
            uint p = 0u;
            for (; kc - p >= 4u; p += 4u) {
                transpose4x4_store<double>(
                    double4(s0[p], s0[p+1u], s0[p+2u], s0[p+3u]),
                    double4(s1[p], s1[p+1u], s1[p+2u], s1[p+3u]),
                    double4(s2[p], s2[p+1u], s2[p+2u], s2[p+3u]),
                    double4(s3[p], s3[p+1u], s3[p+2u], s3[p+3u]),
                    dst4, p, W4, j4);
            }
            for (; p < kc; ++p) {
                dst4[p * W4 + j4] = double4(s0[p], s1[p], s2[p], s3[p]);
            }
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
        Ptr<double> dr = dst + r * kApStrideF64;
        for (uint p = 0u; p < kc; ++p) dr[p] = sr[p];
    }
    if (mr < uint(MR)) {
        for (uint r = mr; r < uint(MR); ++r) {
            Ptr<double> dr = dst + r * kApStrideF64;
            for (uint p = 0u; p < kc; ++p) dr[p] = 0.0;
        }
    }
}

// ============================================================================
// 3. Unified Zero-Spill L1-Packed BLIS Outer-Product Microkernels
// ============================================================================
[ForceInline] void gemm_mk_f32_rank1<let MR : int, let NR : int>(
    Ptr<float, Access.Read> arow[MR], Ptr<float> bp, uint p, inout float c[MR * NR])
{
    Ptr<float> brow = bp + p * uint(NR);
    [ForceUnroll] for (uint r = 0u; r < uint(MR); ++r) {
        float av = arow[r][p];
        for (uint j = 0u; j < uint(NR); ++j) {
            c[r * uint(NR) + j] += av * brow[j];
        }
    }
}

void gemm_mk_f32<let MR : int, let NR : int, let ZERO_INIT : bool>(
    Ptr<float, Access.Read> ap, uint lda, Ptr<float> bp,
    Ptr<float> c_src, uint lds, Ptr<float> c_dst, uint ldd, uint kc)
{
    uint64_t lda64 = uint64_t(lda);
    uint64_t lds64 = uint64_t(lds);
    uint64_t ldd64 = uint64_t(ldd);
    Ptr<float, Access.Read> arow[MR];
    [ForceUnroll] for (uint r = 0u; r < uint(MR); ++r) arow[r] = ap + uint64_t(r) * lda64;
    float c[MR * NR];
    if (ZERO_INIT) {
        for (uint i = 0u; i < uint(MR * NR); ++i) c[i] = 0.0f;
    } else {
        for (uint r = 0u; r < uint(MR); ++r) {
            Ptr<float> cr = c_src + uint64_t(r) * lds64;
            for (uint j = 0u; j < uint(NR); ++j) c[r * uint(NR) + j] = cr[j];
        }
    }
    if (NR >= 64) {
        uint p = 0u;
        for (; kc - p >= 4u; p += 4u) {
            [ForceUnroll] for (uint u = 0u; u < 4u; ++u) {
                gemm_mk_f32_rank1<MR, NR>(arow, bp, p + u, c);
            }
        }
        for (; p < kc; ++p) {
            gemm_mk_f32_rank1<MR, NR>(arow, bp, p, c);
        }
    } else {
        for (uint p = 0u; p < kc; ++p) {
            Ptr<float> brow = bp + p * uint(NR);
            float a[MR];
            [ForceUnroll] for (uint r = 0u; r < uint(MR); ++r) a[r] = arow[r][p];
            for (uint j = 0u; j < uint(NR); ++j) {
                float b = brow[j];
                [ForceUnroll] for (uint r = 0u; r < uint(MR); ++r) {
                    c[r * uint(NR) + j] += a[r] * b;
                }
            }
        }
    }
    for (uint r = 0u; r < uint(MR); ++r) {
        Ptr<float> cr = c_dst + uint64_t(r) * ldd64;
        for (uint j = 0u; j < uint(NR); ++j) cr[j] = c[r * uint(NR) + j];
    }
}

void gemm_mk_f64<let MR : int, let NR : int, let ZERO_INIT : bool>(
    Ptr<double, Access.Read> ap, uint lda, Ptr<double> bp,
    Ptr<double> c_src, uint lds, Ptr<double> c_dst, uint ldd, uint kc)
{
    uint64_t lda64 = uint64_t(lda);
    uint64_t lds64 = uint64_t(lds);
    uint64_t ldd64 = uint64_t(ldd);
    Ptr<double, Access.Read> arow[MR];
    [ForceUnroll] for (uint r = 0u; r < uint(MR); ++r) arow[r] = ap + uint64_t(r) * lda64;
    if (NR >= 32) {
        double c[MR * NR];
        if (ZERO_INIT) {
            for (uint i = 0u; i < uint(MR * NR); ++i) c[i] = 0.0;
        } else {
            for (uint r = 0u; r < uint(MR); ++r) {
                Ptr<double> cr = c_src + uint64_t(r) * lds64;
                for (uint j = 0u; j < uint(NR); ++j) c[r * uint(NR) + j] = cr[j];
            }
        }
        for (uint p = 0u; p < kc; ++p) {
            Ptr<double> brow = bp + p * uint(NR);
            double a[MR];
            [ForceUnroll] for (uint r = 0u; r < uint(MR); ++r) a[r] = arow[r][p];
            for (uint j = 0u; j < uint(NR); ++j) {
                double b = brow[j];
                [ForceUnroll] for (uint r = 0u; r < uint(MR); ++r) {
                    c[r * uint(NR) + j] += a[r] * b;
                }
            }
        }
        for (uint r = 0u; r < uint(MR); ++r) {
            Ptr<double> cr = c_dst + uint64_t(r) * ldd64;
            for (uint j = 0u; j < uint(NR); ++j) cr[j] = c[r * uint(NR) + j];
        }
    } else {
        static const uint W4 = uint(NR) >> 2u;
        double4 c[MR * (NR / 4)];
        if (ZERO_INIT) {
            [ForceUnroll] for (uint i = 0u; i < uint(MR) * W4; ++i) c[i] = double4(0.0);
        } else {
            [ForceUnroll] for (uint r = 0u; r < uint(MR); ++r) {
                Ptr<double> cr = c_src + uint64_t(r) * lds64;
                [ForceUnroll] for (uint j = 0u; j < W4; ++j)
                    c[r * W4 + j] = double4(cr[j*4u], cr[j*4u+1u], cr[j*4u+2u], cr[j*4u+3u]);
            }
        }
        Ptr<double4> bp4 = Ptr<double4>(bp);
        for (uint p = 0u; p < kc; ++p) {
            Ptr<double4> brow4 = bp4 + p * W4;
            double a[MR];
            [ForceUnroll] for (uint r = 0u; r < uint(MR); ++r) a[r] = arow[r][p];
            [ForceUnroll] for (uint j = 0u; j < W4; ++j) {
                double4 b = brow4[j];
                [ForceUnroll] for (uint r = 0u; r < uint(MR); ++r) {
                    c[r * W4 + j] += a[r] * b;
                }
            }
        }
        [ForceUnroll] for (uint r = 0u; r < uint(MR); ++r) {
            Ptr<double> cr = c_dst + uint64_t(r) * ldd64;
            [ForceUnroll] for (uint j = 0u; j < W4; ++j) {
                double4 v = c[r * W4 + j];
                cr[j*4u+0u] = v.x; cr[j*4u+1u] = v.y; cr[j*4u+2u] = v.z; cr[j*4u+3u] = v.w;
            }
        }
    }
}

// Split an MC panel for MR == 6 into 6-row tiles + 4-row tail tiles (rem==2 -> 4+4).
[ForceInline] uint mc_split_m6(uint mc) {
    uint rem = mc % 6u;
    return (rem == 2u && mc >= 8u) ? (mc - 8u) : (mc - rem);
}

void linear_mc_step_f32<let TILE_MR : int, let NR : int, TX : IElement, TY : IElement, Ep : IEpilogue,
                        let ZERO_INIT : bool>(
    Ptr<TX.Storage, Access.Read> x, uint x_stride,
    Ptr<float> ap, Ptr<float, Access.Read> ap_rd, Ptr<float> bp, Ptr<float> c_tile,
    Ptr<TY.Storage> y, uint y_stride,
    uint row0, uint mr, uint col0, uint nr, uint k0, uint kc, bool last_k, EpilogueParams ep)
{
    bool direct_y = last_k && TY.kIsNativeF32 && Ep::isIdentity(ep) &&
                    mr == uint(TILE_MR) && nr == uint(NR);
    Ptr<float> c_dst = direct_y
        ? (Ptr<float>(y) + uint64_t(row0) * uint64_t(y_stride) + uint64_t(col0))
        : c_tile;
    uint ldd = direct_y ? y_stride : uint(NR);

    if (TX.kIsNativeF32 && mr == uint(TILE_MR)) {
        Ptr<float, Access.Read> xp =
            Ptr<float, Access.Read>(x) + uint64_t(row0) * uint64_t(x_stride) + uint64_t(k0);
        gemm_mk_f32<TILE_MR, NR, ZERO_INIT>(xp, x_stride, bp, c_tile, uint(NR), c_dst, ldd, kc);
    } else {
        pack_a_f32<TX, TILE_MR>(x, x_stride, row0, mr, k0, kc, ap);
        gemm_mk_f32<TILE_MR, NR, ZERO_INIT>(ap_rd, kApStrideF32, bp, c_tile, uint(NR), c_dst, ldd, kc);
    }
    if (last_k && !direct_y) {
        for (uint r = 0u; r < mr; ++r) {
            uint rr = row0 + r;
            Ptr<TY.Storage> yr = y + uint64_t(rr) * uint64_t(y_stride) + uint64_t(col0);
            for (uint j = 0u; j < nr; ++j)
                yr[j] = TY::fromF32(Ep::apply(c_tile[r * uint(NR) + j], rr, col0 + j, ep));
        }
    }
}

void linear_mc_panel_f32<let MR : int, let NR : int, TX : IElement, TY : IElement, Ep : IEpilogue,
                         let ZERO_INIT : bool>(
    Ptr<TX.Storage, Access.Read> x, uint x_stride,
    Ptr<float> ap, Ptr<float, Access.Read> ap_rd, Ptr<float> bp, Ptr<float> c_buf,
    Ptr<TY.Storage> y, uint y_stride,
    uint rb, uint mc, uint cb, uint nr, uint k0, uint kc, bool last_k, EpilogueParams ep)
{
    if (MR == 6) {
        uint m6 = mc_split_m6(mc);
        uint rm = 0u;
        for (; rm < m6; rm += 6u) {
            linear_mc_step_f32<6, NR, TX, TY, Ep, ZERO_INIT>(
                x, x_stride, ap, ap_rd, bp, c_buf + rm * uint(NR), y, y_stride,
                rb + rm, 6u, cb, nr, k0, kc, last_k, ep);
        }
        for (; rm < mc; rm += 4u) {
            uint mr = min(4u, mc - rm);
            linear_mc_step_f32<4, NR, TX, TY, Ep, ZERO_INIT>(
                x, x_stride, ap, ap_rd, bp, c_buf + rm * uint(NR), y, y_stride,
                rb + rm, mr, cb, nr, k0, kc, last_k, ep);
        }
    } else {
        for (uint rm = 0u; rm < mc; rm += uint(MR)) {
            uint mr = min(uint(MR), mc - rm);
            linear_mc_step_f32<MR, NR, TX, TY, Ep, ZERO_INIT>(
                x, x_stride, ap, ap_rd, bp, c_buf + rm * uint(NR), y, y_stride,
                rb + rm, mr, cb, nr, k0, kc, last_k, ep);
        }
    }
}

void linear_mc_step_f64<let TILE_MR : int, let NR : int, let ZERO_INIT : bool>(
    Ptr<double, Access.Read> x, uint x_stride,
    Ptr<double> ap, Ptr<double, Access.Read> ap_rd, Ptr<double> bp, Ptr<double> c_tile,
    Ptr<double> y, uint y_stride,
    Ptr<double, Access.Read> bias, uint has_bias,
    uint row0, uint mr, uint col0, uint nr, uint k0, uint kc, bool last_k)
{
    bool direct_y = last_k && has_bias == 0u && mr == uint(TILE_MR) && nr == uint(NR);
    Ptr<double> c_dst = direct_y
        ? (y + uint64_t(row0) * uint64_t(y_stride) + uint64_t(col0))
        : c_tile;
    uint ldd = direct_y ? y_stride : uint(NR);

    if (mr == uint(TILE_MR)) {
        Ptr<double, Access.Read> xp = x + uint64_t(row0) * uint64_t(x_stride) + uint64_t(k0);
        gemm_mk_f64<TILE_MR, NR, ZERO_INIT>(xp, x_stride, bp, c_tile, uint(NR), c_dst, ldd, kc);
    } else {
        pack_a_f64<TILE_MR>(x, x_stride, row0, mr, k0, kc, ap);
        gemm_mk_f64<TILE_MR, NR, ZERO_INIT>(ap_rd, kApStrideF64, bp, c_tile, uint(NR), c_dst, ldd, kc);
    }
    if (last_k && !direct_y) {
        for (uint r = 0u; r < mr; ++r) {
            Ptr<double> yr = y + uint64_t(row0 + r) * uint64_t(y_stride) + uint64_t(col0);
            for (uint j = 0u; j < nr; ++j)
                yr[j] = c_tile[r * uint(NR) + j] + (has_bias != 0u ? bias[col0 + j] : 0.0);
        }
    }
}

void linear_mc_panel_f64<let MR : int, let NR : int, let ZERO_INIT : bool>(
    Ptr<double, Access.Read> x, uint x_stride,
    Ptr<double> ap, Ptr<double, Access.Read> ap_rd, Ptr<double> bp, Ptr<double> c_buf,
    Ptr<double> y, uint y_stride,
    Ptr<double, Access.Read> bias, uint has_bias,
    uint rb, uint mc, uint cb, uint nr, uint k0, uint kc, bool last_k)
{
    if (MR == 6) {
        uint m6 = mc_split_m6(mc);
        uint rm = 0u;
        for (; rm < m6; rm += 6u) {
            linear_mc_step_f64<6, NR, ZERO_INIT>(
                x, x_stride, ap, ap_rd, bp, c_buf + rm * uint(NR), y, y_stride,
                bias, has_bias, rb + rm, 6u, cb, nr, k0, kc, last_k);
        }
        for (; rm < mc; rm += 4u) {
            uint mr = min(4u, mc - rm);
            linear_mc_step_f64<4, NR, ZERO_INIT>(
                x, x_stride, ap, ap_rd, bp, c_buf + rm * uint(NR), y, y_stride,
                bias, has_bias, rb + rm, mr, cb, nr, k0, kc, last_k);
        }
    } else {
        for (uint rm = 0u; rm < mc; rm += uint(MR)) {
            uint mr = min(uint(MR), mc - rm);
            linear_mc_step_f64<MR, NR, ZERO_INIT>(
                x, x_stride, ap, ap_rd, bp, c_buf + rm * uint(NR), y, y_stride,
                bias, has_bias, rb + rm, mr, cb, nr, k0, kc, last_k);
        }
    }
}

// ============================================================================
// 4. Unified Floating-Point Linear (F32, F16->F32, BF16->F32, F32xBF16)
// ============================================================================
// Large-batch 5-loop BLIS helpers (MR == 6, nrows > kMC): pack NC = kNcTiles * NR
// columns of B once per (MB, KC) block and pack non-native A (F16/BF16) once per
// MC rows across all NC strips, avoiding redundant DRAM repacking and conversion.
void pack_a_panel_f32<T : IElement>(
    Ptr<T.Storage, Access.Read> src, uint stride, uint row0, uint mc,
    uint k0, uint kc, Ptr<float> dst)
{
    uint64_t s64 = uint64_t(stride);
    uint64_t k64 = uint64_t(k0);
    for (uint r = 0u; r < mc; ++r) {
        Ptr<T.Storage, Access.Read> sr = src + uint64_t(row0 + r) * s64 + k64;
        Ptr<float> dr = dst + r * kApStrideF32;
        for (uint p = 0u; p < kc; ++p) dr[p] = T::toF32(sr[p]);
    }
    for (uint r = mc; r < mc + 3u; ++r) {
        Ptr<float> dr = dst + r * kApStrideF32;
        for (uint p = 0u; p < kc; ++p) dr[p] = 0.0f;
    }
}

[ForceInline] void blis_step_f32<let TILE_MR : int, let NR : int,
                                 TX : IElement, TY : IElement, Ep : IEpilogue,
                                 let PACK_A : bool>(
    Ptr<TX.Storage, Access.Read> x, uint x_stride,
    Ptr<float> ap_wr, Ptr<float, Access.Read> ap_rd, Ptr<float> bp, Ptr<float> ct_ptr,
    Ptr<TY.Storage> y, uint y_stride,
    uint row0, uint rm, uint mr, uint cb, uint nr, uint k0, uint kc,
    bool first, bool last, EpilogueParams ep)
{
    if (mr == uint(TILE_MR) && nr == uint(NR)) {
        Ptr<float> ytile = Ptr<float>(y) + uint64_t(row0) * uint64_t(y_stride) + uint64_t(cb);
        Ptr<float, Access.Read> aptr = PACK_A
            ? (ap_rd + rm * kApStrideF32)
            : (Ptr<float, Access.Read>(x) + uint64_t(row0) * uint64_t(x_stride) + uint64_t(k0));
        uint lda = PACK_A ? kApStrideF32 : x_stride;
        if (first) gemm_mk_f32<TILE_MR, NR, true>(aptr, lda, bp, ytile, y_stride, ytile, y_stride, kc);
        else gemm_mk_f32<TILE_MR, NR, false>(aptr, lda, bp, ytile, y_stride, ytile, y_stride, kc);
        if (last && !Ep::isIdentity(ep)) {
            for (uint r = 0u; r < uint(TILE_MR); ++r) {
                Ptr<float> yr = ytile + uint64_t(r) * uint64_t(y_stride);
                for (uint j = 0u; j < uint(NR); ++j) yr[j] = Ep::apply(yr[j], row0 + r, cb + j, ep);
            }
        }
        return;
    }
    if (!first) {
        for (uint r = 0u; r < mr; ++r) {
            Ptr<float> yr = Ptr<float>(y) + uint64_t(row0 + r) * uint64_t(y_stride) + uint64_t(cb);
            for (uint j = 0u; j < nr; ++j) ct_ptr[r * uint(NR) + j] = yr[j];
            for (uint j = nr; j < uint(NR); ++j) ct_ptr[r * uint(NR) + j] = 0.0f;
        }
    }
    Ptr<float, Access.Read> aptr = ap_rd + rm * kApStrideF32;
    if (!PACK_A) {
        pack_a_f32<TX, TILE_MR>(x, x_stride, row0, mr, k0, kc, ap_wr);
        aptr = ap_rd;
    }
    if (first) gemm_mk_f32<TILE_MR, NR, true>(aptr, kApStrideF32, bp, ct_ptr, uint(NR), ct_ptr, uint(NR), kc);
    else gemm_mk_f32<TILE_MR, NR, false>(aptr, kApStrideF32, bp, ct_ptr, uint(NR), ct_ptr, uint(NR), kc);
    for (uint r = 0u; r < mr; ++r) {
        Ptr<float> yr = Ptr<float>(y) + uint64_t(row0 + r) * uint64_t(y_stride) + uint64_t(cb);
        if (last && !Ep::isIdentity(ep)) {
            for (uint j = 0u; j < nr; ++j) yr[j] = Ep::apply(ct_ptr[r * uint(NR) + j], row0 + r, cb + j, ep);
        } else {
            for (uint j = 0u; j < nr; ++j) yr[j] = ct_ptr[r * uint(NR) + j];
        }
    }
}

void blis_f32_large<let MR : int, let NR : int,
                    TX : IElement, TW : IElement, TY : IElement, Ep : IEpilogue>(
    Ptr<TX.Storage, Access.Read> x, uint x_stride,
    Ptr<TW.Storage, Access.Read> w, uint w_stride,
    Ptr<TY.Storage> y, uint y_stride,
    uint r0, uint r1, uint c0, uint c1, uint k,
    EpilogueParams ep, bool aligned_w)
{
    static const int kNcTiles = 2;
    static const uint kNC = uint(kNcTiles * NR);
    static const uint kMcBlis = 192u;
    static const uint kMbBlis = 512u;
    static const bool kPackA = !TX.kIsNativeF32;
    float4 bp_all[kNcTiles * VELRI_KC * (NR / 4)];
    float ap_all[(kPackA ? (192 + 3) : 4) * (VELRI_KC + VELRI_AP_PAD_F32)];
    float c_tail[MR * NR];
    bp_all[0] = float4(0.0f); ap_all[0] = 0.0f; c_tail[0] = 0.0f;
    Ptr<float> bp_base = Ptr<float>(&bp_all[0]);
    Ptr<float> ap_wr = &ap_all[0];
    Ptr<float, Access.Read> ap_rd = Ptr<float, Access.Read>(&ap_all[0]);
    Ptr<float> ct_ptr = &c_tail[0];

    for (uint jc = c0; jc < c1;) {
        uint nc = min(kNC, c1 - jc);
        uint n_tiles = (nc + uint(NR) - 1u) / uint(NR);
        for (uint rb = r0; rb < r1;) {
            uint mb = min(kMbBlis, r1 - rb);
            for (uint k0 = 0u; k0 < k;) {
                uint kc = min(kKC, k - k0);
                bool first = (k0 == 0u);
                bool last = (k0 + kc == k);
                for (uint t = 0u; t < n_tiles; ++t) {
                    uint cb = jc + t * uint(NR);
                    uint nr = min(uint(NR), c1 - cb);
                    pack_b_f32<TW, NR>(w, w_stride, cb, nr, k0, kc,
                                       &bp_all[t * kKC * (uint(NR) >> 2u)], aligned_w);
                }
                for (uint ic = 0u; ic < mb;) {
                    uint mc = min(kMcBlis, mb - ic);
                    uint row_base = rb + ic;
                    if (kPackA) pack_a_panel_f32<TX>(x, x_stride, row_base, mc, k0, kc, ap_wr);
                    uint m6 = mc_split_m6(mc);
                    for (uint t = 0u; t < n_tiles; ++t) {
                        uint cb = jc + t * uint(NR);
                        uint nr = min(uint(NR), c1 - cb);
                        Ptr<float> bp = bp_base + t * kKC * uint(NR);
                        uint rm = 0u;
                        if (nr == uint(NR)) {
                            for (; rm < m6; rm += 6u) {
                                blis_step_f32<6, NR, TX, TY, Ep, kPackA>(
                                    x, x_stride, ap_wr, ap_rd, bp, ct_ptr, y, y_stride,
                                    row_base + rm, rm, 6u, cb, nr, k0, kc, first, last, ep);
                            }
                        }
                        for (; rm < mc;) {
                            uint mr = min(4u, mc - rm);
                            blis_step_f32<4, NR, TX, TY, Ep, kPackA>(
                                x, x_stride, ap_wr, ap_rd, bp, ct_ptr, y, y_stride,
                                row_base + rm, rm, mr, cb, nr, k0, kc, first, last, ep);
                            rm += mr;
                        }
                    }
                    ic += mc;
                }
                k0 += kc;
            }
            rb += mb;
        }
        jc += nc;
    }
}

public void linear_slice_cpu<let MR : int, let NR : int,
                             TX : IElement, TW : IElement, TY : IElement, Ep : IEpilogue>(
    Ptr<TX.Storage, Access.Read> x, uint x_stride,
    Ptr<TW.Storage, Access.Read> w, uint w_stride,
    Ptr<TY.Storage> y, uint y_stride,
    uint r0, uint r1, uint c0, uint c1, uint k,
    EpilogueParams ep, bool stream_weights = false, bool aligned_w = false)
{
    static_assert(NR > 0 && (NR % 4) == 0, "NR must be a positive multiple of 4");
    static_assert(VELRI_MC % MR == 0, "VELRI_MC must be divisible by MR");
    if (r0 >= r1 || c0 >= c1) return;
    uint nrows = r1 - r0;
    if (nrows < kGemvRowThreshold || k == 0u) {
        for (uint r = r0; r < r1; ++r) {
            Ptr<TX.Storage, Access.Read> xr = x + uint64_t(r) * uint64_t(x_stride);
            Ptr<TY.Storage> yr = y + uint64_t(r) * uint64_t(y_stride);
            if (stream_weights) {
                // Traverse a dense matrix sequentially while the activation stays in cache.
                for (uint c = c0; c < c1; ++c)
                    gemv_1row<TX, TW, TY, Ep>(xr, w + uint64_t(c) * uint64_t(w_stride),
                                              yr, r, c, k, ep);
                continue;
            }
            uint c = c0;
            for (; c1 - c >= 4u; c += 4u) {
                gemv_4rows<TX, TW, TY, Ep>(xr, w, w_stride, yr, r, c, k, ep);
            }
            for (; c < c1; ++c) {
                gemv_1row<TX, TW, TY, Ep>(xr, w + uint64_t(c) * uint64_t(w_stride), yr, r, c, k, ep);
            }
        }
        return;
    }
    if (MR == 6 && TY.kIsNativeF32 && nrows > kMC) {
        blis_f32_large<MR, NR, TX, TW, TY, Ep>(
            x, x_stride, w, w_stride, y, y_stride, r0, r1, c0, c1, k, ep, aligned_w);
        return;
    }

    float4 bp4[VELRI_KC * (NR / 4)];
    float ap[MR * (VELRI_KC + VELRI_AP_PAD_F32)];
    float c_buf[VELRI_MC_BUF_ROWS * NR];
    bp4[0] = float4(0.0f); ap[0] = 0.0f; c_buf[0] = 0.0f;
    Ptr<float> bp = Ptr<float>(&bp4[0]);
    Ptr<float> ap_wr = &ap[0];
    Ptr<float, Access.Read> ap_rd = Ptr<float, Access.Read>(&ap[0]);
    Ptr<float> cb_ptr = &c_buf[0];

    for (uint cb = c0; cb < c1;) {
        uint nr = min(uint(NR), c1 - cb);
        for (uint rb = r0; rb < r1;) {
            uint mc = min(kMC, r1 - rb);
            for (uint k0 = 0u; k0 < k;) {
                uint kc = min(kKC, k - k0);
                pack_b_f32<TW, NR>(w, w_stride, cb, nr, k0, kc, &bp4[0], aligned_w);
                bool last = (k0 + kc == k);
                if (k0 == 0u) {
                    linear_mc_panel_f32<MR, NR, TX, TY, Ep, true>(
                        x, x_stride, ap_wr, ap_rd, bp, cb_ptr, y, y_stride, rb, mc, cb, nr, k0, kc, last, ep);
                } else {
                    linear_mc_panel_f32<MR, NR, TX, TY, Ep, false>(
                        x, x_stride, ap_wr, ap_rd, bp, cb_ptr, y, y_stride, rb, mc, cb, nr, k0, kc, last, ep);
                }
                k0 += kc;
            }
            rb += mc;
        }
        cb += nr;
    }
}

// ============================================================================
// 5. Exact INT8 * INT8 -> INT32 (modulo 2^32) & W8A8 -> F32
// ============================================================================
// Packed weights store signed bytes in [channel panel][depth][channel] order.
// Each panel is 32 columns wide, including the zero-padded last panel.
#include "w8a8_layout.h"
static const uint kW8A8PanelWidth = VELRI_W8A8_SIGNED_PANEL_WIDTH;

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
    EpilogueParams ep, bool direct_order = false)
{
    static_assert(kKC > 0 && kKC <= 1024, "Each INT8 K block must sum exactly in F32");
    static_assert(NR > 0 && (NR % 4) == 0, "NR must be a positive multiple of 4");
    if (r0 >= r1 || c0 >= c1) return;
    if (direct_order || r1 - r0 < kGemvRowThreshold) {
        if (PACKED) {
            packed_w8a8_gemv(x, x_stride, w, y, y_stride, r0, r1, c0, c1, k, ep);
            return;
        }
        bool stream_rows = w8a8_should_stream_rows(direct_order, c1 - c0, k);
        for (uint r = r0; r < r1; ++r) {
            Ptr<int8_t, Access.Read> xr = x + uint64_t(r) * x_stride;
            Ptr<float> yr = y + uint64_t(r) * y_stride;
            uint c = c0;
            if (!stream_rows) {
                for (; c1 - c >= 4u; c += 4u)
                    w8a8_gemv_columns<4>(xr, w, w_stride, yr, r, c, k, ep);
            }
            for (; c < c1; ++c)
                w8a8_gemv_columns<1>(xr, w, w_stride, yr, r, c, k, ep);
        }
        return;
    }

    float4 bp4[VELRI_KC * (NR / 4)];
    float ap[MR * (VELRI_KC + VELRI_AP_PAD_F32)];
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
                    pack_b_f32<ElemI8, NR, false>(w, w_stride, cb, nr, k0, kc, &bp4[0]);
                }
                for (uint rm = 0u; rm < mc; rm += uint(MR)) {
                    uint mr = min(uint(MR), mc - rm);
                    pack_a_f32<ElemI8, MR>(x, x_stride, rb + rm, mr, k0, kc, &ap[0]);
                    gemm_mk_f32<MR, NR, true>(Ptr<float, Access.Read>(&ap[0]), kApStrideF32, bp,
                                              &c_tmp[0], uint(NR), &c_tmp[0], uint(NR), kc);
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
[ForceInline] void blis_step_f64<let TILE_MR : int, let NR : int>(
    Ptr<double, Access.Read> x, uint x_stride,
    Ptr<double> ap_wr, Ptr<double, Access.Read> ap_rd, Ptr<double> bp, Ptr<double> ct_ptr,
    Ptr<double> y, uint y_stride, Ptr<double, Access.Read> bias, uint has_bias,
    uint row0, uint mr, uint cb, uint nr, uint k0, uint kc, bool first, bool last)
{
    if (mr == uint(TILE_MR) && nr == uint(NR)) {
        Ptr<double> ytile = y + uint64_t(row0) * uint64_t(y_stride) + uint64_t(cb);
        Ptr<double, Access.Read> aptr = x + uint64_t(row0) * uint64_t(x_stride) + uint64_t(k0);
        if (first) gemm_mk_f64<TILE_MR, NR, true>(aptr, x_stride, bp, ytile, y_stride, ytile, y_stride, kc);
        else gemm_mk_f64<TILE_MR, NR, false>(aptr, x_stride, bp, ytile, y_stride, ytile, y_stride, kc);
        if (last && has_bias != 0u) {
            for (uint r = 0u; r < uint(TILE_MR); ++r) {
                Ptr<double> yr = ytile + uint64_t(r) * uint64_t(y_stride);
                for (uint j = 0u; j < uint(NR); ++j) yr[j] += bias[cb + j];
            }
        }
        return;
    }
    if (!first) {
        for (uint r = 0u; r < mr; ++r) {
            Ptr<double> yr = y + uint64_t(row0 + r) * uint64_t(y_stride) + uint64_t(cb);
            for (uint j = 0u; j < nr; ++j) ct_ptr[r * uint(NR) + j] = yr[j];
            for (uint j = nr; j < uint(NR); ++j) ct_ptr[r * uint(NR) + j] = 0.0;
        }
    }
    pack_a_f64<TILE_MR>(x, x_stride, row0, mr, k0, kc, ap_wr);
    if (first) gemm_mk_f64<TILE_MR, NR, true>(ap_rd, kApStrideF64, bp, ct_ptr, uint(NR), ct_ptr, uint(NR), kc);
    else gemm_mk_f64<TILE_MR, NR, false>(ap_rd, kApStrideF64, bp, ct_ptr, uint(NR), ct_ptr, uint(NR), kc);
    for (uint r = 0u; r < mr; ++r) {
        Ptr<double> yr = y + uint64_t(row0 + r) * uint64_t(y_stride) + uint64_t(cb);
        if (last && has_bias != 0u) {
            for (uint j = 0u; j < nr; ++j) yr[j] = ct_ptr[r * uint(NR) + j] + bias[cb + j];
        } else {
            for (uint j = 0u; j < nr; ++j) yr[j] = ct_ptr[r * uint(NR) + j];
        }
    }
}

void blis_f64_large<let MR : int, let NR : int>(
    Ptr<double, Access.Read> x, uint x_stride,
    Ptr<double, Access.Read> w, uint w_stride,
    Ptr<double, Access.Read> bias, uint has_bias,
    Ptr<double> y, uint y_stride,
    uint r0, uint r1, uint c0, uint c1, uint k, bool aligned_w)
{
    static const int kNcTiles = 4;
    static const uint kNC = uint(kNcTiles * NR);
    static const uint kMcBlis = 144u;
    static const uint kMbBlis = 768u;
    double4 bp_all[kNcTiles * VELRI_KC * (NR / 4)];
    double ap_tail[4 * (VELRI_KC + VELRI_AP_PAD_F64)];
    double c_tail[MR * NR];
    bp_all[0] = double4(0.0); ap_tail[0] = 0.0; c_tail[0] = 0.0;
    Ptr<double> bp_base = Ptr<double>(&bp_all[0]);
    Ptr<double> ap_wr = &ap_tail[0];
    Ptr<double, Access.Read> ap_rd = Ptr<double, Access.Read>(&ap_tail[0]);
    Ptr<double> ct_ptr = &c_tail[0];

    for (uint jc = c0; jc < c1;) {
        uint nc = min(kNC, c1 - jc);
        uint n_tiles = (nc + uint(NR) - 1u) / uint(NR);
        for (uint rb = r0; rb < r1;) {
            uint mb = min(kMbBlis, r1 - rb);
            for (uint k0 = 0u; k0 < k;) {
                uint kc = min(kKC, k - k0);
                bool first = (k0 == 0u);
                bool last = (k0 + kc == k);
                for (uint t = 0u; t < n_tiles; ++t) {
                    uint cb = jc + t * uint(NR);
                    uint nr = min(uint(NR), c1 - cb);
                    pack_b_f64<NR>(w, w_stride, cb, nr, k0, kc,
                                   &bp_all[t * kKC * (uint(NR) >> 2u)], aligned_w);
                }
                for (uint ic = 0u; ic < mb;) {
                    uint mc = min(kMcBlis, mb - ic);
                    uint row_base = rb + ic;
                    uint m6 = mc_split_m6(mc);
                    for (uint t = 0u; t < n_tiles; ++t) {
                        uint cb = jc + t * uint(NR);
                        uint nr = min(uint(NR), c1 - cb);
                        Ptr<double> bp = bp_base + t * kKC * uint(NR);
                        uint rm = 0u;
                        if (nr == uint(NR)) {
                            for (; rm < m6; rm += 6u) {
                                blis_step_f64<6, NR>(
                                    x, x_stride, ap_wr, ap_rd, bp, ct_ptr, y, y_stride,
                                    bias, has_bias, row_base + rm, 6u, cb, nr, k0, kc, first, last);
                            }
                        }
                        for (; rm < mc;) {
                            uint mr = min(4u, mc - rm);
                            blis_step_f64<4, NR>(
                                x, x_stride, ap_wr, ap_rd, bp, ct_ptr, y, y_stride,
                                bias, has_bias, row_base + rm, mr, cb, nr, k0, kc, first, last);
                            rm += mr;
                        }
                    }
                    ic += mc;
                }
                k0 += kc;
            }
            rb += mb;
        }
        jc += nc;
    }
}

public void linear_f64_slice_cpu<let MR : int, let NR : int>(
    Ptr<double, Access.Read> x, uint x_stride,
    Ptr<double, Access.Read> w, uint w_stride,
    Ptr<double, Access.Read> bias, uint flags,
    Ptr<double> y, uint y_stride,
    uint r0, uint r1, uint c0, uint c1, uint k,
    bool stream_weights = false, bool aligned_w = false)
{
    static_assert(NR > 0 && (NR % 4) == 0, "NR must be a positive multiple of 4");
    static_assert(VELRI_MC % MR == 0, "VELRI_MC must be divisible by MR");
    if (r0 >= r1 || c0 >= c1) return;
    uint has_bias = flags & kEpFlagBias;
    uint nrows = r1 - r0;
    uint64_t ws64 = uint64_t(w_stride);
    if (nrows < kGemvRowThreshold || k == 0u) {
        for (uint r = r0; r < r1; ++r) {
            Ptr<double, Access.Read> xr = x + uint64_t(r) * uint64_t(x_stride);
            Ptr<double> yr = y + uint64_t(r) * uint64_t(y_stride);
            if (stream_weights) {
                for (uint c = c0; c < c1; ++c) {
                    Ptr<double, Access.Read> wc = w + uint64_t(c) * ws64;
                    double s = 0.0;
                    for (uint i = 0u; i < k; ++i) s += xr[i] * wc[i];
                    yr[c] = s + (has_bias != 0u ? bias[c] : 0.0);
                }
                continue;
            }
            uint c = c0;
            for (; c1 - c >= 4u; c += 4u) {
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
    if (MR == 6 && nrows > kMC) {
        blis_f64_large<MR, NR>(
            x, x_stride, w, w_stride, bias, has_bias, y, y_stride, r0, r1, c0, c1, k, aligned_w);
        return;
    }

    double4 bp4[VELRI_KC * (NR / 4)];
    double ap[MR * (VELRI_KC + VELRI_AP_PAD_F64)];
    double c_buf[VELRI_MC_BUF_ROWS * NR];
    bp4[0] = double4(0.0); ap[0] = 0.0; c_buf[0] = 0.0;
    Ptr<double> bp = Ptr<double>(&bp4[0]);
    Ptr<double> ap_wr = &ap[0];
    Ptr<double, Access.Read> ap_rd = Ptr<double, Access.Read>(&ap[0]);
    Ptr<double> cb_ptr = &c_buf[0];

    for (uint cb = c0; cb < c1;) {
        uint nr = min(uint(NR), c1 - cb);
        for (uint rb = r0; rb < r1;) {
            uint mc = min(kMC, r1 - rb);
            for (uint k0 = 0u; k0 < k;) {
                uint kc = min(kKC, k - k0);
                pack_b_f64<NR>(w, w_stride, cb, nr, k0, kc, &bp4[0], aligned_w);
                bool last = (k0 + kc == k);
                if (k0 == 0u) {
                    linear_mc_panel_f64<MR, NR, true>(
                        x, x_stride, ap_wr, ap_rd, bp, cb_ptr, y, y_stride, bias, has_bias, rb, mc, cb, nr, k0, kc, last);
                } else {
                    linear_mc_panel_f64<MR, NR, false>(
                        x, x_stride, ap_wr, ap_rd, bp, cb_ptr, y, y_stride, bias, has_bias, rb, mc, cb, nr, k0, kc, last);
                }
                k0 += kc;
            }
            rb += mc;
        }
        cb += nr;
    }
}
