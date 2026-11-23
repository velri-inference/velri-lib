// AVX-512 VNNI W8A8 GEMM. Each packed word holds four consecutive K bytes.
import epilogue;

static const uint kMR = 14;
static const uint kNR = 32;
static const uint kMC = 140;
static const uint kKC = 192;
static const uint kTiles = 4;
static const uint kSignBits = 0x80808080u;

[ForceInline] Ptr<uint> cache_line(Ptr<uint> p)
{
    return Ptr<uint>((uint64_t(p) + 63) & ~uint64_t(63));
}

[ForceInline] uint4 load_words(Ptr<int8_t, Access.Read> src)
{
    __target_switch {
        case llvm: __intrinsic_asm "%result = load <4 x i32>, $0, align 1";
    }
}

[ForceInline] void store_words(Ptr<uint> dst, uint4 words)
{
    __target_switch {
        case llvm: __intrinsic_asm "store $1, $0, align 1";
    }
}

// Slang's CPU dot4add fallback unpacks each byte. Keep the non-saturating
// unsigned-by-signed dot explicit in LLVM IR for both CPU AOT pipelines.
[ForceInline] void dot16(Ptr<uint> acc, uint a, Ptr<uint> b)
{
    __target_switch {
        case llvm:
            __intrinsic_asm R"(
                %c = load <16 x i32>, $0, align 1
                %lane = insertelement <16 x i32> poison, $1, i32 0
                %x = shufflevector <16 x i32> %lane, <16 x i32> poison, <16 x i32> zeroinitializer
                %w = load <16 x i32>, $2, align 1
                %d = call <16 x i32> @llvm.x86.avx512.vpdpbusd.512(<16 x i32> %c, <16 x i32> %x, <16 x i32> %w)
                store <16 x i32> %d, $0, align 1
            )";
    }
}

void pack_quads<let WIDTH : int, let FLIP : uint>(
    Ptr<int8_t, Access.Read> src, uint stride, uint rows, uint kc, Ptr<uint> dst)
{
    // Replicate the last input row or output column for an incomplete tile.
    // Only valid rows and columns reach the output epilogue.
    for (uint r = 0; r < uint(WIDTH); r += 4) {
        Ptr<int8_t, Access.Read> s0 = src + uint64_t(min(r, rows - 1)) * stride;
        Ptr<int8_t, Access.Read> s1 = src + uint64_t(min(r + 1, rows - 1)) * stride;
        Ptr<int8_t, Access.Read> s2 = src + uint64_t(min(r + 2, rows - 1)) * stride;
        Ptr<int8_t, Access.Read> s3 = src + uint64_t(min(r + 3, rows - 1)) * stride;
        uint p = 0;
        Ptr<uint> d = dst + r;
        for (; p + 16 <= kc; p += 16) {
            uint4 v0 = load_words(s0 + p) ^ FLIP;
            uint4 v1 = load_words(s1 + p) ^ FLIP;
            uint4 v2 = load_words(s2 + p) ^ FLIP;
            uint4 v3 = load_words(s3 + p) ^ FLIP;
            store_words(d,             uint4(v0.x, v1.x, v2.x, v3.x));
            store_words(d + WIDTH,     uint4(v0.y, v1.y, v2.y, v3.y));
            store_words(d + WIDTH * 2, uint4(v0.z, v1.z, v2.z, v3.z));
            store_words(d + WIDTH * 3, uint4(v0.w, v1.w, v2.w, v3.w));
            d = d + WIDTH * 4;
        }
        for (; p < kc; p += 4) {
            uint4 words = 0;
            for (uint b = 0; b < min(4u, kc - p); ++b) {
                words |= uint4(uint8_t(s0[p + b]), uint8_t(s1[p + b]),
                               uint8_t(s2[p + b]), uint8_t(s3[p + b])) << (8 * b);
            }
            store_words(d, words ^ FLIP);
            d = d + WIDTH;
        }
    }
}

void microkernel(Ptr<uint> ap, Ptr<uint> bp,
                 uint quads, bool first, Ptr<uint> sums)
{
    uint acc[kMR][kNR];
    [ForceUnroll] for (uint r = 0; r < kMR; ++r)
        [ForceUnroll] for (uint c = 0; c < kNR; ++c)
            acc[r][c] = first ? 0 : sums[r * kNR + c];
    for (uint q = 0; q < quads; ++q) {
        [ForceUnroll] for (uint r = 0; r < kMR; ++r) {
            uint a = ap[r];
            dot16(&acc[r][0], a, bp);
            dot16(&acc[r][16], a, bp + 16);
        }
        ap = ap + 16;
        bp = bp + kNR;
    }
    [ForceUnroll] for (uint r = 0; r < kMR; ++r)
        [ForceUnroll] for (uint c = 0; c < kNR; ++c)
            sums[r * kNR + c] = acc[r][c];
}

[DllExport] export __extern_cpp void velri_slang_linear_w8a8_f32_vnni(
    Ptr<int8_t, Access.Read> x, uint xs, Ptr<int8_t, Access.Read> w, uint ws,
    Ptr<float, Access.Read> bias, Ptr<float, Access.Read> x_scale,
    Ptr<float, Access.Read> w_scale, uint flags, Ptr<float> y, uint ys,
    uint r0, uint r1, uint c0, uint c1, uint k)
{
    static_assert(kMC % kMR == 0, "Each microkernel stores all rows of its register tile");
    static_assert(kKC % 4 == 0, "K panels must hold complete packed words");
    if (r0 >= r1 || c0 >= c1) return;
    EpilogueParams ep = make_epilogue_params(bias, x_scale, w_scale, flags);
    uint ap_buf[kKC / 4 * 16 + 15];
    ap_buf[0] = 0;
    Ptr<uint> ap = cache_line(&ap_buf[0]);
    uint bp_buf[kTiles * kKC / 4 * kNR + 15];
    bp_buf[0] = 0;
    Ptr<uint> bp = cache_line(&bp_buf[0]);
    uint sums_buf[kTiles * kMC * kNR + 15];
    sums_buf[0] = 0;
    Ptr<uint> sums = cache_line(&sums_buf[0]);
    uint correction[kTiles * kNR];

    for (uint cb = c0; cb < c1;) {
        uint nc = min(kTiles * kNR, c1 - cb);
        uint tiles = (nc + kNR - 1) / kNR;
        for (uint rb = r0; rb < r1;) {
            uint mc = min(kMC, r1 - rb);
            for (uint c = 0; c < tiles * kNR; ++c) correction[c] = 0;
            for (uint kb = 0; kb < k;) {
                uint kc = min(kKC, k - kb);
                uint quads = (kc + 3) / 4;
                for (uint t = 0; t < tiles; ++t) {
                    uint col = cb + t * kNR;
                    Ptr<uint> packed = &bp[t * quads * kNR];
                    pack_quads<32, 0>(w + uint64_t(col) * ws + kb, ws, min(kNR, c1 - col), kc, packed);
                    // X was biased by 128. Accumulate 128 * sum(W) once per column.
                    for (uint q = 0; q < quads; ++q) {
                        dot16(&correction[t * kNR], kSignBits, packed + q * kNR);
                        dot16(&correction[t * kNR + 16], kSignBits, packed + q * kNR + 16);
                    }
                }
                for (uint r = 0; r < mc; r += kMR) {
                    pack_quads<16, kSignBits>(x + uint64_t(rb + r) * xs + kb, xs, min(kMR, mc - r), kc, &ap[0]);
                    for (uint t = 0; t < tiles; ++t)
                        microkernel(&ap[0], &bp[t * quads * kNR], quads, kb == 0,
                                    &sums[(t * kMC + r) * kNR]);
                }
                kb += kc;
            }
            for (uint t = 0; t < tiles; ++t) {
                uint col = cb + t * kNR;
                uint width = min(kNR, c1 - col);
                for (uint r = 0; r < mc; ++r) {
                    Ptr<float> yr = y + uint64_t(rb + r) * ys + col;
                    for (uint c = 0; c < width; ++c) {
                        uint dot = k == 0 ? 0 : sums[(t * kMC + r) * kNR + c] - correction[t * kNR + c];
                        yr[c] = EpQuantScale::apply(float(int32_t(dot)), rb + r, col + c, ep);
                    }
                }
            }
            rb += mc;
        }
        cb += nc;
    }
}
