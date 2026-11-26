// Signed activation bytes multiply unsigned weight bytes encoding w + 128.
// Subtract 128 * sum(x) once per output, with wrapping 32-bit accumulation.
import epilogue;

static const uint kRowBlock = 128u;
static const uint kRawKC = 4096u;
static const uint kRawTileStride = 4224u;
static const int kTilesPerPanel = 4;
static const int kTileRows = 3;
static const int kTileCols = 8;
static const uint kPanelCols = uint(kTilesPerPanel * kTileCols);

[ForceInline] uint padded_depth(uint k) {
    uint padding = (0u - k) & 63u;
    return k <= 0xffffffffu - padding ? k + padding : k;
}

// Keep additional storage within 1/32 of the logical weights, while breaking
// 512/1024/2048/4096-byte L1D set aliasing on power-of-two K dimensions.
[ForceInline] uint packed_stride(uint k) {
    uint padding = (0u - k) & 63u;
    uint s = padding <= k / 32u && k <= 0xffffffffu - padding ? k + padding : k;
    if (s != 0u && (s & 511u) == 0u && (s - k) + 64u <= k / 32u && s <= 0xffffffffu - 64u) {
        s += 64u;
    }
    return s;
}

[DllExport] export __extern_cpp uint velri_slang_dot_packed_stride(uint k) {
    return packed_stride(k);
}

// Exact signed-by-unsigned byte product: [-128, 127] * [0, 255] = [-32640, 32385]
// fits in int16_t without overflow. Narrowing the product avoids GCC 15's
// widen_mult pattern hijacking acc[0][0] into vpdpwssd, emitting 100% vpdpbusd.
[ForceInline] uint byte_mul_u32(int a, int b) {
    return uint(int(int16_t(a * b)));
}

void dot_tail_panel(
    Ptr<int8_t, Access.Read> xr, uint xs, Ptr<int8_t, Access.Read> w, uint ws,
    Ptr<uint> dst, uint R, uint C, uint k0, uint k)
{
    for (uint r = 0u; r < R; ++r) {
        Ptr<int8_t, Access.Read> xrow = xr + uint64_t(r) * xs;
        for (uint c = 0u; c < C; ++c) {
            Ptr<int8_t, Access.Read> wc = w + uint64_t(c) * ws;
            uint s = 0u;
            for (uint p = k0; p < k; ++p) s += uint(int(xrow[p]) * int(uint8_t(wc[p])));
            dst[r * kPanelCols + c] += s;
        }
    }
}

uint dot_tail_scalar(
    Ptr<int8_t, Access.Read> xr, Ptr<int8_t, Access.Read> wc, uint k0, uint k, bool packed)
{
    uint s = 0u;
    for (uint p = k0; p < k; ++p) {
        uint8_t byte = uint8_t(wc[p]);
        int b = packed ? int(byte) : int(uint8_t(uint(byte) ^ 128u));
        s += uint(int(xr[p]) * b);
    }
    return s;
}

// Panel-mode microkernel: 6 ABI-register arguments (zero stack args, zero imul),
// k64-aligned main loop (zero YMM epilogue spills), 100% vpdpbusd.
void dot_tile_panel<let R : int, let C : int>(
    Ptr<int8_t, Access.Read> xr, uint xs, Ptr<int8_t, Access.Read> w, uint ws,
    Ptr<uint> dst, uint depth)
{
    uint k64 = depth & ~63u;
    uint acc[R][C];
    [ForceUnroll] for (int r = 0; r < R; ++r)
        [ForceUnroll] for (int c = 0; c < C; ++c) acc[r][c] = 0u;
    for (uint p = 0u; p < k64; ++p) {
        int a[R], b[C];
        [ForceUnroll] for (int r = 0; r < R; ++r)
            a[r] = int(xr[uint64_t(r) * xs + p]);
        [ForceUnroll] for (int c = 0; c < C; ++c)
            b[c] = int(uint8_t(w[uint64_t(c) * ws + p]));
        [ForceUnroll] for (int r = 0; r < R; ++r)
            [ForceUnroll] for (int c = 0; c < C; ++c) acc[r][c] += byte_mul_u32(a[r], b[c]);
    }
    [ForceUnroll] for (int r = 0; r < R; ++r)
        [ForceUnroll] for (int c = 0; c < C; ++c)
            dst[uint(r) * kPanelCols + uint(c)] = acc[r][c];
    if (k64 < depth) {
        dot_tail_panel(xr, xs, w, ws, dst, uint(R), uint(C), k64, depth);
    }
}

[ForceInline] void dot_tile_step<let R : int, let C : int>(
    Ptr<int8_t, Access.Read> x, uint xs, Ptr<int8_t, Access.Read> w, uint ws,
    Ptr<uint> sums, uint r, uint dst_col, uint k, uint extended_k, uint remaining_rows, bool accum)
{
    uint depth = r + uint(R) < remaining_rows ? extended_k : k;
    Ptr<int8_t, Access.Read> xr = x + uint64_t(r) * xs;
    Ptr<uint> dst = sums + r * kPanelCols + dst_col;
    if (!accum) {
        dot_tile_panel<R, C>(xr, xs, w, ws, dst, depth);
    } else {
        uint tmp[R * int(kPanelCols)];
        tmp[0] = 0u;
        dot_tile_panel<R, C>(xr, xs, w, ws, &tmp[0], depth);
        [ForceUnroll] for (int rr = 0; rr < R; ++rr)
            [ForceUnroll] for (int cc = 0; cc < C; ++cc)
                dst[uint(rr) * kPanelCols + uint(cc)] += tmp[uint(rr) * kPanelCols + uint(cc)];
    }
}

void dot_rows_panel<let C : int>(
    Ptr<int8_t, Access.Read> x, uint xs, Ptr<int8_t, Access.Read> w, uint ws,
    Ptr<uint> sums, uint height, uint dst_col,
    uint k, uint extended_k, uint remaining_rows, bool accum)
{
    uint r = 0u;
    for (; height - r >= uint(kTileRows); r += uint(kTileRows)) {
        dot_tile_step<kTileRows, C>(x, xs, w, ws, sums, r, dst_col, k, extended_k, remaining_rows, accum);
    }
    if (height - r >= 2u) {
        dot_tile_step<2, C>(x, xs, w, ws, sums, r, dst_col, k, extended_k, remaining_rows, accum);
        r += 2u;
    }
    if (r < height) {
        dot_tile_step<1, C>(x, xs, w, ws, sums, r, dst_col, k, extended_k, remaining_rows, accum);
    }
}

// Direct-output single-row tile used for decode GEMV (height < 3) and overlap-safe direct order.
void dot_tile_direct1<let C : int, let PACKED : bool>(
    Ptr<int8_t, Access.Read> xr, Ptr<int8_t, Access.Read> w, uint ws,
    Ptr<float> yr, uint abs_row, uint dst_col, uint correction, uint k, EpilogueParams ep)
{
    uint k64 = k & ~63u;
    uint acc[C];
    [ForceUnroll] for (int c = 0; c < C; ++c) acc[c] = 0u;
    for (uint p = 0u; p < k64; ++p) {
        int a = int(xr[p]);
        [ForceUnroll] for (int c = 0; c < C; ++c) {
            uint8_t byte = uint8_t(w[uint64_t(c) * ws + p]);
            int b = PACKED ? int(byte) : int(uint8_t(uint(byte) ^ 128u));
            acc[c] += byte_mul_u32(a, b);
        }
    }
    if (k64 < k) {
        [ForceUnroll] for (int c = 0; c < C; ++c) {
            acc[c] += dot_tail_scalar(xr, w + uint64_t(c) * ws, k64, k, PACKED);
        }
    }
    [ForceUnroll] for (int c = 0; c < C; ++c) {
        float value = float(int(acc[c] - correction));
        yr[dst_col + uint(c)] = EpQuantScale::apply(value, abs_row, dst_col + uint(c), ep);
    }
}

void pack_raw_tile<let C : int>(
    Ptr<int8_t, Access.Read> src, uint ws, uint kc, Ptr<int8_t> dst, uint tile_ws)
{
    uint rem = (0u - kc) & 63u;
    [ForceUnroll] for (int c = 0; c < C; ++c) {
        Ptr<int8_t, Access.Read> sr = src + uint64_t(c) * ws;
        Ptr<int8_t> dr = dst + uint64_t(c) * tile_ws;
        for (uint p = 0u; p < kc; ++p) dr[p] = int8_t(uint(sr[p]) ^ 128u);
        for (uint p = 0u; p < rem; ++p) dr[kc + p] = int8_t(0);
    }
}

void row_corrections(
    Ptr<int8_t, Access.Read> x, uint xs, uint height, uint k, Ptr<uint> correction, uint8_t scale128)
{
    int sc = int(scale128);
    for (uint r = 0u; r < height; ++r) {
        Ptr<int8_t, Access.Read> xr = x + uint64_t(r) * xs;
        uint sum = 0u;
        for (uint p = 0u; p < k; ++p) sum += byte_mul_u32(int(xr[p]), sc);
        correction[r] = sum;
    }
}

// Keep w_scale/x_scale as direct pointer locals so -fno-strict-aliasing does not
// assume stores to y can alias the EpilogueParams struct and inhibit vectorization.
[ForceInline] void store_panel(
    Ptr<uint> panel, Ptr<uint> correction, Ptr<float> yb, uint ys,
    uint height, uint cols, uint rb, uint cb, EpilogueParams ep)
{
    if (ep.has_bias == 0u && ep.has_w_scale != 0u && ep.has_x_scale != 0u) {
        Ptr<float, Access.Read> ws_ptr = ep.w_scale + cb;
        Ptr<float, Access.Read> xs_ptr = ep.x_scale + rb;
        for (uint r = 0u; r < height; ++r) {
            Ptr<uint> srow = panel + r * kPanelCols;
            Ptr<float> yrow = yb + uint64_t(r) * ys + cb;
            uint corr = correction[r];
            float x_scale = xs_ptr[r];
            for (uint c = 0u; c < cols; ++c)
                yrow[c] = float(int(srow[c] - corr)) * (ws_ptr[c] * x_scale);
        }
    } else {
        for (uint r = 0u; r < height; ++r) {
            Ptr<uint> srow = panel + r * kPanelCols;
            Ptr<float> yrow = yb + uint64_t(r) * ys + cb;
            uint corr = correction[r];
            for (uint c = 0u; c < cols; ++c)
                yrow[c] = EpQuantScale::apply(float(int(srow[c] - corr)), rb + r, cb + c, ep);
        }
    }
}

void run_raw_l1_panel(
    Ptr<int8_t, Access.Read> xk, uint xs, Ptr<int8_t, Access.Read> wk, uint ws,
    Ptr<int8_t> wt_wr, Ptr<int8_t, Access.Read> wt_rd,
    Ptr<uint> panel, uint height, uint cols, uint kc, uint ext_kc,
    uint tile_ws, uint remaining_rows, bool accum)
{
    uint c = 0u;
    for (; cols - c >= uint(kTileCols); c += uint(kTileCols)) {
        pack_raw_tile<kTileCols>(wk + uint64_t(c) * ws, ws, kc, wt_wr, tile_ws);
        dot_rows_panel<kTileCols>(
            xk, xs, wt_rd, tile_ws, panel, height, c, kc, ext_kc, remaining_rows, accum);
    }
    if (cols - c >= 4u) {
        pack_raw_tile<4>(wk + uint64_t(c) * ws, ws, kc, wt_wr, tile_ws);
        dot_rows_panel<4>(
            xk, xs, wt_rd, tile_ws, panel, height, c, kc, ext_kc, remaining_rows, accum);
        c += 4u;
    }
    if (cols - c >= 2u) {
        pack_raw_tile<2>(wk + uint64_t(c) * ws, ws, kc, wt_wr, tile_ws);
        dot_rows_panel<2>(
            xk, xs, wt_rd, tile_ws, panel, height, c, kc, ext_kc, remaining_rows, accum);
        c += 2u;
    }
    for (; c < cols; ++c) {
        pack_raw_tile<1>(wk + uint64_t(c) * ws, ws, kc, wt_wr, tile_ws);
        dot_rows_panel<1>(
            xk, xs, wt_rd, tile_ws, panel, height, c, kc, ext_kc, remaining_rows, accum);
    }
}

void dot_slice<let PACKED : bool>(
    Ptr<int8_t, Access.Read> x, uint xs, Ptr<int8_t, Access.Read> w, uint ws,
    Ptr<float> y, uint ys, uint r0, uint r1, uint c0, uint c1, uint k,
    EpilogueParams ep, uint flags)
{
    if (r0 >= r1 || c0 >= c1) return;
    bool direct_order = (flags & kLinearFlagDirectOrder) != 0u;
    uint width = c1 - c0;
    uint total_rows = r1 - r0;
    bool direct = direct_order || total_rows < uint(kTileRows) || width < 4u || ys < width || k == 0u;
    // Upper flag byte (bits 24..31) is always zero from cpu_backend.c, giving GCC a full
    // uint8_t range [0, 255] so row_corrections lowers directly to 1-insn vpdpbusd loops.
    uint8_t scale128 = uint8_t(128u + (flags >> 24u));

    if (direct) {
        bool stream_rows = w8a8_should_stream_rows(direct_order, width, k);
        for (uint r = r0; r < r1; ++r) {
            Ptr<int8_t, Access.Read> xr = x + uint64_t(r) * xs;
            Ptr<float> yr = y + uint64_t(r) * ys;
            uint corr = 0u;
            row_corrections(xr, xs, 1u, k, &corr, scale128);
            uint c = c0;
            if (!stream_rows) {
                for (; c1 - c >= uint(kTileCols); c += uint(kTileCols))
                    dot_tile_direct1<kTileCols, PACKED>(xr, w + uint64_t(c) * ws, ws, yr, r, c, corr, k, ep);
                if (c1 - c >= 4u) {
                    dot_tile_direct1<4, PACKED>(xr, w + uint64_t(c) * ws, ws, yr, r, c, corr, k, ep);
                    c += 4u;
                }
            }
            for (; c < c1; ++c)
                dot_tile_direct1<1, PACKED>(xr, w + uint64_t(c) * ws, ws, yr, r, c, corr, k, ep);
        }
        return;
    }

    uint k64 = padded_depth(k);
    uint packed_ext_k = (PACKED && xs == k && k >= 64u && ws >= k64) ? k64 : k;

    for (uint rb = r0; rb < r1;) {
        uint height = min(kRowBlock, r1 - rb);
        Ptr<int8_t, Access.Read> xb = x + uint64_t(rb) * xs;
        Ptr<float> yb = y + uint64_t(rb) * ys;
        uint correction[kRowBlock];
        correction[0] = 0u;
        row_corrections(xb, xs, height, k, &correction[0], scale128);

        uint panel[kRowBlock * kPanelCols];
        panel[0] = 0u;

        if (PACKED) {
            for (uint cb = c0; cb < c1;) {
                uint cols = min(kPanelCols, c1 - cb);
                Ptr<int8_t, Access.Read> wb = w + uint64_t(cb) * ws;
                uint c = 0u;
                for (; cols - c >= uint(kTileCols); c += uint(kTileCols))
                    dot_rows_panel<kTileCols>(
                        xb, xs, wb + uint64_t(c) * ws, ws, &panel[0], height, c,
                        k, packed_ext_k, r1 - rb, false);
                if (cols - c >= 4u) {
                    dot_rows_panel<4>(
                        xb, xs, wb + uint64_t(c) * ws, ws, &panel[0], height, c,
                        k, packed_ext_k, r1 - rb, false);
                    c += 4u;
                }
                if (cols - c >= 2u) {
                    dot_rows_panel<2>(
                        xb, xs, wb + uint64_t(c) * ws, ws, &panel[0], height, c,
                        k, packed_ext_k, r1 - rb, false);
                    c += 2u;
                }
                for (; c < cols; ++c)
                    dot_rows_panel<1>(
                        xb, xs, wb + uint64_t(c) * ws, ws, &panel[0], height, c,
                        k, packed_ext_k, r1 - rb, false);
                store_panel(&panel[0], &correction[0], yb, ys, height, cols, rb, cb, ep);
                cb += cols;
            }
        } else {
            int8_t w_tile[kTileCols * kRawTileStride];
            w_tile[0] = int8_t(0);
            Ptr<int8_t> wt_wr = &w_tile[0];
            Ptr<int8_t, Access.Read> wt_rd = Ptr<int8_t, Access.Read>(&w_tile[0]);
            for (uint cb = c0; cb < c1;) {
                uint cols = min(kPanelCols, c1 - cb);
                Ptr<int8_t, Access.Read> wb = w + uint64_t(cb) * ws;
                for (uint k0 = 0u; k0 < k; k0 += kRawKC) {
                    uint kc = min(kRawKC, k - k0);
                    uint kc64 = kc + ((0u - kc) & 63u);
                    uint tile_ws = (kc64 & 511u) == 0u ? kc64 + 64u : kc64;
                    uint ext_kc = (xs == k && k >= 64u) ? kc64 : kc;
                    run_raw_l1_panel(
                        xb + uint64_t(k0), xs, wb + uint64_t(k0), ws, wt_wr, wt_rd,
                        &panel[0], height, cols, kc, ext_kc, tile_ws,
                        r1 - rb, k0 != 0u);
                }
                store_panel(&panel[0], &correction[0], yb, ys, height, cols, rb, cb, ep);
                cb += cols;
            }
        }
        rb += height;
    }
}

[DllExport] export __extern_cpp void velri_slang_dot_pack_w8a8(
    Ptr<int8_t, Access.Read> w, uint stride, uint channels, uint k, Ptr<int8_t> packed)
{
    uint ws = packed_stride(k);
    for (uint c = 0u; c < channels; ++c) {
        Ptr<int8_t> row = packed + uint64_t(c) * ws;
        for (uint p = 0u; p < k; ++p) row[p] = int8_t(uint(w[uint64_t(c) * stride + p]) ^ 128u);
        for (uint p = k; p < ws; ++p) row[p] = int8_t(0);
    }
}

[DllExport] export __extern_cpp void velri_slang_dot_linear_w8a8_f32(
    Ptr<int8_t, Access.Read> x, uint xs, Ptr<int8_t, Access.Read> w, uint ws,
    Ptr<float, Access.Read> bias, Ptr<float, Access.Read> x_scale, Ptr<float, Access.Read> w_scale,
    uint flags, Ptr<float> y, uint ys, uint r0, uint r1, uint c0, uint c1, uint k)
{
    dot_slice<false>(x, xs, w, ws, y, ys, r0, r1, c0, c1, k,
                     make_epilogue_params(bias, x_scale, w_scale, flags), flags);
}

[DllExport] export __extern_cpp void velri_slang_dot_linear_w8a8_f32_packed(
    Ptr<int8_t, Access.Read> x, uint xs, Ptr<int8_t, Access.Read> w,
    Ptr<float, Access.Read> bias, Ptr<float, Access.Read> x_scale, Ptr<float, Access.Read> w_scale,
    uint flags, Ptr<float> y, uint ys, uint r0, uint r1, uint c0, uint c1, uint k)
{
    dot_slice<true>(x, xs, w, packed_stride(k), y, ys, r0, r1, c0, c1, k,
                    make_epilogue_params(bias, x_scale, w_scale, flags), flags);
}
