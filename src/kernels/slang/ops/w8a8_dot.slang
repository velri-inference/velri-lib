// Exact W8A8 dot products with one independent reduction per output cell.
// Unsigned weight bytes encode w + 128. Subtract 128 * sum(x) once per cell.
import epilogue;

static const uint kRowBlock = 64;
static const int kTileRows = 4;
static const int kTileCols = 5;

[ForceInline] void dot_tile<let R : int, let C : int, let PACKED : bool>(
    Ptr<int8_t, Access.Read> x, uint xs, Ptr<int8_t, Access.Read> w, uint ws,
    Ptr<float> y, uint ys, uint row, uint col, uint k,
    Ptr<uint> correction, EpilogueParams ep)
{
    uint acc[R][C];
    [ForceUnroll] for (int r = 0; r < R; ++r)
        [ForceUnroll] for (int c = 0; c < C; ++c) acc[r][c] = 0;
    for (uint p = 0; p < k; ++p) {
        int a[R], b[C];
        [ForceUnroll] for (int r = 0; r < R; ++r)
            a[r] = int(x[uint64_t(row + r) * xs + p]);
        [ForceUnroll] for (int c = 0; c < C; ++c) {
            uint8_t byte = uint8_t(w[uint64_t(col + c) * ws + p]);
            b[c] = PACKED ? int(byte) : int(uint8_t(uint(byte) ^ 128u));
        }
        [ForceUnroll] for (int r = 0; r < R; ++r)
            [ForceUnroll] for (int c = 0; c < C; ++c) acc[r][c] += uint(a[r] * b[c]);
    }
    [ForceUnroll] for (int r = 0; r < R; ++r)
        [ForceUnroll] for (int c = 0; c < C; ++c) {
            float sum = float(int(acc[r][c] - correction[r]));
            y[uint64_t(row + r) * ys + col + c] = EpQuantScale::apply(sum, row + r, col + c, ep);
        }
}

[ForceInline] void dot_rows<let C : int, let PACKED : bool>(
    Ptr<int8_t, Access.Read> x, uint xs, Ptr<int8_t, Access.Read> w, uint ws,
    Ptr<float> y, uint ys, uint row, uint height, uint col, uint k,
    Ptr<uint> correction, EpilogueParams ep)
{
    uint r = 0;
    for (; height - r >= uint(kTileRows); r += uint(kTileRows))
        dot_tile<kTileRows, C, PACKED>(x, xs, w, ws, y, ys, row + r, col, k, correction + r, ep);
    for (; r < height; ++r)
        dot_tile<1, C, PACKED>(x, xs, w, ws, y, ys, row + r, col, k, correction + r, ep);
}

void dot_slice<let PACKED : bool>(
    Ptr<int8_t, Access.Read> x, uint xs, Ptr<int8_t, Access.Read> w, uint ws,
    Ptr<float> y, uint ys, uint r0, uint r1, uint c0, uint c1, uint k, EpilogueParams ep)
{
    if (r0 >= r1 || c0 >= c1) return;
    for (uint rb = r0; rb < r1;) {
        uint height = min(kRowBlock, r1 - rb);
        uint correction[kRowBlock];
        correction[0] = 0;
        for (uint r = 0; r < height; ++r) {
            Ptr<int8_t, Access.Read> xr = x + uint64_t(rb + r) * xs;
            uint sum = 0;
            for (uint p = 0; p < k; ++p) sum += uint(int(xr[p]));
            correction[r] = sum * 128u;
        }
        uint c = c0;
        for (; c1 - c >= uint(kTileCols); c += uint(kTileCols))
            dot_rows<kTileCols, PACKED>(x, xs, w, ws, y, ys, rb, height, c, k, &correction[0], ep);
        for (; c < c1; ++c)
            dot_rows<1, PACKED>(x, xs, w, ws, y, ys, rb, height, c, k, &correction[0], ep);
        rb += height;
    }
}

[DllExport] export __extern_cpp void velri_slang_dot_pack_w8a8(
    Ptr<int8_t, Access.Read> w, uint stride, uint channels, uint k, Ptr<int8_t> packed)
{
    for (uint c = 0; c < channels; ++c)
        for (uint p = 0; p < k; ++p)
            packed[uint64_t(c) * k + p] = int8_t(uint(w[uint64_t(c) * stride + p]) ^ 128u);
}

[DllExport] export __extern_cpp void velri_slang_dot_linear_w8a8_f32(
    Ptr<int8_t, Access.Read> x, uint xs, Ptr<int8_t, Access.Read> w, uint ws,
    Ptr<float, Access.Read> bias, Ptr<float, Access.Read> x_scale, Ptr<float, Access.Read> w_scale,
    uint flags, Ptr<float> y, uint ys, uint r0, uint r1, uint c0, uint c1, uint k)
{
    dot_slice<false>(x, xs, w, ws, y, ys, r0, r1, c0, c1, k,
                     make_epilogue_params(bias, x_scale, w_scale, flags));
}

[DllExport] export __extern_cpp void velri_slang_dot_linear_w8a8_f32_packed(
    Ptr<int8_t, Access.Read> x, uint xs, Ptr<int8_t, Access.Read> w,
    Ptr<float, Access.Read> bias, Ptr<float, Access.Read> x_scale, Ptr<float, Access.Read> w_scale,
    uint flags, Ptr<float> y, uint ys, uint r0, uint r1, uint c0, uint c1, uint k)
{
    dot_slice<true>(x, xs, w, k, y, ys, r0, r1, c0, c1, k,
                    make_epilogue_params(bias, x_scale, w_scale, flags));
}
