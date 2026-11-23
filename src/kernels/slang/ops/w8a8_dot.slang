// Signed activation bytes multiply unsigned weight bytes encoding w + 128.
// Subtract 128 * sum(x) once per output, with wrapping 32-bit accumulation.
import epilogue;

static const uint kDirectOrder = 8u;
static const uint kRowBlock = 64;
static const uint kColBlock = 40;
static const int kTileRows = 4;
static const int kTileCols = 5;

interface IDotOutput {
    void store(uint row, uint col, uint sum);
}

// The panel uses local coordinates. Direct output retains the absolute channel index.
struct PanelOutput : IDotOutput {
    Ptr<uint> sums;

    [ForceInline] void store(uint row, uint col, uint sum) {
        sums[row * kColBlock + col] = sum;
    }
}

struct DirectOutput : IDotOutput {
    Ptr<float> y;
    uint stride;
    uint first_row;
    Ptr<uint> correction;
    EpilogueParams ep;

    [ForceInline] void store(uint row, uint col, uint sum) {
        float value = float(int(sum - correction[row]));
        y[uint64_t(row) * stride + col] = EpQuantScale::apply(value, first_row + row, col, ep);
    }
}

[ForceInline] void dot_tile<let R : int, let C : int, let PACKED : bool, Output : IDotOutput>(
    Ptr<int8_t, Access.Read> x, uint xs, Ptr<int8_t, Access.Read> w, uint ws,
    Output output, uint row, uint col, uint k)
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
        [ForceUnroll] for (int c = 0; c < C; ++c)
            output.store(row + r, col + c, acc[r][c]);
}

[ForceInline] void dot_rows<let C : int, let PACKED : bool, Output : IDotOutput>(
    Ptr<int8_t, Access.Read> x, uint xs, Ptr<int8_t, Access.Read> w, uint ws,
    Output output, uint height, uint col, uint k)
{
    uint r = 0;
    for (; height - r >= uint(kTileRows); r += uint(kTileRows))
        dot_tile<kTileRows, C, PACKED, Output>(x, xs, w, ws, output, r, col, k);
    for (; r < height; ++r)
        dot_tile<1, C, PACKED, Output>(x, xs, w, ws, output, r, col, k);
}

[ForceInline] void row_corrections(
    Ptr<int8_t, Access.Read> x, uint xs, uint height, uint k, Ptr<uint> correction)
{
    for (uint r = 0; r < height; ++r) {
        Ptr<int8_t, Access.Read> xr = x + uint64_t(r) * xs;
        uint sum = 0;
        for (uint p = 0; p < k; ++p) sum += uint(int(xr[p]));
        correction[r] = sum * 128u;
    }
}

[ForceInline] void store_row(
    Ptr<uint> sums, uint correction, Ptr<float> y, uint width, uint row, uint col, EpilogueParams ep)
{
    if (ep.has_bias == 0u && ep.has_w_scale != 0u && ep.has_x_scale != 0u) {
        float x_scale = ep.x_scale[row];
        for (uint c = 0; c < width; ++c)
            y[c] = float(int(sums[c] - correction)) * (ep.w_scale[col + c] * x_scale);
    } else {
        for (uint c = 0; c < width; ++c)
            y[c] = EpQuantScale::apply(float(int(sums[c] - correction)), row, col + c, ep);
    }
}

void dot_slice<let PACKED : bool>(
    Ptr<int8_t, Access.Read> x, uint xs, Ptr<int8_t, Access.Read> w, uint ws,
    Ptr<float> y, uint ys, uint r0, uint r1, uint c0, uint c1, uint k,
    EpilogueParams ep, bool direct_order)
{
    if (r0 >= r1 || c0 >= c1) return;
    uint width = c1 - c0;
    bool direct = direct_order || r1 - r0 < uint(kTileRows) || width < uint(kTileCols) ||
                  ys < width || k == 0u;
    // Eight activation rows and five weight rows fit in 32 KiB at shorter depths.
    uint row_block = !direct && k <= 32768u / uint(2 * kTileRows + kTileCols)
                   ? uint(2 * kTileRows) : kRowBlock;
    for (uint rb = r0; rb < r1;) {
        uint height = min(row_block, r1 - rb);
        Ptr<int8_t, Access.Read> xb = x + uint64_t(rb) * xs;
        Ptr<float> yb = y + uint64_t(rb) * ys;
        uint correction[kRowBlock];
        correction[0] = 0;
        row_corrections(xb, xs, height, k, &correction[0]);

        if (direct) {
            DirectOutput output = {yb, ys, rb, &correction[0], ep};
            uint c = c0;
            for (; c1 - c >= uint(kTileCols); c += uint(kTileCols))
                dot_rows<kTileCols, PACKED, DirectOutput>(xb, xs, w, ws, output, height, c, k);
            for (; c < c1; ++c)
                dot_rows<1, PACKED, DirectOutput>(xb, xs, w, ws, output, height, c, k);
        } else {
            uint panel[kRowBlock * kColBlock];
            panel[0] = 0;
            PanelOutput output = {&panel[0]};
            for (uint cb = c0; cb < c1;) {
                uint cols = min(kColBlock, c1 - cb);
                Ptr<int8_t, Access.Read> wb = w + uint64_t(cb) * ws;
                uint c = 0;
                for (; cols - c >= uint(kTileCols); c += uint(kTileCols))
                    dot_rows<kTileCols, PACKED, PanelOutput>(xb, xs, wb, ws, output, height, c, k);
                if (cols - c >= 3u) {
                    dot_rows<3, PACKED, PanelOutput>(xb, xs, wb, ws, output, height, c, k);
                    c += 3u;
                }
                for (; c < cols; ++c)
                    dot_rows<1, PACKED, PanelOutput>(xb, xs, wb, ws, output, height, c, k);
                for (uint r = 0; r < height; ++r)
                    store_row(&panel[r * kColBlock], correction[r], yb + uint64_t(r) * ys + cb,
                              cols, rb + r, cb, ep);
                cb += cols;
            }
        }
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
                     make_epilogue_params(bias, x_scale, w_scale, flags), (flags & kDirectOrder) != 0u);
}

[DllExport] export __extern_cpp void velri_slang_dot_linear_w8a8_f32_packed(
    Ptr<int8_t, Access.Read> x, uint xs, Ptr<int8_t, Access.Read> w,
    Ptr<float, Access.Read> bias, Ptr<float, Access.Read> x_scale, Ptr<float, Access.Read> w_scale,
    uint flags, Ptr<float> y, uint ys, uint r0, uint r1, uint c0, uint c1, uint k)
{
    dot_slice<true>(x, xs, w, k, y, ys, r0, r1, c0, c1, k,
                    make_epilogue_params(bias, x_scale, w_scale, flags), (flags & kDirectOrder) != 0u);
}
