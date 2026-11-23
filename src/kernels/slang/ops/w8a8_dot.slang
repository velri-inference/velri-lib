// Signed activation bytes multiply unsigned weight bytes encoding w + 128.
// Subtract 128 * sum(x) once per output, with wrapping 32-bit accumulation.
import epilogue;

static const uint kDirectOrder = 8u;
static const uint kRowBlock = 64;
static const uint kOperandBudget = 32u * 1024u;
static const int kTilesPerPanel = 8;
static const int kPackedTileCols = 6;
static const int kTileRows = 4;
static const int kTileCols = 5;

// Keep additional storage within 1/32 of the logical weights.
[ForceInline] uint packed_stride(uint k) {
    uint padding = (0u - k) & 63u;
    return padding <= k / 32u && k <= 0xffffffffu - padding ? k + padding : k;
}

[DllExport] export __extern_cpp uint velri_slang_dot_packed_stride(uint k) {
    return packed_stride(k);
}

interface IDotOutput {
    void store(uint row, uint col, uint sum);
}

// The panel uses local coordinates. Direct output retains the absolute channel index.
struct PanelOutput<let TILE_COLS : int> : IDotOutput {
    Ptr<uint> sums;

    [ForceInline] void store(uint row, uint col, uint sum) {
        sums[row * uint(kTilesPerPanel * TILE_COLS) + col] = sum;
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
    Output output, uint row, uint col, uint k, uint extended_k, uint remaining_rows)
{
    uint depth = row + uint(R) < remaining_rows ? extended_k : k;
    uint acc[R][C];
    [ForceUnroll] for (int r = 0; r < R; ++r)
        [ForceUnroll] for (int c = 0; c < C; ++c) acc[r][c] = 0;
    for (uint p = 0; p < depth; ++p) {
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

[ForceInline] void dot_rows<let C : int, let PACKED : bool, let TWO_ROW_TAIL : bool, Output : IDotOutput>(
    Ptr<int8_t, Access.Read> x, uint xs, Ptr<int8_t, Access.Read> w, uint ws,
    Output output, uint height, uint col, uint k, uint extended_k, uint remaining_rows)
{
    uint r = 0;
    for (; height - r >= uint(kTileRows); r += uint(kTileRows)) {
        dot_tile<kTileRows, C, PACKED, Output>(x, xs, w, ws, output, r, col, k, extended_k, remaining_rows);
    }
    if (TWO_ROW_TAIL && height - r >= 2u) {
        dot_tile<2, C, PACKED, Output>(x, xs, w, ws, output, r, col, k, extended_k, remaining_rows);
        r += 2u;
    }
    for (; r < height; ++r) {
        dot_tile<1, C, PACKED, Output>(x, xs, w, ws, output, r, col, k, extended_k, remaining_rows);
    }
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
            y[c] = EpQuantScale::apply_scales(float(int(sums[c] - correction)), ep.w_scale[col + c], x_scale);
    } else {
        for (uint c = 0; c < width; ++c)
            y[c] = EpQuantScale::apply(float(int(sums[c] - correction)), row, col + c, ep);
    }
}

// Pass TILE_COLS as a generic argument. Slang 2026.22.1 can reuse the raw
// panel stride for packed output when specializing on a local static constant.
void dot_slice<let PACKED : bool, let TILE_COLS : int>(
    Ptr<int8_t, Access.Read> x, uint xs, Ptr<int8_t, Access.Read> w, uint ws,
    Ptr<float> y, uint ys, uint r0, uint r1, uint c0, uint c1, uint k,
    EpilogueParams ep, bool direct_order)
{
    if (r0 >= r1 || c0 >= c1) return;
    uint panel_cols = uint(kTilesPerPanel * TILE_COLS);
    uint width = c1 - c0;
    bool direct = direct_order || r1 - r0 < uint(kTileRows) || width < uint(kTileCols) ||
                  ys < width || k == 0u;
    // Size the activation block and one weight tile to a 32 KiB operand budget.
    uint row_block = kRowBlock;
    if (!direct && k <= kOperandBudget / uint(2 * kTileRows + TILE_COLS)) {
        if (k <= kOperandBudget / uint(8 * kTileRows + TILE_COLS)) {
            row_block = uint(8 * kTileRows);
        } else if (k <= kOperandBudget / uint(4 * kTileRows + TILE_COLS)) {
            row_block = uint(4 * kTileRows);
        } else {
            row_block = uint(2 * kTileRows);
        }
    }
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
                dot_rows<kTileCols, PACKED, false, DirectOutput>(xb, xs, w, ws, output, height, c, k, k, 0u);
            for (; c < c1; ++c)
                dot_rows<1, PACKED, false, DirectOutput>(xb, xs, w, ws, output, height, c, k, k, 0u);
        } else {
            uint panel[kRowBlock * uint(kTilesPerPanel * TILE_COLS)];
            panel[0] = 0;
            PanelOutput<TILE_COLS> output = {&panel[0]};
            // Zero weight bytes allow a dot to read the beginning of the next
            // contiguous activation row. The final tile stays within the slice.
            uint extended_k = PACKED && xs == k ? ws : k;
            for (uint cb = c0; cb < c1;) {
                uint cols = min(panel_cols, c1 - cb);
                Ptr<int8_t, Access.Read> wb = w + uint64_t(cb) * ws;
                uint c = 0;
                for (; cols - c >= uint(TILE_COLS); c += uint(TILE_COLS))
                    dot_rows<TILE_COLS, PACKED, PACKED, PanelOutput<TILE_COLS>>(
                        xb, xs, wb, ws, output, height, c, k, extended_k, r1 - rb);
                if (PACKED) {
                    if (cols - c >= 4u) {
                        dot_rows<4, true, true, PanelOutput<TILE_COLS>>(
                            xb, xs, wb, ws, output, height, c, k, extended_k, r1 - rb);
                        c += 4u;
                    }
                    if (cols - c >= 2u) {
                        dot_rows<2, true, true, PanelOutput<TILE_COLS>>(
                            xb, xs, wb, ws, output, height, c, k, extended_k, r1 - rb);
                        c += 2u;
                    }
                } else if (cols - c >= 3u) {
                    dot_rows<3, false, false, PanelOutput<TILE_COLS>>(
                        xb, xs, wb, ws, output, height, c, k, extended_k, r1 - rb);
                    c += 3u;
                }
                for (; c < cols; ++c)
                    dot_rows<1, PACKED, PACKED, PanelOutput<TILE_COLS>>(
                        xb, xs, wb, ws, output, height, c, k, extended_k, r1 - rb);
                for (uint r = 0; r < height; ++r)
                    store_row(&panel[r * panel_cols], correction[r], yb + uint64_t(r) * ys + cb,
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
    uint ws = packed_stride(k);
    for (uint c = 0; c < channels; ++c) {
        Ptr<int8_t> row = packed + uint64_t(c) * ws;
        for (uint p = 0; p < k; ++p) row[p] = int8_t(uint(w[uint64_t(c) * stride + p]) ^ 128u);
        for (uint p = k; p < ws; ++p) row[p] = 0;
    }
}

[DllExport] export __extern_cpp void velri_slang_dot_linear_w8a8_f32(
    Ptr<int8_t, Access.Read> x, uint xs, Ptr<int8_t, Access.Read> w, uint ws,
    Ptr<float, Access.Read> bias, Ptr<float, Access.Read> x_scale, Ptr<float, Access.Read> w_scale,
    uint flags, Ptr<float> y, uint ys, uint r0, uint r1, uint c0, uint c1, uint k)
{
    dot_slice<false, kTileCols>(x, xs, w, ws, y, ys, r0, r1, c0, c1, k,
                     make_epilogue_params(bias, x_scale, w_scale, flags), (flags & kDirectOrder) != 0u);
}

[DllExport] export __extern_cpp void velri_slang_dot_linear_w8a8_f32_packed(
    Ptr<int8_t, Access.Read> x, uint xs, Ptr<int8_t, Access.Read> w,
    Ptr<float, Access.Read> bias, Ptr<float, Access.Read> x_scale, Ptr<float, Access.Read> w_scale,
    uint flags, Ptr<float> y, uint ys, uint r0, uint r1, uint c0, uint c1, uint k)
{
    dot_slice<true, kPackedTileCols>(x, xs, w, packed_stride(k), y, ys, r0, r1, c0, c1, k,
                    make_epilogue_params(bias, x_scale, w_scale, flags), (flags & kDirectOrder) != 0u);
}
