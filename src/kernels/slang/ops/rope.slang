// Shared Rotary Position Embedding (RoPE) kernel math for CPU.
// Supports HALF (NeoX/Llama/Qwen) and INTERLEAVED (GPT-J) modes and partial rotary_dim.
module rope;

import numeric;

public void rope_apply_head<T : IElement>(
    Ptr<T.Storage, Access.Read> xh,
    Ptr<T.Storage> yh,
    Ptr<float, Access.Read> cos_row,
    Ptr<float, Access.Read> sin_row,
    uint head_dim,
    uint rotary_dim,
    uint mode)
{
    uint half_rot = rotary_dim >> 1u;
    if (mode == 0u) {
        for (uint i = 0u; i < half_rot; ++i) {
            float x0 = T::toF32(xh[i]);
            float x1 = T::toF32(xh[i + half_rot]);
            float c = cos_row[i];
            float s = sin_row[i];
            yh[i] = T::fromF32(x0 * c - x1 * s);
            yh[i + half_rot] = T::fromF32(x1 * c + x0 * s);
        }
    } else {
        for (uint i = 0u; i < half_rot; ++i) {
            uint idx = i << 1u;
            float x0 = T::toF32(xh[idx]);
            float x1 = T::toF32(xh[idx + 1u]);
            float c = cos_row[i];
            float s = sin_row[i];
            yh[idx] = T::fromF32(x0 * c - x1 * s);
            yh[idx + 1u] = T::fromF32(x1 * c + x0 * s);
        }
    }
    for (uint i = rotary_dim; i < head_dim; ++i) {
        yh[i] = xh[i];
    }
}

public void rope_tokens_cpu<T : IElement>(
    Ptr<T.Storage, Access.Read> x, uint tok_sx, uint head_sx,
    Ptr<float, Access.Read> cos_tbl, Ptr<float, Access.Read> sin_tbl,
    Ptr<T.Storage> y, uint tok_sy, uint head_sy,
    uint t0, uint t1, uint n_heads, uint head_dim, uint rotary_dim,
    uint pos_offset, uint mode)
{
    uint64_t half_rot = uint64_t(rotary_dim >> 1u);
    for (uint t = t0; t < t1; ++t) {
        uint64_t cs_off = (uint64_t(pos_offset) + uint64_t(t)) * half_rot;
        Ptr<float, Access.Read> cr = cos_tbl + cs_off;
        Ptr<float, Access.Read> sr = sin_tbl + cs_off;
        uint64_t x_tok = uint64_t(t) * uint64_t(tok_sx);
        uint64_t y_tok = uint64_t(t) * uint64_t(tok_sy);
        for (uint h = 0u; h < n_heads; ++h) {
            rope_apply_head<T>(
                x + x_tok + uint64_t(h) * uint64_t(head_sx),
                y + y_tok + uint64_t(h) * uint64_t(head_sy),
                cr, sr, head_dim, rotary_dim, mode);
        }
    }
}
