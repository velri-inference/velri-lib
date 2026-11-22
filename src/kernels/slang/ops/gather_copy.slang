// Shared CPU kernels for Embedding table lookup and CopyLastRow.
module gather_copy;

public void embedding_rows_cpu(
    Ptr<float, Access.Read> table, uint table_stride,
    Ptr<int32_t, Access.Read> ids, uint has_ids,
    uint vocab, uint pos_offset,
    Ptr<float> out, uint out_stride,
    uint r0, uint r1, uint dim)
{
    uint64_t ts64 = uint64_t(table_stride);
    uint64_t os64 = uint64_t(out_stride);
    uint64_t vocab64 = uint64_t(vocab);

    if (has_ids != 0u) {
        for (uint r = r0; r < r1; ++r) {
            Ptr<float> dst = out + uint64_t(r) * os64;
            int32_t raw_id = ids[r];
            if (raw_id >= 0 && uint64_t(uint(raw_id)) < vocab64) {
                Ptr<float, Access.Read> src = table + uint64_t(uint(raw_id)) * ts64;
                for (uint c = 0u; c < dim; ++c) {
                    dst[c] = src[c];
                }
            } else {
                for (uint c = 0u; c < dim; ++c) {
                    dst[c] = 0.0f;
                }
            }
        }
    } else {
        for (uint r = r0; r < r1; ++r) {
            Ptr<float> dst = out + uint64_t(r) * os64;
            uint64_t idx64 = uint64_t(pos_offset) + uint64_t(r);
            if (idx64 < vocab64) {
                Ptr<float, Access.Read> src = table + idx64 * ts64;
                for (uint c = 0u; c < dim; ++c) {
                    dst[c] = src[c];
                }
            } else {
                for (uint c = 0u; c < dim; ++c) {
                    dst[c] = 0.0f;
                }
            }
        }
    }
}

public void copy_last_row_cpu(
    Ptr<float, Access.Read> x, uint x_stride,
    Ptr<float> y, uint src_row, uint dim)
{
    Ptr<float, Access.Read> src = x + uint64_t(src_row) * uint64_t(x_stride);
    for (uint c = 0u; c < dim; ++c) {
        y[c] = src[c];
    }
}
