// Shared attention key-index iterators for CPU and Vulkan Slang kernels.
// Compile-time IKeyIter enables zero-overhead dense, sliding-window, and future
// paged / block-sparse KV traversal via one shared attention core.
module key_iter;

public interface IKeyIter {
    uint count();
    uint key_index(uint i);
}

public struct DenseRangeIter : IKeyIter {
    public uint start;
    public uint len;

    public uint count() { return len; }
    public uint key_index(uint i) { return start + i; }
}

public DenseRangeIter make_dense_range_iter(
    uint kv_pos, uint q_row, uint active_rows,
    uint max_ctx, uint sliding_window, uint causal)
{
    uint64_t kv_end64 = (causal != 0u)
                            ? (uint64_t(kv_pos) + uint64_t(q_row) + 1ull)
                            : (uint64_t(kv_pos) + uint64_t(active_rows));
    uint kv_end = uint(min(kv_end64, uint64_t(max_ctx)));
    uint kv_start = (sliding_window > 0u && kv_end > sliding_window)
                        ? (kv_end - sliding_window)
                        : 0u;
    DenseRangeIter it;
    it.start = kv_start;
    it.len = (kv_end > kv_start) ? (kv_end - kv_start) : 0u;
    return it;
}
