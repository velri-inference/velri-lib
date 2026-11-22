module attention_args;

import vk_common;

// Cache storage is specialized at build time. Both formats produce F32 outputs.
#ifdef VELRI_F16_KV
public typealias CacheScalar = half;
#else
public typealias CacheScalar = float;
#endif
public typealias CacheVector = vector<CacheScalar, 4>;

public float4 load_cache4(Ptr<CacheVector, Access.Read> p) {
#ifdef VELRI_F16_KV
    CacheVector x = spirv_asm { result : $$CacheVector = OpLoad $p Aligned 8; };
    return float4(x);
#else
    return load_a16(p);
#endif
}

public uint cache_step_pos(Ptr<uint, Access.Read> step, uint fallback, Ptr<CacheScalar, Access.Read> key) {
#ifdef VELRI_F16_KV
    // Even a one-element cache is a valid fallback address. Never read a float from two bytes.
    Ptr<uint16_t, Access.Read> dummy = Ptr<uint16_t, Access.Read>(key);
    Ptr<uint16_t, Access.Read> word = Ptr<uint16_t, Access.Read>(step);
    uint lo = (step != nullptr ? word + 4u : dummy)[0];
    uint hi = (step != nullptr ? word + 5u : dummy)[0];
    return step != nullptr ? lo | (hi << 16u) : fallback;
#else
    return step_word_or(step, 2u, fallback, key);
#endif
}

public struct AttentionArgs {
    public Ptr<float, Access.Read> q;
    public Ptr<CacheScalar> k_cache;
    public Ptr<CacheScalar> v_cache;
    public Ptr<float> y;
    public Ptr<uint, Access.Read> step_params; // optional [seq_len, pos_offset, kv_pos] for captured graphs
    public uint rows;
    public uint n_heads;
    public uint n_kv_heads;
    public uint d_k;
    public uint d_v;
    public uint max_ctx;
    public uint kv_pos;
    public uint q_stride;
    public uint o_stride;
    public uint k_head_stride;
    public uint k_row_stride;
    public uint v_head_stride;
    public uint v_row_stride;
    public uint sliding_window;
    public uint causal;
    public float scale;
    public uint value_slices;
    public Ptr<float, Access.Read> k_new;
    public Ptr<float, Access.Read> v_new;
};

public uint attention_kv_pos(AttentionArgs a) {
    // Fused append reads the new key while writing the cache; regular attention only reads it.
#ifdef VELRI_F16_KV
    return cache_step_pos(a.step_params, a.kv_pos, Ptr<CacheScalar, Access.Read>(a.k_cache));
#else
    Ptr<float, Access.Read> key = a.k_new != nullptr ? a.k_new : Ptr<float, Access.Read>(a.k_cache);
    return step_word_or(a.step_params, 2u, a.kv_pos, key);
#endif
}

public struct SplitKvArgs {
    public Ptr<float, Access.Read> q;
    public Ptr<CacheScalar, Access.Read> k_cache;
    public Ptr<CacheScalar, Access.Read> v_cache;
    public Ptr<float> partials; // per (h, sp): [m, l, unnormalized_acc[0..d_v-1]], stride = part_stride
    public Ptr<uint, Access.Read> step_params; // optional [seq_len, pos_offset, kv_pos] for captured graphs
    public uint n_heads;
    public uint n_kv_heads;
    public uint d_k;
    public uint d_v;
    public uint max_ctx;
    public uint kv_pos;
    public uint part_stride;
    public uint k_head_stride;
    public uint k_row_stride;
    public uint v_head_stride;
    public uint v_row_stride;
    public uint sliding_window;
    public uint causal;
    public uint n_splits;
    public uint group_heads;
    public float scale;
};
