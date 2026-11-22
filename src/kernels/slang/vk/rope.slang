// Vulkan Rotary Position Embedding (RoPE) kernels (F32 & BF16, HALF & INTERLEAVED).
import numeric;
import vk_common;

struct RopeArgs<T : IElement> {
    Ptr<T.Storage, Access.Read> x;
    Ptr<float, Access.Read> cos_tbl;
    Ptr<float, Access.Read> sin_tbl;
    Ptr<T.Storage> y;
    Ptr<uint, Access.Read> step_params; // optional [seq_len, pos_offset, kv_pos] for captured graphs
    uint n_tokens;
    uint n_heads;
    uint head_dim;
    uint rotary_dim;
    uint pos_offset;
    uint tok_sx;
    uint head_sx;
    uint tok_sy;
    uint head_sy;
    uint mode;
};

void rope_gpu_impl<T : IElement, VIO : IVecIO128<T>>(RopeArgs<T> a, uint3 gid, uint3 gtid) {
    uint half_rot = a.rotary_dim >> 1u;
    uint eff_pos = (a.step_params != nullptr) ? a.step_params[1] : a.pos_offset;
    if ((a.mode & 2u) != 0u) {
        uint v_elems = VIO.kElems;
        uint n_vec = half_rot / v_elems;
        uint idx = gid.x * kRowWgSize + gtid.x;
        uint pair = idx / n_vec;
        uint i = idx - pair * n_vec;
        uint t = pair / a.n_heads;
        if (t >= a.n_tokens) return;
        uint h = pair - t * a.n_heads;
        uint64_t cs_off = (uint64_t(eff_pos) + uint64_t(t)) * uint64_t(half_rot);
        Ptr<T.Storage, Access.Read> xh = a.x + uint64_t(t) * uint64_t(a.tok_sx) + uint64_t(h) * uint64_t(a.head_sx);
        Ptr<T.Storage> yh = a.y + uint64_t(t) * uint64_t(a.tok_sy) + uint64_t(h) * uint64_t(a.head_sy);

        Ptr<uint4, Access.Read> x0_4 = Ptr<uint4, Access.Read>(xh);
        Ptr<uint4, Access.Read> x1_4 = Ptr<uint4, Access.Read>(xh + half_rot);
        Ptr<uint4> y0_4 = Ptr<uint4>(yh);
        Ptr<uint4> y1_4 = Ptr<uint4>(yh + half_rot);
        Ptr<float4, Access.Read> c4 = Ptr<float4, Access.Read>(a.cos_tbl + cs_off);
        Ptr<float4, Access.Read> s4 = Ptr<float4, Access.Read>(a.sin_tbl + cs_off);

        float4 x0l, x0h, x1l, x1h;
        VIO::unpack(load_a16(x0_4 + i), x0l, x0h);
        VIO::unpack(load_a16(x1_4 + i), x1l, x1h);
        if (VIO.kElems == 4u) {
            float4 y0l, y1l;
            rope_rotate_pair(x0l, x1l, load_a16(c4 + i), load_a16(s4 + i), y0l, y1l);
            store_a16(y0_4 + i, VIO::pack(y0l, float4(0.0f)));
            store_a16(y1_4 + i, VIO::pack(y1l, float4(0.0f)));
        } else {
            float4 y0l, y1l, y0h, y1h;
            rope_rotate_pair(x0l, x1l, load_a16(c4 + (i * 2u + 0u)), load_a16(s4 + (i * 2u + 0u)), y0l, y1l);
            rope_rotate_pair(x0h, x1h, load_a16(c4 + (i * 2u + 1u)), load_a16(s4 + (i * 2u + 1u)), y0h, y1h);
            store_a16(y0_4 + i, VIO::pack(y0l, y0h));
            store_a16(y1_4 + i, VIO::pack(y1l, y1h));
        }
        return;
    }

    uint lane = gtid.x & 31u;
    uint grp = gtid.x >> 5u;
    uint t = gid.y;
    uint h = gid.x * 8u + grp;
    if (t >= a.n_tokens || h >= a.n_heads) return;
    uint64_t cs_off = (uint64_t(eff_pos) + uint64_t(t)) * uint64_t(half_rot);
    Ptr<T.Storage, Access.Read> xh = a.x + uint64_t(t) * uint64_t(a.tok_sx) + uint64_t(h) * uint64_t(a.head_sx);
    Ptr<T.Storage> yh = a.y + uint64_t(t) * uint64_t(a.tok_sy) + uint64_t(h) * uint64_t(a.head_sy);

    for (uint i = lane; i < half_rot; i += 32u) {
        float c = a.cos_tbl[cs_off + uint64_t(i)];
        float s = a.sin_tbl[cs_off + uint64_t(i)];
        uint i0 = ((a.mode & 1u) == 0u) ? i : (i << 1u);
        uint i1 = ((a.mode & 1u) == 0u) ? (i + half_rot) : ((i << 1u) + 1u);
        float y0, y1;
        rope_rotate_pair(T::toF32(xh[i0]), T::toF32(xh[i1]), c, s, y0, y1);
        yh[i0] = T::fromF32(y0);
        yh[i1] = T::fromF32(y1);
    }
    for (uint i = a.rotary_dim + lane; i < a.head_dim; i += 32u) {
        yh[i] = xh[i];
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_rope_f32(uniform RopeArgs<ElemF32> a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    rope_gpu_impl<ElemF32, VecIOF32>(a, gid, gtid);
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_rope_bf16(uniform RopeArgs<ElemBF16> a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    rope_gpu_impl<ElemBF16, VecIOBF16>(a, gid, gtid);
}
