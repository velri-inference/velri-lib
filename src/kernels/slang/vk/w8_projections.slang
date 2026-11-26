import vk_common;
import numeric;
import activations;
import epilogue;

int dot_i8x16(uint4 x, uint4 w, int acc) {
    acc = dot4add_i8packed(x.x, w.x, acc);
    acc = dot4add_i8packed(x.y, w.y, acc);
    acc = dot4add_i8packed(x.z, w.z, acc);
    return dot4add_i8packed(x.w, w.w, acc);
}

// A subgroup computes two independent signed dots using each activation load twice.
float2 project_pair(Ptr<uint4, Access.Read> x, Ptr<uint4, Access.Read> first,
                    Ptr<uint4, Access.Read> second, uint k16) {
    int2 sum = 0;
    for (uint i = WaveGetLaneIndex(); i < k16; i += WaveGetLaneCount()) {
        uint4 input = load_a16(x + i);
        sum.x = dot_i8x16(input, load_a16(first + i), sum.x);
        sum.y = dot_i8x16(input, load_a16(second + i), sum.y);
    }
    return float2(WaveActiveSum(sum));
}

struct GatedW8Args {
    Ptr<uint4, Access.Read> x;
    Ptr<uint4, Access.Read> gate_w;
    Ptr<uint4, Access.Read> up_w;
    Ptr<float, Access.Read> x_scale;
    Ptr<float, Access.Read> gate_scale;
    Ptr<float, Access.Read> up_scale;
    Ptr<float> up;
    Ptr<float> y;
    uint m;
    uint k16;
    float clamp_limit;
    uint pad;
};

[shader("compute")] [numthreads(128, 1, 1)]
void vk_gated_w8a8(uniform GatedW8Args a, uint3 gid : SV_GroupID) {
    uint lane = WaveGetLaneIndex(), lanes = WaveGetLaneCount();
    uint ch = gid.x * (kGemvWgSize / lanes) + WaveGetWaveIndex();
    if (ch >= a.m) return;
    uint64_t base = uint64_t(ch) * a.k16;
    float2 totals = project_pair(a.x, a.gate_w + base, a.up_w + base, a.k16);
    if (lane == 0u) {
        float g = EpQuantScale::apply_scales(totals.x, a.gate_scale[ch], a.x_scale[0]);
        float u = EpQuantScale::apply_scales(totals.y, a.up_scale[ch], a.x_scale[0]);
        a.up[ch] = u;
        a.y[ch] = ActSiLU::apply(apply_symmetric_clamp(g, a.clamp_limit)) * apply_symmetric_clamp(u, a.clamp_limit);
    }
}

struct QkvRopeW8Args {
    Ptr<uint4, Access.Read> x;
    Ptr<uint4, Access.Read> qw;
    Ptr<uint4, Access.Read> kw;
    Ptr<uint4, Access.Read> vw;
    Ptr<float, Access.Read> x_scale;
    Ptr<float, Access.Read> q_scale;
    Ptr<float, Access.Read> k_scale;
    Ptr<float, Access.Read> v_scale;
    Ptr<float> q;
    Ptr<float> k;
    Ptr<float> v;
    Ptr<float, Access.Read> cos_tbl;
    Ptr<float, Access.Read> sin_tbl;
    Ptr<uint, Access.Read> step;
    uint q_rows;
    uint kv_rows;
    uint dims; // k16 | (half rotary dimension << 16), to fit the 128-byte push constant range.
    uint pos;
};

[shader("compute")] [numthreads(128, 1, 1)]
void vk_qkv_rope_w8a8(uniform QkvRopeW8Args a, uint3 gid : SV_GroupID) {
    uint lane = WaveGetLaneIndex(), lanes = WaveGetLaneCount();
    uint job = gid.x * (kGemvWgSize / lanes) + WaveGetWaveIndex();
    uint q_pairs = a.q_rows / 2u, kv_pairs = a.kv_rows / 2u;
    bool value = job >= q_pairs + kv_pairs;
    bool query = job < q_pairs;
    uint pair = query ? job : job - q_pairs - (value ? kv_pairs : 0u);
    if (value && pair >= kv_pairs) return;
    uint half_dim = a.dims >> 16u, k16 = a.dims & 65535u;
    uint dim = pair % half_dim, row = (pair / half_dim) * (2u * half_dim) + dim;
    Ptr<uint4, Access.Read> weights = query ? a.qw : (value ? a.vw : a.kw);
    Ptr<uint4, Access.Read> lo = weights + uint64_t(row) * k16;
    Ptr<uint4, Access.Read> hi = lo + uint64_t(half_dim) * k16;
    float2 sum = project_pair(a.x, lo, hi, k16);
    if (lane == 0u) {
        Ptr<float, Access.Read> scale = query ? a.q_scale : (value ? a.v_scale : a.k_scale);
        float2 y = float2(EpQuantScale::apply_scales(sum.x, scale[row], a.x_scale[0]),
                          EpQuantScale::apply_scales(sum.y, scale[row + half_dim], a.x_scale[0]));
        if (!value) {
            uint pos = step_word_or(a.step, 1u, a.pos, a.cos_tbl);
            uint64_t idx = uint64_t(pos) * half_dim + dim;
            float y0, y1;
            rope_rotate_pair(y.x, y.y, a.cos_tbl[idx], a.sin_tbl[idx], y0, y1);
            y = float2(y0, y1);
        }
        Ptr<float> output = query ? a.q : (value ? a.v : a.k);
        output[row] = y.x;
        output[row + half_dim] = y.y;
    }
}
