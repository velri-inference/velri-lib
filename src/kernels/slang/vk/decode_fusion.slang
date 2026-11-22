import vk_common;
import attention_args;
import activations;
import numeric;
import reductions;

struct QkvRopeCacheArgs {
    Ptr<float4, Access.Read> x;
    Ptr<float4, Access.Read> qw;
    Ptr<float4, Access.Read> kw;
    Ptr<float4, Access.Read> vw;
    Ptr<float> q;
    Ptr<float> k;
    Ptr<float> v;
    Ptr<float, Access.Read> c;
    Ptr<float, Access.Read> s;
    Ptr<CacheScalar> k_cache;
    Ptr<CacheScalar> v_cache;
    Ptr<uint, Access.Read> step;
    uint k4;
    uint q_heads;
    uint kv_heads;
    uint head_dim;
    uint max_ctx;
    uint pos;
    uint kv_pos;
    uint pad;
};

// A 256-thread group normalizes the same input row consumed by its projections.
// Group zero also materializes it for other graph consumers and output bindings.
groupshared float4 normalized_input[512];

void cache_rms_input(Ptr<float4, Access.Read> x, Ptr<float4, Access.Read> weight,
                     Ptr<float4> output, uint k4, float eps, uint group) {
    uint tid = WaveGetWaveIndex() * WaveGetLaneCount() + WaveGetLaneIndex();
    float4 values[2], weights[2];
    float partial = 0.0f;
    [ForceUnroll] for (uint j = 0u; j < 2u; ++j) {
        uint i = tid + j * kRowWgSize;
        if (i < k4) {
            values[j] = load_a16(x + i);
            weights[j] = load_a16(weight + i);
            partial += dot(values[j], values[j]);
        }
    }
    float scale = rsqrt(wg_reduce_sum(partial, tid, kRowWgSize) / float(k4 * 4u) + eps);
    [ForceUnroll] for (uint j = 0u; j < 2u; ++j) {
        uint i = tid + j * kRowWgSize;
        if (i < k4) {
            float4 value = values[j] * (scale * weights[j]);
            normalized_input[i] = value;
            if (group == 0u) store_a16(output + i, value);
        }
    }
    GroupMemoryBarrierWithGroupSync();
}

struct ProjectionNorm {
    Ptr<float4, Access.Read> weight;
    Ptr<float4> output;
    float eps;
    uint group;
};

float2 dot_rows<let NORMALIZED : bool>(Ptr<float4, Access.Read> x, Ptr<float4, Access.Read> first,
                Ptr<float4, Access.Read> second, uint k4, ProjectionNorm norm, bool valid) {
    uint lane = WaveGetLaneIndex();
    uint lanes = WaveGetLaneCount();
    uint i = lane;
    float2 sums = 0.0f;
    if (NORMALIZED) {
        // Start streaming projection weights while the workgroup normalizes its input.
        float4 first_weight = 0.0f, second_weight = 0.0f;
        if (valid && lane < k4) {
            first_weight = load_a16(first + lane);
            second_weight = load_a16(second + lane);
        }
        // The final workgroup may contain subgroups with no projection row.
        cache_rms_input(x, norm.weight, norm.output, k4, norm.eps, norm.group);
        if (valid && lane < k4) {
            float4 input = normalized_input[lane];
            sums = float2(dot(input, first_weight), dot(input, second_weight));
        }
        i += lanes;
    }
    if (!valid) return 0.0f;
    for (; i < k4; i += lanes) {
        float4 input = NORMALIZED ? normalized_input[i] : load_a16(x + i);
        sums.x += dot(input, load_a16(first + i));
        sums.y += dot(input, load_a16(second + i));
    }
    return WaveActiveSum(sums);
}

uint subgroup_job(uint group, uint threads) {
    return group * (threads / WaveGetLaneCount()) + WaveGetWaveIndex();
}

// One subgroup projects both halves of a rotary pair. The original Q, K and V
// outputs remain available to other consumers in addition to the two KV caches.
void project_qkv<let NORMALIZED : bool>(QkvRopeCacheArgs a, uint job, ProjectionNorm norm = {}) {
    uint half_dim = a.head_dim / 2u;
    uint q_pairs = a.q_heads * half_dim;
    uint kv_pairs = a.kv_heads * half_dim;
    bool cache = job >= q_pairs;
    bool value = job >= q_pairs + kv_pairs;
    uint pair = value ? job - q_pairs - kv_pairs : (cache ? job - q_pairs : job);
    bool valid = !value || pair < kv_pairs;

    uint head = pair / half_dim;
    uint dim = pair % half_dim;
    uint col = head * a.head_dim + dim;
    Ptr<float4, Access.Read> weights = value ? a.vw : (cache ? a.kw : a.qw);
    Ptr<float4, Access.Read> first = weights + uint64_t(col) * a.k4;
    float2 projected = dot_rows<NORMALIZED>(a.x, first, first + uint64_t(half_dim) * a.k4, a.k4, norm, valid);
    if (valid && WaveGetLaneIndex() == 0u) {
        if (!value) {
            uint pos = step_word_or(a.step, 1u, a.pos, a.c);
            uint64_t index = uint64_t(pos) * half_dim + dim;
            float lo, hi;
            rope_rotate_pair(projected.x, projected.y, a.c[index], a.s[index], lo, hi);
            projected = float2(lo, hi);
        }
        Ptr<float> dst = value ? a.v : (cache ? a.k : a.q);
        dst[col] = projected.x;
        dst[col + half_dim] = projected.y;
        uint kv_pos = step_word_or(a.step, 2u, a.kv_pos, a.c);
        if (cache && kv_pos < a.max_ctx) {
            Ptr<CacheScalar> kv = value ? a.v_cache : a.k_cache;
            uint64_t offset = (uint64_t(head) * a.max_ctx + kv_pos) * a.head_dim + dim;
            kv[offset] = CacheScalar(projected.x);
            kv[offset + half_dim] = CacheScalar(projected.y);
        }
    }
}

[shader("compute")] [numthreads(128, 1, 1)]
void vk_qkv_rope_cache_f32(uniform QkvRopeCacheArgs a, uint3 gid : SV_GroupID) {
    project_qkv<false>(a, subgroup_job(gid.x, kGemvWgSize));
}

// Replay supplies both positions through step. Packing the dimensions keeps the
// entire normalized projection inside Vulkan's 128-byte push constant guarantee.
struct RmsQkvArgs {
    Ptr<float4, Access.Read> x;
    Ptr<float4, Access.Read> qw;
    Ptr<float4, Access.Read> kw;
    Ptr<float4, Access.Read> vw;
    Ptr<float> q;
    Ptr<float> k;
    Ptr<float> v;
    Ptr<float, Access.Read> c;
    Ptr<float, Access.Read> s;
    Ptr<CacheScalar> k_cache;
    Ptr<CacheScalar> v_cache;
    Ptr<uint, Access.Read> step;
    Ptr<float4, Access.Read> norm_weight;
    Ptr<float4> norm_y;
    uint dims;   // k4 | head_dim << 16
    uint heads;  // q_heads | kv_heads << 16
    uint max_ctx;
    float eps;
};

[shader("compute")] [numthreads(256, 1, 1)]
void vk_rms_qkv_f32(uniform RmsQkvArgs a, uint3 gid : SV_GroupID) {
    uint k4 = a.dims & 0xffffu;
    QkvRopeCacheArgs projection = {
        a.x, a.qw, a.kw, a.vw, a.q, a.k, a.v, a.c, a.s, a.k_cache, a.v_cache, a.step,
        k4, a.heads & 0xffffu, a.heads >> 16u, a.dims >> 16u, a.max_ctx, 0u, 0u, 0u
    };
    project_qkv<true>(projection, subgroup_job(gid.x, kRowWgSize),
        { a.norm_weight, a.norm_y, a.eps, gid.x });
}

struct GatedGemvArgs {
    Ptr<float4, Access.Read> x;
    Ptr<float4, Access.Read> gate_w;
    Ptr<float4, Access.Read> up_w;
    Ptr<float> up;
    Ptr<float> y;
    uint m;
    uint k4;
    uint w_stride4;
    float clamp_limit;
    Ptr<float4, Access.Read> norm_weight;
    Ptr<float4> norm_y;
    float eps;
};

void project_gate_up<let NORMALIZED : bool>(GatedGemvArgs a, uint col, ProjectionNorm norm = {}) {
    bool valid = col < a.m;
    uint64_t offset = uint64_t(col) * a.w_stride4;
    float2 projected = dot_rows<NORMALIZED>(a.x, a.gate_w + offset, a.up_w + offset, a.k4, norm, valid);
    if (valid && WaveGetLaneIndex() == 0u) {
        a.up[col] = projected.y;
        a.y[col] = ActSiLU::apply(apply_symmetric_clamp(projected.x, a.clamp_limit))
                   * apply_symmetric_clamp(projected.y, a.clamp_limit);
    }
}

[shader("compute")] [numthreads(128, 1, 1)]
void vk_gated_gemv_f32(uniform GatedGemvArgs a, uint3 gid : SV_GroupID) {
    project_gate_up<false>(a, subgroup_job(gid.x, kGemvWgSize));
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_rms_gated_gemv_f32(uniform GatedGemvArgs a, uint3 gid : SV_GroupID) {
    project_gate_up<true>(a, subgroup_job(gid.x, kRowWgSize),
        { a.norm_weight, a.norm_y, a.eps, gid.x });
}

struct GemvResidualArgs {
    Ptr<float4, Access.Read> x;
    Ptr<float4, Access.Read> w;
    Ptr<float> y;
    Ptr<float> residual;
    uint k4;
    uint m;
};

[shader("compute")] [numthreads(128, 1, 1)]
void vk_gemv_residual_f32(uniform GemvResidualArgs a, uint3 gid : SV_GroupID) {
    uint col = subgroup_job(gid.x, kGemvWgSize);
    if (col >= a.m) return;
    Ptr<float4, Access.Read> weights = a.w + uint64_t(col) * a.k4;
    float sum = 0.0f;
    uint lanes = WaveGetLaneCount();
    uint i = WaveGetLaneIndex();
    for (; i + 2u * lanes < a.k4; i += 3u * lanes) {
        [ForceUnroll] for (uint j = 0u; j < 3u; ++j) {
            uint col = i + j * lanes;
            sum += dot(load_a16(a.x + col), load_a16(weights + col));
        }
    }
    for (; i < a.k4; i += lanes) {
        sum += dot(load_a16(a.x + i), load_a16(weights + i));
    }
    sum = WaveActiveSum(sum);
    if (WaveGetLaneIndex() == 0u) {
        a.y[col] = sum;
        a.residual[col] += sum;
    }
}
