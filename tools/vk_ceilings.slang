// Hardware Ceiling Calibration Shaders (compiled ONLY when VELRI_BUILD_BENCHMARKS=ON).
import vk_common;

struct CeilingBwArgs {
    Ptr<float4, Access.Read> src;
    Ptr<float4> dst;
    uint count_vec4;
};

[shader("compute")] [numthreads(256, 1, 1)]
void vk_ceiling_bw_f32x4(uniform CeilingBwArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint base = gid.x * 1024u + gtid.x;
    [ForceUnroll] for (uint u = 0u; u < 4u; ++u) {
        uint i = base + u * 256u;
        if (i < a.count_vec4) {
            store_a16(a.dst + i, load_a16(a.src + i));
        }
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_ceiling_bw_read_f32x4(uniform CeilingBwArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint stride = 2048u * 256u;
    float4 s = float4(0.0f);
    for (uint i = gid.x * 256u + gtid.x; i < a.count_vec4; i += stride) {
        s += load_a16(a.src + i);
    }
    float r = WaveActiveSum(s.x + s.y + s.z + s.w);
    if (WaveGetLaneIndex() == 0u) {
        uint lane_cnt = WaveGetLaneCount();
        uint warp_global = gid.x * (256u / lane_cnt) + (gtid.x / lane_cnt);
        Ptr<float> sink = Ptr<float>(a.dst);
        sink[warp_global & 16383u] = r;
    }
}

struct CeilingComputeArgs {
    Ptr<float> sink;
    uint iters;
    float seed;
};

[shader("compute")] [numthreads(256, 1, 1)]
void vk_ceiling_fma_f32(uniform CeilingComputeArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    float base = a.seed + float(gtid.x) * 1e-4f;
    float4 c0 = float4(base + 0.01f), c1 = float4(base + 0.02f);
    float4 c2 = float4(base + 0.03f), c3 = float4(base + 0.04f);
    float4 m0 = float4(0.999991f), m1 = float4(0.000013f);
    for (uint i = 0u; i < a.iters; ++i) {
        [ForceUnroll] for (uint u = 0u; u < 16u; ++u) {
            c0 = fma(c0, m0, m1);
            c1 = fma(c1, m0, m1);
            c2 = fma(c2, m0, m1);
            c3 = fma(c3, m0, m1);
        }
    }
    float4 s = c0 + c1 + c2 + c3;
    float r = WaveActiveSum(s.x + s.y + s.z + s.w);
    if (WaveGetLaneIndex() == 0u) {
        uint lane_cnt = WaveGetLaneCount();
        uint warp_global = gid.x * (256u / lane_cnt) + (gtid.x / lane_cnt);
        a.sink[warp_global & 4095u] = r;
    }
}

void ceiling_coopmat_impl<TMat : ICoopElement, TAcc : ICoopElement, let K_STEP : int>(
    CeilingComputeArgs a, TMat va, TMat vb, TAcc zero_acc, uint3 gtid, uint3 gid)
{
    var mA = linalg::CoopMat<TMat, MemoryScope.Subgroup, 16, K_STEP, linalg::CoopMatMatrixUse.MatrixA>(va);
    var mB = linalg::CoopMat<TMat, MemoryScope.Subgroup, K_STEP, 16, linalg::CoopMatMatrixUse.MatrixB>(vb);
    linalg::CoopMat<TAcc, MemoryScope.Subgroup, 16, 16, linalg::CoopMatMatrixUse.MatrixAccumulator> c[8];
    [ForceUnroll] for (uint j = 0u; j < 8u; ++j)
        c[j] = linalg::CoopMat<TAcc, MemoryScope.Subgroup, 16, 16, linalg::CoopMatMatrixUse.MatrixAccumulator>(zero_acc);
    for (uint i = 0u; i < a.iters; ++i) {
        [ForceUnroll] for (uint j = 0u; j < 8u; ++j) {
            c[j] = linalg::coopMatMulAdd<TAcc, false>(mA, mB, c[j]);
        }
    }
    uint lane_cnt = WaveGetLaneCount();
    uint warp_global = gid.x * (256u / lane_cnt) + (gtid.x / lane_cnt);
    Ptr<TAcc> sink_ptr = Ptr<TAcc>(a.sink);
    [ForceUnroll] for (uint j = 0u; j < 8u; ++j) {
        c[j].Store<linalg::CoopMatMatrixLayout.RowMajor>(sink_ptr, ((warp_global & 15u) * 8u + j) * 256u, 16u);
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_ceiling_coopmat_f16(uniform CeilingComputeArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    ceiling_coopmat_impl<half, float, 16>(a, half(a.seed), half(0.999f), 0.0f, gtid, gid);
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_ceiling_coopmat_bf16(uniform CeilingComputeArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    ceiling_coopmat_impl<BFloat16, float, 16>(a, BFloat16(a.seed), BFloat16(0.999f), 0.0f, gtid, gid);
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_ceiling_coopmat_i8(uniform CeilingComputeArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    int s = int(a.seed);
    ceiling_coopmat_impl<int8_t, int32_t, 32>(a, int8_t(s), int8_t(s + 1), int32_t(0), gtid, gid);
}
