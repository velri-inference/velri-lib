// Vulkan Cooperative Matrix Shared-Memory GEMM (FP16, BF16, W8A8 x {128x128, 64x64}).
// Requires 32-lane subgroups (gated on subgroup_size == 32 in vk_backend.c).
import numeric;
import epilogue;
import vk_common;

struct CoopGemmArgs<TMat, TY> {
    Ptr<TMat> x;
    Ptr<TMat> w;
    Ptr<float, Access.Read> bias;
    Ptr<float, Access.Read> x_scale;
    Ptr<float, Access.Read> w_scale;
    Ptr<TY> y;
    uint rows;
    uint m;
    uint k;
    uint x_stride;
    uint w_stride;
    uint y_stride;
    uint flags;
    uint split_k;
};

groupshared uint4 s_cm128_a[128 * 5];
groupshared uint4 s_cm128_b[128 * 5];
groupshared uint4 s_cm64_a[64 * 5];
groupshared uint4 s_cm64_b[64 * 5];

uint4 load_u4_safe<let ELEM_BYTES : int>(
    Ptr<uint8_t, Access.Read> base_b, uint row, uint row_lim,
    uint k_elem, uint k_lim, uint stride_elems, bool aligned16)
{
    if (row >= row_lim || k_elem >= k_lim) return uint4(0u);
    uint elems_per_u4 = uint(16 / ELEM_BYTES);
    uint64_t byte_off = (uint64_t(row) * uint64_t(stride_elems) + uint64_t(k_elem)) * uint64_t(ELEM_BYTES);
    if (aligned16 && (k_elem + elems_per_u4 <= k_lim)) {
        return load_a16(Ptr<uint4, Access.Read>(base_b + byte_off));
    }
    uint max_bytes = min(16u, (k_lim - k_elem) * uint(ELEM_BYTES));
    uint w0 = 0u, w1 = 0u, w2 = 0u, w3 = 0u;
    [ForceUnroll] for (uint b = 0u; b < 16u; ++b) {
        if (b < max_bytes) {
            uint v = uint(base_b[byte_off + uint64_t(b)]) << ((b & 3u) << 3u);
            if (b < 4u) w0 |= v;
            else if (b < 8u) w1 |= v;
            else if (b < 12u) w2 |= v;
            else w3 |= v;
        }
    }
    return uint4(w0, w1, w2, w3);
}

interface ICoopSmemTile<let WG_M : int, let WG_N : int, let WARP_M : int, let WARP_N : int> {
    static void storeA(uint idx, uint4 v);
    static void storeB(uint idx, uint4 v);
    static uint4 readA(uint idx);
    static linalg::CoopMat<T, MemoryScope.Subgroup, 16, K_STEP, linalg::CoopMatMatrixUse.MatrixA>
        loadMatA<T : ICoopElement, let K_STEP : int>(uint off_u4);
    static linalg::CoopMat<T, MemoryScope.Subgroup, K_STEP, 16, linalg::CoopMatMatrixUse.MatrixB>
        loadMatB<T : ICoopElement, let K_STEP : int>(uint off_u4);
    static void storeAcc<TAcc : ICoopElement>(
        linalg::CoopMat<TAcc, MemoryScope.Subgroup, 16, 16, linalg::CoopMatMatrixUse.MatrixAccumulator> mat,
        uint off_u4);
}

#define DEFINE_COOP_SMEM(NAME, SMEM_A, SMEM_B, WGM, WGN, WM, WN)                                  \
struct NAME : ICoopSmemTile<WGM, WGN, WM, WN> {                                                    \
    static void storeA(uint idx, uint4 v) { SMEM_A[idx] = v; }                                     \
    static void storeB(uint idx, uint4 v) { SMEM_B[idx] = v; }                                     \
    static uint4 readA(uint idx) { return SMEM_A[idx]; }                                           \
    static linalg::CoopMat<T, MemoryScope.Subgroup, 16, K_STEP, linalg::CoopMatMatrixUse.MatrixA>  \
        loadMatA<T : ICoopElement, let K_STEP : int>(uint off_u4)                                  \
    {                                                                                              \
        uint lay = 0u, str = 5u;                                                                   \
        return spirv_asm {                                                                         \
            OpExtension "SPV_KHR_cooperative_matrix";                                              \
            OpCapability CooperativeMatrixKHR;                                                     \
            %ptr : $$Ptr<uint4, Access.ReadWrite, AddressSpace.GroupShared> =                      \
                OpAccessChain &SMEM_A $off_u4;                                                     \
            result : $$linalg::CoopMat<T, MemoryScope.Subgroup, 16, K_STEP,                        \
                                       linalg::CoopMatMatrixUse.MatrixA> =                         \
                OpCooperativeMatrixLoadKHR %ptr $lay $str;                                         \
        };                                                                                         \
    }                                                                                              \
    static linalg::CoopMat<T, MemoryScope.Subgroup, K_STEP, 16, linalg::CoopMatMatrixUse.MatrixB>  \
        loadMatB<T : ICoopElement, let K_STEP : int>(uint off_u4)                                  \
    {                                                                                              \
        uint lay = 1u, str = 5u;                                                                   \
        return spirv_asm {                                                                         \
            OpExtension "SPV_KHR_cooperative_matrix";                                              \
            OpCapability CooperativeMatrixKHR;                                                     \
            %ptr : $$Ptr<uint4, Access.ReadWrite, AddressSpace.GroupShared> =                      \
                OpAccessChain &SMEM_B $off_u4;                                                     \
            result : $$linalg::CoopMat<T, MemoryScope.Subgroup, K_STEP, 16,                        \
                                       linalg::CoopMatMatrixUse.MatrixB> =                         \
                OpCooperativeMatrixLoadKHR %ptr $lay $str;                                         \
        };                                                                                         \
    }                                                                                              \
    static void storeAcc<TAcc : ICoopElement>(                                                     \
        linalg::CoopMat<TAcc, MemoryScope.Subgroup, 16, 16,                                        \
                        linalg::CoopMatMatrixUse.MatrixAccumulator> mat,                           \
        uint off_u4)                                                                               \
    {                                                                                              \
        uint lay = 0u, str = 4u;                                                                   \
        spirv_asm {                                                                                \
            OpExtension "SPV_KHR_cooperative_matrix";                                              \
            OpCapability CooperativeMatrixKHR;                                                     \
            %ptr : $$Ptr<uint4, Access.ReadWrite, AddressSpace.GroupShared> =                      \
                OpAccessChain &SMEM_A $off_u4;                                                     \
            OpCooperativeMatrixStoreKHR %ptr $mat $lay $str;                                       \
        };                                                                                         \
    }                                                                                              \
}

DEFINE_COOP_SMEM(CoopSmem128, s_cm128_a, s_cm128_b, 128, 128, 4, 2)
DEFINE_COOP_SMEM(CoopSmem64,  s_cm64_a,  s_cm64_b,   64,  64, 2, 2)

void subgroup_smem_barrier() {
    uint sc = 3u;
    uint sem = 0x108u; // AcquireRelease | WorkgroupMemory
    spirv_asm { OpControlBarrier $sc $sc $sem; };
}

void write_staged_ep_tile<let IS_QUANT : int, Ep : IEpilogue,
                          let WG_M : int, let WG_N : int, let WARP_M : int, let WARP_N : int,
                          S : ICoopSmemTile<WG_M, WG_N, WARP_M, WARP_N>>(
    uint base_u4, Ptr<float> y, uint y_stride, uint rows, uint m,
    uint tr, uint tc, uint lane_id, bool y_aligned4, EpilogueParams ep)
{
    if (tr >= rows || tc >= m) return;
    [ForceUnroll] for (uint u = 0u; u < 2u; ++u) {
        uint idx_u4 = u * 32u + lane_id;
        uint gr = tr + (idx_u4 >> 2u);
        uint gc = tc + ((idx_u4 & 3u) << 2u);
        if (gr >= rows) continue;
        uint4 raw = S::readA(base_u4 + idx_u4);
        float4 fv = (IS_QUANT != 0) ? float4(int4(raw)) : asfloat(raw);
        Ptr<float> yr = y + uint64_t(gr) * uint64_t(y_stride);
        if (y_aligned4 && (gc + 4u <= m)) {
            float4 outv = float4(
                Ep::apply(fv.x, gr, gc + 0u, ep),
                Ep::apply(fv.y, gr, gc + 1u, ep),
                Ep::apply(fv.z, gr, gc + 2u, ep),
                Ep::apply(fv.w, gr, gc + 3u, ep));
            store_a16(Ptr<float4>(yr + gc), outv);
        } else {
            if (gc + 0u < m) yr[gc + 0u] = Ep::apply(fv.x, gr, gc + 0u, ep);
            if (gc + 1u < m) yr[gc + 1u] = Ep::apply(fv.y, gr, gc + 1u, ep);
            if (gc + 2u < m) yr[gc + 2u] = Ep::apply(fv.z, gr, gc + 2u, ep);
            if (gc + 3u < m) yr[gc + 3u] = Ep::apply(fv.w, gr, gc + 3u, ep);
        }
    }
}

void coopmat_mma_step<TMat : ICoopElement, TAcc : ICoopElement, let K_STEP : int,
                      let WG_M : int, let WG_N : int, let WARP_M : int, let WARP_N : int,
                      S : ICoopSmemTile<WG_M, WG_N, WARP_M, WARP_N>>(
    inout linalg::CoopMat<TAcc, MemoryScope.Subgroup, 16, 16, linalg::CoopMatMatrixUse.MatrixAccumulator> acc[WARP_M][WARP_N],
    uint wr, uint wc, uint kk_u4)
{
    linalg::CoopMat<TMat, MemoryScope.Subgroup, 16, K_STEP, linalg::CoopMatMatrixUse.MatrixA> mA[WARP_M];
    linalg::CoopMat<TMat, MemoryScope.Subgroup, K_STEP, 16, linalg::CoopMatMatrixUse.MatrixB> mB[WARP_N];
    [ForceUnroll] for (uint i = 0u; i < uint(WARP_M); ++i)
        mA[i] = S::loadMatA<TMat, K_STEP>((wr * uint(WARP_M * 16) + i * 16u) * 5u + kk_u4);
    [ForceUnroll] for (uint j = 0u; j < uint(WARP_N); ++j)
        mB[j] = S::loadMatB<TMat, K_STEP>((wc * uint(WARP_N * 16) + j * 16u) * 5u + kk_u4);
    [ForceUnroll] for (uint i = 0u; i < uint(WARP_M); ++i)
        [ForceUnroll] for (uint j = 0u; j < uint(WARP_N); ++j)
            acc[i][j] = linalg::coopMatMulAdd<TAcc, false>(mA[i], mB[j], acc[i][j]);
}

void coopmat_gemm_smem<TMat : ICoopElement, TAcc : ICoopElement, Ep : IEpilogue,
                       let K_STEP : int, let ELEM_BYTES : int, let IS_QUANT : int,
                       let WG_M : int, let WG_N : int, let WARP_M : int, let WARP_N : int,
                       let NTHREADS : int,
                       S : ICoopSmemTile<WG_M, WG_N, WARP_M, WARP_N>>(
    CoopGemmArgs<TMat, float> a, TAcc zero_acc, uint3 gtid, uint3 gid)
{
    uint tid = gtid.x;
    uint warp_id = tid >> 5u;
    uint lane_id = tid & 31u;
    uint n_col_warps = uint(WG_N / (WARP_N * 16));
    uint wr = warp_id / n_col_warps;
    uint wc = warp_id - wr * n_col_warps;

    uint wg_m_u = uint(WG_M);
    uint wg_n_u = uint(WG_N);
    uint grid_x = (a.m + wg_n_u - 1u) / wg_n_u;
    uint grid_y = (a.rows + wg_m_u - 1u) / wg_m_u;
    uint2 tile = swizzle_tile_2d(gid, grid_x, grid_y);
    uint row0 = tile.y * wg_m_u;
    uint col0 = tile.x * wg_n_u;

    linalg::CoopMat<TAcc, MemoryScope.Subgroup, 16, 16, linalg::CoopMatMatrixUse.MatrixAccumulator> acc[WARP_M][WARP_N];
    [ForceUnroll] for (uint i = 0u; i < uint(WARP_M); ++i)
        [ForceUnroll] for (uint j = 0u; j < uint(WARP_N); ++j)
            acc[i][j] = linalg::CoopMat<TAcc, MemoryScope.Subgroup, 16, 16, linalg::CoopMatMatrixUse.MatrixAccumulator>(zero_acc);

    Ptr<uint8_t, Access.Read> xb = Ptr<uint8_t, Access.Read>(a.x);
    Ptr<uint8_t, Access.Read> wb = Ptr<uint8_t, Access.Read>(a.w);
    uint elems_per_u4 = uint(16 / ELEM_BYTES);
    uint bk_elems = elems_per_u4 * 4u;
    uint num_k_tiles = (a.k + bk_elems - 1u) / bk_elems;
    bool aligned16 = is_ptr_aligned16(a.x) && is_ptr_aligned16(a.w) &&
                     (((a.x_stride * uint(ELEM_BYTES)) & 15u) == 0u) &&
                     (((a.w_stride * uint(ELEM_BYTES)) & 15u) == 0u);
    bool fast_interior = aligned16 && ((a.k & (bk_elems - 1u)) == 0u) &&
                         (row0 + wg_m_u <= a.rows) && (col0 + wg_n_u <= a.m);

    static const uint kVecA = uint((WG_M * 4) / NTHREADS);
    static const uint kVecB = uint((WG_N * 4) / NTHREADS);
    static const uint kRowStep = uint(NTHREADS / 4);
    uint lr0 = tid >> 2u;
    uint lu0 = tid & 3u;

    if (fast_interior && num_k_tiles > 0u) {
        uint64_t x_off0 = (uint64_t(row0 + lr0) * uint64_t(a.x_stride) + uint64_t(lu0 * elems_per_u4)) * uint64_t(ELEM_BYTES);
        uint64_t w_off0 = (uint64_t(col0 + lr0) * uint64_t(a.w_stride) + uint64_t(lu0 * elems_per_u4)) * uint64_t(ELEM_BYTES);
        Ptr<uint4, Access.Read> xp0 = Ptr<uint4, Access.Read>(xb + x_off0);
        Ptr<uint4, Access.Read> wp0 = Ptr<uint4, Access.Read>(wb + w_off0);
        uint64_t x_step_u4 = (uint64_t(kRowStep) * uint64_t(a.x_stride)) / uint64_t(elems_per_u4);
        uint64_t w_step_u4 = (uint64_t(kRowStep) * uint64_t(a.w_stride)) / uint64_t(elems_per_u4);
        uint s0 = lr0 * 5u + lu0;

        [ForceUnroll] for (uint v = 0u; v < kVecA; ++v)
            S::storeA(s0 + v * (kRowStep * 5u), load_a16(xp0 + uint64_t(v) * x_step_u4));
        [ForceUnroll] for (uint v = 0u; v < kVecB; ++v)
            S::storeB(s0 + v * (kRowStep * 5u), load_a16(wp0 + uint64_t(v) * w_step_u4));
        GroupMemoryBarrierWithGroupSync();

        for (uint kt = 0u; kt < num_k_tiles; ++kt) {
            uint4 pA[kVecA];
            uint4 pB[kVecB];
            [ForceUnroll] for (uint v = 0u; v < kVecA; ++v) pA[v] = uint4(0u);
            [ForceUnroll] for (uint v = 0u; v < kVecB; ++v) pB[v] = uint4(0u);
            bool has_next = (kt + 1u < num_k_tiles);
            if (has_next) {
                uint64_t k_u4 = uint64_t(kt + 1u) << 2u;
                [ForceUnroll] for (uint v = 0u; v < kVecA; ++v)
                    pA[v] = load_a16(xp0 + (uint64_t(v) * x_step_u4 + k_u4));
                [ForceUnroll] for (uint v = 0u; v < kVecB; ++v)
                    pB[v] = load_a16(wp0 + (uint64_t(v) * w_step_u4 + k_u4));
            }
            coopmat_mma_step<TMat, TAcc, K_STEP, WG_M, WG_N, WARP_M, WARP_N, S>(acc, wr, wc, 0u);
            linalg::CoopMat<TMat, MemoryScope.Subgroup, 16, K_STEP, linalg::CoopMatMatrixUse.MatrixA> mA1[WARP_M];
            linalg::CoopMat<TMat, MemoryScope.Subgroup, K_STEP, 16, linalg::CoopMatMatrixUse.MatrixB> mB1[WARP_N];
            [ForceUnroll] for (uint i = 0u; i < uint(WARP_M); ++i)
                mA1[i] = S::loadMatA<TMat, K_STEP>((wr * uint(WARP_M * 16) + i * 16u) * 5u + 2u);
            [ForceUnroll] for (uint j = 0u; j < uint(WARP_N); ++j)
                mB1[j] = S::loadMatB<TMat, K_STEP>((wc * uint(WARP_N * 16) + j * 16u) * 5u + 2u);
            if (has_next) {
                GroupMemoryBarrierWithGroupSync();
                [ForceUnroll] for (uint v = 0u; v < kVecA; ++v)
                    S::storeA(s0 + v * (kRowStep * 5u), pA[v]);
                [ForceUnroll] for (uint v = 0u; v < kVecB; ++v)
                    S::storeB(s0 + v * (kRowStep * 5u), pB[v]);
            }
            [ForceUnroll] for (uint i = 0u; i < uint(WARP_M); ++i)
                [ForceUnroll] for (uint j = 0u; j < uint(WARP_N); ++j)
                    acc[i][j] = linalg::coopMatMulAdd<TAcc, false>(mA1[i], mB1[j], acc[i][j]);
            if (has_next) {
                GroupMemoryBarrierWithGroupSync();
            }
        }
    } else {
        for (uint kt = 0u; kt < num_k_tiles; ++kt) {
            uint k0 = kt * bk_elems;
            [ForceUnroll] for (uint v = 0u; v < kVecA; ++v) {
                uint lr = lr0 + v * kRowStep;
                S::storeA(lr * 5u + lu0, load_u4_safe<ELEM_BYTES>(xb, row0 + lr, a.rows, k0 + lu0 * elems_per_u4, a.k, a.x_stride, aligned16));
            }
            [ForceUnroll] for (uint v = 0u; v < kVecB; ++v) {
                uint lr = lr0 + v * kRowStep;
                S::storeB(lr * 5u + lu0, load_u4_safe<ELEM_BYTES>(wb, col0 + lr, a.m, k0 + lu0 * elems_per_u4, a.k, a.w_stride, aligned16));
            }
            GroupMemoryBarrierWithGroupSync();
            coopmat_mma_step<TMat, TAcc, K_STEP, WG_M, WG_N, WARP_M, WARP_N, S>(acc, wr, wc, 0u);
            coopmat_mma_step<TMat, TAcc, K_STEP, WG_M, WG_N, WARP_M, WARP_N, S>(acc, wr, wc, 2u);
            GroupMemoryBarrierWithGroupSync();
        }
    }

    EpilogueParams ep = make_vk_ep(a.bias, a.x_scale, a.w_scale, a.flags);
    bool y_aligned4 = is_ptr_aligned16(a.y) && ((a.y_stride & 3u) == 0u);
    if ((IS_QUANT == 0) && (a.flags == 0u) && y_aligned4 && (row0 + wg_m_u <= a.rows) && (col0 + wg_n_u <= a.m)) {
        [ForceUnroll] for (uint i = 0u; i < uint(WARP_M); ++i) {
            [ForceUnroll] for (uint j = 0u; j < uint(WARP_N); ++j) {
                uint tr = row0 + wr * uint(WARP_M * 16) + i * 16u;
                uint tc = col0 + wc * uint(WARP_N * 16) + j * 16u;
                Ptr<TAcc> y_tile = Ptr<TAcc>(a.y + uint64_t(tr) * uint64_t(a.y_stride) + uint64_t(tc));
                acc[i][j].Store<linalg::CoopMatMatrixLayout.RowMajor>(y_tile, 0u, a.y_stride);
            }
        }
    } else {
        GroupMemoryBarrierWithGroupSync();
        uint warp_ep_base = warp_id * 64u;
        [ForceUnroll] for (uint i = 0u; i < uint(WARP_M); ++i) {
            uint tr = row0 + wr * uint(WARP_M * 16) + i * 16u;
            [ForceUnroll] for (uint j = 0u; j < uint(WARP_N); ++j) {
                uint tc = col0 + wc * uint(WARP_N * 16) + j * 16u;
                if (tr < a.rows && tc < a.m) {
                    S::storeAcc<TAcc>(acc[i][j], warp_ep_base);
                    subgroup_smem_barrier();
                    write_staged_ep_tile<IS_QUANT, Ep, WG_M, WG_N, WARP_M, WARP_N, S>(
                        warp_ep_base, a.y, a.y_stride, a.rows, a.m, tr, tc, lane_id, y_aligned4, ep);
                    subgroup_smem_barrier();
                }
            }
        }
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_gemm_coopmat_f16(uniform CoopGemmArgs<half, float> a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    coopmat_gemm_smem<half, float, EpBias, 16, 2, 0, 128, 128, 4, 2, 256, CoopSmem128>(a, 0.0f, gtid, gid);
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_gemm_coopmat_bf16(uniform CoopGemmArgs<BFloat16, float> a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    coopmat_gemm_smem<BFloat16, float, EpBias, 16, 2, 0, 128, 128, 4, 2, 256, CoopSmem128>(a, 0.0f, gtid, gid);
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_gemm_coopmat_w8a8(uniform CoopGemmArgs<int8_t, float> a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    coopmat_gemm_smem<int8_t, int32_t, EpQuantScale, 32, 1, 1, 128, 128, 4, 2, 256, CoopSmem128>(a, 0, gtid, gid);
}

[shader("compute")] [numthreads(128, 1, 1)]
void vk_gemm_coopmat_f16_64(uniform CoopGemmArgs<half, float> a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    coopmat_gemm_smem<half, float, EpBias, 16, 2, 0, 64, 64, 2, 2, 128, CoopSmem64>(a, 0.0f, gtid, gid);
}

[shader("compute")] [numthreads(128, 1, 1)]
void vk_gemm_coopmat_bf16_64(uniform CoopGemmArgs<BFloat16, float> a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    coopmat_gemm_smem<BFloat16, float, EpBias, 16, 2, 0, 64, 64, 2, 2, 128, CoopSmem64>(a, 0.0f, gtid, gid);
}

[shader("compute")] [numthreads(128, 1, 1)]
void vk_gemm_coopmat_w8a8_64(uniform CoopGemmArgs<int8_t, float> a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    coopmat_gemm_smem<int8_t, int32_t, EpQuantScale, 32, 1, 1, 64, 64, 2, 2, 128, CoopSmem64>(a, 0, gtid, gid);
}
