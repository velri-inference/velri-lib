// Vulkan High-Bandwidth GEMV kernels (F32, F16, BF16, W8A8 -> F32).
// Reuses IVecIO128::unpack and unified LinearArgs from vk_common.
import numeric;
import epilogue;
import vk_common;

void gemv_float_gpu_impl<T : IElement, VIO : IVecIO128<T>, Ep : IEpilogue>(
    LinearArgs<T.Storage, T.Storage, float> a, uint3 gtid, uint3 gid)
{
    uint lane = WaveGetLaneIndex();
    uint lane_cnt = WaveGetLaneCount();
    uint warp_in_wg = gtid.x / lane_cnt;
    uint warps_per_wg = kGemvWgSize / lane_cnt;
    uint col = gid.x * warps_per_wg + warp_in_wg;
    uint row = gid.y;
    if (row >= a.rows || col >= a.m) return;

    Ptr<T.Storage, Access.Read> xr = a.x + uint64_t(row) * uint64_t(a.x_stride);
    Ptr<T.Storage, Access.Read> wc = a.w + uint64_t(col) * uint64_t(a.w_stride);
    uint mask = VIO.kElems - 1u;
    float acc = 0.0f;
    if (is_ptr_aligned16(xr) && is_ptr_aligned16(wc)) {
        uint k_vec = a.k & ~mask;
        uint n_vec = k_vec / VIO.kElems;
        Ptr<uint4, Access.Read> xr4 = Ptr<uint4, Access.Read>(xr);
        Ptr<uint4, Access.Read> wc4 = Ptr<uint4, Access.Read>(wc);
        for (uint i = lane; i < n_vec; i += lane_cnt) {
            float4 xlo, xhi, wlo, whi;
            VIO::unpack(load_a16(xr4 + i), xlo, xhi);
            VIO::unpack(load_a16(wc4 + i), wlo, whi);
            acc += dot(xlo, wlo);
            if (VIO.kElems == 8u) acc += dot(xhi, whi);
        }
        for (uint i = k_vec + lane; i < a.k; i += lane_cnt) {
            acc = fma(T::toF32(xr[i]), T::toF32(wc[i]), acc);
        }
    } else {
        for (uint i = lane; i < a.k; i += lane_cnt) {
            acc = fma(T::toF32(xr[i]), T::toF32(wc[i]), acc);
        }
    }
    float total = WaveActiveSum(acc);
    if (lane == 0u) {
        a.y[uint64_t(row) * uint64_t(a.y_stride) + uint64_t(col)] =
            Ep::apply(total, row, col, make_epilogue_params(a.bias, a.x_scale, a.w_scale, a.flags));
    }
}

[shader("compute")] [numthreads(128, 1, 1)]
void vk_gemv_f32(uniform LinearArgs<float, float, float> a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    gemv_float_gpu_impl<ElemF32, VecIOF32, EpBias>(a, gtid, gid);
}

[shader("compute")] [numthreads(128, 1, 1)]
void vk_gemv_f16(uniform LinearArgs<ElemF16.Storage, ElemF16.Storage, float> a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    gemv_float_gpu_impl<ElemF16, VecIOF16, EpBias>(a, gtid, gid);
}

[shader("compute")] [numthreads(128, 1, 1)]
void vk_gemv_bf16(uniform LinearArgs<ElemBF16.Storage, ElemBF16.Storage, float> a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    gemv_float_gpu_impl<ElemBF16, VecIOBF16, EpBias>(a, gtid, gid);
}

[shader("compute")] [numthreads(128, 1, 1)]
void vk_gemv_w8a8(uniform LinearArgs<int8_t, int8_t, float> a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint lane = WaveGetLaneIndex();
    uint lane_cnt = WaveGetLaneCount();
    uint warp_in_wg = gtid.x / lane_cnt;
    uint warps_per_wg = kGemvWgSize / lane_cnt;
    uint col = gid.x * warps_per_wg + warp_in_wg;
    uint row = gid.y;
    if (row >= a.rows || col >= a.m) return;

    Ptr<int8_t, Access.Read> xr = a.x + uint64_t(row) * uint64_t(a.x_stride);
    Ptr<int8_t, Access.Read> wc = a.w + uint64_t(col) * uint64_t(a.w_stride);
    int32_t acc = 0;
    uint k16 = a.k & ~15u;
    if (is_ptr_aligned16(xr) && is_ptr_aligned16(wc)) {
        Ptr<uint4, Access.Read> xr16 = Ptr<uint4, Access.Read>(xr);
        Ptr<uint4, Access.Read> wc16 = Ptr<uint4, Access.Read>(wc);
        uint n16 = k16 >> 4u;
        for (uint i = lane; i < n16; i += lane_cnt) {
            uint4 xv = load_a16(xr16 + i);
            uint4 wv = load_a16(wc16 + i);
            acc = dot4add_i8packed(xv.x, wv.x, acc);
            acc = dot4add_i8packed(xv.y, wv.y, acc);
            acc = dot4add_i8packed(xv.z, wv.z, acc);
            acc = dot4add_i8packed(xv.w, wv.w, acc);
        }
        for (uint i = k16 + lane; i < a.k; i += lane_cnt) {
            acc += int32_t(xr[i]) * int32_t(wc[i]);
        }
    } else {
        for (uint i = lane; i < a.k; i += lane_cnt) {
            acc += int32_t(xr[i]) * int32_t(wc[i]);
        }
    }
    int32_t total = WaveActiveSum(acc);
    if (lane == 0u) {
        a.y[uint64_t(row) * uint64_t(a.y_stride) + uint64_t(col)] =
            EpQuantScale::apply(float(total), row, col, make_epilogue_params(a.bias, a.x_scale, a.w_scale, a.flags));
    }
}

// Reuse each aligned weight vector across four prompt rows. A subgroup owns
// one output channel and masks unused rows of the final group.
[shader("compute")] [numthreads(128, 1, 1)]
void vk_gemv_rows_f32(uniform LinearArgs<float, float, float> a, uint3 group : SV_GroupID) {
    uint lanes = WaveGetLaneCount(), lane = WaveGetLaneIndex();
    uint col = group.x * (kGemvWgSize / lanes) + WaveGetWaveIndex();
    uint first = group.y * 4u;
    if (col >= a.m) return;
    Ptr<float4, Access.Read> weights = Ptr<float4, Access.Read>(a.w + uint64_t(col) * a.w_stride);
    float4 sums = float4(0.0f);
    for (uint k = lane; k < a.k / 4u; k += lanes) {
        float4 weight = load_a16(weights + k);
        [ForceUnroll] for (uint r = 0u; r < 4u; ++r) {
            if (first + r < a.rows) {
                Ptr<float4, Access.Read> x = Ptr<float4, Access.Read>(
                    a.x + uint64_t(first + r) * a.x_stride);
                sums[r] += dot(load_a16(x + k), weight);
            }
        }
    }
    sums = WaveActiveSum(sums);
    if (lane == 0u) {
        float bias = ((a.flags & kEpFlagBias) != 0u) ? a.bias[col] : 0.0f;
        [ForceUnroll] for (uint r = 0u; r < 4u; ++r) {
            if (first + r < a.rows) a.y[uint64_t(first + r) * a.y_stride + col] = sums[r] + bias;
        }
    }
}
