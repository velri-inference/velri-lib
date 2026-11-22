// Vulkan Grouped Batched Linear kernels (Y = X @ W^T + b and Y = X @ W + b, F32, 0 spills).
import epilogue;
import reductions_wide;
import vk_common;

struct BatchedLinearArgs {
    Ptr<float, Access.Read> x;
    Ptr<float, Access.Read> w;
    Ptr<float, Access.Read> bias;
    Ptr<float> y;
    uint rows;
    uint n_heads;
    uint n_groups;
    uint in_dim;
    uint out_dim;
    uint x_stride;
    uint w_stride;
    uint y_stride;
    uint flags;
    uint pad;
};

// Non-transposed weight layout W [G * out_dim, in_dim]: Y = X @ W^T + bias.
// Fixed 4 x 32-thread partitions per 128-thread WG (immune to varying Vulkan 1.3 subgroup sizes;
// 0 barriers / single WaveActiveSum on Wave32; vectorized float4 loads when 16B-aligned).
[shader("compute")] [numthreads(128, 1, 1)]
void vk_batched_linear_nt_f32(uniform BatchedLinearArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint t = gid.y;
    uint h = gid.z;
    if (t >= a.rows || h >= a.n_heads) return;

    uint tid = gtid.x;
    uint part = tid >> 5u;
    uint lane = tid & 31u;
    uint m = gid.x * 4u + part;
    bool valid = (m < a.out_dim);

    uint g = h / (a.n_heads / a.n_groups);
    float acc = 0.0f;
    if (valid) {
        Ptr<float, Access.Read> x_ptr =
            a.x + uint64_t(t) * uint64_t(a.x_stride) + uint64_t(h) * uint64_t(a.in_dim);
        Ptr<float, Access.Read> w_row =
            a.w + (uint64_t(g) * uint64_t(a.out_dim) + uint64_t(m)) * uint64_t(a.w_stride);

        uint k4 = a.in_dim >> 2u;
        if (k4 > 0u && is_ptr_aligned16(x_ptr) && is_ptr_aligned16(w_row)) {
            Ptr<float4, Access.Read> x4 = Ptr<float4, Access.Read>(x_ptr);
            Ptr<float4, Access.Read> w4 = Ptr<float4, Access.Read>(w_row);
            for (uint64_t i = uint64_t(lane); i < uint64_t(k4); i += 32ull) {
                acc += dot(load_a16(x4 + i), load_a16(w4 + i));
            }
            uint tail_k = (k4 << 2u) + lane;
            if (tail_k < a.in_dim) {
                acc = fma(x_ptr[tail_k], w_row[tail_k], acc);
            }
        } else {
            for (uint64_t k = uint64_t(lane); k < uint64_t(a.in_dim); k += 32ull) {
                acc = fma(x_ptr[k], w_row[k], acc);
            }
        }
    }

    float sum = part32_reduce_sum(acc, tid, part, lane);
    if (valid && lane == 0u) {
        if ((a.flags & kEpFlagBias) != 0u) {
            sum += a.bias[uint64_t(g) * uint64_t(a.out_dim) + uint64_t(m)];
        }
        a.y[uint64_t(t) * uint64_t(a.y_stride) + uint64_t(h) * uint64_t(a.out_dim) + uint64_t(m)] =
            sum;
    }
}

// Transposed weight layout W [G * in_dim, out_dim]: Y = X @ W + bias.
// One thread per output element m; 100% coalesced reads of W and writes of Y, 0 barriers, 0 smem.
[shader("compute")] [numthreads(256, 1, 1)]
void vk_batched_linear_nn_f32(uniform BatchedLinearArgs a, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint m = gid.x * kRowWgSize + gtid.x;
    uint t = gid.y;
    uint h = gid.z;
    if (t >= a.rows || h >= a.n_heads || m >= a.out_dim) return;

    uint g = h / (a.n_heads / a.n_groups);
    Ptr<float, Access.Read> x_ptr =
        a.x + uint64_t(t) * uint64_t(a.x_stride) + uint64_t(h) * uint64_t(a.in_dim);
    uint64_t ws64 = uint64_t(a.w_stride);
    Ptr<float, Access.Read> w_grp = a.w + (uint64_t(g) * uint64_t(a.in_dim)) * ws64;

    float acc = ((a.flags & kEpFlagBias) != 0u)
                    ? a.bias[uint64_t(g) * uint64_t(a.out_dim) + uint64_t(m)]
                    : 0.0f;
    uint in_dim4 = a.in_dim & ~3u;
    uint k = 0u;
    for (; k < in_dim4; k += 4u) {
        acc = fma(x_ptr[k + 0u], w_grp[uint64_t(k + 0u) * ws64 + uint64_t(m)], acc);
        acc = fma(x_ptr[k + 1u], w_grp[uint64_t(k + 1u) * ws64 + uint64_t(m)], acc);
        acc = fma(x_ptr[k + 2u], w_grp[uint64_t(k + 2u) * ws64 + uint64_t(m)], acc);
        acc = fma(x_ptr[k + 3u], w_grp[uint64_t(k + 3u) * ws64 + uint64_t(m)], acc);
    }
    for (; k < a.in_dim; ++k) {
        acc = fma(x_ptr[k], w_grp[uint64_t(k) * ws64 + uint64_t(m)], acc);
    }
    a.y[uint64_t(t) * uint64_t(a.y_stride) + uint64_t(h) * uint64_t(a.out_dim) + uint64_t(m)] = acc;
}
