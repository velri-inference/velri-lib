// Wide / 32-lane partition GPU workgroup reduction primitives (subgroup-size agnostic).
// Kept separate from common/reductions.slang so 1-scalar norm/softmax kernels do not
// allocate this 1 KiB workgroup buffer; shared by vk/attention.slang & vk/batched_linear.slang.
module reductions_wide;

static const uint kWideWgSize = 256u;
static const uint kWideMaxWaves = 64u;
groupshared float g_part32_buf[kWideWgSize];

// Subgroup-size-agnostic sum + broadcast within each 32-thread partition.
// Zero barriers / zero smem traffic on Wave32; portable tree reduction on all other wave sizes.
public float part32_reduce_sum(float val, uint tid, uint part, uint lane) {
    if (WaveGetLaneCount() == 32u) {
        return WaveActiveSum(val);
    }
    g_part32_buf[tid] = val;
    GroupMemoryBarrierWithGroupSync();
    if (lane < 16u) g_part32_buf[tid] += g_part32_buf[tid + 16u];
    GroupMemoryBarrierWithGroupSync();
    if (lane < 8u)  g_part32_buf[tid] += g_part32_buf[tid + 8u];
    GroupMemoryBarrierWithGroupSync();
    if (lane < 4u)  g_part32_buf[tid] += g_part32_buf[tid + 4u];
    GroupMemoryBarrierWithGroupSync();
    if (lane < 2u)  g_part32_buf[tid] += g_part32_buf[tid + 2u];
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u) g_part32_buf[tid] += g_part32_buf[tid + 1u];
    GroupMemoryBarrierWithGroupSync();
    float res = g_part32_buf[part << 5u];
    GroupMemoryBarrierWithGroupSync();
    return res;
}

// Cross-partition lane sum: sums `val` across P 32-thread partitions for each lane 0..31,
// returning the reduced sum on partition 0 (0.0f on other partitions).
public float part32_cross_sum<let P : uint>(float val, uint tid, uint part, uint lane) {
    g_part32_buf[tid] = val;
    GroupMemoryBarrierWithGroupSync();
    float sum = 0.0f;
    if (part == 0u) {
        [ForceUnroll] for (uint p = 0u; p < P; ++p) {
            sum += g_part32_buf[(p << 5u) + lane];
        }
    }
    GroupMemoryBarrierWithGroupSync();
    return sum;
}

// Loop-safe 4-wide workgroup sum reduction for 256-thread WGs reusing g_part32_buf[256]
// (4 x 64 waves max): 2 barriers per 4 values, zero local-memory spills.
public void wg_reduce_sum4(float v0, float v1, float v2, float v3, uint tid,
                           out float r0, out float r1, out float r2, out float r3)
{
    float w0 = WaveActiveSum(v0);
    float w1 = WaveActiveSum(v1);
    float w2 = WaveActiveSum(v2);
    float w3 = WaveActiveSum(v3);

    uint lane_cnt = WaveGetLaneCount();
    if (lane_cnt >= kWideWgSize) {
        r0 = w0;
        r1 = w1;
        r2 = w2;
        r3 = w3;
        return;
    }

    uint nwaves = kWideWgSize / lane_cnt;
    uint wid = tid / lane_cnt;
    if (WaveIsFirstLane()) {
        g_part32_buf[0u * kWideMaxWaves + wid] = w0;
        g_part32_buf[1u * kWideMaxWaves + wid] = w1;
        g_part32_buf[2u * kWideMaxWaves + wid] = w2;
        g_part32_buf[3u * kWideMaxWaves + wid] = w3;
    }
    GroupMemoryBarrierWithGroupSync();

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint w = 0u; w < nwaves; ++w) {
        a0 += g_part32_buf[0u * kWideMaxWaves + w];
        a1 += g_part32_buf[1u * kWideMaxWaves + w];
        a2 += g_part32_buf[2u * kWideMaxWaves + w];
        a3 += g_part32_buf[3u * kWideMaxWaves + w];
    }
    GroupMemoryBarrierWithGroupSync();
    r0 = a0;
    r1 = a1;
    r2 = a2;
    r3 = a3;
}
