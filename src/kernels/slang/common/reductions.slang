// Shared GPU subgroup + workgroup reduction primitives (subgroup-size agnostic: 4..128).
// Reused by Vulkan RMSNorm, LayerNorm, Softmax, and QuantizeRow kernels.
module reductions;

// Maximum supported subgroups per workgroup (256 threads / min subgroup 4 = 64).
static const uint kMaxSubgroups = 64u;
groupshared float g_reduce_buf[kMaxSubgroups];

public float wg_reduce_sum(float val, uint tid, uint wg_size) {
    float wave_sum = WaveActiveSum(val);
    uint lane_count = WaveGetLaneCount();
    uint lane_id = WaveGetLaneIndex();
    uint wave_id = tid / lane_count;
    uint num_waves = (wg_size + lane_count - 1u) / lane_count;

    if (num_waves <= 1u) {
        return wave_sum;
    }
    if (lane_id == 0u) {
        g_reduce_buf[wave_id] = wave_sum;
    }
    GroupMemoryBarrierWithGroupSync();

    if (wave_id == 0u) {
        float acc = 0.0f;
        for (uint i = lane_id; i < num_waves; i += lane_count) {
            acc += g_reduce_buf[i];
        }
        float total = WaveActiveSum(acc);
        if (lane_id == 0u) {
            g_reduce_buf[0] = total;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    return g_reduce_buf[0];
}

public float wg_reduce_max(float val, uint tid, uint wg_size) {
    float wave_max = WaveActiveMax(val);
    uint lane_count = WaveGetLaneCount();
    uint lane_id = WaveGetLaneIndex();
    uint wave_id = tid / lane_count;
    uint num_waves = (wg_size + lane_count - 1u) / lane_count;

    if (num_waves <= 1u) {
        return wave_max;
    }
    if (lane_id == 0u) {
        g_reduce_buf[wave_id] = wave_max;
    }
    GroupMemoryBarrierWithGroupSync();

    if (wave_id == 0u) {
        float acc = asfloat(0xFF800000u); // -INF
        for (uint i = lane_id; i < num_waves; i += lane_count) {
            acc = max(acc, g_reduce_buf[i]);
        }
        float total = WaveActiveMax(acc);
        if (lane_id == 0u) {
            g_reduce_buf[0] = total;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    return g_reduce_buf[0];
}
