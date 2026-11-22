// Shared GPU subgroup + workgroup reduction primitives (subgroup-size agnostic: 4..128).
// Reused by Vulkan RMSNorm, LayerNorm, Softmax, and QuantizeRow kernels.
module reductions;

// Maximum supported subgroups per workgroup (256 threads / min subgroup 4 = 64).
static const uint kMaxSubgroups = 64u;
groupshared float g_reduce_buf[kMaxSubgroups];

public interface IReduceOp {
    static float identity();
    static float combine(float a, float b);
    static float waveReduce(float v);
}

public struct ReduceAdd : IReduceOp {
    public static float identity() { return 0.0f; }
    public static float combine(float a, float b) { return a + b; }
    public static float waveReduce(float v) { return WaveActiveSum(v); }
}

public struct ReduceMax : IReduceOp {
    public static float identity() { return asfloat(0xFF800000u); } // -INF
    public static float combine(float a, float b) { return max(a, b); }
    public static float waveReduce(float v) { return WaveActiveMax(v); }
}

public float wg_reduce<Op : IReduceOp>(float val, uint tid, uint wg_size) {
    float wave_val = Op::waveReduce(val);
    uint lane_count = WaveGetLaneCount();
    uint lane_id = WaveGetLaneIndex();
    uint wave_id = tid / lane_count;
    uint num_waves = (wg_size + lane_count - 1u) / lane_count;

    if (num_waves <= 1u) {
        return wave_val;
    }
    if (lane_id == 0u) {
        g_reduce_buf[wave_id] = wave_val;
    }
    GroupMemoryBarrierWithGroupSync();

    if (wave_id == 0u) {
        float acc = Op::identity();
        for (uint i = lane_id; i < num_waves; i += lane_count) {
            acc = Op::combine(acc, g_reduce_buf[i]);
        }
        float total = Op::waveReduce(acc);
        if (lane_id == 0u) {
            g_reduce_buf[0] = total;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    float res = g_reduce_buf[0];
    GroupMemoryBarrierWithGroupSync();
    return res;
}

public float wg_reduce_sum(float val, uint tid, uint wg_size) {
    return wg_reduce<ReduceAdd>(val, tid, wg_size);
}

public float wg_reduce_max(float val, uint tid, uint wg_size) {
    return wg_reduce<ReduceMax>(val, tid, wg_size);
}
