// Dense causal mixed-precision attention, 32 queries by 256 keys per workgroup.
// Each subgroup loads two KV tiles once and applies them to both query tiles.
import attention_args;
import flash_attention_half;

typealias LeftTile = linalg::CoopMat<half, MemoryScope.Subgroup, 16, 16, linalg::CoopMatMatrixUse.MatrixA>;
typealias RightTile = linalg::CoopMat<half, MemoryScope.Subgroup, 16, 16, linalg::CoopMatMatrixUse.MatrixB>;
typealias AccTile = linalg::CoopMat<float, MemoryScope.Subgroup, 16, 16, linalg::CoopMatMatrixUse.MatrixAccumulator>;

static const uint kWaves = 8u, kTileKeys = 32u;
static const float kLog2E = 1.44269504f, kLowest = -3.40282346e38f;

float max_value(float x, float y) { return max(x, y); }
float add_value(float x, float y) { return x + y; }
float max_element(uint row, uint col, float x, float y) { return max(x, y); }
float max_abs_element(uint row, uint col, float x, float y) { return max(x, abs(y)); }
float nonzero_scale(uint row, uint col, float x) { return x == 0.0f ? 1.0f : x; }
float exp2_element(uint row, uint col, float x) { return exp2(x); }

struct CausalMask : IFunc<float, uint, uint, float> {
    uint first_query, first_key;
    float operator()(uint row, uint col, float value) {
        return first_key + col <= first_query + row ? value : asfloat(0xff800000u);
    }
}

// Each subgroup publishes a 16x64 output and two broadcast 16x16 row reductions.
static const uint kWaveFloats = 1024u + 256u + 256u;
groupshared float partials[kWaves * kWaveFloats];

void publish_tile(AccTile tile, uint offset, uint stride) {
    uint row_major = 0u;
    spirv_asm {
        OpExtension "SPV_KHR_cooperative_matrix";
        OpCapability CooperativeMatrixKHR;
        %p : $$Ptr<float, Access.ReadWrite, AddressSpace.GroupShared> = OpAccessChain &partials $offset;
        OpCooperativeMatrixStoreKHR %p $tile $row_major $stride;
    };
}

// A masked non-finite V can contaminate an entire output channel in an MMA.
// Repeat the same panel reductions for that channel over the visible keys only.
float repair_score(AttentionArgs a, Ptr<float> q, Ptr<half> keys, uint key, float scale) {
    float dot = 0.0f;
    for (uint j = 0u; j < 64u; ++j) {
        float qh = widen_half(f32tof16(q[j] / scale));
        dot = fma(qh, float(keys[uint64_t(key) * a.k_row_stride + j]), dot);
    }
    return dot * scale * (a.scale * kLog2E);
}

float repair_channel(AttentionArgs a, Ptr<float> q, Ptr<half> keys, Ptr<half> values,
                     uint channel, uint end) {
    float scale = 0.0f;
    for (uint j = 0u; j < 64u; ++j) scale = max(scale, abs(q[j]));
    if (scale == 0.0f) scale = 1.0f;
    float high = kLowest, total = 0.0f, sum = 0.0f;
    for (uint w = 0u; w < kWaves; ++w) {
        float wave_high = kLowest, wave_mass = 0.0f, wave_sum = 0.0f;
        for (uint start = kTileKeys * w; start < end; start += kTileKeys * kWaves) {
            // Recompute scores in the rare repair path to avoid a private array per lane.
            float next_high = wave_high;
            uint count = min(kTileKeys, end - start);
            for (uint n = 0u; n < count; ++n) {
                next_high = max(next_high, repair_score(a, q, keys, start + n, scale));
            }
            float carry = exp2(wave_high - next_high);
            wave_mass *= carry;
            wave_sum *= carry;
            for (uint n = 0u; n < count; ++n) {
                float p = exp2(repair_score(a, q, keys, start + n, scale) - next_high);
                float value = float(values[uint64_t(start + n) * a.v_row_stride + channel]);
                wave_sum = fma(widen_half(f32tof16(p)), value, wave_sum);
                wave_mass += p;
            }
            wave_high = next_high;
        }
        if (wave_mass != 0.0f) {
            float next_high = max(high, wave_high);
            float carry = exp2(high - next_high), weight = exp2(wave_high - next_high);
            sum = sum * carry + wave_sum * weight;
            total = total * carry + wave_mass * weight;
            high = next_high;
        }
    }
    return sum / total;
}

void load_panel<let layout : linalg::CoopMatMatrixLayout>(
    Ptr<half> cache, uint first, uint stride, out RightTile panel[2][4]) {
    [ForceUnroll] for (uint t = 0u; t < 2u; ++t) {
        [ForceUnroll] for (uint j = 0u; j < 4u; ++j) {
            panel[t][j] = RightTile.Load<layout>(
                cache + uint64_t(first + 16u * t) * stride + 16u * j, 0u, stride);
        }
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void vk_prefill_coop_nv_f32(uniform AttentionArgs a, uint3 group : SV_GroupID, uint tid : SV_GroupIndex) {
    uint first = group.x * 32u, head = group.y, kv_head = head / (a.n_heads / a.n_kv_heads);
    uint pos = attention_kv_pos(a), wave = WaveGetWaveIndex();
    Ptr<float> query = Ptr<float>(a.q + uint64_t(first) * a.q_stride + uint64_t(head) * 64u);
    Ptr<half> keys = a.k_cache + uint64_t(kv_head) * a.k_head_stride;
    Ptr<half> values = a.v_cache + uint64_t(kv_head) * a.v_head_stride;

    LeftTile queries[2][4];
    AccTile scales[2], highs[2], masses[2], outputs[2][4];
    [ForceUnroll] for (uint z = 0u; z < 2u; ++z) {
        AccTile q[4], magnitude = AccTile(0.0f);
        [ForceUnroll] for (uint j = 0u; j < 4u; ++j) {
            q[j] = AccTile.Load<linalg::CoopMatMatrixLayout.RowMajor>(
                query + uint64_t(16u * z) * a.q_stride + 16u * j, 0u, a.q_stride);
            magnitude = Tuple<AccTile, AccTile>(magnitude, q[j]).MapElement(max_abs_element);
        }
        scales[z] = magnitude.ReduceRow<16>(max_value).MapElement(nonzero_scale);
        [ForceUnroll] for (uint j = 0u; j < 4u; ++j) {
            queries[z][j] = LeftTile(q[j] / scales[z]);
            outputs[z][j] = AccTile(0.0f);
        }
        highs[z] = AccTile(kLowest);
        masses[z] = AccTile(0.0f);
    }

    uint limit = pos + first + 32u;
    uint first_panel = wave * kTileKeys < limit ? wave * kTileKeys : 0u;
    RightTile key_panel[2][4];
    load_panel<linalg::CoopMatMatrixLayout.ColumnMajor>(keys, first_panel, a.k_row_stride, key_panel);
    for (uint base = wave * kTileKeys; base < limit; base += kWaves * kTileKeys) {
        AccTile scores[2][2];
        [ForceUnroll] for (uint t = 0u; t < 2u; ++t) {
            uint start = base + 16u * t;
            AccTile dot[2] = {AccTile(0.0f), AccTile(0.0f)};
            [ForceUnroll] for (uint j = 0u; j < 4u; ++j) {
                [ForceUnroll] for (uint z = 0u; z < 2u; ++z)
                    dot[z] = linalg::coopMatMulAdd<float, false>(queries[z][j], key_panel[t][j], dot[z]);
            }
            [ForceUnroll] for (uint z = 0u; z < 2u; ++z) {
                uint first_query = pos + first + 16u * z;
                scores[z][t] = (dot[z] * scales[z]) * (a.scale * kLog2E);
                if (start + 15u > first_query) {
                    CausalMask mask = {first_query, start};
                    scores[z][t] = scores[z][t].MapElement(mask);
                }
            }
        }
        // Issue global loads before the softmax arithmetic that can hide their latency.
        RightTile value_panel[2][4];
        load_panel<linalg::CoopMatMatrixLayout.RowMajor>(values, base, a.v_row_stride, value_panel);
        if (limit - base > kWaves * kTileKeys) {
            load_panel<linalg::CoopMatMatrixLayout.ColumnMajor>(
                keys, base + kWaves * kTileKeys, a.k_row_stride, key_panel);
        }
        LeftTile probs[2][2];
        [ForceUnroll] for (uint z = 0u; z < 2u; ++z) {
            AccTile next_high = highs[z];
            [ForceUnroll] for (uint t = 0u; t < 2u; ++t)
                next_high = Tuple<AccTile, AccTile>(next_high, scores[z][t]).MapElement(max_element);
            next_high = next_high.ReduceRow<16>(max_value);
            AccTile carry = (highs[z] - next_high).MapElement(exp2_element);
            [ForceUnroll] for (uint j = 0u; j < 4u; ++j)
                outputs[z][j] = outputs[z][j] * carry;
            AccTile panel_mass = AccTile(0.0f);
            [ForceUnroll] for (uint t = 0u; t < 2u; ++t) {
                AccTile p = (scores[z][t] - next_high).MapElement(exp2_element);
                panel_mass = panel_mass + p;
                probs[z][t] = LeftTile(p);
            }
            // carry is constant along each row, so reduce the accumulated columns once after the loop.
            masses[z] = masses[z] * carry + panel_mass;
            highs[z] = next_high;
        }
        [ForceUnroll] for (uint t = 0u; t < 2u; ++t) {
            [ForceUnroll] for (uint j = 0u; j < 4u; ++j) {
                [ForceUnroll] for (uint z = 0u; z < 2u; ++z)
                    outputs[z][j] = linalg::coopMatMulAdd<float, false>(probs[z][t], value_panel[t][j], outputs[z][j]);
            }
        }
    }

    [ForceUnroll] for (uint z = 0u; z < 2u; ++z) {
        [ForceUnroll] for (uint j = 0u; j < 4u; ++j)
            publish_tile(outputs[z][j], wave * kWaveFloats + 16u * j, 64u);
        publish_tile(highs[z], wave * kWaveFloats + 1024u, 16u);
        publish_tile(masses[z].ReduceRow<16>(add_value), wave * kWaveFloats + 1280u, 16u);
        GroupMemoryBarrierWithGroupSync();
        float results[4];
        [ForceUnroll] for (uint i = 0u; i < 4u; ++i) {
            uint idx = tid + i * 256u, row = idx / 64u, col = idx % 64u;
            float high = kLowest, mass = 0.0f, sum = 0.0f;
            [ForceUnroll] for (uint w = 0u; w < kWaves; ++w)
                high = max(high, partials[w * kWaveFloats + 1024u + row * 16u]);
            [ForceUnroll] for (uint w = 0u; w < kWaves; ++w) {
                float weight = exp2(partials[w * kWaveFloats + 1024u + row * 16u] - high);
                mass += weight * partials[w * kWaveFloats + 1280u + row * 16u];
                sum += weight * partials[w * kWaveFloats + idx];
            }
            results[i] = sum / mass;
            if (!isfinite(results[i])) {
                uint qrow = 16u * z + row;
                results[i] = repair_channel(a, query + uint64_t(qrow) * a.q_stride,
                                            keys, values, col, pos + first + qrow + 1u);
            }
        }
        // Channels in the same workgroup may still read Q during repair when Q aliases Y.
        AllMemoryBarrierWithGroupSync();
        [ForceUnroll] for (uint i = 0u; i < 4u; ++i) {
            uint idx = tid + i * 256u;
            a.y[uint64_t(first + 16u * z + idx / 64u) * a.o_stride + uint64_t(head) * 64u + idx % 64u] = results[i];
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
