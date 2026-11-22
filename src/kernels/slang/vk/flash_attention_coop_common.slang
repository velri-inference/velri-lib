import attention_args;
import flash_attention_half;
import key_iter;
import vk_common;

typealias LeftTile = linalg::CoopMat<half, MemoryScope.Subgroup, 16, 16, linalg::CoopMatMatrixUse.MatrixA>;
typealias RightTile = linalg::CoopMat<half, MemoryScope.Subgroup, 16, 16, linalg::CoopMatMatrixUse.MatrixB>;
typealias AccTile = linalg::CoopMat<float, MemoryScope.Subgroup, 16, 16, linalg::CoopMatMatrixUse.MatrixAccumulator>;

// Q is loaded into registers once. Its storage is then reused for probabilities
// and finally for the four maxima and masses per query row.
#ifdef VELRI_HALF_ATTENTION_OPERANDS
static const bool kCompensated = false;
// Pad the row strides to spread simultaneous row reads across shared-memory banks.
static const uint kScoreStride = 24u, kOutputStride = 72u;
static const uint kWaveFloats = 1536u;
#else
static const bool kCompensated = true;
static const uint kScoreStride = 16u, kOutputStride = 64u;
static const uint kWaveFloats = 1024u;
#endif
groupshared uint packed[kCompensated ? 4096 : 2048];
groupshared float product[4u * kWaveFloats];

LeftTile load_left(uint offset, uint stride) {
    uint layout = 0u;
    return spirv_asm {
        OpExtension "SPV_KHR_cooperative_matrix";
        OpCapability CooperativeMatrixKHR;
        %ptr : $$Ptr<uint, Access.ReadWrite, AddressSpace.GroupShared> = OpAccessChain &packed $offset;
        result : $$LeftTile = OpCooperativeMatrixLoadKHR %ptr $layout $stride;
    };
}

void store_product(AccTile tile, uint offset, uint stride) {
    uint layout = 0u;
    spirv_asm {
        OpExtension "SPV_KHR_cooperative_matrix";
        OpCapability CooperativeMatrixKHR;
        %ptr : $$Ptr<float, Access.ReadWrite, AddressSpace.GroupShared> = OpAccessChain &product $offset;
        OpCooperativeMatrixStoreKHR %ptr $tile $layout $stride;
    };
}

void sync_subgroup_shared() {
    uint subgroup = 3u, acquire_release_workgroup_memory = 0x108u;
    spirv_asm { OpControlBarrier $subgroup $subgroup $acquire_release_workgroup_memory; };
}

void store_pair(uint slot, float x, float y) {
    uint hi = f32tof16(x) | (f32tof16(y) << 16u);
    packed[slot] = hi;
    if (kCompensated) {
        packed[2048u + slot] = f32tof16((x - widen_half(hi)) * 2048.0f) |
                               (f32tof16((y - widen_half(hi >> 16u)) * 2048.0f) << 16u);
    }
}

uint tile_base(uint begin, uint max_ctx) {
    // Matrices always load 16 complete rows. At the allocation boundary, reuse
    // preceding addresses and mask those lanes when publishing the scores.
    return min(begin, max_ctx - 16u);
}

bool visible(uint begin, uint key, DenseRangeIter range) {
    return key >= max(begin, range.start) && key < min(begin + 16u, range.start + range.len);
}

// A masked zero times a non-finite V can poison a matrix accumulator for an
// earlier query. Recompute only affected channels over the visible keys.
float4 masked_value(AttentionArgs a, Ptr<half> v, uint wave, uint wave_base,
                    uint row, uint channel, DenseRangeIter range) {
    float4 result = 0.0f;
    for (uint t = 0u; t < 4u; ++t) {
        uint begin = wave_base + 16u * t, base = tile_base(begin, a.max_ctx);
        for (uint j = 0u; j < 16u; ++j) {
            uint key = base + j;
            if (!visible(begin, key, range)) continue;
            uint slot = (wave * 4u + t) * 128u + row * 8u + j / 2u;
            uint shift = 16u * (j % 2u);
            float p = f16tof32((packed[slot] >> shift) & 65535u);
            if (kCompensated) p += widen_half(packed[2048u + slot] >> shift) / 2048.0f;
            float4 val = load_cache4(Ptr<CacheVector, Access.Read>(v + uint64_t(key) * a.v_row_stride + channel));
            result = fma(float4(p), val, result);
        }
    }
    return result;
}

// A subgroup works on 16 queries and 64 keys. Interior tiles omit per-score
// bounds checks. Causal, sliding-window and allocation boundaries keep them.
float normalize_tile<let Masked : bool>(AttentionArgs a, uint wave_base, DenseRangeIter range,
                                       float scale, inout float high, inout float mass) {
    uint lane = WaveGetLaneIndex(), wave = WaveGetWaveIndex();
    uint row = lane / 2u, side = lane % 2u;
    float scores[4][8], next_high = high, tile_mass = 0.0f;
    [ForceUnroll] for (uint t = 0u; t < 4u; ++t) {
        uint begin = wave_base + 16u * t, base = tile_base(begin, a.max_ctx);
        [ForceUnroll] for (uint j = 0u; j < 8u; ++j) {
            uint col = side * 8u + j;
            float dot = product[wave * kWaveFloats + (16u * t + row) * kScoreStride + col] * scale;
            scores[t][j] = (!Masked || visible(begin, base + col, range)) ? dot * a.scale : asfloat(0xff800000u);
            next_high = max(next_high, scores[t][j]);
        }
    }
    next_high = max(next_high, WaveReadLaneAt(next_high, lane ^ 1u));
    float carry = mass == 0.0f ? 0.0f : exp(high - next_high);
    [ForceUnroll] for (uint t = 0u; t < 4u; ++t) {
        uint begin = wave_base + 16u * t, base = tile_base(begin, a.max_ctx);
        [ForceUnroll] for (uint pair = 0u; pair < 4u; ++pair) {
            uint j = pair * 2u, key = base + side * 8u + j;
            float x = (!Masked || visible(begin, key, range)) ? exp(scores[t][j] - next_high) : 0.0f;
            float y = (!Masked || visible(begin, key + 1u, range)) ? exp(scores[t][j + 1u] - next_high) : 0.0f;
            store_pair((wave * 4u + t) * 128u + row * 8u + side * 4u + pair, x, y);
            tile_mass += x + y;
        }
    }
    mass = mass * carry + (tile_mass + WaveReadLaneAt(tile_mass, lane ^ 1u));
    high = next_high;
    return carry;
}

float pack_query(Ptr<float4, Access.Read> query, uint lane, uint wave, uint row, uint side) {
    float scale = 0.0f;
    [ForceUnroll] for (uint i = 0u; i < 8u; ++i) {
        float4 q = abs(load_a16(query + 8u * side + i));
        scale = max(scale, max(max(q.x, q.y), max(q.z, q.w)));
    }
    scale = max(scale, WaveReadLaneAt(scale, lane ^ 1u));
    if (scale == 0.0f) scale = 1.0f;
    if (wave == 0u) {
        [ForceUnroll] for (uint i = 0u; i < 8u; ++i) {
            float4 q = load_a16(query + 8u * side + i) / scale;
            uint slot = row * 32u + side * 16u + i * 2u;
            store_pair(slot, q.x, q.y);
            store_pair(slot + 1u, q.z, q.w);
        }
    }
    return scale;
}

void merge_waves(AttentionArgs a, uint first, uint head, uint row, uint side, uint wave) {
    if (first + row < a.rows) {
        float largest = asfloat(0xff800000u);
        [ForceUnroll] for (uint w = 0u; w < 4u; ++w) largest = max(largest, asfloat(packed[w * 512u + row]));
        float total = 0.0f;
        float4 sum[2];
        sum[0] = 0.0f;
        sum[1] = 0.0f;
        [ForceUnroll] for (uint w = 0u; w < 4u; ++w) {
            float m = asfloat(packed[w * 512u + 16u + row]);
            float weight = m != 0.0f ? exp(asfloat(packed[w * 512u + row]) - largest) : 0.0f;
            total += m * weight;
            [ForceUnroll] for (uint i = 0u; i < 2u; ++i) {
                uint x = w * kWaveFloats + row * kOutputStride + wave * 16u + side * 8u + i * 4u;
                sum[i] += weight * float4(product[x], product[x + 1u], product[x + 2u], product[x + 3u]);
            }
        }
        Ptr<float> y = a.y + uint64_t(first + row) * a.o_stride + uint64_t(head) * 64u + wave * 16u + side * 8u;
        [ForceUnroll] for (uint i = 0u; i < 2u; ++i)
            store_f32x4_maybe_a16(y + i * 4u, total != 0.0f ? sum[i] / total : float4(0.0f));
    }
}

AccTile load_acc(uint offset, uint stride, uint layout) {
    return spirv_asm {
        OpExtension "SPV_KHR_cooperative_matrix";
        OpCapability CooperativeMatrixKHR;
        %ptr : $$Ptr<float, Access.ReadWrite, AddressSpace.GroupShared> = OpAccessChain &product $offset;
        result : $$AccTile = OpCooperativeMatrixLoadKHR %ptr $layout $stride;
    };
}

[shader("compute")] [numthreads(128, 1, 1)]
void VELRI_PREFILL_KERNEL_ENTRY(uniform AttentionArgs a, uint3 group : SV_GroupID) {
    uint lane = WaveGetLaneIndex(), wave = WaveGetWaveIndex();
    uint first = group.x * 16u, head = group.y, kv_head = head / (a.n_heads / a.n_kv_heads);
    uint row = lane / 2u, side = lane % 2u, src_row = min(first + row, a.rows - 1u);
    uint pos = attention_kv_pos(a);
    DenseRangeIter range = make_dense_range_iter(pos, src_row, a.rows, a.max_ctx, a.sliding_window, a.causal);
    DenseRangeIter first_range = make_dense_range_iter(pos, first, a.rows, a.max_ctx, a.sliding_window, a.causal);
    DenseRangeIter last_range = make_dense_range_iter(pos, min(first + 15u, a.rows - 1u), a.rows,
                                                       a.max_ctx, a.sliding_window, a.causal);
    uint end = last_range.start + last_range.len;
    Ptr<float4, Access.Read> query = Ptr<float4, Access.Read>(a.q + uint64_t(src_row) * a.q_stride + uint64_t(head) * 64u);
    Ptr<half> k = a.k_cache + uint64_t(kv_head) * a.k_head_stride;
    Ptr<half> v = a.v_cache + uint64_t(kv_head) * a.v_head_stride;

    float scale = pack_query(query, lane, wave, row, side);
    GroupMemoryBarrierWithGroupSync();
    LeftTile q_tile[4], q_low[4];
    [ForceUnroll] for (uint i = 0u; i < 4u; ++i) {
        q_tile[i] = load_left(8u * i, 32u);
        if (kCompensated) q_low[i] = load_left(2048u + 8u * i, 32u);
    }
    GroupMemoryBarrierWithGroupSync();

    float high = asfloat(0xff800000u), mass = 0.0f;
    AccTile out_acc[4];
    [ForceUnroll] for (uint c = 0u; c < 4u; ++c) out_acc[c] = AccTile(0.0f);
    for (uint block = first_range.start; block < end;) {
        uint wave_base = block + 64u * wave;
        if (wave_base >= end) break;
        [ForceUnroll] for (uint t = 0u; t < 4u; ++t) {
            uint base = tile_base(wave_base + 16u * t, a.max_ctx);
            AccTile sum = AccTile(0.0f), correction = AccTile(0.0f);
            [ForceUnroll] for (uint i = 0u; i < 4u; ++i) {
                RightTile key = RightTile.Load<linalg::CoopMatMatrixLayout.ColumnMajor>(
                    k + uint64_t(base) * a.k_row_stride + 16u * i, 0u, a.k_row_stride);
                sum = linalg::coopMatMulAdd<float, false>(q_tile[i], key, sum);
                if (kCompensated) correction = linalg::coopMatMulAdd<float, false>(q_low[i], key, correction);
            }
            if (kCompensated) sum = sum + correction * (1.0f / 2048.0f);
            store_product(sum, wave * kWaveFloats + 16u * t * kScoreStride, kScoreStride);
        }
        sync_subgroup_shared();

        bool interior = wave_base >= last_range.start && wave_base + 64u <= first_range.start + first_range.len;
        float carry;
        if (interior)
            carry = normalize_tile<false>(a, wave_base, range, scale, high, mass);
        else
            carry = normalize_tile<true>(a, wave_base, range, scale, high, mass);
        sync_subgroup_shared();

        AccTile sums[4], corrections[4];
        // Column-major stride zero broadcasts one carry per row across all channels.
#ifdef VELRI_HALF_ATTENTION_OPERANDS
        if (side == 0u) product[wave * kWaveFloats + row] = carry;
        sync_subgroup_shared();
        AccTile alpha = load_acc(wave * kWaveFloats, 0u, 1u);
        [ForceUnroll] for (uint c = 0u; c < 4u; ++c) {
            out_acc[c] = out_acc[c] * alpha;
            sums[c] = interior ? out_acc[c] : AccTile(0.0f);
            corrections[c] = AccTile(0.0f);
        }
#else
        // Carry is often unity across the tile. Apply rounding corrections to this
        // tile's PV sum before adding it to the running output.
        if (!WaveActiveAllTrue(carry == 1.0f)) {
            if (side == 0u) product[wave * kWaveFloats + row] = carry;
            sync_subgroup_shared();
            AccTile alpha = load_acc(wave * kWaveFloats, 0u, 1u);
            [ForceUnroll] for (uint c = 0u; c < 4u; ++c) out_acc[c] = out_acc[c] * alpha;
        }
        [ForceUnroll] for (uint c = 0u; c < 4u; ++c) {
            sums[c] = AccTile(0.0f);
            corrections[c] = AccTile(0.0f);
        }
#endif
        [ForceUnroll] for (uint t = 0u; t < 4u; ++t) {
            uint base = tile_base(wave_base + 16u * t, a.max_ctx);
            uint slot = (wave * 4u + t) * 128u;
            LeftTile prob = load_left(slot, 8u), low;
            if (kCompensated) low = load_left(2048u + slot, 8u);
            [ForceUnroll] for (uint c = 0u; c < 4u; ++c) {
                RightTile value = RightTile.Load<linalg::CoopMatMatrixLayout.RowMajor>(
                    v + uint64_t(base) * a.v_row_stride + 16u * c, 0u, a.v_row_stride);
                sums[c] = linalg::coopMatMulAdd<float, false>(prob, value, sums[c]);
                if (kCompensated) corrections[c] = linalg::coopMatMulAdd<float, false>(low, value, corrections[c]);
            }
        }
        if (kCompensated) {
            [ForceUnroll] for (uint c = 0u; c < 4u; ++c)
                sums[c] = sums[c] + corrections[c] * (1.0f / 2048.0f);
        }
        if (interior) {
            [ForceUnroll] for (uint c = 0u; c < 4u; ++c) {
                if (kCompensated) out_acc[c] = out_acc[c] + sums[c];
                else out_acc[c] = sums[c];
            }
        } else {
            // A masked non-finite V may contaminate this tile before it joins the running output.
            [ForceUnroll] for (uint c = 0u; c < 4u; ++c)
                store_product(sums[c], wave * kWaveFloats + 16u * c, kOutputStride);
            sync_subgroup_shared();
            [ForceUnroll] for (uint i = 0u; i < 8u; ++i) {
                uint channel = side * 32u + i * 4u;
                uint x = wave * kWaveFloats + row * kOutputStride + channel;
                float4 v4 = float4(product[x], product[x + 1u], product[x + 2u], product[x + 3u]);
                if (!all(isfinite(v4))) {
                    v4 = masked_value(a, v, wave, wave_base, row, channel, range);
                    [ForceUnroll] for (uint j = 0u; j < 4u; ++j) product[x + j] = v4[j];
                }
            }
            sync_subgroup_shared();
            [ForceUnroll] for (uint c = 0u; c < 4u; ++c)
                out_acc[c] = out_acc[c] + load_acc(wave * kWaveFloats + 16u * c, kOutputStride, 0u);
        }
        if (end - block <= 256u) break;
        block += 256u;
    }
    [ForceUnroll] for (uint c = 0u; c < 4u; ++c)
        store_product(out_acc[c], wave * kWaveFloats + 16u * c, kOutputStride);
    if (side == 0u) {
        packed[wave * 512u + row] = asuint(high);
        packed[wave * 512u + 16u + row] = asuint(mass);
    }
    GroupMemoryBarrierWithGroupSync();

    merge_waves(a, first, head, row, side, wave);
}
