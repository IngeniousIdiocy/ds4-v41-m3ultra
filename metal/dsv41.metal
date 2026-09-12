// DeepSeek V4.1 keeps the RoPE tail in the quantized KV vector. Unlike V4,
// indexer Q/K have no Hadamard transform, and compressed KV has E4M3 scales.
struct ds4_metal_args_dsv41_quantize {
    uint width;
    uint rows;
    uint mode;
};

static inline float dsv41_bf16(float x) {
    uint bits = as_type<uint>(x);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16u) & 1u);
    return as_type<float>(bits & 0xffff0000u);
}

static inline float dsv41_pow2_ceil(float x) {
    const uint bits = as_type<uint>(x);
    return as_type<float>((bits & 0x7f800000u) +
                         ((bits & 0x7fffffu) ? 0x800000u : 0u));
}

kernel void kernel_dsv41_bf16_linear(
        constant ulong &count,
        device uint *x,
        uint gid [[thread_position_in_grid]]) {
    const ulong first = (ulong)gid * 4u;
    if (first + 4u <= count) {
        uint4 bits = *((device uint4 *)(x + first));
        const bool4 finite = (bits & 0x7f800000u) != 0x7f800000u;
        bits += select(uint4(0), uint4(0x7fffu) + ((bits >> 16u) & 1u), finite);
        *((device uint4 *)(x + first)) = bits & 0xffff0000u;
    } else {
        for (ulong i = first; i < count; i++) {
            uint bits = x[i];
            if ((bits & 0x7f800000u) != 0x7f800000u)
                bits += 0x7fffu + ((bits >> 16u) & 1u);
            x[i] = bits & 0xffff0000u;
        }
    }
}

struct ds4_metal_args_dsv41_rope {
    uint width, heads, rows, start, inverse, stride;
    float frequencies[32];
};

kernel void kernel_dsv41_rope(
        constant ds4_metal_args_dsv41_rope &args,
        device float *x,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const float theta = float(args.start + group.y * args.stride) * args.frequencies[lane];
    const float c = precise::cos(theta);
    const float s = args.inverse ? -precise::sin(theta) : precise::sin(theta);
    const ulong i = ((ulong)group.y * args.heads + group.x) * args.width +
                    args.width - 64u + 2u * lane;
    const float re = x[i], im = x[i + 1u];
    x[i] = dsv41_bf16(re * c - im * s);
    x[i + 1u] = dsv41_bf16(re * s + im * c);
}

kernel void kernel_dsv41_quantize(
        constant ds4_metal_args_dsv41_quantize &args,
        device float *x,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const uint block = args.mode == 3u ? 16u : 32u;
    const uint column = group.x * block + lane;
    const bool valid = lane < block && column < args.width;
    const ulong index = (ulong)group.y * args.width + column;
    const float value = valid ? dsv41_bf16(x[index]) : 0.0f;
    const float amax = simd_max(abs(value));
    float result = value;
    if (args.mode == 1u) {
        const float scale = dsv41_pow2_ceil(max(amax, 1.0e-4f) * (1.0f / 448.0f));
        result = copysign(dsv4_e4m3fn_dequant(abs(value) / scale), value) * scale;
    } else if (args.mode == 2u || args.mode == 3u) {
        const float scale = args.mode == 3u
            ? dsv4_e4m3fn_dequant(max(amax, 0.01171875f) / 6.0f)
            : dsv41_pow2_ceil(max(amax, 7.052966104933725e-38f) * (1.0f / 6.0f));
        result = copysign(dsv4_e2m1fn_dequant(abs(value) / scale), value) * scale;
    }
    if (valid) x[index] = dsv41_bf16(result);
}

struct ds4_metal_args_dsv41_engram {
    uint width;
    uint rows;
    float eps;
    uint masked;
};

kernel void kernel_dsv41_engram_add(
        constant ds4_metal_args_dsv41_engram &args,
        device float *residual,
        device const float *kv,
        device const float *q_weight,
        device const float *k_weight,
        device const uchar *mask,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    if (args.masked && !mask[group.x]) return;
    const ulong offset = ((ulong)group.x * 4u + group.y) * args.width;
    const ulong key_offset = ((ulong)group.x * 5u + group.y) * args.width;
    const ulong value_offset = ((ulong)group.x * 5u + 4u) * args.width;
    float h2 = 0.0f, k2 = 0.0f, dot = 0.0f;
    for (uint i = lane; i < args.width; i += 32u) {
        const float h = residual[offset + i];
        const float k = dsv41_bf16(kv[key_offset + i]);
        const uint wi = group.y * args.width + i;
        h2 += h * h;
        k2 += k * k;
        dot += h * (q_weight[wi] * k_weight[wi]) * k;
    }
    h2 = simd_sum(h2);
    k2 = simd_sum(k2);
    dot = simd_sum(dot) * rsqrt(h2 / args.width + args.eps) *
          rsqrt(k2 / args.width + args.eps) * rsqrt(float(args.width));
    const float gate = 1.0f / (1.0f + exp(-copysign(sqrt(max(abs(dot), 1.0e-6f)), dot)));
    for (uint i = lane; i < args.width; i += 32u)
        residual[offset + i] = dsv41_bf16(residual[offset + i] +
            gate * dsv41_bf16(kv[value_offset + i]));
}

struct ds4_metal_args_dsv41_pool {
    uint width;
    uint pairs;
    uint tail;
};

kernel void kernel_dsv41_pool2(
        constant ds4_metal_args_dsv41_pool &args,
        device float *out,
        device const float *kv,
        device const float *scores,
        device const float *previous_kv,
        device const float *previous_scores,
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= args.width || index.y >= args.pairs) return;
    const long a = (long)index.y * 2 - args.tail;
    const ulong b = (ulong)(a + 1) * args.width + index.x;
    const float ka = a < 0 ? previous_kv[index.x] : kv[(ulong)a * args.width + index.x];
    const float sa = a < 0 ? previous_scores[index.x] : scores[(ulong)a * args.width + index.x];
    const float sb = scores[b], peak = max(sa, sb);
    const float ea = exp(sa - peak), eb = exp(sb - peak);
    out[(ulong)index.y * args.width + index.x] = dsv41_bf16((ka * ea + kv[b] * eb) / (ea + eb));
}

struct ds4_metal_args_dsv41_candidates {
    uint width;
    uint rows;
    uint start;
    uint ratio;
};

kernel void kernel_dsv41_candidate_blocks(
        constant ds4_metal_args_dsv41_candidates &args,
        device const float *scores,
        device float *blocks,
        device const float *unused,
        uint2 index [[thread_position_in_grid]]) {
    (void)unused;
    const uint count = (args.width + 7u) / 8u;
    if (index.x >= count || index.y >= args.rows) return;
    const uint visible = min(args.width, (args.start + index.y + 1u) / args.ratio);
    float best = -INFINITY;
    for (uint i = index.x * 8u; i < min(visible, (index.x + 1u) * 8u); i++)
        best = max(best, scores[(ulong)index.y * args.width + i]);
    if (visible && index.x == (visible - 1u) / 8u) best = INFINITY;
    blocks[(ulong)index.y * count + index.x] = best;
}

kernel void kernel_dsv41_candidate_filter(
        constant ds4_metal_args_dsv41_candidates &args,
        device const float *scores,
        device float *out,
        device const float *block_mask,
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= args.width || index.y >= args.rows) return;
    const ulong offset = (ulong)index.y * args.width + index.x;
    const uint blocks = (args.width + 7u) / 8u;
    const uint visible = min(args.width, (args.start + index.y + 1u) / args.ratio);
    out[offset] = index.x < visible &&
        block_mask[(ulong)index.y * blocks + index.x / 8u] == 0.0f
        ? scores[offset] : -INFINITY;
}

struct ds4_metal_args_dsv41_carry {
    uint width, rows, words, format, pack;
};

kernel void kernel_dsv41_carry_copy(
        constant ds4_metal_args_dsv41_carry &args,
        device uint *packed, device float *plain,
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint col = group.x * 128u + tid;
    const ulong row = group.y;
    if (args.format == 0u) {
        if (col >= args.width) return;
        device ushort *p = (device ushort *)(packed + row * args.words);
        if (args.pack) p[col] = ushort(as_type<uint>(plain[row * args.width + col]) >> 16);
        else plain[row * args.width + col] = as_type<float>(uint(p[col]) << 16);
    } else {
        const uint word = col / 32u;
        if (args.pack) {
            const bool allowed = col < args.width && plain[row * args.width + col] == 0.0f;
            const uint bits = simd_sum(allowed ? 1u << lane : 0u);
            if (!lane && word < args.words) packed[row * args.words + word] = bits;
        } else if (col < args.width) {
            const uint bits = packed[row * args.words + word];
            plain[row * args.width + col] = bits & (1u << lane) ? 0.0f : -INFINITY;
        }
    }
}

#ifdef DS4_METAL_HAS_TENSOR
kernel void kernel_dsv41_indexer_pack(
        constant uint4 &args,
        device const float *q, device const float *keys,
        device uint *flags, device bfloat *packed_q, device bfloat *packed_keys,
        threadgroup uint *valid [[threadgroup(0)]],
        uint group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    const bool query = group < args.y;
    const uint count = query ? 32u * 128u : 64u * 128u;
    const ulong offset = query ? (ulong)group * count : (ulong)(group - args.y) * count;
    bool exact = true;
    for (uint i = tid; i < count; i += 128u) {
        const float value = query ? q[offset + i] :
            offset + i < (ulong)args.x * 128u ? keys[offset + i] : 0.0f;
        const bfloat converted = bfloat(value);
        if (query) packed_q[offset + i] = converted;
        else packed_keys[offset + i] = converted;
        exact = exact && float(converted) == value;
    }
    const bool same = simd_all(exact);
    if (!(tid % 32u)) valid[tid / 32u] = same;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!tid) flags[group] = valid[0] && valid[1] && valid[2] && valid[3];
}

kernel void kernel_dsv41_indexer_scores_packed(
        constant uint4 &args, constant uint2 &range,
        device const float *q, device const float *weights,
        device const float *keys, device float *scores,
        device const uint *flags, device const bfloat *packed_q,
        device const bfloat *packed_keys,
        threadgroup float *scratch [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int HEADS = 32, KEYS = 64, DIM = 128;
    const uint width = args.x, token = group.y, row0 = group.x * KEYS;
    const uint visible = (args.z + token + 1u) / args.w;
    if (row0 >= visible) {
        if (tid < KEYS && row0 + tid < width)
            scores[(ulong)token * width + row0 + tid] = -INFINITY;
        return;
    }
    matmul2d<matmul2d_descriptor(HEADS, KEYS, DIM, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> mm;
    auto result = tensor(scratch, dextents<int32_t, 2>(KEYS, HEADS));
    if (flags[range.y + token] && flags[range.x + group.x]) {
        auto query = tensor((device bfloat *)packed_q + (ulong)(range.y + token) * HEADS * DIM,
                             dextents<int32_t, 2>(DIM, HEADS));
        auto key = tensor((device bfloat *)packed_keys + (ulong)row0 * DIM,
                           dextents<int32_t, 2>(DIM, KEYS));
        auto dots = mm.template get_destination_cooperative_tensor<decltype(query), decltype(key), float>();
        for (uint16_t i = 0; i < dots.get_capacity(); i++)
            if (dots.is_valid_element(i)) dots[i] = 0;
        mm.run(query, key, dots);
        dots.store(result);
    } else {
        auto query = tensor((device float *)q + (ulong)token * HEADS * DIM,
                             dextents<int32_t, 2>(DIM, HEADS));
        auto all_keys = tensor((device float *)keys, dextents<int32_t, 2>(DIM, int(width)));
        auto key = all_keys.slice(0, int(row0));
        auto dots = mm.template get_destination_cooperative_tensor<decltype(query), decltype(key), float>();
        for (uint16_t i = 0; i < dots.get_capacity(); i++)
            if (dots.is_valid_element(i)) dots[i] = 0;
        mm.run(query, key, dots);
        dots.store(result);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < KEYS && row0 + tid < width) {
        float sum = 0;
        for (uint h = 0; h < HEADS; h++)
            sum += max(scratch[h * KEYS + tid] * (1.0f / 64.0f), 0.0f) * weights[token * HEADS + h];
        scores[(ulong)token * width + row0 + tid] = row0 + tid < visible ? sum : -INFINITY;
    }
}
#endif

/* ---------------------------------------------------------------------------
 * W2 / D2 -- the V4.1 router chain in one dispatch.
 *
 * V4 Flash collapses its router into kernel_dsv4_router_transform_finalize_
 * weights_one_simd, but that kernel is hard-wired to 256 experts and V4.1 has
 * 384, so V4.1 decode still runs the generic nine-dispatch chain: softplus,
 * sqrt, add bias, kernel_argsort_f32_i32_desc over 512 threads, get_rows,
 * sum_rows, clamp, divide, scale.  This kernel is that chain, textually, in one
 * 512-thread threadgroup:
 *
 *   - the softplus and sqrt expressions are the generic unary kernel's
 *     (metal/unary.metal:245 and :168), applied back to back in registers; both
 *     are single f32 operations, so removing the intermediate store cannot
 *     change a bit;
 *   - the bitonic network below is kernel_argsort_f32_i32<DS4_SORT_ORDER_DESC>
 *     with shuffle = false, causal = false, i00 = 0, ntg.x = 512, width = 384 --
 *     the same stages, the same comparisons, the same out-of-range guard, so
 *     the permutation, and therefore ties, are identical;
 *   - the weight tail reproduces sum_rows' simd_sum over lanes 0..5 (the same
 *     instruction on the same lane layout), then clamp, then divide, then
 *     scale, in that order.
 *
 * DS4_DS41_ROUTER_FUSED=0 (lever router_fused) restores the nine dispatches.
 * ------------------------------------------------------------------------ */
struct ds4_metal_args_dsv41_router_one {
    uint32_t n_expert;
    uint32_t n_expert_used;
    uint32_t has_bias;
    uint32_t write_weights;
    float    scale;
};

kernel void kernel_dsv41_router_select_one_384(
        constant ds4_metal_args_dsv41_router_one & args,
        device const float *logits,
        device       float *probs,
        device const float *bias,
        device     int32_t *selected,
        device       float *weights,
        threadgroup   char *shmem [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]]) {
    constexpr int NT = 512;                 /* argsort threadgroup width */
    const int width = (int)args.n_expert;   /* 384 */

    threadgroup int32_t *sidx =
        (threadgroup int32_t *)shmem;
    threadgroup float *sscore =
        (threadgroup float *)(shmem + NT * sizeof(int32_t));
    threadgroup float *sprob =
        (threadgroup float *)(shmem + NT * sizeof(int32_t) + NT * sizeof(float));

    const int col = (int)tid;
    sidx[col] = col;
    if (col < width) {
        const float x = logits[col];
        const float sp = select(log(1.0f + exp(x)), x, x > 20.0f);
        const float p = sqrt(sp);
        probs[col] = p;
        sprob[col] = p;
        sscore[col] = args.has_bias ? p + bias[col] : p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k = 2; k <= NT; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            const int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (sidx[col] >= width ||
                        (sidx[ixj] < width &&
                         sscore[sidx[col]] < sscore[sidx[ixj]])) {
                        const int32_t t = sidx[col];
                        sidx[col] = sidx[ixj];
                        sidx[ixj] = t;
                    }
                } else {
                    if (sidx[ixj] >= width ||
                        (sidx[col] < width &&
                         sscore[sidx[col]] > sscore[sidx[ixj]])) {
                        const int32_t t = sidx[col];
                        sidx[col] = sidx[ixj];
                        sidx[ixj] = t;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const int used = (int)args.n_expert_used;
    if (col < used) {
        selected[col] = sidx[col];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* sum_rows stages element j in lane j of one simdgroup and reduces with
     * simd_sum; reproduce that lane layout rather than a serial sum. */
    /* sum_rows is dispatched with exactly n_expert_used threads (nth is clamped
     * to ne00 in ds4_gpu_encode_sum_rows_f32), so its simd_sum runs with only
     * those lanes active; reproduce that by calling simd_sum under the same
     * lane predicate rather than padding to a full simdgroup.  The divide and
     * the scale were two kernels with an f32 store between them, so the
     * intermediate is round-tripped through threadgroup memory to keep that
     * rounding boundary. */
    threadgroup volatile float *wtmp =
        (threadgroup volatile float *)(shmem + NT * sizeof(int32_t));
    if (args.write_weights && col < used) {
        const float wv = sprob[sidx[col]];
        const float total = simd_sum(wv);
        const float clamped = clamp(total, 6.103515625e-5f, INFINITY);
        wtmp[col] = wv / clamped;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (args.write_weights && col < used) {
        weights[col] = wtmp[col] * args.scale;
    }
}

/* ---------------------------------------------------------------------------
 * R9 - the V4.1 shared expert's gate and up Q8_0 projections and the SwiGLU
 * that consumes them, in one dispatch.
 *
 * V4's kernel_dsv4_shared_gate_up_swiglu_q8_0 (metal/dense.metal) is the shape
 * template and is reproduced here unchanged: NQ 8, NR0 2, NSG from
 * FC_mul_mv_nsg (4 at decode), 32 x NSG threads, the same K walk, and
 * helper_mv_reduce_and_write's reduction tree (zero the row's scratch from
 * simdgroup 0, simd_sum the per-lane partial, publish it at [sgitg], barrier,
 * simd_sum the published values) in two disjoint 256-byte scratch regions.
 *
 * What V4.1 adds is its rounding boundaries.  The chain this replaces is
 *
 *   ds41_matmul(shared_gate, ffn_gate_shexp, norm, round = true)
 *   ds41_matmul(shared_up,   ffn_up_shexp,   norm, round = true)
 *   ds4_gpu_swiglu_tensor(shared_mid, shared_gate, shared_up, limit, alpha)
 *   ds41_bf16(shared_mid)
 *
 * so gate and up are BF16-rounded *before* the SwiGLU reads them, the SwiGLU
 * is kernel_swiglu_flat_f32's expression (min / clamp under the same
 * limit > 1e-6 guard, x0 / (1 + exp(-x0)), then * x1 * alpha), and the product
 * is BF16-rounded on the way to memory.  All four boundaries are reproduced in
 * the lane that owns the output row, which is what makes the fused dispatch
 * bit-identical to the six it replaces.  shared_gate and shared_up are dead
 * after the SwiGLU in the V4.1 decode graph, so they are not written.
 * ------------------------------------------------------------------------ */
kernel void kernel_dsv41_shared_mid_swiglu_q8_0(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_mid,
        constant     float & clamp_value,
        constant     float & alpha,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short NR0 = N_R0_Q8_0;
    const short NSG = FC_mul_mv_nsg;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;

    const int nb = args.ne00 / QK8_0;
    const int r0 = tgpig.x * NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;
    device const float *y = (device const float *)(src1 + offset1);

    device const block_q8_0 *ag[NR0];
    device const block_q8_0 *au[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row) * args.nb01 +
                                 (i12 / args.r2) * args.nb02 +
                                 (i13 / args.r3) * args.nb03;
        ag[row] = (device const block_q8_0 *)((device const char *)src0_gate + offset0);
        au[row] = (device const block_q8_0 *)((device const char *)src0_up   + offset0);
    }

    float sumg[NR0] = { 0.f };
    float sumu[NR0] = { 0.f };

    const short ix = tiisg / (NW / NQ);
    const short il = tiisg % (NW / NQ);
    const int ib0 = sgitg * NQ + ix;
    float yl[NQ];
    device const float *yb = y + ib0 * QK8_0 + il * NQ;

    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        FOR_UNROLL (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            device const int8_t *qg = ag[row][ib].qs + il * NQ;
            device const int8_t *qu = au[row][ib].qs + il * NQ;

            float sg = 0.f;
            float su = 0.f;
            FOR_UNROLL (short i = 0; i < NQ; ++i) {
                sg += qg[i] * yl[i];
                su += qu[i] * yl[i];
            }

            sumg[row] += sg * ag[row][ib].d;
            sumu[row] += su * au[row][ib].d;
        }

        yb += NSG * NQ * QK8_0;
    }

    threadgroup float *shmem_f32 = (threadgroup float *)shmem;
    threadgroup float *sh_gate[NR0];
    threadgroup float *sh_up[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        sh_gate[row] = shmem_f32 + NW * row;
        sh_up[row]   = shmem_f32 + NW * (NR0 + row);
        if (sgitg == 0) {
            sh_gate[row][tiisg] = 0.0f;
            sh_up[row][tiisg] = 0.0f;
        }
        sumg[row] = simd_sum(sumg[row]);
        sumu[row] = simd_sum(sumu[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            sh_gate[row][sgitg] = sumg[row];
            sh_up[row][sgitg] = sumu[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *mid_f32 = (device float *)dst_mid +
        (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;

    FOR_UNROLL (short row = 0; row < NR0 && r0 + row < args.ne01; ++row) {
        const float gate = simd_sum(sh_gate[row][tiisg]);
        const float up = simd_sum(sh_up[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            const uint out_row = r0 + row;
            /* ds41_matmul(..., round = true) on both projections. */
            float x0 = dsv41_bf16(gate);
            float x1 = dsv41_bf16(up);
            /* kernel_swiglu_flat_f32, limit = DS4_SWIGLU_CLAMP_EXP. */
            if (clamp_value > 1.0e-6f) {
                x0 = min(x0, clamp_value);
                x1 = clamp(x1, -clamp_value, clamp_value);
            }
            const float silu = x0 / (1.0f + exp(-x0));
            /* ds41_bf16(shared_mid). */
            mid_f32[out_row] = dsv41_bf16(silu * x1 * alpha);
        }
    }
}

/* ---------------------------------------------------------------------------
 * R1a dispatch B - the V4.1 HC tail: the new sublayer's Sinkhorn split, the
 * collapse of the four residual streams with the PRECEDING sublayer's pre
 * coefficients, and the weighted RMS norm that consumes the collapse, in one
 * threadgroup.
 *
 * The two halves are independent in V4.1's graph: `ds41_hc_mix` writes this
 * call's split (attn_split / ffn_split) while the collapse that follows it
 * reads the previous sublayer's coefficients (`g->pre` before attention,
 * `attn_split` before the FFN; ds4.c:39993 / :40007, carried forward at
 * :40285).  That is why one dispatch can carry both without reordering
 * anything.
 *
 * Each of the three stages is the production kernel's body, unchanged:
 *   - Sinkhorn: kernel_dsv4_hc_split_sinkhorn's HC == 4 float4 branch
 *     (metal/dsv4_hc.metal:118), including ds4_hc_sigmoid / ds4_hc_twice_sigmoid
 *     -- which follow DS4_METAL_HC_STABLE exactly as the standalone kernel does
 *     -- the initial row max/exp normalization, the loop from iter = 1, and the
 *     reciprocal and epsilon placement.  One designated lane, as today.
 *   - Collapse: kernel_dsv4_hc_weighted_sum_bf16's `acc += x_h * w_h` in h
 *     order followed by dsv41_bf16 on the store (D1's boundary).
 *   - Weighted RMS: kernel_rms_norm_fuse_impl<float4, 4>'s reduction tree at
 *     1024 threads (ds4_gpu_rms_norm_threads(5120) == 1024), reading the
 *     BF16-rounded collapse, and its (x*scale)*w store with dsv41_bf16 applied
 *     per component.
 * The rounded collapse is kept in threadgroup memory so the RMS reads the same
 * words the separate pass would have re-read from device memory.
 * ------------------------------------------------------------------------ */
struct ds4_metal_args_dsv41_hc_tail {
    int32_t n_embd;
    int32_t n_hc;
    int32_t sinkhorn_iters;
    float   hc_eps;
    float   norm_eps;
};

kernel void kernel_dsv41_hc_sinkhorn_collapse_norm(
        constant ds4_metal_args_dsv41_hc_tail & args,
        device const float * mix,
        device const float * scale,
        device const float * base,
        device       float * split,
        device const float * x,
        device const float * pre,
        device       float * collapsed,
        device const float * norm_weight,
        device       float * norm_dst,
        threadgroup  float * shared [[threadgroup(0)]],
        ushort tid   [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort ntg   [[threads_per_threadgroup]]) {
    const uint n_embd = (uint)args.n_embd;
    const uint n4 = n_embd >> 2;

    threadgroup float4 *row = (threadgroup float4 *)shared;
    threadgroup float  *sum_shmem = shared + n_embd;

    /* kernel_rms_norm_fuse_impl zeroes its 32 accumulator slots from
     * simdgroup 0 before the parallel sum. */
    if (sgitg == 0) {
        sum_shmem[tiisg] = 0.0f;
    }

    /* --- Sinkhorn on this sublayer's new mix, one lane, as today. --- */
    if (tid == 0) {
        const float epsv = args.hc_eps;
        const float pre_scale = scale[0];
        const float post_scale = scale[1];
        const float comb_scale = scale[2];

        const float4 pre_z =
            *((device const float4 *) mix) * pre_scale +
            *((device const float4 *) base);
        *((device float4 *) split) = ds4_hc_sigmoid(pre_z) + epsv;

        const float4 post_z =
            *((device const float4 *) (mix  + 4)) * post_scale +
            *((device const float4 *) (base + 4));
        *((device float4 *) (split + 4)) = ds4_hc_twice_sigmoid(post_z);

        float4 r0 = *((device const float4 *) (mix  +  8)) * comb_scale +
                    *((device const float4 *) (base +  8));
        float4 r1 = *((device const float4 *) (mix  + 12)) * comb_scale +
                    *((device const float4 *) (base + 12));
        float4 r2 = *((device const float4 *) (mix  + 16)) * comb_scale +
                    *((device const float4 *) (base + 16));
        float4 r3 = *((device const float4 *) (mix  + 20)) * comb_scale +
                    *((device const float4 *) (base + 20));

        const float m0 = max(max(r0.x, r0.y), max(r0.z, r0.w));
        const float m1 = max(max(r1.x, r1.y), max(r1.z, r1.w));
        const float m2 = max(max(r2.x, r2.y), max(r2.z, r2.w));
        const float m3 = max(max(r3.x, r3.y), max(r3.z, r3.w));

        r0 = exp(r0 - m0);
        r1 = exp(r1 - m1);
        r2 = exp(r2 - m2);
        r3 = exp(r3 - m3);

        r0 = r0 * (1.0f / (r0.x + r0.y + r0.z + r0.w)) + epsv;
        r1 = r1 * (1.0f / (r1.x + r1.y + r1.z + r1.w)) + epsv;
        r2 = r2 * (1.0f / (r2.x + r2.y + r2.z + r2.w)) + epsv;
        r3 = r3 * (1.0f / (r3.x + r3.y + r3.z + r3.w)) + epsv;

        float4 col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
        r0 *= col_inv;
        r1 *= col_inv;
        r2 *= col_inv;
        r3 *= col_inv;

        for (int iter = 1; iter < args.sinkhorn_iters; ++iter) {
            r0 *= 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);
            r1 *= 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);
            r2 *= 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);
            r3 *= 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);

            col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
            r0 *= col_inv;
            r1 *= col_inv;
            r2 *= col_inv;
            r3 *= col_inv;
        }

        *((device float4 *) (split +  8)) = r0;
        *((device float4 *) (split + 12)) = r1;
        *((device float4 *) (split + 16)) = r2;
        *((device float4 *) (split + 20)) = r3;
    }

    /* --- Collapse with the preceding sublayer's pre weights, then its BF16
     * boundary, and the weighted RMS partial sum over the rounded values. --- */
    const float w0 = pre[0], w1 = pre[1], w2 = pre[2], w3 = pre[3];
    device const float4 *x0 = (device const float4 *)(x + 0u * n_embd);
    device const float4 *x1 = (device const float4 *)(x + 1u * n_embd);
    device const float4 *x2 = (device const float4 *)(x + 2u * n_embd);
    device const float4 *x3 = (device const float4 *)(x + 3u * n_embd);
    device float4 *collapsed4 = (device float4 *)collapsed;

    float sumf = 0.0f;
    for (uint i = tid; i < n4; i += ntg) {
        float4 acc = 0.0f;
        acc += x0[i] * w0;
        acc += x1[i] * w1;
        acc += x2[i] * w2;
        acc += x3[i] * w3;
        const float4 v = float4(dsv41_bf16(acc.x), dsv41_bf16(acc.y),
                                dsv41_bf16(acc.z), dsv41_bf16(acc.w));
        collapsed4[i] = v;
        row[i] = v;
        sumf += dot(v, v);
    }

    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        sum_shmem[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = sum_shmem[tiisg];
    sumf = simd_sum(sumf);

    const float mean  = sumf / (float)args.n_embd;
    const float rscale = 1.0f / sqrt(mean + args.norm_eps);

    device const float4 *w = (device const float4 *)norm_weight;
    device float4 *y = (device float4 *)norm_dst;
    for (uint i = tid; i < n4; i += ntg) {
        const float4 v = (row[i] * rscale) * w[i];
        y[i] = float4(dsv41_bf16(v.x), dsv41_bf16(v.y),
                      dsv41_bf16(v.z), dsv41_bf16(v.w));
    }
}

/* ---------------------------------------------------------------------------
 * R1b - the HC expansion with its producer's BF16 boundary folded in.
 *
 * V4.1 rounds the block row to BF16 and then expands it:
 *   attention  ds41_bf16(g->block) (ds4.c:39940)   -> hc_expand_split_bf16
 *   FFN        add(routed, shared), ds41_bf16(g->block) -> hc_expand_split_bf16
 * The rounding pass is a separate 5-threadgroup dispatch over exactly the words
 * the expansion is about to read.  This kernel is kernel_dsv4_hc_expand4_bf16
 * with that round moved onto the load: the same dsv41_bf16 applied to the same
 * f32 word, the block row still written back (dsv41_bf16 is idempotent, so a
 * path that also rounds separately is unaffected), then the production
 * expansion association -- acc = block * post[dst], then += comb[dst][src] * r_src
 * for src 0..3 in that order -- and dsv41_bf16 on the store.
 * ------------------------------------------------------------------------ */
kernel void kernel_dsv41_hc_round_expand4(
        constant ds4_metal_args_dsv4_hc_expand & args,
        device  const char * block_in,
        device  const char * residual,
        device  const char * post,
        device  const char * comb,
        device  const char * block_add,
        device        char * block_out,
        device        char * dst,
        uint gid [[thread_position_in_grid]]) {
    if (args.n_hc != 4) {
        return;
    }

    const int64_t n_elem = args.n_embd * args.n_tokens;
    if ((int64_t) gid >= n_elem) {
        return;
    }

    const int64_t d = ((int64_t) gid) % args.n_embd;
    const int64_t t = ((int64_t) gid) / args.n_embd;

    float block_v = *((device const float *) (block_in + d*args.nb_block0 + t*args.nb_block1));
    if (args.has_add) {
        block_v += *((device const float *) (block_add + d*args.nb_add0 + t*args.nb_add1));
    }
    block_v = dsv41_bf16(block_v);
    *((device float *) (block_out + d*args.nb_block0 + t*args.nb_block1)) = block_v;

    const float r0 = *((device const float *) (residual + d*args.nb_res0 + 0*args.nb_res1 + t*args.nb_res2));
    const float r1 = *((device const float *) (residual + d*args.nb_res0 + 1*args.nb_res1 + t*args.nb_res2));
    const float r2 = *((device const float *) (residual + d*args.nb_res0 + 2*args.nb_res1 + t*args.nb_res2));
    const float r3 = *((device const float *) (residual + d*args.nb_res0 + 3*args.nb_res1 + t*args.nb_res2));

    for (int64_t dst_hc = 0; dst_hc < 4; ++dst_hc) {
        float acc = block_v * *((device const float *) (post + dst_hc*args.nb_post0 + t*args.nb_post1));

        acc += *((device const float *) (comb + dst_hc*args.nb_comb0 + 0*args.nb_comb1 + t*args.nb_comb2)) * r0;
        acc += *((device const float *) (comb + dst_hc*args.nb_comb0 + 1*args.nb_comb1 + t*args.nb_comb2)) * r1;
        acc += *((device const float *) (comb + dst_hc*args.nb_comb0 + 2*args.nb_comb1 + t*args.nb_comb2)) * r2;
        acc += *((device const float *) (comb + dst_hc*args.nb_comb0 + 3*args.nb_comb1 + t*args.nb_comb2)) * r3;

        *((device float *) (dst + d*args.nb0 + dst_hc*args.nb1 + t*args.nb2)) = dsv41_bf16(acc);
    }
}
