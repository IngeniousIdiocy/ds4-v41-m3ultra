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
/* Wave 5: ONE definition of the HC expansion accumulate.  The folded FFN
 * producer and kernel_dsv41_hc_round_expand4 compute the same five terms, but
 * as two separately written expressions the Metal compiler was free to make
 * different fused-multiply-add choices, and the F32 results then differed by
 * about one ULP.  That is invisible after dsv41_bf16 except when a value sits
 * within one ULP of a BF16 midpoint: at the 8k gate exactly ONE word of the
 * 20,480 HC words diverged, first at layer 7, and forty layers of amplification
 * turned it into different text.  Both callers now inline this body, so the
 * contraction choice is made once. */
static inline float dsv41_hc_expand_acc(float block, float post,
                                        float c0, float c1, float c2, float c3,
                                        float r0, float r1, float r2, float r3) {
    float acc = block * post;
    acc += c0 * r0;
    acc += c1 * r1;
    acc += c2 * r2;
    acc += c3 * r3;
    return acc;
}

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
        const float acc = dsv41_hc_expand_acc(
            block_v,
            *((device const float *) (post + dst_hc*args.nb_post0 + t*args.nb_post1)),
            *((device const float *) (comb + dst_hc*args.nb_comb0 + 0*args.nb_comb1 + t*args.nb_comb2)),
            *((device const float *) (comb + dst_hc*args.nb_comb0 + 1*args.nb_comb1 + t*args.nb_comb2)),
            *((device const float *) (comb + dst_hc*args.nb_comb0 + 2*args.nb_comb1 + t*args.nb_comb2)),
            *((device const float *) (comb + dst_hc*args.nb_comb0 + 3*args.nb_comb1 + t*args.nb_comb2)),
            r0, r1, r2, r3);

        *((device float *) (dst + d*args.nb0 + dst_hc*args.nb1 + t*args.nb2)) = dsv41_bf16(acc);
    }
}

// Phase A: exact virtual reductions, twelve HC tasks at the head of a regular
// Q8 grid. Only the last HC producer runs Sinkhorn; bulk tasks never spin.
// Coherent bit-pattern publication follows GLM router/shared, including both
// device fences. This port still requires the two-die runtime stress oracle.
template<short NR0>
static inline void dsv41_arch_coherent_reduce(device float *dst_f32,
        float sumf[NR0], int r0, int ne01, ushort tiisg, ushort sgitg,
        threadgroup char *shmem) {

    constexpr short NW = N_SIMDWIDTH;

    threadgroup float * shmem_f32[NR0];

    for (short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *) shmem + NW*row;

        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }

        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0 && r0 + row < ne01; ++row) {
        float tot = simd_sum(shmem_f32[row][tiisg]);

        if (tiisg == 0 && sgitg == 0) {
            atomic_store_explicit((device atomic_uint *)(dst_f32 + r0 + row),
                                  as_type<uint>(tot), memory_order_relaxed);
        }
    }

}
static inline void dsv41_arch_hc_mix(
        constant ds4_metal_args_hc_norm_mix &args, device const char *x,
        device const char *weight, device char *dst, threadgroup char *shmem,
        uint3 tgpig, ushort tiisg, ushort sgitg) {

    constexpr short NSG = 8;   // ds4_gpu_make_plain_mv_dispatch(16384)
    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NR0 = 2;   // plain mv nr0
    constexpr short NB  = 32;
    constexpr short NF  = 16;
    constexpr short NF4 = NF/4;
    constexpr uint  VTHREADS = 1024u;                 // rms norm threads at n == 16384
    constexpr short VSLICES  = VTHREADS/(NSG*NW);     // virtual 256-thread slices

    const uint n  = (uint)args.n;
    const uint n4 = n >> 2;

    device const float4 *x4 = (device const float4 *)x;

    threadgroup float *norm_shmem = (threadgroup float *)shmem;        // NW slots
    threadgroup float *mv_shmem   = (threadgroup float *)shmem + NW;   // NW*NR0 slots

    // Phase A: exact replica of kernel_rms_norm_f32_4's reduction tree with
    // the 1024 virtual threads folded onto this threadgroup's 8 simdgroups.
    for (short v = 0; v < VSLICES; ++v) {
        const uint vt = (uint)(sgitg + NSG*v)*NW + tiisg;
        float sumf = 0.0f;
        for (uint i00 = vt; i00 < n4; i00 += VTHREADS) {
            sumf += dot(x4[i00], x4[i00]);
        }
        sumf = simd_sum(sumf);
        if (tiisg == 0) {
            norm_shmem[sgitg + NSG*v] = sumf;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = norm_shmem[tiisg];
    total = simd_sum(total);
    const float mean  = total/(float)args.n;
    const float scale = 1.0f/sqrt(mean + args.eps);

    // Phase B: exact replica of kernel_mul_mv_f16_f32_4 (nsg=8, nr0=2) with
    // the normalized operand recomputed as x*scale instead of reloaded.
    const int nb = args.n/NB;
    const int r0 = tgpig.x*NR0;

    device const half4 * ax4[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        ax4[row] = (device const half4 *)
            (weight + (uint64_t)(r0 + row)*(uint64_t)n*sizeof(half));
    }

    float sumf_mv[NR0] = { 0.f };

    const short ix = tiisg/(NW/NF);
    const short il = tiisg%(NW/NF);
    const int ib0 = sgitg*NF + ix;

    for (int ib = ib0; ib < nb; ib += NSG*NF) {
        float4 yl4[NF4];
        FOR_UNROLL (short i = 0; i < NF4; ++i) {
            yl4[i] = x4[(ib*NB + il*NF)/4 + i]*scale;
        }

        FOR_UNROLL (short row = 0; row < NR0; row++) {
            device const half4 * xb4 = ax4[row] + (ib*NB + il*NF)/4;

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NF4; ++i) {
                sumq += dot(float4(xb4[i]), yl4[i]);
            }

            sumf_mv[row] += sumq;
        }
    }

    // n == 16384 makes the scalar tail loop of the original empty.
    device float * dst_f32 = (device float *) dst;
    dsv41_arch_coherent_reduce<NR0>(dst_f32, sumf_mv, r0, args.out_dim,
                                    tiisg, sgitg, (threadgroup char *)mv_shmem);

}
static inline void dsv41_arch_q8(constant ds4_metal_args_mul_mv &args,
        device const char *src0, device const char *src1, device char *dst,
        threadgroup char *shmem, uint3 tgpig, ushort tiisg, ushort sgitg) {
    constexpr short NR0 = N_R0_Q8_0;

    constexpr short NSG = 4;

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;

    const int nb = args.ne00/QK8_0;

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    device const float * y = (device const float *) (src1 + offset1);

    device const block_q8_0 * ax[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

        ax[row] = (device const block_q8_0 *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = { 0.f };

    const short ix = tiisg/(NW/NQ);
    const short il = tiisg%(NW/NQ);

    const int ib0 = sgitg*NQ + ix;

    float yl[NQ];

    device const float * yb = y + ib0*QK8_0 + il*NQ;

    for (int ib = ib0; ib < nb; ib += NSG*NQ) {
        for (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const int8_t * qs = ax[row][ib].qs + il*NQ;

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NQ; ++i) {
                sumq += qs[i] * yl[i];
            }

            sumf[row] += sumq*ax[row][ib].d;
        }

        yb += NSG*NQ*QK8_0;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0, true>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);

}
static inline void dsv41_arch_shared(constant ds4_metal_args_mul_mv &args,
        device const char *src0_gate, device const char *src0_up,
        device const char *src1, device char *dst_mid,
        constant float &clamp_value, constant float &alpha,
        threadgroup char *shmem, uint3 tgpig, ushort tiisg, ushort sgitg) {

    constexpr short NR0 = N_R0_Q8_0;
    constexpr short NSG = 4;
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
static inline void dsv41_arch_sinkhorn(
        constant ds4_metal_args_dsv41_hc_tail &args,
        thread const float *mix, device const float *scale,
        device const float *base, device float *split) {
        const float epsv = args.hc_eps;
        const float pre_scale = scale[0];
        const float post_scale = scale[1];
        const float comb_scale = scale[2];

        const float4 pre_z =
            *((thread const float4 *) mix) * pre_scale +
            *((device const float4 *) base);
        *((device float4 *) split) = ds4_hc_sigmoid(pre_z) + epsv;

        const float4 post_z =
            *((thread const float4 *) (mix  + 4)) * post_scale +
            *((device const float4 *) (base + 4));
        *((device float4 *) (split + 4)) = ds4_hc_twice_sigmoid(post_z);

        float4 r0 = *((thread const float4 *) (mix  +  8)) * comb_scale +
                    *((device const float4 *) (base +  8));
        float4 r1 = *((thread const float4 *) (mix  + 12)) * comb_scale +
                    *((device const float4 *) (base + 12));
        float4 r2 = *((thread const float4 *) (mix  + 16)) * comb_scale +
                    *((device const float4 *) (base + 16));
        float4 r3 = *((thread const float4 *) (mix  + 20)) * comb_scale +
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
kernel void kernel_dsv41_arch_collapse_norm(
        constant ds4_metal_args_dsv41_hc_tail &args,
        device const float *x, device const float *pre,
        device float *collapsed, device const float *norm_weight,
        device float *norm_dst, device atomic_uint *counter,
        threadgroup float *shared [[threadgroup(0)]],
        ushort tid [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort ntg [[threads_per_threadgroup]]) {

    const uint n_embd = (uint)args.n_embd;
    const uint n4 = n_embd >> 2;

    threadgroup float4 *row = (threadgroup float4 *)shared;
    threadgroup float  *sum_shmem = shared + n_embd;

    /* kernel_rms_norm_fuse_impl zeroes its 32 accumulator slots from
     * simdgroup 0 before the parallel sum. */
    if (sgitg == 0) {
        sum_shmem[tiisg] = 0.0f;
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

    if (tid == 0) atomic_store_explicit(counter, 0u, memory_order_relaxed);

}
kernel void kernel_dsv41_arch_hc_stream(
        constant ds4_metal_args_hc_norm_mix &hc_args,
        constant ds4_metal_args_mul_mv &mv_args,
        constant ds4_metal_args_dsv41_hc_tail &tail_args,
        device const char *residual, device const char *hc_weight,
        device char *mix, device const float *scale, device const float *base,
        device float *split, device atomic_uint *counter,
        device const char *weight, device const char *up,
        device const char *input, device char *output,
        constant uint &shared_stream, constant float &clamp_value,
        constant float &alpha,
        threadgroup char *shmem [[threadgroup(0)]],
        threadgroup uint *elected [[threadgroup(1)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    if (tgpig.x >= 12u) {
        // Two independent NSG=4 cohorts, each retaining NR0=2 and its tree.
        const ushort cohort = sgitg >> 2, vsg = sgitg & 3;
        uint3 vtg = uint3((tgpig.x - 12u)*2u + cohort, 0, 0);
        threadgroup char *slice = shmem + cohort*512u;
        if (shared_stream)
            dsv41_arch_shared(mv_args, weight, up, input, output,
                              clamp_value, alpha, slice, vtg, tiisg, vsg);
        else
            dsv41_arch_q8(mv_args, weight, input, output, slice, vtg, tiisg, vsg);
        return;
    }
    dsv41_arch_hc_mix(hc_args, residual, hc_weight, mix, shmem,
                      tgpig, tiisg, sgitg);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup_barrier(mem_flags::mem_device);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    if (tid == 0) {
        const uint ticket = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
        elected[0] = ticket == 11u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!elected[0]) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    if (tid == 0) {
        float4 local4[6];
        thread float *local = (thread float *)local4;
        for (uint i = 0; i < 24u; ++i)
            local[i] = as_type<float>(atomic_load_explicit(
                (device atomic_uint *)mix + i, memory_order_relaxed));
        dsv41_arch_sinkhorn(tail_args, local, scale, base, split);
    }
}

// Flat 640 + 256 row-pair tasks; no max-row padding. Both banks keep the
// ordinary Q8 NSG selected by the caller and the BF16 producer boundary.
kernel void kernel_dsv41_arch_qa_kv_flat(
        constant ds4_metal_args_mul_mv &qa_args,
        constant ds4_metal_args_mul_mv &kv_args,
        device const char *qa_weight, device const char *kv_weight,
        device const char *input, device char *qa, device char *kv,
        threadgroup char *shmem [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    if (tgpig.x < 640u)
        kernel_mul_mv_q8_0_f32_impl<2, constant ds4_metal_args_mul_mv &, true>(
            qa_args, qa_weight, input, qa, shmem, tgpig, tiisg, sgitg);
    else {
        tgpig.x -= 640u;
        kernel_mul_mv_q8_0_f32_impl<2, constant ds4_metal_args_mul_mv &, true>(
            kv_args, kv_weight, input, kv, shmem, tgpig, tiisg, sgitg);
    }
}

// Final producer fold: the group6 K/slot/SIMD tree is copied unchanged.
template<bool WIDE>
kernel void kernel_dsv41_arch_group6_expand(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src00,
        device const char * src01,
        device const char * src02,
        device const char * src03,
        device const char * src04,
        device const char * src05,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        device const float *shared_out,
        device const float *residual,
        device const float *split,
        device float *block_out,
        device float *hc_out,
        device float *next_pre,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr uint32_t expert_group_size = 64;
    const short NSG = FC_mul_mv_nsg;
    constexpr short nr0 = 2; // N_R0_Q4_K in moe.metal
    constexpr int QK_K = 256;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    float sumf[nr0] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        if (expert < 0) {
            continue;
        }
        const uint32_t expert_u = (uint32_t)expert;
        const uint32_t group_id = expert_u / expert_group_size;
        if (group_id >= 6) {
            continue;
        }
        const uint32_t expert_local = expert_u - group_id * expert_group_size;

        device const char *src0_cur = src00;
        switch (group_id) {
        case 1: src0_cur = src01; break;
        case 2: src0_cur = src02; break;
        case 3: src0_cur = src03; break;
        case 4: src0_cur = src04; break;
        case 5: src0_cur = src05; break;
        default: break;
        }

        device const block_q4_K *x =
            (device const block_q4_K *)(src0_cur + (uint64_t)expert_local * args.nb02 + first_row * args.nb01);
        device const float *y = (device const float *)(token_src1 + expert_slot * args.nb11);
        device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }

            device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half *dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    sc16[0] = sc[0] & kmask1;
                    sc16[1] = sc[2] & kmask1;
                    sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                    sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                    device const uint16_t *q2 = q1 + 32;

                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                    if (WIDE) {
                        const ushort4 v1 = *(device const ushort4 *)q1;
                        const ushort4 v2 = *(device const ushort4 *)q2;
                        FOR_UNROLL (short i = 0; i < 4; ++i) {
                            acc1[0] += yl[2 * i + 0] * (v1[i] & 0x000F);
                            acc1[1] += yl[2 * i + 1] * (v1[i] & 0x0F00);
                            acc1[2] += yl[2 * i + 8] * (v1[i] & 0x00F0);
                            acc1[3] += yl[2 * i + 9] * (v1[i] & 0xF000);
                            acc2[0] += yh[2 * i + 0] * (v2[i] & 0x000F);
                            acc2[1] += yh[2 * i + 1] * (v2[i] & 0x0F00);
                            acc2[2] += yh[2 * i + 8] * (v2[i] & 0x00F0);
                            acc2[3] += yh[2 * i + 9] * (v2[i] & 0xF000);
                        }
                    } else {
                    FOR_UNROLL (short i = 0; i < 4; ++i) {
                            acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                            acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                            acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                            acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                            acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                            acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                            acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                            acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
                        }
                    }

                    sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                          (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                          (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                          (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                                 dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                          sumy[2] * sc8[6] + sumy[3] * sc8[7]);
                }

                q1 += args.nb01 / 2;
                sc += args.nb01 / 2;
                dh += args.nb01 / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float *dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            const uint d = first_row + row;
            dst_f32[d] = sum_all;
            // Shared down already publishes BF16; retain that boundary here
            // as well (idempotent) before routed + shared, then BF16(block).
            /* Wave-4 bisect build: the reference operand sourcing exactly --
             * routed re-read from its store, shared used as published. */
            const float block = dsv41_bf16(dst_f32[d] + shared_out[d]);
            block_out[d] = block;
            const float r0 = residual[d], r1 = residual[d+5120u];
            const float r2 = residual[d+10240u], r3 = residual[d+15360u];
            for (uint h = 0; h < 4u; ++h) {
                const float acc = dsv41_hc_expand_acc(
                    block, split[4u+h],
                    split[8u+h], split[12u+h], split[16u+h], split[20u+h],
                    r0, r1, r2, r3);
                hc_out[d+h*5120u] = dsv41_bf16(acc);
            }
            if (d == 0u)
                for (uint h = 0; h < 4u; ++h) next_pre[h] = split[h];
        }
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

typedef decltype(kernel_dsv41_arch_group6_expand<false>) dsv41_arch_group6_t;
template [[host_name("kernel_dsv41_arch_group6_expand")]] kernel dsv41_arch_group6_t kernel_dsv41_arch_group6_expand<false>;
template [[host_name("kernel_dsv41_arch_group6_expand_wide")]] kernel dsv41_arch_group6_t kernel_dsv41_arch_group6_expand<true>;

/* ---------------------------------------------------------------------------
 * R4: routed-down expert-parallel geometry.
 *
 * The production group6 down matvec walks the six selected experts serially
 * inside one threadgroup, so the whole family runs as 1,280 threadgroups --
 * about one scheduling wave on this GPU, with no second wave to hide the Q4_K
 * load latency behind.  This pair splits the expert slot onto grid.z at
 * NSG=2/NR0=1, giving 2,560 x 6 = 15,360 threadgroups, and reduces the six
 * slot partials in a following dispatch.
 *
 * The per-slot body below is the production body with the slot loop peeled:
 * same contiguous K accumulation, same wide/non-wide load choice, same SIMD
 * sum.  What changes is the association -- the reference forms
 * simd_sum(sum over slots of sum over blocks), this forms the sum over slots
 * of simd_sum(sum over blocks) -- which is why R4 is Tier 2 and gates through
 * the four-manifest scorer rather than on byte identity.  Slot order is fixed
 * 0 -> 5 in the reducer so the result is deterministic, and a slot whose
 * expert is invalid publishes a zero partial, matching the `continue` that
 * contributes nothing in the reference.
 *
 * mid[slot][k] already carries the route weight (moe.metal: the pair/SwiGLU
 * producer applies it), so neither kernel multiplies by it again.
 * ------------------------------------------------------------------------ */
template<bool WIDE>
kernel void kernel_dsv41_group6_down_split(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src00,
        device const char * src01,
        device const char * src02,
        device const char * src03,
        device const char * src04,
        device const char * src05,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr uint32_t expert_group_size = 64;
    const short NSG = FC_mul_mv_nsg;
    constexpr int QK_K_ = 256;
    const int nb = args.ne00 / QK_K_;                 // 2304 / 256 = 9
    const uint row = (uint)(tgpig.x * NSG + sgitg);   // NR0 = 1
    const uint token = tgpig.y;
    const uint slot = tgpig.z;
    if (row >= (uint)args.ne0) return;

    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;
    device float *part = (device float *)dst +
        ((uint64_t)token * (uint64_t)args.ne0 + row) * 6u + slot;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    float sumf = 0.f;
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    const int32_t expert = token_ids[slot];
    const uint32_t expert_u = (uint32_t)expert;
    const uint32_t group_id = expert_u / expert_group_size;
    if (expert >= 0 && group_id < 6) {
        const uint32_t expert_local = expert_u - group_id * expert_group_size;
        device const char *src0_cur = src00;
        switch (group_id) {
        case 1: src0_cur = src01; break;
        case 2: src0_cur = src02; break;
        case 3: src0_cur = src03; break;
        case 4: src0_cur = src04; break;
        case 5: src0_cur = src05; break;
        default: break;
        }

        device const block_q4_K *x =
            (device const block_q4_K *)(src0_cur + (uint64_t)expert_local * args.nb02 + row * args.nb01);
        device const float *y = (device const float *)(token_src1 + slot * args.nb11);
        device const float *y4 = y + ix * QK_K_ + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }

            device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half *dh = &x[ib].d;

            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t *q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            if (WIDE) {
                const ushort4 v1 = *(device const ushort4 *)q1;
                const ushort4 v2 = *(device const ushort4 *)q2;
                FOR_UNROLL (short i = 0; i < 4; ++i) {
                    acc1[0] += yl[2 * i + 0] * (v1[i] & 0x000F);
                    acc1[1] += yl[2 * i + 1] * (v1[i] & 0x0F00);
                    acc1[2] += yl[2 * i + 8] * (v1[i] & 0x00F0);
                    acc1[3] += yl[2 * i + 9] * (v1[i] & 0xF000);
                    acc2[0] += yh[2 * i + 0] * (v2[i] & 0x000F);
                    acc2[1] += yh[2 * i + 1] * (v2[i] & 0x0F00);
                    acc2[2] += yh[2 * i + 8] * (v2[i] & 0x00F0);
                    acc2[3] += yh[2 * i + 9] * (v2[i] & 0xF000);
                }
            } else {
                FOR_UNROLL (short i = 0; i < 4; ++i) {
                    acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                    acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                    acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                    acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                    acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                    acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                    acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                    acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
                }
            }

            sumf += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                             (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                             (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                             (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                    dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                             sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            y4 += 4 * QK_K_;
        }
    }

    const float sum_all = simd_sum(sumf);
    if (tiisg == 0) *part = sum_all;
}

typedef decltype(kernel_dsv41_group6_down_split<false>) dsv41_group6_split_t;
template [[host_name("kernel_dsv41_group6_down_split")]] kernel dsv41_group6_split_t kernel_dsv41_group6_down_split<false>;
template [[host_name("kernel_dsv41_group6_down_split_wide")]] kernel dsv41_group6_split_t kernel_dsv41_group6_down_split<true>;

/* Fixed-order slot reduction.  Six scalar loads per row -- the partials are
 * packed slot-minor, so a float4 consumer cannot be used on them.  Writing the
 * routed row lets the existing round/expand/pre chain run unmodified, which
 * keeps every downstream BF16 boundary at its reference value and confines the
 * association change to this reduction. */
kernel void kernel_dsv41_group6_slot_sum(
        constant uint & n_rows,
        device const float * partials,
        device       float * routed,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= n_rows) return;
    device const float *p = partials + (uint64_t)gid * 6u;
    float s = 0.f;
    for (uint slot = 0; slot < 6u; ++slot) s += p[slot];
    routed[gid] = s;
}


// Two token rows share each lane's half4 weight staging. Each virtual
// row retains NSG, NR0, ib0/stride, four dot accumulation steps, then the exact
// existing two-level SIMD reduction. Inputs remain float4, never narrowed.
// Eligibility requires K divisible by 32 and output rows divisible by NR0.
template<short NR0>
static void dsv41_project_f16_rows2_impl(
        constant ds4_metal_args_mul_mv &args,
        device const char *weights, device const char *input, device char *output,
        threadgroup char *scratch, uint2 group, ushort lane, ushort sg) {
    constexpr short NW = 32, NB = 32, NF = 16, NF4 = 4;
    const short NSG = FC_mul_mv_nsg;
    const int r0 = group.x * NR0;
    const int token0 = group.y * 2;
    const int nb = args.ne00 / NB;
    const short ix = lane / (NW / NF), il = lane % (NW / NF);
    const int ib0 = sg * NF + ix;
    float sums[2][NR0] = {};
    for (int ib = ib0; ib < nb; ib += NSG * NF) {
        float4 y[2][NF4];
        for (short t = 0; t < 2; ++t) {
            if (token0 + t < args.ne1) {
                device const float4 *p = (device const float4 *)(input +
                    (ulong)(token0 + t) * args.nb11) + (ib * NB + il * NF) / 4;
                for (short i = 0; i < NF4; ++i) y[t][i] = p[i];
            }
        }
        for (short row = 0; row < NR0; ++row) {
            device const half4 *p = (device const half4 *)(weights +
                (ulong)(r0 + row) * args.nb01) + (ib * NB + il * NF) / 4;
            half4 staged[NF4];
            FOR_UNROLL (short i = 0; i < NF4; ++i) staged[i] = p[i];
            for (short t = 0; t < 2; ++t) {
                if (token0 + t < args.ne1) {
                    float sumq = 0.f;
                    FOR_UNROLL (short i = 0; i < NF4; ++i)
                        sumq += dot(float4(staged[i]), float4(y[t][i]));
                    sums[t][row] += sumq;
                }
            }
        }
    }
    // Disjoint scratch for the two reductions: the original helper has no
    // trailing barrier, so reusing token0's scratch immediately would race.
    for (short t = 0; t < 2; ++t) {
        if (token0 + t < args.ne1) {
            device float *dst = (device float *)output + (ulong)(token0 + t) * args.ne0;
            helper_mv_reduce_and_write<NR0>(dst, sums[t], r0, args.ne01,
                lane, sg, scratch + t * NW * NR0 * sizeof(float));
        }
    }
}

kernel void kernel_dsv41_project_f16_rows2(
        constant ds4_metal_args_mul_mv &args,
        device const char *weights, device const char *input, device char *output,
        threadgroup char *scratch [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (args.nr0 == 2)
        dsv41_project_f16_rows2_impl<2>(args, weights, input, output, scratch, group, lane, sg);
    else if (args.nr0 == 4)
        dsv41_project_f16_rows2_impl<4>(args, weights, input, output, scratch, group, lane, sg);
}


// Copy bits from the unchanged F16 gather, repeat four HC streams, and seed
// each token's preceding mixer. No arithmetic or new quantization boundary.
kernel void kernel_dsv41_repeat_init4(
        constant uint2 &args,
        device const uint4 *rows,
        device uint4 *residual,
        device uint4 *pre,
        uint gid [[thread_position_in_grid]]) {
    const uint vectors = args.x / 4u;
    if (gid >= args.y * vectors) return;
    const uint token = gid / vectors, d = gid % vectors;
    const uint4 value = rows[gid];
    for (uint h = 0; h < 4u; ++h)
        residual[((ulong)token * 4u + h) * vectors + d] = value;
    if (d == 0u) pre[token] = uint4(0x3f800000u, 0u, 0u, 0u);
}


// One F32 componentwise add, then exactly the standalone BF16 bit mapping.
// In particular b is not independently rounded; preserve the parent's words.
kernel void kernel_dsv41_add_bf16_rows4(
        constant uint &vectors,
        device const float4 *a, device const float4 *b, device uint4 *out,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= vectors) return;
    uint4 bits = as_type<uint4>(a[gid] + b[gid]);
    const bool4 finite = (bits & 0x7f800000u) != 0x7f800000u;
    bits += select(uint4(0), uint4(0x7fffu) + ((bits >> 16u) & 1u), finite);
    out[gid] = bits & 0xffff0000u;
}
