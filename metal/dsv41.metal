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
static inline float4 dsv41_bf16(float4 v) {
    return float4(dsv41_bf16(v.x), dsv41_bf16(v.y), dsv41_bf16(v.z), dsv41_bf16(v.w));
}

/* Router softplus, the same formula as the DeepSeek V4 router kernels: a log1p series where exp(x) < 1/32. */
static inline float dsv41_router_softplus(float x) {
    const float ex = exp(x);
    const float em = min(ex, 0.03125f);
    const float poly = em*(1.0f - em*(0.5f - em*(1.0f/3.0f - 0.25f*em)));
    return select(select(log(1.0f + ex), poly, ex < 0.03125f), x, x > 20.0f);
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

struct ds4_metal_args_dsv41_router {
    uint n_expert, top_k, has_bias;
    float scale;
};

/* One-token router for up to 1024 experts in one dispatch: the softplus/sqrt
 * probability transform, the biased top-k through the same bitonic network as
 * kernel_argsort_f32_i32_desc (padding indices sort last), and the weights
 * normalised as the generic chain does: gather, SIMD sum over the top_k lanes,
 * clamp, divide, scale. Threadgroup memory holds nth ints and 2 * nth floats. */
kernel void kernel_dsv41_router_one(
        constant ds4_metal_args_dsv41_router &args,
        device const float *logits,
        device const float *bias,
        device float *probs,
        device int32_t *selected,
        device float *weights,
        threadgroup int32_t *shmem_i32 [[threadgroup(0)]],
        uint2 tg [[threadgroup_position_in_grid]],
        uint2 tpos [[thread_position_in_threadgroup]],
        uint2 tsize [[threads_per_threadgroup]]) {
    const uint col = tpos.x, ntg = tsize.x;
    const int width = (int)args.n_expert;
    logits += tg.y * args.n_expert;   /* one threadgroup per row */
    probs += tg.y * args.n_expert;
    selected += tg.y * args.top_k;
    weights += tg.y * args.top_k;
    threadgroup float *score = (threadgroup float *)(shmem_i32 + ntg);
    threadgroup float *prob = score + ntg;
    shmem_i32[col] = (int)col;
    if ((int)col < width) {
        const float x = logits[col];
        const float sp = dsv41_router_softplus(x);
        const float p = sqrt(sp);
        probs[col] = p;
        prob[col] = p;
        score[col] = args.has_bias ? p + bias[col] : p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k = 2; k <= (int)ntg; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            const int ixj = (int)col ^ j;
            if (ixj > (int)col) {
                const int a = shmem_i32[col], b = shmem_i32[ixj];
                const bool swap = ((int)col & k) == 0 ?
                    (a >= width || (b < width && score[a] < score[b])) :
                    (b >= width || (a < width && score[a] > score[b]));
                if (swap) { shmem_i32[col] = b; shmem_i32[ixj] = a; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (col < args.top_k) {
        const int32_t idx = shmem_i32[col];
        selected[col] = idx;
        const float w = prob[idx];
        const float total = clamp(simd_sum(w), 6.103515625e-5f, INFINITY);
        /* two stores as in the generic chain: the quotient is rounded before the scale */
        threadgroup volatile float *quotient = (threadgroup volatile float *)(prob + ntg) + col;
        *quotient = w / total;
        weights[col] = *quotient * args.scale;
    }
}

/* SIMD nominations for the unique finite top-six case; ties use the original
 * bitonic order. Keep six active lanes in the normalization reduction. */
kernel void kernel_dsv41_router_hier(
        constant ds4_metal_args_dsv41_router & args,
        device const float *logits,
        device const float *bias,
        device       float *probs,
        device     int32_t *selected,
        device       float *weights,
        threadgroup   char *shmem [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]]) {
    const uint lane = tid % 32u, sg = tid / 32u;
    threadgroup float *cscore = (threadgroup float *)shmem;
    threadgroup uint *cid = (threadgroup uint *)(shmem + 128u*4u);
    threadgroup uint *valids = (threadgroup uint *)(shmem + 256u*4u);
    float score = -INFINITY, prob = 0.f;
    bool valid = true;
    if (tid < 384u) {
        const float x = logits[tid];
        const float sp = dsv41_router_softplus(x);
        prob = sqrt(sp);
        probs[tid] = prob;
        score = args.has_bias ? prob + bias[tid] : prob;
        valid = isfinite(prob) && isfinite(score);
    }
    valid = simd_all(valid);
    if (lane == 0u) valids[sg] = valid;
    for (uint k=0; k<6u; ++k) {
        const float best = simd_max(score);
        const uint id = simd_min(score == best ? tid : 0xffffffffu);
        if (lane == 0u) {cscore[sg*6u+k] = best; cid[sg*6u+k] = id;}
        if (tid == id) score = -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (sg == 0u) {
        float v[3]; uint ix[3];
        for (uint j=0;j<3u;++j) {v[j] = cscore[lane+32u*j];ix[j] = cid[lane+32u*j];}
        bool ok = args.n_expert == 384u && args.top_k == 6u && simd_all(lane >= 16u || valids[lane] != 0u);
        float selected_prob=0.f;
        for (uint k=0;k<6u;++k) {
            const float best=simd_max(max(v[0],max(v[1],v[2])));
            uint count=0,id=0xffffffffu;
            for (uint j=0;j<3u;++j) if(v[j]==best){++count;id=min(id,ix[j]);}
            count=simd_sum(count);id=simd_min(id);
            ok = ok && count == 1u && id < 384u;
            const uint safe=min(id,383u);
            if(lane==k){selected[k]=int(safe);selected_prob=probs[safe];}
            for(uint j=0;j<3u;++j) if(ix[j]==id)v[j]=-INFINITY;
        }
        if(lane<6u){
            const float total=simd_sum(selected_prob);
            const float clamped=clamp(total,6.103515625e-5f,INFINITY);
            if(lane<6u) ((threadgroup volatile float *)cscore)[lane]=selected_prob/clamped;
        }
        if(lane==0u)valids[0]=ok;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if(valids[0]){
        if(tid<6u) weights[tid]=((threadgroup volatile float *)cscore)[tid]*args.scale;
        return;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
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
        const float sp = dsv41_router_softplus(x);
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

    const int used = (int)args.top_k;
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
    if (col < used) {
        const float wv = sprob[sidx[col]];
        const float total = simd_sum(wv);
        const float clamped = clamp(total, 6.103515625e-5f, INFINITY);
        wtmp[col] = wv / clamped;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (col < used) {
        weights[col] = wtmp[col] * args.scale;
    }
}

struct ds4_metal_args_dsv41_rope {
    uint width, heads, rows, start, inverse, stride;
    float frequencies[32];
    uint round_all;
};

kernel void kernel_dsv41_rope(
        constant ds4_metal_args_dsv41_rope &args,
        device float *x,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const float theta = float(args.start + group.y * args.stride) * args.frequencies[lane];
    const float c = precise::cos(theta);
    const float s = args.inverse ? -precise::sin(theta) : precise::sin(theta);
    const ulong head = ((ulong)group.y * args.heads + group.x) * args.width;
    if (args.round_all) {
        for (uint k = lane; k + 64u < args.width; k += 32u) x[head + k] = dsv41_bf16(x[head + k]);
    }
    const ulong i = head + args.width - 64u + 2u * lane;
    float re = x[i], im = x[i + 1u];
    if (args.round_all) { re = dsv41_bf16(re); im = dsv41_bf16(im); }
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

struct ds4_metal_args_dsv41_rope_quantize {
    uint width, start, inverse, mode;
    float frequencies[32];
};

/* One row's rope, quantization and store into a cache row: the rotation of
 * the last 64 columns as kernel_dsv41_rope applies it, then the block
 * quantization of kernel_dsv41_quantize, written to dst. */
static inline void dsv41_rope_quantize_body(
        constant ds4_metal_args_dsv41_rope_quantize &args,
        device const float *x,
        device float *dst,
        uint group,
        uint lane) {
    const uint block = args.mode == 3u ? 16u : 32u;
    const uint column = group * block + lane;
    const bool valid = lane < block && column < args.width;
    float value = valid ? x[column] : 0.0f;
    const float other = simd_shuffle_xor(value, 1u);
    const uint rotary = args.width - 64u;
    if (valid && column >= rotary) {
        const float theta = float(args.start) * args.frequencies[(column - rotary) >> 1u];
        const float c = precise::cos(theta);
        const float s = args.inverse ? -precise::sin(theta) : precise::sin(theta);
        const float re = (column & 1u) ? other : value;
        const float im = (column & 1u) ? value : other;
        /* the contraction kernel_dsv41_rope compiles to */
        value = (column & 1u) ? dsv41_bf16(fma(im, c, re * s)) : dsv41_bf16(fma(re, c, -(im * s)));
    }
    value = dsv41_bf16(value);
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
    if (valid) dst[column] = dsv41_bf16(result);
}

kernel void kernel_dsv41_rope_quantize(
        constant ds4_metal_args_dsv41_rope_quantize &args,
        device const float *x,
        device float *dst,
        uint group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    dsv41_rope_quantize_body(args, x, dst, group, lane);
}

/* The decode indexer query's rope and quantization in ONE dispatch over every head, in place: grid
 * (width / block, heads) x 32, each threadgroup one block of one head row. The arithmetic is dsv41_rope_quantize_body
 * with the coefficients read from the step's RoPE coefficient table (kernel_dsv41_rope_coeff_table: float2(cos, sin)
 * of the same theta; the host binds the half the frequencies select, compressed = entries 32..63 at byte 256). Each
 * lane reads its own word before the pair shuffle and writes it last. */
kernel void kernel_dsv41_rope_quantize_heads_tab(
        constant ds4_metal_args_dsv41_rope_quantize &args,
        device float *x,
        device const float2 *tab,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    device float *row = x + (ulong)group.y * args.width;
    const uint block = args.mode == 3u ? 16u : 32u;
    const uint column = group.x * block + lane;
    const bool valid = lane < block && column < args.width;
    float value = valid ? row[column] : 0.0f;
    const float other = simd_shuffle_xor(value, 1u);
    const uint rotary = args.width - 64u;
    if (valid && column >= rotary) {
        const float2 cs = tab[(column - rotary) >> 1u];
        const float c = cs.x;
        const float s = args.inverse ? -cs.y : cs.y;
        const float re = (column & 1u) ? other : value;
        const float im = (column & 1u) ? value : other;
        /* the contraction kernel_dsv41_rope compiles to */
        value = (column & 1u) ? dsv41_bf16(fma(im, c, re * s)) : dsv41_bf16(fma(re, c, -(im * s)));
    }
    value = dsv41_bf16(value);
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
    if (valid) row[column] = dsv41_bf16(result);
}

/* The per-step RoPE coefficient table (512 B): entries 0..31 the raw frequencies, 32..63 the compressed ones,
 * float2(cos, sin) of theta = float(start) * frequency, exactly the low kernels' expression; the consumer keeps the
 * inverse sign. One dispatch of 64 threads per decode step. */
kernel void kernel_dsv41_rope_coeff_table(
        constant ds4_metal_args_mv_rope & raw,
        constant ds4_metal_args_mv_rope & comp,
        device float2 * tab,
        uint tid [[thread_position_in_grid]]) {
    if (tid >= 64u) return;
    constant ds4_metal_args_mv_rope * r = tid < 32u ? &raw : &comp;
    const float theta = float(r->start) * r->frequencies[tid & 31u];
    float c;
    const float sn = precise::sincos(theta, c);
    tab[tid] = float2(c, sn);
}

/* The decode reduce of flash_attn.metal with V4.1's transforming writer. */
kernel void kernel_dsv41_flash_attn_vec_reduce_heads(
        constant ds4_metal_args_flash_attn_ext_vec_reduce & args,
        device  const char * htmp,
        device        char * dst,
        device const float2 * rope_tab,
        uint   tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    ds4_flash_attn_vec_reduce_row<true>(args, htmp, dst, rope_tab, tgpig, tiisg, sgitg,
                                        (short)FC_flash_attn_ext_vec_reduce_NWG,
                                        (short)FC_flash_attn_ext_vec_reduce_DV);
}

// Independent residual collapse can run before the mixer. The mixer and its
// Sinkhorn tail then ride the larger query grid, retaining both reference trees.
// Keep the original block helper whole, including its small norm recomputation:
// removing that code changed the compiled Sinkhorn result in exactness tests. The
// first collapse supplies query inputs; the duplicate work is off that dependency.
struct ds4_deferred_hc_args {
    int32_t n_embd, n_hc, sinkhorn_iters;
    float hc_eps, norm_eps;
};
kernel void kernel_dsv41_deferred_collapse(
        constant ds4_deferred_hc_args &args,
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

/* The split HC collapse: kernel_dsv41_deferred_collapse as two dispatches with the same bits. The single 1024-thread group
 * spends most of its time pulling 100 KB through one core; here the loads spread over 32 groups.
 * Part: group s is the original's simdgroup s (thread t = 32 s + lane owns elements t and t + 1024, as there) and stores
 * the collapsed values and its simd_sum partial in slots[s]. The four-row sum is written as the fma chain the original's
 * `acc += x * w` sequence compiles to under fast math (checked bit for bit against it), so the two cannot drift apart.
 * Norm: every simdgroup reloads the 32 partials, lane = slot, and repeats the original's second simd_sum over the same 32
 * values in the same lanes, then normalizes its elements from the stored collapsed values. Group 0 resets the counter as
 * the original does at its end. Host contract: n_embd 5120, n_hc 4 (as the original). */
kernel void kernel_dsv41_deferred_collapse_part(
        constant ds4_deferred_hc_args &args,
        device const float *x, device const float *pre,
        device float *collapsed, device float *slots,
        uint group [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint n_embd = (uint)args.n_embd, n4 = n_embd >> 2;
    const uint i0 = group * 32u + tiisg, i1 = i0 + 1024u;
    const bool h0 = i0 < n4, h1 = i1 < n4;
    device const float4 *x0 = (device const float4 *)(x + 0u * n_embd);
    device const float4 *x1 = (device const float4 *)(x + 1u * n_embd);
    device const float4 *x2 = (device const float4 *)(x + 2u * n_embd);
    device const float4 *x3 = (device const float4 *)(x + 3u * n_embd);
    const float w0 = pre[0], w1 = pre[1], w2 = pre[2], w3 = pre[3];
    float4 a0 = 0.0f, b0 = 0.0f, c0 = 0.0f, d0 = 0.0f, a1 = 0.0f, b1 = 0.0f, c1 = 0.0f, d1 = 0.0f;
    if (h0) { a0 = x0[i0]; b0 = x1[i0]; c0 = x2[i0]; d0 = x3[i0]; }
    if (h1) { a1 = x0[i1]; b1 = x1[i1]; c1 = x2[i1]; d1 = x3[i1]; }
    device float4 *collapsed4 = (device float4 *)collapsed;
    float sumf = 0.0f;
    if (h0) {
        const float4 v = dsv41_bf16(fma(d0, w3, fma(c0, w2, fma(b0, w1, a0 * w0))));
        collapsed4[i0] = v;
        sumf += dot(v, v);
    }
    if (h1) {
        const float4 v = dsv41_bf16(fma(d1, w3, fma(c1, w2, fma(b1, w1, a1 * w0))));
        collapsed4[i1] = v;
        sumf += dot(v, v);
    }
    sumf = simd_sum(sumf);
    if (tiisg == 0) slots[group] = sumf;
}

kernel void kernel_dsv41_deferred_collapse_norm(
        constant ds4_deferred_hc_args &args,
        device const float *collapsed, device const float *slots,
        device const float *norm_weight, device float *norm_dst, device atomic_uint *counter,
        uint gid [[thread_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint n4 = (uint)args.n_embd >> 2;
    float sumf = slots[tiisg];
    sumf = simd_sum(sumf);
    const float mean  = sumf / (float)args.n_embd;
    const float rscale = 1.0f / sqrt(mean + args.norm_eps);
    if (gid < n4) {
        const float4 r = ((device const float4 *)collapsed)[gid];
        const float4 v = (r * rscale) * ((device const float4 *)norm_weight)[gid];
        ((device float4 *)norm_dst)[gid] = float4(dsv41_bf16(v.x), dsv41_bf16(v.y), dsv41_bf16(v.z), dsv41_bf16(v.w));
    }
    if (gid == 0) atomic_store_explicit(counter, 0u, memory_order_relaxed);
}

/* The attention query projection (1280 -> 32768, bf16, RoPE on the outputs) of a block whose HC input was deferred:
 * the leading 16 threadgroups run the HC block input (mix, split and the collapse again), two more store the KV row
 * roped and quantized, and the rest the q tiles. */
kernel void kernel_dsv41_query_deferred_hc(
        constant ds4_metal_args_hc_norm_mix &hc [[buffer(0)]],
        constant ds4_metal_args_dsv4_hc_split_weighted_sum_norm &split_args [[buffer(1)]],
        device const char *residual [[buffer(2)]], device const char *hc_weight [[buffer(3)]],
        device char *mix [[buffer(4)]], device const float *scale [[buffer(5)]],
        device const float *base [[buffer(6)]], device char *split [[buffer(7)]],
        device atomic_uint *completion [[buffer(8)]],
        constant ds4_metal_args_mul_mv &mv [[buffer(9)]],
        device const char *weight [[buffer(10)]], device const char *x [[buffer(11)]],
        device float *q [[buffer(12)]], constant ds4_metal_args_mv_rope &rope [[buffer(13)]],
        device char *collapsed [[buffer(14)]], device const char *norm_weight [[buffer(15)]],
        device char *norm [[buffer(16)]], device const float *pre [[buffer(17)]],
        constant ds4_metal_args_dsv41_rope_quantize &kv_args [[buffer(18)]],
        device const float *kv [[buffer(19)]], device float *cache [[buffer(20)]],
        constant uint &store_kv [[buffer(21)]],
        threadgroup char *shmem [[threadgroup(0)]], uint3 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]], ushort simd [[simdgroup_index_in_threadgroup]]) {
    if (group.x < 16u) {
        dsv41_hc_block_input_body<2>(hc, split_args, residual, hc_weight, mix,
            scale, base, split, collapsed, norm_weight, norm, pre, completion, shmem,
            group, lane, simd);
        return;
    }
    if (store_kv && group.x < 18u) {
        dsv41_rope_quantize_body(kv_args, kv, cache, (group.x - 16u) * 8u + simd, lane);
        return;
    }
    const uint tile = group.x - 16u - (store_kv ? 2u : 0u);
    const float value = ds4_q8_short_value<40, false>(mv, weight, x, uint3(tile*2u,0,0), lane, simd);
    threadgroup float *pairs = (threadgroup float *)shmem;
    if (lane == 0) pairs[simd] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0 && (simd & 1u) == 0) {
        const uint row = tile * 8u + simd, col = row % rope.width;
        const float re = pairs[simd], im = pairs[simd + 1u];
        if (col >= rope.width - 64u) {
            const float theta = float(rope.start) * rope.frequencies[(col - (rope.width - 64u)) >> 1u];
            const float c = precise::cos(theta);
            const float sn = precise::sin(theta);
            q[row] = ds4_bf16_round(fma(re, c, -(im * sn)));
            q[row + 1u] = ds4_bf16_round(fma(im, c, re * sn));
        } else { q[row] = re; q[row + 1u] = im; }
    }
}

struct ds4_metal_args_dsv41_norm_pair {
    uint n0, n1, ntg0, ntg1;
    float eps;
};

/* Two weighted RMS norms in one dispatch, each row on its own thread count
 * (ntg0 threads then ntg1), so every reduction matches the single-row kernel. */
kernel void kernel_dsv41_norm_pair(
        constant ds4_metal_args_dsv41_norm_pair &args,
        device const float4 *x0,
        device const float4 *w0,
        device float4 *y0,
        device const float4 *x1,
        device const float4 *w1,
        device float4 *y1,
        threadgroup float *shmem [[threadgroup(0)]],
        uint2 tg [[threadgroup_position_in_grid]],
        ushort2 tpos [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const ushort tid = tpos.x;
    const bool second = tid >= args.ntg0;
    const uint ntg = second ? args.ntg1 : args.ntg0;
    const uint t = second ? tid - args.ntg0 : tid;
    const uint sg = second ? sgitg - args.ntg0 / 32u : sgitg;
    const int n = int(second ? args.n1 : args.n0);
    const int ne00_t = n / 4;
    const uint row = tg.y;   /* grid row = activation row */
    device const float4 *x = (second ? x1 : x0) + (ulong)row * ne00_t;
    device const float4 *w = second ? w1 : w0;
    device float4 *y = (second ? y1 : y0) + (ulong)row * ne00_t;
    threadgroup float *sh = shmem + (second ? 32u : 0u);
    if (sg == 0) {
        sh[tiisg] = 0.0f;
    }
    float sumf = 0.0f;
    for (int i = int(t); i < ne00_t; i += int(ntg)) {
        sumf += dot(x[i], x[i]);
    }
    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        sh[sg] = sumf;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumf = sh[tiisg];
    sumf = simd_sum(sumf);
    const float mean  = sumf/n;
    const float scale = 1.0f/sqrt(mean + args.eps);
    for (int i = int(t); i < ne00_t; i += int(ntg)) {
        float4 v = x[i]*scale;
        v = v*w[i];
        y[i] = dsv41_bf16(v);
    }
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

/* A single product, rounded as written: fast math may not reassociate it into a neighbouring product. */
static inline float dsv41_engram_mul(float a, float b) {
#pragma METAL fp math_mode(safe)
    return a * b;
}

/* The decode Engram gate with one 256-thread group per stream instead of one 32-lane SIMD group, with
 * kernel_dsv41_engram_add's bits. Lane l of the 32-lane kernel accumulates elements l, l + 32, ..., l + 32 * 159
 * (j = 0..159) in that order. Under fast math its `h2 += h * h; k2 += k * k; dot += h * (q * w) * k` compiles
 * to fma(h, h, h2), fma(k, k, k2) and fma((h * q) * w, k, dot) (each product rounded), and its residual update
 * to fma(gate, v, h); they are written out here so the different structure cannot change them. Simdgroup s loads segment j = 20 s .. 20 s + 19 of every
 * lane's chain into registers (all loads issued up front), and the segments run in order: simdgroup s takes the
 * three accumulators from s - 1 through threadgroup memory, adds its 20 steps per lane, and hands them on.
 * Simdgroup 7 then does the same simd_sums, norms and gate; every thread updates its own 20 elements of the
 * residual from registers. Width 5120 only, 512 bytes of threadgroup memory. */
kernel void kernel_dsv41_engram_add_tg256(
        constant ds4_metal_args_dsv41_engram &args,
        device float *residual,
        device const float *kv,
        device const float *q_weight,
        device const float *k_weight,
        device const uchar *mask,
        threadgroup float *sh [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    if (args.masked && !mask[group.x]) return;
    constexpr uint W = 5120u, J = W / 256u;
    const uint e0 = sg * 32u * J + lane;
    device float *hp = residual + ((ulong)group.x * 4u + group.y) * W + e0;
    device const float *kp = kv + ((ulong)group.x * 5u + group.y) * W + e0;
    device const float *vp = kv + ((ulong)group.x * 5u + 4u) * W + e0;
    device const float *qp = q_weight + (ulong)group.y * W + e0;
    device const float *wp = k_weight + (ulong)group.y * W + e0;
    float h[J], k[J], p[J], v[J];
    for (uint j = 0; j < J; j++) {
        h[j] = hp[32u * j];
        k[j] = dsv41_bf16(kp[32u * j]);
        p[j] = dsv41_engram_mul(dsv41_engram_mul(h[j], qp[32u * j]), wp[32u * j]);
        v[j] = vp[32u * j];
    }
    float h2 = 0.0f, k2 = 0.0f, dot = 0.0f;
    for (uint s = 0; s < 8u; s++) {
        if (sg == s) {
            if (s) { h2 = sh[lane]; k2 = sh[32u + lane]; dot = sh[64u + lane]; }
            for (uint j = 0; j < J; j++) {
                h2 = fma(h[j], h[j], h2);
                k2 = fma(k[j], k[j], k2);
                dot = fma(p[j], k[j], dot);
            }
            if (s < 7u) { sh[lane] = h2; sh[32u + lane] = k2; sh[64u + lane] = dot; }
        }
        if (s < 7u) threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg == 7u) {
        h2 = simd_sum(h2);
        k2 = simd_sum(k2);
        dot = simd_sum(dot) * rsqrt(h2 / args.width + args.eps) *
              rsqrt(k2 / args.width + args.eps) * rsqrt(float(args.width));
        const float gate = 1.0f / (1.0f + exp(-copysign(sqrt(max(abs(dot), 1.0e-6f)), dot)));
        if (lane == 0u) sh[96] = gate;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float gate = sh[96];
    for (uint j = 0; j < J; j++)
        hp[32u * j] = dsv41_bf16(fma(gate, dsv41_bf16(v[j]), h[j]));
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
