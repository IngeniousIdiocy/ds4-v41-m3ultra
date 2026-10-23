// DeepSeek V4 Flash Vision-Exp operations not covered by shared BF16 matmuls.

struct deepseek4_vision_rows_args {
    uint width;
    uint rows;
};

struct deepseek4_vision_qkv_args {
    uint rows;
    uint grid_width;
};

struct deepseek4_vision_align_args {
    uint grid_height;
    uint grid_width;
    uint output_rows;
};

static inline float deepseek4_vision_round_bf16(float value) {
    uint bits = as_type<uint>(value);
    if ((bits & 0x7f800000u) == 0x7f800000u) return value;
    bits += 0x00007fffu + ((bits >> 16u) & 1u);
    return as_type<float>(bits & 0xffff0000u);
}

static inline float deepseek4_vision_erf(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float a = abs(x);
    const float t = 1.0f / (1.0f + 0.3275911f * a);
    const float p = (((((1.061405429f * t - 1.453152027f) * t) +
                       1.421413741f) * t - 0.284496736f) * t +
                       0.254829592f) * t;
    return sign * (1.0f - p * exp(-a * a));
}

kernel void kernel_deepseek4_vision_qkv_rope(
        constant deepseek4_vision_qkv_args &args,
        device const float                 *qkv,
        device const ushort                *bias,
        device float                       *q,
        device float                       *k,
        device float                       *v,
        uint2 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint row = group.x;
    const uint head = group.y;
    if (row >= args.rows || head >= 16u) return;
    const ulong qkv_base = (ulong)row * 3072u + (ulong)head * 64u;
    const ulong out_base = (ulong)row * 1024u + (ulong)head * 64u;
    const float q0 = deepseek4_vision_round_bf16(
        qkv[qkv_base + lane] +
        glm53_bf16_to_f32(bias[(ulong)head * 64u + lane]));
    const float q1 = deepseek4_vision_round_bf16(
        qkv[qkv_base + lane + 32u] +
        glm53_bf16_to_f32(bias[(ulong)head * 64u + lane + 32u]));
    const float k0 = deepseek4_vision_round_bf16(
        qkv[qkv_base + 1024u + lane] +
        glm53_bf16_to_f32(bias[1024u + (ulong)head * 64u + lane]));
    const float k1 = deepseek4_vision_round_bf16(
        qkv[qkv_base + 1024u + lane + 32u] +
        glm53_bf16_to_f32(bias[1024u + (ulong)head * 64u + lane + 32u]));
    const uint y = row / args.grid_width;
    const uint x = row - y * args.grid_width;
    const uint pos = lane < 16u ? y : x;
    const uint freq_index = lane & 15u;
    const float inv_freq = powr(10000.0f, -(float)freq_index / 16.0f);
    const float angle = (float)pos * inv_freq;
    const float cs = cos(angle);
    const float sn = sin(angle);
    q[out_base + lane] = deepseek4_vision_round_bf16(q0 * cs - q1 * sn);
    q[out_base + lane + 32u] =
        deepseek4_vision_round_bf16(q1 * cs + q0 * sn);
    k[out_base + lane] = deepseek4_vision_round_bf16(k0 * cs - k1 * sn);
    k[out_base + lane + 32u] =
        deepseek4_vision_round_bf16(k1 * cs + k0 * sn);
    v[out_base + lane] = deepseek4_vision_round_bf16(
        qkv[qkv_base + 2048u + lane] +
        glm53_bf16_to_f32(bias[2048u + (ulong)head * 64u + lane]));
    v[out_base + lane + 32u] = deepseek4_vision_round_bf16(
        qkv[qkv_base + 2048u + lane + 32u] +
        glm53_bf16_to_f32(
            bias[2048u + (ulong)head * 64u + lane + 32u]));
}

kernel void kernel_deepseek4_vision_round_bf16(
        constant deepseek4_vision_rows_args &args,
        device float                       *x,
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.width || gid.y >= args.rows) return;
    const ulong off = (ulong)gid.y * args.width + gid.x;
    x[off] = deepseek4_vision_round_bf16(x[off]);
}

kernel void kernel_deepseek4_vision_add_residual(
        constant deepseek4_vision_rows_args &args,
        device float                       *x,
        device const float                 *residual,
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.width || gid.y >= args.rows) return;
    const ulong off = (ulong)gid.y * args.width + gid.x;
    x[off] = deepseek4_vision_round_bf16(x[off] + residual[off]);
}

kernel void kernel_deepseek41_vision_bias_residual(
        constant deepseek4_vision_rows_args &args,
        device float                       *x,
        device const ushort                *bias,
        device const float                 *residual,
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.width || gid.y >= args.rows) return;
    const ulong off = (ulong)gid.y * args.width + gid.x;
    const float projected = deepseek4_vision_round_bf16(
        x[off] + glm53_bf16_to_f32(bias[gid.x]));
    x[off] = deepseek4_vision_round_bf16(projected + residual[off]);
}

static inline void deepseek4_vision_swiglu_split(
        constant deepseek4_vision_rows_args &args,
        device const float                 *gate_up,
        device float                       *out,
        uint2 gid,
        bool round_silu) {
    if (gid.x >= args.width || gid.y >= args.rows) return;
    const ulong source = (ulong)gid.y * args.width * 2u + gid.x;
    const float gate = gate_up[source];
    const float up = gate_up[source + args.width];
    float activated = gate / (1.0f + exp(-gate));
    if (round_silu) activated = deepseek4_vision_round_bf16(activated);
    out[(ulong)gid.y * args.width + gid.x] = deepseek4_vision_round_bf16(activated * up);
}

kernel void kernel_deepseek4_vision_swiglu_split(
        constant deepseek4_vision_rows_args &args,
        device const float                 *gate_up,
        device float                       *out,
        uint2 gid [[thread_position_in_grid]]) {
    deepseek4_vision_swiglu_split(args, gate_up, out, gid, false);
}

kernel void kernel_deepseek41_vision_swiglu_split(
        constant deepseek4_vision_rows_args &args,
        device const float                 *gate_up,
        device float                       *out,
        uint2 gid [[thread_position_in_grid]]) {
    deepseek4_vision_swiglu_split(args, gate_up, out, gid, true);
}

/* Match F.unfold(x, 3, stride=3): channels are outermost, followed by the
 * row-major 3x3 neighborhood. Values outside the ViT grid are zero. */
kernel void kernel_deepseek4_vision_aligner_reorder(
        constant deepseek4_vision_align_args &args,
        device const float                  *x,
        device float                        *out,
        uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x;
    const uint row = gid.y;
    if (d >= 9216u || row >= args.output_rows) return;
    const uint output_width = (args.grid_width + 2u) / 3u;
    const uint block_y = row / output_width;
    const uint block_x = row - block_y * output_width;
    const uint channel = d / 9u;
    const uint within = d - channel * 9u;
    const uint source_y = block_y * 3u + within / 3u;
    const uint source_x = block_x * 3u + within % 3u;
    float value = 0.0f;
    if (source_y < args.grid_height && source_x < args.grid_width) {
        const ulong source_row = (ulong)source_y * args.grid_width + source_x;
        value = x[source_row * 1024u + channel];
    }
    out[(ulong)row * 9216u + d] = value;
}

kernel void kernel_deepseek4_vision_gelu_bias(
        constant deepseek4_vision_rows_args &args,
        device const float                 *x,
        device const ushort                *bias,
        device float                       *out,
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.width || gid.y >= args.rows) return;
    const ulong off = (ulong)gid.y * args.width + gid.x;
    const float value = deepseek4_vision_round_bf16(
        x[off] + glm53_bf16_to_f32(bias[gid.x]));
    out[off] = deepseek4_vision_round_bf16(
        0.5f * value *
        (1.0f + deepseek4_vision_erf(value * 0.7071067811865475f)));
}

/* VISION_ROUND_FOLD (DS4_METAL_DISABLE_V41_VISION_ROUND_FOLD=1 restores the passes): the bf16 rounding that ran as its own pass is applied in
 * the producer's store or the consumer's load. Each kernel below repeats the
 * arithmetic of the kernel it replaces, so the rounded values are the same. */

kernel void kernel_deepseek41_vision_rms_round(
        constant glm53_vision_rows_args &args,
        device const float              *x,
        device const ushort             *weight,
        device float                    *out,
        threadgroup float               *partial,
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    if (row >= args.rows) return;
    device const float *xr = x + (ulong)row * args.width;
    device float *yr = out + (ulong)row * args.width;
    float sum = 0.0f;
    for (uint d = tid; d < args.width; d += 256u) sum = fma(xr[d], xr[d], sum);
    sum = simd_sum(sum);
    if (lane == 0u) partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float v = lane < nsg ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0u) partial[0] = rsqrt(v / (float)args.width + args.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = partial[0];
    for (uint d = tid; d < args.width; d += 256u) {
        yr[d] = deepseek4_vision_round_bf16(xr[d] * inv * glm53_bf16_to_f32(weight[d]));
    }
}

kernel void kernel_deepseek41_vision_add_bias_round(
        constant glm53_vision_rows_args &args,
        device float                    *x,
        device const ushort             *bias,
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.width || gid.y >= args.rows) return;
    const ulong off = (ulong)gid.y * args.width + gid.x;
    float value = x[off];
    value += glm53_bf16_to_f32(bias[gid.x]);
    x[off] = deepseek4_vision_round_bf16(value);
}

/* The gate/up halves are rounded on load (the w1 output's rounding pass). */
kernel void kernel_deepseek41_vision_swiglu_split_rl(
        constant deepseek4_vision_rows_args &args,
        device const float                 *gate_up,
        device float                       *out,
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.width || gid.y >= args.rows) return;
    const ulong source = (ulong)gid.y * args.width * 2u + gid.x;
    const float gate = deepseek4_vision_round_bf16(gate_up[source]);
    const float up = deepseek4_vision_round_bf16(gate_up[source + args.width]);
    float activated = gate / (1.0f + exp(-gate));
    activated = deepseek4_vision_round_bf16(activated);
    out[(ulong)gid.y * args.width + gid.x] = deepseek4_vision_round_bf16(activated * up);
}

/* The w2 output is rounded on load (its rounding pass), then the residual add. */
kernel void kernel_deepseek41_vision_add_residual_rl(
        constant deepseek4_vision_rows_args &args,
        device float                       *x,
        device const float                 *residual,
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.width || gid.y >= args.rows) return;
    const ulong off = (ulong)gid.y * args.width + gid.x;
    x[off] = deepseek4_vision_round_bf16(deepseek4_vision_round_bf16(x[off]) + residual[off]);
}

/* kernel_glm53_vision_attention with the output rounding in its store. */
kernel void kernel_deepseek41_vision_attention_round(
        constant glm53_vision_attention_args &args,
        device const float                   *q,
        device const float                   *k,
        device const float                   *v,
        device float                         *out,
        uint2 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint row = group.x;
    const uint head = group.y;
    if (row >= args.rows || head >= 16u) return;
    const ulong base = (ulong)row * 1024u + (ulong)head * 64u;
    const float q0 = q[base + lane];
    const float q1 = q[base + lane + 32u];
    float acc0 = 0.0f, acc1 = 0.0f;
    float max_score = -INFINITY;
    float denom = 0.0f;
    for (uint key_row = 0; key_row < args.rows; key_row++) {
        const ulong kb = (ulong)key_row * 1024u + (ulong)head * 64u;
        float score = simd_sum(q0 * k[kb + lane] + q1 * k[kb + lane + 32u]);
        score *= args.scale;
        const float next_max = max(max_score, score);
        const float old_scale = max_score == -INFINITY ? 0.0f : exp(max_score - next_max);
        const float new_scale = exp(score - next_max);
        denom = denom * old_scale + new_scale;
        acc0 = acc0 * old_scale + new_scale * v[kb + lane];
        acc1 = acc1 * old_scale + new_scale * v[kb + lane + 32u];
        max_score = next_max;
    }
    out[base + lane] = deepseek4_vision_round_bf16(acc0 / denom);
    out[base + lane + 32u] = deepseek4_vision_round_bf16(acc1 / denom);
}

/* VISION_ATTN_EXACT: the scalar kernel's arithmetic, 32 keys at a time.
 * Lane j forms key j's 32 lane products itself and adds them with simd_sum's
 * adjacent-pairs tree (strides 1, 2, 4, 8, 16), so the score has simd_sum's
 * bits without five shuffles per key. The running maximum is a prefix maximum
 * (order-free), each key's two exponentials are computed by its own lane from
 * the same operands, and the rescaling chain then runs in key order exactly as
 * before, reading the broadcast factors. q and k hold bf16-rounded values,
 * so each lane product is exact in float and contraction cannot change it. */
static inline float deepseek41_vision_tree_dot(
        threadgroup const float *qs,
        threadgroup const float *kr) {
#pragma clang fp reassociate(off)
    float level[5];
    float x = 0.0f;
#pragma unroll
    for (uint l = 0; l < 32u; l++) {
        x = qs[l] * kr[l] + qs[l + 32u] * kr[l + 32u];
#pragma unroll
        for (uint b = 0; b < 5u; b++) {
            if ((l >> b) & 1u) {
                x = level[b] + x;
            } else {
                level[b] = x;
                break;
            }
        }
    }
    return x;
}

/* Eight SIMD groups (eight query rows of one head) share each 32-key K/V tile in
 * threadgroup memory; K rows are padded to 65 floats so lane j's row reads do
 * not collide in a bank. */
template <bool ROUND>
kernel void kernel_deepseek41_vision_attention_exact(
        constant glm53_vision_attention_args &args,
        device const float                   *q,
        device const float                   *k,
        device const float                   *v,
        device float                         *out,
        threadgroup float                    *tg [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint head = group.y;
    const uint rows = args.rows;
    const uint row = group.x * 8u + sg;
    const bool valid = row < rows;
    threadgroup float *Ks = tg;              /* [32][65] */
    threadgroup float *Vs = tg + 32u * 65u;  /* [32][64] */
    threadgroup float *qs = Vs + 2048u + sg * 64u;
    const ulong base = (ulong)(valid ? row : 0u) * 1024u + (ulong)head * 64u;
    qs[lane] = q[base + lane];
    qs[lane + 32u] = q[base + lane + 32u];
    float acc0 = 0.0f, acc1 = 0.0f;
    float max_score = -INFINITY;
    float denom = 0.0f;
    for (uint key0 = 0; key0 < rows; key0 += 32u) {
        const uint n = min(32u, rows - key0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 2048u; i += 256u) {
            const uint kr = i >> 6, d = i & 63u, key = min(key0 + kr, rows - 1u);
            const ulong off = (ulong)key * 1024u + (ulong)head * 64u + d;
            Ks[kr * 65u + d] = k[off];
            Vs[i] = v[off];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float score = deepseek41_vision_tree_dot(qs, Ks + lane * 65u);
        score *= args.scale;
        if (lane >= n) score = -INFINITY;
        float prefix = score;
#pragma unroll
        for (ushort d = 1; d < 32; d <<= 1) {
            const float t = simd_shuffle_up(prefix, d);
            if (lane >= d) prefix = max(prefix, t);
        }
        const float next_max = max(max_score, prefix);
        float prev_max = simd_shuffle_up(next_max, 1);
        if (lane == 0) prev_max = max_score;
        const float old_lane = prev_max == -INFINITY ? 0.0f : exp(prev_max - next_max);
        const float new_lane = exp(score - next_max);
        for (uint j = 0; j < n; j++) {
            const float old_scale = simd_broadcast(old_lane, (ushort)j);
            const float new_scale = simd_broadcast(new_lane, (ushort)j);
            denom = denom * old_scale + new_scale;
            acc0 = acc0 * old_scale + new_scale * Vs[j * 64u + lane];
            acc1 = acc1 * old_scale + new_scale * Vs[j * 64u + lane + 32u];
        }
        max_score = simd_broadcast(next_max, (ushort)(n - 1u));
    }
    if (!valid) return;
    const float o0 = acc0 / denom;
    const float o1 = acc1 / denom;
    out[base + lane] = ROUND ? deepseek4_vision_round_bf16(o0) : o0;
    out[base + lane + 32u] = ROUND ? deepseek4_vision_round_bf16(o1) : o1;
}

typedef decltype(kernel_deepseek41_vision_attention_exact<false>) deepseek41_vision_attention_exact_t;
template [[host_name("kernel_deepseek41_vision_attention_exact")]] kernel deepseek41_vision_attention_exact_t kernel_deepseek41_vision_attention_exact<false>;
template [[host_name("kernel_deepseek41_vision_attention_exact_round")]] kernel deepseek41_vision_attention_exact_t kernel_deepseek41_vision_attention_exact<true>;

/* VISION_ATTN_FLASH (default on; accurate, not bit-exact): tiled attention on simdgroup matrices.
 * A threadgroup of eight SIMD groups takes 64 query rows of one head, 8 per SIMD group, and walks the keys in tiles
 * of 64 staged in threadgroup memory as half (q and k are bf16-rounded, which half holds exactly in its normal
 * range; every product and sum stays in float). The softmax never leaves the SIMD group: in an 8x8 simdgroup
 * matrix lane l holds row (l/4 & 4) + (l/2 % 4) at columns fn, fn+1 with fn = (l/4 & 2)*2 + (l%2)*2, so a row's
 * scores sit in the lanes that differ in bits 0 and 3 and reduce with two shuffles, and the running maximum,
 * the output rescale and the probabilities stay in registers. Q is pre-scaled by 1/8 (exact). */
template <bool ROUND>
kernel void kernel_deepseek41_vision_attention_flash(
        constant glm53_vision_attention_args &args,
        device const float                   *q,
        device const float                   *k,
        device const float                   *v,
        device float                         *out,
        threadgroup half                     *tg [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    constexpr uint KT = 64;                     /* keys per staged tile */
    constexpr uint KP = 72;                     /* padded tile row (halves) */
    const uint head = group.y;
    const uint rows = args.rows;
    const uint row0 = group.x * 64u + sg * 8u;
    threadgroup half *Ks = tg;                  /* [KT][KP] */
    threadgroup half *Vs = tg + KT * KP;        /* [KT][KP] */
    const ushort qid = lane / 4u;
    const ushort fm = (qid & 4u) + ((lane / 2u) % 4u);
    const ushort fn = (qid & 2u) * 2u + (lane % 2u) * 2u;
    const uint qrow = row0 + fm;
    const bool qvalid = qrow < rows;
    device const float *qp = q + (ulong)(qvalid ? qrow : 0u) * 1024u + (ulong)head * 64u;
    simdgroup_float8x8 Qm[8];
#pragma unroll
    for (uint kk = 0; kk < 8u; kk++) {
        thread auto &e = Qm[kk].thread_elements();
        e[0] = qvalid ? qp[kk * 8u + fn] * args.scale : 0.0f;
        e[1] = qvalid ? qp[kk * 8u + fn + 1u] * args.scale : 0.0f;
    }
    simdgroup_float8x8 O[8];
#pragma unroll
    for (uint c = 0; c < 8u; c++) O[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float m = -INFINITY, l = 0.0f;
    for (uint key0 = 0; key0 < rows; key0 += KT) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < KT * 16u; i += 256u) {
            const uint r = i >> 4, c4 = (i & 15u) * 4u, key = key0 + r;
            const ulong off = (ulong)key * 1024u + (ulong)head * 64u + c4;
            const float4 kv = key < rows ? *(device const float4 *)(k + off) : float4(0.0f);
            const float4 vv = key < rows ? *(device const float4 *)(v + off) : float4(0.0f);
            *(threadgroup half4 *)(Ks + r * KP + c4) = half4(kv);
            *(threadgroup half4 *)(Vs + r * KP + c4) = half4(vv);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 S[KT / 8];
#pragma unroll
        for (uint n = 0; n < KT / 8u; n++) {
            S[n] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
#pragma unroll
            for (uint kk = 0; kk < 8u; kk++) {
                simdgroup_half8x8 Kt;
                simdgroup_load(Kt, Ks + n * 8u * KP + kk * 8u, KP, ulong2(0, 0), true);
                simdgroup_float8x8 Kf;
                Kf.thread_elements()[0] = (float)Kt.thread_elements()[0];
                Kf.thread_elements()[1] = (float)Kt.thread_elements()[1];
                simdgroup_multiply_accumulate(S[n], Qm[kk], Kf, S[n]);
            }
        }
        float mx = -INFINITY;
#pragma unroll
        for (uint n = 0; n < KT / 8u; n++) {
            thread auto &e = S[n].thread_elements();
            const uint key = key0 + n * 8u + fn;
            if (key >= rows) e[0] = -INFINITY;
            if (key + 1u >= rows) e[1] = -INFINITY;
            mx = max(mx, max(e[0], e[1]));
        }
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 8));
        const float m_new = max(m, mx);
        const float alpha = exp(m - m_new);
        float sum = 0.0f;
#pragma unroll
        for (uint n = 0; n < KT / 8u; n++) {
            thread auto &e = S[n].thread_elements();
            e[0] = exp(e[0] - m_new);
            e[1] = exp(e[1] - m_new);
            sum += e[0] + e[1];
        }
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 8);
        l = l * alpha + sum;
        m = m_new;
#pragma unroll
        for (uint c = 0; c < 8u; c++) {
            thread auto &e = O[c].thread_elements();
            e[0] *= alpha;
            e[1] *= alpha;
        }
#pragma unroll
        for (uint n = 0; n < KT / 8u; n++) {
#pragma unroll
            for (uint c = 0; c < 8u; c++) {
                simdgroup_half8x8 Vm;
                simdgroup_load(Vm, Vs + n * 8u * KP + c * 8u, KP);
                simdgroup_float8x8 Vf;
                Vf.thread_elements()[0] = (float)Vm.thread_elements()[0];
                Vf.thread_elements()[1] = (float)Vm.thread_elements()[1];
                simdgroup_multiply_accumulate(O[c], S[n], Vf, O[c]);
            }
        }
    }
    if (!qvalid) return;
    device float *op = out + (ulong)qrow * 1024u + (ulong)head * 64u;
    const float inv = 1.0f / l;
#pragma unroll
    for (uint c = 0; c < 8u; c++) {
        thread auto &e = O[c].thread_elements();
        const float o0 = e[0] * inv, o1 = e[1] * inv;
        op[c * 8u + fn] = ROUND ? deepseek4_vision_round_bf16(o0) : o0;
        op[c * 8u + fn + 1u] = ROUND ? deepseek4_vision_round_bf16(o1) : o1;
    }
}

typedef decltype(kernel_deepseek41_vision_attention_flash<false>) deepseek41_vision_attention_flash_t;
template [[host_name("kernel_deepseek41_vision_attention_flash")]] kernel deepseek41_vision_attention_flash_t kernel_deepseek41_vision_attention_flash<false>;
template [[host_name("kernel_deepseek41_vision_attention_flash_round")]] kernel deepseek41_vision_attention_flash_t kernel_deepseek41_vision_attention_flash<true>;
