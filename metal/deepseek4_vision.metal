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
