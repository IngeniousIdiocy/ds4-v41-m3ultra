// DS4 Metal get-rows kernel.

struct ds4_metal_args_get_rows {
    int32_t  ne00t;
    int32_t  ne00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
};

struct ds4_metal_args_get_rows_q8_0 {
    int32_t  n_embd;
    int32_t  n_vocab;
    int32_t  n_tokens;
    uint64_t src_row_bytes;
    uint64_t dst_row_bytes;
    uint64_t token_stride;
};

struct ds4_get_rows_block_q4_0 {
    half d;
    uchar qs[16];
};

struct ds4_get_rows_block_q4_K {
    half d;
    half dmin;
    uchar scales[12];
    uchar qs[128];
};

static inline uchar2 ds4_get_rows_q4_K_scale_min(int j, int k, device const uchar *q) {
    return j < 4 ? uchar2{uchar(q[j + 0 + k] & 63), uchar(q[j + 4 + k] & 63)}
                 : uchar2{uchar((q[j + 4 + k] & 0x0f) | ((q[j - 4 + k] & 0xc0) >> 2)),
                          uchar((q[j + 4 + k] >> 4) | ((q[j - 0 + k] & 0xc0) >> 2))};
}

// Gathers embedding/table rows by integer ids. DS4 uses this for token
// embeddings and small indexed tables such as router/hash lookup outputs.
template<typename T0, typename T>
kernel void kernel_get_rows_f(
        constant ds4_metal_args_get_rows & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3               tgpig[[threadgroup_position_in_grid]],
        ushort              tiitg[[thread_index_in_threadgroup]],
        ushort3             ntg [[threads_per_threadgroup]]) {
    const int32_t iw0 = tgpig.x/args.ne10;
    const int32_t i10 = tgpig.x%args.ne10;
    const int32_t i11 = tgpig.y;
    const int32_t i12 = tgpig.z;

    const int32_t r = ((const device int32_t *) (src1 + i12*args.nb12 + i11*args.nb11 + i10*args.nb10))[0];

    const int32_t i02 = i11;
    const int32_t i03 = i12;

    auto psrc = (const device T0 *) (src0 + i03*args.nb03 + i02*args.nb02 + r*args.nb01);
    auto pdst = (      device T  *) (dst  + i12*args.nb3  + i11*args.nb2  + i10*args.nb1);

    for (int ind = iw0*ntg.x + tiitg; ind < args.ne00t;) {
        pdst[ind] = psrc[ind];

        break;
    }
}

typedef decltype(kernel_get_rows_f<float, float>) get_rows_f_t;

// Host-visible gather variants for F32, F16, and I32 tables.
template [[host_name("kernel_get_rows_f32")]] kernel get_rows_f_t kernel_get_rows_f<float, float>;
template [[host_name("kernel_get_rows_f16")]] kernel get_rows_f_t kernel_get_rows_f<half, float>;
template [[host_name("kernel_get_rows_i32")]] kernel get_rows_f_t kernel_get_rows_f<int32_t, int32_t>;
/* R3a: the same gather writing F16.  The decode attention stages its gathered
 * compressed KV rows into an F16 buffer; producing them as F16 here is the
 * identical float -> half conversion the staging copy applied, one pass
 * earlier, and lets the contiguous staging kernel take the rows verbatim. */
template [[host_name("kernel_get_rows_f32_f16")]] kernel get_rows_f_t kernel_get_rows_f<float, half>;

/* Decode pass 2, C1: the same F32 -> F16 sparse row gather with wide loads.
 * kernel_get_rows_f copies ONE element per thread (one 4 B load, one 2 B
 * store, then `break`), which is why the 512-row/512-column decode KV gather
 * ran at ~140 GB/s.  This twin gives every thread `EPT` contiguous elements
 * through packed_float4 loads and packed_half4 stores.  The per-element
 * conversion is the identical `half(float)`, the row map is the identical
 * ids[] lookup, and each destination element is written by exactly one thread,
 * so the result is bit-identical by construction.
 *
 * Geometry contract: ne00 elements per row, a multiple of 4*EPT; one thread
 * per (row, chunk of 4*EPT elements); grid laid out row-major so a threadgroup
 * stays inside one or two source rows. */
struct ds4_metal_args_gather_kv_wide {
    uint   n_elem;      /* elements per row (512) */
    uint   n_rows;      /* selected rows */
    uint   vec_per_row; /* n_elem / (4*EPT) */
    uint   pad;
    uint64_t src_row_bytes;
    uint64_t dst_row_bytes;
};

template<uint EPT>
kernel void kernel_dsv41_gather_kv_f32_f16_w(
        constant ds4_metal_args_gather_kv_wide & args,
        device const char   * src0,
        device const char   * ids,
        device       char   * dst,
        uint gid [[thread_position_in_grid]]) {
    const uint total = args.n_rows * args.vec_per_row;
    if (gid >= total) return;
    const uint row = gid / args.vec_per_row;
    const uint chunk = gid - row * args.vec_per_row;
    const int  r = ((device const int *)ids)[row];
    device const packed_float4 *psrc =
        (device const packed_float4 *)(src0 + (uint64_t)r * args.src_row_bytes);
    device packed_half4 *pdst =
        (device packed_half4 *)(dst + (uint64_t)row * args.dst_row_bytes);
    const uint base = chunk * EPT;
#pragma clang loop unroll(full)
    for (uint k = 0; k < EPT; ++k) {
        const float4 v = float4(psrc[base + k]);
        pdst[base + k] = packed_half4(half4(v));
    }
}

typedef decltype(kernel_dsv41_gather_kv_f32_f16_w<1>) gather_kv_wide_t;
template [[host_name("kernel_dsv41_gather_kv_f32_f16_w1")]] kernel gather_kv_wide_t kernel_dsv41_gather_kv_f32_f16_w<1>;
template [[host_name("kernel_dsv41_gather_kv_f32_f16_w2")]] kernel gather_kv_wide_t kernel_dsv41_gather_kv_f32_f16_w<2>;
template [[host_name("kernel_dsv41_gather_kv_f32_f16_w4")]] kernel gather_kv_wide_t kernel_dsv41_gather_kv_f32_f16_w<4>;

kernel void kernel_get_rows_q8_0_f32(
        constant ds4_metal_args_get_rows_q8_0 & args,
        device const char    * src0,
        device const char    * src1,
        device       char    * dst,
        uint3                  tgpig[[threadgroup_position_in_grid]],
        ushort                 tiitg[[thread_index_in_threadgroup]],
        ushort3                ntg [[threads_per_threadgroup]]) {
    const int32_t block = (int32_t)tgpig.x;
    const int32_t tok_i = (int32_t)tgpig.y;
    if (tok_i >= args.n_tokens) return;

    const int32_t token =
        ((const device int32_t *)(src1 + (uint64_t)tok_i*args.token_stride))[0];
    if (token < 0 || token >= args.n_vocab) return;

    const device block_q8_0 *row =
        (const device block_q8_0 *)(src0 + (uint64_t)token * args.src_row_bytes);
    const device block_q8_0 *qb = row + block;
    device float *out =
        (device float *)(dst + (uint64_t)tok_i * args.dst_row_bytes);

    const int32_t i0 = block * QK8_0;
    const float d = (float)qb->d;
    for (int32_t i = (int32_t)tiitg; i < QK8_0; i += (int32_t)ntg.x) {
        const int32_t idx = i0 + i;
        if (idx < args.n_embd) {
            out[idx] = d * (float)qb->qs[i];
        }
    }
}

kernel void kernel_get_rows_q4_0_f32(
        constant ds4_metal_args_get_rows_q8_0 & args,
        device const char    * src0,
        device const char    * src1,
        device       char    * dst,
        uint3                  tgpig[[threadgroup_position_in_grid]],
        ushort                 tiitg[[thread_index_in_threadgroup]],
        ushort3                ntg [[threads_per_threadgroup]]) {
    const int32_t block = (int32_t)tgpig.x;
    const int32_t tok_i = (int32_t)tgpig.y;
    if (tok_i >= args.n_tokens) return;

    const int32_t token =
        ((const device int32_t *)(src1 + (uint64_t)tok_i * args.token_stride))[0];
    if (token < 0 || token >= args.n_vocab) return;

    const device ds4_get_rows_block_q4_0 *row =
        (const device ds4_get_rows_block_q4_0 *)(src0 + (uint64_t)token * args.src_row_bytes);
    const device ds4_get_rows_block_q4_0 *qb = row + block;
    device float *out =
        (device float *)(dst + (uint64_t)tok_i * args.dst_row_bytes);

    const int32_t i0 = block * 32;
    const float d = (float)qb->d;
    for (int32_t i = (int32_t)tiitg; i < 32; i += (int32_t)ntg.x) {
        const int32_t idx = i0 + i;
        if (idx < args.n_embd) {
            /* ggml Q4_0: elems 0..15 = low nibbles of qs[0..15], 16..31 = high. */
            const uint8_t packed = qb->qs[(uint)(i & 15)];
            const uint8_t q = (i < 16) ? (packed & 0x0f) : (packed >> 4);
            out[idx] = d * ((float)q - 8.0f);
        }
    }
}

kernel void kernel_get_rows_q4_K_f32(
        constant ds4_metal_args_get_rows_q8_0 & args,
        device const char    * src0,
        device const char    * src1,
        device       char    * dst,
        uint3                  tgpig[[threadgroup_position_in_grid]],
        ushort                 tiitg[[thread_index_in_threadgroup]],
        ushort3                ntg [[threads_per_threadgroup]]) {
    const int32_t block = (int32_t)tgpig.x;
    const int32_t tok_i = (int32_t)tgpig.y;
    if (tok_i >= args.n_tokens) return;

    const int32_t token =
        ((const device int32_t *)(src1 + (uint64_t)tok_i * args.token_stride))[0];
    if (token < 0 || token >= args.n_vocab) return;

    const device ds4_get_rows_block_q4_K *row =
        (const device ds4_get_rows_block_q4_K *)(src0 + (uint64_t)token * args.src_row_bytes);
    const device ds4_get_rows_block_q4_K *qb = row + block;
    device float *out =
        (device float *)(dst + (uint64_t)tok_i * args.dst_row_bytes);

    const int32_t i0 = block * 256;
    for (int32_t i = (int32_t)tiitg; i < 256; i += (int32_t)ntg.x) {
        const int32_t idx = i0 + i;
        if (idx < args.n_embd) {
            const uint group = (uint)i >> 5u;
            const uint l = (uint)i & 31u;
            const uchar2 sm = ds4_get_rows_q4_K_scale_min((int)group, 0, qb->scales);
            const uint byte_off = (group >> 1u) * 32u + l;
            const uint shift = (group & 1u) * 4u;
            const uint q = ((uint)qb->qs[byte_off] >> shift) & 0x0fu;
            out[idx] = (float)qb->d * (float)sm.x * (float)q -
                       (float)qb->dmin * (float)sm.y;
        }
    }
}
