/* The GPU top-p selector of the V4.1 decode (its own library: ds4_gpu_selector_init compiles this file once at load
 * with the default compile options; the engine library is unchanged). After the output head, three dispatches over
 * the logits in the final decode command buffer: K1 (505 threadgroups x 256 threads, one 256-logit block each:
 * finite count, maximum and earliest index per block, coarse histogram H1 of the order keys), R1 (one threadgroup:
 * the global finite / M / best, the coarse bin holding the 512th key, the superset size; clears H1), S3 (505
 * threadgroups: the block mass partials sum exp(v - M) and the compaction of every key in or above that bin into
 * 2048 slots). The CPU consumer in ds4.c validates the summary, decides top-p from it and draws the same token as
 * the production sampler, or falls back to it. Finite is a bit test (the library is fast-math). H1 must be zero
 * before the first call: the host zeroes it at allocation; K1 clears the slot counter. */
#include <metal_stdlib>
using namespace metal;

#define SEL_NV   129280u
#define SEL_BLK  256u
#define SEL_NB   (SEL_NV / SEL_BLK)
#define SEL_CAP  512u
#define SEL_CAPB 1024u

struct sel_part { uint finite; float max; uint best; uint pad; };
struct sel_cand { float logit; uint id; };
struct sel_hdr {
    atomic_uint n;          /* K4 slot counter (all keys >= t; > SEL_CAPB means overflow) */
    uint finite, best, cap; float M;
    uint b1, above1, b2, above2, t, above;
    uint pad[5];
};

static inline bool sel_fin(float v) { return (as_type<uint>(v) & 0x7f800000u) != 0x7f800000u; }
static inline uint sel_key(float v) {
    uint u = as_type<uint>(v);
    if (u == 0x80000000u) u = 0u;
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

/* The selector exp spec (tools/selector_ref.h sel_vexp_spec, bit for bit): the engine vexp's
 * Cody-Waite reduction and degree-7 fused Horner, then one multiply by a normal 2^n giving a normal result; n <= -126 -> 0
 * (no subnormal result, nothing to reassociate). */
static inline float sel_vexp(float x) {
    x = max(x, -104.0f);
    const float n = rint(x * 1.44269504088896341f);
    float r = fma(-n, 0.693145751953125f, x);
    r = fma(-n, 1.428606765330187045e-06f, r);
    float p = 1.0f / 5040.0f;
    p = fma(p, r, 1.0f / 720.0f);
    p = fma(p, r, 1.0f / 120.0f);
    p = fma(p, r, 1.0f / 24.0f);
    p = fma(p, r, 1.0f / 6.0f);
    p = fma(p, r, 0.5f);
    p = fma(p, r, 1.0f);
    p = fma(p, r, 1.0f);
    const int ni = int(n);
    return ni <= -126 ? 0.0f : p * as_type<float>((uint)(ni + 127) << 23);
}

/* (max, earliest index) combine; ties compare ==, so +0/-0 are equal and the lower id wins */
static inline void sel_take(thread float &m, thread uint &bi, thread uint &f, float om, uint obi, uint of) {
    if (of && (!f || om > m || (om == m && obi < bi))) { m = om; bi = obi; }
    f += of;
}
static inline void sel_simd_maxid(thread float &m, thread uint &bi, thread uint &f) {
    for (ushort off = 16; off >= 1; off >>= 1) {
        const float om = simd_shuffle_down(m, off);
        const uint obi = simd_shuffle_down(bi, off), of = simd_shuffle_down(f, off);
        sel_take(m, bi, f, om, obi, of);
    }
}

/* suffix scan of a histogram from the top bin: the bin where the running count first reaches cap,
 * and the count strictly above it. 256 threads, PER = bins per thread (8 for 2048, 4 for 1024). */
template<uint NBIN>
static inline void sel_find_bin(device const atomic_uint *h, uint cap, uint base_above, ushort tid,
                                ushort lane, ushort sg, threadgroup uint *tg_sums, threadgroup uint *out) {
    constexpr uint PER = NBIN / 256u;
    uint c[PER], s = 0;
    for (uint k = 0; k < PER; k++) { c[k] = atomic_load_explicit(&h[NBIN - 1u - (tid * PER + k)], memory_order_relaxed); s += c[k]; }
    const uint ex = simd_prefix_exclusive_sum(s);
    if (lane == 31) tg_sums[sg] = ex + s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint sgbase = 0;
    for (ushort j = 0; j < sg; j++) sgbase += tg_sums[j];
    uint cum = base_above + sgbase + ex;
    if (cum < cap && cum + s >= cap) {
        for (uint k = 0; k < PER; k++) {
            if (cum + c[k] >= cap) { out[0] = NBIN - 1u - (tid * PER + k); out[1] = cum; break; }
            cum += c[k];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void kernel_sel_k1(device const float *lg [[buffer(0)]], device sel_part *part [[buffer(1)]],
                          device atomic_uint *H1 [[buffer(2)]], device atomic_uint *H2 [[buffer(3)]],
                          device atomic_uint *H3 [[buffer(4)]], device sel_hdr *hdr [[buffer(5)]],
                          uint b [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup atomic_uint th[2048];
    threadgroup float tm[8]; threadgroup uint tb[8], tf[8];
    for (uint k = tid; k < 2048u; k += 256u) atomic_store_explicit(&th[k], 0u, memory_order_relaxed);
    if (b == 0) {
        for (uint k = tid; k < 2048u; k += 256u) atomic_store_explicit(&H2[k], 0u, memory_order_relaxed);
        for (uint k = tid; k < 1024u; k += 256u) atomic_store_explicit(&H3[k], 0u, memory_order_relaxed);
        if (tid == 0) atomic_store_explicit(&hdr->n, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint i = b * SEL_BLK + tid;
    const float v = lg[i];
    uint f = sel_fin(v) ? 1u : 0u;
    if (f) atomic_fetch_add_explicit(&th[sel_key(v) >> 21], 1u, memory_order_relaxed);
    float m = v; uint bi = i;
    sel_simd_maxid(m, bi, f);
    if (lane == 0) { tm[sg] = m; tb[sg] = bi; tf[sg] = f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float M = tm[0]; uint B = tb[0], F = tf[0];
        for (ushort j = 1; j < 8; j++) sel_take(M, B, F, tm[j], tb[j], tf[j]);
        part[b] = sel_part{F, M, B, 0u};
    }
    for (uint k = tid; k < 2048u; k += 256u) {
        const uint c = atomic_load_explicit(&th[k], memory_order_relaxed);
        if (c) atomic_fetch_add_explicit(&H1[k], c, memory_order_relaxed);
    }
}

/* every threadgroup reduces the block partials to the same (finite, M, best) */
static inline void sel_global(device const sel_part *part, ushort tid, ushort lane, ushort sg,
                              threadgroup float *tm, threadgroup uint *tb, threadgroup uint *tf,
                              thread uint &F, thread float &M, thread uint &B) {
    float m = 0.0f; uint bi = 0, f = 0;
    for (uint j = tid; j < SEL_NB; j += 256u) { const sel_part q = part[j]; sel_take(m, bi, f, q.max, q.best, q.finite); }
    sel_simd_maxid(m, bi, f);
    if (lane == 0) { tm[sg] = m; tb[sg] = bi; tf[sg] = f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    M = tm[0]; B = tb[0]; F = tf[0];
    for (ushort j = 1; j < 8; j++) sel_take(M, B, F, tm[j], tb[j], tf[j]);
}

/* ---- Superset variant ("superset"): K1 (unchanged) + R1 + S3 ----
 * R1, ONE threadgroup: the ordered reduction of the 505 block records once (sel_global), cap = min(finite, 512),
 * the coarse bin b holding the cap-th finite key and the count above it (sel_find_bin over H1), Ncoarse =
 * above + H1[b]; publishes finite/best/cap/M, b (b1), above (above1), Ncoarse (b2) and the threshold key b << 21 (t);
 * then clears H1 for the next call (K3's job in the base variant). Zero finite: finite = cap = 0, H1 cleared. */
#define SEL_SUPB 2048u
kernel void kernel_sel_r1(device const sel_part *part [[buffer(1)]], device atomic_uint *H1 [[buffer(2)]],
                          device sel_hdr *hdr [[buffer(5)]],
                          ushort tid [[thread_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float tm[8]; threadgroup uint tb[8], tf[8], ts[8], tout[2];
    uint F, B; float M;
    sel_global(part, tid, lane, sg, tm, tb, tf, F, M, B);
    const uint cap = min(F, SEL_CAP);
    if (F != 0u) sel_find_bin<2048>(H1, cap, 0u, tid, lane, sg, ts, tout);   /* uniform branch */
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (tid == 0) {
        const uint b = F ? tout[0] : 0u, above = F ? tout[1] : 0u;
        const uint nc = F ? above + atomic_load_explicit(&H1[b], memory_order_relaxed) : 0u;
        hdr->finite = F; hdr->best = F ? B : 0u; hdr->cap = F ? cap : 0u; hdr->M = F ? M : 0.0f;
        hdr->b1 = b; hdr->above1 = above; hdr->b2 = nc; hdr->t = b << 21;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    for (uint k = tid; k < 2048u; k += 256u) atomic_store_explicit(&H1[k], 0u, memory_order_relaxed);
}

/* S3, 505 threadgroups: the block mass partials exactly as K2 (M from R1; same sel_vexp and shuffle tree, so the
 * same bits), and the compaction of EVERY finite key >= t = b << 21 into SEL_SUPB slots; the slot counter (cleared
 * by K1) counts all of them (> SEL_SUPB = overflow -> production on the CPU). Arrival order is arbitrary. */
kernel void kernel_sel_s3(device const float *lg [[buffer(0)]], device sel_hdr *hdr [[buffer(5)]],
                          device float *P [[buffer(6)]], device sel_cand *cand [[buffer(7)]],
                          uint b [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float tp[8];
    const uint cap = hdr->cap;
    if (cap == 0u) return;   /* zero finite: nothing consumed */
    const float M = hdr->M; const uint t = hdr->t;
    const uint i = b * SEL_BLK + tid;
    const float v = lg[i];
    const bool f = sel_fin(v);
    float a = f ? sel_vexp(v - M) : 0.0f;
    for (ushort off = 16; off >= 1; off >>= 1) a = a + simd_shuffle_down(a, off);
    if (lane == 0) tp[sg] = a;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {   /* the 8 simdgroup sums by the same shuffle tree (offsets 4, 2, 1): not reassociable */
        float s = lane < 8 ? tp[lane] : 0.0f;
        for (ushort off = 4; off >= 1; off >>= 1) s = s + simd_shuffle_down(s, off);
        if (lane == 0) P[b] = s;
    }
    if (f && sel_key(v) >= t) {
        const uint slot = atomic_fetch_add_explicit(&hdr->n, 1u, memory_order_relaxed);
        if (slot < SEL_SUPB) cand[slot] = sel_cand{v, i};
    }
}

/* ---- KEYED_GPU: candidates for the position-keyed Gumbel-max sample (ds4.c sample_keyed, full distribution) ----
 * R rows of 129280 logits, row r keyed by keys[r]. The float score of id i is l * inv_t + G, with G the Gumbel noise
 * -log(-log(u)) of u = ((h >> 11) + 0.5) 2^-53, h = mix(key + i * 0x9e3779b97f4a7c15): the same draw sample_keyed makes
 * in double, its -log(u) taken from 1 - u where u is near 1. KG1 (505 x R threadgroups): the block maximum of the
 * score (finite logits only) and the finite count, and clears the row's slot counter. KG2 (505 x R): the row maximum V
 * from the 505 block records (sel_global), then every finite id whose score is at least V - KG_MARGIN into KG_SLOTS
 * slots, the counter counting all of them. The score's float error is under 1e-4, so the double argmax (the keyed
 * sample) is always a candidate; the CPU evaluates the candidates as sample_keyed does. */
#define KG_SLOTS  64u
#define KG_MARGIN 0.015625f
struct kg_row { atomic_uint n; uint finite; uint pad[2]; uint id[KG_SLOTS]; };

static inline ulong kg_mix(ulong x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9UL;
    x ^= x >> 27; x *= 0x94d049bb133111ebUL;
    return x ^ (x >> 31);
}
static inline float kg_score(float l, float inv_t, ulong key, uint id) {
    const ulong x = kg_mix(key + (ulong)id * 0x9e3779b97f4a7c15UL) >> 11;
    float L;   /* -log(u) */
    if (x >= (1UL << 52)) {
        const float w = ((float)((1UL << 53) - 1UL - x) + 0.5f) * 0x1.0p-53f;   /* 1 - u */
        L = w < 0x1.0p-10f ? w * (1.0f + w * (0.5f + w * (1.0f / 3.0f))) : -precise::log(1.0f - w);
    } else {
        L = -precise::log(((float)x + 0.5f) * 0x1.0p-53f);
    }
    return l * inv_t - precise::log(L);
}

kernel void kernel_kg1(device const float *lg [[buffer(0)]], device sel_part *part [[buffer(1)]],
                       device kg_row *rows [[buffer(2)]], constant ulong *keys [[buffer(3)]],
                       constant float &inv_t [[buffer(4)]],
                       uint2 g [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                       ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float tm[8]; threadgroup uint tb[8], tf[8];
    const uint b = g.x, r = g.y, i = b * SEL_BLK + tid;
    if (b == 0 && tid == 0) atomic_store_explicit(&rows[r].n, 0u, memory_order_relaxed);
    const float v = lg[(ulong)r * SEL_NV + i];
    uint f = sel_fin(v) ? 1u : 0u;
    float m = f ? kg_score(v, inv_t, keys[r], i) : 0.0f; uint bi = i;
    sel_simd_maxid(m, bi, f);
    if (lane == 0) { tm[sg] = m; tb[sg] = bi; tf[sg] = f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float M = tm[0]; uint B = tb[0], F = tf[0];
        for (ushort j = 1; j < 8; j++) sel_take(M, B, F, tm[j], tb[j], tf[j]);
        part[r * SEL_NB + b] = sel_part{F, M, B, 0u};
    }
}

kernel void kernel_kg2(device const float *lg [[buffer(0)]], device const sel_part *part [[buffer(1)]],
                       device kg_row *rows [[buffer(2)]], constant ulong *keys [[buffer(3)]],
                       constant float &inv_t [[buffer(4)]],
                       uint2 g [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                       ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float tm[8]; threadgroup uint tb[8], tf[8];
    const uint b = g.x, r = g.y, i = b * SEL_BLK + tid;
    uint F, B; float V;
    sel_global(part + r * SEL_NB, tid, lane, sg, tm, tb, tf, F, V, B);
    if (b == 0 && tid == 0) rows[r].finite = F;
    const float v = lg[(ulong)r * SEL_NV + i];
    if (F == 0u || !sel_fin(v) || kg_score(v, inv_t, keys[r], i) < V - KG_MARGIN) return;
    const uint slot = atomic_fetch_add_explicit(&rows[r].n, 1u, memory_order_relaxed);
    if (slot < KG_SLOTS) rows[r].id[slot] = i;
}
