#ifndef DS4_DEEPSEEK41_GPU_H
#define DS4_DEEPSEEK41_GPU_H

#include <stdbool.h>
#include <stdint.h>

#ifndef DS4_GPU_TENSOR_DEFINED
#define DS4_GPU_TENSOR_DEFINED
typedef struct ds4_gpu_tensor ds4_gpu_tensor;
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* V4.1 activation/cache formats. Buffers are float-addressable but the
 * rounded values follow the released BF16/FP8/FP4 inference graph. */
typedef enum {
    DS4_V41_BF16 = 0,
    DS4_V41_FP8_E8M0 = 1,
    DS4_V41_FP4_E8M0 = 2,
    DS4_V41_FP4_E4M3 = 3,
} ds4_v41_activation_format;
int ds4_gpu_dsv41_quantize(ds4_gpu_tensor *x, uint32_t width, uint32_t rows,
                          ds4_v41_activation_format format);
#ifdef __APPLE__
/* One-token router: probabilities, biased top-k and normalised weights in one dispatch. */
int ds4_gpu_dsv41_router_one(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
                             ds4_gpu_tensor *probs, const ds4_gpu_tensor *logits,
                             const void *model_map, uint64_t model_size, uint64_t bias_offset,
                             uint32_t n_expert, uint32_t top_k, float scale);
/* The same for a few rows, and the rows' F32 logits with the single-row matvec's reduction. */
int ds4_gpu_dsv41_router_rows(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
                              ds4_gpu_tensor *probs, const ds4_gpu_tensor *logits,
                              const void *model_map, uint64_t model_size, uint64_t bias_offset,
                              uint32_t n_expert, uint32_t top_k, float scale, uint32_t rows);
int ds4_gpu_dsv41_matmul_f32_rows(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                  uint64_t weight_offset, uint32_t in_dim, uint32_t out_dim,
                                  const ds4_gpu_tensor *x, uint32_t rows);
/* Producers that write their outputs already rounded to bf16, the same bits as the producer followed by the
 * rounding pass. One row each (the norm, the sums and the expand take one token). The Q8 bf16io matvec also rounds
 * its activation as it loads it. */
int ds4_gpu_matmul_q8_0_bf16_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                    uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
                                    const ds4_gpu_tensor *x);
int ds4_gpu_matmul_q8_0_bf16io_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                      uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
                                      ds4_gpu_tensor *x);
/* 0 when the shape has no rounding kernel; the caller then rounds separately. */
int ds4_gpu_dsv41_project_f16_bf16(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                   uint64_t weight_offset, uint32_t in_dim, uint32_t out_dim,
                                   const ds4_gpu_tensor *x);
int ds4_gpu_rms_norm_weight_bf16_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
                                        const void *model_map, uint64_t model_size,
                                        uint64_t weight_offset, uint32_t n, float eps);
int ds4_gpu_hc_weighted_sum_bf16_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *residual_hc,
                                        const ds4_gpu_tensor *weights, uint32_t n_embd, uint32_t n_hc);
int ds4_gpu_hc_weighted_sum_split_bf16_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *residual_hc,
                                              const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc);
int ds4_gpu_hc_expand_split_bf16_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out,
                                        const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split,
                                        uint32_t n_embd, uint32_t n_hc);
/* The whole head rounded to bf16, then the RoPE. */
int ds4_gpu_dsv41_rope_bf16(ds4_gpu_tensor *x, uint32_t width, uint32_t heads,
                            uint32_t rows, uint32_t start, bool compressed, bool inverse);
/* QKV_FUSION. The q_a and kv projections of one row in one dispatch (bf16 outputs), and both of their norms in one
 * dispatch; the q projection with its RoPE applied to the bf16 outputs (the last 64 columns of every 512-wide head).
 * Each returns 0, encoding nothing, when its kernel does not apply. */
int ds4_gpu_dsv41_project_pair_q8(ds4_gpu_tensor *out_a, ds4_gpu_tensor *out_b,
                                  const void *model_map, uint64_t model_size,
                                  uint64_t offset_a, uint64_t offset_b, uint32_t in_dim,
                                  uint32_t out_a_dim, uint32_t out_b_dim, const ds4_gpu_tensor *x);
int ds4_gpu_dsv41_norm_pair_rows(ds4_gpu_tensor *out0, const ds4_gpu_tensor *x0, uint64_t weight0_offset, uint32_t n0,
                                 ds4_gpu_tensor *out1, const ds4_gpu_tensor *x1, uint64_t weight1_offset, uint32_t n1,
                                 const void *model_map, uint64_t model_size, float eps, uint32_t rows);
int ds4_gpu_dsv41_project_q(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                            uint64_t weight_offset, uint32_t in_dim, uint32_t out_dim,
                            const ds4_gpu_tensor *x, uint32_t pos, bool compressed);
/* ROPE. rope_quantize: one row's RoPE and quantization stored into dst at dst_offset bytes, x unchanged.
 * rope_coeff_table: one decode step's RoPE coefficients (64 float2: (cos, sin) of float(pos) * frequency, entries
 * 0..31 raw, 32..63 compressed). rope_quantize_heads_tab: RoPE + quantization of heads rows of width in place, in one
 * dispatch reading it. attention_low: the attention output low projection with the heads' bf16 rounding and inverse
 * RoPE applied as it loads them. heads_once_arm: the next single-row decode reduce of 64 heads x 512 writes its heads
 * already bf16-rounded and inverse-roped with the table; take returns 1 when it did (and disarms), and the plain low
 * projection then reads them. reduce_check (test entry): that reduce on tmp's partials, plain or transformed. */
int ds4_gpu_dsv41_rope_quantize(const ds4_gpu_tensor *x, ds4_gpu_tensor *dst, uint64_t dst_offset,
                                uint32_t width, uint32_t start, bool compressed,
                                ds4_v41_activation_format format);
int ds4_gpu_dsv41_rope_coeff_table(ds4_gpu_tensor *tab, uint32_t pos);
int ds4_gpu_dsv41_rope_quantize_heads_tab(ds4_gpu_tensor *x, uint32_t width, uint32_t heads, uint32_t start,
                                          bool compressed, ds4_v41_activation_format format, const ds4_gpu_tensor *tab);
int ds4_gpu_dsv41_attention_low(ds4_gpu_tensor *low, const void *model_map, uint64_t model_size,
                                uint64_t out_a_offset, uint32_t group_dim, uint32_t rank,
                                uint32_t n_groups, ds4_gpu_tensor *heads, uint32_t pos, bool compressed);
void ds4_gpu_dsv41_heads_once_arm(const ds4_gpu_tensor *tab, bool compressed);
int ds4_gpu_dsv41_heads_once_take(void);
int ds4_gpu_dsv41_flash_attn_reduce_check(ds4_gpu_tensor *heads, const ds4_gpu_tensor *tmp, uint32_t nrows,
                                          int transform, const ds4_gpu_tensor *tab, bool compressed);
/* FIXED_SHAPES: ds4_gpu_attention_output_low_q8_tensor at V4.1's decode shape (8 groups of 4096 -> 1024) with that
 * shape as literals; 0 for any other shape. */
int ds4_gpu_dsv41_attention_low_static(ds4_gpu_tensor *low, const void *model_map, uint64_t model_size,
                                       uint64_t out_a_offset, uint32_t group_dim, uint32_t rank, uint32_t n_groups,
                                       const ds4_gpu_tensor *heads);
/* DECODE_ATTENTION: ds4_gpu_attention_decode_heads_tensor of one row over the raw window and the `attended` rows of
 * the f32 compressed cache that ids picks, staged straight from the cache. 0: not run, gather and attend as before. */
int ds4_gpu_dsv41_attention_decode(ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size,
                                   uint64_t sinks_offset, const ds4_gpu_tensor *q,
                                   const ds4_gpu_tensor *raw_kv, uint32_t n_raw, uint32_t raw_cap,
                                   uint32_t raw_start, const ds4_gpu_tensor *comp_cache,
                                   const ds4_gpu_tensor *ids, uint32_t n_comp, uint32_t attended,
                                   uint32_t n_head, uint32_t head_dim);
/* INDEX_CANDIDATES: one decode row's top_k (at most 512) index ids, scored and picked among the candidate blocks of
 * block_mask (more than 2,048 blocks, 2,048 kept) only, as the masked wide scorer and the TOPK radix select pick
 * them; list holds the kept blocks (built unless list_ready). 0: not run, score the full row. */
int ds4_gpu_dsv41_indexer_candidates_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *scores, ds4_gpu_tensor *list,
                                           const ds4_gpu_tensor *q, const ds4_gpu_tensor *weights,
                                           const ds4_gpu_tensor *keys, const ds4_gpu_tensor *view,
                                           const ds4_gpu_tensor *block_mask, uint32_t n_comp, uint32_t top_k,
                                           int list_ready);
/* The index top-k of V4.1's attention (ds4_gpu_indexer_topk_tensor's ids in its order), which TOPK runs as a radix
 * select on rows of 16,384 scores or more and as a shorter sort and merge on one row's top 512. */
int ds4_gpu_dsv41_indexer_topk_tensor(ds4_gpu_tensor *selected, const ds4_gpu_tensor *scores,
                                      uint32_t n_comp, uint32_t n_tokens, uint32_t top_k);
/* INDEX_BF16: index_keys_bf16 writes rows [first, first + rows) of the f32 index keys into their bf16 view (with
 * flag, a key that is not bf16-exact sets *flag); indexer_score_row scores one token over `rows` keys, from the view
 * where the wide scorer runs, else from the f32 keys. */
int ds4_gpu_dsv41_index_keys_bf16(ds4_gpu_tensor *view, const ds4_gpu_tensor *cache, uint32_t first, uint32_t rows,
                                  ds4_gpu_tensor *flag);
int ds4_gpu_dsv41_indexer_score_row(ds4_gpu_tensor *scores, const ds4_gpu_tensor *q, const ds4_gpu_tensor *weights,
                                    const ds4_gpu_tensor *cache, const ds4_gpu_tensor *view, uint32_t rows);
/* INDEX_SCORE_WIDE: one token's index scores over `rows` f32 keys (32 heads of 128, scale 1/64) with the wide
 * scorer, the keys of blocks of 8 that the block mask excludes set to -inf unread. */
int ds4_gpu_dsv41_indexer_score_masked(ds4_gpu_tensor *scores, const ds4_gpu_tensor *q, const ds4_gpu_tensor *weights,
                                       const ds4_gpu_tensor *cache, const ds4_gpu_tensor *mask, uint32_t rows);
/* CONCURRENT_FFN: ds4_gpu_parallel_ffn_start with the shared expert's bf16 boundaries (gate and up rounded before
 * the SwiGLU, the down projection bf16 in and out). */
int ds4_gpu_parallel_ffn_start_bf16(ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid,
                                    ds4_gpu_tensor *shared_out, const void *model_map, uint64_t model_size,
                                    uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
                                    uint32_t model_dim, uint32_t shared_dim, const ds4_gpu_tensor *x, float clamp);
/* SHARED_EXPERT: the shared expert's mid (gate and up rounded to bf16, then the SwiGLU) for up to 65535 rows. */
int ds4_gpu_dsv41_shared_swiglu(ds4_gpu_tensor *mid, const void *model_map, uint64_t model_size,
                                uint64_t gate_offset, uint64_t up_offset, uint32_t in_dim, uint32_t out_dim,
                                const ds4_gpu_tensor *x, float clamp, uint32_t rows);
/* HC_FUSION. hc_block_input(_rows): a block's whole HC input (mix, split, the weighted sum with the previous block's
 * pre weights, its norm) for up to 8 rows. matmul_expand: a bf16 in/out Q8_0 matvec of one row expanded into the HC
 * streams (joined with add first when given). deferred_collapse: the weighted sum and norm alone, ahead of the mix;
 * query_deferred_hc: the attention query projection with RoPE that also runs the deferred block's HC block input and
 * stores its KV row. Each returns 0, encoding nothing, when its kernel does not apply. */
int ds4_gpu_dsv41_hc_block_input_rows(ds4_gpu_tensor *mix, ds4_gpu_tensor *x, ds4_gpu_tensor *norm,
                                      ds4_gpu_tensor *split, const ds4_gpu_tensor *stream,
                                      const ds4_gpu_tensor *pre, const void *model_map, uint64_t model_size,
                                      uint64_t fn_offset, uint64_t scale_offset, uint64_t base_offset,
                                      uint64_t norm_offset, uint32_t n, uint32_t mix_dim, uint32_t n_embd,
                                      uint32_t n_hc, uint32_t sinkhorn_iters, float hc_eps, float norm_eps,
                                      uint32_t rows, uint32_t pre_stride);
int ds4_gpu_dsv41_hc_block_input(ds4_gpu_tensor *mix, ds4_gpu_tensor *x, ds4_gpu_tensor *norm,
                                 ds4_gpu_tensor *split, const ds4_gpu_tensor *stream,
                                 const ds4_gpu_tensor *pre, const void *model_map, uint64_t model_size,
                                 uint64_t fn_offset, uint64_t scale_offset, uint64_t base_offset,
                                 uint64_t norm_offset, uint32_t n, uint32_t mix_dim, uint32_t n_embd,
                                 uint32_t n_hc, uint32_t sinkhorn_iters, float hc_eps, float norm_eps);
int ds4_gpu_dsv41_matmul_expand(ds4_gpu_tensor *out_hc, const void *model_map, uint64_t model_size,
                                uint64_t weight_offset, uint32_t in_dim, uint32_t out_dim,
                                const ds4_gpu_tensor *x, const ds4_gpu_tensor *add,
                                const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_hc);
int ds4_gpu_dsv41_deferred_collapse(ds4_gpu_tensor *x, ds4_gpu_tensor *norm, const ds4_gpu_tensor *residual,
                                    const ds4_gpu_tensor *pre, const void *model_map, uint64_t model_size,
                                    uint64_t norm_offset, float eps);
int ds4_gpu_dsv41_query_deferred_hc(ds4_gpu_tensor *q, ds4_gpu_tensor *mix, ds4_gpu_tensor *split,
                                    const ds4_gpu_tensor *qr, const ds4_gpu_tensor *residual,
                                    const void *model_map, uint64_t model_size, uint64_t q_offset, uint64_t fn_offset,
                                    uint64_t scale_offset, uint64_t base_offset, uint32_t pos, bool compressed,
                                    uint32_t iters, float hc_eps, float norm_eps, ds4_gpu_tensor *x, ds4_gpu_tensor *norm,
                                    const ds4_gpu_tensor *pre, uint64_t norm_offset,
                                    const ds4_gpu_tensor *kv, ds4_gpu_tensor *cache, uint64_t cache_offset);
/* FFN_JOIN: the HC expand of bf16(block_out + block_add), its outputs rounded to bf16. */
int ds4_gpu_hc_expand_add_split_bf16_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out,
                                            const ds4_gpu_tensor *block_add, const ds4_gpu_tensor *residual_hc,
                                            const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc);
/* Weights built at load: the model views' storage options, and a residency set of their own (requested at once and
 * added to the queue); ds4_gpu_weight_residency replaces that set, n == 0 releases it. */
ds4_gpu_tensor *ds4_gpu_tensor_alloc_weights(uint64_t bytes);
int ds4_gpu_weight_residency(ds4_gpu_tensor *const *tensors, uint32_t n);
/* The decode router over BF16 gates, and the Engram projection and indexer q_b over E4M3 codes with E8M0 32x32 block
 * bytes (outputs rounded to bf16): the wide matvecs' bits. 0, encoding nothing, when the shape does not fit. */
int ds4_gpu_matmul_bf16_router_weights(ds4_gpu_tensor *out, const ds4_gpu_tensor *weights,
                                       uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x);
int ds4_gpu_matmul_engram_fp8_weights(ds4_gpu_tensor *out, const ds4_gpu_tensor *codes, const ds4_gpu_tensor *e8m0,
                                      uint32_t in_dim, uint32_t out_dim, const ds4_gpu_tensor *x);
int ds4_gpu_dsv41_project_fp8_bf16_weights(ds4_gpu_tensor *out, const ds4_gpu_tensor *codes, const ds4_gpu_tensor *e8m0,
                                           uint32_t in_dim, uint32_t out_dim, const ds4_gpu_tensor *x);
#endif
#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD) && !defined(DS4_NO_GPU)
/* Same reductions as the ordinary kernels, with the required BF16 store
 * folded into the operation. hc_sum reads either 4- or 24-float row weights. */
int ds4_gpu_dsv41_norm(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const void *map, uint64_t size, uint64_t offset,
        uint32_t width, uint32_t rows, float eps);
int ds4_gpu_dsv41_hc_sum(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *weights, uint32_t width, uint32_t rows,
        uint32_t weight_stride);
int ds4_gpu_dsv41_hc_expand(ds4_gpu_tensor *out, const ds4_gpu_tensor *block,
        const ds4_gpu_tensor *residual, const ds4_gpu_tensor *split,
        uint32_t width, uint32_t rows);
/* CUDA scalar Q8 shared expert. 1: queued; 0: unsupported, no work queued;
 * -1: failure. After 1, keep the input/output tensors alive and unchanged
 * until join, which orders the result before subsequent main-stream work.
 * Other work may run between start and join using separate tensors.
 * Optional ownership picks the rank with fewer selected routed experts.
 * IDs are read on the main stream before later owned-expert filtering. */
typedef struct {
    const ds4_gpu_tensor *selected;
    uint32_t experts, active, rank, tie_owner;
} ds4_v41_shared_owner;
int ds4_gpu_dsv41_shared_start(
        ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid, const ds4_gpu_tensor *x,
        const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint32_t width, uint32_t hidden, float clamp,
        const ds4_v41_shared_owner *owner);
int ds4_gpu_dsv41_shared_join(void);
/* Mask a computed short batch before the owned router filter mutates IDs. */
int ds4_gpu_dsv41_shared_mask(ds4_gpu_tensor *shared, const ds4_gpu_tensor *selected,
        uint32_t width, uint32_t rows, uint32_t experts, uint32_t active,
        uint32_t rank, uint32_t tie_owner);
#endif
/* Full-head prefill, with BF16 rounding between the two Q8 projections. */
int ds4_gpu_dsv41_attention_output_batch(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low,
        const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t out_b_offset,
        const ds4_gpu_tensor *heads, uint32_t n_tokens);
/* Packed local 32-head input and BF16 low projection; output is an unrounded
 * rank partial. The graph sums ranks before rounding the attention block. */
int ds4_gpu_dsv41_attention_output_tp_batch(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low,
        const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t out_b_offset,
        const ds4_gpu_tensor *heads, uint32_t n_tokens, uint32_t tp_rank);
/* Adjacent-pair, unit-magnitude RoPE with the released V4.1 frequencies. */
int ds4_gpu_dsv41_rope(ds4_gpu_tensor *x, uint32_t width, uint32_t heads,
                      uint32_t rows, uint32_t start, bool compressed, bool inverse);
/* Compressed pairs advance two absolute token positions per stored row. */
int ds4_gpu_dsv41_rope_stride(ds4_gpu_tensor *x, uint32_t width, uint32_t heads,
                             uint32_t rows, uint32_t start, uint32_t stride,
                             bool compressed, bool inverse);
int ds4_gpu_dsv41_engram_add(ds4_gpu_tensor *residual,
                           const ds4_gpu_tensor *kv,
                           const ds4_gpu_tensor *q_weight,
                           const ds4_gpu_tensor *k_weight,
                           const ds4_gpu_tensor *mask,
                           uint32_t width, uint32_t rows, float eps);
/* Pool complete pairs and retain the last unpaired projection in previous_*.
 * start is the absolute token position, including earlier chunks. */
int ds4_gpu_dsv41_pool2(ds4_gpu_tensor *out,
                      const ds4_gpu_tensor *kv, const ds4_gpu_tensor *scores,
                      ds4_gpu_tensor *previous_kv, ds4_gpu_tensor *previous_scores,
                       uint32_t width, uint32_t rows, uint32_t start);
/* Candidate blocks contain eight compressed positions. Produce causal block
 * maxima, pinning the newest block; filter consumes a 0/-inf block mask. */
int ds4_gpu_dsv41_candidate_blocks(ds4_gpu_tensor *blocks,
                                  const ds4_gpu_tensor *scores,
                                  uint32_t width, uint32_t rows,
                                  uint32_t start, uint32_t ratio);
int ds4_gpu_dsv41_candidate_filter(ds4_gpu_tensor *scores,
                                  const ds4_gpu_tensor *block_mask,
                                  uint32_t width, uint32_t rows,
                                  uint32_t start, uint32_t ratio);
/* Causal index scores over ratio-1/2 compressed keys, without an extra cast
 * of the already quantized FP4 queries/keys. Scores have source_rows stride. */
int ds4_gpu_dsv41_indexer_scores_batch(ds4_gpu_tensor *scores,
                                     const ds4_gpu_tensor *q,
                                     const ds4_gpu_tensor *weights,
                                     const ds4_gpu_tensor *keys,
                                     uint32_t source_rows, uint32_t rows,
                                     uint32_t start, uint32_t ratio);
int ds4_gpu_dsv41_tensor_ops_available(void);
/* Reuse exact BF16 views of FP4 queries/keys across score tiles. Invalid
 * casts retain the F32 arithmetic for the affected tile. */
uint64_t ds4_gpu_dsv41_indexer_packed_bytes(uint32_t source_rows, uint32_t rows);
int ds4_gpu_dsv41_indexer_pack(ds4_gpu_tensor *packed,
                              const ds4_gpu_tensor *q, const ds4_gpu_tensor *keys,
                              uint32_t source_rows, uint32_t rows);
int ds4_gpu_dsv41_indexer_scores_packed(ds4_gpu_tensor *scores,
                                      const ds4_gpu_tensor *q,
                                      const ds4_gpu_tensor *weights,
                                      const ds4_gpu_tensor *keys,
                                      const ds4_gpu_tensor *packed,
                                      uint32_t source_rows, uint32_t rows,
                                      uint32_t start, uint32_t ratio,
                                      uint32_t packed_rows, uint32_t offset);
/* Exact row-sort ordering with independent causal widths; at least 1024
 * visible keys per row. Output stride is 512 indices. */
int ds4_gpu_dsv41_indexer_topk_batch(ds4_gpu_tensor *selected,
                                    const ds4_gpu_tensor *scores,
                                    uint32_t width, uint32_t rows,
                                    uint32_t start, uint32_t ratio);
enum { DS4_V41_CARRY_BF16, DS4_V41_CARRY_MASK, DS4_V41_CARRY_F32 };
/* Lossless storage for already-BF16 activations or 0/-inf candidate masks.
 * Packed rows are padded to whole uint32_t words. Plain rows remain F32. */
int ds4_gpu_dsv41_carry_copy(ds4_gpu_tensor *packed, uint32_t row_offset,
                            ds4_gpu_tensor *plain, uint32_t width, uint32_t rows,
                            uint32_t format, bool pack);
/* Batched F16 projections with the same arithmetic as individual matvecs. */
int ds4_gpu_dsv41_projection_rows(ds4_gpu_tensor *out,
                                 const void *model_map, uint64_t model_size,
                                 uint64_t weight_offset, uint32_t width,
                                 uint32_t outputs, uint32_t rows,
                                 const ds4_gpu_tensor *in);
/* Gather 512-wide F32 KV rows; IDs must come from top-k over source_rows. */
int ds4_gpu_dsv41_gather_kv(ds4_gpu_tensor *out, const ds4_gpu_tensor *source,
                           const ds4_gpu_tensor *ids, uint32_t source_rows,
                           uint32_t selected_rows);

#ifdef __cplusplus
}
#endif
#endif
