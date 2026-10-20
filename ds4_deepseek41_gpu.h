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
/* One-token router: probabilities, biased top-k and normalised weights in one dispatch. */
int ds4_gpu_dsv41_indexer_score_masked(ds4_gpu_tensor *scores,
        const ds4_gpu_tensor *q, const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *cache, const ds4_gpu_tensor *mask, uint32_t rows);
/* The index-key cache's bf16 view and the scorer over it (the wide scorer's
 * bits). The conversion with a flag tensor sets it nonzero on any key that is
 * not bf16-exact. */
int ds4_gpu_dsv41_index_keys_bf16(ds4_gpu_tensor *view, const ds4_gpu_tensor *cache,
        uint32_t first, uint32_t rows, ds4_gpu_tensor *flag);
int ds4_gpu_dsv41_indexer_score_bf16(ds4_gpu_tensor *scores, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights, const ds4_gpu_tensor *view, uint32_t rows);

int ds4_gpu_dsv41_router_one(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
                             ds4_gpu_tensor *probs, const ds4_gpu_tensor *logits,
                             const void *model_map, uint64_t model_size, uint64_t bias_offset,
                             uint32_t n_expert, uint32_t top_k, float scale);
/* The same for a few rows (Metal only), and the rows' F32 logits with the
 * single-row matvec's reduction. */
int ds4_gpu_dsv41_router_rows(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
                              ds4_gpu_tensor *probs, const ds4_gpu_tensor *logits,
                              const void *model_map, uint64_t model_size, uint64_t bias_offset,
                              uint32_t n_expert, uint32_t top_k, float scale, uint32_t rows);
int ds4_gpu_dsv41_matmul_f32_rows(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                  uint64_t weight_offset, uint32_t in_dim, uint32_t out_dim,
                                  const ds4_gpu_tensor *x, uint32_t rows);
int ds4_gpu_dsv41_quantize(ds4_gpu_tensor *x, uint32_t width, uint32_t rows,
                          ds4_v41_activation_format format);
#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD) && !defined(DS4_NO_GPU)
/* CUDA scalar Q8 shared expert. 1: queued; 0: unsupported, no work queued;
 * -1: failure. After 1, keep the input/output tensors alive and unchanged
 * until join, which orders the result before subsequent main-stream work.
 * Other work may run between start and join using separate tensors. */
int ds4_gpu_dsv41_shared_start(
        ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid, const ds4_gpu_tensor *x,
        const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint32_t width, uint32_t hidden, float clamp);
int ds4_gpu_dsv41_shared_join(void);
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
int ds4_gpu_dsv41_rope_bf16(ds4_gpu_tensor *x, uint32_t width, uint32_t heads,
                            uint32_t rows, uint32_t start, bool compressed, bool inverse);
int ds4_gpu_dsv41_rope_stride(ds4_gpu_tensor *x, uint32_t width, uint32_t heads,
                             uint32_t rows, uint32_t start, uint32_t stride,
                             bool compressed, bool inverse);
int ds4_gpu_dsv41_engram_add(ds4_gpu_tensor *residual,
                           const ds4_gpu_tensor *kv,
                           const ds4_gpu_tensor *q_weight,
                           const ds4_gpu_tensor *k_weight,
                           const ds4_gpu_tensor *mask,
                           uint32_t width, uint32_t rows, float eps);
/* Metal: while on, an unmasked Engram gate whose width is a multiple of 1024 runs with one 256-thread group per
 * stream (set by the V4.1 decode step for its own duration); the count of such dispatches. */
void ds4_gpu_set_engram_add_tg(int on);
int ds4_gpu_get_engram_add_tg(void);
uint64_t ds4_gpu_engram_add_tg_count(void);
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
                                  uint32_t start, uint32_t ratio,
                                  uint32_t mask_stride);
/* Per-row block top-2048 with the causal block count of ratio-1 rows, and
 * the 0/-inf block masks it selects at mask_stride floats per row. */
int ds4_gpu_dsv41_candidate_topk_batch(ds4_gpu_tensor *selected,
                                      const ds4_gpu_tensor *blocks,
                                      uint32_t width, uint32_t rows, uint32_t start);
int ds4_gpu_dsv41_candidate_mask_batch(ds4_gpu_tensor *mask,
                                      const ds4_gpu_tensor *selected,
                                      uint32_t width, uint32_t rows,
                                      uint32_t mask_stride);
/* DSpark. The mean over the hc copies of each row, into out[row][out_off..]
 * at out_stride floats per row. */
int ds4_gpu_dsv41_hc_mean(uint32_t rows, uint32_t dim, uint32_t hc,
                         const ds4_gpu_tensor *stream, ds4_gpu_tensor *out,
                         uint32_t out_stride, uint32_t out_off);
/* Draft `block` tokens in sequence: tokens[step + 1] is the argmax of
 * logits[step] + head . embed[tokens[step]], conf[step] the confidence logit
 * proj . [x[step], embed[tokens[step]]]. Tables are [vocab, rank], F16 or F32;
 * parts holds 2 * n_parts floats of scratch. */
int ds4_gpu_dsv41_markov_chain(uint32_t block, uint32_t vocab, uint32_t rank, uint32_t dim,
                              const ds4_gpu_tensor *logits, const ds4_gpu_tensor *x,
                              const void *model_map, uint64_t model_size,
                              uint64_t embed_offset, uint64_t head_offset, int f16,
                              const ds4_gpu_tensor *conf_proj, ds4_gpu_tensor *tokens,
                              ds4_gpu_tensor *conf, ds4_gpu_tensor *parts, uint32_t n_parts);
/* Causal index scores over ratio-1/2 compressed keys, without an extra cast
 * of the already quantized FP4 queries/keys. Scores have source_rows stride. */
int ds4_gpu_dsv41_indexer_scores_batch(ds4_gpu_tensor *scores,
                                     const ds4_gpu_tensor *q,
                                     const ds4_gpu_tensor *weights,
                                     const ds4_gpu_tensor *keys,
                                     uint32_t source_rows, uint32_t rows,
                                     uint32_t start, uint32_t ratio);
/* The same for 2-7 verify rows over the keys' bf16 view (the same bits); rows
 * from mask_first on may leave -inf where their block-mask row (mask_stride
 * floats) drops a block. 0, scoring nothing, where unavailable (Metal only). */
int ds4_gpu_dsv41_indexer_scores_rows_bf16(ds4_gpu_tensor *scores, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights, const ds4_gpu_tensor *view, const ds4_gpu_tensor *mask,
        uint32_t mask_stride, uint32_t mask_first, uint32_t source_rows, uint32_t rows,
        uint32_t start, uint32_t ratio);
/* The top-k of a row filtered by a block mask of more than 2048 blocks, from
 * the scores of its 2048 candidate blocks only: the same ids in the same order
 * (Metal only). Returns 0, leaving the full row to the caller, where unavailable. */
int ds4_gpu_dsv41_indexer_candidates_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *scores,
                                           ds4_gpu_tensor *list, const ds4_gpu_tensor *q,
                                           const ds4_gpu_tensor *weights, const ds4_gpu_tensor *keys,
                                           const ds4_gpu_tensor *view,
                                           const ds4_gpu_tensor *block_mask, uint32_t n_comp,
                                           float scale, uint32_t top_k, int list_ready);
/* view: keys' bf16 view (ds4_gpu_dsv41_index_keys_bf16) or NULL; the candidate scores read it when set (the same
 * bits). By default two lanes score each key and one threadgroup selects (DS4_METAL_DISABLE_V41_CAND_SPLIT=1 and
 * DS4_METAL_DISABLE_V41_CAND_SELECT_ONE=1 restore the one-lane scorer and the radix select; read per call). */
/* list_ready != 0: list already holds block_mask's candidates (an earlier call since the mask was written); the list
 * is then not rebuilt, and the winners' sort writes the key ids through it (DS4_METAL_DISABLE_V41_INDEX_CAND_ONCE=1
 * rebuilds it every call and maps the ids in their own dispatch). */
/* Test entry implemented by the engine (ds4.c, Metal builds): the decode graph's candidate-only index row at pos on
 * mask, deciding list_ready from the record *list_pos / *list_mask and updating it as the graph does. */
int ds4_test_v41_candidates_row(ds4_gpu_tensor *selected, ds4_gpu_tensor *scores, ds4_gpu_tensor *list,
                                ds4_gpu_tensor *q, ds4_gpu_tensor *weights, ds4_gpu_tensor *keys,
                                ds4_gpu_tensor *mask, uint32_t n_comp, uint32_t pos,
                                uint32_t *list_pos, const ds4_gpu_tensor **list_mask);
/* Test entry implemented by the engine (ds4.c, Metal builds): the decode graph's full index row for owner 0 over
 * keys and their bf16 view, after a restore of n_comp rows when restore is set. 0 failed, 1 scored the f32 keys,
 * 2 scored the view. */
int ds4_test_v41_index_score_row(ds4_gpu_tensor *scores, ds4_gpu_tensor *q, ds4_gpu_tensor *weights,
                                 ds4_gpu_tensor *keys, ds4_gpu_tensor *view, ds4_gpu_tensor *flag,
                                 uint32_t n_comp, int restore);
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
/* Exact row-sort ordering with independent causal widths; more than 512
 * visible keys per row. Output stride is 512 indices. */
int ds4_gpu_dsv41_indexer_topk_batch(ds4_gpu_tensor *selected,
                                    const ds4_gpu_tensor *scores,
                                    uint32_t width, uint32_t rows,
                                    uint32_t start, uint32_t ratio);
/* Rows with at most 512 visible keys select all of them, in key order. */
int ds4_gpu_dsv41_indexer_all_batch(ds4_gpu_tensor *selected, uint32_t rows,
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
/* The HC mixer projection of one row with its norm fused (0: not fused). */
void ds4_gpu_dsv41_verify_rows(int on);
int ds4_gpu_dsv41_hc_project(ds4_gpu_tensor *mix, const ds4_gpu_tensor *residual,
                             const void *model_map, uint64_t model_size, uint64_t fn_offset,
                             uint32_t n, uint32_t mix_dim, float eps);
/* One block's whole HC input in one dispatch: the mixer projection with its
 * norm, then split, weighted sum with `pre`'s first n_hc weights and norm
 * (0: not fused). */
int ds4_gpu_dsv41_hc_block_input(ds4_gpu_tensor *mix, ds4_gpu_tensor *x, ds4_gpu_tensor *norm,
                                 ds4_gpu_tensor *split, const ds4_gpu_tensor *stream,
                                 const ds4_gpu_tensor *pre, const void *model_map, uint64_t model_size,
                                 uint64_t fn_offset, uint64_t scale_offset, uint64_t base_offset,
                                 uint64_t norm_offset, uint32_t n, uint32_t mix_dim, uint32_t n_embd,
                                 uint32_t n_hc, uint32_t sinkhorn_iters, float hc_eps, float norm_eps);
/* The same for a few rows: `pre` advances pre_stride floats per row. */
int ds4_gpu_dsv41_hc_block_input_rows(ds4_gpu_tensor *mix, ds4_gpu_tensor *x, ds4_gpu_tensor *norm,
                                      ds4_gpu_tensor *split, const ds4_gpu_tensor *stream,
                                      const ds4_gpu_tensor *pre, const void *model_map, uint64_t model_size,
                                      uint64_t fn_offset, uint64_t scale_offset, uint64_t base_offset,
                                      uint64_t norm_offset, uint32_t n, uint32_t mix_dim, uint32_t n_embd,
                                      uint32_t n_hc, uint32_t sinkhorn_iters, float hc_eps, float norm_eps,
                                      uint32_t rows, uint32_t pre_stride);
/* One block's input: split, weighted sum with `pre`'s first n_hc weights, norm. */
int ds4_gpu_dsv41_hc_input(ds4_gpu_tensor *x, ds4_gpu_tensor *norm, ds4_gpu_tensor *split,
                           const ds4_gpu_tensor *mix, const ds4_gpu_tensor *residual,
                           const ds4_gpu_tensor *pre, const void *model_map, uint64_t model_size,
                           uint64_t scale_offset, uint64_t base_offset, uint64_t norm_offset,
                           uint32_t n_embd, uint32_t n_hc, uint32_t sinkhorn_iters,
                           float hc_eps, float norm_eps);
/* Two Q8_0 projections of one row, outputs rounded to bf16 (0: run them apart). */
int ds4_gpu_dsv41_project_pair_q8(ds4_gpu_tensor *out_a, ds4_gpu_tensor *out_b,
                                  const void *model_map, uint64_t model_size,
                                  uint64_t offset_a, uint64_t offset_b, uint32_t in_dim,
                                  uint32_t out_a_dim, uint32_t out_b_dim, const ds4_gpu_tensor *x);
/* The shared expert's SwiGLU input from the Q8_0 gate and up weights, as the
 * bf16 matvecs and the SwiGLU pass would produce it; gate/up are scratch. */
int ds4_gpu_dsv41_shared_swiglu(ds4_gpu_tensor *mid, ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
                                const void *model_map, uint64_t model_size, uint64_t gate_offset,
                                uint64_t up_offset, uint32_t in_dim, uint32_t out_dim,
                                const ds4_gpu_tensor *x, float clamp, uint32_t rows);
/* Rope, quantize and store one row into dst at dst_offset bytes; x may be
 * left roped and quantized in place. */
/* The same for `rows` rows at positions start + r: with `ring` row r lands in
 * slot (start + r) % ring of dst, else in row r. */
int ds4_gpu_dsv41_rope_quantize(const ds4_gpu_tensor *x, ds4_gpu_tensor *dst, uint64_t dst_offset,
                                uint32_t width, uint32_t start, bool compressed,
                                ds4_v41_activation_format format);
/* One row's F16 projection rounded to bf16 in the kernel (0: not fused). */
int ds4_gpu_dsv41_project_f16_bf16(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                   uint64_t weight_offset, uint32_t in_dim, uint32_t out_dim,
                                   const ds4_gpu_tensor *x);
/* The same single-token dispatch with the weight read as E4M3 codes
 * [out_dim][in_dim] and E8M0 32x32 scales from fp8_map. */
int ds4_gpu_dsv41_project_fp8_bf16(ds4_gpu_tensor *out, const void *fp8_map, uint64_t fp8_size,
                                   uint64_t codes_off, uint64_t scales_off, uint32_t in_dim, uint32_t out_dim,
                                   const ds4_gpu_tensor *x);
/* Blend-kernel dispatches so far, by decode reader family: router, Engram, indexer q_b (Metal only). */
void ds4_gpu_dsv41_blend_counts(uint64_t out[3]);
/* While on, blend dispatches are neither counted nor announced (load-time warm-up dispatches). */
void ds4_gpu_dsv41_blend_quiet(int on);
/* The same with the codes and E8M0 bytes in tensors (copies built at load). */
int ds4_gpu_dsv41_project_fp8_bf16_weights(ds4_gpu_tensor *out, const ds4_gpu_tensor *codes, const ds4_gpu_tensor *e8m0,
                                           uint32_t in_dim, uint32_t out_dim, const ds4_gpu_tensor *x);
/* Decode attention over the raw ring and the compressed rows selected by
 * ids, staged in one pass (0: not available; `selected` is gather scratch). */
/* The same for `rows` consecutive query rows in one dispatch: row r uses the
 * raw window from raw_start + r and the ids at r * ids_stride. */
int ds4_gpu_dsv41_attention_decode_rows(ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size,
                                        uint64_t sinks_offset, const ds4_gpu_tensor *q,
                                        const ds4_gpu_tensor *raw_kv, uint32_t n_raw, uint32_t raw_cap,
                                        uint32_t raw_start, const ds4_gpu_tensor *comp_cache,
                                        const ds4_gpu_tensor *ids, uint32_t ids_stride, uint32_t n_comp,
                                        uint32_t attended, uint32_t n_head, uint32_t head_dim, uint32_t rows);
int ds4_gpu_dsv41_attention_decode(ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size,
                                   uint64_t sinks_offset, const ds4_gpu_tensor *q,
                                   const ds4_gpu_tensor *raw_kv, uint32_t n_raw, uint32_t raw_cap,
                                   uint32_t raw_start, const ds4_gpu_tensor *comp_cache,
                                   const ds4_gpu_tensor *ids, ds4_gpu_tensor *selected,
                                   uint32_t n_comp, uint32_t attended, uint32_t n_head, uint32_t head_dim);
/* The Q8_0 q projection of one row with the layer's rope on its outputs. */
/* Metal scalar scheduling: collapse now, mix beside the independent query. */
int ds4_gpu_dsv41_deferred_collapse(ds4_gpu_tensor *x, ds4_gpu_tensor *norm,
        const ds4_gpu_tensor *residual, const ds4_gpu_tensor *pre,
        const void *model_map, uint64_t model_size, uint64_t norm_offset, float eps);
int ds4_gpu_dsv41_query_deferred_hc(ds4_gpu_tensor *q, ds4_gpu_tensor *mix, ds4_gpu_tensor *split,
        const ds4_gpu_tensor *qr, const ds4_gpu_tensor *residual,
        const void *model_map, uint64_t model_size, uint64_t q_offset, uint64_t fn_offset,
        uint64_t scale_offset, uint64_t base_offset, uint32_t pos, bool compressed,
        uint32_t iters, float hc_eps, float norm_eps, ds4_gpu_tensor *x, ds4_gpu_tensor *norm,
        const ds4_gpu_tensor *pre, uint64_t norm_offset,
        const ds4_gpu_tensor *kv, ds4_gpu_tensor *cache, uint64_t cache_offset);

int ds4_gpu_dsv41_project_q(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                            uint64_t weight_offset, uint32_t in_dim, uint32_t out_dim,
                            const ds4_gpu_tensor *x, uint32_t pos, bool compressed);
/* A bf16 Q8_0 matvec of one row expanded straight into the n_hc residual
 * streams; `add` (optional) joins the row before the expand. */
int ds4_gpu_dsv41_matmul_expand(ds4_gpu_tensor *out_hc, const void *model_map, uint64_t model_size,
                                uint64_t weight_offset, uint32_t in_dim, uint32_t out_dim,
                                const ds4_gpu_tensor *x, const ds4_gpu_tensor *add,
                                const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split,
                                uint32_t n_hc);
int ds4_gpu_dsv41_matmul_expand_rows(ds4_gpu_tensor *out_hc, const void *model_map, uint64_t model_size,
                                     uint64_t weight_offset, uint32_t in_dim, uint32_t out_dim,
                                     const ds4_gpu_tensor *x, const ds4_gpu_tensor *add,
                                     const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split,
                                     uint32_t n_hc, uint32_t rows);
/* The attention output low projection of one row with the heads' inverse
 * rope (and bf16 rounding) folded into the load; heads may be roped in place. */
int ds4_gpu_dsv41_attention_low(ds4_gpu_tensor *low, const void *model_map, uint64_t model_size,
                                uint64_t out_a_offset, uint32_t group_dim, uint32_t rank,
                                uint32_t n_groups, ds4_gpu_tensor *heads, uint32_t pos, bool compressed);
/* The per-step RoPE coefficient table (64 float2 = 512 B: entries 0..31 the raw frequencies, 32..63 the compressed
 * ones) holds (cos, sin) of theta = float(pos) * frequency, the low kernel's own expression. attention_low_ex with
 * rope_tab reads it (bound at byte 0 raw or 256 compressed) instead of computing them: the same bits; NULL = the
 * inline kernel. */
int ds4_gpu_dsv41_rope_coeff_table(ds4_gpu_tensor *tab, uint32_t pos);
/* Metal: rope + quantization of `heads` rows of `width`, in place, in ONE dispatch with the coefficients from that
 * step's table (bound at byte 0 raw / 256 compressed); 0: not dispatched (the caller runs the separate rope and
 * quantize). The count of such dispatches. */
int ds4_gpu_dsv41_rope_quantize_heads_tab(ds4_gpu_tensor *x, uint32_t width, uint32_t heads, uint32_t start,
                                          bool compressed, ds4_v41_activation_format format, const ds4_gpu_tensor *tab);
uint64_t ds4_gpu_irq_count(void);
int ds4_gpu_dsv41_attention_low_ex(ds4_gpu_tensor *low, const void *model_map, uint64_t model_size,
                                   uint64_t out_a_offset, uint32_t group_dim, uint32_t rank,
                                   uint32_t n_groups, ds4_gpu_tensor *heads, uint32_t pos, bool compressed,
                                   const ds4_gpu_tensor *rope_tab);
/* Heads transformed once (Metal only). arm(tab, compressed) makes the next single-row decode attention reduce of the
 * V4.1 shape (64 heads x 512) write the heads with the inverse RoPE and bf16 rounding applied, reading the table above;
 * take() returns 1 when that reduce ran (and clears the arm). The low projection of those heads is then the plain
 * matvec: attention_low_plain. */
void ds4_gpu_dsv41_heads_once_arm(const ds4_gpu_tensor *tab, bool compressed);
int ds4_gpu_dsv41_heads_once_take(void);
int ds4_gpu_dsv41_attention_low_plain(ds4_gpu_tensor *low, const void *model_map, uint64_t model_size,
                                      uint64_t out_a_offset, uint32_t group_dim, uint32_t rank,
                                      uint32_t n_groups, const ds4_gpu_tensor *heads);
/* Test entry: the decode reduce of nrows heads (512 wide, split 32) from tmp's partials, plain or transformed. */
int ds4_gpu_dsv41_flash_attn_reduce_check(ds4_gpu_tensor *heads, const ds4_gpu_tensor *tmp, uint32_t nrows,
                                          int transform, const ds4_gpu_tensor *tab, bool compressed);
/* Two weighted RMS norms rounded to bf16 in one dispatch. */
int ds4_gpu_dsv41_norm_pair_rows(ds4_gpu_tensor *out0, const ds4_gpu_tensor *x0, uint64_t weight0_offset, uint32_t n0,
                                 ds4_gpu_tensor *out1, const ds4_gpu_tensor *x1, uint64_t weight1_offset, uint32_t n1,
                                 const void *model_map, uint64_t model_size, float eps, uint32_t rows);
int ds4_gpu_dsv41_norm_pair(ds4_gpu_tensor *out0, const ds4_gpu_tensor *x0, uint64_t weight0_offset, uint32_t n0,
                            ds4_gpu_tensor *out1, const ds4_gpu_tensor *x1, uint64_t weight1_offset, uint32_t n1,
                            const void *model_map, uint64_t model_size, float eps);
int ds4_gpu_dsv41_gather_kv(ds4_gpu_tensor *out, const ds4_gpu_tensor *source,
                           const ds4_gpu_tensor *ids, uint32_t source_rows,
                           uint32_t selected_rows);

/* Large-prefill producers. Short-row fused paths remain unchanged. */
int ds4_gpu_dsv41_prefill_folds_available(void);
int ds4_gpu_dsv41_add_bf16_rows(ds4_gpu_tensor *out,
        const ds4_gpu_tensor *a, const ds4_gpu_tensor *b, uint32_t width, uint32_t rows);
int ds4_gpu_dsv41_embed_init_rows(ds4_gpu_tensor *out_hc, ds4_gpu_tensor *pre,
        ds4_gpu_tensor *scratch, const ds4_gpu_tensor *tokens,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t n_vocab, uint32_t rows, uint32_t width);
int ds4_gpu_dsv41_projection_rows2(ds4_gpu_tensor *out,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t width, uint32_t outputs, uint32_t rows, const ds4_gpu_tensor *in);
int ds4_gpu_dsv41_norm_bf16_rows(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const void *map, uint64_t size, uint64_t offset, uint32_t width,
        uint32_t rows, float eps);

/* Complete post-Markov rows for public admission; arithmetic is unchanged. */
int ds4_gpu_dsv41_markov_chain_post(uint32_t block, uint32_t vocab, uint32_t rank, uint32_t dim,
                              const ds4_gpu_tensor *logits, const ds4_gpu_tensor *x,
                              const void *model_map, uint64_t model_size,
                              uint64_t embed_offset, uint64_t head_offset, int f16,
                              const ds4_gpu_tensor *conf_proj, ds4_gpu_tensor *tokens,
                              ds4_gpu_tensor *conf, ds4_gpu_tensor *parts, uint32_t n_parts,
                              ds4_gpu_tensor *post_logits);

#ifdef __cplusplus
}
#endif
#endif
