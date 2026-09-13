#ifndef DS4_H
#define DS4_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "ds4_ssd.h"

/* Public engine boundary.
 *
 * The CLI and server should treat ds4_engine as the loaded model and
 * ds4_session as one mutable inference timeline.  A session owns the live KV
 * cache and logits; callers provide full token prefixes and let
 * ds4_session_sync() reuse, extend, or rebuild the graph state.  Keep this
 * header narrow so HTTP/CLI code does not depend on tensor internals. */

typedef enum {
    DS4_BACKEND_METAL,
    DS4_BACKEND_CUDA,
    DS4_BACKEND_CPU,
} ds4_backend;

typedef enum {
    DS4_THINK_NONE,
    DS4_THINK_HIGH,
    DS4_THINK_MAX,
} ds4_think_mode;
/* Explicit numeric effort lives outside the stable named-mode values. */
#define DS4_THINK_LEVEL_BASE 1000

typedef enum {
    DS4_LOG_DEFAULT,
    DS4_LOG_PREFILL,
    DS4_LOG_GENERATION,
    DS4_LOG_KVCACHE,
    DS4_LOG_TOOL,
    DS4_LOG_WARNING,
    DS4_LOG_TIMING,
    DS4_LOG_OK,
    DS4_LOG_ERROR,
} ds4_log_type;

typedef struct {
    int *v;
    int len;
    int cap;
} ds4_tokens;

typedef struct {
    int id;
    float logit;
    float logprob;
} ds4_token_score;

#define DS4_DEFAULT_TEMPERATURE 1.0f
#define DS4_DEFAULT_TOP_P 1.0f
#define DS4_DEFAULT_MIN_P 0.05f

typedef struct ds4_engine ds4_engine;
typedef struct ds4_session ds4_session;

typedef void (*ds4_session_progress_fn)(void *ud, const char *event, int current, int total);
typedef bool (*ds4_session_cancel_fn)(void *ud);

#define DS4_SESSION_SYNC_INTERRUPTED 2

typedef enum {
    DS4_DISTRIBUTED_NONE = 0,
    DS4_DISTRIBUTED_COORDINATOR,
    DS4_DISTRIBUTED_WORKER,
} ds4_distributed_role;

typedef struct {
    uint32_t start;
    uint32_t end;
    bool has_output;
    bool set;
} ds4_distributed_layers;

typedef struct {
    ds4_distributed_role role;
    ds4_distributed_layers layers;
    const char *listen_host;
    int listen_port;
    const char *coordinator_host;
    int coordinator_port;
    uint32_t prefill_chunk;
    uint32_t prefill_window;
    uint32_t activation_bits;
    bool replay_check;
    bool debug;
} ds4_distributed_options;

/* Tensor parallelism: two identical machines run the model in lockstep and
 * split the heavy per-layer matvecs, exchanging partial sums at gates inside
 * the graph (see misc/METAL_TENSOR_PARALLELISM.md).  Each rank keeps one
 * contiguous half of the routed experts resident; dense and shared weights
 * remain replicated.  The leader owns prompt/sampling and listens; the worker
 * dials in and mirrors every session sync/eval. */
typedef enum {
    DS4_TP_NONE = 0,
    DS4_TP_LEADER,
    DS4_TP_WORKER,
} ds4_tp_role;

typedef enum {
    DS4_TP_TRANSPORT_AUTO = 0,
    DS4_TP_TRANSPORT_RDMA,
    DS4_TP_TRANSPORT_TCP,
} ds4_tp_transport;

typedef struct {
    ds4_tp_role role;
    bool requested;             /* --tensor-parallel with shared role options */
    const char *listen_host;    /* leader listens here for the worker */
    int listen_port;
    const char *leader_host;    /* worker dials the leader */
    int leader_port;
    ds4_tp_transport transport;
    const char *rdma_device;
    int rdma_gid_index;
    bool rdma_gid_index_set;
    bool glm_token_prefill;
    int debug_hash;             /* cross-check hidden state every N tokens */
} ds4_tp_options;

typedef struct {
    const char *model_path;
    const char *mtp_path;
    const char *vision_path;
    ds4_backend backend;
    int n_threads;
    int context_size;
    uint32_t prefill_chunk;
    int mtp_draft_tokens;
    float mtp_margin;
    float dspark_confidence_threshold;
    const char *directional_steering_file;
    const char *expert_profile_path;
    float directional_steering_attn;
    float directional_steering_ffn;
    int power_percent;
    uint32_t ssd_streaming_cache_experts;
    uint64_t ssd_streaming_cache_bytes;
    uint32_t ssd_streaming_full_layers;
    uint32_t ssd_streaming_preload_experts;
    uint64_t simulate_used_memory_bytes;
    bool warm_weights;
    bool quality;
    bool glm_mtp;
    bool glm_mtp_timing;
    bool dspark;
    bool dspark_strict;
    bool dspark_exact_sampling;
    bool dspark_confidence_threshold_set;
    bool cuda_tensor_parallel;
    bool ssd_streaming;
    bool ssd_streaming_cold;
    bool ssd_streaming_full_layers_set;
    bool inspect_only;
    /* Multi-GPU placement uses this to price per-layer KV storage. */
    int placement_ctx_hint;
    /* Number of independently allocated session graphs/caches to reserve. */
    int placement_session_count_hint;
    /* Server batch mode serializes execution and can share prefill scratch. */
    bool share_session_prefill_workspace;
    bool first_token_test;
    bool metal_graph_test;
    bool load_slice;
    uint32_t load_layer_start;
    uint32_t load_layer_end;
    bool load_output;
    ds4_distributed_options distributed;
    ds4_tp_options tp;
} ds4_engine_options;

/* ---------------------------------------------------------------------------
 * DeepSeek V4.1 decode campaign levers.
 *
 * One struct holding every kill switch the campaign toggles, so that a resident
 * server can flip them between requests instead of an A/B costing a full weight
 * load and prefill.  Every field is 1 for the production default.  Historical
 * environment variables still work and still mean exactly what they meant:
 *
 *   queue_layers         DS4_DS41_QUEUE_LAYERS=0 disables          (L1)
 *   engram_async         DS4_DS41_ENGRAM_ASYNC=0 disables          (L2)
 *   round_fuse_norm      DS4_METAL_DISABLE_V41_ROUND_FUSE_NORM=1 disables     (L3/D1)
 *   round_fuse_hcsum     DS4_METAL_DISABLE_V41_ROUND_FUSE_HCSUM=1 disables    (L3/D1)
 *   round_fuse_hcexpand  DS4_METAL_DISABLE_V41_ROUND_FUSE_HCEXPAND=1 disables (L3/D1)
 *
 * Opt-in levers (default 0, the environment variable enables):
 *
 *   q4_grouped   DS4_METAL_ENABLE_Q4_GROUPED_EXPERTS        (L3/W1)
 *   q4_group8    DS4_METAL_ENABLE_Q4_GROUP8_EXPERT_TABLE    (L3/W1)
 *   q4_group24   DS4_METAL_ENABLE_Q4_GROUP24_EXPERT_TABLE   (L3/W1)
 *
 * and one default-on:
 *
 *   q4_group6    DS4_METAL_DISABLE_Q4_GROUP6_EXPERT_TABLE=1 disables (L3/W1)
 *   q4_wide      DS4_DS41_Q4_WIDE=0 disables                          (L3/W2a)
 *   q4_gu_nr1    DS4_DS41_Q4_GU_NR1 enables (default off)             (L3/W2b)
 *   router_fused DS4_DS41_ROUTER_FUSED=0 disables                     (L3/W2/D2)
 *   router_fused_w DS4_DS41_ROUTER_FUSED_W=0 keeps the 5-dispatch tail (L3/W2/D2)
 *   shared_swiglu DS4_DS41_SHARED_SWIGLU=0 disables                   (L3/R9)
 *   hc_norm_mix  DS4_DS41_HC_NORM_MIX=0 disables                     (L3/R1a)
 *   hc_tail      DS4_DS41_HC_TAIL=0 disables                          (L3/R1a)
 *   hc_expand_fold DS4_DS41_HC_EXPAND_FOLD=0 disables                 (L3/R1b)
 *   mv_round     DS4_DS41_MV_ROUND=0 disables                         (L3/R2)
 *   producer_round DS4_DS41_PRODUCER_ROUND=0 disables                  (L3/R8')
 *   ffn_add_fold DS4_DS41_FFN_ADD_FOLD=0 disables                      (L3/R8')
 *   kv_stage_f16 DS4_DS41_KV_STAGE_F16=0 disables                      (L3/R3a)
 *
 * Prefill cost-map levers (prefill/PLAN.md Phase 1; the whole cost map runs
 * inside ONE resident server instead of ~8 process starts):
 *
 *   prefill_8k_chunk       DS4_METAL_DISABLE_V41_8K_CHUNK=1 disables
 *   prefill_decoder_suffix DS4_METAL_DISABLE_V41_DECODER_SUFFIX=1 disables
 *   prefill_stage_profile  DS4_METAL_V41_STAGE_PROFILE=1 enables (default off)
 *   q8_prefill_profile     DS4_METAL_Q8_PREFILL_PROFILE=1 enables (default off)
 *
 * The graph reads g_ds41_levers with plain global loads.  ds41_levers_set() is
 * only ever called between requests, by ds4-server's --debug-levers endpoint.
 * ------------------------------------------------------------------------ */
typedef struct {
    int decode_chunks; /* DS4_DS41_DECODE_CHUNKS=0 restores one-stack-CB path */
    int queue_layers;
    int engram_async;
    int round_fuse_norm;
    int round_fuse_hcsum;
    int round_fuse_hcexpand;
    int q4_grouped;
    int q4_group6;
    int q4_group8;
    int q4_group24;
    int q4_wide;
    int q4_gu_nr1;
    int router_fused;
    int router_fused_w;
    int shared_swiglu;
    int hc_norm_mix;
    int ffn_producer; /* DS4_DS41_FFN_PRODUCER=0 restores separate HC expansion */
    int qa_kv_flat; /* DS4_DS41_QA_KV_FLAT=0 restores separate projections */
    int hc_stream; /* DS4_DS41_HC_STREAM=0 restores pre-wave-A sequence */
    int hc_tail;
    int hc_expand_fold;
    int mv_round;
    int producer_round;
    int ffn_add_fold;
    int kv_stage_f16;
    int routed_down_split; /* DS4_DS41_ROUTED_DOWN_SPLIT=1 arms R4 (opt-in) */
    /* Prefill cost map.  prefill_8k_chunk moves the ENCODER CHUNK only: the
     * graph's batch buffers are sized from the startup value of the same
     * switch, so flipping it at runtime shrinks the chunk without shrinking
     * the allocation (see ds41_encoder_chunk_cap). */
    int prefill_8k_chunk;
    int prefill_decoder_suffix;
    int prefill_stage_profile;  /* default off */
    int q8_prefill_profile;     /* default off */
    int attn_cohort4;      /* DS4_DS41_ATTN_COHORT4=1 arms R6 (opt-in) */
    /* Decode pass 2, C1: elements-per-thread (in packed_float4 units) for the
     * selected-KV gather.  0 = the scalar kernel_get_rows_f32_f16 the wave
     * inherited; 1, 2 or 4 = the wide twin.  Bit-identical at every setting. */
    int gather_wide;
    /* Decode pass 2, C2: perform the selected-row gather inside the contiguous
     * KV staging dispatch.  DS4_DS41_STAGE_GATHER=0 restores the standalone
     * kernel_get_rows gather and its intermediate half buffer. */
    int stage_gather;
    /* Decode pass 2, D1: routed DOWN leg at NR0 = 1 (2,560 threadgroups).
     * DS4_DS41_Q4_DN_NR1=0 restores the two-rows-per-simdgroup grid. */
    int q4_dn_nr1;
    /* Decode pass 2, D2: threadgroup width of kernel_dsv41_hc_round_expand4.
     * 256 (the inherited value) puts 5,120 threads on 20 of this GPU's 80
     * cores; 32/64/128 spread the same threads wider.  Elementwise, so every
     * width is bit-identical. */
    int hc_expand_nth;
    /* Prefill wave 1, bundle item (i): CULL_TAIL_SIMDGROUPS for the Q4_K
     * routed matmuls.  Production default; DS4_METAL_DISABLE_V41_ROUTED_TAIL_CULL=1
     * restores the uncculled kernels. */
    int routed_tail_cull;
    /* Prefill wave 2, lever 1: absorb a sub-2048 remainder into the sweep that
     * precedes it instead of leaving a separate full-depth tail sweep.
     * 62,000 tokens split 32,768 + 28,672 + 560 today; the 560-row sweep
     * re-reads all 40 layers and costs 4.5 % against the prompt's own 32k
     * trend (prefill/phase1/COSTMAP.md S1).  Host-only; no per-row arithmetic
     * changes, but the decoder suffix then approximates over different rows and
     * the frontier logits move -- Tier 2, opt-in with DS4_DS41_PREFILL_TAIL_ABSORB=1
     * until the four-manifest gate passes. */
    int prefill_tail_absorb;
    /* Prefill wave 2, lever 2: concurrent readers in ds4_engram_read_batch.
     * 16 is the historical value; the SSD saturates at 32 (ds4_engram.c).
     * Timing only -- same rows, same order, same values. */
    int engram_readers;
    /* Prefill wave 2, lever 3: stage sixteen K/V rows at a time in the prefill
     * attention core instead of one, by dispatching the existing
     * kernel_dsv4_indexed_mixed_attention_heads8_rb16 (the decode path's
     * production kernel) for multi-token chunks.  Same rows, same order, same
     * online-softmax updates, same dot and reduction trees -- only the gather
     * width and the threadgroup-barrier count change.
     * DS4_METAL_DISABLE_V41_PREFILL_ATTN_RB16=1 restores the one-row kernel. */
    int prefill_attn_rb16;
    int prefill_hc_sum_round; /* wave 3, ADOPTED: default 1, =0 is the kill switch */
    int prefill_hc_expand_round; /* wave 3, ADOPTED: default 1, =0 is the kill switch */
    int prefill_hc_norm_round; /* wave 3, ADOPTED: default 1, =0 is the kill switch */
    int prefill_ffn_add_round; /* wave 3, ADOPTED: default 1, =0 is the kill switch */
    int prefill_embed_init; /* wave 3, ADOPTED: default 1, DS4_DS41_PREFILL_EMBED_INIT=0 kills it */
    int mtp_qa_kv_flat; /* F3: paired q_a/KV banks in one dispatch. */
    int mtp_async_chunks; /* F2: submit every four verifier layers. */
    int mtp_hc_mixed; /* F1: private HC rows mixed with paired Q8 projections. */
    int mtp_hc_rows2; /* Private HC cohorts for two-row verification. */
    int mtp_engram_rows6; /* Six private asynchronous Engram rows. */
    int mtp_engram_rows2; /* Private asynchronous two-row Engram inputs. */
    int dspark_excl_eos; /* Match the serial control's argmax_excluding(eos) rule. */
    int mtp_capture_warmup; /* Complete target capture from decoder warmup inputs. */
    int mtp_state_fix; /* Capture undo, truthful snapshots, seed generation. */
    int mtp_gu_union2; /* Two-row routed gate/up only; down remains ordered. */
    int mtp_q8_pair6; /* Six rows as three independent weight-sharing pairs. */
    int mtp_q8_rows2; /* Opt-in; exactly two target rows. */
    int prefill_f16_rows2; /* wave 3, ADOPTED: default 1, DS4_DS41_PREFILL_F16_ROWS2=0 kills it */
    /* Wave 3, task B.  DIAGNOSTIC ONLY, never adopted: an ablation arm of the
     * prefill attention core.  0 = production kernel; 1..5 select a variant
     * that deletes one stage (pv, exp, gather) or keeps only the gather, plus
     * an unablated control that shares the arms' clamped row scan.  Every
     * non-zero value makes the attention output deliberately WRONG, so no
     * speed or identity claim may ever be made from a run with it set.
     * DS4_DS41_ATTN_DIAG=N, counted lever, per request. */
    int attn_diag;
    /* Wave 3: the register-lean selected-id form of the prefill attention
     * core.  The wave-3 decomposition measured the kernel to be
     * register/occupancy bound; deleting the sixteen-word `uint rows[16]`
     * private array while keeping every row, every order and every arithmetic
     * operation took the 8,192-row call 100.26 -> 84.13 ms.  Same rows, same
     * order, same online-softmax updates -- a sixteen-bit mask replaces the
     * array.  DS4_METAL_DISABLE_V41_PREFILL_ATTN_LEAN_ROWS=1 is the kill
     * switch.  Prefill only: the decode path keeps the wave-2 kernel until
     * the decode gate has seen this. */
    int prefill_attn_lean_rows;
    /* Wave 4, lever 1: bounded cross-sweep Engram prefetch.  A 62,000-token
     * prompt runs three sweeps; wave 3's stage profile shows the layer-1
     * Engram wait exposed 639.0 ms in sweep 2 and 64.7 + 65.2 ms in the 560-row
     * sweep 3, because each sweep starts its module-0 disk delivery only at its
     * own layer 0.  With this on, the tail of a sweep (after its module-1 rows
     * have been consumed and the reader joined at layer 14) starts the NEXT
     * sweep's delivery into the already-owned prefetch slab, using a PRIVATE
     * copy of the hash history seeded from the current sweep's end and a
     * PRIVATE id buffer, so the live history is not advanced early and the
     * reader never shares the id array the next sweep recomputes.  The next
     * sweep adopts the carried rows only after byte-comparing its own freshly
     * hashed ids against the ids the reader actually used; any mismatch joins
     * the carried reader and falls back to the ordinary in-sweep read.  Same
     * rows, same values, same order -- Tier 1 by construction.
     * DS4_DS41_PREFILL_ENGRAM_XSWEEP=0 is the kill switch. */
    int prefill_engram_xsweep;
    /* Wave 4, lever 2: attention-core geometry on the landed lean kernel.
     * 0 = production (8 heads per threadgroup, 16 staged K/V rows, 16 KiB).
     * Every non-zero value selects a different (cohort, staged rows) pair of
     * the SAME kernel: each head still runs its own rows in the same order
     * through the same dot, reduction and online-softmax trees, so the output
     * is byte-identical and the arm is Tier 1 by construction.  Only how many
     * heads share a threadgroup, how many rows are staged per barrier and how
     * much threadgroup memory the dispatch asks for change.  Counted lever,
     * per request: DS4_DS41_ATTN_GEOM=N. */
    int attn_geom;
    /* Stage 5.  dspark_capture = 0 suppresses the target-hidden capture on an
     * armed session, which is the identity A/B for it; verify_wide_prefill
     * relaxes the five count > DS4_TP_BATCH_MAX_ROWS prefill gates so a 6-row
     * verify pass can reach them (VERIFY-COST.md section 5). */
    int dspark_capture;
    int verify_wide_prefill;
    /* dspark_verify_rows = 0 verifies the whole block (1 committed + 5 draft
     * rows); 2..6 selects a prefix of the proposal. 1 and out-of-range
     * widths are invalid; use serial generation for the one-row control.
     * dspark_expert_union = 1 reads the routed selection back per layer. */
    int dspark_verify_rows;
    int dspark_expert_union;
    /* verify_batch_core = 1 runs the verify pass's attention through the
     * prefill batch core instead of six per-row decode passes: faster, but a
     * different attention kernel and therefore not byte-identical to serial.
     * dspark_force_reject = 1 rejects every draft, which commits one token per
     * cycle and exercises the maximal rollback on every cycle. */
    int verify_batch_core;
    int dspark_force_reject;
    /* dspark_draft_trace = 1 prints one line per cycle naming the drafter's
     * block-5 proposal and every verified row's argmax, so per-position
     * agreement can be read paired against the target's own next tokens
     * (ACCEPTANCE.md).  Diagnostic only: it runs `rows` extra host argmaxes
     * and writes to stderr, outside the phase timers. */
    int dspark_draft_trace;
} ds41_levers;

extern ds41_levers g_ds41_levers;

void        ds41_levers_init_from_env(void);
size_t      ds41_levers_count(void);
const char *ds41_levers_name(size_t i);
const char *ds41_levers_env_name(size_t i);
int         ds41_levers_get(const char *name, int *out);
int         ds41_levers_set(const char *name, int value);

/* DS4_KERNEL_LEDGER, per request.  The MODE is process-level: it installs
 * Objective-C swizzles and (mode 2) a counter sample buffer before any
 * pipeline is built, and mode 2 additionally disables encoder batching, so it
 * cannot be changed after ds4_gpu_init().  Reset and dump, however, are free
 * to run per request, which is what lets one resident server produce a
 * separate prefill ledger for every arm of the cost map.
 *   ds4_gpu_kernel_ledger_mode()      0 when the ledger is off.
 *   ds4_gpu_kernel_ledger_reset()     zero every accumulator, keep the
 *                                     pipeline-object -> kernel-name map.
 *   ds4_gpu_kernel_ledger_dump_path() write the ledger to a path (NULL or ""
 *                                     = the DS4_KERNEL_LEDGER_DUMP default);
 *                                     1 on success, 0 when off or unwritable.
 * All three are no-ops returning 0 on non-Metal builds. */
int  ds4_gpu_kernel_ledger_mode(void);
void ds4_gpu_kernel_ledger_reset(void);
int  ds4_gpu_kernel_ledger_dump_path(const char *path);

typedef struct {
    float *data;
    uint32_t token_count;
    uint32_t layout;
    uint32_t grid_width;
    uint32_t grid_height;
    uint32_t width;
    uint32_t height;
    uint32_t content_width;
    uint32_t content_height;
    uint8_t fingerprint[32];
} ds4_vision_embedding;

typedef struct {
    uint32_t token_start;
    ds4_vision_embedding embedding;
} ds4_vision_span;

typedef void (*ds4_token_emit_fn)(void *ud, int token);
typedef void (*ds4_generation_done_fn)(void *ud);

typedef struct {
    uint64_t total_bytes;
    uint64_t raw_bytes;
    uint64_t compressed_bytes;
    uint64_t scratch_bytes;
    uint32_t prefill_cap;
    uint32_t raw_cap;
    uint32_t comp_cap;
} ds4_context_memory;

typedef struct {
    uint8_t *ptr;
    uint64_t len;
    uint64_t cap;
} ds4_session_snapshot;

typedef struct {
    char *path;
    uint64_t bytes;
} ds4_session_payload_file;

int ds4_engine_open(ds4_engine **out, const ds4_engine_options *opt);

/* Multi-GPU pipeline-parallel entry point (wave 2).
 *
 * Accepts an optional ds4_gpu_config (defined in ds4_gpu_mgpu.h) that
 * lets callers describe a multi-GPU placement target. Passing NULL is
 * back-compatible with ds4_engine_open and produces identical engine
 * state — bit-equivalent execution at runtime.
 *
 * When a non-NULL config is supplied AND the computed placement spans
 * more than one tier (either multiple GPUs or any CPU-spill), this
 * wave-2 implementation prints the layout and refuses to open: full
 * multi-tier execution wiring lands in a follow-up task
 * (mgpu-graph-session-execution). Callers receive a non-zero return
 * and a documented stderr notice. */
/* ds4_gpu_config is declared in ds4_gpu_mgpu.h, which callers should
 * include separately. We forward-declare it here so this header can be
 * used as-is (callers passing NULL don't need the struct definition). */
struct ds4_gpu_config;
int ds4_engine_create_with_gpu_config(ds4_engine **out,
                                       const ds4_engine_options *opt,
                                       const struct ds4_gpu_config *gpu_cfg);
void ds4_engine_close(ds4_engine *e);
void ds4_engine_summary(ds4_engine *e);
int ds4_engine_vocab_size(ds4_engine *e);
uint32_t ds4_engine_prefill_chunk(ds4_engine *e);
int ds4_engine_power(ds4_engine *e);
int ds4_engine_set_power(ds4_engine *e, int power_percent);
const char *ds4_engine_model_name(ds4_engine *e);
int ds4_engine_layer_count(ds4_engine *e);
/* Decode gate schedule for the TP transport; see ds4_tp_identity. */
enum { DS4_TP_GATE_MASK_WORDS = 3 };
void ds4_engine_tp_gate_schedule(ds4_engine *e,
                                 uint32_t *start,
                                 uint32_t *step,
                                 uint32_t *per_token,
                                 uint64_t mask[DS4_TP_GATE_MASK_WORDS]);
uint32_t ds4_engine_layer_compress_ratio(ds4_engine *e, uint32_t layer);
uint64_t ds4_engine_hidden_f32_values(ds4_engine *e);
int ds4_engine_embd_dim(ds4_engine *e);
uint64_t ds4_engine_model_bytes(ds4_engine *e);
bool ds4_engine_has_vision(ds4_engine *e);
int ds4_engine_vision_encode_file(ds4_engine *e,
                                  const char *path,
                                  ds4_vision_embedding *out,
                                  char *error,
                                  size_t error_cap);
int ds4_engine_vision_encode_memory(ds4_engine *e,
                                    const uint8_t *encoded,
                                    size_t encoded_len,
                                    ds4_vision_embedding *out,
                                    char *error,
                                    size_t error_cap);
void ds4_vision_embedding_free(ds4_vision_embedding *embedding);
int ds4_prompt_append_vision(ds4_engine *e,
                             ds4_tokens *tokens,
                             ds4_vision_span *span,
                             ds4_vision_embedding *embedding,
                             char *error,
                             size_t error_cap);
/* Append one user or tool message whose text parts alternate with images.
 * text_parts must contain image_count + 1 entries. On success ownership of
 * each embedding is transferred to the corresponding output span. */
int ds4_chat_append_multimodal_message(ds4_engine *e,
                                       ds4_tokens *tokens,
                                       const char *role,
                                       const char *const *text_parts,
                                       ds4_vision_embedding *embeddings,
                                       size_t image_count,
                                       ds4_vision_span *spans,
                                       char *error,
                                       size_t error_cap);
int ds4_engine_tp_vocab_split(ds4_engine *e);
bool ds4_engine_glm_layer_payload_bytes(ds4_engine *e,
                                        uint32_t layer,
                                        uint32_t full_live,
                                        uint32_t key_dim,
                                        uint32_t value_dim,
                                        uint32_t compact_live,
                                        uint32_t index_live,
                                        uint64_t *out);
/* Stable id for cache compatibility.  0 is the original Flash shape, so old
 * KV files with the previously-zero reserved byte remain Flash-compatible;
 * Pro and later shapes must use nonzero ids. */
int ds4_engine_model_id(ds4_engine *e);
bool ds4_engine_is_glm_dsa(ds4_engine *e);
bool ds4_engine_is_glm53(ds4_engine *e);
const char *ds4_backend_name(ds4_backend backend);
bool ds4_think_mode_enabled(ds4_think_mode mode);
int ds4_think_mode_level(ds4_think_mode mode);
bool ds4_think_mode_parse_level(const char *text, ds4_think_mode *out);
const char *ds4_think_mode_name(ds4_think_mode mode);
const char *ds4_think_max_prefix(void);
const char *ds4_glm_reasoning_effort_text(ds4_think_mode mode);
uint32_t ds4_think_max_min_context(void);
ds4_think_mode ds4_think_mode_for_context(ds4_think_mode mode, int ctx_size);
/* Uses the active model shape selected by ds4_engine_open(); call after opening
 * the GGUF so Flash/Pro dimensions are known. */
ds4_context_memory ds4_context_memory_estimate(ds4_backend backend, int ctx_size);
ds4_context_memory ds4_context_memory_estimate_with_prefill(
        ds4_backend backend,
        int ctx_size,
        uint32_t prefill_chunk);
ds4_context_memory ds4_context_memory_estimate_with_prefill_mode(
        ds4_backend backend,
        int ctx_size,
        uint32_t prefill_chunk,
        bool ssd_streaming);
bool ds4_log_is_tty(FILE *fp);
void ds4_log(FILE *fp, ds4_log_type type, const char *fmt, ...);
int ds4_engine_generate_argmax(ds4_engine *e, const ds4_tokens *prompt,
                               int n_predict, int ctx_size,
                               ds4_token_emit_fn emit,
                               ds4_generation_done_fn done,
                               void *emit_ud,
                               ds4_session_progress_fn progress,
                               void *progress_ud);
int ds4_engine_collect_imatrix(ds4_engine *e,
                               const char *dataset_path,
                               const char *output_path,
                               int ctx_size,
                               int max_prompts,
                               int max_tokens,
                               int min_expert_samples);
void ds4_engine_dump_tokens(ds4_engine *e, const ds4_tokens *tokens);
int ds4_dump_text_tokenization(const char *model_path, const char *text, FILE *fp);
int ds4_dump_chat_tokenization(const char *model_path,
                               const char *system,
                               const char *prompt,
                               ds4_think_mode think_mode,
                               int ctx_size,
                               FILE *fp);
int ds4_engine_head_test(ds4_engine *e, const ds4_tokens *prompt);
bool ds4_engine_is_glm_dsa(ds4_engine *e);
bool ds4_engine_is_deepseek41(ds4_engine *e);
const char *ds4_deepseek41_reasoning_effort_text(ds4_think_mode mode);
int ds4_engine_first_token_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_full_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_prompt_test(ds4_engine *e, const ds4_tokens *prompt, int ctx_size);

void ds4_tokens_push(ds4_tokens *tv, int token);
void ds4_tokens_free(ds4_tokens *tv);
void ds4_tokens_copy(ds4_tokens *dst, const ds4_tokens *src);
bool ds4_tokens_starts_with(const ds4_tokens *tokens, const ds4_tokens *prefix);

void ds4_tokenize_text(ds4_engine *e, const char *text, ds4_tokens *out);
void ds4_tokenize_rendered_chat(ds4_engine *e, const char *text, ds4_tokens *out);
void ds4_chat_begin(ds4_engine *e, ds4_tokens *tokens);
void ds4_encode_chat_prompt(
        ds4_engine *e,
        const char *system,
        const char *prompt,
        ds4_think_mode think_mode,
        ds4_tokens *out);
void ds4_chat_append_max_effort_prefix(ds4_engine *e, ds4_tokens *tokens);
void ds4_chat_append_think_prefix(ds4_engine *e, ds4_tokens *tokens, ds4_think_mode mode);
void ds4_chat_append_message(ds4_engine *e, ds4_tokens *tokens, const char *role, const char *content);
void ds4_chat_append_assistant_prefix(ds4_engine *e, ds4_tokens *tokens, ds4_think_mode think_mode);

char *ds4_token_text(ds4_engine *e, int token, size_t *len);
int ds4_token_eos(ds4_engine *e);
bool ds4_token_is_stop(ds4_engine *e, int token);
bool ds4_token_is_thinking_control(ds4_engine *e, int token);
bool ds4_token_is_stop_for_think_mode(ds4_engine *e,
                                      int token,
                                      ds4_think_mode mode);
int ds4_token_user(ds4_engine *e);
int ds4_token_assistant(ds4_engine *e);

/* Tensor-parallel binding: allocates the GPU gate slab, registers it with
 * the transport and arms the per-layer gate machinery.  Call once, after
 * ds4_tp_create() and before any session work.  Transport lifecycle stays
 * with the caller. */
struct ds4_tp;
int ds4_engine_tp_bind(ds4_engine *e, struct ds4_tp *tp, char *err, size_t errlen);

int ds4_session_create(ds4_session **out, ds4_engine *e, int ctx_size);
void ds4_session_free(ds4_session *s);
int ds4_session_power(ds4_session *s);
int ds4_session_set_power(ds4_session *s, int power_percent);
float ds4_session_directional_steering_ffn(ds4_session *s);
/* Change steering for future evaluation without rebuilding the existing KV
 * state. Live changes are currently limited to non-distributed sessions. */
int ds4_session_set_directional_steering_ffn(ds4_session *s, float scale);
bool ds4_session_is_distributed(ds4_session *s);
void ds4_session_set_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud);
/* UI-only progress. It may report fine-grained progress inside a prefill chunk;
 * callers must not treat it as a durable KV checkpoint boundary. */
void ds4_session_set_display_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud);
/* Cooperative cancellation for ds4_session_sync(), which drains pending work
 * before returning DS4_SESSION_SYNC_INTERRUPTED. A complete prefix may remain
 * usable, but a partial layer-major pass is invalidated and the next sync
 * rebuilds it. Do not assume an interrupted session can be saved or decoded. */
void ds4_session_set_cancel(ds4_session *s, ds4_session_cancel_fn fn, void *ud);
void ds4_session_report_progress(ds4_session *s, const char *event, int current, int total);
/* Distributed coordinator sessions return 1 when the full layer route is
 * available, 0 when it is still incomplete, and -1 for a local API error. */
int ds4_session_distributed_route_ready(ds4_session *s, char *err, size_t errlen);

typedef enum {
    DS4_SESSION_REWRITE_ERROR = -1,
    DS4_SESSION_REWRITE_OK = 0,
    /* The live backend state cannot be rewritten safely in place.  The caller should
     * restore an older checkpoint if it has one, then sync to the prompt. */
    DS4_SESSION_REWRITE_REBUILD_NEEDED = 1,
} ds4_session_rewrite_result;

/* Synchronize the live session to a full prompt token prefix.  If the current
 * checkpoint is a prefix, only the suffix is evaluated; otherwise the backend
 * state is refilled from scratch. */
#define DS4_SESSION_SYNC_INTERRUPTED 2
int ds4_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen);
int ds4_session_sync_multimodal(ds4_session *s,
                                const ds4_tokens *prompt,
                                const ds4_vision_span *images,
                                size_t image_count,
                                char *err,
                                size_t errlen);
/* A reusable image prefix has unchanged spans/fingerprints for all historical
 * images, with any new images starting at or after the live token frontier.
 * The caller must also check the token prefix. Invalid checkpoints never match. */
bool ds4_session_vision_prefix_matches(const ds4_session *s,
                                       const ds4_vision_span *images,
                                       size_t image_count);
/* Like the prefix check, but also require exactly the same image count. */
bool ds4_session_vision_state_matches(const ds4_session *s,
                                      const ds4_vision_span *images,
                                      size_t image_count);
/* Restore image positions from an independently authenticated live continuation
 * (for example, matching tool-call IDs). Checks every fingerprint and row count;
 * on failure, leaves spans unchanged. This does not verify the text history. */
bool ds4_session_rebase_vision_state(const ds4_session *s,
                                     ds4_vision_span *images, size_t image_count);
/* True while a session contains, or is actively syncing, image-conditioned
 * state. Such state must not be written to the text-keyed disk KV cache. */
bool ds4_session_has_vision_state(const ds4_session *s);
bool ds4_session_rewrite_requires_rebuild(int live_len, int canonical_len, int common);
ds4_session_rewrite_result ds4_session_rewrite_from_common(
        ds4_session *s, const ds4_tokens *prompt, int common,
        char *err, size_t errlen);
int ds4_session_common_prefix(ds4_session *s, const ds4_tokens *prompt);
int ds4_session_argmax(ds4_session *s);

/* V4.1 DSpark speculative decode (campaign Stage 5).  Greedy only: the emitted
 * continuation is identical to serial greedy by construction.  `out_tokens`
 * must hold at least `gen_tokens` ints; `eos_id` < 0 disables the stop check. */
typedef struct {
    uint32_t cycles;        /* draft + verify passes                       */
    uint32_t committed;     /* target rows committed (tokens per cycle)    */
    uint32_t verified_rows; /* rows evaluated by the verify passes         */
    uint32_t accept_hist[8];/* accepted drafts per cycle, 0..block         */
    double   propose_ms, verify_ms, commit_ms, total_ms;
    double   expert_union;  /* mean routed experts read per layer per pass */
    uint32_t union_layers;  /* layers sampled (0 unless DS4_DS41_EXPERT_UNION) */
} ds4_dspark_decode_stats;

int ds4_session_dspark_generate(ds4_session *s, int gen_tokens, int eos_id,
                                int *out_tokens, int *n_out,
                                ds4_dspark_decode_stats *st,
                                char *err, size_t errlen);
int ds4_session_argmax_excluding(ds4_session *s, int excluded_id);
int ds4_session_argmax_ignoring_eos(ds4_session *s,
                                    ds4_think_mode think_mode);
int ds4_sample_logits(const float *logits, int n_vocab, float temperature,
                      int top_k, float top_p, float min_p, uint64_t *rng);
int ds4_session_sample(ds4_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng);
#ifdef DS4_TEST_HOOKS
int ds4_test_sample_logits(const float *logits, uint32_t n_vocab,
                           float temperature, int top_k,
                           float top_p, float min_p, uint64_t *rng,
                           float *prob_scratch);
int ds4_test_sampling_probabilities(const float *logits, uint32_t n_vocab,
                                    float temperature, int top_k,
                                    float top_p, float min_p, float *probs);
int ds4_test_speculative_sample(const float *target_logits,
                                const float *draft_logits,
                                uint32_t n_vocab,
                                float temperature,
                                int top_k,
                                float top_p,
                                float min_p,
                                uint64_t *rng,
                                float *target_probs,
                                float *draft_probs);
int ds4_test_speculative_delta_sample(const float *target_logits,
                                      uint32_t n_vocab,
                                      int draft_token,
                                      float temperature,
                                      int top_k,
                                      float top_p,
                                      float min_p,
                                      uint64_t *rng,
                                      float *target_probs);
int ds4_test_argmax_excluding_logits(const float *logits, uint32_t n_vocab,
                                     int excluded_id);
uint64_t ds4_test_mixed_native_count(void);
#endif
int ds4_session_top_logprobs(ds4_session *s, ds4_token_score *out, int k);
int ds4_session_token_logprob(ds4_session *s, int token, ds4_token_score *out);
int ds4_session_copy_logits(ds4_session *s, float *out, int cap);
int ds4_session_set_logits(ds4_session *s, const float *logits, int n);
/* Pay the one-time first-submission GPU cost outside any measured window;
 * used by the TP worker right after session create (no-op on CPU/GLM). */
void ds4_session_gpu_warmup(ds4_session *s);
int ds4_session_eval(ds4_session *s, int token, char *err, size_t errlen);

typedef struct {
    ds4_session *session;
    int token;
} ds4_decode_item;

/* Advance independent sessions by one token each. Batch size one is exactly
 * ds4_session_eval(). Backends without native batching use a correctness-first
 * sequential fallback. */
int ds4_sessions_eval_batch(ds4_decode_item *items, int count,
                            char *err, size_t errlen);
/* Advance one resumed prefill suffix and an independent decode batch as one
 * scheduling step. Unsupported combinations use the ordinary serialized
 * session operations. */
int ds4_sessions_eval_batch_with_prefill(
        ds4_decode_item *items, int count,
        ds4_session *prefill_session, const ds4_tokens *prefill_prompt,
        char *err, size_t errlen);
int ds4_session_eval_speculative_argmax(ds4_session *s, int first_token,
                                        int max_tokens, int eos_token,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen);
int ds4_session_eval_speculative_argmax_ignoring_eos(
        ds4_session *s, int first_token, int max_tokens, int eos_token,
        ds4_think_mode think_mode,
        int *accepted, int accepted_cap, char *err, size_t errlen);
/* Evaluate one already-sampled target token and speculatively extend it.
 * Positive-temperature DSpark normally commits greedily verified draft
 * tokens; dspark_exact_sampling selects exact stochastic p/q acceptance for
 * DSpark or an internal GLM MTP block. */
int ds4_session_eval_speculative(ds4_session *s, int first_token,
                                 int max_tokens, int eos_token,
                                 float temperature, int top_k,
                                 float top_p, float min_p, uint64_t *rng,
                                 int *accepted, int accepted_cap,
                                 char *err, size_t errlen);
/* TP worker side of a mirrored speculative-verify block: run its half of the
 * batch verify for KV side effects, then obey the leader's commit frame
 * (keep, or roll back and replay). Only called from ds4_tp_worker_run. */
int ds4_session_tp_spec_cycle(ds4_session *s, const int *drafts, int draft_n,
                              char *err, size_t errlen);
int ds4_session_glm_tp_spec_cycle(ds4_session *s, int token, int limit,
                                 char *err, size_t errlen);
void ds4_session_invalidate(ds4_session *s);
/* Keep the token prefix, restoring recurrent state where possible. Otherwise
 * the checkpoint becomes invalid: sync the retained prefix before eval.
 * Callers retaining images must use sync_multimodal for that rebuild. */
void ds4_session_rewind(ds4_session *s, int pos);
int ds4_session_pos(ds4_session *s);
int ds4_session_ctx(ds4_session *s);
int ds4_session_prefill_cap(ds4_session *s);
int ds4_engine_routed_quant_bits(ds4_engine *e);
bool ds4_engine_has_output_head(ds4_engine *e);
bool ds4_engine_has_mtp(ds4_engine *e);
int ds4_engine_mtp_draft_tokens(ds4_engine *e);
bool ds4_engine_mtp_exact_sampling(ds4_engine *e);
const ds4_tokens *ds4_session_tokens(ds4_session *s);

/* Low-level graph slice entry points used by distributed inference.  The
 * transport/session routing logic lives in ds4_distributed.c. */
int ds4_session_layer_slice_reset(ds4_session *s, char *err, size_t errlen);
int ds4_session_eval_layer_slice(ds4_session *s,
                                 const int *tokens,
                                 uint32_t n_tokens,
                                 uint32_t pos0,
                                 uint32_t layer_start,
                                 uint32_t layer_end,
                                 const float *input_hc,
                                 float *output_hc,
                                 bool output_logits,
                                 float *logits,
                                 char *err,
                                 size_t errlen);
int ds4_session_eval_output_head_from_hc(ds4_session *s,
                                         const float *hidden_hc,
                                         uint32_t n_tokens,
                                         float *logits,
                                         char *err,
                                         size_t errlen);

/* Disk KV payload helpers.  HTTP/agent code owns the outer file header and
 * persistence policy; the engine owns the DS4-specific serialized graph state. */
#define DS4_SESSION_PAYLOAD_MAGIC UINT32_C(0x34565344) /* "DSV4" */
#define DS4_SESSION_PAYLOAD_VERSION UINT32_C(2)
#define DS4_SESSION_PAYLOAD_U32_FIELDS 13u
#define DS4_SESSION_LAYER_PAYLOAD_MAGIC UINT32_C(0x4c565344) /* "DSVL" */
#define DS4_SESSION_LAYER_PAYLOAD_VERSION UINT32_C(1)
#define DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS 14u

uint64_t ds4_session_payload_bytes(ds4_session *s);
int ds4_session_stage_payload(ds4_session *s, ds4_session_payload_file *out,
                              char *err, size_t errlen);
int ds4_session_write_staged_payload(const ds4_session_payload_file *payload,
                                     FILE *fp, char *err, size_t errlen);
void ds4_session_payload_file_free(ds4_session_payload_file *payload);
int ds4_session_save_payload(ds4_session *s, FILE *fp, char *err, size_t errlen);
int ds4_session_load_payload(ds4_session *s, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen);
int ds4_session_save_snapshot(ds4_session *s, ds4_session_snapshot *snap, char *err, size_t errlen);
int ds4_session_load_snapshot(ds4_session *s, const ds4_session_snapshot *snap, char *err, size_t errlen);
void ds4_session_snapshot_free(ds4_session_snapshot *snap);

uint64_t ds4_session_layer_payload_bytes(ds4_session *s,
                                         uint32_t layer_start,
                                         uint32_t layer_end);
int ds4_session_save_layer_payload(ds4_session *s, FILE *fp,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen);
int ds4_session_load_layer_payload(ds4_session *s, FILE *fp,
                                   uint64_t payload_bytes,
                                   const int *tokens, uint32_t n_tokens,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen);

#endif
