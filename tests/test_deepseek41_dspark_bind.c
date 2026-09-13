/* Binding and validation of a V4.1 DSpark support GGUF.
 *
 * Header-only sparse fixtures: the tensor directory is written for real at the
 * V4.1 drafter's true shapes, the payload is a hole. Nothing here reads a weight,
 * allocates on the GPU or runs inference -- `dspark_weights_bind_optional` only
 * walks the directory.
 *
 * What it pins down, all of it new for V4.1:
 *   - the drafter's MoE shape comes from the support file's own metadata
 *     (128 routed / 3 active / 2304 wide), never from the 384/6 backbone;
 *   - `hc_head_*` is absent by design and its absence is not "missing";
 *   - a V4.1 file that carries `hc_head_*` anyway is invalid;
 *   - a V4 support file still binds through the backbone-shape fallback.
 */
#include "../ds4.c"
#include <assert.h>

static void put32(FILE *fp, uint32_t v) { assert(fwrite(&v, 4, 1, fp) == 1); }
static void put64(FILE *fp, uint64_t v) { assert(fwrite(&v, 8, 1, fp) == 1); }

static void putstr(FILE *fp, const char *s) {
    const size_t n = strlen(s);
    put64(fp, n);
    assert(fwrite(s, 1, n, fp) == n);
}

static void kv_string(FILE *fp, const char *key, const char *value) {
    putstr(fp, key);
    put32(fp, GGUF_VALUE_STRING);
    putstr(fp, value);
}

static void kv_u32(FILE *fp, const char *key, uint32_t value) {
    putstr(fp, key);
    put32(fp, GGUF_VALUE_UINT32);
    put32(fp, value);
}

static void kv_u32_array(FILE *fp, const char *key, const uint32_t *v, uint32_t n) {
    putstr(fp, key);
    put32(fp, GGUF_VALUE_ARRAY);
    put32(fp, GGUF_VALUE_UINT32);
    put64(fp, n);
    for (uint32_t i = 0; i < n; i++) put32(fp, v[i]);
}

/* ------------------------------------------------------------------ */

#define FIXTURE_MAX_TENSORS 128
#define FIXTURE_ALIGN 16384u

typedef struct {
    char     name[96];
    uint32_t type;
    uint32_t ndim;
    uint64_t dim[3];
    uint64_t offset;
    uint64_t bytes;
} fixture_tensor;

typedef struct {
    fixture_tensor t[FIXTURE_MAX_TENSORS];
    uint32_t       n;
    uint64_t       next;
} fixture;

static uint64_t type_bytes(uint32_t type, const uint64_t *dim, uint32_t ndim) {
    uint64_t block = 1, block_bytes = 4;
    switch (type) {
    case DS4_TENSOR_F32:  block = 1;   block_bytes = 4;   break;
    case DS4_TENSOR_F16:  block = 1;   block_bytes = 2;   break;
    case DS4_TENSOR_Q8_0: block = 32;  block_bytes = 34;  break;
    case DS4_TENSOR_Q4_K: block = 256; block_bytes = 144; break;
    default: assert(0);
    }
    assert(dim[0] % block == 0);
    uint64_t bytes = dim[0] / block * block_bytes;
    for (uint32_t i = 1; i < ndim; i++) bytes *= dim[i];
    return bytes;
}

static void add(fixture *f, const char *name, uint32_t type,
                uint64_t d0, uint64_t d1, uint64_t d2) {
    assert(f->n < FIXTURE_MAX_TENSORS);
    fixture_tensor *t = &f->t[f->n++];
    assert((size_t)snprintf(t->name, sizeof(t->name), "%s", name) < sizeof(t->name));
    t->type = type;
    t->ndim = d2 ? 3 : (d1 ? 2 : 1);
    t->dim[0] = d0; t->dim[1] = d1; t->dim[2] = d2;
    t->bytes = type_bytes(type, t->dim, t->ndim);
    t->offset = f->next;
    f->next = align_up(f->next + t->bytes, FIXTURE_ALIGN);
}

static void addf(fixture *f, uint32_t stage, const char *suffix, uint32_t type,
                 uint64_t d0, uint64_t d1, uint64_t d2) {
    char name[96];
    assert((size_t)snprintf(name, sizeof(name), "mtp.%u.%s", stage, suffix) < sizeof(name));
    add(f, name, type, d0, d1, d2);
}

typedef enum {
    FIXTURE_V41 = 0,          /* the artifact deepseek41_dspark_quantize.py emits */
    FIXTURE_V41_NO_EXPERT_META,   /* drafter MoE shape not declared */
    FIXTURE_V41_BACKBONE_EXPERTS, /* experts sized 384 while metadata says 128 */
    FIXTURE_V41_WITH_HC_HEAD,     /* hc_head_* present on a V4.1 file */
    FIXTURE_V4,               /* a V4-style file: no arch, hc_head_*, backbone shape */
} fixture_kind;

static void build_tensors(fixture *f, fixture_kind kind, uint32_t experts, uint32_t ff) {
    const uint64_t hc_dim = (uint64_t)DS4_N_EMBD * DS4_N_HC;
    const uint64_t hc_mix = 2u * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t out_low = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;
    const uint32_t stages = 3;

    for (uint32_t s = 0; s < stages; s++) {
        addf(f, s, "hc_attn_fn.weight",   DS4_TENSOR_F16, hc_dim, hc_mix, 0);
        addf(f, s, "hc_attn_scale.weight", DS4_TENSOR_F32, 3, 0, 0);
        addf(f, s, "hc_attn_base.weight",  DS4_TENSOR_F32, hc_mix, 0, 0);
        addf(f, s, "attn_norm.weight",     DS4_TENSOR_F32, DS4_N_EMBD, 0, 0);
        addf(f, s, "attn_sinks.weight",    DS4_TENSOR_F32, DS4_N_HEAD, 0, 0);
        addf(f, s, "attn_q_a.weight",      DS4_TENSOR_Q8_0, DS4_N_EMBD, DS4_N_LORA_Q, 0);
        addf(f, s, "attn_q_a_norm.weight", DS4_TENSOR_F32, DS4_N_LORA_Q, 0, 0);
        addf(f, s, "attn_q_b.weight",      DS4_TENSOR_Q8_0, DS4_N_LORA_Q, q_dim, 0);
        addf(f, s, "attn_kv.weight",       DS4_TENSOR_Q8_0, DS4_N_EMBD, DS4_N_HEAD_DIM, 0);
        addf(f, s, "attn_kv_a_norm.weight", DS4_TENSOR_F32, DS4_N_HEAD_DIM, 0, 0);
        addf(f, s, "attn_output_a.weight", DS4_TENSOR_Q8_0,
             DS4_N_HEAD_DIM * (DS4_N_HEAD / DS4_N_OUT_GROUP), out_low, 0);
        addf(f, s, "attn_output_b.weight", DS4_TENSOR_Q8_0, out_low, DS4_N_EMBD, 0);
        addf(f, s, "hc_ffn_fn.weight",     DS4_TENSOR_F16, hc_dim, hc_mix, 0);
        addf(f, s, "hc_ffn_scale.weight",  DS4_TENSOR_F32, 3, 0, 0);
        addf(f, s, "hc_ffn_base.weight",   DS4_TENSOR_F32, hc_mix, 0, 0);
        addf(f, s, "ffn_norm.weight",      DS4_TENSOR_F32, DS4_N_EMBD, 0, 0);
        addf(f, s, "ffn_gate_inp.weight",  DS4_TENSOR_F32, DS4_N_EMBD, experts, 0);
        addf(f, s, "exp_probs_b.bias",     DS4_TENSOR_F32, experts, 0, 0);
        addf(f, s, "exp_probs_b_vl.bias",  DS4_TENSOR_F32, experts, 0, 0);
        addf(f, s, "ffn_gate_exps.weight", DS4_TENSOR_Q4_K, DS4_N_EMBD, ff, experts);
        addf(f, s, "ffn_up_exps.weight",   DS4_TENSOR_Q4_K, DS4_N_EMBD, ff, experts);
        addf(f, s, "ffn_down_exps.weight", DS4_TENSOR_Q4_K, ff, DS4_N_EMBD, experts);
        addf(f, s, "ffn_gate_shexp.weight", DS4_TENSOR_Q8_0, DS4_N_EMBD, ff, 0);
        addf(f, s, "ffn_up_shexp.weight",  DS4_TENSOR_Q8_0, DS4_N_EMBD, ff, 0);
        addf(f, s, "ffn_down_shexp.weight", DS4_TENSOR_Q8_0, ff, DS4_N_EMBD, 0);
        if (s == 0) {
            addf(f, s, "main_proj.weight", DS4_TENSOR_Q8_0,
                 3u * DS4_N_EMBD, DS4_N_EMBD, 0);
            addf(f, s, "main_norm.weight", DS4_TENSOR_F32, DS4_N_EMBD, 0, 0);
        }
        if (s == stages - 1) {
            addf(f, s, "norm.weight", DS4_TENSOR_F32, DS4_N_EMBD, 0, 0);
            addf(f, s, "markov_head.markov_w1.weight", DS4_TENSOR_Q8_0, 256, DS4_N_VOCAB, 0);
            addf(f, s, "markov_head.markov_w2.weight", DS4_TENSOR_Q8_0, 256, DS4_N_VOCAB, 0);
            addf(f, s, "confidence_head.proj.weight", DS4_TENSOR_F32,
                 (uint64_t)DS4_N_EMBD + 256u, 1, 0);
            if (kind == FIXTURE_V41_WITH_HC_HEAD || kind == FIXTURE_V4) {
                addf(f, s, "hc_head_base.weight", DS4_TENSOR_F32, DS4_N_HC, 0, 0);
                addf(f, s, "hc_head_fn.weight", DS4_TENSOR_F16, hc_dim, DS4_N_HC, 0);
                addf(f, s, "hc_head_scale.weight", DS4_TENSOR_F32, 1, 0, 0);
            }
        }
    }
}

static void write_fixture(const char *path, fixture_kind kind) {
    static const uint32_t targets[3] = { 37, 38, 39 };
    const bool v41 = kind != FIXTURE_V4;
    /* FIXTURE_V41_BACKBONE_EXPERTS emits the target model's expert shape while
     * still declaring the drafter's -- the exact confusion §c.1 warns about. */
    const uint32_t experts = (kind == FIXTURE_V41_BACKBONE_EXPERTS || kind == FIXTURE_V4)
                             ? DS4_N_EXPERT : 128u;
    const uint32_t ff = (kind == FIXTURE_V4) ? DS4_N_FF_EXP : 2304u;

    fixture f = {0};
    build_tensors(&f, kind, experts, ff);

    FILE *fp = fopen(path, "w+b");
    assert(fp);
    uint32_t n_kv = v41 ? 11 : 6;
    if (kind == FIXTURE_V41_NO_EXPERT_META) n_kv -= 3;
    put32(fp, DS4_GGUF_MAGIC);
    put32(fp, 3);
    put64(fp, f.n);
    put64(fp, n_kv);
    kv_u32(fp, "general.alignment", FIXTURE_ALIGN);
    if (v41) {
        kv_string(fp, "general.architecture", "deepseek41-dspark");
        kv_string(fp, "general.name", "DeepSeek V4.1 Flash DSpark");
        kv_u32(fp, "deepseek41.dspark.block_size", 5);
        kv_u32(fp, "deepseek41.dspark.markov_rank", 256);
        kv_u32(fp, "deepseek41.dspark.noise_token_id", 128799);
        kv_u32_array(fp, "deepseek41.dspark.target_layer_ids", targets, 3);
        kv_u32(fp, "deepseek41.dspark.stage_count", 3);
        if (kind != FIXTURE_V41_NO_EXPERT_META) {
            kv_u32(fp, "deepseek41.dspark.n_routed_experts", 128);
            kv_u32(fp, "deepseek41.dspark.num_experts_per_tok", 3);
            kv_u32(fp, "deepseek41.dspark.expert_feed_forward_length", 2304);
        }
    } else {
        kv_string(fp, "general.architecture", "deepseek4-dspark");
        kv_u32(fp, "deepseek4.dspark.block_size", 5);
        kv_u32(fp, "deepseek4.dspark.markov_rank", 256);
        kv_u32(fp, "deepseek4.dspark.noise_token_id", 128799);
        kv_u32_array(fp, "deepseek4.dspark.target_layer_ids", targets, 3);
    }
    for (uint32_t i = 0; i < f.n; i++) {
        const fixture_tensor *t = &f.t[i];
        putstr(fp, t->name);
        put32(fp, t->ndim);
        for (uint32_t d = 0; d < t->ndim; d++) put64(fp, t->dim[d]);
        put32(fp, t->type);
        put64(fp, t->offset);
    }
    const long header = ftell(fp);
    assert(header > 0);
    assert(fflush(fp) == 0);
    /* The payload is a hole: nothing in the bind path reads a weight. */
    const uint64_t data = align_up((uint64_t)header, FIXTURE_ALIGN);
    assert(ftruncate(fileno(fp), (off_t)(data + f.next)) == 0);
    assert(fclose(fp) == 0);
}

/* ------------------------------------------------------------------ */

static ds4_support_kind open_and_bind(const char *path, ds4_dspark_weights *dw,
                                      ds4_dspark_summary *summary, uint32_t *stages) {
    ds4_model m;
    model_open(&m, path, false, false);
    const ds4_support_kind kind = support_model_detect(&m, stages, summary);
    memset(dw, 0, sizeof(*dw));
    if (kind == DS4_SUPPORT_DSPARK) dspark_weights_bind_optional(dw, &m, summary);
    model_close(&m);
    return kind;
}

static void run(fixture_kind kind, const char *label,
                ds4_dspark_weights *dw, ds4_dspark_summary *summary) {
    char path[] = "/tmp/ds41-dspark-bind.XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0 && close(fd) == 0);
    write_fixture(path, kind);
    uint32_t stages = 0;
    const ds4_support_kind detected = open_and_bind(path, dw, summary, &stages);
    assert(unlink(path) == 0);
    assert(detected == DS4_SUPPORT_DSPARK);
    assert(stages == 3 && dw->n_stages == 3);
    printf("%-28s stages=%u block=%u markov_rank=%u experts=%u/%u ff=%u "
           "tensors=%u missing=%u invalid=%u metadata_errors=%u\n",
           label, dw->n_stages, dw->block_size, dw->markov_rank,
           dw->n_routed_experts, dw->n_experts_per_tok, dw->expert_ff_len,
           dw->present_tensors, dw->missing_tensors, dw->invalid_tensors,
           dw->metadata_errors);
}

int main(void) {
    g_ds4_shape = DS4_SHAPE_FLASH41;
    ds4_dspark_weights dw;
    ds4_dspark_summary summary;

    /* The shipped artifact binds clean, with the drafter's own MoE shape. */
    run(FIXTURE_V41, "v4.1 dspark", &dw, &summary);
    assert(summary.is_deepseek41);
    assert(dw.block_size == 5 && dw.markov_rank == 256 && dw.noise_token_id == 128799);
    assert(dw.target_layer_count == 3 && dw.target_layers[0] == 37 &&
           dw.target_layers[1] == 38 && dw.target_layers[2] == 39);
    assert(dw.has_expert_shape);
    assert(dw.n_routed_experts == 128 && dw.n_experts_per_tok == 3);
    assert(dw.expert_ff_len == 2304);
    /* Drafter counts, not the 384/6 backbone the file is drafting for. */
    assert(DS4_N_EXPERT == 384 && DS4_N_EXPERT_USED == 6);
    assert(dw.missing_tensors == 0 && dw.invalid_tensors == 0 && dw.metadata_errors == 0);
    assert(dw.present_tensors == 81 - 3 /* exp_probs_b_vl is emitted but not bound */);
    /* hc_head_* is absent by design and must not count as missing. */
    assert(!summary.has_hc_head);
    assert(dw.stage[2].hc_head_fn == NULL && dw.stage[2].hc_head_base == NULL &&
           dw.stage[2].hc_head_scale == NULL);
    assert(dw.stage[2].norm && dw.stage[2].markov_w1 && dw.stage[2].markov_w2 &&
           dw.stage[2].confidence_proj);
    assert(dw.stage[0].main_proj && dw.stage[0].main_norm);

    /* Without the drafter MoE metadata a V4.1 file is rejected, never silently
     * given the backbone's counts. */
    run(FIXTURE_V41_NO_EXPERT_META, "v4.1 no expert metadata", &dw, &summary);
    assert(!dw.has_expert_shape && dw.metadata_errors > 0);

    /* Experts sized like the backbone against drafter metadata: every routed and
     * router tensor is wrong, and the validator says so. */
    run(FIXTURE_V41_BACKBONE_EXPERTS, "v4.1 backbone expert shape", &dw, &summary);
    assert(dw.metadata_errors == 0 && dw.missing_tensors == 0);
    assert(dw.invalid_tensors == 3 * 5 /* gate_inp, exp_probs_b, 3 routed, per stage */);

    /* A V4.1 file has no head projection; carrying one is a real error. */
    run(FIXTURE_V41_WITH_HC_HEAD, "v4.1 with hc_head_*", &dw, &summary);
    assert(summary.has_hc_head);
    assert(dw.invalid_tensors == 1 && dw.missing_tensors == 0 && dw.metadata_errors == 0);

    /* A V4 support file still binds: no arch string, hc_head_* required, and the
     * drafter shape falls back to the backbone's, which is what V4 meant. */
    run(FIXTURE_V4, "v4 dspark (fallback)", &dw, &summary);
    assert(!summary.is_deepseek41 && !dw.has_expert_shape);
    assert(dw.n_routed_experts == DS4_N_EXPERT &&
           dw.n_experts_per_tok == DS4_N_EXPERT_USED &&
           dw.expert_ff_len == DS4_N_FF_EXP);
    assert(dw.stage[2].hc_head_fn && dw.stage[2].hc_head_base && dw.stage[2].hc_head_scale);
    assert(dw.missing_tensors == 0 && dw.invalid_tensors == 0 && dw.metadata_errors == 0);

    puts("V4.1 DSpark support binding: PASS");
    return 0;
}
