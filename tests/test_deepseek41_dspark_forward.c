/* Stage 3/4 oracle: the V4.1 DSpark drafter forward on Metal against the CPU
 * fp32 reference.
 *
 * Loads the DSpark support GGUF (7.76 GiB) and, from the main GGUF, the single
 * tensor the drafter ties rather than owns -- `output.weight`.  The 550B
 * backbone is never evaluated: the stimulus is the Stage-0 reference's own
 * synthetic `main_hidden` and its token-embedding expansion, both replayed from
 * the dumps, so this test costs one 8 GiB weight load and a few hundred
 * milliseconds of GPU.
 *
 * What it gates on, following SPEC (d) Stage 3/4 and risk R2:
 *   - relative error on the hc streams and the collapsed head input;
 *   - TOP-1 AGREEMENT of the base logits, which is the number that predicts
 *     acceptance -- a numerics regression shows up here and nowhere else;
 *   - exact agreement of the five proposed ids at greedy temperature;
 *   - confidence logits to ~1e-3 relative.
 * Quantization makes exact equality impossible: the reference dequantizes the
 * checkpoint's native FP8-E4M3 / packed-FP4, the runtime reads Q8_0 / Q4_K, and
 * V4.1 pins BF16 activation boundaries the fp32 reference does not model.
 * DS4_DSPARK41_DISABLE_BF16=1 turns the rounding off, which separates weight
 * quantization from the activation boundaries.
 */
#include "../ds4.c"
#include <assert.h>

#define ORACLE_POSITIONS 6
static const uint32_t g_positions[ORACLE_POSITIONS] = {1, 5, 127, 128, 129, 200};

typedef struct {
    double max_abs, max_rel, ref_absmax;
    uint64_t n;
} err_stat;

static float *read_bin(const char *dir, const char *name, uint64_t want_floats) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.bin", dir, name);
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "missing dump %s\n", path); return NULL; }
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return NULL; }
    const long bytes = ftell(fp);
    rewind(fp);
    if (bytes < 0 || (uint64_t)bytes != want_floats * 4u) {
        fprintf(stderr, "%s: %ld bytes, expected %llu\n", path, bytes,
                (unsigned long long)(want_floats * 4u));
        fclose(fp);
        return NULL;
    }
    float *out = malloc((size_t)bytes);
    const bool ok = out && fread(out, 1, (size_t)bytes, fp) == (size_t)bytes;
    fclose(fp);
    if (!ok) { free(out); return NULL; }
    return out;
}

static err_stat compare(const float *got, const float *ref, uint64_t n) {
    err_stat e = {0, 0, 0, n};
    for (uint64_t i = 0; i < n; i++) {
        const double a = got[i], b = ref[i];
        const double d = fabs(a - b);
        if (d > e.max_abs) e.max_abs = d;
        if (fabs(b) > e.ref_absmax) e.ref_absmax = fabs(b);
    }
    /* Relative to the tensor's own scale: a per-element ratio is meaningless
     * where the reference is near zero, and the hc streams are wide. */
    e.max_rel = e.ref_absmax > 0 ? e.max_abs / e.ref_absmax : 0.0;
    return e;
}

static err_stat compare_gpu(const ds4_gpu_tensor *t, uint64_t offset,
                            const float *ref, uint64_t n, bool *ok) {
    err_stat e = {0, 0, 0, n};
    float *got = malloc((size_t)n * sizeof(float));
    if (!got || !ds4_gpu_tensor_read(t, offset, got, n * sizeof(float))) {
        *ok = false;
        free(got);
        return e;
    }
    e = compare(got, ref, n);
    free(got);
    return e;
}

typedef struct {
    const char *name;
    err_stat    e;
} row;

static row g_rows[512];
static uint32_t g_row_count;
static double g_worst_rel;
static const char *g_worst_name = "";

static void note(const char *name, err_stat e) {
    if (g_row_count < 512) {
        g_rows[g_row_count].name = name;
        g_rows[g_row_count].e = e;
        g_row_count++;
    }
    if (e.max_rel > g_worst_rel) { g_worst_rel = e.max_rel; g_worst_name = name; }
    printf("    %-26s max|d| %10.3e  rel %8.3e  (|ref|max %9.3e)\n",
           name, e.max_abs, e.max_rel, e.ref_absmax);
}

static char *dup_name(const char *fmt, uint32_t s) {
    char *p = malloc(64);
    snprintf(p, 64, fmt, s);
    return p;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
            "usage: %s <dspark.gguf> <main.gguf> <oracle-dir>\n", argv[0]);
        return 2;
    }
    const char *dspark_path = argv[1];
    const char *main_path = argv[2];
    const char *oracle_dir = argv[3];

    g_ds4_shape = DS4_SHAPE_FLASH41;

    ds4_model dsm, basem;
    model_open(&dsm, dspark_path, false, false);
    uint32_t stages = 0;
    ds4_dspark_summary summary;
    const ds4_support_kind kind = support_model_detect(&dsm, &stages, &summary);
    if (kind != DS4_SUPPORT_DSPARK) { fprintf(stderr, "not a DSpark file\n"); return 1; }
    ds4_dspark_weights dw;
    memset(&dw, 0, sizeof(dw));
    dspark_weights_bind_optional(&dw, &dsm, &summary);
    if (dw.missing_tensors || dw.invalid_tensors || dw.metadata_errors) {
        fprintf(stderr, "support file did not bind clean: missing=%u invalid=%u meta=%u\n",
                dw.missing_tensors, dw.invalid_tensors, dw.metadata_errors);
        return 1;
    }
    printf("support: stages=%u block=%u experts=%u/%u ff=%u markov_rank=%u\n",
           dw.n_stages, dw.block_size, dw.n_routed_experts, dw.n_experts_per_tok,
           dw.expert_ff_len, dw.markov_rank);

    /* The one tensor read from the main model: the tied output head.  No
     * ds41_memory_admit, no model_warm_weights, no graph. */
    model_open(&basem, main_path, false, false);
    ds4_tensor *base_output = model_find_tensor(&basem, "output.weight");
    if (!base_output) { fprintf(stderr, "main model has no output.weight\n"); return 1; }

    /* Both mmaps must be registered as Metal model views before any kernel can
     * address a weight (ds4_gpu_wrap_model_range walks g_model_views).  The
     * support file is mapped whole, 7.76 GiB; the main model is mapped as a
     * SINGLE SPAN covering only output.weight -- mapping its whole tensor
     * region would ask the accelerator for a 483 GiB VM reservation to read
     * 0.7 GiB. */
    if (!ds4_gpu_set_model_map_range(dsm.map, dsm.size, dsm.tensor_data_pos,
                                     dsm.size - dsm.tensor_data_pos,
                                     dsm.max_tensor_bytes)) {
        fprintf(stderr, "failed to map DSpark support model views\n");
        return 1;
    }
    const uint64_t out_offset = base_output->abs_offset;
    const uint64_t out_bytes = base_output->bytes;
    if (!ds4_gpu_set_model_map_spans(basem.map, basem.size, &out_offset,
                                     &out_bytes, 1, out_bytes)) {
        fprintf(stderr, "failed to map the tied output head\n");
        return 1;
    }
    printf("mapped: support %.2f GiB whole, main output.weight %.2f GiB at %llu\n",
           (double)(dsm.size - dsm.tensor_data_pos) / (double)(1u << 30),
           (double)out_bytes / (double)(1u << 30),
           (unsigned long long)out_offset);

    ds41_dspark d;
    if (!ds41_dspark_alloc(&d, &dsm, &dw)) { fprintf(stderr, "drafter alloc failed\n"); return 1; }
    const uint32_t B = d.block, HC = DS4_N_HC, E = DS4_N_EMBD, V = d.vocab;
    const uint64_t hc_row = (uint64_t)HC * E, mix_hc = 24;
    const uint64_t hidden_dim = (uint64_t)dw.target_layer_count * E;

    ds4_gpu_tensor *hidden = ds4_gpu_tensor_alloc(
        (uint64_t)DS41_DSPARK_WINDOW * hidden_dim * sizeof(float));
    ds4_gpu_tensor *out_hc = ds4_gpu_tensor_alloc((uint64_t)B * hc_row * sizeof(float));
    if (!hidden || !out_hc) { fprintf(stderr, "scratch alloc failed\n"); return 1; }

    int failures = 0;
    uint32_t top1_total = 0, top1_match = 0, ids_total = 0, ids_match = 0;
    const bool rounding = ds41_dspark_round();
    printf("bf16 activation boundaries: %s\n\n", rounding ? "on (production)" : "OFF (diagnostic)");

    for (uint32_t pi = 0; pi < ORACLE_POSITIONS; pi++) {
        const uint32_t p = g_positions[pi];
        char dir[1024];
        snprintf(dir, sizeof(dir), "%s/pos%04u", oracle_dir, p);
        printf("start_pos=%u  (%s)\n", p, dir);

        const uint32_t n_seed = p < DS41_DSPARK_WINDOW ? p : DS41_DSPARK_WINDOW;
        const uint32_t pos0 = p - n_seed;
        bool ok = true;

        float *ref_hidden = read_bin(dir, "main_hidden", (uint64_t)(p + 1u) * hidden_dim);
        float *ref_main_x = read_bin(dir, "main_x", E);
        float *ref_pre_x = read_bin(dir, "main_x_prefill", (uint64_t)p * E);
        float *ref_input = read_bin(dir, "stage_input", (uint64_t)B * hc_row);
        if (!ref_hidden || !ref_main_x || !ref_pre_x || !ref_input) return 1;

        /* main_proj + main_norm over the seeding window and the decode row. */
        ok = ok && ds4_gpu_tensor_write(hidden, 0,
                ref_hidden + (uint64_t)pos0 * hidden_dim,
                (uint64_t)n_seed * hidden_dim * sizeof(float)) != 0;
        ds4_gpu_tensor *hidden_seed = ds4_gpu_tensor_view(hidden, 0,
                (uint64_t)n_seed * hidden_dim * sizeof(float));
        ds4_gpu_tensor *seed_x = ds4_gpu_tensor_view(d.seed_x, 0,
                (uint64_t)n_seed * E * sizeof(float));
        ok = ok && hidden_seed && seed_x &&
             ds41_dspark_project_main(&d, seed_x, hidden_seed, n_seed, false);
        if (ok) note("main_x_prefill",
                     compare_gpu(seed_x, 0, ref_pre_x + (uint64_t)pos0 * E,
                                 (uint64_t)n_seed * E, &ok));

        ok = ok && ds4_gpu_tensor_write(hidden, 0,
                ref_hidden + (uint64_t)p * hidden_dim,
                hidden_dim * sizeof(float)) != 0;
        ds4_gpu_tensor *hidden_one = ds4_gpu_tensor_view(hidden, 0,
                hidden_dim * sizeof(float));
        ok = ok && hidden_one && ds41_dspark_project_main(&d, d.main_x, hidden_one, 1, false);
        if (ok) note("main_x", compare_gpu(d.main_x, 0, ref_main_x, E, &ok));

        /* Seed the three rings, then check each against the reference ring --
         * the FP8-quantized, RoPEd main_kv rows at slot `position % 128`. */
        for (uint32_t s = 0; ok && s < d.stages; s++)
            ok = ds41_dspark_seed(&d, s, seed_x, pos0, n_seed, false);

        /* Draft-token embeddings, expanded over the four hc copies. */
        ok = ok && ds4_gpu_tensor_write(d.hc, 0, ref_input,
                                        (uint64_t)B * hc_row * sizeof(float)) != 0;

        ds4_gpu_tensor *in_hc = d.hc;
        const ds4_gpu_tensor *pre = d.in_split;
        for (uint32_t s = 0; ok && s < d.stages; s++) {
            ok = ds41_dspark_stage(&d, s, p, in_hc, pre, out_hc, false);
            if (!ok) { fprintf(stderr, "stage %u failed at pos %u, step %s\n", s, p, g_dspark_step); break; }

            float *r_post_attn = read_bin(dir, dup_name("stage%u_post_attn", s),
                                          (uint64_t)B * hc_row);
            float *r_post_ffn = read_bin(dir, dup_name("stage%u_post_ffn", s),
                                         (uint64_t)B * hc_row);
            float *r_attn_pre = read_bin(dir, dup_name("stage%u_attn_pre", s), (uint64_t)B * HC);
            float *r_ffn_pre = read_bin(dir, dup_name("stage%u_ffn_pre", s), (uint64_t)B * HC);
            if (!r_post_attn || !r_post_ffn || !r_attn_pre || !r_ffn_pre) return 1;

            /* The ring the attention just read: positions p+1-n_hist .. p at
             * slot `position % 192` here, `position % 128` in the reference.
             * The five draft rows at p+1..p+5 are ours alone and are skipped. */
            {
                float *r_ring = read_bin(dir, dup_name("ring%u", s),
                                         (uint64_t)DS41_DSPARK_WINDOW * DS4_N_HEAD_DIM);
                const uint32_t n_hist = p + 1u < DS41_DSPARK_WINDOW ? p + 1u : DS41_DSPARK_WINDOW;
                float *got = malloc((size_t)DS41_DSPARK_RING_SLOTS * DS4_N_HEAD_DIM * sizeof(float));
                if (r_ring && got &&
                    ds4_gpu_tensor_read(d.ring[s], 0, got,
                        (uint64_t)DS41_DSPARK_RING_SLOTS * DS4_N_HEAD_DIM * sizeof(float))) {
                    err_stat e = {0, 0, 0, 0};
                    for (uint32_t q = p + 1u - n_hist; q <= p; q++) {
                        const err_stat r = compare(
                            got + (uint64_t)(q % DS41_DSPARK_RING_SLOTS) * DS4_N_HEAD_DIM,
                            r_ring + (uint64_t)(q % DS41_DSPARK_WINDOW) * DS4_N_HEAD_DIM,
                            DS4_N_HEAD_DIM);
                        if (r.max_abs > e.max_abs) e.max_abs = r.max_abs;
                        if (r.ref_absmax > e.ref_absmax) e.ref_absmax = r.ref_absmax;
                        e.n += DS4_N_HEAD_DIM;
                    }
                    e.max_rel = e.ref_absmax > 0 ? e.max_abs / e.ref_absmax : 0.0;
                    note(dup_name("ring%u", s), e);
                } else ok = false;
                free(r_ring);
                free(got);
            }
            note(dup_name("stage%u_post_attn", s),
                 compare_gpu(d.after_hc, 0, r_post_attn, (uint64_t)B * hc_row, &ok));
            note(dup_name("stage%u_post_ffn", s),
                 compare_gpu(out_hc, 0, r_post_ffn, (uint64_t)B * hc_row, &ok));
            /* The split's first n_hc entries are `pre`; the rest is post/comb. */
            {
                float *got = malloc((size_t)B * mix_hc * sizeof(float));
                float *packed = malloc((size_t)B * HC * sizeof(float));
                if (got && packed &&
                    ds4_gpu_tensor_read(d.attn_split, 0, got,
                                        (uint64_t)B * mix_hc * sizeof(float))) {
                    for (uint32_t r = 0; r < B; r++)
                        memcpy(packed + r * HC, got + (uint64_t)r * mix_hc,
                               HC * sizeof(float));
                    note(dup_name("stage%u_attn_pre", s),
                         compare(packed, r_attn_pre, (uint64_t)B * HC));
                } else ok = false;
                if (got && packed &&
                    ds4_gpu_tensor_read(d.ffn_split, 0, got,
                                        (uint64_t)B * mix_hc * sizeof(float))) {
                    for (uint32_t r = 0; r < B; r++)
                        memcpy(packed + r * HC, got + (uint64_t)r * mix_hc,
                               HC * sizeof(float));
                    note(dup_name("stage%u_ffn_pre", s),
                         compare(packed, r_ffn_pre, (uint64_t)B * HC));
                } else ok = false;
                free(got);
                free(packed);
            }

            /* Routed-expert selection: compare as sets, since the reference's
             * topk is value-ordered and the kernel's need not be. */
            {
                float *r_ids = read_bin(dir, dup_name("stage%u_expert_ids", s),
                                        (uint64_t)B * d.n_expert_used);
                uint32_t *got = malloc((size_t)B * d.n_expert_used * sizeof(uint32_t));
                if (r_ids && got &&
                    ds4_gpu_tensor_read(d.selected, 0, got,
                        (uint64_t)B * d.n_expert_used * sizeof(uint32_t))) {
                    uint32_t agree = 0, total = 0;
                    for (uint32_t r = 0; r < B; r++)
                        for (uint32_t k = 0; k < d.n_expert_used; k++) {
                            total++;
                            const uint32_t want =
                                (uint32_t)((const int32_t *)r_ids)[r * d.n_expert_used + k];
                            for (uint32_t j = 0; j < d.n_expert_used; j++)
                                if (got[r * d.n_expert_used + j] == want) { agree++; break; }
                        }
                    printf("    %-26s %u/%u routed experts agree\n",
                           dup_name("stage%u_experts", s), agree, total);
                    if (agree != total) failures++;
                } else ok = false;
                free(r_ids);
                free(got);
            }

            free(r_post_attn); free(r_post_ffn); free(r_attn_pre); free(r_ffn_pre);
            pre = d.ffn_split;
            if (s + 1u < d.stages) {
                ok = ok && ds4_gpu_begin_commands() != 0 &&
                     ds4_gpu_tensor_copy(d.hc, 0, out_hc, 0,
                                         (uint64_t)B * hc_row * sizeof(float)) != 0 &&
                     ds4_gpu_end_commands() != 0;
                in_hc = d.hc;
            }
        }

        /* Head: collapse with stage 2's own ffn_pre, pre-head norm, tied head. */
        ok = ok && ds41_dspark_head(&d, out_hc, &basem, base_output, false);
        {
            float *r_head = read_bin(dir, "head_input", (uint64_t)B * E);
            float *r_headn = read_bin(dir, "head_input_normed", (uint64_t)B * E);
            float *r_logits = read_bin(dir, "base_logits", (uint64_t)B * V);
            if (!r_head || !r_headn || !r_logits) return 1;
            if (ok) note("head_input", compare_gpu(d.head_x, 0, r_head, (uint64_t)B * E, &ok));
            if (ok) note("head_input_normed",
                         compare_gpu(d.head_norm, 0, r_headn, (uint64_t)B * E, &ok));
            if (ok) note("base_logits",
                         compare_gpu(d.logits, 0, r_logits, (uint64_t)B * V, &ok));
            /* R2's gate: top-1 of the base logits, per draft row. */
            float *got = malloc((size_t)B * V * sizeof(float));
            if (ok && got &&
                ds4_gpu_tensor_read(d.logits, 0, got, (uint64_t)B * V * sizeof(float))) {
                uint32_t match = 0;
                for (uint32_t r = 0; r < B; r++) {
                    const uint32_t a = dspark_argmax_f32(got + (uint64_t)r * V, V);
                    const uint32_t b = dspark_argmax_f32(r_logits + (uint64_t)r * V, V);
                    if (a == b) match++;
                }
                printf("    %-26s %u/%u rows\n", "base_logits top-1", match, B);
                top1_match += match;
                top1_total += B;
                if (match != B) failures++;
            } else ok = false;
            free(got);
            free(r_head); free(r_headn); free(r_logits);
        }

        /* Stage 4: the Markov chain and the confidence head. */
        {
            char path[1024];
            snprintf(path, sizeof(path), "%s/output_ids.bin", dir);
            FILE *fp = fopen(path, "rb");
            int32_t ref_ids[DS4_DSPARK_MAX_BLOCK_SIZE + 1] = {0};
            const bool have = fp && fread(ref_ids, 4, (size_t)B + 1u, fp) == (size_t)B + 1u;
            if (fp) fclose(fp);
            int32_t proposal[DS4_DSPARK_MAX_BLOCK_SIZE] = {0};
            float *biased = malloc((size_t)B * V * sizeof(float));
            if (ok && have && biased &&
                ds41_dspark_markov_greedy(&d, ref_ids[0], biased, proposal)) {
                float *r_full = read_bin(dir, "logits", (uint64_t)B * V);
                if (r_full) { note("logits (base+markov)",
                                   compare(biased, r_full, (uint64_t)B * V)); free(r_full); }
                uint32_t match = 0;
                for (uint32_t r = 0; r < B; r++)
                    if (proposal[r] == ref_ids[r + 1]) match++;
                printf("    %-26s %u/%u  got", "draft ids", match, B);
                for (uint32_t r = 0; r < B; r++) printf(" %d", proposal[r]);
                printf("  want");
                for (uint32_t r = 0; r < B; r++) printf(" %d", ref_ids[r + 1]);
                printf("\n");
                ids_match += match;
                ids_total += B;
                if (match != B) failures++;

                float *hidden_rows = malloc((size_t)B * E * sizeof(float));
                float *ref_conf = read_bin(dir, "confidence", B);
                float *markov_state = malloc((size_t)d.markov_rank * sizeof(float));
                float *features = malloc(((size_t)E + d.markov_rank) * sizeof(float));
                float conf[DS4_DSPARK_MAX_BLOCK_SIZE] = {0};
                uint32_t conf_len = 0;
                if (hidden_rows && ref_conf && markov_state && features &&
                    ds4_gpu_tensor_read(d.head_x, 0, hidden_rows,
                                        (uint64_t)B * E * sizeof(float)) &&
                    dspark_eval_confidence_probe(conf, hidden_rows, &dsm, &dw,
                        ref_ids[0], proposal, markov_state, features, &conf_len) &&
                    conf_len == B) {
                    note("confidence", compare(conf, ref_conf, B));
                } else { fprintf(stderr, "confidence head failed\n"); failures++; }
                free(hidden_rows); free(ref_conf); free(markov_state); free(features);
            } else { fprintf(stderr, "markov chain failed at pos %u\n", p); failures++; }
            free(biased);
        }

        ds4_gpu_tensor_free(hidden_one);
        ds4_gpu_tensor_free(seed_x);
        ds4_gpu_tensor_free(hidden_seed);
        free(ref_hidden); free(ref_main_x); free(ref_pre_x); free(ref_input);
        if (!ok) { fprintf(stderr, "position %u failed\n", p); failures++; }
        printf("\n");
    }

    printf("worst relative error: %.3e on %s\n", g_worst_rel, g_worst_name);
    printf("base_logits top-1: %u/%u rows\n", top1_match, top1_total);
    printf("draft ids: %u/%u\n", ids_match, ids_total);
    printf("V4.1 DSpark drafter forward: %s\n", failures ? "FAIL" : "PASS");

    ds4_gpu_tensor_free(out_hc);
    ds4_gpu_tensor_free(hidden);
    ds41_dspark_free(&d);
    model_close(&basem);
    model_close(&dsm);
    return failures ? 1 : 0;
}
