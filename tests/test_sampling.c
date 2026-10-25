#include "../ds4.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    int id;
    float logit;
    float prob;
} reference_candidate;

static int failures;

#define CHECK(cond, ...) do {                                                 \
    if (!(cond)) {                                                            \
        fprintf(stderr, "FAIL: " __VA_ARGS__);                               \
        fputc('\n', stderr);                                                  \
        failures++;                                                           \
    }                                                                         \
} while (0)

static uint64_t reference_rng_next(uint64_t *state) {
    uint64_t x = *state;
    if (x == 0) x = 0x9e3779b97f4a7c15ULL;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545f4914f6cdd1dULL;
}

static float reference_rng_f32(uint64_t *state) {
    const uint64_t x = reference_rng_next(state);
    return (float)((x >> 40) & 0xffffffu) / 16777216.0f;
}

static int reference_argmax(const float *logits, uint32_t n_vocab) {
    int best = 0;
    float best_v = -INFINITY;
    for (uint32_t i = 0; i < n_vocab; i++) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best = (int)i;
        }
    }
    return best;
}

static int reference_candidate_cmp_desc(const void *a, const void *b) {
    const reference_candidate *ca = a;
    const reference_candidate *cb = b;
    const int logit_order =
        (cb->logit > ca->logit) - (cb->logit < ca->logit);
    if (logit_order != 0) return logit_order;
    return (ca->id > cb->id) - (ca->id < cb->id);
}

/* This is the sampler immediately before the optimized implementation. */
static int reference_sample(const float *logits, uint32_t n_vocab,
                            float temperature, int top_k,
                            float top_p, float min_p, uint64_t *rng) {
    if (temperature <= 0.0f) return reference_argmax(logits, n_vocab);
    if (top_p <= 0.0f || top_p > 1.0f) top_p = 1.0f;
    if (min_p < 0.0f) min_p = 0.0f;

    if (top_k > 0) {
        if (top_k > 1024) top_k = 1024;
        if ((uint32_t)top_k > n_vocab) top_k = (int)n_vocab;
        int ids[1024];
        float vals[1024];
        int n = 0;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            if (n == top_k && v <= vals[n - 1]) continue;
            int j = n < top_k ? n++ : n - 1;
            while (j > 0 && vals[j - 1] < v) {
                vals[j] = vals[j - 1];
                ids[j] = ids[j - 1];
                j--;
            }
            vals[j] = v;
            ids[j] = (int)i;
        }
        if (n == 0) return reference_argmax(logits, n_vocab);

        float probs[1024];
        const float max_logit = vals[0];
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            probs[i] = expf((vals[i] - max_logit) / temperature);
            sum += probs[i];
        }
        if (sum <= 0.0f || !isfinite(sum)) return ids[0];
        const float min_prob = (probs[0] / sum) * min_p;
        float filtered_sum = 0.0f;
        int filtered = 0;
        for (int i = 0; i < n; i++) {
            const float p = probs[i] / sum;
            if (i > 0 && p < min_prob) break;
            filtered_sum += probs[i];
            filtered++;
            if (filtered_sum / sum >= top_p) break;
        }
        float r = reference_rng_f32(rng) * filtered_sum;
        for (int i = 0; i < filtered; i++) {
            r -= probs[i];
            if (r <= 0.0f) return ids[i];
        }
        return ids[filtered - 1];
    }

    float max_logit = -INFINITY;
    int best = 0;
    uint32_t finite = 0;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        finite++;
        if (v > max_logit) {
            max_logit = v;
            best = (int)i;
        }
    }
    if (finite == 0) return reference_argmax(logits, n_vocab);

    if (top_p >= 1.0f) {
        float sum = 0.0f;
        const float min_rel = min_p > 0.0f ? min_p : 0.0f;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            const float p = expf((v - max_logit) / temperature);
            if (p < min_rel) continue;
            sum += p;
        }
        if (sum <= 0.0f || !isfinite(sum)) return best;
        float r = reference_rng_f32(rng) * sum;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            const float p = expf((v - max_logit) / temperature);
            if (p < min_rel) continue;
            r -= p;
            if (r <= 0.0f) return (int)i;
        }
        return best;
    }

    reference_candidate *cand = malloc((size_t)finite * sizeof(*cand));
    CHECK(cand != NULL, "reference candidate allocation");
    if (!cand) return best;
    uint32_t n = 0;
    float sum = 0.0f;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        const float p = expf((v - max_logit) / temperature);
        cand[n++] = (reference_candidate){(int)i, v, p};
        sum += p;
    }
    if (sum <= 0.0f || !isfinite(sum)) {
        free(cand);
        return best;
    }
    qsort(cand, n, sizeof(*cand), reference_candidate_cmp_desc);
    const float min_prob = (cand[0].prob / sum) * (min_p > 0.0f ? min_p : 0.0f);
    float filtered_sum = 0.0f;
    uint32_t filtered = 0;
    for (uint32_t i = 0; i < n; i++) {
        const float p = cand[i].prob / sum;
        if (i > 0 && p < min_prob) break;
        filtered_sum += cand[i].prob;
        filtered++;
        if (filtered_sum / sum >= top_p) break;
    }
    float r = reference_rng_f32(rng) * filtered_sum;
    for (uint32_t i = 0; i < filtered; i++) {
        r -= cand[i].prob;
        if (r <= 0.0f) {
            const int id = cand[i].id;
            free(cand);
            return id;
        }
    }
    const int id = cand[filtered - 1].id;
    free(cand);
    return id;
}

static uint64_t data_rng_next(uint64_t *state) {
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return *state;
}

static void fill_logits(float *logits, uint32_t n, uint64_t seed) {
    for (uint32_t i = 0; i < n; i++) {
        const uint64_t x = data_rng_next(&seed);
        const int32_t q = (int32_t)(x >> 32) % 250000;
        logits[i] = (float)q / 10000.0f + (float)i * 1.0e-7f;
    }
    if (n > 17) logits[17] = -INFINITY;
    if (n > 113) logits[113] = NAN;
}

static void compare_case(const float *logits, float *scratch, uint32_t n,
                         float temperature, int top_k,
                         float top_p, float min_p, const char *label) {
    for (uint64_t seed = 0; seed < 256; seed++) {
        uint64_t ref_rng = seed;
        uint64_t opt_rng = seed;
        const int ref = reference_sample(logits, n, temperature, top_k,
                                         top_p, min_p, &ref_rng);
        const int opt = ds4_test_sample_logits(logits, n, temperature, top_k,
                                               top_p, min_p, &opt_rng, scratch);
        CHECK(ref == opt,
              "%s seed=%llu token reference=%d optimized=%d",
              label, (unsigned long long)seed, ref, opt);
        CHECK(ref_rng == opt_rng,
              "%s seed=%llu RNG reference=%llu optimized=%llu",
              label, (unsigned long long)seed,
              (unsigned long long)ref_rng, (unsigned long long)opt_rng);
    }
}

static void check_greedy_argmax_case(const float *logits, uint32_t n,
                                     int expected, const char *label) {
    uint64_t unrolled_rng = 0x1234u;
    uint64_t scalar_rng = unrolled_rng;
    CHECK(unsetenv("DS4_CPU_DISABLE_UNROLLED_ARGMAX") == 0,
          "%s select unrolled argmax", label);
    const int unrolled = ds4_test_sample_logits(
            logits, n, 0.0f, 0, 1.0f, 0.0f, &unrolled_rng, NULL);
    CHECK(setenv("DS4_CPU_DISABLE_UNROLLED_ARGMAX", "1", 1) == 0,
          "%s select scalar argmax", label);
    const int scalar = ds4_test_sample_logits(
            logits, n, 0.0f, 0, 1.0f, 0.0f, &scalar_rng, NULL);
    CHECK(unsetenv("DS4_CPU_DISABLE_UNROLLED_ARGMAX") == 0,
          "%s restore unrolled argmax", label);
    CHECK(unrolled == scalar,
          "%s unrolled=%d scalar=%d", label, unrolled, scalar);
    CHECK(unrolled == expected,
          "%s token=%d expected=%d", label, unrolled, expected);
    CHECK(unrolled_rng == 0x1234u && scalar_rng == 0x1234u,
          "%s greedy argmax changed RNG state", label);
}

static void check_excluding_argmax_case(const float *logits, uint32_t n,
                                        int excluded_id, int expected,
                                        const char *label) {
    CHECK(unsetenv("DS4_CPU_DISABLE_UNROLLED_ARGMAX") == 0,
          "%s select unrolled excluding argmax", label);
    const int unrolled = ds4_test_argmax_excluding_logits(
            logits, n, excluded_id);
    CHECK(setenv("DS4_CPU_DISABLE_UNROLLED_ARGMAX", "1", 1) == 0,
          "%s select scalar excluding argmax", label);
    const int scalar = ds4_test_argmax_excluding_logits(
            logits, n, excluded_id);
    CHECK(unsetenv("DS4_CPU_DISABLE_UNROLLED_ARGMAX") == 0,
          "%s restore unrolled excluding argmax", label);
    CHECK(unrolled == scalar,
          "%s excluded=%d unrolled=%d scalar=%d",
          label, excluded_id, unrolled, scalar);
    CHECK(unrolled == expected,
          "%s excluded=%d token=%d expected=%d",
          label, excluded_id, unrolled, expected);
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static void check_speculative_distribution(void) {
    const float filter_logits[] = {
        logf(0.50f), logf(0.30f), logf(0.15f), logf(0.05f),
    };
    float probs[4];
    CHECK(ds4_test_sampling_probabilities(filter_logits, 4, 1.0f,
                                          3, 0.70f, 0.0f, probs) != 0,
          "build top-k/top-p probabilities");
    CHECK(fabsf(probs[0] - 0.625f) < 1e-6f,
          "top-k/top-p p0 %.9g", probs[0]);
    CHECK(fabsf(probs[1] - 0.375f) < 1e-6f,
          "top-k/top-p p1 %.9g", probs[1]);
    CHECK(probs[2] == 0.0f && probs[3] == 0.0f,
          "top-k/top-p filtered tail %.9g %.9g", probs[2], probs[3]);

    CHECK(ds4_test_sampling_probabilities(filter_logits, 4, 1.0f,
                                          0, 1.0f, 0.40f, probs) != 0,
          "build min-p probabilities");
    CHECK(fabsf(probs[0] - 0.625f) < 1e-6f,
          "min-p p0 %.9g", probs[0]);
    CHECK(fabsf(probs[1] - 0.375f) < 1e-6f,
          "min-p p1 %.9g", probs[1]);
    CHECK(probs[2] == 0.0f && probs[3] == 0.0f,
          "min-p filtered tail %.9g %.9g", probs[2], probs[3]);

    const float target_logits[] = {
        logf(0.62f), logf(0.27f), logf(0.11f),
    };
    const float draft_logits[] = {
        logf(0.15f), logf(0.55f), logf(0.30f),
    };
    uint64_t rng = 0x45f17a9d2c6b0381ULL;
    uint32_t counts[3] = {0};
    float target_probs[3];
    float draft_probs[3];
    const uint32_t trials = 100000;
    for (uint32_t i = 0; i < trials; i++) {
        const int token = ds4_test_speculative_sample(
            target_logits, draft_logits, 3, 1.0f, 0, 1.0f, 0.0f,
            &rng, target_probs, draft_probs);
        CHECK(token >= 0 && token < 3,
              "speculative sample token %d", token);
        if (token >= 0 && token < 3) counts[token]++;
    }
    const float expected[] = {0.62f, 0.27f, 0.11f};
    for (uint32_t i = 0; i < 3; i++) {
        const float observed = (float)counts[i] / (float)trials;
        CHECK(fabsf(observed - expected[i]) < 0.006f,
              "speculative distribution token=%u observed=%.6f expected=%.6f",
              i, observed, expected[i]);
    }
    printf("stochastic speculative distribution: %.4f %.4f %.4f\n",
           (double)counts[0] / trials,
           (double)counts[1] / trials,
           (double)counts[2] / trials);

    memset(counts, 0, sizeof(counts));
    rng = 0x8f76c2b5a149d30eULL;
    for (uint32_t i = 0; i < trials; i++) {
        const int token = ds4_test_speculative_delta_sample(
            target_logits, 3, 0, 1.0f, 0, 1.0f, 0.0f,
            &rng, target_probs);
        CHECK(token >= 0 && token < 3,
              "delta speculative sample token %d", token);
        if (token >= 0 && token < 3) counts[token]++;
    }
    for (uint32_t i = 0; i < 3; i++) {
        const float observed = (float)counts[i] / (float)trials;
        CHECK(fabsf(observed - expected[i]) < 0.006f,
              "delta distribution token=%u observed=%.6f expected=%.6f",
              i, observed, expected[i]);
    }
    printf("delta speculative distribution: %.4f %.4f %.4f\n",
           (double)counts[0] / trials,
           (double)counts[1] / trials,
           (double)counts[2] / trials);
}

/* The V4.1 GPU top-p selector's CPU consumer: the production (fast, sort-free) consumer must match the reference
 * consumer on every summary (return path, token, RNG state after, and on paths 0-2 the summary it leaves, bitwise),
 * and a decided step on a summary built from real logits must give the CPU sampler's token and RNG state.
 * Summaries are built the way metal/sampler_selector.metal lays them out; some are then corrupted. */
enum { SELT_NV = 129280u, SELT_NB = 129280u / 256u, SELT_SUPB = 2048u };
typedef struct { uint32_t n, finite, best, cap; float M; uint32_t b1, above1, b2, above2, t, above, pad[5]; } selt_hdr;
typedef struct { float logit; uint32_t id; } selt_cand;
typedef struct { int id; float logit, prob; } selt_sum;
static uint64_t selt_s;
static uint64_t selt_r64(void) {
    uint64_t z = (selt_s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
static double selt_u(void) { return ((double)(selt_r64() >> 11) + 0.5) / 9007199254740992.0; }
static uint32_t selt_ri(uint32_t n) { return (uint32_t)(selt_r64() % n); }
static float selt_bitsf(uint32_t u) { float v; memcpy(&v, &u, 4); return v; }
static uint32_t selt_fbits(float v) { uint32_t u; memcpy(&u, &v, 4); return u; }
static int selt_fin(float v) { return (selt_fbits(v) & 0x7f800000u) != 0x7f800000u; }
static uint32_t selt_okey(float v) {
    uint32_t u = selt_fbits(v);
    if (u == 0x80000000u) u = 0;
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
static int selt_cmp_u32_desc(const void *a, const void *b) {
    const uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x < y) - (x > y);
}
static int selt_cmp_f_desc(const void *a, const void *b) {
    const float x = *(const float *)a, y = *(const float *)b;
    return (x < y) - (x > y);
}

/* the summary of the NG values VALS at ids IDS (LG holds them; finite = NG when FINITE_ALL, else the whole
 * vocabulary). PMODE: 0 random heap share, 1 at the heap-mass boundary, 2 at a nucleus boundary, 3 zero,
 * 4 the true in-order FP32 block partials of LG (FINITE_ALL only). Returns 0 when NG cannot fill a heap. */
static int selt_build(selt_hdr *h, float *P, selt_cand *c, const float *lg, const uint32_t *ids, const float *vals,
                      uint32_t ng, int finite_all, int pmode) {
    static uint32_t keys[4096], order[4096];
    static float top[4096];
    memset(h, 0, sizeof(*h));
    const uint32_t finite = finite_all ? ng : SELT_NV;
    const uint32_t cap = finite < 512u ? finite : 512u;
    if (ng == 0 || ng < cap) return 0;
    for (uint32_t i = 0; i < ng; i++) keys[i] = selt_okey(vals[i]);
    qsort(keys, ng, sizeof(keys[0]), selt_cmp_u32_desc);
    const uint32_t b = keys[cap - 1u] >> 21;
    uint32_t above = 0, n = 0, m = 0;
    for (uint32_t i = 0; i < ng; i++) {
        const uint32_t bk = selt_okey(vals[i]) >> 21;
        above += bk > b;
        n += bk >= b;
        if (bk >= b) order[m++] = i;
    }
    for (uint32_t i = m; i > 1; i--) {   /* the GPU emits in no particular order */
        const uint32_t j = selt_ri(i), x = order[i - 1u];
        order[i - 1u] = order[j];
        order[j] = x;
    }
    memset(c, 0, SELT_SUPB * sizeof(*c));
    for (uint32_t i = 0; i < m && i < SELT_SUPB; i++) c[i] = (selt_cand){vals[order[i]], ids[order[i]]};
    uint32_t bi = 0;
    for (uint32_t i = 1; i < ng; i++)
        if (vals[i] > vals[bi] || (vals[i] == vals[bi] && ids[i] < ids[bi])) bi = i;
    h->finite = finite; h->cap = cap; h->b1 = b; h->above1 = above; h->b2 = n; h->n = n; h->t = b << 21;
    h->best = ids[bi]; h->M = lg[ids[bi]];
    if (pmode == 4) {
        for (uint32_t k = 0; k < SELT_NB; k++) {
            float acc = 0.0f;
            for (uint32_t j = k * 256u; j < k * 256u + 256u; j++)
                if (selt_fin(lg[j])) acc += expf(lg[j] - h->M);
            P[k] = acc;
        }
        return 1;
    }
    for (uint32_t i = 0; i < ng; i++) top[i] = vals[i];
    qsort(top, ng, sizeof(top[0]), selt_cmp_f_desc);
    double H = 0.0, pre[512];
    for (uint32_t i = 0; i < cap; i++) { H += (double)expf(top[i] - h->M); pre[i] = H; }
    const double S = pmode == 0 ? H / (0.80 + 0.20 * selt_u()) :
                     pmode == 1 ? H / 0.95 * (1.0 + (selt_u() - 0.5) * 8e-6) :
                     pmode == 2 ? pre[selt_ri(cap)] / 0.95 * (1.0 + (selt_u() - 0.5) * 8e-6) : 0.0;
    const int ragged = pmode != 3 && selt_u() < 0.5;
    for (uint32_t k = 0; k < SELT_NB; k++) P[k] = (float)(S / SELT_NB * (ragged ? 0.5 + selt_u() : 1.0));
    return 1;
}

/* one corruption (the reference decides what the summary then is) */
static void selt_corrupt(selt_hdr *h, float *P, selt_cand *c, float *lg, int k) {
    const uint32_t ne = h->n < SELT_SUPB ? h->n : SELT_SUPB;
    const uint32_t i = ne ? selt_ri(ne) : 0, j = ne > 1 ? (i + 1u + selt_ri(ne - 1u)) % ne : 0;
    switch (k) {
    case 0: if (ne > 1) c[j] = c[i]; break;                                       /* duplicate candidate */
    case 1: if (ne > 1) c[j].id = c[i].id; break;                                 /* duplicate id, other logit */
    case 2: c[i].id = SELT_NV + selt_ri(1000); break;                             /* id out of range */
    case 3: c[i].logit = selt_bitsf(0x7fc00000u); break;                          /* NaN candidate */
    case 4: lg[c[i].id] = selt_bitsf(selt_fbits(lg[c[i].id]) ^ (1u << selt_ri(32))); break;  /* logits disagree */
    case 5: h->above1 += 1u; break;
    case 6: h->n += 1u; break;
    case 7: h->cap -= 1u; break;
    case 8: h->finite = 0; break;
    case 9: h->M = selt_bitsf(selt_fbits(h->M) ^ 1u); break;
    case 10: P[selt_ri(SELT_NB)] = -1e-3f; break;
    case 11: P[selt_ri(SELT_NB)] = selt_bitsf(0x7f800000u); break;
    case 12: if (h->n <= 2048u) { const uint32_t d = 2049u - h->n; h->n += d; h->b2 += d;
                 if (h->finite < h->b2) h->finite = h->b2; } break;                /* superset overflow */
    default: break;
    }
}

static void check_selector_consumer(void) {
    float *lg = malloc((size_t)SELT_NV * sizeof(*lg));
    float *scratch = malloc((size_t)SELT_NV * sizeof(*scratch));
    float *P = malloc(SELT_NB * sizeof(*P));
    selt_cand *c = malloc(SELT_SUPB * sizeof(*c));
    selt_sum *ref_sum = malloc(SELT_SUPB * sizeof(*ref_sum));
    static uint32_t ids[4096], stamp[SELT_NV], epoch;
    static float vals[4096];
    CHECK(lg && scratch && P && c && ref_sum, "selector test allocation");
    if (!lg || !scratch || !P || !c || !ref_sum) return;
    selt_s = 0x5e1ec7042026ULL;
    uint64_t paths[6] = {0}, cases = 0, differ = 0, prod_cases = 0, prod_differ = 0;
    for (uint32_t it = 0; it < 30000u; it++) {
        const int fam = (int)selt_ri(5), prod = it % 10u == 0;
        uint32_t ng = selt_u() < 0.6 ? 512u + selt_ri(800) : selt_u() < 0.75 ? 1200u + selt_ri(900) : 2000u + selt_ri(900);
        int finite_all = prod || selt_u() < 0.3, clustered = 0;
        if (fam == 4) { ng = 1u + selt_ri(511); finite_all = 1; }
        /* values: 0 realistic tail, 1 heavy ties (and signed zeros), 2 a few ulps apart, 3 any finite bits */
        const float M0 = (float)(-5.0 + 45.0 * selt_u());
        const double sc = 4.0 + 12.0 * selt_u();
        float pool[8];
        const uint32_t npool = 1u + selt_ri(8);
        for (uint32_t q = 0; q < npool; q++)
            pool[q] = selt_u() < 0.15 ? (selt_u() < 0.5 ? 0.0f : -0.0f) : M0 - (float)(3.0 * selt_u());
        for (uint32_t i = 0; i < ng; i++) {
            float v;
            switch (prod ? 0 : fam) {
            case 1: v = pool[selt_ri(npool)]; clustered = it & 1u; break;
            case 2: v = selt_bitsf(selt_fbits(M0) + selt_ri(64)); clustered = 1; break;
            case 3: do v = selt_bitsf((uint32_t)selt_r64()); while (!selt_fin(v)); break;
            default: v = M0 - (float)(sc * pow(-log(selt_u()), 0.7)); break;
            }
            vals[i] = v;
        }
        epoch++;
        if (clustered) {
            const uint32_t s0 = selt_ri(SELT_NV - ng);
            for (uint32_t i = 0; i < ng; i++) ids[i] = s0 + i;
        } else {
            for (uint32_t i = 0; i < ng; i++) {
                uint32_t id;
                do id = selt_ri(SELT_NV); while (stamp[id] == epoch);
                stamp[id] = epoch;
                ids[i] = id;
            }
        }
        /* the other logits: absent (-inf) when only NG are finite, else finite and below every candidate */
        const float rest = finite_all ? -INFINITY : -3.0e38f;
        for (uint32_t i = 0; i < SELT_NV; i++) lg[i] = rest;
        for (uint32_t i = 0; i < ng; i++) lg[ids[i]] = vals[i];
        selt_hdr h;
        const int pmode = prod ? 4 : (int)selt_ri(4);
        if (!selt_build(&h, P, c, lg, ids, vals, ng, finite_all, pmode)) continue;
        if (!prod && selt_u() < 0.35) selt_corrupt(&h, P, c, lg, (int)selt_ri(13));
        const uint64_t seed = selt_r64() | 1u;
        uint64_t r_ref = seed, r_fast = seed;
        int t_ref = -1, t_fast = -1;
        const int p_ref = ds4_test_selector_consume(&h, P, c, lg, &r_ref, &t_ref);
        const uint32_t ncap = h.cap < SELT_SUPB ? h.cap : SELT_SUPB;
        if (p_ref <= 2) memcpy(ref_sum, ds4_test_selector_summary(), ncap * sizeof(*ref_sum));
        const int p_fast = ds4_test_selector_consume_fast(&h, P, c, lg, &r_fast, &t_fast);
        bool same = p_ref == p_fast && r_ref == r_fast && (p_ref != 0 || t_ref == t_fast);
        if (same && p_ref <= 2) same = !memcmp(ref_sum, ds4_test_selector_summary(), ncap * sizeof(*ref_sum));
        cases++;
        if (p_ref >= 0 && p_ref <= 5) paths[p_ref]++;
        if (!same) {
            if (differ < 5) fprintf(stderr, "selector case %u: reference path %d token %d, fast path %d token %d\n",
                                    it, p_ref, t_ref, p_fast, t_fast);
            differ++;
        }
        if (prod && p_ref == 0) {   /* a decided step must be the CPU sampler's */
            uint64_t r_cpu = seed;
            const int t_cpu = ds4_test_sample_logits(lg, SELT_NV, 1.0f, 0, 0.95f, 0.0f, &r_cpu, scratch);
            prod_cases++;
            if (t_cpu != t_ref || r_cpu != r_ref) {
                if (prod_differ < 5) fprintf(stderr, "selector case %u: decided token %d, CPU sampler %d\n", it, t_ref, t_cpu);
                prod_differ++;
            }
        }
    }
    printf("selector consumer: %llu cases (decided %llu, mass short %llu, ambiguous %llu, overflow %llu, invalid %llu,"
           " zero finite %llu), %llu fast/reference differences; %llu decided real-partial cases, %llu differ from the"
           " CPU sampler\n",
           (unsigned long long)cases, (unsigned long long)paths[0], (unsigned long long)paths[1],
           (unsigned long long)paths[2], (unsigned long long)paths[3], (unsigned long long)paths[4],
           (unsigned long long)paths[5], (unsigned long long)differ, (unsigned long long)prod_cases,
           (unsigned long long)prod_differ);
    CHECK(differ == 0, "selector fast consumer differs from the reference in %llu cases", (unsigned long long)differ);
    CHECK(prod_differ == 0, "selector decided %llu steps unlike the CPU sampler", (unsigned long long)prod_differ);
    CHECK(paths[0] > 0 && paths[1] > 0 && paths[2] > 0 && paths[3] > 0 && paths[4] > 0 && paths[5] > 0 && prod_cases > 0,
          "selector test did not reach every consumer path");
    free(lg);
    free(scratch);
    free(P);
    free(c);
    free(ref_sum);
}


/* KEYED_SAMPLING / KEYED_GPU: the position-keyed Gumbel-max draw follows softmax(logits / T) (chi-square over many
 * positions), and the finish of a GPU candidates row gives sample_keyed's token. The rows are built the way
 * metal/sampler_selector.metal kernel_kg1/kg2 build them: every finite id whose float score is within 1/64 of the row's
 * float maximum (the kernel's float score, here with logf for precise::log). */
typedef struct { uint32_t n, finite, pad[2], id[64]; } kgt_row;
static uint64_t kgt_mix(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
static float kgt_score(float l, float inv_t, uint64_t key, uint32_t id) {
    const uint64_t x = kgt_mix(key + (uint64_t)id * 0x9e3779b97f4a7c15ULL) >> 11;
    float L;
    if (x >= (UINT64_C(1) << 52)) {
        const float w = ((float)((UINT64_C(1) << 53) - 1u - x) + 0.5f) * 0x1.0p-53f;
        L = w < 0x1.0p-10f ? w * (1.0f + w * (0.5f + w * (1.0f / 3.0f))) : -logf(1.0f - w);
    } else {
        L = -logf(((float)x + 0.5f) * 0x1.0p-53f);
    }
    return l * inv_t - logf(L);
}
static void kgt_build(kgt_row *r, const float *lg, uint32_t n, float temperature, uint64_t seed, uint32_t pos) {
    const uint64_t key = kgt_mix(seed ^ kgt_mix((uint64_t)pos + 1u));
    const float inv_t = 1.0f / temperature;
    float V = -INFINITY;
    uint32_t finite = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (!(fabsf(lg[i]) <= FLT_MAX)) continue;
        finite++;
        const float v = kgt_score(lg[i], inv_t, key, i);
        if (v > V) V = v;
    }
    memset(r, 0, sizeof(*r));
    r->finite = finite;
    for (uint32_t i = 0; finite && i < n; i++) {
        if (!(fabsf(lg[i]) <= FLT_MAX) || kgt_score(lg[i], inv_t, key, i) < V - 0.015625f) continue;
        if (r->n < 64u) r->id[r->n] = i;
        r->n++;
    }
}
static uint64_t kgt_s;
static double kgt_u(void) { kgt_s = kgt_mix(kgt_s + 0x9e3779b97f4a7c15ULL); return ((double)(kgt_s >> 11) + 0.5) * 0x1.0p-53; }
/* chi-square of COUNTS against softmax(LG / T) over the bins of expected count >= 20 (the rest pooled) */
static double kgt_chi2(const uint32_t *counts, const float *lg, uint32_t n, float temperature, uint32_t draws, int *df) {
    double M = -INFINITY, S = 0.0, chi = 0.0, rest_e = 0.0, rest_o = 0.0;
    for (uint32_t i = 0; i < n; i++) if (fabsf(lg[i]) <= FLT_MAX && lg[i] > M) M = lg[i];
    for (uint32_t i = 0; i < n; i++) if (fabsf(lg[i]) <= FLT_MAX) S += exp(((double)lg[i] - M) / temperature);
    int bins = 0;
    for (uint32_t i = 0; i < n; i++) {
        const double e = fabsf(lg[i]) <= FLT_MAX ? draws * exp(((double)lg[i] - M) / temperature) / S : 0.0;
        if (e >= 20.0) { chi += (counts[i] - e) * (counts[i] - e) / e; bins++; }
        else { rest_e += e; rest_o += counts[i]; }
    }
    if (rest_e > 0.0) { chi += (rest_o - rest_e) * (rest_o - rest_e) / rest_e; bins++; }
    *df = bins - 1;
    return chi;
}

static void check_keyed(void) {
    /* 1: the draw's distribution over 300,000 positions, small vocabulary, two temperatures */
    enum { SN = 4096 };
    static float small[SN];
    static uint32_t counts[129280];
    kgt_s = 0x6b657965645f7631ULL;
    for (uint32_t i = 0; i < SN; i++) small[i] = (float)(6.0 - 3.0 * pow(-log(kgt_u()), 0.8));
    small[5] = small[9];   /* an exact tie */
    small[11] = -INFINITY;
    const float temps[2] = {1.0f, 0.7f};
    for (int ti = 0; ti < 2; ti++) {
        memset(counts, 0, sizeof(counts));
        const uint32_t draws = 300000;
        for (uint32_t pos = 0; pos < draws; pos++) {
            const int t = ds4_test_sample_keyed(small, SN, temps[ti], 0, 1.0f, 0.0f, 0x5eedULL + ti, pos, NULL);
            CHECK(t >= 0 && t < SN && t != 11, "keyed draw %d at %u", t, pos);
            if (t >= 0 && t < SN) counts[t]++;
        }
        int df = 0;
        const double chi = kgt_chi2(counts, small, SN, temps[ti], draws, &df);
        const double bound = df + 6.0 * sqrt(2.0 * df);   /* about p < 1e-7 */
        printf("keyed draw T=%.1f: chi-square %.1f, %d degrees of freedom (bound %.1f), %u draws\n",
               (double)temps[ti], chi, df, bound, draws);
        CHECK(chi < bound, "keyed draw T=%.1f chi-square %.1f over %.1f", (double)temps[ti], chi, bound);
    }
    /* 2: the GPU finish equals sample_keyed on full V4.1-size rows, and its draws follow softmax too */
    const uint32_t NV = 129280;
    float *lg = malloc(NV * sizeof(float));
    CHECK(lg != NULL, "keyed allocation");
    if (!lg) return;
    for (uint32_t i = 0; i < NV; i++) lg[i] = (float)(22.0 - 9.0 * pow(-log(kgt_u()), 0.6));
    for (uint32_t k = 0; k < 6; k++) lg[1000u + 977u * k] = 27.0f - (float)k * 0.7f;   /* a peaked head */
    lg[4321] = lg[1000];   /* tied with the maximum */
    lg[77] = -INFINITY;
    memset(counts, 0, sizeof(counts));
    uint32_t same = 0, decided = 0, overflow = 0, maxn = 0;
    const uint32_t positions = 4000;
    kgt_row row;
    for (uint32_t pos = 0; pos < positions; pos++) {
        const float T = pos & 1u ? 0.7f : 1.0f;
        const uint64_t seed = 0xabcdef12345ULL + (pos % 3u);
        kgt_build(&row, lg, NV, T, seed, pos);
        if (row.n > maxn) maxn = row.n;
        int tf = -1;
        const int path = ds4_test_keyed_finish(&row, lg, T, seed, pos, &tf);
        const int tr = ds4_test_sample_keyed(lg, NV, T, 0, 1.0f, 0.0f, seed, pos, NULL);
        if (path == 1) { overflow++; continue; }
        CHECK(path == 0, "keyed finish path %d at %u", path, pos);
        decided++;
        same += tf == tr;
        if (tf != tr && decided - same <= 5) fprintf(stderr, "keyed finish %d, sample_keyed %d at %u\n", tf, tr, pos);
        if (T == 1.0f && tf >= 0) counts[tf]++;
    }
    int df = 0;
    const double chi = kgt_chi2(counts, lg, NV, 1.0f, positions / 2u, &df);
    const double bound = df + 6.0 * sqrt(2.0 * df);
    printf("keyed GPU finish: %u of %u rows equal sample_keyed (%u overflow, at most %u candidates); "
           "T=1 chi-square %.1f, %d degrees of freedom (bound %.1f)\n",
           same, decided, overflow, maxn, chi, df, bound);
    CHECK(same == decided && decided > positions * 9u / 10u, "keyed finish differs from sample_keyed");
    CHECK(chi < bound, "keyed finish chi-square %.1f over %.1f", chi, bound);
    /* 3: a row with no candidates or a bad id is refused, an all-masked row is the argmax */
    kgt_build(&row, lg, NV, 1.0f, 1, 1);
    kgt_row bad = row;
    int t = -1;
    bad.n = 0;
    CHECK(ds4_test_keyed_finish(&bad, lg, 1.0f, 1, 1, &t) == 2, "keyed finish took an empty row");
    bad = row; bad.id[0] = NV + 5u;
    CHECK(ds4_test_keyed_finish(&bad, lg, 1.0f, 1, 1, &t) == 2, "keyed finish took an out-of-range id");
    bad = row; bad.n = 65;
    CHECK(ds4_test_keyed_finish(&bad, lg, 1.0f, 1, 1, &t) == 1, "keyed finish took an overflowing row");
    free(lg);
}

/* FULL_PROBS: an untruncated build (top_p 1, no top-k or min-p) at vocabulary size is the softmax itself: every
 * finite entry within float rounding of a double softmax, non-finite entries 0, total 1. */
static void check_full_probs(void) {
    enum { FN = 129280 };
    static float lg[FN], pr[FN];
    uint64_t x = 0x9b1f3c27d4e6a805ULL;
    for (uint32_t i = 0; i < FN; i++) {
        x = x * 6364136223846793005ULL + 1442695040888963407ULL;
        const double u = ((double)(x >> 11) + 0.5) * 0x1.0p-53;
        lg[i] = (float)(12.0 * u * u * u - 4.0);   /* a heavy head over a broad tail, like real rows */
        if (i % 9973 == 5) lg[i] = -INFINITY;
        if (i % 30011 == 7) lg[i] = NAN;
    }
    lg[4242] = 14.0f;
    const float temps[] = {1.0f, 0.7f};
    for (uint32_t t = 0; t < 2; t++) {
        const float T = temps[t];
        double mx = -INFINITY, sum = 0.0;
        for (uint32_t i = 0; i < FN; i++) if (isfinite(lg[i]) && lg[i] > mx) mx = lg[i];
        for (uint32_t i = 0; i < FN; i++) if (isfinite(lg[i])) sum += exp(((double)lg[i] - mx) / T);
        const double t0 = now_sec();
        enum { REPS = 50 };
        for (int r = 0; r < REPS; r++)
            CHECK(ds4_test_sampling_probabilities(lg, FN, T, 0, 1.0f, 0.0f, pr) != 0, "full build");
        const double us = (now_sec() - t0) * 1e6 / REPS;
        double worst = 0.0, total = 0.0;
        uint32_t bad = 0;
        for (uint32_t i = 0; i < FN; i++) {
            total += pr[i];
            if (!isfinite(lg[i])) { if (pr[i] != 0.0f) bad++; continue; }
            const double ref = exp(((double)lg[i] - mx) / T) / sum;
            const double rel = fabs(pr[i] - ref) / ref;
            if (rel > worst) worst = rel;
        }
        CHECK(bad == 0, "full build: %u non-finite logits got probability", bad);
        CHECK(worst < 2e-4, "full build T=%.1f: worst relative error %.3g", (double)T, worst);
        CHECK(fabs(total - 1.0) < 1e-4, "full build T=%.1f: total %.9f", (double)T, total);
        printf("full probabilities T=%.1f: %u entries, worst relative error %.2g vs a double softmax, total %.7f, %.0f us per build\n",
               (double)T, FN, worst, total, us);
    }
}

int main(void) {
    check_full_probs();
    check_keyed();
    check_speculative_distribution();
    check_selector_consumer();
    const uint32_t semantic_n = 4096;
    float *logits = malloc((size_t)semantic_n * sizeof(*logits));
    float *scratch = malloc((size_t)semantic_n * sizeof(*scratch));
    CHECK(logits && scratch, "semantic scratch allocation");
    if (!logits || !scratch) return 1;
    fill_logits(logits, semantic_n, 0x123456789abcdef0ULL);

    compare_case(logits, scratch, semantic_n, 1.0f, 0, 1.0f, 0.05f,
                 "default-min-p");
    compare_case(logits, scratch, semantic_n, 0.7f, 0, 1.0f, 0.0f,
                 "full-softmax");
    compare_case(logits, scratch, semantic_n, 1.3f, 0, 0.9f, 0.05f,
                 "top-p-min-p");
    compare_case(logits, scratch, semantic_n, 0.9f, 0, 0.95f, 0.0f,
                 "top-p");
    compare_case(logits, scratch, semantic_n, 0.8f, 64, 0.9f, 0.05f,
                 "top-k");
    CHECK(unsetenv("DS4_CPU_DISABLE_UNROLLED_ARGMAX") == 0,
          "select unrolled argmax default");
    compare_case(logits, scratch, semantic_n, 0.0f, 0, 1.0f, 0.05f,
                 "greedy");
    CHECK(setenv("DS4_CPU_DISABLE_UNROLLED_ARGMAX", "1", 1) == 0,
          "set scalar argmax control");
    compare_case(logits, scratch, semantic_n, 0.0f, 0, 1.0f, 0.05f,
                 "greedy-scalar-control");
    CHECK(unsetenv("DS4_CPU_DISABLE_UNROLLED_ARGMAX") == 0,
          "restore unrolled argmax default");

    const float cross_lane_tie[] = {
        -4.0f, 9.0f, -2.0f, 3.0f, 1.0f, 5.0f, 0.0f, 7.0f,
         9.0f, 4.0f,  6.0f, 2.0f, 8.0f, 1.0f, 3.0f, 0.0f,
         9.0f,
    };
    check_greedy_argmax_case(
            cross_lane_tie,
            (uint32_t)(sizeof(cross_lane_tie) / sizeof(cross_lane_tie[0])),
            1, "greedy-cross-lane-tie");

    const float tail_tie[] = {
        -3.0f, 0.0f, 8.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
         6.0f, 7.0f, 8.0f,
    };
    check_greedy_argmax_case(
            tail_tie,
            (uint32_t)(sizeof(tail_tie) / sizeof(tail_tie[0])),
            2, "greedy-tail-tie");

    const float nonfinite_greedy[] = {
        NAN, -INFINITY, -2.0e30f, INFINITY, INFINITY, 1.0f, NAN, 0.0f, 2.0f,
    };
    check_greedy_argmax_case(
            nonfinite_greedy,
            (uint32_t)(sizeof(nonfinite_greedy) /
                       sizeof(nonfinite_greedy[0])),
            3, "greedy-nonfinite");

    const float below_sentinel[] = {
        -2.0e30f, -1.5e30f, -INFINITY, NAN, -3.0e30f,
    };
    check_greedy_argmax_case(
            below_sentinel,
            (uint32_t)(sizeof(below_sentinel) / sizeof(below_sentinel[0])),
            0, "greedy-below-sentinel");

    const uint32_t excluding_n =
        (uint32_t)(sizeof(cross_lane_tie) / sizeof(cross_lane_tie[0]));
    check_excluding_argmax_case(
            cross_lane_tie, excluding_n, -1, 1, "excluding-none");
    check_excluding_argmax_case(
            cross_lane_tie, excluding_n, 1, 8, "excluding-best");
    check_excluding_argmax_case(
            cross_lane_tie, excluding_n, 8, 1, "excluding-later-tie");
    check_excluding_argmax_case(
            cross_lane_tie, excluding_n, 16, 1, "excluding-tail-tie");
    check_excluding_argmax_case(
            cross_lane_tie, excluding_n, (int)excluding_n, 1,
            "excluding-out-of-range");
    check_excluding_argmax_case(
            nonfinite_greedy,
            (uint32_t)(sizeof(nonfinite_greedy) /
                       sizeof(nonfinite_greedy[0])),
            -1, 0, "excluding-leading-nan");
    check_excluding_argmax_case(
            below_sentinel,
            (uint32_t)(sizeof(below_sentinel) / sizeof(below_sentinel[0])),
            0, 1, "excluding-first-anchor");

    /* Exercise min-p values immediately around expf's cutoff. */
    const float cutoff = logf(0.05f);
    const float boundary_logits[] = {
        0.0f,
        cutoff,
        nextafterf(cutoff, -INFINITY),
        nextafterf(cutoff, INFINITY),
        -1.0f,
        -5.0f,
        -INFINITY,
        NAN,
    };
    float boundary_scratch[sizeof(boundary_logits) / sizeof(boundary_logits[0])];
    compare_case(boundary_logits, boundary_scratch,
                 (uint32_t)(sizeof(boundary_logits) / sizeof(boundary_logits[0])),
                 1.0f, 0, 1.0f, 0.05f, "min-p-boundary");

    const float tied_logits[] = {
        2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 0.0f, -INFINITY, NAN,
    };
    float tied_scratch[sizeof(tied_logits) / sizeof(tied_logits[0])];
    compare_case(tied_logits, tied_scratch,
                 (uint32_t)(sizeof(tied_logits) / sizeof(tied_logits[0])),
                 1.0f, 0, 1.0f, 0.05f, "equal-logits-default");
    compare_case(tied_logits, tied_scratch,
                 (uint32_t)(sizeof(tied_logits) / sizeof(tied_logits[0])),
                 1.0f, 0, 0.8f, 0.05f, "equal-logits-top-p");
    compare_case(tied_logits, tied_scratch,
                 (uint32_t)(sizeof(tied_logits) / sizeof(tied_logits[0])),
                 0.01f, 0, 1.0f, 0.05f, "low-temperature");
    compare_case(tied_logits, tied_scratch,
                 (uint32_t)(sizeof(tied_logits) / sizeof(tied_logits[0])),
                 100.0f, 0, 1.0f, 0.05f, "high-temperature");
    compare_case(tied_logits, tied_scratch,
                 (uint32_t)(sizeof(tied_logits) / sizeof(tied_logits[0])),
                 1.0f, 0, 1.0f, 1.0f, "min-p-one");

    uint64_t null_ref_rng = 42;
    uint64_t null_opt_rng = 42;
    const int null_ref = reference_sample(
            tied_logits,
            (uint32_t)(sizeof(tied_logits) / sizeof(tied_logits[0])),
            1.0f, 0, 1.0f, 0.05f, &null_ref_rng);
    const int null_opt = ds4_test_sample_logits(
            tied_logits,
            (uint32_t)(sizeof(tied_logits) / sizeof(tied_logits[0])),
            1.0f, 0, 1.0f, 0.05f, &null_opt_rng, NULL);
    CHECK(null_ref == null_opt && null_ref_rng == null_opt_rng,
          "null probability scratch fallback mismatch");

    const float nonfinite_logits[] = {-INFINITY, NAN, INFINITY, NAN};
    float nonfinite_scratch[sizeof(nonfinite_logits) /
                            sizeof(nonfinite_logits[0])];
    compare_case(nonfinite_logits, nonfinite_scratch,
                 (uint32_t)(sizeof(nonfinite_logits) /
                            sizeof(nonfinite_logits[0])),
                 1.0f, 0, 1.0f, 0.05f, "all-nonfinite");

    free(logits);
    free(scratch);

    const uint32_t perf_n = 163840;
    const int iterations = 100;
    logits = malloc((size_t)perf_n * sizeof(*logits));
    scratch = malloc((size_t)perf_n * sizeof(*scratch));
    CHECK(logits && scratch, "performance scratch allocation");
    if (!logits || !scratch) return 1;
    fill_logits(logits, perf_n, 0xfeedfacecafebeefULL);

    uint64_t ref_rng = 1234;
    uint64_t checksum = 0;
    double start = now_sec();
    for (int i = 0; i < iterations; i++) {
        checksum += (uint64_t)reference_sample(logits, perf_n, 1.0f, 0,
                                               1.0f, 0.05f, &ref_rng);
    }
    const double reference_ms = (now_sec() - start) * 1000.0;

    uint64_t opt_rng = 1234;
    start = now_sec();
    for (int i = 0; i < iterations; i++) {
        checksum += (uint64_t)ds4_test_sample_logits(logits, perf_n, 1.0f, 0,
                                                     1.0f, 0.05f,
                                                     &opt_rng, scratch);
    }
    const double optimized_ms = (now_sec() - start) * 1000.0;
    CHECK(ref_rng == opt_rng, "performance RNG state mismatch");
    printf("sampling default: reference %.3f ms, optimized %.3f ms, %.2fx, checksum=%llu\n",
           reference_ms, optimized_ms,
           optimized_ms > 0.0 ? reference_ms / optimized_ms : 0.0,
           (unsigned long long)checksum);

    ref_rng = 5678;
    start = now_sec();
    for (int i = 0; i < iterations; i++) {
        checksum += (uint64_t)reference_sample(logits, perf_n, 1.0f, 0,
                                               0.9f, 0.05f, &ref_rng);
    }
    const double reference_top_p_ms = (now_sec() - start) * 1000.0;

    opt_rng = 5678;
    start = now_sec();
    for (int i = 0; i < iterations; i++) {
        checksum += (uint64_t)ds4_test_sample_logits(logits, perf_n, 1.0f, 0,
                                                     0.9f, 0.05f,
                                                     &opt_rng, scratch);
    }
    const double optimized_top_p_ms = (now_sec() - start) * 1000.0;
    CHECK(ref_rng == opt_rng, "top-p performance RNG state mismatch");
    printf("sampling top-p+min-p: reference %.3f ms, optimized %.3f ms, %.2fx, checksum=%llu\n",
           reference_top_p_ms, optimized_top_p_ms,
           optimized_top_p_ms > 0.0 ? reference_top_p_ms / optimized_top_p_ms : 0.0,
           (unsigned long long)checksum);

    free(logits);
    free(scratch);
    if (failures) {
        fprintf(stderr, "%d sampling test(s) failed\n", failures);
        return 1;
    }
    puts("sampling tests: OK");
    return 0;
}
