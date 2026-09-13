#ifndef DS4_DSPARK_CONTROLLER_H
#define DS4_DSPARK_CONTROLLER_H
/* Pure host policy. No learned constants: calibration supplies every economic
 * threshold. Costs and bin counts must describe this binary, quantization,
 * context interval and traffic population. Unknown shapes are unavailable. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static inline bool ds41_ctl_finite(double x) {
    uint64_t bits; memcpy(&bits, &x, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}
#define DS41_CTL_COSTS 128
#define DS41_CTL_BINS 1024
#define DS41_CTL_SLOTS 5

typedef struct {
    uint32_t lo, hi, rows, samples;
    double serial_ms, draft_ms, tail_ms;
} ds41_ctl_cost;
typedef struct {
    uint32_t lo, hi, slot, wins, trials;
    int bin; /* -1: unconditional-on-confidence, still prefix-surviving */
} ds41_ctl_bin;
typedef struct {
    ds41_ctl_cost costs[DS41_CTL_COSTS];
    ds41_ctl_bin bins[DS41_CTL_BINS];
    unsigned nc, nb, min_prefix, fixed_probe;
    double survival, margin, loss_ms, z;
} ds41_ctl_config;
typedef struct {
    double loss[3];
    unsigned count, next, cooldown, level;
    bool thinking;
    char tag[9];
    unsigned tag_len;
    uint64_t attempts, declines, serial_steps;
    double paid_ms;
} ds41_ctl_state;

static inline bool ds41_ctl_load(ds41_ctl_config *c, const char *path) {
    memset(c, 0, sizeof(*c));
    FILE *f = path && *path ? fopen(path, "r") : NULL;
    if (!f) return false;
    char line[512], extra;
    bool ok = true, header = false, policy = false;
    while (ok && fgets(line, sizeof(line), f)) {
        char *p = line; while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\n' || !*p) continue;
        if (!header) { header = !strcmp(p, "ds41-controller-v1\n"); ok = header; continue; }
        if (!strncmp(p, "policy ", 7)) {
            ok = !policy && sscanf(p+7, "%u %lf %lf %lf %u %lf %c",
                &c->min_prefix, &c->survival, &c->margin, &c->loss_ms,
                &c->fixed_probe, &c->z, &extra) == 6;
            policy = true;
            ok = ok && c->min_prefix >= 1 && c->min_prefix <= 5 &&
                ds41_ctl_finite(c->survival) && c->survival > 0 && c->survival <= 1 &&
                ds41_ctl_finite(c->margin) && c->margin >= 0 &&
                ds41_ctl_finite(c->loss_ms) && c->loss_ms >= 0 &&
                c->fixed_probe <= 128 && ds41_ctl_finite(c->z) && c->z >= 0;
        } else if (!strncmp(p, "cost ", 5) && c->nc < DS41_CTL_COSTS) {
            ds41_ctl_cost *v = &c->costs[c->nc];
            ok = sscanf(p+5, "%u %u %u %lf %lf %lf %u %c", &v->lo, &v->hi,
                &v->rows, &v->serial_ms, &v->draft_ms, &v->tail_ms, &v->samples, &extra) == 7;
            ok = ok && v->lo < v->hi && v->rows >= 2 && v->rows <= 6 && v->samples &&
                ds41_ctl_finite(v->serial_ms) && v->serial_ms > 0 &&
                ds41_ctl_finite(v->draft_ms) && v->draft_ms >= 0 &&
                ds41_ctl_finite(v->tail_ms) && v->tail_ms > 0;
            for (unsigned i = 0; ok && i < c->nc; i++) {
                const ds41_ctl_cost *a = &c->costs[i];
                if (a->lo < v->hi && v->lo < a->hi)
                    ok = a->rows != v->rows && a->serial_ms == v->serial_ms;
            }
            c->nc++;
        } else if (!strncmp(p, "bin ", 4) && c->nb < DS41_CTL_BINS) {
            ds41_ctl_bin *v = &c->bins[c->nb];
            ok = sscanf(p+4, "%u %u %u %d %u %u %c", &v->lo, &v->hi, &v->slot,
                         &v->bin, &v->wins, &v->trials, &extra) == 6;
            ok = ok && v->lo < v->hi && v->slot < 5 && v->bin >= -1 && v->bin < 10 &&
                v->trials && v->wins <= v->trials;
            for (unsigned i = 0; ok && i < c->nb; i++) {
                const ds41_ctl_bin *a = &c->bins[i];
                if (a->slot == v->slot && a->bin == v->bin)
                    ok = a->hi <= v->lo || v->hi <= a->lo;
            }
            c->nb++;
        } else ok = false;
    }
    ok = ok && !ferror(f) && header && policy && c->nc && c->nb;
    fclose(f);
    return ok;
}
/* Shared by generation and host boundary tests. Only target policy argmax
 * chooses acceptance; confidence cannot change this rule. */
static inline unsigned ds41_ctl_accept(unsigned rows, unsigned limit, int stop,
        bool force_reject, int (*select)(void *, unsigned), void *select_ctx,
        const int32_t *proposal, bool (*boundary)(void *, int), void *boundary_ctx,
        int *next) {
    if (!rows || !limit) return 0;
    for (unsigned j = 0; j < rows; j++) {
        int v = select(select_ctx, j);
        if (force_reject || j+1 == rows || j+1 == limit || v == stop ||
            v != proposal[j] || (boundary && boundary(boundary_ctx, v))) {
            *next = v; return j+1;
        }
    }
    return 0;
}
static inline double ds41_ctl_sigmoid(double x) {
    if (x >= 0) return 1.0 / (1.0 + exp(-x));
    const double e = exp(x); return e / (1.0 + e);
}
static inline const ds41_ctl_cost *ds41_ctl_cost_at(const ds41_ctl_config *c,
                                                    unsigned p, unsigned rows) {
    for (unsigned i = 0; i < c->nc; i++)
        if (c->costs[i].rows == rows && p >= c->costs[i].lo && p < c->costs[i].hi)
            return &c->costs[i];
    return NULL;
}
static inline double ds41_ctl_probability(const ds41_ctl_config *c, unsigned p,
                                           unsigned slot, double logit, bool confidence) {
    if (confidence && !ds41_ctl_finite(logit)) return -1;
    int bin = confidence ? (int)(ds41_ctl_sigmoid(logit) * 10.0) : -1;
    if (bin == 10) bin = 9;
    for (unsigned i = 0; i < c->nb; i++) {
        const ds41_ctl_bin *b = &c->bins[i];
        if (b->slot != slot || b->bin != bin || p < b->lo || p >= b->hi) continue;
        /* Wilson lower bound, using ONLY observations where all earlier
         * proposals survived. z=0 explicitly requests the point estimate. */
        const double n = b->trials, a = b->wins / n, zz = c->z*c->z;
        return (a + zz/(2*n) - c->z*sqrt(a*(1-a)/n + zz/(4*n*n))) / (1+zz/n);
    }
    return -1;
}
/* Draft cost is sunk at this call. Rank measured widths by full-cycle rate,
 * but compare continuation with decline incrementally. Declining solely because
 * the already-paid draft made this attempt a loss would compound that loss.
 * The three-attempt ledger decides whether to pay for the NEXT draft. */
static inline unsigned ds41_ctl_choose(const ds41_ctl_config *c, unsigned p,
        unsigned cap, unsigned remaining, const float logit[5], bool confidence,
        bool widths, double paid_draft_ms) {
    unsigned best = 0; double best_rate = 0;
    for (unsigned r = 2; r <= cap; r++) {
        if (!widths && r != cap) continue;
        const ds41_ctl_cost *v = ds41_ctl_cost_at(c, p, r);
        if (!v || r <= c->min_prefix || remaining <= c->min_prefix) continue;
        double survival = 1, expected = 1; bool useful = false, known = true;
        for (unsigned j = 0; j+1 < r && j+1 < remaining; j++) {
            double a = ds41_ctl_probability(c, p, j, logit[j], confidence);
            if (a < 0) { known = false; break; }
            survival *= a; expected += survival;
            if (j+1 == c->min_prefix) useful = survival >= c->survival;
        }
        const double full = fmax(v->draft_ms, paid_draft_ms) + v->tail_ms;
        if (!known || !useful || expected*v->serial_ms <= v->tail_ms*(1+c->margin)) continue;
        const double rate = expected/full;
        if (rate > best_rate) { best_rate = rate; best = r; }
    }
    return best;
}
static inline bool ds41_ctl_preflight(const ds41_ctl_config *c, ds41_ctl_state *s,
        unsigned p, unsigned cap, unsigned remaining, bool widths) {
    if (s->thinking) return false;
    if (s->cooldown) { s->cooldown--; return false; }
    for (unsigned r = 2; r <= cap; r++) {
        if ((!widths && r != cap) || r <= c->min_prefix || remaining <= c->min_prefix) continue;
        const ds41_ctl_cost *v = ds41_ctl_cost_at(c, p, r);
        unsigned useful = remaining < r ? remaining : r;
        /* Impossible even at 100% acceptance: do not pay a draft. */
        if (v && useful*v->serial_ms > (v->draft_ms+v->tail_ms)*(1+c->margin)) return true;
    }
    return false;
}
static inline void ds41_ctl_record(const ds41_ctl_config *c, ds41_ctl_state *s,
                                   double wall, unsigned committed, double serial,
                                   bool declined) {
    s->attempts++; s->declines += declined; s->paid_ms += wall;
    s->loss[s->next] = wall - committed*serial;
    s->next = (s->next+1)%3; if (s->count < 3) s->count++;
    if (s->count < 3) return;
    double loss = s->loss[0]+s->loss[1]+s->loss[2];
    if (loss > c->loss_ms) {
        s->cooldown = c->fixed_probe ? c->fixed_probe : 16u << s->level;
        if (s->level < 3) s->level++;
        s->count = s->next = 0;
    } else if (loss <= 0) s->level = 0;
}
/* Feed only committed assistant output bytes. Supports split delimiters.
 * The caller previews this state during acceptance and stops at an opening
 * tag before processing any token inside the reasoning span. */
static inline void ds41_ctl_text(ds41_ctl_state *s, const char *text, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (s->tag_len == sizeof(s->tag)) {
            memmove(s->tag, s->tag+1, sizeof(s->tag)-1); s->tag_len--;
        }
        s->tag[s->tag_len++] = text[i];
        if (s->tag_len >= 7 && !memcmp(s->tag+s->tag_len-7, "<think>", 7)) s->thinking = true;
        if (s->tag_len >= 8 && !memcmp(s->tag+s->tag_len-8, "</think>", 8)) {
            s->thinking = false; s->cooldown = s->level = s->count = s->next = 0;
        }
    }
}
#endif
