#ifndef DS4_DS41_DSPARK_ADAPTIVE_H
#define DS4_DS41_DSPARK_ADAPTIVE_H
/* V4.1 DSpark admission: the GLM-5.3 DFlash2 windowed cost-feedback
 * controller, ported.  Source of truth is `ds4_dflash_adaptive.h` on this
 * project's GLM-5.3-Flash branch — `dflash_adaptive_window_feedback`,
 * `dflash_adaptive_serial_sample`, `dflash_adaptive_backoff`'s windowed arm,
 * `dflash_adaptive_reasoning` and the entry wait of `dflash_adaptive_limit`.
 *
 * The policy, in one paragraph: PROPOSE BY DEFAULT at every eligible greedy
 * position.  Three complete attempts are judged together against the measured
 * serial step: a winning window engages and clears any cooldown. Small losses
 * accumulate without banking earlier wins; exceeding half one serial step
 * backs off for 16, then 32, 64 and 128 consumed serial tokens. The diagnostic
 * loss_budget=0 arm restores immediate backoff on a non-winning window. The
 * serial step cost is the median of the last nine measured serial tokens, so one jittery token cannot
 * move the decision; the full cycle carries an EMA for reporting.  A request
 * starts serial for sixteen tokens so the first window is judged against a
 * measured cost rather than a guess.  `<think>` spans decode serially by
 * default, and leaving reasoning clears the cooldown.
 *
 * Confidence admission is GLM's too (`dflash_adaptive_prefix`): the admitted
 * prefix is the longest run of drafted positions from the anchor whose
 * confidence is at least `p_min`, and a proposal whose run is shorter than
 * `min_draft` is DECLINED before the verify, paying the drafter and one serial
 * step instead of a full six-row target pass.  GLM declines below 4 of its 7
 * draft positions; V4.1 has 5, so 3 of 5 (0.60) is GLM's 0.57 carried over.
 *
 * A decline is accounted exactly as GLM accounts one, and the accounting is the
 * whole subtlety.  `dflash_adaptive_window_feedback` runs for EVERY chosen
 * call, declines included: a declined probe is a chosen call whose wall is the
 * drafter plus one serial token and whose consumed rows are 1, so its net is a
 * loss of the drafter cost, and three of them close a losing window and back
 * off.  That is the mechanism, not a flaw -- it is why GLM skips 79 % of
 * positions on hostile content.  What a decline is NOT is a rejection: it never
 * asked the target to reject anything, so `losing_cycles` counts only verified
 * calls.  An earlier build of this port kept declines out of the window
 * entirely and measured -15 % on prose, because nothing then throttled probing
 * on content where every proposal declines.
 *
 * What still differs from GLM: V4.1 verifies the full trained block, so there
 * is no width search, no `n_min/n_max/n_start` and no per-width cost table.
 * The admitted prefix gates the cycle; it never shortens it. */
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#define DS41_ADAPT_SERIAL_WINDOW 9u    /* GLM DS4_DFLASH_SERIAL_WINDOW */
#define DS41_ADAPT_ENTRY_MAX    64u    /* GLM DS4_DFLASH_ENTRY_MAX */
#define DS41_ADAPT_WINDOW        3u    /* complete attempts judged together */
#define DS41_ADAPT_BACKOFF_MAX   4u    /* 16 << 0..3 = 16, 32, 64, 128 */
#define DS41_ADAPT_BACKOFF_BASE 16u
#define DS41_ADAPT_MS_LIMIT   6.0e7    /* generous bound; anything past it is a fault */

typedef struct {
    uint32_t min_serial_tokens;  /* consumed serial tokens before the first draft */
    bool reasoning_serial;       /* decode <think> serially */
    bool enabled;                /* false: propose everywhere, never back off */
    float p_min;                 /* per-position confidence floor (GLM's 0.75) */
    uint32_t min_draft;          /* admitted prefix below this declines the cycle */
    bool loss_budget;            /* bounded cumulative loss before backoff */
} ds41_adapt_config;

typedef struct {
    ds41_adapt_config config;
    bool active, invalid, engaged, reasoning;
    uint32_t bad_run, skip_remaining, window_calls;
    double window_net_ms;
    double loss_debt_ms;         /* unrepaid window losses; wins repay, never bank credit */
    double serial_ms;            /* median of the last nine serial tokens */
    double serial_samples[DS41_ADAPT_SERIAL_WINDOW];
    uint32_t serial_count, serial_next;
    double cycle_ms;             /* EMA of the full verify cycle, reporting only */
    double net_ms;               /* request-cumulative wall minus rows x serial */
    uint64_t serial_consumed;
    uint64_t calls, attempts, serial_steps, skipped_steps;
    uint64_t losing_cycles, windows, backoffs, committed, declines;
} ds41_dspark_adaptive;

/* GLM's `dflash_adaptive_probability` / `dflash_adaptive_prefix`, verbatim in
 * behaviour: the run stops at the first position whose confidence is missing or
 * below the floor, and a run shorter than `n_min` is no admission at all. */
static inline bool ds41_adapt_probability(float p) {
    return p == p && p >= 0.0f && p <= 1.0f;
}

static inline uint32_t ds41_adapt_prefix(const float *confidence, uint32_t proposed,
                                         float p_min, uint32_t n_min) {
    if (!ds41_adapt_probability(p_min) || !confidence) return 0u;
    uint32_t n = 0;
    while (n < proposed && ds41_adapt_probability(confidence[n]) &&
           confidence[n] >= p_min) n++;
    return n >= n_min ? n : 0u;
}

static inline bool ds41_adapt_timing(double ms) {
    uint64_t bits; memcpy(&bits, &ms, sizeof(bits));
    if ((bits & UINT64_C(0x7ff0000000000000)) == UINT64_C(0x7ff0000000000000)) return false;
    return ms >= 0.0 && ms <= DS41_ADAPT_MS_LIMIT;
}

/* Evidence measured against one conditioning never prices another: the serial
 * window, the cycle EMA and the open window are dropped when the session's
 * prefix stops being an extension of what it was.  Cooldown is request policy
 * and survives, exactly as GLM keeps `skip_remaining` across an evidence reset. */
static inline void ds41_adapt_reset_evidence(ds41_dspark_adaptive *a) {
    a->serial_ms = a->cycle_ms = 0.0;
    a->serial_count = a->serial_next = 0;
    memset(a->serial_samples, 0, sizeof(a->serial_samples));
    a->window_calls = 0;
    a->window_net_ms = 0.0;
    a->loss_debt_ms = 0.0;
    a->engaged = false;
}

static inline void ds41_adapt_begin(ds41_dspark_adaptive *a, ds41_adapt_config c) {
    memset(a, 0, sizeof(*a));
    a->config = c;
    a->active = true;
    a->invalid = c.min_serial_tokens > DS41_ADAPT_ENTRY_MAX;
    ds41_adapt_reset_evidence(a);
}

/* A request starts serial so the first window is judged against a measured
 * serial cost rather than a guess.  GLM's `min_serial_tokens`, same default. */
static inline bool ds41_adapt_entry_wait(const ds41_dspark_adaptive *a) {
    return !a->attempts && a->serial_consumed < a->config.min_serial_tokens;
}

static inline bool ds41_adapt_skip(const ds41_dspark_adaptive *a) {
    return a->skip_remaining != 0;
}

static inline bool ds41_adapt_admit(const ds41_dspark_adaptive *a) {
    if (!a->active || a->invalid) return false;
    if (!a->config.enabled) return true;
    if (a->config.reasoning_serial && a->reasoning) return false;
    return !ds41_adapt_entry_wait(a) && !ds41_adapt_skip(a);
}

/* Leaving a reasoning span clears the cooldown: the evidence that produced it
 * was measured on text the answer span does not have to resemble. */
static inline void ds41_adapt_reasoning(ds41_dspark_adaptive *a, bool inside) {
    if (a->reasoning && !inside) {
        a->window_calls = a->bad_run = a->skip_remaining = 0u;
        a->window_net_ms = 0.0;
        a->loss_debt_ms = 0.0;
        a->engaged = false;
    }
    a->reasoning = inside;
}

/* A bounded central reference resists isolated serial jitter (GLM verbatim):
 * the median of the last nine measured serial tokens. */
static inline void ds41_adapt_serial_sample(ds41_dspark_adaptive *a, double ms) {
    if (!ds41_adapt_timing(ms) || ms <= 0.0) { a->invalid = true; return; }
    a->serial_samples[a->serial_next] = ms;
    a->serial_next = (a->serial_next + 1u) % DS41_ADAPT_SERIAL_WINDOW;
    if (a->serial_count < DS41_ADAPT_SERIAL_WINDOW) a->serial_count++;
    double ordered[DS41_ADAPT_SERIAL_WINDOW];
    for (uint32_t i = 0; i < a->serial_count; i++) {
        uint32_t j = i;
        while (j && ordered[j - 1u] > a->serial_samples[i]) { ordered[j] = ordered[j - 1u]; j--; }
        ordered[j] = a->serial_samples[i];
    }
    const uint32_t middle = a->serial_count / 2u;
    a->serial_ms = a->serial_count & 1u ? ordered[middle]
                                        : 0.5 * (ordered[middle - 1u] + ordered[middle]);
}

/* Judge complete attempts in small windows so an entry cost can be repaid by
 * consecutive verification.  Losing windows back off by token count; no saved
 * time or percentage allowance is required to try again.  This is GLM's
 * `dflash_adaptive_window_feedback`.
 *
 * GLM splits the call in two -- the step declares what it returned, a later ACK
 * confirms how much the request consumed, and the window is priced only when
 * the two agree.  V4.1 has no such gap: `ds41_dspark_step_core` commits inside
 * itself and returns the advance it just made, so the declaration IS the
 * acknowledgement and the guard could never fire.  It is documented here rather
 * than coded as unreachable state.
 *
 * `verified` separates the two kinds of chosen call.  Both are priced -- a
 * declined probe really did cost the drafter and really should discourage the
 * next one -- but only a call that asked the target to check something can be
 * counted as a losing cycle. The bounded loss budget extends the GLM decision
 * rule without changing this accounting. */
static inline void ds41_adapt_window_feedback(ds41_dspark_adaptive *a, uint32_t consumed,
                                              double wall_ms, bool verified) {
    if (a->serial_ms <= 0.0) return;
    const double net = wall_ms - (double)consumed * a->serial_ms;
    a->net_ms += net;
    a->window_net_ms += net;
    a->window_calls++;
    a->losing_cycles += verified && net >= 0.0;
    if (a->window_calls < DS41_ADAPT_WINDOW) return;
    a->windows++;
    /* A tiny measured loss must not immediately discard the next MTP window.
     * Carry losses until they exceed half one serial step. Wins repay this
     * debt without banking credit, so repeated marginal losses still back off.
     * The extra exploration cost before a losing window is bounded by that
     * half-step budget; net_ms continues to report the actual measured cost. */
    if (a->config.loss_budget) {
        a->loss_debt_ms += a->window_net_ms;
        if (a->loss_debt_ms < 0.0) a->loss_debt_ms = 0.0;
    }
    const bool within_budget = a->config.loss_budget &&
                               a->loss_debt_ms <= 0.5 * a->serial_ms;
    if (a->window_net_ms < 0.0 || within_budget) {
        a->engaged = true;
        if (a->window_net_ms < 0.0) a->bad_run = a->skip_remaining = 0u;
    } else {
        a->engaged = false;
        if (a->bad_run < DS41_ADAPT_BACKOFF_MAX) a->bad_run++;
        a->skip_remaining = DS41_ADAPT_BACKOFF_BASE << (a->bad_run - 1u);
        a->backoffs++;
        a->loss_debt_ms = 0.0;
    }
    a->window_calls = 0u;
    a->window_net_ms = 0.0;
}

/* A proposal declined on confidence.  It paid the drafter and produced one
 * serial token.  GLM's accounting, exactly: a chosen call, so it takes a window
 * slot at its drafter cost and three of them back the controller off; not an
 * attempt and never a losing cycle, since it asked the target nothing; and its
 * wall never becomes a serial-cost sample -- it carries the drafter, and pricing
 * serial with it would make every later cycle look cheap. */
static inline void ds41_adapt_record_decline(ds41_dspark_adaptive *a, double wall_ms,
                                             uint32_t rows) {
    if (!a->active) return;
    if (!rows || !ds41_adapt_timing(wall_ms)) { a->invalid = true; return; }
    a->calls++;
    a->committed += rows;
    a->declines++;
    a->serial_consumed += rows;
    ds41_adapt_window_feedback(a, rows, wall_ms, false);
}

/* One decode step has finished.  `rows` is what it produced, `drafted` says
 * whether it was a verify cycle or an ordinary serial token. */
static inline void ds41_adapt_record(ds41_dspark_adaptive *a, double wall_ms,
                                     uint32_t rows, bool drafted) {
    if (!a->active) return;
    if (!rows || !ds41_adapt_timing(wall_ms)) { a->invalid = true; return; }
    a->calls++;
    a->committed += rows;
    if (drafted) {
        a->attempts++;
        a->cycle_ms = a->cycle_ms > 0.0 ? 0.8 * a->cycle_ms + 0.2 * wall_ms : wall_ms;
        ds41_adapt_window_feedback(a, rows, wall_ms, true);
    } else {
        a->serial_steps++;
        a->serial_consumed += rows;
        ds41_adapt_serial_sample(a, wall_ms / (double)rows);
        if (a->skip_remaining) { a->skipped_steps++; a->skip_remaining--; }
    }
}
#endif
