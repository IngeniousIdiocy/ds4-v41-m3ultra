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
 * serial step: a window whose net wall time is below serial engages and clears
 * any cooldown, a window that is not backs off for 16, then 32, 64 and 128
 * consumed serial tokens before proposing again.  The serial step cost is the
 * median of the last nine measured serial tokens, so one jittery token cannot
 * move the decision; the full cycle carries an EMA for reporting.  A request
 * starts serial for sixteen tokens so the first window is judged against a
 * measured cost rather than a guess.  `<think>` spans decode serially by
 * default, and leaving reasoning clears the cooldown.
 *
 * What differs from GLM: V4.1 verifies the full trained block, so there is no
 * width search, no `n_min/n_max/n_start` and no per-width cost table; and there
 * is no confidence admission, because the only calibration V4.1's confidence
 * probe ever had was the Wilson bin file this controller replaces, and reading
 * it costs a GPU readback inside the proposal.  Every eligible position is
 * proposed and the window is what dials it back. */
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
} ds41_adapt_config;

typedef struct {
    ds41_adapt_config config;
    bool active, invalid, engaged, reasoning;
    uint32_t bad_run, skip_remaining, window_calls;
    double window_net_ms;
    double serial_ms;            /* median of the last nine serial tokens */
    double serial_samples[DS41_ADAPT_SERIAL_WINDOW];
    uint32_t serial_count, serial_next;
    double cycle_ms;             /* EMA of the full verify cycle, reporting only */
    double net_ms;               /* request-cumulative wall minus rows x serial */
    uint64_t serial_consumed;
    uint64_t calls, attempts, serial_steps, skipped_steps;
    uint64_t losing_cycles, windows, backoffs, committed;
} ds41_dspark_adaptive;

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
 * time or percentage allowance is required to try again. */
static inline void ds41_adapt_window_feedback(ds41_dspark_adaptive *a,
                                              uint32_t consumed, double wall_ms) {
    if (a->serial_ms <= 0.0) return;
    const double net = wall_ms - (double)consumed * a->serial_ms;
    a->net_ms += net;
    a->window_net_ms += net;
    a->window_calls++;
    a->losing_cycles += net >= 0.0;
    if (a->window_calls < DS41_ADAPT_WINDOW) return;
    a->windows++;
    if (a->window_net_ms < 0.0) {
        a->engaged = true;
        a->bad_run = a->skip_remaining = 0u;
    } else {
        a->engaged = false;
        if (a->bad_run < DS41_ADAPT_BACKOFF_MAX) a->bad_run++;
        a->skip_remaining = DS41_ADAPT_BACKOFF_BASE << (a->bad_run - 1u);
        a->backoffs++;
    }
    a->window_calls = 0u;
    a->window_net_ms = 0.0;
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
        ds41_adapt_window_feedback(a, rows, wall_ms);
    } else {
        a->serial_steps++;
        a->serial_consumed += rows;
        ds41_adapt_serial_sample(a, wall_ms / (double)rows);
        if (a->skip_remaining) { a->skipped_steps++; a->skip_remaining--; }
    }
}
#endif
