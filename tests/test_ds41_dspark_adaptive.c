/* Host policy tests for the ported windowed controller; no model, GPU or
 * timing measurement.  Ported from `tests/test_dflash_windowed.c` on this
 * project's GLM-5.3-Flash branch with V4.1 constants: the full trained block
 * and no width search; confidence admission is ported and covered here. */
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include "../ds4_ds41_dspark_adaptive.h"

static void cycle(ds41_dspark_adaptive *a, double ms, uint32_t rows) {
    ds41_adapt_record(a, ms, rows, true);
}
static void serial(ds41_dspark_adaptive *a, double ms) {
    ds41_adapt_record(a, ms, 1u, false);
}
static void decline(ds41_dspark_adaptive *a, double ms) {
    ds41_adapt_record_decline(a, ms, 1u);
}

int main(void) {
    const ds41_adapt_config c = {16u, true, true, 0.75f, 3u, false};
    ds41_dspark_adaptive a;
    ds41_adapt_begin(&a, c);
    assert(a.active && !a.invalid && !a.engaged);

    /* Entry: a request starts serial until sixteen tokens are consumed. */
    assert(!ds41_adapt_admit(&a) && ds41_adapt_entry_wait(&a));
    for (unsigned i = 0; i < 15; i++) serial(&a, 32.0);
    assert(!ds41_adapt_admit(&a));
    serial(&a, 32.0);
    assert(!ds41_adapt_entry_wait(&a) && ds41_adapt_admit(&a));
    /* Sixteen identical samples leave the median exactly there. */
    assert(fabs(a.serial_ms - 32.0) < 1e-9 && a.serial_steps == 16);
    /* The window is nine deep; one jittery token cannot move the decision. */
    serial(&a, 3200.0);
    assert(fabs(a.serial_ms - 32.0) < 1e-9);

    /* A winning window of three engages and clears any cooldown. */
    cycle(&a, 121.0, 6u);
    assert(a.window_calls == 1 && !a.skip_remaining && !a.engaged);
    cycle(&a, 121.0, 6u);
    assert(a.window_calls == 2 && !a.skip_remaining);
    cycle(&a, 121.0, 6u);
    assert(a.engaged && !a.skip_remaining && !a.window_calls && a.windows == 1);
    assert(a.attempts == 3 && !a.losing_cycles && a.net_ms < 0.0);

    /* Losing windows back off 16, 32, 64, 128 and stop escalating there. */
    for (unsigned episode = 0; episode < 5; episode++) {
        for (unsigned i = 0; i < 3; i++) cycle(&a, 400.0, 2u);
        const uint32_t expect = episode < 4 ? 16u << episode : 128u;
        assert(a.skip_remaining == expect && !a.engaged);
        a.skip_remaining = 0;  /* simulate the cooldown being consumed */
    }
    assert(a.backoffs == 5 && a.losing_cycles == 15);

    /* Cooldown is spent one serial step at a time and blocks admission. */
    a.bad_run = 0;
    for (unsigned i = 0; i < 3; i++) cycle(&a, 400.0, 2u);
    assert(a.skip_remaining == 16 && !ds41_adapt_admit(&a));
    for (unsigned i = 0; i < 16; i++) { assert(!ds41_adapt_admit(&a)); serial(&a, 32.0); }
    assert(!a.skip_remaining && ds41_adapt_admit(&a) && a.skipped_steps == 16);

    /* Reasoning spans decode serially, and leaving one clears the cooldown. */
    a.skip_remaining = 128; a.bad_run = 3; a.window_calls = 2;
    ds41_adapt_reasoning(&a, true);
    assert(!ds41_adapt_admit(&a) && a.skip_remaining == 128);
    ds41_adapt_reasoning(&a, false);
    assert(!a.skip_remaining && !a.bad_run && !a.window_calls && ds41_adapt_admit(&a));

    /* Admission: the longest confident run from the anchor, nothing below the
     * floor, and a run shorter than min_draft is no admission at all. */
    const float conf[5] = {0.99f, 0.80f, 0.74f, 0.99f, 0.99f};
    assert(ds41_adapt_prefix(conf, 5u, 0.75f, 2u) == 2u);
    assert(ds41_adapt_prefix(conf, 5u, 0.75f, 3u) == 0u);
    const float sure[5] = {0.99f, 0.99f, 0.99f, 0.99f, 0.99f};
    assert(ds41_adapt_prefix(sure, 5u, 0.75f, 3u) == 5u);
    const float nan_run[2] = {0.99f, 0.0f / 0.0f};
    assert(ds41_adapt_prefix(nan_run, 2u, 0.75f, 1u) == 1u);
    assert(!ds41_adapt_prefix(sure, 5u, 0.0f / 0.0f, 1u) && !ds41_adapt_prefix(NULL, 5u, 0.75f, 1u));

    /* A decline is a chosen call: it takes a window slot at its drafter cost,
     * three of them back off, and none of them is an attempt or a losing cycle. */
    ds41_dspark_adaptive g;
    ds41_adapt_begin(&g, c);
    for (unsigned i = 0; i < 16; i++) serial(&g, 32.0);
    decline(&g, 40.0); decline(&g, 40.0);
    assert(g.window_calls == 2 && !g.skip_remaining);
    decline(&g, 40.0);
    assert(g.skip_remaining == 16 && g.backoffs == 1 && g.declines == 3);
    assert(!g.attempts && !g.losing_cycles && g.serial_consumed == 19);

    /* With the controller off every eligible position is proposed. */
    ds41_adapt_config off = c; off.enabled = false;
    ds41_dspark_adaptive b;
    ds41_adapt_begin(&b, off);
    assert(ds41_adapt_admit(&b));
    for (unsigned i = 0; i < 9; i++) cycle(&b, 400.0, 1u);
    assert(ds41_adapt_admit(&b));

    /* With reasoning_serial off the controller proposes inside think. */
    ds41_adapt_config think = c; think.reasoning_serial = false;
    ds41_dspark_adaptive d;
    ds41_adapt_begin(&d, think);
    for (unsigned i = 0; i < 16; i++) serial(&d, 32.0);
    ds41_adapt_reasoning(&d, true);
    assert(ds41_adapt_admit(&d));

    /* Faults: a zero-row step, a non-finite wall, an out-of-range entry. */
    ds41_dspark_adaptive e;
    ds41_adapt_begin(&e, c);
    ds41_adapt_record(&e, 32.0, 0u, false);
    assert(e.invalid && !ds41_adapt_admit(&e));
    ds41_adapt_begin(&e, c);
    volatile uint64_t word = UINT64_C(0x7ff0000000000000);
    double bad; memcpy(&bad, (const void *)&word, sizeof(bad));
    assert(!ds41_adapt_timing(bad));
    ds41_adapt_record(&e, bad, 1u, false);
    assert(e.invalid);
    ds41_adapt_config wide = c; wide.min_serial_tokens = DS41_ADAPT_ENTRY_MAX + 1u;
    ds41_adapt_begin(&e, wide);
    assert(e.invalid && !ds41_adapt_admit(&e));

    /* No window can be judged before a serial step has priced one. */
    ds41_dspark_adaptive f;
    ds41_adapt_config now = c; now.min_serial_tokens = 0;
    ds41_adapt_begin(&f, now);
    assert(ds41_adapt_admit(&f));
    for (unsigned i = 0; i < 5; i++) cycle(&f, 400.0, 1u);
    assert(!f.window_calls && !f.skip_remaining && f.attempts == 5);

    /* Evidence resets drop measured costs but keep the request's cooldown. */
    a.skip_remaining = 64;
    ds41_adapt_reset_evidence(&a);
    assert(a.skip_remaining == 64 && a.serial_ms == 0.0 && !a.serial_count);

    /* Marginal losses accumulate: four +4 ms windows exhaust a 16 ms budget
     * only on the fifth window. A clear loss still backs off immediately. */
    ds41_adapt_config tolerant = c; tolerant.loss_budget = true;
    ds41_dspark_adaptive h;
    ds41_adapt_begin(&h, tolerant);
    for (unsigned i = 0; i < 16; i++) serial(&h, 32.0);
    for (unsigned window = 0; window < 4; window++) {
        cycle(&h, 128.0, 4); cycle(&h, 128.0, 4); cycle(&h, 132.0, 4);
        assert(!h.skip_remaining && h.loss_debt_ms == 4.0 * (window + 1));
    }
    cycle(&h, 128.0, 4); cycle(&h, 128.0, 4); cycle(&h, 132.0, 4);
    assert(h.skip_remaining == 16 && h.backoffs == 1 && h.loss_debt_ms == 0.0);
    assert(h.net_ms == 20.0); /* accounting never hides the tolerated losses */
    h.skip_remaining = 0;
    decline(&h, 40.0); decline(&h, 40.0); decline(&h, 40.0);
    assert(h.skip_remaining == 32 && h.backoffs == 2);

    /* A profitable follow-up repays a near-break-even window; large earlier
     * gains cannot subsidize an unlimited run of future losses. */
    ds41_adapt_begin(&h, tolerant);
    for (unsigned i = 0; i < 16; i++) serial(&h, 32.0);
    cycle(&h, 129.0, 4); cycle(&h, 129.0, 4); cycle(&h, 129.0, 4);
    assert(h.loss_debt_ms == 3.0 && !h.skip_remaining);
    cycle(&h, 120.0, 4); cycle(&h, 120.0, 4); cycle(&h, 120.0, 4);
    assert(h.loss_debt_ms == 0.0 && !h.skip_remaining);
    h.loss_debt_ms = 10.0;
    ds41_adapt_reset_evidence(&h);
    assert(h.loss_debt_ms == 0.0);
    h.loss_debt_ms = 10.0; h.reasoning = true;
    ds41_adapt_reasoning(&h, false);
    assert(h.loss_debt_ms == 0.0);

    puts("V4.1 DSpark windowed admission: PASS");
    return 0;
}
