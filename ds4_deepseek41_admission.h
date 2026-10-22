#ifndef DS4_DEEPSEEK41_ADMISSION_H
#define DS4_DEEPSEEK41_ADMISSION_H
/* When DeepSeek V4.1's DSpark drafts on Metal, by measured cost.
 *
 * Drafting is tried wherever it is allowed. Three complete attempts are judged together against the measured cost of
 * one serial token: a window that saves time keeps drafting and clears any back-off. Losses carry over until they
 * exceed half a serial token, and then drafting backs off for 16 serial tokens, then 32, 64 and 128. A serial token
 * costs the median of the last nine. A request decodes its first 16 tokens serially, so the first window is judged
 * against a measured cost.
 *
 * A drafted block is verified whole only when its first three positions are each at least 75% confident. Otherwise it
 * is declined before the verify: that costs the drafter and one serial token instead of a six-row pass. A decline takes
 * its place in the window at that cost, so content where every draft is declined backs off too. */
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#define DS41_ADMIT_FIRST_SERIAL 16u   /* serial tokens of a request before its first draft */
#define DS41_ADMIT_SAMPLES 9u         /* serial tokens in the median that prices one */
#define DS41_ADMIT_WINDOW 3u          /* attempts judged together */
#define DS41_ADMIT_BACKOFF 16u        /* serial tokens after a losing window, doubling up to 128 */
#define DS41_ADMIT_BACKOFF_STEPS 4u
#define DS41_ADMIT_CONFIDENT 3u       /* leading draft positions that must be confident */
#define DS41_ADMIT_CONFIDENCE 0.75f

typedef struct {
    uint32_t backoffs, skip, window_calls, sample_count, sample_next;
    uint64_t serial_tokens, attempts;
    double window_ms, debt_ms, serial_ms, samples[DS41_ADMIT_SAMPLES];
} ds41_admission;

static inline void ds41_admit_begin(ds41_admission *a) {
    memset(a, 0, sizeof(*a));
}

/* Whether to draft at this position. */
static inline bool ds41_admit_draft(const ds41_admission *a) {
    return !a->skip && (a->attempts || a->serial_tokens >= DS41_ADMIT_FIRST_SERIAL);
}

/* Whether a drafted block at draft position `row` with this confidence may still be verified. */
static inline bool ds41_admit_confident(uint32_t row, float confidence) {
    return row >= DS41_ADMIT_CONFIDENT || confidence >= DS41_ADMIT_CONFIDENCE;
}

/* One serial token took `ms`. */
static inline void ds41_admit_serial(ds41_admission *a, double ms) {
    a->serial_tokens++;
    if (a->skip) a->skip--;
    a->samples[a->sample_next] = ms;
    a->sample_next = (a->sample_next + 1u) % DS41_ADMIT_SAMPLES;
    if (a->sample_count < DS41_ADMIT_SAMPLES) a->sample_count++;
    double sorted[DS41_ADMIT_SAMPLES];
    for (uint32_t i = 0; i < a->sample_count; i++) {
        uint32_t j = i;
        for (; j && sorted[j - 1u] > a->samples[i]; j--) sorted[j] = sorted[j - 1u];
        sorted[j] = a->samples[i];
    }
    const uint32_t mid = a->sample_count / 2u;
    a->serial_ms = a->sample_count & 1u ? sorted[mid] : 0.5 * (sorted[mid - 1u] + sorted[mid]);
}

/* One drafted cycle took `ms` and produced `rows` tokens: a verified block, or a declined one that decoded its first
 * token serially. */
static inline void ds41_admit_cycle(ds41_admission *a, double ms, uint32_t rows, bool verified) {
    if (verified) a->attempts++;
    else a->serial_tokens += rows;
    if (a->serial_ms <= 0.0) return;
    a->window_ms += ms - rows * a->serial_ms;
    if (++a->window_calls < DS41_ADMIT_WINDOW) return;
    a->debt_ms = a->debt_ms + a->window_ms > 0.0 ? a->debt_ms + a->window_ms : 0.0;   /* wins repay, never bank */
    if (a->window_ms < 0.0) {
        a->backoffs = a->skip = 0;
    } else if (a->debt_ms > 0.5 * a->serial_ms) {
        if (a->backoffs < DS41_ADMIT_BACKOFF_STEPS) a->backoffs++;
        a->skip = DS41_ADMIT_BACKOFF << (a->backoffs - 1u);
        a->debt_ms = 0.0;
    }
    a->window_calls = 0;
    a->window_ms = 0.0;
}
#endif
