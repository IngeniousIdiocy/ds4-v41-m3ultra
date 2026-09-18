#ifndef DS4_DSPARK_CONTROLLER_H
#define DS4_DSPARK_CONTROLLER_H
/* What survives of the V4.1 DSpark host policy after the calibrated-window
 * controller was deleted: the greedy acceptance rule, which is
 * exactness and not policy, and the reasoning-span tracker the admission
 * controller and the batch boundary both read.  Admission itself now lives in
 * `ds4_ds41_dspark_adaptive.h`, ported from GLM's DFlash2 windowed controller;
 * there is no calibration file, no position window and no Wilson bin. */
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    bool thinking;
    char tag[9];
    unsigned tag_len;
} ds41_ctl_state;

/* Shared by generation and host boundary tests. Only target policy argmax
 * chooses acceptance; nothing else may change this rule. */
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
        if (s->tag_len >= 8 && !memcmp(s->tag+s->tag_len-8, "</think>", 8)) s->thinking = false;
    }
}
#endif
