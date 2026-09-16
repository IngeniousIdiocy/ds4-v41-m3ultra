/* Public release defaults must reproduce the explicitly enabled UAT profile.
 * Rejected experimental APIs must not silently reappear. No model is loaded. */
#include "../ds4.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
    const int expected = argc > 1 ? atoi(argv[1]) : 1;
    const char *names[] = {"mtp_down_wide", "mtp_gu_sort_simd", "mtp_swiglu_round",
        "mtp_q8_round", "mtp_f32_rows", "mtp_pointwise_batch"};
    for (unsigned i = 0; i < sizeof(names)/sizeof(names[0]); i++) {
        int got = -1;
        if (!ds41_levers_get(names[i], &got) || got != expected) {
            fprintf(stderr, "%s: expected %d, got %d\n", names[i], expected, got);
            return 1;
        }
        if (!ds41_levers_set(names[i], 0) || !ds41_levers_get(names[i], &got) || got != 0 ||
            !ds41_levers_set(names[i], 1) || !ds41_levers_get(names[i], &got) || got != 1) return 2;
    }
    int got;
    if (ds41_levers_get("mtp_f16_compact", &got) || ds41_levers_set("mtp_f16_compact", 1) ||
        ds41_levers_get("collapse_split", &got) || ds41_levers_set("collapse_split", 1) ||
        ds41_levers_set("hc_stream_layout", 1) || ds41_levers_set("hc_stream_layout", 3)) return 3;
    puts("PASS release defaults, rollback switches, and rejected-path removal");
    return 0;
}
