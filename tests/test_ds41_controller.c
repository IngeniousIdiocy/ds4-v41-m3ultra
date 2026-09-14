/* Host policy tests; no model, GPU or timing measurement.  What remains here
 * after the calibrated-window controller was deleted is the greedy
 * acceptance rule -- which is exactness, not policy -- and the reasoning-span
 * tracker.  Admission is tested in tests/test_ds41_dspark_adaptive.c. */
#include "../ds4_dspark_controller.h"
#include <assert.h>
#include <stdio.h>

static int select_row(void *ctx, unsigned r) { return ((int *)ctx)[r]; }
static bool stop_boundary(void *ctx, int token) { return token == *(int *)ctx; }

int main(void) {
    int policy[6] = {10,11,12,13,14,15}, next = -1;
    int32_t draft[5] = {10,11,12,13,14};
    for (unsigned r = 2; r <= 6; r++) for (unsigned cap = 1; cap <= r; cap++) {
        assert(ds41_ctl_accept(r,cap,-1,false,select_row,policy,draft,NULL,NULL,&next) == cap);
        assert(next == policy[cap-1]);
        for (unsigned stop = 0; stop < cap; stop++) {
            assert(ds41_ctl_accept(r,cap,policy[stop],false,select_row,policy,draft,NULL,NULL,&next) == stop+1);
            assert(next == policy[stop]);
        }
        assert(ds41_ctl_accept(r,cap,-1,true,select_row,policy,draft,NULL,NULL,&next) == 1);
    }
    int boundary = 12;
    assert(ds41_ctl_accept(6,6,-1,false,select_row,policy,draft,stop_boundary,&boundary,&next) == 3);
    draft[1] = 99;
    assert(ds41_ctl_accept(6,6,-1,false,select_row,policy,draft,NULL,NULL,&next) == 2 && next == 11);

    /* The reasoning tracker, including delimiters split across tokens. */
    ds41_ctl_state s = {0};
    ds41_ctl_text(&s,"<thi",4); assert(!s.thinking);
    ds41_ctl_text(&s,"nk>",3);  assert(s.thinking);
    ds41_ctl_text(&s,"</th",4); assert(s.thinking);
    ds41_ctl_text(&s,"ink>",4); assert(!s.thinking);
    /* A tag longer than the ring still matches on its final bytes. */
    ds41_ctl_text(&s,"xxxxxxxxxxxx<think>",19); assert(s.thinking);

    puts("acceptance rule and reasoning tracker: PASS");
    return 0;
}
