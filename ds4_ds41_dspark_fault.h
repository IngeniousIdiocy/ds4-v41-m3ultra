#ifndef DS4_DS41_DSPARK_FAULT_H
#define DS4_DS41_DSPARK_FAULT_H
/* V4.1 DSpark fault latch: GLM-5.3 DFlash2's `ds4_dflash_fault.h` ported
 * unchanged in shape and meaning.  Source of truth is that header on this
 * project's GLM-5.3-Flash branch.
 *
 * The latch belongs to the allocated session, not to its drafter state, its
 * conditioning generation or its request ledger.  A drafter that failed once
 * must not be silently retried by the next request on the same session, and
 * only destroying the session clears the latch.
 *
 * `unsafe` is the distinction that decides whether a fault is recoverable.  A
 * failure that is drained -- the target's KV, position and checkpoint left
 * exactly as the attempt found them -- can be answered by finishing the request
 * serially, and every later request on that session skips the drafter.  A
 * failure that could not be drained leaves the session's conditioning in an
 * unknown state, and `ds41_fault_skip()` refuses it rather than pretending a
 * serial continuation is equivalent. */
#include <stdbool.h>
#include <stdint.h>

typedef struct {
    bool disabled, unsafe, request_done;
    uint64_t requests, attempts, failures, skips;
} ds41_dspark_fault;

static inline void ds41_fault_begin_request(ds41_dspark_fault *f) {
    f->requests++;
    f->request_done = false;
}

static inline bool ds41_fault_attempt(ds41_dspark_fault *f) {
    if (f->disabled) return false;
    f->attempts++;
    return true;
}

static inline void ds41_fault_latch(ds41_dspark_fault *f, bool drained) {
    f->disabled = true;
    f->unsafe = !drained;
    f->failures++;
}

/* True when the drafter must be skipped and the step taken serially. */
static inline bool ds41_fault_skip(ds41_dspark_fault *f) {
    f->skips++;
    return f->disabled && !f->unsafe;
}

/* One process-scoped hook occurrence, consumed only after a real drafter stage
 * has run, so the injected exit is the same exit a genuine drafter failure
 * takes rather than a shortcut around the work. */
static inline bool ds41_fault_inject_once(bool selected, bool *used) {
    if (!selected || *used) return false;
    *used = true;
    return true;
}

#endif
