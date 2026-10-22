/* DeepSeek V4.1's DSpark verify blocks through the batched attention (VERIFY_INDEXED).
 *
 * Run with:
 *   DS4_TEST_MODEL=/path/to/V4.1.gguf DS4_TEST_DSPARK=/path/to/V4.1-DSpark.gguf \
 *       make test-deepseek41-verify-indexed
 *
 * A verify block attends like a short prefill, and each row keeps the pooling carry its own decode step would leave,
 * so a commit can rewind to any row. From the same saved session past 2,048 tokens (the indexed attention and the
 * batched top-k), at an even and an odd start, a block of at most 4 rows and a block of at most 5 rows are rewound
 * to their first, second and third rows; the next four decode steps must give the same logits, bit for bit, whatever
 * the block held after the rewound row. The row-count-dependent matrix kernels (ROWS_MMA) are off and the public
 * controller is off, so the cycles draft as main's scheduler decides. */
#include "ds4.h"
#include "ds4_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_CTX 4096
#define STEPS 4

static void fail(const char *what) {
    fprintf(stderr, "FAIL: %s\n", what);
    exit(1);
}

static void load(ds4_session *s, char *buf, size_t len) {
    char err[256] = {0};
    FILE *fp = fmemopen(buf, len, "rb");
    if (!fp || ds4_session_load_payload(s, fp, len, err, sizeof(err)) != 0) fail(err[0] ? err : "load");
    fclose(fp);
}

/* restore, one cycle of at most cap rows, rewind to row keep, STEPS decode steps; false: the cycle accepted too few */
static bool run(ds4_session *s, char *buf, size_t len, int cap, int keep, int V, float *logits) {
    char err[256] = {0};
    int tokens[17];
    load(s, buf, len);
    const int start = ds4_session_pos(s);
    const int n = ds4_session_eval_speculative_argmax(s, ds4_session_argmax(s), cap, -1, tokens, 17, err, sizeof(err));
    if (n <= 0) fail(err[0] ? err : "speculative cycle");
    if (n <= keep) return false;
    if (!ds4_session_rewind_speculative(s, start + keep)) fail("rewind into the verify block");
    for (int i = 0; i < STEPS; i++) {
        if (ds4_session_eval(s, ds4_session_argmax(s), err, sizeof(err)) != 0) fail(err);
        if (ds4_session_copy_logits(s, logits + (size_t)i * V, V) != V) fail("copy logits");
    }
    return true;
}

int main(void) {
    const char *model = getenv("DS4_TEST_MODEL"), *support = getenv("DS4_TEST_DSPARK");
    if (!model || !model[0] || !support || !support[0]) {
        fprintf(stderr, "SKIP: DS4_TEST_MODEL or DS4_TEST_DSPARK is not set\n");
        return 0;
    }
    ds4_engine_options opt = {
        .model_path = model,
        .mtp_path = support,
        .dspark = true,
        .backend = DS4_BACKEND_METAL,
        .n_threads = 1,
        .context_size = TEST_CTX,
    };
    ds4_engine *e = NULL;
    char err[256] = {0};
    if (ds4_engine_open(&e, &opt) != 0) fail("engine open");
    ds4_gpu_test_set_v41_off(DS4_GPU_V41_PUBLIC_CONTROLLER | DS4_GPU_V41_ROWS_MMA);
    const int V = ds4_engine_vocab_size(e);
    char *text = malloc(16 * 1100);
    if (!text) fail("alloc");
    text[0] = 0;
    for (int i = 1; i <= 1100; i++) sprintf(text + strlen(text), "%d, ", i);
    ds4_tokens prompt = {0};
    ds4_tokenize_text(e, text, &prompt);
    ds4_session *s = NULL;
    if (ds4_session_create(&s, e, TEST_CTX) != 0 || ds4_session_sync(s, &prompt, err, sizeof(err)) != 0)
        fail(err[0] ? err : "session");
    if (ds4_session_pos(s) < 2100) fail("the prompt is too short for the indexed attention");
    /* the drafter's window warms with ordinary decoding */
    for (int i = 0; i < 128; i++)
        if (ds4_session_eval(s, ds4_session_argmax(s), err, sizeof(err)) != 0) fail(err);

    float *logits = malloc((size_t)2 * STEPS * V * sizeof(float));
    if (!logits) fail("alloc");
    int checked = 0;
    for (int parity = 0; parity < 2; parity++) {
        if ((ds4_session_pos(s) & 1) != parity &&
            ds4_session_eval(s, ds4_session_argmax(s), err, sizeof(err)) != 0) fail(err);
        char *buf = NULL;
        size_t len = 0;
        FILE *fp = open_memstream(&buf, &len);
        if (!fp || ds4_session_save_payload(s, fp, err, sizeof(err)) != 0) fail(err[0] ? err : "save");
        fclose(fp);
        for (int keep = 1; keep <= 3; keep++) {
            if (!run(s, buf, len, 4, keep, V, logits) ||
                !run(s, buf, len, 5, keep, V, logits + (size_t)STEPS * V)) continue;
            if (memcmp(logits, logits + (size_t)STEPS * V, (size_t)STEPS * V * sizeof(float)))
                fail("a block rewound to a row continues differently from a longer block rewound to it");
            checked++;
        }
        load(s, buf, len);
        free(buf);
    }
    if (checked < 4) fail("too few cycles accepted enough rows");
    printf("verify blocks through the batched attention: %d rewinds to rows 1-3 at even and odd starts continue "
           "identically from 4- and 5-row blocks\n", checked);
    ds4_gpu_test_set_v41_off(0);
    free(logits);
    free(text);
    ds4_session_free(s);
    ds4_tokens_free(&prompt);
    ds4_engine_close(e);
    printf("OK\n");
    return 0;
}
