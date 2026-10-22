/* DeepSeek V4.1's drafter window in the session payload.
 *
 * Run with:
 *   DS4_TEST_MODEL=/path/to/V4.1.gguf DS4_TEST_DSPARK=/path/to/V4.1-DSpark.gguf \
 *       make test-deepseek41-dspark-payload
 *
 * With the drafter loaded, a saved payload carries the drafter's window. The test saves one after a prompt and a few
 * steps, runs eight DSpark cycles, restores the payload and runs them again: the cycles must draft and accept the same
 * tokens with the same logits. Then the same payload without its drafter part (the format of payloads saved without a
 * drafter) must still load, and the session must decode its next token serially while the window warms up. The public
 * controller is off, so the cycles draft as main's scheduler decides. */
#include "ds4.h"
#include "ds4_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_CTX 4096
#define CYCLES 8

static const char *prompt_text =
    "The history of the printing press begins in the fifteenth century, when";

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

/* CYCLES DSpark cycles; returns how many accepted more than their first token */
static int cycles(ds4_session *s, int V, int *n, int (*tokens)[17], float *logits) {
    char err[256] = {0};
    int drafted = 0;
    for (int c = 0; c < CYCLES; c++) {
        n[c] = ds4_session_eval_speculative_argmax(s, ds4_session_argmax(s), 16, -1, tokens[c], 17, err, sizeof(err));
        if (n[c] <= 0) fail(err[0] ? err : "speculative cycle");
        if (ds4_session_copy_logits(s, logits + (size_t)c * V, V) != V) fail("copy logits");
        drafted += n[c] > 1;
    }
    return drafted;
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
    ds4_gpu_test_set_v41_off(DS4_GPU_V41_PUBLIC_CONTROLLER);
    const int V = ds4_engine_vocab_size(e);
    /* the drafter part: three rings of 128 positions and 128 rows of features */
    const size_t draft_bytes = 3u * (128u * 4u + 128u * (size_t)ds4_engine_embd_dim(e) * 4u);
    ds4_tokens prompt = {0};
    ds4_tokenize_text(e, prompt_text, &prompt);
    ds4_session *s = NULL;
    if (ds4_session_create(&s, e, TEST_CTX) != 0 || ds4_session_sync(s, &prompt, err, sizeof(err)) != 0)
        fail(err[0] ? err : "session");
    for (int i = 0; i < 16; i++)
        if (ds4_session_eval(s, ds4_session_argmax(s), err, sizeof(err)) != 0) fail(err);

    char *buf = NULL;
    size_t len = 0;
    FILE *fp = open_memstream(&buf, &len);
    if (!fp || ds4_session_save_payload(s, fp, err, sizeof(err)) != 0) fail(err[0] ? err : "save");
    fclose(fp);
    if (len != ds4_session_payload_bytes(s) || len < 64 + draft_bytes) fail("payload size");
    uint32_t field3;
    memcpy(&field3, buf + 12, sizeof(field3));
    if (field3 != 2) fail("the payload does not carry the drafter window");

    int n[2][CYCLES], tokens[2][CYCLES][17];
    float *logits = malloc((size_t)2 * CYCLES * V * sizeof(float));
    if (!logits) fail("alloc");
    const int drafted = cycles(s, V, n[0], tokens[0], logits);
    load(s, buf, len);
    cycles(s, V, n[1], tokens[1], logits + (size_t)CYCLES * V);
    if (!drafted) fail("no cycle drafted");
    for (int c = 0; c < CYCLES; c++)
        if (n[0][c] != n[1][c] || memcmp(tokens[0][c], tokens[1][c], (size_t)n[0][c] * sizeof(int)) ||
            memcmp(logits + (size_t)c * V, logits + (size_t)(CYCLES + c) * V, (size_t)V * sizeof(float)))
            fail("the restored session drafts differently");

    const uint32_t old_format = 1;   /* the same payload without its drafter part */
    memcpy(buf + 12, &old_format, sizeof(old_format));
    load(s, buf, len - draft_bytes);
    int first[17];
    if (ds4_session_eval_speculative_argmax(s, ds4_session_argmax(s), 16, -1, first, 17, err, sizeof(err)) != 1)
        fail(err[0] ? err : "a payload without the drafter window drafted before its window warmed up");

    printf("drafter window in the payload: %zu bytes, %d of %d cycles drafted, restored cycles identical, "
           "old format loads and warms up\n", len, drafted, CYCLES);
    ds4_gpu_test_set_v41_off(0);
    free(logits);
    free(buf);
    ds4_session_free(s);
    ds4_tokens_free(&prompt);
    ds4_engine_close(e);
    printf("OK\n");
    return 0;
}
