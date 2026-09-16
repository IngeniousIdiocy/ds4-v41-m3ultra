/* Exact routed-producer expansion against the standalone production chain.
 * The old implicitly contracted epilogue fails this test at a BF16 boundary. */
#include "../ds4.h"
#include "../ds4_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { D = 5120, H = 2304, E = 384, SELECTED = 6 };
typedef struct { uint16_t d, dmin; uint8_t scales[12], qs[128]; } q4_block;
static uint32_t rng = 1;
static uint32_t random_u32(void) {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return rng;
}

static int check_producer(void) {
    const uint64_t row = D / 256 * sizeof(q4_block);
    const uint64_t down_row = H / 256 * sizeof(q4_block);
    const uint64_t expert = H * row, tensor = E * expert, bytes = 3 * tensor;
    void *model = NULL;
    if (posix_memalign(&model, getpagesize(), bytes)) return 0;
    memset(model, 0, bytes);
    const int32_t active[] = {0, 1, 63, 127, 190, 191, 192, 193, 255, 381, 382, 383};
    for (int w = 0; w < 3; w++) {
        for (unsigned e = 0; e < 12; e++) {
            q4_block *b = (void *)((char *)model + w * tensor + active[e] * expert);
            for (uint64_t j = 0; j < expert / sizeof(*b); j++) {
                b[j].d = b[j].dmin = 0x1800;
                for (unsigned k = 0; k < sizeof(b[j].scales); k++) b[j].scales[k] = random_u32();
                for (unsigned k = 0; k < sizeof(b[j].qs); k++) b[j].qs[k] = random_u32();
            }
        }
    }
    float x[D], reference[D], actual[D], weights[SELECTED];
    int32_t ids[SELECTED];
    float shared[D], residual[4 * D], hc_ref[4 * D], hc_out[4 * D];
    float block_ref[D], block_out[D], split[24], pre[4];
    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc(sizeof(x));
    ds4_gpu_tensor *it = ds4_gpu_tensor_alloc(sizeof(ids));
    ds4_gpu_tensor *wt = ds4_gpu_tensor_alloc(sizeof(weights));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(SELECTED * H * sizeof(float));
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(SELECTED * H * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(SELECTED * H * sizeof(float));
    ds4_gpu_tensor *down = ds4_gpu_tensor_alloc(SELECTED * D * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(actual));
    ds4_gpu_tensor *shared_t = ds4_gpu_tensor_alloc(sizeof(shared));
    ds4_gpu_tensor *res_t = ds4_gpu_tensor_alloc(sizeof(residual));
    ds4_gpu_tensor *split_t = ds4_gpu_tensor_alloc(sizeof(split));
    ds4_gpu_tensor *block_t = ds4_gpu_tensor_alloc(sizeof(block_out));
    ds4_gpu_tensor *hc_t = ds4_gpu_tensor_alloc(sizeof(hc_out));
    ds4_gpu_tensor *pre_t = ds4_gpu_tensor_alloc(sizeof(pre));
    int ok = xt && it && wt && gate && up && mid && down && out &&
             shared_t && res_t && split_t && block_t && hc_t && pre_t;
    unsigned long checked = 0;
    if (!ok) goto done;
    for (unsigned i = 0; i < D; i++) x[i] = ((int)(random_u32() % 101) - 50) / 256.0f;
    for (int s = 0; s < SELECTED; s++) {
        ids[s] = active[s];
        weights[s] = (s + 1) / 21.0f;
    }
    ok = ds4_gpu_set_model_map(model, bytes) &&
         ds4_gpu_tensor_write(xt, 0, x, sizeof(x)) &&
         ds4_gpu_tensor_write(it, 0, ids, sizeof(ids)) &&
         ds4_gpu_tensor_write(wt, 0, weights, sizeof(weights));
    for (unsigned rep = 0; rep < 64 && ok; rep++) {
        for (unsigned j = 0; j < D; j++) shared[j] = ((int)(random_u32() % 20001) - 10000) / 128.0f;
        for (unsigned j = 0; j < 4 * D; j++) residual[j] = ((int)(random_u32() % 20001) - 10000) / 128.0f;
        for (unsigned j = 0; j < 24; j++) split[j] = (random_u32() & 0xffffff) / 16777216.0f;
        ok = ds4_gpu_tensor_write(shared_t, 0, shared, sizeof(shared)) &&
             ds4_gpu_tensor_write(res_t, 0, residual, sizeof(residual)) &&
             ds4_gpu_tensor_write(split_t, 0, split, sizeof(split));
        if (!ok) break;
        /* Reference: routed sum followed by the standalone rounded expansion. */
        ok = ds4_gpu_routed_moe_one_tensor(out, gate, up, mid, down, model, bytes,
                 0, tensor, 2 * tensor, 12, 12, expert, row, expert, down_row,
                 D, H, D, it, wt, E, SELECTED, 7.0f, xt, NULL, 0, true) &&
             ds4_gpu_tensor_read(out, 0, reference, sizeof(reference)) &&
             ds4_gpu_dsv41_hc_round_expand4_tensor(hc_t, block_t, out, shared_t,
                 res_t, split_t, D, 4) &&
             ds4_gpu_tensor_read(hc_t, 0, hc_ref, sizeof(hc_ref)) &&
             ds4_gpu_tensor_read(block_t, 0, block_ref, sizeof(block_ref));
        if (!ok) break;
        ok = ds4_gpu_dsv41_arch_ffn_begin(out, shared_t, res_t, split_t, block_t, hc_t, pre_t) &&
             ds4_gpu_routed_moe_one_tensor(out, gate, up, mid, down, model, bytes,
                 0, tensor, 2 * tensor, 12, 12, expert, row, expert, down_row,
                 D, H, D, it, wt, E, SELECTED, 7.0f, xt, NULL, 0, true);
        int folded = ds4_gpu_dsv41_arch_ffn_end();
        ok = ok && folded &&
             ds4_gpu_tensor_read(out, 0, actual, sizeof(actual)) &&
             ds4_gpu_tensor_read(hc_t, 0, hc_out, sizeof(hc_out)) &&
             ds4_gpu_tensor_read(block_t, 0, block_out, sizeof(block_out)) &&
             ds4_gpu_tensor_read(pre_t, 0, pre, sizeof(pre));
        if (!ok) { fprintf(stderr, "call failed folded=%d\n", folded); break; }
        if (memcmp(actual, reference, sizeof(actual)) ||
            memcmp(block_out, block_ref, sizeof(block_out)) || memcmp(pre, split, sizeof(pre))) {
            fprintf(stderr, "producer operands mismatch rep=%u\n", rep);
            ok = 0;
            break;
        }
        for (unsigned j = 0; j < 4 * D; j++) {
            if (memcmp(hc_ref + j, hc_out + j, sizeof(float))) {
                fprintf(stderr, "HC mismatch rep=%u word=%u expected=%.9g actual=%.9g\n",
                        rep, j, hc_ref[j], hc_out[j]);
                ok = 0;
                break;
            }
        }
        if (ok) checked += D * 6 + 4;
    }
done:
    fprintf(stderr, "%s %lu exact producer-chain words\n", ok ? "PASS" : "FAIL", checked);
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(it); ds4_gpu_tensor_free(wt);
    ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(up); ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(down); ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(shared_t); ds4_gpu_tensor_free(res_t); ds4_gpu_tensor_free(split_t);
    ds4_gpu_tensor_free(block_t); ds4_gpu_tensor_free(hc_t); ds4_gpu_tensor_free(pre_t);
    ds4_gpu_cleanup();
    free(model);
    return ok;
}

int main(void) {
    ds41_levers_init_from_env();
    for (int wide = 0; wide <= 1; wide++) {
        if (!ds4_gpu_init()) return 1;
        ds41_levers_set("q4_wide", wide);
        rng = 1;
        fprintf(stderr, "q4_wide=%d\n", wide);
        if (!check_producer()) return 1;
    }
    return 0;
}
