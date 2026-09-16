/* Production Q4 geometry, same scalar oracle as test_metal_moe_prefill.
 * Single device only. Exercise 1..8 row controls and the reordered six-row path. */
#include "../ds4.h"
#include "../ds4_gpu.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
enum { SELECTED=6 };
typedef struct {uint16_t d,dmin;uint8_t scales[12],qs[128];} q4_block;
typedef struct {uint8_t e,qs[16];} mxfp4_block;
static uint32_t rng=1;
static uint32_t random_u32(void){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
static int check_static_batch(bool q4) {
    const uint32_t D = q4 ? 5120 : 4096, H = q4 ? 2304 : 2048;
    const uint32_t N = q4 ? 8 : 6, E = q4 ? 384 : 256;
    const uint32_t type = q4 ? 12 : 39;
    const uint64_t row = q4 ? D / 256 * sizeof(q4_block) : D / 32 * sizeof(mxfp4_block);
    const uint64_t down_row = q4 ? H / 256 * sizeof(q4_block) : H / 32 * sizeof(mxfp4_block);
    const uint64_t expert = H * row, tensor = E * expert, bytes = 3 * tensor;
    void *model = NULL;
    if (posix_memalign(&model, getpagesize(), bytes)) return 0;
    memset(model, 0, bytes);
    const int32_t mx_active[] = {0, 1, 63, 125, 126, 127, 128, 129, 130, 192, 254, 255};
    const int32_t q4_active[] = {0, 1, 63, 127, 190, 191, 192, 193, 255, 381, 382, 383};
    const int32_t *active = q4 ? q4_active : mx_active;
    for (int w = 0; w < 3; w++) {
        for (unsigned e = 0; e < 12; e++) {
            void *data = (char *)model + w * tensor + active[e] * expert;
            if (q4) {
                q4_block *b = data;
                for (uint64_t j = 0; j < expert / sizeof(*b); j++) {
                    b[j].d = b[j].dmin = 0x1800;
                    for (unsigned k = 0; k < sizeof(b[j].scales); k++) b[j].scales[k] = random_u32();
                    for (unsigned k = 0; k < sizeof(b[j].qs); k++) b[j].qs[k] = random_u32();
                }
            } else {
                mxfp4_block *b = data;
                for (uint64_t j = 0; j < expert / sizeof(*b); j++) {
                    b[j].e = 118 + random_u32() % 5;
                    for (int k = 0; k < 16; k++) b[j].qs[k] = random_u32();
                }
            }
        }
    }
    float *x = malloc(N * D * sizeof(float));
    float *reference = malloc(N * D * sizeof(float));
    float *actual = malloc(N * D * sizeof(float));
    int32_t ids[N * SELECTED];
    float weights[N * SELECTED];
    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc(N * D * sizeof(float));
    ds4_gpu_tensor *it = ds4_gpu_tensor_alloc(sizeof(ids));
    ds4_gpu_tensor *wt = ds4_gpu_tensor_alloc(sizeof(weights));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(N * SELECTED * H * sizeof(float));
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(N * SELECTED * H * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(N * SELECTED * H * sizeof(float));
    ds4_gpu_tensor *down = ds4_gpu_tensor_alloc(N * SELECTED * D * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(N * D * sizeof(float));
    int ok = x && reference && actual && xt && it && wt && gate && up && mid && down && out;
    if (!ok) goto done;
    for (uint32_t i = 0; i < N * D; i++) x[i] = ((int)(random_u32() % 101) - 50) / 256.0f;
    for (uint32_t r = 0; r < N; r++) for (int s = 0; s < SELECTED; s++) {
        ids[r * SELECTED + s] = active[(r * 3 + s) % 12];
        weights[r * SELECTED + s] = q4 ? (s + 1) / 21.0f : 1.0f / SELECTED;
    }
    ok = ds4_gpu_set_model_map(model, bytes) &&
         ds4_gpu_tensor_write(xt, 0, x, N * D * sizeof(float)) &&
         ds4_gpu_tensor_write(it, 0, ids, sizeof(ids)) &&
         ds4_gpu_tensor_write(wt, 0, weights, sizeof(weights));
    setenv("DS4_TP_NO_KEEPALIVE", "1", 1);
    for (int rank = -1; rank < 0 && ok; rank++) {
        if (rank >= 0) ok = ds4_gpu_tp_init((uint32_t)rank, NULL, 0, 0, 0, NULL, NULL);
        for (uint32_t r = 0; r < N && ok; r++) {
            ds4_gpu_tensor *xr = ds4_gpu_tensor_view(xt, r * D * sizeof(float), D * sizeof(float));
            ds4_gpu_tensor *ir = ds4_gpu_tensor_view(it, r * SELECTED * sizeof(int32_t), SELECTED * sizeof(int32_t));
            ds4_gpu_tensor *wr = ds4_gpu_tensor_view(wt, r * SELECTED * sizeof(float), SELECTED * sizeof(float));
            ok = xr && ir && wr && ds4_gpu_routed_moe_one_tensor(
                out, gate, up, mid, down, model, bytes, 0, tensor, 2 * tensor,
                type, type, expert, row, expert, down_row, D, H, D,
                ir, wr, E, SELECTED, 7.0f, xr, NULL, 0, true) &&
                ds4_gpu_tensor_read(out, 0, reference + r * D, D * sizeof(float));
            ds4_gpu_tensor_free(xr);
            ds4_gpu_tensor_free(ir);
            ds4_gpu_tensor_free(wr);
        }
        const uint32_t sizes[] = {6, 2, 5, 3, 4, 6};
        const uint32_t q4_sizes[] = {8, 1, 2, 3, 4, 5, 6, 7, 8};
        const unsigned n_sizes = q4 ? sizeof(q4_sizes) / sizeof(*q4_sizes) : sizeof(sizes) / sizeof(*sizes);
        for (unsigned i = 0; i < n_sizes && ok; i++) {
            bool half_mid = true;
            const uint32_t n = q4 ? q4_sizes[i] : sizes[i];
            ok = ds4_gpu_tensor_fill_f32(mid, NAN, N * SELECTED * H) &&
                 ds4_gpu_tensor_fill_f32(out, NAN, N * D) &&
                 ds4_gpu_routed_moe_batch_tensor(
                    out, gate, up, mid, down, model, bytes, 0, tensor, 2 * tensor,
                    type, type, expert, row, expert, down_row, D, H, D,
                    it, wt, E, SELECTED, 7.0f, xt, 0, n, &half_mid, true) &&
                 !half_mid && ds4_gpu_tensor_read(out, 0, actual, N * D * sizeof(float));
            for (uint32_t j = 0; j < N * D && ok; j++) {
                ok = j >= n * D ? isnan(actual[j]) :
                    isfinite(actual[j]) && isfinite(reference[j]) && memcmp(actual+j,reference+j,sizeof(float)) == 0;
                if (!ok) fprintf(stderr, "decode mismatch index=%u expected=%.9g actual=%.9g\n",
                    j, j < n * D ? reference[j] : NAN, actual[j]);
            }
            fprintf(stderr, "%s static batch rank=%d rows=%u exact: %s\n",
                q4 ? "V4.1 Q4" : "MXFP4", rank, n, ok ? "PASS" : "FAIL");
        }
        if (rank >= 0) ds4_gpu_tp_shutdown();
    }
    unsetenv("DS4_TP_NO_KEEPALIVE");
done:
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(it); ds4_gpu_tensor_free(wt);
    ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(up); ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(down); ds4_gpu_tensor_free(out);
    ds4_gpu_cleanup();
    free(x); free(reference); free(actual); free(model);
    return ok;
}

int main(void) {
 if (!ds4_gpu_init()) return 1;
 ds41_levers_init_from_env();
 g_ds41_levers.mtp_gu_swizzle=1;
 return check_static_batch(true) ? 0 : 1;
}
