#include "ds4_gpu.h"
#include "ds4.h"
#include "ds4_deepseek41_admission.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); return 0; \
} } while (0)

static uint32_t seed = 7919;
static float random_value(void) {
    seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
    return ((int)(seed % 65537) - 32768) / 8192.0f;
}

static float bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, 4);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16) & 1u);
    bits &= 0xffff0000u;
    memcpy(&value, &bits, 4);
    return value;
}

static float nearest(float value, int fp4) {
    const float fp4_values[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    float best_value = 0, best_error = INFINITY;
    int best = 0;
    for (int i = 0; i < (fp4 ? 8 : 127); i++) {
        const float v = fp4 ? fp4_values[i] : i < 8 ? ldexpf(i, -9) :
                        ldexpf(1.0f + (i & 7) / 8.0f, (i >> 3) - 7);
        const float error = fabsf(fabsf(value) - v);
        if (error < best_error || (error == best_error && !(i & 1) && (best & 1))) {
            best = i; best_value = v; best_error = error;
        }
    }
    return copysignf(best_value, value);
}

static ds4_gpu_tensor *upload(const void *data, size_t bytes) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc_managed(bytes);
    if (t && data && !ds4_gpu_tensor_write(t, 0, data, bytes)) {
        ds4_gpu_tensor_free(t);
        return NULL;
    }
    return t;
}

static int check_router(void) {
    enum { ROWS = 129, MAX_EXPERTS = 384, USED = 6 };
    const size_t bias_bytes = (size_t)getpagesize();
    float *bias = NULL;
    CHECK(bias_bytes >= MAX_EXPERTS * sizeof(float));
    CHECK(posix_memalign((void **)&bias, bias_bytes, bias_bytes) == 0);
    memset(bias, 0, bias_bytes);
    float *logits = malloc(ROWS * MAX_EXPERTS * sizeof(float));
    int32_t tokens[ROWS] = {0};
    CHECK(bias && logits);
    for (unsigned e = 0; e < MAX_EXPERTS; e++) bias[e] = (e % 7) * 0.125f;
    CHECK(ds4_gpu_set_model_map(bias, bias_bytes));
    ds4_gpu_tensor *x = upload(NULL, ROWS * MAX_EXPERTS * sizeof(float));
    ds4_gpu_tensor *p = upload(NULL, ROWS * MAX_EXPERTS * sizeof(float));
    ds4_gpu_tensor *ids = upload(NULL, ROWS * USED * sizeof(int32_t));
    ds4_gpu_tensor *weights = upload(NULL, ROWS * USED * sizeof(float));
    ds4_gpu_tensor *tok = upload(tokens, sizeof(tokens));
    CHECK(x && p && ids && weights && tok);
    const unsigned counts[] = {1, 3, 17, ROWS};
    for (unsigned n = 128; n <= MAX_EXPERTS; n += 128) {
        const unsigned USED = n == 128 ? 3 : 6;
        for (unsigned mode = 0; mode < 3; mode++) {
            for (unsigned i = 0; i < ROWS * n; i++) logits[i] = mode == 1 ? 0 :
                mode == 2 ? (i % 2 ? -80 : 40) : random_value() * 8;
            CHECK(ds4_gpu_tensor_write(x, 0, logits, ROWS * n * sizeof(float)));
            for (unsigned ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
                const unsigned rows = counts[ci];
                CHECK(ds4_gpu_router_select_batch_tensor(ids, weights, p, bias,
                    bias_bytes, 0, 0, 0, 0, 0, true, false,
                    x, tok, n, USED, 1.5f, rows));
                CHECK(ds4_gpu_synchronize());
                const int32_t *selected = ds4_gpu_tensor_contents(ids);
                const float *actual = ds4_gpu_tensor_contents(weights);
                const float *probs = ds4_gpu_tensor_contents(p);
                CHECK(selected && actual && probs);
                for (unsigned t = 0; t < rows; t++) {
                    double ref[MAX_EXPERTS];
                    int best[USED];
                    for (unsigned k = 0; k < USED; k++) best[k] = -1;
                    for (unsigned e = 0; e < n; e++) {
                        const double v = logits[t * n + e];
                        ref[e] = sqrt(v > 20 ? v : v < -20 ? exp(v) : log1p(exp(v)));
                        if (!(fabs(probs[t * n + e] - ref[e]) <= 3e-6 * (1 + ref[e])))
                            fprintf(stderr, "router n=%u mode=%u rows=%u row=%u expert=%u logit=%.9g actual=%.9g ref=%.9g\n",
                                    n, mode, rows, t, e, v, probs[t * n + e], ref[e]);
                        CHECK(fabs(probs[t * n + e] - ref[e]) <= 3e-6 * (1 + ref[e]));
                        const double score = ref[e] + bias[e];
                        for (unsigned k = 0; k < USED; k++) {
                            if (best[k] < 0 || score > ref[best[k]] + bias[best[k]]) {
                                for (unsigned j = USED - 1; j > k; j--) best[j] = best[j - 1];
                                best[k] = (int)e;
                                break;
                            }
                        }
                    }
                    double sum = 0;
                    for (unsigned k = 0; k < USED; k++) {
                        const int id = selected[t * USED + k];
                        CHECK(id >= 0 && id < (int)n);
                        for (unsigned j = 0; j < k; j++) CHECK(id != selected[t * USED + j]);
#ifdef __APPLE__
                        /* Bitonic selection does not promise CUDA's lower-ID
                         * tie break. Still require the correct top-k scores. */
                        CHECK(fabs((ref[id] + bias[id]) - (ref[best[k]] + bias[best[k]])) < 3e-6);
#else
                        CHECK(id == best[k]);
#endif
                        sum += ref[id];
                    }
                    for (unsigned k = 0; k < USED; k++) {
                        CHECK(fabs(actual[t * USED + k] - 1.5 * ref[selected[t * USED + k]] /
                              fmax(sum, 0x1p-14)) < 3e-6);
                    }
                }
                int32_t first_ids[USED]; float first_weights[USED];
                memcpy(first_ids, selected, sizeof(first_ids));
                memcpy(first_weights, actual, sizeof(first_weights));
                CHECK(ds4_gpu_router_select_tensor(ids, weights, p, bias,
                    bias_bytes, 0, 0, 0, 0, n, USED, 1.5f,
                    0, 0, true, false, x));
                CHECK(ds4_gpu_synchronize());
                const int32_t *scalar_ids = ds4_gpu_tensor_contents(ids);
                const float *scalar_weights = ds4_gpu_tensor_contents(weights);
#ifdef __APPLE__
                for (unsigned k = 0; k < USED; k++) {
                    CHECK(scalar_ids[k] >= 0 && scalar_ids[k] < (int32_t)n);
                    CHECK(probs[first_ids[k]] + bias[first_ids[k]] ==
                          probs[scalar_ids[k]] + bias[scalar_ids[k]]);
                    CHECK(fabsf(first_weights[k] - scalar_weights[k]) < 3e-6f);
                }
#else
                CHECK(!memcmp(first_ids, scalar_ids, sizeof(first_ids)));
                CHECK(!memcmp(first_weights, scalar_weights, sizeof(first_weights)));
#endif
            }
        }
    }
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(p); ds4_gpu_tensor_free(ids);
    ds4_gpu_tensor_free(weights); ds4_gpu_tensor_free(tok);
    ds4_gpu_cleanup();
    free(bias); free(logits);
    CHECK(ds4_gpu_init());
    fprintf(stderr, "256/384-expert routing, ties/extremes and scalar/batch oracle: PASS\n");
    return 1;
}

static int check_quantization(void) {
    enum { WIDTH = 512, ROWS = 33, N = WIDTH * ROWS };
    float *source = malloc(N * sizeof(float)), *actual = malloc(N * sizeof(float));
    CHECK(source && actual);
    ds4_gpu_tensor *t = upload(NULL, N * sizeof(float));
    CHECK(t);
    for (int mode = 0; mode < 4; mode++) {
        const int block = mode == DS4_V41_FP4_E4M3 ? 16 : 32;
        for (int i = 0; i < N; i++) source[i] = random_value() * (1u << ((i / block) % 4));
        for (int i = 0; i < WIDTH; i++) source[i] = copysignf(0.0f, i & 1 ? -1.0f : 1.0f);
        CHECK(ds4_gpu_tensor_write(t, 0, source, N * sizeof(float)));
        CHECK(ds4_gpu_dsv41_quantize(t, WIDTH, ROWS, (ds4_v41_activation_format)mode));
        CHECK(ds4_gpu_tensor_read(t, 0, actual, N * sizeof(float)));
        for (int start = 0; start < N; start += block) {
            float amax = 0, scale = 1;
            for (int i = 0; i < block; i++) amax = fmaxf(amax, fabsf(bf16(source[start + i])));
            if (mode == DS4_V41_FP8_E8M0)
                scale = exp2f(ceilf(log2f(fmaxf(amax, 1.0e-4f) * (1.0f / 448.0f))));
            if (mode == DS4_V41_FP4_E8M0)
                scale = exp2f(ceilf(log2f(fmaxf(amax, 0x1.8p-124f) * (1.0f / 6.0f))));
            if (mode == DS4_V41_FP4_E4M3) scale = nearest(fmaxf(amax, 6.0f / 512.0f) / 6.0f, 0);
            for (int i = 0; i < block; i++) {
                float expected = bf16(source[start + i]);
                if (mode) expected = bf16(nearest(expected / scale, mode != 1) * scale);
                if (memcmp(&expected, actual + start + i, 4)) {
                    fprintf(stderr, "quantization mode=%d index=%d: %.9g != %.9g\n",
                            mode, start + i, actual[start + i], expected);
                    return 0;
                }
            }
        }
    }
    CHECK(ds4_gpu_dsv41_quantize(t, 24, 1, DS4_V41_BF16));
    CHECK(!ds4_gpu_dsv41_quantize(t, 24, 1, DS4_V41_FP8_E8M0));
    CHECK(!ds4_gpu_dsv41_quantize(t, UINT32_MAX, UINT32_MAX, DS4_V41_BF16));
    CHECK(!ds4_gpu_dsv41_quantize(t, 32, 1, (ds4_v41_activation_format)4));
    ds4_gpu_tensor_free(t);
    free(source); free(actual);
    fprintf(stderr, "V4.1 BF16/FP8/FP4 round trips: exact\n");
    return 1;
}

static double monotonic_seconds(void);

static uint32_t bf16_bits(uint32_t bits) {
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16u) & 1u);
    return bits & 0xffff0000u;
}

static uint32_t bf16_input(size_t i) {
    const uint32_t low[] = {0, 0x7fffu, 0x8000u, 0x8001u, 0xffffu};
    if (i < 65536u * 5u) return ((uint32_t)(i / 5u) << 16u) | low[i % 5u];
    return (uint32_t)i * 2654435761u;
}

static int check_bf16_linear(void) {
    const struct { uint32_t width, rows, offset; } shapes[] = {
        {1, 1, 0}, {3, 1, 0}, {5, 1, 0}, {24, 1, 0}, {31, 3, 0},
        {32, 7, 1}, {129, 3, 4}, {1023, 1, 0}, {1024, 1, 0}, {1025, 1, 0},
        {65536u * 5u, 1, 0}, {5120, 1, 0}, {20480, 512, 0},
        {32768, 4096, 0}, {32768, 8192, 0}
    };
    const unsigned modes[] = {0, 1, 1, 0};
    for (size_t s = 0; s < sizeof(shapes) / sizeof(*shapes); s++) {
        const size_t count = (size_t)shapes[s].width * shapes[s].rows;
        const size_t offset = shapes[s].offset;
        ds4_gpu_tensor *storage = upload(NULL, (count + 8) * sizeof(uint32_t));
        CHECK(storage);
        ds4_gpu_tensor *view = ds4_gpu_tensor_view(storage,
            offset * sizeof(uint32_t), count * sizeof(uint32_t));
        CHECK(view);
        uint32_t *bits = ds4_gpu_tensor_contents(storage);
        CHECK(bits);
        for (unsigned pass = 0; pass < 4; pass++) {
            const unsigned mode = modes[pass];
            if (mode) unsetenv("DS4_METAL_DISABLE_V41_LINEAR_BF16");
            else setenv("DS4_METAL_DISABLE_V41_LINEAR_BF16", "1", 1);
            for (size_t i = 0; i < count + 8; i++) bits[i] = 0x12345678;
            for (size_t i = 0; i < count; i++) bits[offset + i] = bf16_input(i);
            double elapsed = 0;
            for (unsigned repeat = 0; repeat < 5; repeat++) {
                const double begin = monotonic_seconds();
                CHECK(ds4_gpu_dsv41_quantize(view, shapes[s].width, shapes[s].rows, DS4_V41_BF16));
                CHECK(ds4_gpu_synchronize());
                if (repeat) elapsed += (monotonic_seconds() - begin) * 250;
            }
            for (size_t i = 0; i < count; i++) CHECK(bits[offset + i] == bf16_bits(bf16_input(i)));
            for (size_t i = 0; i < offset; i++) CHECK(bits[i] == 0x12345678);
            for (size_t i = offset + count; i < count + 8; i++) CHECK(bits[i] == 0x12345678);
            fprintf(stderr, "BF16 width=%u rows=%u offset=%zu mode=%u pass=%u %.3f ms: exact\n",
                shapes[s].width, shapes[s].rows, offset, mode, pass, elapsed);
        }
        ds4_gpu_tensor_free(view);
        ds4_gpu_tensor_free(storage);
    }
    unsetenv("DS4_METAL_DISABLE_V41_LINEAR_BF16");
    return 1;
}

#ifdef __APPLE__
static int check_hc_scaled(void) {
    enum { WIDTH = 20480, OUT = 24, ROWS = 8192 };
    const size_t weight_bytes = WIDTH * OUT * sizeof(_Float16);
    const size_t page = (size_t)getpagesize();
    const size_t mapped_bytes = (weight_bytes + page - 1) / page * page;
    void *model = NULL;
    CHECK(!posix_memalign(&model, page, mapped_bytes));
    _Float16 *weights = model;
    for (size_t i = 0; i < WIDTH * OUT; i++) weights[i] = (_Float16)(random_value() / 64);
    CHECK(ds4_gpu_set_model_map(model, mapped_bytes));
    ds4_gpu_tensor *x = upload(NULL, (size_t)WIDTH * ROWS * 4);
    ds4_gpu_tensor *norm = upload(NULL, (size_t)WIDTH * ROWS * 4);
    ds4_gpu_tensor *scales = upload(NULL, (ROWS + 1) * 4);
    ds4_gpu_tensor *out = upload(NULL, (OUT * ROWS + 1) * 4);
    ds4_gpu_tensor *ref = upload(NULL, (OUT * ROWS + 1) * 4);
    CHECK(x && norm && scales && out && ref);
    float *input = ds4_gpu_tensor_contents(x), *y = ds4_gpu_tensor_contents(out);
    float *expected = ds4_gpu_tensor_contents(ref), *s = ds4_gpu_tensor_contents(scales);
    for (size_t i = 0; i < (size_t)WIDTH * ROWS; i++)
        input[i] = bf16(random_value() * ((i / WIDTH) % 7 ? 1 : 0x1p-16f));
    const unsigned rows[] = {9, 31, 32, 33, 127, 512, 2048, 4096, 8192};
    for (size_t shape = 0; shape < sizeof(rows) / sizeof(*rows); shape++) {
        const uint32_t n = rows[shape];
        double elapsed[2] = {0};
        for (unsigned repeat = 0; repeat < 7; repeat++) {
            y[n * OUT] = expected[n * OUT] = s[n] = 12345;
            for (unsigned j = 0; j < 2; j++) {
                const unsigned mode = j ^ (repeat & 1);
                const double begin = monotonic_seconds();
                CHECK(ds4_gpu_begin_commands());
                if (mode) {
                    CHECK(ds4_gpu_hc_rms_scale_project_f16_tensor(out, n >= 512u ? scales : norm,
                        model, mapped_bytes, 0, WIDTH, OUT, x, n, 1e-6f));
                } else {
                    CHECK(ds4_gpu_rms_norm_plain_rows_tensor(norm, x, WIDTH, n, 1e-6f));
                    CHECK(ds4_gpu_matmul_f16_tensor(ref, model, mapped_bytes, 0,
                        WIDTH, OUT, norm, n));
                }
                CHECK(ds4_gpu_end_commands());
                if (repeat) elapsed[mode] += (monotonic_seconds() - begin) * (1000.0 / 6);
            }
            CHECK(!memcmp(y, expected, n * OUT * 4));
            CHECK(y[n * OUT] == 12345 && expected[n * OUT] == 12345 && s[n] == 12345);
        }
        fprintf(stderr, "HC scaled rows=%u: exact %.3f -> %.3f ms\n", n, elapsed[0], elapsed[1]);
    }
    ds4_gpu_tensor_free(ref); ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(scales);
    ds4_gpu_tensor_free(norm); ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup(); free(model);
    return 1;
}

#endif

static int check_engram(void) {
    enum { D = 5120, ROWS = 5, N = ROWS * 4 * D };
    float *x = malloc(N * sizeof(float)), *actual = malloc(N * sizeof(float));
    float *kv = malloc(ROWS * 5 * D * sizeof(float));
    float *qw = malloc(4 * D * sizeof(float)), *kw = malloc(4 * D * sizeof(float));
    uint8_t mask[] = {1, 1, 0, 1, 0};
    CHECK(x && actual && kv && qw && kw);
    for (int i = 0; i < N; i++) x[i] = bf16(random_value());
    for (int i = 0; i < ROWS * 5 * D; i++) kv[i] = bf16(random_value());
    for (int i = 0; i < 4 * D; i++) { qw[i] = random_value(); kw[i] = random_value(); }
    memset(x, 0, D * sizeof(float));
    memset(kv + D, 0, D * sizeof(float));
    ds4_gpu_tensor *xt = upload(x, N * sizeof(float));
    ds4_gpu_tensor *kt = upload(kv, ROWS * 5 * D * sizeof(float));
    ds4_gpu_tensor *qwt = upload(qw, 4 * D * sizeof(float));
    ds4_gpu_tensor *kwt = upload(kw, 4 * D * sizeof(float));
    ds4_gpu_tensor *mt = upload(mask, sizeof(mask));
    CHECK(xt && kt && qwt && kwt && mt);
    size_t rounded_differently = 0;
    double error2 = 0, norm2 = 0;
    for (int masked = 0; masked < 2; masked++) {
        CHECK(ds4_gpu_tensor_write(xt, 0, x, N * sizeof(float)));
        CHECK(ds4_gpu_dsv41_engram_add(xt, kt, qwt, kwt, masked ? mt : NULL, D, ROWS, 1e-20f));
        CHECK(ds4_gpu_tensor_read(xt, 0, actual, N * sizeof(float)));
        for (int row = 0; row < ROWS; row++) for (int h = 0; h < 4; h++) {
            double dot = 0, h2 = 0, k2 = 0;
            for (int i = 0; i < D; i++) {
                const double a = x[(row * 4 + h) * D + i], b = kv[(row * 5 + h) * D + i];
                h2 += a * a; k2 += b * b;
                dot += a * (float)(qw[h * D + i] * kw[h * D + i]) * b;
            }
            dot /= sqrt(h2 / D + 1e-20) * sqrt(k2 / D + 1e-20) * sqrt(D);
            double gate = 1 / (1 + exp(-copysign(sqrt(fmax(fabs(dot), 1e-6)), dot)));
            if (masked && !mask[row]) gate = 0;
            for (int i = 0; i < D; i++) {
                const int off = (row * 4 + h) * D + i;
                const float expected = bf16(x[off] + (float)gate * kv[(row * 5 + 4) * D + i]);
                const double error = actual[off] - expected;
                CHECK(isfinite(actual[off]));
                CHECK(fabs(error) <= fmax(1e-6, fabs(expected) / 128));
                if (masked && !mask[row]) CHECK(actual[off] == x[off]);
                rounded_differently += actual[off] != expected;
                error2 += error * error; norm2 += (double)expected * expected;
            }
        }
    }
    CHECK(rounded_differently < N / 1000);
    CHECK(sqrt(error2 / norm2) < 1e-4);
    fprintf(stderr, "V4.1 Engram gate: %zu BF16 boundary differences, relative RMS %.8g\n",
            rounded_differently, sqrt(error2 / norm2));
    CHECK(!ds4_gpu_dsv41_engram_add(xt, kt, qwt, kwt, mt, D, ROWS + 1, 1e-20f));
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(kt); ds4_gpu_tensor_free(qwt);
    ds4_gpu_tensor_free(kwt); ds4_gpu_tensor_free(mt);
    free(x); free(actual); free(kv); free(qw); free(kw);
    return 1;
}

static int check_rope_stride(void) {
    enum { WIDTH = 512, HEADS = 2, ROWS = 129, N = WIDTH * HEADS * ROWS };
    ds4_gpu_tensor *a = upload(NULL, (N + 1u) * sizeof(float));
    ds4_gpu_tensor *b = upload(NULL, (N + 1u) * sizeof(float));
    CHECK(a && b);
    float *x = ds4_gpu_tensor_contents(a), *y = ds4_gpu_tensor_contents(b);
    const uint32_t starts[] = {0, 126, 32766, 1048318};
    for (uint32_t kind = 0; kind < 2; kind++) for (uint32_t inverse = 0; inverse < 2; inverse++) {
        for (size_t i = 0; i < sizeof(starts) / sizeof(*starts); i++) {
            for (size_t j = 0; j < N; j++) x[j] = y[j] = bf16(random_value());
            x[N] = y[N] = 12345;
            CHECK(ds4_gpu_begin_commands());
            CHECK(ds4_gpu_dsv41_rope_stride(a, WIDTH, HEADS, ROWS, starts[i], 2, kind, inverse));
            for (uint32_t row = 0; row < ROWS; row++) {
                ds4_gpu_tensor *v = ds4_gpu_tensor_view(b, (uint64_t)row * WIDTH * HEADS * 4u,
                                                       WIDTH * HEADS * 4u);
                CHECK(v && ds4_gpu_dsv41_rope(v, WIDTH, HEADS, 1, starts[i] + row * 2u, kind, inverse));
                ds4_gpu_tensor_free(v);
            }
            CHECK(ds4_gpu_end_commands());
            CHECK(!memcmp(x, y, (N + 1u) * sizeof(float)) && x[N] == 12345);
        }
    }
    CHECK(!ds4_gpu_dsv41_rope_stride(a, WIDTH, HEADS, ROWS, 0, 0, true, false));
    CHECK(!ds4_gpu_dsv41_rope_stride(a, WIDTH, HEADS, ROWS, 1048320, 2, true, false));
    CHECK(!ds4_gpu_dsv41_rope_stride(a, WIDTH, HEADS, ROWS + 1, 0, 2, true, false));
    CHECK(!ds4_gpu_dsv41_rope_stride(a, WIDTH, HEADS, ROWS, 0, UINT32_MAX, true, false));
    ds4_gpu_tensor_free(b); ds4_gpu_tensor_free(a);
    fprintf(stderr, "V4.1 strided RoPE: exact row parity through 1M positions, guards PASS\n");
    return 1;
}

static int check_pool(void) {
    enum { D = 512, ROWS = 257, PAIRS = ROWS / 2 };
    float *kv = malloc(ROWS * D * sizeof(float)), *scores = malloc(ROWS * D * sizeof(float));
    float *got = malloc(PAIRS * D * sizeof(float)), *reference = malloc(PAIRS * D * sizeof(float));
    CHECK(kv && scores && got && reference);
    for (int i = 0; i < ROWS * D; i++) { kv[i] = random_value(); scores[i] = random_value() * 25; }
    ds4_gpu_tensor *kt = upload(kv, ROWS * D * sizeof(float));
    ds4_gpu_tensor *st = upload(scores, ROWS * D * sizeof(float));
    ds4_gpu_tensor *pk = upload(NULL, D * sizeof(float)), *ps = upload(NULL, D * sizeof(float));
    ds4_gpu_tensor *out = upload(NULL, PAIRS * D * sizeof(float));
    CHECK(kt && st && pk && ps && out);
    const uint32_t chunks[] = {257, 1, 2, 3, 17, 127, 128, 129};
    for (size_t c = 0; c < sizeof(chunks) / sizeof(*chunks); c++) {
        CHECK(ds4_gpu_tensor_fill_f32(pk, NAN, D));
        CHECK(ds4_gpu_tensor_fill_f32(ps, NAN, D));
        CHECK(ds4_gpu_begin_commands());
        for (uint32_t start = 0; start < ROWS;) {
            uint32_t n = chunks[c] < ROWS - start ? chunks[c] : ROWS - start;
            uint32_t pairs = (n + (start & 1u)) / 2;
            ds4_gpu_tensor *k = ds4_gpu_tensor_view(kt, (uint64_t)start * D * 4, (uint64_t)n * D * 4);
            ds4_gpu_tensor *s = ds4_gpu_tensor_view(st, (uint64_t)start * D * 4, (uint64_t)n * D * 4);
            ds4_gpu_tensor *o = pairs ? ds4_gpu_tensor_view(out, (uint64_t)(start / 2) * D * 4,
                                                         (uint64_t)pairs * D * 4) : NULL;
            CHECK(k && s && (!pairs || o));
            CHECK(ds4_gpu_dsv41_pool2(o, k, s, pk, ps, D, n, start));
            ds4_gpu_tensor_free(k); ds4_gpu_tensor_free(s); ds4_gpu_tensor_free(o);
            start += n;
        }
        CHECK(ds4_gpu_end_commands());
        CHECK(ds4_gpu_tensor_read(out, 0, got, PAIRS * D * sizeof(float)));
        if (!c) {
            memcpy(reference, got, PAIRS * D * sizeof(float));
            for (int p = 0; p < PAIRS; p++) for (int i = 0; i < D; i++) {
                const int a = 2 * p * D + i, b = a + D;
                const double gate = 1 / (1 + exp((double)scores[b] - scores[a]));
                const float expected = bf16((float)(kv[a] * gate + kv[b] * (1 - gate)));
                CHECK(isfinite(got[p * D + i]));
                CHECK(fabsf(got[p * D + i] - expected) <= fmaxf(1e-6f, fabsf(expected) / 128));
            }
        } else CHECK(!memcmp(reference, got, PAIRS * D * sizeof(float)));
        float tail[D];
        CHECK(ds4_gpu_tensor_read(pk, 0, tail, sizeof(tail)));
        CHECK(!memcmp(tail, kv + (ROWS - 1) * D, sizeof(tail)));
        CHECK(ds4_gpu_tensor_read(ps, 0, tail, sizeof(tail)));
        CHECK(!memcmp(tail, scores + (ROWS - 1) * D, sizeof(tail)));
    }
    CHECK(ds4_gpu_begin_commands());
    CHECK(ds4_gpu_dsv41_pool2(out, kt, st, pk, ps, D, ROWS - 1, 0));
    CHECK(ds4_gpu_end_commands());
    CHECK(!memcmp(ds4_gpu_tensor_contents(pk), kv + (ROWS - 3) * D, D * sizeof(float)));
    CHECK(!memcmp(ds4_gpu_tensor_contents(ps), scores + (ROWS - 3) * D, D * sizeof(float)));
    CHECK(!memcmp(ds4_gpu_tensor_contents(out), reference, PAIRS * D * sizeof(float)));
    ds4_gpu_tensor_free(kt); ds4_gpu_tensor_free(st); ds4_gpu_tensor_free(pk);
    ds4_gpu_tensor_free(ps); ds4_gpu_tensor_free(out);
    free(kv); free(scores); free(got); free(reference);
    fprintf(stderr, "V4.1 pair pooling: exact across eight chunk sizes\n");
    return 1;
}

typedef struct { float score; uint32_t index; } candidate;
static int candidate_desc(const void *a, const void *b) {
    const candidate *x = a, *y = b;
    return x->score > y->score ? -1 : x->score < y->score ? 1 :
           x->index < y->index ? -1 : x->index > y->index;
}

static int check_candidates(void) {
    const uint32_t widths[] = {1, 7, 8, 9, 127, 16385, 17017};
    for (size_t wi = 0; wi < sizeof(widths) / sizeof(*widths); wi++) {
        const uint32_t n = widths[wi], blocks = (n + 7) / 8, rows = 17;
        const uint32_t top = blocks < 2048 ? blocks : 2048;
        float *scores = malloc((size_t)n * rows * 4), *got = malloc((size_t)n * rows * 4);
        float *maxima = malloc((size_t)blocks * rows * 4);
        candidate *sorted = malloc(blocks * sizeof(candidate));
        uint8_t *kept = malloc(blocks);
        CHECK(scores && got && maxima && sorted && kept);
        /* Unique finite values avoid unspecified top-k tie ordering. */
        for (uint32_t r = 0; r < rows; r++) for (uint32_t i = 0; i < n; i++)
            scores[(size_t)r * n + i] = (float)((i * 7919u + r * 1009u) % 104729u) - 50000;
        ds4_gpu_tensor *s = upload(NULL, (size_t)n * rows * 4);
        ds4_gpu_tensor *b = upload(NULL, (size_t)blocks * rows * 4);
        ds4_gpu_tensor *t = upload(NULL, (size_t)top * rows * 4);
        ds4_gpu_tensor *m = upload(NULL, (size_t)blocks * rows * 4);
        CHECK(s && b && t && m);
        for (uint32_t ratio = 1; ratio <= 2; ratio++) for (int late = 0; late < 2; late++) {
            const uint32_t start = late ? n * ratio - 1 : 0;
            CHECK(ds4_gpu_tensor_write(s, 0, scores, (size_t)n * rows * 4));
            CHECK(ds4_gpu_begin_commands());
            CHECK(ds4_gpu_dsv41_candidate_blocks(b, s, n, rows, start, ratio));
            CHECK(ds4_gpu_indexer_topk_tensor(t, b, blocks, rows, top));
            CHECK(ds4_gpu_dsv4_topk_mask_tensor(m, t, blocks, rows, top));
            CHECK(ds4_gpu_dsv41_candidate_filter(s, m, n, rows, start, ratio, 0));
            CHECK(ds4_gpu_end_commands());
            CHECK(ds4_gpu_tensor_read(b, 0, maxima, (size_t)blocks * rows * 4));
            CHECK(ds4_gpu_tensor_read(s, 0, got, (size_t)n * rows * 4));
            for (uint32_t r = 0; r < rows; r++) {
                uint32_t visible = (start + r + 1) / ratio;
                if (visible > n) visible = n;
                for (uint32_t j = 0; j < blocks; j++) {
                    float best = -INFINITY;
                    for (uint32_t i = j * 8; i < (j + 1) * 8 && i < visible; i++)
                        best = fmaxf(best, scores[(size_t)r * n + i]);
                    if (visible && j == (visible - 1) / 8) best = INFINITY;
                    CHECK(maxima[(size_t)r * blocks + j] == best);
                    sorted[j] = (candidate){best, j};
                }
                qsort(sorted, blocks, sizeof(*sorted), candidate_desc);
                memset(kept, 0, blocks);
                for (uint32_t j = 0; j < top; j++)
                    if (sorted[j].score > -INFINITY) kept[sorted[j].index] = 1;
                for (uint32_t i = 0; i < n; i++) {
                    float expected = i < visible && kept[i / 8] ? scores[(size_t)r * n + i] : -INFINITY;
                    CHECK(got[(size_t)r * n + i] == expected);
                }
            }
        }
        CHECK(!ds4_gpu_dsv41_candidate_blocks(b, s, n, rows, UINT32_MAX, 1));
        CHECK(!ds4_gpu_dsv41_candidate_blocks(b, s, n, rows, 0, 0));
        CHECK(!ds4_gpu_dsv41_candidate_filter(s, m, n, rows + 1, 0, 1, 0));
        ds4_gpu_tensor_free(s); ds4_gpu_tensor_free(b); ds4_gpu_tensor_free(t); ds4_gpu_tensor_free(m);
        free(scores); free(got); free(maxima); free(sorted); free(kept);
    }
    fprintf(stderr, "V4.1 causal candidate blocks and filtering: exact\n");
    return 1;
}

static int check_sparse_gather(void) {
    const uint32_t sizes[] = {1, 3, 511, 512, 513, 8193, 17017};
    for (size_t i = 0; i < sizeof(sizes) / sizeof(*sizes); i++) {
        const uint32_t rows = sizes[i], selected = rows < 512 ? rows : 512;
        ds4_gpu_tensor *source = upload(NULL, (uint64_t)rows * 512 * 4);
        ds4_gpu_tensor *ids = upload(NULL, selected * 4);
        ds4_gpu_tensor *out = upload(NULL, (uint64_t)selected * 512 * 4);
        CHECK(source && ids && out);
        float *input = ds4_gpu_tensor_contents(source);
        int32_t *indices = ds4_gpu_tensor_contents(ids);
        for (uint32_t r = 0; r < rows; r++) for (uint32_t c = 0; c < 512; c++)
            input[(size_t)r * 512 + c] = (float)r + (float)c / 512.0f;
        for (uint32_t r = 0; r < selected; r++) indices[r] = (int32_t)(rows - 1u - r);
        CHECK(ds4_gpu_dsv41_gather_kv(out, source, ids, rows, selected));
        const float *result = ds4_gpu_tensor_contents(out);
        for (uint32_t r = 0; r < selected; r++)
            CHECK(!memcmp(result + r * 512, input + (size_t)indices[r] * 512, 512 * 4));
        CHECK(!ds4_gpu_dsv41_gather_kv(out, source, ids, rows + 1, selected));
        CHECK(!ds4_gpu_dsv41_gather_kv(out, source, ids, rows, selected + 1));
        ds4_gpu_tensor_free(source); ds4_gpu_tensor_free(ids); ds4_gpu_tensor_free(out);
    }
    fprintf(stderr, "V4.1 bounded sparse KV gather: exact\n");
    return 1;
}

static int check_attention_output(bool large) {
    enum { GROUP = 4096, RANK = 1024, GROUPS = 8, OUT = 5120 };
    const uint32_t ROWS = large ? 8192u : 513u;
    typedef struct { uint16_t d; int8_t qs[32]; } q8_block;
    const uint64_t a_bytes = (uint64_t)GROUPS * RANK * GROUP / 32 * sizeof(q8_block);
    const uint64_t b_bytes = (uint64_t)OUT * GROUPS * RANK / 32 * sizeof(q8_block);
    const uint64_t xb = (uint64_t)ROWS * GROUPS * GROUP * 4;
    const uint64_t lb = (uint64_t)ROWS * GROUPS * RANK * 4, ob = (uint64_t)ROWS * OUT * 4;
    void *model = NULL;
    CHECK(posix_memalign(&model, getpagesize(), a_bytes + b_bytes) == 0);
    q8_block *w = model;
    for (uint64_t i = 0; i < (a_bytes + b_bytes) / sizeof(*w); i++) {
        w[i].d = 0x2000; /* Exact 1/128 scale and binary-fraction inputs. */
        for (int j = 0; j < 32; j++) w[i].qs[j] = (int)(random_value() * 8192) % 8;
    }
    float *x = malloc(xb), *reference = malloc(ob), *reference_low = malloc(lb);
    CHECK(x && reference && reference_low);
    for (uint64_t i = 0; i < xb / 4; i++) x[i] = (int)(random_value() * 8192) % 51 / 256.0f;
    ds4_gpu_tensor *xt = upload(x, xb), *low = upload(NULL, lb), *out = upload(NULL, ob);
    CHECK(xt && low && out && ds4_gpu_set_model_map(model, a_bytes + b_bytes));
    CHECK(ds4_gpu_begin_commands());
    for (uint32_t r = 0; r < ROWS; r++) {
        ds4_gpu_tensor *xr = ds4_gpu_tensor_view(xt, (uint64_t)r * GROUPS * GROUP * 4,
                                                GROUPS * GROUP * 4);
        ds4_gpu_tensor *lr = ds4_gpu_tensor_view(low, (uint64_t)r * GROUPS * RANK * 4,
                                                GROUPS * RANK * 4);
        ds4_gpu_tensor *yr = ds4_gpu_tensor_view(out, (uint64_t)r * OUT * 4, OUT * 4);
        CHECK(xr && lr && yr);
        CHECK(ds4_gpu_attention_output_low_q8_tensor(lr, model, a_bytes + b_bytes,
            0, GROUP, RANK, GROUPS, xr));
        CHECK(ds4_gpu_dsv41_quantize(lr, GROUPS * RANK, 1, DS4_V41_BF16));
        CHECK(ds4_gpu_matmul_q8_0_tensor(yr, model, a_bytes + b_bytes,
            a_bytes, GROUPS * RANK, OUT, lr, 1));
        ds4_gpu_tensor_free(xr); ds4_gpu_tensor_free(lr); ds4_gpu_tensor_free(yr);
    }
    CHECK(ds4_gpu_end_commands());
    CHECK(ds4_gpu_tensor_read(out, 0, reference, ob));
    CHECK(ds4_gpu_tensor_read(low, 0, reference_low, lb));
    const uint32_t sizes[] = {1, 2, 3, 5, 6, 8, 31, 32, 63, 64, 65, 257, 512, 513, 8191, 8192};
    for (unsigned n = 0; n < sizeof(sizes) / sizeof(*sizes); n++) {
        const uint32_t rows = sizes[n];
        if (rows > ROWS) break;
        CHECK(ds4_gpu_tensor_fill_f32(out, NAN, ob / 4));
        CHECK(ds4_gpu_tensor_fill_f32(low, NAN, lb / 4));
        CHECK(ds4_gpu_dsv41_attention_output_batch(out, low, model, a_bytes + b_bytes,
            0, a_bytes, xt, rows));
        const float *got = ds4_gpu_tensor_contents(out);
        const float *got_low = ds4_gpu_tensor_contents(low);
        CHECK(!memcmp(got_low, reference_low, (uint64_t)rows * GROUPS * RANK * 4));
        for (uint64_t i = 0; i < (uint64_t)rows * OUT; i++) {
            if (!isfinite(got[i]) || fabsf(got[i] - reference[i]) > 2e-5f * (1 + fabsf(reference[i])))
                fprintf(stderr, "attention output rows=%u index=%llu actual=%.9g reference=%.9g\n",
                    rows, (unsigned long long)i, got[i], reference[i]);
            CHECK(isfinite(got[i]) && fabsf(got[i] - reference[i]) <= 2e-5f * (1 + fabsf(reference[i])));
        }
        if (rows < ROWS) CHECK(isnan(got[(uint64_t)rows * OUT]));
        fprintf(stderr, "V4.1 batched Q8 output, BF16 boundary, rows=%u: PASS\n", rows);
    }
    CHECK(!ds4_gpu_dsv41_attention_output_batch(out, low, model, a_bytes + b_bytes - 1,
        0, a_bytes, xt, ROWS));
    CHECK(!ds4_gpu_dsv41_attention_output_batch(out, low, model, a_bytes + b_bytes,
        0, a_bytes, xt, ROWS + 1));
#ifndef DS4_ROCM_BUILD
    ds4_gpu_tensor *packed = upload(NULL, xb / 2);
    ds4_gpu_tensor *partial[2] = {upload(NULL, ob), upload(NULL, ob)};
    CHECK(packed && partial[0] && partial[1]);
    const uint32_t tp_sizes[] = {1, 2, 3, 5, 6, 8, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257, 512, 513};
    const uint32_t q_half = GROUPS / 2 * GROUP, low_half = GROUPS / 2 * RANK;
    for (size_t n = 0; n < sizeof(tp_sizes) / sizeof(*tp_sizes); n++) {
        const uint32_t rows = tp_sizes[n];
        for (uint32_t rank = 0; rank < 2; rank++) {
            float *px = ds4_gpu_tensor_contents(packed);
            for (uint32_t t = 0; t < rows; t++)
                memcpy(px + (size_t)t * q_half, x + (size_t)t * 2 * q_half + rank * q_half,
                       q_half * sizeof(float));
            CHECK(ds4_gpu_tensor_fill_f32(partial[rank], NAN, ob / 4));
            CHECK(ds4_gpu_tensor_fill_f32(low, NAN, lb / 4));
            CHECK(ds4_gpu_dsv41_attention_output_tp_batch(partial[rank], low,
                model, a_bytes + b_bytes, 0, a_bytes, packed, rows, rank));
            const float *got_low = ds4_gpu_tensor_contents(low);
            const float *got = ds4_gpu_tensor_contents(partial[rank]);
            for (uint32_t t = 0; t < rows; t++) {
                CHECK(!memcmp(got_low + (size_t)t * low_half,
                    reference_low + (size_t)t * 2 * low_half + rank * low_half,
                    low_half * sizeof(float)));
                /* Independent double sums sample both ends and interior output
                 * rows, including rank one's strided input slice. */
                const uint32_t columns[] = {0, 1, 31, 64, 997, 4095, OUT - 1};
                for (size_t c = 0; c < sizeof(columns) / sizeof(*columns); c++) {
                    const uint32_t col = columns[c];
                    const q8_block *bw = (const q8_block *)((const char *)model + a_bytes) +
                        (size_t)col * 2 * low_half / 32 + rank * low_half / 32;
                    double sum = 0;
                    double rounding_bound = 0;
#ifndef __APPLE__
                    /* CUDA's established Q8 dot first rounds each activation
                     * block to signed bytes. TP must keep that same boundary. */
                    for (uint32_t k = 0; k < low_half; k += 32) {
                        const float *v = reference_low + (size_t)t * 2 * low_half + rank * low_half + k;
                        float amax = 0;
                        for (uint32_t j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(v[j]));
                        const float scale = amax / 127.0f;
                        const float inv = scale ? 1.0f / scale : 0;
                        int dot = 0;
                        for (uint32_t j = 0; j < 32; j++) {
                            dot += bw[k / 32].qs[j] * (int)lrintf(v[j] * inv);
                            /* The fast GPU reciprocal can approach an exact
                             * half-integer from the other side. Bound only
                             * those ambiguous bins, not arbitrary dot error. */
                            const double q = amax ? fabs((double)v[j] * 127.0 / amax) : 0;
                            if (q - floor(q) == 0.5)
                                rounding_bound += abs(bw[k / 32].qs[j]) * (double)scale / 128.0;
                        }
                        sum += dot * (double)scale / 128.0;
                    }
#else
                    for (uint32_t k = 0; k < low_half; k++)
                        sum += bw[k / 32].qs[k % 32] / 128.0 *
                            reference_low[(size_t)t * 2 * low_half + rank * low_half + k];
#endif
                    if (!isfinite(got[(size_t)t * OUT + col]) ||
                        fabs(got[(size_t)t * OUT + col] - sum) > rounding_bound + 2e-5 * (1 + fabs(sum)))
                        fprintf(stderr, "TP output rank=%u rows=%u row=%u col=%u actual=%.9g oracle=%.9g tie_bound=%.9g\n",
                            rank, rows, t, col, got[(size_t)t * OUT + col], sum, rounding_bound);
                    CHECK(isfinite(got[(size_t)t * OUT + col]) &&
                        fabs(got[(size_t)t * OUT + col] - sum) <= rounding_bound + 2e-5 * (1 + fabs(sum)));
                }
            }
            CHECK(isnan(got_low[(size_t)rows * low_half]));
            if (rows < ROWS) CHECK(isnan(got[(size_t)rows * OUT]));
        }
        const float *a = ds4_gpu_tensor_contents(partial[0]);
        const float *b = ds4_gpu_tensor_contents(partial[1]);
        for (size_t i = 0; i < (size_t)rows * OUT; i++)
            CHECK(isfinite(a[i]) && isfinite(b[i]) &&
                fabsf(a[i] + b[i] - reference[i]) <= 2e-5f * (1 + fabsf(reference[i])));
        fprintf(stderr, "V4.1 TP Q8 output, both ranks, BF16 and double oracle, rows=%u: PASS\n", rows);
    }
    CHECK(!ds4_gpu_dsv41_attention_output_tp_batch(out, low, model, a_bytes + b_bytes,
        0, a_bytes, packed, 1, 2));
    CHECK(!ds4_gpu_dsv41_attention_output_tp_batch(out, low, model, a_bytes + b_bytes - 1,
        0, a_bytes, packed, 1, 1));
    CHECK(!ds4_gpu_dsv41_attention_output_tp_batch(out, low, model, a_bytes + b_bytes,
        0, a_bytes, packed, ROWS + 1, 0));
    ds4_gpu_tensor_free(partial[0]); ds4_gpu_tensor_free(partial[1]); ds4_gpu_tensor_free(packed);
#else
    fprintf(stderr, "V4.1 TP output projection: skipped (ROCm has no network TP)\n");
#endif
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(low); ds4_gpu_tensor_free(out);
    ds4_gpu_cleanup();
    free(x); free(reference); free(reference_low); free(model);
    return 1;
}

static int check_indexer_batch(void) {
    enum { KEYS = 1025, ROWS = 33, HEADS = 32, DIM = 128 };
    ds4_gpu_tensor *qt = upload(NULL, (size_t)ROWS * HEADS * DIM * sizeof(float));
    ds4_gpu_tensor *kt = upload(NULL, (size_t)KEYS * DIM * sizeof(float));
    ds4_gpu_tensor *wt = upload(NULL, (size_t)ROWS * HEADS * sizeof(float));
    ds4_gpu_tensor *st = upload(NULL, (size_t)(ROWS * KEYS + 1) * sizeof(float));
    const uint64_t packed_bytes = ds4_gpu_dsv41_indexer_packed_bytes(KEYS, ROWS);
    ds4_gpu_tensor *pt = upload(NULL, (size_t)packed_bytes + 4);
    CHECK(qt && kt && wt && st && pt);
#ifndef __APPLE__
    ds4_gpu_tensor *reference = upload(NULL, (size_t)KEYS * sizeof(float));
    CHECK(reference);
#endif
    float *q = ds4_gpu_tensor_contents(qt), *k = ds4_gpu_tensor_contents(kt);
    float *w = ds4_gpu_tensor_contents(wt), *s = ds4_gpu_tensor_contents(st);
    CHECK(q && k && w && s);
    const bool packed_available = ds4_gpu_dsv41_tensor_ops_available();
    for (uint32_t range = 0; range < 3; range++) {
        for (size_t i = 0; i < (size_t)ROWS * HEADS * DIM; i++)
            q[i] = ldexpf(random_value(), range ? (int)((i / 32) % 41) - 20 : 0);
        for (size_t i = 0; i < (size_t)KEYS * DIM; i++)
            k[i] = ldexpf(random_value(), range ? (int)((i / 32) % 41) - 20 : 0);
        for (size_t i = 0; i < (size_t)ROWS * HEADS; i++) w[i] = bf16(random_value());
        CHECK(ds4_gpu_dsv41_quantize(qt, DIM, ROWS * HEADS, DS4_V41_FP4_E8M0));
        CHECK(ds4_gpu_dsv41_quantize(kt, DIM, KEYS, DS4_V41_FP4_E8M0));
        if (range == 2) { q[0] = 1.0001f; k[67u * DIM] = 1.0003f; }
        uint32_t *packed_guard = (uint32_t *)((char *)ds4_gpu_tensor_contents(pt) + packed_bytes);
        *packed_guard = 0xabcdef01;
        if (packed_available) CHECK(ds4_gpu_dsv41_indexer_pack(pt, qt, kt, KEYS, ROWS));
        CHECK(*packed_guard == 0xabcdef01);
        const uint32_t counts[] = {1, 7, 8, 9, 31, 32, 33};
        for (uint32_t ratio = 1; ratio <= 2; ratio++) {
            for (uint32_t early = 0; early <= 1; early++) {
                for (size_t ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
                    const uint32_t rows = counts[ci];
                    const uint32_t start = early ? 0 : KEYS * ratio - rows;
                    for (uint32_t mode = 0; mode < (packed_available ? 3u : 1u); mode++) {
                        const uint32_t offset = mode == 2 && rows < ROWS ? 1u : 0u;
                        ds4_gpu_tensor *qv = ds4_gpu_tensor_view(qt, (uint64_t)offset * HEADS * DIM * 4u,
                            (uint64_t)rows * HEADS * DIM * 4u);
                        ds4_gpu_tensor *wv = ds4_gpu_tensor_view(wt, (uint64_t)offset * HEADS * 4u,
                            (uint64_t)rows * HEADS * 4u);
                        CHECK(qv && wv);
                        s[(size_t)rows * KEYS] = 12345;
                        CHECK(mode ? ds4_gpu_dsv41_indexer_scores_packed(st, qv, wv, kt, pt,
                            KEYS, rows, start, ratio, ROWS, offset) :
                            ds4_gpu_dsv41_indexer_scores_batch(st, qv, wv, kt, KEYS, rows, start, ratio));
                        CHECK(ds4_gpu_synchronize());
                        ds4_gpu_tensor_free(qv); ds4_gpu_tensor_free(wv);
                        CHECK(s[(size_t)rows * KEYS] == 12345);
                        CHECK(*packed_guard == 0xabcdef01);
#ifndef __APPLE__
                        /* Compare the CUDA warp scorer with the original
                         * shared-memory reduction, not only a loose oracle. */
                        ds4_gpu_set_quality(true);
                        for (uint32_t t = 0; t < rows; t++) {
                            const uint32_t visible = (start + t + 1u) / ratio;
                            if (!visible) continue;
                            ds4_gpu_tensor *qr = ds4_gpu_tensor_view(qt,
                                (uint64_t)t * HEADS * DIM * 4u, HEADS * DIM * 4u);
                            ds4_gpu_tensor *wr = ds4_gpu_tensor_view(wt,
                                (uint64_t)t * HEADS * 4u, HEADS * 4u);
                            CHECK(qr && wr && ds4_gpu_glm_indexer_score_one_tensor(
                                reference, qr, wr, kt, visible, HEADS, DIM, 1.0f / 64.0f, false));
                            CHECK(ds4_gpu_synchronize());
                            CHECK(!memcmp(s + (size_t)t * KEYS,
                                ds4_gpu_tensor_contents(reference), visible * sizeof(float)));
                            ds4_gpu_tensor_free(qr); ds4_gpu_tensor_free(wr);
                        }
                        ds4_gpu_set_quality(false);
#endif
                        double worst = 0;
                        for (uint32_t t = 0; t < rows; t++) {
                            const uint32_t visible = (start + t + 1u) / ratio;
                            for (uint32_t j = 0; j < KEYS; j++) {
                                const float actual = s[(size_t)t * KEYS + j];
                                if (j >= visible) { CHECK(actual == -INFINITY); continue; }
                                double expected = 0, magnitude = 0;
                                for (uint32_t h = 0; h < HEADS; h++) {
                                    double dot = 0;
                                    for (uint32_t d = 0; d < DIM; d++)
                                        dot += (double)q[((size_t)(t + offset) * HEADS + h) * DIM + d] *
                                               k[(size_t)j * DIM + d];
                                    const double term = fmax(dot / 64.0, 0) * w[(t + offset) * HEADS + h];
                                    expected += term;
                                    magnitude += fabs(term);
                                }
                                CHECK(isfinite(actual));
                                const double error = fabs(actual - expected) / fmax(magnitude, 1);
                                worst = fmax(worst, error);
                                CHECK(error < 0.00001);
                            }
                        }
                        fprintf(stderr, "V4.1 index scores mode=%u range=%u ratio=%u start=%u rows=%u error=%.9g\n",
                                mode, range, ratio, start, rows, worst);
                    }
                }
            }
        }
    }
    CHECK(!ds4_gpu_dsv41_indexer_scores_batch(st, qt, wt, kt, KEYS, ROWS, 0, 0));
    CHECK(!ds4_gpu_dsv41_indexer_scores_batch(st, qt, wt, kt, KEYS, ROWS, 0, 4));
    CHECK(!ds4_gpu_dsv41_indexer_scores_batch(st, qt, wt, kt, KEYS, ROWS, UINT32_MAX, 1));
    CHECK(!ds4_gpu_dsv41_indexer_scores_batch(st, qt, wt, kt, KEYS, ROWS + 1, 0, 1));
    CHECK(!ds4_gpu_dsv41_indexer_scores_batch(st, qt, wt, kt, KEYS, ROWS, KEYS, 1));
    CHECK(!ds4_gpu_dsv41_indexer_pack(st, qt, kt, KEYS, ROWS));
    CHECK(!ds4_gpu_dsv41_indexer_pack(pt, qt, kt, KEYS, ROWS + 1));
    CHECK(!ds4_gpu_dsv41_indexer_scores_packed(st, qt, wt, kt, pt,
        KEYS, ROWS, 0, 0, ROWS, 0));
    CHECK(!ds4_gpu_dsv41_indexer_scores_packed(st, qt, wt, kt, pt,
        KEYS, ROWS, 0, 1, ROWS, 1));
    CHECK(!ds4_gpu_dsv41_indexer_scores_packed(st, qt, wt, kt, pt,
        KEYS, ROWS, 0, 1, UINT32_MAX, 0));
    CHECK(!ds4_gpu_dsv41_indexer_scores_packed(st, qt, wt, kt, st,
        KEYS, ROWS, 0, 1, ROWS, 0));
    ds4_gpu_tensor_free(pt);
#ifndef __APPLE__
    ds4_gpu_tensor_free(reference);
#endif
    ds4_gpu_tensor_free(st); ds4_gpu_tensor_free(wt);
    ds4_gpu_tensor_free(kt); ds4_gpu_tensor_free(qt);
    fprintf(stderr, "V4.1 causal batched index scores: double-precision oracle PASS\n");
    return 1;
}

static double monotonic_seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

static int check_embedding(void) {
    enum { VOCAB = 19, WIDTH = 5120, HC = 4, ROWS = 257 };
    const size_t row_bytes = WIDTH * HC * sizeof(float);
    const size_t weight_bytes = VOCAB * WIDTH * sizeof(uint16_t);
    uint16_t *model = NULL;
    CHECK(posix_memalign((void **)&model, getpagesize(), weight_bytes) == 0);
    for (unsigned i = 0; i < VOCAB * WIDTH; i++)
        model[i] = (uint16_t)(0x2000u | (i % 1024u) | ((i & 1u) ? 0x8000u : 0));
    int32_t ids[ROWS];
    for (unsigned i = 0; i < ROWS; i++) ids[i] = (int32_t)((i * 7u) % VOCAB);
    CHECK(ds4_gpu_set_model_map(model, weight_bytes));
    ds4_gpu_tensor *tokens = upload(ids, sizeof(ids));
    ds4_gpu_tensor *out = upload(NULL, ROWS * row_bytes + sizeof(float));
    ds4_gpu_tensor *ref = upload(NULL, VOCAB * row_bytes);
    CHECK(tokens && out && ref);
    for (unsigned token = 0; token < VOCAB; token++) {
        ds4_gpu_tensor *row = ds4_gpu_tensor_view(ref, token * row_bytes, row_bytes);
        CHECK(row && ds4_gpu_embed_token_hc_tensor(row, model, weight_bytes, 0,
                                                   VOCAB, token, WIDTH, HC));
        ds4_gpu_tensor_free(row);
    }
    const unsigned counts[] = {1, 7, 127, 128, ROWS};
    const float marker = 12345;
    float *actual = malloc(ROWS * row_bytes + sizeof(float));
    float *expected = malloc(VOCAB * row_bytes);
    CHECK(actual && expected && ds4_gpu_tensor_read(ref, 0, expected, VOCAB * row_bytes));
    for (unsigned i = 0; i < sizeof(counts) / sizeof(*counts); i++) {
        const unsigned rows = counts[i];
        CHECK(ds4_gpu_tensor_write(out, rows * row_bytes, &marker, sizeof(marker)));
        CHECK(ds4_gpu_embed_tokens_hc_tensor(out, tokens, model, weight_bytes, 0,
                                             VOCAB, rows, WIDTH, HC));
        CHECK(ds4_gpu_tensor_read(out, 0, actual, rows * row_bytes + sizeof(float)));
        CHECK(actual[rows * WIDTH * HC] == marker);
        for (unsigned row = 0; row < rows; row++) {
            CHECK(!memcmp(actual + row * WIDTH * HC, expected + ids[row] * WIDTH * HC, row_bytes));
            for (unsigned hc = 0; hc < HC; hc++) for (unsigned d = 0; d < WIDTH; d++) {
                const unsigned w = (unsigned)ids[row] * WIDTH + d;
                const float value = (1.0f + (w % 1024u) / 1024.0f) / 128.0f * ((w & 1u) ? -1 : 1);
                CHECK(actual[(row * HC + hc) * WIDTH + d] == value);
            }
        }
        fprintf(stderr, "V4.1 batched embedding rows=%u: exact scalar and independent oracle PASS\n", rows);
    }
    free(expected); free(actual);
    ds4_gpu_tensor_free(ref); ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(tokens);
    ds4_gpu_cleanup(); free(model);
    CHECK(ds4_gpu_init());
    return 1;
}

static int check_index_projection(void) {
    enum { MAX_ROWS = 4096 };
    const uint32_t widths[] = {1280, 5120, 5120, 512}, outputs[] = {4096, 32, 512, 128};
    const uint32_t counts[] = {1, 31, 64, 65, 513, 2048, MAX_ROWS};
    for (uint32_t shape = 0; shape < 4; shape++) {
        const uint32_t width = widths[shape], output = outputs[shape];
        const size_t weight_bytes = (size_t)width * output * sizeof(uint16_t);
        void *model = NULL;
        CHECK(posix_memalign(&model, getpagesize(), weight_bytes) == 0);
        uint16_t *weights = model;
        for (size_t i = 0; i < weight_bytes / sizeof(*weights); i++) {
            const int value = (int)(random_value() * 8192);
            weights[i] = (uint16_t)(0x2000u | ((unsigned)abs(value) % 1024u) |
                                    (value < 0 ? 0x8000u : 0));
        }
        CHECK(ds4_gpu_set_model_map(model, weight_bytes));
        ds4_gpu_tensor *in = upload(NULL, (size_t)MAX_ROWS * width * sizeof(float));
        ds4_gpu_tensor *out = upload(NULL, ((size_t)MAX_ROWS * output + 1) * sizeof(float));
        ds4_gpu_tensor *ref = upload(NULL, (size_t)MAX_ROWS * output * sizeof(float));
        CHECK(in && out && ref);
        float *x = ds4_gpu_tensor_contents(in), *y = ds4_gpu_tensor_contents(out);
        const float *expected = ds4_gpu_tensor_contents(ref);
        for (size_t i = 0; i < (size_t)MAX_ROWS * width; i++) x[i] = bf16(random_value());
        for (size_t ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
            const uint32_t rows = counts[ci];
            y[(size_t)rows * output] = 12345;
            CHECK(ds4_gpu_begin_commands());
            for (uint32_t t = 0; t < rows; t++) {
                ds4_gpu_tensor *xr = ds4_gpu_tensor_view(in, (size_t)t * width * 4, width * 4);
                ds4_gpu_tensor *yr = ds4_gpu_tensor_view(ref, (size_t)t * output * 4, output * 4);
                CHECK(xr && yr && ds4_gpu_matmul_f16_tensor(yr, model, weight_bytes, 0, width, output, xr, 1));
                ds4_gpu_tensor_free(yr); ds4_gpu_tensor_free(xr);
            }
            CHECK(ds4_gpu_end_commands());
            const double start = monotonic_seconds();
            CHECK(ds4_gpu_dsv41_projection_rows(out, model, weight_bytes, 0, width, output, rows, in));
            CHECK(ds4_gpu_synchronize());
            const double seconds = monotonic_seconds() - start;
            CHECK(y[(size_t)rows * output] == 12345);
            double error = 0, norm = 0;
            for (size_t i = 0; i < (size_t)rows * output; i++) {
                CHECK(isfinite(y[i]) && isfinite(expected[i]));
                const double d = (double)y[i] - expected[i];
                error += d * d; norm += (double)expected[i] * expected[i];
            }
            const double relative = sqrt(error / fmax(norm, 1e-30));
            fprintf(stderr, "V4.1 exact F16 projection width=%u output=%u rows=%u RMS=%.9g %.3f ms\n",
                width, output, rows, relative, seconds * 1000);
            CHECK(memcmp(y, expected, (size_t)rows * output * sizeof(float)) == 0);
            for (uint32_t j = 0; j < 8; j++) {
                const uint32_t o = j * (output / 8), t = rows - 1;
                double sum = 0, magnitude = 0;
                for (uint32_t k = 0; k < width; k++) {
                    const uint16_t w = weights[(size_t)o * width + k];
                    const float value = (1.0f + (w & 1023) / 1024.0f) / 128.0f *
                                        (w & 0x8000 ? -1 : 1);
                    const double term = (double)value * x[(size_t)t * width + k];
                    sum += term; magnitude += fabs(term);
                }
                CHECK(fabs(y[(size_t)t * output + o] - sum) < 0.000001 * fmax(magnitude, 1));
            }
        }
        CHECK(!ds4_gpu_dsv41_projection_rows(out, model, weight_bytes - 1, 0, width, output, 1, in));
        CHECK(!ds4_gpu_dsv41_projection_rows(out, model, weight_bytes, 0, width, output, MAX_ROWS + 1, in));
        ds4_gpu_tensor_free(ref); ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(in);
        ds4_gpu_cleanup(); free(model);
        CHECK(ds4_gpu_init());
    }
    return 1;
}

typedef struct { float value; uint32_t id; } topk_entry;

static int compare_topk_entry(const void *a, const void *b) {
    const topk_entry *x = a, *y = b;
    if (x->value > y->value) return -1;
    if (x->value < y->value) return 1;
    return (x->id > y->id) - (x->id < y->id);
}

static int check_general_topk(void) {
    const uint32_t widths[] = {1, 2, 31, 127, 255, 256, 257, 511, 512, 513,
        1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097, 8191, 8192};
    for (size_t wi = 0; wi < sizeof(widths) / sizeof(*widths); wi++) {
        const uint32_t width = widths[wi], rows = 3;
        ds4_gpu_tensor *scores = upload(NULL, (size_t)width * rows * 4);
        ds4_gpu_tensor *selected = upload(NULL, ((size_t)width * rows + 1) * 4);
        topk_entry *reference = malloc((size_t)width * rows * sizeof(*reference));
        bool *seen = calloc(width, sizeof(*seen));
        CHECK(scores && selected && reference && seen);
        float *s = ds4_gpu_tensor_contents(scores);
        uint32_t *ids = ds4_gpu_tensor_contents(selected);
        const uint32_t counts[] = {1, 2, 31, 128, 511, 512, 513, 2048, width};
        for (uint32_t pattern = 0; pattern < 4; pattern++) {
            for (uint32_t t = 0; t < rows; t++) {
                for (uint32_t i = 0; i < width; i++) {
                    const float value = pattern == 0 ? random_value() :
                        pattern == 1 ? (float)(i % 7) :
                        pattern == 2 ? -INFINITY : (i & 1 ? -0.0f : 0.0f);
                    const float v = pattern == 1 && i % 17 == 0 ? INFINITY : value;
                    s[t * width + i] = v;
                    reference[t * width + i] = (topk_entry){v, i};
                }
                qsort(reference + t * width, width, sizeof(*reference), compare_topk_entry);
            }
            for (size_t ki = 0; ki < sizeof(counts) / sizeof(*counts); ki++) {
                const uint32_t k = counts[ki];
                if (k > width) continue;
                ids[rows * k] = 0xabcdef01;
                CHECK(ds4_gpu_indexer_topk_tensor(selected, scores, width, rows, k));
                CHECK(ds4_gpu_synchronize());
                CHECK(ids[rows * k] == 0xabcdef01);
                for (uint32_t t = 0; t < rows; t++) {
                    memset(seen, 0, width * sizeof(*seen));
                    for (uint32_t i = 0; i < k; i++) {
                        const uint32_t id = ids[t * k + i];
                        CHECK(id < width && !seen[id]);
                        seen[id] = true;
#ifdef __APPLE__
                        /* Metal's bitonic order may permute exactly tied keys. */
                        CHECK(s[t * width + id] == reference[t * width + i].value);
#else
                        if (ids[t * k + i] != reference[t * width + i].id) {
                            fprintf(stderr, "top-k width=%u k=%u pattern=%u row=%u rank=%u: %u vs %u\n",
                                width, k, pattern, t, i, ids[t * k + i], reference[t * width + i].id);
                            CHECK(0);
                        }
#endif
                    }
                }
            }
        }
        fprintf(stderr, "V4.1 variable-k sort width=%u: independent sorted-value/unique-ID oracle PASS\n", width);
        ds4_gpu_tensor_free(selected); ds4_gpu_tensor_free(scores); free(reference); free(seen);
    }
    return 1;
}

static int check_causal_topk(void) {
    const uint32_t frontiers[] = {1024, 1025, 1535, 2047, 2048, 4095, 4096, 8191, 8192, 16383, 32767, 65535};
    const uint32_t counts[] = {1, 2, 31, 32, 33, 127, 128, 129};
    for (uint32_t ratio = 1; ratio <= 2; ratio++) {
    for (size_t fi = 0; fi < sizeof(frontiers) / sizeof(*frontiers); fi++) {
        const uint32_t start = frontiers[fi] * ratio - 1u;
        const uint32_t width = (start + 129u) / ratio + 1024u;
        ds4_gpu_tensor *scores = upload(NULL, (size_t)width * 129u * sizeof(float));
        ds4_gpu_tensor *selected = upload(NULL, (129u * 512u + 1u) * sizeof(int32_t));
        ds4_gpu_tensor *reference = upload(NULL, 129u * 512u * sizeof(int32_t));
        CHECK(scores && selected && reference);
        float *s = ds4_gpu_tensor_contents(scores);
        int32_t *ids = ds4_gpu_tensor_contents(selected);
        const int32_t *ref = ds4_gpu_tensor_contents(reference);
        for (uint32_t pattern = 0; pattern < 3; pattern++) {
            for (uint32_t t = 0; t < 129u; t++) {
                const uint32_t visible = (start + t + 1u) / ratio;
                for (uint32_t j = 0; j < width; j++) {
                    /* Future entries are deliberately attractive. Causality
                     * must come from each row's bounds, not score contents. */
                    float v = pattern == 0 ? random_value() : pattern == 1 ? (float)(j % 7u) : -INFINITY;
                    s[(size_t)t * width + j] = j < visible ? v : 12345.0f;
                }
            }
            for (size_t ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
                const uint32_t rows = counts[ci];
                CHECK(ds4_gpu_begin_commands());
                for (uint32_t t = 0; t < rows; t++) {
                    const uint32_t visible = (start + t + 1u) / ratio;
                    ds4_gpu_tensor *sr = ds4_gpu_tensor_view(scores, (size_t)t * width * 4u, visible * 4u);
                    ds4_gpu_tensor *out = ds4_gpu_tensor_view(reference, (size_t)t * 512u * 4u, 512u * 4u);
                    CHECK(sr && out && ds4_gpu_indexer_topk_tensor(out, sr, visible, 1, 512));
                    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(sr);
                }
                CHECK(ds4_gpu_end_commands());
                ids[rows * 512u] = 123456;
                CHECK(ds4_gpu_dsv41_indexer_topk_batch(selected, scores, width, rows, start, ratio));
                CHECK(ds4_gpu_synchronize());
                CHECK(ids[rows * 512u] == 123456);
                for (uint32_t t = 0; t < rows; t++) {
                    CHECK(s[(size_t)t * width + width - 1u] == 12345.0f);
                    for (uint32_t k = 0; k < 512u; k++) {
                        if (ids[t * 512u + k] != ref[t * 512u + k]) {
                            fprintf(stderr, "top-k mismatch ratio=%u visible=%u rows=%u pattern=%u row=%u rank=%u: %d vs %d\n",
                                ratio, frontiers[fi], rows, pattern, t, k, ids[t * 512u + k], ref[t * 512u + k]);
                            CHECK(0);
                        }
                        CHECK(ids[t * 512u + k] >= 0 && (uint32_t)ids[t * 512u + k] < (start + t + 1u) / ratio);
                    }
                }
            }
        }
        CHECK(!ds4_gpu_dsv41_indexer_topk_batch(selected, scores, width, 130, start, ratio));
        CHECK(!ds4_gpu_dsv41_indexer_topk_batch(selected, scores, width, 1, UINT32_MAX, ratio));
        CHECK(!ds4_gpu_dsv41_indexer_topk_batch(selected, scores, width, 1, 0, ratio));
        CHECK(!ds4_gpu_dsv41_indexer_topk_batch(selected, scores, width, 1, start, 0));
        fprintf(stderr, "V4.1 causal top-k ratio=%u visible=%u: exact IDs including ties PASS\n", ratio, frontiers[fi]);
        ds4_gpu_tensor_free(reference); ds4_gpu_tensor_free(selected); ds4_gpu_tensor_free(scores);
    }
    }
    return 1;
}


static int check_compact_carry(void) {
    const uint32_t widths[] = {1, 31, 32, 33, 127, 128, 129, 20480};
    const uint32_t rows = 129, offset = 2;
    for (uint32_t format = 0; format <= DS4_V41_CARRY_MASK; format++) {
        for (size_t wi = 0; wi < sizeof(widths) / sizeof(*widths); wi++) {
            const uint32_t width = widths[wi];
            const uint32_t words = format == DS4_V41_CARRY_BF16 ? (width + 1) / 2 : (width + 31) / 32;
            const size_t count = (size_t)width * rows;
            const size_t packed_bytes = (size_t)(rows + 4) * words * sizeof(uint32_t);
            ds4_gpu_tensor *plain = upload(NULL, (count + 1) * sizeof(float));
            ds4_gpu_tensor *packed = upload(NULL, packed_bytes);
            uint32_t *expected = malloc(count * sizeof(uint32_t));
            CHECK(plain && packed && expected);
            uint32_t *bits = ds4_gpu_tensor_contents(plain);
            unsigned char *storage = ds4_gpu_tensor_contents(packed);
            for (size_t i = 0; i < count; i++) {
                const uint32_t b = (uint32_t)(i * 40503u + 32768u) & 0xffffu;
                expected[i] = format == DS4_V41_CARRY_BF16 ? b << 16 : (b & 1u ? 0xff800000u : 0u);
            }
            memcpy(bits, expected, count * sizeof(uint32_t));
            bits[count] = 0x12345678u;
            memset(storage, 0xa5, packed_bytes);
            CHECK(ds4_gpu_dsv41_carry_copy(packed, offset, plain, width, rows, format, true));
            CHECK(ds4_gpu_synchronize());
            CHECK(!memcmp(bits, expected, count * sizeof(uint32_t)));
            memset(bits, 0, count * sizeof(uint32_t));
            CHECK(ds4_gpu_dsv41_carry_copy(packed, offset, plain, width, rows, format, false));
            CHECK(ds4_gpu_synchronize());
            CHECK(!memcmp(bits, expected, count * sizeof(uint32_t)) && bits[count] == 0x12345678u);
            for (size_t i = 0; i < (size_t)offset * words * 4u; i++) CHECK(storage[i] == 0xa5);
            for (size_t i = (size_t)(offset + rows) * words * 4u; i < packed_bytes; i++)
                CHECK(storage[i] == 0xa5);
            CHECK(!ds4_gpu_dsv41_carry_copy(packed, UINT32_MAX, plain, width, rows, format, true));
            CHECK(!ds4_gpu_dsv41_carry_copy(packed, offset + 3, plain, width, rows, format, true));
            CHECK(!ds4_gpu_dsv41_carry_copy(packed, offset, plain, 0, rows, format, true));
            CHECK(!ds4_gpu_dsv41_carry_copy(packed, offset, plain, width, 0, format, true));
            CHECK(!ds4_gpu_dsv41_carry_copy(packed, offset, plain, width, rows, DS4_V41_CARRY_F32, true));
            fprintf(stderr, "V4.1 compact carry format=%u width=%u: exact round-trip and bounds PASS\n", format, width);
            free(expected);
            ds4_gpu_tensor_free(plain);
            ds4_gpu_tensor_free(packed);
        }
    }
    return 1;
}

/* Exercise the full-head and compact TP layouts with the same causal keys.
 * Selected rows are shuffled; include masked future rows at odd frontiers. */
static int check_tp_attention(void) {
    enum { D = 512, H = 64, K = 512, C = 2048 };
    const struct { uint32_t rows, selected; } cases[] = {
        {1, K}, {31, K}, {32, K}, {33, K}, {129, K}, {257, K}, {2048, K},
        {1, 0}, {1, 1}, {2, K}, {6, K}, {8, K},
        {2, 0}, {2, 1}, {6, 0}
    };
    float *sinks = NULL;
    CHECK(posix_memalign((void **)&sinks, getpagesize(), getpagesize()) == 0);
    for (int h = 0; h < H; h++) sinks[h] = random_value();
    CHECK(ds4_gpu_set_model_map(sinks, getpagesize()));
    ds4_gpu_set_quality(false);
    for (unsigned s = 0; s < sizeof(cases) / sizeof(*cases); s++) {
        const uint32_t n = cases[s].rows, nr = n + 127, start = 3073;
        const uint32_t ratio = 1u + s % 2u;
        const size_t nq = (size_t)n * H * D;
        float *q = malloc(nq * 4), *actual = malloc(nq * 4);
        float *compact = malloc(nq * 2), *part = malloc(nq * 2);
        float *raw = malloc((size_t)nr * D * 4), *comp = malloc(C * D * 4);
        int32_t *ids = malloc((size_t)n * K * 4);
        CHECK(q && actual && compact && part && raw && comp && ids);
        for (size_t i = 0; i < nq; i++) q[i] = bf16(random_value() / 4);
        for (uint32_t i = 0; i < nr * D; i++) raw[i] = bf16(random_value() / 4);
        for (uint32_t i = 0; i < C * D; i++) comp[i] = bf16(random_value() / 4);
        for (uint32_t t = 0; t < n; t++)
            for (uint32_t j = 0; j < K; j++)
                ids[t * K + j] = j < cases[s].selected &&
                    (cases[s].selected != K || j % 29)
                    ? (int32_t)((j * 127u + t * 17u) % C) : -1;
        ds4_gpu_tensor *qt = upload(q, nq * 4), *rt = upload(raw, (size_t)nr * D * 4);
        ds4_gpu_tensor *ct = upload(comp, C * D * 4), *it = upload(ids, (size_t)n * K * 4);
        ds4_gpu_tensor *out = upload(NULL, nq * 4), *qp = upload(NULL, nq * 2);
        ds4_gpu_tensor *op = upload(NULL, nq * 2);
        CHECK(qt && rt && ct && it && out && qp && op);
        CHECK(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(out, sinks, getpagesize(),
            0, qt, rt, ct, 0, it, n, start, nr, nr, 0, C, K, 128, ratio, H, D));
        CHECK(ds4_gpu_tensor_read(out, 0, actual, nq * 4));
        double max_split = 0, max_oracle = 0;
        for (uint32_t rank = 0; rank < 2; rank++) {
            for (uint32_t t = 0; t < n; t++)
                memcpy(compact + (size_t)t * H/2 * D,
                    q + ((size_t)t * H + rank * H/2) * D, H/2 * D * 4);
            CHECK(ds4_gpu_tensor_write(qp, 0, compact, nq * 2));
            CHECK(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(op, sinks, getpagesize(),
                rank * H/2 * 4, qp, rt, ct, 0, it, n, start, nr, nr, 0, C, K, 128, ratio, H/2, D));
            CHECK(ds4_gpu_tensor_read(op, 0, part, nq * 2));
#ifdef __APPLE__
            if (n > 1) {
                ds4_gpu_set_quality(true); /* Existing eight-head kernel. */
                CHECK(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(op, sinks, getpagesize(),
                    rank * H/2 * 4, qp, rt, ct, 0, it, n, start, nr, nr, 0, C, K, 128, ratio, H/2, D));
                CHECK(!memcmp(part, ds4_gpu_tensor_contents(op), nq * 2));
                ds4_gpu_set_quality(false);
            }
#endif
            if (n == 2048 && rank == 0) {
                const bool controls[] = {true, false, false, true};
                for (unsigned pass = 0; pass < 4; pass++) {
                    ds4_gpu_set_quality(controls[pass]);
                    const double begin = monotonic_seconds();
                    for (int repeat = 0; repeat < 4; repeat++)
                        CHECK(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(op,
                            sinks, getpagesize(), 0, qp, rt, ct, 0, it, n, start,
                            nr, nr, 0, C, K, 128, ratio, H/2, D));
                    CHECK(ds4_gpu_synchronize());
                    fprintf(stderr, "TP indexed attention control=%u %.3f ms\n",
                        controls[pass], (monotonic_seconds() - begin) * 250);
                }
                ds4_gpu_set_quality(false);
            }
            for (uint32_t t = 0; t < n; t++) {
                for (uint32_t h = 0; h < H/2; h++) {
                    const float *got = part + ((size_t)t * H/2 + h) * D;
                    const float *ref = actual + ((size_t)t * H + rank * H/2 + h) * D;
                    for (int d = 0; d < D; d++) {
                        const double err = fabs((double)got[d] - ref[d]);
                        max_split = fmax(max_split, err);
                        CHECK(isfinite(got[d]) && err < 3e-5 * (1 + fabs(ref[d])));
                    }
                    if ((t != 0 && t != n/2 && t + 1 != n) || h % 15) continue;
                    const float *query = compact + ((size_t)t * H/2 + h) * D;
                    double logits[128 + K + 1], values[128 + K + 1];
                    const uint32_t col = (t * 71u + h * 19u) % D;
                    unsigned count = 0;
                    for (uint32_t j = 0; j < 128 + K; j++) {
                        const int32_t id = j < 128 ? (int32_t)(t + j) : ids[t * K + j - 128];
                        if (j >= 128 && (uint32_t)id >= (start + t + 1) / ratio) continue;
                        const float *key = (j < 128 ? raw : comp) + (size_t)id * D;
                        double dot = 0;
                        for (int d = 0; d < D; d++) dot += (double)query[d] * key[d];
                        logits[count] = dot / sqrt(512.0);
                        values[count++] = key[col];
                    }
                    logits[count] = sinks[rank * H/2 + h]; values[count++] = 0;
                    double max = -INFINITY, den = 0, num = 0;
                    for (unsigned j = 0; j < count; j++) max = fmax(max, logits[j]);
                    for (unsigned j = 0; j < count; j++) {
                        const double p = exp(logits[j] - max);
                        den += p; num += p * values[j];
                    }
                    const double err = fabs(got[col] - num / den);
                    max_oracle = fmax(max_oracle, err);
                    CHECK(err < 3e-5 * (1 + fabs(num / den)));
                }
            }
        }
        fprintf(stderr, "V4.1 TP attention rows=%u selected=%u ratio=%u max split=%g oracle=%g: PASS\n",
            n, cases[s].selected, ratio, max_split, max_oracle);
        ds4_gpu_tensor_free(qt); ds4_gpu_tensor_free(rt); ds4_gpu_tensor_free(ct);
        ds4_gpu_tensor_free(it); ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(qp);
        ds4_gpu_tensor_free(op);
        free(q); free(actual); free(compact); free(part); free(raw); free(comp); free(ids);
    }
    ds4_gpu_cleanup();
    free(sinks);
    return 1;
}

#ifdef __APPLE__
/* ---- The V4.1 Metal fast-path groups ----
 * Each check runs a group's path and its off switch (DwarfStar's code) on the same data: equal bits, and the
 * kernel counts show which path ran. Appended after the checks above so they keep their random data. */

/* COMPUTE_TENSOR_COPY: an aligned copy inside a command batch runs as a compute kernel; a misaligned one, or any
 * copy with the switch off, as a blit. */
static int check_compute_copy(void) {
    enum { N = 4096 };
    float *a = malloc(N * sizeof(float)), *b = malloc(N * sizeof(float)), *want = calloc(N, sizeof(float));
    CHECK(a && b && want);
    for (int i = 0; i < N; i++) a[i] = random_value();
    for (int i = 1; i < 4; i++) want[i] = a[i + 1];
    for (int i = 4; i < N - 4; i++) want[i] = a[i + 4];
    ds4_gpu_tensor *src = upload(a, N * sizeof(float)), *dst = upload(NULL, N * sizeof(float));
    CHECK(src && dst);
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    for (int off = 0; off < 2; off++) {
        ds4_gpu_test_set_v41_off(off ? DS4_GPU_V41_COMPUTE_TENSOR_COPY : 0);
        const uint64_t runs = ds4_gpu_test_kernel_runs("kernel_ds4_copy_u4");
        memset(b, 0, N * sizeof(float));
        CHECK(ds4_gpu_tensor_write(dst, 0, b, N * sizeof(float)));
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_tensor_copy(dst, 16, src, 32, (N - 8) * sizeof(float)));
        CHECK(ds4_gpu_tensor_copy(dst, 4, src, 8, 3 * sizeof(float)));
        CHECK(ds4_gpu_end_commands());
        CHECK(ds4_gpu_test_kernel_runs("kernel_ds4_copy_u4") == runs + (off ? 0u : 1u));
        CHECK(ds4_gpu_tensor_read(dst, 0, b, N * sizeof(float)));
        CHECK(!memcmp(b, want, N * sizeof(float)));
    }
    ds4_gpu_test_set_v41_off(0);
    ds4_gpu_test_set_flags(0);
    ds4_gpu_tensor_free(src); ds4_gpu_tensor_free(dst);
    free(a); free(b); free(want);
    fprintf(stderr, "V4.1 compute-encoder tensor copy: exact, kernel and blit\n");
    return 1;
}

/* ROUTER_SELECT: one row's fused select (the hierarchical 384 -> 6 kernel) and the per-row kernel of a few rows,
 * against DwarfStar's router select: probabilities, experts and weights bit for bit, on random, tied and patterned
 * logits. */
static int check_router_select(void) {
    enum { N = 384, K = 6 };
    float *bias = NULL;
    CHECK(posix_memalign((void **)&bias, getpagesize(), getpagesize()) == 0);
    ds4_gpu_tensor *x = upload(NULL, 2 * N * 4), *p[3], *ids[3], *w[3];
    CHECK(x);
    for (int a = 0; a < 3; a++) {
        p[a] = upload(NULL, 2 * N * 4); ids[a] = upload(NULL, 2 * K * 4); w[a] = upload(NULL, 2 * K * 4);
        CHECK(p[a] && ids[a] && w[a]);
    }
    float *lg = ds4_gpu_tensor_contents(x);
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    for (uint32_t draw = 0; draw < 32; draw++) {
        for (int i = 0; i < N; i++) {
            lg[i] = draw % 11 == 0 ? 1.0f : draw % 13 == 0 ? (float)(i % 6) : random_value() * 4;
            bias[i] = draw % 11 == 0 || draw % 13 == 0 ? 0.0f : random_value() / 40.0f;
            lg[N + i] = lg[i];
        }
        CHECK(ds4_gpu_set_model_map(bias, getpagesize()));
        const uint64_t runs = ds4_gpu_test_kernel_runs("kernel_dsv41_router_hier");
        CHECK(ds4_gpu_begin_commands() &&
              ds4_gpu_router_select_tensor(ids[0], w[0], p[0], bias, getpagesize(), 0, 0, 0, 0,
                  N, K, 1.5f, 0, 0, true, false, x) &&
              ds4_gpu_dsv41_router_one(ids[1], w[1], p[1], x, bias, getpagesize(), 0, N, K, 1.5f) &&
              ds4_gpu_dsv41_router_rows(ids[2], w[2], p[2], x, bias, getpagesize(), 0, N, K, 1.5f, 2) &&
              ds4_gpu_end_commands());
        CHECK(ds4_gpu_test_kernel_runs("kernel_dsv41_router_hier") == runs + 1);
        for (int a = 1; a < 3; a++) {
            CHECK(!memcmp(ds4_gpu_tensor_contents(p[0]), ds4_gpu_tensor_contents(p[a]), N * 4));
            CHECK(!memcmp(ds4_gpu_tensor_contents(ids[0]), ds4_gpu_tensor_contents(ids[a]), K * 4));
            CHECK(!memcmp(ds4_gpu_tensor_contents(w[0]), ds4_gpu_tensor_contents(w[a]), K * 4));
        }
        CHECK(!memcmp(ds4_gpu_tensor_contents(p[2]), (float *)ds4_gpu_tensor_contents(p[2]) + N, N * 4));
        CHECK(!memcmp(ds4_gpu_tensor_contents(ids[2]), (int32_t *)ds4_gpu_tensor_contents(ids[2]) + K, K * 4));
        CHECK(!memcmp(ds4_gpu_tensor_contents(w[2]), (float *)ds4_gpu_tensor_contents(w[2]) + K, K * 4));
    }
    ds4_gpu_test_set_flags(0);
    for (int a = 0; a < 3; a++) { ds4_gpu_tensor_free(p[a]); ds4_gpu_tensor_free(ids[a]); ds4_gpu_tensor_free(w[a]); }
    ds4_gpu_tensor_free(x);
    /* bias stays: the model views may wrap it */
    fprintf(stderr, "V4.1 fused router select == router select over 32 draws, one row and two\n");
    return 1;
}

static int same_bits(ds4_gpu_tensor *a, ds4_gpu_tensor *b, uint64_t n) {
    return !memcmp(ds4_gpu_tensor_contents(a), ds4_gpu_tensor_contents(b), n * sizeof(float));
}

/* FUSED_ROUNDING: each producer that writes bf16 directly against the producer followed by the rounding pass,
 * bit for bit; the counts show the rounding kernels ran. */
static int check_fused_rounding(void) {
    enum { IN = 4096, Q8_OUT = 1024, F16_OUT = 512, HC = 4, HEADS = 8, HEAD_DIM = 512 };
    typedef struct { uint16_t d; int8_t qs[32]; } q8_block;
    const uint64_t q8_bytes = (uint64_t)Q8_OUT * IN / 32 * sizeof(q8_block), f16_bytes = (uint64_t)F16_OUT * IN * 2;
    const uint64_t bytes = q8_bytes + f16_bytes + IN * sizeof(float);
    void *model = NULL;
    CHECK(posix_memalign(&model, getpagesize(), bytes) == 0);
    q8_block *q8 = model;
    for (uint64_t i = 0; i < q8_bytes / sizeof(q8_block); i++) {
        q8[i].d = (uint16_t)(0x2000u | (seed & 0x3ffu));
        for (int j = 0; j < 32; j++) q8[i].qs[j] = (int8_t)(random_value() * 32);
    }
    uint16_t *f16 = (uint16_t *)((char *)model + q8_bytes);
    for (uint64_t i = 0; i < (uint64_t)F16_OUT * IN; i++) f16[i] = (uint16_t)(0x2c00u + (seed % 0x1000u) + (random_value() < 0 ? 0x8000u : 0));
    float *norm_w = (float *)((char *)model + q8_bytes + f16_bytes);
    for (int i = 0; i < IN; i++) norm_w[i] = random_value();
    CHECK(ds4_gpu_set_model_map(model, bytes));
    float x[HC * IN], split[24];
    for (int i = 0; i < HC * IN; i++) x[i] = random_value();
    for (int i = 0; i < 24; i++) split[i] = random_value() / 4;
    ds4_gpu_tensor *in = upload(x, sizeof(x)), *in2 = upload(x, sizeof(x)), *w = upload(split, sizeof(split));
    ds4_gpu_tensor *a = upload(NULL, sizeof(x)), *b = upload(NULL, sizeof(x));
    CHECK(in && in2 && w && a && b);
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    const uint64_t q8r = ds4_gpu_test_kernel_runs("kernel_mul_mv_q8_0_f32_bf16");
    const uint64_t q8io = ds4_gpu_test_kernel_runs("kernel_mul_mv_q8_0_f32_bf16io");
    const uint64_t f16r = ds4_gpu_test_kernel_runs("kernel_mul_mv_f16_f32_4_bf16");
    /* Q8 matvec, outputs rounded */
    CHECK(ds4_gpu_matmul_q8_0_tensor(a, model, bytes, 0, IN, Q8_OUT, in, 1) &&
          ds4_gpu_dsv41_quantize(a, Q8_OUT, 1, DS4_V41_BF16) &&
          ds4_gpu_matmul_q8_0_bf16_tensor(b, model, bytes, 0, IN, Q8_OUT, in));
    CHECK(same_bits(a, b, Q8_OUT));
    /* Q8 matvec, activation and outputs rounded; the activation itself stays as it was */
    CHECK(ds4_gpu_matmul_q8_0_bf16io_tensor(b, model, bytes, 0, IN, Q8_OUT, in2) && same_bits(in, in2, IN));
    CHECK(ds4_gpu_dsv41_quantize(in2, IN, 1, DS4_V41_BF16) &&
          ds4_gpu_matmul_q8_0_tensor(a, model, bytes, 0, IN, Q8_OUT, in2, 1) &&
          ds4_gpu_dsv41_quantize(a, Q8_OUT, 1, DS4_V41_BF16));
    CHECK(same_bits(a, b, Q8_OUT));
    CHECK(ds4_gpu_tensor_write(in2, 0, x, sizeof(x)));
    /* F16 matvec */
    CHECK(ds4_gpu_matmul_f16_tensor(a, model, bytes, q8_bytes, IN, F16_OUT, in, 1) &&
          ds4_gpu_dsv41_quantize(a, F16_OUT, 1, DS4_V41_BF16) &&
          ds4_gpu_dsv41_project_f16_bf16(b, model, bytes, q8_bytes, IN, F16_OUT, in));
    CHECK(same_bits(a, b, F16_OUT));
    CHECK(ds4_gpu_test_kernel_runs("kernel_mul_mv_q8_0_f32_bf16") == q8r + 1 &&
          ds4_gpu_test_kernel_runs("kernel_mul_mv_q8_0_f32_bf16io") == q8io + 1 &&
          ds4_gpu_test_kernel_runs("kernel_mul_mv_f16_f32_4_bf16") == f16r + 1);
    /* RMS norm with its weight */
    CHECK(ds4_gpu_rms_norm_weight_tensor(a, in, model, bytes, q8_bytes + f16_bytes, IN, 1e-6f) &&
          ds4_gpu_dsv41_quantize(a, IN, 1, DS4_V41_BF16) &&
          ds4_gpu_rms_norm_weight_bf16_tensor(b, in, model, bytes, q8_bytes + f16_bytes, IN, 1e-6f));
    CHECK(same_bits(a, b, IN));
    /* HC weighted sums: 4 pre weights, and the pre weights inside a 24-wide split row */
    ds4_gpu_tensor *a1 = ds4_gpu_tensor_view(a, 0, IN * sizeof(float)), *b1 = ds4_gpu_tensor_view(b, 0, IN * sizeof(float));
    CHECK(a1 && b1);
    CHECK(ds4_gpu_hc_weighted_sum_tensor(a1, in, w, IN, HC) && ds4_gpu_dsv41_quantize(a1, IN, 1, DS4_V41_BF16) &&
          ds4_gpu_hc_weighted_sum_bf16_tensor(b1, in, w, IN, HC));
    CHECK(same_bits(a1, b1, IN));
    CHECK(ds4_gpu_hc_weighted_sum_split_tensor(a1, in, w, IN, HC) && ds4_gpu_dsv41_quantize(a1, IN, 1, DS4_V41_BF16) &&
          ds4_gpu_hc_weighted_sum_split_bf16_tensor(b1, in, w, IN, HC));
    CHECK(same_bits(a1, b1, IN));
    /* HC expand of a block into the four streams */
    CHECK(ds4_gpu_hc_expand_split_tensor(a, in2, in, w, IN, HC) && ds4_gpu_dsv41_quantize(a, IN * HC, 1, DS4_V41_BF16) &&
          ds4_gpu_hc_expand_split_bf16_tensor(b, in2, in, w, IN, HC));
    CHECK(same_bits(a, b, HC * IN));
    /* the attention heads' bf16 rounding and inverse RoPE, both RoPE kinds */
    for (int compressed = 0; compressed < 2; compressed++) {
        CHECK(ds4_gpu_tensor_write(a, 0, x, sizeof(x)) && ds4_gpu_tensor_write(b, 0, x, sizeof(x)));
        CHECK(ds4_gpu_dsv41_quantize(a, HEADS * HEAD_DIM, 1, DS4_V41_BF16) &&
              ds4_gpu_dsv41_rope(a, HEAD_DIM, HEADS, 1, 4321, compressed, true) &&
              ds4_gpu_dsv41_rope_bf16(b, HEAD_DIM, HEADS, 1, 4321, compressed, true));
        CHECK(same_bits(a, b, HEADS * HEAD_DIM));
    }
    ds4_gpu_test_set_flags(0);
    ds4_gpu_tensor_free(a1); ds4_gpu_tensor_free(b1);
    ds4_gpu_tensor_free(in); ds4_gpu_tensor_free(in2); ds4_gpu_tensor_free(w);
    ds4_gpu_tensor_free(a); ds4_gpu_tensor_free(b);
    /* model stays: the model views may wrap it */
    fprintf(stderr, "V4.1 fused bf16 rounding == separate rounding: Q8, Q8 in/out, F16, norm, HC sums, expand, RoPE\n");
    return 1;
}

/* NARROW_WEIGHTS: the narrow readers against the wide matvecs they replace, bit for bit: the BF16 router copy against
 * the F32 router (gates whose low halves are zero), and the E4M3 Engram and indexer q_b copies against their F16
 * projections rounded to bf16 (weights that are E4M3 codes at their block's scale, exact in F16). */
static int check_narrow_weights(void) {
    enum { R_IN = 4096, R_OUT = 384, X_MAX = 6144 };
    const uint32_t ins[2] = {6144, 1280}, outs[2] = {256, 4096};   /* Engram, indexer q_b */
    const size_t page = (size_t)getpagesize(), r_bytes = (size_t)R_IN * R_OUT * 4;
    size_t off[2], bytes = (r_bytes + page - 1) / page * page;
    for (int k = 0; k < 2; k++) { off[k] = bytes; bytes += ((size_t)ins[k] * outs[k] * 2 + page - 1) / page * page; }
    unsigned char *model = NULL;
    CHECK(posix_memalign((void **)&model, page, bytes) == 0);
    float *gates = (float *)model;
    uint16_t *gates16 = malloc((size_t)R_IN * R_OUT * 2);
    CHECK(gates16);
    for (size_t i = 0; i < (size_t)R_IN * R_OUT; i++) {
        const float v = random_value() / 64;
        uint32_t bits;
        memcpy(&bits, &v, 4);
        bits &= 0xffff0000u;
        memcpy(&gates[i], &bits, 4);
        gates16[i] = (uint16_t)(bits >> 16);
    }
    ds4_gpu_tensor *codes[2], *scales[2];
    for (int k = 0; k < 2; k++) {
        const uint32_t in = ins[k], out = outs[k];
        unsigned char *c = malloc((size_t)in * out), *e = malloc((size_t)(in / 32) * (out / 32));
        CHECK(c && e);
        for (size_t i = 0; i < (size_t)(in / 32) * (out / 32); i++) { (void)random_value(); e[i] = (unsigned char)(112u + seed % 8u); }
        _Float16 *w16 = (_Float16 *)(model + off[k]);
        for (size_t i = 0; i < (size_t)in * out; i++) {
            (void)random_value();
            unsigned char b = (unsigned char)(seed >> 3);
            if ((b & 0x7f) == 0x7f) b ^= 1;   /* no NaN codes */
            c[i] = b;
            const unsigned char a = b & 0x7f;
            const uint32_t ex = (a >> 3) & 15u, mm = a & 7u;
            float v = a == 0 ? 0.0f : ex == 0 ? ldexpf((float)mm, -9) : ldexpf(1.0f + mm / 8.0f, (int)ex - 7);
            if (b & 0x80) v = -v;
            w16[i] = (_Float16)(v * ldexpf(1.0f, (int)e[(i / in / 32) * (in / 32) + i % in / 32] - 127));
        }
        codes[k] = upload(c, (size_t)in * out); scales[k] = upload(e, (size_t)(in / 32) * (out / 32));
        CHECK(codes[k] && scales[k]);
        free(c); free(e);
    }
    CHECK(ds4_gpu_set_model_map(model, bytes));
    float x[X_MAX];
    for (int i = 0; i < X_MAX; i++) x[i] = random_value() / 4;
    ds4_gpu_tensor *xt = upload(x, sizeof(x)), *rt = upload(gates16, (size_t)R_IN * R_OUT * 2);
    ds4_gpu_tensor *a = upload(NULL, 4096 * 4), *b = upload(NULL, 4096 * 4);
    CHECK(xt && rt && a && b);
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    const char *names[3] = {"kernel_router_blend_r1", "kernel_engram_blend_r2_bf16", "kernel_indexer_blend_r2"};
    uint64_t runs[3];
    for (int k = 0; k < 3; k++) runs[k] = ds4_gpu_test_kernel_runs(names[k]);
    CHECK(ds4_gpu_matmul_f32_tensor(a, model, bytes, 0, R_IN, R_OUT, xt, 1) &&
          ds4_gpu_matmul_bf16_router_weights(b, rt, R_IN, R_OUT, xt));
    CHECK(same_bits(a, b, R_OUT));
    for (int k = 0; k < 2; k++) {
        CHECK(ds4_gpu_matmul_f16_tensor(a, model, bytes, off[k], ins[k], outs[k], xt, 1) &&
              ds4_gpu_dsv41_quantize(a, outs[k], 1, DS4_V41_BF16));
        CHECK(k == 0 ? ds4_gpu_matmul_engram_fp8_weights(b, codes[k], scales[k], ins[k], outs[k], xt) :
                       ds4_gpu_dsv41_project_fp8_bf16_weights(b, codes[k], scales[k], ins[k], outs[k], xt));
        CHECK(same_bits(a, b, outs[k]));
    }
    for (int k = 0; k < 3; k++) CHECK(ds4_gpu_test_kernel_runs(names[k]) == runs[k] + 1);
    ds4_gpu_test_set_flags(0);
    for (int k = 0; k < 2; k++) { ds4_gpu_tensor_free(codes[k]); ds4_gpu_tensor_free(scales[k]); }
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(rt); ds4_gpu_tensor_free(a); ds4_gpu_tensor_free(b);
    free(gates16);
    /* model stays: the model views may wrap it */
    fprintf(stderr, "V4.1 narrow readers == wide matvecs: BF16 router, E4M3 Engram and indexer q_b\n");
    return 1;
}

/* QKV_FUSION: the q_a/kv pair against two bf16 matvecs, the paired norm against two bf16 norms (one row and three),
 * and the q projection with RoPE against the bf16 matvec followed by the RoPE kernel (both RoPE kinds), bit for bit. */
static int check_qkv_fusion(void) {
    enum { IN = 4096, OUT_A = 1024, OUT_B = 512, Q_IN = 1024, Q_OUT = 2048, ROWS = 3 };
    typedef struct { uint16_t d; int8_t qs[32]; } q8_block;
    const uint64_t a_bytes = (uint64_t)OUT_A * IN / 32 * sizeof(q8_block), b_bytes = (uint64_t)OUT_B * IN / 32 * sizeof(q8_block);
    const uint64_t q_bytes = (uint64_t)Q_OUT * Q_IN / 32 * sizeof(q8_block);
    const uint64_t n_off = a_bytes + b_bytes + q_bytes, bytes = n_off + (OUT_A + OUT_B) * sizeof(float);
    void *model = NULL;
    CHECK(posix_memalign(&model, getpagesize(), bytes) == 0);
    q8_block *q8 = model;
    for (uint64_t i = 0; i < n_off / sizeof(q8_block); i++) {
        q8[i].d = (uint16_t)(0x2000u | (seed & 0x3ffu));
        for (int j = 0; j < 32; j++) q8[i].qs[j] = (int8_t)(random_value() * 32);
    }
    float *norm_w = (float *)((char *)model + n_off);
    for (int i = 0; i < OUT_A + OUT_B; i++) norm_w[i] = random_value();
    CHECK(ds4_gpu_set_model_map(model, bytes));
    float x[ROWS * IN];
    for (int i = 0; i < ROWS * IN; i++) x[i] = random_value();
    ds4_gpu_tensor *in = upload(x, sizeof(x)), *a0 = upload(NULL, sizeof(x)), *b0 = upload(NULL, sizeof(x));
    ds4_gpu_tensor *a1 = upload(NULL, sizeof(x)), *b1 = upload(NULL, sizeof(x));
    CHECK(in && a0 && b0 && a1 && b1);
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    const char *names[3] = {"kernel_mul_mv_q8_0_f32_bf16_pair", "kernel_dsv41_norm_pair", "kernel_mul_mv_q8_0_f32_bf16_rope"};
    uint64_t runs[3];
    for (int k = 0; k < 3; k++) runs[k] = ds4_gpu_test_kernel_runs(names[k]);
    CHECK(ds4_gpu_matmul_q8_0_bf16_tensor(a0, model, bytes, 0, IN, OUT_A, in) &&
          ds4_gpu_matmul_q8_0_bf16_tensor(a1, model, bytes, a_bytes, IN, OUT_B, in) &&
          ds4_gpu_dsv41_project_pair_q8(b0, b1, model, bytes, 0, a_bytes, IN, OUT_A, OUT_B, in));
    CHECK(same_bits(a0, b0, OUT_A) && same_bits(a1, b1, OUT_B));
    for (uint32_t rows = 1; rows <= ROWS; rows += 2) {
        CHECK(ds4_gpu_rms_norm_weight_rows_tensor(a0, in, model, bytes, n_off, OUT_A, rows, 1e-6f) &&
              ds4_gpu_dsv41_quantize(a0, OUT_A, rows, DS4_V41_BF16) &&
              ds4_gpu_rms_norm_weight_rows_tensor(a1, in, model, bytes, n_off + OUT_A * 4, OUT_B, rows, 1e-6f) &&
              ds4_gpu_dsv41_quantize(a1, OUT_B, rows, DS4_V41_BF16) &&
              ds4_gpu_dsv41_norm_pair_rows(b0, in, n_off, OUT_A, b1, in, n_off + OUT_A * 4, OUT_B, model, bytes, 1e-6f, rows));
        CHECK(same_bits(a0, b0, rows * OUT_A) && same_bits(a1, b1, rows * OUT_B));
    }
    for (int compressed = 0; compressed < 2; compressed++) {
        CHECK(ds4_gpu_matmul_q8_0_bf16_tensor(a0, model, bytes, a_bytes + b_bytes, Q_IN, Q_OUT, in) &&
              ds4_gpu_dsv41_rope(a0, 512, Q_OUT / 512, 1, 777, compressed, false) &&
              ds4_gpu_dsv41_project_q(b0, model, bytes, a_bytes + b_bytes, Q_IN, Q_OUT, in, 777, compressed));
        CHECK(same_bits(a0, b0, Q_OUT));
    }
    CHECK(ds4_gpu_test_kernel_runs(names[0]) == runs[0] + 1 && ds4_gpu_test_kernel_runs(names[1]) == runs[1] + 2 &&
          ds4_gpu_test_kernel_runs(names[2]) == runs[2] + 2);
    ds4_gpu_test_set_flags(0);
    ds4_gpu_tensor_free(in); ds4_gpu_tensor_free(a0); ds4_gpu_tensor_free(b0); ds4_gpu_tensor_free(a1); ds4_gpu_tensor_free(b1);
    /* model stays: the model views may wrap it */
    fprintf(stderr, "V4.1 q_a/kv pair, paired norm and q projection with RoPE == separate kernels\n");
    return 1;
}

static void *q8_weights(uint32_t rows, uint32_t cols, uint64_t *bytes) {
    typedef struct { uint16_t d; int8_t qs[32]; } q8_block;
    *bytes = (uint64_t)rows * cols / 32 * sizeof(q8_block);
    void *model = NULL;
    if (posix_memalign(&model, getpagesize(), *bytes) != 0) return NULL;
    q8_block *w = model;
    for (uint64_t i = 0; i < *bytes / sizeof(*w); i++) {
        w[i].d = (uint16_t)(0x1c00 + (int)(random_value() * 8192) % 1024);
        for (int j = 0; j < 32; j++) w[i].qs[j] = (int8_t)((int)(random_value() * 8192) % 127);
    }
    return model;
}

/* ROPE, bit for bit. A cache row roped, quantized and stored in one dispatch against RoPE, quantization and the copy
 * (three formats, raw and compressed, through 1M positions). With the step's RoPE table, from the same attention
 * partials: the transforming reduce against the plain reduce, bf16 rounding and inverse RoPE; the plain low on its
 * heads and the rope low on the plain heads against the low after the separate RoPE; the index query's one-dispatch
 * rope + quantize against the two. */
static int check_rope(void) {
    enum { GROUP = 4096, RANK = 1024, GROUPS = 8, NROWS = 64, DV = 512, NWG = 32, QN = 32 * 128 };
    const uint64_t hb = (uint64_t)NROWS * DV * 4, lb = (uint64_t)GROUPS * RANK * 4;
    const uint64_t tn = (uint64_t)NROWS * (DV * NWG + 2u * NWG);
    ds4_gpu_tensor *a = upload(NULL, 512 * 4), *b = upload(NULL, 512 * 4), *dst = upload(NULL, 512 * 4);
    CHECK(a && b && dst);
    float *x = ds4_gpu_tensor_contents(a), *y = ds4_gpu_tensor_contents(b);
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    for (uint32_t t = 0; t < 48; t++) {
        const ds4_v41_activation_format format = (ds4_v41_activation_format)(DS4_V41_FP8_E8M0 + t % 3);
        const uint32_t width = format == DS4_V41_FP4_E8M0 ? 128 : 512, pos = (t * 104729u) % 1048576u;
        const bool compressed = (t / 3) & 1;
        const float scale = t % 7 == 0 ? 0.02f : t % 5 == 0 ? 12.0f : 1.0f;
        for (uint32_t i = 0; i < width; i++) x[i] = y[i] = random_value() * scale;
        memset(ds4_gpu_tensor_contents(dst), 0xff, 512 * 4);
        const uint64_t runs = ds4_gpu_test_kernel_runs("kernel_dsv41_rope_quantize");
        CHECK(ds4_gpu_dsv41_rope(a, width, 1, 1, pos, compressed, false) &&
              ds4_gpu_dsv41_quantize(a, width, 1, format) &&
              ds4_gpu_dsv41_rope_quantize(b, dst, 0, width, pos, compressed, format));
        CHECK(ds4_gpu_test_kernel_runs("kernel_dsv41_rope_quantize") == runs + 1);
        CHECK(!memcmp(x, ds4_gpu_tensor_contents(dst), width * 4));
    }
    uint64_t wb = 0;
    void *model = q8_weights(GROUPS * RANK, GROUP, &wb);
    float *t = malloc(tn * 4), *qref = malloc(QN * 4);
    CHECK(model && t && qref && ds4_gpu_set_model_map(model, wb));
    for (uint64_t i = 0; i < (uint64_t)NROWS * DV * NWG; i++) t[i] = random_value();
    for (uint64_t i = (uint64_t)NROWS * DV * NWG; i < tn; i += 2) {
        t[i] = fabsf(random_value()) + 0.25f;   /* S */
        t[i + 1] = random_value();              /* M */
    }
    ds4_gpu_tensor *tmp = upload(t, tn * 4), *plain = upload(NULL, hb), *once = upload(NULL, hb), *roped = upload(NULL, hb);
    ds4_gpu_tensor *q = upload(NULL, QN * 4), *tab = upload(NULL, 512);
    ds4_gpu_tensor *low[3] = {upload(NULL, lb), upload(NULL, lb), upload(NULL, lb)};
    CHECK(tmp && plain && once && roped && q && tab && low[0] && low[1] && low[2]);
    const char *names[4] = {"kernel_dsv41_rope_coeff_table", "kernel_dsv41_flash_attn_vec_reduce_heads",
                            "kernel_dsv4_attn_out_low_q8_0_f32_rope", "kernel_dsv41_rope_quantize_heads_tab"};
    const uint32_t positions[] = {0, 1, 7, 4095, 8191, 65535, 131071, 300000, 1048575};
    for (size_t pi = 0; pi < sizeof(positions) / sizeof(*positions); pi++) for (int compressed = 0; compressed < 2; compressed++) {
        const uint32_t pos = positions[pi];
        uint64_t runs[4];
        for (int k = 0; k < 4; k++) runs[k] = ds4_gpu_test_kernel_runs(names[k]);
        CHECK(ds4_gpu_dsv41_rope_coeff_table(tab, pos));
        CHECK(ds4_gpu_dsv41_flash_attn_reduce_check(plain, tmp, NROWS, 0, NULL, false));
        CHECK(ds4_gpu_dsv41_flash_attn_reduce_check(once, tmp, NROWS, 1, tab, compressed));
        /* the separate path: bf16 rounding, inverse RoPE, then the plain low */
        CHECK(ds4_gpu_tensor_write(roped, 0, ds4_gpu_tensor_contents(plain), hb));
        CHECK(ds4_gpu_dsv41_quantize(roped, NROWS * DV, 1, DS4_V41_BF16) &&
              ds4_gpu_dsv41_rope(roped, DV, NROWS, 1, pos, compressed, true));
        CHECK(!memcmp(ds4_gpu_tensor_contents(roped), ds4_gpu_tensor_contents(once), hb));
        CHECK(ds4_gpu_attention_output_low_q8_tensor(low[0], model, wb, 0, GROUP, RANK, GROUPS, roped) &&
              ds4_gpu_attention_output_low_q8_tensor(low[1], model, wb, 0, GROUP, RANK, GROUPS, once) &&
              ds4_gpu_dsv41_attention_low(low[2], model, wb, 0, GROUP, RANK, GROUPS, plain, pos, compressed));
        CHECK(!memcmp(ds4_gpu_tensor_contents(low[0]), ds4_gpu_tensor_contents(low[1]), lb));
        CHECK(!memcmp(ds4_gpu_tensor_contents(low[0]), ds4_gpu_tensor_contents(low[2]), lb));
        /* the index query from the plain heads */
        CHECK(ds4_gpu_tensor_write(q, 0, ds4_gpu_tensor_contents(plain), QN * 4));
        CHECK(ds4_gpu_dsv41_rope(q, 128, 32, 1, pos, compressed, false) &&
              ds4_gpu_dsv41_quantize(q, 128, 32, DS4_V41_FP4_E8M0) && ds4_gpu_tensor_read(q, 0, qref, QN * 4));
        CHECK(ds4_gpu_tensor_write(q, 0, ds4_gpu_tensor_contents(plain), QN * 4));
        CHECK(ds4_gpu_dsv41_rope_quantize_heads_tab(q, 128, 32, pos, compressed, DS4_V41_FP4_E8M0, tab));
        CHECK(!memcmp(qref, ds4_gpu_tensor_contents(q), QN * 4));
        for (int k = 0; k < 4; k++) CHECK(ds4_gpu_test_kernel_runs(names[k]) == runs[k] + 1);
    }
    ds4_gpu_test_set_flags(0);
    ds4_gpu_tensor *tensors[] = {a, b, dst, tmp, plain, once, roped, q, tab, low[0], low[1], low[2]};
    for (size_t i = 0; i < sizeof(tensors) / sizeof(*tensors); i++) ds4_gpu_tensor_free(tensors[i]);
    free(t); free(qref);
    /* model stays: the model views may wrap it */
    fprintf(stderr, "V4.1 RoPE: cache-row rope + quantize, transforming reduce, rope low, table index query == "
            "separate kernels\n");
    return 1;
}

/* FFN_JOIN: the joined expand against the add, the bf16 rounding, the expand and its rounding, bit for bit. */
static int check_ffn_join(void) {
    enum { N = 5120, HC = 4, SPLIT = 24 };
    float *v = malloc((size_t)(2 * N + HC * N + SPLIT) * sizeof(float));
    CHECK(v);
    for (int i = 0; i < 2 * N + HC * N; i++) v[i] = random_value();
    for (int i = 0; i < SPLIT; i++) v[2 * N + HC * N + i] = fabsf(random_value()) / 4;
    ds4_gpu_tensor *routed = upload(v, N * 4), *shared = upload(v + N, N * 4), *res = upload(v + 2 * N, HC * N * 4);
    ds4_gpu_tensor *split = upload(v + 2 * N + HC * N, SPLIT * 4), *block = upload(NULL, N * 4);
    ds4_gpu_tensor *a = upload(NULL, HC * N * 4), *b = upload(NULL, HC * N * 4);
    CHECK(routed && shared && res && split && block && a && b);
    CHECK(ds4_gpu_add_tensor(block, routed, shared, N) && ds4_gpu_dsv41_quantize(block, N, 1, DS4_V41_BF16) &&
          ds4_gpu_hc_expand_split_tensor(a, block, res, split, N, HC) && ds4_gpu_dsv41_quantize(a, HC * N, 1, DS4_V41_BF16));
    CHECK(ds4_gpu_hc_expand_add_split_bf16_tensor(b, routed, shared, res, split, N, HC));
    CHECK(same_bits(a, b, HC * N));
    ds4_gpu_tensor *tensors[] = {routed, shared, res, split, block, a, b};
    for (size_t i = 0; i < sizeof(tensors) / sizeof(*tensors); i++) ds4_gpu_tensor_free(tensors[i]);
    free(v);
    fprintf(stderr, "V4.1 FFN join: add, bf16, expand and rounding in one dispatch == separate kernels\n");
    return 1;
}

static uint32_t random_bits(void) {
    (void)random_value();
    return seed;
}

/* HC_FUSION, bit for bit, on random weights of the V4.1 shapes: the HC block input against the norm, mix, split,
 * weighted sum and norm kernels (one row with both pre layouts, and three rows against the batch kernels); the
 * expanded Q8 matvec against the matvec and the expand (8192 -> 5120 alone, 2304 -> 5120 joined with an add); and the
 * deferred collapse plus the query kernel against the block input, the q projection and the KV row's RoPE store. */
static int check_hc_fusion(void) {
    const uint64_t qo = 0, ho = qo + 32768ull * 40 * 34, so = ho + 20480ull * 24 * 2, bo = so + 64, no = bo + 128,
                   ao = no + 5120 * 4, eo = ao + 5120ull * 256 * 34, bytes = eo + 5120ull * 72 * 34;
    uint8_t *map = NULL;
    CHECK(posix_memalign((void **)&map, getpagesize(), bytes) == 0);
    memset(map, 0, bytes);
    const uint64_t q8[3][2] = {{qo, 32768ull * 40}, {ao, 5120ull * 256}, {eo, 5120ull * 72}};
    for (int k = 0; k < 3; k++)
        for (uint64_t b = 0; b < q8[k][1]; b++) {   /* Q8_0 blocks: f16 scale, 32 int8 */
            const uint16_t d = (uint16_t)(0x2000u + random_bits() % 0x1800u);
            memcpy(map + q8[k][0] + b * 34, &d, 2);
            for (int j = 0; j < 32; j++) map[q8[k][0] + b * 34 + 2 + j] = (uint8_t)random_bits();
        }
    uint16_t *fn = (uint16_t *)(map + ho);
    for (uint32_t i = 0; i < 20480 * 24; i++)
        fn[i] = (uint16_t)((random_bits() & 0x8000u) | (0x2000u + random_bits() % 0x1800u));
    float *scale = (float *)(map + so), *base = (float *)(map + bo), *nw = (float *)(map + no);
    scale[0] = .04f; scale[1] = .03f; scale[2] = .02f;
    for (int i = 0; i < 24; i++) base[i] = random_value() / 32.0f;
    for (int i = 0; i < 5120; i++) nw[i] = 1.0f + random_value() / 64.0f;
    CHECK(ds4_gpu_set_model_map(map, bytes));
    enum { ROWS = 3 };
    enum { MIX, SPLIT, X, NORM, NT };
    const uint32_t width[NT] = {24, 24, 5120, 5120};
    ds4_gpu_tensor *res = upload(NULL, ROWS * 20480 * 4), *pre = upload(NULL, ROWS * 24 * 4);
    ds4_gpu_tensor *flat = upload(NULL, ROWS * 20480 * 4), *t[2][2][NT];   /* [ref, fused][1 row, ROWS] */
    CHECK(res && pre && flat);
    for (int a = 0; a < 2; a++)
        for (int n = 0; n < 2; n++)
            for (int i = 0; i < NT; i++) CHECK((t[a][n][i] = upload(NULL, (n ? ROWS : 1) * width[i] * 4)) != NULL);
    ds4_gpu_tensor *res1 = ds4_gpu_tensor_view(res, 0, 20480 * 4), *flat1 = ds4_gpu_tensor_view(flat, 0, 20480 * 4);
    ds4_gpu_tensor *pre4 = ds4_gpu_tensor_view(pre, 0, 4 * 4), *pre24 = ds4_gpu_tensor_view(pre, 0, 24 * 4);
    CHECK(res1 && flat1 && pre4 && pre24);
    float *rv = ds4_gpu_tensor_contents(res), *pv = ds4_gpu_tensor_contents(pre);
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    const char *names[4] = {"kernel_dsv41_hc_block_input", "kernel_mul_mv_q8_0_f32_bf16io_hc_expand4",
                            "kernel_dsv41_deferred_collapse", "kernel_dsv41_query_deferred_hc"};
    uint64_t runs[4];
    for (int k = 0; k < 4; k++) runs[k] = ds4_gpu_test_kernel_runs(names[k]);
    ds4_gpu_test_set_v41_off(DS4_GPU_V41_FIXED_SHAPES);   /* the general expand; check_fixed_shapes has the V4.1 ones */
    /* the block input: one row with pre as 4 weights and as a split row, then three rows (split rows) */
    for (int step = 0; step < 3; step++) {
        const int n = step == 2;
        const uint32_t rows = n ? ROWS : 1;
        for (uint32_t i = 0; i < ROWS * 20480; i++) rv[i] = random_value();
        for (uint32_t i = 0; i < ROWS * 24; i++) pv[i] = fabsf(random_value()) / 4;
        ds4_gpu_tensor **r = t[0][n], **d = t[1][n];
        ds4_gpu_tensor *in = n ? res : res1, *w = step == 0 ? pre4 : n ? pre : pre24;
        CHECK((n ? ds4_gpu_rms_norm_plain_rows_tensor(flat, res, 20480, ROWS, 1e-6f) &&
                   ds4_gpu_dsv41_projection_rows(r[MIX], map, bytes, ho, 20480, 24, ROWS, flat) :
                   ds4_gpu_rms_norm_plain_tensor(flat1, res1, 20480, 1e-6f) &&
                   ds4_gpu_matmul_f16_tensor(r[MIX], map, bytes, ho, 20480, 24, flat1, 1)) &&
              ds4_gpu_hc_split_sinkhorn_tensor(r[SPLIT], r[MIX], map, bytes, so, bo, 4, 20, 1e-6f) &&
              (step ? ds4_gpu_hc_weighted_sum_split_tensor(r[X], in, w, 5120, 4) :
                      ds4_gpu_hc_weighted_sum_tensor(r[X], in, w, 5120, 4)) &&
              ds4_gpu_dsv41_quantize(r[X], 5120, rows, DS4_V41_BF16) &&
              ds4_gpu_rms_norm_weight_rows_tensor(r[NORM], r[X], map, bytes, no, 5120, rows, 1e-6f) &&
              ds4_gpu_dsv41_quantize(r[NORM], 5120, rows, DS4_V41_BF16));
        CHECK(n ? ds4_gpu_dsv41_hc_block_input_rows(d[MIX], d[X], d[NORM], d[SPLIT], res, pre, map, bytes, ho, so, bo,
                                                    no, 20480, 24, 5120, 4, 20, 1e-6f, 1e-6f, ROWS, 24) :
                  ds4_gpu_dsv41_hc_block_input(d[MIX], d[X], d[NORM], d[SPLIT], res1, w, map, bytes, ho, so, bo, no,
                                               20480, 24, 5120, 4, 20, 1e-6f, 1e-6f));
        for (int i = 0; i < NT; i++) CHECK(same_bits(r[i], d[i], rows * width[i]));
    }
    /* the expanded matvec: x, the add, the block and the streams */
    ds4_gpu_tensor *xin = upload(NULL, 8192 * 4), *add = upload(NULL, 5120 * 4), *block = upload(NULL, 5120 * 4);
    ds4_gpu_tensor *ref = upload(NULL, 20480 * 4), *got = upload(NULL, 20480 * 4);
    CHECK(xin && add && block && ref && got);
    for (int joined = 0; joined < 2; joined++) {
        const uint32_t k = joined ? 2304 : 8192;
        const uint64_t w = joined ? eo : ao;
        float *xv = ds4_gpu_tensor_contents(xin), *av = ds4_gpu_tensor_contents(add);
        for (uint32_t i = 0; i < k; i++) xv[i] = random_value();
        for (int i = 0; i < 5120; i++) av[i] = random_value();
        for (int i = 0; i < 20480; i++) rv[i] = random_value();
        ds4_gpu_tensor *x = ds4_gpu_tensor_view(xin, 0, k * 4);
        CHECK(x);
        CHECK(ds4_gpu_matmul_q8_0_bf16io_tensor(block, map, bytes, w, k, 5120, x) &&
              (joined ? ds4_gpu_hc_expand_add_split_bf16_tensor(ref, block, add, res1, pre24, 5120, 4) :
                        ds4_gpu_hc_expand_split_bf16_tensor(ref, block, res1, pre24, 5120, 4)));
        CHECK(ds4_gpu_dsv41_matmul_expand(got, map, bytes, w, k, 5120, x, joined ? add : NULL, res1, pre24, 4));
        CHECK(same_bits(ref, got, 20480));
        ds4_gpu_tensor_free(x);
    }
    /* the deferred block input with the query kernel, which also stores the KV row */
    ds4_gpu_tensor *qr = upload(NULL, 1280 * 4), *kv = upload(NULL, 512 * 4), *q[2] = {upload(NULL, 32768 * 4),
                   upload(NULL, 32768 * 4)}, *cache[2] = {upload(NULL, 512 * 4), upload(NULL, 512 * 4)};
    CHECK(qr && kv && q[0] && q[1] && cache[0] && cache[1]);
    for (uint32_t draw = 0; draw < 8; draw++) {
        const bool compressed = draw & 1;
        const uint32_t pos = (draw * 7919u) % 393216u;
        for (int i = 0; i < 20480; i++) rv[i] = random_value();
        for (int i = 0; i < 4; i++) pv[i] = random_value();
        for (int i = 0; i < 1280; i++) ((float *)ds4_gpu_tensor_contents(qr))[i] = random_value();
        for (int i = 0; i < 512; i++) ((float *)ds4_gpu_tensor_contents(kv))[i] = random_value();
        ds4_gpu_tensor **r = t[0][0], **d = t[1][0];
        CHECK(ds4_gpu_dsv41_hc_block_input(r[MIX], r[X], r[NORM], r[SPLIT], res1, pre4, map, bytes, ho, so, bo, no,
                                           20480, 24, 5120, 4, 20, 1e-6f, 1e-6f) &&
              ds4_gpu_dsv41_project_q(q[0], map, bytes, qo, 1280, 32768, qr, pos, compressed) &&
              ds4_gpu_dsv41_rope_quantize(kv, cache[0], 0, 512, pos, compressed, DS4_V41_FP8_E8M0));
        CHECK(ds4_gpu_dsv41_deferred_collapse(d[X], d[NORM], res1, pre4, map, bytes, no, 1e-6f) &&
              ds4_gpu_dsv41_query_deferred_hc(q[1], d[MIX], d[SPLIT], qr, res1, map, bytes, qo, ho, so, bo, pos,
                                              compressed, 20, 1e-6f, 1e-6f, d[X], d[NORM], pre4, no, kv, cache[1], 0));
        for (int i = 0; i < NT; i++) CHECK(same_bits(r[i], d[i], width[i]));
        CHECK(same_bits(q[0], q[1], 32768) && same_bits(cache[0], cache[1], 512));
    }
    CHECK(ds4_gpu_test_kernel_runs(names[0]) == runs[0] + 3 + 8 && ds4_gpu_test_kernel_runs(names[1]) == runs[1] + 2 &&
          ds4_gpu_test_kernel_runs(names[2]) == runs[2] + 8 && ds4_gpu_test_kernel_runs(names[3]) == runs[3] + 8);
    ds4_gpu_test_set_v41_off(0);
    ds4_gpu_test_set_flags(0);
    ds4_gpu_tensor *tensors[] = {res1, flat1, pre4, pre24, res, pre, flat, xin, add, block, ref, got,
                                 qr, kv, q[0], q[1], cache[0], cache[1]};
    for (size_t i = 0; i < sizeof(tensors) / sizeof(*tensors); i++) ds4_gpu_tensor_free(tensors[i]);
    for (int a = 0; a < 2; a++)
        for (int n = 0; n < 2; n++)
            for (int i = 0; i < NT; i++) ds4_gpu_tensor_free(t[a][n][i]);
    /* the model stays: the model views may wrap it */
    fprintf(stderr, "V4.1 HC fusion: block input, expanded matvec, deferred collapse and query == separate kernels\n");
    return 1;
}

/* SPLIT_REDUCE, bit for bit against the single-group forms with the bit off: the Engram gate over 256-thread groups
 * (five rows) and the deferred HC collapse over 32 + 10 groups (fresh inputs each draw, some of them large). */
static int check_split_reduce(void) {
    enum { D = 5120, ROWS = 5 };
    const uint64_t wb = 65536;
    void *model = NULL;
    CHECK(posix_memalign(&model, getpagesize(), wb) == 0);
    float *w = model;
    ds4_gpu_tensor *x[2] = {upload(NULL, ROWS * 4 * D * 4), upload(NULL, ROWS * 4 * D * 4)};
    ds4_gpu_tensor *kv = upload(NULL, ROWS * 5 * D * 4), *qw = upload(NULL, 4 * D * 4), *kw = upload(NULL, 4 * D * 4);
    ds4_gpu_tensor *pre = upload(NULL, 16), *c[2] = {upload(NULL, D * 4), upload(NULL, D * 4)};
    ds4_gpu_tensor *n[2] = {upload(NULL, D * 4), upload(NULL, D * 4)};
    CHECK(x[0] && x[1] && kv && qw && kw && pre && c[0] && c[1] && n[0] && n[1] && ds4_gpu_set_model_map(model, wb));
    float *xv = ds4_gpu_tensor_contents(x[0]), *kvv = ds4_gpu_tensor_contents(kv), *pv = ds4_gpu_tensor_contents(pre);
    for (int i = 0; i < ROWS * 4 * D; i++) xv[i] = bf16(random_value());
    for (int i = 0; i < ROWS * 5 * D; i++) kvv[i] = bf16(random_value());
    for (int i = 0; i < 4 * D; i++) {
        ((float *)ds4_gpu_tensor_contents(qw))[i] = random_value();
        ((float *)ds4_gpu_tensor_contents(kw))[i] = random_value();
    }
    memcpy(ds4_gpu_tensor_contents(x[1]), xv, ROWS * 4 * D * 4);
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    const uint64_t tg = ds4_gpu_test_kernel_runs("kernel_dsv41_engram_add_tg256");
    CHECK(ds4_gpu_dsv41_engram_add(x[0], kv, qw, kw, NULL, D, ROWS, 1e-6f));
    ds4_gpu_test_set_v41_off(DS4_GPU_V41_SPLIT_REDUCE);
    const int ok = ds4_gpu_dsv41_engram_add(x[1], kv, qw, kw, NULL, D, ROWS, 1e-6f);
    ds4_gpu_test_set_v41_off(0);
    CHECK(ok && ds4_gpu_test_kernel_runs("kernel_dsv41_engram_add_tg256") == tg + 1);
    CHECK(same_bits(x[0], x[1], ROWS * 4 * D));
    const uint64_t part = ds4_gpu_test_kernel_runs("kernel_dsv41_deferred_collapse_part");
    for (int draw = 0; draw < 16; draw++) {
        for (int i = 0; i < D; i++) w[i] = 1.0f + random_value() / 8.0f;
        for (int i = 0; i < 4 * D; i++) xv[i] = random_value() * (draw % 4 == 3 ? 64.0f : 1.0f);
        for (int i = 0; i < 4; i++) pv[i] = random_value();
        CHECK(ds4_gpu_dsv41_deferred_collapse(c[0], n[0], x[0], pre, model, wb, 0, 1e-6f));
        ds4_gpu_test_set_v41_off(DS4_GPU_V41_SPLIT_REDUCE);
        const int single = ds4_gpu_dsv41_deferred_collapse(c[1], n[1], x[0], pre, model, wb, 0, 1e-6f);
        ds4_gpu_test_set_v41_off(0);
        CHECK(single && same_bits(c[0], c[1], D) && same_bits(n[0], n[1], D));
    }
    CHECK(ds4_gpu_test_kernel_runs("kernel_dsv41_deferred_collapse_part") == part + 16);
    ds4_gpu_test_set_flags(0);
    ds4_gpu_tensor *tensors[] = {x[0], x[1], kv, qw, kw, pre, c[0], c[1], n[0], n[1]};
    for (size_t i = 0; i < sizeof(tensors) / sizeof(*tensors); i++) ds4_gpu_tensor_free(tensors[i]);
    /* the model stays: the model views may wrap it */
    fprintf(stderr, "V4.1 split reduce: Engram gate and HC collapse == single-group forms\n");
    return 1;
}

/* SHARED_EXPERT, bit for bit: the fused gate, up and SwiGLU against the bf16 Q8 matvecs and the SwiGLU kernel, for
 * one row and for three (5120 -> 2304, the V4.1 shape, clamp 10). */
static int check_shared_expert(void) {
    enum { IN = 5120, OUT = 2304, ROWS = 3 };
    const uint64_t row_bytes = IN / 32 * 34, wb = 2ull * OUT * row_bytes;
    uint8_t *map = NULL;
    CHECK(posix_memalign((void **)&map, getpagesize(), wb) == 0);
    for (uint64_t b = 0; b < wb / 34; b++) {   /* Q8_0 blocks: f16 scale, 32 int8 */
        const uint16_t d = (uint16_t)(0x2000u + random_bits() % 0x1800u);
        memcpy(map + b * 34, &d, 2);
        for (int j = 0; j < 32; j++) map[b * 34 + 2 + j] = (uint8_t)random_bits();
    }
    CHECK(ds4_gpu_set_model_map(map, wb));
    ds4_gpu_tensor *x = upload(NULL, ROWS * IN * 4), *gate = upload(NULL, ROWS * OUT * 4), *up = upload(NULL, ROWS * OUT * 4);
    ds4_gpu_tensor *ref = upload(NULL, ROWS * OUT * 4), *got = upload(NULL, ROWS * OUT * 4);
    CHECK(x && gate && up && ref && got);
    for (int i = 0; i < ROWS * IN; i++) ((float *)ds4_gpu_tensor_contents(x))[i] = bf16(random_value());
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    const uint64_t runs = ds4_gpu_test_kernel_runs("kernel_dsv4_shared_mid_swiglu_q8_0_bf16");
    ds4_gpu_test_set_v41_off(DS4_GPU_V41_ROWS_MMA);   /* the row kernels; check_rows_mma has the matrix form */
    for (uint32_t rows = 1; rows <= ROWS; rows += ROWS - 1) {
        ds4_gpu_tensor *xr = ds4_gpu_tensor_view(x, 0, rows * IN * 4), *mid = ds4_gpu_tensor_view(ref, 0, rows * OUT * 4);
        ds4_gpu_tensor *gr = ds4_gpu_tensor_view(gate, 0, rows * OUT * 4), *ur = ds4_gpu_tensor_view(up, 0, rows * OUT * 4);
        CHECK(xr && mid && gr && ur);
        CHECK((rows == 1 ? ds4_gpu_matmul_q8_0_bf16_tensor(gr, map, wb, 0, IN, OUT, xr) &&
                           ds4_gpu_matmul_q8_0_bf16_tensor(ur, map, wb, OUT * row_bytes, IN, OUT, xr) :
                           ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(gr, map, wb, 0, IN, OUT, xr, rows) &&
                           ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(ur, map, wb, OUT * row_bytes, IN, OUT, xr, rows) &&
                           ds4_gpu_dsv41_quantize(gr, OUT, rows, DS4_V41_BF16) &&
                           ds4_gpu_dsv41_quantize(ur, OUT, rows, DS4_V41_BF16)) &&
              ds4_gpu_swiglu_tensor(mid, gr, ur, rows * OUT, 10.0f, 1.0f));
        CHECK(ds4_gpu_dsv41_shared_swiglu(got, map, wb, 0, OUT * row_bytes, IN, OUT, xr, 10.0f, rows));
        CHECK(same_bits(ref, got, rows * OUT));
        ds4_gpu_tensor *views[] = {xr, mid, gr, ur};
        for (size_t i = 0; i < sizeof(views) / sizeof(*views); i++) ds4_gpu_tensor_free(views[i]);
    }
    CHECK(ds4_gpu_test_kernel_runs("kernel_dsv4_shared_mid_swiglu_q8_0_bf16") == runs + 2);
    ds4_gpu_test_set_v41_off(0);
    ds4_gpu_test_set_flags(0);
    ds4_gpu_tensor *tensors[] = {x, gate, up, ref, got};
    for (size_t i = 0; i < sizeof(tensors) / sizeof(*tensors); i++) ds4_gpu_tensor_free(tensors[i]);
    /* the model stays: the model views may wrap it */
    fprintf(stderr, "V4.1 shared expert: fused gate, up and SwiGLU == bf16 matvecs and SwiGLU, one and three rows\n");
    return 1;
}

/* CONCURRENT_FFN: the concurrent encoder's shared expert, started for V4.1 and joined by a decode routed MoE (six of
 * 384 MXFP4 experts at the V4.1 shapes), against the bf16 shared expert and the routed MoE run one after the other,
 * bit for bit. */
static int check_concurrent_ffn(void) {
    enum { D = 5120, FF = 2304, NE = 384, NU = 6, MXFP4 = 39 /* the tensor type */ };
    const uint64_t q8_row = D / 32 * 34, mx_gate_row = D / 32 * 17, mx_down_row = FF / 32 * 17;
    const uint64_t gate_o = 0, up_o = gate_o + FF * q8_row, down_o = up_o + FF * q8_row,
                   eg_o = down_o + (uint64_t)D * (FF / 32 * 34), eu_o = eg_o + (uint64_t)NE * FF * mx_gate_row,
                   ed_o = eu_o + (uint64_t)NE * FF * mx_gate_row, wb = ed_o + (uint64_t)NE * D * mx_down_row;
    uint8_t *map = NULL;
    CHECK(posix_memalign((void **)&map, getpagesize(), wb) == 0);
    for (uint64_t b = 0; b < eg_o / 34; b++) {   /* Q8_0 blocks: f16 scale, 32 int8 */
        const uint16_t d = (uint16_t)(0x2000u + random_bits() % 0x1800u);
        memcpy(map + b * 34, &d, 2);
        for (int j = 0; j < 32; j++) map[b * 34 + 2 + j] = (uint8_t)random_bits();
    }
    for (uint64_t b = eg_o; b < wb; b += 17) {   /* MXFP4 blocks: e8m0 scale, 16 bytes of nibbles */
        map[b] = (uint8_t)(118u + random_bits() % 8u);
        for (int j = 1; j < 17; j++) map[b + j] = (uint8_t)random_bits();
    }
    CHECK(ds4_gpu_set_model_map(map, wb));
    ds4_gpu_tensor *x = upload(NULL, D * 4), *sg = upload(NULL, FF * 4), *su = upload(NULL, FF * 4);
    ds4_gpu_tensor *mid[2] = {upload(NULL, FF * 4), upload(NULL, FF * 4)}, *shared[2] = {upload(NULL, D * 4), upload(NULL, D * 4)};
    ds4_gpu_tensor *routed[2] = {upload(NULL, D * 4), upload(NULL, D * 4)};
    ds4_gpu_tensor *gate = upload(NULL, NU * FF * 4), *up = upload(NULL, NU * FF * 4), *emid = upload(NULL, NU * FF * 4);
    ds4_gpu_tensor *experts = upload(NULL, NU * D * 4), *sel = upload(NULL, NU * 4), *weights = upload(NULL, NU * 4);
    CHECK(x && sg && su && mid[0] && mid[1] && shared[0] && shared[1] && routed[0] && routed[1] && gate && up &&
          emid && experts && sel && weights);
    for (int i = 0; i < D; i++) ((float *)ds4_gpu_tensor_contents(x))[i] = bf16(random_value());
    for (int i = 0; i < NU; i++) {
        ((int32_t *)ds4_gpu_tensor_contents(sel))[i] = (int32_t)(i * 61 + 7);
        ((float *)ds4_gpu_tensor_contents(weights))[i] = fabsf(random_value()) / 4;
    }
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    const uint64_t runs = ds4_gpu_test_kernel_runs("ds4_gpu_parallel_ffn_start_bf16");
    for (int concurrent = 0; concurrent < 2; concurrent++) {
        const int c = concurrent;
        CHECK(ds4_gpu_begin_commands());
        if (c) CHECK(ds4_gpu_parallel_ffn_start_bf16(sg, su, mid[c], shared[c], map, wb, gate_o, up_o, down_o, D, FF,
                                                     x, 10.0f));
        else CHECK(ds4_gpu_dsv41_shared_swiglu(mid[c], map, wb, gate_o, up_o, D, FF, x, 10.0f, 1) &&
                   ds4_gpu_matmul_q8_0_bf16io_tensor(shared[c], map, wb, down_o, FF, D, mid[c]));
        CHECK(ds4_gpu_routed_moe_one_tensor(routed[c], gate, up, emid, experts, map, wb, eg_o, eu_o, ed_o,
                  MXFP4, MXFP4, mx_gate_row * FF, mx_gate_row, mx_down_row * D, mx_down_row,
                  D, FF, D, sel, weights, NE, NU, 10.0f, x, NULL, 3, 1));
        CHECK(!c || ds4_gpu_parallel_ffn_finish());
        CHECK(ds4_gpu_end_commands() && ds4_gpu_synchronize());
    }
    CHECK(ds4_gpu_test_kernel_runs("ds4_gpu_parallel_ffn_start_bf16") == runs + 1);
    ds4_gpu_test_set_flags(0);
    CHECK(same_bits(shared[0], shared[1], D) && same_bits(routed[0], routed[1], D));
    ds4_gpu_tensor *tensors[] = {x, sg, su, mid[0], mid[1], shared[0], shared[1], routed[0], routed[1], gate, up, emid,
                                 experts, sel, weights};
    for (size_t i = 0; i < sizeof(tensors) / sizeof(*tensors); i++) ds4_gpu_tensor_free(tensors[i]);
    /* the model stays: the model views may wrap it */
    fprintf(stderr, "V4.1 concurrent FFN: shared expert beside the routed MoE == one after the other\n");
    return 1;
}

/* INDEX_SCORE_WIDE: the wide one-token scorer against the per-key kernel (bit off) within float rounding, on
 * 8,192 and 9,000 keys; its masked form gives the wide scores where the block mask keeps a block and -inf where
 * it excludes one, bit for bit. */
static int check_index_score_wide(void) {
    enum { MAXR = 9000 };
    ds4_gpu_tensor *q = upload(NULL, 4096 * 4), *w = upload(NULL, 32 * 4), *keys = upload(NULL, MAXR * 128 * 4);
    ds4_gpu_tensor *mask = upload(NULL, (MAXR + 7) / 8 * 4), *s[3] = {upload(NULL, MAXR * 4), upload(NULL, MAXR * 4),
                                                                      upload(NULL, MAXR * 4)};
    CHECK(q && w && keys && mask && s[0] && s[1] && s[2]);
    float *qv = ds4_gpu_tensor_contents(q), *wv = ds4_gpu_tensor_contents(w), *kv = ds4_gpu_tensor_contents(keys);
    float *mv = ds4_gpu_tensor_contents(mask);
    for (int i = 0; i < 4096; i++) qv[i] = bf16(random_value());
    for (int i = 0; i < 32; i++) wv[i] = random_value() / 8;
    for (int i = 0; i < MAXR * 128; i++) kv[i] = bf16(random_value());
    for (int i = 0; i < (MAXR + 7) / 8; i++) mv[i] = i % 3 == 1 ? 1.0f : 0.0f;
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    const uint64_t wide = ds4_gpu_test_kernel_runs("kernel_glm_indexer_score_one_wide");
    for (uint32_t rows = 8192; rows <= MAXR; rows += MAXR - 8192) {
        CHECK(ds4_gpu_glm_indexer_score_one_tensor(s[0], q, w, keys, rows, 32, 128, 1.0f / 64.0f, false));
        ds4_gpu_test_set_v41_off(DS4_GPU_V41_INDEX_SCORE_WIDE);
        const int direct = ds4_gpu_glm_indexer_score_one_tensor(s[1], q, w, keys, rows, 32, 128, 1.0f / 64.0f, false);
        ds4_gpu_test_set_v41_off(0);
        CHECK(direct && ds4_gpu_dsv41_indexer_score_masked(s[2], q, w, keys, mask, rows));
        const float *a = ds4_gpu_tensor_contents(s[0]), *b = ds4_gpu_tensor_contents(s[1]), *c = ds4_gpu_tensor_contents(s[2]);
        for (uint32_t i = 0; i < rows; i++) {
            CHECK(fabsf(a[i] - b[i]) <= 1e-4f * (1.0f + fabsf(b[i])));
            if (mv[i / 8]) CHECK(isinf(c[i]) && c[i] < 0);
            else CHECK(!memcmp(&a[i], &c[i], 4));
        }
    }
    CHECK(ds4_gpu_test_kernel_runs("kernel_glm_indexer_score_one_wide") == wide + 2);
    ds4_gpu_test_set_flags(0);
    ds4_gpu_tensor *tensors[] = {q, w, keys, mask, s[0], s[1], s[2]};
    for (size_t i = 0; i < sizeof(tensors) / sizeof(*tensors); i++) ds4_gpu_tensor_free(tensors[i]);
    fprintf(stderr, "V4.1 wide index scorer: per-key kernel within rounding, masked form == wide scores and -inf\n");
    return 1;
}

/* INDEX_BF16: the bf16 key copy, its restore check (a key that is not bf16-exact sets the flag), and the bf16
 * scorers against the wide f32 scorer, bit for bit, on each of the three kernel sizes. */
static int check_index_bf16(void) {
    enum { MAXR = 66000 };
    ds4_gpu_tensor *q = upload(NULL, 4096 * 4), *w = upload(NULL, 32 * 4), *keys = upload(NULL, (uint64_t)MAXR * 128 * 4);
    ds4_gpu_tensor *view = upload(NULL, (uint64_t)MAXR * 128 * 2), *flag = upload(NULL, 4);
    ds4_gpu_tensor *s[2] = {upload(NULL, MAXR * 4), upload(NULL, MAXR * 4)};
    CHECK(q && w && keys && view && flag && s[0] && s[1]);
    float *qv = ds4_gpu_tensor_contents(q), *wv = ds4_gpu_tensor_contents(w), *kv = ds4_gpu_tensor_contents(keys);
    for (int i = 0; i < 4096; i++) qv[i] = bf16(random_value());
    for (int i = 0; i < 32; i++) wv[i] = random_value() / 8;
    for (uint64_t i = 0; i < (uint64_t)MAXR * 128; i++) kv[i] = bf16(random_value());
    uint32_t *fv = ds4_gpu_tensor_contents(flag);
    *fv = 0;
    CHECK(ds4_gpu_dsv41_index_keys_bf16(view, keys, 0, MAXR, flag) && ds4_gpu_synchronize() && *fv == 0);
    const float exact = kv[5 * 128 + 7];
    kv[5 * 128 + 7] = 1.0f + 0x1p-12f;   /* not bf16-exact */
    CHECK(ds4_gpu_dsv41_index_keys_bf16(view, keys, 5, 1, flag) && ds4_gpu_synchronize() && *fv != 0);
    kv[5 * 128 + 7] = exact;
    CHECK(ds4_gpu_dsv41_index_keys_bf16(view, keys, 5, 1, NULL));
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    const char *names[3] = {"kernel_dsv41_indexer_score_rows_split4_bf16", "kernel_dsv41_indexer_score_rows_split2_bf16",
                            "kernel_dsv41_indexer_score_bf16"};
    const uint32_t sizes[3] = {9001, 30001, MAXR};
    for (int k = 0; k < 3; k++) {
        const uint64_t runs = ds4_gpu_test_kernel_runs(names[k]);
        CHECK(ds4_gpu_glm_indexer_score_one_tensor(s[0], q, w, keys, sizes[k], 32, 128, 1.0f / 64.0f, false) &&
              ds4_gpu_dsv41_indexer_score_row(s[1], q, w, keys, view, sizes[k]));
        CHECK(ds4_gpu_test_kernel_runs(names[k]) == runs + 1 && same_bits(s[0], s[1], sizes[k]));
    }
    ds4_gpu_test_set_flags(0);
    ds4_gpu_tensor *tensors[] = {q, w, keys, view, flag, s[0], s[1]};
    for (size_t i = 0; i < sizeof(tensors) / sizeof(*tensors); i++) ds4_gpu_tensor_free(tensors[i]);
    fprintf(stderr, "V4.1 bf16 index keys: copy and restore check, bf16 scorers == wide f32 scorer\n");
    return 1;
}

/* TOPK: the V4.1 index top-k against the merge ladder (bit off). One row's top 512 of 4,000 and 12,000 scores
 * (shuffle sort, staged and fused merges) gives the same ids in the same order, ties included. The radix select, on
 * the top 512 of 20,000 and 300,000 (with the gather and the 512-wide sort) and the top 2,048 of 20,000 (its full
 * sort), gives the same scores by rank: exactly tied keys can change places (300,000 scores of 65,537 values tie). */
static int check_topk(void) {
    enum { MAXN = 300000 };
    ds4_gpu_tensor *scores = upload(NULL, MAXN * 4), *sel[2] = {upload(NULL, 2048 * 4), upload(NULL, 2048 * 4)};
    CHECK(scores && sel[0] && sel[1]);
    float *sv = ds4_gpu_tensor_contents(scores);
    const uint32_t n[5] = {4000, 12000, 20000, MAXN, 20000}, k[5] = {512, 512, 512, 512, 2048};
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    const uint64_t radix = ds4_gpu_test_kernel_runs("kernel_topk_select_gather");
    const uint64_t fused = ds4_gpu_test_kernel_runs("kernel_argsort_merge_f32_i32_desc_prefix_fused");
    for (int c = 0; c < 5; c++) {
        for (uint32_t i = 0; i < n[c]; i++) sv[i] = c == 1 && i % 7 == 0 ? 1.0f : random_value();   /* ties */
        CHECK(ds4_gpu_dsv41_indexer_topk_tensor(sel[0], scores, n[c], 1, k[c]));
        ds4_gpu_test_set_v41_off(DS4_GPU_V41_TOPK);
        const int ladder = ds4_gpu_dsv41_indexer_topk_tensor(sel[1], scores, n[c], 1, k[c]);
        ds4_gpu_test_set_v41_off(0);
        const int32_t *a = ds4_gpu_tensor_contents(sel[0]), *b = ds4_gpu_tensor_contents(sel[1]);
        CHECK(ladder);
        for (uint32_t i = 0; i < k[c]; i++) CHECK(c < 2 ? a[i] == b[i] : sv[a[i]] == sv[b[i]]);
    }
    CHECK(ds4_gpu_test_kernel_runs("kernel_topk_select_gather") == radix + 2 &&
          ds4_gpu_test_kernel_runs("kernel_argsort_merge_f32_i32_desc_prefix_fused") == fused + 2);
    ds4_gpu_test_set_flags(0);
    ds4_gpu_tensor *tensors[] = {scores, sel[0], sel[1]};
    for (size_t i = 0; i < sizeof(tensors) / sizeof(*tensors); i++) ds4_gpu_tensor_free(tensors[i]);
    fprintf(stderr, "V4.1 index top-k: radix select and short merge == merge ladder\n");
    return 1;
}

/* INDEX_CANDIDATES: one row of 20,003 keys (2,501 blocks, 2,048 kept, the newest among them) scored and picked
 * among its candidates, from the f32 keys and from their bf16 view, then again with the list reused, gives the masked
 * wide scorer's and the radix select's ids in their order. */
static int check_index_candidates(void) {
    enum { N = 20003, BLOCKS = (N + 7) / 8, K = 512 };
    float *v = malloc(((size_t)N * 128 + 4096 + 32 + BLOCKS) * sizeof(float));
    CHECK(v);
    for (size_t i = 0; i < (size_t)N * 128; i++) v[i] = bf16(random_value());
    float *qv = v + (size_t)N * 128, *wv = qv + 4096, *mv = wv + 32;
    for (int i = 0; i < 4096; i++) qv[i] = random_value();
    for (int i = 0; i < 32; i++) wv[i] = fabsf(random_value());
    for (int b = 0; b < BLOCKS; b++) mv[b] = 0.0f;
    for (int dropped = 0; dropped < BLOCKS - 2048;) {   /* drop 453 blocks, never the newest */
        const int b = (int)((uint32_t)(random_value() * 8192.0f + 32768.0f) % (BLOCKS - 1));
        if (mv[b] == 0.0f) { mv[b] = -INFINITY; dropped++; }
    }
    ds4_gpu_tensor *keys = upload(v, (size_t)N * 128 * 4), *q = upload(qv, 4096 * 4), *w = upload(wv, 32 * 4);
    ds4_gpu_tensor *mask = upload(mv, BLOCKS * 4), *view = upload(NULL, (size_t)N * 128 * 2);
    ds4_gpu_tensor *scores = upload(NULL, N * 4), *list = upload(NULL, 2048 * 4);
    ds4_gpu_tensor *want = upload(NULL, K * 4), *got = upload(NULL, K * 4);
    CHECK(keys && q && w && mask && view && scores && list && want && got);
    CHECK(ds4_gpu_dsv41_index_keys_bf16(view, keys, 0, N, NULL));
    CHECK(ds4_gpu_dsv41_indexer_score_masked(scores, q, w, keys, mask, N) &&
          ds4_gpu_dsv41_indexer_topk_tensor(want, scores, N, 1, K));
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    const uint64_t built = ds4_gpu_test_kernel_runs("kernel_dsv41_candidate_list");
    const uint64_t picked = ds4_gpu_test_kernel_runs("kernel_topk_select_one");
    const ds4_gpu_tensor *views[3] = {NULL, view, view};
    for (int c = 0; c < 3; c++) {
        CHECK(ds4_gpu_dsv41_indexer_candidates_tensor(got, scores, list, q, w, keys, views[c], mask, N, K, c == 2));
        CHECK(!memcmp(ds4_gpu_tensor_contents(want), ds4_gpu_tensor_contents(got), K * 4));
    }
    CHECK(ds4_gpu_test_kernel_runs("kernel_dsv41_candidate_list") == built + 2 &&
          ds4_gpu_test_kernel_runs("kernel_topk_select_one") == picked + 3);
    ds4_gpu_test_set_v41_off(DS4_GPU_V41_INDEX_CANDIDATES);
    CHECK(!ds4_gpu_dsv41_indexer_candidates_tensor(got, scores, list, q, w, keys, view, mask, N, K, 1));
    ds4_gpu_test_set_v41_off(0);
    ds4_gpu_test_set_flags(0);
    ds4_gpu_tensor *tensors[] = {keys, q, w, mask, view, scores, list, want, got};
    for (size_t i = 0; i < sizeof(tensors) / sizeof(*tensors); i++) ds4_gpu_tensor_free(tensors[i]);
    free(v);
    fprintf(stderr, "V4.1 index candidates: candidate-only scores and pick == masked wide scorer and radix select\n");
    return 1;
}

/* DECODE_ATTENTION: one decode row (64 heads, 128 raw rows from a wrapped start, 512 of 4,096 compressed rows)
 * with the rows staged straight from the cache against the gather and the attention after it, bit for bit. */
static int check_decode_attention(void) {
    enum { D = 512, H = 64, R = 128, C = 4096, K = 512 };
    float *sinks = NULL;
    CHECK(posix_memalign((void **)&sinks, getpagesize(), getpagesize()) == 0);
    for (int h = 0; h < H; h++) sinks[h] = random_value();
    CHECK(ds4_gpu_set_model_map(sinks, getpagesize()));
    ds4_gpu_set_quality(false);
    float *v = malloc(((size_t)H * D + (size_t)R * D + (size_t)C * D) * sizeof(float));
    int32_t *ids = malloc(K * sizeof(int32_t));
    CHECK(v && ids);
    for (size_t i = 0; i < (size_t)H * D + (size_t)R * D + (size_t)C * D; i++) v[i] = bf16(random_value() / 4);
    for (int j = 0; j < K; j++) ids[j] = (int32_t)((j * 1031u + 7u) % C);
    ds4_gpu_tensor *q = upload(v, H * D * 4), *raw = upload(v + H * D, R * D * 4);
    ds4_gpu_tensor *cache = upload(v + H * D + R * D, (size_t)C * D * 4), *it = upload(ids, K * 4);
    ds4_gpu_tensor *picked = upload(NULL, K * D * 4), *want = upload(NULL, H * D * 4), *got = upload(NULL, H * D * 4);
    CHECK(q && raw && cache && it && picked && want && got);
    CHECK(ds4_gpu_dsv41_gather_kv(picked, cache, it, C, K) &&
          ds4_gpu_attention_decode_heads_tensor(want, sinks, getpagesize(), 0, q, raw, R, R, 37, picked, 0, K,
                                                NULL, 0, H, D));
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    const uint64_t staged = ds4_gpu_test_kernel_runs("kernel_dsv41_flash_kv_stage");
    CHECK(ds4_gpu_dsv41_attention_decode(got, sinks, getpagesize(), 0, q, raw, R, R, 37, cache, it, C, K, H, D));
    CHECK(ds4_gpu_test_kernel_runs("kernel_dsv41_flash_kv_stage") == staged + 1);
    ds4_gpu_test_set_flags(0);
    CHECK(same_bits(want, got, H * D));
    ds4_gpu_test_set_v41_off(DS4_GPU_V41_DECODE_ATTENTION);
    CHECK(!ds4_gpu_dsv41_attention_decode(got, sinks, getpagesize(), 0, q, raw, R, R, 37, cache, it, C, K, H, D));
    ds4_gpu_test_set_v41_off(0);
    ds4_gpu_tensor *tensors[] = {q, raw, cache, it, picked, want, got};
    for (size_t i = 0; i < sizeof(tensors) / sizeof(*tensors); i++) ds4_gpu_tensor_free(tensors[i]);
    free(ids);
    free(v);
    free(sinks);
    fprintf(stderr, "V4.1 decode attention: rows staged from the cache == gather, then attention\n");
    return 1;
}

/* FIXED_SHAPES: each V4.1-shape kernel against the general kernel it replaces (the bit off), bit for bit: the routed
 * MoE on the fixed-route kernels and the literal pair, the four short-K projections, the q_b projection with RoPE, the
 * q_a/kv pair, the shared down and output-B expands, the FFN join, the low projection and the concurrent shared down. */
static int check_fixed_shapes(void) {
    enum { D = 5120, FF = 2304, NE = 384, NU = 6, MXFP4 = 39 /* the tensor type */ };
    const uint64_t mx_gate_row = D / 32 * 17, mx_down_row = FF / 32 * 17;
    const uint64_t q8 = 5120ull * 256 * 34;   /* the widest Q8_0 matrix here (8192 -> 5120, also 1280 -> 32768) */
    const uint64_t eg_o = q8, eu_o = eg_o + (uint64_t)NE * FF * mx_gate_row,
                   ed_o = eu_o + (uint64_t)NE * FF * mx_gate_row, wb = ed_o + (uint64_t)NE * D * mx_down_row;
    uint8_t *map = NULL;
    CHECK(posix_memalign((void **)&map, getpagesize(), wb) == 0);
    for (uint64_t b = 0; b < q8 / 34; b++) {   /* Q8_0 blocks: f16 scale, 32 int8 */
        const uint16_t d = (uint16_t)(0x2000u + random_bits() % 0x1800u);
        memcpy(map + b * 34, &d, 2);
        for (int j = 0; j < 32; j++) map[b * 34 + 2 + j] = (uint8_t)random_bits();
    }
    for (uint64_t b = eg_o; b < wb; b += 17) {   /* MXFP4 blocks: e8m0 scale, 16 bytes of nibbles */
        map[b] = (uint8_t)(118u + random_bits() % 8u);
        for (int j = 1; j < 17; j++) map[b + j] = (uint8_t)random_bits();
    }
    CHECK(ds4_gpu_set_model_map(map, wb));
    enum { X = 8 * 4096 };
    ds4_gpu_tensor *x = upload(NULL, X * 4), *add = upload(NULL, D * 4), *res = upload(NULL, 4 * D * 4);
    ds4_gpu_tensor *split = upload(NULL, 24 * 4), *out = upload(NULL, 32768 * 4), *out2 = upload(NULL, 512 * 4);
    ds4_gpu_tensor *gate = upload(NULL, NU * FF * 4), *up = upload(NULL, NU * FF * 4), *emid = upload(NULL, NU * FF * 4);
    ds4_gpu_tensor *experts = upload(NULL, NU * D * 4), *sel = upload(NULL, NU * 4), *weights = upload(NULL, NU * 4);
    ds4_gpu_tensor *sg = upload(NULL, FF * 4), *su = upload(NULL, FF * 4), *mid = upload(NULL, FF * 4);
    ds4_gpu_tensor *routed = upload(NULL, D * 4);
    float *want = malloc(32768 * 4);
    CHECK(x && add && res && split && out && out2 && gate && up && emid && experts && sel && weights && sg && su &&
          mid && routed && want);
    float *xv = ds4_gpu_tensor_contents(x), *av = ds4_gpu_tensor_contents(add), *rv = ds4_gpu_tensor_contents(res);
    for (int i = 0; i < X; i++) xv[i] = bf16(random_value());
    for (int i = 0; i < D; i++) av[i] = bf16(random_value());
    for (int i = 0; i < 4 * D; i++) rv[i] = bf16(random_value());
    for (int i = 0; i < 24; i++) ((float *)ds4_gpu_tensor_contents(split))[i] = fabsf(random_value()) / 4;
    for (int i = 0; i < NU; i++) {
        ((int32_t *)ds4_gpu_tensor_contents(sel))[i] = (int32_t)(i * 61 + 7);
        ((float *)ds4_gpu_tensor_contents(weights))[i] = fabsf(random_value()) / 4;
    }
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    /* the call with the bit off, then on: the same bits, and the fixed-shape kernel ran once */
#define FIXED_CASE(kernel, t, n, call) do { \
        ds4_gpu_test_set_v41_off(DS4_GPU_V41_FIXED_SHAPES); \
        CHECK(call); \
        memcpy(want, ds4_gpu_tensor_contents(t), (size_t)(n) * 4); \
        ds4_gpu_test_set_v41_off(0); \
        const uint64_t before = ds4_gpu_test_kernel_runs(kernel); \
        CHECK(call); \
        CHECK(ds4_gpu_test_kernel_runs(kernel) == before + 1 && !memcmp(want, ds4_gpu_tensor_contents(t), (size_t)(n) * 4)); \
    } while (0)
    FIXED_CASE("kernel_mul_mv_id_mxfp4_pair_swiglu_fixed_route_static_v41_f32", routed, D,
        ds4_gpu_routed_moe_one_tensor(routed, gate, up, emid, experts, map, wb, eg_o, eu_o, ed_o, MXFP4, MXFP4,
            mx_gate_row * FF, mx_gate_row, mx_down_row * D, mx_down_row, D, FF, D, sel, weights, NE, NU, 10.0f, x,
            NULL, 3, 1));
    FIXED_CASE("kernel_mul_mv_q8_0_short40_bf16", out, 32768, ds4_gpu_matmul_q8_0_bf16_tensor(out, map, wb, 0, 1280, 32768, x));
    FIXED_CASE("kernel_mul_mv_q8_0_short40_bf16io", out, 32768, ds4_gpu_matmul_q8_0_bf16io_tensor(out, map, wb, 0, 1280, 32768, x));
    FIXED_CASE("kernel_mul_mv_q8_0_short72_bf16", out, D, ds4_gpu_matmul_q8_0_bf16_tensor(out, map, wb, 0, FF, D, x));
    FIXED_CASE("kernel_mul_mv_q8_0_short72_bf16io", out, D, ds4_gpu_matmul_q8_0_bf16io_tensor(out, map, wb, 0, FF, D, x));
    FIXED_CASE("kernel_mul_mv_q8_0_short40_bf16_rope", out, 32768,
        ds4_gpu_dsv41_project_q(out, map, wb, 0, 1280, 32768, x, 1000, true));
    FIXED_CASE("kernel_mul_mv_q8_0_f32_bf16_pair_qa_kv", out, 1280,
        ds4_gpu_dsv41_project_pair_q8(out, out2, map, wb, 0, 34 * 160, D, 1280, 512, x));
    CHECK(ds4_gpu_dsv41_project_pair_q8(out, out2, map, wb, 0, 34 * 160, D, 1280, 512, x));
    memcpy(want, ds4_gpu_tensor_contents(out2), 512 * 4);
    ds4_gpu_test_set_v41_off(DS4_GPU_V41_FIXED_SHAPES);
    CHECK(ds4_gpu_dsv41_project_pair_q8(out, out2, map, wb, 0, 34 * 160, D, 1280, 512, x));
    ds4_gpu_test_set_v41_off(0);
    CHECK(!memcmp(want, ds4_gpu_tensor_contents(out2), 512 * 4));   /* the second matrix's rows too */
    FIXED_CASE("kernel_mul_mv_q8_0_short72_bf16io_hc_expand4", out, 4 * D,
        ds4_gpu_dsv41_matmul_expand(out, map, wb, 0, FF, D, x, add, res, split, 4));
    FIXED_CASE("kernel_mul_mv_q8_0_f32_bf16io_hc_expand4_static", out, 4 * D,
        ds4_gpu_dsv41_matmul_expand(out, map, wb, 0, 8192, D, x, NULL, res, split, 4));
    ds4_gpu_tensor *hc_out = ds4_gpu_tensor_view(out, 0, 4 * D * 4);   /* a whole-token output for the join */
    CHECK(hc_out);
    FIXED_CASE("kernel_dsv4_hc_expand4_serial_add_bf16", hc_out, 4 * D,
        ds4_gpu_hc_expand_add_split_bf16_tensor(hc_out, x, add, res, split, D, 4));
    ds4_gpu_tensor_free(hc_out);
    FIXED_CASE("kernel_dsv4_attn_out_low_q8_0_f32_static", out, 8 * 1024,
        ds4_gpu_dsv41_attention_low_static(out, map, wb, 0, 4096, 1024, 8, x) ||
        ds4_gpu_attention_output_low_q8_tensor(out, map, wb, 0, 4096, 1024, 8, x));
    FIXED_CASE("kernel_mul_mv_q8_0_short72_bf16io", out, D,
        ds4_gpu_begin_commands() &&
        ds4_gpu_parallel_ffn_start_bf16(sg, su, mid, out, map, wb, 0, 0, 0, D, FF, x, 10.0f) &&
        ds4_gpu_routed_moe_one_tensor(routed, gate, up, emid, experts, map, wb, eg_o, eu_o, ed_o, MXFP4, MXFP4,
            mx_gate_row * FF, mx_gate_row, mx_down_row * D, mx_down_row, D, FF, D, sel, weights, NE, NU, 10.0f, x,
            NULL, 3, 1) &&
        ds4_gpu_parallel_ffn_finish() && ds4_gpu_end_commands() && ds4_gpu_synchronize());
#undef FIXED_CASE
    ds4_gpu_test_set_flags(0);
    ds4_gpu_tensor *tensors[] = {x, add, res, split, out, out2, gate, up, emid, experts, sel, weights, sg, su, mid, routed};
    for (size_t i = 0; i < sizeof(tensors) / sizeof(*tensors); i++) ds4_gpu_tensor_free(tensors[i]);
    free(want);
    free(map);
    fprintf(stderr, "V4.1 fixed shapes: each fixed-shape kernel == the general kernel it replaces\n");
    return 1;
}

/* SAMPLER_SCAN: full-vocabulary sampling picks the same token, and leaves the same random state, with the NEON scan
 * as with the scalar loop: random logits, non-finite values, signed zeros everywhere, the maximum in the tail, no
 * finite value, tied maxima in different lanes, and short vocabularies. */
static int check_sampler_scan(void) {
    enum { N = 129280 + 7 };
    float *logits = malloc(N * sizeof(float));
    CHECK(logits);
    const float params[4][3] = {{0.01f, 0.5f, 0.0f}, {0.7f, 1.0f, 0.0f}, {1.0f, 0.9f, 0.0f}, {0.5f, 0.95f, 0.05f}};
    for (int c = 0; c < 8; c++) {
        const uint32_t n = c < 6 ? N : 16u * c + 5u;
        for (uint32_t i = 0; i < n; i++) logits[i] = random_value() * 8;
        if (c == 1) {
            for (uint32_t i = 0; i < n; i += 97) logits[i] = NAN;
            logits[5] = INFINITY; logits[9] = -INFINITY;
        }
        if (c == 2) for (uint32_t i = 0; i < n; i++) logits[i] = i % 3 ? -0.0f : 0.0f;
        if (c == 3 || c == 7) logits[n - 1] = 100;
        if (c == 4) for (uint32_t i = 0; i < n; i++) logits[i] = -INFINITY;
        if (c == 5) logits[7] = logits[40] = logits[1000] = 50;
        for (int k = 0; k < 4; k++)
            for (uint64_t s = 1; s <= 16; s++) {
                uint64_t r0 = s * 0x9e3779b97f4a7c15ull, r1 = r0;
                ds4_gpu_test_set_v41_off(DS4_GPU_V41_SAMPLER_SCAN);
                const int a = ds4_sample_logits(logits, (int)n, params[k][0], 0, params[k][1], params[k][2], &r0);
                ds4_gpu_test_set_v41_off(0);
                const int b = ds4_sample_logits(logits, (int)n, params[k][0], 0, params[k][1], params[k][2], &r1);
                CHECK(a == b && r0 == r1);
            }
    }
    free(logits);
    fprintf(stderr, "V4.1 sampler scan: NEON first pass == scalar loop\n");
    return 1;
}

/* ROW_MATMULS: Q8_0 and F16 matmuls of 2-8 rows with the switch off (base's per-row grid) and on (one weight pass for
 * all rows), the same bits; the bf16 Q8_0 rows == the rows then the rounding pass. */
static int check_row_matmuls(void) {
    enum { Q_IN = 4096, Q_OUT = 1280, H_IN = 20480, H_OUT = 24, ROWS = 8 };
    uint64_t q_bytes = 0;
    uint8_t *q8 = q8_weights(Q_OUT, Q_IN, &q_bytes);
    CHECK(q8);
    const uint64_t h_off = q_bytes, bytes = h_off + (uint64_t)H_IN * H_OUT * 2;
    uint8_t *map = NULL;
    CHECK(posix_memalign((void **)&map, getpagesize(), bytes) == 0);
    memcpy(map, q8, q_bytes);
    free(q8);
    uint16_t *h = (uint16_t *)(map + h_off);
    for (uint32_t i = 0; i < H_IN * H_OUT; i++)
        h[i] = (uint16_t)((random_bits() & 0x8000u) | (0x2000u + random_bits() % 0x1800u));
    CHECK(ds4_gpu_set_model_map(map, bytes));
    ds4_gpu_tensor *x = upload(NULL, ROWS * H_IN * 4), *ref = upload(NULL, ROWS * Q_OUT * 4),
                   *got = upload(NULL, ROWS * Q_OUT * 4);
    CHECK(x && ref && got);
    float *xv = ds4_gpu_tensor_contents(x);
    for (int i = 0; i < ROWS * H_IN; i++) xv[i] = random_value();
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    const char *names[3] = {"kernel_mul_mv_q8_0_f32_rows_seq4", "kernel_mul_mv_q8_0_f32_bf16_rows_seq4",
                            "kernel_mul_mv_f16_f32_4_rows"};
    uint64_t runs[3];
    for (int k = 0; k < 3; k++) runs[k] = ds4_gpu_test_kernel_runs(names[k]);
    for (uint32_t rows = 2; rows <= ROWS; rows++) {
        ds4_gpu_test_set_v41_off(DS4_GPU_V41_ROW_MATMULS);
        CHECK(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(ref, map, bytes, 0, Q_IN, Q_OUT, x, rows));
        CHECK(!ds4_gpu_dsv41_matmul_q8_0_bf16_rows(got, map, bytes, 0, Q_IN, Q_OUT, x, rows));
        ds4_gpu_test_set_v41_off(DS4_GPU_V41_ROWS_MMA | DS4_GPU_V41_SHORT_MMA);   /* the row kernels at eight rows too */
        CHECK(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(got, map, bytes, 0, Q_IN, Q_OUT, x, rows));
        CHECK(same_bits(ref, got, rows * Q_OUT));
        CHECK(ds4_gpu_dsv41_quantize(ref, Q_OUT, rows, DS4_V41_BF16) &&
              ds4_gpu_dsv41_matmul_q8_0_bf16_rows(got, map, bytes, 0, Q_IN, Q_OUT, x, rows));
        CHECK(same_bits(ref, got, rows * Q_OUT));
        ds4_gpu_test_set_v41_off(DS4_GPU_V41_ROW_MATMULS);
        CHECK(ds4_gpu_dsv41_projection_rows(ref, map, bytes, h_off, H_IN, H_OUT, rows, x));
        ds4_gpu_test_set_v41_off(DS4_GPU_V41_ROWS_MMA);
        CHECK(ds4_gpu_dsv41_projection_rows(got, map, bytes, h_off, H_IN, H_OUT, rows, x));
        CHECK(same_bits(ref, got, rows * H_OUT));
    }
    ds4_gpu_test_set_v41_off(0);
    CHECK(ds4_gpu_test_kernel_runs(names[0]) == runs[0] + 7 && ds4_gpu_test_kernel_runs(names[1]) == runs[1] + 7 &&
          ds4_gpu_test_kernel_runs(names[2]) == runs[2] + 7);
    ds4_gpu_test_set_flags(0);
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(ref); ds4_gpu_tensor_free(got);
    /* map stays: the model views may wrap it */
    fprintf(stderr, "V4.1 row matmuls: Q8_0 and F16 rows in one weight pass == per-row grid\n");
    return 1;
}

/* ROWS_MMA: a DSpark block's Q8_0 matmuls of 3-8 rows on simdgroup matrices against the row kernels. The sums run in
 * another order, so the outputs agree to float rounding (and to one bf16 step when rounded), not bit for bit. Any other
 * batch of 3-8 rows, such as a short prefill, keeps the row kernels and their bits. */
static int check_rows_mma(void) {
    enum { ROWS = 8 };
    const uint32_t shapes[3][2] = {{1024, 512}, {4096, 1280}, {8192, 1024}};
    uint64_t bytes = 0;
    uint8_t *map = q8_weights(2560, 8192, &bytes);   /* also the shared expert's two 5120 -> 1280 matrices */
    CHECK(map);
    CHECK(ds4_gpu_set_model_map(map, bytes));
    ds4_gpu_tensor *x = upload(NULL, ROWS * 8192 * 4), *ref = upload(NULL, ROWS * 1280 * 4),
                   *got = upload(NULL, ROWS * 1280 * 4);
    CHECK(x && ref && got);
    float *xv = ds4_gpu_tensor_contents(x);
    for (int i = 0; i < ROWS * 8192; i++) xv[i] = random_value();
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    const char *names[4] = {"kernel_mul_mv_q8_0_f32_mma_rows", "kernel_mul_mv_q8_0_f32_mma_rows8",
                            "kernel_mul_mv_q8_0_f32_bf16_mma_rows", "kernel_mul_mv_q8_0_f32_bf16_mma_rows8"};
    uint64_t runs[4];
    for (int k = 0; k < 4; k++) runs[k] = ds4_gpu_test_kernel_runs(names[k]);
    ds4_gpu_dsv41_dspark_rows(1);
    for (int s = 0; s < 3; s++)
        for (uint32_t rows = 3; rows <= ROWS; rows++)
            for (int round = 0; round < 2; round++) {
                const uint32_t in = shapes[s][0], out = shapes[s][1];
                ds4_gpu_test_set_v41_off(DS4_GPU_V41_ROWS_MMA);
                CHECK(round ? ds4_gpu_dsv41_matmul_q8_0_bf16_rows(ref, map, bytes, 0, in, out, x, rows) :
                              ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(ref, map, bytes, 0, in, out, x, rows));
                ds4_gpu_test_set_v41_off(0);
                CHECK(round ? ds4_gpu_dsv41_matmul_q8_0_bf16_rows(got, map, bytes, 0, in, out, x, rows) :
                              ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(got, map, bytes, 0, in, out, x, rows));
                CHECK(ds4_gpu_synchronize());
                const float *a = ds4_gpu_tensor_contents(got), *b = ds4_gpu_tensor_contents(ref);
                float scale = 0;
                for (uint32_t i = 0; i < rows * out; i++) scale = fmaxf(scale, fabsf(b[i]));
                for (uint32_t i = 0; i < rows * out; i++)
                    CHECK(fabsf(a[i] - b[i]) <= 1e-5f * scale + (round ? fabsf(b[i]) / 128.0f : 0.0f));
            }
    /* the shared expert's gate, up and SwiGLU (5120 -> 1280 here) for 3-8 rows */
    const uint64_t sw_runs = ds4_gpu_test_kernel_runs("kernel_dsv4_shared_mid_swiglu_q8_0_bf16_mma_rows8");
    const uint64_t up_off = 1280ull * 5120 / 32 * 34;
    for (uint32_t rows = 3; rows <= ROWS; rows++) {
        ds4_gpu_test_set_v41_off(DS4_GPU_V41_ROWS_MMA);
        CHECK(ds4_gpu_dsv41_shared_swiglu(ref, map, bytes, 0, up_off, 5120, 1280, x, 10.0f, rows));
        ds4_gpu_test_set_v41_off(0);
        CHECK(ds4_gpu_dsv41_shared_swiglu(got, map, bytes, 0, up_off, 5120, 1280, x, 10.0f, rows));
        CHECK(ds4_gpu_synchronize());
        const float *a = ds4_gpu_tensor_contents(got), *b = ds4_gpu_tensor_contents(ref);
        float scale = 0;
        for (uint32_t i = 0; i < rows * 1280; i++) scale = fmaxf(scale, fabsf(b[i]));
        for (uint32_t i = 0; i < rows * 1280; i++) CHECK(fabsf(a[i] - b[i]) <= 1e-5f * scale + fabsf(b[i]) / 64.0f);
    }
    CHECK(ds4_gpu_test_kernel_runs("kernel_dsv4_shared_mid_swiglu_q8_0_bf16_mma_rows8") == sw_runs + 6);
    /* 1024: 3-8 rows on four simdgroups; 4096: 3-8 rows on eight; 8192: 4-8 rows on eight */
    CHECK(ds4_gpu_test_kernel_runs(names[0]) == runs[0] + 6 && ds4_gpu_test_kernel_runs(names[1]) == runs[1] + 11 &&
          ds4_gpu_test_kernel_runs(names[2]) == runs[2] + 6 && ds4_gpu_test_kernel_runs(names[3]) == runs[3] + 11);
    /* outside a DSpark block the group is on but 3-8 rows keep the row kernels, bit for bit */
    ds4_gpu_dsv41_dspark_rows(0);
    for (int k = 0; k < 4; k++) runs[k] = ds4_gpu_test_kernel_runs(names[k]);
    const uint64_t sw_plain = ds4_gpu_test_kernel_runs("kernel_dsv4_shared_mid_swiglu_q8_0_bf16_mma_rows8");
    for (int s = 0; s < 3; s++)
        for (uint32_t rows = 3; rows <= ROWS; rows++)
            for (int round = 0; round < 2; round++) {
                const uint32_t in = shapes[s][0], out = shapes[s][1];
                ds4_gpu_test_set_v41_off(DS4_GPU_V41_ROWS_MMA);
                CHECK(round ? ds4_gpu_dsv41_matmul_q8_0_bf16_rows(ref, map, bytes, 0, in, out, x, rows) :
                              ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(ref, map, bytes, 0, in, out, x, rows));
                ds4_gpu_test_set_v41_off(0);
                CHECK(round ? ds4_gpu_dsv41_matmul_q8_0_bf16_rows(got, map, bytes, 0, in, out, x, rows) :
                              ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(got, map, bytes, 0, in, out, x, rows));
                CHECK(ds4_gpu_synchronize() && same_bits(ref, got, rows * out));
            }
    for (uint32_t rows = 3; rows <= ROWS; rows++) {
        ds4_gpu_test_set_v41_off(DS4_GPU_V41_ROWS_MMA);
        CHECK(ds4_gpu_dsv41_shared_swiglu(ref, map, bytes, 0, up_off, 5120, 1280, x, 10.0f, rows));
        ds4_gpu_test_set_v41_off(0);
        CHECK(ds4_gpu_dsv41_shared_swiglu(got, map, bytes, 0, up_off, 5120, 1280, x, 10.0f, rows));
        CHECK(ds4_gpu_synchronize() && same_bits(ref, got, rows * 1280));
    }
    for (int k = 0; k < 4; k++) CHECK(ds4_gpu_test_kernel_runs(names[k]) == runs[k]);
    CHECK(ds4_gpu_test_kernel_runs("kernel_dsv4_shared_mid_swiglu_q8_0_bf16_mma_rows8") == sw_plain);
    ds4_gpu_test_set_flags(0);
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(ref); ds4_gpu_tensor_free(got);
    /* map stays: the model views may wrap it */
    fprintf(stderr, "V4.1 rows MMA: DSpark rows on simdgroup matrices == row kernels to float rounding; "
            "other 3-8-row batches keep the row kernels' bits\n");
    return 1;
}

/* VERIFY_KERNELS: a DSpark verify block's 6-8 rows of V4.1's MXFP4 routed experts through the fused decode kernels give
 * every row the bits of that row decoded alone (the one-token path); with the switch off, or outside a DSpark block
 * (a short prefill), they take the general batch kernels. */
static int check_verify_kernels(void) {
    enum { D = 5120, FF = 2304, NE = 384, NU = 6, ROWS = 8, MXFP4 = 39 /* the tensor type */ };
    const uint64_t gate_row = D / 32 * 17, down_row = FF / 32 * 17;
    const uint64_t eg = 0, eu = eg + (uint64_t)NE * FF * gate_row, ed = eu + (uint64_t)NE * FF * gate_row,
                   bytes = ed + (uint64_t)NE * D * down_row;
    uint8_t *map = NULL;
    CHECK(posix_memalign((void **)&map, getpagesize(), bytes) == 0);
    for (uint64_t b = 0; b < bytes; b += 17) {   /* MXFP4 blocks: e8m0 scale, 16 bytes of nibbles */
        map[b] = (uint8_t)(118u + random_bits() % 8u);
        for (int j = 1; j < 17 && b + j < bytes; j++) map[b + j] = (uint8_t)random_bits();
    }
    CHECK(ds4_gpu_set_model_map(map, bytes));
    ds4_gpu_tensor *x = upload(NULL, ROWS * D * 4), *sel = upload(NULL, ROWS * NU * 4), *w = upload(NULL, ROWS * NU * 4);
    ds4_gpu_tensor *gate = upload(NULL, ROWS * NU * FF * 4), *up = upload(NULL, ROWS * NU * FF * 4),
                   *mid = upload(NULL, ROWS * NU * FF * 4), *experts = upload(NULL, ROWS * NU * D * 4);
    ds4_gpu_tensor *ref = upload(NULL, ROWS * D * 4), *got = upload(NULL, ROWS * D * 4);
    CHECK(x && sel && w && gate && up && mid && experts && ref && got);
    float *xv = ds4_gpu_tensor_contents(x), *wv = ds4_gpu_tensor_contents(w);
    int32_t *sv = ds4_gpu_tensor_contents(sel);
    for (int i = 0; i < ROWS * D; i++) xv[i] = bf16(random_value());
    for (int r = 0; r < ROWS; r++)
        for (int k = 0; k < NU; k++) {   /* rows share some experts, as verify rows do */
            sv[r * NU + k] = (int32_t)((r * 37 + k * 61 + (k & 1) * 5) % NE);
            wv[r * NU + k] = fabsf(random_value()) / 4;
        }
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    const uint64_t runs = ds4_gpu_test_kernel_runs("kernel_mul_mv_id_mxfp4_pair_swiglu_f32");
    ds4_gpu_dsv41_dspark_rows(1);
    for (uint32_t rows = 6; rows <= ROWS; rows++) {
        for (uint32_t r = 0; r < rows; r++) {
            ds4_gpu_tensor *o = ds4_gpu_tensor_view(ref, (uint64_t)r * D * 4, D * 4);
            ds4_gpu_tensor *xr = ds4_gpu_tensor_view(x, (uint64_t)r * D * 4, D * 4);
            ds4_gpu_tensor *sr = ds4_gpu_tensor_view(sel, (uint64_t)r * NU * 4, NU * 4);
            ds4_gpu_tensor *wr = ds4_gpu_tensor_view(w, (uint64_t)r * NU * 4, NU * 4);
            CHECK(o && xr && sr && wr);
            CHECK(ds4_gpu_routed_moe_one_tensor(o, gate, up, mid, experts, map, bytes, eg, eu, ed, MXFP4, MXFP4,
                      gate_row * FF, gate_row, down_row * D, down_row, D, FF, D, sr, wr, NE, NU, 10.0f, xr,
                      NULL, 3, 1));
            ds4_gpu_tensor *views[] = {o, xr, sr, wr};
            for (size_t i = 0; i < sizeof(views) / sizeof(*views); i++) ds4_gpu_tensor_free(views[i]);
        }
        bool mid_f16 = false;
        CHECK(ds4_gpu_routed_moe_batch_tensor(got, gate, up, mid, experts, map, bytes, eg, eu, ed, MXFP4, MXFP4,
                  gate_row * FF, gate_row, down_row * D, down_row, D, FF, D, sel, w, NE, NU, 10.0f, x, 3, rows,
                  &mid_f16, true));
        CHECK(same_bits(ref, got, rows * D));
    }
    CHECK(ds4_gpu_test_kernel_runs("kernel_mul_mv_id_mxfp4_pair_swiglu_f32") == runs + 3);
    ds4_gpu_test_set_v41_off(DS4_GPU_V41_VERIFY_KERNELS);
    bool mid_f16 = false;
    CHECK(ds4_gpu_routed_moe_batch_tensor(got, gate, up, mid, experts, map, bytes, eg, eu, ed, MXFP4, MXFP4,
              gate_row * FF, gate_row, down_row * D, down_row, D, FF, D, sel, w, NE, NU, 10.0f, x, 3, ROWS,
              &mid_f16, true));
    ds4_gpu_test_set_v41_off(0);
    CHECK(ds4_gpu_test_kernel_runs("kernel_mul_mv_id_mxfp4_pair_swiglu_f32") == runs + 3);
    /* outside a DSpark block the group is on but 6-8 rows keep the batch kernels, bit for bit */
    ds4_gpu_dsv41_dspark_rows(0);
    ds4_gpu_tensor *plain = upload(NULL, ROWS * D * 4);
    CHECK(plain);
    for (uint32_t rows = 6; rows <= ROWS; rows++) {
        ds4_gpu_test_set_v41_off(DS4_GPU_V41_VERIFY_KERNELS);
        mid_f16 = false;
        CHECK(ds4_gpu_routed_moe_batch_tensor(plain, gate, up, mid, experts, map, bytes, eg, eu, ed, MXFP4, MXFP4,
                  gate_row * FF, gate_row, down_row * D, down_row, D, FF, D, sel, w, NE, NU, 10.0f, x, 3, rows,
                  &mid_f16, true));
        ds4_gpu_test_set_v41_off(0);
        mid_f16 = false;
        CHECK(ds4_gpu_routed_moe_batch_tensor(got, gate, up, mid, experts, map, bytes, eg, eu, ed, MXFP4, MXFP4,
                  gate_row * FF, gate_row, down_row * D, down_row, D, FF, D, sel, w, NE, NU, 10.0f, x, 3, rows,
                  &mid_f16, true));
        CHECK(ds4_gpu_synchronize() && same_bits(plain, got, rows * D));
    }
    CHECK(ds4_gpu_test_kernel_runs("kernel_mul_mv_id_mxfp4_pair_swiglu_f32") == runs + 3);
    ds4_gpu_tensor_free(plain);
    ds4_gpu_test_set_flags(0);
    ds4_gpu_tensor *tensors[] = {x, sel, w, gate, up, mid, experts, ref, got};
    for (size_t i = 0; i < sizeof(tensors) / sizeof(*tensors); i++) ds4_gpu_tensor_free(tensors[i]);
    free(map);
    fprintf(stderr, "V4.1 verify kernels: 6-8 DSpark verify rows through the decode MoE == each row decoded alone; "
            "other 6-8-row batches keep the batch kernels' bits\n");
    return 1;
}

/* PUBLIC_CONTROLLER: DSpark admission fed measured times. A request starts serial, a block needs three confident
 * leading positions, a losing window backs off for 16 and then 32 serial tokens, a small loss is carried, and a
 * winning window clears the back-off. */
static int check_public_controller(void) {
    ds41_admission a;
    ds41_admit_begin(&a);
    for (uint32_t i = 0; i < DS41_ADMIT_FIRST_SERIAL; i++) {
        CHECK(!ds41_admit_draft(&a));
        ds41_admit_serial(&a, i == 3 ? 90.0 : 25.0);   /* one slow token does not move the median */
    }
    CHECK(ds41_admit_draft(&a) && a.serial_ms == 25.0);
    CHECK(ds41_admit_confident(0, 0.75f) && !ds41_admit_confident(2, 0.7f) && ds41_admit_confident(3, 0.1f));
    for (int i = 0; i < 3; i++) ds41_admit_cycle(&a, 60.0, 4, true);    /* 4 tokens for 60 ms, not 100 */
    CHECK(ds41_admit_draft(&a));
    for (int i = 0; i < 3; i++) ds41_admit_cycle(&a, 30.0, 1, false);   /* declines: drafter + one token */
    CHECK(!ds41_admit_draft(&a) && a.skip == 16);
    for (int i = 0; i < 16; i++) {
        CHECK(!ds41_admit_draft(&a));
        ds41_admit_serial(&a, 25.0);
    }
    CHECK(ds41_admit_draft(&a));
    for (int i = 0; i < 3; i++) ds41_admit_cycle(&a, 60.0, 2, true);
    CHECK(a.skip == 32);
    for (int i = 0; i < 32; i++) ds41_admit_serial(&a, 25.0);
    for (int i = 0; i < 3; i++) ds41_admit_cycle(&a, 51.0, 2, true);    /* 3 ms over, within half a token */
    CHECK(ds41_admit_draft(&a) && a.backoffs == 2);
    for (int i = 0; i < 3; i++) ds41_admit_cycle(&a, 60.0, 4, true);
    CHECK(ds41_admit_draft(&a) && a.backoffs == 0);
    fprintf(stderr, "V4.1 DSpark admission: serial start, confident blocks, back-off and recovery by measured cost\n");
    return 1;
}

/* PREFILL_INDEX: index scores of 8 to 40 rows, early and late in the context, ratios 1 and 2, on simdgroup matrices
 * against the tiled kernel, bit for bit. */
static int check_prefill_index(void) {
    enum { KEYS = 1025, ROWS = 40, HEADS = 32, DIM = 128 };
    ds4_gpu_tensor *qt = upload(NULL, (size_t)ROWS * HEADS * DIM * 4), *kt = upload(NULL, (size_t)KEYS * DIM * 4);
    ds4_gpu_tensor *wt = upload(NULL, (size_t)ROWS * HEADS * 4);
    ds4_gpu_tensor *ref = upload(NULL, (size_t)ROWS * KEYS * 4), *got = upload(NULL, (size_t)ROWS * KEYS * 4);
    CHECK(qt && kt && wt && ref && got);
    float *q = ds4_gpu_tensor_contents(qt), *k = ds4_gpu_tensor_contents(kt), *w = ds4_gpu_tensor_contents(wt);
    for (size_t i = 0; i < (size_t)ROWS * HEADS * DIM; i++) q[i] = random_value();
    for (size_t i = 0; i < (size_t)KEYS * DIM; i++) k[i] = random_value();
    for (size_t i = 0; i < (size_t)ROWS * HEADS; i++) w[i] = bf16(random_value());
    CHECK(ds4_gpu_dsv41_quantize(qt, DIM, ROWS * HEADS, DS4_V41_FP4_E8M0));
    CHECK(ds4_gpu_dsv41_quantize(kt, DIM, KEYS, DS4_V41_FP4_E8M0));
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    const uint64_t runs = ds4_gpu_test_kernel_runs("kernel_dsv41_indexer_scores_mma");
    const uint32_t counts[] = {8, 9, 33, 40};
    for (uint32_t ratio = 1; ratio <= 2; ratio++)
        for (uint32_t early = 0; early <= 1; early++)
            for (size_t ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
                const uint32_t rows = counts[ci], start = early ? 0 : KEYS * ratio - rows;
                ds4_gpu_test_set_v41_off(DS4_GPU_V41_PREFILL_INDEX);
                CHECK(ds4_gpu_dsv41_indexer_scores_batch(ref, qt, wt, kt, KEYS, rows, start, ratio));
                ds4_gpu_test_set_v41_off(0);
                CHECK(ds4_gpu_dsv41_indexer_scores_batch(got, qt, wt, kt, KEYS, rows, start, ratio));
                CHECK(ds4_gpu_synchronize());
                CHECK(same_bits(ref, got, (uint64_t)rows * KEYS));
            }
    CHECK(ds4_gpu_test_kernel_runs("kernel_dsv41_indexer_scores_mma") == runs + 16);
    ds4_gpu_test_set_flags(0);
    ds4_gpu_tensor *tensors[] = {qt, kt, wt, ref, got};
    for (size_t i = 0; i < sizeof(tensors) / sizeof(*tensors); i++) ds4_gpu_tensor_free(tensors[i]);
    fprintf(stderr, "V4.1 prefill index: 8-40 rows on simdgroup matrices == the tiled kernel\n");
    return 1;
}

/* PREFILL_ATTENTION: 9 to 129 prefill rows with ratios 1 and 2, every or only some top-k entries selected, staged
 * sixteen rows per barrier against the eight-head kernel, bit for bit. */
static int check_prefill_attention(void) {
    enum { D = 512, H = 64, K = 512, C = 2048 };
    float *sinks = NULL;
    CHECK(posix_memalign((void **)&sinks, getpagesize(), getpagesize()) == 0);
    for (int h = 0; h < H; h++) sinks[h] = random_value();
    CHECK(ds4_gpu_set_model_map(sinks, getpagesize()));
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    const uint64_t runs = ds4_gpu_test_kernel_runs("kernel_dsv41_prefill_attention_rb16");
    const uint64_t mma_runs = ds4_gpu_test_kernel_runs("kernel_dsv41_prefill_attention_mma");
    const uint32_t counts[] = {9, 33, 129};
    for (uint32_t c = 0; c < 3; c++)
        for (uint32_t ratio = 1; ratio <= 2; ratio++)
            for (uint32_t sparse = 0; sparse <= 1; sparse++) {
                const uint32_t n = counts[c], nr = n + 127, start = 3073;
                const size_t nq = (size_t)n * H * D;
                ds4_gpu_tensor *qt = upload(NULL, nq * 4), *rt = upload(NULL, (size_t)nr * D * 4);
                ds4_gpu_tensor *ct = upload(NULL, (size_t)C * D * 4), *it = upload(NULL, (size_t)n * K * 4);
                ds4_gpu_tensor *ref = upload(NULL, nq * 4), *got = upload(NULL, nq * 4);
                CHECK(qt && rt && ct && it && ref && got);
                float *q = ds4_gpu_tensor_contents(qt), *raw = ds4_gpu_tensor_contents(rt);
                float *comp = ds4_gpu_tensor_contents(ct);
                int32_t *ids = ds4_gpu_tensor_contents(it);
                for (size_t i = 0; i < nq; i++) q[i] = bf16(random_value() / 4);
                for (uint32_t i = 0; i < nr * D; i++) raw[i] = bf16(random_value() / 4);
                for (uint32_t i = 0; i < C * D; i++) comp[i] = bf16(random_value() / 4);
                for (uint32_t t = 0; t < n; t++)
                    for (uint32_t j = 0; j < K; j++)
                        ids[t * K + j] = sparse && (j % 7 == 3 || j % 29 == 0) ? -1 :
                            (int32_t)((j * 127u + t * 17u) % C);
                ds4_gpu_test_set_v41_off(DS4_GPU_V41_PREFILL_ATTENTION | DS4_GPU_V41_PREFILL_MMA);
                CHECK(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(ref, sinks, getpagesize(),
                    0, qt, rt, ct, 0, it, n, start, nr, nr, 0, C, K, 128, ratio, H, D));
                ds4_gpu_test_set_v41_off(DS4_GPU_V41_PREFILL_MMA);
                CHECK(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(got, sinks, getpagesize(),
                    0, qt, rt, ct, 0, it, n, start, nr, nr, 0, C, K, 128, ratio, H, D));
                CHECK(ds4_gpu_synchronize());
                CHECK(same_bits(ref, got, nq));
                /* PREFILL_MMA: the same rows on simdgroup matrices, equal up to the reassociated sums */
                ds4_gpu_test_set_v41_off(0);
                CHECK(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(got, sinks, getpagesize(),
                    0, qt, rt, ct, 0, it, n, start, nr, nr, 0, C, K, 128, ratio, H, D));
                CHECK(ds4_gpu_synchronize());
                const float *rv = ds4_gpu_tensor_contents(ref), *gv = ds4_gpu_tensor_contents(got);
                double worst = 0;
                for (size_t i = 0; i < nq; i++) {
                    const double e = fabs((double)gv[i] - rv[i]) / (1e-3 + fabs(rv[i]));
                    if (!(e <= worst)) worst = e;
                }
                if (!(worst <= 2e-2)) fprintf(stderr, "prefill MMA: worst relative error %g (n %u ratio %u)\n",
                                              worst, n, ratio);
                CHECK(worst <= 2e-2);
                ds4_gpu_tensor *tensors[] = {qt, rt, ct, it, ref, got};
                for (size_t i = 0; i < sizeof(tensors) / sizeof(*tensors); i++) ds4_gpu_tensor_free(tensors[i]);
            }
    CHECK(ds4_gpu_test_kernel_runs("kernel_dsv41_prefill_attention_rb16") == runs + 12);
    CHECK(ds4_gpu_test_kernel_runs("kernel_dsv41_prefill_attention_mma") == mma_runs + 12);
    ds4_gpu_test_set_v41_off(0);
    ds4_gpu_test_set_flags(0);
    free(sinks);
    fprintf(stderr, "V4.1 prefill attention: 16 rows per barrier == the eight-head kernel; simdgroup matrices within "
            "reassociation\n");
    return 1;
}

/* PREFILL_FOLDS: 9 and 33 prefill rows through each folded producer against the producer and its rounding pass, bit
 * for bit: the f32 add, the RMS norm, the F16 projection of two rows per threadgroup (512, 1280 and 5120 wide), the
 * HC sums and expand, and the embedding gather repeated into the HC streams. */
static int check_prefill_folds(void) {
    enum { D = 5120, HC = 4, ROWS = 33, VOCAB = 64 };
    const uint64_t norm_off = 0, proj_off = norm_off + D * 2, embd_off = proj_off + (uint64_t)D * 1024 * 2;
    const uint64_t bytes = embd_off + (uint64_t)VOCAB * D * 2;
    _Float16 *map = NULL;
    CHECK(posix_memalign((void **)&map, getpagesize(), bytes) == 0);
    for (uint64_t i = 0; i < bytes / 2; i++) map[i] = (_Float16)random_value();
    CHECK(ds4_gpu_set_model_map(map, bytes));
    ds4_gpu_tensor *a = upload(NULL, ROWS * D * HC * 4), *b = upload(NULL, ROWS * D * HC * 4);
    ds4_gpu_tensor *ref = upload(NULL, ROWS * D * HC * 4), *got = upload(NULL, ROWS * D * HC * 4);
    ds4_gpu_tensor *pre = upload(NULL, ROWS * 24 * 4), *pre2 = upload(NULL, ROWS * HC * 4);
    ds4_gpu_tensor *tok = upload(NULL, ROWS * 4), *x = upload(NULL, ROWS * D * 4);
    CHECK(a && b && ref && got && pre && pre2 && tok && x);
    float *av = ds4_gpu_tensor_contents(a), *bv = ds4_gpu_tensor_contents(b), *pv = ds4_gpu_tensor_contents(pre);
    int32_t *tv = ds4_gpu_tensor_contents(tok);
    for (int i = 0; i < ROWS * D * HC; i++) { av[i] = bf16(random_value()); bv[i] = random_value(); }
    for (int i = 0; i < ROWS * 24; i++) pv[i] = random_value();
    for (int i = 0; i < ROWS; i++) tv[i] = (int32_t)(random_bits() % VOCAB);
    const char *names[] = {"kernel_dsv41_add_bf16_rows4", "kernel_dsv41_project_f16_rows2", "kernel_dsv41_repeat_init4"};
    uint64_t runs[3];
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    for (int i = 0; i < 3; i++) runs[i] = ds4_gpu_test_kernel_runs(names[i]);
    const uint32_t counts[] = {9, ROWS};
    for (int c = 0; c < 2; c++) {
        const uint32_t rows = counts[c];
        /* the HC kernels take their row count from the output's size */
        ds4_gpu_tensor *rv = ds4_gpu_tensor_view(ref, 0, (uint64_t)rows * D * 4);
        ds4_gpu_tensor *gv = ds4_gpu_tensor_view(got, 0, (uint64_t)rows * D * 4);
        ds4_gpu_tensor *rh = ds4_gpu_tensor_view(ref, 0, (uint64_t)rows * D * HC * 4);
        ds4_gpu_tensor *gh = ds4_gpu_tensor_view(got, 0, (uint64_t)rows * D * HC * 4);
        ds4_gpu_tensor *bv1 = ds4_gpu_tensor_view(b, 0, (uint64_t)rows * D * 4);
        CHECK(rv && gv && rh && gh && bv1);
        CHECK(ds4_gpu_add_tensor(rv, a, b, rows * D) && ds4_gpu_dsv41_quantize(rv, D, rows, DS4_V41_BF16));
        CHECK(ds4_gpu_dsv41_add_bf16_rows(got, a, b, D, rows));
        CHECK(ds4_gpu_synchronize() && same_bits(ref, got, (uint64_t)rows * D));
        CHECK(ds4_gpu_rms_norm_weight_rows_tensor(rv, a, map, bytes, norm_off, D, rows, 1e-6f) &&
              ds4_gpu_dsv41_quantize(rv, D, rows, DS4_V41_BF16));
        CHECK(ds4_gpu_dsv41_norm_bf16_rows(got, a, map, bytes, norm_off, D, rows, 1e-6f));
        CHECK(ds4_gpu_synchronize() && same_bits(ref, got, (uint64_t)rows * D));
        const uint32_t widths[] = {512, 1280, 5120}, outs[] = {1024, 512, 1024};
        for (int k = 0; k < 3; k++) {
            CHECK(ds4_gpu_dsv41_projection_rows(ref, map, bytes, proj_off, widths[k], outs[k], rows, b));
            CHECK(ds4_gpu_dsv41_projection_rows2(got, map, bytes, proj_off, widths[k], outs[k], rows, b));
            CHECK(ds4_gpu_synchronize() && same_bits(ref, got, (uint64_t)rows * outs[k]));
        }
        CHECK(ds4_gpu_hc_weighted_sum_split_tensor(rv, a, pre, D, HC) && ds4_gpu_dsv41_quantize(rv, D, rows, DS4_V41_BF16));
        CHECK(ds4_gpu_hc_weighted_sum_split_bf16_tensor(gv, a, pre, D, HC));
        CHECK(ds4_gpu_synchronize() && same_bits(ref, got, (uint64_t)rows * D));
        CHECK(ds4_gpu_hc_expand_split_tensor(rh, bv1, a, pre, D, HC) &&
              ds4_gpu_dsv41_quantize(rh, D * HC, rows, DS4_V41_BF16));
        CHECK(ds4_gpu_hc_expand_split_bf16_tensor(gh, bv1, a, pre, D, HC));
        CHECK(ds4_gpu_synchronize() && same_bits(ref, got, (uint64_t)rows * D * HC));
        for (uint32_t t = 0; t < rows; t++) {
            ds4_gpu_tensor *o = ds4_gpu_tensor_view(ref, (uint64_t)t * D * HC * 4, D * HC * 4);
            CHECK(o && ds4_gpu_embed_token_hc_tensor(o, map, bytes, embd_off, VOCAB, (uint32_t)tv[t], D, HC));
            ds4_gpu_tensor_free(o);
        }
        CHECK(ds4_gpu_dsv41_embed_init_rows(got, pre2, x, tok, map, bytes, embd_off, VOCAB, rows, D));
        CHECK(ds4_gpu_synchronize() && same_bits(ref, got, (uint64_t)rows * D * HC));
        const float *p2 = ds4_gpu_tensor_contents(pre2);
        for (uint32_t t = 0; t < rows; t++)
            CHECK(p2[t * HC] == 1.0f && p2[t * HC + 1] == 0.0f && p2[t * HC + 2] == 0.0f && p2[t * HC + 3] == 0.0f);
        ds4_gpu_tensor *views[] = {rv, gv, rh, gh, bv1};
        for (size_t i = 0; i < sizeof(views) / sizeof(*views); i++) ds4_gpu_tensor_free(views[i]);
    }
    CHECK(ds4_gpu_test_kernel_runs(names[0]) == runs[0] + 2 && ds4_gpu_test_kernel_runs(names[1]) == runs[1] + 6 &&
          ds4_gpu_test_kernel_runs(names[2]) == runs[2] + 2);
    ds4_gpu_test_set_flags(0);
    ds4_gpu_tensor *tensors[] = {a, b, ref, got, pre, pre2, tok, x};
    for (size_t i = 0; i < sizeof(tensors) / sizeof(*tensors); i++) ds4_gpu_tensor_free(tensors[i]);
    free(map);
    fprintf(stderr, "V4.1 prefill folds: each folded producer == the producer and its rounding pass\n");
    return 1;
}

/* MOE_DEDUP: 2-8 MXFP4 rows sharing experts, each distinct expert read once, against the per-pair kernel, bit for
 * bit (6-8 rows take this path only in a DSpark verify block, through the verify kernels). */
static int check_moe_dedup(void) {
    enum { D = 5120, FF = 2304, NE = 384, NU = 6, ROWS = 8, MXFP4 = 39 /* the tensor type */ };
    const uint64_t gate_row = D / 32 * 17, down_row = FF / 32 * 17;
    const uint64_t eg = 0, eu = eg + (uint64_t)NE * FF * gate_row, ed = eu + (uint64_t)NE * FF * gate_row,
                   bytes = ed + (uint64_t)NE * D * down_row;
    uint8_t *map = NULL;
    CHECK(posix_memalign((void **)&map, getpagesize(), bytes) == 0);
    for (uint64_t b = 0; b < bytes; b += 17) {   /* MXFP4 blocks: e8m0 scale, 16 bytes of nibbles */
        map[b] = (uint8_t)(118u + random_bits() % 8u);
        for (int j = 1; j < 17 && b + j < bytes; j++) map[b + j] = (uint8_t)random_bits();
    }
    CHECK(ds4_gpu_set_model_map(map, bytes));
    ds4_gpu_tensor *x = upload(NULL, ROWS * D * 4), *sel = upload(NULL, ROWS * NU * 4), *w = upload(NULL, ROWS * NU * 4);
    ds4_gpu_tensor *gate = upload(NULL, ROWS * NU * FF * 4), *up = upload(NULL, ROWS * NU * FF * 4),
                   *mid = upload(NULL, ROWS * NU * FF * 4), *experts = upload(NULL, ROWS * NU * D * 4);
    ds4_gpu_tensor *ref = upload(NULL, ROWS * D * 4), *got = upload(NULL, ROWS * D * 4);
    CHECK(x && sel && w && gate && up && mid && experts && ref && got);
    float *xv = ds4_gpu_tensor_contents(x), *wv = ds4_gpu_tensor_contents(w);
    int32_t *sv = ds4_gpu_tensor_contents(sel);
    for (int i = 0; i < ROWS * D; i++) xv[i] = bf16(random_value());
    for (int r = 0; r < ROWS; r++)
        for (int k = 0; k < NU; k++) {   /* rows share some experts, as verify rows do */
            sv[r * NU + k] = (int32_t)((r * 37 + k * 61 + (k & 1) * 5) % NE);
            wv[r * NU + k] = fabsf(random_value()) / 4;
        }
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    const uint64_t runs = ds4_gpu_test_kernel_runs("kernel_mul_mv_id_mxfp4_pair_swiglu_dedup_f32");
    ds4_gpu_dsv41_dspark_rows(1);
    for (uint32_t rows = 2; rows <= ROWS; rows++) {
        bool mid_f16 = false;
        ds4_gpu_test_set_v41_off(DS4_GPU_V41_MOE_DEDUP);
        CHECK(ds4_gpu_routed_moe_batch_tensor(ref, gate, up, mid, experts, map, bytes, eg, eu, ed, MXFP4, MXFP4,
                  gate_row * FF, gate_row, down_row * D, down_row, D, FF, D, sel, w, NE, NU, 10.0f, x, 3, rows,
                  &mid_f16, true));
        ds4_gpu_test_set_v41_off(0);
        CHECK(ds4_gpu_routed_moe_batch_tensor(got, gate, up, mid, experts, map, bytes, eg, eu, ed, MXFP4, MXFP4,
                  gate_row * FF, gate_row, down_row * D, down_row, D, FF, D, sel, w, NE, NU, 10.0f, x, 3, rows,
                  &mid_f16, true));
        ds4_gpu_test_set_v41_off(0);
        CHECK(ds4_gpu_synchronize() && same_bits(ref, got, rows * D));
    }
    ds4_gpu_dsv41_dspark_rows(0);
    CHECK(ds4_gpu_test_kernel_runs("kernel_mul_mv_id_mxfp4_pair_swiglu_dedup_f32") == runs + 7);
    ds4_gpu_test_set_flags(0);
    ds4_gpu_tensor *tensors[] = {x, sel, w, gate, up, mid, experts, ref, got};
    for (size_t i = 0; i < sizeof(tensors) / sizeof(*tensors); i++) ds4_gpu_tensor_free(tensors[i]);
    free(map);
    fprintf(stderr, "V4.1 MoE dedup: 2-8 MXFP4 rows, each distinct expert read once == the per-pair kernel\n");
    return 1;
}

/* VERIFY_BATCH: 2-8 rows against each row alone, bit for bit (ROWS_MMA off; on, the matrix kernels run): the
 * projection with the HC expand (the output-B and the shared down with its add), the attention low projection,
 * and the decode attention of consecutive rows, each over its own raw window and selected rows, with eight heads per
 * threadgroup (FLASH_HEAD_SHARE) and without. */
static int check_verify_batch(void) {
    enum { D = 5120, FF = 2304, HC = 4, ROWS = 8, GROUP = 4096, RANK = 1024, GROUPS = 8 };
    uint64_t wb = 0, lb = 0;
    void *map = q8_weights(D, 8192, &wb), *low_map = q8_weights(GROUPS * RANK, GROUP, &lb);
    CHECK(map && low_map);
    ds4_gpu_tensor *x = upload(NULL, ROWS * 8192 * 4), *add = upload(NULL, ROWS * D * 4);
    ds4_gpu_tensor *res = upload(NULL, ROWS * HC * D * 4), *split = upload(NULL, ROWS * 24 * 4);
    ds4_gpu_tensor *ref = upload(NULL, ROWS * HC * D * 4), *got = upload(NULL, ROWS * HC * D * 4);
    ds4_gpu_tensor *heads = upload(NULL, ROWS * GROUPS * GROUP * 4);
    CHECK(x && add && res && split && ref && got && heads);
    float *xv = ds4_gpu_tensor_contents(x), *av = ds4_gpu_tensor_contents(add), *rv = ds4_gpu_tensor_contents(res);
    float *sv = ds4_gpu_tensor_contents(split), *hv = ds4_gpu_tensor_contents(heads);
    for (int i = 0; i < ROWS * 8192; i++) xv[i] = random_value();
    for (int i = 0; i < ROWS * D; i++) av[i] = bf16(random_value());
    for (int i = 0; i < ROWS * HC * D; i++) rv[i] = bf16(random_value());
    for (int i = 0; i < ROWS * 24; i++) sv[i] = random_value();
    for (int i = 0; i < ROWS * GROUPS * GROUP; i++) hv[i] = bf16(random_value());
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    const char *names[] = {"kernel_mul_mv_q8_0_f32_bf16io_hc_expand4_rows_seq4",
                           "kernel_mul_mv_q8_0_f32_bf16io_hc_expand4_mma_rows8", "kernel_dsv4_attn_out_low_q8_0_f32_rows",
                           "kernel_dsv4_attn_out_low_q8_0_f32_mma_rows8", "ds4_gpu_dsv41_attention_decode_rows"};
    uint64_t runs[5];
    for (int k = 0; k < 5; k++) runs[k] = ds4_gpu_test_kernel_runs(names[k]);
    ds4_gpu_dsv41_dspark_rows(1);   /* the matrix forms run only for a DSpark block's rows (ROWS_MMA) */
    for (int shape = 0; shape < 2; shape++)   /* the output-B (8192 -> 5120), the shared down with its add */
        for (uint32_t rows = 2; rows <= ROWS; rows++) {
            const uint32_t k = shape ? FF : 8192;
            CHECK(ds4_gpu_set_model_map(map, wb));
            for (uint32_t r = 0; r < rows; r++) {
                ds4_gpu_tensor *o = ds4_gpu_tensor_view(ref, (uint64_t)r * HC * D * 4, HC * D * 4);
                ds4_gpu_tensor *xr = ds4_gpu_tensor_view(x, (uint64_t)r * k * 4, k * 4);
                ds4_gpu_tensor *ar = ds4_gpu_tensor_view(add, (uint64_t)r * D * 4, D * 4);
                ds4_gpu_tensor *rr = ds4_gpu_tensor_view(res, (uint64_t)r * HC * D * 4, HC * D * 4);
                ds4_gpu_tensor *pr = ds4_gpu_tensor_view(split, (uint64_t)r * 24 * 4, 24 * 4);
                CHECK(o && xr && ar && rr && pr &&
                      ds4_gpu_dsv41_matmul_expand(o, map, wb, 0, k, D, xr, shape ? ar : NULL, rr, pr, HC));
                ds4_gpu_tensor *views[] = {o, xr, ar, rr, pr};
                for (size_t i = 0; i < sizeof(views) / sizeof(*views); i++) ds4_gpu_tensor_free(views[i]);
            }
            ds4_gpu_test_set_v41_off(DS4_GPU_V41_ROWS_MMA);
            CHECK(ds4_gpu_dsv41_matmul_expand_rows(got, map, wb, 0, k, D, x, shape ? add : NULL, res, split, HC, rows));
            ds4_gpu_test_set_v41_off(0);
            CHECK(ds4_gpu_synchronize() && same_bits(ref, got, (uint64_t)rows * HC * D));
            CHECK(ds4_gpu_dsv41_matmul_expand_rows(got, map, wb, 0, k, D, x, shape ? add : NULL, res, split, HC, rows));
            CHECK(ds4_gpu_set_model_map(low_map, lb));
            for (uint32_t r = 0; r < rows; r++) {
                ds4_gpu_tensor *o = ds4_gpu_tensor_view(ref, (uint64_t)r * GROUPS * RANK * 4, GROUPS * RANK * 4);
                ds4_gpu_tensor *h = ds4_gpu_tensor_view(heads, (uint64_t)r * GROUPS * GROUP * 4, GROUPS * GROUP * 4);
                CHECK(o && h && ds4_gpu_attention_output_low_q8_tensor(o, low_map, lb, 0, GROUP, RANK, GROUPS, h));
                ds4_gpu_tensor_free(o);
                ds4_gpu_tensor_free(h);
            }
            ds4_gpu_test_set_v41_off(DS4_GPU_V41_ROWS_MMA);
            CHECK(ds4_gpu_attention_output_low_q8_rows_exact_tensor(got, low_map, lb, 0, GROUP, RANK, GROUPS, 0, GROUPS,
                                                                    heads, rows));
            ds4_gpu_test_set_v41_off(0);
            CHECK(ds4_gpu_synchronize() && same_bits(ref, got, (uint64_t)rows * GROUPS * RANK));
            CHECK(ds4_gpu_attention_output_low_q8_rows_exact_tensor(got, low_map, lb, 0, GROUP, RANK, GROUPS, 0, GROUPS,
                                                                    heads, rows));
        }
    /* rows 2-8 of each: the row kernels with ROWS_MMA off, then the matrix kernels from four (the 8192-wide expand) or three rows (the
     * shared down on its four-simdgroup form, the low projection) */
    CHECK(ds4_gpu_test_kernel_runs(names[0]) == runs[0] + 14 + 3 && ds4_gpu_test_kernel_runs(names[1]) == runs[1] + 5);
    CHECK(ds4_gpu_test_kernel_runs(names[2]) == runs[2] + 14 + 2 && ds4_gpu_test_kernel_runs(names[3]) == runs[3] + 12);
    ds4_gpu_dsv41_dspark_rows(0);
    ds4_gpu_tensor *tensors[] = {x, add, res, split, ref, got, heads};
    for (size_t i = 0; i < sizeof(tensors) / sizeof(*tensors); i++) ds4_gpu_tensor_free(tensors[i]);
    /* the decode attention of consecutive rows: row r's 128 raw keys start at raw row r */
    enum { AD = 512, AH = 64, R = 128, C = 4096, K = 512 };
    float *sinks = NULL;
    CHECK(posix_memalign((void **)&sinks, getpagesize(), getpagesize()) == 0);
    for (int h = 0; h < AH; h++) sinks[h] = random_value();
    CHECK(ds4_gpu_set_model_map(sinks, getpagesize()));
    ds4_gpu_set_quality(false);
    const size_t nv = (size_t)ROWS * AH * AD + (size_t)(R + ROWS) * AD + (size_t)C * AD;
    float *v = malloc(nv * sizeof(float));
    int32_t *ids = malloc(ROWS * K * sizeof(int32_t));
    CHECK(v && ids);
    for (size_t i = 0; i < nv; i++) v[i] = bf16(random_value() / 4);
    for (int r = 0; r < ROWS; r++)
        for (int j = 0; j < K; j++) ids[r * K + j] = (int32_t)((j * 1031u + 7u + r * 97u) % C);
    ds4_gpu_tensor *q = upload(v, (size_t)ROWS * AH * AD * 4);
    ds4_gpu_tensor *raw = upload(v + (size_t)ROWS * AH * AD, (size_t)(R + ROWS) * AD * 4);
    ds4_gpu_tensor *cache = upload(v + (size_t)ROWS * AH * AD + (size_t)(R + ROWS) * AD, (size_t)C * AD * 4);
    ds4_gpu_tensor *it = upload(ids, ROWS * K * 4);
    ds4_gpu_tensor *want = upload(NULL, (size_t)ROWS * AH * AD * 4), *have = upload(NULL, (size_t)ROWS * AH * AD * 4);
    CHECK(q && raw && cache && it && want && have);
    for (uint32_t r = 0; r < ROWS; r++) {
        ds4_gpu_tensor *o = ds4_gpu_tensor_view(want, (uint64_t)r * AH * AD * 4, AH * AD * 4);
        ds4_gpu_tensor *qr = ds4_gpu_tensor_view(q, (uint64_t)r * AH * AD * 4, AH * AD * 4);
        ds4_gpu_tensor *ir = ds4_gpu_tensor_view(it, (uint64_t)r * K * 4, K * 4);
        CHECK(o && qr && ir && ds4_gpu_dsv41_attention_decode(o, sinks, getpagesize(), 0, qr, raw, R, R + ROWS, r,
                                                               cache, ir, C, K, AH, AD));
        ds4_gpu_tensor_free(o); ds4_gpu_tensor_free(qr); ds4_gpu_tensor_free(ir);
    }
    for (uint32_t rows = 2; rows <= ROWS; rows++) {
        CHECK(ds4_gpu_dsv41_attention_decode_rows(have, sinks, getpagesize(), 0, q, raw, R, R + ROWS, 0, cache, it, K,
                                                  C, K, AH, AD, rows));
        CHECK(ds4_gpu_synchronize() && same_bits(want, have, (uint64_t)rows * AH * AD));
    }
    CHECK(ds4_gpu_test_kernel_runs(names[4]) == runs[4] + 7);
    /* FLASH_HEAD_SHARE: the same rows with one head per threadgroup */
    const uint64_t shared = ds4_gpu_test_kernel_runs("flash attention head share");
    ds4_gpu_test_set_v41_off(DS4_GPU_V41_FLASH_HEAD_SHARE);
    for (uint32_t rows = 2; rows <= ROWS; rows++) {
        CHECK(ds4_gpu_dsv41_attention_decode_rows(have, sinks, getpagesize(), 0, q, raw, R, R + ROWS, 0, cache, it, K,
                                                  C, K, AH, AD, rows));
        CHECK(ds4_gpu_synchronize() && same_bits(want, have, (uint64_t)rows * AH * AD));
    }
    ds4_gpu_test_set_v41_off(0);
    CHECK(shared == 7 && ds4_gpu_test_kernel_runs("flash attention head share") == shared);
    ds4_gpu_test_set_v41_off(DS4_GPU_V41_VERIFY_BATCH);
    CHECK(!ds4_gpu_dsv41_attention_decode_rows(have, sinks, getpagesize(), 0, q, raw, R, R + ROWS, 0, cache, it, K,
                                               C, K, AH, AD, 4));
    ds4_gpu_test_set_v41_off(0);
    ds4_gpu_test_set_flags(0);
    ds4_gpu_tensor *atensors[] = {q, raw, cache, it, want, have};
    for (size_t i = 0; i < sizeof(atensors) / sizeof(*atensors); i++) ds4_gpu_tensor_free(atensors[i]);
    free(ids); free(v); free(sinks);
    /* the weight maps stay: the model views may wrap them */
    fprintf(stderr, "V4.1 verify batch: 2-8 rows of the expand, the low projection and the decode attention (heads "
            "shared or not) == each row alone\n");
    return 1;
}

/* PREFILL_MATMUL: 33 and 300 rows of a Q8 matmul through the F16 unpack against the Q8 tiles, bit for bit. */
static int check_prefill_matmul(void) {
    enum { K = 4096, N = 1024, ROWS = 300 };
    uint64_t wb = 0;
    void *map = q8_weights(N, K, &wb);
    CHECK(map && ds4_gpu_set_model_map(map, wb));
    ds4_gpu_tensor *x = upload(NULL, (size_t)ROWS * K * 4), *ref = upload(NULL, (size_t)ROWS * N * 4),
                   *got = upload(NULL, (size_t)ROWS * N * 4);
    CHECK(x && ref && got);
    float *xv = ds4_gpu_tensor_contents(x);
    for (size_t i = 0; i < (size_t)ROWS * K; i++) xv[i] = bf16(random_value());
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    const uint64_t runs = ds4_gpu_test_kernel_runs("ds4_gpu_dsv41_matmul_q8_0_prefill");
    const uint32_t counts[] = {33, ROWS};
    for (int c = 0; c < 2; c++) {
        CHECK(ds4_gpu_matmul_q8_0_tensor(ref, map, wb, 0, K, N, x, counts[c]));
        CHECK(ds4_gpu_dsv41_matmul_q8_0_prefill(got, map, wb, 0, K, N, x, counts[c]));
        CHECK(ds4_gpu_synchronize() && same_bits(ref, got, (uint64_t)counts[c] * N));
    }
    CHECK(ds4_gpu_test_kernel_runs("ds4_gpu_dsv41_matmul_q8_0_prefill") == runs + 2);
    ds4_gpu_test_set_v41_off(DS4_GPU_V41_PREFILL_MATMUL);
    CHECK(!ds4_gpu_dsv41_matmul_q8_0_prefill(got, map, wb, 0, K, N, x, ROWS));
    ds4_gpu_test_set_v41_off(0);
    ds4_gpu_test_set_flags(0);
    ds4_gpu_tensor *tensors[] = {x, ref, got};
    for (size_t i = 0; i < sizeof(tensors) / sizeof(*tensors); i++) ds4_gpu_tensor_free(tensors[i]);
    fprintf(stderr, "V4.1 prefill matmul: Q8 rows through the F16 unpack == the Q8 tiles\n");
    return 1;
}

/* PREFILL_INDEX_BATCH: the batch top-k from 513 visible keys against per-row top-k (the same ids in the same order),
 * select-all for rows that see at most 512 keys, and a batch of ratio-1 rows past 2048 blocks through the batched
 * block top-k and strided masks against the per-row candidates, bit for bit on the filtered scores. */
static int check_prefill_index_batch(void) {
    enum { ROWS = 5, K = 512 };
    for (uint32_t ratio = 1; ratio <= 2; ratio++) {
        const uint32_t start = 600u * ratio - 1u, width = (start + ROWS) / ratio;
        ds4_gpu_tensor *scores = upload(NULL, (size_t)width * ROWS * 4), *sel = upload(NULL, ROWS * K * 4);
        ds4_gpu_tensor *want = upload(NULL, ROWS * K * 4);
        CHECK(scores && sel && want);
        float *s = ds4_gpu_tensor_contents(scores);
        for (uint32_t t = 0; t < ROWS; t++)
            for (uint32_t j = 0; j < width; j++)
                s[(size_t)t * width + j] = j < (start + t + 1u) / ratio ? random_value() : 12345.0f;
        for (uint32_t t = 0; t < ROWS; t++) {
            const uint32_t visible = (start + t + 1u) / ratio;
            ds4_gpu_tensor *sr = ds4_gpu_tensor_view(scores, (size_t)t * width * 4, visible * 4);
            ds4_gpu_tensor *out = ds4_gpu_tensor_view(want, (size_t)t * K * 4, K * 4);
            CHECK(sr && out && ds4_gpu_indexer_topk_tensor(out, sr, visible, 1, K));
            ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(sr);
        }
        ds4_gpu_test_set_v41_off(DS4_GPU_V41_PREFILL_INDEX_BATCH);
        CHECK(!ds4_gpu_dsv41_indexer_topk_batch(sel, scores, width, ROWS, start, ratio));
        ds4_gpu_test_set_v41_off(0);
        CHECK(ds4_gpu_dsv41_indexer_topk_batch(sel, scores, width, ROWS, start, ratio));
        CHECK(ds4_gpu_synchronize() && same_bits(sel, want, ROWS * K));
        ds4_gpu_tensor_free(want); ds4_gpu_tensor_free(sel); ds4_gpu_tensor_free(scores);
    }
    ds4_gpu_tensor *all = upload(NULL, 8 * K * 4);
    CHECK(all);
    int32_t *ids = ds4_gpu_tensor_contents(all);
    for (uint32_t i = 0; i < 8 * K; i++) ids[i] = -7;
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    const uint64_t runs = ds4_gpu_test_kernel_runs("kernel_dsv41_indexer_all");
    CHECK(ds4_gpu_dsv41_indexer_all_batch(all, 8, 1015, 2) && ds4_gpu_synchronize());
    CHECK(ds4_gpu_test_kernel_runs("kernel_dsv41_indexer_all") == runs + 1);
    ds4_gpu_test_set_flags(0);
    for (uint32_t t = 0; t < 8; t++)
        for (uint32_t k = 0; k < K; k++)
            CHECK(ids[t * K + k] == (k < (1015u + t + 1u) / 2u ? (int32_t)k : -7));
    CHECK(!ds4_gpu_dsv41_indexer_all_batch(all, 8, 1020, 2));
    ds4_gpu_tensor_free(all);

    enum { R = 4 };
    const uint32_t start = 16389, n = start + R, blocks = (n + 7u) / 8u, stride = blocks + 3u;
    ds4_gpu_tensor *a = upload(NULL, (size_t)n * R * 4), *b = upload(NULL, (size_t)n * R * 4);
    ds4_gpu_tensor *bs = upload(NULL, (size_t)blocks * R * 4), *bsel = upload(NULL, 2048 * R * 4);
    ds4_gpu_tensor *mask = upload(NULL, (size_t)stride * R * 4);
    ds4_gpu_tensor *rbs = upload(NULL, blocks * 4), *rsel = upload(NULL, 2048 * 4), *rmask = upload(NULL, blocks * 4);
    CHECK(a && b && bs && bsel && mask && rbs && rsel && rmask);
    float *av = ds4_gpu_tensor_contents(a), *bv = ds4_gpu_tensor_contents(b);
    for (uint32_t t = 0; t < R; t++)
        for (uint32_t j = 0; j < n; j++)
            av[(size_t)t * n + j] = bv[(size_t)t * n + j] = j <= start + t ? random_value() : 12345.0f;
    CHECK(ds4_gpu_dsv41_candidate_blocks(bs, a, n, R, start, 1) &&
          ds4_gpu_dsv41_candidate_topk_batch(bsel, bs, blocks, R, start) &&
          ds4_gpu_dsv41_candidate_mask_batch(mask, bsel, blocks, R, stride) &&
          ds4_gpu_dsv41_candidate_filter(a, mask, n, R, start, 1, stride));
    for (uint32_t t = 0; t < R; t++) {
        const uint32_t visible = start + t + 1u, vb = (visible + 7u) / 8u;
        ds4_gpu_tensor *row = ds4_gpu_tensor_view(b, (size_t)t * n * 4, (size_t)visible * 4);
        CHECK(row && ds4_gpu_dsv41_candidate_blocks(rbs, row, visible, 1, start + t, 1) &&
              ds4_gpu_indexer_topk_tensor(rsel, rbs, vb, 1, 2048) &&
              ds4_gpu_dsv4_topk_mask_tensor(rmask, rsel, vb, 1, 2048) &&
              ds4_gpu_dsv41_candidate_filter(row, rmask, visible, 1, start + t, 1, 0));
        CHECK(ds4_gpu_synchronize());
        for (uint32_t j = 0; j < n; j++) {
            const float x = av[(size_t)t * n + j];
            CHECK(j < visible ? memcmp(&x, &bv[(size_t)t * n + j], 4) == 0 : x == -INFINITY);
        }
        ds4_gpu_tensor_free(row);
    }
    ds4_gpu_tensor *tensors[] = {a, b, bs, bsel, mask, rbs, rsel, rmask};
    for (size_t i = 0; i < sizeof(tensors) / sizeof(*tensors); i++) ds4_gpu_tensor_free(tensors[i]);
    fprintf(stderr, "V4.1 prefill index batch: batch top-k, select-all and batched candidates == per row\n");
    return 1;
}

/* SAMPLER_GPU: the GPU top-p selector over random V4.1 logits: the finite count, the maximum and its earliest index
 * are exact, the cap is 512, and every one of the 512 largest logits is among the candidates it emits, with its
 * logit bit for bit. */
static int float_desc(const void *a, const void *b) {
    const float x = *(const float *)a, y = *(const float *)b;
    return x > y ? -1 : x < y;
}

static int check_sampler_gpu(void) {
    enum { NV = 129280, CAP = 512, SUPB = 2048 };
    typedef struct { uint32_t n, finite, best, cap; float M; uint32_t b1, above1, b2, above2, t, above, pad[5]; } hdr;
    typedef struct { float logit; uint32_t id; } cand;
    if (!ds4_gpu_selector_ready() && !ds4_gpu_selector_init()) {
        fprintf(stderr, "V4.1 sampler GPU: metal/sampler_selector.metal not found (run from the tree)\n");
        return 0;
    }
    ds4_gpu_tensor *logits = upload(NULL, NV * 4);
    float *lg = malloc(NV * sizeof(float)), *sorted = malloc(NV * sizeof(float));
    CHECK(logits && lg && sorted);
    /* a bulk of small logits, 600 distinct large ones and a tied maximum (the earliest index wins) */
    for (int i = 0; i < NV; i++) lg[i] = random_value();
    for (int k = 0; k < 600; k++) lg[1000 + k * 211] = 10.0f + (float)k * 0.01f;
    lg[77] = lg[99] = 20.0f;
    memcpy(ds4_gpu_tensor_contents(logits), lg, NV * sizeof(float));
    const uint64_t encoded = ds4_gpu_selector_encoded_count();
    CHECK(ds4_gpu_begin_commands() && ds4_gpu_selector_encode(logits) && ds4_gpu_end_commands());
    CHECK(ds4_gpu_selector_encoded_count() == encoded + 1);
    const hdr *h = ds4_gpu_selector_out(0);
    const cand *c = ds4_gpu_selector_out(2);
    CHECK(h && c && h->finite == NV && h->best == 77 && h->M == 20.0f && h->cap == CAP && h->n >= CAP && h->n <= SUPB);
    memcpy(sorted, lg, NV * sizeof(float));
    qsort(sorted, NV, sizeof(float), float_desc);
    const float threshold = sorted[CAP - 1];
    uint32_t found = 0;
    for (uint32_t i = 0; i < h->n; i++) {
        CHECK(c[i].id < NV && memcmp(&c[i].logit, &lg[c[i].id], 4) == 0);
        found += c[i].logit >= threshold;
    }
    uint32_t want = 0;
    for (int i = 0; i < NV; i++) want += lg[i] >= threshold;
    CHECK(found == want);
    free(sorted); free(lg);
    ds4_gpu_tensor_free(logits);
    fprintf(stderr, "V4.1 sampler GPU: selector summary == the logits' count, maximum and top 512\n");
    return 1;
}

/* TP_PREFILL_ROWS8: the two-Mac attention output's K-slice rows in one dispatch against one call per row, bit for
 * bit: 40 rows (the eight-row kernel on the M3 Ultra) and 5 rows (the two-row kernel). */
static int check_tp_prefill_rows8(void) {
    enum { FULL = 8192, K = 4096, OUT = 5120, SKIP = 3, ROWS = 40 };
    typedef struct { uint16_t d; int8_t qs[32]; } q8_block;
    const uint64_t bytes = (uint64_t)OUT * FULL / 32 * sizeof(q8_block);
    void *model = NULL;
    CHECK(posix_memalign(&model, getpagesize(), bytes) == 0);
    q8_block *q8 = model;
    for (uint64_t i = 0; i < bytes / sizeof(q8_block); i++) {
        q8[i].d = (uint16_t)(0x2000u | (seed & 0x3ffu));
        for (int j = 0; j < 32; j++) q8[i].qs[j] = (int8_t)(random_value() * 32);
    }
    CHECK(ds4_gpu_set_model_map(model, bytes));
    float *x = malloc((size_t)(SKIP + ROWS) * K * sizeof(float));
    CHECK(x != NULL);
    for (int i = 0; i < (SKIP + ROWS) * K; i++) x[i] = random_value();
    ds4_gpu_tensor *in = upload(x, (uint64_t)(SKIP + ROWS) * K * sizeof(float));
    ds4_gpu_tensor *a = upload(NULL, (uint64_t)ROWS * OUT * sizeof(float)), *b = upload(NULL, (uint64_t)ROWS * OUT * sizeof(float));
    CHECK(in && a && b);
    setenv("DS4_METAL_Q8_MV_NSG", "2", 1);   /* the two-Mac default */
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    int rows8 = 0;
    for (int pass = 0; pass < 2; pass++) {
        const uint32_t rows = pass ? 5u : ROWS;
        const uint64_t r8 = ds4_gpu_test_kernel_runs("kernel_mul_mv_q8_0_f32_rows8");
        CHECK(ds4_gpu_matmul_q8_0_kslice_exact_rows_tensor(a, model, bytes, 0, FULL, K, K, OUT, in,
                                                           (uint64_t)SKIP * K, rows));
        rows8 += (int)(ds4_gpu_test_kernel_runs("kernel_mul_mv_q8_0_f32_rows8") - r8);
        for (uint32_t t = 0; t < rows; t++) {
            ds4_gpu_tensor *row = ds4_gpu_tensor_view(b, (uint64_t)t * OUT * sizeof(float), OUT * sizeof(float));
            CHECK(row && ds4_gpu_matmul_q8_0_kslice_tensor(row, model, bytes, 0, FULL, K, K, OUT, in,
                                                           (uint64_t)(SKIP + t) * K));
            ds4_gpu_tensor_free(row);
        }
        CHECK(same_bits(a, b, (uint64_t)rows * OUT));
    }
    ds4_gpu_test_set_flags(0);
    unsetenv("DS4_METAL_Q8_MV_NSG");
    CHECK(rows8 <= 1);
    ds4_gpu_tensor_free(in); ds4_gpu_tensor_free(a); ds4_gpu_tensor_free(b);
    free(x);
    /* model stays: the model views may wrap it */
    fprintf(stderr, "V4.1 two-Mac K-slice rows in one dispatch == one call per row, 40 rows (%s) and 5\n",
            rows8 ? "eight-row kernel" : "two-row kernel");
    return 1;
}

/* VERIFY_INDEXED: a verify block's index rows over the bf16 key view against the batch scorer on the f32 keys in a
 * verify block (the tiled f32 kernel for 2-7 rows, the matrix kernel from 8): the same bits, except -inf in the blocks
 * a masked row's mask drops (which the candidate filter drops anyway); outside a verify block it declines. */
static int check_verify_indexed(void) {
    enum { HEADS = 32, DIM = 128, MAXR = 17 };
    const uint32_t counts[] = {9, 1031, 17011};
    const uint32_t max_keys = 17011, mask_width = (max_keys + 7) / 8;
    ds4_gpu_tensor *qt = upload(NULL, (size_t)MAXR * HEADS * DIM * sizeof(float));
    ds4_gpu_tensor *wt = upload(NULL, (size_t)MAXR * HEADS * sizeof(float));
    ds4_gpu_tensor *kt = upload(NULL, (size_t)max_keys * DIM * sizeof(float));
    ds4_gpu_tensor *vt = upload(NULL, (size_t)max_keys * DIM * sizeof(uint16_t));
    ds4_gpu_tensor *mt = upload(NULL, (size_t)MAXR * mask_width * sizeof(float));
    ds4_gpu_tensor *sr = upload(NULL, (size_t)MAXR * max_keys * sizeof(float));
    ds4_gpu_tensor *sv = upload(NULL, (size_t)MAXR * max_keys * sizeof(float));
    CHECK(qt && wt && kt && vt && mt && sr && sv);
    float *q = ds4_gpu_tensor_contents(qt), *w = ds4_gpu_tensor_contents(wt);
    float *k = ds4_gpu_tensor_contents(kt), *mask = ds4_gpu_tensor_contents(mt);
    float *ref = ds4_gpu_tensor_contents(sr), *got = ds4_gpu_tensor_contents(sv);
    uint64_t skipped = 0;
    ds4_gpu_dsv41_verify_rows(1);
    for (int dist = 0; dist < 2; dist++) {   /* wide and narrow FP4 block scales */
        for (size_t i = 0; i < (size_t)MAXR * HEADS * DIM; i++)
            q[i] = ldexpf(random_value(), dist ? (int)((i / 32 * 7) % 13) - 6 : (int)((i / 32) % 41) - 20);
        for (size_t i = 0; i < (size_t)MAXR * HEADS; i++) w[i] = bf16(random_value());
        for (size_t i = 0; i < (size_t)max_keys * DIM; i++)
            k[i] = ldexpf(random_value(), dist ? (int)((i / 32 * 5 + i / DIM) % 13) - 6 : (int)((i / 32) % 41) - 20);
        CHECK(ds4_gpu_dsv41_quantize(qt, DIM, MAXR * HEADS, DS4_V41_FP4_E8M0));
        CHECK(ds4_gpu_dsv41_quantize(kt, DIM, max_keys, DS4_V41_FP4_E8M0));
        CHECK(ds4_gpu_dsv41_index_keys_bf16(vt, kt, 0, max_keys, NULL));
        for (size_t i = 0; i < (size_t)MAXR * mask_width; i++)   /* about one block in ten kept */
            mask[i] = random_value() < -3.2f ? 0.0f : -INFINITY;
        for (size_t ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++)
            for (uint32_t ratio = 1; ratio <= 2; ratio++)
                for (uint32_t rows = 2; rows <= MAXR && rows <= counts[ci] * ratio; rows++)
                    for (uint32_t mode = 0; mode < 3; mode++) {   /* no mask, every row masked, the later half */
                        const uint32_t n = counts[ci], start = n * ratio - rows;
                        const uint32_t first = mode == 0 ? rows : mode == 1 ? 0 : rows / 2;
                        ds4_gpu_tensor *qv = ds4_gpu_tensor_view(qt, 0, (uint64_t)rows * HEADS * DIM * 4u);
                        ds4_gpu_tensor *wv = ds4_gpu_tensor_view(wt, 0, (uint64_t)rows * HEADS * 4u);
                        CHECK(qv && wv);
                        CHECK(ds4_gpu_dsv41_indexer_scores_batch(sr, qv, wv, kt, n, rows, start, ratio));
                        CHECK(ds4_gpu_dsv41_indexer_scores_rows_bf16(sv, qv, wv, vt, mode ? mt : NULL, mask_width,
                                                                     first, n, rows, start, ratio));
                        CHECK(ds4_gpu_synchronize());
                        ds4_gpu_tensor_free(qv); ds4_gpu_tensor_free(wv);
                        for (uint32_t t = 0; t < rows; t++)
                            for (uint32_t j = 0; j < n; j++) {
                                const float r = ref[(size_t)t * n + j], v = got[(size_t)t * n + j];
                                const bool same = !memcmp(&r, &v, sizeof(r));
                                const bool dropped = mode && t >= first && mask[(size_t)t * mask_width + j / 8] != 0.0f;
                                if (!same && !(dropped && v == -INFINITY)) {
                                    fprintf(stderr, "verify index bf16: n=%u ratio=%u rows=%u mode=%u row=%u key=%u "
                                            "ref=%a got=%a\n", n, ratio, rows, mode, t, j, r, v);
                                    CHECK(0);
                                }
                                skipped += !same;
                            }
                    }
    }
    CHECK(skipped > 0);
    CHECK(!ds4_gpu_dsv41_indexer_scores_rows_bf16(sv, qt, wt, vt, NULL, 0, 0, 1031, 1, 1030, 1));
    ds4_gpu_dsv41_verify_rows(0);
    CHECK(!ds4_gpu_dsv41_indexer_scores_rows_bf16(sv, qt, wt, vt, NULL, 0, 0, 1031, 2, 0, 1));
    ds4_gpu_tensor *tensors[] = {qt, wt, kt, vt, mt, sr, sv};
    for (size_t i = 0; i < sizeof(tensors) / sizeof(*tensors); i++) ds4_gpu_tensor_free(tensors[i]);
    fprintf(stderr, "V4.1 verify indexed: 2-17 rows over the bf16 keys == the batch scorer's bits (%llu masked entries "
            "skipped)\n", (unsigned long long)skipped);
    return 1;
}

/* SHORT_MMA: eight rows outside a DSpark block (a short prefill) of every V4.1 Q8_0 shape on the exact simdgroup
 * matrix kernels against main's kernels, bit for bit: the per-row grid for the plain and bf16 matmuls, the per-row
 * low projection, the row matmuls and SwiGLU for the shared expert, the row-by-row expand for output-B and the shared
 * down. Inside a DSpark block, at other row counts and with the switch off, the kernels do not run. */
static int check_short_mma(void) {
    enum { ROWS = 8, D = 5120, FF = 2304, HC = 4, GROUP = 4096, RANK = 1024, GROUPS = 8 };
    const uint32_t shapes[][2] = {{5120, 1280}, {5120, 512}, {1280, 32768}, {5120, 2304}, {2304, 5120}, {8192, 5120},
                                  {4096, 8192}};
    const int n_shapes = (int)(sizeof(shapes) / sizeof(*shapes));
    uint64_t wb = 0, lb = 0;
    uint8_t *map = q8_weights(5120, 8192, &wb), *low_map = q8_weights(GROUPS * RANK, GROUP, &lb);
    CHECK(map && low_map);
    ds4_gpu_tensor *x = upload(NULL, ROWS * 8192 * 4), *ref = upload(NULL, ROWS * 32768 * 4),
                   *got = upload(NULL, ROWS * 32768 * 4), *gate = upload(NULL, ROWS * FF * 4),
                   *up = upload(NULL, ROWS * FF * 4), *add = upload(NULL, ROWS * D * 4),
                   *res = upload(NULL, ROWS * HC * D * 4), *split = upload(NULL, ROWS * 24 * 4),
                   *heads = upload(NULL, ROWS * GROUPS * GROUP * 4);
    CHECK(x && ref && got && gate && up && add && res && split && heads);
    float *xv = ds4_gpu_tensor_contents(x), *av = ds4_gpu_tensor_contents(add), *rv = ds4_gpu_tensor_contents(res);
    float *sv = ds4_gpu_tensor_contents(split), *hv = ds4_gpu_tensor_contents(heads);
    for (int i = 0; i < ROWS * 8192; i++) xv[i] = random_value();
    for (int i = 0; i < ROWS * D; i++) av[i] = bf16(random_value());
    for (int i = 0; i < ROWS * HC * D; i++) rv[i] = bf16(random_value());
    for (int i = 0; i < ROWS * 24; i++) sv[i] = random_value();
    for (int i = 0; i < ROWS * GROUPS * GROUP; i++) hv[i] = bf16(random_value());
    ds4_gpu_test_set_flags(DS4_GPU_TEST_COUNT_KERNELS);
    const char *names[] = {"kernel_mul_mv_q8_0_f32_mma8_exact", "kernel_mul_mv_q8_0_f32_bf16_mma8_exact",
                           "kernel_mul_mv_q8_0_f32_bf16io_hc_expand4_mma8_exact",
                           "kernel_dsv4_shared_mid_swiglu_q8_0_bf16_mma8_exact",
                           "kernel_dsv4_attn_out_low_q8_0_f32_mma8_exact"};
    enum { NN = sizeof(names) / sizeof(*names) };
    uint64_t runs[NN];
    for (int k = 0; k < NN; k++) runs[k] = ds4_gpu_test_kernel_runs(names[k]);
    ds4_gpu_dsv41_dspark_rows(0);
    CHECK(ds4_gpu_set_model_map(map, wb));
    for (int s = 0; s < n_shapes; s++)
        for (int round = 0; round < 2; round++) {
            const uint32_t in = shapes[s][0], out = shapes[s][1];
            ds4_gpu_test_set_v41_off(DS4_GPU_V41_ROW_MATMULS);   /* main's per-row grid */
            CHECK(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(ref, map, wb, 0, in, out, x, ROWS));
            CHECK(!round || ds4_gpu_dsv41_quantize(ref, out, ROWS, DS4_V41_BF16));
            ds4_gpu_test_set_v41_off(0);
            CHECK(round ? ds4_gpu_dsv41_matmul_q8_0_bf16_rows(got, map, wb, 0, in, out, x, ROWS) :
                          ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(got, map, wb, 0, in, out, x, ROWS));
            CHECK(ds4_gpu_synchronize() && same_bits(ref, got, (uint64_t)ROWS * out));
        }
    /* the shared expert (5120 -> 2304): main's two row matmuls, the bf16 rounding and the SwiGLU */
    const uint64_t up_off = (uint64_t)FF * (D / 32) * 34;
    ds4_gpu_test_set_v41_off(DS4_GPU_V41_ROW_MATMULS);
    CHECK(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(gate, map, wb, 0, D, FF, x, ROWS) &&
          ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(up, map, wb, up_off, D, FF, x, ROWS) &&
          ds4_gpu_dsv41_quantize(gate, FF, ROWS, DS4_V41_BF16) && ds4_gpu_dsv41_quantize(up, FF, ROWS, DS4_V41_BF16) &&
          ds4_gpu_swiglu_tensor(ref, gate, up, ROWS * FF, 10.0f, 1.0f));
    ds4_gpu_test_set_v41_off(0);
    CHECK(ds4_gpu_dsv41_shared_swiglu(got, map, wb, 0, up_off, D, FF, x, 10.0f, ROWS));
    CHECK(ds4_gpu_synchronize() && same_bits(ref, got, (uint64_t)ROWS * FF));
    /* output-B (8192 -> 5120) and the shared down with its add (2304 -> 5120), each row expanded alone */
    for (int shape = 0; shape < 2; shape++) {
        const uint32_t k = shape ? FF : 8192;
        for (uint32_t r = 0; r < ROWS; r++) {
            ds4_gpu_tensor *o = ds4_gpu_tensor_view(ref, (uint64_t)r * HC * D * 4, HC * D * 4);
            ds4_gpu_tensor *xr = ds4_gpu_tensor_view(x, (uint64_t)r * k * 4, k * 4);
            ds4_gpu_tensor *ar = ds4_gpu_tensor_view(add, (uint64_t)r * D * 4, D * 4);
            ds4_gpu_tensor *rr = ds4_gpu_tensor_view(res, (uint64_t)r * HC * D * 4, HC * D * 4);
            ds4_gpu_tensor *pr = ds4_gpu_tensor_view(split, (uint64_t)r * 24 * 4, 24 * 4);
            CHECK(o && xr && ar && rr && pr &&
                  ds4_gpu_dsv41_matmul_expand(o, map, wb, 0, k, D, xr, shape ? ar : NULL, rr, pr, HC));
            ds4_gpu_tensor *views[] = {o, xr, ar, rr, pr};
            for (size_t i = 0; i < sizeof(views) / sizeof(*views); i++) ds4_gpu_tensor_free(views[i]);
        }
        CHECK(ds4_gpu_dsv41_matmul_expand_rows(got, map, wb, 0, k, D, x, shape ? add : NULL, res, split, HC, ROWS));
        CHECK(ds4_gpu_synchronize() && same_bits(ref, got, (uint64_t)ROWS * HC * D));
    }
    /* the low projection (8 groups of 4096 -> 1024), each row alone on main's kernel */
    CHECK(ds4_gpu_set_model_map(low_map, lb));
    for (uint32_t r = 0; r < ROWS; r++) {
        ds4_gpu_tensor *o = ds4_gpu_tensor_view(ref, (uint64_t)r * GROUPS * RANK * 4, GROUPS * RANK * 4);
        ds4_gpu_tensor *h = ds4_gpu_tensor_view(heads, (uint64_t)r * GROUPS * GROUP * 4, GROUPS * GROUP * 4);
        CHECK(o && h && ds4_gpu_attention_output_low_q8_tensor(o, low_map, lb, 0, GROUP, RANK, GROUPS, h));
        ds4_gpu_tensor_free(o);
        ds4_gpu_tensor_free(h);
    }
    CHECK(ds4_gpu_attention_output_low_q8_rows_exact_tensor(got, low_map, lb, 0, GROUP, RANK, GROUPS, 0, GROUPS, heads,
                                                            ROWS));
    CHECK(ds4_gpu_synchronize() && same_bits(ref, got, (uint64_t)ROWS * GROUPS * RANK));
    CHECK(ds4_gpu_test_kernel_runs(names[0]) == runs[0] + n_shapes &&
          ds4_gpu_test_kernel_runs(names[1]) == runs[1] + n_shapes && ds4_gpu_test_kernel_runs(names[2]) == runs[2] + 2 &&
          ds4_gpu_test_kernel_runs(names[3]) == runs[3] + 1 && ds4_gpu_test_kernel_runs(names[4]) == runs[4] + 1);
    /* not inside a DSpark block, not at seven rows, not with the switch off */
    for (int k = 0; k < NN; k++) runs[k] = ds4_gpu_test_kernel_runs(names[k]);
    ds4_gpu_dsv41_dspark_rows(1);
    CHECK(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(got, low_map, lb, 0, 5120, 1280, x, ROWS));
    ds4_gpu_dsv41_dspark_rows(0);
    CHECK(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(got, low_map, lb, 0, 5120, 1280, x, ROWS - 1));
    ds4_gpu_test_set_v41_off(DS4_GPU_V41_SHORT_MMA);
    CHECK(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(got, low_map, lb, 0, 5120, 1280, x, ROWS));
    CHECK(ds4_gpu_attention_output_low_q8_rows_exact_tensor(got, low_map, lb, 0, GROUP, RANK, GROUPS, 0, GROUPS, heads,
                                                            ROWS));
    ds4_gpu_test_set_v41_off(0);
    CHECK(ds4_gpu_synchronize());
    for (int k = 0; k < NN; k++) CHECK(ds4_gpu_test_kernel_runs(names[k]) == runs[k]);
    ds4_gpu_test_set_flags(0);
    ds4_gpu_tensor *tensors[] = {x, ref, got, gate, up, add, res, split, heads};
    for (size_t i = 0; i < sizeof(tensors) / sizeof(*tensors); i++) ds4_gpu_tensor_free(tensors[i]);
    /* the weight maps stay: the model views may wrap them */
    fprintf(stderr, "V4.1 short MMA: eight non-DSpark rows of every V4.1 Q8_0 shape on simdgroup matrices == main's "
            "kernels, bit for bit\n");
    return 1;
}

static int check_v41_groups(void) {
    return check_compute_copy() && check_router_select() && check_fused_rounding() && check_narrow_weights() &&
        check_qkv_fusion() && check_rope() && check_ffn_join() && check_hc_fusion() && check_split_reduce() &&
        check_shared_expert() && check_concurrent_ffn() && check_index_score_wide() &&
        check_index_bf16() && check_topk() && check_index_candidates() &&
        check_decode_attention() && check_fixed_shapes() && check_sampler_scan() &&
        check_row_matmuls() && check_rows_mma() && check_verify_kernels() &&
        check_public_controller() && check_prefill_index() && check_prefill_attention() &&
        check_prefill_folds() &&
        check_moe_dedup() &&
        check_verify_batch() &&
        check_prefill_matmul() &&
        check_prefill_index_batch() &&
        check_sampler_gpu() &&
        check_tp_prefill_rows8() &&
        check_verify_indexed() &&
        check_short_mma();
}
#else
static int check_v41_groups(void) { return 1; }
#endif

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--embedding")) {
        const int ok = ds4_gpu_init() && check_embedding();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--router")) {
        const int ok = ds4_gpu_init() && check_router();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--tp-attention")) {
        const int ok = ds4_gpu_init() && check_tp_attention();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && (!strcmp(argv[1], "--attention-output") ||
                      !strcmp(argv[1], "--attention-output-large"))) {
        const int ok = ds4_gpu_init() && check_attention_output(
            !strcmp(argv[1], "--attention-output-large"));
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
#ifdef __APPLE__
    if (argc == 2 && !strcmp(argv[1], "--hc-scaled")) {
        const int ok = ds4_gpu_init() && check_hc_scaled();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
#endif
    if (argc == 2 && !strcmp(argv[1], "--bf16-linear")) {
        const int ok = ds4_gpu_init() && check_bf16_linear();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--compact-carry")) {
        const int ok = ds4_gpu_init() && check_compact_carry();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--index-scores")) {
        const int ok = ds4_gpu_init() && check_indexer_batch();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--index-topk")) {
        const int ok = ds4_gpu_init() && check_causal_topk();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--general-topk")) {
        const int ok = ds4_gpu_init() && check_general_topk();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--index-projection")) {
        const int ok = ds4_gpu_init() && check_index_projection();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--v41-groups")) {
        const int ok = ds4_gpu_init() && check_v41_groups();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc != 1) return 2;
    int ok = ds4_gpu_init() && check_router() && check_quantization() && check_engram() && check_rope_stride() && check_pool() &&
             check_candidates() && check_sparse_gather() && check_indexer_batch() &&
             check_embedding() && check_index_projection() && check_general_topk() && check_causal_topk() && check_compact_carry() && check_attention_output(false) &&
             check_tp_attention() && check_v41_groups();
    ds4_gpu_cleanup();
    return ok ? 0 : 1;
}
