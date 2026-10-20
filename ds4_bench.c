#include "ds4.h"
#include "ds4_distributed.h"
#include "ds4_gpu_args.h"
#include "ds4_help.h"
#include "ds4_snapstore.h"
#include "ds4_tp.h"

/* Purpose-built throughput benchmark.
 *
 * The benchmark walks one fixed token sequence to configurable context
 * frontiers, measuring only the newest prefill interval at each frontier.  It
 * then snapshots the live session in memory when the payload is small enough,
 * performs a fixed greedy decode run without allowing EOS, restores the
 * snapshot or replays the prefix, and continues to the next frontier.  Snapshot
 * save/restore time is intentionally outside both timing windows.
 */

#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

/* CUDA builds resolve these weak symbols through libcudart; other builds
 * remain independent of CUDA headers and libraries. */
#if defined(__GNUC__) && !defined(__APPLE__)
#define DS4_BENCH_HAVE_CUDA_PROFILER 1
extern int cudaProfilerStart(void) __attribute__((weak));
extern int cudaProfilerStop(void) __attribute__((weak));
#endif

#define DS4_BENCH_DEFAULT_SNAPSHOT_MAX_BYTES (UINT64_C(1) << 30)

typedef struct {
    const char *model_path;
    const char *mtp_path;
    const char *v41_sidecar[DS4_V41_SIDECAR_MAX];
    const char *prompt_path;
    const char *chat_prompt_path;
    const char *system;
    const char *csv_path;
    const char *expert_profile_path;
    const char *gpu_vram_arg;
    const char *gpu_devices_arg;
    ds4_backend backend;
    int threads;
    int ctx_start;
    int ctx_max;
    int ctx_alloc;
    int step_incr;
    int gen_tokens;
    int power_percent;
    uint32_t prefill_chunk;
    uint32_t ssd_streaming_cache_experts;
    uint64_t ssd_streaming_cache_bytes;
    uint32_t ssd_streaming_full_layers;
    uint32_t ssd_streaming_preload_experts;
    uint64_t simulate_used_memory_bytes;
    double step_mul;
    const char *dump_frontier_logits_dir;
    ds4_dist_options dist;
    ds4_tp_options tp;
    bool warm_weights;
    bool quality;
    bool ssd_streaming;
    bool ssd_streaming_cold;
    bool ssd_streaming_full_layers_set;
    bool cuda_tensor_parallel;
    bool show_output;
    bool teacher_forced_decode;
    bool dspark;
    bool dspark_confidence_threshold_set;
    float dspark_confidence_threshold;
    bool sampled;
    float temperature;
    float top_p;
    int top_k;
    float min_p;
    uint64_t seed;
    const char *token_csv_path;
    const char *jobs_path;
} bench_config;

static double bench_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static uint64_t bench_snapshot_max_bytes(void) {
    const char *env = getenv("DS4_BENCH_SNAPSHOT_MAX_BYTES");
    if (!env || env[0] == '\0') return DS4_BENCH_DEFAULT_SNAPSHOT_MAX_BYTES;
    if (!strcmp(env, "unlimited") || !strcmp(env, "UNLIMITED") ||
        !strcmp(env, "inf") || !strcmp(env, "INF")) {
        return UINT64_MAX;
    }
    char *end = NULL;
    unsigned long long v = strtoull(env, &end, 10);
    if (env[0] == '\0' || !end || *end != '\0') {
        fprintf(stderr,
                "ds4-bench: invalid DS4_BENCH_SNAPSHOT_MAX_BYTES=%s; using default %llu\n",
                env,
                (unsigned long long)DS4_BENCH_DEFAULT_SNAPSHOT_MAX_BYTES);
        return DS4_BENCH_DEFAULT_SNAPSHOT_MAX_BYTES;
    }
    return (uint64_t)v;
}

static double bytes_to_gib(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

static void usage(FILE *fp, const char *topic) {
    ds4_help_print(fp, DS4_HELP_BENCH, topic);
}

static int parse_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v <= 0 || v > INT_MAX) {
        fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static int parse_nonnegative_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v < 0 || v > INT_MAX) {
        fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static double parse_double_arg(const char *s, const char *opt) {
    char *end = NULL;
    double v = strtod(s, &end);
    if (s[0] == '\0' || *end != '\0' || !isfinite(v)) {
        fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return v;
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4-bench: %s requires an argument\n", opt);
        exit(2);
    }
    return argv[++*i];
}

static ds4_backend parse_backend(const char *s, const char *opt) {
    if (!strcmp(s, "metal")) return DS4_BACKEND_METAL;
#ifdef DS4_ROCM_BUILD
    if (!strcmp(s, "rocm")) return DS4_BACKEND_CUDA;
#else
    if (!strcmp(s, "cuda")) return DS4_BACKEND_CUDA;
#endif
    if (!strcmp(s, "cpu")) return DS4_BACKEND_CPU;
    fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
#ifdef DS4_ROCM_BUILD
    fprintf(stderr, "ds4-bench: valid backends are: metal, rocm, cpu\n");
#else
    fprintf(stderr, "ds4-bench: valid backends are: metal, cuda, cpu\n");
#endif
    exit(2);
}

static ds4_backend default_backend(void) {
#ifdef DS4_NO_GPU
    return DS4_BACKEND_CPU;
#elif defined(__APPLE__)
    return DS4_BACKEND_METAL;
#else
    return DS4_BACKEND_CUDA;
#endif
}

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ds4-bench: failed to open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "ds4-bench: failed to seek %s\n", path);
        fclose(fp);
        exit(1);
    }
    long n = ftell(fp);
    if (n < 0) {
        fprintf(stderr, "ds4-bench: failed to tell %s\n", path);
        fclose(fp);
        exit(1);
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fprintf(stderr, "ds4-bench: failed to rewind %s\n", path);
        fclose(fp);
        exit(1);
    }
    char *buf = malloc((size_t)n + 1);
    if (!buf) {
        fprintf(stderr, "ds4-bench: out of memory reading %s\n", path);
        fclose(fp);
        exit(1);
    }
    if (fread(buf, 1, (size_t)n, fp) != (size_t)n) {
        fprintf(stderr, "ds4-bench: failed to read %s\n", path);
        free(buf);
        fclose(fp);
        exit(1);
    }
    fclose(fp);
    buf[n] = '\0';
    return buf;
}

static bench_config parse_options(int argc, char **argv) {
    bench_config c = {
        .model_path = "ds4flash.gguf",
        .system = "You are a helpful assistant.",
        .backend = default_backend(),
        .ctx_start = 2048,
        .ctx_max = 32768,
        .step_incr = 2048,
        .gen_tokens = 128,
        .step_mul = 1.0,
        .top_p = 1.0f,
        .seed = 1,
    };

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            const char *topic = (i + 1 < argc && argv[i + 1][0] != '-') ?
                argv[i + 1] : NULL;
            usage(stdout, topic);
            exit(0);
        }
        char dist_parse_err[256] = {0};
        ds4_dist_cli_parse_result dist_parse =
            ds4_dist_parse_cli_arg(arg,
                                   &i,
                                   argc,
                                   argv,
                                   &c.dist,
                                   dist_parse_err,
                                   sizeof(dist_parse_err));
        if (dist_parse == DS4_DIST_CLI_ERROR) {
            fprintf(stderr,
                    "ds4-bench: %s\n",
                    dist_parse_err[0] ? dist_parse_err : "invalid distributed option");
            exit(2);
        }
        if (dist_parse == DS4_DIST_CLI_MATCHED) continue;

        char tp_parse_err[256] = {0};
        ds4_tp_cli_parse_result tp_parse =
            ds4_tp_parse_cli_arg(arg,
                                 &i,
                                 argc,
                                 argv,
                                 &c.tp,
                                 tp_parse_err,
                                 sizeof(tp_parse_err));
        if (tp_parse == DS4_TP_CLI_ERROR) {
            fprintf(stderr,
                    "ds4-bench: %s\n",
                    tp_parse_err[0] ? tp_parse_err : "invalid tensor-parallel option");
            exit(2);
        }
        if (tp_parse == DS4_TP_CLI_MATCHED) continue;

        if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            c.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp-model")) {
            c.mtp_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--v41-sidecar")) {
            if (!ds4_v41_sidecar_add(c.v41_sidecar, need_arg(&i, argc, argv, arg))) {
                fprintf(stderr, "ds4-bench: --v41-sidecar takes at most %d files\n", DS4_V41_SIDECAR_MAX);
                exit(2);
            }
        } else if (!strcmp(arg, "--dspark")) {
            c.dspark = true;
        } else if (!strcmp(arg, "--dspark-confidence")) {
            const double v = parse_double_arg(need_arg(&i, argc, argv, arg), arg);
            if (v < 0.0 || v > 1.0) {
                fprintf(stderr, "ds4-bench: --dspark-confidence must be between 0 and 1\n");
                exit(2);
            }
            c.dspark = true;
            c.dspark_confidence_threshold = (float)v;
            c.dspark_confidence_threshold_set = true;
        } else if (!strcmp(arg, "--prompt-file")) {
            c.prompt_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--chat-prompt-file")) {
            c.chat_prompt_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-sys") || !strcmp(arg, "--system")) {
            c.system = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--ctx-start")) {
            c.ctx_start = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--ctx-max")) {
            c.ctx_max = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--ctx-alloc")) {
            c.ctx_alloc = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--step-incr")) {
            c.step_incr = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--step-mul")) {
            c.step_mul = parse_double_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--gen-tokens") || !strcmp(arg, "--tokens") || !strcmp(arg, "-n")) {
            c.gen_tokens = parse_nonnegative_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--csv")) {
            c.csv_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dump-frontier-logits-dir")) {
            c.dump_frontier_logits_dir = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--expert-profile")) {
            c.expert_profile_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.threads = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--backend")) {
            c.backend = parse_backend(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--metal")) {
            c.backend = DS4_BACKEND_METAL;
#ifdef DS4_ROCM_BUILD
        } else if (!strcmp(arg, "--rocm")) {
            c.backend = DS4_BACKEND_CUDA;
#else
        } else if (!strcmp(arg, "--cuda")) {
            c.backend = DS4_BACKEND_CUDA;
#endif
        } else if (!strcmp(arg, "--gpu-vram")) {
            c.gpu_vram_arg = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--gpu-devices")) {
            c.gpu_devices_arg = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--cuda-tensor-parallel")) {
            c.cuda_tensor_parallel = true;
        } else if (!strcmp(arg, "--cpu")) {
            c.backend = DS4_BACKEND_CPU;
        } else if (!strcmp(arg, "--quality")) {
            c.quality = true;
        } else if (!strcmp(arg, "--ssd-streaming")) {
            c.ssd_streaming = true;
        } else if (!strcmp(arg, "--ssd-streaming-cold")) {
            c.ssd_streaming_cold = true;
        } else if (!strcmp(arg, "--ssd-streaming-cache-experts")) {
            uint32_t experts = 0;
            uint64_t bytes = 0;
            if (!ds4_parse_streaming_cache_experts_arg(
                    need_arg(&i, argc, argv, arg), &experts, &bytes)) {
                fprintf(stderr,
                        "ds4-bench: --ssd-streaming-cache-experts must be a positive count or <number>GB\n");
                exit(2);
            }
            c.ssd_streaming_cache_experts = experts;
            c.ssd_streaming_cache_bytes = bytes;
        } else if (!strcmp(arg, "--ssd-streaming-full-layers")) {
            int v = parse_nonnegative_int(need_arg(&i, argc, argv, arg), arg);
            c.ssd_streaming_full_layers = (uint32_t)v;
            c.ssd_streaming_full_layers_set = true;
        } else if (!strcmp(arg, "--ssd-streaming-preload-experts")) {
            int v = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (v <= 0) {
                fprintf(stderr, "ds4-bench: --ssd-streaming-preload-experts must be positive\n");
                exit(2);
            }
            c.ssd_streaming_preload_experts = (uint32_t)v;
        } else if (!strcmp(arg, "--simulate-used-memory")) {
            if (!ds4_parse_gib_arg(need_arg(&i, argc, argv, arg),
                                   &c.simulate_used_memory_bytes)) {
                fprintf(stderr,
                        "ds4-bench: --simulate-used-memory must be a positive GiB value, e.g. 64GB\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--prefill-chunk")) {
            c.prefill_chunk = (uint32_t)parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--power")) {
            c.power_percent = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (c.power_percent < 1 || c.power_percent > 100) {
                fprintf(stderr, "ds4-bench: --power must be between 1 and 100\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--warm-weights")) {
            c.warm_weights = true;
        } else if (!strcmp(arg, "--show-output")) {
            c.show_output = true;
        } else if (!strcmp(arg, "--teacher-forced-decode")) {
            c.teacher_forced_decode = true;
        } else if (!strcmp(arg, "--temp")) {
            c.temperature = (float)parse_double_arg(need_arg(&i, argc, argv, arg), arg);
            c.sampled = true;
        } else if (!strcmp(arg, "--top-p")) {
            c.top_p = (float)parse_double_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--top-k")) {
            c.top_k = parse_nonnegative_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--min-p")) {
            c.min_p = (float)parse_double_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--seed")) {
            const char *v = need_arg(&i, argc, argv, arg);
            char *end = NULL;
            errno = 0;
            c.seed = strtoull(v, &end, 10);
            if (end == v || *end || errno) {
                fprintf(stderr, "ds4-bench: --seed expects an unsigned integer\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--token-csv")) {
            c.token_csv_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--jobs")) {
            c.jobs_path = need_arg(&i, argc, argv, arg);
        } else {
            fprintf(stderr, "ds4-bench: unknown option: %s\n", arg);
            usage(stderr, NULL);
            exit(2);
        }
    }

    if (!c.jobs_path && !!c.prompt_path == !!c.chat_prompt_path) {
        fprintf(stderr, "ds4-bench: specify exactly one of --prompt-file or --chat-prompt-file\n");
        exit(2);
    }
    if (c.dspark && !c.mtp_path) {
        fprintf(stderr, "ds4-bench: --dspark requires --mtp-model FILE\n");
        exit(2);
    }
    if (c.sampled && c.teacher_forced_decode) {
        fprintf(stderr, "ds4-bench: --temp cannot be combined with --teacher-forced-decode\n");
        exit(2);
    }
    if (c.dspark && c.teacher_forced_decode) {
        fprintf(stderr,
                "ds4-bench: --dspark cannot be combined with "
                "--teacher-forced-decode\n");
        exit(2);
    }
    if (c.ctx_start > c.ctx_max) {
        fprintf(stderr, "ds4-bench: --ctx-start must be <= --ctx-max\n");
        exit(2);
    }
    if (c.step_mul < 1.0) {
        fprintf(stderr, "ds4-bench: --step-mul must be >= 1\n");
        exit(2);
    }
    if (c.step_mul == 1.0 && c.step_incr <= 0) {
        fprintf(stderr, "ds4-bench: --step-incr must be positive when --step-mul is 1\n");
        exit(2);
    }
    if (c.ctx_max > INT_MAX - c.gen_tokens - 1) {
        fprintf(stderr, "ds4-bench: requested context is too large\n");
        exit(2);
    }
    if (c.ctx_alloc == 0) c.ctx_alloc = c.ctx_max + c.gen_tokens + 1;
    if (c.ctx_alloc <= c.ctx_max + c.gen_tokens) {
        fprintf(stderr, "ds4-bench: --ctx-alloc must be greater than ctx-max + gen-tokens\n");
        exit(2);
    }
    char tp_err[256];
    if (!ds4_tp_adopt_distributed_options(&c.tp,
                                          &c.dist,
                                          tp_err,
                                          sizeof(tp_err))) {
        fprintf(stderr, "ds4-bench: %s\n", tp_err);
        exit(2);
    }
    char dist_err[256];
    if (ds4_dist_prepare_engine_options(&c.dist, NULL, dist_err, sizeof(dist_err)) != 0) {
        fprintf(stderr, "ds4-bench: %s\n", dist_err);
        exit(2);
    }
    if (c.dist.role == DS4_DISTRIBUTED_WORKER || c.tp.role == DS4_TP_WORKER) {
        fprintf(stderr, "ds4-bench: --role worker is a serving mode; start workers with ./ds4\n");
        exit(2);
    }
    return c;
}

static void json_write_string(FILE *fp, const char *s) {
    fputc('"', fp);
    if (s) {
        for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
            switch (*p) {
            case '"':  fputs("\\\"", fp); break;
            case '\\': fputs("\\\\", fp); break;
            case '\b': fputs("\\b", fp); break;
            case '\f': fputs("\\f", fp); break;
            case '\n': fputs("\\n", fp); break;
            case '\r': fputs("\\r", fp); break;
            case '\t': fputs("\\t", fp); break;
            default:
                if (*p < 0x20) fprintf(fp, "\\u%04x", (unsigned)*p);
                else fputc((char)*p, fp);
                break;
            }
        }
    }
    fputc('"', fp);
}

static int write_frontier_logits_json(
        const bench_config *cfg,
        ds4_engine         *engine,
        ds4_session        *session,
        int                 frontier,
        int                 previous) {
    if (!cfg->dump_frontier_logits_dir) return 0;

    const int vocab = ds4_engine_vocab_size(engine);
    float *logits = malloc((size_t)vocab * sizeof(logits[0]));
    if (!logits) {
        fprintf(stderr, "ds4-bench: out of memory copying frontier logits\n");
        return 1;
    }
    if (ds4_session_copy_logits(session, logits, vocab) != vocab) {
        fprintf(stderr, "ds4-bench: failed to copy frontier logits at %d\n", frontier);
        free(logits);
        return 1;
    }

    char path[PATH_MAX];
    const int n = snprintf(path,
                           sizeof(path),
                           "%s/frontier_%06d.logits.json",
                           cfg->dump_frontier_logits_dir,
                           frontier);
    if (n <= 0 || (size_t)n >= sizeof(path)) {
        fprintf(stderr, "ds4-bench: frontier logits path is too long\n");
        free(logits);
        return 1;
    }

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4-bench: failed to open %s: %s\n", path, strerror(errno));
        free(logits);
        return 1;
    }

    const int argmax = ds4_session_argmax(session);
    fprintf(fp, "{\n  \"source\":\"ds4-bench\",\n  \"model\":");
    json_write_string(fp, cfg->model_path);
    fprintf(fp,
            ",\n  \"backend\":\"%s\",\n  \"quality\":%s,\n"
            "  \"quant_bits\":%d,\n  \"prompt_tokens\":%d,\n"
            "  \"frontier_tokens\":%d,\n  \"prefill_tokens\":%d,\n"
            "  \"ctx\":%d,\n  \"vocab\":%d,\n"
            "  \"argmax_id\":%d,\n  \"argmax_logit\":%.9g,\n  \"logits\":[",
            ds4_backend_name(cfg->backend),
            cfg->quality ? "true" : "false",
            ds4_engine_routed_quant_bits(engine),
            frontier,
            frontier,
            frontier - previous,
            cfg->ctx_alloc,
            vocab,
            argmax,
            logits[argmax]);
    for (int i = 0; i < vocab; i++) {
        if (i) fputc(',', fp);
        if ((i % 8) == 0) fputs("\n    ", fp);
        if (isfinite(logits[i])) fprintf(fp, "%.9g", logits[i]);
        else fputs("null", fp);
    }
    fputs("\n  ]\n}\n", fp);
    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4-bench: failed to close %s\n", path);
        free(logits);
        return 1;
    }
    free(logits);
    return 0;
}

static int next_frontier(const bench_config *c, int cur) {
    if (cur >= c->ctx_max) return c->ctx_max;
    int next;
    if (c->step_mul == 1.0) {
        if (cur > INT_MAX - c->step_incr) next = c->ctx_max;
        else next = cur + c->step_incr;
    } else {
        const double v = ceil((double)cur * c->step_mul);
        next = v > (double)INT_MAX ? c->ctx_max : (int)v;
        if (next <= cur) next = cur + 1;
    }
    if (next > c->ctx_max) next = c->ctx_max;
    return next;
}

static void log_context_memory(ds4_backend backend,
                               int         ctx_size,
                               uint32_t    prefill_chunk,
                               bool        ssd_streaming) {
    ds4_context_memory m =
        ds4_context_memory_estimate_with_prefill_mode(backend,
                                                      ctx_size,
                                                      prefill_chunk,
                                                      ssd_streaming);
    fprintf(stderr,
            "ds4-bench: context buffers %.2f MiB (ctx=%d, backend=%s, prefill_chunk=%u, raw_kv_rows=%u, compressed_kv_rows=%u)\n",
            (double)m.total_bytes / (1024.0 * 1024.0),
            ctx_size,
            ds4_backend_name(backend),
            m.prefill_cap,
            m.raw_cap,
            m.comp_cap);
}

static int wait_distributed_route(ds4_session *session) {
    char err[256] = {0};
    char last[256] = {0};
    unsigned ticks = 0;
    const struct timespec delay = {0, 250000000L};

    for (;;) {
        int ready = ds4_session_distributed_route_ready(session, err, sizeof(err));
        if (ready > 0) {
            if (ticks) fprintf(stderr, "ds4-bench: distributed route ready\n");
            return 0;
        }
        if (ready < 0) {
            fprintf(stderr,
                    "ds4-bench: distributed route readiness failed: %s\n",
                    err[0] ? err : "unknown error");
            return 1;
        }
        const char *why = err[0] ? err : "route incomplete";
        if (strcmp(last, why) != 0 || (ticks % 20u) == 0) {
            fprintf(stderr, "ds4-bench: waiting for distributed route: %s\n", why);
            snprintf(last, sizeof(last), "%s", why);
        }
        nanosleep(&delay, NULL);
        ticks++;
    }
}

static void maybe_warn_distributed_step_shape(const bench_config *cfg, ds4_session *session) {
    if (!cfg || !session || cfg->dist.role != DS4_DISTRIBUTED_COORDINATOR) return;
    uint32_t chunk = cfg->dist.prefill_chunk;
    if (chunk == 0) {
        const int cap = ds4_session_prefill_cap(session);
        if (cap > 0) chunk = (uint32_t)cap;
    }
    if (chunk == 0) return;
    if (cfg->step_mul == 1.0 &&
        cfg->step_incr > 0 &&
        (uint32_t)cfg->step_incr < chunk &&
        cfg->ctx_start < cfg->ctx_max)
    {
        fprintf(stderr,
                "ds4-bench: note: --step-incr=%d is smaller than distributed prefill chunk %u; "
                "suffix rows will not show multi-chunk pipeline overlap\n",
                cfg->step_incr,
                chunk);
    }
}

/* Same time base as Metal's GPU timestamps on Apple (mach_absolute_time). */
static uint64_t bench_now_ns(void) {
#ifdef __APPLE__
    return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
#endif
}

extern char **environ;

/* ---- Resident job runner (--jobs FILE --token-csv PATH) ----
 * One process, one engine and session, many experiment jobs:
 *   save fixture=PATH depth=N cache=PATH [canary=M]
 *                                           prefill the first N tokens, write the engine
 *                                           payload + a .meta identity file (snapshot format 4,
 *                                           ds4_snapstore.h); canary=M also records the logits of
 *                                           a fresh M-token prefill of the fixture in a side session
 *   load fixture=PATH depth=N cache=PATH [expect=reject]
 *                                           restore a saved frontier of exactly that
 *                                           fixture prefix (no prefill fallback: a missing,
 *                                           incomplete, corrupt, incompatible or unverifiable
 *                                           cache fails the job; expect=reject inverts that
 *                                           for negative checks)
 *   run label=L tokens=N seed=S [mode=sampled|greedy|teacher] [temp= top_p= top_k= min_p=]
 *       [set=NAME=VALUE ...] [unset=NAME ...] [warm=1] [fixture=PATH] [state=1] [dump=PATH]
 *                                           restore the current frontier, apply the
 *                                           environment changes, reset the RNG, decode
 *   quit                                    end the runner (until then it waits for appended jobs)
 * Each run restores the in-memory snapshot of the last save/load, so arms switch at a
 * quiescent boundary. Per token: PATH; per job: PATH.jobs.csv, with the decode paths'
 * dispatch counts of the run (Engram gate form, fused index query, blend readers, GPU
 * selector outcomes, pre-encoded buffers, GPU bridge).
 * Snapshots: payload and meta are each written to a temporary file, fsynced and
 * renamed, never over an existing snapshot; the meta ends with "end". A load checks
 * the producer identity by category, then the payload's size and hash, before the
 * engine sees a byte. A save that extends the frontier of the last save/load records
 * it as "base PATH HASH DEPTH". Format-3 caches (BENCH_CACHE_FORMAT) still load. */
#define BENCH_CACHE_FORMAT "ds4-bench-prefill-cache-3"
/* Versioned contract of what produces a V4.1 prefix frontier. Bump it when a new
 * prefix producer appears (e.g. an FP8 prefill path), never stamp old payloads. */
#define BENCH_CACHE_CONTRACT "v41-prefix-1"
#define BENCH_PE_STATS 13   /* ds4_preencode_stats: prepared, used, pending, then 10 drop reasons */
#define BENCH_JOB_COUNTERS 22

typedef struct {
    char path[PATH_MAX];
    ds4_tokens tokens;
} bench_fixture;

static uint64_t bench_fnv_tokens(const int *v, int n) {
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < n; i++) { h ^= (uint32_t)v[i]; h *= 1099511628211ull; }
    return h;
}

static uint64_t bench_fnv_logits(ds4_session *s, float *buf, int cap) {
    uint64_t h = 1469598103934665603ull;
    const int n = ds4_session_copy_logits(s, buf, cap);
    for (int i = 0; i < n; i++) {
        uint32_t u;
        memcpy(&u, &buf[i], sizeof(u));
        h ^= u;
        h *= 1099511628211ull;
    }
    return h;
}

static const char *job_get(char **kv, int n, const char *key) {
    const size_t k = strlen(key);
    for (int i = 0; i < n; i++)
        if (!strncmp(kv[i], key, k) && kv[i][k] == '=') return kv[i] + k + 1;
    return NULL;
}

static const ds4_tokens *job_fixture(ds4_engine *engine, bench_fixture *fx, const char *path) {
    if (!path) return NULL;
    if (strcmp(fx->path, path) != 0) {
        char *text = read_file(path);
        ds4_tokens_free(&fx->tokens);
        memset(&fx->tokens, 0, sizeof(fx->tokens));
        ds4_tokenize_text(engine, text, &fx->tokens);
        free(text);
        snprintf(fx->path, sizeof(fx->path), "%s", path);
    }
    return &fx->tokens;
}

/* "path size mtime sha256" of a weight file; the sha comes from a recorded
 * PATH.sha256 or $DS4_BENCH_SHA_DIR/BASENAME.sha256 (never rehashed here).
 * Returns 0 ("unverifiable") when the file or a valid recorded digest is missing. */
static int file_identity(const char *path, char *out, size_t cap) {
    if (!path || !path[0]) { snprintf(out, cap, "-"); return 1; }
    struct stat st = {0};
    const int have = stat(path, &st) == 0;
    char sha_path[PATH_MAX], sha[72] = "";
    snprintf(sha_path, sizeof(sha_path), "%s.sha256", path);
    FILE *fp = fopen(sha_path, "rb");
    const char *dir = getenv("DS4_BENCH_SHA_DIR");
    if (!fp && dir) {
        const char *base = strrchr(path, '/');
        snprintf(sha_path, sizeof(sha_path), "%s/%s.sha256", dir, base ? base + 1 : path);
        fp = fopen(sha_path, "rb");
    }
    if (fp) {
        if (fscanf(fp, "%64s", sha) != 1) sha[0] = '\0';
        fclose(fp);
    }
    const int ok = have && strlen(sha) == 64 && strspn(sha, "0123456789abcdef") == 64;
    snprintf(out, cap, "%s %lld %lld %s", path, (long long)st.st_size, (long long)st.st_mtime,
             ok ? sha : "unverifiable");
    return ok;
}

/* Producer compatibility signature (contract BENCH_CACHE_CONTRACT): the inputs that shape a saved
 * prefill frontier. Decode-only switches are absent; saves refuse DS4_METAL_DISABLE_V41_* so every
 * payload comes from the contract's default prefill path. Returns 0 when a weight identity is
 * unverifiable. */
static int cache_identity(const bench_config *cfg, ds4_engine *engine, ds4_session *session,
                          char *out, size_t cap) {
    static const char *numerics_env[] = {
        "DS4_METAL_HC_STABLE", "DS4_METAL_NORM_RSQRT_DISABLE", "DS4_METAL_KV_RAW_F32",
        "DS4_METAL_ROPE_EXP2_LOG2", "DS4_METAL_MATH_SAFE",
    };
    char model[PATH_MAX + 128], drafter[PATH_MAX + 128];
    const int ok = file_identity(cfg->model_path, model, sizeof(model)) &
                   file_identity(cfg->mtp_path, drafter, sizeof(drafter));
    int n = snprintf(out, cap,
             "format %s\ncontract %s\npayload_version %u\nmodel %s\ndrafter %s\n"
             "ctx %d\nprefill_chunk %d\nvocab %d\n",
             BENCH_CACHE_FORMAT, BENCH_CACHE_CONTRACT, (unsigned)DS4_SESSION_PAYLOAD_VERSION, model, drafter,
             ds4_session_ctx(session), ds4_engine_prefill_chunk(engine), ds4_engine_vocab_size(engine));
    for (size_t i = 0; i < sizeof(numerics_env) / sizeof(numerics_env[0]) && n > 0 && (size_t)n < cap; i++) {
        const char *v = getenv(numerics_env[i]);
        n += snprintf(out + n, cap - (size_t)n, "numerics %s=%s\n", numerics_env[i], v ? v : "(unset)");
    }
    return ok;
}

/* Format 4 identity (ds4_snapstore.h): the same producer inputs plus the
 * prefill weight representation; weights compare by size+sha, not path/mtime. */
static int cache_identity4(const bench_config *cfg, ds4_engine *engine, ds4_session *session,
                           char *out, size_t cap) {
    char legacy[8192];
    const int ok = cache_identity(cfg, engine, session, legacy, sizeof(legacy));
    const char *body = strchr(legacy, '\n');   /* drop the format-3 line */
    int n = snprintf(out, cap, "format %s%s", DS4_SS_FORMAT, body ? body : "\n");
    if (n > 0 && (size_t)n < cap)
        snprintf(out + n, cap - (size_t)n, "prefill_repr %s\nprefill_values %s\n",
                 ds4_engine_prefill_repr(engine), ds4_engine_prefill_values(engine));
    return ok;
}

/* Numerics canary: logits hash of a FRESH prefill of the fixture's first len
 * tokens in a small side session of the same engine. A compatible rebuild
 * reproduces it; a prefill numerics change does not. "LEN PREFIX LOGITS". */
static int bench_canary(ds4_engine *engine, const ds4_tokens *all, int len, char *out, size_t cap) {
    if (!all || len <= 0 || len > all->len) return 1;
    ds4_session *cs = NULL;
    char err[256];
    ds4_tokens pre = {.v = all->v, .len = len, .cap = len};
    float *buf = malloc((size_t)(1 << 18) * sizeof(float));
    int rc = !buf || ds4_session_create(&cs, engine, len + 64) != 0 ||
             ds4_session_sync(cs, &pre, err, sizeof(err)) != 0;
    if (!rc) snprintf(out, cap, "%d %016llx %016llx", len,
                      (unsigned long long)bench_fnv_tokens(all->v, len),
                      (unsigned long long)bench_fnv_logits(cs, buf, 1 << 18));
    if (cs) ds4_session_free(cs);
    free(buf);
    return rc;
}

/* The decode paths' dispatch counters, in jobs.csv column order: engram_add_tg, irq_tab, blend_router,
 * blend_engram, blend_indexer, sel_encoded, sel_decided .. sel_stale (9), pe_prep, pe_used, pe_pending,
 * pe_drop, gpu_bridge, gpu_bridge_retired, gpu_bridge_failed. pe_pending is a level, the rest are counts. */
static void job_counters(uint64_t c[BENCH_JOB_COUNTERS], uint64_t pe[BENCH_PE_STATS]) {
    uint64_t b[3], g[9], br[3];
    ds4_blend_counts(b);
    ds4_sampler_gpu_counts(g);
    ds4_preencode_stats(pe, BENCH_PE_STATS);
    ds4_decode_bridge_stats(br);
    int k = 0;
    c[k++] = ds4_engram_add_tg_count();
    c[k++] = ds4_irq_table_count();
    for (int i = 0; i < 3; i++) c[k++] = b[i];
    c[k++] = ds4_sampler_gpu_encoded_count();
    for (int i = 0; i < 9; i++) c[k++] = g[i];
    c[k++] = pe[0];
    c[k++] = pe[1];
    c[k++] = pe[2];
    uint64_t drops = 0;
    for (int i = 3; i < BENCH_PE_STATS; i++) drops += pe[i];
    c[k++] = drops;
    for (int i = 0; i < 3; i++) c[k++] = br[i];
    assert(k == BENCH_JOB_COUNTERS);
}

static int run_jobs(const bench_config *cfg, ds4_engine *engine, ds4_session *session) {
    FILE *jf = fopen(cfg->jobs_path, "rb");
    FILE *tok = cfg->token_csv_path ? fopen(cfg->token_csv_path, "wb") : NULL;
    char jobs_out[PATH_MAX];
    snprintf(jobs_out, sizeof(jobs_out), "%s.jobs.csv", cfg->token_csv_path ? cfg->token_csv_path : "jobs");
    FILE *jo = cfg->token_csv_path ? fopen(jobs_out, "wb") : NULL;
    if (!jf || !tok || !jo) {
        fprintf(stderr, "ds4-bench: --jobs needs a readable job file and --token-csv\n");
        if (jf) fclose(jf);
        if (tok) fclose(tok);
        if (jo) fclose(jo);
        return 1;
    }
    fprintf(tok, "job,label,index,token,sel0_ns,eval0_ns,eval1_ns\n");
    fprintf(jo, "job,verb,label,cache,env,mode,temp,top_p,top_k,min_p,seed,warm,tokens,eos,"
                "setup_ns,restore_ns,first_sel0_ns,last_eval1_ns,decode_tps,pos,prefix_fnv64,logits_fnv64,state_fnv64,"
                "engram_add_tg,irq_tab,blend_router,blend_engram,blend_indexer,"
                "sel_encoded,sel_decided,sel_mass_short,sel_ambiguous,sel_overflow,sel_invalid,sel_zero,sel_unbound,"
                "sel_settings,sel_stale,pe_prep,pe_used,pe_pending,pe_drop,gpu_bridge,gpu_bridge_retired,gpu_bridge_failed,"
                "pe_drops\n");
    const int eos = ds4_token_eos(engine);
    const int logits_cap = 1 << 18;
    float *logits = malloc((size_t)logits_cap * sizeof(float));
    uint64_t *ns = NULL;
    int ns_cap = 0;
    bench_fixture fx = {0};
    ds4_session_snapshot snap = {0};
    bool have_snap = false;
    char cache[PATH_MAX] = "-";
    /* Lineage: the snapshot the session's current frontier came from (load or
     * save), so a longer save records which prefix it extended. */
    char lin_path[PATH_MAX] = "";
    unsigned long long lin_hash = 0;
    int lin_depth = 0;
    char err[256], line[4096];
    int rc = 0, job = 0;
    for (;;) {
        if (rc != 0) break;
        if (!fgets(line, sizeof(line), jf)) {
            /* Resident: wait for appended jobs; a "quit" line ends the runner. */
            clearerr(jf);
            struct timespec pause = {0, 200000000};
            nanosleep(&pause, NULL);
            continue;
        }
        if (!strncmp(line, "quit", 4)) break;
        char *kv[64];
        int n = 0;
        for (char *t = strtok(line, " \t\r\n"); t && n < 64; t = strtok(NULL, " \t\r\n")) kv[n++] = t;
        if (n == 0 || kv[0][0] == '#') continue;
        const char *verb = kv[0];
        char env_desc[2048] = "";
        uint64_t setup_ns = 0, restore_ns = 0;
        job++;
        if (!strcmp(verb, "save") || !strcmp(verb, "load")) {
            const char *path = job_get(kv, n, "cache");
            char meta_path[PATH_MAX], ident[8192];
            if (!path) { fprintf(stderr, "ds4-bench: job %d: %s needs cache=\n", job, verb); rc = 1; break; }
            snprintf(meta_path, sizeof(meta_path), "%s.meta", path);
            const int ident_ok = cache_identity(cfg, engine, session, ident, sizeof(ident));
            const uint64_t t0 = bench_now_ns();
            const ds4_tokens *all = job_fixture(engine, &fx, job_get(kv, n, "fixture"));
            const char *d = job_get(kv, n, "depth");
            const int depth = d ? atoi(d) : 0;
            if (!strcmp(verb, "save")) {
                const char *off = NULL;
                for (char **e = environ; *e && !off; e++)
                    if (!strncmp(*e, "DS4_METAL_DISABLE_V41_", 22)) off = *e;
                if (!all || depth <= 0 || depth > all->len || !ident_ok || off) {
                    fprintf(stderr, "ds4-bench: job %d: save needs fixture= and 0 < depth <= tokens, "
                                    "verifiable model/drafter digests and the default prefill path (%s)\n",
                            job, off ? off : !ident_ok ? "unverifiable digest" : "bad fixture/depth");
                    rc = 1;
                    break;
                }
                ds4_tokens prefix = { .v = all->v, .len = depth, .cap = depth };
                int base_len = 0;   /* > 0: this save extends the session's current frontier */
                {
                    const ds4_tokens *cur = ds4_session_tokens(session);
                    if (cur && cur->len > 0 && cur->len < depth &&
                        !memcmp(cur->v, all->v, (size_t)cur->len * sizeof(int))) base_len = cur->len;
                }
                uint64_t c0[BENCH_JOB_COUNTERS], c1[BENCH_JOB_COUNTERS], pe[BENCH_PE_STATS];
                job_counters(c0, pe);
                if (ds4_session_sync(session, &prefix, err, sizeof(err)) != 0) {
                    fprintf(stderr, "ds4-bench: job %d: prefill failed: %s\n", job, err);
                    rc = 1;
                    break;
                }
                /* the decode-only paths (Engram gate form .. selector) must not run in a prefix producer */
                job_counters(c1, pe);
                int dirty = -1;
                for (int i = 0; i < 6 && dirty < 0; i++) if (c1[i] != c0[i]) dirty = i;
                if (dirty >= 0) {
                    static const char *names[] = {"256-thread Engram gates", "fused index-query ropes",
                        "router blends", "Engram blends", "indexer blends", "GPU selector passes"};
                    fprintf(stderr, "ds4-bench: job %d: prefill dispatched %llu %s; not saving\n", job,
                            (unsigned long long)(c1[dirty] - c0[dirty]), names[dirty]);
                    rc = 1;
                    break;
                }
                const uint64_t t1 = bench_now_ns();
                /* Format 4: payload then meta, each tmp+fsync+rename, never over an
                 * existing snapshot; the meta's last line "end" marks completeness. */
                char ident4[8192], serr[256], canary[96] = "";
                const char *cv = job_get(kv, n, "canary");
                (void)cache_identity4(cfg, engine, session, ident4, sizeof(ident4));
                ds4_session_snapshot fs = {0};
                if (ds4_session_save_snapshot(session, &fs, err, sizeof(err)) != 0 ||
                    ds4_ss_write_file_atomic(path, fs.ptr, (size_t)fs.len, false, serr, sizeof(serr)) != 0) {
                    fprintf(stderr, "ds4-bench: job %d: cannot write %s: %s\n", job, path,
                            fs.len ? serr : err);
                    ds4_session_snapshot_free(&fs);
                    rc = 1;
                    break;
                }
                const unsigned long long ph = ds4_ss_hash_bytes(fs.ptr, (size_t)fs.len);
                const unsigned long long pbytes = fs.len;
                ds4_session_snapshot_free(&fs);
                const uint64_t t2w = bench_now_ns();
                if (cv && atoi(cv) > 0 && bench_canary(engine, all, atoi(cv), canary, sizeof(canary)) != 0) {
                    fprintf(stderr, "ds4-bench: job %d: canary prefill failed\n", job);
                    rc = 1;
                    break;
                }
                const unsigned long long lg = bench_fnv_logits(session, logits, logits_cap);
                size_t mcap = 65536, mn = 0;
                char *mt = malloc(mcap);
                if (!mt) { rc = 1; break; }
#define MT_ADD(...) do { if (mn < mcap) mn += (size_t)snprintf(mt + mn, mcap - mn, __VA_ARGS__); } while (0)
                MT_ADD("%s", ident4);
                if (canary[0]) MT_ADD("canary %s\n", canary);
                MT_ADD("fixture %s\ndepth %d\nprefix_fnv64 %016llx\nlogits_fnv64 %016llx\n"
                       "prefill_ns %llu\nsave_ns %llu\npayload_bytes %llu\npayload_hash %016llx\n",
                       fx.path, depth, (unsigned long long)bench_fnv_tokens(prefix.v, depth), lg,
                       (unsigned long long)(t1 - t0), (unsigned long long)(t2w - t1), pbytes, ph);
                if (base_len > 0 && lin_path[0] && lin_depth == base_len)
                    MT_ADD("base %s %016llx %d\n", lin_path, lin_hash, lin_depth);
                else
                    MT_ADD("base - (fresh prefill from %d tokens)\n", base_len);
                for (char **e = environ; *e; e++)   /* provenance only, not compared */
                    if (!strncmp(*e, "DS4_", 4)) MT_ADD("env %s\n", *e);
                MT_ADD("end\n");
#undef MT_ADD
                if (mn >= mcap || ds4_ss_write_file_atomic(meta_path, mt, mn, false, serr, sizeof(serr)) != 0) {
                    fprintf(stderr, "ds4-bench: job %d: cannot write %s: %s\n", job, meta_path,
                            mn >= mcap ? "metadata too long" : serr);
                    free(mt);
                    rc = 1;
                    break;
                }
                free(mt);
                snprintf(lin_path, sizeof(lin_path), "%s", path);
                lin_hash = ph;
                lin_depth = depth;
            } else {
                /* Outer compatibility check before the engine payload's own: the
                 * producer signature, then the job's own fixture/depth identity. */
                const char *ev = job_get(kv, n, "expect");
                const bool expect_reject = ev && !strcmp(ev, "reject");
                const char *why = NULL;
                char *meta = NULL;
                unsigned long long want_prefix = 0, want_logits = 0;
                unsigned long long load_hash = 0;   /* verified payload hash (format 4; format 3 has none) */
                FILE *mp = fopen(meta_path, "rb");
                if (mp) { fclose(mp); meta = read_file(meta_path); }
                if (!all || depth <= 0 || depth > all->len) why = "load needs fixture= and 0 < depth <= tokens";
                else if (!ident_ok) why = "unverifiable model/drafter digest";
                else if (!meta) why = "no metadata (incomplete snapshot)";
                else if (!strncmp(meta, "format " DS4_SS_FORMAT "\n", strlen("format " DS4_SS_FORMAT "\n"))) {
                    /* Format 4: compatibility by category, completeness, payload integrity. */
                    static char why4[640];
                    char ident4[8192], canary[96] = "";
                    ds4_ss_meta have_m = {0}, need_m = {0};
                    (void)cache_identity4(cfg, engine, session, ident4, sizeof(ident4));
                    const bool pm = ds4_ss_meta_parse(meta, &have_m) == 0;
                    const char *hc = pm ? ds4_ss_meta_get(&have_m, "canary") : NULL;
                    size_t il = strlen(ident4);
                    if (hc && atoi(hc) > 0 && bench_canary(engine, all, atoi(hc), canary, sizeof(canary)) == 0)
                        snprintf(ident4 + il, sizeof(ident4) - il, "canary %s\n", canary);
                    else if (hc)   /* compat then refuses the snapshot as UNVERIFIABLE */
                        fprintf(stderr, "ds4-bench: job %d: canary '%s' could not be recomputed\n", job, hc);
                    const bool nm = ds4_ss_meta_parse(ident4, &need_m) == 0;
                    ds4_ss_verdict vd = pm && nm ? ds4_ss_compat(&have_m, &need_m, why4, sizeof(why4)) : DS4_SS_INCOMPLETE;
                    uint64_t want_b = 0, want_h = 0, got_b = 0, got_h = 0;
                    if (vd == DS4_SS_OK) {
                        if (!ds4_ss_meta_get_u64(&have_m, "payload_bytes", 10, &want_b) ||
                            !ds4_ss_meta_get_u64(&have_m, "payload_hash", 16, &want_h)) {
                            vd = DS4_SS_INCOMPLETE;
                            snprintf(why4, sizeof(why4), "metadata lacks payload_bytes/payload_hash");
                        } else if (ds4_ss_hash_file(path, &got_b, &got_h, why4, sizeof(why4)) != 0 || got_b != want_b) {
                            vd = DS4_SS_INCOMPLETE;
                            if (!why4[0] || got_b != want_b)
                                snprintf(why4, sizeof(why4), "payload is %llu bytes, metadata says %llu",
                                         (unsigned long long)got_b, (unsigned long long)want_b);
                        } else if (got_h != want_h) {
                            vd = DS4_SS_CORRUPT;
                            snprintf(why4, sizeof(why4), "payload hash %016llx, metadata says %016llx",
                                     (unsigned long long)got_h, (unsigned long long)want_h);
                        }
                    }
                    if (vd != DS4_SS_OK) {
                        static char whyv[720];
                        snprintf(whyv, sizeof(whyv), "%s: %s", ds4_ss_verdict_name(vd), why4);
                        why = whyv;
                    } else {
                        uint64_t dv = 0;
                        if (!ds4_ss_meta_get_u64(&have_m, "prefix_fnv64", 16, &dv)) why = "metadata lacks prefix_fnv64";
                        else want_prefix = dv;
                        if (!why && ds4_ss_meta_get_u64(&have_m, "logits_fnv64", 16, &dv)) want_logits = dv;
                        else if (!why) why = "metadata lacks logits_fnv64";
                        if (!why && (!ds4_ss_meta_get_u64(&have_m, "depth", 10, &dv) || (int)dv != depth ||
                                     want_prefix != bench_fnv_tokens(all->v, depth)))
                            why = "cache holds a different fixture prefix or depth";
                        if (!why) load_hash = got_h;
                    }
                    ds4_ss_meta_free(&have_m);
                    ds4_ss_meta_free(&need_m);
                } else if (strncmp(meta, ident, strlen(ident)) != 0) {
                    why = "producer signature differs from this engine/session";
                    size_t k = 0;   /* name the first differing signature line */
                    while (meta[k] && meta[k] == ident[k]) k++;
                    while (k > 0 && ident[k - 1] != '\n') k--;
                    fprintf(stderr, "ds4-bench:   cache has: %.*s\n", (int)strcspn(meta + k, "\n"), meta + k);
                    fprintf(stderr, "ds4-bench:   need:      %.*s\n", (int)strcspn(ident + k, "\n"), ident + k);
                } else {
                    const char *p = strstr(meta, "\nprefix_fnv64 ");
                    const char *q = strstr(meta, "\nlogits_fnv64 ");
                    const char *dp = strstr(meta, "\ndepth ");
                    if (p) want_prefix = strtoull(p + 14, NULL, 16);
                    if (q) want_logits = strtoull(q + 14, NULL, 16);
                    if (!p || !q || !dp || atoi(dp + 7) != depth || want_prefix != bench_fnv_tokens(all->v, depth))
                        why = "cache holds a different fixture prefix or depth";
                }
                free(meta);
                if (why || expect_reject) {
                    fprintf(stderr, "ds4-bench: job %d: cache %s %s (%s)\n", job,
                            !why ? "ACCEPTED, expected a rejection," : expect_reject ? "REJECT (expected)" : "MISS",
                            path, why ? why : "signature and fixture/depth match");
                    if (!why || !expect_reject) { rc = 1; break; }
                    fprintf(jo, "%d,load-reject,,%s,,,,,,,,,0,0,0,0,0,0,0,0,,,", job, path);
                    for (int i = 0; i < BENCH_JOB_COUNTERS; i++) fputs(",0", jo);
                    fputs(",\n", jo);
                    fflush(jo);
                    continue;
                }
                FILE *fp = fopen(path, "rb");
                long bytes = -1;
                if (fp && fseek(fp, 0, SEEK_END) == 0) bytes = ftell(fp);
                if (!fp || bytes <= 0 || fseek(fp, 0, SEEK_SET) != 0 ||
                    ds4_session_load_payload(session, fp, (uint64_t)bytes, err, sizeof(err)) != 0) {
                    fprintf(stderr, "ds4-bench: job %d: cache load %s failed: %s\n", job, path, fp ? err : strerror(errno));
                    if (fp) fclose(fp);
                    rc = 1;
                    break;
                }
                fclose(fp);
                const ds4_tokens *have = ds4_session_tokens(session);
                const unsigned long long got_prefix = bench_fnv_tokens(have->v, have->len);
                const unsigned long long got_logits = bench_fnv_logits(session, logits, logits_cap);
                if (have->len != depth || memcmp(have->v, all->v, (size_t)depth * sizeof(int)) != 0 ||
                    got_prefix != want_prefix || got_logits != want_logits) {
                    fprintf(stderr, "ds4-bench: job %d: cache %s restored a different frontier "
                                    "(prefix %016llx/%016llx logits %016llx/%016llx)\n",
                            job, path, got_prefix, want_prefix, got_logits, want_logits);
                    rc = 1;
                    break;
                }
                fprintf(stderr, "ds4-bench: job %d: cache HIT %s (%d tokens)\n", job, path, have->len);
                snprintf(lin_path, sizeof(lin_path), "%s", path);
                lin_hash = load_hash;
                lin_depth = depth;
            }
            setup_ns = bench_now_ns() - t0;
            const uint64_t t2 = bench_now_ns();
            ds4_session_snapshot_free(&snap);
            memset(&snap, 0, sizeof(snap));
            if (ds4_session_save_snapshot(session, &snap, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-bench: job %d: in-memory snapshot failed: %s\n", job, err);
                rc = 1;
                break;
            }
            restore_ns = bench_now_ns() - t2;
            have_snap = true;
            snprintf(cache, sizeof(cache), "%s", path);
            const ds4_tokens *have = ds4_session_tokens(session);
            fprintf(jo, "%d,%s,,%s,,,,,,,,,0,0,%llu,%llu,0,0,0,%d,%016llx,%016llx,", job, verb, cache,
                    (unsigned long long)setup_ns, (unsigned long long)restore_ns, ds4_session_pos(session),
                    (unsigned long long)bench_fnv_tokens(have->v, have->len),
                    (unsigned long long)bench_fnv_logits(session, logits, logits_cap));
            for (int i = 0; i < BENCH_JOB_COUNTERS; i++) fputs(",0", jo);
            fputs(",\n", jo);
            fflush(jo);
            continue;
        }
        if (strcmp(verb, "run") != 0 || !have_snap) {
            fprintf(stderr, "ds4-bench: job %d: unknown verb %s or run before save/load\n", job, verb);
            rc = 1;
            break;
        }
        const char *label = job_get(kv, n, "label");
        if (label && strpbrk(label, ",\"")) {
            fprintf(stderr, "ds4-bench: job %d: label %s contains a CSV delimiter\n", job, label);
            rc = 1;
            break;
        }
        const char *v;
        const int tokens = (v = job_get(kv, n, "tokens")) ? atoi(v) : 64;
        const uint64_t seed = (v = job_get(kv, n, "seed")) ? strtoull(v, NULL, 10) : 1;
        const char *mode = (v = job_get(kv, n, "mode")) ? v : "sampled";
        const float temp = (v = job_get(kv, n, "temp")) ? (float)atof(v) : 1.0f;
        const float top_p = (v = job_get(kv, n, "top_p")) ? (float)atof(v) : 0.95f;
        const int top_k = (v = job_get(kv, n, "top_k")) ? atoi(v) : 0;
        const float min_p = (v = job_get(kv, n, "min_p")) ? (float)atof(v) : 0.0f;
        const bool warm = (v = job_get(kv, n, "warm")) && atoi(v) != 0;
        /* dump=PATH (diagnostic, not a timing job): the full logit vector each token is
         * sampled from, appended as raw float32 vectors of ds4_session_copy_logits. */
        FILE *dump = (v = job_get(kv, n, "dump")) ? fopen(v, "wb") : NULL;
        if (job_get(kv, n, "dump") && !dump) {
            fprintf(stderr, "ds4-bench: job %d: cannot open dump=%s\n", job, job_get(kv, n, "dump"));
            rc = 1;
            break;
        }
        const bool teacher = !strcmp(mode, "teacher"), greedy = !strcmp(mode, "greedy");
        const ds4_tokens *forced = teacher ? job_fixture(engine, &fx, job_get(kv, n, "fixture")) : NULL;
        if (tokens <= 0 || (teacher && (!forced || forced->len < ds4_session_pos(session) + tokens))) {
            fprintf(stderr, "ds4-bench: job %d: bad tokens= or teacher fixture\n", job);
            if (dump) fclose(dump);
            rc = 1;
            break;
        }
        /* Quiescent boundary: restore the frontier, then switch the arm. */
        const uint64_t t0 = bench_now_ns();
        if (ds4_session_load_snapshot(session, &snap, err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4-bench: job %d: restore failed: %s\n", job, err);
            if (dump) fclose(dump);
            rc = 1;
            break;
        }
        restore_ns = bench_now_ns() - t0;
        for (int i = 1; i < n; i++) {
            if (!strncmp(kv[i], "set=", 4) && strchr(kv[i] + 4, '=')) {
                char name[128];
                const char *eq = strchr(kv[i] + 4, '=');
                snprintf(name, sizeof(name), "%.*s", (int)(eq - kv[i] - 4), kv[i] + 4);
                setenv(name, eq + 1, 1);
            } else if (!strncmp(kv[i], "unset=", 6)) {
                unsetenv(kv[i] + 6);
            } else {
                continue;
            }
            const size_t used = strlen(env_desc);
            snprintf(env_desc + used, sizeof(env_desc) - used, "%s%s", used ? ";" : "", kv[i]);
        }
        if (tokens > ns_cap) {
            free(ns);
            ns = calloc((size_t)tokens * 4u, sizeof(ns[0]));
            ns_cap = ns ? tokens : 0;
            if (!ns) { if (dump) fclose(dump); rc = 1; break; }
        }
        const int pos0 = ds4_session_pos(session);
        uint64_t c0[BENCH_JOB_COUNTERS], c1[BENCH_JOB_COUNTERS], pe0[BENCH_PE_STATS], pe1[BENCH_PE_STATS];
        job_counters(c0, pe0);
        uint64_t rng = seed;
        int done = 0;
        bool eos_hit = false;
        while (done < tokens) {
            if (ds4_session_pos(session) + 1 >= ds4_session_ctx(session)) { rc = 1; break; }
            if (dump) {
                const int nl = ds4_session_copy_logits(session, logits, logits_cap);
                if (nl <= 0 || fwrite(logits, sizeof(float), (size_t)nl, dump) != (size_t)nl) { rc = 1; break; }
            }
            const uint64_t sel0 = bench_now_ns();
            const int t = teacher ? forced->v[pos0 + done] :
                greedy ? ds4_session_argmax_excluding(session, eos) :
                ds4_session_sample(session, temp, top_k, top_p, min_p, &rng);
            if (!teacher && !greedy && t == eos) { eos_hit = true; break; }
            const uint64_t eval0 = bench_now_ns();
            if (t < 0 || ds4_session_eval(session, t, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-bench: job %d: decode failed: %s\n", job, err);
                rc = 1;
                break;
            }
            const uint64_t eval1 = bench_now_ns();
            uint64_t *r = ns + (size_t)done * 4u;
            r[0] = (uint64_t)t; r[1] = sel0; r[2] = eval0; r[3] = eval1;
            done++;
        }
        if (dump && fclose(dump) != 0) rc = 1;
        if (rc != 0) break;
        job_counters(c1, pe1);
        char pestr[160] = "";   /* drops in this job by reason (foreign:pos:latch:state:error:prepare:off:invalid:stepfail:release) */
        for (int i = 3; i < BENCH_PE_STATS; i++) {
            const size_t pn = strlen(pestr);
            snprintf(pestr + pn, sizeof pestr - pn, "%s%llu", i > 3 ? ":" : "", (unsigned long long)(pe1[i] - pe0[i]));
        }
        for (int i = 0; i < done; i++) {
            const uint64_t *r = ns + (size_t)i * 4u;
            fprintf(tok, "%d,%s,%d,%llu,%llu,%llu,%llu\n", job, label ? label : "", i,
                    (unsigned long long)r[0], (unsigned long long)r[1],
                    (unsigned long long)r[2], (unsigned long long)r[3]);
        }
        const uint64_t first = done ? ns[1] : 0, last = done ? ns[(size_t)(done - 1) * 4u + 3u] : 0;
        const ds4_tokens *have = ds4_session_tokens(session);
        /* state=1: digest of the whole session state after the run (KV, draft
         * rows/rings, logits), outside timing, to prove arms leave equal state. */
        char state_hex[24] = "";
        if ((v = job_get(kv, n, "state")) && atoi(v)) {
            ds4_session_snapshot st_snap = {0};
            if (ds4_session_save_snapshot(session, &st_snap, err, sizeof(err)) == 0) {
                uint64_t h = 1469598103934665603ull;
                for (uint64_t i = 0; i < st_snap.len; i++) { h ^= st_snap.ptr[i]; h *= 1099511628211ull; }
                snprintf(state_hex, sizeof(state_hex), "%016llx", (unsigned long long)h);
            }
            ds4_session_snapshot_free(&st_snap);
        }
        fprintf(jo, "%d,run,%s,%s,%s,%s,%g,%g,%d,%g,%llu,%d,%d,%d,0,%llu,%llu,%llu,%.4f,%d,%016llx,%016llx,%s",
                job, label ? label : "", cache, env_desc, mode, (double)temp, (double)top_p, top_k,
                (double)min_p, (unsigned long long)seed, warm ? 1 : 0, done, eos_hit ? 1 : 0,
                (unsigned long long)restore_ns, (unsigned long long)first, (unsigned long long)last,
                last > first ? (double)done * 1e9 / (double)(last - first) : 0.0,
                ds4_session_pos(session), (unsigned long long)bench_fnv_tokens(have->v, pos0),
                (unsigned long long)bench_fnv_logits(session, logits, logits_cap), state_hex);
        for (int i = 0; i < BENCH_JOB_COUNTERS; i++)   /* pe_pending (index 17) is the level after the run */
            fprintf(jo, ",%llu", (unsigned long long)(i == 17 ? c1[i] : c1[i] - c0[i]));
        fprintf(jo, ",%s\n", pestr);
        fflush(tok);
        fflush(jo);
    }
    {   /* process totals of the pre-encoded first buffers */
        uint64_t pt[BENCH_PE_STATS];
        ds4_preencode_stats(pt, BENCH_PE_STATS);
        fprintf(stderr, "ds4-bench: pre-encode totals prepared=%llu used=%llu held=%llu dropped=", (unsigned long long)pt[0],
                (unsigned long long)pt[1], (unsigned long long)pt[2]);
        for (int i = 3; i < BENCH_PE_STATS; i++) fprintf(stderr, "%s%llu", i > 3 ? ":" : "", (unsigned long long)pt[i]);
        fprintf(stderr, "\n");
    }
    fclose(jf);
    fclose(tok);
    fclose(jo);
    free(ns);
    free(logits);
    ds4_tokens_free(&fx.tokens);
    ds4_session_snapshot_free(&snap);
    return rc;
}

static void close_engine(ds4_engine *engine, ds4_tp *tp) {
    if (tp) ds4_tp_send_stop(tp);
    ds4_engine_close(engine);
    ds4_tp_free(tp);
}

int main(int argc, char **argv) {
    bench_config cfg = parse_options(argc, argv);

    /* Hint the packer at the largest ctx this bench run will exercise
     * so per-layer KV bytes are priced for the real session size, not
     * a stale 4096 default. Single-tier and CPU paths ignore this. */
    int placement_ctx_hint = cfg.ctx_max;
    if (cfg.ctx_alloc > placement_ctx_hint) placement_ctx_hint = cfg.ctx_alloc;

    ds4_gpu_config gpu_cfg = {0};
    bool skip_cuda = false;
    const bool have_gpu_config = cfg.gpu_vram_arg || cfg.gpu_devices_arg;
    if (have_gpu_config) {
        char gpu_err[256];
        if (parse_gpu_vram_arg(cfg.gpu_vram_arg, cfg.gpu_devices_arg,
                               &gpu_cfg, &skip_cuda,
                               gpu_err, sizeof(gpu_err)) != 0) {
            fprintf(stderr, "ds4-bench: %s\n", gpu_err);
            return 2;
        }
        cfg.backend = skip_cuda ? DS4_BACKEND_CPU : DS4_BACKEND_CUDA;
    }

    ds4_engine_options opt = {
        .model_path = cfg.model_path,
        .mtp_path = cfg.mtp_path,
        .v41_sidecar = {cfg.v41_sidecar[0], cfg.v41_sidecar[1], cfg.v41_sidecar[2]},
        .backend = cfg.backend,
        .n_threads = cfg.threads,
        .context_size = cfg.ctx_alloc,
        .prefill_chunk = cfg.prefill_chunk,
        .ssd_streaming_cache_experts = cfg.ssd_streaming_cache_experts,
        .ssd_streaming_cache_bytes = cfg.ssd_streaming_cache_bytes,
        .ssd_streaming_full_layers = cfg.ssd_streaming_full_layers,
        .ssd_streaming_preload_experts = cfg.ssd_streaming_preload_experts,
        .simulate_used_memory_bytes = cfg.simulate_used_memory_bytes,
        .power_percent = cfg.power_percent,
        .warm_weights = cfg.warm_weights,
        .quality = cfg.quality,
        .dspark = cfg.dspark,
        .dspark_confidence_threshold = cfg.dspark_confidence_threshold,
        .dspark_confidence_threshold_set = cfg.dspark_confidence_threshold_set,
        .cuda_tensor_parallel = cfg.cuda_tensor_parallel,
        .ssd_streaming = cfg.ssd_streaming,
        .ssd_streaming_cold = cfg.ssd_streaming_cold,
        .ssd_streaming_full_layers_set = cfg.ssd_streaming_full_layers_set,
        .expert_profile_path = cfg.expert_profile_path,
        .distributed = cfg.dist,
        .tp = cfg.tp,
    };
    char dist_err[256];
    if (ds4_dist_prepare_engine_options(&cfg.dist, &opt, dist_err, sizeof(dist_err)) != 0) {
        fprintf(stderr, "ds4-bench: %s\n", dist_err);
        return 2;
    }
    char tp_err[256];
    if (!ds4_tp_validate_engine_options(&opt, tp_err, sizeof(tp_err))) {
        fprintf(stderr, "ds4-bench: %s\n", tp_err);
        return 2;
    }
    ds4_engine *engine = NULL;
    if (have_gpu_config && !skip_cuda) {
        const bool was_auto =
            (cfg.gpu_vram_arg && !strcmp(cfg.gpu_vram_arg, "auto")) ||
            (!cfg.gpu_vram_arg && cfg.gpu_devices_arg);
        char layout[256];
        if (format_gpu_layout_line(&gpu_cfg, was_auto,
                                   layout, sizeof(layout)) > 0) {
            fprintf(stdout, "%s\n", layout);
            fflush(stdout);
        }
        if (ds4_engine_create_with_gpu_config(
                &engine, &opt, &gpu_cfg) != 0) return 1;
    } else if (ds4_engine_open(&engine, &opt) != 0) {
        return 1;
    }
    ds4_tp *tp_leader = NULL;
    if (cfg.tp.role == DS4_TP_LEADER) {
        ds4_tp_identity tp_id = {
            .gguf_bytes = ds4_engine_model_bytes(engine),
            .model_id = (uint32_t)ds4_engine_model_id(engine),
            .n_layer = (uint32_t)ds4_engine_layer_count(engine),
            .n_embd = (uint32_t)ds4_engine_embd_dim(engine),
            .n_vocab = (uint32_t)ds4_engine_vocab_size(engine),
            .quant_bits = (uint32_t)ds4_engine_routed_quant_bits(engine),
            .ctx_size = (uint32_t)cfg.ctx_alloc,
        };
        ds4_engine_tp_gate_schedule(engine,
                                    &tp_id.gate_slot_start,
                                    &tp_id.gate_slot_step,
                                    &tp_id.gates_per_token,
                                    tp_id.gate_slot_mask);
        if (!ds4_tp_create(&tp_leader,
                           &cfg.tp,
                           &tp_id,
                           tp_err,
                           sizeof(tp_err)) ||
            !ds4_engine_tp_bind(engine, tp_leader, tp_err, sizeof(tp_err))) {
            fprintf(stderr, "ds4-bench: %s\n", tp_err);
            close_engine(engine, tp_leader);
            return 1;
        }
    }
    log_context_memory(opt.backend,
                       cfg.ctx_alloc,
                       ds4_engine_prefill_chunk(engine),
                       cfg.ssd_streaming);

    if (cfg.jobs_path) {
        ds4_session *js = NULL;
        if (ds4_session_create(&js, engine, cfg.ctx_alloc) != 0) {
            fprintf(stderr, "ds4-bench: failed to create session\n");
            close_engine(engine, tp_leader);
            return 1;
        }
        const int jrc = run_jobs(&cfg, engine, js);
        ds4_session_free(js);
        close_engine(engine, tp_leader);
        return jrc;
    }

    char *text = read_file(cfg.prompt_path ? cfg.prompt_path : cfg.chat_prompt_path);
    ds4_tokens prompt = {0};
    if (cfg.chat_prompt_path) {
        ds4_encode_chat_prompt(engine, cfg.system, text, DS4_THINK_NONE, &prompt);
    } else {
        ds4_tokenize_text(engine, text, &prompt);
    }
    free(text);

    const int prompt_tokens_needed = cfg.ctx_max +
        (cfg.teacher_forced_decode ? cfg.gen_tokens : 0);
    if (prompt.len < prompt_tokens_needed) {
        fprintf(stderr,
                "ds4-bench: prompt has %d tokens, need at least %d%s\n",
                prompt.len,
                prompt_tokens_needed,
                cfg.teacher_forced_decode ? " for teacher-forced decode" : "");
        ds4_tokens_free(&prompt);
        close_engine(engine, tp_leader);
        return 1;
    }

    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg.ctx_alloc) != 0) {
        fprintf(stderr, "ds4-bench: failed to create session\n");
        ds4_tokens_free(&prompt);
        close_engine(engine, tp_leader);
        return 1;
    }
    if (cfg.dist.role == DS4_DISTRIBUTED_COORDINATOR &&
        wait_distributed_route(session) != 0)
    {
        ds4_session_free(session);
        ds4_tokens_free(&prompt);
        close_engine(engine, tp_leader);
        return 1;
    }
    maybe_warn_distributed_step_shape(&cfg, session);

    FILE *out = stdout;
    if (cfg.csv_path) {
        out = fopen(cfg.csv_path, "wb");
        if (!out) {
            fprintf(stderr, "ds4-bench: failed to open %s: %s\n", cfg.csv_path, strerror(errno));
            ds4_session_free(session);
            ds4_tokens_free(&prompt);
            close_engine(engine, tp_leader);
            return 1;
        }
    }
    fprintf(out, "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_first_ms,gen_steady_tokens,gen_steady_tps,kvcache_bytes\n");
    fflush(out);

    const int eos = ds4_token_eos(engine);
    const bool distributed =
        cfg.dist.role == DS4_DISTRIBUTED_COORDINATOR ||
        cfg.tp.role == DS4_TP_LEADER;
    const bool speculative = cfg.dspark && ds4_engine_mtp_draft_tokens(engine) > 1;
    if (cfg.sampled && speculative) {
        fprintf(stderr, "ds4-bench: --temp needs serial decode (omit --dspark)\n");
        if (out != stdout) fclose(out);
        ds4_session_free(session);
        ds4_tokens_free(&prompt);
        close_engine(engine, tp_leader);
        return 1;
    }
    if (cfg.dspark && !speculative) {
        fprintf(stderr, "ds4-bench: DSpark support model did not enable speculative decoding\n");
        if (out != stdout) fclose(out);
        ds4_session_free(session);
        ds4_tokens_free(&prompt);
        close_engine(engine, tp_leader);
        return 1;
    }
    if (speculative) {
        fprintf(stderr,
                "ds4-bench: DSpark enabled with draft width %d; frontier restoration uses session snapshots\n",
                ds4_engine_mtp_draft_tokens(engine));
    }
    /* --token-csv PATH: per-token timings (PATH) and one row per frontier (PATH.runs.csv) with the frontier prefix and
     * final-logit digests, so arms can be compared bit for bit. */
    FILE *token_csv = NULL, *runs_csv = NULL;
    if (cfg.token_csv_path) {
        char runs_path[PATH_MAX];
        snprintf(runs_path, sizeof(runs_path), "%s.runs.csv", cfg.token_csv_path);
        token_csv = fopen(cfg.token_csv_path, "wb");
        runs_csv = fopen(runs_path, "wb");
        if (!token_csv || !runs_csv) {
            fprintf(stderr, "ds4-bench: failed to open %s or %s\n", cfg.token_csv_path, runs_path);
            if (token_csv) fclose(token_csv);
            if (runs_csv) fclose(runs_csv);
            if (out != stdout) fclose(out);
            ds4_session_free(session);
            ds4_tokens_free(&prompt);
            close_engine(engine, tp_leader);
            return 1;
        }
        fprintf(token_csv, "frontier,index,token,sel0_ns,eval0_ns,eval1_ns\n");
        fprintf(runs_csv, "frontier,mode,temp,top_p,top_k,min_p,seed,tokens,eos,"
                          "first_sel0_ns,last_eval1_ns,decode_tps,gen_tps,prefix_fnv64,snapshot_bytes,logits_fnv64\n");
    }
    const char *mode = cfg.teacher_forced_decode ? "teacher" : cfg.sampled ? "sampled" :
        speculative ? "speculative" : "greedy";
    uint64_t *tok_ns = token_csv && cfg.gen_tokens > 0 ? calloc((size_t)cfg.gen_tokens * 4u, sizeof(tok_ns[0])) : NULL;
    const int logits_cap = 1 << 18;
    float *logits_buf = tok_ns ? malloc((size_t)logits_cap * sizeof(float)) : NULL;
    if (token_csv && cfg.gen_tokens > 0 && (!tok_ns || !logits_buf)) {
        fprintf(stderr, "ds4-bench: out of memory for --token-csv\n");
        free(tok_ns);
        free(logits_buf);
        fclose(token_csv);
        fclose(runs_csv);
        if (out != stdout) fclose(out);
        ds4_session_free(session);
        ds4_tokens_free(&prompt);
        close_engine(engine, tp_leader);
        return 1;
    }
    ds4_session_snapshot snap = {0};
    const uint64_t snapshot_max_bytes = bench_snapshot_max_bytes();
    bool warned_large_snapshot = false;
    char err[256];
    int previous = 0;
    int rc = 0;

    for (int frontier = cfg.ctx_start; ; frontier = next_frontier(&cfg, frontier)) {
        ds4_tokens prefix = {
            .v = prompt.v,
            .len = frontier,
            .cap = frontier,
        };

        const double prefill_t0 = bench_now_sec();
#if defined(DS4_BENCH_HAVE_CUDA_PROFILER)
        const bool cuda_profile_prefill =
            getenv("DS4_BENCH_CUDA_PROFILE_PREFILL") != NULL &&
            cudaProfilerStart && cudaProfilerStop;
        if (cuda_profile_prefill) (void)cudaProfilerStart();
#endif
        const int prefill_rc =
            ds4_session_sync(session, &prefix, err, sizeof(err));
#if defined(DS4_BENCH_HAVE_CUDA_PROFILER)
        if (cuda_profile_prefill) (void)cudaProfilerStop();
#endif
        if (prefill_rc != 0) {
            fprintf(stderr, "ds4-bench: prefill to %d failed: %s\n", frontier, err);
            rc = 1;
            break;
        }
        const double prefill_t1 = bench_now_sec();
        const double prefill_sec = prefill_t1 - prefill_t0;
        if (getenv("DS4_METAL_CB_TIMES"))
            fprintf(stderr, "ds4-bench: prefill window mono %.1f .. %.1f ms\n",
                    prefill_t0 * 1e3, prefill_t1 * 1e3);
        const int prefill_tokens = frontier - previous;

        if (write_frontier_logits_json(&cfg, engine, session, frontier, previous) != 0) {
            rc = 1;
            break;
        }

        const bool need_restore_after_generation =
            cfg.gen_tokens > 0 && frontier < cfg.ctx_max;
        bool have_snapshot = false;
        if (need_restore_after_generation && !distributed &&
            getenv("DS4_BENCH_DISABLE_SNAPSHOT") == NULL) {
            const uint64_t payload_bytes = ds4_session_payload_bytes(session);
            const bool large_snapshot_forced =
                getenv("DS4_BENCH_FORCE_SNAPSHOT") != NULL;
            if (payload_bytes > snapshot_max_bytes && !large_snapshot_forced) {
                if (!warned_large_snapshot) {
                    fprintf(stderr,
                            "ds4-bench: session payload snapshot is %.2f GiB, above the %.2f GiB benchmark limit; "
                            "replaying prefixes instead (set DS4_BENCH_FORCE_SNAPSHOT=1 to force snapshots)\n",
                            bytes_to_gib(payload_bytes),
                            bytes_to_gib(snapshot_max_bytes));
                    warned_large_snapshot = true;
                }
            } else if (payload_bytes > 0) {
                if (ds4_session_save_snapshot(session, &snap, err, sizeof(err)) != 0) {
                    fprintf(stderr, "ds4-bench: snapshot at %d failed: %s\n", frontier, err);
                    rc = 1;
                    break;
                }
                have_snapshot = true;
            }
        }

        uint64_t rng = cfg.seed;   /* the same sampler stream at every frontier */
        bool eos_hit = false;
        const double gen_t0 = bench_now_sec();
        double gen_first_sec = 0.0;
        double gen_steady_sec = 0.0;
        int gen_first_tokens = 0;
        int gen_done = 0;
        int *gen_token_buf = cfg.show_output && cfg.gen_tokens > 0
            ? malloc((size_t)cfg.gen_tokens * sizeof(gen_token_buf[0]))
            : NULL;
        int gen_token_count = 0;
        int cuda_profile_start = -1;
        int cuda_profile_tokens = 0;
        const char *cuda_profile_range =
            getenv("DS4_BENCH_CUDA_PROFILE_RANGE");
        if (cuda_profile_range &&
            (sscanf(cuda_profile_range, "%d:%d",
                    &cuda_profile_start, &cuda_profile_tokens) != 2 ||
             cuda_profile_start < 0 || cuda_profile_tokens <= 0)) {
            fprintf(stderr,
                    "ds4-bench: invalid DS4_BENCH_CUDA_PROFILE_RANGE=%s "
                    "(expected START_TOKEN:TOKEN_COUNT)\n",
                    cuda_profile_range);
            cuda_profile_start = -1;
            cuda_profile_tokens = 0;
        }
        bool generation_stop = false;
        while (gen_done < cfg.gen_tokens && !generation_stop) {
            if (ds4_session_pos(session) + 1 >= ds4_session_ctx(session)) {
                fprintf(stderr, "ds4-bench: generation would exceed allocated context at frontier %d\n", frontier);
                rc = 1;
                break;
            }
            const uint64_t sel0_ns = bench_now_ns();
            const int token = cfg.teacher_forced_decode
                ? prompt.v[frontier + gen_done]
                : cfg.sampled
                ? ds4_session_sample(session, cfg.temperature, cfg.top_k, cfg.top_p, cfg.min_p, &rng)
                : ds4_session_argmax_excluding(session, eos);
            if (cfg.sampled && token == eos) {
                eos_hit = true;
                break;
            }
            if (token < 0) {
                fprintf(stderr, "ds4-bench: failed to choose non-EOS token at frontier %d\n", frontier);
                rc = 1;
                break;
            }
            const uint64_t eval0_ns = bench_now_ns();
            const double token_t0 = bench_now_sec();
#if defined(DS4_BENCH_HAVE_CUDA_PROFILER)
            if (gen_done == cuda_profile_start && cudaProfilerStart) {
                (void)cudaProfilerStart();
            }
#endif
            int toks[17];
            int ntok = 1;
            if (speculative) {
                const double spec_t0 = bench_now_sec();
                ntok = ds4_session_eval_speculative_argmax(
                    session, token, cfg.gen_tokens - gen_done, eos,
                    toks, (int)(sizeof(toks) / sizeof(toks[0])),
                    err, sizeof(err));
                if (getenv("DS4_BENCH_SPEC_TRACE") && ntok > 0 && ntok < 17) {
                    static double call_ms[17];
                    static unsigned call_n[17];
                    static unsigned calls_total;
                    call_ms[ntok] += (bench_now_sec() - spec_t0) * 1e3;
                    call_n[ntok]++;
                    if (++calls_total % 32u == 0u) {
                        fprintf(stderr, "ds4-bench: spec calls %u:", calls_total);
                        for (int k = 1; k < 17; k++)
                            if (call_n[k])
                                fprintf(stderr, " %d-tok x%u avg %.1f ms", k, call_n[k], call_ms[k] / call_n[k]);
                        fprintf(stderr, "\n");
                    }
                }
                if (ntok < 0) {
                    fprintf(stderr, "ds4-bench: DSpark decode at frontier %d failed: %s\n", frontier, err);
                    rc = 1;
                    break;
                }
                if (ntok == 0) {
                    fprintf(stderr, "ds4-bench: DSpark decode at frontier %d accepted no tokens\n", frontier);
                    rc = 1;
                    break;
                }
            } else {
                toks[0] = token;
                if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
                    fprintf(stderr, "ds4-bench: decode at frontier %d failed: %s\n", frontier, err);
                    rc = 1;
                    break;
                }
            }
#if defined(DS4_BENCH_HAVE_CUDA_PROFILER)
            if (cuda_profile_start >= 0 &&
                gen_done + ntok >=
                    cuda_profile_start + cuda_profile_tokens &&
                cudaProfilerStop) {
                (void)cudaProfilerStop();
            }
#endif
            const double token_t1 = bench_now_sec();
            if (tok_ns && ntok == 1) {
                uint64_t *r = tok_ns + (size_t)gen_done * 4u;
                r[0] = (uint64_t)toks[0]; r[1] = sel0_ns; r[2] = eval0_ns; r[3] = bench_now_ns();
            }
            int cycle_tokens = 0;
            for (int j = 0; j < ntok && gen_done < cfg.gen_tokens; j++) {
                if (toks[j] == eos) {
                    generation_stop = true;
                    break;
                }
                if (gen_token_buf) gen_token_buf[gen_token_count++] = toks[j];
                gen_done++;
                cycle_tokens++;
            }
            if (gen_first_tokens == 0) {
                gen_first_sec = token_t1 - token_t0;
                gen_first_tokens = cycle_tokens;
            } else {
                gen_steady_sec += token_t1 - token_t0;
            }
        }
        const double gen_t1 = bench_now_sec();
        if (cfg.show_output && gen_token_buf && gen_token_count > 0) {
            fprintf(stderr, "ds4-bench: gen[ctx=%d] decoded text: \"", frontier);
            for (int i = 0; i < gen_token_count; i++) {
                size_t tlen = 0;
                char *txt = ds4_token_text(engine, gen_token_buf[i], &tlen);
                if (txt) {
                    fwrite(txt, 1, tlen, stderr);
                    free(txt);
                }
            }
            fprintf(stderr, "\"\n");
            fflush(stderr);
        }
        free(gen_token_buf);
        if (rc != 0) break;

        if (token_csv && tok_ns) {
            const bool timed = gen_done > 0 && !speculative;
            const uint64_t first = timed ? tok_ns[1] : 0, last = timed ? tok_ns[(size_t)(gen_done - 1) * 4u + 3u] : 0;
            for (int i = 0; timed && i < gen_done; i++) {
                const uint64_t *r = tok_ns + (size_t)i * 4u;
                fprintf(token_csv, "%d,%d,%llu,%llu,%llu,%llu\n", frontier, i, (unsigned long long)r[0],
                        (unsigned long long)r[1], (unsigned long long)r[2], (unsigned long long)r[3]);
            }
            fprintf(runs_csv, "%d,%s,%g,%g,%d,%g,%llu,%d,%d,%llu,%llu,%.4f,%.4f,%016llx,%llu,%016llx\n",
                    frontier, mode, (double)cfg.temperature, (double)cfg.top_p, cfg.top_k, (double)cfg.min_p,
                    (unsigned long long)cfg.seed, gen_done, eos_hit ? 1 : 0,
                    (unsigned long long)first, (unsigned long long)last,
                    last > first ? (double)gen_done * 1e9 / (double)(last - first) : 0.0,
                    gen_t1 > gen_t0 ? (double)gen_done / (gen_t1 - gen_t0) : 0.0,
                    (unsigned long long)bench_fnv_tokens(prompt.v, frontier),
                    (unsigned long long)(have_snapshot ? snap.len : 0),
                    (unsigned long long)bench_fnv_logits(session, logits_buf, logits_cap));
            fflush(token_csv);
            fflush(runs_csv);
        }

        if (!need_restore_after_generation) {
            /* Nothing later depends on the frontier state. */
        } else if (distributed || !have_snapshot) {
            if (ds4_session_sync(session, &prefix, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-bench: replay restore at %d failed: %s\n", frontier, err);
                rc = 1;
                break;
            }
        } else {
            if (ds4_session_load_snapshot(session, &snap, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-bench: restore at %d failed: %s\n", frontier, err);
                rc = 1;
                break;
            }
        }

        const double gen_sec = gen_t1 - gen_t0;
        const int gen_steady_tokens = gen_done > gen_first_tokens ?
            gen_done - gen_first_tokens : 0;
        fprintf(out,
                "%d,%d,%.2f,%d,%.2f,%.3f,%d,%.2f,%llu\n",
                frontier,
                prefill_tokens,
                prefill_sec > 0.0 ? (double)prefill_tokens / prefill_sec : 0.0,
                gen_done,
                gen_sec > 0.0 ? (double)gen_done / gen_sec : 0.0,
                gen_first_sec * 1000.0,
                gen_steady_tokens,
                gen_steady_sec > 0.0 ? (double)gen_steady_tokens / gen_steady_sec : 0.0,
                (unsigned long long)(have_snapshot ? snap.len : 0));
        fflush(out);

        previous = frontier;
        if (frontier >= cfg.ctx_max) break;
    }

    if (out != stdout) fclose(out);
    if (token_csv) fclose(token_csv);
    if (runs_csv) fclose(runs_csv);
    free(tok_ns);
    free(logits_buf);
    ds4_session_snapshot_free(&snap);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    close_engine(engine, tp_leader);
    return rc;
}
