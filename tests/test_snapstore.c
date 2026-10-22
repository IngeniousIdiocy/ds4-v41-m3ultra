/* CPU unit tests for ds4_snapstore (snapshot integrity + compatibility identity).
 * make test-snapstore   (no model, no GPU) */
#include "../ds4_snapstore.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int failures;
#define CHECK(cond, ...) do { if (!(cond)) { failures++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

static const char *IDENT =
    "format ds4-bench-prefill-cache-4\n"
    "contract v41-prefix-1\n"
    "payload_version 2\n"
    "model /a/model.gguf 501609136128 1792242799 4d16135d76ef1a71ae3109ab5fc02d3014fa806c2caed7a4cbe84e2f695f4344\n"
    "drafter /a/drafter.gguf 7966294016 1792242809 d3efe0b1f6b6fea54f1c58c5cbeb3fded5967b15b64476aed00c95806ad3e3e9\n"
    "armed 1\nctx 393216\nprefill_chunk 8192\nvocab 129280\nprefill_repr v41pf1 engram=f16-gguf indexer=f16-gguf router=f32-gguf\nprefill_values none\n"
    "numerics DS4_METAL_HC_STABLE=(unset)\n"
    "numerics DS4_METAL_KV_RAW_F32=(unset)\n"
    "canary 4096 0123456789abcdef\n";

static ds4_ss_verdict compat_of(const char *have_text, const char *need_text, char *why, size_t n) {
    ds4_ss_meta h, nd;
    ds4_ss_meta_parse(have_text, &h);
    ds4_ss_meta_parse(need_text, &nd);
    const ds4_ss_verdict v = ds4_ss_compat(&h, &nd, why, n);
    ds4_ss_meta_free(&h);
    ds4_ss_meta_free(&nd);
    return v;
}

static char *with_tail(const char *ident, const char *tail) {
    char *s = malloc(strlen(ident) + strlen(tail) + 1);
    strcpy(s, ident);
    strcat(s, tail);
    return s;
}

static char *replace(const char *s, const char *from, const char *to) {
    const char *p = strstr(s, from);
    if (!p) return strdup(s);
    char *r = malloc(strlen(s) - strlen(from) + strlen(to) + 1);
    memcpy(r, s, (size_t)(p - s));
    strcpy(r + (p - s), to);
    strcat(r, p + strlen(from));
    return r;
}

static void test_hash(void) {
    uint8_t buf[1000];
    for (int i = 0; i < 1000; i++) buf[i] = (uint8_t)(i * 37 + 11);
    const uint64_t whole = ds4_ss_hash_bytes(buf, sizeof(buf));
    for (size_t step = 1; step < 20; step++) {
        ds4_ss_hash s;
        ds4_ss_hash_init(&s);
        for (size_t o = 0; o < sizeof(buf); o += step)
            ds4_ss_hash_update(&s, buf + o, o + step > sizeof(buf) ? sizeof(buf) - o : step);
        CHECK(ds4_ss_hash_final(&s) == whole, "chunked hash differs at step %zu", step);
    }
    uint8_t b2[1000];
    memcpy(b2, buf, sizeof(buf));
    b2[999] ^= 1;
    CHECK(ds4_ss_hash_bytes(b2, sizeof(b2)) != whole, "last-byte flip not detected");
    CHECK(ds4_ss_hash_bytes(buf, 999) != whole, "truncation not detected");
    CHECK(ds4_ss_hash_bytes(buf, 0) != ds4_ss_hash_bytes(buf, 1), "empty vs one byte");
    /* Fixed vectors, shared with the kv store's payload digest (ds4_test --server). */
    CHECK(ds4_ss_hash_bytes("abc", 3) == UINT64_C(0xd5ae3d1b0d54b173), "abc vector");
    CHECK(whole == UINT64_C(0x7c47d8383fd6d07d), "1000-byte vector");
}

static void test_files(const char *dir) {
    char path[1024], err[512];
    snprintf(path, sizeof(path), "%s/snap-%ld.bin", dir, (long)getpid());
    const char data[] = "frontier state bytes";
    CHECK(ds4_ss_write_file_atomic(path, data, sizeof(data), false, err, sizeof(err)) == 0, "write: %s", err);
    CHECK(ds4_ss_write_file_atomic(path, data, sizeof(data), false, err, sizeof(err)) != 0, "overwrite allowed");
    CHECK(strstr(err, "immutable") != NULL, "overwrite reason: %s", err);
    uint64_t bytes = 0, h = 0;
    CHECK(ds4_ss_hash_file(path, &bytes, &h, err, sizeof(err)) == 0, "hash file: %s", err);
    CHECK(bytes == sizeof(data) && h == ds4_ss_hash_bytes(data, sizeof(data)), "file hash/bytes");
    char tmp[1100];
    snprintf(tmp, sizeof(tmp), "%s.tmp.%ld", path, (long)getpid());
    struct stat st;
    CHECK(stat(tmp, &st) != 0, "temporary file left behind");
    unlink(path);
}

static void test_compat(void) {
    char why[512];
    char *ok = with_tail(IDENT, "fixture /x\ndepth 8192\nend\n");
    CHECK(compat_of(ok, IDENT, why, sizeof(why)) == DS4_SS_OK, "identical: %s", why);

    /* moved / re-copied model: path and mtime change, weights do not */
    char *moved = replace(ok, "/a/model.gguf 501609136128 1792242799", "/b/m.gguf 501609136128 1800000000");
    CHECK(compat_of(moved, IDENT, why, sizeof(why)) == DS4_SS_OK, "moved model rejected: %s", why);

    char *other = replace(ok, "4d16135d", "5d16135d");
    CHECK(compat_of(other, IDENT, why, sizeof(why)) == DS4_SS_INCOMPATIBLE && strstr(why, "model weights"),
          "different weights: %s", why);

    char *num = replace(ok, "DS4_METAL_KV_RAW_F32=(unset)", "DS4_METAL_KV_RAW_F32=1");
    CHECK(compat_of(num, IDENT, why, sizeof(why)) == DS4_SS_INCOMPATIBLE && strstr(why, "numerics switch"),
          "numerics: %s", why);

    char *can = replace(ok, "0123456789abcdef", "fedcba9876543210");
    CHECK(compat_of(can, IDENT, why, sizeof(why)) == DS4_SS_INCOMPATIBLE && strstr(why, "canary"), "canary: %s", why);

    char *nocan = replace(ok, "canary 4096 0123456789abcdef\n", "");
    CHECK(compat_of(nocan, IDENT, why, sizeof(why)) == DS4_SS_OK, "snapshot without a canary: %s", why);
    /* A recorded canary this process could not recompute is refused. */
    char *need_nocan = replace(IDENT, "canary 4096 0123456789abcdef\n", "");
    CHECK(compat_of(ok, need_nocan, why, sizeof(why)) == DS4_SS_UNVERIFIABLE && strstr(why, "recompute") &&
          strstr(why, "0123456789abcdef"), "unrecomputed canary: %s", why);
    free(need_nocan);

    char *ctx = replace(ok, "ctx 393216", "ctx 65536");
    CHECK(compat_of(ctx, IDENT, why, sizeof(why)) == DS4_SS_INCOMPATIBLE && strstr(why, "context"), "ctx: %s", why);

    char *repr = replace(ok, "engram=f16-gguf", "engram=fp8-f16val");
    CHECK(compat_of(repr, IDENT, why, sizeof(why)) == DS4_SS_INCOMPATIBLE && strstr(why, "representation") &&
          strstr(why, "fp8-f16val") && strstr(why, "f16-gguf"), "repr: %s", why);
    char *vals = replace(ok, "prefill_values none", "prefill_values src:0123456789abcdef");
    CHECK(compat_of(vals, IDENT, why, sizeof(why)) == DS4_SS_INCOMPATIBLE && strstr(why, "value source"),
          "values: %s", why);

    char *unarmed = replace(ok, "armed 1", "armed 0");
    CHECK(compat_of(unarmed, IDENT, why, sizeof(why)) == DS4_SS_INCOMPATIBLE, "armed: %s", why);

    char *noend = replace(ok, "end\n", "");
    CHECK(compat_of(noend, IDENT, why, sizeof(why)) == DS4_SS_INCOMPLETE, "no end: %s", why);

    char *legacy = replace(ok, "ds4-bench-prefill-cache-4", "ds4-bench-prefill-cache-3");
    CHECK(compat_of(legacy, IDENT, why, sizeof(why)) == DS4_SS_LEGACY, "legacy");

    char *unv = replace(ok, "4d16135d76ef1a71ae3109ab5fc02d3014fa806c2caed7a4cbe84e2f695f4344", "unverifiable");
    CHECK(compat_of(unv, IDENT, why, sizeof(why)) == DS4_SS_UNVERIFIABLE, "unverifiable: %s", why);

    /* a switch recorded as set that this build no longer reports */
    char *extra = with_tail(IDENT, "fixture /x\nend\n");
    char *extra2 = replace(extra, "numerics DS4_METAL_KV_RAW_F32=(unset)\n",
                           "numerics DS4_METAL_KV_RAW_F32=(unset)\nnumerics DS4_METAL_OLD=1\n");
    CHECK(compat_of(extra2, IDENT, why, sizeof(why)) == DS4_SS_INCOMPATIBLE, "dropped switch: %s", why);

    ds4_ss_meta m;
    ds4_ss_meta_parse(ok, &m);
    uint64_t d = 0;
    CHECK(ds4_ss_meta_get_u64(&m, "depth", 10, &d) && d == 8192, "depth parse");
    CHECK(m.complete, "complete flag");
    CHECK(ds4_ss_meta_get(&m, "numerics DS4_METAL_HC_STABLE") && !strcmp(ds4_ss_meta_get(&m, "numerics DS4_METAL_HC_STABLE"), "(unset)"),
          "numerics key");
    ds4_ss_meta_free(&m);
    free(ok); free(moved); free(other); free(num); free(can); free(nocan); free(ctx); free(repr); free(vals);
    free(unarmed); free(noend); free(legacy); free(unv); free(extra); free(extra2);
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : ".";
    test_hash();
    test_files(dir);
    test_compat();
    if (failures) { fprintf(stderr, "test_snapstore: %d failure(s)\n", failures); return 1; }
    printf("test_snapstore: ok\n");
    return 0;
}
