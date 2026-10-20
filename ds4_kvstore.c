#include "ds4_kvstore.h"

/* Shared disk KV checkpoint file support.
 *
 * The low-level file layout and payload helpers are intentionally shared.  The
 * ds4-server still owns the automatic byte-prefix cache policy built on top of
 * this file; ds4-agent uses only the same durable format for explicit sessions,
 * with its own policy in ds4_agent.c.  Protocol-specific extras, such as the
 * server's tool-id -> exact DSML trailer, are attached through trailer hooks and
 * still live with the protocol code that owns those mappings. */

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define KV_CACHE_MAGIC0 'K'
#define KV_CACHE_MAGIC1 'V'
#define KV_CACHE_MAGIC2 'C'
/* v2: the header grew to 80 bytes and binds the entry to verified
 * content: the producing weights/representation (ds4_kvstore_engine_identity),
 * the prefill numerics canary and the payload hash, which the loader checks
 * before restoring anything. v1 files carry none of these: they are refused,
 * and stay visible to eviction so their space is reclaimed. */
#define KV_CACHE_VERSION 2u
/* Header byte 20 carries the graph-payload ABI.  It is separate from the outer
 * file version because the KVC envelope can remain stable while the serialized
 * ds4_session internals become unsafe to restore across runtime changes. */
#define KV_CACHE_PAYLOAD_ABI 2u
#define KV_CACHE_DEFAULT_MIN_TOKENS 512
#define KV_CACHE_DEFAULT_COLD_MAX_TOKENS 30000
/* Tokenizers may merge text across the prompt boundary. Trimming a small tail
 * still improves the cheap token-prefix path, while text-prefix lookup handles
 * cases where canonical prompt tokenization spells the same bytes differently.
 * The 2048 alignment also matches the backend prefill chunk schedule, which
 * keeps compressor row finalization identical to a cold full prompt. */
#define KV_CACHE_DEFAULT_BOUNDARY_TRIM_TOKENS 32
#define KV_CACHE_DEFAULT_BOUNDARY_ALIGN_TOKENS 2048
#define KV_CACHE_DEFAULT_CONTINUED_INTERVAL_TOKENS 10000
/* Disk-hit counts are evidence that a checkpoint was useful, but only while
 * the workload still resembles the one that produced those hits. */
#define KV_CACHE_MIN_EFFECTIVE_HITS 0.01
/* A continued checkpoint that is a strict prefix of the incoming store is a
 * routine waypoint on the same path. Keep recent hits meaningful, but make
 * never-hit or stale waypoints cheap victims while pre-evicting for the new
 * store. */
#define KV_CACHE_CONTINUED_PREFIX_MIN_FACTOR 0.05
#define KV_CACHE_CONTINUED_PREFIX_HIT_FACTOR 0.45
/* Cold/evict/shutdown checkpoints are intentional anchors, not just automatic
 * waypoints in a single growing conversation. Give them a soft prior so they
 * survive comparable continued entries, while still allowing pressure and poor
 * density to evict them. */
#define KV_CACHE_ANCHOR_REASON_SCORE_FACTOR 2.0

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} kv_buf;

static void kv_die(const char *msg) {
    fprintf(stderr, "ds4-kvstore: %s\n", msg);
    exit(1);
}

static void *kv_xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) kv_die("out of memory");
    return p;
}

static void *kv_xrealloc(void *p, size_t n) {
    p = realloc(p, n ? n : 1);
    if (!p) kv_die("out of memory");
    return p;
}

static char *kv_xstrdup(const char *s) {
    size_t n = strlen(s);
    char *p = kv_xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

static void kv_buf_reserve(kv_buf *b, size_t add) {
    if (add > SIZE_MAX - b->len - 1) kv_die("buffer overflow");
    size_t need = b->len + add + 1;
    if (need <= b->cap) return;
    size_t cap = b->cap ? b->cap * 2 : 256;
    while (cap < need) cap *= 2;
    b->ptr = kv_xrealloc(b->ptr, cap);
    b->cap = cap;
}

static void kv_buf_append(kv_buf *b, const void *p, size_t n) {
    kv_buf_reserve(b, n);
    memcpy(b->ptr + b->len, p, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}

static void kv_buf_putc(kv_buf *b, char c) {
    kv_buf_append(b, &c, 1);
}

static void kv_buf_puts(kv_buf *b, const char *s) {
    kv_buf_append(b, s, strlen(s));
}

static void kv_buf_printf(kv_buf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0) kv_die("vsnprintf failed");
    kv_buf_reserve(b, (size_t)n);
    vsnprintf(b->ptr + b->len, b->cap - b->len, fmt, ap2);
    va_end(ap2);
    b->len += (size_t)n;
}

static char *kv_buf_take(kv_buf *b) {
    if (!b->ptr) return kv_xstrdup("");
    char *p = b->ptr;
    memset(b, 0, sizeof(*b));
    return p;
}

static double kv_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static const char *kv_log_name(const ds4_kvstore *kc) {
    return kc && kc->log_name ? kc->log_name : "ds4";
}

static void kv_logf(ds4_kvstore *kc, ds4_kvstore_log_type type,
                    const char *fmt, ...) {
    if (!kc || !kc->log) return;
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0) {
        va_end(ap2);
        return;
    }
    char *msg = kv_xmalloc((size_t)n + 1);
    vsnprintf(msg, (size_t)n + 1, fmt, ap2);
    va_end(ap2);
    kc->log(kc->log_ud, type, msg);
    free(msg);
}

ds4_kvstore_options ds4_kvstore_default_options(void) {
    return (ds4_kvstore_options){
        .min_tokens = KV_CACHE_DEFAULT_MIN_TOKENS,
        .cold_max_tokens = KV_CACHE_DEFAULT_COLD_MAX_TOKENS,
        .continued_interval_tokens = KV_CACHE_DEFAULT_CONTINUED_INTERVAL_TOKENS,
        .boundary_trim_tokens = KV_CACHE_DEFAULT_BOUNDARY_TRIM_TOKENS,
        .boundary_align_tokens = KV_CACHE_DEFAULT_BOUNDARY_ALIGN_TOKENS,
    };
}

uint8_t ds4_kvstore_reason_code(const char *reason) {
    if (!reason) return DS4_KVSTORE_REASON_UNKNOWN;
    if (!strcmp(reason, "cold")) return DS4_KVSTORE_REASON_COLD;
    if (!strcmp(reason, "continued")) return DS4_KVSTORE_REASON_CONTINUED;
    if (!strcmp(reason, "evict")) return DS4_KVSTORE_REASON_EVICT;
    if (!strcmp(reason, "shutdown")) return DS4_KVSTORE_REASON_SHUTDOWN;
    if (!strcmp(reason, "agent-system")) return DS4_KVSTORE_REASON_AGENT_SYSTEM;
    if (!strcmp(reason, "agent-session")) return DS4_KVSTORE_REASON_AGENT_SESSION;
    return DS4_KVSTORE_REASON_UNKNOWN;
}

const char *ds4_kvstore_key_kind(uint8_t ext_flags) {
    if (ext_flags & DS4_KVSTORE_EXT_RESPONSES_VISIBLE) return "responses-visible";
    if (ext_flags & DS4_KVSTORE_EXT_THINKING_VISIBLE) return "thinking-visible";
    return "token-text";
}

void ds4_kvstore_le_put32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

static void kv_le_put64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}

uint32_t ds4_kvstore_le_get32(const uint8_t *p) {
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint64_t kv_le_get64(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}

typedef struct {
    uint32_t h[5];
    uint64_t bytes;
    uint8_t block[64];
    size_t used;
} sha1_ctx;

static uint32_t rol32(uint32_t v, int n) {
    return (v << n) | (v >> (32 - n));
}

static void sha1_transform(sha1_ctx *c, const uint8_t block[64]) {
    uint32_t w[80];
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i * 4] << 24) |
               ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) |
               (uint32_t)block[i * 4 + 3];
    }
    for (int i = 16; i < 80; i++)
        w[i] = rol32(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);

    uint32_t a = c->h[0], b = c->h[1], d = c->h[3], e = c->h[4];
    uint32_t cc = c->h[2];
    for (int i = 0; i < 80; i++) {
        uint32_t f, k;
        if (i < 20) {
            f = (b & cc) | ((~b) & d);
            k = 0x5a827999u;
        } else if (i < 40) {
            f = b ^ cc ^ d;
            k = 0x6ed9eba1u;
        } else if (i < 60) {
            f = (b & cc) | (b & d) | (cc & d);
            k = 0x8f1bbcdcu;
        } else {
            f = b ^ cc ^ d;
            k = 0xca62c1d6u;
        }
        uint32_t tmp = rol32(a, 5) + f + e + k + w[i];
        e = d;
        d = cc;
        cc = rol32(b, 30);
        b = a;
        a = tmp;
    }
    c->h[0] += a;
    c->h[1] += b;
    c->h[2] += cc;
    c->h[3] += d;
    c->h[4] += e;
}

static void sha1_init(sha1_ctx *c) {
    c->h[0] = 0x67452301u;
    c->h[1] = 0xefcdab89u;
    c->h[2] = 0x98badcfeu;
    c->h[3] = 0x10325476u;
    c->h[4] = 0xc3d2e1f0u;
    c->bytes = 0;
    c->used = 0;
}

static void sha1_update(sha1_ctx *c, const void *ptr, size_t len) {
    const uint8_t *p = ptr;
    c->bytes += len;
    while (len != 0) {
        size_t n = 64 - c->used;
        if (n > len) n = len;
        memcpy(c->block + c->used, p, n);
        c->used += n;
        p += n;
        len -= n;
        if (c->used == 64) {
            sha1_transform(c, c->block);
            c->used = 0;
        }
    }
}

static void sha1_final(sha1_ctx *c, uint8_t out[20]) {
    uint64_t bits = c->bytes * 8;
    uint8_t one = 0x80;
    uint8_t zero = 0;
    sha1_update(c, &one, 1);
    while (c->used != 56) sha1_update(c, &zero, 1);
    uint8_t len[8];
    for (int i = 0; i < 8; i++) len[7 - i] = (uint8_t)(bits >> (8 * i));
    sha1_update(c, len, sizeof(len));
    for (int i = 0; i < 5; i++) {
        out[i * 4] = (uint8_t)(c->h[i] >> 24);
        out[i * 4 + 1] = (uint8_t)(c->h[i] >> 16);
        out[i * 4 + 2] = (uint8_t)(c->h[i] >> 8);
        out[i * 4 + 3] = (uint8_t)c->h[i];
    }
}

static void hex20(const uint8_t in[20], char out[41]) {
    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < 20; i++) {
        out[i * 2] = hex[in[i] >> 4];
        out[i * 2 + 1] = hex[in[i] & 15];
    }
    out[40] = '\0';
}

void ds4_kvstore_sha1_bytes_hex(const void *ptr, size_t len, char out[41]) {
    sha1_ctx c;
    sha1_init(&c);
    sha1_update(&c, ptr, len);
    uint8_t digest[20];
    sha1_final(&c, digest);
    hex20(digest, out);
}

bool ds4_kvstore_sha_hex_name(const char *name, char sha[41]) {
    if (strlen(name) != 43 || strcmp(name + 40, ".kv")) return false;
    for (int i = 0; i < 40; i++) {
        if (!isxdigit((unsigned char)name[i])) return false;
        sha[i] = (char)tolower((unsigned char)name[i]);
    }
    sha[40] = '\0';
    return true;
}

char *ds4_kvstore_path_join(const char *dir, const char *name) {
    kv_buf b = {0};
    kv_buf_puts(&b, dir);
    if (b.len == 0 || b.ptr[b.len - 1] != '/') kv_buf_putc(&b, '/');
    kv_buf_puts(&b, name);
    return kv_buf_take(&b);
}

char *ds4_kvstore_path_for_sha(ds4_kvstore *kc, const char sha[41]) {
    char name[44];
    memcpy(name, sha, 40);
    memcpy(name + 40, ".kv", 4);
    return ds4_kvstore_path_join(kc->dir, name);
}

static bool kv_mkdir_p(const char *path) {
    if (!path || !path[0]) return false;
    char *tmp = kv_xstrdup(path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p != '/') continue;
        *p = '\0';
        if (mkdir(tmp, 0700) != 0 && errno != EEXIST) {
            free(tmp);
            return false;
        }
        *p = '/';
    }
    bool ok = mkdir(tmp, 0700) == 0 || errno == EEXIST;
    free(tmp);
    return ok;
}

void ds4_kvstore_entry_free(ds4_kvstore_entry *e) {
    free(e->path);
    memset(e, 0, sizeof(*e));
}

void ds4_kvstore_clear(ds4_kvstore *kc) {
    for (int i = 0; i < kc->len; i++) ds4_kvstore_entry_free(&kc->entry[i]);
    free(kc->entry);
    kc->entry = NULL;
    kc->len = 0;
    kc->cap = 0;
}

static void kv_cache_push(ds4_kvstore *kc, ds4_kvstore_entry e) {
    if (kc->len == kc->cap) {
        kc->cap = kc->cap ? kc->cap * 2 : 16;
        kc->entry = kv_xrealloc(kc->entry, (size_t)kc->cap * sizeof(kc->entry[0]));
    }
    kc->entry[kc->len++] = e;
}

/* Process-wide identity of the engine whose state this store holds, and its
 * numerics canary (0 = not bound yet / not computed: identity is then bound
 * by the first store or load, and the canary is not checked). */
static uint64_t g_kv_identity;
static uint64_t g_kv_canary;
/* Canary gate, read by request threads while the canary thread completes it. */
enum { KV_CANARY_NONE, KV_CANARY_PENDING, KV_CANARY_READY, KV_CANARY_FAILED };
static int g_kv_canary_state;

static uint64_t kv_canary(void) { return __atomic_load_n(&g_kv_canary, __ATOMIC_ACQUIRE); }

static uint64_t kv_mix64(uint64_t h, uint64_t v) {
    for (int i = 0; i < 8; i++) { h ^= (uint8_t)(v >> (8 * i)); h *= 1099511628211ull; }
    return h;
}

uint64_t ds4_kvstore_engine_identity(ds4_engine *engine) {
    uint64_t h = 1469598103934665603ull;
    h = kv_mix64(h, ds4_engine_model_fingerprint(engine, 0));   /* GGUF header: metadata, tokenizer, tensor table */
    h = kv_mix64(h, ds4_engine_model_fingerprint(engine, 1));   /* drafter (capture-state producer), 0 = none */
    h = kv_mix64(h, (uint64_t)ds4_engine_routed_quant_bits(engine));
    h = kv_mix64(h, (uint64_t)ds4_engine_model_id(engine));
    for (const char *r = ds4_engine_prefill_repr(engine); r && *r; r++) { h ^= (uint8_t)*r; h *= 1099511628211ull; }
    h ^= 0x1f; h *= 1099511628211ull;   /* field separator */
    for (const char *r = ds4_engine_prefill_values(engine); r && *r; r++) { h ^= (uint8_t)*r; h *= 1099511628211ull; }
    return h ? h : 1;
}

void ds4_kvstore_set_identity(uint64_t identity) { g_kv_identity = identity; }
void ds4_kvstore_canary_pending(void) {
    __atomic_store_n(&g_kv_canary_state, KV_CANARY_PENDING, __ATOMIC_RELEASE);
}

void ds4_kvstore_set_canary(uint64_t canary) {
    __atomic_store_n(&g_kv_canary, canary, __ATOMIC_RELEASE);
    __atomic_store_n(&g_kv_canary_state, KV_CANARY_READY, __ATOMIC_RELEASE);
}

void ds4_kvstore_canary_failed(void) {
    __atomic_store_n(&g_kv_canary_state, KV_CANARY_FAILED, __ATOMIC_RELEASE);
}

bool ds4_kvstore_canary_ready(void) {
    const int st = __atomic_load_n(&g_kv_canary_state, __ATOMIC_ACQUIRE);
    return st == KV_CANARY_NONE || st == KV_CANARY_READY;
}

/* One line per gate state per process: why the disk cache is not used yet. */
static bool kv_canary_gate(ds4_kvstore *kc, const char *what) {
    if (ds4_kvstore_canary_ready()) return true;
    static int logged = -1;
    const int st = __atomic_load_n(&g_kv_canary_state, __ATOMIC_ACQUIRE);
    if (__atomic_exchange_n(&logged, st, __ATOMIC_ACQ_REL) != st)
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE, "%s: kv cache %s refused: %s", kv_log_name(kc), what,
                st == KV_CANARY_PENDING ? "the numerics canary has not completed yet (the cache turns on when it does)" :
                                          "the numerics canary failed (the cache stays off)");
    return false;
}

/* ---- payload digest: a word-wise FNV-style mix ---- */

#define KV_HASH_PRIME UINT64_C(0x100000001b3)
#define KV_HASH_BASIS UINT64_C(0xcbf29ce484222325)

static inline uint64_t kv_hash_mix(uint64_t h, uint64_t w) {
    h ^= w;
    h *= KV_HASH_PRIME;
    h ^= h >> 29;
    return h;
}

static inline uint64_t kv_load64(const uint8_t *p) {
    uint64_t w;
    memcpy(&w, p, 8);
    return w;
}

void ds4_kvstore_hash_init(ds4_kvstore_hash *s) {
    memset(s, 0, sizeof(*s));
    s->h = KV_HASH_BASIS;
}

void ds4_kvstore_hash_update(ds4_kvstore_hash *s, const void *p, size_t n) {
    const uint8_t *b = p;
    s->n += n;
    while (s->tl && n) {
        s->tail[s->tl++] = *b++;
        n--;
        if (s->tl == 8) { s->h = kv_hash_mix(s->h, kv_load64(s->tail)); s->tl = 0; }
    }
    while (n >= 8) { s->h = kv_hash_mix(s->h, kv_load64(b)); b += 8; n -= 8; }
    while (n) { s->tail[s->tl++] = *b++; n--; }
}

uint64_t ds4_kvstore_hash_final(const ds4_kvstore_hash *s) {
    uint64_t h = s->h;
    if (s->tl) {
        uint8_t t[8] = {0};
        memcpy(t, s->tail, s->tl);
        h = kv_hash_mix(h, kv_load64(t));
    }
    return kv_hash_mix(h, s->n);
}

uint64_t ds4_kvstore_payload_hash(const void *p, size_t n) {
    ds4_kvstore_hash s;
    ds4_kvstore_hash_init(&s);
    ds4_kvstore_hash_update(&s, p, n);
    return ds4_kvstore_hash_final(&s);
}

enum { KV_HASH_CHUNK = 8 << 20 };

int ds4_kvstore_verify_payload(FILE *fp, const ds4_kvstore_entry *e,
                               char *why, size_t why_len) {
    const off_t start = ftello(fp);
    if (start < 0) {
        snprintf(why, why_len, "cannot locate the payload");
        return 1;
    }
    if (e->payload_hash == 0) {
        snprintf(why, why_len, "entry records no payload hash");
        return 1;
    }
    uint8_t *buf = malloc(KV_HASH_CHUNK);
    if (!buf) {
        snprintf(why, why_len, "out of memory verifying the payload");
        return 1;
    }
    ds4_kvstore_hash s;
    ds4_kvstore_hash_init(&s);
    uint64_t left = e->payload_bytes;
    while (left > 0) {
        const size_t want = left < KV_HASH_CHUNK ? (size_t)left : KV_HASH_CHUNK;
        if (fread(buf, 1, want, fp) != want) break;
        ds4_kvstore_hash_update(&s, buf, want);
        left -= want;
    }
    free(buf);
    if (left > 0) {
        snprintf(why, why_len, "payload is shorter than the %llu bytes the header records",
                 (unsigned long long)e->payload_bytes);
        return 1;
    }
    const uint64_t got = ds4_kvstore_hash_final(&s);
    if (got != e->payload_hash) {
        snprintf(why, why_len, "payload hash %016llx, header records %016llx (corrupt payload)",
                 (unsigned long long)got, (unsigned long long)e->payload_hash);
        return 1;
    }
    if (fseeko(fp, start, SEEK_SET) != 0) {
        snprintf(why, why_len, "cannot rewind to the payload");
        return 1;
    }
    return 0;
}

/* Copies the staged payload into fp, digesting it on the way. */
static int kv_copy_staged_hashed(const ds4_session_payload_file *staged, FILE *fp,
                                 uint64_t *hash, char *err, size_t err_len) {
    FILE *src = staged && staged->path ? fopen(staged->path, "rb") : NULL;
    if (!src) {
        snprintf(err, err_len, "failed to open staged session payload");
        return 1;
    }
    uint8_t *buf = malloc(KV_HASH_CHUNK);
    ds4_kvstore_hash s;
    ds4_kvstore_hash_init(&s);
    uint64_t left = staged->bytes;
    int rc = buf ? 0 : 1;
    while (rc == 0 && left > 0) {
        const size_t want = left < KV_HASH_CHUNK ? (size_t)left : KV_HASH_CHUNK;
        if (fread(buf, 1, want, src) != want || fwrite(buf, 1, want, fp) != want) {
            rc = 1;
        } else {
            ds4_kvstore_hash_update(&s, buf, want);
            left -= want;
        }
    }
    if (rc) snprintf(err, err_len, "%s", buf ? "failed to copy staged session payload" :
                                               "out of memory copying staged session payload");
    free(buf);
    fclose(src);
    if (rc == 0) *hash = ds4_kvstore_hash_final(&s);
    return rc;
}

/* Header bytes 64..71, written once the payload digest is known. */
static bool kv_put_payload_hash(FILE *fp, uint64_t hash) {
    uint8_t b[8];
    kv_le_put64(b, hash);
    return fflush(fp) == 0 && fseeko(fp, 64, SEEK_SET) == 0 &&
           fwrite(b, 1, sizeof(b), fp) == sizeof(b);
}

int ds4_kvstore_engine_canary(ds4_engine *engine, int tokens, uint64_t *out,
                              char *err, size_t err_len) {
    if (out) *out = 0;
    if (!engine || !out || tokens <= 0) {
        snprintf(err, err_len, "invalid canary request");
        return 1;
    }
    /* Fixed prompt: decimal numbers from a 32-bit LCG, each at least one
     * token, through the bound tokenizer (part of the identity). */
    kv_buf text = {0};
    uint32_t x = 0x9e3779b9u;
    for (int i = 0; i < tokens; i++) {
        x = x * 1664525u + 1013904223u;
        kv_buf_printf(&text, "%05u ", (unsigned)(x % 100000u));
    }
    char *t = kv_buf_take(&text);
    ds4_tokens prompt = {0};
    ds4_tokenize_text(engine, t, &prompt);
    free(t);
    const int vocab = ds4_engine_vocab_size(engine);
    float *logits = vocab > 0 ? malloc((size_t)vocab * sizeof(float)) : NULL;
    ds4_session *cs = NULL;
    char serr[160] = {0};
    int rc = 1;
    if (prompt.len >= tokens) prompt.len = tokens;
    if (prompt.len < tokens) {
        snprintf(err, err_len, "canary prompt tokenized to %d of %d tokens", prompt.len, tokens);
    } else if (!logits) {
        snprintf(err, err_len, "no vocabulary for the canary logits");
    } else if (ds4_session_create(&cs, engine, tokens + 64) != 0) {
        snprintf(err, err_len, "canary session (ctx %d) could not be created", tokens + 64);
    } else if (ds4_session_sync(cs, &prompt, serr, sizeof(serr)) != 0) {
        snprintf(err, err_len, "canary prefill failed: %s", serr[0] ? serr : "unknown error");
    } else {
        const int n = ds4_session_copy_logits(cs, logits, vocab);
        if (n <= 0) {
            snprintf(err, err_len, "canary prefill produced no logits");
        } else {
            ds4_kvstore_hash s;
            ds4_kvstore_hash_init(&s);
            ds4_kvstore_hash_update(&s, prompt.v, (size_t)tokens * sizeof(prompt.v[0]));
            ds4_kvstore_hash_update(&s, logits, (size_t)n * sizeof(float));
            const uint64_t h = ds4_kvstore_hash_final(&s);
            *out = h ? h : 1;
            rc = 0;
        }
    }
    if (cs) ds4_session_free(cs);
    free(logits);
    ds4_tokens_free(&prompt);
    return rc;
}

void ds4_kvstore_fill_header(uint8_t h[DS4_KVSTORE_FIXED_HEADER],
                             uint8_t model_id, uint8_t quant_bits,
                             uint8_t reason, uint8_t ext_flags,
                             uint32_t tokens, uint32_t hits, uint32_t ctx_size,
                             uint64_t created_at, uint64_t last_used,
                             uint64_t payload_bytes) {
    memset(h, 0, DS4_KVSTORE_FIXED_HEADER);
    h[0] = KV_CACHE_MAGIC0;
    h[1] = KV_CACHE_MAGIC1;
    h[2] = KV_CACHE_MAGIC2;
    h[3] = KV_CACHE_VERSION;
    h[4] = quant_bits;
    h[5] = reason;
    h[6] = ext_flags;
    h[7] = model_id;
    ds4_kvstore_le_put32(h + 8, tokens);
    ds4_kvstore_le_put32(h + 12, hits);
    ds4_kvstore_le_put32(h + 16, ctx_size);
    h[20] = KV_CACHE_PAYLOAD_ABI;
    kv_le_put64(h + 24, created_at);
    kv_le_put64(h + 32, last_used);
    kv_le_put64(h + 40, payload_bytes);
    kv_le_put64(h + 48, g_kv_identity);
    kv_le_put64(h + 56, kv_canary());
    /* 64..71: payload hash, written after the payload; 72..79 reserved */
}

/* routed-expert quantizations whose checkpoints the store keys on */
bool ds4_kvstore_quant_bits_supported(int quant_bits) {
    return quant_bits == 2 || quant_bits == 4 || quant_bits == 5 ||
           quant_bits == 6 || quant_bits == 8;
}

/* Header and text length at the file start. false: not an entry this code
 * can read. true with e->refusal set: readable, but this engine must not load
 * it (the fields are still valid for eviction). */
static bool kv_parse_header(FILE *fp, ds4_kvstore_entry *e,
                            uint32_t *text_bytes) {
    uint8_t h[DS4_KVSTORE_FIXED_HEADER];
    if (fread(h, 1, 4, fp) != 4) return false;
    if (h[0] != KV_CACHE_MAGIC0 || h[1] != KV_CACHE_MAGIC1 || h[2] != KV_CACHE_MAGIC2) return false;
    if (h[3] != 1u && h[3] != KV_CACHE_VERSION) return false;
    const size_t hb = h[3] == 1u ? DS4_KVSTORE_V1_HEADER : DS4_KVSTORE_FIXED_HEADER;
    if (fread(h + 4, 1, hb - 4, fp) != hb - 4) return false;
    e->version = h[3];
    e->quant_bits = h[4];
    e->reason = h[5] <= DS4_KVSTORE_REASON_AGENT_SESSION ? h[5] :
                DS4_KVSTORE_REASON_UNKNOWN;
    e->ext_flags = h[6];
    e->model_id = h[7];
    e->tokens = ds4_kvstore_le_get32(h + 8);
    e->hits = ds4_kvstore_le_get32(h + 12);
    e->ctx_size = ds4_kvstore_le_get32(h + 16);
    e->created_at = kv_le_get64(h + 24);
    e->last_used = kv_le_get64(h + 32);
    e->payload_bytes = kv_le_get64(h + 40);
    e->identity = e->canary = e->payload_hash = 0;
    if (hb == DS4_KVSTORE_FIXED_HEADER) {
        e->identity = kv_le_get64(h + 48);
        e->canary = kv_le_get64(h + 56);
        e->payload_hash = kv_le_get64(h + 64);
    }
    uint8_t tb[4];
    if (fread(tb, 1, sizeof(tb), fp) != sizeof(tb)) return false;
    *text_bytes = ds4_kvstore_le_get32(tb);
    e->text_bytes = *text_bytes;
    e->refusal = NULL;
    if (e->version == 1u)
        e->refusal = "v1 entry (no engine identity, canary or payload hash)";
    else if (h[20] != KV_CACHE_PAYLOAD_ABI)
        e->refusal = "graph payload ABI differs";
    else if (g_kv_identity && e->identity != g_kv_identity)
        e->refusal = "engine identity differs (other weights or prefill representation)";
    else if (kv_canary() && e->canary != kv_canary())
        e->refusal = e->canary ? "prefill numerics canary differs" : "entry records no numerics canary";
    return e->tokens != 0 && ds4_kvstore_quant_bits_supported(e->quant_bits);
}

/* One stderr line per refusal kind per process (callers outside the indexed
 * lookup, such as ds4-agent, see refused entries only through this). */
static void kv_note_refusal(const ds4_kvstore_entry *e) {
    static const char *seen[8];
    for (int i = 0; i < 8; i++) {
        if (seen[i] == e->refusal) return;
        if (!seen[i]) { seen[i] = e->refusal; break; }
    }
    fprintf(stderr, "ds4: kv cache entries refused: %s (entry identity %016llx canary %016llx, "
                    "this engine %016llx %016llx); they re-prefill\n",
            e->refusal, (unsigned long long)e->identity, (unsigned long long)e->canary,
            (unsigned long long)g_kv_identity, (unsigned long long)kv_canary());
}

bool ds4_kvstore_read_header(FILE *fp, ds4_kvstore_entry *e,
                             uint32_t *text_bytes) {
    if (!kv_parse_header(fp, e, text_bytes)) return false;
    if (e->refusal) {
        kv_note_refusal(e);
        return false;
    }
    return true;
}

static bool kv_read_entry(const char *path, const char sha[41],
                          ds4_kvstore_entry *out, bool keep_refused) {
    struct stat st;
    if (stat(path, &st) != 0 ||
        st.st_size < (off_t)(DS4_KVSTORE_V1_HEADER + 4))
        return false;
    FILE *fp = fopen(path, "rb");
    if (!fp) return false;
    ds4_kvstore_entry e = {0};
    uint32_t text_bytes = 0;
    bool ok = kv_parse_header(fp, &e, &text_bytes);
    fclose(fp);
    if (!ok) return false;
    if (e.refusal && !keep_refused) {
        kv_note_refusal(&e);
        return false;
    }
    const uint64_t fixed = (e.version == 1u ? DS4_KVSTORE_V1_HEADER : DS4_KVSTORE_FIXED_HEADER) + 4ull;
    if (UINT64_MAX - fixed < (uint64_t)text_bytes ||
        UINT64_MAX - fixed - (uint64_t)text_bytes < e.payload_bytes)
        return false;
    const uint64_t expected = fixed + (uint64_t)text_bytes + e.payload_bytes;
    if ((uint64_t)st.st_size < expected) return false;
    memcpy(e.sha, sha, 41);
    e.path = kv_xstrdup(path);
    e.file_size = (uint64_t)st.st_size;
    *out = e;
    return true;
}

bool ds4_kvstore_read_entry_file(const char *path, const char sha[41],
                                 ds4_kvstore_entry *out) {
    return kv_read_entry(path, sha, out, false);
}

/* Indexes every readable entry, refused ones included: lookups skip those,
 * eviction still counts and reclaims their space. */
static void kv_cache_refresh(ds4_kvstore *kc) {
    if (!kc->enabled) return;
    ds4_kvstore_clear(kc);
    DIR *d = opendir(kc->dir);
    if (!d) return;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        char sha[41];
        if (!ds4_kvstore_sha_hex_name(de->d_name, sha)) continue;
        char *path = ds4_kvstore_path_join(kc->dir, de->d_name);
        ds4_kvstore_entry e = {0};
        if (kv_read_entry(path, sha, &e, true)) kv_cache_push(kc, e);
        free(path);
    }
    closedir(d);
}

/* Only the hit count (12..15) and last use (32..39) change; the identity,
 * canary and payload hash stay as written. */
bool ds4_kvstore_touch_file(const char *path, uint32_t hits) {
    FILE *fp = fopen(path, "r+b");
    if (!fp) return false;
    ds4_kvstore_entry e = {0};
    uint32_t text_bytes = 0;
    bool ok = ds4_kvstore_read_header(fp, &e, &text_bytes);
    if (ok) {
        uint8_t hb[4], tb[8];
        ds4_kvstore_le_put32(hb, hits);
        kv_le_put64(tb, (uint64_t)time(NULL));
        ok = fseek(fp, 12, SEEK_SET) == 0 && fwrite(hb, 1, sizeof(hb), fp) == sizeof(hb) &&
             fseek(fp, 32, SEEK_SET) == 0 && fwrite(tb, 1, sizeof(tb), fp) == sizeof(tb);
    }
    fclose(fp);
    return ok;
}

static bool kv_cache_incoming_supersedes_continued(
        const ds4_kvstore_entry *e,
        const ds4_kvstore_eviction_context *incoming) {
    if (!e || !incoming || !incoming->text) return false;
    if (e->reason != DS4_KVSTORE_REASON_CONTINUED) return false;
    if (e->text_bytes == 0 || e->text_bytes > SIZE_MAX) return false;
    if ((size_t)e->text_bytes >= incoming->text_len) return false;
    if (e->model_id != incoming->model_id) return false;
    if (incoming->reject_different_quant &&
        e->quant_bits != incoming->quant_bits)
        return false;
    /* A smaller-context checkpoint can be loaded by a larger context, but not
     * the reverse.  The incoming checkpoint dominates this one only if it is at
     * least as widely reusable. */
    if (incoming->ctx_size > e->ctx_size) return false;

    char prefix_sha[41];
    ds4_kvstore_sha1_bytes_hex(incoming->text, (size_t)e->text_bytes,
                               prefix_sha);
    return !strcmp(prefix_sha, e->sha);
}

static bool kv_cache_reason_is_anchor(uint8_t reason) {
    return reason == DS4_KVSTORE_REASON_COLD ||
           reason == DS4_KVSTORE_REASON_EVICT ||
           reason == DS4_KVSTORE_REASON_SHUTDOWN;
}

double ds4_kvstore_entry_eviction_score(
        const ds4_kvstore_entry *e,
        const ds4_tokens *live,
        uint64_t now,
        const ds4_kvstore_eviction_context *incoming) {
    if (!e || e->file_size == 0) return 0.0;
    (void)live;
    double effective_hits = (double)e->hits;
    uint64_t used_at = e->last_used ? e->last_used : e->created_at;
    if (used_at == 0) {
        effective_hits = 0.0;
    } else if (now > used_at) {
        double elapsed = (double)(now - used_at);
        effective_hits *= exp2(-elapsed / (double)DS4_KVSTORE_HIT_HALF_LIFE_SECONDS);
        if (effective_hits < KV_CACHE_MIN_EFFECTIVE_HITS) effective_hits = 0.0;
    }
    double score = (effective_hits + 1.0) *
                   (double)e->tokens / (double)e->file_size;
    if (kv_cache_reason_is_anchor(e->reason))
        score *= KV_CACHE_ANCHOR_REASON_SCORE_FACTOR;
    if (kv_cache_incoming_supersedes_continued(e, incoming)) {
        double h = effective_hits > 0.0 ?
            effective_hits / (effective_hits + 1.0) : 0.0;
        score *= KV_CACHE_CONTINUED_PREFIX_MIN_FACTOR +
                 KV_CACHE_CONTINUED_PREFIX_HIT_FACTOR * h;
    }
    return score;
}

void ds4_kvstore_evict(ds4_kvstore *kc, const ds4_tokens *live,
                       uint64_t extra_bytes,
                       const ds4_kvstore_eviction_context *incoming) {
    if (!kc->enabled || kc->budget_bytes == 0) return;
    if (extra_bytes > kc->budget_bytes) return;
    kv_cache_refresh(kc);
    const uint64_t now = (uint64_t)time(NULL);
    uint64_t total = 0;
    for (int i = 0; i < kc->len; i++) total += kc->entry[i].file_size;
    const uint64_t target = kc->budget_bytes - extra_bytes;
    while (total > target && kc->len > 0) {
        int victim = 0;
        double victim_score =
            ds4_kvstore_entry_eviction_score(&kc->entry[0], live, now,
                                             incoming);
        for (int i = 1; i < kc->len; i++) {
            double score =
                ds4_kvstore_entry_eviction_score(&kc->entry[i], live, now,
                                                 incoming);
            if (score < victim_score ||
                (score == victim_score &&
                 kc->entry[i].last_used < kc->entry[victim].last_used))
            {
                victim = i;
                victim_score = score;
            }
        }
        ds4_kvstore_entry e = kc->entry[victim];
        if (unlink(e.path) == 0) {
            kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                    "%s: kv cache evicted reason=disk-cache-full tokens=%u hits=%u size=%.2f MiB file=%s",
                    kv_log_name(kc),
                    e.tokens,
                    e.hits,
                    (double)e.file_size / (1024.0 * 1024.0),
                    e.path ? e.path : "?");
            if (total >= e.file_size) total -= e.file_size;
            else total = 0;
        } else {
            total = 0;
        }
        ds4_kvstore_entry_free(&e);
        memmove(kc->entry + victim, kc->entry + victim + 1,
                (size_t)(kc->len - victim - 1) * sizeof(kc->entry[0]));
        kc->len--;
    }
}

bool ds4_kvstore_open(ds4_kvstore *kc, const char *dir, uint64_t budget_mb,
                      bool reject_different_quant, ds4_kvstore_options opt,
                      const char *log_name,
                      void (*log)(void *ud, ds4_kvstore_log_type type, const char *msg),
                      void *log_ud) {
    memset(kc, 0, sizeof(*kc));
    if (!dir) return false;
    kc->log_name = log_name;
    kc->log = log;
    kc->log_ud = log_ud;
    if (!kv_mkdir_p(dir)) {
        kv_logf(kc, DS4_KVSTORE_LOG_DEFAULT,
                "%s: failed to create KV cache directory %s: %s",
                kv_log_name(kc), dir, strerror(errno));
        return false;
    }
    kc->enabled = true;
    kc->dir = kv_xstrdup(dir);
    if (budget_mb == 0) budget_mb = DS4_KVSTORE_DEFAULT_MB;
    kc->budget_bytes = budget_mb * 1024ull * 1024ull;
    kc->reject_different_quant = reject_different_quant;
    kc->opt = opt;
    ds4_kvstore_evict(kc, NULL, 0, NULL);
    kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
            "%s: KV disk cache %s (budget=%llu MiB, cross-quant=%s, min=%d, cold_max=%d, continued=%d, trim=%d, align=%d, hit_half_life=%llus)",
            kv_log_name(kc),
            kc->dir,
            (unsigned long long)(kc->budget_bytes / (1024ull * 1024ull)),
            reject_different_quant ? "reject" : "accept",
            kc->opt.min_tokens,
            kc->opt.cold_max_tokens,
            kc->opt.continued_interval_tokens,
            kc->opt.boundary_trim_tokens,
            kc->opt.boundary_align_tokens,
            (unsigned long long)DS4_KVSTORE_HIT_HALF_LIFE_SECONDS);
    return true;
}

void ds4_kvstore_close(ds4_kvstore *kc) {
    ds4_kvstore_clear(kc);
    free(kc->dir);
    memset(kc, 0, sizeof(*kc));
}

char *ds4_kvstore_render_tokens_text(ds4_engine *engine,
                                     const ds4_tokens *tokens,
                                     size_t *out_len) {
    kv_buf b = {0};
    for (int i = 0; i < tokens->len; i++) {
        size_t len = 0;
        char *piece = ds4_token_text(engine, tokens->v[i], &len);
        kv_buf_append(&b, piece, len);
        free(piece);
    }
    if (out_len) *out_len = b.len;
    return kv_buf_take(&b);
}

bool ds4_kvstore_byte_prefix_match(const char *text, size_t text_len,
                                   const char *prefix, size_t prefix_len) {
    return prefix_len <= text_len &&
           (prefix_len == 0 || memcmp(text, prefix, prefix_len) == 0);
}

void ds4_kvstore_tokens_copy_prefix(ds4_tokens *dst, const ds4_tokens *src, int n) {
    dst->len = 0;
    if (!src) return;
    if (n > src->len) n = src->len;
    for (int i = 0; i < n; i++) ds4_tokens_push(dst, src->v[i]);
}

static void tokens_append(ds4_tokens *dst, const ds4_tokens *src) {
    if (!dst || !src) return;
    for (int i = 0; i < src->len; i++) ds4_tokens_push(dst, src->v[i]);
}

void ds4_kvstore_build_prompt_from_exact_prefix_and_text_suffix(
        ds4_engine *engine,
        const ds4_tokens *exact_prefix,
        const char *suffix_text,
        ds4_tokens *out) {
    ds4_tokens_copy(out, exact_prefix);

    ds4_tokens suffix = {0};
    /* The suffix may start with DS4 chat markers such as <｜User｜> or
     * </think>, so use the rendered-chat tokenizer, not plain text BPE. */
    ds4_tokenize_rendered_chat(engine, suffix_text ? suffix_text : "", &suffix);
    tokens_append(out, &suffix);
    ds4_tokens_free(&suffix);
}

int ds4_kvstore_store_len(const ds4_kvstore *kc, int tokens) {
    const int trim = kc->opt.boundary_trim_tokens;
    const int align = kc->opt.boundary_align_tokens;
    if (tokens > kc->opt.min_tokens + trim) {
        int stable = tokens - trim;
        if (align > 0) stable -= stable % align;
        if (stable >= kc->opt.min_tokens) return stable;
    }
    return tokens;
}

int ds4_kvstore_chat_anchor_pos(const ds4_kvstore *kc,
                                const ds4_tokens *prompt,
                                int user_token_id,
                                int assistant_token_id) {
    if (!prompt || user_token_id < 0 || assistant_token_id < 0) return -1;

    /* Cold checkpoints maximize reuse across independent agent sessions.  The
     * stable rendered chat prefix is everything before the user message that
     * asks this specific task.  Some clients put stable user-role scaffolding
     * first, so use the last user marker before the first assistant marker. */
    int last_user = -1;
    for (int i = 0; i < prompt->len; i++) {
        const int token = prompt->v[i];
        if (token == assistant_token_id) break;
        if (token == user_token_id) last_user = i;
    }
    return last_user >= kc->opt.min_tokens ? last_user : -1;
}

static int kv_cache_continued_step(const ds4_kvstore *kc) {
    if (!kc->enabled || kc->opt.continued_interval_tokens <= 0) return 0;
    int step = kc->opt.continued_interval_tokens;
    const int align = kc->opt.boundary_align_tokens;
    if (align > 0) {
        step = ((step + align - 1) / align) * align;
        if (step <= 0) step = align;
    }
    return step;
}

int ds4_kvstore_continued_store_target(const ds4_kvstore *kc, int live_tokens) {
    const int step = kv_cache_continued_step(kc);
    if (step <= 0) return 0;
    if (live_tokens < kc->opt.min_tokens) return 0;
    if (live_tokens % step != 0) return 0;
    if (live_tokens <= kc->continued_last_store_tokens) return 0;
    return live_tokens;
}

void ds4_kvstore_note_store(ds4_kvstore *kc, int tokens) {
    if (tokens > kc->continued_last_store_tokens) {
        kc->continued_last_store_tokens = tokens;
    }
}

int ds4_kvstore_suppress_continued_store(ds4_kvstore *kc, int tokens) {
    if (ds4_kvstore_continued_store_target(kc, tokens) != tokens) return -1;
    int old = kc->continued_last_store_tokens;
    ds4_kvstore_note_store(kc, tokens);
    return old;
}

void ds4_kvstore_restore_suppressed_continued(ds4_kvstore *kc,
                                              int old_tokens,
                                              int suppressed_tokens) {
    if (old_tokens >= 0 && kc->continued_last_store_tokens == suppressed_tokens) {
        kc->continued_last_store_tokens = old_tokens;
    }
}

static bool kv_cache_file_size_bytes(uint64_t text_bytes,
                                     uint64_t payload_bytes,
                                     uint64_t trailer_bytes,
                                     uint64_t *file_bytes) {
    const uint64_t fixed = DS4_KVSTORE_FIXED_HEADER + 4ull;
    if (UINT64_MAX - fixed < text_bytes ||
        UINT64_MAX - fixed - text_bytes < payload_bytes ||
        UINT64_MAX - fixed - text_bytes - payload_bytes < trailer_bytes)
        return false;
    if (file_bytes) *file_bytes = fixed + text_bytes + payload_bytes + trailer_bytes;
    return true;
}

static bool kv_cache_budget_required(uint64_t file_bytes,
                                     uint64_t *required_bytes) {
    /* The serialized size is deterministic for one snapshot, including the
     * optional trailer.  Reserve 1% headroom so filesystem/accounting surprises
     * cannot produce a file that is immediately removed by the budget pass. */
    uint64_t slack = file_bytes / 100u;
    if (file_bytes % 100u) slack++;
    if (UINT64_MAX - file_bytes < slack) return false;
    if (required_bytes) *required_bytes = file_bytes + slack;
    return true;
}

bool ds4_kvstore_file_size_fits(const ds4_kvstore *kc,
                                uint64_t text_bytes,
                                uint64_t payload_bytes,
                                uint64_t trailer_bytes,
                                uint64_t *file_bytes_out,
                                uint64_t *required_bytes_out) {
    uint64_t file_bytes = 0;
    if (!kv_cache_file_size_bytes(text_bytes, payload_bytes, trailer_bytes,
                                  &file_bytes))
        return false;
    if (file_bytes_out) *file_bytes_out = file_bytes;
    if (!kc || kc->budget_bytes == 0) return true;
    uint64_t required = 0;
    if (!kv_cache_budget_required(file_bytes, &required)) return false;
    if (required_bytes_out) *required_bytes_out = required;
    return required <= kc->budget_bytes;
}

static bool kv_cache_file_text_matches(const char *path, const char sha[41],
                                       const char *text, size_t text_len) {
    if (text_len > UINT32_MAX) return false;
    FILE *fp = fopen(path, "rb");
    if (!fp) return false;

    ds4_kvstore_entry hdr = {0};
    uint32_t text_bytes = 0;
    bool ok = ds4_kvstore_read_header(fp, &hdr, &text_bytes) &&
              text_bytes == (uint32_t)text_len;
    char *stored = NULL;
    if (ok) {
        stored = kv_xmalloc((size_t)text_bytes + 1);
        ok = fread(stored, 1, text_bytes, fp) == text_bytes;
    }
    fclose(fp);
    if (!ok) {
        free(stored);
        return false;
    }

    char stored_sha[41];
    ds4_kvstore_sha1_bytes_hex(stored, text_bytes, stored_sha);
    ok = !strcmp(stored_sha, sha) &&
         (text_len == 0 || memcmp(stored, text, text_len) == 0);
    free(stored);
    return ok;
}

static bool kv_cache_existing_compatible(ds4_kvstore *kc, const char *path,
                                         const char sha[41],
                                         const char *text, size_t text_len,
                                         int model_id, int quant_bits, int ctx_size) {
    if (access(path, F_OK) != 0) return false;
    ds4_kvstore_entry e = {0};
    if (!ds4_kvstore_read_entry_file(path, sha, &e)) return false;
    bool compatible = e.model_id == (uint8_t)model_id &&
                      (!kc->reject_different_quant ||
                       e.quant_bits == (uint8_t)quant_bits) &&
                      e.ctx_size <= (uint32_t)ctx_size &&
                      kv_cache_file_text_matches(path, sha, text, text_len);
    ds4_kvstore_entry_free(&e);
    if (!compatible) {
        if (unlink(path) == 0) {
            kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                    "%s: kv cache replaced incompatible file %s",
                    kv_log_name(kc), path);
        }
        return false;
    }
    return true;
}

static bool kv_trailer_serialized_size(const ds4_kvstore_trailer_hooks *hooks,
                                       const char *text,
                                       uint64_t *bytes_out) {
    if (bytes_out) *bytes_out = 0;
    if (!hooks || !hooks->serialized_size) return true;
    return hooks->serialized_size(hooks->ud, text, bytes_out);
}

static bool kv_trailer_write(const ds4_kvstore_trailer_hooks *hooks,
                             FILE *fp, const char *text,
                             uint64_t *written_bytes) {
    if (written_bytes) *written_bytes = 0;
    if (!hooks || !hooks->write) return true;
    return hooks->write(hooks->ud, fp, text, written_bytes);
}

static void kv_cache_rewrite_trailer(ds4_kvstore *kc, const char *path,
                                     const char *text,
                                     const ds4_kvstore_trailer_hooks *hooks) {
    uint64_t trailer_est = 0;
    if (!hooks || !hooks->write || !hooks->serialized_size ||
        !kv_trailer_serialized_size(hooks, text, &trailer_est) ||
        trailer_est == 0)
    {
        return;
    }
    FILE *fp = fopen(path, "r+b");
    if (!fp) return;
    ds4_kvstore_entry hdr = {0};
    uint32_t text_bytes = 0;
    bool ok = ds4_kvstore_read_header(fp, &hdr, &text_bytes);
    uint64_t end = DS4_KVSTORE_FIXED_HEADER + 4ull +
                   (uint64_t)text_bytes + hdr.payload_bytes;
    if (ok && end <= (uint64_t)INT64_MAX &&
        fseeko(fp, (off_t)end, SEEK_SET) == 0 &&
        ftruncate(fileno(fp), (off_t)end) == 0)
    {
        uint64_t ignored = 0;
        ok = kv_trailer_write(hooks, fp, text, &ignored) && fflush(fp) == 0;
        if (ok && ignored > 0) {
            /* ext flags (6) and last use (32..39) only: the identity, canary
             * and payload hash stay as written */
            uint8_t ext = (uint8_t)(hdr.ext_flags | hooks->ext_flag), tb[8];
            kv_le_put64(tb, (uint64_t)time(NULL));
            ok = fseeko(fp, 6, SEEK_SET) == 0 && fwrite(&ext, 1, 1, fp) == 1 &&
                 fseeko(fp, 32, SEEK_SET) == 0 && fwrite(tb, 1, sizeof(tb), fp) == sizeof(tb) &&
                 fflush(fp) == 0;
        }
    }
    fclose(fp);
    (void)kc;
    (void)ok;
}

bool ds4_kvstore_write_entry(FILE *fp, const uint8_t h[DS4_KVSTORE_FIXED_HEADER],
                             const char *text, size_t text_len,
                             const ds4_session_payload_file *staged,
                             const ds4_kvstore_trailer_hooks *hooks,
                             uint64_t *trailer_bytes, uint64_t *payload_hash,
                             char *err, size_t err_len) {
    uint8_t tb[4];
    ds4_kvstore_le_put32(tb, (uint32_t)text_len);
    return fwrite(h, 1, DS4_KVSTORE_FIXED_HEADER, fp) == DS4_KVSTORE_FIXED_HEADER &&
           fwrite(tb, 1, sizeof(tb), fp) == sizeof(tb) &&
           fwrite(text, 1, text_len, fp) == text_len &&
           (staged ? kv_copy_staged_hashed(staged, fp, payload_hash, err, err_len) == 0 :
                     (*payload_hash = ds4_kvstore_payload_hash(NULL, 0), true)) &&
           kv_trailer_write(hooks, fp, text, trailer_bytes) &&
           kv_put_payload_hash(fp, *payload_hash);
}

bool ds4_kvstore_store_live_prefix_text(ds4_kvstore *kc,
                                        ds4_engine *engine,
                                        ds4_session *session,
                                        const ds4_tokens *tokens,
                                        int store_len,
                                        const char *reason,
                                        const char *cache_text_override,
                                        uint8_t cache_text_ext,
                                        const char *cache_text_key,
                                        const ds4_kvstore_trailer_hooks *hooks,
                                        char *err,
                                        size_t err_len) {
    if (!kc->enabled) return false;
    if (!tokens || store_len < kc->opt.min_tokens) return false;
    if (!kv_canary_gate(kc, "store")) return false;
    const int original_len = tokens->len;

    ds4_tokens store_tokens = {0};
    ds4_kvstore_tokens_copy_prefix(&store_tokens, tokens, store_len);

    if (!g_kv_identity) g_kv_identity = ds4_kvstore_engine_identity(engine);
    const int quant_bits = ds4_engine_routed_quant_bits(engine);
    if (!ds4_kvstore_quant_bits_supported(quant_bits)) {
        ds4_tokens_free(&store_tokens);
        return false;
    }
    const int model_id = ds4_engine_model_id(engine);

    char save_err[160] = {0};
    const ds4_tokens *live_tokens = ds4_session_tokens(session);
    if (!live_tokens ||
        live_tokens->len != store_tokens.len ||
        !ds4_tokens_starts_with(live_tokens, &store_tokens))
    {
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache skipped tokens=%d reason=%s because live checkpoint is at %d",
                kv_log_name(kc),
                store_tokens.len,
                reason,
                live_tokens ? live_tokens->len : -1);
        ds4_tokens_free(&store_tokens);
        return false;
    }

    size_t text_len = 0;
    char *text = NULL;
    const bool text_override = cache_text_override && cache_text_override[0];
    if (text_override) {
        text = kv_xstrdup(cache_text_override);
        text_len = strlen(text);
    } else {
        text = ds4_kvstore_render_tokens_text(engine, &store_tokens, &text_len);
    }
    if (text_len > UINT32_MAX) {
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache skipped tokens=%d because rendered text is too large",
                kv_log_name(kc), store_tokens.len);
        free(text);
        ds4_tokens_free(&store_tokens);
        return false;
    }

    uint64_t trailer_est_bytes = 0;
    if (!kv_trailer_serialized_size(hooks, text, &trailer_est_bytes)) {
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache skipped tokens=%d reason=%s because tool map size overflowed",
                kv_log_name(kc), store_tokens.len, reason);
        free(text);
        ds4_tokens_free(&store_tokens);
        return false;
    }
    char sha[41];
    ds4_kvstore_sha1_bytes_hex(text, text_len, sha);
    char *path = ds4_kvstore_path_for_sha(kc, sha);
    const uint8_t reason_code = ds4_kvstore_reason_code(reason);

    if (kv_cache_existing_compatible(kc, path, sha, text, text_len,
                                     model_id,
                                     quant_bits, ds4_session_ctx(session))) {
        kv_cache_rewrite_trailer(kc, path, text, hooks);
        free(text);
        free(path);
        ds4_tokens_free(&store_tokens);
        return true;
    }

    ds4_session_payload_file staged = {0};
    if (ds4_session_stage_payload(session, &staged,
                                  save_err, sizeof(save_err)) != 0) {
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache skipped tokens=%d reason=%s because KV payload staging failed: %s",
                kv_log_name(kc),
                store_tokens.len,
                reason,
                save_err[0] ? save_err : "unknown error");
        if (err && err_len) snprintf(err, err_len, "%s",
                                     save_err[0] ? save_err : "unknown error");
        free(text);
        free(path);
        ds4_tokens_free(&store_tokens);
        return false;
    }
    uint64_t payload_bytes = staged.bytes;

    uint64_t est_file_bytes = 0, est_required_bytes = 0;
    if (!ds4_kvstore_file_size_fits(kc, (uint64_t)text_len, payload_bytes,
                                    trailer_est_bytes,
                                    &est_file_bytes, &est_required_bytes)) {
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache skipped tokens=%d reason=%s because estimated file size %.2f MiB (%.2f MiB with safety) exceeds budget %.2f MiB",
                kv_log_name(kc),
                store_tokens.len,
                reason,
                (double)est_file_bytes / (1024.0 * 1024.0),
                (double)est_required_bytes / (1024.0 * 1024.0),
                (double)kc->budget_bytes / (1024.0 * 1024.0));
        ds4_session_payload_file_free(&staged);
        free(text);
        free(path);
        ds4_tokens_free(&store_tokens);
        return false;
    }

    ds4_kvstore_eviction_context incoming = {
        .text = text,
        .text_len = text_len,
        .model_id = (uint8_t)model_id,
        .quant_bits = (uint8_t)quant_bits,
        .ctx_size = (uint32_t)ds4_session_ctx(session),
        .reject_different_quant = kc->reject_different_quant,
    };
    ds4_kvstore_evict(kc, live_tokens, est_file_bytes, &incoming);

    kv_buf tmpb = {0};
    kv_buf_printf(&tmpb, "%s.tmp.%ld", path, (long)getpid());
    char *tmp = kv_buf_take(&tmpb);
    const double save_t0 = kv_now_sec();
    FILE *fp = fopen(tmp, "wb");
    if (!fp) {
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache failed to create %s: %s save=%.1f ms",
                kv_log_name(kc), tmp, strerror(errno),
                (kv_now_sec() - save_t0) * 1000.0);
        ds4_session_payload_file_free(&staged);
        free(tmp);
        free(text);
        free(path);
        ds4_tokens_free(&store_tokens);
        return false;
    }

    const uint64_t now = (uint64_t)time(NULL);
    uint8_t h[DS4_KVSTORE_FIXED_HEADER];
    uint8_t ext_flags = trailer_est_bytes > 0 && hooks ? hooks->ext_flag : 0;
    if (text_override) ext_flags |= cache_text_ext;
    ds4_kvstore_fill_header(h, (uint8_t)model_id, (uint8_t)quant_bits,
                            reason_code, ext_flags,
                            (uint32_t)store_tokens.len, 0,
                            (uint32_t)ds4_session_ctx(session),
                            now, now, payload_bytes);
    uint64_t trailer_bytes = 0, payload_hash = 0;
    errno = 0;
    bool ok = ds4_kvstore_write_entry(fp, h, text, text_len, &staged, hooks,
                                      &trailer_bytes, &payload_hash,
                                      save_err, sizeof(save_err)) &&
              fflush(fp) == 0 && fsync(fileno(fp)) == 0;
    int saved_errno = errno;
    if (fclose(fp) != 0) {
        if (!saved_errno) saved_errno = errno;
        ok = false;
    }
    uint64_t final_file_bytes = 0, final_required_bytes = 0;
    bool final_size_over_budget = false;
    if (ok && !ds4_kvstore_file_size_fits(kc, (uint64_t)text_len, payload_bytes,
                                          trailer_bytes,
                                          &final_file_bytes,
                                          &final_required_bytes))
    {
        final_size_over_budget = true;
        ok = false;
    }
    if (ok && rename(tmp, path) != 0) {
        saved_errno = errno;
        ok = false;
    }
    const double save_ms = (kv_now_sec() - save_t0) * 1000.0;
    if (!ok) {
        if (final_size_over_budget) {
            kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                    "%s: kv cache skipped tokens=%d reason=%s because final file size %.2f MiB (%.2f MiB with safety) exceeds budget %.2f MiB save=%.1f ms",
                    kv_log_name(kc),
                    store_tokens.len,
                    reason,
                    (double)final_file_bytes / (1024.0 * 1024.0),
                    (double)final_required_bytes / (1024.0 * 1024.0),
                    (double)kc->budget_bytes / (1024.0 * 1024.0),
                    save_ms);
        } else {
            kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                    "%s: kv cache store failed (%s): %s save=%.1f ms",
                    kv_log_name(kc),
                    reason,
                    saved_errno ? strerror(saved_errno) :
                    (save_err[0] ? save_err : "unknown error"),
                    save_ms);
        }
        if (err && err_len) {
            snprintf(err, err_len, "%s",
                     saved_errno ? strerror(saved_errno) :
                     (save_err[0] ? save_err : "unknown error"));
        }
        unlink(tmp);
    } else {
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache stored tokens=%d trimmed=%d reason=%s key=%s size=%.2f MiB save=%.1f ms payload_hash=%016llx",
                kv_log_name(kc),
                store_tokens.len,
                original_len - store_tokens.len,
                reason,
                text_override ? (cache_text_key ? cache_text_key : "visible-transcript") : "token-text",
                (double)(DS4_KVSTORE_FIXED_HEADER + 4ull + text_len + payload_bytes + trailer_bytes) / (1024.0 * 1024.0),
                save_ms, (unsigned long long)payload_hash);
    }
    ds4_session_payload_file_free(&staged);
    free(tmp);
    free(text);
    free(path);
    ds4_tokens_free(&store_tokens);
    return ok;
}

bool ds4_kvstore_store_live_prefix(ds4_kvstore *kc,
                                   ds4_engine *engine,
                                   ds4_session *session,
                                   const ds4_tokens *tokens,
                                   int store_len,
                                   const char *reason,
                                   const ds4_kvstore_trailer_hooks *hooks,
                                   char *err,
                                   size_t err_len) {
    return ds4_kvstore_store_live_prefix_text(kc, engine, session, tokens,
                                              store_len, reason, NULL, 0, NULL,
                                              hooks, err, err_len);
}

bool ds4_kvstore_maybe_store_continued(ds4_kvstore *kc,
                                       ds4_engine *engine,
                                       ds4_session *session,
                                       const ds4_kvstore_trailer_hooks *hooks,
                                       char *err,
                                       size_t err_len) {
    const ds4_tokens *tokens = ds4_session_tokens(session);
    if (!tokens) return false;
    const int target = ds4_kvstore_continued_store_target(kc, tokens->len);
    if (target == 0) return false;
    if (ds4_kvstore_store_live_prefix(kc, engine, session, tokens, target,
                                      "continued", hooks, err, err_len))
    {
        ds4_kvstore_note_store(kc, target);
        return true;
    }
    return false;
}

int ds4_kvstore_find_text_prefix(ds4_kvstore *kc, const char *prompt_text,
                                 int model_id, int quant_bits, int ctx_size) {
    if (!prompt_text) return -1;
    const size_t prompt_bytes = strlen(prompt_text);
    kv_cache_refresh(kc);
    int best = -1;
    for (int i = 0; i < kc->len; i++) {
        ds4_kvstore_entry *e = &kc->entry[i];
        if (e->refusal) continue;
        if (e->text_bytes > prompt_bytes || e->text_bytes > SIZE_MAX) continue;
        if ((int)e->tokens < kc->opt.min_tokens) continue;
        if (e->model_id != (uint8_t)model_id) continue;
        if ((uint32_t)ctx_size < e->ctx_size) continue;
        if (kc->reject_different_quant && e->quant_bits != (uint8_t)quant_bits) continue;
        if (best >= 0) {
            ds4_kvstore_entry *b = &kc->entry[best];
            if (e->text_bytes < b->text_bytes) continue;
            if (e->text_bytes == b->text_bytes && e->tokens <= b->tokens) continue;
        }
        char sha[41];
        ds4_kvstore_sha1_bytes_hex(prompt_text, (size_t)e->text_bytes, sha);
        if (!strcmp(sha, e->sha)) best = i;
    }
    /* Name each refused entry that would have served a longer prefix. */
    for (int i = 0; i < kc->len; i++) {
        ds4_kvstore_entry *e = &kc->entry[i];
        if (!e->refusal || e->model_id != (uint8_t)model_id) continue;
        if (e->text_bytes > prompt_bytes || e->text_bytes > SIZE_MAX) continue;
        if (best >= 0 && e->text_bytes <= kc->entry[best].text_bytes) continue;
        char sha[41];
        ds4_kvstore_sha1_bytes_hex(prompt_text, (size_t)e->text_bytes, sha);
        if (strcmp(sha, e->sha)) continue;
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache refused %s tokens=%u: %s (entry identity %016llx canary %016llx, "
                "this engine %016llx %016llx); nothing restored from it",
                kv_log_name(kc), e->path, e->tokens, e->refusal,
                (unsigned long long)e->identity, (unsigned long long)e->canary,
                (unsigned long long)g_kv_identity, (unsigned long long)kv_canary());
    }
    return best;
}

int ds4_kvstore_try_load_text(ds4_kvstore *kc,
                              ds4_engine *engine,
                              ds4_session *session,
                              const char *prompt_text,
                              ds4_tokens *effective_prompt,
                              ds4_kvstore_load_result *result,
                              const ds4_kvstore_trailer_hooks *hooks,
                              bool responses_protocol) {
    if (result) memset(result, 0, sizeof(*result));
    if (effective_prompt) effective_prompt->len = 0;
    if (!kc->enabled || !prompt_text) return 0;
    if (!kv_canary_gate(kc, "load")) return 0;
    if (!g_kv_identity) g_kv_identity = ds4_kvstore_engine_identity(engine);
    const int quant_bits = ds4_engine_routed_quant_bits(engine);
    if (!ds4_kvstore_quant_bits_supported(quant_bits)) return 0;
    const int model_id = ds4_engine_model_id(engine);
    const size_t prompt_bytes = strlen(prompt_text);
    int idx = ds4_kvstore_find_text_prefix(kc, prompt_text, model_id, quant_bits,
                                           ds4_session_ctx(session));
    if (idx < 0) return 0;

    ds4_kvstore_entry e = kc->entry[idx];
    char *path = kv_xstrdup(e.path);
    const double load_t0 = kv_now_sec();
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        free(path);
        return 0;
    }
    uint32_t text_bytes = 0;
    ds4_kvstore_entry hdr = {0};
    const char *fail_reason = "invalid header";
    bool header_ok = kv_parse_header(fp, &hdr, &text_bytes);
    if (header_ok && hdr.refusal) {
        header_ok = false;
        fail_reason = hdr.refusal;
    }
    char *cached_text = NULL;
    if (header_ok) {
        if (hdr.model_id != (uint8_t)model_id) {
            header_ok = false;
            fail_reason = "cached checkpoint was written for a different model";
        } else if ((uint64_t)text_bytes > prompt_bytes) {
            header_ok = false;
            fail_reason = "cached text is longer than prompt";
        } else {
            cached_text = kv_xmalloc((size_t)text_bytes + 1);
            if (fread(cached_text, 1, text_bytes, fp) != text_bytes) {
                header_ok = false;
                fail_reason = "truncated cached text";
            } else {
                cached_text[text_bytes] = '\0';
                char text_sha[41];
                ds4_kvstore_sha1_bytes_hex(cached_text, text_bytes, text_sha);
                if (strcmp(text_sha, e.sha)) {
                    header_ok = false;
                    fail_reason = "cached text hash mismatch";
                } else if (!ds4_kvstore_byte_prefix_match(prompt_text, prompt_bytes,
                                                          cached_text, text_bytes)) {
                    header_ok = false;
                    fail_reason = "cached text prefix mismatch";
                }
            }
        }
    }
    /* Verified content before any restore: a payload that does not hash to
     * the header's digest is refused whole and removed; the session is not
     * touched. */
    char err[160] = {0}, why[160] = {0};
    bool corrupt = false;
    if (header_ok && ds4_kvstore_verify_payload(fp, &hdr, why, sizeof(why)) != 0) {
        header_ok = false;
        corrupt = true;
        fail_reason = why;
    }
    int loaded = 0;
    if (header_ok &&
        ds4_session_load_payload(session, fp, hdr.payload_bytes, err, sizeof(err)) == 0)
    {
        const ds4_tokens *loaded_tokens = ds4_session_tokens(session);
        if (loaded_tokens && loaded_tokens->len == (int)hdr.tokens) {
            loaded = (int)hdr.tokens;
            if (effective_prompt) {
                /* The cache lookup was by bytes, but the graph state is still
                 * the exact token history stored in the payload.  Build the
                 * prompt from that exact history and tokenize only the text
                 * suffix after the byte prefix. */
                ds4_kvstore_build_prompt_from_exact_prefix_and_text_suffix(
                    engine, loaded_tokens, prompt_text + text_bytes,
                    effective_prompt);
            }
            if (hooks && hooks->load && (hdr.ext_flags & hooks->ext_flag)) {
                hooks->load(hooks->ud, fp, hooks->load_wanted);
            }
        } else {
            ds4_session_invalidate(session);
            unlink(path);
            kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                    "%s: kv cache discarded corrupt text-prefix payload%s%s %s",
                    kv_log_name(kc),
                    responses_protocol ? " " : "",
                    responses_protocol ? "RESPPROTO" : "",
                    path);
        }
    } else {
        if (header_ok) ds4_session_invalidate(session);
        if (corrupt) unlink(path);
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache %s%s%s %s: %s%s load=%.1f ms",
                kv_log_name(kc),
                header_ok ? "load failed" : "refused",
                responses_protocol ? " " : "",
                responses_protocol ? "RESPPROTO" : "",
                path,
                header_ok ? err : fail_reason,
                header_ok ? "" : corrupt ? "; nothing restored, file removed" : "; nothing restored",
                (kv_now_sec() - load_t0) * 1000.0);
    }
    fclose(fp);

    if (loaded > 0) {
        const double load_ms = (kv_now_sec() - load_t0) * 1000.0;
        kc->continued_last_store_tokens = loaded;
        const char *key_kind = ds4_kvstore_key_kind(hdr.ext_flags);
        ds4_kvstore_touch_file(path, hdr.hits + 1);
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache hit text%s%s tokens=%d text=%u quant=%u key=%s load=%.1f ms file=%s",
                kv_log_name(kc),
                responses_protocol ? " " : "",
                responses_protocol ? "RESPPROTO" : "",
                loaded, text_bytes, hdr.quant_bits, key_kind, load_ms, path);
        if (result) {
            result->tokens = loaded;
            result->text_bytes = text_bytes;
            result->quant_bits = hdr.quant_bits;
            result->ext_flags = hdr.ext_flags;
            result->load_ms = load_ms;
            result->path = kv_xstrdup(path);
        }
    }
    free(cached_text);
    free(path);
    return loaded;
}

void ds4_kvstore_load_result_free(ds4_kvstore_load_result *result) {
    if (!result) return;
    free(result->path);
    memset(result, 0, sizeof(*result));
}
