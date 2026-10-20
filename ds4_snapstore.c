#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "ds4_snapstore.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* ---- hashing: a word-wise FNV-style 64-bit digest (integrity, not security) ---- */

#define SS_PRIME UINT64_C(0x100000001b3)
#define SS_BASIS UINT64_C(0xcbf29ce484222325)

static inline uint64_t ss_mix(uint64_t h, uint64_t w) {
    h ^= w;
    h *= SS_PRIME;
    h ^= h >> 29;
    return h;
}

static inline uint64_t ss_load64(const uint8_t *p) {
    uint64_t w;
    memcpy(&w, p, 8);
    return w;
}

void ds4_ss_hash_init(ds4_ss_hash *s) {
    memset(s, 0, sizeof(*s));
    s->h = SS_BASIS;
}

void ds4_ss_hash_update(ds4_ss_hash *s, const void *p, size_t n) {
    const uint8_t *b = p;
    s->n += n;
    while (s->tl && n) {
        s->tail[s->tl++] = *b++;
        n--;
        if (s->tl == 8) { s->h = ss_mix(s->h, ss_load64(s->tail)); s->tl = 0; }
    }
    while (n >= 8) { s->h = ss_mix(s->h, ss_load64(b)); b += 8; n -= 8; }
    while (n) { s->tail[s->tl++] = *b++; n--; }
}

uint64_t ds4_ss_hash_final(const ds4_ss_hash *s) {
    uint64_t h = s->h;
    if (s->tl) {
        uint8_t t[8] = {0};
        memcpy(t, s->tail, s->tl);
        h = ss_mix(h, ss_load64(t));
    }
    return ss_mix(h, s->n);
}

uint64_t ds4_ss_hash_bytes(const void *p, size_t n) {
    ds4_ss_hash s;
    ds4_ss_hash_init(&s);
    ds4_ss_hash_update(&s, p, n);
    return ds4_ss_hash_final(&s);
}

static void ss_err(char *err, size_t errlen, const char *what, const char *path) {
    if (err && errlen) snprintf(err, errlen, "%s %s: %s", what, path, strerror(errno));
}

int ds4_ss_hash_file(const char *path, uint64_t *bytes_out, uint64_t *hash_out,
                     char *err, size_t errlen) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { ss_err(err, errlen, "cannot open", path); return 1; }
    enum { CHUNK = 1 << 22 };
    uint8_t *buf = malloc(CHUNK);
    if (!buf) { fclose(fp); if (err && errlen) snprintf(err, errlen, "out of memory"); return 1; }
    ds4_ss_hash s;
    ds4_ss_hash_init(&s);
    size_t got;
    while ((got = fread(buf, 1, CHUNK, fp)) > 0) ds4_ss_hash_update(&s, buf, got);
    const int bad = ferror(fp);
    free(buf);
    fclose(fp);
    if (bad) { ss_err(err, errlen, "cannot read", path); return 1; }
    if (bytes_out) *bytes_out = s.n;
    if (hash_out) *hash_out = ds4_ss_hash_final(&s);
    return 0;
}

int ds4_ss_write_file_atomic(const char *path, const void *buf, size_t n, bool replace,
                             char *err, size_t errlen) {
    struct stat st;
    if (!replace && stat(path, &st) == 0) {
        if (err && errlen) snprintf(err, errlen, "%s exists (snapshots are immutable; pick a new name)", path);
        return 1;
    }
    char tmp[4096];
    if ((size_t)snprintf(tmp, sizeof(tmp), "%s.tmp.%ld", path, (long)getpid()) >= sizeof(tmp)) {
        if (err && errlen) snprintf(err, errlen, "path too long: %s", path);
        return 1;
    }
    const int fd = open(tmp, O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (fd < 0) { ss_err(err, errlen, "cannot create", tmp); return 1; }
    const uint8_t *b = buf;
    size_t left = n;
    while (left) {
        const size_t want = left > ((size_t)1 << 28) ? ((size_t)1 << 28) : left;
        const ssize_t w = write(fd, b, want);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) { ss_err(err, errlen, "cannot write", tmp); close(fd); unlink(tmp); return 1; }
        b += w;
        left -= (size_t)w;
    }
    if (fsync(fd) != 0) { ss_err(err, errlen, "cannot fsync", tmp); close(fd); unlink(tmp); return 1; }
    if (close(fd) != 0) { ss_err(err, errlen, "cannot close", tmp); unlink(tmp); return 1; }
    if (!replace && stat(path, &st) == 0) {
        unlink(tmp);
        if (err && errlen) snprintf(err, errlen, "%s appeared while writing (not replaced)", path);
        return 1;
    }
    if (rename(tmp, path) != 0) { ss_err(err, errlen, "cannot rename onto", path); unlink(tmp); return 1; }
    return 0;
}

/* ---- meta ---- */

static char *ss_strndup(const char *s, size_t n) {
    char *r = malloc(n + 1);
    if (!r) return NULL;
    memcpy(r, s, n);
    r[n] = '\0';
    return r;
}

static bool ss_two_word_key(const char *k, size_t n) {
    return (n == 8 && !memcmp(k, "numerics", 8)) || (n == 3 && !memcmp(k, "env", 3));
}

int ds4_ss_meta_parse(const char *text, ds4_ss_meta *m) {
    memset(m, 0, sizeof(*m));
    const char *last = NULL;
    size_t last_n = 0;
    for (const char *p = text; p && *p;) {
        const char *e = strchr(p, '\n');
        const size_t len = e ? (size_t)(e - p) : strlen(p);
        if (len) {
            const char *sp = memchr(p, ' ', len);
            size_t kn = sp ? (size_t)(sp - p) : len;
            const char *v = sp ? sp + 1 : p + len;
            if (sp && ss_two_word_key(p, kn)) {
                const char *eq = memchr(v, '=', len - (size_t)(v - p));
                if (eq) { kn = (size_t)(eq - p); v = eq + 1; }
            }
            if (m->n == m->cap) {
                m->cap = m->cap ? m->cap * 2 : 32;
                ds4_ss_line *nl = realloc(m->line, (size_t)m->cap * sizeof(*nl));
                if (!nl) { ds4_ss_meta_free(m); return 1; }
                m->line = nl;
            }
            m->line[m->n].key = ss_strndup(p, kn);
            m->line[m->n].val = ss_strndup(v, len - (size_t)(v - p));
            if (!m->line[m->n].key || !m->line[m->n].val) { m->n++; ds4_ss_meta_free(m); return 1; }
            m->n++;
            last = p;
            last_n = len;
        }
        p = e ? e + 1 : NULL;
    }
    m->complete = last && last_n == 3 && !memcmp(last, "end", 3);
    return 0;
}

void ds4_ss_meta_free(ds4_ss_meta *m) {
    for (int i = 0; m && i < m->n; i++) { free(m->line[i].key); free(m->line[i].val); }
    if (m) free(m->line);
    if (m) memset(m, 0, sizeof(*m));
}

const char *ds4_ss_meta_get(const ds4_ss_meta *m, const char *key) {
    for (int i = 0; i < m->n; i++)
        if (!strcmp(m->line[i].key, key)) return m->line[i].val;
    return NULL;
}

bool ds4_ss_meta_get_u64(const ds4_ss_meta *m, const char *key, int base, uint64_t *out) {
    const char *v = ds4_ss_meta_get(m, key);
    if (!v || !*v) return false;
    char *end = NULL;
    errno = 0;
    const unsigned long long x = strtoull(v, &end, base);
    if (errno || end == v || (*end && *end != ' ')) return false;
    *out = (uint64_t)x;
    return true;
}

const char *ds4_ss_verdict_name(ds4_ss_verdict v) {
    switch (v) {
    case DS4_SS_OK: return "compatible";
    case DS4_SS_LEGACY: return "legacy";
    case DS4_SS_INCOMPLETE: return "incomplete";
    case DS4_SS_CORRUPT: return "corrupt";
    case DS4_SS_INCOMPATIBLE: return "incompatible";
    case DS4_SS_UNVERIFIABLE: return "unverifiable";
    }
    return "?";
}

/* "PATH SIZE MTIME SHA" -> "SIZE SHA" (path/mtime are provenance). */
static void ss_weight_key(const char *v, char *out, size_t cap) {
    char path[4096], size[32], mtime[32], sha[80];
    if (!v || !strcmp(v, "-")) { snprintf(out, cap, "-"); return; }
    if (sscanf(v, "%4095s %31s %31s %79s", path, size, mtime, sha) != 4) { snprintf(out, cap, "?%s", v); return; }
    snprintf(out, cap, "%s %s", size, sha);
}

static const char *ss_category(const char *key) {
    if (!strcmp(key, "model")) return "model weights/tokenizer differ";
    if (!strcmp(key, "drafter")) return "drafter (capture state producer) differs";
    if (!strcmp(key, "armed")) return "capture state (armed drafter) differs";
    if (!strcmp(key, "contract")) return "prefill producer contract differs";
    if (!strcmp(key, "payload_version")) return "engine state format differs";
    if (!strcmp(key, "ctx") || !strcmp(key, "prefill_chunk")) return "context allocation / prefill chunking differs";
    if (!strcmp(key, "vocab")) return "vocabulary differs";
    if (!strcmp(key, "prefill_repr")) return "prefill weight representation differs";
    if (!strcmp(key, "prefill_values")) return "prefill narrow-copy value source differs";
    if (!strcmp(key, "canary")) return "prefill numerics changed (canary logits differ)";
    if (!strncmp(key, "numerics ", 9)) return "prefill numerics switch differs";
    return "identity differs";
}

static bool ss_compat_key(const char *key) {
    static const char *keys[] = {"contract", "payload_version", "model", "drafter", "armed", "ctx",
                                 "prefill_chunk", "vocab", "prefill_repr", "prefill_values", "canary"};
    for (size_t i = 0; i < sizeof(keys) / sizeof(*keys); i++) if (!strcmp(key, keys[i])) return true;
    return !strncmp(key, "numerics ", 9);
}

ds4_ss_verdict ds4_ss_compat(const ds4_ss_meta *have, const ds4_ss_meta *need,
                             char *why, size_t why_len) {
    if (why && why_len) why[0] = '\0';
    const char *fmt = ds4_ss_meta_get(have, "format");
    if (!fmt) { if (why && why_len) snprintf(why, why_len, "no format line"); return DS4_SS_INCOMPLETE; }
    if (!strcmp(fmt, DS4_SS_FORMAT_LEGACY)) return DS4_SS_LEGACY;
    if (strcmp(fmt, DS4_SS_FORMAT) != 0) {
        if (why && why_len) snprintf(why, why_len, "unknown snapshot format '%s'", fmt);
        return DS4_SS_INCOMPATIBLE;
    }
    if (!have->complete) {
        if (why && why_len) snprintf(why, why_len, "metadata has no end marker (interrupted save)");
        return DS4_SS_INCOMPLETE;
    }
    for (int i = 0; i < need->n; i++) {
        const char *k = need->line[i].key, *nv = need->line[i].val;
        if (!ss_compat_key(k)) continue;
        const char *hv = ds4_ss_meta_get(have, k);
        if (!strcmp(k, "canary") && !hv) continue;   /* policy belongs to the caller */
        if (!strcmp(k, "model") || !strcmp(k, "drafter")) {
            char a[160], b[160];
            ss_weight_key(hv, a, sizeof(a));
            ss_weight_key(nv, b, sizeof(b));
            if (strstr(b, "unverifiable") || strstr(a, "unverifiable")) {
                if (why && why_len) snprintf(why, why_len, "%s identity has no recorded sha256 (cache %s, need %s)", k, a, b);
                return DS4_SS_UNVERIFIABLE;
            }
            if (strcmp(a, b) != 0) {
                if (why && why_len) snprintf(why, why_len, "%s: cache has '%s', need '%s'", ss_category(k), a, b);
                return DS4_SS_INCOMPATIBLE;
            }
            continue;
        }
        if (!hv || strcmp(hv, nv) != 0) {
            if (why && why_len) snprintf(why, why_len, "%s: cache has %s='%s', need '%s'",
                                         ss_category(k), k, hv ? hv : "(absent)", nv);
            return DS4_SS_INCOMPATIBLE;
        }
    }
    /* A recorded canary binds the snapshot to prefill numerics: if this
     * process could not recompute it, the state is unverified, not accepted. */
    const char *hc = ds4_ss_meta_get(have, "canary");
    if (hc && !ds4_ss_meta_get(need, "canary")) {
        if (why && why_len) snprintf(why, why_len, "snapshot records canary '%s' but this process did not "
                                     "recompute it (canary prefill failed)", hc);
        return DS4_SS_UNVERIFIABLE;
    }
    /* A switch the cache recorded as set but this binary no longer reports. */
    for (int i = 0; i < have->n; i++) {
        const char *k = have->line[i].key;
        if (strncmp(k, "numerics ", 9) != 0 || ds4_ss_meta_get(need, k)) continue;
        if (strcmp(have->line[i].val, "(unset)") != 0) {
            if (why && why_len) snprintf(why, why_len, "%s: cache has %s='%s', this build does not know it",
                                         ss_category(k), k, have->line[i].val);
            return DS4_SS_INCOMPATIBLE;
        }
    }
    return DS4_SS_OK;
}
