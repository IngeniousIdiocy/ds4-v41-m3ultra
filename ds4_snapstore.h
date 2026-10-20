#ifndef DS4_SNAPSTORE_H
#define DS4_SNAPSTORE_H

/* Durable prefill-state snapshots: integrity hashing, atomic files and the
 * compatibility identity that binds a saved frontier to what produced it.
 * Engine-independent (no model, no GPU), so it is unit-tested on the CPU.
 *
 * A snapshot is PATH (the engine's session payload, ds4_session_save_payload)
 * plus PATH.meta (text, "key value" lines). The meta is written last and ends
 * with "end"; a payload without a complete meta is an incomplete snapshot.
 *
 * Compatibility lines (format 4) and how they compare:
 *   contract, payload_version, armed, ctx, prefill_chunk, vocab, prefill_repr,
 *   numerics NAME=VALUE ...      exact
 *   model / drafter PATH SIZE MTIME SHA256   size + sha256 only (a moved or
 *                                re-copied file is the same weights); "-" = none
 *   canary LEN HASH              exact; recorded but not recomputed = UNVERIFIABLE (the logits
 *                                of a fresh LEN-token prefill: catches a prefill
 *                                numerics change without binding the binary)
 * Everything else (fixture path, timings, env provenance) is informational,
 * except the integrity/lineage fields checked by the loader: depth,
 * prefix_fnv64, logits_fnv64, payload_bytes, payload_hash. */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define DS4_SS_FORMAT "ds4-bench-prefill-cache-4"
#define DS4_SS_FORMAT_LEGACY "ds4-bench-prefill-cache-3"

typedef struct {
    uint64_t h, n;
    uint8_t tail[8];
    unsigned tl;
} ds4_ss_hash;

void ds4_ss_hash_init(ds4_ss_hash *s);
void ds4_ss_hash_update(ds4_ss_hash *s, const void *p, size_t n);
uint64_t ds4_ss_hash_final(const ds4_ss_hash *s);
uint64_t ds4_ss_hash_bytes(const void *p, size_t n);
/* Streams the whole file; bytes_out gets its length. 0 on success. */
int ds4_ss_hash_file(const char *path, uint64_t *bytes_out, uint64_t *hash_out,
                     char *err, size_t errlen);

/* Writes PATH.tmp.<pid>, fsyncs, renames to PATH. Refuses an existing PATH
 * unless replace (immutable snapshots). 0 on success. */
int ds4_ss_write_file_atomic(const char *path, const void *buf, size_t n, bool replace,
                             char *err, size_t errlen);

/* Meta lines. key = first word; for "numerics" and "env" the key is the
 * first two words' "numerics NAME" (value after '='). */
typedef struct {
    char *key;
    char *val;
} ds4_ss_line;

typedef struct {
    ds4_ss_line *line;
    int n, cap;
    bool complete;          /* last non-empty line is "end" */
} ds4_ss_meta;

int ds4_ss_meta_parse(const char *text, ds4_ss_meta *m);
void ds4_ss_meta_free(ds4_ss_meta *m);
const char *ds4_ss_meta_get(const ds4_ss_meta *m, const char *key);
bool ds4_ss_meta_get_u64(const ds4_ss_meta *m, const char *key, int base, uint64_t *out);

typedef enum {
    DS4_SS_OK = 0,
    DS4_SS_LEGACY,          /* format 3: compared as the old exact identity prefix */
    DS4_SS_INCOMPLETE,      /* no meta, no "end", or payload size mismatch */
    DS4_SS_CORRUPT,         /* payload hash mismatch */
    DS4_SS_INCOMPATIBLE,    /* a compatibility line differs */
    DS4_SS_UNVERIFIABLE,    /* a weight identity has no recorded digest, or a
                             * recorded canary was not recomputed */
} ds4_ss_verdict;

const char *ds4_ss_verdict_name(ds4_ss_verdict v);

/* Compare the stored meta with the identity this process would write
 * (`need`, same line format). why/why_len receive a one-line reason naming
 * the category and both values. A canary the snapshot records must be in
 * need and equal (missing: UNVERIFIABLE, differs: INCOMPATIBLE); a snapshot
 * without one is not checked for it. */
ds4_ss_verdict ds4_ss_compat(const ds4_ss_meta *have, const ds4_ss_meta *need,
                             char *why, size_t why_len);

#endif
