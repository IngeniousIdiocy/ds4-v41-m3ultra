#ifndef DS4_KVSTORE_H
#define DS4_KVSTORE_H

#include "ds4.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* v2 header (80 bytes): the v1 fields at 0..47, then the verified-content
 * binding: [48] engine identity, [56] numerics canary (0 = not computed),
 * [64] payload hash (ds4_kvstore_payload_hash of the payload bytes), [72]
 * reserved zero. v1 files have the 48-byte header. */
#define DS4_KVSTORE_FIXED_HEADER 80u
#define DS4_KVSTORE_V1_HEADER 48u
/* Tokens of the fixed canary prompt: past 2,048 so the sparse index selects. */
#define DS4_KVSTORE_CANARY_TOKENS 4096
#define DS4_KVSTORE_DEFAULT_MB 4096
#define DS4_KVSTORE_HIT_HALF_LIFE_SECONDS (6ull * 60ull * 60ull)

#define DS4_KVSTORE_EXT_TOOL_MAP          (1u << 0)
#define DS4_KVSTORE_EXT_RESPONSES_VISIBLE (1u << 1)
#define DS4_KVSTORE_EXT_THINKING_VISIBLE  (1u << 2)
#define DS4_KVSTORE_EXT_SESSION_TITLE     (1u << 3)

/* Image identity of a checkpoint. The key text holds each image as text_len
 * bytes of \036 at text_offset (the server's normalized marker); token_start
 * and token_count are its rows in the payload's token history; fingerprint is
 * the live path's (SHA-256 of width, height and decoded RGB). An entry with
 * images carries this list at the front of its verified payload, under payload
 * ABI 3, which older readers refuse; text-only entries keep ABI 2. */
#define DS4_KVSTORE_IMAGES_MAX 16
typedef struct {
    uint32_t text_offset, text_len;
    uint32_t token_start, token_count;
    uint8_t fingerprint[32];
} ds4_kvstore_image;

typedef struct {
    uint32_t len;
    uint64_t encoder;   /* vision GGUF fingerprint (ds4_engine_model_fingerprint 2) */
    ds4_kvstore_image v[DS4_KVSTORE_IMAGES_MAX];
} ds4_kvstore_images;

typedef enum {
    DS4_KVSTORE_REASON_UNKNOWN   = 0,
    DS4_KVSTORE_REASON_COLD      = 1,
    DS4_KVSTORE_REASON_CONTINUED = 2,
    DS4_KVSTORE_REASON_EVICT     = 3,
    DS4_KVSTORE_REASON_SHUTDOWN  = 4,
    DS4_KVSTORE_REASON_AGENT_SYSTEM  = 5,
    DS4_KVSTORE_REASON_AGENT_SESSION = 6,
} ds4_kvstore_reason;

typedef enum {
    DS4_KVSTORE_LOG_DEFAULT,
    DS4_KVSTORE_LOG_KVCACHE,
    DS4_KVSTORE_LOG_WARNING,
} ds4_kvstore_log_type;

typedef struct {
    /* The file name is the rendered byte prefix, not the token sequence. The
     * payload still carries the exact tokens and graph state; the hash only
     * answers "does this checkpoint represent the bytes at the front of the
     * incoming prompt?" */
    char sha[41];
    char *path;
    uint8_t quant_bits;
    /* Stored in header byte 7.  Flash is 0 for backward compatibility with
     * older cache files where this reserved byte was always written as zero. */
    uint8_t model_id;
    uint8_t reason;
    uint32_t tokens;
    uint32_t hits;
    uint32_t ctx_size;
    uint8_t ext_flags;
    uint64_t created_at;
    uint64_t last_used;
    uint64_t payload_bytes;
    uint64_t text_bytes;
    uint64_t file_size;
    uint8_t version;
    uint64_t identity, canary, payload_hash;
    bool images;   /* payload ABI 3: the payload starts with an image list */
    /* NULL when this engine may load the entry; otherwise why it is refused
     * (old version, payload ABI, identity or canary). Refused entries are
     * indexed only so eviction can reclaim their space. */
    const char *refusal;
} ds4_kvstore_entry;

typedef struct {
    int min_tokens;
    int cold_max_tokens;
    int continued_interval_tokens;
    int boundary_trim_tokens;
    int boundary_align_tokens;
} ds4_kvstore_options;

typedef struct {
    bool enabled;
    char *dir;
    uint64_t budget_bytes;
    bool reject_different_quant;
    ds4_kvstore_options opt;
    int continued_last_store_tokens;
    ds4_kvstore_entry *entry;
    int len;
    int cap;
    const char *log_name;
    void *log_ud;
    void (*log)(void *ud, ds4_kvstore_log_type type, const char *msg);
} ds4_kvstore;

typedef struct {
    const char *text;
    size_t text_len;
    uint8_t model_id;
    uint8_t quant_bits;
    uint32_t ctx_size;
    bool reject_different_quant;
    const ds4_kvstore_images *images;   /* the incoming entry's, NULL: none */
} ds4_kvstore_eviction_context;

typedef struct {
    void *ud;
    uint8_t ext_flag;
    bool (*serialized_size)(void *ud, const char *text, uint64_t *bytes_out);
    bool (*write)(void *ud, FILE *fp, const char *text, uint64_t *written_bytes);
    int (*load)(void *ud, FILE *fp, const void *wanted);
    const void *load_wanted;
} ds4_kvstore_trailer_hooks;

typedef struct {
    int tokens;
    uint32_t text_bytes;
    uint8_t quant_bits;
    uint8_t ext_flags;
    double load_ms;
    char *path;
} ds4_kvstore_load_result;

ds4_kvstore_options ds4_kvstore_default_options(void);
uint8_t ds4_kvstore_reason_code(const char *reason);
const char *ds4_kvstore_key_kind(uint8_t ext_flags);

bool ds4_kvstore_open(ds4_kvstore *kc, const char *dir, uint64_t budget_mb,
                      bool reject_different_quant, ds4_kvstore_options opt,
                      const char *log_name,
                      void (*log)(void *ud, ds4_kvstore_log_type type, const char *msg),
                      void *log_ud);
void ds4_kvstore_close(ds4_kvstore *kc);
void ds4_kvstore_clear(ds4_kvstore *kc);
void ds4_kvstore_entry_free(ds4_kvstore_entry *e);

char *ds4_kvstore_render_tokens_text(ds4_engine *engine,
                                     const ds4_tokens *tokens,
                                     size_t *out_len);
bool ds4_kvstore_byte_prefix_match(const char *text, size_t text_len,
                                   const char *prefix, size_t prefix_len);
void ds4_kvstore_tokens_copy_prefix(ds4_tokens *dst, const ds4_tokens *src, int n);
void ds4_kvstore_build_prompt_from_exact_prefix_and_text_suffix(
        ds4_engine *engine,
        const ds4_tokens *exact_prefix,
        const char *suffix_text,
        ds4_tokens *out);

int ds4_kvstore_store_len(const ds4_kvstore *kc, int tokens);
int ds4_kvstore_chat_anchor_pos(const ds4_kvstore *kc,
                                const ds4_tokens *prompt,
                                int user_token_id,
                                int assistant_token_id);
int ds4_kvstore_continued_store_target(const ds4_kvstore *kc, int live_tokens);
void ds4_kvstore_note_store(ds4_kvstore *kc, int tokens);
int ds4_kvstore_suppress_continued_store(ds4_kvstore *kc, int tokens);
void ds4_kvstore_restore_suppressed_continued(ds4_kvstore *kc,
                                              int old_tokens,
                                              int suppressed_tokens);

bool ds4_kvstore_file_size_fits(const ds4_kvstore *kc,
                                uint64_t text_bytes,
                                uint64_t payload_bytes,
                                uint64_t trailer_bytes,
                                uint64_t *file_bytes_out,
                                uint64_t *required_bytes_out);
double ds4_kvstore_entry_eviction_score(const ds4_kvstore_entry *e,
                                        const ds4_tokens *live,
                                        uint64_t now,
                                        const ds4_kvstore_eviction_context *incoming);
void ds4_kvstore_evict(ds4_kvstore *kc, const ds4_tokens *live,
                       uint64_t extra_bytes,
                       const ds4_kvstore_eviction_context *incoming);
int ds4_kvstore_find_text_prefix(ds4_kvstore *kc, const char *prompt_text,
                                 int model_id, int quant_bits, int ctx_size);
/* The longest entry whose key is a byte prefix of prompt_text and whose stored
 * images are exactly the request's images inside that prefix (NULL: none).
 * prompt_text has image markers normalized as in the stored keys. */
int ds4_kvstore_find_image_prefix(ds4_kvstore *kc, const char *prompt_text,
                                  const ds4_kvstore_images *images,
                                  int model_id, int quant_bits, int ctx_size);
/* NULL when stored serves a prompt carrying request's images, else why not:
 * the images inside the first text_bytes, in order, must sit at the same
 * offsets with the same fingerprints and row counts, from the same encoder. */
const char *ds4_kvstore_images_mismatch(const ds4_kvstore_images *stored,
                                        const ds4_kvstore_images *request,
                                        uint64_t text_bytes);
/* The image list at the front of an ABI 3 payload (fp at the payload start;
 * left after the list). Returns its size in bytes, 0 when malformed. */
uint64_t ds4_kvstore_read_images(FILE *fp, uint64_t payload_bytes,
                                 ds4_kvstore_images *out);

bool ds4_kvstore_store_live_prefix_text(ds4_kvstore *kc,
                                        ds4_engine *engine,
                                        ds4_session *session,
                                        const ds4_tokens *tokens,
                                        int store_len,
                                        const char *reason,
                                        const char *cache_text_override,
                                        uint8_t cache_text_ext,
                                        const char *cache_text_key,
                                        const ds4_kvstore_images *images,
                                        const ds4_kvstore_trailer_hooks *hooks,
                                        char *err,
                                        size_t err_len);
bool ds4_kvstore_store_live_prefix(ds4_kvstore *kc,
                                   ds4_engine *engine,
                                   ds4_session *session,
                                   const ds4_tokens *tokens,
                                   int store_len,
                                   const char *reason,
                                   const ds4_kvstore_trailer_hooks *hooks,
                                   char *err,
                                   size_t err_len);
bool ds4_kvstore_maybe_store_continued(ds4_kvstore *kc,
                                       ds4_engine *engine,
                                       ds4_session *session,
                                       const ds4_kvstore_trailer_hooks *hooks,
                                       char *err,
                                       size_t err_len);
int ds4_kvstore_try_load_text(ds4_kvstore *kc,
                              ds4_engine *engine,
                              ds4_session *session,
                              const char *prompt_text,
                              ds4_tokens *effective_prompt,
                              ds4_kvstore_load_result *result,
                              const ds4_kvstore_trailer_hooks *hooks,
                              bool responses_protocol,
                              const ds4_kvstore_images *images);
void ds4_kvstore_load_result_free(ds4_kvstore_load_result *result);

bool ds4_kvstore_read_header(FILE *fp, ds4_kvstore_entry *e,
                             uint32_t *text_bytes);
bool ds4_kvstore_read_entry_file(const char *path, const char sha[41],
                                 ds4_kvstore_entry *out);
/* Identity of the weights/representation this store's entries belong to
 * (model + drafter GGUF header fingerprints, quant, variant, prefill repr).
 * Set it before ds4_kvstore_open so foreign entries are never indexed. */
uint64_t ds4_kvstore_engine_identity(ds4_engine *engine);
void ds4_kvstore_set_identity(uint64_t identity);
/* Numerics canary: logits hash of a fresh one-shot prefill of a fixed
 * tokens-long prompt in a side session. Catches a prefill numerics change
 * (kernels, numerics switches) that the identity cannot see. 0 on success.
 * The caller serializes it with other engine work. */
int ds4_kvstore_engine_canary(ds4_engine *engine, int tokens, uint64_t *out,
                              char *err, size_t err_len);
/* Readiness gate. pending: loads and stores are refused until set_canary
 * (ready) or failed (refused for good); ds4-server keeps requests queued
 * until then, so a pending canary never reaches this gate as a miss. With
 * none of these called (ds4-agent) no canary is required or checked. Once
 * set, entries need an equal canary. */
void ds4_kvstore_canary_pending(void);
void ds4_kvstore_set_canary(uint64_t canary);
void ds4_kvstore_canary_failed(void);
bool ds4_kvstore_canary_ready(void);
/* Word-wise 64-bit FNV-style digest (integrity, not security). */
typedef struct {
    uint64_t h, n;
    uint8_t tail[8];
    unsigned tl;
} ds4_kvstore_hash;
void ds4_kvstore_hash_init(ds4_kvstore_hash *s);
void ds4_kvstore_hash_update(ds4_kvstore_hash *s, const void *p, size_t n);
uint64_t ds4_kvstore_hash_final(const ds4_kvstore_hash *s);
uint64_t ds4_kvstore_payload_hash(const void *p, size_t n);
/* fp at the payload start: hashes e->payload_bytes and compares with the
 * header's payload hash, then seeks back. 0 = verified; otherwise why says
 * why the entry is refused. Runs before any session state is touched. */
int ds4_kvstore_verify_payload(FILE *fp, const ds4_kvstore_entry *e,
                               char *why, size_t why_len);
/* The entry body after a filled header: header, text length, text, the
 * staged payload (digested on the way; NULL = text-only entry), the trailer,
 * then the digest into header bytes 64..71. The caller flushes, fsyncs and
 * renames. */
bool ds4_kvstore_write_entry(FILE *fp, const uint8_t h[DS4_KVSTORE_FIXED_HEADER],
                             const char *text, size_t text_len,
                             const ds4_session_payload_file *staged,
                             const ds4_kvstore_trailer_hooks *hooks,
                             uint64_t *trailer_bytes, uint64_t *payload_hash,
                             char *err, size_t err_len);
/* Bytes an image list adds to the payload (0 for none), and the entry body
 * with one: header ABI 3 and payload_bytes grown by that size, the list
 * digested ahead of the staged payload. NULL or no images writes ABI 2. */
uint64_t ds4_kvstore_images_bytes(const ds4_kvstore_images *images);
bool ds4_kvstore_write_entry_images(FILE *fp, const uint8_t h[DS4_KVSTORE_FIXED_HEADER],
                                    const char *text, size_t text_len,
                                    const ds4_kvstore_images *images,
                                    const ds4_session_payload_file *staged,
                                    const ds4_kvstore_trailer_hooks *hooks,
                                    uint64_t *trailer_bytes, uint64_t *payload_hash,
                                    char *err, size_t err_len);
void ds4_kvstore_fill_header(uint8_t h[DS4_KVSTORE_FIXED_HEADER],
                             uint8_t model_id, uint8_t quant_bits,
                             uint8_t reason, uint8_t ext_flags,
                             uint32_t tokens, uint32_t hits, uint32_t ctx_size,
                             uint64_t created_at, uint64_t last_used,
                             uint64_t payload_bytes);
bool ds4_kvstore_touch_file(const char *path, uint32_t hits);
bool ds4_kvstore_sha_hex_name(const char *name, char sha[41]);
void ds4_kvstore_sha1_bytes_hex(const void *ptr, size_t len, char out[41]);
/* An entry's file name: the SHA-1 of its key text, extended for an image entry
 * by the encoder and each image inside the key (offset, marker length, rows,
 * fingerprint). Text-only names are the plain text SHA-1, as before. */
void ds4_kvstore_entry_sha(const char *text, size_t len,
                           const ds4_kvstore_images *images, char out[41]);
char *ds4_kvstore_path_join(const char *dir, const char *name);
char *ds4_kvstore_path_for_sha(ds4_kvstore *kc, const char sha[41]);
void ds4_kvstore_le_put32(uint8_t *p, uint32_t v);
uint32_t ds4_kvstore_le_get32(const uint8_t *p);

bool ds4_kvstore_quant_bits_supported(int quant_bits);

#endif
