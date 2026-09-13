#ifndef DS4_ENGRAM_H
#define DS4_ENGRAM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

enum {
    DS4_ENGRAM_LAYERS = 2,
    DS4_ENGRAM_NGRAM = 4,
    DS4_ENGRAM_HEADS = 8,
    DS4_ENGRAM_COLS = 24,
    DS4_ENGRAM_DIM = 256,
    DS4_ENGRAM_ROW_BYTES = 264,
    DS4_ENGRAM_DEAD = -1
};

typedef struct {
    const uint32_t *token_map;
    uint32_t vocab_size, compressed_vocab_size, pad_id;
    uint32_t rows[DS4_ENGRAM_LAYERS];
    uint64_t multipliers[DS4_ENGRAM_LAYERS][DS4_ENGRAM_NGRAM];
    uint32_t primes[DS4_ENGRAM_LAYERS][DS4_ENGRAM_COLS];
} ds4_engram_layout;

/* Newest first; copying this value also snapshots the complete hash state. */
typedef struct {
    int32_t tail[DS4_ENGRAM_NGRAM - 1];
} ds4_engram_history;

bool ds4_engram_layout_valid(const ds4_engram_layout *layout);
void ds4_engram_history_reset(ds4_engram_history *history);
/* Validate the layout once at model load. Mask zero breaks all n-grams that
 * cross that position. Output is [token][layer][column], including masked rows;
 * the graph must suppress Engram at masked positions, as in the reference. */
bool ds4_engram_hash(const ds4_engram_layout *layout,
                     ds4_engram_history *history,
                     const int *tokens, const uint8_t *mask, size_t count,
                     uint32_t *rows);

typedef struct {
    int fd;
    uint64_t offset;
    uint32_t rows;
} ds4_engram_table;

/* A separate uncached file descriptor, never an mmap or Metal model view.
 * Each GGUF I8 row is 256 E4M3 bytes followed by 8 original E8M0 scales. */
bool ds4_engram_table_open(ds4_engram_table *table, const char *path,
                           uint64_t offset, uint32_t rows);
void ds4_engram_table_close(ds4_engram_table *table);
/* Output uses F32 storage for the reference's BF16-rounded values. No whole
 * table allocation; caller owns count * DIM floats. Failure invalidates output. */
bool ds4_engram_read(const ds4_engram_table *table, const uint32_t *rows,
                     size_t count, float *out);
/* Read COLS rows per token, restoring token order after deduplicated disk reads.
 * Input stride is in row IDs; output is packed [token][COLS][DIM]. Temporary
 * storage is bounded to 384 KiB, independent of the table and prefix size.
 * On macOS, large batches use bounded concurrent pread readers. */
bool ds4_engram_read_batch(const ds4_engram_table *table, const uint32_t *rows,
                           size_t tokens, size_t stride, float *out);
/* Concurrency of the batch reader (prefill wave 2 lever `engram_readers`).
 * Timing only: the rows read, their order and their values are unchanged.
 * Values outside 1..256 are ignored. */
void ds4_engram_set_readers(unsigned readers);

/* One decode step's DS4_ENGRAM_COLS rows, fetched off the caller's thread on a
 * bounded worker pool. Row order, the e4m3 -> BF16-rounded F32 conversion and
 * the EDOM/EINVAL validation are ds4_engram_read's, unchanged; only the thread
 * and the moment of the pread move. `out` holds COLS * DIM floats and is owned
 * by the caller, which must not touch it between begin and wait. Zero the
 * struct once before first use. */
typedef struct {
    const ds4_engram_table *table;
    float *out;
    uint32_t ids[DS4_ENGRAM_COLS];
    int error[DS4_ENGRAM_COLS];
    void *group;            /* dispatch_group_t where available, else NULL */
    bool inflight;
    uint64_t issue_ns, done_ns;
} ds4_engram_fetch;

/* Issues the fetch and returns immediately. Joins a previous fetch on the same
 * struct first. Falls back to a synchronous ds4_engram_read when no worker can
 * be created; read failures are reported by ds4_engram_fetch_wait. */
bool ds4_engram_fetch_begin(ds4_engram_fetch *fetch, const ds4_engram_table *table,
                            const uint32_t *rows, float *out);
/* Blocks until every row is in place. False with errno set from the
 * lowest-indexed failing row, matching the serial read's first failure. */
bool ds4_engram_fetch_wait(ds4_engram_fetch *fetch);
/* Joins and releases the worker group. Safe on a zeroed struct. */
void ds4_engram_fetch_release(ds4_engram_fetch *fetch);

#endif
