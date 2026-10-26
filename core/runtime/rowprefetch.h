/* rowprefetch.h -- reading a host gather's rows ahead of it, from a table on the mapped tier.
 *
 * A table placed on the mapped tier (--ngram-placement disk, or any weight read in place from the
 * container's mapping) is gathered by a host kernel that touches its rows one after another. Most
 * pages are not resident -- an n-gram table is hundreds of millions of rows, larger than RAM -- and
 * the range is MADV_RANDOM, so each first touch of a page is a fault that reads 4 KiB and waits for
 * it before the kernel moves to the next row. DeepSeek-V4.1's Engram reads 24 rows over two planes
 * a token and layer: ~50 faults in a row, ~40 us each on an idle NVMe (and the queue's latency, not
 * its depth, on a busy one), where the drive answers all of them in about one read's time when they
 * are asked together.
 *
 * So before such a kernel runs, the core hands every page its rows cover to the kernel's page
 * cache with madvise(MADV_WILLNEED): that starts the reads and returns, and the faults the gather
 * then takes wait on reads already in flight, all at once. It is a hint and nothing more: the
 * kernel reads the same bytes in the same order, so its output cannot change, and a page the
 * kernel declines to read ahead is read by the fault as before.
 *
 * Which ops: the row gathers of docs/OPS.md, whose operand 0 holds row ids and whose tables are
 * indexed by id - vocab_offset over n_vocab rows (embed_lookup, embed_lookup_q, embed_lookup_ggml,
 * embed_lookup_planes, each plane of the last its own table). An id outside [0, n_vocab) after the
 * offset reads nothing (the gathers write a zero row), so it is skipped here too. The same ops the
 * planner recognises tables by (place/planner.h, is_ngram_table / is_embedding_table). A kernel that
 * fetches a table's rows itself (libavx's embed_lookup_q: direct reads into its own row cache) says
 * so with rad_kernel_fetches_rows (rad_abi.h) and is left alone: the page cache would read each of
 * its rows a second time.
 *
 * RADIANCE_ROW_PREFETCH=0 turns it off, for an A/B of the gather's cost.
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "rad_types.h"

namespace rad {

static const int kRowTablesMax = 4;

/* One issue's read-ahead: the table operands of a row gather that live on the mapped tier. Built
 * where the op is issued (Ctx::issue_body) and kept with a recorded pass's host op, so a played
 * pass reads ahead as well. */
struct RowPrefetch {
    int      n = 0;                         /* tables; 0 = nothing to read ahead */
    int8_t   opd[kRowTablesMax] = {};       /* their operand positions */
    int64_t  row_bytes[kRowTablesMax] = {};
    int64_t  rows = 0;                      /* ids - voff index [0, rows) */
    int64_t  voff = 0;
};

/* Whether `op` is a row gather this applies to. */
bool row_gather_op(const std::string& op);

/* A page run [first, last), page-aligned. */
using PageRun = std::pair<uintptr_t, uintptr_t>;

/* The reusable buffer the runs are collected in. reserve() once; the step path then does not
 * allocate. */
struct RowPrefetchScratch {
    std::vector<PageRun> runs;
    size_t               cap = 0;
    void reserve(size_t n) { runs.reserve(n); cap = n; }
};

/* Rows [0, m) of `ids` (I32, U32, I64 or I16; any strides), read through `rp` against the bound
 * tensors `t` (the table operands' data). Calls `emit` with every merged run, sorted, when the
 * scratch fills or the ids end; returns the pages covered, or -1 for an id dtype it cannot read
 * (nothing is emitted then). Without syscalls, so the page arithmetic is testable on its own. */
template <class Emit>
int64_t row_page_runs(const RowPrefetch& rp, const RadTensor& ids, int64_t m, const RadTensor* t,
                      int64_t page, RowPrefetchScratch* s, Emit&& emit);

/* row_page_runs with madvise(MADV_WILLNEED) as the emitter: pages advised (>= 0), or -1. */
int64_t row_prefetch(const RowPrefetch& rp, const RadTensor& ids, int64_t m, const RadTensor* t,
                     RowPrefetchScratch* s);

/* The system's page size, read once. */
int64_t row_prefetch_page();

/* ------------------------------------------------------------------ implementation */

/* The id at linear index `i`, row-major over shape and strides. False for a dtype it cannot read. */
bool row_id_at(const RadTensor& ids, int64_t i, int64_t* out);

/* Sort the runs and merge the ones that overlap or touch, in place. */
void row_runs_merge(std::vector<PageRun>* runs);

template <class Emit>
int64_t row_page_runs(const RowPrefetch& rp, const RadTensor& ids, int64_t m, const RadTensor* t,
                      int64_t page, RowPrefetchScratch* s, Emit&& emit) {
    if (rp.n <= 0 || m <= 0 || page <= 0 || !ids.data) return 0;
    int64_t probe;
    if (!row_id_at(ids, 0, &probe)) return -1;
    const uintptr_t mask = ~(uintptr_t)(page - 1);
    int64_t pages = 0;
    auto flush = [&] {
        if (s->runs.empty()) return;
        row_runs_merge(&s->runs);
        for (const PageRun& r : s->runs) {
            pages += (int64_t)((r.second - r.first) / (uintptr_t)page);
            emit(r);
        }
        s->runs.clear();
    };
    const size_t cap = s->cap ? s->cap : 1024;
    for (int64_t i = 0; i < m; ++i) {
        int64_t id;
        row_id_at(ids, i, &id);
        id -= rp.voff;
        if (id < 0 || id >= rp.rows) continue;
        for (int k = 0; k < rp.n; ++k) {
            const uint8_t* base = (const uint8_t*)t[rp.opd[k]].data;
            if (!base || rp.row_bytes[k] <= 0) continue;
            const uintptr_t a = (uintptr_t)(base + id * rp.row_bytes[k]);
            const uintptr_t first = a & mask;
            const uintptr_t last = (a + (uintptr_t)rp.row_bytes[k] + (uintptr_t)page - 1) & mask;
            if (s->runs.size() >= cap) flush();
            s->runs.emplace_back(first, last);
        }
    }
    flush();
    return pages;
}

}  /* namespace rad */
