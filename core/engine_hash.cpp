/* engine_hash.cpp -- RADIANCE_EXPERIMENTAL_DS41_PREFILL_HASH=1: the exactness evidence for the
 * consumer skip (core/sched/consumer_skip.h), usable with or without it. At the end of each
 * prompt's prefill (the step whose entry samples the prompt's first token), before the sampler and
 * before any later step runs, every rank hashes, per KV group and layer, what the prompt left that
 * a later step can read, and the last row's logits (its vocabulary shard, before the sampler):
 *
 *   FULL groups    every whole block of the prompt's positions (a compressed or index group of
 *                  ratio r holds one entry a block of r, written once the block is complete)
 *   WINDOW groups  the last `window` positions (the declared window; a decode reads them all)
 *   LINEAR / CONV  the sequence's state slot (copy 0)
 *
 * and logs, per rank, one line of group hashes (each layer's folded in order) and one of each layer
 * of the window group apart (the consumer skip's layers, where a difference would show), then one
 * line folding every rank's. Two runs with the same requests in the same order compare line by line
 * (tools/ds41-host-experts/prefill_hash_compare.py). Debug only: each rank synchronises its stream
 * and copies layer pools to the host (only referenced physical runs with growing KV), and steps
 * are not issued ahead while it is on. */
#include "engine_priv.h"
#include "mem/kv_layout_receipt.h"
#include "device/device.h"
#include "mem/kv_storage_decode.h"
#include "mem/kv_hash_runs.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace rad {
namespace {

constexpr uint64_t kSeed = 0x9e3779b97f4a7c15ull;

/* A 64-bit hash of a byte run, eight bytes at a time; the tail zero-padded and its length mixed
 * in. Deterministic across runs and machines, which is all it needs to be. */
uint64_t hash_bytes(uint64_t h, const uint8_t* p, int64_t n) {
    const uint64_t k = 0x100000001b3ull;
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t w;
        std::memcpy(&w, p + i, 8);
        h = (h ^ w) * k;
        h ^= h >> 29;
    }
    if (i < n) {
        uint64_t w = 0;
        std::memcpy(&w, p + i, (size_t)(n - i));
        h = (h ^ w) * k;
        h ^= h >> 29;
    }
    h = (h ^ (uint64_t)n) * k;
    return h ^ (h >> 31);
}
uint64_t fold(uint64_t h, uint64_t v) {
    return hash_bytes(h, reinterpret_cast<const uint8_t*>(&v), 8);
}

}  /* namespace */

void Engine::prefill_hash_init() {
    const char* v = std::getenv("RADIANCE_EXPERIMENTAL_DS41_PREFILL_HASH");
    prefill_hash_ = v && !std::strcmp(v, "1");
    if (!prefill_hash_) return;
    hash_job_ = std::make_unique<PrefillHashJob>();
    pipelining_ = false;
    RAD_INFO("prefill hash: on (RADIANCE_EXPERIMENTAL_DS41_PREFILL_HASH) -- at each prompt's last "
             "prefill step every rank hashes its KV groups (each layer; whole blocks of a full "
             "group, the last window of a windowed one, the state slots) and the last row's logits; "
             "steps are not issued ahead while it is on");
}

/* With the ranks parked: the step's entries that end a prompt's prefill, and their tables. */
void Engine::prefill_hash_stage(const StepPlan& plan, const StepOut& so, int step) {
    PrefillHashJob& job = *hash_job_;
    job.step = step;
    job.seqs.clear();
    (void)sched_.locked([&]() -> int {
        const KVManager& kv = ranks_[0]->kv;
        const Program& p = ranks_[0]->program;
        for (size_t i = 0; i < plan.e.size(); ++i) {
            const StepEntry& e = plan.e[i];
            if (!e.is_prefill || !e.produces_token || !e.req || i >= so.first_row.size() ||
                so.first_row[i] < 0)
                continue;
            PrefillHashJob::Seq sq;
            sq.seq = e.seq;
            sq.n = (int64_t)e.ctx_len + e.n_tokens;
            sq.row = so.first_row[i];
            sq.cut = e.consumer_end;
            for (int32_t g = 1; g < (int32_t)p.kv_groups.size(); ++g) {
                const KVGroupPlan* gp = kv.plan(g);
                if (!gp) continue;
                PrefillHashJob::Group gr;
                gr.g = g;
                gr.kind = gp->kind;
                gr.block = gp->block_size;
                gr.window = p.kv_groups[(size_t)g].decl.window;
                if (gp->paged()) {
                    gr.table = kv.block_table(e.seq, g);
                    gr.first = kv.first_block_pos(e.seq, g);
                } else {
                    gr.state = kv.state_slot(e.seq, g);
                }
                sq.groups.push_back(std::move(gr));
            }
            job.seqs.push_back(std::move(sq));
        }
        return RAD_OK;
    });
}

/* On a rank thread, right after it issued the step's forward pass and before its sampler. */
void Engine::prefill_hash_rank(Rank& r, const RadBatch* b) {
    r.hash_lines.clear();
    r.hash_digest.clear();
    const PrefillHashJob& job = *hash_job_;
    if (!b || b->draft_pass != 0 || job.step != b->step || job.seqs.empty()) return;
    if (rad_stream_sync(r.ctx.stream()) < 0) {
        r.hash_lines.push_back(fmt("prefill hash: rank %d: the stream did not drain", r.index));
        return;
    }
    const Program& p = r.program;
    const bool receipt = kv_layout_flag(std::getenv(kKVLayoutEnv)) == 1;
    uint64_t receipt_seq = 0;
    int receipt_group = 0, receipt_layer = -1;
    /* One layer's pool at a time, through a pinned buffer as large as the largest. */
    auto fetch = [&](const void* src, int64_t bytes, const KVHashRuns* sparse = nullptr) -> const uint8_t* {
        const auto record = [&](int64_t copied, bool ok) {
            if (receipt) r.hash_lines.push_back(fmt("DS41_KV_HASH_COPY rank=%d seq=%llu group=%d layer=%d requested=%lld capacity=%lld "
                "copied=%lld sparse=%d runs=%zu ok=%d", r.index, (unsigned long long)receipt_seq,
                receipt_group, receipt_layer, (long long)bytes, (long long)r.hash_cap, (long long)copied, (int)bool(sparse),
                sparse ? sparse->size() : size_t(1), (int)ok));
        };
        if (bytes <= 0 || !src) return nullptr;
        if (bytes > r.hash_cap) {
            if (r.hash_host) rad_dev_free(r.hash_host, RAD_MEM_HOST_PINNED);
            r.hash_host = static_cast<uint8_t*>(rad_dev_alloc(bytes, RAD_MEM_HOST_PINNED));
            r.hash_cap = r.hash_host ? bytes : 0;
            if (!r.hash_host) { record(0, false); return nullptr; }
        }
        int copied = RAD_OK;
        int64_t copy_bytes = sparse ? 0 : bytes;
        if (sparse) {
            for (const auto& run : *sparse)
                if (run.first < 0 || run.second < 0 || run.first > bytes || run.second > bytes - run.first ||
                    run.second == 0 || (receipt && !kv_layout_add(copy_bytes, run.second, &copy_bytes))) {
                    record(0, false); return nullptr;
                }
            for (const auto& run : *sparse) {
                copied = rad_memcpy_async(r.hash_host + run.first, (const char*)src + run.first,
                                          run.second, r.ctx.stream());
                if (copied < 0) break;
            }
        } else copied = rad_memcpy_async(r.hash_host, src, bytes, r.ctx.stream());
        const int drained = rad_stream_sync(r.ctx.stream()); // retire earlier runs even if a later copy failed
        if (copied < 0 || drained < 0) { record(0, false); return nullptr; }
        record(copy_bytes, true);
        return r.hash_host;
    };
    for (const PrefillHashJob::Seq& sq : job.seqs) {
        receipt_seq = sq.seq;
        std::string line = fmt("prefill hash: seq %llu, %lld tokens (%s), rank %d:",
                               (unsigned long long)sq.seq, (long long)sq.n,
                               sq.cut > 0 ? fmt("cut for %lld", (long long)sq.cut).c_str() : "not cut",
                               r.index);
        std::string per_layer;
        uint64_t all = kSeed;
        for (const PrefillHashJob::Group& gr : sq.groups) {
            const KVGroupInfo& gi = p.kv_groups[(size_t)gr.g];
            const std::string& name = gi.name;
            const KVPoolBinding* kb = r.ctx.kv_binding((rad_kvgroup)gr.g);
            const KVGroupPlan* gp = r.kv.plan(gr.g);
            if (!kb || !kb->base || !gp || gp->zero || gp->streamed() || gr.kind == RAD_KV_RING) {
                line += fmt(" %s=-", name.c_str());
                continue;
            }
            const int64_t esz = rad_dtype_bytes(kb->dtype, 1);
            /* The byte runs this prompt left in one layer's pool. */
            std::vector<std::pair<int64_t, int64_t>> runs;
            if (gp->paged()) {
                /* The binding can be physical U8 rank 2. Hash the original logical BF16 layout
                 * and the same per-position runs regardless of storage, including ratio-2 halves. */
                const bool packed = kv_storage_packed(gi.decl);
                const int64_t nh = packed ? gi.decl.n_head_kv : kb->shape[1];
                const int64_t bs = packed ? gp->block_size : kb->shape[2];
                const int64_t row = packed ? 2 * gi.decl.head_dim * 2 : kb->shape[3] * esz;
                int64_t lo = 0, hi = 0;
                if (gr.kind == RAD_KV_FULL) {
                    hi = bs > 0 ? sq.n / bs * bs : 0;
                } else {   /* WINDOW */
                    hi = sq.n;
                    lo = std::max<int64_t>(gr.first, sq.n - (gr.window > 0 ? gr.window : sq.n));
                }
                for (int64_t pos = lo; pos < hi && bs > 0; ++pos) {
                    const int64_t ti = (pos - gr.first) / bs;
                    if (ti < 0 || ti >= (int64_t)gr.table.size() || gr.table[(size_t)ti] < 0) continue;
                    const int64_t blk = gr.table[(size_t)ti];
                    for (int64_t h = 0; h < nh; ++h)
                        runs.push_back({ ((blk * nh + h) * bs + pos % bs) * row, row });
                }
            } else if (gr.state >= 0) {
                const int64_t sb = gp->layer_bytes_per_state();
                runs.push_back({ (int64_t)gr.state * sb, sb });
            }
            uint64_t gh = kSeed;
            std::string lh;
            int n_layers = 0;
            std::vector<uint16_t> decoded(kv_storage_packed(gi.decl)
                ? (size_t)(2 * gp->block_size * gi.decl.head_dim) : 0);
            KVHashRuns physical_runs;
            const bool sparse = kv_growth_ && gp->kind == RAD_KV_FULL;
            const bool valid_runs = !sparse ||
                kv_hash_physical_runs(gi.decl, gp->block_size, kb->layer_stride, runs, &physical_runs);
            for (int32_t l : gi.layers) {
                if (l < 0 || (size_t)l >= kb->layer_slot.size() || kb->layer_slot[(size_t)l] < 0) continue;
                bool backed = valid_runs;
                const int64_t layer_offset = (int64_t)kb->layer_slot[(size_t)l] * kb->layer_stride;
                if (sparse)
                    for (const auto& run : physical_runs)
                        backed &= r.pools.kv().is_backed(gp->pool_offset + layer_offset + run.first, run.second);
                receipt_group = gr.g; receipt_layer = l;
                if (!backed) {
                    RAD_ERR("prefill hash: unbacked or invalid physical runs in group '%s', layer %d",
                            name.c_str(), (int)l);
                    line += " invalid-physical-hash-run";
                }
                const uint8_t* host = backed ? fetch((const char*)kb->base + layer_offset,
                    kb->layer_stride, sparse ? &physical_runs : nullptr) : nullptr;
                uint64_t h = kSeed;
                if (!host) {
                    h = 0;
                } else {
                    for (const auto& run : runs) {
                        if (kv_storage_packed(gi.decl)) {
                            const uint8_t* logical = nullptr;
                            if (!kv_storage_decode_run(gi.decl, gp->block_size, host,
                                    kb->layer_stride, run.first, run.second, decoded.data(),
                                    (int64_t)decoded.size(), &logical)) {
                                RAD_ERR("prefill hash: invalid packed page in group '%s', layer %d",
                                        name.c_str(), (int)l);
                                line += " invalid-packed-page";
                                h = 0;
                                break;
                            }
                            h = hash_bytes(h, logical, run.second);
                        } else if (run.first >= 0 && run.first + run.second <= kb->layer_stride) {
                            h = hash_bytes(h, host + run.first, run.second);
                        }
                    }
                }
                gh = fold(gh, h);
                ++n_layers;
                if (gr.kind == RAD_KV_WINDOW) lh += fmt(" %d:%08llx", (int)l, (unsigned long long)(h >> 32));
            }
            line += fmt(" %s[%d]=%016llx", name.c_str(), n_layers, (unsigned long long)gh);
            all = fold(all, gh);
            if (!lh.empty())
                per_layer = fmt("prefill hash: seq %llu rank %d %s by layer:%s", (unsigned long long)sq.seq,
                                r.index, name.c_str(), lh.c_str());
        }
        /* The last row's logits, this rank's shard, as the head wrote them. */
        const int64_t width = r.sampler.config().n_vocab_local;
        receipt_group = 0; receipt_layer = -1;
        const char* lg = static_cast<const char*>(r.ctx.buf_ptr(p.logits_buf));
        const uint8_t* row = lg && width > 0 && sq.row >= 0
                                 ? fetch(lg + (int64_t)sq.row * width * 4, width * 4) : nullptr;
        const uint64_t lh = row ? hash_bytes(kSeed, row, width * 4) : 0;
        line += fmt(" logits=%016llx", (unsigned long long)lh);
        all = fold(all, lh);
        r.hash_lines.push_back(line);
        if (!per_layer.empty()) r.hash_lines.push_back(per_layer);
        r.hash_digest.push_back(all);
    }
}

/* After the step's join: every rank's lines, then one digest a prompt over the ranks in order. */
void Engine::prefill_hash_print() {
    PrefillHashJob& job = *hash_job_;
    if (job.seqs.empty()) return;
    for (auto& r : ranks_)
        for (const std::string& s : r->hash_lines) RAD_INFO("%s", s.c_str());
    for (size_t i = 0; i < job.seqs.size(); ++i) {
        uint64_t all = kSeed;
        bool whole = true;
        for (auto& r : ranks_) {
            if (i < r->hash_digest.size()) all = fold(all, r->hash_digest[i]);
            else whole = false;
        }
        RAD_INFO("prefill hash: seq %llu, %lld tokens: all ranks %016llx%s",
                 (unsigned long long)job.seqs[i].seq, (long long)job.seqs[i].n,
                 (unsigned long long)all, whole ? "" : " (a rank did not hash)");
    }
    job.seqs.clear();
    job.step = -1;
}

/* The consumer skip's shutdown line: what its cut ran over the prompts it cut. */
void Engine::consumer_skip_report() {
    if (!sched_.consumer_on()) return;
    const BatchBuilder::ConsumerStats& c = sched_.consumer_stats();
    const uint64_t lp = c.full + c.key + c.skipped;
    const SchedMetrics m = sched_.metrics();
    RAD_INFO("consumer skip: %llu prefill passes of cut prompts, %llu of them cut (%llu shared a "
             "step and ran every layer, %llu off the grid): %llu layer-passes run in full, %llu key "
             "ops alone, %llu skipped (%.1f%% of those passes' layer-passes in full); checkpoints "
             "below a prompt's target: %lld written whole, %lld not written; their cones raised "
             "%llu passes by %llu layer-passes in full (the key ops alone stay one a cut pass), %llu "
             "snapshot passes cut at the last layer's keys, %llu run in full off the plan",
             (unsigned long long)c.passes, (unsigned long long)c.cut, (unsigned long long)c.shared,
             (unsigned long long)c.off_grid, (unsigned long long)c.full, (unsigned long long)c.key,
             (unsigned long long)c.skipped, lp ? 100.0 * (double)c.full / (double)lp : 100.0,
             (long long)m.consumer_ckpt_whole, (long long)m.consumer_ckpt_dropped,
             (unsigned long long)c.ckpt_raised, (unsigned long long)c.ckpt_full,
             (unsigned long long)c.ckpt_cut, (unsigned long long)c.ckpt_insured);
    const uint64_t lr = c.rows_full + c.rows_key + c.rows_skipped;
    if (lr)
        RAD_INFO("consumer skip, per row: %llu layer-rows run in full, %llu key ops alone, %llu skipped "
                 "(%.1f%% in full)", (unsigned long long)c.rows_full, (unsigned long long)c.rows_key,
                 (unsigned long long)c.rows_skipped, 100.0 * (double)c.rows_full / (double)lr);
}

}  /* namespace rad */
