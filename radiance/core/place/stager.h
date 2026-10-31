/* stager.h -- the prefill stager: routed layers' non-resident experts copied onto the card ahead
 * of the layers that read them.
 *
 * ---- why prefill needs it ---------------------------------------------------------------------
 *
 * An expert the slab has no room for lives in the pinned host pool, and a GEMM that routes to it
 * reads it across the link in place (Site::DeviceZeroCopy). That suits decode, which reads a few
 * experts per step from the host. A prefill chunk routes to nearly every expert of every layer, so
 * each layer's whole pooled part (~85 MB per card on Qwen3.8-Flash-Next) crosses the link inside
 * its two GEMMs, on the critical path, while the rest of the layer leaves the link idle. With
 * r4d_selftest --perfmoe, a 2048-token gate_up GEMM takes 1.53 ms with every expert on the card
 * and 4.34 ms with the coldest 13% in host memory.
 *
 * So for large enough steps the stager copies the pooled experts of the layers ahead into a VRAM
 * ring while earlier layers run, publishes the copies for each layer's ops, and publishes the host
 * addresses again once those ops are issued. The bytes are the same, so output is bit-identical.
 *
 * ---- the region --------------------------------------------------------------------------------
 *
 * One region, two layers' share of the experts the budget leaves off the card, at the top of the
 * activation arena (Program::arena_stage_off, sized by vram_budget_resolve). A step smaller than
 * the arena's full plan lends that top to the expert slab, so the region belongs to the stager
 * only in full-plan steps (the ones worth staging for). The scheduler says which, per pass
 * (set_ring).
 *
 * Under --prefill-span the ring has a second place. The full plan is sized for a spanned step,
 * and a step of one interval (a chunk that shares its step, a short prompt) runs at the level of
 * that size (Engine::probe_arena_levels). That level keeps the ring right above its own buffers,
 * so everything over it, the span's extra rows included, is lent to the slab while such steps
 * run, and the step is still staged. A pass whose ring is in the other place than the last one's
 * moves it first (rebase).
 *
 * ---- why a ring, not two buffers ---------------------------------------------------------------
 *
 * Layers leave unequal parts off the card: at depth, with the KV cache's loan taken back, the last
 * third of a Qwen3.8-Flash-Next stack pools about twice the median layer. With two fixed buffers,
 * a light layer's copy finishes early and the engine idles, and a heavy layer's copy delays every
 * layer after it. The ring packs each layer at its own size, so copies run ahead through light
 * layers and heavy ones start early. A layer staged beyond the next one takes at most half the
 * ring, so the next layer always has the room two buffers would give it.
 *
 * ---- order and safety --------------------------------------------------------------------------
 *
 *   the pass's first op     stage the first routed layer the pass issues, and the ones after it
 *                           as far as the pace and the ring allow
 *   a layer's first op      wait on its copies; publish the host addresses of earlier layers
 *                           again; stage further layers into the freed room
 *   a layer's last op       record "free" behind it on the op's stream
 *   the pass's end          the same for anything still staged; and, when the next step continues
 *                           a prompt, stage the first layers of its trunk pass
 *
 * A layer's place in the ring is counted in bytes handed out since bring-up, so a copy finds the
 * places it overwrites from the count alone. Before a copy is issued, the mover's stream waits on
 * the "free" event of every earlier place it laps (recorded behind that place's last reader, in
 * this pass or an earlier one). The layer's ops wait on the copies' event through one stream wait
 * before its first op: copies are published settled (see Mover::stage_units), and only the
 * layer's own ops resolve them.
 *
 * ---- how far ahead -----------------------------------------------------------------------------
 *
 * A layer's copies are released once the compute stream reaches the MoE of the layer kLead before
 * it (an event recorded at that layer's first op), not as soon as the ring has room: copies that
 * land long before use slow the layers they overlap. On Qwen3.8-Flash-Next TP2, unpaced copies
 * cost a 26K-token prompt 0.6% and one layer ahead cost a 160K-token prompt 2.4% (its heavy late
 * layers waited); three ahead covered the late layers at the least cost to short prompts.
 *
 * ---- the next step's first layers --------------------------------------------------------------
 *
 * A step carrying a prompt chunk is issued only after the previous step is read back, so its first
 * layer would otherwise wait for all its own copies. When the next step continues a prompt
 * (set_next_prefill), this pass's end stages the next trunk pass's first layers, paced on this
 * pass's last layers as if the passes were one, and publishes them at once. The next trunk pass
 * adopts them when armed; an unarmed trunk pass publishes their host addresses again in
 * begin_pass, which the run phase calls before updating the weight tables. A draft pass in
 * between reads none of the trunk's layers and leaves them.
 *
 * ---- trunk only --------------------------------------------------------------------------------
 *
 * A draft head's routed layer is not staged. The head runs in its own passes; over a prefill chunk
 * (the history pass) it needs its experts only on a prompt's last chunk, for the few rows that
 * draft, so its pooled experts are read in place as at decode. The head's layer is numbered past
 * the trunk's (RadModelMeta::n_layers) and left out at bring-up; a draft pass is never armed.
 */
#pragma once
#include "../runtime/stager.h"
#include "mover.h"
#include "planner.h"
#include "ep_route_counts.h"
#include "dspark_head_tail.h"
#include "host_expert_pool.h"
#include "card_expert_stager.h"
#include "hx_fetch_list.h"
#include "hx_side_store.h"
#include "readahead.h"
#include "route_trace.h"
#include "../sched/tbo.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace rad {

/* One routed layer a stager serves: its units and the span of ops that read them. */
struct StagedLayer {
    int32_t              layer = -1;
    std::vector<int32_t> units;
    int32_t              first_op = -1, last_op = -1;
};

class PrefillStager final : public OpStager {
public:
    ~PrefillStager() override;

    /* Binds the routed layers of `plan` and the ops that read them, with `bytes` of device memory
     * at `region` as the ring -- or at `low`, the ring's place under the span's shared level, when
     * there is one (null otherwise). `min_rows` is the smallest pass it arms for. Leaves enabled()
     * false when no routed layer has a pooled unit to stage. */
    int init(const Program& prog, const Plan& plan, Mover* mover, char* region, char* low,
             int64_t bytes, int64_t min_rows);

    /* Per pass, before release: where the ring is for the step -- the region at the full plan,
     * `low` at the span's shared level -- or null when the step's level lends both. */
    void set_ring(char* base) { want_ = base; }
    /* The ring's two places: at the full plan, and at the span's shared level (the same one when
     * there is no such level). */
    char*   ring_full() const { return region_; }
    char*   ring_low() const { return low_; }
    /* Per step, before release: whether the next step continues a prompt in a pass this stager
     * arms for, so this step's trunk pass stages that pass's first layers. */
    void set_next_prefill(bool more) { next_prefill_ = more; }

    bool    enabled() const { return !layers_.empty(); }
    int64_t ring_bytes() const { return ring_; }
    /* The smallest pass, in tokens, the stager arms for; smaller steps read pooled experts in
     * place. The scheduler keeps a held-back prefill chunk at or above it
     * (Scheduler::set_stage_rows). */
    int64_t min_rows() const { return min_rows_; }
    size_t  n_layers() const { return layers_.size(); }
    /* Whether the last trunk pass armed (copied its pooled experts through the ring). A draft
     * pass leaves it unchanged. */
    bool    staged() const { return staged_; }
    static constexpr int32_t lead() { return kLead; }

    bool begin_pass(const RadBatch* b) override;
    int  before_op(rad_op h, RadStream s, bool* resync) override;
    int  after_op(rad_op h, RadStream s) override;
    int  end_pass(RadStream s) override;

private:
    using Layer = StagedLayer;
    std::vector<Layer>   layers_;      /* the routed layers, in first-op order */
    std::vector<int32_t> first_of_;    /* by op: the layer whose first op it is, or -1 */
    std::vector<int32_t> last_of_;     /* by op: the layer whose last op it is, or -1 */

    /* One layer's place in the ring. [beg, end) is in bytes handed out since bring-up; the region
     * offset is beg modulo its size, and a place never straddles the region's end. */
    struct Place {
        int32_t  li = -1;                      /* index into layers_ */
        int64_t  beg = 0, end = 0;
        RadEvent done = nullptr;               /* behind its copies, on the mover stream */
        RadEvent free = nullptr;               /* behind its last reader */
        bool     free_rec = false;
        /* Its last op has been issued at least once and `free` is recorded behind it. A step
         * longer than the MoE block's pass of rows issues a layer's ops once per pass, so the last
         * op can repeat: each repeat moves `free` behind itself, and the copies are published away
         * only at the next layer's first op. */
        bool     read = false;
        bool     retired = false;              /* its units are published at their pool slots */
        std::vector<int32_t> staged;
    };
    /* More places than a two-share ring can hold at the smallest per-layer share; a full table
     * only stops copies running further ahead. */
    static constexpr int kPlaces = 16;
    Place    place_[kPlaces];
    std::deque<int32_t>  ring_order_;  /* places in the ring, oldest first; retired ones stay until lapped */
    std::vector<int32_t> spare_;       /* places not in the ring */

    Mover*   mover_ = nullptr;
    char*    region_ = nullptr;
    char*    low_ = nullptr;           /* the ring's place at the span's shared level, or null */
    char*    base_ = nullptr;          /* where the places in the ring are */
    char*    want_ = nullptr;          /* where this pass wants it (set_ring) */
    bool     rebase_ = false;          /* the pass moves the ring before its first copy */
    RadEvent moved_t_ = nullptr;       /* the move's two waits: behind the mover's stream, */
    RadEvent moved_c_ = nullptr;       /* and behind the compute stream */
    int64_t  ring_ = 0;
    int64_t  head_ = 0;                /* bytes handed out since bring-up */
    int64_t  min_rows_ = 0;

    /* The pace (see the file header): layer k's copies wait on the compute-stream event recorded
     * at the first op of layer k - kLead. A layer within kLead of the pass's first routed layer
     * goes at once. */
    static constexpr int32_t kLead = 3;
    static constexpr int kGates = 64;
    RadEvent gate_[kGates] = {};
    int32_t  gate_li_[kGates] = {};    /* the layer each gate was recorded at this pass, or -1 */
    int32_t  first_li_ = -1;           /* the pass's first routed layer */

    bool     armed_ = false;
    /* armed_ as the last trunk pass's begin_pass set it; end_pass leaves it, since the engine
     * asks after the pass has ended. */
    bool     staged_ = false;
    bool     next_prefill_ = false;
    bool     first_op_ = false;        /* the pass has issued nothing yet */
    int32_t  next_ = 0;                /* the next layer to stage this pass */
    /* The next trunk pass's first layers, staged at this pass's end: layers [0, n_early_). */
    int32_t  n_early_ = 0;

    enum { kStaged, kNone, kLater };
    int  stage_ahead(int32_t cur, bool* resync);
    int  stage_layer(int32_t li, bool now, RadEvent gate, int* got);
    void withdraw_early();
    int  take(int64_t want, bool partial, int32_t* out);
    int  recycle_front();
    int  rebase(RadStream s);
    void retire(Place& p);
};

/* The read-ahead in pieces' pass policy (FileStager, RADIANCE_EXPERIMENTAL_DS41_PREFETCH=pieces).
 * ra_multirow_rows: RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MULTIROW's value -- 1 unset, empty or "0"
 * (one-token passes only, as always), N for 2 <= N <= limit, -1 for anything else (refused).
 * ra_pass_reads_ahead: every one-token pass, as always, and for max_rows > 1 a trunk decode pass
 * (decode: RAD_PHASE_DECODE; draft_pass 0) of 2..max_rows rows; never a prefill or mixed pass of
 * more rows, nor the drafter's own passes. */
inline int ra_multirow_rows(const char* v, int64_t limit) {
    if (!v || !*v || (v[0] == '0' && !v[1])) return 1;
    int64_t n = 0;
    for (const char* p = v; *p; ++p) {
        if (*p < '0' || *p > '9' || n > 1000) return -1;
        n = n * 10 + (*p - '0');
    }
    return n >= 2 && n <= limit ? (int)n : -1;
}
inline bool ra_pass_reads_ahead(bool decode, int32_t draft_pass, int64_t n_tok, int max_rows) {
    if (n_tok == 1) return true;
    return max_rows > 1 && decode && draft_pass == 0 && n_tok >= 2 && n_tok <= max_rows;
}

/* ---- the file stager: routed layers read from the container -------------------------------------
 *
 * A routed unit on the file tier (--weights-disk-tier) has no address of its own, and a grouped
 * GEMM is handed every expert's pointer at issue. So on every pass, decode included, each routed
 * layer's file-tier units are read into one of the mover's two file buffers before the layer's
 * first expert op and published there, then published as null when the next routed layer starts.
 * The router's picks are not known on the host without a round trip, so the layer's whole
 * file-tier part is read; decode is then paced by the drive, which is the trade the flag makes.
 *
 *   the pass's first op      read the first routed layer the pass issues
 *   a layer's first op       release the layer before (record its buffer free on the stream, publish
 *                            null), wait on this layer's copies, and read the next layer into the
 *                            buffer just released -- its copies queue behind that free event
 *   the pass's end           release whatever is still held
 *
 * A layer is released when the next one starts, not at its own last op: a routed block that slices
 * a large step into passes of `rows` tokens issues its expert ops once per slice, so a release at
 * the last op would leave the next slice reading null. Repeats of the current layer's first op do
 * nothing.
 *
 * The read blocks the issuing thread while the card runs what was issued before it. A layer's
 * file-tier part is gigabytes against milliseconds of compute, so the step is drive-bound either
 * way; a reader thread would gain little and add a second writer of the residency table.
 *
 * A layer is read ahead only when the pass will issue it (a bound learned per draft_pass from the
 * last pass of that kind), so a trunk pass does not read the MTP head's layer. */
class FileStager final : public OpStager {
public:
    ~FileStager() override;

    /* Routing-aware reads: the engine binds a source of the layer's per-expert routing
     * counts for the current pass (the router reports them to the Ctx before the layer's
     * first expert op, which is the stager's trigger). Returns the device pointer of
     * `n_expert` i32 counts, or null when the layer has not reported this pass — the
     * caller then reads the whole layer. Null `routes` reads whole layers. */
    using RouteCounts = const int32_t* (*)(void* ud, int32_t layer, int64_t* n_expert);

    /* A routed layer's card-side counts row for the pass being issued, as Ctx::route_counts_live
     * gives it: the row, the layer's expert count and its top_k, or null when the layer has no
     * row its router writes in stream order. */
    using CountsRow = std::function<const int32_t*(int32_t layer, int64_t* n_expert,
                                                   int64_t* top_k)>;
    using EpCountsRow = std::function<bool(int32_t layer, EpRouteCounts* out)>;
    /* sync: the issuing thread waiting for the card to reach a layer's router (the counts row);
     * stage: its routed units' reads and copies queued (drive reads block it); miss_layers and
     * miss_stage: the layers whose stage read from the drive, and their stage time; hx: the CPU
     * share chosen and handed to the pool. */
    struct EpStats {
        uint64_t reads = 0, zero_local = 0, units = 0, corrupt = 0;
        uint64_t sync_ns = 0, stage_ns = 0, miss_layers = 0, miss_stage_ns = 0, hx_ns = 0;
        /* The same split for one-token passes (decode) alone. */
        uint64_t d_reads = 0, d_sync_ns = 0, d_stage_ns = 0, d_miss_layers = 0, d_miss_stage_ns = 0;
        /* One-token passes with the CPU path, from the counts to the CPU's work submitted: in all
         * (decide), and its parts: the read windows' leases returned, hx_choose, hx_submit, and
         * (..._PREFETCH=pieces, rank 0) the predictor's post after it. */
        uint64_t d_decide_ns = 0, d_retire_ns = 0, d_choose_ns = 0, d_submit_ns = 0, d_post_ns = 0;
        /* ..._MISS_POST: the misses' reads handed to the kept readers, between hx_choose and
         * hx_submit (hx_post_misses; neither part of choose nor the predictor's post above). */
        uint64_t d_miss_post_ns = 0;
    };
    const EpStats& ep_stats() const { return ep_stats_; }
    /* RADIANCE_EXPERIMENTAL_DS41_LCP_PROFILE=1 (an instrument; nothing it measures changes): the routed
     * layers' host-side times by the pass's height -- bucket 0 one token, 1 two to 16, 2 17 to 256, 3 more.
     * layers: routed layer-passes; rows: their routed rows (the pass's rows from the layer's cut);
     * cpu_rows: rows handed to the pool; sync: waiting for the card at the counts (the router wait: the
     * card's work up to this layer's router not hidden behind the host, e.g. the layer before's experts);
     * miss_wait: at the gather for misses still being read; gather: at the gather for the pool's rows;
     * miss_units / miss_read: the misses read and the read call's time (on the worker, overlapping);
     * waves: large-chunk read waves; span: from one routed layer's counts to the next layer's in the same
     * pass (the layer-pass's wall time; a pass's first layer has none). */
    struct PassProfile {
        uint64_t layers = 0, rows = 0, cpu_rows = 0, sync_ns = 0, miss_wait_ns = 0, gather_ns = 0;
        uint64_t miss_units = 0, miss_read_ns = 0, waves = 0, spans = 0, span_ns = 0;
    };
    static constexpr int kProfBuckets = 4;
    bool profile_on() const { return prof_on_; }
    const PassProfile& profile(int b) const { return prof_[b]; }
    /* RADIANCE_EXPERIMENTAL_SCHED_TBO: routed layers' slices B begun (a split pass, stager.h tbo_). */
    uint64_t tbo_slices() const { return tbo_slices_; }
    /* Routed experts the CPU computed (HostExpertPool) instead of a copy to the card. */
    struct HostStats {
        uint64_t layers = 0, units = 0, rows = 0;
        uint64_t miss_units = 0, miss_rows = 0, miss_read_ns = 0;   /* misses computed from planes */
        /* Large-chunk prefill (hx_stream_misses): layers whose misses were streamed, their read
         * waves (past ..._LCP_PLANES units a wave), the units handed to the pool as each landed and
         * the ones handed over after the call (a leg read again), and the issuing thread's waits
         * for an earlier wave's jobs before reusing its planes. */
        uint64_t lcp_layers = 0, lcp_waves = 0, lcp_landed = 0, lcp_late = 0, lcp_wave_wait_ns = 0;
        /* Large-chunk prefill's admission (RadBatch::lcp_admit_row, ..._LCP_ADMIT), over the chunks
         * of more than 16 tokens: cached units leased with an LRU touch (admitted) and without one
         * (read-only); missed units relaid into the cache, computed and let go (read-only), and
         * admitted ones not relaid because ..._LCP_INSERTQ units were already queued (dropped). */
        uint64_t lcp_hit_admit = 0, lcp_hit_ro = 0, lcp_miss_admit = 0, lcp_miss_ro = 0, lcp_miss_dropped = 0;
        /* Pool jobs that run the many-row kernel (HostExpertPool::many_from: four or more rows of
         * one expert with ..._HOST_EXPERTS_MAX_ROWS above 4), and their rows. */
        uint64_t many_jobs = 0, many_rows = 0;
        /* The next-layer predictor (router_predictor.h) against one-token layers: predictions on
         * time, the layers' misses, the misses predicted, the file-tier experts predicted, and of
         * those the ones the layer selected. */
        uint64_t pred_layers = 0, pred_ready = 0, pred_misses = 0, pred_hit = 0;
        uint64_t pred_file = 0, pred_selected = 0;
        uint64_t pred_wasted = 0;   /* predicted file-tier experts not selected and not cached */
        /* By the prediction's rank (best first): uncached file-tier experts predicted, and of them
         * the ones the layer selected -- a read-ahead's precision at each rank. */
        uint64_t pred_pos_unc[16] = {}, pred_pos_hit[16] = {};
        /* The same from two layers back (RouterPredictor::predicted2, shadow mode): layers with
         * the prediction on time, their misses, the misses predicted, and by rank the uncached
         * predictions and the ones the layer selected. */
        uint64_t pred2_ready = 0, pred2_misses = 0, pred2_hit = 0;
        uint64_t pred2_pos_unc[16] = {}, pred2_pos_hit[16] = {};
        /* Read-ahead (..._PREFETCH=on): units read ahead, of them used by a miss, the misses that
         * found theirs still being read and how long they waited. */
        uint64_t pf_read = 0, pf_used = 0, pf_waited = 0, pf_wait_ns = 0;
        /* ...and the entries their layer took before their read began, and the read-ahead's time
         * holding back for demand reads. */
        uint64_t pf_taken = 0, pf_yield_ns = 0;
        /* Read-ahead in pieces (..._PREFETCH=pieces, readahead.h), the layers of passes that read
         * ahead (one-token passes; with ..._PREFETCH_MULTIROW also decode passes of 2..N rows): units
         * planned (predicted, uncached, eligible); planned for a layer this rank had decided
         * already, or past the cap of live units. At their layer, the selected misses: whole
         * (every piece read), finished (partly read, the rest read urgently while the layer
         * waited), read by the layer as any miss (no piece started, or fewer bytes than the plane
         * mirror's share), and re-read whole after a failed piece. Dropped: not selected, selected
         * but a RAM-cache hit, or stale (their layer passed without a decision). The layer's
         * misses, and of them the ones with a planned unit. Bytes the readers had read of the
         * units used and of the ones dropped; the layers' waits for the rest of finished units. */
        uint64_t ra_planned = 0, ra_late = 0, ra_capped = 0;
        uint64_t ra_whole = 0, ra_finished = 0, ra_fresh = 0, ra_refetched = 0;
        uint64_t ra_unselected = 0, ra_cached = 0, ra_stale = 0;
        uint64_t ra_misses = 0, ra_miss_planned = 0;
        uint64_t ra_used_bytes = 0, ra_dropped_bytes = 0, ra_wait_ns = 0;
        /* The layers that finished partly read units (either way): their time from the layer's
         * choice (hx_choose took the units: urgent pieces start there, a rest's read a little
         * later) until every such unit was whole; and of them the layers with one such unit and
         * nothing else to read, which is that unit's own wait: the per-unit wait to compare with
         * and without ..._PREFETCH_URGENT. */
        uint64_t ra_fin_layers = 0, ra_fin_ns = 0, ra_alone = 0, ra_alone_ns = 0;
        /* Of the alone layers, those whose unit had at least the plane mirror's decode share of
         * pieces started when the layer took it (decode_mirror_bytes(), 0 without a mirror): every
         * finished unit without ..._PREFETCH_URGENT=2, which also finishes the barely started units
         * the others read afresh (a rest of ~16-18 MB, a whole demand read's wait). The per-unit
         * wait on the same set of units in every arm. */
        uint64_t ra_alone_b = 0, ra_alone_b_ns = 0;
        /* ..._PREFETCH_URGENT: finished units whose rest their layer's own call read on the kept
         * readers, the rest's bytes, and of them the plane mirror's (ra_wait_ns is then the wait,
         * after that call, for their pieces in flight); the units whose rest the mirror read part of. */
        uint64_t ra_rest = 0, ra_rest_bytes = 0, ra_rest_mirror_bytes = 0, ra_rest_mirror = 0;
        /* ..._SIDE_STORE (hx_side_store.h): misses handed the planes the store holds (no drive
         * read), and of them in one-token passes. */
        uint64_t side_hits = 0, side_d_hits = 0;
        /* The same by the pass's rows (ra_rows[r], r = 1..kRaRows; [0] unused): one-token passes,
         * and with RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MULTIROW the decode passes of 2..N rows.
         * Planned and skipped (too late, or past the live cap) by the rows of the post that
         * predicted them; the rest by the rows of the pass that decided them. used = whole +
         * finished; dropped = not selected + cache hits + stale. */
        struct RaRows {
            uint64_t planned = 0, skipped = 0;
            uint64_t whole = 0, finished = 0, fresh = 0, refetched = 0, dropped = 0;
            uint64_t misses = 0, miss_planned = 0;
            uint64_t used_bytes = 0, dropped_bytes = 0, wait_ns = 0;
        };
        static constexpr int kRaRows = 16;
        RaRows ra_rows[kRaRows + 1];
        static int ra_slot(int64_t rows) { return rows < 1 ? 1 : rows > kRaRows ? kRaRows : (int)rows; }
        uint64_t linked = 0;          /* cached experts the schedule left on the copy link */
        uint64_t missed = 0;          /* routed experts the RAM cache did not hold */
        uint64_t wait_ns = 0;         /* the issuing thread waiting at gathers for the CPU */
        /* ...in one-token passes: the gathers and their wait, [0] layers without a miss, [1] with */
        uint64_t d_waits[2] = {}, d_wait_ns[2] = {};
        uint64_t async_wait_ns = 0;   /* ...for a miss read still on hx_mw_ (async), before that */
        uint64_t verified = 0, differ = 0;   /* ..._POLICY=verify: rows compared with the card's */
        uint64_t moved = 0;           /* a gate_up's operands at new addresses (kept the newest) */
        uint64_t probed = 0;          /* selected units hx_choose borrowed to test for the cache:
                                       * the mapped cache's own hit/miss counts include these */
        /* ..._EARLY_ISSUE: layers whose card ops went ahead of their counts sync (their CPU share
         * chosen at the gather), the issuing thread's wait there for the counts, the layers whose
         * counts had landed before the thread got there (it was behind the card, so issuing the
         * card ops first delayed the CPU's start), and the layers whose rows were fetched again
         * because their gate_up read them at new addresses. */
        uint64_t early = 0, early_wait_ns = 0, early_ready = 0, early_refetch = 0;
        /* ..._FETCH_LIST: round trips copied in one ranged dispatch, range lists (re)written, and
         * round trips left as separate copies (addresses the copy kernel cannot take). */
        uint64_t fl_copies = 0, fl_armed = 0, fl_plain = 0;
        /* ..._HOST_EXPERTS_PREWAKE (hx_wake_*): one-token layers whose router event this thread
         * waited for before the pool's pre-wake, and those whose counts were in by then (no
         * pre-wake); from the router event to the counts as this thread saw them, and its time in
         * the pre-wake calls (the pool's own line counts what they woke). Miss reads (one-token)
         * with a pre-wake armed: those whose reads were still out at its time, so it fired, how long
         * after it the last read landed (the lead it gave: want ~the workers' wake-up, 30-60 us),
         * and those whose reads landed first (`early`, by how much). wk_rt_late: of the layers that
         * called the pre-wake, those whose counts were in when the call returned (it held their
         * decision back by up to its own time). */
        uint64_t wk_rt = 0, wk_rt_ready = 0, wk_rt_gap_ns = 0, wk_rt_call_ns = 0, wk_rt_late = 0;
        uint64_t wk_rd_armed = 0, wk_rd_fired = 0, wk_rd_lead_ns = 0, wk_rd_early_ns = 0;
        /* One-token passes' miss reads (Mover::PlaneReadTiming), [0] those of one missed unit and
         * [1] of more, [2] those with a read-ahead unit's rest (..._PREFETCH_URGENT, whatever else
         * they read): calls; posted at the choice (..._MISS_POST); with the mirror's tail in
         * pieces (..._MIRROR_SPLIT). Of the calls with a mirror leg and a container leg: those the
         * mirror landed last in, and how far the last leg trailed the other, summed by which one
         * it was. Then sums from the dispatch: the mirror leg (its copy included), its longest
         * copy and all its copies, the container leg; the first read, the first container read
         * and the last read begun; the caller back after every read landed (a done_ wake when a
         * reader finished last); a posted call's caller coming to wait; the caller back. */
        struct MissReads {
            uint64_t calls = 0, posted = 0, split = 0;
            uint64_t mirror_calls = 0, mirror_last = 0, mirror_by_ns = 0, container_by_ns = 0;
            uint64_t mirror_ns = 0, copy_ns = 0, copy_sum_ns = 0, container_ns = 0;
            uint64_t first_ns = 0, first_container_ns = 0, last_start_ns = 0, back_ns = 0;
            uint64_t join_ns = 0, total_ns = 0;
        };
        MissReads mr[3];
    };
    const HostStats& host_stats() const { return hx_stats_; }
    bool host_experts() const { return hx_pool_ != nullptr; }
    /* RADIANCE_EXPERIMENTAL_DS41_PREFETCH_URGENT as it took effect (0: off, or no read-ahead in
     * pieces to finish). */
    int  prefetch_urgent() const { return hx_ra_rest_; }
    /* The background inserts of missed units (hx_bg_loop), a pass's (not the warm-up's): units
     * relaid into the RAM cache, of them from one-token passes, and their time from the queue's
     * front to published (waits included). With RADIANCE_EXPERIMENTAL_DS41_INSERT_NT or
     * ..._INSERT_SLICE_KIB: the pieces of the codes planes written by non-temporal stores and their
     * bytes, the pieces that first waited for an idle pool and the time asleep, the inserts whose
     * wait budget ran out with the pool still busy, the bytes (and time) copied while the pool had
     * submitted jobs unfinished (HostExpertPool::work), and the pieces begun while it held only
     * reservations (a miss's read: a wait on backlog() would have slept through them). */
    struct InsertStats {
        uint64_t units = 0, d_units = 0, ns = 0, d_ns = 0;
        uint64_t pieces = 0, nt_bytes = 0, waited = 0, wait_ns = 0, capped = 0;
        uint64_t busy_bytes = 0, busy_ns = 0, reserved = 0;
    };
    InsertStats insert_stats() const;
    /* ..._SIDE_STORE: what the store holds now (the insert threads fill it). */
    bool side_store() const { return hx_side_.on(); }
    HxSideStore::Stats side_store_stats() const { return hx_side_.stats(); }
    /* RADIANCE_EXPERIMENTAL_DS41_ROUTE_TRACE (route_trace.h): this rank's part. Passes and layer
     * records written; of the layers, one-token ones; layers with rank 0's router outputs, with
     * its logits, with a RAM-cache snapshot; bias records; layers whose router op this pass did
     * not show (no route then); the one-token layers' picks by what this rank did with them. */
    struct RtStats {
        uint64_t passes = 0, layers = 0, d_layers = 0, routes = 0, logits = 0, masks = 0;
        uint64_t biases = 0, uncaptured = 0;
        uint64_t copy_failed = 0;   /* rank 0's router-output copies that could not be queued */
        uint64_t picks[rtrace::kStatusCount] = {};
    };
    const RtStats& rt_stats() const { return rt_stats_; }
    bool route_trace() const { return rt_ != nullptr; }

    /* Binds the routed layers with file-tier units and the ops that read them, over the mover's
     * two file buffers. Leaves enabled() false when no routed unit is on the file tier. Refuses a
     * plan whose routed layers are read by interleaved ops, since a file-tier unit has no other
     * address to fall back on. */
    /* With `counts`, a pass of at most kRoutedRows tokens reads only the experts it routes to:
     * at a layer's first expert op the stager copies the layer's counts row to the host behind
     * the router (same stream), waits for it, reads the routed units and aliases the rest
     * (Mover::alias_file_units). A counts row that does not add up to the pass's tokens times
     * top_k, a layer with no live row, or a second issue of the layer's first op (a block that
     * slices the pass) falls back to reading the whole layer. */
    /* The optional private EP provider is used only for exact released native-MXFP4 EP4
     * layers with M<=kEpRouteMaxRows (M<=kEpRouteTallRows under RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL,
     * tall_init). A missing current witness reads the full layer
     * directly; an old global report cannot authorize that EP read. Other layers retain CountsRow verbatim. */
    int init(const Program& prog, const Plan& plan, Mover* mover,
             RouteCounts routes = nullptr, void* routes_ud = nullptr,
             CountsRow counts = nullptr, EpCountsRow ep_counts = nullptr);

    bool    enabled() const { return !layers_.empty(); }
    size_t  n_layers() const { return layers_.size(); }
    int64_t buffer_bytes() const { return cap_; }
    bool    routed_reads() const { return counts_host_ != nullptr; }
    /* The largest pass that reads only its routed experts. A decode step or a short prompt
     * routes to a fraction of a layer; a prefill chunk routes to nearly all of it and keeps the
     * whole-layer read with its read-ahead. */
    static constexpr int64_t kRoutedRows = 64;

    bool begin_pass(const RadBatch* b) override;
    int  before_op(rad_op h, RadStream s, bool* resync) override;
    int  operands(rad_op h, RadStream s, const RadTensor* t, int n_t) override;
    int  launched(rad_op h, RadStream s, const RadArgs* a, int (*launch)(const RadArgs*, RadStream)) override;
    int  after_op(rad_op h, RadStream s) override;
    int  end_pass(RadStream s) override;
    /* ---- prefill Phase 1 (card_expert_stager.h): RADIANCE_EXPERIMENTAL_DS41_P1=1 beside LCP ----------
     * p1_on: this rank took it (p1_init); p1_stats_line: kLogStats's line for the shutdown. */
    bool p1_on() const { return p1_ && p1_->on(); }
    std::string p1_stats_line() const;
    /* Whether pass `b` is armed for the card path at the threshold `min_rows` (..._P1_MIN_ROWS): a pure
     * prefill pass (no decode row, no draft or speculation) of at least min_rows and at most
     * `max_rows` (the chunk) tokens. Its routed layers then each take the card path or the CPU path by
     * their own rows (ds41p1::card_layer); any other pass runs as without the switch. */
    static bool p1_pass_arms(const RadBatch& b, int64_t min_rows, int64_t max_rows) {
        return b.phase == RAD_PHASE_PREFILL && b.n_tok_decode == 0 && b.draft_pass == 0 && b.n_spec == 0 &&
               b.n_tok >= ds41p1::kMinArmRows && b.n_tok >= min_rows && b.n_tok <= max_rows;
    }
    /* The lent arena (..._P1_LEND_ARENA, card_expert_stager.h). Asked for, p1_init lends nothing yet;
     * the engine then binds the arena's smallest level and hands p1_lend_arena the arena's top [lo, lo +
     * bytes) above that level's buffers (steps of up to `rows` tokens) and guard, and `full`, whether
     * the step about to be issued runs at the full plan (Ctx::level() 0): the lending starts, the ring's
     * slots (..._P1_LEND) with the arena's, and every pass at the full plan takes them all back. < 0
     * with *why when no slot fits or no unit is left to lend (the engine refuses to start). */
    bool p1_lend_arena_asked() const { return p1_ && p1_arena_asked_; }
    bool draft_bank_lend_conflict() const { return p1_lend_asked_ || p1_arena_asked_ || (p1_ && p1_->lending()); }
    /* Idle P1 is supported; ring/arena lenders require a combined proof and are refused. */
    int quiesce_for_draft_bank();
    void track_draft_bank_residents() { bank_track_ = true; }
    uint64_t draft_bank_selected_units() const { return bank_selected_; }
    int  p1_lend_arena(char* lo, int64_t bytes, int64_t rows, std::function<bool()> full, std::string* why);
    //Must run before Ctx::shutdown and before Mover/bank/buffer owners disappear.
    int shutdown_head_cache();

private:
    friend struct EpCountsHostProbe;
    using Layer = StagedLayer;
    std::vector<Layer>   layers_;      /* the routed layers with file-tier units, in first-op order */
    std::vector<int32_t> first_of_;    /* by op: the layer whose first op it is, or -1 */

    const Plan* plan_ = nullptr;
    Mover*   mover_ = nullptr;
    char*    buf_[2] = { nullptr, nullptr };
    int64_t  cap_ = 0;
    RouteCounts routes_ = nullptr;     /* routing-aware reads; null reads whole layers */
    void*    routes_ud_ = nullptr;
    int32_t* scratch_ = nullptr;        /* pinned, for the trigger's blocking counts copy */
    int64_t  scratch_n_ = 0;
    std::vector<int32_t> skipped_[2];   /* per buffer: units not read, nulled at release */
    /* Per buffer: units the stager read into hot-cache slots. Unlike the
     * staged ones they keep their publications when the buffer is released — the slot is
     * their address from here on, until heat evicts one. */
    std::vector<int32_t> promoted_[2];
    RadEvent done_[2] = { nullptr, nullptr };   /* behind a buffer's copies, on the mover stream */
    RadEvent free_[2] = { nullptr, nullptr };   /* behind the ops that read a buffer's layer */
    bool     free_rec_[2] = { false, false };
    std::vector<int32_t> staged_[2];
    int  start(int32_t li, RadStream s);
    int32_t  holds_[2] = { -1, -1 };             /* the layer a buffer holds, or -1 */
    bool     partial_[2] = { false, false };     /* its routed experts only, the rest aliased */
    bool     pending_[2] = { false, false };     /* `done_` was recorded for its read */
    int32_t  cur_ = -1;                          /* the layer whose ops are being issued */

    /* Routed-only reads (init's `counts`). */
    CountsRow counts_;
    int32_t*  counts_host_ = nullptr;            /* pinned, one row */
    int64_t   counts_cap_ = 0;                   /* entries it holds */
    RadEvent  counts_ev_ = nullptr;
    bool      routed_ = false;                   /* this pass reads routed experts only */
    int64_t   pass_tok_ = 0;
    /* RADIANCE_EXPERIMENTAL_DS41_LCP_PROFILE (PassProfile above). */
    bool prof_on_ = false;
    PassProfile prof_[kProfBuckets];
    std::chrono::steady_clock::time_point prof_last_counts_{};
    int prof_bucket() const { return pass_tok_ <= 1 ? 0 : pass_tok_ <= 16 ? 1 : pass_tok_ <= 256 ? 2 : 3; }
    void prof_counts(int32_t li, uint64_t sync_ns, std::chrono::steady_clock::time_point t_counts);
public:
    /* LCP's chunk C once this rank's stager admitted it (lcp_init, hx_init), else 0. */
    int64_t lcp_rows() const { return lcp_; }
private:
    /* The per-row consumer cut (RadBatch::consumer_row0, core/sched/consumer_skip.h): routed layer li
     * runs on the pass's rows [layer_r0(li), pass_tok_), its router, scatter and expert GEMMs over
     * those rows alone, and the CPU path with them (hx_fetch from the gate_up's row layer_r0, the
     * jobs over layer_rows); 0 and the pass on every pass without it. */
    std::vector<int32_t> pass_row0_;   /* by model layer; empty: no cut by row */
    int64_t layer_r0(int32_t li) const {
        if (tbo_) return tbo::first_row(tbo_tok_, tbo_slice_);
        if (pass_row0_.empty() || li < 0 || (size_t)li >= layers_.size()) return 0;
        const int32_t l = layers_[(size_t)li].layer;
        return l >= 0 && (size_t)l < pass_row0_.size() ? std::clamp<int64_t>(pass_row0_[(size_t)l], 0, pass_tok_) : 0;
    }
    int64_t layer_rows(int32_t li) const { return tbo_ ? pass_tok_ : pass_tok_ - layer_r0(li); }
    /* RADIANCE_EXPERIMENTAL_SCHED_TBO (core/sched/tbo.h): a split pass issues each routed layer as
     * two slices one after the other, micro-batch A's rows [0, Sa) and then B's [Sa, n_tok), each
     * from its gate_up to its gather (the architecture's other card work between them names no
     * file-tier weight). While a slice runs, pass_tok_ is its rows and layer_r0 its first row, so
     * its counts witness, its rows' copies, the CPU's choice and jobs, its misses and the
     * read-ahead take it as a pass of its own rows. A layer's second gate_up starts slice B
     * (tbo_second); entering a layer starts slice A. */
    bool    tbo_ = false;
    int     tbo_slice_ = 0;
    int64_t tbo_tok_ = 0;     /* the step's tokens */
    void tbo_enter(int slice);
    int  tbo_second(int32_t li, RadStream s);
    /* Whether slice A's read-ahead entries it did not select stay for slice B (which reads ahead). */
    bool tbo_keeps_ra() const;
    uint64_t tbo_slices_ = 0;   /* slices B started (the shutdown report) */
    int32_t n_model_layers_ = 0;
    /* The layers whose MoE this pass runs: RadBatch::consumer_full on a pass the consumer skip
     * cut (core/sched/consumer_skip.h), every layer otherwise. Nothing is read, staged, predicted
     * or read ahead for a layer at or past it. */
    int32_t   pass_full_ = INT32_MAX;
    bool      moe_runs(int32_t layer) const { return layer < pass_full_; }
    bool trace_requested_ = false, trace_target_m1_ = false;
    /* The largest pass the EP counts path takes: 16, or with RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL=1
     * the step (--max-num-batched-tokens, at most kEpRouteTallRows), and so the CPU path's: its
     * pinned rows, sorted order and output rows and the fetch lists are sized by it. */
    int64_t ep_rows_ = kEpRouteMaxRows;
    /* RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL: sets ep_rows_, and refuses a step whose passes would not
     * give a 16-token pass's bits: one taller than kEpRouteTallRows, a routed MXFP4 GEMM band off the
     * decode-band kernel (no twin row moe_gemm_ggml_mxfp4_gemv_tall, or a step taller than its
     * bound), or another op on two kernels other than gemm_nt's m16 / m64; says so at startup.
     * With RADIANCE_EXPERIMENTAL_DS41_LCP=<C> it is lcp_init's. */
    int  tall_init(const Program& prog);
    /* RADIANCE_EXPERIMENTAL_DS41_LCP=<C> (large-chunk prefill, ep_route_counts.h): ep_rows_ = C when the
     * bands say chunks of C give 16-token passes' bits (stager.cpp: the routed GEMMs on the gemv rows,
     * the twin reaching C; any other op on one kernel or on a known byte-identical pair), the step is
     * C, and nothing beside it is outside what was proven (EXACT_TALL other than 1, speculation, the
     * fetch lists, the packed VRAM residents); refused otherwise, with the reason. hx_init then
     * requires the CPU to take every routed unit, and says what was admitted (lcp_report). */
    int  lcp_init(const Program& prog, int64_t C);
    void lcp_report() const;
    /* ---- prefill Phase 1 (card_expert_stager.h) -------------------------------------------------
     *
     * Under RADIANCE_EXPERIMENTAL_DS41_P1=1 (p1_state_ 1, read at lcp_init with the threshold
     * p1_min_rows_), lcp_init admits the P1 twin above 16 rows (ds41p1::kTwinRow, refusing LCP's own
     * twin) and the packed residents' gemv_vp at 16 or fewer; hx_init needs the CPU to take every layer
     * pass below the threshold (p1_cpu_rows); p1_init then checks the rest (every routed layer's three
     * expert ops found, the routed units off the card in the container or the host pool, the ring, the
     * tunables) and starts the path. An armed pass (p1_armed_, p1_pass_arms) hands each routed layer
     * of at least the threshold's rows to the path at start_ep_routed (p1_layer), captures the layer's
     * three launches (launched) and runs its waves behind its down projection (after_op). */
    int  p1_state_ = 0;
    int64_t p1_min_rows_ = ds41p1::kDefMinRows;   /* ..._P1_MIN_ROWS as lcp_init read it */
    bool    p1_lend_asked_ = false;               /* ..._P1_LEND or ..._P1_LEND_ARENA > 0 as lcp_init read it */
    bool bank_track_ = false;
    uint64_t bank_selected_ = 0; // real selected file-tier units skipped because bank loan is resident
    /* The rows the CPU path takes under P1: every layer pass below the threshold (ds41p1::cpu_rows). */
    int64_t p1_cpu_rows() const { return ds41p1::cpu_rows(ep_rows_, p1_min_rows_); }
    /* hx_init's ..._MISSES bound: every pass's rows (ep_rows_), but under P1 without the lent ring only the
     * layer passes below the threshold (p1_cpu_rows); with it a pass that does not arm P1 would stage its
     * misses in the file buffers, whose idle slots then hold published residents. The lent arena keeps
     * the same bound (the copy path refuses whenever a slot is lent). */
    int64_t hx_need_rows() const { return lcp_ && p1_state_ == 1 && !p1_lend_asked_ ? p1_cpu_rows() : ep_rows_; }
    std::unique_ptr<p1::CardExpertPath> p1_;
    bool p1_armed_ = false;
    int32_t p1_li_ = -1;                 /* the layer the path holds this pass, or -1 */
    int  p1_units_ = 0;                  /* its units on the ring */
    struct P1Ops { int32_t sw = -1, dn = -1; };
    std::vector<P1Ops> p1_ops_;          /* by routed layer */
    RadEvent p1_end_ = nullptr;          /* behind an armed pass's last reader of the ring */
    /* The lent ring (card_expert_stager.h, RingLender; ..._P1_LEND): behind the last pass that read a
     * ring resident, recorded at its end; an armed pass's first ring copy waits on it. */
    RadEvent p1_lend_ev_ = nullptr;
    bool     p1_lend_rec_ = false;
    /* The lent ring's slots (..._P1_LEND) and the arena's (`arena`, from p1_lend_arena), the hottest
     * file-tier units to the ring first. */
    int  p1_lend_init(const p1::Tunables& t, const std::vector<char*>& arena, std::string* why);
    /* The lent arena (..._P1_LEND_ARENA): asked for (p1_init then waits for p1_lend_arena), p1_init's
     * tunables, whether the step being issued runs at the full plan, and the level its slots sit above.
     * p1_lend_took_: this pass took the lent slots back (it armed the card path, or ran at the full plan
     * with the lent arena), so the grace restarts at its end. */
    bool    p1_arena_asked_ = false;
    p1::Tunables p1_tun_;
    std::function<bool()> p1_arena_full_;
    int64_t p1_arena_rows_ = 0;
    bool    p1_lend_took_ = false;
    static int64_t p1_now_ms() {
        return (int64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    std::vector<int32_t> p1_nonres_, p1_ahead_;
    /* By routed layer: its routed units the plan pinned in the host pool, which the card would read
     * across the link once a row pass; an armed layer copies the selected ones to the ring as it does
     * a RAM-cache unit (p1_layer). p1_pool_sel_: this layer's selected ones, published skipped. */
    std::vector<std::vector<int32_t>> p1_pool_;
    std::vector<int32_t> p1_pool_sel_;
    int  p1_pool_units_ = 0;
    /* A pooled unit's two weights where the card's DMA reads them (its pool slot), block form; false
     * when it has no pool address or is not a gate_up and down pair. */
    bool p1_pool_source(int32_t u, p1::Source* out) const;
    bool p1_order_ = false;              /* this layer's sorted order fetched for the admission */
    int64_t p1_pass_rows_ = 0;
    std::chrono::steady_clock::time_point p1_t0_{};
    uint64_t p1_touched_ = 0, p1_slab_ = 0, p1_order_moved_ = 0, p1_layers_ = 0;
    int  p1_init(const Program& prog);
    /* Whether routed layer li of this pass is the card path's: an armed pass, and the layer's own rows
     * (layer_rows: under the per-row cut a deep layer can run on fewer) at least the threshold and
     * never 16 or fewer (ds41p1::card_layer). Any other layer of an armed pass takes the CPU path for
     * its non-resident experts as a pass of its height does: plain aliases, no mark, no wave. */
    bool p1_layer_on(int32_t li) const { return p1_armed_ && ds41p1::card_layer(layer_rows(li), p1_min_rows_); }
    /* The rows the CPU path's miss bound (..._MISSES, hx_miss_rows_) is held to: the pass's, or in an
     * armed pass the layer's own (a layer the card path does not take, below the threshold, beside
     * ..._MISSES of at least p1_cpu_rows). */
    int64_t hx_miss_need(int32_t li) const { return p1_armed_ ? layer_rows(li) : pass_tok_; }
    uint64_t p1_cpu_layers_ = 0;         /* this armed pass's routed layers on the CPU path (below the threshold) */
    int  p1_layer(int32_t li, RadStream s);
    bool p1_admits(int32_t li, int32_t e) const;
    /* ..._P1_VERIFY=N: after a layer's waves, N of its RAM-cache units' expert-rows recomputed by the
     * CPU pool from the cache's bytes (borrowed again, read-only) and compared with the card's down
     * rows; the pass waits for the readback. Drive units are not sampled (their pinned slot is reused
     * by then). */
    int  p1_verify(int32_t li, RadStream s);
    std::vector<uint8_t> p1_ram_;        /* parallel to p1_nonres_: a RAM-cache hit */
    int32_t*  p1_vidx_ = nullptr;        /* pinned: the sorted order and the offsets */
    uint16_t* p1_vx_ = nullptr;          /* pinned: the sampled token rows */
    uint16_t* p1_vy_ = nullptr;          /* pinned: the card's down rows */
    uint16_t* p1_vh_ = nullptr;          /* the host's */
    RadEvent  p1_vev_ = nullptr;
    int64_t lcp_ = 0;                  /* C once lcp_init admitted it, else 0 */
    int64_t lcp_reach_ = 0;            /* the twin's bound, as its bands state it */
    int     lcp_moe_ = 0, lcp_twin_ = 0, lcp_split_ = 0, lcp_runs_ = 0, lcp_ops_ = 0, lcp_pair_ops_ = 0;
    std::string lcp_pairs_;            /* the multi-kernel ops admitted, by family */
    /* Batch eligibility only. start() additionally requires this exact native EP layer;
     * selected-routed starts always use Normal. Immutable copy passed per mover call. */
    NativeFileCacheReadPolicy pass_cache_policy_ = NativeFileCacheReadPolicy::Normal;
    bool      warned_sum_ = false, told_ = false;
    std::vector<int32_t> want_, skip_;

    bool     armed_ = false;
    bool     first_op_ = false;
    int32_t  pass_kind_ = 0;
    int32_t  last_issued_ = -1;
    std::map<int32_t, int32_t> pass_last_;   /* by draft_pass: the last op such a pass issued */

    bool will_issue(int32_t li) const;
    int  start(int32_t li);
    /* Reads layer `li`'s routed units into its buffer and aliases the rest; *read false when the
     * counts cannot be trusted and nothing was staged (the caller reads the whole layer). */
    int  start_routed(int32_t li, RadStream s, bool* read);
    int  start_ep_routed(int32_t li, RadStream s, bool* read);
    EpCountsRow ep_counts_;
    const Program* ep_program_ = nullptr;
    std::vector<uint8_t> ep_layer_eligible_;
    EpStats ep_stats_;
    /* RADIANCE_EXPERIMENTAL_DS41_ACCESS_TRACE=<dir>: every EP-routed layer's selected file-tier
     * units, one record each in <dir>/rank<r>.bin (see trace_layer), for replaying cache policies
     * offline. */
    FILE*    trace_fp_ = nullptr;
    uint32_t trace_pass_ = 0;
    std::vector<int32_t> trace_units_;
    void trace_layer(int32_t layer, uint32_t miss_weights);

    /* ---- RADIANCE_EXPERIMENTAL_DS41_ROUTE_TRACE=<prefix> (route_trace.h) --------------------
     *
     * Every EP-routed layer's record, built where the CPU path has just chosen (start_ep_routed,
     * hx_early_decide, hx_deferred): the selected experts this rank owns and what it did with
     * them (rt_choice, from hx_choose's slots, right after it, with the RAM cache's residency of
     * the layer's file-tier units on a one-token pass), written once the misses are submitted
     * (rt_layer), where the issuing thread waits for the pool anyway. Rank 0 adds the router's
     * outputs: the layer's router select op (router_scored_topk: logits, correction bias, ids,
     * weights) shows its operands at operands() (rt_capture), and start_ep_routed queues their
     * copies to pinned memory right behind the counts' event (rt_queue), under an event of their
     * own (rt_ev_) that rt_layer waits for: the CPU's choice does not wait for them, and by the
     * time the record is built they have long landed. The router's buffers hold this layer's
     * outputs until the next layer's router, which the issue order puts after this layer's
     * decision. No byte the card or the pool computes changes. */
    struct RtCap {
        uint32_t pass = 0;       /* the pass the operands were seen in (trace_pass_), 0 never */
        int64_t rows = 0;
        const char *logits = nullptr, *bias = nullptr, *ids = nullptr, *w = nullptr;
        int64_t logits_ld = 0, ids_ld = 0, w_ld = 0;   /* row pitches, in elements */
    };
    RouteTrace* rt_ = nullptr;
    int  rt_rank_ = -1;
    std::vector<int32_t> rt_li_of_op_;                  /* by op: whose router select op, or -1 */
    std::vector<RtCap>   rt_cap_;                       /* by routed layer */
    std::vector<std::array<uint8_t, 96>> rt_tier_;      /* by model layer: each local expert's Tier */
    std::vector<std::array<int32_t, 96>> rt_unit_;      /* by routed layer: a local expert's file-tier unit, or -1 */
    std::vector<std::vector<int32_t>> rt_file_units_;   /* by routed layer: its file-tier units */
    int32_t* rt_ids_ = nullptr;        /* pinned, rank 0: [kEpRouteMaxRows, 6] */
    float*   rt_w_ = nullptr;          /* [kEpRouteMaxRows, 6] */
    float*   rt_logits_ = nullptr;     /* [384]: a one-token pass's */
    float*   rt_bias_ = nullptr;       /* [routed layers, 384], each copied once */
    RadEvent rt_ev_ = nullptr;         /* behind rank 0's copies (host-waited) */
    bool     rt_ev_rec_ = false;
    std::vector<uint8_t> rt_bias_state_;                /* by routed layer: 0, 1 queued, 2 recorded */
    int32_t  rt_q_li_ = -1;            /* the layer whose router outputs are queued, or -1 */
    bool     rt_q_logits_ = false;
    int32_t  rt_step_ = 0;
    uint64_t rt_t0_ns_ = 0;
    uint32_t rt_us_ = 0;               /* the choice, from the pass's start */
    bool     rt_decided_ = false;
    std::vector<int32_t> rt_units_;    /* the layer's selected file-tier units, as hx_choose had them */
    std::vector<uint8_t> rt_status_;   /* parallel to rt_units_ (rtrace::Status) */
    uint32_t rt_ram_[3] = {};
    bool     rt_ram_ok_ = false;
    std::vector<uint8_t> rt_hit_, rt_rec_;
    RtStats  rt_stats_;
    int  rt_init(const Program& prog);
    void rt_pass(const RadBatch* b);
    void rt_capture(int32_t li, const RadTensor* t, int n_t);
    int  rt_queue(int32_t li, RadStream s);
    void rt_choice(int32_t li, bool decided);
    void rt_layer(int32_t li);

    /* ---- routed experts computed on the CPU (host_expert_pool.h) ----------------------------
     *
     * In an EP-routed pass, start_ep_routed copies the pass's token rows, sorted order and expert
     * offsets to the host with the counts row (hx_fetch, one wait for all), gives each selected
     * unit the RAM cache holds to the pool or to the copy link, whichever the schedule expects
     * free sooner (hx_choose), and starts the pool's jobs (hx_submit) before the drive reads that
     * block the thread. The pool's units are aliased like unrouted ones, so the card's GEMMs
     * compute rows from stray bytes for them. At the layer's gather (operands) the stager waits
     * for the pool and copies the CPU's rows over the card's before the gather reads them.
     *
     * The rows' addresses are the gate_up's operands, which hx_capture records at every pass's
     * gate_up and checks against what hx_fetch read; a layer gets CPU units from its second pass
     * on. One layer is in flight at a time; its leases are given back once the pool read them.
     *
     * With ..._EARLY_ISSUE (file buffers capped) the wait, the choice, the submission and the
     * misses' reads move from the gate_up to the gather, behind the layer's card ops
     * (hx_early_decide); with ..._FETCH_LIST the counts row and the rows go in one ranged copy
     * (hx_fetch_list). See the members below. */
    enum { kHxGreedy, kHxAll, kHxVerify };
    /* gather: the ops that may read the layer's down rows first -- its moe_gather, or the
     * reduction that folds it in (ar_gather_hc_write, a form a wire), whichever the pass issues.
     * x .. off: the gate_up's operands as last issued. */
    struct HxOps {
        int32_t gu = -1;
        int32_t gather[3] = { -1, -1, -1 };
        int64_t n_expert = 0, top_k = 0;
        const void* x = nullptr;
        int64_t     x_ld = 0;
        const void* sorted = nullptr;
        const void* off = nullptr;
    };
    static constexpr int kHxUnits = 96;
    /* Sorted rows a pass can have: ep_rows_ tokens at top_k 6. */
    int64_t hx_sorted_max() const { return ep_rows_ * 6; }
    int64_t hx_alloc_rows_ = 0;                   /* the ep_rows_ the pinned buffers below hold */
    int64_t hx_need_misses_ = kEpRouteMaxRows;    /* ..._MISSES the cap needs: ep_rows_ (hx_init) */
    HostExpertPool* hx_pool_ = nullptr;
    int  hx_policy_ = kHxGreedy;
    std::vector<HxOps> hx_ops_;                   /* by routed layer; gu < 0: the card only */
    int32_t hx_li_ = -1;                          /* the layer with CPU units, or -1 */
    int  hx_stage_ = 0;                           /* 1: chosen, 2: handed to the pool */
    bool hx_confirmed_ = false;                   /* its gate_up read the rows the pool was given */
    /* A layer never seen at its gate_up has no addresses for hx_fetch: with the CPU taking every
     * unit (policy all, misses on) its CPU work waits for the gate_up's operands, the card having
     * been given aliases for all of them. hx_defer_li_: the layer waiting, or -1. */
    bool    hx_defer_ok_ = false;
    int32_t hx_defer_li_ = -1;
    /* RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS was accepted: the card's copy path has no room. */
    bool    hx_capped_ = false;
    std::unique_ptr<Mover::HostUnit[]> hx_units_;
    int  hx_n_ = 0;                               /* slots used; a released one has n == 0 */
    int64_t hx_reserved_ = 0;                     /* pool backlog reserved, not yet submitted */
    std::vector<int32_t> hx_keep_, hx_slot_;
    std::vector<HostExpertPool::Job> hx_jobs_;
    std::vector<int32_t> hx_rows_;                /* the sorted row each job writes */
    std::vector<int>     hx_order_;               /* hx_submit's units, longest first */
    HostExpertPool::Ticket hx_ticket_;
    uint16_t* hx_x_ = nullptr;                    /* pinned [ep_rows_, 5120] token rows */
    int32_t*  hx_idx_ = nullptr;                  /* pinned sorted_tok [hx_sorted_max()], expert_offset */
    uint16_t* hx_out_ = nullptr;                  /* pinned [hx_sorted_max(), 5120]: the CPU's rows */
    uint16_t* hx_chk_ = nullptr;                  /* pinned [hx_sorted_max(), 5120]: the card's (verify) */
    RadEvent  hx_in_ev_ = nullptr, hx_out_ev_ = nullptr;
    bool      hx_out_rec_ = false;
    int       hx_told_ = 0;
    HostStats hx_stats_;
    int  hx_init(const Program& prog);
    int  hx_init_impl(const Program& prog);
    /* The first pass's CPU work, at its gate_up's operands (start_ep_routed deferred it). */
    int  hx_deferred(int32_t li, RadStream s);
    int  hx_warm();
    int  hx_submit_misses_now(int rows);
    /* Large-chunk prefill's misses (stager.cpp): each unit's jobs built before its read and handed
     * to the pool the moment its last extent lands (Mover::read_unit_planes' `landed`), longest
     * first, in waves of at most hx_lcp_planes_ units. */
    int  hx_stream_misses(int rows);
    int  hx_lcp_planes_ = 48;     /* RADIANCE_EXPERIMENTAL_DS41_LCP_PLANES */
    /* Large-chunk prefill's causal admission: the pass's RadBatch::lcp_admit_row (0: everything), and
     * by miss (parallel to hx_miss_) whether it is admitted; the cap on the insert queue. */
    int32_t lcp_admit_row_ = 0;
    std::vector<uint8_t> hx_miss_admit_;
    int  hx_lcp_insertq_ = 16;    /* RADIANCE_EXPERIMENTAL_DS41_LCP_INSERTQ */
    /* Whether local expert e of routed layer li, whose sorted order hx_idx_ holds, has a row at or
     * past lcp_admit_row_ (always with no admission row). */
    bool hx_admits(int32_t li, int32_t e) const;
    /* The misses idx[0..n) of hx_miss_ to the insert threads with the pass's tokens `rows`, the
     * admitted ones only and not the side store's, at most hx_lcp_insertq_ queued at once in a
     * large chunk (the rest dropped, counted). */
    void hx_queue_inserts(const size_t* idx, size_t n, int rows);
    /* The same over all of hx_miss_ (hx_submit_misses_now's call). */
    void hx_queue_inserts(int rows) {
        std::vector<size_t> all(hx_miss_.size());
        for (size_t i = 0; i < all.size(); ++i) all[i] = i;
        hx_queue_inserts(all.data(), all.size(), rows);
    }
    /* RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_ASYNC_MISSES=N: in passes of N or more tokens a
     * layer's missed planes are read and their jobs submitted on hx_mw_ (a thread a rank) while
     * the issuing thread goes on issuing the layer's card ops; hx_finish and hx_drain wait for it
     * before the pool's ticket. Measured: prefill gains (the card computes a chunk's resident
     * experts during the reads, TTFT -23%); decode does not (-1%), hence N=2. */
    void hx_mw_loop();
    int  hx_mw_wait();
    bool hx_async_ = false;
    int  hx_async_rows_ = 2;
    std::thread hx_mw_;
    std::mutex hx_mw_mu_;
    std::condition_variable hx_mw_cv_;
    bool hx_mw_req_ = false, hx_mw_stop_ = false;
    int  hx_mw_rc_ = 0, hx_mw_rows_ = 0;
    int  hx_fetch(int32_t li, RadStream s);
    void hx_choose(int32_t li);
    int  hx_submit();
    int  hx_jobs_for(int e, const HostExpertPool::Job& proto);
    int  hx_capture(int32_t li, RadStream s, const RadTensor* t, int n_t);
    int  hx_finish(RadStream s, const RadTensor* t, int n_t);
    void hx_drain();
    /* RADIANCE_EXPERIMENTAL_DS41_EARLY_ISSUE=1 (with the file buffers capped). Without it a
     * routed layer's gate_up waits in start_ep_routed for the counts, the CPU's choice and, in a
     * decode pass, the misses' drive reads, and only then do the layer's card ops (gate_up,
     * swiglu, down, the shared expert, the ffn mix and the next layer's attention prefetch) reach
     * the queue: the card idles from the counts' copy until the host is done. Capped, every
     * file-tier unit is the CPU's, so the card's aliases do not depend on the routing:
     * start_ep_routed queues the counts' and rows' copies, publishes aliases for all the layer's
     * file-tier units and returns at once, and the card computes the resident experts and the
     * shared expert while the host waits. The counts are waited for, the CPU's share chosen and
     * submitted and the misses read at the layer's first gather (hx_early_decide), which then waits
     * for the CPU rows as before (hx_finish). The bytes the card and the pool compute are the same,
     * and the CPU's rows still overwrite the card's stray rows before the gather reads them.
     * hx_early_li_: the layer waiting for its gather, or -1. */
    bool    hx_early_ = false;
    int32_t hx_early_li_ = -1;
    bool    hx_early_seen_ = false;     /* its gate_up read the rows its copies fetched */
    EpRouteCounts hx_early_v_;          /* its counts row's witness, checked at the decision */
    int  hx_early_decide();
    /* RADIANCE_EXPERIMENTAL_DS41_PREFETCH_POST_LATE=1 (with ..._EARLY_ISSUE and ..._PREFETCH=pieces):
     * hx_early_decide posts rank 0's router input to the predictor once the layer's CPU work is
     * submitted, as start_ep_routed does, instead of before hx_choose, where the post (its copy
     * of the input and the predictor's wake, 4-6 us) delays the pool's start (rank 0's choose
     * 8.1-8.9 us against 3.8-4.4 on ranks 1-3 in every Ei arm). The prediction starts ~25 us later
     * (the submit) on a lead of ~1.65 ms. Timed apart (d_post_ns), as start_ep_routed's post. */
    bool    hx_post_late_ = false;
    /* RADIANCE_EXPERIMENTAL_DS41_FETCH_LIST=1: the counts row and the pool's rows in one ranged
     * copy (hx_fetch_list.h) instead of four dispatches. The copy kernel reads its range list when
     * it runs, so each list lives on the card, one a routed layer and pass size (the rows' count
     * sets the ranges), written in stream order from a pinned staging slot when the addresses it
     * was armed for change: the counts row, or the gate_up operands hx_capture records (an
     * operand move re-arms it). An earlier copy reading the old list is ahead of that write on the
     * same stream; the staging slot is rewritten only after hx_fl_ev_ says its last write left. */
    struct HxFl {
        const void* row = nullptr;
        const void* x = nullptr;
        int64_t     x_ld = 0;
        const void* sorted = nullptr;
        const void* off = nullptr;
        RadStream   s = nullptr;
        int         n = 0;                /* ranges armed; 0: not armed */
    };
    /* A list's ranges at the largest pass, and its room in whole 16-byte copies (pairs of ranges):
     * hx_fl_slot(ep_rows_). The stack holds the most any limit needs. */
    static constexpr int kHxFlRangesMax = hx_fetch_n_ranges(kEpRouteTallRows);
    static constexpr int hx_fl_slot(int64_t rows) { return (hx_fetch_n_ranges(rows) + 1) & ~1; }
    int  hx_fl_slot_ = hx_fl_slot(kEpRouteMaxRows);
    bool hx_fl_ = false;
    std::vector<HxFl> hx_fl_key_;         /* by routed layer and pass rows: what its list holds */
    size_t hx_fl_cap_ = 0;                /* lists allocated */
    RadCopyRange* hx_fl_dev_ = nullptr;   /* the lists, on the card */
    RadCopyRange* hx_fl_pin_ = nullptr;   /* their pinned staging */
    RadEvent hx_fl_ev_ = nullptr;         /* behind the last list write */
    bool hx_fl_rec_ = false;
    /* 1: the counts row and rows are queued as one copy; 0: the caller makes the separate copies. */
    int  hx_fetch_list(int32_t li, const int32_t* row, RadStream s);
    int  hx_init_fetch();
    /* Misses on the CPU (..._MISSES=N: passes of at most N tokens): a missed unit's planes are read
     * from the container straight into reused host buffers, computed as urgent jobs, and relaid
     * into the RAM cache by a background thread so later passes hit. */
    int  hx_miss_rows_ = 0;
    /* By routed layer: each local expert's file-tier unit, or -1 (on the card or pinned). */
    std::vector<std::array<int32_t, 96>> hx_unit_of_;
    class RouterPredictor* hx_pred_ = nullptr;
    /* Read-ahead: a thread a rank reads the predicted uncached units of the layer hx_pf_ahead_
     * layers on (hx_pf_loop) for hx_choose to take instead of reading them. One batch a target
     * layer, two kept (by the layer's parity) so a read for L+2 leaves L+1's alone. An entry is
     * queued (0), being read (1), read (2), or taken by its layer before its read began (3): the
     * layer then reads it as any miss. The thread starts no read while a demand read of any rank
     * is in flight (PieceReader::process()'s count, up to 2 ms). hx_miss_ready_ is parallel to hx_miss_: the
     * planes are in already (1: read ahead; 2: the side store's, hx_side_store.h). */
    struct PfEntry { int32_t unit = -1; std::shared_ptr<Mover::HostPlanes> p; int state = 0; bool ok = false; };
    struct PfBatch { uint32_t pass = 0; int32_t li = -1; uint64_t seq = 0; std::vector<PfEntry> entries; };
    std::thread hx_pf_;
    std::mutex hx_pf_mu_;
    std::condition_variable hx_pf_cv_;
    int      hx_pf_ahead_ = 1;          /* ..._PREFETCH_AHEAD: 1 or 2 layers */
    uint64_t hx_pf_seq_ = 0;
    PfBatch  hx_pf_batch_[2];
    std::deque<uint64_t> hx_pf_queue_;  /* batches to read, by seq, oldest first */
    bool hx_pf_stop_ = false;
    std::vector<uint8_t> hx_miss_ready_;
    void hx_pf_loop();
    void hx_pf_target(int32_t li);
    /* ---- read-ahead in pieces (RADIANCE_EXPERIMENTAL_DS41_PREFETCH=pieces, readahead.h) -----
     *
     * The predictor's thread plans this rank's predicted uncached units of layer L+2 as soon as
     * layer L's prediction is ready (hx_ra_plan, through the process's list of stagers) and queues
     * them on the process's PieceReader. hx_choose decides each of a layer's units once its
     * routing is known (hx_ra_take; the rest are cancelled), and hx_submit_misses_now waits for
     * the urgent ones inside its demand read. ..._PREFETCH_QD readers (4) read pieces of
     * ..._PREFETCH_PIECE_KB (512); at most ..._PREFETCH_LIVE units a rank are live (8). */
    struct RaEntry {
        std::shared_ptr<ReadAheadUnit>     u;
        std::shared_ptr<Mover::HostPlanes> p;
        uint32_t pass = 0;
        int32_t  li = -1, unit = -1;
        int32_t  rows = 1;                   /* the rows of the post that predicted it */
    };
    PieceReader* hx_ra_ = nullptr;           /* set in pieces mode once the readers run */
    std::mutex   hx_ra_mu_;                  /* hx_ra_live_, and the ra_ counters the planner bumps */
    std::vector<RaEntry> hx_ra_live_;        /* planned, not yet decided by their layer */
    std::vector<RaEntry> hx_ra_here_;        /* the deciding layer's, taken out of hx_ra_live_ */
    std::vector<int32_t> hx_ra_li_;          /* by model layer: its routed layer index, or -1 */
    std::vector<uint8_t> hx_ra_ok_;          /* by plan unit: read_unit_planes reads it (at init) */
    /* The pass and routed layer this rank last decided, (pass << 32) | (li + 1): a prediction for
     * a layer at or before it is too late to read ahead. */
    std::atomic<uint64_t> hx_ra_at_{ 0 };
    int     hx_ra_cap_ = 12;
    /* ..._PREFETCH_FINISH (1): a partly read unit its layer selects is finished by the readers
     * (urgent); 0 cancels it and the layer reads it afresh, as a unit barely started always is. */
    bool    hx_ra_finish_ = true;
    /* RADIANCE_EXPERIMENTAL_DS41_PREFETCH_URGENT (0): 1 or 2 finish a partly read unit with large
     * reads instead of urgent pieces: hx_ra_take stops it (no further piece starts), and its
     * layer's own call -- the one that reads the layer's other misses, on the mover's kept
     * readers -- reads the rest of each extent past the pieces started (hx_unread_misses, PlanesIn),
     * while the pieces in flight land; hx_ra_settle then waits for those. The same bytes at the
     * same offsets. 1: the rest from the container alone; 2: the plane mirror takes its decode
     * share of the rest beside it (Mover::rest_mirror_tail), so a barely started unit is
     * finished too, never read afresh. Needs ..._PREFETCH_FINISH (on by default). */
    int     hx_ra_rest_ = 0;
    int     hx_ra_handed_ = 0;               /* whole units handed to this layer: no drive read at
                                              * the layer; the epoch is checked before their jobs */
    uint64_t hx_ra_chose_ns_ = 0;            /* when hx_choose took this layer's units (steady ns):
                                              * where the per-unit wait (ra_fin_ns) starts */
    int     hx_ra_fin_b_ = 0;                /* units finished for this layer that every arm finishes
                                              * (HostStats::ra_alone_b) */
    /* Parallel to hx_miss_: the unit the readers are finishing for this layer (with
     * ..._PREFETCH_URGENT: stopped, its rest the layer's call's), or null. */
    std::vector<std::shared_ptr<ReadAheadUnit>> hx_miss_ra_;
    /* ---- RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MULTIROW=N (2..16; needs the read-ahead in pieces):
     * decode passes of 2..N rows read ahead too -- a DSpark verify pass (the last token and its
     * drafts), or one token from each of several sequences. Rank 0 posts every row's router input
     * at the one-token pass's points (start_ep_routed after the CPU's submission, hx_early_decide
     * after the counts); the predictor reads layer L+2's router once for all the rows and hands
     * every rank the union of the rows' top k (router_predictor.h); hx_ra_plan lays it out under
     * the same live cap, and hx_choose decides a layer's units as in a one-token pass: the
     * readers' yields to demand reads, the finishing rule (with the plane mirror's share for the
     * pass's rows) and the epoch check are the same. A one-token pass goes exactly as without the
     * switch; prefill chunks and the drafter's own passes never read ahead. Reads only: the bytes
     * handed to the CPU are the container's either way. */
    int  hx_ra_rows_ = 1;                    /* the largest pass that reads ahead (1: one-token only) */
    bool hx_ra_pass_ = false;                /* this pass reads ahead (ra_pass_reads_ahead) */
    int  hx_ra_rows_init();
    /* hx_ra_take's finishing rule: the bytes of a missed unit this pass's demand read takes from
     * the plane mirror (a one-token pass: the decode share, as always). */
    int64_t hx_ra_mirror_bytes() const;
    static void hx_ra_ready(void* ctx, uint32_t pass, int layer, const int32_t* experts, int n,
                            int rows);
    static int  hx_ra_read(void* ctx, const PlaneExtent& piece);
    /* Layer `layer`'s predicted global experts for pass `pass` (n of them, best first) from a post
     * of `rows` rows: this rank's uncached ones laid out and queued, up to the live cap. */
    void hx_ra_plan(uint32_t pass, int layer, const int32_t* experts, int n, int rows = 1);
    /* At layer li's routing: its live units into hx_ra_here_, stale ones dropped. */
    void hx_ra_begin(int32_t li);
    /* Layer li's selected miss u: its planes when read ahead (whole: *ready; partly read: *urgent,
     * the readers finish it, or with ..._PREFETCH_URGENT it is stopped and the layer's call reads
     * its rest), or null when the layer reads it as any miss. */
    std::shared_ptr<Mover::HostPlanes> hx_ra_take(int32_t u, bool* ready,
                                                  std::shared_ptr<ReadAheadUnit>* urgent);
    /* Drops a unit the layer does not read (hit: a RAM-cache hit, else not selected). */
    void hx_ra_drop(RaEntry& e, bool hit);
    /* Cancels this rank's live units of passes before `pass`, and of `pass` up to routed layer li. */
    void hx_ra_drop_stale(uint32_t pass, int32_t li);
    /* hx_submit_misses_now's part: waits for the urgent units (with ..._PREFETCH_URGENT, for the
     * pieces in flight of the stopped ones, whose rest the layer's call has read), re-reads failed
     * ones whole, checks the container's epoch. False on a failed read. */
    bool hx_ra_settle(int rows);
    std::vector<std::shared_ptr<Mover::HostPlanes>> hx_miss_;     /* this layer's */
    std::vector<std::unique_ptr<Mover::HostPlanes>> hx_planes_free_;
    std::mutex hx_planes_mu_;
    std::shared_ptr<Mover::HostPlanes> hx_planes_get(int32_t unit);
    /* RADIANCE_EXPERIMENTAL_DS41_SIDE_STORE=N (hx_side_store.h): the planes of up to N units the
     * packed RAM cache refused, kept by the insert threads (hx_bg_loop) and handed to their later
     * misses as read (hx_choose: hx_miss_ready_ 2); hx_side_handed_ such misses this layer, which
     * read nothing (the epoch is checked before their jobs, as for units read ahead whole). Its
     * planes go back to hx_planes_free_ when it lets them go (the destructor, before the list). */
    HxSideStore hx_side_;
    int hx_side_handed_ = 0;
    /* hx_init_impl, before the insert threads see a refusal: the switch, refused unless the packed
     * cache and misses on the CPU are on. */
    int hx_side_init();
    std::shared_ptr<Mover::HostPlanes> hx_planes_backed(int32_t unit);
    int  hx_submit_misses();
    /* The misses hx_submit_misses_now reads, into *out, with what each has in already into *in:
     * the layer's not read ahead whole and not being finished by the piece readers (nothing in),
     * and with ..._PREFETCH_URGENT its stopped units that have a rest (their pieces started in).
     * True when the layer finishes partly read units (urgent: hx_ra_settle waits for them, and
     * may give their slots fresh planes). *alone: one such unit and nothing else to read. */
    bool hx_unread_misses(std::vector<Mover::HostPlanes*>* out, std::vector<Mover::PlanesIn>* in,
                          bool* alone = nullptr) const;
    /* RADIANCE_EXPERIMENTAL_DS41_MISS_POST=1: a one-token pass's missed planes start on the
     * mover's kept readers as soon as hx_choose knows them (hx_post_misses, before hx_submit), and
     * hx_submit_misses_now waits for them before anything else reads on those readers
     * (hx_ra_settle); hx_join_miss_post on every other way out (HxMissPostJoin in start_ep_routed
     * and hx_early_decide, hx_drain). The post takes hx_unread_misses' set -- never a unit the
     * piece readers are finishing, whose slot hx_ra_settle may refill, but with
     * ..._PREFETCH_URGENT a stopped unit's rest: no reader starts a piece of it any more, its
     * pieces in flight land before its rest's offsets, and hx_ra_settle, which may refill its
     * slot, runs only after the post's finish -- and is a demand read from the post on
     * (PieceReader::demand_enter; left after the finish). hx_miss_posted_: a posted read awaits its
     * finish; only the issuing thread. (Not the predictor's post: hx_pred_->post, d_post_ns.) */
    bool hx_miss_post_on_ = false;
    bool hx_miss_posted_ = false;
    std::vector<Mover::HostPlanes*> hx_miss_post_ptrs_;
    std::vector<Mover::PlanesIn>    hx_miss_post_in_;
    void hx_post_misses();
    void hx_join_miss_post();
    struct HxMissPostJoin {
        FileStager* s;
        ~HxMissPostJoin() { s->hx_join_miss_post(); }
    };
    void hx_note_reads(const Mover::PlaneReadTiming& t);
    /* ---- RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_PREWAKE=N (host_expert_pool.h, prewake) --------
     *
     * The pool's workers sleep between a layer's bursts, and a submit then pays their wake-up: the
     * waker's broadcast, each worker's C-state exit (C2: 30 us) and, through the condition
     * variable, the lock's convoy. Two points of a one-token layer come shortly before its CPU
     * work, and the pool's pre-wake (N workers, a short window) runs there:
     *  - the card's progress: an event (card-local, host-waited) recorded right before the
     *    layer's router GEMV, the gemm reading blk.L.glm_router.weight (hx_wake_init finds it;
     *    hx_wake_op_). At the decision (hx_early_decide, or start_ep_routed without early issue)
     *    the issuing thread waits for it first, pre-wakes, then waits for the counts, which follow
     *    the router by its GEMV, the select, the scatter and the copies (hx_wake_route). Every rank
     *    records its own; the pool takes the first call of the layer (prewake's window).
     *  - a miss read: while the issuing thread, done with its own share (the mirror's tail, which
     *    lands first in ~78% of calls), waits for the readers, at ..._PREWAKE_LEAD_US before the
     *    read's predicted landing: the median landing, from the dispatch, of the last kWakeRing
     *    reads of one unit (or of more; hx_wake_at), through Mover's wait hook (hx_wake_miss). Not
     *    in a layer whose read-ahead units are being finished (hx_ra_settle waits after the read),
     *    and nothing when the reads land first.
     * Off, nothing here runs: no event, no hook. The bytes the card and the pool compute are the
     * same either way; only when workers wake changes. */
    static constexpr int kWakeRing = 32;
    bool     hx_wake_ = false;
    RadEvent hx_wake_ev_ = nullptr;
    std::vector<int32_t> hx_wake_op_;   /* by op: the routed layer whose router GEMV it is, or -1 */
    int32_t  hx_wake_li_ = -1;          /* the layer whose router event this pass recorded last */
    int64_t  hx_wake_ring_[2][kWakeRing] = {};   /* miss reads' landings (ns from dispatch): one unit, more */
    uint64_t hx_wake_seen_[2] = {};
    int64_t  hx_wake_fired_ns_ = 0;     /* when this read's pre-wake ran (steady ns), 0 none */
    int  hx_wake_init(const Program& prog);
    int  hx_wake_route(int32_t li, std::chrono::steady_clock::time_point* woke);
    int64_t hx_wake_at(size_t units) const;
    static void hx_wake_miss(void* self);
    std::vector<std::thread> hx_bg_;   /* ..._INSERT_THREADS of them (4) */
    std::mutex hx_bg_mu_;
    std::condition_variable hx_bg_cv_;
    /* A unit to relay and the tokens of the pass that missed it (0: the warm-up's). */
    struct HxInsert {
        std::shared_ptr<Mover::HostPlanes> p;
        int rows = 0;
    };
    std::deque<HxInsert> hx_bg_q_;
    bool hx_bg_stop_ = false;
    int  hx_bg_busy_ = 0;
    void hx_bg_loop();
    /* RADIANCE_EXPERIMENTAL_DS41_INSERT_NT=1: the background inserts write a unit's codes planes
     * into the packed RAM cache by non-temporal stores (ds41hx::PackCopy), so the ~18 MB neither
     * reads each line first nor fills the caches the pool reads through. ..._INSERT_SLICE_KIB=N
     * (implies it): in pieces of N KiB, and a one-token pass's insert sleeps before each piece while
     * the pool has submitted jobs unfinished (HostExpertPool::wait_work; not while it holds only the
     * reservations of misses whose planes are being read, when it has nothing to stream), at most
     * ..._INSERT_WAIT_US (4000) an insert; the slot is published whole, after the last piece's
     * sfence, as before.
     * Read by hx_ins_init; only with the packed cache. */
    bool    hx_ins_nt_ = false;
    size_t  hx_ins_slice_ = 0;
    int64_t hx_ins_wait_ns_ = 0;
    void    hx_ins_init();
    struct InsertCounts {
        std::atomic<uint64_t> units{ 0 }, d_units{ 0 }, ns{ 0 }, d_ns{ 0 };
        std::atomic<uint64_t> pieces{ 0 }, nt_bytes{ 0 }, waited{ 0 }, wait_ns{ 0 }, capped{ 0 };
        std::atomic<uint64_t> busy_bytes{ 0 }, busy_ns{ 0 }, reserved{ 0 };
    };
    InsertCounts hx_ins_;   /* the insert threads add to it */
public:
    /* Waits for the background relayouts of missed units; before the mover drains its cache. */
    void drain_host_inserts();
    /* Initial physical-KV mode has no P1 or head/draft passes. Assert the rank CPU issuer ended
     * its FileStager pass (end_pass already joined CPU expert/miss jobs), then drain RAM-cache
     * insert helpers. Never calls Scheduler or waits for its ownership mutex. */
    int quiesce_for_backing();
private:
    int  finish(int32_t b, RadStream s);
    struct HeadOps;
    DsparkHeadTailOwner head_;
    std::vector<int32_t> head_units_, head_staged_;
    RadEvent head_done_ = nullptr;
    RadStream head_compute_ = nullptr;
    using HeadPass = DsparkHeadTailPass;
    HeadPass head_pass_ = HeadPass::Unknown;
    bool head_first_ = false;
    bool head_closed_ = false;
    int head_status_ = RAD_OK;
    int init_head_cache(const Program& prog, const Plan& plan);
    int head_prepare(RadStream s, bool* resync);
    int head_revoke();
};

}  /* namespace rad */
