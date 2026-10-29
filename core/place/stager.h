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
#include "readahead.h"

#include <array>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
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
    };
    const EpStats& ep_stats() const { return ep_stats_; }
    /* Routed experts the CPU computed (HostExpertPool) instead of a copy to the card. */
    struct HostStats {
        uint64_t layers = 0, units = 0, rows = 0;
        uint64_t miss_units = 0, miss_rows = 0, miss_read_ns = 0;   /* misses computed from planes */
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
        /* Read-ahead in pieces (..._PREFETCH=pieces, readahead.h), one-token layers: units
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
        uint64_t linked = 0;          /* cached experts the schedule left on the copy link */
        uint64_t missed = 0;          /* routed experts the RAM cache did not hold */
        uint64_t wait_ns = 0;         /* the issuing thread waiting at gathers for the CPU */
        uint64_t async_wait_ns = 0;   /* ...for a miss read still on hx_mw_ (async), before that */
        uint64_t verified = 0, differ = 0;   /* ..._POLICY=verify: rows compared with the card's */
        uint64_t moved = 0;           /* a gate_up's operands at new addresses (kept the newest) */
        uint64_t probed = 0;          /* selected units hx_choose borrowed to test for the cache:
                                       * the mapped cache's own hit/miss counts include these */
    };
    const HostStats& host_stats() const { return hx_stats_; }
    bool host_experts() const { return hx_pool_ != nullptr; }

    /* The routed layers of `plan` that hold a unit on the file tier and the ops that read them,
     * over the mover's two file buffers. Leaves it disabled -- enabled() false -- when the plan
     * puts no routed unit there; refuses a plan whose routed layers are read by interleaved ops,
     * because a file-tier unit has no other address to fall back on.
     *
     * With `counts`, a pass of at most kRoutedRows tokens reads only the experts it routes to:
     * at a layer's first expert op the stager copies the layer's counts row to the host behind
     * the router (same stream), waits for it, reads the routed units and aliases the rest
     * (Mover::alias_file_units). A counts row that does not add up to the pass's tokens times
     * top_k, a layer with no live row, or a second issue of the layer's first op (a block that
     * slices the pass) falls back to reading the whole layer. */
    /* The optional private EP provider is used only for exact released native-MXFP4 EP4
     * layers with M<=kEpRouteMaxRows. A missing current witness reads the full layer
     * directly; an old global report cannot authorize that EP read. Other layers retain CountsRow verbatim. */
    int init(const Program& prog, const Plan& plan, Mover* mover, CountsRow counts = nullptr,
              EpCountsRow ep_counts = nullptr);

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
    int  end_pass(RadStream s) override;
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
    RadEvent done_[2] = { nullptr, nullptr };   /* behind a buffer's copies, on the mover stream */
    RadEvent free_[2] = { nullptr, nullptr };   /* behind the ops that read a buffer's layer */
    bool     free_rec_[2] = { false, false };
    std::vector<int32_t> staged_[2];
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
    bool trace_requested_ = false, trace_target_m1_ = false;
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
     * on. One layer is in flight at a time; its leases are given back once the pool read them. */
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
    static constexpr int kHxRows = (int)kEpRouteMaxRows * 6;
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
    HostExpertPool::Ticket hx_ticket_;
    uint16_t* hx_x_ = nullptr;                    /* pinned [kEpRouteMaxRows, 5120] token rows */
    int32_t*  hx_idx_ = nullptr;                  /* pinned sorted_tok [kHxRows], expert_offset */
    uint16_t* hx_out_ = nullptr;                  /* pinned [kHxRows, 5120]: the CPU's rows */
    uint16_t* hx_chk_ = nullptr;                  /* pinned [kHxRows, 5120]: the card's (verify) */
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
    int  hx_capture(int32_t li, const RadTensor* t, int n_t);
    int  hx_finish(RadStream s, const RadTensor* t, int n_t);
    void hx_drain();
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
     * planes are in already (read ahead). */
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
     * ..._PREFETCH_PIECE_KB (512); at most ..._PREFETCH_LIVE units a rank are live (12). */
    struct RaEntry {
        std::shared_ptr<ReadAheadUnit>     u;
        std::shared_ptr<Mover::HostPlanes> p;
        uint32_t pass = 0;
        int32_t  li = -1, unit = -1;
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
    int64_t hx_ra_fresh_below_ = 0;          /* bytes started under which a miss reads afresh */
    int     hx_ra_handed_ = 0;               /* whole units handed to this layer: no drive read at
                                              * the layer; the epoch is checked before their jobs */
    /* Parallel to hx_miss_: the unit the readers are finishing for this layer, or null. */
    std::vector<std::shared_ptr<ReadAheadUnit>> hx_miss_ra_;
    static void hx_ra_ready(void* ctx, uint32_t pass, int layer, const int32_t* experts, int k);
    static int  hx_ra_read(void* ctx, const PlaneExtent& piece);
    void hx_ra_plan(uint32_t pass, int layer, const int32_t* experts, int k);
    /* At layer li's routing: its live units into hx_ra_here_, stale ones dropped. */
    void hx_ra_begin(int32_t li);
    /* Layer li's selected miss u: its planes when read ahead (whole: *ready; partly read: *urgent,
     * the readers finish it), or null when the layer reads it as any miss. */
    std::shared_ptr<Mover::HostPlanes> hx_ra_take(int32_t u, bool* ready,
                                                  std::shared_ptr<ReadAheadUnit>* urgent);
    /* Drops a unit the layer does not read (hit: a RAM-cache hit, else not selected). */
    void hx_ra_drop(RaEntry& e, bool hit);
    /* Cancels this rank's live units of passes before `pass`, and of `pass` up to routed layer li. */
    void hx_ra_drop_stale(uint32_t pass, int32_t li);
    /* hx_submit_misses_now's part: waits for the urgent units, re-reads failed ones whole, checks
     * the container's epoch. False on a failed read. */
    bool hx_ra_settle(int rows);
    std::vector<std::shared_ptr<Mover::HostPlanes>> hx_miss_;     /* this layer's */
    std::vector<std::unique_ptr<Mover::HostPlanes>> hx_planes_free_;
    std::mutex hx_planes_mu_;
    std::shared_ptr<Mover::HostPlanes> hx_planes_get(int32_t unit);
    int  hx_submit_misses();
    std::vector<std::thread> hx_bg_;   /* ..._INSERT_THREADS of them (4) */
    std::mutex hx_bg_mu_;
    std::condition_variable hx_bg_cv_;
    std::deque<std::shared_ptr<Mover::HostPlanes>> hx_bg_q_;
    bool hx_bg_stop_ = false;
    int  hx_bg_busy_ = 0;
    void hx_bg_loop();
public:
    /* Waits for the background relayouts of missed units; before the mover drains its cache. */
    void drain_host_inserts();
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
