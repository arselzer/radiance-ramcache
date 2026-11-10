/* mover.h -- residency and the mover: where every weight unit lives and how it moves.
 *
 * Three storage tiers, one address space of slots:
 *
 *   VRAM slab: fixed-stride slots per weight class. Slots within a class are interchangeable, so
 *   a promotion is an integer swap rather than an allocation, and the slab never fragments.
 *
 *   Pinned host pool, same stride. The card reads a non-resident expert from it zero-copy
 *   inside the GEMM. It is owned here, not read through the container's mmap, because under
 *   memory pressure the kernel's paging makes decode reads several times slower, and
 *   MADV_WILLNEED makes it worse.
 *
 *   File tier: O_DIRECT reads into pinned buffers, with predictable latency and no page-cache
 *   copy. No buffered fallback: a tier whose latency depends on the page cache cannot be budgeted.
 *
 * ---- Transfers never synchronise the host ----------------------------------------------------
 *
 * The mover runs on its own stream and the compute stream waits on its events (rad_event_wait,
 * never rad_event_sync), so placement changes without serialising the step. The lead gate is a
 * host wait on how far the host runs ahead, not on a transfer (MoverConfig::max_lead_dispatches).
 *
 * ---- Shadow slots and quarantine -------------------------------------------------------------
 *
 * A promotion copies into a shadow slot (allocated, holding nothing) and only then publishes the
 * new pointer, so a slot is never overwritten while a kernel may read it. Getting this order
 * wrong does not crash; it produces fluent output that differs from run to run.
 *
 * A freed slot goes into quarantine, because kernels issued in earlier layers still name it. The
 * barrier is an event recorded on the compute stream after the publish; when it signals, every
 * launch that could name the slot has retired.
 */
#pragma once
#include "prefetch_trace.h"
#include "../device/device.h"
#include "../device/directio.h"
#include "../mem/pools.h"
#include "heat.h"
#include "native_file_cache.h"
#include "native_mapped_file_cache.h"
#include <array>
#include "native_read_workers.h"
#include "planner.h"
#include "plane_mirror.h"
#include "readahead.h"
#include "residency_iface.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>
#include <sys/stat.h>

namespace ds41hx { struct PackCopy; }

namespace rad {
class DsparkRadBanks; //borrowed immutable two-bank directory/epoch owner

/* ------------------------------------------------------------------ slot bookkeeping */
/* Free-list bookkeeping for the VRAM slab and the pinned host pool. Pure host state, testable
 * without a card.
 *
 * Slots are numbered in segments laid end to end. Segment 0 is the budgeted range; each later
 * one is memory another pool lends (KV cache blocks, the arena's top). Each segment has its own
 * target, below which slots may be handed out, because lenders reclaim independently.
 *
 * The lowest free slot goes first, so occupied slots pack toward a segment's bottom and its top
 * can be returned; a LIFO would keep handing out high slots. `live` is how far a segment's slots
 * exist and `target` how far they should: a shrink lowers `live` only past unheld slots, so it
 * never evicts.
 *
 * The free list is rebuilt from the held flags whenever a target moves; one carried across a
 * shrink and regrow would lose slots freed above the target. take() and add_free() stay heap
 * operations.
 */
class SlotPool {
public:
    /* Segment 0 starts with `n_live` slots, all held (the plan's resident set) until add_free()
     * releases them. `n_cap` is how far segment 0 may grow into address space the caller has
     * reserved and backs on demand; the default is no growth. */
    void init(int64_t n_live, int64_t stride, int64_t n_cap = -1) {
        stride_ = stride;
        n_ = n_cap > n_live ? n_cap : n_live;
        segs_.assign(1, Seg{ 0, n_, n_live > 0 ? n_live : 0, n_live > 0 ? n_live : 0 });
        free_.clear();
        held_.assign((size_t)(n_ > 0 ? n_ : 0), 1);
        n_usable_ = 0;
        n_held_ = n_live > 0 ? n_live : 0;
    }
    /* A segment of `n_slots` after every existing one, covering nothing and holding nothing until
     * a target reaches into it. Returns its index; slot numbers in it start at seg_lo(index). */
    int add_segment(int64_t n_slots) {
        if (n_slots < 0) n_slots = 0;
        segs_.push_back(Seg{ n_, n_ + n_slots, n_, n_ });
        held_.resize((size_t)(n_ + n_slots), 1);
        n_ += n_slots;
        return (int)segs_.size() - 1;
    }
    int64_t seg_lo(int k) const { return segs_[(size_t)k].lo; }

    void add_free(int64_t slot) {
        if (slot < 0 || slot >= n_) return;
        if (!held_[(size_t)slot]) return;
        held_[(size_t)slot] = 0;
        --n_held_;
        Seg& g = segs_[(size_t)seg_of(slot)];
        if (slot < g.target) push(slot);
        else settle(g);
    }
    bool empty() const { return n_usable_ == 0; }
    /* How many slots exist, over every segment. */
    int64_t size() const {
        int64_t n = 0;
        for (const Seg& g : segs_) n += g.live - g.lo;
        return n;
    }
    int64_t capacity() const { return n_; }
    int64_t stride() const { return stride_; }
    int64_t n_free() const { return n_usable_; }
    /* Slots in use: a resident unit, a staged one, or a copy in flight. Fill and drain decide
     * against this, because the working slots must stay free whatever the target is. */
    int64_t held() const { return n_held_; }
    /* Where segment `k`'s covered slots end, as a slot number. */
    int64_t target(int k = 0) const { return segs_[(size_t)k].target; }
    /* Whether `slot` is below its segment's target, i.e. may be handed out. */
    bool covered(int64_t slot) const {
        if (slot < 0 || slot >= n_) return false;
        return slot < segs_[(size_t)seg_of(slot)].target;
    }
    /* The lowest free covered slot, or -1. Starvation is the normal steady state: callers retry
     * on a later dispatch rather than treat -1 as an error. */
    int64_t take() { return take_at_least(0); }

    /* Sets where segment `k`'s covered slots end, as a slot number inside it. Growing is
     * immediate (the caller has already backed the memory). Shrinking lowers `live` only past
     * slots nothing holds; the rest follows as those are released. */
    void set_target(int64_t target) { set_target(0, target); }
    void set_target(int k, int64_t target) {
        Seg& g = segs_[(size_t)k];
        g.target = target < g.lo ? g.lo : (target > g.hi ? g.hi : target);
        /* Nothing holds a slot at or past `live` (take() hands out only covered slots and `live`
         * never falls past a held one), so a set flag there only means the slot never existed. */
        for (int64_t s = g.live; s < g.target; ++s) held_[(size_t)s] = 0;
        if (g.target > g.live) g.live = g.target;
        else settle(g);
        rebuild();
    }

    /* take(), restricted to slots >= `lo`. Free slots below `lo` stay listed; there are at most
     * the working slots of them, so this is a few heap operations. */
    int64_t take_at_least(int64_t lo) {
        std::vector<int64_t> keep;
        int64_t got = -1;
        while (!free_.empty()) {
            std::pop_heap(free_.begin(), free_.end(), std::greater<int64_t>());
            const int64_t s = free_.back();
            free_.pop_back();
            if (!covered(s) || held_[(size_t)s]) continue;   /* never a slot to hand out */
            if (s < lo) { keep.push_back(s); continue; }
            held_[(size_t)s] = 1;
            ++n_held_;
            --n_usable_;
            got = s;
            break;
        }
        for (int64_t s : keep) {
            free_.push_back(s);
            std::push_heap(free_.begin(), free_.end(), std::greater<int64_t>());
        }
        return got;
    }
    /* take(), restricted to the lowest free slot `want` accepts, or -1. A linear scan; the free
     * list holds only slots passing through a swap, a few dozen. */
    template <class Want> int64_t take_where(Want want) {
        size_t best = free_.size();
        for (size_t i = 0; i < free_.size(); ++i) {
            const int64_t s = free_[i];
            if (!covered(s) || held_[(size_t)s] || !want(s)) continue;
            if (best == free_.size() || s < free_[best]) best = i;
        }
        if (best == free_.size()) return -1;
        const int64_t s = free_[best];
        free_[best] = free_.back();
        free_.pop_back();
        std::make_heap(free_.begin(), free_.end(), std::greater<int64_t>());
        held_[(size_t)s] = 1;
        ++n_held_;
        --n_usable_;
        return s;
    }
    /* The highest held slot of segment `k` (a unit, a copy in flight, a quarantined slot), or
     * -1. Scans down from `live`. */
    int64_t top_held(int k = 0) const {
        const Seg& g = segs_[(size_t)k];
        for (int64_t s = g.live; s-- > g.lo;)
            if (held_[(size_t)s]) return s;
        return -1;
    }
    /* How many existing slots at or past `from` something holds, over every segment. */
    int64_t held_from(int64_t from) const {
        int64_t k = 0;
        for (const Seg& g : segs_)
            for (int64_t s = std::max(g.lo, from); s < g.live; ++s) k += held_[(size_t)s];
        return k;
    }

private:
    struct Seg { int64_t lo, hi, target, live; };
    int seg_of(int64_t slot) const {
        for (size_t k = segs_.size(); k-- > 1;)
            if (slot >= segs_[k].lo) return (int)k;
        return 0;
    }
    void push(int64_t s) {
        free_.push_back(s);
        std::push_heap(free_.begin(), free_.end(), std::greater<int64_t>());
        ++n_usable_;
    }
    void settle(Seg& g) { while (g.live > g.target && !held_[(size_t)(g.live - 1)]) --g.live; }
    /* Every covered slot that nothing holds, once each. Ascending order is already a min-heap. */
    void rebuild() {
        free_.clear();
        for (const Seg& g : segs_)
            for (int64_t s = g.lo; s < g.target; ++s)
                if (!held_[(size_t)s]) free_.push_back(s);
        n_usable_ = (int64_t)free_.size();
    }

    int64_t stride_ = 0, n_ = 0, n_usable_ = 0, n_held_ = 0;
    std::vector<Seg> segs_;
    std::vector<int64_t> free_;
    std::vector<uint8_t> held_;
};

/* ------------------------------------------------------------------ the file tier */
/* A thin wrapper over the device layer's DirectFile, which owns the O_DIRECT alignment rules
 * (the prefix cache reads the same way). This adds the tier's accounting: a file-tier read is a
 * blocking pread inside a dispatch, and the counters make its cost visible. */
class FileTier {
public:
    /* Opens the container `path` with O_DIRECT; refuses rather than fall back to buffered reads,
     * whose latency cannot be budgeted. */
    int  open(const std::string& path);
    bool ok() const { return f_.is_open(); }
    int borrowed_fd_for_owner_identity() const { return f_.fd(); }
    void close() { f_.close(); epoch_ready_ = false; }
    int remember_epoch();
    bool same_epoch() const;

    /* One expert-sized read into a pinned buffer. Blocking and counted: a dispatch that meets a
     * file-tier unit stalls until the read returns. This is the cost the plan accepted by putting
     * the cold tail on disk; frequent reads call for a different plan, not a change here. */
    int  read(uint64_t offset, int64_t bytes, void* pinned_dst, int64_t dst_cap);

    /* RadFileEntry offsets are RAD_ALIGN_UNIT-aligned (rad_format.h), which is DirectFile's
     * alignment, so the mover's reads need no bounce buffer -- except the container's final unit,
     * whose bytes can end at an unaligned EOF (the MTP head layer's last expert); that read is
     * clamped to the file end and takes read_at's bounce path, which tolerates an EOF short read. */
    static int64_t direct_align(int64_t v) { return align_up(v, DirectFile::alignment()); }

private:
    DirectFile f_;
    int64_t    size_ = -1;      /* container size, cached at open */
    struct stat epoch_{};
    bool epoch_ready_ = false;
};

/* ------------------------------------------------------------------ residency */
class Residency final : public ResidencyTable {
public:
    void init(const Program& prog);

    const WeightSlot* lookup(rad_weight w) const override {
        return (size_t)w < slot_.size() ? &slot_[w] : nullptr;
    }
    uint64_t epoch() const override { return epoch_; }
    const std::vector<rad_weight>& changes() const override { return changed_; }
    void clear_changes() override {
        for (rad_weight w : changed_) noted_[w] = 0;
        changed_.clear();
    }

    /* The only two writers; both bump the generation. publish() settles a slot; begin_transfer()
     * marks a copy into `ptr` in flight that the compute stream must wait on (`ready`). The run
     * phase memoises its wait per generation, so a new event at an old generation would be
     * skipped (residency_iface.h). */
    void publish(rad_weight w, void* ptr) { begin_transfer(w, ptr, nullptr); }
    void begin_transfer(rad_weight w, void* ptr, RadEvent ready);
    void settle(rad_weight w) {
        if ((size_t)w >= slot_.size()) return;
        slot_[w].ready = nullptr;
        ++epoch_;
    }

private:
    void note(rad_weight w) {
        if (noted_[w]) return;
        noted_[w] = 1;
        changed_.push_back(w);
    }

    std::vector<WeightSlot> slot_;
    uint64_t                epoch_ = 1;
    std::vector<rad_weight> changed_;
    std::vector<uint8_t>    noted_;
};

/* ------------------------------------------------------------------ configuration */
struct MoverConfig {
    /* Shadow and staging slot counts come from the plan (SlabClass::n_shadow, n_stage), which
     * budgets them; a second knob here could only cause an overrun.
     *
     * How far the host may run ahead of the cards, in layer dispatches. A freed slot recycles
     * only when its quarantine event on the compute stream signals, so the move rate is shadow
     * slots / compute queue depth; 0 (unbounded) queues a whole step and starves the mover. 2 is
     * the smallest depth that keeps slots recycling with a dispatch of issue margin. A per-step
     * gate does nothing, since one step is already the whole model's queued work. */
    int max_lead_dispatches = 2;

    /* Lent slots per slab class kept free for swaps to land in (Mover::balance). A knob rather
     * than a plan field because the planner charges nothing for them: they are KV cache memory,
     * free only while it lends. A class whose shadow slots the quarter-of-the-slab cap reduced
     * gets proportionally fewer, so a small loan is not left emptier than it is filled. */
    int turnover_slots = RAD_PLACE_TURNOVER_SLOTS;

    /* The container, for the file tier. Empty means no file tier; a unit planned there is then
     * an error. */
    std::string container_path;
    /* Optional immutable source owner. Must outlive Mover and every checked copy/lease drain.
     * Empty preserves the single-container path. Committed only after old cache retirement. */
    const DsparkRadBanks* source_banks = nullptr;
    /* Pinned staging buffers for file-tier reads, one class stride each; the count the planner
     * charged to --host-pool-mib. */
    int stage_slots = RAD_PLACE_FILE_STAGE_SLOTS;
    /* Native RAM cache capacity for THIS rank, including its entry metadata. Zero disables it.
     * Engine divides the configured server-wide capacity; it is separate from the fixed pool. */
    int64_t native_file_cache_bytes = 0;
    int native_cache_allocation_device = 0; //Engine actual device, never EP rank ordinal
};

/* ------------------------------------------------------------------ the loan */
/* One stretch of another pool's VRAM that may be lent to the slab. The owner keeps it backed,
 * lends from the top down and states the loan in its own measure (tokens, for the KV cache). A
 * loan of T tokens is the top floor(T / unit_tokens) * unit_bytes bytes of every stretch. */
struct LoanStrip {
    char*   top = nullptr;       /* one past the stretch's highest byte */
    int64_t bytes = 0;           /* how far below `top` it runs */
    int64_t unit_bytes = 0;
    int64_t unit_tokens = 0;
};

struct MoverStats {
    uint64_t dispatches = 0;
    uint64_t promotions = 0, demotions = 0;
    uint64_t h2d_bytes = 0, d2h_bytes = 0;
    uint64_t prefetched = 0, prefetch_bytes = 0;
    /* Why a move did not go out: every move record busy, or a free list empty. These separate a
     * starved mover from a satisfied one. */
    uint64_t no_record = 0, starved_slab = 0, starved_pool = 0;
    uint64_t quarantined = 0;
    /* Demotions out of a lent slot that copied nothing (the unit kept its host copy). Also
     * counted in `demotions`. */
    uint64_t dropped = 0;
    /* Host copies given to budgeted victims (Mover::donor_slot); donors moved rather than
     * dropped by a recall or shrink; recalls that found no free slot for a donor. */
    uint64_t donated = 0, relocated = 0, stranded = 0;
    uint64_t ssd_reads = 0, ssd_bytes = 0;
    /* The staging leg of the same bytes: the file stager's copies from its pinned windows
     * into the VRAM staging buffer, which cross the link. Moves out of h2d_bytes so the
     * dashboard's Promotions row shows heat's churn, not the streaming it is spent on. */
    uint64_t ssd_staging_bytes = 0;
    /* The hot VRAM cache's leg: the same extents, staged into a slab slot
     * instead of the per-pass buffer because heat said the expert keeps routing. A promotion
     * the stager charges nothing extra to read for: the unit was read anyway. Pairs with
     * `ssd_staging_bytes`; together with it, the PCIe legs of `ssd_bytes`. */
    uint64_t ssd_promotions = 0, ssd_promoted_bytes = 0;
    /* The SSD tier's deposit leg: file reads whose bytes stayed in a
     * pool slot -- the kept-copy half of the split, like ssd_promoted_bytes, only one tier
     * down. Evictions stay pool_*: they relabel a cold pool unit -- native or deposited, the
     * container is ground truth for both -- back to the file tier, copyless, so they are the
     * pool's ledger, not SSD traffic. */
    uint64_t ssd_deposits = 0, ssd_deposit_bytes = 0, pool_evictions = 0;
    /* The prefill stager (stage_units). `stage_overflow`: units no buffer had room for, read in
     * place. `stage_copies`: copies issued; neighbouring pool slots go as one. */
    uint64_t stage_layers = 0, staged_units = 0, staged_bytes = 0, stage_overflow = 0;
    uint64_t stage_copies = 0;
    /* The file stager (stage_file_units), including the host time spent in preads and waiting
     * for a window's copies to drain. */
    uint64_t ssd_layers = 0, ssd_units = 0, ssd_read_bytes = 0, ssd_preads = 0;
    uint64_t ssd_read_ns = 0, ssd_wait_ns = 0;
    /* Routing-aware reads: units of experts whose routing count was zero at the layer's
     * trigger, not read (their weights are re-published to the layer buffer as a dummy
     * address for the pass, and nulled again with the staged units). */
    uint64_t ssd_skipped_units = 0, ssd_skipped_bytes = 0;
    /* ---- ours (ds41 RAM cache): units a routed-only read left unread (alias_file_units): the pass
     * did not route to them. */
    uint64_t file_aliased = 0;
    /* File-tier weights relaid on the read (Mover::set_file_relayout), and the issuing thread's
     * time in those reads and relayouts, which run on several threads at once. */
    uint64_t file_relaid = 0, file_relaid_bytes = 0, file_relayout_ns = 0;
    uint64_t mapped_source_copies = 0, mapped_source_bytes = 0;
    /* Existing read-only prefill accounting, aggregated after reader retirement. */
    uint64_t file_prefill_ro_hits = 0, file_prefill_ro_hit_bytes = 0;
    uint64_t file_prefill_ro_misses = 0, file_prefill_ro_miss_bytes = 0;
};

/* MoverStats as the mover counts it on its rank thread, readable from the HTTP thread (Relaxed,
 * heat.h); snapshot() is the plain copy the readers take. */
struct MoverCounters {
    Relaxed<uint64_t> dispatches, promotions, demotions, h2d_bytes, d2h_bytes, prefetched,
                      prefetch_bytes, no_record, starved_slab, starved_pool, quarantined, dropped,
                      donated, relocated, stranded, ssd_reads, ssd_bytes, stage_layers,
                      staged_units, staged_bytes, stage_overflow, stage_copies, ssd_staging_bytes,
                      ssd_promotions, ssd_promoted_bytes, ssd_deposits, ssd_deposit_bytes,
                      pool_evictions, ssd_layers, ssd_units, ssd_read_bytes, ssd_preads,
                      ssd_read_ns, ssd_wait_ns, ssd_skipped_units, ssd_skipped_bytes,
                      /* ours */ file_aliased, file_relaid, file_relaid_bytes, file_relayout_ns,
                      mapped_source_copies, mapped_source_bytes, file_prefill_ro_hits,
                      file_prefill_ro_hit_bytes, file_prefill_ro_misses, file_prefill_ro_miss_bytes;

    MoverStats snapshot() const {
        MoverStats s;
        s.dispatches         = dispatches;
        s.promotions         = promotions;
        s.demotions          = demotions;
        s.h2d_bytes          = h2d_bytes;
        s.d2h_bytes          = d2h_bytes;
        s.prefetched         = prefetched;
        s.prefetch_bytes     = prefetch_bytes;
        s.no_record          = no_record;
        s.starved_slab       = starved_slab;
        s.starved_pool       = starved_pool;
        s.quarantined        = quarantined;
        s.dropped            = dropped;
        s.donated            = donated;
        s.relocated          = relocated;
        s.stranded           = stranded;
        s.ssd_reads          = ssd_reads;
        s.ssd_bytes          = ssd_bytes;
        s.stage_layers       = stage_layers;
        s.staged_units       = staged_units;
        s.staged_bytes       = staged_bytes;
        s.stage_overflow     = stage_overflow;
        s.stage_copies       = stage_copies;
        s.ssd_staging_bytes  = ssd_staging_bytes;
        s.ssd_promotions     = ssd_promotions;
        s.ssd_promoted_bytes = ssd_promoted_bytes;
        s.ssd_deposits       = ssd_deposits;
        s.ssd_deposit_bytes  = ssd_deposit_bytes;
        s.pool_evictions     = pool_evictions;
        s.ssd_layers         = ssd_layers;
        s.ssd_units          = ssd_units;
        s.ssd_read_bytes     = ssd_read_bytes;
        s.ssd_preads         = ssd_preads;
        s.ssd_read_ns        = ssd_read_ns;
        s.ssd_wait_ns        = ssd_wait_ns;
        s.ssd_skipped_units  = ssd_skipped_units;
        s.ssd_skipped_bytes  = ssd_skipped_bytes;
        s.file_aliased       = file_aliased;
        s.file_relaid        = file_relaid;
        s.file_relaid_bytes  = file_relaid_bytes;
        s.file_relayout_ns   = file_relayout_ns;
        s.mapped_source_copies = mapped_source_copies;
        s.mapped_source_bytes  = mapped_source_bytes;
        s.file_prefill_ro_hits       = file_prefill_ro_hits;
        s.file_prefill_ro_hit_bytes  = file_prefill_ro_hit_bytes;
        s.file_prefill_ro_misses     = file_prefill_ro_misses;
        s.file_prefill_ro_miss_bytes = file_prefill_ro_miss_bytes;
        return s;
    }
};

/* ------------------------------------------------------------------ the mover */
struct NativeMappedMoverHostTest;
struct AuxStagerProbe;
class Mover {
    friend struct NativeMappedMoverHostTest; //private HOST-only fixture seam
    friend struct AuxStagerProbe; //tests/aux_stager_test.cpp: borrow_hook_ only
    friend class Engine; //drain mapped source consumers before Engine tears down arenas/events
public:
    ~Mover();
    Mover() = default;
    Mover(const Mover&) = delete;
    Mover& operator=(const Mover&) = delete;

    /* Allocates every pool the plan asks for and assigns every unit a slot; the load phase fills
     * the published addresses. On RAD_E_NOMEM the log names the pool.
     *
     * Always pass `budget`: Pools already reserved the weights pool, so allocating beside it
     * spends the budget twice, and reserving from Pools::weights() keeps promotions out of KV
     * blocks (mem/pools.h). Null is the standalone-test path and logs a warning. */
    int init(const Program& prog, const Plan& plan, const MoverConfig& cfg, Pools* budget = nullptr);

    /* The compute stream this rank issues on. Transfers are ordered against it and the
     * quarantine barrier is recorded on it; without one the mover refuses to move. */
    int bind_compute_stream(RadStream s);
    RadStream compute_stream() const { return compute_; }

    /* The mover's stream, to be shared by any subsystem moving bytes off the critical path. Each
     * HIP stream costs a hardware queue, and on RDNA4 a fourth one is expensive: on two R9700s
     * serving Flash-Next, an idle fourth stream takes the decode step from 20.5 to 33.9 ms
     * (GPU_MAX_HW_QUEUES 1-3 are fine, 4 and 8 are not). The engine already uses three (compute,
     * the second compute lane, this one).
     *
     * Null before init() and on a rank with nothing to move; the caller then uses its own stream.
     * The run phase issues here too, so a sharer's large copy delays transfers the compute stream
     * waits on; gate the two rather than add a queue. */
    RadStream transfer_stream() const { return mover_; }
    //Private fixed-head tail owner validates BOTH immutable source banks on warm reuse too.
    bool source_epochs_current() const { return source_file_epoch(); }

    ResidencyTable* residency() { return &res_; }
    /* For Tier::Mapped weights, which live in the container mapping the load phase holds: the
     * one address the mover does not own, recorded so the residency table stays the only record
     * of weight addresses. A mapped unit has no slab class, so nothing republishes over it. */
    void publish_mapped(rad_weight w, void* p) { res_.publish(w, p); }

    /* ---- the run phase, in the order a layer dispatch calls it -----------------------------
     *
     * begin_dispatch  retire what landed, gate the host's lead, record the ordering event
     * prefetch_to_op  release what this op has finished with, then start every transfer the
     *                 exact schedule says is due, while slots are free
     * issue           one heat proposal
     *
     * There is deliberately no end_dispatch: the quarantine barrier must be recorded downstream
     * of the previous dispatch's launches, which begin_dispatch's retire() is.
     *
     * A caller must issue as well as retire every dispatch. The pool free list starts empty, so a
     * loop that only retires promotes into the shadow slots once and never demotes; on the host
     * backend copies land within their dispatch, so this shows only on a real card. The compute
     * stream must also carry the dispatch's real work, or the quarantine barrier and the lead
     * gate are meaningless.
     *
     * In production, begin_dispatch and issue run once a step from Engine::tiering_dispatch,
     * since the mover relocates only between steps (Ctx::run_step syncs the residency table at
     * its top). prefetch_to_op is called only by tests/place_test.cpp; the
     * exact-schedule path is kept built and tested deliberately.
     */
    int begin_dispatch(int dispatch, HeatEngine* heat);
    int prefetch_to_op(int32_t op_index);
    int issue(const Swap& s, HeatEngine* heat);

    /* ---- the loan: VRAM another pool is not using ------------------------------------------
     *
     * An expert that does not fit the slab is read across the link on every step that routes to
     * it, so unused VRAM elsewhere is lent to the slab: the part of the KV cache's carve nothing
     * holds, and the part of the activation arena that steps smaller than the largest do not
     * reach. Both are addresses the lender already owns, so no allocation or mapping is involved.
     * Only the caller that arbitrates the whole card may lend.
     *
     * Each lender lends in its own measure (KV cache: tokens; arena: bytes) into its own slab
     * segment, and they move independently, so every call names its lender.
     *
     * A unit in a lent slot keeps its host copy (weights never change), so giving the slot back
     * is a change of address: a recall waits only for kernels to stop naming the slot. */

    enum Lender { kLendKV = 0, kLendArena = 1, kLenders = 2 };

    /* Registers the stretches a lender may ever lend; once per lender, after its memory is
     * carved. Each stretch is cut into slots of each class's stride from the top down, so a small
     * loan covers a prefix of a large loan's slots and the deepest slot is given back first. */
    int     lend_from(int lender, const std::vector<LoanStrip>& strips);

    /* How much a lender lends, in its own measure. A target, not a transaction: newly covered
     * slots can be filled at once, while uncovered ones empty as their units drop and their
     * quarantine clears. loan_held() is how far into the lender's memory the slab still reaches;
     * the lender may not reuse those bytes until it has fallen to the loan. */
    int     set_loan(int lender, int64_t amount);
    int64_t loan_held(int lender) const;
    int64_t loan(int lender) const { return loan_[lender]; }

    /* set_loan(), completed before returning: units past `amount` are dropped to their host
     * copies and both streams drained. A host wait, for when the lender needs more than it kept
     * in hand (a request larger than the cache's buffer, a step larger than the arena allows). */
    int     reclaim(int lender, int64_t amount, HeatEngine* heat);

    /* Bytes the lent slots hold, bytes the loans offer, and bytes every carved slot could hold.
     * The first two are stored because the HTTP thread reads them for the dashboard and only the
     * rank thread may walk the slot pools. */
    int64_t flex_used_bytes() const { return flex_used_.load(std::memory_order_relaxed); }
    int64_t flex_offered_bytes() const { return flex_offered_.load(std::memory_order_relaxed); }
    int64_t flex_capacity() const;
    /* The lent slots the fill leaves free for the engine to swap into
     * (MoverConfig::turnover_slots), summed over the slab classes. */
    int64_t turnover_slots() const {
        int64_t n = 0;
        for (const ClassPools& cp : pools_) n += cp.n_turn;
        return n;
    }
    /* The loan that covers every slot carved from this lender; more would sit empty. Zero when
     * nothing is carved. */
    int64_t loan_ceiling(int lender) const;

    /* Fills free lent slots and empties slots the loan no longer covers: the half-swaps the heat
     * engine never proposes, since its decisions need a victim. Every free slot past the turnover
     * slots is filled in one call, hottest units first, since an empty slot wastes VRAM every
     * step. The turnover slots stay free for the engine's swaps; filled, they would cap the move
     * rate at the shadow slots. When fewer than the working plus turnover slots are free, the
     * coldest lent units are dropped (an address change, never a copy).
     *
     * `turnover` false fills the turnover slots too, for steps bound by their own staging copies
     * (Engine::tiering_dispatch); the next call with true drops them back out. */
    int balance(HeatEngine* heat, bool turnover = true);

    /* retire() then balance() for an idle rank, so a loan made while idle is filled before the
     * next request. Call only while the rank thread is parked. */
    int idle_tick(HeatEngine* heat);
    /* Whether idle ticks keep the turnover slots filled; the engine holds them through a prompt
     * and the few steps after it (Engine::tiering_dispatch). */
    void hold_turnover(bool held) { hold_turnover_ = held; }
    /* Each class's pinned pool as the card addresses it: [lo, hi) and the unit stride. At most `max`. */
    int pool_ranges(uint64_t* lo, uint64_t* hi, uint64_t* stride, int max) const {
        int n = 0;
        for (const ClassPools& cp : pools_)
            if (cp.pool_dev && cp.stride > 0 && n < max) {
                lo[n] = (uint64_t)(uintptr_t)cp.pool_dev;
                hi[n] = lo[n] + (uint64_t)cp.pool.size() * (uint64_t)cp.stride;
                stride[n] = (uint64_t)cp.stride;
                ++n;
            }
        return n;
    }

    /* Where a unit's bytes are right now, as a device-visible address, or null for a file-tier
     * unit that has not been staged. */
    void* unit_base(int32_t unit) const;

    /* ---- staging within a pass: the prefill stager (core/place/stager.h) ----------------------
     *
     * The only calls that change a unit's address within a step, and only the published one: the
     * unit stays in its pinned pool slot, and one routed layer's ops read a copy in a
     * caller-owned buffer.
     *
     * stage_units   copies each pooled unit of `units` with no slab slot and no copy in flight
     *               into `dst`, packed at the allocation unit; a unit with no room is read in
     *               place. Runs on the mover's stream behind `after` (if set) and records `done`.
     *               Copies are published settled: the caller waits on `done` once instead of
     *               once per weight (a few hundred per layer).
     * unstage_units publishes those units at their pool slots again. The caller must already have
     *               ordered the next writer of `dst` behind every reader of the copies.
     * stage_need    the room stage_units would use with no cap, for sizing shares of a region;
     *               `largest`, if set, gets the largest unit's room.
     *
     * The span form packs into up to two stretches (a ring region that wraps is two).
     */
    struct StageSpan { char* dst; int64_t cap; };
    int  stage_units(const std::vector<int32_t>& units, char* dst, int64_t cap, RadEvent after,
                     RadEvent done, std::vector<int32_t>* staged);
    int  stage_units(const std::vector<int32_t>& units, const StageSpan* spans, int n_spans,
                     RadEvent after, RadEvent done, std::vector<int32_t>* staged);
    void unstage_units(const std::vector<int32_t>& staged);
    int64_t stage_need(const std::vector<int32_t>& units, int64_t* largest = nullptr) const;

    /* ---- routed layers read from the container: the file stager (core/place/stager.h) --------
     *
     * A file-tier unit has no address, but a grouped GEMM takes every expert's pointer at issue.
     * So before a routed layer's first expert op its file-tier units are read into a VRAM buffer
     * and published there; afterwards unstage_units publishes them as null again. The plan
     * charges the two buffers to the weights pool (Plan::file_stage_vram).
     *
     * stage_file_units  reads every file-tier unit of `units` into `dst`, packed at RAD_ALIGN_UNIT
     *                   in the order given, through the pinned read windows: container-order
     *                   runs, each window draining to `dst` on the mover's stream while the next
     *                   fills, behind `after` (if set), with `done` recorded last. Published
     *                   settled. RAD_E_SCRATCH if the units exceed `cap`. The reads block this
     *                   thread.
     *
     * Routing-aware reads: with `ecnt` (host, `n_expert` entries, the layer's per-expert
     * routing counts for the current pass), a unit whose expert is counted zero is not read:
     * it goes to `skipped`, its weights are published at `dst` (a dummy address inside the
     * layer buffer, valid memory a no-run expert never reads through), and the caller nulls
     * it again with the staged units. Null `ecnt` reads the whole layer.
     * file_buffer       buffer 0 or 1 of the plan's two, or null when the plan has none. */
    int   stage_file_units(const std::vector<int32_t>& units, char* dst, int64_t cap,
                           RadEvent after, RadEvent done, std::vector<int32_t>* staged,
                           std::vector<int32_t>* skipped = nullptr,
                           const int32_t* ecnt = nullptr, int32_t n_expert = 0,
                           std::vector<int32_t>* promoted = nullptr,
                           NativeFileCacheReadPolicy policy = NativeFileCacheReadPolicy::Normal,
                           M1PrefetchTrace trace = {});
    void prefetch_trace_route(const M1PrefetchTrace& trace, const int32_t* counts, size_t SSD_units);
    void prefetch_trace_stage_done(const M1PrefetchTrace& trace, int status);
    /* alias_file_units  publishes every file-tier weight of `units` at `base` without reading
     *                   it, for experts a pass does not route to: the grouped GEMM is handed
     *                   every expert's pointer and reads none of a zero-row expert's bytes, so
     *                   the pointer only has to name memory the card owns (the file buffer, at
     *                   least one unit long). Appends them to `staged`; unstage_units takes them
     *                   back like read ones. */
    void  alias_file_units(const std::vector<int32_t>& units, char* base,
                           std::vector<int32_t>* staged);
    /* The same with `mark` in the published address's low bits (tools/ds41-host-experts/ds41_p1.h:
     * kMarkSkip): prefill Phase 1 publishes a touched non-resident expert so that the plugin's own
     * expert ops leave its rows alone (place/card_expert_stager.h). The weight's 64-byte alignment
     * leaves the bits free; only a pass whose rows resolve to the P1 twin may see such an entry. */
    void  alias_file_units(const std::vector<int32_t>& units, char* base,
                           std::vector<int32_t>* staged, uintptr_t mark);
    /* alias_pool_units  the same for units pinned in the host pool, which prefill Phase 1 copies to
     *                   its ring from their pool slots: every weight published at `base` with `mark`,
     *                   appended to `staged`; unstage_units publishes them at their pool slots again.
     *                   A unit not in the pool, or in a slab slot, is left as it is. */
    void  alias_pool_units(const std::vector<int32_t>& units, char* base, std::vector<int32_t>* staged,
                           uintptr_t mark);
    /* borrow_host_unit  for a reader on the host (place/host_expert_pool.h): when the mapped RAM
     *                   cache holds every weight of file-tier unit `u`, a lease on each, which
     *                   keeps its bytes in place until release_host_unit, and their host
     *                   addresses in the unit's weight order. False, with nothing held, when a
     *                   weight is not cached or the cache is off. Borrowing touches the entries,
     *                   as a staged read does. `bytes` is each weight's stored (block_mxfp4) size;
     *                   with the packed cache (packed_cache()) the slots hold the packed form
     *                   instead, `slot` bytes each (ds41_host_experts.h packed_view). */
    struct HostUnit {
        int32_t        unit = -1;
        int            n = 0;
        bool           packed = false;
        const uint8_t* host[2] = { nullptr, nullptr };
        const void*    dev[2] = { nullptr, nullptr };   /* the same bytes as the card's DMA reads them */
        int64_t        bytes[2] = { 0, 0 };
        int64_t        slot[2] = { 0, 0 };
        NativeMappedFileCache::Lease lease[2];
    };
    /* `touch` false: the entries are leased as they are, no move to the LRU front and their second
     * chances kept (borrow_readonly / borrow_packed_readonly), the hits counted all the same. */
    bool  borrow_host_unit(int32_t u, HostUnit* out, bool touch = true);
    void  release_host_unit(HostUnit* hu);
    /* read_unit_planes  for a host reader of units the RAM cache misses: reads each unit's
     *                   weights from the container as the release stores them (codes and scales
     *                   planes, no relayout, no cache) into buffers the HostPlanes owns, every
     *                   weight of every unit on its own thread at once. False when a unit is not
     *                   a plain file-tier ggml_mxfp4 pair or a read fails.
     * insert_unit_planes relays one unit's planes into the stored form and admits it into the
     *                   mapped RAM cache, so a later pass hits; thread-safe (relayouts are pure,
     *                   the cache locks), for a thread beside the issuing one. */
    struct HostPlanes {
        int32_t        unit = -1;
        int            n = 0;
        rad_weight     w[2] = { 0, 0 };
        int64_t        rows[2] = { 0, 0 }, cols[2] = { 0, 0 };   /* elements */
        const uint8_t* codes[2] = { nullptr, nullptr };
        const uint8_t* scales[2] = { nullptr, nullptr };
        uint8_t*       buf[2] = { nullptr, nullptr };
        int64_t        cap[2] = { 0, 0 };
        int64_t        a0[2] = { 0, 0 }, len[2] = { 0, 0 };     /* file range read */
        HostPlanes() = default;
        HostPlanes(const HostPlanes&) = delete;
        HostPlanes& operator=(const HostPlanes&) = delete;
        ~HostPlanes() { for (uint8_t* b : buf) std::free(b); }
    };
    /* One read_unit_planes call's timing, in ns from its dispatch (the readers handed the reads),
     * for the CPU path's counters: which leg lands last, the mirror's copy out of its bounce
     * buffer, the readers' start and the caller's return (RESEARCH-2-RANKED-PLAN #7's gates). */
    struct PlaneReadTiming {
        int     units = 0, reads = 0;
        int     mirror_reads = 0;   /* the mirror's tail reads (pieces, with ..._MIRROR_SPLIT) */
        bool    posted = false;     /* start_unit_planes .. finish_unit_planes */
        int64_t first_ns = 0;       /* the first read begun */
        int64_t first_container_ns = 0, last_start_ns = 0;   /* the first container read, the last of all, begun */
        int64_t container_ns = 0;   /* the last container read landed (0: none) */
        int64_t mirror_ns = 0;      /* the last mirror read landed, its copy included (0: none) */
        int64_t copy_ns = 0;        /* the longest mirror copy (pieces copy at once) */
        int64_t copy_sum_ns = 0;    /* the mirror copies, summed */
        int64_t join_ns = 0;        /* posted: the caller came to wait (finish_unit_planes) */
        int64_t return_ns = 0;      /* the caller back, every read landed */
        int64_t t0_ns = 0;          /* the dispatch itself (steady clock ns) */
        /* Units read past bytes they had in already (PlanesIn: a read-ahead's rest), of them those
         * whose rest the plane mirror read part of, and the bytes read for them, of those the plane
         * mirror's. */
        int     rest = 0, rest_mirror = 0;
        int64_t rest_bytes = 0, rest_mirror_bytes = 0;
    };
    /* A unit whose planes are partly in already (a read-ahead's pieces, readahead.h): the first
     * bytes[k] of weight k's extent -- [a0, a0 + len) as lay_out_unit_planes laid it out -- are
     * read, or being read, by someone else, and a call reads only the rest of each extent, into
     * the same places; each is 0, a multiple of the O_DIRECT alignment, or the whole extent. All
     * zero: the whole unit, as without. */
    struct PlanesIn { int64_t bytes[2] = { 0, 0 }; };
    /* `rows`: the pass's tokens, which decide whether the plane mirror takes a share. `in` (one a
     * unit, or null for none): what each unit has in already; `rest_mirror`: the plane mirror
     * takes its decode share of such a unit's rest (rest_mirror_tail), else the rest is read from
     * the container alone (RADIANCE_EXPERIMENTAL_DS41_PREFETCH_URGENT=2 / 1). A unit with bytes
     * in must be laid out where they are (its buffers big enough already): it is never moved.
     * `hook`: a call on the issuing thread while it waits for the kept readers, its at_ns counted
     * from the call's dispatch (NativeReadWorkers::WaitHook; the CPU pool's pre-wake, stager.h
     * hx_wake_*); none on the threads-a-call fallback. With `landed`, each unit i (its index in
     * `units`) is handed to landed(ctx, i) by the reader that lands its last extent, every extent
     * of it read without error, while the other units' reads go on: a caller can start computing a
     * unit as soon as its bytes are in (large-chunk prefill). A unit one of whose reads failed is
     * not called back (a mirror leg is read again from the container after the readers, and the
     * call's result says whether that worked); nor is any unit of a call that could not start its
     * readers. `landed` runs on a reader thread, at most once a unit, and must not call into this
     * Mover. */
    using ReadWaitHook = NativeReadWorkers::WaitHook;
    using PlaneLanded = void (*)(void* ctx, int i);
    bool  read_unit_planes(HostPlanes* const* units, int n, int rows = 1, PlaneReadTiming* timing = nullptr,
                           const PlanesIn* in = nullptr, bool rest_mirror = false,
                           const ReadWaitHook* hook = nullptr, PlaneLanded landed = nullptr,
                           void* ctx = nullptr);
    /* read_unit_planes in two halves, for the issuing thread (RADIANCE_EXPERIMENTAL_DS41_MISS_POST):
     * start_unit_planes hands the reads to the kept readers and returns at once, so they run while
     * the caller goes on (the pool's jobs for the cached units); finish_unit_planes reads whatever
     * no reader has begun, waits for the rest and returns what read_unit_planes would have. False
     * from start: nothing started (one already in flight, no kept readers, the readers busy with
     * another thread's read_unit_planes -- start never waits for them -- or a unit not eligible),
     * read as before. One call in flight a Mover; the same thread calls both, and calls nothing
     * else that reads planes between them; the units' buffers stay put until the finish. */
    bool  start_unit_planes(HostPlanes* const* units, int n, int rows, const PlanesIn* in = nullptr,
                            bool rest_mirror = false);
    bool  finish_unit_planes(PlaneReadTiming* timing = nullptr, const ReadWaitHook* hook = nullptr);
    /* One read of a weight's extent, in bytes from its a0: the container's, or the plane mirror's
     * copy of the same bytes (mirror). */
    struct PlaneSpan { int64_t off = 0, len = 0; bool mirror = false; };
    /* The reads of an extent of `len` bytes whose first `in` are in already, the plane mirror
     * taking [end - tail, end) (end: its codes plane's end; tail 0: no mirror): the container's
     * head up to the mirror's part, the mirror's part, the container's tail past `end`, each only
     * where bytes are left, into out in that order. Returns how many (0: nothing left). With
     * nothing in, the reads read_unit_planes has always made of a whole extent. */
    static int plane_spans(int64_t len, int64_t in, int64_t end, int64_t tail, PlaneSpan out[3]);
    /* The mirror's part of a unit's rest (RADIANCE_EXPERIMENTAL_DS41_PREFETCH_URGENT=2): the decode
     * tail `tail` scaled by the share of the unit's `total` bytes still to read (`left`), at most
     * the mirrored w3 codes (`w3` bytes ending at `end`) past `in`, rounded down to `align`. */
    static int64_t rest_mirror_tail(int64_t tail, int64_t left, int64_t total, int64_t in,
                                    int64_t end, int64_t w3, int64_t align);
    bool  unit_planes_started() const { return plane_posted_; }
    bool  host_planes_eligible(int32_t u) const;   /* what read_unit_planes accepts */
    /* lay_out_unit_planes  for a read-ahead (readahead.h): unit p->unit laid out in p exactly as
     *                      read_unit_planes lays it out before reading, and each weight's container
     *                      extent in ext[0..1] (the whole extent, from the container: no mirror).
     *                      Reads nothing and reads only the plan and the program, so a thread
     *                      beside the issuing one may call it. False when read_unit_planes would
     *                      refuse the unit's weights.
     * read_piece           one O_DIRECT read of [off, off + len) of container `file` into dst (cap
     *                      bytes there), as read_unit_planes reads an extent; thread-safe.
     * decode_mirror_bytes  the bytes of a missed unit a one-token pass reads from the plane mirror
     *                      instead of the container (0 with no mirror, or one turned off).
     * mirror_bytes         the same for a pass of `rows` tokens, as read_unit_planes picks the share
     *                      (the decode share under the mirror's min_rows, its share from there). */
    bool  lay_out_unit_planes(HostPlanes* p, PlaneExtent ext[2]);
    int   read_piece(uint32_t file, int64_t off, int64_t len, void* dst, int64_t cap);
    int64_t decode_mirror_bytes() const;
    int64_t mirror_bytes(int rows) const;
    /* read_weight_file  a whole weight as the container holds it (every row on this rank) into
     *                   dst, weight_file_bytes(w) long (rad_core.h), for a host reader such as
     *                   the router predictor: a weight stored as is, or the one plane a kernel's
     *                   relayout read (an f32 router gemm_nt_f32bf16 keeps as its bf16 halves). */
    bool  read_weight_file(const WeightInfo& w, void* dst);
    bool  mapped_cache_on() const { return mapped_file_cache_on_; }
    int64_t mapped_cache_bytes() const { return mapped_file_cache_on_ ? cfg_.native_file_cache_bytes : 0; }
    /* The host memory the mapped cache holds now, against its mapped_cache_bytes() budget: its
     * slabs' backing (every slab once ..._PREGROW pinned them at init, the grown ones otherwise),
     * or the entries' payload without slabs. 0 with the cache off. Engine::host_headroom_bytes
     * stops reserving it twice (RADIANCE_EXPERIMENTAL_CACHE_HEADROOM_BACKED). */
    int64_t mapped_cache_backed_bytes() const;
    /* RADIANCE_EXPERIMENTAL_DS41_PACKED_CACHE=1 (slabs only): the mapped cache holds the routed
     * experts in the CPU-only packed form (tools/ds41-host-experts/ds41_host_experts.h), which only
     * borrow_host_unit's readers can use: the card's copy path neither reads nor fills it. The
     * file stager refuses it unless the CPU computes every routed unit (..._FILE_BUFFER_UNITS).
     * ..._PACKED_CACHE_RAW=<gate_up>,<down>: the raw rounds a slot holds (see mover.cpp). */
    bool  packed_cache() const { return mapped_file_cache_on_ && mapped_file_cache_.packed(); }
    /* The packed cache's slot size of class `klass` (0 gate_up, 1 down), what a borrow's slot[] says; 0
     * without the packed cache (the aux expert worker's slot layout, core/place/aux_expert_worker.h). */
    int64_t packed_slot_bytes(int klass) const {
        return packed_cache() && (klass == 0 || klass == 1) ? (int64_t)mapped_file_cache_.slot_bytes()[(size_t)klass] : 0;
    }
    /* One gate_up and one down slot of the mapped cache's slabs, with their guards. */
    int64_t mapped_pair_bytes() const;
    /* Pins the mapped cache's slabs (as ..._PREGROW does; again adds nothing) and returns the
     * pairs it can hold: the fewer of its gate_up and down slots. 0 without slabs. */
    int64_t mapped_cache_pregrow();
    /* Marks `units` profile-hot in the mapped cache and gives each such entry `chances` moves to the
     * front between uses before an insert may evict it (0: plain LRU). The units marked. */
    int     mapped_cache_second_chances(const std::vector<int32_t>& units, int chances);
    /* Whether the mapped RAM cache holds every weight of unit u, without touching it. */
    bool  unit_cached(int32_t u);
    /* unit_cached for n units under one lock of the cache (out[i] 0/1): the routing trace's
     * residency snapshot (route_trace.h). */
    void  units_cached(const int32_t* units, int n, uint8_t* out);
    /* `copy`: how the packed cache's insert writes each weight's codes plane (ds41hx::PackCopy:
     * non-temporal stores, pieces, a hook before each; the background inserts' ..._INSERT_NT and
     * ..._INSERT_SLICE_KIB). The slot's bytes are the same; a block_mxfp4 cache ignores it. */
    int   insert_unit_planes(const HostPlanes& p, std::vector<uint8_t>* scratch,
                             const ds41hx::PackCopy* copy = nullptr);
    /* The packed cache's outcomes, all ranks' threads: weights packed and published; units not
     * cached because a weight had more raw rounds than its slot takes (attempts, and distinct
     * units); weights whose packed scales failed the round trip (never published). */
    struct PackedStats { uint64_t inserted = 0, over_budget = 0, over_budget_units = 0, round_trip_failed = 0; };
    PackedStats packed_stats() const;
    /* The packed cache refused unit u over its raw-round budget (at least once: a refusal is the
     * unit's for good, its scales decide it). */
    bool packed_refused(int32_t u) const {
        return packed_refused_ && u >= 0 && (size_t)u < unit_.size() &&
               packed_refused_[(size_t)u].load(std::memory_order_relaxed) != 0;
    }
    /* retire_file_windows  gives back the RAM-cache leases the read windows keep for their last
     *                      copies, for a caller that knows those copies are done (the file
     *                      stager, past a routing sync the card reached after them). A window
     *                      otherwise keeps them until it comes round again; when the windows sit
     *                      idle -- the CPU takes the misses -- the held entries drift to their
     *                      class's LRU tail, where each one turns every admission away. */
    int   retire_file_windows();
    /* mirror_init  once the plan is known: maps every file-tier unit's gate_up codes to the
     *              plane mirror's copy (place/plane_mirror.h) and samples units against the
     *              container; from then on read_unit_planes reads each gate_up codes plane's tail
     *              from the mirror beside the container. Nothing when no mirror is configured, and
     *              a mismatch turns it off for this rank. */
    int   mirror_init();
    /* set_file_relayout  the function that makes a file-tier weight's stored form from its
     *                    container planes when the two differ (weight_file_relaid): the kernel's
     *                    relayout, as the loader's relayout pool calls it once for a resident
     *                    weight. Every read of such a weight -- a staging slot (stage) or the
     *                    file stager -- reads its planes into a host input, relays them into the
     *                    pinned memory the copy to the card reads (the staging slot, or a read
     *                    window), and copies as usual. Must be pure and thread-safe: the file
     *                    stager relays a window's weights on several threads. Set before init;
     *                    without it a plan with such a weight on the file tier is refused. */
    using FileRelayout = std::function<int(const WeightInfo& w, const RadTensor* planes,
                                           void* dst, int64_t dst_bytes)>;
    void  set_file_relayout(FileRelayout fn) { relayout_ = std::move(fn); }
    /* ---- prefill Phase 1 (place/card_expert_stager.h) ---------------------------------------
     *
     * file_region       the file stager's two buffers as one region (the ring of whole slots), its
     *                   bytes in *bytes; null without buffers.
     * ring_copy         copies `n` segments on the mover's stream (the card's DMA lane, one list),
     *                   behind each of `waits` (null entries skipped), and records `done` after them.
     * mover_wait        the mover's stream waits on `ev` (a hand-over: copies queued later run
     *                   behind it).
     * file_unit_in_slab a file-tier unit the slab or a ring slot holds (set_ring_resident): published
     *                   at its slot, never aliased.
     * set_ring_resident the lent ring (card_expert_stager.h, RingLender): file-tier unit u, whose bytes
     *                   were copied into `slot` (a ring slot of the file region, laid out as the unit is:
     *                   weight i at its offset) and whose copy the host has seen pass, published there
     *                   and from then on a resident -- never aliased, staged or borrowed -- until
     *                   set_ring_resident(u, nullptr) makes it a file-tier unit again (published as
     *                   null; its layer's passes stage or alias it as any other). The issuing thread
     *                   only, between passes; ring_resident may be asked from any thread.
     * relay_unit_block  one missed unit's planes in the stored block form, each weight at its offset in
     *                   the unit, into `dst` (cap bytes, at least the unit's): the bytes a VRAM copy of
     *                   the unit holds. Pure and thread-safe (the lent ring's reload thread).
     * pack_unit_for_ring one missed unit's planes into `dst` (cap bytes, at least ds41p1::kSlotBytes)
     *                   as a ring slot holds it: gate_up at 0, down at ds41p1::kDnOff; in the packed
     *                   RAM cache's form (pack_planes with the cache's raw budgets and its round-trip
     *                   check) when that cache is on and both weights fit their budgets, else in the
     *                   stored block form (the file tier's relayout). *packed says which. Pure and
     *                   thread-safe, for a packer thread beside the issuing one. */
    char* file_region(int64_t* bytes) const {
        if (bytes) *bytes = file_vram_ ? 2 * file_buf_bytes_ : 0;
        return file_vram_;
    }
    /* The lent slab (Plan::lend_slab_vram): the VRAM the plan set aside for the lender's own slots, its
     * bytes without the tail pad in *bytes; null without the switch. */
    char* lend_slab_region(int64_t* bytes) const {
        if (bytes) *bytes = lend_slab_ ? lend_slab_bytes_ : 0;
        return lend_slab_;
    }
    int   ring_copy(const RadCopySeg* segs, int n, const RadEvent* waits, int n_waits, RadEvent done);
    int   mover_wait(RadEvent ev);
    bool  file_unit_in_slab(int32_t u) const {
        return u >= 0 && (size_t)u < unit_.size() && (unit_[(size_t)u].slab_slot >= 0 || ring_resident(u));
    }
    void  set_ring_resident(int32_t u, char* slot);
    bool  ring_resident(int32_t u) const {
        return ring_res_ && u >= 0 && (size_t)u < unit_.size() &&
               ring_res_[(size_t)u].load(std::memory_order_relaxed) != nullptr;
    }
    int   relay_unit_block(const HostPlanes& p, uint8_t* dst, int64_t cap) const;
    int   pack_unit_for_ring(const HostPlanes& p, uint8_t* dst, int64_t cap, bool* packed,
                             int64_t bytes[2]) const;   /* bytes: gate_up's and down's, as written */
    char* file_buffer(int i) const {
        return (file_vram_ && (i == 0 || i == 1)) ? file_vram_ + i * file_buf_bytes_ : nullptr;
    }
    int64_t file_buffer_bytes() const { return file_vram_ ? file_buf_bytes_ : 0; }

    /* A snapshot by value: the counters are atomics read while the rank thread writes them. */
    MoverStats stats() const { return stats_.snapshot(); }
    std::string report() const;

private:
    /* One in-flight relocation: Free -> Copying -> Published -> Quarantine -> Free. Transfers
     * are asynchronous; every transition is made by the dispatch thread at a layer boundary. No
     * mutex on the read side (it is on every layer's critical path) and no background thread,
     * which would make placement depend on host timing. */
    struct Move {
        enum class State { Free, Copying, Published, Quarantine };
        /* Promote/Demote: the heat engine's swaps. Stage: the exact prefetch copying an offloaded
         * unit into a slot; Release: that slot returned after its last use this step. Drop: a
         * unit in a lent slot back to its kept host copy. Release and Drop copy nothing but still
         * need the quarantine, which guards issued kernels, not bytes in flight. Migrate: a donor
         * (Mover::donor_slot) from an uncovered lent slot (`slab_slot`) to a free budgeted one
         * (`to_slot`). */
        enum class Kind { Promote, Demote, Stage, Release, Drop, Migrate, Admit, Evict, Deposit };
        State    state = State::Free;
        Kind     kind = Kind::Promote;
        int32_t  unit = -1;
        int64_t  slab_slot = -1;
        int64_t  pool_slot = -1;
        int64_t  to_slot = -1;
        RadEvent ev = nullptr;

        void clear() {
            state = State::Free;
            unit = -1;
            slab_slot = pool_slot = to_slot = -1;
        }
        /* Whether the move hands `slab_slot` back once its quarantine clears. An Admit keeps
         * its slot: the unit becomes the slot's resident. */
        bool frees_slab() const { return kind != Kind::Promote && kind != Kind::Stage &&
                                          kind != Kind::Admit && kind != Kind::Deposit; }
    };

    struct UnitState {
        Tier    tier = Tier::VRAM;
        Site    site = Site::Device;
        int64_t slab_slot = -1;      /* index into class slab, or -1 */
        int64_t pool_slot = -1;      /* index into class host pool, or -1 */
        int64_t static_off = -1;     /* byte offset into the static VRAM region, or -1 */
        int32_t klass = -1;
        int64_t bytes = 0;
        /* Admitted into the hot VRAM cache: read from the container into a
         * slab slot and left there. Eviction is bookkeeping (kind Evict): re-tier to SSD, free
         * the slot, publish null — the container is ground truth. */
        bool    hot = false;
    };

    /* One weight class's memory. Budgeted slots sit at a fixed stride from one base in the
     * weights pool. Flex slots after them are lent addresses scattered over another pool's memory
     * (KV cache stripes, arena top), so only slot_ptr() can resolve a slot index. Each lender's
     * flex slots form one segment ordered by covering loan, so any loan covers a prefix. */
    struct ClassPools {
        SlotPool slab, pool;
        void*    slab_base = nullptr;
        void*    pool_base = nullptr;
        void*    pool_dev = nullptr;   /* the pinned pool as the device addresses it */
        int64_t  stride = 0;
        int64_t  n_fixed = 0;          /* slots in the budgeted range; the flex ones start here */
        int64_t  n_work = 0;           /* shadow + staging: free slots the mover needs to move at all */
        int64_t  n_turn = 0;           /* lent slots kept free past those: the fill stops short
                                        * of them and balance() drops lent units to rebuild them
                                        * (MoverConfig::turnover_slots) */
        int64_t  n_donated = 0;        /* units in lent slots that gave their host copy away
                                        * (Mover::donor_slot); at most n_turn of them */
        std::vector<char*>   flex_ptr;
        /* One slab segment per lender that carved slots for this class. `need[i]` is the loan,
         * in the lender's measure, that covers the segment's i-th slot; ascending. */
        struct Lent { int lender = 0; int seg = -1; int64_t first = 0; std::vector<int64_t> need; };
        std::vector<Lent>    lent;
        /* Units can sit past the loan only after a shrink, since nothing is placed in an
         * uncovered slot. Set on a shrink; cleared by a balance() sweep that removed them all. */
        bool     strand_check = true;

        char* slot_ptr(int64_t slot) const {
            return slot < n_fixed ? (char*)slab_base + slot * stride
                                  : flex_ptr[(size_t)(slot - n_fixed)];
        }
        bool is_flex(int64_t slot) const { return slot >= n_fixed; }
    };

    const Program* prog_ = nullptr;
    const Plan*    plan_ = nullptr;
    MoverConfig    cfg_;
    Residency      res_;
    MoverCounters  stats_;
    /* The heat engine of the dispatch in flight, bound by begin_dispatch: the file stager's
     * admission test (which units read into the hot cache) runs between dispatches on this
     * same thread. Null when heat is off — no hot cache, no evictable victims. */
    HeatEngine*    heat_ = nullptr;
    Pools*         budget_ = nullptr;
    /* True only when this object called the allocator itself (the test path). A Pool
     * reservation belongs to the Pool; freeing it here would be a double free. */
    bool           owns_memory_ = false;

    RadStream mover_ = nullptr;
    RadStream compute_ = nullptr;
    RadEvent  gate_ = nullptr;          /* recorded on compute, waited on by the mover stream */
    std::vector<RadCopySeg> stage_segs_;   /* stage_units' list, kept so a layer allocates nothing */
    struct StagePick { const char* src; int32_t u; int64_t bytes, need; char* dst; };
    std::vector<StagePick>  stage_picks_;  /* the units stage_units may take, in pool order */
    int       gate_dispatch_ = -1;

    /* A dispatch's swap copies (demotions, promotions, migrations) go to the card's DMA engine as
     * one list (flush_moves), sent at the end of balance() and before anything reads their
     * events. Each move's event is recorded after the list. */
    std::vector<RadCopySeg> move_segs_;
    std::vector<int32_t>    move_recs_;
    void queue_move_copy(void* dst, const void* src, int64_t bytes, int rec);
    int  flush_moves();
    std::vector<ClassPools> pools_;
    std::vector<UnitState>  unit_;
    std::vector<Move>       moves_;
    /* Per unit, how many records are Copying it, so unit_in_flight() needs no walk of the
     * records. The prefill stager asks about every pooled unit it stages, and the record count
     * grows with the move budget. */
    std::vector<uint16_t>   copying_;
    /* balance()'s drop candidates, (heat, unit), kept between calls so a step allocates nothing. */
    std::vector<std::pair<float, int32_t>> drop_cand_;
    bool hold_turnover_ = false;
    /* donor_slot()'s candidates per class, by layer, each layer sorted coldest first (hottest at
     * the back). Ranked once per dispatch; `at` is that dispatch. */
    struct DonorRank { int at = -1; std::vector<std::vector<std::pair<float, int32_t>>> by_layer; };
    std::vector<DonorRank> donors_;
    /* Per class, the layer of the unit each pool slot held at layout. The pool is filled in unit
     * order, so a layer's pooled units start out adjacent (pool_slot_for). */
    std::vector<std::vector<int32_t>> pool_layer_;
    int32_t n_layers_ = 0;

    void*   static_vram_ = nullptr;
    int64_t static_bytes_ = 0;
    void*   pageable_ = nullptr;
    int64_t pageable_bytes_ = 0;

    FileTier file_;
    FileTier draft_file_;
    bool source_file_epoch() const;
    int read_source_file(uint32_t bank,uint64_t offset,int64_t bytes,void*dst,int64_t cap);
    void*    stage_ = nullptr;          /* pinned staging ring for file-tier reads */
    int64_t  stage_stride_ = 0;
    int      stage_next_ = 0;

    /* The file stager's two VRAM buffers, one allocation, and the pinned windows it reads
     * through, each with the event behind the copies out of it. */
    char*    file_vram_ = nullptr;
    int64_t  file_buf_bytes_ = 0;
    char*    lend_slab_ = nullptr;        /* the lent slab's slots (Plan::lend_slab_vram), or null */
    int64_t  lend_slab_bytes_ = 0;
    /* The file tier's relayout (set_file_relayout): a host input per thread, which holds the
     * planes' aligned span as read. */
    struct RelayIn {
        uint8_t* p = nullptr;
        int64_t  cap = 0;
        RelayIn() = default;
        RelayIn(RelayIn&& o) noexcept : p(o.p), cap(o.cap) { o.p = nullptr; o.cap = 0; }
        RelayIn(const RelayIn&) = delete;
        ~RelayIn() { std::free(p); }
    };
    struct Relaid { const WeightInfo* w; char* dst; M1PrefetchTrace trace{}; };
    FileRelayout         relayout_;
    NativeFileCache      native_file_cache_;
    //Allocator context must outlive every window lease. Destruction drains explicitly first.
    NativeMappedFileCache mapped_file_cache_;
    bool mapped_file_cache_on_ = false;
    bool mapped_direct_fill_on_ = false; // captured at init; only slab/Normal/routed leases
    static constexpr size_t kMappedPerWindow = size_t(RAD_PLACE_FILE_WINDOW_BYTES) / NativeMappedFileCache::DN;
    struct MappedWindow {
        std::array<NativeMappedFileCache::Lease,kMappedPerWindow> leases;
        bool copy_may_be_queued = false;
    };
    std::array<MappedWindow,RAD_PLACE_FILE_WINDOWS> mapped_windows_;
    struct MappedCopyHooks {
        void* context = nullptr;
        int (*copy)(void*,void*,const void*,int64_t,RadStream) = nullptr;
        int (*record)(void*,RadEvent,RadStream) = nullptr;
        int (*sync)(void*,RadEvent) = nullptr;
        int (*drain)(void*,RadStream) = nullptr;
    } mapped_copy_hooks_; //only the friend fixture can change these; default real device API
    /* A host test's packed RAM cache (tests/aux_stager_test.cpp, through its friend): set, borrow_host_unit
     * answers from it instead of the mapped cache. The engine never sets it. */
    struct BorrowHook {
        void* context = nullptr;
        bool (*borrow)(void*, int32_t, HostUnit*, bool) = nullptr;
    } borrow_hook_;
    static bool mapped_file_eligible(const WeightInfo& w);
    int mapped_file_drain();
    int retire_file_window(int window);
    int file_copy_event_record(RadEvent event);
    int file_copy_event_sync(RadEvent event);
    int file_copy_stream_sync();
    int mapped_copy_to_vram(void* dst,const void* src,int64_t bytes);
    bool                 native_file_cache_on_ = false;
    std::vector<RelayIn> relay_in_;
    int                  relay_threads_ = 1;
    NativeReadWorkers    relay_workers_;
    /* read_unit_planes' readers: kept, so a miss starts no threads and a mirror read finds its
     * thread's bounce buffer warm (RADIANCE_EXPERIMENTAL_DS41_READ_POOL=0: a thread a read, as
     * before). */
    NativeReadWorkers    plane_workers_;
    /* One read_unit_planes call: its reads, which the readers take in order (the mirror's first:
     * they take the longest), each read's outcome and when it began and landed (ns from t0, the
     * dispatch). The reader is operator(); a posted call (start_unit_planes) lives in plane_call_
     * until its finish. */
    struct PlaneCall {
        struct Read {
            uint32_t file = 0; int64_t off = 0, len = 0; uint8_t* dst = nullptr; int64_t cap = 0;
            const PlaneMirror::Ref* m = nullptr; uint64_t from = 0;   /* m: the mirror's bytes [from, from+len) */
            int rc = 0;
            int64_t start_ns = 0, end_ns = 0, copy_ns = 0;
            int unit = -1;   /* its unit's index in the call */
        };
        /* read_unit_planes' `landed`: each unit's reads left, and whether one failed. */
        PlaneLanded landed = nullptr;
        void* ctx = nullptr;
        std::unique_ptr<std::atomic<int>[]> left;
        std::unique_ptr<std::atomic<uint8_t>[]> failed;
        Mover* mover = nullptr;
        PlaneMirror* mirror = nullptr;
        std::vector<Read> reads;
        std::atomic<size_t> next{ 0 };
        int weights = 0, units = 0;
        int rest = 0, rest_mirror = 0;                   /* PlaneReadTiming::rest, rest_mirror */
        int64_t rest_bytes = 0, rest_mirror_bytes = 0;   /* ...and its bytes */
        bool posted = false;
        int64_t t0 = 0;        /* steady_clock ns: the dispatch */
        int64_t join_ns = 0;   /* posted: the caller came to wait, from t0 */
        void operator()(int);
    };
    PlaneCall plane_call_;
    bool      plane_posted_ = false;
    bool plane_reads(HostPlanes* const* units, int n, int rows, PlaneCall* c,
                     const PlanesIn* in = nullptr, bool rest_mirror = false);
    bool plane_reads_done(PlaneCall* c, bool readers_failed, PlaneReadTiming* timing);
    bool                 native_file_workers_on_ = false;
    int  read_relaid(const WeightInfo& w, void* dst, int64_t dst_cap, RelayIn* in);
    int  read_native_file(const WeightInfo& w, void* dst, int64_t dst_cap, RelayIn* in,
                          bool* cache_hit, NativeFileCacheReadPolicy policy = NativeFileCacheReadPolicy::Normal,
                          NativeMappedFileCache::Lease* mapped_lease = nullptr, M1PrefetchTrace trace = {});
    int  read_native_file_impl(const WeightInfo& w, void* dst, int64_t dst_cap, RelayIn* in,
                          bool* cache_hit, NativeFileCacheReadPolicy policy, NativeMappedFileCache::Lease* mapped_lease);
    bool prefetch_trace_slot(const M1PrefetchTrace& trace);
    std::atomic<uint64_t> prefetch_trace_lines_{0};
    /* read_relaid split, summed over the relay threads that run it at once: preads, relayouts. */
    std::atomic<uint64_t> relaid_read_ns_{0}, relaid_pack_ns_{0}, relaid_reads_{0};
    std::atomic<uint64_t> host_plane_read_ns_{0}, host_plane_reads_{0}, host_plane_inserts_{0};
    /* The packed cache (packed_cache()): each class's raw-round budget, and its outcomes. */
    int  packed_raw_[2] = { 0, 0 };
    std::unique_ptr<std::atomic<uint8_t>[]> packed_refused_;   /* by unit: refused once already */
    /* By unit: the ring slot a lent-ring resident is published at (set_ring_resident), or null. Atomic
     * because the read-ahead threads ask (ring_resident); only the issuing thread writes. */
    std::unique_ptr<std::atomic<char*>[]> ring_res_;
    /* A file-tier unit with no address of its own on the card: neither in a slab slot nor a ring
     * resident. What the file stager's reads, aliases and borrows take. */
    bool file_off_card(int32_t u, const UnitState& us) const {
        return us.tier == Tier::SSD && us.slab_slot < 0 && us.klass >= 0 && !ring_resident(u);
    }
    std::atomic<uint64_t> packed_inserts_{0}, packed_over_budget_{0}, packed_over_budget_units_{0},
                          packed_round_trip_failed_{0};
    int  insert_unit_packed(const HostPlanes& p, const ds41hx::PackCopy* copy);
    /* Weight k (handle h) of a unit's planes laid out in p, its buffer grown as needed: where
     * read_unit_planes reads it. False when it is not a plain pair of MXFP4 planes. */
    bool lay_out_weight(HostPlanes* p, int k, rad_weight h);
    /* Per unit: the mirror's copy of its gate_up's w3 codes (shard -1: none) and which of the
     * unit's two weights is the gate_up; the tail read from the mirror, 4 KiB-aligned. */
    std::vector<PlaneMirror::Ref> mirror_ref_;
    std::vector<int8_t>           mirror_k_;
    int64_t                       mirror_tail_ = 0;          /* passes of min_rows+ tokens */
    int64_t                       mirror_tail_decode_ = 0;   /* smaller passes */
    int  stage_relaid(const std::vector<Relaid>& rl, NativeFileCacheReadPolicy policy);
    char*    windows_ = nullptr;
    int      n_windows_ = 0;
    char*    calm_ = nullptr;      /* zeroed: dummy table entries read zeros, not NaN */
    bool     calm_owned_ = false;  /* ours: calm_ allocated outside the budget (Mover::init) */
    int      win_next_ = 0;
    std::vector<RadEvent> win_ev_;
    std::vector<uint8_t>  win_rec_;

    /* The lead gate's ring: one event a dispatch, and a host wait on the one from
     * max_lead_dispatches ago. */
    std::vector<RadEvent> lead_;
    int                   lead_next_ = 0;
    int32_t               prefetch_cursor_ = 0;
    int32_t               last_op_ = -1;
    int                   dispatch_ = 0;

    /* Each lender's loan in its own measure, and the totals the dashboard reads. */
    int64_t              loan_[kLenders] = {};
    std::atomic<int64_t> flex_used_{0};
    std::atomic<int64_t> flex_offered_{0};
    /* Where the last free record was found. Records free up roughly in the order they were
     * taken, so starting there keeps the search short when lent slots put many in flight. */
    mutable size_t       rec_cursor_ = 0;

    int  retire(HeatEngine* heat);
    int  record_gate();
    void republish(int32_t unit);
    void* reserve(const char* what, int64_t bytes, int kind);
    int  free_move_record() const;
    int  start_stage(int32_t unit, int64_t slab_slot);
    void release_slot(int32_t unit);
    bool unit_in_flight(int32_t unit) const;
    /* Whether stage_units would copy `unit`: pooled, with no slab slot and no copy in flight. */
    bool stageable(int32_t unit) const;
    /* A unit in a lent slot back to the host copy it kept. See Move::Kind::Drop. */
    int  drop(int32_t unit, HeatEngine* heat);
    /* The pool slot a demotion copies `unit` into: one its layer's units started out in, else a
     * same-layer donor's, else any free one, else any donor's; -1 when there is none. */
    int64_t pool_slot_for(int32_t unit, HeatEngine* heat);
    /* A pool slot for a budgeted victim when none is free: the host copy of a unit in a KV-lent
     * slot, which stays in VRAM without it. From a unit of `layer`, or of any layer when
     * negative; -1 when no unit may give one. Rationale at the definition. */
    int64_t donor_slot(size_t klass, int32_t layer, HeatEngine* heat);
    /* A donor out of a lent slot into a free budgeted one, if one is free: false when none is. */
    bool migrate(int32_t unit, HeatEngine* heat);
    /* One promotion out of the unit's pool slot into `slot`, which the caller has taken. */
    int  promote_to(int32_t unit, int64_t slot);
    /* Promote the hottest pooled units of one class into up to `room` free slots. */
    int  fill(size_t klass, int64_t room, HeatEngine* heat);
    void count_flex();
};

}  /* namespace rad */
