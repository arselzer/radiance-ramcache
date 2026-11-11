/* card_expert_stager.h -- prefill Phase 1: every routed expert of a large prefill chunk computed on
 * the cards (tools/ds41-host-experts/ds41_p1.h is the contract with libr4d's twin row; the plan is
 * PHASE1-PLAN.md in the review directory).
 *
 * ---- what it does ------------------------------------------------------------------------------
 *
 * Under RADIANCE_EXPERIMENTAL_DS41_P1=1 beside RADIANCE_EXPERIMENTAL_DS41_LCP=<C>, a routed layer of a
 * pure prefill pass that runs on at least the threshold's rows (RADIANCE_EXPERIMENTAL_DS41_P1_MIN_ROWS,
 * 320; ds41p1::card_layer) gives no routed expert to the CPU; a shorter one takes the CPU path exactly
 * as without the switch. At each such layer the file stager (stager.h) learns the layer's counts as
 * it always does, then:
 *   - VRAM residents run in the plugin's own three expert ops (gate_up, the clamped SwiGLU, down);
 *   - every touched expert that is not resident is published to those ops with kMarkSkip, so the P1
 *     twin leaves its rows alone (T3);
 *   - each such expert is copied into a slot of a VRAM ring on the card's DMA lane: a RAM-cache unit
 *     straight from its pinned slab slot (borrowed read-only, packed or block form), a unit the plan
 *     pinned in the host pool straight from its pool slot (block form; the card would otherwise read
 *     it across the link once a row pass), a drive miss after it was read into host planes and
 *     packed by a packer thread into a pinned ready slot;
 *   - right after the layer's down projection was issued, the stager launches the same three ops
 *     again, in waves of a few ring units, each with its own expert table: the wave's units name
 *     their ring slots (kMarkPacked where packed) and every other entry carries kMarkSkip. The same
 *     kernels and arguments as the plugin's launch, so the rows are the decode band's bits (T1).
 * The SwiGLU relaunch normally runs over all of the layer's slots each wave: the rows already
 * computed come out the same from their unchanged gate_up rows. The default-off SWIGLU_SPAN
 * switch narrows the known row-elementwise, stateless SwiGLU to the covering sorted-slot span of
 * the wave's experts. Gaps still recompute from unchanged gate_up rows; rows outside stay untouched.
 *
 * ---- why after the down projection, not at the gather ------------------------------------------
 *
 * The arena may give the gate_up and SwiGLU outputs' memory to later ops once their last reader (the
 * down projection) is issued. Launching the waves right behind it, before any later op is issued,
 * keeps them inside those buffers' lifetimes on the same stream.
 *
 * ---- the ring ------------------------------------------------------------------------------------
 *
 * The file stager's two layer buffers seen as one region of whole ds41p1::kSlotBytes slots (sized by
 * RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS). Units take slots in the order their copies are
 * queued, slot = (units queued since bring-up) mod n, and waves launch the oldest queued units first,
 * so a slot's previous occupant is always in an earlier wave. A copy is queued only while fewer than
 * n units are queued and not yet launched, and it waits (on the mover's stream) on the slot's free
 * event, recorded on the compute stream behind the down projection of the wave that read the previous
 * occupant. A wave waits (on the compute stream) on the copy events of its units. A unit's source (a
 * RAM-cache lease, a pinned ready slot) is let go only once the host has seen its copy's event pass.
 *
 * ---- testable without a card ---------------------------------------------------------------------
 *
 * CardExpertStager is the scheduling alone, against the Device interface: tests/p1_stager_test.cpp
 * runs it on a simulated pair of streams with random completion, which checks every invariant the
 * plan names (one wave a unit, lapping, release after the copy). CardExpertPath is the engine's side:
 * the real Device (events, the mover's copy lists, the captured launches), the drive reads, the
 * packer and the read-ahead.
 */
#pragma once
#include "../../tools/ds41-host-experts/ds41_p1.h"   /* the P1 contract (also on rad_core's include path) */
#include "mover.h"
#include "../sched/p1_chunk.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace rad {
namespace p1 {

/* ---- tunables ----------------------------------------------------------------------------------- */
inline constexpr const char* kSwigluSpanEnv = "RADIANCE_EXPERIMENTAL_DS41_P1_SWIGLU_SPAN";
struct Tunables {
    int  wave = ds41p1::kDefWave;
    int  inflight = ds41p1::kDefInflight;
    int  ready = ds41p1::kDefReady;
    int  ahead = ds41p1::kDefAhead;
    int  verify = 0;
    int  min_rows = ds41p1::kDefMinRows;
    int  lend = 0;
    int  lend_arena = 0;
    int  grace_ms = ds41p1::kDefGraceMs;
    int  lend_heat = 0;                                         /* 0 off; else retargets a rank a pass */
    int  lend_heat_half = ds41p1::kDefLendHeatHalfRows;         /* the heat's half-life, in decode rows */
    int  lend_heat_margin = ds41p1::kDefLendHeatMarginPct;      /* the hysteresis, percent of the resident's heat */
    int  lend_heat_prior = -1;                                  /* the prior's weight in decode rows (-1: the half-life) */
    int  lend_slab = 0;                                         /* the lent slab's slots a rank (0 off) */
    bool lend_audit = false;                                    /* every lent slot read back and compared */
    int  lend_rest_ms = 0;                                      /* the rest after a long prefill (0 none) */
    bool align_end = true;
    bool swiglu_span = false;
};
/* The tunables from `get` (getenv for the engine): each a whole number in its range (wave 1-16,
 * inflight 1-64, ready 1-32, ahead 0-64, verify 0-4096, min_rows 17-4096, lend 0-64, lend_arena 0-64,
 * grace_ms 0-60000, lend_heat 0-8, lend_heat_half 1-100000000 (unset: 100, or 2000 with lend_slab),
 * lend_heat_margin 100-1000, lend_heat_prior 0-100000000 (unset: the half-life), lend_slab 0-256, lend_rest_ms 0-600000 (unset: 5000 with lend_heat or lend_slab, else
 * 0)), align "end" or "start", swiglu_span and lend_audit 0/off or
 * 1/on (default off). False with the reason on anything else. */
bool parse_tunables(const std::function<const char*(const char*)>& get, Tunables* t, std::string* why);

/* The wave's live sorted slots, from ALL local experts' synchronized counts (their exclusive
 * prefix is scatter's expert_offset). Empty live waves return [0,0); invalid counts or mismatched
 * skip marks refuse the slice. No resident/cache-size assumption is involved. */
struct SwigluSpan { int64_t lo = 0, hi = 0; };
bool swiglu_span_launch_ok(const RadKernelInfo* row, RadLaunchFn launch);
bool wave_swiglu_span(const int32_t* counts, int n, const void* const* gu, const void* const* dn,
                     int64_t rows, SwigluSpan* out);
/* A view for the known stateless swiglu_clamped_bf16 row: retain pitches, columns, n and limit;
 * rebase both bf16 pointers by lo rows and reduce shape[0] and M. The caller skips an empty view.
 * Unsupported layouts leave the outputs alone and take the original full replay instead. */
bool swiglu_span_view(const RadArgs* full, SwigluSpan span, std::vector<RadTensor>* tensors,
                      std::vector<RadParam>* params, RadArgs* out);

/* ---- the scheduling -------------------------------------------------------------------------------- */

/* Where a unit's bytes are for its ring copy: the gate_up and down weights, packed (the packed RAM
 * cache's form) or block form. */
struct Source {
    const void* gu = nullptr;
    int64_t     gu_bytes = 0;
    const void* dn = nullptr;
    int64_t     dn_bytes = 0;
    bool        packed = false;
};
/* Ram: a RAM-cache lease (let go after its copy); Drive: a pinned ready slot (likewise); Pool: a unit
 * the plan pinned in the host pool, read in place (nothing to let go). */
enum class Kind : uint8_t { Ram = 0, Drive = 1, Pool = 2 };

/* The outside world of CardExpertStager. Events are small ids the device hands out. */
class Device {
public:
    virtual ~Device() = default;
    /* A new event: `host` true when the host queries or syncs it (copy events). Its id, or < 0. */
    virtual int  event_create(bool host) = 0;
    /* One unit's copy into ring slot `slot` on the DMA lane: behind event `wait` (-1: none), then
     * gate_up to slot + 0 and down to slot + ds41p1::kDnOff, then `done` recorded. */
    virtual int  copy(int slot, const Source& src, int wait, int done) = 0;
    virtual int  compute_wait(int ev) = 0;
    virtual int  compute_record(int ev) = 0;
    virtual int  query(int ev) = 0;   /* 1 passed, 0 not yet, < 0 error */
    virtual int  sync(int ev) = 0;
    /* One wave on the compute stream: the tables uploaded to row (parity, w), then gate_up, the
     * SwiGLU and down launched again with them. */
    virtual int  wave(int parity, int w, const void* const* gu_tab, const void* const* dn_tab, int n) = 0;
    /* A unit's source may go: its RAM lease, or its pinned ready slot. */
    virtual void release(Kind k, int32_t handle) = 0;
    /* The issuing thread waits for drive units packed for the ring, and hands each to drive_ready()
     * before it returns. 1 when at least one came, < 0 on an error. */
    virtual int  wait_drive() = 0;
};

class CardExpertStager {
public:
    struct Stats {
        uint64_t layers = 0, ram = 0, drive = 0, pool = 0, waves = 0, copies = 0, copy_bytes = 0;
        uint64_t drive_waits = 0, ring_full_waits = 0, release_syncs = 0, max_waves = 0;
    };

    /* `ring_base`: the ring's first slot (n_slots of ds41p1::kSlotBytes); `alias_base`: the address
     * under kMarkSkip for every entry a wave does not compute; `ready_slots`: the pinned slots drive
     * units are packed into, so a full set of them is drained before more RAM units go. */
    int  init(Device* dev, const char* ring_base, int n_slots, int wave, int n_local, const void* alias_base,
              int ready_slots, std::string* why);
    int  n_slots() const { return n_; }

    /* One routed layer of an armed pass: its units, then start (the RAM units' copies, as far as the
     * ring allows), then run (every wave). begin_layer refuses while an earlier layer is unfinished. */
    int  begin_layer(int parity);
    int  add_ram(int e, int32_t unit, int rows, const Source& s, int32_t handle);
    /* A unit pinned in the host pool: copied like a RAM unit, in the same descending-rows order, with
     * no source to let go. */
    int  add_pool(int e, int32_t unit, int rows, const Source& s);
    int  add_drive(int e, int32_t unit, int rows);
    /* A drive unit of this layer is packed in a pinned ready slot (`handle`). */
    int  drive_ready(int32_t unit, const Source& s, int32_t handle);
    int  start();
    int  run();
    bool layer_open() const { return open_; }
    int  layer_units() const { return (int)u_.size(); }
    /* The pass is over: every copy's event seen, every source let go. Refuses (after releasing all)
     * if a layer was left unfinished. */
    int  end_pass();

    const Stats& stats() const { return st_; }
    /* Sources held now (leases and ready slots), for the stats line: 0 after end_pass. */
    int  held() const;

    /* The ring slot a table entry names, or -1 (tests). */
    int  slot_of(const void* entry) const;

private:
    enum class St : uint8_t { Waiting, Ready, Queued, Launched };
    struct U {
        int32_t e = -1, unit = -1, rows = 0, handle = -1;
        Kind    kind = Kind::Ram;
        St      st = St::Waiting;
        Source  src;
        int     slot = -1;
    };
    int  add_ready(Kind k, int e, int32_t unit, int rows, const Source& s, int32_t handle);
    struct Slot {
        int     copy_ev = -1, free_ev = -1;
        bool    free_rec = false;
        bool    held = false;          /* a source not yet let go */
        Kind    kind = Kind::Ram;
        int32_t handle = -1;
    };
    Device*     dev_ = nullptr;
    const char* ring_ = nullptr;
    const void* alias_ = nullptr;
    int         n_ = 0, wave_ = 2, n_local_ = 96, ready_slots_ = 1;
    uint64_t    head_ = 0;             /* units queued since init */
    std::vector<Slot> slot_;
    std::vector<U>    u_;
    std::vector<int>  ram_order_;      /* RAM and pool units by rows, descending */
    size_t            ram_next_ = 0;
    std::deque<int>   drive_q_;        /* drive units ready, in landing order */
    std::deque<int>   queued_;         /* copied, not yet launched, oldest first */
    int   launched_ = 0, waiting_drive_ = 0;
    int   parity_ = 0, wave_no_ = 0;
    bool  open_ = false, started_ = false;
    std::vector<const void*> tab_gu_, tab_dn_;
    Stats st_;

    int  queue_one(int i);
    int  pump();
    int  launch_wave(int n);
    int  let_go(int s, bool wait);
    void poll();
};

/* ---- the lent ring --------------------------------------------------------------------------------
 *
 * Under RADIANCE_EXPERIMENTAL_DS41_P1_LEND=K the ring's slots past the file stager's own two (the ones
 * its aliases name: B1's share) hold K file-tier units as decode residents between the card path's
 * passes: the hottest by the profile, each in a slot of its own, published there once its bytes are
 * in (Mover::set_ring_resident) and from then on computed by the plugin's own ops like any VRAM
 * resident, never aliased or handed to the CPU. A pass that arms the card path takes the slots back
 * before the run phase brings its weight tables up to date: each resident is published as a file-tier
 * unit again, which the card path or the CPU path then takes as it does any other, and the pass's
 * first ring copy waits on the residents' last reader (an event behind the pass before it). Once
 * RADIANCE_EXPERIMENTAL_DS41_P1_GRACE_MS have passed since the card path's last pass, the units are
 * reloaded in the background, so back-to-back prompts do not reload between them: from the RAM cache
 * where it holds the card's block form; where the RAM cache is packed and holds the unit, its lease
 * relaid from the packed form (the same planes, so the same bytes) into one of two pinned buffers by a
 * reload thread; else read from the drive and relaid into such a buffer; then copied into their slots on the DMA lane, behind the card path's
 * last ring reader. Each is published again at a later pass's start, once the host has seen its copy
 * pass. Until then it is non-resident and the CPU path takes it; a reload that fails leaves it there
 * until the next return. A copy begun before the slots were taken back runs before the card path's
 * own copies (one DMA lane) and is never published. Residency moves, bytes do not: every unit is
 * computed from the same bytes wherever it is, so replies stay byte-identical.
 *
 * The lent arena (RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA=K): the same, with up to K slots carved from
 * the activation arena's top (arena_slots), above the buffers of the arena's smallest level (steps of up
 * to 256 rows, the core's arena levels, Engine::probe_arena_levels) and its guard. The arena is sized
 * for the chunk, so a step at the full plan (a card-path pass, or any step above that level's rows)
 * writes there: such a step takes every lent slot back as a card-path pass does (FileStager::begin_pass,
 * before its first op and its weight tables). Its ops run on the compute stream behind the residents'
 * readers (the pass before it, lane 1 joined into lane 0 at its end), but a reload copy still in flight
 * runs on the DMA lane: the compute stream waits on it first (LendHost::fence). And since its arena
 * writes are not behind the DMA lane, the mover's stream waits on the end of every such step (as on an
 * armed pass's end), so a later reload copy lands only after the step's last reader of the arena.
 *
 * The lent slots' heat (RADIANCE_EXPERIMENTAL_DS41_P1_LEND_HEAT=S, default 0: off, the slots keep the units
 * the profile gave them): which units hold the lent slots follows live decode routing. LendHeat keeps, by
 * plan unit of the rank, a decayed count of the decode rows routed to it (FileStager counts each routed
 * trunk layer of a pure decode pass where it learns the layer's counts). Between passes, at the begin of a
 * pass that does not take the slots back and only while the lender is settled (every slot holds its
 * resident: nothing wanted, in flight, landed, copying, cancelled or failed, the slots not lent), up to S
 * of the coldest residents give their slots to the hottest lendable units off the card whose heat is above
 * the margin times the resident's plus half a row (RingLender::retarget):
 *   1. the DMA lane waits on an event recorded on the compute stream at the end of the pass before this
 *      one (LendHost::retarget_fence; the residents' last-reader event that an armed pass's first ring
 *      copy waits on, by the same Mover::mover_wait). That pass's end is behind every kernel that could
 *      read the residents from their slots, and the lane is one queue, so every copy queued from here on
 *      runs behind them: the retargeted slot's own copy, queued passes later when its load has landed, too;
 *   2. the resident is published as a file-tier unit again, as a pass that takes the slots back does it
 *      (LendHost::publish(k, false)), before the run phase brings this pass's weight tables up to date: no
 *      kernel issued from here on reads the slot until its next unit is published there;
 *   3. the slot is the new unit's (LendHost::retarget) and wants a reload: the existing path loads it,
 *      copies it into the slot on the DMA lane and publishes it at a later pass's begin once the host has
 *      seen the copy pass. Until then both units take the CPU path.
 * A pass that takes the slots back in the middle of this finds the slot wanted, reading, landed or copying
 * and treats it as any other (cancelled, a queued copy fenced), and the slot reloads its new unit after the
 * grace. Nothing else changes: which experts a row routes to and every unit's bytes are as without the
 * switch, a lent resident is computed by the plugin's own ops wherever it sits.
 *
 * The lent slab (RADIANCE_EXPERIMENTAL_DS41_P1_LEND_SLAB=N, default 0: off): N slots of VRAM of their own,
 * which the planner set aside by giving up static residents (Plan::lend_slab_vram) and which hold the N
 * hottest file-tier units of that plan (Plan::lend_slab_units). No pass writes there, so these slots are
 * kept: a pass that takes the ring's and the arena's slots back leaves them, their residents stay published
 * through it (the plugin's ops read them at any height: block form, unmarked) and they know no grace. Their
 * units are loaded at bring-up, before the first pass (RingLender::prime), by the path every reload takes.
 * A kept slot's reload begins only while the other slots are not lent out, so an armed pass's drive reads
 * and ring copies have the lane and the drive to themselves; one begun before goes on. With the heat they
 * change units as the other slots do, by the same fence and only while every slot is settled.
 *
 * The read-back audit (RADIANCE_EXPERIMENTAL_DS41_P1_LEND_AUDIT=1, default off: nothing of it exists): the
 * card's copy of every lent unit is read back and compared, twice a residency.
 *   at publish     once the host has seen a slot's copy pass and before its unit is published, the slot is
 *                  copied back to pinned host memory on the DMA lane: behind the copy (the host saw its
 *                  event), with no reader (unpublished) and no writer (the lender queues the slot's only
 *                  copies, and a pass that writes the arena's top takes the slots back first) until the
 *                  host has compared it byte for byte with the bytes that were copied, which the lender
 *                  still holds. One 64-bit digest a 4 KiB page of what was read is kept (LendAudit).
 *   at withdrawal  before a resident is published as a file-tier unit again (a pass that takes the slots
 *                  back, or a retarget), the slot is copied back the same way behind the event the compute
 *                  stream recorded at the end of the pass before: the residents' last reader, the wait an
 *                  armed pass's first ring copy and a retarget's fence make. The host waits for it before
 *                  it withdraws, so before the pass issues an op or the lender queues the slot's next
 *                  copy; its digests are compared with the ones kept. At shutdown every slot that still
 *                  holds a resident is read back the same way (RingLender::audit_residents), so a
 *                  residency no pass ended, a slab slot's above all, is closed by a compare too.
 * A slot that differs, or that cannot be read back, is lost: the host logs rank, slot, unit and the first
 * differing offset, the unit is not published (or is withdrawn) and takes the CPU path, nothing is loaded
 * into the slot again, and the rank's heat policy makes no further retarget (RingLender::stopped,
 * Decline::Audit). Everything else goes on, audited and counted. The read-backs are the issuing thread's,
 * between passes: each costs a copy of the slot off the card and a pass over its bytes, so an audited run
 * is not a timed one.
 *
 * RingLender is the state machine alone, against LendHost (tests/p1_stager_test.cpp drives it with
 * random landings, copy completions and failures); CardExpertPath is the engine's LendHost. */
class LendHost {
public:
    virtual ~LendHost() = default;
    /* Lent slot k's unit read for its slot in the background: 1 begun, 0 not now (every reload
     * buffer busy), < 0 not possible (the unit stays on the CPU path). */
    virtual int  load(int k) = 0;
    /* load(k) only where it reads no drive (the RAM cache holds the unit): 1 begun, 0 not now (not in the
     * RAM cache, or every reload buffer busy), < 0 not possible. While the lender holds its drive reads
     * back (RingLender::set_rest, after a pass that took the slots) it begins loads only so. */
    virtual int  load_ram(int k) {
        (void)k;
        return 0;
    }
    /* A begun load that landed since the last call: its slot (and *ok whether it read), or -1. */
    virtual int  landed(bool* ok) = 0;
    /* Slot k's landed bytes copied into its slot on the DMA lane, an event recorded after. */
    virtual int  copy(int k) = 0;
    /* Slot k's copy: 1 passed, 0 not yet, < 0 an error. */
    virtual int  copied(int k) = 0;
    /* Slot k's landed bytes may go (its buffer back to the reload thread, or its RAM-cache lease). */
    virtual void drop(int k) = 0;
    /* The rest (RingLender::set_rest): on, a load already begun that reads the drive does not start its
     * read until it is off again (one the RAM cache gives goes on); off, it does. */
    virtual void hold_drive(bool on) { (void)on; }
    /* Slot k's unit published at its slot (on), or as a file-tier unit again (off). */
    virtual void publish(int k, bool on) = 0;
    /* A pass that takes the slots back found slot k's copy still queued: whatever writes the slot in
     * that pass must wait for it (an arena slot: the compute stream waits on its event; a ring slot
     * needs nothing, the card path's own copies run behind it on the DMA lane). */
    virtual void fence(int k) = 0;
    /* The heat policy is about to hand slot k, a resident's, to another unit while decode keeps the slots
     * (RingLender::retarget): the slot's next copy must not begin before the last kernel that could read
     * the resident's bytes there has finished. The host makes the DMA lane wait on an event recorded on the
     * compute stream behind the pass before this one, so every copy queued from here on runs behind that
     * pass. 0 done; < 0 it cannot (no such event, or the lane did not take the wait): nothing changes. */
    virtual int  retarget_fence(int k) = 0;
    /* Slot k's resident was withdrawn (publish(k, false)) for the unit the policy named: from here the
     * slot's load, copy and publish are that unit's. */
    virtual void retarget(int k) = 0;
    /* The read-back audit, called only with it on (RingLender::set_audit). publish: slot k's copy has passed
     * and its landed bytes are still held (before drop): the slot is read back and compared with them, and
     * what it holds is remembered. Else slot k's resident is about to be withdrawn (before publish(k,
     * false)): the slot is read back behind the residents' last reader and compared with what was
     * remembered. 0 equal; 1 they differ (the host says where); < 0 the slot could not be read back or
     * compared. */
    virtual int  audit(int k, bool publish) {
        (void)k;
        (void)publish;
        return 0;
    }
};

class RingLender {
public:
    /* Why the heat policy did nothing on a pass (stats().declined): the slots are lent or in the grace (a
     * pass took them back, this one among them); a slot is not its resident's (wanted, reading, landed,
     * copying, or failed until the next return); a draft bank's loan holds ring residents on this rank; a
     * chosen unit can no longer hold a slot; no event behind the pass before this one, or the DMA lane
     * refused the wait; the read-back audit found a slot differing and stopped the policy on this rank. */
    enum class Decline : uint8_t { Lent = 0, Busy = 1, Bank = 2, Candidate = 3, Fence = 4, Audit = 5, Prefill = 6 };
    static constexpr int kDeclines = 7;
    struct Stats {
        uint64_t lends = 0;        /* card-path bursts that took the slots back */
        uint64_t withdrawn = 0;    /* residents published as file-tier units again for them */
        uint64_t returns = 0;      /* reload rounds begun after a grace */
        uint64_t reloaded = 0;     /* units published at their slots */
        uint64_t failed = 0;       /* loads or copies that failed: their units left on the CPU path */
        uint64_t cancelled = 0;    /* loads or copies a card-path pass overtook, never published */
        uint64_t held_back = 0;    /* passes that found the grace still running */
        /* The heat policy's (retarget, decline; all 0 without the switch): */
        uint64_t retarget_withdrawn = 0;   /* residents that gave their slots to another unit */
        uint64_t retargets = 0;            /* those units published at their slots */
        uint64_t declined[kDeclines] = {}; /* passes the policy did nothing on, by Decline */
        /* The read-back audit's (all 0 without the switch): */
        uint64_t audited = 0;        /* slots read back and compared, at publish and at withdrawal */
        uint64_t audit_differ = 0;   /* of them, slots whose bytes differed */
        uint64_t audit_unread = 0;   /* and slots that could not be read back or compared */
        uint64_t lost = 0;           /* slots given up for either: nothing is loaded into them again */
        /* The rest's (all 0 without it): passes at which a slot's reload waited for the drive, loads begun
         * from the RAM cache during it. */
        uint64_t rested = 0, rest_ram = 0;
    };
    /* n slots, the grace in ms; the last `keep` of them (0..ds41p1::kMaxLendSlab) are kept (the slab's:
     * never taken back), the others (at most 2 x ds41p1::kMaxLend) the ring's and the arena's. Every slot
     * that is not kept starts empty and lent, as after a card-path pass long ago: the first pass that does
     * not take the slots begins the reloads. A kept slot starts wanting its unit (prime). */
    int  init(LendHost* host, int n, int64_t grace_ms, int keep = 0);
    int  n() const { return (int)slot_.size(); }
    /* A pass begins at now_ms; p1: it takes the slots back (it arms the card path, or with the lent
     * arena runs at the full plan). With p1 every resident but a kept slot's is published as a file-tier
     * unit again and every load or copy in flight for such a slot is cancelled, each copy still queued
     * fenced first (LendHost::fence); returns 1 when some resident was (the pass's first ring copy must
     * wait on the residents' last reader). Without, once the grace has run out the reloads begin; landed
     * loads are copied, passed copies published, and further loads begun as buffers come free. Never
     * fails: a host call that fails leaves its unit on the CPU path. */
    int  begin_pass(bool p1, int64_t now_ms);
    /* Bring-up, before the first pass: the kept slots' loads worked on (landed ones copied, passed copies
     * published, further loads begun), whatever the other slots' state. Returns the kept slots still on
     * their way (wanted, reading, landed or copying); the caller calls again until 0, or stops and leaves
     * the rest to the passes. */
    int  prime();
    int  kept() const { return keep_; }
    bool keeps(int k) const { return k >= n() - keep_ && k < n(); }
    /* The read-back audit (above), set after init: on, every publish and every withdrawal is behind
     * LendHost::audit. stopped: it found a slot differing or unreadable, so no retarget happens on this
     * rank any more; lost(k): slot k is such a slot. */
    void set_audit(bool on) { audit_ = on; }
    /* The rest after a long prefill (ds41p1::kLendRestEnv), set after init: from a pass that takes the slots
     * back until ms after the last such pass ended (end_p1), no load that reads the drive begins (pump asks
     * LendHost::load_ram instead of load, and the host holds a drive read already queued, hold_drive), and
     * the heat policy declines (Decline::Prefill, heat_turn). 0: none, the lender as without it. resting:
     * now, as of the last begin_pass or end_p1. */
    void set_rest(int64_t ms) { rest_ = ms > 0 ? ms : 0; }
    int64_t rest_ms() const { return rest_; }
    bool resting() const { return rest_ > 0 && any_p1_ && (p1_now_ || now_ - last_p1_ < rest_); }
    bool auditing() const { return audit_; }
    bool stopped() const { return stopped_; }
    bool lost(int k) const { return k >= 0 && k < n() && slot_[(size_t)k].st == St::Lost; }
    /* At shutdown, with no pass in flight: every resident's slot read back and compared as at a withdrawal
     * (a kept slot's resident is never withdrawn by a pass, so this is where its residency is closed). One
     * that differs is withdrawn and lost as there. Nothing without the audit. Returns the slots compared. */
    int  audit_residents();
    /* A pass that took the slots back ended at now_ms: the grace runs from here. */
    void end_p1(int64_t now_ms);
    bool lent() const { return lent_; }
    int  residents() const;
    /* Slots holding a buffer or lease (a load in flight, landed, or a copy in flight). */
    int  busy() const;
    bool resident(int k) const { return k >= 0 && k < n() && slot_[(size_t)k].st == St::Resident; }
    const Stats& stats() const { return st_; }

    /* ---- the heat policy's part (the lent slots' heat, above) ----
     * settled: the slots are not lent and every one holds its resident: nothing wanted, in flight, landed,
     * copying, cancelled or failed. Only then may a slot change its unit. */
    bool settled() const;
    /* Between passes, after begin_pass(false, ..): the residents of slots ks[0..n) give them to other
     * units. Refused whole (0, counted in declined) unless settled(). For each slot in order: the host
     * fences it (retarget_fence; a refusal stops here, counted, the slots before it stay retargeted),
     * withdraws its resident (publish(k, false)) and names the slot's new unit (retarget); the slot then
     * wants a reload, begun at once where the host has a buffer and otherwise as buffers come free. A slot
     * named twice or out of range is skipped. Returns the slots retargeted. */
    int  retarget(const int* ks, int n);
    /* The policy did nothing this pass, for a reason only its caller knows (or the lender's own). */
    void decline(Decline why) { ++st_.declined[(size_t)why]; }

private:
    /* Idle: not resident and nothing wanted (lent, or nothing yet). Want: a reload wanted, not begun
     * (no buffer). Reading: begun; Landed: its bytes in; Copying: its copy queued. cancel: a reading
     * or copying slot a card-path pass overtook, which goes back to Idle or Want once it ends. moved: the
     * heat policy gave the slot another unit, not yet published there (the retargets count). Lost: the
     * read-back audit gave the slot up, for good. */
    enum class St : uint8_t { Idle, Want, Reading, Landed, Copying, Resident, Failed, Lost };
    struct Slot { St st = St::Idle; bool cancel = false; bool moved = false; bool keep = false; };
    LendHost* host_ = nullptr;
    std::vector<Slot> slot_;
    int64_t grace_ = 0, last_p1_ = 0, rest_ = 0, now_ = 0;
    bool    p1_now_ = false, held_ = false;   /* a pass that took the slots is running; the host holds drive reads */
    void hold(bool on);
    int  keep_ = 0;
    bool lent_ = true, any_p1_ = false;
    bool audit_ = false, stopped_ = false;
    Stats st_;
    /* Slot k read back and compared (LendHost::audit); false, counted, with the policy stopped when it
     * differs or could not be read. lose: the slot is given up. */
    bool audit_ok(int k, bool publish);
    void lose(int k);
    /* Slot s is lent out: the slots are, and it is not a kept one. */
    bool away(const Slot& s) const { return lent_ && !s.keep; }
    void settle(int k);
    void poll();
    void pump();
};

/* The lent slots' heat (above): a rank's decayed counts of decode rows by plan unit, and the policy's
 * choice. Pure and device-free (tests/p1_stager_test.cpp). The decay is a growing scale: a count is stored
 * times the scale of its pass and every pass multiplies the scale by 2^(rows / half), so heat(u) =
 * stored / scale halves every `half` decode rows with nothing swept (the stored values are divided back
 * once in ~500 half-lives). */
class LendHeat {
public:
    struct Pair { int slot = -1; int32_t unit = -1; };
    /* The hysteresis' constant: a candidate needs more than margin x the resident's heat plus this many
     * rows, so a unit never seen does not take a slot from another never seen, and equal heat never swaps. */
    static constexpr double kMinRows = 0.5;
    /* n_units plan units; at most max_pairs (1..ds41p1::kMaxLendHeat) retargets a pass; the half-life in
     * decode rows; the margin in percent (100 or more). < 0 on anything else. */
    int  init(size_t n_units, int max_pairs, int half_rows, int margin_pct);
    int  max_pairs() const { return max_; }
    /* A counted pass of `rows` decode rows begins: everything counted so far ages by them. */
    void begin_pass(int64_t rows);
    /* `rows` of the pass's rows were routed to plan unit u. */
    void add(int32_t u, int32_t rows) {
        if (u < 0 || (size_t)u >= h_.size() || rows <= 0) return;
        h_[(size_t)u] += (double)rows * scale_;
        dirty_ = true;
    }
    double heat(int32_t u) const { return u >= 0 && (size_t)u < h_.size() ? h_[(size_t)u] / scale_ : 0.0; }
    /* The prior (ds41p1::kLendHeatPriorEnv), before the first pass: unit u starts as if `rows` decode rows had
     * been routed to it; it ages as a count does. Neither warms the heat nor marks it counted. */
    void prior(int32_t u, double rows) {
        if (u < 0 || (size_t)u >= h_.size() || !(rows > 0.0)) return;
        h_[(size_t)u] += rows * scale_;
    }
    /* The prior from a profile: share[u] (by plan unit; a unit's share of its layer's decode selections, as
     * Planner gives MoveUnit::profile_share) over `rows` decode rows, each routing ds41p1::kExpertsPerRow
     * experts a layer: prior(u, rows x kExpertsPerRow x share[u]). Returns the units seeded. */
    int  seed(const std::vector<float>& share, double rows) {
        int n = 0;
        for (size_t u = 0; u < share.size() && u < h_.size(); ++u) {
            const double r = rows * (double)ds41p1::kExpertsPerRow * (double)share[u];
            if (!(r > 0.0)) continue;
            prior((int32_t)u, r);
            ++n;
        }
        return n;
    }
    /* One half-life of decode rows has been counted: before that the counts say too little to move a unit. */
    bool warm() const { return seen_ >= half_; }
    /* The half-life and the decode rows counted so far (the stats line's). */
    double half_rows() const { return half_; }
    double seen_rows() const { return seen_; }
    /* Something was counted since the policy last looked (looked); again: look once more without it. */
    bool dirty() const { return dirty_; }
    void looked() { dirty_ = false; }
    void again() { dirty_ = true; }
    /* The retargets wanted now. slot_unit[k]: the unit lent slot k holds (n_slots of them); cand: the
     * units a slot may hold; taken: by plan unit, nonzero for one that holds a slot now (never chosen).
     * Pairs the coldest slot with the hottest candidate, the next coldest with the next hottest, up to
     * max_pairs, each only while heat(candidate) > margin x heat(slot's unit) + kMinRows; ties go to the
     * lower slot and the earlier candidate. Returns the pairs written to out (room for max_pairs). */
    int  choose(const int32_t* slot_unit, int n_slots, const int32_t* cand, size_t n_cand, const uint8_t* taken,
                Pair* out) const;

private:
    std::vector<double> h_;        /* by plan unit: its counts, each times the scale of its pass */
    double scale_ = 1.0, half_ = (double)ds41p1::kDefLendHeatHalfRows, margin_ = 1.5, seen_ = 0.0;
    int    max_ = 0;
    bool   dirty_ = false;
};

/* The heat policy's turn between passes, right behind the lender's begin_pass: FileStager::p1_heat_policy is
 * this over the engine's parts, tests/p1_stager_test.cpp runs the same over the fakes. `took`: this pass
 * takes the slots back; `bank`: a draft bank's loan holds ring residents on the rank; `fence_ready`: the
 * residents' last-reader event is the record of the pass right before this one (so the DMA lane's wait on
 * it puts every later slot copy behind every reader of the residents). cand: the units a slot may hold (one
 * found ineligible leaves it); taken: by plan unit, nonzero for one that holds a slot (kept with the
 * retargets). The owner `o`: slot_unit(k), the unit lent slot k is for; eligible(u), whether u may be handed
 * a slot now; retarget(ks, units, n), which names the slots' new units to the host and calls
 * RingLender::retarget, returning the slots retargeted.
 * Nothing happens, counted by reason (RingLender::Decline), once the read-back audit has stopped the policy
 * on the rank, when the pass takes the slots back or they are
 * lent or in the grace, with a draft bank's loan, while any slot is not its resident's, without the fence's
 * event, or when a chosen unit is not eligible; and, uncounted, when nothing was counted since the last look
 * (the choice would be the same), before a half-life of rows has been counted, and when no unit off the card
 * is enough hotter than a resident. Returns the slots retargeted. */
template <class Owner>
int heat_turn(RingLender& l, LendHeat& h, bool took, bool bank, bool fence_ready, std::vector<int32_t>& cand,
              std::vector<uint8_t>& taken, Owner& o) {
    using Decline = RingLender::Decline;
    auto no = [&](Decline why) {
        l.decline(why);
        return 0;
    };
    if (l.stopped()) return no(Decline::Audit);
    if (l.rest_ms() > 0 && (took || l.resting())) return no(Decline::Prefill);
    if (took || l.lent()) return no(Decline::Lent);
    if (bank) return no(Decline::Bank);
    if (!l.settled()) return no(Decline::Busy);
    if (!h.dirty() || !h.warm()) return 0;
    h.looked();
    const int n = l.n();
    if (n < 1 || n > ds41p1::kMaxLendSlots) return 0;
    int32_t slot_unit[ds41p1::kMaxLendSlots];
    for (int k = 0; k < n; ++k) slot_unit[k] = o.slot_unit(k);
    LendHeat::Pair pair[ds41p1::kMaxLendHeat];
    const int np = h.choose(slot_unit, n, cand.data(), cand.size(), taken.data(), pair);
    if (np <= 0) return 0;
    if (!fence_ready) {
        h.again();
        return no(Decline::Fence);
    }
    int     ks[ds41p1::kMaxLendHeat];
    int32_t us[ds41p1::kMaxLendHeat];
    for (int i = 0; i < np; ++i) {
        if (!o.eligible(pair[i].unit)) {
            cand.erase(std::remove(cand.begin(), cand.end(), pair[i].unit), cand.end());
            h.again();
            return no(Decline::Candidate);
        }
        ks[i] = pair[i].slot;
        us[i] = pair[i].unit;
    }
    const int done = o.retarget(ks, us, np);
    for (int i = 0; i < np; ++i) {
        const int32_t was = slot_unit[ks[i]], now = o.slot_unit(ks[i]);
        if (now == was) continue;
        if (was >= 0 && (size_t)was < taken.size()) taken[(size_t)was] = 0;
        if (now >= 0 && (size_t)now < taken.size()) taken[(size_t)now] = 1;
    }
    if (done < np) h.again();
    return done;
}

/* The read-back audit's bytes (above), device-free (tests/p1_stager_test.cpp): the byte compare of a slot
 * read back against the bytes copied, and by lent slot the digests of what it was published holding, one a
 * 4 KiB page of each of the unit's two weights. A digest is h = (h ^ word) x an odd constant over the page's
 * 64-bit words, so a page that differs in one word always has another digest, and two pages that differ in
 * more have the same one with chance 2^-64. */
class LendAudit {
public:
    static constexpr int64_t kPage = 4096;
    /* The first offset at which a[0, n) and b[0, n) differ, or -1. */
    static int64_t first_diff(const uint8_t* a, const uint8_t* b, int64_t n);
    /* One digest a page of p[0, n) (the last may be short), appended to *out. */
    static void digest(const uint8_t* p, int64_t n, std::vector<uint64_t>* out);

    void init(int n_slots) { book_.assign((size_t)(n_slots > 0 ? n_slots : 0), Book{}); }
    /* Slot k was read back equal to what was copied: its two weights w[i][0, n[i]) as read. */
    void remember(int k, const uint8_t* const w[2], const int64_t n[2]);
    void forget(int k) {
        if (k >= 0 && (size_t)k < book_.size()) book_[(size_t)k].known = false;
    }
    bool known(int k) const { return k >= 0 && (size_t)k < book_.size() && book_[(size_t)k].known; }
    /* Slot k read back at withdrawal against what was remembered: 0 equal; 1 they differ, *which the
     * weight and *at the offset in it of the first page that does; < 0 nothing is remembered for the slot,
     * or not for weights of these sizes. */
    int  check(int k, const uint8_t* const w[2], const int64_t n[2], int* which, int64_t* at) const;

private:
    struct Book {
        bool    known = false;
        int64_t n[2] = { 0, 0 };
        std::vector<uint64_t> d;   /* weight 0's pages, then weight 1's */
    };
    std::vector<Book> book_;
};

/* ---- the engine's side ----------------------------------------------------------------------------
 *
 * One a rank, owned by its FileStager. The real Device: the ring's events, the mover's copy lists,
 * the wave tables (pinned rows, uploaded on the compute stream) and the captured launches. The drive
 * misses: a reader thread posts them to the mover's kept readers (Mover::read_unit_planes, at most
 * `inflight` units a call, with the pass's rows for the plane mirror's share), a packer thread packs
 * each landed unit the layer wants into a pinned ready slot (Mover::pack_unit_for_ring), and the
 * issuing thread copies it to the ring. Units read ahead for the next layer wait, read, for that
 * layer's choice. */
class CardExpertPath final : public Device, public LendHost {
public:
    using PlanesGet = std::function<std::shared_ptr<Mover::HostPlanes>(int32_t unit)>;
    using Insert = std::function<void(std::shared_ptr<Mover::HostPlanes> p, int rows)>;
    using LaunchFn = int (*)(const RadArgs*, RadStream);

    ~CardExpertPath() override;
    /* The ring from the mover's file region, the ready slots, the events, the tables and the two
     * threads. < 0 with *why when the path cannot run (the caller refuses P1 then). */
    int  init(Mover* mover, const Tunables& t, int rank, PlanesGet planes, Insert insert, std::string* why);
    bool on() const { return on_; }
    int  ring_slots() const { return st_.n_slots(); }
    const Tunables& tunables() const { return t_; }
    const void* alias_base() const { return alias_; }
    const char* ring_base() const { return ring_; }

    /* An armed pass of `rows` tokens begins; `hand_over` (null entries skipped): events behind the
     * last readers of the file buffers before it, which the first copy waits on. */
    int  begin_pass(int64_t rows, const RadEvent* hand_over, int n_hand_over);
    /* Every copy seen, every source let go, read-ahead dropped, the readers and the packer idle;
     * then `end` is recorded on `s` and the mover's stream waits on it, so a later pass's copies
     * into the file buffers run behind this pass's last reader. */
    int  end_pass(RadStream s, RadEvent end);
    bool pass_on() const { return pass_; }
    /* CPU-side proof only, after end_pass; the owner separately drains every device stream. */
    int quiesce_for_draft_bank();

    /* A routed layer: its units, then start (RAM copies, demand reads posted), then the launches of
     * its three ops captured, then run (the waves) behind the down projection on `s`. */
    int  begin_layer(int32_t li, int parity, const int32_t* counts = nullptr, int n_counts = 0);
    /* A cached unit, borrowed (touch: with the LRU touch an admitted unit gets): false when the RAM
     * cache does not hold it (the caller adds it as a drive miss then). */
    int  add_ram(int e, int32_t unit, int rows, bool touch, bool* hit);
    /* A unit the plan pinned in the host pool, from its pool slot as the card's DMA reads it (`src`,
     * block form; FileStager::p1_pool_source). */
    int  add_pool(int e, int32_t unit, int rows, const Source& src);
    int  add_drive(int e, int32_t unit, int rows, bool admit);
    int  start();
    /* Layer li + 1's likely misses, read after this layer's demand reads (up to `ahead`). */
    int  read_ahead(int32_t next_li, const std::vector<int32_t>& units);
    int  capture(int which, const RadArgs* a, LaunchFn fn, bool swiglu_span_ok = false);   /* 0 gate_up, 1 SwiGLU, 2 down */
    bool captured() const { return cap_[0].ok && cap_[1].ok && cap_[2].ok; }
    /* A captured launch's operands (0 gate_up, 2 down), for the host verify; null when not captured. */
    const RadTensor* captured_operands(int which, int* n) const {
        if (which < 0 || which > 2 || !cap_[which].ok) return nullptr;
        *n = (int)cap_[which].t.size();
        return cap_[which].t.data();
    }
    int  run(RadStream s);
    bool layer_open() const { return st_.layer_open(); }

    /* ---- the lent ring, the lent arena and the lent slab (RingLender's host) ----
     * lend_init: units[k] (its two weights' offsets and sizes in the unit's layout, as the plan lays it
     * out) lent ring slot ring_slots[k] for k below their count, then the arena slots arena_slots[k -
     * that count] (whole ds41p1::kSlotBytes slots of the activation arena's lent top, arena_slots), then
     * the slab's slots slab_slots[..] (whole slots of the mover's lent slab region, kept: never taken
     * back); the ring's and the arena's reloads begin at the first pass that does not take the slots back,
     * the slab's at lend_prime. < 0 with *why when it cannot run (the caller refuses the switch then). */
    struct LendUnit { int32_t unit = -1; int64_t off[2] = { 0, 0 }, bytes[2] = { 0, 0 }; };
    int  lend_init(const std::vector<LendUnit>& units, const std::vector<int>& ring_slots,
                   const std::vector<char*>& arena_slots, const std::vector<char*>& slab_slots, int64_t grace_ms,
                   bool audit, std::string* why);
    /* The read-back audit (`audit` at lend_init: one pinned slot of host memory and an event more): before
     * the lender's begin_pass, the event the compute stream recorded at the end of the pass before (null:
     * none yet), behind which a withdrawn resident's slot is read back. */
    void lend_audit_behind(RadEvent ev) { laud_behind_ = ev; }
    /* The rest after a long prefill (RingLender::set_rest), for the lender. */
    void lend_set_rest(int64_t ms) { lend_.set_rest(ms); }
    /* Bring-up: the slab's units loaded, copied and published before the first pass (RingLender::prime),
     * for at most `most_ms`; what is not in by then comes in at the passes. Returns the slab slots that
     * hold their unit. */
    int  lend_prime(int64_t most_ms);
    int  lend_slab_slots() const { return lend_.kept(); }
    bool lending() const { return lend_.n() > 0; }
    RingLender& lender() { return lend_; }
    const RingLender& lender() const { return lend_; }
    /* Lent slot k's ring slot, or -1 (an arena slot); its address; the arena's lent slots. */
    int  lend_ring_slot(int k) const { return k >= 0 && k < (int)lring_.size() ? lring_[(size_t)k] : -1; }
    char* lend_slot(int k) const { return k >= 0 && k < (int)lat_.size() ? lat_[(size_t)k] : nullptr; }
    int  lend_arena_slots() const { return (int)(lat_.size() - lring_.size()) - lend_.kept(); }
    int32_t lend_unit(int k) const { return k >= 0 && k < (int)lu_.size() ? lu_[(size_t)k].unit : -1; }
    /* The heat policy's retargets (FileStager::p1_heat_policy, between passes): lent slots ks[i], each a
     * resident's, go to units[i] (laid out as lend_init's; n at most ds41p1::kMaxLendHeat, slots and units
     * distinct, no unit one a slot holds). `behind`: an event recorded on the compute stream at the end of
     * the pass before this one, so behind every reader of the residents so far; the DMA lane waits on it
     * before anything changes (retarget_fence), and the slots' copies, queued once their loads have landed,
     * run behind it. Returns the slots retargeted (their residents withdrawn, their reloads wanted); 0
     * with nothing changed when the lender is not settled, an argument is refused or the lane did not take
     * the wait (RingLender::Stats::declined says which). */
    int  lend_retarget(const int* ks, const LendUnit* units, int n, RadEvent behind);
    /* LendHost */
    int  load(int k) override;
    int  load_ram(int k) override;
    void hold_drive(bool on) override;
    int  landed(bool* ok) override;
    int  copy(int k) override;
    int  copied(int k) override;
    void drop(int k) override;
    void publish(int k, bool on) override;
    void fence(int k) override;
    int  retarget_fence(int k) override;
    void retarget(int k) override;
    int  audit(int k, bool publish) override;

    /* Device */
    int  event_create(bool host) override;
    int  copy(int slot, const Source& src, int wait, int done) override;
    int  compute_wait(int ev) override;
    int  compute_record(int ev) override;
    int  query(int ev) override;
    int  sync(int ev) override;
    int  wave(int parity, int w, const void* const* gu_tab, const void* const* dn_tab, int n) override;
    void release(Kind k, int32_t handle) override;
    int  wait_drive() override;

    struct Counts {
        uint64_t passes = 0, layers = 0, ahead_posted = 0, ahead_used = 0, ahead_dropped = 0;
        uint64_t drive_units = 0, drive_bytes = 0, drive_busy_ns = 0, pass_ns = 0;
        uint64_t wait_drive_ns = 0, pack_ns = 0, packed_units = 0, block_units = 0, ready_wait_ns = 0;
        uint64_t admitted = 0, leases_at_end = 0;
        uint64_t verified = 0, differ = 0, verify_skipped = 0;   /* FileStager::p1_verify's */
        uint64_t swiglu_spans = 0, swiglu_rows = 0, swiglu_full_rows = 0, swiglu_fallbacks = 0;
        /* The lent ring's reloads: from the RAM cache (its lease the copy's source where it holds the
         * card's block form; relaid from the packed form into a reload buffer where it is packed, and the
         * reload thread's time in those), read from the drive (and the reload thread's time in them),
         * failed reads or relayouts. */
        uint64_t lend_ram = 0, lend_ram_packed = 0, lend_ram_packed_ns = 0;
        uint64_t lend_drive = 0, lend_drive_ns = 0, lend_read_failed = 0;
        uint64_t lend_audit_ns = 0;   /* the issuing thread's time in the read-back audit */
    };
    Counts& counts_mut() { return c_; }
    const Counts& counts() const { return c_; }
    const CardExpertStager::Stats& sched() const { return st_.stats(); }
    /* kLogStats's line, for this rank. */
    std::string stats_line(int rank) const;

private:
    struct DUnit {
        int32_t unit = -1, li = -1;
        int     rows = 0;
        std::shared_ptr<Mover::HostPlanes> planes;
        int     state = 0;           /* 0 queued to read, 1 being read, 2 landed, 3 failed */
        bool    demanded = false, admit = false, dropped = false, packing = false;
    };
    struct Cap {
        bool ok = false;
        bool swiglu_span_ok = false;
        LaunchFn fn = nullptr;
        RadArgs a{};
        std::vector<RadTensor> t;
        std::vector<RadParam>  p;
    };
    Mover*     mover_ = nullptr;
    Tunables   t_;
    int        rank_ = 0;
    bool       on_ = false, pass_ = false;
    PlanesGet  planes_;
    Insert     insert_;
    CardExpertStager st_;
    char*      ring_ = nullptr;
    const void* alias_ = nullptr;
    RadStream  s_ = nullptr;
    std::vector<RadEvent> ev_;
    std::vector<uint8_t>  ev_host_;
    void**     tab_pin_ = nullptr;   /* [2][96 waves][2][96] */
    void**     tab_dev_ = nullptr;
    Cap        cap_[3];
    std::array<int32_t, 96> layer_counts_{};
    bool       layer_counts_ok_ = false;
    int64_t    rows_ = 0;
    int32_t    li_ = -1;
    std::chrono::steady_clock::time_point t_pass_{};
    /* RAM leases by handle */
    std::vector<std::unique_ptr<Mover::HostUnit>> hu_;
    std::vector<int32_t> hu_free_;
    /* The drive's pipeline, under mu_. */
    std::mutex mu_;
    std::condition_variable cv_read_, cv_pack_, cv_done_, cv_idle_;
    std::deque<std::shared_ptr<DUnit>> read_q_;     /* to read, in order */
    std::deque<std::shared_ptr<DUnit>> pack_q_;     /* landed and demanded, to pack */
    struct Done { int32_t unit; Source src; int32_t handle; };
    std::deque<Done> done_q_;                       /* packed, for the issuing thread */
    std::map<int32_t, std::shared_ptr<DUnit>> by_unit_;   /* this layer's and the next's, by unit */
    std::vector<uint8_t*> ready_;
    std::vector<int32_t>  ready_free_;
    bool reading_ = false, packing_ = false, stop_ = false;
    int  error_ = 0;
    int  owed_ = 0;                                 /* demanded drive units not yet handed over */
    std::thread reader_, packer_;
    Counts c_;
    struct LandCtx {
        CardExpertPath* path = nullptr;
        std::vector<std::shared_ptr<DUnit>> units;   /* by index in the read call */
        std::unique_ptr<std::atomic<uint8_t>[]> done;
    };
    void reader_loop();
    void packer_loop();
    static void landed(void* ctx, int i);
    int  drop_all();
    /* The lent ring and arena: by lent slot its unit, address, copy event (host) and source (a reload
     * buffer, -1 its RAM-cache lease in lhu_, -2 none), whether lhu_ holds a packed RAM-cache lease the
     * reload thread relays from into the slot's buffer (lram_: set at load, let go at landed, both the
     * issuing thread's), and for the ring's (the first lring_.size()) the ring slot; the two pinned reload
     * buffers, each with planes for the drive's read and scratch for the packed form's scales; the reload
     * thread's queue and what it landed. lmu_ guards lbuf_free_, lq_ and ldone_. */
    RingLender lend_;
    std::vector<LendUnit> lu_;
    std::vector<char*> lat_;
    std::vector<int> lring_, lev_, lsrc_;
    std::vector<std::unique_ptr<Mover::HostUnit>> lhu_;
    std::vector<uint8_t*> lbuf_;
    std::vector<std::unique_ptr<Mover::HostPlanes>> lplanes_;
    std::vector<int> lbuf_free_;
    struct LendJob { int k = -1, b = -1; bool ram = false; };
    std::deque<LendJob> lq_;
    std::vector<uint8_t> lram_;
    std::vector<std::vector<uint8_t>> lscales_;
    int  load_from(int k, bool drive);
    std::deque<std::pair<int, bool>> ldone_;
    std::mutex lmu_;
    std::condition_variable lcv_;
    bool lstop_ = false, lhold_ = false;   /* lhold_: drive jobs wait (hold_drive) */
    std::thread lth_;
    void lend_loop();
    void lend_stop();
    /* lend_retarget's call in flight: the slots and their next units, the event the DMA lane waits on,
     * and whether it waits already (one wait covers every copy queued after it). */
    int      lnext_k_[ds41p1::kMaxLendHeat] = {};
    LendUnit lnext_[ds41p1::kMaxLendHeat];
    int      lnext_n_ = 0;
    RadEvent lfence_ = nullptr;
    bool     lfenced_ = false;
    int      lprimed_ = 0;       /* the slab slots lend_prime left holding their unit */
    /* The read-back audit (null and -1 without the switch): a pinned slot of host memory the card's copy
     * is read into, its event, the residents' last-reader event of this pass, and the digests. */
    uint8_t* laud_buf_ = nullptr;
    int      laud_ev_ = -1;
    RadEvent laud_behind_ = nullptr;
    LendAudit laud_;
};

/* The lent arena's slots in [lo, lo + bytes) (the activation arena's top above a level's buffers and
 * guard): whole ds41p1::kSlotBytes slots from the top down, each on a 4 KiB boundary, at most `max`.
 * The deepest slot lies highest, so a smaller count leaves the bytes nearest the buffers unused. */
std::vector<char*> arena_slots(char* lo, int64_t bytes, int max);

}  /* namespace p1 */
}  /* namespace rad */
