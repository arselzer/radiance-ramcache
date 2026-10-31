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
    bool align_end = true;
    bool swiglu_span = false;
};
/* The tunables from `get` (getenv for the engine): each a whole number in its range (wave 1-16,
 * inflight 1-64, ready 1-32, ahead 0-64, verify 0-4096, min_rows 17-4096, lend 0-64, lend_arena 0-64,
 * grace_ms 0-60000), align "end" or "start", swiglu_span 0/off or 1/on (default off).
 * False with the reason on anything else. */
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
    /* Read-only audit notification AFTER copy() accepted the unchanged source/slot. Issuing
     * thread only; no retained Source pointers and no scheduling decision may depend on it. */
    using CopyObserver=void(*)(void*,int expert,int32_t unit,int rows,int slot,Kind,const Source&);
    void set_copy_observer(CopyObserver fn,void*ud){copy_observer_=fn;copy_observer_ud_=ud;}
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
    CopyObserver copy_observer_=nullptr;void*copy_observer_ud_=nullptr;

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
 * where it holds the card's block form, else read from the drive and relaid into one of two pinned
 * buffers by a reload thread, then copied into their slots on the DMA lane, behind the card path's
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
 * RingLender is the state machine alone, against LendHost (tests/p1_stager_test.cpp drives it with
 * random landings, copy completions and failures); CardExpertPath is the engine's LendHost. */
class LendHost {
public:
    virtual ~LendHost() = default;
    /* Lent slot k's unit read for its slot in the background: 1 begun, 0 not now (every reload
     * buffer busy), < 0 not possible (the unit stays on the CPU path). */
    virtual int  load(int k) = 0;
    /* A begun load that landed since the last call: its slot (and *ok whether it read), or -1. */
    virtual int  landed(bool* ok) = 0;
    /* Slot k's landed bytes copied into its slot on the DMA lane, an event recorded after. */
    virtual int  copy(int k) = 0;
    /* Slot k's copy: 1 passed, 0 not yet, < 0 an error. */
    virtual int  copied(int k) = 0;
    /* Slot k's landed bytes may go (its buffer back to the reload thread, or its RAM-cache lease). */
    virtual void drop(int k) = 0;
    /* Slot k's unit published at its slot (on), or as a file-tier unit again (off). */
    virtual void publish(int k, bool on) = 0;
    /* A pass that takes the slots back found slot k's copy still queued: whatever writes the slot in
     * that pass must wait for it (an arena slot: the compute stream waits on its event; a ring slot
     * needs nothing, the card path's own copies run behind it on the DMA lane). */
    virtual void fence(int k) = 0;
};

class RingLender {
public:
    struct Stats {
        uint64_t lends = 0;        /* card-path bursts that took the slots back */
        uint64_t withdrawn = 0;    /* residents published as file-tier units again for them */
        uint64_t returns = 0;      /* reload rounds begun after a grace */
        uint64_t reloaded = 0;     /* units published at their slots */
        uint64_t failed = 0;       /* loads or copies that failed: their units left on the CPU path */
        uint64_t cancelled = 0;    /* loads or copies a card-path pass overtook, never published */
        uint64_t held_back = 0;    /* passes that found the grace still running */
    };
    /* n slots (1..ds41p1::kMaxCombinedLend: the ring's and the arena's), the grace in ms. Every slot starts
     * empty and lent, as after a card-path pass long ago: the first pass that does not take the slots
     * begins the reloads. */
    int  init(LendHost* host, int n, int64_t grace_ms);
    int  n() const { return (int)slot_.size(); }
    /* A pass begins at now_ms; p1: it takes the slots back (it arms the card path, or with the lent
     * arena runs at the full plan). With p1 every resident is published as a file-tier unit again and
     * every load or copy in flight is cancelled, each copy still queued fenced first (LendHost::fence);
     * returns 1 when some resident was (the pass's first ring copy must wait on the residents' last
     * reader). Without, once the grace has run out the reloads begin; landed loads are copied, passed
     * copies published, and further loads begun as buffers come free. Never fails: a host call that
     * fails leaves its unit on the CPU path. */
    int  begin_pass(bool p1, int64_t now_ms);
    /* A pass that took the slots back ended at now_ms: the grace runs from here. */
    void end_p1(int64_t now_ms);
    bool lent() const { return lent_; }
    int  residents() const;
    /* Slots holding a buffer or lease (a load in flight, landed, or a copy in flight). */
    int  busy() const;
    bool resident(int k) const { return k >= 0 && k < n() && slot_[(size_t)k].st == St::Resident; }
    const Stats& stats() const { return st_; }

private:
    /* Idle: not resident and nothing wanted (lent, or nothing yet). Want: a reload wanted, not begun
     * (no buffer). Reading: begun; Landed: its bytes in; Copying: its copy queued. cancel: a reading
     * or copying slot a card-path pass overtook, which goes back to Idle or Want once it ends. */
    enum class St : uint8_t { Idle, Want, Reading, Landed, Copying, Resident, Failed };
    struct Slot { St st = St::Idle; bool cancel = false; };
    LendHost* host_ = nullptr;
    std::vector<Slot> slot_;
    int64_t grace_ = 0, last_p1_ = 0;
    bool lent_ = true, any_p1_ = false;
    Stats st_;
    void settle(int k);
    void poll();
    void pump();
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
    void set_copy_observer(CardExpertStager::CopyObserver fn,void*ud){st_.set_copy_observer(fn,ud);}
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

    /* ---- the lent ring and the lent arena (RingLender's host) ----
     * lend_init: units[k] (its two weights' offsets and sizes in the unit's layout, as the plan lays it
     * out) lent ring slot ring_slots[k] for k below their count, then the arena slots arena_slots[k -
     * that count] (whole ds41p1::kSlotBytes slots of the activation arena's lent top, arena_slots);
     * their reloads begin at the first pass that does not take the slots back. < 0 with *why when it
     * cannot run (the caller refuses the switch then). */
    struct LendUnit { int32_t unit = -1; int64_t off[2] = { 0, 0 }, bytes[2] = { 0, 0 }; };
    int  lend_init(const std::vector<LendUnit>& units, const std::vector<int>& ring_slots,
                   const std::vector<char*>& arena_slots, int64_t grace_ms, std::string* why);
    bool lending() const { return lend_.n() > 0; }
    RingLender& lender() { return lend_; }
    const RingLender& lender() const { return lend_; }
    /* Lent slot k's ring slot, or -1 (an arena slot); its address; the arena's lent slots. */
    int  lend_ring_slot(int k) const { return k >= 0 && k < (int)lring_.size() ? lring_[(size_t)k] : -1; }
    char* lend_slot(int k) const { return k >= 0 && k < (int)lat_.size() ? lat_[(size_t)k] : nullptr; }
    int  lend_arena_slots() const { return (int)(lat_.size() - lring_.size()); }
    int32_t lend_unit(int k) const { return k >= 0 && k < (int)lu_.size() ? lu_[(size_t)k].unit : -1; }
    /* LendHost */
    int  load(int k) override;
    int  landed(bool* ok) override;
    int  copy(int k) override;
    int  copied(int k) override;
    void drop(int k) override;
    void publish(int k, bool on) override;
    void fence(int k) override;

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
        /* The lent ring's reloads: from the RAM cache, read from the drive (and the reload thread's
         * time in them), failed reads. */
        uint64_t lend_ram = 0, lend_drive = 0, lend_drive_ns = 0, lend_read_failed = 0;
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
     * buffer, -1 its RAM-cache lease in lhu_, -2 none), and for the ring's (the first lring_.size()) the
     * ring slot; the two pinned reload buffers, each with planes for the drive's read; the reload
     * thread's queue (lent slot, buffer) and what it landed. lmu_ guards lbuf_free_, lq_ and ldone_. */
    RingLender lend_;
    std::vector<LendUnit> lu_;
    std::vector<char*> lat_;
    std::vector<int> lring_, lev_, lsrc_;
    std::vector<std::unique_ptr<Mover::HostUnit>> lhu_;
    std::vector<uint8_t*> lbuf_;
    std::vector<std::unique_ptr<Mover::HostPlanes>> lplanes_;
    std::vector<int> lbuf_free_;
    std::deque<std::pair<int, int>> lq_;
    std::deque<std::pair<int, bool>> ldone_;
    std::mutex lmu_;
    std::condition_variable lcv_;
    bool lstop_ = false;
    std::thread lth_;
    void lend_loop();
    void lend_stop();
};

/* The lent arena's slots in [lo, lo + bytes) (the activation arena's top above a level's buffers and
 * guard): whole ds41p1::kSlotBytes slots from the top down, each on a 4 KiB boundary, at most `max`.
 * The deepest slot lies highest, so a smaller count leaves the bytes nearest the buffers unused. */
std::vector<char*> arena_slots(char* lo, int64_t bytes, int max);

}  /* namespace p1 */
}  /* namespace rad */
