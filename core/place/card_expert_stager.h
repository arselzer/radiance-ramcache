/* card_expert_stager.h -- prefill Phase 1: every routed expert of a large prefill chunk computed on
 * the cards (tools/ds41-host-experts/ds41_p1.h is the contract with libr4d's twin row; the plan is
 * PHASE1-PLAN.md in the review directory).
 *
 * ---- what it does ------------------------------------------------------------------------------
 *
 * Under RADIANCE_EXPERIMENTAL_DS41_P1=1 beside RADIANCE_EXPERIMENTAL_DS41_LCP=<C>, a pure prefill pass
 * of at least ds41p1::kMinArmRows tokens gives no routed expert to the CPU. At each routed layer the
 * file stager (stager.h) learns the layer's counts as it always does, then:
 *   - VRAM residents run in the plugin's own three expert ops (gate_up, the clamped SwiGLU, down);
 *   - every touched expert that is not resident is published to those ops with kMarkSkip, so the P1
 *     twin leaves its rows alone (T3);
 *   - each such expert is copied into a slot of a VRAM ring on the card's DMA lane: a RAM-cache unit
 *     straight from its pinned slab slot (borrowed read-only, packed or block form), a drive miss
 *     after it was read into host planes and packed by a packer thread into a pinned ready slot;
 *   - right after the layer's down projection was issued, the stager launches the same three ops
 *     again, in waves of a few ring units, each with its own expert table: the wave's units name
 *     their ring slots (kMarkPacked where packed) and every other entry carries kMarkSkip. The same
 *     kernels and arguments as the plugin's launch, so the rows are the decode band's bits (T1).
 * The SwiGLU relaunch runs over all of the layer's slots each wave: the rows already computed come
 * out the same from their unchanged gate_up rows, so it is idempotent.
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
struct Tunables {
    int  wave = ds41p1::kDefWave;
    int  inflight = ds41p1::kDefInflight;
    int  ready = ds41p1::kDefReady;
    int  ahead = ds41p1::kDefAhead;
    int  verify = 0;
    bool align_end = true;
};
/* The tunables from `get` (getenv for the engine): each a whole number in its range (wave 1-16,
 * inflight 1-64, ready 1-32, ahead 0-64, verify 0-4096), align "end" or "start". False with the reason
 * on anything else. */
bool parse_tunables(const std::function<const char*(const char*)>& get, Tunables* t, std::string* why);

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
enum class Kind : uint8_t { Ram = 0, Drive = 1 };

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
        uint64_t layers = 0, ram = 0, drive = 0, waves = 0, copies = 0, copy_bytes = 0;
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
    std::vector<int>  ram_order_;      /* RAM units by rows, descending */
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

/* ---- the engine's side ----------------------------------------------------------------------------
 *
 * One a rank, owned by its FileStager. The real Device: the ring's events, the mover's copy lists,
 * the wave tables (pinned rows, uploaded on the compute stream) and the captured launches. The drive
 * misses: a reader thread posts them to the mover's kept readers (Mover::read_unit_planes, at most
 * `inflight` units a call, with the pass's rows for the plane mirror's share), a packer thread packs
 * each landed unit the layer wants into a pinned ready slot (Mover::pack_unit_for_ring), and the
 * issuing thread copies it to the ring. Units read ahead for the next layer wait, read, for that
 * layer's choice. */
class CardExpertPath final : public Device {
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

    /* An armed pass of `rows` tokens begins; `hand_over` (null entries skipped): events behind the
     * last readers of the file buffers before it, which the first copy waits on. */
    int  begin_pass(int64_t rows, const RadEvent* hand_over, int n_hand_over);
    /* Every copy seen, every source let go, read-ahead dropped, the readers and the packer idle;
     * then `end` is recorded on `s` and the mover's stream waits on it, so a later pass's copies
     * into the file buffers run behind this pass's last reader. */
    int  end_pass(RadStream s, RadEvent end);
    bool pass_on() const { return pass_; }

    /* A routed layer: its units, then start (RAM copies, demand reads posted), then the launches of
     * its three ops captured, then run (the waves) behind the down projection on `s`. */
    int  begin_layer(int32_t li, int parity);
    /* A cached unit, borrowed (touch: with the LRU touch an admitted unit gets): false when the RAM
     * cache does not hold it (the caller adds it as a drive miss then). */
    int  add_ram(int e, int32_t unit, int rows, bool touch, bool* hit);
    int  add_drive(int e, int32_t unit, int rows, bool admit);
    int  start();
    /* Layer li + 1's likely misses, read after this layer's demand reads (up to `ahead`). */
    int  read_ahead(int32_t next_li, const std::vector<int32_t>& units);
    int  capture(int which, const RadArgs* a, LaunchFn fn);   /* 0 gate_up, 1 SwiGLU, 2 down */
    bool captured() const { return cap_[0].ok && cap_[1].ok && cap_[2].ok; }
    /* A captured launch's operands (0 gate_up, 2 down), for the host verify; null when not captured. */
    const RadTensor* captured_operands(int which, int* n) const {
        if (which < 0 || which > 2 || !cap_[which].ok) return nullptr;
        *n = (int)cap_[which].t.size();
        return cap_[which].t.data();
    }
    int  run(RadStream s);
    bool layer_open() const { return st_.layer_open(); }

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
};

}  /* namespace p1 */
}  /* namespace rad */
