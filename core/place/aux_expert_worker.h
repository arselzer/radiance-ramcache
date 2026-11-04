/* aux_expert_worker.h -- the aux expert worker: a spare card of the ranks' ISA holds the hottest CPU-path
 * units in its VRAM and computes their rows for routed layer passes of at most 16 rows
 * (RADIANCE_EXPERIMENTAL_DS41_AUX_EXPERTS=<MiB>; tools/ds41-host-experts/ds41_aux.h has the switches, the
 * pure decision and layouts; work/aux/SPEC.md the design). Off by default; off, nothing here runs.
 *
 * ---- who does what ------------------------------------------------------------------------------
 *
 *   Engine::aux_discover   right after the devices are chosen: parse, the pin set (the extra card maps no
 *                          pinned memory but its own), the decision, the arena reserved (Worker::reserve)
 *   Engine::aux_start      after every rank's FileStager is up: the unit set, the lanes, the threaded load
 *                          with a read-back compare (FileStager::aux_load), the tables, a warm-up job per
 *                          lane checked against the CPU pool's rows, then FileStager::aux_bind per rank
 *   FileStager (hx_*)      per layer pass: hx_choose leases the worker's resident units, hx_submit compacts
 *                          their sorted slots and, after the pool's submit, queues the lane's job;
 *                          hx_finish places the rows (or compares them, verify mode) or has the CPU compute
 *                          them from the kept leases when the job is late or failed
 *   Worker::~Worker        after every stager is unbound: each lane drained for at most 2 s (its job, then a
 *                          probe behind everything queued, an idle lane's too), then streams, events, pinned
 *                          buffers and the arena freed (left to the process exit otherwise: a queue that never
 *                          answers is not destroyed, since rad_stream_destroy's host wait has no bound)
 *
 * ---- exactness ----------------------------------------------------------------------------------
 *
 * The worker computes only rows the CPU path computes (selected, cached, file-tier units of trunk routed
 * layers in passes of at most 16 rows), with libr4d's decode-band rows (moe_gemm_ggml_mxfp4_gemv_vp,
 * swiglu_clamped_bf16) from the code object the rank cards run, on weights that are the RAM cache's packed
 * slots read back identical, from a byte copy of the rows the CPU reads. Each output value of those rows
 * depends on its weight row, activation row and K only, so a compacted launch gives the full launch's bytes
 * (the device check `aux` holds both, and gemv_vp to gemv, on the extra card and on an XT). A late or failed
 * job's rows are the CPU's. So hx_out_ holds the CPU path's bytes whatever the worker does. */
#pragma once
#include "../../tools/ds41-host-experts/ds41_aux.h"   /* the switches, decision and layouts (also on rad_core's path) */
#include "rad_abi.h"
#include "rad_device.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace rad {
namespace aux {

using Config = ds41aux::Config;

/* Every visible device's facts as the decision reads them (fresh free VRAM, PCI path). I/O. */
std::vector<ds41aux::Card> inventory(const std::string& sysfs = "/sys");
/* The selector as a PCI address: cfg.device itself, or its UUID through the KFD topology ("" when no node
 * has it, or no selector). */
std::string selector_bdf(const Config& cfg, const std::string& sysfs = "/sys");

/* One kernel row a layer job relaunches: the band row of the rank's own op at the job's height
 * (FileStager::aux_resolve), as the issue path would call it. */
struct Launch {
    int (*fn)(const RadArgs*, RadStream) = nullptr;
    const RadParam* p = nullptr;
    int n_p = 0;
    void* instance = nullptr;
};

/* One layer job: the pass's token rows, the sorted order compacted to the worker's experts (ds41aux::compact)
 * and the layer's tables on the extra card. */
struct Job {
    int AR = 0, T = 0;
    const uint16_t* x = nullptr;        /* AR rows of kEmbd bf16 (the rank's hx_x_) */
    const int32_t* sorted = nullptr;    /* T */
    const int32_t* off = nullptr;       /* kExperts + 1 */
    Launch gu, sw, dn;
    RadTensor gu_w{}, dn_w{};           /* the weight-table operands, data the layer's tables */
    int rank = 0, world = 1;
};

/* Card sharing's gate (work/aux-dyn/SPEC.md §3.3): the control thread shuts it before it drains the lanes and
 * frees the arena; a rank holds its in_lane entry from hx_choose's decision to the layer's last lane call (a
 * Dekker pair: seq_cst on both stores and both loads), and the step boundary holds it on a parked rank's behalf
 * while it polls the lane. Made only with a controller (Config::controller()); null otherwise. */
constexpr int kMaxGateRanks = 8;
struct Gate {
    std::atomic<bool>    closed{ false };
    std::atomic<uint8_t> in_lane[kMaxGateRanks]{};
    /* The rank side: false (nothing held) when the gate is shut. */
    bool enter(int r) {
        in_lane[r].store(1, std::memory_order_seq_cst);
        if (closed.load(std::memory_order_seq_cst)) {
            in_lane[r].store(0, std::memory_order_seq_cst);
            return false;
        }
        return true;
    }
    void leave(int r) { in_lane[r].store(0, std::memory_order_release); }
    /* The control side, after closed.store(true): whether any rank still holds its lane. */
    bool busy(int n) const {
        for (int r = 0; r < n && r < kMaxGateRanks; ++r)
            if (in_lane[r].load(std::memory_order_seq_cst)) return true;
        return false;
    }
};

/* A rank's queue on the extra card (DeviceLane), or a fake in the tests. One job in flight at most: one
 * routed layer is in flight per rank, and the queue is in order. */
class Lane {
public:
    virtual ~Lane() = default;
    enum State { kReady = 0, kBusy = 1, kBackoff = 2, kDead = 3 };
    /* Whether a layer job may be submitted now. Polls an abandoned job: once it completed, the lane backs
     * off (if the abandon asked for it) and is ready again after that many declined layer jobs. */
    virtual State state() = 0;
    /* One layer job declined while backing off: counts the backoff down. */
    virtual void declined() = 0;
    /* Queues the job: the record (a host copy), one H2D, the three launches, the D2H of the rows, the
     * event. Every step's status is the job's: < 0 means a step failed, the job is failed (wait answers
     * < 0 without querying, rows() is never read) and the lane is dead, said once. */
    virtual int submit(const Job& j) = 0;
    /* 1: done, rows() holds its T rows (read only for a job whose own record completed); 0: still
     * pending at `deadline_ns` (steady clock); < 0: failed. Queries at least once whatever the deadline. */
    virtual int wait(int64_t deadline_ns) = 0;
    virtual const uint16_t* rows() const = 0;
    /* The job in flight is given up: the lane stays busy until it completes (its rows are never read),
     * then declines `backoff` layer jobs when asked. */
    virtual void abandon(bool backoff) = 0;
    /* Teardown: polls the job in flight, then a probe recorded behind everything queued (an idle lane's too: a
     * card that hung after its last job, with no fault the host sees, completes nothing), for at most
     * timeout_ns (the probe at least 100 ms). True when the probe completed: nothing is left on the queue and
     * it may be destroyed; false leaves the lane to the process exit. */
    virtual bool drain(int64_t timeout_ns) = 0;
    /* Why the lane died ("" while alive). */
    virtual std::string why_dead() const = 0;
    /* A release's drain (work/aux-dyn/SPEC.md A.5): as drain(), and a lane retired by a host-side refusal (a job's
     * shape, a copy or launch the queue refused, with no fault seen) is probed too, so a release can free the
     * arena; a lane that faulted (a failed query, a device fault) is never drained. */
    virtual bool drain_release(int64_t timeout_ns) { return drain(timeout_ns); }
    /* The lane died of a device fault or a failed query (not drainable). */
    virtual bool faulted() const { return false; }
};

/* One rank's part of the worker: which of its plan units are resident, their slots, and the layer tables. */
struct RankView {
    std::vector<uint8_t>     resident;   /* by plan unit */
    std::vector<int32_t>     slot;       /* by plan unit: the arena slot, or -1 */
    std::vector<const void*> gu_tab, dn_tab;   /* by routed layer: the tables' device addresses, or null */
    /* By routed layer x kExperts: a resident unit's weight in its slot | ds41p1::kMarkPacked, or null
     * (never read: a job's compacted offsets give every other expert an empty range). The loader fills
     * them; Worker::load_tables writes them to the card. */
    std::vector<const void*> gu_tab_entries, dn_tab_entries;
    int64_t units = 0;
    double  share = 0, share_all = 0;    /* the profile share held, and of every candidate */
    bool is_resident(int32_t u) const {
        return u >= 0 && (size_t)u < resident.size() && resident[(size_t)u] != 0;
    }
};

/* A rank's counters (FileStager::aux_stats, the shutdown line). */
struct Stats {
    uint64_t jobs = 0, units = 0, rows = 0, placed = 0, eligible_rows = 0;
    uint64_t declined_tall = 0, declined_draft = 0, declined_rows = 0, declined_busy = 0, declined_dead = 0;
    uint64_t uncached = 0, late = 0, forced_late = 0, fallback_units = 0, fallback_rows = 0, fallback_ns = 0;
    uint64_t failed = 0, wait_ns = 0, submit_ns = 0, rt_sum_ns = 0, rt_max_ns = 0, rt_n = 0;
    uint64_t verified = 0, differ_rows = 0, differ_values = 0, not_compared = 0;
    uint64_t declined_released = 0;   /* card sharing: a layer whose rank found the gate shut */
};
/* The shutdown line of rank r (ds41aux::kLogRank); `released`: with a controller, the declines while released
 * ("..., L while released" after "... after it failed"). */
std::string stats_line(int r, const Stats& s, bool released = false);

class DeviceLane;

/* The process's worker: the arena on the extra card, the tables, the lanes, the ranks' views. */
class Worker {
public:
    Worker(const Config& cfg, const ds41aux::Card& card, const std::vector<int>& ranks);
    ~Worker();
    Worker(const Worker&) = delete;
    Worker& operator=(const Worker&) = delete;

    /* Discovery's reservation (work/aux/SPEC.md §3.5b): the arena of cap MiB on the extra card, and free VRAM
     * of at least the headroom after it. The caller's current device is restored. < 0 with *why: off. */
    int  reserve(int restore_device, std::string* why);
    /* The arena's regions for `ranks` x `layers` and `slots` units of `L`'s stride, and a lane per rank (on
     * the extra card; the caller's device restored). < 0 with *why: off. carve() then make_lanes(). */
    int  build(int layers, int64_t slots, const ds41aux::Layout& L, int restore_device, std::string* why);
    /* work/aux-dyn/SPEC.md §3.6. carve: the regions of the arena (reserve()d) and the views' tables; the lanes made
     * already get their workspaces (set_ws: their queues idle, the gate shut). make_lanes: a lane per rank (queues,
     * events, pinned record and rows: nothing in the arena; a lane made without an arena has no workspace until
     * a carve). release_arena: every lane and the swap queue drained (the caller's word), the arena freed with the
     * extra card bound (then restore_device bound, when >= 0); the lanes keep their queues. has_arena: an arena is
     * held. */
    int  carve(int layers, int64_t slots, const ds41aux::Layout& L, std::string* why);
    /* A released start (no arena): the shape carve() will use at the first acquire, and the views (no unit, no
     * table); then make_lanes(). */
    void plan(int layers, int64_t slots, const ds41aux::Layout& L);
    int  make_lanes(int restore_device, std::string* why);
    void release_arena(int restore_device = -1);   /* -1: the extra card stays the current device */
    bool has_arena() const { return arena_ != nullptr; }
    /* The tables written from `image` (the control thread's mirror: ranks x layers x 2 x kExperts words), read back
     * and compared, as load_tables() does from the views. */
    int  load_tables_image(const uint64_t* image, std::string* why);
    /* The views' entries as a table image (the control thread's mirror after the startup load). */
    std::vector<uint64_t> table_image() const;
    int  layers() const { return layers_; }
    /* The control thread's teardown: its swap queue did not drain, so the arena stays (finish() says so). */
    void set_swap_undrained() { swap_undrained_ = true; }
    /* The units held, as the control thread counts them (-1: the views' count, the merged worker's). */
    void set_units_held(int64_t n) { units_held_.store(n, std::memory_order_relaxed); }
    /* finish() with a controller: lanes retired by a refusal are probed (Lane::drain_release). */
    void set_release_drain(bool on) { release_drain_ = on; }
    /* The tables of every rank (the views' entries) to the card, read back and compared. */
    int  load_tables(std::string* why);
    /* A free-VRAM check on the extra card against the headroom (after the lanes' code objects loaded). */
    int  check_headroom(std::string* why) const;

    const Config&        cfg() const { return cfg_; }
    const ds41aux::Card& card() const { return card_; }
    int                  device() const { return card_.ord; }
    int                  n_ranks() const { return (int)ranks_.size(); }
    int64_t              arena_bytes() const { return arena_bytes_; }
    int64_t              free_after() const { return free_after_; }
    const ds41aux::Layout& layout() const { return layout_; }
    char*                slot(int64_t k) const { return slots_ + k * layout_.stride; }
    int64_t              n_slots() const { return n_slots_; }
    char*                table(int r, int li, int which) const;   /* which 0 gate_up, 1 down */
    Lane*                lane(int r) const;
    DeviceLane*          device_lane(int r) const;
    RankView&            view(int r) { return views_[(size_t)r]; }
    const RankView&      view(int r) const { return views_[(size_t)r]; }
    int64_t              units() const;
    /* The process line at exit (ds41aux::kLogDevice), and the lanes drained (at most 2 s each). */
    std::string          finish(uint64_t jobs, uint64_t rows, uint64_t late);
    /* finish() saw every lane drained: the streams and the arena may be freed. */
    bool                 drained() const { return drained_; }
    /* The loader's bound on each wait for its copies (FileStager::aux_load, load_tables): 10 s. */
    int64_t              load_wait_ns() const { return load_wait_ns_; }
    void                 set_load_wait_ns(int64_t ns) { load_wait_ns_ = ns; }   /* a host test's shorter bound */

private:
    Config cfg_;
    ds41aux::Card card_;
    std::vector<int> ranks_;
    char*   arena_ = nullptr;
    int64_t arena_bytes_ = 0, free_after_ = -1;
    int     layers_ = 0;
    ds41aux::Layout layout_{};
    char*   tables_ = nullptr;
    char*   ws_ = nullptr;
    char*   slots_ = nullptr;
    int64_t n_slots_ = 0;
    std::vector<std::unique_ptr<DeviceLane>> lanes_;
    std::vector<RankView> views_;
    bool    drained_ = true;   /* finish() found every lane idle: streams and arena may be freed */
    bool    finished_ = false;
    int64_t load_wait_ns_ = 10'000'000'000LL;
    bool    swap_undrained_ = false, release_drain_ = false;
    std::atomic<int64_t> units_held_{ -1 };
    int  load_tables_with(void (*fill)(const Worker*, const void*, uint64_t*), const void* ctx, std::string* why);
};

/* A rank's lane on the extra card: an AQL queue, a host-waitable event, the pinned record and output rows
 * (mapped on the extra card only), its VRAM workspace (in the arena). */
class DeviceLane final : public Lane {
public:
    DeviceLane(int device, int rank, char* ws, int64_t backoff);
    ~DeviceLane() override;
    /* Creates the queue (on the current device: the caller sets the extra card), the event and the pinned
     * buffers. */
    int init(std::string* why);
    State state() override;
    void declined() override;
    int  submit(const Job& j) override;
    int  wait(int64_t deadline_ns) override;
    const uint16_t* rows() const override { return out_; }
    void abandon(bool backoff) override;
    bool drain(int64_t timeout_ns) override;
    std::string why_dead() const override { return why_dead_; }
    bool drain_release(int64_t timeout_ns) override;
    bool faulted() const override { return dead_ && fault_; }
    /* A new workspace (an acquire's arena; null at a release): the lane idle, the gate shut. */
    void set_ws(char* ws) { ws_ = ws; }
    /* The loader's use of the queue (FileStager::aux_load): copies on it and a bounded wait for them. */
    RadStream stream() const { return s_; }
    RadEvent  event() const { return ev_; }
    /* Records the event behind everything queued and polls it for at most timeout_ns: 1 done, 0 timed out,
     * < 0 failed. A wait that did not see the event complete leaves the lane not idle: timed out, what it
     * waited for stays the lane's work in flight (drain() polls this same event, and a lane it does not see
     * drain is left to the process exit with the arena); a failed record or wait retires the lane (dead: never
     * drained). The caller frees what those copies read or write only after a 1 (work/aux/SPEC.md §7.5). */
    int  sync_bounded(int64_t timeout_ns);
    /* Leaves the queue, event and buffers to the process exit (a lane that did not drain). */
    void leak() { leaked_ = true; }
    bool dead() const { return dead_; }

private:
    /* `fault`: a failed query or a device fault (the queue's state unknown: never drained); otherwise a host-side
     * refusal, which drain_release probes. */
    int  retire(const char* step, int rc, bool fault = false);
    int  dev_, rank_;
    char* ws_;                  /* VRAM: in (ds41aux::ws()'s record layout), gu_out, act, dn_out */
    RadStream s_ = nullptr;
    RadEvent  ev_ = nullptr;
    char*     rec_ = nullptr;   /* pinned, the extra card's only: the record */
    uint16_t* out_ = nullptr;   /* pinned, the extra card's only: the rows */
    bool      pending_ = false; /* a job was submitted and not yet consumed */
    bool      job_ok_ = false;  /* its every step succeeded, its record included */
    bool      abandoned_ = false, backoff_after_ = false;
    int64_t   backoff_ = 0, backoff_left_ = 0;
    bool      dead_ = false, leaked_ = false, fault_ = false;
    std::string why_dead_;
};

}  // namespace aux
}  // namespace rad
