/* aux_control.h -- the aux expert worker's control thread (work/aux-dyn/SPEC.md §3, parts A and B): card sharing
 * (the probe, the state machine, the control file and holders, the state file), the slot pool, the mailbox the step
 * boundary applies, the copy executor on a swap queue of its own, and the online unit set's scores and planner.
 *
 * Made only when a switch asks for it (Config::control_thread(): RADIANCE_EXPERIMENTAL_DS41_AUX_SHARE exclusive or
 * allow, ..._AUX_CONTROL, or ..._AUX_ONLINE=1); with none of them the engine has no Control and the worker is the
 * merged one (no thread, queue, allocation, sysfs read or log line of this file).
 *
 * ---- who touches what (§3.1) ---------------------------------------------------------------------------------
 *
 *   rank thread r      its stager, lane r while it holds the gate's in_lane[r], view r (read only), its hit array
 *   scheduler thread   boundary(): the views (the only writer once bound), the hit hand-off, the gate's reopening,
 *                      the lanes' quiet checks (holding in_lane on a parked rank's behalf); never waits
 *   control thread     everything else here: the probe, the state machine, the swap queue and its bounces, the
 *                      arena (reserve, carve, release), the lanes while the gate is shut and no rank holds one
 *
 * ---- exactness (§5) -------------------------------------------------------------------------------------------
 *
 * A view changes only in boundary(), with every rank parked, so a pass sees one view. A unit enters a view only once
 * its slot and its two table entries were written and the swap queue's event seen (and, sampled, read back equal);
 * a resident unit's slot and entries are never written; a slot a view gave up is written again only after its
 * rank's lane was idle at a boundary after that (no abandoned job can still read it); the arena is freed only with
 * the gate shut, every lane drained and the swap queue drained. A rank that finds the gate shut computes its layer
 * on the CPU path, as a declined layer does. So every row the worker places is the CPU path's. */
#pragma once
#include "aux_expert_worker.h"
#include "../../tools/ds41-host-experts/ds41_aux_online.h"
#include "../../tools/ds41-host-experts/ds41_aux_share.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace rad {
namespace aux {

enum class SlotState : uint8_t { kFree = 0, kFilling, kReady, kResident, kEvicting, kLost };

/* What the control thread needs from the engine (core/engine_aux.cpp), injectable (tests/aux_fence_test.cpp). Every
 * call comes from the control thread except hit_tokens / swap_hits / guard_tripped, which boundary() makes. */
struct Hooks {
    struct Source {
        const uint8_t* gu = nullptr;
        const uint8_t* dn = nullptr;
        int64_t        gu_n = 0, dn_n = 0;
        void*          lease = nullptr;
    };
    /* Unit u's packed bytes on rank r for a copy: a read-only lease (no LRU move, second chances kept); false when
     * the RAM cache does not hold it in the worker's form. release() gives it back once the bytes are copied. */
    std::function<bool(int r, int32_t u, Source* s)> source;
    std::function<void(int r, Source* s)>            release;
    /* A warm-up job of unit u (routed layer li) on rank r's lane with the new arena's tables, its rows equal to the
     * CPU pool's: < 0 with *why. Null: none (a test). */
    std::function<int(int r, int32_t u, int32_t li, const void* gu_tab, const void* dn_tab, std::string* why)> warm;
    /* boundary(): rank r's decode tokens counted since the last hand-off; its hit array swapped for `empty` (the same
     * size, zeroed), returning those tokens; whether its stager stopped taking jobs (aux_guard). */
    std::function<uint64_t(int r)>                             hit_tokens;
    std::function<uint64_t(int r, std::vector<uint32_t>* empty)> swap_hits;
    std::function<bool(int r)>                                 guard_tripped;
    /* The driver's figure for what the card still gives (AMDKFD_IOC_AVAILABLE_MEMORY), -1 unknown. */
    std::function<int64_t()> kfd_available;
};

/* One rank's part: its candidates in ds41aux::before order with each one's routed layer index, its quota, and the
 * plan's unit count (its view's and hit array's size). */
struct RankUnits {
    std::vector<ds41aux::Cand> cands;
    std::vector<int32_t>       li;
    int64_t                    quota = 0;
    int32_t                    n_plan_units = 0;
};

struct Roots {   /* where the probe reads (a test's fake trees) */
    std::string sysfs = "/sys", procfs = "/proc";
};

class Control {
public:
    Control(const Config& cfg, const ds41aux::Card& card, int n_ranks);
    ~Control();
    Control(const Control&) = delete;
    Control& operator=(const Control&) = delete;

    /* ---- the controller's startup (Engine::aux_discover; work/aux-dyn/SPEC.md A.5 "startup") ----
     * init_share: the probe, the allow list, the control file, the holders' directory and the state file (with its
     * lock held for the process's life). < 0 with *why only when the state file cannot be written.
     * may_reserve: the mode (holders, the control file) and the probe's verdict before any reserve: false (with *why)
     * for a released start. watch: writes `starting` (or `released` for a released start) and starts the thread,
     * which watches the probe, the holders and the control file from now on; `reserving` true while the caller
     * reserves the arena. reserved: after the reserve; false when the watch asked for a release meanwhile (the
     * caller frees the arena: a released start). claim (Engine::aux_start, before its load): false when the watch
     * released the arena since; otherwise the watch may only ask the load to stop (load_abort). */
    int  init_share(const Roots& roots, long pid, std::string* why);
    bool may_reserve(std::string* why);
    void watch(Worker* w, bool reserving);
    bool reserved(bool ok);
    bool claim(std::string* why);
    const std::atomic<bool>* load_abort() const { return &load_abort_; }

    /* ---- attach (Engine::aux_start, every stager bound) ----
     * `released`: started released (nothing loaded), `why` the reason; otherwise the views hold the startup load.
     * < 0 with *err: the worker goes off. */
    int  attach(Worker* w, Hooks hooks, std::vector<RankUnits> units, int layers, int64_t slots,
                const ds41aux::Layout& L, bool released, const std::string& why, std::string* err);
    /* The thread in its attached loop (started by watch() already, or now). */
    void start();
    /* Bounded (the copies' and drains' bounds): the thread joined, the swap queue probed; a queue that did not drain
     * keeps the arena (Worker::set_swap_undrained). Idempotent. */
    void stop();
    /* The step boundary (§3.2), on the scheduler thread with every rank parked; never waits. */
    void boundary(bool idle);
    Gate* gate() { return gate_.get(); }
    /* The startup lines (A.6, B.8), after the "on:" / "released at start" line; the shutdown lines. */
    std::vector<std::string> startup_lines() const;
    std::vector<std::string> report() const;
    /* The state file's last word at teardown: `stopped` (after Worker::finish). */
    void write_final(const std::string& state, const std::string& reason);
    /* For tests: the slot pool's and the mailbox's state. */
    struct Snapshot {
        int64_t free = 0, filling = 0, ready = 0, resident = 0, evicting = 0, lost = 0;
        uint64_t epoch = 0, swaps = 0, fills = 0, releases = 0, acquires = 0, readbacks = 0, differ = 0;
        std::string state;
        bool online_stopped = false;
    };
    Snapshot snapshot();
    /* Tests: the probe's roots and a clock offset are fixed at init_share; this forces a poll now. */
    void test_poll_now() { force_poll_.store(true, std::memory_order_release); }

private:
    using Mode = ds41aux::share::Mode;
    using St = ds41aux::share::St;
    struct Slot {
        SlotState st = SlotState::kFree;
        int8_t    rank = -1;
        int32_t   cand = -1;
        uint64_t  evict_epoch = 0;
        int8_t    prev_rank = -1;   /* the unit whose entries may still name this slot (nulled at reuse) */
        int32_t   prev_cand = -1;
    };
    struct Flip {
        int8_t  rank = -1;
        int32_t in = -1, out = -1, slot = -1;   /* candidate indices; out -1: a fill */
        uint64_t seq = 0;                       /* the post that carried it */
        int64_t  t_dec_ns = 0;                  /* when it was decided */
    };
    struct Applied {
        uint64_t seq = 0, epoch = 0;
    };
    struct Mail {   /* under mail_mu_ */
        bool                      reset = false;
        bool                      clear = false;    /* every unit out of the views, tables kept (online alone: dead) */
        std::vector<const void*>  gu_tab, dn_tab;   /* rank x layer: the new arena's tables */
        std::vector<Flip>         pending;          /* control -> boundary */
        bool                      open = false;
        uint64_t                  post_seq = 0;
        std::vector<Applied>      applied;          /* boundary -> control */
        uint64_t                  epoch = 0;
        std::vector<uint64_t>     quiet_epoch;      /* by rank: the last boundary its lane was idle at */
        std::vector<uint8_t>      excluded;         /* by rank: 1 the lane died, 2 the guard tripped */
        uint32_t                  quiet_mask = 0;   /* ranks with slots awaiting their lane idle */
    };
    struct Inbox {   /* the hit hand-off: the boundary fills it when !full, the control thread empties it */
        std::vector<std::vector<uint32_t>> hits;
        std::vector<uint64_t>              tok;
        bool                               full = false;
    };
    enum Kind : int { kFill = 0, kSwap = 1, kTables = 2 };
    struct Job {
        int      kind = kFill;
        int      rank = -1;
        int32_t  cin = -1, cout = -1, slot = -1;
        bool     sampled = false;
        int64_t  t_dec_ns = 0;
    };
    struct InFlight {
        bool     used = false;
        Job      job;
        int64_t  t0_ns = 0, gu_n = 0, dn_n = 0;
        uint64_t gen = 0;
    };
    static constexpr int kBounce = 4;

    void loop();
    void tune_thread();
    void poll_share(int64_t now_ms);
    bool service(int64_t now_ms);
    void do_release(const ds41aux::share::Step& st);
    void do_acquire(const ds41aux::share::Step& st);
    bool drain_for_release(std::string* why, int64_t* gate_us, int64_t* lanes_ms, int64_t* swap_ms);
    bool free_after_drain();   /* the arena freed, the pool reset; false: not drained (dead held) */
    void take_reports();
    void merge_and_plan(int64_t now_ns);
    void plan_fills(int64_t now_ns);
    bool executor_step(int64_t now_ns);
    int  start_copy(const Job& j, int b, int64_t now_ns);   /* 1 started, 0 dropped (not cached), < 0 failed */
    bool finish_copy(int b, int64_t now_ns, bool post_flip = true);   /* false: a read-back that differs */
    bool wait_copies(int64_t per_ns);                       /* every copy in flight seen complete */
    bool sync_copy(const Job& j, std::string* why);         /* one copy started and waited for (the acquire) */
    void post(const std::vector<Flip>& flips, bool reset, bool open, bool clear = false);
    void stop_online(const std::string& why);
    void dead(bool freed, const std::string& why);
    void publish(const std::string& state, const std::string& reason);
    void want_update();   /* under mail_mu_ */
    int  free_slot();
    int64_t table_word(int r, int32_t cand, int which) const;
    bool rank_ok(int r) const;
    int64_t fills_pending(int r) const;   /* fills queued, copying, or posted and not yet applied */
    bool full(int r) const;               /* its quota resident, or its list exhausted with nothing pending */
    void note_state();
    void snap();
    std::string clients_str() const;
    int64_t self_mib() const;
    std::string state_word() const;

    Config         cfg_;
    ds41aux::Card  card_;
    int            R_ = 0;
    bool           ctl_ = false, online_ = false;
    long           pid_ = 0;
    unsigned long long starttime_ = 0;
    /* sharing */
    Roots                         roots_;
    ds41aux::share::Probe         probe_;
    ds41aux::share::Allow         allow_;
    ds41aux::share::ControlFile   cfile_;
    std::string                   holders_dir_, state_path_, lock_path_;
    int                           lock_fd_ = -1;
    ds41aux::share::Fsm           fsm_;
    Mode                          mode_ = Mode::kAuto;
    std::string                   mode_src_ = "default";
    std::string                   last_published_;
    ds41aux::share::Poll          last_poll_;
    int64_t                       next_poll_ms_ = 0;
    std::atomic<bool>             force_poll_{ false };
    /* startup hand-off (under start_mu_) */
    std::mutex                    start_mu_;
    enum Phase : int { kNoArena = 0, kReserving, kReserved, kLoading, kAttached };
    Phase                         phase_ = kNoArena;
    bool                          early_release_ = false;
    std::string                   early_why_;
    std::atomic<bool>             load_abort_{ false };
    std::atomic<bool>             attached_{ false };
    /* the worker, its units */
    Worker*                       w_ = nullptr;
    Hooks                         hooks_;
    std::vector<RankUnits>        units_;
    std::vector<std::vector<int32_t>> cidx_;   /* rank x plan unit: candidate index, -1 */
    int                           layers_ = 0;
    int64_t                       n_slots_ = 0;
    ds41aux::Layout               L_{};
    std::unique_ptr<Gate>         gate_;
    /* the slot pool (control thread) */
    std::vector<Slot>             slots_;
    std::vector<int32_t>          free_;
    std::vector<std::vector<int32_t>> slot_of_;   /* rank x cand: its slot (filling, ready or resident), -1 */
    std::vector<std::vector<uint8_t>> leaving_;   /* rank x cand: chosen to leave, not yet flipped */
    std::vector<std::vector<int32_t>> cool_;      /* rank x cand: decisions left in cool-down */
    /* rank x cand: the EVICTING slot its table entries may still name for a late job (-1 none). Such a unit is not
     * copied in again until that slot frees (its rank's lane idle): a late job that listed it reads its entries,
     * and they are not rewritten under it. */
    std::vector<std::vector<int32_t>> evict_of_;
    std::vector<int64_t>          resident_n_;    /* by rank: resident units */
    std::vector<uint64_t>         mirror_;        /* the device tables' host image */
    std::vector<int32_t>          owner_;         /* by table word: rank * 2^24 + cand of the unit it belongs to, -1 */
    std::vector<Flip>             posted_;        /* posted, not yet seen applied */
    uint64_t                      open_seq_ = 0;  /* the acquire's open, awaited */
    bool                          await_open_ = false;
    std::vector<uint8_t>          excluded_;      /* by rank, from the boundary */
    std::vector<uint8_t>          excluded_told_;
    /* fills */
    bool                          filling_ = false;
    std::vector<std::vector<int32_t>> fill_list_; /* by rank: candidates to fill, in order */
    std::vector<size_t>           fill_at_;
    int                           fill_rr_ = 0;
    int64_t                       fill_t0_ns_ = 0, fill_bytes_ = 0;
    int64_t                       acquire_t0_ns_ = 0;
    /* the mailbox */
    std::mutex                    mail_mu_;
    Mail                          mail_;
    Inbox                         inbox_;
    std::atomic<uint32_t>         want_{ 0 };
    /* the executor */
    RadStream                     swap_ = nullptr;
    RadEvent                      ev_[kBounce] = {};
    char*                         bounce_[kBounce] = {};
    char*                         back_ = nullptr;     /* the sampled read-back: a stride and the two entries */
    char*                         zero_ = nullptr;     /* 4 KiB of zeros */
    uint64_t*                     tback_ = nullptr;    /* the whole tables' read-back */
    std::vector<uint64_t>         tsnap_;              /* the mirror when that read-back was queued */
    InFlight                      fl_[kBounce];
    int                           fl_head_ = 0, fl_n_ = 0;
    std::deque<Job>               todo_;
    ds41aux::online::Bucket       swap_budget_, fill_budget_;
    bool                          swap_suspect_ = false;   /* a copy failed or never completed */
    bool                          swap_leaked_ = false;
    uint64_t                      gen_ = 0;
    /* the online set */
    std::vector<ds41aux::online::RankScores> scores_;
    ds41aux::online::Credit       credit_;
    uint64_t                      merges_ = 0, tok_since_ = 0;
    bool                          online_stopped_ = false;
    std::string                   online_why_;
    int                           sample_first_ = 0;   /* read back every copy until this many after an acquire */
    std::vector<uint32_t>         delta_;
    /* the thread */
    std::thread                   th_;
    std::atomic<bool>             stop_{ false };
    bool                          stopped_ = false, started_ = false, tuned_attached_ = false;
    /* counters */
    struct Stats {
        uint64_t releases[ds41aux::share::kWhyN] = {};
        uint64_t acquires = 0, refused = 0, dead = 0;
        double   rel_ms_sum = 0, rel_ms_max = 0;
        uint64_t rel_n = 0;
        double   gate_us_sum = 0, gate_us_max = 0;
        uint64_t gate_n = 0;
        int64_t  in_state_ms[6] = {};
        int64_t  state_since_ms = 0;
        int      last_state = 0;
        /* online */
        uint64_t decisions = 0, swaps = 0, swap_bytes = 0, uncached = 0, no_slot = 0, over_budget = 0;
        uint64_t fills = 0, fill_bytes = 0, readbacks = 0, differ = 0, table_checks = 0, leases = 0;
        double   copy_ms_sum = 0, copy_ms_max = 0;
        uint64_t copy_n = 0;
        double   lag_ms_sum = 0, lag_ms_max = 0;
        uint64_t lag_n = 0;
        uint64_t next_line = 1000;
    } st_;
    /* the boundary's own counters (scheduler thread; read at report) */
    std::atomic<uint64_t> b_n_{ 0 }, b_busy_{ 0 }, b_ns_sum_{ 0 }, b_ns_max_{ 0 };
    double  startup_share_ = 0;
    std::string dead_word_;                        /* "dead freed" / "dead held <MiB>" once dead */
    std::atomic<int> state_pub_{ 0 };              /* fsm_.state() for other threads */
    std::mutex   snap_mu_;
    Snapshot     snap_;
    int64_t      snap_ms_ = 0;
};

}  // namespace aux
}  // namespace rad
