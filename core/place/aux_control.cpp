/* aux_control.cpp -- see aux_control.h. */
#include "aux_control.h"

#include "device/device.h"
#include "rad_internal.h"
#include "../../tools/ds41-host-experts/ds41_host_experts.h"
#include "../../tools/ds41-host-experts/ds41_p1.h"

#include <dirent.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <sys/file.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstring>

namespace rad {
namespace aux {

namespace sh = ds41aux::share;
namespace on = ds41aux::online;

namespace {

int64_t mono_ns() {
    return (int64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}
int64_t mono_ms() { return mono_ns() / 1000000; }
int64_t unix_ms() {
    return (int64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch()).count();
}

/* The boundary's fast path: nothing to do unless one of these is set (§3.2). */
enum : uint32_t { kWantPost = 1, kWantHits = 2, kWantQuiet = 4, kWantLanes = 8 };
constexpr int64_t kPieceBytes = (int64_t)2 << 20;        /* a copy's pieces: a lane's job waits behind one at most */
constexpr int64_t kCopyTimeoutNs = 2'000'000'000LL;      /* a copy not complete in this long is lost */
constexpr int64_t kDrainNs = 2'000'000'000LL;            /* a lane's or the swap queue's drain at a release */
constexpr int64_t kGateWarnNs = 5'000'000'000LL;
constexpr uint64_t kTableCheckEvery = 1000;              /* swaps between whole-table read-backs */
constexpr int     kSampleFirst = 8;                       /* copies read back after each acquire */

int st_index(sh::St s) { return (int)s; }

}  // namespace

/* ------------------------------------------------------------------ construction */

Control::Control(const Config& cfg, const ds41aux::Card& card, int n_ranks)
    : cfg_(cfg), card_(card), R_(n_ranks), ctl_(cfg.controller()), online_(cfg.online),
      fsm_(cfg.release_polls, cfg.reacquire_s * 1000) {
    mail_.quiet_epoch.assign((size_t)R_, 0);
    mail_.excluded.assign((size_t)R_, 0);
    excluded_.assign((size_t)R_, 0);
    excluded_told_.assign((size_t)R_, 0);
    if (ctl_) {
        gate_.reset(new Gate);
        gate_->closed.store(true, std::memory_order_seq_cst);   /* attach() opens it when the worker serves */
    }
    st_.state_since_ms = mono_ms();
}

Control::~Control() {
    stop();
    if (!swap_leaked_) {
        for (int b = 0; b < kBounce; ++b) {
            if (ev_[b]) rad_event_destroy(ev_[b]);
            if (bounce_[b]) rad_dev_free(bounce_[b], RAD_MEM_HOST_PINNED);
        }
        if (swap_) rad_stream_destroy(swap_);
        if (back_) rad_dev_free(back_, RAD_MEM_HOST_PINNED);
        if (zero_) rad_dev_free(zero_, RAD_MEM_HOST_PINNED);
        if (tback_) rad_dev_free(tback_, RAD_MEM_HOST_PINNED);
    }
    if (lock_fd_ >= 0) ::close(lock_fd_);   /* the lock goes with it */
}

/* ------------------------------------------------------------------ the controller's startup */

int Control::init_share(const Roots& roots, long pid, std::string* why) {
    roots_ = roots;
    pid_ = pid;
    starttime_ = sh::proc_starttime(roots.procfs, pid);
    (void)allow_.set(cfg_.allow);
    std::string pw;
    if (probe_.init(roots.sysfs, roots.procfs, card_.bdf, pid, cfg_.share != ds41aux::kShareAlways, &pw) < 0)
        RAD_WARN("aux experts: sharing: %s; the card's room is unknown: no acquire until it can be read", pw.c_str());
    if (cfg_.share != ds41aux::kShareAlways && !probe_.self_ok())
        RAD_WARN("aux experts: sharing: %s; the worker stays released unless the control file says on",
                 probe_.why_self().c_str());
    if (cfg_.control.empty()) return RAD_OK;
    cfile_ = sh::ControlFile(cfg_.control);
    holders_dir_ = cfg_.control + ".d";
    state_path_ = cfg_.control + ".state";
    lock_path_ = cfg_.control + ".lock";
    if (::mkdir(holders_dir_.c_str(), 0777) != 0 && errno != EEXIST) {
        *why = fmt("%s=%s: its holders' directory %s cannot be made: %s", ds41aux::kControlEnv, cfg_.control.c_str(),
                   holders_dir_.c_str(), std::strerror(errno));
        return RAD_E_INVAL;
    }
    /* The server's lock, held for the process's life: aux_yield.sh tells a live server from a gone one by it,
     * across pid namespaces (a shared lock it can take means no server). */
    lock_fd_ = ::open(lock_path_.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    bool locked = false;
    for (int i = 0; lock_fd_ >= 0 && i < 100 && !locked; ++i) {
        locked = ::flock(lock_fd_, LOCK_EX | LOCK_NB) == 0;
        if (!locked) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!locked) {
        *why = fmt("%s=%s: %s is held by another process (one server a control path)", ds41aux::kControlEnv,
                   cfg_.control.c_str(), lock_path_.c_str());
        return RAD_E_INVAL;
    }
    if (!sh::write_atomic(state_path_, sh::state_line("starting", unix_ms(), pid_, starttime_, Mode::kAuto, "default",
                                                      "before the mode was read"))) {
        *why = fmt("%s=%s: the state file %s cannot be written", ds41aux::kControlEnv, cfg_.control.c_str(),
                   state_path_.c_str());
        return RAD_E_INVAL;
    }
    return RAD_OK;
}

void Control::publish(const std::string& state, const std::string& reason) {
    if (state_path_.empty()) return;
    const std::string key = state + "|" + sh::mode_name(mode_) + "|" + mode_src_ + "|" + reason;
    if (key == last_published_) return;
    last_published_ = key;
    (void)sh::write_atomic(state_path_, sh::state_line(state, unix_ms(), pid_, starttime_, mode_, mode_src_, reason));
}

std::string Control::state_word() const {
    switch (fsm_.state()) {
        case St::kStarting: return phase_ == kNoArena ? "released" : "starting";
        case St::kServing: return "serving";
        case St::kDraining: return "draining";
        case St::kReleased: return "released";
        case St::kLoading: return "loading";
        default: return dead_word_.empty() ? std::string("dead freed") : dead_word_;
    }
}

int64_t Control::self_mib() const { return last_poll_.self >= 0 ? last_poll_.self >> 20 : -1; }

std::string Control::clients_str() const {
    if (!last_poll_.ok_clients) return cfg_.share == ds41aux::kShareAlways ? "not watched" : "unknown";
    std::string s;
    for (const sh::Client& c : last_poll_.clients) s += (s.empty() ? "" : ", ") + sh::client_str(c);
    return s.empty() ? "none" : s;
}

bool Control::may_reserve(std::string* why) {
    const int64_t now = mono_ms();
    std::vector<std::string> h = holders_dir_.empty() ? std::vector<std::string>{} : sh::live_holders(holders_dir_);
    Mode fm = Mode::kAuto;
    std::string garbage;
    if (!cfg_.control.empty()) (void)cfile_.poll(&fm, &garbage);
    if (!garbage.empty())
        RAD_WARN("aux experts: sharing: the control file %s says '%s': not on, off or auto; ignored",
                 cfg_.control.c_str(), garbage.c_str());
    mode_ = h.empty() ? fm : Mode::kOff;
    mode_src_ = !h.empty() ? (h.size() == 1 ? "holder " + h[0] : fmt("%zu holders", h.size()))
                           : cfg_.control.empty() ? "default" : "control file";
    last_poll_ = probe_.poll(now, cfg_.min_mib << 20);
    const int64_t kfd = dev_card_available(card_.ord);
    const sh::Verdict v = sh::judge(cfg_, allow_, last_poll_, /*serving=*/false, kfd);
    if (mode_ == Mode::kOff) {
        *why = "manual off (" + mode_src_ + ")";
        return false;
    }
    if (mode_ == Mode::kOn || cfg_.share == ds41aux::kShareAlways) {
        if (!v.room()) {
            *why = v.why;
            return false;
        }
        return true;
    }
    if (!v.free_for_auto()) {
        *why = v.why;
        return false;
    }
    return true;
}

void Control::watch(Worker* w, bool reserving) {
    w_ = w;
    {
        std::lock_guard<std::mutex> lk(start_mu_);
        phase_ = reserving ? kReserving : kNoArena;
        fsm_.start(reserving ? St::kStarting : St::kReleased, mono_ms(), !reserving);
        early_why_.clear();
    }
    publish(reserving ? "starting" : "released", reserving ? "the arena reserved, the load to come" : "at start");
    if (!started_) {
        started_ = true;
        th_ = std::thread([this] { loop(); });
    }
}

bool Control::reserved(bool ok) {
    std::lock_guard<std::mutex> lk(start_mu_);
    if (!ok || early_release_) {
        phase_ = kNoArena;
        if (fsm_.state() != St::kReleased) fsm_.done_release(mono_ms(), true);
        return false;
    }
    phase_ = kReserved;
    return true;
}

bool Control::claim(std::string* why) {
    std::lock_guard<std::mutex> lk(start_mu_);
    if (phase_ == kNoArena) {
        *why = early_why_.empty() ? std::string("released before the load") : early_why_;
        return false;
    }
    phase_ = kLoading;
    return true;
}

/* ------------------------------------------------------------------ attach */

int64_t Control::table_word(int r, int32_t cand, int which) const {
    const RankUnits& u = units_[(size_t)r];
    return (((int64_t)r * layers_ + u.li[(size_t)cand]) * 2 + which) * ds41aux::kExperts + u.cands[(size_t)cand].expert;
}

int Control::attach(Worker* w, Hooks hooks, std::vector<RankUnits> units, int layers, int64_t slots,
                    const ds41aux::Layout& L, bool released, const std::string& why, std::string* err) {
    std::lock_guard<std::mutex> lk(start_mu_);
    w_ = w;
    hooks_ = std::move(hooks);
    units_ = std::move(units);
    layers_ = layers;
    n_slots_ = slots;
    L_ = L;
    if ((int)units_.size() != R_ || R_ > kMaxGateRanks) {
        *err = "the control thread's ranks";
        return RAD_E_STATE;
    }
    cidx_.assign((size_t)R_, {});
    slot_of_.assign((size_t)R_, {});
    leaving_.assign((size_t)R_, {});
    cool_.assign((size_t)R_, {});
    evict_of_.assign((size_t)R_, {});
    resident_n_.assign((size_t)R_, 0);
    scores_.assign((size_t)R_, {});
    fill_list_.assign((size_t)R_, {});
    fill_at_.assign((size_t)R_, 0);
    owner_.assign((size_t)(ds41aux::tables_bytes(R_, layers_) / 8), -1);
    mirror_.assign(owner_.size(), 0);
    inbox_.hits.assign((size_t)R_, {});
    inbox_.tok.assign((size_t)R_, 0);
    for (int r = 0; r < R_; ++r) {
        const RankUnits& u = units_[(size_t)r];
        if (u.n_plan_units < 0 || u.li.size() != u.cands.size()) {
            *err = "the control thread's candidates";
            return RAD_E_STATE;
        }
        if (released) {
            /* A busy/off startup used Worker::plan(), never aux_load(), so its fresh views have no plan-unit
             * domain yet. Size it before the stagers run: later boundary flips must be able to publish units. */
            RankView& v = w_->view(r);
            v.resident.assign((size_t)u.n_plan_units, 0);
            v.slot.assign((size_t)u.n_plan_units, -1);
            v.gu_tab_entries.assign((size_t)layers_ * ds41aux::kExperts, nullptr);
            v.dn_tab_entries.assign((size_t)layers_ * ds41aux::kExperts, nullptr);
            v.units = 0;
            v.share = 0;
        }
        cidx_[(size_t)r].assign((size_t)u.n_plan_units, -1);
        for (size_t i = 0; i < u.cands.size(); ++i) {
            const int32_t pu = u.cands[i].unit;
            if (pu < 0 || pu >= u.n_plan_units || u.li[i] < 0 || u.li[i] >= layers_ || u.cands[i].expert < 0 ||
                u.cands[i].expert >= ds41aux::kExperts) {
                *err = fmt("rank %d candidate %zu (unit %d) is outside the plan or the tables", r, i, pu);
                return RAD_E_STATE;
            }
            cidx_[(size_t)r][(size_t)pu] = (int32_t)i;
            for (int which = 0; which < 2; ++which) owner_[(size_t)table_word(r, (int32_t)i, which)] = (r << 24) | (int32_t)i;
        }
        slot_of_[(size_t)r].assign(u.cands.size(), -1);
        leaving_[(size_t)r].assign(u.cands.size(), 0);
        cool_[(size_t)r].assign(u.cands.size(), 0);
        evict_of_[(size_t)r].assign(u.cands.size(), -1);
        if (online_) {
            scores_[(size_t)r].init(u.cands, cfg_.prior);
            inbox_.hits[(size_t)r].assign((size_t)u.n_plan_units, 0);
        }
    }
    slots_.assign((size_t)slots, Slot{});
    if (!released && w_->has_arena()) {
        mirror_ = w_->table_image();
        for (int r = 0; r < R_; ++r) {
            const RankView& v = w_->view(r);
            for (size_t i = 0; i < units_[(size_t)r].cands.size(); ++i) {
                const int32_t pu = units_[(size_t)r].cands[i].unit;
                if (!v.is_resident(pu)) continue;
                const int32_t k = v.slot[(size_t)pu];
                if (k < 0 || k >= slots) continue;
                slots_[(size_t)k].st = SlotState::kResident;
                slots_[(size_t)k].rank = (int8_t)r;
                slots_[(size_t)k].cand = (int32_t)i;
                slot_of_[(size_t)r][i] = k;
                ++resident_n_[(size_t)r];
            }
        }
        for (int r = 0; r < R_; ++r) startup_share_ += w_->view(r).share;
    }
    free_.clear();
    for (int64_t k = slots - 1; k >= 0; --k)
        if (slots_[(size_t)k].st == SlotState::kFree) free_.push_back((int32_t)k);
    /* The swap queue and its buffers (§3.5), on the extra card; pinned memory mapped there only. */
    if (rad_dev_set(card_.ord) < 0 || rad_stream_create(&swap_, 0) < 0) {
        swap_ = nullptr;
        *err = fmt("the swap queue on device %d: %s", card_.ord, rad_dev_last_error());
        return RAD_E_DEVICE;
    }
    for (int b = 0; b < kBounce; ++b) {
        if (rad_event_create(&ev_[b]) < 0) {
            ev_[b] = nullptr;
            *err = fmt("the swap queue's events: %s", rad_dev_last_error());
            return RAD_E_DEVICE;
        }
        bounce_[b] = (char*)dev_alloc_pinned_on(L_.stride + 4096, { card_.ord });
    }
    back_ = (char*)dev_alloc_pinned_on(L_.stride + 4096, { card_.ord });
    zero_ = (char*)dev_alloc_pinned_on(4096, { card_.ord });
    tback_ = (uint64_t*)dev_alloc_pinned_on(ds41aux::tables_bytes(R_, layers_), { card_.ord });
    for (int b = 0; b < kBounce; ++b)
        if (!bounce_[b]) back_ = nullptr;
    if (!back_ || !zero_ || !tback_) {
        *err = fmt("the swap queue's pinned bounce buffers (device %d only): %s", card_.ord, rad_dev_last_error());
        return RAD_E_NOMEM;
    }
    std::memset(zero_, 0, 4096);
    swap_budget_.init(cfg_.mbps);
    fill_budget_.init(cfg_.fill_mbps);
    if (gate_) gate_->closed.store(released, std::memory_order_seq_cst);
    phase_ = kAttached;
    const int64_t now = mono_ms();
    if (ctl_) {
        fsm_.attach(released ? St::kReleased : St::kServing, now);
        publish(released ? "released" : "serving", released ? why : "the startup set loaded");
    }
    w_->set_units_held(released ? 0 : -1);
    note_state();
    attached_.store(true, std::memory_order_release);
    return RAD_OK;
}

void Control::start() {
    if (!started_) {
        started_ = true;
        th_ = std::thread([this] { loop(); });
    }
}

void Control::stop() {
    if (stopped_) return;
    stopped_ = true;
    stop_.store(true, std::memory_order_release);
    if (th_.joinable()) th_.join();
    /* The swap queue: a probe behind everything queued, as a lane's drain. One that does not complete keeps the
     * arena (its copies may still write it) and its own buffers, to the process exit. */
    if (swap_ && ev_[0]) {
        bool ok = !swap_suspect_;
        if (ok) {
            ok = rad_event_record(ev_[0], swap_) >= 0;
            const int64_t end = mono_ns() + kDrainNs;
            while (ok) {
                const int q = rad_event_query(ev_[0]);
                if (q == 1) break;
                if (q < 0 || mono_ns() >= end) ok = false;
                else std::this_thread::sleep_for(std::chrono::microseconds(500));
            }
        }
        if (!ok) {
            swap_leaked_ = true;
            if (w_) w_->set_swap_undrained();
        }
    }
    if (w_ && ctl_) w_->set_release_drain(true);
    if (attached_.load(std::memory_order_acquire)) snap();   /* the last state for a reader (the thread is gone) */
}

void Control::write_final(const std::string& state, const std::string& reason) { publish(state, reason); }

/* ------------------------------------------------------------------ the thread */

void Control::tune_thread() {
    /* Low priority, and off the expert pool's cores (its workers are pinned one a core, named rad-hx-N): the copies
     * and the probe never take a core the CPU path computes on. */
    (void)setpriority(PRIO_PROCESS, (id_t)syscall(SYS_gettid), 10);
    cpu_set_t pool;
    CPU_ZERO(&pool);
    int pinned = 0;
    if (DIR* d = opendir("/proc/self/task")) {
        while (dirent* e = readdir(d)) {
            if (e->d_name[0] == '.') continue;
            std::string comm;
            if (!sh::read_file(std::string("/proc/self/task/") + e->d_name + "/comm", &comm, 64)) continue;
            if (comm.compare(0, 7, "rad-hx-") != 0) continue;
            cpu_set_t s;
            CPU_ZERO(&s);
            if (sched_getaffinity((pid_t)std::atoi(e->d_name), sizeof s, &s) != 0 || CPU_COUNT(&s) != 1) continue;
            CPU_OR(&pool, &pool, &s);
            ++pinned;
        }
        closedir(d);
    }
    if (!pinned) return;
    cpu_set_t mine;
    CPU_ZERO(&mine);
    if (sched_getaffinity(0, sizeof mine, &mine) != 0) return;
    cpu_set_t rest;
    CPU_ZERO(&rest);
    for (int c = 0; c < CPU_SETSIZE; ++c)
        if (CPU_ISSET(c, &mine) && !CPU_ISSET(c, &pool)) CPU_SET(c, &rest);
    if (CPU_COUNT(&rest) > 0) (void)sched_setaffinity(0, sizeof rest, &rest);
}

void Control::loop() {
    pthread_setname_np(pthread_self(), "rad-aux-ctl");
    tune_thread();
    (void)rad_dev_set(card_.ord);
    while (!stop_.load(std::memory_order_acquire)) {
        const int64_t now = mono_ms();
        bool busy = false;
        const bool att = attached_.load(std::memory_order_acquire);
        if (att && !tuned_attached_) {   /* the pool started since: off its cores */
            tuned_attached_ = true;
            tune_thread();
        }
        if (ctl_ && (now >= next_poll_ms_ || force_poll_.exchange(false))) {
            poll_share(now);
            next_poll_ms_ = now + cfg_.poll_ms;
        }
        if (att) busy = service(now);
        const int64_t until = ctl_ ? std::max<int64_t>(1, next_poll_ms_ - mono_ms()) : 10;
        std::this_thread::sleep_for(std::chrono::milliseconds(busy ? 1 : std::min<int64_t>(until, 10)));
    }
}

/* ------------------------------------------------------------------ sharing: a poll */

void Control::poll_share(int64_t now) {
    std::vector<std::string> h = holders_dir_.empty() ? std::vector<std::string>{} : sh::live_holders(holders_dir_);
    Mode fm = Mode::kAuto;
    std::string garbage;
    if (!cfg_.control.empty()) (void)cfile_.poll(&fm, &garbage);
    if (!garbage.empty())
        RAD_WARN("aux experts: sharing: the control file %s says '%s': not on, off or auto; ignored",
                 cfg_.control.c_str(), garbage.c_str());
    const Mode m = h.empty() ? fm : Mode::kOff;
    const std::string src = !h.empty() ? (h.size() == 1 ? "holder " + h[0] : fmt("%zu holders", h.size()))
                                       : cfg_.control.empty() ? "default" : "control file";
    std::unique_lock<std::mutex> lk(start_mu_);
    /* attach() publishes the startup state under this mutex too; its mode/source reads must not race a poll. */
    const bool mode_changed = m != mode_ || src != mode_src_;
    mode_ = m;
    mode_src_ = src;
    const St s0 = fsm_.state();
    const bool serving_like = s0 == St::kServing || s0 == St::kLoading || s0 == St::kStarting || s0 == St::kDraining;
    sh::Poll p = probe_.poll(now, cfg_.min_mib << 20);
    for (const sh::Client& c : p.appeared)
        RAD_INFO("aux experts: sharing: client %s appeared (%s)", sh::client_str(c).c_str(), sh::st_name(s0));
    for (const sh::Client& c : p.gone)
        RAD_INFO("aux experts: sharing: client %s (pid %ld) gone (%s)", c.comm.c_str(), c.pid, sh::st_name(s0));
    last_poll_ = p;
    const int64_t kfd = serving_like ? -1 : (hooks_.kfd_available ? hooks_.kfd_available() : dev_card_available(card_.ord));
    const sh::Verdict v = sh::judge(cfg_, allow_, p, serving_like && !(s0 == St::kStarting && phase_ == kNoArena), kfd);
    if (phase_ != kAttached) {
        /* The startup watch: a release before aux_start attached (no acquire before then). */
        if (phase_ == kNoArena) {
            if (mode_changed) publish("released", early_why_.empty() ? "at start" : early_why_);
            return;
        }
        const sh::Step st = fsm_.step(now, v, m);
        if (st.act == sh::Act::kRelease) {
            early_why_ = st.reason;
            ++st_.releases[st.why];
            if (phase_ == kReserving) {
                early_release_ = true;
            } else if (phase_ == kReserved) {
                const int64_t t0 = mono_ns();
                w_->release_arena();
                phase_ = kNoArena;
                fsm_.done_release(now, true);
                RAD_INFO("aux experts: releasing (%s) before the load: the arena of %lld MiB freed in %.1f ms",
                         st.reason.c_str(), (long long)cfg_.cap_mib, (double)(mono_ns() - t0) * 1e-6);
                publish("released", st.reason);
            } else if (phase_ == kLoading) {
                load_abort_.store(true, std::memory_order_release);
                RAD_INFO("aux experts: releasing (%s) during the load: the load stops at its next unit",
                         st.reason.c_str());
                publish("draining", st.reason);
            }
        } else if (mode_changed) {
            publish(state_word(), "");
        }
        return;
    }
    const sh::Step st = fsm_.step(now, v, m);
    lk.unlock();
    if (st.act == sh::Act::kRelease) {
        do_release(st);
    } else if (st.act == sh::Act::kAcquire) {
        do_acquire(st);
    } else if (mode_changed) {
        const St s = fsm_.state();
        publish(state_word(), s == St::kReleased ? fmt("this process holds %lld MiB on the card", (long long)self_mib()) : "");
    }
    note_state();
    /* time in states */
    const int cur = st_index(fsm_.state());
    const int64_t t = mono_ms();
    st_.in_state_ms[st_.last_state] += t - st_.state_since_ms;
    st_.state_since_ms = t;
    st_.last_state = cur;
}

/* ------------------------------------------------------------------ release */

bool Control::drain_for_release(std::string* why, int64_t* gate_us, int64_t* lanes_ms, int64_t* swap_ms) {
    /* 1. The gate: no rank touches a lane once this returns (§3.3). */
    const int64_t g0 = mono_ns();
    if (gate_) {
        gate_->closed.store(true, std::memory_order_seq_cst);
        bool warned = false;
        while (gate_->busy(R_)) {
            if (stop_.load(std::memory_order_acquire)) {
                *why = "stopping while a rank held its lane";
                return false;
            }
            if (!warned && mono_ns() - g0 > kGateWarnNs) {
                warned = true;
                int r = 0;
                while (r < R_ && !gate_->in_lane[r].load()) ++r;
                RAD_WARN("aux experts: releasing: rank %d has held its lane for 5 s; the release waits for it", r);
            }
            std::this_thread::sleep_for(std::chrono::microseconds(20));
        }
    }
    *gate_us = (mono_ns() - g0) / 1000;
    /* 2. The swap queue: every copy in flight seen complete (2 s each). */
    const int64_t s0 = mono_ns();
    const bool swap_ok = !swap_suspect_ && wait_copies(kCopyTimeoutNs);
    *swap_ms = (mono_ns() - s0) / 1000000;
    /* 3. The lanes: the job in flight (an abandoned late one), then a probe. */
    const int64_t l0 = mono_ns();
    int stuck = 0;
    std::string lw;
    for (int r = 0; r < R_; ++r) {
        Lane* l = w_->lane(r);
        if (l && !l->drain_release(kDrainNs)) {
            ++stuck;
            lw += fmt("%s rank %d%s", lw.empty() ? "" : ",", r, l->faulted() ? " (a device fault)" : "");
        }
    }
    *lanes_ms = (mono_ns() - l0) / 1000000;
    if (!swap_ok || stuck) {
        *why = !swap_ok ? std::string("the swap queue did not drain") : "";
        if (stuck) *why += (why->empty() ? "" : "; ") + fmt("lanes not drained:%s", lw.c_str());
        return false;
    }
    return true;
}

bool Control::free_after_drain() {
    w_->release_arena();
    ++gen_;
    for (Slot& s : slots_) s = Slot{};
    free_.clear();
    for (int64_t k = n_slots_ - 1; k >= 0; --k) free_.push_back((int32_t)k);
    for (int r = 0; r < R_; ++r) {
        std::fill(slot_of_[(size_t)r].begin(), slot_of_[(size_t)r].end(), -1);
        std::fill(leaving_[(size_t)r].begin(), leaving_[(size_t)r].end(), 0);
        std::fill(evict_of_[(size_t)r].begin(), evict_of_[(size_t)r].end(), -1);
        resident_n_[(size_t)r] = 0;
    }
    std::fill(mirror_.begin(), mirror_.end(), 0);
    posted_.clear();
    todo_.clear();
    w_->set_units_held(0);
    {
        std::lock_guard<std::mutex> lk(mail_mu_);
        mail_.quiet_mask = 0;
        want_update();
    }
    return true;
}

void Control::do_release(const sh::Step& st) {
    const int64_t t0 = mono_ns();
    ++st_.releases[st.why];
    RAD_INFO("aux experts: releasing (%s%s)", st.why == sh::kWhyManual ? "" : (std::string(sh::why_name(st.why)) + ": ").c_str(),
             st.reason.c_str());
    publish("draining", st.reason);
    /* 0. Whatever the boundary has not applied yet is withdrawn: a view reset, flips, the open. */
    {
        std::lock_guard<std::mutex> lk(mail_mu_);
        mail_.reset = false;
        mail_.pending.clear();
        mail_.open = false;
        want_update();
    }
    await_open_ = false;
    filling_ = false;
    std::string why;
    int64_t gate_us = 0, lanes_ms = 0, swap_ms = 0;
    const bool ok = drain_for_release(&why, &gate_us, &lanes_ms, &swap_ms);
    st_.gate_us_sum += (double)gate_us;
    st_.gate_us_max = std::max(st_.gate_us_max, (double)gate_us);
    ++st_.gate_n;
    if (!ok) {
        fsm_.done_release(mono_ms(), false);
        dead(false, why);
        return;
    }
    free_after_drain();
    fsm_.done_release(mono_ms(), true);
    const double ms = (double)(mono_ns() - t0) * 1e-6;
    st_.rel_ms_sum += ms;
    st_.rel_ms_max = std::max(st_.rel_ms_max, ms);
    ++st_.rel_n;
    /* The counters after the free: the next poll's. */
    last_poll_ = probe_.poll(mono_ms(), cfg_.min_mib << 20);
    RAD_INFO("aux experts: released in %.1f ms (gate shut in %.3f ms, lanes drained in %lld ms, swap queue in %lld ms): "
             "the arena of %lld MiB freed; the card has %lld MiB free; this process holds %lld MiB there",
             ms, (double)gate_us * 1e-3, (long long)lanes_ms, (long long)swap_ms, (long long)cfg_.cap_mib,
             (long long)(last_poll_.free_b >= 0 ? last_poll_.free_b >> 20 : -1), (long long)self_mib());
    publish("released", fmt("%s; this process holds %lld MiB on the card", st.reason.c_str(), (long long)self_mib()));
}

void Control::dead(bool freed, const std::string& why) {
    ++st_.dead;
    if (gate_) gate_->closed.store(true, std::memory_order_seq_cst);
    else post({}, false, false, /*clear=*/true);   /* online alone: no gate; every unit leaves the views */
    filling_ = false;
    online_stopped_ = true;
    if (online_why_.empty()) online_why_ = "dead";
    const int64_t self = self_mib();
    dead_word_ = freed ? std::string("dead freed") : fmt("dead held %lld", (long long)(self >= 0 ? self : cfg_.cap_mib));
    RAD_WARN("aux experts: dead: %s; the CPU path computes every row from here on%s", why.c_str(),
             freed ? "" : "; the arena is held to the process exit (the card stays held until the server restarts)");
    note_state();
    publish(dead_word_, why);
}

/* ------------------------------------------------------------------ acquire */

bool Control::rank_ok(int r) const { return !excluded_[(size_t)r]; }

int64_t Control::fills_pending(int r) const {
    int64_t n = 0;
    for (const Job& j : todo_)
        if (j.rank == r && j.kind == kFill) ++n;
    for (int b = 0; b < kBounce; ++b)
        if (fl_[b].used && fl_[b].job.rank == r && fl_[b].job.kind == kFill) ++n;
    for (const Flip& f : posted_)
        if (f.rank == r && f.out < 0) ++n;
    return n;
}

bool Control::full(int r) const {
    if (resident_n_[(size_t)r] >= units_[(size_t)r].quota) return true;
    return fill_at_[(size_t)r] >= fill_list_[(size_t)r].size() && fills_pending(r) == 0;
}

void Control::note_state() { state_pub_.store(ctl_ ? (int)fsm_.state() : (int)St::kServing, std::memory_order_release); }

void Control::do_acquire(const sh::Step& st) {
    acquire_t0_ns_ = mono_ns();
    RAD_INFO("aux experts: acquiring (%s; clients: %s; %lld MiB free)", st.reason.c_str(), clients_str().c_str(),
             (long long)(last_poll_.free_b >= 0 ? last_poll_.free_b >> 20 : -1));
    publish("loading", st.reason);
    std::string why;
    auto refused = [&](const std::string& w, bool held) {
        ++st_.refused;
        if (held) {
            /* the arena taken: give it back (its lanes idle: nothing ran) */
            int64_t g = 0, l = 0, s = 0;
            std::string dw;
            if (!drain_for_release(&dw, &g, &l, &s)) {
                fsm_.done_acquire(mono_ms(), -1);
                dead(false, w + "; then " + dw);
                return;
            }
            free_after_drain();
        }
        fsm_.done_acquire(mono_ms(), 0);
        RAD_INFO("aux experts: acquire refused: %s; next try after %lld s free", w.c_str(),
                 (long long)(fsm_.reacquire_ms() / 1000));
        publish("released", "acquire refused: " + w);
    };
    if (w_->reserve(card_.ord, &why) < 0) return refused(why, false);
    if (w_->carve(layers_, n_slots_, L_, &why) < 0) return refused(why, true);
    ++gen_;
    std::fill(mirror_.begin(), mirror_.end(), 0);
    if (w_->load_tables_image(mirror_.data(), &why) < 0) return refused("the tables: " + why, true);
    /* 4. The fill order: the static set, each rank's candidates in the startup order up to its quota; the online
     * set, each rank's best by score (score desc, then the startup order). Cached units only. */
    for (int r = 0; r < R_; ++r) {
        std::vector<int32_t>& f = fill_list_[(size_t)r];
        f.clear();
        fill_at_[(size_t)r] = 0;
        const size_t n = units_[(size_t)r].cands.size();
        for (size_t i = 0; i < n; ++i) f.push_back((int32_t)i);
        if (online_) {
            const on::RankScores& s = scores_[(size_t)r];
            std::stable_sort(f.begin(), f.end(), [&](int32_t a, int32_t b) { return s.score((size_t)a) > s.score((size_t)b); });
        }
    }
    /* 5. A unit a rank, then its warm-up job against the CPU pool's rows (the code object on the card, the new
     * tables), the lanes the control thread's while the gate is shut. */
    std::vector<Flip> first;
    sample_first_ = kSampleFirst;
    for (int r = 0; r < R_; ++r) {
        /* The lanes are this thread's while the gate is shut: a dead one's rank takes no unit. */
        if (const DeviceLane* dl = w_->device_lane(r); dl && dl->dead()) excluded_[(size_t)r] |= 1;
        if (!rank_ok(r)) continue;
        while (fill_at_[(size_t)r] < fill_list_[(size_t)r].size() && resident_n_[(size_t)r] < units_[(size_t)r].quota) {
            const int32_t c = fill_list_[(size_t)r][fill_at_[(size_t)r]++];
            if (slot_of_[(size_t)r][(size_t)c] >= 0) continue;
            const int k = free_slot();
            if (k < 0) break;
            Job j;
            j.kind = kFill;
            j.rank = r;
            j.cin = c;
            j.slot = k;
            j.t_dec_ns = mono_ns();
            std::string cw;
            if (!sync_copy(j, &cw)) {
                if (!cw.empty()) {   /* a copy that failed: the swap queue is suspect */
                    fsm_.done_acquire(mono_ms(), -1);
                    int64_t g = 0, l = 0, s = 0;
                    std::string dw;
                    const bool dr = drain_for_release(&dw, &g, &l, &s);
                    if (dr) free_after_drain();
                    dead(dr, "the acquire's first copy: " + cw + (dr ? "" : "; " + dw));
                    return;
                }
                continue;   /* not cached: the rank's next */
            }
            const int32_t li = units_[(size_t)r].li[(size_t)c];
            if (hooks_.warm) {
                const int rc = hooks_.warm(r, units_[(size_t)r].cands[(size_t)c].unit, li, w_->table(r, li, 0),
                                           w_->table(r, li, 1), &cw);
                if (rc < 0) {
                    /* §6: a warm-up that differs or times out leaves the worker dead; its jobs completed or were
                     * abandoned, so the drains free the arena (dead freed) unless a lane faulted. */
                    fsm_.done_acquire(mono_ms(), -1);
                    int64_t g = 0, l = 0, s = 0;
                    std::string dw;
                    const bool dr = drain_for_release(&dw, &g, &l, &s);
                    if (dr) free_after_drain();
                    dead(dr, "the acquire's warm-up: " + cw + (dr ? "" : "; " + dw));
                    return;
                }
            }
            Flip f;
            f.rank = (int8_t)r;
            f.in = c;
            f.slot = k;
            f.t_dec_ns = j.t_dec_ns;
            first.push_back(f);
            break;
        }
    }
    if (w_->check_headroom(&why) < 0) return refused(why, true);
    /* 6. The view reset with the new tables, the first units, and the open: one boundary applies them all. */
    post(first, /*reset=*/true, /*open=*/true);
    await_open_ = true;
    filling_ = true;
    fill_t0_ns_ = mono_ns();
    fill_bytes_ = 0;
    w_->set_units_held(-1);
}

/* ------------------------------------------------------------------ the mailbox */

void Control::want_update() {
    uint32_t w = 0;
    if (mail_.reset || mail_.clear || mail_.open || !mail_.pending.empty()) w |= kWantPost;
    if (online_ && !inbox_.full) w |= kWantHits;
    if (mail_.quiet_mask) w |= kWantQuiet;
    if (online_) w |= kWantLanes;
    want_.store(w, std::memory_order_release);
}

void Control::post(const std::vector<Flip>& flips, bool reset, bool open, bool clear) {
    std::lock_guard<std::mutex> lk(mail_mu_);
    const uint64_t seq = ++mail_.post_seq;
    if (clear) {
        mail_.clear = true;
        mail_.pending.clear();
    }
    if (reset) {
        mail_.reset = true;
        mail_.gu_tab.assign((size_t)R_ * (size_t)layers_, nullptr);
        mail_.dn_tab.assign((size_t)R_ * (size_t)layers_, nullptr);
        for (int r = 0; r < R_; ++r)
            for (int li = 0; li < layers_; ++li) {
                mail_.gu_tab[(size_t)r * layers_ + li] = w_->table(r, li, 0);
                mail_.dn_tab[(size_t)r * layers_ + li] = w_->table(r, li, 1);
            }
    }
    for (Flip f : flips) {
        f.seq = seq;
        mail_.pending.push_back(f);
        posted_.push_back(f);
    }
    if (open) {
        mail_.open = true;
        open_seq_ = seq;
    }
    want_update();
}

void Control::boundary(bool) {
    if (!want_.load(std::memory_order_acquire) || !attached_.load(std::memory_order_acquire)) return;
    std::unique_lock<std::mutex> lk(mail_mu_, std::try_to_lock);
    if (!lk.owns_lock()) {   /* the control thread edits a list: the next boundary applies it */
        b_busy_.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const int64_t t0 = mono_ns();
    const uint64_t E = mail_.epoch + 1;
    /* 1. The hits hand-off (B.2): only when the inbox is empty, and only with decode tokens counted. */
    if (online_ && !inbox_.full && hooks_.hit_tokens && hooks_.hit_tokens(0) > 0) {
        for (int r = 0; r < R_; ++r) inbox_.tok[(size_t)r] = hooks_.swap_hits(r, &inbox_.hits[(size_t)r]);
        inbox_.full = true;
    }
    /* 2. The lanes: idle (no job in flight) and alive, each polled holding the parked rank's in_lane entry; a gate
     * shut meanwhile (a release) skips them all. */
    const bool lanes = online_ || mail_.quiet_mask;
    for (int r = 0; lanes && r < R_; ++r) {
        if (!online_ && !(mail_.quiet_mask & (1u << r))) continue;
        if (gate_) {
            if (gate_->in_lane[r].load(std::memory_order_seq_cst)) {   /* a parked rank holds it: a stager bug */
                b_busy_.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            if (!gate_->enter(r)) break;
        }
        Lane* l = w_->lane(r);
        const Lane::State s = l ? l->state() : Lane::kDead;
        if (gate_) gate_->leave(r);
        if (s == Lane::kReady || s == Lane::kBackoff) mail_.quiet_epoch[(size_t)r] = E;
        if (s == Lane::kDead) mail_.excluded[(size_t)r] |= 1;
        if (hooks_.guard_tripped && hooks_.guard_tripped(r)) mail_.excluded[(size_t)r] |= 2;
    }
    /* 4. A view reset (an acquire): no unit, the new arena's tables; a clear: no unit, the tables kept. */
    bool applied = false;
    if (mail_.clear) {
        for (int r = 0; r < R_; ++r) {
            RankView& v = w_->view(r);
            std::fill(v.resident.begin(), v.resident.end(), (uint8_t)0);
            std::fill(v.slot.begin(), v.slot.end(), -1);
            v.units = 0;
        }
        mail_.clear = false;
        applied = true;
    }
    if (mail_.reset) {
        for (int r = 0; r < R_; ++r) {
            RankView& v = w_->view(r);
            std::fill(v.resident.begin(), v.resident.end(), (uint8_t)0);
            std::fill(v.slot.begin(), v.slot.end(), -1);
            v.units = 0;
            for (int li = 0; li < layers_ && (size_t)li < v.gu_tab.size(); ++li) {
                v.gu_tab[(size_t)li] = mail_.gu_tab[(size_t)r * layers_ + li];
                v.dn_tab[(size_t)li] = mail_.dn_tab[(size_t)r * layers_ + li];
            }
        }
        mail_.reset = false;
        applied = true;
    }
    /* 5. The flips: both units of a swap change for the next pass. */
    for (const Flip& f : mail_.pending) {
        RankView& v = w_->view(f.rank);
        const RankUnits& u = units_[(size_t)f.rank];
        const int32_t pin = u.cands[(size_t)f.in].unit;
        if ((size_t)pin < v.resident.size()) {
            v.resident[(size_t)pin] = 1;
            v.slot[(size_t)pin] = f.slot;
            ++v.units;
        }
        if (f.out >= 0) {
            const int32_t pout = u.cands[(size_t)f.out].unit;
            if ((size_t)pout < v.resident.size() && v.resident[(size_t)pout]) {
                v.resident[(size_t)pout] = 0;
                v.slot[(size_t)pout] = -1;
                --v.units;
            }
        }
        applied = true;
    }
    mail_.pending.clear();
    /* 6. The open, after the reset and the flips. */
    if (mail_.open) {
        if (gate_) gate_->closed.store(false, std::memory_order_seq_cst);
        mail_.open = false;
        applied = true;
    }
    if (applied) mail_.applied.push_back(Applied{ mail_.post_seq, E });
    mail_.epoch = E;
    want_update();
    const uint64_t ns = (uint64_t)(mono_ns() - t0);
    b_n_.fetch_add(1, std::memory_order_relaxed);
    b_ns_sum_.fetch_add(ns, std::memory_order_relaxed);
    if (ns > b_ns_max_.load(std::memory_order_relaxed)) b_ns_max_.store(ns, std::memory_order_relaxed);
}

void Control::take_reports() {
    std::vector<Applied> applied;
    std::vector<uint64_t> quiet;
    std::vector<uint8_t> excl;
    {
        std::lock_guard<std::mutex> lk(mail_mu_);
        applied.swap(mail_.applied);
        quiet = mail_.quiet_epoch;
        excl = mail_.excluded;
    }
    /* Flips seen applied: in units resident, out units evicting from that epoch. */
    for (const Applied& a : applied) {
        size_t keep = 0;
        for (size_t i = 0; i < posted_.size(); ++i) {
            const Flip& f = posted_[i];
            if (f.seq > a.seq) {
                posted_[keep++] = f;
                continue;
            }
            Slot& s = slots_[(size_t)f.slot];
            if (s.st == SlotState::kReady && s.rank == f.rank && s.cand == f.in) {
                s.st = SlotState::kResident;
                ++resident_n_[(size_t)f.rank];
            }
            if (f.out >= 0) {
                const int32_t ko = slot_of_[(size_t)f.rank][(size_t)f.out];
                leaving_[(size_t)f.rank][(size_t)f.out] = 0;
                if (ko >= 0) {
                    Slot& so = slots_[(size_t)ko];
                    so.st = SlotState::kEvicting;
                    so.evict_epoch = a.epoch;
                    so.prev_rank = so.rank;
                    so.prev_cand = so.cand;
                    slot_of_[(size_t)f.rank][(size_t)f.out] = -1;
                    evict_of_[(size_t)f.rank][(size_t)f.out] = ko;
                    --resident_n_[(size_t)f.rank];
                }
                const double lag = (double)(mono_ns() - f.t_dec_ns) * 1e-6;
                st_.lag_ms_sum += lag;
                st_.lag_ms_max = std::max(st_.lag_ms_max, lag);
                ++st_.lag_n;
            }
        }
        posted_.resize(keep);
        if (await_open_ && a.seq >= open_seq_) {
            await_open_ = false;
            if (ctl_) {
                fsm_.done_acquire(mono_ms(), 1);
                ++st_.acquires;
                int64_t n = 0;
                for (int r = 0; r < R_; ++r) n += resident_n_[(size_t)r];
                int64_t target = 0;
                for (int r = 0; r < R_; ++r) target += units_[(size_t)r].quota;
                RAD_INFO("aux experts: serving again in %.1f s: %lld units of the %lld targeted, warm-up rows equal the "
                         "CPU pool's", (double)(mono_ns() - acquire_t0_ns_) * 1e-9, (long long)n, (long long)target);
                publish("serving", "a partial set, filling");
            }
        }
    }
    /* Evicting slots whose rank's lane was idle at a boundary at or after their eviction: free. */
    uint32_t mask = 0;
    for (size_t k = 0; k < slots_.size(); ++k) {
        Slot& s = slots_[k];
        if (s.st != SlotState::kEvicting) continue;
        if (quiet[(size_t)s.rank] >= s.evict_epoch) {
            if (s.cand >= 0 && evict_of_[(size_t)s.rank][(size_t)s.cand] == (int32_t)k)
                evict_of_[(size_t)s.rank][(size_t)s.cand] = -1;
            s.st = SlotState::kFree;
            s.rank = -1;
            s.cand = -1;
            free_.push_back((int32_t)k);
        } else {
            mask |= 1u << s.rank;
        }
    }
    for (int r = 0; r < R_; ++r) {
        if (excl[(size_t)r] && !excluded_[(size_t)r]) {
            excluded_[(size_t)r] = excl[(size_t)r];
            if (online_ && !excluded_told_[(size_t)r]) {
                excluded_told_[(size_t)r] = 1;
                const Lane* l = w_->lane(r);
                RAD_WARN("%srank %d excluded: %s; its swaps stop, its evicted slots stay out of the pool",
                         ds41aux::kLogOnline, r,
                         (excl[(size_t)r] & 1) ? ("its lane died (" + (l ? l->why_dead() : std::string("?")) + ")").c_str()
                                               : "its stager's guard stopped its jobs");
            }
        }
    }
    {
        std::lock_guard<std::mutex> lk(mail_mu_);
        mail_.quiet_mask = mask;
        want_update();
    }
}

/* ------------------------------------------------------------------ the service loop */

bool Control::service(int64_t) {
    const int64_t now_ns = mono_ns();
    take_reports();
    const St s = ctl_ ? fsm_.state() : St::kServing;
    const bool arena = dead_word_.empty() && w_->has_arena() && (s == St::kServing || s == St::kLoading);
    if (online_) merge_and_plan(now_ns);
    if (arena && filling_ && !await_open_) plan_fills(now_ns);
    bool busy = arena && executor_step(now_ns);
    busy = busy || await_open_ || !posted_.empty();
    note_state();
    if (now_ns / 1000000 - snap_ms_ >= 10) {
        snap_ms_ = now_ns / 1000000;
        snap();
    }
    return busy;
}

int Control::free_slot() {
    if (free_.empty()) return -1;
    const int k = free_.back();
    free_.pop_back();
    return k;
}

void Control::plan_fills(int64_t now_ns) {
    /* Ranks interleaved, so every rank serves early; the fill budget is _SHARE_FILL_MBPS. */
    bool any = false;
    for (int n = 0; n < R_ && todo_.size() < (size_t)kBounce * 2; ++n) {
        const int r = fill_rr_;
        fill_rr_ = (fill_rr_ + 1) % R_;
        if (!rank_ok(r)) continue;
        const int64_t pending = fills_pending(r);
        while (fill_at_[(size_t)r] < fill_list_[(size_t)r].size() &&
               (slot_of_[(size_t)r][(size_t)fill_list_[(size_t)r][fill_at_[(size_t)r]]] >= 0 ||
                evict_of_[(size_t)r][(size_t)fill_list_[(size_t)r][fill_at_[(size_t)r]]] >= 0))
            ++fill_at_[(size_t)r];
        if (resident_n_[(size_t)r] + pending >= units_[(size_t)r].quota || fill_at_[(size_t)r] >= fill_list_[(size_t)r].size())
            continue;
        any = true;
        const int k = free_slot();
        if (k < 0) break;
        Job j;
        j.kind = kFill;
        j.rank = r;
        j.cin = fill_list_[(size_t)r][fill_at_[(size_t)r]++];
        j.slot = k;
        j.t_dec_ns = now_ns;
        slots_[(size_t)k].st = SlotState::kFilling;
        slots_[(size_t)k].rank = (int8_t)r;
        slots_[(size_t)k].cand = j.cin;
        slot_of_[(size_t)r][(size_t)j.cin] = k;
        todo_.push_back(j);
    }
    bool left = any;
    for (int r = 0; r < R_ && !left; ++r) left = fills_pending(r) > 0;
    if (!left) {
        filling_ = false;
        int64_t n = 0;
        for (int r = 0; r < R_; ++r) n += resident_n_[(size_t)r];
        const double secs = (double)(now_ns - fill_t0_ns_) * 1e-9;
        RAD_INFO("aux experts: full set in %.1f s (%lld units, %.1f GiB at %.0f MB/s)",
                 (double)(now_ns - acquire_t0_ns_) * 1e-9, (long long)n, (double)fill_bytes_ / (1 << 30),
                 secs > 0 ? (double)fill_bytes_ / secs * 1e-6 : 0.0);
        if (ctl_) publish("serving", "the full set");
    }
}

/* ------------------------------------------------------------------ the online set */

void Control::stop_online(const std::string& why) {
    if (online_stopped_) return;
    online_stopped_ = true;
    online_why_ = why;
    RAD_WARN("%sstopped: %s; serving goes on with the current set", ds41aux::kLogOnline, why.c_str());
}

void Control::merge_and_plan(int64_t now_ns) {
    {
        std::lock_guard<std::mutex> lk(mail_mu_);
        if (!inbox_.full) return;
    }
    const uint64_t T = inbox_.tok[0];
    const double f = on::decay_factor(T, cfg_.halflife);
    for (int r = 0; r < R_; ++r) {
        std::vector<uint32_t>& h = inbox_.hits[(size_t)r];
        const std::vector<int32_t>& ci = cidx_[(size_t)r];
        delta_.assign(units_[(size_t)r].cands.size(), 0);
        for (size_t u = 0; u < h.size() && u < ci.size(); ++u)
            if (h[u] && ci[u] >= 0) delta_[(size_t)ci[u]] += h[u];
        on::merge(&scores_[(size_t)r], delta_.data(), f);
        std::fill(h.begin(), h.end(), 0u);
    }
    {
        std::lock_guard<std::mutex> lk(mail_mu_);
        for (int r = 0; r < R_; ++r) inbox_.tok[(size_t)r] = 0;
        inbox_.full = false;
        want_update();
    }
    if (T == 0) return;
    ++merges_;
    tok_since_ += T;
    const St s = ctl_ ? fsm_.state() : St::kServing;
    const bool can = !online_stopped_ && w_->has_arena() && s == St::kServing && !await_open_ && !swap_suspect_;
    if (!can) {
        tok_since_ = 0;
        return;
    }
    if (merges_ % (uint64_t)cfg_.every) return;
    int n = credit_.grant(tok_since_, cfg_.rate, cfg_.omax);
    tok_since_ = 0;
    ++st_.decisions;
    for (int r = 0; r < R_; ++r)
        for (int32_t& c : cool_[(size_t)r])
            if (c > 0) --c;
    const int nfree = (int)free_.size();
    if (n > nfree) {
        ++st_.no_slot;
        n = nfree;
    }
    if (n <= 0) return;
    /* The planner's states (B.4): resident units that may leave, others that may come in. A rank still filling,
     * excluded (its lane dead or its guard tripped) or holding a slot that cannot free is left out. */
    std::vector<std::vector<uint8_t>> state((size_t)R_);
    std::vector<on::RankIn> in((size_t)R_);
    for (int r = 0; r < R_; ++r) {
        const size_t nc = units_[(size_t)r].cands.size();
        std::vector<uint8_t>& stv = state[(size_t)r];
        stv.assign(nc, on::kNon);
        for (size_t i = 0; i < nc; ++i) {
            const int32_t k = slot_of_[(size_t)r][i];
            if (k >= 0) stv[i] = slots_[(size_t)k].st == SlotState::kResident && !leaving_[(size_t)r][i] ? on::kRes : on::kPinned;
            else if (cool_[(size_t)r][i] > 0 || evict_of_[(size_t)r][i] >= 0) stv[i] = on::kPinned;
        }
        in[(size_t)r].s = &scores_[(size_t)r];
        in[(size_t)r].cand = &units_[(size_t)r].cands;
        in[(size_t)r].state = stv.data();
        in[(size_t)r].ok = rank_ok(r) && full(r);
    }
    const std::vector<on::Pair> pairs = on::plan(in, n, cfg_.margin);
    for (const on::Pair& p : pairs) {
        const int k = free_slot();
        if (k < 0) break;
        /* §3.5: the entries the copy writes belong to the in unit, which no view lists (I4). */
        const int64_t wg = table_word(p.rank, p.in, 0), wd = table_word(p.rank, p.in, 1);
        if (owner_[(size_t)wg] != ((p.rank << 24) | p.in) || owner_[(size_t)wd] != ((p.rank << 24) | p.in) ||
            slot_of_[(size_t)p.rank][(size_t)p.in] >= 0 || evict_of_[(size_t)p.rank][(size_t)p.in] >= 0) {
            free_.push_back(k);
            stop_online(fmt("rank %d candidate %d: its table entries are not its own or it holds a slot", p.rank, p.in));
            return;
        }
        Job j;
        j.kind = kSwap;
        j.rank = p.rank;
        j.cin = p.in;
        j.cout = p.out;
        j.slot = k;
        j.t_dec_ns = now_ns;
        slots_[(size_t)k].st = SlotState::kFilling;
        slots_[(size_t)k].rank = (int8_t)p.rank;
        slots_[(size_t)k].cand = p.in;
        slot_of_[(size_t)p.rank][(size_t)p.in] = k;
        leaving_[(size_t)p.rank][(size_t)p.out] = 1;
        todo_.push_back(j);
    }
    credit_.spend((int)pairs.size());
}

/* ------------------------------------------------------------------ the executor (§3.5) */

int Control::start_copy(const Job& j, int b, int64_t now_ns) {
    InFlight& f = fl_[b];
    const RankUnits& u = units_[(size_t)j.rank];
    const int32_t pu = u.cands[(size_t)j.cin].unit;
    const int32_t li = u.li[(size_t)j.cin], e = u.cands[(size_t)j.cin].expert;
    Hooks::Source src;
    if (!hooks_.source || !hooks_.source(j.rank, pu, &src)) return 0;
    ++st_.leases;
    if (src.gu_n <= 0 || src.dn_n <= 0 || src.gu_n > L_.gu_slot || src.dn_n > L_.dn_slot) {
        if (hooks_.release) hooks_.release(j.rank, &src);
        return 0;
    }
    char* const bb = bounce_[b];
    ds41hx::copy_nt((uint8_t*)bb, src.gu, (size_t)src.gu_n);
    ds41hx::copy_nt((uint8_t*)bb + L_.dn_off, src.dn, (size_t)src.dn_n);
    const int64_t gu_n = src.gu_n, dn_n = src.dn_n;
    if (hooks_.release) hooks_.release(j.rank, &src);
    char* const slot = w_->slot(j.slot);
    uint64_t* tail = (uint64_t*)(bb + L_.stride);
    tail[0] = (uint64_t)(uintptr_t)ds41p1::mark(slot, ds41p1::kMarkPacked);
    tail[1] = (uint64_t)(uintptr_t)ds41p1::mark(slot + L_.dn_off, ds41p1::kMarkPacked);
    int rc = RAD_OK;
    /* (a) The slot's previous unit's entries, when they still name this slot: zero. */
    Slot& s = slots_[(size_t)j.slot];
    if (s.prev_rank >= 0 && s.prev_cand >= 0) {
        for (int which = 0; which < 2 && rc >= 0; ++which) {
            const int64_t w = table_word(s.prev_rank, s.prev_cand, which);
            if (mirror_[(size_t)w] != (uint64_t)(uintptr_t)ds41p1::mark(slot + (which ? L_.dn_off : 0), ds41p1::kMarkPacked))
                continue;
            const int32_t pk = slot_of_[(size_t)s.prev_rank][(size_t)s.prev_cand];
            if (pk >= 0) continue;   /* resident elsewhere: its entries name that slot, not this one */
            rc = rad_memcpy_async(w_->table(s.prev_rank, units_[(size_t)s.prev_rank].li[(size_t)s.prev_cand], which) +
                                      (int64_t)units_[(size_t)s.prev_rank].cands[(size_t)s.prev_cand].expert * 8,
                                  zero_, 8, swap_);
            if (rc >= 0) mirror_[(size_t)w] = 0;
        }
        s.prev_rank = -1;
        s.prev_cand = -1;
    }
    /* (b) The bounce to the slot in pieces of at most 2 MiB. */
    auto pieces = [&](int64_t off, int64_t n) {
        for (int64_t at = 0; at < n && rc >= 0; at += kPieceBytes)
            rc = rad_memcpy_async(slot + off + at, bb + off + at, std::min(kPieceBytes, n - at), swap_);
    };
    pieces(0, gu_n);
    pieces(L_.dn_off, dn_n);
    /* (c) The unit's two entries. */
    char* const tg = w_->table(j.rank, li, 0) + (int64_t)e * 8;
    char* const td = w_->table(j.rank, li, 1) + (int64_t)e * 8;
    if (rc >= 0) rc = rad_memcpy_async(tg, &tail[0], 8, swap_);
    if (rc >= 0) rc = rad_memcpy_async(td, &tail[1], 8, swap_);
    if (rc >= 0) {
        mirror_[(size_t)table_word(j.rank, j.cin, 0)] = tail[0];
        mirror_[(size_t)table_word(j.rank, j.cin, 1)] = tail[1];
    }
    /* (d) Sampled: the slot and the entries read back. */
    if (rc >= 0 && j.sampled) {
        rc = rad_memcpy_async(back_, slot, gu_n, swap_);
        if (rc >= 0) rc = rad_memcpy_async(back_ + L_.dn_off, slot + L_.dn_off, dn_n, swap_);
        if (rc >= 0) rc = rad_memcpy_async(back_ + L_.stride, tg, 8, swap_);
        if (rc >= 0) rc = rad_memcpy_async(back_ + L_.stride + 8, td, 8, swap_);
    }
    /* (e) The bounce's event. */
    if (rc >= 0) rc = rad_event_record(ev_[b], swap_);
    f.used = true;
    f.job = j;
    f.t0_ns = now_ns;
    f.gu_n = gu_n;
    f.dn_n = dn_n;
    f.gen = gen_;
    if (rc < 0) {
        /* Some of the job's copies may be queued: its slot is lost and the queue suspect (a release then cannot
         * free the arena: dead held). The bounce is not tracked in flight. */
        f.used = false;
        swap_suspect_ = true;
        return -1;
    }
    return 1;
}

bool Control::finish_copy(int b, int64_t now_ns, bool post_flip) {
    InFlight& f = fl_[b];
    const Job j = f.job;
    f.used = false;
    const double ms = (double)(now_ns - f.t0_ns) * 1e-6;
    st_.copy_ms_sum += ms;
    st_.copy_ms_max = std::max(st_.copy_ms_max, ms);
    ++st_.copy_n;
    if (j.kind == kTables) {
        ++st_.table_checks;
        size_t at = 0;
        while (at < tsnap_.size() && tback_[at] == tsnap_[at]) ++at;
        if (at < tsnap_.size()) {
            RAD_WARN("%sthe tables read back differ from the control thread's image at word %zu (0x%016llx, the image "
                     "0x%016llx)", ds41aux::kLogOnline, at, (unsigned long long)tback_[at], (unsigned long long)tsnap_[at]);
            return false;
        }
        return true;
    }
    Slot& s = slots_[(size_t)j.slot];
    if (j.sampled) {
        ++st_.readbacks;
        const char* bb = bounce_[b];
        const uint64_t* tail = (const uint64_t*)(bb + L_.stride);
        const uint64_t* got = (const uint64_t*)(back_ + L_.stride);
        int64_t at = -1;
        if (std::memcmp(back_, bb, (size_t)f.gu_n) != 0)
            for (int64_t i = 0; i < f.gu_n && at < 0; ++i)
                if (back_[i] != bb[i]) at = i;
        if (at < 0 && std::memcmp(back_ + L_.dn_off, bb + L_.dn_off, (size_t)f.dn_n) != 0)
            for (int64_t i = 0; i < f.dn_n && at < 0; ++i)
                if (back_[L_.dn_off + i] != bb[L_.dn_off + i]) at = L_.dn_off + i;
        const bool entries = got[0] == tail[0] && got[1] == tail[1] &&
                             got[0] == mirror_[(size_t)table_word(j.rank, j.cin, 0)] &&
                             got[1] == mirror_[(size_t)table_word(j.rank, j.cin, 1)];
        if (at >= 0 || !entries) {
            ++st_.differ;
            s.st = SlotState::kLost;
            slot_of_[(size_t)j.rank][(size_t)j.cin] = -1;
            if (j.cout >= 0) leaving_[(size_t)j.rank][(size_t)j.cout] = 0;
            RAD_WARN("%srank %d unit %d slot %d: %s", ds41aux::kLogOnline, j.rank, units_[(size_t)j.rank].cands[(size_t)j.cin].unit,
                     j.slot, at >= 0 ? fmt("the slot read back differs at byte %lld (0x%02x, written 0x%02x)", (long long)at,
                                           (unsigned)(uint8_t)back_[at], (unsigned)(uint8_t)bb[at]).c_str()
                                     : "its table entries read back differ from what was written");
            return false;
        }
    }
    if (j.kind == kFill) {
        ++st_.fills;
        st_.fill_bytes += (uint64_t)(f.gu_n + f.dn_n);
        fill_bytes_ += f.gu_n + f.dn_n;
    } else {
        ++st_.swaps;
        st_.swap_bytes += (uint64_t)(f.gu_n + f.dn_n);
    }
    if (j.kind == kSwap && cfg_.dry) {
        /* The diagnostic: copied, never flipped. The unit's entries name this slot (the next use zeroes them). */
        s.st = SlotState::kFree;
        s.prev_rank = (int8_t)j.rank;
        s.prev_cand = j.cin;
        s.rank = -1;
        s.cand = -1;
        free_.push_back(j.slot);
        slot_of_[(size_t)j.rank][(size_t)j.cin] = -1;
        leaving_[(size_t)j.rank][(size_t)j.cout] = 0;
        return true;
    }
    s.st = SlotState::kReady;
    if (!post_flip) return true;
    Flip fl;
    fl.rank = (int8_t)j.rank;
    fl.in = j.cin;
    fl.out = j.kind == kSwap ? j.cout : -1;
    fl.slot = j.slot;
    fl.t_dec_ns = j.t_dec_ns;
    post({ fl }, false, false);
    if (st_.swaps >= st_.next_line) {
        st_.next_line += 1000;
        double share = 0, share_all = 0;
        for (int r = 0; r < R_; ++r)
            for (size_t i = 0; i < units_[(size_t)r].cands.size(); ++i) {
                share_all += units_[(size_t)r].cands[i].share;
                if (slot_of_[(size_t)r][i] >= 0) share += units_[(size_t)r].cands[i].share;
            }
        RAD_INFO("%s%llu swaps so far, resident share %.4f of the ranks' %.4f (startup %.4f)", ds41aux::kLogOnline,
                 (unsigned long long)st_.swaps, share, share_all, startup_share_);
    }
    return true;
}

bool Control::executor_step(int64_t now_ns) {
    /* Completions, oldest first (the swap queue is in order). */
    while (fl_n_ > 0) {
        InFlight& f = fl_[fl_head_];
        const int q = rad_event_query(ev_[fl_head_]);
        if (q == 0 && now_ns - f.t0_ns < kCopyTimeoutNs) break;
        const int b = fl_head_;
        fl_head_ = (fl_head_ + 1) % kBounce;
        --fl_n_;
        if (q != 1) {
            /* Lost: the slot is never reused, the online set stops, the queue is suspect (a release needs its drain
             * and does not get it: dead held). */
            f.used = false;
            swap_suspect_ = true;
            if (f.job.kind != kTables) {
                slots_[(size_t)f.job.slot].st = SlotState::kLost;
                slot_of_[(size_t)f.job.rank][(size_t)f.job.cin] = -1;
                if (f.job.cout >= 0) leaving_[(size_t)f.job.rank][(size_t)f.job.cout] = 0;
            }
            stop_online(q < 0 ? "a copy on the swap queue failed" : "a copy on the swap queue did not complete in 2 s");
            filling_ = false;
            todo_.clear();
            continue;
        }
        if (f.gen != gen_) {   /* a release came in between (it waited for this copy): nothing of it stays */
            f.used = false;
            continue;
        }
        if (!finish_copy(b, now_ns)) {
            const bool tables = f.job.kind == kTables;
            stop_online(tables ? "the tables read back differ" : "a read-back differs");
            if (tables) {
                /* A table the jobs read may be wrong: the CPU path from here on. */
                if (!gate_) {
                    /* Online alone has no card-sharing gate: the rank threads still own their lanes. Ask the
                     * next boundary to clear every view and retain the arena until teardown after ranks join;
                     * this thread cannot drain/free it concurrently with their submissions. */
                    dead(false, "the tables read back differ from what was written");
                    return false;
                }
                int64_t g = 0, l = 0, s = 0;
                std::string dw;
                const bool dr = drain_for_release(&dw, &g, &l, &s);
                if (dr) free_after_drain();
                if (ctl_) fsm_.done_release(mono_ms(), false);
                dead(dr, "the tables read back differ from what was written" + (dr ? std::string() : "; " + dw));
                return false;
            }
        }
    }
    /* Starts: a free bounce, the budget, one sampled copy at a time (one read-back buffer). */
    while (!todo_.empty() && fl_n_ < kBounce && !swap_suspect_) {
        Job j = todo_.front();
        const int b = (fl_head_ + fl_n_) % kBounce;
        bool sampled_busy = false;
        for (int i = 0; i < kBounce; ++i) sampled_busy = sampled_busy || (fl_[i].used && fl_[i].job.sampled);
        const uint64_t nth = st_.swaps + st_.fills + 1;
        j.sampled = cfg_.verify || sample_first_ > 0 || (cfg_.check > 0 && nth % (uint64_t)cfg_.check == 0);
        if (j.sampled && sampled_busy) break;
        ds41aux::online::Bucket& bk = j.kind == kFill ? fill_budget_ : swap_budget_;
        if (!bk.take(now_ns, L_.gu_slot + L_.dn_slot)) {
            if (j.kind == kSwap) ++st_.over_budget;
            break;
        }
        todo_.pop_front();
        if (j.sampled && sample_first_ > 0) --sample_first_;
        const int rc = start_copy(j, b, now_ns);
        if (rc > 0) {
            ++fl_n_;
            continue;
        }
        /* Not started: the slot back, the in unit cooling down (a swap) or the rank's next (a fill). */
        Slot& s = slots_[(size_t)j.slot];
        slot_of_[(size_t)j.rank][(size_t)j.cin] = -1;
        if (j.cout >= 0) leaving_[(size_t)j.rank][(size_t)j.cout] = 0;
        if (rc == 0) {
            s.st = SlotState::kFree;
            s.rank = -1;
            s.cand = -1;
            free_.push_back(j.slot);
            ++st_.uncached;
            if (j.kind == kSwap) cool_[(size_t)j.rank][(size_t)j.cin] = on::kCoolDown;
        } else {
            s.st = SlotState::kLost;
            stop_online("a copy could not be queued on the swap queue");
            filling_ = false;
            todo_.clear();
            break;
        }
    }
    /* Every kTableCheckEvery swaps: the whole tables read back and compared with the image. */
    if (online_ && !swap_suspect_ && !online_stopped_ && st_.swaps >= (st_.table_checks + 1) * kTableCheckEvery &&
        fl_n_ < kBounce && todo_.empty()) {
        bool tables_busy = false;
        for (int i = 0; i < kBounce; ++i) tables_busy = tables_busy || (fl_[i].used && fl_[i].job.kind == kTables);
        if (!tables_busy) {
            const int b = (fl_head_ + fl_n_) % kBounce;
            tsnap_ = mirror_;
            int rc = rad_memcpy_async(tback_, w_->table(0, 0, 0), ds41aux::tables_bytes(R_, layers_), swap_);
            if (rc >= 0) rc = rad_event_record(ev_[b], swap_);
            if (rc >= 0) {
                fl_[b].used = true;
                fl_[b].job = Job{};
                fl_[b].job.kind = kTables;
                fl_[b].t0_ns = now_ns;
                fl_[b].gen = gen_;
                ++fl_n_;
            } else {
                swap_suspect_ = true;
                stop_online("the tables' read-back could not be queued");
            }
        }
    }
    return fl_n_ > 0 || !todo_.empty();
}

bool Control::wait_copies(int64_t per_ns) {
    bool ok = true;
    while (fl_n_ > 0) {
        InFlight& f = fl_[fl_head_];
        const int64_t end = mono_ns() + per_ns;
        int q = 0;
        while ((q = rad_event_query(ev_[fl_head_])) == 0 && mono_ns() < end && !stop_.load(std::memory_order_acquire))
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        if (q != 1) {
            ok = false;
            swap_suspect_ = true;
        }
        f.used = false;
        fl_head_ = (fl_head_ + 1) % kBounce;
        --fl_n_;
        if (!ok) break;
    }
    todo_.clear();
    return ok;
}

bool Control::sync_copy(const Job& j0, std::string* why) {
    Job j = j0;
    j.sampled = true;   /* the acquire's first units are read back */
    if (!wait_copies(kCopyTimeoutNs)) {
        *why = "the swap queue did not drain";
        return false;
    }
    Slot& s = slots_[(size_t)j.slot];
    s.st = SlotState::kFilling;
    s.rank = (int8_t)j.rank;
    s.cand = j.cin;
    slot_of_[(size_t)j.rank][(size_t)j.cin] = j.slot;
    const int b = fl_head_;
    const int rc = start_copy(j, b, mono_ns());
    if (rc <= 0) {
        slot_of_[(size_t)j.rank][(size_t)j.cin] = -1;
        s.st = rc == 0 ? SlotState::kFree : SlotState::kLost;
        if (rc == 0) {
            s.rank = -1;
            s.cand = -1;
            free_.push_back(j.slot);
            ++st_.uncached;
            return false;
        }
        *why = "a copy could not be queued";
        return false;
    }
    ++fl_n_;
    const int64_t end = mono_ns() + kCopyTimeoutNs;
    int q = 0;
    while ((q = rad_event_query(ev_[b])) == 0 && mono_ns() < end)
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    fl_head_ = (fl_head_ + 1) % kBounce;
    --fl_n_;
    if (q != 1) {
        fl_[b].used = false;
        swap_suspect_ = true;
        s.st = SlotState::kLost;
        slot_of_[(size_t)j.rank][(size_t)j.cin] = -1;
        *why = q < 0 ? "a copy failed" : "a copy did not complete in 2 s";
        return false;
    }
    /* Read back and compared; the acquire posts its flip with the view reset. */
    const bool same = finish_copy(b, mono_ns(), /*post_flip=*/false);
    if (!same) {
        *why = "the read-back differs";
        return false;
    }
    return true;
}

/* ------------------------------------------------------------------ lines */

std::vector<std::string> Control::startup_lines() const {
    std::vector<std::string> out;
    if (ctl_) {
        std::string pol = ds41aux::share_name(cfg_.share);
        if (cfg_.share == ds41aux::kShareAllow)
            pol += fmt(" (%s, foreign at most %lld MiB)", cfg_.allow.c_str(), (long long)cfg_.foreign_mib);
        out.push_back(fmt("%spolicy %s, clients from %lld MiB, floor %lld MiB, polls every %lld ms (a client the policy "
                          "refuses at once, the floor, the cap or an unknown probe after %lld), reacquire after %lld s "
                          "free, fill %lld MB/s, control %s, card gpu_id %u",
                          ds41aux::kLogSharing, pol.c_str(), (long long)cfg_.min_mib, (long long)cfg_.floor_mib,
                          (long long)cfg_.poll_ms, (long long)cfg_.release_polls, (long long)cfg_.reacquire_s,
                          (long long)cfg_.fill_mbps,
                          cfg_.control.empty() ? "none"
                                               : (cfg_.control + " (holders " + holders_dir_ + ", state " + state_path_ + ")").c_str(),
                          probe_.gpu_id()));
    }
    if (online_) {
        int64_t held = 0;
        for (const RankUnits& u : units_) held += u.quota;
        std::string chk = cfg_.verify ? "every swap" : cfg_.check == 0 ? "none" : cfg_.check == 1 ? "every swap"
                                                    : fmt("every %lldth swap", (long long)cfg_.check);
        out.push_back(fmt("%s%.2f swaps a token, at most %lld a decision every %lld step(s), half-life %lld tokens, prior "
                          "%lld tokens, margin %lld%%, %lld spare slots (%lld units held of %lld slots), copies at most "
                          "%lld MB/s, read-back %s%s",
                          ds41aux::kLogOnlineOn, (double)cfg_.rate / 100.0, (long long)cfg_.omax, (long long)cfg_.every,
                          (long long)cfg_.halflife, (long long)cfg_.prior, (long long)cfg_.margin, (long long)cfg_.spare,
                          (long long)held, (long long)n_slots_, (long long)cfg_.mbps, chk.c_str(),
                          cfg_.dry ? "; dry: copies and read-backs only, no unit ever flips" : ""));
    }
    return out;
}

std::vector<std::string> Control::report() const {
    std::vector<std::string> out;
    const double bn = (double)b_n_.load();
    if (ctl_) {
        uint64_t rel = 0;
        for (int i = 0; i < sh::kWhyN; ++i) rel += st_.releases[i];
        out.push_back(fmt("%s%llu releases (%llu foreign, %llu foreign cap, %llu floor, %llu unknown, %llu manual), %llu "
                          "acquires (%llu refused, %llu dead); serving %.1f s, released %.1f s, draining %.1f s, loading "
                          "%.1f s; release latency mean %.1f ms, max %.1f ms (decision to freed); gate shut mean %.1f us, "
                          "max %.1f us; boundary %.2f us mean, %.1f us max over %.0f boundaries (%llu skipped busy)",
                          ds41aux::kLogSharing, (unsigned long long)rel, (unsigned long long)st_.releases[sh::kWhyForeign],
                          (unsigned long long)st_.releases[sh::kWhyCap], (unsigned long long)st_.releases[sh::kWhyFloor],
                          (unsigned long long)st_.releases[sh::kWhyUnknown], (unsigned long long)st_.releases[sh::kWhyManual],
                          (unsigned long long)st_.acquires, (unsigned long long)st_.refused, (unsigned long long)st_.dead,
                          (double)st_.in_state_ms[(int)St::kServing] * 1e-3, (double)st_.in_state_ms[(int)St::kReleased] * 1e-3,
                          (double)st_.in_state_ms[(int)St::kDraining] * 1e-3, (double)st_.in_state_ms[(int)St::kLoading] * 1e-3,
                          st_.rel_n ? st_.rel_ms_sum / (double)st_.rel_n : 0.0, st_.rel_ms_max,
                          st_.gate_n ? st_.gate_us_sum / (double)st_.gate_n : 0.0, st_.gate_us_max,
                          bn > 0 ? (double)b_ns_sum_.load() / bn * 1e-3 : 0.0, (double)b_ns_max_.load() * 1e-3, bn,
                          (unsigned long long)b_busy_.load()));
    }
    if (online_) {
        double share = 0, share_all = 0;
        for (int r = 0; r < R_; ++r)
            for (size_t i = 0; i < units_[(size_t)r].cands.size(); ++i) {
                share_all += units_[(size_t)r].cands[i].share;
                if (slot_of_[(size_t)r][i] >= 0) share += units_[(size_t)r].cands[i].share;
            }
        out.push_back(fmt("%s%llu decisions, %llu swaps (%.1f GiB, %.2f ms copy mean, %.2f ms max; decision to flip %.1f ms "
                          "mean, %.1f ms max; %llu uncached, %llu decisions short of a free slot, %llu waits on the "
                          "budget), %llu read back (%llu differ), %llu table checks, %llu fills (%.1f GiB); resident share "
                          "%.4f of the ranks' %.4f (startup %.4f); executor leases %llu (counted among the RAM cache's "
                          "hits)%s%s",
                          ds41aux::kLogOnline, (unsigned long long)st_.decisions, (unsigned long long)st_.swaps,
                          (double)st_.swap_bytes / (1 << 30), st_.copy_n ? st_.copy_ms_sum / (double)st_.copy_n : 0.0,
                          st_.copy_ms_max, st_.lag_n ? st_.lag_ms_sum / (double)st_.lag_n : 0.0, st_.lag_ms_max,
                          (unsigned long long)st_.uncached, (unsigned long long)st_.no_slot,
                          (unsigned long long)st_.over_budget, (unsigned long long)st_.readbacks,
                          (unsigned long long)st_.differ, (unsigned long long)st_.table_checks,
                          (unsigned long long)st_.fills, (double)st_.fill_bytes / (1 << 30), share, share_all,
                          startup_share_, (unsigned long long)st_.leases, cfg_.dry ? "; dry (no flip)" : "",
                          online_stopped_ ? ("; stopped: " + online_why_).c_str() : ""));
    }
    return out;
}

Control::Snapshot Control::snapshot() {
    std::lock_guard<std::mutex> lk(snap_mu_);
    return snap_;
}

void Control::snap() {
    Snapshot s;
    for (const Slot& x : slots_) {
        switch (x.st) {
            case SlotState::kFree: ++s.free; break;
            case SlotState::kFilling: ++s.filling; break;
            case SlotState::kReady: ++s.ready; break;
            case SlotState::kResident: ++s.resident; break;
            case SlotState::kEvicting: ++s.evicting; break;
            default: ++s.lost; break;
        }
    }
    {
        std::lock_guard<std::mutex> lk(mail_mu_);
        s.epoch = mail_.epoch;
    }
    s.swaps = st_.swaps;
    s.fills = st_.fills;
    s.acquires = st_.acquires;
    for (int i = 0; i < sh::kWhyN; ++i) s.releases += st_.releases[i];
    s.readbacks = st_.readbacks;
    s.differ = st_.differ;
    s.state = ctl_ ? state_word() : "serving";
    s.online_stopped = online_stopped_;
    std::lock_guard<std::mutex> lk(snap_mu_);
    snap_ = s;
}

}  // namespace aux
}  // namespace rad
