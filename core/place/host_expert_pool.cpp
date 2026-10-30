/* host_expert_pool.cpp -- see host_expert_pool.h. */
#include "host_expert_pool.h"

#include "ds41_host_experts.h"
#include "../rad_internal.h"

#include <immintrin.h>
#include <linux/futex.h>
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace rad {

static_assert(HostExpertPool::kMaxRowsCap == ds41hx::kGemvRows, "a job's rows are one many-row kernel call");
static_assert(HostExpertPool::kMaxRowsDefault == ds41hx::kFewRows, "jobs of up to four rows can take the few-row kernel");

namespace {

constexpr const char* kEnvThreads = "RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS";
constexpr const char* kEnvSilu    = "RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_SILU";
constexpr const char* kEnvCpus    = "RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_CPUS";
constexpr const char* kEnvGbps    = "RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_GBPS";
constexpr const char* kEnvLink    = "RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_LINK_GBPS";
constexpr const char* kEnvSpin    = "RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_SPIN_US";
constexpr const char* kEnvNice    = "RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_NICE";
constexpr const char* kEnvMaxRows = "RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MAX_ROWS";
constexpr const char* kEnvLean    = "RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_LEAN_TILES";
constexpr const char* kEnvFutex   = "RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_FUTEX";
constexpr const char* kEnvPrewake = "RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_PREWAKE";
constexpr const char* kEnvPreSpin = "RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_PREWAKE_SPIN_US";
constexpr const char* kEnvPreLead = "RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_PREWAKE_LEAD_US";

/* ..._FUTEX: the futex word is the atomic's own 32 bits. */
static_assert(sizeof(std::atomic<uint32_t>) == sizeof(uint32_t) && std::atomic<uint32_t>::is_always_lock_free,
              "a futex word is a plain 32-bit integer");
long futex_op(std::atomic<uint32_t>* word, int op, uint32_t val) {
    return syscall(SYS_futex, reinterpret_cast<uint32_t*>(word), op, val, nullptr, nullptr, 0);
}

int64_t now_ns() {
    return (int64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

/* A whole-number setting into *out: `dflt` when unset or empty; false (and *out untouched) when it
 * is not a number in [lo, hi]. */
bool env_num(const char* k, long dflt, long lo, long hi, long* out) {
    const char* s = std::getenv(k);
    if (!s || !*s) { *out = dflt; return true; }
    char* e = nullptr;
    const long v = std::strtol(s, &e, 10);
    if (e == s || *e || v < lo || v > hi) return false;
    *out = v;
    return true;
}

std::atomic<HostExpertPool*> g_running{ nullptr };

/* MAX_ROWS: 1 .. cap, or 0 for a value that is not one (unset: the default). */
int parse_max_rows(int dflt, int cap) {
    const char* s = std::getenv(kEnvMaxRows);
    if (!s || !*s) return dflt;
    char* e = nullptr;
    const long v = std::strtol(s, &e, 10);
    return e != s && !*e && v >= 1 && v <= cap ? (int)v : 0;
}

/* "4-23" or "4,5,6" or "0-5,12-17"; empty on a malformed list. */
std::vector<int> parse_cpus(const char* s) {
    std::vector<int> out;
    if (!s || !*s) return out;
    const char* p = s;
    while (*p) {
        char* e = nullptr;
        const long a = std::strtol(p, &e, 10);
        if (e == p || a < 0 || a > 4095) return {};
        long b = a;
        p = e;
        if (*p == '-') {
            b = std::strtol(p + 1, &e, 10);
            if (e == p + 1 || b < a || b > 4095) return {};
            p = e;
        }
        for (long c = a; c <= b; ++c) out.push_back((int)c);
        if (*p == ',') ++p;
        else if (*p) return {};
    }
    return out;
}

double env_f(const char* k, double dflt) {
    const char* s = std::getenv(k);
    if (!s || !*s) return dflt;
    char* e = nullptr;
    const double v = std::strtod(s, &e);
    return e != s && v > 0 ? v : dflt;
}

int64_t g_spin_ns = 200000;
int     g_nice = 0;     /* workers' nice value: positive yields the cores to the drive readers */

/* A count only its own worker writes (lean_stats reads it from another thread). */
inline void bump(std::atomic<uint64_t>& c, uint64_t v = 1) {
    c.store(c.load(std::memory_order_relaxed) + v, std::memory_order_relaxed);
}

/* ..._LEAN_TILES: the lanes this worker laid out for its last tile, under their job's sequence and
 * projection (seq * 2 + down; sequences start at 1, so 0 is none). */
struct alignas(32) X8 { float f[8]; };
struct TileLanes {
    uint64_t key = 0;
    std::vector<X8> buf;
};
thread_local TileLanes t_lanes;

}  /* namespace */

thread_local HostExpertPool::WorkerCounts* HostExpertPool::t_counts_ = nullptr;

int HostExpertPool::max_rows() {
    static const int m = [] {
        const int v = parse_max_rows(kMaxRowsDefault, kMaxRowsCap);
        return v > 0 ? v : kMaxRowsDefault;
    }();
    return m;
}

HostExpertPool* HostExpertPool::get() {
    static HostExpertPool* pool = [] () -> HostExpertPool* {
        const char* t = std::getenv(kEnvThreads);
        const int threads = t && *t ? std::atoi(t) : 0;
        if (threads <= 0) return nullptr;
        if (threads > 256) {
            RAD_ERR("%s=%s: at most 256 threads; host experts off", kEnvThreads, t);
            return nullptr;
        }
        if (!__builtin_cpu_supports("avx2")) {
            RAD_ERR("%s: this CPU has no AVX2, which the host expert kernels need; host experts off",
                    kEnvThreads);
            return nullptr;
        }
        const char* path = std::getenv(kEnvSilu);
        if (!path || !*path) {
            RAD_ERR("%s is set without %s, the card's silu table; host experts off", kEnvThreads,
                    kEnvSilu);
            return nullptr;
        }
        if (parse_max_rows(kMaxRowsDefault, kMaxRowsCap) <= 0) {
            RAD_ERR("%s=%s: token rows a job, 1 .. %d; host experts off", kEnvMaxRows,
                    std::getenv(kEnvMaxRows), kMaxRowsCap);
            return nullptr;
        }
        /* The wake-up's switches (host_expert_pool.h): a value they do not take turns the pool off
         * rather than running without what was asked for. */
        long fx = 0, pw = 0, ps = 100, pl = 60;
        if (!env_num(kEnvFutex, 0, 0, 1, &fx) || !env_num(kEnvPrewake, 0, 0, 256, &pw) ||
            !env_num(kEnvPreSpin, 100, 1, 100000, &ps) || !env_num(kEnvPreLead, 60, 0, 100000, &pl)) {
            RAD_ERR("host experts: %s (0 or 1), %s (0 off, or 1 .. 256 workers), %s (1 .. 100000) or %s "
                    "(0 .. 100000) is not a value it takes; host experts off", kEnvFutex, kEnvPrewake,
                    kEnvPreSpin, kEnvPreLead);
            return nullptr;
        }
        auto* p = new HostExpertPool();
        p->futex_ = fx == 1;
        p->prewake_n_ = (int)std::min<long>(pw, threads);
        p->prewake_spin_ns_ = (int64_t)ps * 1000;
        p->prewake_lead_ns_ = (int64_t)pl * 1000;
        p->silu_.resize(65536);
        FILE* f = std::fopen(path, "rb");
        const size_t got = f ? std::fread(p->silu_.data(), sizeof(float), 65536, f) : 0;
        const bool extra = f && std::fgetc(f) != EOF;
        if (f) std::fclose(f);
        if (got != 65536 || extra) {
            RAD_ERR("%s: %s is not a silu table (65536 f32 values, 256 KiB); host experts off",
                    kEnvSilu, path);
            delete p;
            return nullptr;
        }
        /* The kernel streams ~4.2 GB/s a thread up to ~78 GB/s of DRAM at 24 threads (EPYC 7443,
         * DDR4-2133 x8); an override replaces the estimate. */
        p->cpu_bpns_ = env_f(kEnvGbps, std::min(4.2 * threads, 78.0));
        p->link_bpns_ = env_f(kEnvLink, 20.0);
        g_spin_ns = (int64_t)(env_f(kEnvSpin, 200.0) * 1000.0);
        if (const char* n = std::getenv(kEnvNice); n && *n) g_nice = std::clamp(std::atoi(n), 0, 19);
        const char* cl = std::getenv(kEnvCpus);
        const std::vector<int> cpus = parse_cpus(cl);
        if (cl && *cl && cpus.empty())
            RAD_WARN("%s=%s is not a CPU list (\"4-23\", \"4,5,6\"); the workers float", kEnvCpus, cl);
        if (const char* l = std::getenv(kEnvLean); l && !std::strcmp(l, "1")) p->lean_ = true;
        if (p->start(threads, cpus) < 0) {
            delete p;
            return nullptr;
        }
        RAD_INFO("host experts: %d CPU threads%s compute RAM-resident DeepSeek-V4.1 experts "
                 "(silu table %s; scheduling at %.0f GB/s against %.0f GB/s a copy link)",
                 threads, cpus.empty() ? "" : (" pinned to " + std::string(cl)).c_str(), path,
                 p->cpu_bpns_, p->link_bpns_);
        if (max_rows() != kMaxRowsDefault)
            RAD_INFO("host experts: jobs of up to %d token rows of one expert (%s)%s", max_rows(), kEnvMaxRows,
                     max_rows() > kMaxRowsDefault ? "; four or more through the many-row kernel" : "");
        if (p->futex_)
            RAD_INFO("host experts: idle workers sleep on a futex and a submit wakes them without a lock "
                     "(%s)", kEnvFutex);
        if (p->prewake_n_)
            RAD_INFO("host experts: pre-wake of up to %d workers before a one-token layer's CPU work, at "
                     "its router and %s; pre-woken workers spin %lld us for work (%s%s)", p->prewake_n_,
                     p->prewake_lead_ns_ > 0
                         ? fmt("%lld us before a miss read's predicted end", (long long)(p->prewake_lead_ns_ / 1000)).c_str()
                         : "not in miss reads",
                     (long long)(p->prewake_spin_ns_ / 1000), kEnvPrewake,
                     pw > threads ? fmt("; %ld asked, %d threads", pw, threads).c_str() : "");
        if (p->lean_)
            RAD_INFO("host experts: lean tiles: a worker reuses the x lanes of its last tile on a tile of the "
                     "same job and projection, and claims look only at the slots holding a job (%s)", kEnvLean);
        g_running.store(p, std::memory_order_release);
        return p;
    }();
    return pool;
}

HostExpertPool* HostExpertPool::running() { return g_running.load(std::memory_order_acquire); }

int HostExpertPool::start(int threads, const std::vector<int>& cpus) {
    slots_.reset(new (std::nothrow) Slot[kSlots]);
    if (!slots_) return RAD_E_NOMEM;
    if (lean_) {
        counts_.reset(new (std::nothrow) WorkerCounts[(size_t)threads]);
        if (!counts_) return RAD_E_NOMEM;
    }
    /* Each slot's gate_up and act rows, 64-byte aligned (a slot's rows take max_rows() * 13824 bytes:
     * 7 MB in all at four rows, 57 MB at 32). */
    const size_t per = (size_t)max_rows() * 3 * kFf;
    void* rb = nullptr;
    if (posix_memalign(&rb, 64, (size_t)kSlots * per * sizeof(uint16_t)) != 0) return RAD_E_NOMEM;
    rows_ = std::unique_ptr<uint16_t, void (*)(void*)>(static_cast<uint16_t*>(rb), std::free);
    for (int i = 0; i < kSlots; ++i) {
        slots_[i].claim.store(kDone, std::memory_order_relaxed);
        slots_[i].gate_up = rows_.get() + (size_t)i * per;
        slots_[i].act = slots_[i].gate_up + (size_t)max_rows() * 2 * kFf;
    }
    wstats_.reset(new (std::nothrow) WorkerStats[(size_t)threads]);
    if (!wstats_) return RAD_E_NOMEM;
    try {
        for (int i = 0; i < threads; ++i)
            workers_.emplace_back(&HostExpertPool::worker, this, i,
                                  cpus.empty() ? -1 : cpus[(size_t)i % cpus.size()]);
    } catch (...) {
        stop_workers();
        for (std::thread& w : workers_) w.join();
        workers_.clear();
        RAD_ERR("host experts: could not start %d threads", threads);
        return RAD_E_NOMEM;
    }
    return RAD_OK;
}

HostExpertPool::~HostExpertPool() {
    stop_workers();
    for (std::thread& w : workers_) w.join();
}

/* stop_ under idle_mu_ (a condition-variable sleeper checks it under the lock: no lost wakeup), then
 * every sleeper woken either way. */
void HostExpertPool::stop_workers() {
    {
        std::lock_guard<std::mutex> lk(idle_mu_);
        stop_ = true;
    }
    if (futex_) {
        work_gen_.fetch_add(1);
        futex_op(&work_gen_, FUTEX_WAKE_PRIVATE, INT_MAX);
    }
    idle_cv_.notify_all();
}

int HostExpertPool::submit(const Job* jobs, int n, Ticket* t) {
    if (!t || n < 0 || (n > 0 && !jobs)) return RAD_E_INVAL;
    if (n == 0) return RAD_OK;
    for (int j = 0; j < n; ++j) {
        const Job& b = jobs[j];
        if (!b.gu || !b.dn || !b.x || !b.y || (!b.gu_scales) != (!b.dn_scales) || b.more < 0 ||
            b.more >= max_rows())
            return RAD_E_INVAL;
        for (int r = 0; r < b.more; ++r)
            if (!b.xs[r] || !b.ys[r]) return RAD_E_INVAL;
        /* A packed job's slots must describe these weights: tile() reads them through the same view. */
        ds41hx::Mx v;
        if (b.packed && (b.gu_scales || b.gu_bytes <= 0 || b.dn_bytes <= 0 ||
                         !ds41hx::packed_view(b.gu, (size_t)b.gu_bytes, 2 * kFf, kEmbd, &v) ||
                         !ds41hx::packed_view(b.dn, (size_t)b.dn_bytes, kEmbd, kFf, &v)))
            return RAD_E_INVAL;
    }
    /* Counted before any job is visible, so a waiter never sees zero while one is still to come. */
    t->left.fetch_add(n, std::memory_order_relaxed);
    std::unique_lock<std::mutex> lk(submit_mu_);
    /* The wake-up's counters (stats()): whether every worker slept when this submit came. */
    const bool cold = sleepers_.load(std::memory_order_relaxed) >= threads();
    int slot = 0;
    for (int j = 0; j < n; ++j) {
        for (;;) {
            for (; slot < kSlots; ++slot)
                if (slots_[slot].state.load(std::memory_order_acquire) == 0) break;
            if (slot < kSlots) break;
            /* Every slot taken: the workers must be awake to free one, including for the jobs
             * this call has published so far. */
            wake();
            std::unique_lock<std::mutex> sl(slot_mu_);
            slot_cv_.wait_for(sl, std::chrono::microseconds(50));
            slot = 0;
        }
        Slot& s = slots_[slot];
        s.state.store(1, std::memory_order_relaxed);
        if ((uint32_t)(seq_ + 1) == 0) ++seq_;   /* a claim's sequence is never 0, `open`'s reset */
        const uint64_t seq = ++seq_;
        s.seq.store(seq, std::memory_order_relaxed);
        s.job = jobs[j];
        if (!s.job.cost) s.job.cost = job_cost(1 + s.job.more);
        s.ticket = t;
        s.gu_left.store(kGuTiles, std::memory_order_relaxed);
        s.dn_left.store(kDnTiles, std::memory_order_relaxed);
        s.open.store(0, std::memory_order_relaxed);
        s.urgent.store(jobs[j].urgent ? 1 : 0, std::memory_order_relaxed);
        s.head = j == 0;
        s.cold = cold;
        if (j == 0) s.pub_ns = now_ns();
        /* Publishes everything above: a worker reads the job only after claiming a tile. */
        s.claim.store((uint64_t)(uint32_t)seq << 32, std::memory_order_release);
        /* ..._LEAN_TILES: claims look at this slot from now; its job's last tile clears the bit
         * before the slot is free, so a later job's bit is never cleared by an earlier one. */
        if (lean_) active_[slot / 64].fetch_or((uint64_t)1 << (slot % 64), std::memory_order_release);
    }
    lk.unlock();
    /* Outside the lock: waking ~40 sleeping workers takes the waker ~30 us, which another rank's
     * submit need not wait out. */
    const int64_t w0 = now_ns();
    const bool woke = wake();
    pstats_.submits.fetch_add(1, std::memory_order_relaxed);
    if (woke) {
        pstats_.submit_woke.fetch_add(1, std::memory_order_relaxed);
        pstats_.wake_ns.fetch_add((uint64_t)(now_ns() - w0), std::memory_order_relaxed);
    }
    return RAD_OK;
}

/* After jobs were published: a worker that saw no work before them either sees the new
 * generation before it sleeps or is woken here (both sides use sequentially consistent
 * operations on work_gen_ and sleepers_). With ..._FUTEX the generation is the futex word: a
 * sleeper whose word moved on does not sleep (the kernel compares it under the futex's lock), and
 * one already asleep is woken here, with no lock taken in user space. */
bool HostExpertPool::wake() {
    work_gen_.fetch_add(1);
    if (sleepers_.load() <= 0) return false;
    if (futex_) {
        futex_op(&work_gen_, FUTEX_WAKE_PRIVATE, INT_MAX);
        return true;
    }
    std::lock_guard<std::mutex> il(idle_mu_);
    idle_cv_.notify_all();
    return true;
}

/* ..._PREWAKE (host_expert_pool.h). The window: one call wakes workers, the others that come while
 * its workers are in their window (the other ranks' for the same layer) wake nobody. Then up to N
 * awake workers in all, counting the ones computing or spinning already. FUTEX_WAKE or notify_one
 * picks the sleepers; each woken sleeper sees prewake_gen_ moved and takes one of the call's tokens
 * (take_prewake), so as many count as pre-woken as the call woke, and spin its window if no work
 * has come. A sleeper the call did not pick but a submit woke right after may take a token in its
 * place; both then have work, and the counts stay right. */
int HostExpertPool::prewake(int trigger) {
    if (prewake_n_ <= 0 || trigger < 0 || trigger >= kWakeTriggers) return 0;
    const int64_t t0 = now_ns();
    int64_t until = prewake_until_ns_.load(std::memory_order_relaxed);
    const int asleep = sleepers_.load();
    const int want = std::min(prewake_n_ - (threads() - asleep), asleep);
    if (t0 < until || want <= 0 ||
        !prewake_until_ns_.compare_exchange_strong(until, t0 + prewake_spin_ns_)) {
        pstats_.pre_skipped[trigger].fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    prewake_trigger_.store(trigger, std::memory_order_relaxed);
    prewake_tokens_.store(want, std::memory_order_relaxed);
    prewake_gen_.fetch_add(1);   /* publishes the trigger and the tokens to the sleepers it wakes */
    long woken = 0;
    if (futex_) {
        woken = futex_op(&work_gen_, FUTEX_WAKE_PRIVATE, (uint32_t)want);
    } else {
        std::lock_guard<std::mutex> il(idle_mu_);
        for (int k = 0; k < want; ++k) idle_cv_.notify_one();
        woken = want;
    }
    if (woken < want) prewake_tokens_.fetch_sub(want - (int)std::max(woken, 0L), std::memory_order_relaxed);
    pstats_.pre_calls[trigger].fetch_add(1, std::memory_order_relaxed);
    pstats_.pre_woken[trigger].fetch_add((uint64_t)std::max(woken, 0L), std::memory_order_relaxed);
    pstats_.pre_call_ns.fetch_add((uint64_t)(now_ns() - t0), std::memory_order_relaxed);
    return (int)std::max(woken, 0L);
}

void HostExpertPool::wait(const Ticket* t) const {
    for (int i = 0; !done(t); ++i) {
        if (i < 4096) _mm_pause();
        else std::this_thread::yield();
    }
}

/* Claims one tile of the queued job with the lowest sequence number that has one: the next
 * gate_up tile, or once the job's gate_up has all finished, the next down tile. The claim word
 * pairs the job's sequence with the next tile, so a worker holding a stale view of a slot that
 * was reused cannot claim a tile of the new job by accident. */
bool HostExpertPool::run_one() {
    for (;;) {
        /* Urgent jobs first, then the oldest: the key is the sequence with the top bit clear for
         * urgent ones. */
        Slot* best = nullptr;
        uint64_t best_claim = 0, best_key = UINT64_MAX;
        auto consider = [&](Slot& s) {
            const uint64_t c = s.claim.load(std::memory_order_acquire);
            const uint32_t k = (uint32_t)c;
            if (k >= (uint32_t)(kGuTiles + kDnTiles)) return;
            if (k >= (uint32_t)kGuTiles &&
                s.open.load(std::memory_order_acquire) != (uint32_t)(c >> 32)) return;
            const uint64_t key = s.seq.load(std::memory_order_relaxed) |
                                 (s.urgent.load(std::memory_order_relaxed) ? 0 : (uint64_t)1 << 63);
            if (key >= best_key) return;
            best = &s;
            best_claim = c;
            best_key = key;
        };
        uint64_t looked = 0;
        if (lean_) {
            /* ..._LEAN_TILES: the slots holding a job. One published after this read is looked
             * for again before the worker sleeps (work_gen_), as a claim word read too early is. */
            for (int w = 0; w < kSlots / 64; ++w)
                for (uint64_t b = active_[w].load(std::memory_order_acquire); b; b &= b - 1, ++looked)
                    consider(slots_[w * 64 + __builtin_ctzll(b)]);
        } else {
            for (int i = 0; i < kSlots; ++i) consider(slots_[i]);
        }
        if (!best) return false;
        if (!best->claim.compare_exchange_strong(best_claim, best_claim + 1,
                                                 std::memory_order_acq_rel)) continue;
        const int k = (int)(uint32_t)best_claim;
        if (t_counts_) {
            bump(t_counts_->claims);
            bump(t_counts_->scanned, looked);
        }
        if (best->head && (k == 0 || k == kRampTiles - 1)) claimed(*best, k);
        tile(*best, (uint32_t)(best_claim >> 32), k);
        return true;
    }
}

/* A submit's first job: its first tile and its kRampTiles-th claimed (a gate_up tile, 72 a job), the
 * second about when that many workers are on it if the pool was idle. */
void HostExpertPool::claimed(const Slot& s, int k) {
    const uint64_t d = (uint64_t)std::max<int64_t>(now_ns() - s.pub_ns, 0);
    PoolStats& p = pstats_;
    auto add = [](std::atomic<uint64_t>& n, std::atomic<uint64_t>& sum, uint64_t v) {
        n.fetch_add(1, std::memory_order_relaxed);
        sum.fetch_add(v, std::memory_order_relaxed);
    };
    if (k == 0) {
        add(p.first_n, p.first_ns, d);
        if (s.cold) add(p.cold_first_n, p.cold_first_ns, d);
    } else {
        add(p.ramp_n, p.ramp_ns, d);
        if (s.cold) add(p.cold_ramp_n, p.cold_ramp_ns, d);
    }
}

void HostExpertPool::lanes_gemv(const Slot& s, bool down, const ds41hx::Mx& m, const uint16_t* const* xs,
                                uint16_t* const* ys, int rows, int n0, int n1) {
    TileLanes& l = t_lanes;
    const uint64_t key = s.seq.load(std::memory_order_relaxed) * 2 + (down ? 1 : 0);
    if (l.key != key) {
        if (l.buf.empty()) l.buf.resize((size_t)ds41hx::kFewRows * ds41hx::few_lanes_floats(kEmbd) / 8);
        ds41hx::few_lanes(xs, rows, m.K, l.buf.data()->f);
        l.key = key;
        if (t_counts_) bump(t_counts_->built);
    } else if (t_counts_) {
        bump(t_counts_->reused);
    }
    ds41hx::gemv_avx2_few_lanes(m, l.buf.data()->f, ys, rows, n0, n1);
}

HostExpertPool::LeanStats HostExpertPool::lean_stats() const {
    LeanStats st;
    if (!counts_) return st;
    for (size_t i = 0; i < workers_.size(); ++i) {
        const WorkerCounts& c = counts_[i];
        st.claims += c.claims.load(std::memory_order_relaxed);
        st.scanned += c.scanned.load(std::memory_order_relaxed);
        st.lanes_built += c.built.load(std::memory_order_relaxed);
        st.lanes_reused += c.reused.load(std::memory_order_relaxed);
    }
    return st;
}

/* A packed slot's view: submit checked it, and the slot is leased and unchanged until the job is done. */
static void packed_or_die(const uint8_t* p, int64_t bytes, int N, int K, ds41hx::Mx* m) {
    if (ds41hx::packed_view(p, (size_t)bytes, N, K, m)) return;
    RAD_ERR("host experts: a packed slot changed under a job (%dx%d); stopping", N, K);
    std::abort();
}

void HostExpertPool::tile(Slot& s, uint32_t gen, int k) {
    ds41hx::Mx m;
    const bool planes = s.job.gu_scales != nullptr;
    /* ..._LEAN_TILES: a job the few-row kernel runs takes it on this worker's lanes of its x. */
    const bool lanes = lean_ && 1 + s.job.more < many_from();
    if (k < kGuTiles) {
        m.N = 2 * kFf;
        m.K = kEmbd;
        if (s.job.packed) packed_or_die(s.job.gu, s.job.gu_bytes, 2 * kFf, kEmbd, &m);
        else if (planes) { m.codes = s.job.gu; m.scales = s.job.gu_scales; }
        else m.blocks = s.job.gu;
        const uint16_t* xs[kMaxRowsCap];
        uint16_t* ys[kMaxRowsCap];
        for (int r = 0; r <= s.job.more; ++r) { xs[r] = r ? s.job.xs[r - 1] : s.job.x; ys[r] = s.gate_up + (size_t)r * 2 * kFf; }
        if (lanes)
            lanes_gemv(s, false, m, xs, ys, 1 + s.job.more, k * kTile, (k + 1) * kTile);
        else
            (1 + s.job.more >= many_from() ? ds41hx::gemv_avx2_many : ds41hx::gemv_avx2_few)(m, xs, ys, 1 + s.job.more,
                                                                                             k * kTile, (k + 1) * kTile);
        if (s.gu_left.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            /* The job's down tiles wait for this, so it runs eight values a step (12 -> ~2 us a
             * row); RADIANCE_EXPERIMENTAL_DS41_SWIGLU_SCALAR=1 keeps the scalar loop (an A/B switch). */
            static const bool scalar = [] {
                const char* v = std::getenv("RADIANCE_EXPERIMENTAL_DS41_SWIGLU_SCALAR");
                return v && !std::strcmp(v, "1");
            }();
            for (int r = 0; r <= s.job.more; ++r)
                (scalar ? ds41hx::swiglu : ds41hx::swiglu_avx2)(silu_.data(), s.gate_up + (size_t)r * 2 * kFf,
                                                                 s.act + (size_t)r * kFf, kFf, limit_);
            s.open.store(gen, std::memory_order_release);
            wake();   /* workers that slept through the gate_up tail take down tiles too */
        }
        return;
    }
    const int d = k - kGuTiles;
    m.N = kEmbd;
    m.K = kFf;
    if (s.job.packed) packed_or_die(s.job.dn, s.job.dn_bytes, kEmbd, kFf, &m);
    else if (planes) { m.codes = s.job.dn; m.scales = s.job.dn_scales; }
    else m.blocks = s.job.dn;
    const uint16_t* xs[kMaxRowsCap];
    uint16_t* ys[kMaxRowsCap];
    for (int r = 0; r <= s.job.more; ++r) { xs[r] = s.act + (size_t)r * kFf; ys[r] = r ? s.job.ys[r - 1] : s.job.y; }
    if (lanes)
        lanes_gemv(s, true, m, xs, ys, 1 + s.job.more, d * kTile, (d + 1) * kTile);
    else
        (1 + s.job.more >= many_from() ? ds41hx::gemv_avx2_many : ds41hx::gemv_avx2_few)(m, xs, ys, 1 + s.job.more,
                                                                                         d * kTile, (d + 1) * kTile);
    if (s.dn_left.fetch_sub(1, std::memory_order_acq_rel) != 1) return;
    /* The job's last tile: nothing reads the slot after this. */
    Ticket* t = s.ticket;
    const int64_t left = backlog_.fetch_sub(s.job.cost, std::memory_order_relaxed) - s.job.cost;
    const bool late = quiet_late_.load(std::memory_order_relaxed);
    if (!late) drained(left);
    if (lean_) {
        const int i = (int)(&s - slots_.get());
        active_[i / 64].fetch_and(~((uint64_t)1 << (i % 64)), std::memory_order_relaxed);
    }
    s.state.store(0, std::memory_order_release);
    t->left.fetch_sub(1, std::memory_order_acq_rel);
    slot_cv_.notify_one();
    /* quiet_after_tickets: the ticket's waiter goes on first. */
    if (late) drained(left);
}

void HostExpertPool::drained(int64_t left) {
    if (left > 0) return;
    std::atomic_thread_fence(std::memory_order_seq_cst);   /* against wait_idle's count, then check */
    if (quiet_waiters_.load(std::memory_order_relaxed) == 0) return;
    std::lock_guard<std::mutex> lk(quiet_mu_);
    quiet_cv_.notify_all();
}

void HostExpertPool::wait_idle(int64_t max_ns) {
    if (max_ns <= 0 || backlog_.load(std::memory_order_relaxed) <= 0) return;
    std::unique_lock<std::mutex> lk(quiet_mu_);
    quiet_waiters_.fetch_add(1);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    quiet_cv_.wait_for(lk, std::chrono::nanoseconds(max_ns), [&] {
        return backlog_.load(std::memory_order_relaxed) <= 0 || stop_.load(std::memory_order_relaxed);
    });
    quiet_waiters_.fetch_sub(1);
}

void HostExpertPool::worker(int i, int cpu) {
    t_counts_ = counts_ ? &counts_[(size_t)i] : nullptr;
    if (cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        if (pthread_setaffinity_np(pthread_self(), sizeof set, &set) != 0 && i == 0)
            RAD_WARN("host experts: could not pin a worker to CPU %d; the workers float", cpu);
    }
    char name[16];
    std::snprintf(name, sizeof name, "rad-hx-%d", i);
    pthread_setname_np(pthread_self(), name);
    /* Linux nice is per thread: below the issuing and relay threads, which serve the misses. */
    if (g_nice > 0 && setpriority(PRIO_PROCESS, (id_t)syscall(SYS_gettid), g_nice) != 0 && i == 0)
        RAD_WARN("host experts: could not lower the workers' priority to nice %d", g_nice);
    WorkerStats& ws = wstats_[(size_t)i];
    int pre = -1;   /* this wake-up was a pre-wake, from that trigger (sleep()) */
    while (!stop_.load(std::memory_order_relaxed)) {
        const uint32_t seen = work_gen_.load();
        bool got = run_one();
        if (!got) {
            /* Idle: spin a while, since a layer's jobs come in bursts a few milliseconds apart, then
             * sleep until the next submit. A pre-woken worker spins its window (..._PREWAKE_SPIN_US). */
            const int64_t limit = pre >= 0 ? prewake_spin_ns_ : g_spin_ns;
            int64_t spun = 0;
            const auto t0 = std::chrono::steady_clock::now();
            while (!got && !stop_.load(std::memory_order_relaxed)) {
                for (int r = 0; r < 64; ++r) _mm_pause();
                spun = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - t0).count();
                got = run_one();
                if (spun > limit) break;
            }
            ws.spin_ns.fetch_add((uint64_t)spun, std::memory_order_relaxed);
            if (pre >= 0) ws.pre_spin_ns[pre].fetch_add((uint64_t)spun, std::memory_order_relaxed);
        }
        if (pre >= 0) {
            (got ? ws.pre_used : ws.pre_slept)[pre].fetch_add(1, std::memory_order_relaxed);
            pre = -1;
        }
        if (got) continue;
        pre = sleep(seen, ws);
    }
}

/* Through the condition variable as before, or with ..._FUTEX on the work generation itself. A
 * pre-wake ends the sleep too (prewake_gen_ moved on since the worker came here). A worker that
 * slept through a pre-wake counts as pre-woken if it takes one of its tokens, whether a submit's
 * work has come by the time it runs or not (then it spins its window). A futex sleeper woken for
 * nothing (a signal, a pre-wake from before it slept) sleeps on. */
int HostExpertPool::sleep(uint32_t seen, WorkerStats& ws) {
    ws.sleeps.fetch_add(1, std::memory_order_relaxed);
    const uint32_t pw = prewake_gen_.load();
    if (futex_) {
        sleepers_.fetch_add(1);
        while (!stop_.load() && work_gen_.load() == seen) {
            futex_op(&work_gen_, FUTEX_WAIT_PRIVATE, seen);
            if (prewake_gen_.load() != pw) break;
        }
        sleepers_.fetch_sub(1);
    } else {
        std::unique_lock<std::mutex> lk(idle_mu_);
        sleepers_.fetch_add(1);
        idle_cv_.wait(lk, [&] {
            return stop_.load() || work_gen_.load() != seen || (prewake_n_ > 0 && prewake_gen_.load() != pw);
        });
        sleepers_.fetch_sub(1);
    }
    if (prewake_n_ <= 0 || stop_.load() || prewake_gen_.load() == pw || !take_prewake()) return -1;
    return prewake_trigger_.load(std::memory_order_relaxed);
}

/* One token of the last pre-wake, if any is left and the call was recent (within two windows: a
 * worker it woke runs well inside its own window; one that slept through an older call and wakes for
 * a later submit takes none). */
bool HostExpertPool::take_prewake() {
    if (now_ns() > prewake_until_ns_.load(std::memory_order_relaxed) + prewake_spin_ns_) return false;
    int t = prewake_tokens_.load(std::memory_order_relaxed);
    while (t > 0)
        if (prewake_tokens_.compare_exchange_weak(t, t - 1, std::memory_order_relaxed)) return true;
    return false;
}

HostExpertPool::Stats HostExpertPool::stats() const {
    Stats s;
    const PoolStats& p = pstats_;
    auto ld = [](const std::atomic<uint64_t>& a) { return a.load(std::memory_order_relaxed); };
    s.submits = ld(p.submits);
    s.submit_woke = ld(p.submit_woke);
    s.wake_ns = ld(p.wake_ns);
    s.first_n = ld(p.first_n);
    s.first_ns = ld(p.first_ns);
    s.ramp_n = ld(p.ramp_n);
    s.ramp_ns = ld(p.ramp_ns);
    s.cold_first_n = ld(p.cold_first_n);
    s.cold_first_ns = ld(p.cold_first_ns);
    s.cold_ramp_n = ld(p.cold_ramp_n);
    s.cold_ramp_ns = ld(p.cold_ramp_ns);
    s.pre_call_ns = ld(p.pre_call_ns);
    for (int t = 0; t < kWakeTriggers; ++t) {
        s.pre_calls[t] = ld(p.pre_calls[t]);
        s.pre_skipped[t] = ld(p.pre_skipped[t]);
        s.pre_woken[t] = ld(p.pre_woken[t]);
    }
    for (size_t i = 0; wstats_ && i < workers_.size(); ++i) {
        const WorkerStats& w = wstats_[i];
        s.spin_ns += ld(w.spin_ns);
        s.sleeps += ld(w.sleeps);
        for (int t = 0; t < kWakeTriggers; ++t) {
            s.pre_used[t] += ld(w.pre_used[t]);
            s.pre_slept[t] += ld(w.pre_slept[t]);
            s.pre_spin_ns[t] += ld(w.pre_spin_ns[t]);
        }
    }
    return s;
}

/* One line for the shutdown report: the wake-up, the first tiles, the pre-wakes, the idle spin. */
std::string HostExpertPool::report() const {
    const Stats s = stats();
    auto us = [](uint64_t ns, uint64_t n) { return n ? (double)ns * 1e-3 / (double)n : 0.0; };
    std::string pre;
    static const char* const kTrig[kWakeTriggers] = { "router", "miss" };
    for (int t = 0; t < kWakeTriggers; ++t) {
        if (!prewake_n_) break;
        const uint64_t ended = s.pre_used[t] + s.pre_slept[t];
        pre += fmt("; %s pre-wakes: %llu calls woke %llu workers (%llu calls woke nobody), %llu of them "
                   "took a tile in their window and %llu slept again, %.1f us spinning each",
                   kTrig[t], (unsigned long long)s.pre_calls[t], (unsigned long long)s.pre_woken[t],
                   (unsigned long long)s.pre_skipped[t], (unsigned long long)s.pre_used[t],
                   (unsigned long long)s.pre_slept[t], us(s.pre_spin_ns[t], ended));
    }
    if (prewake_n_) {
        const uint64_t calls = s.pre_calls[kWakeRouter] + s.pre_calls[kWakeMiss];
        pre += fmt(" (%.1f us a call that woke workers)", us(s.pre_call_ns, calls));
    }
    return fmt("host expert pool: %d workers, %s wake%s, pre-wake %d (..._PREWAKE); %llu submits, %llu "
               "of them woke sleeping workers (%.1f us in the wake each); a submit's first job: its first "
               "tile %.1f us after it, its %dth %.1f us (%llu / %llu), from an all-asleep pool %.1f and "
               "%.1f us (%llu / %llu)%s; idle spin %.1f ms in all, %llu sleeps",
               threads(), futex_ ? "futex" : "condition-variable", futex_ ? " (..._FUTEX)" : "",
               prewake_n_, (unsigned long long)s.submits, (unsigned long long)s.submit_woke,
               us(s.wake_ns, s.submit_woke), us(s.first_ns, s.first_n), kRampTiles,
               us(s.ramp_ns, s.ramp_n), (unsigned long long)s.first_n, (unsigned long long)s.ramp_n,
               us(s.cold_first_ns, s.cold_first_n), us(s.cold_ramp_ns, s.cold_ramp_n),
               (unsigned long long)s.cold_first_n, (unsigned long long)s.cold_ramp_n, pre.c_str(),
               (double)s.spin_ns * 1e-6, (unsigned long long)s.sleeps);
}

}  /* namespace rad */
