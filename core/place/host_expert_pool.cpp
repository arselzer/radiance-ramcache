/* host_expert_pool.cpp -- see host_expert_pool.h. */
#include "host_expert_pool.h"

#include "ds41_host_experts.h"
#include "../rad_internal.h"

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace rad {

namespace {

constexpr const char* kEnvThreads = "RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS";
constexpr const char* kEnvSilu    = "RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_SILU";
constexpr const char* kEnvCpus    = "RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_CPUS";
constexpr const char* kEnvGbps    = "RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_GBPS";
constexpr const char* kEnvLink    = "RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_LINK_GBPS";
constexpr const char* kEnvSpin    = "RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_SPIN_US";
constexpr const char* kEnvNice    = "RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_NICE";

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

}  /* namespace */

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
        auto* p = new HostExpertPool();
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
        if (p->start(threads, cpus) < 0) {
            delete p;
            return nullptr;
        }
        RAD_INFO("host experts: %d CPU threads%s compute RAM-resident DeepSeek-V4.1 experts "
                 "(silu table %s; scheduling at %.0f GB/s against %.0f GB/s a copy link)",
                 threads, cpus.empty() ? "" : (" pinned to " + std::string(cl)).c_str(), path,
                 p->cpu_bpns_, p->link_bpns_);
        return p;
    }();
    return pool;
}

int HostExpertPool::start(int threads, const std::vector<int>& cpus) {
    slots_.reset(new (std::nothrow) Slot[kSlots]);
    if (!slots_) return RAD_E_NOMEM;
    for (int i = 0; i < kSlots; ++i) slots_[i].claim.store(kDone, std::memory_order_relaxed);
    try {
        for (int i = 0; i < threads; ++i)
            workers_.emplace_back(&HostExpertPool::worker, this, i,
                                  cpus.empty() ? -1 : cpus[(size_t)i % cpus.size()]);
    } catch (...) {
        {
            std::lock_guard<std::mutex> lk(idle_mu_);   /* as the destructor: no lost wakeup */
            stop_ = true;
        }
        idle_cv_.notify_all();
        for (std::thread& w : workers_) w.join();
        workers_.clear();
        RAD_ERR("host experts: could not start %d threads", threads);
        return RAD_E_NOMEM;
    }
    return RAD_OK;
}

HostExpertPool::~HostExpertPool() {
    {
        std::lock_guard<std::mutex> lk(idle_mu_);
        stop_ = true;
    }
    idle_cv_.notify_all();
    for (std::thread& w : workers_) w.join();
}

int HostExpertPool::submit(const Job* jobs, int n, Ticket* t) {
    if (!t || n < 0 || (n > 0 && !jobs)) return RAD_E_INVAL;
    if (n == 0) return RAD_OK;
    for (int j = 0; j < n; ++j) {
        const Job& b = jobs[j];
        if (!b.gu || !b.dn || !b.x || !b.y || (!b.gu_scales) != (!b.dn_scales) || b.more < 0 ||
            b.more >= kMaxRows)
            return RAD_E_INVAL;
        for (int r = 0; r < b.more; ++r)
            if (!b.xs[r] || !b.ys[r]) return RAD_E_INVAL;
    }
    /* Counted before any job is visible, so a waiter never sees zero while one is still to come. */
    t->left.fetch_add(n, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(submit_mu_);
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
        /* Publishes everything above: a worker reads the job only after claiming a tile. */
        s.claim.store((uint64_t)(uint32_t)seq << 32, std::memory_order_release);
    }
    wake();
    return RAD_OK;
}

/* After jobs were published: a worker that saw no work before them either sees the new
 * generation before it sleeps or is woken here (both sides use sequentially consistent
 * operations on work_gen_ and sleepers_). */
void HostExpertPool::wake() {
    work_gen_.fetch_add(1);
    if (sleepers_.load() > 0) {
        std::lock_guard<std::mutex> il(idle_mu_);
        idle_cv_.notify_all();
    }
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
        for (int i = 0; i < kSlots; ++i) {
            Slot& s = slots_[i];
            const uint64_t c = s.claim.load(std::memory_order_acquire);
            const uint32_t k = (uint32_t)c;
            if (k >= (uint32_t)(kGuTiles + kDnTiles)) continue;
            if (k >= (uint32_t)kGuTiles &&
                s.open.load(std::memory_order_acquire) != (uint32_t)(c >> 32)) continue;
            const uint64_t key = s.seq.load(std::memory_order_relaxed) |
                                 (s.urgent.load(std::memory_order_relaxed) ? 0 : (uint64_t)1 << 63);
            if (key >= best_key) continue;
            best = &s;
            best_claim = c;
            best_key = key;
        }
        if (!best) return false;
        if (!best->claim.compare_exchange_strong(best_claim, best_claim + 1,
                                                 std::memory_order_acq_rel)) continue;
        tile(*best, (uint32_t)(best_claim >> 32), (int)(uint32_t)best_claim);
        return true;
    }
}

void HostExpertPool::tile(Slot& s, uint32_t gen, int k) {
    ds41hx::Mx m;
    const bool planes = s.job.gu_scales != nullptr;
    if (k < kGuTiles) {
        m.N = 2 * kFf;
        m.K = kEmbd;
        if (planes) { m.codes = s.job.gu; m.scales = s.job.gu_scales; }
        else m.blocks = s.job.gu;
        for (int r = 0; r <= s.job.more; ++r)
            ds41hx::gemv_avx2(m, r ? s.job.xs[r - 1] : s.job.x, s.gate_up[r], k * kTile, (k + 1) * kTile);
        if (s.gu_left.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            for (int r = 0; r <= s.job.more; ++r)
                ds41hx::swiglu(silu_.data(), s.gate_up[r], s.act[r], kFf, limit_);
            s.open.store(gen, std::memory_order_release);
            wake();   /* workers that slept through the gate_up tail take down tiles too */
        }
        return;
    }
    const int d = k - kGuTiles;
    m.N = kEmbd;
    m.K = kFf;
    if (planes) { m.codes = s.job.dn; m.scales = s.job.dn_scales; }
    else m.blocks = s.job.dn;
    for (int r = 0; r <= s.job.more; ++r)
        ds41hx::gemv_avx2(m, s.act[r], r ? s.job.ys[r - 1] : s.job.y, d * kTile, (d + 1) * kTile);
    if (s.dn_left.fetch_sub(1, std::memory_order_acq_rel) != 1) return;
    /* The job's last tile: nothing reads the slot after this. */
    Ticket* t = s.ticket;
    backlog_.fetch_sub(s.job.cost, std::memory_order_relaxed);
    s.state.store(0, std::memory_order_release);
    t->left.fetch_sub(1, std::memory_order_acq_rel);
    slot_cv_.notify_one();
}

void HostExpertPool::worker(int i, int cpu) {
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
    while (!stop_.load(std::memory_order_relaxed)) {
        const uint64_t seen = work_gen_.load();
        if (run_one()) continue;
        /* Idle: spin a while, since a layer's jobs come in bursts a few milliseconds apart, then
         * sleep until the next submit. */
        bool got = false;
        const auto t0 = std::chrono::steady_clock::now();
        while (!got && !stop_.load(std::memory_order_relaxed)) {
            for (int r = 0; r < 64; ++r) _mm_pause();
            got = run_one();
            if (std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - t0).count() > g_spin_ns) break;
        }
        if (got) continue;
        std::unique_lock<std::mutex> lk(idle_mu_);
        sleepers_.fetch_add(1);
        idle_cv_.wait(lk, [&] { return stop_.load() || work_gen_.load() != seen; });
        sleepers_.fetch_sub(1);
    }
}

}  /* namespace rad */
