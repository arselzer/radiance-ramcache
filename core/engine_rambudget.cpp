/* engine_rambudget.cpp -- RADIANCE_EXPERIMENTAL_DS41_RAM_BUDGET: the expert RAM cache sized from the
 * host memory the box has, and the guard that stops its growth when MemAvailable runs low
 * (core/ram_budget.h has the rule and why).
 *
 * Off (the default: no `auto`, no switch) nothing here runs and nothing else changes: the cache
 * keeps its configured budget and no growth limit (SIZE_MAX) is ever set.
 *
 * On: start() takes the cap (the number beside the switch, or MemTotal for `auto` alone) as each
 * rank's cache budget, with a growth limit of 0, so nothing grows during startup and no startup
 * check reserves the cache (Engine::host_headroom_bytes). serve() then measures MemAvailable with
 * every startup allocation already out of it, subtracts what grows later and the reserve, and sets
 * each rank's growth limit to its share. The guard thread samples MemAvailable every
 * RADIANCE_EXPERIMENTAL_DS41_RAM_GUARD_MS (250) and, below the floor (half the reserve by default),
 * lowers every limit to what the cache holds: it stops growing and replaces within its slabs, and
 * a unit it refuses is read from the container, so replies keep their bytes. */
#include "engine_priv.h"
#include "device/device.h"
#include "engine_hostmem.h"
#include "kvstream/bringup.h"
#include "ram_budget.h"
#include "runtime/arena.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace rad {

namespace {

constexpr int64_t kMiB = (int64_t)1 << 20;

/* An environment value as a whole number in [lo, hi]; `def` when unset or empty. False (and an
 * error logged) when set to anything else. */
bool env_num(const char* name, int64_t lo, int64_t hi, int64_t def, int64_t* out) {
    const char* v = std::getenv(name);
    *out = def;
    if (!v || !*v) return true;
    char* end = nullptr;
    const long long n = std::strtoll(v, &end, 10);
    if (!end || *end || n < lo || n > hi) {
        RAD_ERR("%s=%s is not a whole number in [%lld, %lld]", name, v, (long long)lo, (long long)hi);
        return false;
    }
    *out = (int64_t)n;
    return true;
}

std::string hb(int64_t b) { return b < 0 ? std::string("unreadable") : humanb(b); }

}  /* namespace */

int Engine::ram_budget_begin() {
    const char* sw = std::getenv("RADIANCE_EXPERIMENTAL_DS41_RAM_BUDGET");
    if (sw && *sw && std::strcmp(sw, "0") && std::strcmp(sw, "1")) {
        RAD_ERR("RADIANCE_EXPERIMENTAL_DS41_RAM_BUDGET=%s: 1 sizes the expert RAM cache from the host "
                "memory startup leaves (the --expert-ram-cache-total-mib figure then caps it); 0 or "
                "unset keeps the figure as it is", sw);
        return RAD_E_INVAL;
    }
    const bool env_on = sw && !std::strcmp(sw, "1");
    ram_budget_on_ = cfg_.expert_ram_cache_auto || (env_on && cfg_.expert_ram_cache_total_mib > 0);
    if (env_on && !ram_budget_on_)
        RAD_WARN("ram budget: RADIANCE_EXPERIMENTAL_DS41_RAM_BUDGET=1, but no expert RAM cache is "
                 "configured (--expert-ram-cache-total-mib N or auto): nothing to size");
    if (!ram_budget_on_) return RAD_OK;
    int64_t reserve_mib = 0, floor_mib = 0;
    if (!env_num("RADIANCE_EXPERIMENTAL_DS41_RAM_RESERVE_MIB", 0, (int64_t)1 << 30, 10240, &reserve_mib))
        return RAD_E_INVAL;
    if (!env_num("RADIANCE_EXPERIMENTAL_DS41_RAM_GUARD_FLOOR_MIB", 0, reserve_mib, reserve_mib / 2,
                 &floor_mib) ||
        !env_num("RADIANCE_EXPERIMENTAL_DS41_RAM_GUARD_MS", 10, 60000, 250, &ram_guard_ms_) ||
        /* --checkpoint-slots auto's session length (Engine::size_checkpoint_store). */
        !env_num("RADIANCE_EXPERIMENTAL_DS41_RAM_BUDGET_SESSION_TOKENS", 1, (int64_t)1 << 32, 0,
                 &ram_session_tokens_))
        return RAD_E_INVAL;
    ram_reserve_bytes_ = reserve_mib * kMiB;
    ram_floor_bytes_ = floor_mib * kMiB;
    ram_cap_given_ = cfg_.expert_ram_cache_total_mib > 0;
    if (!ram_cap_given_) {
        const int64_t total = meminfo_now("MemTotal");
        if (total <= 0) {
            RAD_ERR("--expert-ram-cache-total-mib auto: /proc/meminfo has no MemTotal to cap the "
                    "cache by; give a number");
            return RAD_E_STATE;
        }
        cfg_.expert_ram_cache_total_mib = total / kMiB;   /* the mover's budget: the cap */
    }
    ram_avail_at_start_ = meminfo_now("MemAvailable");
    ram_rss_at_start_ = proc_status_now("VmRSS");
    RAD_INFO("ram budget: on (%s): the expert RAM cache is sized at serve from the host memory "
             "startup leaves, at most %s; MemAvailable %s now; reserve %s; the guard holds the "
             "cache's growth below %s of MemAvailable, sampled every %lld ms",
             cfg_.expert_ram_cache_auto ? "--expert-ram-cache-total-mib auto"
                                        : "RADIANCE_EXPERIMENTAL_DS41_RAM_BUDGET=1",
             ram_cap_given_ ? humanb(cfg_.expert_ram_cache_total_mib * kMiB).c_str()
                            : fmt("MemTotal (%s)", humanb(cfg_.expert_ram_cache_total_mib * kMiB).c_str()).c_str(),
             hb(ram_avail_at_start_).c_str(), humanb(ram_reserve_bytes_).c_str(),
             humanb(ram_floor_bytes_).c_str(), (long long)ram_guard_ms_);
    return RAD_OK;
}

int Engine::ram_budget_serve() {
    if (!ram_budget_on_) return RAD_OK;
    RamBudgetIn in;
    std::string room_why;
    in.mem_available = host_room_now(&room_why);
    in.swap_free = meminfo_now("SwapFree");
    in.pin_room = host_pin_room_now();
    in.reserve = ram_reserve_bytes_;
    in.cap = ram_cap_given_ ? cfg_.expert_ram_cache_total_mib * kMiB : 0;
    const int64_t rank_cap = cfg_.tp > 0 ? cfg_.expert_ram_cache_total_mib * kMiB / cfg_.tp /
                                               RAD_ALIGN_UNIT * RAD_ALIGN_UNIT : 0;
    int64_t arenas = 0, scored = 0;
    int active = 0;
    KVStreamHostUse stream;
    for (auto& r : ranks_) {
        const bool on = r->mover.ram_cache_on();
        active += on ? 1 : 0;
        in.backed.push_back(on ? r->mover.cache_growth_charged() : 0);
        in.rank_cap.push_back(on ? rank_cap : 0);
        arenas += arena_host_bytes(r->program);
        kvstream_host_add(&stream, r->kv);
        scored += cfg_.max_tok * r->sampler.config().n_vocab_local * (int64_t)sizeof(float);
    }
    if (!active) {
        RAD_WARN("ram budget: no rank's plan reads experts from the file tier, so there is no RAM "
                 "cache to size");
        ram_budget_on_ = false;
        return RAD_OK;
    }
    /* Already out of MemAvailable (informational: the decision measures them by construction). */
    int64_t pinned = 0, mapped = 0, n_pinned = 0, n_mapped = 0;
    rad_dev_alloc_totals(RAD_MEM_HOST_PINNED, &pinned, &n_pinned);
    rad_dev_alloc_totals(RAD_MEM_HOST_MAPPED, &mapped, &n_mapped);
    const int64_t tp = (int64_t)ranks_.size();
    in.held.push_back({ "host pools, pinned once a rank", tp * cfg_.host_pool_total_mib() * kMiB,
                        "core/mem/pools.cpp, --host-pool-mib + RAM tables" });
    in.held.push_back({ fmt("prefix-cache snapshot store, %lld slots of %s", (long long)ck_store_.capacity(),
                            humanb(ck_store_.slot_bytes()).c_str()), ck_store_.cap_bytes(),
                        "engine_bringup.cpp configure_kv ck_store_.open, --checkpoint-slots" });
    if (cfg_.prefix_cache && cfg_.prefix_cache_mib > 0)
        in.held.push_back({ "prefix cache host KV tier", cfg_.prefix_cache_mib * kMiB,
                            "core/mem/kvtier.h, --prefix-cache-mib" });
    in.held.push_back({ "host activation arenas", arenas, "core/runtime/ctx.cpp host_arena_.init" });
    if (stream.sum.host_bytes > 0)
        in.held.push_back({ "KV stream mirrors", stream.sum.host_bytes, "core/kvstream/mirror.cpp alloc" });
    int64_t backed = 0;
    for (int64_t b : in.backed) backed += b;
    in.held.push_back({ "expert RAM cache slabs so far", backed, "core/place/native_mapped_file_cache.h" });
    in.held.push_back({ fmt("all rad_dev_alloc pinned, %lld buffers (sampler stages, batch ring, stager "
                            "slots, the pools and store above...)", (long long)n_pinned),
                        pinned, "core/device/dispatch.cpp ledger, never counts down" });
    in.held.push_back({ fmt("all rad_dev_alloc mapped, %lld buffers", (long long)n_mapped), mapped,
                        "core/device/dispatch.cpp ledger, never counts down" });
    const int64_t avail_now = meminfo_now("MemAvailable");
    if (ram_avail_at_start_ >= 0 && avail_now >= 0)
        in.held.push_back({ "measured: MemAvailable fall since start()", ram_avail_at_start_ - avail_now,
                            fmt("%s at start, %s now; other processes included", hb(ram_avail_at_start_).c_str(),
                                hb(avail_now).c_str()) });
    const int64_t rss = proc_status_now("VmRSS");
    if (rss >= 0)
        in.held.push_back({ "measured: this process's RSS", rss,
                            fmt("/proc/self/status VmRSS, %s at start()", hb(ram_rss_at_start_).c_str()) });
    /* Allocated after this point: subtracted. */
    int64_t per_seq_mib = 0;
    if (!env_num("RADIANCE_EXPERIMENTAL_DS41_RAM_PER_SEQ_MIB", 0, 1 << 20, 100, &per_seq_mib))
        return RAD_E_INVAL;
    in.later.push_back({ fmt("per-sequence growth, %lld MiB x --max-num-seqs %lld", (long long)per_seq_mib,
                             (long long)cfg_.max_seqs),
                         per_seq_mib * kMiB * cfg_.max_seqs,
                         "measured 2026-11-12 (MemAvailable per stream), not itemized; "
                         "RADIANCE_EXPERIMENTAL_DS41_RAM_PER_SEQ_MIB" });
    in.later.push_back({ "scored rows' pinned logits, grown on the first scored step", scored,
                         "core/engine.cpp stage_scored_rows, --max-num-batched-tokens x vocab shard" });

    const RamBudget b = ram_budget_resolve(in);
    for (const std::string& line : b.lines) RAD_INFO("ram budget: %s", line.c_str());
    ram_rank_limit_ = b.rank_limit;
    ram_limits_apply(false);
    if (b.ok)
        RAD_INFO("ram budget: the expert RAM cache may hold %s (%s a rank on %d rank(s)) of %s%s; room "
                 "bound by %s", humanb(b.total).c_str(), humanb(b.total / std::max(active, 1)).c_str(),
                 active, humanb(cfg_.expert_ram_cache_total_mib * kMiB).c_str(),
                 ram_cap_given_ ? " configured" : " (MemTotal)", room_why.c_str());
    else
        RAD_WARN("ram budget: MemAvailable is unreadable, so the expert RAM cache stays at the %s "
                 "it holds", humanb(b.total).c_str());
    /* ..._PREGROW and ..._CACHE_WARM found a limit of 0 at init: they run now, inside the budget. */
    const char* pg = std::getenv("RADIANCE_EXPERIMENTAL_NATIVE_CACHE_PREGROW");
    const char* w = std::getenv("RADIANCE_EXPERIMENTAL_DS41_CACHE_WARM");
    const bool pregrow = pg && !std::strcmp(pg, "1"), warm = w && !std::strcmp(w, "1");
    for (auto& r : ranks_) {
        if (!r->mover.ram_cache_on() || (!pregrow && !warm)) continue;
        LogRank lr(r->index);
        RAD_TRY(rad_dev_set(r->device));
        if (pregrow) {
            const int64_t pairs = r->mover.mapped_cache_pregrow();
            RAD_INFO("ram budget: rank %d pinned its cache's slabs within the budget (%lld pairs, %s held)",
                     r->index, (long long)pairs, humanb(r->mover.cache_growth_charged()).c_str());
        }
        if (warm) RAD_TRY(r->fstager.warm_ram_cache());
    }
    if (!ranks_.empty()) RAD_TRY(rad_dev_set(ranks_[0]->device));
    {
        std::lock_guard<std::mutex> lk(ram_guard_mu_);
        ram_guard_quit_ = false;
    }
    ram_guard_thread_ = std::thread([this] { ram_guard_loop(); });
    return RAD_OK;
}

void Engine::ram_limits_apply(bool held) {
    for (size_t i = 0; i < ranks_.size() && i < ram_rank_limit_.size(); ++i) {
        Mover& m = ranks_[i]->mover;
        if (!m.ram_cache_on()) continue;
        const int64_t lim = held ? std::min(ram_rank_limit_[i], m.cache_growth_charged()) : ram_rank_limit_[i];
        m.set_cache_growth_limit(lim);
    }
}

void Engine::ram_guard_loop() {
    RamGuardState g(ram_floor_bytes_, ram_reserve_bytes_);
    int64_t peak_rss = 0;
    std::unique_lock<std::mutex> lk(ram_guard_mu_);
    while (!ram_guard_quit_) {
        lk.unlock();
        const int64_t avail = meminfo_now("MemAvailable");
        peak_rss = std::max(peak_rss, proc_status_now("VmRSS"));
        int64_t held = 0;
        switch (g.observe(avail)) {
        case RamGuardState::Event::Hold:
            ram_limits_apply(true);
            for (auto& r : ranks_) held += r->mover.cache_growth_charged();
            RAD_WARN("ram guard: MemAvailable %s is below the floor of %s: the expert RAM cache stops "
                     "growing at the %s it holds, and replaces within it (replies unchanged)",
                     humanb(avail).c_str(), humanb(g.floor()).c_str(), humanb(held).c_str());
            break;
        case RamGuardState::Event::Release:
            ram_limits_apply(false);
            RAD_INFO("ram guard: MemAvailable %s is back at the reserve of %s: the expert RAM cache "
                     "may grow to its budget again", humanb(avail).c_str(), humanb(g.release()).c_str());
            break;
        case RamGuardState::Event::None:
            break;
        }
        lk.lock();
        ram_guard_cv_.wait_for(lk, std::chrono::milliseconds(ram_guard_ms_), [this] { return ram_guard_quit_; });
    }
    lk.unlock();
    int64_t holds_now = 0, limit = 0;
    uint64_t refused = 0;
    for (size_t i = 0; i < ranks_.size(); ++i) {
        holds_now += ranks_[i]->mover.cache_growth_charged();
        refused += ranks_[i]->mover.cache_growth_refusals();
        if (i < ram_rank_limit_.size()) limit += ram_rank_limit_[i];
    }
    RAD_INFO("ram guard: stopped after %llu samples: lowest MemAvailable %s (floor %s), held %llu "
             "time(s)%s; the cache holds %s of its %s budget, %llu growth(s) refused; peak RSS %s "
             "(%s at start)", (unsigned long long)g.samples(), hb(g.lowest()).c_str(),
             humanb(g.floor()).c_str(), (unsigned long long)g.holds(), g.held() ? ", held at the end" : "",
             humanb(holds_now).c_str(), humanb(limit).c_str(), (unsigned long long)refused,
             hb(peak_rss).c_str(), hb(ram_rss_at_start_).c_str());
}

void Engine::ram_guard_stop() {
    if (!ram_guard_thread_.joinable()) return;
    {
        std::lock_guard<std::mutex> lk(ram_guard_mu_);
        ram_guard_quit_ = true;
    }
    ram_guard_cv_.notify_all();
    ram_guard_thread_.join();
}

}  /* namespace rad */
