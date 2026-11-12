/* ram_budget.h -- the expert RAM cache sized from the host memory the box has
 * (RADIANCE_EXPERIMENTAL_DS41_RAM_BUDGET=1, or --expert-ram-cache-total-mib auto), and the runtime
 * guard that stops the cache growing when MemAvailable falls below a floor.
 *
 * Why: on 2026-11-12 an arm with 256 sequences and a 192 GiB cache ran the host out of RAM; the
 * OOM killer fired and an amdgpu bug on that path panicked the kernel. A fixed cache figure cannot
 * know what the rest of the configuration takes; this mode decides the cache last, at serve(),
 * after every startup allocation is already out of MemAvailable, so only what grows later has to
 * be counted (RamBudgetIn::later) besides the reserve.
 *
 * Exactness: both only bound how far the cache's slabs grow. A unit the cache refuses is read
 * from the container as before (the mover's window path), so a reply's bytes never change.
 *
 * Pure arithmetic: no /proc reads, no allocation of host memory beyond the result, so the host
 * tests drive it with synthetic figures (tests/ram_budget_test.cpp). */
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace rad {

struct RamBudgetItem {
    std::string name;     /* what it is */
    int64_t     bytes = 0;
    std::string source;   /* where the code allocates it (file:line or the switch that sizes it) */
};

struct RamBudgetIn {
    /* The host memory free for the cache now: MemAvailable, or less when a memory cgroup leaves
     * less (lockable room). Negative: unreadable, and the cache is held where it is. */
    int64_t mem_available = -1;
    int64_t swap_free = -1;          /* logged only: a pinned slab never swaps */
    /* What the amdgpu driver still lets this process pin (its cap less what is pinned), INT64_MAX
     * when lifted or unknown. The cache's slabs are pinned (RAD_MEM_HOST_MAPPED). */
    int64_t pin_room = INT64_MAX;
    int64_t reserve = 0;             /* left free for the system and the guard's floor */
    int64_t cap = 0;                 /* the configured total (all ranks); 0 is none */
    /* Per rank: what its cache already holds (already gone from mem_available), and the most it
     * may hold (its share of `cap`, as the mover was configured; 0 is a rank without a cache, which
     * takes no share; negative, or missing, is no cap). */
    std::vector<int64_t> backed;
    std::vector<int64_t> rank_cap;
    std::vector<RamBudgetItem> held;   /* allocated before the decision: in mem_available already */
    std::vector<RamBudgetItem> later;  /* allocated after it: subtracted */
};

struct RamBudget {
    bool    ok = false;              /* false: mem_available unreadable; every rank held at backed */
    int64_t later = 0;               /* sum of RamBudgetIn::later, saturated */
    int64_t room = 0;                /* min(mem_available, pin_room) - later - reserve, >= 0 */
    int64_t growth = 0;              /* what the caches may still grow, all ranks: <= room */
    int64_t total = 0;               /* sum of rank_limit */
    std::vector<int64_t> rank_limit; /* each rank's payload limit: backed + its share of room */
    std::vector<std::string> lines;  /* the log, item by item */
};

/* The rule: room = min(mem_available, pin_room) - sum(later) - reserve, split evenly across the
 * ranks in whole 2 MiB, each rank's limit min(rank_cap, backed + share). Never more growth than
 * room: growth + sum(later) + reserve <= mem_available. */
RamBudget ram_budget_resolve(const RamBudgetIn& in);

/* The runtime guard's state machine, one MemAvailable sample at a time. Below `floor` the cache
 * is held (stops growing; it keeps replacing within what it holds); at or above `release` it may
 * grow again up to the startup budget. release >= floor gives the hysteresis. A negative sample
 * (unreadable) changes nothing. */
class RamGuardState {
public:
    enum class Event { None, Hold, Release };
    RamGuardState(int64_t floor, int64_t release)
        : floor_(floor), release_(release < floor ? floor : release) {}
    Event observe(int64_t avail);
    bool     held() const { return held_; }
    int64_t  lowest() const { return lowest_; }   /* -1 before a readable sample */
    uint64_t holds() const { return holds_; }
    uint64_t samples() const { return samples_; }
    int64_t  floor() const { return floor_; }
    int64_t  release() const { return release_; }

private:
    int64_t  floor_, release_;
    bool     held_ = false;
    int64_t  lowest_ = -1;
    uint64_t holds_ = 0, samples_ = 0;
};

/* A /proc/meminfo field ("MemAvailable", "SwapFree") in bytes from the file's text, -1 if absent. */
int64_t meminfo_field(const std::string& text, const char* key);
/* The same from /proc/meminfo now; and a /proc/self/status field ("VmRSS"). -1 if unreadable. */
int64_t meminfo_now(const char* key);
int64_t proc_status_now(const char* key);

}  /* namespace rad */
