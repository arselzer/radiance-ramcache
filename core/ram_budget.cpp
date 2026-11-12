/* ram_budget.cpp -- see ram_budget.h. */
#include "ram_budget.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace rad {

namespace {

constexpr int64_t kMiB = (int64_t)1 << 20;
constexpr int64_t kGrain = (int64_t)2 << 20;   /* a slab's charge is whole 2 MiB */

int64_t sat_add(int64_t a, int64_t b) {
    if (b > 0 && a > INT64_MAX - b) return INT64_MAX;
    return a + b;
}

std::string mib(int64_t b) {
    char s[64];
    if (b == INT64_MAX) return "unbounded";
    std::snprintf(s, sizeof s, "%lld MiB", (long long)(b / kMiB));
    return s;
}

}  /* namespace */

RamBudget ram_budget_resolve(const RamBudgetIn& in) {
    RamBudget out;
    const size_t n = std::max(in.backed.size(), in.rank_cap.size());
    out.rank_limit.assign(n, 0);
    for (size_t i = 0; i < n; ++i)
        out.rank_limit[i] = i < in.backed.size() ? std::max<int64_t>(in.backed[i], 0) : 0;

    for (const RamBudgetItem& it : in.held)
        out.lines.push_back("held  " + mib(it.bytes) + "  " + it.name + " (" + it.source + ")");
    for (const RamBudgetItem& it : in.later) {
        out.later = sat_add(out.later, std::max<int64_t>(it.bytes, 0));
        out.lines.push_back("later " + mib(it.bytes) + "  " + it.name + " (" + it.source + ")");
    }
    out.lines.push_back("reserve " + mib(in.reserve));
    out.lines.push_back("MemAvailable " + (in.mem_available < 0 ? std::string("unreadable")
                                                                 : mib(in.mem_available)) +
                        ", SwapFree " + (in.swap_free < 0 ? std::string("unreadable")
                                                          : mib(in.swap_free)) +
                        " (not counted: a pinned slab does not swap), pin room " + mib(in.pin_room));

    if (in.mem_available < 0 || n == 0 || in.reserve < 0) {
        out.total = 0;
        for (int64_t l : out.rank_limit) out.total = sat_add(out.total, l);
        out.lines.push_back("no readable MemAvailable: each cache held at what it holds");
        return out;
    }
    out.ok = true;
    const int64_t avail = std::min(in.mem_available, in.pin_room < 0 ? 0 : in.pin_room);
    const __int128 r = (__int128)avail - out.later - in.reserve;
    out.room = r <= 0 ? 0 : (int64_t)r;
    auto cap_of = [&](size_t i) {
        return i >= in.rank_cap.size() || in.rank_cap[i] < 0 ? INT64_MAX : in.rank_cap[i];
    };
    int64_t active = 0;
    for (size_t i = 0; i < n; ++i) active += cap_of(i) != 0 ? 1 : 0;
    const int64_t share = active ? out.room / active / kGrain * kGrain : 0;
    for (size_t i = 0; i < n; ++i) {
        const int64_t backed = out.rank_limit[i];
        const int64_t cap = cap_of(i);
        /* A rank already past its cap keeps what it holds: the limit never asks for an eviction. */
        const int64_t lim = std::max(backed, std::min(cap, sat_add(backed, share)));
        out.rank_limit[i] = lim;
        out.growth = sat_add(out.growth, lim - backed);
    }
    for (int64_t l : out.rank_limit) out.total = sat_add(out.total, l);
    if (in.cap > 0 && out.total > in.cap) out.lines.push_back("the configured cap binds");
    out.lines.push_back("room " + mib(out.room) + " = " + mib(avail) + " - later " + mib(out.later) +
                        " - reserve " + mib(in.reserve) + "; the caches may grow " +
                        mib(out.growth) + ", to " + mib(out.total) + " in all");
    return out;
}

RamGuardState::Event RamGuardState::observe(int64_t avail) {
    if (avail < 0) return Event::None;
    ++samples_;
    if (lowest_ < 0 || avail < lowest_) lowest_ = avail;
    if (!held_ && avail < floor_) {
        held_ = true;
        ++holds_;
        return Event::Hold;
    }
    if (held_ && avail >= release_) {
        held_ = false;
        return Event::Release;
    }
    return Event::None;
}

int64_t meminfo_field(const std::string& text, const char* key) {
    std::istringstream f(text);
    const std::string want = std::string(key) + ": %lld kB";
    std::string line;
    long long kb = 0;
    while (std::getline(f, line))
        if (std::sscanf(line.c_str(), want.c_str(), &kb) == 1) return (int64_t)kb * 1024;
    return -1;
}

static int64_t file_field(const char* path, const char* key) {
    std::ifstream f(path);
    if (!f) return -1;
    std::stringstream ss;
    ss << f.rdbuf();
    return meminfo_field(ss.str(), key);
}

int64_t meminfo_now(const char* key) { return file_field("/proc/meminfo", key); }
int64_t proc_status_now(const char* key) { return file_field("/proc/self/status", key); }

}  /* namespace rad */
