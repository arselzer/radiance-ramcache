/* ram_budget_test.cpp -- RADIANCE_EXPERIMENTAL_DS41_RAM_BUDGET's arithmetic and the guard's state
 * machine (core/ram_budget.h), on synthetic MemAvailable figures: no /proc, no card.
 *
 * The property that matters: whatever the figures, the caches' growth plus everything counted to
 * grow later plus the reserve never exceeds MemAvailable, so the configuration cannot drive the
 * host into the OOM killer (2026-11-12: a 192 GiB cache with 256 sequences did, and an amdgpu bug
 * on the OOM path panicked the kernel). */
#include "rad_test.h"

#include "ram_budget.h"

#include <algorithm>
#include <cstdint>
#include <random>

using namespace rad;

namespace {

constexpr int64_t MiB = (int64_t)1 << 20;
constexpr int64_t GiB = (int64_t)1 << 30;

RamBudgetIn four_ranks(int64_t avail, int64_t reserve, int64_t cap_total) {
    RamBudgetIn in;
    in.mem_available = avail;
    in.reserve = reserve;
    in.cap = cap_total;
    in.backed.assign(4, 0);
    in.rank_cap.assign(4, cap_total > 0 ? cap_total / 4 : -1);
    return in;
}

int64_t later_sum(const RamBudgetIn& in) {
    int64_t s = 0;
    for (const auto& it : in.later) s += it.bytes;
    return s;
}

bool within(const RamBudgetIn& in, const RamBudget& b) {
    int64_t growth = 0;
    for (size_t i = 0; i < b.rank_limit.size(); ++i) {
        const int64_t d = b.rank_limit[i] - (i < in.backed.size() ? in.backed[i] : 0);
        if (d < 0) return false;
        growth += d;
    }
    if (growth != b.growth) return false;
    if (growth == 0) return true;
    const int64_t avail = std::min(in.mem_available, in.pin_room);
    return (__int128)growth + later_sum(in) + in.reserve <= (__int128)avail;
}

}  /* namespace */

/* The 2026-11-12 box: 200 GiB available after startup, 256 sequences at 100 MiB, a 10 GiB
 * reserve. The cache gets the rest in whole 2 MiB a rank, and the configured 192 GiB would not
 * have fit. */
TEST(the_cache_gets_what_is_left_after_later_growth_and_the_reserve) {
    RamBudgetIn in = four_ranks(200 * GiB, 10 * GiB, 192 * GiB);
    in.later.push_back({ "per-sequence", 256 * 100 * MiB, "test" });
    const RamBudget b = ram_budget_resolve(in);
    REQUIRE(b.ok);
    CHECK_EQ(b.later, 256 * 100 * MiB);
    CHECK_EQ(b.room, 200 * GiB - 256 * 100 * MiB - 10 * GiB);
    REQUIRE(b.rank_limit.size() == 4u);
    for (int64_t l : b.rank_limit) {
        CHECK_EQ(l % (2 * MiB), 0);
        CHECK(l <= 48 * GiB);
        CHECK_EQ(l, b.room / 4 / (2 * MiB) * (2 * MiB));
    }
    CHECK(b.total < 192 * GiB);
    CHECK(within(in, b));
    CHECK(!b.lines.empty());
}

/* With room to spare the configured figure binds, rank by rank. */
TEST(a_configured_cap_binds_when_the_host_has_more) {
    RamBudgetIn in = four_ranks(240 * GiB, 10 * GiB, 64 * GiB);
    const RamBudget b = ram_budget_resolve(in);
    REQUIRE(b.ok);
    for (int64_t l : b.rank_limit) CHECK_EQ(l, 16 * GiB);
    CHECK_EQ(b.total, 64 * GiB);
    CHECK_EQ(b.growth, 64 * GiB);
}

/* Less available than later growth and the reserve: no growth at all, never a negative limit. */
TEST(a_host_short_of_the_reserve_grows_nothing) {
    RamBudgetIn in = four_ranks(8 * GiB, 10 * GiB, 0);
    in.later.push_back({ "x", 1 * GiB, "test" });
    const RamBudget b = ram_budget_resolve(in);
    REQUIRE(b.ok);
    CHECK_EQ(b.room, 0);
    CHECK_EQ(b.growth, 0);
    for (int64_t l : b.rank_limit) CHECK_EQ(l, 0);
}

/* What a rank's cache holds is already out of MemAvailable: it keeps it, and only grows by its
 * share. A rank past its cap keeps what it holds (no eviction is asked for). */
TEST(backed_slabs_are_kept_and_counted_once) {
    RamBudgetIn in = four_ranks(100 * GiB, 10 * GiB, 160 * GiB);
    in.backed = { 4 * GiB, 0, 45 * GiB, 30 * GiB };
    const RamBudget b = ram_budget_resolve(in);
    REQUIRE(b.ok);
    const int64_t share = 90 * GiB / 4;
    CHECK_EQ(share % (2 * MiB), 0);
    CHECK_EQ(b.rank_limit[0], 4 * GiB + share);
    CHECK_EQ(b.rank_limit[1], share);
    CHECK_EQ(b.rank_limit[2], 45 * GiB);      /* past its 40 GiB cap: held where it is */
    CHECK_EQ(b.rank_limit[3], 40 * GiB);      /* capped */
    CHECK_EQ(b.growth, 2 * share + 10 * GiB);
    CHECK(within(in, b));
}

/* A rank without a cache (rank_cap 0) takes no share: the others split the room. */
TEST(a_rank_without_a_cache_takes_no_share) {
    RamBudgetIn in = four_ranks(50 * GiB, 10 * GiB, 0);
    in.rank_cap = { -1, 0, -1, 0 };
    const RamBudget b = ram_budget_resolve(in);
    REQUIRE(b.ok);
    CHECK_EQ(b.rank_limit[1], 0);
    CHECK_EQ(b.rank_limit[3], 0);
    CHECK_EQ(b.rank_limit[0], 20 * GiB);
    CHECK_EQ(b.rank_limit[2], 20 * GiB);
}

/* The amdgpu pin cap binds like MemAvailable: the slabs are pinned. */
TEST(the_pin_room_binds_below_mem_available) {
    RamBudgetIn in = four_ranks(200 * GiB, 10 * GiB, 0);
    in.pin_room = 50 * GiB;
    const RamBudget b = ram_budget_resolve(in);
    REQUIRE(b.ok);
    CHECK_EQ(b.room, 40 * GiB);
    CHECK(b.growth <= 40 * GiB);
}

/* Unreadable MemAvailable: not ok, every rank held at what it holds. */
TEST(an_unreadable_mem_available_holds_every_cache) {
    RamBudgetIn in = four_ranks(-1, 10 * GiB, 64 * GiB);
    in.backed = { 1 * GiB, 2 * GiB, 0, 0 };
    const RamBudget b = ram_budget_resolve(in);
    CHECK(!b.ok);
    CHECK_EQ(b.rank_limit[0], 1 * GiB);
    CHECK_EQ(b.rank_limit[1], 2 * GiB);
    CHECK_EQ(b.rank_limit[2], 0);
    CHECK_EQ(b.growth, 0);
}

/* Saturation: a later item near INT64_MAX does not wrap into room. */
TEST(huge_later_items_saturate_rather_than_wrap) {
    RamBudgetIn in = four_ranks(200 * GiB, 10 * GiB, 0);
    in.later.push_back({ "a", INT64_MAX - 5, "test" });
    in.later.push_back({ "b", INT64_MAX - 5, "test" });
    const RamBudget b = ram_budget_resolve(in);
    REQUIRE(b.ok);
    CHECK_EQ(b.later, INT64_MAX);
    CHECK_EQ(b.growth, 0);
}

/* The property, over many random configurations: growth + later + reserve <= MemAvailable, every
 * limit at least what is backed and within its cap unless backed already exceeds it. */
TEST(random_figures_never_exceed_mem_available_less_the_reserve) {
    std::mt19937_64 rng(20261112);
    for (int k = 0; k < 20000; ++k) {
        const int n = 1 + (int)(rng() % 5);
        RamBudgetIn in;
        in.mem_available = (int64_t)(rng() % (256 * GiB));
        in.reserve = (int64_t)(rng() % (16 * GiB));
        in.pin_room = rng() % 4 ? INT64_MAX : (int64_t)(rng() % (256 * GiB));
        for (int i = 0; i < n; ++i) {
            in.backed.push_back((int64_t)(rng() % (8 * GiB)));
            const int c = (int)(rng() % 3);
            in.rank_cap.push_back(c == 0 ? -1 : c == 1 ? 0 : (int64_t)(rng() % (64 * GiB)));
        }
        const int m = (int)(rng() % 4);
        for (int i = 0; i < m; ++i) in.later.push_back({ "r", (int64_t)(rng() % (64 * GiB)), "test" });
        const RamBudget b = ram_budget_resolve(in);
        REQUIRE(b.ok);
        int64_t growth = 0;
        for (int i = 0; i < n; ++i) {
            const int64_t d = b.rank_limit[(size_t)i] - in.backed[(size_t)i];
            REQUIRE(d >= 0);
            if (in.rank_cap[(size_t)i] == 0) REQUIRE(d == 0);
            if (in.rank_cap[(size_t)i] > 0 && in.backed[(size_t)i] <= in.rank_cap[(size_t)i])
                REQUIRE(b.rank_limit[(size_t)i] <= in.rank_cap[(size_t)i]);
            growth += d;
        }
        REQUIRE(growth == b.growth);
        const int64_t avail = std::min(in.mem_available, in.pin_room);
        if (growth > 0) REQUIRE((__int128)growth + b.later + in.reserve <= (__int128)avail);
        REQUIRE(growth <= b.room);
    }
}

/* The guard: hold below the floor, once; stay held between floor and release; release at the
 * reserve; unreadable samples change nothing; lowest tracked. */
TEST(the_guard_holds_below_the_floor_and_releases_at_the_reserve) {
    RamGuardState g(5 * GiB, 10 * GiB);
    CHECK(g.observe(20 * GiB) == RamGuardState::Event::None);
    CHECK(!g.held());
    CHECK(g.observe(6 * GiB) == RamGuardState::Event::None);
    CHECK(g.observe(5 * GiB) == RamGuardState::Event::None);            /* at the floor: not below */
    CHECK(g.observe(5 * GiB - 1) == RamGuardState::Event::Hold);
    CHECK(g.held());
    CHECK(g.observe(3 * GiB) == RamGuardState::Event::None);            /* already held */
    CHECK(g.observe(-1) == RamGuardState::Event::None);                 /* unreadable */
    CHECK(g.held());
    CHECK(g.observe(9 * GiB) == RamGuardState::Event::None);            /* hysteresis */
    CHECK(g.observe(10 * GiB) == RamGuardState::Event::Release);
    CHECK(!g.held());
    CHECK(g.observe(4 * GiB) == RamGuardState::Event::Hold);
    CHECK_EQ(g.holds(), (uint64_t)2);
    CHECK_EQ(g.lowest(), 3 * GiB);
    CHECK_EQ(g.samples(), (uint64_t)10);
}

/* A release below the floor is raised to the floor: no oscillation on one sample. */
TEST(the_guard_release_is_never_below_its_floor) {
    RamGuardState g(8 * GiB, 1 * GiB);
    CHECK_EQ(g.release(), 8 * GiB);
    CHECK(g.observe(7 * GiB) == RamGuardState::Event::Hold);
    CHECK(g.observe(7 * GiB) == RamGuardState::Event::None);
    CHECK(g.observe(8 * GiB) == RamGuardState::Event::Release);
    RamGuardState never(-1, -1);
    CHECK(never.observe(0) == RamGuardState::Event::None);
    CHECK_EQ(never.lowest(), 0);
}

TEST(meminfo_fields_parse_in_bytes) {
    const std::string t = "MemTotal:       263742468 kB\nMemFree:         1000 kB\n"
                          "MemAvailable:   209715200 kB\nSwapFree:        4096 kB\n";
    CHECK_EQ(meminfo_field(t, "MemAvailable"), (int64_t)209715200 * 1024);
    CHECK_EQ(meminfo_field(t, "SwapFree"), (int64_t)4096 * 1024);
    CHECK_EQ(meminfo_field(t, "MemTotal"), (int64_t)263742468 * 1024);
    CHECK_EQ(meminfo_field(t, "Cached"), -1);
    CHECK_EQ(meminfo_field("VmRSS:\t  12345 kB\n", "VmRSS"), (int64_t)12345 * 1024);
}

RAD_TEST_MAIN()
