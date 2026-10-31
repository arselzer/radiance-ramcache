/* Device-free policy proof. This does not map VRAM or enable runtime KV growth. */
#include "rad_test.h"
#include "mem/kv_growth.h"
#include <numeric>

using namespace rad;

namespace {
constexpr int64_t G = 256;
KVBackingLimits limits(int n = 16) { return {n * G, G, 2 * G, n * G, G, 2 * G}; }
void apply(std::vector<uint8_t>& map, const std::vector<KVBackingRange>& rows, uint8_t value) {
    for (const auto& r : rows) {
        REQUIRE_EQ(r.offset % G, int64_t(0));
        REQUIRE_EQ(r.bytes % G, int64_t(0));
        for (int64_t i = r.offset / G; i < (r.offset + r.bytes) / G; ++i) {
            REQUIRE(i >= 0 && size_t(i) < map.size());
            CHECK(map[size_t(i)] != value);
            map[size_t(i)] = value;
        }
    }
}
int64_t bytes(const std::vector<uint8_t>& m) {
    return std::accumulate(m.begin(), m.end(), int64_t(0)) * G;
}
void covers(const std::vector<uint8_t>& m, const std::vector<KVBackingRange>& need) {
    for (const auto& r : need) if (r.bytes)
        for (int64_t i = r.offset / G; i <= (r.offset + r.bytes - 1) / G; ++i)
            CHECK_EQ(m[size_t(i)], uint8_t(1));
}
}

TEST(kv_growth_partial_pages_share_mapping_granules_and_keep_fixed_and_prefix_holders) {
    auto lim = limits();
    std::vector<uint8_t> before(16, 0);
    before[0] = before[1] = before[8] = 1;
    // Two tightly packed rows can straddle a granule. Retained prefixes are still holders.
    const std::vector<KVBackingRange> holders{{0, 80}, {G - 5, 68}, {8 * G + 7, 1}, {12 * G, G}};
    KVBackingPlan p;
    REQUIRE(kv_backing_plan(lim, before, holders, &p) == KVBackingResult::Ok);
    CHECK_EQ(p.required_bytes, 4 * G);
    CHECK_EQ(p.target_bytes, 5 * G);
    auto actual = before;
    apply(actual, p.release_first, 0);
    covers(actual, {{0, 80}, {G - 5, 68}, {8 * G + 7, 1}});
    apply(actual, p.commit, 1);
    covers(actual, holders);
    apply(actual, p.release_after, 0);
    CHECK(actual == p.target);
    CHECK(bytes(actual) <= lim.maximum_bytes);
}

TEST(kv_growth_never_releases_a_granule_with_a_live_byte_and_defers_small_shrink) {
    auto lim = limits(8);
    std::vector<uint8_t> before(8, 0);
    for (int i = 0; i < 4; ++i) before[size_t(i)] = 1;
    KVBackingPlan p;
    REQUIRE(kv_backing_plan(lim, before, {{G + 255, 1}}, &p) == KVBackingResult::Ok);
    CHECK(p.target == before); // mandatory1 + warm1/min2, excess2 == hysteresis
    CHECK(p.commit.empty() && p.release_first.empty() && p.release_after.empty());
    lim.shrink_hysteresis = 0;
    REQUIRE(kv_backing_plan(lim, before, {{G + 255, 1}}, &p) == KVBackingResult::Ok);
    CHECK_EQ(p.target_bytes, 2 * G);
    CHECK_EQ(p.target[1], uint8_t(1));
}

TEST(kv_growth_budget_reduction_and_disjoint_writes_release_only_unheld_room_before_commit) {
    auto lim = limits(8);
    lim.maximum_bytes = 3 * G;
    lim.warm_bytes = 0;
    lim.shrink_hysteresis = 0;
    std::vector<uint8_t> before(8, 0);
    for (int i = 0; i < 5; ++i) before[size_t(i)] = 1;
    KVBackingPlan p;
    const std::vector<KVBackingRange> need{{0, G}, {7 * G, G}};
    REQUIRE(kv_backing_plan(lim, before, need, &p) == KVBackingResult::Ok);
    CHECK_EQ(p.peak_bytes, 3 * G);
    auto actual = before;
    apply(actual, p.release_first, 0);
    covers(actual, {{0, G}});
    apply(actual, p.commit, 1);
    CHECK(bytes(actual) <= lim.maximum_bytes);
    covers(actual, need);
    apply(actual, p.release_after, 0);
    CHECK(actual == p.target);
}

TEST(kv_growth_refuses_missing_capacity_and_bad_ranges_without_changing_the_plan) {
    auto lim = limits(4);
    lim.maximum_bytes = 2 * G;
    std::vector<uint8_t> before(4, 0);
    KVBackingPlan p;
    p.target_bytes = 77;
    CHECK(kv_backing_plan(lim, before, {{0, 3 * G}}, &p) == KVBackingResult::BudgetExceeded);
    CHECK_EQ(p.target_bytes, int64_t(77));
    for (auto r : std::vector<KVBackingRange>{{-1, 1}, {0, -1}, {4 * G, 1}, {1, INT64_MAX}}) {
        CHECK(kv_backing_plan(lim, before, {r}, &p) == KVBackingResult::Invalid);
        CHECK_EQ(p.target_bytes, int64_t(77));
    }
    before[0] = 2;
    CHECK(kv_backing_plan(lim, before, {}, &p) == KVBackingResult::Invalid);
    lim.minimum_bytes = G + 1; lim.maximum_bytes = 2 * G - 1;
    before[0] = 0;
    CHECK(kv_backing_plan(lim, before, {}, &p) == KVBackingResult::BudgetExceeded);
}

TEST(kv_growth_sequence_and_prefix_release_sweeps_preserve_held_bytes_and_joint_peak_budget) {
    uint32_t state = 0x4132f09u;
    auto draw = [&]() { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; };
    for (int trial = 0; trial < 3000; ++trial) {
        auto lim = limits(32);
        lim.minimum_bytes = int64_t(draw() % 4) * G;
        lim.maximum_bytes = int64_t(4 + draw() % 29) * G;
        lim.warm_bytes = int64_t(draw() % 8) * G;
        lim.shrink_hysteresis = int64_t(draw() % 5) * G;
        std::vector<uint8_t> before(32, 0), mandatory(32, 0);
        std::vector<KVBackingRange> need;
        for (int i = 0; i < 32; ++i) {
            before[size_t(i)] = (draw() % 3 == 0);
            if (draw() % 7 == 0) {
                mandatory[size_t(i)] = 1;
                need.push_back({i * G + int64_t(draw() % G), 1});
            }
        }
        KVBackingPlan p;
        const auto rc = kv_backing_plan(lim, before, need, &p);
        if (bytes(mandatory) > lim.maximum_bytes) {
            CHECK(rc == KVBackingResult::BudgetExceeded);
            continue;
        }
        REQUIRE(rc == KVBackingResult::Ok);
        auto actual = before;
        apply(actual, p.release_first, 0);
        for (size_t i = 0; i < actual.size(); ++i)
            if (mandatory[i] && before[i]) CHECK_EQ(actual[i], uint8_t(1));
        apply(actual, p.commit, 1);
        CHECK(bytes(actual) <= lim.maximum_bytes);
        CHECK_EQ(bytes(actual), p.peak_bytes);
        covers(actual, need);
        apply(actual, p.release_after, 0);
        CHECK(actual == p.target);
        CHECK(bytes(actual) >= lim.minimum_bytes);
    }
}

RAD_TEST_MAIN()
