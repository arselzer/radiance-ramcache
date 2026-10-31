#include "rad_test.h"
#include "mem/kv_live_budget.h"
#include <map>

using namespace rad;
namespace {
std::map<std::string, std::string> options() {
    return { { kKVLiveEnv, "1" }, { kKVLiveMinEnv, "64" }, { kKVLiveMaxEnv, "188" },
             { kKVLiveWarmEnv, "8" }, { kKVLiveHysteresisEnv, "4" }, { kKVLiveHeadroomEnv, "32" } };
}
bool parse(const std::map<std::string, std::string>& env, KVLiveOptions* out) {
    return kv_live_options([&](const char* key) -> const char* {
        const auto it = env.find(key); return it == env.end() ? nullptr : it->second.c_str();
    }, out, nullptr);
}
}  // namespace

TEST(kv_live_growth_defaults_off_and_requires_all_explicit_physical_budgets) {
    KVLiveOptions o;
    CHECK(parse({}, &o));
    CHECK(!o.enabled);
    auto env = options();
    REQUIRE(parse(env, &o));
    CHECK(o.enabled);
    CHECK_EQ(o.maximum_bytes, 188 * kKVLiveMiB);
    for (const char* key : { kKVLiveMinEnv, kKVLiveMaxEnv, kKVLiveWarmEnv,
                             kKVLiveHysteresisEnv, kKVLiveHeadroomEnv }) {
        auto bad = env; bad.erase(key); CHECK(!parse(bad, &o));
        bad = env; bad[key] = "-1"; CHECK(!parse(bad, &o));
        bad = env; bad[key] = "1 "; CHECK(!parse(bad, &o));
        bad = env; bad[key] = "8796093022208"; CHECK(!parse(bad, &o));
    }
    auto bad = env; bad[kKVLiveEnv] = "2"; CHECK(!parse(bad, &o));
    bad = env; bad[kKVLiveEnv] = "0"; CHECK(!parse(bad, &o));
    bad = env; bad[kKVLiveMaxEnv] = "0"; CHECK(!parse(bad, &o));
    bad = env; bad[kKVLiveMinEnv] = "189"; CHECK(!parse(bad, &o));
}

TEST(kv_live_budget_charges_partial_tail_actual_max_and_all_headroom_without_overflow) {
    KVLiveOptions o;
    REQUIRE(parse(options(), &o));
    KVLiveBudget b{ 187 * kKVLiveMiB, 188 * kKVLiveMiB, 187 * kKVLiveMiB,
                    2 * kKVLiveMiB, 0, 800 * kKVLiveMiB, 512 * kKVLiveMiB, 64 * kKVLiveMiB };
    CHECK(kv_live_budget_error(o, b) == nullptr);
    auto short_cap = o; short_cap.maximum_bytes = 187 * kKVLiveMiB;
    CHECK(kv_live_budget_error(short_cap, b) != nullptr);
    auto too_big = o; too_big.maximum_bytes = 190 * kKVLiveMiB;
    CHECK(kv_live_budget_error(too_big, b) != nullptr);
    b.device_free_bytes = (188 + 512 + 64) * kKVLiveMiB;
    CHECK(kv_live_budget_error(o, b) == nullptr);
    --b.device_free_bytes;
    CHECK(kv_live_budget_error(o, b) != nullptr);
    b.currently_backed_bytes = 100 * kKVLiveMiB;
    b.device_free_bytes = (88 + 512 + 64) * kKVLiveMiB;
    CHECK(kv_live_budget_error(o, b) == nullptr);
    b.late_bytes = INT64_MAX;
    CHECK(kv_live_budget_error(o, b) != nullptr);
    b = { 1, 2, 1, 2, 0, INT64_MAX, INT64_MAX, INT64_MAX };
    o.minimum_bytes = 0; o.maximum_bytes = 2;
    CHECK(kv_live_budget_error(o, b) != nullptr);
}

TEST(kv_live_runtime_funding_keeps_headroom_for_growth_and_allows_release_under_pressure) {
    CHECK(kv_live_funding_fits(80, 96, 188, 48, 32, true));
    CHECK(!kv_live_funding_fits(80, 97, 188, 48, 32, true));
    CHECK(!kv_live_funding_fits(80, 189, 188, 1000, 0, true));
    CHECK(kv_live_funding_fits(180, 100, 188, 0, 32, false));
    CHECK(!kv_live_funding_fits(189, 100, 188, 0, 32, false));
    CHECK(!kv_live_funding_fits(80, 96, 188, -1, 32, true));
}

RAD_TEST_MAIN()
