#include "rad_test.h"
#include "../core/sched/draft_experiment.h"
#include "../core/sched/draft_measure.h"
#include <limits>

using namespace rad;

TEST(dspark_prefix_policy_explicit_config_and_mutual_exclusion) {
    for (const char* off : { (const char*)nullptr, "off", "0" }) {
        DraftPrefixPolicy p; REQUIRE(DraftPrefixPolicy::parse(off, &p)); CHECK(!p.on);
        CHECK(DraftExperiment::capture(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                      nullptr, nullptr, nullptr, off).valid());
    }
    for (const char* bad : { "", "1", "v1", "v2:0.25:0.75", "v1:0.25", "v1:0.25:0.75:",
                            "v1:NaN:0.5", "v1:0.5:inf", "v1:-0.1:0.5", "v1:0.5:1.01",
                            "v1: 0.2:0.5", "v1:0x1:0.5", "v1:0.2:0.5\n", "v1:0.2:0.5junk" }) {
        DraftPrefixPolicy p; CHECK(!DraftPrefixPolicy::parse(bad, &p));
    }
    constexpr const char* spec = "v1:0.25:0.75";
    auto capture = [](const char* off, const char* cap, const char* gate, const char* conf,
                      const char* load, const char* defer, const char* prob, const char* early) {
        return DraftExperiment::capture(nullptr, off, cap, gate, conf, load, defer, prob, early,
                                        "v1:0.25:0.75");
    };
    auto p = capture(nullptr, nullptr, nullptr, nullptr, nullptr, "1", "1", nullptr);
    REQUIRE(p.valid()); CHECK(p.early_boundary()); CHECK(p.scope_required());
    CHECK(p.prefix_policy().on); CHECK_EQ(p.early_verify(), -1);
    CHECK_EQ(p.runtime_depth(5), 5); CHECK_EQ(std::strcmp(p.prefix_policy().spec, spec), 0);
    CHECK(!capture("1", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr).valid());
    CHECK(!capture(nullptr, "0", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr).valid());
    CHECK(!capture(nullptr, "2", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr).valid());
    CHECK(!capture(nullptr, nullptr, "1", nullptr, nullptr, nullptr, nullptr, nullptr).valid());
    CHECK(!capture(nullptr, nullptr, nullptr, "1", nullptr, nullptr, nullptr, nullptr).valid());
    CHECK(!capture(nullptr, nullptr, nullptr, nullptr, "1", nullptr, nullptr, nullptr).valid());
    for (const char* k : { "0", "1", "2" })
        CHECK(!capture(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, k).valid());
    char mutable_spec[] = "v1:0.25:0.75";
    const auto immutable = DraftExperiment::capture(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                   nullptr, nullptr, nullptr, mutable_spec);
    mutable_spec[5] = '9';
    CHECK_EQ(std::strcmp(immutable.prefix_policy().spec, spec), 0);
    CHECK(draft_prefix_policy_profile(1, false, 1, 5, 5, 2));
    CHECK(!draft_prefix_policy_profile(2, false, 1, 5, 5, 2));
    CHECK(!draft_prefix_policy_profile(1, true, 1, 5, 5, 2));
    CHECK(!draft_prefix_policy_profile(1, false, 0, 5, 5, 2));
    CHECK(!draft_prefix_policy_profile(1, false, 1, 5, 5, 3));
    CHECK(!draft_prefix_policy_profile(1, false, 1, 5, 4, 2));
}

TEST(dspark_prefix_policy_hierarchical_boundaries_and_numeric_fallback) {
    DraftPrefixPolicy p; REQUIRE(DraftPrefixPolicy::parse("v1:0.25:0.75", &p));
    CHECK_EQ(p.select(.125f, .875f, 0), 0);
    CHECK_EQ(p.select(.25f, .5f, 0), 1);
    CHECK_EQ(p.select(.25f, .75f, 0), 2);
    CHECK_EQ(p.select(1, 0, 0), 1); CHECK_EQ(p.select(0, 1, 0), 0);
    for (unsigned mask : { 1u, 2u, 3u }) {
        bool fallback = false; CHECK_EQ(p.select(0, 0, mask, &fallback), 2); CHECK(fallback);
    }
    for (float bad : { std::numeric_limits<float>::quiet_NaN(),
                       std::numeric_limits<float>::infinity(), -1.f, 2.f }) {
        bool fallback = false; CHECK_EQ(p.select(bad, .9f, 0, &fallback), 2); CHECK(fallback);
        CHECK_EQ(p.select(0, bad, 0, &fallback), 2); CHECK(fallback);
    }
    float obs[5] = { .5f, 4.f, 1.f, 10.f, 9.f };
    CHECK(draft_measure_signal(obs, 11, 9) == DraftMeasureSignal::PathMismatch);
    obs[0] = std::numeric_limits<float>::quiet_NaN();
    CHECK(draft_measure_signal(obs, 11, 9) == DraftMeasureSignal::PathMismatch);
    CHECK(draft_measure_signal(obs, 10, 9) == DraftMeasureSignal::InvalidNumeric);
    obs[0] = .5f; obs[1] = std::numeric_limits<float>::infinity();
    CHECK(draft_measure_signal(obs, 10, 9) == DraftMeasureSignal::InvalidNumeric);
}

TEST(dspark_prefix_policy_epoch_binding_and_mixed_action_costs) {
    DraftMeasure meter, fixed;
    DraftMeasureFrame f; f.seq = 7; f.card = 0; f.ctx = 50; f.anchor = 9;
    f.ids[0] = 10; f.ids[1] = 11; f.k = 0; f.query_rows = 2;
    f.epoch = 1; f.ready_ns = 100; f.wait_ns = 10; f.pure_decode = true;
    CHECK(draft_measure_frame_matches(f, 7, 0, 50, 9, f.ids, 2));
    CHECK(!draft_measure_frame_matches(f, 8, 0, 50, 9, f.ids, 2));
    CHECK(!draft_measure_frame_matches(f, 7, 1, 50, 9, f.ids, 2));
    CHECK(!draft_measure_frame_matches(f, 7, 0, 51, 9, f.ids, 2));
    CHECK(!draft_measure_frame_matches(f, 7, 0, 50, 8, f.ids, 2));
    int32_t wrong[2] = { 10, 12 };
    CHECK(!draft_measure_frame_matches(f, 7, 0, 50, 9, wrong, 2));
    auto stale = f; stale.epoch = 0;
    CHECK(!draft_measure_frame_matches(stale, 7, 0, 50, 9, f.ids, 2));
    for (int k : { 0, 2, 1, 2 }) {
        f.k = k; f.invalid_depths = f.epoch == 2 ? 2 : 0;
        meter.ready(f, true, true); fixed.ready(f, true);
        const int accepted = meter.close ? meter.closing.k : 0;
        const uint64_t output = f.epoch == 1 ? 1 : meter.closing.output + 1 + accepted;
        meter.committed(true, true, output, accepted, true);
        fixed.committed(true, true, output, accepted);
        ++f.epoch; f.ready_ns += 100;
    }
    CHECK_EQ(meter.stats.n[0], (uint64_t)1); CHECK_EQ(meter.stats.n[1], (uint64_t)1);
    CHECK_EQ(meter.stats.n[2], (uint64_t)1); CHECK_EQ(meter.stats.invalid_queries, (uint64_t)1);
    CHECK_EQ(fixed.stats.n[0] + fixed.stats.n[1] + fixed.stats.n[2], (uint64_t)0);
    meter.no_query(); CHECK_EQ(meter.pending.epoch, (uint64_t)0);
    meter.ready(f, false, true); CHECK(!meter.close);
    meter.committed(false, false, 0, 0, true); CHECK_EQ(meter.pending.epoch, (uint64_t)0);
}

RAD_TEST_MAIN()
