#include "rad_test.h"
#include "sched/draft_experiment.h"
using namespace rad;

TEST(dspark_probability_capture_is_explicit_and_never_a_depth_control) {
    for (const char* flag : { (const char*)nullptr, "0", "1" }) {
        const auto x = DraftExperiment::capture(nullptr, nullptr, "2", nullptr, nullptr, nullptr, nullptr, flag);
        CHECK(x.valid());
        CHECK_EQ(x.prob_capture(), flag && flag[0] == '1');
        CHECK_EQ(x.runtime_depth(5), 2);
        CHECK(x.head_enabled(true));
    }
    CHECK(!DraftExperiment::capture(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, "yes").valid());
    CHECK(!DraftExperiment::capture(nullptr, "1", nullptr, nullptr, nullptr, nullptr, nullptr, "1").valid());
}

TEST(dspark_probability_pairs_ids_positions_and_reached_prefix) {
    DraftProbPending p;
    p.epoch = 17; p.anchor = 40; p.n = 2; p.token[0] = 7; p.token[1] = 8;
    const int32_t tokens[] = { 7, 8 }, changed[] = { 9, 8 };
    for (int accepted = 0; accepted <= 2; ++accepted)
        for (int j = 0; j < 2; ++j) {
            const auto o = draft_prob_outcome(p, j, 40, tokens, 2, accepted);
            CHECK(o.matched); CHECK(o.verified);
            CHECK_EQ(o.reached, j <= accepted);
            CHECK_EQ(o.accepted, j < accepted);
        }
    CHECK(!draft_prob_outcome(p, 0, 41, tokens, 2, 2).matched);
    CHECK(!draft_prob_outcome(p, 1, 40, changed, 2, 2).matched);
    CHECK(!draft_prob_outcome(p, 0, 40, nullptr, 2, 2).matched);
    const auto capped = draft_prob_outcome(p, 1, 40, tokens, 1, 1);
    CHECK(capped.matched); CHECK(!capped.verified); CHECK(!capped.reached);
    const auto held = draft_prob_outcome(p, 0, 40, nullptr, 0, 0);
    CHECK(held.matched); CHECK(!held.verified); CHECK(!held.reached);
}
RAD_TEST_MAIN()
