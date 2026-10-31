#include "rad_test.h"
#include "sched/draft_experiment.h"
#include <limits>
using namespace rad;

namespace {
DraftExperiment nq(const char* flag = "1", const char* early = "0", const char* defer = "1",
                   const char* prob = nullptr, const char* policy = nullptr) {
    return DraftExperiment::capture(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                    defer, prob, early, policy, flag);
}
DraftQueryFreeFrame frame(uint64_t boundary, int step = -1) {
    DraftQueryFreeFrame f;
    f.seq = 17; f.slot = 2; f.card = 0; f.boundary = boundary;
    f.step = step < 0 ? (int)boundary : step;
    f.input_ctx = 100 + (int)boundary; f.ctx = f.input_ctx + 1;
    f.output_before = boundary + 3; f.ready_ns = 100 * boundary; f.wait_ns = 7;
    f.token = 10 + (int)boundary; f.context_rows = 1; f.pure_decode = true;
    return f;
}
}
TEST(query_free_is_explicit_fixed0_dx1_only_and_does_not_disable_the_head) {
    const auto x = nq(); REQUIRE(x.valid()); CHECK(x.query_free()); CHECK(!x.compute_off());
    CHECK(x.head_enabled(true)); CHECK(x.early_boundary()); CHECK_EQ(x.runtime_depth(5), 0);
    CHECK(!x.prob_capture()); CHECK(!x.prefix_policy().on);
    for (const char* bad : { "", "on", "2", "01" }) CHECK(!nq(bad).valid());
    for (const char* bad : { "off", "1", "2" }) CHECK(!nq("1", bad).valid());
    CHECK(!nq("1", "0", "0").valid()); CHECK(!nq("1", "0", nullptr).valid());
    CHECK(!nq("1", "0", "1", "1").valid());
    CHECK(!nq("1", "0", "1", nullptr, "v1:0.25:0.75").valid());
    CHECK(!nq("0").query_free());
    CHECK(!DraftExperiment::capture(nullptr, "1", nullptr, nullptr, nullptr, nullptr,
                                    "1", nullptr, "0", nullptr, "1").valid());
}
TEST(query_free_profile_rejects_unsupported_pipeline_shapes) {
    CHECK(draft_query_free_profile(1, false, 1, 5, 5, 2, true, true, true, true));
    CHECK(!draft_query_free_profile(4, false, 1, 5, 5, 2, true, true, true, true));
    CHECK(!draft_query_free_profile(1, true, 1, 5, 5, 2, true, true, true, true));
    CHECK(!draft_query_free_profile(1, false, 0, 5, 5, 2, true, true, true, true));
    CHECK(!draft_query_free_profile(1, false, 1, 5, 4, 2, true, true, true, true));
    CHECK(!draft_query_free_profile(1, false, 1, 5, 5, 3, true, true, true, true));
    CHECK(!draft_query_free_profile(1, false, 1, 5, 5, 2, false, true, true, true));
    CHECK(!draft_query_free_profile(1, false, 1, 5, 5, 2, true, false, true, true));
    CHECK(!draft_query_free_profile(1, false, 1, 5, 5, 2, true, true, false, true));
    CHECK(!draft_query_free_profile(1, false, 1, 5, 5, 2, true, true, true, false));
}
TEST(query_free_checks_actual_request_sampling_and_extra_state) {
    Request r; CHECK(!draft_query_free_request(r)); r.sp.temp = 0;
    CHECK(draft_query_free_request(r)); r.sp.temp = 1; r.sp.top_k = 1;
    CHECK(draft_query_free_request(r)); r.sp.temp = std::numeric_limits<float>::quiet_NaN();
    CHECK(!draft_query_free_request(r)); r.sp.temp = 0;
    r.sp.xtc_probability = .1f; CHECK(!draft_query_free_request(r)); r.sp.xtc_probability = 0;
    r.sp.grammar = "root ::= 'x'"; CHECK(!draft_query_free_request(r)); r.sp.grammar.clear();
    r.sp.grammar_lazy = true; CHECK(!draft_query_free_request(r)); r.sp.grammar_lazy = false;
    r.sp.thinking_budget = 0; CHECK(!draft_query_free_request(r)); r.sp.thinking_budget = -1;
    r.sp.use_control_vector = true; CHECK(!draft_query_free_request(r)); r.sp.use_control_vector = false;
    r.sp.n_probs = 0; CHECK(!draft_query_free_request(r)); r.sp.n_probs = -1;
    r.lora_slot = 0; CHECK(!draft_query_free_request(r)); r.lora_slot = -1;
    r.score = true; CHECK(!draft_query_free_request(r)); r.score = false;
    r.n_draft = 1; CHECK(!draft_query_free_request(r)); r.n_draft = 0;
    r.may_speculate = false; CHECK(!draft_query_free_request(r)); r.may_speculate = true;
    CHECK(draft_query_free_request(r));
}
TEST(query_free_cycles_have_no_proposal_epoch_and_bind_two_actual_commits) {
    DraftQueryFreeMeter m;
    auto a = frame(1); m.ready(a); CHECK(!m.committed(true, a.output_before + 1));
    auto b = frame(2); m.ready(b); CHECK(m.committed(true, b.output_before + 1));
    CHECK_EQ(m.n, (uint64_t)1); CHECK_EQ(m.ns, (uint64_t)100);
    CHECK_EQ(m.emitted, (uint64_t)1); CHECK_EQ(m.wait_ns, (uint64_t)7);
    CHECK_EQ(m.boundaries, (uint64_t)2); CHECK_EQ(m.waits, (uint64_t)2);
    CHECK_EQ(m.context_rows, (uint64_t)2); CHECK_EQ(m.pending.boundary, (uint64_t)2);
}
TEST(query_free_meter_rejects_gaps_reuse_prefill_time_and_context_discontinuity) {
    for (int fault = 0; fault < 10; ++fault) {
        DraftQueryFreeMeter m; auto a = frame(1); auto b = frame(2);
        m.ready(a); CHECK(!m.committed(true, a.output_before + 1));
        switch (fault) {
            case 0: b.seq++; break; case 1: b.slot++; break; case 2: b.card++; break;
            case 3: b.step++; break; case 4: b.boundary++; break;
            case 5: b.ready_ns = a.ready_ns; break; case 6: b.input_ctx++; break;
            case 7: b.output_before++; break; case 8: b.pure_decode = false; break;
            case 9: m.pause(); break;
        }
        m.ready(b); CHECK(!m.committed(true, b.output_before + 1)); CHECK_EQ(m.n, (uint64_t)0);
    }
}
TEST(query_free_meter_excludes_cancel_terminal_wrong_commit_and_first_decode) {
    for (int fault = 0; fault < 3; ++fault) {
        DraftQueryFreeMeter m; auto a = frame(1); auto b = frame(2); auto c = frame(3);
        if (fault == 2) a.pure_decode = false;
        m.ready(a); CHECK(!m.committed(true, a.output_before + 1));
        m.ready(b); CHECK(!m.committed(fault == 2, b.output_before + (fault == 1 ? 2 : 1)));
        m.ready(c); CHECK_EQ(m.committed(true, c.output_before + 1), fault == 2);
        CHECK_EQ(m.end_cancel, (uint64_t)(fault == 2 ? 0 : 1));
    }
}
RAD_TEST_MAIN()
