#include "rad_test.h"
#include "sched/draft_experiment.h"
using namespace rad;
TEST(query_copy_opt_in_requires_the_same_fixed0_dx_profile) {
    const auto capture = [](const char* value, const char* early, const char* dx, const char* nq = "0") {
        return DraftExperiment::capture(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                        dx, nullptr, early, nullptr, nq, value);
    };
    CHECK(!capture(nullptr, "0", "1").query_copy_stats());
    CHECK(capture("0", "0", "1").valid());
    CHECK(capture("1", "0", "1").query_copy_stats()); CHECK(capture("1", "0", "1", "1").valid());
    CHECK(!capture("1", "1", "1").valid()); CHECK(!capture("1", "0", "0").valid());
    CHECK(!capture("yes", "0", "1").valid());
}
TEST(query_copy_configured_nq_has_no_pending_owner_or_fake_epoch) {
    DraftQueryCopyStats s; REQUIRE(s.configure(40, 128));
    CHECK(!s.pending()); CHECK_EQ(s.started, (uint64_t)0); CHECK_EQ(s.bound_epochs, (uint64_t)0);
    CHECK_EQ(s.last_epoch, (uint64_t)0); CHECK_EQ(s.last_seq, (uint64_t)0);
    CHECK_EQ(s.observer_calls+s.slot_calls+s.observer_bytes+s.slot_bytes+s.unbound()+s.owner_mismatches, (uint64_t)0);
}
TEST(query_copy_success_records_actual_rank0_bytes_then_binds_the_real_epoch) {
    DraftQueryCopyStats s; REQUIRE(s.configure(40, 128));
    for (int i = 1; i <= 3; ++i) {
        REQUIRE(s.begin(i, i < 3 ? 7 : 8, 0, 0)); CHECK_EQ(s.unbound(), (uint64_t)1);
        CHECK(s.copied(0, DraftQueryCopyStats::Observer, 40)); CHECK(s.copied(0, DraftQueryCopyStats::Slots, 128));
        CHECK(s.bind(i, i < 3 ? 7 : 8, 0, 0, i)); CHECK(!s.pending()); CHECK_EQ(s.unbound(), (uint64_t)0);
    }
    CHECK_EQ(s.observer_calls, (uint64_t)3); CHECK_EQ(s.observer_bytes, (uint64_t)120);
    CHECK_EQ(s.slot_calls, (uint64_t)3); CHECK_EQ(s.slot_bytes, (uint64_t)384);
    CHECK_EQ(s.bound_epochs, (uint64_t)3); CHECK_EQ(s.last_seq, (uint64_t)8);
}
TEST(query_copy_failed_enqueue_does_not_increment_calls_or_bytes) {
    DraftQueryCopyStats s; REQUIRE(s.configure(40, 128)); REQUIRE(s.begin(1, 7, 0, 0));
    CHECK(!s.copied(-1, DraftQueryCopyStats::Observer, 40));
    CHECK_EQ(s.observer_calls, (uint64_t)0); CHECK_EQ(s.observer_bytes, (uint64_t)0);
    CHECK(s.copied(0, DraftQueryCopyStats::Observer, 40));
    CHECK(!s.copied(-1, DraftQueryCopyStats::Slots, 128));
    CHECK_EQ(s.slot_calls, (uint64_t)0); CHECK_EQ(s.slot_bytes, (uint64_t)0);
    CHECK(!s.bind(1, 7, 0, 0, 1)); CHECK_EQ(s.bound_epochs, (uint64_t)0); CHECK_EQ(s.unbound(), (uint64_t)1);
}
TEST(query_copy_orphan_duplicate_and_wrong_width_are_real_successes_but_invalid_receipts) {
    for (int fault = 0; fault < 3; ++fault) {
        DraftQueryCopyStats s; REQUIRE(s.configure(40, 128));
        if (fault) REQUIRE(s.begin(1, 7, 0, 0));
        const bool ok = s.copied(0, DraftQueryCopyStats::Observer, fault == 2 ? 39 : 40);
        CHECK_EQ(ok, fault == 1);
        if (fault == 1) CHECK(!s.copied(0, DraftQueryCopyStats::Observer, 40));
        CHECK_EQ(s.observer_calls, (uint64_t)(fault == 1 ? 2 : 1));
        CHECK(s.owner_mismatches > 0); CHECK(!s.bind(1, 7, 0, 0, 1));
        CHECK_EQ(s.bound_epochs, (uint64_t)0);
    }
}
TEST(query_copy_wrong_owner_or_epoch_never_rebinds_a_successful_pair) {
    for (int fault = 0; fault < 5; ++fault) {
        DraftQueryCopyStats s; REQUIRE(s.configure(40, 128)); REQUIRE(s.begin(1, 7, 2, 0));
        CHECK(s.copied(0, DraftQueryCopyStats::Observer, 40)); CHECK(s.copied(0, DraftQueryCopyStats::Slots, 128));
        CHECK(!s.bind(fault == 0 ? 2 : 1, fault == 1 ? 8 : 7, fault == 2 ? 3 : 2,
                      fault == 3 ? 1 : 0, fault == 4 ? 2 : 1));
        CHECK_EQ(s.bound_epochs, (uint64_t)0); CHECK_EQ(s.unbound(), (uint64_t)1);
        CHECK(!s.begin(2, 8, 2, 0)); // no silent discard/repair of a failed pending frame
    }
}
RAD_TEST_MAIN()
