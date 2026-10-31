/* Prepared arithmetic/inventory cases only; not a Phase2 runtime or numerical oracle. */
#include "rad_test.h"
#include "place/ds41_phase2_retention.h"
using namespace rad::ds41_phase2;

TEST(phase2_retention_includes_fused_next_norm_six_sorted_slots_and_context_keep) {
    Geometry g;
    Layout p;
    REQUIRE(retention_layout(g, 8192, 2, 4096, &p));
    CHECK_EQ(p.keep_words, int64_t(33));
    CHECK_EQ(p.row_bytes, int64_t(176446));
    CHECK_EQ(p.buffers[size_t(Buffer::NextNormed)].row_bytes, int64_t(10240));
    CHECK_EQ(p.buffers[size_t(Buffer::ExpertDown)].row_bytes, int64_t(61440));
    CHECK_EQ(p.buffers[size_t(Buffer::SharedGate)].row_bytes, int64_t(2));
    CHECK_EQ(p.route_metadata.bytes, int64_t(2*772));
    int64_t end = 0;
    for (const auto& b : p.buffers) {
        CHECK_EQ(b.offset % 4096, int64_t(0));
        CHECK(b.offset >= end); end = b.offset + b.bytes;
    }
    CHECK(p.route_metadata.offset >= end);
    CHECK(p.total_bytes >= p.route_metadata.offset + p.route_metadata.bytes);
    auto long_context = g; long_context.max_ctx = 1048576;
    Layout q; REQUIRE(retention_layout(long_context, 8192, 2, 4096, &q));
    CHECK_EQ(q.keep_words, int64_t(4097));
    CHECK(q.total_bytes > p.total_bytes);
    auto span16 = g; span16.max_ctx = 16384;
    Layout p16; REQUIRE(retention_layout(span16, 16384, 4, 4096, &p16));
    CHECK_EQ(p16.keep_words, int64_t(65));
    CHECK_EQ(p16.row_bytes, int64_t(176574));
    CHECK(!retention_layout(g, 16384, 4, 4096, &p16)); // rows cannot exceed the context
}

TEST(phase2_retention_refuses_unclosed_geometry_span_and_overflow_without_mutating_output) {
    Geometry g; Layout p;
    REQUIRE(retention_layout(g, 6817, 2, 4096, &p));
    const auto old = p.total_bytes;
    for (int64_t rows : {int64_t(0), int64_t(16385), INT64_MAX}) {
        CHECK(!retention_layout(g, rows, 2, 4096, &p)); CHECK_EQ(p.total_bytes, old);
    }
    CHECK(!retention_layout(g, 8192, 1, 4096, &p));
    CHECK(!retention_layout(g, 8192, 2, 3, &p));
    auto bad = g; bad.top_k = 8; CHECK(!retention_layout(bad, 8192, 2, 4096, &p));
    bad = g; bad.max_ctx = INT64_MAX; CHECK(!retention_layout(bad, 8192, 2, 4096, &p));
    bad = g; bad.max_ctx = INT64_MAX-1; CHECK(!retention_layout(bad, 16384, 4, 4096, &p));
    CHECK_EQ(p.total_bytes, old);
}

TEST(phase2_budget_keeps_variable_cache_future_growth_and_refuses_unfunded_arena_assumptions) {
    Layout p; REQUIRE(retention_layout({}, 8192, 2, 4096, &p));
    Budget b;
    b.device_room = p.total_bytes + 512; b.device_headroom = 256; b.additional_device_bytes = 256;
    b.host_headroom = 8192; b.future_expert_cache_bytes = 192; b.additional_host_bytes = 64;
    b.host_available = b.host_headroom + b.future_expert_cache_bytes + b.additional_host_bytes;
    CHECK(fits_budget(p, b));
    --b.device_room; CHECK(!fits_budget(p, b)); ++b.device_room;
    --b.host_available; CHECK(!fits_budget(p, b)); ++b.host_available;
    b.future_expert_cache_bytes = 96; CHECK(fits_budget(p, b));
    b.replaceable_piece_bytes = p.total_bytes + 1; CHECK(!fits_budget(p, b));
    b.replaceable_piece_bytes = 0; b.additional_device_bytes = INT64_MAX; CHECK(!fits_budget(p, b));
    b.additional_device_bytes = 0; b.host_headroom = INT64_MAX; CHECK(!fits_budget(p, b));
}
RAD_TEST_MAIN()
