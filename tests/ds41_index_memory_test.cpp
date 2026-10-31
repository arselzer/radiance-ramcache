#include "rad_test.h"
#include "../arch/deepseek41/ds41_index_memory.h"
#include "../core/place/ds41_index_memory.h"
#include <functional>
using namespace rad;
namespace {
int twin_launch(const RadArgs*, RadStream) { return RAD_OK; }
int index_launch(const RadArgs*, RadStream) { return RAD_OK; }
int alien_launch(const RadArgs*, RadStream) { return RAD_E_UNSUPPORTED; }
const int module_a = 0, module_b = 1;
const void* fake_origin(RadLaunchFn f) { return f ? f == alien_launch ? &module_b : &module_a : nullptr; }
const void* failed_origin(RadLaunchFn) { return nullptr; }
struct Fixture {
    Program p;
    RadKernelInfo info[4]{};
    KernelRow row[4]{};
    Fixture() {
        const char* ops[] = { "moe_gemm_ggml", "index_score", "index_candidates", "index_topk" };
        const char* names[] = { "moe_gemm_ggml_mxfp4_gemv_tall", "ds41_index_score_d128_bf16",
                                "ds41_index_candidates", "ds41_index_topk" };
        for (int i = 0; i < 4; ++i) {
            info[i].name = names[i]; info[i].op = ops[i]; info[i].domain = RAD_DOMAIN_DEVICE;
            info[i].launch = i ? index_launch : twin_launch;
            row[i].info = &info[i]; row[i].plugin = "libr4d"; row[i].plugin_order = 0;
            OpInfo op; op.op = ops[i]; Band b; b.hi = 64;
            b.dom[RAD_DOMAIN_DEVICE].row = &row[i]; op.bands.push_back(b); p.ops.push_back(op);
        }
    }
};
} // namespace

TEST(ds41_index_memory_budget_quotient_and_default_floor_are_preserved) {
    using namespace deepseek41;
    bool requested = true;
    for (const char* off : { (const char*)nullptr, "", "0" }) {
        REQUIRE(index_compact_parse(off, &requested)); CHECK(!requested);
    }
    REQUIRE(index_compact_parse("1", &requested)); CHECK(requested);
    for (const char* bad : { "on", "-1", "2", "1x", " 1" }) CHECK(!index_compact_parse(bad, &requested));
    for (int64_t C : { int64_t(16), int64_t(256), int64_t(2048), int64_t(4096) }) {
        for (int64_t ctx : { int64_t(32768), int64_t(524288), int64_t(1048576) }) {
            const int64_t width = ctx + 1;
            CHECK_EQ(index_plane_rows(C, width, false), std::min(C, std::max(int64_t(256), kScorePlane / (4 * width))));
            if (ctx == 32768) CHECK_EQ(index_plane_rows(C, width, true), index_plane_rows(C, width, false));
        }
    }
    CHECK_EQ(index_plane_rows(2048, 524289, true), int64_t(127));
    CHECK_EQ(index_plane_rows(2048, 1048577, true), int64_t(64));
    CHECK_EQ(index_plane_rows(2048, 1048577, true) * 1048577 * 4, (int64_t(256) << 20) + 256);
    CHECK_EQ(index_plane_rows(2048, 524289, true) * 524289 * 4, (int64_t(254) << 20) + 508);
    CHECK_EQ(index_plane_rows(2048, 0, true), int64_t(0));
}

TEST(ds41_index_memory_actual_module_and_all_known_band_admission) {
    Fixture f; std::string why;
    REQUIRE(ds41_index_memory_admit(f.p, &why, fake_origin)); CHECK(why.empty());
    /* dladdr is exercised on real executable text, without a library/device/model load. */
    REQUIRE(ds41_index_module(twin_launch) != nullptr);
    CHECK_EQ(ds41_index_module(twin_launch), ds41_index_module(index_launch));
    CHECK(ds41_index_memory_admit(f.p, &why));
    f.p.ops[1].op = "index_score_kvpack"; f.info[1].name = "ds41_index_score_d128_kvpack";
    CHECK(ds41_index_memory_admit(f.p, &why, fake_origin));
    f.info[0].name = "moe_gemm_ggml_mxfp4_gemv_tall_p1";
    CHECK(ds41_index_memory_admit(f.p, &why, fake_origin));
    f.p.ops[1].bands.push_back(f.p.ops[1].bands[0]);
    CHECK(ds41_index_memory_admit(f.p, &why, fake_origin));
}

TEST(ds41_index_memory_refuses_unproven_origins_axes_and_missing_bands) {
    const std::vector<std::function<void(Fixture&)>> bad = {
        [](Fixture& f) { f.info[0].launch = nullptr; },
        [](Fixture& f) { f.info[1].launch = nullptr; },
        [](Fixture& f) { f.info[1].launch = alien_launch; },
        [](Fixture& f) { f.row[1].plugin = "override"; },
        [](Fixture& f) { f.row[0].plugin.clear(); },
        [](Fixture& f) { for (auto& row : f.row) row.plugin = "another_library"; },
        [](Fixture& f) { f.row[2].plugin_order = 1; },
        [](Fixture& f) { f.info[3].name = "unknown_topk"; },
        [](Fixture& f) { f.info[2].n_tunables = 1; },
        [](Fixture& f) { f.p.ops[1].bands[0].dom[RAD_DOMAIN_DEVICE].tune_source = "RADIANCE_TUNE"; },
        [](Fixture& f) { f.p.ops[2].bands[0].dom[RAD_DOMAIN_DEVICE].choice = "x=1"; },
        [](Fixture& f) { f.p.ops[3].bands.clear(); },
        [](Fixture& f) { f.p.ops[1].bands[0].dom[RAD_DOMAIN_DEVICE].row = nullptr; },
        [](Fixture& f) { f.info[0].name = "no_lcp_twin"; },
        [](Fixture& f) { f.p.ops[1].op = "no_score_chain"; },
        [](Fixture& f) { f.p.ops[3].op = "no_selection_chain"; },
        [](Fixture& f) { auto b = f.p.ops[1].bands[0]; b.dom[RAD_DOMAIN_DEVICE].row = &f.row[3];
                         f.p.ops[1].bands.push_back(b); },
    };
    for (const auto& mutate : bad) {
        Fixture f; mutate(f); std::string why;
        CHECK(!ds41_index_memory_admit(f.p, &why, fake_origin)); CHECK(!why.empty());
    }
    Fixture f; std::string why;
    CHECK(!ds41_index_memory_admit(f.p, &why, failed_origin)); CHECK(!why.empty());
    CHECK(!ds41_index_memory_admit(f.p, &why, nullptr)); CHECK(!why.empty());
}

RAD_TEST_MAIN()
