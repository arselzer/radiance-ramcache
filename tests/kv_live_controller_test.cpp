/* Actual host Pool/VMem + manager/publisher/controller composition, not a model execution. */
#include "rad_test.h"
#include "mem/kv_live_controller.h"
#include "mem/kv.h"
#include "mem/pools.h"
#include "device/device.h"
#include "rad_internal.h"

#include <memory>

using namespace rad;
namespace {
struct RankData { Program program; Pools pools; KVManager kv; };
struct Fixture {
    std::vector<std::unique_ptr<RankData>> ranks;
    KVLiveController controller; // destroyed before ranks/pools
    int64_t free_bytes = int64_t(1) << 40;
    int64_t G = 0;
    KVLiveOptions options;
    int build() {
        G = VMem::granularity();
        if (G <= 0 || G % kKVLiveMiB) return RAD_E_UNSUPPORTED;
        Config c;
        c.vram_weights_mib = c.gpu_headroom_mib = 1;
        // Both index-layer stripes exceed G; the minimum's first granule alone cannot expose
        // an index ID. A short ensure must therefore exercise both groups' grow callbacks.
        c.vram_kv_mib = 16 * G / kKVLiveMiB;
        c.host_pool_mib = 0; c.max_seqs = 1; c.max_tok = 128; c.max_ctx = 4096; c.n_spec = 0;
        std::vector<KVGroupInfo> groups(1);
        for (uint32_t format : { uint32_t(RAD_KV_STORAGE_DS41_FP4_POW2_G32),
                                 uint32_t(RAD_KV_STORAGE_DS41_FP4_E4M3_G16) }) {
            KVGroupInfo g;
            g.name = format == 2 ? "index" : "latent";
            g.decl.kind = RAD_KV_FULL; g.decl.dtype = RAD_BF16; g.decl.n_head_kv = 1;
            g.decl.head_dim = format == 2 ? 32 : 128;
            g.decl.storage_format = format; g.decl.storage_block_bytes = format == 2 ? 68 : 288;
            g.block_size = 2; g.layers = { 2, 8 };
            groups.push_back(g);
        }
        KVGroupInfo tail;
        tail.name = "raw_f32"; tail.decl.kind = RAD_KV_LINEAR; tail.decl.dtype = RAD_F32;
        tail.decl.n_head_kv = 1; tail.decl.state_dim[0] = 2; tail.decl.state_dim[1] = 16;
        tail.layers = { 2 }; groups.push_back(tail);
        KVGroupInfo win;
        win.name = "window"; win.decl.kind = RAD_KV_WINDOW; win.decl.dtype = RAD_BF16;
        win.decl.n_head_kv = 1; win.decl.head_dim = 8; win.decl.window = 8;
        win.block_size = 4; win.layers = { 2 }; groups.push_back(win);
        std::vector<KVLiveController::Rank> cohort;
        for (int i = 0; i < 4; ++i) {
            auto r = std::make_unique<RankData>();
            r->program.kv_groups = groups;
            RAD_TRY(r->pools.configure(c, 0, &rad_malloc_allocator(), &rad_malloc_allocator()));
            RAD_TRY(r->kv.plan_groups(groups, c));
            RAD_TRY(r->kv.enable_backing_mode());
            RAD_TRY(r->kv.carve(r->pools.kv()));
            if (r->pools.kv().committed() != 0) return RAD_E_STATE;
            cohort.push_back({ 0, &r->pools.kv(), &r->kv, &r->program });
            ranks.push_back(std::move(r));
        }
        options = { true, G, 16 * G, 0, 0, 0 };
        return controller.start(options, cohort, 0, 0, 0,
            [this](int, int64_t* out) { *out = free_bytes; return int(RAD_OK); });
    }
};
bool host_vmem() { return device_is_host() && VMem::supported(); }
}  // namespace

TEST(kv_live_minimum_then_real_all_rank_growth_and_prefix_pinned_shrink) {
    if (!host_vmem()) SKIP("actual controller/VMem composition requires host elastic memory");
    Fixture f;
    REQUIRE_EQ(f.build(), RAD_OK);
    const int64_t initial = f.ranks[0]->pools.kv().committed();
    REQUIRE(initial > 0 && initial < f.options.maximum_bytes);
    std::vector<void*> bases;
    std::vector<int64_t> strides;
    for (const auto& r : f.ranks) {
        CHECK_EQ(r->pools.kv().committed(), initial);
        CHECK_EQ(r->kv.free_blocks(1), 0ll); // min/fixed tail cannot expose an unbacked full page
        bases.push_back(r->pools.kv().base());
        strides.push_back(r->kv.plan(1)->layer_stride());
    }
    REQUIRE_EQ(f.controller.begin_plan(), RAD_OK);
    KVManager& lead = f.ranks[0]->kv;
    REQUIRE_EQ(lead.add_sequence(1), RAD_OK);
    REQUIRE_EQ(lead.ensure(1, 128), RAD_OK); // two FULL groups: later demand must retain earlier candidates
    f.controller.finish_plan();
    const int64_t grown = f.ranks[0]->pools.kv().committed();
    CHECK(grown > initial && grown < f.options.maximum_bytes);
    for (size_t i = 0; i < f.ranks.size(); ++i) {
        const auto& r = f.ranks[i];
        CHECK_EQ(r->pools.kv().committed(), grown);
        CHECK(r->pools.kv().base() == bases[i]);
        CHECK_EQ(r->kv.plan(1)->layer_stride(), strides[i]);
        for (int32_t g : { 1, 2 })
            for (int32_t b : lead.block_table(1, g)) CHECK(r->kv.block_backed(g, b));
    }
    const int32_t retained = lead.block_table(1, 1).front();
    REQUIRE_EQ(lead.retain_block(1, retained), RAD_OK);
    lead.free_sequence(1);
    REQUIRE_EQ(f.controller.begin_plan(), RAD_OK);
    f.controller.finish_plan();
    for (const auto& r : f.ranks) CHECK(r->kv.block_backed(1, retained));
    const int64_t with_prefix = f.ranks[0]->pools.kv().committed();
    CHECK(with_prefix > initial && with_prefix <= grown);
    lead.release_block(1, retained);
    REQUIRE_EQ(f.controller.begin_plan(), RAD_OK);
    f.controller.finish_plan();
    for (size_t i = 0; i < f.ranks.size(); ++i) {
        CHECK_EQ(f.ranks[i]->pools.kv().committed(), initial);
        CHECK(f.ranks[i]->pools.kv().base() == bases[i]);
        CHECK_EQ(f.ranks[i]->kv.plan(1)->layer_stride(), strides[i]);
    }
}

TEST(kv_live_failed_growth_funding_withdraws_every_rank_before_any_candidate_is_allocated) {
    if (!host_vmem()) SKIP("actual controller/VMem composition requires host elastic memory");
    Fixture f;
    REQUIRE_EQ(f.build(), RAD_OK);
    const int64_t initial = f.ranks[0]->pools.kv().committed();
    REQUIRE_EQ(f.controller.begin_plan(), RAD_OK);
    f.free_bytes = 0; // after startup proof; controlled runtime funding refusal
    KVManager& lead = f.ranks[0]->kv;
    REQUIRE_EQ(lead.add_sequence(1), RAD_OK);
    CHECK(lead.ensure(1, 128) < 0);
    CHECK(f.controller.status() < 0);
    f.controller.finish_plan();
    for (const auto& r : f.ranks) {
        CHECK_EQ(r->pools.kv().committed(), initial);
        CHECK(r->kv.backing_error() < 0);
        for (int32_t g : { 1, 2 }) CHECK_EQ(r->kv.free_blocks(g), 0ll);
    }
    CHECK(lead.block_table(1, 1).empty());
    CHECK(lead.block_table(1, 2).empty());
}

TEST(kv_live_allocation_outside_the_preplan_lease_never_maps_and_poison_closes_all_ranks) {
    if (!host_vmem()) SKIP("actual controller/VMem composition requires host elastic memory");
    Fixture f;
    REQUIRE_EQ(f.build(), RAD_OK);
    const int64_t initial = f.ranks[0]->pools.kv().committed();
    KVManager& lead = f.ranks[0]->kv;
    REQUIRE_EQ(lead.add_sequence(1), RAD_OK);
    CHECK(lead.ensure(1, 128) < 0); // no begin_plan
    for (const auto& r : f.ranks) {
        CHECK_EQ(r->pools.kv().committed(), initial);
        CHECK(r->kv.backing_error() < 0);
        CHECK_EQ(r->kv.free_blocks(1), 0ll);
    }
}

RAD_TEST_MAIN()
