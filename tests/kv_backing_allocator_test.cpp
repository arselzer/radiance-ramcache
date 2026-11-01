/* Real manager/free-heap adapter against metadata mapping transactions. No device work. */
#include "rad_test.h"
#include "mem/kv.h"
#include "mem/kv_backing_allocator.h"
#include "device/device.h"
#include "rad_internal.h"
#include <algorithm>
using namespace rad;
namespace {
struct Fixture {
    Pools pools;
    KVManager kv;
    KVManagerBackingPublisher pub{&kv};
    KVBackingSnapshot map;
    int grow_calls = 0;
    int build(bool window = true, bool whole_index = false) {
        Config cfg; cfg.vram_weights_mib = cfg.gpu_headroom_mib = 1; cfg.vram_kv_mib = 8;
        cfg.host_pool_mib = 0; cfg.max_seqs = 2; cfg.max_tok = 4; cfg.max_ctx = 131072;
        std::vector<KVGroupInfo> groups(1);
        KVGroupInfo full; full.name = "packed_index"; full.decl.kind = RAD_KV_FULL;
        full.decl.dtype = RAD_BF16; full.decl.n_head_kv = 1; full.decl.head_dim = 32;
        full.decl.storage_format = RAD_KV_STORAGE_DS41_FP4_POW2_G32; full.decl.storage_block_bytes = 68;
        full.block_size = 2; full.layers = {2, 8}; groups.push_back(full);
        if (whole_index) { groups.back().block_size = 1; groups.back().decl.head_dim = 64; }
        full.name = "packed_latent"; full.decl.head_dim = 256; full.block_size = 1; full.layers = {20};
        full.decl.storage_format = RAD_KV_STORAGE_DS41_FP4_E4M3_G16; full.decl.storage_block_bytes = 288;
        groups.push_back(full);
        KVGroupInfo state; state.name = "raw_state"; state.decl.kind = RAD_KV_LINEAR;
        state.decl.dtype = RAD_F32; state.decl.n_head_kv = 1; state.decl.state_dim[0] = 4;
        state.decl.state_dim[1] = 4; state.layers = {2}; groups.push_back(state);
        KVGroupInfo win; win.name = "window"; win.decl.kind = RAD_KV_WINDOW; win.decl.dtype = RAD_BF16;
        win.decl.n_head_kv = 1; win.decl.head_dim = 8; win.decl.window = 8;
        win.block_size = 4; win.layers = {2}; if (window) groups.push_back(win);
        int rc = pools.configure(cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator()); if (rc < 0) return rc;
        rc = kv.plan_groups(groups, cfg); if (rc < 0) return rc;
        rc = kv.enable_backing_mode(); if (rc < 0) return rc;
        rc = kv.carve(pools.kv()); if (rc < 0) return rc;
        map.reserved_bytes = pools.kv().backing_reserved_bytes();
        map.granule_bytes = pools.kv().backing_granule_bytes();
        if (!map.granule_bytes) return RAD_E_UNSUPPORTED;
        map.backed.assign((size_t)(map.reserved_bytes / map.granule_bytes), 0);
        return RAD_OK;
    }
    int publish(const std::vector<KVBackingBlock>& wanted = {}) {
        std::vector<KVBackingRange> required, extra;
        int rc = kv.backing_required_ranges(kv, &required); if (rc < 0) return rc;
        rc = kv.backing_block_ranges(wanted, &extra); if (rc < 0) return rc;
        required.insert(required.end(), extra.begin(), extra.end());
        KVBackingLimits lim{map.reserved_bytes, map.granule_bytes, 0, map.reserved_bytes, 0, 0};
        KVBackingPlan plan;
        if (kv_backing_plan(lim, map.backed, required, &plan) != KVBackingResult::Ok) return RAD_E_STATE;
        auto target = map; target.backed = plan.target;
        rc = pub.stage(map, target); if (rc < 0) return rc;
        pub.withdraw();
        for (int g = 1; g < kv.n_groups(); ++g) CHECK_EQ(kv.free_blocks(g), int64_t(0));
        map = target; // metadata backend; no pool bytes are touched by these cases
        pub.publish_target();
        return kv.backing_error();
    }
    void grow() {
        kv.set_backing_grow([this](const std::vector<KVBackingBlock>& ids) {
            ++grow_calls;
            const int rc = publish(ids);
            return KVBackingGrowResult{rc, rc < 0};
        });
    }
    int32_t unbacked(int g) const {
        for (int32_t b = 0; b < kv.total_blocks(g); ++b) if (!kv.block_backed(g, b)) return b;
        return -1;
    }
};
} // namespace

TEST(kv_backing_allocator_bootstrap_hides_ids_and_requires_fixed_floor) {
    if (!device_is_host() || !VMem::supported()) SKIP("host VMem metadata fixture required");
    Fixture f; REQUIRE_EQ(f.build(), RAD_OK);
    CHECK_EQ(f.pools.kv().committed(), int64_t(0));
    CHECK_EQ(f.kv.free_blocks(1), int64_t(0)); CHECK_EQ(f.kv.free_states(3), int64_t(0));
    CHECK(f.pub.stage(f.map, f.map) < 0); // unmapped raw/window floor cannot be published
    REQUIRE_EQ(f.publish(), RAD_OK); f.grow();
    CHECK(f.kv.free_states(3) > 0);
    CHECK_EQ(f.kv.set_loan(1), RAD_E_UNSUPPORTED);
    REQUIRE_EQ(f.kv.add_sequence(1), RAD_OK);
    REQUIRE_EQ(f.kv.ensure(1, 1), RAD_OK);
    CHECK(f.grow_calls >= 1);
    for (int g : {1, 2, 4}) for (int32_t b : f.kv.block_table(1, g)) CHECK(f.kv.block_backed(g, b));
    CHECK_EQ(f.kv.held_blocks(1), int64_t(1));
    // Granule rounding can back an entire small group while mapping just its requested pages.
    // Count physical publication independently from the map, rather than require a gap in that
    // particular group. Logical availability includes unbacked private candidates; the public
    // free count and backed mask must not advertise them as physically allocatable.
    int64_t hidden_full = 0;
    for (int g : {1, 2, 4}) {
        const auto* p = f.kv.plan(g);
        const int64_t frag = p->layer_bytes_per_block();
        int64_t backed_free = 0, logical_free = 0, unbacked_free = 0;
        for (int32_t b = 0; b < p->n_blocks; ++b) {
            bool mapped = true;
            for (int64_t l = 0; l < p->n_layers; ++l) {
                const int64_t off = p->pool_offset + l * p->layer_stride() + b * frag;
                const int64_t first = off / f.map.granule_bytes;
                const int64_t last = (off + frag - 1) / f.map.granule_bytes;
                for (int64_t i = first; i <= last; ++i)
                    mapped &= f.map.backed[(size_t)i] != 0;
            }
            CHECK_EQ(f.kv.block_backed(g, b), mapped);
            if (f.kv.block_refcount(g, b) == 0) {
                ++logical_free;
                if (mapped) ++backed_free;
                else ++unbacked_free;
            }
        }
        CHECK_EQ(f.kv.free_blocks(g), backed_free);
        CHECK_EQ(f.kv.available_blocks(g), logical_free);
        CHECK_EQ(logical_free - backed_free, unbacked_free);
        if (p->kind == RAD_KV_FULL && unbacked_free > 0) {
            CHECK(f.kv.free_blocks(g) < f.kv.available_blocks(g));
            hidden_full += unbacked_free;
        }
    }
    CHECK(hidden_full > 0); // actual future FULL capacity remains unpublished
}

TEST(kv_backing_allocator_cow_prefix_and_inflight_demand_survive_later_group_transactions) {
    if (!device_is_host() || !VMem::supported()) SKIP("host VMem metadata fixture required");
    Fixture f; REQUIRE_EQ(f.build(), RAD_OK); REQUIRE_EQ(f.publish(), RAD_OK); f.grow();
    REQUIRE_EQ(f.kv.add_sequence(1), RAD_OK); REQUIRE_EQ(f.kv.ensure(1, 1), RAD_OK);
    const int32_t source = f.kv.block_table(1, 1)[0];
    REQUIRE_EQ(f.kv.retain_block(1, source), RAD_OK);
    REQUIRE_EQ(f.kv.fork(1, 2), RAD_OK); REQUIRE_EQ(f.kv.ensure(2, 2), RAD_OK);
    const int32_t copy = f.kv.block_table(2, 1)[0]; CHECK(copy != source);
    f.kv.free_sequence(1); f.kv.free_sequence(2);
    REQUIRE_EQ(f.publish(), RAD_OK); // pending COW source/destination remain pinned
    CHECK(f.kv.block_backed(1, source)); CHECK(f.kv.block_backed(1, copy));
    std::vector<KVBackingRange> fragments;
    REQUIRE_EQ(f.kv.backing_block_ranges({{1, source}}, &fragments), RAD_OK);
    REQUIRE_EQ(fragments.size(), size_t(2));
    auto invalid = f.map;
    invalid.backed[(size_t)(fragments[0].offset / invalid.granule_bytes)] = 0;
    CHECK(f.pub.stage(f.map, invalid) < 0); // retained prefix / pending copy cannot be unmapped
    (void)f.kv.take_pending_copies(); // stand-in for the engine's checked drain
    f.kv.release_block(1, source);
    REQUIRE_EQ(f.publish(), RAD_OK);
    CHECK_EQ(f.kv.held_blocks(1), int64_t(0));
}

TEST(kv_backing_allocator_multi_group_preflight_pins_unowned_candidates_until_commit) {
    if (!device_is_host() || !VMem::supported()) SKIP("host VMem metadata fixture required");
    Fixture f; REQUIRE_EQ(f.build(false, true), RAD_OK); REQUIRE_EQ(f.publish(), RAD_OK); f.grow();
    /* The fixed raw state backs the last granule, giving the second full group some free IDs.
     * The first group's new candidates live in earlier granules that a later group's growth
     * would otherwise be allowed to trim before either group's first refcount is installed. */
    const int64_t tokens = f.kv.free_blocks(2) + 1;
    REQUIRE(tokens > 1); REQUIRE(tokens < f.kv.total_blocks(1));
    REQUIRE_EQ(f.kv.add_sequence(1), RAD_OK); REQUIRE_EQ(f.kv.ensure(1, tokens), RAD_OK);
    CHECK(f.grow_calls >= 2);
    for (int g : {1, 2}) {
        CHECK_EQ(f.kv.held_blocks(g), tokens);
        for (int32_t b : f.kv.block_table(1, g)) CHECK(f.kv.block_backed(g, b));
    }
}

TEST(kv_backing_allocator_retained_prefix_adopts_only_backed_whole_blocks) {
    if (!device_is_host() || !VMem::supported()) SKIP("host VMem metadata fixture required");
    Fixture f; REQUIRE_EQ(f.build(), RAD_OK); REQUIRE_EQ(f.publish(), RAD_OK); f.grow();
    REQUIRE_EQ(f.kv.add_sequence(1), RAD_OK); REQUIRE_EQ(f.kv.ensure(1, 2), RAD_OK);
    const int32_t prefix = f.kv.block_table(1, 1)[0];
    REQUIRE_EQ(f.kv.retain_block(1, prefix), RAD_OK); f.kv.free_sequence(1);
    REQUIRE_EQ(f.publish(), RAD_OK);
    REQUIRE_EQ(f.kv.add_sequence(2), RAD_OK);
    REQUIRE_EQ(f.kv.adopt(2, 1, {prefix}, 2), RAD_OK);
    CHECK_EQ(f.kv.block_refcount(1, prefix), 2);
    REQUIRE_EQ(f.kv.ensure(2, 3), RAD_OK);
    CHECK_EQ(f.kv.block_table(2, 1)[0], prefix);
    for (int32_t b : f.kv.block_table(2, 1)) CHECK(f.kv.block_backed(1, b));
    f.kv.free_sequence(2); CHECK_EQ(f.kv.block_refcount(1, prefix), 1);
    f.kv.release_block(1, prefix); REQUIRE_EQ(f.publish(), RAD_OK);
    CHECK_EQ(f.kv.held_blocks(1), int64_t(0));
}

TEST(kv_backing_allocator_specific_restore_budget_refusal_and_fatal_poison_are_distinct) {
    if (!device_is_host() || !VMem::supported()) SKIP("host VMem metadata fixture required");
    Fixture f; REQUIRE_EQ(f.build(), RAD_OK); REQUIRE_EQ(f.publish(), RAD_OK);
    const int32_t b = f.unbacked(1); REQUIRE(b >= 0);
    f.kv.set_backing_grow([](const std::vector<KVBackingBlock>&) { return KVBackingGrowResult{RAD_E_FULL, false}; });
    CHECK_EQ(f.kv.reserve_specific_block(1, b), RAD_E_FULL);
    CHECK_EQ(f.kv.backing_error(), RAD_OK); CHECK_EQ(f.kv.block_refcount(1, b), 0);
    f.grow(); REQUIRE_EQ(f.kv.reserve_specific_block(1, b), RAD_OK);
    CHECK(f.kv.block_backed(1, b)); CHECK_EQ(f.kv.block_refcount(1, b), 1);
    f.kv.release_block(1, b); REQUIRE_EQ(f.publish(), RAD_OK);
    const int32_t missing = f.unbacked(1); REQUIRE(missing >= 0);
    f.kv.set_backing_grow([](const std::vector<KVBackingBlock>&) { return KVBackingGrowResult{RAD_E_DEVICE, true}; });
    CHECK_EQ(f.kv.reserve_specific_block(1, missing), RAD_E_DEVICE);
    CHECK_EQ(f.kv.backing_error(), RAD_E_DEVICE); CHECK_EQ(f.kv.free_blocks(1), int64_t(0));
    int32_t out = 99; CHECK_EQ(f.kv.reserve_block(1, &out), RAD_E_DEVICE); CHECK_EQ(out, -1);
    f.pub.poison(); f.pub.poison(); CHECK_EQ(f.kv.free_blocks(2), int64_t(0));
}

TEST(kv_backing_allocator_old_publication_and_follow_require_common_backed_ids) {
    if (!device_is_host() || !VMem::supported()) SKIP("host VMem metadata fixture required");
    Fixture a, b; REQUIRE_EQ(a.build(), RAD_OK); REQUIRE_EQ(b.build(), RAD_OK);
    REQUIRE_EQ(a.publish(), RAD_OK); REQUIRE_EQ(b.publish(), RAD_OK);
    CHECK_EQ(b.kv.follow(a.kv), RAD_OK);
    const int32_t id = a.unbacked(1); REQUIRE(id >= 0);
    std::vector<KVBackingRange> ranges; REQUIRE_EQ(a.kv.backing_block_ranges({{1, id}}, &ranges), RAD_OK);
    auto target = a.map;
    for (const auto& r : ranges)
        for (int64_t i = r.offset / target.granule_bytes; i <= (r.offset + r.bytes - 1) / target.granule_bytes; ++i)
            target.backed[(size_t)i] = 1;
    const auto old_free = a.kv.free_blocks(1);
    REQUIRE_EQ(a.pub.stage(a.map, target), RAD_OK); a.pub.withdraw(); a.pub.publish_old();
    CHECK_EQ(a.kv.free_blocks(1), old_free); CHECK_EQ(b.kv.follow(a.kv), RAD_OK);
    a.grow(); REQUIRE_EQ(a.kv.reserve_specific_block(1, id), RAD_OK);
    CHECK_EQ(b.kv.follow(a.kv), RAD_E_STATE); // leader IDs are not published on the follower yet
}

TEST(kv_backing_allocator_page_crossing_requires_every_granule_before_specific_restore) {
    if (!device_is_host() || !VMem::supported()) SKIP("host VMem metadata fixture required");
    Fixture f; REQUIRE_EQ(f.build(), RAD_OK); REQUIRE_EQ(f.publish(), RAD_OK); f.grow();
    int32_t crossing = -1; std::vector<KVBackingRange> fragments;
    for (int32_t b = 0; b < f.kv.total_blocks(2); ++b) {
        REQUIRE_EQ(f.kv.backing_block_ranges({{2, b}}, &fragments), RAD_OK);
        const auto r = fragments[0];
        if (r.offset / f.map.granule_bytes != (r.offset + r.bytes - 1) / f.map.granule_bytes) {
            crossing = b; break;
        }
    }
    REQUIRE(crossing >= 0);
    REQUIRE_EQ(f.kv.reserve_specific_block(2, crossing), RAD_OK);
    CHECK(f.kv.block_backed(2, crossing));
    auto partial = f.map;
    partial.backed[(size_t)(fragments[0].offset / partial.granule_bytes)] = 0;
    CHECK(f.pub.stage(f.map, partial) < 0);
    CHECK_EQ(f.kv.block_refcount(2, crossing), 1);
}

TEST(kv_backing_allocator_real_publishers_rollback_all_rank_heaps_before_fatal_stop) {
    if (!device_is_host() || !VMem::supported()) SKIP("host VMem metadata fixture required");
    Fixture a, b; REQUIRE_EQ(a.build(), RAD_OK); REQUIRE_EQ(b.build(), RAD_OK);
    REQUIRE_EQ(a.publish(), RAD_OK); REQUIRE_EQ(b.publish(), RAD_OK); a.grow(); b.grow();
    const int32_t prefix = a.unbacked(1); REQUIRE(prefix >= 0);
    REQUIRE_EQ(a.kv.reserve_specific_block(1, prefix), RAD_OK); a.kv.cache_reserved_block(1, prefix);
    REQUIRE_EQ(b.kv.reserve_specific_block(1, prefix), RAD_OK); b.kv.release_block(1, prefix);
    REQUIRE(a.map.backed == b.map.backed);
    const int32_t wanted = a.unbacked(2); REQUIRE(wanted >= 0);
    struct Quiet final : KVBackingQuiescence {
        bool held = false;
        int enter() override { held = true; return RAD_OK; }
        void leave() noexcept override { held = false; }
        void poison() noexcept override {}
    } quiet;
    struct Funding final : KVBackingFunding {
        int reserve(const std::vector<KVBackingSnapshot>&, const std::vector<KVBackingPlan>&) override { return RAD_OK; }
        void committed() noexcept override {} void rolled_back() noexcept override {} void poison() noexcept override {}
    } funding;
    struct Backend final : KVBackingBackend {
        Fixture* f; Fixture* peer; Quiet* q; bool fail = false;
        int read(KVBackingSnapshot* out) override { *out = f->map; return RAD_OK; }
        void before() {
            CHECK(q->held);
            for (int g : {1, 2}) { CHECK_EQ(f->kv.free_blocks(g), int64_t(0)); CHECK_EQ(peer->kv.free_blocks(g), int64_t(0)); }
        }
        int commit(int64_t off, int64_t) override {
            before(); f->map.backed[(size_t)(off / f->map.granule_bytes)] = 1;
            if (fail) { fail = false; return RAD_E_DEVICE; } // physical presence is not successful access
            return RAD_OK;
        }
        int release(int64_t off, int64_t) override {
            before(); f->map.backed[(size_t)(off / f->map.granule_bytes)] = 0; return RAD_OK;
        }
    } ba, bb;
    ba.f = &a; ba.peer = &b; ba.q = &quiet;
    bb.f = &b; bb.peer = &a; bb.q = &quiet; bb.fail = true;
    auto participant = [&](Fixture& f, Backend* backend) {
        KVBackingParticipant p; p.backend = backend; p.publisher = &f.pub;
        p.limits = {f.map.reserved_bytes, f.map.granule_bytes, 0, f.map.reserved_bytes, 0, 0};
        p.inventory = [&f, &a, wanted](std::vector<KVBackingRange>* out) {
            int rc = f.kv.backing_required_ranges(a.kv, out); if (rc < 0) return rc;
            std::vector<KVBackingRange> add;
            rc = f.kv.backing_block_ranges({{2, wanted}}, &add); if (rc < 0) return rc;
            out->insert(out->end(), add.begin(), add.end()); return int(RAD_OK);
        };
        return p;
    };
    const auto old = a.map.backed;
    const int64_t free_a = a.kv.free_blocks(1), free_b = b.kv.free_blocks(1);
    const auto report = kv_backing_transact({participant(a, &ba), participant(b, &bb)}, &quiet, &funding);
    CHECK(report.result == KVBackingTransactionResult::RolledBack); CHECK_EQ(report.status, RAD_E_DEVICE);
    CHECK(a.map.backed == old && b.map.backed == old);
    CHECK_EQ(a.kv.free_blocks(1), free_a); CHECK_EQ(b.kv.free_blocks(1), free_b);
    CHECK_EQ(a.kv.block_refcount(1, prefix), 1); CHECK_EQ(b.kv.follow(a.kv), RAD_OK);
    a.pub.poison(); b.pub.poison(); // controller's fatal mapping result stops EVERY rank
    CHECK_EQ(a.kv.free_blocks(1), int64_t(0)); CHECK_EQ(b.kv.free_blocks(1), int64_t(0));
}

RAD_TEST_MAIN()
