#include "rad_test.h"
#include "mem/kv_backing_transaction.h"
#include "mem/kv.h"
#include "mem/pools.h"
#include "device/device.h"
#include "rad_internal.h"

#include <algorithm>
#include <cstring>

using namespace rad;
namespace {
constexpr int64_t G = 64;

struct Quiet final : KVBackingQuiescence {
    bool held = false, fatal = false, refuse = false;
    int leaves = 0;
    int enter() override { if (refuse) return RAD_E_STATE; held = true; return RAD_OK; }
    void leave() noexcept override { CHECK(held); held = false; ++leaves; }
    void poison() noexcept override { CHECK(held); fatal = true; }
};
struct Funding final : KVBackingFunding {
    Quiet* quiet;
    bool reserved = false, fatal = false, refuse = false;
    int accepted = 0, restored = 0;
    explicit Funding(Quiet* q) : quiet(q) {}
    int reserve(const std::vector<KVBackingSnapshot>& old,
                const std::vector<KVBackingPlan>& plans) override {
        CHECK(quiet->held);
        CHECK_EQ(old.size(), plans.size());
        if (refuse) return RAD_E_NOMEM;
        reserved = true;
        return RAD_OK;
    }
    void committed() noexcept override { CHECK(reserved); reserved = false; ++accepted; }
    void rolled_back() noexcept override { CHECK(reserved); reserved = false; ++restored; }
    void poison() noexcept override { CHECK(reserved); fatal = true; }
};
struct Publisher;
struct Backend final : KVBackingBackend {
    Quiet* quiet;
    Funding* funding;
    const std::vector<Publisher*>* publishers = nullptr;
    std::vector<uint8_t> physical, accessible, held;
    int commits = 0, releases = 0, changes = 0, peak = 0;
    int fail_commit = -1, fail_release = -1;
    bool map_on_failure = false, unreadable_after_change = false;
    explicit Backend(Quiet* q, Funding* f, std::vector<uint8_t> bits)
        : quiet(q), funding(f), physical(std::move(bits)), accessible(physical), held(physical.size(), 0) {
        peak = (int)std::count(physical.begin(), physical.end(), uint8_t(1));
    }
    int read(KVBackingSnapshot* out) override {
        if (unreadable_after_change && changes > 0) return RAD_E_DEVICE;
        out->reserved_bytes = (int64_t)physical.size() * G;
        out->granule_bytes = G;
        out->backed = physical;
        return RAD_OK;
    }
    void before();
    int commit(int64_t off, int64_t bytes) override {
        before();
        CHECK_EQ(bytes, G);
        const size_t b = (size_t)(off / G);
        const bool fail = ++commits == fail_commit;
        if (!fail || map_on_failure) {
            physical[b] = 1;
            accessible[b] = fail ? 0 : 1;
            ++changes;
            peak = std::max(peak, (int)std::count(physical.begin(), physical.end(), uint8_t(1)));
        }
        return fail ? RAD_E_DEVICE : RAD_OK;
    }
    int release(int64_t off, int64_t bytes) override {
        before();
        CHECK_EQ(bytes, G);
        const size_t b = (size_t)(off / G);
        CHECK(!held[b]);
        if (++releases == fail_release) return RAD_E_DEVICE;
        physical[b] = accessible[b] = 0;
        ++changes;
        return RAD_OK;
    }
};
struct Publisher final : KVBackingPublisher {
    Quiet* quiet;
    KVBackingSnapshot old, target;
    std::vector<uint8_t> publication;
    bool frozen = false, fatal = false, refuse = false, staged = false;
    int target_publications = 0, old_publications = 0;
    explicit Publisher(Quiet* q, std::vector<uint8_t> published) : quiet(q), publication(std::move(published)) {}
    int stage(const KVBackingSnapshot& a, const KVBackingSnapshot& b) override {
        CHECK(quiet->held);
        if (refuse) return RAD_E_INVAL;
        old = a; target = b; staged = true;
        return RAD_OK;
    }
    void discard() noexcept override { CHECK(!frozen); staged = false; }
    void withdraw() noexcept override { CHECK(staged); CHECK(quiet->held); frozen = true; publication.clear(); }
    void publish_target() noexcept override {
        CHECK(quiet->held); CHECK(frozen);
        publication.swap(target.backed); frozen = false; staged = false; ++target_publications;
    }
    void publish_old() noexcept override {
        CHECK(quiet->held); CHECK(frozen);
        publication.swap(old.backed); frozen = false; staged = false; ++old_publications;
    }
    void poison() noexcept override { fatal = true; frozen = true; publication.clear(); }
};
void Backend::before() {
    CHECK(quiet->held);
    CHECK(funding->reserved);
    REQUIRE(publishers);
    for (const Publisher* p : *publishers) { CHECK(p->frozen); CHECK(p->publication.empty()); }
}

KVBackingParticipant participant(KVBackingBackend* b, Publisher* p,
                                 std::vector<KVBackingRange> required, int64_t bytes = 8 * G,
                                 int64_t max_bytes = 8 * G) {
    KVBackingParticipant r;
    r.backend = b;
    r.publisher = p;
    r.limits = { bytes, G, 0, max_bytes, 0, 0 };
    r.inventory = [p, required = std::move(required)](std::vector<KVBackingRange>* out) {
        CHECK(p->quiet->held); // inventory is captured after the ownership lock/drain
        *out = required;
        return int(RAD_OK);
    };
    return r;
}
}  // namespace

TEST(kv_backing_all_ranks_withdraw_before_changes_and_publish_only_after_commit) {
    Quiet q;
    Funding f(&q);
    const std::vector<uint8_t> old = { 1, 1, 1, 1, 0, 0, 0, 0 };
    Backend a(&q, &f, old), b(&q, &f, old);
    Publisher pa(&q, old), pb(&q, old);
    const std::vector<Publisher*> pubs = { &pa, &pb };
    a.publishers = b.publishers = &pubs;
    a.held[0] = b.held[0] = 1; // retained prefix on every rank
    const std::vector<KVBackingRange> required = { { 0, 1 }, { 5 * G - 12, 68 } }; // tight index page crosses granules
    const auto report = kv_backing_transact({ participant(&a, &pa, required, 8 * G, 4 * G),
                                              participant(&b, &pb, required, 8 * G, 4 * G) }, &q, &f);
    CHECK(report.result == KVBackingTransactionResult::Committed);
    CHECK_EQ(report.status, RAD_OK);
    CHECK_EQ(q.leaves, 1);
    CHECK(!q.held && !q.fatal);
    CHECK_EQ(f.accepted, 1);
    CHECK_EQ(pa.target_publications, 1);
    CHECK_EQ(pb.target_publications, 1);
    CHECK(a.physical == report.plans[0].target && b.physical == report.plans[1].target);
    CHECK(pa.publication == a.physical && pb.publication == b.physical);
    CHECK(a.accessible == a.physical && b.accessible == b.physical);
    CHECK(a.peak <= 4 && b.peak <= 4); // early release is funded before additions
    CHECK_EQ(a.physical[0], uint8_t(1));
    CHECK_EQ(a.physical[4], uint8_t(1));
    CHECK_EQ(a.physical[5], uint8_t(1));
}

TEST(kv_backing_failed_later_rank_commit_rolls_back_every_rank_before_old_publication) {
    for (bool inaccessible_map : { false, true }) {
        Quiet q;
        Funding f(&q);
        const std::vector<uint8_t> old = { 1, 1, 1, 1, 0, 0, 0, 0 };
        Backend a(&q, &f, old), b(&q, &f, old);
        Publisher pa(&q, old), pb(&q, old);
        const std::vector<Publisher*> pubs = { &pa, &pb };
        a.publishers = b.publishers = &pubs;
        a.held[0] = b.held[0] = 1;
        b.fail_commit = 1;
        b.map_on_failure = inaccessible_map; // physical present does NOT mean access granted
        const std::vector<KVBackingRange> required = { { 0, 1 }, { 6 * G, G } };
        const auto report = kv_backing_transact({ participant(&a, &pa, required, 8 * G, 4 * G),
                                                  participant(&b, &pb, required, 8 * G, 4 * G) }, &q, &f);
        CHECK(report.result == KVBackingTransactionResult::RolledBack);
        CHECK_EQ(report.status, RAD_E_DEVICE);
        CHECK_EQ(f.restored, 1);
        CHECK_EQ(q.leaves, 1);
        CHECK(!q.fatal);
        CHECK(a.physical == old && b.physical == old);
        CHECK(a.accessible == old && b.accessible == old);
        CHECK(pa.publication == old && pb.publication == old);
        CHECK_EQ(pa.target_publications + pb.target_publications, 0);
        CHECK_EQ(pa.old_publications + pb.old_publications, 2);
    }
}

TEST(kv_backing_failed_rollback_unmap_keeps_all_publishers_frozen_and_stops_before_restore) {
    Quiet q;
    Funding f(&q);
    const std::vector<uint8_t> old = { 1, 1, 1, 1, 0, 0, 0, 0 };
    Backend a(&q, &f, old), b(&q, &f, old);
    Publisher pa(&q, old), pb(&q, old);
    const std::vector<Publisher*> pubs = { &pa, &pb };
    a.publishers = b.publishers = &pubs;
    b.fail_commit = 1;
    b.map_on_failure = true;
    // One early release per rank, then rollback unmap of a's new extent fails.
    a.fail_release = 2;
    const auto report = kv_backing_transact({ participant(&a, &pa, { { 0, G }, { 6 * G, G } }, 8 * G, 4 * G),
                                              participant(&b, &pb, { { 0, G }, { 6 * G, G } }, 8 * G, 4 * G) }, &q, &f);
    CHECK(report.result == KVBackingTransactionResult::Poisoned);
    CHECK(q.fatal && q.held && f.fatal);
    CHECK_EQ(q.leaves, 0);
    CHECK(pa.fatal && pb.fatal && pa.frozen && pb.frozen);
    CHECK(pa.publication.empty() && pb.publication.empty());
    CHECK_EQ(pa.old_publications + pb.old_publications, 0);
    CHECK_EQ(a.commits, 1); // no old-backing restore can exceed peak beside the retained new map
    CHECK(report.actual_valid[0]);
    CHECK(report.actual[0].backed == a.physical);
}

TEST(kv_backing_read_failure_after_mutation_is_poisoned_not_guessed_from_old_map) {
    Quiet q;
    Funding f(&q);
    const std::vector<uint8_t> old(8, 0);
    Backend b(&q, &f, old);
    Publisher p(&q, old);
    const std::vector<Publisher*> pubs = { &p };
    b.publishers = &pubs;
    b.unreadable_after_change = true;
    const auto report = kv_backing_transact({ participant(&b, &p, { { G, G } }) }, &q, &f);
    CHECK(report.result == KVBackingTransactionResult::Poisoned);
    CHECK(q.fatal && p.fatal && f.fatal);
    CHECK(!report.actual_valid[0]);
    CHECK_EQ(p.target_publications + p.old_publications, 0);
}

TEST(kv_backing_failed_release_restores_old_map_but_failed_access_on_restore_poisoned) {
    for (bool fail_restore : { false, true }) {
        Quiet q;
        Funding f(&q);
        const std::vector<uint8_t> old = { 1, 1, 1, 1, 0, 0, 0, 0 };
        Backend b(&q, &f, old);
        Publisher p(&q, old);
        const std::vector<Publisher*> pubs = { &p };
        b.publishers = &pubs;
        b.held[0] = 1;
        b.fail_release = 2; // one old unheld granule released, then a later unmap fails
        if (fail_restore) { b.fail_commit = 1; b.map_on_failure = true; }
        const auto report = kv_backing_transact({ participant(&b, &p, { { 0, G } }) }, &q, &f);
        CHECK(b.physical == old); // presence alone is deliberately insufficient in the failure case
        if (!fail_restore) {
            CHECK(report.result == KVBackingTransactionResult::RolledBack);
            CHECK(b.accessible == old && p.publication == old);
            CHECK_EQ(f.restored, 1);
        } else {
            CHECK(report.result == KVBackingTransactionResult::Poisoned);
            CHECK(b.accessible != old);
            CHECK(p.publication.empty() && q.fatal && f.fatal);
            CHECK_EQ(q.leaves, 0);
        }
    }
}

TEST(kv_backing_refusals_do_not_mutate_maps_or_visible_publication) {
    for (int refusal = 0; refusal < 5; ++refusal) {
        Quiet q;
        Funding f(&q);
        const std::vector<uint8_t> old(8, 1);
        Backend b(&q, &f, old);
        Publisher p(&q, old);
        const std::vector<Publisher*> pubs = { &p };
        b.publishers = &pubs;
        auto r = participant(&b, &p, { { 0, G } });
        if (refusal == 0) q.refuse = true;
        if (refusal == 1) r.limits.granule_bytes = G / 2; // actual backend granule is authoritative
        if (refusal == 2) p.refuse = true;
        if (refusal == 3) f.refuse = true;
        if (refusal == 4) r.inventory = [](std::vector<KVBackingRange>*) { return int(RAD_E_STATE); };
        const auto report = kv_backing_transact({ r }, &q, &f);
        CHECK(report.result == KVBackingTransactionResult::Refused);
        CHECK(report.status < 0);
        CHECK(b.physical == old && p.publication == old);
        CHECK_EQ(b.changes, 0);
        CHECK(!p.frozen && !p.fatal && !q.held && !q.fatal && !f.reserved);
    }
}

TEST(kv_backing_partial_logical_tail_charges_the_full_granule_and_rolls_back_failed_tail_map) {
    for (int mode = 0; mode < 3; ++mode) {
        Quiet q;
        Funding f(&q);
        const std::vector<uint8_t> old(8, 0);
        Backend b(&q, &f, old);
        Publisher p(&q, old);
        const std::vector<Publisher*> pubs = { &p };
        b.publishers = &pubs;
        const int64_t logical_bytes = 7 * G + 1; // the VA reservation is8G, not this449-byte budget
        const int64_t maximum_physical = mode == 0 ? logical_bytes : 8 * G;
        auto r = participant(&b, &p, { { 0, logical_bytes } }, 8 * G, maximum_physical);
        if (mode == 2) { b.fail_commit = 8; b.map_on_failure = true; }
        const auto report = kv_backing_transact({ r }, &q, &f);
        if (mode == 0) {
            CHECK(report.result == KVBackingTransactionResult::Refused);
            CHECK_EQ(report.status, RAD_E_NOMEM);
            CHECK_EQ(b.changes, 0);
        } else if (mode == 1) {
            CHECK(report.result == KVBackingTransactionResult::Committed);
            CHECK_EQ(report.plans[0].required_bytes, 8 * G);
            CHECK_EQ(report.plans[0].target_bytes, 8 * G);
            CHECK_EQ(b.peak, 8);
        } else {
            CHECK(report.result == KVBackingTransactionResult::RolledBack);
            CHECK(b.physical == old && b.accessible == old && p.publication == old);
            CHECK_EQ(b.releases, 8); // tail's failed accessible-map is rolled back too
        }
    }
}

TEST(kv_backing_pool_adapter_maps_and_releases_real_host_vmem_without_moving_addresses) {
    if (!device_is_host()) SKIP("real Pool/VMem backing case requires the host device backend");
    if (!VMem::supported()) SKIP("host backend has no elastic memory");
    Config cfg;
    cfg.vram_weights_mib = cfg.vram_kv_mib = cfg.gpu_headroom_mib = 1;
    cfg.host_pool_mib = 0;
    Pools pools;
    REQUIRE_EQ(pools.configure(cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator()), RAD_OK);
    Pool& pool = pools.kv();
    REQUIRE(pool.elastic());
    void* const base = pool.base();
    PoolKVBackingBackend backend(&pool, 0);
    KVBackingSnapshot initial;
    REQUIRE_EQ(backend.read(&initial), RAD_OK);
    REQUIRE_EQ(pool.committed(), 0ll);
    Quiet q;
    Funding f(&q);
    Publisher p(&q, initial.backed);
    KVBackingParticipant r;
    r.backend = &backend; r.publisher = &p;
    r.limits = { initial.reserved_bytes, initial.granule_bytes, 0, initial.reserved_bytes, 0, 0 };
    r.inventory = [&](std::vector<KVBackingRange>* out) {
        CHECK(q.held);
        *out = { { 0, 1 }, { pool.bytes() - 1, 1 } };
        return int(RAD_OK);
    };
    auto report = kv_backing_transact({ r }, &q, &f);
    REQUIRE(report.result == KVBackingTransactionResult::Committed);
    const int64_t touched = (pool.bytes() - 1) / initial.granule_bytes > 0 ? 2 : 1;
    CHECK_EQ(pool.committed(), touched * initial.granule_bytes);
    CHECK(pool.base() == base);
    CHECK(pool.is_backed(0, 1) && pool.is_backed(pool.bytes() - 1, 1));
    r.inventory = [](std::vector<KVBackingRange>* out) { out->clear(); return int(RAD_OK); };
    report = kv_backing_transact({ r }, &q, &f);
    CHECK(report.result == KVBackingTransactionResult::Committed);
    CHECK_EQ(pool.committed(), 0ll);
    CHECK(pool.base() == base);
}

TEST(kv_backing_manager_inventory_pins_prefix_refs_pending_cow_and_fixed_state_window) {
    if (!device_is_host()) SKIP("manager inventory fixture requires the host device backend");
    Config cfg;
    cfg.vram_weights_mib = cfg.vram_kv_mib = cfg.gpu_headroom_mib = 1;
    cfg.host_pool_mib = 0;
    cfg.max_seqs = 2; cfg.max_tok = 4; cfg.max_ctx = 128;
    std::vector<KVGroupInfo> groups(1);
    KVGroupInfo full;
    full.name = "packed68"; full.decl.kind = RAD_KV_FULL; full.decl.dtype = RAD_BF16;
    full.decl.n_head_kv = 1; full.decl.head_dim = 32;
    full.decl.storage_format = RAD_KV_STORAGE_DS41_FP4_POW2_G32;
    full.decl.storage_block_bytes = 68;
    full.block_size = 2; full.layers = { 2, 8 };
    groups.push_back(full);
    KVGroupInfo tail;
    tail.name = "raw_f32"; tail.decl.kind = RAD_KV_LINEAR; tail.decl.dtype = RAD_F32;
    tail.decl.n_head_kv = 1; tail.decl.state_dim[0] = 2; tail.decl.state_dim[1] = 16;
    tail.layers = { 2 };
    groups.push_back(tail);
    KVGroupInfo win;
    win.name = "window"; win.decl.kind = RAD_KV_WINDOW; win.decl.dtype = RAD_BF16;
    win.decl.n_head_kv = 1; win.decl.head_dim = 8; win.decl.window = 8;
    win.block_size = 4; win.layers = { 2 };
    groups.push_back(win);
    Pools pools, follower_pools;
    REQUIRE_EQ(pools.configure(cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator()), RAD_OK);
    REQUIRE_EQ(follower_pools.configure(cfg, 0, &rad_malloc_allocator(), &rad_malloc_allocator()), RAD_OK);
    KVManager kv, follower;
    REQUIRE_EQ(kv.configure(groups, cfg, pools.kv()), RAD_OK);
    REQUIRE_EQ(follower.configure(groups, cfg, follower_pools.kv()), RAD_OK);
    REQUIRE_EQ(kv.add_sequence(1), RAD_OK);
    REQUIRE_EQ(kv.ensure(1, 3), RAD_OK);
    const auto original = kv.block_table(1, 1);
    REQUIRE_EQ(original.size(), 2u);
    REQUIRE_EQ(kv.retain_block(1, original[0]), RAD_OK); // prefix owner survives sequence cancellation
    REQUIRE_EQ(kv.fork(1, 2), RAD_OK);
    REQUIRE_EQ(kv.ensure(2, 4), RAD_OK); // queued COW source/dest
    kv.free_sequence(1);
    kv.free_sequence(2);
    std::vector<KVBackingRange> required, on_follower;
    REQUIRE_EQ(kv.backing_required_ranges(kv, &required), RAD_OK);
    REQUIRE_EQ(follower.backing_required_ranges(kv, &on_follower), RAD_OK);
    const auto covers = [](const std::vector<KVBackingRange>& ranges, int64_t off, int64_t bytes) {
        for (const auto& r : ranges) if (off >= r.offset && bytes <= r.bytes - (off - r.offset)) return true;
        return false;
    };
    for (const KVManager* manager : { &kv, &follower }) {
        const auto& ranges = manager == &kv ? required : on_follower;
        const auto* p = manager->plan(1);
        for (int64_t l = 0; l < p->n_layers; ++l)
            for (int32_t b : original)
                CHECK(covers(ranges, p->pool_offset + l * p->layer_stride() + b * 68, 68));
        for (int32_t g : { 2, 3 }) {
            const auto* fixed = manager->plan(g);
            CHECK(covers(ranges, fixed->pool_offset, fixed->pool_bytes));
        }
    }
    const auto copies = kv.take_pending_copies();
    REQUIRE_EQ(copies.size(), 1u);
    const auto* p = kv.plan(1);
    for (int64_t l = 0; l < p->n_layers; ++l)
        CHECK(covers(required, p->pool_offset + l * p->layer_stride() + copies[0].dst_block * 68, 68));
    kv.release_block(1, original[0]);
    REQUIRE_EQ(kv.backing_required_ranges(kv, &required), RAD_OK);
    CHECK(!covers(required, p->pool_offset + original[0] * 68, 68));
}

RAD_TEST_MAIN()
