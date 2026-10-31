/* SOURCE host oracle: no model payload, GPU, or large native backup allocation.
 * Pure native Program/Plan addresses feed the same admission function used by
 * the real wrapper. Tiny raw spans exercise its actual copy/fault mechanism. */
#include "rad_test.h"
#include "dspark_bank_plan_fixture.h"
#include "place/draft_bank_backup.h"
#include "place/draft_bank_complement.h"
#include "place/planner.h"

#include <array>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

using namespace rad;
namespace d = rad::draft_bank_backup_detail;
namespace {
void* at(uintptr_t p) { return reinterpret_cast<void*>(p); }
struct Native {
    Program p;
    Plan plan;
    std::vector<void*> units, weights;
    Native(int n, int rank = 0) : p(dspark_scope_fixture::make(rank)) {
        void* first[3][2] = {};
        for (int s = 0; s < 3; ++s) for (int e = 0; e < 32; ++e) {
            MoveUnit mu;
            mu.layer = 40 + s; mu.expert = e; mu.bytes = kDraftPairBytes;
            mu.weights = {uint32_t(1 + 64 * s + e), uint32_t(1 + 64 * s + 32 + e)};
            mu.offset = {0, 12533760};
            mu.draft_vram = e < n; mu.draft_parked = e >= n;
            mu.tier = e < n ? Tier::VRAM : Tier::Mapped;
            mu.slab_class = e < n ? 0 : -1; mu.slot_bytes = e < n ? kDraftPairBytes : 0;
            const uintptr_t base = uintptr_t{0x10000000} + uintptr_t(s * 32 + e) * uintptr_t{0x02000000};
            units.push_back(e < n ? at(base) : nullptr);
            for (int k = 0; k < 2; ++k) {
                if (e < n && !first[s][k]) first[s][k] = at(base + uintptr_t(mu.offset[k]));
                auto& w = p.weights[mu.weights[k]];
                w.tier = mu.tier; w.site = Site::Device;
                w.ptr = e < n ? at(base + uintptr_t(mu.offset[k])) : first[s][k];
            }
            plan.units.push_back(std::move(mu));
        }
        for (int s = 0; s < 3; ++s) {
            WeightInfo w; w.name = "mask." + std::to_string(s);
            w.decl.group.layer = 40 + s; w.decl.group.expert = -1;
            w.decl.dtype = RAD_F32; w.decl.rank = 1; w.decl.shape[0] = 128;
            w.tier = Tier::VRAM; w.site = Site::Device; w.stored_bytes = 512;
            w.ptr = at(uintptr_t{0xf0000000} + uintptr_t(s) * 4096);
            OpInfo op; op.op = "router_scored_topk"; op.weights = {uint32_t(p.weights.size())};
            p.ops.push_back(op); p.weights.push_back(w);
        }
        for (const auto& w : p.weights) weights.push_back(w.ptr);
    }
    d::Addresses addresses() {
        return {this,
            [](void* self, int32_t u) -> void* {
                auto& v = static_cast<Native*>(self)->units;
                return u >= 0 && size_t(u) < v.size() ? v[size_t(u)] : nullptr;
            },
            [](void* self, uint32_t w) -> void* {
                auto& v = static_cast<Native*>(self)->weights;
                return w < v.size() ? v[w] : nullptr;
            }};
    }
    int derive(int n, d::Layout* out, std::string* why) {
        return d::derive_layout(p, plan, n, addresses(), out, why);
    }
};

struct Tiny {
    struct Rank {
        std::array<uint8_t, 17> a{};
        std::array<uint8_t, 33> b{};
        std::array<uint8_t, 9> mask{};
    };
    std::array<Rank, 4> data{};
    std::array<d::ImageRank, 4> image{};
    Tiny() {
        for (int r = 0; r < 4; ++r) {
            for (size_t i = 0; i < data[r].a.size(); ++i) data[r].a[i] = uint8_t(11 * r + i);
            for (size_t i = 0; i < data[r].b.size(); ++i) data[r].b[i] = uint8_t(0xff - 17 * r - i);
            // Include zero/0xff tails so a fixed poison cannot pass this oracle.
            data[r].a.back() = 0; data[r].b.back() = 0xff;
            for (size_t i = 0; i < data[r].mask.size(); ++i) data[r].mask[i] = uint8_t(3 * r + i);
            image[r].spans[0] = {data[r].a.data(), int64_t(data[r].a.size()), true};
            image[r].spans[1] = {data[r].b.data(), int64_t(data[r].b.size()), true};
            image[r].spans[2] = {data[r].mask.data(), int64_t(data[r].mask.size()), false};
            image[r].nspans = 3; image[r].scratch_bytes = 33;
        }
    }
    void lend() {
        for (auto& r : data) { r.a.fill(0x73); r.b.fill(0x19); }
    }
};

struct IO : DraftBankBackupIO {
    int rooms = 0, allocs = 0, frees = 0, copies = 0;
    int fail_alloc = -1, fail_copy = -1, short_copy = -1;
    bool deny = false, undercharge = false;
    int64_t extra_charge = 16;
    int64_t need = -1, still = -1, margin = -1;
    int64_t allocation_charge(int64_t n) const noexcept override {
        if (undercharge) return n - 1;
        return n > INT64_MAX - 31 - extra_charge ? -1 : ((n + 15) / 16) * 16 + extra_charge;
    }
    bool room(int64_t n, int64_t s, int64_t m, std::string* why) override {
        ++rooms; need = n; still = s; margin = m;
        if (deny && why) *why = "injected MemAvailable/cgroup refusal";
        return !deny;
    }
    void* alloc_pinned(int64_t n) override {
        ++allocs;
        return allocs == fail_alloc ? nullptr : std::malloc(size_t(n));
    }
    void free_pinned(void* p) noexcept override { ++frees; std::free(p); }
    int copy_sync(int rank, void* dst, const void* src, int64_t bytes) override {
        CHECK(rank >= 0 && rank < 4);
        ++copies;
        const bool fail = copies == fail_copy;
        const int64_t n = (copies == short_copy || fail) ? bytes - 1 : bytes;
        if (n > 0) std::memcpy(dst, src, size_t(n));
        return fail ? RAD_E_IO : RAD_OK;
    }
};

DraftBankBackupBudget budget() {
    DraftBankBackupBudget b;
    b.backup_mib = 1;
    b.expert_cache_budget_bytes = int64_t{96} << 30;
    b.expert_cache_backed_bytes = int64_t{90} << 30;
    b.other_future_bytes = 987;
    b.system_headroom_bytes = int64_t{2} << 30;
    b.controller_metadata_bytes = 2176;
    return b;
}
int capture(d::Image& out, Tiny& t, IO& io, std::string* why) {
    return out.capture(t.image, budget(), sizeof(out), &io, why);
}
} // namespace

TEST(draft_bank_backup_derives_every_native_n_without_allocating_payload) {
    for (int rank = 0; rank < 4; ++rank) for (int n = 1; n <= 32; ++n) {
        Native f(n, rank); d::Layout out; std::string why;
        REQUIRE_EQ(f.derive(n, &out, &why), RAD_OK);
        CHECK_EQ(out.nslots, 3 * n); CHECK_EQ(out.nbindings, 192);
        int64_t bytes = 0;
        for (int i = 0; i < out.nslots; ++i) {
            const auto& s = out.slots[i];
            CHECK_EQ(s.bytes, kDraftPairBytes);
            CHECK_EQ(s.base, f.units[size_t(s.hot_unit)]);
            CHECK_EQ(s.stage, i / n); CHECK_EQ(s.expert, i % n);
            bytes += s.bytes;
        }
        CHECK_EQ(bytes, int64_t(3 * n) * kDraftPairBytes);
        REQUIRE_EQ(d::check_bindings(f.p, out, f.addresses(), &why), RAD_OK);
    }
}

TEST(draft_bank_backup_refuses_nonraw_unbounded_and_incomplete_native_layouts) {
    auto refused = [](auto change) {
        Native f(2); change(f); d::Layout out; std::string why;
        CHECK(f.derive(2, &out, &why) < 0); CHECK(!why.empty());
    };
    refused([](Native& f) { f.plan.units[0].slot_bytes--; });
    refused([](Native& f) { f.plan.units[0].bytes--; });
    refused([](Native& f) { f.plan.units[0].vram_bytes = 4096; });
    refused([](Native& f) { f.plan.units[0].vram_offset = {0, 4096}; });
    refused([](Native& f) { f.plan.units[0].offset[1]++; });
    refused([](Native& f) { f.plan.units[0].movable = true; });
    refused([](Native& f) { f.plan.units[0].host_forced = true; });
    refused([](Native& f) { f.plan.units[0].slab_slot = 0; });
    refused([](Native& f) { f.plan.units[2].slot_bytes = kDraftPairBytes; });
    refused([](Native& f) { f.plan.units[2].slab_class = 0; });
    refused([](Native& f) { f.units[1] = f.units[0]; });
    refused([](Native& f) { f.units[0] = at(UINTPTR_MAX - 255); });
    refused([](Native& f) { f.plan.units.pop_back(); });
    refused([](Native& f) { f.p.weights[1].vram_pk_cap = 0; });
    refused([](Native& f) { f.p.weights[1].layout_tag = "unknown"; });
    refused([](Native& f) { f.p.weights[1].tier = Tier::HostPinned; });
    refused([](Native& f) { f.p.weights[3].ptr = f.p.weights[2].ptr; }); // parked alias
    refused([](Native& f) { f.weights[3] = f.weights[2]; });
    refused([](Native& f) { f.p.ctx.world_size = 2; });
    refused([](Native& f) { f.p.weights.back().ptr = f.units[0]; }); // mask overlap/binding
    refused([](Native& f) {
        f.p.weights.back().ptr = f.units[0]; f.weights.back() = f.units[0];
    }); // matching publications still cannot overlap a lendable slot
    Native f(1); d::Layout out; std::string why;
    CHECK(f.derive(0, &out, &why) < 0); CHECK(f.derive(33, &out, &why) < 0);
    CHECK(f.derive(2, &out, &why) < 0);
}

TEST(draft_bank_backup_detects_all_hot_parked_and_mask_pointer_mutations) {
    Native f(1); d::Layout out; std::string why;
    REQUIRE_EQ(f.derive(1, &out, &why), RAD_OK);
    for (const auto& b : out.bindings) {
        f.p.weights[b.weight].ptr = at(uintptr_t(b.ptr) + 256);
        CHECK(d::check_bindings(f.p, out, f.addresses(), &why) < 0);
        f.p.weights[b.weight].ptr = b.ptr;
        f.weights[b.weight] = at(uintptr_t(b.ptr) + 256);
        CHECK(d::check_bindings(f.p, out, f.addresses(), &why) < 0);
        f.weights[b.weight] = b.ptr;
        f.p.weights[b.weight].tier = Tier::HostPinned;
        CHECK(d::check_bindings(f.p, out, f.addresses(), &why) < 0);
        f.p.weights[b.weight].tier = b.hot ? Tier::VRAM : Tier::Mapped;
    }
    for (const auto& b : out.planes) {
        f.weights[b.weight] = at(uintptr_t(b.ptr) + 256);
        CHECK(d::check_bindings(f.p, out, f.addresses(), &why) < 0);
        f.weights[b.weight] = b.ptr;
    }
    f.units[0] = at(uintptr_t(f.units[0]) + 256);
    CHECK(d::check_bindings(f.p, out, f.addresses(), &why) < 0);
}

TEST(draft_bank_backup_caps_all_roles_and_reserves_only_future_cache) {
    Tiny t; IO io; d::Image out; std::string why;
    REQUIRE_EQ(capture(out, t, io, &why), RAD_OK);
    const auto& s = out.stats();
    CHECK_EQ(io.rooms, 1); CHECK_EQ(io.allocs, 4);
    CHECK_EQ(s.raw_bytes, int64_t(4 * (17 + 33)));
    CHECK_EQ(s.mask_bytes, int64_t(4 * 9));
    CHECK_EQ(s.shared_scratch_bytes, int64_t(4 * 33));
    CHECK_EQ(s.relay_bytes, s.shared_scratch_bytes);
    CHECK_EQ(s.readback_bytes, s.shared_scratch_bytes);
    CHECK_EQ(s.pinned_requested_bytes, s.raw_bytes + s.mask_bytes + s.shared_scratch_bytes);
    CHECK_EQ(s.total_charge_bytes, s.pinned_charge_bytes + s.metadata_bytes);
    CHECK_EQ(s.controller_metadata_bytes, int64_t{2176});
    CHECK_EQ(s.metadata_bytes, int64_t(sizeof(out)) + 2176);
    CHECK_EQ(io.need, s.total_charge_bytes);
    CHECK_EQ(io.still, (int64_t{6} << 30) + 987);
    CHECK_EQ(io.margin, int64_t{2} << 30);
    CHECK_EQ(s.capture_copy_bytes, s.raw_bytes + s.mask_bytes);
    CHECK_EQ(s.verify_copy_bytes, s.capture_copy_bytes);
    CHECK(out.ready()); CHECK(!out.failed());
    for (int r = 0; r < 4; ++r) { CHECK(out.scratch(r)); CHECK_EQ(out.scratch_bytes(r), 33); }
    CHECK(!out.scratch(-1)); CHECK(!out.scratch(4));
    // Variable budgets, including rounded backed storage above the budget.
    for (int64_t gib : {int64_t{96}, int64_t{128}, int64_t{192}}) {
        IO adapter; d::Image image; auto b = budget();
        b.expert_cache_budget_bytes = gib << 30; b.expert_cache_backed_bytes = (gib + 1) << 30;
        REQUIRE_EQ(image.capture(t.image, b, sizeof(image), &adapter, &why), RAD_OK);
        CHECK_EQ(adapter.still, b.other_future_bytes);
    }
}

TEST(draft_bank_backup_refuses_budget_room_and_allocation_before_reading_bank) {
    Tiny t; std::string why;
    auto refused = [&](DraftBankBackupBudget b) {
        IO io; d::Image image;
        CHECK(image.capture(t.image, b, sizeof(image), &io, &why) < 0);
        CHECK_EQ(io.allocs, 0); CHECK_EQ(io.copies, 0);
        CHECK(image.failed()); CHECK(!image.ready());
    };
    auto b = budget(); b.backup_mib = 0; refused(b);
    b = budget(); b.backup_mib = INT64_MAX; refused(b);
    b = budget(); b.expert_cache_budget_bytes = -1; refused(b);
    b = budget(); b.system_headroom_bytes = -1; refused(b);
    b = budget(); b.controller_metadata_bytes = -1; refused(b);
    b = budget(); b.controller_metadata_bytes = INT64_MAX; refused(b);
    b = budget(); b.expert_cache_budget_bytes = INT64_MAX; b.expert_cache_backed_bytes = 0;
    b.other_future_bytes = 1; refused(b);
    { IO io; io.extra_charge = int64_t{1} << 20; d::Image image;
      CHECK(capture(image, t, io, &why) < 0); CHECK_EQ(io.allocs, 0); CHECK_EQ(io.rooms, 0); }
    { IO io; io.undercharge = true; d::Image image;
      CHECK(capture(image, t, io, &why) < 0); CHECK_EQ(io.allocs, 0); CHECK_EQ(io.rooms, 0); }
    { IO io; io.deny = true; d::Image image;
      CHECK_EQ(capture(image, t, io, &why), RAD_E_NOMEM);
      CHECK_EQ(io.allocs, 0); CHECK_EQ(io.copies, 0); CHECK(why.find("cgroup") != std::string::npos); }
    for (int rank = 1; rank <= 4; ++rank) {
        IO io; io.fail_alloc = rank;
        { d::Image image; CHECK_EQ(capture(image, t, io, &why), RAD_E_NOMEM);
          CHECK(image.failed()); CHECK_EQ(io.copies, 0); }
        CHECK_EQ(io.frees, rank - 1);
    }
}

TEST(draft_bank_backup_restores_every_byte_same_address_and_reuses_one_scratch) {
    Tiny t; const auto original = t.data; IO io; d::Image out; std::string why;
    REQUIRE_EQ(capture(out, t, io, &why), RAD_OK);
    const auto hash = out.stats().image_hash;
    std::array<void*, 4> scratch{};
    for (int r = 0; r < 4; ++r) scratch[r] = out.scratch(r);
    for (int cycle = 0; cycle < 3; ++cycle) {
        out.mark_borrowed(); CHECK(!out.ready()); CHECK(!out.failed());
        t.lend();
        // Controller relay use may overwrite every scratch byte; never image bytes.
        for (int r = 0; r < 4; ++r) std::memset(out.scratch(r), cycle, 33);
        REQUIRE_EQ(out.restore_verify_all(&why), RAD_OK);
        CHECK(out.ready()); CHECK_EQ(out.stats().image_hash, hash);
        CHECK_EQ(out.stats().completed_restores, uint64_t(cycle + 1));
        for (int r = 0; r < 4; ++r) {
            CHECK_EQ(out.scratch(r), scratch[r]);
            CHECK_EQ(std::memcmp(t.data[r].a.data(), original[r].a.data(), 17), 0);
            CHECK_EQ(std::memcmp(t.data[r].b.data(), original[r].b.data(), 33), 0);
            CHECK_EQ(std::memcmp(t.data[r].mask.data(), original[r].mask.data(), 9), 0);
        }
    }
    CHECK_EQ(io.allocs, 4); // no transition allocation
    CHECK_EQ(out.stats().restore_copy_bytes, 3 * out.stats().raw_bytes);
}

TEST(draft_bank_backup_short_capture_and_readback_are_fatal_including_zero_tails) {
    for (int copy = 1; copy <= 24; ++copy) {
        Tiny t; IO io; io.short_copy = copy; d::Image image; std::string why;
        // First-capture truncation matters when it loses a nonzero byte. A
        // skipped zero written into an already-zero snapshot is still exact.
        if (copy % 2) for (auto& r : t.data) {
            r.a.back() = 0x61; r.b.back() = 0x72; r.mask.back() = 0x83;
        }
        CHECK(capture(image, t, io, &why) < 0);
        CHECK(image.failed()); CHECK(!image.ready());
        CHECK(image.restore_verify_all(&why) < 0);
    }
}

TEST(draft_bank_backup_partial_restore_and_mask_change_never_publish_ready) {
    for (bool short_success : {false, true}) for (int rank = 0; rank < 4; ++rank) {
        Tiny t; IO io; d::Image image; std::string why;
        REQUIRE_EQ(capture(image, t, io, &why), RAD_OK);
        image.mark_borrowed(); t.lend();
        // Restore begins with four unchanged-mask reads; each rank then has two
        // write/readback pairs. Fail its second raw slot after a partial write.
        const int call = io.copies + 4 + 4 * rank + 3;
        if (short_success) io.short_copy = call; else io.fail_copy = call;
        CHECK(image.restore_verify_all(&why) < 0);
        CHECK(image.failed()); CHECK(!image.ready());
        CHECK_EQ(image.stats().completed_restores, uint64_t{0});
        io.fail_copy = io.short_copy = -1;
        CHECK(image.restore_verify_all(&why) < 0); // failure latch is irreversible
    }
    Tiny t; IO io; d::Image image; std::string why;
    REQUIRE_EQ(capture(image, t, io, &why), RAD_OK);
    image.mark_borrowed(); t.lend(); t.data[3].mask[8] ^= 0x80;
    const int64_t restored = image.stats().restore_copy_bytes;
    CHECK(image.restore_verify_all(&why) < 0);
    CHECK_EQ(image.stats().restore_copy_bytes, restored);
    CHECK(image.failed()); CHECK(!image.ready());
}

TEST(draft_bank_backup_rejects_overlap_bad_spans_and_recapture) {
    Tiny t; std::string why;
    { IO io; d::Image out; auto ranks = t.image;
      ranks[0].spans[1] = ranks[0].spans[0];
      CHECK(out.capture(ranks, budget(), sizeof(out), &io, &why) < 0); CHECK_EQ(io.allocs, 0); }
    { IO io; d::Image out; auto ranks = t.image; ranks[3].nspans = 0;
      CHECK(out.capture(ranks, budget(), sizeof(out), &io, &why) < 0); CHECK_EQ(io.allocs, 0); }
    { IO io; d::Image out; auto ranks = t.image; ranks[1].scratch_bytes = 32;
      CHECK(out.capture(ranks, budget(), sizeof(out), &io, &why) < 0); CHECK_EQ(io.allocs, 0); }
    { IO io; d::Image out; auto ranks = t.image; ranks[1].spans[0].base = at(UINTPTR_MAX - 1);
      CHECK(out.capture(ranks, budget(), sizeof(out), &io, &why) < 0); CHECK_EQ(io.allocs, 0); }
    { IO io; d::Image out;
      REQUIRE_EQ(capture(out, t, io, &why), RAD_OK);
      CHECK(capture(out, t, io, &why) < 0); CHECK(out.failed()); CHECK(!out.ready()); }
}

TEST(draft_bank_backup_real_allocator_charge_covers_no_svm_padding) {
    DraftBankBackupDeviceIO io; // pure accounting; constructor does not open a device
    constexpr int64_t align = int64_t{2} << 20;
    CHECK_EQ(io.allocation_charge(1), 2 * align);
    CHECK_EQ(io.allocation_charge(align), 2 * align);
    CHECK_EQ(io.allocation_charge(align + 1), 3 * align);
    CHECK_EQ(io.allocation_charge(0), int64_t{-1});
    CHECK_EQ(io.allocation_charge(INT64_MAX), int64_t{-1});
}

TEST(draft_bank_backup_timings_off_preserves_hash_bytes_and_exact_copy_order) {
    struct TraceIO : IO {
        Tiny* source=nullptr;
        std::vector<uint64_t> trace;
        int copy_sync(int rank,void* dst,const void* src,int64_t bytes) override {
            const uintptr_t first=uintptr_t(source->data.data()), last=first+sizeof(source->data);
            const bool write=uintptr_t(dst)>=first&&uintptr_t(dst)<last;
            trace.push_back((uint64_t(rank)<<48)|(uint64_t(bytes)<<1)|uint64_t(write));
            return IO::copy_sync(rank,dst,src,bytes);
        }
    };
    Tiny a,b;TraceIO off,on;off.source=&a;on.source=&b;d::Image ia,ib;std::string why;
    auto ba=budget(),bb=budget();bb.timings=true;
    REQUIRE_EQ(ia.capture(a.image,ba,sizeof(ia),&off,&why),RAD_OK);
    REQUIRE_EQ(ib.capture(b.image,bb,sizeof(ib),&on,&why),RAD_OK);
    CHECK_EQ(ia.stats().image_hash,ib.stats().image_hash);CHECK(off.trace==on.trace);
    ia.mark_borrowed();ib.mark_borrowed();a.lend();b.lend();
    REQUIRE_EQ(ia.restore_verify_all(&why),RAD_OK);REQUIRE_EQ(ib.restore_verify_all(&why),RAD_OK);
    CHECK(off.trace==on.trace);CHECK_EQ(std::memcmp(a.data.data(),b.data.data(),sizeof(a.data)),0);
    CHECK_EQ(ia.stats().capture_copy_bytes,ib.stats().capture_copy_bytes);
    CHECK_EQ(ia.stats().restore_copy_bytes,ib.stats().restore_copy_bytes);
    CHECK_EQ(ia.stats().verify_copy_bytes,ib.stats().verify_copy_bytes);
    CHECK(!ia.stats().timings.enabled);CHECK(!ia.stats().timings.overflow);
    for(const auto& c:ia.stats().timings.part){CHECK_EQ(c.ns,uint64_t{0});CHECK_EQ(c.attempts,uint64_t{0});
        CHECK_EQ(c.ok,uint64_t{0});CHECK_EQ(c.requested_bytes,uint64_t{0});CHECK_EQ(c.ok_bytes,uint64_t{0});}
    CHECK_EQ(ib.stats().metadata_bytes-ia.stats().metadata_bytes,kDraftBankTimingWorkspaceBytes);
}

TEST(draft_bank_backup_timings_counter_overflow_is_transactional_and_latched) {
    auto fields=[](const DraftBankTimingCounter& c){return std::array<uint64_t,5>{c.ns,c.attempts,c.ok,c.requested_bytes,c.ok_bytes};};
    for(int field=0;field<5;++field){DraftBankTimingCounter c;
        switch(field){case 0:c.ns=UINT64_MAX;break;case 1:c.attempts=UINT64_MAX;break;
        case 2:c.ok=UINT64_MAX;break;case 3:c.requested_bytes=UINT64_MAX;break;case 4:c.ok_bytes=UINT64_MAX;break;}
        const auto before=fields(c);CHECK(!c.record(1,1,true));CHECK(fields(c)==before);}
    DraftBankTimings t;
    CHECK(t.record(DraftBankTimingPart::CaptureD2H,3,4,true));CHECK_EQ(t.part[2].attempts,uint64_t{0});
    t.enabled=true;t.part[2].requested_bytes=UINT64_MAX;
    CHECK(!t.record(DraftBankTimingPart::CaptureD2H,1,1,true));CHECK(t.overflow);
    CHECK(!t.record(DraftBankTimingPart::Zero,0,0,true));CHECK_EQ(t.part[1].attempts,uint64_t{0});
    DraftBankTimings fresh;fresh.enabled=true;
    CHECK(fresh.record(DraftBankTimingPart::Zero,0,0,true));CHECK(!fresh.overflow);
    CHECK_EQ(fresh.part[1].attempts,uint64_t{1});CHECK_EQ(fresh.part[1].ok,uint64_t{1});
}

TEST(draft_bank_backup_timings_reconcile_exclusive_copy_counts_and_honest_workspace) {
    Tiny t;IO io;d::Image image;std::string why;auto b=budget();b.timings=true;
    REQUIRE_EQ(image.capture(t.image,b,sizeof(image),&io,&why),RAD_OK);
    auto c=[&](DraftBankTimingPart p)->const DraftBankTimingCounter&{return image.stats().timings.part[size_t(p)];};
    CHECK_EQ(c(DraftBankTimingPart::Allocation).attempts,uint64_t{4});
    CHECK_EQ(c(DraftBankTimingPart::Allocation).ok_bytes,uint64_t{368});
    for(auto p:{DraftBankTimingPart::Zero,DraftBankTimingPart::CaptureD2H,DraftBankTimingPart::CaptureFNV,
                DraftBankTimingPart::CaptureComplement,DraftBankTimingPart::CaptureVerifyD2H,DraftBankTimingPart::CaptureMemcmp}) {
        CHECK_EQ(c(p).attempts,uint64_t{12});CHECK_EQ(c(p).ok,uint64_t{12});
        CHECK_EQ(c(p).requested_bytes,uint64_t{236});CHECK_EQ(c(p).ok_bytes,uint64_t{236});}
    CHECK_EQ(image.stats().metadata_bytes,int64_t(sizeof(image))+b.controller_metadata_bytes+kDraftBankTimingWorkspaceBytes);
    CHECK_EQ(io.need,image.stats().total_charge_bytes); // includes all retained stats + fixed log/logger stack BEFORE allocation
    REQUIRE_EQ(image.verify_unchanged(&why),RAD_OK); // pre-lend invariant check
    image.mark_borrowed();t.lend();REQUIRE_EQ(image.restore_verify_all(&why),RAD_OK);
    for(auto p:{DraftBankTimingPart::RestoreH2D,DraftBankTimingPart::RestoreComplement,DraftBankTimingPart::RestoreD2H,DraftBankTimingPart::RestoreMemcmp}) {
        CHECK_EQ(c(p).attempts,uint64_t{8});CHECK_EQ(c(p).ok_bytes,uint64_t{200});}
    for(auto p:{DraftBankTimingPart::InvariantComplement,DraftBankTimingPart::InvariantD2H,DraftBankTimingPart::InvariantMemcmp}) {
        CHECK_EQ(c(p).attempts,uint64_t{12});CHECK_EQ(c(p).ok_bytes,uint64_t{108});}
    CHECK_EQ(uint64_t(image.stats().verify_copy_bytes),c(DraftBankTimingPart::CaptureVerifyD2H).ok_bytes+
        c(DraftBankTimingPart::RestoreD2H).ok_bytes+c(DraftBankTimingPart::InvariantD2H).ok_bytes);
    CHECK_EQ(uint64_t(image.stats().restore_copy_bytes),c(DraftBankTimingPart::RestoreH2D).ok_bytes);
    CHECK(image.ready());CHECK(!image.stats().timings.overflow);
}

TEST(draft_bank_backup_timings_failed_allocation_copy_and_recapture_keep_partial_proof) {
    std::string why;
    {Tiny t;IO io;io.fail_alloc=3;d::Image image;auto b=budget();b.timings=true;
        CHECK_EQ(image.capture(t.image,b,sizeof(image),&io,&why),RAD_E_NOMEM);
        const auto before=image.stats().timings;
        CHECK_EQ(before.part[size_t(DraftBankTimingPart::Allocation)].attempts,uint64_t{3});
        CHECK_EQ(before.part[size_t(DraftBankTimingPart::Allocation)].ok,uint64_t{2});
        CHECK_EQ(io.frees,2);CHECK_EQ(io.copies,0);b.timings=false;
        CHECK(image.capture(t.image,b,sizeof(image),&io,&why)<0);
        CHECK(image.stats().timings.enabled);CHECK_EQ(image.stats().timings.part[0].attempts,before.part[0].attempts);
        CHECK(image.failed());CHECK(!image.ready());}
    {Tiny t;IO io;io.fail_copy=1;d::Image image;auto b=budget();b.timings=true;
        CHECK_EQ(image.capture(t.image,b,sizeof(image),&io,&why),RAD_E_IO);
        const auto& c=image.stats().timings.part[size_t(DraftBankTimingPart::CaptureD2H)];
        CHECK_EQ(c.attempts,uint64_t{1});CHECK_EQ(c.ok,uint64_t{0});CHECK_EQ(c.requested_bytes,uint64_t{17});CHECK_EQ(c.ok_bytes,uint64_t{0});
        CHECK_EQ(io.frees,4);CHECK_EQ(io.copies,1);CHECK(image.failed());CHECK(!image.ready());}
}

TEST(draft_bank_backup_timings_short_restore_records_reported_copy_and_failed_memcmp) {
    Tiny t;IO io;d::Image image;auto b=budget();b.timings=true;std::string why;
    REQUIRE_EQ(image.capture(t.image,b,sizeof(image),&io,&why),RAD_OK);
    image.mark_borrowed();t.lend();io.short_copy=io.copies+5; // four invariant reads, then first raw H2D
    CHECK(image.restore_verify_all(&why)<0);CHECK(image.failed());CHECK(!image.ready());
    const auto& w=image.stats().timings.part[size_t(DraftBankTimingPart::RestoreH2D)];
    const auto& cmp=image.stats().timings.part[size_t(DraftBankTimingPart::RestoreMemcmp)];
    CHECK_EQ(w.attempts,uint64_t{1});CHECK_EQ(w.ok,uint64_t{1});CHECK_EQ(w.ok_bytes,uint64_t{17});
    CHECK_EQ(cmp.attempts,uint64_t{1});CHECK_EQ(cmp.ok,uint64_t{0});CHECK_EQ(cmp.requested_bytes,uint64_t{17});
    const int copies=io.copies;CHECK(image.restore_verify_all(&why)<0);CHECK_EQ(io.copies,copies);
}

TEST(draft_bank_backup_timings_image_counter_overflow_poison_survives_retry) {
    Tiny t;IO io;d::Image image;auto b=budget();b.timings=true;std::string why;
    REQUIRE_EQ(image.capture(t.image,b,sizeof(image),&io,&why),RAD_OK);
    // Host-only fault injection into a nonconst underlying Image. No runtime setter/env bypass.
    auto& counters=const_cast<DraftBankTimings&>(image.stats().timings);
    counters.part[size_t(DraftBankTimingPart::RestoreH2D)].attempts=UINT64_MAX;
    image.mark_borrowed();t.lend();CHECK(image.restore_verify_all(&why)<0);
    CHECK(image.stats().timings.overflow);CHECK(image.failed());CHECK(!image.ready());
    const int copies=io.copies;CHECK(image.restore_verify_all(&why)<0);CHECK_EQ(io.copies,copies);
    CHECK(image.capture(t.image,b,sizeof(image),&io,&why)<0);CHECK(image.stats().timings.overflow);
    d::Image fresh;CHECK(!fresh.stats().timings.enabled);CHECK(!fresh.stats().timings.overflow);
    CHECK_EQ(fresh.stats().timings.part[0].attempts,uint64_t{0});
}

TEST(draft_bank_complement_words_matches_every_byte_unaligned_tail_and_guards) {
    for (int pattern = 0; pattern < 4; ++pattern) {
        std::array<uint8_t, 320> source{};
        for (size_t i = 0; i < source.size(); ++i)
            source[i] = pattern == 0 ? uint8_t(i) : pattern == 1 ? uint8_t(255-i) : pattern == 2 ? 0 : 255;
        const auto immutable = source;
        for (int so = 0; so < 16; ++so) for (int offset = 0; offset < 16; ++offset)
            for (int bytes : {0,1,2,7,8,9,15,16,17,31,32,33,63,64,65,127,128,129,256}) {
                std::array<uint8_t, 320> expected, actual;
                expected.fill(0x5a); actual = expected;
                for (int j = 0; j < bytes; ++j) expected[size_t(offset+j)] = uint8_t(~source[size_t(so+j)]);
                CHECK(d::complement_words(actual.data()+offset,source.data()+so,bytes));
                CHECK(actual == expected); // includes every untouched prefix/tail guard
                CHECK(source == immutable);
            }
    }
}

TEST(draft_bank_complement_words_preserves_forward_overlap_and_empty_range_semantics) {
    CHECK(d::complement_words(nullptr,nullptr,0));
    CHECK(!d::complement_words(nullptr,nullptr,1));
    CHECK(!d::complement_words(nullptr,nullptr,-1));
    CHECK(!d::complement_words(at(UINTPTR_MAX-3),at(16),8));
    CHECK(!d::complement_words(at(16),at(UINTPTR_MAX-3),8));
    for (int delta = -17; delta <= 17; ++delta) for (int bytes : {0,1,7,8,9,16,17,63,127}) {
        std::array<uint8_t, 256> expected{}, actual{};
        for (size_t i = 0; i < actual.size(); ++i) actual[i] = uint8_t(37*i+11);
        expected = actual;
        for (int j = 0; j < bytes; ++j) expected[size_t(64+delta+j)] = uint8_t(~expected[size_t(64+j)]);
        CHECK(d::complement_words(actual.data()+64+delta,actual.data()+64,bytes));
        CHECK(actual == expected); // especially output one byte after the source
    }
}

TEST(draft_bank_complement_words_image_preserves_hash_copy_order_bytes_and_fixed_workspace) {
    struct TraceIO : IO {
        Tiny* source = nullptr;
        std::vector<uint64_t> trace;
        int copy_sync(int rank,void* dst,const void* src,int64_t bytes) override {
            const uintptr_t first=uintptr_t(source->data.data()),last=first+sizeof(source->data);
            const bool write=uintptr_t(dst)>=first&&uintptr_t(dst)<last;
            const uintptr_t device=write?uintptr_t(dst):uintptr_t(src);
            trace.push_back((uint64_t(rank)<<48)|(uint64_t(bytes)<<16)|((device-first)<<1)|uint64_t(write));
            return IO::copy_sync(rank,dst,src,bytes);
        }
    };
    Tiny a,b;TraceIO off,on;off.source=&a;on.source=&b;d::Image ia,ib;std::string why;
    auto ba=budget(),bb=budget();ba.timings=bb.timings=true;bb.complement_words=true;
    REQUIRE_EQ(ia.capture(a.image,ba,sizeof(ia),&off,&why),RAD_OK);
    REQUIRE_EQ(ib.capture(b.image,bb,sizeof(ib),&on,&why),RAD_OK);
    CHECK(!ia.stats().complement_words);CHECK(ib.stats().complement_words);
    CHECK_EQ(ia.stats().image_hash,ib.stats().image_hash);CHECK(off.trace==on.trace);
    CHECK_EQ(ib.stats().metadata_bytes-ia.stats().metadata_bytes,d::kComplementWordWorkspaceBytes);
    CHECK_EQ(ib.stats().total_charge_bytes-ia.stats().total_charge_bytes,d::kComplementWordWorkspaceBytes);
    CHECK_EQ(off.allocs,on.allocs);CHECK_EQ(off.need+d::kComplementWordWorkspaceBytes,on.need);
    for(int pass=0;pass<3;++pass){
        ia.mark_borrowed();ib.mark_borrowed();a.lend();b.lend();
        REQUIRE_EQ(ia.restore_verify_all(&why),RAD_OK);
        REQUIRE_EQ(ib.restore_verify_all(&why),RAD_OK);
        CHECK(off.trace==on.trace);CHECK_EQ(std::memcmp(a.data.data(),b.data.data(),sizeof(a.data)),0);
    }
    CHECK_EQ(ia.stats().capture_copy_bytes,ib.stats().capture_copy_bytes);
    CHECK_EQ(ia.stats().restore_copy_bytes,ib.stats().restore_copy_bytes);
    CHECK_EQ(ia.stats().verify_copy_bytes,ib.stats().verify_copy_bytes);
    for(size_t i=0;i<ia.stats().timings.part.size();++i){
        const auto& x=ia.stats().timings.part[i];const auto& y=ib.stats().timings.part[i];
        CHECK_EQ(x.attempts,y.attempts);CHECK_EQ(x.ok,y.ok);
        CHECK_EQ(x.requested_bytes,y.requested_bytes);CHECK_EQ(x.ok_bytes,y.ok_bytes);
    }
}

TEST(draft_bank_complement_words_short_capture_and_restore_readbacks_stay_fatal) {
    for (bool words : {false,true}) for (int rank=0;rank<4;++rank) {
        std::string why;auto b=budget();b.complement_words=words;
        { Tiny t;IO io;d::Image image;io.short_copy=6*rank+2; // first capture readback on this rank
          CHECK(image.capture(t.image,b,sizeof(image),&io,&why)<0);
          CHECK(image.failed());CHECK(!image.ready());CHECK(why.find("strict byte")!=std::string::npos); }
        for(int direction=0;direction<2;++direction){
            Tiny t;IO io;d::Image image;
            REQUIRE_EQ(image.capture(t.image,b,sizeof(image),&io,&why),RAD_OK);
            image.mark_borrowed();t.lend();
            io.short_copy=io.copies+4+4*rank+1+direction; // H2D or D2H after four invariant reads
            CHECK(image.restore_verify_all(&why)<0);CHECK(image.failed());CHECK(!image.ready());
            CHECK(why.find("strict byte")!=std::string::npos);
            const int calls=io.copies;CHECK(image.restore_verify_all(&why)<0);CHECK_EQ(io.copies,calls);
        }
    }
}

TEST(draft_bank_complement_words_workspace_cap_refuses_before_any_allocation) {
    Tiny t;IO old_io;d::Image old;auto b=budget();std::string why;
    int64_t pinned=0;
    for(const auto& rank:t.image){
        int64_t requested=rank.scratch_bytes;
        for(int i=0;i<rank.nspans;++i)requested+=rank.spans[i].bytes;
        pinned+=old_io.allocation_charge(requested);
    }
    const int64_t metadata=(b.backup_mib<<20)-pinned-b.controller_metadata_bytes;
    REQUIRE_EQ(old.capture(t.image,b,metadata,&old_io,&why),RAD_OK);
    CHECK_EQ(old.stats().total_charge_bytes,old.stats().cap_bytes);
    IO word_io;d::Image words;b.complement_words=true;
    CHECK(words.capture(t.image,b,metadata,&word_io,&why)<0);
    CHECK_EQ(word_io.rooms,0);CHECK_EQ(word_io.allocs,0);CHECK_EQ(word_io.copies,0);
    CHECK(words.failed());CHECK(!words.ready());
    CHECK_EQ(words.stats().total_charge_bytes,old.stats().cap_bytes+d::kComplementWordWorkspaceBytes);
}

RAD_TEST_MAIN()
