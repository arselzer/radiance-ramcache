/* Real-card VMem/Pool/allocator oracle. This is not a model/context-length gate.
 * Run only inside gpu_cmd_window.sh, with both research locks held and QFN stopped. */
#include "mem/kv.h"
#include "mem/kv_live_controller.h"
#include "mem/kv_backing_transaction.h"
#include "mem/pools.h"
#include "rad_internal.h"
#include "rad_device.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace rad;
namespace {
constexpr int R = 4;
constexpr int64_t scratch_bytes = 64 * 1024;
void need(bool value, const char* why) { if (!value) throw std::runtime_error(why); }
void checked(int status, const char* why) { need(status >= 0, why); }
int64_t bytes(const KVBackingSnapshot& s) {
    return int64_t(std::count(s.backed.begin(), s.backed.end(), uint8_t(1))) * s.granule_bytes;
}
struct Rank {
    int device = -1;
    Program program;
    Pools pools;
    KVManager kv;
    RadStream stream = nullptr;
    void* base = nullptr;
    int64_t stride = 0;
    ~Rank() {
        if (device >= 0 && rad_dev_set(device) >= 0 && stream) {
            (void)rad_stream_sync(stream);
            rad_stream_destroy(stream);
        }
        // Members destroy after this body, with their owning device still selected.
    }
};
struct Cards {
    std::array<std::unique_ptr<Rank>, R> ranks;
    // Controller is destroyed before managers/pools; it removes its leader callback.
    KVLiveController controller;
    KVLiveOptions options;
    void* seed = nullptr;
    void* readback = nullptr;
    int64_t G = 0, initial = 0;
    ~Cards() {
        if (rad_dev_set(0) >= 0) {
            if (seed) rad_dev_free(seed, RAD_MEM_HOST_PINNED);
            if (readback) rad_dev_free(readback, RAD_MEM_HOST_PINNED);
        }
    }
    void build() {
        need(rad_dev_count() == R, "requires exactly four visible real cards");
        for (int r = 0; r < R; ++r) {
            RadDeviceProps p{};
            checked(rad_dev_props(r, &p), "device properties failed");
            need(!p.is_host_backend && !std::strcmp(p.arch, "gfx1201"), "four gfx1201 cards required; host fallback is not evidence");
            checked(rad_dev_set(r), "select card failed");
            need(VMem::supported(), "real device VMem is unsupported");
            const int64_t g = VMem::granularity();
            need(g > 0 && g % kKVLiveMiB == 0 && g <= 64 * kKVLiveMiB && (!G || G == g), "unsupported/common granule size");
            G = g;
            need(p.vram_free >= 16 * G + 64 * kKVLiveMiB, "insufficient card room for bounded oracle plus headroom");
        }
        Config c;
        c.vram_weights_mib = c.gpu_headroom_mib = 1;
        c.vram_kv_mib = 16 * G / kKVLiveMiB;
        c.host_pool_mib = 0; c.max_seqs = 1; c.max_tok = 128; c.max_ctx = 4096; c.n_spec = 0;
        std::vector<KVGroupInfo> groups(1);
        for (int i = 0; i < 2; ++i) {
            KVGroupInfo g;
            g.name = i ? "latent" : "index";
            g.decl.kind = RAD_KV_FULL; g.decl.dtype = RAD_BF16;
            g.decl.n_head_kv = 1; g.decl.head_dim = i ? 128 : 32;
            g.decl.storage_format = i ? RAD_KV_STORAGE_DS41_FP4_E4M3_G16 : RAD_KV_STORAGE_DS41_FP4_POW2_G32;
            g.decl.storage_block_bytes = i ? 288 : 68;
            g.block_size = 2; g.layers = {2, 8}; groups.push_back(g);
        }
        KVGroupInfo raw;
        raw.name = "raw_f32"; raw.decl.kind = RAD_KV_LINEAR; raw.decl.dtype = RAD_F32;
        raw.decl.n_head_kv = 1; raw.decl.state_dim[0] = 2; raw.decl.state_dim[1] = 16;
        raw.layers = {2}; groups.push_back(raw);
        KVGroupInfo window;
        window.name = "bf16_window"; window.decl.kind = RAD_KV_WINDOW;
        window.decl.dtype = RAD_BF16; window.decl.n_head_kv = 1; window.decl.head_dim = 8;
        window.decl.window = 8; window.block_size = 4; window.layers = {2}; groups.push_back(window);
        std::vector<KVLiveController::Rank> cohort;
        for (int r = 0; r < R; ++r) {
            checked(rad_dev_set(r), "select card failed");
            auto rank = std::make_unique<Rank>(); rank->device = r; rank->program.kv_groups = groups;
            checked(rank->pools.configure(c, 0), "pool configuration failed");
            need(rank->pools.kv().elastic(), "fixed allocation is not a VMem gate");
            checked(rank->kv.plan_groups(groups, c), "KV plan failed");
            checked(rank->kv.enable_backing_mode(), "backing admission failed");
            checked(rank->kv.carve(rank->pools.kv()), "deferred KV carve failed");
            need(rank->pools.kv().committed() == 0, "startup eagerly backed the KV pool");
            rank->base = rank->pools.kv().base(); rank->stride = rank->kv.plan(1)->layer_stride();
            checked(rad_stream_create(&rank->stream, 0), "stream creation failed");
            cohort.push_back({r, &rank->pools.kv(), &rank->kv, &rank->program});
            ranks[r] = std::move(rank);
        }
        checked(rad_dev_set(0), "select rank0 failed");
        seed = rad_dev_alloc(scratch_bytes, RAD_MEM_HOST_PINNED);
        readback = rad_dev_alloc(scratch_bytes, RAD_MEM_HOST_PINNED);
        need(seed && readback, "bounded pinned scratch allocation failed");
        options = {true, G, 16 * G, 0, 0, 64 * kKVLiveMiB};
        checked(controller.start(options, cohort, 0, 0, 0), "actual all-rank startup failed");
        initial = ranks[0]->pools.kv().committed();
        need(initial > 0 && initial < options.maximum_bytes, "minimum startup did not save physical backing");
        stable(initial);
    }
    KVBackingSnapshot snapshot(int r) {
        PoolKVBackingBackend b(&ranks[r]->pools.kv(), r);
        KVBackingSnapshot s; checked(b.read(&s), "actual backing ledger read failed"); return s;
    }
    void stable(int64_t physical) {
        for (int r = 0; r < R; ++r) {
            need(ranks[r]->pools.kv().base() == ranks[r]->base &&
                 ranks[r]->kv.plan(1)->layer_stride() == ranks[r]->stride, "stable VA/stride changed");
            need(bytes(snapshot(r)) == physical && ranks[r]->pools.kv().committed() == physical,
                 "actual all-rank physical totals differ");
        }
    }
    void transfer(int r, int64_t off, int64_t count, bool write) {
        checked(rad_dev_set(r), "transfer device selection failed");
        need(ranks[r]->pools.kv().is_backed(off, count), "attempt to touch an unbacked extent");
        for (int64_t done = 0; done < count;) {
            const int64_t n = std::min(scratch_bytes, count - done);
            auto* p = static_cast<unsigned char*>(seed);
            for (int64_t j = 0; j < n; ++j)
                p[j] = uint8_t((uint64_t(off + done + j) * 29 + uint64_t(r) * 31) ^ (uint64_t(off + done + j) >> 8));
            void* where = static_cast<char*>(ranks[r]->base) + off + done;
            checked(rad_memcpy_async(write ? where : readback, write ? seed : where, n, ranks[r]->stream), "actual device transfer failed");
            checked(rad_stream_sync(ranks[r]->stream), "device transfer drain failed");
            if (!write) need(!std::memcmp(seed, readback, size_t(n)), "held GPU bytes changed across transaction");
            done += n;
        }
    }
    void fill_mapped() {
        for (int r = 0; r < R; ++r) {
            const auto s = snapshot(r);
            for (size_t i = 0; i < s.backed.size(); ++i) if (s.backed[i]) transfer(r, int64_t(i) * G, G, true);
        }
    }
    void verify_required() {
        for (int r = 0; r < R; ++r) {
            std::vector<KVBackingRange> held;
            checked(ranks[r]->kv.backing_required_ranges(ranks[0]->kv, &held), "holder inventory failed");
            for (const auto& h : held) transfer(r, h.offset, h.bytes, false);
        }
        checked(rad_dev_set(0), "restore rank0 failed");
    }
    void live_case() {
        auto& lead = ranks[0]->kv;
        need(lead.free_blocks(1) == 0, "fixture would not force a grow callback");
        checked(controller.begin_plan(), "begin grow lease failed");
        checked(lead.add_sequence(1), "sequence creation failed");
        checked(lead.ensure(1, 128), "real all-rank candidate growth failed");
        controller.finish_plan();
        const int64_t grown = ranks[0]->pools.kv().committed();
        need(grown > initial && grown < options.maximum_bytes, "no actual bounded growth"); stable(grown);
        for (int r = 0; r < R; ++r) for (int g : {1, 2})
            for (int b : lead.block_table(1, g)) need(ranks[r]->kv.block_backed(g, b), "published an unbacked ID");
        fill_mapped(); verify_required();
        const int held = lead.block_table(1, 1).front();
        checked(lead.retain_block(1, held), "prefix retain failed");
        lead.free_sequence(1);
        checked(controller.begin_plan(), "prefix-pinned shrink failed"); controller.finish_plan();
        const int64_t pinned = ranks[0]->pools.kv().committed();
        need(pinned > initial && pinned <= grown, "prefix did not pin physical backing"); stable(pinned);
        for (int r = 0; r < R; ++r) need(ranks[r]->kv.block_backed(1, held), "prefix publication was lost");
        verify_required();
        lead.release_block(1, held);
        checked(controller.begin_plan(), "released-prefix shrink failed"); controller.finish_plan();
        stable(initial); verify_required();
        std::printf("KV_BACKING_CARD live ranks=4 startup=%lld grown=%lld prefix=%lld shrunk=%lld granule=%lld bytes_exact=1 stable_va=1\n",
            (long long)initial, (long long)grown, (long long)pinned, (long long)initial, (long long)G);
    }
};

// Faults wrap actual device mappings. Returning an error after rank3 mapped a new granule
// proves rollback consults the real changed ledger, rather than assuming failure was atomic.
struct Audit;
struct Backend final : KVBackingBackend {
    Audit* audit; int rank; PoolKVBackingBackend real;
    bool fail_map = false, fail_unmap = false, fired = false;
    Backend(Audit* a, int r, Pool* p) : audit(a), rank(r), real(p, r) {}
    int read(KVBackingSnapshot* s) override { return real.read(s); }
    int commit(int64_t, int64_t) override;
    int release(int64_t, int64_t) override;
};
struct Publisher final : KVBackingPublisher {
    Audit* audit; int rank; KVManagerBackingPublisher real;
    KVBackingSnapshot old_map, target, observed;
    Publisher(Audit* a, int r, KVManager* m) : audit(a), rank(r), real(m) {}
    int stage(const KVBackingSnapshot&, const KVBackingSnapshot&) override;
    void discard() noexcept override { real.discard(); }
    void withdraw() noexcept override;
    void publish_target() noexcept override;
    void publish_old() noexcept override;
    void poison() noexcept override;
};
struct Audit final : KVBackingQuiescence, KVBackingFunding {
    Cards& cards;
    std::array<std::unique_ptr<Backend>, R> backend;
    std::array<std::unique_ptr<Publisher>, R> publisher;
    std::array<bool, R> withdrawn{}, poisoned{};
    bool held = false, bad = false, fatal = false;
    int targets = 0, old_publications = 0;
    explicit Audit(Cards& c) : cards(c) {
        for (int r = 0; r < R; ++r) {
            backend[r] = std::make_unique<Backend>(this, r, &c.ranks[r]->pools.kv());
            publisher[r] = std::make_unique<Publisher>(this, r, &c.ranks[r]->kv);
        }
    }
    bool all_withdrawn() const { return std::all_of(withdrawn.begin(), withdrawn.end(), [](bool b) { return b; }); }
    int enter() override {
        if (held || fatal) return RAD_E_STATE;
        for (int r = 0; r < R; ++r) {
            RAD_TRY(rad_dev_set(r)); RAD_TRY(rad_stream_sync(cards.ranks[r]->stream));
        }
        held = true; return RAD_OK;
    }
    void leave() noexcept override { held = false; (void)rad_dev_set(0); }
    void poison() noexcept override { fatal = true; } // shared quiescence/funding fatal latch
    int reserve(const std::vector<KVBackingSnapshot>& old, const std::vector<KVBackingPlan>& plan) override {
        if (!held || old.size() != R || plan.size() != R) return RAD_E_STATE;
        for (int r = 0; r < R; ++r) {
            RadDeviceProps p{}; RAD_TRY(rad_dev_props(r, &p));
            if (plan[r].peak_bytes > cards.options.maximum_bytes ||
                !kv_live_funding_fits(bytes(old[r]), plan[r].peak_bytes, cards.options.maximum_bytes,
                                     p.vram_free, 64 * kKVLiveMiB, !plan[r].commit.empty())) return RAD_E_NOMEM;
        }
        return RAD_OK;
    }
    void committed() noexcept override {}
    void rolled_back() noexcept override {}
    void changed() noexcept {
        if (!held || !all_withdrawn()) bad = true;
        for (const auto& r : cards.ranks) if (r->kv.free_blocks(1) || r->kv.free_blocks(2)) bad = true;
    }
    void all_maps_equal(bool old) noexcept {
        // Reuse preallocated target-size maps; the production callback remains a swap.
        for (int r = 0; r < R; ++r) {
            auto& observed = publisher[r]->observed;
            const auto& expected = old ? publisher[r]->old_map : publisher[r]->target;
            if (backend[r]->read(&observed) < 0 || observed.backed != expected.backed) bad = true;
        }
    }
    KVBackingTransactionReport run(int failure) {
        std::vector<KVBackingParticipant> participants(R);
        const auto base = cards.snapshot(0);
        auto absent = std::find(base.backed.begin(), base.backed.end(), uint8_t(0));
        need(absent != base.backed.end(), "failure fixture has no absent granule");
        const int64_t extra = int64_t(absent - base.backed.begin()) * cards.G;
        for (int r = 0; r < R; ++r) {
            auto& p = participants[r]; p.backend = backend[r].get(); p.publisher = publisher[r].get();
            p.limits = {base.reserved_bytes, cards.G, cards.initial, base.reserved_bytes, 0, 0};
            p.inventory = [this, r, extra](std::vector<KVBackingRange>* out) {
                if (!held) return int(RAD_E_STATE);
                const int rc = cards.ranks[r]->kv.backing_required_ranges(cards.ranks[0]->kv, out);
                if (rc >= 0) out->push_back({extra, 1});
                return rc;
            };
        }
        backend[3]->fail_map = failure != 0; backend[3]->fail_unmap = failure == 2;
        return kv_backing_transact(participants, this, this);
    }
};
int Backend::commit(int64_t off, int64_t n) {
    audit->changed(); const int rc = real.commit(off, n);
    if (rc >= 0 && fail_map) { fail_map = false; fired = true; return RAD_E_DEVICE; }
    return rc;
}
int Backend::release(int64_t off, int64_t n) {
    audit->changed();
    if (fail_unmap) { fail_unmap = false; return RAD_E_DEVICE; }
    return real.release(off, n);
}
int Publisher::stage(const KVBackingSnapshot& old, const KVBackingSnapshot& next) {
    if (!audit->held) audit->bad = true;
    old_map = old; target = next; observed = next; return real.stage(old, next);
}
void Publisher::withdraw() noexcept { real.withdraw(); audit->withdrawn[rank] = true; }
void Publisher::publish_target() noexcept {
    if (!audit->held) audit->bad = true;
    audit->all_maps_equal(false); ++audit->targets; real.publish_target();
}
void Publisher::publish_old() noexcept {
    if (!audit->held) audit->bad = true;
    audit->all_maps_equal(true); ++audit->old_publications; real.publish_old();
}
void Publisher::poison() noexcept { audit->poisoned[rank] = true; real.poison(); }
void failures(Cards& cards) {
    {
        Audit a(cards); const auto report = a.run(0);
        need(report.result == KVBackingTransactionResult::Committed && report.status >= 0,
             "audited real all-rank map did not commit");
        need(!a.bad && !a.fatal && a.targets == R && a.old_publications == 0,
             "publication preceded all-rank actual maps or did not reach every rank");
        // This success fixture may keep a different optional spare, so record its actual map
        // as the rollback baseline rather than pretending a count identifies physical pages.
        cards.fill_mapped(); cards.verify_required();
    }
    std::array<KVBackingSnapshot, R> before;
    for (int r = 0; r < R; ++r) before[r] = cards.snapshot(r);
    {
        Audit a(cards); const auto report = a.run(1);
        need(report.result == KVBackingTransactionResult::RolledBack && report.status < 0 && a.backend[3]->fired,
             "late real mapped-error did not roll back");
        need(!a.bad && !a.fatal && a.targets == 0 && a.old_publications == R, "failure published partial target or escaped withdrawal");
        for (int r = 0; r < R; ++r) {
            need(cards.snapshot(r).backed == before[r].backed, "actual old physical maps were not restored");
            need(cards.ranks[r]->kv.backing_error() >= 0, "repaired transaction poisoned publication unexpectedly");
        }
        cards.stable(bytes(before[0])); cards.verify_required();
    }
    {
        Audit a(cards); const auto report = a.run(2);
        need(report.result == KVBackingTransactionResult::Poisoned && report.status < 0 && a.backend[3]->fired,
             "failed real rollback-unmap did not poison");
        need(!a.bad && a.fatal && a.targets == 0 && a.old_publications == 0, "unsafe failed rollback republished IDs");
        for (int r = 0; r < R; ++r) {
            need(a.poisoned[r] && cards.ranks[r]->kv.backing_error() < 0, "poison did not close every rank");
            need(!cards.ranks[r]->kv.free_blocks(1) && !cards.ranks[r]->kv.free_blocks(2), "poison left an allocatable heap");
        }
        need(cards.snapshot(3).backed != before[3].backed, "unmap-failure test did not retain a truthful real map");
        // Deliberately no read of failed mappings; Pool destruction releases real retained VA.
    }
}
} // namespace
int main(int argc, char** argv) {
    try {
        need(argc == 2 && !std::strcmp(argv[1], "--inside-guarded-window"), "requires --inside-guarded-window before any device access");
        Cards cards; cards.build(); cards.live_case(); failures(cards);
        std::printf("PASS: DS41_KV_BACKING_CARD ranks=4 startup_min=1 grow=1 prefix_pin=1 shrink=1 stable_va=1 bytes_exact=1 late_map_rollback=1 rollback_unmap_poison=1\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL DS41_KV_BACKING_CARD: %s\n", e.what()); return 1;
    }
}
