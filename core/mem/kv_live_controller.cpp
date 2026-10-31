#include "mem/kv_live_controller.h"
#include "mem/kv.h"
#include "mem/pools.h"
#include "mem/kv_backing_transaction.h"
#include "device/device.h"
#include "rad_internal.h"

#include <algorithm>
#include <limits>

namespace rad {
namespace {
int64_t physical_bytes(const KVBackingSnapshot& s) {
    int64_t n = 0;
    for (uint8_t b : s.backed) n += b != 0;
    return n * s.granule_bytes;
}
bool same_groups(const Program& a, const Program& b) {
    if (a.kv_groups.size() != b.kv_groups.size()) return false;
    for (size_t g = 1; g < a.kv_groups.size(); ++g) {
        const auto& x = a.kv_groups[g];
        const auto& y = b.kv_groups[g];
        const auto& xd = x.decl;
        const auto& yd = y.decl;
        if (x.layers != y.layers || x.zero || y.zero || xd.kind != yd.kind || xd.dtype != yd.dtype ||
            xd.window != yd.window || xd.n_head_kv != yd.n_head_kv || xd.head_dim != yd.head_dim ||
            xd.state_dim[0] != yd.state_dim[0] || xd.state_dim[1] != yd.state_dim[1] ||
            xd.conv_width != yd.conv_width || xd.storage_format != yd.storage_format ||
            xd.storage_block_bytes != yd.storage_block_bytes) return false;
    }
    return true;
}
bool same_plans(const KVManager& a, const KVManager& b) {
    if (a.plans().size() != b.plans().size()) return false;
    for (size_t g = 1; g < a.plans().size(); ++g) {
        const auto& x = a.plans()[g];
        const auto& y = b.plans()[g];
        if (x.kind != y.kind || x.n_layers != y.n_layers || x.block_size != y.block_size ||
            x.n_blocks != y.n_blocks || x.n_states != y.n_states || x.bytes_per_block != y.bytes_per_block ||
            x.bytes_per_state != y.bytes_per_state || x.storage_format != y.storage_format ||
            x.pool_offset != y.pool_offset || x.pool_bytes != y.pool_bytes ||
            x.layer_stride() != y.layer_stride() || x.zero || y.zero || x.streamed() || y.streamed()) return false;
    }
    return true;
}
}  // namespace

struct KVLiveController::Impl {
    KVLiveOptions options;
    std::vector<Rank> ranks;
    std::vector<std::unique_ptr<PoolKVBackingBackend>> backends;
    std::vector<std::unique_ptr<KVManagerBackingPublisher>> publishers;
    FreeBytes free_bytes;
    int64_t startup_headroom = 0, late_startup = 0, late_runtime = 0;
    bool startup = false, planning = false, transaction = false, funded = false;
    int failure = RAD_OK;
    uint64_t epoch = 0;

    int poison(int status) noexcept {
        if (failure >= 0) failure = status < 0 ? status : RAD_E_STATE;
        for (auto& p : publishers) p->poison();
        if (!ranks.empty()) (void)rad_dev_set(ranks[0].device);
        return failure;
    }
    struct Quiet final : KVBackingQuiescence {
        Impl* p;
        explicit Quiet(Impl* owner) : p(owner) {}
        int enter() override {
            if (p->failure < 0 || p->transaction || (!p->startup && !p->planning)) return RAD_E_STATE;
            p->transaction = true;
            return RAD_OK;
        }
        void leave() noexcept override { p->transaction = false; }
        void poison() noexcept override { p->poison(RAD_E_DEVICE); }
    } quiet{this};
    struct Funding final : KVBackingFunding {
        Impl* p;
        explicit Funding(Impl* owner) : p(owner) {}
        int reserve(const std::vector<KVBackingSnapshot>& old,
                    const std::vector<KVBackingPlan>& plans) override {
            if (old.size() != p->ranks.size() || plans.size() != old.size() || p->funded)
                return RAD_E_STATE;
            for (size_t i = 0; i < old.size(); ++i) {
                int64_t free = 0;
                RAD_TRY(p->free_bytes(p->ranks[i].device, &free));
                const int64_t before = physical_bytes(old[i]);
                if (p->startup) {
                    const auto& r = p->ranks[i];
                    KVLiveBudget b{ r.pool->bytes(), old[i].reserved_bytes, r.pool->used(),
                        old[i].granule_bytes, before, free, p->startup_headroom, p->late_startup };
                    if (kv_live_budget_error(p->options, b)) return RAD_E_NOMEM;
                } else {
                    const int64_t reserve = p->options.runtime_headroom_bytes + p->late_runtime;
                    if (!kv_live_funding_fits(before, plans[i].peak_bytes, p->options.maximum_bytes,
                                              free, reserve, !plans[i].commit.empty())) return RAD_E_NOMEM;
                }
            }
            p->funded = true;
            return RAD_OK;
        }
        void committed() noexcept override { p->funded = false; }
        void rolled_back() noexcept override { p->funded = false; }
        void poison() noexcept override { p->poison(RAD_E_DEVICE); }
    } funding{this};

    int inventory(size_t i, const std::vector<KVBackingBlock>& wanted,
                   std::vector<KVBackingRange>* out) {
        RAD_TRY(ranks[i].manager->backing_required_ranges(*ranks[0].manager, out));
        std::vector<KVBackingRange> more;
        RAD_TRY(ranks[i].manager->backing_block_ranges(wanted, &more));
        out->insert(out->end(), more.begin(), more.end());
        return RAD_OK;
    }

    int resize(const std::vector<KVBackingBlock>& wanted, const char* cause) {
        try { return resize_checked(wanted, cause); }
        catch (...) { return poison(RAD_E_NOMEM); }
    }

    int resize_checked(const std::vector<KVBackingBlock>& wanted, const char* cause) {
        if (failure < 0) return failure;
        if ((!startup && !planning) || transaction) return poison(RAD_E_STATE);
        std::vector<KVBackingSnapshot> before(ranks.size());
        std::vector<KVBackingParticipant> participants(ranks.size());
        std::vector<KVBackingPlan> preview(ranks.size());
        bool changes = false;
        for (size_t i = 0; i < ranks.size(); ++i) {
            const int rc = backends[i]->read(&before[i]);
            if (rc < 0) return poison(rc);
            if (i && (before[i].granule_bytes != before[0].granule_bytes ||
                      before[i].reserved_bytes != before[0].reserved_bytes ||
                      before[i].backed != before[0].backed)) return poison(RAD_E_STATE);
            auto& part = participants[i];
            part.backend = backends[i].get();
            part.publisher = publishers[i].get();
            part.limits = { before[i].reserved_bytes, before[i].granule_bytes,
                options.minimum_bytes, options.maximum_bytes,
                startup ? 0 : options.warm_bytes, startup ? 0 : options.hysteresis_bytes };
            std::vector<KVBackingRange> required;
            const int irc = inventory(i, wanted, &required);
            if (irc < 0) return poison(irc);
            const auto planned = kv_backing_plan(part.limits, before[i].backed, required, &preview[i]);
            if (planned != KVBackingResult::Ok)
                return poison(planned == KVBackingResult::BudgetExceeded ? RAD_E_NOMEM : RAD_E_INVAL);
            if (i && preview[i].target != preview[0].target) return poison(RAD_E_STATE);
            changes |= preview[i].target != before[i].backed;
            part.inventory = [this, i, wanted](std::vector<KVBackingRange>* out) {
                return inventory(i, wanted, out); // Engine still holds scheduler ownership
            };
        }
        if (!changes && !startup) {
            for (const auto& id : wanted)
                for (const auto& r : ranks)
                    if (!r.manager->block_backed(id.group, id.block)) return poison(RAD_E_STATE);
            const int restored = rad_dev_set(ranks[0].device);
            return restored < 0 ? poison(restored) : int(RAD_OK);
        }
        auto report = kv_backing_transact(participants, &quiet, &funding);
        if (report.result != KVBackingTransactionResult::Committed || report.status < 0)
            return poison(report.status < 0 ? report.status : RAD_E_DEVICE);
        for (const auto& id : wanted)
            for (const auto& r : ranks)
                if (!r.manager->block_backed(id.group, id.block)) return poison(RAD_E_STATE);
        ++epoch;
        for (size_t i = 0; i < ranks.size(); ++i)
            RAD_INFO("DS41_KV_GROW rank=%zu epoch=%llu cause=%s logical=%lld reserved=%lld "
                     "granule=%lld old=%lld actual=%lld min=%lld max=%lld fixed_experts=1",
                     i, (unsigned long long)epoch, cause, (long long)ranks[i].pool->bytes(),
                     (long long)report.actual[i].reserved_bytes, (long long)report.actual[i].granule_bytes,
                     (long long)physical_bytes(before[i]), (long long)physical_bytes(report.actual[i]),
                     (long long)options.minimum_bytes, (long long)options.maximum_bytes);
        const int restored = rad_dev_set(ranks[0].device);
        if (restored < 0) return poison(restored);
        return RAD_OK;
    }
};

KVLiveController::KVLiveController() : p_(std::make_unique<Impl>()) {}
KVLiveController::~KVLiveController() {
    if (p_ && !p_->ranks.empty()) p_->ranks[0].manager->set_backing_grow({});
}

int KVLiveController::start(const KVLiveOptions& options, const std::vector<Rank>& ranks,
                           int64_t startup_headroom, int64_t late_startup, int64_t late_runtime,
                           FreeBytes free_bytes) {
    if (!p_->ranks.empty() || !options.enabled || ranks.empty() || startup_headroom < 0 ||
        late_startup < 0 || late_runtime < 0 ||
        options.runtime_headroom_bytes > std::numeric_limits<int64_t>::max() - late_runtime)
        return RAD_E_INVAL;
    p_->options = options;
    p_->startup_headroom = startup_headroom;
    p_->late_startup = late_startup;
    p_->late_runtime = late_runtime;
    p_->free_bytes = free_bytes ? std::move(free_bytes) : FreeBytes([](int device, int64_t* out) {
        RadDeviceProps props{};
        const int rc = rad_dev_props(device, &props);
        if (rc < 0) return rc;
        if (props.vram_free < 0) return int(RAD_E_DEVICE);
        *out = props.vram_free;
        return int(RAD_OK);
    });
    for (const auto& r : ranks)
        if (!r.pool || !r.manager || !r.program || !r.pool->elastic() ||
            !r.manager->backing_mode() || r.manager->loan_tokens() != 0) return RAD_E_UNSUPPORTED;
    p_->ranks = ranks;
    for (const auto& r : ranks) {
        if (r.pool->bytes() != ranks[0].pool->bytes() ||
            !same_groups(*r.program, *ranks[0].program) ||
            !same_plans(*r.manager, *ranks[0].manager)) return p_->poison(RAD_E_UNSUPPORTED);
        p_->backends.push_back(std::make_unique<PoolKVBackingBackend>(r.pool, r.device));
        p_->publishers.push_back(std::make_unique<KVManagerBackingPublisher>(r.manager));
    }
    for (size_t i = 0; i < ranks.size(); ++i) {
        KVBackingSnapshot s;
        RAD_TRY(p_->backends[i]->read(&s));
        int64_t free = 0;
        RAD_TRY(p_->free_bytes(ranks[i].device, &free));
        KVLiveBudget b{ ranks[i].pool->bytes(), s.reserved_bytes, ranks[i].pool->used(),
                       s.granule_bytes, physical_bytes(s), free, startup_headroom, late_startup };
        if (const char* why = kv_live_budget_error(options, b)) {
            RAD_ERR("%s refused on rank %zu: %s (logical=%lld, reserved=%lld, G=%lld, explicit max=%lld)",
                kKVLiveEnv, i, why, (long long)b.logical_pool_bytes, (long long)b.reserved_bytes,
                (long long)b.granule_bytes, (long long)options.maximum_bytes);
            return p_->poison(RAD_E_NOMEM);
        }
        if (physical_bytes(s) != 0) return p_->poison(RAD_E_STATE); // bootstrap must really defer backing
    }
    p_->startup = true;
    const int rc = p_->resize({}, "startup");
    p_->startup = false;
    if (rc < 0) return rc;
    ranks[0].manager->set_backing_grow([this](const std::vector<KVBackingBlock>& wanted) {
        const int rc = p_->resize(wanted, "grow");
        return KVBackingGrowResult{ rc, rc < 0 }; // every fatal result withdraws ALL ranks above
    });
    return RAD_OK;
}

int KVLiveController::begin_plan() {
    if (p_->failure < 0 || p_->startup || p_->planning || p_->transaction || p_->ranks.empty())
        return p_->poison(RAD_E_STATE);
    p_->planning = true;
    return p_->resize({}, "shrink");
}
void KVLiveController::finish_plan() noexcept { p_->planning = false; }
int KVLiveController::status() const { return p_->failure; }
}  // namespace rad
