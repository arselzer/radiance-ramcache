#include "mem/kv_backing_transaction.h"
#include "mem/pools.h"
#include "device/device.h"

#include <exception>
#include <limits>

namespace rad {
int PoolKVBackingBackend::read(KVBackingSnapshot* out) {
    if (!pool_ || !out) return RAD_E_INVAL;
    const int rc = rad_dev_set(device_);
    if (rc < 0) return rc;
    if (!pool_->elastic()) return RAD_E_UNSUPPORTED;
    const int64_t G = pool_->backing_granule_bytes(), bytes = pool_->backing_reserved_bytes();
    if (G <= 0 || bytes <= 0 || bytes % G ||
        uint64_t(bytes / G) > uint64_t(std::numeric_limits<size_t>::max())) return RAD_E_STATE;
    out->reserved_bytes = bytes;
    out->granule_bytes = G;
    out->backed.resize((size_t)(bytes / G));
    for (size_t i = 0; i < out->backed.size(); ++i)
        out->backed[i] = pool_->is_backed(int64_t(i) * G, G) ? 1 : 0;
    return RAD_OK;
}
int PoolKVBackingBackend::commit(int64_t off, int64_t bytes) {
    if (!pool_) return RAD_E_INVAL;
    const int rc = rad_dev_set(device_);
    if (rc < 0) return rc;
    if (bytes != pool_->backing_granule_bytes()) return RAD_E_INVAL;
    return pool_->commit_backing_granule(off);
}
int PoolKVBackingBackend::release(int64_t off, int64_t bytes) {
    if (!pool_) return RAD_E_INVAL;
    const int rc = rad_dev_set(device_);
    if (rc < 0) return rc;
    if (bytes != pool_->backing_granule_bytes()) return RAD_E_INVAL;
    return pool_->release_backing_granule(off);
}

namespace {
bool valid_snapshot(const KVBackingSnapshot& s) {
    if (s.granule_bytes <= 0 || s.reserved_bytes <= 0 || s.reserved_bytes % s.granule_bytes ||
        uint64_t(s.reserved_bytes / s.granule_bytes) != uint64_t(s.backed.size())) return false;
    for (uint8_t b : s.backed) if (b > 1) return false;
    return true;
}
bool same_layout(const KVBackingSnapshot& a, const KVBackingSnapshot& b) {
    return a.reserved_bytes == b.reserved_bytes && a.granule_bytes == b.granule_bytes &&
           a.backed.size() == b.backed.size();
}
}  // namespace

KVBackingTransactionReport kv_backing_transact(const std::vector<KVBackingParticipant>& ranks,
                                              KVBackingQuiescence* quiet,
                                              KVBackingFunding* funding) {
    KVBackingTransactionReport report;
    if (!quiet || !funding || ranks.empty()) return report;
    for (size_t i = 0; i < ranks.size(); ++i) {
        if (!ranks[i].backend || !ranks[i].publisher || !ranks[i].inventory) return report;
        for (size_t j = 0; j < i; ++j)
            if (ranks[i].backend == ranks[j].backend || ranks[i].publisher == ranks[j].publisher)
                return report;
    }
    report.status = quiet->enter();
    if (report.status < 0) return report;
    bool withdrawn = false, funded = false;
    size_t staged = 0;
    std::vector<KVBackingSnapshot> old;
    const auto discard = [&]() noexcept {
        for (size_t i = 0; i < staged; ++i) ranks[i].publisher->discard();
    };
    const auto poison = [&]() noexcept {
        for (const auto& r : ranks) r.publisher->poison();
        if (funded) funding->poison();
        quiet->poison();
        report.result = KVBackingTransactionResult::Poisoned;
    };
    const auto read = [&](size_t i) {
        report.actual_valid[i] = 0;
        const int rc = ranks[i].backend->read(&report.actual[i]);
        report.actual_valid[i] = rc >= 0 && valid_snapshot(report.actual[i]) &&
                                 same_layout(report.actual[i], old[i]);
        return report.actual_valid[i] ? int(RAD_OK) : rc < 0 ? rc : int(RAD_E_STATE);
    };
    try {
        old.resize(ranks.size());
        report.actual.resize(ranks.size());
        report.actual_valid.assign(ranks.size(), 0);
        report.plans.resize(ranks.size());
        for (size_t i = 0; i < ranks.size(); ++i) {
            int rc = ranks[i].backend->read(&old[i]);
            if (rc >= 0 && !valid_snapshot(old[i])) rc = RAD_E_STATE;
            const auto& lim = ranks[i].limits;
            if (rc >= 0 && (lim.reserved_bytes != old[i].reserved_bytes ||
                            lim.granule_bytes != old[i].granule_bytes)) rc = RAD_E_INVAL;
            if (rc >= 0) {
                std::vector<KVBackingRange> required;
                rc = ranks[i].inventory(&required);
                const auto planned = rc < 0 ? KVBackingResult::Invalid
                    : kv_backing_plan(lim, old[i].backed, required,
                                                      &report.plans[i]);
                if (rc >= 0 && planned != KVBackingResult::Ok)
                    rc = planned == KVBackingResult::BudgetExceeded ? RAD_E_NOMEM : RAD_E_INVAL;
            }
            if (rc < 0) {
                report.status = rc;
                discard();
                quiet->leave();
                return report;
            }
            report.actual[i] = old[i]; // allocate bitmap capacity BEFORE any mapping changes
            report.actual_valid[i] = 1;
            KVBackingSnapshot target = old[i];
            target.backed = report.plans[i].target;
            rc = ranks[i].publisher->stage(old[i], target);
            if (rc < 0) {
                ranks[i].publisher->discard();
                report.status = rc;
                discard();
                quiet->leave();
                return report;
            }
            ++staged;
        }
        report.status = funding->reserve(old, report.plans);
        if (report.status < 0) { discard(); quiet->leave(); return report; }
        funded = true;
        for (const auto& r : ranks) r.publisher->withdraw();
        withdrawn = true;

        /* One granule per call: VMem can partially complete a larger call. Always re-read the
         * actual ledger, including after failure. Publication stays empty on ALL ranks. */
        const auto apply = [&](size_t i, const std::vector<KVBackingRange>& ranges, bool add) {
            const int64_t G = old[i].granule_bytes;
            for (const auto& range : ranges)
                for (int64_t off = range.offset; off < range.offset + range.bytes; off += G) {
                    const int rc = add ? ranks[i].backend->commit(off, G)
                                       : ranks[i].backend->release(off, G);
                    const int observed = read(i);
                    if (rc < 0) return rc; // mapped != accessible after a failed commit
                    if (observed < 0) return observed;
                    if (report.actual[i].backed[(size_t)(off / G)] != uint8_t(add))
                        return int(RAD_E_STATE);
                }
            return int(RAD_OK);
        };
        int rc = RAD_OK;
        for (int phase = 0; phase < 3 && rc >= 0; ++phase)
            for (size_t i = 0; i < ranks.size() && rc >= 0; ++i) {
                const auto& p = report.plans[i];
                rc = apply(i, phase == 0 ? p.release_first : phase == 1 ? p.commit : p.release_after,
                           phase == 1);
            }
        if (rc >= 0)
            for (size_t i = 0; i < ranks.size() && rc >= 0; ++i) {
                rc = read(i);
                if (rc >= 0 && report.actual[i].backed != report.plans[i].target) rc = RAD_E_STATE;
            }
        if (rc >= 0) {
            for (const auto& r : ranks) r.publisher->publish_target();
            funding->committed();
            quiet->leave();
            report.result = KVBackingTransactionResult::Committed;
            report.status = RAD_OK;
            return report;
        }

        report.status = rc;
        bool repaired = true;
        // Remove EVERY new extent, including an inaccessible mapping left by a failed commit.
        for (size_t i = 0; i < ranks.size(); ++i) {
            if (read(i) < 0) { repaired = false; continue; }
            const int64_t G = old[i].granule_bytes;
            for (size_t b = 0; b < old[i].backed.size(); ++b) {
                if (old[i].backed[b] || !report.actual[i].backed[b]) continue;
                const int urc = ranks[i].backend->release(int64_t(b) * G, G);
                const int observed = read(i);
                // A failed unmap is fatal even if the physical ledger later looks unchanged.
                if (urc < 0 || observed < 0 || report.actual[i].backed[b]) repaired = false;
            }
        }
        // Do not add old mappings beside a new extent rollback could not release: that can
        // exceed the funded peak. Leave every publisher withdrawn and stop inference instead.
        if (!repaired) { poison(); return report; }
        // Early release touched only unheld data; restore its old backing, not its stale bytes.
        for (size_t i = 0; i < ranks.size(); ++i) {
            if (read(i) < 0) { repaired = false; continue; }
            const int64_t G = old[i].granule_bytes;
            for (size_t b = 0; b < old[i].backed.size(); ++b) {
                if (!old[i].backed[b] || report.actual[i].backed[b]) continue;
                const int crc = ranks[i].backend->commit(int64_t(b) * G, G);
                const int observed = read(i);
                if (crc < 0 || observed < 0 || !report.actual[i].backed[b]) repaired = false;
            }
        }
        for (size_t i = 0; i < ranks.size(); ++i)
            if (read(i) < 0 || report.actual[i].backed != old[i].backed) repaired = false;
        if (!repaired) { poison(); return report; }
        for (const auto& r : ranks) r.publisher->publish_old();
        funding->rolled_back();
        quiet->leave();
        report.result = KVBackingTransactionResult::RolledBack;
        return report;
    } catch (const std::exception&) {
        report.status = RAD_E_NOMEM;
        // A throwing backend/publisher violates the post-withdraw no-allocation contract. Never
        // resume from an unknown partial mutation; retained actual snapshots are diagnostic only.
        if (withdrawn) {
            std::fill(report.actual_valid.begin(), report.actual_valid.end(), uint8_t(0));
            poison();
            return report;
        }
        discard();
        if (funded) funding->rolled_back();
        quiet->leave();
        return report;
    } catch (...) {
        report.status = RAD_E_STATE;
        if (withdrawn) {
            std::fill(report.actual_valid.begin(), report.actual_valid.end(), uint8_t(0));
            poison();
            return report;
        }
        discard();
        if (funded) funding->rolled_back();
        quiet->leave();
        return report;
    }
}
}  // namespace rad
