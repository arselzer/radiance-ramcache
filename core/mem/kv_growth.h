/* kv_growth.h -- physical backing policy for a fixed-address KV reservation.
 *
 * Source-only prerequisite: no engine caller enables this policy yet. The owner must supply
 * EVERY referenced sequence/prefix/COW range and fixed window/state range, plus the next step's
 * writes. Plans are applied only after readers, copies and expert loans have drained, on every
 * TP rank before publishing any newly backed block. Logical capacity and addresses never move.
 * The joint KV/expert ledger must fund mappings; this policy does not change expert residency.
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace rad {

struct KVBackingRange {
    int64_t offset = 0;
    int64_t bytes = 0;
};

struct KVBackingLimits {
    int64_t reserved_bytes = 0;       // fixed virtual reservation, whole granules
    int64_t granule_bytes = 0;        // actual backend mapping granule
    int64_t minimum_bytes = 0;        // rounded UP; retain this much physical backing
    int64_t maximum_bytes = 0;        // rounded DOWN; never exceed this physical budget
    int64_t warm_bytes = 0;           // spare mapped room above all mandatory granules
    int64_t shrink_hysteresis = 0;    // defer trimming until this much excess accumulates
};

enum class KVBackingResult { Ok, Invalid, BudgetExceeded };

struct KVBackingPlan {
    // Complete desired mapping, useful for rank-local layout audits and failure bookkeeping.
    std::vector<uint8_t> target;
    // Only unreferenced granules occur here. Apply this first if add-before-drop exceeds max.
    std::vector<KVBackingRange> release_first;
    std::vector<KVBackingRange> commit;
    std::vector<KVBackingRange> release_after;
    int64_t required_bytes = 0;
    int64_t target_bytes = 0;
    int64_t peak_bytes = 0;
};

/* No mutation on refusal, no backend/device calls, no inference from currently free VRAM.
 * `backed` is the ACTUAL mapping, not a logical pool size. Bits must be exactly0 or1. The
 * returned release ranges are safe only if `required` really includes every outstanding holder.
 * A failed mapping transaction must leave actual-map bookkeeping truthful and must not publish
 * new free-list entries. Cross-rank commit/rollback and admission belong to the runtime owner.
 */
inline KVBackingResult kv_backing_plan(const KVBackingLimits& lim,
                                       const std::vector<uint8_t>& backed,
                                       const std::vector<KVBackingRange>& required,
                                       KVBackingPlan* out) {
    if (!out || lim.granule_bytes <= 0 || lim.reserved_bytes <= 0 ||
        lim.reserved_bytes % lim.granule_bytes || lim.minimum_bytes < 0 ||
        lim.maximum_bytes < 0 || lim.maximum_bytes > lim.reserved_bytes ||
        lim.minimum_bytes > lim.maximum_bytes || lim.warm_bytes < 0 ||
        lim.shrink_hysteresis < 0)
        return KVBackingResult::Invalid;
    const int64_t G = lim.granule_bytes;
    const int64_t N = lim.reserved_bytes / G;
    if (uint64_t(N) > uint64_t(std::numeric_limits<size_t>::max()) ||
        backed.size() != size_t(N)) return KVBackingResult::Invalid;
    // Quotient/remainder avoids x+G-1 overflow for valid int64 byte budgets.
    const auto ceil_g = [G](int64_t x) { return x / G + (x % G != 0); };
    const int64_t min_g = ceil_g(lim.minimum_bytes), max_g = lim.maximum_bytes / G;
    if (min_g > max_g) return KVBackingResult::BudgetExceeded;
    int64_t old_g = 0;
    for (uint8_t b : backed) {
        if (b > 1) return KVBackingResult::Invalid;
        old_g += b;
    }
    KVBackingPlan p;
    p.target.assign(size_t(N), 0);
    for (const auto& r : required) {
        if (r.offset < 0 || r.bytes < 0 || r.offset > lim.reserved_bytes ||
            r.bytes > lim.reserved_bytes - r.offset) return KVBackingResult::Invalid;
        if (!r.bytes) continue;
        const int64_t lo = r.offset / G, hi = ceil_g(r.offset + r.bytes);
        for (int64_t i = lo; i < hi; ++i) p.target[size_t(i)] = 1;
    }
    int64_t need_g = 0;
    bool missing = false;
    for (size_t i = 0; i < p.target.size(); ++i) {
        need_g += p.target[i];
        missing |= p.target[i] && !backed[i];
    }
    if (need_g > max_g) return KVBackingResult::BudgetExceeded;
    // Clamp optional warm room; mandatory pages are never clipped to fit a budget.
    const int64_t warm_g = std::min(ceil_g(lim.warm_bytes), max_g - need_g);
    int64_t target_g = std::max(min_g, need_g + warm_g);
    const int64_t hysteresis_g = ceil_g(lim.shrink_hysteresis);
    const bool keep_old = !missing && old_g <= max_g && old_g >= target_g &&
                          old_g - target_g <= hysteresis_g;
    if (keep_old) {
        p.target = backed;
        target_g = old_g;
    } else {
        int64_t got = need_g;
        // Keep already-backed spare granules first, avoiding a release/remap just for ordering.
        for (size_t i = 0; i < p.target.size() && got < target_g; ++i)
            if (backed[i] && !p.target[i]) { p.target[i] = 1; ++got; }
        for (size_t i = 0; i < p.target.size() && got < target_g; ++i)
            if (!p.target[i]) { p.target[i] = 1; ++got; }
    }
    std::vector<uint8_t> add(size_t(N), 0), drop(size_t(N), 0), first(size_t(N), 0);
    int64_t add_g = 0, drop_g = 0;
    for (size_t i = 0; i < p.target.size(); ++i) {
        add[i] = p.target[i] && !backed[i]; add_g += add[i];
        drop[i] = backed[i] && !p.target[i]; drop_g += drop[i];
    }
    // old_g+add_g can overflow at extreme valid sizes; subtract before adding.
    int64_t early_g = old_g > max_g ? old_g - max_g + add_g
        : add_g > max_g - old_g ? add_g - (max_g - old_g) : 0;
    if (early_g > drop_g) return KVBackingResult::BudgetExceeded;
    const int64_t release_g = early_g;
    for (size_t i = drop.size(); i-- > 0 && early_g > 0;)
        if (drop[i]) { first[i] = 1; drop[i] = 0; --early_g; }
    const auto ranges = [G](const std::vector<uint8_t>& bits) {
        std::vector<KVBackingRange> rows;
        size_t i = 0;
        while (i < bits.size()) {
            if (!bits[i]) { ++i; continue; }
            const size_t lo = i++;
            while (i < bits.size() && bits[i]) ++i;
            rows.push_back({int64_t(lo) * G, int64_t(i - lo) * G});
        }
        return rows;
    };
    p.release_first = ranges(first);
    p.commit = ranges(add);
    p.release_after = ranges(drop);
    p.required_bytes = need_g * G;
    p.target_bytes = target_g * G;
    p.peak_bytes = (old_g - release_g + add_g) * G;
    *out = std::move(p);
    return KVBackingResult::Ok;
}

} // namespace rad
