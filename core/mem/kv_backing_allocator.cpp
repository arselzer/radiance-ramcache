#include "mem/kv.h"
#include <algorithm>
#include <exception>
#include <functional>

namespace rad {
namespace {
bool valid(const KVBackingSnapshot& s) {
    return s.granule_bytes > 0 && s.reserved_bytes > 0 && s.reserved_bytes % s.granule_bytes == 0 &&
           uint64_t(s.reserved_bytes / s.granule_bytes) == uint64_t(s.backed.size()) &&
           std::all_of(s.backed.begin(), s.backed.end(), [](uint8_t b) { return b <= 1; });
}
bool mapped(const KVBackingSnapshot& s, int64_t off, int64_t bytes) {
    if (off < 0 || bytes < 0 || off > s.reserved_bytes || bytes > s.reserved_bytes - off) return false;
    if (!bytes) return true;
    const int64_t end = (off + bytes - 1) / s.granule_bytes;
    for (int64_t i = off / s.granule_bytes; i <= end; ++i) if (!s.backed[(size_t)i]) return false;
    return true;
}
} // namespace
int KVManager::enable_backing_mode() {
    if (configured_ || plans_.empty() || loan_ != 0) return RAD_E_STATE;
    for (const auto& p : plans_) if (p.streamed()) return RAD_E_UNSUPPORTED;
    backing_enabled_ = true;
    backing_withdrawn_ = true;
    backing_fixed_ready_ = false;
    backing_error_ = RAD_OK;
    return RAD_OK;
}
bool KVManager::block_backed(int32_t g, int32_t b) const {
    if (g < 0 || (size_t)g >= plans_.size() || b < 0 || b >= plans_[(size_t)g].n_blocks) return false;
    return !backing_enabled_ || (!backing_withdrawn_ && backing_error_ >= 0 && (size_t)g < backed_ids_.size() &&
           (size_t)b < backed_ids_[(size_t)g].size() && backed_ids_[(size_t)g][(size_t)b]);
}
void KVManager::backing_poison(int status) noexcept {
    backing_error_ = status < 0 ? status : int(RAD_E_STATE);
    backing_withdrawn_ = true;
}
int KVManager::backing_ready() {
    if (!backing_enabled_) return RAD_OK;
    if (backing_error_ < 0) return backing_error_;
    if (backing_withdrawn_ || !backing_fixed_ready_) {
        backing_poison(RAD_E_STATE); return backing_error_;
    }
    return RAD_OK;
}
int KVManager::backing_block_ranges(const std::vector<KVBackingBlock>& ids,
                                    std::vector<KVBackingRange>* out) const {
    if (!configured_ || !out || !pool_) return RAD_E_STATE;
    std::vector<KVBackingRange> ranges;
    for (const auto& id : ids) {
        const auto* p = plan(id.group);
        if (!p || !p->paged() || id.block < 0 || id.block >= p->n_blocks) return RAD_E_INVAL;
        if (p->zero) continue;
        const int64_t frag = p->layer_bytes_per_block();
        if (frag <= 0 || p->n_layers <= 0) return RAD_E_STATE;
        for (int64_t l = 0; l < p->n_layers; ++l) {
            const int64_t off = p->pool_offset + l * p->layer_stride() + int64_t(id.block) * frag;
            if (off < 0 || off > pool_->bytes() || frag > pool_->bytes() - off) return RAD_E_STATE;
            ranges.push_back({off, frag});
        }
    }
    *out = std::move(ranges);
    return RAD_OK;
}
int KVManager::backing_stage(const KVBackingSnapshot& old, const KVBackingSnapshot& target) {
    if (!backing_enabled_ || !configured_ || backing_error_ < 0 || backing_staged_ ||
        !valid(old) || !valid(target) || old.reserved_bytes != target.reserved_bytes ||
        old.granule_bytes != target.granule_bytes || !pool_ ||
        old.reserved_bytes != pool_->backing_reserved_bytes() ||
        (pool_->backing_granule_bytes() > 0 && old.granule_bytes != pool_->backing_granule_bytes()))
        return RAD_E_STATE;
    const auto build = [&](const KVBackingSnapshot& map, BackingPublication* out) -> int {
        out->mask.resize(plans_.size()); out->heap.resize(plans_.size());
        out->usable.assign(plans_.size(), 0); out->fixed_ready = true;
        for (size_t g = 0; g < plans_.size(); ++g) {
            const auto& p = plans_[g];
            if (!p.zero && p.kind != RAD_KV_FULL && p.pool_bytes > 0 &&
                !mapped(map, p.pool_offset, p.pool_bytes)) out->fixed_ready = false;
            if (!p.paged()) continue;
            out->mask[g].assign((size_t)p.n_blocks, 0);
            out->heap[g].reserve((size_t)p.n_blocks);
            for (int64_t b = 0; b < p.n_blocks; ++b) {
                bool backed = true;
                if (!p.zero) {
                    const int64_t frag = p.layer_bytes_per_block();
                    if (frag <= 0 || p.n_layers <= 0) return RAD_E_STATE;
                    for (int64_t l = 0; l < p.n_layers && backed; ++l)
                        backed = mapped(map, p.pool_offset + l * p.layer_stride() + b * frag, frag);
                }
                out->mask[g][(size_t)b] = backed;
                if (!backed && refcount_[g][(size_t)b] > 0) return RAD_E_STATE;
                if (b < live_[g] && refcount_[g][(size_t)b] == 0) {
                    // Private candidates retain the legacy global minimum-free-ID order.
                    // Eligibility is still exclusively the separate backed mask/count: an
                    // unbacked candidate must pass backing_need before grab can return it.
                    out->heap[g].push_back((int32_t)b);
                    if (backed) ++out->usable[g];
                }
            }
        }
        /* Sources whose final logical reference was dropped remain readers until COW copies
         * drain. The engine lease separately covers copies already taken from pending_. */
        for (const auto& copy : pending_)
            for (int32_t b : {copy.src_block, copy.dst_block})
                if (copy.group < 0 || (size_t)copy.group >= out->mask.size() || b < 0 ||
                    (size_t)b >= out->mask[(size_t)copy.group].size() || !out->mask[(size_t)copy.group][(size_t)b])
                    return RAD_E_STATE;
        if (!out->fixed_ready) {
            for (const auto& s : seqs_) for (int32_t slot : s.second.state) if (slot >= 0) return RAD_E_STATE;
            if (!clears_.empty()) return RAD_E_STATE;
        }
        return RAD_OK;
    };
    try {
        BackingPublication a, b;
        int rc = build(old, &a); if (rc < 0) return rc;
        rc = build(target, &b); if (rc < 0) return rc;
        if (!b.fixed_ready) return RAD_E_STATE;
        for (const auto& id : backing_intent_)
            if (id.group < 0 || (size_t)id.group >= b.mask.size() || id.block < 0 ||
                (size_t)id.block >= b.mask[(size_t)id.group].size() || !b.mask[(size_t)id.group][(size_t)id.block])
                return RAD_E_STATE;
        backing_old_ = std::move(a); backing_target_ = std::move(b);
        backing_staged_ = true;
        return RAD_OK;
    } catch (const std::exception&) { return RAD_E_NOMEM; }
}
void KVManager::backing_discard() noexcept {
    backing_staged_ = false;
    backing_old_ = {}; backing_target_ = {};
}
void KVManager::backing_publish(BackingPublication& p) noexcept {
    if (!backing_staged_ || !backing_withdrawn_ || backing_error_ < 0) { backing_poison(RAD_E_STATE); return; }
    backed_ids_.swap(p.mask); free_blocks_.swap(p.heap); free_usable_.swap(p.usable);
    backing_fixed_ready_ = p.fixed_ready;
    backing_withdrawn_ = false; backing_staged_ = false;
}
int KVManager::backing_need_ids(const std::vector<KVBackingBlock>& ids) {
    RAD_TRY(backing_ready());
    bool missing = false;
    for (const auto& id : ids) {
        const auto* p = plan(id.group);
        if (!p || !p->paged() || id.block < 0 || id.block >= p->n_blocks) return RAD_E_INVAL;
        missing |= !block_backed(id.group, id.block);
    }
    if (backing_collect_intent_) backing_intent_.insert(backing_intent_.end(), ids.begin(), ids.end());
    if (!missing) return RAD_OK;
    if (!backing_grow_) { backing_poison(RAD_E_STATE); return backing_error_; }
    const KVBackingGrowResult result = backing_grow_(ids);
    if (result.fatal) { backing_poison(result.status); return backing_error_; }
    if (backing_error_ < 0) return backing_error_;
    if (result.status < 0) return result.status;
    for (const auto& id : ids) if (!block_backed(id.group, id.block)) {
        backing_poison(RAD_E_STATE); return backing_error_;
    }
    return backing_ready();
}
int KVManager::backing_need(int32_t g, int64_t free) {
    RAD_TRY(backing_ready());
    if (free <= 0) return RAD_OK;
    if (g < 0 || (size_t)g >= plans_.size() || !plans_[(size_t)g].paged()) return RAD_E_INVAL;
    if (free > available_blocks(g)) return RAD_E_FULL;
    const auto& p = plans_[(size_t)g];
    auto& heap = free_blocks_[(size_t)g];
    // reserve_specific leaves a stale heap entry, just as on the legacy path. Remove only
    // stale held entries at the front; never search the whole refcount array for candidates.
    const auto valid_id = [&](int32_t b) { return b >= 0 && b < p.n_blocks && b < live_[(size_t)g]; };
    while (!heap.empty()) {
        const int32_t b = heap.front();
        if (!valid_id(b)) { backing_poison(RAD_E_STATE); return backing_error_; }
        if (refcount_[(size_t)g][(size_t)b] == 0) break;
        std::pop_heap(heap.begin(), heap.end(), std::greater<int32_t>()); heap.pop_back();
    }
    if (!backing_collect_intent_ && free == 1 && !heap.empty() && block_backed(g, heap.front()))
        return RAD_OK;
    std::vector<KVBackingBlock> ids;
    // Allocate the existing N-candidate workspace before removing anything. Restoring these
    // trivial entries cannot grow the heap beyond its original size/capacity. A growth callback
    // can swap the entire publication, so no heap reference is used after backing_need_ids.
    try { ids.reserve((size_t)free); }
    catch (const std::exception&) { return RAD_E_NOMEM; }
    bool invalid = false;
    while (!heap.empty() && (int64_t)ids.size() < free) {
        std::pop_heap(heap.begin(), heap.end(), std::greater<int32_t>());
        const int32_t b = heap.back(); heap.pop_back();
        if (!valid_id(b)) { invalid = true; break; }
        if (refcount_[(size_t)g][(size_t)b] != 0) continue;
        // A specific reservation followed by release can leave duplicate stale entries.
        // Extraction is sorted, so distinct candidates need no second heap or bitmap.
        if (ids.empty() || ids.back().block != b) ids.push_back({g, b});
    }
    for (const auto& id : ids) {
        heap.push_back(id.block); std::push_heap(heap.begin(), heap.end(), std::greater<int32_t>());
    }
    if (invalid || (int64_t)ids.size() != free) { backing_poison(RAD_E_STATE); return backing_error_; }
    RAD_TRY(backing_need_ids(ids));
    if (free_usable_[(size_t)g] < free) { backing_poison(RAD_E_STATE); return backing_error_; }
    return RAD_OK;
}
int KVManagerBackingPublisher::stage(const KVBackingSnapshot& a, const KVBackingSnapshot& b) {
    return manager_ ? manager_->backing_stage(a, b) : int(RAD_E_INVAL);
}
void KVManagerBackingPublisher::discard() noexcept { if (manager_) manager_->backing_discard(); }
void KVManagerBackingPublisher::withdraw() noexcept { if (manager_) manager_->backing_withdrawn_ = true; }
void KVManagerBackingPublisher::publish_target() noexcept { if (manager_) manager_->backing_publish(manager_->backing_target_); }
void KVManagerBackingPublisher::publish_old() noexcept { if (manager_) manager_->backing_publish(manager_->backing_old_); }
void KVManagerBackingPublisher::poison() noexcept { if (manager_) manager_->backing_poison(RAD_E_DEVICE); }
} // namespace rad
