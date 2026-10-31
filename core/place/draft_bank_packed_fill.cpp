#include "draft_bank_packed_fill.h"
#include <limits>

namespace rad::draft_bank_packed_fill {
namespace {
bool span(const void* p, size_t n) noexcept {
    return p && n && n <= UINTPTR_MAX - uintptr_t(p);
}
bool apart(const void* a, size_t an, const void* b, size_t bn) noexcept {
    return uintptr_t(a) + an <= uintptr_t(b) || uintptr_t(b) + bn <= uintptr_t(a);
}
bool weight(const WeightInfo& w, int layer, int expert, int slot) noexcept {
    const auto& d = w.decl;
    return d.group.layer == layer && d.group.expert == expert && d.group.slot == slot &&
        d.access == RAD_ACCESS_CONDITIONAL && d.shard == RAD_SHARD_NONE && d.rank == 2 &&
        d.shape[0] == (slot ? 5120 : 4608) && d.shape[1] == (slot ? 2304 : 5120) &&
        w.stored_bytes == (slot ? 6266880 : 12533760) && w.layout_tag == "ggml_mxfp4" &&
        w.stored_dtype == RAD_U8 && w.stored_rank == 2 &&
        w.stored_shape[0] == d.shape[0] && w.stored_shape[1] == d.shape[1] / 32 * 17 &&
        !w.widen && !w.narrow && !w.vram_bytes && w.vram_pk_cap == -1 && w.file_id == 0;
}
} // namespace
int prepare(const Program& p, const Plan& plan, int32_t u, const Mover::HostUnit& h,
            uint8_t* dst, size_t bytes, Workspace* work, EpochCheck epochs,
            void* epoch_context, Inverse inverse) noexcept {
    if (!work || !span(work, sizeof(*work)) || !span(dst, bytes) ||
        !apart(work, sizeof(*work), dst, bytes)) return RAD_E_INVAL;
    for (int k = 0; k < 2; ++k)
        if (h.slot[k] > 0 && uint64_t(h.slot[k]) <= uint64_t(SIZE_MAX) &&
            span(h.host[k], size_t(h.slot[k])) &&
            !apart(work, sizeof(*work), h.host[k], size_t(h.slot[k]))) return RAD_E_INVAL;
    *work = Workspace{};
    auto no = [&](Refusal why) { work->refusal = why; return RAD_E_STATE; };
    if (!epochs || !epochs(epoch_context)) return no(Refusal::Epoch);
    if (!inverse || u < 0 || size_t(u) >= plan.units.size() || h.unit != u || h.n != 2 || !h.packed)
        return no(Refusal::Unit);
    const MoveUnit& m = plan.units[size_t(u)];
    if (m.layer < 0 || m.layer >= kDraftFirstLayer || m.expert < 0 || m.expert >= 96 ||
        m.tier != Tier::SSD || m.slab_slot >= 0 || m.slab_class < 0 || m.draft_head || m.movable || m.host_forced ||
        m.draft_vram || m.draft_parked || m.vram_bytes || !m.vram_offset.empty() ||
        m.weights.size() != 2 || m.offset.size() != 2 || m.weights[0] == m.weights[1] ||
        m.bytes != kDraftPairBytes || m.slot_bytes != kDraftPairBytes ||
        m.offset[0] != 0 || m.offset[1] != 12533760 || bytes != size_t(kDraftPairBytes))
        return no(Refusal::Geometry);
    for (int k = 0; k < 2; ++k) {
        const auto wi = m.weights[size_t(k)];
        if (!wi || size_t(wi) >= p.weights.size() || !weight(p.weights[wi], m.layer, m.expert, k))
            return no(Refusal::Geometry);
        const auto& l = h.lease[k];
        if (!l.active() || !l.packed() || h.slot[k] <= 0 || h.slot[k] % 64 ||
            uint64_t(h.slot[k]) > uint64_t(SIZE_MAX) || l.bytes() != size_t(h.slot[k]) ||
            h.host[k] != l.host() || !h.host[k] ||
            h.dev[k] != (l.device() ? l.device() : static_cast<const void*>(l.host())) ||
            h.bytes[k] != p.weights[wi].stored_bytes || !span(h.host[k], size_t(h.slot[k])))
            return no(Refusal::Lease);
        if (!apart(h.host[k], size_t(h.slot[k]), dst, bytes) ||
            !apart(h.host[k], size_t(h.slot[k]), work, sizeof(*work))) return no(Refusal::Alias);
    }
    if (!apart(h.host[0], size_t(h.slot[0]), h.host[1], size_t(h.slot[1]))) return no(Refusal::Alias);
    if (p.weights[m.weights[0]].source_bank_generation != p.weights[m.weights[1]].source_bank_generation)
        return no(Refusal::Epoch);
    /* Both source leases/metadata are validated before either conversion. The
     * inverse separately checks each complete packed header and exception list. */
    for (int k = 0; k < 2; ++k) {
        const auto& w = p.weights[m.weights[size_t(k)]];
        if (inverse(h.host[k], size_t(h.slot[k]), int(w.decl.shape[0]), int(w.decl.shape[1]),
                    dst + m.offset[size_t(k)], size_t(w.stored_bytes), &work->inverse[k]) < 0)
            return no(Refusal::Inverse);
        work->source_bytes += uint64_t(h.slot[k]);
        work->raw_bytes += uint64_t(w.stored_bytes);
    }
    if (!epochs(epoch_context)) return no(Refusal::Epoch);
    return RAD_OK;
}
} // namespace rad::draft_bank_packed_fill
