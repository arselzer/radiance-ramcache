#include "draft_bank_backup.h"
#include "draft_vram_load.h"
#include "dspark_native_scope.h"
#include "mover.h"
#include "planner.h"
#include "../rad_internal.h"
#include "../engine_hostmem.h"
#include "rad_format.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>

namespace rad {
namespace {
constexpr int64_t kMiB = int64_t{1} << 20;
bool add(int64_t a, int64_t b, int64_t* out) noexcept {
    if (a < 0 || b < 0 || a > INT64_MAX - b) return false;
    *out = a + b;
    return true;
}
int no(std::string* why, const char* message, int rc = RAD_E_STATE) {
    if (why) *why = message;
    return rc;
}
void count_bytes(int64_t* count, int64_t bytes) noexcept {
    *count = *count > INT64_MAX - bytes ? INT64_MAX : *count + bytes;
}
bool valid_range(const void* p, int64_t bytes) noexcept {
    return p && bytes > 0 && uint64_t(bytes) <= UINTPTR_MAX - uintptr_t(p);
}
bool overlap(const void* a, int64_t na, const void* b, int64_t nb) noexcept {
    return uintptr_t(a) < uintptr_t(b) + uint64_t(nb) &&
           uintptr_t(b) < uintptr_t(a) + uint64_t(na);
}
void* unit_address(void* context, int32_t u) {
    return static_cast<Mover*>(context)->unit_base(u);
}
void* weight_address(void* context, uint32_t w) {
    const auto* slot = static_cast<Mover*>(context)->residency()->lookup(w);
    return slot ? slot->ptr : nullptr;
}
draft_bank_backup_detail::Addresses addresses(Mover* m) {
    return {m, unit_address, weight_address};
}
} // namespace

int64_t DraftBankBackupDeviceIO::allocation_charge(int64_t bytes) const noexcept {
    // Covers AQL SVM (4KiB physical rounding), no-SVM pin_whole (below),
    // and the host backend. No assumption about driver SVM availability.
    constexpr int64_t a = RAD_ALIGN_POOL;
    if (bytes <= 0 || bytes > INT64_MAX - (2 * a - 1)) return -1;
    return ((bytes + a - 1) / a) * a + a;
}
bool DraftBankBackupDeviceIO::room(int64_t need, int64_t still, int64_t margin, std::string* why) {
    return lockable_room(need, still, margin, false, why);
}
void* DraftBankBackupDeviceIO::alloc_pinned(int64_t bytes) {
    if (rad_dev_set(ranks_[0].device) < 0) return nullptr;
    return rad_dev_alloc(bytes, RAD_MEM_HOST_PINNED);
}
void DraftBankBackupDeviceIO::free_pinned(void* ptr) noexcept {
    if (!ptr) return;
    (void)rad_dev_set(ranks_[0].device);
    rad_dev_free(ptr, RAD_MEM_HOST_PINNED);
}
int DraftBankBackupDeviceIO::copy_sync(int rank, void* dst, const void* src, int64_t bytes) {
    if (rank < 0 || rank >= kDraftRanks || ranks_[rank].rank != rank ||
        ranks_[rank].device < 0 || !ranks_[rank].stream ||
        !valid_range(dst, bytes) || !valid_range(src, bytes)) return RAD_E_INVAL;
    int rc = rad_dev_set(ranks_[rank].device);
    if (rc >= 0) {
        rc = rad_memcpy_async(dst, src, bytes, ranks_[rank].stream);
        // Drain even an enqueue failure before any scratch reuse/free.
        const int synced = rad_stream_sync(ranks_[rank].stream);
        if (rc >= 0) rc = synced;
    }
    const int reset = rad_dev_set(ranks_[0].device);
    return rc < 0 ? rc : reset;
}

namespace draft_bank_backup_detail {
int derive_layout(const Program& p, const Plan& plan, int n, const Addresses& a,
                  Layout* out, std::string* why) {
    if (!out || !a.unit || !a.weight || n < 1 || n > kDraftRankExperts)
        return no(why, "invalid bank layout request", RAD_E_INVAL);
    *out = {};
    if (!dspark_native_program_scope(p) || plan.units.size() > size_t(INT32_MAX))
        return no(why, "bank is not native DSpark EP4");
    bool seen[kDraftStages][kDraftRankExperts] = {};
    int hot[kDraftStages] = {};
    void* first[kDraftStages][2] = {};
    // In Plan order, exactly as draft_vram_aliases chooses a parked alias.
    for (size_t u = 0; u < plan.units.size(); ++u) {
        const auto& m = plan.units[u];
        if (m.layer < kDraftFirstLayer || m.expert < 0) {
            if (m.draft_vram || m.draft_parked) return no(why, "bank flag outside native experts");
            continue;
        }
        if (m.layer >= kDraftFirstLayer + kDraftStages || m.expert >= kDraftRankExperts ||
            m.weights.size() != 2 || m.offset.size() != 2 || m.draft_vram == m.draft_parked ||
            m.vram_bytes != 0 || !m.vram_offset.empty() || m.movable || m.host_forced ||
            m.site != Site::Device || m.slab_slot != -1)
            return no(why, "unsupported bank unit layout or ownership");
        const int stage = m.layer - kDraftFirstLayer;
        if (seen[stage][m.expert]) return no(why, "duplicate bank expert unit");
        seen[stage][m.expert] = true;
        bool weight_slot[2] = {};
        for (size_t k = 0; k < 2; ++k) {
            const uint32_t w = m.weights[k];
            if (!w || w >= p.weights.size()) return no(why, "bank weight outside Program");
            const auto& wi = p.weights[w];
            const int slot = wi.decl.group.slot;
            if (!dspark_native_weight_scope(wi) || wi.decl.group.layer != m.layer ||
                wi.decl.group.expert != m.expert || wi.vram_bytes || wi.vram_pk_cap != -1 ||
                wi.tier != m.tier || wi.site != Site::Device || wi.slab_slot != -1 ||
                slot < 0 || slot > 1 || weight_slot[slot] ||
                m.offset[k] != (slot ? int64_t{12533760} : int64_t{0}))
                return no(why, "bank is not an unchanged native GU/DN pair");
            weight_slot[slot] = true;
        }
        if (m.draft_parked) {
            if (m.tier != Tier::Mapped || m.slab_class != -1 || m.slot_bytes != 0)
                return no(why, "parked bank unit owns storage");
            continue;
        }
        if (m.tier != Tier::VRAM || m.slot_bytes != kDraftPairBytes || m.bytes != kDraftPairBytes)
            return no(why, "hot bank unit is not a full static raw slot");
        void* base = a.unit(a.context, int32_t(u));
        if (!valid_range(base, kDraftPairBytes) || uintptr_t(base) % RAD_ALIGN_UNIT)
            return no(why, "hot bank slot is missing, unaligned or overflows");
        for (int i = 0; i < out->nslots; ++i)
            if (overlap(base, kDraftPairBytes, out->slots[i].base, out->slots[i].bytes))
                return no(why, "overlapping hot bank slots");
        if (out->nslots == kSlots) return no(why, "too many hot bank slots");
        out->slots[out->nslots++] = {base, kDraftPairBytes, int32_t(u), stage, m.expert};
        ++hot[stage];
        for (size_t k = 0; k < 2; ++k) {
            const int slot = p.weights[m.weights[k]].decl.group.slot;
            if (!first[stage][slot]) first[stage][slot] = reinterpret_cast<void*>(uintptr_t(base) + uint64_t(m.offset[k]));
        }
    }
    for (int s = 0; s < kDraftStages; ++s) {
        if (hot[s] != n) return no(why, "bank resident count differs from explicit N");
        for (bool v : seen[s]) if (!v) return no(why, "bank expert unit missing from Plan");
    }
    for (size_t u = 0; u < plan.units.size(); ++u) {
        const auto& m = plan.units[u];
        if (!m.draft_vram && !m.draft_parked) continue;
        for (size_t k = 0; k < 2; ++k) {
            const uint32_t w = m.weights[k];
            const int s = m.layer - kDraftFirstLayer, slot = p.weights[w].decl.group.slot;
            void* expected = m.draft_vram ? reinterpret_cast<void*>(uintptr_t(a.unit(a.context, int32_t(u))) + uint64_t(m.offset[k]))
                                        : first[s][slot];
            if (p.weights[w].ptr != expected || p.weights[w].host_ptr ||
                a.weight(a.context, w) != expected)
                return no(why, "hot or parked bank Program/Residency alias differs");
            out->bindings[out->nbindings++] = {w, expected, m.draft_vram};
        }
    }
    if (out->nbindings != kBindings) return no(why, "bank binding inventory incomplete");
    rad_weight planes[kDraftStages];
    if (draft_vram_find_planes(p, planes, why) < 0) return RAD_E_STATE;
    for (int s = 0; s < kDraftStages; ++s) {
        const auto& w = p.weights[planes[s]];
        constexpr int64_t bytes = kDraftExperts * int64_t(sizeof(float));
        if (w.tier != Tier::VRAM || w.site != Site::Device || w.stored_bytes != bytes ||
            !valid_range(w.ptr, bytes) || w.host_ptr || a.weight(a.context, planes[s]) != w.ptr)
            return no(why, "router mask plane is not resident at its exact published bytes");
        for (int i = 0; i < out->nslots; ++i)
            if (overlap(w.ptr, bytes, out->slots[i].base, out->slots[i].bytes))
                return no(why, "router mask overlaps a lendable bank slot");
        for (int j = 0; j < s; ++j)
            if (overlap(w.ptr, bytes, out->planes[j].ptr, out->planes[j].bytes))
                return no(why, "router mask planes overlap");
        out->planes[s] = {planes[s], w.ptr, bytes};
    }
    return RAD_OK;
}

int check_bindings(const Program& p, const Layout& layout, const Addresses& a, std::string* why) {
    if (!a.unit || !a.weight || !dspark_native_program_scope(p))
        return no(why, "bank metadata scope changed");
    for (int i = 0; i < layout.nslots; ++i) {
        const auto& slot = layout.slots[i];
        if (a.unit(a.context, slot.hot_unit) != slot.base)
            return no(why, "a hot bank slot base moved");
    }
    for (int i = 0; i < layout.nbindings; ++i) {
        const auto& b = layout.bindings[i];
        if (b.weight >= p.weights.size()) return no(why, "bank weight binding disappeared");
        const auto& w = p.weights[b.weight];
        if (w.ptr != b.ptr || w.host_ptr || w.site != Site::Device || w.slab_slot != -1 ||
            w.tier != (b.hot ? Tier::VRAM : Tier::Mapped) || w.vram_bytes || w.vram_pk_cap != -1 ||
            a.weight(a.context, b.weight) != b.ptr)
            return no(why, "bank Program/Residency alias changed");
    }
    for (const auto& b : layout.planes) {
        if (!b.weight || b.weight >= p.weights.size() || p.weights[b.weight].ptr != b.ptr ||
            p.weights[b.weight].stored_bytes != b.bytes || p.weights[b.weight].host_ptr ||
            p.weights[b.weight].tier != Tier::VRAM || p.weights[b.weight].site != Site::Device ||
            p.weights[b.weight].decl.dtype != RAD_F32 || p.weights[b.weight].decl.rank != 1 ||
            p.weights[b.weight].decl.shape[0] != kDraftExperts ||
            a.weight(a.context, b.weight) != b.ptr)
            return no(why, "bank mask plane binding changed");
    }
    return RAD_OK;
}

Image::~Image() { release(); }
void Image::release() noexcept {
    if (io_) for (void*& p : allocation_) if (p) { io_->free_pinned(p); p = nullptr; }
    ready_ = false;
}
int Image::poison(const char* why, std::string* out) {
    ready_ = false;
    failed_ = true;
    return no(out, why);
}
void* Image::scratch(int rank) const noexcept {
    return rank >= 0 && rank < kDraftRanks && allocation_[rank]
        ? static_cast<char*>(allocation_[rank]) + snapshot_bytes_[rank] : nullptr;
}
int64_t Image::scratch_bytes(int rank) const noexcept {
    return rank >= 0 && rank < kDraftRanks ? ranks_[rank].scratch_bytes : 0;
}
int64_t Image::timing_begin() const noexcept {
    if (!stats_.timings.enabled) return 0;
    const uint64_t now=rad_mono_ns(); // existing steady_clock nanosecond source
    return now<=uint64_t(INT64_MAX)?int64_t(now):-1;
}
int Image::timing_end(DraftBankTimingPart part, int64_t start, int64_t bytes, bool success, std::string* why) {
    if (!stats_.timings.enabled) return RAD_OK;
    const int64_t end = timing_begin();
    if (start < 0 || end < start || bytes < 0 ||
        !stats_.timings.record(part, uint64_t(end)-uint64_t(start), uint64_t(bytes), success)) {
        stats_.timings.overflow = true;
        return poison("bank timing counter overflow or steady-clock regression", why);
    }
    return RAD_OK;
}
void Image::timing_log(const char* operation, int status) const noexcept {
    if (!stats_.timings.enabled) return;
    static constexpr const char* names[] = {"alloc", "zero", "cap_d2h", "cap_fnv", "cap_complement", "cap_verify_d2h", "cap_memcmp",
        "restore_h2d", "restore_complement", "restore_d2h", "restore_memcmp", "invariant_complement", "invariant_d2h", "invariant_memcmp"};
    static_assert(sizeof(names)/sizeof(names[0]) == size_t(DraftBankTimingPart::Count));
    char line[kDraftBankTimingLogBytes];
    int used = std::snprintf(line,sizeof(line),
        "DSBANKTIME operation=%s enabled=1 cumulative=1 clock=steady_ns status=%d overflow=%d ready=%d failed=%d "
        "workspace=%lld stats_bytes=%zu restores=%llu hash=%016llx columns=ns/attempts/ok/requested/okbytes",
        operation,status,stats_.timings.overflow,ready_,failed_,(long long)kDraftBankTimingWorkspaceBytes,
        sizeof(stats_.timings),(unsigned long long)stats_.completed_restores,(unsigned long long)stats_.image_hash);
    bool complete = used > 0 && size_t(used) < sizeof(line);
    for (size_t i = 0; complete && i < stats_.timings.part.size(); ++i) {
        const auto& t = stats_.timings.part[i];
        const int n = std::snprintf(line+used,sizeof(line)-size_t(used)," %s=%llu/%llu/%llu/%llu/%llu",names[i],
            (unsigned long long)t.ns,(unsigned long long)t.attempts,(unsigned long long)t.ok,
            (unsigned long long)t.requested_bytes,(unsigned long long)t.ok_bytes);
        complete = n > 0 && size_t(n) < sizeof(line)-size_t(used);
        if (complete) used += n;
    }
    if (complete) RAD_INFO("%s",line);
    else RAD_ERR("DSBANKTIME operation=%s enabled=1 invalid=log_overflow",operation);
}
int Image::capture(const std::array<ImageRank, kDraftRanks>& ranks, const DraftBankBackupBudget& b,
                   int64_t metadata, DraftBankBackupIO* io, std::string* why) {
    if (attempted_) return poison("bank image capture is one-shot", why);
    attempted_ = true;
    stats_.timings.enabled = b.timings;
    auto fail = [&](const char* message) { release(); return poison(message, why); };
    auto measure = [&](DraftBankTimingPart p,int64_t start,int64_t bytes,bool ok) -> int {
        const int rc=timing_end(p,start,bytes,ok,why);if(rc<0)release();return rc;
    };
    if (!io || b.backup_mib <= 0 || b.backup_mib > INT64_MAX / kMiB ||
        b.expert_cache_budget_bytes < 0 || b.expert_cache_backed_bytes < 0 ||
        b.other_future_bytes < 0 || b.system_headroom_bytes < 0 || b.controller_metadata_bytes < 0 ||
        metadata < int64_t(sizeof(Image)))
        return fail("invalid explicit bank backup budget");
    io_ = io;
    ranks_ = ranks;
    stats_.cap_bytes = b.backup_mib * kMiB;
    if (!add(metadata, b.controller_metadata_bytes, &stats_.metadata_bytes))
        return fail("bank metadata accounting overflow");
    if (b.timings && !add(stats_.metadata_bytes,kDraftBankTimingWorkspaceBytes,&stats_.metadata_bytes))
        return fail("bank timing workspace accounting overflow");
    stats_.controller_metadata_bytes = b.controller_metadata_bytes;
    stats_.total_charge_bytes = stats_.metadata_bytes;
    stats_.future_cache_bytes = std::max<int64_t>(0, b.expert_cache_budget_bytes - b.expert_cache_backed_bytes);
    stats_.future_other_bytes = b.other_future_bytes;
    stats_.system_headroom_bytes = b.system_headroom_bytes;
    std::array<int64_t, kDraftRanks> requested{};
    for (int r = 0; r < kDraftRanks; ++r) {
        const auto& rank = ranks_[r];
        if (rank.nspans <= 0 || rank.nspans > kSpans || rank.scratch_bytes <= 0)
            return fail("incomplete all-rank bank image");
        for (int i = 0; i < rank.nspans; ++i) {
            const auto& s = rank.spans[i];
            if (!valid_range(s.base, s.bytes) || s.bytes > rank.scratch_bytes)
                return fail("invalid bank snapshot span or bounded scratch");
            for (int j = 0; j < i; ++j)
                if (overlap(s.base, s.bytes, rank.spans[j].base, rank.spans[j].bytes))
                    return fail("overlapping bank snapshot spans");
            offsets_[r][i] = snapshot_bytes_[r];
            if (!add(snapshot_bytes_[r], s.bytes, &snapshot_bytes_[r]) ||
                !add(s.restore ? stats_.raw_bytes : stats_.mask_bytes, s.bytes,
                     s.restore ? &stats_.raw_bytes : &stats_.mask_bytes))
                return fail("bank snapshot size overflow");
        }
        if (!add(snapshot_bytes_[r], rank.scratch_bytes, &requested[r]))
            return fail("bank allocation size overflow");
        const int64_t charge = io_->allocation_charge(requested[r]);
        if (charge < requested[r] || !add(stats_.pinned_requested_bytes, requested[r], &stats_.pinned_requested_bytes) ||
            !add(stats_.pinned_charge_bytes, charge, &stats_.pinned_charge_bytes) ||
            !add(stats_.total_charge_bytes, charge, &stats_.total_charge_bytes) ||
            !add(stats_.shared_scratch_bytes, rank.scratch_bytes, &stats_.shared_scratch_bytes))
            return fail("bank allocation accounting overflow");
    }
    stats_.relay_bytes = stats_.readback_bytes = stats_.shared_scratch_bytes;
    if (stats_.total_charge_bytes > stats_.cap_bytes) return fail("bank backup cap does not cover all allocations and metadata");
    int64_t still = 0;
    if (!add(stats_.future_cache_bytes, b.other_future_bytes, &still)) return fail("future host reservation overflow");
    // Single admission before ANY pinned allocation; the complete cohort charge
    // includes every later rank allocation and both roles of the shared scratch.
    if (!io_->room(stats_.total_charge_bytes, still, b.system_headroom_bytes, why)) {
        ready_ = false; failed_ = true; return RAD_E_NOMEM;
    }
    for (int r = 0; r < kDraftRanks; ++r) {
        const int64_t timed=timing_begin();
        allocation_[r] = io_->alloc_pinned(requested[r]);
        const int recorded=measure(DraftBankTimingPart::Allocation,timed,requested[r],allocation_[r]!=nullptr);
        if(recorded<0)return recorded;
        if (!allocation_[r]) { fail("bank backup pinned allocation failed"); return RAD_E_NOMEM; }
    }
    uint64_t hash = UINT64_C(14695981039346656037);
    for (int r = 0; r < kDraftRanks; ++r) {
        for (int i = 0; i < ranks_[r].nspans; ++i) {
            const auto& s = ranks_[r].spans[i];
            auto* saved = static_cast<uint8_t*>(allocation_[r]) + offsets_[r][i];
            int64_t timed=timing_begin();
            std::memset(saved, 0, size_t(s.bytes));
            int recorded=measure(DraftBankTimingPart::Zero,timed,s.bytes,true);
            if(recorded<0)return recorded;
            timed=timing_begin();
            int rc = io_->copy_sync(r, saved, s.base, s.bytes);
            recorded=measure(DraftBankTimingPart::CaptureD2H,timed,s.bytes,rc>=0);
            if(recorded<0)return recorded;
            if (rc < 0) { fail("bank backup capture copy failed"); return rc; }
            count_bytes(&stats_.capture_copy_bytes, s.bytes);
            if ((rc = verify(r, i, why, VerifyPhase::Capture)) < 0) { release(); return rc; }
            timed=timing_begin();
            for (int64_t j = 0; j < s.bytes; ++j) { hash ^= saved[j]; hash *= UINT64_C(1099511628211); }
            recorded=measure(DraftBankTimingPart::CaptureFNV,timed,s.bytes,true);
            if(recorded<0)return recorded;
        }
    }
    stats_.image_hash = hash;
    captured_ = ready_ = true;
    return RAD_OK;
}
int Image::verify(int rank, int span, std::string* why, VerifyPhase phase) {
    const auto& s = ranks_[rank].spans[span];
    const auto* saved = static_cast<const uint8_t*>(allocation_[rank]) + offsets_[rank][span];
    auto* back = static_cast<uint8_t*>(scratch(rank));
    const auto complement = phase==VerifyPhase::Capture?DraftBankTimingPart::CaptureComplement:
        phase==VerifyPhase::Restore?DraftBankTimingPart::RestoreComplement:DraftBankTimingPart::InvariantComplement;
    const auto read = phase==VerifyPhase::Capture?DraftBankTimingPart::CaptureVerifyD2H:
        phase==VerifyPhase::Restore?DraftBankTimingPart::RestoreD2H:DraftBankTimingPart::InvariantD2H;
    const auto compare = phase==VerifyPhase::Capture?DraftBankTimingPart::CaptureMemcmp:
        phase==VerifyPhase::Restore?DraftBankTimingPart::RestoreMemcmp:DraftBankTimingPart::InvariantMemcmp;
    // Every not-copied byte MUST disagree with its reference, including zero
    // tails. Ordinary constant poison cannot guarantee this for arbitrary data.
    int64_t timed=timing_begin();
    for (int64_t j = 0; j < s.bytes; ++j) back[j] = uint8_t(~saved[j]);
    RAD_TRY(timing_end(complement,timed,s.bytes,true,why));
    timed=timing_begin();
    const int rc = io_->copy_sync(rank, back, s.base, s.bytes);
    RAD_TRY(timing_end(read,timed,s.bytes,rc>=0,why));
    if (rc < 0) { poison("bank backup readback copy failed", why); return rc; }
    count_bytes(&stats_.verify_copy_bytes, s.bytes);
    timed=timing_begin();
    const int different=std::memcmp(saved, back, size_t(s.bytes));
    RAD_TRY(timing_end(compare,timed,s.bytes,different==0,why));
    if (different) return poison("bank backup strict byte readback differs", why);
    return RAD_OK;
}
int Image::verify_unchanged(std::string* why) {
    if (!captured_ || failed_) return poison("bank image is not captured and valid", why);
    for (int r = 0; r < kDraftRanks; ++r)
        for (int i = 0; i < ranks_[r].nspans; ++i)
            if (!ranks_[r].spans[i].restore) {
                const int rc = verify(r, i, why, VerifyPhase::Invariant);
                if (rc < 0) return rc;
            }
    return RAD_OK;
}
int Image::restore_verify_all(std::string* why) {
    if (!captured_ || failed_) return poison("bank image is not captured and valid", why);
    ready_ = false;
    int rc = verify_unchanged(why); // invariant planes are never silently repaired
    if (rc < 0) return rc;
    for (int r = 0; r < kDraftRanks; ++r) {
        for (int i = 0; i < ranks_[r].nspans; ++i) {
            const auto& s = ranks_[r].spans[i];
            if (!s.restore) continue;
            const auto* saved = static_cast<const char*>(allocation_[r]) + offsets_[r][i];
            const int64_t timed=timing_begin();
            rc = io_->copy_sync(r, s.base, saved, s.bytes);
            RAD_TRY(timing_end(DraftBankTimingPart::RestoreH2D,timed,s.bytes,rc>=0,why));
            if (rc < 0) { poison("bank backup restore copy failed", why); return rc; }
            count_bytes(&stats_.restore_copy_bytes, s.bytes);
            if ((rc = verify(r, i, why, VerifyPhase::Restore)) < 0) return rc;
        }
    }
    if ((rc = verify_unchanged(why)) < 0) return rc;
    if (stats_.completed_restores == UINT64_MAX) return poison("bank restore generation exhausted", why);
    ++stats_.completed_restores;
    ready_ = true; // cohort publication only after EVERY rank restores and verifies
    return RAD_OK;
}
} // namespace draft_bank_backup_detail

int DraftBankBackup::capture(const DraftBankBackupRanks& ranks, int n, const DraftBankBackupBudget& budget,
                             DraftBankBackupIO* io, std::string* why) {
    if (attempted_) return image_.poison("bank capture is one-shot", why);
    attempted_ = true;
    ranks_ = ranks;
    std::array<draft_bank_backup_detail::ImageRank, kDraftRanks> image_ranks{};
    for (int r = 0; r < kDraftRanks; ++r) {
        const auto& rank = ranks_[r];
        if (rank.rank != r || rank.device < 0 || !rank.program || !rank.plan || !rank.mover ||
            !rank.stream || rank.program->ctx.rank != r)
            return image_.poison("bank backup requires a complete native rank cohort", why);
        for (int j = 0; j < r; ++j)
            if (ranks_[j].device == rank.device || ranks_[j].mover == rank.mover || ranks_[j].program == rank.program)
                return image_.poison("bank backup ranks alias a device or owner", why);
        const int rc = draft_bank_backup_detail::derive_layout(*rank.program, *rank.plan, n,
                                                              addresses(rank.mover), &layout_[r], why);
        if (rc < 0) { image_.poison("bank layout refused", nullptr); return rc; }
        auto& target = image_ranks[r];
        target.scratch_bytes = kDraftPairBytes;
        for (int i = 0; i < layout_[r].nslots; ++i) {
            const auto& s = layout_[r].slots[i];
            target.spans[target.nspans++] = {s.base, s.bytes, true};
        }
        for (const auto& p : layout_[r].planes)
            target.spans[target.nspans++] = {p.ptr, p.bytes, false};
    }
    device_io_.bind(ranks_);
    // Charge retained metadata PLUS the bounded capture descriptor on the stack.
    const int rc = image_.capture(image_ranks, budget, sizeof(*this) + sizeof(image_ranks),
                                  io ? io : &device_io_, why);
    const int result=rc<0?rc:validate_invariants(why);
    image_.timing_log("capture",result);
    return result;
}
int DraftBankBackup::check_pointers(std::string* why) {
    if (!attempted_ || image_.failed()) return image_.poison("bank backup is not valid", why);
    for (int r = 0; r < kDraftRanks; ++r) {
        if (!ranks_[r].program || !ranks_[r].mover ||
            draft_bank_backup_detail::check_bindings(*ranks_[r].program, layout_[r], addresses(ranks_[r].mover), why) < 0) {
            image_.poison("bank pointer invariant failed", nullptr);
            return RAD_E_STATE;
        }
    }
    return RAD_OK;
}
int DraftBankBackup::validate_invariants(std::string* why) {
    const int rc = check_pointers(why);
    return rc < 0 ? rc : image_.verify_unchanged(why);
}
int DraftBankBackup::restore_verify_all(std::string* why) {
    int rc = check_pointers(why);
    if (rc >= 0) rc = image_.restore_verify_all(why);
    if (rc >= 0) rc = validate_invariants(why);
    image_.timing_log("restore",rc);
    return rc;
}
int DraftBankBackup::slots(int rank) const noexcept {
    return rank >= 0 && rank < kDraftRanks ? layout_[rank].nslots : 0;
}
const DraftBankBackupSlot& DraftBankBackup::slot(int rank, int i) const {
    static const DraftBankBackupSlot empty{};
    return rank >= 0 && rank < kDraftRanks && i >= 0 && i < layout_[rank].nslots ? layout_[rank].slots[i] : empty;
}
} // namespace rad
