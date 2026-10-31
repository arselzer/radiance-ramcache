/* Native DSpark bank lending. No query/depth/routing arithmetic changes. The only
 * reuse interval is a Scheduler frame with the existing solo gate held AND no
 * context/query work. Each transition holds Scheduler's mutex and a parked CPU
 * cohort, then drains every compute/copy producer before changing residency. */
#include "engine_priv.h"
#include "device/device.h"
#include "engine_hostmem.h"
#include "place/draft_bank_complement.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>

namespace rad {
namespace {
int64_t bank_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
const char* bank_action(draft_bank::Action a) {
    return a == draft_bank::Action::Lend ? "lend" : "restore";
}
}

int Engine::draft_bank_lend_drain(bool startup, const char* action, uint64_t epoch) {
    if (!draft_bank_loan_) return RAD_E_STATE;
    /* gen even is the paired release/join boundary, NOT merely all queues empty.
     * No code here calls arrive_and_wait: workers remain parked throughout. On
     * startup/after all joins no workers exist and the same stream drains apply. */
    if (!startup && (!barrier_ || !barrier_->parked_for_scheduler())) return RAD_E_STATE;
    const int64_t t = bank_now_ns();
    for (auto& p : ranks_) {
        Rank& r = *p;
        RAD_TRY(rad_dev_set(r.device));
        RAD_TRY(r.fstager.quiesce_for_draft_bank());
        if (r.mover.unit_planes_started()) return RAD_E_STATE;
        RAD_TRY(r.ctx.quiesce_for_backing()); // includes rowprefetch and BOTH lanes
        RAD_TRY(r.mover.mapped_file_drain());
        if (r.mover.transfer_stream()) RAD_TRY(rad_stream_sync(r.mover.transfer_stream()));
        RAD_TRY(r.tierx.quiesce_checked());
        if (action) RAD_INFO("DSBANKLEND rank=%d epoch=%llu action=%s stage=drained cpu_parked=1",
            r.index, (unsigned long long)epoch, action);
    }
    draft_bank_loan_->drain_ns += (uint64_t)(bank_now_ns() - t);
    return rad_dev_set(ranks_[0]->device);
}

int Engine::draft_bank_lend_open() {
    if (!draft_bank_config_.on) return RAD_OK;
    const int64_t startup_t=bank_now_ns();
    auto no = [&](const char* why) {
        RAD_ERR("DSBANKLEND startup refused: %s", why); return RAD_E_UNSUPPORTED;
    };
    if (draft_bank_loan_ || draft_vram_n_ < 1 || draft_vram_n_ > kDraftRankExperts ||
        cfg_.tp != kDraftRanks || ranks_.size() != kDraftRanks || !block_drafting() ||
        !sched_.draft_solo() || !sched_.ctx_deferred() || gen_ || tree_ ||
        !cfg_.deterministic || cfg_.lora.enabled() || cfg_.media_flags_set())
        return no("requires native EP4 hot bank, block DSpark beside concurrency, explicit Dx, text-only chain");
    if (lending_) return no("generic KV/arena expert loans need a combined ownership proof");
    for (const auto& p : ranks_) {
        const Rank& r = *p;
        if (r.plan.dynamic || r.program.two_batch || r.fstager.draft_bank_lend_conflict() || r.kv.loan_tokens()!=0 ||
            r.mover.flex_capacity() > 0)
            return no("dynamic placement, TBO, P1 ring/arena lenders, or generic flex slots conflict");
    }
    std::string preallocation_why;
    int64_t future_cache = 0;
    for (const auto& r : ranks_) {
        const int64_t future=std::max<int64_t>(0,r->mover.mapped_cache_bytes()-r->mover.mapped_cache_backed_bytes());
        if(future_cache>INT64_MAX-future)return RAD_E_INVAL;
        future_cache+=future;
    }
    if(cfg_.host_headroom_mib<0||cfg_.host_headroom_mib>INT64_MAX/(1ll<<20)||
       (int64_t)sizeof(DraftBankLoan)>draft_bank_config_.backup_cap ||
       !lockable_room(sizeof(DraftBankLoan),future_cache,cfg_.host_headroom_mib*(1ll<<20),false,&preallocation_why)) {
        RAD_ERR("DSBANKLEND controller preallocation refused: %s",preallocation_why.c_str());
        return RAD_E_NOMEM;
    }
    auto loan = std::make_unique<DraftBankLoan>(draft_bank_config_);
    /* Bounded insertion ranking avoids a new vector proportional to model size.
     * Existing profile order is deterministic; ties use immutable unit identity. */
    for (int ri = 0; ri < kDraftRanks; ++ri) {
        Rank& r = *ranks_[(size_t)ri];
        const int want = kDraftStages * draft_vram_n_;
        auto better = [&](int32_t a, int32_t b) {
            const MoveUnit& x = r.plan.units[(size_t)a];
            const MoveUnit& y = r.plan.units[(size_t)b];
            return x.profile_share > y.profile_share || (x.profile_share == y.profile_share && a < b);
        };
        for (size_t u = 0; u < r.plan.units.size(); ++u) {
            const MoveUnit& m = r.plan.units[u];
            if (m.layer < 0 || m.layer >= kDraftFirstLayer || m.expert < 0 ||
                m.tier != Tier::SSD || m.slab_slot >= 0 || m.slab_class < 0 || m.draft_head ||
                m.draft_vram || m.draft_parked || m.vram_bytes || m.weights.size() != 2 ||
                m.offset.size() != 2 || m.bytes != kDraftPairBytes || m.slot_bytes != kDraftPairBytes ||
                m.offset[0] != 0 || m.offset[1] != 12533760 || r.mover.file_unit_in_slab((int32_t)u) ||
                !r.mover.host_planes_eligible((int32_t)u)) continue;
            const WeightInfo& gu = r.program.weights[m.weights[0]];
            const WeightInfo& dn = r.program.weights[m.weights[1]];
            if (gu.stored_bytes != 12533760 || dn.stored_bytes != 6266880) continue;
            int& n = loan->count[(size_t)ri];
            int at = n;
            while (at > 0 && better((int32_t)u, loan->target[(size_t)ri][(size_t)at-1])) --at;
            if (at >= want) continue;
            for (int j = std::min(n, want-1); j > at; --j)
                loan->target[(size_t)ri][(size_t)j] = loan->target[(size_t)ri][(size_t)j-1];
            loan->target[(size_t)ri][(size_t)at] = (int32_t)u;
            n = std::min(n+1, want);
        }
        if (loan->count[(size_t)ri] != want) return no("not enough native file-tier target slots of unchanged bank geometry");
    }
    draft_bank_loan_ = std::move(loan);
    RAD_TRY(draft_bank_lend_drain(/*startup=*/true, nullptr, 0));
    DraftBankBackupRanks io{};
    DraftBankBackupBudget budget{};
    budget.backup_mib = draft_bank_config_.backup_cap / (1ll<<20);
    budget.timings = draft_bank_config_.timings;
    budget.complement_words = draft_bank_config_.complement_words;
    budget.controller_metadata_bytes = sizeof(DraftBankLoan) + DraftBankLoan::kSourceBound;
    if (draft_bank_config_.packed_fill)
        budget.controller_metadata_bytes += draft_bank_packed_fill::kCallMetadataBound;
    budget.system_headroom_bytes = std::max<int64_t>(cfg_.host_headroom_mib, 0) * (1ll<<20);
    for (int ri = 0; ri < kDraftRanks; ++ri) {
        Rank& r = *ranks_[(size_t)ri];
        io[(size_t)ri] = {ri, r.device, &r.program, &r.plan, &r.mover,
                         r.mover.transfer_stream() ? r.mover.transfer_stream() : r.ctx.stream()};
        const int64_t cache = r.mover.mapped_cache_bytes(), backed = r.mover.mapped_cache_backed_bytes();
        if (cache < 0 || backed < 0 || budget.expert_cache_budget_bytes > INT64_MAX-cache ||
            budget.expert_cache_backed_bytes > INT64_MAX-backed) return RAD_E_INVAL;
        budget.expert_cache_budget_bytes += cache;
        budget.expert_cache_backed_bytes += backed;
    }
    std::string why;
    const int captured=draft_bank_loan_->backup.capture(io, draft_vram_n_, budget, nullptr, &why);
    if(captured<0){RAD_ERR("DSBANKLEND capture refused: %s",why.c_str());return captured;}
    DraftBankLoan& l = *draft_bank_loan_;
    /* Preallocate/reuse the exact two extent buffers, after cap/headroom preflight.
     * All candidates have the same native stored geometry. No generation-time
     * source-buffer growth is permitted; one shared pair is charged in full. */
    for (int ri = 0; ri < kDraftRanks; ++ri) for (int i = 0; i < l.count[(size_t)ri]; ++i) {
        l.source.unit = l.target[(size_t)ri][(size_t)i];
        PlaneExtent ext[2];
        if (!ranks_[(size_t)ri]->mover.lay_out_unit_planes(&l.source, ext) ||
            l.source.cap[0] > DraftBankLoan::kSourcePayload - l.source.cap[1])
            return no("source extent buffers exceed the explicitly charged bound");
    }
    RAD_TRY(sched_.enable_draft_bank_observation());
    for(auto&r:ranks_)r->fstager.track_draft_bank_residents();
    const auto& s = l.backup.stats();
    l.startup_ns=(uint64_t)(bank_now_ns()-startup_t);
    RAD_INFO("DSBANKLEND startup ranks=4 N=%d bank_bytes=%lld mask_bytes=%lld relay_bytes=%lld readback_bytes=%lld "
             "shared_scratch_bytes=%lld pinned_requested=%lld pinned_charge=%lld metadata_bytes=%lld "
             "total_charge=%lld cap=%lld future_cache=%lld system_headroom=%lld grace_ms=%lld cooldown_ms=%lld "
             "source_extent_bound=%lld startup_ns=%llu restore=sync_strict_memcmp explicit_Dx=1",
        draft_vram_n_, (long long)s.raw_bytes, (long long)s.mask_bytes, (long long)s.relay_bytes,
        (long long)s.readback_bytes, (long long)s.shared_scratch_bytes, (long long)s.pinned_requested_bytes,
        (long long)s.pinned_charge_bytes, (long long)s.metadata_bytes, (long long)s.total_charge_bytes,
        (long long)s.cap_bytes, (long long)s.future_cache_bytes, (long long)s.system_headroom_bytes,
        (long long)draft_bank_config_.grace_ms, (long long)draft_bank_config_.cooldown_ms,
        (long long)DraftBankLoan::kSourceBound,(unsigned long long)l.startup_ns);
    if (draft_bank_config_.packed_fill)
        RAD_INFO("DSBANKPACK startup enabled=1 native_gu=4608x5120 native_dn=5120x2304 pair_bytes=%lld "
                 "relay=backup_shared_pair extra_payload_bytes=0 call_metadata_bound=%lld leases=through_sync_copy",
                 (long long)kDraftPairBytes, (long long)draft_bank_packed_fill::kCallMetadataBound);
    if (draft_bank_config_.complement_words)
        RAD_INFO("DSBANKCOMPLEMENT startup enabled=1 word_bytes=8 workspace=%lld alignment=unaligned "
                 "overlap=legacy_byte tail=legacy_byte readback=full_memcmp hash=unchanged copies=unchanged",
                 (long long)draft_bank_backup_detail::kComplementWordWorkspaceBytes);
    return RAD_OK;
}

int Engine::draft_bank_lend_transition(const draft_bank::Work& w, draft_bank::Action a, bool startup) {
    using namespace draft_bank;
    DraftBankLoan& l = *draft_bank_loan_;
    const uint64_t ticket = l.controller.epoch();
    if(a==Action::Lend)l.loan_epoch=ticket;
    if(!l.loan_epoch){l.controller.fatal();return RAD_E_STATE;}
    const uint64_t epoch = l.loan_epoch; // stable copy/loan identity across later restore frames
    const char* action = bank_action(a);
    RAD_INFO("DSBANKLEND transition epoch=%llu step=%lld seq=%llu action=%s phase=begin pending_ctx=%lld "
             "pending_query=%lld solo_held=%d deferred=%d all_ranks=4 frame_epoch=%llu",
        (unsigned long long)epoch, (long long)w.step, (unsigned long long)w.query_seq, action,
        (long long)w.context_rows, (long long)w.query_rows, w.solo_held, w.deferred,(unsigned long long)w.epoch);
    const int64_t t = bank_now_ns();
    int rc = draft_bank_lend_drain(startup, action, epoch);
    if (rc < 0) { l.controller.fatal(); return rc; }
    std::string why;
    auto publish_all = [&](bool on) {
        for(int ri=0;ri<kDraftRanks;++ri) for(int i=0;i<l.count[(size_t)ri];++i)
            ranks_[(size_t)ri]->mover.set_ring_resident(l.target[(size_t)ri][(size_t)i],on?
                static_cast<char*>(l.backup.slot(ri,i).base):nullptr);
        l.published=on;
        for(int ri=0;ri<kDraftRanks;++ri) for(int i=0;i<l.count[(size_t)ri];++i) {
            Rank&r=*ranks_[(size_t)ri];const int32_t u=l.target[(size_t)ri][(size_t)i];
            const MoveUnit&m=r.plan.units[(size_t)u];
            char*base=on?static_cast<char*>(l.backup.slot(ri,i).base):nullptr;
            if(r.mover.ring_resident(u)!=on||r.mover.unit_base(u)!=base)return RAD_E_STATE;
            for(size_t k=0;k<m.weights.size();++k){
                const WeightSlot*ws=r.mover.residency()->lookup(m.weights[k]);
                if(!ws||ws->ptr!=(base?base+m.offset[k]:nullptr))return RAD_E_STATE;
            }
        }
        return RAD_OK;
    };
    if (a == Action::Lend) {
        if (!w.lendable() || l.published || !l.backup.ready()) { l.controller.fatal(); return RAD_E_STATE; }
        const int valid=l.backup.validate_invariants(&why);
        if(valid<0){l.controller.fatal();RAD_ERR("DSBANKLEND fatal invariant: %s",why.c_str());return valid;}
        l.backup.mark_borrowed();
        for(int ri=0;ri<kDraftRanks;++ri)l.selected_at_lend[(size_t)ri]=ranks_[(size_t)ri]->fstager.draft_bank_selected_units();
        /* Every source and copy lands before ANY target is published. Cancels/new
         * arrivals cannot race this cohort because the scheduler lock is held. */
        for (int ri = 0; ri < kDraftRanks && rc >= 0; ++ri) {
            Rank& r = *ranks_[(size_t)ri];
            const uint64_t hits_before=l.packed_hits, refusals_before=l.packed_refusals, misses_before=l.packed_misses;
            const uint64_t ram_before=l.source_ram_bytes, file_before=l.source_drive_bytes, fill_before=l.fill_bytes;
            for (int i = 0; i < l.count[(size_t)ri] && rc >= 0; ++i) {
                const int32_t u = l.target[(size_t)ri][(size_t)i];
                const MoveUnit& m = r.plan.units[(size_t)u];
                auto* scratch = static_cast<uint8_t*>(l.backup.scratch(ri));
                if (!scratch || r.mover.file_unit_in_slab(u) || !r.mover.source_epochs_current()) { rc=RAD_E_STATE; break; }
                std::memset(scratch, 0, (size_t)kDraftPairBytes);
                Mover::HostUnit hu;
                const bool borrowed = r.mover.borrow_host_unit(u, &hu, /*touch=*/false);
                bool packed_ready = false;
                if (draft_bank_config_.packed_fill && borrowed && hu.packed) {
                    const int64_t inverse_t = bank_now_ns();
                    const int prepared = draft_bank_packed_fill::prepare(r.program, r.plan, u, hu,
                        scratch, (size_t)kDraftPairBytes, &l.packed_fill,
                        [](void* p) noexcept { return static_cast<Mover*>(p)->source_epochs_current(); }, &r.mover);
                    l.packed_inverse_ns += (uint64_t)(bank_now_ns() - inverse_t);
                    if (prepared >= 0) {
                        packed_ready = true;
                        ++l.packed_hits;
                        l.packed_source_bytes += l.packed_fill.source_bytes;
                        l.packed_raw_bytes += l.packed_fill.raw_bytes;
                    } else {
                        ++l.packed_refusals;
                        // The first inverse may have completed before the second refused.
                        // Never copy a partial pair; the existing file relay fills a clean pair.
                        std::memset(scratch, 0, (size_t)kDraftPairBytes);
                        if (l.packed_fill.refusal == draft_bank_packed_fill::Refusal::Epoch ||
                            !r.mover.source_epochs_current()) rc = RAD_E_STATE;
                    }
                } else if (draft_bank_config_.packed_fill && !borrowed) ++l.packed_misses;
                if (!packed_ready && rc >= 0 && borrowed && !hu.packed && hu.n == 2) {
                    for (int k = 0; k < 2; ++k) {
                        if (!hu.host[k] || hu.bytes[k] != r.program.weights[m.weights[(size_t)k]].stored_bytes) { rc=RAD_E_STATE; break; }
                        std::memcpy(scratch + m.offset[(size_t)k], hu.host[k], (size_t)hu.bytes[k]);
                        l.source_ram_bytes += (uint64_t)hu.bytes[k];
                    }
                } else if (!packed_ready && rc >= 0) {
                    // Packed CPU cache is never read as BLOCK: original source + exact relay.
                    l.source.unit = u;
                    PlaneExtent ext[2];
                    const int64_t c0=l.source.cap[0],c1=l.source.cap[1];
                    if (!r.mover.lay_out_unit_planes(&l.source, ext) || l.source.cap[0]!=c0 || l.source.cap[1]!=c1)
                        rc=RAD_E_STATE;
                    for (int k = 0; k < 2 && rc >= 0; ++k) {
                        rc=r.mover.read_piece(ext[k].file, ext[k].off, ext[k].len, ext[k].dst, ext[k].cap);
                        if (rc >= 0) l.source_drive_bytes += (uint64_t)ext[k].len;
                    }
                    if (rc >= 0) rc=r.mover.relay_unit_block(l.source,scratch,kDraftPairBytes);
                    if (rc >= 0) l.relay_bytes += (uint64_t)kDraftPairBytes;
                }
                // Preserve the flag0 path. Enabled packed fill retains BOTH leases even
                // through file fallback and any synchronously failing H2D copy.
                if (borrowed && !draft_bank_config_.packed_fill) r.mover.release_host_unit(&hu);
                if (rc >= 0 && draft_bank_config_.packed_fill && !r.mover.source_epochs_current()) rc=RAD_E_STATE;
                if (rc >= 0) rc=l.backup.device_io().copy_sync(ri,l.backup.slot(ri,i).base,scratch,kDraftPairBytes);
                if (borrowed && draft_bank_config_.packed_fill) r.mover.release_host_unit(&hu);
                if (rc >= 0 && draft_bank_config_.packed_fill && !r.mover.source_epochs_current()) rc=RAD_E_STATE;
                if (rc >= 0) l.fill_bytes += (uint64_t)kDraftPairBytes;
            }
            if (draft_bank_config_.packed_fill)
                RAD_INFO("DSBANKPACK rank=%d epoch=%llu stage=fill packed_hits=%llu packed_refusals=%llu packed_misses=%llu "
                    "raw_ram_bytes=%llu source_file_bytes=%llu whole_pair_fill_bytes=%llu status=%d",
                    ri,(unsigned long long)epoch,(unsigned long long)(l.packed_hits-hits_before),
                    (unsigned long long)(l.packed_refusals-refusals_before),(unsigned long long)(l.packed_misses-misses_before),
                    (unsigned long long)(l.source_ram_bytes-ram_before),(unsigned long long)(l.source_drive_bytes-file_before),
                    (unsigned long long)(l.fill_bytes-fill_before),rc);
        }
        if (rc >= 0 && draft_bank_config_.packed_fill)
            for (const auto& rank : ranks_)
                if (!rank->mover.source_epochs_current()) { rc=RAD_E_STATE; break; }
        if (rc >= 0) {
            // Fixed metadata/atomic publications cannot fail. Validate every actual pointer.
            rc=publish_all(true);
        }
        if(rc<0) {
            // Includes partially copied/published cohort: unpublish then verify ALL originals.
            const int revoked=publish_all(false);
            if(revoked<0){l.controller.fatal();return revoked;}
            const int back=l.backup.restore_verify_all(&why);
            ++l.rollbacks;
            if(back<0){l.controller.fatal();RAD_ERR("DSBANKLEND fatal rollback: %s",why.c_str());return back;}
            RAD_TRY(l.controller.finish(a,ticket,false,bank_now_ns()/1000000));
            RAD_ERR("DSBANKLEND rollback epoch=%llu restored=1 fill_error=%d",(unsigned long long)epoch,rc);
            return RAD_OK; // unchanged bank ready, no partial loan is usable
        }
        RAD_TRY(l.controller.finish(a,ticket,true,bank_now_ns()/1000000));
        ++l.lends;l.fill_ns+=(uint64_t)(bank_now_ns()-t);
        for(int ri=0;ri<kDraftRanks;++ri)RAD_INFO("DSBANKLEND rank=%d epoch=%llu stage=lent slots=%d borrowed_bytes=%lld bank_bytes=%lld copied=1",
            ri,(unsigned long long)epoch,l.count[(size_t)ri],
            (long long)l.count[(size_t)ri]*kDraftPairBytes,(long long)l.backup.slots(ri)*kDraftPairBytes);
    } else {
        const int revoked=publish_all(false);
        if(revoked<0){l.controller.fatal();return revoked;}
        rc=l.backup.restore_verify_all(&why);
        const int finished=l.controller.finish(a,ticket,rc>=0,bank_now_ns()/1000000);
        if(rc<0||finished<0){l.controller.fatal();RAD_ERR("DSBANKLEND fatal restore: %s",why.c_str());return rc<0?rc:finished;}
        ++l.restores;l.restore_ns+=(uint64_t)(bank_now_ns()-t);
        l.resume_pending=true;l.resume_epoch=epoch;
        for(int ri=0;ri<kDraftRanks;++ri)RAD_INFO("DSBANKLEND rank=%d epoch=%llu stage=restored bytes=%lld verified=1 aliases_ok=1 masks_ok=1 "
            "hash=%016llx expected_hash=%016llx hash_scope=cohort strict_memcmp=1 selected_units=%llu restored_bytes=%lld invariant_bytes=%d",
            ri,(unsigned long long)epoch,(long long)l.backup.slots(ri)*kDraftPairBytes+kDraftStages*512,
            (unsigned long long)l.backup.stats().image_hash,(unsigned long long)l.backup.stats().image_hash,
            (unsigned long long)(ranks_[(size_t)ri]->fstager.draft_bank_selected_units()-l.selected_at_lend[(size_t)ri]),
            (long long)l.backup.slots(ri)*kDraftPairBytes,kDraftStages*512);
    }
    RAD_INFO("DSBANKLEND transition epoch=%llu step=%lld seq=%llu action=%s phase=ready pending_ctx=%lld pending_query=%lld "
             "solo_held=%d deferred=%d all_ranks=4 frame_epoch=%llu",
        (unsigned long long)epoch,(long long)w.step,(unsigned long long)w.query_seq,action,
        (long long)w.context_rows,(long long)w.query_rows,w.solo_held,w.deferred,(unsigned long long)w.epoch);
    return RAD_OK;
}

int Engine::draft_bank_lend_step(int step) {
    return sched_.with_draft_bank_frame(step,[&](const draft_bank::Work&w) -> int {
        DraftBankLoan& l=*draft_bank_loan_;
        l.query_step=w.step;l.query_seq=w.query_seq;
        draft_bank::Action a;
        RAD_TRY(l.controller.decide(w,bank_now_ns()/1000000,&a));
        if(a!=draft_bank::Action::None) RAD_TRY(draft_bank_lend_transition(w,a));
        if((w.context_rows||w.query_rows) && (!l.controller.ready()||!l.backup.ready()||l.published))return RAD_E_STATE;
        return RAD_OK;
    });
}

int Engine::draft_bank_lend_idle() {
    return sched_.with_draft_bank_idle([&]() -> int {
        DraftBankLoan& l=*draft_bank_loan_;
        draft_bank::Action a;RAD_TRY(l.controller.restore_idle(l.controller.epoch(),&a));
        if(a==draft_bank::Action::None)return RAD_OK;
        draft_bank::Work w{};w.deferred=true;
        return draft_bank_lend_transition(w,a);
    });
}

int Engine::draft_bank_lend_query_issued(int step,int64_t rows) {
    DraftBankLoan& l=*draft_bank_loan_;
    if(!l.controller.ready()||!l.backup.ready()||l.published||l.query_step!=step||!l.query_seq||rows<=0)return RAD_E_STATE;
    if(l.resume_pending){
        RAD_INFO("DSBANKLEND resume epoch=%llu step=%d seq=%llu query_rows=%lld ready=1",
            (unsigned long long)l.resume_epoch,step,(unsigned long long)l.query_seq,(long long)rows);
        l.resume_pending=false;
    }
    return RAD_OK;
}

int Engine::draft_bank_lend_close() {
    DraftBankLoan& l=*draft_bank_loan_;
    int rc=RAD_OK;
    if(l.published){
        draft_bank::Action a;
        rc=l.controller.restore_idle(l.controller.epoch(),&a);
        if(rc>=0&&a==draft_bank::Action::Restore){draft_bank::Work w{};w.deferred=true;
            rc=draft_bank_lend_transition(w,a,/*startup/no workers=*/true);}
    }
    const auto&s=l.backup.stats();
    RAD_INFO("DSBANKLEND totals lends=%llu restores=%llu rollbacks=%llu borrowed_bytes=%lld fill_bytes=%llu "
        "source_drive_bytes=%llu source_ram_bytes=%llu relay_bytes=%llu restore_bytes=%lld verify_bytes=%lld "
        "startup_ns=%llu drain_ns=%llu fill_ns=%llu restore_ns=%llu ready=%d published=%d",
        (unsigned long long)l.lends,(unsigned long long)l.restores,(unsigned long long)l.rollbacks,
        l.published?(long long)s.raw_bytes:0ll,(unsigned long long)l.fill_bytes,(unsigned long long)l.source_drive_bytes,
        (unsigned long long)l.source_ram_bytes,(unsigned long long)l.relay_bytes,(long long)s.restore_copy_bytes,
        (long long)s.verify_copy_bytes,(unsigned long long)l.startup_ns,(unsigned long long)l.drain_ns,(unsigned long long)l.fill_ns,
        (unsigned long long)l.restore_ns,l.backup.ready(),l.published);
    if (draft_bank_config_.packed_fill)
        RAD_INFO("DSBANKPACK totals enabled=1 packed_hits=%llu packed_refusals=%llu packed_misses=%llu "
            "packed_source_bytes=%llu packed_raw_bytes=%llu raw_ram_bytes=%llu source_file_bytes=%llu "
            "inverse_ns=%llu whole_pair_fill_bytes=%llu",
            (unsigned long long)l.packed_hits, (unsigned long long)l.packed_refusals,
            (unsigned long long)l.packed_misses, (unsigned long long)l.packed_source_bytes,
            (unsigned long long)l.packed_raw_bytes, (unsigned long long)l.source_ram_bytes,
            (unsigned long long)l.source_drive_bytes, (unsigned long long)l.packed_inverse_ns,
            (unsigned long long)l.fill_bytes);
    draft_bank_loan_.reset();return rc;
}
} // namespace rad
