#include "engine_priv.h"
#include "device/device.h"

#include <cstring>

namespace rad {
int Engine::kv_quiescent_plan(const std::function<int()>& change, const RadBatch** out) {
    if (!out) return RAD_E_INVAL;
    *out = nullptr;
    if (!change) return RAD_E_INVAL;
    /* This first seam is deliberately narrower than the normal engine. A future public mode
     * must validate these at startup as well, and turn pipelining off before using the seam. */
    if (!meta_.arch_id || std::strcmp(meta_.arch_id, "deepseek41") || cfg_.tp != 4 || ranks_.size() != 4 ||
        cfg_.max_seqs != 1 || !cfg_.deterministic || cfg_.n_spec != 0 || pipelining_ ||
        serial_drafting() || block_drafting() || tree_ || gen_ || embedding_ ||
        cfg_.prefix_cache_mib > 0 || cfg_.kv_resident > 0 || lending_ ||
        !barrier_ || !barrier_->parked_for_scheduler()) return RAD_E_UNSUPPORTED;
    for (const auto& r : ranks_)
        if (r->stager.enabled() || r->fstager.p1_on() || r->fstager.p1_lend_arena_asked() ||
            r->tierx.enabled() || r->kv.loan_tokens() != 0 || r->program.encoder.modalities ||
            r->program.audio_encoder.modalities) return RAD_E_UNSUPPORTED;

    return sched_.step_quiescent([&]() -> int {
        /* The sole step thread has not arrived at the next barrier, so parked ranks cannot
         * restart. HTTP cancel/reap/prefix changes wait on this ownership lock. None of the
         * helpers below calls Scheduler: CPU miss jobs ended before the rank join, RAM insert
         * helpers use only mover/cache locks, and GPU events need no host scheduler callback. */
        if (!barrier_->parked_for_scheduler()) return RAD_E_STATE;
        for (auto& r : ranks_) {
            const int rc = [&]() -> int {
                RAD_TRY(rad_dev_set(r->device));
                RAD_TRY(r->fstager.quiesce_for_backing());
                RAD_TRY(r->ctx.quiesce_for_backing());
                RAD_TRY(r->mover.mapped_file_drain());
                if (RadStream transfer = r->mover.transfer_stream()) RAD_TRY(rad_stream_sync(transfer));
                return r->tierx.quiesce_checked();
            }();
            if (rc < 0) { (void)rad_dev_set(ranks_[0]->device); return rc; }
        }
        RAD_TRY(rad_dev_set(ranks_[0]->device));
        /* May call the backing transaction with an already-held quiescence lease. It MUST NOT
         * call public Scheduler methods, prepare_release, prefill_hash_stage or tier_tick. */
        const int rc = change();
        const int restored = rad_dev_set(ranks_[0]->device);
        return rc < 0 ? rc : restored;
    }, out);
}
}  // namespace rad
