#include "engine_priv.h"
#include "device/device.h"

#include <cstring>
#include <cstdlib>
#include <limits>

namespace rad {
int Engine::kv_growth_parse() {
    std::string why;
    if (!kv_live_options([](const char* key) { return std::getenv(key); }, &kv_growth_options_, &why)) {
        RAD_ERR("%s refused: %s", kKVLiveEnv, why.c_str());
        return RAD_E_INVAL;
    }
    if (!kv_growth_options_.enabled) return RAD_OK;
    if (cfg_.tp != 4 || cfg_.max_seqs != 1 || !cfg_.deterministic || cfg_.n_spec != 0 ||
        cfg_.prefix_cache_mib > 0 || cfg_.kv_resident > 0 || cfg_.lora.enabled()) {
        RAD_ERR("%s requires explicit TP4, max-num-seqs1, deterministic fixed experts, speculative "
                "depth0, no host-prefix tier/KV streaming/LoRA", kKVLiveEnv);
        return RAD_E_UNSUPPORTED;
    }
    for (const char* key : { "RADIANCE_EXPERIMENTAL_DS41_P1",
                             "RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA",
                             "RADIANCE_EXPERIMENTAL_SCHED_TBO" }) {
        const char* v = std::getenv(key);
        if (v && *v && std::strcmp(v, "0")) {
            RAD_ERR("%s refuses %s in its first bounded cohort", kKVLiveEnv, key);
            return RAD_E_UNSUPPORTED;
        }
    }
    return RAD_OK;
}

int Engine::kv_growth_devices(const std::vector<int>& devices) {
    if (!kv_growth_options_.enabled) return RAD_OK;
    int64_t common = 0;
    for (int i = 0; i < cfg_.tp; ++i) {
        RAD_TRY(rad_dev_set(devices[(size_t)i]));
        if (!VMem::supported()) {
            (void)rad_dev_set(devices[0]);
            RAD_ERR("%s requires actual elastic VMem on every rank; fixed allocation fallback is "
                    "available with this switch off", kKVLiveEnv);
            return RAD_E_UNSUPPORTED;
        }
        const int64_t G = VMem::granularity();
        if (G <= 0 || (common > 0 && common != G) || kv_growth_options_.maximum_bytes % G) {
            (void)rad_dev_set(devices[0]);
            RAD_ERR("%s requires one common backend granule and a MAX_MIB that funds whole physical "
                    "granules", kKVLiveEnv);
            return RAD_E_UNSUPPORTED;
        }
        common = G;
    }
    return rad_dev_set(devices[0]);
}

int Engine::kv_growth_scope(bool runtime) const {
    if (!kv_growth_options_.enabled) return RAD_OK;
    if (!meta_.arch_id || std::strcmp(meta_.arch_id, "deepseek41") || cfg_.tp != 4 || ranks_.size() != 4 ||
        cfg_.max_seqs != 1 || !cfg_.deterministic || cfg_.n_spec != 0 ||
        serial_drafting() || block_drafting() || tree_ || gen_ || embedding_ ||
        cfg_.prefix_cache_mib > 0 || cfg_.kv_resident > 0 || cfg_.lora.enabled() || lending_ ||
        (runtime && (pipelining_ || !barrier_ || !barrier_->parked_for_scheduler())))
        return RAD_E_UNSUPPORTED;
    for (const auto& r : ranks_)
        if (r->stager.enabled() || r->fstager.p1_on() || r->fstager.p1_lend_arena_asked() ||
            r->tierx.enabled() || r->kv.loan_tokens() != 0 || r->program.encoder.modalities ||
            r->program.audio_encoder.modalities || r->program.pooled_buf != RAD_NULL_HANDLE ||
            r->program.is_pass_driven() || r->program.lane1_w1 >= 0) return RAD_E_UNSUPPORTED;
    return RAD_OK;
}

int Engine::kv_growth_layout() {
    if (!kv_growth_options_.enabled) return RAD_OK;
    RAD_TRY(kv_growth_scope(false));
    int64_t common_G = 0, common_pool = 0;
    for (auto& r : ranks_) {
        RAD_TRY(rad_dev_set(r->device));
        const int64_t G = r->pools.kv().backing_granule_bytes();
        const int64_t reserve = r->pools.kv().backing_reserved_bytes();
        if (!r->pools.kv().elastic() || G <= 0 || (common_G && G != common_G) ||
            (common_pool && reserve != common_pool) || reserve != kv_growth_options_.maximum_bytes) {
            RAD_ERR("%s rank%d requires explicit MAX_MIB=%lld for its actual rounded KV reservation "
                    "(logical %lld bytes, granule %lld); all four layouts must agree", kKVLiveEnv,
                    r->index, (long long)(reserve / kKVLiveMiB), (long long)r->pools.kv().bytes(),
                    (long long)G);
            (void)rad_dev_set(ranks_[0]->device);
            return RAD_E_UNSUPPORTED;
        }
        common_G = G; common_pool = reserve;
    }
    return rad_dev_set(ranks_[0]->device);
}

int Engine::kv_growth_drain() {
    /* Caller either has not started rank threads (startup) or owns scheduler admission with all
     * paired-barrier CPU issuers parked. None of these helpers calls Scheduler. */
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
    return rad_dev_set(ranks_[0]->device);
}

int Engine::kv_growth_start() {
    if (!kv_growth_options_.enabled) return RAD_OK;
    RAD_TRY(kv_growth_scope(false));
    RAD_TRY(kv_growth_drain());
    if (late_step_bytes_ < 0 || late_lazy_bytes_ < 0 ||
        late_step_bytes_ > std::numeric_limits<int64_t>::max() - late_lazy_bytes_ ||
        cfg_.gpu_headroom_mib < 0 || cfg_.gpu_headroom_mib > INT64_MAX / kKVLiveMiB)
        return RAD_E_INVAL;
    std::vector<KVLiveController::Rank> ranks;
    for (auto& r : ranks_)
        ranks.push_back({ r->device, &r->pools.kv(), &r->kv, &r->program });
    try {
        auto controller = std::make_unique<KVLiveController>();
        RAD_TRY(controller->start(kv_growth_options_, ranks, cfg_.gpu_headroom_mib * kKVLiveMiB,
                                  late_step_bytes_ + late_lazy_bytes_, late_lazy_bytes_));
        kv_growth_ = std::move(controller);
    } catch (...) { return RAD_E_NOMEM; }
    RAD_INFO("DS41_KV_GROW_READY min=%lld max=%lld warm=%lld hysteresis=%lld runtime_headroom=%lld "
             "fixed_experts=1 pipelines=0 cohort=TP4-seq1-noP1-noDraft-noTier-noMedia",
             (long long)kv_growth_options_.minimum_bytes, (long long)kv_growth_options_.maximum_bytes,
             (long long)kv_growth_options_.warm_bytes, (long long)kv_growth_options_.hysteresis_bytes,
             (long long)kv_growth_options_.runtime_headroom_bytes);
    return RAD_OK;
}

int Engine::kv_quiescent_plan(const std::function<int()>& change, const std::function<void()>& after_plan,
                             const RadBatch** out) {
    if (!out) return RAD_E_INVAL;
    *out = nullptr;
    if (!change) return RAD_E_INVAL;
    RAD_TRY(kv_growth_scope(true));

    return sched_.step_quiescent([&]() -> int {
        /* The sole step thread has not arrived at the next barrier, so parked ranks cannot
         * restart. HTTP cancel/reap/prefix changes wait on this ownership lock. None of the
         * helpers below calls Scheduler: CPU miss jobs ended before the rank join, RAM insert
         * helpers use only mover/cache locks, and GPU events need no host scheduler callback. */
        if (!barrier_->parked_for_scheduler()) return RAD_E_STATE;
        RAD_TRY(kv_growth_drain());
        /* May call the backing transaction with an already-held quiescence lease. It MUST NOT
         * call public Scheduler methods, prepare_release, prefill_hash_stage or tier_tick. */
        const int rc = change();
        const int restored = rad_dev_set(ranks_[0]->device);
        return rc < 0 ? rc : restored;
    }, out, after_plan);
}
}  // namespace rad
