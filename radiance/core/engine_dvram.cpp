/* engine_dvram.cpp -- RADIANCE_EXPERIMENTAL_DS41_DRAFT_VRAM (place/draft_vram.h): the DSpark draft
 * bank's hot experts as VRAM residents, the rest never loaded, and the draft router masked to the
 * residents.
 *
 *   draft_vram_open    before the planner: the switch, the hot-set file, and every rank's part,
 *                      each listed expert checked against the container's directory (the rank must
 *                      hold it); the planner then places them (Planner::place_draft_bank_vram)
 *   load_weights       publishes each parked draft expert at a resident's address on its card and
 *                      reads nothing for it (engine_bringup.cpp)
 *   draft_vram_mask    at the load's end: every rank's three router correction planes read back,
 *                      written with -inf at every expert no rank holds, and read back again
 *   draft_vram_credit  each credited draft routing histogram: the rank's selections of each stage
 *                      on its resident experts and on parked ones; the router replicated, a
 *                      selection of a parked expert would name weights the card never loaded
 *   draft_vram_close   after the rank threads stop: the last banks, and one line a rank
 *
 * The tally reads the histograms the core already copies back once a pass (rad_route_report); with
 * the heat engine off and no --expert-stats it drains the three draft layers' banks itself, once a
 * step on the rank thread, a few hundred bytes. */
#include "engine_priv.h"
#include "device/device.h"
#include "place/draft_vram_load.h"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace rad {

int Engine::draft_vram_open() {
    if (draft_bank::Config::parse(std::getenv(draft_bank::kLendEnv),
            std::getenv(draft_bank::kBackupEnv), std::getenv(draft_bank::kGraceEnv),
            std::getenv(draft_bank::kCooldownEnv), &draft_bank_config_) < 0) {
        RAD_ERR("DSBANKLEND invalid flag/budget/hysteresis: explicit flag=1 and positive BACKUP_MIB required");
        return RAD_E_INVAL;
    }
    int n = 0;
    const char* v = std::getenv(kDraftVramEnv);
    const char* file = std::getenv(kDraftHotsetEnv);
    if (draft_vram_switch(v, &n) < 0) {
        RAD_ERR("%s=%s: 1..32 keeps that many experts of each DSpark draft stage a rank in VRAM, the "
                "rest unloaded and masked out of the draft router; 0 or unset leaves the bank as "
                "--draft-bank-expert-placement places it", kDraftVramEnv, v);
        return RAD_E_INVAL;
    }
    if (n == 0) {
        if (draft_bank_config_.on) {
            RAD_ERR("DSBANKLEND requires an explicit native DRAFT_VRAM hot bank");
            return RAD_E_UNSUPPORTED;
        }
        if (file && *file) {
            RAD_ERR("%s=%s is set but %s is 0 or unset: the hot set would not be used", kDraftHotsetEnv,
                    file, kDraftVramEnv);
            return RAD_E_INVAL;
        }
        return RAD_OK;
    }
    if (!source_banks_ || !source_banks_->dual() || cfg_.draft_bank.empty() ||
        cfg_.draft_bank_expert_placement != "host" || cfg_.tp != 4 || ranks_.size() != 4) {
        RAD_ERR("%s=%d needs a bound DSpark bank (--draft-bank) with --draft-bank-expert-placement host, "
                "whose pinned pool it replaces, at --tp 4", kDraftVramEnv, n);
        return RAD_E_UNSUPPORTED;
    }
    if (!file || !*file) {
        RAD_ERR("%s=%d needs %s=<hot-set file> (tools/ds41-host-experts/draft_hotset.py writes one from "
                "--expert-stats captures)", kDraftVramEnv, n, kDraftHotsetEnv);
        return RAD_E_INVAL;
    }
    std::string why;
    if (const int st = draft_hotset_load(file, &draft_hotset_, &why); st < 0) {
        RAD_ERR("%s=%s refused: %s", kDraftHotsetEnv, file, why.c_str());
        return st;
    }
    if (const int st = draft_vram_allowed(draft_hotset_, n, &draft_vram_allowed_, &why); st < 0) {
        RAD_ERR("%s=%d over %s refused: %s", kDraftVramEnv, n, file, why.c_str());
        return st;
    }
    draft_vram_rank_.assign(ranks_.size(), DraftVramRank{});
    for (size_t ri = 0; ri < ranks_.size(); ++ri) {
        const Program& P = ranks_[ri]->program;
        /* Each draft unit's model-wide id: the container's directory entry of its weights, as
         * --expert-stats numbers them (Engine::xstats_open); every weight of a unit must agree. */
        constexpr int32_t kUnset = INT32_MIN;
        std::vector<std::vector<int32_t>> gid((size_t)kDraftStages,
                                              std::vector<int32_t>((size_t)kDraftRankExperts, kUnset));
        for (size_t w = 1; w < P.weights.size(); ++w) {
            const WeightInfo& wi = P.weights[w];
            const int32_t l = wi.decl.group.layer, e = wi.decl.group.expert;
            if (l < kDraftFirstLayer || l >= kDraftFirstLayer + kDraftStages || e < 0) continue;
            if (e >= kDraftRankExperts) {
                RAD_ERR("%s: rank %d declares draft expert %d of layer %d, past the bank's 32 a rank",
                        kDraftVramEnv, ranks_[ri]->index, (int)e, (int)l);
                return RAD_E_UNSUPPORTED;
            }
            const RadFileEntry* fe = source_entry(wi.source);
            const int32_t g = fe && fe->layer == l && fe->expert >= 0 ? fe->expert : -1;
            int32_t& at = gid[(size_t)(l - kDraftFirstLayer)][(size_t)e];
            at = at == kUnset ? g : at == g ? at : -1;
        }
        auto global_of = [&](int32_t layer, int32_t local) -> int32_t {
            const int32_t g = gid[(size_t)(layer - kDraftFirstLayer)][(size_t)local];
            return g == kUnset ? -1 : g;
        };
        if (const int st = draft_vram_rank(draft_hotset_, n, ranks_[ri]->index, global_of,
                                           &draft_vram_rank_[ri], &why); st < 0) {
            RAD_ERR("%s=%d over %s refused: %s", kDraftVramEnv, n, file, why.c_str());
            return st;
        }
    }
    draft_vram_n_ = n;
    draft_vram_tally_.assign(ranks_.size(), DraftVramTally{});
    int kept[kDraftStages] = {};
    for (int s = 0; s < kDraftStages; ++s)
        for (uint8_t a : draft_vram_allowed_[(size_t)s]) kept[s] += a;
    RAD_INFO("draft VRAM: %s=%d from %s: every rank keeps %d experts of each DSpark stage in VRAM (%d "
             "pairs, %s a rank, charged to the weights pool), leaves the other %d unloaded and pins "
             "nothing of the bank in the host pool; the draft router is masked to %d/%d/%d of %d experts "
             "(stages 40/41/42)", kDraftVramEnv, n, file, n, n * kDraftStages,
             humanb((int64_t)n * kDraftStages * kDraftPairBytes).c_str(),
             (kDraftRankExperts - n) * kDraftStages, kept[0], kept[1], kept[2], kDraftExperts);
    for (size_t ri = 0; ri < ranks_.size(); ++ri) {
        std::string lists;
        for (int s = 0; s < kDraftStages; ++s) {
            lists += fmt("%s%d [", s ? "; " : "", kDraftFirstLayer + s);
            for (int i = 0; i < n; ++i)
                lists += fmt("%s%d", i ? "," : "", draft_hotset_.hot[(size_t)s][(size_t)ranks_[ri]->index][(size_t)i]);
            lists += "]";
        }
        RAD_INFO("draft VRAM: rank %d keeps experts %s", ranks_[ri]->index, lists.c_str());
    }
    return RAD_OK;
}

int Engine::draft_vram_mask() {
    if (draft_vram_n_ <= 0) return RAD_OK;
    for (auto& r : ranks_) {
        LogRank lr(r->index);
        RAD_TRY(rad_dev_set(r->device));
        int kept[kDraftStages] = {}, nonzero = 0;
        std::string why;
        if (const int st = draft_vram_mask_planes(r->program, draft_vram_allowed_, kept, &nonzero, &why); st < 0) {
            RAD_ERR("draft VRAM: rank %d: the router's mask refused: %s", r->index,
                    why.empty() ? rad_strerror(st) : why.c_str());
            return st;
        }
        draft_vram_tally_[(size_t)r->index].masked_planes += kDraftStages;
        RAD_INFO("draft VRAM: rank %d: the router correction planes of stages 40/41/42 masked: %d/%d/%d of "
                 "%d experts keep their loaded value (%d of them nonzero), the rest -inf; read back exact",
                 r->index, kept[0], kept[1], kept[2], (int)kDraftExperts, nonzero);
    }
    if (ranks_.size() > 1) RAD_TRY(rad_dev_set(ranks_[0]->device));
    return RAD_OK;
}

void Engine::draft_vram_credit(Rank& r, int layer, const int32_t* h, int64_t n) {
    if (draft_vram_n_ <= 0 || !h || layer < kDraftFirstLayer || layer >= kDraftFirstLayer + kDraftStages ||
        (size_t)r.index >= draft_vram_tally_.size())
        return;
    const DraftVramRank& dv = draft_vram_rank_[(size_t)r.index];
    DraftVramTally& t = draft_vram_tally_[(size_t)r.index];
    const int s = layer - kDraftFirstLayer;
    ++t.hist[s];
    const int64_t m = std::min<int64_t>(n, kDraftRankExperts);
    for (int64_t e = 0; e < m; ++e) {
        if (h[e] <= 0) continue;
        if (dv.resident[(size_t)s][(size_t)e]) t.resident[s] += (uint64_t)h[e];
        else                                   t.parked[s] += (uint64_t)h[e];
    }
}

void Engine::draft_vram_drain(Rank& r, int before_step) {
    const int n_layer = r.ctx.n_route_layers();
    if ((int)r.heat_seen.size() != n_layer) r.heat_seen.assign((size_t)n_layer, -1);
    int8_t landed[kRouteBanks];
    std::memset(landed, -1, sizeof landed);
    const int last = std::min(n_layer, (int)(kDraftFirstLayer + kDraftStages));
    for (int l = kDraftFirstLayer; l < last; ++l) {
        int seen = r.heat_seen[(size_t)l];
        for (;;) {
            const int bank = r.ctx.next_histogram(l, seen, before_step, landed);
            if (bank < 0) break;
            const int32_t* h = r.ctx.last_histogram(l, bank);
            if (!h) break;
            const RadRouting* rt = r.ctx.last_routing(l);
            draft_vram_credit(r, l, h, rt ? std::min<int64_t>(rt->n_expert, r.ctx.route_n_expert()) : 0);
            seen = r.ctx.last_histogram_step(l, bank);
            r.ctx.mark_credited(l, bank);
        }
        r.heat_seen[(size_t)l] = seen;
    }
}

void Engine::draft_vram_close() {
    if (draft_vram_n_ <= 0) return;
    /* The last steps' banks are still in the ring: drain each rank's stream, then credit them (an
     * --expert-stats run's xstats_close has, through draft_vram_credit). Not after a step-loop
     * error, where a stream may never drain. */
    if (loop_status_.load(std::memory_order_acquire) >= 0 && !xstats_) {
        for (auto& r : ranks_) {
            if (r->heat.enabled()) continue;
            if (rad_dev_set(r->device) < 0 || rad_stream_sync(r->ctx.stream()) < 0) continue;
            draft_vram_drain(*r, INT_MAX);
        }
        if (ranks_.size() > 1) (void)rad_dev_set(ranks_[0]->device);
    }
    for (auto& r : ranks_) {
        if ((size_t)r->index >= draft_vram_tally_.size()) continue;
        LogRank lr(r->index);
        const DraftVramTally& t = draft_vram_tally_[(size_t)r->index];
        const uint64_t parked = t.parked[0] + t.parked[1] + t.parked[2];
        RAD_INFO("draft VRAM: rank %d at exit: %llu correction planes masked; draft histograms %llu/%llu/%llu "
                 "(stages 40/41/42), selections on its %d resident experts a stage %llu/%llu/%llu, on its "
                 "parked experts %llu/%llu/%llu%s", r->index, (unsigned long long)t.masked_planes,
                 (unsigned long long)t.hist[0], (unsigned long long)t.hist[1], (unsigned long long)t.hist[2],
                 draft_vram_n_, (unsigned long long)t.resident[0], (unsigned long long)t.resident[1],
                 (unsigned long long)t.resident[2], (unsigned long long)t.parked[0],
                 (unsigned long long)t.parked[1], (unsigned long long)t.parked[2],
                 parked ? " -- A PARKED EXPERT WAS ROUTED TO" : " (none: the mask held)");
    }
}

}  /* namespace rad */
