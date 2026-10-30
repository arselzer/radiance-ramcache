/* Startup-only authority for the experimental bank-loaded OFF control. */
#pragma once
#include "dspark_native_scope.h"
#include "planner.h"
#include <array>
namespace rad {
/* `draft_vram_n` > 0 (RADIANCE_EXPERIMENTAL_DS41_DRAFT_VRAM, place/draft_vram.h): the bank's experts
 * are not the 96 pinned pairs but N static VRAM residents of every stage, the rest parked on the
 * mapped tier (Planner::place_draft_bank_vram); the rest of the scope is the same. Any
 * --max-num-seqs: a step that drafts carries one sequence (sched/draft_conf.h, one sequence alone). */
inline bool dspark_draft_control_scope(const Program& prog, const Plan& plan,
                                       const Config& cfg, bool banks_bound,
                                       int draft_vram_n = 0) noexcept {
    if (!banks_bound || cfg.draft_bank.empty() || cfg.draft_bank_expert_placement != "host" ||
        cfg.n_spec != 5 || cfg.tp != 4 || cfg.max_seqs < 1 || cfg.kld() ||
        cfg.spec_ngram != 0 || cfg.draft_coupling || cfg.kv_cache_dtype != "bf16" ||
        cfg.draft_window != DraftWindow::Fixed || !cfg.debug_draft_oracle.empty() ||
        !dspark_native_program_scope(prog) || plan.world_size != 4 || plan.rank != prog.ctx.rank ||
        draft_vram_n < 0 || draft_vram_n > 32)
        return false;
    std::array<std::array<bool,32>,3> seen{};
    std::array<int,3> hot{};
    int count = 0;
    for (const auto& mu : plan.units) {
        if (mu.layer < 40 || mu.expert < 0) continue;
        if (mu.layer >= 43 || mu.expert >= 32 || mu.weights.size() != 2 || mu.movable) return false;
        if (draft_vram_n == 0) {
            if (mu.slot_bytes != 18800640 || mu.tier != Tier::HostPinned ||
                mu.site != Site::DeviceZeroCopy || !mu.host_forced) return false;
        } else if (mu.draft_vram) {
            if (mu.draft_parked || mu.host_forced || mu.slot_bytes != 18800640 ||
                mu.tier != Tier::VRAM || mu.site != Site::Device) return false;
            ++hot[size_t(mu.layer-40)];
        } else if (!mu.draft_parked || mu.host_forced || mu.tier != Tier::Mapped || mu.slot_bytes != 0) {
            return false;
        }
        bool& entry = seen[size_t(mu.layer-40)][size_t(mu.expert)];
        if (entry) return false;
        entry = true; ++count;
        unsigned slots = 0;
        for (auto h : mu.weights) {
            if (!h || size_t(h) >= prog.weights.size()) return false;
            const auto& w = prog.weights[size_t(h)];
            if (!dspark_native_weight_scope(w) || w.decl.group.layer != mu.layer ||
                w.decl.group.expert != mu.expert) return false;
            const unsigned bit = 1u << w.decl.group.slot;
            if (slots & bit) return false;
            slots |= bit;
        }
        if (slots != 3) return false;
    }
    if (draft_vram_n > 0)
        for (int s : hot) if (s != draft_vram_n) return false;
    return count == 96;
}
} // namespace rad
