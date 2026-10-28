/* Startup-only authority for the experimental bank-loaded OFF control. */
#pragma once
#include "dspark_native_scope.h"
#include "planner.h"
#include <array>
namespace rad {
inline bool dspark_draft_control_scope(const Program& prog, const Plan& plan,
                                       const Config& cfg, bool banks_bound) noexcept {
    if (!banks_bound || cfg.draft_bank.empty() || cfg.draft_bank_expert_placement != "host" ||
        cfg.n_spec != 5 || cfg.tp != 4 || cfg.max_seqs != 1 || cfg.kld() ||
        cfg.spec_ngram != 0 || cfg.draft_coupling || cfg.kv_cache_dtype != "bf16" ||
        cfg.draft_window != DraftWindow::Fixed || !cfg.debug_draft_oracle.empty() ||
        !dspark_native_program_scope(prog) || plan.world_size != 4 || plan.rank != prog.ctx.rank)
        return false;
    std::array<std::array<bool,32>,3> seen{};
    int count = 0;
    for (const auto& mu : plan.units) {
        if (mu.layer < 40 || mu.expert < 0) continue;
        if (mu.layer >= 43 || mu.expert >= 32 || mu.weights.size() != 2 ||
            mu.slot_bytes != 18800640 || mu.tier != Tier::HostPinned ||
            mu.site != Site::DeviceZeroCopy || !mu.host_forced || mu.movable) return false;
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
    return count == 96;
}
} // namespace rad
