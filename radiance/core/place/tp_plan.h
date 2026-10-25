/* Joint static initial MoE placement prototype. CPU-only metadata, no engine wiring.
 * Uses real Planner costs, profiles, spares and refusal rules without padding weights.
 * It does NOT coordinate subsequent heat/promotions/loans or validate TP execution. */
#pragma once
#include "planner.h"
#include <algorithm>
#include <climits>
#include <iterator>
#include <span>

namespace rad {
inline ExpertEligibility resident_experts(const Plan& plan) {
    ExpertEligibility out;
    for(const auto& u:plan.units)
        if(u.layer>=0&&u.expert>=0&&u.tier==Tier::VRAM)out.emplace(u.layer,u.expert);
    return out;
}
struct TpInitialPlanRound {
    std::vector<size_t> resident_counts;
    size_t intersection_count=0;
};
struct TpInitialPlanDiagnostics {
    int status=RAD_OK;
    std::string why;
    std::vector<int> rank_status;
    std::vector<TpInitialPlanRound> rounds;
    size_t planner_rounds_attempted=0;
    ExpertEligibility common_resident;
    bool initial_placement_only=true;
    bool heat_safe=false;
};

/* Output plans are committed together only on success; every input object/set is read-only.
 * Supports explicit expert_tiered TP worlds with identical canonical expert ownership.
 * Caller eligibility filters are refused rather than silently replaced. Profiles may differ:
 * each Planner preserves its own ranking, and their intersection is deliberately conservative.
 * This is a feasible initial-set search, not an optimality or performance guarantee. */
inline int coordinate_initial_expert_placement(std::span<const PlannerInput> inputs,
                                               std::vector<Plan>* output,
                                               TpInitialPlanDiagnostics* diagnostics=nullptr) {
    TpInitialPlanDiagnostics d;
    auto refuse=[&](int status,std::string why) {
        d.status=status;d.why=std::move(why);if(diagnostics)*diagnostics=d;return status;
    };
    if(!output||inputs.size()<2||inputs.size()>size_t(INT_MAX))
        return refuse(RAD_E_INVAL,"joint initial placement needs an output and at least two rank inputs");
    const int world=int(inputs.size());
    std::set<int> ranks;ExpertEligibility universe;
    for(size_t r=0;r<inputs.size();++r) {
        const auto& in=inputs[r];
        if(!in.program||!in.config||in.world_size!=world||in.rank<0||in.rank>=world||
           !ranks.insert(in.rank).second||in.program->ctx.rank!=in.rank||
           in.program->ctx.world_size!=world||in.n_profile<0||(in.n_profile>0&&!in.profile))
            return refuse(RAD_E_INVAL,"inconsistent/null TP rank, program, configuration or profile input");
        if(in.resident_eligibility)
            return refuse(RAD_E_INVAL,"coordinator owns the initial eligibility filter; caller filters are unsupported");
        Strategy strategy;
        if(strategy_parse(in.config->placement.c_str(),&strategy)!=RAD_OK||strategy!=Strategy::ExpertTiered)
            return refuse(RAD_E_INVAL,"joint initial placement requires explicit expert_tiered on every rank");
        ExpertEligibility keys;
        for(size_t w=1;w<in.program->weights.size();++w) {
            const auto& wi=in.program->weights[w];
            if(wi.decl.group.expert<0)continue;
            if(wi.decl.group.layer<0||wi.decl.access!=RAD_ACCESS_CONDITIONAL)
                return refuse(RAD_E_INVAL,"canonical experts must be conditional layer-owned movement units");
            keys.emplace(wi.decl.group.layer,wi.decl.group.expert);
        }
        if(keys.empty()||(r>0&&keys!=universe))
            return refuse(RAD_E_INVAL,"TP ranks must own exactly the same nonempty canonical expert universe");
        universe=std::move(keys);
    }

    ExpertEligibility eligible;bool restricted=false;
    for(;;) {
        ++d.planner_rounds_attempted;
        std::vector<Plan> plans(inputs.size());
        std::vector<ExpertEligibility> resident(inputs.size());
        d.rank_status.assign(inputs.size(),RAD_OK);int failure=RAD_OK;
        for(size_t r=0;r<inputs.size();++r) {
            auto in=inputs[r];if(restricted)in.resident_eligibility=&eligible;
            int status=Planner(in).plan(&plans[r]);
            if(status==RAD_OK&&!plans[r].fits())status=RAD_E_NOMEM;
            d.rank_status[r]=status;if(status<0&&failure==RAD_OK)failure=status;
            resident[r]=resident_experts(plans[r]);
        }
        if(failure<0)return refuse(failure,"a real rank Planner refused; no joint plans were published");
        ExpertEligibility common=resident.front();TpInitialPlanRound round;bool agree=true;
        for(const auto& set:resident) {
            round.resident_counts.push_back(set.size());
            if(set!=resident.front())agree=false;
            ExpertEligibility intersection;
            std::set_intersection(common.begin(),common.end(),set.begin(),set.end(),
                                  std::inserter(intersection,intersection.end()));
            common=std::move(intersection);
            if(restricted&&!std::includes(eligible.begin(),eligible.end(),set.begin(),set.end()))
                return refuse(RAD_E_STATE,"Planner produced an expert outside shared initial eligibility");
        }
        round.intersection_count=common.size();d.rounds.push_back(std::move(round));
        if(agree) {
            d.common_resident=std::move(common);d.why="static initial resident IDs agree; heat/loan execution is unvalidated";
            *output=std::move(plans);if(diagnostics)*diagnostics=std::move(d);return RAD_OK;
        }
        // After the first restriction, disagreement strictly shrinks eligibility.
        // A finite canonical universe therefore terminates without a heuristic limit.
        if(restricted&&common==eligible)
            return refuse(RAD_E_STATE,"shared eligibility failed to shrink despite resident-set disagreement");
        eligible=std::move(common);restricted=true;
    }
}
} // namespace rad
