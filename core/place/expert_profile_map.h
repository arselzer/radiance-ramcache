/* Translate model-wide popularity keys to a Program's movement-group keys.
 * The source directory is authoritative: no expert-name parsing or EP rank arithmetic.
 * Routing/table/count indices remain local and unchanged. Header-only for CPU metadata tests.
 */
#pragma once
#include "rad_format.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <new>
#include <utility>
#include <vector>

namespace rad {

enum class ExpertProfileMapStatus {
    Complete, InvalidArguments, MissingSource, InvalidSource, ConflictingGroup,
    AmbiguousSource, InvalidProfile, NoMemory
};
inline const char* expert_profile_map_status(ExpertProfileMapStatus status) noexcept {
    switch (status) {
        case ExpertProfileMapStatus::Complete: return "complete";
        case ExpertProfileMapStatus::InvalidArguments: return "invalid profile arguments";
        case ExpertProfileMapStatus::MissingSource: return "required expert source is absent";
        case ExpertProfileMapStatus::InvalidSource: return "source expert key is invalid";
        case ExpertProfileMapStatus::ConflictingGroup: return "one local group names different source experts";
        case ExpertProfileMapStatus::AmbiguousSource: return "one source expert names different local groups";
        case ExpertProfileMapStatus::InvalidProfile: return "profile key or fraction is invalid";
        case ExpertProfileMapStatus::NoMemory: return "profile map allocation failed";
    }
    return "unknown profile map status";
}

/* Weights is the actual Program::weights vector; find(source) returns its RadFileEntry.
 * The minimal field contract also permits standalone fixtures using this same implementation.
 * n_trunk/n_model_experts bound the known trunk domain; optional draft layers retain the
 * directory's nonnegative bounded ids. Every contributing code/scale/GU/DN declaration must
 * agree on the same source expert. Slot/plane ids need not match the directory's tensor slot.
 * Output is transactional on refusal and shares are never renormalized per EP rank.
 */
template <class Weights, class Find>
ExpertProfileMapStatus local_expert_profile(const Weights& weights, Find find,
                                            const RadFileProfile* profile, int64_t n_profile,
                                            int64_t n_trunk, int64_t n_model_experts,
                                            std::vector<RadFileProfile>* output) {
    using Status = ExpertProfileMapStatus;
    if (!output || n_profile < 0 || (n_profile && !profile) || n_trunk < 0 ||
        n_model_experts < 0 || (n_trunk && !n_model_experts)) return Status::InvalidArguments;
    if (!n_profile) { output->clear(); return Status::Complete; }
    constexpr int32_t kMaxId = 1 << 20;  // same bounded key domain as upstream XsMap
    auto key_valid = [&](int32_t layer, int32_t expert) {
        return layer >= 0 && layer < kMaxId && expert >= 0 && expert < kMaxId &&
               (layer >= n_trunk || expert < n_model_experts);
    };
    try {
        using Key = std::pair<int32_t, int32_t>;
        std::map<Key, Key> forward, reverse;
        for (std::size_t i = 1; i < weights.size(); ++i) {
            const auto& weight = weights[i];
            const int32_t layer = weight.decl.group.layer, expert = weight.decl.group.expert;
            if (expert < 0) continue;
            if (!key_valid(layer, expert)) return Status::InvalidSource;
            const RadFileEntry* source = find(weight.source);
            if (!source) {
                if (weight.decl.optional) continue;  // the loader also skips an absent optional
                return Status::MissingSource;
            }
            if (!key_valid(source->layer, source->expert) || source->layer != layer)
                return Status::InvalidSource;
            const Key local{layer, expert}, global{source->layer, source->expert};
            const auto a = forward.emplace(local, global);
            if (!a.second && a.first->second != global) return Status::ConflictingGroup;
            const auto b = reverse.emplace(global, local);
            if (!b.second && b.first->second != local) return Status::AmbiguousSource;
        }
        std::vector<RadFileProfile> translated;
        translated.reserve(reverse.size());
        for (int64_t i = 0; i < n_profile; ++i) {
            const auto& row = profile[i];
            if (!key_valid(row.layer, row.expert) || !std::isfinite(row.share) ||
                row.share < 0 || row.share > 1) return Status::InvalidProfile;
            const auto at = reverse.find(Key{row.layer, row.expert});
            if (at == reverse.end()) continue;  // another rank, or an undeclared optional expert
            auto local = row;
            local.layer = at->second.first;
            local.expert = at->second.second;
            translated.push_back(local);  // preserve order/duplicates: Planner's last row wins
        }
        output->swap(translated);
        return Status::Complete;
    } catch (const std::bad_alloc&) {
        return Status::NoMemory;
    }
}

}  // namespace rad
