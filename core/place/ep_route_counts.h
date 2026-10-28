/* Private counts contract for released DeepSeek native EP4, M<=16. No SDK ABI change. */
#pragma once
#include <cstdint>

namespace rad {

/* Private native EP producer/consumer domain. Larger M must fall back until
 * kernel bands and actual activation/placement bounds are independently admitted. */
inline constexpr int64_t kEpRouteMaxRows = 16;

struct EpRouteCounts {
    const void* program = nullptr;
    const int32_t* row = nullptr;
    void* stream = nullptr;
    uint64_t generation = 0, producer_serial = 0;
    int64_t layer = -1, rank = -1, world = 0;
    int64_t global_experts = 0, local_experts = 0, expert_base = -1;
    int64_t top_k = 0, rows = 0;
    bool successfully_issued = false;
};

inline bool ep_route_current(const EpRouteCounts& v, const void* program, void* stream,
                              uint64_t generation, int64_t layer, int64_t rows) {
    return program && v.program == program && v.row && v.stream == stream &&
           generation != 0 && v.generation == generation && v.producer_serial != 0 &&
           v.successfully_issued && v.layer == layer && layer >= 0 && layer < 40 &&
           v.world == 4 && v.rank >= 0 && v.rank < 4 &&
           v.global_experts == 384 && v.local_experts == 96 &&
           v.expert_base == v.rank * 96 && v.top_k == 6 &&
           rows > 0 && rows <= kEpRouteMaxRows && v.rows == rows;
}

/* Every count belongs to this rank's LOCAL numbering. Zero is complete local evidence,
 * not an incomplete global report. Unique global top-k implies one expert <=rows. */
inline bool ep_route_counts_valid(const EpRouteCounts& v, const int32_t* counts,
                                   int64_t capacity, int64_t* total) {
    if (!counts || !total || v.local_experts != 96 || capacity < 96 ||
        v.rows <= 0 || v.rows > kEpRouteMaxRows || v.top_k != 6) return false;
    int64_t sum = 0;
    for (int64_t e = 0; e < 96; ++e) {
        if (counts[e] < 0 || counts[e] > v.rows) return false;
        sum += counts[e];
        if (sum > v.rows * 6) return false;
    }
    *total = sum;
    return true;
}

inline bool ep_route_unit_selected(const int32_t* counts, int32_t local_expert,
                                    bool* selected) {
    if (!counts || !selected || local_expert < 0 || local_expert >= 96) return false;
    *selected = counts[local_expert] != 0;
    return true;
}

} // namespace rad
