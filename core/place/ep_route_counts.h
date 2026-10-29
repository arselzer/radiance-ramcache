/* Private counts contract for released DeepSeek native EP4, M<=16 (M<=48 under
 * RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL=1, as far as the step goes; M<=C under
 * RADIANCE_EXPERIMENTAL_DS41_LCP=C). No SDK ABI change. */
#pragma once
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace rad {

/* Private native EP producer/consumer domain. Larger M must fall back until
 * kernel bands and actual activation/placement bounds are independently admitted. */
inline constexpr int64_t kEpRouteMaxRows = 16;

/* RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL=1 admits passes of up to kEpRouteTallRows tokens, as tall as
 * --max-num-batched-tokens lets them be, and only when each gives a 16-token pass's bits: libr4d
 * keeps the routed GEMMs on their decode-band kernel (the twin row moe_gemm_ggml_mxfp4_gemv_tall,
 * bounded at the tallest pass libr4d keeps every ds41 kernel exact at), the sparse attention and the
 * indexer score on their exact scalar forms, and the norms on their 32-row width
 * (libr4d/r4d_ds41_tall.h). The CPU path's rows are the gemv's bits at any height. FileStager::
 * tall_init checks the bands the step resolved and refuses otherwise. The counts witness admits
 * ep_route_max_rows(); the file stager sizes the CPU path's token rows, sorted order, output rows
 * and fetch lists by its own limit (the step, at least 16). */
inline constexpr int64_t kEpRouteTallRows = 48;
inline constexpr const char* kExactTallEnv = "RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL";

namespace detail {
/* Tests only (ds41_exact_tall_set_for_test): -2 reads the environment. */
inline std::atomic<int> g_exact_tall_forced{ -2 };
}  // namespace detail

/* The switch: 1 on, 0 off (unset, empty or "0"), -1 any other value (refused at FileStager::init).
 * The environment is read once. */
inline int ds41_exact_tall_state() {
    const int forced = detail::g_exact_tall_forced.load(std::memory_order_relaxed);
    if (forced != -2) return forced;
    static const int from_env = [] {
        const char* e = std::getenv(kExactTallEnv);
        if (!e || !*e || !std::strcmp(e, "0")) return 0;
        return std::strcmp(e, "1") == 0 ? 1 : -1;
    }();
    return from_env;
}
/* Tests: the switch as `state` (1, 0 or -1) until called again with -2, which goes back to the
 * environment. Set before anything sized by the limit is made. */
inline void ds41_exact_tall_set_for_test(int state) {
    detail::g_exact_tall_forced.store(state, std::memory_order_relaxed);
}
/* ---- large-chunk prefill: RADIANCE_EXPERIMENTAL_DS41_LCP=<C> (FileStager::lcp_init) ----------------
 *
 * Prefill in chunks of C tokens (a multiple of 16 in [kEpRouteLcpMin, kEpRouteLcpMax]), each giving
 * the bits of the same tokens in 16-token passes: libr4d keeps EXACT_TALL's pins at every height (its
 * twin row reaches kEpRouteLcpMax, libr4d/r4d_ds41_tall.h), and the core admits a step only when the
 * bands it resolved say so (lcp_init: the routed GEMMs on the gemv rows to C, every other op on one
 * kernel or on a pair known to give the same bits), --max-num-batched-tokens is C, the chunk geometry
 * makes chunks of C, and the CPU computes every routed unit of every pass. The counts witness then
 * admits passes of up to C tokens. */
inline constexpr int64_t kEpRouteLcpMin = 64;
inline constexpr int64_t kEpRouteLcpMax = 2048;
inline constexpr const char* kLcpEnv = "RADIANCE_EXPERIMENTAL_DS41_LCP";

namespace detail {
/* Tests only (ds41_lcp_set_for_test): -2 reads the environment. */
inline std::atomic<int64_t> g_lcp_forced{ -2 };
/* The ranks whose file stager admitted LCP (FileStager::lcp_init and hx_init): the scheduler refuses
 * LCP unless every rank's did (lcp_admitted_ranks). */
inline std::atomic<int> g_lcp_admitted{ 0 };
}  // namespace detail

/* The switch: C when it is a multiple of 16 in [kEpRouteLcpMin, kEpRouteLcpMax], 0 off (unset,
 * empty or "0"), -1 any other value (refused at FileStager::init). The environment is read once. */
inline int64_t ds41_lcp_state() {
    const int64_t forced = detail::g_lcp_forced.load(std::memory_order_relaxed);
    if (forced != -2) return forced;
    static const int64_t from_env = [] () -> int64_t {
        const char* e = std::getenv(kLcpEnv);
        if (!e || !*e || !std::strcmp(e, "0")) return 0;
        char* end = nullptr;
        const long long v = std::strtoll(e, &end, 10);
        if (end == e || *end || v < kEpRouteLcpMin || v > kEpRouteLcpMax || v % 16) return -1;
        return (int64_t)v;
    }();
    return from_env;
}
/* Tests: the switch as `state` (C, 0 or -1) until called again with -2, which goes back to the
 * environment. Set before anything sized by the limit is made. */
inline void ds41_lcp_set_for_test(int64_t state) {
    detail::g_lcp_forced.store(state, std::memory_order_relaxed);
}
/* C, or 0 without (a valid) LCP. */
inline int64_t ds41_lcp_rows() {
    const int64_t s = ds41_lcp_state();
    return s > 0 ? s : 0;
}
inline int  lcp_admitted_ranks() { return detail::g_lcp_admitted.load(std::memory_order_acquire); }
inline void lcp_note_admitted() { detail::g_lcp_admitted.fetch_add(1, std::memory_order_acq_rel); }
inline void lcp_admitted_reset_for_test(int ranks) { detail::g_lcp_admitted.store(ranks, std::memory_order_release); }

/* The largest pass the counts contract admits: C under LCP, kEpRouteTallRows with EXACT_TALL, else
 * kEpRouteMaxRows. */
inline int64_t ep_route_max_rows() {
    if (const int64_t c = ds41_lcp_rows()) return c;
    return ds41_exact_tall_state() == 1 ? kEpRouteTallRows : kEpRouteMaxRows;
}

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
           rows > 0 && rows <= ep_route_max_rows() && v.rows == rows;
}

/* Every count belongs to this rank's LOCAL numbering. Zero is complete local evidence,
 * not an incomplete global report. Unique global top-k implies one expert <=rows. */
inline bool ep_route_counts_valid(const EpRouteCounts& v, const int32_t* counts,
                                   int64_t capacity, int64_t* total) {
    if (!counts || !total || v.local_experts != 96 || capacity < 96 ||
        v.rows <= 0 || v.rows > ep_route_max_rows() || v.top_k != 6) return false;
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
