/* Private matched measurement: no proposal, query epoch or probability observation exists.
 * This clock measures target-commit-bound consecutive opportunities, not a query cycle. */
#pragma once
#include "../rad_core.h"
#include <cmath>
#include <cstdint>

namespace rad {
inline constexpr char kDraftQueryFreeEnv[] = "RADIANCE_EXPERIMENTAL_DS41_DSPARK_QUERY_FREE";
inline bool draft_query_free_profile(int64_t seqs, bool tree, int from, int full, int depth,
                                     int short_rows, bool defer, bool native, bool deterministic,
                                     bool no_tbo) noexcept {
    return seqs == 1 && !tree && from == 1 && full == 5 && depth == 5 && short_rows == 2 &&
           defer && native && deterministic && no_tbo;
}
/* Check the admitted request, not the engine's placement/deterministic switch. No RNG draw is
 * assumed harmless: this first instrument refuses stochastic and stateful sampler extras. */
inline bool draft_query_free_request(const Request& r) noexcept {
    const SamplingParams& s = r.sp;
    return std::isfinite(s.temp) && s.greedy() && s.xtc_probability == 0 &&
           s.grammar.empty() && !s.grammar_lazy && s.grammar_triggers.empty() &&
           s.grammar_prefix.empty() && !s.grammar_program && s.thinking_budget < 0 &&
           !s.use_control_vector && s.ngram_ban_n == 0 && s.n_probs < 0 &&
           !r.media && r.lora_slot < 0 && !r.lora_uid && !r.embed && !r.score &&
           !r.token_scores && r.plugin_words.empty() && r.may_speculate && r.n_draft == 0;
}
struct DraftQueryFreeFrame {
    uint64_t seq = 0, boundary = 0, ready_ns = 0, wait_ns = 0;
    uint64_t output_before = 0, output_after = 0;
    int32_t step = -1, slot = -1, card = -1, input_ctx = 0, ctx = 0, token = -1;
    int64_t context_rows = 0;
    bool pure_decode = false;
};
inline bool draft_query_free_consecutive(const DraftQueryFreeFrame& a,
                                         const DraftQueryFreeFrame& b) noexcept {
    return a.seq && a.seq == b.seq && a.slot == b.slot && a.card == b.card &&
           a.boundary && a.boundary != UINT64_MAX && b.boundary == a.boundary + 1 &&
           a.step >= 0 && a.step < INT32_MAX && b.step == a.step + 1 &&
           a.pure_decode && b.pure_decode && b.ready_ns > a.ready_ns &&
           a.ctx == b.input_ctx && a.output_before != UINT64_MAX && a.output_after == a.output_before + 1 &&
           b.output_before == a.output_after;
}
struct DraftQueryFreeMeter {
    DraftQueryFreeFrame pending{}, current{}, closing{};
    uint64_t boundaries = 0, waits = 0, wait_all_ns = 0, context_rows = 0;
    uint64_t n = 0, ns = 0, wait_ns = 0, emitted = 0, transition = 0, end_cancel = 0, no_opportunity = 0;
    void pause() noexcept { pending = {}; current = {}; closing = {}; ++no_opportunity; }
    void ready(const DraftQueryFreeFrame& frame) noexcept {
        current = frame;
        closing = draft_query_free_consecutive(pending, frame) ? pending : DraftQueryFreeFrame{};
        ++boundaries; ++waits; wait_all_ns += frame.wait_ns;
        context_rows += (uint64_t)frame.context_rows;
    }
    bool committed(bool matched, uint64_t output_after) noexcept {
        if (!current.seq) return false;
        const bool ok = matched && current.output_before != UINT64_MAX && output_after == current.output_before + 1;
        const bool count = ok && closing.seq;
        if (!ok) ++end_cancel;
        else if (!count) ++transition;
        if (count) {
            ++n; ns += current.ready_ns - closing.ready_ns;
            wait_ns += current.wait_ns; ++emitted;
        }
        current.output_after = output_after;
        pending = ok ? current : DraftQueryFreeFrame{};
        current = {}; closing = {};
        return count;
    }
};
} // namespace rad
