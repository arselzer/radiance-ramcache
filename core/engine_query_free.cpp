/* Default-off fixed0 query-cost control. Target IDs use the existing collect_issue readback;
 * context passes, snapshots, reservations and the before-ahead step_end event are unchanged. */
#include "engine_priv.h"
#include "device/device.h"
#include <climits>

namespace rad {
int Engine::dnq_ready(const RadBatch* batch, const StepOut& so) {
    const auto refused = [&](const char* reason) {
        RAD_ERR("DSpark query-free refused: stage=ready reason=%s step=%d seq=%llu "
                "block_rows=%d first_rows=%llu readback_rows=%d boundary=%llu",
                reason, batch ? batch->step : -1, (unsigned long long)dnq_candidate_.seq,
                block_rows_, (unsigned long long)so.first_row.size(), so.n_rows,
                (unsigned long long)dnq_boundary_);
        return RAD_E_STATE;
    };
    if (!draft_experiment_.query_free() || !batch || block_rows_ != 0) return refused("mode_batch_rows");
    if (dquery_copy_.pending() || dquery_copy_.started || dquery_copy_.observer_calls || dquery_copy_.slot_calls)
        return refused("query_copy"); // NQ cannot carry even a pending query-copy frame
    if (!dnq_candidate_.seq) { dnq_.pause(); return RAD_OK; }
    if (dnq_candidate_.step != batch->step || so.first_row.size() != 1 ||
        so.first_row[0] != 0 || !so.rb || so.n_rows != 1 || dnq_boundary_ == UINT64_MAX)
        return refused("target_binding");
    const uint64_t begin = rad_mono_ns();
    RAD_TRY(rad_event_sync(step_end_)); // same opportunity boundary as matched fixed0Q
    const uint64_t ready = rad_mono_ns();
    dnq_candidate_.boundary = ++dnq_boundary_; // deliberately NOT a proposal/query epoch
    dnq_candidate_.ready_ns = ready; dnq_candidate_.wait_ns = ready - begin;
    dnq_candidate_.token = so.rb[so.first_row[0]];
    if (!dnq_candidate_.boundary || dnq_candidate_.token < 0) return refused("target_token");
    dnq_.ready(dnq_candidate_);
    if (dnq_boundary_ == 1)
        RAD_INFO("DSpark query-free first: seq=%llu step=%d boundary=1 slot=%d card=%d "
                 "query_rows=0 observer_rows=0 query_epochs=0 context_rows=%lld wait=step_end",
                 (unsigned long long)dnq_candidate_.seq, dnq_candidate_.step,
                 dnq_candidate_.slot, dnq_candidate_.card, (long long)dnq_candidate_.context_rows);
    return RAD_OK;
}

int Engine::dnq_finish(const StepOut& so) {
    const auto refused = [&](const char* reason) {
        RAD_ERR("DSpark query-free refused: stage=finish reason=%s seq=%llu boundary=%llu "
                "tokens=%llu accepted_rows=%llu",
                reason, (unsigned long long)dnq_.current.seq,
                (unsigned long long)dnq_.current.boundary,
                (unsigned long long)so.token.size(), (unsigned long long)so.accepted.size());
        return RAD_E_STATE;
    };
    if (!draft_experiment_.query_free()) return refused("mode");
    if (!dnq_.current.seq) return RAD_OK;
    if (so.token.size() != 1 || so.accepted.size() != 1 || so.accepted[0] != 0 ||
        so.token[0] != dnq_.current.token) return refused("target_commit");
    uint64_t output = 0;
    const bool matched = sched_.query_free_committed(dnq_.current, &output);
    dnq_.committed(matched, output); // terminal/cancel/reuse can never close a cost cycle
    dnq_candidate_ = {};
    return RAD_OK;
}

void Engine::dnq_report() const {
    const DraftQueryFreeMeter& s = dnq_;
    const std::string query_copy = dquery_copy_summary();
    RAD_INFO("DSpark query-free totals: opportunities=%llu waits=%llu wait_ns=%llu context_rows=%llu "
             "query_issues=0 query_rows=0 observers=0 query_epochs=0 publications=0 "
             "transition=%llu no_opportunity=%llu end_cancel=%llu unclosed_boundary=%llu%s",
             (unsigned long long)s.boundaries, (unsigned long long)s.waits,
             (unsigned long long)s.wait_all_ns, (unsigned long long)s.context_rows,
             (unsigned long long)s.transition, (unsigned long long)s.no_opportunity,
             (unsigned long long)s.end_cancel, (unsigned long long)s.pending.boundary, query_copy.c_str());
    RAD_INFO("DSpark query-free costs: label=C0NQ n=%llu cycle_ns=%llu wait_ns=%llu emitted=%llu "
             "mean_ms=%.6f; consecutive target-commit-bound pure-decode opportunities only; "
             "context retained, early wait retained, no query/observer/query readback",
             (unsigned long long)s.n, (unsigned long long)s.ns, (unsigned long long)s.wait_ns,
             (unsigned long long)s.emitted, s.n ? (double)s.ns / (double)s.n * 1e-6 : 0.);
}
} // namespace rad
