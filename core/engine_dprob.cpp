/* Read-only observations, fixed matched controls and a separately opt-in prefix policy. The architecture
 * op runs after unchanged selection. Copy behind it, retire after step_end_, and pair with the
 * same StepSlot's ID/position before the proposal's verification can commit. */
#include "engine_priv.h"
#include "device/device.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace rad {
int Engine::dpolicy_concurrent_open() {
    if (!dpolicy_concurrent_) return RAD_OK;
    if (!draft_experiment_.prefix_policy().on || cfg_.max_seqs <= 1 || cfg_.tp != 4 ||
        ranks_.size() != 4 || !sched_.draft_solo() || !sched_.ctx_deferred() || tree_ ||
        !block_drafting() || !draft_experiment_.ctx_defer()) return RAD_E_UNSUPPORTED;
    RAD_INFO("DSpark prefix policy concurrent: {\"max_seqs\":%lld,\"tp\":4,\"dsolo\":true,"
             "\"deferred\":true,\"query_rows\":2,\"full_rows\":5,\"depth\":5,\"from\":1,"
             "\"tbo\":false,\"native_bank\":true,\"bank_hot_n\":%d}",
             (long long)cfg_.max_seqs, draft_vram_n_);
    return RAD_OK;
}

void Engine::dpolicy_pause(int step, const char* reason) {
    if (!dpolicy_concurrent_) return;
    const DraftMeasureFrame previous = dmeasure_.pending;
    dmeasure_.no_query(); // clears a pending cost even if this is an idle rather than model frame
    if (!previous.epoch) return;
    dpolicy_pause_seq_ = previous.seq; dpolicy_pause_epoch_ = previous.epoch;
    dpolicy_paused_ = true;
    ++dpolicy_span_;
    RAD_INFO("DSpark prefix policy pause: {\"seq\":%llu,\"epoch\":%llu,\"step\":%d,"
             "\"frame\":%llu,\"previous_span\":%llu,\"next_span\":%llu,\"reason\":\"%s\",\"query_rows\":0}",
             (unsigned long long)previous.seq, (unsigned long long)previous.epoch, step,
             (unsigned long long)(step < 0 ? 0 : dpolicy_frame_),
             (unsigned long long)previous.solo_span, (unsigned long long)dpolicy_span_, reason);
}

int Engine::dprob_init() {
    if (!draft_experiment_.prob_capture() && !draft_experiment_.early_boundary()) return RAD_OK;
    const bool policy = draft_experiment_.prefix_policy().on;
    const char* instrument = policy ? rad_dflash_observe::kPolicyEnv : draft_experiment_.early_verify() >= 0
                           ? rad_dflash_observe::kEarlyEnv : rad_dflash_observe::kEnv;
    if (!block_drafting() || tree_ || ranks_.empty() || !drafter_.name ||
        std::strcmp(drafter_.name, "dspark")) {
        RAD_ERR("%s requires DSpark's chain block drafter", instrument);
        return RAD_E_UNSUPPORTED;
    }
    const Program& p = ranks_[0]->program;
    for (size_t i = 1; i < p.buffers.size(); ++i) {
        const std::string& name = p.buffers[i].name;
        const size_t len = std::strlen(rad_dflash_observe::kBuffer);
        if (name.size() >= len && name.compare(name.size() - len, len, rad_dflash_observe::kBuffer) == 0)
            dprob_buf_ = (rad_buf)i;
    }
    if (!dprob_buf_) {
        RAD_ERR("%s: architecture declares no proposal observations; rebuild all plugins together",
                instrument);
        return RAD_E_UNSUPPORTED;
    }
    const RadBufDecl& d = p.buffers[dprob_buf_].decl;
    if (d.kind != RAD_BUF_PERSIST || d.dtype != RAD_F32 || d.rank != 2 ||
        d.shape[0] < cfg_.max_seqs || d.shape[1] != rad_dflash_observe::kWidth)
        return RAD_E_SHAPE;
    for (const OpInfo& op : p.ops)
        if (op.op == "dflash_observe") {
            long long candidates = 0;
            if (op.base.get_i("top_k", &candidates)) dprob_candidates_ = (int)candidates;
        }
    if (dprob_candidates_ < 1 || dprob_candidates_ > 32) return RAD_E_STATE;
    dprob_rows_ = d.shape[0];
    dprob_host_ = static_cast<float*>(rad_dev_alloc(
        dprob_rows_ * rad_dflash_observe::kWidth * (int64_t)sizeof(float), RAD_MEM_HOST_PINNED));
    if (!dprob_host_) return RAD_E_NOMEM;
    if (draft_experiment_.prob_capture() && !policy)
        RAD_INFO("DSpark read-only proposal capture: on (%s); depths 1/2, candidate_p is softmax at "
             "temperature 1 over the existing %d Markov-conditioned candidates, NOT full-vocabulary "
             "or acceptance probability; score and margin logged; no proposal, verification or "
             "depth decision reads it", rad_dflash_observe::kEnv, dprob_candidates_);
    if (policy && draft_experiment_.prob_capture())
        RAD_INFO("DSpark read-only proposal capture: on (%s); policy action telemetry is separate; "
                 "candidate_p is candidate-set share, never target acceptance probability",
                 rad_dflash_observe::kEnv);
    if (draft_experiment_.early_boundary()) {
        dmeasure_ = {};
        dpolicy_concurrent_ = policy && cfg_.max_seqs > 1;
        dpolicy_paused_ = false;
        dpolicy_frame_ = 0; dpolicy_span_ = 1;
        dpolicy_pause_seq_ = dpolicy_pause_epoch_ = 0;
        dmeasure_query_rows_ = (int)(short_from_ == 1 && drafter_short_ > 0 ? drafter_short_ : drafter_.block);
        if (dmeasure_query_rows_ < 2 || dmeasure_query_rows_ > 5) return RAD_E_UNSUPPORTED;
        if (policy)
            RAD_INFO("DSpark prefix policy: on config=%s (%s); query_rows=2 unchanged at every action, "
                     "observer=1, wait=step_end before issue_ahead; candidate_p=candidate-set-share, "
                     "invalid_numeric=full-prefix2; action/cycle logging=on, fixed_measurement=off",
                     draft_experiment_.prefix_policy().spec, rad_dflash_observe::kPolicyEnv);
        else RAD_INFO("DSpark matched early verify: on k=%d (%s); query_rows=%d unchanged at every action, "
                 "observer=1, wait=step_end before issue_ahead, logging=%s; fixed measurement, no probability policy",
                 draft_experiment_.early_verify(), rad_dflash_observe::kEarlyEnv, dmeasure_query_rows_,
                 draft_experiment_.prob_capture() ? "capture" : "quiet");
    }
    return RAD_OK;
}

void Engine::dprob_fini() {
    if (dprob_host_) rad_dev_free(dprob_host_, RAD_MEM_HOST_PINNED);
    dprob_host_ = nullptr; dprob_buf_ = 0; dprob_rows_ = 0;
}

int Engine::dprob_issue(int n) {
    if (!dprob_host_ || n <= 0) return RAD_OK;
    if (n > dprob_rows_) return RAD_E_SHAPE;
    Rank& r0 = *ranks_[0];
    const void* src = r0.ctx.buf_ptr(dprob_buf_);
    if (!src) return RAD_E_STATE;
    return rad_memcpy_async(dprob_host_, src,
                           (int64_t)n * rad_dflash_observe::kWidth * (int64_t)sizeof(float), r0.ctx.stream());
}

void Engine::dprob_finish(int row, const int32_t* tokens, int n, bool published) {
    if (!dprob_host_ || !draft_experiment_.prob_capture()) return;
    const Scheduler::BlockRow& b = block_row_[(size_t)row];
    const StepSlot& s = slots_host_[b.card];
    sched_.draft_prob_publish(b.seq, s.ctx, s.tok[0], tokens, n,
                              dprob_host_ + (int64_t)row * rad_dflash_observe::kWidth,
                              published, dprob_candidates_, draft_experiment_.early_boundary()
                                  ? dmeasure_.current.epoch : 0);
}

/* Called AFTER mark_step_end and BEFORE issue_ahead. Existing step_end covers both the observation
 * and complete StepSlot copies. No new event, stream synchronization, or host wait on the off path.
 * Later collect/so.early waits are deliberately retained (already complete in this branch). */
int Engine::dmeasure_ready(const RadBatch* batch) {
    if (block_rows_ == 0) {
        if (dpolicy_concurrent_) dpolicy_pause(batch->step, "no_query");
        else dmeasure_.no_query();
        return RAD_OK;
    }
    if (block_rows_ != 1 || block_steps_ != dmeasure_query_rows_ || !dprob_host_) return RAD_E_STATE;
    if (dpolicy_concurrent_) {
        uint64_t observed = 0;
        if (!sched_.policy_block_owner(batch->step, block_row_.data(), block_rows_, &observed) ||
            observed != dpolicy_frame_ || !dpolicy_span_) return RAD_E_STATE;
    }
    const uint64_t begin = rad_mono_ns();
    RAD_TRY(rad_event_sync(step_end_));
    const uint64_t ready = rad_mono_ns();
    const Scheduler::BlockRow& row = block_row_[0];
    const StepSlot& slot = slots_host_[row.card];
    const bool policy = draft_experiment_.prefix_policy().on;
    const char* label = policy ? "DSpark prefix policy" : "DSpark matched early verify";
    bool previous_matches = false;
    if (!sched_.measure_block_binding(row, slot.ctx, slot.acc, dmeasure_.pending, &previous_matches)) {
        RAD_ERR("%s: sequence/slot/ctx binding failed for sequence %llu", label, (unsigned long long)row.seq);
        return RAD_E_STATE;
    }
    uint32_t invalid_depths = 0;
    for (int j = 0; j < rad_dflash_observe::kDepths; ++j) {
        const float* obs = dprob_host_ + j * rad_dflash_observe::kFields;
        const DraftMeasureSignal signal = draft_measure_signal(obs, slot.tok[j + 1], slot.tok[j]);
        if (signal == DraftMeasureSignal::PathMismatch) {
            RAD_ERR("%s: echoed observation/path IDs differ at depth %d", label, j + 1);
            return RAD_E_STATE;
        }
        if (signal == DraftMeasureSignal::InvalidNumeric) invalid_depths |= 1u << j;
    }
    DraftMeasureFrame frame;
    frame.seq = row.seq; frame.card = row.card; frame.step = batch->step;
    frame.epoch = sched_.draft_prob_reserve_epoch();
    frame.ctx = slot.ctx; frame.anchor = slot.tok[0];
    if (dpolicy_concurrent_) {
        frame.query_frame = dpolicy_frame_; frame.solo_span = dpolicy_span_; frame.owner_slot = row.slot;
    }
    frame.k = draft_experiment_.early_verify(); frame.query_rows = dmeasure_query_rows_;
    bool fallback = false;
    const float p1 = dprob_host_[rad_dflash_observe::kCandidateP];
    const float p2 = dprob_host_[rad_dflash_observe::kFields + rad_dflash_observe::kCandidateP];
    if (policy) frame.k = draft_experiment_.prefix_policy().select(p1, p2, invalid_depths, &fallback);
    frame.invalid_depths = invalid_depths;
    frame.ready_ns = ready; frame.wait_ns = ready - begin;
    std::copy(slot.tok + 1, slot.tok + 1 + block_steps_, frame.ids);
    const StepPlan& plan = sched_.plan();
    frame.pure_decode = plan.phase == RAD_PHASE_DECODE && plan.e.size() == 1 &&
                        !plan.e[0].is_prefill && !plan.e[0].ngram && plan.e[0].produces_token;
    dmeasure_.ready(frame, previous_matches, policy, dpolicy_concurrent_);
    if (policy) {
        /* JSON stays valid for nonfinite observations; invalid_depths identifies the fallback.
         * This action belongs to the complete Db2 query, not to a subsequently reused slot. */
        const std::string p1_json = std::isfinite(p1) ? fmt("%.17g", (double)p1) : "null";
        const std::string p2_json = std::isfinite(p2) ? fmt("%.17g", (double)p2) : "null";
        const std::string concurrent = dpolicy_concurrent_ ? fmt(",\"frame\":%llu,\"solo_span\":%llu,\"owner_slot\":%d",
            (unsigned long long)frame.query_frame, (unsigned long long)frame.solo_span, frame.owner_slot) : "";
        RAD_INFO("DSpark prefix policy action: {\"seq\":%llu,\"epoch\":%llu,\"step\":%d,\"card\":%d,"
                 "\"ctx\":%d,\"anchor\":%d,\"query_rows\":%d,\"ids\":[%d,%d],\"k\":%d,"
                 "\"candidate_p\":[%s,%s],\"invalid_depths\":%u,\"fallback\":%s,\"wait_ns\":%llu,"
                 "\"pure_decode\":%s%s}",
                 (unsigned long long)frame.seq, (unsigned long long)frame.epoch, frame.step, frame.card,
                 frame.ctx, frame.anchor, frame.query_rows, frame.ids[0], frame.ids[1], frame.k,
                 p1_json.c_str(), p2_json.c_str(), (unsigned)invalid_depths, fallback ? "true" : "false",
                 (unsigned long long)frame.wait_ns, frame.pure_decode ? "true" : "false", concurrent.c_str());
        if (dpolicy_concurrent_ && dpolicy_paused_) {
            RAD_INFO("DSpark prefix policy resume: {\"seq\":%llu,\"epoch\":%llu,\"step\":%d,"
                     "\"frame\":%llu,\"solo_span\":%llu,\"card\":%d,\"owner_slot\":%d,\"ctx\":%d,"
                     "\"previous_seq\":%llu,\"previous_epoch\":%llu,\"query_rows\":2,\"ready\":true}",
                     (unsigned long long)frame.seq, (unsigned long long)frame.epoch, frame.step,
                     (unsigned long long)frame.query_frame, (unsigned long long)frame.solo_span,
                     frame.card, frame.owner_slot, frame.ctx, (unsigned long long)dpolicy_pause_seq_,
                     (unsigned long long)dpolicy_pause_epoch_);
            dpolicy_paused_ = false;
        }
    }
    return RAD_OK;
}

int Engine::dmeasure_check(int row, const int32_t* tokens, int n) const {
    if (!draft_experiment_.early_boundary()) return RAD_OK;
    const DraftMeasureFrame& frame = dmeasure_.current;
    const Scheduler::BlockRow& b = block_row_[(size_t)row];
    const StepSlot& s = slots_host_[b.card];
    if (dpolicy_concurrent_ && (row != 0 || frame.owner_slot != b.slot ||
        frame.query_frame != dpolicy_frame_ || frame.solo_span != dpolicy_span_)) return RAD_E_STATE;
    if (!draft_measure_frame_matches(frame, b.seq, b.card, s.ctx, s.tok[0], tokens, n)) return RAD_E_STATE;
    return RAD_OK;
}

void Engine::dmeasure_finish(const StepOut& so) {
    if (!draft_experiment_.early_boundary() || !dmeasure_.current.epoch) return;
    const bool policy = draft_experiment_.prefix_policy().on;
    const DraftMeasureFrame current = dmeasure_.current, previous = dmeasure_.closing;
    const bool closing = dmeasure_.close;
    const uint64_t cycle_ns = dmeasure_.closing_ns;
    uint64_t output = 0;
    bool live = false;
    const bool committed = sched_.committed_seq(current.seq, &output, &live);
    const bool published = sched_.measure_published(current, &output);
    int accepted = so.accepted.empty() ? -1 : so.accepted[0];
    if (dpolicy_concurrent_) {
        accepted = -1;
        const auto& entries = sched_.plan().e;
        for (size_t i = 0; i < entries.size(); ++i) {
            const auto& e = entries[i];
            if (e.seq == current.seq && e.card == current.card && e.slot == current.owner_slot &&
                i < so.accepted.size()) accepted = so.accepted[i];
        }
    }
    const bool counted = dmeasure_.committed(committed && live, published, output, accepted, policy);
    if (policy) {
        const std::string concurrent = dpolicy_concurrent_ ? fmt(",\"frame\":%llu,\"solo_span\":%llu,\"previous_span\":%llu",
            (unsigned long long)current.query_frame, (unsigned long long)current.solo_span,
            (unsigned long long)(closing ? previous.solo_span : 0)) : "";
        RAD_INFO("DSpark prefix policy cycle: {\"seq\":%llu,\"epoch\":%llu,\"verify_step\":%d,"
                 "\"next_epoch\":%llu,\"k\":%d,\"next_k\":%d,\"cycle_ns\":%llu,\"wait_ns\":%llu,"
                 "\"cost_valid\":%s,\"live\":%s,\"published\":%s,\"accepted\":%d,\"output\":%llu,"
                 "\"proposal_invalid_depths\":%u,\"next_invalid_depths\":%u%s}",
                 (unsigned long long)current.seq, (unsigned long long)(closing ? previous.epoch : 0), current.step,
                 (unsigned long long)current.epoch, closing ? previous.k : -1, current.k,
                 (unsigned long long)(closing ? cycle_ns : 0), (unsigned long long)current.wait_ns,
                 counted ? "true" : "false", committed && live ? "true" : "false", published ? "true" : "false",
                 accepted, (unsigned long long)output, (unsigned)previous.invalid_depths, (unsigned)current.invalid_depths,
                 concurrent.c_str());
    }
    else if (draft_experiment_.prob_capture() && closing)
        RAD_INFO("DSpark matched early cycle: {\"seq\":%llu,\"epoch\":%llu,\"verify_step\":%d,"
                 "\"next_epoch\":%llu,\"k\":%d,\"cycle_ns\":%llu,\"wait_ns\":%llu,"
                 "\"steady\":%s,\"live\":%s,\"published\":%s,\"accepted\":%d,"
                 "\"proposal_invalid_depths\":%u,\"next_invalid_depths\":%u}",
                 (unsigned long long)previous.seq, (unsigned long long)previous.epoch, current.step,
                 (unsigned long long)current.epoch, previous.k, (unsigned long long)cycle_ns,
                 (unsigned long long)current.wait_ns, counted ? "true" : "false",
                 committed && live ? "true" : "false", published ? "true" : "false", accepted,
                 (unsigned)previous.invalid_depths, (unsigned)current.invalid_depths);
}

void Engine::dmeasure_report() const {
    if (!draft_experiment_.early_boundary()) return;
    const DraftMeasureStats& s = dmeasure_.stats;
    if (draft_experiment_.prefix_policy().on) {
        RAD_INFO("DSpark prefix policy totals: queries=%llu query_rows=%llu observers=%llu waits=%llu "
                 "wait_ns=%llu transition=%llu no_query=%llu end_cancel=%llu unpublished=%llu "
                 "invalid_queries=%llu unclosed_epoch=%llu costs=[%llu,%llu,%llu]; "
                 "on-policy action/cycle telemetry, never fixed TRAIN costs",
                 (unsigned long long)s.queries, (unsigned long long)s.query_rows, (unsigned long long)s.queries,
                 (unsigned long long)s.waits, (unsigned long long)s.wait_all_ns, (unsigned long long)s.transition,
                 (unsigned long long)s.no_query, (unsigned long long)s.end_cancel, (unsigned long long)s.unpublished,
                 (unsigned long long)s.invalid_queries, (unsigned long long)dmeasure_.pending.epoch,
                 (unsigned long long)s.n[0], (unsigned long long)s.n[1], (unsigned long long)s.n[2]);
        return;
    }
    RAD_INFO("DSpark matched early totals: queries=%llu query_rows=%llu observers=%llu waits=%llu "
             "wait_ns=%llu transition=%llu no_query=%llu end_cancel=%llu unpublished=%llu "
             "invalid_queries=%llu invalid_cycles=%llu unclosed_epoch=%llu",
             (unsigned long long)s.queries, (unsigned long long)s.query_rows, (unsigned long long)s.queries,
             (unsigned long long)s.waits, (unsigned long long)s.wait_all_ns, (unsigned long long)s.transition,
             (unsigned long long)s.no_query, (unsigned long long)s.end_cancel, (unsigned long long)s.unpublished,
             (unsigned long long)s.invalid_queries, (unsigned long long)s.invalid_cycles,
             (unsigned long long)dmeasure_.pending.epoch);
    for (int k = 0; k < 3; ++k)
        RAD_INFO("DSpark matched early costs: k=%d n=%llu cycle_ns=%llu wait_ns=%llu accepted=%llu emitted=%llu "
                 "mean_ms=%.6f; consecutive same-sequence/epoch-bound pure-decode cycles only; includes query/observer/wait and logging=%s",
                 k, (unsigned long long)s.n[k], (unsigned long long)s.ns[k], (unsigned long long)s.wait_ns[k],
                 (unsigned long long)s.accepted[k], (unsigned long long)s.emitted[k],
                 s.n[k] ? (double)s.ns[k] / (double)s.n[k] * 1e-6 : 0.,
                 draft_experiment_.prob_capture() ? "capture" : "quiet (debug logs follow RADIANCE_LOG/-v)");
}
}
