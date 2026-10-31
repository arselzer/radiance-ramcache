/* Step 2A: read-only proposal observations, NOT a cost/depth policy. The optional architecture
 * op runs after unchanged selection. Copy behind it, retire after step_end_, and pair with the
 * same StepSlot's ID/position before the proposal's verification can commit. */
#include "engine_priv.h"
#include "device/device.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace rad {
int Engine::dprob_init() {
    if (!draft_experiment_.prob_capture() && draft_experiment_.early_verify() < 0) return RAD_OK;
    const char* instrument = draft_experiment_.early_verify() >= 0
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
    if (draft_experiment_.prob_capture())
        RAD_INFO("DSpark read-only proposal capture: on (%s); depths 1/2, candidate_p is softmax at "
             "temperature 1 over the existing %d Markov-conditioned candidates, NOT full-vocabulary "
             "or acceptance probability; score and margin logged; no proposal, verification or "
             "depth decision reads it", rad_dflash_observe::kEnv, dprob_candidates_);
    if (draft_experiment_.early_verify() >= 0) {
        dmeasure_ = {};
        dmeasure_query_rows_ = (int)(short_from_ == 1 && drafter_short_ > 0 ? drafter_short_ : drafter_.block);
        if (dmeasure_query_rows_ < 2 || dmeasure_query_rows_ > 5) return RAD_E_UNSUPPORTED;
        RAD_INFO("DSpark matched early verify: on k=%d (%s); query_rows=%d unchanged at every action, "
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
                              published, dprob_candidates_, draft_experiment_.early_verify() >= 0
                                  ? dmeasure_.current.epoch : 0);
}

/* Called AFTER mark_step_end and BEFORE issue_ahead. Existing step_end covers both the observation
 * and complete StepSlot copies. No new event, stream synchronization, or host wait on the off path.
 * Later collect/so.early waits are deliberately retained (already complete in this branch). */
int Engine::dmeasure_ready(const RadBatch* batch) {
    if (block_rows_ == 0) { dmeasure_.no_query(); return RAD_OK; }
    if (block_rows_ != 1 || block_steps_ != dmeasure_query_rows_ || !dprob_host_) return RAD_E_STATE;
    const uint64_t begin = rad_mono_ns();
    RAD_TRY(rad_event_sync(step_end_));
    const uint64_t ready = rad_mono_ns();
    const Scheduler::BlockRow& row = block_row_[0];
    const StepSlot& slot = slots_host_[row.card];
    bool previous_matches = false;
    if (!sched_.measure_block_binding(row, slot.ctx, slot.acc, dmeasure_.pending, &previous_matches)) {
        RAD_ERR("DSpark matched early verify: sequence/slot/ctx binding failed for sequence %llu",
                (unsigned long long)row.seq);
        return RAD_E_STATE;
    }
    uint32_t invalid_depths = 0;
    for (int j = 0; j < rad_dflash_observe::kDepths; ++j) {
        const float* obs = dprob_host_ + j * rad_dflash_observe::kFields;
        const DraftMeasureSignal signal = draft_measure_signal(obs, slot.tok[j + 1], slot.tok[j]);
        if (signal == DraftMeasureSignal::PathMismatch) {
            RAD_ERR("DSpark matched early verify: echoed observation/path IDs differ at depth %d", j + 1);
            return RAD_E_STATE;
        }
        if (signal == DraftMeasureSignal::InvalidNumeric) invalid_depths |= 1u << j;
    }
    DraftMeasureFrame frame;
    frame.seq = row.seq; frame.card = row.card; frame.step = batch->step;
    frame.epoch = sched_.draft_prob_reserve_epoch();
    frame.ctx = slot.ctx; frame.anchor = slot.tok[0];
    frame.k = draft_experiment_.early_verify(); frame.query_rows = dmeasure_query_rows_;
    frame.invalid_depths = invalid_depths;
    frame.ready_ns = ready; frame.wait_ns = ready - begin;
    std::copy(slot.tok + 1, slot.tok + 1 + block_steps_, frame.ids);
    const StepPlan& plan = sched_.plan();
    frame.pure_decode = plan.phase == RAD_PHASE_DECODE && plan.e.size() == 1 &&
                        !plan.e[0].is_prefill && !plan.e[0].ngram && plan.e[0].produces_token;
    dmeasure_.ready(frame, previous_matches);
    return RAD_OK;
}

int Engine::dmeasure_check(int row, const int32_t* tokens, int n) const {
    if (draft_experiment_.early_verify() < 0) return RAD_OK;
    const DraftMeasureFrame& frame = dmeasure_.current;
    const Scheduler::BlockRow& b = block_row_[(size_t)row];
    const StepSlot& s = slots_host_[b.card];
    if (!frame.epoch || frame.seq != b.seq || frame.card != b.card || frame.ctx != s.ctx ||
        frame.anchor != s.tok[0] || n != frame.query_rows ||
        std::memcmp(frame.ids, tokens, (size_t)n * sizeof(int32_t))) return RAD_E_STATE;
    return RAD_OK;
}

void Engine::dmeasure_finish(const StepOut& so) {
    if (draft_experiment_.early_verify() < 0 || !dmeasure_.current.epoch) return;
    const DraftMeasureFrame current = dmeasure_.current, previous = dmeasure_.closing;
    const bool closing = dmeasure_.close;
    const uint64_t cycle_ns = dmeasure_.closing_ns;
    uint64_t output = 0;
    bool live = false;
    const bool committed = sched_.committed_seq(current.seq, &output, &live);
    const bool published = sched_.measure_published(current, &output);
    const int accepted = so.accepted.empty() ? -1 : so.accepted[0];
    const bool counted = dmeasure_.committed(committed && live, published, output, accepted);
    if (draft_experiment_.prob_capture() && closing)
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
    if (draft_experiment_.early_verify() < 0) return;
    const DraftMeasureStats& s = dmeasure_.stats;
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
