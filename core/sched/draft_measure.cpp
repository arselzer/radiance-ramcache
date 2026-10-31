#include "scheduler.h"

namespace rad {
bool Scheduler::policy_block_owner(int step, const BlockRow* rows, int n, uint64_t* frame) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (!frame || !draft_experiment_.prefix_policy().on || cfg_.max_seqs <= 1 || !dsolo_ ||
        !dctx_.on || !step_open_ || next_open_ || !policy_frame_open_ ||
        policy_frame_step_ != step || !policy_frame_epoch_ || n < 0 || n > 1 || (n && !rows)) return false;
    int owners = 0;
    for (const BlockSeq& b : block_) {
        if (!b.can_draft) continue;
        if (++owners > 1 || n != 1 || policy_frame_solo_ || block_len_ != 2 || block_depth_ != 2 ||
            b.mirror || !block_req(b) || b.seq != rows[0].seq || b.slot != rows[0].slot ||
            b.card != rows[0].card) return false;
    }
    if (owners != n) return false;
    *frame = policy_frame_epoch_;
    return true;
}
/* The block snapshot keeps the Request alive; identity/card/position are nevertheless checked
 * under mu_ so cancellation/reuse cannot validate another sequence's StepSlot. */
bool Scheduler::measure_block_binding(const BlockRow& row, int32_t ctx, int accepted,
                                       const DraftMeasureFrame& previous, bool* previous_matches) const {
    std::lock_guard<std::mutex> lk(mu_);
    *previous_matches = false;
    if (row.slot < 0 || (size_t)row.slot >= reqs_.size() || !reqs_[(size_t)row.slot]) return false;
    const SchedReq& s = *reqs_[(size_t)row.slot];
    if (!s.req || s.req->id != row.seq || s.card != row.card) return false;
    for (const StepEntry& e : plan_.e) {
        if (e.seq != row.seq || e.slot != row.slot || e.card != row.card) continue;
        const int64_t input = e.dev ? (int64_t)s.req->n_computed : e.ctx_len;
        if (accepted < 0 || accepted > e.n_spec ||
            ctx != input + e.n_tokens - e.n_spec + accepted) return false;
        bool match = previous.epoch && previous.seq == row.seq && previous.card == row.card &&
                     previous.ctx == input && previous.k == e.n_spec;
        for (int j = 0; match && j < previous.k; ++j)
            match = e.draft && e.draft[j] == previous.ids[j];
        *previous_matches = match;
        return true;
    }
    return false;
}

bool Scheduler::measure_published(const DraftMeasureFrame& row, uint64_t* output) const {
    std::lock_guard<std::mutex> lk(mu_);
    const auto it = by_id_.find(row.seq);
    if (it == by_id_.end()) return false;
    const SchedReq& s = *reqs_[(size_t)it->second];
    if (!s.req || s.req->state != ReqState::Running || !s.req->may_speculate ||
        !s.req->speculation_fixed || s.card != row.card || s.req->n_computed != row.ctx ||
        s.req->output.empty() || s.req->output.back() != row.anchor || s.n_draft != row.k) return false;
    for (int j = 0; j < row.k; ++j) if (s.draft[j] != row.ids[j]) return false;
    *output = (uint64_t)s.req->output.size();
    return true;
}
}
