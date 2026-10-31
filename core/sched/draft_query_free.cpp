#include "scheduler.h"
#include <climits>

namespace rad {
int Scheduler::query_free_opportunity(int step, const BlockRow* rows, int n,
                                       DraftQueryFreeFrame* out) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (!out) return RAD_E_INVAL;
    *out = {};
    if (!draft_experiment_.query_free() || !step_open_ || next_open_ || step != step_ - 1 ||
        cfg_.max_seqs != 1 || !dctx_.on || block_len_ != 2 || block_depth_ != 2 ||
        dprob_epoch_ || n < 0 || n > 1 || (n && !rows)) return RAD_E_STATE;
    int owners = 0;
    for (const BlockSeq& b : block_) {
        if (!b.can_draft) continue;
        if (++owners > 1 || n != 1 || b.mirror || b.seq != rows[0].seq ||
            b.slot != rows[0].slot || b.card != rows[0].card || b.n_spec != 0) return RAD_E_STATE;
        const Request* r = block_req(b);
        if (!r || r->state != ReqState::Running) return RAD_OK; // cancelled owner; never adopt a reused slot
        if (b.ctx_done != b.ctx_n || b.ctx_done <= 0 || !r->speculation_fixed ||
            !draft_query_free_request(*r) || plan_.e.size() != 1) return RAD_E_STATE;
        for (const StepEntry& e : plan_.e) {
            if (e.seq != b.seq || e.slot != b.slot || e.card != b.card) continue;
            const int64_t input = e.dev ? r->n_computed : e.ctx_len;
            const int64_t ctx = input + e.n_tokens;
            if (e.n_spec || e.ngram || !e.produces_token || input < 0 || ctx > INT32_MAX)
                return RAD_E_STATE;
            out->seq = b.seq; out->slot = b.slot; out->card = b.card; out->step = step;
            out->input_ctx = (int32_t)input; out->ctx = (int32_t)ctx;
            out->output_before = (uint64_t)r->output.size(); out->context_rows = b.ctx_done;
            out->pure_decode = plan_.phase == RAD_PHASE_DECODE && plan_.e.size() == 1 && !e.is_prefill;
            return RAD_OK;
        }
        return RAD_E_STATE;
    }
    return owners == n ? RAD_OK : RAD_E_STATE;
}

bool Scheduler::query_free_committed(const DraftQueryFreeFrame& row, uint64_t* output) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (!output || !row.seq || !row.boundary || row.output_before == UINT64_MAX || dprob_epoch_) return false;
    const auto it = by_id_.find(row.seq);
    if (it == by_id_.end() || it->second != row.slot) return false;
    const SchedReq& s = *reqs_[(size_t)it->second];
    if (!s.req || s.req->id != row.seq || s.card != row.card || s.n_draft ||
        s.req->state != ReqState::Running || !s.req->speculation_fixed || !draft_query_free_request(*s.req) ||
        s.req->n_computed != row.ctx || s.req->output.empty() || s.req->output.size() != row.output_before + 1 ||
        s.req->output.back() != row.token) return false;
    *output = (uint64_t)s.req->output.size();
    return true;
}
} // namespace rad
