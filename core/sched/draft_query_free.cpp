#include "scheduler.h"
#include <climits>

namespace rad {
int Scheduler::query_free_opportunity(int step, const BlockRow* rows, int n,
                                       DraftQueryFreeFrame* out) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (!out) return RAD_E_INVAL;
    *out = {};
    const auto refused = [&](const char* reason) {
        RAD_ERR("DSpark query-free refused: stage=opportunity reason=%s step=%d current_step=%d "
                "step_open=%d next_open=%d block=%d depth=%d n=%d epoch=%llu",
                reason, step, step_ - 1, (int)step_open_, (int)next_open_, block_len_, block_depth_, n,
                (unsigned long long)dprob_epoch_);
        return RAD_E_STATE;
    };
    if (!draft_experiment_.query_free()) return refused("mode");
    if (!step_open_ || next_open_ || step != step_ - 1) return refused("step_phase");
    if (cfg_.max_seqs != 1 || !dctx_.on) return refused("profile");
    if (dprob_epoch_) return refused("query_epoch");
    if (n < 0 || n > 1 || (n && !rows)) return refused("rows");
    /* Engine::block_shape keeps the declared full5 shape while decoded_seqs()==0: a
     * nonproducing prefill piece has no query owner. Dx retains its tap history until the
     * producing piece catches up. Do not turn that valid pause into an opportunity or run
     * context early. A real owner still requires the admitted two-row query shape. */
    const bool short_shape = block_len_ == 2 && block_depth_ == 2;
    const bool full_pause = n == 0 && block_len_ == 5 && block_depth_ == 5;
    if (!short_shape && !full_pause) return refused("block_shape");
    int owners = 0;
    for (const BlockSeq& b : block_) {
        if (!b.can_draft) continue;
        if (++owners > 1 || n != 1 || b.mirror || b.seq != rows[0].seq ||
            b.slot != rows[0].slot || b.card != rows[0].card || b.n_spec != 0)
            return refused("owner_binding");
        const Request* r = block_req(b);
        if (!r || r->state != ReqState::Running) return RAD_OK; // cancelled owner; never adopt a reused slot
        if (b.ctx_done != b.ctx_n || b.ctx_done <= 0 || !r->speculation_fixed ||
            !draft_query_free_request(*r) || plan_.e.size() != 1) return refused("owner_context");
        for (const StepEntry& e : plan_.e) {
            if (e.seq != b.seq || e.slot != b.slot || e.card != b.card) continue;
            const int64_t input = e.dev ? r->n_computed : e.ctx_len;
            const int64_t ctx = input + e.n_tokens;
            if (e.n_spec || e.ngram || !e.produces_token || input < 0 || ctx > INT32_MAX)
                return refused("target_binding");
            out->seq = b.seq; out->slot = b.slot; out->card = b.card; out->step = step;
            out->input_ctx = (int32_t)input; out->ctx = (int32_t)ctx;
            out->output_before = (uint64_t)r->output.size(); out->context_rows = b.ctx_done;
            out->pure_decode = plan_.phase == RAD_PHASE_DECODE && plan_.e.size() == 1 && !e.is_prefill;
            return RAD_OK;
        }
        return refused("owner_plan");
    }
    return owners == n ? RAD_OK : refused("owner_count");
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
