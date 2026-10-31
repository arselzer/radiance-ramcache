/* Step 2A: read-only proposal observations, NOT a cost/depth policy. The optional architecture
 * op runs after unchanged selection. Copy behind it, retire after step_end_, and pair with the
 * same StepSlot's ID/position before the proposal's verification can commit. */
#include "engine_priv.h"
#include "device/device.h"
#include <cstring>

namespace rad {
int Engine::dprob_init() {
    if (!draft_experiment_.prob_capture()) return RAD_OK;
    if (!block_drafting() || tree_ || ranks_.empty() || !drafter_.name ||
        std::strcmp(drafter_.name, "dspark")) {
        RAD_ERR("%s requires DSpark's chain block drafter", rad_dflash_observe::kEnv);
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
                rad_dflash_observe::kEnv);
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
    RAD_INFO("DSpark read-only proposal capture: on (%s); depths 1/2, candidate_p is softmax at "
             "temperature 1 over the existing %d Markov-conditioned candidates, NOT full-vocabulary "
             "or acceptance probability; score and margin logged; no proposal, verification or "
             "depth decision reads it", rad_dflash_observe::kEnv, dprob_candidates_);
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
    if (!dprob_host_) return;
    const Scheduler::BlockRow& b = block_row_[(size_t)row];
    const StepSlot& s = slots_host_[b.card];
    sched_.draft_prob_publish(b.seq, s.ctx, s.tok[0], tokens, n,
                              dprob_host_ + (int64_t)row * rad_dflash_observe::kWidth,
                              published, dprob_candidates_);
}
}
