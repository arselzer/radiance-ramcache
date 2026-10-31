/* engine_dconf.cpp -- RADIANCE_EXPERIMENTAL_DS41_DSPARK_CONF (sched/draft_conf.h): DSpark's confidence
 * head, evaluated on the host for every query pass.
 *
 * The reference (inference/model.py DSparkConfidenceHead) scores block row i as
 * proj . [x_i, markov_embed(output_ids[i])] in fp32 with no bias, x_i the last stage's hc collapse
 * before its norm. Neither the head nor a use for it is in the draft bank, so the engine takes it from
 * a small file (..._CONF_HEAD; tools/ds41-host-experts/dspark_conf_extract.py writes it from the
 * checkpoint): `w`, the head's first `dim` weights, and `mscore`, its last `rank` folded through the
 * Markov embedding table, one score a token. The plugin keeps `x` (its buffer "dspark_collapsed")
 * persistent while the switch is set, and the query pass's first two rows a drafting sequence are
 * read back with the proposals; p = sigmoid(w . x + mscore[token]), token the row's input (the
 * anchor for row 0, the first proposal for row 1). Host work: two 5120-long dot products a drafting
 * sequence a step.
 *
 * It decides nothing here; the scheduler's controller does (Scheduler::draft_conf_scored). Nothing a
 * reply contains depends on it.
 */
#include "engine_priv.h"

#include "sched/draft_conf_head.h"
#include "sched/scheduler.h"
#include "device/device.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace rad {

/* At start, after the programs are declared: the head, the buffer and the pinned rows. */
int Engine::dconf_init() {
    if (!draft_experiment_.conf().on) return RAD_OK;
    if (!block_drafting() || ranks_.empty()) {
        RAD_ERR("%s: no block drafter (DSpark) runs in this deployment", kDraftConfEnv);
        return RAD_E_UNSUPPORTED;
    }
    const Program& prog = ranks_[0]->program;
    rad_buf h = 0;
    for (size_t i = 0; i < prog.buffers.size(); ++i) {
        const std::string& n = prog.buffers[i].name;
        const char* suf = "dspark_collapsed";
        const size_t k = std::strlen(suf);
        if (n.size() >= k && n.compare(n.size() - k, k, suf) == 0) h = (rad_buf)i;
    }
    if (!h) {
        RAD_ERR("%s: the architecture declares no 'dspark_collapsed' rows (not DeepSeek-V4.1's DSpark)",
                kDraftConfEnv);
        return RAD_E_UNSUPPORTED;
    }
    const RadBufDecl& d = prog.buffers[h].decl;
    if (d.kind != RAD_BUF_PERSIST || d.dtype != RAD_BF16 || d.rank != 2 || d.shape[0] < 2) {
        RAD_ERR("%s: the drafter's collapsed rows are not kept past the query pass (a deepseek41 plugin "
                "without the switch: it must declare them persistent bf16 [rows, dim])", kDraftConfEnv);
        return RAD_E_UNSUPPORTED;
    }
    std::string why;
    const char* path = std::getenv(kDraftConfHeadEnv);
    if (const int st = dconf_head_load(path, d.shape[1], &dconf_head_, &why); st < 0) {
        RAD_ERR("%s=%s: %s", kDraftConfHeadEnv, path ? path : "(unset)", why.c_str());
        return st;
    }
    dconf_x_ = h;
    dconf_rows_ = d.shape[0];
    dconf_host_ = static_cast<uint16_t*>(
        rad_dev_alloc(dconf_rows_ * dconf_head_.dim * (int64_t)sizeof(uint16_t), RAD_MEM_HOST_PINNED));
    if (!dconf_host_) return RAD_E_NOMEM;
    RAD_INFO("DSpark confidence head: %lld weights and %lld token scores from %s, evaluated on the host "
             "for the first two rows a drafting sequence of every query pass", (long long)dconf_head_.dim,
             (long long)dconf_head_.vocab, path);
    return RAD_OK;
}

void Engine::dconf_fini() {
    if (dconf_host_) rad_dev_free(dconf_host_, RAD_MEM_HOST_PINNED);
    dconf_host_ = nullptr;
    dconf_n_ = 0;
}

/* Behind the query pass on rank 0's stream, before the step's end event: the drafting rows' first
 * `block` rows each (rows i * block .. i * block + block - 1). */
int Engine::dconf_issue(int n, int block) {
    dconf_n_ = 0;
    if (!dconf_host_ || n <= 0 || block < 2) return RAD_OK;
    const int64_t rows = (int64_t)n * block;
    if (rows > dconf_rows_) return RAD_OK;
    Rank& r0 = *ranks_[0];
    const void* src = r0.ctx.buf_ptr(dconf_x_);
    if (!src) return RAD_E_STATE;
    RAD_TRY(rad_memcpy_async(dconf_host_, src, rows * dconf_head_.dim * (int64_t)sizeof(uint16_t),
                             r0.ctx.stream()));
    dconf_n_ = n;
    dconf_block_ = block;
    return RAD_OK;
}

/* After the step's end event, with the slots read: each drafting row's p for its first two
 * proposals, to the scheduler. */
void Engine::dconf_finish() {
    if (dconf_n_ <= 0) return;
    const int64_t D = dconf_head_.dim;
    const int steps = block_steps_;
    for (int i = 0; i < dconf_n_ && i < block_rows_; ++i) {
        const Scheduler::BlockRow& row = block_row_[(size_t)i];
        const StepSlot& S = slots_host_[row.card];
        const uint16_t* x0 = dconf_host_ + (int64_t)i * dconf_block_ * D;
        const float p0 = dconf_score(dconf_head_, x0, S.tok[0]);
        const float p1 = steps >= 2 ? dconf_score(dconf_head_, x0 + D, S.tok[1]) : -1.0f;
        sched_.draft_conf_scored(row.seq, p0, p1);
    }
    dconf_n_ = 0;
}

}  /* namespace rad */
