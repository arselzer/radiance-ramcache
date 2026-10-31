#include "scheduler.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace rad {
namespace {
const char* json_float(float v, char (&buf)[32]) {
    if (!std::isfinite(v)) return "null";
    std::snprintf(buf, sizeof buf, "%.9g", (double)v);
    return buf;
}
}

void Scheduler::dprob_record(uint64_t seq, const DraftProbPending& p, int depth, const char* event,
                              const char* status, DraftProbOutcome o, bool cancelled) {
    using namespace rad_dflash_observe;
    const float* x = p.signal + depth * kFields;
    const int32_t prev = depth ? p.token[depth - 1] : p.anchor_token;
    const bool valid = std::isfinite(x[kCandidateP]) && x[kCandidateP] >= 0 && x[kCandidateP] <= 1 &&
                       std::isfinite(x[kScore]) && !std::isnan(x[kMargin]) && x[kMargin] >= 0 &&
                       x[kToken] == (float)p.token[depth] && x[kPredecessor] == (float)prev;
    const bool proposal = !std::strcmp(event, "proposal");
    if (proposal) ++dprob_stats_.proposed;
    else {
        ++dprob_stats_.resolved;
        dprob_stats_.verified += o.verified;
        dprob_stats_.reached += o.reached;
        dprob_stats_.accepted += o.accepted;
    }
    char a[32], b[32], c[32];
    RAD_INFO("DSpark proposal capture: {\"v\":1,\"event\":\"%s\",\"seq\":%llu,\"epoch\":%llu,"
             "\"anchor_pos\":%lld,\"pos\":%lld,\"depth\":%d,\"token\":%d,\"predecessor\":%d,"
             "\"candidate_count\":%d,\"candidate_p\":%s,\"score\":%s,\"margin\":%s,"
             "\"margin_unbounded\":%s,\"signal_valid\":%s,\"status\":\"%s\",\"verified\":%s,"
             "\"reached\":%s,\"accepted\":%s,\"cancelled\":%s}",
             event, (unsigned long long)seq, (unsigned long long)p.epoch, (long long)p.anchor,
             (long long)(p.anchor + depth + 1), depth + 1, p.token[depth], prev, p.candidates,
             json_float(x[kCandidateP], a), json_float(x[kScore], b), json_float(x[kMargin], c),
             std::isinf(x[kMargin]) && x[kMargin] > 0 ? "true" : "false", valid ? "true" : "false",
             status, proposal ? "null" : o.verified ? "true" : "false",
             proposal ? "null" : o.reached ? "true" : "false",
             !proposal && o.reached ? (o.accepted ? "true" : "false") : "null",
             cancelled ? "true" : "false");
}

void Scheduler::dprob_discard(SchedReq& s, const char* reason, bool cancelled) {
    if (!dprob_on_ || !s.req) return;
    for (int j = 0; j < s.dprob.n; ++j)
        dprob_record(s.req->id, s.dprob, j, "outcome", reason, {},
                      cancelled || s.req->state == ReqState::Cancelled);
    s.dprob = {};
}

void Scheduler::draft_prob_publish(uint64_t id, int64_t anchor, int32_t anchor_token,
                                   const int32_t* tokens, int n, const float* signal,
                                   bool published, int candidates, uint64_t reserved_epoch) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!dprob_on_ || !tokens || !signal || n <= 0) return;
    DraftProbPending p;
    p.epoch = reserved_epoch ? reserved_epoch : ++dprob_epoch_;
    p.anchor = anchor; p.anchor_token = anchor_token;
    p.n = std::min(n, (int)rad_dflash_observe::kDepths); p.candidates = candidates;
    std::copy(tokens, tokens + p.n, p.token);
    std::copy(signal, signal + rad_dflash_observe::kWidth, p.signal);
    const auto it = by_id_.find(id);
    SchedReq* s = it == by_id_.end() ? nullptr : reqs_[(size_t)it->second].get();
    if (s) dprob_discard(*s, "superseded");
    const bool cancelled = s && s->req && s->req->state == ReqState::Cancelled;
    const bool live = s && s->req && (s->req->state == ReqState::Running ||
                                      (next_open_ && s->entry_next >= 0));
    const int installed = published && live ? std::min(p.n, s->n_draft) : 0;
    for (int j = 0; j < p.n; ++j) {
        dprob_record(id, p, j, "proposal", "pending", {}, cancelled);
        if (j >= installed)
            dprob_record(id, p, j, "outcome", !live ? "request_closed" :
                          !published ? "not_published" : "verify_cap", {}, cancelled);
    }
    if (installed > 0) { p.n = installed; s->dprob = p; }
}

void Scheduler::dprob_verify(SchedReq& s, const StepEntry& e, int64_t anchor, int accepted) {
    const DraftProbPending& p = s.dprob;
    for (int j = 0; j < p.n; ++j) {
        const DraftProbOutcome o = e.ngram ? DraftProbOutcome{} :
            draft_prob_outcome(p, j, anchor, e.draft, e.n_spec, accepted);
        const char* status = !o.matched ? "proposal_mismatch" : !o.verified ? "not_verified" :
                             !o.reached ? "unreached" : o.accepted ? "accepted" : "rejected";
        dprob_record(e.seq, p, j, "outcome", status, o, s.req->state == ReqState::Cancelled);
    }
    s.dprob = {};
}

DraftProbStats Scheduler::draft_prob_stats() const {
    std::lock_guard<std::mutex> lk(mu_);
    return dprob_stats_;
}

uint64_t Scheduler::draft_prob_reserve_epoch() {
    std::lock_guard<std::mutex> lk(mu_);
    return dprob_on_ || draft_experiment_.early_boundary() ? ++dprob_epoch_ : 0;
}
}
