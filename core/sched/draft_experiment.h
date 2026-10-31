/* Private opt-in draft attribution and bank-loaded compute-OFF control.
 * This object is captured once by Engine, then copied into Scheduler at init.
 * It never changes Config, Program, allocator bounds or the declared maximum depth. */
#pragma once
#include <cstdint>
#include <cstring>

#include "draft_gate.h"
#include "draft_conf.h"
#include "draft_prob.h"

namespace rad {
inline constexpr char kDraftOriginEnv[] = "RADIANCE_EXPERIMENTAL_DS41_DRAFT_ORIGIN_STATS";
inline constexpr char kDraftComputeOffEnv[] = "RADIANCE_EXPERIMENTAL_DS41_DSPARK_COMPUTE_OFF";
inline constexpr char kDraftVerifyCapEnv[] = "RADIANCE_EXPERIMENTAL_DS41_DSPARK_VERIFY_CAP";
class DraftExperiment {
public:
    DraftExperiment() = default;
    static DraftExperiment capture(const char* stats, const char* off,
                                   const char* verify_cap = nullptr,
                                   const char* gate = nullptr,
                                   const char* conf = nullptr,
                                   const char* load = nullptr,
                                   const char* defer = nullptr,
                                   const char* prob = nullptr,
                                   const char* early = nullptr) noexcept {
        DraftExperiment x;
        const bool valid_cap = !verify_cap ||
            (verify_cap[0] >= '0' && verify_cap[0] <= '5' && verify_cap[1] == '\0');
        x.cap_ = valid_cap && verify_cap ? verify_cap[0] - '0' : -1;
        // The acceptance gate (draft_gate.h): a spec it cannot read is refused, not ignored.
        const bool valid_gate = DraftGateCfg::parse(gate, &x.gate_);
        // A positive cap and explicit compute-OFF conflict rather than silently overriding.
        x.valid_ = flag(stats) && flag(off) && valid_cap && !(one(off) && x.cap_ > 0) && valid_gate;
        x.off_ = one(off) || x.cap_ == 0;
        // Nothing drafts with compute OFF, so a gate there would gate nothing under its name.
        if (x.off_ && x.gate_.on) x.valid_ = false;
        // draft_conf.h: the confidence controller, load-aware drafting, the deferred context pass.
        // Each refused if unreadable or with compute OFF; the controller and the acceptance gate
        // both decide whether a query pass runs, so not together.
        const bool valid_conf = DraftConfCfg::parse(conf, &x.conf_);
        const bool valid_load = draft_load_parse(load, &x.load_from_);
        const bool valid_defer = draft_ctx_defer_parse(defer, &x.ctx_defer_);
        if (!valid_conf || !valid_load || !valid_defer) x.valid_ = false;
        if (x.off_ && (x.conf_.on || x.load_from_ > 0 || x.ctx_defer_)) x.valid_ = false;
        if (x.conf_.on && x.gate_.on) x.valid_ = false;
        if (!flag(prob) || (one(prob) && x.off_)) x.valid_ = false;
        x.prob_ = one(prob);
        x.early_ = rad_dflash_observe::early_fixed(early);
        if (x.early_ < -1 || (x.early_ >= 0 && (x.off_ || x.cap_ >= 0 || x.gate_.on ||
                                               x.conf_.on || x.load_from_ > 0))) x.valid_ = false;
        x.observe_ = one(stats) || x.off_ || x.cap_ >= 0;
        return x;
    }
    bool valid() const noexcept { return valid_; }
    bool compute_off() const noexcept { return off_; }
    bool observe() const noexcept { return observe_; }
    bool head_enabled(bool declared) const noexcept { return declared && !off_; }
    int verify_cap() const noexcept { return cap_; } // -1 means the original uncapped path
    const DraftGateCfg& gate() const noexcept { return gate_; } // the acceptance gate; off by default
    const DraftConfCfg& conf() const noexcept { return conf_; } // the confidence controller; off by default
    int load_from() const noexcept { return load_from_; }       // load-aware drafting from N decoders; 0 off
    bool ctx_defer() const noexcept { return ctx_defer_; }      // the deferred context pass; off by default
    bool prob_capture() const noexcept { return prob_; }      // read-only candidate-set signal
    int early_verify() const noexcept { return early_; }      // fixed matched measurement, absent/off=-1
    bool scope_required() const noexcept { return off_ || cap_ >= 0 || early_ >= 0; }
    int runtime_depth(int configured) const noexcept {
        if (off_) return 0;
        if (early_ >= 0 && configured > early_) return early_;
        return cap_ >= 0 && configured > cap_ ? cap_ : configured;
    }
    bool target_pipelining(bool eligible, bool serial, bool block, int configured) const noexcept {
        return eligible && (serial || block || configured <= 0 || off_);
    }
private:
    bool valid_ = true, off_ = false, observe_ = false;
    int cap_ = -1;
    DraftGateCfg gate_{};
    DraftConfCfg conf_{};
    int load_from_ = 0;
    bool ctx_defer_ = false;
    bool prob_ = false;
    int early_ = -1;
    static bool one(const char* x) noexcept { return x && std::strcmp(x,"1") == 0; }
    static bool flag(const char* x) noexcept {
        return !x || std::strcmp(x,"0") == 0 || std::strcmp(x,"1") == 0;
    }
};

/* Only successful ownership commits call these methods. Published/offered tokens
 * can be overwritten, held, capped or cancelled; they are not inferred consumes.
 * Fallback includes prompt lookup AND the debug oracle, never a head-origin claim.
 * The owner is Scheduler::mu_; readers see only the normal published snapshot. */
struct DraftOriginStats {
    bool enabled = false;
    bool compute_off = false;
    int verify_cap = -1; // configured private control, not a measured verify-depth histogram
    int64_t learned_publications = 0, learned_published_tokens = 0;
    int64_t fallback_offers = 0, fallback_offered_tokens = 0;
    int64_t fallback_publications = 0, fallback_published_tokens = 0;
    int64_t learned_verified_tokens = 0, learned_accepted_tokens = 0;
    int64_t fallback_verified_tokens = 0, fallback_accepted_tokens = 0;
    int64_t block_context_issues = 0, block_context_rows = 0;
    int64_t block_query_issues = 0, block_query_rows = 0;
    void learned_publish(int n) noexcept {
        if (enabled && n > 0) { ++learned_publications; learned_published_tokens += n; }
    }
    void fallback_offer(int n) noexcept {
        if (enabled && n > 0) { ++fallback_offers; fallback_offered_tokens += n; }
    }
    void fallback_publish(int n) noexcept {
        if (enabled && n > 0) { ++fallback_publications; fallback_published_tokens += n; }
    }
    void committed(bool fallback, int verified, int accepted) noexcept {
        if (!enabled || verified <= 0) return;
        if (fallback) { fallback_verified_tokens += verified; fallback_accepted_tokens += accepted; }
        else { learned_verified_tokens += verified; learned_accepted_tokens += accepted; }
    }
    // Known0 all-rank issue, not a completion, GPU time, or per-rank sum.
    void block_issue(int pass, int64_t rows) noexcept {
        if (!enabled || rows <= 0) return;
        if (pass == -1) { ++block_context_issues; block_context_rows += rows; }
        else if (pass == 1) { ++block_query_issues; block_query_rows += rows; }
    }
};
} // namespace rad
