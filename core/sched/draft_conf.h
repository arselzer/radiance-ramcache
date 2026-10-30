/* draft_conf.h -- three opt-in controls for a block drafter (DSpark), each off by default:
 *
 *   RADIANCE_EXPERIMENTAL_DS41_DSPARK_CONF=T[,R]        the confidence controller
 *   RADIANCE_EXPERIMENTAL_DS41_DSPARK_LOAD=N            load-aware drafting
 *   RADIANCE_EXPERIMENTAL_DS41_DSPARK_CTX_DEFER=1       the context pass deferred while nothing drafts
 *
 * THE CONFIDENCE CONTROLLER. DSpark's last stage carries a confidence head (the reference's
 * DSparkConfidenceHead): for block row i, proj . [x_i, markov_embed(output_ids[i])], a logit for
 * whether the proposal that row makes is accepted. The engine evaluates it on the host for the rows
 * of every query pass (engine_dconf.cpp: the rows' collapsed hidden states read back with the
 * proposals, the head from the checkpoint by tools/ds41-host-experts/dspark_conf_extract.py) and
 * hands the scheduler p = sigmoid(logit) for the first two proposals of each drafting sequence.
 *
 * The decision is per sequence and per step, in Scheduler::block_begin, before the query pass is
 * staged: the query pass is the cost (window ab-20261030T134728Z: a step with a draft and a verify
 * 90.2 ms against 59.0 ms for a plain one, while a verify without a draft cost 56.5 ms: the verify
 * row is nearly free, the draft is not). The step issued ahead already verifies the draft the last
 * query pass made, before the host can read its confidence, so the newest estimate the decision can
 * use is that draft's: p of the proposal now being verified. A sequence drafts while that p is at
 * or above T and stops below it, and then has no newer estimate: every R-th step of a closed run it
 * drafts anyway (a probe), whose confidence reopens it or not. A sequence starts open (no estimate).
 *
 * Not done: a second proposal verified when p is high. The short block proposes two, but the verify
 * cap is applied at eight sites (step planning and its token budget, expect_draft, set_draft,
 * step_next, the adaptive window) that must agree on a per-sequence count; the shutdown report's
 * second-proposal table says first whether it would pay.
 *
 * LOAD-AWARE DRAFTING. With N or more sequences decoding (running and past their prompt), no
 * sequence drafts: on this engine every decode row brings its own experts, so a draft row competes
 * with the other streams' rows for the step, and concurrent throughput comes first.
 *
 * THE CONTEXT PASS DEFERRED (the plugin's half: arch/deepseek41 Dsp::ctx_defer). The drafter's
 * context pass, a pass of its own after every trunk step, keeps the drafter's window cache current
 * for every committed position whether or not anything drafts. Deferred, a step whose sequences
 * draft nothing skips it; the trunk's tapped rows wait in the plugin's source buffer (appended in
 * issue order), and the next pass that runs covers them:
 *   - one sequence decoding alone, plain steps (no drafts verified): the waiting rows are its own,
 *     positions contiguous, and the next pass is a catch-up, one entry over all of them. Exact
 *     drafter state: the same keys as one pass a step.
 *   - anything else (several sequences, a prefill chunk, another sequence): the waiting rows are
 *     dropped (the next pass covers its own step's rows only) and each sequence that missed rows
 *     has a hole: it drafts again only once its window has been rebuilt past the hole, as after a
 *     prefix-cache hit (Scheduler::draft_floor).
 * At most `cap` rows wait (the plugin declares the room: its source rows less the step budget), so a
 * pass runs at least every `cap` rows.
 *
 * Exactness: none of the three changes a reply. A draft is verified like any other; these choose
 * only whether a draft is made and how many proposals a verify takes, and a deferred pass computes
 * the same keys from the same rows (any rounding a multi-row pass brings can move a proposal, never
 * an emitted token: core/sample/accept.h). Off, every path is the branch's.
 */
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace rad {

inline constexpr char kDraftConfEnv[]     = "RADIANCE_EXPERIMENTAL_DS41_DSPARK_CONF";
inline constexpr char kDraftConfHeadEnv[] = "RADIANCE_EXPERIMENTAL_DS41_DSPARK_CONF_HEAD";
inline constexpr char kDraftLoadEnv[]     = "RADIANCE_EXPERIMENTAL_DS41_DSPARK_LOAD";
inline constexpr char kDraftCtxDeferEnv[] = "RADIANCE_EXPERIMENTAL_DS41_DSPARK_CTX_DEFER";

struct DraftConfCfg {
    enum { kMaxProbe = 4096 };
    /* The defaults. T: a draft step against a plain one with the bank loaded is 90.2 / 59.0 ms in
     * window ab-20261030T134728Z, a break-even of 0.53 accepted a draft; 0.55 keeps a little margin
     * for a head whose calibration on this engine is not measured yet (the shutdown report's table
     * measures it). R: as the acceptance gate's probe. */
    static constexpr double kThreshold = 0.55;
    static constexpr int    kProbe = 8;

    bool   on = false;
    double threshold = kThreshold;   /* T: drafts while the newest p >= T */
    int    probe = kProbe;           /* R: a closed run drafts every R-th step */

    /* Unset, "" or "0": off. "1": on at the defaults. Otherwise "T[,R]": T in (0, 1), R in 2..4096.
     * False for anything else, which the engine refuses at startup. */
    static bool parse(const char* s, DraftConfCfg* out) {
        DraftConfCfg c;
        if (!s || !*s || !std::strcmp(s, "0")) { *out = c; return true; }
        c.on = true;
        if (!std::strcmp(s, "1")) { *out = c; return true; }
        char* e = nullptr;
        const double t = std::strtod(s, &e);
        if (e == s || !(t > 0.0) || !(t < 1.0)) return false;   /* NaN fails both */
        c.threshold = t;
        if (*e == ',') {
            const char* p = e + 1;
            const long r = std::strtol(p, &e, 10);
            if (e == p || r < 2 || r > kMaxProbe) return false;
            c.probe = (int)r;
        }
        if (*e) return false;
        *out = c;
        return true;
    }
};

/* What an ask decided: no draft; a draft with no estimate yet, open, or as a closed run's probe. */
enum DraftConfKind { kDraftConfGated = 0, kDraftConfWarm, kDraftConfOpen, kDraftConfProbe };

/* The calibration tables' resolution: p in tenths. */
inline constexpr int kDraftConfBins = 10;

inline int draft_conf_bin(double p) {
    if (!(p >= 0.0)) return 0;
    /* A float p a hair under a tenth (0.7f is 0.69999998) belongs to that tenth. */
    const int b = (int)(p * kDraftConfBins + 1e-6);
    return b >= kDraftConfBins ? kDraftConfBins - 1 : b;
}

/* One sequence's controller. It describes the text, so preemption keeps it and a new request starts
 * it over (Scheduler::add_locked). */
struct DraftConfSeq {
    /* The newest estimate: p of the newest draft's first and second proposals, -1 before the first.
     * `prev` / `prev1`: the estimate before it, the one in force when the draft the next commit
     * verifies was decided. `fresh`: `p` belongs to the draft the next commit verifies. */
    float p = -1.0f, p1 = -1.0f, prev = -1.0f, prev1 = -1.0f;
    bool  fresh = false;
    /* Steps asked since the controller closed; 0 while open. */
    int32_t closed = 0;
    /* The scheduler step whose verify carries a probe's draft, -1 for none (as DraftGateSeq). */
    int64_t probe_step = -1;
    /* Counters: steps asked; of them drafted with no estimate, open, as a probe, and gated; closed
     * runs begun and ended; drafts that got an estimate; verifies of head drafts with the proposals
     * verified and accepted, of those the probes'. */
    int64_t asks = 0, warm = 0, open = 0, probes = 0, gated = 0, closes = 0, reopens = 0;
    int64_t scored = 0;
    int64_t verifies = 0, verified = 0, accepted = 0, probe_verified = 0, probe_accepted = 0;
    /* Calibration, [bin][0] verifies and [bin][1] of them whose first proposal was accepted, by
     * the draft's own p (`own`), by the estimate that decided it (`dec`, its predecessor's p), and
     * by its predecessor's second-proposal p (`dec1`). */
    int64_t own[kDraftConfBins][2] = {}, dec[kDraftConfBins][2] = {}, dec1[kDraftConfBins][2] = {};
};

/* Whether the sequence drafts this step, and why (DraftConfKind); counts the ask. */
inline int draft_conf_ask(const DraftConfCfg& c, DraftConfSeq& g) {
    ++g.asks;
    if (g.p < 0.0f) { ++g.warm; return kDraftConfWarm; }
    if ((double)g.p >= c.threshold) {
        if (g.closed) { ++g.reopens; g.closed = 0; }
        ++g.open;
        return kDraftConfOpen;
    }
    if (!g.closed) ++g.closes;
    ++g.closed;
    if (g.closed % c.probe == 0) { ++g.probes; return kDraftConfProbe; }
    ++g.gated;
    return kDraftConfGated;
}

/* An ask that drafted whose draft was not staged after all (its KV refused). */
inline void draft_conf_unask(DraftConfSeq& g, int kind) {
    switch (kind) {
    case kDraftConfWarm:  --g.warm; break;
    case kDraftConfOpen:  --g.open; break;
    case kDraftConfProbe: --g.probes; break;
    default: return;
    }
    --g.asks;
}

/* The estimate for the draft just made (p0: its first proposal, p1: its second, -1 for none). */
inline void draft_conf_scored(DraftConfSeq& g, float p0, float p1) {
    g.prev = g.p;
    g.prev1 = g.p1;
    g.p = p0;
    g.p1 = p1;
    g.fresh = true;
    ++g.scored;
}

/* A verify of `verified` head proposals, `accepted` of them taken; `probe` when a probe drafted it. */
inline void draft_conf_verified(DraftConfSeq& g, int verified, int accepted, bool probe) {
    if (verified <= 0) return;
    if (accepted < 0) accepted = 0;
    if (accepted > verified) accepted = verified;
    ++g.verifies;
    g.verified += verified;
    g.accepted += accepted;
    if (probe) {
        g.probe_verified += verified;
        g.probe_accepted += accepted;
    }
    if (g.fresh) {
        const int hit = accepted > 0 ? 1 : 0;
        if (g.p >= 0.0f)     { ++g.own[draft_conf_bin(g.p)][0];      g.own[draft_conf_bin(g.p)][1] += hit; }
        if (g.prev >= 0.0f)  { ++g.dec[draft_conf_bin(g.prev)][0];   g.dec[draft_conf_bin(g.prev)][1] += hit; }
        if (g.prev1 >= 0.0f) { ++g.dec1[draft_conf_bin(g.prev1)][0]; g.dec1[draft_conf_bin(g.prev1)][1] += hit; }
        g.fresh = false;
    }
}

/* The logistic of the head's logit, guarded against overflow. */
inline float draft_conf_sigmoid(double z) {
    if (z >= 40.0) return 1.0f;
    if (z <= -40.0) return 0.0f;
    return (float)(1.0 / (1.0 + std::exp(-z)));
}

/* RADIANCE_EXPERIMENTAL_DS41_DSPARK_LOAD: unset, "" or "0" off (0); "1" from two sequences; "N"
 * (2..64) from N. False for anything else. */
inline bool draft_load_parse(const char* s, int* from) {
    *from = 0;
    if (!s || !*s || !std::strcmp(s, "0")) return true;
    if (!std::strcmp(s, "1")) { *from = 2; return true; }
    char* e = nullptr;
    const long n = std::strtol(s, &e, 10);
    if (e == s || *e || n < 2 || n > 64) return false;
    *from = (int)n;
    return true;
}

/* RADIANCE_EXPERIMENTAL_DS41_DSPARK_CTX_DEFER: unset, "" or "0" off; "1" on. False otherwise. */
inline bool draft_ctx_defer_parse(const char* s, bool* on) {
    *on = false;
    if (!s || !*s || !std::strcmp(s, "0")) return true;
    if (!std::strcmp(s, "1")) { *on = true; return true; }
    return false;
}

/* The context pass this step (Scheduler::block_begin decides, block_context builds). */
enum DraftCtxMode { kDraftCtxRun = 0, kDraftCtxSkip, kDraftCtxCatchUp };

/* The deferral's bookkeeping, in the core's count of the plugin's waiting rows (Dsp::tap_rows). */
struct DraftCtxDefer {
    bool     on = false;
    int64_t  cap = 0;          /* rows that may wait: the plugin's source rows less the step budget */
    int64_t  rows = 0;         /* rows the trunk appended since the last pass */
    uint64_t seq = 0;          /* while `clean`: the one sequence they belong to, from `from` */
    int64_t  from = -1;
    bool     clean = true;
    int      mode = kDraftCtxRun;
};

/* Whether the rows waiting plus this step's `t` can still wait for a later pass: the plugin holds
 * cap + one step budget, and the next trunk step may append a whole budget. */
inline bool draft_ctx_may_wait(const DraftCtxDefer& d, int64_t t) {
    return d.rows + t <= d.cap;
}

/* Every controller's totals over the run, for metrics and the shutdown report. */
struct DraftConfStats {
    bool    on = false, deferred = false;
    int     load_from = 0;
    double  threshold = 0.0;
    int     probe = 0;
    int64_t seqs = 0;
    int64_t asks = 0, warm = 0, open = 0, probes = 0, gated = 0, closes = 0, reopens = 0;
    int64_t scored = 0, verifies = 0, verified = 0, accepted = 0, probe_verified = 0, probe_accepted = 0;
    int64_t own[kDraftConfBins][2] = {}, dec[kDraftConfBins][2] = {}, dec1[kDraftConfBins][2] = {};
    /* Load-aware: steps in which N or more sequences decoded, and the drafts it held back. */
    int64_t load_steps = 0, load_held = 0;
    /* The context pass: run, skipped, caught up (and the rows those covered), dropped (and the
     * rows lost), and run only because the rows waiting had reached the cap. */
    int64_t ctx_run = 0, ctx_skip = 0, ctx_catch = 0, ctx_catch_rows = 0, ctx_drop = 0,
            ctx_drop_rows = 0, ctx_full = 0;
    /* Decode steps of one sequence by whether they verified head drafts and whether a query pass ran
     * behind them, with their wall time (as DraftGateStats). */
    int64_t step_n[2][2] = {}, step_ns[2][2] = {};

    void add(const DraftConfSeq& g) {
        ++seqs;
        asks += g.asks; warm += g.warm; open += g.open; probes += g.probes;
        gated += g.gated; closes += g.closes; reopens += g.reopens; scored += g.scored;
        verifies += g.verifies; verified += g.verified; accepted += g.accepted;
        probe_verified += g.probe_verified; probe_accepted += g.probe_accepted;
        for (int b = 0; b < kDraftConfBins; ++b)
            for (int k = 0; k < 2; ++k) {
                own[b][k] += g.own[b][k];
                dec[b][k] += g.dec[b][k];
                dec1[b][k] += g.dec1[b][k];
            }
    }
    double step_ms(int v, int d) const {
        return step_n[v][d] > 0 ? (double)step_ns[v][d] * 1e-6 / (double)step_n[v][d] : 0.0;
    }
};

}  /* namespace rad */
