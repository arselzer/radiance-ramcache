/* draft_gate.h -- RADIANCE_EXPERIMENTAL_DS41_DSPARK_GATE: a per-sequence running-mean acceptance
 * gate for a block drafter (DSpark).
 *
 * A drafting step pays when 1 + a > rho + d: it emits 1 + a tokens (a: drafts accepted) for a
 * verify pass of 1 + k rows (rho plain steps) plus the draft (d). On ds41's best settings with the
 * bank pinned (window ab-20261029T220909Z, cap 1 with K1's draft GEMV) a drafting step took 110.5
 * ms against 62.1 ms for a plain one on the held-out prompts, 126.4 against 71.4 on chat: rho + d
 * = 1.78, so a draft pays only above 0.78 accepted a verify. Acceptance per reply ran from 0.36 to
 * 1.00 in the same window, so any one policy for every reply loses on some of them.
 *
 * The gate keeps, per sequence, the drafts accepted by each of its last W verifies of head drafts.
 * While their mean is under the threshold the sequence drafts nothing: its query pass is skipped
 * and the step after verifies no drafts for it, while the context pass, which fills the drafter's
 * window cache for every committed position, still runs. Every R-th step of a closed run it drafts
 * anyway, a probe whose verify enters the mean, so a reply that turns repetitive reopens. A
 * sequence starts open and stays open until W verifies are in.
 *
 * The decision is the scheduler's, in Scheduler::block_begin, before the query pass is staged.
 * DSpark proposes its whole block in one pass, so the draft-side decision is per block (a
 * per-token stop would save nothing), and a running mean needs no per-step signal from the draft
 * just run. At that point the host has committed the step before (the open step's acceptance is
 * still on the card), so the mean lags one verify; nothing waits on the card, and the issue-ahead
 * pipeline plans the next verify from the drafter's rows as before: a declined sequence gets no
 * expect_draft(), so the step issued ahead verifies 0 drafts for it.
 *
 * Exactness: only how many drafts a step verifies changes. Every emitted token is the target's
 * (core/sample/accept.h), and a row's result does not depend on how many rows its pass carries.
 */
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace rad {

inline constexpr char kDraftGateEnv[] = "RADIANCE_EXPERIMENTAL_DS41_DSPARK_GATE";

struct DraftGateCfg {
    enum { kMaxWindow = 64, kMaxProbe = 4096 };
    /* The defaults. Threshold: rho + d - 1 from the step times above (the cap-1 break-even with
     * the full block; K1's two-row block breaks even at 0.70, and a deeper cap has its own, since
     * the unit is drafts accepted a verify). Window: about the drafts it takes to tell a 0.4 reply
     * from a 0.8 one; at 8 and 0.78 a sequence drafts with at most one rejection among its last 8
     * verifies. Probe: one step in 8 of a closed run, which costs a reply that accepts 0.4 about
     * (1.78 - 1.4) / 8 = 0.05 of a plain step a step at today's costs. */
    static constexpr double kThreshold = 0.78;
    static constexpr int kWindow = 8;
    static constexpr int kProbe = 8;

    bool   on = false;
    double threshold = kThreshold;   /* mean drafts accepted a verify, at or above which it drafts */
    int    window = kWindow;         /* W: the verifies the mean is over */
    int    probe = kProbe;           /* R: a closed run drafts every R-th step */

    /* Unset, "" or "0": off. "1": on at the defaults. Otherwise "T[,W[,R]]": the threshold T in
     * (0, 64] (so 1.0, not 1, for a threshold of one), W in 1..64, R in 1..4096. False for
     * anything else, which the engine refuses at startup. */
    static bool parse(const char* s, DraftGateCfg* out) {
        DraftGateCfg c;
        if (!s || !*s || !std::strcmp(s, "0")) { *out = c; return true; }
        c.on = true;
        if (!std::strcmp(s, "1")) { *out = c; return true; }
        char* e = nullptr;
        const double t = std::strtod(s, &e);
        if (e == s || !(t > 0.0) || !(t <= 64.0)) return false;   /* NaN fails both */
        c.threshold = t;
        if (*e == ',') {
            const char* p = e + 1;
            const long w = std::strtol(p, &e, 10);
            if (e == p || w < 1 || w > kMaxWindow) return false;
            c.window = (int)w;
            if (*e == ',') {
                p = e + 1;
                const long r = std::strtol(p, &e, 10);
                if (e == p || r < 1 || r > kMaxProbe) return false;
                c.probe = (int)r;
            }
        }
        if (*e) return false;
        *out = c;
        return true;
    }
};

/* What an ask decided: no draft, or a draft while warming up (fewer than W verifies in), open, or
 * as a closed run's probe. */
enum DraftGateKind { kDraftGateGated = 0, kDraftGateWarm, kDraftGateOpen, kDraftGateProbe };

/* One sequence's gate. It describes the text, so preemption keeps it and a new request starts it
 * over (Scheduler::add_locked). */
struct DraftGateSeq {
    /* The drafts accepted by each of its last `window` verifies of head drafts: a ring of `n`
     * entries from `head`, summing to `sum`. */
    uint8_t acc[DraftGateCfg::kMaxWindow] = {};
    int32_t n = 0, head = 0, sum = 0;
    /* Steps asked since the gate closed; 0 while open. */
    int32_t closed = 0;
    /* The scheduler step whose verify carries this sequence's probe draft, -1 for none: the
     * probe's outcome told apart from an open draft's (Scheduler::block_begin). */
    int64_t probe_step = -1;
    /* Counters. Steps asked; of them drafted while warming up, open, or as a probe, and gated;
     * closed runs begun and ended; verifies of head drafts with the drafts they verified and
     * accepted, and of those the probes'. */
    int64_t asks = 0, warm = 0, open = 0, probes = 0, gated = 0, closes = 0, reopens = 0;
    int64_t verifies = 0, verified = 0, accepted = 0, probe_verified = 0, probe_accepted = 0;
};

/* The mean drafts accepted a verify over what the window holds; 0 when empty. */
inline double draft_gate_mean(const DraftGateSeq& g) {
    return g.n > 0 ? (double)g.sum / (double)g.n : 0.0;
}

/* Whether the window's mean is at the threshold or above. The sum is a whole number and
 * threshold * n need not be, so equality is decided with a little slack. */
inline bool draft_gate_open(const DraftGateCfg& c, const DraftGateSeq& g) {
    return (double)g.sum + 1e-9 >= c.threshold * (double)g.n;
}

/* Whether the sequence drafts this step, and why (DraftGateKind); counts the ask. */
inline int draft_gate_ask(const DraftGateCfg& c, DraftGateSeq& g) {
    ++g.asks;
    if (g.n < c.window) { ++g.warm; return kDraftGateWarm; }
    if (draft_gate_open(c, g)) {
        if (g.closed) { ++g.reopens; g.closed = 0; }
        ++g.open;
        return kDraftGateOpen;
    }
    if (!g.closed) ++g.closes;
    ++g.closed;
    if (g.closed % c.probe == 0) { ++g.probes; return kDraftGateProbe; }
    ++g.gated;
    return kDraftGateGated;
}

/* An ask that drafted whose draft was not staged after all (Scheduler::block_begin: its KV
 * refused). Its count goes; the closed run it may have extended stays. */
inline void draft_gate_unask(DraftGateSeq& g, int kind) {
    switch (kind) {
    case kDraftGateWarm:  --g.warm; break;
    case kDraftGateOpen:  --g.open; break;
    case kDraftGateProbe: --g.probes; break;
    default: return;
    }
    --g.asks;
}

/* A verify of `verified` head drafts, `accepted` of them taken; `probe` when it verified a
 * probe's draft. */
inline void draft_gate_observe(const DraftGateCfg& c, DraftGateSeq& g, int verified, int accepted,
                               bool probe) {
    if (verified <= 0 || c.window < 1 || c.window > DraftGateCfg::kMaxWindow) return;
    if (accepted < 0) accepted = 0;
    if (accepted > verified) accepted = verified;
    ++g.verifies;
    g.verified += verified;
    g.accepted += accepted;
    if (probe) {
        g.probe_verified += verified;
        g.probe_accepted += accepted;
    }
    const int a = accepted > 255 ? 255 : accepted;
    if (g.n < c.window) {
        g.acc[(g.head + g.n) % c.window] = (uint8_t)a;
        ++g.n;
        g.sum += a;
    } else {
        g.sum += a - (int)g.acc[g.head];
        g.acc[g.head] = (uint8_t)a;
        g.head = (g.head + 1) % c.window;
    }
}

/* The gate over every sequence, for metrics and the shutdown report: the sequences' counters
 * summed, and decode steps of one sequence by whether they verified drafts and whether their
 * drafter ran a query pass, with their wall time (Scheduler::step_finished). steady[1][1] / [0][0]
 * is rho + d as measured, the break-even acceptance plus one. */
struct DraftGateStats {
    bool    on = false;
    double  threshold = 0.0;
    int     window = 0, probe = 0;
    int64_t seqs = 0;
    int64_t asks = 0, warm = 0, open = 0, probes = 0, gated = 0, closes = 0, reopens = 0;
    int64_t verifies = 0, verified = 0, accepted = 0, probe_verified = 0, probe_accepted = 0;
    int64_t step_n[2][2] = {}, step_ns[2][2] = {};   /* [verified drafts][drafted] */

    void add(const DraftGateSeq& g) {
        ++seqs;
        asks += g.asks; warm += g.warm; open += g.open; probes += g.probes; gated += g.gated;
        closes += g.closes; reopens += g.reopens;
        verifies += g.verifies; verified += g.verified; accepted += g.accepted;
        probe_verified += g.probe_verified; probe_accepted += g.probe_accepted;
    }
    double step_ms(int v, int d) const {
        return step_n[v][d] > 0 ? (double)step_ns[v][d] * 1e-6 / (double)step_n[v][d] : 0.0;
    }
};

}  /* namespace rad */
