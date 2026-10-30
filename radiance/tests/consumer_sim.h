/* consumer_sim.h -- a brute-force dependency simulation of the consumer-suffix skip
 * (core/sched/consumer_skip.h), row by row and layer by layer, shared by consumer_skip_test (the
 * schedule over random grids) and sched_test (the scheduler's own passes and snapshots).
 *
 * The model is the declared contract (abi/rad_builder.h, rad_declare_consumer_suffix): n layers,
 * boundary B, window W. A pass [s, e) with F runs layers [0, F) in full on its rows and layer F's
 * key ops alone. In a layer a pass runs, the key ops come first: each row's key is stored from
 * the row's input to the layer (exact iff that input is). A full layer's row is exact iff its
 * input is, every key of the layer at q - W + 1 .. q is exact, and (past B) every key of B so far
 * is. A row of layer L's input is layer L-1's output; layer 0's is the embedding (exact). Keys are
 * written once, by the pass holding their row; positions before the resume point p0 hold a prefix
 * hit's restored state, exact for every layer.
 *
 * A snapshot at P (always a pass end) holds every layer's keys at max(0, P - W) .. P - 1 and the
 * state of each layer at or below B (a recurrent state folds in every row, so it is the base
 * engine's iff every key of that layer below P is: the stand-in here); it is the base engine's iff
 * all of those are exact. Keys below P are final once the pass ending at P ran, so they are read
 * from the end state. Pure host arithmetic. */
#pragma once

#include "sched/consumer_skip.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace rad {
namespace csim {

using Pass = std::pair<int64_t, int64_t>;   /* [s, e) */

struct Sim {
    bool ok = true;
    std::string why;
    int64_t garbage = 0;       /* computed rows or stored keys that are not the base engine's */
    int64_t stale = 0;         /* full rows that read a key never written in this request */
    int64_t full = 0, key = 0, total = 0;   /* layer-passes */
    std::vector<int32_t> F;    /* each pass's F */
    std::vector<int64_t> bad_snaps;   /* snapshots that are not the base engine's */
};

/* Runs the passes `g` of one prompt of `n` rows resumed at `p0` (a multiple of the grid, or 0),
 * pass k with F[k] (n_layer: every layer), and checks: every computed row and stored key exact, no
 * row reading an unwritten key, the prompt's last row exact (its last pass must run every layer),
 * the window a decode at n reads complete, and every snapshot in `snaps` (pass ends, or n) the base
 * engine's. */
inline Sim simulate_passes(const ConsumerShape& c, int64_t n, int64_t p0, const std::vector<Pass>& g,
                           const std::vector<int32_t>& F, const std::vector<int64_t>& snaps) {
    Sim out;
    const int32_t nl = c.n_layer, B = c.boundary;
    const int64_t W = c.window;
    auto fail = [&](const std::string& w) {
        if (out.ok) out.why = w;
        out.ok = false;
    };
    if (F.size() != g.size()) { fail("one F a pass"); return out; }
    /* key[L][p]: 0 never written, 1 the base engine's bytes, 2 anything else. pre[L] counts exact
     * keys over [0, p), settled up to upto[L]. */
    std::vector<std::vector<uint8_t>> key((size_t)nl, std::vector<uint8_t>((size_t)n, 0));
    std::vector<std::vector<int32_t>> pre((size_t)nl, std::vector<int32_t>((size_t)n + 1, 0));
    std::vector<int64_t> upto((size_t)nl, 0);
    for (int32_t L = 0; L < nl; ++L)
        for (int64_t p = 0; p < p0 && p < n; ++p) key[(size_t)L][(size_t)p] = 1;
    auto settle = [&](int32_t L, int64_t to) {
        auto& k = key[(size_t)L];
        auto& q = pre[(size_t)L];
        for (int64_t p = upto[(size_t)L]; p < to; ++p) q[(size_t)p + 1] = q[(size_t)p] + (k[(size_t)p] == 1);
        upto[(size_t)L] = std::max(upto[(size_t)L], to);
    };
    bool bkeys = true;   /* every key of B written so far is exact */
    std::vector<uint8_t> in, nx;   /* this pass's rows into and out of a layer: 1 exact, 2 not */
    for (size_t k = 0; k < g.size(); ++k) {
        const int64_t s = g[k].first, e = g[k].second, T = e - s;
        const int32_t f = F[k];
        out.F.push_back(f);
        if (f < B || f > nl) { fail("F outside [B, n]"); return out; }
        if (s < 0 || e > n || T <= 0) { fail("a pass outside the prompt"); return out; }
        out.total += nl;
        in.assign((size_t)T, 1);   /* the embedding: every row exact */
        for (int32_t L = 0; L < nl; ++L) {
            const bool full = L < f, keys = full || L == f;
            if (!keys) break;
            settle(L, s);
            for (int64_t i = 0; i < T; ++i) {
                uint8_t& kk = key[(size_t)L][(size_t)(s + i)];
                if (kk != 0) { fail("a key written twice"); return out; }
                kk = in[(size_t)i];
                if (kk != 1) ++out.garbage;
                if (L == B && kk != 1) bkeys = false;
            }
            settle(L, e);
            if (!full) { ++out.key; break; }
            ++out.full;
            nx.assign((size_t)T, 1);
            for (int64_t i = 0; i < T; ++i) {
                const int64_t q = s + i, lo = std::max<int64_t>(0, q - W + 1);
                const int64_t exact = pre[(size_t)L][(size_t)q + 1] - pre[(size_t)L][(size_t)lo];
                bool unwritten = false;
                if (exact != q + 1 - lo)
                    for (int64_t p = lo; p <= q; ++p) unwritten |= key[(size_t)L][(size_t)p] == 0;
                const bool row_ok = in[(size_t)i] == 1 && exact == q + 1 - lo && (L < B || bkeys);
                nx[(size_t)i] = row_ok ? 1 : 2;
                if (!row_ok) ++out.garbage;
                if (unwritten) ++out.stale;
            }
            in.swap(nx);
        }
        if (f == nl && e == n && in[(size_t)(T - 1)] != 1) fail("the prompt's last row is not exact");
        if (f < nl && e == n) fail("the prompt's last pass was cut");
    }
    if (out.garbage) fail("a computed row or stored key is not the base engine's");
    if (out.stale) fail("a computed row read a key this request never wrote");
    for (int32_t L = 0; L < nl; ++L) settle(L, n);
    auto whole = [&](int64_t P) {
        for (int32_t L = 0; L < nl; ++L) {
            const int64_t lo = std::max<int64_t>(0, P - W);
            if (pre[(size_t)L][(size_t)P] - pre[(size_t)L][(size_t)lo] != P - lo) return false;
            if (L <= B && pre[(size_t)L][(size_t)P] != P) return false;   /* its state */
        }
        return true;
    };
    if (!whole(n)) fail("the decode window is incomplete");
    for (int64_t P : snaps) {
        bool at_end = P == n;
        for (const Pass& ps : g) at_end |= ps.second == P;
        if (P < 1 || P > n || (!at_end && P > p0)) { fail("a snapshot off the passes' ends"); continue; }
        if (!whole(P)) {
            out.bad_snaps.push_back(P);
            fail("a snapshot is not the base engine's");
        }
    }
    if (!bkeys) fail("a boundary key is not exact");
    return out;
}

/* Each pass's F under the schedule: the target's thresholds `f` and, for a pass with a
 * checkpoint of `whole` (ascending, below the target) at or after its end, the first such
 * checkpoint's (consumer_skip.h, "complete checkpoints"). Empty `f`: every layer. */
inline std::vector<int32_t> schedule_F(const ConsumerShape& c, const std::vector<Pass>& g,
                                       const std::vector<int64_t>& f,
                                       const std::vector<int64_t>& whole) {
    std::vector<int32_t> F;
    F.reserve(g.size());
    int64_t gk = -1;
    std::vector<int64_t> gt;
    for (const Pass& ps : g) {
        if (f.empty()) { F.push_back(c.n_layer); continue; }
        const auto it = std::lower_bound(whole.begin(), whole.end(), ps.second);
        if (it == whole.end()) { F.push_back(consumer_full_layers(c, f, ps.second - 1)); continue; }
        if (gk != *it) { gt = consumer_ckpt_thresholds(c, *it); gk = *it; }
        F.push_back(consumer_pass_layers(c, f, gt, ps.second - 1));
    }
    return F;
}

/* Layer-passes run in full over the passes (every layer below F; layer F's key ops apart). */
inline int64_t full_layer_passes(const std::vector<int32_t>& F) {
    int64_t s = 0;
    for (int32_t f : F) s += f;
    return s;
}

/* DeepSeek-V4.1 as the ds41 A/Bs serve it: an idle prefill from `p0` to `n` in 16-token chunks on
 * the 16 grid (--max-num-batched-tokens 16), with linear state beside the window, and the
 * snapshots the scheduler's plan_chunk writes at checkpoint interval `I` (a multiple of 16):
 * --prefix-resume nearest at each chunk end on the interval, and at the prompt's end when it ends
 * on the grid at least max(16, I/4) past the interval below (the tail rule; a tail inside the last
 * chunk is never a chunk end); exact at each chunk end on the interval and at the tail, the start
 * of the last chunk, past the first interval. sched_test holds the real scheduler to this list. */
inline std::vector<int64_t> ds41_idle_snapshots(int64_t p0, int64_t n, int64_t I, bool exact) {
    std::vector<int64_t> out;
    const int64_t tail = (n - 1) / 16 * 16;
    for (int64_t pos = p0; pos < n;) {
        const int64_t e = std::min<int64_t>(n, pos + 16);
        bool ck = e % I == 0;
        if (exact && e == tail && tail > I) ck = true;
        if (!exact && e == n && n % 16 == 0 && n > I && n % I >= std::max<int64_t>(16, I / 4))
            ck = true;
        if (ck) out.push_back(e);
        pos = e;
    }
    return out;
}

/* Its 16-row passes. */
inline std::vector<Pass> ds41_idle_passes(int64_t p0, int64_t n) {
    std::vector<Pass> g;
    for (int64_t pos = p0; pos < n; pos = std::min<int64_t>(n, pos + 16))
        g.push_back({ pos, std::min<int64_t>(n, pos + 16) });
    return g;
}

}  /* namespace csim */
}  /* namespace rad */
