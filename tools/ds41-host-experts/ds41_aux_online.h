/* ds41_aux_online.h -- the aux expert worker's online unit set (work/aux-dyn/SPEC.md part B): the decayed scores,
 * the swap planner, the swap credit and the copy budget. Header-only, plain C++17, no HIP and no core types (the
 * ds41_aux.h pattern): core/place/aux_control.cpp, the host tests and the replay's fixture share the rule.
 *
 * The rule (B.3, B.4), the same operations in the same order as work/aux-dyn/replay/q4_online_rank.py's
 * sim_engine(), so the C++ planner and the replay agree bit for bit (tests/aux_online_test.cpp holds them to it on a
 * slice of the Oct 29 traces):
 *   score(u)  = hits(u) + prior(u), prior(u) = 6 * share(u) * n0 (doubles; share the plan's float);
 *   a merge of T tokens: hits *= 0.5 ^ (T / H), then hits += the rows routed to u since the last merge;
 *   every EVERY merges with tokens: credit = min(MAX, credit + RATE / 100 * T); n = floor(credit), at most the free
 *   slots; per rank (a rank whose quota is filled, whose lane lives and whose guard holds) ins = the n best
 *   non-resident candidates with hits > 0 (score desc, then ds41aux::before), outs = the n worst resident ones
 *   (score asc, then ds41aux::before reversed); pairs across ranks by gain score(in) - score(out), the best first
 *   (ties: the lower rank), while score(in) > MARGIN / 100 * score(out); credit -= pairs. */
#pragma once
#include "ds41_aux.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace ds41aux {
namespace online {

/* 6 selections a token: the replay's per-token rate of a unit with profile share s is 6 * s. */
constexpr double kSelections = 6.0;
/* A unit whose lease failed at copy time is left out of the ins for this many decisions. */
constexpr int kCoolDown = 64;

/* One rank's candidates (by candidate index: the rank's list in ds41aux::before order). */
struct RankScores {
    std::vector<double> hits, prior;
    void init(const std::vector<Cand>& cands, int64_t n0) {
        hits.assign(cands.size(), 0.0);
        prior.resize(cands.size());
        for (size_t i = 0; i < cands.size(); ++i) prior[i] = kSelections * (double)cands[i].share * (double)n0;
    }
    double score(size_t i) const { return hits[i] + prior[i]; }
};

inline double decay_factor(uint64_t tokens, int64_t halflife) {
    return std::pow(0.5, (double)tokens / (double)halflife);
}
/* A merge: `delta` the rows each candidate met since the last one (null: none), `f` decay_factor(T, H). */
inline void merge(RankScores* s, const uint32_t* delta, double f) {
    const size_t n = s->hits.size();
    if (f != 1.0)
        for (size_t i = 0; i < n; ++i) s->hits[i] = s->hits[i] * f;
    if (delta)
        for (size_t i = 0; i < n; ++i)
            if (delta[i]) s->hits[i] = s->hits[i] + (double)delta[i];
}

/* A candidate's place for the planner. */
enum : uint8_t {
    kNon = 0,      /* not resident, may come in (when hit) */
    kRes = 1,      /* resident, may leave */
    kPinned = 2,   /* neither: copying in, chosen to leave, cooling down, or not cached */
};

struct RankIn {
    const RankScores*        s = nullptr;
    const std::vector<Cand>* cand = nullptr;
    const uint8_t*           state = nullptr;   /* by candidate index */
    bool                     ok = false;        /* planned at all */
};
struct Pair {
    int     rank = -1;
    int32_t in = -1, out = -1;   /* candidate indices */
    double  gain = 0;
};

/* B.4 steps 2-3: at most n pairs. */
inline std::vector<Pair> plan(const std::vector<RankIn>& ranks, int n, int64_t margin_pct) {
    std::vector<Pair> chosen;
    if (n <= 0) return chosen;
    const double margin = (double)margin_pct / 100.0;
    const size_t R = ranks.size();
    std::vector<std::vector<int32_t>> ins(R), outs(R);
    for (size_t r = 0; r < R; ++r) {
        const RankIn& k = ranks[r];
        if (!k.ok || !k.s || !k.cand || !k.state) continue;
        const RankScores& s = *k.s;
        const std::vector<Cand>& c = *k.cand;
        std::vector<int32_t> a, b;
        for (size_t i = 0; i < s.hits.size(); ++i) {
            if (k.state[i] == kNon && s.hits[i] > 0.0) a.push_back((int32_t)i);
            else if (k.state[i] == kRes) b.push_back((int32_t)i);
        }
        auto better_in = [&](int32_t x, int32_t y) {
            const double sx = s.score((size_t)x), sy = s.score((size_t)y);
            if (sx != sy) return sx > sy;
            return before(c[(size_t)x], c[(size_t)y]);
        };
        auto worse_out = [&](int32_t x, int32_t y) {
            const double sx = s.score((size_t)x), sy = s.score((size_t)y);
            if (sx != sy) return sx < sy;
            return before(c[(size_t)y], c[(size_t)x]);
        };
        const size_t na = std::min(a.size(), (size_t)n), nb = std::min(b.size(), (size_t)n);
        std::partial_sort(a.begin(), a.begin() + (ptrdiff_t)na, a.end(), better_in);
        std::partial_sort(b.begin(), b.begin() + (ptrdiff_t)nb, b.end(), worse_out);
        a.resize(na);
        b.resize(nb);
        ins[r].swap(a);
        outs[r].swap(b);
    }
    std::vector<size_t> idx(R, 0);
    while ((int)chosen.size() < n) {
        int best = -1;
        double best_gain = 0;
        for (size_t r = 0; r < R; ++r) {
            const size_t i = idx[r];
            if (i >= ins[r].size() || i >= outs[r].size()) continue;
            const double sa = ranks[r].s->score((size_t)ins[r][i]), sb = ranks[r].s->score((size_t)outs[r][i]);
            if (!(sa > sb * margin)) continue;
            const double g = sa - sb;
            if (best < 0 || g > best_gain) {
                best = (int)r;
                best_gain = g;
            }
        }
        if (best < 0) break;
        Pair p;
        p.rank = best;
        p.in = ins[(size_t)best][idx[(size_t)best]];
        p.out = outs[(size_t)best][idx[(size_t)best]];
        p.gain = best_gain;
        ++idx[(size_t)best];
        chosen.push_back(p);
    }
    return chosen;
}

/* B.4 step 1: the swap credit, RATE swaps per 100 tokens, at most MAX banked. */
struct Credit {
    double credit = 0;
    /* After a merge of T tokens: the swaps this decision may make (before the free slots' bound). */
    int grant(uint64_t tokens, int64_t rate, int64_t max) {
        credit = std::min((double)max, credit + (double)rate / 100.0 * (double)tokens);
        return (int)std::floor(credit);
    }
    void spend(int n) { credit -= (double)n; }
};

/* A copy budget in bytes per second, one second deep (B.5: _ONLINE_MBPS for swaps, _SHARE_FILL_MBPS for fills). */
struct Bucket {
    double  rate = 0, level = 0, depth = 0;   /* bytes per ns, bytes, bytes */
    int64_t last_ns = -1;
    void init(int64_t mbps) {
        rate = (double)mbps * 1e6 / 1e9;
        depth = (double)mbps * 1e6;
        level = depth;
        last_ns = -1;
    }
    void refill(int64_t now_ns) {
        if (last_ns >= 0 && now_ns > last_ns) level = std::min(depth, level + rate * (double)(now_ns - last_ns));
        last_ns = now_ns;
    }
    /* Takes `bytes` when the level allows (a copy larger than the depth goes once the bucket is full). */
    bool take(int64_t now_ns, int64_t bytes) {
        refill(now_ns);
        if (level >= (double)bytes || level >= depth) {
            level -= (double)bytes;
            return true;
        }
        return false;
    }
};

}  // namespace online
}  // namespace ds41aux
