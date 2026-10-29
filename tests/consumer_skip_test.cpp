/* consumer_skip_test.cpp -- the consumer-suffix schedule (core/sched/consumer_skip.h) against a
 * brute-force dependency simulation, row by row and layer by layer, over pass grids the engine's
 * rules allow (passes of at most M rows starting on a multiple of A, every one but the prompt's
 * last ending on one, a split at the tail checkpoint, a resume from a prefix hit):
 *   - every row the schedule computes, and every key it stores, is the base engine's (the
 *     simulation tracks "exact" through every read: a layer's input row, its window of W keys, the
 *     boundary's keys);
 *   - no computed row reads a key this request did not write;
 *   - the prompt's last row is exact, and so is every layer's window the next step reads (the last
 *     W positions) and the one a checkpoint at the target holds;
 *   - at M = A = 16, W = 128 it is PREFILL-ENGINE-SURVEY.md's schedule (5,400 of 7,720 layer-passes
 *     of a 3,086-token prompt), and at one-row passes the minimal dependency cone;
 *   - and the simulation is not blind: a schedule one window position short, or one that assumes
 *     narrower passes than the grid has, is caught, and so is a checkpoint below the target.
 * Pure host arithmetic. */
#include "rad_test.h"

#include "sched/consumer_skip.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

using namespace rad;

namespace {

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 6364136223846793005ull + 1442695040888963407ull) {}
    uint64_t next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return s >> 17;
    }
    /* Uniform in [lo, hi]. */
    int64_t in(int64_t lo, int64_t hi) { return hi <= lo ? lo : lo + (int64_t)(next() % (uint64_t)(hi - lo + 1)); }
};

using Pass = std::pair<int64_t, int64_t>;   /* [s, e) */

/* A pass grid from `p0` (a multiple of A) to `n` under the engine's rules: a pass holds at most M
 * rows, every pass but the last ends on a multiple of A, and a pass ends at `split` when p0 <
 * split < n (the scheduler cuts a chunk at its tail checkpoint). `widest`: each pass as long as
 * the rules allow (an idle engine's grid); otherwise a random length (a step shared with others,
 * a capped chunk). */
std::vector<Pass> grid(int64_t p0, int64_t n, int64_t M, int64_t A, int64_t split, bool widest,
                       Rng& r) {
    std::vector<Pass> g;
    int64_t s = p0;
    while (s < n) {
        int64_t limit = std::min(n, s + M);
        const bool forced = split > s && split < n && split <= limit;
        if (forced) limit = split;
        int64_t e;
        if (limit == n || forced) {
            /* The end itself, or (randomly) an aligned point before it. */
            const int64_t top = (limit - 1) / A * A;   /* the last multiple of A short of limit */
            if (!widest && top > s && r.in(0, 2) == 0) e = s + A * r.in(1, (top - s) / A);
            else e = limit;
        } else {
            const int64_t top = limit / A * A;
            CHECK(top > s);   /* M >= A and s on the grid: one quantum always fits */
            if (top <= s) break;
            e = widest ? top : s + A * r.in(1, (top - s) / A);
        }
        g.push_back({ s, e });
        s = e;
    }
    return g;
}

struct Sim {
    bool ok = true;
    std::string why;
    int64_t garbage = 0;       /* computed rows or stored keys that are not the base engine's */
    int64_t stale = 0;         /* full rows that read a key never written in this request */
    int64_t full = 0, key = 0, total = 0;   /* layer-passes */
    std::vector<int32_t> F;    /* each pass's F */
};

/* Runs the passes of one prompt, layer by layer, with the schedule's F for each pass (or every
 * layer when `f` is empty), and checks what the schedule promises. `target` is the cut's target:
 * the prompt's end `n`, or a checkpoint below it whose window must be complete. */
Sim simulate(const ConsumerShape& c, int64_t n, int64_t target, int64_t p0,
             const std::vector<Pass>& g, const std::vector<int64_t>& f) {
    Sim out;
    const int32_t nl = c.n_layer, B = c.boundary;
    const int64_t W = c.window;
    /* key[L][p]: 0 never written, 1 the base engine's bytes, 2 anything else. Positions before p0
     * are a prefix hit's restored state, complete. pre[L] counts exact keys over [0, p). */
    std::vector<std::vector<uint8_t>> key((size_t)nl, std::vector<uint8_t>((size_t)n, 0));
    std::vector<std::vector<int32_t>> pre((size_t)nl, std::vector<int32_t>((size_t)n + 1, 0));
    std::vector<int64_t> upto((size_t)nl, 0);
    for (int32_t L = 0; L < nl; ++L)
        for (int64_t p = 0; p < p0; ++p) key[(size_t)L][(size_t)p] = 1;
    auto settle = [&](int32_t L, int64_t to) {   /* pre[L] final up to `to` */
        auto& k = key[(size_t)L];
        auto& q = pre[(size_t)L];
        for (int64_t p = upto[(size_t)L]; p < to; ++p) q[(size_t)p + 1] = q[(size_t)p] + (k[(size_t)p] == 1);
        upto[(size_t)L] = std::max(upto[(size_t)L], to);
    };
    bool bkeys = true;   /* every key of B written so far is exact */
    auto fail = [&](const std::string& w) {
        if (out.ok) out.why = w;
        out.ok = false;
    };
    std::vector<uint8_t> in, nx;   /* this pass's rows into and out of a layer: 0 none, 1 exact, 2 not */
    for (const Pass& ps : g) {
        const int64_t s = ps.first, e = ps.second, T = e - s;
        const int32_t F = f.empty() ? nl : consumer_full_layers(c, f, e - 1);
        out.F.push_back(F);
        if (F < B || F > nl) { fail("F outside [B, n]"); return out; }
        out.total += nl;
        in.assign((size_t)T, 1);   /* the embedding: every row exact */
        for (int32_t L = 0; L < nl; ++L) {
            const bool full = L < F, keys = full || L == F;
            if (!keys) break;
            for (int64_t i = 0; i < T; ++i)
                if (in[(size_t)i] == 0) { fail("a layer runs on a row its previous layer did not compute"); return out; }
            /* The key ops, before any row's attention. */
            settle(L, s);
            for (int64_t i = 0; i < T; ++i) {
                uint8_t& k = key[(size_t)L][(size_t)(s + i)];
                if (k != 0) { fail("a key written twice"); return out; }
                k = in[(size_t)i];
                if (k != 1) ++out.garbage;
                if (L == B && k != 1) bkeys = false;
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
        if (F == nl && e == n && in[(size_t)(T - 1)] != 1) fail("the prompt's last row is not exact");
        if (F < nl && e == n) fail("the prompt's last pass was cut");
    }
    if (out.garbage) fail("a computed row or stored key is not the base engine's");
    if (out.stale) fail("a computed row read a key this request never wrote");
    /* The windows the next step and the checkpoint at the target read. */
    for (int32_t L = 0; L < nl; ++L) {
        for (int64_t p = std::max<int64_t>(0, n - W); p < n; ++p)
            if (key[(size_t)L][(size_t)p] != 1) { fail("the decode window is incomplete"); break; }
        for (int64_t p = std::max<int64_t>(0, target - W); p < target; ++p)
            if (key[(size_t)L][(size_t)p] != 1) { fail("the target's window is incomplete"); break; }
    }
    if (!bkeys) fail("a boundary key is not exact");
    return out;
}

/* The tail checkpoint a scheduler would cut a prompt at: some multiple of A past p0 and short of
 * n, or n (none). */
int64_t pick_target(int64_t p0, int64_t n, int64_t A, Rng& r) {
    const int64_t top = (n - 1) / A * A;
    if (top <= p0 || r.in(0, 2) == 0) return n;
    return std::max<int64_t>(p0 + A, top - A * r.in(0, 2));
}

}  /* namespace */

/* PREFILL-ENGINE-SURVEY.md's schedule at the measured configuration: 16-row passes on the 16 grid,
 * a 128-row window, boundary 20 of 40. Layer L runs in full from the pass 8 (39 - L) before the
 * last, and stores its keys alone in the 8 passes before that; layer 20 stores its keys in every
 * pass before it runs in full. A 3,086-token prompt: 5,400 of 7,720 layer-passes in full. */
TEST(consumer_plan_is_the_survey_schedule_at_16_row_passes) {
    const ConsumerShape c{ 40, 20, 128, 16, 16 };
    const int64_t n = 3086;
    const std::vector<int64_t> f = consumer_thresholds(c, n);
    REQUIRE_EQ(f.size(), (size_t)40);
    Rng r(1);
    const std::vector<Pass> g = grid(0, n, 16, 16, n, true, r);
    REQUIRE_EQ(g.size(), (size_t)193);
    const Sim s = simulate(c, n, n, 0, g, f);
    CHECK(s.ok);
    if (!s.ok) fprintf(stderr, "    %s\n", s.why.c_str());
    CHECK_EQ(s.full, (int64_t)5400);
    CHECK_EQ(s.total, (int64_t)7720);
    const int64_t last = (int64_t)g.size() - 1;
    for (int32_t L = 21; L < 40; ++L) {
        /* The first pass that runs L in full, and the first that stores its keys. */
        int64_t first_full = -1, first_key = -1;
        for (int64_t k = 0; k < (int64_t)g.size(); ++k) {
            if (first_full < 0 && s.F[(size_t)k] > L) first_full = k;
            if (first_key < 0 && s.F[(size_t)k] >= L) first_key = k;
        }
        CHECK_EQ(first_full, last - 8 * (39 - L));
        CHECK_EQ(first_key, last - 8 * (40 - L));
    }
    for (int32_t F : s.F) CHECK(F >= 20);
    CHECK_EQ(s.F.back(), 40);
    CHECK_EQ(s.F[0], 20);   /* the first passes: layers 0..19, and layer 20's keys */
    /* The rows a layer computes against the dependency cone 1 + 127 (39 - L): at most a pass and a
     * window's rounding more a layer. */
    for (int32_t L = 20; L < 40; ++L) {
        const int64_t rows = n - (int64_t)16 * (last - 8 * (39 - L));
        CHECK(rows >= 1 + 127 * (39 - L));
        CHECK(rows <= 1 + 127 * (39 - L) + (39 - L) + 16);
    }
}

/* One-row passes cut at the dependency cone itself: layer L runs on the last 1 + (W-1)(n-1-L) rows
 * and stores its keys on W - 1 more. */
TEST(one_row_passes_cut_at_the_dependency_cone) {
    for (const ConsumerShape& c : { ConsumerShape{ 40, 20, 128, 1, 1 }, ConsumerShape{ 9, 3, 4, 1, 1 } }) {
        for (int64_t n : { (int64_t)1, (int64_t)7, (int64_t)100, (int64_t)2600, (int64_t)3000 }) {
            const std::vector<int64_t> f = consumer_thresholds(c, n);
            for (int32_t L = c.boundary; L < c.n_layer; ++L) {
                const int64_t cone = n - 1 - (int64_t)(c.window - 1) * (c.n_layer - 1 - L);
                CHECK_EQ(f[(size_t)L], cone < 0 ? (int64_t)-1 : cone);
            }
            Rng r(2);
            const Sim s = simulate(c, n, n, 0, grid(0, n, 1, 1, n, true, r), f);
            CHECK(s.ok);
            if (!s.ok) fprintf(stderr, "    n %lld: %s\n", (long long)n, s.why.c_str());
        }
    }
}

/* The schedule's promise on every grid the rules allow, over shapes, budgets, quanta, prompt
 * lengths, prefix hits and tail checkpoints: exact, garbage-free, complete windows. */
TEST(consumer_plan_is_exact_on_every_grid_the_rules_allow) {
    struct Case { ConsumerShape c; int64_t n_max; int reps; };
    const Case cases[] = {
        { { 40, 20, 128, 16, 16 }, 3200, 24 },
        { { 40, 20, 128, 48, 16 }, 3200, 16 },
        { { 40, 20, 128, 64, 16 }, 3500, 12 },
        { { 40, 20, 128, 17, 16 }, 2900, 12 },
        { { 40, 20, 128, 5, 1 }, 2700, 10 },
        { { 12, 4, 5, 4, 2 }, 160, 300 },
        { { 12, 4, 5, 7, 3 }, 160, 300 },
        { { 8, 2, 3, 1, 1 }, 60, 300 },
        { { 8, 2, 3, 6, 2 }, 60, 300 },
        { { 6, 0, 2, 3, 3 }, 40, 300 },
        { { 5, 3, 9, 8, 4 }, 90, 300 },
        { { 3, 1, 1, 2, 1 }, 20, 200 },
    };
    Rng r(3);
    int runs = 0, cut = 0, bad = 0;
    for (const Case& k : cases) {
        REQUIRE(k.c.valid());
        for (int rep = 0; rep < k.reps; ++rep) {
            const int64_t n = r.in(1, k.n_max);
            /* A prefix hit resumes on a multiple of A short of the end, now and then. */
            const int64_t p0 = r.in(0, 3) == 0 && n > k.c.align ? k.c.align * r.in(0, (n - 1) / k.c.align) : 0;
            const int64_t target = pick_target(p0, n, k.c.align, r);
            const std::vector<int64_t> f = consumer_thresholds(k.c, target);
            for (bool widest : { true, false }) {
                const std::vector<Pass> g = grid(p0, n, k.c.max_rows, k.c.align, target, widest, r);
                const Sim s = simulate(k.c, n, target, p0, g, f);
                ++runs;
                cut += s.full < s.total;
                if (!s.ok) {
                    if (++bad <= 5)
                        fprintf(stderr, "    n %d B %d W %d M %lld A %lld: n %lld p0 %lld target %lld: %s\n",
                                k.c.n_layer, k.c.boundary, k.c.window, (long long)k.c.max_rows,
                                (long long)k.c.align, (long long)n, (long long)p0, (long long)target,
                                s.why.c_str());
                }
            }
        }
    }
    CHECK_EQ(bad, 0);
    CHECK(runs > 4000);
    CHECK(cut > runs / 3);   /* the cut is taken, not always declined */
}

/* The simulation is not blind: a schedule derived for a shorter window than the model's (by one
 * position at one-row passes, where nothing rounds it away; by a pass at 16-row passes), for
 * narrower passes than the grid runs, for a coarser grid, or ignoring the budget, computes rows
 * that are not the base engine's on some grid. (A window one position short at 16-row passes is
 * not caught, and is no error there: the passes' 16-row rounding covers the position.) */
TEST(a_schedule_one_step_short_is_caught) {
    struct Wrong { ConsumerShape real, assumed; };
    const Wrong cases[] = {
        { { 8, 2, 3, 1, 1 }, { 8, 2, 2, 1, 1 } },               /* the window one short */
        { { 40, 20, 128, 16, 16 }, { 40, 20, 112, 16, 16 } },   /* the window a pass short */
        { { 40, 20, 128, 48, 16 }, { 40, 20, 128, 16, 16 } },   /* passes three times wider */
        { { 12, 4, 5, 7, 3 }, { 12, 4, 5, 7, 7 } },             /* starts assumed on a coarser grid */
        { { 8, 2, 3, 6, 2 }, { 8, 2, 3, 2, 2 } },               /* the budget ignored */
    };
    Rng r(4);
    for (const Wrong& w : cases) {
        int caught = 0;
        for (int rep = 0; rep < 400 && !caught; ++rep) {
            const int64_t n = r.in(1, w.real.n_layer == 40 ? 3200 : 200);
            const std::vector<int64_t> f = consumer_thresholds(w.assumed, n);
            const std::vector<Pass> g = grid(0, n, w.real.max_rows, w.real.align, n, rep % 2 == 0, r);
            caught += !simulate(w.real, n, n, 0, g, f).ok;
        }
        CHECK(caught > 0);
    }
}

/* A checkpoint below the cut's target holds an incomplete window: the cut must target it (the
 * scheduler then cuts for min(end, tail checkpoint)) or not write it. */
TEST(a_checkpoint_below_the_target_is_incomplete) {
    const ConsumerShape c{ 40, 20, 128, 16, 16 };
    Rng r(5);
    const int64_t n = 3086, tail = 3056;   /* the window-snapshot tail: a block short of the end */
    const std::vector<Pass> g = grid(0, n, 16, 16, tail, true, r);
    /* Cut for the end: the window at 3056 misses layer 39's keys below 2944. */
    const Sim at_end = simulate(c, n, tail, 0, g, consumer_thresholds(c, n));
    CHECK(!at_end.ok);
    CHECK(at_end.why == "the target's window is incomplete");
    /* Cut for the checkpoint: complete, and still exact at the end. */
    const Sim at_tail = simulate(c, n, tail, 0, g, consumer_thresholds(c, tail));
    CHECK(at_tail.ok);
    if (!at_tail.ok) fprintf(stderr, "    %s\n", at_tail.why.c_str());
    CHECK_EQ(at_tail.full, at_end.full + 40);   /* two passes earlier, each of layers 20..39 */
    CHECK(consumer_checkpoint_complete(tail, tail));
    CHECK(!consumer_checkpoint_complete(tail - 16, tail));
}

/* Edges: no cut for a shape outside the contract or an empty target; the last pass and anything
 * past the target run every layer; F never drops below the boundary and grows with the pass. */
TEST(consumer_plan_edges) {
    CHECK(consumer_thresholds(ConsumerShape{ 40, 39, 128, 16, 16 }, 100).empty());   /* B the last layer */
    CHECK(consumer_thresholds(ConsumerShape{ 40, 20, 0, 16, 16 }, 100).empty());
    CHECK(consumer_thresholds(ConsumerShape{ 40, 20, 128, 8, 16 }, 100).empty());    /* M < A */
    CHECK(consumer_thresholds(ConsumerShape{ 40, 20, 128, 16, 16 }, 0).empty());
    const ConsumerShape c{ 40, 20, 128, 16, 16 };
    CHECK_EQ(consumer_full_layers(c, std::vector<int64_t>{}, 5), 40);
    CHECK_EQ(consumer_full_layers(c, (const int64_t*)nullptr, 5), 40);
    const std::vector<int64_t> f = consumer_thresholds(c, 8000);
    int32_t prev = 20;
    for (int64_t last = 0; last < 9000; last += 7) {
        const int32_t F = consumer_full_layers(c, f, last);
        CHECK(F >= prev);
        CHECK(F >= 20 && F <= 40);
        prev = F;
    }
    CHECK_EQ(consumer_full_layers(c, f, 7999), 40);
    CHECK_EQ(consumer_full_layers(c, f, 12000), 40);
    /* A short prompt: only the last layers can be cut, and a one-pass prompt is not. */
    const std::vector<int64_t> g = consumer_thresholds(c, 16);
    CHECK_EQ(consumer_full_layers(c, g, 15), 40);
    const std::vector<int64_t> h = consumer_thresholds(c, 64);
    CHECK_EQ(consumer_full_layers(c, h, 15), 39);   /* layer 39 stores its keys alone */
    CHECK_EQ(consumer_full_layers(c, h, 63), 40);
}

RAD_TEST_MAIN()
