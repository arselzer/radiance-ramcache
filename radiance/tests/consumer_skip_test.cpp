/* consumer_skip_test.cpp -- the consumer-suffix schedule (core/sched/consumer_skip.h) against a
 * brute-force dependency simulation (consumer_sim.h), row by row and layer by layer, over pass
 * grids the engine's rules allow (passes of at most M rows starting on a multiple of A, every one
 * but the prompt's last ending on one, a split at the tail checkpoint, a resume from a prefix hit):
 *   - every row the schedule computes, and every key it stores, is the base engine's (the
 *     simulation tracks "exact" through every read: a layer's input row, its window of W keys, the
 *     boundary's keys);
 *   - no computed row reads a key this request did not write;
 *   - the prompt's last row is exact, and so is every layer's window the next step reads (the last
 *     W positions) and the one a checkpoint at the target holds;
 *   - with checkpoints below the target made whole ("complete checkpoints"), every snapshot holds
 *     the base engine's window and state bytes, for any set of them, on every grid, and at the
 *     checkpoint intervals 16, 256 and 2048 the scheduler writes in both resume modes; only the
 *     first checkpoint ahead of a pass counts;
 *   - at M = A = 16, W = 128 it is PREFILL-ENGINE-SURVEY.md's schedule (5,400 of 7,720 layer-passes
 *     of a 3,086-token prompt), and at one-row passes the minimal dependency cone;
 *   - and the simulation is not blind: a schedule one window position short, or one that assumes
 *     narrower passes than the grid has, is caught, and so is a checkpoint below the target, one
 *     given only its keys (no cone), one whose cone is a pass short, or one built for narrower
 *     passes.
 * Pure host arithmetic. */
#include "rad_test.h"

#include "sched/consumer_skip.h"
#include "consumer_sim.h"

#include <algorithm>
#include <cstdlib>
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

using Pass = csim::Pass;

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

using csim::Sim;

/* Runs the passes of one prompt with the schedule's F for each pass (every layer when `f` is
 * empty) and the checkpoints of `whole` made whole, and checks what the schedule promises
 * (csim::simulate_passes): `target` is the cut's target, the prompt's end `n` or a checkpoint
 * below it, and its snapshot, with every one in `whole`, must be the base engine's. */
Sim simulate(const ConsumerShape& c, int64_t n, int64_t target, int64_t p0,
             const std::vector<Pass>& g, const std::vector<int64_t>& f,
             const std::vector<int64_t>& whole = {}) {
    std::vector<int64_t> snaps = whole;
    snaps.push_back(target);
    return csim::simulate_passes(c, n, p0, g, csim::schedule_F(c, g, f, whole), snaps);
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
    CHECK(at_end.why == "a snapshot is not the base engine's");
    CHECK(at_end.bad_snaps == std::vector<int64_t>{ tail });
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

/* ================================================================== complete checkpoints */

/* Any set of pass ends below the target made whole: every snapshot of the set, and the target's,
 * holds the base engine's window and state bytes on every grid the rules allow, with every
 * computed row exact and no unwritten key read; and each pass's F is the brute-force maximum over
 * every checkpoint of the set ahead of it, which is the first one's (only the first counts). */
TEST(complete_checkpoints_are_the_base_engines_snapshots_on_every_grid) {
    struct Case { ConsumerShape c; int64_t n_max; int reps; };
    const Case cases[] = {
        { { 40, 20, 128, 16, 16 }, 3200, 14 },
        { { 40, 20, 128, 48, 16 }, 3200, 8 },
        { { 40, 20, 128, 17, 16 }, 2900, 8 },
        { { 40, 20, 128, 5, 1 }, 1600, 4 },
        { { 12, 4, 5, 4, 2 }, 160, 300 },
        { { 12, 4, 5, 7, 3 }, 160, 300 },
        { { 8, 2, 3, 1, 1 }, 60, 300 },
        { { 8, 2, 3, 6, 2 }, 60, 300 },
        { { 6, 0, 2, 3, 3 }, 40, 300 },
        { { 5, 3, 9, 8, 4 }, 90, 300 },
        { { 3, 1, 1, 2, 1 }, 20, 200 },
    };
    Rng r(6);
    int runs = 0, with = 0, bad = 0, not_first = 0, raised = 0;
    int64_t snaps = 0;
    for (const Case& k : cases) {
        REQUIRE(k.c.valid());
        for (int rep = 0; rep < k.reps; ++rep) {
            const int64_t n = r.in(1, k.n_max);
            const int64_t p0 = r.in(0, 3) == 0 && n > k.c.align ? k.c.align * r.in(0, (n - 1) / k.c.align) : 0;
            const int64_t target = pick_target(p0, n, k.c.align, r);
            const std::vector<int64_t> f = consumer_thresholds(k.c, target);
            for (bool widest : { true, false }) {
                const std::vector<Pass> g = grid(p0, n, k.c.max_rows, k.c.align, target, widest, r);
                /* Every pass end below the target, or a random part of them. */
                const int64_t mode = r.in(0, 3);
                std::vector<int64_t> whole;
                for (const Pass& ps : g)
                    if (ps.second > p0 && ps.second < target && (mode == 0 || r.in(0, mode) == 0))
                        whole.push_back(ps.second);
                const std::vector<int32_t> F = csim::schedule_F(k.c, g, f, whole);
                for (size_t i = 0; i < g.size(); ++i) {
                    int32_t best = consumer_full_layers(k.c, f, g[i].second - 1);
                    for (int64_t P : whole)
                        if (P >= g[i].second)
                            best = std::max(best, consumer_full_layers(
                                                      k.c, consumer_ckpt_thresholds(k.c, P),
                                                      g[i].second - 1));
                    not_first += best != F[i];
                }
                const Sim s = simulate(k.c, n, target, p0, g, f, whole);
                const Sim t = simulate(k.c, n, target, p0, g, f);   /* the target alone */
                ++runs;
                with += !whole.empty();
                snaps += (int64_t)whole.size();
                raised += s.full > t.full;
                if (!s.ok && ++bad <= 5)
                    fprintf(stderr, "    n %d B %d W %d M %lld A %lld: n %lld p0 %lld target %lld, "
                                    "%zu whole: %s\n",
                            k.c.n_layer, k.c.boundary, k.c.window, (long long)k.c.max_rows,
                            (long long)k.c.align, (long long)n, (long long)p0, (long long)target,
                            whole.size(), s.why.c_str());
            }
        }
    }
    CHECK_EQ(bad, 0);
    CHECK_EQ(not_first, 0);
    CHECK(runs > 4000);
    CHECK(with > runs / 2);
    CHECK(snaps > 20000);
    CHECK(raised > runs / 4);   /* the cones cost layers, and the simulation sees them run */
}

/* DeepSeek-V4.1's own shape and grid: the snapshots the scheduler writes for an idle prefill
 * (csim::ds41_idle_snapshots; sched_test holds the real scheduler to it) at checkpoint intervals
 * 16, 256 and 2048 in both resume modes, cold and resumed from one of them. The cut's target is
 * the deepest and every other is made whole: each one's window and state bytes are the base
 * engine's, every computed row is, and the work is what consumer_skip.h says it costs. */
TEST(complete_checkpoints_at_the_scheduler_intervals_in_both_resume_modes) {
    const ConsumerShape c{ 40, 20, 128, 16, 16 };
    const int64_t lengths[] = { 19, 40, 129, 255, 256, 257, 300, 1100, 1221, 2047, 2048, 2050,
                                2064, 2065, 2655, 3086, 3163, 4096, 5000 };
    int runs = 0, bad = 0;
    int64_t whole_total = 0;
    for (const bool exact : { false, true }) {
        for (const int64_t I : { 16, 256, 2048 }) {
            for (const int64_t n : lengths) {
                const std::vector<int64_t> cold = csim::ds41_idle_snapshots(0, n, I, exact);
                /* Cold, and resumed at each of the first three snapshots below the end. */
                std::vector<int64_t> starts{ 0 };
                for (size_t i = 0; i < cold.size() && starts.size() < 4; ++i)
                    if (cold[i] < n) starts.push_back(cold[i]);
                for (const int64_t p0 : starts) {
                    const std::vector<int64_t> snap = csim::ds41_idle_snapshots(p0, n, I, exact);
                    int64_t T = n;
                    if (!snap.empty() && snap.back() < n) T = snap.back();
                    std::vector<int64_t> whole;
                    for (int64_t P : snap) if (P < T) whole.push_back(P);
                    const std::vector<Pass> g = csim::ds41_idle_passes(p0, n);
                    const std::vector<int64_t> f = consumer_thresholds(c, T);
                    const std::vector<int32_t> F = csim::schedule_F(c, g, f, whole);
                    const Sim s = csim::simulate_passes(c, n, p0, g, F, snap);
                    ++runs;
                    whole_total += (int64_t)whole.size();
                    if (!s.ok && ++bad <= 5)
                        fprintf(stderr, "    %s I %lld n %lld p0 %lld T %lld: %s\n",
                                exact ? "exact" : "nearest", (long long)I, (long long)n,
                                (long long)p0, (long long)T, s.why.c_str());
                    /* The pass ending at a whole snapshot stores every layer's keys. */
                    for (size_t k = 0; k < g.size(); ++k)
                        if (std::binary_search(whole.begin(), whole.end(), g[k].second) &&
                            F[k] < c.n_layer - 1)
                            ++bad;
                }
            }
        }
    }
    CHECK_EQ(bad, 0);
    CHECK(runs > 2 * 3 * 19 * 2);   /* every case, cold and resumed */
    CHECK(whole_total > 3000);

    /* What it costs, in layer-passes run in full, on a 3,086-token prompt (exact; the target is
     * the tail snapshot at 3072 at every interval): the target alone runs 5,420 of 7,720; with
     * every snapshot whole, at an interval of 16 every layer but the last on every pass outside
     * the target's reach (7,529), at 256 every layer but the last two or so (7,434), at 2048 the
     * one snapshot at 2048 adds its cone past the target's (932). */
    {
        const int64_t n = 3086;
        auto cost = [&](int64_t I, bool all) {
            const std::vector<int64_t> snap = csim::ds41_idle_snapshots(0, n, I, true);
            const int64_t T = snap.empty() ? n : snap.back();
            std::vector<int64_t> whole;
            if (all) for (int64_t P : snap) if (P < T) whole.push_back(P);
            const std::vector<Pass> g = csim::ds41_idle_passes(0, n);
            return csim::full_layer_passes(csim::schedule_F(c, g, consumer_thresholds(c, T), whole));
        };
        CHECK_EQ(cost(16, false), (int64_t)5420);
        CHECK_EQ(cost(256, false), (int64_t)5420);
        CHECK_EQ(cost(2048, false), (int64_t)5420);
        const int64_t a16 = cost(16, true), a256 = cost(256, true), a2048 = cost(2048, true);
        CHECK_EQ(a16, (int64_t)7529);     /* 97.5% of 7,720 */
        CHECK_EQ(a256, (int64_t)7434);    /* 96.3% */
        CHECK_EQ(a2048, (int64_t)6352);   /* 82.3% */
    }
}

/* A checkpoint needs its whole cone: the target's schedule alone leaves its snapshot short of
 * layer 39's keys; storing every layer's keys on the passes behind it alone (layers below in full
 * there, nothing before: the key ops on the rows [P - W, P) without the cone) reads keys never
 * written and so computes rows that are not the base engine's; a cone one pass short (layer n-2
 * in full from P - W + A), or one derived for narrower passes than the grid runs, leaves a
 * snapshot or a row wrong on some grid. */
TEST(a_checkpoint_needs_its_whole_cone) {
    const ConsumerShape c{ 40, 20, 128, 16, 16 };
    const int64_t n = 4096, T = 4080, P = 2048;
    const std::vector<Pass> g = csim::ds41_idle_passes(0, n);
    const std::vector<int64_t> f = consumer_thresholds(c, T);
    /* The target alone. */
    {
        const Sim s = csim::simulate_passes(c, n, 0, g, csim::schedule_F(c, g, f, {}), { P, T });
        CHECK(!s.ok);
        CHECK(s.bad_snaps == std::vector<int64_t>{ P });
    }
    /* The key ops alone on the rows behind P. */
    {
        std::vector<int32_t> F = csim::schedule_F(c, g, f, {});
        for (size_t k = 0; k < g.size(); ++k)
            if (g[k].second > P - c.window && g[k].second <= P) F[k] = std::max(F[k], c.n_layer - 1);
        const Sim s = csim::simulate_passes(c, n, 0, g, F, { P, T });
        CHECK(!s.ok);
        CHECK(s.stale > 0);
        CHECK(s.garbage > 0);
    }
    /* The whole cone: right. */
    {
        const Sim s = csim::simulate_passes(c, n, 0, g, csim::schedule_F(c, g, f, { P }), { P, T });
        CHECK(s.ok);
        if (!s.ok) fprintf(stderr, "    %s\n", s.why.c_str());
    }
    /* A cone one pass short at the top. */
    {
        std::vector<int64_t> gt = consumer_ckpt_thresholds(c, P);
        for (int32_t L = c.boundary; L < c.n_layer - 1; ++L)
            if (gt[(size_t)L] >= 0) gt[(size_t)L] += c.align;
        std::vector<int32_t> F;
        for (const Pass& ps : g)
            F.push_back(ps.second <= P ? consumer_pass_layers(c, f, gt, ps.second - 1)
                                       : consumer_full_layers(c, f, ps.second - 1));
        const Sim s = csim::simulate_passes(c, n, 0, g, F, { P, T });
        CHECK(!s.ok);
    }
    /* A cone for 16-row passes on a grid of 48-row passes. */
    {
        const ConsumerShape wide{ 40, 20, 128, 48, 16 };
        Rng r(7);
        int caught = 0;
        for (int rep = 0; rep < 40 && !caught; ++rep) {
            const std::vector<Pass> gw = grid(0, n, 48, 16, T, rep % 2 == 0, r);
            std::vector<int64_t> whole;
            for (const Pass& ps : gw)
                if (ps.second < T && ps.second % 256 == 0) whole.push_back(ps.second);
            const std::vector<int64_t> fw = consumer_thresholds(wide, T);
            std::vector<int32_t> F;
            for (const Pass& ps : gw) {
                const auto it = std::lower_bound(whole.begin(), whole.end(), ps.second);
                F.push_back(it == whole.end()
                                ? consumer_full_layers(wide, fw, ps.second - 1)
                                : consumer_pass_layers(wide, fw, consumer_ckpt_thresholds(c, *it),
                                                       ps.second - 1));
            }
            std::vector<int64_t> snaps = whole;
            snaps.push_back(T);
            caught += !csim::simulate_passes(wide, n, 0, gw, F, snaps).ok;
        }
        CHECK(caught > 0);
    }
}

/* The checkpoint thresholds' edges, their order in P, and the rule a cut step that writes a
 * snapshot is held to. */
TEST(complete_checkpoint_edges) {
    const ConsumerShape c{ 40, 20, 128, 16, 16 };
    CHECK(consumer_ckpt_thresholds(c, 0).empty());
    CHECK(consumer_ckpt_thresholds(ConsumerShape{ 40, 39, 128, 16, 16 }, 100).empty());
    /* A snapshot inside the first window: every pass before it runs layers 0..38 in full. */
    const std::vector<int64_t> g0 = consumer_ckpt_thresholds(c, 96);
    REQUIRE_EQ(g0.size(), (size_t)40);
    for (int32_t L = 0; L < 39; ++L) CHECK_EQ(g0[(size_t)L], (int64_t)-1);
    CHECK_EQ(g0[39], INT64_MAX);
    CHECK_EQ(consumer_full_layers(c, g0, 15), 39);
    /* Far from the start: layer 38 in full on the 8 passes before P, layer 37 on 16, ... */
    const int64_t P = 8192;
    const std::vector<int64_t> g = consumer_ckpt_thresholds(c, P);
    CHECK_EQ(g[38], P - 128);
    CHECK_EQ(g[37], P - 255);
    CHECK_EQ(g[20], P - 2431);
    for (int32_t L = 21; L < 39; ++L) {
        int passes = 0;
        for (int64_t e = 16; e <= P; e += 16) passes += consumer_full_layers(c, g, e - 1) > L;
        CHECK_EQ(passes, 8 * (39 - L));
    }
    /* Non-decreasing in P, layer by layer, so G_P is non-increasing in P. */
    Rng r(8);
    for (int rep = 0; rep < 2000; ++rep) {
        const ConsumerShape k{ (int32_t)r.in(3, 40), 0, (int32_t)r.in(1, 130), r.in(1, 64), 1 };
        ConsumerShape s = k;
        s.boundary = (int32_t)r.in(0, k.n_layer - 2);
        s.align = r.in(1, s.max_rows);
        const int64_t p1 = r.in(1, 5000), p2 = p1 + r.in(0, 300);
        const std::vector<int64_t> a = consumer_ckpt_thresholds(s, p1), b = consumer_ckpt_thresholds(s, p2);
        REQUIRE_EQ(a.size(), b.size());
        for (size_t L = 0; L < a.size(); ++L) CHECK(a[L] <= b[L]);
        const int64_t last = r.in(0, 5300);
        CHECK(consumer_full_layers(s, a, last) >= consumer_full_layers(s, b, last));
    }
    /* A cut step may write the snapshot it was planned to make whole, at its end, storing every
     * layer's keys; at or past the target any; nothing else. */
    CHECK(consumer_ckpt_covered(c, 2048, 2047, 3000, 2048, 39));
    CHECK(consumer_ckpt_covered(c, 2048, 2047, 3000, 2048, 40));
    CHECK(!consumer_ckpt_covered(c, 2048, 2047, 3000, 2048, 38));
    CHECK(!consumer_ckpt_covered(c, 2048, 2047, 3000, 2304, 39));
    CHECK(!consumer_ckpt_covered(c, 2048, 2047, 3000, 0, 39));
    CHECK(!consumer_ckpt_covered(c, 2040, 2047, 3000, 2040, 39));
    CHECK(consumer_ckpt_covered(c, 3000, 2999, 3000, 0, 20));
    CHECK(consumer_pass_layers(c, consumer_thresholds(c, 3000), std::vector<int64_t>{}, 100) ==
          consumer_full_layers(c, consumer_thresholds(c, 3000), 100));
}

/* What complete checkpoints cost at DeepSeek-V4.1's shape, by prompt length, interval, resume
 * mode and policy (every snapshot, those within N tokens below the target, none), in layer-passes
 * run in full against the base engine's (every layer, every pass): the passes and snapshots of an
 * idle prefill (csim::ds41_idle_snapshots), the schedule's F. The ordering holds everywhere (none <=
 * a band <= a wider band <= all <= base), every snapshot written below the target costs nothing
 * in key ops alone (one a cut pass, as without it), and with CSCKPT_TABLE=1 the table is printed:
 * "base" the base engine's layer-passes, then each policy's share of them in full. */
TEST(complete_checkpoint_cost_by_length_and_interval) {
    const ConsumerShape c{ 40, 20, 128, 16, 16 };
    const bool print = std::getenv("CSCKPT_TABLE") != nullptr;
    const int64_t bands[] = { 0, 256, 512, 1024, 2048, -1 };
    if (print)
        fprintf(stderr, "    mode    I     n      T  snaps  base   | none   256    512    1024   2048   all\n");
    for (const bool exact : { true, false }) {
        for (const int64_t I : { 16, 256, 2048 }) {
            for (const int64_t n : { 1221, 2048, 3086, 4096, 8192, 16384, 32768 }) {
                const std::vector<int64_t> snap = csim::ds41_idle_snapshots(0, n, I, exact);
                const int64_t T = !snap.empty() && snap.back() < n ? snap.back() : n;
                const std::vector<Pass> g = csim::ds41_idle_passes(0, n);
                const std::vector<int64_t> f = consumer_thresholds(c, T);
                const int64_t base = (int64_t)g.size() * c.n_layer;
                int64_t prev = 0, keys0 = -1;
                char row[256];
                int at = snprintf(row, sizeof row, "    %-7s %-5lld %-5lld %-5lld %-5zu %-6lld |",
                                  exact ? "exact" : "nearest", (long long)I, (long long)n,
                                  (long long)T, snap.size(), (long long)base);
                for (const int64_t band : bands) {
                    std::vector<int64_t> whole;
                    for (int64_t P : snap)
                        if (P < T && (band < 0 || (band > 0 && P >= T - band))) whole.push_back(P);
                    const std::vector<int32_t> F = csim::schedule_F(c, g, f, whole);
                    const int64_t full = csim::full_layer_passes(F);
                    int64_t keys = 0;
                    for (int32_t x : F) keys += x < c.n_layer;
                    if (keys0 < 0) keys0 = keys;
                    CHECK(full >= prev);
                    CHECK(full <= base);
                    CHECK_EQ(keys, keys0);   /* one layer of key ops alone a cut pass, as before */
                    prev = full;
                    at += snprintf(row + at, sizeof row - (size_t)at, " %5.1f%%",
                                   100.0 * (double)full / (double)base);
                }
                if (print) fprintf(stderr, "%s\n", row);
            }
        }
    }
}


/* ================================================================== the per-row cut */
/* Large-chunk prefill's row mode (consumer_row_starts): tall passes on the 16-row grid, each layer on
 * a suffix of the pass's rows. The same row-level simulation, with each layer's key ops on rows
 * [row0(L - 1), T) and its full rows [row0(L), T). */
namespace {

/* 0 none, 1 the key ops alone, 2 in full: by layer and position. */
using Set = std::vector<std::vector<uint8_t>>;

/* With `whole` (pass ends below the target, ascending), each pass runs under consumer_row_thresholds'
 * h for the first of them at or after its end (claude/ds41-csckpt's cones under the per-row cut), and
 * each of them must hold the base engine's window on every layer and be consumer_ckpt_covered_rows
 * at the pass that ends there. */
Sim simulate_rows(const ConsumerShape& c16, int64_t n, int64_t target, int64_t p0, const std::vector<Pass>& g,
                  const std::vector<int64_t>& f, Set* set, const std::vector<int64_t>& whole = {}) {
    Sim out;
    const int32_t nl = c16.n_layer, B = c16.boundary;
    const int64_t W = c16.window;
    std::vector<std::vector<uint8_t>> key((size_t)nl, std::vector<uint8_t>((size_t)n, 0));
    std::vector<std::vector<int32_t>> pre((size_t)nl, std::vector<int32_t>((size_t)n + 1, 0));
    std::vector<int64_t> upto((size_t)nl, 0);
    set->assign((size_t)nl, std::vector<uint8_t>((size_t)n, 0));
    for (int32_t L = 0; L < nl; ++L)
        for (int64_t p = 0; p < p0; ++p) key[(size_t)L][(size_t)p] = 1;
    auto settle = [&](int32_t L, int64_t to) {
        auto& k = key[(size_t)L];
        auto& q = pre[(size_t)L];
        for (int64_t p = upto[(size_t)L]; p < to; ++p) q[(size_t)p + 1] = q[(size_t)p] + (k[(size_t)p] == 1);
        upto[(size_t)L] = std::max(upto[(size_t)L], to);
    };
    bool bkeys = true;
    auto fail = [&](const std::string& w) {
        if (out.ok) out.why = w;
        out.ok = false;
    };
    std::vector<int32_t> row0((size_t)nl);
    std::vector<uint8_t> in, nx;
    std::vector<int64_t> gk((size_t)nl), hk((size_t)nl);
    for (const Pass& ps : g) {
        const int64_t s = ps.first, e = ps.second, T = e - s;
        const int64_t* thr = f.empty() ? nullptr : f.data();
        const auto ahead = std::lower_bound(whole.begin(), whole.end(), e);
        if (thr && ahead != whole.end() && *ahead < target) {
            if (!consumer_ckpt_thresholds(c16, *ahead, gk.data()) ||
                !consumer_row_thresholds(c16, f.data(), gk.data(), hk.data())) { fail("no cone"); return out; }
            thr = hk.data();
        }
        if (!thr) std::fill(row0.begin(), row0.end(), 0);
        else if (!consumer_row_starts(c16, thr, s, e, row0.data())) { fail("no row starts"); return out; }
        if (thr && ahead != whole.end() && *ahead == e &&
            !consumer_ckpt_covered_rows(c16, e, s, e, target, *ahead, row0.data()))
            { fail("a pass ending at a whole checkpoint is not covered"); return out; }
        for (int32_t L = B + 1; L < nl; ++L)
            if (row0[(size_t)L] < row0[(size_t)L - 1]) { fail("a layer's rows do not nest in the one below"); return out; }
        for (int32_t L = 0; L < B; ++L)
            if (row0[(size_t)L] != 0) { fail("a layer below the boundary is cut"); return out; }
        if (thr && consumer_full_layers(c16, thr, e - 1) !=
                              (int32_t)(std::find_if(row0.begin() + B, row0.end(),
                                                     [&](int32_t r) { return r >= T; }) - row0.begin()))
            { fail("F is not the first layer with no row"); return out; }
        out.total += nl;
        in.assign((size_t)T, 1);
        for (int32_t L = 0; L < nl; ++L) {
            const int64_t kf = L <= B ? 0 : row0[(size_t)L - 1], ff = row0[(size_t)L];
            if (kf >= T) break;
            for (int64_t i = kf; i < T; ++i)
                if (in[(size_t)i] == 0) { fail("a layer runs on a row its previous layer did not compute"); return out; }
            settle(L, s);
            for (int64_t i = kf; i < T; ++i) {
                uint8_t& k = key[(size_t)L][(size_t)(s + i)];
                if (k != 0) { fail("a key written twice"); return out; }
                k = in[(size_t)i];
                if (k != 1) ++out.garbage;
                if (L == B && k != 1) bkeys = false;
                (*set)[(size_t)L][(size_t)(s + i)] = i >= ff ? 2 : 1;
            }
            settle(L, e);
            out.full += ff < T;
            out.key += ff >= T;
            nx.assign((size_t)T, 0);
            for (int64_t i = ff; i < T; ++i) {
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
            if (ff >= T) break;   /* the key layer: nothing past it */
        }
        if (e == n) {
            for (int32_t L = 0; L < nl; ++L)
                if (row0[(size_t)L] > T - 1) { fail("the prompt's last row is outside a layer's rows"); return out; }
            if (in[(size_t)(T - 1)] != 1) fail("the prompt's last row is not exact");
        }
    }
    if (out.garbage) fail("a computed row or stored key is not the base engine's");
    if (out.stale) fail("a computed row read a key this request never wrote");
    for (int32_t L = 0; L < nl; ++L) {
        for (int64_t p = std::max<int64_t>(0, n - W); p < n; ++p)
            if (key[(size_t)L][(size_t)p] != 1) { fail("the decode window is incomplete"); break; }
        for (int64_t p = std::max<int64_t>(0, target - W); p < target; ++p)
            if (key[(size_t)L][(size_t)p] != 1) { fail("the target's window is incomplete"); break; }
    }
    if (!bkeys) fail("a boundary key is not exact");
    for (const int64_t P : whole) {
        for (int32_t L = 0; L < nl; ++L)
            for (int64_t p = std::max<int64_t>(0, P - W); p < P; ++p)
                if (key[(size_t)L][(size_t)p] != 1) {
                    out.bad_snaps.push_back(P);
                    fail("a whole checkpoint's window is incomplete");
                    L = nl;
                    break;
                }
    }
    return out;
}

/* The (layer, position) set the 16-row passes compute under the 16-row schedule: the reference. */
Set set_at_16(const ConsumerShape& c16, int64_t n, int64_t p0, const std::vector<Pass>& g16, const std::vector<int64_t>& f) {
    Set set((size_t)c16.n_layer, std::vector<uint8_t>((size_t)n, 0));
    for (const Pass& ps : g16) {
        const int32_t F = consumer_full_layers(c16, f, ps.second - 1);
        for (int32_t L = 0; L < c16.n_layer && L <= F; ++L)
            for (int64_t q = ps.first; q < ps.second; ++q) set[(size_t)L][(size_t)q] = L < F ? 2 : 1;
    }
    (void)p0;
    return set;
}

}  /* namespace */

/* Row mode against the row-level simulation, over prompts of 1 to 16,384 tokens, chunks of 64 to
 * 2,048, prefix hits resuming on the 16-row grid, tail checkpoints and short last chunks: exact,
 * nothing unwritten read, the windows complete, each layer's rows nested in the one below's, the
 * prompt's last row inside every layer's rows, and the (layer, row) set the 16-row passes compute. */
TEST(row_cut_is_the_16_row_schedule_in_every_tall_pass) {
    const ConsumerShape c16{ 40, 20, 128, 16, 16 };
    Rng r(11);
    int runs = 0, bad = 0, differ = 0, cut = 0;
    for (const int64_t C : { 64, 256, 512, 1024, 2048 }) {
        for (int rep = 0; rep < 24; ++rep) {
            const int64_t n = rep < 3 ? (int64_t)(rep == 0 ? 3086 : rep == 1 ? 6817 : 16384) : r.in(1, 16384);
            const int64_t p0 = r.in(0, 3) == 0 && n > 16 ? 16 * r.in(0, (n - 1) / 16) : 0;
            const int64_t target = pick_target(p0, n, 16, r);
            const std::vector<int64_t> f = consumer_thresholds(c16, target);
            const std::vector<Pass> g16 = grid(p0, n, 16, 16, target, true, r);
            const Set ref = set_at_16(c16, n, p0, g16, f);
            for (bool widest : { true, false }) {
                const std::vector<Pass> g = grid(p0, n, C, 16, target, widest, r);
                Set got;
                const Sim s = simulate_rows(c16, n, target, p0, g, f, &got);
                ++runs;
                cut += s.full < s.total;
                if (!s.ok && ++bad <= 5)
                    fprintf(stderr, "    C %lld n %lld p0 %lld target %lld: %s\n", (long long)C, (long long)n,
                            (long long)p0, (long long)target, s.why.c_str());
                differ += got != ref;
            }
        }
    }
    CHECK_EQ(bad, 0);
    CHECK_EQ(differ, 0);
    CHECK(runs >= 240);
    CHECK(cut > runs / 2);
}

/* At C = 1,024 a 3,086-token prompt: chunk 0 runs layers 20..22 on 384, 256 and 128 of its rows and
 * stores layer 23's keys, chunk 3 (14 rows) runs every layer on every row, and the layer-rows equal the
 * 16-row passes' 30% cut (5,400 of 7,720 layer-passes, 16 rows each but the last). Without rows (a
 * whole-pass cut at M = 1,024) the same prompt keeps almost nothing cut. */
TEST(row_cut_at_1024_keeps_the_16_row_cut) {
    const ConsumerShape c16{ 40, 20, 128, 16, 16 };
    const int64_t n = 3086;
    const std::vector<int64_t> f = consumer_thresholds(c16, n);
    std::vector<int32_t> row0(40);
    REQUIRE(consumer_row_starts(c16, f.data(), 0, 1024, row0.data()));
    for (int32_t L = 0; L < 20; ++L) CHECK_EQ(row0[(size_t)L], 0);
    CHECK_EQ(1024 - row0[20], 384);
    CHECK_EQ(1024 - row0[21], 256);
    CHECK_EQ(1024 - row0[22], 128);
    CHECK_EQ(row0[23], 1024);
    CHECK_EQ(consumer_full_layers(c16, f.data(), 1023), 23);
    REQUIRE(consumer_row_starts(c16, f.data(), 3072, 3086, row0.data()));
    for (int32_t L = 0; L < 40; ++L) CHECK_EQ(row0[(size_t)L], 0);
    /* Layer-rows over the four chunks against the 16-row passes'. */
    Rng r(12);
    Set got;
    const std::vector<Pass> g = grid(0, n, 1024, 16, n, true, r);
    REQUIRE_EQ(g.size(), (size_t)4);
    const Sim s = simulate_rows(c16, n, n, 0, g, f, &got);
    CHECK(s.ok);
    if (!s.ok) fprintf(stderr, "    %s\n", s.why.c_str());
    int64_t rows_full = 0;
    for (const auto& L : got)
        for (uint8_t v : L) rows_full += v == 2;
    const Set ref = set_at_16(c16, n, 0, grid(0, n, 16, 16, n, true, r), f);
    int64_t ref_full = 0;
    for (const auto& L : ref)
        for (uint8_t v : L) ref_full += v == 2;
    CHECK_EQ(rows_full, ref_full);
    CHECK(got == ref);
    /* The whole-pass cut at M = 1,024 for comparison: far more layer-rows in full. */
    const ConsumerShape cM{ 40, 20, 128, 1024, 16 };
    const std::vector<int64_t> fM = consumer_thresholds(cM, n);
    int64_t whole = 0;
    for (const Pass& ps : g) whole += (int64_t)consumer_full_layers(cM, fM, ps.second - 1) * (ps.second - ps.first);
    CHECK(whole > rows_full);
    CHECK(whole >= (int64_t)40 * n * 96 / 100);   /* keeps at most ~4% cut */
}

/* Complete checkpoints (claude/ds41-csckpt) under the per-row cut: tall passes on the 16-row grid at
 * C of 64 to 2,048, every pass end below the target or a random part of them made whole with the
 * 16-row cone applied to each group (consumer_row_thresholds): every computed row exact, no unwritten
 * key read, the target's and the decode window complete, every whole checkpoint's window complete on
 * every layer and covered at its pass (consumer_ckpt_covered_rows). Negative control: the target's
 * rows alone leave some checkpoint below it incomplete. */
TEST(row_cut_with_complete_checkpoints_keeps_every_snapshot_whole) {
    const ConsumerShape c16{ 40, 20, 128, 16, 16 };
    Rng r(13);
    int runs = 0, bad = 0, with = 0, short_alone = 0;
    for (const int64_t C : { 64, 256, 512, 1024, 2048 }) {
        for (int rep = 0; rep < 16; ++rep) {
            const int64_t n = rep < 2 ? (int64_t)(rep == 0 ? 3086 : 6817) : r.in(1, 12000);
            const int64_t p0 = r.in(0, 3) == 0 && n > 16 ? 16 * r.in(0, (n - 1) / 16) : 0;
            const int64_t target = pick_target(p0, n, 16, r);
            const std::vector<int64_t> f = consumer_thresholds(c16, target);
            for (bool widest : { true, false }) {
                const std::vector<Pass> g = grid(p0, n, C, 16, target, widest, r);
                const int64_t mode = r.in(0, 3);
                std::vector<int64_t> whole;
                for (const Pass& ps : g)
                    if (ps.second > p0 && ps.second < target && (mode == 0 || r.in(0, mode) == 0))
                        whole.push_back(ps.second);
                Set got;
                const Sim s = simulate_rows(c16, n, target, p0, g, f, &got, whole);
                ++runs;
                with += !whole.empty();
                if (!s.ok && ++bad <= 5)
                    fprintf(stderr, "    C %lld n %lld p0 %lld target %lld, %zu whole: %s\n", (long long)C,
                            (long long)n, (long long)p0, (long long)target, whole.size(), s.why.c_str());
                /* The target's rows alone: the snapshots below it are checked, not made whole. */
                if (!whole.empty()) {
                    Set alone;
                    std::vector<int64_t> none;
                    const Sim t = simulate_rows(c16, n, target, p0, g, f, &alone, none);
                    bool incomplete = false;
                    for (const int64_t P : whole)
                        for (int32_t L = 0; L < c16.n_layer && !incomplete; ++L)
                            for (int64_t q = std::max<int64_t>(0, P - c16.window); q < P && !incomplete; ++q)
                                incomplete = alone[(size_t)L][(size_t)q] == 0 ||
                                             (L > 0 && alone[(size_t)L - 1][(size_t)q] != 2);
                    short_alone += t.ok && incomplete;
                }
            }
        }
    }
    CHECK_EQ(bad, 0);
    CHECK(runs >= 160);
    CHECK(with > runs / 3);
    CHECK(short_alone > 0);
}

RAD_TEST_MAIN()
