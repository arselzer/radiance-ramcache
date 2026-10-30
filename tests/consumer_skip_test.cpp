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

RAD_TEST_MAIN()
