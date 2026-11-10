/* aux_online_test.cpp -- the aux expert worker's online unit set, pure parts (tools/ds41-host-experts/
 * ds41_aux_online.h; work/aux-dyn/SPEC.md part B, §7.2). No card, no core.
 *  - the scores: the prior (6 x share x n0), a merge's decay and its hits, against hand values;
 *  - the credit (RATE / 100 x tokens, at most MAX, floor) and the copy budget (bytes per second, a second deep);
 *  - the planner: ins only from units hit at least once, pairs by gain across ranks under the margin, at most n,
 *    ties by ds41aux::before (ins) and its reverse (outs), the lower rank on equal gains, pinned units (copying,
 *    leaving, cooling down) and ranks not planned left out;
 *  - the replay fixture (tests/data/aux_online/fixture.bin, written by work/aux-dyn/replay/q5_engine_rule.py
 *    --fixture from 200 held-out decode tokens of the Oct 29 traces on plan L1024's four rank lists): the C++ rule
 *    -- scores, credit, planner, the engine's lag (flips two steps after the hits that decided them) -- makes the
 *    replay's swaps at every step, at 1 and at 8 streams, and the same absorption. */
#include "rad_test.h"

#include "../tools/ds41-host-experts/ds41_aux_online.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace on = ds41aux::online;

namespace {

std::vector<ds41aux::Cand> cands(int rank, const std::vector<float>& shares) {
    std::vector<ds41aux::Cand> c;
    for (size_t i = 0; i < shares.size(); ++i) {
        ds41aux::Cand x;
        x.rank = rank;
        x.unit = (int32_t)i;
        x.share = shares[i];
        x.expert = (int32_t)i;
        x.layer = 3;
        c.push_back(x);
    }
    return c;
}

}  // namespace

TEST(scores_prior_decay_and_hits) {
    on::RankScores s;
    s.init(cands(0, { 0.01f, 0.0f }), 256);
    CHECK_EQ(s.prior[0], 6.0 * (double)0.01f * 256.0);
    CHECK_EQ(s.prior[1], 0.0);
    const uint32_t d[2] = { 8, 0 };
    on::merge(&s, d, on::decay_factor(8, 512));
    CHECK_EQ(s.hits[0], 8.0);
    CHECK_EQ(s.hits[1], 0.0);
    on::merge(&s, nullptr, on::decay_factor(512, 512));   /* a half-life */
    CHECK_EQ(s.hits[0], 4.0);
    const uint32_t d2[2] = { 0, 3 };
    on::merge(&s, d2, 1.0);
    CHECK_EQ(s.hits[0], 4.0);
    CHECK_EQ(s.hits[1], 3.0);
    CHECK_EQ(s.score(1), 3.0);
    CHECK_EQ(on::decay_factor(0, 512), 1.0);
}

TEST(credit_and_budget) {
    on::Credit c;
    CHECK_EQ(c.grant(1, 200, 16), 2);   /* 2 a token */
    c.spend(2);
    CHECK_EQ(c.grant(8, 200, 16), 16);  /* at most MAX */
    c.spend(10);
    CHECK_EQ(c.grant(1, 50, 16), 6);    /* 6.5: the floor */
    on::Bucket b;
    b.init(1000);   /* 1000 MB/s, a second deep */
    CHECK(b.take(0, 600'000'000));
    CHECK(!b.take(0, 600'000'000));
    CHECK(b.take(200'000'000, 600'000'000));   /* 200 ms refilled 200 MB */
    CHECK(!b.take(200'000'000, 1));
    on::Bucket big;
    big.init(100);
    CHECK(big.take(0, 500'000'000));   /* larger than the depth: once full */
}

TEST(planner_rules) {
    /* rank 0: units 0-3 resident (scores by prior), 4-7 not; rank 1 likewise. */
    std::vector<ds41aux::Cand> c0 = cands(0, { 0.08f, 0.07f, 0.06f, 0.05f, 0.04f, 0.03f, 0.02f, 0.01f });
    std::vector<ds41aux::Cand> c1 = cands(1, { 0.08f, 0.07f, 0.06f, 0.05f, 0.04f, 0.03f, 0.02f, 0.01f });
    on::RankScores s0, s1;
    s0.init(c0, 1);
    s1.init(c1, 1);
    std::vector<uint8_t> st0 = { 1, 1, 1, 1, 0, 0, 0, 0 }, st1 = st0;
    std::vector<on::RankIn> in(2);
    in[0] = { &s0, &c0, st0.data(), true };
    in[1] = { &s1, &c1, st1.data(), true };
    /* no hits: nothing comes in, whatever its prior */
    CHECK(on::plan(in, 4, 100).empty());
    /* rank 0's unit 6 hit 1.0 (score 1.0 + 0.12), rank 1's unit 7 hit 0.5 */
    s0.hits[6] = 1.0;
    s1.hits[7] = 0.5;
    std::vector<on::Pair> p = on::plan(in, 4, 100);
    REQUIRE_EQ(p.size(), (size_t)2);
    CHECK_EQ(p[0].rank, 0);   /* the larger gain first */
    CHECK_EQ(p[0].in, 6);
    CHECK_EQ(p[0].out, 3);    /* the coldest resident */
    CHECK_EQ(p[1].rank, 1);
    CHECK_EQ(p[1].in, 7);
    CHECK_EQ(p[1].out, 3);
    CHECK(on::plan(in, 1, 100).size() == 1);   /* at most n */
    /* the margin: 0.5 + 0.06 against unit 3's 0.3: under 200% */
    std::vector<on::Pair> pm = on::plan(in, 4, 200);
    REQUIRE_EQ(pm.size(), (size_t)1);
    CHECK_EQ(pm[0].rank, 0);
    /* pinned and not planned */
    st0[6] = on::kPinned;
    p = on::plan(in, 4, 100);
    REQUIRE_EQ(p.size(), (size_t)1);
    CHECK_EQ(p[0].rank, 1);
    in[1].ok = false;
    CHECK(on::plan(in, 4, 100).empty());
    in[1].ok = true;
    st0[6] = 0;
    /* ties: equal scores in come in the startup order (before: share desc, then expert), out the reverse */
    on::RankScores t;
    std::vector<ds41aux::Cand> ct = cands(0, { 0.02f, 0.02f, 0.02f, 0.02f });
    t.init(ct, 0);
    t.hits = { 0.0, 0.0, 1.0, 1.0 };
    std::vector<uint8_t> stt = { 1, 1, 0, 0 };
    std::vector<on::RankIn> it(1);
    it[0] = { &t, &ct, stt.data(), true };
    std::vector<on::Pair> q = on::plan(it, 2, 100);
    REQUIRE_EQ(q.size(), (size_t)2);
    CHECK_EQ(q[0].in, 2);
    CHECK_EQ(q[0].out, 1);   /* among equal outs the later in the startup order leaves first */
    CHECK_EQ(q[1].in, 3);
    CHECK_EQ(q[1].out, 0);
    /* equal gains across ranks: the lower rank first */
    on::RankScores u0, u1;
    u0.init(ct, 0);
    u1.init(ct, 0);
    u0.hits = { 0.0, 0.0, 1.0, 0.0 };
    u1.hits = { 0.0, 0.0, 1.0, 0.0 };
    std::vector<uint8_t> su = { 1, 1, 0, 0 };
    std::vector<on::RankIn> iu(2);
    iu[0] = { &u0, &ct, su.data(), true };
    iu[1] = { &u1, &ct, su.data(), true };
    std::vector<on::Pair> r = on::plan(iu, 1, 100);
    REQUIRE_EQ(r.size(), (size_t)1);
    CHECK_EQ(r[0].rank, 0);
}

/* ---- the replay fixture ---- */

namespace {

struct Reader {
    std::vector<uint8_t> b;
    size_t at = 0;
    bool ok = true;
    template <class T> T get() {
        T v{};
        if (at + sizeof(T) > b.size()) {
            ok = false;
            return v;
        }
        std::memcpy(&v, b.data() + at, sizeof(T));
        at += sizeof(T);
        return v;
    }
};

struct Fixture {
    uint32_t NR = 0, NL = 0, PER = 0, ntok = 0;
    std::vector<uint16_t> picks;   /* ntok x NL x 6 */
    std::vector<std::vector<ds41aux::Cand>> cand;   /* by rank, ds41aux::before order */
    std::map<std::pair<int, int>, std::pair<int, int>> index;   /* (layer, global expert) -> (rank, candidate) */
    uint32_t N = 0, spare = 0, H = 0, n0 = 0, rate = 0, smax = 0, every = 0, margin = 0;
    struct Step {
        std::vector<uint32_t> toks;
        std::vector<std::array<uint32_t, 3>> swaps;
    };
    struct Run {
        uint32_t C = 0;
        std::vector<Step> steps;
        uint64_t rows = 0, hit = 0;
    };
    std::vector<Run> runs;

    bool load(const char* path) {
        Reader r;
        FILE* f = std::fopen(path, "rb");
        if (!f) return false;
        uint8_t buf[65536];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) r.b.insert(r.b.end(), buf, buf + n);
        std::fclose(f);
        if (r.b.size() < 8 || std::memcmp(r.b.data(), "AUXONL1\0", 8) != 0) return false;
        r.at = 8;
        NR = r.get<uint32_t>();
        NL = r.get<uint32_t>();
        PER = r.get<uint32_t>();
        ntok = r.get<uint32_t>();
        picks.resize((size_t)ntok * NL * 6);
        for (uint16_t& p : picks) p = r.get<uint16_t>();
        cand.resize(NR);
        for (uint32_t k = 0; k < NR; ++k) {
            const uint32_t nc = r.get<uint32_t>();
            for (uint32_t i = 0; i < nc; ++i) {
                ds41aux::Cand c;
                c.layer = r.get<uint8_t>();
                c.expert = r.get<uint8_t>();
                (void)r.get<uint16_t>();
                c.share = r.get<float>();
                c.rank = (int)k;
                c.unit = c.layer * (int32_t)PER + c.expert;
                index[{ c.layer, (int)(k * PER + (uint32_t)c.expert) }] = { (int)k, (int)i };
                cand[k].push_back(c);
            }
        }
        N = r.get<uint32_t>();
        spare = r.get<uint32_t>();
        H = r.get<uint32_t>();
        n0 = r.get<uint32_t>();
        rate = r.get<uint32_t>();
        smax = r.get<uint32_t>();
        every = r.get<uint32_t>();
        margin = r.get<uint32_t>();
        const uint32_t nr = r.get<uint32_t>();
        for (uint32_t k = 0; k < nr && r.ok; ++k) {
            Run run;
            run.C = r.get<uint32_t>();
            const uint32_t ns = r.get<uint32_t>();
            for (uint32_t s = 0; s < ns && r.ok; ++s) {
                Step st;
                const uint32_t nt = r.get<uint32_t>();
                for (uint32_t t = 0; t < nt; ++t) st.toks.push_back(r.get<uint32_t>());
                const uint32_t nw = r.get<uint32_t>();
                for (uint32_t w = 0; w < nw; ++w) {
                    std::array<uint32_t, 3> x;
                    x[0] = r.get<uint32_t>();
                    x[1] = r.get<uint32_t>();
                    x[2] = r.get<uint32_t>();
                    st.swaps.push_back(x);
                }
                run.steps.push_back(st);
            }
            run.rows = r.get<uint64_t>();
            run.hit = r.get<uint64_t>();
            runs.push_back(run);
        }
        return r.ok && r.at == r.b.size();
    }
};

}  // namespace

TEST(the_cpp_rule_makes_the_replays_swaps) {
#ifndef RAD_AUX_ONLINE_FIXTURE
    CHECK(!"RAD_AUX_ONLINE_FIXTURE is not set");
#else
    Fixture fx;
    REQUIRE(fx.load(RAD_AUX_ONLINE_FIXTURE));
    REQUIRE_EQ(fx.NR, 4u);
    REQUIRE_EQ(fx.runs.size(), (size_t)2);
    for (uint32_t r = 0; r < fx.NR; ++r)   /* the lists are in the startup order */
        for (size_t i = 1; i < fx.cand[r].size(); ++i) REQUIRE(ds41aux::before(fx.cand[r][i - 1], fx.cand[r][i]));
    for (const Fixture::Run& run : fx.runs) {
        /* the sealed set: the global order's first N - spare (ds41aux::quotas), each rank's first quota */
        std::vector<ds41aux::Cand> all;
        for (const auto& v : fx.cand) all.insert(all.end(), v.begin(), v.end());
        const std::vector<int64_t> q = ds41aux::quotas(all, (int64_t)fx.N - fx.spare, (int)fx.NR);
        std::vector<std::vector<uint8_t>> res(fx.NR);
        std::vector<on::RankScores> sc(fx.NR);
        for (uint32_t r = 0; r < fx.NR; ++r) {
            res[r].assign(fx.cand[r].size(), 0);
            for (int64_t i = 0; i < q[r]; ++i) res[r][(size_t)i] = 1;
            sc[r].init(fx.cand[r], fx.n0);
        }
        std::map<size_t, std::vector<std::array<uint32_t, 3>>> pending;
        on::Credit credit;
        uint64_t merges = 0, tok_since = 0, rows = 0, hit = 0;
        int bad_steps = 0;
        size_t first_bad = (size_t)-1;
        for (size_t si = 0; si < run.steps.size(); ++si) {
            const Fixture::Step& st = run.steps[si];
            std::vector<std::vector<uint32_t>> delta(fx.NR);
            for (uint32_t r = 0; r < fx.NR; ++r) delta[r].assign(fx.cand[r].size(), 0);
            for (uint32_t t : st.toks)
                for (uint32_t l = 0; l < fx.NL; ++l)
                    for (int k = 0; k < 6; ++k) {
                        const uint16_t x = fx.picks[((size_t)t * fx.NL + l) * 6 + (size_t)k];
                        if (x == 0xFFFF) continue;
                        auto it = fx.index.find({ (int)l, (int)x });
                        if (it == fx.index.end()) continue;
                        ++rows;
                        hit += res[(size_t)it->second.first][(size_t)it->second.second];
                        ++delta[(size_t)it->second.first][(size_t)it->second.second];
                    }
            const uint64_t T = st.toks.size();
            const double f = on::decay_factor(T, fx.H);
            for (uint32_t r = 0; r < fx.NR; ++r) on::merge(&sc[r], delta[r].data(), f);
            /* the boundary before s + 1: the flips decided at s - 1 */
            for (const auto& w : pending[si + 1]) {
                res[w[0]][w[1]] = 1;
                res[w[0]][w[2]] = 0;
            }
            pending.erase(si + 1);
            ++merges;
            tok_since += T;
            std::vector<std::array<uint32_t, 3>> chosen;
            if (merges % fx.every == 0) {
                int n = credit.grant(tok_since, fx.rate, fx.smax);
                tok_since = 0;
                int64_t held = 0;
                for (uint32_t r = 0; r < fx.NR; ++r)
                    for (uint8_t x : res[r]) held += x;
                n = (int)std::min<int64_t>(n, (int64_t)fx.N - held);
                if (n > 0) {
                    std::vector<on::RankIn> in(fx.NR);
                    for (uint32_t r = 0; r < fx.NR; ++r) in[r] = { &sc[r], &fx.cand[r], res[r].data(), true };
                    const std::vector<on::Pair> p = on::plan(in, n, fx.margin);
                    credit.spend((int)p.size());
                    for (const on::Pair& x : p) chosen.push_back({ (uint32_t)x.rank, (uint32_t)x.in, (uint32_t)x.out });
                    for (const auto& w : chosen) pending[si + 2].push_back(w);
                }
            }
            if (chosen != st.swaps) {
                ++bad_steps;
                if (first_bad == (size_t)-1) first_bad = si;
            }
        }
        if (bad_steps)
            std::fprintf(stderr, "C=%u: %d of %zu steps differ from the replay, the first at step %zu\n", run.C,
                         bad_steps, run.steps.size(), first_bad);
        CHECK_EQ(bad_steps, 0);
        CHECK_EQ(rows, run.rows);
        CHECK_EQ(hit, run.hit);
        size_t swaps = 0;
        for (const auto& s : run.steps) swaps += s.swaps.size();
        CHECK(swaps > 0);
    }
#endif
}

RAD_TEST_MAIN()
