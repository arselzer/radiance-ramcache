/* aux_compact_test.cpp -- the aux expert worker's compaction and placement (tools/ds41-host-experts/ds41_aux.h
 * compact(); work/aux/SPEC.md §9.2, §10.2, §11), pure:
 *  - compaction against FileStager::hx_jobs_for's slot rule on routings moe_scatter could write: dropped slots
 *    (-1 inside a range), slots past the live end (a token at or past AR), the rank's last expert (whose range
 *    runs to T over the other ranks' slots), empty layers, and the count-mismatch error; libr4d's mx_range and
 *    the live-tail ranges on the compacted order give each held expert exactly its run, and every slot reads
 *    the token row the CPU's job reads;
 *  - the run copy: the worker's rows merged into the CPU's give the same y_expert bytes as copying both sets,
 *    and the placement writes exactly the right 10,240-byte rows of hx_out_;
 *  - the verify count: one flipped bit is one row and one value. */
#include "rad_test.h"
#include "place/aux_expert_worker.h"

#include <algorithm>
#include <cstring>
#include <random>
#include <vector>

using namespace ds41aux;

namespace {

/* A pass's sorted order for rank 0 of four (local experts 0..95 = global 0..95), as moe_scatter writes it:
 * each local expert's slots (token * 6 + k) in token order, then the other ranks' slots as -1. `drop` puts a
 * dropped slot (-1) inside some ranges, `past` a slot naming a token at AR. counts: the live slots. */
struct Route {
    int AR = 0, T = 0;
    std::vector<int32_t> sorted, off, counts;
};
Route route(int AR, uint32_t seed, bool drop, bool past) {
    Route r;
    r.AR = AR;
    r.T = AR * kTopK;
    std::mt19937 rng(seed);
    std::vector<std::vector<int32_t>> by_e(kExperts);
    for (int t = 0; t < AR; ++t) {
        std::vector<int> g(384);
        for (int i = 0; i < 384; ++i) g[(size_t)i] = i;
        std::shuffle(g.begin(), g.end(), rng);
        for (int k = 0; k < kTopK; ++k)
            if (g[(size_t)k] < kExperts) by_e[(size_t)g[(size_t)k]].push_back(t * kTopK + k);
    }
    r.counts.assign(kExperts, 0);
    r.off.assign(kExperts + 1, 0);
    int live = 0;
    for (const auto& v : by_e) live += (int)v.size();
    int room = r.T - live;   /* slots the other ranks own: room for the injected ones */
    int at = 0;
    for (int e = 0; e < kExperts; ++e) {
        r.off[(size_t)e] = at;
        for (int32_t s : by_e[(size_t)e]) {
            r.sorted.push_back(s);
            ++at;
            ++r.counts[(size_t)e];
        }
        /* A dropped slot or one past the live end, inside expert e's range, never counted. */
        if (drop && e % 17 == 3 && room > 0) { r.sorted.push_back(-1); ++at; --room; }
        if (past && e % 23 == 5 && room > 0) { r.sorted.push_back(AR * kTopK + 2); ++at; --room; }
    }
    r.off[kExperts] = at;
    while ((int)r.sorted.size() < r.T) r.sorted.push_back(-1);
    r.sorted.resize((size_t)r.T);
    return r;
}

/* hx_jobs_for's rows of expert e: its live slots. */
std::vector<int32_t> cpu_rows(const Route& r, int e) {
    int lo = e == 0 ? 0 : r.off[(size_t)e], hi = e == kExperts - 1 ? r.T : r.off[(size_t)e + 1];
    lo = std::clamp(lo, 0, r.T);
    hi = std::clamp(hi, lo, r.T);
    std::vector<int32_t> v;
    for (int i = lo; i < hi; ++i)
        if (r.sorted[(size_t)i] >= 0 && r.sorted[(size_t)i] / kTopK < r.AR) v.push_back(i);
    return v;
}
/* libr4d's mx_range, and the live tail's range (r4d_ds41_live_tail_policy.h). */
void mx_range(const int32_t* off, int e, int ne, int T, int* s, int* t) {
    int a = e == 0 ? 0 : off[e], b = e == ne - 1 ? T : off[e + 1];
    a = a < 0 ? 0 : (a > T ? T : a);
    b = b < a ? a : (b > T ? T : b);
    *s = a;
    *t = b;
}
void tail_range(const int32_t* off, int e, int T, int* s, int* t) {
    auto clamp = [&](int v) { return v < 0 ? 0 : (v > T ? T : v); };
    int a = e == 0 ? 0 : clamp(off[e]), b = clamp(off[e + 1]);
    *s = a;
    *t = b < a ? a : b;
}

void check_compaction(const Route& r, const std::vector<uint8_t>& worker, int max_rows = kMaxRows) {
    static Compact c;   /* 24 KiB: off the stack */
    REQUIRE_EQ(compact(r.sorted.data(), r.off.data(), r.AR, kTopK, kExperts, worker.data(), r.counts.data(), kExperts, &c,
                       max_rows), 0);
    int j = 0;
    CHECK_EQ(c.off[0], 0);
    for (int e = 0; e < kExperts; ++e) {
        const std::vector<int32_t> want = worker[(size_t)e] ? cpu_rows(r, e) : std::vector<int32_t>{};
        CHECK_EQ(c.off[e + 1] - c.off[e], (int32_t)want.size());
        for (size_t k = 0; k < want.size(); ++k, ++j) {
            CHECK_EQ(c.row_of[j], want[k]);
            CHECK_EQ(c.sorted[j], r.sorted[(size_t)want[k]]);
            /* gate_up's slot j reads token row sorted'[j] / 6: the row the CPU's job reads. */
            CHECK_EQ(c.sorted[j] / kTopK, r.sorted[(size_t)want[k]] / kTopK);
            CHECK(c.sorted[j] >= 0 && c.sorted[j] / kTopK < r.AR);
        }
        int s = 0, t = 0;
        mx_range(c.off, e, kExperts, c.T, &s, &t);
        CHECK_EQ(s, c.off[e]);
        CHECK_EQ(t, c.off[e + 1]);
        tail_range(c.off, e, c.T, &s, &t);
        CHECK_EQ(s, c.off[e]);
        CHECK_EQ(t, c.off[e + 1]);
    }
    CHECK_EQ(c.T, j);
    CHECK_EQ(c.off[kExperts], c.T);
}

}  // namespace

TEST(compaction_follows_the_cpu_rows_slot_rule) {
    for (int AR : { 1, 2, 4, 8, 16 })
        for (uint32_t seed = 1; seed <= 6; ++seed)
            for (int form = 0; form < 4; ++form) {
                const Route r = route(AR, seed * 101 + (uint32_t)AR, form & 1, form & 2);
                /* Every live expert, every other one, one, and the rank's last. */
                std::vector<uint8_t> all(kExperts, 0), half(kExperts, 0), one(kExperts, 0), last(kExperts, 0);
                int first_live = -1;
                for (int e = 0; e < kExperts; ++e) {
                    all[(size_t)e] = r.counts[(size_t)e] > 0;
                    half[(size_t)e] = r.counts[(size_t)e] > 0 && e % 2;
                    if (r.counts[(size_t)e] > 0 && first_live < 0) first_live = e;
                }
                if (first_live >= 0) one[(size_t)first_live] = 1;
                last[kExperts - 1] = r.counts[kExperts - 1] > 0;
                check_compaction(r, all);
                check_compaction(r, half);
                check_compaction(r, one);
                check_compaction(r, last);
            }
}

/* RADIANCE_EXPERIMENTAL_DS41_AUX_ROWS above 16: the same slot rule at 17..512 rows with the bound raised; a pass
 * taller than the bound (or than kRowsCap) is a shape outside the worker's, and the default bound is today's 16. */
TEST(taller_passes_compact_by_the_same_rule) {
    for (int AR : { 17, 24, 33, 64, 100, 128, 200, 256, 512 })
        for (uint32_t seed = 1; seed <= 3; ++seed)
            for (int form = 0; form < 4; ++form) {
                const Route r = route(AR, seed * 977 + (uint32_t)AR, form & 1, form & 2);
                std::vector<uint8_t> all(kExperts, 0), third(kExperts, 0), last(kExperts, 0);
                for (int e = 0; e < kExperts; ++e) {
                    all[(size_t)e] = r.counts[(size_t)e] > 0;
                    third[(size_t)e] = r.counts[(size_t)e] > 0 && e % 3 == 1;
                }
                last[kExperts - 1] = r.counts[kExperts - 1] > 0;
                check_compaction(r, all, AR);
                check_compaction(r, third, 512);
                check_compaction(r, last, std::max(AR, 256));
            }
    static Compact c;
    const Route r = route(128, 5, false, false);
    std::vector<uint8_t> all(kExperts, 1);
    CHECK_EQ(compact(r.sorted.data(), r.off.data(), r.AR, kTopK, kExperts, all.data(), r.counts.data(), kExperts, &c),
             kCompactShape);                                              /* the default bound: 16 */
    CHECK_EQ(compact(r.sorted.data(), r.off.data(), r.AR, kTopK, kExperts, all.data(), r.counts.data(), kExperts, &c, 127),
             kCompactShape);
    CHECK_EQ(compact(r.sorted.data(), r.off.data(), r.AR, kTopK, kExperts, all.data(), r.counts.data(), kExperts, &c, 128), 0);
    CHECK_EQ(compact(r.sorted.data(), r.off.data(), r.AR, kTopK, kExperts, all.data(), r.counts.data(), kExperts, &c,
                     kRowsCap + 1), kCompactShape);                       /* past Compact's arrays */
    CHECK_EQ(compact(r.sorted.data(), r.off.data(), r.AR, kTopK, kExperts, all.data(), r.counts.data(), kExperts, &c, 0),
             kCompactShape);
    /* A whole pass of 512 rows all on this rank's experts: 3072 slots, exactly the arrays' capacity. */
    Route w;
    w.AR = 512;
    w.T = 512 * kTopK;
    w.counts.assign(kExperts, 0);
    w.off.assign(kExperts + 1, 0);
    std::vector<std::vector<int32_t>> by(kExperts);
    for (int t = 0; t < w.AR; ++t)
        for (int k = 0; k < kTopK; ++k) by[(size_t)((t + 16 * k) % kExperts)].push_back(t * kTopK + k);
    for (int e = 0; e < kExperts; ++e) {
        w.off[(size_t)e] = (int32_t)w.sorted.size();
        for (int32_t x : by[(size_t)e]) w.sorted.push_back(x);
        w.counts[(size_t)e] = (int32_t)by[(size_t)e].size();
    }
    w.off[kExperts] = (int32_t)w.sorted.size();
    REQUIRE_EQ((int)w.sorted.size(), w.T);
    check_compaction(w, all, 512);
}

TEST(the_ranks_last_expert_runs_to_T_over_other_ranks_slots) {
    /* Expert 95 with two live slots, then four of other ranks' (-1) and one past the live end. */
    Route r;
    r.AR = 2;
    r.T = 12;
    r.sorted = { 0, 1, 2, 3, 4, 5, 6, -1, -1, 11, -1, 14 };
    r.off.assign(kExperts + 1, 0);
    for (int e = 1; e <= kExperts; ++e) r.off[(size_t)e] = 7;   /* experts 1..95 start at 7 */
    r.off[0] = 0;
    r.counts.assign(kExperts, 0);
    r.counts[0] = 7;                    /* expert 0: slots 0..6 */
    r.off[kExperts] = 9;                /* the local slots end at 9 */
    std::vector<uint8_t> w(kExperts, 0);
    w[kExperts - 1] = 1;
    /* Expert 95's range is [off[95], T) = [7, 12): live slots 9 (11 / 6 = 1 < 2); 14 / 6 = 2 is past AR. */
    r.counts[kExperts - 1] = 1;
    Compact c;
    REQUIRE_EQ(compact(r.sorted.data(), r.off.data(), r.AR, kTopK, kExperts, w.data(), r.counts.data(), kExperts, &c), 0);
    CHECK_EQ(c.T, 1);
    CHECK_EQ(c.row_of[0], 9);
    CHECK_EQ(c.sorted[0], 11);
    CHECK_EQ(c.off[kExperts - 1], 0);
    CHECK_EQ(c.off[kExperts], 1);
}

TEST(empty_layers_and_count_mismatch_and_shapes) {
    const Route r = route(4, 77, true, true);
    std::vector<uint8_t> none(kExperts, 0);
    Compact c;
    CHECK_EQ(compact(r.sorted.data(), r.off.data(), r.AR, kTopK, kExperts, none.data(), r.counts.data(), kExperts, &c), 0);
    CHECK_EQ(c.T, 0);
    for (int e = 0; e <= kExperts; ++e) CHECK_EQ(c.off[e], 0);
    /* A held expert whose live slots disagree with its count: -(e + 1). A non-held one is not checked. */
    int e0 = -1, e1 = -1;
    for (int e = 0; e < kExperts && (e0 < 0 || e1 < 0); ++e)
        if (r.counts[(size_t)e] > 0) (e0 < 0 ? e0 : e1) = e;
    REQUIRE(e0 >= 0 && e1 >= 0);
    std::vector<int32_t> counts = r.counts;
    counts[(size_t)e0] += 1;
    std::vector<uint8_t> w(kExperts, 0);
    w[(size_t)e0] = 1;
    CHECK_EQ(compact(r.sorted.data(), r.off.data(), r.AR, kTopK, kExperts, w.data(), counts.data(), kExperts, &c), -(e0 + 1));
    w[(size_t)e0] = 0;
    w[(size_t)e1] = 1;
    CHECK_EQ(compact(r.sorted.data(), r.off.data(), r.AR, kTopK, kExperts, w.data(), counts.data(), kExperts, &c), 0);
    /* A count array too short for a held expert. */
    CHECK_EQ(compact(r.sorted.data(), r.off.data(), r.AR, kTopK, kExperts, w.data(), counts.data(), e1, &c), -(e1 + 1));
    /* Shapes outside the worker's: no rows, more than 16, another top_k or expert count. */
    CHECK_EQ(compact(r.sorted.data(), r.off.data(), 0, kTopK, kExperts, w.data(), r.counts.data(), kExperts, &c), kCompactShape);
    CHECK_EQ(compact(r.sorted.data(), r.off.data(), 17, kTopK, kExperts, w.data(), r.counts.data(), kExperts, &c), kCompactShape);
    CHECK_EQ(compact(r.sorted.data(), r.off.data(), 4, 5, kExperts, w.data(), r.counts.data(), kExperts, &c), kCompactShape);
    CHECK_EQ(compact(r.sorted.data(), r.off.data(), 4, kTopK, 32, w.data(), r.counts.data(), kExperts, &c), kCompactShape);
}

TEST(run_copy_and_placement) {
    const Route r = route(16, 4242, true, false);
    std::vector<uint8_t> w(kExperts, 0);
    std::vector<int32_t> cpu;   /* the CPU's rows: every live expert the worker does not hold */
    for (int e = 0; e < kExperts; ++e) {
        if (!r.counts[(size_t)e]) continue;
        if (e % 3 == 0) w[(size_t)e] = 1;
        else for (int32_t i : cpu_rows(r, e)) cpu.push_back(i);
    }
    Compact c;
    REQUIRE_EQ(compact(r.sorted.data(), r.off.data(), r.AR, kTopK, kExperts, w.data(), r.counts.data(), kExperts, &c), 0);
    REQUIRE(c.T > 0);
    const int64_t row = kEmbd;
    const size_t rb = (size_t)row * 2;
    /* hx_out_: the CPU's rows written with one pattern, every other row with garbage; the worker's out rows. */
    std::vector<uint16_t> hx_out((size_t)r.T * row, 0xdead), out((size_t)c.T * row);
    for (int32_t i : cpu)
        for (int64_t k = 0; k < row; ++k) hx_out[(size_t)(i * row + k)] = (uint16_t)(i * 7 + k);
    for (int j = 0; j < c.T; ++j)
        for (int64_t k = 0; k < row; ++k) out[(size_t)(j * row + k)] = (uint16_t)(0x4000 + j * 13 + k);
    const std::vector<uint16_t> before = hx_out;
    /* The placement (aux_finish): out's row j to hx_out_'s row row_of[j]. */
    for (int j = 0; j < c.T; ++j)
        std::memcpy(hx_out.data() + (int64_t)c.row_of[j] * row, out.data() + (int64_t)j * row, rb);
    std::vector<uint8_t> is_w((size_t)r.T, 0), is_c((size_t)r.T, 0);
    for (int j = 0; j < c.T; ++j) is_w[(size_t)c.row_of[j]] = 1;
    for (int32_t i : cpu) is_c[(size_t)i] = 1;
    for (int i = 0; i < r.T; ++i) {
        CHECK(!(is_w[(size_t)i] && is_c[(size_t)i]));   /* disjoint */
        const bool same = std::memcmp(hx_out.data() + (int64_t)i * row, before.data() + (int64_t)i * row, rb) == 0;
        if (!is_w[(size_t)i]) CHECK(same);               /* nothing else written */
    }
    for (int j = 0; j < c.T; ++j)
        CHECK(std::memcmp(hx_out.data() + (int64_t)c.row_of[j] * row, out.data() + (int64_t)j * row, rb) == 0);
    /* hx_finish's runs over the merged, sorted rows give y_expert the bytes of copying both sets. */
    std::vector<int32_t> rows = cpu;
    for (int j = 0; j < c.T; ++j) rows.push_back(c.row_of[j]);
    std::sort(rows.begin(), rows.end());
    CHECK(std::adjacent_find(rows.begin(), rows.end()) == rows.end());
    std::vector<uint16_t> y1((size_t)r.T * row, 0x1111), y2 = y1;
    for (size_t i = 0; i < rows.size();) {
        size_t j = i + 1;
        while (j < rows.size() && rows[j] == rows[j - 1] + 1) ++j;
        std::memcpy(y1.data() + (int64_t)rows[i] * row, hx_out.data() + (int64_t)rows[i] * row, rb * (j - i));
        i = j;
    }
    for (int32_t i : cpu) std::memcpy(y2.data() + (int64_t)i * row, hx_out.data() + (int64_t)i * row, rb);
    for (int j = 0; j < c.T; ++j)
        std::memcpy(y2.data() + (int64_t)c.row_of[j] * row, out.data() + (int64_t)j * row, rb);
    CHECK(y1 == y2);
}

TEST(verify_counts_one_flipped_bit_as_one_row_and_one_value) {
    std::vector<uint16_t> a((size_t)kEmbd), b;
    for (int i = 0; i < kEmbd; ++i) a[(size_t)i] = (uint16_t)(i * 31 + 7);
    b = a;
    int first = -1;
    CHECK_EQ(row_diff(a.data(), b.data(), kEmbd, &first), 0);
    CHECK_EQ(first, -1);
    b[1234] ^= 0x0001;
    CHECK_EQ(row_diff(a.data(), b.data(), kEmbd, &first), 1);
    CHECK_EQ(first, 1234);
    b[4000] ^= 0x8000;
    CHECK_EQ(row_diff(a.data(), b.data(), kEmbd, &first), 2);
    CHECK_EQ(first, 1234);
}

RAD_TEST_MAIN()
