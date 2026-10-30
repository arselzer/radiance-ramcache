/* router_predictor_rows_test.cpp -- the read-ahead in multi-row decode passes
 * (RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MULTIROW), its host arithmetic and pass policy: the router
 * logits of several rows in one sweep of the weights are, row by row, bit for bit those of the
 * single-row kernel (with the AVX2 kernel and the scalar loop, rows split into groups of six and
 * the 8-wide tail taken); a row's top k is the router key's order; the union of the rows' top k
 * runs position by position, row by row, each expert once; the switch's value is 1 (off), 2..limit
 * or refused; and only one-token passes, or with the switch trunk decode passes of 2..N rows, read
 * ahead -- never a prefill or mixed pass, nor the drafter's own. */
#include "rad_test.h"

#include "place/router_predictor.h"
#include "place/stager.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using rad::RouterPredictor;

namespace {

uint16_t bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fff + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
    float next() {   /* in [-1, 1) */
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return (float)((double)(s >> 11) / (double)(1ull << 53)) * 2.0f - 1.0f;
    }
};

/* A router of n experts over embd inputs (bf16) and `rows` inputs. */
struct Case {
    int n, embd, rows;
    std::vector<uint16_t> w;
    std::vector<float> x;
    std::vector<const float*> xs;
    Case(int n_, int embd_, int rows_, uint64_t seed) : n(n_), embd(embd_), rows(rows_) {
        Rng g(seed);
        w.resize((size_t)n * embd);
        for (auto& v : w) v = bf16(g.next() * 0.05f);
        x.resize((size_t)rows * embd);
        for (auto& v : x) v = g.next() * 3.0f;
        for (int r = 0; r < rows; ++r) xs.push_back(&x[(size_t)r * embd]);
    }
};

bool same_bits(float a, float b) { return std::memcmp(&a, &b, 4) == 0; }

/* Each row's logits from one multi-row call against the same row in a one-row call. */
int rows_differ(const Case& c, bool avx2) {
    std::vector<float> all((size_t)c.rows * c.n), one((size_t)c.n);
    rad::router_logits_rows(c.w.data(), c.n, c.embd, c.xs.data(), c.rows, avx2, all.data());
    int bad = 0;
    for (int r = 0; r < c.rows; ++r) {
        const float* xr = c.xs[(size_t)r];
        rad::router_logits_rows(c.w.data(), c.n, c.embd, &xr, 1, avx2, one.data());
        for (int e = 0; e < c.n; ++e) bad += !same_bits(all[(size_t)r * c.n + e], one[(size_t)e]);
    }
    return bad;
}

bool have_avx2() { return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"); }

}  // namespace

/* Rows 2..16 (groups of six and their remainders), a width with the 8-wide tail (1032 = 64 x 16 +
 * 8) and the model's (5120): every row's logits equal the one-row kernel's, bit for bit. */
TEST(multirow_logits_equal_the_single_row_kernel_bit_for_bit) {
    if (!have_avx2()) SKIP("no AVX2/FMA on this CPU");
    for (int rows : { 2, 3, 5, 6, 7, 12, 13, 16 })
        for (int embd : { 1032, 5120 }) {
            const Case c(48, embd, rows, (uint64_t)(rows * 131 + embd));
            CHECK_EQ(rows_differ(c, true), 0);
        }
}

TEST(multirow_logits_scalar_loop_equals_its_single_row) {
    const Case c(24, 1032, 6, 7);
    CHECK_EQ(rows_differ(c, false), 0);
}

/* The AVX2 rows against a double-precision reference: the same numbers, to rounding. */
TEST(multirow_logits_are_the_dot_products) {
    if (!have_avx2()) SKIP("no AVX2/FMA on this CPU");
    const Case c(16, 5120, 4, 11);
    std::vector<float> all((size_t)c.rows * c.n);
    rad::router_logits_rows(c.w.data(), c.n, c.embd, c.xs.data(), c.rows, true, all.data());
    for (int r = 0; r < c.rows; ++r)
        for (int e = 0; e < c.n; ++e) {
            double ref = 0;
            for (int k = 0; k < c.embd; ++k) {
                uint32_t u = (uint32_t)c.w[(size_t)e * c.embd + k] << 16;
                float wf;
                std::memcpy(&wf, &u, 4);
                ref += (double)wf * (double)c.x[(size_t)r * c.embd + k];
            }
            CHECK_NEAR(all[(size_t)r * c.n + e], ref, 1e-3 * (1.0 + std::fabs(ref)));
        }
}

/* The router key's best k, best first, with the correction selecting. */
TEST(router_topk_takes_the_router_keys_best_first) {
    const int n = 384, k = 6;
    std::vector<float> logit(n), corr(n, 0.0f);
    /* Distinct logits 0.01 apart (7919 is prime to 384), so no two keys tie. */
    for (int e = 0; e < n; ++e) logit[(size_t)e] = (float)((e * 7919) % 384) * 0.01f - 1.9f;
    corr[300] = 10.0f;   /* the correction alone puts 300 first */
    int32_t top[k];
    rad::router_topk(logit.data(), corr.data(), n, k, top);
    std::vector<std::pair<double, int>> key;
    for (int e = 0; e < n; ++e) {
        const double l = logit[(size_t)e];
        key.push_back({ std::sqrt(std::log1p(std::exp(l))) + corr[(size_t)e], e });
    }
    std::sort(key.begin(), key.end(), [](auto& a, auto& b) { return a.first > b.first; });
    CHECK_EQ(top[0], 300);
    for (int i = 0; i < k; ++i) CHECK_EQ(top[i], key[(size_t)i].second);
}

/* A multi-row call's row r gives the same top k as row r alone (the read-ahead's row 0 of a verify
 * pass is what a one-token pass of that row would read ahead). */
TEST(multirow_topk_per_row_equals_the_row_alone) {
    const bool avx2 = have_avx2();
    const Case c(RouterPredictor::kExperts, 1032, 5, 23);
    std::vector<float> corr((size_t)c.n);
    Rng g(5);
    for (auto& v : corr) v = g.next() * 0.01f;
    std::vector<float> all((size_t)c.rows * c.n), one((size_t)c.n);
    rad::router_logits_rows(c.w.data(), c.n, c.embd, c.xs.data(), c.rows, avx2, all.data());
    for (int r = 0; r < c.rows; ++r) {
        const float* xr = c.xs[(size_t)r];
        rad::router_logits_rows(c.w.data(), c.n, c.embd, &xr, 1, avx2, one.data());
        int32_t a[8], b[8];
        rad::router_topk(&all[(size_t)r * c.n], corr.data(), c.n, 8, a);
        rad::router_topk(one.data(), corr.data(), c.n, 8, b);
        CHECK(std::memcmp(a, b, sizeof a) == 0);
    }
}

/* The union: every row's best before any row's second, each expert once, ids outside the router
 * left out; one row is its own list. */
TEST(topk_union_runs_position_by_position_row_by_row) {
    const int32_t top[3 * 4] = { 5, 9, 2, 7,      /* row 0 */
                                 9, 5, 3, 7,      /* row 1 */
                                 1, 400, 2, -1 }; /* row 2: an id past the router, a negative one */
    int32_t u[12];
    const int n = rad::router_topk_union(top, 3, 4, u);
    /* Position 0: 5, 9, 1; position 1: nothing new (400 is past the router); 2: 2 (row 0), 3
     * (row 1); 3: 7. */
    const int32_t want[] = { 5, 9, 1, 2, 3, 7 };
    REQUIRE_EQ(n, 6);
    for (int i = 0; i < n; ++i) CHECK_EQ(u[i], want[i]);
    int32_t one[4];
    CHECK_EQ(rad::router_topk_union(top, 1, 4, one), 4);
    CHECK(std::memcmp(one, top, sizeof one) == 0);
    /* The most a post carries: 16 rows of 16, all distinct, fits the order key's byte. */
    std::vector<int32_t> big(16 * 16), out(16 * 16);
    for (int i = 0; i < 256; ++i) big[(size_t)i] = i;
    CHECK_EQ(rad::router_topk_union(big.data(), 16, 16, out.data()), 256);
    CHECK_EQ(out[1], 16);   /* row 1's best second */
}

TEST(multirow_switch_value_is_off_n_or_refused) {
    CHECK_EQ(rad::ra_multirow_rows(nullptr, 16), 1);
    CHECK_EQ(rad::ra_multirow_rows("", 16), 1);
    CHECK_EQ(rad::ra_multirow_rows("0", 16), 1);
    CHECK_EQ(rad::ra_multirow_rows("6", 16), 6);
    CHECK_EQ(rad::ra_multirow_rows("2", 16), 2);
    CHECK_EQ(rad::ra_multirow_rows("16", 16), 16);
    CHECK_EQ(rad::ra_multirow_rows("06", 16), 6);
    CHECK_EQ(rad::ra_multirow_rows("1", 16), -1);    /* one row is what a plain run reads ahead */
    CHECK_EQ(rad::ra_multirow_rows("17", 16), -1);   /* past the EP path's rows */
    CHECK_EQ(rad::ra_multirow_rows("7", 6), -1);
    CHECK_EQ(rad::ra_multirow_rows("on", 16), -1);
    CHECK_EQ(rad::ra_multirow_rows("-3", 16), -1);
    CHECK_EQ(rad::ra_multirow_rows("99999999999999", 16), -1);
}

/* What a prediction costs by its rows (the read-ahead's choice: every row of a verify pass, not row
 * 0 with a wider k): the model's 40 routers (157 MB of bf16, so every layer's 3.9 MB comes from
 * memory as in the engine), each layer scored for R rows in one sweep against R one-row sweeps.
 * Prints the times; asserts only that one sweep of R rows costs no more than R sweeps of one. */
TEST(perf_multirow_logits_cost) {
    if (!have_avx2()) SKIP("no AVX2/FMA on this CPU");
    const int L = RouterPredictor::kLayers, n = RouterPredictor::kExperts, embd = RouterPredictor::kEmbd;
    std::vector<uint16_t> w((size_t)L * n * embd);
    Rng g(3);
    for (auto& v : w) v = bf16(g.next() * 0.05f);
    const Case rowsx(1, embd, 6, 9);   /* only its six inputs */
    std::vector<float> out((size_t)6 * n);
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto us = [](auto a, auto b) { return std::chrono::duration<double, std::micro>(b - a).count(); };
    double one_row = 0;
    for (int R : { 1, 2, 3, 4, 6 }) {
        double fused = 0, separate = 0;
        for (int rep = 0; rep < 3; ++rep) {
            auto t0 = now();
            for (int l = 0; l < L; ++l)
                rad::router_logits_rows(&w[(size_t)l * n * embd], n, embd, rowsx.xs.data(), R, true, out.data());
            auto t1 = now();
            for (int l = 0; l < L; ++l)
                for (int r = 0; r < R; ++r)
                    rad::router_logits_rows(&w[(size_t)l * n * embd], n, embd, rowsx.xs.data() + r, 1, true,
                                            out.data() + (size_t)r * n);
            auto t2 = now();
            if (rep) { fused += us(t0, t1) / L / 2; separate += us(t1, t2) / L / 2; }
        }
        if (R == 1) one_row = fused;
        std::printf("router logits, %d row%s a layer: one sweep %.1f us (%.2f one-row sweeps), %d one-row "
                    "sweeps %.1f us\n", R, R == 1 ? "" : "s", fused, one_row > 0 ? fused / one_row : 1.0, R,
                    separate);
        CHECK(fused <= separate * 1.10 + 5.0);
    }
}

TEST(multirow_passes_that_read_ahead) {
    /* Off: one-token passes only, of any kind, as always. */
    CHECK(rad::ra_pass_reads_ahead(true, 0, 1, 1));
    CHECK(rad::ra_pass_reads_ahead(false, 0, 1, 1));
    CHECK(!rad::ra_pass_reads_ahead(true, 0, 2, 1));
    /* On (N = 6): trunk decode passes of 2..6 rows too. */
    CHECK(rad::ra_pass_reads_ahead(true, 0, 1, 6));
    CHECK(rad::ra_pass_reads_ahead(true, 0, 2, 6));
    CHECK(rad::ra_pass_reads_ahead(true, 0, 6, 6));
    CHECK(!rad::ra_pass_reads_ahead(true, 0, 7, 6));
    CHECK(!rad::ra_pass_reads_ahead(false, 0, 2, 6));   /* a prefill chunk (or mixed) */
    CHECK(!rad::ra_pass_reads_ahead(true, 1, 5, 6));    /* the drafter's query pass */
    CHECK(!rad::ra_pass_reads_ahead(true, -1, 2, 6));   /* its context pass */
    CHECK(!rad::ra_pass_reads_ahead(true, 0, 0, 6));
}

RAD_TEST_MAIN()
