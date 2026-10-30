/* router_predictor_test.cpp -- the router predictor's quantized scoring
 * (RADIANCE_EXPERIMENTAL_DS41_PREFETCH_SCORE, core/place/router_score.h, router_predictor.h): the
 * integer dots equal their scalar references bit for bit, at the int32 lanes' bound too; quantized
 * rows reconstruct within half a step (int4: the clip that wins has no more squared error than
 * max|w|'s); the input's int16 copy; and a predictor of each kind on 40 synthetic routers -- the
 * quantized ones give bf16's own picks whenever bf16's are among the rescored, the worker hands
 * over what predict() computes, and the check counts what the test counts. */
#include "rad_test.h"

#include "place/router_predictor.h"
#include "place/router_score.h"
#include "rad_core.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using rad::RouterPredictor;
namespace rs = rad::rscore;

namespace {

/* A deterministic stream: xorshift64*, uniform in [0, 1), and a rough normal (Irwin-Hall of 4). */
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
    uint64_t next() { s ^= s >> 12; s ^= s << 25; s ^= s >> 27; return s * 0x2545F4914F6CDD1Dull; }
    float uni() { return (float)(next() >> 40) / (float)(1 << 24); }
    float gauss() { return (uni() + uni() + uni() + uni() - 2.0f) * 1.7320508f; }
};

bool avx2() { return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"); }

constexpr int L = RouterPredictor::kLayers, E = RouterPredictor::kExperts, D = RouterPredictor::kEmbd;

/* Layer l's synthetic router [E, D] (f32, bf16-valued as the container's), its correction [E] and
 * ffn_norm gain [D]: experts of different norms, a few strong channels in the gain. */
void make_router(int l, float* w) {
    Rng r(1000 + (uint64_t)l);
    for (int e = 0; e < E; ++e) {
        const float g = 0.015f + 0.01f * r.uni();
        for (int k = 0; k < D; ++k) w[(size_t)e * D + k] = rs::bf16_to_f32(rs::f32_to_bf16_rne(g * r.gauss()));
    }
}
void make_corr(int l, float* c) {
    Rng r(2000 + (uint64_t)l);
    for (int e = 0; e < E; ++e) c[e] = 0.05f * (r.uni() - 0.5f);
}
void make_gain(int l, float* g) {
    Rng r(3000 + (uint64_t)l);
    for (int k = 0; k < D; ++k) g[k] = k % 509 == 7 ? 3.0f + r.uni() : 0.5f + r.uni();
}

/* A Program holding the three weights a layer as the container does (f32, read whole), and the
 * read function the predictor's load takes. */
struct Routers {
    rad::Program prog;
    Routers() {
        prog.weights.emplace_back();   /* the sentinel */
        auto add = [&](const std::string& name, int64_t bytes) {
            rad::WeightInfo w;
            w.name = name;
            w.identity = true;
            w.stored_bytes = bytes;
            prog.weights.push_back(w);
        };
        for (int l = 0; l < L; ++l) {
            const std::string b = "blk." + std::to_string(l) + ".";
            add(b + "glm_router.weight", (int64_t)E * D * 4);
            add(b + "glm_router.correction", (int64_t)E * 4);
            add(b + "ffn_norm.weight", (int64_t)D * 4);
        }
    }
    static bool read(const rad::WeightInfo& w, void* dst) {
        int l = -1;
        char what[64] = {};
        if (std::sscanf(w.name.c_str(), "blk.%d.%63s", &l, what) != 2) return false;
        if (!std::strcmp(what, "glm_router.weight")) make_router(l, (float*)dst);
        else if (!std::strcmp(what, "glm_router.correction")) make_corr(l, (float*)dst);
        else if (!std::strcmp(what, "ffn_norm.weight")) make_gain(l, (float*)dst);
        else return false;
        return true;
    }
};

std::unique_ptr<RouterPredictor> loaded(RouterPredictor::Score score, int check, const Routers& r) {
    RouterPredictor::Options o;
    o.pieces = true;
    o.shadow = false;
    o.k = 4;
    o.ahead = 2;
    o.score = score;
    o.rescore = 12;
    o.check = check;
    std::unique_ptr<RouterPredictor> p = RouterPredictor::create(o);
    if (!p->load(r.prog, &Routers::read)) return nullptr;
    return p;
}

/* Inputs as the predictor sees them at layer l (bf16 router input, a few large channels), and the
 * stand-in it scores layer l + 2 with: x * g_{l+2} / g_l. */
std::vector<uint16_t> input(int i) {
    Rng r(5000 + (uint64_t)i);
    std::vector<uint16_t> x(D);
    for (int k = 0; k < D; ++k) {
        float v = r.gauss();
        if (k % 1021 == 3) v *= 20.0f;
        x[(size_t)k] = rs::f32_to_bf16_rne(v);
    }
    return x;
}
std::vector<float> standin(const std::vector<uint16_t>& x, int l) {
    std::vector<float> g0(D), g2(D), xh(D);
    make_gain(l, g0.data());
    make_gain(l + 2, g2.data());
    for (int k = 0; k < D; ++k) {
        const float a = g0[(size_t)k], ratio = a != 0.0f ? g2[(size_t)k] / a : 0.0f;
        xh[(size_t)k] = rs::bf16_to_f32(x[(size_t)k]) * ratio;
    }
    return xh;
}

bool same_set(const int32_t* a, const int32_t* b, int k) {
    for (int i = 0; i < k; ++i)
        if (std::find(b, b + k, a[i]) == b + k) return false;
    return true;
}

}  /* namespace */

TEST(router_score_integer_dots_equal_their_references) {
    if (!avx2()) SKIP("no AVX2");
    Rng r(7);
    for (int n : { 64, 128, 1024, 5120, 5184, rs::kMaxDotElems }) {
        std::vector<int8_t> q8((size_t)n);
        std::vector<uint8_t> q4((size_t)n / 2);
        std::vector<int16_t> x((size_t)n);
        for (int t = 0; t < 4; ++t) {
            for (int k = 0; k < n; ++k) {
                q8[(size_t)k] = (int8_t)((int)(r.next() % 255) - 127);
                x[(size_t)k] = (int16_t)((int)(r.next() % 65535) - 32767);
            }
            for (auto& b : q4) b = (uint8_t)r.next();
            CHECK_EQ(rs::dot_i8_avx2(q8.data(), x.data(), n), rs::dot_i8_ref(q8.data(), x.data(), n));
            CHECK_EQ(rs::dot_u4_avx2(q4.data(), x.data(), n), rs::dot_u4_ref(q4.data(), x.data(), n));
        }
        /* The bound: every product at its largest, of one sign and of the other. */
        for (int sgn : { 1, -1 }) {
            std::fill(q8.begin(), q8.end(), (int8_t)(127 * sgn));
            std::fill(q4.begin(), q4.end(), (uint8_t)0xff);
            std::fill(x.begin(), x.end(), (int16_t)(32767 * sgn));
            const int64_t want8 = (int64_t)n * 127 * 32767, want4 = (int64_t)n * 15 * 32767 * sgn;
            CHECK_EQ(rs::dot_i8_avx2(q8.data(), x.data(), n), want8);
            CHECK_EQ(rs::dot_i8_ref(q8.data(), x.data(), n), want8);
            CHECK_EQ(rs::dot_u4_avx2(q4.data(), x.data(), n), want4);
        }
    }
}

TEST(router_score_quantized_rows_reconstruct_within_half_a_step) {
    Rng r(11);
    std::vector<float> w(D);
    std::vector<int8_t> q8(D);
    std::vector<uint8_t> q4(D / 2);
    for (int t = 0; t < 32; ++t) {
        const float g = 0.01f + 0.02f * r.uni();
        for (int k = 0; k < D; ++k) w[(size_t)k] = g * r.gauss() * (k % 701 == 5 ? 6.0f : 1.0f);
        float amax = 0;
        for (float v : w) amax = std::max(amax, std::fabs(v));
        /* int8: max|w| is +-127 steps; every weight within half a step */
        const float s8 = rs::quantize_row_i8(w.data(), D, q8.data());
        CHECK_NEAR(s8, amax / 127.0f, 1e-9);
        float worst8 = 0;
        for (int k = 0; k < D; ++k) worst8 = std::max(worst8, std::fabs(w[(size_t)k] - s8 * q8[(size_t)k]));
        CHECK(worst8 <= 0.5f * s8 * 1.0001f);
        /* int4: within half a step where not clipped; no more squared error than max|w|'s scale */
        const float s4 = rs::quantize_row_u4(w.data(), D, q4.data());
        CHECK(s4 > 0.0f && s4 <= amax / 7.0f * 1.0001f);
        double err = 0, err_abs = 0;
        const float sa = amax / 7.0f;
        for (int k = 0; k < D; ++k) {
            const float v = w[(size_t)k], back = s4 * rs::u4_at(q4.data(), k);
            if (v <= 7.0f * s4 && v >= -8.0f * s4) CHECK(std::fabs(v - back) <= 0.5f * s4 * 1.0001f);
            err += (double)(v - back) * (v - back);
            const float qa = (float)std::clamp(std::lrint(v / sa), -8L, 7L);
            err_abs += (double)(v - qa * sa) * (v - qa * sa);
        }
        CHECK(err <= err_abs * 1.0001);
    }
    /* A row of zeros: scale 0, every value 0 */
    std::fill(w.begin(), w.end(), 0.0f);
    CHECK_EQ(rs::quantize_row_i8(w.data(), D, q8.data()), 0.0f);
    CHECK_EQ(rs::quantize_row_u4(w.data(), D, q4.data()), 0.0f);
    CHECK_EQ(rs::u4_at(q4.data(), 17), 0);
}

TEST(router_score_int4_clip_error_avx2_matches_the_reference) {
    if (!avx2()) SKIP("no AVX2");
    Rng r(19);
    std::vector<float> w(D);
    for (int t = 0; t < 16; ++t) {
        for (int k = 0; k < D; ++k) w[(size_t)k] = 0.02f * r.gauss() * (k % 997 == 3 ? 8.0f : 1.0f);
        float amax = 0;
        for (float v : w) amax = std::max(amax, std::fabs(v));
        for (float c : rs::kU4Clip) {
            const float s = amax * c / 7.0f;
            const double a = rs::u4_sq_err_avx2(w.data(), D, s), b = rs::u4_sq_err_ref(w.data(), D, s);
            CHECK_NEAR(a, b, 1e-4 * b + 1e-12);   /* the same roundings; f32 lanes sum in another order */
        }
    }
}

TEST(router_score_the_inputs_int16_copy) {
    std::vector<float> x(D);
    Rng r(13);
    for (auto& v : x) v = r.gauss();
    x[100] = -9.5f;   /* the largest magnitude */
    std::vector<int16_t> xq(D);
    int64_t sum = 0;
    const float sx = rs::quantize_x16(x.data(), D, xq.data(), &sum);
    CHECK_NEAR(sx, 9.5f / 32767.0f, 1e-9);
    CHECK_EQ((int)xq[100], -32767);
    int64_t s = 0;
    float worst = 0;
    for (int k = 0; k < D; ++k) {
        s += xq[(size_t)k];
        worst = std::max(worst, std::fabs(x[(size_t)k] - sx * xq[(size_t)k]));
    }
    CHECK_EQ(sum, s);
    CHECK(worst <= 0.5f * sx * 1.0001f);
    /* Not finite counts as 0; all zeros give scale 0 */
    x[5] = NAN;
    x[6] = INFINITY;
    rs::quantize_x16(x.data(), D, xq.data(), &sum);
    CHECK_EQ((int)xq[5], 0);
    CHECK_EQ((int)xq[6], 0);
    std::fill(x.begin(), x.end(), 0.0f);
    CHECK_EQ(rs::quantize_x16(x.data(), D, xq.data(), &sum), 0.0f);
    CHECK_EQ(sum, (int64_t)0);
}

/* quantize_x16 (the AVX2 form where it applies) gives quantize_x16_ref's values, scale and sum, bit for
 * bit: gaussian rows with outliers, values at the rounding midpoints, not-finite lanes, all zeros, the
 * largest magnitude negative, a row too short or not a multiple of 16 (the reference's own path). */
TEST(router_score_the_inputs_int16_copy_by_avx2_is_the_references) {
    if (!__builtin_cpu_supports("avx2")) { SKIP("no AVX2"); return; }
    Rng r(29);
    for (int n : { D, 16, 4096 + 16, 65536, D + 8, 7 }) {
        std::vector<float> x((size_t)n);
        for (int t = 0; t < 12; ++t) {
            for (auto& v : x) v = r.gauss();
            if (t % 3 == 1) x[(size_t)(t % n)] = 300.0f * (t & 4 ? -1.0f : 1.0f);   /* an outlier channel */
            if (t % 4 == 2 && n >= 16) { x[3] = NAN; x[9] = INFINITY; x[n - 1] = -INFINITY; }
            if (t == 5) {   /* every value at a rounding midpoint of the int16 grid */
                const float amax = 32767.0f;
                for (int k = 0; k < n; ++k) x[(size_t)k] = (k == 0 ? -amax : (float)(k % 64) + 0.5f);
            }
            if (t == 7) std::fill(x.begin(), x.end(), 0.0f);
            if (t == 8) { std::fill(x.begin(), x.end(), NAN); }
            std::vector<int16_t> a((size_t)n, 7), b((size_t)n, 9);
            int64_t sa = -1, sb = -2;
            const float fa = rs::quantize_x16_ref(x.data(), n, a.data(), &sa);
            const float fb = rs::quantize_x16(x.data(), n, b.data(), &sb);
            CHECK(a == b);
            CHECK_EQ(sa, sb);
            CHECK_EQ(std::memcmp(&fa, &fb, sizeof fa), 0);
        }
    }
}

TEST(router_score_quantized_logits_are_the_dequantized_dot) {
    /* s sx (sum q xq) and s sx (sum n xq - 8 sum xq) are the dot of the dequantized row and input,
     * and that lands near the f32 dot: int8 within 5% of the logits' spread (|w| rms(x)), int4
     * within 60% (one scale a row, 16 levels; the predictor rescores its best 12 in bf16). */
    Rng r(17);
    std::vector<float> w(D), x(D);
    std::vector<int8_t> q8(D);
    std::vector<uint8_t> q4(D / 2);
    std::vector<int16_t> xq(D);
    for (int t = 0; t < 16; ++t) {
        for (int k = 0; k < D; ++k) { w[(size_t)k] = 0.02f * r.gauss(); x[(size_t)k] = r.gauss(); }
        double ref = 0, wn = 0, xn = 0;
        for (int k = 0; k < D; ++k) {
            ref += (double)w[(size_t)k] * x[(size_t)k];
            wn += (double)w[(size_t)k] * w[(size_t)k];
            xn += (double)x[(size_t)k] * x[(size_t)k];
        }
        const double spread = std::sqrt(wn) * std::sqrt(xn / D);
        int64_t sum = 0;
        const float sx = rs::quantize_x16(x.data(), D, xq.data(), &sum);
        const float s8 = rs::quantize_row_i8(w.data(), D, q8.data());
        const float s4 = rs::quantize_row_u4(w.data(), D, q4.data());
        const double l8 = (double)s8 * sx * (double)rs::dot_i8_ref(q8.data(), xq.data(), D);
        const double l4 = (double)s4 * sx * (double)(rs::dot_u4_ref(q4.data(), xq.data(), D) - 8 * sum);
        double d8 = 0, d4 = 0;
        for (int k = 0; k < D; ++k) {
            const double xv = (double)sx * xq[(size_t)k];
            d8 += (double)s8 * q8[(size_t)k] * xv;
            d4 += (double)s4 * rs::u4_at(q4.data(), k) * xv;
        }
        CHECK_NEAR(l8, d8, 1e-9 * spread);
        CHECK_NEAR(l4, d4, 1e-9 * spread);
        CHECK(std::fabs(l8 - ref) <= 0.05 * spread);
        CHECK(std::fabs(l4 - ref) <= 0.60 * spread);
    }
}

TEST(router_predictor_quantized_picks_are_bf16s_when_among_the_rescored) {
    if (!avx2()) SKIP("no AVX2");
    const Routers r;
    constexpr int kInputs = 48;
    std::vector<std::vector<uint16_t>> xs;
    for (int i = 0; i < kInputs; ++i) xs.push_back(input(i));
    auto layer_of = [](int i) { return (i * 7) % (L - 2); };   /* predicting layer_of + 2 */
    /* bf16's own picks, from a bf16 predictor */
    std::vector<std::array<int32_t, 4>> ref(kInputs);
    {
        std::unique_ptr<RouterPredictor> p = loaded(RouterPredictor::kScoreBf16, 0, r);
        REQUIRE(p != nullptr);
        CHECK_EQ(p->quant_bytes(), (int64_t)0);
        for (int i = 0; i < kInputs; ++i) {
            const std::vector<float> xh = standin(xs[(size_t)i], layer_of(i));
            p->predict(layer_of(i) + 2, xh.data(), ref[(size_t)i].data());
        }
    }
    for (RouterPredictor::Score sc : { RouterPredictor::kScoreInt8, RouterPredictor::kScoreInt4 }) {
        std::unique_ptr<RouterPredictor> p = loaded(sc, 1, r);
        REQUIRE(p != nullptr);
        CHECK_EQ(p->quant_bytes(), (int64_t)L * E * D / (sc == RouterPredictor::kScoreInt8 ? 1 : 2));
        CHECK_EQ(p->rescore(), 12);
        /* The worker: each posted input's prediction handed over, as predict() gives it. */
        struct Got { std::mutex mu; std::condition_variable cv; std::map<int, std::array<int32_t, 4>> by_pass; };
        Got got;
        p->on_ready([](void* ctx, uint32_t pass, int, const int32_t* e, int k) {
            Got* g = static_cast<Got*>(ctx);
            std::array<int32_t, 4> a{};
            std::copy(e, e + std::min(k, 4), a.begin());
            std::lock_guard<std::mutex> lk(g->mu);
            g->by_pass[(int)pass] = a;
            g->cv.notify_all();
        }, &got);
        int same = 0, same_direct = 0;
        for (int i = 0; i < kInputs; ++i) {
            const std::vector<float> xh = standin(xs[(size_t)i], layer_of(i));
            std::array<int32_t, 4> direct{};
            p->predict(layer_of(i) + 2, xh.data(), direct.data());
            same_direct += same_set(direct.data(), ref[(size_t)i].data(), 4);
            p->post((uint32_t)i + 1, layer_of(i), xs[(size_t)i].data());
            std::unique_lock<std::mutex> lk(got.mu);
            CHECK(got.cv.wait_for(lk, std::chrono::seconds(20), [&] { return got.by_pass.count(i + 1) != 0; }));
            CHECK(got.by_pass[i + 1] == direct);
            same += same_set(got.by_pass[i + 1].data(), ref[(size_t)i].data(), 4);
        }
        p->on_ready(nullptr, nullptr);
        /* The check runs after each hand-over; wait for the last one. */
        RouterPredictor::Stats s = p->stats();
        for (int t = 0; t < 2000 && s.checked < (uint64_t)kInputs; ++t) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            s = p->stats();
        }
        CHECK_EQ(s.computed, (uint64_t)kInputs);
        CHECK_EQ(s.checked, (uint64_t)kInputs);
        CHECK_EQ(s.check_set, (uint64_t)same);
        CHECK_EQ(same, same_direct);
        uint64_t picks = 0;
        for (uint64_t b : s.check_rank) picks += b;
        CHECK_EQ(picks, (uint64_t)kInputs * 4);
        CHECK(s.check_order <= s.check_set);
        CHECK(s.quant_ns > 0 && s.rescore_ns > 0);
        /* A pick of bf16's outside the rescored is the only way to differ. */
        CHECK(s.check_outside >= (uint64_t)(kInputs - same));
        if (sc == RouterPredictor::kScoreInt8) CHECK_EQ(same, kInputs);
        else CHECK(same >= kInputs * 9 / 10);
        std::printf("  %s: %d of %d predictions bf16's set, %llu in its order; bf16's picks at or past "
                    "the 12 rescored %llu\n", sc == RouterPredictor::kScoreInt8 ? "int8" : "int4", same,
                    kInputs, (unsigned long long)s.check_order, (unsigned long long)s.check_outside);
    }
}

TEST(router_predictor_options_are_clamped) {
    RouterPredictor::Options o;
    o.k = 4;
    o.rescore = 2;    /* below k: k */
    o.check = 5;      /* bf16: nothing to check */
    std::unique_ptr<RouterPredictor> a = RouterPredictor::create(o);
    CHECK_EQ(a->rescore(), 4);
    CHECK_EQ(a->check_every(), 0);
    CHECK_EQ((int)a->score_mode(), (int)RouterPredictor::kScoreBf16);
    o.score = RouterPredictor::kScoreInt4;
    o.rescore = 1000;   /* at most kMaxRescore */
    std::unique_ptr<RouterPredictor> b = RouterPredictor::create(o);
    CHECK_EQ(b->rescore(), RouterPredictor::kMaxRescore);
    CHECK_EQ(b->check_every(), 5);
    CHECK(!b->loaded());
    int32_t e[4] = { -1, -1, -1, -1 };
    std::vector<float> xh(D, 1.0f);
    b->predict(2, xh.data(), e);   /* not loaded: nothing */
    CHECK_EQ(e[0], -1);
}

RAD_TEST_MAIN()
