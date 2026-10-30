/* tests/r4d_emu: router_scored_topk (r4d_ds41_elem.hip, the card's DSpark draft router) under
 * RADIANCE_EXPERIMENTAL_DS41_DRAFT_VRAM's mask (core/place/draft_vram.h) on the host SIMT emulator.
 *
 * The engine writes -inf into a draft stage's correction plane at every expert no rank holds in VRAM
 * (draft_mask_plane), and refuses a stage that keeps fewer than top_k. Here the kernel, at the draft
 * stages' shape (128 experts, top-3, sqrt(softplus) scores, norm, route scale 1.5, a query pass's 5
 * rows and taller), over seeded logits and masks built by the engine's own helpers from ranked
 * per-rank lists at 1 to 32 experts a rank:
 *   - never names a masked expert, and names three distinct kept ones;
 *   - names the three best kept experts by score + correction, in that order, wherever the third and
 *     fourth best are apart by more than the scores' float rounding (a serial model in double);
 *   - weights them by their unbiased scores renormalised over the three (sum = route_scale);
 *   - unmasked (an all-zero plane, as the bank ships), the same model holds, so the model is the
 *     kernel's;
 * and with two kept experts (what the engine refuses) its third pick is not a kept expert: the mask
 * cannot hold below top_k. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include "../../core/place/draft_vram.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace {

constexpr int NE = 128, K = 3;
constexpr float kScale = 1.5f;

double score_of(float x) { return std::sqrt(x > 20.0f ? (double)x : std::log1p(std::exp((double)x))); }

struct Launch {
    int rc = 0;
    std::vector<int> ids;
    std::vector<float> w;
};

Launch launch(const std::vector<float>& lg, int M, const std::vector<float>& bias) {
    const int lp = NE + 3, ip = K + 2, wp = K + 1;
    std::vector<float> L((size_t)M * lp, 0.0f);
    for (int r = 0; r < M; ++r)
        std::memcpy(&L[(size_t)r * lp], &lg[(size_t)r * NE], NE * sizeof(float));
    RadTensor t[4]{};
    t[0].data = L.data();
    t[0].dtype = RAD_F32;
    t[0].rank = 2;
    t[0].shape[0] = M;
    t[0].shape[1] = NE;
    t[0].stride[0] = lp;
    t[0].stride[1] = 1;
    t[1].data = const_cast<float*>(bias.data());
    t[1].dtype = RAD_F32;
    t[1].rank = 1;
    t[1].shape[0] = NE;
    t[1].stride[0] = 1;
    t[2].dtype = RAD_I32;
    t[2].rank = 2;
    t[2].shape[0] = M;
    t[2].shape[1] = K;
    t[2].stride[0] = ip;
    t[2].stride[1] = 1;
    t[3].dtype = RAD_F32;
    t[3].rank = 2;
    t[3].shape[0] = M;
    t[3].shape[1] = K;
    t[3].stride[0] = wp;
    t[3].stride[1] = 1;
    RadParam p[7]{};
    p[0] = { "M", RAD_P_INT, M };
    p[1] = { "n_expert", RAD_P_INT, NE };
    p[2] = { "top_k", RAD_P_INT, K };
    p[3] = { "norm", RAD_P_INT, 1 };
    p[4].key = "route_scale";
    p[4].kind = RAD_P_F64;
    p[4].dval = kScale;
    p[5].key = "score";
    p[5].kind = RAD_P_STR;
    p[5].sval = "sqrtsoftplus";
    p[6].key = "dtype";
    p[6].kind = RAD_P_STR;
    p[6].sval = "f32";
    RadArgs a{};
    a.t = t;
    a.n_t = 4;
    a.p = p;
    a.n_p = 7;
    std::vector<int> ids((size_t)M * ip, -77);
    std::vector<float> w((size_t)M * wp, -7.25f);
    t[2].data = ids.data();
    t[3].data = w.data();
    Launch out;
    out.rc = r4d_rad_router_scored_topk(&a, nullptr);
    out.ids.resize((size_t)M * K);
    out.w.resize((size_t)M * K);
    for (int r = 0; r < M; ++r)
        for (int j = 0; j < K; ++j) {
            out.ids[(size_t)r * K + j] = ids[(size_t)r * ip + j];
            out.w[(size_t)r * K + j] = w[(size_t)r * wp + j];
        }
    return out;
}

/* The rows the launch got wrong against the serial model, and the picks of a masked expert. */
struct Verdict {
    int rows = 0, masked_picks = 0, wrong_set = 0, wrong_order = 0, wrong_weights = 0, tied = 0;
};

Verdict judge(const Launch& l, const std::vector<float>& lg, int M, const std::vector<float>& bias) {
    Verdict v;
    for (int r = 0; r < M; ++r) {
        ++v.rows;
        const float* x = &lg[(size_t)r * NE];
        std::vector<double> score(NE), key(NE);
        std::vector<int> order;
        for (int e = 0; e < NE; ++e) {
            score[(size_t)e] = score_of(x[e]);
            key[(size_t)e] = score[(size_t)e] + (double)bias[(size_t)e];
            if (key[(size_t)e] > -INFINITY) order.push_back(e);
        }
        std::stable_sort(order.begin(), order.end(),
                         [&](int a, int b) { return key[(size_t)a] > key[(size_t)b]; });
        const int* got = &l.ids[(size_t)r * K];
        bool bad = false;
        for (int j = 0; j < K; ++j) {
            const int e = got[j];
            if (e < 0 || e >= NE || !(bias[(size_t)e] > -INFINITY)) { ++v.masked_picks; bad = true; }
            for (int i = 0; i < j; ++i) if (got[i] == e) bad = true;
        }
        if (bad) { ++v.wrong_set; continue; }
        /* The best three, in order, unless the scores' float rounding could reorder them. */
        const double eps = 1e-5;
        bool close = false;
        for (int j = 0; j < K && j + 1 < (int)order.size(); ++j)
            close = close || key[(size_t)order[(size_t)j]] - key[(size_t)order[(size_t)j + 1]] < eps;
        if (close) { ++v.tied; continue; }
        for (int j = 0; j < K; ++j)
            if (got[j] != order[(size_t)j]) { ++v.wrong_order; bad = true; break; }
        if (bad) continue;
        double sum = 0;
        for (int j = 0; j < K; ++j) sum += score[(size_t)got[j]];
        double wsum = 0;
        for (int j = 0; j < K; ++j) {
            const double want = score[(size_t)got[j]] * kScale / sum, w = l.w[(size_t)r * K + j];
            wsum += w;
            if (std::fabs(w - want) > 1e-5 * std::max(1.0, std::fabs(want))) bad = true;
        }
        if (bad || std::fabs(wsum - kScale) > 1e-5) ++v.wrong_weights;
    }
    return v;
}

}  // namespace

int main() {
    std::mt19937 rng(1700);
    std::normal_distribution<float> nd(0.f, 2.f);
    int fails = 0, cases = 0, rows = 0, tied = 0;
    const int heights[] = { 5, 15, 64 };
    /* Experts a stage a rank: 1 (the fewest the engine keeps, 4 a stage), 8, 16, 24, 32 (every one). */
    const int keeps[] = { 1, 8, 16, 24, 32 };
    for (int M : heights)
        for (int n : keeps)
            for (int plane = 0; plane < 2; ++plane) {
                /* A hot set as draft_hotset.py writes it: each rank's 32 experts in a random order. */
                rad::DraftHotset hs;
                hs.hot.assign(3, std::vector<std::vector<int32_t>>(4));
                for (int s = 0; s < 3; ++s)
                    for (int r = 0; r < 4; ++r) {
                        std::vector<int32_t> l(32);
                        std::iota(l.begin(), l.end(), 32 * r);
                        std::shuffle(l.begin(), l.end(), rng);
                        hs.hot[(size_t)s][(size_t)r] = l;
                    }
                std::vector<std::vector<uint8_t>> allowed;
                std::string why;
                if (rad::draft_vram_allowed(hs, n, &allowed, &why) != RAD_OK) {
                    std::printf("FAIL draft_vram_allowed at n=%d: %s\n", n, why.c_str());
                    ++fails;
                    continue;
                }
                /* The bank's planes are zero; a nonzero one checks the kept values are kept. */
                std::vector<float> orig(NE, 0.0f), bias(NE);
                if (plane)
                    for (auto& b : orig) b = nd(rng) * 0.05f;
                for (int s = 0; s < 3; ++s) {
                    rad::draft_mask_plane(orig.data(), allowed[(size_t)s].data(), NE, bias.data());
                    std::vector<float> lg((size_t)M * NE);
                    for (auto& v : lg) v = nd(rng);
                    if (s == 2) for (int i = 0; i < M; ++i) lg[(size_t)i * NE + (size_t)(i % NE)] = 25.0f;  /* softplus' x itself */
                    const Launch l = launch(lg, M, bias);
                    ++cases;
                    if (l.rc != RAD_OK) {
                        std::printf("FAIL router_scored_topk launch M=%d n=%d rc %d\n", M, n, l.rc);
                        ++fails;
                        continue;
                    }
                    const Verdict v = judge(l, lg, M, bias);
                    rows += v.rows;
                    tied += v.tied;
                    if (v.masked_picks || v.wrong_set || v.wrong_order || v.wrong_weights) {
                        std::printf("FAIL masked router M=%d keep %d a rank stage %d%s: %d masked picks, %d "
                                    "rows with a wrong set, %d out of order, %d wrong weights\n",
                                    M, n, 40 + s, plane ? " (nonzero plane)" : "", v.masked_picks,
                                    v.wrong_set, v.wrong_order, v.wrong_weights);
                        ++fails;
                    }
                }
            }
    /* Unmasked: the bank's all-zero plane; the model must be the kernel's. */
    {
        const int M = 64;
        std::vector<float> lg((size_t)M * NE), bias(NE, 0.0f);
        for (auto& v : lg) v = nd(rng);
        const Launch l = launch(lg, M, bias);
        const Verdict v = judge(l, lg, M, bias);
        ++cases;
        rows += v.rows;
        tied += v.tied;
        if (l.rc != RAD_OK || v.masked_picks || v.wrong_set || v.wrong_order || v.wrong_weights) {
            std::printf("FAIL unmasked router against the model: rc %d, %d wrong sets, %d out of order, %d "
                        "wrong weights\n", l.rc, v.wrong_set, v.wrong_order, v.wrong_weights);
            ++fails;
        }
    }
    /* Two kept experts (the engine refuses a stage below top_k): the third pick cannot be a kept one. */
    {
        const int M = 5;
        std::vector<float> lg((size_t)M * NE), bias(NE, -INFINITY);
        for (auto& v : lg) v = nd(rng);
        bias[37] = 0.0f;
        bias[90] = 0.0f;
        const Launch l = launch(lg, M, bias);
        int third_kept = 0;
        for (int r = 0; r < M; ++r) {
            const int e = l.ids[(size_t)r * K + 2];
            third_kept += e == 37 || e == 90;
        }
        ++cases;
        if (l.rc != RAD_OK || third_kept) {
            std::printf("FAIL two kept experts: rc %d, %d rows took a kept third\n", l.rc, third_kept);
            ++fails;
        } else {
            std::printf("  two kept experts: the third pick of every row is expert %d, which is masked: "
                        "the engine's refusal below top-3 is needed\n", l.ids[2]);
        }
    }
    std::printf("%s: router_scored_topk under the draft mask, %d cases, %d rows (%d with a near tie "
                "checked for the set only), %d failed\n",
                fails ? "FAIL" : "PASS", cases, rows, tied, fails);
    return fails ? 1 : 0;
}
