/* Actual selector + read-only observer against an independent lane-fold oracle. This binary is
 * host-only SIMT emulation, not run by the source-preparation agent. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include "ref_host.h"
#include "../../arch/common/rad_dflash_observe.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static RadTensor plane(void* p, uint32_t dtype, int rows, int width, int pitch) {
    RadTensor t{}; t.data = p; t.dtype = dtype; t.rank = 2;
    t.shape[0] = rows; t.shape[1] = width; t.stride[0] = pitch; t.stride[1] = 1;
    return t;
}
static float edge(const uint16_t* pred, const uint16_t* succ, const uint16_t* hp, int R) {
    float lanes[32]{};
    for (int l = 0; l < 32; ++l)
        for (int r = l; r < R; r += 32) {
            const float p = hp ? b2f(pred[r]) * b2f(hp[r]) : b2f(pred[r]);
            lanes[l] += p * b2f(succ[r]);
        }
    for (int mask = 16; mask; mask >>= 1) {
        float next[32];
        for (int l = 0; l < 32; ++l) next[l] = lanes[l] + lanes[l ^ mask];
        std::copy(next, next + 32, lanes);
    }
    return lanes[0];
}
static bool one(int steps, int K, int R, bool projected, bool ties, bool nonfinite) {
    using namespace rad_dflash_observe;
    constexpr int M = 2, V = 64, astr = 7, op = kWidth + 4;
    const int cp = K + 3, hpitch = R + 5, tp = steps + 3, rows = M * steps;
    std::vector<int32_t> cand((size_t)rows * cp, -3), anchor(M * astr, 63), tokens(M * tp, -7);
    std::vector<float> unary((size_t)rows * cp, -77), obs(M * op, -77);
    std::vector<uint16_t> pred((size_t)V * R), succ((size_t)V * R), hp((size_t)rows * hpitch);
    for (int s = 0; s < M; ++s) anchor[s * astr] = 2 + s;
    for (int t = 0; t < V; ++t)
        for (int r = 0; r < R; ++r) {
            pred[(size_t)t * R + r] = f2b(ties ? 0.f : .25f * (float)((t + r) % 5 - 2));
            succ[(size_t)t * R + r] = f2b(.25f * (float)((t * 3 + r) % 7 - 3));
        }
    for (int row = 0; row < rows; ++row) {
        for (int c = 0; c < K; ++c) {
            cand[(size_t)row * cp + c] = (row * 7 + c) % V;
            unary[(size_t)row * cp + c] = ties ? 0.f : .5f * (float)((row + c * 3) % 5);
        }
        for (int r = 0; r < hpitch; ++r) hp[(size_t)row * hpitch + r] = f2b(.5f * (1 + r % 3));
    }
    if (nonfinite) unary[0] = NAN;
    const auto old_cand = cand, old_anchor = anchor;
    const auto old_unary = unary;
    const auto old_pred = pred, old_succ = succ, old_hp = hp;
    RadTensor t[8] = { plane(cand.data(), RAD_I32, rows, K, cp), plane(unary.data(), RAD_F32, rows, K, cp),
                      {}, {}, plane(pred.data(), RAD_BF16, V, R, R), plane(succ.data(), RAD_BF16, V, R, R),
                      plane(tokens.data(), RAD_I32, M, steps, tp), plane(obs.data(), RAD_F32, M, kWidth, op) };
    if (projected) t[2] = plane(hp.data(), RAD_BF16, rows, R, hpitch);
    t[3].data = anchor.data(); t[3].dtype = RAD_I32; t[3].rank = 1;
    t[3].shape[0] = M * astr; t[3].stride[0] = 1;
    RadParam p[7]{};
    p[0] = { "M", RAD_P_INT, M }; p[1] = { "steps", RAD_P_INT, steps };
    p[2] = { "top_k", RAD_P_INT, K }; p[3] = { "rank", RAD_P_INT, R };
    p[4] = { "n_vocab", RAD_P_INT, V }; p[5] = { "anchor_stride", RAD_P_INT, astr };
    p[6].key = "dtype"; p[6].kind = RAD_P_STR; p[6].sval = "bf16";
    RadArgs a{}; a.t = t; a.n_t = 7; a.p = p; a.n_p = 7;
    if (r4d_rad_dflash_select(&a, nullptr) != RAD_OK) return false;
    const auto selected = tokens;
    a.n_t = 8;
    if (r4d_rad_dflash_observe(&a, nullptr) != RAD_OK || tokens != selected) return false;
    if (cand != old_cand || anchor != old_anchor || pred != old_pred || succ != old_succ || hp != old_hp ||
        std::memcmp(unary.data(), old_unary.data(), unary.size() * sizeof(float))) return false;
    for (int s = 0; s < M; ++s) {
        int pid = anchor[s * astr];
        for (int l = 0; l < steps; ++l) {
            const int row = s * steps + l;
            float score[32]; int best = 0; float bv = -INFINITY; bool finite = true;
            for (int c = 0; c < K; ++c) {
                const int id = cand[(size_t)row * cp + c];
                score[c] = edge(pred.data() + pid * R, succ.data() + id * R,
                                projected ? hp.data() + row * hpitch : nullptr, R) + unary[(size_t)row * cp + c];
                finite = finite && std::isfinite(score[c]);
                if (score[c] > bv) { bv = score[c]; best = c; }
            }
            const int id = cand[(size_t)row * cp + best];
            if (tokens[s * tp + l] != id) return false;
            if (l < kDepths) {
                const float* y = obs.data() + s * op + l * kFields;
                if (y[kToken] != id || y[kPredecessor] != pid) return false;
                if (!finite) {
                    if (!std::isnan(y[kCandidateP]) || !std::isnan(y[kScore]) || !std::isnan(y[kMargin])) return false;
                } else {
                    float den = 0.f, runner = -INFINITY;
                    for (int c = 0; c < K; ++c) {
                        den += std::exp(score[c] - bv);
                        if (c != best) runner = std::max(runner, score[c]);
                    }
                    if (y[kScore] != bv || y[kMargin] != bv - runner ||
                        std::fabs(y[kCandidateP] - 1.f / den) > 2e-6f) return false;
                }
            }
            pid = id;
        }
        for (int l = steps; l < kDepths; ++l) {
            const float* y = obs.data() + s * op + l * kFields;
            if (!std::isnan(y[kCandidateP]) || y[kToken] != -1.f) return false;
        }
        for (int j = kWidth; j < op; ++j) if (obs[s * op + j] != -77.f) return false;
    }
    /* A different path is not silently given the original winner's score. */
    tokens[0] = V + 1;
    if (r4d_rad_dflash_observe(&a, nullptr) != RAD_OK || !std::isnan(obs[0])) return false;
    tokens = selected;
    if (K > 1) {
        cand[1] = cand[0];
        if (r4d_rad_dflash_observe(&a, nullptr) != RAD_OK || !std::isnan(obs[0])) return false;
        cand = old_cand;
    }
    t[7].data = tokens.data();
    if (r4d_rad_dflash_observe(&a, nullptr) >= 0) return false;
    t[7].data = obs.data();
    t[1].stride[0] = cp + 1;
    if (r4d_rad_dflash_observe(&a, nullptr) != RAD_E_STRIDE) return false;
    t[1].stride[0] = cp;
    t[7].shape[1] = kWidth - 1;
    if (r4d_rad_dflash_observe(&a, nullptr) >= 0) return false;
    return true;
}
int main() {
    int failures = 0, cases = 0;
    for (int steps : { 1, 3, 5 })
        for (int K : { 1, 4, 16, 32 })
            for (bool projected : { false, true }) {
                const int R = K < 16 ? 32 : K == 16 ? 256 : 512;
                ++cases;
                if (!one(steps, K, R, projected, false, false)) ++failures;
            }
    ++cases; if (!one(3, 4, 32, false, true, false)) ++failures;
    ++cases; if (!one(3, 4, 32, false, false, true)) ++failures;
    std::printf("dprob: %d cases, %d failures\n", cases, failures);
    return failures != 0;
}
