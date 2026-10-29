/* router_predictor.cpp -- see router_predictor.h. */
#include "router_predictor.h"

#include "../rad_internal.h"
#include "../rad_core.h"

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace rad {

namespace {

inline uint16_t f32_to_bf16_rne(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fff + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}

inline float bf16_to_f32(uint16_t h) {
    const uint32_t u = (uint32_t)h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

/* dot(w bf16, x f32) for n a multiple of 8; FMA is fine here, the result only ranks experts. */
__attribute__((target("avx2,fma"))) float dot_avx2(const uint16_t* w, const float* x, int n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    int k = 0;
    for (; k + 16 <= n; k += 16) {
        const __m256 w0 = _mm256_castsi256_ps(_mm256_slli_epi32(
            _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(w + k))), 16));
        const __m256 w1 = _mm256_castsi256_ps(_mm256_slli_epi32(
            _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(w + k + 8))), 16));
        a0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x + k), a0);
        a1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x + k + 8), a1);
    }
    for (; k + 8 <= n; k += 8) {
        const __m256 w0 = _mm256_castsi256_ps(_mm256_slli_epi32(
            _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(w + k))), 16));
        a0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x + k), a0);
    }
    a0 = _mm256_add_ps(a0, a1);
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps(a0, 1));
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

float dot_ref(const uint16_t* w, const float* x, int n) {
    float a = 0;
    for (int k = 0; k < n; ++k) a += bf16_to_f32(w[k]) * x[k];
    return a;
}

}  /* namespace */

RouterPredictor* RouterPredictor::get() {
    static RouterPredictor* p = []() -> RouterPredictor* {
        const char* m = std::getenv("RADIANCE_EXPERIMENTAL_DS41_PREFETCH");
        if (!m || !*m || !std::strcmp(m, "0") || !std::strcmp(m, "off")) return nullptr;
        auto* r = new RouterPredictor();
        r->shadow_ = std::strcmp(m, "on") != 0;
        if (const char* k = std::getenv("RADIANCE_EXPERIMENTAL_DS41_PREFETCH_K"); k && *k)
            r->k_ = std::clamp(std::atoi(k), 1, (int)kMaxK);
        if (const char* a = std::getenv("RADIANCE_EXPERIMENTAL_DS41_PREFETCH_AHEAD"); a && *a)
            r->ahead_ = std::atoi(a) == 2 ? 2 : 1;
        r->avx2_ = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
        r->worker_ = std::thread(&RouterPredictor::loop, r);
        return r;
    }();
    return p;
}

RouterPredictor::~RouterPredictor() {
    {
        std::lock_guard<std::mutex> lk(q_mu_);
        stop_ = true;
    }
    q_cv_.notify_all();
    if (worker_.joinable()) worker_.join();
}

bool RouterPredictor::load(const Program& prog,
                           const std::function<bool(const WeightInfo&, void*)>& read) {
    std::call_once(load_once_, [&] {
        auto find = [&](const std::string& name) -> const WeightInfo* {
            for (const WeightInfo& w : prog.weights)
                if (w.name == name) return &w;
            return nullptr;
        };
        w_.assign((size_t)kLayers * kExperts * kEmbd, 0);
        corr_.assign((size_t)kLayers * kExperts, 0.0f);
        ratio_.assign((size_t)kLayers * kEmbd, 1.0f);
        ratio2_.assign((size_t)kLayers * kEmbd, 1.0f);
        std::vector<float> wf((size_t)kExperts * kEmbd), gain((size_t)kLayers * kEmbd);
        for (int l = 0; l < kLayers; ++l) {
            const std::string b = "blk." + std::to_string(l) + ".";
            const WeightInfo* wr = find(b + "glm_router.weight");
            const WeightInfo* wc = find(b + "glm_router.correction");
            const WeightInfo* wg = find(b + "ffn_norm.weight");
            if (!wr || !wc || !wg || wr->stored_bytes != (int64_t)kExperts * kEmbd * 4 ||
                wc->stored_bytes != kExperts * 4 || wg->stored_bytes != kEmbd * 4 ||
                !read(*wr, wf.data()) || !read(*wc, &corr_[(size_t)l * kExperts]) ||
                !read(*wg, &gain[(size_t)l * kEmbd])) {
                RAD_WARN("router predictor: layer %d's router or ffn_norm is not f32 [384, 5120] / "
                         "[384] / [5120] in the container; no prediction", l);
                return;
            }
            uint16_t* dst = &w_[(size_t)l * kExperts * kEmbd];
            for (size_t i = 0; i < wf.size(); ++i) dst[i] = f32_to_bf16_rne(wf[i]);
        }
        for (int l = 0; l + 1 < kLayers; ++l)
            for (int k = 0; k < kEmbd; ++k) {
                const float a = gain[(size_t)l * kEmbd + k], b = gain[(size_t)(l + 1) * kEmbd + k];
                ratio_[(size_t)l * kEmbd + k] = a != 0.0f ? b / a : 0.0f;
                if (l + 2 < kLayers) {
                    const float c = gain[(size_t)(l + 2) * kEmbd + k];
                    ratio2_[(size_t)l * kEmbd + k] = a != 0.0f ? c / a : 0.0f;
                }
            }
        load_ok_ = true;
        loaded_.store(true, std::memory_order_release);
        RAD_INFO("router predictor: %d routers loaded (bf16, %.0f MB); layer L's router input "
                 "predicts layer L+1's top %d (%s)", kLayers, w_.size() * 2 / 1e6, k_,
                 shadow_ ? "shadow: compared, nothing read" : "predicted misses are read ahead");
    });
    return load_ok_;
}

void RouterPredictor::post(uint32_t pass, int layer, const uint16_t* x) {
    if (!loaded() || layer < 0 || layer + 1 >= kLayers || !x) return;
    posted_.fetch_add(1, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(q_mu_);
        if (q_.size() >= 4) {
            q_.pop_front();
            dropped_.fetch_add(1, std::memory_order_relaxed);
        }
        Req r;
        r.pass = pass;
        r.layer = layer;
        r.x.assign(x, x + kEmbd);
        q_.push_back(std::move(r));
    }
    q_cv_.notify_one();
}

void RouterPredictor::loop() {
    std::unique_lock<std::mutex> lk(q_mu_);
    for (;;) {
        q_cv_.wait(lk, [&] { return stop_ || !q_.empty(); });
        if (stop_) return;
        Req r = std::move(q_.front());
        q_.pop_front();
        lk.unlock();
        compute(r.pass, r.layer, r.x.data());
        lk.lock();
    }
}

/* Layer `target`'s router on the stand-in input xh: its best k into *out. */
void RouterPredictor::score(int target, const float* xh, Result* out, uint32_t pass) {
    const uint16_t* w = &w_[(size_t)target * kExperts * kEmbd];
    const float* corr = &corr_[(size_t)target * kExperts];
    std::pair<float, int32_t> key[kExperts];
    for (int e = 0; e < kExperts; ++e) {
        const float logit = avx2_ ? dot_avx2(w + (size_t)e * kEmbd, xh, kEmbd)
                                  : dot_ref(w + (size_t)e * kEmbd, xh, kEmbd);
        /* sqrtsoftplus, as the router scores; the correction only selects. */
        const float sp = logit > 0 ? logit + std::log1p(std::exp(-logit)) : std::log1p(std::exp(logit));
        key[e] = { std::sqrt(sp) + corr[e], e };
    }
    std::partial_sort(key, key + k_, key + kExperts,
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    std::lock_guard<std::mutex> lk(res_mu_);
    out->pass = pass;
    out->valid = true;
    for (int i = 0; i < k_; ++i) out->e[i] = key[i].second;
}

void RouterPredictor::compute(uint32_t pass, int layer, const uint16_t* x) {
    const auto t0 = std::chrono::steady_clock::now();
    float xh[kEmbd];
    const float* ratio = &ratio_[(size_t)layer * kEmbd];
    for (int k = 0; k < kEmbd; ++k) xh[k] = bf16_to_f32(x[k]) * ratio[k];
    score(layer + 1, xh, &res_[layer + 1], pass);
    /* Two layers ahead from the same input: in shadow mode for the statistics, and for a read-ahead
     * two layers on. */
    if ((shadow_ || ahead_ == 2) && layer + 2 < kLayers) {
        const float* ratio2 = &ratio2_[(size_t)layer * kEmbd];
        for (int k = 0; k < kEmbd; ++k) xh[k] = bf16_to_f32(x[k]) * ratio2[k];
        score(layer + 2, xh, &res2_[layer + 2], pass);
    }
    computed_.fetch_add(1, std::memory_order_relaxed);
    ns_.fetch_add((uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now() - t0).count(), std::memory_order_relaxed);
}

bool RouterPredictor::predicted(uint32_t pass, int layer, int32_t* experts) const {
    if (layer < 0 || layer >= kLayers) return false;
    std::lock_guard<std::mutex> lk(res_mu_);
    const Result& r = res_[layer];
    if (!r.valid || r.pass != pass) return false;
    std::memcpy(experts, r.e, sizeof(int32_t) * (size_t)k_);
    return true;
}

bool RouterPredictor::predicted2(uint32_t pass, int layer, int32_t* experts) const {
    if (layer < 0 || layer >= kLayers) return false;
    std::lock_guard<std::mutex> lk(res_mu_);
    const Result& r = res2_[layer];
    if (!r.valid || r.pass != pass) return false;
    std::memcpy(experts, r.e, sizeof(int32_t) * (size_t)k_);
    return true;
}

RouterPredictor::Stats RouterPredictor::stats() const {
    Stats s;
    s.posted = posted_.load();
    s.computed = computed_.load();
    s.dropped = dropped_.load();
    s.compute_ns = ns_.load();
    return s;
}

}  /* namespace rad */
