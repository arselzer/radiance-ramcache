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

/* dot_avx2 for R rows at once: the weights converted once, and each row's two accumulators taking
 * exactly dot_avx2's operations in its order, so out[r] is dot_avx2(w, x[r], n) bit for bit. R <=
 * 6 keeps the twelve accumulators and the two weight vectors in the sixteen ymm registers. */
template <int R>
__attribute__((target("avx2,fma"))) void dot_rows_avx2(const uint16_t* w, const float* const* x, int n,
                                                        float* out) {
    __m256 a0[R], a1[R];
#pragma GCC unroll 6
    for (int r = 0; r < R; ++r) {
        a0[r] = _mm256_setzero_ps();
        a1[r] = _mm256_setzero_ps();
    }
    int k = 0;
    for (; k + 16 <= n; k += 16) {
        const __m256 w0 = _mm256_castsi256_ps(_mm256_slli_epi32(
            _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(w + k))), 16));
        const __m256 w1 = _mm256_castsi256_ps(_mm256_slli_epi32(
            _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(w + k + 8))), 16));
#pragma GCC unroll 6
        for (int r = 0; r < R; ++r) {
            a0[r] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x[r] + k), a0[r]);
            a1[r] = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x[r] + k + 8), a1[r]);
        }
    }
    for (; k + 8 <= n; k += 8) {
        const __m256 w0 = _mm256_castsi256_ps(_mm256_slli_epi32(
            _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(w + k))), 16));
#pragma GCC unroll 6
        for (int r = 0; r < R; ++r) a0[r] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x[r] + k), a0[r]);
    }
#pragma GCC unroll 6
    for (int r = 0; r < R; ++r) {
        const __m256 a = _mm256_add_ps(a0[r], a1[r]);
        __m128 s = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        out[r] = _mm_cvtss_f32(s);
    }
}

}  /* namespace */

void router_logits_rows(const uint16_t* w, int n_expert, int embd, const float* const* xs, int rows,
                        bool avx2, float* logits) {
    rows = std::clamp(rows, 0, (int)RouterPredictor::kMaxRows);
    for (int e = 0; e < n_expert; ++e) {
        const uint16_t* we = w + (size_t)e * (size_t)embd;
        if (!avx2) {
            for (int r = 0; r < rows; ++r) logits[(size_t)r * n_expert + e] = dot_ref(we, xs[r], embd);
            continue;
        }
        if (rows == 1) {   /* the single-row kernel itself */
            logits[e] = dot_avx2(we, xs[0], embd);
            continue;
        }
        float out[RouterPredictor::kMaxRows];
        for (int r0 = 0; r0 < rows; r0 += 6) {
            const float* const* x = xs + r0;
            switch (std::min(6, rows - r0)) {
                case 1: dot_rows_avx2<1>(we, x, embd, out + r0); break;
                case 2: dot_rows_avx2<2>(we, x, embd, out + r0); break;
                case 3: dot_rows_avx2<3>(we, x, embd, out + r0); break;
                case 4: dot_rows_avx2<4>(we, x, embd, out + r0); break;
                case 5: dot_rows_avx2<5>(we, x, embd, out + r0); break;
                default: dot_rows_avx2<6>(we, x, embd, out + r0); break;
            }
        }
        for (int r = 0; r < rows; ++r) logits[(size_t)r * n_expert + e] = out[r];
    }
}

void router_topk(const float* logits, const float* corr, int n, int k, int32_t* out) {
    n = std::clamp(n, 0, (int)RouterPredictor::kExperts);
    k = std::clamp(k, 0, n);
    std::pair<float, int32_t> key[RouterPredictor::kExperts];
    for (int e = 0; e < n; ++e) {
        const float logit = logits[e];
        /* sqrtsoftplus, as the router scores; the correction only selects (as score()). */
        const float sp = logit > 0 ? logit + std::log1p(std::exp(-logit)) : std::log1p(std::exp(logit));
        key[e] = { std::sqrt(sp) + corr[e], e };
    }
    std::partial_sort(key, key + k, key + n,
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    for (int i = 0; i < k; ++i) out[i] = key[i].second;
}

int router_topk_union(const int32_t* top, int rows, int k, int32_t* out) {
    bool seen[RouterPredictor::kExperts] = {};
    int n = 0;
    for (int pos = 0; pos < k; ++pos)
        for (int r = 0; r < rows; ++r) {
            const int32_t e = top[(size_t)r * k + pos];
            if (e < 0 || e >= RouterPredictor::kExperts || seen[e]) continue;
            seen[e] = true;
            out[n++] = e;
        }
    return n;
}

RouterPredictor* RouterPredictor::get() {
    static RouterPredictor* p = []() -> RouterPredictor* {
        const char* m = std::getenv("RADIANCE_EXPERIMENTAL_DS41_PREFETCH");
        if (!m || !*m || !std::strcmp(m, "0") || !std::strcmp(m, "off")) return nullptr;
        auto* r = new RouterPredictor();
        r->pieces_ = std::strcmp(m, "pieces") == 0;
        r->shadow_ = std::strcmp(m, "on") != 0 && !r->pieces_;
        if (r->pieces_) r->k_ = 4;
        if (const char* k = std::getenv("RADIANCE_EXPERIMENTAL_DS41_PREFETCH_K"); k && *k)
            r->k_ = std::clamp(std::atoi(k), 1, (int)kMaxK);
        if (const char* a = std::getenv("RADIANCE_EXPERIMENTAL_DS41_PREFETCH_AHEAD"); a && *a)
            r->ahead_ = std::atoi(a) == 2 ? 2 : 1;
        if (r->pieces_) r->ahead_ = 2;
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
            /* As the container holds them (weight_file_bytes): a router kept on the card as its
             * bf16 halves (RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16) is read as its f32 plane, which
             * rounds to the same bf16 here. */
            if (!wr || !wc || !wg || weight_file_bytes(*wr) != (int64_t)kExperts * kEmbd * 4 ||
                weight_file_bytes(*wc) != kExperts * 4 || weight_file_bytes(*wg) != kEmbd * 4 ||
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
        if (pieces_)
            RAD_INFO("router predictor: %d routers loaded (bf16, %.0f MB); layer L's router input "
                     "predicts layer L+2's top %d, read ahead in pieces", kLayers, w_.size() * 2 / 1e6, k_);
        else
            RAD_INFO("router predictor: %d routers loaded (bf16, %.0f MB); layer L's router input "
                     "predicts layer L+1's top %d (%s)", kLayers, w_.size() * 2 / 1e6, k_,
                     shadow_ ? "shadow: compared, nothing read" : "predicted misses are read ahead");
    });
    return load_ok_;
}

void RouterPredictor::on_ready(ReadyFn fn, void* ctx) {
    std::lock_guard<std::mutex> lk(ready_mu_);
    ready_fn_ = fn;
    ready_ctx_ = ctx;
}

void RouterPredictor::post(uint32_t pass, int layer, const uint16_t* x, int rows) {
    if (!loaded() || layer < 0 || layer + (pieces_ ? 2 : 1) >= kLayers || !x) return;
    if (rows < 1 || rows > kMaxRows || (rows > 1 && !pieces_)) return;
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
        r.rows = rows;
        r.x.assign(x, x + (size_t)kEmbd * (size_t)rows);
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
        if (r.rows > 1) compute_rows(r.pass, r.layer, r.x.data(), r.rows);
        else compute(r.pass, r.layer, r.x.data());
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
    if (pieces_) {   /* layer + 2 only, handed straight to the read-ahead */
        if (layer + 2 >= kLayers) return;
        const float* ratio2 = &ratio2_[(size_t)layer * kEmbd];
        for (int k = 0; k < kEmbd; ++k) xh[k] = bf16_to_f32(x[k]) * ratio2[k];
        Result* r = &res2_[layer + 2];
        score(layer + 2, xh, r, pass);
        int32_t e[kMaxK];
        std::memcpy(e, r->e, sizeof(int32_t) * (size_t)k_);   /* score() wrote r; only this thread does */
        const auto t1 = std::chrono::steady_clock::now();
        computed_.fetch_add(1, std::memory_order_relaxed);
        ns_.fetch_add((uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
                      std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(ready_mu_);
            if (ready_fn_) ready_fn_(ready_ctx_, pass, layer + 2, e, k_, 1);
        }
        ready_ns_.fetch_add((uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - t1).count(), std::memory_order_relaxed);
        return;
    }
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

/* A multi-row decode pass's rows (..._PREFETCH_MULTIROW): each row's stand-in for layer + 2 as the
 * one-row path makes it, every row's logits in one sweep of the layer's router, each row's top k,
 * and their union to on_ready. The one-row path's res2_ is left alone (nothing reads it in pieces
 * mode, and a one-row post never comes here). */
void RouterPredictor::compute_rows(uint32_t pass, int layer, const uint16_t* x, int rows) {
    if (!pieces_ || layer + 2 >= kLayers || rows < 2 || rows > kMaxRows) return;
    const auto t0 = std::chrono::steady_clock::now();
    const int target = layer + 2;
    xh_rows_.resize((size_t)kMaxRows * kEmbd);
    logits_rows_.resize((size_t)kMaxRows * kExperts);
    const float* ratio2 = &ratio2_[(size_t)layer * kEmbd];
    const float* xs[kMaxRows];
    for (int r = 0; r < rows; ++r) {
        float* xh = &xh_rows_[(size_t)r * kEmbd];
        const uint16_t* xr = x + (size_t)r * kEmbd;
        for (int k = 0; k < kEmbd; ++k) xh[k] = bf16_to_f32(xr[k]) * ratio2[k];
        xs[r] = xh;
    }
    router_logits_rows(&w_[(size_t)target * kExperts * kEmbd], kExperts, kEmbd, xs, rows, avx2_,
                       logits_rows_.data());
    int32_t top[kMaxRows * kMaxK], u[kMaxRows * kMaxK];
    const float* corr = &corr_[(size_t)target * kExperts];
    for (int r = 0; r < rows; ++r)
        router_topk(&logits_rows_[(size_t)r * kExperts], corr, kExperts, k_, top + (size_t)r * k_);
    const int n = router_topk_union(top, rows, k_, u);
    const auto t1 = std::chrono::steady_clock::now();
    const uint64_t ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    computed_.fetch_add(1, std::memory_order_relaxed);
    ns_.fetch_add(ns, std::memory_order_relaxed);
    multi_.fetch_add(1, std::memory_order_relaxed);
    multi_rows_.fetch_add((uint64_t)rows, std::memory_order_relaxed);
    multi_ns_.fetch_add(ns, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(ready_mu_);
        if (ready_fn_) ready_fn_(ready_ctx_, pass, target, u, n, rows);
    }
    ready_ns_.fetch_add((uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - t1).count(), std::memory_order_relaxed);
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
    s.ready_ns = ready_ns_.load();
    s.multi = multi_.load();
    s.multi_rows = multi_rows_.load();
    s.multi_ns = multi_ns_.load();
    return s;
}

}  /* namespace rad */
