/* router_predictor.cpp -- see router_predictor.h. */
#include "router_predictor.h"
#include "router_score.h"

#include "../rad_internal.h"
#include "../rad_core.h"

#include <sys/mman.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace rad {

using rscore::bf16_to_f32;
using rscore::f32_to_bf16_rne;

namespace {

static_assert(RouterPredictor::kEmbd % 64 == 0 && RouterPredictor::kEmbd <= rscore::kMaxDotElems,
              "the integer dots take whole steps of 64 within their int32 bound");

uint64_t ns_since(std::chrono::steady_clock::time_point t0) {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - t0).count();
}

/* ..._PREFETCH_SCORE: bf16 when unset or off, the mode, or -1 for a value it does not take. */
int parse_score(const char* v) {
    if (!v || !*v || !std::strcmp(v, "0") || !std::strcmp(v, "off") || !std::strcmp(v, "bf16"))
        return RouterPredictor::kScoreBf16;
    if (!std::strcmp(v, "int8")) return RouterPredictor::kScoreInt8;
    if (!std::strcmp(v, "int4")) return RouterPredictor::kScoreInt4;
    return -1;
}

}  /* namespace */

RouterPredictor* RouterPredictor::get() {
    static RouterPredictor* p = []() -> RouterPredictor* {
        const char* m = std::getenv("RADIANCE_EXPERIMENTAL_DS41_PREFETCH");
        if (!m || !*m || !std::strcmp(m, "0") || !std::strcmp(m, "off")) return nullptr;
        Options o;
        o.pieces = std::strcmp(m, "pieces") == 0;
        o.shadow = std::strcmp(m, "on") != 0 && !o.pieces;
        if (o.pieces) o.k = 4;
        if (const char* k = std::getenv("RADIANCE_EXPERIMENTAL_DS41_PREFETCH_K"); k && *k)
            o.k = std::clamp(std::atoi(k), 1, (int)kMaxK);
        if (const char* a = std::getenv("RADIANCE_EXPERIMENTAL_DS41_PREFETCH_AHEAD"); a && *a)
            o.ahead = std::atoi(a) == 2 ? 2 : 1;
        if (o.pieces) o.ahead = 2;
        const char* sc = std::getenv("RADIANCE_EXPERIMENTAL_DS41_PREFETCH_SCORE");
        const int mode = parse_score(sc);
        if (mode < 0) {
            RAD_ERR("RADIANCE_EXPERIMENTAL_DS41_PREFETCH_SCORE=%s: bf16 (or 0), int8 or int4; no router "
                    "predictor", sc);
            return nullptr;
        }
        o.score = (Score)mode;
        if (const char* r = std::getenv("RADIANCE_EXPERIMENTAL_DS41_PREFETCH_RESCORE"); r && *r)
            o.rescore = std::atoi(r);
        if (const char* c = std::getenv("RADIANCE_EXPERIMENTAL_DS41_PREFETCH_SCORE_CHECK"); c && *c)
            o.check = std::max(std::atoi(c), 0);
        if (o.check > 0 && o.score == kScoreBf16) {
            RAD_WARN("RADIANCE_EXPERIMENTAL_DS41_PREFETCH_SCORE_CHECK=%d compares a quantized score "
                     "with bf16 (..._PREFETCH_SCORE=int8 or int4); nothing to check", o.check);
            o.check = 0;
        }
        if (const char* rec = std::getenv("RADIANCE_EXPERIMENTAL_DS41_PREFETCH_RECORD"); rec && *rec)
            o.record = rec;
        return make(o);
    }();
    return p;
}

std::unique_ptr<RouterPredictor> RouterPredictor::create(const Options& o) {
    return std::unique_ptr<RouterPredictor>(make(o));
}

RouterPredictor* RouterPredictor::make(const Options& o) {
    auto* r = new RouterPredictor();
    r->shadow_ = o.shadow;
    r->pieces_ = o.pieces;
    r->k_ = std::clamp(o.k, 1, (int)kMaxK);
    r->ahead_ = o.ahead == 2 ? 2 : 1;
    r->score_ = o.score;
    r->rescore_ = std::clamp(o.rescore, r->k_, (int)kMaxRescore);
    r->check_ = r->score_ == kScoreBf16 ? 0 : std::max(o.check, 0);
    r->record_path_ = o.record;
    r->avx2_ = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    r->worker_ = std::thread(&RouterPredictor::loop, r);
    return r;
}

RouterPredictor::~RouterPredictor() {
    {
        std::lock_guard<std::mutex> lk(q_mu_);
        stop_ = true;
    }
    q_cv_.notify_all();
    if (worker_.joinable()) worker_.join();
    std::lock_guard<std::mutex> lk(rec_mu_);
    if (rec_fp_) std::fclose(rec_fp_);
    rec_fp_ = nullptr;
}

/* ..._PREFETCH_RECORD: one input after its prediction was handed over (router_predictor.h). A
 * failed write closes the file, said once. */
void RouterPredictor::record(uint32_t pass, int layer, const uint16_t* x) {
    std::lock_guard<std::mutex> lk(rec_mu_);
    if (!rec_fp_) return;
    const int32_t l = layer;
    if (std::fwrite(&pass, 4, 1, rec_fp_) != 1 || std::fwrite(&l, 4, 1, rec_fp_) != 1 ||
        std::fwrite(x, 2, kEmbd, rec_fp_) != (size_t)kEmbd) {
        RAD_WARN("router predictor: writing %s failed; no more inputs recorded", record_path_.c_str());
        std::fclose(rec_fp_);
        rec_fp_ = nullptr;
        return;
    }
    recorded_.fetch_add(1, std::memory_order_relaxed);
}

void RouterPredictor::flush_record() {
    std::lock_guard<std::mutex> lk(rec_mu_);
    if (rec_fp_) std::fflush(rec_fp_);
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
        /* The quantized copy, on 2 MiB pages where the kernel grants them: a prediction streams
         * one layer's 0.98-1.97 MB of it. */
        if (score_ != kScoreBf16) {
            qw_row_ = score_ == kScoreInt8 ? kEmbd : kEmbd / 2;
            const size_t bytes = (size_t)kLayers * kExperts * (size_t)qw_row_, huge = (size_t)2 << 20;
            qw_.reset(static_cast<uint8_t*>(std::aligned_alloc(huge, (bytes + huge - 1) / huge * huge)));
            if (!qw_) {
                RAD_WARN("router predictor: no memory for the %s copy of the routers (%.0f MB); no "
                         "prediction", score_ == kScoreInt8 ? "int8" : "int4", bytes / 1e6);
                return;
            }
            (void)::madvise(qw_.get(), (bytes + huge - 1) / huge * huge, MADV_HUGEPAGE);
            qw_bytes_ = bytes;
            qw_scale_.assign((size_t)kLayers * kExperts, 0.0f);
        }
        uint64_t quant_ns = 0;
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
            /* Quantized from the f32 weights (the container's routers are bf16-valued, so the same
             * values the bf16 copy holds). */
            if (qw_) {
                const auto t0 = std::chrono::steady_clock::now();
                for (int e = 0; e < kExperts; ++e) {
                    const size_t r = (size_t)l * kExperts + (size_t)e;
                    uint8_t* q = qw_.get() + r * (size_t)qw_row_;
                    const float* w = &wf[(size_t)e * kEmbd];
                    qw_scale_[r] = score_ == kScoreInt8 ? rscore::quantize_row_i8(w, kEmbd, (int8_t*)q)
                                                       : rscore::quantize_row_u4(w, kEmbd, q);
                }
                quant_ns += ns_since(t0);
            }
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
        /* ..._PREFETCH_RECORD: the file and its header, before the first post can come. */
        if (!record_path_.empty()) {
            std::lock_guard<std::mutex> lk(rec_mu_);
            rec_fp_ = std::fopen(record_path_.c_str(), "wb");
            uint32_t h[16] = {};
            std::memcpy(h, "RADPRDR1", 8);
            h[2] = 1;
            h[3] = kLayers;
            h[4] = kExperts;
            h[5] = kEmbd;
            h[6] = (uint32_t)(pieces_ ? 2 : ahead_);
            h[7] = (uint32_t)k_;
            if (!rec_fp_ || std::fwrite(h, sizeof h, 1, rec_fp_) != 1) {
                RAD_WARN("router predictor: cannot write %s; no inputs recorded", record_path_.c_str());
                if (rec_fp_) std::fclose(rec_fp_);
                rec_fp_ = nullptr;
            } else {
                RAD_INFO("router predictor: every posted router input is recorded to %s "
                         "(RADIANCE_EXPERIMENTAL_DS41_PREFETCH_RECORD)", record_path_.c_str());
            }
        }
        load_ok_ = true;
        loaded_.store(true, std::memory_order_release);
        /* ..._PREFETCH_SCORE: the copy each prediction scores first (ab_window.sh's Pi parts look
         * for "scored in int8" / "scored in int4" here, and a check arm's Pk for "checked"). */
        std::string q;
        if (qw_)
            q = fmt("; scored in %s (one scale an expert, %.0f MB, quantized in %.2f s), the best %d "
                    "scored again in bf16%s", score_ == kScoreInt8 ? "int8" : "int4", qw_bytes_ / 1e6,
                    (double)quant_ns * 1e-9, rescore_,
                    check_ ? fmt("; one prediction in %d checked against bf16 alone", check_).c_str() : "");
        if (pieces_)
            RAD_INFO("router predictor: %d routers loaded (bf16, %.0f MB)%s; layer L's router input "
                     "predicts layer L+2's top %d, read ahead in pieces", kLayers, w_.size() * 2 / 1e6,
                     q.c_str(), k_);
        else
            RAD_INFO("router predictor: %d routers loaded (bf16, %.0f MB)%s; layer L's router input "
                     "predicts layer L+1's top %d (%s)", kLayers, w_.size() * 2 / 1e6, q.c_str(), k_,
                     shadow_ ? "shadow: compared, nothing read" : "predicted misses are read ahead");
    });
    return load_ok_;
}

void RouterPredictor::on_ready(ReadyFn fn, void* ctx) {
    std::lock_guard<std::mutex> lk(ready_mu_);
    ready_fn_ = fn;
    ready_ctx_ = ctx;
}

void RouterPredictor::post(uint32_t pass, int layer, const uint16_t* x) {
    if (!loaded() || layer < 0 || layer + (pieces_ ? 2 : 1) >= kLayers || !x) return;
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

/* Layer `target`'s router on the stand-in input xh: its best k_ into e, best first. bf16: every
 * expert's key from its bf16 row. Quantized (router_score.h): every expert's key from its quantized
 * row and the input quantized to int16, then the best rescore_ of those keyed again from their
 * bf16 rows, and the best k_ of the rescored. */
void RouterPredictor::rank(int target, const float* xh, int32_t* e, float* qkey, uint64_t* q_ns,
                           uint64_t* r_ns) const {
    const uint16_t* w = &w_[(size_t)target * kExperts * kEmbd];
    const float* corr = &corr_[(size_t)target * kExperts];
    std::pair<float, int32_t> key[kExperts];
    const auto best = [](const std::pair<float, int32_t>& a, const std::pair<float, int32_t>& b) {
        return a.first > b.first;
    };
    /* sqrtsoftplus, as the router scores; the correction only selects. */
    const auto bf16_key = [&](int x) {
        const float logit = avx2_ ? rscore::dot_bf16_avx2(w + (size_t)x * kEmbd, xh, kEmbd)
                                  : rscore::dot_bf16_ref(w + (size_t)x * kEmbd, xh, kEmbd);
        return rscore::key_of(logit, corr[x]);
    };
    if (!qw_) {
        for (int x = 0; x < kExperts; ++x) key[x] = { bf16_key(x), x };
        std::partial_sort(key, key + k_, key + kExperts, best);
        for (int i = 0; i < k_; ++i) e[i] = key[i].second;
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    alignas(32) int16_t xq[kEmbd];
    int64_t sum = 0;
    const float sx = rscore::quantize_x16(xh, kEmbd, xq, &sum);
    const size_t r0 = (size_t)target * kExperts;
    const uint8_t* q = qw_.get() + r0 * (size_t)qw_row_;
    const float* sc = &qw_scale_[r0];
    if (score_ == kScoreInt8) {
        for (int x = 0; x < kExperts; ++x) {
            const int8_t* qx = (const int8_t*)(q + (size_t)x * (size_t)qw_row_);
            const int64_t d = avx2_ ? rscore::dot_i8_avx2(qx, xq, kEmbd) : rscore::dot_i8_ref(qx, xq, kEmbd);
            key[x] = { rscore::key_of(sc[x] * sx * (float)d, corr[x]), x };
        }
    } else {   /* w ~ s (n - 8): the dot less 8 times the input's sum */
        const int64_t off = 8 * sum;
        for (int x = 0; x < kExperts; ++x) {
            const uint8_t* qx = q + (size_t)x * (size_t)qw_row_;
            const int64_t d = avx2_ ? rscore::dot_u4_avx2(qx, xq, kEmbd) : rscore::dot_u4_ref(qx, xq, kEmbd);
            key[x] = { rscore::key_of(sc[x] * sx * (float)(d - off), corr[x]), x };
        }
    }
    if (qkey)
        for (int x = 0; x < kExperts; ++x) qkey[x] = key[x].first;
    std::partial_sort(key, key + rescore_, key + kExperts, best);
    const auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < rescore_; ++i) key[i].first = bf16_key(key[i].second);
    std::partial_sort(key, key + k_, key + rescore_, best);
    for (int i = 0; i < k_; ++i) e[i] = key[i].second;
    if (q_ns) *q_ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    if (r_ns) *r_ns = ns_since(t1);
}

void RouterPredictor::predict(int target, const float* xh, int32_t* experts) const {
    if (!loaded() || target < 0 || target >= kLayers || !xh || !experts) return;
    rank(target, xh, experts, nullptr, nullptr, nullptr);
}

/* Layer `target`'s router on the stand-in input xh: its best k into *out. */
void RouterPredictor::score(int target, const float* xh, Result* out, uint32_t pass, float* qkey) {
    int32_t e[kMaxK];
    uint64_t q_ns = 0, r_ns = 0;
    rank(target, xh, e, qkey, &q_ns, &r_ns);
    if (qw_) {
        quant_ns_.fetch_add(q_ns, std::memory_order_relaxed);
        rescore_ns_.fetch_add(r_ns, std::memory_order_relaxed);
    }
    std::lock_guard<std::mutex> lk(res_mu_);
    out->pass = pass;
    out->valid = true;
    for (int i = 0; i < k_; ++i) out->e[i] = e[i];
}

/* ..._PREFETCH_SCORE_CHECK: bf16 alone on the same input, against the prediction e made from the
 * quantized keys qkey. */
void RouterPredictor::check(int target, const float* xh, const int32_t* e, const float* qkey) {
    const auto t0 = std::chrono::steady_clock::now();
    const uint16_t* w = &w_[(size_t)target * kExperts * kEmbd];
    const float* corr = &corr_[(size_t)target * kExperts];
    std::pair<float, int32_t> key[kExperts];
    for (int x = 0; x < kExperts; ++x) {
        const float logit = avx2_ ? rscore::dot_bf16_avx2(w + (size_t)x * kEmbd, xh, kEmbd)
                                  : rscore::dot_bf16_ref(w + (size_t)x * kEmbd, xh, kEmbd);
        key[x] = { rscore::key_of(logit, corr[x]), x };
    }
    std::partial_sort(key, key + k_, key + kExperts,
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    bool set = true, order = true;
    for (int i = 0; i < k_; ++i) {
        const int32_t b = key[i].second;
        order = order && e[i] == b;
        bool in = false;
        for (int j = 0; j < k_; ++j) in = in || e[j] == b;
        set = set && in;
        /* b's rank among the quantized keys: the experts that scored above it there. */
        int r = 0;
        for (int x = 0; x < kExperts; ++x) r += qkey[x] > qkey[b];
        int bucket = 0;
        while (bucket < kRankBuckets - 1 && r >= kRankEdge[bucket]) ++bucket;
        check_rank_[bucket].fetch_add(1, std::memory_order_relaxed);
        if (r >= rescore_) check_outside_.fetch_add(1, std::memory_order_relaxed);
    }
    checked_.fetch_add(1, std::memory_order_relaxed);
    check_set_.fetch_add(set, std::memory_order_relaxed);
    check_order_.fetch_add(order, std::memory_order_relaxed);
    check_ns_.fetch_add(ns_since(t0), std::memory_order_relaxed);
}

void RouterPredictor::compute(uint32_t pass, int layer, const uint16_t* x) {
    const auto t0 = std::chrono::steady_clock::now();
    float xh[kEmbd];
    float qkey[kExperts];
    /* ..._PREFETCH_SCORE_CHECK: this prediction's turn (only this thread counts them). */
    const bool chk = check_ > 0 && check_tick_++ % (uint64_t)check_ == 0;
    if (pieces_) {   /* layer + 2 only, handed straight to the read-ahead */
        if (layer + 2 >= kLayers) return;
        const float* ratio2 = &ratio2_[(size_t)layer * kEmbd];
        for (int k = 0; k < kEmbd; ++k) xh[k] = bf16_to_f32(x[k]) * ratio2[k];
        Result* r = &res2_[layer + 2];
        score(layer + 2, xh, r, pass, chk ? qkey : nullptr);
        int32_t e[kMaxK];
        std::memcpy(e, r->e, sizeof(int32_t) * (size_t)k_);   /* score() wrote r; only this thread does */
        const auto t1 = std::chrono::steady_clock::now();
        computed_.fetch_add(1, std::memory_order_relaxed);
        ns_.fetch_add((uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
                      std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(ready_mu_);
            if (ready_fn_) ready_fn_(ready_ctx_, pass, layer + 2, e, k_);
        }
        ready_ns_.fetch_add((uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - t1).count(), std::memory_order_relaxed);
        if (chk) check(layer + 2, xh, e, qkey);   /* after the hand-over: the read-ahead never waits for it */
        record(pass, layer, x);
        return;
    }
    const float* ratio = &ratio_[(size_t)layer * kEmbd];
    for (int k = 0; k < kEmbd; ++k) xh[k] = bf16_to_f32(x[k]) * ratio[k];
    score(layer + 1, xh, &res_[layer + 1], pass, chk ? qkey : nullptr);
    uint64_t check_ns = 0;
    if (chk) {   /* the next layer's prediction against bf16 alone (only this thread writes res_) */
        const uint64_t c0 = check_ns_.load(std::memory_order_relaxed);
        check(layer + 1, xh, res_[layer + 1].e, qkey);
        check_ns = check_ns_.load(std::memory_order_relaxed) - c0;
    }
    /* Two layers ahead from the same input: in shadow mode for the statistics, and for a read-ahead
     * two layers on. */
    if ((shadow_ || ahead_ == 2) && layer + 2 < kLayers) {
        const float* ratio2 = &ratio2_[(size_t)layer * kEmbd];
        for (int k = 0; k < kEmbd; ++k) xh[k] = bf16_to_f32(x[k]) * ratio2[k];
        score(layer + 2, xh, &res2_[layer + 2], pass, nullptr);
    }
    computed_.fetch_add(1, std::memory_order_relaxed);
    ns_.fetch_add(ns_since(t0) - check_ns, std::memory_order_relaxed);
    record(pass, layer, x);
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
    s.quant_ns = quant_ns_.load();
    s.rescore_ns = rescore_ns_.load();
    s.checked = checked_.load();
    s.check_set = check_set_.load();
    s.check_order = check_order_.load();
    s.check_outside = check_outside_.load();
    s.check_ns = check_ns_.load();
    for (int b = 0; b < kRankBuckets; ++b) s.check_rank[b] = check_rank_[b].load();
    s.recorded = recorded_.load();
    return s;
}

}  /* namespace rad */
