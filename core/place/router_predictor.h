/* router_predictor.h -- DeepSeek-V4.1's routed experts predicted one layer ahead on the CPU, so the
 * experts the RAM cache misses can be read before their layer asks for them.
 *
 * At layer L's routing sync the file stager holds layer L's router input on the host (the CPU
 * expert path copies it with the counts row): x_L = rmsnorm(h_L) * g_L, g the layer's ffn_norm
 * gain. Layer L+1's router reads rmsnorm(h_{L+1}) * g_{L+1}, and the residual stream moves little
 * in one layer, so x_L * g_{L+1} / g_L stands in for it. A worker scores every expert as the router
 * does (sqrt(softplus(w . x)) + correction) and keeps the best k. The prediction is a hint for
 * reads only: what the card routes is untouched.
 *
 * One predictor per process: the router input is the same on every rank, so rank 0 posts it and
 * each rank reads the prediction for its own experts. The 40 routers are kept as bf16 (157 MB).
 *
 * RADIANCE_EXPERIMENTAL_DS41_PREFETCH=shadow predicts and compares with what the layer then misses,
 * reading nothing; =on also reads the predicted misses (file stager). ..._PREFETCH_K: experts kept
 * (10; at most 16).
 *
 * =pieces (readahead.h): layer L's input predicts layer L+2 only (the read-ahead's lead), posted
 * once layer L's CPU work is submitted, and each prediction goes straight to the callback
 * (on_ready) instead of waiting to be polled. ..._PREFETCH_K defaults to 4 there.
 *
 * ---- multi-row decode passes (RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MULTIROW, pieces mode) --------
 *
 * A decode pass of 2..kMaxRows rows (a DSpark verify pass: the last token and its drafts; or one
 * token from each of several sequences) needs every row's experts, and a draft row brings its own.
 * Rank 0 posts all the rows' router inputs at once (post's `rows`). The worker reads layer L+2's
 * router weights once for all of them (router_logits_rows; each row's logits are bit for bit what
 * that row posted alone would give), takes each row's top k (router_topk), and hands on_ready
 * their union (router_topk_union): position by position, row by row, each expert once, so every
 * row's best prediction comes before any row's second. Why every row and not row 0 with a wider
 * k: a draft row's experts are its own (they do not amortise), and the router's weights, not the
 * rows, are what a prediction costs (3.9 MB a layer streamed from memory), so a second row costs
 * a fraction of a second prediction. A one-row post goes the way it always went. */
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace rad {

struct Program;
struct WeightInfo;

/* Router logits of `rows` inputs xs[r] (embd floats each, embd a multiple of 8) against one
 * layer's router (bf16 [n_expert][embd]): logits[r * n_expert + e] = w_e . xs[r]. With avx2 the
 * weights are read once for all the rows and each row takes the single-row kernel's operations in
 * its order (so its logits are bit for bit a one-row call's); without, the scalar loop. rows <=
 * RouterPredictor::kMaxRows. */
void router_logits_rows(const uint16_t* w, int n_expert, int embd, const float* const* xs, int rows,
                        bool avx2, float* logits);
/* The best k of n experts by the router's key, sqrt(softplus(logit)) + corr, best first, into out
 * (1 <= k <= n <= RouterPredictor::kExperts); equal keys in partial_sort's order. */
void router_topk(const float* logits, const float* corr, int n, int k, int32_t* out);
/* The union of `rows` lists of k experts (top[r * k + pos], each best first): position by
 * position, row by row, each expert once (ids in [0, RouterPredictor::kExperts)). Its length, at
 * most rows * k, is returned. */
int router_topk_union(const int32_t* top, int rows, int k, int32_t* out);

class RouterPredictor {
public:
    static constexpr int kLayers = 40, kExperts = 384, kEmbd = 5120, kMaxK = 16;
    /* The most rows a post carries (..._PREFETCH_MULTIROW): a pass the EP counts path takes. */
    static constexpr int kMaxRows = 16;

    /* The process's predictor, or null when RADIANCE_EXPERIMENTAL_DS41_PREFETCH is unset. */
    static RouterPredictor* get();
    bool shadow() const { return shadow_; }
    int  k() const { return k_; }
    /* ..._PREFETCH_AHEAD: how many layers ahead the read-ahead reads (1, or 2 from predicted2). */
    int  ahead() const { return ahead_; }
    /* ..._PREFETCH=pieces: two layers ahead only, each prediction handed to on_ready. */
    bool pieces() const { return pieces_; }
    /* The function a finished prediction is handed to (pieces mode), on the predictor's thread:
     * the pass, the layer predicted and its n best global experts, best first, from a post of
     * `rows` rows: a one-row post's k() best, a multi-row post's union (router_topk_union). Set
     * before the first post; null stops the calls. */
    using ReadyFn = void (*)(void* ctx, uint32_t pass, int layer, const int32_t* experts, int n,
                             int rows);
    void on_ready(ReadyFn fn, void* ctx);

    /* Once for the process (later calls return the first's answer): every layer's router weight,
     * correction and ffn_norm gain, each read through `read` (the weight's stored bytes). */
    bool load(const Program& prog, const std::function<bool(const WeightInfo&, void*)>& read);
    bool loaded() const { return loaded_.load(std::memory_order_acquire); }

    /* Layer `layer`'s router input for pass `pass` (kEmbd bf16): the worker predicts layer + 1.
     * `rows` > 1 (pieces mode only, at most kMaxRows): that many inputs, kEmbd apart, from a
     * multi-row decode pass, predicted together (see the file header). */
    void post(uint32_t pass, int layer, const uint16_t* x, int rows = 1);
    /* The k predicted global experts of `layer` for `pass`, best first; false until ready. */
    bool predicted(uint32_t pass, int layer, int32_t* experts) const;
    /* The same from two layers back (layer - 2's router input, rescaled by g_layer / g_{layer-2}):
     * a longer lead for a read, computed in shadow mode only. */
    bool predicted2(uint32_t pass, int layer, int32_t* experts) const;

    /* posted, computed, dropped (a full queue); compute_ns scoring, ready_ns in on_ready. Of the
     * computed, the multi-row posts' (..._PREFETCH_MULTIROW): predictions, their rows, their
     * scoring. */
    struct Stats {
        uint64_t posted = 0, computed = 0, dropped = 0, compute_ns = 0, ready_ns = 0;
        uint64_t multi = 0, multi_rows = 0, multi_ns = 0;
    };
    Stats stats() const;
    ~RouterPredictor();

private:
    RouterPredictor() = default;
    void loop();
    void compute(uint32_t pass, int layer, const uint16_t* x);
    /* Pieces mode, a post of rows > 1: every row predicts layer + 2; their union to on_ready. */
    void compute_rows(uint32_t pass, int layer, const uint16_t* x, int rows);

    bool shadow_ = true;
    bool pieces_ = false;
    int  k_ = 10;
    int  ahead_ = 1;
    std::mutex ready_mu_;
    ReadyFn    ready_fn_ = nullptr;
    void*      ready_ctx_ = nullptr;
    bool avx2_ = false;
    std::once_flag load_once_;
    std::atomic<bool> loaded_{ false };
    bool load_ok_ = false;
    std::vector<uint16_t> w_;      /* [kLayers][kExperts][kEmbd] bf16 */
    std::vector<float>    corr_;   /* [kLayers][kExperts] */
    std::vector<float>    ratio_;  /* [kLayers][kEmbd]: g_{L+1} / g_L, the stand-in's rescale */
    std::vector<float>    ratio2_; /* [kLayers][kEmbd]: g_{L+2} / g_L */

    struct Result { uint32_t pass = 0; bool valid = false; int32_t e[kMaxK] = {}; };
    Result res_[kLayers];
    Result res2_[kLayers];         /* by target layer, from two layers back */
    void score(int target, const float* xh, Result* out, uint32_t pass);
    mutable std::mutex res_mu_;

    struct Req { uint32_t pass = 0; int layer = 0; int rows = 1; std::vector<uint16_t> x; };
    std::deque<Req> q_;
    std::mutex q_mu_;
    std::condition_variable q_cv_;
    bool stop_ = false;
    std::thread worker_;
    std::atomic<uint64_t> posted_{ 0 }, computed_{ 0 }, dropped_{ 0 }, ns_{ 0 }, ready_ns_{ 0 };
    std::atomic<uint64_t> multi_{ 0 }, multi_rows_{ 0 }, multi_ns_{ 0 };
    /* compute_rows' scratch (the worker's only): the rows' stand-in inputs and their logits. */
    std::vector<float> xh_rows_, logits_rows_;
};

}  /* namespace rad */
