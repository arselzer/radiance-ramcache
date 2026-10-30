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
 * RADIANCE_EXPERIMENTAL_DS41_PREFETCH_SCORE=int8|int4 (bf16, 0 or unset: as above): every expert is
 * first scored from a quantized copy of the routers (router_score.h: one scale an expert, integer
 * dots, 79 / 39 MB beside the bf16 copy, which stays), and only the best ..._PREFETCH_RESCORE (12)
 * are scored again in bf16; the best k of those are the prediction. A prediction reads 1.97 /
 * 0.98 MB instead of 3.93 MB from DRAM, so it lands sooner and takes less of the bandwidth the
 * pool's bursts are bound by; it is bf16's own whenever bf16's best k are among the rescored.
 * ..._PREFETCH_SCORE_CHECK=N (0): every Nth prediction is also scored in bf16 alone, after it was
 * handed over, and compared (stats(): the same set, the same order, and where bf16's picks ranked
 * in the quantized scores) -- a check arm's, not a timed one's.
 *
 * RADIANCE_EXPERIMENTAL_DS41_PREFETCH_RECORD=<path>: every posted router input is appended to <path>
 * after its prediction was handed over, for scoring offline (tools/ds41-host-experts/
 * router_score_eval.cpp, which reads the routers from the container): a 64-byte header, char[8]
 * "RADPRDR1", u32 version 1, layers 40, experts 384, embd 5120, ahead (the layer predicted is the
 * input's + ahead), k, then zeros; then records of u32 pass, i32 layer, u16 x[5120] (bf16), 10,248
 * bytes each. An instrument: nothing it writes is read back. */
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace rad {

struct Program;
struct WeightInfo;

class RouterPredictor {
public:
    static constexpr int kLayers = 40, kExperts = 384, kEmbd = 5120, kMaxK = 16;
    /* How a prediction scores the experts (..._PREFETCH_SCORE); the value is the bits a weight. */
    enum Score { kScoreBf16 = 16, kScoreInt8 = 8, kScoreInt4 = 4 };
    /* The quantized scores rescored in bf16 at most; the check's buckets for the rank bf16's picks
     * had in the quantized scores: under 4, 8, 12, 16, 24, 32, and 32 or more. */
    static constexpr int kMaxRescore = 64, kRankBuckets = 7;
    static constexpr int kRankEdge[kRankBuckets - 1] = { 4, 8, 12, 16, 24, 32 };

    /* What get() reads from the environment (see above); create() takes it from a caller. */
    struct Options {
        bool  shadow = true;
        bool  pieces = false;
        int   k = 10;
        int   ahead = 1;
        Score score = kScoreBf16;
        int   rescore = 12;   /* clamped to [k, kMaxRescore] */
        int   check = 0;      /* every check-th prediction compared with bf16 alone; 0 never */
        std::string record;   /* ..._PREFETCH_RECORD: the inputs' file, or empty */
    };
    /* The process's predictor, or null when RADIANCE_EXPERIMENTAL_DS41_PREFETCH is unset (or a
     * modifier has a value it does not take, which it says). */
    static RouterPredictor* get();
    /* A predictor of its own, its worker running (tests, tools). */
    static std::unique_ptr<RouterPredictor> create(const Options& o);
    bool shadow() const { return shadow_; }
    int  k() const { return k_; }
    Score score_mode() const { return score_; }
    int  rescore() const { return rescore_; }
    int  check_every() const { return check_; }
    /* The quantized copy's bytes (0 with bf16 scoring). */
    int64_t quant_bytes() const { return (int64_t)qw_bytes_; }
    /* ..._PREFETCH_RECORD: the file (empty: none), and its records flushed to it (before the
     * shutdown report reads the count). */
    const std::string& record_path() const { return record_path_; }
    void flush_record();
    /* ..._PREFETCH_AHEAD: how many layers ahead the read-ahead reads (1, or 2 from predicted2). */
    int  ahead() const { return ahead_; }
    /* ..._PREFETCH=pieces: two layers ahead only, each prediction handed to on_ready. */
    bool pieces() const { return pieces_; }
    /* The function a finished prediction is handed to (pieces mode), on the predictor's thread:
     * the pass, the layer predicted and its k() best global experts, best first. Set before the
     * first post; null stops the calls. */
    using ReadyFn = void (*)(void* ctx, uint32_t pass, int layer, const int32_t* experts, int k);
    void on_ready(ReadyFn fn, void* ctx);

    /* Once for the process (later calls return the first's answer): every layer's router weight,
     * correction and ffn_norm gain, each read through `read` (the weight's stored bytes); with a
     * quantized score, the quantized copy made from the f32 weights. */
    bool load(const Program& prog, const std::function<bool(const WeightInfo&, void*)>& read);
    bool loaded() const { return loaded_.load(std::memory_order_acquire); }

    /* Layer `target`'s best k() experts for the stand-in input xh (kEmbd f32, already rescaled to
     * the target's gain), best first, as the worker computes them: on the caller's thread, no
     * statistics. A loaded predictor only. */
    void predict(int target, const float* xh, int32_t* experts) const;

    /* Layer `layer`'s router input for pass `pass` (kEmbd bf16): the worker predicts layer + 1. */
    void post(uint32_t pass, int layer, const uint16_t* x);
    /* The k predicted global experts of `layer` for `pass`, best first; false until ready. */
    bool predicted(uint32_t pass, int layer, int32_t* experts) const;
    /* The same from two layers back (layer - 2's router input, rescaled by g_layer / g_{layer-2}):
     * a longer lead for a read, computed in shadow mode only. */
    bool predicted2(uint32_t pass, int layer, int32_t* experts) const;

    /* posted, computed, dropped (a full queue); compute_ns scoring, ready_ns in on_ready. With a
     * quantized score, compute_ns's parts: quant_ns the quantized pass (the input's quantization,
     * the dots, the keys and the sort for the rescored), rescore_ns the bf16 rescoring and the
     * final sort. The check (..._SCORE_CHECK, not in compute_ns): predictions checked, of them
     * with bf16's own set of k and its order, bf16's picks by the rank they had in the quantized
     * scores (kRankEdge), those at or past the rescored, and the check's time. */
    struct Stats {
        uint64_t posted = 0, computed = 0, dropped = 0, compute_ns = 0, ready_ns = 0;
        uint64_t quant_ns = 0, rescore_ns = 0;
        uint64_t checked = 0, check_set = 0, check_order = 0, check_outside = 0, check_ns = 0;
        uint64_t check_rank[kRankBuckets] = {};
        uint64_t recorded = 0;   /* ..._PREFETCH_RECORD: inputs written */
    };
    Stats stats() const;
    ~RouterPredictor();

private:
    RouterPredictor() = default;
    static RouterPredictor* make(const Options& o);
    void loop();
    void compute(uint32_t pass, int layer, const uint16_t* x);
    /* Layer target's best k_ for xh into e; quantized: each expert's quantized key into qkey
     * (when given), the parts' times into *q_ns and *r_ns. */
    void rank(int target, const float* xh, int32_t* e, float* qkey, uint64_t* q_ns, uint64_t* r_ns) const;
    /* The check, after the prediction e was handed over: bf16 alone against it. */
    void check(int target, const float* xh, const int32_t* e, const float* qkey);

    bool shadow_ = true;
    bool pieces_ = false;
    int  k_ = 10;
    int  ahead_ = 1;
    Score score_ = kScoreBf16;
    int  rescore_ = 12;
    int  check_ = 0;
    uint64_t check_tick_ = 0;   /* the worker's predictions, for the check's turn */
    /* ..._PREFETCH_RECORD: opened by load(), written by the worker (record()), under rec_mu_. */
    std::string record_path_;
    std::mutex  rec_mu_;
    FILE*       rec_fp_ = nullptr;
    void record(uint32_t pass, int layer, const uint16_t* x);
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
    /* The quantized copy (score_ int8 / int4): [kLayers][kExperts][qw_row_] bytes on 2 MiB pages
     * (router_score.h's layout), and each expert's scale [kLayers][kExperts]. */
    struct QFree { void operator()(uint8_t* p) const { std::free(p); } };
    std::unique_ptr<uint8_t, QFree> qw_;
    size_t qw_bytes_ = 0;
    int    qw_row_ = 0;
    std::vector<float> qw_scale_;

    struct Result { uint32_t pass = 0; bool valid = false; int32_t e[kMaxK] = {}; };
    Result res_[kLayers];
    Result res2_[kLayers];         /* by target layer, from two layers back */
    void score(int target, const float* xh, Result* out, uint32_t pass, float* qkey);
    mutable std::mutex res_mu_;

    struct Req { uint32_t pass = 0; int layer = 0; std::vector<uint16_t> x; };
    std::deque<Req> q_;
    std::mutex q_mu_;
    std::condition_variable q_cv_;
    bool stop_ = false;
    std::thread worker_;
    std::atomic<uint64_t> posted_{ 0 }, computed_{ 0 }, dropped_{ 0 }, ns_{ 0 }, ready_ns_{ 0 };
    std::atomic<uint64_t> quant_ns_{ 0 }, rescore_ns_{ 0 };
    std::atomic<uint64_t> checked_{ 0 }, check_set_{ 0 }, check_order_{ 0 }, check_outside_{ 0 },
                          check_ns_{ 0 };
    std::atomic<uint64_t> check_rank_[kRankBuckets] = {};
    std::atomic<uint64_t> recorded_{ 0 };
};

}  /* namespace rad */
