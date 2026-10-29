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
 * (10; at most 16). */
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

class RouterPredictor {
public:
    static constexpr int kLayers = 40, kExperts = 384, kEmbd = 5120, kMaxK = 16;

    /* The process's predictor, or null when RADIANCE_EXPERIMENTAL_DS41_PREFETCH is unset. */
    static RouterPredictor* get();
    bool shadow() const { return shadow_; }
    int  k() const { return k_; }

    /* Once for the process (later calls return the first's answer): every layer's router weight,
     * correction and ffn_norm gain, each read through `read` (the weight's stored bytes). */
    bool load(const Program& prog, const std::function<bool(const WeightInfo&, void*)>& read);
    bool loaded() const { return loaded_.load(std::memory_order_acquire); }

    /* Layer `layer`'s router input for pass `pass` (kEmbd bf16): the worker predicts layer + 1. */
    void post(uint32_t pass, int layer, const uint16_t* x);
    /* The k predicted global experts of `layer` for `pass`, best first; false until ready. */
    bool predicted(uint32_t pass, int layer, int32_t* experts) const;

    struct Stats { uint64_t posted = 0, computed = 0, dropped = 0, compute_ns = 0; };
    Stats stats() const;
    ~RouterPredictor();

private:
    RouterPredictor() = default;
    void loop();
    void compute(uint32_t pass, int layer, const uint16_t* x);

    bool shadow_ = true;
    int  k_ = 10;
    bool avx2_ = false;
    std::once_flag load_once_;
    std::atomic<bool> loaded_{ false };
    bool load_ok_ = false;
    std::vector<uint16_t> w_;      /* [kLayers][kExperts][kEmbd] bf16 */
    std::vector<float>    corr_;   /* [kLayers][kExperts] */
    std::vector<float>    ratio_;  /* [kLayers][kEmbd]: g_{L+1} / g_L, the stand-in's rescale */

    struct Result { uint32_t pass = 0; bool valid = false; int32_t e[kMaxK] = {}; };
    Result res_[kLayers];
    mutable std::mutex res_mu_;

    struct Req { uint32_t pass = 0; int layer = 0; std::vector<uint16_t> x; };
    std::deque<Req> q_;
    std::mutex q_mu_;
    std::condition_variable q_cv_;
    bool stop_ = false;
    std::thread worker_;
    std::atomic<uint64_t> posted_{ 0 }, computed_{ 0 }, dropped_{ 0 }, ns_{ 0 };
};

}  /* namespace rad */
