/* CPU mathematical reference for DeepSeek-V4.1 (1017).
 * Algorithm source: official inference/model.py and kernel.py at
 * deepseek-ai/DeepSeek-V4.1-Flash-1017 1bae0e1088993b9b82de08f59de5be5a34e42170.
 * This is an independently written semantic oracle, not a serving implementation or a
 * reproduction of TileLang's BF16 probability/activation quantization. Precision boundaries
 * are explicit. New GPU kernels must be compared with this AND the packed arithmetic path.
 */
#pragma once
#include "rad_abi.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace rad::deepseek41::ref {
using Floats = std::span<const float>;
using Mutable = std::span<float>;
inline int fail(int code, const char* text, std::string* why) {
    if (why) *why = text;
    return code;
}
inline bool extent(std::initializer_list<size_t> dims, size_t* out) {
    size_t n = 1;
    for (auto d : dims) {
        if (d && n > SIZE_MAX / d) return false;
        n *= d;
    }
    *out = n;
    return true;
}
inline bool finite(Floats x) {
    return std::all_of(x.begin(), x.end(), [](float v) { return std::isfinite(v); });
}
enum class Precision { f32, bf16 };
inline float rounded(float x, Precision precision) {
    if (precision == Precision::f32) return x;
    uint32_t bits = std::bit_cast<uint32_t>(x);
    if ((bits & 0x7fffffffU) > 0x7f800000U) return std::bit_cast<float>((bits & 0xffff0000U) | 0x00400000U);
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return std::bit_cast<float>(bits & 0xffff0000U);
}
inline float sigmoid(float x) {
    if (x >= 0) return 1.f / (1.f + std::exp(-x));
    const float e = std::exp(x);
    return e / (1.f + e);
}
inline float softplus(float x) { return x > 20.f ? x : std::log1p(std::exp(x)); }
inline int rms_norm(Floats x, Floats weight, float eps, Mutable out,
                     Precision precision = Precision::f32, std::string* why = nullptr) {
    if (x.empty() || weight.size() != x.size() || out.size() != x.size())
        return fail(RAD_E_SHAPE, "RMSNorm shape mismatch", why);
    if (!(eps > 0) || !std::isfinite(eps) || !finite(x) || !finite(weight))
        return fail(RAD_E_INVAL, "nonfinite RMSNorm input or invalid epsilon", why);
    float sum = 0;
    for (float value : x) sum += value * value;
    const float scale = 1.f / std::sqrt(sum / float(x.size()) + eps);
    for (size_t i = 0; i < x.size(); ++i) out[i] = rounded((x[i] * scale) * weight[i], precision);
    return RAD_OK;
}
/* Row-major [out,in] weights. Accumulation order is explicit; no hidden activation quantizer. */
inline int linear(Floats x, Floats weight, size_t outputs, Mutable out,
                   Precision precision = Precision::f32, std::string* why = nullptr) {
    size_t elements = 0;
    if (x.empty() || !outputs || !extent({outputs, x.size()}, &elements) ||
        weight.size() != elements || out.size() != outputs)
        return fail(RAD_E_SHAPE, "linear shape mismatch", why);
    if (!finite(x) || !finite(weight)) return fail(RAD_E_INVAL, "nonfinite linear input", why);
    for (size_t r = 0; r < outputs; ++r) {
        float value = 0;
        for (size_t c = 0; c < x.size(); ++c) value += x[c] * weight[r * x.size() + c];
        out[r] = rounded(value, precision);
    }
    return RAD_OK;
}

struct MhcMix {
    size_t copies = 0;
    std::vector<float> pre, post, comb; // comb[source,destination]
};
inline int mhc_prepare(Floats h, size_t dim, size_t copies, Floats fn, Floats scale,
                        Floats base, int iterations, float norm_eps, float hc_eps,
                        MhcMix* out, std::string* why = nullptr) {
    size_t width = 0, mixes = 0, elements = 0;
    if (!out || !dim || copies < 2 || copies > 8 || !extent({copies, dim}, &width) ||
        !extent({copies, copies + 2}, &mixes) || !extent({mixes, width}, &elements) ||
        h.size() != width || fn.size() != elements || scale.size() != 3 || base.size() != mixes ||
        iterations < 1 || iterations > 256)
        return fail(RAD_E_SHAPE, "mHC shape or iteration mismatch", why);
    if (!(norm_eps > 0) || hc_eps < 0 || !std::isfinite(norm_eps) || !std::isfinite(hc_eps) ||
        !finite(h) || !finite(fn) || !finite(scale) || !finite(base))
        return fail(RAD_E_INVAL, "nonfinite mHC input or invalid epsilon", why);
    float square = 0;
    for (float v : h) square += v * v;
    const float inverse_rms = 1.f / std::sqrt(square / float(width) + norm_eps);
    std::vector<float> coeff(mixes);
    for (size_t r = 0; r < mixes; ++r) {
        float value = 0;
        for (size_t c = 0; c < width; ++c) value += h[c] * fn[r * width + c];
        coeff[r] = value * inverse_rms;
    }
    MhcMix result;
    result.copies = copies; result.pre.resize(copies); result.post.resize(copies);
    result.comb.resize(copies * copies);
    for (size_t i = 0; i < copies; ++i) {
        result.pre[i] = sigmoid(coeff[i] * scale[0] + base[i]) + hc_eps;
        result.post[i] = 2.f * sigmoid(coeff[copies + i] * scale[1] + base[copies + i]);
    }
    for (size_t src = 0; src < copies; ++src) {
        float maximum = -INFINITY;
        for (size_t dst = 0; dst < copies; ++dst) {
            const size_t index = 2 * copies + src * copies + dst;
            auto& v = result.comb[src * copies + dst];
            v = coeff[index] * scale[2] + base[index]; maximum = std::max(maximum, v);
        }
        float sum = 0;
        for (size_t dst = 0; dst < copies; ++dst) {
            auto& v = result.comb[src * copies + dst]; v = std::exp(v - maximum); sum += v;
        }
        for (size_t dst = 0; dst < copies; ++dst) result.comb[src * copies + dst] = result.comb[src * copies + dst] / sum + hc_eps;
    }
    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (iteration) for (size_t src = 0; src < copies; ++src) {
            float sum = 0;
            for (size_t dst = 0; dst < copies; ++dst) sum += result.comb[src * copies + dst];
            for (size_t dst = 0; dst < copies; ++dst) result.comb[src * copies + dst] /= sum + hc_eps;
        }
        for (size_t dst = 0; dst < copies; ++dst) {
            float sum = 0;
            for (size_t src = 0; src < copies; ++src) sum += result.comb[src * copies + dst];
            for (size_t src = 0; src < copies; ++src) result.comb[src * copies + dst] /= sum + hc_eps;
        }
    }
    *out = std::move(result);
    return RAD_OK;
}
inline int mhc_collapse(Floats h, Floats previous_pre, Mutable out,
                         Precision precision = Precision::f32, std::string* why = nullptr) {
    size_t total = 0;
    if (out.empty() || previous_pre.empty() || !extent({out.size(), previous_pre.size()}, &total) || h.size() != total)
        return fail(RAD_E_SHAPE, "mHC collapse shape mismatch", why);
    if (!finite(h) || !finite(previous_pre)) return fail(RAD_E_INVAL, "nonfinite mHC collapse input", why);
    for (size_t d = 0; d < out.size(); ++d) {
        float value = 0;
        for (size_t copy = 0; copy < previous_pre.size(); ++copy) value += previous_pre[copy] * h[copy * out.size() + d];
        out[d] = rounded(value, precision);
    }
    return RAD_OK;
}
inline int mhc_expand(Floats y, Floats residual, const MhcMix& mix, Mutable out,
                       Precision precision = Precision::f32, std::string* why = nullptr) {
    size_t total = 0;
    if (y.empty() || !mix.copies || !extent({y.size(), mix.copies}, &total) || residual.size() != total ||
        out.size() != total || mix.post.size() != mix.copies || mix.comb.size() != mix.copies * mix.copies)
        return fail(RAD_E_SHAPE, "mHC expansion shape mismatch", why);
    if (!finite(y) || !finite(residual) || !finite(mix.post) || !finite(mix.comb))
        return fail(RAD_E_INVAL, "nonfinite mHC expansion input", why);
    // Stage the result: out may alias residual, but must not overwrite another source copy.
    std::vector<float> result(total);
    for (size_t dst = 0; dst < mix.copies; ++dst) for (size_t d = 0; d < y.size(); ++d) {
        float carried = 0;
        for (size_t src = 0; src < mix.copies; ++src) carried += mix.comb[src * mix.copies + dst] * residual[src * y.size() + d];
        result[dst * y.size() + d] = rounded(mix.post[dst] * y[d] + carried, precision);
    }
    std::copy(result.begin(), result.end(), out.begin());
    return RAD_OK;
}
/* One shifted sublayer: current coefficients govern the residual write and NEXT collapse.
 * The supplied previous_pre governs THIS collapse. Same rule applies to the final head. */
inline int shifted_sublayer(Floats h, Floats previous_pre, const MhcMix& current,
                            const std::function<int(Floats, Mutable)>& sublayer, Mutable out,
                            Precision precision = Precision::f32, std::string* why = nullptr) {
    if (!current.copies || previous_pre.size() != current.copies || h.size() % current.copies || !sublayer)
        return fail(RAD_E_SHAPE, "invalid shifted sublayer", why);
    std::vector<float> x(h.size() / current.copies), y(x.size());
    int status = mhc_collapse(h, previous_pre, x, precision, why);
    if (status == RAD_OK) status = sublayer(x, y);
    if (status == RAD_OK) status = mhc_expand(y, h, current, out, precision, why);
    return status;
}

struct Routing { std::vector<int32_t> ids; std::vector<float> weights; };
inline int route(Floats logits, Floats correction, size_t topk, float temperature,
                  float route_scale, bool normalize, Routing* out, std::string* why = nullptr) {
    if (!out || !topk || topk > logits.size() || logits.size() > INT32_MAX || correction.size() != logits.size())
        return fail(RAD_E_SHAPE, "routing shape mismatch", why);
    if (!(temperature > 0) || !std::isfinite(temperature) || !(route_scale > 0) || !std::isfinite(route_scale) ||
        !finite(logits) || !finite(correction)) return fail(RAD_E_INVAL, "invalid routing input", why);
    std::vector<float> unbiased(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) unbiased[i] = std::sqrt(softplus(logits[i] / temperature));
    Routing result; result.ids.resize(logits.size()); std::iota(result.ids.begin(), result.ids.end(), 0);
    std::partial_sort(result.ids.begin(), result.ids.begin() + topk, result.ids.end(), [&](int32_t a, int32_t b) {
        const float sa = unbiased[a] + correction[a], sb = unbiased[b] + correction[b];
        return sa != sb ? sa > sb : a < b; // deterministic oracle tie; framework topk ties are unspecified
    });
    result.ids.resize(topk); result.weights.resize(topk);
    float total = 0;
    for (size_t i = 0; i < topk; ++i) { result.weights[i] = unbiased[result.ids[i]]; total += result.weights[i]; }
    for (auto& weight : result.weights) {
        if (normalize && topk > 1) weight /= total + 1e-20f;
        weight *= route_scale;
    }
    *out = std::move(result); return RAD_OK;
}
inline int swiglu(Floats gate, Floats up, float limit, Mutable out,
                   Precision precision = Precision::f32, std::string* why = nullptr) {
    if (gate.size() != up.size() || out.size() != gate.size()) return fail(RAD_E_SHAPE, "SwiGLU shape mismatch", why);
    if (!finite(gate) || !finite(up) || !std::isfinite(limit) || limit < 0) return fail(RAD_E_INVAL, "invalid SwiGLU input", why);
    for (size_t i = 0; i < gate.size(); ++i) {
        const float g = limit > 0 ? std::min(gate[i], limit) : gate[i];
        const float u = limit > 0 ? std::clamp(up[i], -limit, limit) : up[i];
        out[i] = rounded((g * sigmoid(g)) * u, precision);
    }
    return RAD_OK;
}

struct Rope { size_t rotary_dim = 0; float theta = 10000, factor = 1, beta_fast = 32, beta_slow = 1; int64_t original_context = 0; };
inline int rotary(Mutable x, int64_t position, const Rope& cfg, bool inverse = false, std::string* why = nullptr) {
    if (!cfg.rotary_dim || cfg.rotary_dim % 2 || cfg.rotary_dim > x.size() || position < 0 ||
        !(cfg.theta > 1) || !(cfg.factor > 0) || !(cfg.beta_fast > 0) || !(cfg.beta_slow > 0) ||
        !std::isfinite(cfg.theta) || !std::isfinite(cfg.factor) || !std::isfinite(cfg.beta_fast) ||
        !std::isfinite(cfg.beta_slow) || cfg.original_context < 0 || !finite(x))
        return fail(RAD_E_INVAL, "invalid rotary input or configuration", why);
    float low = 0, high = 0;
    if (cfg.original_context) {
        auto corrected = [&](float rotations) {
            return double(cfg.rotary_dim) * std::log(double(cfg.original_context) / (double(rotations) * 2 * std::acos(-1.))) /
                         (2 * std::log(double(cfg.theta)));
        };
        low = float(std::max(std::floor(corrected(cfg.beta_fast)), 0.));
        high = float(std::min(std::ceil(corrected(cfg.beta_slow)), double(cfg.rotary_dim - 1)));
    }
    const size_t start = x.size() - cfg.rotary_dim;
    for (size_t pair = 0; pair < cfg.rotary_dim / 2; ++pair) {
        float freq = 1.f / std::pow(cfg.theta, float(2 * pair) / float(cfg.rotary_dim));
        if (cfg.original_context) {
            const float smooth = 1.f - std::clamp((float(pair) - low) / std::max(high - low, 1e-3f), 0.f, 1.f);
            freq = (freq / cfg.factor) * (1.f - smooth) + freq * smooth;
        }
        const float angle = float(position) * freq * (inverse ? -1.f : 1.f);
        const float cosine = std::cos(angle), sine = std::sin(angle);
        const float a = x[start + 2 * pair], b = x[start + 2 * pair + 1];
        x[start + 2 * pair] = a * cosine - b * sine;
        x[start + 2 * pair + 1] = a * sine + b * cosine;
    }
    return RAD_OK;
}

/* kv supplies both keys and values, each of width dim. sink is a logit on a virtual zero value.
 * -1 is the only padding index; invalid positive indices fail rather than aliasing another row. */
inline int sparse_attention(Floats query, Floats kv, size_t heads, size_t dim,
                             Floats sinks, std::span<const int32_t> selected, float scale,
                             Mutable out, std::string* why = nullptr) {
    size_t qsize = 0;
    if (!heads || !dim || !extent({heads, dim}, &qsize) || query.size() != qsize || out.size() != qsize ||
        kv.size() % dim || sinks.size() != heads) return fail(RAD_E_SHAPE, "attention shape mismatch", why);
    const size_t rows = kv.size() / dim;
    if (!(scale > 0) || !std::isfinite(scale) || !finite(query)) return fail(RAD_E_INVAL, "invalid attention input", why);
    for (float sink : sinks) if (!std::isfinite(sink) && sink != -INFINITY) return fail(RAD_E_INVAL, "invalid attention sink", why);
    for (auto id : selected) if (id < -1 || (id >= 0 && size_t(id) >= rows)) return fail(RAD_E_SHAPE, "attention index out of range", why);
    for (auto id : selected) if (id >= 0 && !finite(kv.subspan(size_t(id) * dim, dim)))
        return fail(RAD_E_INVAL, "nonfinite selected attention KV", why);
    std::vector<float> result(qsize, 0.f), scores(selected.size());
    for (size_t head = 0; head < heads; ++head) {
        float maximum = sinks[head];
        for (size_t i = 0; i < selected.size(); ++i) {
            if (selected[i] < 0) { scores[i] = -INFINITY; continue; }
            float dot = 0;
            for (size_t d = 0; d < dim; ++d) dot += query[head * dim + d] * kv[size_t(selected[i]) * dim + d];
            scores[i] = dot * scale; maximum = std::max(maximum, scores[i]);
        }
        if (maximum == -INFINITY) continue; // no valid keys and disabled sink -> training zero convention
        float denominator = std::exp(sinks[head] - maximum);
        for (size_t i = 0; i < selected.size(); ++i) if (selected[i] >= 0) {
            const float probability = std::exp(scores[i] - maximum); denominator += probability;
            for (size_t d = 0; d < dim; ++d) result[head * dim + d] += probability * kv[size_t(selected[i]) * dim + d];
        }
        for (size_t d = 0; d < dim; ++d) result[head * dim + d] /= denominator;
    }
    std::copy(result.begin(), result.end(), out.begin()); return RAD_OK;
}
inline int index_scores(Floats query, Floats keys, Floats head_weights, size_t dim,
                         size_t global_heads, Mutable scores, std::string* why = nullptr) {
    const size_t heads = head_weights.size();
    size_t count = 0;
    if (!dim || !heads || global_heads < heads || !extent({heads, dim}, &count) || query.size() != count ||
        keys.size() % dim || scores.size() != keys.size() / dim) return fail(RAD_E_SHAPE, "index score shape mismatch", why);
    if (!finite(query) || !finite(keys) || !finite(head_weights)) return fail(RAD_E_INVAL, "invalid index score input", why);
    const float scale = (1.f / std::sqrt(float(dim))) * (1.f / std::sqrt(float(global_heads)));
    for (size_t pos = 0; pos < scores.size(); ++pos) {
        float total = 0;
        for (size_t head = 0; head < heads; ++head) {
            float dot = 0;
            for (size_t d = 0; d < dim; ++d) dot += query[head * dim + d] * keys[pos * dim + d];
            total += std::max(dot, 0.f) * (head_weights[head] * scale);
        }
        scores[pos] = total;
    }
    return RAD_OK;
}
inline int candidate_blocks(Floats scores, size_t visible, size_t block_size,
                             size_t top_blocks, std::span<uint8_t> keep, std::string* why = nullptr) {
    if (!block_size || !top_blocks || visible > scores.size() || keep.size() != scores.size())
        return fail(RAD_E_SHAPE, "candidate block shape mismatch", why);
    for (size_t i = 0; i < visible; ++i) if (!std::isfinite(scores[i])) return fail(RAD_E_INVAL, "invalid candidate score", why);
    std::fill(keep.begin(), keep.end(), 0);
    if (!visible) return RAD_OK;
    const size_t count = visible / block_size + (visible % block_size != 0), newest = (visible - 1) / block_size;
    std::vector<float> best(count, -INFINITY); std::vector<size_t> blocks(count); std::iota(blocks.begin(), blocks.end(), 0);
    for (size_t pos = 0; pos < visible; ++pos) best[pos / block_size] = std::max(best[pos / block_size], scores[pos]);
    best[newest] = INFINITY;
    const size_t selected = std::min(top_blocks, count);
    std::partial_sort(blocks.begin(), blocks.begin() + selected, blocks.end(), [&](size_t a, size_t b) {
        return best[a] != best[b] ? best[a] > best[b] : a < b;
    });
    // The official mask includes the full selected block, including its not-yet-visible tail;
    // visibility is a separate mask applied during top-k/attention.
    for (size_t b = 0; b < selected; ++b) {
        const size_t begin = blocks[b] * block_size;
        const size_t end = begin + std::min(block_size, scores.size() - begin);
        std::fill(keep.begin() + begin, keep.begin() + end, 1);
    }
    return RAD_OK;
}
inline int index_topk(Floats scores, size_t visible, size_t topk, std::span<const uint8_t> candidates,
                       std::vector<int32_t>* out, std::string* why = nullptr) {
    if (!out || !topk || visible > scores.size() || scores.size() > INT32_MAX ||
        (!candidates.empty() && candidates.size() != scores.size())) return fail(RAD_E_SHAPE, "index top-k shape mismatch", why);
    std::vector<int32_t> ids;
    for (size_t i = 0; i < visible; ++i) {
        if (!std::isfinite(scores[i])) return fail(RAD_E_INVAL, "invalid visible index score", why);
        if (candidates.empty() || candidates[i]) ids.push_back(int32_t(i));
    }
    const size_t count = std::min(topk, ids.size());
    std::partial_sort(ids.begin(), ids.begin() + count, ids.end(), [&](int32_t a, int32_t b) {
        return scores[a] != scores[b] ? scores[a] > scores[b] : a < b;
    });
    ids.resize(count); std::sort(ids.begin(), ids.end());
    ids.resize(std::min(topk, scores.size()), -1); *out = std::move(ids); return RAD_OK;
}

/* Grouped wo_a is block-diagonal over heads, followed by dense wo_b. */
inline int grouped_output(Floats heads, size_t n_heads, size_t head_dim, size_t groups,
                            size_t output_rank, Floats wa, Floats wb, Mutable out,
                            Precision precision = Precision::f32, std::string* why = nullptr) {
    size_t count = 0, wa_size = 0, intermediate = 0, wb_size = 0;
    if (!n_heads || !head_dim || !groups || n_heads % groups || !output_rank || out.empty() ||
        !extent({n_heads, head_dim}, &count) || heads.size() != count ||
        !extent({groups, output_rank}, &intermediate) ||
        !extent({intermediate, count / groups}, &wa_size) || wa.size() != wa_size ||
        !extent({out.size(), intermediate}, &wb_size) || wb.size() != wb_size)
        return fail(RAD_E_SHAPE, "grouped output shape mismatch", why);
    std::vector<float> latent(intermediate);
    const size_t width = count / groups;
    for (size_t group = 0; group < groups; ++group) {
        const int status = linear(heads.subspan(group * width, width),
                                   wa.subspan(group * output_rank * width, output_rank * width),
                                   output_rank, Mutable(latent).subspan(group * output_rank, output_rank), precision, why);
        if (status != RAD_OK) return status;
    }
    return linear(latent, wb, out.size(), out, precision, why);
}

/* The ring is independent per sequence and layer. Its checkpoint includes wrap position;
 * restoring only its payload would attend to the wrong tokens after prefix reuse. */
class WindowCache {
public:
    struct Snapshot {
        std::string identity;
        size_t window = 0, dim = 0;
        int64_t next_position = 0;
        std::vector<float> slots;
    };
    WindowCache(size_t window, size_t dim, std::string identity)
        : window_(window), dim_(dim), identity_(std::move(identity)) {
        size_t count = 0;
        if (!window || !dim || window > INT32_MAX || identity_.empty() || !extent({window, dim}, &count))
            throw std::invalid_argument("invalid DeepSeek window cache");
        slots_.resize(count, 0.f);
    }
    int append(int64_t position, Floats kv, std::string* why = nullptr) {
        if (position != next_position_ || position < 0 || position == INT64_MAX)
            return fail(RAD_E_STATE, "window position must advance by one", why);
        if (kv.size() != dim_) return fail(RAD_E_SHAPE, "window KV shape mismatch", why);
        if (!finite(kv)) return fail(RAD_E_INVAL, "nonfinite window KV", why);
        std::copy(kv.begin(), kv.end(), slots_.begin() + (size_t(position) % window_) * dim_);
        ++next_position_; return RAD_OK;
    }
    std::vector<int32_t> selected() const {
        std::vector<int32_t> ids;
        const int64_t first = std::max<int64_t>(0, next_position_ - int64_t(window_));
        for (int64_t position = first; position < next_position_; ++position)
            ids.push_back(int32_t(position % int64_t(window_)));
        return ids;
    }
    Floats values() const { return slots_; }
    int64_t next_position() const { return next_position_; }
    Snapshot snapshot() const { return {identity_, window_, dim_, next_position_, slots_}; }
    int restore(const Snapshot& saved, std::string* why = nullptr) {
        if (saved.identity != identity_ || saved.window != window_ || saved.dim != dim_)
            return fail(RAD_E_INVAL, "window snapshot belongs to another sequence/layer", why);
        if (saved.next_position < 0 || saved.slots.size() != slots_.size() || !finite(saved.slots))
            return fail(RAD_E_FORMAT, "invalid window snapshot", why);
        slots_ = saved.slots; next_position_ = saved.next_position; return RAD_OK;
    }
private:
    size_t window_, dim_;
    std::string identity_;
    std::vector<float> slots_;
    int64_t next_position_ = 0;
};

/* A bounded per-source compressor. Its partial group is sequence/prefix state, never global.
 * Projection weights and the pre-RoPE latent are deliberately outside this state container. */
class Compressor {
public:
    struct Snapshot {
        std::string identity;
        size_t ratio = 0, dim = 0, pending = 0;
        int64_t next_position = 0;
        std::vector<float> kv, scores;
    };
    Compressor(size_t ratio, Floats norm_weight, float norm_eps, std::string identity,
                 Precision precision = Precision::f32)
        : ratio_(ratio), dim_(norm_weight.size()), norm_(norm_weight.begin(), norm_weight.end()),
          eps_(norm_eps), identity_(std::move(identity)), precision_(precision) {
        if ((ratio != 1 && ratio != 2) || !dim_ || identity_.empty() || !finite(norm_) || !(eps_ > 0) || !std::isfinite(eps_))
            throw std::invalid_argument("invalid DeepSeek compressor configuration");
        kv_.resize(ratio_ * dim_); scores_.resize(ratio_ * dim_);
    }
    int push(Floats kv, Floats score, std::vector<float>* latent, int64_t* first_position, std::string* why = nullptr) {
        if (!latent || !first_position || kv.size() != dim_ || (ratio_ == 2 && score.size() != dim_) ||
            (ratio_ == 1 && !score.empty())) return fail(RAD_E_SHAPE, "compressor input shape mismatch", why);
        if (!finite(kv) || !finite(score) || next_position_ == INT64_MAX) return fail(RAD_E_INVAL, "invalid compressor input", why);
        const bool complete = pending_ + 1 == ratio_;
        std::vector<float> normalized;
        if (complete) {
            std::vector<float> pooled(dim_);
            for (size_t d = 0; d < dim_; ++d) {
                float value = kv[d];
                if (ratio_ == 2) {
                    const float maximum = std::max(scores_[d], score[d]);
                    const float a = std::exp(scores_[d] - maximum), b = std::exp(score[d] - maximum);
                    value = (kv_[d] * (a / (a + b))) + (kv[d] * (b / (a + b)));
                }
                pooled[d] = rounded(value, precision_);
            }
            normalized.resize(dim_);
            const int status = rms_norm(pooled, norm_, eps_, normalized, precision_, why);
            if (status != RAD_OK) return status;
        }
        if (!complete) {
            std::copy(kv.begin(), kv.end(), kv_.begin() + pending_ * dim_);
            std::copy(score.begin(), score.end(), scores_.begin() + pending_ * dim_);
        }
        ++next_position_; pending_ = complete ? 0 : pending_ + 1;
        *first_position = complete ? next_position_ - int64_t(ratio_) : -1;
        *latent = std::move(normalized); return RAD_OK;
    }
    Snapshot snapshot() const {
        return {identity_, ratio_, dim_, pending_, next_position_,
                {kv_.begin(), kv_.begin() + pending_ * dim_}, {scores_.begin(), scores_.begin() + pending_ * dim_}};
    }
    int restore(const Snapshot& saved, std::string* why = nullptr) {
        if (saved.identity != identity_ || saved.ratio != ratio_ || saved.dim != dim_)
            return fail(RAD_E_INVAL, "compressor snapshot belongs to another source", why);
        if (saved.next_position < 0 || saved.pending >= ratio_ || saved.pending != size_t(saved.next_position % int64_t(ratio_)) ||
            saved.kv.size() != saved.pending * dim_ || saved.scores.size() != saved.pending * dim_ ||
            !finite(saved.kv) || !finite(saved.scores)) return fail(RAD_E_FORMAT, "invalid compressor snapshot", why);
        std::fill(kv_.begin(), kv_.end(), 0); std::fill(scores_.begin(), scores_.end(), 0);
        std::copy(saved.kv.begin(), saved.kv.end(), kv_.begin()); std::copy(saved.scores.begin(), saved.scores.end(), scores_.begin());
        pending_ = saved.pending; next_position_ = saved.next_position; return RAD_OK;
    }
    int64_t next_position() const { return next_position_; }
private:
    size_t ratio_, dim_, pending_ = 0;
    std::vector<float> norm_, kv_, scores_;
    float eps_;
    std::string identity_;
    Precision precision_;
    int64_t next_position_ = 0;
};
} // namespace rad::deepseek41::ref
