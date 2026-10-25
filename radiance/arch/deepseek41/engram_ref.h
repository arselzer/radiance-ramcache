/* Engram CPU reference for DeepSeek V4.1-Flash-1017, not a serving plugin.
 * Math: DeepSeek-AI official model 1bae0e1088993b9b82de08f59de5be5a34e42170,
 * inference/engram.py:NgramHashState.forward and inference/model.py:Engram.forward.
 *
 * Hash controls come from the checkpoint, not a replacement RNG/token normalizer.
 * A sequence stores only max_ngram_size-1 compressed tokens. Prefix snapshots carry
 * that history, including DEAD barriers, and the exact immutable model-control owner.
 * Table lookup remains a caller-supplied row operation: this reference never loads
 * or allocates a complete Engram table. There is no convolution in this model.
 */
#pragma once
#include "rad_abi.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace rad::deepseek41::engram {

inline int refuse(std::string* why, int status, const char* text) {
    if (why) *why = text;
    return status;
}
inline bool extent(int64_t a, int64_t b, int64_t* out) {
    if (a < 0 || b < 0 || (a && b > INT64_MAX / a)) return false;
    *out = a * b;
    return uint64_t(*out) <= SIZE_MAX;
}

struct LayerHashControls {
    int layer_id = -1;
    int64_t table_rows = 0;
    // N-gram order first, head second, exactly the checkpoint's row-major I64 controls.
    std::vector<int64_t> primes, offsets, multipliers;
};
struct HashControls {
    // Expected checkpoint + quant + table asset identity; no cross-model cache namespace.
    std::string asset_identity;
    int64_t compressed_vocab_size = 0;
    int64_t raw_pad_id = -1;
    int64_t max_ngram_size = 0, n_heads = 0, head_dim = 0;
    std::vector<int32_t> token_map;
    std::vector<LayerHashControls> layers;
};

class ModelControls {
public:
    static int create(HashControls controls, std::shared_ptr<const ModelControls>* out,
                       std::string* why = nullptr) {
        if (!out) return refuse(why, RAD_E_INVAL, "missing Engram controls destination");
        if (controls.asset_identity.empty() || controls.compressed_vocab_size <= 0 ||
            controls.compressed_vocab_size > INT32_MAX || controls.raw_pad_id < 0 ||
            uint64_t(controls.raw_pad_id) >= controls.token_map.size() ||
            controls.max_ngram_size < 2 || controls.max_ngram_size > 64 ||
            controls.n_heads <= 0 || controls.head_dim <= 0 || controls.layers.empty())
            return refuse(why, RAD_E_SHAPE, "invalid Engram hash geometry or asset identity");
        int64_t columns = 0, embed_dim = 0;
        if (!extent(controls.max_ngram_size - 1, controls.n_heads, &columns) ||
            !extent(columns, controls.head_dim, &embed_dim))
            return refuse(why, RAD_E_SHAPE, "Engram embedding geometry overflows");
        if (uint64_t(controls.compressed_vocab_size) > controls.token_map.size())
            return refuse(why, RAD_E_SHAPE, "Engram compressed vocabulary exceeds token_map");
        std::vector<uint8_t> present(size_t(controls.compressed_vocab_size), 0);
        for (int32_t token : controls.token_map) {
            if (token < 0 || token >= controls.compressed_vocab_size)
                return refuse(why, RAD_E_FORMAT, "Engram token map leaves its compressed vocabulary");
            present[size_t(token)] = 1;
        }
        if (std::find(present.begin(), present.end(), uint8_t(0)) != present.end())
            return refuse(why, RAD_E_FORMAT, "Engram compressed vocabulary has unused IDs");
        for (size_t li = 0; li < controls.layers.size(); ++li) {
            const auto& layer = controls.layers[li];
            if (layer.layer_id < 0 || layer.primes.size() != uint64_t(columns) ||
                layer.offsets.size() != uint64_t(columns) ||
                layer.multipliers.size() != uint64_t(controls.max_ngram_size))
                return refuse(why, RAD_E_SHAPE, "Engram hash control shape mismatch");
            for (size_t prev = 0; prev < li; ++prev)
                if (controls.layers[prev].layer_id == layer.layer_id)
                    return refuse(why, RAD_E_DUPLICATE, "duplicate Engram layer controls");
            int64_t next = 0;
            for (int64_t col = 0; col < columns; ++col) {
                const int64_t prime = layer.primes[size_t(col)];
                if (prime <= 1 || layer.offsets[size_t(col)] != next || prime > INT64_MAX - next)
                    return refuse(why, RAD_E_FORMAT, "Engram bucket ranges overlap, gap or overflow");
                next += prime;
            }
            if (next != layer.table_rows)
                return refuse(why, RAD_E_SHAPE, "Engram table rows do not match its bucket ranges");
            // Official multipliers are positive odd numbers bounded to keep each product
            // below INT64_MAX. Reject incompatible controls rather than invoke signed overflow.
            const int64_t max_token = controls.compressed_vocab_size - 1;
            for (int64_t multiplier : layer.multipliers)
                if (multiplier <= 0 || !(multiplier & 1) ||
                    (max_token && multiplier > INT64_MAX / max_token))
                    return refuse(why, RAD_E_FORMAT, "Engram multiplier violates the int64 product contract");
        }
        *out = std::shared_ptr<const ModelControls>(new ModelControls(std::move(controls), columns));
        return RAD_OK;
    }
    const HashControls& get() const { return controls_; }
    int64_t columns() const { return columns_; }
    int32_t pad_id() const { return controls_.token_map[size_t(controls_.raw_pad_id)]; }
private:
    ModelControls(HashControls controls, int64_t columns)
        : controls_(std::move(controls)), columns_(columns) {}
    HashControls controls_;
    int64_t columns_;
};

struct HistorySnapshot {
    std::shared_ptr<const ModelControls> model;
    int64_t position = 0;
    std::vector<int32_t> history; // oldest first; DEAD=-1 is retained, not converted to pad
};

class HashState {
public:
    static constexpr int32_t DEAD = -1;
    explicit HashState(std::shared_ptr<const ModelControls> model) : model_(std::move(model)) {}
    int64_t position() const { return position_; }
    size_t history_size() const { return history_.size(); }
    void reset() { position_ = 0; history_.clear(); }
    HistorySnapshot snapshot() const { return {model_, position_, history_}; }
    int restore(const HistorySnapshot& snapshot, std::string* why = nullptr) {
        if (!model_ || snapshot.model != model_)
            return refuse(why, RAD_E_STATE, "Engram history belongs to different model controls");
        const auto& c = model_->get();
        if (snapshot.position < 0 || snapshot.history.size() !=
            uint64_t(std::min(snapshot.position, c.max_ngram_size - 1)))
            return refuse(why, RAD_E_SHAPE, "Engram snapshot has incomplete history");
        for (int32_t token : snapshot.history)
            if (token != DEAD && (token < 0 || token >= c.compressed_vocab_size))
                return refuse(why, RAD_E_FORMAT, "invalid compressed token in Engram history");
        position_ = snapshot.position;
        history_ = snapshot.history;
        return RAD_OK;
    }
    // Output layout [tokens, layers, (ngram_size-1)*heads]. An absent mask means all
    // text; a zero mask entry is a DEAD barrier, even if its raw token is a real token.
    // Invalid inputs leave both state and output untouched. Prefill and decode use
    // this same operation, so chunk boundaries and prefix restores preserve hashes.
    int consume(std::span<const int32_t> raw_ids, std::span<const uint8_t> token_mask,
                 std::vector<int64_t>* hashes, std::string* why = nullptr) {
        if (!model_ || !hashes) return refuse(why, RAD_E_STATE, "Engram hash state is not configured");
        const auto& c = model_->get();
        if ((!token_mask.empty() && token_mask.size() != raw_ids.size()) ||
            raw_ids.size() > uint64_t(INT64_MAX - position_))
            return refuse(why, RAD_E_SHAPE, "Engram token mask or position mismatch");
        for (int32_t raw : raw_ids)
            if (raw < 0 || uint64_t(raw) >= c.token_map.size())
                return refuse(why, RAD_E_FORMAT, "Engram raw token is outside token_map");
        int64_t stride = 0, count = 0;
        if (!extent(int64_t(c.layers.size()), model_->columns(), &stride) ||
            !extent(int64_t(raw_ids.size()), stride, &count) || uint64_t(count) > SIZE_MAX / sizeof(int64_t))
            return refuse(why, RAD_E_SHAPE, "Engram hash output overflows");
        std::vector<int64_t> result(size_t(count), 0);
        auto history = history_;
        std::vector<int32_t> tokens(size_t(c.max_ngram_size), model_->pad_id());
        for (size_t ti = 0; ti < raw_ids.size(); ++ti) {
            const int32_t current = !token_mask.empty() && !token_mask[ti]
                                      ? DEAD : c.token_map[size_t(raw_ids[ti])];
            bool blocked = false;
            for (int64_t shift = 0; shift < c.max_ngram_size; ++shift) {
                int32_t token = model_->pad_id();
                if (shift == 0) token = current;
                else if (uint64_t(shift) <= history.size()) token = history[history.size() - size_t(shift)];
                else blocked = true;
                blocked = blocked || token == DEAD;
                tokens[size_t(shift)] = blocked ? model_->pad_id() : token;
            }
            for (size_t li = 0; li < c.layers.size(); ++li) {
                const auto& layer = c.layers[li];
                uint64_t rolling = uint64_t(tokens[0]) * uint64_t(layer.multipliers[0]);
                for (int64_t shift = 1; shift < c.max_ngram_size; ++shift) {
                    rolling ^= uint64_t(tokens[size_t(shift)]) * uint64_t(layer.multipliers[size_t(shift)]);
                    for (int64_t head = 0; head < c.n_heads; ++head) {
                        const size_t col = size_t((shift - 1) * c.n_heads + head);
                        result[ti * size_t(stride) + li * size_t(model_->columns()) + col] =
                            int64_t(rolling % uint64_t(layer.primes[col])) + layer.offsets[col];
                    }
                }
            }
            history.push_back(current);
            if (history.size() >= uint64_t(c.max_ngram_size)) history.erase(history.begin());
        }
        position_ += int64_t(raw_ids.size());
        history_ = std::move(history);
        *hashes = std::move(result);
        return RAD_OK;
    }
private:
    std::shared_ptr<const ModelControls> model_;
    int64_t position_ = 0;
    std::vector<int32_t> history_;
};

// Callback serves only requested rows, in supplied order, with duplicate IDs retained.
// It may coalesce reads/cache internally. These are layer-local IDs for ISTA's two
// separate tables; Unsloth's fused table/global offsets need a separate input adapter.
using RowLookup = std::function<int(int layer_id, std::span<const int64_t> rows,
                                    int64_t width, std::span<float> out, std::string* why)>;
inline int gather_rows(const ModelControls& model, size_t layer_index, int64_t tokens,
                        std::span<const int64_t> rows, const RowLookup& lookup,
                        std::span<float> flattened, std::string* why = nullptr) {
    const auto& c = model.get();
    int64_t ids = 0, values = 0;
    if (layer_index >= c.layers.size() || !extent(tokens, model.columns(), &ids) ||
        !extent(ids, c.head_dim, &values) || rows.size() != uint64_t(ids) ||
        flattened.size() != uint64_t(values) || !lookup)
        return refuse(why, RAD_E_SHAPE, "Engram row gather shape mismatch");
    const auto& layer = c.layers[layer_index];
    for (int64_t i = 0; i < ids; ++i) {
        const size_t col = size_t(i % model.columns());
        if (rows[size_t(i)] < layer.offsets[col] ||
            rows[size_t(i)] >= layer.offsets[col] + layer.primes[col])
            return refuse(why, RAD_E_FORMAT, "Engram row lies outside its hash-column bucket");
    }
    return lookup(layer.layer_id, rows, c.head_dim, flattened, why);
}

inline float bf16(float value) {
    uint32_t bits = std::bit_cast<uint32_t>(value);
    if ((bits & 0x7f800000u) == 0x7f800000u) {
        if (bits & 0x007fffffu) bits |= 0x00400000u;
    } else bits += 0x7fffu + ((bits >> 16) & 1u);
    return std::bit_cast<float>(bits & 0xffff0000u);
}
inline float gate_from_dot(float dot, float clamp = 1e-6f) {
    const float rooted = std::copysign(std::sqrt(std::max(std::abs(dot), clamp)), dot);
    if (rooted >= 0.0f) return 1.0f / (1.0f + std::exp(-rooted));
    const float exp = std::exp(rooted);
    return exp / (1.0f + exp);
}

// x: [tokens,copies,dim], projected_kv: [tokens,copies+1,dim]. The last
// projected row is one value shared by all copies. Norms are per-copy over dim.
// q_weight and k_weight: [copies,dim], combined in f32, not applied before RMS.
// With round_bf16=true, only the final residual is rounded as x.to(bfloat16).
inline int gate_residual(int64_t tokens, int64_t copies, int64_t dim,
                          std::span<const float> x, std::span<const float> projected_kv,
                          std::span<const float> q_weight, std::span<const float> k_weight,
                          std::span<const uint8_t> token_mask, float eps, bool round_bf16,
                          std::span<float> out, std::span<float> gates = {},
                          std::string* why = nullptr) {
    int64_t width = 0, nx = 0, kv_width = 0, nkv = 0, ng = 0;
    if (copies <= 0 || dim <= 0 || copies == INT64_MAX || !extent(copies,dim,&width) ||
        !extent(tokens,width,&nx) || !extent(copies+1,dim,&kv_width) ||
        !extent(tokens,kv_width,&nkv) || !extent(tokens,copies,&ng) ||
        x.size()!=uint64_t(nx) || projected_kv.size()!=uint64_t(nkv) ||
        q_weight.size()!=uint64_t(width) || k_weight.size()!=uint64_t(width) || out.size()!=uint64_t(nx) ||
        (!gates.empty() && gates.size()!=uint64_t(ng)) ||
        (!token_mask.empty() && token_mask.size()!=uint64_t(tokens)) || !std::isfinite(eps) || eps<=0)
        return refuse(why,RAD_E_SHAPE,"Engram gate shape or epsilon mismatch");
    auto finite=[](std::span<const float> values) {
        return std::all_of(values.begin(),values.end(),[](float v){return std::isfinite(v);});
    };
    if (!finite(x) || !finite(projected_kv) || !finite(q_weight) || !finite(k_weight))
        return refuse(why,RAD_E_FORMAT,"Engram gate input is non-finite");
    const float inv_dim=1.0f/float(dim), scale=1.0f/std::sqrt(float(dim));
    for (int64_t token=0;token<tokens;++token) {
        const float* value=projected_kv.data()+token*kv_width+width;
        for (int64_t copy=0;copy<copies;++copy) {
            const int64_t offset=token*width+copy*dim;
            const float* key=projected_kv.data()+token*kv_width+copy*dim;
            float hs=0,ks=0,dot=0;
            for(int64_t d=0;d<dim;++d) {
                const float h=x[size_t(offset+d)],k=key[d];
                hs+=h*h;ks+=k*k;
                const float weight=q_weight[size_t(copy*dim+d)]*k_weight[size_t(copy*dim+d)];
                dot+=(h*weight)*k;
            }
            const float rstd=(1.0f/std::sqrt(hs*inv_dim+eps))*(1.0f/std::sqrt(ks*inv_dim+eps));
            dot=(dot*rstd)*scale;
            const float gate=!token_mask.empty() && !token_mask[size_t(token)] ? 0.0f : gate_from_dot(dot);
            if(!gates.empty())gates[size_t(token*copies+copy)]=gate;
            for(int64_t d=0;d<dim;++d) {
                const float residual=x[size_t(offset+d)]+gate*value[d];
                out[size_t(offset+d)]=round_bf16 ? bf16(residual) : residual;
            }
        }
    }
    return RAD_OK;
}

} // namespace rad::deepseek41::engram
