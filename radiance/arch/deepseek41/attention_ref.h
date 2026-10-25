/* Small serial FP32 attention reference for DeepSeek V4.1-Flash-1017. This composes the explicit
 * mathematical helpers; it is not a serving plugin or an optimized/packed numerical path.
 * In particular, the official FP8 window KV, FP4 index/latent quantizers and output activation
 * quantizer are intentionally absent. GPU arithmetic needs a separate packed-path quality gate.
 * Sources: official inference/model.py at 1bae0e1088993b9b82de08f59de5be5a34e42170,
 * DeepInfer source topology at 9bd9cf243fc53ef9afbf8f04d391d989117d6ab9.
 */
#pragma once
#include "config.h"
#include "../common/rad_ds41_ref.h"
#include <memory>
#include <optional>
#include <stdexcept>

namespace rad::deepseek41::attention {
using ref::Floats;
using ref::Mutable;

struct LayerWeights {
    // All matrices [output,input], with wo_a logically [group,rank,group_head_width].
    std::vector<float> q_a, q_norm, q_b, kv, kv_norm, out_a, out_b, sinks;
    std::vector<float> compress_kv, compress_gate, compress_norm; // only KV producers
    std::vector<float> index_q, index_head_weights;              // only top-k producers
    std::vector<float> index_k, index_k_norm;                    // only index-K producers
};
struct Trace {
    std::vector<float> q_latent, query, window_row, compressed_latent, compressed_row;
    std::vector<float> index_query, index_key, scores, attended, unrotated_heads, output;
    std::vector<int32_t> compressed_indices, window_indices, combined_indices;
    std::vector<uint8_t> candidates;
    int64_t compressed_first_position = -1;
};

class Model {
public:
    static int create(Config config, std::vector<LayerWeights> weights, std::string identity,
                       std::shared_ptr<const Model>* out, std::string* why = nullptr) {
        using namespace config_detail;
        if (!out || identity.empty()) return fail(why,"missing attention model identity or destination");
        int status=validate_config(&config,why);
        if (status!=RAD_OK)return status;
        if (weights.size()!=size_t(config.n_layers))return fail(why,"attention weight list does not cover all layers",RAD_E_SHAPE);
        const size_t d=size_t(config.n_embd),hd=size_t(config.head_dim),qr=size_t(config.q_lora_rank);
        const size_t heads=size_t(config.n_head),groups=size_t(config.o_groups),orank=size_t(config.o_lora_rank);
        const size_t ih=size_t(config.index_n_heads),id=size_t(config.index_head_dim);
        for (int layer=0;layer<config.n_layers;++layer) {
            const auto& w=weights[layer];const auto& s=config.layers[layer];
            auto sized=[&](const std::vector<float>& values,std::initializer_list<size_t> shape) {
                size_t n=0;return ref::extent(shape,&n)&&values.size()==n&&ref::finite(values);
            };
            if (!sized(w.q_a,{qr,d}) || !sized(w.q_norm,{qr}) || !sized(w.q_b,{heads,hd,qr}) ||
                !sized(w.kv,{hd,d}) || !sized(w.kv_norm,{hd}) || !sized(w.out_a,{groups,orank,heads/groups,hd}) ||
                !sized(w.out_b,{d,groups,orank}) || w.sinks.size()!=heads)
                return fail(why,"attention projection shape or finite-value mismatch",RAD_E_SHAPE);
            for (float sink:w.sinks) if (!std::isfinite(sink)&&sink!=-INFINITY)
                return fail(why,"invalid attention sink",RAD_E_INVAL);
            if (s.writes_kv(layer)) {
                if (!sized(w.compress_kv,{hd,d}) || !sized(w.compress_norm,{hd}) ||
                    (s.compress_ratio==2 ? !sized(w.compress_gate,{hd,d}) : !w.compress_gate.empty()))
                    return fail(why,"compressor projection shape mismatch",RAD_E_SHAPE);
            } else if (!w.compress_kv.empty() || !w.compress_gate.empty() || !w.compress_norm.empty()) {
                return fail(why,"compressor weights supplied for a KV consumer",RAD_E_SHAPE);
            }
            if (s.writes_index(layer)) {
                if (!sized(w.index_q,{ih,id,qr}) || !sized(w.index_head_weights,{ih,d}))
                    return fail(why,"index-query projection shape mismatch",RAD_E_SHAPE);
            } else if (!w.index_q.empty() || !w.index_head_weights.empty()) {
                return fail(why,"index-query weights supplied for a top-k consumer",RAD_E_SHAPE);
            }
            if (s.writes_index_k(layer)) {
                if (!sized(w.index_k,{id,hd}) || !sized(w.index_k_norm,{id}))
                    return fail(why,"index-key projection shape mismatch",RAD_E_SHAPE);
            } else if (!w.index_k.empty() || !w.index_k_norm.empty()) {
                return fail(why,"index-key weights supplied for an index-K consumer",RAD_E_SHAPE);
            }
        }
        auto model=std::shared_ptr<Model>(new Model);
        model->config_=std::move(config);model->weights_=std::move(weights);model->identity_=std::move(identity);
        *out=std::move(model);if(why)why->clear();
        return RAD_OK;
    }
    const Config& config() const { return config_; }
    const std::string& identity() const { return identity_; }
    const LayerWeights& layer(int i) const { return weights_.at(size_t(i)); }
private:
    Config config_;
    std::vector<LayerWeights> weights_;
    std::string identity_;
};

class Sequence {
public:
    struct Snapshot {
        std::shared_ptr<const Model> model;
        std::string sequence_identity;
        int64_t position=0;
        std::vector<ref::WindowCache::Snapshot> windows;
        std::vector<std::optional<ref::Compressor::Snapshot>> compressors;
        std::vector<std::vector<float>> compressed_kv, index_keys;
    };
    Sequence(std::shared_ptr<const Model> model,std::string sequence_identity,int64_t context_limit)
        : model_(std::move(model)),sequence_identity_(std::move(sequence_identity)),limit_(context_limit) {
        if (!model_ || sequence_identity_.empty() || limit_<=0 || limit_>model_->config().context_length)
            throw std::invalid_argument("invalid DeepSeek attention sequence");
        const auto& c=model_->config();
        for (int layer=0;layer<c.n_layers;++layer) {
            const auto identity=model_->identity()+"/"+sequence_identity_+"/layer/"+std::to_string(layer);
            windows_.emplace_back(size_t(c.window_size),size_t(c.head_dim),identity+"/window");
            if (c.layers[layer].writes_kv(layer))
                compressors_.emplace_back(std::in_place,size_t(c.layers[layer].compress_ratio),
                    model_->layer(layer).compress_norm,float(c.norm_eps),identity+"/compressor");
            else compressors_.emplace_back();
        }
        compressed_kv_.resize(size_t(c.n_layers));index_keys_.resize(size_t(c.n_layers));
        frame_.resize(size_t(c.n_layers));
    }
    int64_t next_position() const { return position_; }
    int next_layer() const { return next_layer_; }
    const std::shared_ptr<const Model>& model() const { return model_; }

    /* A token traverses layers 0..n_layers-1, then position advances. Stage all changes so a bad
     * weight/input/state cannot poison later requests. Copying caches is intentional in this tiny
     * oracle; the serving implementation needs scoped ownership and asynchronous lifetimes. */
    int step(int layer,int64_t position,Floats normalized_input,Mutable output,
              Trace* trace=nullptr,std::string* why=nullptr) {
        if (layer!=next_layer_ || layer<0 || layer>=model_->config().n_layers || position!=position_ ||
            position<0 || position>=limit_)
            return ref::fail(RAD_E_STATE,"attention sequence/layer order or context bound mismatch",why);
        if (normalized_input.size()!=size_t(model_->config().n_embd) || output.size()!=normalized_input.size())
            return ref::fail(RAD_E_SHAPE,"attention input/output shape mismatch",why);
        if (!ref::finite(normalized_input))return ref::fail(RAD_E_INVAL,"nonfinite attention input",why);
        Sequence staged=*this;Trace detail;std::vector<float> result(output.size());
        const int status=staged.step_in_place(layer,position,normalized_input,result,&detail,why);
        if (status!=RAD_OK)return status;
        *this=std::move(staged);std::copy(result.begin(),result.end(),output.begin());
        if(trace)*trace=std::move(detail);
        if(why)why->clear();
        return RAD_OK;
    }
    int snapshot(Snapshot* out,std::string* why=nullptr) const {
        if (!out || next_layer_!=0)return ref::fail(RAD_E_STATE,"attention snapshots require a completed token boundary",why);
        Snapshot saved;saved.model=model_;saved.sequence_identity=sequence_identity_;saved.position=position_;
        for(const auto& window:windows_)saved.windows.push_back(window.snapshot());
        for(const auto& compressor:compressors_)saved.compressors.push_back(compressor?std::optional(compressor->snapshot()):std::nullopt);
        saved.compressed_kv=compressed_kv_;saved.index_keys=index_keys_;*out=std::move(saved);
        if(why)why->clear();
        return RAD_OK;
    }
    int restore(const Snapshot& saved,std::string* why=nullptr) {
        if (saved.model!=model_ || saved.sequence_identity!=sequence_identity_)
            return ref::fail(RAD_E_INVAL,"attention prefix belongs to another model or sequence",why);
        const auto& c=model_->config();const size_t count=size_t(c.n_layers);
        if (saved.position<0 || saved.position>limit_ || saved.windows.size()!=count ||
            saved.compressors.size()!=count || saved.compressed_kv.size()!=count || saved.index_keys.size()!=count)
            return ref::fail(RAD_E_FORMAT,"attention prefix shape/position mismatch",why);
        Sequence staged=*this;
        for(size_t layer=0;layer<count;++layer) {
            const auto& s=c.layers[layer];
            if (saved.windows[layer].next_position!=saved.position)
                return ref::fail(RAD_E_FORMAT,"window prefix positions disagree",why);
            int status=staged.windows_[layer].restore(saved.windows[layer],why);
            if(status!=RAD_OK)return status;
            if (bool(staged.compressors_[layer])!=bool(saved.compressors[layer]))
                return ref::fail(RAD_E_FORMAT,"prefix compressor ownership changed",why);
            if(staged.compressors_[layer]) {
                if(saved.compressors[layer]->next_position!=saved.position)
                    return ref::fail(RAD_E_FORMAT,"compressor prefix positions disagree",why);
                status=staged.compressors_[layer]->restore(*saved.compressors[layer],why);
                if(status!=RAD_OK)return status;
            }
            const size_t rows=s.compress_ratio?size_t(saved.position/s.compress_ratio):0;
            const size_t kv_size=s.writes_kv(int(layer))?rows*size_t(c.head_dim):0;
            const size_t key_size=s.writes_index_k(int(layer))?rows*size_t(c.index_head_dim):0;
            if(saved.compressed_kv[layer].size()!=kv_size || saved.index_keys[layer].size()!=key_size ||
                !ref::finite(saved.compressed_kv[layer]) || !ref::finite(saved.index_keys[layer]))
                return ref::fail(RAD_E_FORMAT,"prefix shared-cache extent or values are invalid",why);
        }
        staged.compressed_kv_=saved.compressed_kv;staged.index_keys_=saved.index_keys;
        staged.position_=saved.position;staged.next_layer_=0;staged.frame_.assign(count,{});
        *this=std::move(staged);if(why)why->clear();
        return RAD_OK;
    }
private:
    struct Frame {
        bool kv_ready=false,key_ready=false,index_ready=false,candidates_ready=false;
        int64_t first_position=-1;
        std::vector<float> latent;
        std::vector<int32_t> indices;
        std::vector<uint8_t> candidates;
    };
    ref::Rope rope(bool compressed) const {
        const auto& c=model_->config();
        return {size_t(c.rope_head_dim),float(compressed?c.compress_rope_theta:c.rope_theta),
                float(c.rope_factor),float(c.beta_fast),float(c.beta_slow),compressed?c.original_seq_len:0};
    }
    int step_in_place(int layer,int64_t position,Floats x,Mutable output,Trace* trace,std::string* why) {
        const auto& c=model_->config();const auto& s=c.layers[layer];const auto& w=model_->layer(layer);
        const size_t hd=size_t(c.head_dim),heads=size_t(c.n_head),qr=size_t(c.q_lora_rank);
        const size_t ih=size_t(c.index_n_heads),id=size_t(c.index_head_dim);
        const auto rotation=rope(s.compress_ratio!=0);
        auto linear=[&](Floats input,const std::vector<float>& weight,size_t outputs,std::vector<float>* dest) {
            dest->resize(outputs);return ref::linear(input,weight,outputs,*dest,ref::Precision::f32,why);
        };
        auto norm=[&](std::vector<float>* input,const std::vector<float>& weight) {
            return ref::rms_norm(*input,weight,float(c.norm_eps),*input,ref::Precision::f32,why);
        };
        int status=linear(x,w.q_a,qr,&trace->q_latent);if(status!=RAD_OK)return status;
        status=norm(&trace->q_latent,w.q_norm);if(status!=RAD_OK)return status;
        status=linear(trace->q_latent,w.q_b,heads*hd,&trace->query);if(status!=RAD_OK)return status;
        for(size_t h=0;h<heads;++h) {
            status=ref::rotary(Mutable(trace->query).subspan(h*hd,hd),position,rotation,false,why);
            if(status!=RAD_OK)return status;
        }
        status=linear(x,w.kv,hd,&trace->window_row);if(status!=RAD_OK)return status;
        status=norm(&trace->window_row,w.kv_norm);if(status!=RAD_OK)return status;
        status=ref::rotary(trace->window_row,position,rotation,false,why);if(status!=RAD_OK)return status;
        status=windows_[layer].append(position,trace->window_row,why);if(status!=RAD_OK)return status;
        trace->window_indices=windows_[layer].selected();trace->combined_indices=trace->window_indices;
        std::vector<float> joint(windows_[layer].values().begin(),windows_[layer].values().end());
        if(s.compress_ratio) {
            const size_t visible=size_t((position+1)/s.compress_ratio);
            if(s.writes_kv(layer)) {
                std::vector<float> kv,score;
                status=linear(x,w.compress_kv,hd,&kv);if(status!=RAD_OK)return status;
                if(s.compress_ratio==2) {status=linear(x,w.compress_gate,hd,&score);if(status!=RAD_OK)return status;}
                status=compressors_[layer]->push(kv,score,&frame_[layer].latent,&frame_[layer].first_position,why);
                if(status!=RAD_OK)return status;
                frame_[layer].kv_ready=true;
                trace->compressed_latent=frame_[layer].latent;
                trace->compressed_first_position=frame_[layer].first_position;
            }
            if(!frame_[s.kv].kv_ready)return ref::fail(RAD_E_STATE,"compressed KV producer has not run for this token",why);
            if(s.writes_index_k(layer)) {
                const auto& source=frame_[s.kv];
                if(!source.latent.empty()) {
                    status=linear(source.latent,w.index_k,id,&trace->index_key);if(status!=RAD_OK)return status;
                    status=norm(&trace->index_key,w.index_k_norm);if(status!=RAD_OK)return status;
                    status=ref::rotary(trace->index_key,source.first_position,rotation,false,why);if(status!=RAD_OK)return status;
                    index_keys_[layer].insert(index_keys_[layer].end(),trace->index_key.begin(),trace->index_key.end());
                }
                frame_[layer].key_ready=true;
            }
            if(!frame_[s.index_k].key_ready || index_keys_[s.index_k].size()!=visible*id)
                return ref::fail(RAD_E_STATE,"shared index-K producer/extent mismatch",why);
            if(s.writes_index(layer)) {
                trace->scores.resize(visible);
                if(visible) {
                    status=linear(trace->q_latent,w.index_q,ih*id,&trace->index_query);if(status!=RAD_OK)return status;
                    for(size_t h=0;h<ih;++h) {
                        status=ref::rotary(Mutable(trace->index_query).subspan(h*id,id),position,rotation,false,why);
                        if(status!=RAD_OK)return status;
                    }
                    std::vector<float> head_weights;
                    status=linear(x,w.index_head_weights,ih,&head_weights);if(status!=RAD_OK)return status;
                    status=ref::index_scores(trace->index_query,index_keys_[s.index_k],head_weights,id,ih,trace->scores,why);
                    if(status!=RAD_OK)return status;
                }
                if(s.writes_candidates(layer)) {
                    frame_[layer].candidates.resize(visible);
                    status=ref::candidate_blocks(trace->scores,visible,size_t(c.candidate_block_size),
                               size_t(c.candidate_topk_blocks),frame_[layer].candidates,why);
                    if(status!=RAD_OK)return status;
                    frame_[layer].candidates_ready=true;
                }
                std::span<const uint8_t> candidates;
                if(s.uses_candidates(layer)) {
                    if(!frame_[s.candidate].candidates_ready || frame_[s.candidate].candidates.size()!=visible)
                        return ref::fail(RAD_E_STATE,"candidate producer/extent mismatch",why);
                    candidates=frame_[s.candidate].candidates;
                }
                status=ref::index_topk(trace->scores,visible,size_t(c.index_topk),candidates,&frame_[layer].indices,why);
                if(status!=RAD_OK)return status;
                frame_[layer].index_ready=true;
            }
            if(!frame_[s.index].index_ready)return ref::fail(RAD_E_STATE,"index selection producer has not run for this token",why);
            trace->compressed_indices=frame_[s.index].indices;
            if(s.candidate>=0 && frame_[s.candidate].candidates_ready)trace->candidates=frame_[s.candidate].candidates;
            // Index keys above consume the RoPE-free latent. Only now rotate a copy for the KV cache.
            if(s.writes_kv(layer) && !frame_[layer].latent.empty()) {
                trace->compressed_row=frame_[layer].latent;
                status=ref::rotary(trace->compressed_row,frame_[layer].first_position,rotation,false,why);
                if(status!=RAD_OK)return status;
                compressed_kv_[layer].insert(compressed_kv_[layer].end(),trace->compressed_row.begin(),trace->compressed_row.end());
            }
            if(compressed_kv_[s.kv].size()!=visible*hd)
                return ref::fail(RAD_E_STATE,"shared compressed KV extent mismatch",why);
            joint.insert(joint.end(),compressed_kv_[s.kv].begin(),compressed_kv_[s.kv].end());
            const size_t offset=size_t(c.window_size);
            for(auto index:trace->compressed_indices) {
                if(index>=0 && (size_t(index)>=visible || size_t(index)>size_t(INT32_MAX)-offset))
                    return ref::fail(RAD_E_SHAPE,"compressed index exceeds visible range",why);
                trace->combined_indices.push_back(index<0?-1:int32_t(offset+size_t(index)));
            }
        }
        trace->attended.resize(heads*hd);
        status=ref::sparse_attention(trace->query,joint,heads,hd,w.sinks,trace->combined_indices,
                                   1.f/std::sqrt(float(hd)),trace->attended,why);
        if(status!=RAD_OK)return status;
        trace->unrotated_heads=trace->attended;
        for(size_t h=0;h<heads;++h) {
            status=ref::rotary(Mutable(trace->unrotated_heads).subspan(h*hd,hd),position,rotation,true,why);
            if(status!=RAD_OK)return status;
        }
        status=ref::grouped_output(trace->unrotated_heads,heads,hd,size_t(c.o_groups),size_t(c.o_lora_rank),
                                    w.out_a,w.out_b,output,ref::Precision::f32,why);
        if(status!=RAD_OK)return status;
        if(!ref::finite(output))return ref::fail(RAD_E_INVAL,"nonfinite attention output",why);
        trace->output.assign(output.begin(),output.end());
        if(++next_layer_==c.n_layers) {next_layer_=0;++position_;frame_.assign(size_t(c.n_layers),{});}
        return RAD_OK;
    }
    std::shared_ptr<const Model> model_;
    std::string sequence_identity_;
    int64_t limit_=0,position_=0;
    int next_layer_=0;
    std::vector<ref::WindowCache> windows_;
    std::vector<std::optional<ref::Compressor>> compressors_;
    std::vector<std::vector<float>> compressed_kv_,index_keys_;
    std::vector<Frame> frame_;
};
} // namespace rad::deepseek41::attention
