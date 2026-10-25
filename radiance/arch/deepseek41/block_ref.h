/* Tiny serial DeepSeek1017 block/FFN/head oracle. This is a logical-weight CPU path,
 * not a serving architecture or a model-size allocation path. Linear weights are
 * supplied as [out,in] FP32 values (e.g. decoded packed weights in a tiny fixture).
 * No FP4/FP8 activation quantization is emulated. Precision::bf16 makes the model's
 * BF16 boundaries explicit; production packed arithmetic remains a separate gate.
 * Source: official inference/model.py Block, Gate, Expert, MoE and Transformer at
 * 1bae0e1088993b9b82de08f59de5be5a34e42170. No speculative/vision path is implied.
 */
#pragma once
#include "../common/rad_ds41_ref.h"
#include "engram_ref.h"
#include <optional>

namespace rad::deepseek41::ref {
struct ExpertWeights { std::vector<float> gate, up, down; };
struct ExpertTrace { float route_weight=0; std::vector<float> gate,up,activated,down_input,output; };
struct MoeWeights {
    size_t dim=0, intermediate=0, topk=0;
    float temperature=1, route_scale=1, routed_clamp=0, shared_clamp=0;
    bool normalize=true;
    std::vector<float> router, correction;
    std::vector<ExpertWeights> experts;
    ExpertWeights shared;
};
struct MoeTrace { std::vector<float> logits; Routing routing; std::vector<ExpertTrace> experts; ExpertTrace shared; };

inline int validate_expert(const ExpertWeights& w,size_t dim,size_t intermediate,std::string* why=nullptr) {
    size_t n=0;
    if(!dim||!intermediate||!extent({dim,intermediate},&n)||w.gate.size()!=n||w.up.size()!=n||w.down.size()!=n)
        return fail(RAD_E_SHAPE,"DeepSeek expert matrix shape mismatch",why);
    if(!finite(w.gate)||!finite(w.up)||!finite(w.down))return fail(RAD_E_INVAL,"nonfinite DeepSeek expert weight",why);
    return RAD_OK;
}
inline int validate_moe(const MoeWeights& w,std::string* why=nullptr) {
    size_t n=0;
    if(!w.dim||!w.intermediate||w.experts.empty()||!w.topk||w.topk>w.experts.size()||
       !extent({w.experts.size(),w.dim},&n)||w.router.size()!=n||w.correction.size()!=w.experts.size()||
       !(w.temperature>0)||!(w.route_scale>0)||w.routed_clamp<0||w.shared_clamp<0||
       !std::isfinite(w.temperature)||!std::isfinite(w.route_scale)||!std::isfinite(w.routed_clamp)||
       !std::isfinite(w.shared_clamp)||!finite(w.router)||!finite(w.correction))
        return fail(RAD_E_SHAPE,"invalid DeepSeek MoE geometry or routing parameters",why);
    for(const auto& expert:w.experts){const int st=validate_expert(expert,w.dim,w.intermediate,why);if(st!=RAD_OK)return st;}
    return validate_expert(w.shared,w.dim,w.intermediate,why);
}
inline int expert_forward(Floats x,const ExpertWeights& w,size_t intermediate,float clamp,float routing_weight,
                           Mutable out,Precision precision=Precision::f32,ExpertTrace* trace=nullptr,
                           std::string* why=nullptr) {
    int status=validate_expert(w,x.size(),intermediate,why);
    if(status!=RAD_OK)return status;
    if(out.size()!=x.size()||!std::isfinite(routing_weight)||routing_weight<0)
        return fail(RAD_E_SHAPE,"invalid expert output or routing weight",why);
    ExpertTrace staged;staged.route_weight=routing_weight;
    staged.gate.resize(intermediate);staged.up.resize(intermediate);staged.activated.resize(intermediate);
    staged.down_input.resize(intermediate);staged.output.resize(x.size());
    status=linear(x,w.gate,intermediate,staged.gate,precision,why);
    if(status==RAD_OK)status=linear(x,w.up,intermediate,staged.up,precision,why);
    if(status==RAD_OK)status=swiglu(staged.gate,staged.up,clamp,staged.activated,Precision::f32,why);
    if(status!=RAD_OK)return status;
    // Official Expert multiplies routing weight into the activation BEFORE the
    // down-projection input cast. Moving the weight after down changes BF16/FP4 math.
    for(size_t i=0;i<intermediate;++i)staged.down_input[i]=rounded(staged.activated[i]*routing_weight,precision);
    status=linear(staged.down_input,w.down,x.size(),staged.output,precision,why);
    if(status!=RAD_OK)return status;
    std::copy(staged.output.begin(),staged.output.end(),out.begin());
    if(trace)*trace=std::move(staged);
    return RAD_OK;
}
inline int moe_forward(Floats x,const MoeWeights& w,Mutable out,Precision precision=Precision::f32,
                        MoeTrace* trace=nullptr,std::string* why=nullptr) {
    int status=validate_moe(w,why);
    if(status!=RAD_OK)return status;
    if(x.size()!=w.dim||out.size()!=w.dim)return fail(RAD_E_SHAPE,"DeepSeek MoE input/output shape mismatch",why);
    MoeTrace staged;staged.logits.resize(w.experts.size());
    status=linear(x,w.router,w.experts.size(),staged.logits,Precision::f32,why);
    if(status==RAD_OK)status=route(staged.logits,w.correction,w.topk,w.temperature,w.route_scale,w.normalize,&staged.routing,why);
    if(status!=RAD_OK)return status;
    std::vector<float> sum(w.dim,0),part(w.dim);if(trace)staged.experts.resize(w.experts.size());
    // Match official MoE's ascending expert-ID accumulation, independent of top-k order.
    // Every configured expert can be selected; there is no subset/reduced-expert shortcut.
    for(size_t expert=0;expert<w.experts.size();++expert) {
        const auto found=std::find(staged.routing.ids.begin(),staged.routing.ids.end(),int32_t(expert));
        if(found==staged.routing.ids.end())continue;
        const size_t selected=size_t(found-staged.routing.ids.begin());
        status=expert_forward(x,w.experts[expert],w.intermediate,w.routed_clamp,staged.routing.weights[selected],
                              part,precision,trace?&staged.experts[expert]:nullptr,why);
        if(status!=RAD_OK)return status;
        for(size_t d=0;d<w.dim;++d)sum[d]+=part[d];
    }
    status=expert_forward(x,w.shared,w.intermediate,w.shared_clamp,1.f,part,precision,trace?&staged.shared:nullptr,why);
    if(status!=RAD_OK)return status;
    for(size_t d=0;d<w.dim;++d)out[d]=rounded(sum[d]+part[d],precision);
    if(trace)*trace=std::move(staged);
    return RAD_OK;
}

struct MhcWeights { std::vector<float> fn,scale,base; };
struct EngramWeights { size_t control_layer=0; std::vector<float> projection,q_weight,k_weight; };
struct BlockWeights {
    int layer_id=-1;size_t dim=0,copies=4;int sinkhorn_iterations=20;
    float norm_eps=1e-6f,hc_eps=1e-6f;
    MhcWeights attention_hc,ffn_hc;
    std::vector<float> attention_norm,ffn_norm;
    MoeWeights moe;
    std::optional<EngramWeights> engram;
};
struct SerialState { std::vector<float> h,previous_pre; };
struct EngramToken {
    std::shared_ptr<const engram::ModelControls> model;
    std::span<const int64_t> hashes; // one token, [Engram layers, hash columns]
    engram::RowLookup lookup;
    bool alive=true;
};
using AttentionCallback=std::function<int(int layer_id,int64_t position,Floats normalized,Mutable out,std::string* why)>;
struct BlockTrace {
    std::vector<float> after_engram,attention_input,attention_normed,attention_output,after_attention;
    MhcMix attention_mix,ffn_mix;
    std::vector<float> ffn_input,ffn_normed,ffn_output;
    MoeTrace moe;
};
inline int init_token(Floats embedding,size_t copies,SerialState* out,Precision precision=Precision::f32,
                       std::string* why=nullptr) {
    size_t width=0;
    if(!out||embedding.empty()||copies<2||copies>8||!extent({embedding.size(),copies},&width)||!finite(embedding))
        return fail(RAD_E_SHAPE,"invalid DeepSeek initial residual geometry",why);
    SerialState state;state.h.resize(width);state.previous_pre.assign(copies,0.f);state.previous_pre[0]=1.f;
    for(size_t copy=0;copy<copies;++copy)for(size_t d=0;d<embedding.size();++d)state.h[copy*embedding.size()+d]=rounded(embedding[d],precision);
    *out=std::move(state);return RAD_OK;
}
inline int validate_block(const BlockWeights& w,std::string* why=nullptr) {
    size_t width=0,mixes=0,n=0;
    if(w.layer_id<0||!w.dim||w.dim>INT64_MAX||w.copies<2||w.copies>8||
       !extent({w.copies,w.dim},&width)||!extent({w.copies,w.copies+2},&mixes)||!extent({mixes,width},&n)||
       w.sinkhorn_iterations<1||w.sinkhorn_iterations>256||!(w.norm_eps>0)||w.hc_eps<0||
       !std::isfinite(w.norm_eps)||!std::isfinite(w.hc_eps)||w.attention_norm.size()!=w.dim||
       w.ffn_norm.size()!=w.dim||!finite(w.attention_norm)||!finite(w.ffn_norm)||w.moe.dim!=w.dim)
        return fail(RAD_E_SHAPE,"invalid DeepSeek block geometry",why);
    for(const auto* hc:{&w.attention_hc,&w.ffn_hc})
        if(hc->fn.size()!=n||hc->scale.size()!=3||hc->base.size()!=mixes||!finite(hc->fn)||!finite(hc->scale)||!finite(hc->base))
            return fail(RAD_E_SHAPE,"DeepSeek block mHC weight shape mismatch",why);
    return validate_moe(w.moe,why);
}
inline int apply_engram(const BlockWeights& w,Floats h,const EngramToken& token,Mutable out,
                         Precision precision,std::string* why=nullptr) {
    if(!w.engram||!token.model||!token.lookup)return fail(RAD_E_STATE,"missing DeepSeek Engram rows/controls",why);
    const auto& weights=*w.engram;const auto& c=token.model->get();
    size_t width=0,embed=0,projected=0,n=0,hashes=0;
    if(weights.control_layer>=c.layers.size()||c.layers[weights.control_layer].layer_id!=w.layer_id||
       !extent({w.copies,w.dim},&width)||h.size()!=width||out.size()!=width||
       !extent({size_t(token.model->columns()),size_t(c.head_dim)},&embed)||!extent({w.copies+1,w.dim},&projected)||
       !extent({embed,projected},&n)||!extent({c.layers.size(),size_t(token.model->columns())},&hashes)||token.hashes.size()!=hashes||
       weights.projection.size()!=n||weights.q_weight.size()!=width||weights.k_weight.size()!=width||
       !finite(weights.projection)||!finite(weights.q_weight)||!finite(weights.k_weight))
        return fail(RAD_E_SHAPE,"DeepSeek Engram block weight/hash shape mismatch",why);
    const auto rows=token.hashes.subspan(weights.control_layer*size_t(token.model->columns()),size_t(token.model->columns()));
    std::vector<float> embedding(embed),kv(projected);
    int status=engram::gather_rows(*token.model,weights.control_layer,1,rows,token.lookup,embedding,why);
    if(status!=RAD_OK)return status;
    for(auto& value:embedding)value=rounded(value,precision);
    status=linear(embedding,weights.projection,projected,kv,precision,why);
    const uint8_t alive=token.alive;
    if(status==RAD_OK)status=engram::gate_residual(1,int64_t(w.copies),int64_t(w.dim),h,kv,
        weights.q_weight,weights.k_weight,std::span(&alive,1),w.norm_eps,precision==Precision::bf16,out,{},why);
    return status;
}
// Local residual/pre outputs and trace are committed only after both sublayers succeed.
// A stateful attention callback owns its own transaction/rollback (its KV state is external).
inline int block_step(const BlockWeights& w,int64_t position,const SerialState& input,
                        const AttentionCallback& attention,SerialState* out,const EngramToken* token=nullptr,
                        Precision precision=Precision::f32,BlockTrace* trace=nullptr,std::string* why=nullptr) {
    int status=validate_block(w,why);if(status!=RAD_OK)return status;
    if(!out||!attention||position<0||input.previous_pre.size()!=w.copies||input.h.size()!=w.copies*w.dim||
       !finite(input.h)||!finite(input.previous_pre))return fail(RAD_E_SHAPE,"DeepSeek block input/state mismatch",why);
    BlockTrace staged;staged.after_engram=input.h;
    if(w.engram) {
        if(!token)return fail(RAD_E_STATE,"DeepSeek block needs its Engram token context",why);
        status=apply_engram(w,input.h,*token,staged.after_engram,precision,why);if(status!=RAD_OK)return status;
    }
    status=mhc_prepare(staged.after_engram,w.dim,w.copies,w.attention_hc.fn,w.attention_hc.scale,w.attention_hc.base,
                         w.sinkhorn_iterations,w.norm_eps,w.hc_eps,&staged.attention_mix,why);
    staged.attention_input.resize(w.dim);staged.attention_normed.resize(w.dim);staged.attention_output.resize(w.dim);
    staged.after_attention.resize(input.h.size());
    if(status==RAD_OK)status=mhc_collapse(staged.after_engram,input.previous_pre,staged.attention_input,precision,why);
    if(status==RAD_OK)status=rms_norm(staged.attention_input,w.attention_norm,w.norm_eps,staged.attention_normed,precision,why);
    if(status==RAD_OK)status=attention(w.layer_id,position,staged.attention_normed,staged.attention_output,why);
    if(status==RAD_OK)status=mhc_expand(staged.attention_output,staged.after_engram,staged.attention_mix,staged.after_attention,precision,why);
    if(status!=RAD_OK)return status;
    status=mhc_prepare(staged.after_attention,w.dim,w.copies,w.ffn_hc.fn,w.ffn_hc.scale,w.ffn_hc.base,
                         w.sinkhorn_iterations,w.norm_eps,w.hc_eps,&staged.ffn_mix,why);
    staged.ffn_input.resize(w.dim);staged.ffn_normed.resize(w.dim);staged.ffn_output.resize(w.dim);
    if(status==RAD_OK)status=mhc_collapse(staged.after_attention,staged.attention_mix.pre,staged.ffn_input,precision,why);
    if(status==RAD_OK)status=rms_norm(staged.ffn_input,w.ffn_norm,w.norm_eps,staged.ffn_normed,precision,why);
    if(status==RAD_OK)status=moe_forward(staged.ffn_normed,w.moe,staged.ffn_output,precision,trace?&staged.moe:nullptr,why);
    SerialState result;result.h.resize(input.h.size());result.previous_pre=staged.ffn_mix.pre;
    if(status==RAD_OK)status=mhc_expand(staged.ffn_output,staged.after_attention,staged.ffn_mix,result.h,precision,why);
    if(status!=RAD_OK)return status;
    *out=std::move(result);if(trace)*trace=std::move(staged);return RAD_OK;
}
struct HeadWeights { size_t dim=0,vocab=0;float norm_eps=1e-6f;std::vector<float> norm,projection; };
inline int validate_head(const HeadWeights& weights,std::string* why=nullptr) {
    size_t n=0;
    if(!weights.dim||!weights.vocab||!extent({weights.dim,weights.vocab},&n)||weights.projection.size()!=n||
       weights.norm.size()!=weights.dim||!(weights.norm_eps>0)||!std::isfinite(weights.norm_eps)||
       !finite(weights.norm)||!finite(weights.projection))
        return fail(RAD_E_SHAPE,"DeepSeek final head shape or parameter mismatch",why);
    return RAD_OK;
}
inline int final_head(const SerialState& state,const HeadWeights& weights,Mutable logits,
                        Precision precision=Precision::f32,std::string* why=nullptr) {
    const int checked=validate_head(weights,why);if(checked!=RAD_OK)return checked;
    if(logits.size()!=weights.vocab)
        return fail(RAD_E_SHAPE,"DeepSeek final head shape mismatch",why);
    std::vector<float> collapsed(weights.dim),normalized(weights.dim);
    int status=mhc_collapse(state.h,state.previous_pre,collapsed,precision,why);
    if(status==RAD_OK)status=rms_norm(collapsed,weights.norm,weights.norm_eps,normalized,precision,why);
    // Official ParallelHead promotes x and weights to FP32, even in BF16 mode.
    if(status==RAD_OK)status=linear(normalized,weights.projection,weights.vocab,logits,Precision::f32,why);
    return status;
}
struct SerialTrace { std::vector<BlockTrace> layers;SerialState final_state; };
inline int serial_token(Floats embedding,std::span<const BlockWeights> weights,int64_t position,
                          const HeadWeights& head,const AttentionCallback& attention,Mutable logits,
                          const EngramToken* engram_token=nullptr,Precision precision=Precision::f32,
                          SerialTrace* trace=nullptr,std::string* why=nullptr) {
    if(weights.empty()||head.dim!=embedding.size()||logits.size()!=head.vocab||position<0||!attention)
        return fail(RAD_E_SHAPE,"empty or inconsistent DeepSeek serial model",why);
    const int checked=validate_head(head,why);if(checked!=RAD_OK)return checked;
    for(size_t i=0;i<weights.size();++i) {
        if(weights[i].layer_id!=int(i)||weights[i].dim!=embedding.size()||weights[i].copies!=weights[0].copies)
            return fail(RAD_E_SHAPE,"DeepSeek serial layers are not an ordered uniform backbone",why);
        const int st=validate_block(weights[i],why);if(st!=RAD_OK)return st;
    }
    SerialState state;int status=init_token(embedding,weights[0].copies,&state,precision,why);
    SerialTrace staged;if(trace)staged.layers.resize(weights.size());
    for(size_t i=0;i<weights.size()&&status==RAD_OK;++i)
        status=block_step(weights[i],position,state,attention,&state,engram_token,precision,trace?&staged.layers[i]:nullptr,why);
    if(status==RAD_OK)status=final_head(state,head,logits,precision,why);
    if(status==RAD_OK&&trace){staged.final_state=std::move(state);*trace=std::move(staged);}
    return status;
}
} // namespace rad::deepseek41::ref
