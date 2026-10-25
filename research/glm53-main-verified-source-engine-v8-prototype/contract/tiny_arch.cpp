// Genuine synthetic FFN graph, staged for declaration/quote ONLY. No Engine hook.
#include "profile.h"
#include "tiny_arch.h"
#include "rad_model_failure_policy.h"
extern "C" uint32_t rad_core_model_failure_policy_version(void) __attribute__((weak));
extern "C" int rad_require_permanent_model_failure(RadBuilder*) __attribute__((weak));
#include "rad_arch.h"
#include "rad_block_moe_fp8.h"
#include <array>
#include <algorithm>
namespace glm53_tiny_native_arch {
using namespace rad::arch;
using namespace glm53::tiny_startup;
struct Model {
    Names names{""};Geom g{};
    std::array<MoeFP8,L> layers{};
    rad_buf x=0,h=0,hout=0,logits=0;
    rad_weight embed=0,norm=0,head=0;
    rad_op op_embed=0,op_norm=0,op_gather=0,op_head=0;
};
static Model model;
int declare(RadBuilder* b,const RadModelMeta* meta,const RadBuildCtx* ctx){
    if(!b||admit_declaration_shape(meta,ctx)<0)return RAD_E_UNSUPPORTED;
    if(!rad_core_model_failure_policy_version||!rad_require_permanent_model_failure||
       rad_core_model_failure_policy_version()<RAD_MODEL_FAILURE_POLICY_VERSION)return RAD_E_ABI;
    RAD_ARCH_TRY(rad_require_permanent_model_failure(b));
    model=Model{};auto& m=model;auto& nm=m.names;auto& g=m.g;
    // Only actual FFN geometry is initialized. No fictitious attention heads.
    g.n_layer=L;g.n_embd=H;g.n_ff=g.n_ff_all=F;g.n_vocab=g.n_vocab_all=V;
    g.max_tok=T;g.max_seqs=1;g.max_ctx=T;g.max_logit_rows=1;
    g.dtype="bf16";g.act_dtype=RAD_BF16;g.eps=1e-5f;g.w_bf16=false;
    m.x=decl_b(b,"tiny.x",RAD_BF16,{T,H});m.h=decl_b(b,"tiny.final_h",RAD_BF16,{T,H});
    m.hout=decl_b(b,"tiny.h_out",RAD_BF16,{1,H});m.logits=decl_b(b,"tiny.logits",RAD_F32,{1,V});
    if(!m.x||!m.h||!m.hout||!m.logits)return RAD_E_NOMEM;
    RAD_ARCH_TRY(rad_declare_logits(b,m.logits));
    RAD_ARCH_TRY(map_copy(b,"tiny.token_embd.weight","tiny.embed.weight"));
    RAD_ARCH_TRY(map_copy(b,"tiny.output_norm.weight","tiny.final_norm.weight"));
    RAD_ARCH_TRY(map_copy(b,"tiny.output.weight","tiny.head.weight"));
    m.embed=decl_w(b,"tiny.token_embd.weight",RAD_BF16,{V,H},RAD_ACCESS_VOCAB,RAD_SHARD_ROW,grp_model());
    m.norm=decl_w(b,"tiny.output_norm.weight",RAD_F32,{H},RAD_ACCESS_PER_TOKEN,RAD_SHARD_NONE,grp_model());
    m.head=decl_w(b,"tiny.output.weight",RAD_BF16,{V,H},RAD_ACCESS_VOCAB,RAD_SHARD_ROW,grp_model());
    if(!m.embed||!m.norm||!m.head)return RAD_E_SHAPE;
    m.op_embed=rw(b,RAD_OP(b,"embed_lookup",RAD_PARAMS(RAD_RANGE("M",1,T),RAD_INT("n_embd",H),RAD_INT("n_vocab",V),RAD_STR("dtype","bf16"),RAD_INT("vocab_offset",0)),RAD_WEIGHTS(m.embed)),{},{m.x});
    for(int l=0;l<L;++l){auto& moe=m.layers[size_t(l)];MoeFP8::Wire w{};w.x=m.x;
        auto buf=[&](const char* name,uint32_t dtype,std::initializer_list<int64_t> shape){return decl_b(b,nm.f("tiny.layer.%d.%s",l,name),dtype,shape);};
        w.h.x=buf("normed",RAD_BF16,{T,H});RAD_ARCH_TRY(w.h.declare_qs(b,nm,g,nm.f("tiny.layer.%d.input",l),H));
        w.logits=buf("router_logits",RAD_F32,{T,E});w.ids=buf("ids",RAD_I32,{T,K});w.ew=buf("route_weights",RAD_F32,{T,K});
        w.ecnt=buf("count",RAD_I32,{E});w.eoff=buf("offset",RAD_I32,{E+1});w.sorted=buf("sorted",RAD_I32,{R*K});
        w.egu=buf("GU",RAD_BF16,{R*K,2*F});w.eff.x=buf("product",RAD_BF16,{R*K,F});
        RAD_ARCH_TRY(w.eff.declare_qs(b,nm,g,nm.f("tiny.layer.%d.ff",l),F,R*K));w.edn=buf("DN",RAD_BF16,{R*K,H});
        MoeFP8::Src src{};src.norm=nm.ckpt("tiny.layers.%d.ffn_norm.weight",l);
        src.router=nm.ckpt("tiny.layers.%d.router.weight",l);src.router_bias=nm.ckpt("tiny.layers.%d.router.correction",l);
        src.experts=nm.ckpt("tiny.layers.%d.experts",l);
        auto& c=src.cfg;c.n_expert=c.n_expert_all=E;c.n_ff_exp=F;c.top_k=K;c.rows=R;
        c.expert_mx=1;c.sigmoid_router=1;c.route_scale=2.5;c.swiglu_limit=10;c.swiglu_quant=1;c.route_prepare=1;
        for(int parity=0;parity<2;++parity){c.ff_lo[parity]=0;c.ff_hi[parity]=F;}
        RAD_ARCH_TRY(moe.declare(b,nm,g,l,w,src,false,true));
    }
    m.op_norm=rw(b,RAD_OP(b,"rmsnorm",RAD_PARAMS(RAD_RANGE("M",1,T),RAD_INT("n",H),RAD_F64("eps",1e-5),RAD_F64("wadd",0),RAD_STR("dtype","bf16")),RAD_WEIGHTS(m.norm)),{m.x},{m.h});
    m.op_gather=rw(b,RAD_OP(b,"gather_rows",RAD_PARAMS(RAD_RANGE("M",1,1),RAD_INT("n",H),RAD_STR("dtype","bf16")),RAD_NOWEIGHTS),{m.h},{m.hout});
    m.op_head=rw(b,RAD_OP(b,"logits_gemm",RAD_PARAMS(RAD_RANGE("M",1,1),RAD_INT("n_vocab",V),RAD_INT("n_embd",H),RAD_STR("dtype","bf16")),RAD_WEIGHTS(m.head)),{m.hout},{m.logits});
    return m.op_embed&&m.op_norm&&m.op_gather&&m.op_head?RAD_OK:RAD_E_SHAPE;
}
void step(RadCtx* c,const RadBatch* batch){
    // This package never invokes step. The actual graph is present for later
    // independently approved execution; no no-op/fake-logits success shim.
    auto& m=model;if(!c||!batch)return;
    RAD_ISSUE(c,m.op_embed,praw(batch->token_ids,RAD_I32,batch->n_tok),RAD_W(m.embed),brows(m.x,batch->n_tok));
    for(const auto& layer:m.layers)for(int64_t first=0;first<batch->n_tok;first+=R)
        layer.pass(c,batch->n_tok,first,std::min<int64_t>(R,batch->n_tok-first));
    RAD_ISSUE(c,m.op_norm,brows(m.x,batch->n_tok),RAD_W(m.norm),brows(m.h,batch->n_tok));
    if(batch->n_out>0){RAD_ISSUE_N(c,m.op_gather,batch->n_out,brows(m.h,batch->n_tok),praw(batch->out_ids,RAD_I32,batch->n_out),brows(m.hout,batch->n_out));
        RAD_ISSUE_N(c,m.op_head,batch->n_out,brows(m.hout,batch->n_out),RAD_W(m.head),brows(m.logits,batch->n_out));}
}
} // namespace glm53_tiny_native_arch
#ifndef TINY_STARTUP_ARCH_NO_EXPORTS
RAD_ARCH_PLUGIN(glm53_tiny_native_arch,"glm53nativetiny","","source-only-v1","Synthetic no-attention pre-norm residual Native FFN graph; declaration/quote gate only")
RAD_ARCH_SERVES_IMPORTS(1)
extern "C" int rad_plugin_open(){return rad_core_model_failure_policy_version&&rad_require_permanent_model_failure&&rad_core_model_failure_policy_version()>=RAD_MODEL_FAILURE_POLICY_VERSION?RAD_OK:RAD_E_ABI;}
#endif
