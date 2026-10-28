/* Exact native DSpark1017 EP4 geometry. Private source-only prototype; no SDK change. */
#pragma once
#include "../rad_core.h"
#include <array>
#include <charconv>
#include <cstring>
#include <cstdint>

namespace rad {
inline constexpr char kDsparkSelectedEnv[]="RADIANCE_EXPERIMENTAL_DS41_DSPARK_EP_SELECTED";
inline bool dspark_exact_one(const char* value) noexcept { return value&&std::strcmp(value,"1")==0; }

/* Aliases must be present, exact decimal integers and mutually consistent. The
 * architecture reads normalized deepseek41.* or the official release spellings. */
inline bool dspark_meta_integer(const RadModelMeta& m,const char* normal,const char* official,int64_t expected) noexcept {
    if(m.n_kv<0||m.n_kv>65536||(m.n_kv&&(!m.kv_key||!m.kv_val)))return false;
    bool found=false,seen_normal=false,seen_official=false;
    for(int i=0;i<m.n_kv;++i){
        if(!m.kv_key[i]||!m.kv_val[i])return false;
        const bool a=std::strcmp(m.kv_key[i],normal)==0,b=std::strcmp(m.kv_key[i],official)==0;
        if(!a&&!b)continue;
        if((a&&seen_normal)||(b&&seen_official))return false;
        seen_normal|=a;seen_official|=b;
        const char* s=m.kv_val[i];int64_t value=0;
        const auto parsed=std::from_chars(s,s+std::strlen(s),value);
        if(parsed.ec!=std::errc{}||parsed.ptr==s||*parsed.ptr||value!=expected)return false;
        found=true;
    }
    return found;
}
inline bool dspark_native_weight_scope(const WeightInfo& w) noexcept {
    const auto& d=w.decl;
    if(d.group.layer<40||d.group.layer>=43||d.group.expert<0||d.group.expert>=32||
       d.access!=RAD_ACCESS_CONDITIONAL||d.shard!=RAD_SHARD_NONE||d.rank!=2||
       w.widen||w.narrow||w.layout_tag!="ggml_mxfp4")return false;
    return (d.group.slot==0&&d.shape[0]==4608&&d.shape[1]==5120&&w.stored_bytes==12533760)||
           (d.group.slot==1&&d.shape[0]==5120&&d.shape[1]==2304&&w.stored_bytes==6266880);
}

/* Called at owning initialization, never on a cache hit. This validates actual
 * final Program/WTAB geometry in addition to advertised checkpoint metadata.
 * Each stage has exactly32 whole native GU/DN units and one32-entry op per slot.
 * Unsupported declarations must retain the conservative full-layer fallback. */
inline bool dspark_native_program_scope(const Program& p) noexcept {
    const auto& m=p.meta;const auto& c=p.ctx;const auto& d=p.drafter;
    if(!m.arch_id||std::strcmp(m.arch_id,"deepseek41")||m.n_layers!=40||m.n_embd!=5120||m.n_ff!=2304||
       m.n_expert!=384||m.n_expert_used!=6||m.n_vocab!=129280||c.world_size!=4||c.rank<0||c.rank>=4||
       c.max_seqs!=1||c.max_tok<5||c.max_spec!=5||c.kv_dtype!=RAD_BF16||p.drafter_name!="dspark"||
       d.kind!=RAD_DRAFT_BLOCK||d.depth!=5||d.block!=5||d.proposal_pitch!=5||!d.proposal||
       d.mask_token!=128799||d.window!=129)return false;
    if(!dspark_meta_integer(m,"deepseek41.nextn_predict_layers","num_nextn_predict_layers",3)||
       !dspark_meta_integer(m,"deepseek41.dspark.block_size","dspark_block_size",5)||
       !dspark_meta_integer(m,"deepseek41.dspark.expert_count","dspark_n_routed_experts",128)||
       !dspark_meta_integer(m,"deepseek41.dspark.expert_used_count","dspark_num_experts_per_tok",3))return false;
    std::array<std::array<uint8_t,64>,3> weights{};
    for(size_t h=1;h<p.weights.size();++h){const auto& w=p.weights[h];
        if(w.decl.group.expert<0||w.decl.group.layer<40)continue;
        if(!dspark_native_weight_scope(w))return false;
        auto& seen=weights[size_t(w.decl.group.layer-40)][size_t(w.decl.group.expert*2+w.decl.group.slot)];
        if(seen)return false;
        seen=1;
    }
    for(const auto& layer:weights)for(uint8_t seen:layer)if(seen!=1)return false;
    std::array<std::array<uint8_t,2>,3> ops{};
    for(const auto& op:p.ops){
        if(op.weights.empty())continue;
        bool draft=false;
        for(auto h:op.weights){if(h==0||size_t(h)>=p.weights.size())continue;
            const auto& w=p.weights[h];draft|=w.decl.group.layer>=40&&w.decl.group.expert>=0;}
        if(!draft)continue;
        for(auto h:op.weights)if(h==0||size_t(h)>=p.weights.size())return false;
        long long ne=0,top=0,N=0,K=0;
        if(op.op!="moe_gemm_ggml"||op.weights.size()!=32||!op.base.get_i("n_expert",&ne)||ne!=32||
           !op.base.get_i("top_k",&top)||top!=3||!op.base.get_i("N",&N)||!op.base.get_i("K",&K))return false;
        const char* type=op.base.get_s("type"),*order=op.base.get_s("a_order"),*dtype=op.base.get_s("dtype");
        if(!type||std::strcmp(type,"mxfp4")||!dtype||std::strcmp(dtype,"bf16"))return false;
        const int slot=N==4608&&K==5120?0:N==5120&&K==2304?1:-1;
        if(slot<0||!order||std::strcmp(order,slot?"sorted":"token"))return false;
        const int layer=p.weights[op.weights[0]].decl.group.layer;
        if(layer<40||layer>=43)return false;
        for(size_t e=0;e<32;++e){const auto& w=p.weights[op.weights[e]];
            if(!dspark_native_weight_scope(w)||w.decl.group.layer!=layer||w.decl.group.expert!=int(e)||w.decl.group.slot!=slot)return false;}
        if(ops[size_t(layer-40)][size_t(slot)])return false;
        ops[size_t(layer-40)][size_t(slot)]=1;
    }
    for(const auto& layer:ops)for(uint8_t seen:layer)if(seen!=1)return false;
    return true;
}
inline bool dspark_query_batch(const RadBatch* b) noexcept {
    // Scheduler::block_query declares this exact one-sequence5-row mask query.
    return b&&b->phase==RAD_PHASE_DECODE&&b->draft_pass==1&&b->n_seq==1&&b->n_tok==5&&
        b->n_spec==0&&b->n_seq_decode==0&&b->n_tok_decode==0;
}
} //namespace rad
