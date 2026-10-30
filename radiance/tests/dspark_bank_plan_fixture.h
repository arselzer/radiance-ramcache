/* Shared synthetic actual Program/Geometry construction for HOST scope controls.
 * Metadata and native expert declarations only: no model payload, allocator or GPU. */
#pragma once
#include "place/dspark_native_scope.h"
#include <string>
#include <utility>

namespace dspark_scope_fixture {
inline rad::Program make(int rank=0,bool official=false){
    static const char* normalized[]={"deepseek41.nextn_predict_layers","deepseek41.dspark.block_size","deepseek41.dspark.expert_count","deepseek41.dspark.expert_used_count"};
    static const char* release[]={"num_nextn_predict_layers","dspark_block_size","dspark_n_routed_experts","dspark_num_experts_per_tok"};
    static const char* values[]={"3","5","128","3"};
    rad::Program p;p.ctx.rank=rank;p.ctx.world_size=4;p.ctx.max_tok=16;p.ctx.max_seqs=1;p.ctx.max_spec=5;p.ctx.kv_dtype=RAD_BF16;
    p.meta.arch_id="deepseek41";p.meta.n_layers=40;p.meta.n_embd=5120;p.meta.n_ff=2304;p.meta.n_expert=384;p.meta.n_expert_used=6;p.meta.n_vocab=129280;
    p.meta.n_kv=4;p.meta.kv_key=official?release:normalized;p.meta.kv_val=values;
    p.drafter_name="dspark";p.drafter.kind=RAD_DRAFT_BLOCK;p.drafter.depth=5;p.drafter.block=5;p.drafter.proposal_pitch=5;
    p.drafter.proposal=1;p.drafter.mask_token=128799;p.drafter.window=129;p.weights.resize(1);
    for(int layer=40;layer<43;++layer)for(int slot=0;slot<2;++slot){
        rad::OpInfo op;op.op="moe_gemm_ggml";op.index=int(p.ops.size());
        op.base.set_i("n_expert",32);op.base.set_i("top_k",3);op.base.set_i("N",slot?5120:4608);op.base.set_i("K",slot?2304:5120);
        op.base.set_s("type","mxfp4");op.base.set_s("dtype","bf16");op.base.set_s("a_order",slot?"sorted":"token");
        for(int expert=0;expert<32;++expert){
            rad::WeightInfo w;w.name="blk."+std::to_string(layer)+(slot?".ffn_down_exps.":".ffn_gate_up_exps.")+std::to_string(rank*32+expert)+".weight";w.source=w.name;
            w.decl.dtype=RAD_U8;w.decl.rank=2;w.decl.shape[0]=slot?5120:4608;w.decl.shape[1]=slot?2304:5120;
            w.decl.access=RAD_ACCESS_CONDITIONAL;w.decl.shard=RAD_SHARD_NONE;w.decl.group.layer=layer;w.decl.group.expert=expert;w.decl.group.slot=slot;
            w.layout_tag="ggml_mxfp4";w.stored_bytes=slot?6266880:12533760;w.identity=false;
            op.weights.push_back(rad_weight(p.weights.size()));p.weights.push_back(std::move(w));
        }
        p.ops.push_back(std::move(op));
    }
    return p;
}
} //namespace dspark_scope_fixture
