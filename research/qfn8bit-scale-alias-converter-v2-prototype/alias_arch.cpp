// Converter-only synthetic fixture, outside serving scan. No model execution.
#include "rad_builder.h"
#include <cstring>
namespace {
constexpr const char* ng="blk.0.ple_ngram.weight";
constexpr const char* prefix="model.language_model.layers.1.ple.ple_embedding.ngram_embedding.";
int mapped(RadBuilder* b,const char* logical,const char* const* names,int n){
    RadNameMap map{};map.declared=logical;map.mode=n==1?RAD_MAP_COPY:RAD_MAP_CONCAT;map.n_src=n;
    for(int& index:map.src_index)index=-1;
    for(int i=0;i<n;++i)map.src[i]=names[i];return rad_decl_name_map(b,&map);
}
int declare_weight(RadBuilder* b,const char* name,bool global){
    RadEncoding enc{};uint32_t rank=0;int64_t shape[RAD_MAX_RANK]{};
    if(rad_weight_encoding(b,name,&enc,shape,&rank)!=RAD_OK||rank!=2)return RAD_E_SHAPE;
    if(global){const auto* scale=rad_enc_plane(&enc,"scale");
        if(enc.n_planes!=2||enc.plane[0].dtype!=RAD_F8E4M3||!scale||scale->dtype!=RAD_BF16||scale->block[0]||scale->block[1])return RAD_E_DTYPE;}
    RadWeightDecl w{};w.dtype=enc.plane[0].dtype;w.rank=rank;for(uint32_t i=0;i<rank;++i)w.shape[i]=shape[i];
    w.access=RAD_ACCESS_PER_TOKEN;w.shard=RAD_SHARD_NONE;w.group.layer=w.group.expert=-1;
    const auto h=rad_decl_weight(b,name,&w);if(!h)return RAD_E_INVAL;
    return rad_decl_op(b,"test_import_read",nullptr,0,&h,1)?RAD_OK:RAD_E_INVAL;
}
}
extern "C" {
uint32_t rad_plugin_abi_version(){return RAD_ABI_VERSION;}
const RadPluginInfo* rad_plugin_info(){static const RadPluginInfo i{RAD_PLUGIN_ARCH,"qfn8bit_alias_fixture","1","bounded converter-only PLE/expert fixture","host"};return &i;}
const char* rad_arch_id(){return "qfn8bitfixture";}
const char* rad_arch_quant(){return "";}
int rad_arch_declare(RadBuilder* b,const RadModelMeta*,const RadBuildCtx*){
    (void)prefix;
    const char* shards[]={"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_1.weight",
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_2.weight"};
    if(mapped(b,ng,shards,3)!=RAD_OK)return RAD_E_INVAL;
    const char* gu[]={"model.language_model.layers.0.mlp.experts.0.gate_proj.weight","model.language_model.layers.0.mlp.experts.0.up_proj.weight"};
    const char* dn[]={"model.language_model.layers.0.mlp.experts.0.down_proj.weight"};
    if(mapped(b,"fixture.expert.gate_up.weight",gu,2)!=RAD_OK||mapped(b,"fixture.expert.down.weight",dn,1)!=RAD_OK)return RAD_E_INVAL;
    const int a=declare_weight(b,ng,true);if(a<0)return a;
    const int c=declare_weight(b,"fixture.expert.gate_up.weight",false);if(c<0)return c;
    return declare_weight(b,"fixture.expert.down.weight",false);
}
void rad_arch_step(RadCtx*,const RadBatch*){} // Never invoked; no inference fixture.
}
