/* A converter-only fixture: one packed weight and no execution graph. Kept outside arch/ so it
 * cannot be discovered by a serving process. Tests supply a private RADIANCE_HOME explicitly. */
#include "rad_builder.h"
#include "rad_import.h"
extern "C" {
uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
const RadPluginInfo* rad_plugin_info(void) {
    static const RadPluginInfo info{RAD_PLUGIN_ARCH,"radtest_importarch","0.1","lossless importer conversion fixture","host"};
    return &info;
}
const char* rad_arch_id(void) { return "radtest"; }
const char* rad_arch_quant(void) { return ""; }
int rad_arch_serves_imports(void) { return 1; }
int rad_arch_declare(RadBuilder* b, const RadModelMeta*, const RadBuildCtx*) {
    RadNameMap map{};map.declared="logical";map.mode=RAD_MAP_COPY;map.n_src=1;map.src[0]="packed";
    for(int& i:map.src_index)i=-1;
    if(rad_decl_name_map(b,&map)!=RAD_OK)return RAD_E_INVAL;
    RadEncoding enc{};int64_t shape[RAD_MAX_RANK]{};uint32_t rank=0;
    if(rad_weight_encoding(b,"logical",&enc,shape,&rank)!=RAD_OK)return RAD_E_INVAL;
    RadWeightDecl weight{};weight.dtype=enc.plane[0].dtype;weight.rank=rank;
    for(uint32_t i=0;i<rank;++i)weight.shape[i]=shape[i];
    weight.access=RAD_ACCESS_PER_TOKEN;weight.shard=RAD_SHARD_NONE;
    weight.group.layer=weight.group.expert=-1;
    const rad_weight handle=rad_decl_weight(b,"logical",&weight);
    if(!handle)return RAD_E_INVAL;
    return rad_decl_op(b,"test_import_read",nullptr,0,&handle,1)?RAD_OK:RAD_E_INVAL;
}
void rad_arch_step(RadCtx*,const RadBatch*) {}
}
