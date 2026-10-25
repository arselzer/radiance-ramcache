/* Isolated full-head floating index selection over FULL paged keys. No dense score/mask plane,
 * architecture registration, hidden allocation or partial-head TP reduction. */
#pragma once
#include "ds41_index_paged_impl.h"
#include "ds41_device_plan.h"
#include <cstring>
namespace rad::ds41::index_stream {
enum class Mode {source,consumer,plain};
enum Operand {query=0,cache,head_weights,sequence_ids,visible,block_table,origins,candidates_in,
              selected_out,candidates_out,operand_count};
inline constexpr int64_t max_topk=1024,max_top_blocks=2048,tile_keys=128,scratch_bytes=4;
using Entry=index::Entry;
struct Plan {
    Mode mode{};paged_index::Plan paged;RadTensor t[operand_count]{};int64_t spans[operand_count]{};
    int64_t topk=0,top_blocks=0,candidate_block_size=0;
};
inline int64_t shared_bytes(const Plan& p){return (p.topk+p.top_blocks)*int64_t(sizeof(Entry))+tile_keys*4;}
static_assert((max_topk+max_top_blocks)*sizeof(Entry)+tile_keys*4+32<32768);
inline int validate(const RadArgs* a,Plan* out){
    if(!a||!out||!a->t||a->n_t!=operand_count||!a->p||a->n_p<=0)return RAD_E_INVAL;
    Plan p;for(int i=0;i<operand_count;++i)p.t[i]=a->t[i];
    const char* mode=rad_param_gets(a->p,a->n_p,"mode",nullptr);
    if(!mode)return RAD_E_INVAL;
    if(!std::strcmp(mode,"source"))p.mode=Mode::source;else if(!std::strcmp(mode,"consumer"))p.mode=Mode::consumer;
    else if(!std::strcmp(mode,"plain"))p.mode=Mode::plain;else return RAD_E_INVAL;
    auto gi=[&](const char* k){return int64_t(rad_param_geti(a->p,a->n_p,k,0));};
    auto& b=p.paged;const int64_t capacity=gi("M"),seq_capacity=gi("sequences"),global_heads=gi("global_heads");
    b.keys=gi("key_rows");b.heads=gi("heads");b.dim=gi("head_dim");b.ratio=gi("ratio");b.block_size=gi("block_size");
    p.topk=gi("topk");p.top_blocks=gi("top_blocks");p.candidate_block_size=gi("candidate_block_size");
    if(global_heads<=0||b.heads<=0||global_heads<b.heads)return RAD_E_SHAPE;
    if(global_heads!=b.heads)return RAD_E_UNSUPPORTED;
    const auto dtype=rad_dtype_parse(rad_param_gets(a->p,a->n_p,"dtype",nullptr));if(!math::scalar_dtype(dtype))return RAD_E_DTYPE;
    if(capacity<=0||capacity>index::max_rows||seq_capacity<=0||seq_capacity>index::max_sequences||
       b.keys<=0||b.keys>index::max_keys||b.heads>index::max_heads||b.dim<=0||b.dim>index::max_dimension||
       (b.ratio!=1&&b.ratio!=2)||b.dim%(2*b.ratio)||b.block_size<=0||b.block_size>65536||b.block_size%b.ratio||
       p.topk<=0||p.topk>max_topk||p.topk>b.keys||p.top_blocks<0||p.top_blocks>max_top_blocks||
       (p.mode==Mode::plain?p.top_blocks!=0:p.top_blocks==0)||p.candidate_block_size<=0||p.candidate_block_size>256||
       p.t[query].rank!=3||p.t[cache].rank!=4||p.t[block_table].rank!=2)return RAD_E_SHAPE;
    b.rows=p.t[query].shape[0];b.physical_blocks=p.t[cache].shape[0];b.sequences=p.t[block_table].shape[0];b.logical_blocks=p.t[block_table].shape[1];
    if(b.rows<=0||b.rows>capacity||b.sequences<=0||b.sequences>seq_capacity||b.physical_blocks<=0||b.physical_blocks>INT_MAX||
       b.logical_blocks<=0||b.logical_blocks>index::max_keys*2)return RAD_E_SHAPE;
    int status=RAD_OK;auto check=[&](int i,uint32_t d,std::initializer_list<int64_t> shape){if(status==RAD_OK)status=math::tensor(p.t[i],d,shape,&p.spans[i]);};
    check(query,dtype,{b.rows,b.heads,b.dim});check(cache,dtype,{b.physical_blocks,1,b.block_size,b.dim/b.ratio});
    check(head_weights,RAD_F32,{b.rows,b.heads});check(sequence_ids,RAD_I32,{b.rows});check(visible,RAD_I32,{b.rows});
    check(block_table,RAD_I32,{b.sequences,b.logical_blocks});b.origins=p.t[origins].data||p.t[origins].rank;
    if(b.origins)check(origins,RAD_I32,{b.sequences});
    if(p.mode==Mode::consumer)check(candidates_in,RAD_I32,{b.rows,p.top_blocks});
    else if(p.t[candidates_in].data||p.t[candidates_in].rank)return RAD_E_SHAPE;
    check(selected_out,RAD_I32,{b.rows,p.topk});
    if(p.mode==Mode::source)check(candidates_out,RAD_I32,{b.rows,p.top_blocks});
    else if(p.t[candidates_out].data||p.t[candidates_out].rank)return RAD_E_SHAPE;
    if(status!=RAD_OK)return status;
    for(int i=selected_out;i<operand_count;++i)for(int j=0;j<operand_count;++j)
        if(i!=j&&math::overlaps(p.t[i].data,p.spans[i],p.t[j].data,p.spans[j]))return RAD_E_INVAL;
    if(!a->scratch||a->scratch_bytes<scratch_bytes)return RAD_E_INVAL;
    if(uintptr_t(a->scratch)%alignof(int32_t))return RAD_E_ALIGN;
    if(uintptr_t(a->scratch)>UINTPTR_MAX-4)return RAD_E_SHAPE;
    for(int i=0;i<operand_count;++i)if(math::overlaps(a->scratch,4,p.t[i].data,p.spans[i]))return RAD_E_INVAL;
    b.scale=(1.f/std::sqrt(float(b.dim)))*(1.f/std::sqrt(float(global_heads)));b.status=static_cast<int32_t*>(a->scratch);
    for(int i=0;i<=origins;++i)b.t[i]=p.t[i];
    *out=p;return RAD_OK;
}
} // namespace rad::ds41::index_stream
extern "C" {
int ds41_index_stream(const RadArgs*,RadStream);
inline int64_t ds41_index_stream_scratch(const RadArgs*){return rad::ds41::gpu_scratch_bytes<rad::ds41::index_stream::Plan>();}
}
