/* Experimental floating DS41 shared K=V attention: ordinary fixed WINDOW plus FULL compressed
 * cache. One common block_size uses the existing core consumer contract. No ring/speculation. */
#pragma once
#include "ds41_device_plan.h"
#include "ds41_math_impl.h"
#include <algorithm>
namespace rad::ds41::attention_paged {
enum Operand {query=0,window_cache,window_table,window_used,compressed_cache,compressed_table,
              compressed_used,selected,positions,cu,ctx_lens,sinks,output,tensors_count};
inline constexpr int64_t max_rows=65536,max_sequences=65536,max_context=1048576;
inline constexpr int64_t max_heads=1024,max_dimension=1024,max_window=4096,max_topk=4096;
inline constexpr int64_t scratch_bytes=4,status_offset=0;
struct Plan {
    RadTensor t[tensors_count]{};int64_t spans[tensors_count]{};
    int64_t rows=0,sequences=0,heads=0,dim=0,window=0,topk=0,ratio=0,block=0;
    int64_t window_blocks=0,window_columns=0,compressed_blocks=0,compressed_columns=0;
    float scale=0;int32_t* status=nullptr;
};
inline int validate(const RadArgs* a,Plan* out) {
    if(!a||!out||!a->p||a->n_p<=0||!a->t||a->n_t!=tensors_count)return RAD_E_INVAL;
    Plan p;for(int i=0;i<tensors_count;++i)p.t[i]=a->t[i];
    auto gi=[&](const char* k,int64_t fallback=0){return int64_t(rad_param_geti(a->p,a->n_p,k,fallback));};
    const int64_t capacity=gi("M"),seq_capacity=gi("sequences");
    p.heads=gi("heads");p.dim=gi("head_dim");p.window=gi("window");p.topk=gi("topk");p.ratio=gi("ratio");p.block=gi("block_size");
    p.scale=float(rad_param_getf(a->p,a->n_p,"scale",0));
    if(gi("n_spec",-1)!=0)return RAD_E_UNSUPPORTED;
    const auto dtype=rad_dtype_parse(rad_param_gets(a->p,a->n_p,"dtype",nullptr));if(!math::scalar_dtype(dtype))return RAD_E_DTYPE;
    if(capacity<=0||capacity>max_rows||seq_capacity<=0||seq_capacity>max_sequences||
       p.heads<=0||p.heads>max_heads||p.dim<=0||p.dim>max_dimension||(p.dim&1)||
       p.window<=0||p.window>max_window||p.ratio<0||p.ratio>2||p.block<=0||p.block>65536||
       p.topk<0||p.topk>max_topk||(p.ratio?(p.topk==0||p.block%p.ratio||p.dim%(2*p.ratio)):p.topk!=0)||
       p.t[query].rank!=3||p.t[window_cache].rank!=4||p.t[window_table].rank!=2)return RAD_E_SHAPE;
    if(!(p.scale>0)||!std::isfinite(p.scale))return RAD_E_INVAL;
    p.rows=p.t[query].shape[0];p.sequences=p.t[window_table].shape[0];
    p.window_columns=p.t[window_table].shape[1];p.window_blocks=p.t[window_cache].shape[0];
    if(p.rows<=0||p.rows>capacity||p.sequences<=0||p.sequences>seq_capacity||
       p.window_columns<=0||p.window_columns>max_context*2||p.window_blocks<=0||p.window_blocks>INT_MAX)return RAD_E_SHAPE;
    int rc=RAD_OK;auto check=[&](int i,uint32_t d,std::initializer_list<int64_t> shape){if(rc==RAD_OK)rc=math::tensor(p.t[i],d,shape,&p.spans[i]);};
    check(query,dtype,{p.rows,p.heads,p.dim});check(window_cache,dtype,{p.window_blocks,1,p.block,p.dim});
    check(window_table,RAD_I32,{p.sequences,p.window_columns});check(window_used,RAD_I32,{p.sequences});
    if(p.ratio) {
        if(p.t[compressed_cache].rank!=4||p.t[compressed_table].rank!=2)return RAD_E_SHAPE;
        p.compressed_blocks=p.t[compressed_cache].shape[0];p.compressed_columns=p.t[compressed_table].shape[1];
        if(p.compressed_blocks<=0||p.compressed_blocks>INT_MAX||p.compressed_columns<=0||p.compressed_columns>max_context*2)return RAD_E_SHAPE;
        check(compressed_cache,dtype,{p.compressed_blocks,1,p.block,p.dim/p.ratio});
        check(compressed_table,RAD_I32,{p.sequences,p.compressed_columns});check(compressed_used,RAD_I32,{p.sequences});
        check(selected,RAD_I32,{p.rows,p.topk});
    } else for(int i=compressed_cache;i<=selected;++i)if(p.t[i].data||p.t[i].rank)return RAD_E_SHAPE;
    check(positions,RAD_I32,{p.rows});check(cu,RAD_I32,{p.sequences+1});check(ctx_lens,RAD_I32,{p.sequences});
    check(sinks,RAD_F32,{p.heads});check(output,dtype,{p.rows,p.heads,p.dim});if(rc!=RAD_OK)return rc;
    for(int i=0;i<output;++i)if(math::overlaps(p.t[output].data,p.spans[output],p.t[i].data,p.spans[i]))return RAD_E_INVAL;
    if(!a->scratch||a->scratch_bytes<scratch_bytes)return RAD_E_INVAL;
    if(uintptr_t(a->scratch)%alignof(int32_t))return RAD_E_ALIGN;
    if(uintptr_t(a->scratch)>UINTPTR_MAX-uint64_t(scratch_bytes))return RAD_E_SHAPE;
    for(int i=0;i<tensors_count;++i)if(math::overlaps(a->scratch,scratch_bytes,p.t[i].data,p.spans[i]))return RAD_E_INVAL;
    p.status=static_cast<int32_t*>(a->scratch);*out=p;return RAD_OK;
}
} // namespace rad::ds41::attention_paged
extern "C" {
int ds41_sparse_attn_paged(const RadArgs*,RadStream);
inline int64_t ds41_attention_paged_scratch(const RadArgs*) {return rad::ds41::gpu_scratch_bytes<rad::ds41::attention_paged::Plan>();}
}
