#pragma once
#include "ds41_latent.h"
#include "ds41_packed_args.h"
namespace rad::ds41::latent {
inline int validate(const RadArgs* a,bool store,Launch* out) {
    if(!a||!a->t||!out||a->n_t!=(store?5:4))return RAD_E_INVAL;
    const int64_t cap=rad_param_geti(a->p,a->n_p,"M",0),d=rad_param_geti(a->p,a->n_p,"dimension",0);
    const int64_t ratio=rad_param_geti(a->p,a->n_p,"ratio",0),b=rad_param_geti(a->p,a->n_p,"block_size",0);
    const auto& cache_input=a->t[store?3:0];
    const int64_t blocks=cache_input.rank==4?cache_input.shape[0]:0;
    const int64_t block_cap=rad_param_geti(a->p,a->n_p,"n_blocks",0);
    const auto dtype=rad_dtype_parse(rad_param_gets(a->p,a->n_p,"dtype",nullptr));
    if(dtype!=RAD_F32&&dtype!=RAD_F16&&dtype!=RAD_BF16)return RAD_E_DTYPE;
    if(cap<=0||cap>4096||d<=0||d>1024||b<=0||b>65536||blocks<=0||blocks>INT_MAX||
       block_cap<0||(block_cap>0&&blocks>block_cap))return RAD_E_SHAPE;
    Launch l;l.dtype=dtype;l.blocks=blocks;
    int status=deepseek41::state::make_layout(d,ratio,b,rad_dtype_bits(dtype)/8,&l.layout);
    if(status!=RAD_OK)return status;
    int64_t bytes=0;
    if(!deepseek41::state::product(blocks,l.layout.page_bytes,&bytes)||
       !deepseek41::state::product(blocks,b,&l.slots))return RAD_E_SHAPE;
    const int cache_i=store?3:0,x_i=store?0:2,status_i=store?4:3;
    const auto* t=a->t;const auto& cache=t[cache_i];
    if(!cache.data)return RAD_E_INVAL;
    if(cache.dtype!=dtype)return RAD_E_DTYPE;
    if(cache.rank!=4||cache.shape[0]!=blocks||cache.shape[1]!=1||cache.shape[2]!=b||cache.shape[3]!=d/ratio)
        return RAD_E_SHAPE;
    if(!rad_tensor_is_contiguous(&cache))return RAD_E_STRIDE;
    if(uintptr_t(cache.data)%unsigned(l.layout.element_bytes))return RAD_E_ALIGN;
    if(t[x_i].rank!=2)return RAD_E_SHAPE;
    l.rows=t[x_i].shape[0];if(l.rows<=0||l.rows>cap)return RAD_E_SHAPE;
    int64_t spans[5]{};spans[cache_i]=bytes;
    status=packed::matrix(t[x_i],dtype,l.rows,d,&spans[x_i]);if(status!=RAD_OK)return status;
    l.pitch=t[x_i].stride[0];
    if(store) {
        for(int i:{1,2}) {status=packed::vector(t[i],RAD_I64,l.rows,&spans[i]);if(status!=RAD_OK)return status;}
    } else {status=packed::vector(t[1],RAD_I64,l.rows,&spans[1]);if(status!=RAD_OK)return status;}
    status=packed::vector(t[status_i],RAD_I32,l.rows,&spans[status_i]);if(status!=RAD_OK)return status;
    const int dest=store?cache_i:x_i;
    for(int i=0;i<a->n_t;++i) {
        if(i!=dest&&packed::alias(t[dest].data,spans[dest],t[i].data,spans[i]))return RAD_E_INVAL;
        if(i!=status_i&&packed::alias(t[status_i].data,spans[status_i],t[i].data,spans[i]))return RAD_E_INVAL;
    }
    *out=l;return RAD_OK;
}
inline int store_row(const Launch& l,int64_t position,int64_t final_slot,int64_t* first) {
    const int64_t ratio=l.layout.ratio,b=l.layout.block_size;
    if(position<0||position%ratio!=ratio-1||final_slot<ratio-1||final_slot>=l.slots||
       final_slot%b!=position%b)return RAD_E_SHAPE;
    const int64_t start=final_slot-ratio+1;
    if(start/b!=final_slot/b)return RAD_E_SHAPE;
    *first=start;return RAD_OK;
}
inline int gather_row(const Launch& l,int64_t first) {
    return first>=0&&first<=l.slots-l.layout.ratio&&first%l.layout.block_size%l.layout.ratio==0?RAD_OK:RAD_E_SHAPE;
}
} // namespace rad::ds41::latent
