#pragma once
#include "ds41_latent_emit.h"
namespace rad::ds41::latent_emit {
inline int validate(const RadArgs* a,Plan* out) {
    if(!a||!out||!a->t||a->n_t!=operand_count||!a->p||a->n_p<=0)return RAD_E_INVAL;
    Plan p;for(int i=0;i<operand_count;++i)p.t[i]=a->t[i];
    const int64_t capacity=rad_param_geti(a->p,a->n_p,"M",0),d=rad_param_geti(a->p,a->n_p,"dimension",0),
                  ratio=rad_param_geti(a->p,a->n_p,"ratio",0),b=rad_param_geti(a->p,a->n_p,"block_size",0),
                  block_cap=rad_param_geti(a->p,a->n_p,"n_blocks",0);
    const auto dtype=rad_dtype_parse(rad_param_gets(a->p,a->n_p,"dtype",nullptr));
    if(!math::scalar_dtype(dtype))return RAD_E_DTYPE;
    if(capacity<=0||capacity>max_rows||d<=0||d>1024||b!=4||block_cap<0||p.t[x].rank!=2||p.t[cache].rank!=4)return RAD_E_SHAPE;
    auto& l=p.latent;l.rows=p.t[x].shape[0];l.blocks=p.t[cache].shape[0];l.dtype=dtype;
    if(l.rows<=0||l.rows>capacity||l.blocks<=0||l.blocks>INT_MAX||(block_cap&&l.blocks>block_cap))return RAD_E_SHAPE;
    int rc=deepseek41::state::make_layout(d,ratio,b,rad_dtype_bits(dtype)/8,&l.layout);if(rc!=RAD_OK)return rc;
    if(!deepseek41::state::product(l.blocks,b,&l.slots))return RAD_E_SHAPE;
    rc=math::tensor(p.t[x],dtype,{l.rows,d},&p.spans[x]);if(rc!=RAD_OK)return rc;l.pitch=p.t[x].stride[0];
    for(int i:{positions,slot_mapping}) {
        const auto type=p.t[i].dtype;if(type!=RAD_I32&&type!=RAD_I64)return RAD_E_DTYPE;
        rc=math::tensor(p.t[i],type,{l.rows},&p.spans[i]);if(rc!=RAD_OK)return rc;
    }
    for(int i:{emit,status}) {rc=math::tensor(p.t[i],RAD_I32,{l.rows},&p.spans[i]);if(rc!=RAD_OK)return rc;}
    rc=math::tensor(p.t[cache],dtype,{l.blocks,1,b,d/ratio},&p.spans[cache]);if(rc!=RAD_OK)return rc;
    int64_t stride=1;
    for(int axis=3;axis>=0;--axis){if(p.t[cache].stride[axis]!=stride)return RAD_E_STRIDE;stride*=p.t[cache].shape[axis];}
    for(int out:{cache,status})for(int i=0;i<operand_count;++i)
        if(i!=out&&math::overlaps(p.t[out].data,p.spans[out],p.t[i].data,p.spans[i]))return RAD_E_INVAL;
    if(!a->scratch||a->scratch_bytes<scratch_bytes)return RAD_E_INVAL;
    if(uintptr_t(a->scratch)%alignof(int32_t))return RAD_E_ALIGN;
    if(uintptr_t(a->scratch)>UINTPTR_MAX-uint64_t(scratch_bytes))return RAD_E_SHAPE;
    for(int i=0;i<operand_count;++i)if(math::overlaps(a->scratch,scratch_bytes,p.t[i].data,p.spans[i]))return RAD_E_INVAL;
    p.flag=static_cast<int32_t*>(a->scratch);*out=p;return RAD_OK;
}
} // namespace rad::ds41::latent_emit
