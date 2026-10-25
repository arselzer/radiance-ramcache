/* Isolated scalar-F32 Engram gate/residual baseline. Projection, hashes, table IO,
 * cache/staging ownership and official BF16 intermediate fidelity are outside this op.
 * Payload errors are asynchronous on GPU: scratch status must be collected before
 * publishing outputs. A successful enqueue is not successful device validation. */
#pragma once
#include "ds41_device_plan.h"
#include "ds41_math_impl.h"

namespace rad::ds41::engram_gate {
inline constexpr int64_t max_tokens=65536,max_copies=8,max_dimension=65536;
inline constexpr int64_t scratch_bytes=4,status_offset=0;
enum Operand {input=0,projected_kv=1,q_weight=2,k_weight=3,mask_operand=4,output=5,gate_output=6,operand_count=7};
struct Plan {
    RadTensor t[7]{};int64_t spans[7]{};
    int nt=operand_count,mask_index=-1,output_index=output,gates_index=-1;
    int64_t tokens=0,copies=0,dim=0,width=0,kv_width=0;
    float eps=0,clamp=0;int32_t* status=nullptr;
};
inline int validate(const RadArgs* args,Plan* out) {
    if(!args||!out||!args->t||!args->p||args->n_p<=0)return RAD_E_INVAL;
    auto integer=[&](const char* key,int fallback=0){return int64_t(rad_param_geti(args->p,args->n_p,key,fallback));};
    const int64_t capacity=integer("M"),mask=integer("token_mask",-1),gates=integer("gates",-1);
    if(mask<0||mask>1||gates<0||gates>1)return RAD_E_INVAL;
    Plan p;if(args->n_t!=operand_count)return RAD_E_INVAL;
    if(capacity<=0||capacity>max_tokens||args->t[0].rank!=3)return RAD_E_SHAPE;
    p.tokens=args->t[0].shape[0];p.copies=integer("copies");p.dim=integer("n");
    if(p.tokens<=0||p.tokens>capacity||p.copies<=0||p.copies>max_copies||p.dim<=0||p.dim>max_dimension)return RAD_E_SHAPE;
    p.width=p.copies*p.dim;p.kv_width=(p.copies+1)*p.dim;
    p.eps=float(rad_param_getf(args->p,args->n_p,"eps",0));
    p.clamp=float(rad_param_getf(args->p,args->n_p,"clamp",0));
    if(!(p.eps>0)||!std::isfinite(p.eps)||!(p.clamp>0)||!std::isfinite(p.clamp))return RAD_E_INVAL;
    const auto dtype=rad_dtype_parse(rad_param_gets(args->p,args->n_p,"dtype",nullptr));
    if(!math::scalar_dtype(dtype))return RAD_E_DTYPE;
    if(mask)p.mask_index=mask_operand;
    if(gates)p.gates_index=gate_output;
    for(int i=0;i<p.nt;++i)p.t[i]=args->t[i];
    if((!mask&&(p.t[mask_operand].data||p.t[mask_operand].rank))||
       (!gates&&(p.t[gate_output].data||p.t[gate_output].rank)))return RAD_E_INVAL;
    int rc=RAD_OK;
    auto check=[&](int i,uint32_t type,std::initializer_list<int64_t> shape){
        if(rc==RAD_OK)rc=math::tensor(p.t[i],type,shape,&p.spans[i]);
    };
    check(input,dtype,{p.tokens,p.copies,p.dim});
    check(projected_kv,dtype,{p.tokens,p.copies+1,p.dim});
    check(q_weight,RAD_F32,{p.copies,p.dim});check(k_weight,RAD_F32,{p.copies,p.dim});
    if(mask)check(p.mask_index,RAD_I32,{p.tokens});
    check(p.output_index,dtype,{p.tokens,p.copies,p.dim});
    if(gates)check(p.gates_index,RAD_F32,{p.tokens,p.copies});
    if(rc!=RAD_OK)return rc;
    for(int i=p.output_index;i<p.nt;++i)for(int j=0;j<p.nt;++j)
        if(i!=j&&math::overlaps(p.t[i].data,p.spans[i],p.t[j].data,p.spans[j]))return RAD_E_INVAL;
    if(!args->scratch||args->scratch_bytes<scratch_bytes)return RAD_E_INVAL;
    if(uintptr_t(args->scratch)%alignof(int32_t))return RAD_E_ALIGN;
    if(uintptr_t(args->scratch)>UINTPTR_MAX-uint64_t(scratch_bytes))return RAD_E_SHAPE;
    for(int i=0;i<p.nt;++i)
        if(math::overlaps(args->scratch,scratch_bytes,p.t[i].data,p.spans[i]))return RAD_E_INVAL;
    p.status=static_cast<int32_t*>(args->scratch);*out=p;return RAD_OK;
}
} // namespace rad::ds41::engram_gate
extern "C" {
int ds41_engram_gate(const RadArgs*,RadStream);
int ds41_ref_engram_gate(const RadArgs*,RadStream) noexcept;
inline int64_t ds41_engram_gate_scratch(const RadArgs*) {
    return rad::ds41::gpu_scratch_bytes<rad::ds41::engram_gate::Plan>();
}
}
