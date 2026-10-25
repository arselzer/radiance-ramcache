/* Local DS41 FFN boundaries, plus the existing mhc_post schema's scalar baseline. This is not a
 * distributed expert implementation: complete expert outputs must already be rounded before
 * combine, and the initial combine profile explicitly refuses tp_world != 1. */
#pragma once
#include "ds41_device_plan.h"
#include "ds41_math_impl.h"
namespace rad::ds41::ffn {
enum class Op {expert_act,moe_combine,mhc_post};
inline constexpr int64_t scratch_bytes=4,status_offset=0;
inline constexpr int max_experts=4096;
struct Plan {
    Op op{};RadTensor t[5]{};int64_t spans[5]{};int nt=0;
    int64_t rows=0,routes=0,n=0,experts=0,hc=0;float limit=0;int32_t* status=nullptr;
};
inline int validate(Op op,const RadArgs* a,Plan* out) {
    if(static_cast<unsigned>(op)>2||!a||!out||!a->t||!a->p||a->n_p<=0)return RAD_E_INVAL;
    const int nt=op==Op::expert_act?4:5;if(a->n_t!=nt)return RAD_E_INVAL;
    Plan p;p.op=op;p.nt=nt;for(int i=0;i<nt;++i)p.t[i]=a->t[i];
    const auto gi=[&](const char* key){return int64_t(rad_param_geti(a->p,a->n_p,key,0));};
    const auto dtype=rad_dtype_parse(rad_param_gets(a->p,a->n_p,"dtype",nullptr));
    if(!math::scalar_dtype(dtype))return RAD_E_DTYPE;
    const int64_t capacity=gi("M");p.n=gi("n");
    if(capacity<=0||capacity>INT_MAX||p.n<=0||p.n>INT_MAX)return RAD_E_SHAPE;
    const auto& row_source=p.t[op==Op::moe_combine?3:0];
    if(row_source.rank!=2)return RAD_E_SHAPE;
    p.rows=row_source.shape[0];
    if(p.rows<=0||p.rows>capacity||p.rows>INT_MAX/p.n)return RAD_E_SHAPE;
    int rc=RAD_OK;
    auto check=[&](int i,uint32_t d,std::initializer_list<int64_t> shape) {
        if(rc==RAD_OK)rc=math::tensor(p.t[i],d,shape,&p.spans[i]);
    };
    if(op==Op::expert_act) {
        p.limit=float(rad_param_getf(a->p,a->n_p,"limit",-1));
        if(p.limit<0||!std::isfinite(p.limit))return RAD_E_INVAL;
        check(0,dtype,{p.rows,p.n});check(1,dtype,{p.rows,p.n});check(2,RAD_F32,{p.rows});check(3,dtype,{p.rows,p.n});
    } else if(op==Op::moe_combine) {
        const auto* world=rad_param_find(a->p,a->n_p,"tp_world");
        if(!world||world->kind!=RAD_P_INT)return RAD_E_INVAL;
        if(world->ival!=1)return RAD_E_UNSUPPORTED;
        if(p.t[0].rank!=2)return RAD_E_SHAPE;
        p.routes=p.t[0].shape[0];p.experts=gi("n_expert");
        if(p.routes<=0||p.routes>gi("R")||gi("R")>INT_MAX||p.experts<=0||p.experts>max_experts)return RAD_E_SHAPE;
        check(0,dtype,{p.routes,p.n});check(1,RAD_I32,{p.routes});check(2,RAD_I32,{p.routes});
        check(3,dtype,{p.rows,p.n});check(4,dtype,{p.rows,p.n});
    } else {
        p.hc=gi("hc");if(p.hc<2||p.hc>8||p.n>INT_MAX/p.hc||p.rows>INT_MAX/(p.n*p.hc))return RAD_E_SHAPE;
        check(0,dtype,{p.rows,p.n});check(1,dtype,{p.rows,p.hc*p.n});check(2,RAD_F32,{p.rows,p.hc});
        check(3,RAD_F32,{p.rows,p.hc*p.hc});check(4,dtype,{p.rows,p.hc*p.n});
    }
    if(rc!=RAD_OK)return rc;
    const int yi=nt-1;for(int i=0;i<yi;++i)
        if(math::overlaps(p.t[yi].data,p.spans[yi],p.t[i].data,p.spans[i]))return RAD_E_INVAL;
    if(!a->scratch||a->scratch_bytes<scratch_bytes)return RAD_E_INVAL;
    if(uintptr_t(a->scratch)%alignof(int32_t))return RAD_E_ALIGN;
    if(uintptr_t(a->scratch)>UINTPTR_MAX-uint64_t(scratch_bytes))return RAD_E_SHAPE;
    for(int i=0;i<nt;++i)if(math::overlaps(a->scratch,scratch_bytes,p.t[i].data,p.spans[i]))return RAD_E_INVAL;
    p.status=static_cast<int32_t*>(a->scratch);*out=p;return RAD_OK;
}
} // namespace rad::ds41::ffn
extern "C" {
int ds41_expert_act(const RadArgs*,RadStream);
int ds41_moe_combine(const RadArgs*,RadStream);
// Identical operands/parameters to existing mhc_post, not a new operator schema.
int ds41_mhc_post(const RadArgs*,RadStream);
// GPU reservation includes the staged descriptor; CPU emulate still needs only4B.
inline int64_t ds41_ffn_scratch(const RadArgs*) {return rad::ds41::gpu_scratch_bytes<rad::ds41::ffn::Plan>();}
}
