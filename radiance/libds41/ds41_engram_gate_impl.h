#pragma once
#include "ds41_engram_gate.h"
#if defined(__HIPCC__)
#define DS41_ENGRAM_GATE_HD __host__ __device__ inline
#else
#define DS41_ENGRAM_GATE_HD inline
#endif
namespace rad::ds41::engram_gate {
DS41_ENGRAM_GATE_HD int64_t preflight_items(const Plan& p) {
    int64_t count=0;
    for(int i=0;i<p.output_index;++i)if(i!=mask_operand||p.mask_index>=0)count+=math::count(p.t[i]);
    return count;
}
DS41_ENGRAM_GATE_HD int check_item(const Plan& p,int64_t item) {
    for(int i=0;i<p.output_index;++i) {
        if(i==mask_operand&&p.mask_index<0)continue;
        const int64_t count=math::count(p.t[i]);
        if(item<count) {
            if(i==p.mask_index) {const auto mask=math::id(p.t[i],item);return mask==0||mask==1?RAD_OK:RAD_E_INVAL;}
            return math::finite(math::load(p.t[i],item))?RAD_OK:RAD_E_INVAL;
        }
        item-=count;
    }
    return RAD_E_INVAL;
}
/* copysign is intentional: +0 and -0 are different clamped gate inputs in the
 * pinned model.py:362. sign(dot)*sqrt(abs(dot)) would change both zero cases. */
DS41_ENGRAM_GATE_HD float from_dot(float dot,float clamp) {
    const float rooted=::copysignf(::sqrtf(::fmaxf(::fabsf(dot),clamp)),dot);
    return math::sigmoid(rooted);
}
DS41_ENGRAM_GATE_HD int evaluate(const Plan& p,int64_t job,float* gate) {
    const int64_t token=job/p.copies,copy=job%p.copies;
    float hs=0,ks=0,dot=0;
    for(int64_t d=0;d<p.dim;++d) {
        const float h=math::load(p.t[input],job*p.dim+d);
        const float k=math::load(p.t[projected_kv],token*p.kv_width+copy*p.dim+d);
        hs=hs+h*h;ks=ks+k*k;
        const float weight=math::load(p.t[q_weight],copy*p.dim+d)*math::load(p.t[k_weight],copy*p.dim+d);
        dot=dot+(h*weight)*k;
    }
    if(!math::finite(hs)||!math::finite(ks)||!math::finite(dot))return RAD_E_INVAL;
    const float inv_dim=1.f/float(p.dim),scale=1.f/::sqrtf(float(p.dim));
    const float h_denominator=hs*inv_dim+p.eps,k_denominator=ks*inv_dim+p.eps;
    // A finite square sum plus finite epsilon can still overflow. Checking only
    // rstd would accept sqrt(Inf)->Inf followed by its finite reciprocal zero.
    if(!math::finite(h_denominator)||!math::finite(k_denominator))return RAD_E_INVAL;
    const float rstd=(1.f/::sqrtf(h_denominator))*(1.f/::sqrtf(k_denominator));
    dot=(dot*rstd)*scale;
    if(!math::finite(rstd)||!math::finite(dot))return RAD_E_INVAL;
    *gate=p.mask_index>=0&&!math::id(p.t[p.mask_index],token)?0.f:from_dot(dot,p.clamp);
    return math::finite(*gate)?RAD_OK:RAD_E_INVAL;
}
DS41_ENGRAM_GATE_HD float residual(const Plan& p,int64_t job,int64_t d,float gate) {
    return math::load(p.t[input],job*p.dim+d)+gate*math::load(p.t[projected_kv],(job/p.copies)*p.kv_width+p.width+d);
}
DS41_ENGRAM_GATE_HD bool representable(const Plan& p,float value) {
    if(!math::finite(value))return false;
    const auto dtype=p.t[p.output_index].dtype;
    if(dtype==RAD_F16)return math::finite(packed::half(packed::to_half(value)));
    if(dtype==RAD_BF16)return math::finite(packed::from_bits(uint32_t(packed::to_bf16(value))<<16));
    return true;
}
/* A second read-only phase rejects overflowing intermediate/output conversions
 * across the entire batch before residuals OR optional gates are written. */
DS41_ENGRAM_GATE_HD int check_job(const Plan& p,int64_t job) {
    float gate=0;const int rc=evaluate(p,job,&gate);if(rc!=RAD_OK)return rc;
    for(int64_t d=0;d<p.dim;++d)if(!representable(p,residual(p,job,d,gate)))return RAD_E_INVAL;
    return RAD_OK;
}
DS41_ENGRAM_GATE_HD void write_element(const Plan& p,int64_t job,int64_t d,float gate) {
    math::store(p.t[p.output_index],job*p.dim+d,residual(p,job,d,gate));
}
DS41_ENGRAM_GATE_HD void write_gate(const Plan& p,int64_t job,float gate) {
    if(p.gates_index>=0)math::store(p.t[p.gates_index],job,gate);
}
} // namespace rad::ds41::engram_gate
#undef DS41_ENGRAM_GATE_HD
