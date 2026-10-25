#pragma once
#include "ds41_transform.h"
#if defined(__HIPCC__)
#define DS41_TRANSFORM_HD __host__ __device__ inline
#else
#define DS41_TRANSFORM_HD inline
#endif
namespace rad::ds41::transform {
DS41_TRANSFORM_HD int64_t preflight_items(const Plan& p) {
    return math::count(p.t[0])+(p.op==Op::rmsnorm?p.dim:p.rows*(1+p.rotary/2));
}
DS41_TRANSFORM_HD int check_item(const Plan& p,int64_t i) {
    const int64_t values=math::count(p.t[0]);
    if(i<values)return math::finite(math::load(p.t[0],i))?RAD_OK:RAD_E_INVAL;
    i-=values;
    if(p.op==Op::rmsnorm)return math::finite(math::load(p.t[1],i))?RAD_OK:RAD_E_INVAL;
    const int64_t row=i/(1+p.rotary/2),pair=i%(1+p.rotary/2)-1;
    const int32_t pos=math::id(p.t[1],row);if(pos<0||pos>=p.positions)return RAD_E_SHAPE;
    if(pair<0)return RAD_OK;
    const float cosine=math::load(p.t[2],int64_t(pos)*p.rotary+2*pair),sine=math::load(p.t[2],int64_t(pos)*p.rotary+2*pair+1);
    const float norm=cosine*cosine+sine*sine;
    return math::finite(cosine)&&math::finite(sine)&&math::finite(norm)&&::fabsf(norm-1.f)<=2e-5f?RAD_OK:RAD_E_INVAL;
}
DS41_TRANSFORM_HD float inverse_rms(const Plan& p,int64_t row) {
    float square=0;for(int64_t d=0;d<p.dim;++d) {const float value=math::load(p.t[0],row*p.dim+d);square=square+value*value;}
    return 1.f/::sqrtf(square/float(p.dim)+p.eps);
}
DS41_TRANSFORM_HD void norm_element(const Plan& p,int64_t row,int64_t d,float inverse) {
    math::store(p.t[2],row*p.dim+d,(math::load(p.t[0],row*p.dim+d)*inverse)*math::load(p.t[1],d));
}
DS41_TRANSFORM_HD void rope_element(const Plan& p,int64_t at) {
    const int64_t d=at%p.dim,tail=p.dim-p.rotary;
    if(d<tail) {
        const int bytes=p.t[0].dtype==RAD_F32?4:2;
        const auto* src=static_cast<const uint8_t*>(p.t[0].data)+math::offset(p.t[0],at)*bytes;
        auto* dst=static_cast<uint8_t*>(p.t[3].data)+math::offset(p.t[3],at)*bytes;
        for(int i=0;i<bytes;++i)dst[i]=src[i];
        return;
    }
    const int64_t relative=d-tail;if(relative&1)return;
    const int64_t row=at/(p.heads*p.dim),pair=relative/2;const int32_t pos=math::id(p.t[1],row);
    const float cosine=math::load(p.t[2],int64_t(pos)*p.rotary+pair*2);
    float sine=math::load(p.t[2],int64_t(pos)*p.rotary+pair*2+1);if(p.inverse)sine=-sine;
    const float a=math::load(p.t[0],at),b=math::load(p.t[0],at+1);
    math::store(p.t[3],at,a*cosine-b*sine);math::store(p.t[3],at+1,a*sine+b*cosine);
}
} // namespace rad::ds41::transform
#undef DS41_TRANSFORM_HD
