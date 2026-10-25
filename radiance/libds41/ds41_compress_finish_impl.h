#pragma once
#include "ds41_compress_finish.h"
#if defined(__HIPCC__)
#define DS41_FINISH_HD __host__ __device__ inline
#else
#define DS41_FINISH_HD inline
#endif
namespace rad::ds41::compress_finish {
DS41_FINISH_HD float rounded(float x,uint32_t dtype){
    return dtype==RAD_F32?x:dtype==RAD_F16?math::packed::half(math::packed::to_half(x)):
        math::packed::from_bits(uint32_t(math::packed::to_bf16(x))<<16);
}
DS41_FINISH_HD float input(const Plan& p,int64_t row,int64_t d){return rounded(math::load(p.t[pooled],row*p.dim+d),p.dtype);}
DS41_FINISH_HD float inverse_rms(const Plan& p,int64_t row){
    float square=0;for(int64_t d=0;d<p.dim;++d){const float x=input(p,row,d);square=square+x*x;}return 1.f/::sqrtf(square/float(p.dim)+p.eps);
}
DS41_FINISH_HD float normalized(const Plan& p,int64_t row,int64_t d,float inverse){return rounded((input(p,row,d)*inverse)*math::load(p.t[gain],d),p.dtype);}
DS41_FINISH_HD int check_row(const Plan& p,int64_t row){
    const int32_t mask=math::id(p.t[emit],row);if(mask!=0&&mask!=1)return RAD_E_INVAL;
    if(!mask)return RAD_OK; // Absolutely no pooled/position/table payload read for inactive rows.
    const int32_t pos=math::id(p.t[first_positions],row);if(pos<0||pos>=p.positions||pos%p.ratio)return RAD_E_SHAPE;
    float square=0;
    for(int64_t d=0;d<p.dim;++d){const float source=math::load(p.t[pooled],row*p.dim+d);if(!math::finite(source))return RAD_E_INVAL;
        const float x=rounded(source,p.dtype);if(!math::finite(x))return RAD_E_INVAL;square=square+x*x;if(!math::finite(square))return RAD_E_INVAL;}
    const float denominator=square/float(p.dim)+p.eps,inverse=1.f/::sqrtf(denominator);
    if(!(denominator>0)||!math::finite(denominator)||!math::finite(inverse))return RAD_E_INVAL;
    for(int64_t d=0;d<p.dim;++d){const float raw=(input(p,row,d)*inverse)*math::load(p.t[gain],d);if(!math::finite(raw)||!math::finite(rounded(raw,p.dtype)))return RAD_E_INVAL;}
    const int64_t tail=p.dim-p.rotary;
    for(int64_t pair=0;pair<p.rotary/2;++pair){
        const float cosine=math::load(p.t[pairs],int64_t(pos)*p.rotary+pair*2),sine=math::load(p.t[pairs],int64_t(pos)*p.rotary+pair*2+1),unit=cosine*cosine+sine*sine;
        if(!math::finite(cosine)||!math::finite(sine)||!math::finite(unit)||::fabsf(unit-1.f)>2e-5f)return RAD_E_INVAL;
        const float a=normalized(p,row,tail+pair*2,inverse),b=normalized(p,row,tail+pair*2+1,inverse),x=a*cosine-b*sine,y=a*sine+b*cosine;
        if(!math::finite(x)||!math::finite(y)||!math::finite(rounded(x,p.dtype))||!math::finite(rounded(y,p.dtype)))return RAD_E_INVAL;
    }
    return RAD_OK;
}
DS41_FINISH_HD int check_item(const Plan& p,int64_t item){return item<p.dim?(math::finite(math::load(p.t[gain],item))?RAD_OK:RAD_E_INVAL):check_row(p,item-p.dim);}
DS41_FINISH_HD void output_element(const Plan& p,int64_t row,int64_t d,float inverse,bool active){
    const int64_t at=row*p.dim+d;
    if(!active){math::store(p.t[unrotated],at,0.f);math::store(p.t[rotated],at,0.f);return;}
    const int64_t tail=p.dim-p.rotary;
    if(d<tail){const float u=normalized(p,row,d,inverse);math::store(p.t[unrotated],at,u);math::store(p.t[rotated],at,u);return;}
    if((d-tail)&1)return;
    const float a=normalized(p,row,d,inverse),b=normalized(p,row,d+1,inverse);const int32_t pos=math::id(p.t[first_positions],row);
    const int64_t pair=(d-tail)/2;const float cosine=math::load(p.t[pairs],int64_t(pos)*p.rotary+pair*2),sine=math::load(p.t[pairs],int64_t(pos)*p.rotary+pair*2+1);
    math::store(p.t[unrotated],at,a);math::store(p.t[unrotated],at+1,b);
    math::store(p.t[rotated],at,a*cosine-b*sine);math::store(p.t[rotated],at+1,a*sine+b*cosine);
}
} // namespace rad::ds41::compress_finish
#undef DS41_FINISH_HD
