/* Arithmetic-only portable GPU candidate, separately checked against the CPU
 * interpreted reference, actual Torch casts and AMD's host FP4 implementation.
 * No TileLang execution/quality claim. Callers own descriptors, environment,
 * alias checks, whole-batch transactionality and optional row masking. */
#pragma once
#include "ds41_packed.h"
#if defined(__HIPCC__)
#define DS41_QDQ_HD __host__ __device__ inline
#else
#define DS41_QDQ_HD inline
#endif
namespace rad::ds41::qdq_math {
enum class Profile : uint32_t {window_fp8,compressed_fp4,index_fp4};
DS41_QDQ_HD bool valid(Profile p){return uint32_t(p)<=uint32_t(Profile::index_fp4);}
DS41_QDQ_HD int width(Profile p){return p==Profile::compressed_fp4?16:32;}
DS41_QDQ_HD bool finite(float x){return (packed::float_bits(x)&0x7fffffffu)<0x7f800000u;}
DS41_QDQ_HD float magnitude(float x){return packed::from_bits(packed::float_bits(x)&0x7fffffffu);}
DS41_QDQ_HD float bf16(uint16_t x){return packed::from_bits(uint32_t(x)<<16);}
DS41_QDQ_HD float divide(float a,float b){
    // Compile WITHOUT fast-math/reciprocal reassociation. This local toolchain
    // has no __fdiv_rn or linkable OCML RTE helper. Actual device tests must
    // prove quotient ties across all126 positive E4M3 scale codes, in addition
    // to primitive cast tests; CPU agreement alone is insufficient.
    return a/b;
}
DS41_QDQ_HD int round_bf16(float x,uint16_t* out){
    if(!finite(x))return RAD_E_INVAL;
    const auto code=packed::to_bf16(x);if(!finite(bf16(code)))return RAD_E_INVAL;
    *out=code;return RAD_OK;
}
DS41_QDQ_HD float e4m3(uint8_t code){
    const uint32_t exp=(code>>3)&15u,man=code&7u,sign=uint32_t(code&128u)<<24;
    if((code&127u)==127u)return packed::from_bits(sign|0x7fc00000u);
    if(exp)return packed::from_bits(sign|((exp+120u)<<23)|(man<<20));
    const float value=float(man)*0x1p-9f;
    return packed::from_bits(packed::float_bits(value)|sign);
}
DS41_QDQ_HD int round_e4m3(float x,uint8_t* out){
    if(!finite(x))return RAD_E_INVAL;
    const float a=magnitude(x);if(a>464.f)return RAD_E_INVAL;
    const uint32_t raw=packed::float_bits(a),sign=(packed::float_bits(x)>>24)&128u;
    uint32_t code=0;
    if(a>=448.f)code=126;
    else if(a<0x1p-6f){
        // The subnormal E4M3 quantum is exactly2^-9. The integer conversion
        // below is bounded[0,8), then explicit RNE chooses a tie's even code.
        const float scaled=a*512.f;code=uint32_t(scaled);
        const float remainder=scaled-float(code);
        if(remainder>.5f||(remainder==.5f&&(code&1u)))++code;
    }else{
        const uint32_t rounded=raw+0x7ffffu+((raw>>20)&1u);
        code=(rounded>>20)-(120u<<3);
    }
    *out=uint8_t(sign|code);return RAD_OK;
}
DS41_QDQ_HD float e2m1(uint8_t code){
    const unsigned value=code&7u;
    const float decoded=value<4?float(value)*.5f:value==4?2.f:value==5?3.f:value==6?4.f:6.f;
    return packed::from_bits(packed::float_bits(decoded)|(uint32_t(code&8u)<<28));
}
DS41_QDQ_HD int round_e2m1(float x,uint8_t* out){
    if(!finite(x))return RAD_E_INVAL;
    const float a=magnitude(x);unsigned code=7;
    // Midpoints and parity are exact F32 values. Saturation is part of E2M1;
    // QDQ itself has already clamped its finite quotient to[-6,6].
    const float cuts[]{.25f,.75f,1.25f,1.75f,2.5f,3.5f,5.f};
    for(unsigned lower=0;lower<7;++lower){
        if(a<cuts[lower]){code=lower;break;}
        if(a==cuts[lower]){code=lower+(lower&1u);break;}
    }
    *out=uint8_t(code|((packed::float_bits(x)>>28)&8u));return RAD_OK;
}
DS41_QDQ_HD int scale(Profile profile,float amax,float* out){
    if(!valid(profile)||!finite(amax)||amax<0)return RAD_E_INVAL;
    if(profile==Profile::compressed_fp4){
        if(amax<6.f*0x1p-9f)amax=6.f*0x1p-9f;
        uint8_t code=0;const int rc=round_e4m3(divide(amax,6.f),&code);if(rc!=RAD_OK)return rc;
        *out=e4m3(code);return *out>0&&finite(*out)?RAD_OK:RAD_E_INVAL;
    }
    const bool fp8=profile==Profile::window_fp8;
    const float floor=fp8?1e-4f:6.f*0x1p-126f;if(amax<floor)amax=floor;
    const float product=amax*(fp8?(1.f/448.f):(1.f/6.f));
    const uint32_t raw=packed::float_bits(product),exp=(raw>>23)&255u;
    if(!finite(product)||!(product>0)||!exp)return RAD_E_INVAL;
    const uint32_t rounded=exp+uint32_t((raw&0x7fffffu)!=0);
    if(!rounded||rounded>254u)return RAD_E_INVAL;
    *out=packed::from_bits(rounded<<23);return RAD_OK;
}
DS41_QDQ_HD int value(Profile profile,float input,float factor,uint16_t* out){
    if(!valid(profile)||!finite(input)||!finite(factor)||!(factor>0))return RAD_E_INVAL;
    float normalized=divide(input,factor);if(!finite(normalized))return RAD_E_INVAL;
    const bool fp8=profile==Profile::window_fp8;const float limit=fp8?448.f:6.f;
    if(normalized>limit)normalized=limit;
    if(normalized< -limit)normalized=-limit;
    uint8_t code=0;const int rc=fp8?round_e4m3(normalized,&code):round_e2m1(normalized,&code);
    if(rc!=RAD_OK)return rc;
    return round_bf16((fp8?e4m3(code):e2m1(code))*factor,out);
}
// Small arithmetic test adapter only. GPU kernels use a wave to reduce amax
// and keep one scalar per lane, avoiding a32-element private array per thread.
DS41_QDQ_HD int group(Profile profile,const uint16_t* input,uint16_t* output){
    if(!valid(profile)||!input||!output)return RAD_E_INVAL;
    const int n=width(profile);float maximum=0;
    for(int i=0;i<n;++i){const float x=bf16(input[i]);if(!finite(x))return RAD_E_INVAL;const float a=magnitude(x);if(a>maximum)maximum=a;}
    float factor=0;const int checked=scale(profile,maximum,&factor);if(checked!=RAD_OK)return checked;
    uint16_t temporary[32]{};
    for(int i=0;i<n;++i){const int rc=value(profile,bf16(input[i]),factor,temporary+i);if(rc!=RAD_OK)return rc;}
    for(int i=0;i<n;++i)output[i]=temporary[i];
    return RAD_OK;
}
} // namespace rad::ds41::qdq_math
#undef DS41_QDQ_HD
