#pragma once
// Independent bounded scalar reference: canonical record bytes + supplied inputs only.
// No GPU stage/readback is used to construct an expected later stage.
#include "rad_plugin.h"
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "The bounded FFN scalar oracle requires no-fast-math and finite-math-only disabled"
#endif
#include <algorithm>
#include <array>
#include <bit>
#include <bitset>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>
namespace ffn_reference {
constexpr int tokens=5,top=2,rows=tokens*top,hidden=128,experts=4;
struct Result {
    std::array<uint16_t,rows*2*hidden> gu{};
    std::array<uint16_t,rows*hidden> product{},dn{};
    std::array<uint8_t,rows*hidden> q{};
    std::array<float,rows> scale{};
    std::array<uint16_t,tokens*hidden> gathered{};
    size_t positive_gates=0,negative_gates=0,clamped_gates=0,clamped_ups=0,unclamped_pairs=0,positive_products=0,negative_products=0,distinct_codes=0,fp8_ties=0,math_bf16_disagreements=0;
    double bf16_midpoint_margin=1,fp8_midpoint_margin=1;
};
inline uint16_t bf16(float x){const uint32_t bits=std::bit_cast<uint32_t>(x);return uint16_t((bits+0x7fffu+((bits>>16)&1u))>>16);}
inline float f32(uint16_t x){return std::bit_cast<float>(uint32_t(x)<<16);}
inline float fp8(uint8_t x){const int e=(x>>3)&15,m=x&7;if(e==15&&m==7)throw std::runtime_error("NaN FP8 input outside bounded oracle");const float v=std::ldexp(float(e?8+m:m),e?e-10:-9);return x&128?-v:v;}
inline float fp4(uint8_t x){constexpr float magnitude[]{0,.5f,1,1.5f,2,3,4,6};return magnitude[x&7]*(x&8?-1.f:1.f);}
inline double bf16_margin(float x){
    const uint32_t u=std::bit_cast<uint32_t>(std::fabs(x));if(!u)return 1;
    const long double lo=std::bit_cast<float>(u&0xffff0000u),hi=std::bit_cast<float>((u&0xffff0000u)+0x10000u);
    return double(std::fabs((long double)std::fabs(x)-(lo+hi)/2)/(hi-lo));
}
inline uint8_t quantize(float value,Result& out){
    const float a=std::fabs(value);
    if(a>0&&a<448){
        double distance=1;
        for(unsigned i=0;i<126;++i){const long double lo=fp8(uint8_t(i)),hi=fp8(uint8_t(i+1));distance=std::min(distance,double(std::fabs((long double)a-(lo+hi)/2)/(hi-lo)));}
        out.fp8_midpoint_margin=std::min(out.fp8_midpoint_margin,distance);out.fp8_ties+=distance==0;
    }
    // Public scalar RNE converter is independently implemented from GPU packed conversion.
    return rad_f32_to_fp8e4m3(value);
}
inline Result evaluate(const std::array<std::vector<std::byte>,experts>& records,
                       const std::array<int32_t,rows>& routes,std::span<const uint8_t> input,
                       std::span<const float> input_scales,std::span<const float> weights){
    if(input.size()!=tokens*hidden||input_scales.size()!=tokens||weights.size()!=rows)throw std::runtime_error("oracle exact input span");
    for(const auto& r:records)if(r.size()!=32768)throw std::runtime_error("oracle record stride");
    for(auto e:routes)if(e<0||e>=experts)throw std::runtime_error("oracle route range");
    Result out;std::array<int32_t,rows> sorted{};int at=0;
    for(int e=0;e<experts;++e)for(int flat=0;flat<rows;++flat)if(routes[size_t(flat)]==e)sorted[size_t(at++)]=flat;
    auto projection=[&](int p,int e,int n,int input_row,std::span<const uint8_t> q,float activation_scale){
        const auto& bytes=records[size_t(e)];const size_t codes=p?20480:0,scales=p?28672:16384;float value=0;
        for(int block=0;block<4;++block){float dot=0;
            for(int k=block*32;k<(block+1)*32;++k){const uint8_t pair=std::to_integer<uint8_t>(bytes[codes+size_t(n*hidden+k)/2]);const uint8_t code=uint8_t((pair>>((k&1)*4))&15);dot+=fp8(q[size_t(input_row*hidden+k)])*fp4(code);}
            const auto exp=std::to_integer<uint8_t>(bytes[scales+size_t(n*4+block)]);const float ws=std::ldexp(1.f,int(exp)-127);value+=(dot*activation_scale)*ws;
        }
        if(!std::isfinite(value))throw std::runtime_error("oracle projection nonfinite");
        return bf16(value);
    };
    for(int row=0;row<rows;++row){const int flat=sorted[size_t(row)],token=flat/top,e=routes[size_t(flat)];
        for(int n=0;n<2*hidden;++n)out.gu[size_t(row*2*hidden+n)]=projection(0,e,n,token,input,input_scales[size_t(token)]);
        float maximum=0;
        for(int n=0;n<hidden;++n){const float gate=f32(out.gu[size_t(row*2*hidden+n)]),up=f32(out.gu[size_t(row*2*hidden+hidden+n)]);out.positive_gates+=gate>0;out.negative_gates+=gate<0;out.clamped_gates+=gate>10;out.clamped_ups+=std::fabs(up)>10;out.unclamped_pairs+=gate<=10&&std::fabs(up)<=10;
            const float g=std::min(gate,10.f),u=std::clamp(up,-10.f,10.f);const float denominator=1.f+std::exp(-g);const float product=(g/denominator)*u;
            if(!std::isfinite(product))throw std::runtime_error("oracle product nonfinite");
            const uint16_t bits=bf16(product);out.product[size_t(row*hidden+n)]=bits;const float rounded=f32(bits);maximum=std::max(maximum,std::fabs(rounded));out.positive_products+=rounded>0;out.negative_products+=rounded<0;out.bf16_midpoint_margin=std::min(out.bf16_midpoint_margin,bf16_margin(product));
            const long double mathematical=((long double)g/(1+std::exp(-(long double)g)))*(long double)u;
            out.math_bf16_disagreements+=bf16(float(mathematical))!=bits;
        }
        // Preserve the pinned F32 multiply-by-reciprocal constant, reciprocal, multiply, clamp.
        const float scale=maximum>0?maximum*(1.f/448.f):1.f,reciprocal=1.f/scale;out.scale[size_t(row)]=scale;
        for(int n=0;n<hidden;++n){const float normalized=std::clamp(f32(out.product[size_t(row*hidden+n)])*reciprocal,-448.f,448.f);out.q[size_t(row*hidden+n)]=quantize(normalized,out);}
    }
    std::bitset<256> codes;for(auto code:out.q)codes.set(code);out.distinct_codes=codes.count();
    for(int row=0;row<rows;++row){const int e=routes[size_t(sorted[size_t(row)])];for(int n=0;n<hidden;++n)out.dn[size_t(row*hidden+n)]=projection(1,e,n,row,out.q,out.scale[size_t(row)]);}
    std::array<int32_t,rows> inverse{};for(int i=0;i<rows;++i)inverse[size_t(sorted[size_t(i)])]=i;
    for(int token=0;token<tokens;++token)for(int n=0;n<hidden;++n){float sum=0;
        for(int slot=0;slot<top;++slot){const int flat=token*top+slot;sum=std::fma(weights[size_t(flat)],f32(out.dn[size_t(inverse[size_t(flat)]*hidden+n)]),sum);}
        if(!std::isfinite(sum))throw std::runtime_error("oracle gather nonfinite");
        out.gathered[size_t(token*hidden+n)]=bf16(sum);
    }
    return out;
}
} // namespace ffn_reference
