/* Interpreted scalar CPU candidate for pinned official1017 cache QDQ math.
 * This does NOT execute TileLang, register a runtime profile, or establish model
 * quality/device bit parity. BF16 in/out; portable software finite E4M3/E2M1 RNE.
 * See qdq_ref.README.txt for source pins, stronger refusals and format evidence. */
#pragma once
#include "rad_abi.h"
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>

namespace rad::deepseek41::qdq {
static_assert(sizeof(float)==4&&std::numeric_limits<float>::is_iec559);
inline constexpr size_t max_row_width=4096,max_group_width=32,max_groups=max_row_width/16;
enum class Profile {window_fp8,compressed_fp4,index_fp4};
struct Geometry {size_t width=0;bool fp8=false,e4m3_scale=false;};
inline int geometry(Profile profile,Geometry* out) {
    if(!out)return RAD_E_INVAL;
    switch(profile) {
        case Profile::window_fp8:*out={32,true,false};return RAD_OK;
        case Profile::compressed_fp4:*out={16,false,true};return RAD_OK;
        case Profile::index_fp4:*out={32,false,false};return RAD_OK;
    }
    return RAD_E_INVAL;
}
inline uint32_t bits(float x){return std::bit_cast<uint32_t>(x);}
inline float from_bits(uint32_t x){return std::bit_cast<float>(x);}
inline bool finite(float x){return (bits(x)&0x7fffffffu)<0x7f800000u;}
inline float bf16_value(uint16_t x){return from_bits(uint32_t(x)<<16);}
inline int bf16_rne(float value,uint16_t* out) {
    if(!out||!finite(value))return RAD_E_INVAL;
    const uint32_t raw=bits(value),rounded=(raw+0x7fffu+((raw>>16)&1u))&0xffff0000u;
    if(!finite(from_bits(rounded)))return RAD_E_INVAL;
    *out=uint16_t(rounded>>16);return RAD_OK;
}
inline int e4m3_value(uint8_t code,float* out) {
    if(!out)return RAD_E_INVAL;
    const unsigned positive=code&127u;if(positive==127u)return RAD_E_FORMAT;
    const unsigned exp=positive>>3,man=positive&7u;
    const float value=exp?std::ldexp(float(8u+man),int(exp)-10):float(man)*0x1p-9f;
    *out=std::copysign(value,code&128u?-1.f:1.f);return RAD_OK;
}
inline int e4m3_rne(float value,uint8_t* out) {
    if(!out||!finite(value))return RAD_E_INVAL;
    const float magnitude=std::fabs(value);const uint8_t sign=uint8_t((bits(value)>>24)&128u);
    // Checked finite-RNE candidate: E4M3FN's next mathematical value480 is a
    // reserved NaN code; midpoint464 chooses even code126. Beyond that we REFUSE,
    // independently of backend saturation/NaN policy. In particular the observed
    // CPU Torch build saturates even Inf: this does not adopt its overflow mode.
    if(magnitude>464.f)return RAD_E_INVAL;
    if(magnitude>=448.f){*out=uint8_t(sign|126u);return RAD_OK;}
    unsigned hi=126,lo=0;
    while(lo+1<hi) {
        const unsigned middle=(lo+hi)/2;float point=0;e4m3_value(uint8_t(middle),&point);
        if(point<magnitude)lo=middle;else hi=middle;
    }
    float lower=0,upper=0;e4m3_value(uint8_t(lo),&lower);e4m3_value(uint8_t(hi),&upper);
    const double down=double(magnitude)-double(lower),up=double(upper)-double(magnitude);
    const unsigned code=down<up?lo:up<down?hi:(lo&1u?hi:lo);
    *out=uint8_t(sign|code);return RAD_OK;
}
inline constexpr std::array<float,8> e2m1_positive{0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
inline int e2m1_value(uint8_t code,float* out) {
    if(!out)return RAD_E_INVAL;
    if(code>15u)return RAD_E_FORMAT;
    *out=std::copysign(e2m1_positive[code&7u],code&8u?-1.f:1.f);return RAD_OK;
}
inline int e2m1_rne(float value,uint8_t* out) {
    if(!out||!finite(value))return RAD_E_INVAL;
    const float magnitude=std::fabs(value);const uint8_t sign=uint8_t((bits(value)>>28)&8u);
    if(magnitude>=6.f){*out=uint8_t(sign|7u);return RAD_OK;} // finite-only saturated format
    unsigned hi=1;while(e2m1_positive[hi]<magnitude)++hi;
    const unsigned lo=hi-1;
    const double down=double(magnitude)-e2m1_positive[lo],up=double(e2m1_positive[hi])-magnitude;
    const unsigned code=down<up?lo:up<down?hi:(lo&1u?hi:lo);
    *out=uint8_t(sign|code);return RAD_OK;
}
inline int e8m0_value(uint8_t code,float* out) {
    if(!out)return RAD_E_INVAL;
    if(code==255u)return RAD_E_FORMAT;
    *out=from_bits(code?uint32_t(code)<<23:0x00400000u);return RAD_OK;
}
/* Exact pinned IEEE-bit expression, not a general subnormal log2 algorithm.
 * Profile amax floors make its product normal. A subnormal argument would return
 * -126 just as the source formula does; callers must not relabel this std::ceil. */
inline int log2_ceil_bits(float positive,int* out) {
    if(!out||!(positive>0)||!finite(positive))return RAD_E_INVAL;
    const uint32_t raw=bits(positive);*out=int((raw>>23)&255u)-127+int((raw&0x7fffffu)!=0);return RAD_OK;
}
inline int pow2_normal(int exponent,float* out,uint8_t* code=nullptr) {
    if(!out||exponent< -126||exponent>127)return RAD_E_INVAL;
    const auto encoded=uint8_t(exponent+127);*out=from_bits(uint32_t(encoded)<<23);
    if(code)*code=encoded;
    return RAD_OK;
}
inline bool supported_f32_environment() {
    if(std::fegetround()!=FE_TONEAREST)return false;
    // Detect FTZ and DAZ without changing floating-point CONTROL settings.
    // Probe arithmetic, like the QDQ arithmetic itself, may set exception flags.
    volatile float normal=from_bits(0x00800000u),half=.5f,tiny=from_bits(0x00400000u),two=2.f;
    return bits(normal*half)==0x00400000u&&bits(tiny*two)==0x00800000u;
}
struct GroupResult {
    std::array<uint16_t,max_group_width> bf16{};
    std::array<uint8_t,max_group_width> codes{}; // one LOGICAL code per byte, not packed row ABI
    float scale=0;uint8_t scale_code=0,width=0;
};
struct Range {uintptr_t begin=0;size_t bytes=0;};
inline int range(const void* pointer,size_t bytes,size_t alignment,Range* out) {
    if(!out||(!pointer&&bytes))return RAD_E_INVAL;
    const auto begin=reinterpret_cast<uintptr_t>(pointer);
    if(bytes&&begin%alignment)return RAD_E_ALIGN;
    if(bytes>UINTPTR_MAX-begin)return RAD_E_SHAPE;
    *out={begin,bytes};return RAD_OK;
}
inline bool overlaps(Range a,Range b) {
    return a.bytes&&b.bytes&&(a.begin<=b.begin?b.begin-a.begin<a.bytes:a.begin-b.begin<b.bytes);
}
namespace detail {
/* Arithmetic-only body. The public group wrapper validates geometry, spans and
 * host F32 environment first. A future device port must prove its own denormal
 * behavior; it must not import the host fenv/span guards as a GPU proof. */
inline int interpret_group(Geometry spec,std::span<const uint16_t> input,GroupResult* out) {
    int rc=RAD_OK;GroupResult result;result.width=uint8_t(spec.width);float amax=0;
    for(uint16_t raw:input) {
        const float x=bf16_value(raw);if(!finite(x))return RAD_E_INVAL;
        const float magnitude=std::fabs(x);if(magnitude>amax)amax=magnitude;
    }
    if(spec.e4m3_scale) {
        if(amax<6.f*0x1p-9f)amax=6.f*0x1p-9f;
        const float unrounded=amax/6.f;
        if((rc=e4m3_rne(unrounded,&result.scale_code))!=RAD_OK)return rc;
        e4m3_value(result.scale_code,&result.scale);
    } else {
        const float floor=spec.fp8?1e-4f:6.f*0x1p-126f;
        if(amax<floor)amax=floor;
        const float scaled=amax*(spec.fp8?(1.f/448.f):(1.f/6.f));
        // Fixed profile floors must avoid the source bit helper's subnormal case.
        if(!finite(scaled)||!(scaled>0)||(bits(scaled)&0x7f800000u)==0)return RAD_E_INVAL;
        int exponent=0;if((rc=log2_ceil_bits(scaled,&exponent))!=RAD_OK)return rc;
        if((rc=pow2_normal(exponent,&result.scale,&result.scale_code))!=RAD_OK)return rc;
    }
    if(!(result.scale>0)||!finite(result.scale))return RAD_E_INVAL;
    const float limit=spec.fp8?448.f:6.f;
    for(size_t i=0;i<input.size();++i) {
        float normalized=bf16_value(input[i])/result.scale;
        if(!finite(normalized))return RAD_E_INVAL;
        if(normalized>limit)normalized=limit;
        if(normalized< -limit)normalized=-limit;
        if((rc=spec.fp8?e4m3_rne(normalized,&result.codes[i]):e2m1_rne(normalized,&result.codes[i]))!=RAD_OK)return rc;
        float decoded=0;
        if(spec.fp8)e4m3_value(result.codes[i],&decoded);else e2m1_value(result.codes[i],&decoded);
        const float dequantized=decoded*result.scale;
        if((rc=bf16_rne(dequantized,&result.bf16[i]))!=RAD_OK)return rc;
    }
    *out=result;return RAD_OK;
}
} // namespace detail
inline int group(Profile profile,std::span<const uint16_t> input,GroupResult* out) {
    Geometry spec;int rc=geometry(profile,&spec);if(rc!=RAD_OK)return rc;
    if(!out||input.size()!=spec.width)return RAD_E_SHAPE;
    Range source,destination;
    if((rc=range(input.data(),input.size_bytes(),alignof(uint16_t),&source))!=RAD_OK||
       (rc=range(out,sizeof(*out),alignof(GroupResult),&destination))!=RAD_OK)return rc;
    if(overlaps(source,destination))return RAD_E_INVAL;
    if(!supported_f32_environment())return RAD_E_UNSUPPORTED;
    return detail::interpret_group(spec,input,out);
}
struct RowOutput {
    std::span<uint16_t> bf16;
    std::span<uint8_t> codes{},scale_codes{};std::span<float> scales{};
};
inline int row_ranges(Profile profile,size_t width,Range input,RowOutput out) {
    Geometry spec;int rc=geometry(profile,&spec);if(rc!=RAD_OK)return rc;
    if(!width||width>max_row_width||width%spec.width||out.bf16.size()!=width||
       (!out.codes.empty()&&out.codes.size()!=width)||
       (!out.scale_codes.empty()&&out.scale_codes.size()!=width/spec.width)||
       (!out.scales.empty()&&out.scales.size()!=width/spec.width))return RAD_E_SHAPE;
    std::array<Range,4> outputs{};
    if((rc=range(out.bf16.data(),out.bf16.size_bytes(),alignof(uint16_t),&outputs[0]))!=RAD_OK||
       (rc=range(out.codes.data(),out.codes.size_bytes(),1,&outputs[1]))!=RAD_OK||
       (rc=range(out.scale_codes.data(),out.scale_codes.size_bytes(),1,&outputs[2]))!=RAD_OK||
       (rc=range(out.scales.data(),out.scales.size_bytes(),alignof(float),&outputs[3]))!=RAD_OK)return rc;
    for(size_t i=0;i<outputs.size();++i) {
        if(overlaps(input,outputs[i]))return RAD_E_INVAL;
        for(size_t j=0;j<i;++j)if(overlaps(outputs[i],outputs[j]))return RAD_E_INVAL;
    }
    return RAD_OK;
}
/* Synchronous, out-of-place, whole-row transactional. Bounded ~13KiB stack,
 * no allocation. Inputs must remain immutable for this call. No outputs change
 * on malformed metadata, unsupported F32 environment, nonfinite or overflow. */
inline int row(Profile profile,std::span<const uint16_t> input,RowOutput out) {
    if(input.size()>max_row_width)return RAD_E_SHAPE;
    Range source;int rc=range(input.data(),input.size_bytes(),alignof(uint16_t),&source);if(rc!=RAD_OK)return rc;
    if((rc=row_ranges(profile,input.size(),source,out))!=RAD_OK)return rc;
    Geometry spec;geometry(profile,&spec);
    std::array<uint16_t,max_row_width> values{};std::array<uint8_t,max_row_width> codes{};
    std::array<uint8_t,max_groups> scale_codes{};std::array<float,max_groups> scales{};
    for(size_t start=0;start<input.size();start+=spec.width) {
        GroupResult result;if((rc=group(profile,input.subspan(start,spec.width),&result))!=RAD_OK)return rc;
        for(size_t i=0;i<spec.width;++i){values[start+i]=result.bf16[i];codes[start+i]=result.codes[i];}
        scale_codes[start/spec.width]=result.scale_code;scales[start/spec.width]=result.scale;
    }
    std::memcpy(out.bf16.data(),values.data(),input.size_bytes());
    if(!out.codes.empty())std::memcpy(out.codes.data(),codes.data(),out.codes.size_bytes());
    if(!out.scale_codes.empty())std::memcpy(out.scale_codes.data(),scale_codes.data(),out.scale_codes.size_bytes());
    if(!out.scales.empty())std::memcpy(out.scales.data(),scales.data(),out.scales.size_bytes());
    return RAD_OK;
}
/* Explicit optional F32->BF16 input boundary, itself checked before QDQ. This
 * adds <=8KiB bounded stack and refuses input BF16-conversion overflow. */
inline int row_from_f32(Profile profile,std::span<const float> input,RowOutput out) {
    if(input.size()>max_row_width)return RAD_E_SHAPE;
    Range source;int rc=range(input.data(),input.size_bytes(),alignof(float),&source);if(rc!=RAD_OK)return rc;
    if((rc=row_ranges(profile,input.size(),source,out))!=RAD_OK)return rc;
    std::array<uint16_t,max_row_width> converted{};
    for(size_t i=0;i<input.size();++i)if((rc=bf16_rne(input[i],&converted[i]))!=RAD_OK)return rc;
    return row(profile,std::span<const uint16_t>(converted.data(),input.size()),out);
}
} // namespace rad::deepseek41::qdq
