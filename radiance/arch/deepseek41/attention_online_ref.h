/* Bounded scalar mathematical candidate for pinned official1017 kernel.py:311-389.
 * Online64 F32 max/denominator, BF16 UNnormalized exp before PV, sink last.
 * CPU std::exp and explicit scalar reductions are NOT TileLang/GEMM/device parity.
 * See attention_online_ref.README.txt for pin, refusals and precise arithmetic policy. */
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
namespace rad::deepseek41::attention_online {
static_assert(sizeof(float)==4&&std::numeric_limits<float>::is_iec559);
inline constexpr int64_t block=64,max_heads=32,max_dimension=1024,max_candidates=640;
struct View {
    int64_t heads=0,dimension=0,kv_rows=0;
    std::span<const uint16_t> query;      // Exact BF16[heads,dimension].
    std::span<const uint16_t> kv;         // Exact BF16[kv_rows,dimension], shared K=V.
    std::span<const float> sinks;        // Exact F32[heads], finite or negative infinity.
    std::span<const int32_t> selected;   // At most640; order/multiplicity/-1 slots retained.
    float scale=0;
};
struct Workspace {
    std::array<float,max_dimension> numerator{};
    std::array<float,block> scores{};
    std::array<uint16_t,block> probability_bf16{};
    std::array<int32_t,block> ids{};
    std::array<uint16_t,max_heads*max_dimension> transactional_output{};
};
namespace detail {
inline uint32_t bits(float x){return std::bit_cast<uint32_t>(x);}
inline float bf16(uint16_t x){return std::bit_cast<float>(uint32_t(x)<<16);}
inline bool finite(float x){return (bits(x)&0x7FFFFFFFu)<0x7F800000u;}
inline bool negative_infinity(float x){return bits(x)==0xFF800000u;}
inline int encode(float x,uint16_t* out){
    if(!out||!finite(x))return RAD_E_INVAL;
    const uint32_t raw=bits(x),rounded=raw+0x7FFFu+((raw>>16)&1u);
    if((rounded&0x7F800000u)==0x7F800000u)return RAD_E_INVAL;
    *out=uint16_t(rounded>>16);return RAD_OK;
}
// Volatile destinations specify individual F32 boundaries, including no fused multiply-add.
// This still does not prescribe the native GEMM/reduction tree or device exp lowering.
inline float add(float a,float b){volatile float r=a+b;return r;}
inline float subtract(float a,float b){volatile float r=a-b;return r;}
inline float multiply(float a,float b){volatile float r=a*b;return r;}
inline float divide(float a,float b){volatile float r=a/b;return r;}
inline float exponential(float a){volatile float r=std::exp(a);return r;}
inline bool supported_environment(){
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
    // Compile-time arithmetic assumptions must not erase the checked finite/IEEE profile.
    return false;
#else
    if(std::fegetround()!=FE_TONEAREST)return false;
    volatile float normal=std::bit_cast<float>(0x00800000u),half=.5f,tiny=std::bit_cast<float>(0x00400000u),two=2.f;
    return bits(normal*half)==0x00400000u&&bits(tiny*two)==0x00800000u;
#endif
}
struct Range {uintptr_t begin=0;size_t bytes=0;};
inline int range(const void* data,size_t count,size_t width,size_t alignment,Range* out){
    if(!out||count>SIZE_MAX/width||(!data&&count))return RAD_E_INVAL;
    const auto begin=reinterpret_cast<uintptr_t>(data);const size_t bytes=count*width;
    if(count&&begin%alignment)return RAD_E_ALIGN;
    if(bytes>UINTPTR_MAX-begin)return RAD_E_SHAPE;
    *out={begin,bytes};return RAD_OK;
}
inline bool overlaps(Range a,Range b){
    return a.bytes&&b.bytes&&(a.begin<=b.begin?uint64_t(b.begin-a.begin)<a.bytes:uint64_t(a.begin-b.begin)<b.bytes);
}
inline int validate(const View& v,std::span<uint16_t> out,Workspace* workspace){
    if(!workspace)return RAD_E_INVAL;
    if(v.heads<=0||v.heads>max_heads||v.dimension<=0||v.dimension>max_dimension||v.kv_rows<0||v.kv_rows>INT32_MAX||
       v.selected.size()>size_t(max_candidates))return RAD_E_SHAPE;
    const uint64_t qcount=uint64_t(v.heads)*uint64_t(v.dimension),kcount=uint64_t(v.kv_rows)*uint64_t(v.dimension);
    if(qcount>SIZE_MAX/sizeof(uint16_t)||kcount>SIZE_MAX/sizeof(uint16_t)||v.query.size()!=qcount||v.kv.size()!=kcount||
       v.sinks.size()!=size_t(v.heads)||out.size()!=qcount)return RAD_E_SHAPE;
    if(!(v.scale>0)||!finite(v.scale))return RAD_E_INVAL;
    std::array<Range,6> ranges{};int rc=range(v.query.data(),size_t(qcount),2,alignof(uint16_t),&ranges[0]);if(rc!=RAD_OK)return rc;
    rc=range(v.kv.data(),size_t(kcount),2,alignof(uint16_t),&ranges[1]);if(rc!=RAD_OK)return rc;
    rc=range(v.sinks.data(),v.sinks.size(),4,alignof(float),&ranges[2]);if(rc!=RAD_OK)return rc;
    rc=range(v.selected.data(),v.selected.size(),4,alignof(int32_t),&ranges[3]);if(rc!=RAD_OK)return rc;
    rc=range(out.data(),out.size(),2,alignof(uint16_t),&ranges[4]);if(rc!=RAD_OK)return rc;
    rc=range(workspace,1,sizeof(Workspace),alignof(Workspace),&ranges[5]);if(rc!=RAD_OK)return rc;
    for(int i=0;i<4;++i)if(overlaps(ranges[4],ranges[size_t(i)]))return RAD_E_INVAL;
    for(int i=0;i<5;++i)if(overlaps(ranges[5],ranges[size_t(i)]))return RAD_E_INVAL;
    if(!supported_environment())return RAD_E_UNSUPPORTED;
    // Establish every ID before reading ANY selected KV payload (protected unused rows allowed).
    for(int32_t id:v.selected)if(id< -1||(id>=0&&int64_t(id)>=v.kv_rows))return RAD_E_SHAPE;
    for(uint16_t x:v.query)if(!finite(bf16(x)))return RAD_E_INVAL;
    for(float sink:v.sinks)if(!finite(sink)&&!negative_infinity(sink))return RAD_E_INVAL;
    for(int32_t id:v.selected)if(id>=0)for(int64_t d=0;d<v.dimension;++d)
        if(!finite(bf16(v.kv[size_t(int64_t(id)*v.dimension+d)])))return RAD_E_INVAL;
    return RAD_OK;
}
} // namespace detail
/* No allocation. Workspace must be exclusive to this call and byte-disjoint from all live
 * input spans and out; its contents are unspecified on refusal. Caller output changes ONLY
 * after the entire candidate has passed metadata, payload and derived-arithmetic validation. */
inline int evaluate(const View& v,std::span<uint16_t> out,Workspace* workspace){
    const int validated=detail::validate(v,out,workspace);if(validated!=RAD_OK)return validated;
    auto& w=*workspace;
    for(int64_t head=0;head<v.heads;++head){
        for(int64_t d=0;d<v.dimension;++d)w.numerator[size_t(d)]=0.f;
        float maximum=-1e30f,denominator=0.f;
        for(size_t base=0;base<v.selected.size();base+=size_t(block)){
            float next_maximum=maximum;
            for(int64_t k=0;k<block;++k){
                const int32_t id=base+size_t(k)<v.selected.size()?v.selected[base+size_t(k)]:-1;w.ids[size_t(k)]=id;
                float score=-std::numeric_limits<float>::infinity();
                if(id>=0){float dot=0;
                    for(int64_t d=0;d<v.dimension;++d){
                        const float product=detail::multiply(detail::bf16(v.query[size_t(head*v.dimension+d)]),detail::bf16(v.kv[size_t(int64_t(id)*v.dimension+d)]));
                        if(!detail::finite(product))return RAD_E_INVAL;
                        dot=detail::add(dot,product);if(!detail::finite(dot))return RAD_E_INVAL;
                    }
                    score=detail::multiply(dot,v.scale);if(!detail::finite(score))return RAD_E_INVAL;
                }
                w.scores[size_t(k)]=score;if(score>next_maximum)next_maximum=score;
            }
            const float alpha=detail::exponential(detail::subtract(maximum,next_maximum));
            if(!detail::finite(alpha)||alpha<0||alpha>1)return RAD_E_INVAL;
            float block_sum=0;
            for(int64_t k=0;k<block;++k){
                const float probability=detail::exponential(detail::subtract(w.scores[size_t(k)],next_maximum));
                if(!detail::finite(probability)||probability<0)return RAD_E_INVAL;
                block_sum=detail::add(block_sum,probability);if(!detail::finite(block_sum))return RAD_E_INVAL;
                const int encoded=detail::encode(probability,&w.probability_bf16[size_t(k)]);if(encoded!=RAD_OK)return encoded;
            }
            denominator=detail::add(detail::multiply(denominator,alpha),block_sum);if(!detail::finite(denominator))return RAD_E_INVAL;
            for(int64_t d=0;d<v.dimension;++d){w.numerator[size_t(d)]=detail::multiply(w.numerator[size_t(d)],alpha);if(!detail::finite(w.numerator[size_t(d)]))return RAD_E_INVAL;}
            // Preserve all64 slots, including padded KV=+0 contributions. Do not sort/dedup IDs.
            for(int64_t k=0;k<block;++k)for(int64_t d=0;d<v.dimension;++d){
                const float value=w.ids[size_t(k)]<0?0.f:detail::bf16(v.kv[size_t(int64_t(w.ids[size_t(k)])*v.dimension+d)]);
                const float product=detail::multiply(detail::bf16(w.probability_bf16[size_t(k)]),value);
                if(!detail::finite(product))return RAD_E_INVAL;
                w.numerator[size_t(d)]=detail::add(w.numerator[size_t(d)],product);if(!detail::finite(w.numerator[size_t(d)]))return RAD_E_INVAL;
            }
            maximum=next_maximum;
        }
        const float sink_difference=detail::subtract(v.sinks[size_t(head)],maximum),sink_exp=detail::exponential(sink_difference);
        if(std::isnan(sink_difference)||std::isnan(sink_exp)||sink_exp<0)return RAD_E_INVAL;
        // Positive infinity is allowed ONLY at this final sink stage. Literal finite-sink,
        // all-padding rows often reach it because the running maximum stays at-1e30.
        denominator=detail::add(denominator,sink_exp);if(std::isnan(denominator)||!(denominator>0))return RAD_E_INVAL;
        for(int64_t d=0;d<v.dimension;++d){
            const float value=detail::divide(w.numerator[size_t(d)],denominator);
            const int encoded=detail::encode(value,&w.transactional_output[size_t(head*v.dimension+d)]);if(encoded!=RAD_OK)return encoded;
        }
    }
    std::memcpy(out.data(),w.transactional_output.data(),out.size_bytes());return RAD_OK;
}
} // namespace rad::deepseek41::attention_online
