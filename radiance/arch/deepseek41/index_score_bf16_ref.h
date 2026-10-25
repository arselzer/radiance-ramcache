/* Bounded scalar BF16 score-boundary candidate, not native GEMM/TP/model execution.
 * Pinned Indexer.forward model.py:556-558; see README and actual CPU Torch fixture.
 * BF16 projected-weight scale, BF16 dot/ReLU/product, F32 head sum -> BF16.
 * All outputs are transactional; caller-owned bounded stage Workspace, no allocation. */
#pragma once
#include "attention_online_ref.h"
namespace rad::deepseek41::index_score_bf16 {
inline constexpr int64_t max_heads=32,max_dimension=1024,max_keys=640;
struct View {
    int64_t heads=0,dimension=0,keys=0;
    std::span<const uint16_t> query;             // [heads,D] BF16, already prepared.
    std::span<const uint16_t> key;               // [keys,D] BF16, already prepared.
    std::span<const uint16_t> projected_weights; // [heads] BF16, BEFORE supplied factor.
    float factor=0;
};
struct Workspace {
    std::array<uint16_t,max_heads> scaled_weights{};
    // Bounded ONE-query stage receipt. Each head has max_keys reserved slots.
    std::array<uint16_t,max_heads*max_keys> dot{},relu{},products{};
    std::array<uint16_t,max_keys> transactional_scores{};
};
namespace detail {
namespace fp=rad::deepseek41::attention_online::detail;
inline int validate(const View& v,std::span<uint16_t> out,Workspace* w){
    if(!w)return RAD_E_INVAL;
    if(v.heads<=0||v.heads>max_heads||v.dimension<=0||v.dimension>max_dimension||v.keys<0||v.keys>max_keys)return RAD_E_SHAPE;
    const uint64_t q=uint64_t(v.heads)*uint64_t(v.dimension),k=uint64_t(v.keys)*uint64_t(v.dimension);
    if(q>SIZE_MAX/2||k>SIZE_MAX/2||v.query.size()!=q||v.key.size()!=k||v.projected_weights.size()!=size_t(v.heads)||out.size()!=size_t(v.keys))return RAD_E_SHAPE;
    if(!(v.factor>0)||!fp::finite(v.factor))return RAD_E_INVAL;
    std::array<fp::Range,5> spans{};int rc=fp::range(v.query.data(),size_t(q),2,alignof(uint16_t),&spans[0]);if(rc!=RAD_OK)return rc;
    rc=fp::range(v.key.data(),size_t(k),2,alignof(uint16_t),&spans[1]);if(rc!=RAD_OK)return rc;
    rc=fp::range(v.projected_weights.data(),size_t(v.heads),2,alignof(uint16_t),&spans[2]);if(rc!=RAD_OK)return rc;
    rc=fp::range(out.data(),out.size(),2,alignof(uint16_t),&spans[3]);if(rc!=RAD_OK)return rc;
    rc=fp::range(w,1,sizeof(Workspace),alignof(Workspace),&spans[4]);if(rc!=RAD_OK)return rc;
    for(int i=0;i<3;++i)if(fp::overlaps(spans[3],spans[size_t(i)]))return RAD_E_INVAL;
    for(int i=0;i<4;++i)if(fp::overlaps(spans[4],spans[size_t(i)]))return RAD_E_INVAL;
    if(!fp::supported_environment())return RAD_E_UNSUPPORTED;
    for(uint16_t x:v.query)if(!fp::finite(fp::bf16(x)))return RAD_E_INVAL;
    for(uint16_t x:v.key)if(!fp::finite(fp::bf16(x)))return RAD_E_INVAL;
    for(uint16_t x:v.projected_weights)if(!fp::finite(fp::bf16(x)))return RAD_E_INVAL;
    return RAD_OK;
}
} // namespace detail
inline int evaluate(const View& v,std::span<uint16_t> out,Workspace* workspace){
    const int checked=detail::validate(v,out,workspace);if(checked!=RAD_OK)return checked;
    namespace fp=detail::fp;auto& w=*workspace;
    for(int64_t h=0;h<v.heads;++h){
        const float scaled=fp::multiply(fp::bf16(v.projected_weights[size_t(h)]),v.factor);
        const int rc=fp::encode(scaled,&w.scaled_weights[size_t(h)]);if(rc!=RAD_OK)return rc;
    }
    for(int64_t h=0;h<v.heads;++h)for(int64_t k=0;k<v.keys;++k){
        float sum=0;
        for(int64_t d=0;d<v.dimension;++d){const float product=fp::multiply(fp::bf16(v.query[size_t(h*v.dimension+d)]),fp::bf16(v.key[size_t(k*v.dimension+d)]));
            if(!fp::finite(product))return RAD_E_INVAL;
            sum=fp::add(sum,product);if(!fp::finite(sum))return RAD_E_INVAL;}
        const size_t at=size_t(h*max_keys+k);int rc=fp::encode(sum,&w.dot[at]);if(rc!=RAD_OK)return rc;
        // Preserve -0 under ReLU, matching the supplied CPU expression's boundary.
        w.relu[at]=fp::bf16(w.dot[at])<0?0:w.dot[at];
        const float product=fp::multiply(fp::bf16(w.relu[at]),fp::bf16(w.scaled_weights[size_t(h)]));
        rc=fp::encode(product,&w.products[at]);if(rc!=RAD_OK)return rc;
    }
    for(int64_t k=0;k<v.keys;++k){float sum=0;
        for(int64_t h=0;h<v.heads;++h){sum=fp::add(sum,fp::bf16(w.products[size_t(h*max_keys+k)]));if(!fp::finite(sum))return RAD_E_INVAL;}
        const int rc=fp::encode(sum,&w.transactional_scores[size_t(k)]);if(rc!=RAD_OK)return rc;
    }
    if(!out.empty())std::memcpy(out.data(),w.transactional_scores.data(),out.size_bytes());
    return RAD_OK;
}
} // namespace rad::deepseek41::index_score_bf16
