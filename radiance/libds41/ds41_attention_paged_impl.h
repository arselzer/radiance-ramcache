#pragma once
#include "ds41_attention_paged.h"
#if defined(__HIPCC__)
#define DS41_ATTN_PAGED_HD __host__ __device__ inline
#else
#define DS41_ATTN_PAGED_HD inline
#endif
namespace rad::ds41::attention_paged {
/* Phase 1 consumes only bounded sequence/row metadata, never tables or cache payloads. All
 * sequences complete this phase before phase 2 may use cu for ownership or dereference pages. */
DS41_ATTN_PAGED_HD int check_sequence(const Plan& p,int64_t seq) {
    const int64_t begin=math::id(p.t[cu],seq),end=math::id(p.t[cu],seq+1);
    if(begin<0||end<begin||end>p.rows||(seq==0&&begin!=0)||(seq+1==p.sequences&&end!=p.rows))return RAD_E_SHAPE;
    if(begin==end)return RAD_OK; // unused sequences' context, lengths and table contents are not consumed
    const int64_t ctx=math::id(p.t[ctx_lens],seq),chunk_end=ctx+end-begin;
    if(ctx<0||chunk_end>max_context)return RAD_E_SHAPE;
    const int64_t used=math::id(p.t[window_used],seq),first=chunk_end-used;
    const int64_t oldest=ctx>=p.window?ctx-p.window+1:0;
    if(used<=0||used>p.window_columns*p.block||first<0||first%p.block||first>oldest)return RAD_E_STATE;
    if(p.ratio) {
        const int64_t complete=math::id(p.t[compressed_used],seq);
        if(complete!=chunk_end||complete>p.compressed_columns*p.block)return RAD_E_STATE;
    }
    for(int64_t row=begin;row<end;++row) {
        const int64_t pos=math::id(p.t[positions],row);
        if(pos!=ctx+row-begin)return RAD_E_SHAPE;
        if(p.ratio)for(int64_t k=0;k<p.topk;++k) {
            const int64_t id=math::id(p.t[selected],row*p.topk+k);
            if(id< -1||id>=(pos+1)/p.ratio)return RAD_E_SHAPE;
        }
    }
    return RAD_OK;
}
DS41_ATTN_PAGED_HD int64_t sequence_of(const Plan& p,int64_t row) {
    int64_t lo=0,hi=p.sequences;
    while(lo<hi) {const int64_t mid=(lo+hi)/2;if(row>=math::id(p.t[cu],mid+1))lo=mid+1;else hi=mid;}
    return lo;
}
DS41_ATTN_PAGED_HD int64_t first_window_position(const Plan& p,int64_t seq) {
    return int64_t(math::id(p.t[ctx_lens],seq))+math::id(p.t[cu],seq+1)-math::id(p.t[cu],seq)-math::id(p.t[window_used],seq);
}
DS41_ATTN_PAGED_HD int64_t window_count(const Plan& p,int64_t row) {
    const int64_t available=int64_t(math::id(p.t[positions],row))+1;return available<p.window?available:p.window;
}
DS41_ATTN_PAGED_HD int64_t candidate_count(const Plan& p,int64_t row) {return window_count(p,row)+p.topk;}
struct Candidate {int cache=window_cache;int64_t scalar=0;bool present=false;};
/* Called only after complete phase-1 validation. Each candidate maps independently, so even a
 * corrupt selected page cannot cause an out-of-bounds cache read in the page-validation phase. */
DS41_ATTN_PAGED_HD int map_candidate(const Plan& p,int64_t row,int64_t at,Candidate* out) {
    Candidate value;const int64_t seq=sequence_of(p,row),count=window_count(p,row);
    int64_t original=0,logical=0,block=0;
    if(at<count) {
        original=int64_t(math::id(p.t[positions],row))+1-count+at;
        const int64_t first=first_window_position(p,seq);logical=(original-first)/p.block;
        if(original<first||logical<0||logical>=p.window_columns)return RAD_E_STATE;
        block=math::id(p.t[window_table],seq*p.window_columns+logical);
        if(block<0||block>=p.window_blocks)return RAD_E_STATE;
        value.scalar=(block*p.block+original%p.block)*p.dim;
    } else {
        const int64_t id=math::id(p.t[selected],row*p.topk+at-count);
        if(id<0) {*out=value;return RAD_OK;}
        original=id*p.ratio;logical=original/p.block;
        if(logical>=p.compressed_columns)return RAD_E_STATE;
        block=math::id(p.t[compressed_table],seq*p.compressed_columns+logical);
        if(block<0||block>=p.compressed_blocks)return RAD_E_STATE;
        value.cache=compressed_cache;value.scalar=(block*p.block+original%p.block)*(p.dim/p.ratio);
    }
    value.present=true;*out=value;return RAD_OK;
}
DS41_ATTN_PAGED_HD int check_pages(const Plan& p,int64_t row) {
    for(int64_t k=0;k<candidate_count(p,row);++k) {Candidate c;const int rc=map_candidate(p,row,k,&c);if(rc!=RAD_OK)return rc;}
    return RAD_OK;
}
/* Phase 3 starts only after every selected page passed phase 2. Unselected pages/payloads and
 * the unused portion of a partial C2 group are never read or required finite. */
DS41_ATTN_PAGED_HD int check_payload(const Plan& p,int64_t row) {
    for(int64_t h=0;h<p.heads;++h) {
        const float sink=math::load(p.t[sinks],h);if(!math::finite(sink)&&!math::negative_infinity(sink))return RAD_E_INVAL;
        for(int64_t d=0;d<p.dim;++d)if(!math::finite(math::load(p.t[query],(row*p.heads+h)*p.dim+d)))return RAD_E_INVAL;
    }
    for(int64_t k=0;k<candidate_count(p,row);++k) {
        Candidate c;const int rc=map_candidate(p,row,k,&c);if(rc!=RAD_OK)return rc;
        if(c.present)for(int64_t d=0;d<p.dim;++d)if(!math::finite(math::load(p.t[c.cache],c.scalar+d)))return RAD_E_INVAL;
    }
    return RAD_OK;
}
DS41_ATTN_PAGED_HD float score(const Plan& p,int64_t row,int64_t head,int64_t at) {
    Candidate c;map_candidate(p,row,at,&c);if(!c.present)return packed::from_bits(0xFF800000u);
    float dot=0;for(int64_t d=0;d<p.dim;++d)dot=dot+math::load(p.t[query],(row*p.heads+head)*p.dim+d)*math::load(p.t[c.cache],c.scalar+d);
    return dot*p.scale;
}
DS41_ATTN_PAGED_HD float maximum(const Plan& p,int64_t head,const float* scores,int64_t count) {
    float value=math::load(p.t[sinks],head);for(int64_t k=0;k<count;++k)if(scores[k]>value)value=scores[k];return value;
}
DS41_ATTN_PAGED_HD float denominator(const Plan& p,int64_t head,const float* probabilities,int64_t count,float maximum) {
    if(maximum==packed::from_bits(0xFF800000u))return 0;
    float value=::expf(math::load(p.t[sinks],head)-maximum);
    for(int64_t k=0;k<count;++k)value=value+probabilities[k];
    return value;
}
DS41_ATTN_PAGED_HD void attend_column(const Plan& p,int64_t row,int64_t head,int64_t d,const float* probabilities,int64_t count,float den) {
    float value=0;
    if(den!=0)for(int64_t k=0;k<count;++k) {
        Candidate c;map_candidate(p,row,k,&c);if(c.present)value=value+probabilities[k]*math::load(p.t[c.cache],c.scalar+d);
    }
    math::store(p.t[output],(row*p.heads+head)*p.dim+d,den==0?0.f:value/den);
}
} // namespace rad::ds41::attention_paged
#undef DS41_ATTN_PAGED_HD
