#pragma once
#include "ds41_compress.h"
#if defined(__HIPCC__)
#define DS41_COMPRESS_HD __host__ __device__ inline
#else
#define DS41_COMPRESS_HD inline
#endif
namespace rad::ds41::compress {
DS41_COMPRESS_HD uint32_t magic(int64_t ratio) {return 0x44534300u|uint32_t(ratio);}
DS41_COMPRESS_HD int64_t state_offset(const Plan& p,int64_t slot,int64_t plane,int64_t column) {
    return math::offset(p.t[pending_state],(slot*2+plane)*(p.dimension+1)+column);
}
DS41_COMPRESS_HD uint32_t word(const Plan& p,int64_t slot,int64_t plane,int64_t column) {
    return static_cast<const uint32_t*>(p.t[pending_state].data)[state_offset(p,slot,plane,column)];
}
DS41_COMPRESS_HD void put_word(const Plan& p,int64_t slot,int64_t plane,int64_t column,uint32_t value) {
    static_cast<uint32_t*>(p.t[pending_state].data)[state_offset(p,slot,plane,column)]=value;
}
DS41_COMPRESS_HD void put_int(const RadTensor& t,int64_t i,int32_t value) {
    static_cast<int32_t*>(t.data)[math::offset(t,i)]=value;
}
DS41_COMPRESS_HD int check_sequence(const Plan& p,int64_t s) {
    const int64_t begin=math::id(p.t[cu],s),end=math::id(p.t[cu],s+1);
    if(math::id(p.t[cu],0)!=0||math::id(p.t[cu],p.sequences)!=p.tokens||
       begin<0||end<begin||end>p.tokens)return RAD_E_SHAPE;
    if(begin==end)return RAD_OK; // unused slots/has_init/accepted are deliberately ignored
    const int32_t slot=math::id(p.t[state_ids],s),warm=math::id(p.t[has_init],s);
    if(slot<0||slot>=p.states||warm<0)return RAD_E_STATE;
    if(p.accepted&&math::id(p.t[num_accepted],s)!=1)return RAD_E_UNSUPPORTED;
    for(int64_t other=0;other<s;++other) {
        const int64_t lo=math::id(p.t[cu],other),hi=math::id(p.t[cu],other+1);
        if(lo>=0&&hi>lo&&hi<=p.tokens&&math::id(p.t[state_ids],other)==slot)return RAD_E_INVAL;
    }
    const int64_t first=math::id(p.t[positions],begin);
    if(first<0||first+(end-begin)>INT32_MAX)return RAD_E_SHAPE;
    for(int64_t row=begin;row<end;++row)
        if(math::id(p.t[positions],row)!=first+(row-begin))return RAD_E_STATE;
    if(!warm&&first!=0)return RAD_E_STATE;
    if(warm) {
        if(word(p,slot,1,p.dimension)!=magic(p.ratio)||word(p,slot,0,p.dimension)!=uint32_t(first))return RAD_E_STATE;
        if(p.ratio==2&&(first&1))for(int64_t d=0;d<p.dimension;++d)
            if(!math::finite(packed::from_bits(word(p,slot,0,d)))||!math::finite(packed::from_bits(word(p,slot,1,d))))return RAD_E_INVAL;
    }
    return RAD_OK;
}
DS41_COMPRESS_HD int64_t preflight_items(const Plan& p) {
    return p.tokens*p.dimension*(p.ratio==2?2:1)+p.sequences;
}
DS41_COMPRESS_HD int check_item(const Plan& p,int64_t i) {
    const int64_t values=p.tokens*p.dimension;
    if(i<values)return math::finite(math::load(p.t[projected_kv],i))?RAD_OK:RAD_E_INVAL;
    i-=values;
    if(p.ratio==2) {if(i<values)return math::finite(math::load(p.t[gate_scores],i))?RAD_OK:RAD_E_INVAL;i-=values;}
    return check_sequence(p,i);
}
DS41_COMPRESS_HD float pool(float previous,float current,float old_score,float score) {
    const float maximum=old_score>score?old_score:score;
    const float a=::expf(old_score-maximum),b=::expf(score-maximum);
    return previous*(a/(a+b))+current*(b/(a+b));
}
/* Caller captures begin/end/slot/first before any thread can publish a new header. Columns are
 * independent; each owns its pending pair and iterates original token rows in order. */
DS41_COMPRESS_HD void column(const Plan& p,int64_t begin,int64_t end,int64_t slot,int64_t first,int64_t d) {
    float previous=0,old_score=0;
    if(p.ratio==2&&(first&1)) {previous=packed::from_bits(word(p,slot,0,d));old_score=packed::from_bits(word(p,slot,1,d));}
    for(int64_t row=begin;row<end;++row) {
        const float kv=math::load(p.t[projected_kv],row*p.dimension+d);float value=kv;
        if(p.ratio==2) {
            const float score=math::load(p.t[gate_scores],row*p.dimension+d);
            if(((first+row-begin)&1)==0) {previous=kv;old_score=score;value=0.f;}
            else value=pool(previous,kv,old_score,score);
        }
        math::store(p.t[pooled],row*p.dimension+d,value);
    }
    if(p.ratio==2) {
        put_word(p,slot,0,d,packed::float_bits(previous));put_word(p,slot,1,d,packed::float_bits(old_score));
    }
}
DS41_COMPRESS_HD void finish_sequence(const Plan& p,int64_t begin,int64_t end,int64_t slot,int64_t first) {
    for(int64_t row=begin;row<end;++row) {
        const int64_t pos=first+row-begin;const bool emitted=p.ratio==1||(pos&1);
        put_int(p.t[emit],row,emitted?1:0);put_int(p.t[first_position],row,emitted?int32_t(pos+1-p.ratio):-1);
    }
    put_word(p,slot,0,p.dimension,uint32_t(first+end-begin));put_word(p,slot,1,p.dimension,magic(p.ratio));
}
} // namespace rad::ds41::compress
#undef DS41_COMPRESS_HD
