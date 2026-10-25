#pragma once
#include "ds41_index_paged.h"
#if defined(__HIPCC__)
#define DS41_PAGED_HD __host__ __device__ inline
#else
#define DS41_PAGED_HD inline
#endif
namespace rad::ds41::paged_index {
DS41_PAGED_HD int check_row(const Plan& p,int64_t row) {
    const int64_t seq=math::id(p.t[3],row),n=math::id(p.t[4],row);
    if(seq<0||seq>=p.sequences)return RAD_E_STATE;
    if(n<0||n>p.keys)return RAD_E_SHAPE;
    if(p.origins&&math::id(p.t[6],seq)!=0)return RAD_E_UNSUPPORTED;
    return RAD_OK;
}
DS41_PAGED_HD int physical_first(const Plan& p,int64_t row,int64_t key,int64_t* slot) {
    const int64_t pos=key*p.ratio,logical=pos/p.block_size;
    if(logical>=p.logical_blocks)return RAD_E_SHAPE;
    const int64_t block=math::id(p.t[5],int64_t(math::id(p.t[3],row))*p.logical_blocks+logical);
    if(block<0||block>=p.physical_blocks)return RAD_E_STATE;
    *slot=block*p.block_size+pos%p.block_size;return RAD_OK;
}
DS41_PAGED_HD int64_t preflight_items(const Plan& p){return p.rows+p.rows*p.keys+math::count(p.t[0])+math::count(p.t[2]);}
DS41_PAGED_HD int check_item(const Plan& p,int64_t i) {
    if(i<p.rows)return check_row(p,i);
    i-=p.rows;
    if(i<p.rows*p.keys){
        const int64_t row=i/p.keys,key=i%p.keys;const int rc=check_row(p,row);if(rc!=RAD_OK)return rc;
        if(key>=math::id(p.t[4],row))return RAD_OK;
        int64_t first=0;const int mapped=physical_first(p,row,key,&first);if(mapped!=RAD_OK)return mapped;
        for(int64_t d=0;d<p.dim;++d)if(!math::finite(math::load(p.t[1],first*(p.dim/p.ratio)+d)))return RAD_E_INVAL;
        return RAD_OK;
    }
    i-=p.rows*p.keys;const int64_t q=math::count(p.t[0]);
    return math::finite(i<q?math::load(p.t[0],i):math::load(p.t[2],i-q))?RAD_OK:RAD_E_INVAL;
}
DS41_PAGED_HD void score_element(const Plan& p,int64_t at) {
    const int64_t row=at/p.keys,key=at%p.keys;float value=packed::from_bits(0xFF800000u);
    if(key<math::id(p.t[4],row)){
        int64_t first=0;physical_first(p,row,key,&first);value=0;
        for(int64_t h=0;h<p.heads;++h){float dot=0;
            for(int64_t d=0;d<p.dim;++d)dot=dot+math::load(p.t[0],(row*p.heads+h)*p.dim+d)*math::load(p.t[1],first*(p.dim/p.ratio)+d);
            value=value+(dot>0?dot:0.f)*(math::load(p.t[2],row*p.heads+h)*p.scale);
        }
    }
    math::store(p.t[7],at,value);
}
}
#undef DS41_PAGED_HD
