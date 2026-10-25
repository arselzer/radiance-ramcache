#pragma once
#include "ds41_index.h"
#if defined(__HIPCC__)
#define DS41_INDEX_HD __host__ __device__ inline
#else
#define DS41_INDEX_HD inline
#endif
namespace rad::ds41::index {
DS41_INDEX_HD int32_t visible(const Plan& p,int64_t row) {return math::id(p.t[p.op==Op::scores?4:1],row);}
DS41_INDEX_HD int check_row(const Plan& p,int64_t row) {
    const int64_t n=visible(p,row);if(n<0||n>p.keys)return RAD_E_SHAPE;
    if(p.op==Op::scores){const int64_t s=math::id(p.t[3],row);if(s<0||s>=p.sequences)return RAD_E_STATE;}
    return RAD_OK;
}
DS41_INDEX_HD int64_t preflight_items(const Plan& p) {
    return p.rows+p.rows*p.keys+(p.op==Op::scores?math::count(p.t[0])+math::count(p.t[2]):0);
}
DS41_INDEX_HD int check_item(const Plan& p,int64_t i) {
    if(i<p.rows)return check_row(p,i);
    i-=p.rows;
    if(i<p.rows*p.keys) {
        const int64_t row=i/p.keys,key=i%p.keys;const int rc=check_row(p,row);if(rc!=RAD_OK)return rc;
        if(key>=visible(p,row))return RAD_OK;
        if(p.op!=Op::scores)return math::finite(math::load(p.t[0],i))?RAD_OK:RAD_E_INVAL;
        const int64_t seq=math::id(p.t[3],row);
        for(int64_t d=0;d<p.dim;++d)if(!math::finite(math::load(p.t[1],(seq*p.keys+key)*p.dim+d)))return RAD_E_INVAL;
        return RAD_OK;
    }
    i-=p.rows*p.keys;
    const int64_t q=math::count(p.t[0]);
    return math::finite(i<q?math::load(p.t[0],i):math::load(p.t[2],i-q))?RAD_OK:RAD_E_INVAL;
}
DS41_INDEX_HD void score_element(const Plan& p,int64_t at) {
    const int64_t row=at/p.keys,key=at%p.keys;float total=packed::from_bits(0xFF800000u);
    if(key<visible(p,row)) {
        total=0;const int64_t seq=math::id(p.t[3],row);
        for(int64_t h=0;h<p.heads;++h) {
            float dot=0;
            for(int64_t d=0;d<p.dim;++d)dot=dot+math::load(p.t[0],(row*p.heads+h)*p.dim+d)*math::load(p.t[1],(seq*p.keys+key)*p.dim+d);
            total=total+(dot>0?dot:0.f)*(math::load(p.t[2],row*p.heads+h)*p.scale);
        }
    }
    math::store(p.t[5],at,total);
}
DS41_INDEX_HD bool better(Entry a,Entry b) {return a.score!=b.score?a.score>b.score:a.id<b.id;}
DS41_INDEX_HD void sift(Entry* heap,int64_t size,int64_t root) {
    for(;;) {
        const int64_t left=root*2+1;if(left>=size)return;
        const int64_t right=left+1;int64_t worse=left;
        if(right<size&&better(heap[left],heap[right]))worse=right;
        if(!better(heap[root],heap[worse]))return;
        const Entry temp=heap[root];heap[root]=heap[worse];heap[worse]=temp;root=worse;
    }
}
DS41_INDEX_HD void offer(Entry* heap,int64_t capacity,int64_t* count,Entry entry) {
    if(*count<capacity) {
        int64_t at=(*count)++;heap[at]=entry;
        while(at) {const int64_t parent=(at-1)/2;if(!better(heap[parent],heap[at]))break;
            const Entry temp=heap[parent];heap[parent]=heap[at];heap[at]=temp;at=parent;}
    } else if(better(entry,heap[0])){heap[0]=entry;sift(heap,*count,0);}
}
DS41_INDEX_HD uint8_t byte(const RadTensor& t,int64_t at) {return static_cast<const uint8_t*>(t.data)[math::offset(t,at)];}
DS41_INDEX_HD void put_byte(const RadTensor& t,int64_t at,uint8_t value) {static_cast<uint8_t*>(t.data)[math::offset(t,at)]=value;}
DS41_INDEX_HD void put_id(const RadTensor& t,int64_t at,int32_t value) {static_cast<int32_t*>(t.data)[math::offset(t,at)]=value;}
DS41_INDEX_HD void candidate_row(const Plan& p,int64_t row,Entry* heap) {
    const int64_t n=visible(p,row);int64_t count=0;
    for(int64_t key=0;key<p.keys;++key)put_byte(p.t[2],row*p.keys+key,0);
    const int64_t blocks=n/p.block_size+(n%p.block_size!=0);
    for(int64_t b=0;b<blocks;++b) {
        float best=packed::from_bits(0xFF800000u);
        const int64_t begin=b*p.block_size,end=begin+p.block_size<n?begin+p.block_size:n;
        for(int64_t key=begin;key<end;++key){const float v=math::load(p.t[0],row*p.keys+key);if(v>best)best=v;}
        if(b==blocks-1)best=packed::from_bits(0x7F800000u);
        offer(heap,p.topk,&count,{best,int32_t(b)});
    }
    for(int64_t i=0;i<count;++i) {
        const int64_t begin=int64_t(heap[i].id)*p.block_size,end=begin+p.block_size<p.keys?begin+p.block_size:p.keys;
        for(int64_t key=begin;key<end;++key)put_byte(p.t[2],row*p.keys+key,1);
    }
}
DS41_INDEX_HD void topk_row(const Plan& p,int64_t row,Entry* heap) {
    int64_t count=0;
    for(int64_t key=0;key<visible(p,row);++key)
        if(!p.mask||byte(p.t[2],row*p.keys+key))offer(heap,p.topk,&count,{math::load(p.t[0],row*p.keys+key),int32_t(key)});
    // Position order is independent of score order. Insertion-sort at most4096 selected IDs.
    for(int64_t i=1;i<count;++i){const Entry value=heap[i];int64_t j=i;
        while(j&&heap[j-1].id>value.id){heap[j]=heap[j-1];--j;}heap[j]=value;}
    for(int64_t i=0;i<p.topk;++i)put_id(p.t[3],row*p.topk+i,i<count?heap[i].id:-1);
}
} // namespace rad::ds41::index
#undef DS41_INDEX_HD
