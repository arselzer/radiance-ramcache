#pragma once
#include "ds41_index_stream.h"
#include "ds41_index_impl.h"
#if defined(__HIPCC__)
#define DS41_INDEX_STREAM_HD __host__ __device__ inline
#else
#define DS41_INDEX_STREAM_HD inline
#endif
namespace rad::ds41::index_stream {
DS41_INDEX_STREAM_HD int check_row(const Plan& p,int64_t row){
    const int rc=paged_index::check_row(p.paged,row);if(rc!=RAD_OK)return rc;
    if(p.mode==Mode::consumer){
        const int64_t n=math::id(p.t[visible],row),block_count=(n+p.candidate_block_size-1)/p.candidate_block_size;
        int32_t previous=-1;bool ended=false;
        for(int64_t i=0;i<p.top_blocks;++i){const int32_t id=math::id(p.t[candidates_in],row*p.top_blocks+i);
            if(id== -1){ended=true;continue;}
            if(id<0||id<=previous||ended||id>=block_count)return RAD_E_SHAPE;
            previous=id;}
    }
    return RAD_OK;
}
DS41_INDEX_STREAM_HD int64_t items(const Plan& p,int64_t row){
    if(p.mode!=Mode::consumer)return math::id(p.t[visible],row);
    int64_t count=0;while(count<p.top_blocks&&math::id(p.t[candidates_in],row*p.top_blocks+count)>=0)++count;
    return count*p.candidate_block_size;
}
/* Consumer block tails remain represented, but not-yet-visible keys never map/read a page. */
DS41_INDEX_STREAM_HD int64_t key_at(const Plan& p,int64_t row,int64_t ordinal){
    if(p.mode!=Mode::consumer)return ordinal;
    const int64_t block=math::id(p.t[candidates_in],row*p.top_blocks+ordinal/p.candidate_block_size);
    const int64_t key=block*p.candidate_block_size+ordinal%p.candidate_block_size;
    return key<math::id(p.t[visible],row)?key:-1;
}
DS41_INDEX_STREAM_HD int check_page(const Plan& p,int64_t row,int64_t ordinal){
    const int64_t key=key_at(p,row,ordinal);if(key<0)return RAD_OK;
    int64_t first=0;return paged_index::physical_first(p.paged,row,key,&first);
}
DS41_INDEX_STREAM_HD int check_input(const Plan& p,int64_t row,int64_t i){
    const int64_t q=p.paged.heads*p.paged.dim;
    return math::finite(i<q?math::load(p.t[query],row*q+i):math::load(p.t[head_weights],row*p.paged.heads+i-q))?RAD_OK:RAD_E_INVAL;
}
DS41_INDEX_STREAM_HD float score(const Plan& p,int64_t row,int64_t key){
    const auto& b=p.paged;int64_t first=0;paged_index::physical_first(b,row,key,&first);float total=0;
    for(int64_t h=0;h<b.heads;++h){float dot=0;
        for(int64_t d=0;d<b.dim;++d)dot=dot+math::load(p.t[query],(row*b.heads+h)*b.dim+d)*math::load(p.t[cache],first*(b.dim/b.ratio)+d);
        total=total+(dot>0?dot:0.f)*(math::load(p.t[head_weights],row*b.heads+h)*b.scale);}
    return total;
}
DS41_INDEX_STREAM_HD int check_value(const Plan& p,int64_t row,int64_t ordinal){
    const int64_t key=key_at(p,row,ordinal);if(key<0)return RAD_OK;
    const auto& b=p.paged;int64_t first=0;const int rc=paged_index::physical_first(b,row,key,&first);if(rc!=RAD_OK)return rc;
    for(int64_t d=0;d<b.dim;++d)if(!math::finite(math::load(p.t[cache],first*(b.dim/b.ratio)+d)))return RAD_E_INVAL;
    // Dense score -> candidate/topk rejects nonfinite computed scores. Establish this before
    // any query can publish output, without retaining a dense score plane.
    return math::finite(score(p,row,key))?RAD_OK:RAD_E_INVAL;
}
struct Working {int64_t selected_count,block_count;float block_best;};
DS41_INDEX_STREAM_HD void merge_key(const Plan& p,int64_t row,int64_t key,float value,Entry* selected,Entry* blocks,Working* w){
    index::offer(selected,p.topk,&w->selected_count,{value,int32_t(key)});
    if(p.mode==Mode::source){
        if(value>w->block_best)w->block_best=value;
        const int64_t n=math::id(p.t[visible],row);
        if((key+1)%p.candidate_block_size==0||key+1==n){
            const int64_t block=key/p.candidate_block_size,newest=(n-1)/p.candidate_block_size;
            index::offer(blocks,p.top_blocks,&w->block_count,{block==newest?packed::from_bits(0x7F800000u):w->block_best,int32_t(block)});
            w->block_best=packed::from_bits(0xFF800000u);
        }
    }
}
DS41_INDEX_STREAM_HD void sort_ids(Entry* heap,int64_t count){
    for(int64_t i=1;i<count;++i){const Entry value=heap[i];int64_t j=i;while(j&&heap[j-1].id>value.id){heap[j]=heap[j-1];--j;}heap[j]=value;}
}
DS41_INDEX_STREAM_HD void publish(const Plan& p,int64_t row,Entry* selected,Entry* blocks,const Working& w){
    sort_ids(selected,w.selected_count);
    for(int64_t i=0;i<p.topk;++i)static_cast<int32_t*>(p.t[selected_out].data)[math::offset(p.t[selected_out],row*p.topk+i)]=i<w.selected_count?selected[i].id:-1;
    if(p.mode==Mode::source){sort_ids(blocks,w.block_count);
        for(int64_t i=0;i<p.top_blocks;++i)static_cast<int32_t*>(p.t[candidates_out].data)[math::offset(p.t[candidates_out],row*p.top_blocks+i)]=i<w.block_count?blocks[i].id:-1;}
}
} // namespace rad::ds41::index_stream
#undef DS41_INDEX_STREAM_HD
