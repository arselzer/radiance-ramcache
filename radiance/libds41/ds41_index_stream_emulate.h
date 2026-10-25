#pragma once
#include "ds41_index_stream_impl.h"
#include <array>
namespace rad::ds41::index_stream {
inline int emulate(const RadArgs* a){
    Plan p;const int rc=validate(a,&p);if(rc!=RAD_OK)return rc;*p.paged.status=RAD_OK;
    auto checked=[&](int status){if(status!=RAD_OK)*p.paged.status=status;return status;};
    for(int64_t r=0;r<p.paged.rows;++r)if(checked(check_row(p,r))!=RAD_OK)return *p.paged.status;
    for(int64_t r=0;r<p.paged.rows;++r)for(int64_t i=0;i<items(p,r);++i)if(checked(check_page(p,r,i))!=RAD_OK)return *p.paged.status;
    for(int64_t r=0;r<p.paged.rows;++r){
        for(int64_t i=0;i<p.paged.heads*(p.paged.dim+1);++i)if(checked(check_input(p,r,i))!=RAD_OK)return *p.paged.status;
        for(int64_t i=0;i<items(p,r);++i)if(checked(check_value(p,r,i))!=RAD_OK)return *p.paged.status;
    }
    std::array<Entry,size_t(max_topk)> selected{};std::array<Entry,size_t(max_top_blocks)> blocks{};
    for(int64_t r=0;r<p.paged.rows;++r){Working w{0,0,packed::from_bits(0xFF800000u)};
        for(int64_t i=0;i<items(p,r);++i){const int64_t key=key_at(p,r,i);if(key>=0)merge_key(p,r,key,score(p,r,key),selected.data(),blocks.data(),&w);}
        publish(p,r,selected.data(),blocks.data(),w);}
    return RAD_OK;
}
} // namespace rad::ds41::index_stream
