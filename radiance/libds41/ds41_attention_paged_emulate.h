#pragma once
#include "ds41_attention_paged_impl.h"
#include <array>
namespace rad::ds41::attention_paged {
inline int emulate(const RadArgs* a) {
    Plan p;const int rc=validate(a,&p);if(rc!=RAD_OK)return rc;*p.status=RAD_OK;
    auto checked=[&](int status) {if(status!=RAD_OK)*p.status=status;return status;};
    for(int64_t seq=0;seq<p.sequences;++seq)if(checked(check_sequence(p,seq))!=RAD_OK)return *p.status;
    for(int64_t row=0;row<p.rows;++row)if(checked(check_pages(p,row))!=RAD_OK)return *p.status;
    for(int64_t row=0;row<p.rows;++row)if(checked(check_payload(p,row))!=RAD_OK)return *p.status;
    std::array<float,size_t(max_window+max_topk)> probabilities{};
    for(int64_t row=0;row<p.rows;++row)for(int64_t h=0;h<p.heads;++h) {
        const int64_t count=candidate_count(p,row);for(int64_t k=0;k<count;++k)probabilities[size_t(k)]=score(p,row,h,k);
        const float mx=maximum(p,h,probabilities.data(),count);
        for(int64_t k=0;k<count;++k)probabilities[size_t(k)]=mx==packed::from_bits(0xFF800000u)?0.f : ::expf(probabilities[size_t(k)]-mx);
        const float den=denominator(p,h,probabilities.data(),count,mx);
        for(int64_t d=0;d<p.dim;++d)attend_column(p,row,h,d,probabilities.data(),count,den);
    }
    return RAD_OK;
}
} // namespace rad::ds41::attention_paged
