#pragma once
#include "ds41_reorder_heads.h"
#if defined(__HIPCC__)
#define DS41_REORDER_HD __host__ __device__ inline
#else
#define DS41_REORDER_HD inline
#endif
namespace rad::ds41::reorder_heads {
DS41_REORDER_HD void copy_element(const Launch& p,int64_t output_index) {
    const int64_t d=output_index%p.dim,head_row=output_index/p.dim,
                  head=head_row%(p.tp*p.heads),row=head_row/(p.tp*p.heads),
                  rank=head/p.heads,local=head%p.heads,
                  input_head=(rank*p.rows+row)*p.heads+local;
    const int64_t input_index=input_head*p.dim+d;
    auto* destination=static_cast<uint8_t*>(p.y)+output_index*p.element_bytes;
    const auto* source=static_cast<const uint8_t*>(p.q)+input_index*p.element_bytes;
    if(p.element_bytes==4)__builtin_memcpy(destination,source,4);
    else __builtin_memcpy(destination,source,2);
    if(!d)__builtin_memcpy(static_cast<uint8_t*>(p.v)+head_row*4,static_cast<const uint8_t*>(p.w)+input_head*4,4);
}
inline int emulate(const RadArgs* a) {
    Launch p;const int rc=validate(a,&p);if(rc!=RAD_OK)return rc;
    for(int64_t i=0;i<p.query_elements;++i)copy_element(p,i);
    return RAD_OK;
}
} // namespace rad::ds41::reorder_heads
#undef DS41_REORDER_HD
