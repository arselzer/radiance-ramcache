#pragma once
#include "ds41_latent_emit.h"
#include "ds41_math_impl.h"
#if !defined(__HIPCC__)
#include "ds41_latent_args.h"
#endif
#if defined(__HIPCC__)
#define DS41_LATENT_EMIT_HD __host__ __device__ inline
#else
#define DS41_LATENT_EMIT_HD inline
#endif
namespace rad::ds41::latent_emit {
DS41_LATENT_EMIT_HD int64_t integer(const RadTensor& t,int64_t row) {
    const auto at=row*t.stride[0];
    return t.dtype==RAD_I32?int64_t(static_cast<const int32_t*>(t.data)[at]):static_cast<const int64_t*>(t.data)[at];
}
DS41_LATENT_EMIT_HD int checked_first(const Plan& p,int64_t position,int64_t final_slot,int64_t* first) {
#if defined(__HIPCC__)
    // Exact existing latent::store_row contract, kept device-local because the original
    // host-only metadata header also contains host canonical-table/re-layout code.
    const auto& l=p.latent;const int64_t ratio=l.layout.ratio,b=l.layout.block_size;
    if(position<0||position%ratio!=ratio-1||final_slot<ratio-1||final_slot>=l.slots||final_slot%b!=position%b)return RAD_E_SHAPE;
    const int64_t start=final_slot-ratio+1;if(start/b!=final_slot/b)return RAD_E_SHAPE;
    *first=start;return RAD_OK;
#else
    return latent::store_row(p.latent,position,final_slot,first);
#endif
}
DS41_LATENT_EMIT_HD int check_mask(const Plan& p,int64_t row) {
    const int64_t value=integer(p.t[emit],row);return value==0||value==1?RAD_OK:RAD_E_INVAL;
}
DS41_LATENT_EMIT_HD int check_coordinate(const Plan& p,int64_t row) {
    if(!integer(p.t[emit],row))return RAD_OK;
    int64_t first=0;return checked_first(p,integer(p.t[positions],row),integer(p.t[slot_mapping],row),&first);
}
DS41_LATENT_EMIT_HD int check_duplicate(const Plan& p,int64_t row) {
    if(!integer(p.t[emit],row))return RAD_OK;
    const int64_t slot=integer(p.t[slot_mapping],row);
    for(int64_t other=0;other<row;++other)
        if(integer(p.t[emit],other)&&integer(p.t[slot_mapping],other)==slot)return RAD_E_INVAL;
    return RAD_OK;
}
DS41_LATENT_EMIT_HD void seal_row(const Plan& p,int64_t row) {
    static_cast<int32_t*>(p.t[status].data)[row*p.t[status].stride[0]]=*p.flag;
}
DS41_LATENT_EMIT_HD void copy_element(const Plan& p,int64_t row,int64_t col) {
    const int64_t first=integer(p.t[slot_mapping],row)-p.latent.layout.ratio+1,
                  source=math::offset(p.t[x],row*p.latent.layout.dimension+col),
                  target=first*(p.latent.layout.dimension/p.latent.layout.ratio)+col;
    const int width=int(p.latent.layout.element_bytes);
    const auto* src=static_cast<const uint8_t*>(p.t[x].data)+source*width;
    auto* dst=static_cast<uint8_t*>(p.t[cache].data)+target*width;
    if(width==4)__builtin_memcpy(dst,src,4);else __builtin_memcpy(dst,src,2);
}
} // namespace rad::ds41::latent_emit
#undef DS41_LATENT_EMIT_HD
