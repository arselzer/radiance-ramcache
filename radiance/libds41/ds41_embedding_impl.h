#pragma once
#include "ds41_embedding.h"
#include "ds41_math_impl.h"
#if defined(__HIPCC__)
#define DS41_EMBEDDING_HD __host__ __device__ inline
#else
#define DS41_EMBEDDING_HD inline
#endif
namespace rad::ds41::embedding {
DS41_EMBEDDING_HD int check_id(const Plan& p,int64_t row) {
    const int32_t id=math::id(p.t[token_ids],row);
    return id>=0&&id<p.n?RAD_OK:RAD_E_SHAPE;
}
DS41_EMBEDDING_HD float value(const Plan& p,int64_t row,int64_t column,const packed::Tables& tables) {
    const int64_t id=math::id(p.t[token_ids],row),values=packed::block_values(p.type);
    const auto* block=static_cast<const uint8_t*>(p.t[weight].data)+id*p.row_bytes+
                      (column/values)*packed::block_bytes(p.type);
    return packed::decode(p.type,block,int(column%values),tables);
}
} // namespace rad::ds41::embedding
#undef DS41_EMBEDDING_HD
