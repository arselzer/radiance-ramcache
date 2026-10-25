#pragma once
#include "ds41_embedding_args.h"
#include "ds41_embedding_impl.h"
#include "../arch/common/rad_gguf_iq_tables.h"
namespace rad::ds41::embedding {
inline int emulate(const RadArgs* a) {
    Plan p;const int rc=validate(a,&p);if(rc!=RAD_OK)return rc;
    *p.status=RAD_OK;
    for(int64_t row=0;row<p.rows;++row){const int checked=check_id(p,row);if(checked!=RAD_OK){*p.status=checked;return checked;}}
    const packed::Tables tables{gguf::tables::iq2xxs_grid,gguf::tables::iq2xs_grid,
        gguf::tables::iq2s_grid,gguf::tables::iq3xxs_grid,gguf::tables::ksigns_iq2xs};
    for(int64_t row=0;row<p.rows;++row)for(int64_t col=0;col<p.k;++col)
        math::store(p.t[output],row*p.k+col,value(p,row,col,tables));
    return RAD_OK;
}
} // namespace rad::ds41::embedding
