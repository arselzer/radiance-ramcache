#pragma once
#include "ds41_latent_emit_args.h"
#include "ds41_latent_emit_impl.h"
namespace rad::ds41::latent_emit {
inline int emulate(const RadArgs* a) {
    Plan p;const int rc=validate(a,&p);if(rc!=RAD_OK)return rc;*p.flag=RAD_OK;
    for(int phase=0;phase<3&&*p.flag==RAD_OK;++phase)for(int64_t row=0;row<p.latent.rows;++row) {
        const int checked=phase==0?check_mask(p,row):phase==1?check_coordinate(p,row):check_duplicate(p,row);
        if(checked!=RAD_OK){*p.flag=checked;break;}
    }
    for(int64_t row=0;row<p.latent.rows;++row)seal_row(p,row);
    if(*p.flag!=RAD_OK)return *p.flag;
    for(int64_t row=0;row<p.latent.rows;++row)if(integer(p.t[emit],row))
        for(int64_t col=0;col<p.latent.layout.dimension;++col)copy_element(p,row,col);
    return RAD_OK;
}
} // namespace rad::ds41::latent_emit
