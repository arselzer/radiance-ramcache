#pragma once
#include "ds41_compress_finish_impl.h"
namespace rad::ds41::compress_finish {
inline int emulate(const RadArgs* a){
    Plan p;const int rc=validate(a,&p);if(rc!=RAD_OK)return rc;*p.status=RAD_OK;
    for(int64_t item=0;item<p.dim+p.rows;++item){const int checked=check_item(p,item);if(checked!=RAD_OK){*p.status=checked;return checked;}}
    for(int64_t row=0;row<p.rows;++row){const bool active=math::id(p.t[emit],row)!=0;const float inverse=active?inverse_rms(p,row):0.f;for(int64_t d=0;d<p.dim;++d)output_element(p,row,d,inverse,active);}
    return RAD_OK;
}
} // namespace rad::ds41::compress_finish
