#pragma once
#include "ds41_transform_impl.h"
namespace rad::ds41::transform {
inline int emulate(Op op,const RadArgs* a) {
    Plan p;const int status=validate(op,a,&p);if(status!=RAD_OK)return status;*p.status=RAD_OK;
    for(int64_t i=0;i<preflight_items(p);++i) {const int rc=check_item(p,i);if(rc!=RAD_OK) {*p.status=rc;return rc;}}
    if(op==Op::rmsnorm)for(int64_t row=0;row<p.rows;++row) {
        const float inverse=inverse_rms(p,row);for(int64_t d=0;d<p.dim;++d)norm_element(p,row,d,inverse);
    } else for(int64_t i=0;i<p.rows*p.heads*p.dim;++i)rope_element(p,i);
    return RAD_OK;
}
} // namespace rad::ds41::transform
