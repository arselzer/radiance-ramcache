#pragma once
#include "ds41_compress_impl.h"
namespace rad::ds41::compress {
inline int emulate(const RadArgs* a) {
    Plan p;const int rc=validate(a,&p);if(rc!=RAD_OK)return rc;*p.status=RAD_OK;
    for(int64_t i=0;i<preflight_items(p);++i) {
        const int error=check_item(p,i);if(error!=RAD_OK) {*p.status=error;return error;}
    }
    for(int64_t s=0;s<p.sequences;++s) {
        const int64_t begin=math::id(p.t[cu],s),end=math::id(p.t[cu],s+1);if(begin==end)continue;
        const int64_t slot=math::id(p.t[state_ids],s),first=math::id(p.t[positions],begin);
        for(int64_t d=0;d<p.dimension;++d)column(p,begin,end,slot,first,d);
        finish_sequence(p,begin,end,slot,first);
    }
    return RAD_OK;
}
} // namespace rad::ds41::compress
