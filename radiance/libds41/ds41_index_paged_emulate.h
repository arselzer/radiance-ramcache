#pragma once
#include "ds41_index_paged_impl.h"
namespace rad::ds41::paged_index {
inline int emulate(const RadArgs* args){
    Plan p;const int rc=validate(args,&p);if(rc!=RAD_OK)return rc;*p.status=RAD_OK;
    for(int64_t i=0;i<preflight_items(p);++i){const int status=check_item(p,i);if(status!=RAD_OK){*p.status=status;return status;}}
    for(int64_t i=0;i<p.rows*p.keys;++i)score_element(p,i);
    return RAD_OK;
}
}
