#pragma once
#include "ds41_index_impl.h"
#include <vector>
namespace rad::ds41::index {
inline int emulate(Op op,const RadArgs* args) {
    Plan p;const int rc=validate(op,args,&p);if(rc!=RAD_OK)return rc;
    *p.status=RAD_OK;
    for(int64_t i=0;i<preflight_items(p);++i){const int s=check_item(p,i);if(s!=RAD_OK){*p.status=s;return s;}}
    if(op==Op::scores)for(int64_t i=0;i<p.rows*p.keys;++i)score_element(p,i);
    else {
        std::vector<Entry> heap(size_t(p.topk));
        for(int64_t row=0;row<p.rows;++row)if(op==Op::candidates)candidate_row(p,row,heap.data());else topk_row(p,row,heap.data());
    }
    return RAD_OK;
}
}
