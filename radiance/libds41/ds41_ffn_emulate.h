#pragma once
#include "ds41_ffn_impl.h"
#include <vector>
namespace rad::ds41::ffn {
inline int emulate(Op op,const RadArgs* a) {
    Plan p;const int rc=validate(op,a,&p);if(rc!=RAD_OK)return rc;*p.status=RAD_OK;
    for(int64_t i=0;i<preflight_items(p);++i) {
        const int error=check_item(p,i);if(error!=RAD_OK) {*p.status=error;return error;}
    }
    if(op==Op::expert_act)for(int64_t i=0;i<p.rows*p.n;++i)activate(p,i);
    else if(op==Op::mhc_post)for(int64_t i=0;i<p.rows*p.hc*p.n;++i)post(p,i);
    else {
        std::vector<int32_t> order(size_t(p.experts));
        for(int64_t row=0;row<p.rows;++row) {const int count=order_routes(p,row,order.data());
            if(count<0) {*p.status=RAD_E_STATE;return RAD_E_STATE;}
            for(int64_t d=0;d<p.n;++d)combine(p,row,d,order.data(),count);
        }
    }
    return RAD_OK;
}
} // namespace rad::ds41::ffn
