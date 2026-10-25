#pragma once
#include "ds41_ffn.h"
#if defined(__HIPCC__)
#define DS41_FFN_HD __host__ __device__ inline
#else
#define DS41_FFN_HD inline
#endif
namespace rad::ds41::ffn {
DS41_FFN_HD int64_t preflight_items(const Plan& p) {
    if(p.op==Op::expert_act)return 2*p.rows*p.n+p.rows;
    if(p.op==Op::moe_combine)return (p.routes+p.rows)*p.n+p.routes;
    return p.rows*p.n+p.rows*p.hc*p.n+p.rows*p.hc+p.rows*p.hc*p.hc;
}
DS41_FFN_HD int check_item(const Plan& p,int64_t item) {
    if(p.op==Op::moe_combine) {
        const int64_t route_values=p.routes*p.n,shared_values=p.rows*p.n;
        if(item<route_values)return math::finite(math::load(p.t[0],item))?RAD_OK:RAD_E_INVAL;
        item-=route_values;
        if(item<shared_values)return math::finite(math::load(p.t[3],item))?RAD_OK:RAD_E_INVAL;
        item-=shared_values;
        const int32_t e=math::id(p.t[1],item),row=math::id(p.t[2],item);
        if(e<0||e>=p.experts||row<0||row>=p.rows)return RAD_E_SHAPE;
        for(int64_t prior=0;prior<item;++prior)
            if(math::id(p.t[1],prior)==e&&math::id(p.t[2],prior)==row)return RAD_E_INVAL;
        return RAD_OK;
    }
    for(int i=0;i<p.nt-1;++i) {
        const int64_t n=math::count(p.t[i]);
        if(item<n) {
            const float v=math::load(p.t[i],item);
            return math::finite(v)&&!(p.op==Op::expert_act&&i==2&&v<0)?RAD_OK:RAD_E_INVAL;
        }
        item-=n;
    }
    return RAD_E_INVAL;
}
DS41_FFN_HD void activate(const Plan& p,int64_t i) {
    float gate=math::load(p.t[0],i),up=math::load(p.t[1],i);
    if(p.limit>0) {
        if(gate>p.limit)gate=p.limit;
        if(up>p.limit)up=p.limit;else if(up< -p.limit)up=-p.limit;
    }
    const float activated=(gate*math::sigmoid(gate))*up;
    const float weighted=activated*math::load(p.t[2],i/p.n);
    math::store(p.t[3],i,weighted);
}
DS41_FFN_HD int order_routes(const Plan& p,int64_t row,int32_t* order) {
    int count=0;
    for(int64_t j=0;j<p.routes;++j)if(math::id(p.t[2],j)==row) {
        if(count>=p.experts)return -1;
        int at=count;
        while(at>0&&math::id(p.t[1],order[at-1])>math::id(p.t[1],j)) {order[at]=order[at-1];--at;}
        order[at]=int32_t(j);++count;
    }
    return count;
}
DS41_FFN_HD void combine(const Plan& p,int64_t row,int64_t d,const int32_t* order,int count) {
    float sum=0;
    for(int i=0;i<count;++i)sum=sum+math::load(p.t[0],int64_t(order[i])*p.n+d);
    // Inputs are completed, individually rounded expert outputs. No route weights belong here.
    math::store(p.t[4],row*p.n+d,sum+math::load(p.t[3],row*p.n+d));
}
DS41_FFN_HD void post(const Plan& p,int64_t i) {
    const int64_t row=i/(p.hc*p.n),dst=(i/p.n)%p.hc,d=i%p.n;float sum=0;
    for(int64_t src=0;src<p.hc;++src)
        sum=sum+math::load(p.t[3],row*p.hc*p.hc+src*p.hc+dst)*math::load(p.t[1],row*p.hc*p.n+src*p.n+d);
    sum=sum+math::load(p.t[2],row*p.hc+dst)*math::load(p.t[0],row*p.n+d);
    math::store(p.t[4],i,sum);
}
} // namespace rad::ds41::ffn
#undef DS41_FFN_HD
