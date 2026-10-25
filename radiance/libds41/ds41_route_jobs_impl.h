#pragma once
#include "ds41_route_jobs.h"
#include "ds41_math_impl.h"
#if defined(__HIPCC__)
#define DS41_ROUTE_JOBS_HD __host__ __device__ inline
#else
#define DS41_ROUTE_JOBS_HD inline
#endif
namespace rad::ds41::route_jobs {
DS41_ROUTE_JOBS_HD int check_job(const Plan& p,int64_t job){
    const int32_t expert=math::id(p.t[ids],job);const float weight=math::load(p.t[weights],job);
    if(expert<0||expert>=p.experts)return RAD_E_SHAPE;
    if(!math::finite(weight)||weight<0)return RAD_E_INVAL;
    const int64_t row=job/p.topk,choice=job%p.topk;
    for(int64_t prior=0;prior<choice;++prior)if(math::id(p.t[ids],row*p.topk+prior)==expert)return RAD_E_INVAL;
    return RAD_OK;
}
DS41_ROUTE_JOBS_HD void clear_expert(const Plan& p,int64_t expert){
    static_cast<int32_t*>(p.t[expert_counts].data)[math::offset(p.t[expert_counts],expert)]=0;
}
DS41_ROUTE_JOBS_HD int32_t write_job(const Plan& p,int64_t job){
    const int32_t expert=math::id(p.t[ids],job);
    static_cast<int32_t*>(p.t[expert_ids].data)[math::offset(p.t[expert_ids],job)]=expert;
    static_cast<int32_t*>(p.t[token_ids].data)[math::offset(p.t[token_ids],job)]=int32_t(job/p.topk);
    static_cast<int32_t*>(p.t[identity_rows].data)[math::offset(p.t[identity_rows],job)]=int32_t(job);
    math::store(p.t[job_weights],job,math::load(p.t[weights],job));
    return expert;
}
} // namespace rad::ds41::route_jobs
#undef DS41_ROUTE_JOBS_HD
