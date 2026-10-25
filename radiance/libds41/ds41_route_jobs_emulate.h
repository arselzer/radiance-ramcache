/* CPU-only transactional emulator. No serving row or host registration. */
#pragma once
#include "ds41_route_jobs_impl.h"
namespace rad::ds41::route_jobs {
inline int emulate(const RadArgs* args){
    Plan p;const int status=validate(args,&p);if(status!=RAD_OK)return status;
    *p.status=RAD_OK;
    for(int64_t job=0;job<p.jobs;++job){const int result=check_job(p,job);if(result!=RAD_OK){*p.status=result;return result;}}
    for(int64_t expert=0;expert<p.experts;++expert)clear_expert(p,expert);
    for(int64_t job=0;job<p.jobs;++job){const int32_t expert=write_job(p,job);++static_cast<int32_t*>(p.t[expert_counts].data)[math::offset(p.t[expert_counts],expert)];}
    return RAD_OK;
}
}
