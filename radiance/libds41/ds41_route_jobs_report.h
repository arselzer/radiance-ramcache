/* Host metadata handoff. Successful validation here is NOT completion of the
 * device payload check; the core deferred-status gate must still run. */
#pragma once
#include "ds41_route_jobs.h"
#include "rad_runtime.h"
namespace rad::ds41::route_jobs {
struct ContiguousJobs {
    RadRouting routing{};const int32_t* token_ids=nullptr;const int32_t* identity_rows=nullptr;int64_t jobs=0,rows=0;
};
inline int contiguous_view(const RadArgs* args,ContiguousJobs* out){
    if(!out)return RAD_E_INVAL;
    Plan p;const int status=validate(args,&p);if(status!=RAD_OK)return status;
    // RadRouting has no strides; both packed_routed ID vectors also require1.
    for(int i=expert_ids;i<operand_count;++i)if(p.t[i].stride[0]!=1)return RAD_E_STRIDE;
    ContiguousJobs view;view.routing.expert_ids=static_cast<const int32_t*>(p.t[expert_ids].data);
    view.routing.expert_w=static_cast<const float*>(p.t[job_weights].data);
    view.routing.expert_count=static_cast<const int32_t*>(p.t[expert_counts].data);
    view.routing.top_k=p.topk;view.routing.n_expert=p.experts;
    view.token_ids=static_cast<const int32_t*>(p.t[token_ids].data);view.identity_rows=static_cast<const int32_t*>(p.t[identity_rows].data);
    view.jobs=p.jobs;view.rows=p.rows;*out=view;return RAD_OK;
}
} // namespace rad::ds41::route_jobs
