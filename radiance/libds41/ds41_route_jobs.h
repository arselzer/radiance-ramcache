/* DS41 router -> token-major packed-expert jobs. Source/outputs carry logical
 * elements and explicit positive pitches; allocation capacities remain caller-owned. */
#pragma once
#include "ds41_device_plan.h"
#include "ds41_math.h"
namespace rad::ds41::route_jobs {
inline constexpr int64_t scratch_bytes=4,status_offset=0,max_experts=4096;
inline constexpr int operand_count=7;
enum Operand {ids=0,weights=1,expert_ids=2,token_ids=3,job_weights=4,expert_counts=5,identity_rows=6};
struct Plan {
    RadTensor t[operand_count]{};int64_t spans[operand_count]{};
    int64_t rows=0,topk=0,experts=0,jobs=0;int32_t* status=nullptr;
};
inline int validate(const RadArgs* args,Plan* out){
    if(!args||!out||!args->t||args->n_t!=operand_count||!args->p||args->n_p<=0)return RAD_E_INVAL;
    const int64_t capacity=rad_param_geti(args->p,args->n_p,"M",0),topk=rad_param_geti(args->p,args->n_p,"topk",0),
                  experts=rad_param_geti(args->p,args->n_p,"n_expert",0);
    if(capacity<=0||capacity>INT_MAX||experts<=0||experts>max_experts||topk<=0||topk>experts||capacity>INT_MAX/topk)return RAD_E_SHAPE;
    if(args->t[ids].rank!=2)return RAD_E_SHAPE;
    Plan p;p.rows=args->t[ids].shape[0];p.topk=topk;p.experts=experts;
    if(p.rows<=0||p.rows>capacity)return RAD_E_SHAPE;
    p.jobs=p.rows*p.topk;for(int i=0;i<operand_count;++i)p.t[i]=args->t[i];
    int status=math::tensor(p.t[ids],RAD_I32,{p.rows,p.topk},&p.spans[ids]);
    if(status==RAD_OK)status=math::tensor(p.t[weights],RAD_F32,{p.rows,p.topk},&p.spans[weights]);
    for(int i=expert_ids;i<operand_count&&status==RAD_OK;++i)
        status=math::tensor(p.t[i],i==job_weights?RAD_F32:RAD_I32,{i==expert_counts?p.experts:p.jobs},&p.spans[i]);
    if(status!=RAD_OK)return status;
    // Every output (including the histogram and identity rows) is transactional:
    // reject overlap with inputs or another output before any flag/payload write.
    for(int i=expert_ids;i<operand_count;++i)for(int j=0;j<operand_count;++j)
        if(i!=j&&math::overlaps(p.t[i].data,p.spans[i],p.t[j].data,p.spans[j]))return RAD_E_INVAL;
    if(!args->scratch||args->scratch_bytes<scratch_bytes)return RAD_E_INVAL;
    if(uintptr_t(args->scratch)%alignof(int32_t))return RAD_E_ALIGN;
    if(uintptr_t(args->scratch)>UINTPTR_MAX-uint64_t(scratch_bytes))return RAD_E_SHAPE;
    for(int i=0;i<operand_count;++i)if(math::overlaps(args->scratch,scratch_bytes,p.t[i].data,p.spans[i]))return RAD_E_INVAL;
    p.status=static_cast<int32_t*>(args->scratch);*out=p;return RAD_OK;
}
} // namespace rad::ds41::route_jobs
extern "C" {
int ds41_route_jobs(const RadArgs*,RadStream);
inline int64_t ds41_route_jobs_scratch(const RadArgs*){return rad::ds41::gpu_scratch_bytes<rad::ds41::route_jobs::Plan>();}
}
