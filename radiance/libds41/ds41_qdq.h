/* Explicit BF16 out-of-place mathematical QDQ candidate. No packed cache ABI,
 * default model behavior, official TileLang parity or model-quality claim. */
#pragma once
#include "ds41_device_plan.h"
#include "ds41_math_impl.h"
#include "ds41_qdq_math.h"
namespace rad::ds41::qdq {
enum Operand {input=0,emit,output,operand_count};
inline constexpr int64_t max_rows=65536,max_dimension=1024,scratch_bytes=4;
struct Plan {
    RadTensor t[operand_count]{};int64_t spans[operand_count]{};
    int64_t rows=0,dim=0,groups=0;qdq_math::Profile profile{};bool masked=false;int32_t* status=nullptr;
};
inline int parse_profile(const char* text,qdq_math::Profile* out){
    if(!text||!out)return RAD_E_INVAL;
    if(!std::strcmp(text,"window_fp8"))*out=qdq_math::Profile::window_fp8;
    else if(!std::strcmp(text,"compressed_fp4"))*out=qdq_math::Profile::compressed_fp4;
    else if(!std::strcmp(text,"index_fp4"))*out=qdq_math::Profile::index_fp4;
    else return RAD_E_INVAL;
    return RAD_OK;
}
inline int validate(const RadArgs* a,Plan* out){
    if(!a||!out||!a->p||a->n_p<=0||!a->t||a->n_t!=operand_count)return RAD_E_INVAL;
    Plan p;for(int i=0;i<operand_count;++i)p.t[i]=a->t[i];
    const int64_t capacity=rad_param_geti(a->p,a->n_p,"M",0),masked=rad_param_geti(a->p,a->n_p,"masked",-1);
    p.dim=rad_param_geti(a->p,a->n_p,"n",0);
    const int profile=parse_profile(rad_param_gets(a->p,a->n_p,"profile",nullptr),&p.profile);if(profile!=RAD_OK)return profile;
    if(masked<0||masked>1)return RAD_E_INVAL;
    p.masked=masked!=0;
    if(capacity<=0||capacity>max_rows||p.dim<=0||p.dim>max_dimension||p.dim%qdq_math::width(p.profile)||p.t[input].rank!=2)return RAD_E_SHAPE;
    p.rows=p.t[input].shape[0];p.groups=p.dim/qdq_math::width(p.profile);
    if(p.rows<=0||p.rows>capacity)return RAD_E_SHAPE;
    int rc=math::tensor(p.t[input],RAD_BF16,{p.rows,p.dim},&p.spans[input]);if(rc!=RAD_OK)return rc;
    rc=math::tensor(p.t[output],RAD_BF16,{p.rows,p.dim},&p.spans[output]);if(rc!=RAD_OK)return rc;
    if(p.masked){rc=math::tensor(p.t[emit],RAD_I32,{p.rows},&p.spans[emit]);if(rc!=RAD_OK)return rc;}
    else if(p.t[emit].rank||p.t[emit].data)return RAD_E_INVAL;
    for(int i=0;i<output;++i)if(math::overlaps(p.t[output].data,p.spans[output],p.t[i].data,p.spans[i]))return RAD_E_INVAL;
    if(!a->scratch||a->scratch_bytes<scratch_bytes)return RAD_E_INVAL;
    if(uintptr_t(a->scratch)%alignof(int32_t))return RAD_E_ALIGN;
    if(uintptr_t(a->scratch)>UINTPTR_MAX-uint64_t(scratch_bytes))return RAD_E_SHAPE;
    for(int i=0;i<operand_count;++i)if(math::overlaps(a->scratch,scratch_bytes,p.t[i].data,p.spans[i]))return RAD_E_INVAL;
    p.status=static_cast<int32_t*>(a->scratch);*out=p;return RAD_OK;
}
inline const RadParamSpec params[]{{"M",RAD_P_INT,RAD_REQUIRED,RAD_PROLE_CAPACITY},
    {"n",RAD_P_INT,RAD_REQUIRED},{"profile",RAD_P_STR,RAD_REQUIRED},{"masked",RAD_P_INT,RAD_REQUIRED}};
inline const RadOperandSpec operands[]{{"input",RAD_OPD_IN,0},{"emit",RAD_OPD_IN,1},{"output",RAD_OPD_OUT,0}};
inline const RadOpSchema schema{"ds41_qdq_candidate",params,4,operands,operand_count,
    "Explicit BF16 QDQ mathematical candidate: window FP8/E8M0, compressed FP4/E4M3, index FP4/E8M0; inactive +0, full-batch preflight. No TileLang parity claim."};
} // namespace rad::ds41::qdq
extern "C" {
int ds41_qdq_candidate(const RadArgs*,RadStream);
int ds41_ref_qdq_candidate(const RadArgs*,RadStream) noexcept;
inline int64_t ds41_qdq_scratch(const RadArgs*){return rad::ds41::gpu_scratch_bytes<rad::ds41::qdq::Plan>();}
}
