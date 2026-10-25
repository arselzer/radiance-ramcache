/* Explicit DS41 norm/rotary storage boundaries. No projection, cache quantization or model
 * lifecycle is hidden here; inputs are already at the graph's intended stored precision. */
#pragma once
#include "ds41_device_plan.h"
#include "ds41_math_impl.h"
namespace rad::ds41::transform {
enum class Op {rmsnorm,rope_tail};
inline constexpr int64_t max_rows=65536,max_norm_dimension=65536,max_heads=1024,max_head_dim=1024,max_positions=1048576;
inline constexpr int64_t scratch_bytes=4,status_offset=0;
struct Plan {
    Op op{};int nt=0;RadTensor t[4]{};int64_t spans[4]{};
    int64_t rows=0,heads=0,dim=0,rotary=0,positions=0;float eps=0;bool inverse=false;int32_t* status=nullptr;
};
inline int validate(Op op,const RadArgs* a,Plan* out) {
    if(unsigned(op)>1||!a||!out||!a->p||a->n_p<=0||!a->t)return RAD_E_INVAL;
    const int nt=op==Op::rmsnorm?3:4;if(a->n_t!=nt)return RAD_E_INVAL;
    Plan p;p.op=op;p.nt=nt;for(int i=0;i<nt;++i)p.t[i]=a->t[i];
    auto gi=[&](const char* key){return int64_t(rad_param_geti(a->p,a->n_p,key,0));};
    const int64_t capacity=gi("M");if(capacity<=0||capacity>max_rows||p.t[0].rank!=(op==Op::rmsnorm?2u:3u))return RAD_E_SHAPE;
    p.rows=p.t[0].shape[0];if(p.rows<=0||p.rows>capacity)return RAD_E_SHAPE;
    int rc=RAD_OK;auto check=[&](int i,uint32_t d,std::initializer_list<int64_t> shape) {
        if(rc==RAD_OK)rc=math::tensor(p.t[i],d,shape,&p.spans[i]);
    };
    if(op==Op::rmsnorm) {
        p.dim=gi("n");p.eps=float(rad_param_getf(a->p,a->n_p,"eps",0));
        const auto input=rad_dtype_parse(rad_param_gets(a->p,a->n_p,"a_dtype",nullptr));
        const auto output=rad_dtype_parse(rad_param_gets(a->p,a->n_p,"y_dtype",nullptr));
        if(!math::scalar_dtype(input)||!math::scalar_dtype(output))return RAD_E_DTYPE;
        if(p.dim<=0||p.dim>max_norm_dimension)return RAD_E_SHAPE;
        if(!(p.eps>0)||!std::isfinite(p.eps))return RAD_E_INVAL;
        check(0,input,{p.rows,p.dim});check(1,RAD_F32,{p.dim});check(2,output,{p.rows,p.dim});
    } else {
        p.heads=gi("heads");p.dim=gi("head_dim");p.rotary=gi("rotary_dim");const int64_t inverse=rad_param_geti(a->p,a->n_p,"inverse",-1);
        const auto dtype=rad_dtype_parse(rad_param_gets(a->p,a->n_p,"dtype",nullptr));if(!math::scalar_dtype(dtype))return RAD_E_DTYPE;
        if(p.heads<=0||p.heads>max_heads||p.dim<=0||p.dim>max_head_dim||p.rotary<=0||p.rotary>p.dim||(p.rotary&1)||
           inverse<0||inverse>1||p.t[2].rank!=3)return RAD_E_SHAPE;
        p.inverse=inverse!=0;p.positions=p.t[2].shape[0];if(p.positions<=0||p.positions>max_positions)return RAD_E_SHAPE;
        check(0,dtype,{p.rows,p.heads,p.dim});check(1,RAD_I32,{p.rows});
        check(2,RAD_F32,{p.positions,p.rotary/2,2});check(3,dtype,{p.rows,p.heads,p.dim});
    }
    if(rc!=RAD_OK)return rc;
    const int yi=nt-1;
    for(int i=0;i<yi;++i)if(math::overlaps(p.t[yi].data,p.spans[yi],p.t[i].data,p.spans[i]))return RAD_E_INVAL;
    if(!a->scratch||a->scratch_bytes<scratch_bytes)return RAD_E_INVAL;
    if(uintptr_t(a->scratch)%alignof(int32_t))return RAD_E_ALIGN;
    if(uintptr_t(a->scratch)>UINTPTR_MAX-uint64_t(scratch_bytes))return RAD_E_SHAPE;
    for(int i=0;i<nt;++i)if(math::overlaps(a->scratch,scratch_bytes,p.t[i].data,p.spans[i]))return RAD_E_INVAL;
    p.status=static_cast<int32_t*>(a->scratch);*out=p;return RAD_OK;
}
} // namespace rad::ds41::transform
extern "C" {
int ds41_rmsnorm(const RadArgs*,RadStream);
int ds41_rope_tail(const RadArgs*,RadStream);
inline int64_t ds41_transform_scratch(const RadArgs*) {return rad::ds41::gpu_scratch_bytes<rad::ds41::transform::Plan>();}
}
