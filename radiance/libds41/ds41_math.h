/* Experimental correctness baselines for the existing libref DS41 schemas. Metadata errors
 * return synchronously. Device payload errors live in scratch[0:4] and leave all outputs
 * unchanged; successful enqueue alone does not establish successful device validation.
 * No runtime status propagation or production architecture/lifecycle is implemented here. */
#pragma once
#include "ds41_device_plan.h"
#include "rad_plugin.h"
#include "ds41_packed_decode.h"
#include <cmath>
#include <climits>
#include <initializer_list>

namespace rad::ds41::math {
enum class Op {hc_prepare,hc_collapse,router_topk,sparse_attn};
inline constexpr int64_t status_offset=0,status_bytes=4,scratch_bytes=4;
inline constexpr int max_router_experts=4096,max_sparse_topk=4096,max_sparse_dim=1024;
struct Plan {
    Op op{};int nt=0;RadTensor t[7]{};int64_t spans[7]{};
    int64_t rows=0,n=0,hc=0,iterations=0,experts=0,topk=0,heads=0,dim=0,kv_rows=0;
    float eps=0,hc_eps=0,temperature=0,route_scale=0,scale=0;bool normalize=false;
    int32_t* status=nullptr;
};
inline bool scalar_dtype(uint32_t d) {return d==RAD_F32||d==RAD_F16||d==RAD_BF16;}
inline bool multiply(int64_t a,int64_t b,int64_t* out) {
    if(a<0||b<0||(b&&a>INT64_MAX/b))return false;
    *out=a*b;return true;
}
inline bool overlaps(const void* a,int64_t na,const void* b,int64_t nb) {
    const auto x=uintptr_t(a),y=uintptr_t(b);
    return na>0&&nb>0&&(x<=y?uint64_t(y-x)<uint64_t(na):uint64_t(x-y)<uint64_t(nb));
}
inline int tensor(const RadTensor& t,uint32_t dtype,std::initializer_list<int64_t> shape,int64_t* span) {
    if(!t.data)return RAD_E_INVAL;
    if(t.dtype!=dtype)return RAD_E_DTYPE;
    if(t.rank!=shape.size())return RAD_E_SHAPE;
    unsigned index=0;for(auto d:shape)if(d<=0||t.shape[index++]!=d)return RAD_E_SHAPE;
    const int bytes=rad_dtype_bits(dtype)/8;
    if(bytes<=0||uintptr_t(t.data)%unsigned(bytes))return RAD_E_ALIGN;
    int64_t elements=1,minimum=1;
    for(int axis=int(t.rank)-1;axis>=0;--axis) {
        if(t.stride[axis]<minimum)return RAD_E_STRIDE;
        int64_t extent=0;
        if(!multiply(t.shape[axis]-1,t.stride[axis],&extent)||elements>INT64_MAX-extent)return RAD_E_SHAPE;
        elements+=extent;
        if(!multiply(t.shape[axis],t.stride[axis],&minimum))return RAD_E_SHAPE;
    }
    if(!multiply(elements,bytes,span)||uintptr_t(t.data)>UINTPTR_MAX-uint64_t(*span))return RAD_E_SHAPE;
    return RAD_OK;
}
inline int validate(Op op,const RadArgs* a,Plan* out) {
    if(static_cast<unsigned>(op)>3)return RAD_E_INVAL;
    const int nt=op==Op::hc_prepare?7:op==Op::hc_collapse?3:op==Op::router_topk?4:5;
    if(!a||!out||!a->t||a->n_t!=nt||!a->p||a->n_p<=0)return RAD_E_INVAL;
    Plan p;p.op=op;p.nt=nt;for(int i=0;i<nt;++i)p.t[i]=a->t[i];
    auto gi=[&](const char* key,int64_t fallback=0){return int64_t(rad_param_geti(a->p,a->n_p,key,fallback));};
    auto gf=[&](const char* key){return float(rad_param_getf(a->p,a->n_p,key,0));};
    const int64_t capacity=gi("M");
    if(p.t[0].rank<1||capacity<=0||capacity>INT_MAX)return RAD_E_SHAPE;
    p.rows=p.t[0].shape[0];if(p.rows<=0||p.rows>capacity)return RAD_E_SHAPE;
    const uint32_t dtype=rad_dtype_parse(rad_param_gets(a->p,a->n_p,"dtype","f32"));
    if(op!=Op::router_topk&&!scalar_dtype(dtype))return RAD_E_DTYPE;
    int rc=RAD_OK;
    auto check=[&](int i,uint32_t d,std::initializer_list<int64_t> shape) {
        if(rc==RAD_OK)rc=tensor(p.t[i],d,shape,&p.spans[i]);
    };
    if(op==Op::hc_prepare||op==Op::hc_collapse) {
        p.n=gi("n");p.hc=gi("hc");
        if(p.n<=0||p.hc<2||p.hc>8||p.n>INT_MAX/p.hc)return RAD_E_SHAPE;
        const int64_t width=p.n*p.hc,mixes=p.hc*(p.hc+2);
        check(0,dtype,{p.rows,width});
        if(op==Op::hc_prepare) {
            p.iterations=gi("iterations");p.eps=gf("eps");p.hc_eps=gf("hc_eps");
            if(p.iterations<1||p.iterations>256||!(p.eps>0)||!std::isfinite(p.eps)||
               p.hc_eps<0||!std::isfinite(p.hc_eps))return RAD_E_INVAL;
            check(1,RAD_F32,{mixes,width});check(2,RAD_F32,{3});check(3,RAD_F32,{mixes});
            check(4,RAD_F32,{p.rows,p.hc});check(5,RAD_F32,{p.rows,p.hc});check(6,RAD_F32,{p.rows,p.hc*p.hc});
        } else {
            if(p.rows>INT_MAX/p.n)return RAD_E_SHAPE;
            check(1,RAD_F32,{p.rows,p.hc});check(2,dtype,{p.rows,p.n});
        }
    } else if(op==Op::router_topk) {
        p.experts=gi("experts");p.topk=gi("topk");const auto norm=gi("normalize",1);
        p.temperature=gf("temperature");p.route_scale=gf("route_scale");p.normalize=norm!=0;
        if(p.experts<=0||p.experts>max_router_experts||p.topk<=0||p.topk>p.experts||norm<0||norm>1)return RAD_E_SHAPE;
        if(!(p.temperature>0)||!(p.route_scale>0)||!std::isfinite(p.temperature)||!std::isfinite(p.route_scale))return RAD_E_INVAL;
        check(0,RAD_F32,{p.rows,p.experts});check(1,RAD_F32,{p.experts});
        check(2,RAD_I32,{p.rows,p.topk});check(3,RAD_F32,{p.rows,p.topk});
    } else {
        p.heads=gi("heads");p.dim=gi("head_dim");p.kv_rows=gi("kv_rows");p.topk=gi("topk");p.scale=gf("scale");
        if(p.heads<=0||p.heads>INT_MAX||p.dim<=0||p.dim>max_sparse_dim||p.kv_rows<=0||p.kv_rows>INT_MAX||
           p.topk<=0||p.topk>max_sparse_topk||p.rows>INT_MAX/p.heads)return RAD_E_SHAPE;
        if(!(p.scale>0)||!std::isfinite(p.scale))return RAD_E_INVAL;
        check(0,dtype,{p.rows,p.heads,p.dim});check(1,dtype,{p.kv_rows,p.dim});check(2,RAD_F32,{p.heads});
        check(3,RAD_I32,{p.rows,p.topk});check(4,dtype,{p.rows,p.heads,p.dim});
    }
    if(rc!=RAD_OK)return rc;
    const int first_output=op==Op::hc_prepare?4:op==Op::hc_collapse?2:op==Op::router_topk?2:4;
    for(int i=first_output;i<nt;++i)for(int j=0;j<nt;++j)
        if(i!=j&&overlaps(p.t[i].data,p.spans[i],p.t[j].data,p.spans[j]))return RAD_E_INVAL;
    if(!a->scratch||a->scratch_bytes<scratch_bytes)return RAD_E_INVAL;
    if(uintptr_t(a->scratch)%alignof(int32_t))return RAD_E_ALIGN;
    if(uintptr_t(a->scratch)>UINTPTR_MAX-uint64_t(scratch_bytes))return RAD_E_SHAPE;
    for(int i=0;i<nt;++i)if(overlaps(a->scratch,scratch_bytes,p.t[i].data,p.spans[i]))return RAD_E_INVAL;
    p.status=static_cast<int32_t*>(a->scratch);*out=p;return RAD_OK;
}
} // namespace rad::ds41::math
extern "C" {
int ds41_hc_prepare(const RadArgs*,RadStream);
int ds41_hc_collapse(const RadArgs*,RadStream);
int ds41_router_topk(const RadArgs*,RadStream);
int ds41_sparse_attn(const RadArgs*,RadStream);
// Called at declare without live operands: status prefix plus staged device Plan.
// CPU emulator scratch_bytes remains the independent four-byte semantic flag.
inline int64_t ds41_math_scratch(const RadArgs*) {return rad::ds41::gpu_scratch_bytes<rad::ds41::math::Plan>();}
}
