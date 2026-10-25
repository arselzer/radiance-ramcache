/* Floating topology completion for emitted compressor rows, not official QDQ.
 * No pending state, compaction, cache store or graph ownership is hidden here. */
#pragma once
#include "ds41_device_plan.h"
#include "ds41_math_impl.h"

namespace rad::ds41::compress_finish {
enum Operand {pooled=0,emit,first_positions,gain,pairs,unrotated,rotated,operand_count};
inline constexpr int64_t max_rows=65536,max_dimension=1024,max_positions=1048576;
inline constexpr int64_t status_offset=0,scratch_bytes=4;
struct Geometry {int64_t capacity=0,dim=0,rotary=0,ratio=0;uint32_t input_dtype=0,dtype=0;float eps=0;};
struct Plan {
    RadTensor t[operand_count]{};int64_t spans[operand_count]{};
    int64_t rows=0,dim=0,rotary=0,ratio=0,positions=0;uint32_t dtype=0;float eps=0;
    int32_t* status=nullptr;
};
inline int geometry(const RadParam* p,int np,Geometry* out,bool declaring=false){
    if(!p||np<=0||!out)return RAD_E_INVAL;
    auto integer=[&](const char* key){return declaring?rad_param_getdim(p,np,key,0):rad_param_geti(p,np,key,0);};
    Geometry g;g.capacity=integer("M");g.dim=integer("n");g.rotary=integer("rotary_dim");g.ratio=integer("ratio");
    g.input_dtype=rad_dtype_parse(rad_param_gets(p,np,"a_dtype",nullptr));g.dtype=rad_dtype_parse(rad_param_gets(p,np,"dtype",nullptr));g.eps=float(rad_param_getf(p,np,"eps",0));
    if(!math::scalar_dtype(g.input_dtype)||!math::scalar_dtype(g.dtype))return RAD_E_DTYPE;
    if(g.capacity<=0||g.capacity>max_rows||g.dim<=0||g.dim>max_dimension||g.rotary<=0||g.rotary>g.dim||(g.rotary&1)||(g.ratio!=1&&g.ratio!=2))return RAD_E_SHAPE;
    if(!(g.eps>0)||!std::isfinite(g.eps))return RAD_E_INVAL;
    *out=g;return RAD_OK;
}
inline int validate(const RadArgs* a,Plan* out){
    if(!a||!out||!a->t||a->n_t!=operand_count)return RAD_E_INVAL;
    Geometry g;const int rc=geometry(a->p,a->n_p,&g);if(rc!=RAD_OK)return rc;
    if(a->t[pooled].rank!=2||a->t[pairs].rank!=3)return RAD_E_SHAPE;
    Plan p;for(int i=0;i<operand_count;++i)p.t[i]=a->t[i];
    p.rows=p.t[pooled].shape[0];p.positions=p.t[pairs].shape[0];p.dim=g.dim;p.rotary=g.rotary;p.ratio=g.ratio;p.dtype=g.dtype;p.eps=g.eps;
    if(p.rows<=0||p.rows>g.capacity||p.positions<=0||p.positions>max_positions)return RAD_E_SHAPE;
    int checked=RAD_OK;auto tensor=[&](int i,uint32_t dtype,std::initializer_list<int64_t> shape){if(checked==RAD_OK)checked=math::tensor(p.t[i],dtype,shape,&p.spans[i]);};
    tensor(pooled,g.input_dtype,{p.rows,p.dim});tensor(emit,RAD_I32,{p.rows});tensor(first_positions,RAD_I32,{p.rows});
    tensor(gain,RAD_F32,{p.dim});tensor(pairs,RAD_F32,{p.positions,p.rotary/2,2});tensor(unrotated,g.dtype,{p.rows,p.dim});tensor(rotated,g.dtype,{p.rows,p.dim});
    if(checked!=RAD_OK)return checked;
    for(int i=unrotated;i<=rotated;++i)for(int j=0;j<operand_count;++j)if(i!=j&&math::overlaps(p.t[i].data,p.spans[i],p.t[j].data,p.spans[j]))return RAD_E_INVAL;
    if(!a->scratch||a->scratch_bytes<scratch_bytes)return RAD_E_INVAL;
    if(uintptr_t(a->scratch)%alignof(int32_t))return RAD_E_ALIGN;
    if(uintptr_t(a->scratch)>UINTPTR_MAX-uint64_t(scratch_bytes))return RAD_E_SHAPE;
    for(int i=0;i<operand_count;++i)if(math::overlaps(a->scratch,scratch_bytes,p.t[i].data,p.spans[i]))return RAD_E_INVAL;
    p.status=static_cast<int32_t*>(a->scratch);*out=p;return RAD_OK;
}
inline int shape(const RadParam* p,int np,int operand,RadOpdDesc* out){
    if(!out||operand<0||operand>=operand_count)return RAD_E_INVAL;
    Geometry g;const int rc=geometry(p,np,&g,true);if(rc!=RAD_OK)return rc;
    // P is a caller-owned table extent, absent from this fixed parameter contract.
    // Generic fixture tools must refuse rather than allocate a guessed full-context table.
    if(operand==pairs)return RAD_E_UNSUPPORTED;
    RadOpdDesc d{};d.idx_const=-1;
    if(operand==pooled||operand==unrotated||operand==rotated){d.dtype=operand==pooled?g.input_dtype:g.dtype;d.rank=2;d.shape[0]=g.capacity;d.shape[1]=g.dim;}
    else if(operand==emit||operand==first_positions){d.dtype=RAD_I32;d.rank=1;d.shape[0]=g.capacity;}
    else if(operand==gain){d.dtype=RAD_F32;d.rank=1;d.shape[0]=g.dim;}
    *out=d;return RAD_OK;
}
inline const RadParamSpec params[]{
    {"M",RAD_P_INT,RAD_REQUIRED,RAD_PROLE_CAPACITY},{"n",RAD_P_INT,RAD_REQUIRED},
    {"rotary_dim",RAD_P_INT,RAD_REQUIRED},{"ratio",RAD_P_INT,RAD_REQUIRED},
    {"a_dtype",RAD_P_STR,RAD_REQUIRED},{"dtype",RAD_P_STR,RAD_REQUIRED},{"eps",RAD_P_F64,RAD_REQUIRED}};
inline const RadOperandSpec operands[]{
    {"pooled",RAD_OPD_IN,0},{"emit",RAD_OPD_IN,0},{"first_positions",RAD_OPD_IN,0},{"gain",RAD_OPD_IN,0},{"pairs",RAD_OPD_IN,0},
    {"unrotated",RAD_OPD_OUT,0},{"rotated",RAD_OPD_OUT,0}};
inline const RadOpSchema schema{"ds41_compress_finish",params,7,operands,operand_count,
    "Mask compressor completions before payload reads; cast before RMS, round before tail RoPE; inactive outputs are +0. Floating topology only."};
} // namespace rad::ds41::compress_finish
extern "C" {
int ds41_compress_finish(const RadArgs*,RadStream);
int ds41_ref_compress_finish(const RadArgs*,RadStream) noexcept;
inline int64_t ds41_compress_finish_scratch(const RadArgs*){return rad::ds41::gpu_scratch_bytes<rad::ds41::compress_finish::Plan>();}
}
