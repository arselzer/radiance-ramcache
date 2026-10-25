/* Non-speculative DS41 ratio-1/2 pooling over supplied projections. Opaque U32 LINEAR state
 * preserves pending float bits and an exact original-token cursor through raw checkpoint copies.
 * This is an experimental standalone op, not a production architecture or lifecycle. */
#pragma once
#include "ds41_device_plan.h"
#include "ds41_math_impl.h"
#include "rad_builder.h"

namespace rad::ds41::compress {
inline constexpr int tensors_count=11,max_dimension=1024;
inline constexpr int64_t status_offset=0,status_bytes=4,scratch_bytes=4;
enum Operand {projected_kv=0,gate_scores=1,pending_state=2,state_ids=3,has_init=4,cu=5,
              positions=6,num_accepted=7,pooled=8,emit=9,first_position=10};
struct Plan {
    RadTensor t[tensors_count]{};int64_t spans[tensors_count]{};
    int64_t tokens=0,dimension=0,ratio=0,sequences=0,states=0;bool accepted=false;
    int32_t* status=nullptr;
};
inline bool absent(const RadTensor& t) {return !t.data&&t.rank==0;}
inline int validate(const RadArgs* a,Plan* out) {
    if(!a||!out||!a->t||a->n_t!=tensors_count||!a->p||a->n_p<=0)return RAD_E_INVAL;
    const auto* spec=rad_param_find(a->p,a->n_p,"n_spec");
    if(!spec||spec->kind!=RAD_P_INT)return RAD_E_INVAL;
    if(spec->ival!=0)return RAD_E_UNSUPPORTED;
    Plan p;for(int i=0;i<tensors_count;++i)p.t[i]=a->t[i];
    const int64_t capacity=rad_param_geti(a->p,a->n_p,"M",0);
    p.dimension=rad_param_geti(a->p,a->n_p,"dim",0);p.ratio=rad_param_geti(a->p,a->n_p,"ratio",0);
    const uint32_t dtype=rad_dtype_parse(rad_param_gets(a->p,a->n_p,"dtype",nullptr));
    if(!math::scalar_dtype(dtype)||(p.ratio==2&&dtype!=RAD_F32))return RAD_E_DTYPE;
    if(capacity<=0||capacity>INT_MAX||p.dimension<=0||p.dimension>max_dimension||
       (p.ratio!=1&&p.ratio!=2)||p.t[projected_kv].rank!=2||p.t[cu].rank!=1||p.t[pending_state].rank!=4)
        return RAD_E_SHAPE;
    p.tokens=p.t[projected_kv].shape[0];p.sequences=p.t[cu].shape[0]-1;p.states=p.t[pending_state].shape[0];
    if(p.tokens<=0||p.tokens>capacity||p.sequences<=0||p.sequences>=INT_MAX||p.states<=0||p.states>INT_MAX)
        return RAD_E_SHAPE;
    int rc=RAD_OK;
    auto check=[&](int i,uint32_t d,std::initializer_list<int64_t> shape) {
        if(rc==RAD_OK)rc=math::tensor(p.t[i],d,shape,&p.spans[i]);
    };
    check(projected_kv,dtype,{p.tokens,p.dimension});
    if(p.ratio==2)check(gate_scores,RAD_F32,{p.tokens,p.dimension});
    else if(!absent(p.t[gate_scores]))return RAD_E_INVAL;
    check(pending_state,RAD_U32,{p.states,1,2,p.dimension+1});
    check(state_ids,RAD_I32,{p.sequences,1});check(has_init,RAD_I32,{p.sequences});
    check(cu,RAD_I32,{p.sequences+1});check(positions,RAD_I32,{p.tokens});
    p.accepted=!absent(p.t[num_accepted]);if(p.accepted)check(num_accepted,RAD_I32,{p.sequences});
    check(pooled,RAD_F32,{p.tokens,p.dimension});check(emit,RAD_I32,{p.tokens});check(first_position,RAD_I32,{p.tokens});
    if(rc!=RAD_OK)return rc;
    for(int i:{pending_state,pooled,emit,first_position})for(int j=0;j<tensors_count;++j)
        if(i!=j&&math::overlaps(p.t[i].data,p.spans[i],p.t[j].data,p.spans[j]))return RAD_E_INVAL;
    if(!a->scratch||a->scratch_bytes<scratch_bytes)return RAD_E_INVAL;
    if(uintptr_t(a->scratch)%alignof(int32_t))return RAD_E_ALIGN;
    if(uintptr_t(a->scratch)>UINTPTR_MAX-uint64_t(scratch_bytes))return RAD_E_SHAPE;
    for(int i=0;i<tensors_count;++i)if(math::overlaps(a->scratch,scratch_bytes,p.t[i].data,p.spans[i]))return RAD_E_INVAL;
    p.status=static_cast<int32_t*>(a->scratch);*out=p;return RAD_OK;
}
inline int state_decl(int64_t dimension,uint32_t requested_state_dtype,int max_spec,RadKVGroupDecl* out) {
    if(!out)return RAD_E_INVAL;
    if(max_spec!=0||(requested_state_dtype!=0&&requested_state_dtype!=RAD_U32))return RAD_E_UNSUPPORTED;
    if(dimension<=0||dimension>max_dimension)return RAD_E_SHAPE;
    RadKVGroupDecl d{};d.kind=RAD_KV_LINEAR;d.dtype=RAD_U32;d.n_head_kv=1;d.state_dim[0]=2;d.state_dim[1]=dimension+1;
    *out=d;return RAD_OK;
}
inline const RadParamSpec params[]{
    {"M",RAD_P_INT,RAD_REQUIRED,RAD_PROLE_CAPACITY},{"dim",RAD_P_INT,RAD_REQUIRED},
    {"ratio",RAD_P_INT,RAD_REQUIRED},{"dtype",RAD_P_STR,RAD_REQUIRED},{"n_spec",RAD_P_INT,RAD_REQUIRED}};
inline const RadOperandSpec operands[]{
    {"projected_kv",RAD_OPD_IN,0},{"gate_scores",RAD_OPD_IN,1},{"pending_state",RAD_OPD_INOUT,0},
    {"state_ids",RAD_OPD_IN,0},{"has_init",RAD_OPD_IN,0},{"cu",RAD_OPD_IN,0},{"positions",RAD_OPD_IN,0},
    {"num_accepted",RAD_OPD_IN,1},{"pooled",RAD_OPD_OUT,0},{"emit",RAD_OPD_OUT,0},{"first_position",RAD_OPD_OUT,0}};
inline const RadOpSchema schema{"ds41_compress_pool",params,5,operands,tensors_count,
    "non-speculative projected ratio-1/2 pool; original-row outputs, opaque U32 pending state; norm/RoPE remain separate"};
} // namespace rad::ds41::compress
extern "C" {
int ds41_compress_pool(const RadArgs*,RadStream);
// GPU reservation includes the staged descriptor; CPU emulate still needs only4B.
inline int64_t ds41_compress_scratch(const RadArgs*) {return rad::ds41::gpu_scratch_bytes<rad::ds41::compress::Plan>();}
}
