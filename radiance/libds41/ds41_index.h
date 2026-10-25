/* Experimental floating-point DS41 index scoring and deterministic selection.
 * No compressed-cache allocation, routing lifecycle or model graph is implied. */
#pragma once
#include "ds41_device_plan.h"
#include "ds41_math_impl.h"
#include <algorithm>
namespace rad::ds41::index {
enum class Op {scores,candidates,topk};
inline constexpr int max_heads=1024,max_dimension=1024,max_topk=4096;
inline constexpr int64_t max_keys=1048576,max_rows=65536,max_sequences=65536;
struct Entry {float score;int32_t id;};
struct Plan {
    Op op{};int nt=0;RadTensor t[6]{};int64_t spans[6]{};
    int64_t rows=0,keys=0,heads=0,dim=0,sequences=0,topk=0,block_size=0;
    float scale=0;bool mask=false;int32_t* status=nullptr;
};
inline int validate(Op op,const RadArgs* a,Plan* out) {
    if(unsigned(op)>2||!a||!out||!a->t||!a->p||a->n_p<=0)return RAD_E_INVAL;
    const int nt=op==Op::scores?6:op==Op::candidates?3:4;
    if(a->n_t!=nt)return RAD_E_INVAL;
    Plan p;p.op=op;p.nt=nt;for(int i=0;i<nt;++i)p.t[i]=a->t[i];
    auto gi=[&](const char* key){return int64_t(rad_param_geti(a->p,a->n_p,key,0));};
    const int64_t capacity=gi("M"),key_capacity=gi("key_rows");
    if(capacity<=0||capacity>max_rows||key_capacity<=0||key_capacity>max_keys||p.t[0].rank<2)return RAD_E_SHAPE;
    p.rows=p.t[0].shape[0];if(p.rows<=0||p.rows>capacity)return RAD_E_SHAPE;
    int rc=RAD_OK;
    auto check=[&](int i,uint32_t dtype,std::initializer_list<int64_t> shape){
        if(rc==RAD_OK)rc=math::tensor(p.t[i],dtype,shape,&p.spans[i]);
    };
    int first_output=0;
    if(op==Op::scores) {
        p.heads=gi("heads");p.dim=gi("head_dim");const int64_t global_heads=gi("global_heads"),seq_capacity=gi("sequences");
        const auto dtype=rad_dtype_parse(rad_param_gets(a->p,a->n_p,"dtype",nullptr));
        if(!math::scalar_dtype(dtype))return RAD_E_DTYPE;
        if(p.heads<=0||p.heads>max_heads||p.dim<=0||p.dim>max_dimension||global_heads<p.heads||global_heads>max_sequences||
           seq_capacity<=0||seq_capacity>max_sequences||p.t[1].rank!=3)return RAD_E_SHAPE;
        p.sequences=p.t[1].shape[0];p.keys=p.t[1].shape[1];
        if(p.sequences<=0||p.sequences>seq_capacity||p.keys<=0||p.keys>key_capacity)return RAD_E_SHAPE;
        p.scale=(1.f/std::sqrt(float(p.dim)))*(1.f/std::sqrt(float(global_heads)));
        check(0,dtype,{p.rows,p.heads,p.dim});check(1,dtype,{p.sequences,p.keys,p.dim});
        check(2,RAD_F32,{p.rows,p.heads});check(3,RAD_I32,{p.rows});check(4,RAD_I32,{p.rows});
        check(5,RAD_F32,{p.rows,p.keys});first_output=5;
    } else {
        p.keys=p.t[0].shape[1];if(p.keys<=0||p.keys>key_capacity)return RAD_E_SHAPE;
        check(0,RAD_F32,{p.rows,p.keys});check(1,RAD_I32,{p.rows});
        if(op==Op::candidates) {
            p.block_size=gi("block_size");p.topk=gi("top_blocks");
            if(p.block_size<=0||p.block_size>max_keys||p.topk<=0||p.topk>max_topk)return RAD_E_SHAPE;
            check(2,RAD_U8,{p.rows,p.keys});first_output=2;
        } else {
            const int64_t desired=gi("topk");if(desired<=0||desired>max_topk)return RAD_E_SHAPE;
            p.topk=std::min(desired,p.keys);p.mask=p.t[2].data||p.t[2].rank;
            if(p.mask)check(2,RAD_U8,{p.rows,p.keys});
            check(3,RAD_I32,{p.rows,p.topk});first_output=3;
        }
    }
    if(rc!=RAD_OK)return rc;
    for(int i=first_output;i<nt;++i)for(int j=0;j<nt;++j)
        if(i!=j&&math::overlaps(p.t[i].data,p.spans[i],p.t[j].data,p.spans[j]))return RAD_E_INVAL;
    if(!a->scratch||a->scratch_bytes<4)return RAD_E_INVAL;
    if(uintptr_t(a->scratch)%alignof(int32_t))return RAD_E_ALIGN;
    if(uintptr_t(a->scratch)>UINTPTR_MAX-4)return RAD_E_SHAPE;
    for(int i=0;i<nt;++i)if(math::overlaps(a->scratch,4,p.t[i].data,p.spans[i]))return RAD_E_INVAL;
    p.status=static_cast<int32_t*>(a->scratch);*out=p;return RAD_OK;
}
inline const RadParamSpec score_params[]{
    {"M",RAD_P_INT,RAD_REQUIRED,RAD_PROLE_CAPACITY},{"key_rows",RAD_P_INT,RAD_REQUIRED},
    {"heads",RAD_P_INT,RAD_REQUIRED},{"global_heads",RAD_P_INT,RAD_REQUIRED},
    {"head_dim",RAD_P_INT,RAD_REQUIRED},{"sequences",RAD_P_INT,RAD_REQUIRED},{"dtype",RAD_P_STR,RAD_REQUIRED}};
inline const RadParamSpec candidate_params[]{
    {"M",RAD_P_INT,RAD_REQUIRED,RAD_PROLE_CAPACITY},{"key_rows",RAD_P_INT,RAD_REQUIRED},
    {"block_size",RAD_P_INT,RAD_REQUIRED},{"top_blocks",RAD_P_INT,RAD_REQUIRED}};
inline const RadParamSpec topk_params[]{
    {"M",RAD_P_INT,RAD_REQUIRED,RAD_PROLE_CAPACITY},{"key_rows",RAD_P_INT,RAD_REQUIRED},{"topk",RAD_P_INT,RAD_REQUIRED}};
inline const RadOperandSpec score_operands[]{
    {"query",RAD_OPD_IN,0},{"keys",RAD_OPD_IN,0},{"head_weights",RAD_OPD_IN,0},
    {"sequence_ids",RAD_OPD_IN,0},{"visible",RAD_OPD_IN,0},{"scores",RAD_OPD_OUT,0}};
inline const RadOperandSpec candidate_operands[]{{"scores",RAD_OPD_IN,0},{"visible",RAD_OPD_IN,0},{"keep",RAD_OPD_OUT,0}};
inline const RadOperandSpec topk_operands[]{{"scores",RAD_OPD_IN,0},{"visible",RAD_OPD_IN,0},
    {"candidates",RAD_OPD_IN,1},{"indices",RAD_OPD_OUT,0}};
inline const RadOpSchema schemas[]{
    {"ds41_index_scores",score_params,7,score_operands,6,"Floating index scores; supplied unscaled F32 head weights, global head scale; TP score reduction is external."},
    {"ds41_candidate_blocks",candidate_params,4,candidate_operands,3,"Visible block maxima; newest block forced; selected block includes its invisible tail."},
    {"ds41_index_topk",topk_params,3,topk_operands,4,"Highest scores with lower-position ties, then position sort; -1 padding, optional candidate mask."}};
} // namespace rad::ds41::index
extern "C" {
int ds41_index_scores(const RadArgs*,RadStream);
int ds41_candidate_blocks(const RadArgs*,RadStream);
int ds41_index_topk(const RadArgs*,RadStream);
inline int64_t ds41_index_scratch(const RadArgs*) {return rad::ds41::gpu_scratch_bytes<rad::ds41::index::Plan>();}
}
