/* Direct shared index-key reads from core's FULL paged KV representation.
 * Logical compressed entry j starts at original token j*ratio, never j/ratio. */
#pragma once
#include "ds41_index.h"
namespace rad::ds41::paged_index {
struct Plan {
    RadTensor t[8]{};int64_t spans[8]{};
    int64_t rows=0,keys=0,heads=0,dim=0,sequences=0,logical_blocks=0,physical_blocks=0,block_size=0,ratio=0;
    float scale=0;bool origins=false;int32_t* status=nullptr;
};
inline int validate(const RadArgs* a,Plan* out) {
    if(!a||!out||!a->t||a->n_t!=8||!a->p||a->n_p<=0)return RAD_E_INVAL;
    Plan p;for(int i=0;i<8;++i)p.t[i]=a->t[i];
    auto gi=[&](const char* k){return int64_t(rad_param_geti(a->p,a->n_p,k,0));};
    const int64_t capacity=gi("M"),key_capacity=gi("key_rows"),seq_capacity=gi("sequences"),global=gi("global_heads");
    p.heads=gi("heads");p.dim=gi("head_dim");p.ratio=gi("ratio");p.block_size=gi("block_size");
    const auto dtype=rad_dtype_parse(rad_param_gets(a->p,a->n_p,"dtype",nullptr));
    if(!math::scalar_dtype(dtype))return RAD_E_DTYPE;
    if(capacity<=0||capacity>index::max_rows||key_capacity<=0||key_capacity>index::max_keys||
       seq_capacity<=0||seq_capacity>index::max_sequences||p.heads<=0||p.heads>index::max_heads||global<p.heads||global>index::max_sequences||
       p.dim<=0||p.dim>index::max_dimension||(p.ratio!=1&&p.ratio!=2)||p.dim%(2*p.ratio)||
       p.block_size<=0||p.block_size>65536||p.block_size%p.ratio||
       p.t[0].rank!=3||p.t[1].rank!=4||p.t[5].rank!=2||p.t[7].rank!=2)return RAD_E_SHAPE;
    p.rows=p.t[0].shape[0];p.physical_blocks=p.t[1].shape[0];p.sequences=p.t[5].shape[0];
    p.logical_blocks=p.t[5].shape[1];p.keys=p.t[7].shape[1];
    if(p.rows<=0||p.rows>capacity||p.keys<=0||p.keys>key_capacity||p.sequences<=0||p.sequences>seq_capacity||
       p.physical_blocks<=0||p.physical_blocks>INT_MAX||p.logical_blocks<=0||p.logical_blocks>index::max_keys*2)return RAD_E_SHAPE;
    int rc=RAD_OK;auto check=[&](int i,uint32_t d,std::initializer_list<int64_t> shape){if(rc==RAD_OK)rc=math::tensor(p.t[i],d,shape,&p.spans[i]);};
    check(0,dtype,{p.rows,p.heads,p.dim});check(1,dtype,{p.physical_blocks,1,p.block_size,p.dim/p.ratio});
    check(2,RAD_F32,{p.rows,p.heads});check(3,RAD_I32,{p.rows});check(4,RAD_I32,{p.rows});
    check(5,RAD_I32,{p.sequences,p.logical_blocks});p.origins=p.t[6].data||p.t[6].rank;
    if(p.origins)check(6,RAD_I32,{p.sequences});
    check(7,RAD_F32,{p.rows,p.keys});
    if(rc!=RAD_OK)return rc;
    for(int i=0;i<7;++i)if(math::overlaps(p.t[7].data,p.spans[7],p.t[i].data,p.spans[i]))return RAD_E_INVAL;
    if(!a->scratch||a->scratch_bytes<4)return RAD_E_INVAL;
    if(uintptr_t(a->scratch)%alignof(int32_t))return RAD_E_ALIGN;
    if(uintptr_t(a->scratch)>UINTPTR_MAX-4)return RAD_E_SHAPE;
    for(int i=0;i<8;++i)if(math::overlaps(a->scratch,4,p.t[i].data,p.spans[i]))return RAD_E_INVAL;
    p.scale=(1.f/std::sqrt(float(p.dim)))*(1.f/std::sqrt(float(global)));p.status=static_cast<int32_t*>(a->scratch);*out=p;return RAD_OK;
}
inline const RadParamSpec params[]{
    {"M",RAD_P_INT,RAD_REQUIRED,RAD_PROLE_CAPACITY},{"key_rows",RAD_P_INT,RAD_REQUIRED},
    {"heads",RAD_P_INT,RAD_REQUIRED},{"global_heads",RAD_P_INT,RAD_REQUIRED},{"head_dim",RAD_P_INT,RAD_REQUIRED},
    {"sequences",RAD_P_INT,RAD_REQUIRED},{"dtype",RAD_P_STR,RAD_REQUIRED},{"ratio",RAD_P_INT,RAD_REQUIRED},
    {"block_size",RAD_P_INT,RAD_DERIVED}};
inline const RadOperandSpec operands[]{
    {"query",RAD_OPD_IN,0},{"cache",RAD_OPD_IN,0},{"head_weights",RAD_OPD_IN,0},{"sequence_ids",RAD_OPD_IN,0},
    {"visible",RAD_OPD_IN,0},{"block_table",RAD_OPD_IN,0},{"table_first_pos",RAD_OPD_IN,1},{"scores",RAD_OPD_OUT,0}};
inline const RadOpSchema schema{"ds41_index_scores_paged",params,9,operands,8,
    "Direct FULL paged shared-key reads; original-token block coordinates; floating score/scale contract; external TP reduction."};
}
extern "C" {
int ds41_index_scores_paged(const RadArgs*,RadStream);
inline int64_t ds41_index_paged_scratch(const RadArgs*) {return rad::ds41::gpu_scratch_bytes<rad::ds41::paged_index::Plan>();}
}
