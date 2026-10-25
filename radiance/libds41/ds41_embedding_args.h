/* Host metadata validation/layout adapters only. The ABI describes trusted allocation extents,
 * not physical capacities. Native rows must be present for all declared vocabulary rows. */
#pragma once
#include "ds41_embedding.h"
#include "ds41_packed_args.h"
#include <array>
namespace rad::ds41::embedding {
struct Geometry {packed::Type type{};int64_t capacity=0,n=0,k=0,row_bytes=0;uint32_t dtype=0;};
inline int geometry(const RadParam* p,int np,Geometry* out,bool declaring=false) {
    if(!p||np<=0||!out)return RAD_E_INVAL;
    Geometry g;
    auto get=[&](const char* key){return declaring?rad_param_getdim(p,np,key,0):rad_param_geti(p,np,key,0);};
    g.capacity=get("M");g.n=get("N");g.k=get("K");
    if(!packed::parse_type(rad_param_gets(p,np,"format",nullptr),&g.type))return RAD_E_DTYPE;
    g.dtype=rad_dtype_parse(rad_param_gets(p,np,"y_dtype",nullptr));
    if(!math::scalar_dtype(g.dtype))return RAD_E_DTYPE;
    if(g.capacity<=0||g.capacity>max_rows||g.n<=0||g.n>INT_MAX||g.k<=0||g.k>INT_MAX||
       g.k%packed::block_values(g.type))return RAD_E_SHAPE;
    g.row_bytes=(g.k/packed::block_values(g.type))*packed::block_bytes(g.type);
    if(g.n>INT64_MAX/g.row_bytes)return RAD_E_SHAPE;
    *out=g;return RAD_OK;
}
inline int validate(const RadArgs* a,Plan* out) {
    if(!a||!out||!a->t||a->n_t!=operand_count)return RAD_E_INVAL;
    Geometry g;int rc=geometry(a->p,a->n_p,&g);if(rc!=RAD_OK)return rc;
    Plan p;for(int i=0;i<operand_count;++i)p.t[i]=a->t[i];
    if(p.t[token_ids].rank!=1)return RAD_E_SHAPE;
    p.rows=p.t[token_ids].shape[0];p.n=g.n;p.k=g.k;p.type=g.type;p.row_bytes=g.row_bytes;p.y_dtype=g.dtype;
    if(p.rows<=0||p.rows>g.capacity)return RAD_E_SHAPE;
    rc=math::tensor(p.t[token_ids],RAD_I32,{p.rows},&p.spans[token_ids]);if(rc!=RAD_OK)return rc;
    rc=math::tensor(p.t[output],g.dtype,{p.rows,g.k},&p.spans[output]);if(rc!=RAD_OK)return rc;
    const auto& w=p.t[weight];
    if(!w.data)return RAD_E_INVAL;
    if(w.dtype!=packed::storage_dtype(g.type))return RAD_E_DTYPE;
    if(w.rank!=2||w.shape[0]!=g.n||w.shape[1]!=g.k)return RAD_E_SHAPE;
    if(w.stride[0]!=g.k||w.stride[1]!=1)return RAD_E_STRIDE;
    // Decoder uses byte loads. No stronger alignment is required to read a native GGUF row;
    // the canonical->native layout still requests align256 from the allocator.
    p.spans[weight]=g.n*g.row_bytes;
    if(uintptr_t(w.data)>UINTPTR_MAX-uint64_t(p.spans[weight]))return RAD_E_SHAPE;
    for(int i=0;i<operand_count;++i)for(int j=0;j<i;++j)
        if(math::overlaps(p.t[i].data,p.spans[i],p.t[j].data,p.spans[j]))return RAD_E_INVAL;
    if(!a->scratch||a->scratch_bytes<scratch_bytes)return RAD_E_INVAL;
    if(uintptr_t(a->scratch)%alignof(int32_t))return RAD_E_ALIGN;
    if(uintptr_t(a->scratch)>UINTPTR_MAX-uint64_t(scratch_bytes))return RAD_E_SHAPE;
    for(int i=0;i<operand_count;++i)
        if(math::overlaps(a->scratch,scratch_bytes,p.t[i].data,p.spans[i]))return RAD_E_INVAL;
    p.status=static_cast<int32_t*>(a->scratch);*out=p;return RAD_OK;
}
/* All canonical planes are required, exactly as for packed GEMM. The internal adapter only
 * supplies the unrelated GEMM activation metadata; callers never need an a_dtype parameter. */
inline std::array<RadParam,6> packed_params(const Geometry& g) {
    return {RAD_INT("M",1),RAD_INT("N",g.n),RAD_INT("K",g.k),
            RAD_STR("format",packed::type_name(g.type)),RAD_STR("a_dtype","f32"),RAD_STR("y_dtype",rad_dtype_name(g.dtype))};
}
inline int layout(const RadParam* p,int np,int operand,const RadEncoding* enc,const int* sel,
                  const RadTensor* planes,int n,RadLayout* out) {
    if(operand!=weight)return RAD_E_INVAL;
    Geometry g;const int rc=geometry(p,np,&g,true);if(rc!=RAD_OK)return rc;
    const auto bridge=packed_params(g);return packed::layout(bridge.data(),bridge.size(),weight,enc,sel,planes,n,out);
}
inline int relayout(const RadParam* p,int np,int operand,const RadEncoding* enc,const int* sel,
                    const RadTensor* planes,int n,void* dst,int64_t bytes) {
    if(operand!=weight)return RAD_E_INVAL;
    Geometry g;const int rc=geometry(p,np,&g,true);if(rc!=RAD_OK)return rc;
    const auto bridge=packed_params(g);return packed::relayout(bridge.data(),bridge.size(),weight,enc,sel,planes,n,dst,bytes);
}
inline int unrelayout(const RadParam* p,int np,int operand,const RadEncoding* enc,const int* sel,
                      const void* src,int64_t bytes,const RadTensor* planes,int n) {
    if(operand!=weight)return RAD_E_INVAL;
    Geometry g;const int rc=geometry(p,np,&g,true);if(rc!=RAD_OK)return rc;
    const auto bridge=packed_params(g);return packed::unrelayout(bridge.data(),bridge.size(),weight,enc,sel,src,bytes,planes,n);
}
inline int shape(const RadParam* p,int np,int operand,RadOpdDesc* out) {
    if(!out||operand<0||operand>=operand_count)return RAD_E_INVAL;
    Geometry g;const int rc=geometry(p,np,&g,true);if(rc!=RAD_OK)return rc;
    if(operand==weight)return RAD_E_UNSUPPORTED;
    RadOpdDesc d{};d.idx_const=-1;
    if(operand==token_ids){d.dtype=RAD_I32;d.rank=1;d.shape[0]=g.capacity;d.fill=RAD_FILL_INDEX;d.idx_max=g.n;}
    else {d.dtype=g.dtype;d.rank=2;d.shape[0]=g.capacity;d.shape[1]=g.k;}
    *out=d;return RAD_OK;
}
} // namespace rad::ds41::embedding
