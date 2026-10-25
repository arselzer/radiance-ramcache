/* Pure host validation and layout hooks for experimental packed GEMMs. The public ABI
 * advertises geometry, not allocation capacity: payload extents remain a caller guarantee. */
#pragma once
#include "rad_plugin.h"
#include "ds41_packed.h"
#include "../arch/common/rad_gguf_relayout.h"
#include <climits>

namespace rad::ds41::packed {
// Runtime tensors retain logical N,K even for opaque stored weights. A private dtype keeps
// generic scalar diagnostics from treating N*K as readable U8 payload bytes (cf. r4d_args.h).
inline constexpr uint32_t storage_dtype(Type type) {
    return static_cast<unsigned>(type)<6?RAD_DT_PLUGIN_BASE+0x4100u+static_cast<unsigned>(type):uint32_t(RAD_DT_INVALID);
}
inline const char* runtime_tag(Type type) {
    static constexpr const char* tags[]{"ds41.gguf.iq2_xxs.v1","ds41.gguf.iq2_xs.v1",
        "ds41.gguf.iq2_s.v1","ds41.gguf.iq3_xxs.v1","ds41.gguf.q5_K.v1","ds41.gguf.q8_0.v1"};
    return static_cast<unsigned>(type)<6?tags[static_cast<unsigned>(type)]:nullptr;
}
inline const char* type_name(Type t) {
    static constexpr const char* names[]{"iq2_xxs","iq2_xs","iq2_s","iq3_xxs","q5_K","q8_0"};
    return static_cast<unsigned>(t)<6?names[static_cast<unsigned>(t)]:nullptr;
}
inline bool parse_type(const char* text,Type* out) {
    if(text&&out)for(int i=0;i<6;++i)if(!std::strcmp(text,type_name(Type(i)))) {*out=Type(i);return true;}
    return false;
}
inline int geometry(const RadParam* p,int np,bool routed,Geometry* out,bool shape_hook=false) {
    if(!p||np<=0||!out)return RAD_E_INVAL;
    Geometry g;
    auto integer=[&](const char* key) {
        return shape_hook?rad_param_getdim(p,np,key,0):rad_param_geti(p,np,key,0);
    };
    g.m=integer("M");g.n=integer("N");g.k=integer("K");
    g.r=routed?integer("R"):g.m;g.experts=routed?integer("n_expert"):1;
    if(!parse_type(rad_param_gets(p,np,"format",nullptr),&g.type))return RAD_E_DTYPE;
    g.a_dtype=rad_dtype_parse(rad_param_gets(p,np,"a_dtype",nullptr));
    g.y_dtype=rad_dtype_parse(rad_param_gets(p,np,"y_dtype",nullptr));
    auto scalar=[](uint32_t d){return d==RAD_F32||d==RAD_BF16||d==RAD_F16;};
    if(!scalar(g.a_dtype)||!scalar(g.y_dtype))return RAD_E_DTYPE;
    if(g.m<=0||g.n<=0||g.k<=0||g.r<=0||g.experts<=0||
       g.m>INT_MAX||g.n>INT_MAX||g.k>INT_MAX||g.r>INT_MAX||g.experts>INT_MAX||
       g.k%block_values(g.type))return RAD_E_SHAPE;
    g.row_bytes=(g.k/block_values(g.type))*block_bytes(g.type);
    if(g.n>INT64_MAX/g.row_bytes||g.r>int64_t(INT_MAX)*4/g.n)return RAD_E_SHAPE;
    *out=g;return RAD_OK;
}
inline gguf::Format gguf_type(Type t) {
    switch(t) {
    case Type::iq2_xxs:return gguf::Format::iq2_xxs;case Type::iq2_xs:return gguf::Format::iq2_xs;
    case Type::iq2_s:return gguf::Format::iq2_s;case Type::iq3_xxs:return gguf::Format::iq3_xxs;
    case Type::q5_k:return gguf::Format::q5_k;case Type::q8_0:return gguf::Format::q8_0;
    default:return static_cast<gguf::Format>(999);
    }
}
inline int packed_plan(const RadParam* p,int np,int operand,const RadEncoding* enc,const int* sel,
                        const RadTensor* planes,int n,gguf::Layout* result) {
    if(operand!=1)return RAD_E_INVAL;
    // Both schemas include M/N/K. Layout uses the same format and rank-local weight geometry.
    const bool routed=rad_param_find(p,np,"n_expert")!=nullptr;
    Geometry g;int status=geometry(p,np,routed,&g,true);if(status!=RAD_OK)return status;
    status=gguf::detail::abi_plan(gguf_type(g.type),enc,sel,planes,n,result);
    if(status!=RAD_OK)return status;
    return result->rows==g.n&&result->cols==g.k?RAD_OK:RAD_E_SHAPE;
}
inline int layout(const RadParam* p,int np,int operand,const RadEncoding* enc,const int* sel,
                   const RadTensor* planes,int n,RadLayout* out) {
    if(!out)return RAD_E_INVAL;
    gguf::Layout l;int status=packed_plan(p,np,operand,enc,sel,planes,n,&l);
    if(status!=RAD_OK)return status;
    Geometry g;const bool routed=rad_param_find(p,np,"n_expert")!=nullptr;
    if(geometry(p,np,routed,&g,true)!=RAD_OK)return RAD_E_SHAPE;
    RadLayout r{};r.tag=runtime_tag(g.type);r.dtype=storage_dtype(g.type);r.rank=2;r.shape[0]=l.rows;r.shape[1]=l.row_bytes;
    r.align=256;r.bytes=l.bytes;*out=r;return RAD_OK;
}
inline int relayout(const RadParam* p,int np,int operand,const RadEncoding* enc,const int* sel,
                     const RadTensor* planes,int n,void* dst,int64_t bytes) {
    gguf::Layout l;const int status=packed_plan(p,np,operand,enc,sel,planes,n,&l);
    if(status!=RAD_OK)return status;
    Geometry g;const bool routed=rad_param_find(p,np,"n_expert")!=nullptr;
    if(geometry(p,np,routed,&g,true)!=RAD_OK)return RAD_E_SHAPE;
#define DS41_COPY(f) case Type::f:return gguf::abi_relayout<gguf::Format::f>(p,np,operand,enc,sel,planes,n,dst,bytes)
    switch(g.type) {
        DS41_COPY(iq2_xxs);DS41_COPY(iq2_xs);DS41_COPY(iq2_s);DS41_COPY(iq3_xxs);DS41_COPY(q5_k);DS41_COPY(q8_0);
    }
#undef DS41_COPY
    return RAD_E_DTYPE;
}
inline int unrelayout(const RadParam* p,int np,int operand,const RadEncoding* enc,const int* sel,
                       const void* src,int64_t bytes,const RadTensor* planes,int n) {
    gguf::Layout l;const int status=packed_plan(p,np,operand,enc,sel,planes,n,&l);
    if(status!=RAD_OK)return status;
    Geometry g;const bool routed=rad_param_find(p,np,"n_expert")!=nullptr;
    if(geometry(p,np,routed,&g,true)!=RAD_OK)return RAD_E_SHAPE;
#define DS41_COPY(f) case Type::f:return gguf::abi_unrelayout<gguf::Format::f>(p,np,operand,enc,sel,src,bytes,planes,n)
    switch(g.type) {
        DS41_COPY(iq2_xxs);DS41_COPY(iq2_xs);DS41_COPY(iq2_s);DS41_COPY(iq3_xxs);DS41_COPY(q5_k);DS41_COPY(q8_0);
    }
#undef DS41_COPY
    return RAD_E_DTYPE;
}
inline int matrix(const RadTensor& t,uint32_t dtype,int64_t rows,int64_t cols,int64_t* bytes=nullptr) {
    if(!t.data)return RAD_E_INVAL;
    if(t.dtype!=dtype)return RAD_E_DTYPE;
    if(t.rank!=2||t.shape[0]!=rows||t.shape[1]!=cols||rows<=0||cols<=0)return RAD_E_SHAPE;
    if(t.stride[1]!=1||t.stride[0]<cols)return RAD_E_STRIDE;
    const int element_bytes=rad_dtype_bits(dtype)/8;
    if(element_bytes<=0||uintptr_t(t.data)%unsigned(element_bytes))return RAD_E_ALIGN;
    if(rows-1>(INT64_MAX-cols)/t.stride[0])return RAD_E_SHAPE;
    const int64_t elements=(rows-1)*t.stride[0]+cols;
    if(elements>INT64_MAX/element_bytes)return RAD_E_SHAPE;
    if(bytes)*bytes=elements*element_bytes;
    return RAD_OK;
}
inline int vector(const RadTensor& t,uint32_t dtype,int64_t count,int64_t* bytes=nullptr) {
    if(!t.data)return RAD_E_INVAL;
    if(t.dtype!=dtype)return RAD_E_DTYPE;
    if(t.rank!=1||t.shape[0]!=count||count<=0)return RAD_E_SHAPE;
    if(t.stride[0]!=1)return RAD_E_STRIDE;
    const int width=rad_dtype_bits(dtype)/8;
    if(width<=0||uintptr_t(t.data)%unsigned(width))return RAD_E_ALIGN;
    if(count>INT64_MAX/width)return RAD_E_SHAPE;
    if(bytes)*bytes=count*width;
    return RAD_OK;
}
inline bool alias(const void* a,int64_t na,const void* b,int64_t nb) {
    const auto x=uintptr_t(a),y=uintptr_t(b);
    return na>0&&nb>0&&(x<=y?uint64_t(y-x)<uint64_t(na):uint64_t(x-y)<uint64_t(nb));
}
inline int validate(const RadArgs* args,bool routed,Launch* out) {
    if(!args||!out||!args->t||args->n_t!=(routed?6:3))return RAD_E_INVAL;
    Launch l;int status=geometry(args->p,args->n_p,routed,&l.g);if(status!=RAD_OK)return status;
    const auto* t=args->t;
    if(t[0].rank!=2)return RAD_E_SHAPE;
    l.input_rows=t[0].shape[0];l.jobs=routed?t[2].shape[0]:l.input_rows;
    if(l.input_rows<=0||l.input_rows>l.g.m||l.jobs<=0||l.jobs>l.g.r)return RAD_E_SHAPE;
    int64_t spans[6]{};
    status=matrix(t[0],l.g.a_dtype,l.input_rows,l.g.k,&spans[0]);if(status!=RAD_OK)return status;
    status=matrix(t[routed?4:2],l.g.y_dtype,l.jobs,l.g.n,&spans[routed?4:2]);
    if(status!=RAD_OK)return status;
    l.a_pitch=t[0].stride[0];l.y_pitch=t[routed?4:2].stride[0];
    if(routed) {
        if(!t[1].data)return RAD_E_INVAL;
        const bool logical=t[1].dtype==storage_dtype(l.g.type);
        if(!logical&&t[1].dtype!=RAD_U8)return RAD_E_DTYPE;
        const int64_t descriptor_cols=logical?l.g.k:l.g.row_bytes;
        if(t[1].rank!=3||t[1].shape[0]!=l.g.experts||t[1].shape[1]!=l.g.n||
           t[1].shape[2]!=descriptor_cols)return RAD_E_SHAPE;
        if(t[1].stride[0]!=0||t[1].stride[1]!=descriptor_cols||t[1].stride[2]!=1)return RAD_E_STRIDE;
        if(uintptr_t(t[1].data)%alignof(void*))return RAD_E_ALIGN;
        spans[1]=l.g.experts*int64_t(sizeof(void*));
        for(int i:{2,3,5}) {status=vector(t[i],RAD_I32,l.jobs,&spans[i]);if(status!=RAD_OK)return status;}
    } else {
        if(!t[1].data)return RAD_E_INVAL;
        const bool logical=t[1].dtype==storage_dtype(l.g.type);
        if(!logical&&t[1].dtype!=RAD_U8)return RAD_E_DTYPE;
        const int64_t descriptor_cols=logical?l.g.k:l.g.row_bytes;
        if(t[1].rank!=2||t[1].shape[0]!=l.g.n||t[1].shape[1]!=descriptor_cols)return RAD_E_SHAPE;
        if(t[1].stride[0]!=descriptor_cols||t[1].stride[1]!=1)return RAD_E_STRIDE;
        spans[1]=l.g.n*l.g.row_bytes;
    }
    const int y=routed?4:2;
    for(int i=0;i<args->n_t;++i)if(i!=y&&alias(t[y].data,spans[y],t[i].data,spans[i]))return RAD_E_INVAL;
    if(routed)for(int i=0;i<5;++i)if(alias(t[5].data,spans[5],t[i].data,spans[i]))return RAD_E_INVAL;
    *out=l;return RAD_OK;
}
inline int shape(const RadParam* p,int np,int operand,RadOpdDesc* out,bool routed) {
    if(!out||operand<0||operand>=(routed?6:3))return RAD_E_INVAL;
    Geometry g;const int status=geometry(p,np,routed,&g,true);if(status!=RAD_OK)return status;
    RadOpdDesc d{};d.idx_const=-1;
    if(operand==0) {d.dtype=g.a_dtype;d.rank=2;d.shape[0]=g.m;d.shape[1]=g.k;}
    else if(operand==1) {
        // A cold normal F32 draw is not this exact canonical encoding. Recorded importer
        // fixtures/dedicated checks must provide it; never advertise a false weight dtype.
        return RAD_E_UNSUPPORTED;
    } else if(operand==(routed?4:2)) {d.dtype=g.y_dtype;d.rank=2;d.shape[0]=g.r;d.shape[1]=g.n;}
    else {
        d.dtype=RAD_I32;d.rank=1;d.shape[0]=g.r;
        if(operand==2||operand==3) {d.fill=RAD_FILL_INDEX;d.idx_max=operand==2?g.experts:g.m;}
    }
    *out=d;return RAD_OK;
}
} // namespace rad::ds41::packed
