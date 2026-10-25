/* Checked HOST metadata only. No Builder mutation, payload verification, operator
 * registration, allocation of tensor data, projection or collective execution. */
#pragma once
#include "../common/rad_gguf_repack_base.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <string>

namespace rad::deepseek41::handoff {
inline constexpr int64_t max_tokens=65536,max_dimension=65536,max_groups=64;
enum class StorageLayout {plain_row_major,gguf_block_rows};
struct Source {
    // Caller-declared immutable asset/revision/hash identity; NOT verified bytes.
    std::string asset_identity,tensor_name;
    RadEncoding encoding{};
    uint32_t rank=0;
    std::array<int64_t,3> shape{}; // normalized [N,K] or [groups,N,K], NEVER GGUF order
    StorageLayout layout=StorageLayout::plain_row_major;
    uint64_t payload_bytes=0; // EXACT original plain/GGUF-block payload, no header/padding
};
struct MatrixSlice {
    uint32_t dtype=0;
    int64_t rows=0,cols=0,element_offset=0,row_stride=0;
    uint64_t byte_offset=0,span_bytes=0; // from slice start; includes inter-row gaps
};
struct Parameters {int64_t M=0,N=0,K=0;uint32_t input_dtype=0,compute_dtype=RAD_F32,output_dtype=0;};
enum class PlainRole {generic,c2_projection,router_logits,index_bf16};
struct PlainSpec {
    int64_t tokens=0,inputs=0,outputs=0;
    // Defaults intentionally keep C2/router projection results F32.
    uint32_t input_dtype=RAD_BF16,compute_dtype=RAD_F32,output_dtype=RAD_F32;
    PlainRole role=PlainRole::generic;
};
struct PlainPlan {
    Source source;
    Parameters params;
    PlainRole role=PlainRole::generic;
    uint32_t weight_storage_dtype=0;
    bool promote_input_to_f32=false,promote_weight_to_f32=false;
    MatrixSlice input,weight,output;
};
struct GroupedSpec {
    int64_t tokens=0,groups=0,inputs_per_group=0,outputs_per_group=0;
    int rank=0,world_size=1; // metadata ownership only; no collective is implied
    uint32_t input_dtype=RAD_BF16,compute_dtype=RAD_F32,output_dtype=RAD_BF16;
};
struct GroupSlice {
    int64_t global_group=0;
    MatrixSlice input,output; // each token's OWN group in full global token-major arrays
    gguf::Region logical_weight; // row slice of normalized flattened [G*N,K]
    uint64_t original_block_byte_offset=0,original_block_bytes=0;
    Parameters params;
};
struct GroupedPlan {
    Source source;
    bool packed=false;
    gguf::Format format{}; // meaningful only when packed=true
    uint32_t plain_weight_dtype=0;
    int64_t group_begin=0,group_count=0,global_groups=0;
    uint64_t full_input_bytes=0,full_output_bytes=0;
    std::array<GroupSlice,size_t(max_groups)> slices{}; // only group_count entries live
};
namespace detail {
inline int fail(int code,const char* text,std::string* why) noexcept {
    // Diagnostic storage must not turn a checked allocation refusal into an
    // escaping exception. Retaining the caller's output plan takes precedence.
    if(why)try{*why=text;}catch(const std::bad_alloc&){why->clear();}
    return code;
}
inline bool scalar(uint32_t dtype){return dtype==RAD_BF16||dtype==RAD_F32;}
inline uint64_t width(uint32_t dtype){return dtype==RAD_F32?4:dtype==RAD_BF16?2:0;}
inline bool multiply(uint64_t a,uint64_t b,uint64_t* out){
    if(b&&a>uint64_t(INT64_MAX)/b)return false;
    *out=a*b;return *out<=SIZE_MAX;
}
inline bool name(const std::string& x,size_t limit){return !x.empty()&&x.size()<=limit&&x.find('\0')==std::string::npos;}
inline int source(const Source& s,uint32_t rank,std::string* why){
    if(!name(s.asset_identity,256)||!name(s.tensor_name,512))return fail(RAD_E_INVAL,"bounded asset/tensor identity is required",why);
    if(s.rank!=rank)return fail(RAD_E_SHAPE,"source rank is not the explicit normalized projection rank",why);
    for(uint32_t i=0;i<rank;++i)if(s.shape[i]<=0||s.shape[i]>(rank==3&&i==0?max_groups:max_dimension))
        return fail(RAD_E_SHAPE,"source dimension is outside the bounded metadata profile",why);
    for(uint32_t i=rank;i<s.shape.size();++i)if(s.shape[i])return fail(RAD_E_SHAPE,"unused source dimensions must be zero",why);
    if(!s.payload_bytes||s.payload_bytes>uint64_t(INT64_MAX)||s.payload_bytes>SIZE_MAX)
        return fail(RAD_E_SHAPE,"source payload extent is invalid",why);
    return RAD_OK;
}
inline uint32_t plain_dtype(const RadEncoding& encoding){
    for(uint32_t dtype:{uint32_t(RAD_BF16),uint32_t(RAD_F32)}){
        const auto exact=rad_enc_plain(dtype);if(rad_enc_equal(&encoding,&exact))return dtype;
    }
    return 0;
}
inline int matrix(int64_t rows,int64_t cols,int64_t stride,int64_t offset,uint32_t dtype,
                  uint64_t total,MatrixSlice* out){
    if(rows<=0||cols<=0||stride<cols||offset<0||!scalar(dtype))return RAD_E_SHAPE;
    uint64_t skip=0,span=0,byte_offset=0;
    if(!multiply(uint64_t(rows-1),uint64_t(stride),&skip)||uint64_t(cols)>uint64_t(INT64_MAX)-skip||
       !multiply(skip+uint64_t(cols),width(dtype),&span)||!multiply(uint64_t(offset),width(dtype),&byte_offset)||
       byte_offset>total||span>total-byte_offset)return RAD_E_SHAPE;
    *out={dtype,rows,cols,offset,stride,byte_offset,span};return RAD_OK;
}
inline int extents(int64_t tokens,int64_t cols,uint32_t dtype,uint64_t* bytes){
    uint64_t values=0;
    return multiply(uint64_t(tokens),uint64_t(cols),&values)&&multiply(values,width(dtype),bytes)?RAD_OK:RAD_E_SHAPE;
}
} // namespace detail

inline int plan_plain(const Source& source,const PlainSpec& spec,PlainPlan* out,std::string* why=nullptr){
    if(!out)return detail::fail(RAD_E_INVAL,"missing plain projection plan",why);
    int rc=detail::source(source,2,why);if(rc!=RAD_OK)return rc;
    if(spec.tokens<=0||spec.tokens>max_tokens||spec.inputs!=source.shape[1]||spec.outputs!=source.shape[0])
        return detail::fail(RAD_E_SHAPE,"plain projection geometry does not match its explicit source",why);
    const uint32_t weight=detail::plain_dtype(source.encoding);
    if(source.layout!=StorageLayout::plain_row_major||!weight||!detail::scalar(spec.input_dtype)||
       !detail::scalar(spec.output_dtype)||spec.compute_dtype!=RAD_F32)return detail::fail(RAD_E_DTYPE,"plain BF16/F32 storage and F32 compute are required",why);
    switch(spec.role){
        case PlainRole::generic:break;
        case PlainRole::c2_projection:case PlainRole::router_logits:
            if(spec.output_dtype!=RAD_F32)return detail::fail(RAD_E_DTYPE,"C2/router projection output must stay F32",why);
            break;
        case PlainRole::index_bf16:
            if(spec.output_dtype!=RAD_BF16)return detail::fail(RAD_E_DTYPE,"explicit BF16 index projection boundary required",why);
            break;
        default:return detail::fail(RAD_E_INVAL,"unknown plain projection role",why);
    }
    uint64_t weights=0,input_bytes=0,output_bytes=0;
    if(detail::extents(spec.outputs,spec.inputs,weight,&weights)!=RAD_OK||weights!=source.payload_bytes||
       detail::extents(spec.tokens,spec.inputs,spec.input_dtype,&input_bytes)!=RAD_OK||
       detail::extents(spec.tokens,spec.outputs,spec.output_dtype,&output_bytes)!=RAD_OK)
        return detail::fail(RAD_E_SHAPE,"plain payload or activation byte extent mismatch",why);
    try{
        PlainPlan result;result.source=source;result.params={spec.tokens,spec.outputs,spec.inputs,spec.input_dtype,spec.compute_dtype,spec.output_dtype};
        result.role=spec.role;result.weight_storage_dtype=weight;result.promote_input_to_f32=spec.input_dtype!=RAD_F32;result.promote_weight_to_f32=weight!=RAD_F32;
        if((rc=detail::matrix(spec.tokens,spec.inputs,spec.inputs,0,spec.input_dtype,input_bytes,&result.input))!=RAD_OK||
           (rc=detail::matrix(spec.outputs,spec.inputs,spec.inputs,0,weight,weights,&result.weight))!=RAD_OK||
           (rc=detail::matrix(spec.tokens,spec.outputs,spec.outputs,0,spec.output_dtype,output_bytes,&result.output))!=RAD_OK)
            return detail::fail(rc,"plain activation slice overflows",why);
        *out=std::move(result);if(why)why->clear();return RAD_OK;
    }catch(const std::bad_alloc&){return detail::fail(RAD_E_NOMEM,"cannot retain projection source identity",why);}
}

inline int plan_grouped(const Source& source,const GroupedSpec& spec,GroupedPlan* out,std::string* why=nullptr){
    if(!out)return detail::fail(RAD_E_INVAL,"missing grouped projection plan",why);
    int rc=detail::source(source,3,why);if(rc!=RAD_OK)return rc;
    if(spec.tokens<=0||spec.tokens>max_tokens||spec.groups!=source.shape[0]||spec.outputs_per_group!=source.shape[1]||spec.inputs_per_group!=source.shape[2])
        return detail::fail(RAD_E_SHAPE,"grouped projection geometry does not match normalized [G,N,K] source",why);
    if((spec.world_size!=1&&spec.world_size!=4)||spec.rank<0||spec.rank>=spec.world_size||spec.groups%spec.world_size)
        return detail::fail(RAD_E_UNSUPPORTED,"only TP1/4 with complete equal group ownership is planned",why);
    if(!detail::scalar(spec.input_dtype)||!detail::scalar(spec.output_dtype)||spec.compute_dtype!=RAD_F32)
        return detail::fail(RAD_E_DTYPE,"grouped activations require BF16/F32 storage and F32 compute",why);
    const uint32_t plain=detail::plain_dtype(source.encoding);bool packed=false;gguf::Format format{};uint64_t group_bytes=0;
    uint64_t rows=0,input_cols=0,output_cols=0,input_bytes=0,output_bytes=0,total=0;
    if(!detail::multiply(uint64_t(spec.groups),uint64_t(spec.outputs_per_group),&rows)||
       !detail::multiply(uint64_t(spec.groups),uint64_t(spec.inputs_per_group),&input_cols))return RAD_E_SHAPE;
    output_cols=rows;
    if(source.layout==StorageLayout::plain_row_major&&plain){
        if(detail::extents(spec.outputs_per_group,spec.inputs_per_group,plain,&group_bytes)!=RAD_OK)return RAD_E_SHAPE;
    }else if(source.layout==StorageLayout::gguf_block_rows){
        bool found=false;
        for(auto f:{gguf::Format::iq2_xxs,gguf::Format::iq2_xs,gguf::Format::iq2_s,gguf::Format::iq3_xxs,gguf::Format::q5_k,gguf::Format::q8_0}){
            const auto exact=gguf::canonical_encoding(f);if(rad_enc_equal(&source.encoding,&exact)){format=f;found=true;break;}
        }
        if(!found)return detail::fail(RAD_E_DTYPE,"grouped encoding is not one of the exact six GGUF block forms",why);
        gguf::Layout layout;rc=gguf::plan(format,source.encoding,int64_t(rows),spec.inputs_per_group,
            {0,spec.outputs_per_group,0,spec.inputs_per_group},&layout);
        if(rc!=RAD_OK)return detail::fail(rc,"grouped original-block geometry is invalid",why);
        packed=true;group_bytes=uint64_t(layout.bytes);
    }else return detail::fail(RAD_E_DTYPE,"grouped source layout/storage is unsupported",why);
    if(!detail::multiply(group_bytes,uint64_t(spec.groups),&total)||total!=source.payload_bytes||
       detail::extents(spec.tokens,int64_t(input_cols),spec.input_dtype,&input_bytes)!=RAD_OK||
       detail::extents(spec.tokens,int64_t(output_cols),spec.output_dtype,&output_bytes)!=RAD_OK)
        return detail::fail(RAD_E_SHAPE,"grouped source/activation byte extents overflow or mismatch",why);
    try{
        GroupedPlan result;result.source=source;result.packed=packed;result.format=format;result.plain_weight_dtype=packed?0:plain;
        result.global_groups=spec.groups;result.group_count=spec.groups/spec.world_size;result.group_begin=spec.rank*result.group_count;
        result.full_input_bytes=input_bytes;result.full_output_bytes=output_bytes;
        for(int64_t local=0;local<result.group_count;++local){
            auto& slice=result.slices[size_t(local)];const int64_t group=result.group_begin+local;slice.global_group=group;
            if((rc=detail::matrix(spec.tokens,spec.inputs_per_group,int64_t(input_cols),group*spec.inputs_per_group,spec.input_dtype,input_bytes,&slice.input))!=RAD_OK||
               (rc=detail::matrix(spec.tokens,spec.outputs_per_group,int64_t(output_cols),group*spec.outputs_per_group,spec.output_dtype,output_bytes,&slice.output))!=RAD_OK)return detail::fail(rc,"grouped activation slice overflows",why);
            slice.logical_weight={group*spec.outputs_per_group,spec.outputs_per_group,0,spec.inputs_per_group};
            slice.original_block_byte_offset=uint64_t(group)*group_bytes;slice.original_block_bytes=group_bytes;
            slice.params={spec.tokens,spec.outputs_per_group,spec.inputs_per_group,spec.input_dtype,spec.compute_dtype,spec.output_dtype};
        }
        *out=std::move(result);if(why)why->clear();return RAD_OK;
    }catch(const std::bad_alloc&){return detail::fail(RAD_E_NOMEM,"cannot retain grouped source identity",why);}
}
} // namespace rad::deepseek41::handoff
