/* Small public-ABI declaration helpers, NOT a DS41 architecture/plugin.
 * Caller declares explicit source name maps FIRST and retains name strings through Builder
 * finalization. Real source encodings choose the existing six native packed operator formats.
 * No quantization, padding, gather, activation, collective, cast, cache or model lifecycle is
 * hidden here. All selected canonical planes form ONE logical [N,K] weight; the resolved DS41
 * row's layout assigns its opaque private stored dtype/tag. Do not substitute plain-U8 weights.
 * Output handles are published only on success. A Builder failure can leave partial declare
 * state: discard that Builder instead of attempting rollback/reusing failed handles. */
#pragma once
#include "config.h"
#include "../common/rad_gguf_repack_base.h"
#include "rad_runtime.h"
#include <array>
#include <climits>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace rad::deepseek41::packed {
namespace gg=rad::gguf;
struct Source {const char* name=nullptr;const char* source=nullptr;};
struct Description {gg::Format format{};RadEncoding encoding{};int64_t rows=0,cols=0,groups=1,group_rows=0;};
inline int fail(int code,const std::string& text,std::string* why){if(why)*why=text;return code;}
inline bool scalar(uint32_t dtype){return dtype==RAD_F32||dtype==RAD_F16||dtype==RAD_BF16;}
inline bool valid_name(const char* name){return name&&*name;}
inline bool context(const RadBuildCtx& c){return(c.world_size==1||c.world_size==4)&&c.rank>=0&&c.rank<c.world_size;}
inline int describe(RadBuilder* b,const Source& s,Description* out,std::string* why=nullptr){
    if(!b||!out||!valid_name(s.name)||!valid_name(s.source))return fail(RAD_E_INVAL,"explicit weight and mapped-source names required",why);
    RadEncoding encoding{};int64_t shape[RAD_MAX_RANK]{};uint32_t rank=0;
    const int rc=rad_weight_encoding(b,s.source,&encoding,shape,&rank);
    if(rc!=RAD_OK)return fail(rc,"source encoding unavailable after name map: "+std::string(s.source),why);
    Description d;bool found=false;
    for(const auto format:{gg::Format::iq2_xxs,gg::Format::iq2_xs,gg::Format::iq2_s,
                           gg::Format::iq3_xxs,gg::Format::q5_k,gg::Format::q8_0}){
        const auto expected=gg::canonical_encoding(format);
        // Compare the fixed representation FIRST; do not interpret untrusted unrelated metadata.
        if(rad_enc_equal(&encoding,&expected)){d.format=format;found=true;break;}
    }
    if(!found)return fail(RAD_E_DTYPE,"source is not one of the exact six native GGUF encodings: "+std::string(s.source),why);
    if(rank!=2&&rank!=3)return fail(RAD_E_SHAPE,"native source must be an explicit matrix or reviewed grouped matrix",why);
    for(uint32_t i=0;i<rank;++i)if(shape[i]<=0||shape[i]>INT_MAX)return fail(RAD_E_SHAPE,"invalid native source dimension",why);
    d.rows=shape[0];d.cols=shape[1];
    if(rank==3){d.groups=shape[0];d.group_rows=shape[1];if(shape[0]>INT64_MAX/shape[1])return fail(RAD_E_SHAPE,"grouped row overflow",why);d.rows=shape[0]*shape[1];d.cols=shape[2];}
    if(d.rows>INT_MAX||d.rows>(INT64_MAX-7)/64/d.cols||d.cols%gg::geometry(d.format)->values)
        return fail(RAD_E_SHAPE,"native source byte arithmetic or block width is invalid",why);
    d.encoding=encoding;*out=d;if(why)why->clear();return RAD_OK;
}
enum class DenseOwnership {replicated,reviewed_rows};
struct LinearSpec {
    Source weight;
    int64_t rows=0,cols=0,max_rows=0; // EXPECTED full source extents, not guessed from model name.
    int layer=-1,slot=0;
    uint32_t input_dtype=RAD_BF16,output_dtype=RAD_BF16;
    DenseOwnership ownership=DenseOwnership::replicated;
    int64_t source_groups=1; // Explicit grouped [groups,group_rows,K] flattening only.
    // ROW opt-in is a CALLER semantic assertion, not proof that gathers are implemented.
    bool full_input_and_output_reconstruction_reviewed=false;
    int64_t whole_output_unit_rows=0; // e.g512 for complete heads,1024 for output_a groups.
};
struct NativeLinear {
    rad_weight weight=0;rad_op op=0;Description source;
    int64_t rows=0,cols=0,max_rows=0,row_begin=0;
    bool row_sharded=false;
};
inline int declare_native_linear(RadBuilder* b,const RadBuildCtx& ctx,const LinearSpec& s,
                                  NativeLinear* out,std::string* why=nullptr){
    if(!out||!context(ctx)||s.rows<=0||s.cols<=0||s.max_rows<=0||s.max_rows>INT_MAX||s.layer< -1||s.slot<0||
       !scalar(s.input_dtype)||!scalar(s.output_dtype)||s.source_groups<=0)
        return fail(RAD_E_INVAL,"invalid explicit native linear declaration",why);
    NativeLinear result;int rc=describe(b,s.weight,&result.source,why);if(rc!=RAD_OK)return rc;
    const auto& source=result.source;
    if(source.rows!=s.rows||source.cols!=s.cols||source.groups!=s.source_groups)
        return fail(RAD_E_SHAPE,"native linear expected source/group geometry mismatch",why);
    result.rows=s.rows;result.cols=s.cols;result.max_rows=s.max_rows;
    if(s.ownership==DenseOwnership::reviewed_rows){
        const int64_t unit=s.whole_output_unit_rows;
        if(!s.full_input_and_output_reconstruction_reviewed||unit<=0||s.rows%ctx.world_size)
            return fail(RAD_E_UNSUPPORTED,"ROW requires explicit full-input/output reconstruction review and divisible rows",why);
        result.rows=s.rows/ctx.world_size;result.row_begin=ctx.rank*result.rows;result.row_sharded=ctx.world_size>1;
        if(result.rows%unit||result.row_begin%unit||(source.groups>1&&unit!=source.group_rows))
            return fail(RAD_E_SHAPE,"ROW slice cuts a caller-declared complete head/group unit",why);
    }else if(s.ownership!=DenseOwnership::replicated)return fail(RAD_E_UNSUPPORTED,"unreviewed dense shard policy",why);
    if(s.max_rows>int64_t(INT_MAX)*4/result.rows)
        return fail(RAD_E_SHAPE,"linear output capacity exceeds native kernel addressing",why);
    gg::Layout layout;rc=gg::plan(source.format,source.encoding,s.rows,s.cols,
        {result.row_begin,result.rows,0,s.cols},&layout,why);if(rc!=RAD_OK)return rc;
    if(!rad_op_declared(b,"ds41_gemm_packed"))return fail(RAD_E_NOKERNEL,"native linear schema is unavailable",why);
    RadWeightDecl d{};d.dtype=source.encoding.plane[0].dtype;d.rank=2;d.shape[0]=result.rows;d.shape[1]=s.cols;
    d.access=RAD_ACCESS_PER_TOKEN;d.shard=result.row_sharded?RAD_SHARD_ROW:RAD_SHARD_NONE;
    d.group.layer=s.layer;d.group.expert=-1;d.group.slot=s.slot;d.source=s.weight.source;
    result.weight=rad_decl_weight(b,s.weight.name,&d);
    if(!result.weight)return fail(RAD_E_STATE,"Builder refused all-plane native linear weight",why);
    if(result.row_sharded&&rad_weight_shard_span(b,result.weight,result.row_begin,result.row_begin+result.rows)!=RAD_OK)
        return fail(RAD_E_SHAPE,"Builder refused native linear ROW span",why);
    const RadParam p[]{RAD_RANGE("M",1,s.max_rows),RAD_INT("N",result.rows),RAD_INT("K",s.cols),
        RAD_STR("format",gg::geometry(source.format)->name),RAD_STR("a_dtype",rad_dtype_name(s.input_dtype)),
        RAD_STR("y_dtype",rad_dtype_name(s.output_dtype))};
    result.op=rad_decl_op(b,"ds41_gemm_packed",p,6,&result.weight,1);
    if(!result.op||!rad_op_resolved(b,result.op))return fail(RAD_E_NOKERNEL,"native linear row did not resolve",why);
    *out=result;if(why)why->clear();return RAD_OK;
}
struct ExpertSources {Source gate,up,down;};
struct RoutedSpec {
    int layer=-1;
    int64_t hidden=0,intermediate=0,max_tokens=0,topk=0;
    uint32_t input_dtype=RAD_BF16,activation_dtype=RAD_BF16;
    std::vector<ExpertSources> experts; // CANONICAL expert-ID order, explicit caller name maps.
};
struct RoutedProjection {
    std::vector<rad_weight> weights;rad_op op=0;gg::Format format{};
    int64_t rows=0,cols=0,input_capacity=0,route_capacity=0;
};
struct RoutedFFN {RoutedProjection gate,up,down;Partition partition;int layer=-1;};
inline int declare_routed_ffn(RadBuilder* b,const RadBuildCtx& ctx,const RoutedSpec& s,
                              RoutedFFN* out,std::string* why=nullptr){
    if(!b||!out||!context(ctx)||s.layer<0||s.hidden<=0||s.intermediate<=0||s.max_tokens<=0||
       s.max_tokens>INT_MAX||s.topk<=0||s.experts.empty()||s.experts.size()>4096||s.topk>int64_t(s.experts.size())||
       s.max_tokens>INT_MAX/s.topk||!scalar(s.input_dtype)||!scalar(s.activation_dtype))
        return fail(RAD_E_INVAL,"invalid explicit routed FFN declaration",why);
    RoutedFFN result;result.layer=s.layer;
    int rc=ffn_partition(s.intermediate,ctx.world_size,ctx.rank,s.layer,256,&result.partition,why);
    if(rc!=RAD_OK)return rc;
    const int64_t width=result.partition.width,lo=result.partition.offset,routes=s.max_tokens*s.topk;
    // Preflight ALL sources before ANY weight/op declaration. One WTAB per projection must
    // share a format/shape, while gate, up and down may use DIFFERENT native encodings.
    std::array<std::vector<Description>,3> descriptions;std::set<std::string> names;
    for(size_t expert=0;expert<s.experts.size();++expert){
        const auto& e=s.experts[expert];const Source sources[]{e.gate,e.up,e.down};
        for(int projection=0;projection<3;++projection){
            Description desc;rc=describe(b,sources[projection],&desc,why);if(rc!=RAD_OK)return rc;
            if(!names.insert(sources[projection].name).second)return fail(RAD_E_DUPLICATE,"duplicate declared FFN weight name",why);
            if(desc.groups!=1||desc.rows!=(projection==2?s.hidden:s.intermediate)||desc.cols!=(projection==2?s.intermediate:s.hidden))
                return fail(RAD_E_SHAPE,"routed source does not match explicit expert matrix geometry",why);
            if(!descriptions[projection].empty()&&descriptions[projection][0].format!=desc.format)
                return fail(RAD_E_DTYPE,"WTAB requires one native format within each projection",why);
            descriptions[projection].push_back(desc);
        }
    }
    RoutedProjection* projections[]{&result.gate,&result.up,&result.down};
    for(int projection=0;projection<3;++projection){
        auto& p=*projections[projection];p.format=descriptions[projection][0].format;
        p.rows=projection==2?s.hidden:width;p.cols=projection==2?width:s.hidden;
        p.input_capacity=projection==2?routes:s.max_tokens;p.route_capacity=routes;
        if(p.rows>INT_MAX||p.cols>INT_MAX||routes>int64_t(INT_MAX)*4/p.rows)
            return fail(RAD_E_SHAPE,"routed operator capacity exceeds native kernel addressing",why);
        gg::Layout layout;rc=gg::plan(p.format,descriptions[projection][0].encoding,
            projection==2?s.hidden:s.intermediate,projection==2?s.intermediate:s.hidden,
            projection==2?gg::Region{0,s.hidden,lo,width}:gg::Region{lo,width,0,s.hidden},&layout,why);
        if(rc!=RAD_OK)return rc;
    }
    if(!rad_op_declared(b,"ds41_routed_packed"))return fail(RAD_E_NOKERNEL,"native routed schema is unavailable",why);
    for(int projection=0;projection<3;++projection){
        auto& p=*projections[projection];
        for(size_t expert=0;expert<s.experts.size();++expert){
            const auto& e=s.experts[expert];const Source source=projection==0?e.gate:projection==1?e.up:e.down;
            RadWeightDecl d{};d.dtype=descriptions[projection][expert].encoding.plane[0].dtype;d.rank=2;
            d.shape[0]=p.rows;d.shape[1]=p.cols;d.access=RAD_ACCESS_CONDITIONAL;
            d.shard=ctx.world_size==1?RAD_SHARD_NONE:projection==2?RAD_SHARD_COL:RAD_SHARD_ROW;
            d.group.layer=s.layer;d.group.expert=int(expert);d.group.slot=projection;d.source=source.source;
            const auto w=rad_decl_weight(b,source.name,&d);
            if(!w)return fail(RAD_E_STATE,"Builder refused all-plane expert weight",why);
            if(ctx.world_size>1&&rad_weight_shard_span(b,w,lo,lo+width)!=RAD_OK)
                return fail(RAD_E_SHAPE,"Builder refused expert ROW/COL span",why);
            if(!p.weights.empty()&&w!=p.weights[0]+expert)return fail(RAD_E_STATE,"WTAB expert handles are not consecutive",why);
            p.weights.push_back(w);
        }
        const RadParam params[]{RAD_RANGE("M",1,p.input_capacity),RAD_INT("N",p.rows),RAD_INT("K",p.cols),
            RAD_INT("R",p.route_capacity),RAD_INT("n_expert",int64_t(s.experts.size())),
            RAD_STR("format",gg::geometry(p.format)->name),
            RAD_STR("a_dtype",rad_dtype_name(projection==2?s.activation_dtype:s.input_dtype)),
            RAD_STR("y_dtype",rad_dtype_name(projection==2?uint32_t(RAD_F32):s.activation_dtype))};
        p.op=rad_decl_op(b,"ds41_routed_packed",params,8,p.weights.data(),int(p.weights.size()));
        if(!p.op||!rad_op_resolved(b,p.op))return fail(RAD_E_NOKERNEL,"native routed projection row did not resolve",why);
    }
    *out=std::move(result);if(why)why->clear();return RAD_OK;
}
// Optional issue wiring only. Caller supplies descriptors/row slices and declares buffer
// liveness via rad_op_reads/writes. Gate/up use token row IDs; down MUST use identity job rows.
inline int issue_native_linear(RadCtx* c,const NativeLinear& p,RadOperand x,RadOperand y,int64_t rows){
    if(!c||!p.op||!p.weight||rows<=0||rows>p.max_rows)return RAD_E_INVAL;
    const RadOperand operands[]{x,RAD_W(p.weight),y};return rad_issue(c,p.op,operands,3,rows);
}
inline int issue_routed_projection(RadCtx* c,const RoutedProjection& p,RadOperand x,RadOperand expert_ids,
    RadOperand row_ids,RadOperand y,RadOperand status,int64_t input_rows){
    if(!c||!p.op||p.weights.empty()||input_rows<=0||input_rows>p.input_capacity)return RAD_E_INVAL;
    const RadOperand operands[]{x,RAD_WTAB(p.weights[0],int64_t(p.weights.size())),expert_ids,row_ids,y,status};
    return rad_issue(c,p.op,operands,6,input_rows);
}
inline constexpr const char* routed_completion_obligation=
    "Gate/up rounded outputs -> F32 clamp/SiLU/product/route-weight -> ONE activation cast -> "
    "down F32 partial PER EXPERT JOB -> exact F32 TP sum matching canonical token/expert job order -> "
    "cast EACH completed expert -> ascending expert-ID combine + completed shared expert. "
    "No gather/reduction/cast/combine is implemented by these declarations. Partial local expert "
    "sums cannot be combined/cast as a substitute. MTP/FP4 activation arithmetic is not implied.";
} // namespace rad::deepseek41::packed
