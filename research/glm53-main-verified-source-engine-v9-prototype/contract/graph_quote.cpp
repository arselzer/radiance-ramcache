#include "graph_quote.h"
#include "build/startup.h"
#include "format/share.h"
#include "place/planner.h"
#include "sample/sampler.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <new>
#include <string_view>
#include <vector>
#include <utility>
namespace glm53::tiny_startup {
namespace {
struct Failure {int status;const char* why;};
void need(bool b,const char* why){if(!b)throw Failure{RAD_E_SHAPE,why};}
void explain(std::string* out,const char* s)noexcept{if(out)try{*out=s;}catch(...){}}
// This is a bounded specification, not a WeightInfo clone. Runtime strings,
// layout hooks, encoding and selected rectangles always come from actual Core.
struct View {
    std::string name,source,physical0,physical1;
    uint32_t dtype=0,rank=0;int64_t rows=0,cols=0;
    int layer=-1,expert=-1,slot=0,selected=0,shard=RAD_SHARD_NONE,access=RAD_ACCESS_PER_TOKEN;
};
std::vector<View> views(){
    std::vector<View> out;out.reserve(155);
    auto ordinary=[&](std::string n,std::string physical,uint32_t dt,int64_t rows,int64_t cols,int rank,int l=-1,int shard=RAD_SHARD_NONE,int access=RAD_ACCESS_PER_TOKEN){
        out.push_back({n,n,std::move(physical),{},dt,uint32_t(rank),rows,cols,l,-1,0,0,shard,access});};
    ordinary("tiny.token_embd.weight","tiny.embed.weight",RAD_BF16,V,H,2,-1,RAD_SHARD_ROW,RAD_ACCESS_VOCAB);
    ordinary("tiny.output_norm.weight","tiny.final_norm.weight",RAD_F32,H,1,1);
    ordinary("tiny.output.weight","tiny.head.weight",RAD_BF16,V,H,2,-1,RAD_SHARD_ROW,RAD_ACCESS_VOCAB);
    for(int l=0;l<L;++l){const auto block="blk."+std::to_string(l),src="tiny.layers."+std::to_string(l);
        ordinary(block+".ffn_norm.weight",src+".ffn_norm.weight",RAD_F32,H,1,1,l);
        ordinary(block+".glm_router.weight",src+".router.weight",RAD_F32,E,H,2,l);
        ordinary(block+".glm_router.correction",src+".router.correction",RAD_F32,E,1,1,l);
        for(int e=0;e<E;++e)for(int slot=0;slot<2;++slot){const auto stem=block+(slot?".ffn_down_exps.":".ffn_gate_up_exps.")+std::to_string(e);
            const auto source=stem+".weight",physical=src+".experts."+std::to_string(e)+".";
            for(int scale=0;scale<2;++scale){View v;v.name=stem+(scale?".scale":".weight");v.source=source;
                v.physical0=physical+(slot?"down":"gate")+"_proj.weight";if(!slot)v.physical1=physical+"up_proj.weight";
                v.dtype=scale?RAD_E8M0:RAD_FP4E2M1;v.rank=2;v.rows=slot?H:2*F;v.cols=scale?H/32:H;
                v.layer=l;v.expert=e;v.slot=slot;v.selected=scale;v.shard=slot?RAD_SHARD_COL:RAD_SHARD_ROW;v.access=RAD_ACCESS_CONDITIONAL;out.push_back(std::move(v));}
        }
    }return out;
}
bool bounded_equal(const char* p,const std::string& s){if(!p)return false;const auto n=::strnlen(p,513);return n<=512&&n==s.size()&&!std::memcmp(p,s.data(),n);}
bool allowed_op(std::string_view op){
    constexpr std::array names={"embed_lookup","rmsnorm_quant_fp8","gemm_nt_f32","router_sigmoid_topk",
      "moe_scatter","moe_scatter_inv","moe_gemm_q","swiglu_clamped_quant_fp8","swiglu_clamped",
      "quant_act_fp8","silu_mul","silu_mul_quant_fp8","moe_gather","moe_gather_inv",
      "router_sigmoid_topk_scatter","add","rmsnorm","gather_rows","logits_gemm",
      "sample_penalties","sample_dry","sample_ngram_ban","sample_temp","sample_mask","sample_topk",
      "sample_argmax","sample_topp","sample_minp","sample_typical","sample_xtc","sample_pick","sample_logprobs"};
    return std::find(names.begin(),names.end(),op)!=names.end();
}
void exact_maps(const rad::Program& p,const std::vector<View>& expected){
    need(p.name_map.size()==91,"exact direct logical name-map count");
    std::array<bool,155> seen{};size_t n=0;
    for(const auto& m:p.name_map){size_t i=0;for(;i<expected.size();++i)if(expected[i].selected==0&&bounded_equal(m.declared,expected[i].source))break;
        need(i<expected.size()&&!seen[i],"unknown or duplicate logical name map");seen[i]=true;const auto& v=expected[i];const bool gu=!v.physical1.empty();
        need(m.mode==(gu?RAD_MAP_CONCAT:RAD_MAP_COPY)&&m.n_src==(gu?2:1)&&m.concat_dim==0&&bounded_equal(m.src[0],v.physical0)&&(!gu||bounded_equal(m.src[1],v.physical1)),"exact physical map source order/mode");
        for(int s=0;s<8;++s)need(m.src_index[s]==-1&&m.src_rows[s]==0&&(s<m.n_src||m.src[s]==nullptr),"exact full physical map slice extent");++n;
    }need(n==91,"exact logical source map coverage");
}
} // namespace
int quote_declared_graph(const ColdSourceSeal& seal,const rad::Program& p,const rad::Plan& plan,
                        const BoundSources& sources,GraphQuote* out,std::string* why){
    try{
        need(out&&seal.permits_zero_head_metadata(p.meta,p.ctx),"sealed no-attention declaration profile");
        need(sources.owner_identity()==&seal.owner(),"bound resolver owns another Checkpoint");
        const int audit=seal.owner().audit(why);if(audit<0)return audit;auto& opened=seal.owner().checkpoint();
        need(p.weights.size()==156&&p.weights.size()<=512&&p.ops.size()<=256&&p.buffers.size()<=256&&plan.units.size()<=128,"closed graph declaration counts");
        need(p.requires_permanent_model_failure()&&p.kv_groups.size()==1&&p.expert_route_maps.size()==L&&p.drafter.kind==RAD_DRAFT_NONE&&
             !p.drafter_shared_kv&&!p.encoder.modalities&&!p.audio_encoder.modalities&&!p.is_pass_driven()&&
             p.lora_sites.empty()&&p.lora_pools.empty()&&p.lora_t.empty()&&p.lora_roots.empty()&&p.lora_aliases.empty()&&
             p.scratch_regions==1&&p.lane1_w0==-1&&p.lane1_w1==-1&&p.logits_buf>0&&!p.pooled_buf,"unexpected attention/KV/draft/media/adapter/lane role");
        need(p.arena_bytes>=0&&p.host_arena_bytes>=0&&p.scratch_bytes>=0&&p.arena_bytes+p.host_arena_bytes+p.scratch_bytes<=int64_t(4*MiB),"actual arena and scratch quote");
        need(plan.rank==0&&plan.world_size==1&&plan.strategy==rad::Strategy::AllVram&&!plan.dynamic&&plan.fits()&&!plan.native_inventory&&
             plan.weight.size()==p.weights.size()&&plan.unit_of_weight.size()==p.weights.size()&&plan.unit_of_weight[0]==-1,"matching pure metadata plan");
        const auto expected=views();need(expected.size()==155,"bounded view specification");exact_maps(p,expected);
        GraphQuote q;q.physical_tensors=seal.source().physical_tensors;q.weights=p.weights.size();q.units=plan.units.size();q.actual_arena_scratch_bytes=p.arena_bytes+p.host_arena_bytes+p.scratch_bytes;
        std::array<bool,155> seen{};std::array<bool,32> experts{};std::array<bool,155> source_seen{};
        for(size_t i=1;i<p.ops.size();++i){const auto& op=p.ops[i];need(op.op.size()<=512&&allowed_op(op.op),"operation outside closed FFN/sampler allowlist");q.bands+=op.bands.size();}
        need(q.bands<=512,"closed actual selected-band cap");
        struct SamplerBuf {const char* name;uint32_t dtype;int64_t columns;int kind;};
        const SamplerBuf sampler_buffers[]={{"params",RAD_U32,int64_t(sizeof(rad::DeviceSampleParams)/4),RAD_BUF_PERSIST},{"params_late",RAD_U32,int64_t(sizeof(rad::DeviceSampleParams)/4),RAD_BUF_PERSIST},
            {"history",RAD_I32,2048,RAD_BUF_PERSIST},{"dry_breakers",RAD_U8,2048,RAD_BUF_PERSIST},{"ngram",RAD_I32,2,RAD_BUF_PERSIST},
            {"ngram_wl",RAD_U32,10,RAD_BUF_PERSIST},{"mask",RAD_U32,10,RAD_BUF_PERSIST},{"cand_idx",RAD_I32,1024,RAD_BUF_TRANSIENT},
            {"cand_val",RAD_F32,1024,RAD_BUF_TRANSIENT},{"logprobs",RAD_F32,44,RAD_BUF_PERSIST}};
        for(const auto& expected_buffer:sampler_buffers){const auto name=std::string("sampler.")+expected_buffer.name;const auto b=std::find_if(p.buffers.begin(),p.buffers.end(),[&](const auto& x){return x.name==name;});
            need(b!=p.buffers.end()&&std::count_if(p.buffers.begin(),p.buffers.end(),[&](const auto& x){return x.name==name;})==1&&b->decl.dtype==expected_buffer.dtype&&b->decl.rank==2&&
                 b->decl.shape[0]==1&&b->decl.shape[1]==expected_buffer.columns&&b->decl.kind==expected_buffer.kind&&b->decl.domain==RAD_DOMAIN_DEVICE,"actual same-Builder sampler declaration binding");}
        // This gate quotes graph wiring only; builder still must resolve all actual
        // HOST holes. Optional fallback declaration misses alone are not execution.
        need(p.device_holes(true).empty()&&sources.refused().empty(),"unresolved actual HOST graph or checkpoint source");
        for(size_t h=1;h<p.weights.size();++h){const auto& w=p.weights[h];size_t i=0;for(;i<expected.size()&&expected[i].name!=w.name;++i){}
            need(i<expected.size()&&!seen[i],"unexpected or duplicate declared weight name");seen[i]=true;const auto& v=expected[i];
            need(w.source==v.source&&w.name.size()<=512&&w.source.size()<=512&&w.layout_tag.size()<=512&&w.decl.dtype==v.dtype&&w.decl.rank==v.rank&&
                 w.decl.shape[0]==v.rows&&(v.rank==1||w.decl.shape[1]==v.cols)&&w.decl.access==v.access&&w.decl.shard==v.shard&&
                 w.decl.group.layer==v.layer&&w.decl.group.expert==v.expert&&w.decl.group.slot==v.slot&&!w.decl.optional&&
                 w.enc_known&&w.n_sel==1&&w.sel[0]==v.selected&&w.identity&&!w.widen&&w.layout_tag.empty()&&
                 w.stored_dtype==v.dtype&&w.stored_rank==v.rank&&w.stored_shape[0]==v.rows&&(v.rank==1||w.stored_shape[1]==v.cols)&&
                 w.align==RAD_ALIGN_UNIT&&w.stored_bytes>0,
                 "exact declared group/slot/name/source/type/selection/layout");
            for(uint32_t d=v.rank;d<RAD_MAX_RANK;++d)need(w.decl.shape[d]==0&&w.stored_shape[d]==0,"unused declaration/stored extent");
            const auto* cw=sources.find(w.source);need(cw&&cw->name==v.source&&cw->trivial&&rad_enc_equal(&cw->enc,&w.enc)&&cw->rank==v.rank,"actual cached logical import encoding");
            if(v.expert>=0){const size_t count=v.slot?1:2;need(cw->pieces.size()==count&&cw->scales.size()==count,"bound cached physical plane count");
                for(size_t piece=0;piece<count;++piece){const auto& name=piece?v.physical1:v.physical0;const rad::SrcTensor *code=nullptr,*scale=nullptr;
                    need(opened.import_piece(name,&code,&scale,why)==RAD_OK&&code&&scale,"bound retained Checkpoint import planes");
                    const auto* raw_code=opened.find(name);const auto* raw_scale=opened.find(name+"_scale");
                    need(cw->pieces[piece]==code&&cw->scales[piece]==scale&&code->name==name&&scale->name==name+"_scale"&&
                         raw_code&&raw_scale&&code->data==raw_code->data&&scale->data==raw_scale->data&&code->bytes==raw_code->bytes&&scale->bytes==raw_scale->bytes,
                         "bound cached code/scale NAME ORDER object and physical byte span");}
            }else{need(cw->pieces.size()==1&&cw->pieces[0]==opened.find(v.physical0)&&cw->pieces[0]&&cw->pieces[0]->name==v.physical0,
                        "bound ordinary cached Checkpoint object and named byte span");}
            if(!source_seen[i]&&v.selected==0){++q.logical_sources;source_seen[i]=true;}
            const auto uid=plan.unit_of_weight[h];need(uid>=0&&size_t(uid)<plan.units.size()&&plan.weight[h].unit==uid,"actual unit map coverage");
            const auto& u=plan.units[size_t(uid)];need(u.layer==v.layer&&u.expert==v.expert&&u.weights.size()==u.offset.size()&&
                 std::count(u.weights.begin(),u.weights.end(),rad_weight(h))==1,"exact actual group member mapping");
            const auto member=size_t(std::find(u.weights.begin(),u.weights.end(),rad_weight(h))-u.weights.begin());
            need(w.align>0&&u.offset[member]>=0&&u.offset[member]%w.align==0&&u.offset[member]<=u.bytes&&w.stored_bytes<=u.bytes-u.offset[member],"actual layout offsets within group");
            rad::PlaneShare share[RAD_ENC_MAX_PLANES];need(rad::weight_share(w,cw->enc,cw->rank,cw->shape,1,0,share,why)==RAD_OK,"actual selected full-share rectangles");
            need(share[0].dtype==v.dtype&&share[0].dense_bytes()==w.stored_bytes&&w.logical_bytes==w.stored_bytes,"actual identity stored-byte contract");q.selected_plane_rects+=share[0].rects.size();
            q.all_stored_bytes+=uint64_t(w.stored_bytes);
            if(v.expert>=0){++q.expert_views;const auto e=size_t(v.layer*E+v.expert);experts[e]=true;
                need(cw->rank==2&&cw->shape[0]==(v.slot?H:2*F)&&cw->shape[1]==H&&cw->pieces.size()==size_t(v.slot?1:2)&&
                     cw->scales.size()==cw->pieces.size()&&!cw->imported&&w.enc.n_planes==2,"actual byte-trivial Quark gate/up/down sources");
                for(size_t j=0;j<cw->pieces.size();++j){const auto* codes=cw->pieces[j];const auto* scale=cw->scales[j];
                    need(codes&&scale&&codes->rad_dtype==RAD_FP4E2M1&&scale->rad_dtype==RAD_E8M0&&codes->shape==std::vector<int64_t>({H,H})&&
                         scale->shape==std::vector<int64_t>({H,H/32})&&codes->bytes==8192&&scale->bytes==512&&codes->data&&scale->data,"actual Quark code/scale source plane metadata");}
                need(w.shard_lo==0&&w.shard_hi==(v.slot&&v.selected?H/32:F)&&w.n_part_spans==0,"actual explicit full-share span coordinates");++q.explicit_full_spans;
                need(w.decl.n_row_parts==(v.slot?0:2),"actual gate/up row-part count");
                if(!v.slot){need(w.decl.row_parts[0]==F&&w.decl.row_parts[1]==F,"actual gate/up part extents");++q.gate_up_rowparts;}
            }else{++q.nonexpert_views;need(cw->pieces.size()==1&&cw->scales.size()==1&&cw->scales[0]==nullptr&&!cw->imported&&w.shard_lo==0&&w.shard_hi==0&&w.n_part_spans==0&&w.decl.n_row_parts==0,"ordinary physical full-source metadata");}
            if(v.expert<0||v.expert==0){++q.ordinary_views;q.ordinary_stored_bytes+=uint64_t(w.stored_bytes);q.ordinary_page_upper+=(uint64_t(w.stored_bytes)+4095)/4096*4096;}
        }
        q.expert_units=std::count(experts.begin(),experts.end(),true);
        // Actual Planner contract: EACH model-level weight owns a distinct unit.
        // Parent diagnostic measured3model+8layer+32expert=43, not a model fusion.
        std::array<bool,L> layer_units_seen{};
        for(const auto& u:plan.units){if(u.layer<0){need(u.layer==-1&&u.expert==-1&&u.weights.size()==1,"exact single-member model unit");++q.model_units;}
            else if(u.expert<0){need(u.layer<L&&u.expert==-1&&u.weights.size()==3&&!layer_units_seen[size_t(u.layer)],"exact three-member nonexpert layer unit");layer_units_seen[size_t(u.layer)]=true;++q.layer_units;}
            else need(u.layer<L&&u.expert<E&&u.weights.size()==4,"exact four-member expert unit");}
        std::array<bool,L> route_seen{};
        for(const auto& route:p.expert_route_maps){const auto& d=route.decl;
            need(d.layer>=0&&d.layer<L&&!route_seen[size_t(d.layer)]&&d.canonical_experts==E&&d.local_experts==E&&d.expert_base==0&&d.regular_experts==E&&d.classes==2&&
                 d.max_rows==R&&d.top_k==K&&d.n_entries==E&&route.entries.size()==E&&d.gu>0&&d.gu<p.ops.size()&&p.ops[d.gu].op=="moe_gemm_q","actual current-route declaration");route_seen[size_t(d.layer)]=true;
            const auto prefix="tiny.layer."+std::to_string(d.layer)+".";
            for(const auto [handle,suffix]:std::array<std::pair<rad_buf,const char*>,4>{{{d.counts,"count"},{d.offsets,"offset"},{d.sorted,"sorted"},{d.ids,"ids"}}})
                need(handle&&handle<p.buffers.size()&&p.buffers[handle].name==prefix+suffix&&p.buffers[handle].decl.dtype==RAD_I32,"actual current-route buffer binding");
            for(int e=0;e<E;++e){const auto& entry=route.entries[size_t(e)];need(entry.canonical==e&&entry.slot==e&&entry.class_index==e%2&&!entry.protected_expert,"actual current-route expert ordering");
                for(int member=0;member<4;++member){const auto h=entry.members[member];need(h&&h<p.weights.size(),"actual current-route member range");const auto& w=p.weights[h];
                    need(w.decl.group.layer==d.layer&&w.decl.group.expert==e&&w.decl.group.slot==member/2&&w.sel[0]==member%2,"actual current-route member identity");}
            }
        }
        need(std::all_of(seen.begin(),seen.end(),[](bool b){return b;})&&q.expert_views==128&&q.nonexpert_views==27&&q.logical_sources==91&&q.expert_units==32&&q.units==43&&q.model_units==3&&q.layer_units==8&&
             q.explicit_full_spans==128&&q.gate_up_rowparts==64&&q.all_stored_bytes==1020544&&q.ordinary_views==59&&q.ordinary_stored_bytes==393856&&q.ordinary_page_upper==528384,
             "actual W/source/unit/span/ordinary-byte topology differs from reviewed budget");
        const int st=rad::NativeWeightAuthority::inspect(p.weights,plan.unit_of_weight,plan.units,4*MiB,&q.authority,why);if(st<0)return st;
        need(q.authority.weights==q.weights&&q.authority.geometry_size==sizeof(rad::WeightInfo)&&q.authority.admissible&&q.authority.construction_peak_bytes<=4*MiB,"actual compact metadata precharge");
        const int final_audit=seal.owner().audit(why);if(final_audit<0)return final_audit;
        q.engine_eligible=false;*out=q;return RAD_OK;
    }catch(const Failure& e){explain(why,e.why);return e.status;}catch(const std::bad_alloc&){return RAD_E_NOMEM;}catch(...){return RAD_E_STATE;}
}
} // namespace glm53::tiny_startup
