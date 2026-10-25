// Explicit actual checkpoint metadata / optional complete-data prewalk only.
// Existing bounded path, plugin and JSON helpers are reused byte-for-byte.
#define main preserved_stock_metadata_main
#include "../glm53-verified-main-source-prototype/stock/stock_plan.cpp"
#undef main
#include "build/glm_main_ordinary_loader.h"
#include "build/glm_main_source_data/main_bundle_reader.h"
#include "build/tiny_component_quotes.h"
#include "place/native_inventory.h"
#include "runtime/ctx.h"
#include "runtime/cache_prepare4_session.h"
#include "sample/sampler.h"
#include <atomic>
#include <cstdlib>
#include <functional>
namespace {
std::atomic<uint64_t> backend_allocations{0},payload_reads{0};
size_t component_checks=0,component_refusals=0;
void checked(bool b,const std::string& why){++component_checks;need(b,why);}
struct Actual {
    std::unique_ptr<Checkpoint> checkpoint;
    std::unique_ptr<CheckpointSources> sources;
    LoadedPlugins plugins;
    // Programs borrow RadModelMeta's string/KV pointers. This owner is stable
    // on Actual's heap and destroys AFTER programs (reverse member order).
    MetaOwn declaration_meta;
    std::array<Program,4> programs;
    std::array<Sampler,4> samplers;
    std::array<TinySamplerQuote,4> sampler_quotes;
    std::array<TinyCtxQuote,4> context_quotes;
};
SamplerConfig sampler_config(int rank){SamplerConfig c;c.n_vocab=154880;c.n_vocab_local=38720;c.max_seqs=1;
    c.rank=rank;c.world_size=4;c.vocab_off=int64_t(rank)*38720;return c;}
CtxDesc context_descriptor(Program& p){CtxDesc d;d.program=&p;d.rank=p.ctx.rank;d.world_size=4;d.device=0;
    d.record_passes=false;d.profile_ops=false;d.expert_prepare_limits.max_rows=1;d.expert_prepare_limits.max_experts=288;
    d.expert_prepare_limits.max_top_k=8;d.expert_prepare_limits.max_maps=42;d.expert_prepare_limits.max_entries=12096;return d;}
void context_guard_diagnostic(const CtxDesc& d){
    // Read-only witness BEFORE the original quote. No normalization, quote retry,
    // Runtime/owner construction or acceptance change. Fixed metadata/log caps.
    const auto& p=*d.program;const auto& limits=d.expert_prepare_limits;
    Json report{{"schema","glm53-actual-context-quote-guard-diagnostic-v1"},
        {"rank",d.rank},{"world_size",d.world_size},{"device",d.device},
        {"max_tok",p.ctx.max_tok},{"max_seqs",p.ctx.max_seqs},{"max_ctx",p.ctx.max_ctx},{"max_spec",p.ctx.max_spec},
        {"is_draft",p.ctx.is_draft},{"requires_GLM_fail_stop",p.requires_glm_main_tp4_cache_fail_stop()},
        {"requires_PermanentV1",p.requires_permanent_model_failure()},
        {"record_passes",d.record_passes},{"profile_ops",d.profile_ops},{"tiny_retirement",d.tiny_retirement},
        {"bound_KV_count",d.kv.size()},{"has_stream",d.stream!=nullptr},{"has_prepare_owner",d.expert_prepare_owner!=nullptr},
        {"weights",p.weights.size()},{"buffers",p.buffers.size()},{"ops",p.ops.size()},{"KV_group_count",p.kv_groups.size()},
        {"arena_bytes",p.arena_bytes},{"arena_fixed_bytes",p.arena_fixed_bytes},{"arena_stage_offset",p.arena_stage_off},
        {"arena_stage_bytes",p.arena_stage_bytes},{"host_arena_bytes",p.host_arena_bytes},
        {"scratch_bytes",p.scratch_bytes},{"scratch_regions",p.scratch_regions},{"scratch_bytes_lane1",p.scratch_bytes_lane1},
        {"lane1_first_op",p.lane1_w0},{"lane1_last_op",p.lane1_w1},
        {"expert_prepare_limits",{{"max_rows",limits.max_rows},{"max_experts",limits.max_experts},
            {"max_top_k",limits.max_top_k},{"max_maps",limits.max_maps},{"max_entries",limits.max_entries},
            {"max_readback_bytes",limits.max_readback_bytes},{"max_metadata_bytes",limits.max_metadata_bytes}}},
        {"route_maps",p.expert_route_maps.size()},{"KV_geometry_scope","actual declared group fields; no runtime pool binding or carving"},{"KV_groups",Json::array()},
        {"first_unsupported_schema",nullptr},{"first_unsupported_row",nullptr},{"first_scratch_violation",nullptr}};
    for(const char* key:{"RADIANCE_OP_ORACLE","RADIANCE_OP_REPEAT","RADIANCE_DEBUG_ROUTING"}){
        const auto* value=std::getenv(key);report["nonempty_environment_guards"][key]=bool(value&&*value);}
    const size_t groups=std::min<size_t>(p.kv_groups.size(),128);
    for(size_t g=0;g<groups;++g){const auto& group=p.kv_groups[g];const auto& decl=group.decl;
        std::vector<int> layers(group.layers.begin(),group.layers.begin()+std::min<size_t>(group.layers.size(),46));
        report["KV_groups"].push_back({{"group",g},{"name",group.name.substr(0,128)},
            {"kind",decl.kind},{"dtype",decl.dtype},{"window",decl.window},{"n_head_kv",decl.n_head_kv},
            {"head_dim",decl.head_dim},{"state_dim",{decl.state_dim[0],decl.state_dim[1]}},{"conv_width",decl.conv_width},
            {"block_size",group.block_size},{"bytes_per_block",group.bytes_per_block},{"bytes_per_state",group.bytes_per_state},
            {"zero",group.zero},{"replicated",group.replicated},{"drafter",group.drafter},
            {"layers",std::move(layers)},{"layer_count",group.layers.size()},{"layers_truncated",group.layers.size()>46},
            {"supported_kind_in_current_quote",!g||decl.kind==RAD_KV_FULL||decl.kind==RAD_KV_WINDOW||decl.kind==RAD_KV_LINEAR||decl.kind==RAD_KV_CONV},
            {"head_dim_exceeds_current_4096_cap",decl.head_dim>4096}});}
    report["KV_groups_truncated"]=p.kv_groups.size()>groups;
    uint64_t bands=0,rows=0,init_rows=0,collective_rows=0;bool truncated=false;
    for(size_t oi=0;oi<p.ops.size()&&oi<50000;++oi){const auto& op=p.ops[oi];
        if(oi&&report["first_unsupported_schema"].is_null()&&(!op.schema||op.schema->n_operands<0||
           op.schema->n_operands>rad::MAX_OPERANDS||(op.schema->n_operands&&!op.schema->operands)||
           op.weights.size()!=op.weight_opd.size()||op.weights.size()>100000))
            report["first_unsupported_schema"]={{"op_index",oi},{"op",op.op.substr(0,128)},
                {"has_schema",op.schema!=nullptr},{"n_operands",op.schema?op.schema->n_operands:-1},
                {"has_operands",op.schema&&op.schema->operands},{"weights",op.weights.size()},{"weight_operands",op.weight_opd.size()}};
        for(size_t bi=0;bi<op.bands.size();++bi){if(bands>=100000){truncated=true;break;}++bands;
            for(size_t domain=0;domain<RAD_N_DOMAINS;++domain){const auto& resolved=op.bands[bi].dom[domain];
                if(resolved.scratch_bytes<0||resolved.scratch_bytes>p.scratch_bytes)
                    if(report["first_scratch_violation"].is_null())report["first_scratch_violation"]={{"op_index",oi},{"op",op.op.substr(0,128)},
                        {"band",bi},{"domain",domain},{"scratch_bytes",resolved.scratch_bytes}};
                if(!resolved.row)continue;
                ++rows;
                const auto* info=resolved.row->info;init_rows+=bool(info&&info->init);collective_rows+=resolved.row->collective;
                if(report["first_unsupported_row"].is_null()&&(!info||resolved.row->deferred_status))
                    report["first_unsupported_row"]={{"op_index",oi},{"op",op.op.substr(0,128)},{"band",bi},{"domain",domain},
                        {"plugin",resolved.row->plugin.substr(0,128)},{"has_info",info!=nullptr},{"deferred_status",resolved.row->deferred_status!=nullptr},
                        {"collective",resolved.row->collective},{"releases_link",resolved.row->releases_link}};
            }}
        if(truncated)break;
    }
    report["scanned_bands"]=bands;report["resolved_rows"]=rows;report["rows_with_init"]=init_rows;
    report["collective_rows"]=collective_rows;report["op_scan_truncated"]=truncated||p.ops.size()>50000;
    report["diagnostic_only_no_quote_or_runtime_called"]=true;
    const auto raw=report.dump();need(raw.size()<=65536,"bounded Context guard diagnostic64KiB");
    std::cerr<<"CTX_QUOTE_DIAGNOSTIC "<<raw<<'\n';std::cerr.flush();
}
std::unique_ptr<Actual> declare_actual(const std::string& checkpoint,const std::string& home){
    checked(!device_is_host()&&std::string(device_backend_name())=="aql","explicit actual AQL metadata mode");
    RadDeviceProps props{};good(rad_dev_props(0,&props),"metadata eligibility card");
    checked(!std::strcmp(props.arch,"gfx1201"),"pinned actual9070XT reader/layout eligibility");
    auto a=std::make_unique<Actual>();a->checkpoint=std::make_unique<Checkpoint>();good(a->checkpoint->open(checkpoint),"same actual checkpoint map/header owner");
    auto& registry=rad_tools_registry();registry.set_code_device(0);registry.set_no_device_code(false);registry.set_device_fact(RAD_FACT_P2P,0);
    good(rad_tools_load_plugins(home,{},a->checkpoint->meta().arch,a->checkpoint->meta().quant,&a->plugins,
         a->checkpoint->imports()!=Checkpoint::Import::None),"exclusive cache architecture plugins");
    checked(a->plugins.arch_plugin=="glm5next_cache","HOME must exclusively select actual glm5next_cache, never relabel stock graph");
    a->checkpoint->set_importers(&registry);
    a->declaration_meta=a->checkpoint->meta();auto& meta=a->declaration_meta;
    if(a->checkpoint->imports()!=Checkpoint::Import::None)meta.quant=a->plugins.arch_quant;
    meta.finish(); // republish ALL pointers into the retained owner, including non-imported input
    checked(meta.m.n_vocab==154880,"actual pinned GLM vocabulary");
    a->sources=std::make_unique<CheckpointSources>(*a->checkpoint);
    uint64_t pinned_upper=0;
    for(int rank=0;rank<4;++rank){good(Sampler::glm_main_startup_quote(sampler_config(rank),&a->sampler_quotes[size_t(rank)]),"pure predeclare Sampler reserve");
        pinned_upper+=a->sampler_quotes[size_t(rank)].storage.pinned_page_bytes;}
    checked(pinned_upper<1u<<20,"named four-Sampler pinned staging upper before declaration");
    const auto before=tracked_allocations();const auto instances=registry.n_instances();
    for(int rank=0;rank<4;++rank){auto bc=context(rank,512,1);bc.tp_wire_min_bytes=0;
        good(rad_tools_declare(a->plugins,meta.m,bc,&a->programs[size_t(rank)],
            [&,rank](RadBuilder* b,int64_t width){
                checked(width==38720,"actual contiguous per-rank logits width");
                good(a->samplers[size_t(rank)].declare(b,sampler_config(rank)),"real Sampler declaration/stage_alloc");
                RadBufDecl out{};out.dtype=RAD_I32;out.rank=1;out.shape[0]=1;out.kind=RAD_BUF_PERSIST;
                return rad_decl_buffer(b,"sampler.tokens",&out)?RAD_OK:RAD_E_STATE;
            },nullptr,a->sources->fn(),false,nullptr),"actual declaration, no instance initialization");
        auto& p=a->programs[size_t(rank)];good(glm_main_tp4_cache_program_profile(p),"actual42-layer/parity/current-route topology");
        checked(p.device_holes(false).empty(),"actual DEVICE selection complete");
        good(rad_gate_reference_kernels(p,false,"component metadata"),"no hidden reference execution fallback");
        checked(p.scratch_regions==2&&p.scratch_bytes_lane1==0&&p.lane1_w0==-1&&p.lane1_w1==-1,
            "genuine actual shared-Moe two-region declaration, no active second-lane window");
        auto d=context_descriptor(p);const auto alloc=backend_allocations.load(),reads=payload_reads.load();
        if(rank==0)context_guard_diagnostic(d);
        good(Ctx::glm_main_startup_quote(d,&a->context_quotes[size_t(rank)]),"actual private Context storage quote");
        checked(backend_allocations.load()==alloc&&payload_reads.load()==reads,"Ctx quote itself is pure metadata");
        TinySamplerQuote member;good(a->samplers[size_t(rank)].glm_main_startup_quote(&member),"postdeclare member quote");
        checked(member.storage.pinned_bytes==a->sampler_quotes[size_t(rank)].storage.pinned_bytes&&
                member.candidates==a->sampler_quotes[size_t(rank)].candidates,"predeclare/postdeclare Sampler clamp closure");
    }
    checked(a->sources->refused().empty(),"actual source import/encoding coverage");
    const auto after=tracked_allocations();checked(after[0]==before[0]&&after[3]==before[3],"no tracked model DEVICE/mapped buffer allocation");
    checked(after[1]>=before[1]&&uint64_t(after[1]-before[1])<=pinned_upper,"only named bounded Sampler pinned staging");
    checked(registry.n_instances()==instances,"no kernel instances or collective initialization");return a;
}
void metadata_owner_lifetime(const Actual& a){
    const auto& own=a.declaration_meta;const auto& m=own.m;
    checked(m.arch_id==own.arch.c_str()&&m.name==own.name.c_str()&&m.quant==own.quant.c_str(),
            "retained MetaOwn string pointer identity after declaration helper return");
    const std::string counts=" (n_kv="+std::to_string(m.n_kv)+", keys="+std::to_string(own.keys.size())+
        ", vals="+std::to_string(own.vals.size())+", key_pointers="+std::to_string(own.kp.size())+
        ", value_pointers="+std::to_string(own.vp.size())+")";
    checked(own.keys.size()==own.vals.size()&&own.keys.size()==own.kp.size()&&own.keys.size()==own.vp.size(),
            "retained MetaOwn exact KV vector counts after declaration helper return"+counts);
    checked(m.n_kv>=0&&size_t(m.n_kv)==own.keys.size(),
            "retained MetaOwn nonnegative exact metadata KV count after declaration helper return"+counts);
    checked(m.kv_key==own.kp.data()&&m.kv_val==own.vp.data(),
            "retained MetaOwn exact KV pointer arrays after declaration helper return"+counts);
    for(size_t i=0;i<own.keys.size();++i)
        checked(own.kp[i]==own.keys[i].c_str()&&own.vp[i]==own.vals[i].c_str(),"retained MetaOwn KV string owner");
    for(const auto& p:a.programs){
        // Compare owners BEFORE dereferencing these borrowed Program pointers.
        checked(p.meta.arch_id==m.arch_id&&p.meta.name==m.name&&p.meta.quant==m.quant&&
                p.meta.n_kv==m.n_kv&&p.meta.kv_key==m.kv_key&&p.meta.kv_val==m.kv_val,
                "Program metadata borrows exact retained Actual owner");
        checked(!std::strcmp(p.meta.arch_id,"glm5next"),"owned actual architecture identity after return");
        good(glm_main_tp4_cache_program_profile(p),"post-return actual GLM profile with retained metadata owner");
    }
}
template<class F>void bad_context(Program& p,F mutation){
    auto d=context_descriptor(p);TinyCtxQuote q;std::memset(&q,0x6b,sizeof q);const auto prior=q;
    const auto alloc=backend_allocations.load(),reads=payload_reads.load();
    mutation(d,[&]{checked(Ctx::glm_main_startup_quote(d,&q)<0,"actual Ctx metadata mutation refused");});
    checked(!std::memcmp(&q,&prior,sizeof q),"actual Ctx refusal preserved output");
    checked(backend_allocations.load()==alloc&&payload_reads.load()==reads,"Ctx refusal had backend/payload side effects");++component_refusals;
}
void context_refusals(Actual& a){auto& p=a.programs[0];
    bad_context(p,[](auto& d,auto run){d.record_passes=true;run();});
    bad_context(p,[](auto& d,auto run){d.world_size=3;run();});
    bad_context(p,[](auto& d,auto run){d.tiny_retirement=true;run();});
    bad_context(p,[](auto& d,auto run){const auto old=d.program->ctx.max_tok;d.program->ctx.max_tok=2;run();d.program->ctx.max_tok=old;});
    bad_context(p,[](auto& d,auto run){const auto old=d.program->ctx.max_ctx;d.program->ctx.max_ctx=2049;run();d.program->ctx.max_ctx=old;});
    bad_context(p,[](auto& d,auto run){auto& x=d.program->expert_route_maps[0].entries[0].class_index;const auto old=x;x=1;run();x=old;});
    bad_context(p,[](auto& d,auto run){auto& x=d.program->scratch_regions;const auto old=x;x=3;run();x=old;});
    bad_context(p,[](auto& d,auto run){auto& x=d.program->buffers[1].arena_offset;const auto old=x;x=3ll<<30;run();x=old;});
    bad_context(p,[](auto& d,auto run){auto& x=d.program->weights[1].stored_bytes;const auto old=x;x=(2ll<<30)+1;run();x=old;});
    checked(p.kv_groups.size()>1,"actual GLM has real state/cache groups");
    bad_context(p,[](auto& d,auto run){auto& x=d.program->kv_groups[1].decl.head_dim;const auto old=x;x=-1;run();x=old;});
    // Locate the ACTUAL declared tuple; never normalize/rebuild it for a positive.
    size_t conv=p.kv_groups.size();
    for(size_t g=1;g<p.kv_groups.size();++g)if(p.kv_groups[g].decl.kind==RAD_KV_CONV&&p.kv_groups[g].decl.head_dim==6144){
        checked(conv==p.kv_groups.size(),"single actual GLM6144 convolution group");conv=g;}
    checked(conv<p.kv_groups.size(),"actual GLM6144 convolution group exists");
    const auto& actual=p.kv_groups[conv].decl;
    checked(actual.dtype==RAD_BF16&&actual.n_head_kv==1&&actual.conv_width==4&&actual.window==0&&
            actual.state_dim[0]==0&&actual.state_dim[1]==0,"genuine exact CONV6144 tuple retained");
    const std::array<std::function<void(RadKVGroupDecl&)>,10> mutations{
        [](auto& x){x.kind=RAD_KV_LINEAR;},[](auto& x){x.dtype=RAD_F32;},
        [](auto& x){x.n_head_kv=2;},[](auto& x){x.head_dim=6143;},[](auto& x){x.head_dim=6145;},
        [](auto& x){x.conv_width=3;},[](auto& x){x.conv_width=5;},[](auto& x){x.window=1;},
        [](auto& x){x.state_dim[0]=1;},[](auto& x){x.state_dim[1]=1;}};
    for(const auto& mutate:mutations)bad_context(p,[&](auto& d,auto run){
        auto& decl=d.program->kv_groups[conv].decl;
        struct Restore{RadKVGroupDecl& target;RadKVGroupDecl prior;~Restore(){target=prior;}} restore{decl,decl};
        mutate(decl);run();});
}
Plan metadata_plan(Program& p,int rank){Config c=config(512,1);c.placement="all_vram";c.vram_weights_mib=0;c.host_pool_mib=0;
    PlannerInput input{&p,&c,rank,4,nullptr,0,false,false};Plan result;good(Planner(input).plan(&result),"uncapped metadata layout, not physical fit");return result;}
Json native_metadata_diagnostic(Actual& a){
    metadata_owner_lifetime(a);
    Json result=Json::array();
    for(int rank=0;rank<4;++rank){auto& p=a.programs[size_t(rank)];const auto alloc=backend_allocations.load(),reads=payload_reads.load();
        auto plan=metadata_plan(p,rank);
        Json row{{"rank",rank},{"program_world",p.ctx.world_size},{"plan_world",plan.world_size},
            {"plan_rank",plan.rank},{"strategy",strategy_name(plan.strategy)},{"dynamic",plan.dynamic},
            {"program_profile",glm_main_tp4_cache_program_profile(p)},{"units",plan.units.size()},
            {"classes",plan.classes.size()},{"first_static_refusal",nullptr},{"static_refusal_count",0},
            {"first_reader_refusal",nullptr},{"reader_refusal_count",0},{"no_affirmative_reader_count",0},
            {"layout_callback_status_counts",Json::object()},{"tier_counts",Json::object()},
            {"native_inventory_bind_called",false},{"complete_Reader_or_source_scan_called",false}};
        size_t static_bad=0,reader_bad=0;std::vector<uint8_t> claimed(p.weights.size(),0);
        for(size_t ui=0;ui<plan.units.size();++ui){const auto& u=plan.units[ui];const auto tier=tier_name(u.tier);
            auto& count=row["tier_counts"][tier];count=count.is_number_unsigned()?count.get<uint64_t>()+1:uint64_t(1);
            if(u.tier==Tier::VRAM&&!u.movable&&u.slab_slot<0&&u.bytes>0)continue;
            ++static_bad;
            if(!row["first_static_refusal"].is_null())continue;
            Json members=Json::array();for(size_t i=0;i<u.weights.size()&&i<8;++i){const auto h=u.weights[i];
                if(!h||h>=p.weights.size()){members.push_back({{"handle",h},{"outside",true}});continue;}
                const auto& w=p.weights[h];members.push_back({{"handle",h},{"name",w.name.substr(0,128)},
                    {"source",w.source.substr(0,128)},{"optional",w.decl.optional},{"stored_bytes",w.stored_bytes},
                    {"logical_bytes",w.logical_bytes},{"identity",w.identity},{"stored_dtype",w.stored_dtype}});}
            row["first_static_refusal"]={{"unit",ui},{"layer",u.layer},{"expert",u.expert},{"tier",tier},
                {"site",int(u.site)},{"bytes",u.bytes},{"slot_bytes",u.slot_bytes},{"movable",u.movable},
                {"slab_slot",u.slab_slot},{"slab_class",u.slab_class},{"host_forced",u.host_forced},
                {"reason",std::string(u.reason?u.reason:"",u.reason?strnlen(u.reason,160):0)},{"member_count",u.weights.size()},{"members",std::move(members)}};
        }
        for(size_t oi=0;oi<p.ops.size();++oi){const auto& op=p.ops[oi];
            for(size_t wi=0;wi<op.weights.size();++wi){const auto h=op.weights[wi];
                checked(h<p.weights.size(),"diagnostic actual selected weight handle");if(!h)continue;
                const auto& w=p.weights[h];if(w.decl.group.expert<0)continue;
                const int operand=wi<op.weight_opd.size()&&op.weight_opd[wi]>=0?op.weight_opd[wi]:int(wi);
                for(size_t bi=0;bi<op.bands.size();++bi){const auto& resolved=op.bands[bi].dom[RAD_DOMAIN_DEVICE];
                    int status=RAD_E_UNSUPPORTED;bool missing=!resolved.row||!resolved.row->info;
                    if(!missing){RadTensor planes[RAD_ENC_MAX_PLANES]{};weight_planes(w,planes);RadLayout layout{};
                        const auto* info=resolved.row->info;status=info->layout?info->layout(resolved.geom.params(),
                            resolved.geom.n_params(),operand,&w.enc,w.sel,planes,w.n_sel,&layout):RAD_E_UNSUPPORTED;}
                    auto& count=row["layout_callback_status_counts"][std::to_string(status)];
                    count=count.is_number_unsigned()?count.get<uint64_t>()+1:uint64_t(1);
                    if(!missing&&status==RAD_LAYOUT_ANY)continue;
                    if(!missing&&status==RAD_E_UNSUPPORTED){claimed[h]=1;continue;}
                    ++reader_bad;
                    if(row["first_reader_refusal"].is_null())row["first_reader_refusal"]={{"op",op.op.substr(0,128)},
                        {"op_index",oi},{"band",bi},{"handle",h},{"weight",w.name.substr(0,128)},
                        {"operand",operand},{"missing_DEVICE_reader",missing},{"layout_status",status}};
                }
            }
        }
        size_t unclaimed=0;for(size_t h=1;h<p.weights.size();++h)if(p.weights[h].decl.group.expert>=0&&!claimed[h])++unclaimed;
        row["static_refusal_count"]=static_bad;row["reader_refusal_count"]=reader_bad;row["no_affirmative_reader_count"]=unclaimed;
        checked(backend_allocations.load()==alloc&&payload_reads.load()==reads,"native metadata diagnostic has no backend/payload side effects");
        result.push_back(std::move(row));
    }
    return {{"schema","glm53-native-authority-metadata-diagnostic-v1"},{"rank_reports",std::move(result)},
        {"Program_or_Plan_normalized",false},{"NativeInventory_or_Session_capability_granted",false}};
}
uint64_t id(const char* s){const std::string text=s;checked(!text.empty()&&text.size()<=20&&std::all_of(text.begin(),text.end(),[](unsigned char c){return c>='0'&&c<='9';}),"bounded trusted bundle ID");const auto value=std::stoull(text);checked(value,"nonzero trusted ID");return value;}
Json complete_prewalk(Actual& a,const std::string& bundle,const std::string& proof,uint64_t expected){
    namespace data=rad::glm_main_data;
    std::array<Plan,4> metadata,final;
    std::array<std::shared_ptr<const data::VerifiedRankMain>,4> ranks;
    std::array<std::shared_ptr<const NativeInventory>,4> inventories;
    data::Request request{bundle,proof,expected};data::ReaderLimits reader_limits;
    std::shared_ptr<const data::VerifiedBundle> verified;std::string why;
    const auto opened=data::VerifiedBundle::open(request,*a.checkpoint,reader_limits,&verified,&why);
    checked(opened==data::Error::ok,"REAL complete-data verification: "+why);
    metadata_owner_lifetime(a); // source verification never shortens the declaration owner's lifetime
    uint64_t source_cut_bytes=0;
    for(size_t rank=0;rank<4;++rank){auto& p=a.programs[rank];metadata[rank]=metadata_plan(p,int(rank));
        checked(verified->bind_rank(*a.checkpoint,*a.sources,p,metadata[rank],&ranks[rank],&why)==data::Error::ok,"actual once/rank source/cut binding: "+why);
        checked(ranks[rank]->receipt().source_cut_bytes_hashed==40428896256ull,"exact actual per-rank selected source bytes hashed");
        source_cut_bytes+=ranks[rank]->receipt().source_cut_bytes_hashed;
        NativeInventoryLimits cap;cap.backend_profile=NativeBackendProfile::GlmMainTp4AqlFailStopV1;cap.max_rows=1;cap.top_k=8;
        cap.max_owned_units=12096;cap.max_classes=1;cap.max_regions=9;cap.max_device_bytes=cap.max_landing_bytes=32u<<20;
        cap.max_weights_bytes=8ull<<30;cap.max_metadata_bytes=128u<<20;cap.hot_bytes=0;cap.cold_bytes=8ull*3342336;
        cap.max_host_bytes=cap.cold_bytes+cap.max_landing_bytes+cap.max_metadata_bytes;
        why.clear();
        const int native_status=NativeInventory::bind(p,metadata[rank],ranks[rank]->descriptors(),ranks[rank]->owned(),{},cap,&inventories[rank],&why);
        good(native_status,"real source-bound native authority: "+why);
        Config c=config(512,1);c.placement="all_vram";c.vram_kv_mib=1;Config resolved;good(inventories[rank]->configure_pools(c,&resolved),"pure native footprint configuration");
        PlannerInput input{&p,&resolved,int(rank),4,nullptr,0,false,false};input.native_inventory=inventories[rank];good(Planner(input).plan(&final[rank]),"final Native plan, not allocated or Engine-ready");
    }
    std::array<GlmMainOrdinaryRankSpec,4> jobs;
    for(size_t rank=0;rank<4;++rank)jobs[rank]={&a.programs[rank],&final[rank],a.sources.get()};
    GlmMainOrdinaryLoaderLimits caps;GlmMainOrdinaryLoaderQuote q;std::unique_ptr<GlmMainOrdinaryLoader> loader;
    const auto alloc=backend_allocations.load(),reads=payload_reads.load();
    good(GlmMainOrdinaryLoader::prepare(jobs,caps,&loader,&q,&why),"actual full ordinary prewalk: "+why);
    checked(backend_allocations.load()==alloc&&payload_reads.load()==reads,"prewalk reads no payload/allocates no backend buffers");
    uint64_t weights=0;for(const auto& p:a.programs)weights+=p.weights.size()-1;
    checked(q.jobs+q.omitted_views+q.optional_missing==weights&&q.omitted_views==4ull*48384&&
            q.omitted_source_bytes==161715585024ull,"complete typed omissions and remaining job arithmetic");
    checked(q.buffer_bytes==q.pinned_page_bytes+q.column_bytes&&q.input_bytes&&q.ordinary_source_bytes&&q.ordinary_stored_bytes,"bounded actual input/output/column maxima");
    auto refusal=[&](const auto& limits){std::unique_ptr<GlmMainOrdinaryLoader> output;GlmMainOrdinaryLoaderQuote quoted;
        std::memset(&quoted,0x41,sizeof quoted);const auto old=quoted;const auto aa=backend_allocations.load(),rr=payload_reads.load();
        checked(GlmMainOrdinaryLoader::prepare(jobs,limits,&output,&quoted)<0,"actual cap-minus-one prewalk refused");
        checked(!output&&!std::memcmp(&old,&quoted,sizeof old),"prewalk refusal output atomic");
        checked(backend_allocations.load()==aa&&payload_reads.load()==rr,"cap refusal before payload/backend work");++component_refusals;};
    auto bad=caps;bad.max_metadata_bytes=q.metadata_requested-1;refusal(bad);
    bad=caps;bad.max_input_bytes=q.input_bytes-1;refusal(bad);
    if(q.output_bytes){bad=caps;bad.max_output_bytes=q.output_bytes-1;refusal(bad);}
    if(q.column_bytes){bad=caps;bad.max_column_bytes=q.column_bytes-1;refusal(bad);}
    bad=caps;bad.max_buffer_bytes=q.buffer_bytes-1;refusal(bad);
    auto saved=jobs[1].plan;jobs[1].plan=jobs[0].plan;refusal(caps);jobs[1].plan=saved;
    const auto owned=ranks[0]->owned()[0].unit;const auto handle=final[0].units[size_t(owned)].weights[0];
    auto* source=const_cast<CkptWeight*>(a.sources->find(a.programs[0].weights[handle].source));checked(source,"actual owned source cache object");
    const auto encoding=source->enc;source->enc.plane[0].dtype=RAD_DT_INVALID;refusal(caps);source->enc=encoding;
    return {{"complete_bundle_id",expected},{"verified_native_bytes",verified->receipt().native_bytes_verified},
        {"source_rank_cut_hash_bytes",source_cut_bytes},{"jobs",q.jobs},{"omitted_views",q.omitted_views},
        {"omitted_source_bytes",q.omitted_source_bytes},{"ordinary_source_bytes",q.ordinary_source_bytes},{"ordinary_stored_bytes",q.ordinary_stored_bytes},
        {"max_input_bytes",q.input_bytes},{"max_nonidentity_output_bytes",q.output_bytes},{"column_scratch_bytes",q.column_bytes},
        {"pinned_page_bytes",q.pinned_page_bytes},{"buffer_reservation",q.buffer_bytes},{"metadata_reservation",q.metadata_requested},
        {"identity_jobs",q.identity_jobs},{"widen_jobs",q.widen_jobs},{"relayout_jobs",q.relayout_jobs},{"loader_execute_called",false}};
}
}
extern "C" void* __real_rad_dev_alloc(int64_t,int);
extern "C" void* __wrap_rad_dev_alloc(int64_t n,int k){++backend_allocations;return __real_rad_dev_alloc(n,k);}
extern "C" ssize_t __real_pread(int,void*,size_t,off_t);
extern "C" ssize_t __wrap_pread(int fd,void* p,size_t n,off_t off){++payload_reads;return __real_pread(fd,p,n,off);}
int main(int argc,char** argv){try{
    const bool complete=argc==8&&!std::strcmp(argv[1],"--run-complete-prewalk");
    const bool native_diagnostic=argc==5&&!std::strcmp(argv[1],"--run-native-metadata-diagnostic");
    checked(complete||native_diagnostic||(argc==5&&!std::strcmp(argv[1],"--run-actual-metadata")),
        "usage: --run-actual-metadata CHECKPOINT EXCLUSIVE_HOME NEW_JSON | --run-native-metadata-diagnostic CHECKPOINT EXCLUSIVE_HOME NEW_JSON | --run-complete-prewalk CHECKPOINT EXCLUSIVE_HOME BUNDLE PROOF ID NEW_JSON");
    const std::string checkpoint=argv[2],home=argv[3],output=argv[complete?7:4];bounded(checkpoint);bounded(home);bounded(output);
    uint64_t expected_id=0;
    if(complete){bounded(argv[4]);bounded(argv[5]);expected_id=id(argv[6]);}
    const auto destination=fs::canonical(fs::path(output).parent_path())/fs::path(output).filename();
    const auto source=fs::canonical(checkpoint);checked(!fs::exists(destination),"new report only");
    for(auto at=destination;!at.empty();){checked(at!=source,"report cannot mutate source generation");if(complete)checked(at!=fs::canonical(argv[4]),"report cannot mutate native generation");const auto next=at.parent_path();if(next==at)break;at=next;}
    auto graphs=declare_actual(checkpoint,home);
    metadata_owner_lifetime(*graphs); // before any expensive complete Reader/source scan
    context_refusals(*graphs);
    Json output_json{{"schema","glm53-main-verified-components-v1"},{"passed",true},{"runtime_ready",false},
        {"execution_mode",native_diagnostic?"native_metadata_diagnostic":complete?"complete_prewalk":"actual_metadata"},
        {"runtime_SourceFactory_or_Session_or_Engine_admitted",false},{"actual_logical_rank_graphs",4},{"kernel_instances_or_launches",0},
        {"sampler_staging_is_real_bounded_pinned_allocation",true},{"loader_payload_execution",false},
        {"retained_declaration_metadata",{{"n_kv",graphs->declaration_meta.m.n_kv},
            {"keys",graphs->declaration_meta.keys.size()},{"values",graphs->declaration_meta.vals.size()},
            {"key_pointers",graphs->declaration_meta.kp.size()},{"value_pointers",graphs->declaration_meta.vp.size()},
            {"exact_owner_and_count_checks_passed",true}}}};
    Json ranks=Json::array();for(size_t i=0;i<4;++i){const auto& q=graphs->context_quotes[i];ranks.push_back({{"rank",i},
        {"weights",q.weights},{"buffers",q.buffers},{"ops",q.ops},{"WTAB_tables",q.tables},{"WTAB_entries",q.table_entries},
        {"actual_KV_groups",graphs->programs[i].kv_groups.size()},{"arena_bytes",q.arena_device},{"host_arena_bytes",q.arena_host},
        {"context_requested_upper",q.storage.page_and_heap_total},{"prepare_maps",q.prepare_maps},{"prepare_entries",q.prepare_entries},
        {"sampler_requests",graphs->sampler_quotes[i].storage.pinned_requests},{"sampler_pinned_bytes",graphs->sampler_quotes[i].storage.pinned_bytes}});}
    output_json["ranks"]=std::move(ranks);
    if(native_diagnostic)output_json["native_metadata_diagnostic"]=native_metadata_diagnostic(*graphs);
    if(complete){metadata_owner_lifetime(*graphs);output_json["actual_complete_data_prewalk"]=complete_prewalk(*graphs,argv[4],argv[5],expected_id);}
    output_json["checks"]=component_checks+1;output_json["refusals"]=component_refusals;
    checked(output_json.dump().size()<=65536,"component report64KiB cap");graphs.reset();rad_tools_registry().close();publish(destination.string(),output_json);
    std::cout<<"PASS actual graph components "<<component_checks<<" checks "<<component_refusals<<" refusals; no Engine/Session/model pass\n";return 0;
}catch(const std::exception& e){std::cerr<<"FAIL actual components checks="<<component_checks<<" refusals="<<component_refusals<<": "<<e.what()<<'\n';return 1;}}
