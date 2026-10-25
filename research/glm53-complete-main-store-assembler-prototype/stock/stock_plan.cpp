// Standalone planning only. No Engine, Pools, Ctx, sampler instance, queue,
// kernel init/launch, upload, conversion, model allocation or source payload copy.
#include "build/startup.h"
#include "device/device.h"
#include "format/checkpoint.h"
#include "format/encoding.h"
#include "mem/kv.h"
#include "place/planner.h"
#include "runtime/arena.h"
#include "sched/batch.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>

using Json=nlohmann::ordered_json;
namespace fs=std::filesystem;
using namespace rad;
namespace {
constexpr int world=4;
constexpr int64_t MiB=1ll<<20,GiB=1ll<<30;
constexpr int64_t sampler_margin=16*MiB;
struct Options {std::string checkpoint,home,output;int64_t ctx=512,batch=256;};
void need(bool b,const std::string& why){if(!b)throw std::runtime_error(why);}
void good(int rc,const std::string& why){need(rc==RAD_OK,why+" status="+std::to_string(rc));}
int64_t plus(int64_t a,int64_t b){need(a>=0&&b>=0&&a<=INT64_MAX-b,"nonnegative quote overflow");return a+b;}
int64_t round_mib(int64_t n){return plus(n,MiB-1)/MiB*MiB;}
int64_t meminfo(const char* key){std::ifstream in("/proc/meminfo");std::string name,unit;int64_t n=0;
    while(in>>name>>n>>unit){if(name==std::string(key)+":")return n*1024;}return -1;}
bool suffix(const std::string& s,const char* end){const size_t n=std::strlen(end);return s.size()>=n&&s.compare(s.size()-n,n,end)==0;}
std::array<int64_t,4> tracked_allocations(){std::array<int64_t,4> a{};for(int k=0;k<4;++k)rad_dev_alloc_totals(k,&a[size_t(k)],nullptr);return a;}
Json allocations(const std::array<int64_t,4>& a){return {{"device",a[0]},{"pinned",a[1]},{"host",a[2]},{"mapped",a[3]}};}
void bounded(const std::string& p){need(!p.empty()&&p.size()<=4096&&p.find('\0')==std::string::npos&&fs::path(p).is_absolute(),"absolute bounded local path required");}
Options options(int argc,char** argv){Options o;need(argc==8||argc==10||argc==12,
    "usage: stock_glm53_plan --run-device-metadata --checkpoint ABS_COMPLETE_CHECKPOINT --home ABS_SHIPPED_HOME --output ABS_NEW_JSON [--ctx 512|2048] [--batch-tokens 128|256]");
    need(!std::strcmp(argv[1],"--run-device-metadata"),"explicit device-metadata gate required");
    for(int i=2;i<argc;i+=2){need(i+1<argc,"missing option value");const std::string k=argv[i],v=argv[i+1];
        if(k=="--checkpoint")o.checkpoint=v;else if(k=="--home")o.home=v;else if(k=="--output")o.output=v;
        else if(k=="--ctx"){need(v=="512"||v=="2048","closed tiny context512 or2048");o.ctx=std::stoll(v);}
        else if(k=="--batch-tokens"){need(v=="128"||v=="256","closed batch128 or256");o.batch=std::stoll(v);}else need(false,"unknown option: "+k);}
    bounded(o.checkpoint);bounded(o.home);bounded(o.output);need(!fs::exists(o.output),"output must be a NEW path");return o;}
void publish(const std::string& path,const Json& j){const auto text=j.dump(2)+"\n";need(text.size()<=16u<<20,"report16MiB cap");
    const int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600);need(fd>=0,"new output open");
    bool complete=false;try{size_t at=0;while(at<text.size()){const auto n=::write(fd,text.data()+at,text.size()-at);if(n<0&&errno==EINTR)continue;need(n>0,"output write");at+=size_t(n);}need(!::fsync(fd),"output fsync");complete=true;}
    catch(...){::close(fd);::unlink(path.c_str());throw;}::close(fd);need(complete,"complete report");}
Config config(int64_t ctx,int64_t batch){Config c;c.tp=world;c.max_tok=batch;c.max_seqs=1;c.max_ctx=ctx;c.n_spec=0;
    c.tp_wire_exact=1;c.tp_attn_rows=0;c.p2p=RAD_P2P_OFF;c.kv_cache_dtype="bf16";c.state_cache_dtype="f32";
    c.mm_max_patches=0;c.mm_audio_frames=0;c.prefix_cache=false;c.prefix_cache_mib=0;c.checkpoint_slots=0;
    c.embedding_placement=c.embedding_resolved=TablePlacement::Vram;c.ngram_placement=TablePlacement::Disk;
    c.placement="expert_tiered";c.weights_disk_tier=false;c.deterministic=true;return c;}
RadBuildCtx context(int rank,int64_t ctx,int64_t batch){RadBuildCtx c{};c.rank=rank;c.world_size=world;c.max_tok=batch;c.max_seqs=1;c.max_ctx=ctx;
    c.max_spec=0;c.kv_dtype=RAD_BF16;c.state_dtype=0;c.scope="";c.tp_wire_lossy=0;c.tp_wire_min_bytes=128*1024;return c;}
Json footprint(const WeightFootprint& w){return {{"statics",w.statics},{"experts",w.experts},{"expert_spare_dynamic_upper_bound",w.expert_spare},
    {"expert_unit_max",w.expert_unit_max},{"expert_layer_max",w.expert_layer_max},{"layer_unit_max",w.layer_unit_max},
    {"embedding",w.embedding},{"tower",w.tower},{"hosted",w.hosted},{"mapped",w.mapped},{"routed",w.routed},{"units",w.n_units}};}
Json plan_summary(const Plan& p,int status){Json unfit=Json::array();for(const auto& u:p.unfit)unfit.push_back({{"class",u.what},{"why",u.why},{"count",u.count},{"bytes",u.bytes}});
    Json tiers=Json::object();for(int t=0;t<kTierCount;++t)tiers[tier_name(Tier(t))]=p.totals.bytes[t];
    uint64_t stranded=0;for(const auto& u:p.unfit){need(u.bytes>=0,"negative unfit bytes");stranded+=uint64_t(u.bytes);}
    return {{"status",status},{"fits",status==RAD_OK&&p.fits()},{"notes",p.notes},{"unfit",unfit},{"unfit_bytes",stranded},
      {"tier_bytes",tiers},{"weight_budget",p.vram_budget},{"host_budget_after_staging",p.host_budget},
      {"slab_overhead",p.slab_overhead},{"file_stage_vram",p.file_stage_vram},{"file_stage_host",p.file_stage_reserve},
      {"units",p.units.size()},{"classes",p.classes.size()},{"failed_plan_ssd_tier_is_not_successful_ssd_support",status<0}};
}
struct RankQuote {
    Json json;std::unique_ptr<Builder> builder;WeightFootprint weights{};
    int64_t arena=0,host_arena=0,kv=0,batch=0,checkpoint=0,capacity=0;
    bool device_complete=false;
};
RankQuote declare_rank(Registry& registry,const Plugin& architecture,Checkpoint& checkpoint,CheckpointSources& sources,
                       const RadModelMeta& meta,const RadDeviceProps& device,int rank,const Options& o){
    RankQuote q;q.capacity=device.vram_bytes;q.builder=std::make_unique<Builder>(registry,meta,context(rank,o.ctx,o.batch));q.builder->set_instances(false);
    auto actual=sources.fn();std::map<std::string,Json> casts;
    q.builder->set_sources([&](const std::string& name,const Program& p,WeightSource* out) -> int {
        const int rc=actual(name,p,out);if(rc!=RAD_OK)return rc;
        const auto* raw=sources.find(name);if(!raw||!raw->derived.empty()||!raw->wnorm_scale.empty())return RAD_E_UNSUPPORTED;
        if(suffix(name,"attn_kv_b.weight")){
            // EXACT stock recipe's ONLY cast. This is an explicit metadata
            // hypothesis about a future converted source, never payload conversion.
            if(out->rank!=2||out->shape[0]<=0||out->shape[1]<=0)return RAD_E_SHAPE;
            const auto before=out->enc;out->enc=rad_enc_plain(RAD_BF16);
            casts[name]={{"logical_source",name},{"raw_encoding",enc_name(before)},{"planned_encoding","bf16"},
                {"shape",{out->shape[0],out->shape[1]}},{"payload_converted",false}};
        }
        return RAD_OK;
    });
    const int declared=architecture.arch_declare(q.builder.get(),&meta,&q.builder->program().ctx);
    if(declared<0){for(const auto& e:q.builder->errors())std::cerr<<e<<"\n";good(declared,"actual stock glm5next declaration rank"+std::to_string(rank));}
    const int finished=q.builder->finish();if(finished<0){for(const auto& e:q.builder->errors())std::cerr<<e<<"\n";good(finished,"actual Builder finish");}
    auto& p=q.builder->program();need(p.weights.size()<100000&&p.ops.size()<50000,"closed fullgraph metadata counts");
    Json holes=Json::array();for(const auto* hole:p.device_holes(false))holes.push_back(hole->line());
    std::vector<std::string> reference_ops;const int reference_instances=rad_reference_ops(p,&reference_ops);
    q.device_complete=holes.empty()&&reference_instances==0;
    Json by_kernel=Json::object();size_t initialized=0,reference_device_bands=0;for(size_t op=1;op<p.ops.size();++op)for(const auto& band:p.ops[op].bands){
        for(int d=0;d<RAD_N_DOMAINS;++d){const auto& r=band.dom[d];initialized+=r.instance!=nullptr;
            if(r.row&&d==RAD_DOMAIN_DEVICE){const std::string key=r.row->plugin+"/"+r.row->info->name;
                if(!by_kernel.contains(key))by_kernel[key]=0;
                by_kernel[key]=by_kernel[key].get<size_t>()+1;reference_device_bands+=r.row->reference;}}}
    need(initialized==0,"kernel instances must remain disabled");
    weight_footprint(p,&q.weights,TablePlacement::Vram,TablePlacement::Disk,false,false);
    q.arena=arena_plan_bytes(p,nullptr,nullptr);q.host_arena=arena_host_bytes(p);
    Config c=config(o.ctx,o.batch);KVManager kv;for(size_t g=1;g<p.kv_groups.size();++g)if(p.kv_groups[g].decl.kind==RAD_KV_RING)
        good(kv.declare_runtime_window(int32_t(g),p.kv_groups[g].decl.window),"actual ring window planning");
    good(kv.plan_groups(p.kv_groups,c),"actual pure KVManager::plan_groups");q.kv=round_mib(kv.ceiling(o.ctx));q.checkpoint=kv.checkpoint_bytes();
    KVGeom geometry=KVGeom::of(p.kv_groups);for(size_t g=0;g<geometry.g.size();++g){const int64_t width=kv.state_index_width(int32_t(g));if(width>0)geometry.g[g].sidx_width=width;}
    q.batch=BatchBuilder::device_bytes(p,c,geometry);need(q.arena>=0&&q.host_arena>=0&&q.kv>=0&&q.batch>=0,"actual nonnegative state/arena/batch costs");
    Json states=Json::array();for(const auto& g:kv.plans())if(g.index>0)states.push_back({{"name",g.name},{"kind",g.kind},{"layers",g.n_layers},{"block_size",g.block_size},
        {"bytes_per_block",g.bytes_per_block},{"bytes_per_sequence_state",g.bytes_per_state},{"states",g.n_states},{"zero",g.zero},{"replicated",g.replicated}});
    int64_t selected=0,identity_bytes=0,nonidentity_bytes=0,widen_bytes=0,nonidentity_max=0,widen_max=0;
    size_t used=0,widened=0,identity_count=0,nonidentity_count=0;std::set<std::string> source_names;
    std::vector<const WeightInfo*> largest;largest.reserve(11);
    for(size_t h=1;h<p.weights.size();++h){const auto& w=p.weights[h];need(w.stored_bytes>=0,"negative stored bytes");
        selected=plus(selected,w.stored_bytes);used+=w.first_use_op>=0;widened+=w.widen;source_names.insert(w.source);
        if(w.identity&&!w.widen){++identity_count;identity_bytes=plus(identity_bytes,w.stored_bytes);}
        else{++nonidentity_count;nonidentity_bytes=plus(nonidentity_bytes,w.stored_bytes);nonidentity_max=std::max(nonidentity_max,w.stored_bytes);
            if(w.widen){widen_bytes=plus(widen_bytes,w.stored_bytes);widen_max=std::max(widen_max,w.stored_bytes);}
            largest.push_back(&w);std::sort(largest.begin(),largest.end(),[](const auto* a,const auto* b){return a->stored_bytes!=b->stored_bytes?a->stored_bytes>b->stored_bytes:a->name<b->name;});
            if(largest.size()>10)largest.pop_back();}}
    Json top=Json::array();for(const auto* w:largest)top.push_back({{"name",w->name},{"source",w->source},{"stored_bytes",w->stored_bytes},
        {"logical_bytes",w->logical_bytes},{"layout",w->layout_tag},{"identity_flag",w->identity},{"widen",w->widen},
        {"dtype",w->stored_dtype},{"first_use",w->first_use_op},{"last_use",w->last_use_op}});
    Json loader={{"direct_identity_count",identity_count},{"direct_identity_stored_bytes",identity_bytes},
        {"nonidentity_OR_widen_count",nonidentity_count},{"nonidentity_OR_widen_stored_bytes",nonidentity_bytes},
        {"nonidentity_OR_widen_max_stored_bytes",nonidentity_max},{"widen_count",widened},{"widen_stored_bytes",widen_bytes},
        {"widen_max_stored_bytes",widen_max},{"top10_nonidentity_OR_widen",top},
        {"scope","stored output/layout metadata only; not measured worker/input/bounce/retained-output residency or recipe conversion peak"}};
    Json cast_list=Json::array();for(const auto& [name,cast]:casts){(void)name;cast_list.push_back(cast);}need(casts.size()==11,"MTP0 full trunk must use exactly11 MLA kv_b casts");
    q.json={{"rank",rank},{"device",device.name},{"capacity_bytes",device.vram_bytes},{"eligibility_card0_free_snapshot_NOT_rank_free",device.vram_free},
      {"weights_with_sentinel",p.weights.size()},{"used_weights",used},{"logical_sources",source_names.size()},{"widened_weights",widened},
      {"selected_stored_weight_bytes_sum",selected},{"ordinary_loader_metadata",loader},{"ops_with_sentinel",p.ops.size()},{"buffers_with_sentinel",p.buffers.size()},
      {"device_selection_complete",q.device_complete},{"device_holes",holes},{"device_hole_free",holes.empty()},
      {"stock_reference_gate_passed",reference_instances==0},{"stock_reference_only_op_instances",reference_instances},{"stock_reference_only_op_names",reference_ops},
      {"reference_device_bands",reference_device_bands},{"device_bands_by_kernel",by_kernel},
      {"weight_footprint",footprint(q.weights)},{"arena_plus_scratch_bytes",q.arena},{"host_arena_bytes",q.host_arena},
      {"kv_fixed_bytes",kv.fixed_bytes()},{"kv_ceiling_bytes",kv.ceiling(o.ctx)},{"kv_pool_mib_rounded_bytes",q.kv},
      {"checkpoint_one_slot_bytes_NOT_ALLOCATED",q.checkpoint},{"batch_staging_device_bytes",q.batch},
      {"state_groups",states},{"planned_recipe_casts",cast_list},{"sampler_declared",false},{"sampler_margin_device_bytes_assumed",sampler_margin},
      {"sampler_margin_host_bytes_assumed",sampler_margin},{"kernel_instances_initialized",initialized}};
    // No lambda with references to local casts/actual may escape this function.
    q.builder->set_sources({});return q;
}
} // namespace

int main(int argc,char** argv){Json report;std::string output;
    try{
        const auto o=options(argc,argv);output=o.output;const auto before=tracked_allocations();
        need(!device_is_host()&&std::string(device_backend_name())=="aql","actual AQL backend required; no HOST/synthetic DEVICE rows");
        const int visible=rad_dev_count();need(visible>=1,"one visible gfx1201 card required for actual code eligibility");
        RadDeviceProps eligibility{};good(rad_dev_props(0,&eligibility),"one actual device metadata query");
        need(!eligibility.is_host_backend&&!std::strcmp(eligibility.arch,"gfx1201")&&eligibility.vram_bytes>0,"closed gfx1201 eligibility card");
        std::array<RadDeviceProps,world> devices{};devices.fill(eligibility);
        Checkpoint checkpoint;good(checkpoint.open(o.checkpoint),"actual complete checkpoint metadata/map open");
        need(checkpoint.meta().arch=="glm5next"&&checkpoint.tensors().size()<=100000&&checkpoint.tensors().size()>70000,"actual full stockGLM checkpoint identity/metadata bound");
        Registry& registry=rad_tools_registry();registry.set_code_device(0);registry.set_no_device_code(false);
        // An actual stock --p2p off / --tp-wire exact planning configuration.
        // No enable-peer call or P2P mapping is made for this quote.
        registry.set_device_fact(RAD_FACT_P2P,0);LoadedPlugins plugins;
        good(rad_tools_load_plugins(o.home,{},checkpoint.meta().arch,checkpoint.meta().quant,&plugins,checkpoint.imports()!=Checkpoint::Import::None),"actual shipped plugin loading/DEVICE code eligibility");
        checkpoint.set_importers(&registry);const auto* architecture=registry.plugin(plugins.arch_plugin);
        need(architecture&&architecture->arch_declare&&architecture->arch_id=="glm5next","actual stock glm5next plugin selected");
        auto meta=checkpoint.meta();if(checkpoint.imports()!=Checkpoint::Import::None){meta.quant=plugins.arch_quant;meta.finish();}
        CheckpointSources sources(checkpoint);std::array<RankQuote,world> quotes;
        bool selection=true;int64_t total_host_arenas=0,total_physical=0,total_selected=0;
        report={{"schema","glm53-stock-fullgraph-tp4-planning-v1"},{"checkpoint",o.checkpoint},{"home",o.home},
          {"mode","actual DEVICE metadata selection; no kernel instances or inference"},{"ctx",o.ctx},{"max_tok",o.batch},{"max_seqs",1},{"mtp_depth",0},{"deterministic",true},
          {"kv_dtype","bf16"},{"state_dtype","f32"},{"embedding","vram"},{"tp_wire","exact"},{"p2p","off"},{"vision",false},
          {"prefix_cache",false},{"checkpoint_slots",0},{"raw_physical_tensors",checkpoint.tensors().size()},
          {"visible_devices_reported",visible},{"eligibility_device_queried",0},
          {"four_card_geometry_assumption","logical ranks0..3 use identical gfx1201/capacity to visible card0; parent separately verified4physical identical boards; no other card/device binding/peer query"},
          {"recipe_scope","hypothetical preconversion of exactly *attn_kv_b.weight toBF16 per stock glm53-mxfp4.recipe; no payload conversion"},
          {"planner_container",false},{"weights_disk_tier",false},{"raw_checkpoint_direct_serve_claim",false},
          {"model_allocations_requested_by_tool",false},{"queues_created_by_tool",false},{"kernel_launches",false},
          {"sampler_instances",false},{"sampler_omission_reserve_is_assumed",true},{"ranks",Json::array()},{"scenarios",Json::array()}};
        Json modules=Json::array();for(const auto& plugin:registry.plugins())modules.push_back({{"name",plugin.name},{"version",plugin.version},{"path",plugin.path},{"kind",plugin.kind},{"device_code_absent",plugin.device_code_absent}});report["plugins"]=modules;
        for(int r=0;r<world;++r){quotes[size_t(r)]=declare_rank(registry,*architecture,checkpoint,sources,meta.m,devices[size_t(r)],r,o);
            const auto& q=quotes[size_t(r)];selection=selection&&q.device_complete;total_host_arenas=plus(total_host_arenas,q.host_arena);total_physical=plus(total_physical,q.capacity);
            total_selected=plus(total_selected,q.json.at("selected_stored_weight_bytes_sum").get<int64_t>());report["ranks"].push_back(q.json);}
        report["source_refusals"]=sources.refused();selection=selection&&sources.refused().empty();
        const int64_t host_total=meminfo("MemTotal"),host_available=meminfo("MemAvailable");
        report["host_memtotal_bytes"]=host_total;report["host_memavailable_snapshot_bytes"]=host_available;report["four_card_physical_bytes"]=total_physical;
        report["four_rank_selected_weight_bytes_sum"]=total_selected;
        report["host_nonpool_margin_bytes_ASSUMED"]=8*GiB;report["host_margin_scope"]="OS/current nonmodel consumers/loader temporaries:8GiB illustrative allowance, NOT a proved startup peak; current production muststop separately";
        for(const int64_t reserve_mib:{512,1024,1536,2048})for(const int64_t host_mib:{26624,27136,27648,28160,28672}){
            Json scenario{{"runtime_driver_collective_headroom_mib_ASSUMED",reserve_mib},{"host_pool_mib_per_rank",host_mib},{"ranks",Json::array()}};
            bool fits=selection;int64_t all_host=plus(plus(host_mib*MiB*world,total_host_arenas),world*sampler_margin);all_host=plus(all_host,8*GiB);
            for(int r=0;r<world;++r){auto& q=quotes[size_t(r)];const int64_t nonweight=plus(plus(plus(q.arena,q.kv),q.batch),plus(reserve_mib*MiB,sampler_margin));
                const int64_t weight_mib=std::max<int64_t>(0,q.capacity-nonweight)/MiB;Config c=config(o.ctx,o.batch);c.host_pool_mib=host_mib;c.vram_weights_mib=weight_mib;
                PlannerInput input{&q.builder->program(),&c,r,world,nullptr,0,false,false};Plan plan;
                const int status=weight_mib>0?Planner(input).plan(&plan):RAD_E_NOMEM;fits=fits&&status==RAD_OK&&plan.fits();
                auto entry=plan_summary(plan,status);entry["rank"]=r;entry["nonweight_device_budget_bytes"]=nonweight;
                entry["weight_budget_mib_derived"]=weight_mib;entry["dynamic_spare_conservative_floor_helper_NOT_deterministic_minimum"]=expert_vram_floor(q.weights,host_mib*MiB);
                entry["deterministic_expert_payload_lower_bound_NOT_first_fit_cost"]=std::max<int64_t>(0,q.weights.experts-host_mib*MiB);
                scenario["ranks"].push_back(std::move(entry));}
            scenario["all_rank_weight_plans_fit"]=fits;scenario["host_pools_plus_hostarena_sampler_and_ASSUMED8GiB_bytes"]=all_host;
            scenario["within_physical_memtotal_under_ASSUMED_margin"]=host_total>0&&all_host<=host_total;
            scenario["within_current_memavailable_without_stopping_anything"]=host_available>0&&all_host<=host_available;
            scenario["planning_candidate_only"]=fits&&host_total>0&&all_host<=host_total;report["scenarios"].push_back(std::move(scenario));}
        const auto after=tracked_allocations();report["core_tracked_allocated_bytes_before_queries"]=allocations(before);
        report["core_tracked_allocated_bytes_after_planning"]=allocations(after);report["core_tracked_allocation_bytes_unchanged"]=before==after;
        report["actual_device_selection_complete"]=selection;
        report["hidden_runtime_effects"]={"AQL discovery calls hsa_init/agent/pool discovery and code-object ISA validation; host metadata and driver-internal state canallocate.",
          "Plugin dlopen/static registration, source metadata/cache and kernel layout/scratch callbacks execute; opaque plugin/HSA allocations are not a process-wide allocation proof.",
          "No Pool.configure/carve, Engine.start, kernel init_instance, Ctx, rad_stream_create, copy or kernel launch is called by this tool.",
          "Checkpoint::open maps real payload files in virtual address space and reads headers/config; source resolver metadata is real; tensor data are not converted/copied by this tool.",
          "Sampler graph omitted and covered with separate16MiB/rank device +16MiB/rank host assumptions; driver/queue/collective/module/loader peaks remain declared margins, not measurements.",
          "deterministic=true disables dynamic heat/shadow slots, XP prefetch and dynamic expert-layer lending/staging baseline; actual Planner still reports any required ordinary staged ring/slab costs. Kernel collective/driver/queues remain explicit assumed margins."};
        report["not_proved"]={"actual startup/Pools allocations","whole-model inference correctness","TP communicator initialization","Loader/pin/startup peak","current hardware fit while production serves","throughput"};
        report["completed"]=true;publish(output,report);std::cout<<"WROTE full stockGLM TP4 planning report "<<output<<" selection="<<selection<<" trackedAllocUnchanged="<<(before==after)<<"\n";
        return selection&&before==after?0:1;
    }catch(const std::exception& e){std::cerr<<"FAIL stockGLM planning: "<<e.what()<<"\n";
        if(!output.empty()&&!fs::exists(output)){try{report["completed"]=false;report["error"]=e.what();publish(output,report);}catch(...){}}
        return 1;}
}
