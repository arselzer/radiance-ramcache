// Standalone planning only. No Engine, Pools, Ctx, sampler instance, queue,
// kernel init/launch, upload, conversion, model allocation or source payload copy.
#include "build/startup.h"
#include "device/device.h"
#include "format/checkpoint.h"
#include "format/share.h"
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
struct Options {std::string model,home,output;int64_t ctx=512,batch=128,seqs=1,spec=0;};
void need(bool b,const std::string& why){if(!b)throw std::runtime_error(why);}
void good(int rc,const std::string& why){need(rc==RAD_OK,why+" status="+std::to_string(rc));}
int64_t plus(int64_t a,int64_t b){need(a>=0&&b>=0&&a<=INT64_MAX-b,"nonnegative quote overflow");return a+b;}
int64_t round_mib(int64_t n){return plus(n,MiB-1)/MiB*MiB;}
int64_t meminfo(const char* key){std::ifstream in("/proc/meminfo");std::string name,unit;int64_t n=0;
    while(in>>name>>n>>unit){if(name==std::string(key)+":")return n*1024;}return -1;}
std::array<int64_t,4> tracked_allocations(){std::array<int64_t,4> a{};for(int k=0;k<4;++k)rad_dev_alloc_totals(k,&a[size_t(k)],nullptr);return a;}
Json allocations(const std::array<int64_t,4>& a){return {{"device",a[0]},{"pinned",a[1]},{"host",a[2]},{"mapped",a[3]}};}
void bounded(const std::string& p){need(!p.empty()&&p.size()<=4096&&p.find('\0')==std::string::npos&&fs::path(p).is_absolute(),"absolute bounded local path required");}
Options options(int argc,char** argv){Options o;
    need(argc>=8&&argc<=16&&(argc%2)==0,"usage: qfn8bit_plan --run-device-metadata --model ABS_PRODUCTION_RAD --home ABS_SHIPPED_HOME --output ABS_NEW_JSON [--ctx 512|2048|262144] [--batch-tokens 128|256|2048|4096] [--max-seqs 1|4|8] [--mtp-depth 0|3]");
    need(!std::strcmp(argv[1],"--run-device-metadata"),"explicit device-metadata gate required");
    std::set<std::string> seen;
    for(int i=2;i<argc;i+=2){need(i+1<argc,"missing option value");const std::string k=argv[i],v=argv[i+1];need(seen.insert(k).second,"duplicate option");
        if(k=="--model")o.model=v;else if(k=="--home")o.home=v;else if(k=="--output")o.output=v;
        else if(k=="--ctx"){need(v=="512"||v=="2048"||v=="262144","closed context");o.ctx=std::stoll(v);}
        else if(k=="--batch-tokens"){need(v=="128"||v=="256"||v=="2048"||v=="4096","closed batch");o.batch=std::stoll(v);}
        else if(k=="--max-seqs"){need(v=="1"||v=="4"||v=="8","closed sequence bound");o.seqs=std::stoll(v);}
        else if(k=="--mtp-depth"){need(v=="0"||v=="3","closed draft depth");o.spec=std::stoll(v);}
        else need(false,"unknown option: "+k);}
    bounded(o.model);bounded(o.home);bounded(o.output);need(!fs::exists(o.output),"output must be NEW");return o;}
void publish(const std::string& path,const Json& j){const auto text=j.dump(2)+"\n";need(text.size()<=16u<<20,"report16MiB cap");
    const int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600);need(fd>=0,"new output open");
    bool complete=false;try{size_t at=0;while(at<text.size()){const auto n=::write(fd,text.data()+at,text.size()-at);if(n<0&&errno==EINTR)continue;need(n>0,"output write");at+=size_t(n);}need(!::fsync(fd),"output fsync");complete=true;}
    catch(...){::close(fd);::unlink(path.c_str());throw;}::close(fd);need(complete,"complete report");}

// The existing file is a shape/encoding source ONLY. None of its payload accessors
// are used. The returned hypothetical Program must never be used for loading.
struct ExpertName {int layer=-1,expert=-1,projection=-1;};
bool expert_name(const std::string& name,ExpertName* out){
    int l=-1,e=-1,n=-1;int projection=-1;
    if(std::sscanf(name.c_str(),"blk.%d.ffn_gate_up_exps.%d.weight%n",&l,&e,&n)==2&&n==int(name.size()))projection=0;
    else {n=-1;if(std::sscanf(name.c_str(),"blk.%d.ffn_down_exps.%d.weight%n",&l,&e,&n)==2&&n==int(name.size()))projection=1;}
    if(projection<0)return false;
    need(l>=0&&l<=48&&e>=0&&e<512,"expert identity outside closed QFN source");*out={l,e,projection};return true;
}
bool protected_expert(int layer,int expert){
    return (layer==34&&(expert==407||expert==496))||(layer==44&&(expert==292||expert==350))||
        (layer==46&&(expert==290||expert==392))||
        (layer==47&&(expert==122||expert==143||expert==399||expert==445));
}
bool plain_bf16(const RadEncoding& e){return rad_enc_is(&e,"plain")&&e.n_planes==1&&e.plane[0].dtype==RAD_BF16;}
bool production_w4(const RadEncoding& e){
    const auto* s=rad_enc_plane(&e,"scale");const auto* s2=rad_enc_plane(&e,"scale.1");const auto* table=rad_enc_plane(&e,"table");
    return rad_enc_is(&e,"affine")&&e.n_planes==4&&e.plane[0].dtype==RAD_U4&&
        !std::strcmp(e.transform,"fwht128")&&s&&s->dtype==RAD_F8E4M3&&s->block[0]==1&&s->block[1]==64&&
        s2&&s2->dtype==RAD_F32&&s2->block[0]==0&&s2->block[1]==0&&table&&table->dtype==RAD_F32&&
        table->kind==RAD_PLANE_TABLE&&table->extent[0]==1&&table->extent[1]==16;
}
void source_geometry(const WeightSource& w,const ExpertName& e){
    need(w.rank==2&&w.shape[0]==(e.projection?2560:1280)&&w.shape[1]==(e.projection?640:2560),"actual QFN expert logical shape");
    need(protected_expert(e.layer,e.expert)?plain_bf16(w.enc):production_w4(w.enc),"production source protected/W4 encoding disagreement");
}
void metadata(const RadFile& f,MetaOwn* out){
    out->arch=f.str(f.header().arch_id);out->name=f.str(f.header().model_name);out->quant=f.str(f.header().quant);
    need(out->arch=="qwen4exp"&&f.meta_count()>0&&f.meta_count()<8192&&f.entry_count()>49000&&f.entry_count()<100000,"closed QFN container metadata");
    auto mi=[&](const char* key,int64_t fallback=0){int64_t n=fallback;f.get_i(key,&n);return n;};
    auto mf=[&](const char* key,double fallback){double n=fallback;f.get_f(key,&n);return float(n);};
    auto& m=out->m;m.n_layers=mi("n_layers");m.n_embd=mi("n_embd");m.n_head=mi("n_head");m.n_head_kv=mi("n_head_kv");m.head_dim=mi("head_dim");
    m.n_ff=mi("n_ff");m.n_vocab=mi("n_vocab");m.n_ctx_train=mi("n_ctx_train");m.n_expert=mi("n_expert");m.n_expert_used=mi("n_expert_used");m.n_expert_shared=mi("n_expert_shared");
    m.rms_eps=mf("rms_eps",1e-6);m.rope_theta=mf("rope_theta",10000);m.rope_scale=1;
    need(m.n_layers==48&&m.n_embd==2560&&m.n_expert==512&&m.n_expert_used==10&&m.n_vocab==248320&&m.head_dim==256,"actual Flash-Next geometry");
    for(int64_t i=0;i<f.meta_count();++i){const auto& v=f.meta_at(i);const char* key=f.str(v.key);need(std::strlen(key)<=4096,"metadata key cap");
        if(v.type==RAD_P_INT)out->kv(key,std::to_string(v.v.i));
        else if(v.type==RAD_KV_F64)out->kv(key,fmt("%.17g",v.v.f));
        else {const char* text=f.str(v.v.s);need(std::strlen(text)<=65536,"metadata value cap");out->kv(key,text);}}
    out->finish();
}
Config config(const Options& o,TablePlacement ng){Config c;c.tp=world;c.max_tok=o.batch;c.max_seqs=o.seqs;c.max_ctx=o.ctx;c.n_spec=o.spec;
    c.tp_wire_exact=1;c.tp_attn_rows=0;c.p2p=RAD_P2P_OFF;c.kv_cache_dtype="fp8";c.state_cache_dtype="f32";
    c.mm_max_patches=0;c.mm_audio_frames=0;c.prefix_cache=false;c.prefix_cache_mib=0;c.checkpoint_slots=0;
    c.embedding_placement=c.embedding_resolved=TablePlacement::Vram;c.ngram_placement=c.ngram_resolved=ng;
    c.placement="expert_tiered";c.weights_disk_tier=false;c.deterministic=true;return c;}
RadBuildCtx context(int rank,const Options& o){RadBuildCtx c{};c.rank=rank;c.world_size=world;c.max_tok=o.batch;c.max_seqs=o.seqs;c.max_ctx=o.ctx;
    c.max_spec=o.spec;c.kv_dtype=RAD_F8E4M3;c.state_dtype=0;c.scope="";c.tp_wire_lossy=0;c.tp_wire_min_bytes=128*1024;return c;}
Json footprint(const WeightFootprint& w){return {{"statics",w.statics},{"experts",w.experts},{"expert_spare_dynamic_upper_bound",w.expert_spare},
    {"expert_unit_max",w.expert_unit_max},{"expert_layer_max",w.expert_layer_max},{"layer_unit_max",w.layer_unit_max},
    {"embedding",w.embedding},{"ngram",w.ngram},{"ngram_device_reader",w.ngram_device},{"ngram_host_reader",w.ngram_host},
    {"tower",w.tower},{"hosted",w.hosted},{"mapped",w.mapped},{"units",w.n_units}};}
Json plan_summary(const Plan& p,int status){Json unfit=Json::array();for(const auto& u:p.unfit)unfit.push_back({{"class",u.what},{"why",u.why},{"count",u.count},{"bytes",u.bytes}});
    Json tiers=Json::object();for(int t=0;t<kTierCount;++t)tiers[tier_name(Tier(t))]=p.totals.bytes[t];
    return {{"status",status},{"fits",status==RAD_OK&&p.fits()},{"notes",p.notes},{"unfit",unfit},{"tier_bytes",tiers},
        {"weight_budget",p.vram_budget},{"host_budget_after_staging",p.host_budget},{"slab_overhead",p.slab_overhead},
        {"file_stage_vram",p.file_stage_vram},{"file_stage_host",p.file_stage_reserve},{"ngram_copy_bytes",p.ngram_copy_bytes},
        {"units",p.units.size()},{"classes",p.classes.size()},{"SSD_in_failed_plan_is_not_successful_backing",status<0}};}
struct RankQuote {
    Json json;std::unique_ptr<Builder> builder;WeightFootprint weights{};
    int64_t arena=0,host_arena=0,kv=0,batch=0,capacity=0,ngram_copy_raw=0,ngram_copy_page_upper=0;bool complete=false;
};
RankQuote declare_rank(Registry& registry,const Plugin& arch,const RadFile& file,const RadModelMeta& meta,
                       const RadDeviceProps& device,int rank,const Options& o){
    RankQuote q;q.capacity=device.vram_bytes;q.builder=std::make_unique<Builder>(registry,meta,context(rank,o));q.builder->set_instances(false);
    auto actual=rad_container_sources(file);std::array<uint8_t,49*512*2> seen{};uint64_t overridden=0,protected_sources=0;
    q.builder->set_sources([&](const std::string& name,const Program& p,WeightSource* out)->int {
        const int rc=actual(name,p,out);if(rc!=RAD_OK)return rc;ExpertName e;
        if(expert_name(name,&e)){
            source_geometry(*out,e);need(e.layer<48||o.spec>0,"unexpected MTP source under MTP0");
            const size_t index=size_t((e.layer*512+e.expert)*2+e.projection);
            if(!seen[index]){seen[index]=1;if(protected_expert(e.layer,e.expert))++protected_sources;else ++overridden;}
            if(!protected_expert(e.layer,e.expert))out->enc=rad_enc_affine(RAD_F8E4M3,RAD_BF16,128,128);
        }return RAD_OK;
    });
    const int declared=arch.arch_declare(q.builder.get(),&meta,&q.builder->program().ctx);
    if(declared<0){for(const auto& e:q.builder->errors())std::cerr<<e<<"\n";good(declared,"actual QFN declaration");}
    const int finished=q.builder->finish();if(finished<0){for(const auto& e:q.builder->errors())std::cerr<<e<<"\n";good(finished,"actual QFN Builder finish");}
    auto& p=q.builder->program();need(p.weights.size()<150000&&p.ops.size()<50000&&p.buffers.size()<50000,"closed fullgraph metadata counts");
    const uint64_t expert_pairs=uint64_t(48+(o.spec?1:0))*512-10;
    need(overridden==2*expert_pairs&&protected_sources==20,"complete FP8 override/protected BF16 source coverage");
    Json holes=Json::array();for(const auto* h:p.device_holes(false))holes.push_back(h->line());
    std::vector<std::string> references;const int ref_count=rad_reference_ops(p,&references);q.complete=holes.empty()&&ref_count==0;
    uint64_t initialized=0,fp8_ops=0,fp8_gu=0,fp8_dn=0,protected_views=0;std::array<uint64_t,2> gu_width{},dn_width{};
    Json kernels=Json::object(),samples=Json::array(),ngram_members=Json::array();int64_t selected=0;uint64_t ng_codes=0,ng_scales=0;
    for(size_t i=1;i<p.ops.size();++i){const auto& op=p.ops[i];
        if(op.op=="moe_gemm_q_mod4"){need(op.base.get_s("dtype")&&!std::strcmp(op.base.get_s("dtype"),"fp8a8"),"four-class expert op must be fp8a8");++fp8_ops;}
        for(const auto& band:op.bands)for(int domain=0;domain<RAD_N_DOMAINS;++domain){const auto& res=band.dom[domain];initialized+=res.instance!=nullptr;
            if(res.row&&domain==RAD_DOMAIN_DEVICE){const auto key=res.row->plugin+"/"+res.row->info->name;if(!kernels.contains(key))kernels[key]=0;kernels[key]=kernels[key].get<uint64_t>()+1;}}}
    need(initialized==0&&fp8_ops>=96,"instances disabled and actual mod4 FP8 ops selected");
    const auto expected=rad_enc_affine(RAD_F8E4M3,RAD_BF16,128,128);
    for(size_t h=1;h<p.weights.size();++h){const auto& w=p.weights[h];selected=plus(selected,w.stored_bytes);
        if(is_ngram_table(p,rad_weight(h))){
            need(w.decl.group.layer<0&&w.identity&&!w.widen&&w.n_sel==1,"classified ngram must be an unchanged model-level plane");
            const auto& plane=w.enc.plane[w.sel[0]];
            if(!std::strcmp(plane.role,"codes")){need(plane.dtype==RAD_F8E4M3&&w.stored_bytes==51200245760ll,"actual FP8 PLE code bytes");++ng_codes;}
            else if(!std::strcmp(plane.role,"scale")){need(plane.dtype==RAD_BF16&&w.stored_bytes==2,"actual PLE BF16 scalar bytes");++ng_scales;}
            else need(false,"unexpected classified ngram plane");
            q.ngram_copy_raw=plus(q.ngram_copy_raw,w.stored_bytes);
            q.ngram_copy_page_upper=plus(q.ngram_copy_page_upper,plus(w.stored_bytes,RAD_ALIGN_UNIT-1)/RAD_ALIGN_UNIT*RAD_ALIGN_UNIT);
            ngram_members.push_back({{"name",w.name},{"role",plane.role},{"source",w.source},{"stored_bytes",w.stored_bytes},{"alignment",w.align}});
        }
        ExpertName e;if(!expert_name(w.source,&e))continue;
        if(protected_expert(e.layer,e.expert)){need(plain_bf16(w.enc),"protected BF16 source changed");++protected_views;continue;}
        need(rad_enc_equal(&w.enc,&expected),"hypothetical expert encoding drift");
        const auto* source=file.find(w.source);need(source,"original logical source metadata disappeared");
        PlaneShare share[RAD_ENC_MAX_PLANES];std::string why;good(weight_share(w,w.enc,source->rank,source->shape,world,rank,share,&why),"actual hypothetical TP4 plane share: "+why);
        if(w.n_sel!=1||w.sel[0]!=0)continue;
        const int64_t width=w.shard_hi-w.shard_lo;need(width==128||width==256,"FP8 TP4 must use whole128-column blocks");
        if(!e.projection){need(w.decl.rank==2&&w.decl.shape[0]==2*width&&w.decl.shape[1]==2560&&w.decl.n_row_parts==2,"GU halves are split equally on this rank");++fp8_gu;++gu_width[size_t(width==256)];}
        else {need(w.decl.rank==2&&w.decl.shape[0]==2560&&w.decl.shape[1]==width,"DN COL shard width");++fp8_dn;++dn_width[size_t(width==256)];}
        if(samples.size()<8&&e.layer==0&&e.expert<4){Json rectangles=Json::array();for(const auto& rect:share[0].rects)rectangles.push_back({{"row0",rect.row0},{"rows",rect.rows},{"byte0",rect.byte0},{"bytes",rect.bytes}});
            samples.push_back({{"source",w.source},{"rank",rank},{"projection",e.projection?"down":"gate_up"},{"expert_mod4",e.expert%4},{"shard_lo",w.shard_lo},{"shard_hi",w.shard_hi},
                {"stored_bytes",w.stored_bytes},{"layout",w.layout_tag},{"stored_dtype",w.stored_dtype},{"code_rectangles",rectangles}});}}
    need(fp8_gu==expert_pairs&&fp8_dn==expert_pairs&&protected_views==20,"actual complete expert weight views");
    weight_footprint(p,&q.weights,TablePlacement::Vram,TablePlacement::Disk,false,true);
    need(ng_codes==1&&ng_scales==1&&q.ngram_copy_raw==51200245762ll&&q.ngram_copy_page_upper==51200249856ll&&
         q.weights.ngram==q.ngram_copy_page_upper&&q.weights.ngram_host,
         "actual PLE classified members: codes="+std::to_string(ng_codes)+" scales="+std::to_string(ng_scales)+
         " raw="+std::to_string(q.ngram_copy_raw)+" pages="+std::to_string(q.ngram_copy_page_upper)+
         " footprint="+std::to_string(q.weights.ngram)+" host="+std::to_string(q.weights.ngram_host)+" device="+std::to_string(q.weights.ngram_device));
    q.arena=arena_plan_bytes(p,nullptr,nullptr);q.host_arena=arena_host_bytes(p);Config c=config(o,TablePlacement::Disk);KVManager kv;
    for(size_t g=1;g<p.kv_groups.size();++g)if(p.kv_groups[g].decl.kind==RAD_KV_RING)good(kv.declare_runtime_window(int32_t(g),p.kv_groups[g].decl.window),"actual runtime window metadata");
    good(kv.plan_groups(p.kv_groups,c),"actual KV plan");q.kv=round_mib(kv.ceiling(o.ctx));
    KVGeom geometry=KVGeom::of(p.kv_groups);for(size_t g=0;g<geometry.g.size();++g){const auto width=kv.state_index_width(int32_t(g));if(width>0)geometry.g[g].sidx_width=width;}
    q.batch=BatchBuilder::device_bytes(p,c,geometry);need(q.arena>=0&&q.host_arena>=0&&q.batch>=0,"nonnegative actual runtime quote");
    Json states=Json::array();for(const auto& g:kv.plans())if(g.index>0)states.push_back({{"name",g.name},{"kind",g.kind},{"layers",g.n_layers},{"bytes_per_block",g.bytes_per_block},{"bytes_per_sequence_state",g.bytes_per_state},{"states",g.n_states},{"zero",g.zero}});
    q.json={{"rank",rank},{"physical_card_capacity_assumed",q.capacity},{"weights_with_sentinel",p.weights.size()},{"buffers_with_sentinel",p.buffers.size()},{"ops_with_sentinel",p.ops.size()},
        {"source_FP8_overrides",overridden},{"protected_BF16_sources_unchanged",protected_sources},{"protected_BF16_views",protected_views},{"FP8_mod4_ops",fp8_ops},
        {"FP8_GU_views",fp8_gu},{"FP8_DN_views",fp8_dn},{"GU_width128_count",gu_width[0]},{"GU_width256_count",gu_width[1]},{"DN_width128_count",dn_width[0]},{"DN_width256_count",dn_width[1]},
        {"actual_ragged_code_rectangles",samples},{"device_holes",holes},{"reference_only_ops",references},{"stock_reference_gate_passed",ref_count==0},{"device_selection_complete",q.complete},
        {"device_bands_by_kernel",kernels},{"kernel_instances_initialized",initialized},{"selected_stored_weight_sum_WITH_replicated_PLE",selected},
        {"weight_footprint",footprint(q.weights)},{"classified_ngram_members",ngram_members},{"ngram_raw_copy_bytes",q.ngram_copy_raw},{"ngram_copy_page_upper",q.ngram_copy_page_upper},{"arena_device_including_scratch",q.arena},{"arena_host",q.host_arena},{"KV_fixed",kv.fixed_bytes()},
        {"KV_ceiling",kv.ceiling(o.ctx)},{"KV_pool_rounded",q.kv},{"state_groups",states},{"batch_device",q.batch},{"one_checkpoint_bytes_NOT_ALLOCATED",kv.checkpoint_bytes()},
        {"sampler_graph_omitted",true},{"sampler_device_margin_ASSUMED",sampler_margin},{"sampler_host_margin_ASSUMED",sampler_margin}};
    q.builder->set_sources({});return q;
}
} // namespace

int main(int argc,char** argv){Json report;std::string output;
    try {const auto o=options(argc,argv);output=o.output;const auto before=tracked_allocations();
        need(!device_is_host()&&std::string(device_backend_name())=="aql","actual AQL metadata backend required");
        const int visible=rad_dev_count();need(visible>=1,"one visible eligibility card required");RadDeviceProps device{};good(rad_dev_props(0,&device),"actual device metadata");
        need(!device.is_host_backend&&!std::strcmp(device.arch,"gfx1201")&&device.vram_bytes>0,"gfx1201 metadata profile");
        RadFile file;good(file.open(o.model.c_str()),"existing production container metadata open");need(file.size()==121969901568ll,"exact parent-specified production container size");
        MetaOwn meta;metadata(file,&meta);Registry& registry=rad_tools_registry();registry.set_code_device(0);registry.set_no_device_code(false);registry.set_device_fact(RAD_FACT_P2P,0);
        LoadedPlugins plugins;good(rad_tools_load_plugins(o.home,{},meta.arch,meta.quant,&plugins,false),"actual shipped QFN and kernel plugins");
        const auto* arch=registry.plugin(plugins.arch_plugin);need(arch&&arch->arch_declare&&arch->arch_id=="qwen4exp","actual QFN architecture");
        report={{"schema","qfn8bit-hypothetical-postconversion-TP4-layout-v1"},{"HYPOTHETICAL_ONLY",true},{"source_model",o.model},{"source_container_bytes",file.size()},
            {"source_payload_hash_verified",false},{"hypothesis","Replace ONLY production W4NL expert source encodings by E4M3/BF16[128,128]; ten BF16 experts and all nonexperts unchanged"},
            {"home",o.home},{"ctx",o.ctx},{"max_tok",o.batch},{"max_seqs",o.seqs},{"mtp_depth",o.spec},{"kv_dtype","fp8"},{"state_dtype","f32"},
            {"TP",4},{"tp_wire","exact"},{"P2P","off"},{"deterministic",true},{"vision",false},{"embedding","vram"},{"prefix_cache",false},{"checkpoint_slots",0},
            {"future_container_assumption",true},{"weights_disk_tier",false},{"payload_read_or_conversion",false},{"kernel_init_or_launch",false},{"Pools_Ctx_Engine_or_sampler_instance",false},
            {"visible_devices_reported",visible},{"eligibility_card_queried",0},{"four_card_geometry_assumption","All4 logical ranks use visible card0 ISA/capacity; parent separately verifies boards; no peer mapping or device binding"},
            {"KV_pool_scope","KV ceiling is worst-case simultaneous FULL configured context for ALL max-seqs, not stock Engine minimum pool/admission; a negative c8 quote does not mean c8 cannot be configured with shared KV capacity"},
            {"nonexpert_precision_scope","Controlled production nonexperts retained; this is NOT the complete official FP8 checkpoint's different trunk/HC/head precision"},
            {"ranks",Json::array()},{"scenarios",Json::array()}};
        Json loaded=Json::array();for(const auto& p:registry.plugins())loaded.push_back({{"name",p.name},{"version",p.version},{"path",p.path},{"kind",p.kind},{"device_code_absent",p.device_code_absent}});report["plugins"]=loaded;
        std::array<RankQuote,world> ranks;bool complete=true;int64_t host_arenas=0;
        for(int rank=0;rank<world;++rank){ranks[size_t(rank)]=declare_rank(registry,*arch,file,meta.m,device,rank,o);const auto& q=ranks[size_t(rank)];
            complete=complete&&q.complete;host_arenas=plus(host_arenas,q.host_arena);report["ranks"].push_back(q.json);}
        const int64_t host_total=meminfo("MemTotal"),available=meminfo("MemAvailable");
        report["host_memtotal"]=host_total;report["host_memavailable_snapshot"]=available;report["host_available_after_production_estimate_NOT_MEASURED"]=int64_t(113600)*GiB/1000;
        report["shared_disk_rowcache_capacity_source_bytes"]=int64_t(1<<19)*160;report["shared_disk_rowcache_metadata_source_bytes"]=int64_t(1<<19)*9+(1<<16);
        report["shared_disk_rowcache_full_capacity_reserve_ASSUMED"]=96*MiB;report["host_nonpool_margin_ASSUMED"]=8*GiB;
        for(const auto ng:{TablePlacement::Ram,TablePlacement::Disk})for(const int64_t reserve:{512,1024,1536,2048})for(const int64_t host_mib:{12288,14336,16384,18432,20480,24576,28672}){
            Json scenario{{"ngram",table_placement_name(ng)},{"host_pool_mib_PER_RANK",host_mib},{"runtime_driver_collective_headroom_mib_ASSUMED",reserve},{"ranks",Json::array()}};
            bool fits=complete;int64_t shared_ngram=ng==TablePlacement::Ram?ranks[0].ngram_copy_page_upper:96*MiB;
            int64_t host=plus(plus(host_mib*MiB*world,host_arenas),plus(world*sampler_margin,plus(8*GiB,shared_ngram)));
            for(int rank=0;rank<world;++rank){auto& q=ranks[size_t(rank)];const auto overhead=plus(plus(q.arena,q.kv),plus(q.batch,plus(reserve*MiB,sampler_margin)));
                Config c=config(o,ng);c.host_pool_mib=host_mib;c.vram_weights_mib=std::max<int64_t>(0,q.capacity-overhead)/MiB;
                PlannerInput input{&q.builder->program(),&c,rank,world,nullptr,0,true,false};Plan plan;
                const int status=c.vram_weights_mib>0?Planner(input).plan(&plan):RAD_E_NOMEM;fits=fits&&status==RAD_OK&&plan.fits();
                if(status==RAD_OK)need(ng==TablePlacement::Ram?plan.ngram_copy_bytes==q.ngram_copy_raw:plan.ngram_copy_bytes==0,"actual Planner PLE copy charge rank="+std::to_string(rank)+" got="+std::to_string(plan.ngram_copy_bytes)+" expected="+std::to_string(ng==TablePlacement::Ram?q.ngram_copy_raw:0));
                auto entry=plan_summary(plan,status);entry["rank"]=rank;entry["nonweight_device_budget"]=overhead;entry["derived_weights_mib"]=c.vram_weights_mib;scenario["ranks"].push_back(std::move(entry));}
            scenario["all_weight_plans_fit"]=fits;scenario["ngram_host_shared_ONCE_outside_pools"]=shared_ngram;scenario["all_host_bytes_under_ASSUMED_allowances"]=host;
            scenario["within_MemTotal_under_ASSUMED_allowances"]=host_total>0&&host<=host_total;
            scenario["within_current_MemAvailable"]=available>0&&host<=available;
            scenario["within_parent_estimated_poststop113_6GiB"]=host<=int64_t(113600)*GiB/1000;
            scenario["hypothetical_planning_candidate_only"]=fits&&host_total>0&&host<=host_total&&host<=int64_t(113600)*GiB/1000;report["scenarios"].push_back(std::move(scenario));}
        const auto after=tracked_allocations();report["tracked_allocations_before"]=allocations(before);report["tracked_allocations_after"]=allocations(after);report["tracked_allocations_unchanged"]=before==after;
        report["actual_selection_complete"]=complete;report["not_proved"]={"conversion/import/precision","payload correctness or checksum","actual Pools/Engine startup","TP collective initialization","loader peak or physical RSS","live hardware fit","inference/throughput"};
        report["side_effects_scope"]={"Actual AQL HSA discovery/ISA eligibility and plugin dlopen/registration can allocate opaque host/driver storage. Core tracked allocation equality is not a process allocator proof.",
            "No queues, backend instances, pools, payload accessor/read_at, transfers, kernel initialization, model allocation or inference are called by this tool.",
            "Original .rad is mapped for validated metadata; hypothetical Program sources no longer describe its payload, so loading that Program is expressly invalid and has no path here.",
            "Ngram codes and scalar RAM are classified/charged once outside4 host pools with per-member page upper. Disk reserves96MiB once for the shared80MiB row cache plus metadata; per-thread readers and pagecache remain in the illustrative8GiB margin.",
            "Sampler omitted with16MiB/device and16MiB/host per-rank margins. Driver/queue/collective/loader allowances are assumptions, not measurements."};
        report["completed"]=true;publish(output,report);std::cout<<"WROTE HYPOTHETICAL QFN8bit TP4 layout quote "<<output<<" selection="<<complete<<" trackedUnchanged="<<(before==after)<<"\n";
        return complete&&before==after?0:1;
    }catch(const std::exception& e){std::cerr<<"FAIL hypothetical QFN layout quote: "<<e.what()<<"\n";if(!output.empty()&&!fs::exists(output))try{report["completed"]=false;report["error"]=e.what();publish(output,report);}catch(...){}return 1;}
}
