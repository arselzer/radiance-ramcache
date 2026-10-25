// Explicit expensive REAL full-data check. No kernel instances/launches, Pools,
// Ctx, NativeInventory/Session or Engine activation. Parent-only invocation.
#define main preserved_stock_metadata_main
#include "stock/stock_plan.cpp"
#undef main
#include "main_bundle_reader.h"
#include "native_source.h"
namespace vm=glm53::verified_main;namespace sc=glm53::sharded;
namespace {
Plan metadata_layout(const Program& p,int rank){Config c;c.placement="all_vram";c.deterministic=true;c.vram_weights_mib=0;c.host_pool_mib=0;PlannerInput input{&p,&c,rank,4,nullptr,0,false,false};Plan result;good(Planner(input).plan(&result),"pure uncapped layout, not physical fit");return result;}
uint64_t fnv(std::span<const std::byte> bytes){uint64_t h=14695981039346656037ull;for(auto b:bytes){h^=std::to_integer<uint8_t>(b);h*=1099511628211ull;}return h;}
void verified(vm::Error e,const std::string& why){need(e==vm::Error::ok,why+": "+vm::name(e));}
void source_status(int e,const std::string& why){good(e,why);}
uint64_t decimal(const char* value){const std::string s=value;need(!s.empty()&&s.size()<=20&&std::all_of(s.begin(),s.end(),[](unsigned char c){return c>='0'&&c<='9';}),"bounded trusted complete bundle ID");const auto n=std::stoull(s);need(n,"nonzero trusted complete bundle ID");return n;}
}
int main(int argc,char** argv){try{
 need(argc==8&&!std::strcmp(argv[1],"--run-complete-reader"),"usage: glm53_complete_main_source_check --run-complete-reader MODEL MATCHING_AQL_HOME COMPLETE_BUNDLE PRODUCER_PROOF EXPECTED_ID NEW_REPORT_JSON");
 Options o;o.checkpoint=argv[2];o.home=argv[3];o.ctx=512;o.batch=1;o.output=argv[7];bounded(o.checkpoint);bounded(o.home);bounded(o.output);bounded(argv[4]);bounded(argv[5]);
 const auto source_root=fs::canonical(o.checkpoint),bundle_root=fs::canonical(argv[4]);const auto report=fs::canonical(fs::path(o.output).parent_path())/fs::path(o.output).filename();
 for(auto at=report;!at.empty();){need(at!=source_root&&at!=bundle_root,"report outside both immutable generations");const auto up=at.parent_path();if(up==at)break;at=up;}need(!fs::exists(report),"new report only");
 need(!rad::device_is_host()&&std::string(rad::device_backend_name())=="aql","actual AQL metadata selection; parent idle gate required");const auto before=tracked_allocations();RadDeviceProps props{};good(rad_dev_props(0,&props),"actual metadata eligibility card");need(!std::strcmp(props.arch,"gfx1201"),"actual9070XT DEVICE layout eligibility");
 auto checkpoint=std::make_unique<Checkpoint>();good(checkpoint->open(o.checkpoint),"actual pinned checkpoint metadata/map open");auto& registry=rad_tools_registry();registry.set_code_device(0);registry.set_no_device_code(false);registry.set_device_fact(RAD_FACT_P2P,0);LoadedPlugins plugins;
 good(rad_tools_load_plugins(o.home,{},checkpoint->meta().arch,checkpoint->meta().quant,&plugins,checkpoint->imports()!=Checkpoint::Import::None),"actual GLM metadata plugins");checkpoint->set_importers(&registry);const auto* architecture=registry.plugin(plugins.arch_plugin);need(architecture&&architecture->arch_declare&&architecture->arch_id=="glm5next","actual GLM declaration");auto meta=checkpoint->meta();if(checkpoint->imports()!=Checkpoint::Import::None){meta.quant=plugins.arch_quant;meta.finish();}
 auto source_cache=std::make_unique<CheckpointSources>(*checkpoint);
 std::array<RankQuote,4> graphs;std::array<Plan,4> plans;
 // Refuse actual metadata/kernel holes before the expensive full data scans.
 for(int rank=0;rank<4;++rank){graphs[size_t(rank)]=declare_rank(registry,*architecture,*checkpoint,*source_cache,meta.m,props,rank,o);need(graphs[size_t(rank)].device_complete,"actual rank DEVICE resolution complete, no model fit claim");plans[size_t(rank)]=metadata_layout(graphs[size_t(rank)].builder->program(),rank);}
 need(source_cache->refused().empty(),"actual ordinary/source encoding resolution before full scan");
 vm::Request request{bundle_root.string(),argv[5],decimal(argv[6])};vm::ReaderLimits caps;
 std::shared_ptr<const vm::VerifiedBundle> bundle;std::string why;const auto opened=vm::VerifiedBundle::open(request,*checkpoint,caps,&bundle,&why);verified(opened,"actual complete persisted reopen "+why);const auto receipt=bundle->receipt();need(receipt.complete_main_domain&&receipt.records==48384&&receipt.shards==168&&receipt.native_bytes_verified==161715585024ull&&receipt.raw_tensors==76108,"actual complete reopened data domain");
 std::array<std::shared_ptr<const vm::VerifiedRankMain>,4> ranks;
 std::array<std::shared_ptr<vm::GlmMainNativeSource>,4> sources;std::array<NativeRecordBatch,4> held;
 std::array<std::array<uint64_t,8>,4> checksums{};
 sc::ProcessLimits process_limits;process_limits.max_rank_owners=4;process_limits.max_open_fds=32;process_limits.max_metadata_bytes=1ull<<30;process_limits.max_cache_reservations=4ull*16*3342336;
 std::shared_ptr<sc::ProcessBudget> process;need(sc::ProcessBudget::create(process_limits,&process)==sc::Error::ok,"explicit common provider budget");
 constexpr std::array<uint32_t,8> ids{0,1,288,289,4321,6789,12094,12095};uint64_t cache_checked=0;
 Json rank_reports=Json::array();
 for(int rank=0;rank<4;++rank){auto& bound=ranks[size_t(rank)];const auto binding=bundle->bind_rank(*checkpoint,*source_cache,graphs[size_t(rank)].builder->program(),plans[size_t(rank)],&bound,&why);verified(binding,"once/rank real source+cuts binding "+why);need(bound->descriptors().size()==12096&&bound->owned().size()==12096&&bound->receipt().source_cut_bytes_hashed==40428896256ull,"complete all-owned/empty-resident source-cut CRC domain");
  sc::Limits limits;limits.hot_bytes=8ull*3342336;limits.cold_bytes=8ull*3342336;limits.max_record_bytes=3342336;limits.max_metadata_bytes=256u<<20;limits.max_records=12096;limits.max_batch=8;limits.max_shards=42;limits.max_segments=42;limits.max_payload_fds=2;limits.max_rank_fd_credits=12;limits.max_live_records=12096;limits.max_live_leases=64;limits.io_chunk_bytes=65536;
  auto& src=sources[size_t(rank)];const auto admitted=vm::GlmMainNativeSource::open(process,bound,limits,&src,&why);source_status(admitted,"opaque normalized actual source "+why);need(src->bootstrap(0,{})==RAD_E_UNSUPPORTED,"no fabricated fixed-resident bootstrap");need(src->acquire(ids,&held[size_t(rank)])<0,"setup trim/audit required before source acquisition");source_status(src->finish_bootstrap(),"vacuous bootstrap trim/audit");need(src->finish_bootstrap()==RAD_E_STATE,"setup completion exactly once");source_status(src->acquire(ids,&held[size_t(rank)]),"actual selected cache misses across layer shards");need(held[size_t(rank)].leases.size()==ids.size(),"exact returned union");
  for(size_t i=0;i<ids.size();++i){const auto& lease=held[size_t(rank)].leases[i];checksums[size_t(rank)][i]=src->catalog()[ids[i]].checksum;need(lease.record==ids[i]&&lease.bytes.size()==3342336&&lease.lifetime&&fnv(lease.bytes)==checksums[size_t(rank)][i],"actual cached native bytes and shared lease");cache_checked+=lease.bytes.size();}
  const auto first=src->stats();NativeRecordBatch repeated;const std::array<uint32_t,8> duplicate{ids[3],ids[0],ids[3],ids[7],ids[0],ids[7],ids[1],ids[1]};source_status(src->acquire(duplicate,&repeated),"actual HOT/weak repeat first-occurrence dedup");need(repeated.leases.size()==4&&repeated.leases[0].record==ids[3]&&repeated.leases[1].record==ids[0]&&repeated.leases[2].record==ids[7]&&repeated.leases[3].record==ids[1]&&src->stats().payload_read_bytes==first.payload_read_bytes,"repeat does not reread native payload");
  const auto stats=src->stats();rank_reports.push_back({{"rank",rank},{"records",12096},{"bundle_id",bound->generation()},{"source_cut_bytes_hashed",bound->receipt().source_cut_bytes_hashed},{"binding_peak_reservation",bound->receipt().metadata_reservation},{"binding_immutable_reservation",bound->receipt().immutable_metadata_reservation},{"shared_bundle_per_token_allowance",bound->receipt().retained_bundle_reservation},{"provider_metadata_reserved",stats.usage.metadata_reserved},{"hot_hits",stats.hot_hits},{"cold_hits",stats.cold_hits},{"misses",stats.misses},{"payload_read_bytes",stats.payload_read_bytes},{"cached_payload_fds",stats.cached_payload_fds}});
 }
 need(source_cache->refused().empty()&&tracked_allocations()==before,"no Core-tracked model allocation/kernel work");
 for(auto& src:sources)source_status(src->close(),"close new acquisitions, retain existing leases");for(auto& src:sources)src.reset();for(auto& bound:ranks)bound.reset();bundle.reset();
 // Raw Checkpoint/cache/Program owners actually die BEFORE retained byte leases.
 for(auto& graph:graphs)graph=RankQuote{};for(auto& plan:plans)plan=Plan{};source_cache.reset();checkpoint.reset();
 need(process->usage().rank_owners==4,"every returned lease retains same rank budget/authority after source close");
 for(size_t rank=0;rank<4;++rank)for(size_t i=0;i<8;++i){need(fnv(held[rank].leases[i].bytes)==checksums[rank][i],"leases survive actual source/checkpoint/Program destruction");cache_checked+=held[rank].leases[i].bytes.size();}
 for(auto& batch:held)batch=NativeRecordBatch{};const auto retired=process->usage();need(!retired.rank_owners&&!retired.open_fd_credits&&!retired.cache_reserved&&retired.metadata_reserved==256+4*128,"all actual provider/authority tokens retired");
 Json output{{"schema","glm53-verified-complete-main-source-check-v1"},{"passed",true},{"runtime_ready",false},{"Engine_or_Session_admitted",false},{"kernel_instances_or_launches",0},{"complete_native_bytes_verified",receipt.native_bytes_verified},{"raw_mapped_source_bytes_compared",receipt.mapped_source_bytes_compared},{"raw_tensors",receipt.raw_tensors},{"bundle_id",receipt.bundle_id},{"reader_peak_requested_reservation",receipt.metadata_reservation},{"reader_immutable_reservation",receipt.immutable_metadata_reservation},{"rank_reports",rank_reports},{"cache_bytes_compared_twice",cache_checked},{"process_metadata_limit",process_limits.max_metadata_bytes},{"shared_bundle_allowance_scope","conservatively charged per rank token, not four physical copies"},{"OS_page_cache_and_caller_metadata_RSS_excluded",true},{"actual_tracked_model_bytes_unchanged",true}};
 publish(report.string(),output);std::cout<<"PASS complete actual main DATA reader/source; no Engine/Session/GPU inference\n";return 0;
}catch(const std::exception& e){std::cerr<<"FAIL complete actual main DATA reader/source: "<<e.what()<<'\n';return 1;}}
