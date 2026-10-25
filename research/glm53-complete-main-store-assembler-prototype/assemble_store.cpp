#include "bundle_records.h"
#include "allocation_hooks.h"
#include <sys/stat.h>
#include <sys/statvfs.h>
#define main inherited_metadata_driver_not_called
#include "stock/stock_plan.cpp"
#undef main
namespace mb=glm53::main_bundle;
namespace {
uint64_t assertions=0,declared_graphs=0,geometry_quotes=0,compared_records=0,compared_bytes=0,reads=0,writes=0,triggers=0;
uint64_t positive_assembly_verify_reads=0,positive_assembly_verify_writes=0;
int fault=0;bool segment_written=false,manifest_written=false;std::string source_index;
Json kv_b_metadata=Json::array();bool kv_b_already_bf16=true;
void check(bool ok,const std::string& why){++assertions;if(!ok)throw std::runtime_error(why);}
void status(mb::Status s,const std::string& why){check(s==mb::Status::ok,why+": "+glm53::main_records::name(s));}
void metadata_evidence(const RankQuote& graph){const auto& casts=graph.json.at("planned_recipe_casts");check(casts.is_array()&&casts.size()==11,"actual inherited nonexpert metadata-hook coverage");
    if(kv_b_metadata.empty())kv_b_metadata=casts;else check(kv_b_metadata==casts,"actual raw nonexpert metadata stable across declarations");
    for(const auto& item:casts)kv_b_already_bf16=kv_b_already_bf16&&item.at("raw_encoding")==item.at("planned_encoding")&&item.at("planned_encoding")=="bf16";
}
struct Stage {
    std::string path;bool published=false;
    explicit Stage(const std::string& destination){auto pattern=(fs::path(destination).parent_path()/(fs::path(destination).filename().string()+".validation.XXXXXX")).string();std::vector<char> name(pattern.begin(),pattern.end());name.push_back(0);const auto* p=::mkdtemp(name.data());check(p,"private checker root");try{path=p;}catch(...){::rmdir(p);throw;}}
    ~Stage(){if(!published){std::error_code error;fs::remove_all(path,error);}}
};
struct OwnedFd {int n=-1;explicit OwnedFd(int fd):n(fd){}~OwnedFd(){if(n>=0)::close(n);}OwnedFd(const OwnedFd&)=delete;};
std::string raw_name(int layer,int expert,int projection,int plane){char name[160];const int n=std::snprintf(name,sizeof name,"model.language_model.layers.%d.mlp.experts.%d.%s_proj.weight%s",layer,expert,projection==0?"gate":projection==1?"up":"down",plane?"_scale":"");check(n>0&&size_t(n)<sizeof name,"oracle source name cap");return name;}
std::vector<std::byte> oracle(rad::Checkpoint& ck,const mb::Record& record){
    constexpr int64_t offsets[]{0,2097152,2228224,3276800};constexpr int64_t sizes[]{2097152,131072,1048576,65536};
    const auto& key=record.native.key;check(key.world==4&&key.rank<4&&key.layer>=3&&key.layer<=44&&key.expert>=0&&key.expert<288&&record.native.length==3342336,"independent canonical oracle key/size");
    std::vector<std::byte> expected(3342336,std::byte{0});
    for(int m=0;m<4;++m){check(record.native.member_offsets[size_t(m)]==uint64_t(offsets[m])&&record.native.member_bytes[size_t(m)]==uint64_t(sizes[m]),"independent oracle four canonical member positions");
        const bool gu=m<2;const int plane=m%2;const int parts=gu?2:1;const int64_t width=gu?(plane?128:2048):(plane?64:1024),take=gu?width:(plane?16:256),rows=gu?512:4096;
        for(int part=0;part<parts;++part){const auto* tensor=ck.find(raw_name(key.layer,key.expert,gu?part:2,plane));check(tensor&&tensor->data&&tensor->rad_dtype==RAD_U8&&tensor->shape==std::vector<int64_t>({gu?2048:4096,width})&&tensor->bytes==(gu?2048:4096)*width,"independent physical tensor extents");
            for(int64_t r=0;r<rows;++r){const auto from=(r+(gu?int64_t(key.rank)*512:0))*width+(gu?0:int64_t(key.rank)*take);const auto to=offsets[m]+(int64_t(part)*rows+r)*take;
                check(from>=0&&take<=tensor->bytes-from&&to>=0&&take<=int64_t(expected.size())-to,"independent physical oracle row/column bounds");
                std::memcpy(expected.data()+to,static_cast<const std::byte*>(tensor->data)+from,size_t(take));}
        }
    }
    return expected;
}
void exact_records(const std::string& root,rad::Checkpoint& ck,const mb::PublishedBundle& bundle){
    for(const auto& record:bundle.records()){check(record.shard<bundle.shards().size(),"published record shard ordinal");const auto path=fs::path(root)/bundle.shards()[record.shard].name/"segment-00000000.bin";
        OwnedFd fd(::open(path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW));check(fd.n>=0,"independent native record descriptor");
        std::vector<std::byte> actual(size_t(record.native.length));size_t done=0;
        while(done<actual.size()){const auto n=::pread(fd.n,actual.data()+done,actual.size()-done,off_t(record.native.offset+done));if(n<0&&errno==EINTR)continue;if(n<=0)throw std::runtime_error("independent complete native read");done+=size_t(n);}
        const auto expected=oracle(ck,record);check(actual==expected,"every native byte versus independent physical oracle");++compared_records;compared_bytes+=actual.size();
    }
}
bool suffix_fd(int fd,const char* suffix){char link[64],target[4097];const int n=std::snprintf(link,sizeof link,"/proc/self/fd/%d",fd);if(n<=0||size_t(n)>=sizeof link)return false;const auto used=::readlink(link,target,sizeof target-1);const auto count=std::strlen(suffix);return used>=0&&size_t(used)>=count&&!std::memcmp(target+used-count,suffix,count);}
bool index_fd(int fd){char link[64],target[4097];const int n=std::snprintf(link,sizeof link,"/proc/self/fd/%d",fd);if(n<=0||size_t(n)>=sizeof link)return false;const auto used=::readlink(link,target,sizeof target-1);return used>=0&&size_t(used)==source_index.size()&&!std::memcmp(target,source_index.data(),source_index.size());}
void clean_staging(const std::string& root){for(const auto& entry:fs::directory_iterator(root))check(entry.path().filename().string().find(".bundle.")==std::string::npos,"failed assembler staging cleanup");}
rad::Plan layout_plan(rad::Program& program,int rank){auto cfg=config(512,128);cfg.placement="all_vram";cfg.vram_weights_mib=0;cfg.host_pool_mib=0;rad::PlannerInput input{&program,&cfg,rank,4,nullptr,0,false,false};rad::Plan plan;good(rad::Planner(input).plan(&plan),"actual uncapped native-layout Plan, not fit");return plan;}
void fault_cases(rad::Checkpoint& ck,rad::CheckpointSources& sources,rad::Program& program,const rad::Plan& plan,
                 const std::string& proof,const std::string& root){
    for(int which:{1,2,3}){const auto path=root+"/fault-"+std::to_string(which);std::unique_ptr<mb::Assembler> assembler;status(mb::Assembler::begin(ck,proof,mb::Mode::SubsetComponent,path,{},&assembler),"fault assembly begin");
        segment_written=false;manifest_written=false;const auto before=triggers;fault=which;const auto result=assembler->append_rank(ck,sources,program,plan);fault=0;++geometry_quotes;
        check(result==mb::Status::io&&triggers==before+1&&segment_written,"intended exact late native I/O failure trigger");if(which==3)check(manifest_written,"final source audit error is AFTER real layer manifest write");
        mb::PublishedBundle prior;check(assembler->finish(&prior)!=mb::Status::ok&&prior.records().empty()&&!fs::exists(path),"failed rank cannot publish a subset bundle");assembler.reset();clean_staging(root);
    }
    const auto never=root+"/never";std::unique_ptr<mb::Assembler> absent;const auto r0=reads,w0=writes;
    allocation_fault::remaining=0;const auto oom=mb::Assembler::begin(ck,proof,mb::Mode::SubsetComponent,never,{},&absent);const bool triggered=allocation_fault::remaining==-1;allocation_fault::remaining=-1;
    check(triggered&&oom==mb::Status::no_memory&&!absent&&reads==r0&&writes==w0&&!fs::exists(never),"first allocation failure before file I/O preserves output");
    auto small=mb::Limits{};small.max_metadata_bytes=1;check(mb::Assembler::begin(ck,proof,mb::Mode::SubsetComponent,never,small,&absent)==mb::Status::budget&&reads==r0&&writes==w0&&!absent,"metadata refusal before I/O");
    check(mb::Assembler::begin(ck,proof,mb::Mode(99),never,{},&absent)==mb::Status::unsupported&&!absent&&reads==r0&&writes==w0,"invalid mode closed before I/O");
    status(mb::Assembler::begin(ck,proof,mb::Mode::SubsetComponent,never,{},&absent),"owner mismatch begin");rad::Checkpoint foreign;
    const auto before=reads;check(absent->append_rank(foreign,sources,program,plan)==mb::Status::identity&&reads==before,"foreign unopened source owner refuses before payload read");absent.reset();clean_staging(root);
    status(mb::Assembler::begin(ck,proof,mb::Mode::SubsetComponent,never,{},&absent),"omitted ranks begin");mb::PublishedBundle prior;
    check(absent->finish(&prior)==mb::Status::identity&&prior.records().empty()&&!fs::exists(never),"omitted ranks never finalize");absent.reset();clean_staging(root);
    auto full=mb::Limits{};full.max_records=48384;full.max_total_bytes=161715585024ull;
    const auto full_start=mb::Assembler::begin(ck,proof,mb::Mode::CompleteMainData,never,full,&absent);
    if(full_start==mb::Status::identity)check(!absent&&!fs::exists(never),"subset producer transcript cannot begin complete main assembly");
    else{status(full_start,"a valid complete producer transcript may begin only");check(bool(absent)&&absent->finish(&prior)==mb::Status::identity&&prior.records().empty()&&!fs::exists(never),"even complete producer proof needs all four complete catalogs");absent.reset();clean_staging(root);}
}
mb::PublishedBundle assemble(rad::Registry& registry,const rad::Plugin& architecture,rad::Checkpoint& ck,rad::CheckpointSources& sources,
                             const RadModelMeta& meta,const RadDeviceProps& eligibility,const Options& options,const std::string& proof,
                             const std::string& output,mb::Mode mode,const mb::Limits& limits,bool collision){
    const auto transaction_reads=reads,transaction_writes=writes;
    std::unique_ptr<mb::Assembler> assembler;std::string why;status(mb::Assembler::begin(ck,proof,mode,output,limits,&assembler,&why),"bundle begin "+why);
    for(int rank=0;rank<4;++rank){auto graph=declare_rank(registry,architecture,ck,sources,meta,eligibility,rank,options);++declared_graphs;metadata_evidence(graph);check(graph.device_complete,"actual complete DEVICE selection for main source/layout");auto plan=layout_plan(graph.builder->program(),rank);
        const auto packed=assembler->append_rank(ck,sources,graph.builder->program(),plan,&why);++geometry_quotes;status(packed,"once/rank complete source/data append "+why);}
    const auto prior_triggers=triggers;segment_written=true;if(collision)fault=4;mb::PublishedBundle output_data;const auto done=assembler->finish(&output_data,&why);fault=0;
    if(collision){check(done==mb::Status::exists&&triggers==prior_triggers+1&&output_data.records().empty(),"actual final assembler NOREPLACE collision refuses publication");
        check(fs::is_regular_file(fs::path(output)/"marker"),"late destination marker preserved");check(::unlink((output+"/marker").c_str())==0&&::rmdir(output.c_str())==0,"task-owned collision marker cleanup");assembler.reset();clean_staging(fs::path(output).parent_path().string());return {};}
    status(done,"complete final bundle commit "+why);const auto receipt=output_data.receipt();check(receipt.graph_quotes==4&&receipt.records==(mode==mb::Mode::CompleteMainData?48384u:8u)&&receipt.total_bytes==receipt.records*3342336ull&&receipt.total_bytes==receipt.selected_copy_bytes&&receipt.total_bytes==receipt.complete_tensor_read_bytes,"exact data/quote/byte-domain receipt closure");
    check(receipt.complete_main_domain==(mode==mb::Mode::CompleteMainData)&&output_data.layouts().size()==4&&receipt.peak_input_fds<=2&&receipt.io_window_bytes==65536,"explicit complete-vs-component data profile");
    assembler.reset(); // close broker2+staging FD/window BEFORE the independent verifier owns6FDs.
    const auto r0=reads,w0=writes;auto small=limits;small.max_metadata_bytes=1;check(mb::verify(output,output_data,small)==mb::Status::budget&&reads==r0&&writes==w0,"verifier metadata refusal before reads/allocations");
    small=limits;check(receipt.metadata_reserved>1,"actual verifier metadata reservation");small.max_metadata_bytes=receipt.metadata_reserved-1;
    check(mb::verify(output,output_data,small)==mb::Status::budget&&reads==r0&&writes==w0,"actual verifier metadata-minus-one refuses before I/O");
    status(mb::verify(output,output_data,limits,&why),"strict complete data verification "+why);
    positive_assembly_verify_reads=reads-transaction_reads;positive_assembly_verify_writes=writes-transaction_writes;return output_data;
}
void tamper_case(const std::string& root,const mb::PublishedBundle& bundle){const auto path=fs::path(root)/bundle.shards()[0].name/"segment-00000000.bin";const int fd=::open(path.c_str(),O_RDWR|O_CLOEXEC|O_NOFOLLOW);check(fd>=0,"task-owned tamper descriptor");
    uint8_t original=0;check(::pread(fd,&original,1,0)==1,"task-owned original byte");const uint8_t changed=original^1;check(::pwrite(fd,&changed,1,0)==1&&!::fsync(fd),"task-only corrupt native byte");
    check(mb::verify(root,bundle,{})==mb::Status::checksum,"late corrupted record refuses strict verification");check(::pwrite(fd,&original,1,0)==1&&!::fsync(fd)&&!::close(fd),"task-only exact restore");status(mb::verify(root,bundle,{}),"good-bad-good actual record verification");}
}
extern "C" ssize_t __real_pread(int,void*,size_t,off_t);
extern "C" ssize_t __real_pwrite(int,const void*,size_t,off_t);
extern "C" int __real_fstat(int,struct stat*);
extern "C" int __real_renameat2(int,const char*,int,const char*,unsigned);
extern "C" ssize_t __wrap_pread(int fd,void* data,size_t n,off_t at){++reads;if(fault==1&&segment_written){fault=0;++triggers;errno=EIO;return -1;}return __real_pread(fd,data,n,at);}
extern "C" ssize_t __wrap_pwrite(int fd,const void* data,size_t n,off_t at){++writes;if(fault==2&&segment_written){fault=0;++triggers;errno=EIO;return -1;}const auto result=__real_pwrite(fd,data,n,at);if(result>0){if(suffix_fd(fd,"/segment-00000000.bin"))segment_written=true;if(suffix_fd(fd,"/manifest.json"))manifest_written=true;}return result;}
extern "C" int __wrap_fstat(int fd,struct stat* value){const auto result=__real_fstat(fd,value);if(!result&&fault==3&&manifest_written&&index_fd(fd)){fault=0;++triggers;errno=EIO;return -1;}return result;}
extern "C" int __wrap_renameat2(int a,const char* from,int b,const char* to,unsigned flags){if(fault==4&&segment_written){fault=0;++triggers;if(::mkdir(to,0700))return -1;const auto marker=std::string(to)+"/marker";const int fd=::open(marker.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);if(fd<0)return -1;(void)::write(fd,"original",8);::close(fd);}return __real_renameat2(a,from,b,to,flags);}
int main(int argc,char** argv){try{
    const bool complete=argc>1&&!std::strcmp(argv[1],"--assemble-complete-main-bundle");
    check((!complete&&argc==10&&!std::strcmp(argv[1],"--run-subset-validation"))||(complete&&argc==12),"explicit subset or separately scheduled complete assembler CLI");
    check(!std::strcmp(argv[2],"--checkpoint")&&!std::strcmp(argv[4],"--home")&&!std::strcmp(argv[6],"--source-proof")&&!std::strcmp(argv[8],"--output"),"fixed bounded CLI fields");
    Options options;options.checkpoint=argv[3];options.home=argv[5];options.output=argv[9];options.ctx=512;options.batch=128;const std::string proof=argv[7];bounded(options.checkpoint);bounded(options.home);bounded(options.output);bounded(proof);
    mb::Limits limits;const auto mode=complete?mb::Mode::CompleteMainData:mb::Mode::SubsetComponent;
    if(complete){check(!std::strcmp(argv[10],"--max-native-bytes"),"explicit complete native payload cap");const std::string n=argv[11];check(!n.empty()&&n.size()<=12&&std::all_of(n.begin(),n.end(),[](unsigned char c){return c>='0'&&c<='9';}),"bounded decimal complete cap");limits.max_records=48384;limits.max_total_bytes=std::stoull(n);check(limits.max_total_bytes>=161715585024ull&&limits.max_total_bytes<=200ull<<30,"complete native-byte cap (metadata separate)");}
    rad::Checkpoint checkpoint;good(checkpoint.open(options.checkpoint),"actual complete checkpoint metadata open");const auto source_root=fs::canonical(checkpoint.path());const auto output=fs::canonical(fs::path(options.output).parent_path())/fs::path(options.output).filename();
    for(auto at=output;!at.empty();){check(at!=source_root,"checker output outside original source BEFORE staging");const auto up=at.parent_path();if(up==at)break;at=up;}options.output=output.string();check(!fs::exists(output),"new final task output");source_index=(source_root/"model.safetensors.index.json").string();
    if(complete){struct statvfs space{};check(!::statvfs(output.parent_path().c_str(),&space)&&space.f_frsize&&uint64_t(space.f_bavail)<=UINT64_MAX/uint64_t(space.f_frsize),"full output checked free-space preflight");const uint64_t available=uint64_t(space.f_bavail)*uint64_t(space.f_frsize);check(available>=161715585024ull+169ull*limits.max_manifest_bytes+limits.max_proof_bytes+(32ull<<30),"native+bounded metadata+32GiB disk reserve");}
    check(!rad::device_is_host()&&std::string(rad::device_backend_name())=="aql","actual AQL metadata selection only");const auto before=tracked_allocations();RadDeviceProps eligibility{};good(rad_dev_props(0,&eligibility),"actual metadata eligibility card");check(!std::strcmp(eligibility.arch,"gfx1201"),"actual target native-reader eligibility");
    auto& registry=rad::rad_tools_registry();registry.set_code_device(0);registry.set_no_device_code(false);registry.set_device_fact(RAD_FACT_P2P,0);rad::LoadedPlugins plugins;good(rad::rad_tools_load_plugins(options.home,{},checkpoint.meta().arch,checkpoint.meta().quant,&plugins,checkpoint.imports()!=rad::Checkpoint::Import::None),"actual stock plugins");checkpoint.set_importers(&registry);
    const auto* architecture=registry.plugin(plugins.arch_plugin);check(architecture&&architecture->arch_declare&&architecture->arch_id=="glm5next","actual pinned main GLM architecture");auto meta=checkpoint.meta();if(checkpoint.imports()!=rad::Checkpoint::Import::None){meta.quant=plugins.arch_quant;meta.finish();}rad::CheckpointSources sources(checkpoint);Stage stage(options.output);
    if(!complete){auto graph=declare_rank(registry,*architecture,checkpoint,sources,meta.m,eligibility,0,options);++declared_graphs;metadata_evidence(graph);check(graph.device_complete,"actual rank0 fault fixture DEVICE rows");auto plan=layout_plan(graph.builder->program(),0);fault_cases(checkpoint,sources,graph.builder->program(),plan,proof,stage.path);}
    if(!complete)(void)assemble(registry,*architecture,checkpoint,sources,meta.m,eligibility,options,proof,stage.path+"/collision",mode,limits,true);
    const auto data=stage.path+"/data";auto publication=assemble(registry,*architecture,checkpoint,sources,meta.m,eligibility,options,proof,data,mode,limits,false);exact_records(data,checkpoint,publication);
    if(!complete){tamper_case(data,publication);std::unique_ptr<mb::Assembler> duplicate;const auto r0=reads,w0=writes;check(mb::Assembler::begin(checkpoint,proof,mode,data,limits,&duplicate)==mb::Status::exists&&!duplicate&&reads==r0&&writes==w0,"existing generation never overwritten");
        status(mb::Assembler::begin(checkpoint,proof,mode,stage.path+"/incomplete",limits,&duplicate),"prior-publication preservation setup");auto retained=publication;const auto id=retained.receipt().bundle_id;
        check(duplicate->finish(&retained)==mb::Status::identity&&retained.receipt().bundle_id==id&&retained.records().size()==8,"failed finish preserves caller's prior publication");duplicate.reset();clean_staging(stage.path);}
    check(sources.refused().empty()&&tracked_allocations()==before,"no source resolution refusals or Core-tracked model allocations");const auto receipt=publication.receipt();check(compared_records==receipt.records&&compared_bytes==receipt.total_bytes,"independent oracle covers EVERY published record byte");if(!complete)check(triggers==4&&declared_graphs==9&&geometry_quotes==11,"honest subset fault/declaration/quote counters");
    Json report{{"schema",complete?"glm53-main-store-assembler-complete-check-v1":"glm53-main-store-assembler-subset-check-v1"},{"passed",true},{"runtime_ready",false},
        {"mode",complete?"complete_main_data":"subset_component_NONREADY"},{"records",receipt.records},{"shards",receipt.shards},{"native_bytes",receipt.total_bytes},{"bundle_id",receipt.bundle_id},{"graph_quotes",receipt.graph_quotes},
        {"total_graph_declarations",declared_graphs},{"total_geometry_quotes_including_faults",geometry_quotes},{"checks",assertions+1},{"fault_triggers",triggers},{"compared_records",compared_records},{"compared_native_bytes",compared_bytes},
        {"metadata_requested_peak",receipt.metadata_reserved},{"metadata_on_disk_actual",receipt.metadata_written_bytes},{"metadata_on_disk_upper",receipt.metadata_output_upper_bytes},
        {"own_complete_tensor_read_bytes",receipt.complete_tensor_read_bytes},{"own_source_band_read_bytes",receipt.source_band_read_bytes},{"native_copy_bytes",receipt.selected_copy_bytes},
        {"actual_total_checker_pread_calls_before_receipt",reads},{"actual_total_checker_pwrite_calls_before_receipt",writes},
        {"positive_assembly_and_verify_pread_calls",positive_assembly_verify_reads},{"positive_assembly_and_verify_pwrite_calls",positive_assembly_verify_writes},
        {"producer_source_proof_hashed_bytes",receipt.source_proof_hashed_bytes},{"trusted_source_proof_complete",receipt.source_proof_complete},{"full_checkpoint_payload_hashes_independently_computed_by_Cpp",false},
        {"input_payload_fd_peak",receipt.peak_input_fds},{"assembler_owned_fd_upper",8},{"verifier_owned_fds",6},{"assembler_destroyed_before_verifier",true},{"window_bytes",receipt.io_window_bytes},
        {"upstream_Checkpoint_Program_metadata_and_mapping_owners_separate",true},{"fixture_independent_oracle_buffers_upper",8u<<20},
        {"nonexpert_kv_b_metadata_scope","inherited declaration hook only; actual raw/planned encodings recorded, no payload conversion/load"},
        {"nonexpert_kv_b_metadata_evidence",kv_b_metadata},{"nonexpert_kv_b_all_raw_already_planned_BF16",kv_b_already_bf16},
        {"model_kernel_or_Engine_execution",false},{"content_ID_excludes_runtime_ctx_batch_residency_path_inode_stat",true},{"future_SourceFactory_still_absent",true}};
    publish(stage.path+"/receipt.json",report);check(::renameat2(AT_FDCWD,stage.path.c_str(),AT_FDCWD,options.output.c_str(),1u)==0,"outer validation NOREPLACE commit");stage.published=true;
    std::cout<<"PASS GLM main store data records="<<receipt.records<<" nativeBytes="<<receipt.total_bytes<<" checks="<<assertions<<" faults="<<triggers<<" runtimeReady=false\n";return 0;
}catch(const std::exception& e){fault=0;allocation_fault::remaining=-1;std::cerr<<"FAIL main store assembler: "<<e.what()<<" checks="<<assertions<<"\n";return 1;}}
