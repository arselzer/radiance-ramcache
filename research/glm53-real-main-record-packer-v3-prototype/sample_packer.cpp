#include "main_records.h"
#include "allocation_hooks.h"
#include <sys/stat.h>
// Reuse byte-identical real stock declaration/source/Planner helpers. Its old
// main is renamed and NEVER called; the explicit payload gate below is separate.
#define main inherited_metadata_only_main_not_called
#include "stock/stock_plan.cpp"
#undef main
namespace mr=glm53::main_records;
namespace {
uint64_t checks=0,compared_bytes=0,failures_seen=0,reads=0,writes=0;
int fault=0;bool wrote=false;
bool manifest_written=false;std::string source_index_for_fault;
void check(bool value,const char* why){++checks;if(!value)throw std::runtime_error(why);}
void packed(mr::Status s,const char* why){check(s==mr::Status::ok,why);}
struct Stage {
    std::string path;bool published=false;
    explicit Stage(const std::string& destination){const fs::path dest(destination);std::string pattern=(dest.parent_path()/(dest.filename().string()+".sample.XXXXXX")).string();std::vector<char> name(pattern.begin(),pattern.end());name.push_back(0);
        char* p=::mkdtemp(name.data());check(p,"task sample staging root");try{path=p;}catch(...){::rmdir(p);throw;}}
    ~Stage(){if(published||path.empty())return;std::error_code error;fs::remove_all(path,error);}
};
std::string raw(int layer,int expert,int projection,int plane){char name[160];const int n=std::snprintf(name,sizeof name,"model.language_model.layers.%d.mlp.experts.%d.%s_proj.weight%s",layer,expert,projection==0?"gate":projection==1?"up":"down",plane?"_scale":"");check(n>0&&size_t(n)<sizeof name,"oracle name bound");return name;}
std::vector<std::byte> oracle(rad::Checkpoint& ck,const rad::Plan& plan,int32_t unit,int rank){
    const auto& u=plan.units[size_t(unit)];check(u.weights.size()==4&&u.offset.size()==4&&u.slot_bytes==3342336,"independent actual main oracle geometry");
    std::vector<std::byte> expected(size_t(u.slot_bytes),std::byte{0});
    for(int member=0;member<4;++member){const bool gu=member<2;const int plane=member%2;const int parts=gu?2:1;
        const int64_t raw_width=gu?(plane?128:2048):(plane?64:1024),take=gu?raw_width:(plane?16:256),rows=gu?512:4096;
        for(int part=0;part<parts;++part){const auto name=raw(u.layer,u.expert,gu?part:2,plane);const auto* tensor=ck.find(name);
            check(tensor&&tensor->data&&tensor->rad_dtype==RAD_U8&&tensor->shape==std::vector<int64_t>({gu?2048:4096,raw_width})&&tensor->bytes==(gu?2048:4096)*raw_width,"independent actual raw source tensor shape/extent");
            for(int64_t row=0;row<rows;++row){const auto from=(row+(gu?int64_t(rank)*512:0))*raw_width+(gu?0:int64_t(rank)*take);
                const auto to=u.offset[size_t(member)]+(int64_t(part)*rows+row)*take;
                check(from>=0&&take<=tensor->bytes-from&&to>=0&&take<=int64_t(expected.size())-to,"independent oracle byte range");
                std::memcpy(expected.data()+to,static_cast<const std::byte*>(tensor->data)+from,size_t(take));}
        }
    }
    return expected;
}
void compare_reader(mr::MainReader& reader,rad::Checkpoint& ck,const rad::Plan& plan,int32_t unit,int rank){
    const auto info=reader.records()[0];std::vector<std::byte> actual;packed(reader.read(info.key,&actual),"actual verified one-record read");
    const auto expected=oracle(ck,plan,unit,rank);check(actual==expected,"all actual native bytes versus independent raw row/column oracle");compared_bytes+=actual.size();
    std::vector<std::byte> preserved(3,std::byte{0x67});const auto* pointer=preserved.data();auto bad=info.key;bad.rank=(bad.rank+1)%4;
    check(reader.read(bad,&preserved)==mr::Status::identity&&preserved.size()==3&&preserved.data()==pointer&&preserved[0]==std::byte{0x67},"foreign-rank read preserves caller");
}
void fault_cases(rad::Checkpoint& ck,const rad::CheckpointSources& sources,const rad::Program& program,const rad::Plan& plan,int32_t unit,
                 const std::string& stage,const mr::PublishedRecords& original){
    source_index_for_fault=fs::canonical(ck.path()).string()+"/model.safetensors.index.json";
    for(int which=1;which<=4;++which){mr::PublishedRecords output=original;const auto retained=output.receipt().manifest_checksum;const std::string path=stage+"/fault-"+std::to_string(which);
        wrote=false;manifest_written=false;fault=which;const auto before=failures_seen;const auto status=mr::pack_main_records(ck,sources,program,plan,{&unit,1},path,mr::Limits{},&output);fault=0;
        check(status==(which==3?mr::Status::exists:mr::Status::io)&&failures_seen==before+1&&wrote,"intended actual late I/O/publication refusal witnessed");
        check(output.receipt().manifest_checksum==retained,"late failure preserves independent publication owner");
        if(which==4)check(manifest_written,"semantic source-index stat refusal occurs AFTER real manifest write");
        if(which==3){check(fs::is_regular_file(path+"/marker"),"late rename collision preserves real destination");check(::unlink((path+"/marker").c_str())==0&&::rmdir(path.c_str())==0,"collision fixture cleanup");}
        else check(!fs::exists(path),"failed actual output was not published");
        for(const auto& entry:fs::directory_iterator(stage))check(entry.path().filename().string().find(".staging.")==std::string::npos,"owned staging directory cleaned after late refusal");
    }
    mr::PublishedRecords output=original;const std::string oom_path=stage+"/fault-oom";const auto before_reads=reads,before_writes=writes;
    allocation_fault::remaining=0;const auto oom=mr::pack_main_records(ck,sources,program,plan,{&unit,1},oom_path,mr::Limits{},&output);const bool triggered=allocation_fault::remaining==-1;allocation_fault::remaining=-1;
    check(triggered&&oom==mr::Status::no_memory&&reads==before_reads&&writes==before_writes&&!fs::exists(oom_path)&&output.receipt().manifest_checksum==original.receipt().manifest_checksum,"actual first C++ allocation refusal before source I/O/publication");
    auto small=mr::Limits{};small.max_metadata_bytes=1;
    check(mr::pack_main_records(ck,sources,program,plan,{&unit,1},oom_path,small,&output)==mr::Status::budget&&reads==before_reads&&writes==before_writes&&!fs::exists(oom_path),"metadata budget refusal before source I/O");
    auto config_small=mr::Limits{};const auto actual_config=fs::file_size(fs::path(ck.path())/"config.json");check(actual_config>1&&actual_config<=config_small.max_config_bytes,"actual source config bounded by derived2MiB limit");config_small.max_config_bytes=size_t(actual_config-1);
    check(mr::pack_main_records(ck,sources,program,plan,{&unit,1},oom_path,config_small,&output)==mr::Status::budget&&!fs::exists(oom_path)&&output.receipt().manifest_checksum==original.receipt().manifest_checksum,"actual config-size-minus-one refusal preserves publication");
    const std::string collision=stage+"/rank-0";check(mr::pack_main_records(ck,sources,program,plan,{&unit,1},collision,mr::Limits{},&output)==mr::Status::exists,"existing generation NOREPLACE refusal");
}
} // namespace
extern "C" ssize_t __real_pread(int,void*,size_t,off_t);
extern "C" ssize_t __real_pwrite(int,const void*,size_t,off_t);
extern "C" int __real_renameat2(int,const char*,int,const char*,unsigned);
extern "C" int __real_fstat(int,struct stat*);
namespace {
bool fd_has_path(int fd,const std::string& expected){char link[64],target[4097];const int length=std::snprintf(link,sizeof link,"/proc/self/fd/%d",fd);if(length<=0||size_t(length)>=sizeof link)return false;
    const auto n=::readlink(link,target,sizeof target-1);return n>=0&&size_t(n)==expected.size()&&!std::memcmp(target,expected.data(),size_t(n));}
bool manifest_fd(int fd){char link[64],target[4097];const int length=std::snprintf(link,sizeof link,"/proc/self/fd/%d",fd);if(length<=0||size_t(length)>=sizeof link)return false;
    const auto n=::readlink(link,target,sizeof target-1);constexpr const char suffix[]="/manifest.json";return n>=ssize_t(sizeof suffix-1)&&!std::memcmp(target+n-(sizeof suffix-1),suffix,sizeof suffix-1);}
}
extern "C" ssize_t __wrap_pread(int fd,void* data,size_t bytes,off_t offset){++reads;if(fault==1&&wrote){fault=0;++failures_seen;errno=EIO;return -1;}return __real_pread(fd,data,bytes,offset);}
extern "C" ssize_t __wrap_pwrite(int fd,const void* data,size_t bytes,off_t offset){++writes;if(fault==2&&wrote){fault=0;++failures_seen;errno=EIO;return -1;}const auto result=__real_pwrite(fd,data,bytes,offset);if(result>0){wrote=true;if(fault==4&&manifest_fd(fd))manifest_written=true;}return result;}
extern "C" int __wrap_fstat(int fd,struct stat* out){const int status=__real_fstat(fd,out);
    // Semantic returned-error seam after an ACTUAL successful index stat and
    // manifest pwrite. Never modify real checkpoint bytes/inodes or fabricate a
    // "complete clone". This tests final audit presence, not a source-mutation claim.
    if(status==0&&fault==4&&manifest_written&&fd_has_path(fd,source_index_for_fault)){fault=0;++failures_seen;errno=EIO;return -1;}return status;
}
extern "C" int __wrap_renameat2(int from,const char* oldpath,int to,const char* newpath,unsigned flags){
    if(fault==3&&wrote){fault=0;++failures_seen;if(::mkdir(newpath,0700))return -1;const std::string marker=std::string(newpath)+"/marker";const int fd=::open(marker.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);if(fd<0)return -1;(void)::write(fd,"original",8);::close(fd);}
    return __real_renameat2(from,oldpath,to,newpath,flags);
}
int main(int argc,char** argv){try{
    check(argc==8&&!std::strcmp(argv[1],"--run-complete-record-pack")&&!std::strcmp(argv[2],"--checkpoint")&&!std::strcmp(argv[4],"--home")&&!std::strcmp(argv[6],"--output"),"usage: --run-complete-record-pack --checkpoint ABS_COMPLETE_PINNED_CHECKPOINT --home ABS_SHIPPED_HOME --output ABS_NEW_SAMPLE_DIR");
    Options o;o.checkpoint=argv[3];o.home=argv[5];o.output=argv[7];o.ctx=512;o.batch=128;bounded(o.checkpoint);bounded(o.home);bounded(o.output);check(!fs::exists(o.output),"NEW output generation only");
    check(!rad::device_is_host()&&std::string(rad::device_backend_name())=="aql","actual AQL DEVICE metadata selection required; no HOST/synthetic eligibility rows");
    const auto tracked_before=tracked_allocations();RadDeviceProps eligibility{};good(rad_dev_props(0,&eligibility),"actual eligibility device query");check(!std::strcmp(eligibility.arch,"gfx1201"),"actual gfx1201 code eligibility");
    rad::Checkpoint checkpoint;good(checkpoint.open(o.checkpoint),"actual complete pinned checkpoint open");rad::Registry& registry=rad::rad_tools_registry();registry.set_code_device(0);registry.set_no_device_code(false);registry.set_device_fact(RAD_FACT_P2P,0);
    rad::LoadedPlugins plugins;good(rad::rad_tools_load_plugins(o.home,{},checkpoint.meta().arch,checkpoint.meta().quant,&plugins,checkpoint.imports()!=rad::Checkpoint::Import::None),"actual stock plugin/kernel metadata");checkpoint.set_importers(&registry);
    const auto* architecture=registry.plugin(plugins.arch_plugin);check(architecture&&architecture->arch_declare&&architecture->arch_id=="glm5next","actual stock GLM architecture");
    auto meta=checkpoint.meta();if(checkpoint.imports()!=rad::Checkpoint::Import::None){meta.quant=plugins.arch_quant;meta.finish();}
    rad::CheckpointSources sources(checkpoint);Stage stage(o.output);Json report{{"schema","glm53-real-one-expert-all4logicalrank-record-proof-v1"},{"checkpoint",o.checkpoint},{"actual_raw_tensor_reads",true},{"actual_source_full_checkpoint_SHA_verified",false},
        {"one_complete_expert_per_rank",true},{"logical_ranks",4},{"layer",3},{"expert",0},{"GPU_model_or_kernel_execution",false},{"Engine_or_NativeSource_or_ready",false},{"ranks",Json::array()}};
    uint64_t generation=0;
    for(int rank=0;rank<4;++rank){auto actual=declare_rank(registry,*architecture,checkpoint,sources,meta.m,eligibility,rank,o);check(actual.device_complete,"actual stock main DEVICE selection/reference admission");
        rad::Config cfg=config(512,128);cfg.placement="all_vram";cfg.vram_weights_mib=0;cfg.host_pool_mib=0;
        rad::PlannerInput input{&actual.builder->program(),&cfg,rank,4,nullptr,0,false,false};rad::Plan plan;good(rad::Planner(input).plan(&plan),"actual uncapped layout-only Plan; NOT hardware fit");
        const auto found=std::find_if(plan.units.begin(),plan.units.end(),[](const auto& u){return u.layer==3&&u.expert==0;});check(found!=plan.units.end(),"actual selected full-shape expert unit");const int32_t unit=int32_t(found-plan.units.begin());
        const std::string path=stage.path+"/rank-"+std::to_string(rank);mr::PublishedRecords publication;std::string why;
        const auto status=mr::pack_main_records(checkpoint,sources,actual.builder->program(),plan,{&unit,1},path,mr::Limits{},&publication,&why);
        if(status!=mr::Status::ok)throw std::runtime_error(std::string("actual pack refused: ")+mr::name(status)+" "+why);
        const auto receipt=publication.receipt();check(receipt.total_bytes==3342336&&receipt.complete_tensor_read_bytes==13369344&&receipt.selected_copy_bytes==3342336&&receipt.records==1&&receipt.physical_tensors==6&&receipt.peak_io_buffer==65536&&receipt.peak_input_fds<=2,"actual full-shape bounded source/record receipts");
        if(!rank)generation=receipt.source_generation;else check(receipt.source_generation==generation,"all4 rank cuts derive SAME actual physical source generation");
        std::unique_ptr<mr::MainReader> reader;packed(mr::MainReader::open(path,publication,mr::Limits{},&reader,&why),"strict actual-source Reader and padding/checksum scan");compare_reader(*reader,checkpoint,plan,unit,rank);
        std::shared_ptr<const mr::VerifiedCatalog> normalized;packed(reader->export_catalog(&normalized),"normalized verified stamp/record export");check(normalized&&normalized->record().key.rank==uint32_t(rank)&&normalized->record().length==3342336,"normalized catalog exact actual rank/record");
        if(rank==0)fault_cases(checkpoint,sources,actual.builder->program(),plan,unit,stage.path,publication);
        auto tiny=mr::Limits{};tiny.max_read_bytes=3342335;std::unique_ptr<mr::MainReader> capped;packed(mr::MainReader::open(path,publication,tiny,&capped),"one-record read-cap setup uses bounded scan");std::vector<std::byte> prior(1,std::byte{0x45});
        check(capped->read(normalized->record().key,&prior)==mr::Status::budget&&prior==std::vector<std::byte>({std::byte{0x45}}),"actual one-record read-minus-one refusal preserves caller");
        publication=mr::PublishedRecords{};compare_reader(*reader,checkpoint,plan,unit,rank);reader.reset();check(normalized->record().length==3342336,"normalized recipe remains owned after Reader/publication destruction");
        report["ranks"].push_back({{"rank",rank},{"unit",unit},{"native_bytes",receipt.total_bytes},{"source_complete_tensor_bytes",receipt.complete_tensor_read_bytes},{"source_generation",receipt.source_generation},
            {"manifest_checksum",receipt.manifest_checksum},{"metadata_reserved",receipt.metadata_reserved},{"source_config_bytes",receipt.source_config_bytes},{"source_index_bytes",receipt.source_index_bytes},{"window",receipt.peak_io_buffer},{"input_fds",receipt.peak_input_fds},{"uncapped_layout_plan_is_NOT_hardware_fit",true}});
    }
    check(sources.refused().empty(),"actual source resolver has no refusals");check(tracked_allocations()==tracked_before,"no Core-tracked model/device/pinned/mapped allocation by sample pack/metadata");
    check(failures_seen==4&&compared_bytes==26738688,"complete positive byte comparisons and four witnessed late faults");report["checks"]=checks;report["compared_native_bytes"]=compared_bytes;report["late_faults"]=failures_seen;
    report["late_source_audit_case"]="actual successful index fstat reported as EIO AFTER manifest pwrite; semantic fault, no checkpoint mutation/clone";report["full_model_or_inference_proof"]=false;
    const std::string summary=stage.path+"/sample-receipt.json";publish(summary,report);
    // Parent-level transaction publishes all four completed task-owned samples
    // together. No source/kernel/Engine work occurs after this final commit.
    char completion[256];const int n=std::snprintf(completion,sizeof completion,"PASS actual complete expert all4logicalranks checks=%llu comparedBytes=%llu lateFaults=%llu noGPU/model/Engine/NativeSourceReady\n",
        (unsigned long long)checks,(unsigned long long)compared_bytes,(unsigned long long)failures_seen);check(n>0&&size_t(n)<sizeof completion,"completion receipt bound");
    check(::renameat2(AT_FDCWD,stage.path.c_str(),AT_FDCWD,o.output.c_str(),1u)==0,"final all4 sample NOREPLACE publication");stage.published=true;
    (void)::write(STDOUT_FILENO,completion,size_t(n));return 0;
}catch(const std::exception& e){fault=0;allocation_fault::remaining=-1;std::cerr<<"FAIL actual main record proof: "<<e.what()<<" checks="<<checks<<"\n";return 1;}}
