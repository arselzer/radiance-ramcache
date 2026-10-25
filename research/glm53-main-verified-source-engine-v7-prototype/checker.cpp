#include "engine.h"
#include "engine_tiny_native.h"
#include "device/device.h"
#include "util/stallwatch.h"
#include "fixture_writer.h"
#include "build/rad_build.h"
#include "build/startup.h"
#include "runtime/ctx.h"
#include "regressions/compact/allocation_hooks.h"
#include <array>
#include <atomic>
#include <cstdarg>
#include <fcntl.h>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <unistd.h>
namespace fs=std::filesystem;
namespace {
size_t checks=0,groups=0,kv_quote_refusals=0;
void need(bool condition,const std::string& why){++checks;if(!condition)throw std::runtime_error(why);}
void good(int status,const std::string& why){need(status==RAD_OK,why+" status="+std::to_string(status));}
struct Allocation{void* pointer=nullptr;int kind=0;};
std::mutex mutex;
std::array<Allocation,512> allocations{};
bool tracing=false,unknown_free=false;
int64_t fail_at=-1;uint64_t attempts=0,live=0,failures_seen=0;
bool fail_eog_copy=false;uint64_t eog_copy_witness=0;
std::atomic<bool> trace_source_io{false};
std::atomic<uint64_t> source_io_calls{0},copy_calls{0};
void begin(int64_t failure=-1){std::lock_guard lock(mutex);need(!live,"old device backing remains");allocations={};tracing=true;unknown_free=false;fail_at=failure;attempts=failures_seen=0;}
void end(){std::lock_guard lock(mutex);need(!live&&!unknown_free,"actual device/pinned/mapped allocations not fully retired");tracing=false;}
struct Scratch{
    fs::path root,checkpoint,home,work;
    Scratch(const std::string& core_home,const std::string& architecture){char pattern[]="/tmp/glm53-real-engine-XXXXXX";const char* p=::mkdtemp(pattern);if(!p)throw std::runtime_error("scratch root");root=p;checkpoint=root/"checkpoint";home=root/"home";work=root/"work";
        fs::create_directories(home/"kernels");fs::create_directories(home/"architectures");fs::create_directories(home/"quantizers");fs::create_directory(work);
        fs::copy_file(fs::path(core_home)/"kernels/libref.so",home/"kernels/libref.so");
        fs::copy_file(fs::path(core_home)/"quantizers/libquant.so",home/"quantizers/libquant.so");
        fs::copy_file(architecture,home/"architectures/libglm53_tiny_startup_arch.so");
        glm53::tiny_startup::FixtureFiles files;good(glm53::tiny_startup::write_fixture(checkpoint.string(),&files),"actual219tensor four-file tiny checkpoint");
    }
    ~Scratch(){std::error_code error;fs::remove_all(root,error);}
    rad::TinyEngineOptions options(rad::TinyEnginePhase fault=rad::TinyEnginePhase::Requested)const{return {checkpoint.string(),home.string(),work.string(),fault};}
};
rad::TinyEngineSnapshot snapshot(const rad::Engine& e){rad::TinyEngineSnapshot out;good(rad::TinyNativeHarness::snapshot(e,&out),"bounded Engine snapshot");return out;}
void retired(const rad::TinyEngineSnapshot& s){need(s.retired&&!s.ready,"Engine is retired, no Ready authority");
    for(size_t i=0;i<s.budget.live.size();++i){uint64_t expected=0;if(i==(size_t)rad::TinyCategory::EngineMetadata)expected=8u<<20;
        if(i==(size_t)rad::TinyCategory::Controls)expected=s.budget.admission_controls;
        need(s.budget.live[i]==expected,"retained payload/ticket beyond documented Engine metadata baseline");}
    need(!s.budget.streams&&!s.budget.aux_threads,"Engine stream/thread roles not retired");}
void baseline(const Scratch& scratch){++groups;begin();std::unique_ptr<rad::Engine> engine;good(rad::TinyNativeHarness::create(scratch.options(),&engine),"fixed precharged actual Engine construction");
    const int started=engine->start();
    if(started!=RAD_OK){const auto d=snapshot(*engine);
        std::cerr<<"ACTUAL_ENGINE_START_REFUSAL status="<<started<<" failedPhase="<<uint32_t(d.failed_phase)<<" phase="<<uint32_t(d.phase)
            <<" attempts="<<d.attempts<<" retired="<<d.retired<<" ordinary="<<d.ordinary_checked<<" residentMembers="<<d.resident_members_checked
            <<" kvBindings="<<d.ctx_kv_bindings<<" sentinelBasePresent="<<d.ctx_sentinel_base_present<<" sentinelBytes="<<d.ctx_sentinel_bytes
            <<" sentinelStride="<<d.ctx_sentinel_stride<<" sentinelRank="<<d.ctx_sentinel_rank<<" sentinelLayers="<<d.ctx_sentinel_layers
            <<" sentinelDtype="<<d.ctx_sentinel_dtype<<" sentinelNonzeroShapes="<<d.ctx_sentinel_nonzero_shapes
            <<" ctxQuote="<<d.ctx_quote_status<<" ctxInit="<<d.ctx_init_status<<"\n";}
    good(started,"actual Engine::start transport pipeline");const auto s=snapshot(*engine);
    need(s.ready&&!s.retired&&s.phase==rad::TinyEnginePhase::TransportReady&&s.attempts==1,"actual Engine transportReady publication");
    need(s.weights==156&&s.units==43&&s.owned_units==24&&s.resident_units==8,"actual Engine graph/native partition");
    need(s.omitted_views==96&&s.omitted_source_bytes==626688&&s.ordinary_views==59&&s.ordinary_source_bytes==393856,"typed source/share-before-omission and ordinary load receipts");
    need(s.ordinary_checked==59&&s.resident_members_checked==32,"actual Loader resident completion proof");
    need(s.loader.all_retired&&s.loader.workers_started==2&&s.loader.workers_joined==2&&s.loader.real_uplink_drains==1&&s.loader.live_bounces_at_final_drain==2,"actual two-worker loader final real drain");
    need(s.bos==257&&s.eos==256&&s.eog_count==2&&s.kv_committed==0,"actual loaded Vocab and truthful zero-KV state");
    need(s.ctx_kv_bindings==1&&s.ctx_sentinel_base_present&&!s.ctx_sentinel_bytes&&!s.ctx_sentinel_stride&&
         !s.ctx_sentinel_rank&&!s.ctx_sentinel_layers&&!s.ctx_sentinel_dtype&&!s.ctx_sentinel_nonzero_shapes&&
         s.ctx_quote_status==RAD_OK&&s.ctx_init_status==RAD_OK,
         "actual zero-byte group0 carve retains reserved base, never a real KV extent");
    need(s.pool_device_requested+s.pool_host_requested+s.kv_va_reserved<=6u*1024u*1024u&&s.runtime_quote<=4u*1024u*1024u&&s.arena_quote<=4u*1024u*1024u,"actual owner/pool/arena quotes within old categories");
    const auto before=attempts;need(engine->start()==RAD_E_STATE&&attempts==before&&snapshot(*engine).attempts==1,"successful same Engine cannot restart");
    need(engine->serve(nullptr)==RAD_E_UNSUPPORTED,"transport-only Engine cannot serve or run sampler/model step");
    std::cout<<"ACTUAL_ENGINE W="<<s.weights<<" buffers="<<s.buffers<<" ops="<<s.ops<<" bands="<<s.bands<<" units="<<s.units<<" owned="<<s.owned_units<<" resident="<<s.resident_units
        <<" omitted="<<s.omitted_views<<" omittedBytes="<<s.omitted_source_bytes<<" ordinary="<<s.ordinary_views<<" ordinaryBytes="<<s.ordinary_source_bytes<<" checked="<<s.ordinary_checked<<" residentMembers="<<s.resident_members_checked
        <<" poolDevice="<<s.pool_device_requested<<" poolHost="<<s.pool_host_requested<<" kvVA="<<s.kv_va_reserved<<" kvCommit="<<s.kv_committed<<" runtimeQuote="<<s.runtime_quote<<" arenaQuote="<<s.arena_quote<<" loaderContexts="<<s.loader_context_bytes<<"\n";
    good(rad::TinyNativeHarness::retire(*engine),"explicit actual Engine teardown");retired(snapshot(*engine));
    need(fs::is_empty(scratch.work),"source work generation retained after normal last reader");engine.reset();end();
}
void retire_before_start(const Scratch& scratch){++groups;begin();
    std::unique_ptr<rad::Engine> engine;
    good(rad::TinyNativeHarness::create(scratch.options(),&engine),"retire-before-start actual Engine construction");
    good(rad::TinyNativeHarness::retire(*engine),"unstarted Engine retirement");
    const auto before=snapshot(*engine);retired(before);
    need(!before.attempts&&!before.source_fingerprint&&before.phase==rad::TinyEnginePhase::Retired,
         "pre-start retirement has no attempted source/graph startup");
    // Calibrate the actual Core source open/pread seams and complete C++
    // allocation-family witness before bracketing the refused Engine call.
    const auto file=(scratch.checkpoint/"model.safetensors").string();
    source_io_calls=0;trace_source_io=true;
    const int fd=::open(file.c_str(),O_RDONLY|O_CLOEXEC);char byte=0;
    const auto got=fd>=0?::pread(fd,&byte,1,0):-1;if(fd>=0)::close(fd);
    trace_source_io=false;
    need(fd>=0&&got==1&&source_io_calls==2,"source open/pread witness is active");
    allocation_fault::begin();void* witness=::operator new(1);::operator delete(witness);
    const auto heap_witness=allocation_fault::calls;allocation_fault::end();
    need(heap_witness==1,"complete C++ allocation-family witness is active");
    const auto device_before=attempts;const auto copies_before=copy_calls.load();
    source_io_calls=0;trace_source_io=true;allocation_fault::begin();
    const int status=engine->start();
    const auto heap_calls=allocation_fault::calls;allocation_fault::end();trace_source_io=false;
    need(status==RAD_E_STATE&&!heap_calls&&attempts==device_before&&copy_calls==copies_before&&source_io_calls==0,
         "retired unstarted Engine refuses before C++/device allocation, upload or source open/read");
    const auto after=snapshot(*engine);retired(after);
    need(!after.attempts&&!after.source_fingerprint&&after.phase==before.phase&&
         after.first_status==before.first_status&&after.budget.live==before.budget.live,
         "retired Engine authority and released-credit state remain unchanged");
    need(fs::is_empty(scratch.work),"retire-before-start cannot publish a source work generation");
    good(rad::TinyNativeHarness::retire(*engine),"repeated unstarted retirement is idempotent");
    engine.reset();end();
}
void phase_faults(const Scratch& scratch){++groups;using P=rad::TinyEnginePhase;
    for(const auto phase:{P::ColdAdmitted,P::Declared,P::SourcePrepared,P::PoolsReady,P::MoverReady,P::ResidentLoaded,P::OwnerBound,P::ContextBound,P::StepStateReady}){
        begin();std::unique_ptr<rad::Engine> engine;good(rad::TinyNativeHarness::create(scratch.options(phase),&engine),"phase-fault fixed Engine construction");
        const int status=engine->start();need(status<0,"requested actual startup phase refusal");const auto failed=snapshot(*engine);
        need(failed.first_status==status&&failed.failed_phase==phase&&failed.faults_triggered==1&&failed.attempts==1,"intended phase fault exactly once");retired(failed);
        const auto before=attempts;need(engine->start()==status&&attempts==before&&snapshot(*engine).faults_triggered==1,"first error sticky before/after Ctx without retry work");
        need(fs::is_empty(scratch.work),"private source work leaked after partial startup refusal");engine.reset();end();
    }
}
void eog_source_fault(const Scratch& scratch){++groups;begin();std::unique_ptr<rad::Engine> engine;
    good(rad::TinyNativeHarness::create(scratch.options(),&engine),"EOG source-lifetime Engine construction");
    fail_eog_copy=true;eog_copy_witness=0;const int status=engine->start();fail_eog_copy=false;
    need(status<0&&eog_copy_witness==1,"actual EOG copy first enqueues successfully then reports refusal");
    const auto s=snapshot(*engine);need(s.failed_phase==rad::TinyEnginePhase::ContextBound&&!s.faults_triggered,"EOG fault reached intended real step-state stage");retired(s);
    const auto before=attempts;need(engine->start()==status&&attempts==before&&eog_copy_witness==1,"EOG failure cannot retry or release source before retirement");
    engine.reset();need(fs::is_empty(scratch.work),"EOG failure retained private generation");end();
}
void allocation_faults(const Scratch& scratch){++groups;
    // Actual Sampler first pin, pool first backing and Loader first pin. These
    // are observed ABI allocation refusals, not invented status-only wrappers.
    for(const int64_t at:{0,11,13}){begin(at);std::unique_ptr<rad::Engine> engine;good(rad::TinyNativeHarness::create(scratch.options(),&engine),"allocator-fault Engine construction");
        const int status=engine->start();need(status<0&&failures_seen==1,"actual intended device allocation refusal witnessed");retired(snapshot(*engine));
        const auto before=attempts;need(engine->start()==status&&attempts==before,"allocator failure permanently poisons same Engine startup");
        engine.reset();need(fs::is_empty(scratch.work),"allocation refusal retained source generation");end();}
}
void sentinel_quote_cases(const Scratch& scratch){++groups;
    // Separate pure quote API coverage on an ACTUAL tiny declaration. This
    // opaque marker is never dereferenced and is not a forged device backing;
    // baseline() independently demands the real Engine/KVManager-carved base.
    rad::Checkpoint checkpoint;good(checkpoint.open(scratch.checkpoint.string()),"sentinel quote actual source open");
    rad::Registry registry;registry.set_no_device_code(true);
    good(registry.load_plugin((scratch.home/"kernels/libref.so").string(),0),"sentinel quote actual reference plugin");
    good(registry.load_plugin((scratch.home/"architectures/libglm53_tiny_startup_arch.so").string(),1),"sentinel quote actual tiny architecture");
    checkpoint.set_importers(&registry);const auto* architecture=registry.plugin("glm53_tiny_native_arch");
    need(architecture&&architecture->arch_declare,"sentinel quote actual architecture callback");
    RadBuildCtx context{};context.world_size=1;context.max_tok=context.max_ctx=5;context.max_seqs=1;
    rad::CheckpointSources sources(checkpoint);rad::Builder builder(registry,checkpoint.meta().m,context);
    builder.set_sources(sources.fn());builder.set_instances(false);
    good(architecture->arch_declare(&builder,&checkpoint.meta().m,&builder.program().ctx),"sentinel quote actual declaration");
    good(builder.finish(),"sentinel quote finished actual Program");auto& program=builder.program();
    need(program.kv_groups.size()==1,"actual recipe has only group0 sentinel");const auto original=program.kv_groups[0];
    rad::CtxDesc desc;desc.program=&program;desc.record_passes=false;
    auto& limits=desc.expert_prepare_limits;limits.max_rows=2;limits.max_experts=4;limits.max_top_k=2;limits.max_maps=8;limits.max_entries=32;limits.max_readback_bytes=4096;limits.max_metadata_bytes=65536;
    rad::TinyCtxQuote quoted{};good(rad::Ctx::tiny_startup_quote(desc,true,&quoted),"empty preconfigure KV bindings accepted only with true Program sentinel");
    char opaque=0;desc.kv.resize(1);desc.kv[0].base=&opaque;
    good(rad::Ctx::tiny_startup_quote(desc,true,&quoted),"pure zeroextent sentinel base is ignored without dereference");
    for(int fault=0;fault<18;++fault){program.kv_groups.resize(1);program.kv_groups[0]=original;desc.kv.assign(1,rad::KVPoolBinding{});desc.kv[0].base=&opaque;
        switch(fault){case 0:desc.kv[0].bytes=1;break;case 1:desc.kv[0].layer_stride=1;break;
        case 2:desc.kv[0].rank=1;break;case 3:desc.kv[0].shape[0]=1;break;case 4:desc.kv[0].shape[RAD_MAX_RANK-1]=1;break;
        case 5:desc.kv[0].layer_slot.push_back(0);break;case 6:desc.kv[0].dtype=RAD_BF16;break;
        case 7:desc.kv.push_back({});break;case 8:program.kv_groups.push_back({});break;
        case 9:program.kv_groups[0].decl.n_head_kv=1;break;case 10:program.kv_groups[0].layers.push_back(0);break;
        case 11:program.kv_groups[0].block_size=1;break;case 12:program.kv_groups[0].bytes_per_state=1;break;
        case 13:program.kv_groups[0].decl.dtype=RAD_BF16;break;case 14:program.kv_groups[0].name="real-kv";break;
        case 15:program.kv_groups[0].zero=true;break;case 16:program.kv_groups[0].replicated=true;break;case 17:program.kv_groups[0].drafter=true;break;}
        rad::TinyCtxQuote refused{};refused.weights=777;
        need(rad::Ctx::tiny_startup_quote(desc,true,&refused)<0&&refused.weights==777,"nonzero/real KV quote refuses and preserves caller");++kv_quote_refusals;
    }
    program.kv_groups.resize(1);program.kv_groups[0]=original;desc.kv.assign(1,rad::KVPoolBinding{});desc.kv[0].base=&opaque;
    good(rad::Ctx::tiny_startup_quote(desc,true,&quoted),"actual sentinel restored after focused metadata mutations");
}
}
extern "C" void* __real_rad_dev_alloc(int64_t,int);
extern "C" void __real_rad_dev_free(void*,int);
extern "C" void* __wrap_rad_dev_alloc(int64_t bytes,int kind){
    {std::lock_guard lock(mutex);if(tracing&&int64_t(attempts++)==fail_at){++failures_seen;return nullptr;}}
    void* p=__real_rad_dev_alloc(bytes,kind);
    if(p){std::lock_guard lock(mutex);if(tracing){bool recorded=false;for(auto& a:allocations)if(!a.pointer){a={p,kind};++live;recorded=true;break;}if(!recorded)std::abort();}}
    return p;
}
extern "C" void __wrap_rad_dev_free(void* p,int kind){if(p){std::lock_guard lock(mutex);if(tracing){bool found=false;for(auto& a:allocations)if(a.pointer==p){if(a.kind!=kind)unknown_free=true;a.pointer=nullptr;--live;found=true;break;}if(!found)unknown_free=true;}}
    __real_rad_dev_free(p,kind);
}
extern "C" int __real_rad_memcpy_async(void*,const void*,int64_t,RadStream);
extern "C" int __wrap_rad_memcpy_async(void* destination,const void* source,int64_t bytes,RadStream stream){
    ++copy_calls;
    const int status=__real_rad_memcpy_async(destination,source,bytes,stream);
    // Closed recipe's only8-byte upload is its two real loaded EOG IDs. The
    // checker demands an actual successful submission before returning error.
    if(status>=0&&fail_eog_copy&&bytes==8){fail_eog_copy=false;++eog_copy_witness;return RAD_E_STATE;}
    return status;
}
extern "C" int __real_open(const char*,int,...);
extern "C" ssize_t __real_pread(int,void*,size_t,off_t);
extern "C" int __wrap_open(const char* path,int flags,...){
    mode_t mode=0;if((flags&O_CREAT)||(flags&O_TMPFILE)==O_TMPFILE){va_list args;va_start(args,flags);mode=va_arg(args,mode_t);va_end(args);}
    if(trace_source_io.load())++source_io_calls;
    return __real_open(path,flags,mode);
}
extern "C" ssize_t __wrap_pread(int fd,void* buffer,size_t bytes,off_t offset){
    if(trace_source_io.load())++source_io_calls;
    return __real_pread(fd,buffer,bytes,offset);
}
int main(int argc,char** argv){try{
    need(argc==6&&!std::strcmp(argv[1],"--run-host")&&!std::strcmp(argv[2],"--radiance-home")&&!std::strcmp(argv[4],"--architecture"),"usage: --run-host --radiance-home MATCHING_HOST_HOME --architecture TINY_MODULE");
    need(rad::device_is_host(),"actual Engine transport checker requires HOST build");Scratch scratch(argv[3],argv[5]);
    baseline(scratch);retire_before_start(scratch);phase_faults(scratch);allocation_faults(scratch);eog_source_fault(scratch);sentinel_quote_cases(scratch);
    need(groups==6&&kv_quote_refusals==18,"complete Engine/sentinel quote matrix");std::cout<<"PASS actual Engine transport groups="<<groups<<" checks="<<checks<<" sentinelQuoteRefusals="<<kv_quote_refusals<<" no model/sampler step/serve/GPU\n";return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<" checks="<<checks<<"\n";return 1;}}
