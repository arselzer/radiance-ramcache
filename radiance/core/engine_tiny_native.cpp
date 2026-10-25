#include "engine_priv.h"
#include "build/tiny_native_source/source_factory.h"
#include "format/share.h"
#include "runtime/kvbind.h"
#include "place/native_inventory.h"
#include "rad_internal.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <new>
#include <system_error>
#include <unistd.h>
extern char** environ;
namespace rad {
namespace {
constexpr uint64_t MiB=uint64_t(1)<<20;
bool bounded_path(const std::string& s){return !s.empty()&&s.size()<=480&&s.find('\0')==std::string::npos&&s.find(':')==std::string::npos&&std::filesystem::path(s).is_absolute();}
int environment_profile(){size_t n=0;for(char** p=environ;p&&*p;++p){if(++n>4096)return RAD_E_UNSUPPORTED;
    if(!std::strncmp(*p,"RADIANCE_",9))return RAD_E_UNSUPPORTED;}return RAD_OK;}
int home_profile(const std::string& path){
    namespace fs=std::filesystem;
    const std::array<std::pair<const char*,const char*>,3> expected{{{"kernels","libref.so"},{"architectures","libglm53_tiny_startup_arch.so"},{"quantizers","libquant.so"}}};
    size_t roots=0;for(const auto& directory:fs::directory_iterator(path)) {
        if(++roots>3||directory.is_symlink()||!directory.is_directory())return RAD_E_UNSUPPORTED;
        const auto name=directory.path().filename().string();
        const auto found=std::find_if(expected.begin(),expected.end(),[&](const auto& e){return name==e.first;});
        if(found==expected.end())return RAD_E_UNSUPPORTED;
        size_t files=0;for(const auto& file:fs::directory_iterator(directory.path()))
            if(++files>1||file.is_symlink()||!file.is_regular_file()||file.path().filename()!=found->second)return RAD_E_UNSUPPORTED;
        if(files!=1)return RAD_E_UNSUPPORTED;
    }
    return roots==3?RAD_OK:RAD_E_UNSUPPORTED;
}
SamplerConfig tiny_sampler(){SamplerConfig s;s.n_vocab=s.n_vocab_local=320;s.max_seqs=1;s.max_hist=2048;s.max_candidates=1024;return s;}
ExpertPrepareLimits tiny_prepare_limits(){ExpertPrepareLimits l;l.max_rows=2;l.max_experts=4;l.max_top_k=2;l.max_maps=8;l.max_entries=32;l.max_readback_bytes=68;l.max_metadata_bytes=65536;return l;}
uint64_t runtime_without_arena(const TinyCtxQuote& q){return q.storage.page_and_heap_total-((q.arena_device+4095)&~uint64_t(4095))-q.arena_host;}
} // namespace
class TinyPoolAllocator final:public MemAllocator {
    struct Allocation{void* pointer=nullptr;int kind=0;};
    std::array<Allocation,4> allocations_{};
    uint64_t va_=0,live_=0;
public:
    uint64_t device=0,host=0;
    explicit TinyPoolAllocator(uint64_t va):va_(va){}
    ~TinyPoolAllocator(){if(live_)tiny_unretired_fail_stop();}
    void* alloc(int64_t bytes,int kind) override {
        if(bytes<=0||live_>=allocations_.size())return nullptr;
        const uint64_t n=(uint64_t(bytes)+4095)&~uint64_t(4095);
        if(n>6*MiB-va_-device-host)return nullptr;
        void* p=rad_device_allocator().alloc(bytes,kind);
        if(p){for(auto& a:allocations_)if(!a.pointer){a={p,kind};break;}
            ++live_;if(kind==RAD_MEM_DEVICE)device+=n;else host+=n;}return p;
    }
    void free(void* p,int kind) override {if(!p)return;
        for(auto& a:allocations_)if(a.pointer==p){if(a.kind!=kind)tiny_unretired_fail_stop();
            rad_device_allocator().free(p,kind);a.pointer=nullptr;--live_;return;}
        tiny_unretired_fail_stop();
    }
    const char* name()const override{return "tiny-counted-real-device";}
};
EngineTinyState::~EngineTinyState()=default;
int TinyNativeHarness::create(const TinyEngineOptions& options,std::unique_ptr<Engine>* out){
    if(!out||*out||!bounded_path(options.checkpoint)||!bounded_path(options.home)||!bounded_path(options.work_parent)||
       uint32_t(options.fail_after)>uint32_t(TinyEnginePhase::TransportReady))return RAD_E_INVAL;
    if(environment_profile()<0||!device_is_host())return RAD_E_UNSUPPORTED;
    try {
        // The fixed bootstrap state is the only bounded control preceding its
        // own ledger. No source, stream, tensor or Engine exists yet.
        static_assert(sizeof(EngineTinyState)<65536);
        auto state=std::make_unique<EngineTinyState>();
        RAD_TRY(state->budget.admit());
        RAD_TRY(state->budget.take(TinyCategory::EngineMetadata,8*MiB,&state->engine_metadata));
        RAD_TRY(state->budget.take(TinyCategory::Runtime,4*MiB,&state->runtime));
        RAD_TRY(state->budget.take(TinyCategory::Arena,4*MiB,&state->arena));
        RAD_TRY(state->budget.take(TinyCategory::Pools,6*MiB,&state->pools));
        RAD_TRY(state->budget.take(TinyCategory::NativeMetadata,4*MiB,&state->native_metadata));
        const auto backend=state->budget.backend();
        if(!backend.host||!backend.event_size||backend.event_size>65536)return RAD_E_UNSUPPORTED;
        RAD_TRY(state->budget.take(TinyCategory::Controls,128*backend.event_size,&state->events));
        state->options=options;
        RAD_TRY(home_profile(options.home));
        if(!std::filesystem::is_directory(options.work_parent)||std::filesystem::is_symlink(options.work_parent))return RAD_E_INVAL;
        const auto model=std::filesystem::weakly_canonical(options.checkpoint).string();
        const auto work=std::filesystem::canonical(options.work_parent).string();
        if(work==model||work.starts_with(model+"/"))return RAD_E_INVAL;
        state->result.bootstrap_control_bytes=sizeof(EngineTinyState)+sizeof(Engine)+sizeof(Engine::Rank);
        if(state->result.bootstrap_control_bytes>8*MiB)return RAD_E_UNSUPPORTED;
        RAD_TRY(Sampler::tiny_startup_quote(tiny_sampler(),&state->sampler_quote));
        Config cfg;cfg.model=options.checkpoint;cfg.radiance_home=options.home;cfg.kernel_hierarchy={"libref"};
        cfg.tp=1;cfg.max_tok=5;cfg.max_seqs=1;cfg.max_ctx=5;cfg.n_spec=0;cfg.deterministic=true;
        cfg.placement="all_vram";cfg.vram_weights_mib=4;cfg.vram_kv_mib=1;
        cfg.embedding_placement=cfg.embedding_resolved=TablePlacement::Vram;
        cfg.embedding_host_mib=cfg.tower_host_mib=cfg.host_pool_mib=0;
        cfg.prefix_cache=false;cfg.prefix_cache_mib=0;cfg.checkpoint_slots=0;cfg.checkpoint_interval=0;
        cfg.mm_max_patches=0;cfg.accept_reference_kernels=true;cfg.host_headroom_mib=0;
        auto engine=std::make_unique<Engine>(std::move(cfg));
        engine->tiny_=std::move(state);*out=std::move(engine);return RAD_OK;
    }catch(const std::bad_alloc&){return RAD_E_NOMEM;}catch(...){return RAD_E_STATE;}
}
int TinyNativeHarness::snapshot(const Engine& e,TinyEngineSnapshot* out){if(!out||!e.tiny_)return RAD_E_INVAL;
    auto q=e.tiny_->result;q.budget=e.tiny_->budget.snapshot();*out=q;return RAD_OK;}
int TinyNativeHarness::retire(Engine& e){if(!e.tiny_)return RAD_E_INVAL;e.tiny_shutdown();return RAD_OK;}
int Engine::tiny_phase(uint32_t phase){auto& t=*tiny_;t.result.phase=TinyEnginePhase(phase);
    if(t.options.fail_after!=TinyEnginePhase::Requested&&t.options.fail_after==t.result.phase){++t.result.faults_triggered;return RAD_E_STATE;}return RAD_OK;}
int Engine::start_tiny_transport(){auto& t=*tiny_;
    // Retirement is terminal even before the first attempted start: its credits
    // and owners are already released. Refuse before source/allocator activity.
    if(t.result.retired||t.retiring||t.result.phase==TinyEnginePhase::Terminal||
       t.result.phase==TinyEnginePhase::Retired||t.attempted)
        return t.result.first_status<0?t.result.first_status:RAD_E_STATE;
    t.attempted=true;t.result.attempts=1;int status=RAD_E_STATE;
    try{status=tiny_transport_body();}catch(const std::bad_alloc&){status=RAD_E_NOMEM;}catch(...){status=RAD_E_STATE;}
    if(status<0){t.result.first_status=status;t.result.failed_phase=t.result.phase;t.result.ready=false;
        t.result.phase=TinyEnginePhase::Terminal;tiny_shutdown();}
    return status;
}
int Engine::tiny_transport_body(){auto& t=*tiny_;using P=TinyEnginePhase;
    RAD_TRY(tiny_phase(uint32_t(P::ColdAdmitted)));
    RAD_TRY(glm53::engine_source::CheckpointBinding::open(cfg_.model,t.budget,&t.binding));
    t.result.source_fingerprint=t.binding->receipt().source_fingerprint;
    RAD_TRY(read_metadata());RAD_TRY(load_vocab());
    if(!vocab_||vocab_->n_tokens()!=320||vocab_->bos()!=257||vocab_->eos()!=256||vocab_->eot()!=256||vocab_->add_bos()||vocab_->add_eos())return RAD_E_FORMAT;
    auto eog=vocab_->eog();std::sort(eog.begin(),eog.end());eog.erase(std::unique(eog.begin(),eog.end()),eog.end());
    if(eog!=std::vector<int32_t>({256,258}))return RAD_E_FORMAT;
    t.result.bos=vocab_->bos();t.result.eos=vocab_->eos();t.result.eog_count=eog.size();
    RAD_TRY(tiny_phase(uint32_t(P::SourceOwned)));
    rad_tools_registry().set_code_device(0);rad_tools_registry().set_no_device_code(true);
    RAD_TRY(load_plugins());if(!arch_step_||plugins_.arch_plugin!="glm53_tiny_native_arch")return RAD_E_UNSUPPORTED;
    RAD_TRY(tiny_phase(uint32_t(P::PluginsLoaded)));
    auto rank=std::make_unique<Rank>();rank->device=rank->index=0;ranks_.push_back(std::move(rank));
    RAD_TRY(declare());auto& r=*ranks_[0];
    if(!r.tok_buf||r.tok_buf>=r.program.buffers.size())return RAD_E_SHAPE;
    const auto& tokens=r.program.buffers[r.tok_buf];
    if(tokens.name!="sampler.tokens"||tokens.decl.dtype!=RAD_I32||tokens.decl.rank!=1||tokens.decl.shape[0]!=1||tokens.decl.kind!=RAD_BUF_PERSIST)return RAD_E_SHAPE;
    t.result.weights=r.program.weights.size();t.result.buffers=r.program.buffers.size();t.result.ops=r.program.ops.size();
    for(const auto& op:r.program.ops)t.result.bands+=op.bands.size();
    CtxDesc desc;desc.program=&r.program;desc.record_passes=false;desc.expert_prepare_limits=tiny_prepare_limits();
    RAD_TRY(Ctx::tiny_startup_quote(desc,true,&t.ctx_quote));
    RAD_TRY(initialize_program_instances(rad_tools_registry(),r.program));
    RAD_TRY(tiny_phase(uint32_t(P::Declared)));
    PlannerInput metadata{&r.program,&cfg_,0,1,nullptr,0,false,true};
    RAD_TRY(Planner(metadata).plan(&t.metadata_plan));
    t.result.units=t.metadata_plan.units.size();if(t.result.weights!=156||t.result.units!=43)return RAD_E_SHAPE;
    RAD_TRY(tiny_phase(uint32_t(P::MetadataPlanned)));
    RAD_TRY(glm53::engine_source::PreparedRecords::open(t.binding,r.program,t.metadata_plan,t.options.work_parent,t.budget,&t.records));
    if(t.records->owned().size()!=24||t.records->resident_units().size()!=8||t.records->catalog().size()!=24)return RAD_E_STATE;
    t.result.owned_units=t.records->owned().size();t.result.resident_units=t.records->resident_units().size();
    RAD_TRY(tiny_phase(uint32_t(P::SourcePrepared)));
    NativeInventoryLimits limits;limits.max_rows=2;limits.top_k=2;limits.max_owned_units=24;limits.max_classes=8;limits.max_regions=16;
    limits.max_device_bytes=limits.max_landing_bytes=4*MiB;limits.max_weights_bytes=MiB;
    limits.max_metadata_bytes=4*MiB;limits.max_host_bytes=8*MiB;limits.hot_bytes=65536;limits.cold_bytes=98304;
    RAD_TRY(NativeInventory::bind(r.program,t.metadata_plan,t.records->catalog(),t.records->owned(),t.records->resident_units(),limits,&t.inventory));
    Config resolved;RAD_TRY(t.inventory->configure_pools(cfg_,&resolved));cfg_=std::move(resolved);
    PlannerInput native{&r.program,&cfg_,0,1,nullptr,0,false,true};native.native_inventory=t.inventory;
    RAD_TRY(Planner(native).plan(&r.plan));RAD_TRY(t.inventory->validate(r.program,r.plan));
    rank_cfg_.push_back(cfg_);
    RAD_TRY(NativeBackingOwner::tiny_startup_quote(r.program,r.plan,&t.owner_quote));
    MoverConfig mover_config;mover_config.tiny_budget=&t.budget;
    RAD_TRY(Mover::tiny_startup_quote(r.program,r.plan,mover_config,&t.mover_quote));
    RAD_TRY(tiny_step_state_quote(vocab_->eog(),&t.step_quote));
    t.result.arena_quote=((t.ctx_quote.arena_device+4095)&~uint64_t(4095))+t.ctx_quote.arena_host;
    t.result.runtime_quote=t.sampler_quote.storage.page_and_heap_total+runtime_without_arena(t.ctx_quote)+
        t.owner_quote.storage.page_and_heap_total+t.mover_quote.page_and_heap_total+t.step_quote.storage.page_and_heap_total+262144;
    const uint64_t events=t.ctx_quote.storage.event_count+t.mover_quote.event_count+t.step_quote.storage.event_count+4;
    t.result.event_quote=events*t.budget.backend().event_size;
    if(t.result.arena_quote>4*MiB||t.result.runtime_quote>4*MiB||events>128)return RAD_E_NOMEM;
    RAD_TRY(tiny_phase(uint32_t(P::NativePlanned)));
    // Actual VMem is used by HOST too. Pool owns the real singleton allocator;
    // startup counts requested backings and KV VA separately, never malloc stand-ins.
    if(cfg_.vram_weights_mib!=1||cfg_.vram_kv_mib!=1)return RAD_E_SHAPE;
    const uint64_t landing=t.inventory->footprint().landing_bytes;
    const int64_t gran=VMem::granularity();if(gran<=0||gran>int64_t(2*MiB))return RAD_E_UNSUPPORTED;
    const uint64_t kv_va=(MiB+uint64_t(gran)-1)/uint64_t(gran)*uint64_t(gran);
    if(MiB+landing+kv_va>6*MiB)return RAD_E_NOMEM;
    t.pool_allocator=std::make_unique<TinyPoolAllocator>(kv_va);
    RAD_TRY(r.pools.configure_native(cfg_,0,r.plan,t.pool_allocator.get(),t.pool_allocator.get()));
    t.result.pool_device_requested=t.pool_allocator->device;t.result.pool_host_requested=t.pool_allocator->host;
    t.result.kv_va_reserved=kv_va;t.result.kv_committed=r.pools.kv().committed();if(t.result.kv_committed)return RAD_E_STATE;
    RAD_TRY(tiny_phase(uint32_t(P::PoolsReady)));
    RAD_TRY(r.tiny_compute.open(t.budget,0));
    RAD_TRY(load_weights()); // actual method: native omissions + ordinary loader + resident checks
    RAD_TRY(tiny_phase(uint32_t(P::ResidentLoaded)));
    RAD_TRY(t.records->finish_bootstrap());RAD_TRY(t.records->audit());RAD_TRY(t.binding->audit());
    RAD_TRY(tiny_phase(uint32_t(P::BootstrapTrimmed)));
    RAD_TRY(NativeBackingOwner::open(r.program,r.plan,r.mover,t.records->source(),&r.native_owner));
    RAD_TRY(tiny_phase(uint32_t(P::OwnerBound)));
    RAD_TRY(configure_kv());
    desc.stream=r.tiny_compute.get();desc.residency=r.mover.residency();desc.arch_step=arch_step_;
    desc.registry=&rad_tools_registry();desc.expert_prepare_owner=r.native_owner.get();desc.tiny_retirement=true;
    bind_kv_pools(r.program,[&](int32_t g){return r.kv.plan(g);},&desc.kv);
    // Diagnostic-only receipt of the REAL post-configure binding. It neither
    // normalizes the sentinel nor changes the quote/init acceptance predicates.
    t.result.ctx_kv_bindings=desc.kv.size();
    if(desc.kv.size()==1){const auto& k=desc.kv[0];t.result.ctx_sentinel_base_present=k.base!=nullptr;
        t.result.ctx_sentinel_rank=k.rank;t.result.ctx_sentinel_layers=k.layer_slot.size();t.result.ctx_sentinel_dtype=k.dtype;
        t.result.ctx_sentinel_bytes=k.bytes;t.result.ctx_sentinel_stride=k.layer_stride;
        for(const auto extent:k.shape)t.result.ctx_sentinel_nonzero_shapes+=extent!=0;}
    t.result.ctx_quote_status=Ctx::tiny_startup_quote(desc,true,&t.ctx_quote);
    RAD_TRY(t.result.ctx_quote_status);
    t.result.ctx_init_status=r.ctx.init(desc);RAD_TRY(t.result.ctx_init_status);
    RAD_TRY(tiny_phase(uint32_t(P::ContextBound)));
    RAD_TRY(step_state_init());
    RAD_TRY(tiny_phase(uint32_t(P::StepStateReady)));
    RAD_TRY(t.records->audit());RAD_TRY(t.binding->audit());RAD_TRY(t.inventory->validate(r.program,r.plan));
    RAD_TRY(tiny_phase(uint32_t(P::TransportReady)));t.result.ready=true;return RAD_OK;
}
void Engine::tiny_shutdown() noexcept {if(!tiny_||tiny_->result.retired||tiny_->retiring)return;auto& t=*tiny_;t.retiring=true;
    try {
        for(auto& r:ranks_){r->mover.tiny_retire_or_fail();if(r->tiny_compute.get())tiny_drain_or_fail(r->tiny_compute.get(),r->device);
            if(r->native_owner&&r->native_owner->quiesce()<0)tiny_unretired_fail_stop();}
        for(auto& r:ranks_){r->ctx.shutdown();if(r->native_owner&&r->native_owner->close_after_context_shutdown()<0)tiny_unretired_fail_stop();}
        step_state_fini();t.eog_stage.clear();
        if(t.records&&t.records->close()<0)tiny_unretired_fail_stop();
        for(auto& r:ranks_)r->native_owner.reset();
        ranks_.clear();t.pool_allocator.reset();t.inventory.reset();t.records.reset();t.metadata_plan=Plan{};t.binding.reset();
        t.events.reset();t.native_metadata.reset();t.pools.reset();t.arena.reset();t.runtime.reset();
        t.result.ready=false;t.result.retired=true;if(t.result.first_status>=0)t.result.phase=TinyEnginePhase::Retired;
        t.retiring=false;
    }catch(...){tiny_unretired_fail_stop();}
}
}
