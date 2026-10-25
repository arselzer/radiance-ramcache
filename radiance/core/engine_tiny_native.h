#pragma once
// Internal synthetic HOST transport fixture. No public ABI/CLI or serving activation.
#include "build/tiny_startup.h"
#include "build/tiny_loader.h"
#include "build/tiny_component_quotes.h"
#include "place/planner.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <type_traits>
namespace glm53::engine_source {class CheckpointBinding;class PreparedRecords;}
namespace rad {
class Engine;
class TinyPoolAllocator;
enum class TinyEnginePhase:uint32_t {Requested,ColdAdmitted,SourceOwned,PluginsLoaded,Declared,
    MetadataPlanned,SourcePrepared,NativePlanned,PoolsReady,MoverReady,ResidentLoaded,
    BootstrapTrimmed,OwnerBound,ContextBound,StepStateReady,TransportReady,Terminal,Retired};
struct TinyEngineOptions {
    std::string checkpoint,home,work_parent;
    // Explicit manual phase fault, never a production configuration flag.
    TinyEnginePhase fail_after=TinyEnginePhase::Requested;
};
struct TinyEngineSnapshot {
    TinyEnginePhase phase=TinyEnginePhase::Requested,failed_phase=TinyEnginePhase::Requested;
    int first_status=RAD_OK;
    uint64_t attempts=0,faults_triggered=0;
    uint64_t weights=0,buffers=0,ops=0,bands=0,units=0,owned_units=0,resident_units=0;
    uint64_t omitted_views=0,omitted_source_bytes=0,ordinary_views=0,ordinary_source_bytes=0;
    uint64_t ordinary_checked=0,resident_members_checked=0,source_fingerprint=0,loader_context_bytes=0;
    uint64_t runtime_quote=0,arena_quote=0,event_quote=0;
    uint64_t pool_device_requested=0,pool_host_requested=0,kv_va_reserved=0,kv_committed=0;
    uint64_t bos=0,eos=0,eog_count=0,bootstrap_control_bytes=0,vocab_index_bytes=0;
    uint64_t ctx_kv_bindings=0,ctx_sentinel_base_present=0,ctx_sentinel_rank=0,ctx_sentinel_layers=0,ctx_sentinel_dtype=0,ctx_sentinel_nonzero_shapes=0;
    int64_t ctx_sentinel_bytes=0,ctx_sentinel_stride=0;
    int ctx_quote_status=RAD_OK,ctx_init_status=RAD_OK;
    bool ready=false,retired=false;
    TinyLoaderReceipt loader{};
    TinyBudgetSnapshot budget{};
};
static_assert(std::is_trivially_copyable_v<TinyEngineSnapshot> && sizeof(TinyEngineSnapshot)<=4096);
// State precedes ranks_ in Engine so the ledger/source outlive every rank even
// during partial destruction. Its actual fixed bootstrap allocation is bounded
// and explicitly charged in EngineMetadata before any Engine/source construction.
struct EngineTinyState {
    TinyBudget budget;
    TinyTicket engine_metadata,runtime,arena,pools,native_metadata,events;
    TinyEngineOptions options;
    TinyEngineSnapshot result;
    std::shared_ptr<glm53::engine_source::CheckpointBinding> binding;
    std::unique_ptr<glm53::engine_source::PreparedRecords> records;
    Plan metadata_plan; // stable borrowed identity for the actual P3 source
    std::shared_ptr<const NativeInventory> inventory;
    std::vector<int32_t> eog_stage; // survives every possibly enqueued upload
    TinySamplerQuote sampler_quote{};
    TinyCtxQuote ctx_quote{};
    TinyOwnerQuote owner_quote{};
    TinyStorageQuote mover_quote{};
    TinyStepQuote step_quote{};
    std::unique_ptr<TinyPoolAllocator> pool_allocator;
    bool attempted=false,retiring=false;
    ~EngineTinyState();
};
// Fixed options factory, not a caller-supplied provider/catalog/Ready callback.
class TinyNativeHarness {
public:
    static int create(const TinyEngineOptions&,std::unique_ptr<Engine>* out);
    static int snapshot(const Engine&,TinyEngineSnapshot* out);
    static int retire(Engine&);
};
}
