#pragma once
// Manual internal actual-GLM research entry. No CLI, HTTP admission or auto-run.
#include "build/glm_main_source_data/native_source.h"
#include "build/glm_main_ordinary_loader.h"
#include "build/tiny_component_quotes.h"
#include "build/tiny_startup.h"
#include "place/native_inventory.h"
#include "sched/geometry.h"
#include <array>
namespace rad {
int glm_main_one_prompt_admission(std::span<const int32_t>,int64_t vocabulary,int64_t context)noexcept;
// Pure metadata facts. No payload pointer, source capability or Ready state.
struct GlmMainVocabularyFacts {
    int64_t model_rows=0,model_width=0,tokenizer_tokens=0,view_tokens=0;
    int64_t embedding_rows=0,embedding_cols=0,embedding_bytes=0;
    int64_t head_rows=0,head_cols=0,head_bytes=0;
    uint32_t embedding_dtype=0,head_dtype=0;
    bool embedding_physical_plain=false,head_physical_plain=false;
};
struct GlmMainVocabularyAdmission {
    int64_t model_rows=0,valid_tokens=0,padded_rows=0,local_rows=0;
    int32_t special_ids=0,eog_ids=0;
};
int glm_main_vocabulary_guard(const GlmMainVocabularyFacts&,std::span<const int32_t> specials,
    std::span<const int32_t> eog,GlmMainVocabularyAdmission*)noexcept;
class Engine;class CachePrepare4RankBinding;class GlmMainTp4VerifiedSourceFactory;
enum class GlmMainEnginePhase:uint32_t {Requested,ColdAdmitted,MetadataOwned,Plugins,Declared,SourceVerified,NativePlanned,SourceAdmitted,Instances,Pools,Mover,OrdinaryLoaded,Contexts,TransportReady,TextComplete,Terminal,Retired};
struct GlmMainEngineOptions {
    std::string checkpoint,home,bundle_root,source_proof;
    uint64_t expected_bundle_id=0;int64_t context=512;
    // Exact first functional profile: batch1/seq1/spec0/exact wire/no residents.
    uint64_t hot_bytes_per_rank=64u<<20;
    uint64_t metadata_limit=4ull<<30,host_payload_limit=8ull<<30;
    uint64_t gpu_headroom=256u<<20;
    uint64_t loader_input_limit=2ull<<30,loader_output_limit=2ull<<30,loader_column_limit=64u<<20;
    uint64_t thread_stack_limit=64u<<20;
    uint32_t manual_timeout_seconds=1800;
};
struct GlmMainEngineSnapshot {
    GlmMainEnginePhase phase=GlmMainEnginePhase::Requested,failed_phase=GlmMainEnginePhase::Requested;
    int first_status=RAD_OK;
    uint64_t attempts=0,bundle_id=0,source_bytes_verified=0,mapped_bytes_compared=0;
    uint64_t metadata_reserved=0,host_payload_reserved=0,thread_stack_bytes=0;
    uint64_t backend_stream_requested=0,backend_control_reserved=0,sampler_requested=0;
    uint64_t batch_pinned_requested=0,batch_device_requested=0,batch_events=0,event_requested=0;
    uint64_t actual_quantum=0,actual_state_chunk=0,actual_attention_block=0,actual_generic_minimum=0;
    std::array<uint64_t,4> device_requested{},weights_requested{},kv_requested{},arena_requested{};
    std::array<uint64_t,4> inventory_metadata{},context_requested{},owner_requested{},mover_requested{};
    std::array<uint64_t,4> ordinary_jobs{},omitted_views{};
    GlmMainOrdinaryLoaderReceipt loader;
    GlmMainVocabularyAdmission vocabulary;
    bool source_admitted=false,transport_ready=false,text_complete=false,retired=false;
    int32_t sampled_token=-1;
};
// Pool/stream owners are real; Ctx only borrows this stream. Binding survives it.
class GlmMainComputeStream final {
public:
    ~GlmMainComputeStream();int open(int device,CachePrepare4RankBinding*);void retire()noexcept;
    RadStream get()const noexcept{return stream_;}
private:
    RadStream stream_=nullptr;int device_=-1;CachePrepare4RankBinding* binding_=nullptr;
};
struct EngineGlmMainState {
    GlmMainEngineOptions options;GlmMainEngineSnapshot result;
    glm_main_data::ReaderLimits reader_limits;
    std::shared_ptr<const glm_main_data::VerifiedBundle> bundle;
    std::array<ChunkGeometry,4> geometry;
    std::array<Plan,4> metadata_plans; // stable through last source.close
    std::array<std::shared_ptr<const glm_main_data::VerifiedRankMain>,4> authorities;
    std::array<std::shared_ptr<glm_main_data::GlmMainNativeSource>,4> sources;
    std::array<std::shared_ptr<const NativeInventory>,4> inventories;
    std::shared_ptr<glm_main_provider::ProcessBudget> process;
    std::unique_ptr<GlmMainOrdinaryLoader> ordinary;
    std::array<TinySamplerQuote,4> sampler_quotes;std::array<TinyCtxQuote,4> context_quotes;
    std::array<TinyOwnerQuote,4> owner_quotes;std::array<TinyStorageQuote,4> mover_quotes;
    TinyBackendQuote backend;
    std::vector<int32_t> eog_stage; // survives every possibly queued copy
    std::string manual_prompt;
    bool attempted=false,retiring=false,async_started=false,manual_requested=false;
    bool registry_owned=false,process_lease=false;
};
class GlmMainEngineHarness {
public:
    static int create(const GlmMainEngineOptions&,std::unique_ptr<Engine>* out)noexcept;
    static int snapshot(const Engine&,GlmMainEngineSnapshot*)noexcept;
    // CPU-only actual Checkpoint+Tokenizer parse and SAME production guard.
    // Never calls start(), loads plugins, selects a device, creates a stream,
    // admits a source or allocates a model. Output has no borrowed lifetime.
    static int vocabulary_preflight(const std::string& checkpoint,GlmMainVocabularyAdmission* out)noexcept;
    static int retire(Engine&)noexcept;
    // One bounded request through existing scheduler/model/sampler/collect.
    // Never opens an HTTP server. Requires successful start.
    static int one_token(Engine&,const std::string& prompt,int32_t* token)noexcept;
};
}
