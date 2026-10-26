#include "engine_priv.h"
#include "build/glm_main_quote.h"
#include "mem/vram_budget.h"
#include "runtime/kvbind.h"
#include "util/stallwatch.h"
#include "server/sink.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <new>
#include <thread>
#include <unistd.h>
namespace rad {
namespace {
constexpr uint64_t MiB=1ull<<20,GiB=1ull<<30,stride=3342336,cold=8*stride;
std::atomic<bool> manual_process_owner{false};
// Requested planning envelopes, admitted BEFORE Checkpoint/Tokenizer/Sampler.
// Includes shared bundle allowance PER rank token, not physical duplicate bytes.
constexpr uint64_t metadata_envelope=(256+4*192+256+4*128+4*128+1024+4*128+64)*MiB;
bool path_text(const std::string& s)noexcept{return !s.empty()&&s.size()<=4096&&s.find('\0')==std::string::npos;}
int request_valid(const GlmMainEngineOptions& o)noexcept {
    if(!path_text(o.checkpoint)||!path_text(o.home)||!path_text(o.bundle_root)||!path_text(o.source_proof)||
       !o.expected_bundle_id||o.context<128||o.context>512||o.context%128||
       o.hot_bytes_per_rank>GiB||o.metadata_limit<metadata_envelope||o.metadata_limit>8*GiB||
       o.host_payload_limit<GiB||o.host_payload_limit>16*GiB||
       o.loader_input_limit<stride||o.loader_input_limit>2*GiB||o.loader_output_limit<stride||o.loader_output_limit>2*GiB||
       !o.loader_column_limit||o.loader_column_limit>64*MiB||o.gpu_headroom<96*MiB||o.gpu_headroom>GiB||
       !o.thread_stack_limit||o.thread_stack_limit>64*MiB||!o.manual_timeout_seconds||o.manual_timeout_seconds>7200)return RAD_E_INVAL;
    uint64_t host=o.loader_input_limit+o.loader_output_limit+o.loader_column_limit+8*MiB;
    if(!tiny_quote::add(host,4*(o.hot_bytes_per_rank+cold+30228480))||host>o.host_payload_limit)return RAD_E_NOMEM;
    return RAD_OK;
}
int data_status(glm_main_data::Error e)noexcept {
    switch(e){case glm_main_data::Error::ok:return RAD_OK;case glm_main_data::Error::no_memory:return RAD_E_NOMEM;
    case glm_main_data::Error::budget:return RAD_E_SCRATCH;case glm_main_data::Error::io:case glm_main_data::Error::checksum:return RAD_E_IO;
    case glm_main_data::Error::unsupported:return RAD_E_UNSUPPORTED;default:return RAD_E_INVAL;}
}
[[noreturn]]void startup_failure(void* pointer,int status)noexcept {
    auto* binding=static_cast<CachePrepare4RankBinding*>(pointer);
    if(binding)binding->fail_stop(status);
    glm53::cache_prepare4::process_fail_stop();
}
void phase(EngineGlmMainState& s,GlmMainEnginePhase p)noexcept{s.result.phase=p;}
uint64_t sum(uint64_t a,uint64_t b){if(b>UINT64_MAX-a)throw std::bad_alloc();return a+b;}
uint64_t round_page(uint64_t n){return sum(n,4095)/4096*4096;}
}
int glm_main_vocabulary_guard(const GlmMainVocabularyFacts& f,std::span<const int32_t> specials,
    std::span<const int32_t> eog,GlmMainVocabularyAdmission* out)noexcept {
    if(!out)return RAD_E_INVAL;
    constexpr int64_t rows=154880,width=4096,tokens=154856,bytes=rows*width*2;
    if(f.model_rows!=rows||f.model_width!=width||f.tokenizer_tokens!=tokens||f.view_tokens!=tokens||
       f.embedding_rows!=rows||f.embedding_cols!=width||f.embedding_bytes!=bytes||f.embedding_dtype!=RAD_BF16||!f.embedding_physical_plain||
       f.head_rows!=rows||f.head_cols!=width||f.head_bytes!=bytes||f.head_dtype!=RAD_BF16||!f.head_physical_plain||
       specials.size()!=5||eog.empty()||eog.size()>64)return RAD_E_FORMAT;
    for(const int32_t id:specials)if(id < -1||id>=tokens)return RAD_E_FORMAT;
    for(size_t i=0;i<eog.size();++i){if(eog[i]<0||eog[i]>=tokens)return RAD_E_FORMAT;
        for(size_t j=0;j<i;++j)if(eog[j]==eog[i])return RAD_E_FORMAT;}
    GlmMainVocabularyAdmission admitted;admitted.model_rows=rows;admitted.valid_tokens=tokens;
    admitted.padded_rows=rows-tokens;admitted.local_rows=rows/4;
    admitted.special_ids=int32_t(specials.size());admitted.eog_ids=int32_t(eog.size());
    *out=admitted;return RAD_OK;
}
int Engine::glm_main_vocabulary_admit(){
    if(!glm_main_||!ckpt_||!vocab_||!vocab_view_)return RAD_E_STATE;
    if(!meta_.arch_id||std::strcmp(meta_.arch_id,"glm5next")||meta_.n_layers!=45||
       meta_.n_embd!=4096||meta_.n_expert!=288||meta_.n_expert_used!=8)return RAD_E_SHAPE;
    GlmMainVocabularyFacts f;f.model_rows=meta_.n_vocab;f.model_width=meta_.n_embd;
    f.tokenizer_tokens=vocab_->n_tokens();f.view_tokens=vocab_view_->n_tokens();
    auto tensor=[&](const SrcTensor* t,int64_t& rows,int64_t& cols,int64_t& bytes,uint32_t& dtype,bool& plain){
        if(!t)return;bytes=t->bytes;dtype=t->rad_dtype;
        if(t->shape.size()==2){rows=t->shape[0];cols=t->shape[1];}
        plain=t->data&&!t->ggml&&!t->gptq_bits&&!t->gptq_group&&!t->ct_group&&!t->ct_zero&&!t->pbits&&
            !t->aux[0]&&!t->aux[1]&&!t->aux[2]&&t->shape.size()==2;
    };
    tensor(ckpt_->find("model.language_model.embed_tokens.weight"),f.embedding_rows,f.embedding_cols,f.embedding_bytes,f.embedding_dtype,f.embedding_physical_plain);
    tensor(ckpt_->find("lm_head.weight"),f.head_rows,f.head_cols,f.head_bytes,f.head_dtype,f.head_physical_plain);
    const std::array<int32_t,5> specials{vocab_->bos(),vocab_->eos(),vocab_->eot(),vocab_->unk(),vocab_->pad()};
    return glm_main_vocabulary_guard(f,specials,vocab_->eog(),&glm_main_->result.vocabulary);
}
int GlmMainEngineHarness::vocabulary_preflight(const std::string& checkpoint,GlmMainVocabularyAdmission* out)noexcept {
    if(!out||!path_text(checkpoint))return RAD_E_INVAL;
    try{
        Config c;c.model=checkpoint;auto e=std::make_unique<Engine>(std::move(c));
        e->glm_main_=std::make_unique<EngineGlmMainState>();
        int st=e->read_metadata();if(st<0)return st;
        st=e->load_vocab();if(st<0)return st;
        st=e->glm_main_vocabulary_admit();if(st<0)return st;
        *out=e->glm_main_->result.vocabulary;return RAD_OK;
    }catch(const std::bad_alloc&){return RAD_E_NOMEM;}catch(...){return RAD_E_STATE;}
}
int glm_main_one_prompt_admission(std::span<const int32_t> tokens,int64_t vocabulary,int64_t context)noexcept {
    return tokens.size()==1&&vocabulary>0&&context>=2&&tokens[0]>=0&&tokens[0]<vocabulary?RAD_OK:RAD_E_SHAPE;
}
GlmMainComputeStream::~GlmMainComputeStream(){retire();}
int GlmMainComputeStream::open(int device,CachePrepare4RankBinding* binding){
    if(stream_||binding_||device<0||!binding||!binding->runtime_admitted())return RAD_E_STATE;
    binding_=binding;device_=device;
    int st=rad_dev_set(device);if(st<0)binding_->fail_stop(st);
    st=rad_stream_create(&stream_,0);if(st<0)binding_->fail_stop(st);
    if(!stream_)binding_->fail_stop(RAD_E_STATE);return RAD_OK;
}
void GlmMainComputeStream::retire()noexcept {
    if(!stream_)return;
    if(!binding_)glm53::cache_prepare4::process_fail_stop();
    if(const int s=rad_dev_set(device_);s<0)binding_->fail_stop(s);
    if(const int s=rad_stream_sync(stream_);s<0)binding_->fail_stop(s);
    rad_stream_destroy(stream_);stream_=nullptr;
}
int GlmMainEngineHarness::create(const GlmMainEngineOptions& o,std::unique_ptr<Engine>* out)noexcept {
    if(!out||*out)return RAD_E_INVAL;
    const int valid=request_valid(o);if(valid<0)return valid;
    try {
        auto state=std::make_unique<EngineGlmMainState>();state->options=o;
        state->result.metadata_reserved=metadata_envelope;
        state->result.host_payload_reserved=o.loader_input_limit+o.loader_output_limit+o.loader_column_limit+8*MiB+4*(o.hot_bytes_per_rank+cold+30228480);
        Config c;c.model=o.checkpoint;c.radiance_home=o.home;c.tp=4;c.tp_wire_exact=1;c.tp_wire_min_kb=0;
        c.max_tok=1;c.max_seqs=1;c.max_ctx=o.context;c.n_spec=0;c.deterministic=true;
        c.placement="all_vram";c.host_pool_mib=0;c.weights_disk_tier=false;
        c.embedding_placement=c.embedding_resolved=TablePlacement::Vram;c.embedding_host_mib=c.tower_host_mib=0;
        c.tower_placement=TablePlacement::Vram;c.prefix_cache=false;c.prefix_cache_mib=0;
        c.checkpoint_slots=0;c.checkpoint_interval=0;c.prefill_span=1;c.mm_max_patches=0;c.mm_audio_frames=0;
        c.draft_coupling=false;c.kv_cache_dtype="bf16";c.gpu_headroom_mib=int64_t(o.gpu_headroom/MiB);
        auto e=std::make_unique<Engine>(std::move(c));e->glm_main_=std::move(state);*out=std::move(e);return RAD_OK;
    }catch(const std::bad_alloc&){return RAD_E_NOMEM;}catch(...){return RAD_E_STATE;}
}
int GlmMainEngineHarness::snapshot(const Engine& e,GlmMainEngineSnapshot* out)noexcept {
    if(!out||!e.glm_main_)return RAD_E_INVAL;*out=e.glm_main_->result;return RAD_OK;
}
int GlmMainEngineHarness::retire(Engine& e)noexcept {if(!e.glm_main_)return RAD_E_INVAL;e.glm_main_shutdown();return RAD_OK;}
int Engine::start_glm_main(){auto& s=*glm_main_;
    if(s.attempted||s.retiring||s.result.retired||s.result.phase==GlmMainEnginePhase::Terminal)
        return s.result.first_status<0?s.result.first_status:RAD_E_STATE;
    s.attempted=true;s.result.attempts=1;int status=RAD_E_STATE;
    try{status=glm_main_body();}catch(const std::bad_alloc&){status=RAD_E_NOMEM;}catch(...){status=RAD_E_STATE;}
    if(status<0){
        // A known error after peer init/async work must never join or unwind live
        // peer collectives. Before that boundary metadata failures are returnable.
        if(s.async_started&&cache_prepare4_)cache_prepare4_->fail_stop(-1,status);
        s.result.first_status=status;s.result.failed_phase=s.result.phase;s.result.phase=GlmMainEnginePhase::Terminal;
        glm_main_shutdown();
    }
    return status;
}
int Engine::glm_main_declare(){auto& s=*glm_main_;
    if(ranks_.size()!=4||!ckpt_||!ckpt_src_||plugins_.arch_plugin!="glm5next_cache")return RAD_E_UNSUPPORTED;
    // No opaque instances: serial metadata declarations use the actual card's
    // selection facts, actual source cache and actual Sampler/PERSIST output.
    for(auto& r:ranks_){RAD_TRY(rad_dev_set(r->device));
        RAD_TRY(rad_tools_declare(plugins_,meta_,build_ctx(r->index,1),&r->program,
            [this,index=r->index](RadBuilder* b,int64_t width){return declare_sampler(index,b,width);},
            nullptr,sources(),false,nullptr));
        RAD_TRY(glm_main_tp4_cache_program_profile(r->program));
        TinySamplerQuote sampler_actual;RAD_TRY(r->sampler.glm_main_startup_quote(&sampler_actual));
        if(sampler_actual.storage.page_and_heap_total!=s.sampler_quotes[size_t(r->index)].storage.page_and_heap_total||
           sampler_actual.candidates!=s.sampler_quotes[size_t(r->index)].candidates)return RAD_E_STATE;
        if(r->program.ctx.max_tok!=1||r->program.buffers.size()>20000||r->program.ops.size()>10000||r->program.weights.size()>60000)return RAD_E_SHAPE;
        if(!r->program.device_holes(false).empty())return RAD_E_NOTFOUND; // exact predicate underlying miss_report
        RAD_TRY(rad_gate_reference_kernels(r->program,false,"actual GLM main cache"));
        if(!r->tok_buf||r->tok_buf>=r->program.buffers.size())return RAD_E_SHAPE;
        const auto& token=r->program.buffers[r->tok_buf];
        if(token.name!="sampler.tokens"||token.decl.kind!=RAD_BUF_PERSIST||token.decl.dtype!=RAD_I32||token.decl.rank!=1||token.decl.shape[0]!=1)return RAD_E_SHAPE;
    }
    if(!ckpt_src_->refused().empty())return RAD_E_UNSUPPORTED;
    // Validate all actual row paths/constraints BEFORE source scan, Native fit,
    // opaque collective init, Pools or ordinary payload allocations.
    for(size_t i=0;i<4;++i){RAD_TRY(glm_main_final_token_geometry_resolve(ranks_[i]->program,cfg_,&s.geometry[i]));
        if(i&&(s.geometry[i].quantum!=s.geometry[0].quantum||s.geometry[i].state_chunk!=s.geometry[0].state_chunk||
               s.geometry[i].attn_block!=s.geometry[0].attn_block||s.geometry[i].max_chunk!=s.geometry[0].max_chunk))return RAD_E_SHAPE;}
    s.result.actual_quantum=s.geometry[0].quantum;s.result.actual_state_chunk=s.geometry[0].state_chunk;
    s.result.actual_attention_block=s.geometry[0].attn_block;s.result.actual_generic_minimum=s.geometry[0].min_max_tok;
    phase(s,GlmMainEnginePhase::Declared);return RAD_OK;
}
// Sole Session-admission friend. No public catalog/scalar/Ready setter.
class GlmMainTp4VerifiedSourceFactory final {
public:
    static int prepare(Engine& e){auto& s=*e.glm_main_;std::string why;
        if(!e.ckpt_||!e.ckpt_src_||e.ranks_.size()!=4||e.cache_prepare4_)return RAD_E_STATE;
        for(size_t rank=0;rank<4;++rank){auto& r=*e.ranks_[rank];
            Config c=e.cfg_;c.vram_weights_mib=0;c.vram_kv_mib=0;
            PlannerInput in{&r.program,&c,int(rank),4,nullptr,0,false,false};
            RAD_TRY(Planner(in).plan(&s.metadata_plans[rank]));
        }
        glm_main_data::Request request{s.options.bundle_root,s.options.source_proof,s.options.expected_bundle_id};
        RAD_TRY(data_status(glm_main_data::VerifiedBundle::open(request,*e.ckpt_,s.reader_limits,&s.bundle,&why)));
        const auto receipt=s.bundle->receipt();
        if(!receipt.complete_main_domain||receipt.records!=48384||receipt.shards!=168||receipt.native_bytes_verified!=161715585024ull||receipt.bundle_id!=s.options.expected_bundle_id)return RAD_E_INVAL;
        s.result.bundle_id=receipt.bundle_id;s.result.source_bytes_verified=receipt.native_bytes_verified;s.result.mapped_bytes_compared=receipt.mapped_source_bytes_compared;
        glm_main_provider::ProcessLimits process_limits;process_limits.max_rank_owners=4;process_limits.max_open_fds=48;
        process_limits.max_metadata_bytes=GiB;process_limits.max_cache_reservations=4*(s.options.hot_bytes_per_rank+cold);
        if(glm_main_provider::ProcessBudget::create(process_limits,&s.process)!=glm_main_provider::Error::ok)return RAD_E_NOMEM;
        e.rank_cfg_.resize(4);
        for(size_t rank=0;rank<4;++rank){auto& r=*e.ranks_[rank];
            RAD_TRY(data_status(s.bundle->bind_rank(*e.ckpt_,*e.ckpt_src_,r.program,s.metadata_plans[rank],&s.authorities[rank],&why)));
            if(s.authorities[rank]->generation()!=receipt.bundle_id||s.authorities[rank]->owned().size()!=12096)return RAD_E_INVAL;
            glm_main_provider::Limits limits;limits.hot_bytes=s.options.hot_bytes_per_rank;limits.cold_bytes=cold;limits.max_record_bytes=stride;
            limits.max_metadata_bytes=256*MiB;limits.max_records=12096;limits.max_batch=8;limits.max_shards=42;limits.max_segments=42;
            limits.max_payload_fds=2;limits.max_rank_fd_credits=12;limits.max_live_records=12096;limits.max_live_leases=64;limits.io_chunk_bytes=65536;
            RAD_TRY(glm_main_data::GlmMainNativeSource::open(s.process,s.authorities[rank],limits,&s.sources[rank],&why));
            NativeInventoryLimits caps;caps.backend_profile=NativeBackendProfile::GlmMainTp4AqlFailStopV1;
            caps.max_rows=1;caps.top_k=8;caps.max_owned_units=12096;caps.max_classes=1;caps.max_regions=9;
            caps.max_device_bytes=caps.max_landing_bytes=32*MiB;caps.max_weights_bytes=8*GiB;
            caps.max_metadata_bytes=128*MiB;caps.hot_bytes=s.options.hot_bytes_per_rank;caps.cold_bytes=cold;
            caps.max_host_bytes=s.options.hot_bytes_per_rank+cold+32*MiB+128*MiB;
            RAD_TRY(NativeInventory::bind(r.program,s.metadata_plans[rank],s.sources[rank]->catalog(),s.authorities[rank]->owned(),{},caps,&s.inventories[rank],&why));
            RAD_TRY(rad_dev_set(r.device));
            RAD_TRY(r.kv.plan_groups(r.program.kv_groups,e.cfg_));const int64_t kv=r.kv.ceiling(e.cfg_.max_ctx);
            if(kv<=0||uint64_t(kv)>GiB)return RAD_E_SCRATCH;
            Config base=e.cfg_;base.vram_kv_mib=int64_t((uint64_t(kv)+MiB-1)/MiB);
            RAD_TRY(s.inventories[rank]->configure_pools(base,&e.rank_cfg_[rank]));
            PlannerInput final{&r.program,&e.rank_cfg_[rank],int(rank),4,nullptr,0,false,false};final.native_inventory=s.inventories[rank];
            RAD_TRY(Planner(final).plan(&r.plan));RAD_TRY(s.inventories[rank]->validate(r.program,r.plan));
            s.result.inventory_metadata[rank]=s.inventories[rank]->footprint().metadata_reservation;
        }
        phase(s,GlmMainEnginePhase::SourceVerified);
        std::array<GlmMainOrdinaryRankSpec,4> jobs;
        for(size_t i=0;i<4;++i)jobs[i]={&e.ranks_[i]->program,&e.ranks_[i]->plan,e.ckpt_src_.get()};
        GlmMainOrdinaryLoaderLimits load_limits;load_limits.max_input_bytes=s.options.loader_input_limit;load_limits.max_output_bytes=s.options.loader_output_limit;
        load_limits.max_column_bytes=s.options.loader_column_limit;load_limits.max_buffer_bytes=load_limits.max_input_bytes+load_limits.max_output_bytes+load_limits.max_column_bytes;
        GlmMainOrdinaryLoaderQuote load_quote;RAD_TRY(GlmMainOrdinaryLoader::prepare(jobs,load_limits,&s.ordinary,&load_quote,&why));
        s.result.loader.quote=load_quote;
        // Actual backend quote is per actual bound card; all requested owners
        // and stream backing are checked before opaque init or Pools allocation.
        uint64_t stack=0,guard=0;RAD_TRY(tiny_thread_defaults(&stack,&guard));
        if(stack>s.options.thread_stack_limit||guard>s.options.thread_stack_limit-stack)return RAD_E_SCRATCH;
        s.result.thread_stack_bytes=5*(stack+guard); //4rank +stepper, init4 not concurrent with them
        s.result.backend_control_reserved=4*MiB;
        uint64_t extra_host=0;
        for(size_t i=0;i<4;++i){auto& r=*e.ranks_[i];RAD_TRY(rad_dev_set(r.device));TinyBackendQuote backend;
            RAD_TRY(tiny_backend_quote(&backend));if(backend.host||backend.backend_threads_per_stream||!backend.stream_size||!backend.event_size||!backend.stream_karg_bytes||!backend.stream_bounce_bytes)return RAD_E_UNSUPPORTED;
            s.backend=backend;
            const auto stream=sum(backend.stream_packet_bytes,sum(backend.stream_karg_bytes,sum(backend.stream_bounce_bytes,sum(backend.stream_size,backend.stream_control_allowance))));
            s.result.backend_stream_requested=sum(s.result.backend_stream_requested,3*stream); //compute+mover+per-card backend spare/shared lane allowance
            extra_host=sum(extra_host,3*stream);
            CtxDesc desc;desc.program=&r.program;desc.rank=int(i);desc.world_size=4;desc.device=r.device;desc.record_passes=false;desc.profile_ops=false;
            desc.expert_prepare_limits.max_rows=1;desc.expert_prepare_limits.max_top_k=8;desc.expert_prepare_limits.max_experts=288;
            desc.expert_prepare_limits.max_maps=42;desc.expert_prepare_limits.max_entries=12096;
            RAD_TRY(Ctx::glm_main_startup_quote(desc,&s.context_quotes[i]));
            RAD_TRY(NativeBackingOwner::glm_main_startup_quote(r.program,r.plan,&s.owner_quotes[i]));
            RAD_TRY(Mover::glm_main_startup_quote(r.program,r.plan,&s.mover_quotes[i]));
            const auto& q=s.context_quotes[i];s.result.context_requested[i]=q.storage.page_and_heap_total;s.result.owner_requested[i]=s.owner_quotes[i].storage.page_and_heap_total;s.result.mover_requested[i]=s.mover_quotes[i].page_and_heap_total;
            const auto weights=uint64_t(e.rank_cfg_[i].vram_weights_mib)*MiB,cache=uint64_t(e.rank_cfg_[i].vram_kv_mib)*MiB;
            KVGeom kg=KVGeom::of(r.program.kv_groups);
            for(size_t group=0;group<kg.g.size();++group){const auto width=r.kv.state_index_width(int32_t(group));if(width>0)kg.g[group].sidx_width=width;}
            uint64_t batch=0,pinned=0,events=0;RAD_TRY(BatchBuilder::glm_main_staging_quote(r.program,e.cfg_,kg,&batch,&pinned,&events));
            if(i&&(batch!=s.result.batch_device_requested||pinned!=s.result.batch_pinned_requested||events!=s.result.batch_events))return RAD_E_UNSUPPORTED;
            s.result.batch_device_requested=batch;s.result.batch_pinned_requested=pinned;s.result.batch_events=events;
            const uint64_t event_count=q.storage.event_count+s.mover_quotes[i].event_count+2+events/4;
            const uint64_t event_bytes=event_count*backend.event_size;if(backend.event_size&&event_bytes/backend.event_size!=event_count)return RAD_E_SHAPE;
            s.result.event_requested=sum(s.result.event_requested,event_bytes);
            uint64_t device=sum(weights,sum(cache,q.storage.device_page_bytes));
            device=sum(device,round_page(uint64_t(batch)));
            // Actual StepSlot+EOG and two per-card streams' non-HOST kargs. The
            // packet ring/driver backing is conservatively covered as well.
            device=sum(device,round_page(sizeof(StepSlot))+round_page(std::max<size_t>(1,e.vocab_->eog().size())*sizeof(int32_t)));
            device=sum(device,3*(backend.stream_packet_bytes+(backend.karg_host?0:backend.stream_karg_bytes)));
            device=sum(device,s.options.gpu_headroom);
            RadDeviceProps props{};RAD_TRY(rad_dev_props(r.device,&props));
            if(props.vram_free<=0||device>uint64_t(props.vram_free))return RAD_E_NOMEM;
            s.result.device_requested[i]=device;s.result.weights_requested[i]=weights;s.result.kv_requested[i]=cache;s.result.arena_requested[i]=q.arena_device;
            extra_host=sum(extra_host,q.storage.pinned_page_bytes+q.storage.mapped_page_bytes+s.inventories[i]->footprint().landing_bytes);
        }
        extra_host=sum(extra_host,round_page(s.result.batch_pinned_requested));
        if(s.result.event_requested>s.result.backend_control_reserved)return RAD_E_NOMEM;
        if(extra_host>s.options.host_payload_limit-s.result.host_payload_reserved)return RAD_E_NOMEM;
        s.result.host_payload_reserved+=extra_host;
        if(s.result.thread_stack_bytes>s.options.metadata_limit-s.result.metadata_reserved)return RAD_E_NOMEM;
        s.result.metadata_reserved+=s.result.thread_stack_bytes;
        phase(s,GlmMainEnginePhase::NativePlanned);
        // Prepare allocation BEFORE final audits and private capability commit.
        static std::atomic<uint64_t> model_ids{1};uint64_t model=model_ids.load(std::memory_order_relaxed);
        for(;;){if(!model||model==UINT64_MAX)return RAD_E_STATE;
            if(model_ids.compare_exchange_weak(model,model+1,std::memory_order_relaxed))break;}
        auto session=std::unique_ptr<CachePrepare4Session>(new CachePrepare4Session(model,receipt.bundle_id));
        RAD_TRY(data_status(s.bundle->audit()));
        for(size_t i=0;i<4;++i){RAD_TRY(data_status(s.authorities[i]->audit()));RAD_TRY(s.inventories[i]->validate(e.ranks_[i]->program,e.ranks_[i]->plan));}
        session->source_admitted_=true;e.cache_prepare4_=std::move(session);s.result.source_admitted=true;
        phase(s,GlmMainEnginePhase::SourceAdmitted);return RAD_OK;
    }
};
int Engine::glm_main_body(){auto& s=*glm_main_;
    RAD_TRY(request_valid(s.options));phase(s,GlmMainEnginePhase::ColdAdmitted);
    // HOST negative path stops before any source open, plugin load or Sampler.
    if(device_is_host()||std::strcmp(device_backend_name(),"aql")||rad_dev_count()<4)return RAD_E_UNSUPPORTED;
    for(const char* key:{"RADIANCE_OP_ORACLE","RADIANCE_OP_REPEAT"}){const char* v=std::getenv(key);if(v&&*v)return RAD_E_UNSUPPORTED;}
    bool expected=false;if(!manual_process_owner.compare_exchange_strong(expected,true))return RAD_E_STATE;
    s.process_lease=true;
    // Registry is process-global. This manual profile never adopts another
    // Engine's loaded plugins/instances or closes their ownership on refusal.
    if(!rad_tools_registry().plugins().empty()||rad_tools_registry().n_instances())return RAD_E_STATE;
    s.registry_owned=true;
    RAD_TRY(read_metadata());RAD_TRY(load_vocab());if(!vocab_||!vocab_view_||meta_.n_layers!=45||meta_.n_embd!=4096||meta_.n_expert!=288||meta_.n_expert_used!=8)return RAD_E_SHAPE;
    RAD_TRY(glm_main_vocabulary_admit());
    phase(s,GlmMainEnginePhase::MetadataOwned);
    rad_tools_registry().set_code_device(0);rad_tools_registry().set_no_device_code(false);
    // Deliberately conservative host-mailbox collectives for first functionality.
    rad_tools_registry().set_device_fact(RAD_FACT_P2P,0);
    RAD_TRY(load_plugins());RAD_TRY(glm_main_tp4_cold_source_gate());if(!arch_step_||!ckpt_)return RAD_E_UNSUPPORTED;
    ckpt_->set_importers(&rad_tools_registry());phase(s,GlmMainEnginePhase::Plugins);
    ranks_.reserve(4);for(int i=0;i<4;++i){RadDeviceProps props{};RAD_TRY(rad_dev_props(i,&props));if(std::strcmp(props.arch,"gfx1201"))return RAD_E_UNSUPPORTED;
        auto r=std::make_unique<Rank>();r->index=r->device=i;ranks_.push_back(std::move(r));}
    RAD_TRY(declare());RAD_TRY(GlmMainTp4VerifiedSourceFactory::prepare(*this));
    size_t count=0;for(const auto& rank:ranks_)for(const auto& op:rank->program.ops)for(const auto& band:op.bands)for(const auto& row:band.dom)
        if(row.row&&row.row->info&&row.row->info->init){if(count==SIZE_MAX)return RAD_E_SHAPE;++count;}
    RAD_TRY(rad_tools_registry().reserve_instance_tracking(count));
    std::vector<std::thread> workers;workers.reserve(4);
    s.async_started=true; // before any peer init can rendezvous; all later errors exit88
    try{for(size_t rank=0;rank<4;++rank)workers.emplace_back([this,rank]{auto* binding=cache_prepare4_->binding(int(rank));
        try{const int bound=rad_dev_set(ranks_[rank]->device);if(bound<0)binding->fail_stop(bound);
            const int status=initialize_program_instances(rad_tools_registry(),ranks_[rank]->program,{},startup_failure,binding);
            if(status<0)binding->fail_stop(status);
        }catch(...){binding->fail_stop(RAD_E_STATE);}});
        for(auto& worker:workers)worker.join();
    }catch(...){cache_prepare4_->fail_stop(-1,RAD_E_STATE);}
    phase(s,GlmMainEnginePhase::Instances);
    // Kernel init may allocate opaque vendor memory: recheck the same requested
    // model budget against actual current free bytes before any Pools commit.
    for(size_t rank=0;rank<4;++rank){auto& r=*ranks_[rank];RAD_TRY(rad_dev_set(r.device));RadDeviceProps props{};RAD_TRY(rad_dev_props(r.device,&props));
        if(props.vram_free<=0||s.result.device_requested[rank]>uint64_t(props.vram_free))return RAD_E_NOMEM;
        RAD_TRY(r.pools.configure_native(rank_cfg_[rank],props.vram_bytes,r.plan));
    }
    phase(s,GlmMainEnginePhase::Pools);
    for(size_t rank=0;rank<4;++rank){auto& r=*ranks_[rank];auto* binding=cache_prepare4_->binding(int(rank));
        RAD_TRY(r.glm_compute.open(r.device,binding));MoverConfig config;config.glm_main_binding=binding;config.glm_main_device=r.device;
        RAD_TRY(r.mover.init(r.program,r.plan,config,&r.pools));RAD_TRY(r.mover.bind_compute_stream(r.glm_compute.get()));}
    phase(s,GlmMainEnginePhase::Mover);
    std::array<GlmMainOrdinaryRuntimeRank,4> runtime;
    for(size_t rank=0;rank<4;++rank){auto& r=*ranks_[rank];runtime[rank]={&r.mover,&r.pools,r.device,r.glm_compute.get(),cache_prepare4_->binding(int(rank))};}
    RAD_TRY(s.ordinary->execute(runtime,&s.result.loader));
    if(s.result.loader.completed_jobs!=s.result.loader.quote.jobs||s.result.loader.quote.omitted_views!=4ull*48384)return RAD_E_STATE;
    phase(s,GlmMainEnginePhase::OrdinaryLoaded);
    for(size_t rank=0;rank<4;++rank){auto& r=*ranks_[rank];RAD_TRY(rad_dev_set(r.device));RAD_TRY(s.sources[rank]->finish_bootstrap());
        RAD_TRY(NativeBackingOwner::open(r.program,r.plan,r.mover,s.sources[rank],&r.native_owner));}
    RAD_TRY(configure_kv());
    for(size_t rank=0;rank<4;++rank){auto& r=*ranks_[rank];RAD_TRY(rad_dev_set(r.device));CtxDesc d;
        d.program=&r.program;d.rank=int(rank);d.world_size=4;d.device=r.device;d.stream=r.glm_compute.get();d.residency=r.mover.residency();
        d.arch_step=arch_step_;d.record_passes=false;d.profile_ops=false;d.registry=&rad_tools_registry();d.expert_prepare_owner=r.native_owner.get();d.cache_prepare4=cache_prepare4_->binding(int(rank));
        d.expert_prepare_limits.max_rows=1;d.expert_prepare_limits.max_top_k=8;d.expert_prepare_limits.max_experts=288;d.expert_prepare_limits.max_maps=42;d.expert_prepare_limits.max_entries=12096;
        bind_kv_pools(r.program,[&](int32_t g){return r.kv.plan(g);},&d.kv);
        TinyCtxQuote actual;RAD_TRY(Ctx::glm_main_startup_quote(d,&actual));const auto& admitted=s.context_quotes[rank];
        if(actual.storage.device_page_bytes!=admitted.storage.device_page_bytes||actual.storage.pinned_page_bytes!=admitted.storage.pinned_page_bytes||
           actual.storage.mapped_page_bytes!=admitted.storage.mapped_page_bytes||actual.storage.heap_payload_bytes!=admitted.storage.heap_payload_bytes)return RAD_E_STATE;
        RAD_TRY(r.ctx.init(d));
    }
    phase(s,GlmMainEnginePhase::Contexts);RAD_TRY(step_state_init());
    for(size_t rank=0;rank<4;++rank){RAD_TRY(data_status(s.authorities[rank]->validate_program()));RAD_TRY(s.inventories[rank]->validate(ranks_[rank]->program,ranks_[rank]->plan));}
    RAD_TRY(data_status(s.bundle->audit()));RAD_TRY(model_policy_admission());
    phase(s,GlmMainEnginePhase::TransportReady);s.result.transport_ready=true;return RAD_OK;
}
void Engine::glm_main_shutdown()noexcept {
    if(!glm_main_||glm_main_->retiring||glm_main_->result.retired)return;auto& s=*glm_main_;s.retiring=true;
    try{
        if(cache_prepare4_)cache_prepare4_->stop_admission();
        stopping_.store(true,std::memory_order_release);if(barrier_)barrier_->abort();
        for(auto& worker:rank_threads_)if(worker.joinable())worker.join();rank_threads_.clear();
        for(auto& r:ranks_){if(rad_dev_set(r->device)<0&&cache_prepare4_)cache_prepare4_->fail_stop(r->index,RAD_E_DEVICE);
            r->mover.glm_main_retire_or_fail();
            if(r->glm_compute.get()&&rad_stream_sync(r->glm_compute.get())<0)cache_prepare4_->fail_stop(r->index,RAD_E_DEVICE);
            if(r->native_owner){const int status=r->native_owner->quiesce();if(status<0)cache_prepare4_->fail_stop(r->index,status);}}
        for(auto& r:ranks_){r->ctx.shutdown();if(r->native_owner){const int status=r->native_owner->close_after_context_shutdown();if(status<0)cache_prepare4_->fail_stop(r->index,status);}}
        step_state_fini();s.eog_stage.clear();
        // Scheduler/BatchBuilder buffers may refer to Program/streams/Pools.
        sched_.fini();kv_iface_.reset();pc_iface_.reset();
        for(auto& src:s.sources)if(src&&src->close()<0)cache_prepare4_->fail_stop(-1,RAD_E_STATE);
        for(auto& r:ranks_)r->native_owner.reset();
        // Successful init/fini pairs run while Program geometry, Pools and the
        // rank-owned streams still exist, after every model/Ctx consumer stops.
        if(s.registry_owned)rad_tools_registry().fini_instances();
        s.ordinary.reset();ranks_.clear();s.inventories={};s.sources={};s.authorities={};
        for(auto& plan:s.metadata_plans)plan=Plan{};s.bundle.reset();s.process.reset();
        // Registry-owned opaque init instances may retain checkpoint-independent
        // device buffers; fini while all device/runtime support still exists.
        if(s.registry_owned){rad_tools_registry().close();s.registry_owned=false;}
        ckpt_src_.reset();ckpt_.reset();cache_prepare4_.reset();
        if(s.process_lease){manual_process_owner.store(false,std::memory_order_release);s.process_lease=false;}
        s.result.transport_ready=false;s.result.retired=true;
        if(s.result.first_status>=0)s.result.phase=GlmMainEnginePhase::Retired;s.retiring=false;
    }catch(...){if(cache_prepare4_)cache_prepare4_->fail_stop(-1,RAD_E_STATE);glm53::cache_prepare4::process_fail_stop();}
}
int GlmMainEngineHarness::one_token(Engine& e,const std::string& prompt,int32_t* token)noexcept {
    if(!token||!e.glm_main_||!e.glm_main_->result.transport_ready||e.glm_main_->manual_requested||prompt.empty()||prompt.size()>4096)return RAD_E_INVAL;
    auto& s=*e.glm_main_;
    try{s.manual_prompt=prompt;s.manual_requested=true;const int status=e.serve(nullptr);
        if(status<0){if(e.cache_prepare4_)e.cache_prepare4_->fail_stop(-1,status);return status;}
        if(!s.result.text_complete||s.result.sampled_token<0)return RAD_E_STATE;*token=s.result.sampled_token;return RAD_OK;
    }catch(...){if(e.cache_prepare4_)e.cache_prepare4_->fail_stop(-1,RAD_E_STATE);return RAD_E_STATE;}
}
int Engine::glm_main_run_request(){auto& s=*glm_main_;
    try{
        auto request=std::make_unique<Request>();request->id=1;request->max_tokens=1;request->ignore_eos=true;
        request->sp.temp=0;request->sp.top_k=1;request->sp.seed=1;request->may_speculate=false;
        RAD_TRY(vocab_view_->tokenize(s.manual_prompt,&request->prompt));
        RAD_TRY(glm_main_one_prompt_admission(request->prompt,s.result.vocabulary.valid_tokens,cfg_.max_ctx));
        server::Sink sink(8);request->sink=&sink;sink.id=request->id;
        const int added=sched_.add(std::move(request));if(added<0){sink.fail(added);return added;}
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(s.options.manual_timeout_seconds);
        while(!sink.done()){
            if(std::chrono::steady_clock::now()>=deadline||loop_status_.load(std::memory_order_acquire)<0)cache_prepare4_->fail_stop(-1,RAD_E_STATE);
            (void)sink.wait(25);
        }
        while(cache_prepare4_->active()){
            if(std::chrono::steady_clock::now()>=deadline)cache_prepare4_->fail_stop(-1,RAD_E_STATE);
            std::this_thread::yield();
        }
        if(sink.status()<0)cache_prepare4_->fail_stop(-1,sink.status());
        server::SinkToken token;const auto n=sink.read(&token,1);
        if(n!=1||sink.n_produced()!=1||token.token<0||token.token>=s.result.vocabulary.valid_tokens)cache_prepare4_->fail_stop(-1,RAD_E_STATE);
        const int reaped=sched_.reap(1);if(reaped<0)cache_prepare4_->fail_stop(-1,reaped);
        s.result.sampled_token=token.token;s.result.text_complete=true;phase(s,GlmMainEnginePhase::TextComplete);return RAD_OK;
    }catch(...){cache_prepare4_->fail_stop(-1,RAD_E_STATE);}
}
} // namespace rad
