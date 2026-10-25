#include "cache_prepare4_session.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <new>
namespace rad {
namespace cp=glm53::cache_prepare4;
namespace {
int status(int s)noexcept{return s<=-10000?RAD_E_STATE:s;}
bool batch_pass(const RadBatch& b,uint64_t model,uint64_t source,uint64_t epoch,cp::Pass* out)noexcept{
    if(!out||b.step<0||b.n_tok<1||b.n_tok>128||b.n_seq!=1||b.n_out<0||b.n_out>1||b.draft_pass||b.n_spec||b.n_ahead<0||b.n_ahead>128||
       (b.phase!=RAD_PHASE_PREFILL&&b.phase!=RAD_PHASE_DECODE)||(b.phase==RAD_PHASE_DECODE&&(b.n_tok!=1||b.n_ahead)))return false;
    cp::Pass p;p.model_generation=model;p.source_generation=source;p.epoch=epoch;p.step=b.step;p.draft_pass=b.draft_pass;p.n_spec=b.n_spec;p.n_ahead=b.n_ahead;
    p.phase=b.phase==RAD_PHASE_PREFILL?cp::Phase::prefill:cp::Phase::decode;p.total_rows=b.n_tok;p.sequences=b.n_seq;p.n_out=b.n_out;*out=p;return true;
}
}
int glm_main_tp4_cache_program_profile(const Program& p)noexcept{
    if(!p.requires_glm_main_tp4_cache_fail_stop()||p.requires_permanent_model_failure()||p.ctx.world_size!=4||p.ctx.rank<0||p.ctx.rank>=4||p.ctx.max_tok<1||p.ctx.max_tok>128||p.ctx.max_seqs!=1||p.ctx.max_spec||p.ctx.is_draft||
       p.meta.n_layers!=45||p.meta.n_embd!=4096||p.meta.n_expert!=288||p.meta.n_expert_used!=8||!p.meta.arch_id||std::strcmp(p.meta.arch_id,"glm5next")||
       p.weights.size()>100000||p.ops.size()>50000||p.expert_route_maps.size()!=42||p.is_pass_driven()||p.encoder.modalities||p.audio_encoder.modalities||p.pooled_buf||!p.logits_buf||
       p.drafter.kind||!p.lora_sites.empty()||!p.lora_pools.empty()||!p.lora_t.empty()||p.lora_rows||p.lane1_w0>=0||p.lane1_w1>=0)return RAD_E_UNSUPPORTED;
    uint64_t layers=0;
    for(const auto& map:p.expert_route_maps){const auto& d=map.decl;
        if(d.layer<3||d.layer>44||(layers&(uint64_t(1)<<(d.layer-3)))||d.canonical_experts!=288||d.local_experts!=288||d.regular_experts!=288||d.expert_base||d.top_k!=8||d.max_rows!=p.ctx.max_tok||
           d.classes!=2||d.n_entries!=288||map.entries.size()!=288||!d.gu||d.gu>=p.ops.size())return RAD_E_UNSUPPORTED;
        layers|=uint64_t(1)<<(d.layer-3);std::array<uint8_t,288> seen{};
        for(size_t e=0;e<map.entries.size();++e){const auto& entry=map.entries[e];
            if(entry.slot!=int32_t(e)||entry.canonical<0||entry.canonical>=288||seen[size_t(entry.canonical)]++||entry.protected_expert||entry.class_index!=int32_t(e%size_t(d.classes)))return RAD_E_INVAL;
            for(size_t m=0;m<4;++m){const auto h=entry.members[m];if(!h||h>=p.weights.size())return RAD_E_INVAL;const auto& w=p.weights[h];
                if(w.decl.group.layer!=d.layer||w.decl.group.expert!=entry.slot||w.decl.group.slot!=int(m/2))return RAD_E_INVAL;}
        }
        for(const auto b:{d.counts,d.offsets,d.sorted,d.ids})if(!b||b>=p.buffers.size()||p.buffers[b].decl.dtype!=RAD_I32)return RAD_E_DTYPE;
    }
    if(layers!=((uint64_t(1)<<42)-1))return RAD_E_SHAPE;
    for(const auto& op:p.ops)for(const auto& band:op.bands)for(const auto& resolved:band.dom)
        if(resolved.row&&resolved.row->deferred_status)return RAD_E_UNSUPPORTED;
    return RAD_OK;
}
CachePrepare4Session::CachePrepare4Session(uint64_t model,uint64_t source):model_(model),source_(source),coordinator_(model,source){
    for(int r=0;r<4;++r){bindings_[size_t(r)].session_=this;bindings_[size_t(r)].rank_=r;}
}
CachePrepare4Session::~CachePrepare4Session(){if(source_admitted_)stop_admission();else coordinator_.close();}
int CachePrepare4ProtocolHarness::create(uint64_t model,uint64_t source,std::unique_ptr<CachePrepare4Session>* out)noexcept{
    if(!out||*out||!model||!source)return RAD_E_INVAL;
    try{auto result=std::unique_ptr<CachePrepare4Session>(new CachePrepare4Session(model,source));*out=std::move(result);return RAD_OK;}catch(const std::bad_alloc&){return RAD_E_NOMEM;}catch(...){return RAD_E_STATE;}
}
cp::Pass CachePrepare4Session::pass()const{std::lock_guard lock(pass_mutex_);return pass_;}
int CachePrepare4Session::begin(const RadBatch& batch){std::lock_guard lock(pass_mutex_);
    if(admission_closed())return RAD_E_STATE;
    if(active()||epoch_==std::numeric_limits<uint64_t>::max())return status(coordinator_.fail(-1,RAD_E_STATE,cp::Reason::bad_pass));
    cp::Pass proposed;if(!batch_pass(batch,model_,source_,epoch_+1,&proposed))return status(coordinator_.fail(-1,RAD_E_UNSUPPORTED,cp::Reason::bad_pass));
    const int result=coordinator_.begin(proposed);if(result<0)return status(result);
    pass_=proposed;epoch_=proposed.epoch;for(auto& binding:bindings_)binding.entered_=false;active_.store(true,std::memory_order_release);return RAD_OK;
}
int CachePrepare4Session::complete_frame(const RadBatch& batch){std::lock_guard lock(pass_mutex_);
    cp::Pass proposed;const auto state=coordinator_.observe();
    if(!active()||!batch_pass(batch,model_,source_,epoch_,&proposed)||proposed!=pass_||state.first.status<0||state.active||state.finished!=cp::all||state.ordinal!=cp::layers)
        return status(coordinator_.fail(-1,RAD_E_STATE,cp::Reason::bad_pass));
    active_.store(false,std::memory_order_release);return RAD_OK;
}
CachePrepare4RankBinding* CachePrepare4Session::binding(int rank)noexcept{return rank>=0&&rank<4?&bindings_[size_t(rank)]:nullptr;}
[[noreturn]]void CachePrepare4Session::fail_stop(int rank,int error)noexcept{(void)coordinator_.fail(rank,error<0?error:RAD_E_STATE);cp::process_fail_stop();}
void CachePrepare4Session::stop_if_active()noexcept{if(active())fail_stop(-1,RAD_E_STATE);}
void CachePrepare4Session::stop_admission()noexcept{std::lock_guard lock(pass_mutex_);
    if(active())fail_stop(-1,RAD_E_STATE);
    closed_.store(true,std::memory_order_release);coordinator_.close();
}
bool CachePrepare4RankBinding::runtime_admitted()const noexcept{return session_&&session_->runtime_admitted();}
bool CachePrepare4RankBinding::active()const noexcept{return session_&&session_->active();}
uint64_t CachePrepare4RankBinding::model_generation()const noexcept{return session_?session_->model_generation():0;}
uint64_t CachePrepare4RankBinding::source_generation()const noexcept{return session_?session_->source_generation():0;}
int CachePrepare4RankBinding::check_batch(const RadBatch& b)const noexcept{
    if(!session_||!session_->active())return RAD_E_STATE;
    try{const auto current=session_->pass();cp::Pass proposed;return batch_pass(b,model_generation(),source_generation(),current.epoch,&proposed)&&proposed==current?RAD_OK:RAD_E_UNSUPPORTED;}
    catch(...){return RAD_E_STATE;}
}
int CachePrepare4RankBinding::enter(int layer,int64_t first,int64_t rows,std::span<const int32_t> ids){
    if(!session_)return RAD_E_STATE;
    if(entered_||layer<3||layer>44)return status(session_->coordinator_.fail(rank_,RAD_E_STATE,cp::Reason::bad_key));
    const auto current=session_->pass();
    // Derive from fixed model layer, never a peer's possibly previous ordinal.
    cp::Key proposed{current,uint32_t(layer-3),layer,first,rows};
    const int result=session_->coordinator_.enter(rank_,proposed,ids);if(result<0)return status(result);key_=proposed;entered_=true;return RAD_OK;
}
int CachePrepare4RankBinding::ready(int local_status){if(!session_)return RAD_E_STATE;if(!entered_)return status(session_->coordinator_.fail(rank_,RAD_E_STATE,cp::Reason::bad_key));
    const int result=session_->coordinator_.ready(rank_,key_,local_status);if(result<0)return status(result);entered_=false;return RAD_OK;
}
int CachePrepare4RankBinding::finish(const RadBatch& b,int local_status){if(!session_)return RAD_E_STATE;
    if(local_status<0)return status(session_->coordinator_.fail(rank_,local_status,cp::Reason::early_exit));
    if(check_batch(b)<0)return status(session_->coordinator_.fail(rank_,RAD_E_STATE,cp::Reason::bad_pass));
    const int result=session_->coordinator_.finish(rank_,session_->pass(),local_status);if(result<0)return status(result);
    // Coordinator closes MODEL gates; Session stays frame-active through
    // sampler and final collect so a server stop cannot abort peers in flight.
    return RAD_OK;
}
[[noreturn]]void CachePrepare4RankBinding::fail_stop(int error)const noexcept{if(session_)session_->fail_stop(rank_,error);cp::process_fail_stop();}
} // namespace rad
