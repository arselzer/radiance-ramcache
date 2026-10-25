#include "ctx.h"
#include <algorithm>
#include <cstring>
#include <limits>
namespace rad {
namespace {
uint64_t bank_bytes(int64_t experts,int64_t rows,int64_t top){return uint64_t(2*experts+1+2*rows*top)*sizeof(int32_t);}
bool limits_ok(const ExpertPrepareLimits& l){return l.max_rows>0&&l.max_rows<=65536&&l.max_experts>0&&l.max_experts<=4096&&l.max_top_k>0&&l.max_top_k<=512&&l.max_maps>0&&l.max_maps<=256&&l.max_entries>0&&l.max_entries<=65536&&l.max_readback_bytes>0&&l.max_readback_bytes<=(16u<<20)&&l.max_metadata_bytes>0&&l.max_metadata_bytes<=(32u<<20);}
}
ExpertPrepareState::~ExpertPrepareState(){if(ready)rad_event_destroy(ready);if(allocation)rad_dev_free(allocation,RAD_MEM_HOST_MAPPED);}
int Ctx::expert_prepare_open(ExpertPrepareOwner* owner,const ExpertPrepareLimits& l){
    const auto* native=residency_?residency_->native_inventory():nullptr;
    if(native && (!owner || !owner->accepts_native_inventory(native)))return RAD_E_UNSUPPORTED;
    if(!owner)return RAD_OK;
    if(world_size_!=1||rank_!=0||record_passes_||checks_device_status()||!limits_ok(l)||program_->lane1_w0>=0||program_->expert_route_maps.empty()||program_->expert_route_maps.size()>l.max_maps)return RAD_E_UNSUPPORTED;
    try{
        const int accepted=owner->check_program(*program_);
        if (accepted < 0) return accepted;
        auto state=std::make_unique<ExpertPrepareState>();state->owner=owner;state->limits=l;size_t entries=0;uint64_t bytes=0;
        for(const auto& map:program_->expert_route_maps){const auto& d=map.decl;
            if(d.expert_base||d.canonical_experts!=d.local_experts||d.local_experts<=0||d.local_experts>l.max_experts||d.max_rows<=0||d.max_rows>l.max_rows||d.top_k<=0||d.top_k>l.max_top_k||map.entries.size()!=size_t(d.local_experts)||d.n_entries!=d.local_experts||d.regular_experts<0||d.regular_experts>d.local_experts||d.classes<1||d.classes>4||!d.gu||d.gu>=program_->ops.size())return RAD_E_UNSUPPORTED;
            entries+=map.entries.size();if(entries>l.max_entries)return RAD_E_SHAPE;
            bytes=std::max(bytes,bank_bytes(d.local_experts,d.max_rows,d.top_k));
            std::vector<uint8_t> unique(size_t(d.local_experts),0);
            for(size_t i=0;i<map.entries.size();++i){const auto& e=map.entries[i];
                if(e.slot!=int32_t(i)||e.canonical<0||e.canonical>=d.local_experts||unique[size_t(e.canonical)]||e.protected_expert!=int(i>=size_t(d.regular_experts))||e.class_index<-1||e.class_index>=d.classes||(e.class_index>=0&&e.class_index!=int32_t(i%size_t(d.classes)))||!e.members[0]||!e.members[2]||bool(e.members[1])!=bool(e.members[3])||(e.protected_expert&&(e.members[1]||e.members[3])))return RAD_E_INVAL;
                unique[size_t(e.canonical)]=1;
                for(const auto h:e.members)if(h){if(h>=program_->weights.size()||program_->weights[h].decl.group.layer!=d.layer)return RAD_E_INVAL;}
                for(size_t member=0;member<4;++member){const auto h=e.members[member];if(!h)continue;const auto& w=program_->weights[h];
                    if(!e.protected_expert&&(w.decl.group.expert!=e.slot||w.decl.group.slot!=int(member/2)))return RAD_E_INVAL;
                    if(e.protected_expert&&(w.decl.group.expert!=-1||w.stored_dtype!=RAD_BF16))return RAD_E_INVAL;
                }
            }
            for(const auto h:{d.counts,d.offsets,d.sorted,d.ids})if(!h||h>=program_->buffers.size()||program_->buffers[h].decl.dtype!=RAD_I32)return RAD_E_DTYPE;
        }
        const uint64_t metadata=uint64_t(entries)*sizeof(RadExpertRouteEntry)+uint64_t(l.max_rows)*l.max_top_k+program_->expert_route_maps.size()*sizeof(ExpertRouteMap);
        if(bytes>l.max_readback_bytes||metadata>l.max_metadata_bytes)return RAD_E_SHAPE;
        state->maps=program_->expert_route_maps;state->seen.resize(size_t(l.max_rows*l.max_top_k));state->bytes=bytes;
        state->allocation=rad_dev_alloc(int64_t(bytes),RAD_MEM_HOST_MAPPED);if(!state->allocation)return RAD_E_NOMEM;
        state->host=static_cast<int32_t*>(rad_dev_host_ptr(state->allocation));state->device=static_cast<int32_t*>(rad_dev_device_ptr(state->allocation));if(!state->host||!state->device)return RAD_E_STATE;
        const int status=rad_event_create(&state->ready);if(status<0)return status;
        if(native){const int bound=owner->bind_native_context(native,this,stream_);if(bound<0)return bound;}
        expert_prepare_=std::move(state);return RAD_OK;
    }catch(const std::bad_alloc&){return RAD_E_NOMEM;}catch(...){return RAD_E_STATE;}
}
void Ctx::expert_prepare_close(bool drained){
    if(!expert_prepare_)return;
    if(!drained){(void)expert_prepare_.release();return;} // quarantine, never free possibly live DMA
    expert_prepare_.reset();
}
int Ctx::expert_prepare_fail(int status,const char* why){
    if(const int first=terminal_status();first<0)return first;
    // A later Native refusal must not replace an earlier synchronous pass
    // failure in the owner's sticky state, including default Legacy programs.
    const int first_step=step_status_<0?step_status_:status;
    if(expert_prepare_&&expert_prepare_->fatal>=0)expert_prepare_->fatal=first_step;
    if(step_status_<0)return remember_model_failure(step_status_,why);
    try{return abort_step_msg(0,status,why);}
    catch(...){step_status_=status;step_fail_.clear();return remember_model_failure(status,why);}
}
int Ctx::expert_prepare_admit(const RadBatch* batch){
    if(const int first=terminal_status();first<0)return first;
    if(!expert_prepare_)return RAD_OK;
    if(expert_prepare_->fatal<0)return remember_model_failure(expert_prepare_->fatal,"current expert prepare is already fatal");
    if(!batch||world_size_!=1||lane_!=0||batch->n_spec!=0||batch->draft_pass!=0||batch->n_ahead!=0||stager_||xp_||tape_on_||rec_)return expert_prepare_fail(RAD_E_UNSUPPORTED,"current expert prepare requires TP1,lane0,eager,no speculation/ahead/stager/XP");
    return RAD_OK;
}
void Ctx::expert_prepare_begin(){
    // run_step has completed pure admission and deferred begin; no callbacks,
    // allocation, fallible work, or new refusal may open an unretired frame here.
    if(expert_prepare_)expert_prepare_->active_pass=true;
}
int Ctx::expert_prepare_end(){
    if(!expert_prepare_||!expert_prepare_->active_pass)return RAD_OK;
    auto& state=*expert_prepare_;state.active_pass=false;int result=RAD_OK;
    try{result=state.owner->end_pass(stream_,step_status_);}catch(const std::bad_alloc&){result=RAD_E_NOMEM;}catch(...){result=RAD_E_STATE;}
    if(step_status_<0&&state.fatal>=0)state.fatal=step_status_;
    if(result<0)return expert_prepare_fail(result,"current expert prepare owner end_pass failed");
    return RAD_OK;
}
int Ctx::expert_prepare(const RadExpertPrepare* request){
    if(const int first=terminal_status();first<0)return first;
    if(!expert_prepare_)return RAD_OK;
    if(step_status_<0)return remember_model_failure(step_status_,"current expert prepare follows a failed step");
    if(!request||!batch_||!expert_prepare_->active_pass)return expert_prepare_fail(RAD_E_STATE,"current expert prepare outside active pass");
    auto& state=*expert_prepare_;
    auto fail=[&](int status,const char* why){return expert_prepare_fail(status,why);};
    try{
        if(state.fatal<0)return remember_model_failure(state.fatal,"current expert prepare is already fatal");
        if(lane_!=0||stager_||xp_||tape_on_||rec_)return fail(RAD_E_UNSUPPORTED,"current expert prepare profile changed within pass");
        const auto found=std::find_if(state.maps.begin(),state.maps.end(),[&](const auto& m){return m.decl.layer==request->layer;});
        if(found==state.maps.end())return fail(RAD_E_INVAL,"undeclared expert route layer");
        const auto& map=*found;const auto& d=map.decl;const auto& r=*request;
        if(r.total_rows<=0||r.total_rows>batch_->n_tok||r.total_rows>program_->ctx.max_tok||r.first_row<0||r.first_row>INT64_MAX/d.top_k||r.rows<=0||r.rows>d.max_rows||r.first_row>r.total_rows||r.rows>r.total_rows-r.first_row||state.generation==UINT64_MAX)return fail(RAD_E_SHAPE,"current expert route slice bounds/generation");
        const int64_t routes=r.rows*d.top_k;
        auto bind=[&](const RadOperand& o,rad_buf expected,int64_t rows,int64_t cols,int64_t offset,RadTensor* out,bool counts)->int{
            if(counts&&o.kind==RAD_OPK_RAW){
                if(o.raw!=route_counts(d.layer,d.local_experts)||!o.raw||o.dtype!=RAD_I32||o.offset||o.cols||o.rows!=rows||o.handle)return RAD_E_INVAL;
                *out={};out->data=o.raw;out->dtype=RAD_I32;out->rank=1;out->shape[0]=rows;out->stride[0]=1;return RAD_OK;
            }
            if(o.kind!=RAD_OPK_BUF||o.handle!=expected||!expected||expected>=buf_tmpl_.size()||o.raw||o.offset!=offset||o.rows!=rows||o.cols)return RAD_E_INVAL;
            RadTensor t=buf_tmpl_[expected];const int rank=cols?2:1;
            if(t.dtype!=RAD_I32||t.rank!=uint32_t(rank)||!t.data||offset<0||t.stride[rank-1]!=1||(cols&&(t.shape[1]!=cols||t.stride[0]!=cols)))return RAD_E_SHAPE;
            const int64_t width=cols?cols:1;
            if(offset%width||rows>t.shape[0]||offset/width>t.shape[0]-rows||uint64_t(offset)>uint64_t(INT64_MAX)/4||uint64_t(rows)>uint64_t(INT64_MAX)/uint64_t(width)/4||uintptr_t(t.data)>UINTPTR_MAX-uint64_t(offset)*4-uint64_t(rows)*uint64_t(width)*4)return RAD_E_SHAPE;
            const auto& buffer=program_->buffers[expected];if(buffer.first_def>=0&&(int64_t(d.gu)<buffer.first_def||int64_t(d.gu)>buffer.last_use))return RAD_E_STATE;
            t.data=static_cast<char*>(t.data)+offset*4;t.shape[0]=rows;*out=t;return RAD_OK;
        };
        RadTensor tensors[4]{};
        int rc=bind(r.counts,d.counts,d.local_experts,0,0,&tensors[0],true);if(rc<0)return fail(rc,"counts operand is not declared/Core-owned current row");
        rc=bind(r.offsets,d.offsets,d.local_experts+1,0,0,&tensors[1],false);if(rc<0)return fail(rc,"invalid current expert offsets binding");
        rc=bind(r.sorted,d.sorted,routes,0,0,&tensors[2],false);if(rc<0)return fail(rc,"invalid pass-relative sorted binding");
        rc=bind(r.ids,d.ids,r.rows,d.top_k,r.first_row*d.top_k,&tensors[3],false);if(rc<0)return fail(rc,"invalid step-relative IDs slice binding");
        const int64_t sizes[]{d.local_experts,d.local_experts+1,routes,routes};int64_t at=0;
        for(int i=0;i<4;++i){rc=rad_memcpy_async(state.device+at,tensors[i].data,sizes[i]*4,cur_stream());if(rc<0)return fail(rc,"current route producer copy failed");at+=sizes[i];++state.copies;state.copied_bytes+=uint64_t(sizes[i])*4;}
        rc=rad_event_record(state.ready,cur_stream());if(rc<0)return fail(rc,"current route producer event record failed");
        rc=rad_event_sync(state.ready);if(rc<0)return fail(rc,"current route producer event did not complete");
        // No payload access occurs before successful same-producer-stream completion.
        const auto* counts=state.host;const auto* offsets=counts+d.local_experts;const auto* sorted=offsets+d.local_experts+1;const auto* ids=sorted+routes;
        std::fill(state.seen.begin(),state.seen.begin()+routes,uint8_t(0));
        int64_t live=0;for(int64_t flat=0;flat<routes;++flat){if(ids[flat]>=d.canonical_experts)return fail(RAD_E_INVAL,"selected current expert ID outside full local domain");if(ids[flat]>=0)++live;}
        int64_t sum=0;
        for(int slot=0;slot<d.local_experts;++slot){if(counts[slot]<0||counts[slot]>routes-sum||offsets[slot]!=sum)return fail(RAD_E_INVAL,"current expert count/prefix mismatch");const int64_t end=sum+counts[slot];
            for(int64_t i=sum;i<end;++i){const int flat=sorted[i];if(flat<0||flat>=routes||state.seen[size_t(flat)]||ids[flat]!=map.entries[size_t(slot)].canonical)return fail(RAD_E_INVAL,"current sorted coverage/canonical expert mapping mismatch");state.seen[size_t(flat)]=1;}sum=end;
        }
        if(sum!=live||offsets[d.local_experts]!=live)return fail(RAD_E_INVAL,"current route live count mismatch");
        for(int64_t i=live;i<routes;++i)if(sorted[i]!=-1)return fail(RAD_E_INVAL,"current dropped-padding sorted tail is not -1");
        for(int64_t i=0;i<routes;++i)if(bool(state.seen[size_t(i)])!=(ids[i]>=0))return fail(RAD_E_INVAL,"current live slot omitted/negative padding included");
        ++state.generation;bool changed=false;
        ExpertPrepareView view{program_,this,d,map.entries,batch_->step,r.total_rows,r.first_row,r.rows,state.generation,cur_stream(),tensors[0],tensors[1],tensors[2],tensors[3],{counts,size_t(d.local_experts)},{offsets,size_t(d.local_experts+1)},{sorted,size_t(routes)},{ids,size_t(routes)}};
        ++state.callbacks;rc=state.owner->prepare(view,&changed);if(rc<0)return fail(rc,"current expert prepare owner refused before GU");
        if(changed){sync_residency();}
        return RAD_OK;
    }catch(const std::bad_alloc&){return fail(RAD_E_NOMEM,"current expert prepare allocation failure");}catch(...){return fail(RAD_E_STATE,"current expert prepare owner exception");}
}
}
extern "C" int rad_expert_prepare(RadCtx* c,const RadExpertPrepare* r){return c?static_cast<rad::Ctx*>(c)->expert_prepare(r):RAD_E_INVAL;}
