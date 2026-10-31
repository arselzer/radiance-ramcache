/* Optional readback ownership and actual Plan binding; no executor or scheduling choice. */
#include "stager.h"
#include <cstring>

namespace rad {
int FileStager::p2_init(const Program&prog){
    namespace a=ds41_phase2::audit;
    if(!a::requested())return RAD_OK;
    auto refuse=[&](const char*why){RAD_ERR("DSP2AUDIT refused rank=%d reason=%s",plan_?plan_->rank:-1,why);return RAD_E_INVAL;};
    if(!a::prefix_valid(a::prefix()))return refuse("absolute_prefix_required");
    if(!plan_||plan_->world_size!=4||plan_->rank<0||plan_->rank>=4||plan_->dynamic||prog.meta.n_layers!=40||
       !prog.meta.arch_id||std::strcmp(prog.meta.arch_id,"deepseek41")||prog.meta.n_embd!=5120||
       prog.meta.n_expert!=384||prog.meta.n_expert_used!=6||
       !p1_on()||lcp_!=4096||ep_rows_!=4096||!ep_counts_||!counts_host_||counts_cap_<96||
       hx_early_||prog.two_batch||layers_.size()!=40)return refuse("requires_static_native_4K_P1_no_early_issue_or_TBO");
    // A complete immutable local96 Plan map per layer; no guessing from file-only lists.
    std::array<std::array<uint8_t,96>,40>seen{};
    for(const MoveUnit&u:plan_->units)if(u.access==RAD_ACCESS_CONDITIONAL&&u.layer>=0&&u.layer<40&&u.expert>=0&&u.expert<96){
        if(seen[u.layer][u.expert]++||u.bytes<=0)return refuse("ambiguous_native_plan_units");
    }
    for(int l=0;l<40;++l)for(int e=0;e<96;++e)if(seen[l][e]!=1)return refuse("incomplete_native_plan_units");
    const std::string path=std::string(a::prefix())+".rank"+std::to_string(plan_->rank)+".jsonl";
    // Exclusive create protects an earlier receipt. A failed/partial run is never overwritten.
    p2_file_=std::fopen(path.c_str(),"wx");if(!p2_file_)return refuse("output_open_failed_or_exists");
    constexpr int64_t words=2*ds41_phase2::routes::kMaxSlots+97;
    p2_readback_=(int32_t*)rad_dev_alloc(words*4,RAD_MEM_HOST_PINNED);
    if(!p2_readback_)return refuse("bounded_readback_allocation_failed");
    p2_program_=&prog;
    p2_audit_.reset(new a::Trace(p2_file_,plan_->rank,(uint64_t)(uintptr_t)&prog,(uint64_t)(uintptr_t)plan_));
    p1_->set_copy_observer([](void*ud,int e,int32_t unit,int rows,int slot,p1::Kind kind,const p1::Source&src){
        auto*f=static_cast<FileStager*>(ud);
        a::Copy c;c.expert=e;c.unit=unit;c.rows=rows;c.slot=slot;c.kind=int(kind);
        c.gu_bytes=src.gu_bytes;c.dn_bytes=src.dn_bytes;c.packed=src.packed;
        f->p2_audit_->copy(c);
    },this);
    RAD_INFO("DSP2AUDIT enabled rank=%d file=%s program=%llu plan=%llu readback_bytes=%lld metadata_bytes=%zu "
             "max_output_bytes=%llu max_frames=%d max_spans=%d reuse=completed_counts_and_matching_order "
             "extra=ids_and_unavailable_order performance_eligible=0 executor=0",
        plan_->rank,path.c_str(),(unsigned long long)(uintptr_t)&prog,(unsigned long long)(uintptr_t)plan_,
        (long long)words*4,sizeof(a::Trace),(unsigned long long)a::kMaxBytes,a::kMaxFrames,a::kMaxSpans);
    return RAD_OK;
}
void FileStager::p2_queue(const EpRouteCounts&v,bool order_queued,int32_t li,RadStream s){
    namespace r=ds41_phase2::routes;
    p2_queued_=false;p2_reused_order_=false;p2_extra_bytes_=0;
    if(!p2_audit_||!p2_audit_->collecting())return;
    if(v.program!=p2_program_||v.stream!=s||!v.audit_ids||!v.audit_sorted||!v.audit_offsets||!mover_->source_epochs_current()||
       v.rows!=layer_rows(li)||v.rows<1||v.rows>4096){p2_audit_->incomplete("producer_operands_unavailable");return;}
    const int64_t slots=v.rows*6;
    auto copy=[&](void*dst,const void*src,int64_t n){
        if(rad_memcpy_async(dst,src,n,s)<0){p2_audit_->incomplete("extra_readback_enqueue_failed");return false;}
        p2_extra_bytes_+=(uint64_t)n;return true;
    };
    if(!copy(p2_readback_,v.audit_ids,slots*4))return;
    p2_reused_order_=order_queued&&hx_idx_&&(size_t)li<hx_ops_.size()&&
        hx_ops_[(size_t)li].sorted==v.audit_sorted&&hx_ops_[(size_t)li].off==v.audit_offsets;
    if(!p2_reused_order_){
        if(!copy(p2_readback_+r::kMaxSlots,v.audit_sorted,slots*4)||
           !copy(p2_readback_+2*r::kMaxSlots,v.audit_offsets,97*4))return;
    }
    p2_queued_=true;
}
void FileStager::p2_freeze(const EpRouteCounts&v,bool p1,int32_t li,uint64_t counts_wait_ns){
    namespace a=ds41_phase2::audit;namespace r=ds41_phase2::routes;
    if(!p2_audit_||!p2_audit_->collecting()||!p2_queued_)return;
    a::UnitMap map;
    for(size_t u=0;u<plan_->units.size();++u){
        const MoveUnit&mu=plan_->units[u];if(mu.access!=RAD_ACCESS_CONDITIONAL||mu.layer!=v.layer||mu.expert<0||mu.expert>=96)continue;
        a::Unit&to=map[(size_t)mu.expert];
        if(to.unit>=0){p2_audit_->incomplete("plan_unit_alias");return;}
        to.unit=(int32_t)u;to.tier=int(mu.tier);to.native_bytes=mu.bytes;
        const bool resident=mu.tier==Tier::VRAM||mover_->ring_resident((int32_t)u)||mover_->file_unit_in_slab((int32_t)u);
        to.residence=resident?r::Residence::Resident:r::Residence::Nonresident;
    }
    const int64_t slots=v.rows*6;
    const int32_t*sorted=p2_reused_order_?hx_idx_:p2_readback_+r::kMaxSlots;
    const int32_t*offsets=p2_reused_order_?hx_idx_+hx_sorted_max():p2_readback_+2*r::kMaxSlots;
    const r::View view{p2_readback_,(size_t)slots,sorted,(size_t)slots,counts_host_,96,offsets,97};
    p2_audit_->frame(v.generation,v.producer_serial,(int)v.layer,(int)layer_r0(li),view,map,p1,p2_extra_bytes_,p2_reused_order_,counts_wait_ns);
    p2_queued_=false;
}
} // namespace rad
