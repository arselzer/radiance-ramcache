#include "tiny_component_quotes.h"
#include "../rad_core.h"
#include "../sched/advance.h"
#include <algorithm>
#include <limits>
namespace rad::tiny_quote {
bool add(uint64_t& value,uint64_t n) noexcept {if(n>UINT64_MAX-value)return false;value+=n;return true;}
bool mul(uint64_t a,uint64_t b,uint64_t* out) noexcept {if(!out||(a&&b>UINT64_MAX/a))return false;*out=a*b;return true;}
namespace {
bool request(uint64_t& total,uint64_t& rounded,uint64_t& calls,uint64_t bytes,uint64_t n) noexcept {
    if(!bytes||!n||bytes>UINT64_MAX-(page-1))return false;
    uint64_t a=0,b=0;
    return mul(bytes,n,&a)&&mul((bytes+page-1)/page*page,n,&b)&&add(total,a)&&add(rounded,b)&&add(calls,n);
}
}
bool device(TinyStorageQuote& q,uint64_t b,uint64_t n) noexcept{return request(q.device_bytes,q.device_page_bytes,q.device_requests,b,n);}
bool pinned(TinyStorageQuote& q,uint64_t b,uint64_t n) noexcept{return request(q.pinned_bytes,q.pinned_page_bytes,q.pinned_requests,b,n);}
bool mapped(TinyStorageQuote& q,uint64_t b,uint64_t n) noexcept{return request(q.mapped_bytes,q.mapped_page_bytes,q.mapped_requests,b,n);}
bool heap(TinyStorageQuote& q,uint64_t count,uint64_t element) noexcept {uint64_t n=0;return mul(count,element,&n)&&add(q.heap_payload_bytes,n);}
bool finish(TinyStorageQuote& q) noexcept {
    uint64_t meta=0;for(auto n:{q.heap_payload_bytes,q.embedded_object_bytes,q.heap_growth_allowance,q.control_allowance})if(!add(meta,n))return false;
    if(meta>control_cap)return false;
    uint64_t requested=meta,rounded=meta;
    for(auto n:{q.device_bytes,q.pinned_bytes,q.mapped_bytes})if(!add(requested,n))return false;
    for(auto n:{q.device_page_bytes,q.pinned_page_bytes,q.mapped_page_bytes})if(!add(rounded,n))return false;
    q.requested_total=requested;q.page_and_heap_total=rounded;return true;
}
int program_profile(const Program& p) noexcept {
    if(!p.requires_permanent_model_failure()||p.ctx.world_size!=1||p.ctx.rank||p.ctx.max_tok!=5||p.ctx.max_seqs!=1||p.ctx.max_ctx!=5||p.ctx.max_spec||p.ctx.is_draft||
       p.meta.n_layers!=8||p.meta.n_embd!=128||p.meta.n_expert!=4||p.meta.n_vocab!=320||
       p.weights.empty()||p.weights.size()>max_weights||p.buffers.empty()||p.buffers.size()>max_buffers||p.ops.empty()||p.ops.size()>max_ops||
       p.kv_groups.size()>1||!p.lora_sites.empty()||!p.lora_pools.empty()||!p.lora_t.empty()||p.lora_rows||p.drafter.kind||p.is_pass_driven()||p.encoder.modalities||
       p.lane1_w0>=0||p.lane1_w1>=0||p.scratch_regions!=1||p.arena_stage_bytes||p.arena_bytes<0||p.arena_bytes>int64_t(arena_cap)||p.scratch_bytes<0||p.scratch_bytes>int64_t(arena_cap))return RAD_E_UNSUPPORTED;
    return RAD_OK;
}
}
namespace rad {
int tiny_step_state_quote(std::span<const int32_t> eog,TinyStepQuote* out) noexcept {
    if(!out||eog.size()>64)return RAD_E_INVAL;
    TinyStepQuote q;q.sizeof_step_slot=sizeof(StepSlot);q.slots=1;q.input_eog=eog.size();
    for(size_t i=0;i<eog.size();++i){if(eog[i]<0||eog[i]>=320)return RAD_E_INVAL;bool earlier=false;for(size_t j=0;j<i;++j)earlier|=eog[j]==eog[i];if(!earlier)++q.unique_eog;}
    auto& s=q.storage;
    if(!tiny_quote::device(s,sizeof(StepSlot))||!tiny_quote::device(s,std::max<uint64_t>(q.unique_eog,1)*sizeof(int32_t))||!tiny_quote::pinned(s,sizeof(StepSlot))||!tiny_quote::heap(s,eog.size(),sizeof(int32_t)))return RAD_E_SHAPE;
    s.event_count=1;s.control_allowance=256;
    if(!tiny_quote::finish(s))return RAD_E_SCRATCH;
    *out=q;return RAD_OK;
}
}
