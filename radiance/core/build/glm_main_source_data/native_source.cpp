#include "../glm_main_profile_limits.h"
#include "native_source.h"
#include <algorithm>
#include <array>
#include <limits>
#include <optional>

namespace rad::glm_main_data {
namespace {
thread_local bool in_main_facade=false;
struct Active {
    Active(){in_main_facade=true;}
    ~Active(){in_main_facade=false;}
};
constexpr size_t records=42*288,max_request=glm_main_max_union,max_holders=64;
constexpr uint64_t record_bytes=3342336;
uint64_t sum(uint64_t a,uint64_t b){if(b>UINT64_MAX-a)throw std::bad_alloc();return a+b;}
uint64_t product(uint64_t a,uint64_t b){if(a&&b>UINT64_MAX/a)throw std::bad_alloc();return a*b;}
void explanation(std::string* out,const char* text)noexcept{if(out)try{*out=text;}catch(...){}}
}
int native_status(sc::Error e)noexcept{
    switch(e){
        case sc::Error::ok:return RAD_OK;
        case sc::Error::budget:return RAD_E_SCRATCH;
        case sc::Error::no_memory:return RAD_E_NOMEM;
        case sc::Error::io:case sc::Error::checksum:return RAD_E_IO;
        case sc::Error::identity:case sc::Error::invalid:return RAD_E_INVAL;
        default:return RAD_E_STATE;
    }
}
struct GlmMainNativeSource::Quota {
    mutable std::mutex mutex;
    AdapterUsage usage;
    // Preserve the SAME rank token through Holder teardown, including the small
    // interval after its provider Batch is destroyed but before credit return.
    std::optional<sc::Accounting> reservation;
};
struct GlmMainNativeSource::Holder {
    std::shared_ptr<Quota> quota;
    sc::Batch batch;
    explicit Holder(std::shared_ptr<Quota> q):quota(std::move(q)){}
    ~Holder(){
        // Return provider vector/payload credits before the facade holder credit.
        batch=sc::Batch{};
        std::lock_guard lock(quota->mutex);
        --quota->usage.live_holders;
    }
};
GlmMainNativeSource::GlmMainNativeSource()=default;
GlmMainNativeSource::~GlmMainNativeSource(){(void)close();}
int GlmMainNativeSource::open(std::shared_ptr<sc::ProcessBudget> process,
    std::shared_ptr<const VerifiedRankMain> authority,const sc::Limits& limits,
    std::shared_ptr<GlmMainNativeSource>* out,std::string* why)noexcept{
    if(in_main_facade)return RAD_E_STATE;
    Active active;
    if(!process||!authority||!out||*out||!authority->program()||!authority->metadata_plan()){
        explanation(why,"complete-main facade authority/output");return RAD_E_INVAL;
    }
    const auto& p=*authority->program();
    if(p.ctx.world_size!=4||p.ctx.rank<0||p.ctx.rank>=4||!glm_main_rows_supported(p.ctx.max_tok)||p.ctx.max_seqs!=1||p.ctx.max_spec||
       authority->rank()!=uint32_t(p.ctx.rank)||authority->descriptors().size()!=records||
       authority->owned().size()!=records||authority->units().size()!=records||
       !limits.max_batch||limits.max_batch>size_t(p.ctx.max_tok)*8||limits.max_record_bytes<record_bytes){
        explanation(why,"closed TP4 main/max-token1/max-sequence1/no-spec profile");return RAD_E_INVAL;
    }
    try{
        // Source-owned storage plus conservative shared-control allocation
        // allowances. Caller-retained NativeRecordBatch vector capacity remains
        // caller storage (the public ABI has no retained-capacity credit hook).
        // Provider Batch vectors have their own explicit retained slot credits.
        const uint64_t holder=sizeof(Holder)+128;
        const uint64_t metadata=sum(sizeof(GlmMainNativeSource)+128,
            sum(sizeof(Quota)+128,sum(product(records,sizeof(sc::BoundRecord)),
                product(max_holders,holder))));
        // Only the fixed Source shell exists before provider/token admission.
        // All O(records) facade allocations occur AFTER that same token owns
        // their full reservation, and no source is published before completion.
        auto result=std::shared_ptr<GlmMainNativeSource>(new GlmMainNativeSource);
        result->process_=std::move(process);result->authority_=std::move(authority);
        const auto opened=sc::RankNativeStore::open_verified_main(result->process_,result->authority_,limits,
            metadata,&result->store_,why);
        if(opened!=sc::Error::ok)return native_status(opened);
        result->quota_=std::make_shared<Quota>();result->quota_->usage.metadata_reservation=metadata;
        result->quota_->reservation=result->store_->accounting();
        result->request_limit_=limits.max_batch;result->handles_.resize(records);
        const auto bound=result->store_->bind_main_handles(result->handles_);
        if(bound!=sc::Error::ok)return native_status(bound);
        // Quota/control blocks/handle storage all exist before the final
        // nonallocating metadata/stamp audit and caller publication.
        if(result->authority_->validate_program()!=Error::ok)return RAD_E_INVAL;
        const auto audited=result->store_->audit_all();
        if(audited!=sc::Error::ok)return native_status(audited);
        *out=std::move(result);return RAD_OK;
    }catch(const std::bad_alloc&){explanation(why,"complete-main facade allocation refused");return RAD_E_NOMEM;}
    catch(...){explanation(why,"unexpected complete-main facade failure");return RAD_E_STATE;}
}
std::span<const rad::NativeRecordDescriptor> GlmMainNativeSource::catalog()const noexcept{
    return authority_?authority_->descriptors():std::span<const rad::NativeRecordDescriptor>{};
}
int GlmMainNativeSource::bootstrap(int32_t,std::span<std::byte>)noexcept{return RAD_E_UNSUPPORTED;}
int GlmMainNativeSource::finish_bootstrap()noexcept{
    if(in_main_facade)return RAD_E_STATE;
    Active active;std::lock_guard lock(operation_);
    if(closed_||bootstrap_done_||!store_||!authority_)return RAD_E_STATE;
    const auto checked=authority_->validate_program();
    if(checked!=Error::ok){closed_=true;(void)store_->close();return RAD_E_INVAL;}
    const auto audited=store_->audit_all();
    if(audited!=sc::Error::ok){closed_=true;(void)store_->close();return native_status(audited);}
    const auto trimmed=store_->trim();
    if(trimmed!=sc::Error::ok){closed_=true;(void)store_->close();return native_status(trimmed);}
    bootstrap_done_=true;return RAD_OK;
}
int GlmMainNativeSource::acquire(std::span<const uint32_t> request,rad::NativeRecordBatch* out){
    if(in_main_facade)return RAD_E_STATE;
    Active active;std::lock_guard lock(operation_);
    if(closed_||!bootstrap_done_||!store_)return RAD_E_STATE;
    if(!out||!request.data()||request.empty()||request.size()>request_limit_)return RAD_E_INVAL;
    std::array<uint32_t,max_request> ids{};
    std::array<sc::BoundRecord,max_request> selected{};
    size_t count=0;
    // Validate the ENTIRE ID list before credit, I/O or any ready-cell change.
    for(const auto id:request){
        if(id>=handles_.size())return RAD_E_INVAL;
        if(std::find(ids.begin(),ids.begin()+count,id)!=ids.begin()+count)continue;
        ids[count]=id;selected[count]=handles_[id];++count;
    }
    bool reserved=false;
    try{
        {
            std::lock_guard quota_lock(quota_->mutex);
            if(quota_->usage.live_holders>=max_holders)return RAD_E_SCRATCH;
            ++quota_->usage.live_holders;reserved=true;
            quota_->usage.peak_holders=std::max(quota_->usage.peak_holders,quota_->usage.live_holders);
        }
        // Every fallible facade allocation precedes Store.acquire and its ready
        // commit. No allocation follows a successful underlying transaction.
        auto holder=std::make_shared<Holder>(quota_);reserved=false;
        rad::NativeRecordBatch result;result.leases.reserve(count);
        const auto acquired=store_->acquire({selected.data(),count},&holder->batch);
        if(acquired!=sc::Error::ok)return native_status(acquired);
        // A provider invariant violation may leave a verified cache fill ready,
        // but must never permit out-of-range access or publish a partial caller
        // batch. It is not a recoverable facade transaction/OOM path.
        if(holder->batch.leases.size()!=count)return RAD_E_STATE;
        // Immutable provider contract: unique first-occurrence order is retained.
        // Reserved inserts/shared_ptr copies/swap below are nonallocating.
        for(size_t i=0;i<count;++i)result.leases.push_back({ids[i],holder->batch.leases[i].bytes(),holder});
        out->leases.swap(result.leases);return RAD_OK;
    }catch(const std::bad_alloc&){
        if(reserved){std::lock_guard quota_lock(quota_->mutex);--quota_->usage.live_holders;}
        return RAD_E_NOMEM;
    }catch(...){
        if(reserved){std::lock_guard quota_lock(quota_->mutex);--quota_->usage.live_holders;}
        return RAD_E_STATE;
    }
}
int GlmMainNativeSource::close()noexcept{
    if(in_main_facade)return RAD_E_STATE;
    Active active;std::lock_guard lock(operation_);
    if(closed_)return RAD_OK;
    closed_=true;return store_?native_status(store_->close()):RAD_OK;
}
sc::Stats GlmMainNativeSource::stats()const noexcept{return store_->stats();}
sc::ProcessUsage GlmMainNativeSource::process_usage()const noexcept{return process_->usage();}
sc::Accounting GlmMainNativeSource::accounting()const noexcept{return store_->accounting();}
AdapterUsage GlmMainNativeSource::adapter_usage()const noexcept{
    std::lock_guard lock(quota_->mutex);return quota_->usage;
}
} // namespace rad::glm_main_data
