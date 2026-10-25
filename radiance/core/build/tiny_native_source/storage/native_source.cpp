#include "native_source.h"
#include <algorithm>
#include <cstring>
namespace glm53::engine_core_p3 {
namespace {thread_local bool in_facade=false;struct Active {Active(){in_facade=true;}~Active(){in_facade=false;}};}
int status(sc::Error e)noexcept{switch(e){case sc::Error::ok:return RAD_OK;case sc::Error::budget:return RAD_E_SCRATCH;case sc::Error::no_memory:return RAD_E_NOMEM;case sc::Error::io:case sc::Error::checksum:return RAD_E_IO;case sc::Error::identity:case sc::Error::invalid:return RAD_E_INVAL;default:return RAD_E_STATE;}}
struct ShardedNativeSource::Quota {mutable std::mutex mutex;AdapterUsage usage;size_t cap=16;};
struct ShardedNativeSource::Holder {
    std::shared_ptr<Quota> quota;sc::Batch batch;
    explicit Holder(std::shared_ptr<Quota> q):quota(std::move(q)){}
    ~Holder(){batch=sc::Batch{};std::lock_guard lock(quota->mutex);--quota->usage.live_holders;}
};
ShardedNativeSource::ShardedNativeSource():quota_(std::make_shared<Quota>()){}
ShardedNativeSource::~ShardedNativeSource(){(void)close();}
int ShardedNativeSource::open(const rad::Program& program,const rad::Plan& plan,const std::string& root,
                             std::span<const sc::ShardInput> inputs,uint64_t cold,uint64_t hot,
                             std::shared_ptr<ShardedNativeSource>* out,std::string* why){
    if(!out||*out||program.ctx.world_size!=1||program.ctx.rank||inputs.size()!=3||
       plan.units.size()>128||program.weights.size()>512||!cold||cold>98304||hot>65536)return RAD_E_INVAL;
    try{
        struct Typed {int32_t unit;rad::NativeRecordDescriptor descriptor;};
        std::vector<Typed> typed;typed.reserve(64);
        uint64_t generation=14695981039346656037ull;
        auto byte=[&](uint8_t b){generation^=b;generation*=1099511628211ull;};
        for(const auto& input:inputs){for(const unsigned char c:input.name)byte(c);byte(0);
            for(unsigned s=0;s<64;s+=8)byte(uint8_t(input.manifest_checksum>>s));}
        if(!generation)generation=1;
        // Full typed capture is BEFORE verified Store.open, never derived after verification.
        for(size_t u=0;u<plan.units.size();++u){const auto& unit=plan.units[u];if(unit.expert<0)continue;
            if(typed.size()>=64||unit.weights.size()!=4||unit.offset.size()!=4||unit.slot_bytes!=32768||unit.expert>=4||unit.layer<0||unit.layer>=8)return RAD_E_SHAPE;
            rad::NativeRecordDescriptor d;d.source_generation=generation;d.world=1;d.rank=0;
            d.layer=unit.layer;d.expert=unit.expert;d.stride=unit.slot_bytes;
            for(size_t m=0;m<4;++m){const auto w=unit.weights[m];if(!w||w>=program.weights.size())return RAD_E_INVAL;
                const auto& info=program.weights[w];const auto main=rad_enc_affine(RAD_FP4E2M1,RAD_E8M0,1,32);
                if(info.stored_rank!=2||!info.identity||!info.enc_known||!rad_enc_equal(&info.enc,&main)||
                   info.n_sel!=1||info.sel[0]!=int(m%2))return RAD_E_SHAPE;
                d.offsets[m]=unit.offset[m];d.bytes[m]=info.stored_bytes;d.dtype[m]=info.stored_dtype;
                d.shape[m]={info.stored_shape[0],info.stored_shape[1]};
                if(d.offsets[m]<0||d.bytes[m]<=0||d.offsets[m]>d.stride||d.bytes[m]>d.stride-d.offsets[m]||d.shape[m][0]<=0||d.shape[m][1]<=0)return RAD_E_SHAPE;}
            typed.push_back({int32_t(u),d});
        }
        if(typed.size()!=32)return RAD_E_SHAPE;
        auto result=std::shared_ptr<ShardedNativeSource>(new ShardedNativeSource);
        sc::ProcessLimits process_limits;process_limits.max_metadata_bytes=24u<<20;
        process_limits.max_cache_reservations=hot+cold;
        const auto created=sc::ProcessBudget::create(process_limits,&result->process_);
        if(created!=sc::Error::ok)return status(created);
        sc::Limits limits;limits.hot_bytes=hot;limits.cold_bytes=cold;limits.max_records=64;
        limits.max_live_records=32;limits.max_batch=3;limits.max_live_leases=16;
        limits.max_metadata_bytes=24u<<20;limits.max_shards=3;limits.max_segments=32;
        const auto opened=sc::RankNativeStore::open(result->process_,program,plan,root,generation,inputs,limits,&result->store_,why);
        if(opened!=sc::Error::ok)return status(opened);
        result->handles_.reserve(64);result->descriptors_.reserve(64);result->bootstrap_.reserve(8);
        result->request_.reserve(3);result->ids_.reserve(3);result->last_.reserve(3);
        for(const auto& item:typed){const auto& d=item.descriptor;
            const sc::native::Key key{1,0,d.layer,d.expert,sc::native::RecordClass::main_mxfp4_e8m0};
            sc::BoundRecord handle;sc::BoundInfo info;
            if(result->store_->bind(key,&handle)!=sc::Error::ok||result->store_->describe(handle,&info)!=sc::Error::ok||
               info.program!=&program||info.unit!=item.unit||info.key!=key||info.length!=uint64_t(d.stride))return RAD_E_STATE;
            auto descriptor=d;descriptor.checksum=info.checksum;
            for(size_t m=0;m<4;++m)if(info.offsets[m]!=uint64_t(d.offsets[m])||info.bytes[m]!=uint64_t(d.bytes[m]))return RAD_E_STATE;
            if(d.expert==0)result->bootstrap_.push_back({item.unit,handle});
            else{result->handles_.push_back(handle);result->descriptors_.push_back(descriptor);}
        }
        if(result->descriptors_.empty()||result->bootstrap_.size()*3!=result->descriptors_.size())return RAD_E_STATE;
        *out=std::move(result);return RAD_OK;
    }catch(const std::bad_alloc&){return RAD_E_NOMEM;}catch(...){return RAD_E_STATE;}
}
int ShardedNativeSource::bootstrap(int32_t unit,std::span<std::byte> target){
    if(in_facade)return RAD_E_STATE;Active active;std::lock_guard lock(operation_);if(bootstrap_done_||closed_||!target.data())return RAD_E_STATE;
    const auto found=std::find_if(bootstrap_.begin(),bootstrap_.end(),[&](const auto& item){return item.first==unit;});
    if(found==bootstrap_.end())return RAD_E_INVAL;
    sc::BoundInfo info;if(store_->describe(found->second,&info)!=sc::Error::ok)return RAD_E_SHAPE;
    uint64_t end=0;for(size_t m=0;m<4;++m){if(info.offsets[m]>info.length||info.bytes[m]>info.length-info.offsets[m])return RAD_E_STATE;end=std::max(end,info.offsets[m]+info.bytes[m]);}
    if(target.size()!=end||target.size()>UINTPTR_MAX-uintptr_t(target.data()))return RAD_E_SHAPE;
    sc::Batch ready;const auto e=store_->acquire({&found->second,1},&ready);if(e!=sc::Error::ok)return status(e);
    if(ready.leases.size()!=1)return RAD_E_STATE;
    const auto bytes=ready.leases[0].bytes();const auto lo=uintptr_t(target.data()),hi=lo+target.size(),source=uintptr_t(bytes.data());
    if(bytes.size()>UINTPTR_MAX-source||(lo<source+bytes.size()&&source<hi))return RAD_E_INVAL;
    for(size_t m=0;m<4;++m)std::memcpy(target.data()+info.offsets[m],bytes.data()+info.offsets[m],size_t(info.bytes[m]));
    return RAD_OK;
}
int ShardedNativeSource::finish_bootstrap(){if(in_facade)return RAD_E_STATE;Active active;std::lock_guard lock(operation_);if(bootstrap_done_||closed_)return RAD_E_STATE;
    const auto e=store_->trim();if(e!=sc::Error::ok)return status(e);bootstrap_done_=true;return RAD_OK;}
int ShardedNativeSource::acquire(std::span<const uint32_t> request,rad::NativeRecordBatch* out){
    if(in_facade)return RAD_E_STATE;Active active;std::lock_guard lock(operation_);if(!out||!bootstrap_done_||closed_||request.empty()||request.size()>3)return RAD_E_INVAL;
    request_.clear();ids_.clear();
    for(const auto id:request){if(id>=handles_.size())return RAD_E_INVAL;
        if(std::find(ids_.begin(),ids_.end(),id)==ids_.end()){ids_.push_back(id);request_.push_back(handles_[id]);}}
    bool reserved=false;
    try{
        {std::lock_guard quota_lock(quota_->mutex);if(quota_->usage.live_holders>=quota_->cap)return RAD_E_SCRATCH;
            ++quota_->usage.live_holders;quota_->usage.peak_holders=std::max(quota_->usage.peak_holders,quota_->usage.live_holders);reserved=true;}
        // Prepare EVERY fallible facade allocation before Store can commit ready/HOT cells.
        auto holder=std::make_shared<Holder>(quota_);reserved=false;
        rad::NativeRecordBatch result;result.leases.reserve(ids_.size());
        const auto e=store_->acquire(request_,&holder->batch);
        if(e!=sc::Error::ok)return status(e);
        if(holder->batch.leases.size()!=ids_.size())return RAD_E_STATE;
        // Below: immutable reads, reserved-capacity inserts, shared_ptr copies and swap only.
        for(size_t n=0;n<ids_.size();++n)result.leases.push_back({ids_[n],holder->batch.leases[n].bytes(),holder});
        last_.clear();for(const auto id:ids_)last_.push_back(id);out->leases.swap(result.leases);return RAD_OK;
    }catch(const std::bad_alloc&){if(reserved){std::lock_guard quota_lock(quota_->mutex);--quota_->usage.live_holders;}return RAD_E_NOMEM;}
    catch(...){if(reserved){std::lock_guard quota_lock(quota_->mutex);--quota_->usage.live_holders;}return RAD_E_STATE;}
}
int ShardedNativeSource::close(){if(in_facade)return RAD_E_STATE;Active active;std::lock_guard lock(operation_);if(closed_)return RAD_OK;
    closed_=true;return store_?status(store_->close()):RAD_OK;}
AdapterUsage ShardedNativeSource::adapter_usage()const noexcept{std::lock_guard lock(quota_->mutex);return quota_->usage;}
} // namespace glm53::engine_core_p3
