#include "mover.h"
#include "native_inventory.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
namespace rad {
uint32_t native_device_owner_profile_version()noexcept{return kNativeDeviceOwnerProfileVersion;}
namespace {
bool native_backend_allowed(const NativeInventory& inventory){
    if(inventory.limits().backend_profile==NativeBackendProfile::HostOnly)return device_is_host();
    return (inventory.limits().backend_profile==NativeBackendProfile::TinyAqlV1||
            inventory.limits().backend_profile==NativeBackendProfile::GlmMainTp4AqlFailStopV1)&&
        !device_is_host()&&!std::strcmp(device_backend_name(),"aql");
}
}
void* Mover::native_sentinel(int32_t unit)const{
    const auto* inventory=budget_?budget_->native_inventory():nullptr;
    if(native_closed_||!plan_||!inventory||plan_->native_inventory.get()!=inventory)return nullptr;
    const auto* owned=inventory->find(unit);
    if(!owned)return nullptr;
    const auto& cls=inventory->classes()[size_t(owned->klass)];
    if(cls.first_region<0||size_t(cls.first_region)>=native_regions_.size())return nullptr;
    return native_regions_[size_t(cls.first_region)].base;
}
int Mover::native_record_init(){
    if(!plan_->native_inventory)return RAD_OK;
    if(!budget_||!mover_||!native_regions_.empty()||!native_backend_allowed(*plan_->native_inventory))return RAD_E_UNSUPPORTED;
    const auto& inventory=*plan_->native_inventory;
    RAD_TRY(inventory.validate(*prog_,*plan_));
    const auto& f=inventory.footprint();
    try{
        native_regions_.reserve(f.regions);native_landing_.reserve(f.regions);
        native_landing_device_.reserve(f.regions);native_device_allocation_.reserve(f.regions);
        native_host_allocation_.reserve(f.regions);native_weights_.reserve(inventory.owned().size()*4);
        native_uploaded_unit_.assign(f.regions,-1);
        native_region_busy_.assign(f.regions,0);
        for(const auto& owned:inventory.owned())
            for(const auto w:plan_->units[size_t(owned.unit)].weights)native_weights_.push_back(w);
        for(const auto& cls:inventory.classes()){
            if(cls.first_region!=int32_t(native_regions_.size()))return RAD_E_STATE;
            const auto& mu=plan_->units[size_t(cls.exemplar)];
            for(int32_t i=0;i<=cls.slots;++i){
                auto* host=static_cast<char*>(reserve("native record landing",cls.allocation_bytes,RAD_MEM_HOST_PINNED));
                auto* device=static_cast<char*>(reserve("native record consumer",cls.allocation_bytes,RAD_MEM_DEVICE));
                if(!host||!device)return RAD_E_NOMEM;
                // AQL owns pooled pinned ranges, including interior reservations. HOST's
                // existing test allocator can use unregistered equal-address HOST storage.
                auto* host_view=device_is_host()?host:static_cast<char*>(rad_dev_host_ptr(host));
                auto* on_card=device_is_host()?host:static_cast<char*>(rad_dev_device_ptr(host));
                if(!host_view||!on_card||uintptr_t(host_view)%4096||uintptr_t(device)%4096||uintptr_t(on_card)%4096)
                    return RAD_E_ALIGN;
                std::memset(host_view,0xbd,size_t(cls.allocation_bytes));
                auto* body=host_view+4096;
                std::memset(body,0,size_t(cls.region_bytes));
                if(i==0){
                    for(size_t m=0;m<mu.weights.size();++m){
                        const auto& w=prog_->weights[mu.weights[m]];
                        if(w.stored_dtype==RAD_E8M0)
                            std::memset(body+mu.offset[m],127,size_t(w.stored_bytes));
                        else if(w.stored_dtype==RAD_F32){
                            auto* scales=reinterpret_cast<float*>(body+mu.offset[m]);
                            for(int64_t n=0;n<w.stored_bytes/4;++n)scales[n]=1.f;
                        }
                    }
                }
                RadEvent ready=nullptr;
                if(i)RAD_TRY(rad_event_create(&ready));
                native_regions_.push_back({cls.exemplar,device+4096,cls.region_bytes,ready,i==0});
                native_landing_.push_back(body);native_landing_device_.push_back(on_card+4096);
                native_device_allocation_.push_back(device);native_host_allocation_.push_back(host_view);
                rad_dev_host_wrote();
                const int copied=rad_memcpy_async(device,on_card,cls.allocation_bytes,mover_);
                if(cfg_.tiny_budget){
                    // Installed before this first asynchronous bootstrap copy.
                    // A failed REAL drain cannot unwind landing/pool storage.
                    tiny_drain_or_fail(mover_,0);
                    if(copied<0)return copied;
                    continue;
                }
                const int drained=rad_stream_sync(mover_);
                if(drained<0)return drained; // HOST-only profile; no GPU fault-recovery claim.
                if(copied<0)return copied;
            }
        }
        return RAD_OK;
    }catch(const std::bad_alloc&){return RAD_E_NOMEM;}
    catch(...){return RAD_E_STATE;}
}
int Mover::native_record_session_open(uint64_t* token){
    if(!token||record_session_.token||!plan_||!plan_->native_inventory||!compute_||
       native_closed_||native_regions_.empty()||!native_backend_allowed(*plan_->native_inventory)||
       plan_->dynamic||plan_->strategy!=Strategy::NativeBacked||!cfg_.container_path.empty()||
       stage_||pageable_||file_vram_||windows_)return RAD_E_UNSUPPORTED;
    if(!budget_||plan_->native_inventory.get()!=budget_->native_inventory()||
       res_.native_inventory()!=budget_->native_inventory())return RAD_E_STATE;
    RAD_TRY(plan_->native_inventory->validate(*prog_,*plan_));
    RAD_TRY(rad_stream_sync(compute_));
    RAD_TRY(rad_stream_sync(mover_));
    try{
        RecordSession next;
        const auto& inventory=*plan_->native_inventory;
        next.claims.reserve(inventory.owned().size());next.regions=native_regions_;
        next.published.reserve(inventory.footprint().consumer_slots);
        for(const auto& owned:inventory.owned()){
            const auto& cls=inventory.classes()[size_t(owned.klass)];
            const auto& mu=plan_->units[size_t(owned.unit)];
            for(size_t m=0;m<mu.weights.size();++m){
                const auto* slot=res_.lookup(mu.weights[m]);
                if(!slot||slot->ptr!=static_cast<char*>(native_sentinel(owned.unit))+mu.offset[m]||
                   slot->ready)return RAD_E_STATE;
            }
            next.claims.push_back({owned.unit,cls.first_region});
        }
        if(record_next_token_==UINT64_MAX)return RAD_E_STATE;
        next.token=++record_next_token_;next.native_only=true;
        record_session_=std::move(next);*token=record_session_.token;
        return RAD_OK;
    }catch(const std::bad_alloc&){return RAD_E_NOMEM;}
    catch(...){return RAD_E_STATE;}
}
int Mover::native_record_upload(uint64_t token,int32_t unit,int32_t region,std::span<const std::byte> bytes){
    if(!token||token!=record_session_.token||!record_session_.native_only||
       !record_session_.quiesced||region<0||size_t(region)>=native_regions_.size()||
       native_regions_[size_t(region)].sentinel||!bytes.data())return RAD_E_INVAL;
    if(!budget_||plan_->native_inventory.get()!=budget_->native_inventory()||
       res_.native_inventory()!=budget_->native_inventory())return RAD_E_STATE;
    if(record_session_.upload_failed||native_region_busy_[size_t(region)])return RAD_E_STATE;
    const auto& r=native_regions_[size_t(region)];
    if(plan_->native_inventory->validate_unit(*prog_,*plan_,r.exemplar)<0||
       plan_->native_inventory->validate_unit(*prog_,*plan_,unit)<0)return RAD_E_STATE;
    const auto& cls=plan_->units[size_t(r.exemplar)];
    const auto* owned=plan_->native_inventory->find(unit);
    const auto* exemplar=plan_->native_inventory->find(r.exemplar);
    if(!owned||!exemplar||owned->klass!=exemplar->klass)return RAD_E_INVAL;
    if(bytes.size()!=uint64_t(cls.slot_bytes)||bytes.size()>UINTPTR_MAX-uintptr_t(bytes.data()))
        return RAD_E_SHAPE;
    uint64_t checksum=14695981039346656037ull;
    for(const auto b:bytes){checksum^=std::to_integer<uint8_t>(b);checksum*=1099511628211ull;}
    if(checksum!=owned->descriptor.checksum)return RAD_E_INVAL;
    const auto lo=uintptr_t(bytes.data()),hi=lo+bytes.size();
    for(size_t i=0;i<native_regions_.size();++i){
        const auto* owned=plan_->native_inventory->find(native_regions_[i].exemplar);
        if(!owned)return RAD_E_STATE;
        const auto& c=plan_->native_inventory->classes()[size_t(owned->klass)];
        for(const auto base:{native_host_allocation_[i],native_device_allocation_[i]}){
            const auto a=uintptr_t(base),z=a+uint64_t(c.allocation_bytes);
            if(lo<z&&a<hi)return RAD_E_INVAL;
        }
    }
    // Busy precedes the FIRST host mutation. Neither successful nor failed queued work may
    // share this landing region with another upload until BOTH streams successfully drain.
    native_region_busy_[size_t(region)]=1;
    native_uploaded_unit_[size_t(region)]=-1;
    std::memcpy(native_landing_[size_t(region)],bytes.data(),bytes.size());
    // The fixed initialized reader tail remains zero; outer guards stay untouched.
    rad_dev_host_wrote();
    const int copied=rad_memcpy_async(r.base,native_landing_device_[size_t(region)],r.bytes,mover_);
    if(copied<0){record_session_.upload_failed=true;return copied;}
    const int recorded=rad_event_record(r.ready,mover_);
    if(recorded>=0)native_uploaded_unit_[size_t(region)]=unit;
    else record_session_.upload_failed=true;
    return recorded;
}
int Mover::native_record_wait_prepared(uint64_t token,std::span<const RecordPublication> expected){
    if(!token||token!=record_session_.token||!record_session_.native_only||native_closed_||
       !prog_||!plan_||!budget_||!mover_||!compute_||!plan_->native_inventory||
       plan_->native_inventory.get()!=budget_->native_inventory()||
       res_.native_inventory()!=budget_->native_inventory()||
       !native_backend_allowed(*plan_->native_inventory)||record_session_.upload_failed)
        return RAD_E_STATE;
    const auto& inventory=*plan_->native_inventory;
    if(inventory.validate_runtime_header(*prog_,*plan_)<0||native_regions_.size()>512||
       native_uploaded_unit_.size()!=native_regions_.size()||native_region_busy_.size()!=native_regions_.size()||
       expected.size()>512||expected.size()!=record_session_.published.size()||
       (expected.empty()!=record_session_.quiesced))return RAD_E_STATE;
    // Single rank-thread contract. Preserve exact generations across the blocking
    // transfer fence; no pointer/event generation is settled or republished here.
    std::array<uint8_t,512> selected{};
    std::array<rad_weight,2048> weights{};
    std::array<uint64_t,2048> generations{};
    size_t count=0;
    auto validate=[&](bool capture)->int{
        if(inventory.validate_runtime_header(*prog_,*plan_)<0||record_session_.upload_failed)return RAD_E_STATE;
        size_t at=0;
        for(size_t i=0;i<expected.size();++i){const auto& wanted=expected[i];const auto& published=record_session_.published[i];
            if(wanted.unit!=published.unit||wanted.region!=published.region||wanted.region<0||
               size_t(wanted.region)>=native_regions_.size())return RAD_E_STATE;
            const auto region_index=size_t(wanted.region);const auto& region=native_regions_[region_index];
            if(region.sentinel||!region.base||!region.ready||!native_region_busy_[region_index]||
               native_uploaded_unit_[region_index]!=wanted.unit||
               inventory.validate_unit(*prog_,*plan_,wanted.unit)<0||
               inventory.validate_unit(*prog_,*plan_,region.exemplar)<0)return RAD_E_STATE;
            if(capture){if(selected[region_index])return RAD_E_STATE;selected[region_index]=1;}
            const auto& unit=plan_->units[size_t(wanted.unit)];
            if(unit.weights.size()!=4||at>weights.size()-4)return RAD_E_STATE;
            for(size_t m=0;m<4;++m){const auto w=unit.weights[m];const auto* slot=res_.lookup(w);
                if(!slot||slot->ptr!=static_cast<char*>(region.base)+unit.offset[m]||
                   slot->ready!=region.ready||!slot->generation)return RAD_E_STATE;
                if(capture){weights[at]=w;generations[at]=slot->generation;}
                else if(weights[at]!=w||generations[at]!=slot->generation)return RAD_E_STATE;
                ++at;
            }
        }
        if(!capture&&at!=count)return RAD_E_STATE;
        for(size_t r=0;r<native_regions_.size();++r)
            if(bool(native_region_busy_[r])!=bool(selected[r])||(!selected[r]&&native_uploaded_unit_[r]!=-1))return RAD_E_STATE;
        if(capture)count=at;
        return RAD_OK;
    };
    RAD_TRY(validate(true));
    const int completed=rad_stream_sync(mover_);
    if(completed<0){record_session_.upload_failed=true;return completed;}
    const int unchanged=validate(false);
    if(unchanged<0)record_session_.upload_failed=true;
    return unchanged;
}
int Mover::native_record_guards_host()const{
    const auto* inventory=budget_?budget_->native_inventory():nullptr;
    if(!device_is_host()||!inventory)return RAD_E_UNSUPPORTED;
    for(size_t i=0;i<native_regions_.size();++i){
        const auto* owned=inventory->find(native_regions_[i].exemplar);
        if(!owned)return RAD_E_STATE;
        const auto& cls=inventory->classes()[size_t(owned->klass)];
        for(const auto* base:{native_host_allocation_[i],native_device_allocation_[i]}){
            const auto* raw=reinterpret_cast<const uint8_t*>(base);
            const auto guarded=[](uint8_t x){return x==0xbd;};
            if(!std::all_of(raw,raw+4096,guarded)||
               !std::all_of(raw+4096+cls.region_bytes,raw+cls.allocation_bytes,guarded))
                return RAD_E_STATE;
            if(!std::all_of(raw+4096+cls.stride,raw+4096+cls.region_bytes,
                            [](uint8_t x){return x==0;}))return RAD_E_STATE;
        }
    }
    return RAD_OK;
}
int Mover::native_record_guards_aql(void* host,void* device,int64_t capacity,RadStream stream)const{
    const auto* inventory=budget_?budget_->native_inventory():nullptr;
    if(!inventory||inventory->limits().backend_profile!=NativeBackendProfile::TinyAqlV1||
       !native_backend_allowed(*inventory)||!host||!device||stream!=compute_||capacity<49152||
       rad_dev_host_ptr(host)!=host||rad_dev_device_ptr(host)!=device||
       uintptr_t(host)>UINTPTR_MAX-uint64_t(capacity)||uintptr_t(device)>UINTPTR_MAX-uint64_t(capacity))
        return RAD_E_UNSUPPORTED;
    for(size_t i=0;i<native_regions_.size();++i){
        const auto* owned=inventory->find(native_regions_[i].exemplar);if(!owned)return RAD_E_STATE;
        const auto& cls=inventory->classes()[size_t(owned->klass)];
        for(const auto base:{native_host_allocation_[i],native_device_allocation_[i]}){
            const auto a=uintptr_t(base),z=a+uint64_t(cls.allocation_bytes);
            const auto lo=uintptr_t(host),hi=lo+uint64_t(capacity);
            const auto dl=uintptr_t(device),dh=dl+uint64_t(capacity);
            if((lo<z&&a<hi)||(dl<z&&a<dh))return RAD_E_INVAL;
        }
    }
    RAD_TRY(rad_stream_sync(compute_));RAD_TRY(rad_stream_sync(mover_));
    const auto checked=[](const uint8_t* raw,const NativeClassReservation& cls){
        return std::all_of(raw,raw+4096,[](uint8_t x){return x==0xbd;})&&
            std::all_of(raw+4096+cls.region_bytes,raw+cls.allocation_bytes,[](uint8_t x){return x==0xbd;})&&
            std::all_of(raw+4096+cls.stride,raw+4096+cls.region_bytes,[](uint8_t x){return x==0;});
    };
    for(size_t i=0;i<native_regions_.size();++i){
        const auto* owned=inventory->find(native_regions_[i].exemplar);const auto& cls=inventory->classes()[size_t(owned->klass)];
        if(cls.allocation_bytes>capacity||!checked(reinterpret_cast<const uint8_t*>(native_host_allocation_[i]),cls))return RAD_E_STATE;
        RAD_TRY(rad_memcpy_async(device,native_device_allocation_[i],cls.allocation_bytes,stream));
        RAD_TRY(rad_stream_sync(stream));
        if(!checked(static_cast<const uint8_t*>(host),cls))return RAD_E_STATE;
    }
    return RAD_OK;
}
} // namespace rad
