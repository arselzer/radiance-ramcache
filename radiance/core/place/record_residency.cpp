#include "mover.h"
#include "native_inventory.h"
#include <algorithm>
#include <limits>
namespace rad {
    namespace {
        bool geometry(const Program& p,const MoveUnit& a,const MoveUnit& b){
            if(a.slot_bytes!=b.slot_bytes||a.weights.size()!=4||b.weights.size()!=4||a.offset!=b.offset)return false;
            for(size_t i=0;i<4;++i){const auto& x=p.weights[a.weights[i]];
                const auto& y=p.weights[b.weights[i]];
                if(!x.identity||!y.identity||x.widen||y.widen||x.layout_tag!=y.layout_tag||x.stored_dtype!=y.stored_dtype||x.stored_rank!=y.stored_rank||x.stored_rank>RAD_MAX_RANK||x.stored_bytes!=y.stored_bytes||x.align!=y.align||x.n_sel!=y.n_sel||x.n_sel!=1||x.sel[0]!=y.sel[0]||!rad_enc_equal(&x.enc,&y.enc))return false;
                for(uint32_t d=0;d<x.stored_rank;++d)if(x.stored_shape[d]!=y.stored_shape[d])return false;
            }
            return true;
        }
        bool range(const RecordRegion& r,uintptr_t* begin,uintptr_t* end){
            if(!r.base||r.bytes<=0||uintptr_t(r.base)%RAD_ALIGN_UNIT||uint64_t(r.bytes)>UINTPTR_MAX-uintptr_t(r.base))return false;
            *begin=uintptr_t(r.base);
            *end=*begin+uintptr_t(r.bytes);
            return true;
        }
    }
    int Residency::reserve_record_changes(){try{changed_.reserve(slot_.size());
            return RAD_OK;
        }catch(...){return RAD_E_NOMEM;
    }}
    bool Residency::record_batch_ready(std::span<const rad_weight> weights,bool known_unique)const{
        if(weights.size()>UINT64_MAX-epoch_||changed_.capacity()<slot_.size())return false;
        for(size_t i=0;i<weights.size();++i){const auto w=weights[i];
            if(!w||w>=slot_.size()||slot_[w].generation==UINT32_MAX)return false;
            if(!known_unique)for(size_t j=0;j<i;++j)if(weights[j]==w)return false;
        }
        return true;
    }
    uint64_t Mover::record_base_allocation_bytes()const{
        uint64_t bytes=uint64_t(std::max<int64_t>(static_bytes_,0));
        for(const auto& p:pools_){if(p.slab_base)bytes+=uint64_t(p.n_fixed*p.stride+RAD_ALIGN_UNIT);
            if(p.pool_base)bytes+=uint64_t(p.pool.capacity()*p.stride+RAD_ALIGN_UNIT);
        }
        return bytes;
        // static test profile only; no file/pageable/stage/loan allocations permitted
    }
    int Mover::record_session_open(std::span<const RecordClaim> claims,std::span<const RecordRegion> regions,uint64_t* token){
        if(!token||record_session_.token||!prog_||!plan_||plan_->dynamic||!compute_||claims.empty()||claims.size()>512||regions.empty()||regions.size()>1024||!cfg_.container_path.empty()||file_vram_||windows_||pageable_||stage_)return RAD_E_UNSUPPORTED;
        try{
            for(const auto& c:plan_->classes)if(c.n_shadow||c.n_stage)return RAD_E_UNSUPPORTED;
            for(const auto& mu:plan_->units)if(mu.tier!=Tier::VRAM)return RAD_E_UNSUPPORTED;
            for(const auto& pool:pools_)if(!pool.flex_ptr.empty())return RAD_E_UNSUPPORTED;
            for(const auto& move:moves_)if(move.state!=Move::State::Free)return RAD_E_STATE;
            int drained=rad_stream_sync(compute_);
            if(drained<0)return drained;
            drained=rad_stream_sync(mover_);
            if(drained<0)return drained;
            std::vector<rad_weight> changed;
            changed.reserve(claims.size()*4);
            for(size_t i=0;i<claims.size();++i){const auto& c=claims[i];
                if(c.unit<0||size_t(c.unit)>=unit_.size()||c.sentinel_region<0||size_t(c.sentinel_region)>=regions.size()||!regions[size_t(c.sentinel_region)].sentinel)return RAD_E_INVAL;
                for(size_t j=0;j<i;++j)if(claims[j].unit==c.unit)return RAD_E_INVAL;
                const auto& mu=plan_->units[size_t(c.unit)];
                if(mu.expert<0||mu.weights.size()!=4||mu.offset.size()!=4||mu.slot_bytes<=0||mu.slot_bytes%RAD_ALIGN_UNIT||mu.movable||unit_in_flight(c.unit))return RAD_E_INVAL;
                for(size_t m=0;m<mu.weights.size();++m){const auto h=mu.weights[m];
                    if(!h||h>=prog_->weights.size())return RAD_E_INVAL;
                    const auto& w=prog_->weights[h];
                    if(!w.identity||w.widen||!w.layout_tag.empty()||w.stored_bytes<=0||w.stored_rank!=2||w.align<=0||w.align>RAD_ALIGN_UNIT||mu.offset[m]<0||mu.offset[m]%w.align||mu.offset[m]>mu.slot_bytes||w.stored_bytes>mu.slot_bytes-mu.offset[m]||(m&&mu.offset[m]<mu.offset[m-1]+prog_->weights[mu.weights[m-1]].stored_bytes))return RAD_E_SHAPE;
                    changed.push_back(h);
                }
            }
            for(size_t i=0;i<regions.size();++i){const auto& r=regions[i];
                if(r.exemplar<0||size_t(r.exemplar)>=plan_->units.size()||std::none_of(claims.begin(),claims.end(),[&](const auto& c){return c.unit==r.exemplar;}))return RAD_E_INVAL;
                const auto& mu=plan_->units[size_t(r.exemplar)];
                uintptr_t lo=0,hi=0;
                if(!range(r,&lo,&hi)||mu.slot_bytes>INT64_MAX-RAD_ALIGN_UNIT||r.bytes<mu.slot_bytes+RAD_ALIGN_UNIT||(!r.sentinel&&!r.ready)||(r.sentinel&&r.ready))return RAD_E_SHAPE;
                for(size_t j=0;j<i;++j){uintptr_t a=0,b=0;
                    if(!range(regions[j],&a,&b)||(lo<b&&a<hi))return RAD_E_INVAL;
                }
                for(size_t u=0;u<plan_->units.size();++u){const auto base=reinterpret_cast<uintptr_t>(unit_base(int32_t(u)));
                    const auto n=unit_[u].slab_slot>=0?plan_->units[u].slot_bytes:plan_->units[u].bytes;
                    if(base&&n>0&&(uint64_t(n)>UINTPTR_MAX-base||(lo<base+uint64_t(n)&&base<hi)))return RAD_E_INVAL;
                }
            }
            for(const auto& c:claims)if(!geometry(*prog_,plan_->units[size_t(c.unit)],plan_->units[size_t(regions[size_t(c.sentinel_region)].exemplar)]))return RAD_E_SHAPE;
            RecordSession prepared;
            prepared.regions.assign(regions.begin(),regions.end());
            prepared.claims.assign(claims.begin(),claims.end());
            const int reserved=res_.reserve_record_changes();
            if(reserved<0)return reserved;
            if(!res_.record_batch_ready(changed))return RAD_E_STATE;
            if(record_next_token_==UINT64_MAX)return RAD_E_STATE;
            prepared.token=++record_next_token_;
            // No fallible operation remains: all member pointers publish through Mover's writer.
            record_session_=std::move(prepared);
            for(const auto& c:record_session_.claims){const auto& mu=plan_->units[size_t(c.unit)];
                auto* base=static_cast<char*>(record_session_.regions[size_t(c.sentinel_region)].base);
                for(size_t j=0;j<mu.weights.size();++j)res_.begin_transfer(mu.weights[j],base+mu.offset[j],nullptr);
            }
            *token=record_session_.token;
            return RAD_OK;
        }catch(const std::bad_alloc&){return RAD_E_NOMEM;
        }catch(...){return RAD_E_STATE;
        }
    }
    int Mover::record_publish(uint64_t token,std::span<const RecordPublication> batch){
        if(!token||token!=record_session_.token||batch.empty()||batch.size()>record_session_.claims.size())return RAD_E_INVAL;
        if(batch.size()>512)return RAD_E_INVAL;
        if(record_session_.native_only &&
           (!record_session_.quiesced || record_session_.upload_failed ||
            !budget_ || plan_->native_inventory.get()!=budget_->native_inventory() ||
            res_.native_inventory()!=budget_->native_inventory() ||
            plan_->native_inventory->validate_runtime_header(*prog_,*plan_)<0))return RAD_E_STATE;
        // Fixed stack bound512units*4weights; no allocation occurs between preflight and commit.
        rad_weight weights[2048];
        size_t count=0;
        for(size_t i=0;i<batch.size();++i){const auto& p=batch[i];
            if(p.region<0||size_t(p.region)>=record_session_.regions.size()||record_session_.regions[size_t(p.region)].sentinel||std::none_of(record_session_.claims.begin(),record_session_.claims.end(),[&](const auto& c){return c.unit==p.unit;}))return RAD_E_INVAL;
            for(size_t j=0;j<i;++j)if(batch[j].unit==p.unit||batch[j].region==p.region)return RAD_E_INVAL;
            const auto& region=record_session_.regions[size_t(p.region)];
            const auto& mu=plan_->units[size_t(p.unit)];
            if(record_session_.native_only &&
               (native_uploaded_unit_[size_t(p.region)]!=p.unit ||
                plan_->native_inventory->validate_unit(*prog_,*plan_,p.unit)<0 ||
                plan_->native_inventory->validate_unit(*prog_,*plan_,region.exemplar)<0))return RAD_E_STATE;

            if(!geometry(*prog_,mu,plan_->units[size_t(region.exemplar)]))return RAD_E_SHAPE;
            for(const auto w:mu.weights)weights[count++]=w;
        }
        if(!res_.record_batch_ready({weights,count}))return RAD_E_STATE;
        if(record_session_.native_only){
            if(batch.size()>record_session_.published.capacity())return RAD_E_STATE;
            record_session_.published.assign(batch.begin(),batch.end());
        }
        for(const auto& p:batch){const auto& mu=plan_->units[size_t(p.unit)];
            const auto& region=record_session_.regions[size_t(p.region)];
            auto* base=static_cast<char*>(region.base);
            for(size_t j=0;j<mu.weights.size();++j)res_.begin_transfer(mu.weights[j],base+mu.offset[j],region.ready);
        }
        record_session_.quiesced=false;
        return RAD_OK;
    }
    int Mover::record_restore(uint64_t token){
        if(!token||token!=record_session_.token)return RAD_E_INVAL;
        // HOST completion is deliberate in this correctness baseline. No restore/release on failure.
        int status=rad_stream_sync(compute_);
        if(status<0)return status;
        status=rad_stream_sync(mover_);
        if(status<0)return status;

        if(record_session_.native_only){
            auto clear_uploads=[&]{
                std::fill(native_uploaded_unit_.begin(),native_uploaded_unit_.end(),-1);
                std::fill(native_region_busy_.begin(),native_region_busy_.end(),uint8_t(0));
                record_session_.upload_failed=false;
            };
            if(record_session_.quiesced){clear_uploads();return RAD_OK;}
            if(!budget_ || plan_->native_inventory.get()!=budget_->native_inventory() ||
               res_.native_inventory()!=budget_->native_inventory())return RAD_E_STATE;
            if(plan_->native_inventory->validate_runtime_header(*prog_,*plan_)<0)return RAD_E_STATE;
            rad_weight weights[2048];size_t count=0;
            for(const auto& p:record_session_.published){
                if(plan_->native_inventory->validate_unit(*prog_,*plan_,p.unit)<0)return RAD_E_STATE;
                for(const auto w:plan_->units[size_t(p.unit)].weights)weights[count++]=w;
            }
            if(!res_.record_batch_ready({weights,count}))return RAD_E_STATE;
            for(const auto& p:record_session_.published){
                const auto& mu=plan_->units[size_t(p.unit)];
                auto* base=static_cast<char*>(native_sentinel(p.unit));
                if(!base)return RAD_E_STATE;
                for(size_t m=0;m<mu.weights.size();++m)
                    res_.begin_transfer(mu.weights[m],base+mu.offset[m],nullptr);
            }
            record_session_.published.clear();record_session_.quiesced=true;clear_uploads();
            return RAD_OK;
        }
        if(record_session_.quiesced)return RAD_OK;
        rad_weight weights[2048];
        size_t count=0;
        for(const auto& c:record_session_.claims)for(const auto w:plan_->units[size_t(c.unit)].weights)weights[count++]=w;
        if(!res_.record_batch_ready({weights,count}))return RAD_E_STATE;
        for(const auto& c:record_session_.claims){const auto& mu=plan_->units[size_t(c.unit)];
            auto* base=static_cast<char*>(record_session_.regions[size_t(c.sentinel_region)].base);
            for(size_t j=0;j<mu.weights.size();++j)res_.begin_transfer(mu.weights[j],base+mu.offset[j],nullptr);
        }
        record_session_.quiesced=true;
        return RAD_OK;
    }
    int Mover::record_session_close(uint64_t token){
        if(!token||token!=record_session_.token||!record_session_.quiesced)return RAD_E_STATE;
        if(record_session_.native_only){
            RAD_TRY(record_restore(token));
            if(!res_.record_batch_ready(native_weights_,true))return RAD_E_STATE;
            for(const auto w:native_weights_)res_.begin_transfer(w,nullptr,nullptr);
            native_closed_=true;record_session_=RecordSession{};
            return RAD_OK;
        }
        // Caller has shut down Ctx and its WTAB mirrors. Restore valid ORIGINAL allocated bases
        // before releasing consumer backing; this is NOT permission to run an unloaded expert.
        rad_weight weights[2048];
        size_t count=0;
        for(const auto& c:record_session_.claims)for(const auto w:plan_->units[size_t(c.unit)].weights)weights[count++]=w;
        if(!res_.record_batch_ready({weights,count}))return RAD_E_STATE;
        for(const auto& c:record_session_.claims){auto* base=static_cast<char*>(unit_base(c.unit));
            if(!base)return RAD_E_STATE;
        }
        for(const auto& c:record_session_.claims)republish(c.unit);
        record_session_=RecordSession{};
        return RAD_OK;
    }
} // namespace rad
