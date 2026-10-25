#include "native_inventory.h"
#include "device/device.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
namespace rad {
namespace {
struct Failure {int status;const char* message;};
void require(bool b,int status,const char* why){if(!b)throw Failure{status,why};}
uint64_t add(uint64_t a,uint64_t b){require(b<=UINT64_MAX-a,RAD_E_SHAPE,"native byte addition overflow");return a+b;}
uint64_t mul(uint64_t a,uint64_t b){require(!a||b<=UINT64_MAX/a,RAD_E_SHAPE,"native byte product overflow");return a*b;}
uint64_t aligned(uint64_t n){return add(n,4095)/4096*4096;}
template<class T,size_t N> bool same(const T(&a)[N],const T(&b)[N]){
    return std::equal(a,a+N,b);
}
bool metadata(const WeightInfo& a,const WeightInfo& b,bool names){
    return native_weight_metadata_equal(a,b,names);
}
bool unit_metadata(const MoveUnit& a,const MoveUnit& b){
    return a.layer==b.layer&&a.expert==b.expert&&a.weights==b.weights&&a.offset==b.offset&&
        a.bytes==b.bytes&&a.slot_bytes==b.slot_bytes&&a.access==b.access&&
        a.first_use_op==b.first_use_op&&a.last_use_op==b.last_use_op&&a.slab_class==b.slab_class;
}
bool class_metadata(const SlabClass& a,const SlabClass& b){
    return a.name==b.name&&a.stride==b.stride&&a.n_units==b.n_units&&
        a.n_slots==b.n_slots&&a.n_shadow==b.n_shadow&&a.n_stage==b.n_stage&&a.movable==b.movable;
}
bool geometry(const Program& p,const MoveUnit& a,const MoveUnit& b){
    if(a.slot_bytes!=b.slot_bytes||a.offset!=b.offset||a.weights.size()!=b.weights.size())return false;
    for(size_t i=0;i<a.weights.size();++i)
        if(!metadata(p.weights[a.weights[i]],p.weights[b.weights[i]],false))return false;
    return true;
}
void explain(std::string* why,const char* s)noexcept{if(why)try{*why=s;}catch(...){}}
}
struct NativeInventory::Impl {
    const Program* program=nullptr;
    NativeInventoryLimits limits;
    NativeBackingFootprint footprint;
    std::unique_ptr<NativeWeightAuthority> authority;
    std::vector<MoveUnit> units;
    std::vector<int32_t> unit_of_weight;
    std::vector<SlabClass> plan_classes;
    std::vector<int32_t> owned_index;
    std::vector<NativeOwnedUnit> owned;
    std::vector<NativeClassReservation> classes;
    std::vector<ExpertRouteMap> routes;
};
NativeInventory::NativeInventory(std::unique_ptr<Impl> p):p_(std::move(p)){}
NativeInventory::~NativeInventory()=default;
const Program* NativeInventory::program()const noexcept{return p_->program;}
std::span<const NativeOwnedUnit> NativeInventory::owned()const noexcept{return p_->owned;}
std::span<const NativeClassReservation> NativeInventory::classes()const noexcept{return p_->classes;}
const NativeBackingFootprint& NativeInventory::footprint()const noexcept{return p_->footprint;}
const NativeAuthorityStats& NativeInventory::authority_stats()const noexcept{return p_->authority->stats();}
const NativeInventoryLimits& NativeInventory::limits()const noexcept{return p_->limits;}
const NativeOwnedUnit* NativeInventory::find(int32_t unit)const noexcept{
    if(unit<0||size_t(unit)>=p_->owned_index.size())return nullptr;
    const auto i=p_->owned_index[size_t(unit)];
    return i<0?nullptr:&p_->owned[size_t(i)];
}
int NativeInventory::bind(const Program& program,const Plan& layout,
                         std::span<const NativeRecordDescriptor> catalog,
                         std::span<const RecordOwnerBinding> owned,
                         std::span<const int32_t> residents,const NativeInventoryLimits& limits,
                         std::shared_ptr<const NativeInventory>* out,std::string* why){
    try{
        require(out&&!*out,RAD_E_INVAL,"native inventory output must be empty");
        require(program.ctx.world_size==1&&program.ctx.rank==0&&layout.world_size==1&&
                layout.rank==0&&!layout.dynamic&&layout.strategy==Strategy::AllVram,
                RAD_E_UNSUPPORTED,"initial native inventory requires truthful static TP1 layout");
        require(limits.backend_profile==NativeBackendProfile::HostOnly||
                limits.backend_profile==NativeBackendProfile::TinyAqlV1,
                RAD_E_UNSUPPORTED,"unknown native backend profile");
        if(limits.backend_profile==NativeBackendProfile::TinyAqlV1){
            require(!device_is_host()&&!std::strcmp(device_backend_name(),"aql")&&
                    program.requires_permanent_model_failure()&&program.ctx.max_tok==5&&program.ctx.max_seqs==1&&program.ctx.max_ctx==5&&limits.max_rows==2&&limits.top_k==2&&
                    owned.size()==24&&catalog.size()==24&&residents.size()==8&&
                    layout.units.size()<=128&&program.weights.size()<=512&&
                    program.expert_route_maps.size()==8&&limits.hot_bytes<=65536&&
                    limits.cold_bytes<=98304&&limits.max_metadata_bytes<=(4u<<20)&&
                    limits.max_device_bytes<=(4u<<20)&&limits.max_landing_bytes<=(4u<<20)&&
                    limits.max_weights_bytes<=(4u<<20)&&limits.max_host_bytes<=(64u<<20),
                    RAD_E_UNSUPPORTED,"tiny AQL Native profile/capability/topology/budget");
            bool seen[8][4]{};size_t experts=0;constexpr int64_t offsets[]{0,16384,20480,28672};
            for(const auto& unit:layout.units)if(unit.expert>=0){
                require(unit.layer>=0&&unit.layer<8&&unit.expert<4&&!seen[unit.layer][unit.expert]&&
                        unit.slot_bytes==32768&&unit.weights.size()==4&&
                        unit.offset.size()==4&&std::equal(unit.offset.begin(),unit.offset.end(),offsets),
                        RAD_E_SHAPE,"tiny AQL exact expert record geometry");
                seen[unit.layer][unit.expert]=true;++experts;
                constexpr int64_t shape[4][2]{{256,128},{256,4},{128,128},{128,4}};
                constexpr int64_t bytes[]{16384,1024,8192,512};
                for(size_t m=0;m<4;++m){const auto h=unit.weights[m];require(h&&h<program.weights.size(),RAD_E_INVAL,"tiny AQL member handle");const auto& w=program.weights[h];
                    require(w.stored_rank==2&&w.stored_shape[0]==shape[m][0]&&w.stored_shape[1]==shape[m][1]&&w.stored_bytes==bytes[m],
                            RAD_E_SHAPE,"tiny AQL H128/localFF128 member shapes");
                    const auto main=rad_enc_affine(RAD_FP4E2M1,RAD_E8M0,1,32);
                    const auto dtype=m%2?RAD_E8M0:RAD_FP4E2M1;
                    require(w.enc_known&&rad_enc_equal(&w.enc,&main)&&w.identity&&!w.widen&&
                            w.layout_tag.empty()&&w.n_sel==1&&w.sel[0]==int(m%2)&&
                            w.stored_dtype==dtype&&w.decl.dtype==dtype&&w.decl.rank==2&&
                            w.decl.shape[0]==shape[m][0]&&w.decl.shape[1]==shape[m][1]&&
                            w.decl.shard==(m<2?RAD_SHARD_ROW:RAD_SHARD_COL)&&
                            w.decl.group.layer==unit.layer&&w.decl.group.expert==unit.expert&&
                            w.decl.group.slot==int(m/2),RAD_E_DTYPE,
                            "tiny AQL canonical main members including fixed residents");
                }

            }
            require(experts==32,RAD_E_SHAPE,"tiny AQL complete32 expert units");
            for(const auto u:residents)require(u>=0&&size_t(u)<layout.units.size()&&
                layout.units[size_t(u)].expert==0,RAD_E_INVAL,"tiny AQL expert0 residents only");
            for(const auto& route:program.expert_route_maps)require(route.decl.layer>=0&&route.decl.layer<8&&
                route.decl.canonical_experts==4&&route.decl.local_experts==4&&route.decl.regular_experts==4&&
                route.decl.expert_base==0&&route.decl.max_rows==2&&route.decl.top_k==2,
                RAD_E_SHAPE,"tiny AQL full local current-route profile");
        }
        require(limits.max_rows&&limits.max_rows<=2048&&limits.top_k&&limits.top_k<=32&&
                limits.max_owned_units&&limits.max_owned_units<=16384&&limits.max_classes&&
                limits.max_classes<=64&&limits.max_regions>=2&&limits.max_regions<=512&&
                limits.max_device_bytes&&limits.max_device_bytes<=(uint64_t(8)<<30)&&
                limits.max_landing_bytes&&limits.max_landing_bytes<=(uint64_t(8)<<30)&&
                limits.max_weights_bytes&&limits.max_weights_bytes<=(uint64_t(128)<<30)&&
                limits.max_host_bytes&&limits.max_host_bytes<=(uint64_t(128)<<30)&&
                limits.max_metadata_bytes&&limits.max_metadata_bytes<=(uint64_t(128)<<20)&&
                limits.cold_bytes&&limits.cold_bytes<=uint64_t(INT64_MAX)&&
                limits.hot_bytes<=uint64_t(INT64_MAX),RAD_E_INVAL,"native limits invalid");
        require(!owned.empty()&&owned.size()<=limits.max_owned_units&&
                !catalog.empty()&&catalog.size()<=65536&&layout.units.size()<=65536&&
                program.weights.size()>1&&program.weights.size()<=262144&&
                layout.unit_of_weight.size()==program.weights.size()&&layout.unit_of_weight[0]==-1&&
                residents.size()<=layout.units.size()&&!layout.classes.empty()&&
                layout.classes.size()<=layout.units.size()&&layout.classes.size()<=4096,
                RAD_E_SHAPE,"native inventory count caps");
        // Preserve the old non-weight/Owner/Mover reservations. Weight authority is
        // separately scanned with bounded borrowed scratch and exact-size owned arrays.
        uint64_t reservation=65536;
        reservation=add(reservation,mul(program.weights.size(),128));
        reservation=add(reservation,mul(layout.units.size(),2*sizeof(MoveUnit)+128));
        reservation=add(reservation,mul(layout.unit_of_weight.size(),2*sizeof(int32_t)));
        reservation=add(reservation,mul(layout.classes.size(),2*sizeof(SlabClass)+128));
        reservation=add(reservation,mul(layout.classes.size(),sizeof(size_t)));
        reservation=add(reservation,mul(limits.max_regions,sizeof(uint8_t)));
        reservation=add(reservation,mul(owned.size(),2*sizeof(NativeOwnedUnit)+64));
        reservation=add(reservation,mul(limits.max_regions,1024));
        for(const auto& w:program.weights){
            require(w.name.size()<=512&&w.source.size()<=512&&w.layout_tag.size()<=512,
                    RAD_E_SHAPE,"native metadata string cap");
        }
        for(const auto& u:layout.units){
            require(u.weights.size()==u.offset.size()&&u.weights.size()<=65536,
                    RAD_E_SHAPE,"unit member count");
            reservation=add(reservation,mul(u.weights.size(),2*(sizeof(rad_weight)+sizeof(int64_t))));
        }
        for(const auto& cls:layout.classes){
            require(cls.name.size()<=512&&cls.stride>0&&cls.stride%4096==0&&
                    !cls.movable&&!cls.n_slots&&!cls.n_shadow&&!cls.n_stage,
                    RAD_E_SHAPE,"actual static class metadata cap");
            reservation=add(reservation,mul(cls.name.size(),4));
        }
        require(program.expert_route_maps.size()<=256,RAD_E_SHAPE,"native route map cap");
        for(const auto& route:program.expert_route_maps){
            require(route.entries.size()<=512,RAD_E_SHAPE,"native route entry cap");
            reservation=add(reservation,add(2*sizeof(ExpertRouteMap),
                mul(route.entries.size(),2*sizeof(RadExpertRouteEntry)+16)));
        }
        reservation=add(reservation,catalog.size());
        require(reservation<=limits.max_metadata_bytes,RAD_E_SCRATCH,"native metadata reservation cap");
        // Recheck the pure Planner's grouping/offset result rather than accepting a forged
        // same-sized Plan. Each actual Program weight occurs exactly once in its declared group.
        std::vector<size_t> class_unit_counts(layout.classes.size(),0);
        for(size_t u=0;u<layout.units.size();++u){
            const auto& unit=layout.units[u];uint64_t end=0;
            require(!unit.weights.empty()&&unit.slab_class>=0&&
                    size_t(unit.slab_class)<layout.classes.size(),RAD_E_SHAPE,"empty/outside actual unit class");
            ++class_unit_counts[size_t(unit.slab_class)];
            for(size_t m=0;m<unit.weights.size();++m){
                const auto h=unit.weights[m];
                require(h&&h<program.weights.size()&&layout.unit_of_weight[h]==int32_t(u),
                        RAD_E_SHAPE,"actual unit weight map mismatch");
                const auto& w=program.weights[h];
                require(w.align>0&&w.align<=4096&&4096%w.align==0&&w.stored_bytes>0&&
                        (w.decl.group.layer<0 ? unit.layer<0 :
                         unit.layer==w.decl.group.layer&&unit.expert==w.decl.group.expert),
                        RAD_E_SHAPE,"actual unit group/alignment mismatch");
                const auto at=add(end,uint64_t(w.align)-1)/uint64_t(w.align)*uint64_t(w.align);
                require(unit.offset[m]>=0&&uint64_t(unit.offset[m])==at,
                        RAD_E_SHAPE,"actual unit member offset mismatch");
                end=add(at,uint64_t(w.stored_bytes));
            }
            require(end==uint64_t(unit.bytes)&&aligned(end)==uint64_t(unit.slot_bytes),
                    RAD_E_SHAPE,"actual unit byte/stride mismatch");
            require(layout.classes[size_t(unit.slab_class)].stride==unit.slot_bytes,
                    RAD_E_SHAPE,"actual unit class stride mismatch");
        }
        for(size_t c=0;c<layout.classes.size();++c)
            require(class_unit_counts[c]&&layout.classes[c].n_units==int64_t(class_unit_counts[c]),
                    RAD_E_SHAPE,"unused/inconsistent actual class");
        for(size_t h=1;h<program.weights.size();++h){
            const auto u=layout.unit_of_weight[h];
            require(u>=0&&size_t(u)<layout.units.size()&&
                    std::count(layout.units[size_t(u)].weights.begin(),layout.units[size_t(u)].weights.end(),
                               rad_weight(h))==1,RAD_E_SHAPE,"actual Program weight coverage mismatch");
        }
        std::unique_ptr<NativeWeightAuthority> authority;
        require(reservation<limits.max_metadata_bytes,RAD_E_SCRATCH,"native authority budget exhausted before scratch");
        const int captured=NativeWeightAuthority::capture(program.weights,layout.unit_of_weight,
            layout.units,limits.max_metadata_bytes-reservation,&authority,why);
        if(captured<0)return captured;
        reservation=add(reservation,authority->stats().construction_peak_bytes);
        require(reservation<=limits.max_metadata_bytes,RAD_E_SCRATCH,"native compact metadata reservation cap");
        auto p=std::make_unique<Impl>();p->program=&program;p->limits=limits;
        p->authority=std::move(authority);
        p->footprint.metadata_reservation=reservation;
        p->owned_index.assign(layout.units.size(),-1);
        p->owned.reserve(owned.size());p->classes.reserve(limits.max_classes);
        std::vector<uint8_t> coverage(layout.units.size(),0);
        std::vector<uint8_t> used_record(catalog.size(),0);
        for(const auto u:residents){
            require(u>=0&&size_t(u)<layout.units.size()&&layout.units[size_t(u)].expert>=0&&
                    !coverage[size_t(u)],RAD_E_INVAL,"resident mask duplicate/nonexpert/outside unit");
            coverage[size_t(u)]=1;
        }
        for(const auto& b:owned){
            require(b.unit>=0&&size_t(b.unit)<layout.units.size()&&b.record<catalog.size()&&
                    !coverage[size_t(b.unit)],RAD_E_INVAL,"native ownership duplicate/resident overlap");
            require(!used_record[b.record],RAD_E_INVAL,"native record aliased across units");
            used_record[b.record]=1;
            const auto& u=layout.units[size_t(b.unit)];const auto& d=catalog[b.record];
            require(u.layer>=0&&u.expert>=0&&u.access==RAD_ACCESS_CONDITIONAL&&
                    u.tier==Tier::VRAM&&!u.movable&&u.weights.size()==4&&u.offset.size()==4&&
                    u.slot_bytes>0&&u.slot_bytes%4096==0&&u.bytes>0,
                    RAD_E_SHAPE,"native ownership requires four-member actual native expert");
            require(d.world==1&&d.rank==0&&d.layer==u.layer&&d.expert==u.expert&&
                    d.source_generation&&d.stride==u.slot_bytes,RAD_E_INVAL,"native source/rank/stride identity");
            require(p->owned.empty()||d.source_generation==p->owned.front().descriptor.source_generation,
                    RAD_E_INVAL,"mixed native source generations");
            require(u.slot_bytes<=INT64_MAX-16384,RAD_E_SHAPE,"native region extent overflow");
            for(size_t m=0;m<4;++m){
                const auto h=u.weights[m];
                require(h&&h<program.weights.size()&&layout.unit_of_weight[h]==b.unit,
                        RAD_E_INVAL,"native member ownership");
                const auto& w=program.weights[h];
                const auto mx=rad_enc_affine(RAD_FP4E2M1,RAD_E8M0,1,32);
                const auto mtp=rad_enc_affine(RAD_F8E4M3,RAD_F32,128,128);
                require(w.enc_known&&(rad_enc_equal(&w.enc,&mx)||rad_enc_equal(&w.enc,&mtp))&&
                        w.identity&&!w.widen&&w.layout_tag.empty()&&w.n_sel==1&&
                        w.sel[0]==int(m%2)&&w.decl.rank==2&&w.stored_rank==2&&
                        w.stored_dtype==w.enc.plane[w.sel[0]].dtype&&
                        w.decl.dtype==w.stored_dtype&&w.decl.group.layer==u.layer&&
                        w.decl.group.expert==u.expert&&w.align>0&&w.align<=4096&&
                        w.stored_bytes>0&&u.offset[m]>=0&&u.offset[m]%4096==0&&
                        u.offset[m]<=u.slot_bytes&&w.stored_bytes<=u.slot_bytes-u.offset[m],
                        RAD_E_SHAPE,"native member canonical layout/encoding");
                require(d.offsets[m]==u.offset[m]&&d.bytes[m]==w.stored_bytes&&
                        d.dtype[m]==w.stored_dtype&&d.shape[m][0]==w.stored_shape[0]&&
                        d.shape[m][1]==w.stored_shape[1],RAD_E_SHAPE,"native descriptor member mismatch");
                if(m)require(u.offset[m]>=u.offset[m-1]+program.weights[u.weights[m-1]].stored_bytes,
                             RAD_E_SHAPE,"native member overlap");
            }
            const auto map=std::find_if(program.expert_route_maps.begin(),program.expert_route_maps.end(),
                                       [&](const auto& x){return x.decl.layer==u.layer;});
            require(map!=program.expert_route_maps.end()&&map->decl.top_k==limits.top_k&&
                    map->decl.max_rows>=limits.max_rows&&map->decl.expert_base==0&&
                    map->decl.local_experts==map->decl.canonical_experts&&
                    size_t(u.expert)<map->entries.size(),RAD_E_SHAPE,"owned actual route map/top-k");
            const auto& entry=map->entries[size_t(u.expert)];
            require(!entry.protected_expert&&std::equal(u.weights.begin(),u.weights.end(),entry.members),
                    RAD_E_INVAL,"native canonical/runtime route members");
            int32_t klass=-1;
            for(size_t c=0;c<p->classes.size();++c)
                if(geometry(program,u,layout.units[size_t(p->classes[c].exemplar)])){klass=int32_t(c);break;}
            if(klass<0){
                require(p->classes.size()<limits.max_classes,RAD_E_SHAPE,"native exact layout class cap");
                klass=int32_t(p->classes.size());
                p->classes.push_back({b.unit,0,0,u.slot_bytes,0,0});
            }
            p->owned_index[size_t(b.unit)]=int32_t(p->owned.size());
            p->owned.push_back({b.unit,klass,b.record,d});coverage[size_t(b.unit)]=2;
            p->footprint.native_payload_bytes=add(p->footprint.native_payload_bytes,uint64_t(d.stride));
        }
        uint64_t static_need=0;
        for(size_t i=0;i<layout.units.size();++i){
            const auto& u=layout.units[i];
            require(u.tier==Tier::VRAM&&!u.movable&&u.slab_slot<0&&u.bytes>0,
                    RAD_E_UNSUPPORTED,"initial native profile has only static original residents");
            require(u.expert<0||coverage[i],RAD_E_INVAL,"every expert requires owned or explicit resident coverage");
            if(coverage[i]==2)continue;
            static_need=add(aligned(static_need),uint64_t(u.bytes));
            if(u.expert>=0)p->footprint.resident_expert_bytes=add(p->footprint.resident_expert_bytes,uint64_t(u.bytes));
        }
        p->footprint.original_static_bytes=static_need;
        std::vector<std::pair<int32_t,int32_t>> class_layers;
        class_layers.reserve(p->owned.size());
        for(const auto& owned_unit:p->owned)
            class_layers.push_back({owned_unit.klass,layout.units[size_t(owned_unit.unit)].layer});
        std::sort(class_layers.begin(),class_layers.end());
        std::vector<size_t> class_max(p->classes.size(),0);
        for(size_t i=0;i<class_layers.size();){
            size_t end=i+1;while(end<class_layers.size()&&class_layers[end]==class_layers[i])++end;
            class_max[size_t(class_layers[i].first)]=std::max(class_max[size_t(class_layers[i].first)],end-i);
            i=end;
        }
        for(size_t c=0;c<p->classes.size();++c){
            auto& cls=p->classes[c];size_t max_layer=0;
            max_layer=class_max[c];
            cls.slots=int32_t(std::min<uint64_t>(max_layer,mul(limits.max_rows,limits.top_k)));
            require(cls.slots>0,RAD_E_SHAPE,"empty native consumer class");
            cls.first_region=int32_t(p->footprint.regions);
            cls.region_bytes=int64_t(add(uint64_t(cls.stride),4096));
            cls.allocation_bytes=int64_t(add(uint64_t(cls.stride),16384));
            p->footprint.regions+=size_t(cls.slots)+1;
            p->footprint.consumer_slots+=size_t(cls.slots);
            const auto pool=mul(uint64_t(cls.slots)+1,uint64_t(cls.allocation_bytes));
            p->footprint.native_device_bytes=add(p->footprint.native_device_bytes,pool);
            p->footprint.landing_bytes=add(p->footprint.landing_bytes,pool);
            p->footprint.cold_union_bytes=add(p->footprint.cold_union_bytes,mul(uint64_t(cls.slots),uint64_t(cls.stride)));
        }
        require(p->footprint.regions<=limits.max_regions&&p->footprint.consumer_slots<=512&&
                p->footprint.native_device_bytes<=limits.max_device_bytes&&
                p->footprint.landing_bytes<=limits.max_landing_bytes&&
                p->footprint.cold_union_bytes<=limits.cold_bytes&&
                add(add(limits.hot_bytes,limits.cold_bytes),
                    add(p->footprint.landing_bytes,reservation))<=limits.max_host_bytes&&
                add(aligned(static_need),p->footprint.native_device_bytes)<=limits.max_weights_bytes,
                RAD_E_SCRATCH,"native reusable pool/cold-union admission cap");
        if(limits.backend_profile==NativeBackendProfile::TinyAqlV1){
            require(p->classes.size()==1&&p->classes[0].stride==32768&&
                    p->footprint.consumer_slots==3&&p->footprint.regions==4&&
                    p->footprint.native_device_bytes==196608&&p->footprint.landing_bytes==196608,
                    RAD_E_SHAPE,"tiny AQL exact reusable consumer footprint");
            const auto mx=rad_enc_affine(RAD_FP4E2M1,RAD_E8M0,1,32);
            for(const auto& owned:p->owned)for(const auto h:layout.units[size_t(owned.unit)].weights)
                require(rad_enc_equal(&program.weights[h].enc,&mx),RAD_E_DTYPE,"tiny AQL main MXFP4 only");
        }
        p->footprint.hot_bytes=limits.hot_bytes;p->footprint.cold_bytes=limits.cold_bytes;
        p->footprint.owned_units=p->owned.size();
        p->units=layout.units;
        p->unit_of_weight=layout.unit_of_weight;
        p->plan_classes=layout.classes;
        p->routes=program.expert_route_maps;
        for(auto& route:p->routes)route.decl.entries=nullptr;
        auto ready=std::shared_ptr<const NativeInventory>(new NativeInventory(std::move(p)));
        const int final_check=ready->validate(program,layout,why);
        if(final_check<0)return final_check;
        *out=std::move(ready);
        return RAD_OK;
    }catch(const Failure& f){explain(why,f.message);return f.status;}
    catch(const std::bad_alloc&){explain(why,"native snapshot allocation failed");return RAD_E_NOMEM;}
    catch(...){explain(why,"native snapshot exception");return RAD_E_STATE;}
}
int NativeInventory::validate(const Program& program,const Plan& layout,std::string* why)const{
    if(&program!=p_->program||program.ctx.world_size!=1||program.ctx.rank!=0||
       layout.rank!=0||layout.world_size!=1||layout.units.size()!=p_->units.size()||
       program.weights.size()!=p_->authority->size()||
       layout.classes.size()!=p_->plan_classes.size()||
       layout.unit_of_weight.size()!=p_->unit_of_weight.size()||
       layout.unit_of_weight!=p_->unit_of_weight){
        explain(why,"native exact Program/rank/layout/weight-unit-map identity");return RAD_E_INVAL;
    }
    for(size_t w=1;w<program.weights.size();++w)
        if(!p_->authority->matches(rad_weight(w),program.weights[w])||
           p_->authority->unit_for_weight(rad_weight(w))!=p_->unit_of_weight[w]){
            explain(why,"native Program metadata changed");return RAD_E_STATE;
        }
    if(program.expert_route_maps.size()!=p_->routes.size()){
        explain(why,"native route maps changed");return RAD_E_STATE;
    }
    for(size_t r=0;r<p_->routes.size();++r){
        const auto& a=program.expert_route_maps[r];const auto& b=p_->routes[r];
        const auto& x=a.decl;const auto& y=b.decl;
        if(x.layer!=y.layer||x.canonical_experts!=y.canonical_experts||
           x.local_experts!=y.local_experts||x.expert_base!=y.expert_base||
           x.regular_experts!=y.regular_experts||x.classes!=y.classes||
           x.max_rows!=y.max_rows||x.top_k!=y.top_k||x.gu!=y.gu||
           x.counts!=y.counts||x.offsets!=y.offsets||x.sorted!=y.sorted||
           x.ids!=y.ids||x.n_entries!=y.n_entries||a.entries.size()!=b.entries.size()){
            explain(why,"native route declaration changed");return RAD_E_STATE;
        }
        for(size_t e=0;e<a.entries.size();++e){
            const auto& u=a.entries[e];const auto& v=b.entries[e];
            if(u.canonical!=v.canonical||u.slot!=v.slot||u.class_index!=v.class_index||
               u.protected_expert!=v.protected_expert||!same(u.members,v.members)){
                explain(why,"native route identity changed");return RAD_E_STATE;
            }
        }
    }
    for(size_t u=0;u<layout.units.size();++u)
        if(!unit_metadata(layout.units[u],p_->units[u])){explain(why,"native unit layout changed");return RAD_E_STATE;}
    for(size_t c=0;c<layout.classes.size();++c)
        if(!class_metadata(layout.classes[c],p_->plan_classes[c])){
            explain(why,"native exact class table changed");return RAD_E_STATE;
        }
    for(const auto& unit:layout.units){
        if(unit.slab_class<0||size_t(unit.slab_class)>=layout.classes.size()||
           layout.classes[size_t(unit.slab_class)].stride!=unit.slot_bytes){
            explain(why,"native class/stride changed");return RAD_E_STATE;
        }
    }
    if(layout.native_inventory){
        if(layout.native_inventory.get()!=this||layout.strategy!=Strategy::NativeBacked||
           layout.dynamic||layout.weight.size()!=program.weights.size()){
            explain(why,"native plan capability changed");return RAD_E_STATE;
        }
        for(size_t u=0;u<layout.units.size();++u){
            const auto& unit=layout.units[u];const auto tier=find(int32_t(u))?Tier::NativeStore:Tier::VRAM;
            if(unit.tier!=tier||unit.site!=Site::Device||unit.movable||unit.slab_slot>=0||
               unit.host_forced){explain(why,"native placement changed");return RAD_E_STATE;}
            for(const auto w:unit.weights){
                const auto& placement=layout.weight[w];
                if(placement.unit!=int32_t(u)||placement.tier!=tier||
                   placement.site!=Site::Device||placement.slab_slot>=0||
                   placement.bytes!=p_->authority->stored_bytes(w)){
                    explain(why,"native member placement changed");return RAD_E_STATE;
                }
            }
        }
        for(const auto& cls:layout.classes)
            if(cls.n_slots||cls.n_shadow||cls.n_stage||cls.movable){
                explain(why,"competing native mover class storage");return RAD_E_STATE;
            }
    }
    return RAD_OK;
}
int NativeInventory::validate_runtime_header(const Program& program,const Plan& plan)const{
    if(&program!=p_->program||program.ctx.world_size!=1||program.ctx.rank!=0||
       program.weights.size()!=p_->authority->size()||plan.rank!=0||plan.world_size!=1||
       plan.units.size()!=p_->units.size()||plan.unit_of_weight.size()!=p_->unit_of_weight.size()||
       plan.classes.size()!=p_->plan_classes.size()||plan.native_inventory.get()!=this||
       plan.strategy!=Strategy::NativeBacked||plan.dynamic||
       plan.weight.size()!=program.weights.size()||plan.unit_of_weight[0]!=p_->unit_of_weight[0])
        return RAD_E_INVAL;
    return RAD_OK;
}
int32_t NativeInventory::unit_for_weight(rad_weight weight)const noexcept{
    return p_->authority->unit_for_weight(weight);
}
int NativeInventory::validate_unit(const Program& program,const Plan& plan,int32_t unit)const{
    const int header=validate_runtime_header(program,plan);
    if(header<0)return header;
    if(unit<0||size_t(unit)>=p_->units.size())return RAD_E_INVAL;
    const auto& expected=p_->units[size_t(unit)];const auto& current=plan.units[size_t(unit)];
    if(!unit_metadata(current,expected)||current.slab_class<0||
       size_t(current.slab_class)>=p_->plan_classes.size()||
       !class_metadata(plan.classes[size_t(current.slab_class)],p_->plan_classes[size_t(current.slab_class)]))
        return RAD_E_STATE;
    const auto tier=find(unit)?Tier::NativeStore:Tier::VRAM;
    if(current.tier!=tier||current.site!=Site::Device||current.movable||
       current.slab_slot>=0||current.host_forced)return RAD_E_STATE;
    for(size_t m=0;m<expected.weights.size();++m){const auto w=expected.weights[m];
        if(plan.unit_of_weight[w]!=p_->unit_of_weight[w]||
           p_->authority->unit_for_weight(w)!=p_->unit_of_weight[w]||
           p_->authority->member_index(w)!=int32_t(m)||
           !p_->authority->matches(w,program.weights[w]))return RAD_E_STATE;
        const auto& placement=plan.weight[w];
        if(placement.unit!=unit||placement.tier!=tier||placement.site!=Site::Device||
           placement.slab_slot>=0||placement.bytes!=p_->authority->stored_bytes(w))
            return RAD_E_STATE;
    }
    return RAD_OK;
}
int NativeInventory::configure_pools(const Config& base,Config* out)const{
    if(!out||base.host_pool_mib||base.weights_disk_tier||!base.deterministic||
       base.placement!="all_vram"||base.vram_kv_mib<=0||
       base.vram_kv_mib>INT64_MAX/(1<<20)||base.embedding_host_mib<0||
       base.tower_host_mib<0||uint64_t(base.tower_host_mib)>uint64_t(INT64_MAX)/(1<<20)||
       uint64_t(base.embedding_host_mib)>uint64_t(INT64_MAX)/(1<<20)-uint64_t(base.tower_host_mib))
        return RAD_E_INVAL;
    try{
        Config result=base;
        const auto n=p_->footprint.weights_minimum();
        result.vram_weights_mib=int64_t(std::max<uint64_t>(1,add(n,(1u<<20)-1)/(1u<<20)));
        if(result.vram_kv_mib>INT64_MAX/(1<<20)-result.vram_weights_mib)return RAD_E_SHAPE;
        result.host_pool_mib=0;*out=std::move(result);return RAD_OK;
    }catch(...){return RAD_E_NOMEM;}
}
} // namespace rad
