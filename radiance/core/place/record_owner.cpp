#include "record_owner.h"
#include "runtime/ctx.h"
#include <algorithm>
#include <cstring>
#include <limits>
namespace rad {
    namespace {
        uint64_t fnv(std::span<const std::byte> data){uint64_t h=14695981039346656037ull;
            for(auto b:data){h^=std::to_integer<uint8_t>(b);
                h*=1099511628211ull;
            }return h;
        }
        bool plus(uint64_t a,uint64_t b,uint64_t* out){if(b>UINT64_MAX-a)return false;
            *out=a+b;
            return true;
        }
        struct Buffer {
            void* allocation=nullptr;
            char* host=nullptr;
            char* device=nullptr;
            int64_t bytes=0,allocated=0,offset=0;
            int kind=0;
            bool safe=true;
            int init(int64_t n,int k,RadStream stream){
                if(n<=0||n>INT64_MAX-16384)return RAD_E_SHAPE;
                bytes=n;
                allocated=n+12288;
                kind=k;
                allocation=rad_dev_alloc(allocated,k);
                if(!allocation)return RAD_E_NOMEM;
                auto* dev=static_cast<char*>(rad_dev_device_ptr(allocation));
                if(!dev&&k==RAD_MEM_DEVICE)dev=static_cast<char*>(allocation);
                if(!dev||uintptr_t(dev)>UINTPTR_MAX-8191)return RAD_E_STATE;
                offset=int64_t(((uintptr_t(dev)+8191)&~uintptr_t(4095))-uintptr_t(dev));
                device=dev+offset;
                if(k!=RAD_MEM_DEVICE){auto* base=static_cast<char*>(rad_dev_host_ptr(allocation));
                    if(!base)return RAD_E_STATE;
                    host=base+offset;
                    if(uintptr_t(host)%4096)return RAD_E_ALIGN;
                    std::memset(base,0xbd,size_t(allocated));
                }
                else{safe=false;
                    const int result=rad_memset_async(allocation,0xbd,allocated,stream);
                    const int drained=rad_stream_sync(stream);
                    if(drained<0)return drained;
                    safe=true;
                    if(result<0)return result;
                }
                return RAD_OK;
            }
            bool guards_host()const{const auto* raw=kind==RAD_MEM_DEVICE?static_cast<const uint8_t*>(allocation):reinterpret_cast<const uint8_t*>(host)-offset;
                return raw&&std::all_of(raw,raw+offset,[](auto x){return x==0xbd;})&&std::all_of(raw+offset+bytes,raw+allocated,[](auto x){return x==0xbd;});
            }
            ~Buffer(){if(allocation&&safe)rad_dev_free(allocation,kind);
            }
        };
        struct Slot {Buffer landing,native;
            RadEvent ready=nullptr;
            int32_t exemplar=-1;
            bool sentinel=false,initialized=false;
            ~Slot(){if(ready)rad_event_destroy(ready);
        }};
        struct Binding {int32_t unit;
            uint32_t record;
            NativeRecordDescriptor descriptor;
            int32_t region;
        };
    }
    struct ExpertRecordOwner::Impl {
        Program* program;
        const Plan* plan;
        Mover* mover;
        std::shared_ptr<NativeRecordSource> source;
        RecordOwnerLimits limits;
        std::vector<Binding> bindings;
        std::vector<std::pair<rad_weight,WeightSlot>> fixed;
        std::vector<std::unique_ptr<Slot>> slots;
        std::vector<uint32_t> request;
        std::vector<RecordPublication> publication;
        std::vector<RecordClaim> claims;
        NativeRecordBatch held;
        RecordOwnerStats counters;
        uint64_t token=0;
        bool quiesced=true;
        int fatal=RAD_OK;
        RadStream transfer()const{return mover->transfer_stream();
        }
    };
    ExpertRecordOwner::ExpertRecordOwner(std::unique_ptr<Impl> p):impl_(std::move(p)){}
    ExpertRecordOwner::~ExpertRecordOwner(){
        if(!impl_)return;
        if(!impl_->quiesced){(void)impl_.release();
            return;
        } // quarantine pending buffers/leases/events
        if(impl_->token&&impl_->mover->record_session_close(impl_->token)<0){(void)impl_.release();
        }
    }
    int ExpertRecordOwner::open(Program& program,const Plan& plan,Mover& mover,std::shared_ptr<NativeRecordSource> source,std::span<const RecordOwnerBinding> bindings,const RecordOwnerLimits& limits,std::unique_ptr<ExpertRecordOwner>* out){
        if(!mover.record_plan_matches(program,plan))return RAD_E_INVAL;
        if(!out||*out||!source||bindings.empty()||bindings.size()>limits.max_units||limits.max_units>512||limits.max_regions>1024||limits.max_regions<2||!limits.max_landing_bytes||limits.max_landing_bytes>(256u<<20)||!limits.max_device_bytes||limits.max_device_bytes>(256u<<20)||!limits.max_metadata_bytes||limits.max_metadata_bytes>(64u<<20)||program.weights.size()>262144||program.ctx.world_size!=1||program.ctx.rank!=0||plan.dynamic||plan.world_size!=1||plan.rank!=0||!mover.transfer_stream())return RAD_E_UNSUPPORTED;
        try{
            auto p=std::make_unique<Impl>();
            p->program=&program;
            p->plan=&plan;
            p->mover=&mover;
            p->source=std::move(source);
            p->limits=limits;
            const auto catalog=p->source->catalog();
            if(catalog.empty()||catalog.size()>65536)return RAD_E_SHAPE;
            std::vector<int32_t> exemplars;
            for(size_t i=0;i<bindings.size();++i){const auto b=bindings[i];
                if(b.unit<0||size_t(b.unit)>=plan.units.size()||b.record>=catalog.size())return RAD_E_INVAL;
                for(size_t j=0;j<i;++j)if(bindings[j].unit==b.unit||bindings[j].record==b.record)return RAD_E_INVAL;
                const auto& mu=plan.units[size_t(b.unit)];
                const auto& d=catalog[b.record];
                if(d.world!=1||d.rank||d.layer!=mu.layer||d.expert!=mu.expert||!d.source_generation||d.stride<=0||mu.slot_bytes<=0||mu.bytes<=0||d.stride!=mu.slot_bytes||mu.weights.size()!=4||mu.offset.size()!=4||mu.expert<0||mu.movable)return RAD_E_SHAPE;
                for(size_t m=0;m<4;++m){if(!mu.weights[m]||mu.weights[m]>=program.weights.size())return RAD_E_INVAL;
                    const auto& w=program.weights[mu.weights[m]];
                    if(!w.identity||w.widen||!w.layout_tag.empty()||w.n_sel!=1||w.sel[0]!=int(m%2)||w.stored_rank!=2||w.stored_dtype!=w.enc.plane[w.sel[0]].dtype||d.offsets[m]!=mu.offset[m]||d.bytes[m]!=w.stored_bytes||d.dtype[m]!=w.stored_dtype||d.shape[m][0]!=w.stored_shape[0]||d.shape[m][1]!=w.stored_shape[1])return RAD_E_SHAPE;
                    const auto main=rad_enc_affine(RAD_FP4E2M1,RAD_E8M0,1,32),mtp=rad_enc_affine(RAD_F8E4M3,RAD_F32,128,128);
                    if(!rad_enc_equal(&w.enc,&main)&&!rad_enc_equal(&w.enc,&mtp))return RAD_E_DTYPE;
                }
                const auto routing=std::find_if(program.expert_route_maps.begin(),program.expert_route_maps.end(),[&](const auto& map){return map.decl.layer==mu.layer;});
                if(routing==program.expert_route_maps.end()||size_t(mu.expert)>=routing->entries.size())return RAD_E_STATE;
                const auto& entry=routing->entries[size_t(mu.expert)];
                if(entry.protected_expert)return RAD_E_UNSUPPORTED;
                for(size_t m=0;m<4;++m)if(mu.weights[m]!=entry.members[m])return RAD_E_STATE;
                if(std::none_of(exemplars.begin(),exemplars.end(),[&](auto u){return plan.units[size_t(u)].slab_class==mu.slab_class;}))exemplars.push_back(b.unit);
                p->bindings.push_back({b.unit,b.record,d,-1});
            }
            // One immutable class sentinel, one full native region per maximum simultaneous bound
            // unit. These worst-case reservations happen BEFORE admission/copies/publication.
            if(exemplars.size()+bindings.size()>limits.max_regions)return RAD_E_SHAPE;
            uint64_t landing=0,device=0;
            for(const auto unit:exemplars){const auto n=plan.units[size_t(unit)].slot_bytes;
                if(n>INT64_MAX-16384||!plus(landing,uint64_t(n+16384),&landing)||!plus(device,uint64_t(n+16384),&device))return RAD_E_SHAPE;
            }
            for(const auto& b:p->bindings){const auto n=b.descriptor.stride;
                if(n>INT64_MAX-16384||!plus(landing,uint64_t(n+16384),&landing)||!plus(device,uint64_t(n+16384),&device))return RAD_E_SHAPE;
            }
            const uint64_t metadata=uint64_t(bindings.size())*(sizeof(Binding)+sizeof(RecordPublication)+sizeof(RecordClaim)+sizeof(uint32_t))+uint64_t(exemplars.size()+bindings.size())*(sizeof(Slot)+sizeof(RecordRegion))+uint64_t(program.weights.size())*sizeof(std::pair<rad_weight,WeightSlot>);
            if(landing>limits.max_landing_bytes||device>limits.max_device_bytes||metadata>limits.max_metadata_bytes)return RAD_E_SCRATCH;
            p->counters.landing_bytes=landing;
            p->counters.device_bytes=device;
            p->counters.metadata_bytes=metadata;
            p->counters.original_mover_base_bytes=mover.record_base_allocation_bytes();
            for(const auto& b:p->bindings)p->counters.retained_cache_unit_bytes+=uint64_t(plan.units[size_t(b.unit)].bytes);
            p->request.reserve(bindings.size());
            p->publication.reserve(bindings.size());
            p->claims.reserve(bindings.size());
            p->held.leases.reserve(bindings.size());
            p->slots.reserve(exemplars.size()+bindings.size());
            std::vector<RecordRegion> regions;
            regions.reserve(exemplars.size()+bindings.size());
            auto allocate=[&](int32_t unit,bool sentinel)->int{
                auto slot=std::make_unique<Slot>();
                slot->exemplar=unit;
                slot->sentinel=sentinel;
                const auto& mu=plan.units[size_t(unit)];
                const auto bytes=mu.slot_bytes+4096;
                int status=slot->landing.init(bytes,RAD_MEM_HOST_MAPPED,p->transfer());
                if(status<0)return status;
                status=slot->native.init(bytes,RAD_MEM_DEVICE,p->transfer());
                if(status<0)return status;
                std::memset(slot->landing.host,0,size_t(bytes));
                if(sentinel){for(size_t m=0;m<4;++m){if(!mu.weights[m]||mu.weights[m]>=program.weights.size())return RAD_E_INVAL;
                        const auto& w=program.weights[mu.weights[m]];
                        if(w.stored_dtype==RAD_E8M0)std::memset(slot->landing.host+mu.offset[m],127,size_t(w.stored_bytes));
                        else if(w.stored_dtype==RAD_F32){auto* q=reinterpret_cast<float*>(slot->landing.host+mu.offset[m]);
                            for(int64_t j=0;j<w.stored_bytes/4;++j)q[j]=1.f;
                    }}
                    rad_dev_host_wrote();
                    status=rad_memcpy_async(slot->native.device,slot->landing.device,bytes,p->transfer());
                    const int drained=rad_stream_sync(p->transfer());
                    if(drained<0){(void)slot.release();
                        return drained;
                    }if(status<0)return status;
                    slot->initialized=true;
                }else{status=rad_event_create(&slot->ready);
                    if(status<0)return status;
                }
                regions.push_back({unit,slot->native.device,bytes,slot->ready,sentinel});
                p->slots.push_back(std::move(slot));
                return RAD_OK;
            };
            for(const auto unit:exemplars){const int status=allocate(unit,true);
                if(status<0)return status;
            }
            for(auto& b:p->bindings){const int status=allocate(b.unit,false);
                if(status<0)return status;
                b.region=int32_t(regions.size()-1);
                const auto& mu=plan.units[size_t(b.unit)];
                const auto exemplar=std::find_if(exemplars.begin(),exemplars.end(),[&](auto u){return plan.units[size_t(u)].slab_class==mu.slab_class;});
                p->claims.push_back({b.unit,int32_t(exemplar-exemplars.begin())});
            }
            p->fixed.reserve(program.weights.size());
            for(size_t h=1;h<program.weights.size();++h){const bool owned=std::any_of(p->bindings.begin(),p->bindings.end(),[&](const auto& b){const auto& mu=plan.units[size_t(b.unit)];return std::find(mu.weights.begin(),mu.weights.end(),rad_weight(h))!=mu.weights.end();});
                if(!owned){const auto* slot=mover.residency()->lookup(rad_weight(h));
                    if(!slot||!slot->ptr||slot->ready)return RAD_E_STATE;
                    p->fixed.emplace_back(rad_weight(h),*slot);
            }}
            auto result=std::unique_ptr<ExpertRecordOwner>(new ExpertRecordOwner(std::move(p)));
            const int opened=mover.record_session_open(result->impl_->claims,regions,&result->impl_->token);
            if(opened<0)return opened;
            *out=std::move(result);
            return RAD_OK;
        }catch(const std::bad_alloc&){return RAD_E_NOMEM;
        }catch(...){return RAD_E_STATE;
        }
    }
    int ExpertRecordOwner::quiesce(){
        auto& p=*impl_;
        if(!p.token)return RAD_E_STATE;
        const int status=p.mover->record_restore(p.token);
        if(status<0){p.quiesced=false;
            if(p.fatal>=0)p.fatal=status;
            return status;
        }p.held.leases.clear();
        p.counters.held_leases=0;
        p.quiesced=true;
        ++p.counters.retirements;
        return RAD_OK;
    }
    int ExpertRecordOwner::end_pass(RadStream,int status){const int result=quiesce();
        if(status<0&&impl_->fatal>=0)impl_->fatal=status;
        return result;
    }
    int ExpertRecordOwner::check_program(const Program& program)const{return &program==impl_->program?RAD_OK:RAD_E_INVAL;}
    int ExpertRecordOwner::prepare(const ExpertPrepareView& view,bool* changed){
        auto& p=*impl_;
        if(view.program!=p.program||!view.context||view.context->program()!=p.program||view.context->cur_stream()!=view.producer_stream)return RAD_E_INVAL;
        if(!changed)return RAD_E_INVAL;
        *changed=false;
        if(p.fatal<0)return p.fatal;
        ++p.counters.prepares;
        const auto previous_epoch=p.mover->residency()->epoch();
        int status=quiesce();
        if(status<0)return status;
        *changed=p.mover->residency()->epoch()!=previous_epoch;
        try{
            p.request.clear();
            p.publication.clear();
            for(const auto& entry:view.map){if(view.counts[size_t(entry.slot)]<=0)continue;
                const auto found=std::find_if(p.bindings.begin(),p.bindings.end(),[&](const auto& b){const auto& mu=p.plan->units[size_t(b.unit)];return mu.layer==view.declaration.layer&&mu.expert==entry.slot;});
                if(found==p.bindings.end()){
                    for(const auto h:entry.members){if(!h)continue;
                        const auto fixed=std::lower_bound(p.fixed.begin(),p.fixed.end(),h,[](const auto& a,rad_weight b){return a.first<b;});
                        const auto* slot=p.mover->residency()->lookup(h);
                        if(fixed==p.fixed.end()||fixed->first!=h||!slot||slot->ptr!=fixed->second.ptr||slot->ready!=fixed->second.ready||slot->generation!=fixed->second.generation)return RAD_E_STATE;
                    }
                    continue;
                }const auto& mu=p.plan->units[size_t(found->unit)];
                for(size_t m=0;m<4;++m)if(mu.weights[m]!=entry.members[m])return RAD_E_STATE;
                p.request.push_back(found->record);
                p.publication.push_back({found->unit,found->region});
            }
            if(p.request.empty())return RAD_OK;
            // Catalogue is immutable for the source lifetime; detect ordinary version drift before I/O.
            const auto catalog=p.source->catalog();
            for(const auto id:p.request){const auto binding=std::find_if(p.bindings.begin(),p.bindings.end(),[&](const auto& b){return b.record==id;});
                if(id>=catalog.size()||binding==p.bindings.end()||catalog[id]!=binding->descriptor)return RAD_E_STATE;
            }
            NativeRecordBatch ready;
            ++p.counters.source_batches;
            status=p.source->acquire(p.request,&ready);
            if(status<0)return status;
            if(ready.leases.size()!=p.request.size())return RAD_E_STATE;
            for(size_t i=0;i<ready.leases.size();++i){const auto& lease=ready.leases[i];
                const auto& publication=p.publication[i];
                const auto binding=std::find_if(p.bindings.begin(),p.bindings.end(),[&](const auto& b){return b.unit==publication.unit;});
                if(lease.record!=p.request[i]||!lease.lifetime||lease.bytes.size()!=uint64_t(binding->descriptor.stride)||fnv(lease.bytes)!=binding->descriptor.checksum)return RAD_E_INVAL;
            }
            p.held=std::move(ready);
            p.counters.held_leases=p.held.leases.size();
            p.quiesced=false;
            for(size_t i=0;i<p.held.leases.size();++i){auto& slot=*p.slots[size_t(p.publication[i].region)];
                const auto& lease=p.held.leases[i];
                std::memcpy(slot.landing.host,lease.bytes.data(),lease.bytes.size());
                p.counters.source_bytes+=lease.bytes.size();
                rad_dev_host_wrote();
                status=rad_memcpy_async(slot.native.device,slot.landing.device,slot.native.bytes,p.transfer());
                if(status<0)return status;
                status=rad_event_record(slot.ready,p.transfer());
                if(status<0)return status;
                slot.initialized=true;
                p.counters.copy_bytes+=uint64_t(slot.native.bytes);
            }
            status=p.mover->record_publish(p.token,p.publication);
            if(status<0)return status;
            p.counters.published_units+=p.publication.size();
            *changed=true;
            return RAD_OK;
        }catch(const std::bad_alloc&){return RAD_E_NOMEM;
        }catch(...){return RAD_E_STATE;
        }
    }
    int ExpertRecordOwner::check_guards_host()const{if(!device_is_host())return RAD_E_UNSUPPORTED;
        for(const auto& slot:impl_->slots)if(!slot->landing.guards_host()||!slot->native.guards_host()||(impl_->fatal>=0&&slot->initialized&&std::memcmp(slot->landing.host,slot->native.device,size_t(slot->native.bytes))!=0))return RAD_E_STATE;
        return RAD_OK;
    }
    RecordOwnerStats ExpertRecordOwner::stats()const noexcept{return impl_->counters;
    }
} // namespace rad
