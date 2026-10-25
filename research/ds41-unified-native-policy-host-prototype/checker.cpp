// Actual HOST MoeFP8 current handoff -> S1Reader/cache -> Mover Residency -> native libref MXFP4.
#include "place/native_backing_owner.h"
#include "mem/pools.h"
#include "server/liveview.h"
#include "build/rad_build.h"
#include "plugin/registry.h"
#include "runtime/ctx.h"
#include "rad_block_moe_fp8.h"
#include "record_cache_bridge.h"
#include "rad_plugin.h"
#include "rad_model_failure_policy.h"
#include <new>
#include <string>
#include <dlfcn.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
// Test-only rank-thread allocation fault. Replace the COMPLETE allocation
// family consistently: otherwise std::stable_sort's nothrow new can allocate
// through ASan's C++ allocator while our sized delete calls free.
namespace allocation_fault {
thread_local bool next=false;thread_local size_t failures=0;
void check(){if(next){next=false;++failures;throw std::bad_alloc();}}
void* allocate(std::size_t n){check();if(void* p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void* allocate_aligned(std::size_t n,std::align_val_t requested){
    check();const auto alignment=std::max(std::size_t(requested),alignof(void*));
    void* p=nullptr;
    if((alignment&(alignment-1))==0&&posix_memalign(&p,alignment,n?n:1)==0)return p;
    throw std::bad_alloc();
}
}
void* operator new(std::size_t n){return allocation_fault::allocate(n);}
void* operator new[](std::size_t n){return allocation_fault::allocate(n);}
void* operator new(std::size_t n,const std::nothrow_t&) noexcept{try{return allocation_fault::allocate(n);}catch(...){return nullptr;}}
void* operator new[](std::size_t n,const std::nothrow_t&) noexcept{try{return allocation_fault::allocate(n);}catch(...){return nullptr;}}
void* operator new(std::size_t n,std::align_val_t a){return allocation_fault::allocate_aligned(n,a);}
void* operator new[](std::size_t n,std::align_val_t a){return allocation_fault::allocate_aligned(n,a);}
void* operator new(std::size_t n,std::align_val_t a,const std::nothrow_t&) noexcept{try{return allocation_fault::allocate_aligned(n,a);}catch(...){return nullptr;}}
void* operator new[](std::size_t n,std::align_val_t a,const std::nothrow_t&) noexcept{try{return allocation_fault::allocate_aligned(n,a);}catch(...){return nullptr;}}
void operator delete(void* p) noexcept{std::free(p);}
void operator delete[](void* p) noexcept{std::free(p);}
void operator delete(void* p,std::size_t) noexcept{std::free(p);}
void operator delete[](void* p,std::size_t) noexcept{std::free(p);}
void operator delete(void* p,const std::nothrow_t&) noexcept{std::free(p);}
void operator delete[](void* p,const std::nothrow_t&) noexcept{std::free(p);}
void operator delete(void* p,std::align_val_t) noexcept{std::free(p);}
void operator delete[](void* p,std::align_val_t) noexcept{std::free(p);}
void operator delete(void* p,std::size_t,std::align_val_t) noexcept{std::free(p);}
void operator delete[](void* p,std::size_t,std::align_val_t) noexcept{std::free(p);}
void operator delete(void* p,std::align_val_t,const std::nothrow_t&) noexcept{std::free(p);}
void operator delete[](void* p,std::align_val_t,const std::nothrow_t&) noexcept{std::free(p);}

using namespace rad;
using namespace rad::arch;
namespace copy_fault {inline RadStream stream=nullptr;
    inline int remaining=-1;
    inline size_t failed=0;
}
namespace event_fault {inline RadStream stream=nullptr;inline int remaining=-1;inline size_t failed=0;}
extern "C" int __real_rad_event_record(RadEvent,RadStream);
extern "C" int __wrap_rad_event_record(RadEvent event,RadStream stream){
    if(stream==event_fault::stream&&event_fault::remaining>=0&&event_fault::remaining--==0){
        event_fault::remaining=-1;++event_fault::failed;return RAD_E_IO;
    }
    return __real_rad_event_record(event,stream);
}
extern "C" int __real_rad_memcpy_async(void*,const void*,int64_t,RadStream);
extern "C" int __wrap_rad_memcpy_async(void* dst,const void* src,int64_t bytes,RadStream stream){
    if(stream==copy_fault::stream&&bytes==32768+4096&&copy_fault::remaining>=0&&copy_fault::remaining--==0){copy_fault::remaining=-1;
        ++copy_fault::failed;
        return RAD_E_IO;
    }
    return __real_rad_memcpy_async(dst,src,bytes,stream);
}

namespace es=rad::expert_store;
namespace br=glm53::record_bridge;
namespace ec=glm53::cache;
namespace fs=std::filesystem;
namespace {
    constexpr int64_t H=128,E=4,K=2,T=5,R=2;
    size_t cases=0,checks=0;
    void need(bool v,const char* why){++checks;
        if(!v)throw std::runtime_error(why);
    }
    void good(int s,const char* why){
        const std::string message=std::string(why)+": "+rad_strerror(s)+" ("+std::to_string(s)+")";
        need(s==RAD_OK,message.c_str());
    }
    void good(es::Status s,const std::string& why){
        const std::string message=why+": "+es::name(s)+" ("+std::to_string(static_cast<int>(s))+")";
        need(s==es::Status::ok,message.c_str());
    }
    void allocation_hook_contract(){
        // Direct calls prevent allocation elision and cover the formerly
        // mismatched nothrow-new -> sized-delete pair before loading plugins.
        auto touch=[](void* p){need(p!=nullptr,"allocation-family contract returned null");std::memset(p,0xA5,8);};
        void* p=::operator new(8);touch(p);::operator delete(p,std::size_t(8));
        p=::operator new[](8);touch(p);::operator delete[](p,std::size_t(8));
        p=::operator new(8,std::nothrow);touch(p);::operator delete(p,std::size_t(8));
        p=::operator new[](8,std::nothrow);touch(p);::operator delete[](p,std::nothrow);
        constexpr auto alignment=std::align_val_t(64);
        auto aligned=[&](void* value){touch(value);need(uintptr_t(value)%64==0,"aligned allocation family lost alignment");};
        p=::operator new(8,alignment);aligned(p);::operator delete(p,std::size_t(8),alignment);
        p=::operator new[](8,alignment);aligned(p);::operator delete[](p,alignment);
        p=::operator new(8,alignment,std::nothrow);aligned(p);::operator delete(p,alignment,std::nothrow);
        p=::operator new[](8,alignment,std::nothrow);aligned(p);::operator delete[](p,std::size_t(8),alignment);
        const auto before=allocation_fault::failures;
        allocation_fault::next=true;p=::operator new(8,std::nothrow);
        need(!p&&!allocation_fault::next&&allocation_fault::failures==before+1,"scalar nothrow allocation did not contain injected throw");
        allocation_fault::next=true;p=::operator new[](8,std::nothrow);
        need(!p&&!allocation_fault::next&&allocation_fault::failures==before+2,"array nothrow allocation did not contain injected throw");
        allocation_fault::next=true;p=::operator new(8,alignment,std::nothrow);
        need(!p&&!allocation_fault::next&&allocation_fault::failures==before+3,"aligned scalar nothrow allocation did not contain injected throw");
        allocation_fault::next=true;p=::operator new[](8,alignment,std::nothrow);
        need(!p&&!allocation_fault::next&&allocation_fault::failures==before+4,"aligned array nothrow allocation did not contain injected throw");
    }
    struct Temp {fs::path path;
        Temp(){char n[]="/tmp/glm53-owner-b1-XXXXXX";
            const auto* p=::mkdtemp(n);
            need(p,"tiny temp directory");
            path=p;
        }~Temp(){std::error_code e;
            fs::remove_all(path,e);
    }};
    void write(const fs::path& path,std::span<const std::byte> bytes){std::ofstream out(path,std::ios::binary|std::ios::trunc);
        need(bool(out),"synthetic source file open");
        out.write(reinterpret_cast<const char*>(bytes.data()),std::streamsize(bytes.size()));
        need(bool(out),"synthetic file write");
    }
    struct Stream {RadStream value=nullptr;
        Stream(){good(rad_stream_create(&value,0),"actual HOST compute stream");
        }~Stream(){if(value){if(rad_stream_sync(value)<0)std::abort();
                rad_stream_destroy(value);
    }}};
    struct Gate {
        void* value=nullptr;
        explicit Gate(RadStream stream){good(rad_stream_gate(stream,&value),"actual HOST transfer gate");}
        void open(){if(value){void* old=value;value=nullptr;good(rad_gate_open(old),"open actual HOST gate");}}
        ~Gate(){if(value&&rad_gate_open(value)<0)std::abort();}
    };
    struct Source final:NativeRecordSource {
        std::vector<es::RecordSpec> specs;
        es::SourceCatalog sources;
        es::Receipt receipt;
        std::unique_ptr<br::RecordCache> bridge;
        std::vector<NativeRecordDescriptor> descriptors;
        std::vector<ec::RecordRef> refs;
        std::vector<uint32_t> last_request;
        bool fail_acquire_and_next_allocation=false;
        Source(const fs::path& store,std::vector<es::RecordSpec> s,es::SourceCatalog c,es::Receipt r,uint64_t cold):specs(std::move(s)),sources(std::move(c)),receipt(r){
            br::Limits limits;
            limits.world=1;
            limits.rank=0;
            limits.cache.hot_bytes=2*32768;
            limits.cache.cold_bytes=cold;
            need(br::RecordCache::open(store.string(),{specs,sources,receipt.manifest_checksum},limits,&bridge)==ec::Error::ok,"new TP1 S1/cache bridge");
            refs.assign(bridge->records().begin(),bridge->records().end());
            for(const auto& ref:refs){const auto it=std::find_if(specs.begin(),specs.end(),[&](const auto& x){const auto k=x.key();return k.layer==ref.key.layer&&k.expert==ref.key.expert&&k.rank==ref.key.rank&&k.world==ref.key.tp;});
                need(it!=specs.end(),"bound native record spec");
                NativeRecordDescriptor d;
                d.source_generation=receipt.manifest_checksum;
                d.checksum=ref.checksum;
                d.world=ref.key.tp;
                d.rank=ref.key.rank;
                d.layer=ref.key.layer;
                d.expert=ref.key.expert;
                d.stride=int64_t(ref.length);
                const auto offsets=it->member_offsets(),bytes=it->member_bytes();
                const auto metadata=nlohmann::json::parse(it->metadata_json());
                for(size_t m=0;m<4;++m){d.offsets[m]=int64_t(offsets[m]);
                    d.bytes[m]=int64_t(bytes[m]);
                    d.dtype[m]=metadata["members"][m]["stored_dtype"].get<uint32_t>();
                    d.shape[m]={metadata["members"][m]["stored_shape"][0].get<int64_t>(),metadata["members"][m]["stored_shape"][1].get<int64_t>()};
                }descriptors.push_back(d);
            }
            last_request.reserve(E);
        }
        std::span<const NativeRecordDescriptor> catalog()const noexcept override{return descriptors;
        }
        int acquire(std::span<const uint32_t> ids,NativeRecordBatch* out)override{
            if(!out||ids.empty()||ids.size()>E)return RAD_E_INVAL;
            if(fail_acquire_and_next_allocation){fail_acquire_and_next_allocation=false;
                allocation_fault::next=true;return RAD_E_NOMEM;}
            std::vector<ec::RecordRef> request;
            request.reserve(ids.size());
            for(const auto id:ids){if(id>=refs.size())return RAD_E_INVAL;
                request.push_back(refs[id]);
            }
            auto keep=std::make_shared<ec::Batch>();
            const auto status=bridge->acquire(request,{specs,sources,receipt.manifest_checksum},keep.get());
            if(status!=ec::Error::ok)return status==ec::Error::budget?RAD_E_SCRATCH:RAD_E_IO;
            NativeRecordBatch ready;
            ready.leases.reserve(keep->leases.size());
            if(keep->leases.size()!=ids.size())return RAD_E_STATE;
            for(size_t i=0;i<ids.size();++i)ready.leases.push_back({ids[i],keep->leases[i].bytes(),keep});
            last_request.assign(ids.begin(),ids.end());
            out->leases.swap(ready.leases);
            return RAD_OK;
        }
    };
    struct CountingAllocator final:MemAllocator {
        struct Request{void* ptr;int64_t bytes,physical;int kind;};
        std::vector<Request> live;uint64_t requested=0,physical=0,calls=0;
        int64_t fail_at=-1;
        void* alloc(int64_t bytes,int kind)override{
            ++calls;if(int64_t(calls)==fail_at)return nullptr;
            void* p=rad_malloc_allocator().alloc(bytes,kind);
            if(p){const int64_t n=align_up(bytes,int64_t(RAD_ALIGN_POOL));
                try{live.push_back({p,bytes,n,kind});}
                catch(...){rad_malloc_allocator().free(p,kind);return nullptr;}
                requested+=uint64_t(bytes);physical+=uint64_t(n);}
            return p;
        }
        void free(void* p,int kind)override{
            const auto i=std::find_if(live.begin(),live.end(),[&](const auto& x){return x.ptr==p;});
            if(i==live.end())std::abort();
            live.erase(i);rad_malloc_allocator().free(p,kind);
        }
        const char* name()const override{return "counted actual HOST pool allocator";}
    };
    using PayloadTotals=std::array<std::pair<int64_t,int64_t>,4>;
    PayloadTotals payload_totals(){
        PayloadTotals result{};
        const int kinds[]{RAD_MEM_DEVICE,RAD_MEM_HOST_PINNED,RAD_MEM_HOST,RAD_MEM_HOST_MAPPED};
        for(size_t i=0;i<result.size();++i)
            rad_dev_alloc_totals(kinds[i],&result[i].first,&result[i].second);
        return result;
    }
    struct Fixture;
    thread_local Fixture* active=nullptr;
    void step(RadCtx*,const RadBatch*);
    struct Fixture {
        Temp temp;
        Stream compute;
        Registry registry;
        Names names;
        std::vector<MoeFP8> moes;int active_layer=0;
        std::unique_ptr<Builder> builder;
        Plan metadata_plan,plan;
        std::shared_ptr<const NativeInventory> inventory;
        Config pool_config;
        CountingAllocator device_allocator,host_allocator;
        Pools pools;
        Mover mover;
        PayloadTotals before_mover{},after_mover{};
        int64_t vmem_before_pool=0,vmem_after_pool=0;
        std::vector<std::vector<std::byte>> initial;
        es::SourceCatalog source_catalog;
        std::vector<es::RecordSpec> specs;
        es::Receipt receipt;
        std::shared_ptr<Source> source;
        std::unique_ptr<NativeBackingOwner> owner;
        uint64_t original_expert_reads=0;
        std::array<uint16_t,T*H> caller{};
        rad_buf bad_dn=0;
        bool fail_dn=false;
        int mode=0,step_mode=0,late_prepare_status=RAD_OK;size_t step_entries=0;
        Ctx ctx;
        Fixture(const std::string& home,bool owned,uint64_t cold=3*32768,int layers=1,
                bool initialize_owner_context=true,bool permanent=true,bool collector=false):names(""){
            need(device_is_host(),"HOST-only, no device execution permitted");
            good(registry.load_plugin(home+"/kernels/libref.so",0),"actual matching HOST libref");
            if(collector)good(registry.load_plugin(UNIFIED_POLICY_COLLECTOR_MODULE,0),"actual admission-only V1 DEVICE collector module");
            RadModelMeta meta{};
            meta.arch_id="synthetic-record-owner";
            meta.name="not-a-model";
            need(layers>0&&layers<=8,"bounded actual S1 fixture layers");
            meta.n_layers=layers;
            meta.n_embd=H;
            meta.n_expert=E;
            RadBuildCtx bc{};
            bc.rank=0;
            bc.world_size=1;
            bc.max_tok=T;
            bc.max_seqs=1;
            bc.max_ctx=T;
            bc.scope="";
            builder=std::make_unique<Builder>(registry,meta,bc);
            if(permanent)good(builder->require_permanent_model_failure(),"real immutable Program policy BEFORE declaration");
            builder->set_instances(false);
            builder->set_sources([](const std::string& name,const Program&,WeightSource* out){const bool gu=name.find("ffn_gate_up_exps.")!=std::string::npos,dn=name.find("ffn_down_exps.")!=std::string::npos;if(!gu&&!dn)return RAD_E_NOTFOUND;*out={};out->enc=rad_enc_affine(RAD_FP4E2M1,RAD_E8M0,1,32);out->rank=2;out->shape[0]=gu?2*H:H;out->shape[1]=H;return RAD_OK;});
            moes.resize(size_t(layers));
            for(int l=0;l<layers;++l){
            active_layer=l;
            auto& moe=moes[size_t(l)];
            Geom g{};
            g.n_layer=layers;
            g.n_embd=H;
            g.n_ff=g.n_ff_all=H;
            g.max_tok=T;
            g.max_seqs=1;
            g.max_ctx=T;
            g.dtype="bf16";
            g.act_dtype=RAD_BF16;
            auto buf=[&](const char* name,uint32_t dtype,std::initializer_list<int64_t> shape){const std::string label="layer."+std::to_string(l)+"."+name;return decl_b(builder.get(),label.c_str(),dtype,shape,RAD_BUF_PERSIST);
            };
            MoeFP8::Wire w{};
            w.h.x=buf("input",RAD_BF16,{T,H});
            good(w.h.declare_qs(builder.get(),names,g,names.f("layer.%d.input",l),H),"input FP8 pair");
            w.logits=buf("logits",RAD_F32,{T,E});
            w.ids=buf("ids",RAD_I32,{T,K});
            w.ew=buf("route.weights",RAD_F32,{T,K});
            w.ecnt=buf("count",RAD_I32,{E});
            w.eoff=buf("offsets",RAD_I32,{E+1});
            w.sorted=buf("sort",RAD_I32,{R*K});
            w.egu=buf("GU",RAD_BF16,{R*K,2*H});
            w.eff.x=buf("product",RAD_BF16,{R*K,H});
            good(w.eff.declare_qs(builder.get(),names,g,names.f("layer.%d.ff",l),H,R*K),"down FP8 pair");
            w.edn=buf("DN",RAD_BF16,{R*K,H});
            bad_dn=buf("invalid.late.DN",RAD_BF16,{1,H});
            MoeFP8::Src s{};
            s.experts="synthetic.experts";
            s.router="synthetic.router";
            s.router_bias="synthetic.correction";
            s.cfg.n_expert=s.cfg.n_expert_all=E;
            s.cfg.n_ff_exp=H;
            s.cfg.top_k=K;
            s.cfg.rows=R;
            s.cfg.expert_mx=1;
            s.cfg.sigmoid_router=1;
            s.cfg.route_scale=2.5;
            s.cfg.swiglu_limit=10;
            s.cfg.route_prepare=1;
            for(int q=0;q<2;++q){s.cfg.ff_lo[q]=0;
                s.cfg.ff_hi[q]=H;
            }
            moe.ext_in=true;
            good(moe.declare(builder.get(),names,g,l,w,s,false,false),"actual TP1 MXFP4 MoeFP8");
            }
            active_layer=0;
            if(collector)need(builder->decl_op("unified_status_fixture",nullptr,0,nullptr,0)!=0,
                              "actual V1 DEVICE collector op selection");
            good(builder->finish(),"actual HOST graph finish");
            auto& program=builder->program();
            Config config;
            config.placement="all_vram";
            config.deterministic=true;
            PlannerInput pi{&program,&config,0,1,nullptr,0,false,true};
            // These fresh synthetic planes are checkpoint-style sources, not a mapped .rad
            // container. On the actual HOST backend the native destination is host memory.
            // Pretending a DEVICE backend/container would pin these HOST-only reader units
            // to Tier::Mapped, deliberately leaving them without a native class/slot stride.
            good(Planner(pi).plan(&metadata_plan),"real static pure Planner");
            plan=metadata_plan;
            build_source();
            good(es::pack(specs,source_catalog,(temp.path/"store").string(),es::Limits{},&receipt),"S1 real TP1 native pack");
            source=std::make_shared<Source>(temp.path/"store",specs,source_catalog,receipt,cold);
            Config base=config;base.vram_kv_mib=1;base.embedding_placement=TablePlacement::Vram;
            vmem_before_pool=rad_dev_vmem_committed();
            if(owned){
                const auto bindings=bound(*source);std::vector<int32_t> residents;
                for(size_t u=0;u<metadata_plan.units.size();++u)
                    if(metadata_plan.units[u].expert==0)residents.push_back(int32_t(u));
                NativeInventoryLimits limits;limits.max_rows=R;limits.top_k=K;
                limits.hot_bytes=2*32768;limits.cold_bytes=cold;
                std::string why;
                const int status=NativeInventory::bind(program,metadata_plan,source->catalog(),bindings,
                                                       residents,limits,&inventory,&why);
                if(status<0)throw std::runtime_error("native inventory: "+why);
                good(inventory->configure_pools(base,&pool_config),"actual reduced native footprint config");
                pi.config=&pool_config;pi.native_inventory=inventory;
                good(Planner(pi).plan(&plan),"actual typed native-aware Planner");
                good(pools.configure_native(pool_config,0,plan,&device_allocator,&host_allocator),
                     "actual reduced Pools with separate native landing");
            }else{
                uint64_t bytes=0;for(const auto& u:metadata_plan.units)bytes+=uint64_t(u.slot_bytes);
                base.vram_weights_mib=int64_t((bytes+(1<<20)-1)/(1<<20));pool_config=base;
                pi.config=&pool_config;
                good(Planner(pi).plan(&plan),"actual bounded all-resident baseline Planner");
                good(pools.configure(pool_config,0,&device_allocator,&host_allocator),
                     "actual baseline Pools");
            }
            vmem_after_pool=rad_dev_vmem_committed();
            MoverConfig mc;mc.turnover_slots=0;
            before_mover=payload_totals();
            good(mover.init(program,plan,mc,&pools),"actual budgeted Mover, native units omitted");
            after_mover=payload_totals();
            good(mover.bind_compute_stream(compute.value),"Mover actual compute stream");
            initial.resize(program.weights.size());
            for(size_t h=1;h<program.weights.size();++h){auto& info=program.weights[h];
                if(info.decl.group.expert>=0)continue;
                initial[h].resize(size_t(info.stored_bytes),std::byte{0});
                if(info.stored_dtype==RAD_F32){auto* p=reinterpret_cast<float*>(initial[h].data());
                    for(size_t i=0;i<initial[h].size()/4;++i)p[i]=0;
                }
                if(std::any_of(moes.begin(),moes.end(),[&](const auto& m){return h==m.glm_router.projection;})){auto* p=reinterpret_cast<float*>(initial[h].data());
                    p[0*H+0]=4;
                    p[1*H+0]=3;
                    p[2*H+1]=3;
                    p[3*H+1]=1;
                }
            }
            std::unique_ptr<es::Reader> reader;
            good(es::Reader::open((temp.path/"store").string(),specs,source_catalog,receipt.manifest_checksum,es::Limits{},&reader),"read exact native source for original baseline bases");
            for(size_t u=0;u<plan.units.size();++u){const auto& unit=plan.units[u];
                auto* base=static_cast<std::byte*>(mover.unit_base(int32_t(u)));
                if(inventory&&inventory->find(int32_t(u))){
                    need(!base,"native unit unexpectedly retained original Mover allocation");
                    for(const auto w:unit.weights){const auto* slot=mover.residency()->lookup(w);
                        need(slot&&slot->ptr&&!slot->ready,"native WTAB sentinel not initialized before Ctx");}
                    continue;
                }
                need(base,"required original static/resident Mover base absent");
                std::memset(base,0,size_t(unit.bytes));
                if(unit.expert>=0){const auto it=std::find_if(specs.begin(),specs.end(),[&](const auto& x){return x.key().layer==unit.layer&&x.key().expert==unit.expert;});
                    need(it!=specs.end(),"TP1 S1 spec for actual unit");
                    std::vector<std::byte> record;
                    good(reader->read(it->key(),&record),"read original baseline unit");++original_expert_reads;
                    for(size_t member=0;member<unit.weights.size();++member)std::memcpy(base+unit.offset[member],record.data()+unit.offset[member],size_t(program.weights[unit.weights[member]].stored_bytes));
                }
                else for(size_t m=0;m<unit.weights.size();++m){const auto h=unit.weights[m];
                    std::memcpy(base+unit.offset[m],initial[h].data(),initial[h].size());
                }
            }
            if(owned&&!initialize_owner_context)return;
            if(owned)good(NativeBackingOwner::open(program,plan,mover,source,&owner),
                           "typed native rank owner");
            CtxDesc d{};
            d.program=&program;
            d.stream=compute.value;
            d.residency=mover.residency();
            d.arch_step=step;
            d.record_passes=false;
            d.expert_prepare_owner=owner.get();
            d.expert_prepare_limits.max_rows=R;
            d.expert_prepare_limits.max_experts=E;
            d.expert_prepare_limits.max_top_k=K;
            d.expert_prepare_limits.max_readback_bytes=68;
            good(ctx.init(d),"real prepared Ctx with Mover Residency");
            need(program.requires_permanent_model_failure()==permanent&&ctx.requires_permanent_model_failure()==permanent&&
                 !ctx.checks_device_status()&&!ctx.records_passes(),"actual declared zero-collector eager Program/Ctx profile");
            good(ctx.bind_thread(),"HOST rank thread");
            for(size_t op=1;op<program.ops.size();++op)good(ctx.set_op_domain(rad_op(op),RAD_DOMAIN_HOST),"actual HOST rows");
            caller.fill(0x5a5a);
            reset();
        }
        ~Fixture(){if(owner&&owner->quiesce()<0)std::abort();
            ctx.shutdown();
            if(owner){good(mover.native_record_guards_host(),"fixed native pool guards");
                good(owner->close_after_context_shutdown(),"withdraw only after actual Ctx/WTAB teardown");
                owner.reset();
        }}
        void build_source(){
            source_catalog.source_id="synthetic-native-owner-tp1";
            source_catalog.revision="synthetic-not-checkpoint";
            std::vector<std::byte> image;
            for(const auto& unit:plan.units)if(unit.expert>=0){for(int projection=0;projection<2;++projection){const auto& info=builder->program().weights[unit.weights[size_t(projection*2)]];
                    es::SourceTensor tensor;
                    tensor.logical=info.source;
                    tensor.projection=projection?es::Projection::down:es::Projection::gate_up;
                    tensor.encoding=info.enc;
                    tensor.shape={projection?H:2*H,H};
                    for(int plane=0;plane<2;++plane){const auto rows=tensor.shape[0],row_bytes=plane?H/32:H/2;
                        const auto offset=size_t(align_up(int64_t(image.size()),4096));
                        const auto count=size_t(rows*row_bytes);
                        image.resize(offset+count,std::byte{0});
                        for(int64_t row=0;row<rows;++row)for(int64_t col=0;col<row_bytes;++col){const uint8_t b=plane?uint8_t(124+(unit.expert+row+col)%3):uint8_t(0x11u+uint8_t((unit.expert*17+projection*31+row*7+col*13)%7)*0x11u);
                            image[offset+size_t(row*row_bytes+col)]=std::byte(b);
                        }tensor.planes[size_t(plane)]={(temp.path/"source.bin").string(),offset,count,es::checksum({image.data()+offset,count})};
                    }source_catalog.tensors.push_back(std::move(tensor));
            }}
            write(temp.path/"source.bin",image);
            for(size_t u=0;u<plan.units.size();++u)if(plan.units[u].expert>=0){es::RecordSpec spec;
                std::string why;
                const auto captured=es::capture_unit(builder->program(),plan,u,source_catalog,&spec,&why);
                good(captured,"actual TP1 ROWparts2/COL source capture, unit "+std::to_string(u)+": "+why);
                need(spec.key().world==1&&spec.key().rank==0&&spec.stride()==32768,"truthful TP1 native class geometry");
                specs.push_back(std::move(spec));
            }
        }
        std::vector<RecordOwnerBinding> bound(Source& from){std::vector<RecordOwnerBinding> result;
            for(size_t u=0;u<plan.units.size();++u)if(plan.units[u].expert>0){const auto i=std::find_if(from.descriptors.begin(),from.descriptors.end(),[&](const auto& d){return d.layer==plan.units[u].layer&&d.expert==plan.units[u].expert;});
                need(i!=from.descriptors.end(),"actual unit bound to immutable native descriptor");
                result.push_back({int32_t(u),uint32_t(i-from.descriptors.begin())});
            }return result;
        }
        MoeFP8& selected(){return moes[size_t(active_layer)];}
        void reset(){for(int64_t row=0;row<T;++row){const int row_mode=mode==3?(row%2?2:0):mode;
                for(int64_t h=0;h<H;++h){const float x=h==0?(row_mode==2?0.f:1.f):h==1?(row_mode==0?0.f:row_mode==1?2.f:1.f):float(int((row+h)%5)-2)*.03125f;
                    static_cast<uint16_t*>(ctx.buf_ptr(selected().w.h.x))[row*H+h]=rad_f32_to_bf16(x);
                    static_cast<uint8_t*>(ctx.buf_ptr(selected().w.h.q))[row*H+h]=rad_f32_to_fp8e4m3(x);
                }static_cast<float*>(ctx.buf_ptr(selected().w.h.s))[row]=1.f;
            }std::memset(ctx.buf_ptr(selected().w.egu),0x5a,size_t(R*K*2*H*2));
        }
        int run(){reset();
            RadBatch b{};
            b.n_tok=T;
            b.n_seq=1;
            b.phase=RAD_PHASE_PREFILL;
            b.step=9;
            active=this;
            const int status=ctx.run_step(&b);
            active=nullptr;
            good(rad_stream_sync(compute.value),"final actual HOST completion");
            if(status==RAD_OK){auto* output=static_cast<uint16_t*>(ctx.buf_ptr(selected().w.h.x));
                std::copy(output,output+caller.size(),caller.begin());
            }return status;
        }
    };
    void step(RadCtx* c,const RadBatch* b){auto& f=*active;
        ++f.step_entries;
        if(f.step_mode==1){rad_step_fail(c,"first error before Native prepare");
            f.late_prepare_status=rad_expert_prepare(c,nullptr);return;}
        if(f.step_mode==2){rad_step_fail(c,"Legacy first A before lane B");
            f.late_prepare_status=rad_lane(c,1);return;}
        const auto down=f.selected().w.edn;
        if(f.fail_dn)f.selected().w.edn=f.bad_dn;
        for(int64_t r0=0;r0<b->n_tok;r0+=R)f.selected().pass(c,b->n_tok,r0,std::min<int64_t>(R,b->n_tok-r0));
        f.selected().w.edn=down;
    }
    std::vector<int32_t> residents(const Fixture& f){
        std::vector<int32_t> result;
        for(size_t u=0;u<f.metadata_plan.units.size();++u)
            if(f.metadata_plan.units[u].expert==0)result.push_back(int32_t(u));
        return result;
    }
    void success(const std::string& home){++cases;
        Fixture reference(home,false,3*32768,8),f(home,true,3*32768,8);
        need(f.original_expert_reads==8&&reference.original_expert_reads==32,
             "native bootstrap still loaded owned experts into original backing");
        const auto& bytes=f.inventory->footprint();
        need(bytes.owned_units==24&&bytes.consumer_slots==3&&bytes.regions==4,
             "consumer region count tracks all records instead of per-slice union");
        need(bytes.native_device_bytes==4*49152&&bytes.landing_bytes==4*49152,
             "full native region/tail/guards not charged");
        need(f.pools.host_pinned().bytes()==int64_t(bytes.landing_bytes)&&
             f.pools.host_pinned().used()==int64_t(bytes.landing_bytes)&&f.pool_config.host_pool_mib==0,
             "old expert host pool still allocated or landing hidden");
        need(f.pools.weights().used()==int64_t(bytes.weights_minimum())&&
             f.mover.record_base_allocation_bytes()==bytes.original_static_bytes,
             "native payload retained in static/slab/pinned original allocations");
        need(f.pools.kv().used()==0&&f.pools.kv().bytes()==1<<20&&
             f.pool_config.vram_kv_mib==reference.pool_config.vram_kv_mib,
             "expert consumer reserved or changed KV pool");
        const bool elastic=f.pools.kv().elastic();
        const uint64_t expected_device_calls=elastic?1:2;
        const uint64_t kv_allocator_bytes=elastic?0:uint64_t(f.pools.kv().bytes());
        const uint64_t kv_va=elastic?uint64_t(align_up(f.pools.kv().bytes(),VMem::granularity())):0;
        const auto ledger=[&](){
            std::string out="device calls="+std::to_string(f.device_allocator.calls)+
                " expected="+std::to_string(expected_device_calls)+
                " requested="+std::to_string(f.device_allocator.requested)+
                " rounded="+std::to_string(f.device_allocator.physical)+
                "; host calls="+std::to_string(f.host_allocator.calls)+
                " requested="+std::to_string(f.host_allocator.requested)+
                " rounded="+std::to_string(f.host_allocator.physical)+
                "; KV elastic="+std::to_string(elastic)+
                " logical="+std::to_string(f.pools.kv().bytes())+
                " committed="+std::to_string(f.pools.kv().committed())+
                " VA-rounded-from-granularity="+std::to_string(kv_va);
            for(const auto& x:f.device_allocator.live)
                out+="; device-allocation(kind="+std::to_string(x.kind)+",bytes="+
                    std::to_string(x.bytes)+",rounded="+std::to_string(x.physical)+")";
            for(const auto& x:f.host_allocator.live)
                out+="; host-allocation(kind="+std::to_string(x.kind)+",bytes="+
                    std::to_string(x.bytes)+",rounded="+std::to_string(x.physical)+")";
            return out;
        }();
        need(f.device_allocator.calls==expected_device_calls&&f.host_allocator.calls==1,
             ("unreported pool allocation path: "+ledger).c_str());
        need(f.device_allocator.live.size()==expected_device_calls&&f.host_allocator.live.size()==1&&
             std::all_of(f.device_allocator.live.begin(),f.device_allocator.live.end(),
                         [](const auto& x){return x.kind==RAD_MEM_DEVICE;})&&
             f.host_allocator.live[0].kind==RAD_MEM_HOST_MAPPED,
             ("wrong actual Pool allocation kinds/counts: "+ledger).c_str());
        need(f.device_allocator.requested==uint64_t(f.pools.weights().bytes())+kv_allocator_bytes&&
             f.host_allocator.requested==bytes.landing_bytes,
             ("actual HOST allocator request receipts disagree: "+ledger).c_str());
        need(!elastic||(f.pools.kv().base()&&f.pools.kv().committed()==0&&
                        f.vmem_before_pool==f.vmem_after_pool),
             ("KV virtual reservation hid committed payload: "+ledger).c_str());
        need(f.before_mover==f.after_mover,
             "Mover added rad_dev_alloc payload outside charged Pools (DEVICE/pinned/pageable/mapped)");
        need(f.pools.weights().bytes()<reference.pools.weights().bytes(),
             "reduced footprint still requested full all-record weights pool");
        std::vector<std::pair<rad_weight,WeightSlot>> resident_slots;
        std::vector<std::pair<int32_t,std::vector<std::byte>>> resident_bytes;
        for(size_t u=0;u<f.plan.units.size();++u){const auto& mu=f.plan.units[u];
            if(mu.expert==0){const auto* data=static_cast<const std::byte*>(f.mover.unit_base(int32_t(u)));
                resident_bytes.emplace_back(int32_t(u),std::vector<std::byte>(data,data+mu.bytes));
                for(const auto w:mu.weights)resident_slots.emplace_back(w,*f.mover.residency()->lookup(w));
            }else if(mu.expert>0){
                need(mu.tier==Tier::NativeStore&&!f.mover.unit_base(int32_t(u)),
                     "native-only record unexpectedly has an original fallback base");
            }
        }
        size_t pass=0;
        for(const auto [layer,mode]:std::array<std::pair<int,int>,8>{{{0,0},{0,0},{7,0},{0,0},{7,0},{7,2},{7,2},{3,3}}}){
            reference.active_layer=f.active_layer=layer;reference.mode=f.mode=mode;
            good(reference.run(),"same exact native bytes, actual all-resident HOST FFN");
            const auto reads=f.source->bridge->stats().provider_reads;
            good(f.run(),"actual native-only backing/current routes/cache/Mover/FFN");
            need(f.caller==reference.caller,"native backing changed actual HOST FFN values");
            if(pass==1||pass==3||pass==4||pass==6)
                need(f.source->bridge->stats().provider_reads==reads,
                     "rank-global HOT hit reread layer/shard record");
            for(const auto id:f.source->last_request)
                need(f.source->descriptors[id].layer==layer&&f.source->descriptors[id].expert>0,
                     "current route union included other layer or static expert");
            need(f.owner->stats().held_leases==0,"last reader did not release cache lease");
            good(f.mover.native_record_guards_host(),"native consumer/landing/tail guards");
            ++pass;
        }
        for(const auto& x:resident_bytes)
            need(!std::memcmp(f.mover.unit_base(x.first),x.second.data(),x.second.size()),
                 "fixed resident expert payload changed");
        for(const auto& x:resident_slots){const auto* now=f.mover.residency()->lookup(x.first);
            need(now->ptr==x.second.ptr&&now->ready==x.second.ready&&now->generation==x.second.generation,
                 "fixed resident pointer/generation changed");
        }
        const auto cache=f.source->bridge->stats().cache;
        need(f.pools.kv().used()==0&&(!elastic||f.pools.kv().committed()==0),
             "native FFN changed or committed the separate KV pool");
        need(cache.hot_hits&&cache.evictions&&cache.peak_hot_bytes<=2*32768&&
             cache.peak_cold_bytes==3*32768,"global HOT/COLD ledger receipts/worst miss union");
        need(f.owner->stats().peak_held_leases==3&&f.owner->stats().published_units,
             "full admitted three-record current-route union was not exercised");
        std::cout<<"RECEIPT owned="<<bytes.owned_units<<" regions="<<bytes.regions
                 <<" native="<<bytes.native_device_bytes<<" landing="<<bytes.landing_bytes
                 <<" static="<<bytes.original_static_bytes<<" metadata-reservation="<<bytes.metadata_reservation
                 <<" HOT="<<bytes.hot_bytes<<" COLD="<<bytes.cold_bytes
                 <<" source-metadata-reservation="<<f.source->bridge->stats().metadata_reservation
                 <<" source-staging-peak="<<f.source->bridge->stats().peak_staging_bytes
                 <<" pool-requested="<<f.device_allocator.requested+f.host_allocator.requested
                 <<" HOST-allocator-rounded="<<f.device_allocator.physical+f.host_allocator.physical
                 <<" KV-elastic="<<elastic<<" KV-logical="<<f.pools.kv().bytes()
                 <<" KV-VA-rounded-from-granularity="<<kv_va
                 <<" KV-committed="<<f.pools.kv().committed()
                 <<" (2MiB allocation alignment is actual backing, not reclaimed bytes)\n";
    }
    void growth(const std::string& home){++cases;
        Fixture a(home,true),b(home,true,3*32768,8);
        const auto& x=a.inventory->footprint();const auto& y=b.inventory->footprint();
        need(x.owned_units==3&&y.owned_units==24&&x.native_device_bytes==y.native_device_bytes&&
             x.landing_bytes==y.landing_bytes&&x.consumer_slots==y.consumer_slots,
             "native storage grows one region per catalog record/layer");
        need(y.native_payload_bytes==8*x.native_payload_bytes&&y.metadata_reservation>x.metadata_reservation,
             "inventory growth not attributed to durable payload and metadata");
        need(y.original_static_bytes>x.original_static_bytes&&
             b.mover.record_base_allocation_bytes()==y.original_static_bytes,
             "growth hid original resident/static payload");
        need(a.pools.kv().bytes()==b.pools.kv().bytes()&&
             a.host_allocator.requested==b.host_allocator.requested,
             "catalog growth changed KV or hidden old host allocation");
    }
    void admission(const std::string& home){++cases;
        Fixture f(home,true);auto& program=f.builder->program();
        const auto bound=f.bound(*f.source);const auto fixed=residents(f);
        auto catalog=f.source->descriptors;auto limits=f.inventory->limits();
        auto refuse=[&](const auto& desc,const auto& claims,const auto& mask,const auto& cap){
            std::shared_ptr<const NativeInventory> out;std::string why;
            need(NativeInventory::bind(program,f.metadata_plan,desc,claims,mask,cap,&out,&why)<0&&
                 !out&&!why.empty(),"invalid capability admitted/partially returned");
        };
        const auto record=bound.front().record;
        auto bad=catalog;bad[record].rank=1;refuse(bad,bound,fixed,limits);
        bad=catalog;bad[record].world=4;refuse(bad,bound,fixed,limits);
        bad=catalog;bad[record].dtype[1]=RAD_F32;refuse(bad,bound,fixed,limits);
        bad=catalog;bad[record].shape[0][1]+=32;refuse(bad,bound,fixed,limits);
        bad=catalog;bad[record].source_generation^=1;refuse(bad,bound,fixed,limits);
        auto fewer=bound;fewer.pop_back();refuse(catalog,fewer,fixed,limits);
        auto duplicate=bound;duplicate.push_back(bound.front());refuse(catalog,duplicate,fixed,limits);
        auto overlap=fixed;overlap.push_back(bound.front().unit);refuse(catalog,bound,overlap,limits);
        auto cap=limits;cap.cold_bytes=32768;refuse(catalog,bound,fixed,cap);
        cap=limits;cap.max_device_bytes=1;refuse(catalog,bound,fixed,cap);
        cap=limits;cap.max_landing_bytes=1;refuse(catalog,bound,fixed,cap);
        cap=limits;cap.max_metadata_bytes=1;refuse(catalog,bound,fixed,cap);
        cap=limits;cap.max_regions=2;refuse(catalog,bound,fixed,cap);
        cap=limits;cap.max_weights_bytes=1;refuse(catalog,bound,fixed,cap);
        cap=limits;cap.max_host_bytes=1;refuse(catalog,bound,fixed,cap);
        const auto w=f.metadata_plan.units[size_t(bound.front().unit)].weights[0];
        const auto saved=program.weights[w].stored_bytes;program.weights[w].stored_bytes+=1;
        need(f.inventory->validate(program,f.plan)<0,"changed actual Program metadata admitted");
        program.weights[w].stored_bytes=saved;
        const auto canonical=program.expert_route_maps[0].entries[0].canonical;
        program.expert_route_maps[0].entries[0].canonical^=1;
        need(f.inventory->validate(program,f.plan)<0,"canonical/runtime route map drift ignored");
        program.expert_route_maps[0].entries[0].canonical=canonical;
        good(f.inventory->validate(program,f.plan),"valid actual identity after local negative fixtures restored");
        Config out=f.pool_config;const auto old=out.vram_weights_mib;
        Config wrong=out;wrong.host_pool_mib=1;
        need(f.inventory->configure_pools(wrong,&out)<0&&out.vram_weights_mib==old,
             "old host budget silently preserved/replaced");
        wrong=f.pool_config;wrong.vram_kv_mib=INT64_MAX;
        need(f.inventory->configure_pools(wrong,&out)<0,"KV arithmetic overflow admitted");
    }
    void pools_and_placement(const std::string& home){++cases;
        Fixture f(home,true);
        CountingAllocator device,host;
        {
            Pools refused;auto bad=f.pool_config;bad.host_pool_mib=1;
            need(refused.configure_native(bad,0,f.plan,&device,&host)<0&&
                 !device.calls&&!host.calls,"old host pool refusal allocated payload");
            bad=f.pool_config;bad.vram_weights_mib=0;
            need(refused.configure_native(bad,0,f.plan,&device,&host)<0&&
                 !device.calls&&!host.calls,"unresolved native pool allocated payload");
            bad=f.pool_config;bad.embedding_host_mib=INT64_MAX;
            need(refused.configure_native(bad,0,f.plan,&device,&host)<0&&
                 !device.calls&&!host.calls,"overflow host budget allocated payload");
        }
        {
            Pools oom;host.fail_at=1;
            need(oom.configure_native(f.pool_config,0,f.plan,&device,&host)==RAD_E_NOMEM,
                 "actual HOST landing allocation fault hidden");
            need(!oom.configured()&&!oom.native_inventory(),"failed typed Pools published authority");
        }
        need(device.live.empty()&&host.live.empty(),"failed Pool allocation leaked actual HOST backing");
        {
            Pools wrong;good(wrong.configure(f.pool_config,0,&device,&host),"legacy pool mismatch fixture");
            Mover m;MoverConfig mc;
            need(m.init(f.builder->program(),f.plan,mc,&wrong)<0,
                 "native Mover accepted untyped Pools with missing landing reservation");
        }
        auto changed=f.plan;const auto owned=f.inventory->owned().front().unit;
        changed.units[size_t(owned)].tier=Tier::VRAM;
        Mover bad;MoverConfig mc;
        need(bad.init(f.builder->program(),changed,mc,&f.pools)<0,
             "native owned unit regained an unverified permanent allocation");
        Strategy parsed=Strategy::AllVram;
        need(strategy_parse("native-backed",&parsed)<0&&parsed==Strategy::AllVram,
             "internal native capability exposed as a CLI strategy");
        need(std::string(tier_name(Tier::NativeStore))=="native-store"&&
             f.plan.report(f.builder->program()).find("native consumers")!=std::string::npos,
             "native durable/consumer costs absent from plan report");
    }
    struct NoopOwner final:ExpertPrepareOwner{
        int prepare(const ExpertPrepareView&,bool*)override{return RAD_OK;}
    };
    void owner_admission(const std::string& home){++cases;
        Fixture f(home,true),other(home,true);
        const auto epoch=f.mover.residency()->epoch();
        const auto reads=f.source->bridge->stats().provider_reads;
        auto desc=[&](ExpertPrepareOwner* owner,Program* program,ResidencyTable* residency){
            CtxDesc d{};d.program=program;d.stream=f.compute.value;d.residency=residency;
            d.arch_step=step;d.record_passes=false;d.expert_prepare_owner=owner;
            d.expert_prepare_limits.max_rows=R;d.expert_prepare_limits.max_experts=E;
            d.expert_prepare_limits.max_top_k=K;d.expert_prepare_limits.max_readback_bytes=68;
            return d;
        };
        {Ctx rejected;need(rejected.init(desc(nullptr,&f.builder->program(),f.mover.residency()))<0,
                           "native residency accepted omitted Owner");}
        {NoopOwner owner;Ctx rejected;need(rejected.init(desc(&owner,&f.builder->program(),f.mover.residency()))<0,
                                         "native residency accepted default no-op Owner");}
        {Ctx rejected;need(rejected.init(desc(other.owner.get(),&f.builder->program(),f.mover.residency()))<0,
                           "native residency accepted foreign inventory Owner");}
        {Ctx rejected;need(rejected.init(desc(f.owner.get(),&other.builder->program(),f.mover.residency()))<0,
                           "native authority accepted same-shaped foreign Program/handle collision");}
        {Ctx rejected;need(rejected.init(desc(f.owner.get(),&f.builder->program(),f.mover.residency()))<0,
                           "native authority accepted a second WTAB/context reader");}
        need(f.mover.residency()->epoch()==epoch&&f.source->bridge->stats().provider_reads==reads,
             "Owner admission rejection changed residency/performed payload read");
        good(f.run(),"valid native Owner remains usable after foreign Ctx refusals");
    }
    void copy_failure(const std::string& home){++cases;
        Fixture f(home,true);f.mode=2;
        const auto before=f.caller;const auto epoch=f.mover.residency()->epoch();
        const auto faults=copy_fault::failed;
        copy_fault::stream=f.mover.transfer_stream();copy_fault::remaining=1;
        const int status=f.run();copy_fault::stream=nullptr;copy_fault::remaining=-1;
        need(status==RAD_E_IO&&copy_fault::failed==faults+1&&f.caller==before&&
             f.mover.residency()->epoch()==epoch,"late native copy fault partially published/committed");
        need(f.owner->stats().published_units==0&&f.owner->stats().copy_bytes==36864&&
             f.owner->stats().held_leases==0,"native fault retired before last copy or skipped cleanup");
        good(f.mover.native_record_guards_host(),"copy refusal native guards");
        need(f.run()<0,"native failed Ctx pass was reused");
    }
    void source_failure(const std::string& home){++cases;
        Fixture f(home,true);const auto before=f.caller;
        const auto epoch=f.mover.residency()->epoch();
        const auto reads=f.source->bridge->stats().provider_reads;
        const auto selected=std::find_if(f.source->descriptors.begin(),f.source->descriptors.end(),
                                         [](const auto& d){return d.expert==1;});
        need(selected!=f.source->descriptors.end(),"selected native source descriptor");
        auto& record=*selected;
        ++record.source_generation;
        need(f.run()<0&&f.caller==before&&f.mover.residency()->epoch()==epoch&&
             f.source->bridge->stats().provider_reads==reads,
             "native source generation drift reached acquire/publication");
        need(f.owner->stats().held_leases==0,"source refusal retained payload lease");
    }
    void writer(const std::string& home){++cases;
        Fixture f(home,true,3*32768,1,false);
        uint64_t token=0;good(f.mover.native_record_session_open(&token),"native session actual bounded regions");
        const auto a=f.inventory->owned()[0],b=f.inventory->owned()[1];
        const auto cls=f.inventory->classes()[size_t(a.klass)];
        need(a.klass==b.klass,"tiny fixture native class");
        const auto epoch=f.mover.residency()->epoch();
        const RecordPublication sentinel{a.unit,cls.first_region};
        need(f.mover.record_publish(token,{&sentinel,1})<0&&f.mover.residency()->epoch()==epoch,
             "selected sentinel fallback was admitted");
        const RecordPublication unready{a.unit,cls.first_region+1};
        need(f.mover.record_publish(token,{&unready,1})<0&&f.mover.residency()->epoch()==epoch,
             "unverified/unuploaded native region was admitted");
        const std::array<uint32_t,2> ids{a.record,b.record};NativeRecordBatch ready;
        good(f.source->acquire(ids,&ready),"actual S1/cache complete native union");
        need(ready.leases.size()==2,"manual actual native lease count");
        auto corrupt=std::vector<std::byte>(ready.leases[0].bytes.begin(),ready.leases[0].bytes.end());
        if(corrupt.empty())throw std::runtime_error("empty native checksum corruption fixture");
        need(corrupt.size()==size_t(cls.stride),"bounded checksum fixture must contain one complete record");
        corrupt.back()^=std::byte{1};
        need(f.mover.native_record_upload(token,a.unit,cls.first_region+1,corrupt)<0,
             "wrong whole-record checksum admitted to native landing");
        need(f.mover.native_record_upload(token,a.unit,cls.first_region,ready.leases[0].bytes)<0,
             "sentinel overwritten by record upload");
        good(f.mover.native_record_upload(token,a.unit,cls.first_region+1,ready.leases[0].bytes),
             "actual native record first upload");
        good(f.mover.native_record_upload(token,b.unit,cls.first_region+2,ready.leases[1].bytes),
             "actual native record second upload");
        std::array<RecordPublication,2> batch{{{a.unit,cls.first_region+1},{b.unit,INT32_MAX}}};
        need(f.mover.record_publish(token,batch)<0&&f.mover.residency()->epoch()==epoch,
             "late native region failure partially published");
        batch[1].region=cls.first_region+2;
        {
            const Plan exact=f.plan;
            const auto exemplar=cls.exemplar;
            for(int variant=0;variant<4;++variant){
                if(variant==0)f.plan.units[size_t(exemplar)].offset.clear();
                if(variant==1)f.plan.units[size_t(exemplar)].offset[0]=INT64_MAX;
                if(variant==2)f.plan.unit_of_weight[f.plan.units[size_t(a.unit)].weights[0]]=residents(f).front();
                if(variant==3)f.plan.classes.push_back(SlabClass{});
                need(f.mover.record_publish(token,batch)<0&&f.mover.residency()->epoch()==epoch,
                     "post-upload native unit/map/class drift reached atomic publication");
                f.plan=exact;
            }
        }
        const auto weight=f.plan.units[size_t(b.unit)].weights.back();
        auto* slot=const_cast<WeightSlot*>(f.mover.residency()->lookup(weight));
        const auto generation=slot->generation;slot->generation=UINT32_MAX;
        need(f.mover.record_publish(token,batch)<0&&f.mover.residency()->epoch()==epoch,
             "native generation wrap partially published");
        slot->generation=generation;
        good(f.mover.record_publish(token,batch),"verified whole-union native publication");
        need(f.mover.residency()->epoch()>epoch,"native writer did not publish through Residency");
        good(f.mover.record_restore(token),"native both-stream last-reader drain");
        const auto restored=f.mover.residency()->epoch();
        need(f.mover.record_publish(token,batch)<0&&f.mover.residency()->epoch()==restored,
             "retired native upload reused without current verified record");
        ready.leases.clear();
        f.ctx.shutdown();
        good(f.mover.record_session_close(token),"withdraw native capability after empty Ctx teardown");
        for(const auto& unit:f.inventory->owned())for(const auto w:f.plan.units[size_t(unit.unit)].weights)
            need(!f.mover.residency()->lookup(w)->ptr,"native-only close restored original base");
        good(f.mover.native_record_guards_host(),"manual native writer guards");
    }
    void late_failure(const std::string& home){++cases;
        Fixture f(home,true);const auto before=f.caller;
        f.fail_dn=true;
        need(f.run()<0&&f.caller==before,"late native DN issue committed caller result");
        need(f.owner->stats().published_units&&f.owner->stats().held_leases==0,
             "native late issue missed last-reader/end-pass cleanup");
        good(f.mover.native_record_guards_host(),"late native issue guard check");
        need(f.run()<0,"native failed architecture pass was reused");
    }
    void withdraw(const std::string& home){++cases;
        Fixture f(home,true);good(f.run(),"native positive pass before withdrawal");
        auto* resident=f.mover.unit_base(residents(f).front());
        good(f.owner->quiesce(),"native last reader drained");
        f.ctx.shutdown();
        good(f.owner->close_after_context_shutdown(),"native close after real WTAB shutdown");
        f.owner.reset();
        for(const auto& owned:f.inventory->owned())for(const auto w:f.plan.units[size_t(owned.unit)].weights)
            need(!f.mover.residency()->lookup(w)->ptr&&!f.mover.unit_base(owned.unit),
                 "native close restored a nonexistent original base/sentinel");
        need(f.mover.unit_base(residents(f).front())==resident,"native withdrawal changed fixed resident");
        uint64_t token=0;
        need(f.mover.native_record_session_open(&token)<0&&!token,
             "closed native capability was silently reactivated");
        good(f.mover.native_record_guards_host(),"withdrawal retained guarded fixed backing");
    }
    void failed_bringup(const std::string& home){++cases;
        Fixture f(home,true,3*32768,1,false);
        std::unique_ptr<NativeBackingOwner> owner;
        good(NativeBackingOwner::open(f.builder->program(),f.plan,f.mover,f.source,&owner),
             "native owner before failing Ctx admission");
        const auto reads=f.source->bridge->stats().provider_reads;
        {
            Ctx failed;CtxDesc d{};d.program=&f.builder->program();d.stream=f.compute.value;
            d.residency=f.mover.residency();d.arch_step=step;d.record_passes=false;
            d.expert_prepare_owner=owner.get();d.expert_prepare_limits.max_rows=R;
            d.expert_prepare_limits.max_experts=E;d.expert_prepare_limits.max_top_k=K;
            d.expert_prepare_limits.max_readback_bytes=1;
            need(failed.init(d)<0,"insufficient handoff bank unexpectedly initialized");
        }
        // No Ctx accepted this Owner; its ordinary failed-startup destructor may drain/close.
        owner.reset();
        need(f.source->bridge->stats().provider_reads==reads,
             "failed native Ctx bringup acquired payload");
        for(const auto& owned:f.inventory->owned())for(const auto w:f.plan.units[size_t(owned.unit)].weights)
            need(!f.mover.residency()->lookup(w)->ptr,
                 "failed unbound Owner did not close/withdraw capability");
        good(f.mover.native_record_guards_host(),"failed bringup guards");
    }
    void mutable_authority(const std::string& home){++cases;
        {
            Fixture f(home,true,3*32768,1,false);
            const Plan saved=f.plan;const auto exemplar=f.inventory->classes()[0].exemplar;
            const auto fixed=residents(f).front();
            for(int variant=0;variant<7;++variant){
                if(variant==0)f.plan.unit_of_weight.clear();
                if(variant==1)f.plan.unit_of_weight.resize(f.plan.unit_of_weight.size()-1);
                if(variant==2)for(const auto& owned:f.inventory->owned())
                    for(const auto w:f.plan.units[size_t(owned.unit)].weights)f.plan.unit_of_weight[w]=fixed;
                if(variant==3)f.plan.units[size_t(exemplar)].offset.clear();
                if(variant==4)f.plan.units[size_t(exemplar)].offset[0]=INT64_MAX;
                if(variant==5)f.plan.units[size_t(exemplar)].slab_class=INT32_MAX;
                if(variant==6)f.plan.classes.push_back(SlabClass{});
                const auto epoch=f.mover.residency()->epoch();
                const auto reads=f.source->bridge->stats().provider_reads;
                const auto before=payload_totals();
                const auto weights_used=f.pools.weights().used();
                const auto host_used=f.pools.host_pinned().used();
                std::unique_ptr<NativeBackingOwner> out;
                need(f.inventory->validate(f.builder->program(),f.plan)<0,
                     "full authority validation accepted changed map/unit/class table");
                need(NativeBackingOwner::open(f.builder->program(),f.plan,f.mover,f.source,&out)<0&&!out,
                     "changed startup authority reached Owner fixed-pointer snapshot");
                {
                    Mover rejected;MoverConfig config;
                    need(rejected.init(f.builder->program(),f.plan,config,&f.pools)<0,
                         "changed class/map authority reached proportional Mover allocation");
                }
                need(f.mover.residency()->epoch()==epoch&&f.source->bridge->stats().provider_reads==reads&&
                     payload_totals()==before&&f.pools.weights().used()==weights_used&&f.pools.host_pinned().used()==host_used,
                     "startup authority refusal read/published/allocated or reserved payload");
                f.plan=saved;
            }
        }
        for(int variant=0;variant<12;++variant){
            Fixture f(home,true);const Plan saved=f.plan;
            const auto exemplar=f.inventory->classes()[0].exemplar;
            const auto fixed=residents(f).front();
            if(variant==0)f.plan.unit_of_weight.clear();
            if(variant==1)f.plan.unit_of_weight.resize(f.plan.unit_of_weight.size()-1);
            if(variant==2)for(const auto& owned:f.inventory->owned())
                for(const auto w:f.plan.units[size_t(owned.unit)].weights)f.plan.unit_of_weight[w]=fixed;
            if(variant==3)f.plan.units[size_t(exemplar)].offset.clear();
            if(variant==4)f.plan.units[size_t(exemplar)].offset[0]=INT64_MAX;
            if(variant==5)f.plan.units[size_t(exemplar)].weights[0]=f.plan.units[size_t(fixed)].weights[0];
            if(variant==6)f.plan.units[size_t(exemplar)].slab_class=INT32_MAX;
            if(variant==7)++f.plan.classes[size_t(f.plan.units[size_t(exemplar)].slab_class)].stride;
            if(variant==8)f.plan.classes.push_back(SlabClass{});
            if(variant==9)++f.plan.units[size_t(exemplar)].bytes;
            if(variant==10)f.plan.units[size_t(exemplar)].slot_bytes+=4096;
            if(variant==11)f.plan.weight[f.plan.units[size_t(exemplar)].weights[0]].unit=fixed;
            const auto caller=f.caller;const auto epoch=f.mover.residency()->epoch();
            const auto reads=f.source->bridge->stats().provider_reads;
            const int status=f.run();
            f.plan=saved; // restore only metadata for safe teardown; failed Ctx remains sealed.
            need(status<0&&f.caller==caller&&f.mover.residency()->epoch()==epoch&&
                 f.source->bridge->stats().provider_reads==reads&&f.owner->stats().published_units==0,
                 "after-open map/unit/class drift acquired/published selected sentinels or committed output");
            need(f.owner->stats().held_leases==0&&f.run()<0,"authority failure retained lease or reused failed pass");
        }
    }
    void landing_lifetime(const std::string& home){++cases;
        Fixture f(home,true,3*32768,1,false);
        uint64_t token=0;good(f.mover.native_record_session_open(&token),"native upload lifetime session");
        const auto a=f.inventory->owned()[0],b=f.inventory->owned()[1];
        const auto cls=f.inventory->classes()[size_t(a.klass)];
        const int region=cls.first_region+1;const std::array<uint32_t,2> ids{a.record,b.record};
        NativeRecordBatch ready;good(f.source->acquire(ids,&ready),"native lifetime exact source union");
        if(ready.leases.size()!=2)throw std::runtime_error("native lifetime source union size");
        need(ready.leases[0].bytes.size()==size_t(cls.stride)&&
             ready.leases[1].bytes.size()==size_t(cls.stride)&&
             !std::equal(ready.leases[0].bytes.begin(),ready.leases[0].bytes.end(),
                         ready.leases[1].bytes.begin(),ready.leases[1].bytes.end()),
             "queued overwrite witness must use distinct complete native records");
        const auto& target=f.mover.native_record_regions()[size_t(region)];
        {
            const Plan exact=f.plan;
            std::vector<std::byte> untouched(size_t(cls.region_bytes));
            std::memcpy(untouched.data(),target.base,untouched.size());
            for(int variant=0;variant<4;++variant){
                if(variant==0)f.plan.units[size_t(cls.exemplar)].offset.clear();
                if(variant==1)f.plan.units[size_t(cls.exemplar)].offset[0]=INT64_MAX;
                if(variant==2)f.plan.unit_of_weight[f.plan.units[size_t(a.unit)].weights[0]]=residents(f).front();
                if(variant==3)f.plan.classes.push_back(SlabClass{});
                need(f.mover.native_record_upload(token,a.unit,region,ready.leases[0].bytes)<0&&
                     !std::memcmp(target.base,untouched.data(),untouched.size()),
                     "native layout/map/class drift reached landing upload");
                f.plan=exact;
            }
        }
        {
            Gate gate(f.mover.transfer_stream());
            good(f.mover.native_record_upload(token,a.unit,region,ready.leases[0].bytes),
                 "first gated actual native upload");
            need(f.mover.native_record_upload(token,b.unit,region,ready.leases[1].bytes)==RAD_E_STATE,
                 "same busy landing overwritten while queued copy still reads it");
            gate.open();good(rad_stream_sync(f.mover.transfer_stream()),"first native queued copy completed");
            need(!std::memcmp(target.base,ready.leases[0].bytes.data(),ready.leases[0].bytes.size()),
                 "rejected duplicate upload mutated the first copy source");
            const RecordPublication publication{a.unit,region};
            good(f.mover.record_publish(token,{&publication,1}),"first verified native upload publication");
            good(f.mover.record_restore(token),"successful first native retirement");
        }
        const auto faults=event_fault::failed;
        {
            Gate gate(f.mover.transfer_stream());
            event_fault::stream=f.mover.transfer_stream();event_fault::remaining=0;
            const int status=f.mover.native_record_upload(token,b.unit,region,ready.leases[1].bytes);
            event_fault::stream=nullptr;event_fault::remaining=-1;
            need(status==RAD_E_IO&&event_fault::failed==faults+1,"actual native event-record failure not injected");
            need(f.mover.native_record_upload(token,a.unit,region,ready.leases[0].bytes)==RAD_E_STATE,
                 "failed-record landing reused before queued copy completed");
            const RecordPublication stale{a.unit,region},failed{b.unit,region};
            const auto epoch=f.mover.residency()->epoch();
            need(f.mover.record_publish(token,{&stale,1})<0&&f.mover.record_publish(token,{&failed,1})<0&&
                 f.mover.residency()->epoch()==epoch,"failed event retained stale verified publication");
            need(f.mover.record_restore(token)==RAD_E_STATE,"closed-gate drain did not refuse");
            need(f.mover.native_record_upload(token,a.unit,region,ready.leases[0].bytes)==RAD_E_STATE,
                 "failed drain cleared busy landing state");
            gate.open();good(f.mover.record_restore(token),"two-stream drain clears failed upload state");
            need(!std::memcmp(target.base,ready.leases[1].bytes.data(),ready.leases[1].bytes.size()),
                 "failed-record source overwritten before actual queued copy finished");
        }
        good(f.mover.native_record_upload(token,a.unit,region,ready.leases[0].bytes),"good native upload after explicit drain");
        const RecordPublication fresh{a.unit,region};
        good(f.mover.record_publish(token,{&fresh,1}),"fresh verified native publication after good-bad-good");
        good(f.mover.record_restore(token),"last native upload reader/copy retirement");
        ready.leases.clear();f.ctx.shutdown();
        good(f.mover.record_session_close(token),"close native upload lifetime session");
        good(f.mover.native_record_guards_host(),"busy/failure native landing guards");
    }
    void terminal_guards(Fixture& f,int first){
        need(first<0&&f.ctx.terminal_status()==first,"strict Native failure not permanently latched");
        const std::string diagnostic=f.ctx.step_error();need(!diagnostic.empty(),"first Native diagnostic absent");
        const auto payload=payload_totals();const auto epoch=f.mover.residency()->epoch();
        const auto reads=f.source->bridge->stats().provider_reads;
        const auto copies=f.ctx.expert_prepare_copies();const auto entries=f.step_entries;
        const auto waits=f.ctx.stream_waits();
        const auto weight=f.plan.units[size_t(residents(f).front())].weights.front();
        RadBatch b{};b.n_tok=T;b.n_seq=1;b.phase=RAD_PHASE_PREFILL;b.step=99;
        active=&f;const int retry=f.ctx.run_step(&b);active=nullptr;
        need(retry==first&&f.ctx.expert_prepare(nullptr)==first&&f.ctx.set_lane(1)==first&&
             f.ctx.weight_ptr(weight)==nullptr,"terminal Native API guard replaced/ignored first status");
        need(std::string(f.ctx.step_error())==diagnostic&&f.ctx.expert_prepare_copies()==copies&&
             f.step_entries==entries&&f.ctx.stream_waits()==waits&&
             f.mover.residency()->epoch()==epoch&&f.source->bridge->stats().provider_reads==reads&&
             payload_totals()==payload,"terminal guard entered prepare/architecture/readiness/allocation/publication work");
        auto* program=&f.builder->program();f.ctx.shutdown();
        const auto after_retirement=payload_totals();
        CtxDesc d{};d.program=program;d.stream=f.compute.value;d.residency=f.mover.residency();
        d.arch_step=step;d.record_passes=false;d.expert_prepare_owner=f.owner.get();
        need(f.ctx.init(d)==first&&payload_totals()==after_retirement&&
             std::string(f.ctx.step_error())==diagnostic,"terminal Native Ctx reinitialized/allocated/lost first diagnostic");
        need(f.mover.residency()->epoch()==epoch&&f.source->bridge->stats().provider_reads==reads,
             "terminal retirement/reinit performed source/publication work");
    }
    void policy_begin_refusal(const std::string& home){++cases;
        Fixture f(home,true);const auto caller=f.caller;const auto epoch=f.mover.residency()->epoch();
        RadBatch b{};b.n_tok=T;b.n_seq=1;b.phase=RAD_PHASE_PREFILL;b.n_ahead=1;
        active=&f;const int status=f.ctx.run_step(&b);active=nullptr;
        need(status==RAD_E_UNSUPPORTED&&f.step_entries==0&&f.ctx.expert_prepare_copies()==0&&
             f.owner->stats().prepares==0&&f.owner->stats().held_leases==0&&f.caller==caller&&
             f.mover.residency()->epoch()==epoch,"unsupported Native begin entered/mutated active owner work");
        terminal_guards(f,status);
    }
    void legacy_begin_refusal(const std::string& home){++cases;
        Fixture f(home,true,3*32768,1,true,false);
        RadBatch b{};b.n_tok=T;b.n_seq=1;b.phase=RAD_PHASE_PREFILL;b.n_ahead=1;
        active=&f;const int status=f.ctx.run_step(&b);active=nullptr;
        need(status==RAD_E_UNSUPPORTED&&f.ctx.terminal_status()==RAD_OK&&f.step_entries==0&&
             f.ctx.expert_prepare_copies()==0&&f.owner->stats().prepares==0,
             "Legacy Native admission changed into permanent-policy behavior/entered owner");
        active=&f;const int retry=f.ctx.run_step(&b);active=nullptr;
        need(retry==status&&f.ctx.terminal_status()==RAD_OK&&f.step_entries==0,
             "Legacy Native fatal admission was reused or relabeled permanent");
    }
    void legacy_first_error(const std::string& home){++cases;
        Fixture f(home,true,3*32768,1,true,false);f.step_mode=2;const auto caller=f.caller;
        const int first=f.run();
        need(first==RAD_E_SHAPE&&f.late_prepare_status==first&&f.ctx.terminal_status()==RAD_OK&&
             f.step_entries==1&&f.ctx.expert_prepare_copies()==0&&f.caller==caller&&
             std::string(f.ctx.step_error())=="Legacy first A before lane B",
             "Legacy Native lane refusal replaced first synchronous pass error");
        RadBatch b{};b.n_tok=T;b.n_seq=1;b.phase=RAD_PHASE_PREFILL;b.step=100;
        active=&f;const int retry=f.ctx.run_step(&b);active=nullptr;
        need(retry==first&&f.ctx.terminal_status()==RAD_OK&&f.step_entries==1&&
             std::string(f.ctx.step_error())=="Legacy first A before lane B"&&
             f.ctx.expert_prepare_copies()==0&&f.owner->stats().held_leases==0&&f.caller==caller,
             "Legacy Native owner/end/retry lost first A to later Unsupported B");
    }
    void collector_owner_refusal(const std::string& home,bool permanent){++cases;
        Fixture f(home,true,3*32768,1,false,permanent,true);
        bool selected=false;for(const auto& op:f.builder->program().ops)for(const auto& band:op.bands)
            for(const auto& resolved:band.dom)selected|=resolved.row&&resolved.row->deferred_status;
        need(selected,"real collector metadata absent from combination test");
        std::unique_ptr<NativeBackingOwner> owner;
        good(NativeBackingOwner::open(f.builder->program(),f.plan,f.mover,f.source,&owner),"actual Native Owner before collector admission refusal");
        void* module=dlopen(UNIFIED_POLICY_COLLECTOR_MODULE,RTLD_NOW|RTLD_NOLOAD);
        need(module!=nullptr,"retain selected metadata fixture handle");
        struct Close {void* handle;~Close(){if(handle)dlclose(handle);}} close{module};
        const auto bodies=reinterpret_cast<uint64_t(*)()>(dlsym(module,"unified_status_bodies"));
        const auto collectors=reinterpret_cast<uint64_t(*)()>(dlsym(module,"unified_status_collectors"));
        need(bodies&&collectors,"actual fixture counter exports");
        const auto before_bodies=bodies(),before_collectors=collectors();
        const auto epoch=f.mover.residency()->epoch(),reads=f.source->bridge->stats().provider_reads;
        Ctx rejected;CtxDesc d{};d.program=&f.builder->program();d.stream=f.compute.value;
        d.residency=f.mover.residency();d.arch_step=step;d.record_passes=false;d.expert_prepare_owner=owner.get();
        d.expert_prepare_limits.max_rows=R;d.expert_prepare_limits.max_experts=E;
        d.expert_prepare_limits.max_top_k=K;d.expert_prepare_limits.max_readback_bytes=68;
        need(rejected.init(d)==RAD_E_UNSUPPORTED&&bodies()==before_bodies&&collectors()==before_collectors&&
             owner->stats().prepares==0&&f.mover.residency()->epoch()==epoch&&
             f.source->bridge->stats().provider_reads==reads,
             "collector/Native unsupported admission executed body/collector/Owner/source/publication");
        // This is the actual existing refusal, not a fake valid combined frame.
        // No context was accepted/bound, so the ordinary owner destructor closes it.
    }
    void policy_outside_prepare(const std::string& home){++cases;
        Fixture f(home,true);const auto caller=f.caller;
        const int status=f.ctx.expert_prepare(nullptr);
        need(status==RAD_E_STATE&&f.step_entries==0&&f.ctx.expert_prepare_copies()==0&&f.caller==caller,
             "direct inactive Native prepare did work or failed to latch");
        terminal_guards(f,status);
    }
    void policy_diagnostic_oom(const std::string& home){++cases;
        Fixture f(home,true);f.mode=2;f.source->fail_acquire_and_next_allocation=true;
        const auto failures=allocation_fault::failures;const auto caller=f.caller;
        const auto epoch=f.mover.residency()->epoch();const int status=f.run();
        need(!allocation_fault::next&&allocation_fault::failures==failures+1&&status==RAD_E_NOMEM&&
             f.ctx.terminal_status()==RAD_E_NOMEM&&f.caller==caller&&f.mover.residency()->epoch()==epoch&&
             f.owner->stats().held_leases==0,"diagnostic OOM catch lost permanent Native error/published output");
        need(std::strstr(f.ctx.step_error(),"current expert prepare owner refused before GU")!=nullptr,
             "allocation-free fallback diagnostic not retained");
        terminal_guards(f,status);
    }
    void policy_prior_error(const std::string& home,bool owned){++cases;
        Fixture f(home,owned);f.step_mode=1;const auto caller=f.caller;
        const int status=f.run();
        need(status==RAD_E_SHAPE&&f.late_prepare_status==status&&f.ctx.expert_prepare_copies()==0&&
             f.caller==caller&&std::string(f.ctx.step_error())=="first error before Native prepare",
             "prior step failure did not guard Native/no-owner prepare before work");
        terminal_guards(f,status);
    }
    void reporting(){++cases;
        using namespace rad::server;
        static_assert(kLiveTiers==kTierCount);
        LiveSnapshot snapshot;auto& plane=snapshot.experts;
        plane.n_layers=1;plane.n_experts=3;plane.n_units=3;plane.total_bytes=96;
        plane.units[int(Tier::VRAM)]=1;plane.bytes[int(Tier::VRAM)]=32;
        plane.units[int(Tier::NativeStore)]=2;plane.bytes[int(Tier::NativeStore)]=64;
        plane.layer_units={3};plane.layer_resident={1};
        plane.tier={uint8_t(Tier::VRAM),uint8_t(Tier::NativeStore),uint8_t(Tier::NativeStore)};
        LiveView view([](LiveSnapshot&){});
        const auto line=view.render_line(snapshot),screen=view.render_screen(snapshot);
        need(line.find("native-store=2")!=std::string::npos&&line.find("ssd=0")!=std::string::npos,
             "native-store count folded into original SSD tier");
        need(screen.find("native-store")!=std::string::npos&&screen.find("\033[94mN")!=std::string::npos,
             "native-store terminal name/color/glyph missing");
    }
}
int main(int argc,char** argv){
    try{
        need(argc==3&&!std::strcmp(argv[1],"--radiance-home"),
             "usage: native_backed_plan_test --radiance-home MATCHING_B2_HOST_HOME");
        need(device_is_host(),"HOST-only checker refuses DEVICE backend");
        need(rad_core_model_failure_policy_version()==RAD_MODEL_FAILURE_POLICY_VERSION,
             "matching Core+Engine permanent-policy capability required");
        allocation_hook_contract();
        success(argv[2]);growth(argv[2]);admission(argv[2]);pools_and_placement(argv[2]);
        owner_admission(argv[2]);copy_failure(argv[2]);source_failure(argv[2]);writer(argv[2]);
        late_failure(argv[2]);withdraw(argv[2]);failed_bringup(argv[2]);reporting();
        mutable_authority(argv[2]);landing_lifetime(argv[2]);
        policy_begin_refusal(argv[2]);policy_outside_prepare(argv[2]);policy_diagnostic_oom(argv[2]);
        policy_prior_error(argv[2],true);policy_prior_error(argv[2],false);
        legacy_begin_refusal(argv[2]);collector_owner_refusal(argv[2],true);collector_owner_refusal(argv[2],false);
        legacy_first_error(argv[2]);
        need(cases==23,"14 preserved Native groups plus5 policy,2 Legacy,2 combined-admission groups");
        std::cout<<"PASS "<<cases<<" actual HOST unified strict-policy Native plan/allocation/owner cases, "
                 <<checks<<" checks; no Engine Loader/production/full-model/TP/GPU claim\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
