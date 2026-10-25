#pragma once
// Actual HOST MoeFP8 current handoff -> S1Reader/cache -> Mover Residency -> native libref MXFP4.
#include "place/native_backing_owner.h"
#include "mem/pools.h"
#include "server/liveview.h"
#include "build/rad_build.h"
#include "plugin/registry.h"
#include "runtime/ctx.h"
#include "rad_block_moe_fp8.h"
#include "native_source.h"
#include <fcntl.h>
#include <sys/stat.h>
#include "rad_plugin.h"
#include "rad_model_failure_policy.h"
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

namespace allocation_fault {extern thread_local int64_t remaining;}
namespace es=rad::expert_store_candidate;
namespace sc=glm53::sharded;
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
    using Source=glm53::p3::ShardedNativeSource;
    // Only the two model-pass allocation fixtures use this boundary decorator.
    // Normal eight cases still bind the actual facade directly to the Owner.
    struct ModelAllocationSource final:NativeRecordSource {
        std::shared_ptr<Source> real;int64_t armed=-1;uint64_t callbacks=0;
        explicit ModelAllocationSource(std::shared_ptr<Source> source):real(std::move(source)){}
        std::span<const NativeRecordDescriptor> catalog()const noexcept override{return real->catalog();}
        int acquire(std::span<const uint32_t> ids,NativeRecordBatch* out)override{
            ++callbacks;if(armed<0)return real->acquire(ids,out);
            const auto position=armed;armed=-1;
            struct Disarm {~Disarm(){allocation_fault::remaining=-1;}} disarm;
            allocation_fault::remaining=position;return real->acquire(ids,out);
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
        std::array<std::vector<es::RecordSpec>,3> shard_specs;
        std::array<sc::ShardInput,3> shard_inputs;
        size_t producer_retained=0,producer_peak=0;
        PayloadTotals before_ctx{},after_ctx{};
        static constexpr uint64_t admitted=64u<<20;
        static constexpr uint64_t strict_extra=32768; // decorator/counters/tiny terminal snapshots
        static constexpr uint64_t declared_upper=strict_extra+(4u+4u+4u+24u+4u+2u)*(1u<<20)+2*65536+163840;
        static_assert(declared_upper<=admitted);
        std::shared_ptr<Source> source;
        std::shared_ptr<ModelAllocationSource> model_allocation_source;
        std::unique_ptr<NativeBackingOwner> owner;
        uint64_t original_expert_reads=0;
        std::array<uint16_t,T*H> caller{};
        rad_buf bad_dn=0;
        bool fail_dn=false;
        int mode=0;uint64_t caller_version=0,step_entries=0;
        Ctx ctx;
        Fixture(const std::string& home,bool owned,uint64_t cold=3*32768,int layers=8,uint64_t hot=65536,bool model_allocation=false,bool raw_session=false):names(""){
            need(device_is_host(),"HOST-only, no device execution permitted");
            good(registry.load_plugin(home+"/kernels/libref.so",0),"actual matching HOST libref");
            RadModelMeta meta{};
            meta.arch_id="synthetic-record-owner";
            meta.name="not-a-model";
            need(layers==8&&cold>0&&cold<=98304&&hot<=65536,"bounded composite64MiB profile before source I/O/allocation");
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
            good(builder->require_permanent_model_failure(),"immutable PermanentV1 before runtime declarations");
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
            good(builder->finish(),"actual HOST graph finish");
            auto& program=builder->program();
            need(program.arena_bytes>=0&&program.arena_bytes<4*(1<<20)&&program.host_arena_bytes>=0&&program.host_arena_bytes<4*(1<<20)&&program.scratch_bytes>=0&&program.scratch_bytes<4*(1<<20)&&
                 program.arena_bytes+program.host_arena_bytes+program.scratch_bytes<4*(1<<20),
                 "Core arena/scratch quote exceeds admitted profile before allocation");
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
            if(owned){
                const auto root=temp.path/"generation";need(fs::create_directory(root),"three-shard root");
                for(const auto& spec:specs)shard_specs[size_t(spec.key().layer%3)].push_back(spec);
                for(size_t n=0;n<3;++n){auto& input=shard_inputs[n];input.name="layer_shard_"+std::to_string(n);
                    input.specs=shard_specs[n];input.sources=&source_catalog;input.limits.max_segment_bytes=65536;
                    es::Receipt ready;good(es::pack(input.specs,source_catalog,(root/input.name).string(),input.limits,&ready),"fresh three-shard native store");
                    input.manifest_checksum=ready.manifest_checksum;input.limits.max_segments=ready.segments;}
                std::string why;const int opened=Source::open(program,metadata_plan,root.string(),shard_inputs,cold,hot,&source,&why);good(opened,why.c_str());
                need(source->catalog().size()==24&&source->bootstrap_count()==8&&source->stats().catalog_records==32,
                     "static bootstrap records leaked into owned NativeRecordSource catalog");
            }else good(es::pack(specs,source_catalog,(temp.path/"store").string(),es::Limits{},&receipt),"reference native store");
            Config base=config;base.vram_kv_mib=1;base.embedding_placement=TablePlacement::Vram;
            vmem_before_pool=rad_dev_vmem_committed();
            if(owned){
                const auto bindings=bound(*source);std::vector<int32_t> residents;
                for(size_t u=0;u<metadata_plan.units.size();++u)
                    if(metadata_plan.units[u].expert==0)residents.push_back(int32_t(u));
                NativeInventoryLimits limits;limits.max_rows=R;limits.top_k=K;
                limits.hot_bytes=hot;limits.cold_bytes=cold;
                limits.max_metadata_bytes=4u<<20;limits.max_weights_bytes=4u<<20;limits.max_device_bytes=4u<<20;
                limits.max_landing_bytes=4u<<20;limits.max_host_bytes=64u<<20;
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
            if(!owned)good(es::Reader::open((temp.path/"store").string(),specs,source_catalog,receipt.manifest_checksum,es::Limits{},&reader),"reference native bootstrap reader");
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
                if(unit.expert>=0){
                    if(owned)good(source->bootstrap(int32_t(u),{base,size_t(unit.bytes)}),"private static expert0 bootstrap");
                    else{const auto it=std::find_if(specs.begin(),specs.end(),[&](const auto& x){return x.key().layer==unit.layer&&x.key().expert==unit.expert;});
                        need(it!=specs.end(),"reference unit source");std::vector<std::byte> record;good(reader->read(it->key(),&record),"reference payload bootstrap");
                        for(size_t m=0;m<unit.weights.size();++m)std::memcpy(base+unit.offset[m],record.data()+unit.offset[m],size_t(program.weights[unit.weights[m]].stored_bytes));}
                    ++original_expert_reads;
                }
                else for(size_t m=0;m<unit.weights.size();++m){const auto h=unit.weights[m];
                    std::memcpy(base+unit.offset[m],initial[h].data(),initial[h].size());
                }
            }
            reader.reset();
            if(owned){good(source->finish_bootstrap(),"seal bootstrap and trim original cache payload");
                need(source->accounting().usage().hot_bytes==0&&source->accounting().usage().cold_bytes==0,
                     "bootstrap retained source payload before forward");}
            std::vector<std::vector<std::byte>>().swap(initial);
            for(auto& input:shard_inputs){input.specs={};input.sources=nullptr;}
            for(auto& group:shard_specs)std::vector<es::RecordSpec>().swap(group);
            std::vector<es::RecordSpec>().swap(specs);source_catalog={};
            // New fence-only fixture: actual source/Pools/Mover/bootstrap above; no Owner/Ctx is opened.
            if(raw_session)return;
            if(owned){
                if(model_allocation)model_allocation_source=std::make_shared<ModelAllocationSource>(source);
                const std::shared_ptr<NativeRecordSource> owner_source=model_allocation_source?
                    std::static_pointer_cast<NativeRecordSource>(model_allocation_source):
                    std::static_pointer_cast<NativeRecordSource>(source);
                good(NativeBackingOwner::open(program,plan,mover,owner_source,&owner),"typed native rank owner");
            }
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
            before_ctx=payload_totals();
            good(ctx.init(d),"actual checked-wait native Ctx");
            need(program.requires_permanent_model_failure()&&ctx.requires_permanent_model_failure()&&
                 !ctx.checks_device_status()&&!ctx.records_passes(),"actual strict TP1 eager zero-collector profile");
            after_ctx=payload_totals();
            for(size_t n=0;n<before_ctx.size();++n)need(after_ctx[n].second-before_ctx[n].second>=0&&
                after_ctx[n].second-before_ctx[n].second<4*(1<<20),"actual Ctx side backing exceeds admitted arena/work bound");
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
            producer_peak=image.capacity();write(temp.path/"source.bin",image);
            std::vector<std::byte>().swap(image);producer_retained=image.capacity();
            for(size_t u=0;u<plan.units.size();++u)if(plan.units[u].expert>=0){es::RecordSpec spec;
                std::string why;
                const auto captured=es::capture_unit(builder->program(),plan,u,source_catalog,&spec,&why);
                good(captured,"actual TP1 ROWparts2/COL source capture, unit "+std::to_string(u)+": "+why);
                need(spec.key().world==1&&spec.key().rank==0&&spec.stride()==32768,"truthful TP1 native class geometry");
                specs.push_back(std::move(spec));
            }
        }
        std::vector<RecordOwnerBinding> bound(Source& from){std::vector<RecordOwnerBinding> result;
            for(size_t u=0;u<plan.units.size();++u)if(plan.units[u].expert>0){const auto i=std::find_if(from.catalog().begin(),from.catalog().end(),[&](const auto& d){return d.layer==plan.units[u].layer&&d.expert==plan.units[u].expert;});
                need(i!=from.catalog().end(),"actual unit bound to immutable native descriptor");
                result.push_back({int32_t(u),uint32_t(i-from.catalog().begin())});
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
        int run(){
            if(const int first=ctx.terminal_status();first<0)return first; // caller preparation guard; direct Core guards tested separately
            reset();
            RadBatch b{};
            b.n_tok=T;
            b.n_seq=1;
            b.phase=RAD_PHASE_PREFILL;
            b.step=9;
            active=this;
            const int status=ctx.run_step(&b);
            active=nullptr;
            const int complete=ctx.await_device_status();
            good(rad_stream_sync(compute.value),"final actual HOST completion");
            if(status==RAD_OK&&complete==RAD_OK){auto* output=static_cast<uint16_t*>(ctx.buf_ptr(selected().w.h.x));
                std::copy(output,output+caller.size(),caller.begin());++caller_version;
            }return status<0?status:complete;
        }
    };
    void step(RadCtx* c,const RadBatch* b){auto& f=*active;++f.step_entries;
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
}
