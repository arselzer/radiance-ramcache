#pragma once
// Actual tiny AQL MoeFP8 current handoff -> S1/cache -> Mover/NativeBackingOwner -> original DEVICE readers.
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
#include "device/device.h"
#include <bit>
#include <cmath>
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

namespace es=rad::expert_store_candidate;
namespace sc=glm53::sharded;
namespace fs=std::filesystem;
namespace {
    constexpr int64_t H=128,E=4,K=2,T=5,R=2;
    size_t cases=0,checks=0;
    void need(bool v,const char* why){++checks;
        if(!v)throw std::runtime_error(why);
    }
    void need(bool v,const std::string& why){need(v,why.c_str());}
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
    [[noreturn]] void contain(const char* why){std::fprintf(stderr,"Native DEVICE checker fail-stop: %s; no ordinary live-backing teardown\n",why);std::fflush(stderr);std::_Exit(3);}
    struct Stream {RadStream value=nullptr;
        Stream(){good(rad_stream_create(&value,0),"actual AQL compute stream");
        }~Stream(){if(value){if(rad_stream_sync(value)<0)contain("compute stream did not drain");
                rad_stream_destroy(value);
    }}};
    struct Stage {
        RadStream stream;void* allocation=nullptr;uint8_t* host=nullptr;uint8_t* device=nullptr;
        static constexpr int64_t bytes=65536,allocation_bytes=bytes+8192;
        explicit Stage(RadStream st):stream(st){allocation=rad_dev_alloc(allocation_bytes,RAD_MEM_HOST_MAPPED);need(allocation,"charged mapped DEVICE staging");
            host=static_cast<uint8_t*>(rad_dev_host_ptr(allocation));device=static_cast<uint8_t*>(rad_dev_device_ptr(allocation));
            if(!host||!device||uintptr_t(host)%4096||uintptr_t(device)%4096){rad_dev_free(allocation,RAD_MEM_HOST_MAPPED);allocation=nullptr;throw std::runtime_error("real mapped allocation views/alignment");}
            std::memset(host,0xbd,allocation_bytes);host+=4096;device+=4096;rad_dev_host_wrote();
        }
        ~Stage(){if(allocation){if(rad_stream_sync(stream)<0)contain("staging reader did not drain");rad_dev_free(allocation,RAD_MEM_HOST_MAPPED);}}
        void guards(){need(std::all_of(host-4096,host,[](uint8_t b){return b==0xbd;})&&std::all_of(host+bytes,host+bytes+4096,[](uint8_t b){return b==0xbd;}),"mapped staging guards");}
        void upload(void* target,std::span<const std::byte> source){need(target&&source.size()<=bytes,"bounded H2D staging span");guards();std::memcpy(host,source.data(),source.size());rad_dev_host_wrote();
            const int copied=rad_memcpy_async(target,device,int64_t(source.size()),stream);if(rad_stream_sync(stream)<0){contain("H2D staging copy did not drain");}
            good(copied,"actual H2D bootstrap/input copy");guards();}
        std::vector<std::byte> read(void* source,size_t n){need(source&&n<=bytes,"bounded DEVICE readback");const int copied=rad_memcpy_async(device,source,int64_t(n),stream);
            if(rad_stream_sync(stream)<0){contain("DEVICE readback did not drain");}
            good(copied,"actual DEVICE completed readback");guards();return {reinterpret_cast<std::byte*>(host),reinterpret_cast<std::byte*>(host)+n};}
    };
    unsigned native_code(int layer,int expert,int64_t n,int64_t k){uint32_t v=uint32_t(expert+1)*0x9e3779b9u^uint32_t(n+5)*0x85ebca6bu^uint32_t(k+11)*0xc2b2ae35u^uint32_t(layer)*97u;v^=v>>16;v*=0x7feb352du;v^=v>>15;return unsigned(1+v%7u)|unsigned(((v>>13)&1u)<<3);}
    int native_exponent(int layer,int expert,int64_t n,int64_t group){return 124+(layer+expert+int(2*n)+int(group))%3;}
    using Source=glm53::p3::ShardedNativeSource;
    struct CountingAllocator final:MemAllocator {
        struct Request{void* ptr;int64_t bytes,physical;int kind;};
        std::vector<Request> live;uint64_t requested=0,physical=0,calls=0;
        int64_t fail_at=-1;
        void* alloc(int64_t bytes,int kind)override{
            ++calls;if(int64_t(calls)==fail_at)return nullptr;
            if(bytes<=0)return nullptr;
            const int64_t n=align_up(bytes,int64_t(RAD_ALIGN_POOL));
            void* p=rad_device_allocator().alloc(n,kind);
            if(p){
                const bool mapped=kind==RAD_MEM_HOST_MAPPED||kind==RAD_MEM_HOST_PINNED;
                if(uintptr_t(p)%4096||(mapped&&(!rad_dev_host_ptr(p)||rad_dev_host_ptr(p)!=p||!rad_dev_device_ptr(p)||uintptr_t(rad_dev_device_ptr(p))%4096))){rad_device_allocator().free(p,kind);return nullptr;}
                try{live.push_back({p,bytes,n,kind});}
                catch(...){rad_device_allocator().free(p,kind);return nullptr;}
                requested+=uint64_t(bytes);physical+=uint64_t(n);
            }
            return p;
        }
        void free(void* p,int kind)override{
            const auto i=std::find_if(live.begin(),live.end(),[&](const auto& x){return x.ptr==p;});
            if(i==live.end())std::abort();
            live.erase(i);rad_device_allocator().free(p,kind);
        }
        const char* name()const override{return "counted actual AQL pool allocator";}
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
        static constexpr uint64_t strict_extra=(2u<<20); // decorator/counters/tiny terminal snapshots
        static constexpr uint64_t declared_upper=strict_extra+(4u+4u+4u+24u+4u+2u)*(1u<<20)+2*65536+163840;
        static_assert(declared_upper<=admitted);
        std::shared_ptr<Source> source;
        std::unique_ptr<NativeBackingOwner> owner;
        uint64_t original_expert_reads=0;
        std::array<uint16_t,T*H> caller{};
        rad_buf bad_dn=0;
        bool fail_dn=false;
        int mode=0;uint64_t caller_version=0,step_entries=0;
        std::unique_ptr<Stage> staging,readback,snapshots;
        struct SliceSnapshot {int64_t first=0,rows=0;size_t sorted=0,gu=0,q=0,scale=0,dn=0,gather=0,input_q=0,input_s=0;};
        std::array<SliceSnapshot,(T+R-1)/R> slices{};size_t n_slices=0,snapshot_used=0;
        std::array<uint16_t,T*H> original_input{};
        std::array<uint8_t,T*H> original_q{};std::array<float,T> original_scale{};
        std::array<int32_t,T*K> expected_ids{};std::array<float,T*K> expected_weights{};
        bool learned=false;
        Ctx ctx;
        Fixture(const std::string& home,const std::string& plugin):names(""){
            constexpr bool owned=true;constexpr uint64_t cold=98304,hot=65536;constexpr int layers=8;
            need(!device_is_host()&&!std::strcmp(device_backend_name(),"aql"),"TinyAqlV1 requires actual AQL");
            need(native_device_owner_profile_version()==kNativeDeviceOwnerProfileVersion,"matching tiny AQL Core version");
            good(registry.load_plugin(plugin,0),"actual unchanged DEVICE FFN/router rows");
            good(registry.load_plugin(home+"/kernels/libref.so",1),"actual matching HOST schema module");
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
            // This fixture owns ALL T external input rows before any sliced pass. Their
            // lifetime must cover every GU/GQ/DN/gather slice; TRANSIENT one-GU
            // storage may otherwise be reused by the down-input quantizer.
            w.h.q=buf("input.q",RAD_F8E4M3,{T,H});
            w.h.s=buf("input.s",RAD_F32,{T,1});
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
            s.cfg.swiglu_quant=1;
            s.cfg.route_prepare=1;
            for(int q=0;q<2;++q){s.cfg.ff_lo[q]=0;
                s.cfg.ff_hi[q]=H;
            }
            moe.ext_in=true;
            good(moe.declare(builder.get(),names,g,l,w,s,false,false),"actual TP1 MXFP4 MoeFP8");
            }
            active_layer=0;
            good(builder->finish(),"actual DEVICE graph finish");
            auto& program=builder->program();
            need(program.arena_bytes>=0&&program.arena_bytes<4*(1<<20)&&program.host_arena_bytes>=0&&program.host_arena_bytes<4*(1<<20)&&program.scratch_bytes>=0&&program.scratch_bytes<4*(1<<20)&&
                 program.arena_bytes+program.host_arena_bytes+program.scratch_bytes<4*(1<<20),
                 "Core arena/scratch quote exceeds admitted profile before allocation");
            Config config;
            config.placement="all_vram";
            config.deterministic=true;
            PlannerInput pi{&program,&config,0,1,nullptr,0,false,false};
            // Truthful DEVICE/checkpoint-style metadata. Original affine source encoding
            // is resolved by the actual DEVICE reader callback; no dense fragment relayout.
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
                NativeInventoryLimits limits;limits.backend_profile=NativeBackendProfile::TinyAqlV1;limits.max_rows=R;limits.top_k=K;
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
            staging=std::make_unique<Stage>(mover.transfer_stream());readback=std::make_unique<Stage>(compute.value);snapshots=std::make_unique<Stage>(compute.value);
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
                need(unit.bytes>0&&unit.bytes<=Stage::bytes,"bounded static bootstrap");
                std::memset(staging->host,0,size_t(unit.bytes));auto* bootstrap= reinterpret_cast<std::byte*>(staging->host);
                if(unit.expert>=0){
                    if(owned)good(source->bootstrap(int32_t(u),{bootstrap,size_t(unit.bytes)}),"private static expert0 bootstrap");
                    else{const auto it=std::find_if(specs.begin(),specs.end(),[&](const auto& x){return x.key().layer==unit.layer&&x.key().expert==unit.expert;});
                        need(it!=specs.end(),"reference unit source");std::vector<std::byte> record;good(reader->read(it->key(),&record),"reference payload bootstrap");
                        for(size_t m=0;m<unit.weights.size();++m)std::memcpy(bootstrap+unit.offset[m],record.data()+unit.offset[m],size_t(program.weights[unit.weights[m]].stored_bytes));}
                    ++original_expert_reads;
                }
                else for(size_t m=0;m<unit.weights.size();++m){const auto h=unit.weights[m];
                    std::memcpy(bootstrap+unit.offset[m],initial[h].data(),initial[h].size());
                }
                rad_dev_host_wrote();const int copied=rad_memcpy_async(base,staging->device,unit.bytes,mover.transfer_stream());
                if(rad_stream_sync(mover.transfer_stream())<0){contain("static/bootstrap upload did not drain");}
                good(copied,"complete actual H2D before Owner fixed-slot snapshot");staging->guards();
            }
            reader.reset();
            if(owned){good(source->finish_bootstrap(),"seal bootstrap and trim original cache payload");
                need(source->accounting().usage().hot_bytes==0&&source->accounting().usage().cold_bytes==0,
                     "bootstrap retained source payload before forward");}
            std::vector<std::vector<std::byte>>().swap(initial);
            for(auto& input:shard_inputs){input.specs={};input.sources=nullptr;}
            for(auto& group:shard_specs)std::vector<es::RecordSpec>().swap(group);
            std::vector<es::RecordSpec>().swap(specs);source_catalog={};
            good(NativeBackingOwner::open(program,plan,mover,source,&owner),"typed actual AQL native owner");
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
            good(ctx.bind_thread(),"AQL rank thread");
            for(const auto& moe:moes){need(moe.gq_exp.clamped_fused&&!moe.gq_exp.op_quant&&!moe.scatter_inv&&!moe.op_sigmoid_scatter&&!moe.op_topk_scatter,"closed six-row DEVICE graph");
                for(const auto op:{moe.glm_router.op_project,moe.glm_router.op_select,moe.op_scatter,moe.op_gu,moe.gq_exp.op,moe.op_dn,moe.op_gather}){
                    good(ctx.set_op_domain(op,RAD_DOMAIN_DEVICE),"actual DEVICE graph row");for(const auto& band:program.ops[op].bands)need(band.dom[RAD_DOMAIN_DEVICE].row&&band.dom[RAD_DOMAIN_DEVICE].row->plugin=="glm53_native_device_rows","no HOST fallback in DEVICE path");}}
            // Truthful external full-pass input storage cannot alias any FFN writer.
            for(const auto& moe:moes){const auto& w=moe.w;
                for(const auto input:{w.h.q,w.h.s}){
                    need(program.buffers[input].decl.kind==RAD_BUF_PERSIST,"external input pair must persist across all slices");
                    const auto a=uintptr_t(ctx.buf_ptr(input));const auto na=uint64_t(program.buffers[input].bytes);
                    for(const auto output:{w.egu,w.eff.q,w.eff.s,w.eff.x,w.edn,w.h.x}){const auto b=uintptr_t(ctx.buf_ptr(output));const auto nb=uint64_t(program.buffers[output].bytes);
                        need(a>=b+nb||b>=a+na,"external input pair aliases a current FFN output");}
                }
            }
            check_guards();
            caller.fill(0x5a5a);
            reset();
        }
        ~Fixture(){if(owner&&owner->quiesce()<0)contain("Owner quiescence failed");
            ctx.shutdown();
            if(owner){check_guards();
                good(owner->close_after_context_shutdown(),"withdraw only after actual Ctx/WTAB teardown");
                owner.reset();
        }}
        void build_source(){
            source_catalog.source_id="synthetic-native-device-owner-tp1-v1";
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
                        for(int64_t row=0;row<rows;++row)for(int64_t col=0;col<row_bytes;++col){const uint8_t b=plane?uint8_t(native_exponent(unit.layer,unit.expert+projection,row,col)):uint8_t(native_code(unit.layer,unit.expert+projection,row,col*2)|(native_code(unit.layer,unit.expert+projection,row,col*2+1)<<4));
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
        void check_guards(){good(mover.native_record_guards_aql(readback->host,readback->device,Stage::bytes,compute.value),"completed DEVICE consumer guards");staging->guards();readback->guards();snapshots->guards();}
        template<class X> void upload(rad_buf target,std::span<const X> x){staging->upload(ctx.buf_ptr(target),std::as_bytes(x));}
        // Only enqueues same-stream snapshots. No extra per-stage or per-slice sync:
        // normal producer/Owner retirement and final run drain remain authoritative.
        size_t save_snapshot(const void* source,size_t bytes){
            const auto offset=size_t(align_up(int64_t(snapshot_used),16));
            need(source&&bytes&&offset<=size_t(Stage::bytes)&&bytes<=size_t(Stage::bytes)-offset,"bounded asynchronous stage snapshot bank");
            good(rad_memcpy_async(snapshots->device+offset,source,int64_t(bytes),compute.value),"enqueue ordered independent stage snapshot");
            snapshot_used=offset+bytes;return offset;
        }
        void capture_slice(int64_t first,int64_t rows){
            need(n_slices<slices.size()&&first>=0&&rows>0&&rows<=R&&first+rows<=T,"exact bounded snapshot slice");
            const auto& w=selected().w;auto& s=slices[n_slices++];s.first=first;s.rows=rows;
            s.sorted=save_snapshot(ctx.buf_ptr(w.sorted),size_t(rows*K*4));
            s.gu=save_snapshot(ctx.buf_ptr(w.egu),size_t(rows*K*2*H*2));
            s.q=save_snapshot(ctx.buf_ptr(w.eff.q),size_t(rows*K*H));s.scale=save_snapshot(ctx.buf_ptr(w.eff.s),size_t(rows*K*4));
            s.dn=save_snapshot(ctx.buf_ptr(w.edn),size_t(rows*K*H*2));
            s.gather=save_snapshot(static_cast<const char*>(ctx.buf_ptr(w.h.x))+first*H*2,size_t(rows*H*2));
            s.input_q=save_snapshot(ctx.buf_ptr(w.h.q),size_t(T*H));s.input_s=save_snapshot(ctx.buf_ptr(w.h.s),size_t(T*4));
        }
        template<class X>std::vector<X> saved(size_t offset,size_t count){
            need(offset<=snapshot_used&&count<=((snapshot_used-offset)/sizeof(X)),"bounded completed snapshot read");
            std::vector<X> out(count);std::memcpy(out.data(),snapshots->host+offset,count*sizeof(X));return out;
        }
        void reset(){
            n_slices=0;snapshot_used=0;slices={};
            // The previous run is fully drained before a successful reset/reuse.
            std::memset(snapshots->host,0xA5,size_t(Stage::bytes));rad_dev_host_wrote();
            for(int64_t row=0;row<T;++row){const int row_mode=mode==3?(row%2?2:0):mode;
                for(int64_t h=0;h<H;++h){const float x=h==0?(row_mode==2?0.f:1.f):h==1?(row_mode==0?0.f:row_mode==1?2.f:1.f):float(int((row+h)%5)-2)*.25f;
                    original_input[size_t(row*H+h)]=rad_f32_to_bf16(x);original_q[size_t(row*H+h)]=rad_f32_to_fp8e4m3(x);}
                original_scale[size_t(row)]=1.f;
                if(learned){const float x=rad_bf16_to_f32(original_input[size_t(row*H)]),z=rad_bf16_to_f32(original_input[size_t(row*H+1)]);
                    const std::array<float,E> logits{4*x,3*x,3*z,z};std::array<float,E> probability{};std::array<int,E> order{0,1,2,3};
                    for(int e=0;e<E;++e)probability[size_t(e)]=1.f/(1.f+std::exp(-logits[size_t(e)]));
                    std::sort(order.begin(),order.end(),[&](int a,int b){return probability[size_t(a)]!=probability[size_t(b)]?probability[size_t(a)]>probability[size_t(b)]:a<b;});
                    const float mult=2.5f/(probability[size_t(order[0])]+probability[size_t(order[1])]+1e-20f);
                    for(int k=0;k<K;++k){expected_ids[size_t(row*K+k)]=order[size_t(k)];expected_weights[size_t(row*K+k)]=probability[size_t(order[size_t(k)])]*mult;}
                }else for(int k=0;k<K;++k){expected_ids[size_t(row*K+k)]=row_mode==2?2+k:k;expected_weights[size_t(row*K+k)]=k?.97353125f:1.52646875f;}
            }
            upload(selected().w.h.x,std::span<const uint16_t>(original_input));upload(selected().w.h.q,std::span<const uint8_t>(original_q));upload(selected().w.h.s,std::span<const float>(original_scale));
            if(!learned){upload(selected().w.ids,std::span<const int32_t>(expected_ids));upload(selected().w.ew,std::span<const float>(expected_weights));}
            std::array<std::byte,R*K*2*H*2> poison;poison.fill(std::byte{0x5a});staging->upload(ctx.buf_ptr(selected().w.egu),poison);
        }
        int run(){
            if(const int first=ctx.terminal_status();first<0)return first;
            reset();RadBatch b{};b.n_tok=T;b.n_seq=1;b.phase=RAD_PHASE_PREFILL;b.step=9;
            active=this;const int status=ctx.run_step(&b);active=nullptr;
            const int complete=ctx.await_device_status();if(rad_stream_sync(compute.value)<0)contain("final DEVICE completion did not drain");
            if(status==RAD_OK&&complete==RAD_OK){const auto bytes=readback->read(ctx.buf_ptr(selected().w.h.x),caller.size()*2);std::memcpy(caller.data(),bytes.data(),bytes.size());++caller_version;}
            return status<0?status:complete;
        }

    };
    void supplied_pass(Fixture& f,RadCtx* c,int64_t first,int64_t rows){const auto& m=f.selected();const auto& w=m.w;
        int32_t* counts=rad_route_counts(c,m.layer,E);const RadOperand ecnt=counts?praw(counts,RAD_I32,E):brows(w.ecnt,E);
        RAD_ISSUE_N(c,m.op_scatter,rows,brow_slice(w.ids,first,rows,K),brows(w.sorted,rows*K),brows(w.eoff,E+1),ecnt);
        const RadExpertPrepare current{m.layer,T,first,rows,ecnt,brows(w.eoff,E+1),brows(w.sorted,rows*K),brow_slice(w.ids,first,rows,K)};
        if(rad_expert_prepare(c,&current)<0)return;
        RAD_ISSUE_N(c,m.op_gu,rows,brow_slice(w.h.q,first,rows,H),brow_slice(w.h.s,first,rows,1),RAD_WTAB(m.w_gu[0],E/2),RAD_WTAB(m.w_gus[0],E/2),brows(w.sorted,rows*K),brows(w.eoff,E+1),RAD_WTAB(m.w_gu[1],E/2),RAD_WTAB(m.w_gus[1],E/2),brows(w.egu,rows*K));
        m.gq_exp.step(c,w.egu,rows*K,0,rows*K);
        RAD_ISSUE_N(c,m.op_dn,rows,brows(w.eff.q,rows*K),brows(w.eff.s,rows*K),RAD_WTAB(m.w_dn[0],E/2),RAD_WTAB(m.w_dns[0],E/2),brows(w.sorted,rows*K),brows(w.eoff,E+1),RAD_WTAB(m.w_dn[1],E/2),RAD_WTAB(m.w_dns[1],E/2),brows(w.edn,rows*K));
        RAD_ISSUE_N(c,m.op_gather,rows,brows(w.edn,rows*K),brow_slice(w.ew,first,rows,K),brows(w.sorted,rows*K),RAD_NONE,RAD_NONE,brow_slice(w.h.x,first,rows,H));
    }
    void step(RadCtx* c,const RadBatch* b){auto& f=*active;++f.step_entries;const auto down=f.selected().w.edn;if(f.fail_dn)f.selected().w.edn=f.bad_dn;
        for(int64_t first=0;first<b->n_tok;first+=R){const auto rows=std::min<int64_t>(R,b->n_tok-first);if(f.learned)f.selected().pass(c,b->n_tok,first,rows);else supplied_pass(f,c,first,rows);if(f.ctx.terminal_status()>=0)f.capture_slice(first,rows);}
        f.selected().w.edn=down;
    }

    std::vector<int32_t> residents(const Fixture& f){
        std::vector<int32_t> result;
        for(size_t u=0;u<f.metadata_plan.units.size();++u)
            if(f.metadata_plan.units[u].expert==0)result.push_back(int32_t(u));
        return result;
    }
}
