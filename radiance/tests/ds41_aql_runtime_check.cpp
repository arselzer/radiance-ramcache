/* Explicit, isolated GPU gate. This is deliberately NOT an automatic *_test.cpp. */
#include "build/rad_build.h"
#include "runtime/ctx.h"
#include "runtime/residency.h"
#include "device/device.h"
#include "../libds41/ds41_packed_args.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace rad;
namespace pk=rad::ds41::packed;
namespace {
constexpr int64_t kM=3,kN=2,kK=256,kExperts=2,kGuard=256,kExplicitLimit=1<<20;
constexpr uint8_t kPoison=0xbd;
int64_t live_explicit=0,peak_explicit=0;
size_t checked_launches=0,checked_outputs=0;
void require(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
void check(int status,const char* operation){
    if(status<0)throw std::runtime_error(std::string(operation)+": "+rad_strerror(status));
}
struct Stream {
    RadStream value=nullptr;
    Stream(){check(rad_stream_create(&value,0),"create core stream");require(value!=nullptr,"null core stream");}
    ~Stream(){if(value)rad_stream_destroy(value);}
};
struct Buffer {
    void* allocation=nullptr;uint8_t* accessible=nullptr;int64_t bytes=0;int kind=RAD_MEM_DEVICE;
    Buffer(RadStream stream,int64_t count,int memory=RAD_MEM_DEVICE):bytes(count),kind(memory){
        require(bytes>0&&bytes<=65536,"explicit fixture allocation is not tiny");
        require(live_explicit<=kExplicitLimit-bytes-2*kGuard,"explicit allocation cap exceeded");
        allocation=rad_dev_alloc(bytes+2*kGuard,kind);require(allocation!=nullptr,"tiny allocation failed");
        accessible=static_cast<uint8_t*>(kind==RAD_MEM_HOST_MAPPED?rad_dev_device_ptr(allocation):allocation);
        if(!accessible){rad_dev_free(allocation,kind);allocation=nullptr;throw std::runtime_error("mapped allocation has no device alias");}
        live_explicit+=bytes+2*kGuard;peak_explicit=std::max(peak_explicit,live_explicit);
        const int status=rad_memset_async(accessible,kPoison,bytes+2*kGuard,stream);
        if(status<0){rad_dev_free(allocation,kind);allocation=nullptr;live_explicit-=bytes+2*kGuard;check(status,"initialize guarded allocation");}
    }
    ~Buffer(){if(allocation){rad_dev_free(allocation,kind);live_explicit-=bytes+2*kGuard;}}
    Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    void* data()const{return accessible+kGuard;}
    void put(const void* source,int64_t count,RadStream stream){
        require(source&&count==bytes,"bad tiny upload extent");check(rad_memcpy_async(data(),source,bytes,stream),"upload fixture");
    }
    std::vector<uint8_t> read(RadStream stream)const{
        std::vector<uint8_t> all((size_t)(bytes+2*kGuard));
        check(rad_memcpy_async(all.data(),accessible,(int64_t)all.size(),stream),"read guarded fixture");
        check(rad_stream_sync(stream),"finish readback");
        require(std::all_of(all.begin(),all.begin()+kGuard,[](uint8_t b){return b==kPoison;})&&
                std::all_of(all.end()-kGuard,all.end(),[](uint8_t b){return b==kPoison;}),"device redzone changed");
        return {all.begin()+kGuard,all.end()-kGuard};
    }
};
template<class T>std::vector<T> values(const std::vector<uint8_t>& bytes){
    require(bytes.size()%sizeof(T)==0,"bad typed readback size");std::vector<T> out(bytes.size()/sizeof(T));
    std::memcpy(out.data(),bytes.data(),bytes.size());return out;
}
std::vector<uint8_t> unhex(const std::string& text){
    require(text.size()%2==0&&text.size()<=65536,"invalid bounded fixture hex");
    std::vector<uint8_t> out(text.size()/2);
    auto nibble=[](char c)->unsigned{if(c>='0'&&c<='9')return unsigned(c-'0');if(c>='a'&&c<='f')return unsigned(c-'a'+10);if(c>='A'&&c<='F')return unsigned(c-'A'+10);throw std::runtime_error("invalid fixture hex digit");};
    for(size_t i=0;i<out.size();++i)out[i]=uint8_t((nibble(text[2*i])<<4)|nibble(text[2*i+1]));
    return out;
}
struct Golden {std::string format;std::vector<uint8_t> packed;std::vector<float> logical;};
std::vector<Golden> read_goldens(const std::string& path){
    require(std::filesystem::file_size(path)<=2u*1024u*1024u,"golden JSON exceeds bounded metadata cap");
    std::ifstream file(path);require(bool(file),"cannot open tiny golden fixture");
    const auto json=nlohmann::json::parse(file);
    std::vector<Golden> out;
    for(const char* name:{"iq2_xxs","iq2_xs","iq2_s","iq3_xxs","q5_K","q8_0"}){
        bool found=false;
        for(const auto& entry:json.at("formats"))if(entry.at("name")==name){
            Golden g;g.format=name;g.packed=unhex(entry.at("packed_hex").get<std::string>());
            g.logical=values<float>(unhex(entry.at("f32_hex").get<std::string>()));
            require(g.logical.size()==size_t(2*kN*kK),"expected four bounded golden rows");
            require(std::all_of(g.logical.begin(),g.logical.end(),[](float f){return std::isfinite(f);}),"nonfinite golden weights");
            out.push_back(std::move(g));found=true;break;
        }
        require(found,std::string("missing native format fixture: ")+name);
    }
    return out;
}
struct Residency final:ResidencyTable {
    std::vector<WeightSlot> slots;std::vector<rad_weight> dirty;uint64_t current_epoch=1;
    const WeightSlot* lookup(rad_weight h)const override{return h<slots.size()?&slots[h]:nullptr;}
    uint64_t epoch()const override{return current_epoch;}
    const std::vector<rad_weight>& changes()const override{return dirty;}
    void clear_changes()override{dirty.clear();}
    void move(rad_weight h,void* pointer){
        slots[h].ptr=pointer;++slots[h].generation;++current_epoch;dirty.push_back(h);
    }
};
struct Fixture;
Fixture* active=nullptr;
int real_launch_capture(const RadArgs*,RadStream);
void step(RadCtx*,const RadBatch*);
struct WrappedRow {RadKernelInfo info{};KernelRow row;};
struct Fixture {
    Registry& registry;const Golden& golden;bool routed=false;rad_op op=0;
    std::unique_ptr<Builder> builder;std::vector<rad_weight> handles;
    std::array<std::vector<uint8_t>,2> native;
    std::array<std::vector<float>,2> logical,current_logical;
    std::vector<std::unique_ptr<Buffer>> weight_storage;
    std::vector<std::vector<uint8_t>> original_weights;
    std::array<Buffer*,2> current_weight{};
    Residency residency;std::deque<WrappedRow> wrappers;RadLaunchFn actual_launch=nullptr;
    std::vector<RadTensor> observed;RadStream observed_stream=nullptr;
    std::unique_ptr<Buffer> input,output,expert_ids,row_ids,status;
    std::vector<float> host_input;
    std::array<int32_t,3> host_experts{1,0,1},host_rows{0,1,2};
    std::vector<RadOperand> operands;
    Ctx ctx; // Must drain/destroy before any buffers or copied kernel rows disappear.
    Fixture(Registry& reg,const Golden& g,bool table,RadStream stream,bool tapes):registry(reg),golden(g),routed(table){
        const auto* imp=registry.checkpoint_importer("gguf",golden.format);require(imp!=nullptr,"actual GGUF importer is absent");
        RadCheckpointImportWeight iw{"aql-fixture",golden.format.c_str(),kN,kK};
        WeightSource source{};source.rank=2;source.shape[0]=kN;source.shape[1]=kK;
        check(imp->info->encoding(&iw,&source.enc),"actual importer metadata");
        RadModelMeta meta{};meta.arch_id="ds41-aql-fixture";meta.quant="native";meta.name="synthetic";
        RadBuildCtx bc{};bc.rank=0;bc.world_size=1;bc.max_tok=kM;bc.max_seqs=1;bc.max_ctx=kM;bc.scope="";
        builder=std::make_unique<Builder>(registry,meta,bc);builder->set_instances(false);
        builder->set_sources([source](const std::string&,const Program&,WeightSource* out){*out=source;return RAD_OK;});
        for(int e=0;e<(routed?2:1);++e){
            RadWeightDecl d{};d.dtype=source.enc.plane[0].dtype;d.rank=2;d.shape[0]=kN;d.shape[1]=kK;
            d.access=RAD_ACCESS_CONDITIONAL;d.group.layer=0;d.group.expert=e;
            const auto handle=builder->decl_weight(("tiny.weight."+std::to_string(e)).c_str(),&d);
            require(handle!=0,"actual Builder refused weight");handles.push_back(handle);
        }
        RadParam params[]{RAD_RANGE("M",1,kM),RAD_INT("N",kN),RAD_INT("K",kK),
            RAD_STR("format",golden.format.c_str()),RAD_STR("a_dtype","f32"),RAD_STR("y_dtype","f32"),
            RAD_INT("R",kM),RAD_INT("n_expert",kExperts)};
        op=builder->decl_op(routed?"ds41_routed_packed":"ds41_gemm_packed",params,routed?8:6,handles.data(),(int)handles.size());
        require(op!=0,"actual Builder refused op");check(builder->finish(),"actual Builder finish");
        auto& program=builder->program();
        require(program.arena_bytes>=0&&program.arena_bytes<=65536&&program.scratch_bytes>=0&&program.scratch_bytes<=65536,
                "program-sized fixture arena unexpectedly large");
        require(program.host_arena_bytes>=0&&program.host_arena_bytes<=65536,"fixture host arena unexpectedly large");
        auto& first=program.ops[op].bands[0].dom[RAD_DOMAIN_DEVICE];
        require(first.row&&first.row->info&&first.row->info->relayout,"real libds41 device row did not resolve");
        actual_launch=first.row->info->launch;require(actual_launch!=nullptr,"real libds41 launch is missing");
        for(int e=0;e<2;++e){
            const auto& weight=program.weights[handles[0]];
            require(weight.stored_bytes>0&&weight.stored_bytes<=4096&&weight.stored_dtype>=RAD_DT_PLUGIN_BASE&&
                    rad_dtype_bits(weight.stored_dtype)==0,"native runtime storage is not bounded/opaque");
            require(golden.packed.size()==size_t(2*weight.stored_bytes),"golden native extent differs from actual layout");
            std::array<std::vector<uint8_t>,RAD_ENC_MAX_PLANES> canonical;
            void* pointers[RAD_ENC_MAX_PLANES]{};
            for(int p=0;p<source.enc.n_planes;++p){
                const auto bytes=rad_enc_plane_bytes(&source.enc.plane[p],kN,kK);
                require(bytes>0&&bytes<=8192,"canonical fixture plane not tiny");
                canonical[(size_t)p].assign(size_t(bytes),0);pointers[p]=canonical[(size_t)p].data();
            }
            check(imp->info->import_rows(&iw,golden.packed.data()+e*weight.stored_bytes,weight.stored_bytes,kN,pointers),"actual block import");
            RadTensor planes[RAD_ENC_MAX_PLANES]{};
            for(int p=0;p<weight.n_sel;++p){
                const int selected=weight.sel[p];auto& t=planes[p];t.data=pointers[selected];
                t.dtype=source.enc.plane[selected].dtype;t.rank=2;t.shape[0]=weight.sel_rows[p];t.shape[1]=weight.sel_cols[p];rad_tensor_pack(&t);
            }
            native[(size_t)e].assign(size_t(weight.stored_bytes),0);
            check(first.row->info->relayout(params,routed?8:6,1,&source.enc,weight.sel,planes,weight.n_sel,
                                          native[(size_t)e].data(),weight.stored_bytes),"actual native relayout");
            require(std::equal(native[(size_t)e].begin(),native[(size_t)e].end(),golden.packed.begin()+e*weight.stored_bytes),"lossless relayout changed original blocks");
            logical[(size_t)e].assign(golden.logical.begin()+e*kN*kK,golden.logical.begin()+(e+1)*kN*kK);
        }
        residency.slots.resize(program.weights.size());
        for(size_t e=0;e<handles.size();++e){
            weight_storage.push_back(std::make_unique<Buffer>(stream,(int64_t)native[e].size()));
            original_weights.push_back(native[e]);
            current_weight[e]=weight_storage.back().get();current_weight[e]->put(native[e].data(),(int64_t)native[e].size(),stream);
            program.weights[handles[e]].ptr=current_weight[e]->data();
            residency.slots[handles[e]]={current_weight[e]->data(),nullptr,1};current_logical[e]=logical[e];
        }
        // Capture the descriptors but delegate every actual launch. Preserve the collector field.
        for(auto& band:program.ops[op].bands){
            auto& resolved=band.dom[RAD_DOMAIN_DEVICE];if(!resolved.row)continue;
            require(resolved.row->info->launch==actual_launch,"unexpected different native launch band");
            wrappers.emplace_back();auto& wrapped=wrappers.back();wrapped.info=*resolved.row->info;
            wrapped.info.launch=real_launch_capture;wrapped.row=*resolved.row;wrapped.row.info=&wrapped.info;resolved.row=&wrapped.row;
        }
        check(rad_stream_sync(stream),"settle tiny weights before Ctx binding");
        CtxDesc cd{};cd.program=&program;cd.device=0;cd.world_size=1;cd.stream=stream;
        cd.arch_step=step;cd.residency=&residency;cd.record_passes=tapes;
        check(ctx.init(cd),"actual Ctx init");check(ctx.bind_thread(),"actual Ctx bind thread");
        require(ctx.stream()==stream,"Ctx did not retain core-owned stream");
        require(ctx.checks_device_status()==routed,"unexpected deferred status contract");
        require(routed?!ctx.records_passes():ctx.records_passes()==tapes,"unexpected recorded-pass policy");
        input=std::make_unique<Buffer>(stream,kM*kK*int64_t(sizeof(float)));
        output=std::make_unique<Buffer>(stream,kM*kN*int64_t(sizeof(float)));
        host_input.resize(size_t(kM*kK));
        for(int64_t r=0;r<kM;++r)for(int64_t k=0;k<kK;++k)host_input[size_t(r*kK+k)]=float((r*37+k*13)%19-9)*0.03125f;
        input->put(host_input.data(),kM*kK*int64_t(sizeof(float)),stream);
        operands={RAD_P_T2(input->data(),RAD_F32,kM,kK),routed?RAD_WTAB(handles[0],kExperts):RAD_W(handles[0])};
        if(routed){
            expert_ids=std::make_unique<Buffer>(stream,kM*4);row_ids=std::make_unique<Buffer>(stream,kM*4);status=std::make_unique<Buffer>(stream,kM*4);
            auto vector=[](void* data){auto operand=RAD_P_T(data,RAD_I32);operand.rows=kM;return operand;};
            operands.push_back(vector(expert_ids->data()));operands.push_back(vector(row_ids->data()));
            operands.push_back(RAD_P_T2(output->data(),RAD_F32,kM,kN));operands.push_back(vector(status->data()));
        }else operands.push_back(RAD_P_T2(output->data(),RAD_F32,kM,kN));
        std::printf("  %s %s: explicit_live=%lld B, program_arena=%lld B, program_scratch=%lld B; no Pools/KV carve\n",
                    golden.format.c_str(),routed?"routed":"dense",(long long)live_explicit,
                    (long long)program.arena_bytes,(long long)program.scratch_bytes);
    }
    std::vector<float> expected()const{
        std::vector<float> out(size_t(kM*kN));
        for(int64_t j=0;j<kM;++j)for(int64_t n=0;n<kN;++n){
            const int e=routed?host_experts[size_t(j)]:0;const int64_t row=routed?host_rows[size_t(j)]:j;
            float sum=0;
            for(int64_t k=0;k<kK;++k){const float product=host_input[size_t(row*kK+k)]*current_logical[size_t(e)][size_t(n*kK+k)];sum=sum+product;}
            out[size_t(j*kN+n)]=sum;
        }
        return out;
    }
    void relocate(int expert,int source,bool mapped){
        require(expert>=0&&size_t(expert)<handles.size()&&source>=0&&source<2,"invalid tiny relocation");
        weight_storage.push_back(std::make_unique<Buffer>(ctx.stream(),(int64_t)native[size_t(source)].size(),mapped?RAD_MEM_HOST_MAPPED:RAD_MEM_DEVICE));
        original_weights.push_back(native[size_t(source)]);
        auto* next=weight_storage.back().get();next->put(native[size_t(source)].data(),(int64_t)native[size_t(source)].size(),ctx.stream());
        check(rad_stream_sync(ctx.stream()),"settle relocated expert");
        residency.move(handles[size_t(expert)],next->data());current_weight[size_t(expert)]=next;current_logical[size_t(expert)]=logical[size_t(source)];
    }
    void run(int ordinal,bool malformed=false){
        const auto want=expected();
        check(rad_memset_async(output->data(),0xd3,output->bytes,ctx.stream()),"poison output");
        if(routed){auto ids=host_experts;if(malformed)ids.back()=int32_t(kExperts);
            expert_ids->put(ids.data(),kM*4,ctx.stream());row_ids->put(host_rows.data(),kM*4,ctx.stream());
            check(rad_memset_async(status->data(),0x7f,status->bytes,ctx.stream()),"poison route status");}
        active=this;RadBatch batch{};batch.phase=RAD_PHASE_PREFILL;batch.n_tok=kM;batch.step=ordinal;
        const int submitted=ctx.run_step(&batch);const int verified=ctx.await_device_status();active=nullptr;
        require(submitted>=0,std::string("actual Ctx issue failed: ")+ctx.step_error());
        if(malformed)require(verified==RAD_E_SHAPE,"deferred collector missed malformed route");else check(verified,"await actual device status");
        check(rad_stream_sync(ctx.stream()),"finish Ctx AQL pass");
        const auto got=values<float>(output->read(ctx.stream()));
        for(size_t i=0;i<want.size();++i){
            if(malformed&&i>=size_t((kM-1)*kN)){require(std::isnan(got[i]),"malformed route output was not NaN");continue;}
            require(std::isfinite(got[i])&&std::abs(got[i]-want[i])<=0.00004f+0.0002f*std::abs(want[i]),"actual Ctx/AQL output differs from independent golden CPU math");++checked_outputs;
        }
        require(!observed.empty()&&observed_stream==ctx.stream(),"real launch was never observed on core stream");
        const auto& t=observed[1];const auto& w=builder->program().weights[handles[0]];
        require(t.dtype==w.stored_dtype&&t.dtype>=RAD_DT_PLUGIN_BASE&&rad_dtype_bits(t.dtype)==0,"Ctx lost private native storage dtype");
        require(t.rank==(routed?3u:2u)&&t.shape[routed?1:0]==kN&&t.shape[routed?2:1]==kK&&t.stride[routed?1:0]==kK,"Ctx changed logical native weight descriptor");
        if(routed){
            require(t.stride[0]==0,"routed operand is not a pointer table");std::array<void*,2> table{};
            check(rad_memcpy_async(table.data(),t.data,int64_t(sizeof table),ctx.stream()),"read actual Ctx weight table");check(rad_stream_sync(ctx.stream()),"finish weight-table readback");
            for(size_t e=0;e<2;++e)require(table[e]==current_weight[e]->data(),"Ctx weight table did not follow residency relocation");
            const auto codes=values<int32_t>(status->read(ctx.stream()));
            for(size_t j=0;j<codes.size();++j)require(codes[j]==(malformed&&j==codes.size()-1?RAD_E_SHAPE:RAD_OK),"explicit route status mismatch");
            require(ctx.pass_counts().played==0&&!ctx.records_passes(),"status-opted graph replayed before validation");
        }else require(t.data==current_weight[0]->data(),"dense native pointer did not follow relocation");
        require(values<float>(input->read(ctx.stream()))==host_input,"read-only input changed");
        for(size_t i=0;i<weight_storage.size();++i)
            require(weight_storage[i]->read(ctx.stream())==original_weights[i],"read-only native weight changed");
    }
};
int real_launch_capture(const RadArgs* args,RadStream stream){
    if(!active||!args||!stream)return RAD_E_STATE;
    active->observed.assign(args->t,args->t+args->n_t);active->observed_stream=stream;++checked_launches;
    return active->actual_launch(args,stream);
}
void step(RadCtx* ctx,const RadBatch*){if(active)(void)static_cast<Ctx*>(ctx)->issue(active->op,active->operands.data(),int(active->operands.size()),kM);}
void usage(){std::puts("ds41-aql-runtime-check --run-gpu --home ISOLATED_BUILD_HOME --fixtures gguf-import.json [--host-kargs]\nTiny actual Builder/Ctx/AQL check only; requires exactly one visible gfx1201. No model/service/Pools.");}
}
int main(int argc,char** argv){
    bool approved=false,host_kargs=false;std::string home,fixtures;
    // No explicit Registry, Ctx, query, stream, allocation or launch before this main gate.
    // Linked HIP fatbinary registration constructors may have run before main; see the note.
    for(int i=1;i<argc;++i){const std::string argument=argv[i];
        if(argument=="--help"||argument=="-h"){usage();return 0;}
        if(argument=="--run-gpu")approved=true;
        else if(argument=="--host-kargs")host_kargs=true;
        else if((argument=="--home"||argument=="--fixtures")&&i+1<argc){(argument=="--home"?home:fixtures)=argv[++i];}
        else {std::fprintf(stderr,"unknown/incomplete option: %s\n",argument.c_str());return 2;}
    }
    if(!approved||home.empty()||fixtures.empty()){usage();std::fputs("Explicit --run-gpu, isolated --home and bounded --fixtures are required.\n",stderr);return 2;}
    try{
        const auto golden=read_goldens(fixtures);
#if defined(RAD_HAVE_HIP) && RAD_HAVE_HIP
        if(host_kargs)check(aql_force_host_kargs(),"select host argument rings");
#else
        (void)host_kargs;throw std::runtime_error("checker must link HIP-enabled rad_core");
#endif
        require(rad_dev_count()==1,"expose exactly one GPU with ROCR_VISIBLE_DEVICES");
        require(std::strcmp(device_backend_name(),"aql")==0&&!device_is_host(),"actual AQL backend required; host fallback is not a pass");
        RadDeviceProps props{};check(rad_dev_props(0,&props),"read selected device");
        require(!props.is_host_backend&&std::strcmp(props.arch,"gfx1201")==0,"checker requires one visible gfx1201");
        check(rad_dev_set(0),"select only visible card");
        std::printf("Explicit tiny check: backend=%s arch=%s, one core stream; no production config/model/service.\n",device_backend_name(),props.arch);
        Registry registry;
        check(registry.load_plugin(home+"/quantizers/libquant.so",0),"load isolated libquant");
        check(registry.load_plugin(home+"/experimental/kernels/libds41.so",0),"load isolated real libds41");
        Stream stream;
        for(const auto& g:golden){
            {Fixture dense(registry,g,false,stream.value,true);dense.run(0);
                const auto before=dense.expected();dense.relocate(0,1,false);require(dense.expected()!=before,"degenerate relocation golden");dense.run(1);
                dense.relocate(0,0,true);for(int i=2;i<9;++i)dense.run(i);
                require(dense.ctx.pass_counts().played>0,"dense graph never reached real AQL replay");}
            {Fixture routed(registry,g,true,stream.value,true);routed.run(0);
                const auto before=routed.expected();routed.relocate(1,0,true);require(routed.expected()!=before,"degenerate routed relocation golden");routed.run(1);
                routed.run(2,true);routed.run(3);}
        }
        check(rad_stream_sync(stream.value),"final AQL stream drain");
        require(live_explicit==0,"tiny explicit allocations were not released");
        std::printf("PASS: six formats, real private bindings, VRAM/mapped relocation, routed status refusal/reset, dense replay; %zu delegated launches, %zu numeric outputs; explicit_peak=%lld B.\n",checked_launches,checked_outputs,(long long)peak_explicit);
        std::puts("Receipt excludes core stream/runtime overhead (AQL argument ring about 4 MiB and pinned bounce 8 MiB) and tiny Ctx WTAB/status allocations. No full graph/context/performance claim.");
        return 0;
    }catch(const std::exception& error){active=nullptr;std::fprintf(stderr,"FAIL: %s\n",error.what());return 1;}
}
