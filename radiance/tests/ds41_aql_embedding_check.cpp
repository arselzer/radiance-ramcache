/* Explicit tiny manual check, NOT an automatic *_test.cpp. Actual unmodified embedding row:
 * GGML recorded blocks -> real importer -> Builder private layout -> Ctx/rad_issue/AQL. */
#include "build/rad_build.h"
#include "runtime/ctx.h"
#include "runtime/residency.h"
#include "device/device.h"
#include "../libds41/ds41_embedding_args.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace rad;
namespace e=rad::ds41::embedding;
namespace pk=rad::ds41::packed;
namespace {
constexpr int64_t kM=5,kN=4,kK=256,kGuard=256,kIdGuard=64,kExplicitLimit=1<<20,kArenaLimit=128<<10;
constexpr uint8_t kPoison=0xbd,kOutputPoison=0xd3;
int64_t live_bytes=0,peak_bytes=0;size_t good_passes=0,refused_passes=0,checked_values=0;
void require(bool okay,const std::string& why){if(!okay)throw std::runtime_error(why);}
void check(int status,const char* operation){if(status<0)throw std::runtime_error(std::string(operation)+": "+rad_strerror(status));}
void charge(int64_t count){require(count>0&&count<=kExplicitLimit&&live_bytes<=kExplicitLimit-count,"aggregate tiny data/arena cap exceeded");live_bytes+=count;peak_bytes=std::max(peak_bytes,live_bytes);}
struct Stream {
    RadStream value=nullptr;
    Stream(){check(rad_stream_create(&value,0),"create one core stream");require(value!=nullptr,"null core stream");}
    ~Stream(){if(value)rad_stream_destroy(value);}
};
struct Charge {
    int64_t count=0;
    void take(int64_t n){require(count==0,"duplicate arena charge");charge(n);count=n;}
    ~Charge(){live_bytes-=count;}
};
struct WeightStorage {
    RadStream stream;void* allocation=nullptr;uint8_t* device=nullptr;int64_t bytes=0;int kind=RAD_MEM_DEVICE;
    WeightStorage(RadStream s,int64_t n,bool mapped):stream(s),bytes(n),kind(mapped?RAD_MEM_HOST_MAPPED:RAD_MEM_DEVICE){
        require(n>0&&n<=65536,"native fixture record is not tiny");charge(n+2*kGuard);
        allocation=rad_dev_alloc(n+2*kGuard,kind);
        if(!allocation){live_bytes-=n+2*kGuard;throw std::runtime_error("native fixture allocation failed");}
        device=static_cast<uint8_t*>(mapped?rad_dev_device_ptr(allocation):allocation);
        if(!device){rad_dev_free(allocation,kind);allocation=nullptr;live_bytes-=n+2*kGuard;throw std::runtime_error("mapped native fixture has no device alias");}
        const int initialized=rad_memset_async(device,kPoison,n+2*kGuard,stream);
        if(initialized<0){
            if(rad_stream_sync(stream)==RAD_OK){rad_dev_free(allocation,kind);live_bytes-=n+2*kGuard;}
            // An unproven reader keeps storage until process exit on this fatal path.
            allocation=nullptr;device=nullptr;check(initialized,"poison native allocation and redzones");
        }
    }
    WeightStorage(const WeightStorage&)=delete;WeightStorage& operator=(const WeightStorage&)=delete;
    ~WeightStorage(){if(allocation){if(rad_stream_sync(stream)<0){std::fputs("failed native fixture drain; retain storage until process exit\n",stderr);return;}
        rad_dev_free(allocation,kind);live_bytes-=bytes+2*kGuard;}}
    void* data()const{return device+kGuard;}
    void put(const std::vector<uint8_t>& native){require(native.size()==size_t(bytes),"wrong native record size");check(rad_memcpy_async(data(),native.data(),bytes,stream),"upload native fixture");}
    std::vector<uint8_t> read()const{
        std::vector<uint8_t> all(size_t(bytes+2*kGuard));check(rad_memcpy_async(all.data(),device,int64_t(all.size()),stream),"read native fixture and redzones");check(rad_stream_sync(stream),"finish native fixture readback");
        require(std::all_of(all.begin(),all.begin()+kGuard,[](uint8_t v){return v==kPoison;})&&
                std::all_of(all.end()-kGuard,all.end(),[](uint8_t v){return v==kPoison;}),"native weight redzone changed");
        return {all.begin()+kGuard,all.end()-kGuard};
    }
};
std::vector<uint8_t> unhex(const std::string& input){
    require(input.size()%2==0&&input.size()<=65536,"invalid bounded fixture hex extent");std::vector<uint8_t> out(input.size()/2);
    auto nibble=[](char c)->uint8_t{if(c>='0'&&c<='9')return uint8_t(c-'0');if(c>='a'&&c<='f')return uint8_t(c-'a'+10);if(c>='A'&&c<='F')return uint8_t(c-'A'+10);throw std::runtime_error("invalid fixture hex digit");};
    for(size_t i=0;i<out.size();++i)out[i]=uint8_t((nibble(input[2*i])<<4)|nibble(input[2*i+1]));
    return out;
}
struct Golden {std::string format;std::vector<uint8_t> raw;std::vector<float> values;};
std::vector<Golden> goldens(const std::string& path){
    require(std::filesystem::file_size(path)<=2*1024*1024,"golden metadata exceeds2MiB cap");std::ifstream file(path);require(bool(file),"cannot open golden fixture");
    const auto json=nlohmann::json::parse(file);std::vector<Golden> out;
    for(const char* format:{"iq2_xxs","iq2_xs","iq2_s","iq3_xxs","q5_K","q8_0"}){
        bool found=false;for(const auto& entry:json.at("formats"))if(entry.at("name")==format){
            require(entry.at("rows")==kN&&entry.at("cols")==kK,"golden geometry must be four256-wide rows");
            Golden g;g.format=format;g.raw=unhex(entry.at("packed_hex").get<std::string>());const auto raw_values=unhex(entry.at("f32_hex").get<std::string>());
            require(raw_values.size()==size_t(kN*kK*4),"golden F32 extent mismatch");g.values.resize(size_t(kN*kK));std::memcpy(g.values.data(),raw_values.data(),raw_values.size());
            require(std::all_of(g.values.begin(),g.values.end(),[](float v){return std::isfinite(v);}),"golden values must be finite");out.push_back(std::move(g));found=true;break;
        }require(found,std::string("missing native golden format ")+format);
    }return out;
}
struct Residency final:ResidencyTable {
    std::vector<WeightSlot> slots;std::vector<rad_weight> changed;uint64_t generation=1;
    const WeightSlot* lookup(rad_weight h)const override{return h<slots.size()?&slots[h]:nullptr;}
    uint64_t epoch()const override{return generation;}
    const std::vector<rad_weight>& changes()const override{return changed;}
    void clear_changes()override{changed.clear();}
    void move(rad_weight h,void* pointer){slots[h].ptr=pointer;++slots[h].generation;++generation;changed.push_back(h);}
};
struct Fixture;Fixture* active=nullptr;void step(RadCtx*,const RadBatch*);
struct Fixture {
    Registry& registry;const Golden& golden;RadStream stream;uint32_t dtype;int width=0;
    std::unique_ptr<Builder> builder;rad_weight weight=0;rad_buf ids=0,output=0;rad_op op=0;
    std::array<std::vector<uint8_t>,2> native;std::vector<std::unique_ptr<WeightStorage>> stores;std::vector<int> store_variant;
    WeightStorage* current=nullptr;int variant=0;Residency residency;std::array<RadOperand,3> operands{};
    Charge arena_charge;Ctx ctx; // Ctx drains/destroys BEFORE arena charge and explicit storage.
    Fixture(Registry& r,const Golden& g,RadStream s,uint32_t type):registry(r),golden(g),stream(s),dtype(type){
        width=rad_dtype_bits(dtype)/8;require(width==2||width==4,"wrong scalar output type");
        const auto* importer=registry.checkpoint_importer("gguf",golden.format);require(importer!=nullptr,"real libquant importer absent");
        RadCheckpointImportWeight imported_weight{"embedding-golden",golden.format.c_str(),kN,kK};WeightSource source{};source.rank=2;source.shape[0]=kN;source.shape[1]=kK;
        check(importer->info->encoding(&imported_weight,&source.enc),"real importer encoding");
        RadModelMeta meta{};meta.arch_id="ds41-aql-embedding-check";meta.name="tiny-synthetic-only";meta.quant="native-GGUF";
        RadBuildCtx build{};build.rank=0;build.world_size=1;build.max_tok=kM;build.max_seqs=1;build.max_ctx=kM;build.scope="";
        builder=std::make_unique<Builder>(registry,meta,build);builder->set_instances(false);
        builder->set_sources([source](const std::string&,const Program&,WeightSource* out){*out=source;return RAD_OK;});
        RadWeightDecl wd{};wd.dtype=source.enc.plane[0].dtype;wd.rank=2;wd.shape[0]=kN;wd.shape[1]=kK;wd.access=RAD_ACCESS_PER_TOKEN;wd.group.layer=-1;wd.group.expert=-1;
        weight=builder->decl_weight("tiny.embedding.weight",&wd);require(weight!=0,"Builder refused native embedding weight");
        RadBufDecl id_decl{};id_decl.dtype=RAD_I32;id_decl.rank=1;id_decl.shape[0]=kM+2*kIdGuard;id_decl.kind=RAD_BUF_TRANSIENT;id_decl.domain=RAD_DOMAIN_DEVICE;
        ids=builder->decl_buffer("guarded.token.ids",&id_decl);
        RadBufDecl y_decl{};y_decl.dtype=dtype;y_decl.rank=2;y_decl.shape[0]=kM+2;y_decl.shape[1]=kK;y_decl.kind=RAD_BUF_TRANSIENT;y_decl.domain=RAD_DOMAIN_DEVICE;
        output=builder->decl_buffer("guarded.embedding.output",&y_decl);require(ids&&output,"Builder refused shaped guarded buffers");
        const RadParam params[]{RAD_RANGE("M",1,kM),RAD_INT("N",kN),RAD_INT("K",kK),RAD_STR("format",golden.format.c_str()),RAD_STR("y_dtype",rad_dtype_name(dtype))};
        op=builder->decl_op("ds41_embedding",params,5,&weight,1);require(op!=0,"Builder refused actual registered embedding op");
        check(rad_op_reads(builder.get(),op,&ids,1),"declare ID read lifetime");check(rad_op_writes(builder.get(),op,&output,1),"declare output write lifetime");
        check(builder->finish(),"finish actual embedding graph");auto& p=builder->program();
        require(p.ops.size()==2&&p.kv_groups.size()==1&&p.host_arena_bytes==0,"fixture unexpectedly has extra ops/KV/host arena");
        require(p.scratch_bytes>=ds41_embedding_scratch(nullptr),"embedding Plan scratch was not reserved");
        const auto& w=p.weights[weight];pk::Type packed_type{};require(pk::parse_type(golden.format.c_str(),&packed_type),"unknown golden native type");
        const int64_t row_bytes=kK/pk::block_values(packed_type)*pk::block_bytes(packed_type);
        require(w.n_sel==source.enc.n_planes&&!w.identity&&w.stored_dtype==pk::storage_dtype(packed_type)&&rad_dtype_bits(w.stored_dtype)==0&&
                w.layout_tag==pk::runtime_tag(packed_type)&&w.stored_bytes==kN*row_bytes&&w.align==256,"Builder lost exact all-plane private embedding layout");
        require(w.decl.rank==2&&w.decl.shape[0]==kN&&w.decl.shape[1]==kK&&w.stored_shape[1]==row_bytes,"Builder logical/native geometry mismatch");
        require(golden.raw.size()==size_t(w.stored_bytes),"golden native byte extent mismatch");
        const auto& resolved=p.ops[op].bands[0].dom[RAD_DOMAIN_DEVICE];require(resolved.row&&resolved.row->info&&resolved.row->plugin=="libds41"&&
            resolved.row->info->launch&&resolved.row->info->layout&&resolved.row->info->relayout&&resolved.row->deferred_status,"real embedding row/layout/collector did not resolve");
        for(const auto& band:p.ops[op].bands)require(band.dom[RAD_DOMAIN_DEVICE].row&&band.dom[RAD_DOMAIN_DEVICE].row->info->launch==resolved.row->info->launch,
            "unexpected embedding launch band change");
        for(int which=0;which<2;++which){
            std::vector<uint8_t> raw(golden.raw.size());for(int64_t row=0;row<kN;++row){const int64_t from=(row+which)%kN;
                std::copy_n(golden.raw.data()+from*row_bytes,size_t(row_bytes),raw.data()+row*row_bytes);}
            std::array<std::vector<uint8_t>,RAD_ENC_MAX_PLANES> canonical;void* plane_data[RAD_ENC_MAX_PLANES]{};int64_t canonical_bytes=0;
            for(int plane=0;plane<source.enc.n_planes;++plane){const auto size=rad_enc_plane_bytes(&source.enc.plane[plane],kN,kK);
                require(size>0&&size<=16384&&canonical_bytes<=65536-size,"canonical fixture planes exceed64KiB");canonical_bytes+=size;
                canonical[size_t(plane)].resize(size_t(size));plane_data[plane]=canonical[size_t(plane)].data();}
            check(importer->info->import_rows(&imported_weight,raw.data(),raw.size(),kN,plane_data),"real golden block import");
            RadTensor planes[RAD_ENC_MAX_PLANES]{};for(int selected=0;selected<w.n_sel;++selected){const int plane=w.sel[selected];auto& t=planes[selected];
                t.data=plane_data[plane];t.dtype=source.enc.plane[plane].dtype;t.rank=2;t.shape[0]=w.sel_rows[selected];t.shape[1]=w.sel_cols[selected];rad_tensor_pack(&t);}
            native[size_t(which)].resize(size_t(w.stored_bytes));check(resolved.row->info->relayout(params,5,e::weight,&source.enc,w.sel,planes,w.n_sel,
                native[size_t(which)].data(),w.stored_bytes),"actual embedding canonical->native relayout");
            require(native[size_t(which)]==raw,"embedding relayout changed golden block bytes");
        }
        stores.push_back(std::make_unique<WeightStorage>(stream,w.stored_bytes,false));store_variant.push_back(0);current=stores.back().get();current->put(native[0]);
        p.weights[weight].ptr=current->data();residency.slots.resize(p.weights.size());residency.slots[weight]={current->data(),nullptr,1};
        const auto arena_bytes=arena_plan_bytes(p,nullptr,nullptr);require(arena_bytes>0&&arena_bytes<=kArenaLimit,"program-sized embedding arena is not tiny");arena_charge.take(arena_bytes);
        check(rad_stream_sync(stream),"settle initial native record");CtxDesc cd{};cd.program=&p;cd.device=0;cd.rank=0;cd.world_size=1;cd.stream=stream;cd.arch_step=step;cd.residency=&residency;cd.record_passes=true;
        check(ctx.init(cd),"initialize actual embedding Ctx");check(ctx.bind_thread(),"bind actual rank thread");
        require(ctx.stream()==stream&&ctx.checks_device_status()&&!ctx.records_passes(),"embedding stream/deferred-status/no-replay contract missing");
        require(rad_buf_ptr(&ctx,ids)&&rad_buf_ptr(&ctx,output)&&ctx.scratch(),"actual shaped arena/scratch binding absent");
        operands={RAD_B_N(ids,kIdGuard,kM),RAD_W(weight),RAD_B_N(output,kK,kM)};
        std::printf("  %s ->%s: native=%lld B, actual Program arena/scratch=%lld B, charged live=%lld B; no launch replacement\n",
            golden.format.c_str(),rad_dtype_name(dtype),(long long)w.stored_bytes,(long long)arena_bytes,(long long)live_bytes);
    }
    std::vector<uint8_t> expected(const std::array<int32_t,kM>& tokens)const{
        std::vector<uint8_t> out(size_t(kM*kK*width));for(int64_t row=0;row<kM;++row)for(int64_t col=0;col<kK;++col)
            rad_store_f32(out.data(),dtype,row*kK+col,golden.values[size_t(((tokens[size_t(row)]+variant)%kN)*kK+col)]);
        return out;
    }
    void relocate(int which,bool mapped){
        require(which>=0&&which<2,"invalid relocation variant");stores.push_back(std::make_unique<WeightStorage>(stream,int64_t(native[size_t(which)].size()),mapped));
        store_variant.push_back(which);auto* next=stores.back().get();next->put(native[size_t(which)]);check(rad_stream_sync(stream),"settle relocated record");
        residency.move(weight,next->data());current=next;variant=which;
    }
    std::vector<uint8_t> read(void* pointer,int64_t count)const{
        require(pointer&&count>0&&count<=65536,"readback extent is not tiny");std::vector<uint8_t> out(size_t(count),0);
        check(rad_memcpy_async(out.data(),pointer,count,stream),"read actual Ctx fixture");check(rad_stream_sync(stream),"finish actual Ctx readback");return out;
    }
    void run(int ordinal,bool bad=false){
        std::array<int32_t,kM> tokens{3,0,1,3,2};if(bad)tokens.back()=int32_t(kN);
        const auto& p=builder->program();const int64_t id_bytes=p.buffers[ids].bytes,y_bytes=p.buffers[output].bytes;
        std::vector<uint8_t> id_before(size_t(id_bytes),kPoison);std::memcpy(id_before.data()+kIdGuard*4,tokens.data(),sizeof tokens);
        check(rad_memcpy_async(rad_buf_ptr(&ctx,ids),id_before.data(),id_bytes,stream),"upload guarded IDs");
        check(rad_memset_async(rad_buf_ptr(&ctx,output),kOutputPoison,y_bytes,stream),"poison entire guarded output");
        active=this;RadBatch batch{};batch.phase=RAD_PHASE_PREFILL;batch.n_tok=kM;batch.n_seq=1;batch.step=ordinal;
        const int submitted=ctx.run_step(&batch),verified=ctx.await_device_status();active=nullptr;
        require(submitted>=0,std::string("real rad_issue refused synchronously: ")+ctx.step_error());
        if(bad){require(verified==RAD_E_SHAPE&&std::string(ctx.step_error()).find("ds41_embedding")!=std::string::npos,"deferred embedding failure attribution missing");++refused_passes;}
        else{check(verified,"await actual embedding payload status");++good_passes;}
        require(!ctx.records_passes()&&ctx.pass_counts().played==0,"status-opted embedding graph replayed");
        const auto got=read(rad_buf_ptr(&ctx,output),y_bytes);const size_t begin=size_t(kK*width),end=size_t((kM+1)*kK*width);
        require(std::all_of(got.begin(),got.begin()+begin,[](uint8_t v){return v==kOutputPoison;})&&
                std::all_of(got.begin()+end,got.end(),[](uint8_t v){return v==kOutputPoison;}),"embedding output guard row changed");
        if(bad)require(std::all_of(got.begin(),got.end(),[](uint8_t v){return v==kOutputPoison;}),"bad LAST ID changed earlier output rows");
        else{const auto want=expected(tokens);for(int64_t i=0;i<kM*kK;++i){const float actual=rad_load_f32(got.data()+begin,dtype,i),reference=rad_load_f32(want.data(),dtype,i);
                require(std::isfinite(actual),"nonfinite embedding output");
                if(dtype==RAD_F32)require(std::abs(actual-reference)<=2e-6f+2e-5f*std::abs(reference),"embedding output differs from independent GGML golden");
                else require(std::memcmp(got.data()+begin+i*width,want.data()+i*width,size_t(width))==0,"embedding output cast differs from independent golden");
                ++checked_values;}}
        require(read(rad_buf_ptr(&ctx,ids),id_bytes)==id_before,"readonly IDs or their guards changed");
        for(size_t i=0;i<stores.size();++i)require(stores[i]->read()==native[size_t(store_variant[i])],"readonly native record changed");
        // ONE op/pass: run_pass resets scratch cursor0, so this is the real descriptor staged
        // by the unmodified launch. Readback observes actual Ctx binding without a launch shim.
        e::Plan plan{};check(rad_memcpy_async(&plan,static_cast<uint8_t*>(ctx.scratch())+rad::ds41::device_plan_offset,sizeof plan,stream),"read actual staged embedding Plan");
        int32_t flag=INT32_MAX;check(rad_memcpy_async(&flag,ctx.scratch(),4,stream),"read actual scalar validation flag");check(rad_stream_sync(stream),"finish Plan/status readback");
        const auto& w=p.weights[weight];const auto& bound=plan.t[e::weight];
        require(plan.rows==kM&&plan.n==kN&&plan.k==kK&&plan.y_dtype==dtype&&plan.status==ctx.scratch(),"actual GPU Plan scalar/flag binding differs");
        require(bound.data==current->data()&&bound.dtype==w.stored_dtype&&rad_dtype_bits(bound.dtype)==0&&bound.rank==2&&
            bound.shape[0]==kN&&bound.shape[1]==kK&&bound.stride[0]==kK&&bound.stride[1]==1,"actual Ctx logical private embedding descriptor/relocation mismatch");
        require(plan.spans[e::weight]==w.stored_bytes&&flag==(bad?RAD_E_SHAPE:RAD_OK),"actual native weight span or independent scalar flag mismatch");
        require(plan.t[e::token_ids].data==static_cast<uint8_t*>(rad_buf_ptr(&ctx,ids))+kIdGuard*4&&plan.t[e::token_ids].shape[0]==kM&&
            plan.t[e::output].data==static_cast<uint8_t*>(rad_buf_ptr(&ctx,output))+kK*width&&plan.t[e::output].shape[0]==kM,
            "actual Ctx operand row slices differ");
        require(rad_weight_ptr(&ctx,weight)==current->data(),"public weight resolution did not follow residency generation");
    }
};
void step(RadCtx* c,const RadBatch*){if(active)(void)rad_issue(c,active->op,active->operands.data(),3,kM);else rad_step_fail(c,"missing embedding checker owner");}
void usage(){std::puts("ds41-aql-embedding-check --run-gpu --home ISOLATED_BUILD_HOME --fixtures gguf-import.json [--host-kargs]\nActual tiny embedding Builder/Ctx/AQL gate; one visible gfx1201; no model/Pools/service.\n");}
}
int main(int argc,char** argv){
    bool approved=false,host_kargs=false;std::string home,fixtures;
    // Main gate precedes explicit queries/allocations. Linked HIP registration may precede main.
    for(int i=1;i<argc;++i){const std::string option=argv[i];if(option=="--help"||option=="-h"){usage();return 0;}
        if(option=="--run-gpu")approved=true;else if(option=="--host-kargs")host_kargs=true;
        else if((option=="--home"||option=="--fixtures")&&i+1<argc)(option=="--home"?home:fixtures)=argv[++i];
        else{std::fprintf(stderr,"unknown/incomplete option: %s\n",option.c_str());return 2;}}
    if(!approved||home.empty()||fixtures.empty()){usage();return 2;}
    try{
        const auto fixture=goldens(fixtures);
#if defined(RAD_HAVE_HIP) && RAD_HAVE_HIP
        if(host_kargs)check(aql_force_host_kargs(),"force host argument storage");
#else
        (void)host_kargs;throw std::runtime_error("requires HIP-enabled rad_core");
#endif
        require(rad_dev_count()==1,"expose exactly one gfx1201 via ROCR_VISIBLE_DEVICES");require(std::strcmp(device_backend_name(),"aql")==0&&!device_is_host(),"AQL backend required, host fallback is not a pass");
        RadDeviceProps props{};check(rad_dev_props(0,&props),"read one visible device");require(!props.is_host_backend&&std::strcmp(props.arch,"gfx1201")==0,"requires exactly one visible gfx1201");check(rad_dev_set(0),"select only visible GPU");
        Registry registry;check(registry.load_plugin(home+"/quantizers/libquant.so",0),"load isolated real importer");check(registry.load_plugin(home+"/experimental/kernels/libds41.so",0),"load isolated unmodified libds41");
        Stream stream;std::printf("Explicit embedding gate backend=%s arch=%s; one core stream, no launch replacement\n",device_backend_name(),props.arch);
        for(const auto& g:fixture)for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16}){
            Fixture f(registry,g,stream.value,dtype);f.run(0);f.run(1,true);f.run(2);
            const std::array<int32_t,kM> tokens{3,0,1,3,2};const auto original=f.expected(tokens);f.relocate(1,false);require(f.expected(tokens)!=original,"degenerate relocated-row golden");f.run(3);
            f.relocate(0,true);f.run(4);f.run(5,true);f.run(6);
        }
        check(rad_stream_sync(stream.value),"final one-stream drain");require(live_bytes==0,"fixture data/arena charge leaked");
        require(good_passes==90&&refused_passes==36&&checked_values==115200,"embedding checker coverage count drifted");
        std::printf("PASS: six native formats x three output types, actual importer/layout/Builder/Ctx/AQL, unmodified launches, VRAM/mapped relocation and bad-last-ID refusal/reset; %zu good/%zu refused passes, %zu independent golden outputs; peak charged data+arena=%lld B.\n",
            good_passes,refused_passes,checked_values,(long long)peak_bytes);
        std::puts("Receipt excludes core stream argument/bounce rings, modules/driver/page metadata and fixed Ctx status resources. No full graph, TP4, model/context or throughput claim.");return 0;
    }catch(const std::exception& error){active=nullptr;std::fprintf(stderr,"FAIL: %s\n",error.what());return 1;}
}
