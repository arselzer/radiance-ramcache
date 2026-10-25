/* Real HOST Builder/Ctx -> registered libds41_ref dense rows. Distinct per-group
 * weights are selected through actual checkpoint maps/import/relayout, not by
 * slicing the opaque private descriptor. Logical TP cases are sequential CPU
 * ownership tests, not communication, actual GPU TP or a complete model graph. */
#include "quant_fixture.h"
#include "build/rad_build.h"
#include "format/checkpoint.h"
#include "format/share.h"
#include "runtime/ctx.h"
#include "../arch/deepseek41/packed_projections.h"
#include "../arch/deepseek41/projection_handoffs.h"
#include "../libds41/ds41_packed_args.h"
#include <nlohmann/json.hpp>
#include <array>
#include <fstream>
#include <map>
#include <memory>
#include <unistd.h>

using namespace rad;
namespace pp=rad::deepseek41::packed;
namespace hand=rad::deepseek41::handoff;
namespace gg=rad::gguf;
namespace pk=rad::ds41::packed;
namespace {
constexpr int64_t N=3,K=256,capacity=3;
void need(bool value,const std::string& message){if(!value)throw std::runtime_error(message);}
void okay(int status,const char* message){need(status==RAD_OK,std::string(message)+": "+rad_strerror(status));}
std::vector<uint8_t> unhex(const std::string& text){
    need(text.size()%2==0&&text.size()<65536,"bounded golden hex required");
    auto digit=[](char c)->unsigned{if(c>='0'&&c<='9')return unsigned(c-'0');if(c>='a'&&c<='f')return unsigned(c-'a'+10);throw std::runtime_error("bad golden hex");};
    std::vector<uint8_t> bytes(text.size()/2);
    for(size_t i=0;i<bytes.size();++i)bytes[i]=uint8_t(digit(text[2*i])*16+digit(text[2*i+1]));
    return bytes;
}
const nlohmann::json& goldens(){
    static const auto j=[] {const auto path=std::string(RAD_GOLDEN_FIXTURES)+"/gguf-import.json";
        need(std::filesystem::file_size(path)<2*1024*1024,"golden size cap");std::ifstream f(path);need(bool(f),"missing independent GGML golden");return nlohmann::json::parse(f);}();return j;
}
const nlohmann::json& golden(gg::Format format){
    for(const auto& f:goldens().at("formats"))if(f.at("name")==gg::geometry(format)->name)return f;
    throw std::runtime_error("missing expected GGML format");
}
struct PackedSource {
    gg::Format format;int64_t groups,row_bytes;std::vector<uint8_t> packed;std::vector<float> expected;
    PackedSource(gg::Format f,int64_t g):format(f),groups(g),row_bytes(K/gg::geometry(f)->values*gg::geometry(f)->bytes){
        const auto& j=golden(f);need(j.at("rows")==4&&j.at("cols")==K,"independent fixture geometry differs");
        const auto bytes=unhex(j.at("packed_hex")),floats=unhex(j.at("f32_hex"));std::array<float,4*K> decoded{};
        need(bytes.size()==size_t(4*row_bytes)&&floats.size()==decoded.size()*4,"golden extents differ");std::memcpy(decoded.data(),floats.data(),floats.size());
        packed.resize(size_t(groups*N*row_bytes));expected.resize(size_t(groups*N*K));
        for(int64_t row=0;row<groups*N;++row){const auto selected=row%4;
            std::memcpy(packed.data()+row*row_bytes,bytes.data()+selected*row_bytes,size_t(row_bytes));
            std::copy_n(decoded.data()+selected*K,K,expected.data()+row*K);
        }
    }
    std::string file()const{
        auto path=radtest::tmp_path("grouped_woa.XXXXXX.gguf");const int fd=mkstemps(path.data(),5);need(fd>=0,"cannot create tiny GGUF");close(fd);
        std::vector<uint8_t> bytes;
        auto put=[&]<class T>(T value){const auto* p=reinterpret_cast<const uint8_t*>(&value);bytes.insert(bytes.end(),p,p+sizeof(value));};
        auto string=[&](const std::string& s){put(uint64_t(s.size()));bytes.insert(bytes.end(),s.begin(),s.end());};
        put(uint32_t(0x46554747));put(uint32_t(3));put(uint64_t(1));put(uint64_t(1));
        string("general.architecture");put(uint32_t(GGUF_STR));string("ds41-grouped-test");
        string("woa.groups");put(uint32_t(3));put(uint64_t(K));put(uint64_t(N));put(uint64_t(groups));
        put(golden(format).at("type").get<uint32_t>());put(uint64_t(0));bytes.resize((bytes.size()+31)/32*32);
        bytes.insert(bytes.end(),packed.begin(),packed.end());std::ofstream out(path,std::ios::binary);out.write(reinterpret_cast<const char*>(bytes.data()),std::streamsize(bytes.size()));out.close();need(bool(out),"tiny GGUF write failed");return path;
    }
};
struct Native {
    std::vector<uint8_t> allocation,before;uint8_t* data=nullptr;
    explicit Native(int64_t bytes):allocation(size_t(bytes)+768,0xB7){need(bytes>0&&bytes<65536,"native weight cap");data=reinterpret_cast<uint8_t*>((uintptr_t(allocation.data())+511)&~uintptr_t(255));}
    void seal(){before=allocation;}
    void check()const{need(before==allocation,"native private weight or redzone changed");}
};
std::function<void(RadCtx*)> active;
void step(RadCtx* c,const RadBatch*){if(active)active(c);else rad_step_fail(c,"missing grouped HOST test step");}
struct Harness {
    const PackedSource& raw;Registry registry;Checkpoint checkpoint;RadBuildCtx context{};
    std::map<std::string,std::string> names;std::unique_ptr<Builder> builder;
    hand::GroupedPlan plan;std::vector<pp::NativeLinear> projections;
    std::map<rad_weight,std::unique_ptr<Native>> storage;
    std::vector<std::pair<const RadKernelInfo*,RadLaunchFn>> launches;
    rad_buf input=0,output=0,wrong=0;int64_t input_pitch=0,output_pitch=0;Ctx ctx;
    Harness(const PackedSource& source,int rank=0,int world=1):raw(source){
        need(device_is_host(),"HOST backend required");okay(registry.load_plugin(home()+"/experimental/kernels/libds41_ref.so",0),"load actual HOST DS41 rows");
        okay(checkpoint.open(raw.file()),"open actual tiny stacked GGUF");checkpoint.set_importers(&qfix::reg());
        context.rank=rank;context.world_size=world;context.max_tok=capacity;context.max_ctx=capacity;context.max_seqs=1;context.scope="";
        RadModelMeta meta{};meta.arch_id="ds41-grouped-runtime-test";meta.name="synthetic-not-model";meta.quant="native-GGUF";meta.n_layers=1;meta.n_embd=K;
        builder=std::make_unique<Builder>(registry,meta,context);builder->set_instances(false);
        // Explicit conversion/import metadata seam. Normal CheckpointSources
        // intentionally refuses raw imported GGUF serving; this does not change it.
        builder->set_sources([this](const std::string& name,const Program& p,WeightSource* out)->int{
            CkptWeight resolved;std::string why;const int rc=ckpt_resolve(checkpoint,p,name,false,&resolved,&why);if(rc!=RAD_OK)return rc;
            if(!resolved.imported||!resolved.block_importer||resolved.rank!=2)return RAD_E_UNSUPPORTED;
            out->enc=resolved.enc;out->rank=resolved.rank;
            for(uint32_t i=0;i<resolved.rank;++i)out->shape[i]=resolved.shape[i];
            return RAD_OK;
        });
        hand::Source source_meta;source_meta.asset_identity="tiny-independent-GGML-blocks";source_meta.tensor_name="woa.groups";
        source_meta.encoding=gg::canonical_encoding(raw.format);source_meta.rank=3;source_meta.shape={raw.groups,N,K};source_meta.layout=hand::StorageLayout::gguf_block_rows;source_meta.payload_bytes=raw.packed.size();
        hand::GroupedSpec request{capacity,raw.groups,K,N,rank,world};request.input_dtype=RAD_F32;request.output_dtype=RAD_F32;
        okay(hand::plan_grouped(source_meta,request,&plan),"checked actual grouped handoff");
        input_pitch=raw.groups*K+7;output_pitch=raw.groups*N+5;
        input=buffer("grouped.input.pitched",input_pitch);output=buffer("grouped.output.pitched",output_pitch);wrong=buffer("common-input.negative.pitched",output_pitch);
        for(int64_t local=0;local<plan.group_count;++local){const auto& group=plan.slices[size_t(local)];const auto alias="woa.group"+std::to_string(group.global_group);
            RadNameMap map{};map.declared=keep(alias);map.mode=RAD_MAP_COPY;map.n_src=1;map.src[0]=keep("woa.groups");map.src_index[0]=int(group.global_group);
            okay(rad_decl_name_map(builder.get(),&map),"declare actual checkpoint group map");
            pp::LinearSpec spec;spec.weight={map.declared,map.declared};spec.rows=N;spec.cols=K;spec.max_rows=capacity;spec.input_dtype=RAD_F32;spec.output_dtype=RAD_F32;
            pp::NativeLinear projection;okay(pp::declare_native_linear(builder.get(),context,spec,&projection),"declare per-group actual dense row");projections.push_back(projection);
            const rad_buf writes[]{output,wrong};okay(rad_op_reads(builder.get(),projection.op,&input,1),"declare grouped input lifetime");okay(rad_op_writes(builder.get(),projection.op,writes,2),"declare grouped outputs lifetime");
        }
        finish();
    }
    const char* keep(const std::string& name){return names.emplace(name,name).first->second.c_str();}
    rad_buf buffer(const char* name,int64_t pitch){RadBufDecl d{};d.dtype=RAD_F32;d.kind=RAD_BUF_PERSIST;d.domain=RAD_DOMAIN_HOST;d.rank=2;d.shape[0]=capacity;d.shape[1]=pitch;const auto b=rad_decl_buffer(builder.get(),name,&d);need(b!=0,"declare actual pitched arena buffer");return b;}
    void finish(){
        okay(builder->finish(),"finish actual grouped Builder");auto& p=builder->program();
        need(arena_plan_bytes(p,nullptr,nullptr)+align_up(arena_host_bytes(p),RAD_ALIGN_UNIT)<=1024*1024,"tiny actual arena exceeds1MiB");
        const auto* original=checkpoint.find("woa.groups");need(original&&original->shape==std::vector<int64_t>{raw.groups,N,K},"actual original source shape");
        for(size_t local=0;local<projections.size();++local){auto& w=p.weights[projections[local].weight];const auto& group=plan.slices[local];
            CkptWeight source;std::string why;okay(ckpt_resolve(checkpoint,p,w.source,false,&source,&why),"actual group checkpoint resolve");
            need(source.pieces.size()==1&&source.rows==N&&source.cols==K&&source.rank==2&&source.imported,"wrong actual group source region");
            need(source.pieces[0]->data==static_cast<const uint8_t*>(original->data)+group.original_block_byte_offset&&uint64_t(source.pieces[0]->bytes)==group.original_block_bytes,"checkpoint group slice is not checked original-block extent");
            std::vector<std::vector<uint8_t>> planes;okay(ckpt_import_band(source,0,N,&planes,&why),"real grouped source canonical import");
            PlaneShare shares[RAD_ENC_MAX_PLANES]{};okay(weight_share(w,source.enc,source.rank,source.shape,context.world_size,context.rank,shares,&why),"real Core source share");
            RadTensor tensors[RAD_ENC_MAX_PLANES]{};weight_planes(w,tensors);std::array<std::vector<uint8_t>,RAD_ENC_MAX_PLANES> selected;
            need(w.n_sel==source.enc.n_planes&&w.stored_dtype==pk::storage_dtype(static_cast<pk::Type>(raw.format)),"real all-plane private descriptor selection missing");
            for(int plane=0;plane<w.n_sel;++plane){auto& bytes=selected[size_t(plane)];const auto& share=shares[plane];bytes.resize(size_t(share.dense_bytes()));size_t at=0;
                for(const auto& rect:share.rects)for(int64_t r=0;r<rect.rows;++r){const int64_t off=(rect.row0+r)*share.row_bytes+rect.byte0;const auto& plane_input=planes[size_t(w.sel[plane])];
                    need(off>=0&&rect.bytes>0&&size_t(off+rect.bytes)<=plane_input.size()&&at+size_t(rect.bytes)<=bytes.size(),"bounded canonical region copy");std::memcpy(bytes.data()+at,plane_input.data()+off,size_t(rect.bytes));at+=size_t(rect.bytes);}
                need(at==bytes.size(),"source rectangles failed to cover plane");tensors[plane].data=bytes.data();
            }
            const auto& resolved=p.ops[size_t(w.lay_op)].bands[size_t(w.lay_band)].dom[w.lay_dom];need(resolved.row&&resolved.row->plugin=="libds41_ref"&&resolved.row->info->relayout,"actual HOST private relayout missing");
            auto native=std::make_unique<Native>(w.stored_bytes);okay(resolved.row->info->relayout(resolved.geom.params(),resolved.geom.n_params(),w.lay_operand,&w.enc,w.sel,tensors,w.n_sel,native->data,w.stored_bytes),"registered per-group private relayout");
            need(uint64_t(w.stored_bytes)==group.original_block_bytes&&std::memcmp(native->data,raw.packed.data()+group.original_block_byte_offset,size_t(w.stored_bytes))==0,"native relayout changed independent original group blocks");
            native->seal();w.ptr=native->data;storage.emplace(projections[local].weight,std::move(native));
            for(const auto& band:p.ops[projections[local].op].bands){const auto& host=band.dom[RAD_DOMAIN_HOST];need(host.row&&host.row->plugin=="libds41_ref"&&host.row->info->launch,"real HOST launch missing");launches.emplace_back(host.row->info,host.row->info->launch);}
        }
        CtxDesc desc{};desc.program=&p;desc.rank=context.rank;desc.world_size=context.world_size;desc.device=0;desc.arch_step=step;desc.record_passes=false;
        okay(ctx.init(desc),"real grouped Ctx init");okay(ctx.bind_thread(),"real HOST rank bind");need(!ctx.checks_device_status(),"HOST row must return synchronously");
    }
    float* ptr(rad_buf b){auto* p=static_cast<float*>(rad_buf_ptr(&ctx,b));need(p,"actual arena pointer missing");return p;}
    void run(int64_t live,bool common,int variation){
        constexpr float guard=-12345.25f;auto* x=ptr(input);auto* y=ptr(common?wrong:output);
        std::fill_n(x,capacity*input_pitch,guard);std::fill_n(y,capacity*output_pitch,guard);
        for(int64_t row=0;row<live;++row)for(int64_t group=0;group<raw.groups;++group)for(int64_t k=0;k<K;++k)x[row*input_pitch+group*K+k]=float((row*11+group*17+k*7+variation*5)%31-15)/256.f;
        const std::vector<float> before(x,x+capacity*input_pitch);std::vector<int> codes(projections.size(),RAD_E_STATE);
        active=[&](RadCtx* c){for(size_t local=0;local<projections.size();++local){const auto& g=plan.slices[local];auto a=RAD_B_COL(input,common?0:g.input.element_offset,K);a.rows=live;
            auto out=RAD_B_COL(common?wrong:output,g.output.element_offset,N);out.rows=live;codes[local]=pp::issue_native_linear(c,projections[local],a,out,live);}};
        RadBatch batch{};batch.phase=RAD_PHASE_PREFILL;batch.n_tok=live;const int status=ctx.run_step(&batch);active={};okay(status,"actual grouped rad_issue pass");okay(ctx.await_device_status(),"HOST await");okay(rad_stream_sync(ctx.stream()),"HOST stream drain");
        for(int code:codes)okay(code,"registered group dense issue");
        need(std::equal(before.begin(),before.end(),x),"pitched readonly input modified");bool common_diff=false;
        for(int64_t row=0;row<capacity;++row)for(int64_t column=0;column<output_pitch;++column){
            const int64_t group=column/N,n=column%N;const bool owned=group>=plan.group_begin&&group<plan.group_begin+plan.group_count&&column<raw.groups*N&&row<live;
            if(!owned){CHECK_EQ(y[row*output_pitch+column],guard);continue;}
            float expected=0,correct=0;for(int64_t k=0;k<K;++k){const float weight=raw.expected[size_t((group*N+n)*K+k)];expected=expected+x[row*input_pitch+(common?0:group*K)+k]*weight;correct=correct+x[row*input_pitch+group*K+k]*weight;}
            CHECK_EQ(y[row*output_pitch+column],expected);common_diff|=expected!=correct;
        }
        if(common&&plan.group_begin+plan.group_count>1)CHECK(common_diff);
        for(const auto& [handle,native]:storage){(void)handle;native->check();}
        for(const auto& [info,launch]:launches)CHECK(info->launch==launch);
        CHECK_EQ(ctx.pass_counts().played,0);
    }
};
}

TEST(ds41_grouped_projection_real_host_import_relayout_issue_covers_all_six_formats_and_pitched_groups){
    if(!device_is_host())SKIP("HOST-only actual grouped projection proof");
    for(auto format:{gg::Format::iq2_xxs,gg::Format::iq2_xs,gg::Format::iq2_s,gg::Format::iq3_xxs,gg::Format::q5_k,gg::Format::q8_0}){
        PackedSource source(format,2);Harness fixture(source);fixture.run(1,false,0);fixture.run(3,false,1);fixture.run(2,true,2);
    }
}
TEST(ds41_grouped_projection_logical_tp4_keeps_complete_groups_and_global_output_columns){
    if(!device_is_host())SKIP("HOST-only sequential logical TP4 ownership proof");
    PackedSource source(gg::Format::q5_k,8);
    for(int rank=0;rank<4;++rank){Harness fixture(source,rank,4);CHECK_EQ(fixture.plan.group_begin,rank*2);CHECK_EQ(fixture.plan.group_count,2);fixture.run(2,false,rank);fixture.run(1,true,rank+1);}
}
RAD_TEST_MAIN()
