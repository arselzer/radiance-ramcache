/* Actual HOST Builder/Ctx/libds41_ref projection issue. Small generated source records are
 * imported through libquant, cut by Core weight_share and passed to the registered relayout.
 * Logical TP ranks execute sequentially on ONE host device; this is not a TP collective test. */
#include "quant_fixture.h"
#include "build/rad_build.h"
#include "format/share.h"
#include "runtime/ctx.h"
#include "../arch/deepseek41/packed_projections.h"
#include "../libds41/ds41_packed_args.h"
#include <nlohmann/json.hpp>
#include <array>
#include <fstream>
#include <map>
#include <memory>

using namespace rad;
namespace pp=rad::deepseek41::packed;
namespace gg=rad::gguf;
namespace pk=rad::ds41::packed;
namespace {
constexpr int64_t tokens=2,routes=4,hidden=256,intermediate=1280;
constexpr int layer=2;
void need(bool condition,const std::string& text){if(!condition)throw std::runtime_error(text);}
void okay(int status,const char* text){need(status==RAD_OK,std::string(text)+": "+rad_strerror(status));}
std::vector<uint8_t> unhex(const std::string& text){
    need(text.size()%2==0&&text.size()<=65536,"unbounded/malformed tiny golden hex");
    auto digit=[](char c)->unsigned{if(c>='0'&&c<='9')return unsigned(c-'0');if(c>='a'&&c<='f')return unsigned(c-'a'+10);if(c>='A'&&c<='F')return unsigned(c-'A'+10);throw std::runtime_error("invalid hex");};
    std::vector<uint8_t> result(text.size()/2);for(size_t i=0;i<result.size();++i)result[i]=uint8_t(digit(text[2*i])*16+digit(text[2*i+1]));return result;
}
const nlohmann::json& goldens(){
    static const auto json=[] {
        const auto path=std::string(RAD_GOLDEN_FIXTURES)+"/gguf-import.json";
        need(std::filesystem::file_size(path)<=2u*1024u*1024u,"golden metadata cap exceeded");
        std::ifstream file(path);need(bool(file),"missing independent GGML goldens");return nlohmann::json::parse(file);
    }();return json;
}
uint16_t bf16(float f){uint32_t bits;std::memcpy(&bits,&f,4);need(std::isfinite(f),"nonfinite test arithmetic");return uint16_t((bits+0x7FFFu+((bits>>16)&1u))>>16);}
float from_bf16(uint16_t h){const uint32_t bits=uint32_t(h)<<16;float f;std::memcpy(&f,&bits,4);return f;}
// Scaling both d and dmin by a power of two preserves the independent golden values exactly.
// The tiny fixture includes zero/subnormal half scales. Only exact upward shifts are used;
// refuse any case requiring normalization/rounding rather than invent another encoder.
void scale_half(uint8_t* p,int exponent){
    uint16_t bits=uint16_t(p[0])|uint16_t(p[1])<<8;const int e=(bits>>10)&31;
    need(exponent>=0&&exponent<=2&&e<31&&e+exponent<31,"golden scale is outside exact power-of-two profile");
    if(!e){const unsigned mantissa=bits&1023u;need((mantissa<<exponent)<=1023u,"subnormal test shift would normalize");bits=uint16_t((bits&0x8000u)|(mantissa<<exponent));}
    else bits=uint16_t(int(bits)+exponent*1024);
    p[0]=uint8_t(bits);p[1]=uint8_t(bits>>8);
}
struct SourceData {
    gg::Format format;int64_t rows,cols;WeightSource metadata{};
    std::vector<uint8_t> packed;std::vector<float> expected;
    std::array<std::vector<uint8_t>,RAD_ENC_MAX_PLANES> planes;
    SourceData(gg::Format f,int64_t nr,int64_t nc,int seed):format(f),rows(nr),cols(nc){
        const auto* geom=gg::geometry(f);const nlohmann::json* golden=nullptr;
        for(const auto& entry:goldens().at("formats"))if(entry.at("name")==geom->name){golden=&entry;break;}
        need(golden&&golden->at("rows")==4&&golden->at("cols")==256,"missing expected four-row golden");
        const auto bytes=unhex(golden->at("packed_hex").get<std::string>()),values=unhex(golden->at("f32_hex").get<std::string>());
        need(values.size()==4*256*sizeof(float),"wrong golden float extent");std::array<float,4*256> logical{};std::memcpy(logical.data(),values.data(),values.size());
        const int64_t golden_row=256/geom->values*geom->bytes,rb=cols/geom->values*geom->bytes;
        need(cols>0&&cols%256==0&&rows>0&&rows<=1280&&cols<=1280&&bytes.size()==size_t(4*golden_row),"generated source exceeds bounded fixture");
        packed.resize(size_t(rows*rb));expected.resize(size_t(rows*cols));
        for(int64_t n=0;n<rows;++n)for(int64_t chunk=0;chunk<cols/256;++chunk){
            const int selected=int((n+3*chunk+seed)%4),exponent=int((n/256+chunk+seed)%3);
            auto* at=packed.data()+n*rb+chunk*golden_row;std::memcpy(at,bytes.data()+selected*golden_row,size_t(golden_row));
            for(int64_t block=0;block<256/geom->values;++block){scale_half(at+block*geom->bytes,exponent);if(f==gg::Format::q5_k)scale_half(at+block*geom->bytes+2,exponent);}
            for(int64_t k=0;k<256;++k)expected[size_t(n*cols+chunk*256+k)]=std::ldexp(logical[size_t(selected*256+k)],exponent);
        }
        const auto* imp=qfix::reg().checkpoint_importer("gguf",geom->name);need(imp&&imp->info,"missing real libquant importer");
        RadCheckpointImportWeight iw{"projection-runtime-source",geom->name,rows,cols};okay(imp->info->encoding(&iw,&metadata.enc),"source encoding");
        metadata.rank=2;metadata.shape[0]=rows;metadata.shape[1]=cols;void* pointers[RAD_ENC_MAX_PLANES]{};const void* decoded_planes[RAD_ENC_MAX_PLANES]{};
        for(int p=0;p<metadata.enc.n_planes;++p){const auto count=rad_enc_plane_bytes(&metadata.enc.plane[p],rows,cols);need(count>0&&count<2*1024*1024,"canonical plane cap");planes[size_t(p)].resize(size_t(count));pointers[p]=planes[size_t(p)].data();decoded_planes[p]=pointers[p];}
        okay(imp->info->import_rows(&iw,packed.data(),int64_t(packed.size()),rows,pointers),"actual source import");
        EncodedView view;enc_view(metadata.enc,rows,cols,decoded_planes,&view);std::vector<float> decoded(expected.size());std::string why;
        okay(enc_decode_rows(view,0,rows,decoded.data(),&why),"independent canonical source decode");
        need(decoded==expected,"source importer/generic decoder disagrees with independent scaled GGML goldens");
    }
};
using Sources=std::map<std::string,std::shared_ptr<SourceData>>;
struct Native {
    static constexpr size_t guard=256;std::vector<uint8_t> allocation,expected;uint8_t* data=nullptr;int64_t bytes=0;
    explicit Native(int64_t count):allocation(size_t(count)+3*guard,0xB7),bytes(count){need(count>0&&count<1024*1024,"native fixture size cap");const uintptr_t p=(uintptr_t(allocation.data())+2*guard-1)&~uintptr_t(guard-1);data=reinterpret_cast<uint8_t*>(p);}
    void seal(){expected=allocation;}
    void check()const{need(allocation==expected,"native weights or native guard bytes changed during issue");}
};
std::function<void(RadCtx*)> active_step;
void step(RadCtx* c,const RadBatch*){if(active_step)active_step(c);else rad_step_fail(c,"projection test has no active step");}
struct Fixture {
    Registry registry;Sources sources;std::map<std::string,std::string> names;
    RadBuildCtx context{};std::unique_ptr<Builder> b;std::map<rad_weight,std::unique_ptr<Native>> native;
    std::vector<std::pair<const RadKernelInfo*,RadLaunchFn>> launches;Ctx ctx;
    explicit Fixture(int rank=0,int world=1){
        need(device_is_host(),"HOST-only fixture");okay(registry.load_plugin(home()+"/experimental/kernels/libds41_ref.so",0),"load real HOST DS41 plugin");
        context.rank=rank;context.world_size=world;context.max_tok=tokens;context.max_seqs=1;context.max_ctx=tokens;context.scope="";
        RadModelMeta meta{};meta.arch_id="projection-runtime-test";meta.name="generated-not-model";meta.quant="native-GGUF";
        meta.n_layers=4;meta.n_embd=hidden;meta.n_ff=intermediate;meta.n_expert=2;
        b=std::make_unique<Builder>(registry,meta,context);b->set_instances(false);
        b->set_sources([this](const std::string& name,const Program& p,WeightSource* out){
            auto found=sources.find(name);
            if(found==sources.end())for(const auto& map:p.name_map)if(map.declared&&name==map.declared&&map.mode==RAD_MAP_COPY&&map.n_src==1){found=sources.find(map.src[0]);break;}
            if(found==sources.end())return RAD_E_NOTFOUND;
            *out=found->second->metadata;
            return RAD_OK;
        });
    }
    const char* keep(const std::string& text){return names.emplace(text,text).first->second.c_str();}
    pp::Source add(const std::string& name,std::shared_ptr<SourceData> source){
        const char* alias=keep(name),*raw=keep("raw."+name);sources.emplace(raw,std::move(source));
        RadNameMap map{};map.declared=alias;map.mode=RAD_MAP_COPY;map.n_src=1;map.src[0]=raw;map.src_index[0]=-1;
        okay(rad_decl_name_map(b.get(),&map),"explicit source name map");return {alias,alias};
    }
    const SourceData& source_for(const WeightInfo& w)const{
        for(const auto& map:b->program().name_map)if(map.declared&&w.source==map.declared)return *sources.at(map.src[0]);
        return *sources.at(w.source);
    }
    rad_buf buffer(const char* name,uint32_t dtype,int64_t rows,int64_t cols=0){
        RadBufDecl d{};d.dtype=dtype;d.kind=RAD_BUF_PERSIST;d.domain=RAD_DOMAIN_HOST;d.rank=cols?2:1;d.shape[0]=rows;if(cols)d.shape[1]=cols;
        const auto h=rad_decl_buffer(b.get(),name,&d);need(h!=0,"declare real HOST buffer");return h;
    }
    void hazards(rad_op op,std::initializer_list<rad_buf> in,std::initializer_list<rad_buf> out){
        okay(rad_op_reads(b.get(),op,in.begin(),int(in.size())),"declare input lifetimes");okay(rad_op_writes(b.get(),op,out.begin(),int(out.size())),"declare output lifetimes");
    }
    void finish(){
        okay(b->finish(),"finish actual projection Builder");auto& program=b->program();
        const int64_t arena=arena_plan_bytes(program,nullptr,nullptr)+align_up(arena_host_bytes(program),RAD_ALIGN_UNIT);need(arena>=0&&arena<=1024*1024,"tiny Program arena cap exceeded");
        for(size_t wi=1;wi<program.weights.size();++wi){
            auto& w=program.weights[wi];const auto& source=source_for(w);const auto* geom=gg::geometry(source.format);
            need(w.stored_dtype==pk::storage_dtype(static_cast<pk::Type>(source.format))&&rad_dtype_bits(w.stored_dtype)==0&&w.n_sel==source.metadata.enc.n_planes,"actual private/all-plane layout not selected");
            PlaneShare shares[RAD_ENC_MAX_PLANES]{};std::string why;
            okay(weight_share(w,source.metadata.enc,2,source.metadata.shape,context.world_size,context.rank,shares,&why),"Core canonical source shard");
            RadTensor tensors[RAD_ENC_MAX_PLANES]{};weight_planes(w,tensors);std::array<std::vector<uint8_t>,RAD_ENC_MAX_PLANES> selected;
            for(int p=0;p<w.n_sel;++p){
                const auto& share=shares[p];need(share.rows==w.sel_rows[p]&&share.cols==w.sel_cols[p]&&share.dtype==tensors[p].dtype,"Core share/relayout plane shape disagreement");
                auto& bytes=selected[size_t(p)];bytes.resize(size_t(share.dense_bytes()));size_t at=0;
                const auto& all=source.planes[size_t(w.sel[p])];
                for(const auto& rect:share.rects)for(int64_t r=0;r<rect.rows;++r){
                    const int64_t off=(rect.row0+r)*share.row_bytes+rect.byte0;
                    need(off>=0&&rect.bytes>0&&size_t(off+rect.bytes)<=all.size()&&at+size_t(rect.bytes)<=bytes.size(),"bounded Core rectangle copy");
                    std::memcpy(bytes.data()+at,all.data()+off,size_t(rect.bytes));at+=size_t(rect.bytes);
                }
                need(at==bytes.size(),"Core rectangles failed to fill complete selected plane");tensors[p].data=bytes.data();
            }
            auto storage=std::make_unique<Native>(w.stored_bytes);const auto& resolved=program.ops[size_t(w.lay_op)].bands[size_t(w.lay_band)].dom[w.lay_dom];
            need(resolved.row&&resolved.row->plugin=="libds41_ref"&&resolved.row->info&&resolved.row->info->relayout&&resolved.row->info->launch,"actual registered HOST row missing");
            const auto* info=resolved.row->info;okay(info->relayout(resolved.geom.params(),resolved.geom.n_params(),w.lay_operand,&w.enc,w.sel,tensors,w.n_sel,storage->data,w.stored_bytes),"actual registered native relayout");
            const int64_t row0=w.decl.shard==RAD_SHARD_ROW?w.shard_lo:0,col0=w.decl.shard==RAD_SHARD_COL?w.shard_lo:0,
                          src_rb=source.cols/geom->values*geom->bytes,dst_rb=w.decl.shape[1]/geom->values*geom->bytes;
            need(w.stored_bytes==w.decl.shape[0]*dst_rb&&col0%geom->values==0,"native source slice geometry");
            for(int64_t r=0;r<w.decl.shape[0];++r)need(std::memcmp(storage->data+r*dst_rb,source.packed.data()+(row0+r)*src_rb+col0/geom->values*geom->bytes,size_t(dst_rb))==0,"Core source share plus native relayout changed selected GGUF blocks");
            storage->seal();w.ptr=storage->data;native.emplace(rad_weight(wi),std::move(storage));
        }
        for(size_t oi=1;oi<program.ops.size();++oi)for(const auto& band:program.ops[oi].bands){const auto& resolved=band.dom[RAD_DOMAIN_HOST];need(resolved.row&&resolved.row->plugin=="libds41_ref"&&resolved.row->info&&resolved.row->info->launch,"real HOST launch missing");launches.emplace_back(resolved.row->info,resolved.row->info->launch);}
        CtxDesc cd{};cd.program=&program;cd.rank=context.rank;cd.world_size=context.world_size;cd.device=0;cd.arch_step=step;cd.record_passes=false;
        okay(ctx.init(cd),"actual Ctx init");okay(ctx.bind_thread(),"actual Ctx thread binding");need(!ctx.checks_device_status(),"HOST rows must report errors synchronously");
    }
    template<class T>T* ptr(rad_buf h){auto* p=static_cast<T*>(rad_buf_ptr(&ctx,h));need(p!=nullptr,"missing actual arena buffer");return p;}
    void run(const std::function<void(RadCtx*)>& body){
        active_step=body;RadBatch batch{};batch.phase=RAD_PHASE_PREFILL;batch.n_tok=tokens;
        const int rc=ctx.run_step(&batch);active_step={};need(rc==RAD_OK,std::string("real helper issue failed: ")+ctx.step_error());okay(ctx.await_device_status(),"HOST await");okay(rad_stream_sync(ctx.stream()),"HOST stream drain");
        for(const auto& [handle,storage]:native){(void)handle;storage->check();}for(const auto& [info,launch]:launches)need(info->launch==launch,"test replaced a real launch");
    }
};
float dot(const std::vector<float>& weight,int64_t pitch,int64_t row,int64_t begin,const float* x,int64_t count){
    float sum=0;for(int64_t k=0;k<count;++k){const float product=x[k]*weight[size_t(row*pitch+begin+k)];sum=sum+product;}return sum;
}
std::vector<float> input(){std::vector<float> x(size_t(tokens*hidden));for(int64_t t=0;t<tokens;++t)for(int64_t k=0;k<hidden;++k)x[size_t(t*hidden+k)]=float((t*17+k*7)%23-11)/256;return x;}
constexpr std::array<int32_t,routes> expert_ids{1,0,0,1},token_ids{1,0,1,0},identity{0,1,2,3};
float activation(float gate,float up,int64_t job){return (gate/(1.0f+std::exp(-gate)))*up*(job%2?0.375f:0.625f);}
}

TEST(ds41_projection_runtime_dense_six_formats_use_real_host_issue_and_source_relayout){
    if(!device_is_host())SKIP("HOST-only actual projection runtime proof");
    for(const auto format:{gg::Format::iq2_xxs,gg::Format::iq2_xs,gg::Format::iq2_s,gg::Format::iq3_xxs,gg::Format::q5_k,gg::Format::q8_0}){
        auto source=std::make_shared<SourceData>(format,4,hidden,1);Fixture f;
        pp::LinearSpec spec;spec.weight=f.add("dense",source);spec.rows=4;spec.cols=hidden;spec.max_rows=tokens;spec.input_dtype=RAD_F32;spec.output_dtype=RAD_BF16;
        pp::NativeLinear p;std::string why;REQUIRE_EQ(pp::declare_native_linear(f.b.get(),f.context,spec,&p,&why),RAD_OK);
        const auto xh=f.buffer("dense.x",RAD_F32,tokens,hidden),yh=f.buffer("dense.y",RAD_BF16,tokens,4);f.hazards(p.op,{xh},{yh});f.finish();
        const auto x=input();std::memcpy(f.ptr<float>(xh),x.data(),x.size()*4);std::fill_n(f.ptr<uint16_t>(yh),tokens*4,uint16_t(0x7FC1));int status=RAD_E_STATE;
        f.run([&](RadCtx* c){status=pp::issue_native_linear(c,p,RAD_B_N(xh,0,tokens),RAD_B_N(yh,0,tokens),tokens);});REQUIRE_EQ(status,RAD_OK);
        for(int64_t t=0;t<tokens;++t)for(int64_t n=0;n<4;++n)CHECK_EQ(f.ptr<uint16_t>(yh)[t*4+n],bf16(dot(source->expected,hidden,n,0,x.data()+t*hidden,hidden)));
        CHECK_EQ(f.ctx.pass_counts().played,0);
    }
}

TEST(ds41_projection_runtime_routed_down_uses_route_capacity_and_identity_rows_with_unequal_tp4_shards){
    if(!device_is_host())SKIP("HOST-only logical TP4 projection proof");
    std::array<std::array<std::shared_ptr<SourceData>,2>,3> source;
    const gg::Format formats[]{gg::Format::iq2_xs,gg::Format::iq3_xxs,gg::Format::q5_k};
    for(int projection=0;projection<3;++projection)for(int e=0;e<2;++e)source[size_t(projection)][size_t(e)]=std::make_shared<SourceData>(formats[projection],projection==2?hidden:intermediate,projection==2?intermediate:hidden,projection+e+1);
    const auto x=input();std::array<std::vector<float>,4> partial,oracle_partial;std::array<int64_t,4> widths{},offsets{};
    for(int rank=0;rank<4;++rank){
        Fixture f(rank,4);pp::RoutedSpec spec;spec.layer=layer;spec.hidden=hidden;spec.intermediate=intermediate;spec.max_tokens=tokens;spec.topk=2;spec.input_dtype=RAD_F32;spec.activation_dtype=RAD_BF16;
        for(int e=0;e<2;++e){const auto root="e"+std::to_string(e)+".";spec.experts.push_back({f.add(root+"gate",source[0][size_t(e)]),f.add(root+"up",source[1][size_t(e)]),f.add(root+"down",source[2][size_t(e)])});}
        pp::RoutedFFN p;std::string why;REQUIRE_EQ(pp::declare_routed_ffn(f.b.get(),f.context,spec,&p,&why),RAD_OK);
        const int64_t width=p.partition.width,lo=p.partition.offset;widths[size_t(rank)]=width;offsets[size_t(rank)]=lo;
        CHECK_EQ(width,rank==layer?512:256);CHECK_EQ(p.gate.input_capacity,tokens);CHECK_EQ(p.up.input_capacity,tokens);CHECK_EQ(p.down.input_capacity,routes);CHECK_EQ(p.down.route_capacity,routes);
        const auto xh=f.buffer("tokens",RAD_F32,tokens,hidden),eh=f.buffer("experts",RAD_I32,routes),th=f.buffer("token-rows",RAD_I32,routes),ih=f.buffer("identity-rows",RAD_I32,routes),
            gh=f.buffer("gate",RAD_BF16,routes,width),uh=f.buffer("up",RAD_BF16,routes,width),ah=f.buffer("CPU-supplied-rounded-activation",RAD_BF16,routes,width),dh=f.buffer("down-F32-partials",RAD_F32,routes,hidden),
            gs=f.buffer("gate-status",RAD_I32,routes),us=f.buffer("up-status",RAD_I32,routes),ds=f.buffer("down-status",RAD_I32,routes);
        f.hazards(p.gate.op,{xh,eh,th},{gh,gs});f.hazards(p.up.op,{xh,eh,th},{uh,us});f.hazards(p.down.op,{ah,eh,ih},{dh,ds});f.finish();
        const auto& down_op=f.b->program().ops[p.down.op];CHECK_EQ(down_op.range_hi,routes);CHECK_EQ(std::string(down_op.base.get_s("y_dtype")),std::string("f32"));
        std::memcpy(f.ptr<float>(xh),x.data(),x.size()*4);std::memcpy(f.ptr<int32_t>(eh),expert_ids.data(),routes*4);std::memcpy(f.ptr<int32_t>(th),token_ids.data(),routes*4);std::memcpy(f.ptr<int32_t>(ih),identity.data(),routes*4);
        std::fill_n(f.ptr<int32_t>(gs),routes,-99);std::fill_n(f.ptr<int32_t>(us),routes,-99);std::fill_n(f.ptr<int32_t>(ds),routes,-99);std::fill_n(f.ptr<float>(dh),routes*hidden,-12345.0f);
        std::vector<uint16_t> expected_gate(size_t(routes*width)),expected_up(size_t(routes*width)),expected_act(size_t(routes*width));
        for(int64_t job=0;job<routes;++job)for(int64_t n=0;n<width;++n){const int e=expert_ids[size_t(job)],t=token_ids[size_t(job)];const size_t at=size_t(job*width+n);
            expected_gate[at]=bf16(dot(source[0][size_t(e)]->expected,hidden,lo+n,0,x.data()+t*hidden,hidden));expected_up[at]=bf16(dot(source[1][size_t(e)]->expected,hidden,lo+n,0,x.data()+t*hidden,hidden));
            expected_act[at]=bf16(activation(from_bf16(expected_gate[at]),from_bf16(expected_up[at]),job));}
        std::array<int,3> statuses{RAD_E_STATE,RAD_E_STATE,RAD_E_STATE};
        f.run([&](RadCtx* c){
            statuses[0]=pp::issue_routed_projection(c,p.gate,RAD_B_N(xh,0,tokens),RAD_B_N(eh,0,routes),RAD_B_N(th,0,routes),RAD_B_N(gh,0,routes),RAD_B_N(gs,0,routes),tokens);
            statuses[1]=pp::issue_routed_projection(c,p.up,RAD_B_N(xh,0,tokens),RAD_B_N(eh,0,routes),RAD_B_N(th,0,routes),RAD_B_N(uh,0,routes),RAD_B_N(us,0,routes),tokens);
            if(statuses[0]!=RAD_OK||statuses[1]!=RAD_OK)return;
            // Deliberate HOST test scaffold, not an undeclared production device graph op.
            // The projection test supplies completed/rounded activation rows synchronously.
            for(int64_t at=0;at<routes*width;++at)f.ptr<uint16_t>(ah)[at]=bf16(activation(from_bf16(f.ptr<uint16_t>(gh)[at]),from_bf16(f.ptr<uint16_t>(uh)[at]),at/width));
            statuses[2]=pp::issue_routed_projection(c,p.down,RAD_B_N(ah,0,routes),RAD_B_N(eh,0,routes),RAD_B_N(ih,0,routes),RAD_B_N(dh,0,routes),RAD_B_N(ds,0,routes),routes);
        });for(int status:statuses)REQUIRE_EQ(status,RAD_OK);
        for(size_t at=0;at<expected_gate.size();++at){CHECK_EQ(f.ptr<uint16_t>(gh)[at],expected_gate[at]);CHECK_EQ(f.ptr<uint16_t>(uh)[at],expected_up[at]);CHECK_EQ(f.ptr<uint16_t>(ah)[at],expected_act[at]);}
        for(int64_t job=0;job<routes;++job){CHECK_EQ(f.ptr<int32_t>(gs)[job],RAD_OK);CHECK_EQ(f.ptr<int32_t>(us)[job],RAD_OK);CHECK_EQ(f.ptr<int32_t>(ds)[job],RAD_OK);}
        CHECK(std::equal(x.begin(),x.end(),f.ptr<float>(xh)));CHECK(std::equal(expert_ids.begin(),expert_ids.end(),f.ptr<int32_t>(eh)));
        CHECK(std::equal(token_ids.begin(),token_ids.end(),f.ptr<int32_t>(th)));CHECK(std::equal(identity.begin(),identity.end(),f.ptr<int32_t>(ih)));
        partial[size_t(rank)].assign(f.ptr<float>(dh),f.ptr<float>(dh)+routes*hidden);auto& expected=oracle_partial[size_t(rank)];expected.resize(size_t(routes*hidden));std::vector<float> decoded_act(size_t(routes*width));
        for(size_t at=0;at<decoded_act.size();++at)decoded_act[at]=from_bf16(expected_act[at]);
        for(int64_t job=0;job<routes;++job)for(int64_t n=0;n<hidden;++n){const size_t at=size_t(job*hidden+n);expected[at]=dot(source[2][size_t(expert_ids[size_t(job)])]->expected,intermediate,n,lo,decoded_act.data()+job*width,width);CHECK_EQ(partial[size_t(rank)][at],expected[at]);}
        CHECK_EQ(f.ctx.pass_counts().played,0);
        // Helper-level over-capacity refusal does not mutate output or require a Core issue.
        const auto before=partial[size_t(rank)];CHECK_EQ(pp::issue_routed_projection(&f.ctx,p.down,RAD_B(ah),RAD_B(eh),RAD_B(ih),RAD_B(dh),RAD_B(ds),routes+1),RAD_E_INVAL);
        CHECK(std::equal(before.begin(),before.end(),f.ptr<float>(dh)));
    }
    CHECK_EQ(widths[0]+widths[1]+widths[2]+widths[3],intermediate);CHECK_EQ(offsets[0],0);for(int rank=1;rank<4;++rank)CHECK_EQ(offsets[size_t(rank)],offsets[size_t(rank-1)]+widths[size_t(rank-1)]);
    std::array<std::vector<uint16_t>,2> complete;for(auto& value:complete)value.resize(size_t(routes*hidden));bool premature_cast_differs=false;
    for(int64_t at=0;at<routes*hidden;++at){float got=0,expected=0,premature=0;for(int rank=0;rank<4;++rank){got=got+partial[size_t(rank)][size_t(at)];expected=expected+oracle_partial[size_t(rank)][size_t(at)];premature=premature+from_bf16(bf16(partial[size_t(rank)][size_t(at)]));}
        complete[0][size_t(at)]=bf16(got);complete[1][size_t(at)]=bf16(expected);CHECK_EQ(complete[0][size_t(at)],complete[1][size_t(at)]);premature_cast_differs|=bf16(premature)!=bf16(got);}
    CHECK(premature_cast_differs); // This fixture distinguishes cast-after-TP-sum from cast-per-rank.
    // Canonical expert-ID order per token, after each completed expert has been rounded once.
    for(int64_t token=0;token<tokens;++token)for(int64_t n=0;n<hidden;++n){float got=0,expected=0;for(int e=0;e<2;++e)for(int64_t job=0;job<routes;++job)if(expert_ids[size_t(job)]==e&&token_ids[size_t(job)]==token){got=got+from_bf16(complete[0][size_t(job*hidden+n)]);expected=expected+from_bf16(complete[1][size_t(job*hidden+n)]);}CHECK_EQ(got,expected);}
}
RAD_TEST_MAIN()
