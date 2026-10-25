/* Existing plain HOST projection rows through the real Builder and Ctx. The source is BF16
 * metadata plus a tiny generated BF16 payload. The test manually supplies its exact F32 widening;
 * it proves Builder accounting/binding and real issue, NOT execution of Engine's private loader. */
#include "rad_test.h"
#include "device/device.h"
#include "build/rad_build.h"
#include "format/encoding.h"
#include "runtime/arena.h"
#include "runtime/ctx.h"
#include <array>
#include <memory>

using namespace rad;
namespace {
constexpr int64_t max_rows=5, max_issue=3, n=4;
void need(bool value,const std::string& why){if(!value)throw std::runtime_error(why);}
void okay(int status,const char* why){need(status==RAD_OK,std::string(why)+": "+rad_strerror(status));}
uint16_t bf16(float value){uint32_t bits;std::memcpy(&bits,&value,4);return uint16_t((bits+0x7FFFu+((bits>>16)&1u))>>16);}
float widen(uint16_t value){const uint32_t bits=uint32_t(value)<<16;float result;std::memcpy(&result,&bits,4);return result;}
int x_numerator(int64_t row,int64_t col){return int((row*7+col*3)%13)-6;}
int w_numerator(int64_t row,int64_t col,int generation){return (int((row*5+col*2)%11)-5)*(generation? -1:1);}
float expected(int64_t row,int64_t col,int64_t k,int generation){
    int64_t sum=0;for(int64_t j=0;j<k;++j)sum+=int64_t(x_numerator(row,j))*w_numerator(col,j,generation);
    return float(sum)/128.0f; // products of exact BF16 x/8 and w/16; the tiny sum is exact in F32.
}
struct Table final:ResidencyTable {
    std::vector<WeightSlot> slots;std::vector<rad_weight> changed;uint64_t version=1;
    const WeightSlot* lookup(rad_weight h)const override{return size_t(h)<slots.size()?&slots[size_t(h)]:nullptr;}
    uint64_t epoch()const override{return version;}
    const std::vector<rad_weight>& changes()const override{return changed;}
    void clear_changes()override{changed.clear();}
    void move(rad_weight h,void* ptr){auto& slot=slots.at(size_t(h));slot.ptr=ptr;++slot.generation;++version;changed.push_back(h);}
};
std::function<void(RadCtx*)> body;
void step(RadCtx* c,const RadBatch*){if(body)body(c);else rad_step_fail(c,"plain projection test has no body");}
struct Fixture {
    Registry registry;std::unique_ptr<Builder> builder;
    std::array<std::vector<uint16_t>,2> source;
    std::array<std::vector<float>,2> converted;
    Table table;Ctx ctx;
    const bool f32;const int64_t k,xpitch,ypitch=n+3;
    rad_weight weight=0;rad_op op=0;rad_buf x=0,y=0;
    std::vector<std::pair<const RadKernelInfo*,RadLaunchFn>> launches;
    Fixture(bool output_f32,int64_t width):f32(output_f32),k(width),xpitch(width+3){
        need(device_is_host(),"HOST-only fixture");
        okay(registry.load_plugin(home()+"/kernels/libref.so",0),"load real libref");
        RadBuildCtx bc{};bc.rank=0;bc.world_size=1;bc.max_tok=max_issue;bc.max_seqs=1;bc.max_ctx=max_issue;bc.scope="";
        RadModelMeta meta{};meta.arch_id="ds41-plain-projection-test";meta.name="tiny-generated-not-model";meta.quant="plain-BF16";meta.n_embd=k;meta.n_layers=1;
        builder=std::make_unique<Builder>(registry,meta,bc);builder->set_instances(false);
        builder->set_sources([this](const std::string& name,const Program&,WeightSource* out){
            if(name!="source.bf16")return RAD_E_NOTFOUND;
            *out={};out->enc=rad_enc_plain(RAD_BF16);out->rank=2;out->shape[0]=n;out->shape[1]=k;return RAD_OK;
        });
        RadWeightDecl wd{};wd.dtype=f32?RAD_F32:RAD_BF16;wd.rank=2;wd.shape[0]=n;wd.shape[1]=k;wd.access=RAD_ACCESS_PER_TOKEN;wd.shard=RAD_SHARD_NONE;wd.group=rad_group_layer(0);wd.source="source.bf16";
        weight=rad_decl_weight(builder.get(),"projection.weight",&wd);need(weight!=0,"declare actual source-backed weight");
        RadBufDecl bd{};bd.dtype=RAD_BF16;bd.rank=2;bd.shape[0]=max_rows;bd.shape[1]=xpitch;bd.kind=RAD_BUF_PERSIST;bd.domain=RAD_DOMAIN_HOST;
        x=rad_decl_buffer(builder.get(),"pitched.input",&bd);bd.dtype=f32?RAD_F32:RAD_BF16;bd.shape[1]=ypitch;y=rad_decl_buffer(builder.get(),"pitched.output",&bd);need(x&&y,"declare actual pitched buffers");
        const RadParam params[]{RAD_RANGE("M",1,max_issue),RAD_INT("N",n),RAD_INT("K",k),RAD_STR("dtype","bf16")};
        op=rad_decl_op(builder.get(),f32?"gemm_nt_f32":"gemm_nt",params,4,&weight,1);need(op!=0,"declare existing plain op");
        okay(rad_op_reads(builder.get(),op,&x,1),"declare input lifetime");okay(rad_op_writes(builder.get(),op,&y,1),"declare output lifetime");
        okay(builder->finish(),"finish actual Builder");auto& program=builder->program();auto& wi=program.weights.at(weight);
        need(wi.widen==f32,"Builder widening must follow explicit declaration");
        need(wi.enc.n_planes==1&&wi.enc.plane[0].dtype==RAD_BF16&&wi.n_sel==1,"source encoding must remain BF16");
        need(wi.stored_dtype==(f32?RAD_F32:RAD_BF16)&&wi.stored_bytes==n*k*(f32?4:2),"Builder must bill the actual stored scalar width");
        need(wi.identity&&wi.layout_tag.empty(),"existing plain row must not require private relayout");
        for(int generation=0;generation<2;++generation){
            auto& raw=source[size_t(generation)];auto& wide=converted[size_t(generation)];raw.resize(size_t(n*k));wide.resize(size_t(n*k));
            for(int64_t row=0;row<n;++row)for(int64_t col=0;col<k;++col){const size_t at=size_t(row*k+col);const float value=float(w_numerator(row,col,generation))/16.0f;raw[at]=bf16(value);wide[at]=widen(raw[at]);need(wide[at]==value,"test source is not an exact BF16 dyadic");}
        }
        // This exact conversion belongs to the test scaffold, not a call into Engine loading.
        wi.ptr=f32?static_cast<void*>(converted[0].data()):static_cast<void*>(source[0].data());
        table.slots.resize(program.weights.size());table.slots.at(weight)={wi.ptr,nullptr,1};
        for(const auto& band:program.ops.at(op).bands){const auto& resolved=band.dom[RAD_DOMAIN_HOST];need(resolved.row&&resolved.row->plugin=="libref"&&resolved.row->info&&resolved.row->info->launch,"must select the real libref HOST row");launches.emplace_back(resolved.row->info,resolved.row->info->launch);}
        const int64_t arena=arena_plan_bytes(program,nullptr,nullptr)+align_up(arena_host_bytes(program),RAD_ALIGN_UNIT);need(arena>=0&&arena<=64*1024,"plain fixture arena cap");
        CtxDesc cd{};cd.program=&program;cd.rank=0;cd.world_size=1;cd.device=0;cd.arch_step=step;cd.record_passes=false;cd.residency=&table;
        okay(ctx.init(cd),"actual Ctx init");okay(ctx.bind_thread(),"actual Ctx thread binding");need(!ctx.checks_device_status(),"existing HOST rows use synchronous errors");
    }
    template<class T>T* ptr(rad_buf handle){auto* result=static_cast<T*>(rad_buf_ptr(&ctx,handle));need(result!=nullptr,"actual buffer binding");return result;}
    void run(int64_t rows,int generation){
        auto* input=ptr<uint16_t>(x);std::fill_n(input,max_rows*xpitch,uint16_t(0x7FC1));
        for(int64_t row=0;row<rows;++row)for(int64_t col=0;col<k;++col)input[(row+1)*xpitch+1+col]=bf16(float(x_numerator(row,col))/8.0f);
        const std::vector<uint16_t> original_input(input,input+max_rows*xpitch);
        const auto raw_before=source;const auto wide_before=converted;
        if(f32)std::fill_n(ptr<float>(y),max_rows*ypitch,-12345.0f);else std::fill_n(ptr<uint16_t>(y),max_rows*ypitch,uint16_t(0x7FC1));
        int status=RAD_E_STATE;
        body=[&](RadCtx* c){RadOperand operands[]{RAD_B_N(x,xpitch+1,rows),RAD_W(weight),RAD_B_N(y,ypitch+2,rows)};operands[0].cols=k;operands[2].cols=n;status=rad_issue(c,op,operands,3,rows);};
        RadBatch batch{};batch.phase=RAD_PHASE_PREFILL;batch.n_tok=rows;int rc=RAD_E_STATE;
        try{rc=ctx.run_step(&batch);}catch(...){body={};throw;}body={};
        need(rc==RAD_OK,std::string("Ctx plain projection issue: ")+ctx.step_error());okay(status,"existing row issue");okay(ctx.await_device_status(),"HOST await");okay(rad_stream_sync(ctx.stream()),"HOST drain");
        for(int64_t row=0;row<max_rows;++row)for(int64_t col=0;col<ypitch;++col){const bool active=row>=1&&row<rows+1&&col>=2&&col<2+n;const int64_t at=row*ypitch+col;const float want=active?expected(row-1,col-2,k,generation):-12345.0f;
            if(f32)CHECK_EQ(ptr<float>(y)[at],want);else CHECK_EQ(ptr<uint16_t>(y)[at],active?bf16(want):uint16_t(0x7FC1));
        }
        CHECK(std::equal(original_input.begin(),original_input.end(),input));CHECK(source==raw_before);CHECK(converted==wide_before);
        for(const auto& [info,launch]:launches)CHECK(info->launch==launch);
        CHECK_EQ(ctx.pass_counts().played,0);
    }
    void relocate(){table.move(weight,f32?static_cast<void*>(converted[1].data()):static_cast<void*>(source[1].data()));}
};
}

TEST(ds41_plain_projection_runtime_explicit_f32_declaration_and_existing_f32_output_row){
    if(!device_is_host())SKIP("HOST-only existing plain projection proof");
    Fixture fixture(true,17);fixture.run(1,0);fixture.run(3,0);fixture.relocate();fixture.run(3,1);
}
TEST(ds41_plain_projection_runtime_bf16_index_output_and_existing_generic_row){
    if(!device_is_host())SKIP("HOST-only existing BF16 projection proof");
    Fixture fixture(false,256);fixture.run(1,0);fixture.run(3,0);fixture.relocate();fixture.run(3,1);
}
RAD_TEST_MAIN()
