/* Explicit manual integration checker. Never an automatic *_test.cpp.
 * Uses unmodified registered rows through Builder -> Ctx -> rad_issue -> AQL. */
#include "build/rad_build.h"
#include "runtime/ctx.h"
#include "device/device.h"
#include "../libds41/ds41_route_jobs_fixture.h"
#include "../libds41/ds41_math_emulate.h"
#include "../libds41/ds41_compress_fixture.h"
#include "../libds41/ds41_ffn_fixture.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace rad;
namespace route=rad::ds41::route_jobs;
namespace math=rad::ds41::math;
namespace compression=rad::ds41::compress;
namespace ffn=rad::ds41::ffn;
namespace {
constexpr int64_t kGuard=256,kExplicitLimit=1<<20,kArenaLimit=128<<10;
constexpr uint8_t kPoison=0xbd;
int64_t live_explicit=0,peak_explicit=0;
size_t good_passes=0,bad_passes=0,published_passes=0,numeric_outputs=0;
void require(bool okay,const std::string& why){if(!okay)throw std::runtime_error(why);}
void check(int status,const char* operation){
    if(status<0)throw std::runtime_error(std::string(operation)+": "+rad_strerror(status));
}
void charge(int64_t bytes){
    require(bytes>0&&bytes<=kExplicitLimit&&live_explicit<=kExplicitLimit-bytes,"tiny explicit live allocation cap exceeded");
    live_explicit+=bytes;peak_explicit=std::max(peak_explicit,live_explicit);
}
struct Stream {
    RadStream value=nullptr;
    Stream(){check(rad_stream_create(&value,0),"create core stream");require(value!=nullptr,"null core stream");}
    ~Stream(){if(value)rad_stream_destroy(value);}
};
// The request count is explicit; runtime/page/driver metadata overhead is separate.
struct Owned {
    RadStream stream;void* allocation=nullptr;void* device=nullptr;void* host=nullptr;int64_t bytes;int kind;
    Owned(RadStream s,int64_t count,int memory):stream(s),bytes(count),kind(memory){
        require(bytes>0&&bytes<=65536,"individual fixture allocation is not tiny");
        charge(bytes);
        allocation=rad_dev_alloc(bytes,kind);
        if(!allocation){live_explicit-=bytes;throw std::runtime_error("tiny rad_dev_alloc failed");}
        device=kind==RAD_MEM_DEVICE?allocation:rad_dev_device_ptr(allocation);
        host=kind==RAD_MEM_DEVICE?nullptr:rad_dev_host_ptr(allocation);
        if(!device||(kind!=RAD_MEM_DEVICE&&!host)){
            rad_dev_free(allocation,kind);allocation=nullptr;live_explicit-=bytes;
            throw std::runtime_error("mapped fixture allocation has no checked host/device views");
        }
    }
    Owned(const Owned&)=delete;Owned& operator=(const Owned&)=delete;
    ~Owned(){
        if(!allocation)return;
        // Keep resources charged/alive if quiescence cannot be established.
        if(rad_stream_sync(stream)<0){std::fputs("fixture teardown: failed drain; allocation retained until process exit\n",stderr);return;}
        rad_dev_free(allocation,kind);live_explicit-=bytes;
    }
};
struct ArenaCharge {
    int64_t bytes=0;
    void take(int64_t n){require(bytes==0,"arena charged twice");charge(n);bytes=n;}
    ~ArenaCharge(){live_explicit-=bytes;}
};
int64_t elements(const RadTensor& t){
    require(t.rank>0&&t.rank<=RAD_MAX_RANK,"fixture rank is invalid");int64_t count=1,stride=1;
    for(int d=int(t.rank)-1;d>=0;--d){
        require(t.shape[d]>0&&t.shape[d]<=65536&&t.stride[d]==stride,"fixture must be small and densely packed");
        require(count<=65536/t.shape[d],"fixture element count is not tiny");
        count*=t.shape[d];stride*=t.shape[d];
    }
    return count;
}
struct Tensor {
    RadTensor shape{};std::vector<uint8_t> initial,expected;
    bool exists()const{return shape.rank!=0;}
};
struct Case {
    std::string name,op;int64_t rows=0;std::vector<RadParam> params;std::vector<Tensor> tensors;int expected_status=RAD_OK;
};
std::vector<uint8_t> payload(const RadTensor& t){
    const auto count=elements(t);const auto bytes=rad_dtype_bytes(t.dtype,count);
    require(t.data&&bytes>0&&bytes<=65536,"fixture payload is not bounded");
    std::vector<uint8_t> out(size_t(bytes),0);std::memcpy(out.data(),t.data,size_t(bytes));return out;
}
template<class Fixture>Case capture(std::string name,std::string op,int64_t rows,Fixture fixture){
    auto before=fixture.args();Case out;out.name=std::move(name);out.op=std::move(op);out.rows=rows;
    out.params.assign(before.p,before.p+before.n_p);out.tensors.resize(size_t(before.n_t));
    for(int i=0;i<before.n_t;++i)if(before.t[i].rank){
        auto& value=out.tensors[size_t(i)];value.shape=before.t[i];value.shape.data=nullptr;value.initial=payload(before.t[i]);
    }
    out.expected_status=fixture.run();const auto after=fixture.args();
    for(int i=0;i<after.n_t;++i)if(after.t[i].rank)out.tensors[size_t(i)].expected=payload(after.t[i]);
    return out;
}
uint16_t halfword(const uint8_t* p){uint16_t value;std::memcpy(&value,p,2);return value;}
float scalar(const uint8_t* p,uint32_t dtype){
    if(dtype==RAD_F32){float value;std::memcpy(&value,p,4);return value;}
    const auto value=halfword(p);return dtype==RAD_F16?rad_f16_to_f32(value):rad_bf16_to_f32(value);
}
int ordered(uint16_t value){return value&0x8000u?32768-int(value&0x7fff):32768+int(value);}
void compare_floats(const std::vector<uint8_t>& actual,const Tensor& reference,const std::string& label){
    const int width=rad_dtype_bits(reference.shape.dtype)/8;
    for(size_t at=0;at<actual.size();at+=size_t(width)){
        const float got=scalar(actual.data()+at,reference.shape.dtype),want=scalar(reference.expected.data()+at,reference.shape.dtype);
        if(std::isinf(want)){require(got==want,label+": infinite score/padding differs");continue;}
        require(std::isfinite(got)&&std::isfinite(want),label+": nonfinite numeric output");
        if(reference.shape.dtype==RAD_F32)
            require(std::abs(got-want)<=4e-6f+4e-5f*std::abs(want),label+": F32 output differs from component CPU oracle");
        else require(std::abs(ordered(halfword(actual.data()+at))-ordered(halfword(reference.expected.data()+at)))<=1,
                     label+": storage differs by more than one representable step");
        if(got==0&&want==0)require(std::signbit(got)==std::signbit(want),label+": zero sign differs");
        ++numeric_outputs;
    }
}
struct Binding {
    bool present=false,weight=false,output=false;uint32_t handle=0;int64_t offset_elements=0,offset_bytes=0,bytes=0,logical_bytes=0;
    void* base=nullptr;std::unique_ptr<Owned> storage,staging;
};
struct Harness;
Harness* active=nullptr;
void step(RadCtx*,const RadBatch*);
struct Harness {
    std::string op_name;int64_t rows;int32_t sequence_count=1;std::unique_ptr<Builder> builder;rad_op op=0;
    std::vector<Binding> bindings;std::vector<RadOperand> operands;ArenaCharge arena_charge;
    Ctx ctx; // Destroy/drain before staging, weight storage, Builder and charged arena receipt.
    Harness(Registry& registry,const Case& initial,RadStream stream,bool mapped_weights):op_name(initial.op),rows(initial.rows){
        require(initial.expected_status==RAD_OK,"Harness needs a valid initial case");
        if(op_name=="ds41_compress_pool")sequence_count=int32_t(initial.tensors[compression::cu].shape.shape[0]-1);
        const auto* schema=registry.schema(op_name);require(schema&&schema->n_operands==int(initial.tensors.size()),"real op schema is absent/wrong arity");
        RadModelMeta meta{};meta.arch_id="ds41-aql-math-state-fixture";meta.name="tiny-synthetic";meta.quant="native";
        RadBuildCtx bc{};bc.rank=0;bc.world_size=1;bc.max_tok=rad_param_geti(initial.params.data(),int(initial.params.size()),"M",rows);
        bc.max_seqs=8;bc.max_ctx=64;bc.scope="";
        require(bc.max_tok>=rows&&bc.max_tok<=64,"bad tiny capacity");
        builder=std::make_unique<Builder>(registry,meta,bc);builder->set_instances(false);
        bindings.resize(initial.tensors.size());operands.resize(initial.tensors.size());std::vector<rad_weight> weights;
        for(size_t i=0;i<bindings.size();++i){
            auto& b=bindings[i];const auto& input=initial.tensors[i];const auto role=schema->operands[i].role;
            b.output=role==RAD_OPD_OUT||role==RAD_OPD_INOUT;
            if(!input.exists()){require(schema->operands[i].optional,"required fixture operand absent");operands[i]=RAD_NONE;continue;}
            b.present=true;b.weight=role==RAD_OPD_WEIGHT;require(role!=RAD_OPD_WTAB,"unexpected weight-table fixture");
            b.logical_bytes=int64_t(input.initial.size());
            if(b.weight){
                RadWeightDecl decl{};decl.dtype=input.shape.dtype;decl.rank=input.shape.rank;decl.access=RAD_ACCESS_PER_TOKEN;decl.group={-1,-1,0};
                for(unsigned d=0;d<decl.rank;++d)decl.shape[d]=input.shape.shape[d];
                b.handle=builder->decl_weight(("fixture.weight."+std::to_string(i)).c_str(),&decl);require(b.handle!=0,"Builder refused real weight");
                weights.push_back(b.handle);b.offset_bytes=kGuard;b.bytes=b.logical_bytes+2*kGuard;
                b.storage=std::make_unique<Owned>(stream,b.bytes,mapped_weights?RAD_MEM_HOST_MAPPED:RAD_MEM_DEVICE);b.base=b.storage->device;
                operands[i]=RAD_W(b.handle);
            }else{
                const int width=rad_dtype_bits(input.shape.dtype)/8;require(width>0,"opaque activation fixture refused");
                const int64_t row_elements=elements(input.shape)/input.shape.shape[0],row_bytes=row_elements*width;
                const int64_t guard_rows=(kGuard+row_bytes-1)/row_bytes;
                RadBufDecl decl{};decl.dtype=input.shape.dtype;decl.rank=input.shape.rank;decl.kind=RAD_BUF_PERSIST;decl.domain=RAD_DOMAIN_DEVICE;
                for(unsigned d=0;d<decl.rank;++d)decl.shape[d]=input.shape.shape[d];
                decl.shape[0]+=2*guard_rows;b.offset_elements=guard_rows*row_elements;b.offset_bytes=b.offset_elements*width;
                b.bytes=b.logical_bytes+2*b.offset_bytes;
                b.handle=builder->decl_buffer(("fixture.tensor."+std::to_string(i)).c_str(),&decl);require(b.handle!=0,"Builder refused actual shaped buffer");
                operands[i]=RAD_B_N(b.handle,b.offset_elements,input.shape.shape[0]);
            }
            b.staging=std::make_unique<Owned>(stream,b.bytes,RAD_MEM_HOST_MAPPED);
        }
        auto params=initial.params;
        for(auto& p:params)if(!std::strcmp(p.key,"M"))p=RAD_RANGE("M",1,p.ival);
        op=builder->decl_op(op_name.c_str(),params.data(),int(params.size()),weights.data(),int(weights.size()));
        require(op!=0,"Builder refused actual op");check(builder->finish(),"finish actual op Builder");
        auto& program=builder->program();
        require(program.host_arena_bytes==0,"unexpected host activation arena");
        const bool latent=op_name=="ds41_latent_store"||op_name=="ds41_latent_gather";
        require(program.scratch_bytes>=(latent?0:4)&&program.scratch_bytes<=65536,"missing/big op scratch reservation");
        for(const auto& band:program.ops[op].bands){
            const auto& resolved=band.dom[RAD_DOMAIN_DEVICE];
            require(resolved.row&&resolved.row->info&&resolved.row->plugin=="libds41"&&resolved.row->info->launch&&resolved.row->deferred_status,
                    "real libds41 device row/collector did not resolve");
        }
        for(auto& b:bindings)if(b.present&&b.weight){
            auto& w=program.weights[b.handle];require(w.stored_bytes==b.logical_bytes&&w.stored_dtype==w.decl.dtype,"unexpected plain fixture weight layout");
            w.ptr=static_cast<uint8_t*>(b.base)+b.offset_bytes;
        }
        const auto arena_bytes=arena_plan_bytes(program,nullptr,nullptr);require(arena_bytes>0&&arena_bytes<=kArenaLimit,"actual Ctx arena is not tiny");
        arena_charge.take(arena_bytes);
        CtxDesc cd{};cd.program=&program;cd.rank=0;cd.world_size=1;cd.device=0;cd.stream=stream;cd.arch_step=step;cd.record_passes=true;
        check(ctx.init(cd),"initialize actual Ctx");check(ctx.bind_thread(),"bind actual rank thread");
        require(ctx.stream()==stream&&ctx.checks_device_status()&&!ctx.records_passes(),"actual stream/status/tape contract missing");
        for(auto& b:bindings)if(b.present&&!b.weight){b.base=rad_buf_ptr(&ctx,b.handle);require(b.base!=nullptr,"actual arena binding absent");}
        std::printf("  %s input=%s ratio=%lld mapped_weights=%d: Ctx_arena_and_scratch=%lld B, aggregate_explicit_live=%lld B; no launch replacement\n",
                    op_name.c_str(),rad_dtype_name(initial.tensors[0].shape.dtype),
                    rad_param_geti(initial.params.data(),int(initial.params.size()),"ratio",0),int(mapped_weights),
                    (long long)arena_bytes,(long long)live_explicit);
    }
    void pass(const Case& test,int ordinal,int retained_operand=-1){
        require(test.op==op_name&&test.rows==rows&&test.tensors.size()==bindings.size(),"case/Harness geometry differs");
        if(retained_operand>=0){
            require(size_t(retained_operand)<bindings.size(),"retained state operand out of range");
            auto& state=bindings[size_t(retained_operand)];require(state.present&&state.output,"retained operand is not mutable state");
            check(rad_memcpy_async(state.staging->device,state.base,state.bytes,ctx.stream()),"inspect retained GPU state before continuation");
            check(rad_stream_sync(ctx.stream()),"finish retained-state inspection");
            const auto* raw=static_cast<const uint8_t*>(state.staging->host);
            require(std::vector<uint8_t>(raw+state.offset_bytes,raw+state.offset_bytes+state.logical_bytes)==test.tensors[size_t(retained_operand)].initial,
                    "actual retained GPU state differs from expected prior chunk; refusing artificial continuation");
        }
        for(size_t i=0;i<bindings.size();++i){
            auto& b=bindings[i];const auto& input=test.tensors[i];require(b.present==input.exists(),"case optionality differs");if(!b.present)continue;
            require(int64_t(input.initial.size())==b.logical_bytes,"case payload extent differs");
            if(int(i)==retained_operand)continue; // Preserve actual device state, not a CPU re-upload.
            std::memset(b.staging->host,kPoison,size_t(b.bytes));std::memcpy(static_cast<uint8_t*>(b.staging->host)+b.offset_bytes,input.initial.data(),size_t(b.logical_bytes));
            rad_dev_host_wrote();check(rad_memcpy_async(b.base,b.staging->device,b.bytes,ctx.stream()),"upload mapped, owned fixture staging");
        }
        const auto publication_before=published_passes;active=this;
        RadBatch batch{};batch.phase=RAD_PHASE_PREFILL;batch.n_tok=rows;batch.n_seq=sequence_count;batch.step=ordinal;
        const int submitted=ctx.run_step(&batch);active=nullptr;
        require(submitted>=0,std::string("actual rad_issue failed synchronously for ")+test.name+": "+ctx.step_error());
        const int verified=ctx.await_device_status();
        require(verified==test.expected_status,test.name+": deferred result="+std::to_string(verified)+
                ", expected="+std::to_string(test.expected_status)+", pass="+std::to_string(ordinal));
        if(verified<0){++bad_passes;require(published_passes==publication_before,"failed pass reached publication");}
        else {++good_passes;++published_passes;}
        // Diagnostic readback is after the validation gate; it is not publication.
        check(rad_stream_sync(ctx.stream()),"finish actual AQL pass");
        for(size_t i=0;i<bindings.size();++i){
            auto& b=bindings[i];if(!b.present)continue;const auto& reference=test.tensors[i];
            check(rad_memcpy_async(b.staging->device,b.base,b.bytes,ctx.stream()),"read guarded arena/weight fixture");
            check(rad_stream_sync(ctx.stream()),"finish mapped fixture readback");
            const auto* bytes=static_cast<const uint8_t*>(b.staging->host);
            require(std::all_of(bytes,bytes+b.offset_bytes,[](uint8_t v){return v==kPoison;})&&
                    std::all_of(bytes+b.offset_bytes+b.logical_bytes,bytes+b.bytes,[](uint8_t v){return v==kPoison;}),test.name+": shaped-buffer/weight redzone changed");
            std::vector<uint8_t> actual(bytes+b.offset_bytes,bytes+b.offset_bytes+b.logical_bytes);
            const bool floating=reference.shape.dtype==RAD_F32||reference.shape.dtype==RAD_BF16||reference.shape.dtype==RAD_F16;
            const bool latent=op_name=="ds41_latent_store"||op_name=="ds41_latent_gather";
            const bool explicit_status=latent&&i+1==bindings.size();
            if(!b.output||verified<0||!floating||latent)
                require(actual==(b.output&&(verified==RAD_OK||explicit_status)?reference.expected:reference.initial),test.name+": read-only/transactional/exact output bytes changed");
            else compare_floats(actual,reference,test.name+" operand "+std::to_string(i));
        }
        require(ctx.pass_counts().played==0&&!ctx.records_passes(),"validation-opted op entered replay");
    }
};
void step(RadCtx* context,const RadBatch*){
    if(!active){rad_step_fail(context,"missing tiny fixture owner");return;}
    (void)rad_issue(context,active->op,active->operands.data(),int(active->operands.size()),active->rows);
}
template<class Good,class Bad>void exercise(Registry& registry,RadStream stream,Good good,Bad bad,bool mapped_weights=false){
    require(good.expected_status==RAD_OK&&bad.expected_status<0,"checker fixture status construction failed");
    Harness fixture(registry,good,stream,mapped_weights);fixture.pass(good,0);fixture.pass(bad,1);fixture.pass(good,2);
}
struct MathFixture {
    math::Op op;int64_t rows=2;std::vector<route::testing::Buffer> buffers;std::vector<RadTensor> tensors;std::vector<RadParam> params;std::array<int32_t,4> flag{911,912,913,914};
    explicit MathFixture(math::Op kind):op(kind){
        auto add=[&](uint32_t dt,std::initializer_list<int64_t> shape){buffers.emplace_back(dt,shape,0);};int first_output=0;
        if(op==math::Op::hc_prepare||op==math::Op::hc_collapse){
            params={RAD_INT("M",4),RAD_INT("n",3),RAD_INT("hc",4),RAD_STR("dtype","f32")};add(RAD_F32,{2,12});
            if(op==math::Op::hc_prepare){params.insert(params.end(),{RAD_INT("iterations",3),RAD_F64("eps",1e-6),RAD_F64("hc_eps",1e-6)});
                add(RAD_F32,{24,12});add(RAD_F32,{3});add(RAD_F32,{24});add(RAD_F32,{2,4});add(RAD_F32,{2,4});add(RAD_F32,{2,16});first_output=4;}
            else {add(RAD_F32,{2,4});add(RAD_F32,{2,3});first_output=2;}
        }else if(op==math::Op::router_topk){
            params={RAD_INT("M",4),RAD_INT("experts",5),RAD_INT("topk",2),RAD_F64("temperature",1),RAD_F64("route_scale",1),RAD_INT("normalize",1)};
            add(RAD_F32,{2,5});add(RAD_F32,{5});add(RAD_I32,{2,2});add(RAD_F32,{2,2});first_output=2;
        }else{
            params={RAD_INT("M",4),RAD_INT("heads",2),RAD_INT("head_dim",4),RAD_INT("kv_rows",5),RAD_INT("topk",3),RAD_F64("scale",.5),RAD_STR("dtype","f32")};
            add(RAD_F32,{2,2,4});add(RAD_F32,{5,4});add(RAD_F32,{2});add(RAD_I32,{2,3});add(RAD_F32,{2,2,4});first_output=4;
        }
        for(int i=0;i<first_output;++i)for(int64_t j=0;j<buffers[size_t(i)].count();++j)
            if(buffers[size_t(i)].t.dtype==RAD_F32)buffers[size_t(i)].weight(j,float(int((j*7+i*11)%17)-8)/16.f);
        if(op==math::Op::hc_prepare){buffers[2].weight(0,.6f);buffers[2].weight(1,.9f);buffers[2].weight(2,1.2f);}
        if(op==math::Op::sparse_attn){const int32_t ids[]{-1,0,2,1,1,4};for(int i=0;i<6;++i)buffers[3].id(i,ids[i]);buffers[2].weight(1,-INFINITY);}
    }
    RadArgs args(){tensors.clear();for(auto& b:buffers){b.t.data=b.words.data();tensors.push_back(b.t);}RadArgs a{};a.t=tensors.data();a.n_t=int(tensors.size());a.p=params.data();a.n_p=int(params.size());a.scratch=flag.data();a.scratch_bytes=4;return a;}
    int run(){const auto a=args();return math::emulate(op,&a);}
};
// Deliberately opaque scalar payloads: the latent row must copy all IEEE bits,
// including NaN/subnormal/sign bits, without interpreting them as arithmetic.
struct LatentFixture {
    struct Raw {
        RadTensor t{};std::vector<uint64_t> words;
        Raw(uint32_t dtype,std::initializer_list<int64_t> shape){
            t.dtype=dtype;t.rank=shape.size();unsigned at=0;for(auto d:shape)t.shape[at++]=d;rad_tensor_pack(&t);
            int64_t n=1;for(auto d:shape)n*=d;const auto bytes=rad_dtype_bytes(dtype,n);words.resize(size_t((bytes+7)/8)+8,0xA5A5A5A5A5A5A5A5ull);t.data=words.data();
        }
        void set64(int64_t at,int64_t value){std::memcpy(reinterpret_cast<uint8_t*>(words.data())+at*8,&value,8);}
        int64_t get64(int64_t at)const{int64_t value;std::memcpy(&value,reinterpret_cast<const uint8_t*>(words.data())+at*8,8);return value;}
    };
    bool store;int64_t rows=3,dimension=8,ratio,block=4,blocks=3;uint32_t dtype;std::vector<Raw> buffers;std::vector<RadTensor> tensors;std::vector<RadParam> params;
    explicit LatentFixture(bool writing,int64_t compression,uint32_t dt):store(writing),ratio(compression),dtype(dt){
        params={RAD_INT("M",5),RAD_INT("dimension",dimension),RAD_INT("ratio",ratio),RAD_INT("block_size",block),RAD_INT("n_blocks",blocks),RAD_STR("dtype",rad_dtype_name(dtype))};
        const auto add=[&](uint32_t d,std::initializer_list<int64_t> shape){buffers.emplace_back(d,shape);};
        if(store){add(dtype,{rows,dimension});add(RAD_I64,{rows});add(RAD_I64,{rows});add(dtype,{blocks,1,block,dimension/ratio});add(RAD_I32,{rows});}
        else {add(dtype,{blocks,1,block,dimension/ratio});add(RAD_I64,{rows});add(dtype,{rows,dimension});add(RAD_I32,{rows});}
        auto& source=buffers[0];const int64_t source_bytes=store?rows*dimension*(rad_dtype_bits(dtype)/8):blocks*block*(dimension/ratio)*(rad_dtype_bits(dtype)/8);
        for(int64_t i=0;i<source_bytes;++i)reinterpret_cast<uint8_t*>(source.words.data())[i]=uint8_t((i*71+19)%256);
        for(int64_t row=0;row<rows;++row){
            if(store){buffers[1].set64(row,100*block+ratio-1);buffers[2].set64(row,row*block+ratio-1);}
            else buffers[1].set64(row,row*block);
        }
    }
    RadArgs args(){tensors.clear();for(auto& b:buffers){b.t.data=b.words.data();tensors.push_back(b.t);}RadArgs a{};a.t=tensors.data();a.n_t=int(tensors.size());a.p=params.data();a.n_p=int(params.size());return a;}
    int run(){
        int error=RAD_OK;const int slots_operand=store?2:1;
        for(int64_t row=0;row<rows&&error==RAD_OK;++row){const int64_t slot=buffers[size_t(slots_operand)].get64(row);
            if(store){const auto position=buffers[1].get64(row);
                if(position<0||position%ratio!=ratio-1||slot<ratio-1||slot>=blocks*block||slot%block!=position%block)error=RAD_E_SHAPE;
                if(error==RAD_OK)for(int64_t prior=0;prior<row;++prior)if(buffers[2].get64(prior)==slot)error=RAD_E_INVAL;
            }else if(slot<0||slot>blocks*block-ratio||slot%block%ratio)error=RAD_E_SHAPE;
        }
        for(int64_t row=0;row<rows;++row)std::memcpy(reinterpret_cast<uint8_t*>(buffers.back().words.data())+row*4,&error,4);
        if(error!=RAD_OK)return error;
        const int width=rad_dtype_bits(dtype)/8;
        for(int64_t row=0;row<rows;++row){const auto first=buffers[size_t(slots_operand)].get64(row)-(store?ratio-1:0);
            auto* cache=reinterpret_cast<uint8_t*>(buffers[size_t(store?3:0)].words.data());auto* x=reinterpret_cast<uint8_t*>(buffers[size_t(store?0:2)].words.data());
            if(store)std::memcpy(cache+first*(dimension/ratio)*width,x+row*dimension*width,size_t(dimension*width));
            else std::memcpy(x+row*dimension*width,cache+first*(dimension/ratio)*width,size_t(dimension*width));
        }
        return RAD_OK;
    }
};
void run_cases(Registry& registry,RadStream stream){
    const char* math_names[]{"ds41_hc_prepare","ds41_hc_collapse","ds41_router_topk","ds41_sparse_attn"};
    for(int op=0;op<4;++op){
        MathFixture good{math::Op(op)};auto bad=good;bad.buffers[0].weight(bad.buffers[0].count()-1,NAN);
        exercise(registry,stream,capture("math valid",math_names[op],2,good),capture("math NaN preflight",math_names[op],2,bad));
    }
    for(const auto profile:{std::array<uint32_t,2>{1,RAD_F32},{1,RAD_BF16},{2,RAD_F32}}){
        const int64_t ratio=profile[0];const uint32_t dtype=profile[1];
        compression::testing::Fixture first(5,ratio,dtype);
        const auto initial=capture("compressor initial chunk","ds41_compress_pool",first.tokens,first);
        Harness harness(registry,initial,stream,false);harness.pass(initial,0);
        const auto continue_case=[&](const Case& previous,std::vector<int32_t> starts){
            compression::testing::Fixture next(5,ratio,dtype,0,{2,2,0},std::move(starts),{0,2,-1});
            const auto& state=previous.tensors[compression::pending_state].expected;
            require(state.size()<=next.buffer(compression::pending_state).words.size()*4,"CPU continuation state extent mismatch");
            std::memcpy(next.buffer(compression::pending_state).words.data(),state.data(),state.size());return next;
        };
        auto warm=continue_case(initial,{3,1,0});
        const auto valid_warm=capture("compressor warm continuation","ds41_compress_pool",warm.tokens,warm);
        auto malformed=warm;malformed.buffer(compression::cu).seti(malformed.sequences,int32_t(malformed.tokens+1));
        const auto bad=capture("compressor invalid warm cu","ds41_compress_pool",malformed.tokens,malformed);
        require(bad.expected_status<0,"compressor malformed fixture was accepted");
        harness.pass(bad,1,compression::pending_state);
        harness.pass(valid_warm,2,compression::pending_state);
        auto third=continue_case(valid_warm,{5,3,0});
        harness.pass(capture("compressor second warm continuation","ds41_compress_pool",third.tokens,third),3,compression::pending_state);
        harness.pass(initial,4); // Explicit cold reset after genuine retained-state continuations.
    }
    const char* ffn_names[]{"ds41_expert_act","ds41_moe_combine","mhc_post"};
    for(int op=0;op<3;++op)for(uint32_t dtype:{RAD_F32,RAD_BF16}){
        ffn::testing::Fixture good{ffn::Op(op),dtype};auto bad=good;bad.buffers[0].setf(bad.buffers[0].count()-1,NAN);
        const int64_t rows=op==2?2:3;
        exercise(registry,stream,capture("FFN valid",ffn_names[op],rows,good),capture("FFN NaN preflight",ffn_names[op],rows,bad));
    }
    for(bool store:{false,true})for(int ratio:{1,2})for(uint32_t dtype:{RAD_F32,RAD_BF16}){
        LatentFixture good(store,ratio,dtype),bad=good;bad.buffers[size_t(store?2:1)].set64(2,good.blocks*good.block);
        const char* op=store?"ds41_latent_store":"ds41_latent_gather";
        exercise(registry,stream,capture("opaque latent copy valid",op,3,good),capture("opaque latent invalid physical slot",op,3,bad));
    }
}

void usage(){std::puts("ds41-aql-math-state-check --run-gpu --home ISOLATED_BUILD_HOME [--host-kargs]\nTiny actual Builder/Ctx/AQL math/state/FFN check; exactly one visible gfx1201; no model, Pools or service.");}
} // namespace
int main(int argc,char** argv){
    bool approved=false,host_kargs=false;std::string home;
    // No explicit device query, Registry, stream, allocation or launch before this gate.
    for(int i=1;i<argc;++i){const std::string argument=argv[i];
        if(argument=="--help"||argument=="-h"){usage();return 0;}
        if(argument=="--run-gpu")approved=true;
        else if(argument=="--host-kargs")host_kargs=true;
        else if(argument=="--home"&&i+1<argc)home=argv[++i];
        else {std::fprintf(stderr,"unknown/incomplete option: %s\n",argument.c_str());return 2;}
    }
    if(!approved||home.empty()){usage();std::fputs("Explicit --run-gpu and isolated --home are required.\n",stderr);return 2;}
    try{
#if defined(RAD_HAVE_HIP) && RAD_HAVE_HIP
        if(host_kargs)check(aql_force_host_kargs(),"select host argument rings");
#else
        (void)host_kargs;throw std::runtime_error("checker must link HIP-enabled rad_core");
#endif
        require(rad_dev_count()==1,"expose exactly one GPU with ROCR_VISIBLE_DEVICES");
        require(std::strcmp(device_backend_name(),"aql")==0&&!device_is_host(),"actual AQL backend required; host fallback is not a pass");
        RadDeviceProps props{};check(rad_dev_props(0,&props),"read selected card");
        require(!props.is_host_backend&&std::strcmp(props.arch,"gfx1201")==0,"requires one visible gfx1201");check(rad_dev_set(0),"select only visible card");
        Registry registry;check(registry.load_plugin(home+"/quantizers/libquant.so",0),"load isolated libquant");
        check(registry.load_plugin(home+"/experimental/kernels/libds41.so",0),"load isolated real libds41");
        Stream stream;run_cases(registry,stream.value);check(rad_stream_sync(stream.value),"final core AQL drain");
        require(live_explicit==0,"fixture allocations/arena receipts were not released");
        require(bad_passes==21&&good_passes==48&&published_passes==good_passes,"checker pass/publication count drifted");
        std::printf("PASS: actual math/FFN rows and retained compressor state on core AQL; %zu good/%zu refused passes, %zu publications, %zu numeric outputs; peak charged fixture bytes=%lld.\n",
                    good_passes,bad_passes,published_passes,numeric_outputs,(long long)peak_explicit);
        std::puts("Byte cap includes padded Ctx arena/scratch and requested weight/mapped staging extents; excludes driver/page metadata, core stream argument/bounce rings and fixed Ctx validation resources. No serving/full graph/throughput claim.");
        return 0;
    }catch(const std::exception& error){active=nullptr;std::fprintf(stderr,"FAIL: %s\n",error.what());return 1;}
}
