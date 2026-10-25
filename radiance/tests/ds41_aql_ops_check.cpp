/* Explicit manual integration checker. Never an automatic *_test.cpp.
 * Uses unmodified registered rows through Builder -> Ctx -> rad_issue -> AQL. */
#include "build/rad_build.h"
#include "runtime/ctx.h"
#include "device/device.h"
#include "../libds41/ds41_route_jobs_fixture.h"
#include "../libds41/ds41_index_stream_fixture.h"
#include "../libds41/ds41_transform_fixture.h"
#include "../libds41/ds41_index_paged_fixture.h"
#include "../libds41/ds41_attention_paged_fixture.h"
#include "../libds41/ds41_compress_finish_fixture.h"
#include "../libds41/ds41_latent_emit_fixture.h"
#include "../libds41/ds41_engram_gate_fixture.h"
#include "../libds41/ds41_qdq_fixture.h"
#include "../libds41/ds41_math_emulate.h"
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
namespace transform=rad::ds41::transform;
namespace indexer=rad::ds41::index;
namespace paged=rad::ds41::paged_index;
namespace attention=rad::ds41::attention_paged;
namespace math=rad::ds41::math;
namespace {
constexpr int64_t kGuard=256,kExplicitLimit=1<<20,kArenaLimit=128<<10;
constexpr uint8_t kPoison=0xbd;
int64_t live_explicit=0,peak_explicit=0;
size_t good_passes=0,bad_passes=0,published_passes=0,numeric_outputs=0,math_fault_cycles=0;
size_t exact_qdq_outputs=0,qdq_subnormal_proofs=0;
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
template<class Fixture>Case capture(std::string name,std::string op,Fixture fixture){
    auto before=fixture.args();Case out;out.name=std::move(name);out.op=std::move(op);
    if constexpr(requires{fixture.rows;})out.rows=fixture.rows;else out.rows=fixture.tokens;
    out.params.assign(before.p,before.p+before.n_p);out.tensors.resize(size_t(before.n_t));
    for(int i=0;i<before.n_t;++i)if(before.t[i].rank){
        auto& value=out.tensors[size_t(i)];value.shape=before.t[i];value.shape.data=nullptr;value.initial=payload(before.t[i]);
    }
    out.expected_status=fixture.run();const auto after=fixture.args();
    for(int i=0;i<after.n_t;++i)if(after.t[i].rank)out.tensors[size_t(i)].expected=payload(after.t[i]);
    return out;
}
enum class MathFault{prepare,collapse_arithmetic,collapse_cast,router_quotient,router_scale,sparse_numerator};
struct MathFixture{
    math::Op op;int64_t rows=3;
    std::vector<transform::testing::Buffer> buffers;std::vector<RadTensor> tensors;
    std::vector<RadParam> params;std::array<int32_t,4> status{911,912,913,914};
    MathFixture(math::Op operation,uint32_t dtype,MathFault fault,bool broken):op(operation){
        auto add=[&](uint32_t dt,std::initializer_list<int64_t> shape){buffers.emplace_back(dt,shape,0);};
        if(op==math::Op::hc_prepare||op==math::Op::hc_collapse){
            params={RAD_INT("M",4),RAD_INT("n",3),RAD_INT("hc",2),RAD_STR("dtype",rad_dtype_name(dtype))};
            add(dtype,{rows,6});
            if(op==math::Op::hc_prepare){
                params.push_back(RAD_INT("iterations",2));params.push_back(RAD_F64("eps",1e-6));params.push_back(RAD_F64("hc_eps",1e-6));
                add(RAD_F32,{8,6});add(RAD_F32,{3});add(RAD_F32,{8});add(RAD_F32,{rows,2});add(RAD_F32,{rows,2});add(RAD_F32,{rows,4});
            }else{add(RAD_F32,{rows,2});add(dtype,{rows,3});}
        }else if(op==math::Op::router_topk){
            const int64_t topk=fault==MathFault::router_scale?1:2;
            params={RAD_INT("M",4),RAD_INT("experts",4),RAD_INT("topk",topk),
                    RAD_F64("temperature",fault==MathFault::router_quotient?std::ldexp(1.,-126):1.),
                    RAD_F64("route_scale",fault==MathFault::router_scale?std::ldexp(1.,66):1.5),RAD_INT("normalize",1)};
            add(RAD_F32,{rows,4});add(RAD_F32,{4});add(RAD_I32,{rows,topk});add(RAD_F32,{rows,topk});
        }else{
            params={RAD_INT("M",4),RAD_INT("heads",2),RAD_INT("head_dim",4),RAD_INT("kv_rows",3),RAD_INT("topk",2),RAD_F64("scale",.5),RAD_STR("dtype",rad_dtype_name(dtype))};
            add(dtype,{rows,2,4});add(dtype,{3,4});add(RAD_F32,{2});add(RAD_I32,{rows,2});add(dtype,{rows,2,4});
        }
        const int first=first_output();
        for(int i=0;i<first;++i)for(int64_t j=0;j<buffers[size_t(i)].count();++j){
            if(buffers[size_t(i)].t.dtype==RAD_I32)buffers[size_t(i)].id(j,0);else buffers[size_t(i)].set(j,.125f);
        }
        if(op==math::Op::router_topk)for(int64_t i=0;i<4;++i)buffers[1].set(i,0.f);
        if(broken&&fault==MathFault::prepare){
            for(int64_t i=0;i<rows*6;++i)buffers[0].set(i,0.f);
            for(int64_t i=0;i<8*6;++i)buffers[1].set(i,std::ldexp(1.f,dtype==RAD_F16?127:100));
            for(int64_t i=(rows-1)*6;i<rows*6;++i)buffers[0].set(i,dtype==RAD_F16?2.f:std::ldexp(1.f,100));
        }
        if(broken&&(fault==MathFault::collapse_arithmetic||fault==MathFault::collapse_cast)){
            for(int64_t i=0;i<rows*2;++i)buffers[1].set(i,0.f);
            buffers[1].set(rows*2-1,fault==MathFault::collapse_cast&&dtype==RAD_BF16?257.f/256.f:2.f);
            const float large=fault==MathFault::collapse_cast?(dtype==RAD_F16?65504.f:rad_bf16_to_f32(0x7f7f)):std::ldexp(1.f,127);
            buffers[0].set(rows*6-1,large);
        }
        if(broken&&(fault==MathFault::router_quotient||fault==MathFault::router_scale)){
            for(int64_t i=0;i<rows*4;++i)buffers[0].set(i,0.f);
            buffers[0].set(rows*4-1,fault==MathFault::router_quotient?4.f:std::ldexp(1.f,126));
        }
        if(broken&&fault==MathFault::sparse_numerator){
            for(int64_t i=0;i<rows*2*4;++i)buffers[0].set(i,0.f);
            buffers[2].set(0,std::numeric_limits<float>::max());buffers[2].set(1,-INFINITY);
            buffers[3].id((rows-1)*2,1);buffers[3].id((rows-1)*2+1,2);
            for(int64_t i=4;i<12;++i)buffers[1].set(i,std::ldexp(1.f,127));
        }
    }
    int first_output()const{return op==math::Op::hc_prepare?4:op==math::Op::sparse_attn?4:2;}
    RadArgs args(){tensors.clear();for(auto& buffer:buffers){buffer.t.data=buffer.words.data();tensors.push_back(buffer.t);}RadArgs a{};a.t=tensors.data();a.n_t=int(tensors.size());a.p=params.data();a.n_p=int(params.size());a.scratch=status.data();a.scratch_bytes=math::scratch_bytes;return a;}
    int run(){auto a=args();return math::emulate(op,&a);}
    void check_finite_witness(){auto a=args();math::Plan p;require(math::validate(op,&a,&p)==RAD_OK,"math derived witness metadata invalid");for(int64_t i=0;i<math::preflight_items(p);++i)require(math::check_item(p,i)==RAD_OK,"math witness has an input payload error");}
};
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
    std::vector<uint8_t> last_contents;
};
struct Harness;
Harness* active=nullptr;
void step(RadCtx*,const RadBatch*);
struct Harness {
    std::string op_name;int64_t rows;std::unique_ptr<Builder> builder;rad_op op=0;
    std::vector<Binding> bindings;std::vector<RadOperand> operands;ArenaCharge arena_charge;
    Ctx ctx; // Destroy/drain before staging, weight storage, Builder and charged arena receipt.
    Harness(Registry& registry,const Case& initial,RadStream stream,bool mapped_weights):op_name(initial.op),rows(initial.rows){
        require(initial.expected_status==RAD_OK,"Harness needs a valid initial case");
        const auto* schema=registry.schema(op_name);require(schema&&schema->n_operands==int(initial.tensors.size()),"real op schema is absent/wrong arity");
        RadModelMeta meta{};meta.arch_id="ds41-aql-ops-fixture";meta.name="tiny-synthetic";meta.quant="native";
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
        require(program.scratch_bytes>=4&&program.scratch_bytes<=65536,"missing/big scalar scratch reservation");
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
    void pass(const Case& test,int ordinal,bool preserve_outputs=false){
        require(test.op==op_name&&test.rows==rows&&test.tensors.size()==bindings.size(),"case/Harness geometry differs");
        for(size_t i=0;i<bindings.size();++i){
            auto& b=bindings[i];const auto& input=test.tensors[i];require(b.present==input.exists(),"case optionality differs");if(!b.present)continue;
            require(int64_t(input.initial.size())==b.logical_bytes,"case payload extent differs");
            if(preserve_outputs&&b.output){require(int64_t(b.last_contents.size())==b.logical_bytes,"cannot preserve output before a completed pass");continue;}
            std::memset(b.staging->host,kPoison,size_t(b.bytes));std::memcpy(static_cast<uint8_t*>(b.staging->host)+b.offset_bytes,input.initial.data(),size_t(b.logical_bytes));
            rad_dev_host_wrote();check(rad_memcpy_async(b.base,b.staging->device,b.bytes,ctx.stream()),"upload mapped, owned fixture staging");
        }
        const auto publication_before=published_passes;active=this;
        RadBatch batch{};batch.phase=RAD_PHASE_PREFILL;batch.n_tok=rows;batch.n_seq=1;batch.step=ordinal;
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
            const bool failure_status=op_name=="ds41_latent_store_emitted"&&i==rad::ds41::latent_emit::status;
            if(preserve_outputs&&b.output&&verified<0)
                require(actual==b.last_contents,test.name+": late derived fault changed an earlier completed output");
            else if(!b.output||verified<0||!floating||op_name=="ds41_route_jobs"||op_name=="ds41_latent_store_emitted"||op_name=="ds41_qdq_candidate")
                require(actual==(b.output&&(verified==RAD_OK||failure_status)?reference.expected:reference.initial),test.name+": read-only/transactional/exact output bytes changed");
            else compare_floats(actual,reference,test.name+" operand "+std::to_string(i));
            if(verified==RAD_OK&&op_name=="ds41_qdq_candidate"&&i==rad::ds41::qdq::output){
                exact_qdq_outputs+=actual.size()/2;
                if(!std::strcmp(rad_param_gets(test.params.data(),int(test.params.size()),"profile",""),"index_fp4")){
                    require(halfword(actual.data())==0x0040&&halfword(actual.data()+2)==0x8040,"actual AQL QDQ flushed index +/-2^-127 to zero");
                    qdq_subnormal_proofs+=2;
                }
            }
            b.last_contents=std::move(actual);
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
void run_cases(Registry& registry,RadStream stream){
    // Existing four baseline rows, not additional registered operations. Reuse each actual
    // Ctx, device buffers and status allocation for good -> late derived fault -> good.
    auto derived=[&](math::Op op,uint32_t dtype,MathFault fault){
        const char* names[]{"ds41_hc_prepare","ds41_hc_collapse","ds41_router_topk","ds41_sparse_attn"};
        MathFixture good(op,dtype,fault,false),bad(op,dtype,fault,true);bad.check_finite_witness();
        const std::string label=std::string(names[size_t(op)])+" "+rad_dtype_name(dtype)+" fault"+std::to_string(int(fault));
        const auto valid=capture(label+" valid",names[size_t(op)],good),invalid=capture(label+" late derived overflow",names[size_t(op)],bad);
        require(valid.expected_status==RAD_OK&&invalid.expected_status==RAD_E_INVAL,"math derived-fault CPU oracle did not refuse");
        Harness fixture(registry,valid,stream,dtype==RAD_BF16);fixture.pass(valid,0);fixture.pass(invalid,1,true);fixture.pass(valid,2);++math_fault_cycles;
    };
    for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})derived(math::Op::hc_prepare,dtype,MathFault::prepare);
    for(uint32_t dtype:{RAD_F32,RAD_BF16})derived(math::Op::hc_collapse,dtype,MathFault::collapse_arithmetic);
    for(uint32_t dtype:{RAD_F16,RAD_BF16})derived(math::Op::hc_collapse,dtype,MathFault::collapse_cast);
    derived(math::Op::router_topk,RAD_F32,MathFault::router_quotient);derived(math::Op::router_topk,RAD_F32,MathFault::router_scale);
    for(uint32_t dtype:{RAD_F32,RAD_BF16})derived(math::Op::sparse_attn,dtype,MathFault::sparse_numerator);
    {
        route::testing::Fixture good;
        good.buffers[route::weights].weight(0,-0.f);
        good.buffers[route::weights].weight(1,std::numeric_limits<float>::denorm_min());
        good.buffers[route::weights].weight(5,std::numeric_limits<float>::max());
        auto bad=good;bad.buffers[route::ids].id(5,bad.buffers[route::ids].id(4));
        const auto okay=capture("route jobs valid","ds41_route_jobs",good),invalid=capture("route jobs duplicate","ds41_route_jobs",bad);
        // Literal CPU mapping/count oracle is independent of the materializer primitives.
        const std::array<int32_t,6> ids{0,1,3,4,1,2},tokens{0,0,1,1,2,2},identity{0,1,2,3,4,5};const std::array<int32_t,5> counts{1,2,1,1,1};
        require(okay.tensors[route::expert_ids].expected==std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(ids.data()),reinterpret_cast<const uint8_t*>(ids.data())+sizeof ids),"literal expert-ID oracle differs");
        require(okay.tensors[route::token_ids].expected==std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(tokens.data()),reinterpret_cast<const uint8_t*>(tokens.data())+sizeof tokens),"literal token-row oracle differs");
        require(okay.tensors[route::identity_rows].expected==std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(identity.data()),reinterpret_cast<const uint8_t*>(identity.data())+sizeof identity),"literal identity-row oracle differs");
        require(okay.tensors[route::expert_counts].expected==std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(counts.data()),reinterpret_cast<const uint8_t*>(counts.data())+sizeof counts),"literal histogram oracle differs");
        Harness fixture(registry,okay,stream,false);fixture.pass(okay,0);fixture.pass(invalid,1);fixture.pass(okay,2);
        route::testing::Fixture outside;outside.buffers[route::ids].id(5,5);fixture.pass(capture("route jobs bad ID","ds41_route_jobs",outside),3);fixture.pass(okay,4);
    }
    for(const auto types:{std::array<uint32_t,2>{RAD_F32,RAD_F32},{RAD_BF16,RAD_F32},{RAD_F32,RAD_BF16}}){
        transform::testing::Fixture good(transform::Op::rmsnorm,types[0],types[1]),bad=good;
        bad.buffers[0].set(bad.buffers[0].count()-1,NAN);
        exercise(registry,stream,capture("RMS valid","ds41_rmsnorm",good),capture("RMS NaN input","ds41_rmsnorm",bad),types[0]==RAD_BF16);
    }
    for(uint32_t dtype:{RAD_F32,RAD_BF16}){
        transform::testing::Fixture good(transform::Op::rope_tail,dtype,dtype,0,dtype==RAD_BF16),bad=good;
        bad.buffers[1].id(2,int32_t(bad.table_positions));
        exercise(registry,stream,capture("RoPE valid","ds41_rope_tail",good),capture("RoPE position outside table","ds41_rope_tail",bad));
    }
    for(uint32_t dtype:{RAD_F32,RAD_BF16}){
        indexer::testing::Fixture good(indexer::Op::scores,dtype,0,3,11,4,true,8),bad=good;bad.buffer(3).id(2,3);
        exercise(registry,stream,capture("dense index valid","ds41_index_scores",good),capture("dense index invalid sequence","ds41_index_scores",bad));
        for(int ratio:{1,2}){
            paged::testing::Fixture pages(dtype,0,ratio),broken=pages;broken.buffer(5).id(2*broken.blocks_per_sequence,broken.physical_blocks);
            const auto dense_reference=capture("dense canonical index","ds41_index_scores",good),paged_reference=capture("paged index valid","ds41_index_scores_paged",pages);
            require(dense_reference.tensors[5].expected==paged_reference.tensors[7].expected,"permuted paged index differs from canonical dense CPU oracle");
            exercise(registry,stream,paged_reference,capture("paged index invalid visible page","ds41_index_scores_paged",broken));
        }
    }
    for(auto kind:{indexer::Op::candidates,indexer::Op::topk})for(bool mask:{false,true}){
        if(kind==indexer::Op::candidates&&!mask)continue;
        indexer::testing::Fixture good(kind,RAD_F32,0,3,11,4,mask),bad=good;
        bad.visible().id(2,int32_t(bad.keys+1));
        const char* op=kind==indexer::Op::candidates?"ds41_candidate_blocks":"ds41_index_topk";
        exercise(registry,stream,capture("index selection valid",op,good),capture("index selection invalid visible length",op,bad));
    }
    for(uint32_t dtype:{RAD_F32,RAD_BF16})for(int ratio:{0,1,2}){
        attention::testing::Fixture good(dtype,0,ratio),bad=good;
        bad.buffer(attention::window_table).id(2*bad.buffer(attention::window_table).t.shape[1],-1);
        exercise(registry,stream,capture("paged attention valid","ds41_sparse_attn_paged",good),
                 capture("paged attention invalid visible window page","ds41_sparse_attn_paged",bad),dtype==RAD_BF16);
    }
    namespace finish=rad::ds41::compress_finish;
    for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(int ratio:{1,2}){
        finish::testing::Fixture good(RAD_F32,dtype,128,ratio),bad=good;
        bad.buffers[finish::gain].set(127,NAN);
        exercise(registry,stream,capture("compress finish valid","ds41_compress_finish",good),
                 capture("compress finish invalid gain","ds41_compress_finish",bad));
    }
    namespace emitted=rad::ds41::latent_emit;
    for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(int ratio:{1,2})for(uint32_t pd:{RAD_I32,RAD_I64}){
        emitted::testing::Fixture good(dtype,128,ratio,0,pd,pd==RAD_I32?RAD_I64:RAD_I32),bad=good;
        bad.buffers[emitted::emit].integer(bad.rows-1,2);
        exercise(registry,stream,capture("emitted store valid","ds41_latent_store_emitted",good),
                 capture("emitted store invalid mask","ds41_latent_store_emitted",bad));
    }
    namespace engram=rad::ds41::engram_gate;
    for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(bool mask:{false,true})for(bool gates:{false,true}){
        engram::testing::Fixture good(dtype,0,mask,gates),bad=good;
        bad.buffers[engram::input].set(bad.buffers[engram::input].count()-1,NAN);
        exercise(registry,stream,capture("Engram gate valid","ds41_engram_gate",good),
                 capture("Engram gate invalid input","ds41_engram_gate",bad),dtype==RAD_BF16);
    }
    namespace qdq=rad::ds41::qdq;namespace qm=rad::ds41::qdq_math;
    for(auto profile:{qm::Profile::window_fp8,qm::Profile::compressed_fp4,qm::Profile::index_fp4})
        for(bool mask:{false,true})for(int width:{32,128,512}){
            qdq::testing::Fixture good(profile,width,0,mask),bad=good;
            bad.buffers[qdq::input].raw(5*width+width-1,0x7FC1);
            exercise(registry,stream,capture("QDQ candidate valid","ds41_qdq_candidate",good),
                     capture("QDQ candidate late NaN","ds41_qdq_candidate",bad));
        }
    namespace st=rad::ds41::index_stream;
    for(auto mode:{st::Mode::source,st::Mode::consumer,st::Mode::plain})
        for(uint32_t dtype:{RAD_F32,RAD_BF16})for(int ratio:{1,2}) {
            st::testing::Fixture good(mode,dtype,0,ratio),bad=good;
            bad.buffer(st::query).set(bad.buffer(st::query).count()-1,NAN);
            exercise(registry,stream,capture("stream index valid","ds41_index_stream",good),
                     capture("stream index nonfinite query","ds41_index_stream",bad));
        }

}
void usage(){std::puts("ds41-aql-ops-check --run-gpu --home ISOLATED_BUILD_HOME [--host-kargs]\nTiny actual Builder/Ctx/AQL op/status check; exactly one visible gfx1201; no model, Pools or service.");}
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
        require(math_fault_cycles==11&&bad_passes==93&&good_passes==185&&published_passes==good_passes,"checker pass/publication count drifted");
        require(exact_qdq_outputs==56448&&qdq_subnormal_proofs==24,"actual QDQ exact-output/underflow proof count drifted");
        std::printf("PASS: actual registered rows, shaped bindings and core AQL deferred status; %zu good/%zu refused passes, %zu publications, %zu numeric outputs; peak charged fixture bytes=%lld.\n",
                    good_passes,bad_passes,published_passes,numeric_outputs,(long long)peak_explicit);
        std::printf("QDQ mathematical candidate: %zu exact BF16 outputs, %zu signed index2^-127 proofs; not official TileLang parity.\n",exact_qdq_outputs,qdq_subnormal_proofs);
        std::printf("Math derived faults: %zu good-bad-good cycles on existing four rows; every failed pass preserves all prior output bytes and resets on recovery.\n",math_fault_cycles);
        std::puts("Byte cap includes padded Ctx arena/scratch and requested weight/mapped staging extents; excludes driver/page metadata, core stream argument/bounce rings and fixed Ctx validation resources. No serving/full graph/throughput claim.");
        return 0;
    }catch(const std::exception& error){active=nullptr;std::fprintf(stderr,"FAIL: %s\n",error.what());return 1;}
}
