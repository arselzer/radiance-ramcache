/* Actual host-reference plugin/Builder/Ctx bindings. No launch replacement,
 * GPU work, model pools or service state. CPU component tests own math oracles. */
#include "rad_test.h"
#include "build/rad_build.h"
#include "runtime/ctx.h"
#include "device/device.h"
#include "../libds41/ds41_host_ops.h"
#include "../libds41/ds41_math_emulate.h"
#include "../libds41/ds41_compress_fixture.h"
#include "../libds41/ds41_ffn_fixture.h"
#include "../libds41/ds41_route_jobs_fixture.h"
#include "../libds41/ds41_index_stream_fixture.h"
#include "../libds41/ds41_transform_fixture.h"
#include "../libds41/ds41_index_paged_fixture.h"
#include "../libds41/ds41_attention_paged_fixture.h"
#include "../libds41/ds41_compress_finish_fixture.h"
#include "../libds41/ds41_latent_emit_fixture.h"
#include "../libds41/ds41_engram_gate_fixture.h"
#include "../libds41/ds41_qdq_fixture.h"
#include <limits>
#include <memory>

using namespace rad;
namespace math=rad::ds41::math;
namespace compress=rad::ds41::compress;
namespace ffn=rad::ds41::ffn;
namespace route=rad::ds41::route_jobs;
namespace transform=rad::ds41::transform;
namespace indexer=rad::ds41::index;
namespace paged=rad::ds41::paged_index;
namespace attention=rad::ds41::attention_paged;
namespace {
constexpr int64_t guard=256,cap=1<<20;
constexpr uint8_t poison=0xbd;
void need(bool okay,const std::string& message){if(!okay)throw std::runtime_error(message);}
void okay(int status,const char* operation){need(status==RAD_OK,std::string(operation)+": "+rad_strerror(status));}
Registry& registry(){
    static Registry value;static const bool loaded=[&]{
        okay(value.load_plugin(home()+"/quantizers/libquant.so",0),"load real libquant");
        okay(value.load_plugin(home()+"/experimental/kernels/libds41_ref.so",0),"load real libds41_ref");return true;
    }();(void)loaded;return value;
}
struct Tensor {RadTensor shape{};std::vector<uint8_t> before,after;bool exists()const{return shape.rank!=0;}};
struct Case {std::string op;int64_t rows=0;std::vector<RadParam> params;std::vector<Tensor> tensors;int status=RAD_OK;};
int64_t count(const RadTensor& t){
    need(t.rank>0&&t.rank<=RAD_MAX_RANK,"bad fixture rank");int64_t n=1,stride=1;
    for(int d=int(t.rank)-1;d>=0;--d){need(t.shape[d]>0&&t.shape[d]<=65536&&t.stride[d]==stride,"host binding fixture must be packed and tiny");
        need(n<=65536/t.shape[d],"host binding fixture is too large");n*=t.shape[d];stride*=t.shape[d];}
    return n;
}
std::vector<uint8_t> copy(const RadTensor& t){
    const auto bytes=rad_dtype_bytes(t.dtype,count(t));need(t.data&&bytes>0&&bytes<=65536,"bad host fixture extent");
    const auto* begin=static_cast<const uint8_t*>(t.data);return {begin,begin+bytes};
}
template<class Fixture>Case capture(const char* op,int64_t rows,Fixture fixture){
    const auto before=fixture.args();Case out;out.op=op;out.rows=rows;out.params.assign(before.p,before.p+before.n_p);out.tensors.resize(size_t(before.n_t));
    for(int i=0;i<before.n_t;++i)if(before.t[i].rank){auto& t=out.tensors[size_t(i)];t.shape=before.t[i];t.shape.data=nullptr;t.before=copy(before.t[i]);}
    out.status=fixture.run();const auto after=fixture.args();
    for(int i=0;i<after.n_t;++i)if(after.t[i].rank)out.tensors[size_t(i)].after=copy(after.t[i]);
    return out;
}
struct Binding {
    bool present=false,weight=false,output=false;uint32_t handle=0;int64_t offset_elements=0,offset_bytes=0,bytes=0,logical=0;
    void* base=nullptr;std::vector<uint32_t> weight_words;
};
struct Harness;
thread_local Harness* active=nullptr;
void step(RadCtx*,const RadBatch*);
struct Harness {
    std::unique_ptr<Builder> builder;rad_op op=0;int64_t rows;std::string name;
    std::vector<Binding> bindings;std::vector<RadOperand> operands;std::vector<RadLaunchFn> original_launches;int published=0;
    Ctx ctx; // Drains before the real weight owners, Builder or Registry disappear.
    explicit Harness(const Case& good):rows(good.rows),name(good.op){
        need(device_is_host()&&good.status==RAD_OK,"host Harness needs host backend and good fixture");
        auto& reg=registry();const auto* schema=reg.schema(name);need(schema&&schema->n_operands==int(good.tensors.size()),"real host schema missing");
        RadModelMeta meta{};meta.arch_id="ds41-host-op-fixture";meta.quant="native";meta.name="tiny-host-only";
        RadBuildCtx bc{};bc.rank=0;bc.world_size=1;bc.max_tok=rad_param_geti(good.params.data(),int(good.params.size()),"M",rows);bc.max_seqs=8;bc.max_ctx=64;bc.scope="";
        builder=std::make_unique<Builder>(reg,meta,bc);builder->set_instances(false);bindings.resize(good.tensors.size());operands.resize(good.tensors.size());
        std::vector<rad_weight> weights;int64_t explicit_weight_bytes=0;
        for(size_t i=0;i<bindings.size();++i){auto& b=bindings[i];const auto& source=good.tensors[i];const int role=schema->operands[i].role;
            b.output=role==RAD_OPD_OUT||role==RAD_OPD_INOUT;
            if(!source.exists()){need(schema->operands[i].optional,"required host operand absent");operands[i]=RAD_NONE;continue;}
            b.present=true;b.weight=role==RAD_OPD_WEIGHT;need(role!=RAD_OPD_WTAB,"unexpected host table operand");b.logical=int64_t(source.before.size());
            if(b.weight){
                RadWeightDecl decl{};decl.dtype=source.shape.dtype;decl.rank=source.shape.rank;decl.access=RAD_ACCESS_PER_TOKEN;decl.group={-1,-1,0};
                for(unsigned d=0;d<decl.rank;++d)decl.shape[d]=source.shape.shape[d];
                b.handle=builder->decl_weight(("host.weight."+std::to_string(i)).c_str(),&decl);need(b.handle!=0,"Builder refused real host weight");weights.push_back(b.handle);
                b.offset_bytes=guard;b.bytes=b.logical+2*guard;b.weight_words.resize(size_t((b.bytes+3)/4));b.base=b.weight_words.data();explicit_weight_bytes+=int64_t(b.weight_words.size()*4);
                operands[i]=RAD_W(b.handle);
            }else{
                const int width=rad_dtype_bits(source.shape.dtype)/8;need(width>0,"opaque host activation refused");
                const auto row_elements=count(source.shape)/source.shape.shape[0],row_bytes=row_elements*width,guard_rows=(guard+row_bytes-1)/row_bytes;
                RadBufDecl decl{};decl.dtype=source.shape.dtype;decl.rank=source.shape.rank;decl.kind=RAD_BUF_PERSIST;decl.domain=RAD_DOMAIN_HOST;
                for(unsigned d=0;d<decl.rank;++d)decl.shape[d]=source.shape.shape[d];
                decl.shape[0]+=2*guard_rows;b.offset_elements=guard_rows*row_elements;b.offset_bytes=b.offset_elements*width;b.bytes=b.logical+2*b.offset_bytes;
                b.handle=builder->decl_buffer(("host.tensor."+std::to_string(i)).c_str(),&decl);need(b.handle!=0,"Builder refused real host buffer");
                operands[i]=RAD_B_N(b.handle,b.offset_elements,source.shape.shape[0]);
            }
        }
        auto params=good.params;for(auto& p:params)if(!std::strcmp(p.key,"M"))p=RAD_RANGE("M",1,p.ival);
        op=builder->decl_op(name.c_str(),params.data(),int(params.size()),weights.data(),int(weights.size()));need(op!=0,"actual host op failed to resolve");okay(builder->finish(),"finish host Builder");
        auto& program=builder->program();
        const auto device_arena=arena_plan_bytes(program,nullptr,nullptr),host_arena=align_up(arena_host_bytes(program),RAD_ALIGN_UNIT);
        need(device_arena>=0&&host_arena>=0&&device_arena+host_arena+explicit_weight_bytes<=cap,"host fixture arena exceeds cap");
        for(const auto& band:program.ops[op].bands){const auto& resolved=band.dom[RAD_DOMAIN_HOST];
            need(resolved.row&&resolved.row->plugin=="libds41_ref"&&resolved.row->reference&&resolved.row->info&&resolved.row->info->launch&&!resolved.row->deferred_status,
                 "host reference row/marker/direct launch contract missing");original_launches.push_back(resolved.row->info->launch);
        }
        for(auto& b:bindings)if(b.present&&b.weight){auto& weight=program.weights[b.handle];need(weight.stored_bytes==b.logical&&weight.stored_dtype==weight.decl.dtype,"unexpected host weight relayout");
            weight.ptr=static_cast<uint8_t*>(b.base)+b.offset_bytes;weight.site=Site::Host;}
        CtxDesc desc{};desc.program=&program;desc.rank=0;desc.world_size=1;desc.device=0;desc.arch_step=step;desc.record_passes=false;
        okay(ctx.init(desc),"initialize actual host Ctx");okay(ctx.bind_thread(),"bind actual host rank");need(!ctx.checks_device_status(),"host rows accidentally enabled GPU collectors");
        for(auto& b:bindings)if(b.present&&!b.weight){b.base=rad_buf_ptr(&ctx,b.handle);need(b.base!=nullptr,"actual host arena tensor missing");}
    }
    int pass(const Case& test,int ordinal){
        need(test.op==name&&test.rows==rows&&test.tensors.size()==bindings.size(),"host case geometry differs");
        okay(rad_stream_sync(ctx.stream()),"drain before CPU fixture writes");
        for(size_t i=0;i<bindings.size();++i){auto& b=bindings[i];const auto& t=test.tensors[i];need(b.present==t.exists(),"host optionality differs");if(!b.present)continue;
            need(int64_t(t.before.size())==b.logical,"host fixture payload size differs");std::memset(b.base,poison,size_t(b.bytes));std::memcpy(static_cast<uint8_t*>(b.base)+b.offset_bytes,t.before.data(),size_t(b.logical));}
        rad_dev_host_wrote();const int before=published;active=this;RadBatch batch{};batch.phase=RAD_PHASE_PREFILL;batch.n_tok=rows;batch.n_seq=1;batch.step=ordinal;
        const int result=ctx.run_step(&batch);active=nullptr;need(result==test.status,name+": direct host payload result differs");
        need(ctx.await_device_status()==RAD_OK&&!ctx.checks_device_status(),"host error incorrectly depends on deferred device readback");
        if(result==RAD_OK)++published;else need(published==before,"failed host pass published state");
        okay(rad_stream_sync(ctx.stream()),"finish host fixture pass");
        for(size_t i=0;i<bindings.size();++i){const auto& b=bindings[i];if(!b.present)continue;const auto* bytes=static_cast<const uint8_t*>(b.base);const auto& t=test.tensors[i];
            need(std::all_of(bytes,bytes+b.offset_bytes,[](uint8_t x){return x==poison;})&&
                 std::all_of(bytes+b.offset_bytes+b.logical,bytes+b.bytes,[](uint8_t x){return x==poison;}),name+": host guard changed");
            const std::vector<uint8_t> actual(bytes+b.offset_bytes,bytes+b.offset_bytes+b.logical);
            const bool failure_status=name=="ds41_latent_store_emitted"&&i==rad::ds41::latent_emit::status;
            need(actual==(b.output&&(result==RAD_OK||failure_status)?t.after:t.before),name+": host output/source bytes differ");
        }
        size_t band_index=0;for(const auto& band:builder->program().ops[op].bands)
            need(band.dom[RAD_DOMAIN_HOST].row->info->launch==original_launches[band_index++],"checker/runtime replaced the legacy host launch");
        return result;
    }
};
void step(RadCtx* c,const RadBatch*){if(active)(void)rad_issue(c,active->op,active->operands.data(),int(active->operands.size()),active->rows);else rad_step_fail(c,"missing host fixture owner");}
void exercise(const Case& good,const Case& bad){
    need(good.status==RAD_OK&&bad.status<0,"host fixture status setup failed");Harness fixture(good);
    CHECK_EQ(fixture.pass(good,0),RAD_OK);CHECK_EQ(fixture.pass(bad,1),bad.status);CHECK_EQ(fixture.pass(good,2),RAD_OK);CHECK_EQ(fixture.published,2);
}
// Compact synthetic math inputs, supplied explicitly (component math tests own the
// independent official/golden equations). This fixture only proves runtime binding.
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
}
TEST(ds41_host_ops_real_plugin_registers_reference_rows_without_deferred_collectors){
    if(!device_is_host())SKIP("host backend only");
    const std::array<const char*,27> names{"ds41_gemm_packed","ds41_routed_packed","ds41_latent_store","ds41_latent_gather","ds41_hc_prepare","ds41_hc_collapse","ds41_router_topk","ds41_sparse_attn",
        "ds41_compress_pool","ds41_expert_act","ds41_moe_combine","mhc_post","ds41_index_scores","ds41_candidate_blocks","ds41_index_topk","ds41_index_scores_paged","ds41_rmsnorm","ds41_rope_tail","ds41_sparse_attn_paged","ds41_route_jobs","ds41_index_stream","ds41_embedding","ds41_reorder_heads","ds41_compress_finish","ds41_latent_store_emitted","ds41_engram_gate","ds41_qdq_candidate"};
    auto& reg=registry();CHECK_EQ(reg.rows().size(),names.size());
    for(const char* name:names){int matches=0;for(const auto& row:reg.rows())if(!std::strcmp(row.info->op,name)){
            ++matches;CHECK(row.plugin=="libds41_ref");CHECK(row.reference);CHECK_EQ(row.info->domain,RAD_DOMAIN_HOST);CHECK(row.info->launch!=nullptr);CHECK(row.deferred_status==nullptr);CHECK_EQ(row.info->launch(nullptr,nullptr),RAD_E_INVAL);}
        CHECK_EQ(matches,1);CHECK(reg.schema(name)!=nullptr);}
}
TEST(ds41_host_ops_route_jobs_real_bindings_fail_directly_and_reset_without_collectors){
    if(!device_is_host())SKIP("host backend only");
    route::testing::Fixture good;good.buffers[route::weights].weight(0,-0.f);good.buffers[route::weights].weight(1,std::numeric_limits<float>::denorm_min());
    auto bad=good;bad.buffers[route::ids].id(5,bad.buffers[route::ids].id(4));const auto valid=capture("ds41_route_jobs",3,good),duplicate=capture("ds41_route_jobs",3,bad);
    Harness fixture(valid);CHECK_EQ(fixture.pass(valid,0),RAD_OK);CHECK_EQ(fixture.pass(duplicate,1),RAD_E_INVAL);CHECK_EQ(fixture.pass(valid,2),RAD_OK);
    bad=good;bad.buffers[route::ids].id(5,5);CHECK_EQ(fixture.pass(capture("ds41_route_jobs",3,bad),3),RAD_E_SHAPE);CHECK_EQ(fixture.pass(valid,4),RAD_OK);CHECK_EQ(fixture.published,3);
}
TEST(ds41_host_ops_never_dereference_or_validate_device_arena_caller_scratch){
    if(!device_is_host())SKIP("host backend only");
    RadLaunchFn launch=nullptr;for(const auto& row:registry().rows())if(!std::strcmp(row.info->op,"ds41_route_jobs"))launch=row.info->launch;
    REQUIRE(launch!=nullptr);
    route::testing::Fixture good;const auto flag=good.flag;auto a=good.args();a.scratch=reinterpret_cast<void*>(uintptr_t(1));a.scratch_bytes=INT64_MIN;
    CHECK_EQ(launch(&a,nullptr),RAD_OK);CHECK(good.flag==flag);CHECK_EQ(good.buffers[route::identity_rows].id(5),5);
    route::testing::Fixture bad;bad.buffers[route::ids].id(5,5);a=bad.args();a.scratch=reinterpret_cast<void*>(UINTPTR_MAX);a.scratch_bytes=INT64_MAX;
    const auto before=bad.buffers;CHECK_EQ(launch(&a,nullptr),RAD_E_SHAPE);CHECK(bad.flag==flag);
    for(size_t i=0;i<before.size();++i)CHECK(bad.buffers[i].words==before[i].words);
}
TEST(ds41_host_ops_math_real_bindings_keep_shifted_mhc_routing_and_sparse_contracts){
    if(!device_is_host())SKIP("host backend only");
    const char* names[]{"ds41_hc_prepare","ds41_hc_collapse","ds41_router_topk","ds41_sparse_attn"};
    for(int op=0;op<4;++op){MathFixture good{math::Op(op)};auto bad=good;bad.buffers[0].weight(bad.buffers[0].count()-1,NAN);exercise(capture(names[op],2,good),capture(names[op],2,bad));}
}
TEST(ds41_host_ops_compressor_real_host_state_bindings_are_transactional){
    if(!device_is_host())SKIP("host backend only");
    for(int ratio:{1,2}){compress::testing::Fixture good(5,ratio),bad=good;bad.buffer(compress::cu).seti(bad.sequences,int32_t(bad.tokens+1));
        exercise(capture("ds41_compress_pool",good.tokens,good),capture("ds41_compress_pool",bad.tokens,bad));}
}
TEST(ds41_host_ops_compressor_finish_preserves_original_row_mask_and_cast_boundaries){
    if(!device_is_host())SKIP("host backend only");
    namespace c=rad::ds41::compress_finish;
    for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(int ratio:{1,2}){
        c::testing::Fixture good(RAD_F32,dtype,128,ratio),bad=good;
        bad.buffers[c::gain].set(127,NAN);
        exercise(capture("ds41_compress_finish",good.rows,good),capture("ds41_compress_finish",bad.rows,bad));
    }
}
TEST(ds41_host_ops_emitted_store_uses_core_coordinate_widths_and_seals_failure_status){
    if(!device_is_host())SKIP("host backend only");
    namespace e=rad::ds41::latent_emit;
    for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(int ratio:{1,2})for(uint32_t pd:{RAD_I32,RAD_I64}){
        e::testing::Fixture good(dtype,128,ratio,0,pd,pd==RAD_I32?RAD_I64:RAD_I32),bad=good;
        bad.buffers[e::emit].integer(bad.rows-1,2);
        exercise(capture("ds41_latent_store_emitted",good.rows,good),capture("ds41_latent_store_emitted",bad.rows,bad));
    }
}
TEST(ds41_host_ops_engram_gate_binds_both_real_weights_and_fixed_optional_slots){
    if(!device_is_host())SKIP("host backend only");
    namespace e=rad::ds41::engram_gate;
    for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(bool mask:{false,true})for(bool gates:{false,true}){
        e::testing::Fixture good(dtype,0,mask,gates),bad=good;
        bad.buffers[e::input].set(bad.buffers[e::input].count()-1,NAN);
        exercise(capture("ds41_engram_gate",good.tokens,good),capture("ds41_engram_gate",bad.tokens,bad));
    }
}
TEST(ds41_host_ops_qdq_candidate_preserves_bits_masks_and_private_direct_failure){
    if(!device_is_host())SKIP("host backend only");
    namespace q=rad::ds41::qdq;namespace m=rad::ds41::qdq_math;
    for(auto profile:{m::Profile::window_fp8,m::Profile::compressed_fp4,m::Profile::index_fp4})
        for(bool mask:{false,true})for(int width:{32,128,512}){
            q::testing::Fixture good(profile,width,0,mask),bad=good;
            bad.buffers[q::input].raw(5*width+width-1,0x7FC1);
            exercise(capture("ds41_qdq_candidate",good.rows,good),capture("ds41_qdq_candidate",bad.rows,bad));
        }
}
TEST(ds41_host_ops_ffn_real_bindings_return_payload_errors_before_publication){
    if(!device_is_host())SKIP("host backend only");
    const char* names[]{"ds41_expert_act","ds41_moe_combine","mhc_post"};
    for(int op=0;op<3;++op){ffn::testing::Fixture good{ffn::Op(op)};auto bad=good;bad.buffers[0].setf(bad.buffers[0].count()-1,NAN);const int rows=op==2?2:3;
        exercise(capture(names[op],rows,good),capture(names[op],rows,bad));}
}
TEST(ds41_host_ops_rmsnorm_real_weight_and_storage_bindings){
    if(!device_is_host())SKIP("host backend only");
    for(const auto types:{std::array<uint32_t,2>{RAD_F32,RAD_F32},{RAD_BF16,RAD_F32},{RAD_F32,RAD_BF16}}){
        transform::testing::Fixture good(transform::Op::rmsnorm,types[0],types[1]),bad=good;bad.buffers[0].set(bad.buffers[0].count()-1,NAN);
        exercise(capture("ds41_rmsnorm",3,good),capture("ds41_rmsnorm",3,bad));}
}
TEST(ds41_host_ops_rope_real_rank_three_tables_and_inverse_bindings){
    if(!device_is_host())SKIP("host backend only");
    for(uint32_t dtype:{RAD_F32,RAD_BF16})for(bool inverse:{false,true}){transform::testing::Fixture good(transform::Op::rope_tail,dtype,dtype,0,inverse),bad=good;
        bad.buffers[1].id(2,int32_t(bad.table_positions));exercise(capture("ds41_rope_tail",3,good),capture("ds41_rope_tail",3,bad));}
}
TEST(ds41_host_ops_dense_and_paged_index_real_rank_three_and_four_bindings){
    if(!device_is_host())SKIP("host backend only");
    for(uint32_t dtype:{RAD_F32,RAD_BF16}){
        indexer::testing::Fixture good(indexer::Op::scores,dtype,0,3,11,4,true,8),bad=good;bad.buffer(3).id(2,3);
        exercise(capture("ds41_index_scores",3,good),capture("ds41_index_scores",3,bad));
        for(int ratio:{1,2}){paged::testing::Fixture pages(dtype,0,ratio),broken=pages;broken.buffer(5).id(2*broken.blocks_per_sequence,broken.physical_blocks);
            exercise(capture("ds41_index_scores_paged",3,pages),capture("ds41_index_scores_paged",3,broken));}
    }
}
TEST(ds41_host_ops_candidate_and_topk_real_optional_mask_bindings){
    if(!device_is_host())SKIP("host backend only");
    for(auto op:{indexer::Op::candidates,indexer::Op::topk})for(bool mask:{false,true}){if(op==indexer::Op::candidates&&!mask)continue;
        indexer::testing::Fixture good(op,RAD_F32,0,3,11,4,mask),bad=good;bad.visible().id(2,int32_t(bad.keys+1));const char* name=op==indexer::Op::candidates?"ds41_candidate_blocks":"ds41_index_topk";
        exercise(capture(name,3,good),capture(name,3,bad));}
}
TEST(ds41_host_ops_paged_attention_real_optional_cache_and_sink_weight_bindings){
    if(!device_is_host())SKIP("host backend only");
    for(uint32_t dtype:{RAD_F32,RAD_BF16})for(int ratio:{0,1,2}){attention::testing::Fixture good(dtype,0,ratio),bad=good;
        bad.buffer(attention::window_table).id(2*bad.buffer(attention::window_table).t.shape[1],-1);
        exercise(capture("ds41_sparse_attn_paged",good.rows,good),capture("ds41_sparse_attn_paged",bad.rows,bad));}
}
TEST(ds41_host_ops_compact_stream_real_host_bindings_preserve_both_selection_outputs){
    if(!device_is_host())SKIP("host backend only");
    namespace st=rad::ds41::index_stream;
    for(auto mode:{st::Mode::source,st::Mode::consumer,st::Mode::plain}) {
        st::testing::Fixture good(mode),bad=good;bad.buffer(st::query).set(bad.buffer(st::query).count()-1,NAN);
        exercise(capture("ds41_index_stream",good.rows,good),capture("ds41_index_stream",bad.rows,bad));
    }
}
RAD_TEST_MAIN()
