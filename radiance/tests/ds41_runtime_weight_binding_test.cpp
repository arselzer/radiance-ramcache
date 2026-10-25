/* Actual Ctx weight/table binding for the real DS41 metadata rows. Host backend only:
 * tiny synthetic native payloads, no GPU kernels, model pools or service state. */
#include "quant_fixture.h"
#include "build/rad_build.h"
#include "runtime/ctx.h"
#include "runtime/residency.h"
#include "../libds41/ds41_packed_args.h"
#include <functional>

using namespace rad;
namespace pk=rad::ds41::packed;
namespace {
struct Capture {int status=RAD_E_STATE;std::vector<RadTensor> tensors;};
Capture* active_capture=nullptr;
bool active_routed=false;
std::function<void(Ctx&)> step_body;
int capture_launch(const RadArgs* a,RadStream) {
    if(!active_capture)return RAD_E_STATE;
    active_capture->tensors.assign(a->t,a->t+a->n_t);
    pk::Launch launch;
    active_capture->status=pk::validate(a,active_routed,&launch);
    return active_capture->status;
}
void step(RadCtx* c,const RadBatch*) {if(step_body)step_body(*static_cast<Ctx*>(c));}
struct ResidencyFixture final:ResidencyTable {
    std::vector<WeightSlot> slots;std::vector<rad_weight> changed;uint64_t ep=1;
    const WeightSlot* lookup(rad_weight w)const override{return w<slots.size()?&slots[w]:nullptr;}
    uint64_t epoch()const override{return ep;}
    const std::vector<rad_weight>& changes()const override{return changed;}
    void clear_changes()override{changed.clear();}
    void move(rad_weight w,void* p){slots[w].ptr=p;++slots[w].generation;++ep;changed.push_back(w);}
};
struct Fixture {
    Registry registry;
    std::unique_ptr<Builder> builder;
    std::vector<rad_weight> weights;
    std::vector<std::vector<uint8_t>> payload;
    ResidencyFixture residency;
    RadKernelInfo capture_info{};KernelRow capture_row;
    rad_op op=0;bool routed=false;
    Capture captured;Ctx ctx;
    explicit Fixture(const char* format,bool table,int rank=0):routed(table) {
        if(!device_is_host())throw std::runtime_error("this regression requires the host backend");
        if(registry.load_plugin(home()+"/testkernels/ds41meta/radtest_ds41meta.so",0)!=RAD_OK)
            throw std::runtime_error("missing actual libds41 device metadata fixture");
        const auto* importer=qfix::reg().checkpoint_importer("gguf",format);
        if(!importer)throw std::runtime_error("missing actual GGUF importer");
        WeightSource src{};src.rank=2;src.shape[0]=2;src.shape[1]=256;
        const RadCheckpointImportWeight iw{"runtime-weight-binding",format,2,256};
        if(importer->info->encoding(&iw,&src.enc)!=RAD_OK)throw std::runtime_error("encoding metadata failed");
        RadModelMeta meta{};meta.arch_id="runtime-binding-test";meta.quant="native";meta.name="no-model";
        RadBuildCtx bc{};bc.rank=rank;bc.world_size=2;bc.max_tok=2;bc.max_seqs=1;bc.max_ctx=2;bc.scope="";
        builder=std::make_unique<Builder>(registry,meta,bc);builder->set_instances(false);
        builder->set_sources([src](const std::string&,const Program&,WeightSource* out){*out=src;return RAD_OK;});
        for(int e=0;e<(table?2:1);++e) {
            RadWeightDecl d{};d.dtype=src.enc.plane[0].dtype;d.rank=2;d.shape[0]=2;d.shape[1]=256;
            d.access=RAD_ACCESS_CONDITIONAL;d.group.layer=0;d.group.expert=e;
            const auto w=builder->decl_weight(("weight."+std::to_string(e)).c_str(),&d);
            if(!w)throw std::runtime_error("actual Builder refused weight");
            weights.push_back(w);
        }
        RadParam params[]{RAD_RANGE("M",1,2),RAD_INT("N",2),RAD_INT("K",256),
            RAD_STR("format",format),RAD_STR("a_dtype","f32"),RAD_STR("y_dtype","f32"),
            RAD_INT("R",2),RAD_INT("n_expert",2)};
        op=builder->decl_op(table?"ds41_routed_packed":"ds41_gemm_packed",params,table?8:6,weights.data(),int(weights.size()));
        if(!op||builder->finish()!=RAD_OK)throw std::runtime_error("actual DS41 layout declaration failed");
        auto& program=builder->program();
        for(auto w:weights) {
            const auto n=program.weights[w].stored_bytes;
            if(n<=0||n>4096)throw std::runtime_error("unexpected native fixture extent");
            payload.emplace_back(size_t(n),0);program.weights[w].ptr=payload.back().data();
        }
        residency.slots.resize(program.weights.size());
        for(auto w:weights)residency.slots[w].ptr=program.weights[w].ptr;
        auto& resolved=program.ops[op].bands[0].dom[RAD_DOMAIN_DEVICE];
        if(!resolved.row||!resolved.row->info)throw std::runtime_error("actual device metadata did not resolve");
        // Preserve the real DS41 layout/unrelayout/opd-shape metadata. Only replace
        // the never-launch stub, so the kernel observes what Ctx actually binds.
        capture_info=*resolved.row->info;capture_info.launch=capture_launch;
        capture_row=*resolved.row;capture_row.info=&capture_info;resolved.row=&capture_row;
        CtxDesc cd{};cd.program=&program;cd.rank=rank;cd.world_size=2;cd.arch_step=step;
        cd.residency=&residency;cd.record_passes=false;
        if(ctx.init(cd)!=RAD_OK)throw std::runtime_error("actual Ctx bind failed");
    }
    int run(int ordinal=0) {
        std::vector<float> a(2*256,1),y(2*2,0);int32_t experts[]{1,0},rows[]{0,1},status[]{-1,-1};
        auto vector=[](void* ptr) {RadOperand o=RAD_P_T(ptr,RAD_I32);o.rows=2;return o;};
        std::vector<RadOperand> operands{RAD_P_T2(a.data(),RAD_F32,2,256),routed?RAD_WTAB(weights[0],2):RAD_W(weights[0])};
        if(routed){operands.push_back(vector(experts));operands.push_back(vector(rows));}
        operands.push_back(RAD_P_T2(y.data(),RAD_F32,2,2));if(routed)operands.push_back(vector(status));
        active_capture=&captured;active_routed=routed;
        step_body=[&](Ctx& c){(void)c.issue(op,operands.data(),int(operands.size()),2);};
        RadBatch batch{};batch.phase=RAD_PHASE_PREFILL;batch.n_tok=2;batch.step=ordinal;
        const int rc=ctx.run_step(&batch);const int sync=rad_stream_sync(ctx.stream());
        step_body={};active_capture=nullptr;return rc<0?rc:sync;
    }
};
}
TEST(ds41_runtime_dense_binding_preserves_logical_opaque_descriptor) {
    if(!device_is_host())SKIP("CPU-only runtime binding fixture");
    for(const char* format:{"iq2_xxs","iq2_xs","iq2_s","iq3_xxs","q5_K","q8_0"}) {
        Fixture f(format,false);REQUIRE_EQ(f.run(),RAD_OK);REQUIRE_EQ(f.captured.status,RAD_OK);
        REQUIRE_EQ(f.captured.tensors.size(),3u);const auto& t=f.captured.tensors[1];
        const auto& w=f.builder->program().weights[f.weights[0]];
        CHECK_EQ(t.dtype,w.stored_dtype);CHECK(t.dtype>=RAD_DT_PLUGIN_BASE);CHECK_EQ(rad_dtype_bits(t.dtype),0);
        CHECK_EQ(t.rank,2u);CHECK_EQ(t.shape[0],2);CHECK_EQ(t.shape[1],256);
        CHECK_EQ(t.stride[0],256);CHECK_EQ(t.stride[1],1);CHECK(t.data==f.payload[0].data());
        CHECK(!w.layout_tag.empty());CHECK_EQ(w.stored_bytes,int64_t(f.payload[0].size()));
    }
}
TEST(ds41_runtime_routed_binding_tracks_rank_local_expert_pointers_and_moves) {
    if(!device_is_host())SKIP("CPU-only runtime binding fixture");
    Fixture a("iq2_xs",true,0),b("iq2_xs",true,1);
    REQUIRE_EQ(a.run(),RAD_OK);REQUIRE_EQ(b.run(),RAD_OK);
    for(auto* f:{&a,&b}) {
        REQUIRE_EQ(f->captured.status,RAD_OK);REQUIRE_EQ(f->captured.tensors.size(),6u);
        const auto& t=f->captured.tensors[1];CHECK(t.dtype>=RAD_DT_PLUGIN_BASE);CHECK_EQ(rad_dtype_bits(t.dtype),0);
        CHECK_EQ(t.rank,3u);CHECK_EQ(t.shape[0],2);CHECK_EQ(t.shape[1],2);CHECK_EQ(t.shape[2],256);
        CHECK_EQ(t.stride[0],0);CHECK_EQ(t.stride[1],256);CHECK_EQ(t.stride[2],1);
        const auto* pointers=static_cast<void* const*>(t.data);
        CHECK(pointers[0]==f->payload[0].data());CHECK(pointers[1]==f->payload[1].data());
    }
    const auto old_b=static_cast<void* const*>(b.captured.tensors[1].data)[1];
    std::vector<uint8_t> moved(a.payload[1].size(),0);
    a.residency.move(a.weights[1],moved.data());REQUIRE_EQ(a.run(1),RAD_OK);
    const auto* pa=static_cast<void* const*>(a.captured.tensors[1].data);
    CHECK(pa[0]==a.payload[0].data());CHECK(pa[1]==moved.data());
    CHECK(static_cast<void* const*>(b.captured.tensors[1].data)[1]==old_b);
}
RAD_TEST_MAIN()
