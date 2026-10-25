/* Real Registry/Builder integration for native GGUF layout hooks. No payloads, pool allocation,
 * device kernels or inference: this tests the metadata the loader and planner will consume. */
#include "quant_fixture.h"
#include "build/rad_build.h"
#include "../arch/common/rad_gguf_relayout.h"
#include <map>

using namespace rad;
namespace gg = rad::gguf;
namespace {
constexpr gg::Format formats[]{gg::Format::iq2_xxs,gg::Format::iq2_xs,gg::Format::iq2_s,
                              gg::Format::iq3_xxs,gg::Format::q5_k,gg::Format::q8_0};
RadModelMeta meta() {
    RadModelMeta m{};m.arch_id="repack_fixture";m.name="metadata-only";m.quant="gguf";
    m.n_layers=4;m.n_embd=5120;m.n_ff=2304;return m;
}
RadBuildCtx context(int rank=0,int world=1) {
    RadBuildCtx c{};c.rank=rank;c.world_size=world;c.max_tok=16;c.max_seqs=1;c.max_ctx=256;c.scope="";
    return c;
}
int load(Registry& reg) {
    return reg.load_dir(home()+"/testkernels/repack",{"radtest_repackkernel"});
}
WeightSource source(gg::Format format,int64_t rows,int64_t cols) {
    WeightSource result{};const auto* g=gg::geometry(format);
    const auto* importer=qfix::reg().checkpoint_importer("gguf",g->name);
    if(!importer)throw std::runtime_error(std::string("missing GGUF importer ")+g->name);
    RadCheckpointImportWeight weight{"builder-layout-fixture",g->name,rows,cols};
    if(importer->info->encoding(&weight,&result.enc)!=RAD_OK)
        throw std::runtime_error("cannot describe fixture encoding");
    result.rank=2;result.shape[0]=rows;result.shape[1]=cols;return result;
}
using Sources=std::map<std::string,WeightSource>;
void sources(Builder& b,const Sources& inputs) {
    b.set_sources([inputs](const std::string& name,const Program&,WeightSource* out) {
        const auto found=inputs.find(name);if(found==inputs.end())return RAD_E_NOTFOUND;
        *out=found->second;return RAD_OK;
    });
    b.set_instances(false);
}
rad_weight weight(Builder& b,const char* name,const WeightSource& src,int64_t rows,int64_t cols,
                  int shard,RadWeightGroup group) {
    RadWeightDecl d{};d.dtype=src.enc.plane[0].dtype;d.rank=2;d.shape[0]=rows;d.shape[1]=cols;
    d.access=RAD_ACCESS_PER_TOKEN;d.shard=shard;d.group=group;
    // A null planes list selects every canonical plane, in encoding order.
    return b.decl_weight(name,&d);
}
void verify(Builder& b,rad_weight handle,gg::Format format,int64_t rows,int64_t cols) {
    const auto* w=b.program().weight(handle);REQUIRE(w);
    const auto* g=gg::geometry(format);const int64_t rb=cols/g->values*g->bytes;
    CHECK(w->enc_known);CHECK(!w->identity);CHECK(!w->widen);
    CHECK_EQ(w->n_sel,w->enc.n_planes);CHECK_EQ(w->layout_tag,std::string(g->tag));
    CHECK_EQ(w->stored_dtype,uint32_t(RAD_U8));CHECK_EQ(w->stored_rank,2u);
    CHECK_EQ(w->stored_shape[0],rows);CHECK_EQ(w->stored_shape[1],rb);
    CHECK_EQ(w->stored_bytes,rows*rb);CHECK_EQ(w->align,256);
    CHECK_EQ(w->lay_dom,int(RAD_DOMAIN_HOST));CHECK_EQ(w->lay_operand,0);
    CHECK(w->lay_op>0);CHECK_EQ(w->first_use_op,w->lay_op);CHECK_EQ(w->last_use_op,w->lay_op);
    CHECK(!weight_file_direct(*w,b.program().ctx.world_size));
    RadTensor planes[RAD_ENC_MAX_PLANES]{};weight_planes(*w,planes);
    for(int p=0;p<w->n_sel;++p) {
        int64_t pr=0,pc=0;rad_enc_plane_dims(&w->enc.plane[p],rows,cols,&pr,&pc);
        CHECK_EQ(w->sel[p],p);CHECK_EQ(planes[p].shape[0],pr);CHECK_EQ(planes[p].shape[1],pc);
        CHECK_EQ(planes[p].dtype,w->enc.plane[p].dtype);CHECK_EQ(planes[p].rank,2u);
        CHECK_EQ(planes[p].stride[0],pc);CHECK_EQ(planes[p].stride[1],1);
        CHECK(planes[p].data==nullptr);
    }
    const auto* op=b.program().op(rad_op(w->lay_op));REQUIRE(op);REQUIRE_EQ(op->bands.size(),1u);
    CHECK(!op->bands[0].dom[RAD_DOMAIN_DEVICE]);
    const auto& resolved=op->bands[0].dom[RAD_DOMAIN_HOST];REQUIRE(resolved.row);
    CHECK_EQ(resolved.row->plugin,std::string("radtest_repackkernel"));
    REQUIRE(resolved.row->info->relayout);
    REQUIRE(resolved.row->info->unrelayout);
    CHECK_EQ(resolved.row->info->launch(nullptr,nullptr),RAD_E_STATE);
}
}

TEST(gguf_relayout_builder_selects_native_storage_for_all_six_importer_encodings) {
    Registry reg;if(load(reg)<0)ARTIFACT_MISSING("radtest_repackkernel in testkernels/repack");
    for(const auto format:formats) {
        const auto src=source(format,4,768);Builder b(reg,meta(),context());sources(b,{{"weight",src}});
        const auto w=weight(b,"weight",src,4,768,RAD_SHARD_NONE,rad_group_layer(0));REQUIRE(w!=0);
        REQUIRE(b.decl_op("test_repack_layout",nullptr,0,&w,1)!=0);
        REQUIRE_EQ(b.finish(),RAD_OK);CHECK(b.errors().empty());
        verify(b,w,format,4,768);
    }
}
TEST(gguf_relayout_builder_preserves_iq_tp4_expert_gate_up_down_geometry) {
    Registry reg;if(load(reg)<0)ARTIFACT_MISSING("radtest_repackkernel in testkernels/repack");
    constexpr int64_t nff=2304,hidden=5120;
    for(int f=0;f<4;++f)for(int rank=0;rank<4;++rank) {
        const auto format=formats[f];Builder b(reg,meta(),context(rank,4));Sources input;
        for(int layer=0;layer<4;++layer)for(int expert=0;expert<2;++expert)
            for(const std::string kind:{"gate","up","down"}) {
                const auto name="blk."+std::to_string(layer)+".experts."+std::to_string(expert)+"."+kind;
                input.emplace(name,source(format,kind=="down"?hidden:nff,kind=="down"?nff:hidden));
            }
        sources(b,input);
        struct Expected {rad_weight w;int64_t rows,cols,lo,hi;int layer,expert,shard;};
        std::vector<Expected> expected;
        for(int layer=0;layer<4;++layer) {
            int64_t lo=0;for(int r=0;r<rank;++r)lo+=r==layer%4?768:512;
            const int64_t width=rank==layer%4?768:512,hi=lo+width;
            CHECK_EQ(lo%256,0);CHECK_EQ(hi%256,0);
            for(int expert=0;expert<2;++expert)for(const std::string kind:{"gate","up","down"}) {
                const auto name="blk."+std::to_string(layer)+".experts."+std::to_string(expert)+"."+kind;
                const bool down=kind=="down";const int64_t rows=down?hidden:width,cols=down?width:hidden;
                const int shard=down?RAD_SHARD_COL:RAD_SHARD_ROW;
                const auto w=weight(b,name.c_str(),input.at(name),rows,cols,shard,rad_group_expert(layer,expert));
                REQUIRE(w!=0);REQUIRE_EQ(rad_weight_shard_span(&b,w,lo,hi),RAD_OK);
                REQUIRE(b.decl_op("test_repack_layout",nullptr,0,&w,1)!=0);
                expected.push_back({w,rows,cols,lo,hi,layer,expert,shard});
            }
        }
        REQUIRE_EQ(b.finish(),RAD_OK);CHECK(b.errors().empty());
        int64_t total=0;
        for(const auto& e:expected) {
            verify(b,e.w,format,e.rows,e.cols);const auto* w=b.program().weight(e.w);
            CHECK_EQ(w->shard_lo,e.lo);CHECK_EQ(w->shard_hi,e.hi);CHECK_EQ(w->decl.shard,e.shard);
            CHECK_EQ(w->decl.group.layer,e.layer);CHECK_EQ(w->decl.group.expert,e.expert);
            total+=w->stored_bytes;
        }
        // Across all four layer rotations each rank owns exactly 2304 FFN rows/columns.
        const auto* g=gg::geometry(format);
        CHECK_EQ(total,2*3*nff*hidden/g->values*g->bytes);
    }
}
RAD_TEST_MAIN()
