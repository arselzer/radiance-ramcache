/* Actual Builder/source-encoding/DS41 private layout tests. Metadata only: no payloads,
 * device enumeration, pools, launches, full architecture, gather or TP collective. */
#include "quant_fixture.h"
#include "build/rad_build.h"
#include "place/planner.h"
#include "../arch/deepseek41/packed_projections.h"
#include "../libds41/ds41_packed_args.h"
#include <map>
#include <memory>

using namespace rad;
namespace pp=rad::deepseek41::packed;
namespace pk=rad::ds41::packed;
namespace gg=rad::gguf;
namespace {
struct Fixture {
    Registry registry;std::map<std::string,WeightSource> sources;
    std::map<std::string,std::string> names;RadBuildCtx context{};std::unique_ptr<Builder> b;
    explicit Fixture(int rank=0,int world=1,bool rows=true) {
        if(rows&&registry.load_plugin(home()+"/testkernels/ds41meta/radtest_ds41meta.so",0)!=RAD_OK)
            throw std::runtime_error("missing actual libds41 metadata fixture");
        context.rank=rank;context.world_size=world;context.max_tok=8;context.max_seqs=1;context.max_ctx=8;context.scope="";
        RadModelMeta m{};m.arch_id="ds41-projection-declaration-test";m.name="synthetic-no-model";m.quant="native-GGUF";
        m.n_layers=4;m.n_embd=5120;m.n_ff=2304;m.n_expert=3;
        b=std::make_unique<Builder>(registry,m,context);b->set_instances(false);
        b->set_sources([this](const std::string& name,const Program& p,WeightSource* out){
            auto found=sources.find(name);
            if(found==sources.end())for(const auto& map:p.name_map)if(map.declared&&name==map.declared&&map.mode==RAD_MAP_COPY&&map.n_src==1){found=sources.find(map.src[0]);break;}
            if(found==sources.end())return RAD_E_NOTFOUND;
            *out=found->second;return RAD_OK;
        });
    }
    const char* keep(const std::string& value){return names.emplace(value,value).first->second.c_str();}
    pp::Source add(const std::string& name,gg::Format format,int64_t rows,int64_t cols,bool mapped=false){
        const auto* importer=qfix::reg().checkpoint_importer("gguf",gg::geometry(format)->name);
        if(!importer)throw std::runtime_error("missing actual GGUF importer");
        WeightSource source{};source.rank=2;source.shape[0]=rows;source.shape[1]=cols;
        const RadCheckpointImportWeight desc{name.c_str(),gg::geometry(format)->name,rows,cols};
        if(importer->info->encoding(&desc,&source.enc)!=RAD_OK)throw std::runtime_error("importer metadata refused");
        const auto raw=mapped?"raw-"+name:name;sources.emplace(raw,source);
        const char* n=keep(name);
        if(mapped){RadNameMap map{};map.declared=n;map.mode=RAD_MAP_COPY;map.n_src=1;map.src[0]=keep(raw);map.src_index[0]=-1;
            if(rad_decl_name_map(b.get(),&map)!=RAD_OK)throw std::runtime_error("name map refused");}
        return {n,n};
    }
    pp::RoutedSpec ffn(int layer=0,gg::Format gate=gg::Format::iq2_xs,gg::Format up=gg::Format::iq3_xxs,gg::Format down=gg::Format::q5_k){
        pp::RoutedSpec s;s.layer=layer;s.hidden=5120;s.intermediate=2304;s.max_tokens=8;s.topk=2;
        for(int expert=0;expert<3;++expert){
            const std::string root="layer"+std::to_string(layer)+".expert"+std::to_string(expert)+".";
            s.experts.push_back({add(root+"gate",gate,2304,5120),add(root+"up",up,2304,5120),add(root+"down",down,5120,2304)});
        }
        return s;
    }
};
pk::Type kernel_type(gg::Format f){return static_cast<pk::Type>(static_cast<unsigned>(f));}
void verify_weight(const WeightInfo& w,gg::Format f,int64_t rows,int64_t cols){
    const auto* g=gg::geometry(f);CHECK(w.enc_known);CHECK(!w.identity);CHECK_EQ(w.n_sel,w.enc.n_planes);
    CHECK_EQ(w.decl.rank,2u);CHECK_EQ(w.decl.shape[0],rows);CHECK_EQ(w.decl.shape[1],cols);
    CHECK_EQ(w.stored_dtype,pk::storage_dtype(kernel_type(f)));CHECK_EQ(rad_dtype_bits(w.stored_dtype),0);
    CHECK_EQ(w.layout_tag,std::string(pk::runtime_tag(kernel_type(f))));
    CHECK_EQ(w.stored_bytes,rows*(cols/g->values)*g->bytes);CHECK_EQ(w.align,256);
    CHECK_EQ(w.stored_shape[0],rows);CHECK_EQ(w.stored_shape[1],(cols/g->values)*g->bytes);
    for(int plane=0;plane<w.n_sel;++plane)CHECK_EQ(w.sel[plane],plane);
}
}
TEST(ds41_native_linear_chooses_all_six_actual_mapped_source_encodings_and_private_layouts){
    for(const auto format:{gg::Format::iq2_xxs,gg::Format::iq2_xs,gg::Format::iq2_s,gg::Format::iq3_xxs,gg::Format::q5_k,gg::Format::q8_0}){
        Fixture f;pp::LinearSpec s;s.weight=f.add("linear",format,8,768,true);s.rows=8;s.cols=768;s.max_rows=8;s.input_dtype=RAD_F32;s.output_dtype=RAD_BF16;
        pp::NativeLinear result;std::string why;REQUIRE_EQ(pp::declare_native_linear(f.b.get(),f.context,s,&result,&why),RAD_OK);
        REQUIRE_EQ(f.b->finish(),RAD_OK);const auto& p=f.b->program();verify_weight(p.weights[result.weight],format,8,768);
        CHECK_EQ(p.weights[result.weight].source,std::string("linear"));CHECK_EQ(p.weights[result.weight].decl.shard,RAD_SHARD_NONE);
        const auto& op=p.ops[result.op];CHECK_EQ(op.op,std::string("ds41_gemm_packed"));CHECK_EQ(op.ranged_key,std::string("M"));
        CHECK_EQ(std::string(op.base.get_s("format")),std::string(gg::geometry(format)->name));
        REQUIRE(!op.bands.empty());const auto& row=op.bands[0].dom[RAD_DOMAIN_DEVICE];REQUIRE(row.row);REQUIRE(row.row->info->opd_shape);
        RadOpdDesc out{};REQUIRE_EQ(row.row->info->opd_shape(row.geom.params(),row.geom.n_params(),2,&out),RAD_OK);
        CHECK_EQ(out.shape[1],8);CHECK_EQ(out.dtype,uint32_t(RAD_BF16));
    }
}
TEST(ds41_native_linear_reviewed_rows_keep_whole_K_and_whole_output_groups){
    for(int rank=0;rank<4;++rank){
        Fixture f(rank,4);pp::LinearSpec s;s.weight=f.add("shared_down",gg::Format::q5_k,5120,2304);s.rows=5120;s.cols=2304;s.max_rows=8;
        s.ownership=pp::DenseOwnership::reviewed_rows;s.whole_output_unit_rows=1;s.full_input_and_output_reconstruction_reviewed=true;
        pp::NativeLinear result;REQUIRE_EQ(pp::declare_native_linear(f.b.get(),f.context,s,&result),RAD_OK);REQUIRE_EQ(f.b->finish(),RAD_OK);
        const auto& w=f.b->program().weights[result.weight];verify_weight(w,gg::Format::q5_k,1280,2304);
        CHECK_EQ(w.decl.shard,RAD_SHARD_ROW);CHECK_EQ(w.shard_lo,rank*1280);CHECK_EQ(w.shard_hi,(rank+1)*1280);
        CHECK_EQ(result.cols,2304); // Never cut shared-down K to576.
    }
    Fixture f(1,4);pp::LinearSpec s;s.weight=f.add("grouped_out",gg::Format::q5_k,8192,4096);
    auto& source=f.sources.at("grouped_out");source.rank=3;source.shape[0]=8;source.shape[1]=1024;source.shape[2]=4096;
    s.rows=8192;s.cols=4096;s.max_rows=8;s.source_groups=8;s.ownership=pp::DenseOwnership::reviewed_rows;
    s.whole_output_unit_rows=1024;s.full_input_and_output_reconstruction_reviewed=true;
    pp::NativeLinear result;REQUIRE_EQ(pp::declare_native_linear(f.b.get(),f.context,s,&result),RAD_OK);REQUIRE_EQ(f.b->finish(),RAD_OK);
    verify_weight(f.b->program().weights[result.weight],gg::Format::q5_k,2048,4096);CHECK_EQ(result.row_begin,2048);
}
TEST(ds41_routed_declaration_preserves_mixed_projection_formats_and_rotating_256_block_ownership){
    for(int layer=0;layer<4;++layer)for(int rank=0;rank<4;++rank){
        Fixture f(rank,4);auto s=f.ffn(layer);pp::RoutedFFN result;
        REQUIRE_EQ(pp::declare_routed_ffn(f.b.get(),f.context,s,&result),RAD_OK);REQUIRE_EQ(f.b->finish(),RAD_OK);
        const int64_t width=rank==layer?768:512;int64_t lo=0;for(int r=0;r<rank;++r)lo+=r==layer?768:512;
        CHECK_EQ(result.partition.width,width);CHECK_EQ(result.partition.offset,lo);CHECK_EQ(width%256,0);
        const pp::RoutedProjection* projections[]{&result.gate,&result.up,&result.down};
        const gg::Format formats[]{gg::Format::iq2_xs,gg::Format::iq3_xxs,gg::Format::q5_k};
        const auto& program=f.b->program();
        for(int p=0;p<3;++p){const auto& projection=*projections[p];REQUIRE_EQ(projection.weights.size(),3u);
            CHECK_EQ(projection.input_capacity,p==2?16:8);CHECK_EQ(projection.route_capacity,16);
            const auto& op=program.ops[projection.op];CHECK_EQ(op.op,std::string("ds41_routed_packed"));CHECK_EQ(op.ranged_key,std::string("M"));
            CHECK_EQ(std::string(op.base.get_s("y_dtype")),std::string(p==2?"f32":"bf16"));
            for(int e=0;e<3;++e){const auto& w=program.weights[projection.weights[size_t(e)]];
                CHECK_EQ(projection.weights[size_t(e)],projection.weights[0]+e);verify_weight(w,formats[p],p==2?5120:width,p==2?width:5120);
                CHECK_EQ(w.decl.group.layer,layer);CHECK_EQ(w.decl.group.expert,e);CHECK_EQ(w.decl.group.slot,p);
                CHECK_EQ(w.decl.shard,p==2?RAD_SHARD_COL:RAD_SHARD_ROW);CHECK_EQ(w.shard_lo,lo);CHECK_EQ(w.shard_hi,lo+width);
            }
        }
        Config c;c.placement="all_vram";c.vram_weights_mib=0;c.host_pool_mib=0;Plan plan;
        PlannerInput in{&program,&c,rank,4,nullptr,0,true,false};REQUIRE_EQ(Planner(in).plan(&plan),RAD_OK);REQUIRE_EQ(plan.units.size(),3u);
        for(const auto& unit:plan.units){REQUIRE_EQ(unit.weights.size(),3u);CHECK_EQ(unit.layer,layer);CHECK(!unit.file_direct);}
    }
}
TEST(ds41_projection_preflight_rejects_missing_unknown_mixed_table_and_wrong_geometry_without_declarations){
    for(int fault=0;fault<6;++fault){
        Fixture f(0,4);auto s=f.ffn();pp::RoutedFFN out;out.gate.weights={77};out.partition.width=99;
        if(fault==0)f.sources.erase(s.experts[2].up.source);
        if(fault==1)f.sources.at(s.experts[2].up.source).enc=rad_enc_plain(RAD_BF16);
        if(fault==2)f.sources.at(s.experts[2].up.source).enc=gg::canonical_encoding(gg::Format::q5_k);
        if(fault==3)f.sources.at(s.experts[2].down.source).shape[1]=2048;
        if(fault==4)s.intermediate=2305;
        if(fault==5)s.experts[2].down.name=s.experts[0].gate.name;
        const auto weights=f.b->program().weights.size(),ops=f.b->program().ops.size();
        CHECK(pp::declare_routed_ffn(f.b.get(),f.context,s,&out)<0);CHECK_EQ(f.b->program().weights.size(),weights);CHECK_EQ(f.b->program().ops.size(),ops);
        CHECK(out.gate.weights==std::vector<rad_weight>{77});CHECK_EQ(out.partition.width,99);
    }
}
TEST(ds41_dense_preflight_requires_explicit_head_group_review_and_preserves_outputs_on_failure){
    for(int fault=0;fault<5;++fault){
        Fixture f(0,4);pp::LinearSpec s;s.weight=f.add("heads",gg::Format::q5_k,32768,1280);s.rows=32768;s.cols=1280;s.max_rows=8;
        s.ownership=pp::DenseOwnership::reviewed_rows;s.whole_output_unit_rows=512;s.full_input_and_output_reconstruction_reviewed=true;
        if(fault==0)s.full_input_and_output_reconstruction_reviewed=false;
        if(fault==1)s.whole_output_unit_rows=513;
        if(fault==2)s.rows=32767;
        if(fault==3)s.ownership=static_cast<pp::DenseOwnership>(999);
        if(fault==4)s.source_groups=8;
        pp::NativeLinear out;out.weight=77;out.op=88;CHECK(pp::declare_native_linear(f.b.get(),f.context,s,&out)<0);
        CHECK_EQ(out.weight,77u);CHECK_EQ(out.op,88u);CHECK_EQ(f.b->program().weights.size(),1u);
    }
}
TEST(ds41_projection_helpers_refuse_missing_native_schema_without_declaring_a_fake_fallback){
    Fixture f(0,1,false);pp::LinearSpec s;s.weight=f.add("no-row",gg::Format::iq2_s,8,768);s.rows=8;s.cols=768;s.max_rows=8;
    pp::NativeLinear linear;linear.op=77;CHECK_EQ(pp::declare_native_linear(f.b.get(),f.context,s,&linear),RAD_E_NOKERNEL);
    CHECK_EQ(linear.op,77u);CHECK_EQ(f.b->program().weights.size(),1u);CHECK_EQ(f.b->program().ops.size(),1u);
    auto spec=f.ffn();pp::RoutedFFN routed;routed.layer=99;
    CHECK_EQ(pp::declare_routed_ffn(f.b.get(),f.context,spec,&routed),RAD_E_NOKERNEL);
    CHECK_EQ(routed.layer,99);CHECK_EQ(f.b->program().weights.size(),1u);CHECK_EQ(f.b->program().ops.size(),1u);
}
RAD_TEST_MAIN()
