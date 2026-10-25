#include "rad_test.h"
#include "../arch/deepseek41/projection_handoffs.h"
#include <nlohmann/json.hpp>
#include <fstream>

namespace h=rad::deepseek41::handoff;
namespace {
const nlohmann::json& inventory(){
    static const auto j=[] {std::ifstream file(std::string(RAD_GOLDEN_FIXTURES)+"/gguf-weight-inventory.json");
        if(!file)throw std::runtime_error("missing pinned metadata-only inventory");
        return nlohmann::json::parse(file);}();return j;
}
h::Source source(const char* tensor){
    const auto& data=inventory();
    for(const auto& row:data.at("tensors"))if(row[0].get<std::string>()==tensor){
        h::Source result;result.asset_identity=data.at("source_repo").get<std::string>()+"@"+data.at("source_revision").get<std::string>();
        result.tensor_name=tensor;const auto type=row[1].get<std::string>();const auto dims=row[2].get<std::vector<int64_t>>();
        result.rank=uint32_t(dims.size());if(dims.size()>3)throw std::runtime_error("fixture projection rank");
        for(size_t i=0;i<dims.size();++i)result.shape[i]=dims[dims.size()-1-i];
        result.payload_bytes=row[3].get<uint64_t>();
        if(type=="BF16"||type=="F32")result.encoding=rad_enc_plain(type=="BF16"?RAD_BF16:RAD_F32);
        else if(type=="Q5_K"){result.encoding=rad::gguf::canonical_encoding(rad::gguf::Format::q5_k);result.layout=h::StorageLayout::gguf_block_rows;}
        else throw std::runtime_error("fixture source type outside test cases");
        return result;
    }
    throw std::runtime_error("missing pinned tensor metadata");
}
h::Source tiny_grouped(){
    h::Source s;s.asset_identity="synthetic-independent-group-witness";s.tensor_name="groups.weight";
    s.encoding=rad_enc_plain(RAD_F32);s.rank=3;s.shape={2,2,3};s.payload_bytes=48;return s;
}
}

TEST(deepseek41_projection_handoffs_pinned_plain_matrices_preserve_storage_and_computation_boundaries){
    CHECK_EQ(inventory().at("source_revision").get<std::string>(),"cb02422c900df489cea5833e8a92171224059b3c");
    CHECK(!inventory().at("payload_downloaded").get<bool>());CHECK(!inventory().at("payload_hashes_verified").get<bool>());
    struct Case {const char* name;h::PlainRole role;uint32_t output;};
    const Case cases[]{
        {"blk.2.attn_compressor_kv.weight",h::PlainRole::c2_projection,RAD_F32},
        {"blk.2.attn_compressor_gate.weight",h::PlainRole::c2_projection,RAD_F32},
        {"blk.2.ffn_gate_inp.weight",h::PlainRole::router_logits,RAD_F32},
        {"blk.2.indexer.attn_q_b.weight",h::PlainRole::index_bf16,RAD_BF16},
        {"blk.2.indexer.attn_k.weight",h::PlainRole::index_bf16,RAD_BF16},
        {"blk.2.indexer.proj.weight",h::PlainRole::index_bf16,RAD_BF16},
        {"blk.2.hc_attn_fn.weight",h::PlainRole::generic,RAD_F32}};
    for(const auto& test:cases){
        const auto s=source(test.name);h::PlainSpec request;request.tokens=3;request.inputs=s.shape[1];request.outputs=s.shape[0];
        request.role=test.role;request.output_dtype=test.output;h::PlainPlan plan;
        REQUIRE_EQ(h::plan_plain(s,request,&plan),RAD_OK);
        CHECK_EQ(plan.source.asset_identity,s.asset_identity);CHECK_EQ(plan.source.tensor_name,s.tensor_name);
        CHECK_EQ(plan.params.compute_dtype,uint32_t(RAD_F32));CHECK_EQ(plan.params.output_dtype,test.output);
        CHECK_EQ(plan.input.cols,s.shape[1]);CHECK_EQ(plan.weight.rows,s.shape[0]);CHECK_EQ(plan.weight.span_bytes,s.payload_bytes);
        CHECK_EQ(plan.output.span_bytes,uint64_t(3*s.shape[0]*(test.output==RAD_F32?4:2)));
        CHECK(plan.promote_input_to_f32);CHECK_EQ(plan.promote_weight_to_f32,plan.weight_storage_dtype==RAD_BF16);
    }
}

TEST(deepseek41_projection_handoffs_c2_router_default_to_f32_and_reject_premature_output_cast){
    for(auto role:{h::PlainRole::c2_projection,h::PlainRole::router_logits}){
        const auto s=source(role==h::PlainRole::c2_projection?"blk.2.attn_compressor_kv.weight":"blk.2.ffn_gate_inp.weight");
        h::PlainSpec request;request.tokens=1;request.inputs=s.shape[1];request.outputs=s.shape[0];request.role=role;
        h::PlainPlan plan;REQUIRE_EQ(h::plan_plain(s,request,&plan),RAD_OK);CHECK_EQ(plan.output.dtype,uint32_t(RAD_F32));
        const auto saved=plan.source.tensor_name;request.output_dtype=RAD_BF16;
        CHECK_EQ(h::plan_plain(s,request,&plan),RAD_E_DTYPE);CHECK_EQ(plan.source.tensor_name,saved);CHECK_EQ(plan.output.dtype,uint32_t(RAD_F32));
        request.role=h::PlainRole::generic;REQUIRE_EQ(h::plan_plain(s,request,&plan),RAD_OK);CHECK_EQ(plan.output.dtype,uint32_t(RAD_BF16));
    }
    auto s=source("blk.2.indexer.attn_k.weight");h::PlainSpec request{1,s.shape[1],s.shape[0]};request.role=h::PlainRole::index_bf16;
    h::PlainPlan plan;CHECK_EQ(h::plan_plain(s,request,&plan),RAD_E_DTYPE);request.output_dtype=RAD_BF16;REQUIRE_EQ(h::plan_plain(s,request,&plan),RAD_OK);
}

TEST(deepseek41_projection_handoffs_pinned_grouped_woa_has_distinct_input_output_and_original_block_slices){
    const auto s=source("blk.2.attn_output_a.weight");
    for(int world:{1,4})for(int rank=0;rank<world;++rank){
        h::GroupedSpec request{3,8,4096,1024,rank,world};h::GroupedPlan plan;
        REQUIRE_EQ(h::plan_grouped(s,request,&plan),RAD_OK);CHECK(plan.packed);CHECK_EQ(plan.plain_weight_dtype,0u);
        CHECK_EQ(plan.group_count,8/world);CHECK_EQ(plan.group_begin,rank*(8/world));
        CHECK_EQ(plan.full_input_bytes,uint64_t(3*8*4096*2));CHECK_EQ(plan.full_output_bytes,uint64_t(3*8*1024*2));
        for(int64_t local=0;local<plan.group_count;++local){
            const auto& g=plan.slices[size_t(local)];const auto group=plan.group_begin+local;
            CHECK_EQ(g.global_group,group);CHECK_EQ(g.input.element_offset,group*4096);CHECK_EQ(g.input.row_stride,8*4096);
            CHECK_EQ(g.output.element_offset,group*1024);CHECK_EQ(g.output.row_stride,8*1024);
            CHECK_EQ(g.logical_weight.row,group*1024);CHECK_EQ(g.logical_weight.rows,1024);CHECK_EQ(g.logical_weight.cols,4096);
            CHECK_EQ(g.original_block_bytes,uint64_t(1024*(4096/256)*176));
            CHECK_EQ(g.original_block_byte_offset,uint64_t(group)*g.original_block_bytes);
            CHECK(g.input.byte_offset+g.input.span_bytes<=plan.full_input_bytes);
            CHECK(g.output.byte_offset+g.output.span_bytes<=plan.full_output_bytes);
            CHECK(g.original_block_byte_offset+g.original_block_bytes<=s.payload_bytes);
        }
    }
}

TEST(deepseek41_projection_handoffs_independent_group_witness_exposes_flattened_common_input_error){
    const auto s=tiny_grouped();h::GroupedSpec request{2,2,3,2};request.input_dtype=request.output_dtype=RAD_F32;
    h::GroupedPlan plan;REQUIRE_EQ(h::plan_grouped(s,request,&plan),RAD_OK);CHECK(!plan.packed);
    // Independent explicit contractions: group0 identity first2 coordinates;
    // group1 uses its OWN vector and asymmetric weighted rows.
    const std::array<float,12> x{1,2,3, 7,11,13, 4,5,6, 17,19,23};
    const std::array<float,12> w{1,0,0, 0,1,0, 1,2,3, -1,1,2};
    const std::array<float,8> expected{1,2,68,30, 4,5,124,48};
    std::array<float,8> actual{},wrong{};
    for(int64_t group=0;group<plan.group_count;++group){
        const auto& g=plan.slices[size_t(group)];
        for(int64_t row=0;row<2;++row)for(int64_t n=0;n<2;++n){
            float value=0,flat=0;
            for(int64_t k=0;k<3;++k){
                const float weight=w[size_t((g.logical_weight.row+n)*3+k)];
                value+=x[size_t(g.input.element_offset+row*g.input.row_stride+k)]*weight;
                flat+=x[size_t(row*6+k)]*weight; // incorrect common group0 input
            }
            const auto out=size_t(g.output.element_offset+row*g.output.row_stride+n);actual[out]=value;wrong[out]=flat;
        }
    }
    CHECK(actual==expected);CHECK(wrong!=expected);CHECK_EQ(wrong[2],14.f);CHECK_EQ(actual[2],68.f);
    // This is a CPU arithmetic witness of metadata slices, NOT a GEMM/graph test.
}

TEST(deepseek41_projection_handoffs_bad_source_layout_shape_identity_and_payload_preserve_output){
    const auto original=source("blk.2.attn_compressor_kv.weight");h::PlainSpec request{2,5120,512};request.role=h::PlainRole::c2_projection;
    h::PlainPlan output;REQUIRE_EQ(h::plan_plain(original,request,&output),RAD_OK);const auto saved=output.source.tensor_name;
    for(int error=0;error<9;++error){
        auto s=original;
        if(error==0)s.shape[0]=INT64_MAX;
        if(error==1)s.rank=3;
        if(error==2)s.shape[2]=1;
        if(error==3)--s.payload_bytes;
        if(error==4)s.encoding=rad_enc_plain(RAD_F16);
        if(error==5)s.layout=h::StorageLayout::gguf_block_rows;
        if(error==6)s.asset_identity.assign(257,'x');
        if(error==7)s.tensor_name.assign(513,'x');
        if(error==8)s.tensor_name=std::string("x\0hidden",8);
        CHECK(h::plan_plain(s,request,&output)<0);CHECK_EQ(output.source.tensor_name,saved);CHECK_EQ(output.output.dtype,uint32_t(RAD_F32));
    }
    request.tokens=INT64_MAX;CHECK(h::plan_plain(original,request,&output)<0);CHECK_EQ(output.source.tensor_name,saved);
    CHECK_EQ(h::plan_plain(original,request,nullptr),RAD_E_INVAL);
}

TEST(deepseek41_projection_handoffs_grouped_rejects_incomplete_groups_formats_and_unbounded_geometry_atomically){
    const auto original=source("blk.2.attn_output_a.weight");const h::GroupedSpec initial{2,8,4096,1024};h::GroupedPlan output;
    REQUIRE_EQ(h::plan_grouped(original,initial,&output),RAD_OK);const auto saved=output.source.tensor_name;
    for(int error=0;error<11;++error){
        auto s=original;auto request=initial;
        if(error==0)request.world_size=3;
        if(error==1)request.rank=1;
        if(error==2){s.shape[0]=6;s.payload_bytes=s.payload_bytes/8*6;request.groups=6;request.world_size=4;}
        if(error==3)s.shape[2]=INT64_MAX;
        if(error==4)request.tokens=h::max_tokens+1;
        if(error==5)--s.payload_bytes;
        if(error==6)s.layout=h::StorageLayout::plain_row_major;
        if(error==7)s.encoding=rad_enc_plain(RAD_F32);
        if(error==8)request.input_dtype=RAD_F16;
        if(error==9){s.shape[2]=4095;request.inputs_per_group=4095;}
        if(error==10)s.asset_identity.clear();
        CHECK(h::plan_grouped(s,request,&output)<0);CHECK_EQ(output.source.tensor_name,saved);
        CHECK_EQ(output.group_count,8);CHECK_EQ(output.slices[1].input.element_offset,4096);
    }
    CHECK_EQ(h::plan_grouped(original,initial,nullptr),RAD_E_INVAL);
}
RAD_TEST_MAIN()
