#include "rad_test.h"
#include "plugin/registry.h"
TEST(ds41_device_metadata_matches_the_existing_libref_schemas) {
    rad::Registry reg;
    REQUIRE_EQ(reg.load_plugin(home()+"/kernels/libref.so",0),RAD_OK);
    // This uses the real device declaration source with never-launch stubs, so schema names,
    // order, requiredness and operand roles are checked by the actual Registry on the CPU.
    REQUIRE_EQ(reg.load_plugin(home()+"/testkernels/ds41meta/radtest_ds41meta.so",0),RAD_OK);
}
TEST(ds41_existing_reference_schemas_match_when_device_metadata_loads_first) {
    rad::Registry reg;
    REQUIRE_EQ(reg.load_plugin(home()+"/testkernels/ds41meta/radtest_ds41meta.so",0),RAD_OK);
    REQUIRE_EQ(reg.load_plugin(home()+"/kernels/libref.so",0),RAD_OK);
}
TEST(ds41_compressor_schema_and_selection_refuse_unsupported_state_profiles) {
    rad::Registry reg;
    REQUIRE_EQ(reg.load_plugin(home()+"/testkernels/ds41meta/radtest_ds41meta.so",0),RAD_OK);
    const auto* s=reg.schema("ds41_compress_pool");REQUIRE(s);
    CHECK_EQ(s->n_operands,11);CHECK_EQ(s->n_params,5);
    CHECK_EQ(std::string(s->operands[2].name),"pending_state");
    CHECK_EQ(s->operands[2].role,RAD_OPD_INOUT);
    rad::Geometry g;g.set_i("M",8);g.set_i("dim",512);g.set_i("ratio",2);g.set_i("n_spec",0);g.set_s("dtype","f32");
    REQUIRE(reg.select("ds41_compress_pool",g,RAD_DOMAIN_DEVICE));
    g.set_i("n_spec",3);CHECK(!reg.select("ds41_compress_pool",g,RAD_DOMAIN_DEVICE));
    g.set_i("n_spec",0);g.set_i("ratio",3);CHECK(!reg.select("ds41_compress_pool",g,RAD_DOMAIN_DEVICE));
    g.set_i("ratio",2);g.set_i("dim",1025);CHECK(!reg.select("ds41_compress_pool",g,RAD_DOMAIN_DEVICE));
}
TEST(ds41_paged_attention_profile_excludes_speculation_and_runtime_layout_inference) {
    rad::Registry reg;REQUIRE_EQ(reg.load_plugin(home()+"/testkernels/ds41meta/radtest_ds41meta.so",0),RAD_OK);
    const auto* s=reg.schema("ds41_sparse_attn_paged");REQUIRE(s);CHECK_EQ(s->n_params,11);CHECK_EQ(s->n_operands,13);
    CHECK_EQ(std::string(s->operands[1].name),"window_cache");CHECK_EQ(std::string(s->operands[4].name),"compressed_cache");
    CHECK_EQ(std::string(s->operands[12].name),"output");CHECK_EQ(s->operands[12].role,RAD_OPD_OUT);
    rad::Geometry g;g.set_i("M",8);g.set_i("sequences",2);g.set_i("heads",4);g.set_i("head_dim",512);
    g.set_i("window",128);g.set_i("topk",512);g.set_i("ratio",2);g.set_i("block_size",4);g.set_i("n_spec",0);g.set_s("dtype","bf16");
    REQUIRE(reg.select("ds41_sparse_attn_paged",g,RAD_DOMAIN_DEVICE));
    g.set_i("n_spec",3);CHECK(!reg.select("ds41_sparse_attn_paged",g,RAD_DOMAIN_DEVICE));
    g.set_i("n_spec",0);g.set_i("ratio",3);CHECK(!reg.select("ds41_sparse_attn_paged",g,RAD_DOMAIN_DEVICE));
    g.set_i("ratio",0);g.set_i("topk",0);CHECK(reg.select("ds41_sparse_attn_paged",g,RAD_DOMAIN_DEVICE));
}
TEST(ds41_route_job_schema_keeps_both_gather_maps_and_histogram_distinct) {
    rad::Registry reg;REQUIRE_EQ(reg.load_plugin(home()+"/testkernels/ds41meta/radtest_ds41meta.so",0),RAD_OK);
    const auto* s=reg.schema("ds41_route_jobs");REQUIRE(s);CHECK_EQ(s->n_params,3);CHECK_EQ(s->n_operands,7);
    CHECK_EQ(std::string(s->operands[3].name),"token_ids");CHECK_EQ(std::string(s->operands[5].name),"expert_counts");
    CHECK_EQ(std::string(s->operands[6].name),"identity_rows");
    for(int i=0;i<7;++i)CHECK_EQ(s->operands[i].role,i<2?RAD_OPD_IN:RAD_OPD_OUT);
    rad::Geometry g;g.set_i("M",8);g.set_i("topk",8);g.set_i("n_expert",256);
    REQUIRE(reg.select("ds41_route_jobs",g,RAD_DOMAIN_DEVICE));
}
RAD_TEST_MAIN()
