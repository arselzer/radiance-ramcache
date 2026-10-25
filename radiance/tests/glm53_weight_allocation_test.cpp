#include "rad_test.h"
#include "../tools/glm53_weight_allocation.h"
namespace gp=rad::glm53_weight_probe;
namespace {
gp::Inventory toy() {
    gp::Inventory i;i.revision="synthetic-small-count-same-geometry";i.experts=16;i.topk=8;
    i.layers={{3,gp::Format::mxfp4},{45,gp::Format::fp8_f32}};return i;
}
}
TEST(glm53_actual_builder_keeps_main_and_mtp_selected_planes_and_fused_row_parts) {
    rad::Registry registry;const auto i=toy();
    for(int rank=0;rank<4;++rank) {
        auto b=gp::build(registry,i,rank);const auto& p=b->program();REQUIRE_EQ(p.weights.size(),129u);
        for(size_t h=1;h<p.weights.size();++h) {
            const auto& w=p.weights[h];const bool mtp=w.decl.group.layer==45;
            CHECK(w.identity);CHECK(w.layout_tag.empty());CHECK_EQ(w.n_sel,1);CHECK(!rad::weight_file_direct(w,4));
            CHECK_EQ(w.enc.plane[1].dtype,uint32_t(mtp?RAD_F32:RAD_E8M0));
            CHECK_EQ(w.enc.plane[1].block[0],mtp?128:1);CHECK_EQ(w.enc.plane[1].block[1],mtp?128:32);
            const bool scale=w.sel[0]==1,down=w.decl.group.slot==1;
            const int divisor=scale?(mtp?128:down?32:1):1;
            CHECK_EQ(w.shard_lo,rank*512/divisor);CHECK_EQ(w.shard_hi,(rank+1)*512/divisor);
            CHECK_EQ(w.decl.shard,down?RAD_SHARD_COL:RAD_SHARD_ROW);
            if(!down){CHECK_EQ(w.decl.n_row_parts,2);CHECK_EQ(w.decl.row_parts[0],mtp&&scale?4:512);CHECK_EQ(w.decl.row_parts[1],w.decl.row_parts[0]);}
        }
    }
}
TEST(glm53_real_planner_records_pin_canonical_member_offsets_and_distinct_class_padding) {
    rad::Registry registry;const auto i=toy();auto b=gp::build(registry,i,0);rad::Config c;c.placement="all_vram";c.vram_weights_mib=0;c.host_pool_mib=0;
    const auto r=gp::report(i,b->program(),c);CHECK(r["fits_weight_only_assumptions"].get<bool>());
    REQUIRE_EQ(r["movement_unit_count"].get<int>(),32);REQUIRE_EQ(r["proposed_record_classes"].size(),2u);
    for(const auto& rec:r["proposed_record_classes"]) {
        const bool mtp=rec["format"]=="fp8_e4m3_f32_b128";
        const std::array<int64_t,4> offsets=mtp?std::array<int64_t,4>{0,4*gp::wp::MiB,4*gp::wp::MiB+4096,6*gp::wp::MiB+4096}:
            std::array<int64_t,4>{0,2*gp::wp::MiB,2*gp::wp::MiB+131072,3*gp::wp::MiB+131072};
        REQUIRE_EQ(rec["members"].size(),4u);
        for(size_t j=0;j<4;++j)CHECK_EQ(rec["members"][j]["offset"].get<int64_t>(),offsets[j]);
        CHECK_EQ(rec["record_stride_bytes"].get<int64_t>(),mtp?6*gp::wp::MiB+8192:3*gp::wp::MiB+196608);
        CHECK_EQ(rec["file_direct"].get<bool>(),false);
    }
}
TEST(glm53_routed_probe_reports_real_budget_refusal_without_claiming_ssd_ram_cache) {
    rad::Registry registry;const auto i=toy();auto b=gp::build(registry,i,0);rad::Config c;c.placement="expert_tiered";c.vram_weights_mib=8;c.host_pool_mib=1;
    const auto r=gp::report(i,b->program(),c);CHECK(!r["fits_weight_only_assumptions"].get<bool>());
    CHECK(!r["reservation_counts_valid_for_allocation"].get<bool>());CHECK(r["planner_status"].get<int>()<0);CHECK(!r["unfit"].empty());
    CHECK_EQ(c.vram_weights_mib,8);CHECK_EQ(c.host_pool_mib,1);
}
RAD_TEST_MAIN()
