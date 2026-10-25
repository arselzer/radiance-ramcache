/* CPU-only real Planner tests for joint initial residency. No plugins or model allocations. */
#include "rad_test.h"
#include "place/tp_plan.h"
#include <array>

using namespace rad;
namespace {
constexpr int64_t MiB=1ll<<20;
const KernelRow* device_row() {
    static const RadKernelInfo info{.name="tp_plan_fixture",.op="gemm",.family="gemm",.domain=RAD_DOMAIN_DEVICE};
    static const KernelRow row=[] {KernelRow r;r.info=&info;return r;}();return &row;
}
struct Fixture {
    Program program;
    Config config;
    std::vector<RadFileProfile> profile;
    Fixture(int rank,std::vector<int64_t> costs,int64_t vram=9,int64_t host=16,bool dynamic=false) {
        program.ctx.rank=rank;program.ctx.world_size=2;program.weights.resize(1);program.ops.resize(1);
        config.placement="expert_tiered";config.vram_weights_mib=vram;config.host_pool_mib=host;
        config.deterministic=!dynamic;
        add("dense",4096,-1,-1,RAD_ACCESS_PER_TOKEN);
        for(size_t e=0;e<costs.size();++e)add("expert."+std::to_string(e),costs[e],0,int(e),RAD_ACCESS_CONDITIONAL);
    }
    void add(std::string name,int64_t bytes,int layer,int expert,int access) {
        WeightInfo w;w.name=std::move(name);w.decl.dtype=RAD_U8;w.decl.rank=1;w.decl.shape[0]=bytes;
        w.decl.access=access;w.decl.group.layer=layer;w.decl.group.expert=expert;
        w.stored_bytes=bytes;w.logical_bytes=bytes;w.align=RAD_ALIGN_UNIT;
        w.identity=false;w.layout_tag="tp-fixture-native"; // Not file-direct: honest host refusal.
        const auto h=rad_weight(program.weights.size());program.weights.push_back(w);
        OpInfo o;o.op="gemm";o.index=int32_t(program.ops.size());o.weights={h};
        Band b;b.hi=1;b.dom[RAD_DOMAIN_DEVICE].row=device_row();o.bands.push_back(b);
        program.weights[h].first_use_op=o.index;program.weights[h].last_use_op=o.index;program.ops.push_back(o);
    }
    PlannerInput input() const {
        return {&program,&config,program.ctx.rank,2,profile.empty()?nullptr:profile.data(),int64_t(profile.size()),true,false};
    }
};
std::array<PlannerInput,2> inputs(const Fixture& a,const Fixture& b){return {a.input(),b.input()};}
void check_costs(const Plan& plan,const Fixture& f) {
    for(const auto& u:plan.units) {
        REQUIRE_EQ(u.weights.size(),1u);
        CHECK_EQ(u.bytes,f.program.weights[u.weights[0]].stored_bytes);
        CHECK_EQ(u.slot_bytes,(u.bytes+RAD_ALIGN_UNIT-1)/RAD_ALIGN_UNIT*RAD_ALIGN_UNIT);
    }
}
}
TEST(tp_plan_default_null_and_complete_filter_preserve_existing_plan) {
    Fixture f(0,{6*MiB,4*MiB,4*MiB});auto in=f.input();Plan baseline,explicit_null,complete;
    REQUIRE_EQ(Planner(in).plan(&baseline),RAD_OK);
    in.resident_eligibility=nullptr;REQUIRE_EQ(Planner(in).plan(&explicit_null),RAD_OK);
    ExpertEligibility all{{0,0},{0,1},{0,2}};in.resident_eligibility=&all;
    REQUIRE_EQ(Planner(in).plan(&complete),RAD_OK);
    CHECK_EQ(baseline.report(f.program),explicit_null.report(f.program));
    CHECK_EQ(baseline.report(f.program),complete.report(f.program));
    CHECK_EQ(all.size(),3u);check_costs(complete,f);
    const ExpertEligibility empty;in.resident_eligibility=&empty;Plan off;
    REQUIRE_EQ(Planner(in).plan(&off),RAD_OK);CHECK(resident_experts(off).empty());
    CHECK(off.weight[1].tier==Tier::VRAM);CHECK_EQ(off.totals.bytes[int(Tier::HostPinned)],14*MiB);
    check_costs(off,f);
}
TEST(tp_plan_intersection_coordinates_unequal_costs_without_padding) {
    Fixture a(0,{6*MiB,4*MiB,4*MiB}),b(1,{4*MiB,6*MiB,4*MiB});
    const auto in=inputs(a,b);Plan pa,pb;
    REQUIRE_EQ(Planner(in[0]).plan(&pa),RAD_OK);REQUIRE_EQ(Planner(in[1]).plan(&pb),RAD_OK);
    CHECK(resident_experts(pa)!=resident_experts(pb));
    std::vector<Plan> plans;TpInitialPlanDiagnostics d;
    REQUIRE_EQ(coordinate_initial_expert_placement(in,&plans,&d),RAD_OK);
    REQUIRE_EQ(plans.size(),2u);CHECK_EQ(d.rounds.size(),2u);
    CHECK(resident_experts(plans[0])==resident_experts(plans[1]));
    CHECK(d.common_resident==ExpertEligibility({{0,0}}));
    CHECK(d.initial_placement_only);CHECK(!d.heat_safe);
    CHECK_EQ(plans[0].weight[2].bytes,6*MiB);CHECK_EQ(plans[1].weight[2].bytes,4*MiB);
    check_costs(plans[0],a);check_costs(plans[1],b);
    CHECK_EQ(a.config.host_pool_mib,16);CHECK(in[0].resident_eligibility==nullptr);
}
TEST(tp_plan_refusal_preserves_output_programs_configs_profiles_and_filters) {
    Fixture a(0,{6*MiB,4*MiB,4*MiB},9,8),b(1,{4*MiB,6*MiB,4*MiB},9,8);
    const auto in=inputs(a,b);Plan independent;
    REQUIRE_EQ(Planner(in[0]).plan(&independent),RAD_OK);REQUIRE_EQ(Planner(in[1]).plan(&independent),RAD_OK);
    std::vector<Plan> untouched(1);untouched[0].rank=77;untouched[0].notes={"original"};
    TpInitialPlanDiagnostics d;
    CHECK_EQ(coordinate_initial_expert_placement(in,&untouched,&d),RAD_E_NOMEM);
    REQUIRE_EQ(untouched.size(),1u);CHECK_EQ(untouched[0].rank,77);CHECK_EQ(untouched[0].notes[0],std::string("original"));
    CHECK_EQ(d.rank_status.size(),2u);CHECK_EQ(d.rank_status[1],RAD_E_NOMEM);
    CHECK_EQ(a.config.host_pool_mib,8);CHECK_EQ(b.program.weights[2].stored_bytes,4*MiB);
    const ExpertEligibility allowed{{0,0}};auto filtered=in;filtered[0].resident_eligibility=&allowed;
    CHECK_EQ(coordinate_initial_expert_placement(filtered,&untouched,&d),RAD_E_INVAL);
    CHECK(allowed==ExpertEligibility({{0,0}}));CHECK_EQ(untouched[0].rank,77);
    CHECK_EQ(coordinate_initial_expert_placement(in,nullptr,&d),RAD_E_INVAL);
    CHECK_EQ(coordinate_initial_expert_placement({},&untouched,&d),RAD_E_INVAL);CHECK_EQ(untouched[0].rank,77);
    auto invalid=in;invalid[1].config=nullptr;
    CHECK_EQ(coordinate_initial_expert_placement(invalid,&untouched,&d),RAD_E_INVAL);CHECK_EQ(untouched[0].rank,77);
}
TEST(tp_plan_rejects_mismatched_ownership_world_and_strategy) {
    Fixture a(0,{6*MiB,4*MiB,4*MiB}),b(1,{4*MiB,6*MiB});
    std::vector<Plan> out(1);out[0].rank=91;TpInitialPlanDiagnostics d;
    CHECK_EQ(coordinate_initial_expert_placement(inputs(a,b),&out,&d),RAD_E_INVAL);CHECK_EQ(out[0].rank,91);
    b.add("missing",4*MiB,0,2,RAD_ACCESS_CONDITIONAL);
    auto wrong=inputs(a,b);wrong[1].world_size=4;
    CHECK_EQ(coordinate_initial_expert_placement(wrong,&out,&d),RAD_E_INVAL);
    b.config.placement="all_vram";
    CHECK_EQ(coordinate_initial_expert_placement(inputs(a,b),&out,&d),RAD_E_INVAL);
    ExpertEligibility all{{0,0},{0,1},{0,2}};auto bad=b.input();bad.resident_eligibility=&all;
    Plan local;CHECK_EQ(Planner(bad).plan(&local),RAD_E_INVAL);CHECK_EQ(out[0].rank,91);
}
TEST(tp_plan_retains_profile_order_and_real_dynamic_spare_accounting) {
    std::vector<int64_t> ca(64,2*MiB),cb(64,2*MiB);
    for(size_t i=0;i<64;++i){if(i%8==0)ca[i]=3*MiB;if(i%8==7)cb[i]=3*MiB;}
    Fixture a(0,ca,32,256,true),b(1,cb,32,256,true);
    RadFileProfile hot{};hot.layer=0;hot.expert=63;hot.share=100;
    a.profile={hot};b.profile={hot};const auto original=a.profile;
    std::vector<Plan> plans;TpInitialPlanDiagnostics d;
    REQUIRE_EQ(coordinate_initial_expert_placement(inputs(a,b),&plans,&d),RAD_OK);
    REQUIRE_EQ(plans.size(),2u);CHECK(resident_experts(plans[0])==resident_experts(plans[1]));
    CHECK(d.common_resident.count({0,63})!=0);CHECK_EQ(a.profile[0].share,original[0].share);
    for(size_t r=0;r<plans.size();++r) {
        const auto& p=plans[r];CHECK(p.dynamic);CHECK(p.slab_overhead>0);
        CHECK(p.totals.bytes[int(Tier::VRAM)]+p.slab_overhead<=32*MiB);
        for(const auto& c:p.classes)if(c.n_slots>0)
            CHECK_EQ(c.n_shadow,std::min<int64_t>(RAD_PLACE_SHADOW_SLOTS,std::max<int64_t>(1,c.n_slots/4)));
    }
    check_costs(plans[0],a);check_costs(plans[1],b);CHECK(!d.heat_safe);
}
RAD_TEST_MAIN()
