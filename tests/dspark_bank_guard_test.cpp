/* Conflict-only startup guards. The early refusal precedes device or file discovery. */
#include "rad_test.h"
#include "rad_core.h"
#include "engine.h"
using namespace rad;
TEST(dspark_manual_placement_conflict_stops_engine_before_pool_growth_or_model_open) {
    for(int mode=0;mode<3;++mode) {
        Config cfg;cfg.model="/must-not-open-dspark-placement-conflict.rad";
        cfg.draft_bank="/must-not-open-donor.rad";cfg.host_pool_mib=1234;
        cfg.draft_bank_expert_placement=mode==2?"auto":"host";
        cfg.draft_expert_placement=mode==1?"auto":"host";
        cfg.draft_expert_placement_given=mode==1;
        if(mode==2)cfg.given["--draft-bank-expert-placement"]="command line";
        REQUIRE(draft_placement_modes_conflict(cfg));
        Engine engine(cfg);CHECK_EQ(engine.start(),RAD_E_INVAL);
        CHECK_EQ(engine.config().host_pool_mib,int64_t(1234));
        CHECK_EQ(engine.config().n_spec,cfg.n_spec);CHECK(engine.config().model==cfg.model);
    }
}
TEST(dspark_single_manual_mode_preserves_distinct_budget_contracts) {
    Config cfg;CHECK(!draft_placement_modes_conflict(cfg));
    cfg.draft_bank_expert_placement="host";CHECK(!draft_placement_modes_conflict(cfg));
    cfg.draft_bank_expert_placement="auto";cfg.draft_expert_placement="host";
    cfg.draft_expert_placement_given=true;CHECK(!draft_placement_modes_conflict(cfg));
}
RAD_TEST_MAIN()
