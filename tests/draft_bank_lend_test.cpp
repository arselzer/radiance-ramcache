#include "rad_test.h"
#include "place/draft_bank_lend.h"
using namespace rad::draft_bank;
namespace {
Config enabled(int64_t grace=10,int64_t cooldown=20){Config c;c.on=true;c.backup_cap=8ll<<30;c.grace_ms=grace;c.cooldown_ms=cooldown;return c;}
Work held(uint64_t epoch=1){Work w;w.step=(int64_t)epoch-1;w.epoch=epoch;w.deferred=true;w.solo_held=true;return w;}
}
TEST(bank_lend_configuration_is_explicit_and_disabled_path_ignores_tunables) {
    Config c;
    CHECK_OK(Config::parse(nullptr,"garbage","bad","-1",&c));CHECK(!c.on);
    CHECK_OK(Config::parse("0","garbage","bad","-1",&c));CHECK(!c.on);
    for(const char* s:{"","2","true","-1"})CHECK_EQ(Config::parse(s,"6000",nullptr,nullptr,&c),RAD_E_INVAL);
    for(const char* b:{"","0","-1","1x","99999999999999999999999999"})CHECK_EQ(Config::parse("1",b,nullptr,nullptr,&c),RAD_E_INVAL);
    CHECK_OK(Config::parse("1","6144","0","600000",&c));CHECK(c.on);CHECK_EQ(c.backup_cap,6144ll<<20);
    CHECK_EQ(Config::parse("1","6144","600001",nullptr,&c),RAD_E_INVAL);
    Controller off;Action a;CHECK_OK(off.decide(held(),1000,&a));CHECK(a==Action::None);CHECK(off.ready());
}
TEST(bank_lend_requires_existing_solo_hold_and_zero_context_query_work) {
    for(int condition=0;condition<4;++condition){Controller c(enabled(0,0));Work w=held();
        if(condition==0)w.deferred=false;if(condition==1)w.solo_held=false;
        if(condition==2)w.context_rows=1;if(condition==3){w.query_rows=2;w.query_seq=42;}
        Action a;CHECK_OK(c.decide(w,100,&a));CHECK(a==Action::None);CHECK(c.ready());}
    Work bad=held();bad.query_seq=42;Controller c(enabled(0,0));Action a;
    CHECK_EQ(c.decide(bad,100,&a),RAD_E_INVAL);bad.query_seq=0;bad.query_rows=2;CHECK_EQ(c.decide(bad,100,&a),RAD_E_INVAL);
}
TEST(bank_lend_dwell_cooldown_duplicate_epoch_and_double_transition) {
    Controller c(enabled());Action a;Work w=held();
    CHECK_OK(c.decide(w,100,&a));CHECK(a==Action::None);
    CHECK_OK(c.decide(w,200,&a));CHECK(a==Action::None); // duplicate never advances a dwell decision
    w=held(2);CHECK_OK(c.decide(w,109,&a));CHECK(a==Action::None);
    w=held(3);CHECK_OK(c.decide(w,110,&a));CHECK(a==Action::Lend);
    CHECK_EQ(c.decide(held(4),111,&a),RAD_E_STATE);
    CHECK_EQ(c.finish(Action::Lend,2,true,111),RAD_E_STATE);
    CHECK_OK(c.finish(Action::Lend,3,true,111));CHECK(c.lent());
    CHECK_EQ(c.finish(Action::Lend,3,true,111),RAD_E_STATE);
    CHECK_OK(c.decide(held(4),112,&a));CHECK(a==Action::None);
    w=held(5);w.solo_held=false;w.query_rows=2;w.query_seq=77;
    CHECK_OK(c.decide(w,113,&a));CHECK(a==Action::Restore); // no grace on restore
    CHECK_OK(c.finish(Action::Restore,5,true,120));CHECK(c.ready());
    CHECK_OK(c.decide(held(6),121,&a));CHECK(a==Action::None);
    CHECK_OK(c.decide(held(7),139,&a));CHECK(a==Action::None);
    CHECK_OK(c.decide(held(8),140,&a));CHECK(a==Action::Lend);
}
TEST(bank_lend_cancel_end_idle_clear_dwell_and_restore_before_reuse) {
    Controller c(enabled(10,0));Action a;
    CHECK_OK(c.decide(held(1),100,&a));CHECK(a==Action::None);
    CHECK_OK(c.restore_idle(0,&a));CHECK(a==Action::None);
    CHECK_OK(c.decide(held(2),200,&a));CHECK(a==Action::None); // idle did not leave old dwell armed
    CHECK_OK(c.decide(held(3),210,&a));CHECK(a==Action::Lend);
    CHECK_OK(c.finish(Action::Lend,3,true,211));
    CHECK_OK(c.restore_idle(3,&a));CHECK(a==Action::Restore);
    CHECK_EQ(c.restore_idle(3,&a),RAD_E_STATE);
    CHECK_OK(c.finish(Action::Restore,3,true,212));CHECK(c.ready());
    CHECK_OK(c.decide(held(4),213,&a));CHECK(a==Action::None);
}
TEST(bank_lend_failed_fill_requires_verified_rollback_and_failed_restore_latches) {
    Controller c(enabled(0,0));Action a;
    CHECK_OK(c.decide(held(1),100,&a));CHECK(a==Action::Lend);
    CHECK_OK(c.finish(Action::Lend,1,false,101));CHECK(c.ready()); // caller already restored+memcmp
    CHECK_OK(c.decide(held(2),102,&a));CHECK(a==Action::Lend);
    CHECK_OK(c.finish(Action::Lend,2,true,103));
    Work w=held(3);w.solo_held=false;
    CHECK_OK(c.decide(w,104,&a));CHECK(a==Action::Restore);
    CHECK_EQ(c.finish(Action::Restore,3,false,105),RAD_E_STATE);CHECK(c.phase()==Phase::Failed);
    CHECK_EQ(c.decide(held(4),106,&a),RAD_E_STATE);
    CHECK_EQ(c.restore_idle(4,&a),RAD_E_STATE);
    Controller copied(enabled(0,0));CHECK_OK(copied.decide(held(),100,&a));copied.fatal();
    CHECK_EQ(copied.finish(Action::Lend,1,true,101),RAD_E_STATE);
}
TEST(bank_lend_rejects_stale_epoch_or_clock_before_overwriting_another_frame) {
    Controller c(enabled(10,0));Action a;
    CHECK_OK(c.decide(held(4),100,&a));
    CHECK_EQ(c.decide(held(3),101,&a),RAD_E_STATE);
    CHECK_EQ(c.decide(held(5),99,&a),RAD_E_STATE);
}
TEST(bank_lend_restore_is_required_when_query_cap_or_cancellation_leaves_zero_rows) {
    Controller c(enabled(0,0));Action a;
    CHECK_OK(c.decide(held(),100,&a));CHECK_OK(c.finish(Action::Lend,1,true,101));
    Work capped=held(2);capped.solo_held=false; // no current query, but bank becomes eligible again
    CHECK_EQ(capped.context_rows,0);CHECK_EQ(capped.query_rows,0);
    CHECK_OK(c.decide(capped,102,&a));CHECK(a==Action::Restore);
    CHECK_OK(c.finish(Action::Restore,2,true,103));CHECK(c.ready());
}
RAD_TEST_MAIN()
