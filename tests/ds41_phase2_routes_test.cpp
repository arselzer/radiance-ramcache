/* Prepared host metadata cases only. No device, model, replay/simulation or executor. */
#include "rad_test.h"
#include "place/ds41_phase2_routes.h"
#include <algorithm>
#include <functional>
#include <vector>
using namespace rad::ds41_phase2::routes;
namespace {
struct Raw {
    std::vector<int32_t> ids,sorted;
    std::array<int32_t,kLocalExperts> counts{};
    std::array<int32_t,kLocalExperts+1> offsets{};
    View view()const{return {ids.data(),ids.size(),sorted.data(),sorted.size(),
        counts.data(),counts.size(),offsets.data(),offsets.size()};}
    void prefix(){offsets[0]=0;for(int e=0;e<kLocalExperts;++e)offsets[e+1]=offsets[e]+counts[e];}
};
Raw first() {
    Raw r;r.ids={3,100,0,101,102,103,9,3,104,105,106,107};r.sorted.assign(12,-555);
    r.counts[0]=1;r.counts[3]=2;r.counts[9]=1;r.prefix();
    const int32_t live[]={2,0,7,6};std::copy(live,live+4,r.sorted.begin());return r;
}
Raw second() {
    Raw r;r.ids={9,108,109,110,111,112,5,9,113,114,115,116};r.sorted.assign(12,-777);
    r.counts[5]=1;r.counts[9]=2;r.prefix();
    const int32_t live[]={6,0,7};std::copy(live,live+3,r.sorted.begin());return r;
}
Binding binding(int piece=0,int64_t begin=0,int rows=2,int from=0,int rank=0) {
    Binding b;b.sequence=17;b.span_epoch=23;b.frame_epoch=100+piece;b.plan_cookie=42;b.layer=2;b.rank=rank;b.piece=piece;
    b.begin=begin;b.query_rows=rows;b.row_from=from;b.phase=Phase::Prefill;return b;
}
Scope scope(int pieces=2,int64_t end=4,int rank=0){return {17,23,42,2,rank,pieces,0,end};}
UnitMap mapping(int rank=0) {
    UnitMap m;m.plan_cookie=42;m.layer=2;m.rank=rank;
    for(int e=0;e<kLocalExperts;++e){m.unit[e]=rank*1000+e;m.residence[e]=Residence::Nonresident;}
    return m;
}
}
TEST(phase2_routes_freezes_original_local_sort_without_borrowing_or_touching_dead_tail) {
    Raw r=first();Frame f;const char*why=nullptr;
    REQUIRE(freeze(binding(),r.view(),&f,&why));CHECK(why==nullptr);CHECK(f.valid());
    CHECK_EQ(f.rows(),2);CHECK_EQ(f.slots(),12);CHECK_EQ(f.live_slots(),4);
    CHECK_EQ(f.counts()[0],1);CHECK_EQ(f.counts()[3],2);CHECK_EQ(f.offsets()[9],3);
    CHECK_EQ(f.sorted()[0],2);CHECK_EQ(f.sorted()[1],0);CHECK_EQ(f.sorted()[2],7);CHECK_EQ(f.sorted()[11],-555);
    // These are the buffers another piece/router/readback would overwrite.
    std::fill(r.ids.begin(),r.ids.end(),383);std::fill(r.sorted.begin(),r.sorted.end(),0);
    r.counts.fill(0);r.offsets.fill(0);
    CHECK_EQ(f.ids()[0],3);CHECK_EQ(f.ids()[2],0);CHECK_EQ(f.sorted()[11],-555);CHECK_EQ(f.counts()[3],2);
    Frame copied=f;CHECK_EQ(copied.binding().frame_epoch,uint64_t(100));CHECK_EQ(copied.sorted()[2],7);
}
TEST(phase2_routes_union_deduplicates_expert_units_but_keeps_each_original_piece_range) {
    std::array<Frame,3> f;Raw a=first(),b=second(),cut;
    REQUIRE(freeze(binding(0,0),a.view(),&f[0]));REQUIRE(freeze(binding(1,2),b.view(),&f[1]));
    REQUIRE(freeze(binding(2,4,16,16),cut.view(),&f[2]));
    auto m=mapping();m.residence[3]=Residence::Resident;Union u;
    REQUIRE(make_union(scope(3,20),f.data(),f.size(),m,&u));
    REQUIRE_EQ(u.size,4);CHECK_EQ(u.expert_occurrences,5);CHECK_EQ(u.duplicate_occurrences,1);
    CHECK_EQ(u.nonresident_occurrences,4);CHECK_EQ(u.unique_nonresident,3);CHECK_EQ(u.duplicate_nonresident,1);
    CHECK_EQ(u.local_rows,int64_t(7));CHECK_EQ(u.frame_epochs[0],uint64_t(100));CHECK_EQ(u.frame_epochs[1],uint64_t(101));
    CHECK_EQ(u.entries[0].global_expert,0);CHECK_EQ(u.entries[0].unit,0); // unit0/expert0 are valid
    CHECK_EQ(u.entries[1].local_expert,3);CHECK(u.entries[1].residence==Residence::Resident);
    CHECK_EQ(u.entries[1].original_ranges[0].lo,1);CHECK_EQ(u.entries[1].original_ranges[0].hi,3);
    CHECK_EQ(u.entries[1].original_ranges[1].lo,0);CHECK_EQ(u.entries[1].original_ranges[1].hi,0);
    const auto&nine=u.entries[3];CHECK_EQ(nine.global_expert,9);CHECK_EQ(nine.rows,int64_t(3));
    CHECK_EQ(nine.pieces_mask,uint8_t(3));CHECK_EQ(nine.original_ranges[0].lo,3);CHECK_EQ(nine.original_ranges[0].hi,4);
    CHECK_EQ(nine.original_ranges[1].lo,1);CHECK_EQ(nine.original_ranges[1].hi,3);
    CHECK_EQ(nine.original_ranges[2].lo,0);CHECK_EQ(nine.original_ranges[2].hi,0);
    // No giant sort/output row is built: both saved local permutations remain byte-identical.
    for(int j=0;j<12;++j){CHECK_EQ(f[0].sorted()[j],a.sorted[j]);CHECK_EQ(f[1].sorted()[j],b.sorted[j]);}
}
TEST(phase2_routes_refuses_bad_topk_counts_offsets_or_binding_before_replacing_a_snapshot) {
    const std::vector<std::function<void(Raw&)>> bad={
        [](Raw&r){r.ids[0]=384;},[](Raw&r){r.ids[0]=-1;},[](Raw&r){r.ids[0]=-2;},[](Raw&r){r.ids[1]=r.ids[0];},
        [](Raw&r){r.counts[3]=-1;},[](Raw&r){r.counts[3]=12;},[](Raw&r){r.offsets[0]=1;},
        [](Raw&r){++r.offsets[4];},[](Raw&r){r.sorted[0]=12;},[](Raw&r){r.sorted[0]=-1;},
        [](Raw&r){r.sorted[2]=r.sorted[1];},[](Raw&r){r.sorted[0]=6;},
        [](Raw&r){r.counts.fill(0);r.offsets.fill(0);},[](Raw&r){r.sorted.pop_back();},[](Raw&r){r.ids.pop_back();}
    };
    Frame old;Raw good=first();REQUIRE(freeze(binding(),good.view(),&old));
    for(const auto&mutate:bad){Raw r=first();mutate(r);const char*why=nullptr;
        CHECK(!freeze(binding(),r.view(),&old,&why));CHECK(why!=nullptr);
        CHECK(old.valid());CHECK_EQ(old.ids()[2],0);CHECK_EQ(old.sorted()[2],7);CHECK_EQ(old.binding().span_epoch,uint64_t(23));}
    const std::vector<std::function<void(Binding&)>> bad_key={
        [](Binding&b){b.span_epoch=0;},[](Binding&b){b.frame_epoch=0;},[](Binding&b){b.plan_cookie=0;},[](Binding&b){b.layer=40;},
        [](Binding&b){b.rank=4;},[](Binding&b){b.piece=4;},[](Binding&b){b.begin=-1;},
        [](Binding&b){b.begin=INT32_MAX;},[](Binding&b){b.query_rows=4097;},
        [](Binding&b){b.row_from=3;},[](Binding&b){b.phase=Phase::Decode;},[](Binding&b){b.phase=Phase::Draft;}
    };
    for(const auto&mutate:bad_key){auto b=binding();mutate(b);CHECK(!freeze(b,good.view(),&old));CHECK_EQ(old.ids()[0],3);}
    auto zero=binding();zero.sequence=0;CHECK(freeze(zero,good.view(),&old));CHECK_EQ(old.binding().sequence,uint64_t(0));
    auto view=good.view();view.counts_size=95;CHECK(!freeze(binding(),view,&old));
    view=good.view();view.offsets=nullptr;CHECK(!freeze(binding(),view,&old));
}
TEST(phase2_routes_span_binding_and_selected_unit_alias_refusals_are_atomic) {
    const std::vector<std::function<void(Binding&)>> other={
        [](Binding&b){++b.sequence;},[](Binding&b){++b.span_epoch;},[](Binding&b){b.frame_epoch=99;},[](Binding&b){++b.plan_cookie;},
        [](Binding&b){++b.layer;},[](Binding&b){++b.piece;},[](Binding&b){++b.begin;}
    };
    Raw a=first(),b=second();std::array<Frame,2> f;REQUIRE(freeze(binding(),a.view(),&f[0]));
    Union u;u.size=77;auto m=mapping();
    for(const auto&mutate:other){auto key=binding(1,2);mutate(key);REQUIRE(freeze(key,b.view(),&f[1]));
        CHECK(!make_union(scope(),f.data(),f.size(),m,&u));CHECK_EQ(u.size,77);}
    auto same=binding(1,2);same.frame_epoch=100;REQUIRE(freeze(same,b.view(),&f[1]));
    CHECK(make_union(scope(),f.data(),f.size(),m,&u));CHECK_EQ(u.frame_epochs[1],uint64_t(100));
    u.size=77;REQUIRE(freeze(binding(1,2),b.view(),&f[1]));
    auto s=scope();++s.rank;CHECK(!make_union(s,f.data(),f.size(),m,&u));
    s=scope();++s.end;CHECK(!make_union(s,f.data(),f.size(),m,&u));
    s=scope();s.end=8193;CHECK(!make_union(s,f.data(),f.size(),m,&u));
    CHECK(!make_union(scope(),f.data(),1,m,&u));
    for(int kind=0;kind<5;++kind){auto broken=m;
        if(kind==0)broken.unit[9]=broken.unit[3];if(kind==1)broken.unit[9]=-1;
        if(kind==2)broken.residence[9]=Residence::Unknown;if(kind==3)++broken.layer;if(kind==4)++broken.plan_cookie;
        CHECK(!make_union(scope(),f.data(),f.size(),broken,&u));CHECK_EQ(u.size,77);}
}
TEST(phase2_routes_foreign_only_and_key_only_pieces_need_no_fake_local_expert) {
    Raw foreign;foreign.ids={-1,-1,98,99,100,101};foreign.sorted={INT32_MIN,-99,99999,0,42,17};
    Raw cut;std::array<Frame,2> f;REQUIRE(freeze(binding(0,0,1),foreign.view(),&f[0]));
    REQUIRE(freeze(binding(1,1,16,16),cut.view(),&f[1]));
    auto m=mapping();m.unit.fill(-1);m.residence.fill(Residence::Unknown);Union u;
    REQUIRE(make_union(scope(2,17),f.data(),f.size(),m,&u));CHECK_EQ(u.size,0);
    CHECK_EQ(u.local_rows,int64_t(0));CHECK_EQ(u.expert_occurrences,0);CHECK_EQ(u.duplicate_nonresident,0);
    CHECK_EQ(f[0].sorted()[0],INT32_MIN); // dead sorted tail is not an expert row
}
TEST(phase2_routes_8k_bound_all96_experts_and_all_logical_ranks_keep_piece_shapes) {
    Raw r;r.ids.resize(kMaxSlots);r.sorted.resize(kMaxSlots);
    for(int e=0;e<kLocalExperts;++e){r.counts[e]=256;r.offsets[e]=e*256;
        for(int j=0;j<256;++j)r.sorted[e*256+j]=j*96+e;}r.offsets[96]=kMaxSlots;
    std::array<Frame,2> f;
    for(int rank=0;rank<4;++rank){for(int j=0;j<kMaxSlots;++j)r.ids[j]=rank*96+j%96;
        REQUIRE(freeze(binding(0,0,4096,0,rank),r.view(),&f[0]));
        REQUIRE(freeze(binding(1,4096,4096,0,rank),r.view(),&f[1]));
        Union u;auto m=mapping(rank);REQUIRE(make_union(scope(2,8192,rank),f.data(),f.size(),m,&u));
        CHECK_EQ(u.size,96);CHECK_EQ(u.expert_occurrences,192);CHECK_EQ(u.duplicate_nonresident,96);
        CHECK_EQ(u.local_rows,int64_t(49152));CHECK_EQ(u.entries[95].global_expert,rank*96+95);
        CHECK_EQ(u.entries[95].rows,int64_t(512));CHECK_EQ(u.entries[95].original_ranges[1].hi,kMaxSlots);}
    int64_t bytes=0;REQUIRE(metadata_bytes(4,&bytes));CHECK(bytes>=int64_t(5)*int64_t(sizeof(Frame)));
    const auto old=bytes;CHECK(!metadata_bytes(5,&bytes));CHECK_EQ(bytes,old);
}
RAD_TEST_MAIN()
