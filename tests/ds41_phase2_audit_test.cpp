/* Prepared host/source-only cases. No device, model, executor, benchmark or routing replay. */
#include "rad_test.h"
#include "place/ds41_phase2_audit.h"
#include "place/card_expert_stager.h"
#include "sched/batch.h"
#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>
namespace a=rad::ds41_phase2::audit;
namespace r=rad::ds41_phase2::routes;
namespace {
struct File {FILE*p=std::tmpfile();File(){if(!p)throw std::runtime_error("audit fixture tmpfile failed");}
    ~File(){if(p)std::fclose(p);}std::string text(){
    std::fflush(p);std::rewind(p);std::string s;char b[4096];size_t n;
    while((n=std::fread(b,1,sizeof b,p)))s.append(b,n);return s;
}};
struct Raw {
    std::vector<int32_t> ids{0,9,100,101,102,103,9,104,105,106,107,108};
    std::vector<int32_t> sorted{0,1,6,-71,-71,-71,-71,-71,-71,-71,-71,-71};
    std::array<int32_t,96> counts{};std::array<int32_t,97> offsets{};
    Raw(){counts[0]=1;counts[9]=2;prefix();}
    void prefix(){for(int e=0;e<96;++e)offsets[e+1]=offsets[e]+counts[e];}
    r::View view()const{return {ids.data(),ids.size(),sorted.data(),sorted.size(),counts.data(),96,offsets.data(),97};}
};
a::Identity id(int step=1,int begin=0,int end=4,uint64_t seq=0){
    a::Identity i;i.valid=true;i.step=step;i.card=3;i.sequence=seq;i.begin=begin;i.rows=2;i.prompt_end=end;return i;
}
a::UnitMap map(){a::UnitMap m;for(int e=0;e<96;++e){m[e].unit=e;m[e].tier=3;m[e].native_bytes=18800640;m[e].residence=r::Residence::Nonresident;}
    m[0].tier=0;m[0].residence=r::Residence::Resident;return m;}
a::Copy copy(int64_t gu=8,int64_t dn=4,bool packed=false){a::Copy c;c.expert=9;c.unit=9;c.rows=2;c.slot=0;c.kind=0;c.gu_bytes=gu;c.dn_bytes=dn;c.packed=packed;return c;}
void piece(a::Trace&t,int step,int begin,int end,uint64_t epoch,const a::Copy&c=copy()){
    Raw raw;t.bind(id(step,begin,end));t.begin(step,2,true,nullptr,1);
    t.frame(epoch,epoch*40,0,0,raw.view(),map(),true,96,false);t.copy(c);t.layer_done(0,true);t.end(true);
}
}
TEST(phase2_audit_pairs_real_generations_seq_zero_plan_and_copy_units){
    File f;CHECK(f.p);a::Trace t(f.p,0,100,200);
    piece(t,11,0,4,101);piece(t,13,2,4,106);CHECK(!t.failed());
    CHECK_EQ(t.totals().copies,2u);CHECK_EQ(t.totals().copy_bytes,24u);
    CHECK_EQ(t.totals().duplicate_same_representation_bytes,12u);
    CHECK_EQ(t.totals().union_min_observed_bytes,12u);CHECK_EQ(t.totals().union_max_observed_bytes,12u);
    CHECK_EQ(t.totals().selected_nonresident,2u);CHECK_EQ(t.totals().unique_selected_nonresident,1u);
    const auto s=f.text();CHECK(s.find("\"seq\":0")!=std::string::npos);CHECK(s.find("\"epoch\":101")!=std::string::npos);
    CHECK(s.find("\"epoch\":106")!=std::string::npos);CHECK(s.find("\"span_epoch\":101")!=std::string::npos);
    CHECK(s.find("\"sorted\":[0,1,6,-71")!=std::string::npos);CHECK(s.find("\"saving_claim\":false")!=std::string::npos);
}
TEST(phase2_audit_separates_mixed_formats_and_repeated_actual_layouts){
    File f;CHECK(f.p);a::Trace t(f.p,0,100,200);
    piece(t,1,0,6,1,copy(8,4));piece(t,2,2,6,2,copy(6,4,true));piece(t,3,4,6,3,copy(8,4));
    CHECK(!t.failed());CHECK_EQ(t.totals().copy_bytes,34u);CHECK_EQ(t.totals().duplicate_same_representation_bytes,12u);
    CHECK_EQ(t.totals().union_min_observed_bytes,10u);CHECK_EQ(t.totals().union_max_observed_bytes,12u);
    CHECK_EQ(t.totals().mixed_representation_units,1u);
}
TEST(phase2_audit_rejects_stale_generation_unit_alias_duplicate_and_missing_copy){
    {File f;a::Trace t(f.p,0,1,2);piece(t,1,0,4,7);Raw raw;t.bind(id(2,2));t.begin(2,2,true,nullptr,1);
     t.frame(7,99,0,0,raw.view(),map(),true,0,true);CHECK(t.failed());}
    for(int fault=0;fault<4;++fault){File f;a::Trace t(f.p,0,1,2);Raw raw;t.bind(id(1,0,2));t.begin(1,2,true,nullptr,1);
        auto m=map();if(fault==0)m[9].unit=0;
        t.frame(1,1,0,0,raw.view(),m,true,0,true);
        if(fault==1){auto c=copy();c.unit=10;t.copy(c);}
        if(fault==2){t.copy(copy());t.copy(copy());}
        if(fault==3)t.layer_done(0,true);
        CHECK(t.failed());CHECK(f.text().find("\"type\":\"incomplete\"")!=std::string::npos);
    }
}
TEST(phase2_audit_freezes_reused_readback_and_keeps_cpu_foreign_and_key_only_distinct){
    File f;CHECK(f.p);a::Trace t(f.p,0,1,2);Raw raw;
    raw.ids={100,101,102,103,104,105,-1,-1,-1,-1,-1,-1};raw.counts={};raw.offsets={};
    int32_t cuts[40]={};cuts[1]=2;
    t.bind(id(1,0,2));t.begin(1,2,true,cuts,2);
    t.frame(9,3,0,0,raw.view(),map(),false,48,true);
    std::fill(raw.ids.begin(),raw.ids.end(),7);std::fill(raw.sorted.begin(),raw.sorted.end(),99);
    t.end(true);CHECK(!t.failed());CHECK_EQ(t.totals().copies,0u);
    const auto s=f.text();CHECK(s.find("\"ids\":[100,101,102,103,104,105,-1")!=std::string::npos);
    CHECK(s.find("\"p1\":false")!=std::string::npos);CHECK(s.find("\"reason\":\"key_only\"")!=std::string::npos);
}
TEST(phase2_audit_unbound_short_cancel_and_caps_never_invent_complete_frames){
    {File f;a::Trace t(f.p,0,1,2);piece(t,1,0,3,1);auto i=id(2,2,3);i.rows=1;t.bind(i);t.begin(2,1,false,nullptr,40);
     CHECK(!t.failed());const auto s=f.text();CHECK(s.find("\"whole_prompt_covered\":false")!=std::string::npos);
     CHECK(s.find("\"reason\":\"non_p1_pass\"")!=std::string::npos);}
    {File f;a::Trace t(f.p,0,1,2);t.begin(12,4096,true,nullptr,40);CHECK(!t.collecting());CHECK(t.failed());
     CHECK(f.text().find("unbound_p1_identity")!=std::string::npos);}
    {File f;a::Trace t(f.p,0,1,2);Raw raw;t.bind(id(1,0,2));t.begin(1,2,true,nullptr,1);
     t.frame(1,1,0,0,raw.view(),map(),true,0,true);t.copy(copy());t.layer_done(0,true);t.end(false);CHECK(t.failed());}
    {File f;a::Trace t(f.p,0,1,2);for(int n=0;n<a::kMaxSpans;++n){auto i=id(n,0,2,uint64_t(n));t.bind(i);t.begin(n,2,true,nullptr,1);
      Raw raw;t.frame(n+1,n+1,0,0,raw.view(),map(),true,0,true);t.copy(copy());t.layer_done(0,true);t.end(true);}
     CHECK(!t.failed());t.bind(id(30,0,2,30));t.begin(30,2,true,nullptr,1);CHECK(t.failed());}
}
TEST(phase2_audit_prefix_validation_and_all_four_ranks_bind_their_actual_experts){
    CHECK(a::prefix_valid(nullptr));CHECK(a::prefix_valid("0"));CHECK(a::prefix_valid("/tmp/fresh-p2"));
    CHECK(!a::prefix_valid("relative"));CHECK(!a::prefix_valid("/tmp/bad\npath"));CHECK(!a::prefix_valid("/tmp/\"bad"));
    for(int rank=0;rank<4;++rank){File f;a::Trace t(f.p,rank,100+rank,200+rank);Raw raw;
        for(auto&e:raw.ids)if(e>=0)e=(e+rank*96)%384;
        t.bind(id(1,0,2));t.begin(1,2,true,nullptr,1);t.frame(1,1,0,0,raw.view(),map(),true,0,true);
        t.copy(copy());t.layer_done(0,true);t.end(true);CHECK(!t.failed());
    }
}
TEST(phase2_audit_actual_host_plan_identity_rejects_ahead_decode_mixed_dev_and_stale_seq){
    RadBatch b{};b.step=11;b.phase=RAD_PHASE_PREFILL;b.n_tok=4080;b.n_seq=1;
    rad::StepEntry e;e.seq=0;e.card=3;e.is_prefill=true;e.n_tokens=4080;e.ctx_len=2736;e.prompt_end=6817;
    uint64_t seq=0;
    auto good=a::make_identity(&b,&e,1,&seq,1,false);CHECK(good.valid);CHECK_EQ(good.sequence,0u);
    CHECK_EQ(good.step,11);CHECK_EQ(good.begin,2736);CHECK_EQ(good.rows,4080);CHECK_EQ(good.prompt_end,6817);
    for(int fault=0;fault<11;++fault){auto x=b;auto y=e;uint64_t s=seq;size_t n=1,ns=1;bool gen=false;
        switch(fault){case 0:x.phase=RAD_PHASE_DECODE;break;case 1:x.n_seq_decode=1;break;case 2:y.dev=true;break;
        case 3:s=99;break;case 4:x.draft_pass=1;break;case 5:x.n_spec=1;break;case 6:n=2;break;
        case 7:ns=0;break;case 8:gen=true;break;case 9:y.prompt_end=6815;break;case 10:y.n_tokens=4079;break;}
        CHECK(!a::make_identity(&x,&y,n,&s,ns,gen).valid);
    }
}
TEST(phase2_audit_byte_frame_and_complete_footer_caps_are_fail_closed){
    {File f;a::Limits limits;limits.bytes=4096;a::Trace t(f.p,0,1,2,limits);Raw raw;
     t.bind(id(1,0,2));t.begin(1,2,true,nullptr,1);t.frame(3,9,0,0,raw.view(),map(),true,0,true);
     CHECK(t.failed());t.close();const auto s=f.text();CHECK(s.find("output_cap")!=std::string::npos);
     CHECK(s.find("\"type\":\"audit_end\",\"rank\":0,\"complete\":false")!=std::string::npos);}
    {File f;a::Limits limits;limits.frames=1;a::Trace t(f.p,0,1,2,limits);piece(t,1,0,4,1);
     piece(t,2,2,4,2);CHECK(t.failed());CHECK(f.text().find("frame_cap")!=std::string::npos);}
    {File f;a::Trace t(f.p,0,1,2);piece(t,1,0,2,1);t.close();CHECK(!t.failed());
     CHECK(f.text().find("\"type\":\"audit_end\",\"rank\":0,\"complete\":true")!=std::string::npos);}
}
namespace {
struct ImmediateDevice:rad::p1::Device {
    bool fail=false;int events=0,copies=0;
    int event_create(bool)override{return events++;}
    int copy(int,const rad::p1::Source&,int,int)override{++copies;return fail?-1:0;}
    int compute_wait(int)override{return 0;}int compute_record(int)override{return 0;}
    int query(int)override{return 1;}int sync(int)override{return 0;}
    int wave(int,int,const void*const*,const void*const*,int)override{return 0;}
    void release(rad::p1::Kind,int32_t)override{}int wait_drive()override{return -1;}
};
}
TEST(phase2_audit_copy_observer_sees_only_successful_unchanged_source_enqueues){
    for(bool fail:{false,true}){ImmediateDevice d;d.fail=fail;rad::p1::CardExpertStager st;std::string why;
        CHECK_EQ(st.init(&d,(const char*)uintptr_t(0x10000),2,1,96,(void*)uintptr_t(0x20000),1,&why),0);
        struct Obs{int n=0;int32_t unit=-1;int64_t bytes=0;}obs;
        st.set_copy_observer([](void*p,int e,int32_t u,int rows,int slot,rad::p1::Kind kind,const rad::p1::Source&s){
            auto&o=*static_cast<Obs*>(p);CHECK_EQ(e,9);CHECK_EQ(rows,2);CHECK_EQ(slot,0);CHECK(kind==rad::p1::Kind::Ram);
            ++o.n;o.unit=u;o.bytes=s.gu_bytes+s.dn_bytes;
        },&obs);
        rad::p1::Source src;src.gu=(void*)uintptr_t(0x30000);src.dn=(void*)uintptr_t(0x40000);src.gu_bytes=8;src.dn_bytes=4;
        CHECK_EQ(st.begin_layer(0),0);CHECK_EQ(st.add_ram(9,0,2,src,1),0);
        if(fail){CHECK(st.start()<0);CHECK_EQ(obs.n,0);}else{CHECK_EQ(st.start(),0);CHECK_EQ(st.run(),0);
            CHECK_EQ(obs.n,1);CHECK_EQ(obs.unit,0);CHECK_EQ(obs.bytes,12);CHECK_EQ(st.end_pass(),0);}
        CHECK_EQ(d.copies,1);
    }
}
