#include "rad_test.h"
#include "place/draft_bank_packed_fill.h"
#include "place/draft_bank_lend.h"
#include "../tools/ds41-host-experts/ds41_host_experts.h"
#include "../libquant/lq_ggml_planes.h"
#include <array>
#include <algorithm>
#include <cstring>
#include <string>
#include <stdexcept>
#include <vector>
using namespace rad;
namespace pf = rad::draft_bank_packed_fill;
namespace {
void need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
struct Environment {
    const char* name = "RADIANCE_EXPERIMENTAL_NATIVE_CACHE_MAPPED_SLABS";
    bool present = false;
    std::string old;
    Environment() { const char* s = std::getenv(name); present = s; old = s ? s : ""; ::setenv(name,"1",1); }
    ~Environment() { if (present) ::setenv(name,old.c_str(),1); else ::unsetenv(name); }
};
struct Epoch {
    int calls = 0, fail_on = 0;
    static bool check(void* p) noexcept { auto& e = *static_cast<Epoch*>(p); return ++e.calls != e.fail_on; }
};
struct Fixture {
    Environment env;
    Program p;
    Plan plan;
    NativeMappedFileCache cache;
    Mover::HostUnit h;
    std::vector<uint8_t> raw = std::vector<uint8_t>(size_t(kDraftPairBytes));
    std::vector<uint8_t> relay = std::vector<uint8_t>(size_t(kDraftPairBytes)+64,0xa5);
    Epoch epoch;
    pf::Workspace work;
    Fixture() {
        p.weights.resize(3); plan.units.resize(1);
        auto& m = plan.units[0]; m.layer=7; m.expert=13; m.tier=Tier::SSD; m.slab_class=0;
        m.bytes=m.slot_bytes=kDraftPairBytes; m.weights={1,2}; m.offset={0,12533760};
        const ds41_mapped::SlabCache::Sizes sizes{ds41hx::packed_bytes(4608,5120,512),ds41hx::packed_bytes(5120,2304,320)};
        need(cache.configure(22u<<20,4,0,0,NativeMappedFileCache::Layout::Packed,sizes)==RAD_OK,"real packed cache configuration");
        const auto pack=lq::ggml::pack_of(lq::ggml::GGML_TYPE_MXFP4); need(pack,"original BLOCK relayout");
        for(int k=0;k<2;++k) {
            auto& w=p.weights[size_t(k)+1]; auto& d=w.decl;
            const int n=k?5120:4608, cols=k?2304:5120, blocks=cols/32;
            d.group.layer=7; d.group.expert=13; d.group.slot=k; d.rank=2;
            d.shape[0]=n; d.shape[1]=cols; d.shard=RAD_SHARD_NONE; d.access=RAD_ACCESS_CONDITIONAL;
            w.layout_tag="ggml_mxfp4"; w.stored_bytes=k?6266880:12533760; w.source_bank_generation=123;
            w.stored_dtype=RAD_U8;w.stored_rank=2;w.stored_shape[0]=n;w.stored_shape[1]=blocks*17;
            std::vector<uint8_t> codes(size_t(n)*size_t(cols/2)),scales(size_t(n)*size_t(blocks));
            for(size_t i=0;i<codes.size();++i)codes[i]=uint8_t(i*131+i/17+k);
            for(size_t i=0;i<scales.size();++i)scales[i]=uint8_t(120+i%4);
            scales.front()=0;scales.back()=255;
            ds41hx::PackInfo info; need(ds41hx::pack_plan(scales.data(),n,cols,&info),"packed scale plan");
            NativeMappedFileCache::Write wr;
            need(cache.reserve_packed(size_t(k)+1,k,&wr)==NativeMappedFileCache::Status::Ok,"real paired cache reservation");
            need(ds41hx::pack_planes(codes.data(),scales.data(),n,cols,k?320:512,info,wr.host(),wr.bytes())==ds41hx::kPackOk,"pack immutable native matrix");
            need(wr.publish(&h.lease[k])==NativeMappedFileCache::Status::Ok,"publish real packed lease");
            h.host[k]=h.lease[k].host();h.dev[k]=h.lease[k].device()?h.lease[k].device():h.host[k];
            h.bytes[k]=w.stored_bytes;h.slot[k]=int64_t(h.lease[k].bytes());
            for(int row=0;row<n;++row){const uint8_t* planes[2]={codes.data()+size_t(row)*size_t(cols/2),scales.data()+size_t(row)*size_t(blocks)};
                pack(planes,blocks,raw.data()+m.offset[size_t(k)]+size_t(row)*size_t(blocks)*17);}
        }
        h.unit=0;h.n=2;h.packed=true;
    }
    ~Fixture(){for(auto& l:h.lease)(void)l.release();}
    int prepare(pf::Inverse inverse=ds41_packed_inverse::to_block) {
        epoch.calls=0;
        return pf::prepare(p,plan,0,h,relay.data(),size_t(kDraftPairBytes),&work,Epoch::check,&epoch,inverse);
    }
    void untouched(){CHECK(std::all_of(relay.begin(),relay.end(),[](uint8_t x){return x==uint8_t{0xa5};}));}
    void held(){CHECK(h.lease[0].active());CHECK(h.lease[1].active());}
};
int inverse_calls=0,fail_inverse_on=0;
int inverse_fault(const uint8_t* p,size_t bytes,int n,int k,uint8_t* out,size_t cap,ds41_packed_inverse::Stats* stats) noexcept {
    if(++inverse_calls==fail_inverse_on)return RAD_E_STATE;
    return ds41_packed_inverse::to_block(p,bytes,n,k,out,cap,stats);
}
}
TEST(bank_packed_fill_flag_is_strict_explicit_and_requires_lending) {
    draft_bank::Config c;
    for(const char* f:std::array<const char*,2>{nullptr,"0"}) {
        CHECK_OK(draft_bank::Config::parse(nullptr,"bad","bad","bad",&c,f)); CHECK(!c.on); CHECK(!c.packed_fill);
        CHECK_OK(draft_bank::Config::parse("1","8192",nullptr,nullptr,&c,f)); CHECK(c.on); CHECK(!c.packed_fill);
    }
    for(const char* f:{"","2","true","-1"," 1","01"})
        CHECK_EQ(draft_bank::Config::parse("1","8192",nullptr,nullptr,&c,f),RAD_E_INVAL);
    for(const char* f:std::array<const char*,2>{nullptr,"0"})
        CHECK_EQ(draft_bank::Config::parse(f,"8192",nullptr,nullptr,&c,"1"),RAD_E_INVAL);
    CHECK_OK(draft_bank::Config::parse("1","8192",nullptr,nullptr,&c,"1"));CHECK(c.on);CHECK(c.packed_fill);
}
TEST(bank_packed_fill_actual_leases_reconstruct_every_native_byte_and_keep_guards) {
    Fixture f; REQUIRE_EQ(f.prepare(),RAD_OK); f.held();
    CHECK_EQ(std::memcmp(f.relay.data(),f.raw.data(),f.raw.size()),0);
    for(size_t i=f.raw.size();i<f.relay.size();++i)CHECK_EQ(f.relay[i],uint8_t{0xa5});
    CHECK_EQ(f.work.raw_bytes,uint64_t(kDraftPairBytes));
    CHECK_EQ(f.work.source_bytes,uint64_t(f.h.slot[0]+f.h.slot[1]));
    CHECK_EQ(f.work.inverse[0].raw_rounds,uint64_t{2});CHECK_EQ(f.work.inverse[1].raw_rounds,uint64_t{2});
    CHECK_EQ(f.epoch.calls,2); CHECK(f.work.refusal==pf::Refusal::None);
}
TEST(bank_packed_fill_refuses_wrong_plan_shapes_offsets_and_source_bank) {
    Fixture f;
    for(int fault=0;fault<15;++fault) {
        const auto m=f.plan.units[0]; const auto w0=f.p.weights[1],w1=f.p.weights[2];
        auto& unit=f.plan.units[0];auto& gu=f.p.weights[1];auto& dn=f.p.weights[2];
        switch(fault){case 0:unit.offset[1]++;break;case 1:unit.slot_bytes--;break;
        case 2:unit.weights[1]=unit.weights[0];break;case 3:gu.decl.shape[0]--;break;
        case 4:dn.decl.group.slot=0;break;case 5:dn.stored_bytes--;break;
        case 6:gu.vram_bytes=64;break;case 7:gu.widen=true;break;
        case 8:unit.draft_parked=true;break;case 9:gu.file_id=1;break;
        case 10:dn.source_bank_generation++;break;case 11:unit.layer=40;break;
        case 12:gu.stored_dtype=RAD_BF16;break;case 13:dn.stored_shape[1]++;break;case 14:unit.movable=true;break;}
        CHECK(f.prepare()<0);f.untouched();f.held();
        f.plan.units[0]=m;f.p.weights[1]=w0;f.p.weights[2]=w1;
    }
}
TEST(bank_packed_fill_refuses_wrong_lease_metadata_before_either_inverse) {
    Fixture f;
    for(int fault=0;fault<8;++fault){const auto* host=f.h.host[1];const auto* dev=f.h.dev[1];
        const auto bytes=f.h.bytes[1],slot=f.h.slot[1];
        switch(fault){case 0:f.h.host[1]=nullptr;break;case 1:f.h.host[1]++;break;
        case 2:f.h.dev[1]=nullptr;break;case 3:f.h.bytes[1]--;break;case 4:f.h.slot[1]--;break;
        case 5:f.h.n=1;break;case 6:f.h.packed=false;break;case 7:f.h.unit=1;break;}
        inverse_calls=0;fail_inverse_on=0;CHECK(f.prepare(inverse_fault)<0);CHECK_EQ(inverse_calls,0);
        f.untouched();f.held();f.h.host[1]=host;f.h.dev[1]=dev;f.h.bytes[1]=bytes;f.h.slot[1]=slot;f.h.n=2;f.h.packed=true;f.h.unit=0;
    }
    REQUIRE(f.h.lease[1].release()==NativeMappedFileCache::Status::Ok);
    CHECK(f.prepare()<0);f.untouched();CHECK(f.h.lease[0].active());
}
TEST(bank_packed_fill_second_inverse_fault_is_not_a_complete_pair_and_keeps_leases) {
    Fixture f;inverse_calls=0;fail_inverse_on=2;
    CHECK(f.prepare(inverse_fault)<0);CHECK_EQ(inverse_calls,2);CHECK(f.work.refusal==pf::Refusal::Inverse);
    CHECK_EQ(std::memcmp(f.relay.data(),f.raw.data(),12533760),0);
    CHECK_EQ(f.work.raw_bytes,uint64_t{12533760}); f.held();
    CHECK(std::all_of(f.relay.begin()+12533760,f.relay.end(),[](uint8_t x){return x==uint8_t{0xa5};}));
    // The engine must clean the ENTIRE shared relay before its existing file fallback.
    // A later valid preparation overwrites both halves and remains independently exact.
    std::memset(f.relay.data(),0,size_t(kDraftPairBytes));REQUIRE_EQ(f.prepare(),RAD_OK);f.held();
    CHECK_EQ(std::memcmp(f.relay.data(),f.raw.data(),f.raw.size()),0);
}
TEST(bank_packed_fill_truncated_or_corrupt_second_packed_header_is_refused_before_copy) {
    Fixture f;
    auto* data=const_cast<uint8_t*>(f.h.host[1]);const uint8_t saved=data[0];data[0]^=1;
    CHECK(f.prepare()<0);CHECK(f.work.refusal==pf::Refusal::Inverse); f.held();data[0]=saved;
    std::memset(f.relay.data(),0xa5,f.relay.size());
    const auto saved_slot=f.h.slot[1];f.h.slot[1]-=64;
    CHECK(f.prepare()<0);f.untouched();f.held();f.h.slot[1]=saved_slot;
    REQUIRE_EQ(f.prepare(),RAD_OK);CHECK_EQ(std::memcmp(f.relay.data(),f.raw.data(),f.raw.size()),0);
}
TEST(bank_packed_fill_source_epoch_fails_before_or_after_both_inverses) {
    Fixture f;
    f.epoch.fail_on=1;CHECK(f.prepare()<0);CHECK(f.work.refusal==pf::Refusal::Epoch);f.untouched();f.held();
    f.epoch.fail_on=2;CHECK(f.prepare()<0);CHECK(f.work.refusal==pf::Refusal::Epoch);f.held();
    CHECK_EQ(std::memcmp(f.relay.data(),f.raw.data(),f.raw.size()),0); // correct bytes still MUST NOT publish stale source
}
TEST(bank_packed_fill_refuses_relay_workspace_and_source_aliases_without_source_writes) {
    Fixture f;
    CHECK(pf::prepare(f.p,f.plan,0,f.h,const_cast<uint8_t*>(f.h.host[0]),size_t(kDraftPairBytes),
                      &f.work,Epoch::check,&f.epoch)<0); f.held();f.untouched();
    auto* alias=reinterpret_cast<pf::Workspace*>(const_cast<uint8_t*>(f.h.host[0]));
    CHECK(pf::prepare(f.p,f.plan,0,f.h,f.relay.data(),size_t(kDraftPairBytes),alias,Epoch::check,&f.epoch)<0);
    f.untouched();REQUIRE_EQ(f.prepare(),RAD_OK);CHECK_EQ(std::memcmp(f.relay.data(),f.raw.data(),f.raw.size()),0);
    CHECK(pf::prepare(f.p,f.plan,0,f.h,f.relay.data(),size_t(kDraftPairBytes)-1,&f.work,Epoch::check,&f.epoch)<0);
}
RAD_TEST_MAIN()
