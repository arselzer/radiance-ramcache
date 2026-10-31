#include "rad_test.h"
#include "place/ds41_window_packed.h"
#include "mem/kv_storage_decode.h"
#include "runtime/kv_storage_guard.h"
#include <functional>
#include <vector>
using namespace rad;
namespace {
int launch(const RadArgs*,RadStream){return RAD_OK;}
int alien(const RadArgs*,RadStream){return RAD_E_UNSUPPORTED;}
int miswired(const RadArgs*,RadStream){return RAD_OK;}
const int one=0,two=0;
const void* origin(RadLaunchFn f){return !f?nullptr:f==alien?&two:&one;}
bool identity(RadLaunchFn f,const char*s){return f==launch&&s&&std::strstr(s,"windowpack");}
struct Fixture {
    Program p;RadKernelInfo info[3]{};KernelRow rows[3]{};
    Fixture(){
        KVGroupInfo g;g.name="kv_window";g.decl.kind=RAD_KV_WINDOW;g.decl.dtype=RAD_BF16;
        g.decl.n_head_kv=1;g.decl.head_dim=256;g.decl.window=132;g.block_size=16;g.layers={2};
        g.decl.storage_format=RAD_KV_STORAGE_DS41_FP8_WINDOW_G32;g.decl.storage_block_bytes=8448;
        g.bytes_per_block=8448;p.kv_groups.push_back(g);
        const char*ops[]={"quant_roundtrip_windowpack","kv_store_windowpack","attn_sparse_sink_windowpack"};
        const char*names[]={"quant_roundtrip_fp8_windowpack","ds41_kv_store_windowpack","ds41_attn_sparse_sink_d512_windowpack"};
        for(int i=0;i<3;++i){info[i].op=ops[i];info[i].name=names[i];info[i].domain=RAD_DOMAIN_DEVICE;info[i].launch=launch;
            rows[i].info=&info[i];rows[i].plugin="libr4d";OpInfo o;o.op=ops[i];o.base.set_i(i==2?"window_format":"kv_format",3);
            Band b;b.hi=4096;b.dom[RAD_DOMAIN_DEVICE].row=&rows[i];o.bands.push_back(b);p.ops.push_back(o);}
    }
};
}
TEST(window_packed_all_bands_require_native_geometry_and_actual_export_identity) {
    Fixture f;Ds41WindowPackedProof proof;std::string why;
    REQUIRE(ds41_window_packed_admit(f.p,&proof,&why,origin,identity));
    CHECK_EQ(proof.groups,1);CHECK_EQ(proof.layers,1);CHECK_EQ(proof.producers,1);CHECK_EQ(proof.stores,1);CHECK_EQ(proof.readers,1);
    f.p.ops[2].bands.push_back(f.p.ops[2].bands[0]);CHECK(ds41_window_packed_admit(f.p,&proof,&why,origin,identity));
    const std::vector<std::function<void(Fixture&)>> bad={
        [](Fixture&f){f.p.kv_groups[0].decl.kind=RAD_KV_FULL;},
        [](Fixture&f){f.p.kv_groups[0].block_size=8;},
        [](Fixture&f){f.p.kv_groups[0].decl.head_dim=128;},
        [](Fixture&f){f.p.kv_groups[0].bytes_per_block=16384;},
        [](Fixture&f){f.p.kv_groups.push_back(f.p.kv_groups[0]);},
        [](Fixture&f){f.info[1].launch=miswired;},
        [](Fixture&f){f.info[1].launch=alien;},
        [](Fixture&f){f.info[0].name="quant_roundtrip_fp8";},
        [](Fixture&f){f.info[2].op="attn_sparse_sink";},
        [](Fixture&f){f.rows[1].plugin="override";},
        [](Fixture&f){f.rows[1].plugin_order=1;},
        [](Fixture&f){f.info[0].n_tunables=1;},
        [](Fixture&f){f.p.ops[2].bands[0].dom[RAD_DOMAIN_DEVICE].choice="x=1";},
        [](Fixture&f){f.p.ops[0].bands[0].dom[RAD_DOMAIN_HOST].row=&f.rows[0];},
        [](Fixture&f){f.p.ops[1].base.set_i("kv_format",1);},
        [](Fixture&f){f.p.ops[2].bands.clear();},
        [](Fixture&f){auto b=f.p.ops[2].bands[0];b.dom[RAD_DOMAIN_DEVICE].row=&f.rows[1];f.p.ops[2].bands.push_back(b);},
        [](Fixture&f){f.p.ops.pop_back();},
    };
    for(const auto&m:bad){Fixture x;m(x);CHECK(!ds41_window_packed_admit(x.p,&proof,&why,origin,identity));CHECK(!why.empty());}
}
TEST(window_packed_logical_hash_decode_keeps_all_sixteen_slots_raw_bits_and_run_boundaries) {
    Fixture f;const auto&d=f.p.kv_groups[0].decl;std::vector<uint8_t> p(2*8448);
    std::vector<uint16_t> expected(2*16*512),scratch(16*512);
    for(int r=0;r<32;++r){uint8_t*row=p.data()+r*528;
        for(int g=0;g<16;++g){uint8_t*payload=row+g*32;
            const uint16_t a=(r&1)?0x8000:0x7fc1,b=(r&1)?0xffc2:0;
            payload[0]=uint8_t(a);payload[1]=uint8_t(a>>8);payload[2]=uint8_t(b);payload[3]=uint8_t(b>>8);
            payload[4]=0xa5;payload[5]=0x5a;payload[6]=0x66;payload[7]=0x99;}
        REQUIRE(rad_ds41_window_codec::decode_row(row,512,expected.data()+r*512));}
    for(int page:{1,0})for(int slot:{0,7,15})for(int half:{0,1}){
        const int64_t off=(int64_t(page)*16+slot)*1024+half*512;const uint8_t*out=nullptr;
        REQUIRE(kv_storage_decode_run(d,16,p.data(),p.size(),off,512,scratch.data(),scratch.size(),&out));
        CHECK_EQ(std::memcmp(out,reinterpret_cast<const uint8_t*>(expected.data())+off,512),0);
    }
    p[512]=1;const uint8_t*out=nullptr;
    CHECK(!kv_storage_decode_run(d,16,p.data(),p.size(),0,512,scratch.data(),scratch.size(),&out));
    CHECK(out==nullptr);
}
TEST(window_packed_operand_roles_and_flag_validation_fail_closed) {
    bool b=true;
    for(const char*e:{(const char*)nullptr,"","0"}){CHECK(ds41_window_packed_parse(e,&b));CHECK(!b);}
    CHECK(ds41_window_packed_parse("1",&b));CHECK(b);
    for(const char*e:{"2","on"," 1","1x"})CHECK(!ds41_window_packed_parse(e,&b));
    CHECK(kv_storage_operand_error(3,"kv_store_windowpack","ds41_kv_store_windowpack","kv_store_windowpack",3,3,3)==nullptr);
    CHECK(kv_storage_operand_error(3,"attn_sparse_sink_windowpack","ds41_attn_sparse_sink_d512_windowpack","attn_sparse_sink_windowpack",1,3,3)==nullptr);
    CHECK(kv_storage_operand_error(1,"attn_sparse_sink_windowpack","ds41_attn_sparse_sink_d512_windowpack","attn_sparse_sink_windowpack",4,1,1)==nullptr);
    for(int i:{0,2,4})CHECK(kv_storage_operand_error(3,"kv_store_windowpack","ds41_kv_store_windowpack","kv_store_windowpack",i,3,3)!=nullptr);
    CHECK(kv_storage_operand_error(3,"attn_sparse_sink","ds41_attn_sparse_sink_d512_bf16","attn_sparse_sink",1,3,3)!=nullptr);
    CHECK(kv_storage_operand_error(3,"attn_sparse_sink_windowpack","ds41_attn_sparse_sink_d512_windowpack","attn_sparse_sink_windowpack",4,3,3)!=nullptr);
}
RAD_TEST_MAIN()
