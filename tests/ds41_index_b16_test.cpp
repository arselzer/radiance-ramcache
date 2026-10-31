#include "rad_test.h"
#include "place/ds41_index_b16.h"
#include "../arch/common/rad_ds41_score_b16.h"
using namespace rad;
namespace {
int launch(const RadArgs*,RadStream){return RAD_OK;}
int alien(const RadArgs*,RadStream){return RAD_E_UNSUPPORTED;}
int miswired_same_module(const RadArgs*,RadStream){return RAD_OK;}
const int one=0,two=0;
const void* origin(RadLaunchFn f){return !f?nullptr:f==alien?&two:&one;}
bool identity(RadLaunchFn f,const char*s){return f==launch&&s&&std::strstr(s,"r4d_rad_ds41_index_")==s;}
struct Fixture {
    Program p;RadKernelInfo info[3]{};KernelRow row[3]{};
    Fixture(){
        BufferInfo plane;plane.name="index_score";plane.decl.dtype=RAD_BF16;plane.decl.rank=2;
        plane.decl.shape[0]=2047;plane.decl.shape[1]=32769;p.buffers.push_back(plane);
        const char*ops[]={"index_score_b16","index_candidates_b16","index_topk_b16"};
        for(int i=0;i<3;++i){info[i].op=ops[i];info[i].name=ds41_index_known_row(ops[i]);
            info[i].domain=RAD_DOMAIN_DEVICE;info[i].launch=launch;row[i].info=&info[i];row[i].plugin="libr4d";
            OpInfo o;o.op=ops[i];Band b;b.hi=4096;b.dom[RAD_DOMAIN_DEVICE].row=&row[i];o.bands.push_back(b);p.ops.push_back(o);}
    }
};
}
TEST(index_b16_all_score_bit_patterns_expand_without_rounding_or_payload_changes){
    using namespace rad_ds41_score_b16;
    for(uint32_t i=0;i<65536;++i){const uint32_t old=i<<16;CHECK(representable(old));CHECK_EQ(expand_bits(high_bits(old)),old);}
    for(uint32_t u:{0u,0x80000000u,0x7f800000u,0xff800000u,0x7fc10000u,0xffc10000u,0x00010000u,0x80010000u})
        CHECK_EQ(expand_bits(high_bits(u)),u);
    CHECK(!representable(0x3f800001u)); // not permission to round an arbitrary f32 score
}
TEST(index_b16_admission_requires_the_complete_known_same_module_chain_in_every_band){
    Fixture f;Ds41IndexB16Proof p;std::string why;
    CHECK(ds41_index_b16_admit(f.p,&p,&why,origin,identity));CHECK(why.empty());
    CHECK_EQ(p.rows,2047);CHECK_EQ(p.columns,32769);CHECK_EQ(p.writers,1);CHECK_EQ(p.candidates,1);CHECK_EQ(p.topk,1);
    f.p.ops[0].bands.push_back(f.p.ops[0].bands[0]);CHECK(ds41_index_b16_admit(f.p,&p,&why,origin,identity));
    CHECK_EQ(p.rows*p.columns*2,134156286ll); // same original run graph, exactly half bytes
}
TEST(index_b16_admission_refuses_old_packed_unknown_tuned_or_cross_library_paths){
    const std::vector<std::function<void(Fixture&)>> bad={
        [](Fixture&f){f.p.buffers[0].decl.dtype=RAD_F32;},
        [](Fixture&f){f.p.buffers.push_back(f.p.buffers[0]);},
        [](Fixture&f){f.p.ops[0].op="index_score_kvpack";},
        [](Fixture&f){f.p.ops[0].op="index_score";},
        [](Fixture&f){f.p.ops[2].op="index_topk_unknown";},
        [](Fixture&f){f.info[1].name="index_candidates";},
        [](Fixture&f){f.info[2].op="index_topk";},
        [](Fixture&f){f.info[1].launch=alien;},
        [](Fixture&f){f.info[1].launch=miswired_same_module;},
        [](Fixture&f){f.info[0].launch=nullptr;},
        [](Fixture&f){f.row[1].plugin="override";},
        [](Fixture&f){f.row[1].plugin_order=1;},
        [](Fixture&f){f.info[0].n_tunables=1;},
        [](Fixture&f){f.p.ops[0].bands[0].dom[RAD_DOMAIN_DEVICE].choice="M=64";},
        [](Fixture&f){f.p.ops[1].bands[0].dom[RAD_DOMAIN_DEVICE].tune_source="RADIANCE_TUNE";},
        [](Fixture&f){f.p.ops[0].bands[0].dom[RAD_DOMAIN_HOST].row=&f.row[0];},
        [](Fixture&f){f.p.ops[0].bands.clear();},
        [](Fixture&f){auto b=f.p.ops[0].bands[0];b.dom[RAD_DOMAIN_DEVICE].row=&f.row[1];f.p.ops[0].bands.push_back(b);},
        [](Fixture&f){f.p.ops.pop_back();},
    };
    for(const auto&mutate:bad){Fixture f;mutate(f);Ds41IndexB16Proof p;std::string why;
        CHECK(!ds41_index_b16_admit(f.p,&p,&why,origin,identity));CHECK(!why.empty());}
}
RAD_TEST_MAIN()
