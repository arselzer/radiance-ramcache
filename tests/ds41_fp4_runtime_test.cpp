#include "rad_test.h"
#include "place/ds41_fp4_classify.h"
#include <array>
using namespace rad;
namespace f=rad_ds41_fp4_runtime;
namespace {
int launch(const RadArgs*,RadStream){return RAD_OK;}
int alien(const RadArgs*,RadStream){return RAD_OK;}
const int one=1,two=2;
const void* origin(RadLaunchFn x){return x==launch?&one:x==alien?&two:nullptr;}
const void* missing_origin(RadLaunchFn){return nullptr;}
bool identity(RadLaunchFn x,const char* s){return x==launch&&s&&std::string(s)=="r4d_rad_quant_roundtrip_fp4_classify";}
bool both_identity(RadLaunchFn x,const char* s){return (x==launch||x==alien)&&s&&std::string(s)=="r4d_rad_quant_roundtrip_fp4_classify";}
struct Fixture {
    Program p;RadKernelInfo info{};KernelRow row{};RadOpSchema schema{};
    RadOperandSpec operands[2]={{"x",RAD_OPD_INOUT,0},{"classifier_audit",RAD_OPD_INOUT,1}};
    Fixture(){schema.n_operands=2;schema.operands=operands;
        info.name=f::kRows[f::Quant];info.op=f::kNames[f::Quant];info.domain=RAD_DOMAIN_DEVICE;info.launch=launch;
        row.info=&info;row.plugin="libr4d";row.plugin_order=0;
        OpInfo o;o.op=info.op;o.schema=&schema;o.ranged_key="M";o.range_lo=1;o.range_hi=4096;
        o.base.set_i("n",512);o.base.set_i("block",16);o.base.set_s("format","fp4_e2m1");
        o.base.set_s("scale","e4m3");o.base.set_s("dtype","bf16");
        for(int hi:{1,128,2048,4096}){Band b;b.hi=hi;b.dom[RAD_DOMAIN_DEVICE].row=&row;o.bands.push_back(b);}
        p.ops.push_back(o);
    }
    bool admit(Ds41Fp4Origin look=origin,Ds41Fp4Identity check=identity){int n=0;std::string why;return ds41_fp4_classify_admit(p,&n,&why,look,check)&&n==1;}
};
}
TEST(ds41_fp4_runtime_config_is_strict_default_off_and_scope_is_native) {
    bool on=true;CHECK(f::parse(nullptr,&on));CHECK(!on);CHECK(f::parse("0",&on));CHECK(!on);
    CHECK(f::parse("1",&on));CHECK(on);
    for(const char* s:{"","01","2","-1","true","1 "," 1"})CHECK(!f::parse(s,&on));
    CHECK(!f::parse("1",nullptr));
    CHECK(f::declare_quant(true,512,16,"fp4_e2m1","e4m3",false,0));
    CHECK(f::declare_quant(true,128,32,"fp4_e2m1","ue8m0",true,2));
    CHECK(f::declare_quant(true,4096,32,"fp4_e2m1","ue8m0",false,0));
    CHECK(!f::declare_quant(true,4096,32,"fp4_e2m1","ue8m0",true,2));
    for(int n:{64,256,1024})CHECK(!f::declare_quant(true,n,32,"fp4_e2m1","ue8m0",false,0));
    CHECK(!f::declare_quant(false,512,16,"fp4_e2m1","e4m3",false,0));
    CHECK(!f::declare_quant(true,512,16,"fp8_e4m3","e4m3",false,0));
    CHECK(!f::declare_quant(true,512,16,"fp4_e2m1","f32",false,0));
    CHECK(!f::declare_quant(true,512,16,"fp4_e2m1","e4m3",true,2));
}
TEST(ds41_fp4_runtime_rope_guards_preserve_original_resolver_for_unknown_sites) {
    CHECK(f::rope_scope(512,512,32,448,32,true,1.0f));
    CHECK(f::rope_scope(128,128,32,64,32,true,1.0f));
    CHECK(f::rope_scope(4096,128,32,64,1024,true,1.0f));
    CHECK(!f::rope_scope(4096,128,32,64,32,true,1.0f));
    CHECK(!f::rope_scope(128,128,16,96,16,true,1.0f));
    CHECK(!f::rope_scope(128,128,32,64,32,false,1.0f));
    CHECK(!f::rope_scope(128,128,32,64,32,true,-1.0f));
}
TEST(ds41_fp4_runtime_audit_counter_contract_refuses_wrap_wrong_capture_and_empty_work) {
    std::array<uint64_t,f::FieldCount> p{};CHECK(f::valid_row(p.data(),false));
    p[f::Fast]=256;p[f::Blocks]=1;CHECK(f::valid_row(p.data(),false));CHECK(!f::valid_row(p.data(),true));
    p[f::CapturedFast]=256;CHECK(f::valid_row(p.data(),true));CHECK(!f::valid_row(p.data(),false));
    p[f::Overflow]=1;CHECK(!f::valid_row(p.data(),true));p[f::Overflow]=0;
    p[f::Blocks]=0;CHECK(!f::valid_row(p.data(),true));p[f::Blocks]=1;
    p[f::Fast]=UINT64_MAX;p[f::CapturedFast]=UINT64_MAX;p[f::Fallback]=1;CHECK(!f::valid_row(p.data(),true));
    CHECK(!f::valid_row(nullptr,false));CHECK_EQ(f::kBytes,480);
    CHECK_EQ(f::slot(f::Rope,512),8);CHECK_EQ(f::slot(f::Rope,4096),9);
    CHECK_EQ(f::slot(f::Norm,512),-1);CHECK_EQ(f::slot(f::QuantCapture,4096),-1);
}
TEST(ds41_fp4_runtime_all_actual_bands_require_known_export_and_module) {
    Fixture a;CHECK(a.admit());CHECK(!a.admit(missing_origin));
    a.p.ops[0].bands.back().dom[RAD_DOMAIN_DEVICE].row=nullptr;CHECK(!a.admit());
    Fixture b;b.info.launch=alien;CHECK(!b.admit());
    Fixture c;c.row.plugin="libref";CHECK(!c.admit());
    Fixture d;d.info.name="quant_roundtrip_fp4_bf16";CHECK(!d.admit());
    Fixture e;e.row.plugin_order=-1;CHECK(!e.admit());
    Fixture g;auto info=g.info;info.launch=alien;auto row=g.row;row.info=&info;
    g.p.ops[0].bands.back().dom[RAD_DOMAIN_DEVICE].row=&row;CHECK(!g.admit(origin,both_identity));
    info.launch=launch;row.plugin_order=1;CHECK(!g.admit());
}
TEST(ds41_fp4_runtime_admission_refuses_unknown_tuning_host_and_operand_contract) {
    Fixture a;a.p.ops[0].bands[1].dom[RAD_DOMAIN_DEVICE].choice="x=1";CHECK(!a.admit());
    Fixture b;b.p.ops[0].bands[0].dom[RAD_DOMAIN_HOST].row=&b.row;CHECK(!b.admit());
    Fixture c;c.operands[1].optional=0;CHECK(!c.admit());
    Fixture d;d.operands[1].role=RAD_OPD_OUT;CHECK(!d.admit());
    Fixture e;e.p.ops[0].base.set_i("n",256);CHECK(!e.admit());
    Fixture g;g.p.ops[0].base.set_s("scale","ue8m0");CHECK(!g.admit());
    Fixture h;h.p.ops[0].op="quant_roundtrip_fp4_classify_unknown";CHECK(!h.admit());
    Fixture i;i.p.ops[0].op="quant_roundtrip";CHECK(!i.admit());
    Fixture j;j.schema.operands=nullptr;CHECK(!j.admit());
    Fixture k;k.p.ops[0].range_hi=int64_t(INT_MAX)+1;CHECK(!k.admit());
}
RAD_TEST_MAIN()
