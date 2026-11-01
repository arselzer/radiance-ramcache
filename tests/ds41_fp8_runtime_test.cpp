#include "rad_test.h"
#include "place/ds41_fp8_convert.h"
#include <array>
using namespace rad;
namespace f=rad_ds41_fp8_runtime;
namespace {
int launch(const RadArgs*,RadStream){return RAD_OK;}
int alien(const RadArgs*,RadStream){return RAD_OK;}
const int one=1,two=2;
const void*origin(RadLaunchFn fn){return fn==launch?&one:fn==alien?&two:nullptr;}
bool identity(RadLaunchFn fn,const char*s){return fn==launch&&s&&std::string(s)=="r4d_rad_rope_rows_roundtrip_fp8_convert";}
bool both(RadLaunchFn fn,const char*s){return (fn==launch||fn==alien)&&s&&std::string(s)=="r4d_rad_rope_rows_roundtrip_fp8_convert";}
struct Fixture {
    Program p;RadKernelInfo info{};KernelRow row{};RadOpSchema schema{};
    RadOperandSpec args[4]={{"x",RAD_OPD_INOUT,0},{"cos_sin",RAD_OPD_IN,0},
        {"positions",RAD_OPD_IN,0},{"convert_audit",RAD_OPD_INOUT,1}};
    Fixture(){info.name=f::kRows[f::Rope];info.op=f::kNames[f::Rope];info.domain=RAD_DOMAIN_DEVICE;info.launch=launch;
        row.info=&info;row.plugin="libr4d";row.plugin_order=0;schema.n_operands=4;schema.operands=args;
        OpInfo o;o.op=info.op;o.schema=&schema;o.ranged_key="M";o.range_lo=1;o.range_hi=4096;
        o.base.set_s("format","fp8_e4m3");o.base.set_s("scale","ue8m0");o.base.set_s("dtype","bf16");
        for(auto kv:{std::pair<const char*,int>{"block",32},{"n_head",16},{"head_dim",512},{"rot",64},{"off",448},{"inverse",1}})
            o.base.set_i(kv.first,kv.second);
        o.base.set_s("mode","gptj");
        for(int hi:{1,128,2048,4096}){Band b;b.hi=hi;b.dom[RAD_DOMAIN_DEVICE].row=&row;o.bands.push_back(b);}
        for(int i=0;i<40;++i)p.ops.push_back(o);
    }
    bool admit(Ds41Fp8Identity test=identity){int count=0;std::string why;
        return ds41_fp8_convert_admit(p,&count,&why,origin,test)&&count==40;}
};
}
TEST(ds41_fp8_runtime_strict_default_off_and_native_only) {
    bool on=true;CHECK(f::parse(nullptr,&on));CHECK(!on);CHECK(f::parse("0",&on));CHECK(!on);
    CHECK(f::parse("1",&on));CHECK(on);CHECK(!f::parse("1",nullptr));
    for(const char*s:{"","01","2","true","1 "," 1"})CHECK(!f::parse(s,&on));
    for(int n:{512,8192})CHECK(f::declare_quant(true,n,32,"fp8_e4m3","ue8m0",false));
    for(int n:{128,256,4096,8193})CHECK(!f::declare_quant(true,n,32,"fp8_e4m3","ue8m0",false));
    CHECK(!f::declare_quant(false,512,32,"fp8_e4m3","ue8m0",false));
    CHECK(!f::declare_quant(true,512,32,"fp8_e4m3","ue8m0",true));
    CHECK(!f::declare_quant(true,512,16,"fp8_e4m3","ue8m0",false));
    CHECK(!f::declare_quant(true,512,32,"fp4_e2m1","ue8m0",false));
    CHECK(!f::declare_quant(true,512,32,"fp8_e4m3","f32",false));
    CHECK(f::rope_scope(8192,512,32,448,512,true,-1.f));
    CHECK(!f::rope_scope(8192,512,32,448,512,true,1.f));
    CHECK(!f::rope_scope(8192,512,32,448,512,false,-1.f));
    CHECK(!f::rope_scope(512,512,32,448,32,true,-1.f));
}
TEST(ds41_fp8_runtime_domain_keeps_all_exception_and_non_normal_scale_fallbacks) {
    for(uint32_t v:{0u,0x80000000u,1u,0x007fffffu,0x00800000u,0x7f7fffffu,0xff7fffffu})
        for(uint32_t s:{0x00800000u,0x3f800000u,0x7f7fffffu})CHECK(f::hardware_domain(v,s));
    for(uint32_t v:{0x7f800000u,0xff800000u,0x7f800001u,0xffc00001u})CHECK(!f::hardware_domain(v,0x3f800000u));
    for(uint32_t s:{0u,0x80000000u,1u,0x007fffffu,0xbf800000u,0x7f800000u,0x7f800001u})
        CHECK(!f::hardware_domain(0x3f800000u,s));
}
TEST(ds41_fp8_runtime_all_40_native_sites_all_bands_and_exact_module_are_required) {
    Fixture a;CHECK(a.admit());a.p.ops.pop_back();CHECK(!a.admit());
    Fixture b;b.p.ops[39].bands.back().dom[RAD_DOMAIN_DEVICE].row=nullptr;CHECK(!b.admit());
    Fixture c;c.info.launch=alien;CHECK(!c.admit());
    Fixture d;auto info=d.info;info.launch=alien;auto row=d.row;row.info=&info;
    d.p.ops[39].bands[1].dom[RAD_DOMAIN_DEVICE].row=&row;CHECK(!d.admit(both));
    Fixture e;e.row.plugin="libref";CHECK(!e.admit());
    Fixture g;g.info.name="rope_rows_roundtrip_fp8_bf16";CHECK(!g.admit());
    Fixture h;h.row.plugin_order=-1;CHECK(!h.admit());
}
TEST(ds41_fp8_runtime_unknown_tuning_geometry_and_audit_contract_refuse) {
    Fixture a;a.p.ops[8].base.set_i("inverse",0);CHECK(!a.admit());
    Fixture b;b.p.ops[9].base.set_i("block",16);CHECK(!b.admit());
    Fixture c;c.p.ops[10].bands[0].dom[RAD_DOMAIN_DEVICE].choice="x=1";CHECK(!c.admit());
    Fixture d;d.p.ops[11].bands[0].dom[RAD_DOMAIN_HOST].row=&d.row;CHECK(!d.admit());
    Fixture e;e.args[3].optional=0;CHECK(!e.admit());
    Fixture g;g.args[3].role=RAD_OPD_OUT;CHECK(!g.admit());
    Fixture h;h.p.ops[12].op="rope_rows_roundtrip_fp8_convert_unknown";CHECK(!h.admit());
    Fixture i;i.p.ops[13].range_hi=int64_t(INT_MAX)+1;CHECK(!i.admit());
}
TEST(ds41_fp8_runtime_audit_width_slots_zero_rows_and_overflow_are_explicit) {
    std::array<uint64_t,f::FieldCount> row{};CHECK(f::valid_row(row.data()));
    row[f::Fast]=1;CHECK(!f::valid_row(row.data()));row[f::Blocks]=1;CHECK(f::valid_row(row.data()));
    row[f::Fallback]=2;CHECK(f::valid_row(row.data()));row[f::Overflow]=1;CHECK(!f::valid_row(row.data()));
    row[f::Overflow]=0;row[f::Fast]=UINT64_MAX;CHECK(!f::valid_row(row.data()));CHECK(!f::valid_row(nullptr));
    CHECK_EQ(f::slot(f::Quant,512),0);CHECK_EQ(f::slot(f::Quant,8192),1);CHECK_EQ(f::slot(f::Rope,8192),2);
    CHECK_EQ(f::slot(f::Rope,512),-1);CHECK_EQ(f::kBytes,96);
}
RAD_TEST_MAIN()
