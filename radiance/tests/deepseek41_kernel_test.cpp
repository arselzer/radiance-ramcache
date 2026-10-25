/* Actual libref plugin entrypoints and cold operand descriptions for the new DS41 oracles. */
#include "rad_test.h"
#include "plugin/registry.h"
#include "../arch/common/rad_ds41_ref.h"
#include "rad_plugin.h"
#include <nlohmann/json.hpp>
#include <fstream>
using namespace rad;
namespace ds=rad::deepseek41::ref;
namespace {
using Json=nlohmann::json;
const Json& fixture() {
    static const auto value=[] {std::ifstream file(std::string(RAD_GOLDEN_FIXTURES)+"/deepseek41-reference.json");return Json::parse(file);}();return value;
}
void flatten(const Json& x,std::vector<float>& out) {if(x.is_array())for(const auto& v:x)flatten(v,out);else out.push_back(x.get<float>());}
std::vector<float> floats(const Json& x) {std::vector<float> out;flatten(x,out);return out;}
Registry& registry() {
    static Registry r;static bool loaded=false;
    if(!loaded){loaded=true;if(r.load_plugin(home()+"/kernels/libref.so",0)!=RAD_OK)throw std::runtime_error("cannot load libref");}
    return r;
}
const RadKernelInfo* kernel(const char* op) {
    for(const auto& row:registry().rows())if(std::strcmp(row.info->op,op)==0)return row.info;
    throw std::runtime_error(std::string("missing reference op ")+op);
}
RadTensor tensor(void* data,uint32_t dt,std::initializer_list<int64_t> shape) {
    RadTensor t{};t.data=data;t.dtype=dt;t.rank=uint32_t(shape.size());size_t i=0;
    for(auto dim:shape)t.shape[i++]=dim;
    int64_t stride=1;
    for(int d=int(t.rank)-1;d>=0;--d){t.stride[d]=stride;stride*=t.shape[d];}return t;
}
int launch(const char* op,std::vector<RadParam> p,std::vector<RadTensor> t) {
    RadArgs a{};a.p=p.data();a.n_p=int(p.size());a.t=t.data();a.n_t=int(t.size());return kernel(op)->launch(&a,{});
}
void near(ds::Floats actual,ds::Floats expected) {
    REQUIRE_EQ(actual.size(),expected.size());for(size_t i=0;i<actual.size();++i)CHECK_NEAR(actual[i],expected[i],5e-6+2e-5*std::fabs(expected[i]));
}
std::vector<RadParam> hcparams() {return {RAD_INT("M",8),RAD_INT("n",6),RAD_INT("hc",4),RAD_INT("iterations",20),RAD_F64("eps",1e-6),RAD_F64("hc_eps",1e-6),RAD_STR("dtype","f32")};}
}
TEST(deepseek_plugin_mhc_prepare_exports_next_pre_without_current_collapse) {
    const auto& f=fixture()["mhc"][0];auto h=floats(f["h"]),fn=floats(f["fn"]),scale=floats(f["scale"]),base=floats(f["base"]);
    std::vector<float> pre(4),post(4),comb(16);auto p=hcparams();
    CHECK_EQ(launch("ds41_hc_prepare",p,{tensor(h.data(),RAD_F32,{1,24}),tensor(fn.data(),RAD_F32,{24,24}),tensor(scale.data(),RAD_F32,{3}),
        tensor(base.data(),RAD_F32,{24}),tensor(pre.data(),RAD_F32,{1,4}),tensor(post.data(),RAD_F32,{1,4}),tensor(comb.data(),RAD_F32,{1,16})}),RAD_OK);
    near(pre,floats(f["pre"]));near(post,floats(f["post"]));near(comb,floats(f["comb"]));
    const auto* schema=registry().schema("ds41_hc_prepare");REQUIRE(schema);CHECK_EQ(std::string(schema->operands[4].name),std::string("pre"));
    const auto* k=kernel("ds41_hc_prepare");REQUIRE(k->opd_shape);
    for(int operand=0;operand<7;++operand){RadOpdDesc d{};CHECK_EQ(k->opd_shape(p.data(),p.size(),operand,&d),RAD_OK);CHECK_EQ(d.dtype,uint32_t(RAD_F32));}
}
TEST(deepseek_plugin_collapse_supports_strided_tensors_and_bf16_rounding) {
    auto raw=floats(fixture()["mhc"][0]["h"]);auto previous=floats(fixture()["mhc"][0]["previous_pre"]);
    std::vector<float> h(48,-999),pre(8,-999),out(12,-999);for(size_t i=0;i<raw.size();++i)h[2*i]=raw[i];for(size_t i=0;i<4;++i)pre[2*i]=previous[i];
    auto ht=tensor(h.data(),RAD_F32,{1,24}),pt=tensor(pre.data(),RAD_F32,{1,4}),ot=tensor(out.data(),RAD_F32,{1,6});
    ht.stride[1]=pt.stride[1]=ot.stride[1]=2;ht.stride[0]=48;pt.stride[0]=8;ot.stride[0]=12;
    CHECK_EQ(launch("ds41_hc_collapse",hcparams(),{ht,pt,ot}),RAD_OK);
    std::vector<float> unpacked;for(size_t i=0;i<6;++i){unpacked.push_back(out[2*i]);CHECK_EQ(out[2*i+1],-999);}
    near(unpacked,floats(fixture()["mhc"][0]["collapsed"]));
    std::vector<uint16_t> bf16(24),bfout(6);for(size_t i=0;i<24;++i)bf16[i]=uint16_t(std::bit_cast<uint32_t>(ds::rounded(raw[i],ds::Precision::bf16))>>16);
    auto p=hcparams();p.back()=RAD_STR("dtype","bf16");
    CHECK_EQ(launch("ds41_hc_collapse",p,{tensor(bf16.data(),RAD_BF16,{1,24}),tensor(previous.data(),RAD_F32,{1,4}),tensor(bfout.data(),RAD_BF16,{1,6})}),RAD_OK);
    for(size_t i=0;i<6;++i)CHECK_EQ(bfout[i],bf16[i]);
}
TEST(deepseek_plugin_router_matches_independent_ids_and_unbiased_weights) {
    const auto& f=fixture()["routing"][1];auto logits=floats(f["logits"]),bias=floats(f["bias"]);std::vector<int32_t> ids(6);std::vector<float> weights(6);
    std::vector<RadParam> p{RAD_INT("M",1),RAD_INT("experts",12),RAD_INT("topk",6),RAD_F64("temperature",.75),RAD_F64("route_scale",1.5),RAD_INT("normalize",1)};
    CHECK_EQ(launch("ds41_router_topk",p,{tensor(logits.data(),RAD_F32,{1,12}),tensor(bias.data(),RAD_F32,{12}),tensor(ids.data(),RAD_I32,{1,6}),tensor(weights.data(),RAD_F32,{1,6})}),RAD_OK);
    CHECK(ids==f["ids"].get<std::vector<int32_t>>());near(weights,floats(f["weights"]));
    const auto* k=kernel("ds41_router_topk");REQUIRE(k->opd_shape);RadOpdDesc d{};CHECK_EQ(k->opd_shape(p.data(),p.size(),2,&d),RAD_OK);CHECK_EQ(d.dtype,uint32_t(RAD_I32));CHECK_EQ(d.idx_max,12);
}
TEST(deepseek_plugin_sparse_attention_validates_indices_and_output_bounds) {
    const auto& f=fixture()["attention"][1];auto q=floats(f["q"]),kv=floats(f["kv"]),sinks=floats(f["sinks"]);auto ids=f["ids"].get<std::vector<int32_t>>();
    std::vector<float> guard(q.size()+2,1234);auto p=std::vector<RadParam>{RAD_INT("M",1),RAD_INT("heads",3),RAD_INT("head_dim",8),RAD_INT("kv_rows",7),RAD_INT("topk",4),RAD_F64("scale",f["scale"].get<double>()),RAD_STR("dtype","f32")};
    auto operands=std::vector<RadTensor>{tensor(q.data(),RAD_F32,{1,3,8}),tensor(kv.data(),RAD_F32,{7,8}),tensor(sinks.data(),RAD_F32,{3}),tensor(ids.data(),RAD_I32,{1,4}),tensor(guard.data()+1,RAD_F32,{1,3,8})};
    CHECK_EQ(launch("ds41_sparse_attn",p,operands),RAD_OK);near(ds::Floats(guard).subspan(1,q.size()),floats(f["out"]));CHECK_EQ(guard.front(),1234);CHECK_EQ(guard.back(),1234);
    std::fill(guard.begin(),guard.end(),1234);ids[0]=7;CHECK(launch("ds41_sparse_attn",p,operands)<0);CHECK(std::all_of(guard.begin(),guard.end(),[](float x){return x==1234;}));
    const auto* k=kernel("ds41_sparse_attn");REQUIRE(k->opd_shape);RadOpdDesc d{};CHECK_EQ(k->opd_shape(p.data(),p.size(),3,&d),RAD_OK);CHECK_EQ(d.idx_max,7);
}
TEST(deepseek_plugin_refuses_bad_geometry_and_unsupported_io_dtype) {
    std::vector<float> h(24),pre(4),out(6,42);auto p=hcparams();p[1]=RAD_INT("n",7);
    CHECK(launch("ds41_hc_collapse",p,{tensor(h.data(),RAD_F32,{1,24}),tensor(pre.data(),RAD_F32,{1,4}),tensor(out.data(),RAD_F32,{1,6})})<0);CHECK_EQ(out[0],42);
    p=hcparams();p.back()=RAD_STR("dtype","i8");CHECK(launch("ds41_hc_collapse",p,{tensor(h.data(),RAD_F32,{1,24}),tensor(pre.data(),RAD_F32,{1,4}),tensor(out.data(),RAD_F32,{1,6})})<0);
}
RAD_TEST_MAIN()
