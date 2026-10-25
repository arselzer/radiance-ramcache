/* Independently executed, unchanged pinned official Torch CPU bodies. These
 * checks do not execute TileLang, model weights, collectives or a serving graph.
 * BF16 <=1 ULP and small F32 tolerances are numerical baselines, NOT bit parity. */
#include "rad_test.h"
#include "../libds41/ds41_transform_emulate.h"
#include "../libds41/ds41_transform_table.h"
#include "../libds41/ds41_engram_gate_emulate.h"
#include <nlohmann/json.hpp>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

namespace tr=rad::ds41::transform;
namespace eg=rad::ds41::engram_gate;
namespace {
using Json=nlohmann::json;
void need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
const Json& oracle(){
    static const Json data=[] {
        const auto path=std::string(RAD_GOLDEN_FIXTURES)+"/ds41-official-torch-cpu.json";
        need(std::filesystem::file_size(path)<=300000,"official tiny CPU fixture exceeds cap");
        std::ifstream file(path);need(bool(file),"official CPU fixture missing");
        Json j=Json::parse(file);
        need(j.at("official_revision")=="1bae0e1088993b9b82de08f59de5be5a34e42170","official fixture source pin changed");
        need(j.at("gpu_execution")==false&&j.at("model_payload_read")==false&&
             j.at("model_quality_or_gpu_bit_parity_established")==false,"fixture scope metadata changed");
        return j;
    }();return data;
}
uint32_t dtype(const std::string& name){
    if(name=="float32")return RAD_F32;
    if(name=="bfloat16")return RAD_BF16;
    throw std::runtime_error("unsupported official CPU fixture dtype");
}
struct Tensor {
    static constexpr size_t guard=64;
    RadTensor t{};std::vector<uint32_t> words;int64_t count=0,bytes=0;
    Tensor()=default;
    Tensor(uint32_t type,std::initializer_list<int64_t> shape){init(type,std::vector<int64_t>(shape));}
    void init(uint32_t type,const std::vector<int64_t>& shape){
        need(!shape.empty()&&shape.size()<=4,"fixture rank outside cap");
        t.dtype=type;t.rank=unsigned(shape.size());int64_t stride=1;
        for(int d=int(shape.size())-1;d>=0;--d){need(shape[size_t(d)]>0&&stride<=4096/shape[size_t(d)],"fixture element cap exceeded");t.shape[d]=shape[size_t(d)];t.stride[d]=stride;stride*=shape[size_t(d)];}
        count=stride;bytes=rad_dtype_bytes(type,count);need(bytes>0,"unsupported fixture storage");
        words.assign(size_t((bytes+2*guard+3)/4),0xBDBDBDBDu);t.data=reinterpret_cast<uint8_t*>(words.data())+guard;
    }
    explicit Tensor(const Json& source,bool drop_batch=false){
        auto shape=source.at("shape").get<std::vector<int64_t>>();
        if(drop_batch){need(shape.size()==4&&shape[0]==1,"only a single leading batch may be flattened");shape.erase(shape.begin());}
        init(dtype(source.at("dtype").get<std::string>()),shape);
        const auto raw=source.at("raw_u8").get<std::vector<uint8_t>>();need(int64_t(raw.size())==bytes,"raw oracle extent differs");
        std::memcpy(t.data,raw.data(),raw.size());
    }
    void check()const{
        const auto* p=reinterpret_cast<const uint8_t*>(words.data());
        for(size_t i=0;i<guard;++i)need(p[i]==0xBD&&p[size_t(bytes)+guard+i]==0xBD,"official fixture redzone changed");
    }
    float value(int64_t at)const{return rad_load_f32(t.data,t.dtype,at);}
    void value(int64_t at,float x){rad_store_f32(t.data,t.dtype,at,x);}
};
int ordered(uint16_t bits){return bits&0x8000u?32768-int(bits&0x7FFFu):32768+int(bits);}
void compare(const Tensor& actual,const Json& expected){
    Tensor reference(expected);need(reference.bytes==actual.bytes&&reference.t.dtype==actual.t.dtype,"oracle output storage differs");
    for(int64_t i=0;i<actual.count;++i){
        const float a=actual.value(i),b=reference.value(i);need(std::isfinite(a)&&std::isfinite(b),"unexpected nonfinite ordinary oracle output");
        if(b==0.f){CHECK_EQ(std::signbit(a),std::signbit(b));CHECK_EQ(a,0.f);}
        else if(actual.t.dtype==RAD_BF16){
            uint16_t ah=0,bh=0;std::memcpy(&ah,static_cast<uint8_t*>(actual.t.data)+i*2,2);std::memcpy(&bh,static_cast<uint8_t*>(reference.t.data)+i*2,2);
            CHECK(std::abs(ordered(ah)-ordered(bh))<=1);
        }else CHECK(std::fabs(a-b)<=3e-6f*std::max(1.f,std::fabs(b)));
    }
    actual.check();reference.check();
}
RadArgs args(RadTensor* tensors,int nt,std::vector<RadParam>& params,std::array<int32_t,4>& flag){
    RadArgs a{};a.t=tensors;a.n_t=nt;a.p=params.data();a.n_p=int(params.size());a.scratch=flag.data();a.scratch_bytes=4;return a;
}
}
TEST(ds41_norm_matches_unchanged_official_torch_cpu_storage_boundaries){
    REQUIRE_EQ(oracle().at("rmsnorm").size(),12u);
    for(const auto& entry:oracle().at("rmsnorm")){
        Tensor x(entry.at("input"));const int64_t rows=x.t.shape[0],dim=x.t.shape[1];
        Tensor gain(RAD_F32,{dim}),out(x.t.dtype,{rows,dim});
        const auto values=entry.at("gain").at("values_f32").get<std::vector<float>>();REQUIRE_EQ(values.size(),size_t(dim));
        for(int64_t i=0;i<dim;++i)gain.value(i,values[size_t(i)]);
        std::vector<RadParam> params{RAD_INT("M",rows),RAD_INT("n",dim),RAD_F64("eps",entry.at("epsilon").get<double>()),
            RAD_STR("a_dtype",rad_dtype_name(x.t.dtype)),RAD_STR("y_dtype",rad_dtype_name(x.t.dtype))};
        RadTensor tensors[]{x.t,gain.t,out.t};std::array<int32_t,4> flag{999,998,997,996};auto a=args(tensors,3,params,flag);
        REQUIRE_EQ(tr::emulate(tr::Op::rmsnorm,&a),RAD_OK);CHECK_EQ(flag[0],RAD_OK);CHECK_EQ(flag[1],998);
        compare(out,entry.at("output"));x.check();gain.check();
    }
}
TEST(ds41_tail_rope_and_host_table_match_unchanged_official_torch_cpu_frequencies){
    REQUIRE_EQ(oracle().at("rope").size(),8u);
    for(const auto& entry:oracle().at("rope")){
        Tensor x(entry.at("input"),true),out(x.t.dtype,{3,2,8}),positions(RAD_I32,{3}),pairs(RAD_F32,{64,2,2});
        REQUIRE(x.t.shape[0]==3&&x.t.shape[1]==2&&x.t.shape[2]==8);
        REQUIRE(entry.at("selected_frequencies").at("shape")==Json::array({3,2}));
        const auto selected=entry.at("selected_frequencies").at("real_imag_f32").get<std::vector<float>>();
        const auto original=entry.at("positions").get<std::vector<int32_t>>();REQUIRE_EQ(selected.size(),12u);REQUIRE_EQ(original.size(),3u);
        for(int row=0;row<3;++row){static_cast<int32_t*>(positions.t.data)[row]=original[size_t(row)];for(int d=0;d<4;++d)pairs.value(int64_t(original[size_t(row)])*4+d,selected[size_t(row*4+d)]);}
        const auto& s=entry.at("spec");tr::RopeSpec spec{s.at("rotary_dim").get<int64_t>(),s.at("original_context").get<int64_t>(),s.at("theta").get<float>(),s.at("factor").get<float>(),s.at("beta_fast").get<float>(),s.at("beta_slow").get<float>()};
        std::vector<float> built;REQUIRE_EQ(tr::build_pairs(spec,64,64*4*4,&built),RAD_OK);
        for(int row=0;row<3;++row)for(int d=0;d<4;++d)CHECK(std::fabs(built[size_t(original[size_t(row)]*4+d)]-selected[size_t(row*4+d)])<=2e-6f);
        std::vector<RadParam> params{RAD_INT("M",3),RAD_INT("heads",2),RAD_INT("head_dim",8),RAD_INT("rotary_dim",4),RAD_STR("dtype",rad_dtype_name(x.t.dtype)),RAD_INT("inverse",entry.at("inverse").get<bool>()?1:0)};
        RadTensor tensors[]{x.t,positions.t,pairs.t,out.t};std::array<int32_t,4> flag{999,998,997,996};auto a=args(tensors,4,params,flag);
        REQUIRE_EQ(tr::emulate(tr::Op::rope_tail,&a),RAD_OK);compare(out,entry.at("output"));
        const int width=rad_dtype_bits(x.t.dtype)/8;for(int row=0;row<6;++row)CHECK(std::memcmp(static_cast<uint8_t*>(x.t.data)+row*8*width,static_cast<uint8_t*>(out.t.data)+row*8*width,size_t(4*width))==0);
        x.check();positions.check();pairs.check();
    }
}
TEST(ds41_engram_gate_matches_unchanged_official_torch_cpu_forward_with_supplied_projections){
    REQUIRE_EQ(oracle().at("engram").size(),18u);
    for(const auto& entry:oracle().at("engram")){
        Tensor x(entry.at("input"),true),kv(entry.at("projected_kv")),q(RAD_F32,{2,4}),k(RAD_F32,{2,4}),mask(RAD_I32,{3}),out(x.t.dtype,{3,2,4}),gates(RAD_F32,{3,2});
        REQUIRE(x.t.shape[0]==3&&x.t.shape[1]==2&&x.t.shape[2]==4);
        REQUIRE(kv.t.rank==3&&kv.t.shape[0]==1&&kv.t.shape[1]==3&&kv.t.shape[2]==12);
        REQUIRE(entry.at("q_weight").at("shape")==Json::array({2,4}));
        REQUIRE(entry.at("k_weight").at("shape")==Json::array({2,4}));
        // The official KV fixture is [1,T,(copies+1)*D]; the operator's explicit
        // supplied projection descriptor is [T,copies+1,D] over the SAME bytes.
        kv.t.rank=3;kv.t.shape[0]=3;kv.t.shape[1]=3;kv.t.shape[2]=4;kv.t.stride[0]=12;kv.t.stride[1]=4;kv.t.stride[2]=1;
        for(int i=0;i<8;++i){q.value(i,entry.at("q_weight").at("values_f32")[i].get<float>());k.value(i,entry.at("k_weight").at("values_f32")[i].get<float>());}
        const bool masked=!entry.at("mask").is_null();if(masked)for(int i=0;i<3;++i)static_cast<int32_t*>(mask.t.data)[i]=entry.at("mask")[i].get<int32_t>();
        std::vector<RadParam> params{RAD_INT("M",3),RAD_INT("copies",2),RAD_INT("n",4),RAD_STR("dtype",rad_dtype_name(x.t.dtype)),
            RAD_F64("eps",entry.at("epsilon").get<double>()),RAD_F64("clamp",entry.at("clamp").get<double>()),RAD_INT("token_mask",masked?1:0),RAD_INT("gates",1)};
        RadTensor tensors[]{x.t,kv.t,q.t,k.t,masked?mask.t:RadTensor{},out.t,gates.t};std::array<int32_t,4> flag{999,998,997,996};auto a=args(tensors,7,params,flag);
        REQUIRE_EQ(eg::emulate(&a),RAD_OK);compare(out,entry.at("output"));compare(gates,entry.at("observed_unmodified_forward_locals").at("gate"));
        x.check();kv.check();q.check();k.check();mask.check();
    }
}
TEST(ds41_engram_signed_zero_clamp_matches_the_literal_official_torch_assignment){
    REQUIRE_EQ(oracle().at("literal_dot_gate_assignment").size(),3u);
    for(const auto& entry:oracle().at("literal_dot_gate_assignment")){
        const auto input=entry.at("dot").at("values_f32").get<std::vector<float>>();const auto output=entry.at("gate").at("values_f32").get<std::vector<float>>();
        REQUIRE_EQ(input.size(),output.size());
        for(size_t i=0;i<input.size();++i)CHECK(std::fabs(eg::from_dot(input[i],entry.at("clamp").get<float>())-output[i])<=1e-7f);
    }
}
RAD_TEST_MAIN()
