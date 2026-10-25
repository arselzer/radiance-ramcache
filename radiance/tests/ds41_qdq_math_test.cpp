/* The device-candidate bit encoder is independent of the CPU reference's
 * binary-search/Fraction codebook path. Actual Torch primitives remain pinned
 * in the fixtures; no official TileLang/full-model parity is asserted. */
#include "rad_test.h"
#include "../libds41/ds41_qdq_math.h"
#include "../arch/deepseek41/qdq_ref.h"
#include <nlohmann/json.hpp>
#include <array>
#include <fstream>
namespace m=rad::ds41::qdq_math;
namespace q=rad::deepseek41::qdq;
namespace {
const nlohmann::json& fixtures(){
    static const auto data=[] {
        const auto path=std::string(RAD_GOLDEN_FIXTURES)+"/deepseek41-qdq.json";
        if(std::filesystem::file_size(path)>1000000)throw std::runtime_error("QDQ fixture cap");
        std::ifstream file(path);if(!file)throw std::runtime_error("missing QDQ primitive fixtures");
        return nlohmann::json::parse(file);
    }();return data;
}
m::Profile profile(const std::string& text){
    if(text=="window_fp8")return m::Profile::window_fp8;
    if(text=="compressed_fp4")return m::Profile::compressed_fp4;
    if(text=="index_fp4")return m::Profile::index_fp4;
    throw std::runtime_error("unknown QDQ fixture profile");
}
}
TEST(ds41_qdq_fast_e4m3_encoder_matches_actual_torch_codes_boundaries_and_all_bf16_inputs){
    for(const auto& item:fixtures().at("e4m3_codes")){
        if(!item.at("finite").get<bool>())continue;
        const auto code=item.at("code").get<uint8_t>();CHECK_EQ(q::bits(m::e4m3(code)),item.at("value_f32_bits").get<uint32_t>());
        uint8_t encoded=0xAB;REQUIRE_EQ(m::round_e4m3(m::e4m3(code),&encoded),RAD_OK);CHECK_EQ(encoded,code);
    }
    for(const auto& item:fixtures().at("e4m3_boundaries")){
        uint8_t code=0xAB;const int result=m::round_e4m3(q::from_bits(item.at("input_f32_bits").get<uint32_t>()),&code);
        if(item.at("candidate_accepts").get<bool>()){REQUIRE_EQ(result,RAD_OK);CHECK_EQ(code,item.at("code").get<uint8_t>());}
        else{CHECK(result<0);CHECK_EQ(code,0xAB);}
    }
    // All65536 BF16 bit patterns, including signs, subnormals and nonfinite
    // encodings: the independent reference performs nearest-value search.
    for(uint32_t bits=0;bits<65536;++bits){
        const float x=q::bf16_value(uint16_t(bits));uint8_t a=0xAB,b=0xAB;
        CHECK_EQ(m::round_e4m3(x,&a),q::e4m3_rne(x,&b));CHECK_EQ(a,b);
    }
}
TEST(ds41_qdq_fast_fp4_encoder_matches_independent_boundaries_and_all_bf16_inputs){
    for(const auto& item:fixtures().at("fp4_boundaries")){
        const float x=q::from_bits(item.at("input_f32_bits").get<uint32_t>());uint8_t code=0xAB;
        REQUIRE_EQ(m::round_e2m1(x,&code),RAD_OK);CHECK_EQ(code,item.at("code").get<uint8_t>());
        CHECK_EQ(q::bits(m::e2m1(code)),item.at("decoded_f32_bits").get<uint32_t>());
    }
    for(uint32_t bits=0;bits<65536;++bits){
        const float x=q::bf16_value(uint16_t(bits));uint8_t a=0xAB,b=0xAB;
        CHECK_EQ(m::round_e2m1(x,&a),q::e2m1_rne(x,&b));CHECK_EQ(a,b);
    }
}
TEST(ds41_qdq_device_math_groups_match_interpreted_profiles_with_gradual_underflow){
    REQUIRE(q::supported_f32_environment());
    for(const auto& item:fixtures().at("rows")){
        const auto kind=profile(item.at("profile").get<std::string>());
        const auto input=item.at("input_bf16_bits").get<std::vector<uint16_t>>();const auto reference_kind=q::Profile(uint32_t(kind));
        const int width=m::width(kind);REQUIRE(input.size()%size_t(width)==0);
        for(size_t first=0;first<input.size();first+=size_t(width)){
            q::GroupResult expected;std::array<uint16_t,32> got;got.fill(0xBEEF);
            const int reference=q::group(reference_kind,std::span(input).subspan(first,size_t(width)),&expected);
            const int actual=m::group(kind,input.data()+first,got.data());CHECK_EQ(actual,reference);
            for(int i=0;i<width;++i)CHECK_EQ(got[size_t(i)],actual==RAD_OK?expected.bf16[size_t(i)]:0xBEEF);
            for(int i=width;i<32;++i)CHECK_EQ(got[size_t(i)],0xBEEF);
        }
    }
    std::array<uint16_t,32> tiny{},out{};tiny[0]=0x0040;tiny[1]=0x8040;
    REQUIRE_EQ(m::group(m::Profile::index_fp4,tiny.data(),out.data()),RAD_OK);
    CHECK_EQ(out[0],0x0040);CHECK_EQ(out[1],0x8040);
    for(auto kind:{m::Profile::window_fp8,m::Profile::compressed_fp4,m::Profile::index_fp4}){
        for(int i=0;i<32;++i)tiny[size_t(i)]=i%2?0x8000:0;
        REQUIRE_EQ(m::group(kind,tiny.data(),out.data()),RAD_OK);
        for(int i=0;i<m::width(kind);++i)CHECK_EQ(out[size_t(i)],tiny[size_t(i)]);
    }
}
RAD_TEST_MAIN()
