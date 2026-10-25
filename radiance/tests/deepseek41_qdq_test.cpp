#include "rad_test.h"
#include "../arch/deepseek41/qdq_ref.h"
#include <nlohmann/json.hpp>
#include <fstream>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif
namespace q=rad::deepseek41::qdq;
namespace {
const nlohmann::json& fixtures() {
    static const auto json=[] {
        const auto path=std::string(RAD_GOLDEN_FIXTURES)+"/deepseek41-qdq.json";
        if(std::filesystem::file_size(path)>1000000)throw std::runtime_error("QDQ fixture exceeds cap");
        std::ifstream input(path);if(!input)throw std::runtime_error("missing QDQ candidate fixture");
        auto result=nlohmann::json::parse(input);
        if(result.at("schema")!="deepseek41-cache-qdq-candidate-fixtures-v1"||
           result.at("official_tilelang_executed")!=false||result.at("gpu_execution")!=false)
            throw std::runtime_error("QDQ fixture provenance changed");
        return result;
    }();return json;
}
q::Profile profile(const std::string& name) {
    if(name=="window_fp8")return q::Profile::window_fp8;
    if(name=="compressed_fp4")return q::Profile::compressed_fp4;
    if(name=="index_fp4")return q::Profile::index_fp4;
    throw std::runtime_error("unknown fixture profile");
}
struct Output {
    std::vector<uint16_t> values;std::vector<uint8_t> codes,scale_codes;std::vector<float> scales;
    Output(size_t width,size_t groups):values(width+2,0xbeef),codes(width+2,0xab),scale_codes(groups+2,0xcd),scales(groups+2,1234.f){}
    q::RowOutput view() {
        return {std::span(values).subspan(1,values.size()-2),std::span(codes).subspan(1,codes.size()-2),
                std::span(scale_codes).subspan(1,scale_codes.size()-2),std::span(scales).subspan(1,scales.size()-2)};
    }
    bool same(const Output& other)const{return values==other.values&&codes==other.codes&&scale_codes==other.scale_codes&&scales==other.scales;}
    void guards()const {
        CHECK_EQ(values.front(),0xbeef);CHECK_EQ(values.back(),0xbeef);CHECK_EQ(codes.front(),0xab);CHECK_EQ(codes.back(),0xab);
        CHECK_EQ(scale_codes.front(),0xcd);CHECK_EQ(scale_codes.back(),0xcd);CHECK_EQ(scales.front(),1234.f);CHECK_EQ(scales.back(),1234.f);
    }
};
}
TEST(deepseek41_qdq_e4m3_all_finite_codes_and_rne_boundaries_match_actual_cpu_torch_casts) {
    REQUIRE_EQ(fixtures().at("e4m3_codes").size(),256u);
    for(const auto& entry:fixtures().at("e4m3_codes")) {
        const auto code=entry.at("code").get<uint8_t>();float decoded=1234.f;
        if(entry.at("finite").get<bool>()) {
            REQUIRE_EQ(q::e4m3_value(code,&decoded),RAD_OK);CHECK_EQ(q::bits(decoded),entry.at("value_f32_bits").get<uint32_t>());
            uint8_t roundtrip=255;REQUIRE_EQ(q::e4m3_rne(decoded,&roundtrip),RAD_OK);CHECK_EQ(roundtrip,code);
        } else {CHECK_EQ(q::e4m3_value(code,&decoded),RAD_E_FORMAT);CHECK_EQ(decoded,1234.f);}
    }
    for(const auto& entry:fixtures().at("e4m3_boundaries")) {
        uint8_t code=0xab;const auto input=q::from_bits(entry.at("input_f32_bits"));
        const int result=q::e4m3_rne(input,&code);
        if(entry.at("candidate_accepts").get<bool>()){REQUIRE_EQ(result,RAD_OK);CHECK_EQ(code,entry.at("code").get<uint8_t>());}
        else{CHECK(result<0);CHECK_EQ(code,0xab);}
    }
}
TEST(deepseek41_qdq_e2m1_codebook_ties_nextafter_and_signs_match_independent_fraction_oracle) {
    REQUIRE_EQ(fixtures().at("fp4_boundaries").size(),64u);
    for(const auto& entry:fixtures().at("fp4_boundaries")) {
        const float input=q::from_bits(entry.at("input_f32_bits"));uint8_t encoded=255;
        REQUIRE_EQ(q::e2m1_rne(input,&encoded),RAD_OK);CHECK_EQ(encoded,entry.at("code").get<uint8_t>());
        float decoded=0;REQUIRE_EQ(q::e2m1_value(encoded,&decoded),RAD_OK);CHECK_EQ(q::bits(decoded),entry.at("decoded_f32_bits").get<uint32_t>());
    }
    for(unsigned code=0;code<16;++code) {
        float value=0;uint8_t back=255;REQUIRE_EQ(q::e2m1_value(uint8_t(code),&value),RAD_OK);
        REQUIRE_EQ(q::e2m1_rne(value,&back),RAD_OK);CHECK_EQ(back,code);
    }
    float unchanged=1234;CHECK_EQ(q::e2m1_value(16,&unchanged),RAD_E_FORMAT);CHECK_EQ(unchanged,1234.f);
}
TEST(deepseek41_qdq_bf16_boundaries_and_e8m0_decode_match_actual_cpu_torch_formats) {
    for(const auto& entry:fixtures().at("bf16_boundaries")) {
        uint16_t output=0xabcd;const int status=q::bf16_rne(q::from_bits(entry.at("input_f32_bits")),&output);
        if(entry.at("candidate_accepts").get<bool>()){REQUIRE_EQ(status,RAD_OK);CHECK_EQ(output,entry.at("output_bf16_bits").get<uint16_t>());}
        else{CHECK(status<0);CHECK_EQ(output,0xabcd);}
    }
    for(const auto& entry:fixtures().at("e8m0_codes")) {
        float decoded=1234;const int status=q::e8m0_value(entry.at("code"),&decoded);
        if(entry.at("finite").get<bool>()){REQUIRE_EQ(status,RAD_OK);CHECK_EQ(q::bits(decoded),entry.at("value_f32_bits").get<uint32_t>());}
        else{CHECK(status<0);CHECK_EQ(decoded,1234.f);}
    }
    for(float bad:{INFINITY,-INFINITY,NAN}) {
        uint8_t code=0xab;CHECK(q::e4m3_rne(bad,&code)<0);CHECK_EQ(code,0xab);CHECK(q::e2m1_rne(bad,&code)<0);CHECK_EQ(code,0xab);
    }
}
TEST(deepseek41_qdq_all_profiles_match_explicitly_interpreted_torch_and_codebook_rows_transactionally) {
    REQUIRE_EQ(fixtures().at("rows").size(),30u);CHECK(q::supported_f32_environment());
    for(const auto& entry:fixtures().at("rows")) {
        const auto kind=profile(entry.at("profile"));q::Geometry geometry;REQUIRE_EQ(q::geometry(kind,&geometry),RAD_OK);
        const auto input=entry.at("input_bf16_bits").get<std::vector<uint16_t>>();const auto before=input;
        Output output(input.size(),input.size()/geometry.width);const auto unchanged=output;
        const int status=q::row(kind,input,output.view());CHECK(input==before);
        if(entry.at("candidate_status")=="ok") {
            REQUIRE_EQ(status,RAD_OK);const auto values=entry.at("output_bf16_bits").get<std::vector<uint16_t>>();
            const auto codes=entry.at("logical_codes").get<std::vector<uint8_t>>(),scale_codes=entry.at("scale_codes").get<std::vector<uint8_t>>();
            const auto scales=entry.at("scale_f32_bits").get<std::vector<uint32_t>>();
            for(size_t i=0;i<input.size();++i){CHECK_EQ(output.values[i+1],values[i]);CHECK_EQ(output.codes[i+1],codes[i]);}
            for(size_t i=0;i<scales.size();++i){CHECK_EQ(q::bits(output.scales[i+1]),scales[i]);CHECK_EQ(output.scale_codes[i+1],scale_codes[i]);}
            Output from_float(input.size(),input.size()/geometry.width);std::vector<float> promoted;
            for(auto raw:input)promoted.push_back(q::bf16_value(raw));
            REQUIRE_EQ(q::row_from_f32(kind,promoted,from_float.view()),RAD_OK);CHECK(output.same(from_float));
        } else {CHECK(status<0);CHECK(output.same(unchanged));}
        output.guards();
    }
}
TEST(deepseek41_qdq_index_half_min_normal_requires_gradual_underflow_not_ftz_or_daz) {
    std::array<uint16_t,32> input{};input[0]=0x0040;input[1]=0x8040;
    q::GroupResult result;REQUIRE_EQ(q::group(q::Profile::index_fp4,input,&result),RAD_OK);
    CHECK_EQ(q::bits(result.scale),0x00800000u);CHECK_EQ(result.scale_code,1u);
    CHECK_EQ(result.codes[0],1u);CHECK_EQ(result.codes[1],9u);
    CHECK_EQ(result.bf16[0],0x0040);CHECK_EQ(result.bf16[1],0x8040);
    for(auto kind:{q::Profile::window_fp8,q::Profile::compressed_fp4,q::Profile::index_fp4}) {
        q::Geometry geometry;REQUIRE_EQ(q::geometry(kind,&geometry),RAD_OK);std::vector<uint16_t> zeros(geometry.width);
        for(size_t i=1;i<zeros.size();i+=2)zeros[i]=0x8000;
        REQUIRE_EQ(q::group(kind,zeros,&result),RAD_OK);CHECK(result.scale>0);
        for(size_t i=0;i<zeros.size();++i)CHECK_EQ(result.bf16[i],zeros[i]);
    }
}
TEST(deepseek41_qdq_exact_ieee_scale_bit_formula_and_overflow_bounds_are_explicit) {
    for(int exponent=-126;exponent<=127;++exponent) {
        float value=0;uint8_t code=0;REQUIRE_EQ(q::pow2_normal(exponent,&value,&code),RAD_OK);
        CHECK_EQ(code,exponent+127);int rounded=0;REQUIRE_EQ(q::log2_ceil_bits(value,&rounded),RAD_OK);CHECK_EQ(rounded,exponent);
        if(exponent<127){REQUIRE_EQ(q::log2_ceil_bits(std::nextafter(value,INFINITY),&rounded),RAD_OK);CHECK_EQ(rounded,exponent+1);}
    }
    int exact_source=0;REQUIRE_EQ(q::log2_ceil_bits(q::from_bits(1),&exact_source),RAD_OK);CHECK_EQ(exact_source,-126);
    float unchanged=1234.f;CHECK(q::pow2_normal(-127,&unchanged)<0);CHECK(q::pow2_normal(128,&unchanged)<0);CHECK_EQ(unchanged,1234.f);
    for(float bad:{0.f,-1.f,INFINITY,NAN}){int out=1234;CHECK(q::log2_ceil_bits(bad,&out)<0);CHECK_EQ(out,1234);}
}
TEST(deepseek41_qdq_late_nonfinite_scale_and_output_overflow_refusals_preserve_every_output) {
    for(auto kind:{q::Profile::window_fp8,q::Profile::compressed_fp4,q::Profile::index_fp4}) {
        q::Geometry geometry;REQUIRE_EQ(q::geometry(kind,&geometry),RAD_OK);
        for(uint16_t invalid:{uint16_t(0x7f80),uint16_t(0xff80),uint16_t(0x7fc1),uint16_t(0x7f7f)}) {
            std::vector<uint16_t> input(geometry.width*2,0x3f80);input.back()=invalid;
            Output out(input.size(),2);const auto before=out;CHECK(q::row(kind,input,out.view())<0);CHECK(out.same(before));
        }
        std::vector<float> input(geometry.width*2,1.f);input.back()=q::from_bits(0x7f7fffffu);
        Output out(input.size(),2);const auto before=out;CHECK(q::row_from_f32(kind,input,out.view())<0);CHECK(out.same(before));
    }
    std::vector<uint16_t> scale_overflow(32,0x3f80);scale_overflow.back()=0x4580; // finite4096
    Output out(32,2);const auto before=out;
    CHECK(q::row(q::Profile::compressed_fp4,scale_overflow,out.view())<0);CHECK(out.same(before));
}
TEST(deepseek41_qdq_metadata_alias_and_shape_refusals_are_out_of_place_and_bounded) {
    std::vector<uint16_t> input(64,0x3f80);Output output(64,2);const auto before=output;
    auto out=output.view();out.bf16=input;CHECK(q::row(q::Profile::window_fp8,input,out)<0);CHECK(output.same(before));
    out=output.view();out.codes=std::span(reinterpret_cast<uint8_t*>(out.bf16.data()),64);
    CHECK(q::row(q::Profile::window_fp8,input,out)<0);CHECK(output.same(before));
    out=output.view();out.scales=std::span(reinterpret_cast<float*>(output.values.data()),2);
    CHECK(q::row(q::Profile::window_fp8,input,out)<0);CHECK(output.same(before));
    out=output.view();out.codes=out.codes.first(63);CHECK(q::row(q::Profile::window_fp8,input,out)<0);CHECK(output.same(before));
    CHECK(q::row(q::Profile::window_fp8,std::span(input).first(63),output.view())<0);CHECK(output.same(before));
    CHECK(q::row(static_cast<q::Profile>(999),input,output.view())<0);CHECK(output.same(before));
    const auto* wrap=reinterpret_cast<const uint16_t*>(UINTPTR_MAX-1);
    CHECK(q::row(q::Profile::window_fp8,std::span(wrap,64),output.view())<0);CHECK(output.same(before));
    std::vector<uint16_t> too_wide(q::max_row_width+1,0x3f80);
    CHECK(q::row(q::Profile::window_fp8,too_wide,output.view())<0);CHECK(output.same(before));
    std::vector<uint16_t> maximum(q::max_row_width,0x3f80),result(q::max_row_width);
    REQUIRE_EQ(q::row(q::Profile::window_fp8,maximum,{result}),RAD_OK);
}
TEST(deepseek41_qdq_refuses_non_rne_or_flush_environment_without_changing_it) {
    struct Rounding {int original=std::fegetround();~Rounding(){std::fesetround(original);}} rounding;
    std::vector<uint16_t> input(32,0x3f80);Output output(32,1);const auto before=output;
    if(std::fesetround(FE_UPWARD)==0) {
        CHECK_EQ(q::row(q::Profile::window_fp8,input,output.view()),RAD_E_UNSUPPORTED);CHECK(output.same(before));
        CHECK_EQ(std::fegetround(),FE_UPWARD);
    }
    std::fesetround(rounding.original);
#if defined(__SSE__)
    struct Mxcsr {unsigned original=_mm_getcsr();~Mxcsr(){_mm_setcsr(original);}} mxcsr;
    for(unsigned flush:{0x8000u,0x0040u}) {
        _mm_setcsr(mxcsr.original|flush);
        CHECK_EQ(q::row(q::Profile::index_fp4,input,output.view()),RAD_E_UNSUPPORTED);CHECK(output.same(before));
    }
#endif
}
RAD_TEST_MAIN()
