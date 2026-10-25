/* Real unchanged official Compressor/RMSNorm Torch CPU bodies, supplied tiny
 * F.linear projections. Independent scalar C++ projection/state comparison;
 * this does not execute official Linear/TileLang, a graph, GPU or model payload. */
#include "rad_test.h"
#include "../arch/common/rad_ds41_ref.h"
#include <nlohmann/json.hpp>
#include <bit>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>

namespace ref=rad::deepseek41::ref;
namespace {
using Json=nlohmann::json;
const Json& oracle(){
    static const Json value=[] {
        const auto path=std::string(RAD_GOLDEN_FIXTURES)+"/ds41-official-compressor-cpu.json";
        if(std::filesystem::file_size(path)>500000)throw std::runtime_error("compressor CPU fixture exceeds500KB");
        std::ifstream file(path);if(!file)throw std::runtime_error("missing official CPU compressor fixture");
        return Json::parse(file);
    }();return value;
}
struct Tensor {
    std::string dtype;std::vector<size_t> shape;std::vector<uint8_t> raw;std::vector<float> values;
    explicit Tensor(const Json& tensor):dtype(tensor.at("dtype")),shape(tensor.at("shape").get<std::vector<size_t>>()),raw(tensor.at("raw_u8").get<std::vector<uint8_t>>()){
        if(dtype!="float32"&&dtype!="bfloat16")throw std::runtime_error("unsupported compressor fixture dtype");
        if(shape.empty()||shape.size()>3)throw std::runtime_error("compressor fixture rank outside cap");
        size_t count=1;
        for(size_t dim:shape){if(dim>4096||count>4096/(dim?dim:1))throw std::runtime_error("compressor fixture element cap");count*=dim;}
        const size_t width=dtype=="float32"?4:2;
        if(raw.size()!=count*width)throw std::runtime_error("compressor raw extent mismatch");
        values.resize(count);
        for(size_t i=0;i<count;++i){
            uint32_t bits=0;for(size_t byte=0;byte<width;++byte)bits|=uint32_t(raw[i*width+byte])<<(8*byte);
            values[i]=std::bit_cast<float>(dtype=="bfloat16"?bits<<16:bits);
            if(!std::isfinite(values[i]))throw std::runtime_error("unexpected nonfinite semantic compressor tensor");
        }
    }
};
int ordered(uint16_t value){return value&0x8000u?32768-int(value&0x7fffu):32768+int(value);}
void exact(ref::Floats actual,const Tensor& expected){
    REQUIRE_EQ(actual.size(),expected.values.size());
    for(size_t i=0;i<actual.size();++i)CHECK_EQ(std::bit_cast<uint32_t>(actual[i]),std::bit_cast<uint32_t>(expected.values[i]));
}
void bf16_baseline(ref::Floats actual,const Tensor& expected){
    REQUIRE_EQ(expected.dtype,"bfloat16");REQUIRE_EQ(actual.size(),expected.values.size());
    for(size_t i=0;i<actual.size();++i){
        const float a=actual[i],b=expected.values[i];REQUIRE(std::isfinite(a));
        const uint32_t bits=std::bit_cast<uint32_t>(a);CHECK_EQ(bits&0xffffu,0u);
        const auto ah=uint16_t(bits>>16),bh=uint16_t(std::bit_cast<uint32_t>(b)>>16);
        if(b==0.f){CHECK_EQ(a,0.f);CHECK_EQ(std::signbit(a),std::signbit(b));}
        else CHECK(std::abs(ordered(ah)-ordered(bh))<=1);
    }
}
std::vector<float> linear(const Tensor& x,const Tensor& weight,bool bf16){
    if(x.shape.size()!=3||x.shape[0]!=1||weight.shape.size()!=2||x.shape[2]!=weight.shape[1])throw std::runtime_error("projection fixture shape differs");
    const size_t rows=x.shape[1],outputs=weight.shape[0],inputs=weight.shape[1];std::vector<float> y(rows*outputs);
    // This is independent scalar FP32 arithmetic, not a call to ref::linear or
    // official Linear. Chosen dyadic BF16 fixtures make these dot sums exact.
    for(size_t row=0;row<rows;++row)for(size_t out=0;out<outputs;++out){
        float sum=0;for(size_t k=0;k<inputs;++k)sum=sum+x.values[row*inputs+k]*weight.values[out*inputs+k];
        y[row*outputs+out]=ref::rounded(sum,bf16?ref::Precision::bf16:ref::Precision::f32);
    }
    return y;
}
}

TEST(ds41_official_compressor_fixture_records_exact_cpu_execution_and_explicit_projection_boundary){
    const auto& j=oracle();CHECK_EQ(j.at("schema").get<std::string>(),"ds41-official-compressor-cpu-v1");
    CHECK_EQ(j.at("official_revision").get<std::string>(),"1bae0e1088993b9b82de08f59de5be5a34e42170");
    CHECK_EQ(j.at("official_model_sha256").get<std::string>(),"e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d");
    for(const char* flag:{"official_linear_executed","tilelang_executed","gpu_execution","model_payload_read","model_quality_or_gpu_bit_parity_established"})CHECK(!j.at(flag).get<bool>());
    for(const char* method:{"Compressor.forward","RMSNorm.forward"}){
        const auto& body=j.at("executed_functions").at(method);CHECK(body.at("body_unchanged").get<bool>());CHECK(body.at("removed_decorators").empty());
        CHECK_EQ(body.at("body_ast_sha256").get<std::string>().size(),64u);
    }
    CHECK_EQ(j.at("environment").at("torch").get<std::string>(),"2.14.1+cpu");
    CHECK(j.at("environment").at("torch_hip").is_null());CHECK(j.at("environment").at("torch_cuda").is_null());
    CHECK_EQ(j.at("owners").size(),5u);CHECK_EQ(j.at("execution_steps").size(),25u);
}

TEST(ds41_official_compressor_supplied_projection_dots_and_premature_bf16_pending_witness){
    for(const auto& step:oracle().at("execution_steps")){
        const auto& owner=oracle().at("owners").at(step.at("owner").get<std::string>());const int ratio=owner.at("ratio");
        const Tensor x(step.at("input")),weight(owner.at("source_weight_bf16")),expected(step.at("projected_kv"));
        CHECK_EQ(step.at("observed_projection_input_dtype").get<std::string>(),ratio==2?"float32":"bfloat16");
        exact(linear(x,weight,ratio==1),expected);
        if(ratio==2){const Tensor gate(owner.at("source_gate_bf16")),score(step.at("projected_score"));exact(linear(x,gate,false),score);}
    }
    const auto& witness=oracle().at("premature_bf16_pending_witness");const Tensor correct(witness.at("correct_f32_pending_kv")),wrong(witness.at("incorrect_bf16_then_f32_pending_kv"));
    REQUIRE_EQ(correct.values.size(),wrong.values.size());size_t differences=0;
    for(size_t i=0;i<correct.values.size();++i){CHECK_EQ(ref::rounded(correct.values[i],ref::Precision::bf16),wrong.values[i]);differences+=std::bit_cast<uint32_t>(correct.values[i])!=std::bit_cast<uint32_t>(wrong.values[i]);}
    CHECK(differences>0);CHECK_EQ(differences,witness.at("differing_values").get<size_t>());
}

TEST(ds41_official_compressor_c1_c2_odd_prefill_decode_and_independent_owners_match_reference){
    std::map<std::string,std::unique_ptr<ref::Compressor>> references;size_t partial_steps=0,emitted_steps=0;
    for(auto it=oracle().at("owners").begin();it!=oracle().at("owners").end();++it){
        const Tensor gain(it.value().at("norm_gain"));
        references.emplace(it.key(),std::make_unique<ref::Compressor>(it.value().at("ratio").get<size_t>(),gain.values,
            it.value().at("epsilon").get<float>(),it.key(),ref::Precision::bf16));
    }
    for(const auto& step:oracle().at("execution_steps")){
        const auto name=step.at("owner").get<std::string>();auto& reference=*references.at(name);const auto& owner=oracle().at("owners").at(name);
        const size_t ratio=owner.at("ratio"),dim=owner.at("dim");const int64_t start=step.at("start_position");
        const Tensor x(step.at("input")),weight(owner.at("source_weight_bf16"));const auto kv=linear(x,weight,ratio==1);
        std::vector<float> scores;if(ratio==2)scores=linear(x,Tensor(owner.at("source_gate_bf16")),false);
        REQUIRE_EQ(reference.next_position(),start);std::vector<float> emitted;
        for(size_t row=0;row<x.shape[1];++row){
            std::vector<float> latent;int64_t first=-1;
            REQUIRE_EQ(reference.push(ref::Floats(kv).subspan(row*dim,dim),ratio==2?ref::Floats(scores).subspan(row*dim,dim):ref::Floats{},&latent,&first),RAD_OK);
            const bool complete=(size_t(start)+row+1)%ratio==0;
            CHECK_EQ(first,complete?start+int64_t(row)+1-int64_t(ratio):-1);CHECK_EQ(latent.empty(),!complete);
            emitted.insert(emitted.end(),latent.begin(),latent.end());
        }
        if(step.at("output").is_null()){CHECK(emitted.empty());++partial_steps;}
        else{bf16_baseline(emitted,Tensor(step.at("output")));++emitted_steps;}
        const auto saved=reference.snapshot();const auto& pending=step.at("semantic_pending");
        CHECK_EQ(saved.next_position,pending.at("next_position").get<int64_t>());CHECK_EQ(saved.pending,pending.at("pending").get<size_t>());
        if(ratio==2){
            const Tensor pkv(pending.at("kv")),pgate(pending.at("score"));exact(saved.kv,pkv);exact(saved.scores,pgate);
            // Official buffers retain irrelevant old completed slots; compare
            // precisely the live pending prefix, never demand stale-byte parity.
            const auto full=step.at("official_state_after").at("kv").at("raw_u8").get<std::vector<uint8_t>>();
            REQUIRE(full.size()>=pkv.raw.size());CHECK(std::equal(pkv.raw.begin(),pkv.raw.end(),full.begin()));
        }else{CHECK(saved.kv.empty());CHECK(saved.scores.empty());}
        ref::Compressor restored(ratio,Tensor(owner.at("norm_gain")).values,owner.at("epsilon").get<float>(),name,ref::Precision::bf16);
        REQUIRE_EQ(restored.restore(saved),RAD_OK);CHECK_EQ(restored.next_position(),reference.next_position());
    }
    CHECK(partial_steps>0);CHECK(emitted_steps>0);
}
RAD_TEST_MAIN()
