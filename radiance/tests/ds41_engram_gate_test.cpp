#include "rad_test.h"
#include "../libds41/ds41_engram_gate_fixture.h"
#include "../libds41/ds41_test_scratch.h"
#include <nlohmann/json.hpp>
#include <bit>
#include <fstream>
#include <limits>

namespace eg=rad::ds41::engram_gate;
namespace fix=eg::testing;
namespace ref=rad::deepseek41::engram;
namespace {
const nlohmann::json& golden() {
    static const auto data=[] {
        std::ifstream file(std::string(RAD_GOLDEN_FIXTURES)+"/deepseek41-engram.json");
        if(!file)throw std::runtime_error("missing independent Engram fixture");
        return nlohmann::json::parse(file);
    }();return data;
}
void compare(fix::Fixture& fixture) {
    auto expected=fixture;REQUIRE_EQ(fix::reference(expected),RAD_OK);const auto before=fixture.buffers;
    REQUIRE_EQ(fixture.run(),RAD_OK);CHECK_EQ(fixture.flag[0],RAD_OK);
    for(int i=0;i<eg::operand_count;++i) {
        if(i==eg::output||(i==eg::gate_output&&fixture.gates))CHECK(fixture.buffers[i].words==expected.buffers[i].words);
        else CHECK(fixture.buffers[i].words==before[i].words);
    }
    CHECK_EQ(fixture.flag[1],912);CHECK_EQ(fixture.flag[2],913);CHECK_EQ(fixture.flag[3],914);
}
void reject_payload(fix::Fixture& fixture) {
    const auto before=fixture.buffers;CHECK(fixture.run()<0);CHECK(fixture.flag[0]<0);
    for(int i=0;i<eg::operand_count;++i)CHECK(fixture.buffers[i].words==before[i].words);
    CHECK_EQ(fixture.flag[1],912);CHECK_EQ(fixture.flag[2],913);CHECK_EQ(fixture.flag[3],914);
}
void reject_args(fix::Fixture& fixture,RadArgs& args) {
    const auto before=fixture.buffers;const auto status=fixture.flag;CHECK(eg::emulate(&args)<0);
    for(int i=0;i<eg::operand_count;++i)CHECK(fixture.buffers[i].words==before[i].words);
    CHECK(fixture.flag==status);
}
}

TEST(ds41_engram_gate_all_dtypes_pitches_optional_slots_and_primary_shape_match_reference) {
    for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(int pitch=0;pitch<3;++pitch)
        for(bool mask:{false,true})for(bool gates:{false,true}) {
            fix::Fixture fixture(dtype,pitch,mask,gates);compare(fixture);
        }
    for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16}) {
        fix::Fixture primary(dtype,0,true,true,5120,4,1);compare(primary);
        fix::Fixture maximum_copies(dtype,2,true,true,13,8,2);compare(maximum_copies);
        fix::Fixture one(dtype,1,false,false,1,1,1);compare(one);
    }
}
TEST(ds41_engram_gate_matches_existing_official_forward_goldens_and_masked_tokens) {
    REQUIRE_EQ(golden().at("gate_cases").size(),4u);
    for(const auto& data:golden().at("gate_cases")) {
        const auto dtype=data.at("dtype")=="bf16"?RAD_BF16:RAD_F32;
        fix::Fixture fixture(dtype,2,true,true,4,2,3);
        const int slots[]{eg::input,eg::projected_kv,eg::q_weight,eg::k_weight};
        const char* keys[]{"x","projected_kv","q_weight","k_weight"};
        for(int i=0;i<4;++i) {
            const auto values=data.at(keys[i]).get<std::vector<float>>();
            for(size_t j=0;j<values.size();++j)fixture.buffers[slots[i]].set(int64_t(j),values[j]);
        }
        const auto mask=data.at("token_mask").get<std::vector<int32_t>>();
        for(size_t i=0;i<mask.size();++i)fixture.buffers[eg::mask_operand].id(int64_t(i),mask[i]);
        REQUIRE_EQ(fixture.run(),RAD_OK);
        const auto expected=data.at("output").get<std::vector<float>>(),gates=data.at("gates").get<std::vector<float>>();
        for(size_t i=0;i<expected.size();++i) {
            if(dtype==RAD_BF16)CHECK_EQ(fixture.buffers[eg::output].get(int64_t(i)),expected[i]);
            else CHECK_NEAR(fixture.buffers[eg::output].get(int64_t(i)),expected[i],3e-6);
        }
        for(size_t i=0;i<gates.size();++i)CHECK_NEAR(fixture.buffers[eg::gate_output].get(int64_t(i)),gates[i],2e-6);
        for(int i=8;i<16;++i)CHECK_EQ(fixture.buffers[eg::output].get(i),fixture.buffers[eg::input].get(i));
    }
}
TEST(ds41_engram_gate_signed_zero_clamp_and_negative_dot_are_not_sign_times_root) {
    for(float clamp:{1e-6f,.25f,4.f}) {
        CHECK(eg::from_dot(0.f,clamp)>.5f);CHECK(eg::from_dot(-0.f,clamp)<.5f);
        CHECK_EQ(eg::from_dot(0.f,clamp),ref::gate_from_dot(0.f,clamp));
        CHECK_EQ(eg::from_dot(-0.f,clamp),ref::gate_from_dot(-0.f,clamp));
        for(float dot:{-100.f,-4.f,-1e-8f,1e-8f,4.f,100.f})CHECK_EQ(eg::from_dot(dot,clamp),ref::gate_from_dot(dot,clamp));
    }
    fix::Fixture zeros(RAD_F32,1,false,true,4,2,1);
    for(int64_t i=0;i<zeros.buffers[eg::input].count();++i)zeros.buffers[eg::input].set(i,-0.f);
    compare(zeros);for(int i=0;i<2;++i)CHECK(zeros.buffers[eg::gate_output].get(i)>.5f);
    fix::Fixture clamp(RAD_F32,0,false,true,1,1,1);
    clamp.buffers[eg::input].set(0,0.f);clamp.params[5].dval=.25;
    REQUIRE_EQ(clamp.run(),RAD_OK);CHECK_EQ(clamp.buffers[eg::gate_output].get(0),ref::gate_from_dot(0,.25f));
}
TEST(ds41_engram_gate_nonfinite_late_masked_inputs_and_bad_mask_are_batch_atomic) {
    for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(int slot:{eg::input,eg::projected_kv,eg::q_weight,eg::k_weight})
        for(float bad:{NAN,INFINITY,-INFINITY}) {
            fix::Fixture fixture(dtype,2);fixture.buffers[slot].set(fixture.buffers[slot].count()-1,bad);reject_payload(fixture);
        }
    for(int32_t mask:{-1,2,INT32_MAX}) {
        fix::Fixture fixture;fixture.buffers[eg::mask_operand].id(2,mask);reject_payload(fixture);
    }
    fix::Fixture masked;masked.buffers[eg::mask_operand].id(2,0);
    masked.buffers[eg::input].set(masked.buffers[eg::input].count()-1,NAN);reject_payload(masked);
}
TEST(ds41_engram_gate_finite_overflow_is_refused_before_residual_or_gate_publication) {
    fix::Fixture norm;norm.buffers[eg::input].set(norm.buffers[eg::input].count()-1,std::numeric_limits<float>::max());reject_payload(norm);
    fix::Fixture dot;dot.buffers[eg::q_weight].set(dot.dim*dot.copies-1,std::numeric_limits<float>::max());
    dot.buffers[eg::k_weight].set(dot.dim*dot.copies-1,std::numeric_limits<float>::max());reject_payload(dot);
    fix::Fixture tiny(RAD_F32,0,false,true);tiny.params[4].dval=std::numeric_limits<float>::denorm_min();
    for(int64_t i=0;i<tiny.buffers[eg::input].count();++i)tiny.buffers[eg::input].set(i,0);
    for(int64_t i=0;i<tiny.buffers[eg::projected_kv].count();++i)tiny.buffers[eg::projected_kv].set(i,0);
    reject_payload(tiny);
    fix::Fixture half(RAD_F16,0,false,true,1,1,2);
    half.buffers[eg::input].set(1,60000);half.buffers[eg::projected_kv].set(2,1);half.buffers[eg::projected_kv].set(3,60000);
    reject_payload(half);
    // Only the last job has a finite sum large enough to overflow after adding
    // the separately finite epsilon. Earlier jobs must not publish any outputs.
    for(bool key:{false,true}) {
        fix::Fixture denominator(RAD_F32,2,true,true,1,2,3);
        denominator.params[4].dval=std::numeric_limits<float>::max();
        const int64_t at=key?(denominator.tokens-1)*(denominator.copies+1)+(denominator.copies-1):
                              denominator.tokens*denominator.copies-1;
        denominator.buffers[key?eg::projected_kv:eg::input].set(at,1e19f);
        volatile float large=1e19f;const float square=large*large;
        CHECK(std::isfinite(square));
        CHECK(!std::isfinite(square+std::numeric_limits<float>::max()));
        reject_payload(denominator);
    }
}
TEST(ds41_engram_gate_descriptor_dtype_pitch_alias_and_optional_contract_are_checked_without_writes) {
    for(int fault=0;fault<17;++fault) {
        fix::Fixture fixture;auto args=fixture.args();
        if(fault==0)fixture.tensors[eg::output].data=fixture.tensors[eg::input].data;
        if(fault==1)fixture.tensors[eg::gate_output].data=fixture.tensors[eg::output].data;
        if(fault==2)fixture.tensors[eg::input].stride[0]=INT64_MAX;
        if(fault==3)fixture.tensors[eg::input].stride[2]=0;
        if(fault==4)fixture.tensors[eg::q_weight].dtype=RAD_BF16;
        if(fault==5)fixture.tensors[eg::mask_operand].dtype=RAD_U32;
        if(fault==6)fixture.tensors[eg::output].dtype=RAD_F16;
        if(fault==7)args.n_t=6;
        if(fault==8)args.scratch_bytes=3;
        if(fault==9)args.scratch=reinterpret_cast<void*>(UINTPTR_MAX-3);
        if(fault==10)args.scratch=fixture.tensors[eg::input].data;
        if(fault==11)fixture.params[0].ival=0;
        if(fault==12)fixture.params[1].ival=9;
        if(fault==13)fixture.params[2].ival=eg::max_dimension+1;
        if(fault==14)fixture.params[6].ival=0; // present mask contradicts absent metadata
        if(fault==15)fixture.params[7].ival=0;
        if(fault==16)fixture.tensors[eg::input].data=reinterpret_cast<void*>(UINTPTR_MAX-3);
        reject_args(fixture,args);
    }
    for(int key:{4,5})for(double bad:{0.,-1.,double(NAN),double(INFINITY),1e-100}) {
        fix::Fixture fixture;auto args=fixture.args();fixture.params[size_t(key)].dval=bad;reject_args(fixture,args);
    }
    fix::Fixture dtype;auto args=dtype.args();dtype.params[3].sval="i32";reject_args(dtype,args);
    CHECK_EQ(ds41_engram_gate_scratch(nullptr),rad::ds41::gpu_scratch_bytes<eg::Plan>());
}
TEST(ds41_engram_gate_full_gpu_plan_span_is_disjoint_before_any_staging) {
    fix::Fixture fixture;auto args=fixture.args();eg::Plan plan;REQUIRE_EQ(eg::validate(&args,&plan),RAD_OK);
    alignas(16) std::array<uint8_t,size_t(rad::ds41::gpu_scratch_bytes<eg::Plan>())> scratch{};
    args.scratch=scratch.data();args.scratch_bytes=scratch.size();
    CHECK_EQ(rad::ds41::validate_plan_storage(&args,plan),RAD_OK);
    auto bad=plan;bad.t[eg::output].data=scratch.data()+16;bad.spans[eg::output]=4;
    CHECK_EQ(rad::ds41::validate_plan_storage(&args,bad),RAD_E_INVAL);
    args.scratch_bytes=eg::scratch_bytes;CHECK_EQ(rad::ds41::validate_plan_storage(&args,plan),RAD_E_INVAL);
    CHECK_EQ(eg::scratch_bytes,4);CHECK_EQ(eg::status_offset,0);
}
TEST(ds41_engram_gate_host_wrapper_ignores_unreadable_device_scratch_and_returns_payload_failure) {
    fix::Fixture fixture;auto args=fixture.args();args.scratch=reinterpret_cast<void*>(uintptr_t(1));args.scratch_bytes=INT64_MAX;
    REQUIRE_EQ(ds41_ref_engram_gate(&args,nullptr),RAD_OK);CHECK_EQ(fixture.flag[0],911);
    const auto before=fixture.buffers;fixture.buffers[eg::input].set(fixture.buffers[eg::input].count()-1,NAN);
    const auto bad=fixture.buffers;args=fixture.args();args.scratch=reinterpret_cast<void*>(uintptr_t(1));
    CHECK(ds41_ref_engram_gate(&args,nullptr)<0);
    for(int i=0;i<eg::operand_count;++i)CHECK(fixture.buffers[i].words==bad[i].words);
    CHECK(fixture.buffers[eg::output].words==before[eg::output].words);CHECK_EQ(fixture.flag[0],911);
    CHECK_EQ(ds41_ref_engram_gate(nullptr,nullptr),RAD_E_INVAL);
}
RAD_TEST_MAIN()
