/* Manual native-HIP operator gate. This is not a Ctx/AQL/graph integration test.
 * No model weights, files, services or cache lifecycle. One visible gfx1201 only. */
#include <hip/hip_runtime.h>
#include "ds41_engram_gate_fixture.h"
#include "ds41_test_scratch.h"
#include <cstdio>
#include <cstring>
#include <memory>
#include <limits>
#include <span>
#include <stdexcept>

namespace eg=rad::ds41::engram_gate;
namespace fix=eg::testing;
namespace {
constexpr size_t expected_payload_cases=136,expected_metadata_cases=10;
void require(bool yes,const char* message){if(!yes)throw std::runtime_error(message);}
void check(hipError_t rc){if(rc!=hipSuccess)throw std::runtime_error(hipGetErrorString(rc));}
template<class V>std::vector<uint8_t> bytes(const V& source) {
    std::vector<uint8_t> out(source.size()*sizeof(*source.data()));
    if(!out.empty())std::memcpy(out.data(),source.data(),out.size());
    return out;
}
struct Device {
    static constexpr size_t guard=64;void* base=nullptr;size_t size=0;
    explicit Device(const std::vector<uint8_t>& source):size(source.size()) {
        require(size>0,"empty device fixture");check(hipMalloc(&base,size+guard*2));
        auto rc=hipMemset(base,0xA7,size+guard*2);
        if(rc==hipSuccess)rc=hipMemcpy(data(),source.data(),size,hipMemcpyHostToDevice);
        if(rc!=hipSuccess){static_cast<void>(hipFree(base));base=nullptr;check(rc);}
    }
    Device(const Device&)=delete;Device& operator=(const Device&)=delete;
    ~Device(){if(base)static_cast<void>(hipFree(base));}
    void* data()const{return static_cast<uint8_t*>(base)+guard;}
    std::vector<uint8_t> read()const {
        std::vector<uint8_t> all(size+guard*2);check(hipMemcpy(all.data(),base,all.size(),hipMemcpyDeviceToHost));
        for(size_t i=0;i<guard;++i)require(all[i]==0xA7&&all[guard+size+i]==0xA7,"Engram device allocation redzone changed");
        return {all.begin()+guard,all.end()-guard};
    }
};
void promote(fix::Fixture& fixture) {
    for(int slot:{eg::input,eg::projected_kv,eg::output}) {
        auto& b=fixture.buffers[slot];const auto values=b.values();b.t.dtype=RAD_F32;
        for(size_t i=0;i<values.size();++i)b.set(int64_t(i),values[i]);
    }
    fixture.dtype=RAD_F32;fixture.params[3].sval="f32";fixture.refresh();
}
uint16_t halfword(const std::vector<uint8_t>& source,size_t offset) {
    uint16_t out;std::memcpy(&out,source.data()+offset,2);return out;
}
int ordered(uint16_t code){return code&0x8000u?32768-int(code&0x7fffu):32768+int(code);}
float scalar(const std::vector<uint8_t>& source,uint32_t dtype,size_t offset) {
    if(dtype==RAD_F32){float out;std::memcpy(&out,source.data()+offset*4,4);return out;}
    const auto h=halfword(source,offset*2);return dtype==RAD_F16?rad_f16_to_f32(h):rad_bf16_to_f32(h);
}
void output(const fix::Buffer& initial,const fix::Buffer& oracle,const fix::Buffer& raw,
             const std::vector<uint8_t>& actual) {
    const auto before=bytes(initial.words);std::vector<uint8_t> touched(actual.size(),0);
    const int width=rad_dtype_bits(initial.t.dtype)/8;
    for(int64_t i=0;i<initial.count();++i) {
        const size_t at=size_t(initial.offset(i));const float want=oracle.get(i),unrounded=raw.get(i),got=scalar(actual,initial.t.dtype,at);
        const float tolerance=3e-6f+3e-5f*std::fabs(unrounded);
        require(std::isfinite(got)&&std::isfinite(want)&&std::isfinite(unrounded),"nonfinite Engram output");
        if(initial.t.dtype==RAD_F32)require(std::fabs(got-want)<=tolerance,"Engram F32 output differs from independent CPU reference");
        else {
            const uint16_t encoded=initial.t.dtype==RAD_F16?rad_f32_to_f16(unrounded):rad_f32_to_bf16(unrounded);
            const uint16_t received=halfword(actual,at*2);
            if(received!=encoded) {
                require(std::abs(ordered(received)-ordered(encoded))<=1,"Engram output exceeds one storage rounding step");
                require(std::fabs(unrounded-.5f*(got+want))<=tolerance,"Engram storage differs away from rounding boundary");
            }
        }
        if(got==0&&want==0)require(std::signbit(got)==std::signbit(want),"Engram output zero sign differs");
        for(int byte=0;byte<width;++byte)touched[at*size_t(width)+size_t(byte)]=1;
    }
    for(size_t byte=0;byte<actual.size();++byte)if(!touched[byte])require(actual[byte]==before[byte],"Engram output pitch padding changed");
}
void run(fix::Fixture input,hipStream_t stream,int metadata=-1) {
    auto emulator=input;const int expected=emulator.run();auto oracle=input,raw=input;
    if(expected==RAD_OK) {
        require(fix::reference(oracle)==RAD_OK,"independent Engram reference refused valid fixture");
        promote(raw);require(fix::reference(raw)==RAD_OK,"unrounded independent Engram reference refused fixture");
    }
    const auto before=rad::ds41::testing::make_gpu_scratch<eg::Plan>(bytes(input.flag));
    size_t budget=before.size()+2*Device::guard;
    for(int i=0;i<eg::operand_count;++i)if(input.buffers[i].t.rank)budget+=input.buffers[i].words.size()*4+2*Device::guard;
    require(budget<=1048576,"Engram checker exceeded explicit1MiB device fixture budget");
    std::array<std::unique_ptr<Device>,eg::operand_count> device;
    for(int i=0;i<eg::operand_count;++i)if(input.buffers[i].t.rank)
        device[i]=std::make_unique<Device>(bytes(input.buffers[i].words));
    Device scratch(before);auto args=input.args();
    for(int i=0;i<eg::operand_count;++i)if(device[i])input.tensors[i].data=device[i]->data();
    args.scratch=scratch.data();args.scratch_bytes=rad::ds41::gpu_scratch_bytes<eg::Plan>();
    if(metadata==0)input.tensors[eg::output].data=input.tensors[eg::input].data;
    if(metadata==1)input.tensors[eg::gate_output].data=input.tensors[eg::output].data;
    if(metadata==2)input.tensors[eg::input].stride[0]=INT64_MAX;
    if(metadata==3)input.tensors[eg::q_weight].dtype=RAD_BF16;
    if(metadata==4)args.scratch_bytes=3;
    if(metadata==5)args.scratch=input.tensors[eg::input].data;
    if(metadata==6)input.params[6].ival=0;
    if(metadata==7)input.params[1].ival=9;
    if(metadata==8)input.tensors[eg::q_weight].data=static_cast<uint8_t*>(scratch.data())+16;
    if(metadata==9)args.scratch_bytes=rad::ds41::gpu_scratch_bytes<eg::Plan>()-1;
    const int submitted=ds41_engram_gate(&args,stream);
    require(metadata<0?submitted==RAD_OK:submitted<0,"Engram descriptor acceptance mismatch");
    check(hipStreamSynchronize(stream));const auto flag=scratch.read();
    require(rad::ds41::testing::gpu_scratch_guards<eg::Plan>(flag,before),"Engram scratch guards/staged Plan tail changed");
    int32_t status=0;std::memcpy(&status,flag.data(),4);
    require(metadata<0?status==expected:flag==before,"Engram status differs or descriptor refusal wrote scratch");
    for(int i=0;i<eg::operand_count;++i)if(device[i]) {
        const auto actual=device[i]->read();
        if(metadata>=0||expected!=RAD_OK||(i!=eg::output&&i!=eg::gate_output))
            require(actual==bytes(input.buffers[i].words),"Engram input changed or refusal wrote output");
        else output(input.buffers[i],oracle.buffers[i],raw.buffers[i],actual);
    }
}
enum class Payload {input_nan,kv_inf,q_nan,k_inf,mask_negative,mask_two,masked_nan,weight_overflow};
void corrupt(fix::Fixture& c,Payload p) {
    if(p==Payload::input_nan)c.buffers[eg::input].set(c.buffers[eg::input].count()-1,NAN);
    if(p==Payload::kv_inf)c.buffers[eg::projected_kv].set(c.buffers[eg::projected_kv].count()-1,INFINITY);
    if(p==Payload::q_nan)c.buffers[eg::q_weight].set(c.copies*c.dim-1,NAN);
    if(p==Payload::k_inf)c.buffers[eg::k_weight].set(c.copies*c.dim-1,-INFINITY);
    if(p==Payload::mask_negative)c.buffers[eg::mask_operand].id(c.tokens-1,-1);
    if(p==Payload::mask_two)c.buffers[eg::mask_operand].id(c.tokens-1,2);
    if(p==Payload::masked_nan) {
        c.buffers[eg::mask_operand].id(c.tokens-1,0);c.buffers[eg::input].set(c.buffers[eg::input].count()-1,NAN);
    }
    if(p==Payload::weight_overflow) {
        c.buffers[eg::q_weight].set(c.copies*c.dim-1,std::numeric_limits<float>::max());
        c.buffers[eg::k_weight].set(c.copies*c.dim-1,std::numeric_limits<float>::max());
    }
}
}
int main(int argc,char** argv) {
    if(argc!=2||std::strcmp(argv[1],"--run-gpu")) {
        std::fprintf(stderr,"usage: ds41-engram-gate-selftest --run-gpu\n"
                            "Tiny native HIP work only; require one visible gfx1201 and an approved idle window.\n");return 2;
    }
    hipStream_t stream=nullptr;
    try {
        int devices=0;check(hipGetDeviceCount(&devices));require(devices==1,"requires exactly one visible GPU");
        check(hipSetDevice(0));hipDeviceProp_t props{};check(hipGetDeviceProperties(&props,0));
        require(!std::strncmp(props.gcnArchName,"gfx1201",7)&&(!props.gcnArchName[7]||props.gcnArchName[7]==':'),"requires gfx1201");
        check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));size_t payload=0,metadata=0;
        const auto one=[&](fix::Fixture fixture) {
            try{run(fixture,stream);++payload;}
            catch(const std::exception& e) {
                std::fprintf(stderr,"Engram case=%zu dtype=%s pitch=%d copies=%lld dim=%lld: %s\n",payload+1,
                    rad_dtype_name(fixture.dtype),fixture.pitch,(long long)fixture.copies,(long long)fixture.dim,e.what());throw;
            }
        };
        for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(int pitch=0;pitch<3;++pitch) {
            for(bool mask:{false,true})for(bool gates:{false,true})one(fix::Fixture(dtype,pitch,mask,gates)); //36
            fix::Fixture zeros(dtype,pitch);
            for(int64_t i=0;i<zeros.buffers[eg::input].count();++i)zeros.buffers[eg::input].set(i,-0.f);
            one(zeros); //9
            fix::Fixture negative(dtype,pitch,false,true);
            for(int64_t i=0;i<negative.buffers[eg::input].count();++i)negative.buffers[eg::input].set(i,1.f);
            for(int64_t i=0;i<negative.buffers[eg::projected_kv].count();++i)negative.buffers[eg::projected_kv].set(i,-1.f);
            for(int64_t i=0;i<negative.copies*negative.dim;++i) {
                negative.buffers[eg::q_weight].set(i,1.f);negative.buffers[eg::k_weight].set(i,1.f);
            }
            one(negative); //9
            for(auto bad:{Payload::input_nan,Payload::kv_inf,Payload::q_nan,Payload::k_inf,Payload::mask_negative,
                          Payload::mask_two,Payload::masked_nan,Payload::weight_overflow}) {
                fix::Fixture fixture(dtype,pitch);corrupt(fixture,bad);one(fixture); //72
            }
        }
        for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16}) {
            one(fix::Fixture(dtype,0,true,true,5120,4,1)); //3
            one(fix::Fixture(dtype,2,true,true,13,8,2)); //3
        }
        fix::Fixture norm;norm.buffers[eg::input].set(norm.buffers[eg::input].count()-1,std::numeric_limits<float>::max());one(norm);
        fix::Fixture half(RAD_F16,0,false,true,1,1,2);
        half.buffers[eg::input].set(1,60000);half.buffers[eg::projected_kv].set(2,1);half.buffers[eg::projected_kv].set(3,60000);one(half);
        for(bool key:{false,true}) {
            fix::Fixture denominator(RAD_F32,2,true,true,1,2,3);
            denominator.params[4].dval=std::numeric_limits<float>::max();
            const int64_t at=key?(denominator.tokens-1)*(denominator.copies+1)+(denominator.copies-1):
                                  denominator.tokens*denominator.copies-1;
            denominator.buffers[key?eg::projected_kv:eg::input].set(at,1e19f);one(denominator);
        }
        for(int fault=0;fault<int(expected_metadata_cases);++fault){run(fix::Fixture(),stream,fault);++metadata;}
        require(payload==expected_payload_cases&&metadata==expected_metadata_cases,"Engram checker case count drift");
        check(hipStreamSynchronize(stream));check(hipStreamDestroy(stream));stream=nullptr;
        std::printf("PASS: %zu tiny native Engram gate cases plus %zu descriptor failures; independent CPU equations, guards and explicit status\n",payload,metadata);
        return 0;
    } catch(const std::exception& error) {
        if(stream){static_cast<void>(hipStreamSynchronize(stream));static_cast<void>(hipStreamDestroy(stream));}
        std::fprintf(stderr,"Engram gate checker: %s\n",error.what());return 1;
    }
}
