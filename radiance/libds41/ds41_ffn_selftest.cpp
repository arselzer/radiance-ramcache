/* Staged device checker, never automatic. Tiny synthetic tensors only; explicit scratch flag
 * checks distinguish enqueue success from batch-valid output. Exactly 141 launch cases. */
#include <hip/hip_runtime.h>
#include "ds41_test_scratch.h"
#include "ds41_ffn_fixture.h"
#include <dlfcn.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
namespace f=rad::ds41::ffn;
namespace fix=f::testing;
namespace {
constexpr size_t expected_cases=141;
void check(hipError_t rc) {if(rc!=hipSuccess)throw std::runtime_error(hipGetErrorString(rc));}
void require(bool okay,const char* text) {if(!okay)throw std::runtime_error(text);}
template<class V>std::vector<uint8_t> bytes(const V& v) {
    std::vector<uint8_t> out(v.size()*sizeof(*v.data()));std::memcpy(out.data(),v.data(),out.size());return out;
}
struct Device {
    void* base=nullptr;size_t size=0;static constexpr size_t guard=64;
    explicit Device(const std::vector<uint8_t>& v):size(v.size()) {
        check(hipMalloc(&base,size+2*guard));auto rc=hipMemset(base,0xA7,size+2*guard);
        if(rc==hipSuccess)rc=hipMemcpy(data(),v.data(),size,hipMemcpyHostToDevice);
        if(rc!=hipSuccess) {static_cast<void>(hipFree(base));base=nullptr;check(rc);}
    }
    Device(const Device&)=delete;Device& operator=(const Device&)=delete;
    ~Device() {if(base)static_cast<void>(hipFree(base));}
    void* data() const {return static_cast<uint8_t*>(base)+guard;}
    std::vector<uint8_t> read() const {
        std::vector<uint8_t> all(size+2*guard);check(hipMemcpy(all.data(),base,all.size(),hipMemcpyDeviceToHost));
        require(std::all_of(all.begin(),all.begin()+guard,[](uint8_t b){return b==0xA7;})&&
                std::all_of(all.end()-guard,all.end(),[](uint8_t b){return b==0xA7;}),"GPU red zone changed");
        return {all.begin()+guard,all.end()-guard};
    }
};
enum class Scenario {normal,no_limit,large_limit,negative_gate,zero_weight,expert_order,shared_only,post_zero,
                      input_nan,bad_expert,bad_token,duplicate,negative_weight,post_inf,comb_nan,shared_inf,up_nan};
void arrange(fix::Fixture& c,Scenario s) {
    if(s==Scenario::no_limit)c.real("limit",0);
    if(s==Scenario::large_limit)c.real("limit",100);
    if(s==Scenario::negative_gate) {
        for(int64_t i=0;i<c.buffers[0].count();++i) {c.buffers[0].setf(i,-20);c.buffers[1].setf(i,1.f);}
    }
    if(s==Scenario::zero_weight)for(int64_t i=0;i<c.buffers[2].count();++i)c.buffers[2].setf(i,0.f);
    if(s==Scenario::expert_order) {
        const float big=c.dtype==RAD_F16?65504.f:1e8f,small=c.dtype==RAD_F16?1.f/1048576.f:1.f;
        c.buffers[0].setf(0,small);c.buffers[0].setf(7,big);c.buffers[0].setf(3*7,-big);c.buffers[3].setf(0,0);
    }
    if(s==Scenario::shared_only) {c.buffers[2].seti(4,1);c.buffers[2].seti(5,1);}
    if(s==Scenario::post_zero)for(int operand:{2,3})for(int64_t i=0;i<c.buffers[size_t(operand)].count();++i)c.buffers[size_t(operand)].setf(i,0);
    if(s==Scenario::input_nan)c.buffers[0].setf(c.buffers[0].count()-1,NAN);
    if(s==Scenario::bad_expert)c.buffers[1].seti(5,4);
    if(s==Scenario::bad_token)c.buffers[2].seti(5,3);
    if(s==Scenario::duplicate) {c.buffers[1].seti(5,0);c.buffers[2].seti(5,2);}
    if(s==Scenario::negative_weight)c.buffers[2].setf(2,-1);
    if(s==Scenario::post_inf)c.buffers[2].setf(c.buffers[2].count()-1,INFINITY);
    if(s==Scenario::comb_nan)c.buffers[3].setf(c.buffers[3].count()-1,NAN);
    if(s==Scenario::shared_inf)c.buffers[3].setf(c.buffers[3].count()-1,INFINITY);
    if(s==Scenario::up_nan)c.buffers[1].setf(c.buffers[1].count()-1,NAN);
}
void promote(fix::Fixture& c) {
    for(auto& b:c.buffers)if(b.t.dtype==RAD_F16||b.t.dtype==RAD_BF16) {
        const auto values=b.values();b.t.dtype=RAD_F32;for(int64_t i=0;i<b.count();++i)b.setf(i,values[size_t(i)]);
    }
    for(auto& p:c.params)if(!std::strcmp(p.key,"dtype"))p.sval="f32";
    c.refresh();
}
uint16_t halfword(const std::vector<uint8_t>& v,size_t at) {uint16_t h;std::memcpy(&h,v.data()+at,2);return h;}
int ordered(uint16_t h) {return h&0x8000u?32768-int(h&0x7FFFu):32768+int(h);}
void run(RadLaunchFn launch,fix::Fixture input,Scenario scenario,hipStream_t stream) {
    arrange(input,scenario);fix::Fixture oracle=input;const int expected=oracle.run();
    fix::Fixture raw=input;promote(raw);const int shadow=raw.run();require((shadow==RAD_OK)==(expected==RAD_OK),"CPU FP32 shadow validation disagrees");
    std::vector<std::unique_ptr<Device>> device;for(const auto& b:input.buffers)device.push_back(std::make_unique<Device>(bytes(b.words)));
    const auto flag_before=rad::ds41::testing::make_gpu_scratch<f::Plan>(bytes(input.flag));Device scratch(flag_before);auto args=input.args();
    for(size_t i=0;i<device.size();++i)input.tensors[i].data=device[i]->data();
    args.scratch=scratch.data();args.scratch_bytes=rad::ds41::gpu_scratch_bytes<f::Plan>();
    require(launch(&args,stream)==RAD_OK,"GPU enqueue failed for valid metadata");check(hipStreamSynchronize(stream));
    const auto flag=scratch.read();int32_t status=0;std::memcpy(&status,flag.data()+f::status_offset,4);
    require(status==expected,"GPU payload status differs from CPU preflight");
    require(rad::ds41::testing::gpu_scratch_guards<f::Plan>(flag,flag_before),"GPU changed status sentinels or staged Plan tail guards");
    for(size_t i=0;i<device.size();++i) {
        const auto actual=device[i]->read(),initial=bytes(input.buffers[i].words);
        if(i+1<device.size()||expected!=RAD_OK) {require(actual==initial,"GPU changed read-only input or wrote invalid-batch output");continue;}
        const auto& b=input.buffers[i];std::vector<bool> touched(actual.size(),false);const int width=rad_dtype_bits(b.t.dtype)/8;
        for(int64_t j=0;j<b.count();++j) {
            const size_t at=size_t(b.offset(j));const float reference=oracle.buffers[i].getf(j),unrounded=raw.buffers[i].getf(j);
            float got=0;if(b.t.dtype==RAD_F32)std::memcpy(&got,actual.data()+4*at,4);
            else {const auto h=halfword(actual,at*2);got=b.t.dtype==RAD_F16?rad_f16_to_f32(h):rad_bf16_to_f32(h);}
            require(std::isfinite(got)&&std::isfinite(reference)&&std::isfinite(unrounded),"nonfinite output");
            const float tolerance=4e-6f+4e-5f*std::fabs(unrounded);
            if(b.t.dtype==RAD_F32)require(std::fabs(got-reference)<=tolerance,"FP32 FFN output mismatch");
            else {
                const uint16_t encoded=b.t.dtype==RAD_F16?rad_f32_to_f16(unrounded):rad_f32_to_bf16(unrounded),received=halfword(actual,at*2);
                if(received!=encoded) {
                    require(std::abs(ordered(received)-ordered(encoded))<=1,"FFN storage differs by more than one representable step");
                    require(std::fabs(unrounded-.5f*(got+reference))<=tolerance,"FFN storage mismatch away from a rounding boundary");
                    if(got==0&&reference==0)require(std::signbit(got)==std::signbit(reference),"zero storage sign differs");
                }
            }
            if(scenario==Scenario::post_zero)require(got==0&&!std::signbit(got),"mHC positive-zero result differs");
            if(scenario==Scenario::expert_order&&j==0)require(got==(b.t.dtype==RAD_F16?1.f/1048576.f:1.f),"expert-ID order changed cancellation");
            for(int byte=0;byte<width;++byte)touched[at*size_t(width)+size_t(byte)]=true;
        }
        for(size_t byte=0;byte<actual.size();++byte)if(!touched[byte])require(actual[byte]==initial[byte],"output padding/canary changed");
    }
}
}
int main(int argc,char** argv) {
    if((argc!=3&&argc!=4)||std::strcmp(argv[1],"--run-gpu")) {
        std::fprintf(stderr,"usage: ds41-ffn-selftest --run-gpu /path/libds41.so [device]\n"
                            "Runs exactly 141 tiny synthetic FFN/mHC cases; never run by a build or CTest.\n");return 2;
    }
    void* plugin=nullptr;hipStream_t stream=nullptr;
    try {
        int device=0;if(argc==4) {size_t used=0;device=std::stoi(argv[3],&used);require(used==std::strlen(argv[3])&&device>=0,"invalid device");}
        check(hipSetDevice(device));hipDeviceProp_t props{};check(hipGetDeviceProperties(&props,device));
        require(!std::strncmp(props.gcnArchName,"gfx1201",7)&&(!props.gcnArchName[7]||props.gcnArchName[7]==':'),"checker requires gfx1201");
        plugin=dlopen(argv[2],RTLD_NOW|RTLD_LOCAL);if(!plugin)throw std::runtime_error(dlerror());
        const auto abi=reinterpret_cast<uint32_t(*)()>(dlsym(plugin,"rad_plugin_abi_version"));require(abi&&abi()==RAD_ABI_VERSION,"plugin ABI mismatch");
        const char* names[]{"ds41_expert_act","ds41_moe_combine","ds41_mhc_post"};std::array<RadLaunchFn,3> launch{};
        for(size_t i=0;i<launch.size();++i) {launch[i]=reinterpret_cast<RadLaunchFn>(dlsym(plugin,names[i]));require(launch[i],"missing FFN launch symbol");}
        check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));size_t cases=0;
        auto one=[&](f::Op op,uint32_t dtype,int pitch,Scenario scenario=Scenario::normal) {
            try {run(launch[size_t(op)],fix::Fixture(op,dtype,pitch),scenario,stream);++cases;}
            catch(const std::exception& e) {std::fprintf(stderr,"case=%zu op=%s dtype=%s pitch=%d scenario=%u: %s\n",cases+1,names[size_t(op)],rad_dtype_name(dtype),pitch,unsigned(scenario),e.what());throw;}
        };
        for(auto op:{f::Op::expert_act,f::Op::moe_combine,f::Op::mhc_post})for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(int pitch=0;pitch<3;++pitch)one(op,dtype,pitch); //27
        for(auto scenario:{Scenario::no_limit,Scenario::large_limit,Scenario::negative_gate,Scenario::zero_weight})
            for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(int pitch=0;pitch<3;++pitch)one(f::Op::expert_act,dtype,pitch,scenario); //36
        for(auto scenario:{Scenario::expert_order,Scenario::shared_only})for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(int pitch=0;pitch<3;++pitch)one(f::Op::moe_combine,dtype,pitch,scenario); //18
        for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(int pitch=0;pitch<3;++pitch)one(f::Op::mhc_post,dtype,pitch,Scenario::post_zero); //9
        for(auto op:{f::Op::expert_act,f::Op::moe_combine,f::Op::mhc_post})for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(int pitch=0;pitch<3;++pitch)one(op,dtype,pitch,Scenario::input_nan); //27
        for(auto scenario:{Scenario::bad_expert,Scenario::bad_token,Scenario::duplicate})for(int pitch=0;pitch<3;++pitch)one(f::Op::moe_combine,RAD_F32,pitch,scenario); //9
        for(int pitch=0;pitch<3;++pitch)one(f::Op::expert_act,RAD_F32,pitch,Scenario::negative_weight); //3
        for(auto scenario:{Scenario::post_inf,Scenario::comb_nan})for(int pitch=0;pitch<3;++pitch)one(f::Op::mhc_post,RAD_F32,pitch,scenario); //6
        for(int pitch=0;pitch<3;++pitch)one(f::Op::moe_combine,RAD_F32,pitch,Scenario::shared_inf); //3
        for(int pitch=0;pitch<3;++pitch)one(f::Op::expert_act,RAD_F32,pitch,Scenario::up_nan); //3
        require(cases==expected_cases,"case count drifted");check(hipStreamSynchronize(stream));check(hipStreamDestroy(stream));stream=nullptr;
        dlclose(plugin);plugin=nullptr;std::printf("PASS: %zu tiny DS41 FFN/mHC GPU cases on %s; guarded operands, rounding boundaries and validation flags\n",cases,props.gcnArchName);return 0;
    } catch(const std::exception& e) {
        if(stream) {static_cast<void>(hipStreamSynchronize(stream));static_cast<void>(hipStreamDestroy(stream));}
        if(plugin)dlclose(plugin);
        std::fprintf(stderr,"ds41-ffn-selftest: %s\n",e.what());return 1;
    }
}
