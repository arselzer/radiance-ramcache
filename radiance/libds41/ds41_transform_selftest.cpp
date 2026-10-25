/* Explicitly gated synthetic GPU checker. Exactly 261 device-payload cases plus 14 host
 * descriptor failures; guarded inputs, output padding and the separate device status flag. */
#include <cstring>
#include <hip/hip_runtime.h>
#include "ds41_test_scratch.h"
#include "ds41_transform_fixture.h"
#include <dlfcn.h>
#include <cstdio>
#include <memory>
#include <string>
namespace t=rad::ds41::transform;
namespace fix=t::testing;
namespace {
constexpr size_t expected_cases=261,expected_metadata_cases=14;
void check(hipError_t rc) {if(rc!=hipSuccess)throw std::runtime_error(hipGetErrorString(rc));}
void require(bool okay,const char* message) {if(!okay)throw std::runtime_error(message);}
template<class V>std::vector<uint8_t> bytes(const V& v) {
    std::vector<uint8_t> out(v.size()*sizeof(*v.data()));std::memcpy(out.data(),v.data(),out.size());return out;
}
struct Device {
    void* base=nullptr;size_t size=0;static constexpr size_t guard=64;
    explicit Device(const std::vector<uint8_t>& value):size(value.size()) {
        check(hipMalloc(&base,size+2*guard));auto rc=hipMemset(base,0xA7,size+2*guard);
        if(rc==hipSuccess)rc=hipMemcpy(data(),value.data(),size,hipMemcpyHostToDevice);
        if(rc!=hipSuccess) {static_cast<void>(hipFree(base));base=nullptr;check(rc);}
    }
    Device(const Device&)=delete;Device& operator=(const Device&)=delete;
    ~Device() {if(base)static_cast<void>(hipFree(base));}
    void* data() const {return static_cast<uint8_t*>(base)+guard;}
    std::vector<uint8_t> read() const {
        std::vector<uint8_t> all(size+2*guard);check(hipMemcpy(all.data(),base,all.size(),hipMemcpyDeviceToHost));
        require(std::all_of(all.begin(),all.begin()+guard,[](uint8_t b){return b==0xA7;})&&
                std::all_of(all.end()-guard,all.end(),[](uint8_t b){return b==0xA7;}),"GPU allocation red zone changed");
        return {all.begin()+guard,all.end()-guard};
    }
};
enum class Scenario {normal,zeros,no_yarn,last_position,input_nan,gain_inf,bad_pos_negative,bad_pos_end,
                     bad_pos_max,pair_nan,pair_inf,pair_nonunit,unused_pair_nan};
void arrange(fix::Fixture& c,Scenario s) {
    if(s==Scenario::zeros)for(int64_t i=0;i<c.buffers[0].count();++i)c.buffers[0].set(i,-0.f);
    if(s==Scenario::no_yarn) {c.spec.original_context=0;c.rebuild();}
    if(s==Scenario::last_position)c.buffers[1].id(2,int32_t(c.table_positions-1));
    if(s==Scenario::input_nan)c.buffers[0].set(c.buffers[0].count()-1,NAN);
    if(s==Scenario::gain_inf)c.buffers[1].set(c.dim-1,INFINITY);
    if(s==Scenario::bad_pos_negative)c.buffers[1].id(2,-1);
    if(s==Scenario::bad_pos_end)c.buffers[1].id(2,int32_t(c.table_positions));
    if(s==Scenario::bad_pos_max)c.buffers[1].id(2,INT32_MAX);
    if(s==Scenario::pair_nan)c.buffers[2].set(37*c.rotary+c.rotary-1,NAN);
    if(s==Scenario::pair_inf)c.buffers[2].set(37*c.rotary+c.rotary-1,INFINITY);
    if(s==Scenario::pair_nonunit)c.buffers[2].set(37*c.rotary,2.f);
    if(s==Scenario::unused_pair_nan)c.buffers[2].set(63*c.rotary,NAN);
}
void promote(fix::Fixture& c) {
    for(auto& b:c.buffers)if(b.t.dtype==RAD_F16||b.t.dtype==RAD_BF16) {
        const auto values=b.values();b.t.dtype=RAD_F32;
        for(int64_t i=0;i<b.count();++i)b.set(i,values[size_t(i)]);
    }
    for(auto& param:c.params)if(!std::strcmp(param.key,"dtype")||!std::strcmp(param.key,"a_dtype")||!std::strcmp(param.key,"y_dtype"))param.sval="f32";
    c.refresh();
}
uint16_t halfword(const std::vector<uint8_t>& value,size_t offset) {uint16_t h;std::memcpy(&h,value.data()+offset,2);return h;}
int ordered(uint16_t h) {return h&0x8000u?32768-int(h&0x7FFFu):32768+int(h);}
float scalar(const std::vector<uint8_t>& value,uint32_t dtype,size_t element) {
    if(dtype==RAD_F32) {float result;std::memcpy(&result,value.data()+4*element,4);return result;}
    const auto h=halfword(value,2*element);return dtype==RAD_F16?rad_f16_to_f32(h):rad_bf16_to_f32(h);
}
void run(RadLaunchFn launch,fix::Fixture input,Scenario scenario,hipStream_t stream) {
    arrange(input,scenario);auto oracle=input;const int expected=oracle.run();auto raw=input;promote(raw);
    require((raw.run()==RAD_OK)==(expected==RAD_OK),"FP32 shadow preflight disagrees");
    std::vector<std::unique_ptr<Device>> device;
    for(const auto& b:input.buffers)device.push_back(std::make_unique<Device>(bytes(b.words)));
    const auto flag_before=rad::ds41::testing::make_gpu_scratch<t::Plan>(bytes(input.flag));Device scratch(flag_before);auto args=input.args();
    for(size_t i=0;i<device.size();++i)input.tensors[i].data=device[i]->data();args.scratch=scratch.data();args.scratch_bytes=scratch.size;
    require(launch(&args,stream)==RAD_OK,"GPU enqueue failed for valid metadata");check(hipStreamSynchronize(stream));
    const auto flag=scratch.read();int32_t status=0;std::memcpy(&status,flag.data()+t::status_offset,4);
    require(status==expected,"GPU validation flag differs from CPU preflight");
    require(rad::ds41::testing::gpu_scratch_guards<t::Plan>(flag,flag_before),"GPU plan staging changed scratch guards");
    for(size_t i=0;i<device.size();++i) {
        const auto actual=device[i]->read(),initial=bytes(input.buffers[i].words);
        if(i+1<device.size()||expected!=RAD_OK) {require(actual==initial,"GPU changed input or wrote output for invalid batch");continue;}
        const auto& b=input.buffers[i];const int width=rad_dtype_bits(b.t.dtype)/8;std::vector<bool> touched(actual.size(),false);
        const auto source=bytes(input.buffers[0].words);
        for(int64_t j=0;j<b.count();++j) {
            const size_t at=size_t(b.offset(j));const float reference=oracle.buffers[i].get(j),unrounded=raw.buffers[i].get(j);
            const float got=scalar(actual,b.t.dtype,at);
            require(std::isfinite(got)&&std::isfinite(reference)&&std::isfinite(unrounded),"nonfinite transform output");
            const float tolerance=4e-6f+4e-5f*std::fabs(unrounded);
            if(b.t.dtype==RAD_F32)require(std::fabs(got-reference)<=tolerance,"FP32 transform differs from independent CPU oracle");
            else {
                const uint16_t encoded=b.t.dtype==RAD_F16?rad_f32_to_f16(unrounded):rad_f32_to_bf16(unrounded),received=halfword(actual,at*2);
                if(received!=encoded) {
                    require(std::abs(ordered(received)-ordered(encoded))<=1,"transform storage differs by more than one representable step");
                    require(std::fabs(unrounded-.5f*(got+reference))<=tolerance,"transform storage mismatch away from rounding boundary");
                }
            }
            if(got==0&&reference==0)require(std::signbit(got)==std::signbit(reference),"transform zero storage sign differs");
            if(input.op==t::Op::rope_tail&&j%input.dim<input.dim-input.rotary) {
                const size_t src=size_t(input.buffers[0].offset(j))*size_t(width);
                require(std::equal(actual.begin()+at*size_t(width),actual.begin()+(at+1)*size_t(width),source.begin()+src),"unrotated prefix bytes changed");
            }
            for(int byte=0;byte<width;++byte)touched[at*size_t(width)+size_t(byte)]=true;
        }
        for(size_t byte=0;byte<actual.size();++byte)if(!touched[byte])require(actual[byte]==initial[byte],"GPU output padding/canary changed");
    }
}
enum class Metadata {alias,bad_pitch,bad_dtype,short_scratch,scratch_alias,scratch_wrap,zero_capacity};
void reject_metadata(RadLaunchFn launch,t::Op op,Metadata invalid,hipStream_t stream) {
    fix::Fixture input(op);std::vector<std::unique_ptr<Device>> device;
    for(const auto& b:input.buffers)device.push_back(std::make_unique<Device>(bytes(b.words)));
    const auto before=rad::ds41::testing::make_gpu_scratch<t::Plan>(bytes(input.flag));Device scratch(before);auto args=input.args();
    for(size_t i=0;i<device.size();++i)input.tensors[i].data=device[i]->data();args.scratch=scratch.data();args.scratch_bytes=scratch.size;
    if(invalid==Metadata::alias)input.tensors.back().data=input.tensors[0].data;
    if(invalid==Metadata::bad_pitch)input.tensors[0].stride[0]=INT64_MAX;
    if(invalid==Metadata::bad_dtype)input.tensors.back().dtype=RAD_U32;
    if(invalid==Metadata::short_scratch)args.scratch_bytes=3;
    if(invalid==Metadata::scratch_alias)args.scratch=input.tensors[0].data;
    if(invalid==Metadata::scratch_wrap)args.scratch=reinterpret_cast<void*>(UINTPTR_MAX-3);
    if(invalid==Metadata::zero_capacity)input.params[0].ival=0;
    require(launch(&args,stream)<0,"bad descriptor was accepted");check(hipStreamSynchronize(stream));
    require(scratch.read()==before,"host validation failure changed device status");
    for(size_t i=0;i<device.size();++i)require(device[i]->read()==bytes(input.buffers[i].words),"host validation failure changed an operand");
}
} // namespace
int main(int argc,char** argv) {
    if((argc!=3&&argc!=4)||std::strcmp(argv[1],"--run-gpu")) {
        std::fprintf(stderr,"usage: ds41-transform-selftest --run-gpu /path/libds41.so [device]\n"
                            "Runs exactly 261 tiny device cases and 14 host validation failures; never run automatically.\n");return 2;
    }
    void* plugin=nullptr;hipStream_t stream=nullptr;
    try {
        int device=0;if(argc==4) {size_t used=0;device=std::stoi(argv[3],&used);require(used==std::strlen(argv[3])&&device>=0,"invalid device");}
        check(hipSetDevice(device));hipDeviceProp_t props{};check(hipGetDeviceProperties(&props,device));
        require(!std::strncmp(props.gcnArchName,"gfx1201",7)&&(!props.gcnArchName[7]||props.gcnArchName[7]==':'),"checker requires gfx1201");
        plugin=dlopen(argv[2],RTLD_NOW|RTLD_LOCAL);if(!plugin)throw std::runtime_error(dlerror());
        const auto abi=reinterpret_cast<uint32_t(*)()>(dlsym(plugin,"rad_plugin_abi_version"));require(abi&&abi()==RAD_ABI_VERSION,"plugin ABI mismatch");
        const char* names[]{"ds41_rmsnorm","ds41_rope_tail"};std::array<RadLaunchFn,2> launch{};
        for(size_t i=0;i<launch.size();++i) {launch[i]=reinterpret_cast<RadLaunchFn>(dlsym(plugin,names[i]));require(launch[i],"missing transform launch symbol");}
        check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));size_t cases=0,metadata_cases=0;
        auto one=[&](t::Op op,uint32_t ad,uint32_t yd,int pitch,Scenario scenario=Scenario::normal,
                     bool inverse=false,int64_t width=0,int64_t rotary=4,int64_t heads=2) {
            try {run(launch[size_t(op)],fix::Fixture(op,ad,yd,pitch,inverse,width,rotary,heads),scenario,stream);++cases;}
            catch(const std::exception& e) {std::fprintf(stderr,"case=%zu op=%s input=%s output=%s pitch=%d scenario=%u inverse=%d width=%lld: %s\n",
                cases+1,names[size_t(op)],rad_dtype_name(ad),rad_dtype_name(yd),pitch,unsigned(scenario),inverse,(long long)width,e.what());throw;}
        };
        for(uint32_t ad:{RAD_F32,RAD_F16,RAD_BF16})for(uint32_t yd:{RAD_F32,RAD_F16,RAD_BF16})for(int p=0;p<3;++p)one(t::Op::rmsnorm,ad,yd,p); //27
        for(int64_t width:{512,1536,4096})for(uint32_t ad:{RAD_F32,RAD_F16,RAD_BF16})for(uint32_t yd:{RAD_F32,RAD_F16,RAD_BF16})one(t::Op::rmsnorm,ad,yd,1,Scenario::normal,false,width); //27
        for(uint32_t ad:{RAD_F32,RAD_F16,RAD_BF16})for(uint32_t yd:{RAD_F32,RAD_F16,RAD_BF16})one(t::Op::rmsnorm,ad,yd,2,Scenario::zeros); //9
        for(auto scenario:{Scenario::normal,Scenario::no_yarn,Scenario::last_position})for(uint32_t d:{RAD_F32,RAD_F16,RAD_BF16})for(int p=0;p<3;++p)for(bool inverse:{false,true})one(t::Op::rope_tail,d,d,p,scenario,inverse); //54
        for(uint32_t d:{RAD_F32,RAD_F16,RAD_BF16})for(int p=0;p<3;++p)for(bool inverse:{false,true})one(t::Op::rope_tail,d,d,p,Scenario::normal,inverse,4,4,1); //18
        for(uint32_t d:{RAD_F32,RAD_F16,RAD_BF16})for(int p=0;p<3;++p)for(bool inverse:{false,true})one(t::Op::rope_tail,d,d,p,Scenario::normal,inverse,128,128,3); //18
        for(uint32_t ad:{RAD_F32,RAD_F16,RAD_BF16})for(uint32_t yd:{RAD_F32,RAD_F16,RAD_BF16})for(int p=0;p<3;++p)one(t::Op::rmsnorm,ad,yd,p,Scenario::input_nan); //27
        for(uint32_t d:{RAD_F32,RAD_F16,RAD_BF16})for(int p=0;p<3;++p)one(t::Op::rope_tail,d,d,p,Scenario::input_nan); //9
        for(uint32_t ad:{RAD_F32,RAD_F16,RAD_BF16})for(uint32_t yd:{RAD_F32,RAD_F16,RAD_BF16})one(t::Op::rmsnorm,ad,yd,2,Scenario::gain_inf); //9
        for(auto scenario:{Scenario::bad_pos_negative,Scenario::bad_pos_end,Scenario::bad_pos_max,Scenario::pair_nan,Scenario::pair_inf,Scenario::pair_nonunit})for(uint32_t d:{RAD_F32,RAD_F16,RAD_BF16})for(int p=0;p<3;++p)one(t::Op::rope_tail,d,d,p,scenario); //54
        for(uint32_t d:{RAD_F32,RAD_F16,RAD_BF16})for(int p=0;p<3;++p)one(t::Op::rope_tail,d,d,p,Scenario::unused_pair_nan); //9
        for(auto op:{t::Op::rmsnorm,t::Op::rope_tail})for(auto invalid:{Metadata::alias,Metadata::bad_pitch,Metadata::bad_dtype,Metadata::short_scratch,Metadata::scratch_alias,Metadata::scratch_wrap,Metadata::zero_capacity}) {
            reject_metadata(launch[size_t(op)],op,invalid,stream);++metadata_cases;
        }
        require(cases==expected_cases&&metadata_cases==expected_metadata_cases,"checker case count drifted");
        check(hipStreamSynchronize(stream));check(hipStreamDestroy(stream));stream=nullptr;dlclose(plugin);plugin=nullptr;
        std::printf("PASS: %zu tiny DS41 norm/RoPE GPU cases plus %zu descriptor failures on %s; guarded operands and explicit status flags\n",cases,metadata_cases,props.gcnArchName);return 0;
    } catch(const std::exception& e) {
        if(stream) {static_cast<void>(hipStreamSynchronize(stream));static_cast<void>(hipStreamDestroy(stream));}
        if(plugin)dlclose(plugin);std::fprintf(stderr,"ds41-transform-selftest: %s\n",e.what());return 1;
    }
}
