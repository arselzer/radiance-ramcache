/* Staged standalone device checker. No automatic execution, model data, network or server API.
 * Exactly 167 tiny launch cases include persistent multi-call GPU state, chunk reordering,
 * input/output/state red zones and a separately read asynchronous validation flag. */
#include <hip/hip_runtime.h>
#include "ds41_test_scratch.h"
#include "ds41_compress_fixture.h"
#include <dlfcn.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

namespace c=rad::ds41::compress;
namespace f=c::testing;
namespace {
constexpr size_t expected_cases=167;
void check(hipError_t rc) {if(rc!=hipSuccess)throw std::runtime_error(hipGetErrorString(rc));}
void require(bool okay,const char* why) {if(!okay)throw std::runtime_error(why);}
template<class Container>std::vector<uint8_t> bytes(const Container& v) {
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
                std::all_of(all.end()-guard,all.end(),[](uint8_t b){return b==0xA7;}),"GPU allocation red zone changed");
        return {all.begin()+guard,all.end()-guard};
    }
};
std::vector<uint32_t> run(RadLaunchFn launch,f::Fixture input,Device& persistent,hipStream_t stream) {
    require(persistent.size==input.buffer(c::pending_state).words.size()*4,"persistent GPU state shape changed");
    f::Fixture oracle=input;const int reference=oracle.run();
    std::array<std::unique_ptr<Device>,c::tensors_count> device;
    for(int i=0;i<c::tensors_count;++i)if(i!=c::pending_state&&input.buffers[size_t(i)])
        device[size_t(i)]=std::make_unique<Device>(bytes(input.buffer(i).words));
    const auto flag_before=rad::ds41::testing::make_gpu_scratch<c::Plan>(bytes(input.flag));Device scratch(flag_before);auto args=input.args();
    for(int i=0;i<c::tensors_count;++i)if(input.buffers[size_t(i)])
        input.tensors[size_t(i)].data=i==c::pending_state?persistent.data():device[size_t(i)]->data();
    args.scratch=scratch.data();args.scratch_bytes=rad::ds41::gpu_scratch_bytes<c::Plan>();require(launch(&args,stream)==RAD_OK,"GPU enqueue failed despite valid metadata");
    check(hipStreamSynchronize(stream));
    const auto flag=scratch.read();int32_t status=0;std::memcpy(&status,flag.data()+c::status_offset,4);
    require(status==reference,"GPU validation flag disagrees with CPU preflight");
    require(rad::ds41::testing::gpu_scratch_guards<c::Plan>(flag,flag_before),"GPU changed status sentinels or staged Plan tail guards");
    require(persistent.read()==bytes(oracle.buffer(c::pending_state).words),"GPU persistent state/header or unused state bytes differ");
    for(int i=0;i<c::tensors_count;++i)if(device[size_t(i)]) {
        const auto received=device[size_t(i)]->read();const auto initial=bytes(input.buffer(i).words);
        if(i<c::pooled||reference!=RAD_OK) {require(received==initial,"GPU changed input or wrote during invalid batch");continue;}
        if(i!=c::pooled) {require(received==bytes(oracle.buffer(i).words),"GPU emit/position or their canaries differ");continue;}
        const auto& out=input.buffer(i);std::vector<bool> touched(received.size(),false);
        for(int64_t j=0;j<out.count();++j) {
            const size_t byte=size_t(out.offset(j))*4;float got=0;std::memcpy(&got,received.data()+byte,4);
            const float expected=oracle.buffer(i).getf(j);
            require(std::isfinite(got)&&std::isfinite(expected)&&std::fabs(got-expected)<=4e-6f+4e-5f*std::fabs(expected),"GPU pooled value mismatch");
            if(!oracle.buffer(c::emit).geti(j/input.dimension))require(got==0.f&&!std::signbit(got),"incomplete row is not positive zero");
            for(size_t b=0;b<4;++b)touched[byte+b]=true;
        }
        for(size_t b=0;b<received.size();++b)if(!touched[b])require(received[b]==initial[b],"GPU pooled stride hole/tail changed");
    }
    return oracle.buffer(c::pending_state).words;
}
void fault(f::Fixture& f,int number) {
    const int64_t last=f.tokens-1;
    if(number==0)f.buffer(c::projected_kv).setf(f.tokens*f.dimension-1,NAN);
    if(number==1)f.buffer(c::gate_scores).setf(f.tokens*f.dimension-1,INFINITY);
    if(number==2)f.buffer(c::state_ids).seti(1,4);
    if(number==3)f.buffer(c::state_ids).seti(1,-1);
    if(number==4)f.buffer(c::state_ids).seti(1,2);
    if(number==5)f.buffer(c::cu).seti(0,1);
    if(number==6)f.buffer(c::cu).seti(f.sequences,f.tokens+1);
    if(number==7)f.buffer(c::cu).seti(1,f.tokens+1);
    if(number==8)f.buffer(c::positions).seti(last,7);
    if(number==9)f.buffer(c::positions).seti(last,INT32_MAX);
    if(number==10)f.buffer(c::positions).seti(0,-1);
    if(number==11)f.buffer(c::has_init).seti(1,-1);
    if(number==12)f.buffer(c::num_accepted).seti(1,2);
    if(number==13)f.buffer(c::has_init).seti(1,1);
    if(number==14) {f.buffer(c::has_init).seti(1,3);f.seed(0,0);f.state_set(0,0,f.dimension,1);}
    if(number==15) {f.buffer(c::has_init).seti(1,3);f.seed(0,0);f.state_set(0,1,f.dimension,c::magic(1));}
}
}
int main(int argc,char** argv) {
    if((argc!=3&&argc!=4)||std::strcmp(argv[1],"--run-gpu")) {
        std::fprintf(stderr,"usage: ds41-compress-selftest --run-gpu /path/libds41.so [device]\n"
                            "Runs exactly 167 tiny synthetic compressor cases. Never called by a build/CTest.\n");return 2;
    }
    void* plugin=nullptr;hipStream_t stream=nullptr;
    try {
        int device=0;if(argc==4) {size_t used=0;device=std::stoi(argv[3],&used);require(used==std::strlen(argv[3])&&device>=0,"invalid device");}
        check(hipSetDevice(device));hipDeviceProp_t props{};check(hipGetDeviceProperties(&props,device));
        require(!std::strncmp(props.gcnArchName,"gfx1201",7)&&(!props.gcnArchName[7]||props.gcnArchName[7]==':'),"checker requires gfx1201");
        plugin=dlopen(argv[2],RTLD_NOW|RTLD_LOCAL);if(!plugin)throw std::runtime_error(dlerror());
        const auto abi=reinterpret_cast<uint32_t(*)()>(dlsym(plugin,"rad_plugin_abi_version"));
        const auto launch=reinterpret_cast<RadLaunchFn>(dlsym(plugin,"ds41_compress_pool"));
        require(abi&&abi()==RAD_ABI_VERSION&&launch,"missing compressor launch or wrong ABI");
        check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));size_t cases=0;
        auto one=[&](f::Fixture input) {
            Device state(bytes(input.buffer(c::pending_state).words));
            static_cast<void>(run(launch,std::move(input),state,stream));++cases;
        };
        for(int ratio:{1,2})for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})if(ratio==1||dtype==RAD_F32)
            for(int64_t dim:{1,128,512})for(int pitch=0;pitch<3;++pitch) {
                f::Fixture a(dim,ratio,dtype,pitch);Device state(bytes(a.buffer(c::pending_state).words));
                auto saved=run(launch,a,state,stream);++cases;
                f::Fixture b(dim,ratio,dtype,pitch,{2,3,0},{3,1,0},{0,2,-1});b.buffer(c::pending_state).words=saved;
                saved=run(launch,b,state,stream);++cases;
                f::Fixture d(dim,ratio,dtype,pitch,{1,1},{4,5},{2,0});d.buffer(c::pending_state).words=saved;
                saved=run(launch,d,state,stream);++cases;
                f::Fixture e(dim,ratio,dtype,pitch,{2,3},{6,5},{0,2});e.buffer(c::pending_state).words=saved;
                static_cast<void>(run(launch,e,state,stream));++cases;
            } // 36 profiles, 4 persistent-state chunks each = 144 launch cases
        for(int32_t start:{16777217,2139095041,INT32_MAX-1}) {
            f::Fixture test(5,2,RAD_F32,2,{1},{start},{2});test.seed(2,start,(start&1)!=0);one(std::move(test));
        }
        f::Fixture extreme(512,2,RAD_F32,2,{2},{0},{2});
        for(int64_t row=0;row<2;++row)for(int64_t d=0;d<512;++d)extreme.buffer(c::gate_scores).setf(row*512+d,((row+d)&1)?1000.f:-1000.f);
        one(std::move(extreme));
        one(f::Fixture(128,1,RAD_BF16,1,{3},{0},{2},false));one(f::Fixture(128,2,RAD_F32,1,{3},{0},{2},false));
        for(int i=0;i<16;++i) {f::Fixture test(5,2,RAD_F32,2);fault(test,i);one(std::move(test));}
        f::Fixture pending(5,2,RAD_F32,2,{1},{1},{2});pending.seed(2,1,false);one(std::move(pending));
        require(cases==expected_cases,"checker case count drifted");check(hipStreamSynchronize(stream));check(hipStreamDestroy(stream));stream=nullptr;
        dlclose(plugin);plugin=nullptr;
        std::printf("PASS: %zu tiny persistent compressor GPU cases on %s, guarded U32 state/inputs/outputs and validation flags\n",cases,props.gcnArchName);return 0;
    } catch(const std::exception& e) {
        if(stream) {static_cast<void>(hipStreamSynchronize(stream));static_cast<void>(hipStreamDestroy(stream));}
        if(plugin)dlclose(plugin);
        std::fprintf(stderr,"ds41-compress-selftest: %s\n",e.what());return 1;
    }
}
