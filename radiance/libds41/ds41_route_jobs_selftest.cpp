/* Standalone explicit-gated GPU checker; never CTest, never model/service work.
 * Link ds41_route_jobs.hip directly. CPU tests independently pin the mappings. */
#include <hip/hip_runtime.h>
#include "ds41_test_scratch.h"
#include "ds41_route_jobs_fixture.h"
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
namespace r=rad::ds41::route_jobs;
namespace f=r::testing;
namespace {
template<class V>std::vector<uint8_t> bytes(const V& value){std::vector<uint8_t> out(value.size()*sizeof(*value.data()));std::memcpy(out.data(),value.data(),out.size());return out;}
std::vector<uint32_t> words(const std::vector<uint8_t>& value){std::vector<uint32_t> out(value.size()/4);std::memcpy(out.data(),value.data(),value.size());return out;}
void require(bool yes,const char* why){if(!yes)throw std::runtime_error(why);}
void check(hipError_t status){if(status!=hipSuccess)throw std::runtime_error(hipGetErrorString(status));}
struct Device {
    static constexpr size_t guard=64;void* base=nullptr;size_t bytes;
    explicit Device(std::span<const uint32_t> words):bytes(words.size_bytes()){
        std::vector<uint8_t> host(bytes+2*guard,0xC7);
        std::memcpy(host.data()+guard,words.data(),bytes);
        check(hipMalloc(&base,bytes+2*guard));
        const auto copied=hipMemcpy(base,host.data(),host.size(),hipMemcpyHostToDevice);
        if(copied!=hipSuccess){static_cast<void>(hipFree(base));base=nullptr;check(copied);}
    }
    ~Device(){if(base)static_cast<void>(hipFree(base));}
    void* data()const{return static_cast<uint8_t*>(base)+guard;}
    std::vector<uint32_t> read()const{
        std::vector<uint8_t> host(bytes+2*guard);check(hipMemcpy(host.data(),base,host.size(),hipMemcpyDeviceToHost));
        for(size_t i=0;i<guard;++i)require(host[i]==0xC7&&host[guard+bytes+i]==0xC7,"route-job allocation redzone changed");
        std::vector<uint32_t> result(bytes/4);std::memcpy(result.data(),host.data()+guard,bytes);return result;
    }
};
void run(f::Fixture test,hipStream_t stream,int metadata_fault=-1){
    auto oracle=test;const int expected=oracle.run();auto args=test.args();
    std::vector<std::unique_ptr<Device>> allocations;
    for(int i=0;i<r::operand_count;++i){allocations.push_back(std::make_unique<Device>(test.buffers[i].words));test.tensors[i].data=allocations.back()->data();}
    const std::array<uint32_t,4> sentinels{911,912,913,914};const auto flag_before=rad::ds41::testing::make_gpu_scratch<r::Plan>(bytes(sentinels));Device flag(words(flag_before));args.scratch=flag.data();args.scratch_bytes=flag_before.size();
    if(metadata_fault==0)test.tensors[r::expert_counts].data=test.tensors[r::ids].data;
    if(metadata_fault==1)args.scratch=test.tensors[r::ids].data;
    if(metadata_fault==2)args.scratch_bytes=3;
    if(metadata_fault==3)test.tensors[r::identity_rows].stride[0]=0;
    if(metadata_fault==4)test.tensors[r::identity_rows].dtype=RAD_F32;
    const int submitted=ds41_route_jobs(&args,stream);
    require(metadata_fault<0?submitted==RAD_OK:submitted<0,"route-job metadata acceptance mismatch");check(hipStreamSynchronize(stream));
    const auto status=flag.read();require(rad::ds41::testing::gpu_scratch_guards<r::Plan>(bytes(status),flag_before),"route-job staging changed scratch guards");if(metadata_fault>=0)require(bytes(status)==flag_before,"metadata refusal staged a plan");require(metadata_fault<0?int32_t(status[0])==expected:status[0]==911,"route-job payload flag mismatch");
    for(int i=1;i<4;++i)require(status[size_t(i)]==sentinels[size_t(i)],"route-job scratch exceeded4bytes");
    for(int i=0;i<r::operand_count;++i){const auto actual=allocations[size_t(i)]->read();
        const auto& wanted=i<2||expected!=RAD_OK||metadata_fault>=0?test.buffers[i].words:oracle.buffers[i].words;
        require(actual==wanted,"route-job output/input/pitch guard differs from independent CPU fixture");
    }
}
}
int main(int argc,char** argv){
    if((argc!=2&&argc!=3)||std::strcmp(argv[1],"--run-gpu")){
        std::fprintf(stderr,"usage: ds41-route-jobs-selftest --run-gpu [device]\nTiny synthetic GPU work; approved idle window only.\n");return 2;
    }
    hipStream_t stream=nullptr;
    try{
        int device=0;if(argc==3){size_t used=0;device=std::stoi(argv[2],&used);require(used==std::strlen(argv[2])&&device>=0,"invalid device");}
        check(hipSetDevice(device));hipDeviceProp_t properties{};check(hipGetDeviceProperties(&properties,device));
        require(!std::strncmp(properties.gcnArchName,"gfx1201",7)&&(!properties.gcnArchName[7]||properties.gcnArchName[7]==':'),"requires gfx1201");
        check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));size_t cases=0;
        const auto one=[&](f::Fixture test,int metadata=-1){run(std::move(test),stream,metadata);++cases;};
        for(int pitch=0;pitch<3;++pitch){
            one(f::Fixture(pitch));one(f::Fixture(pitch,4,1,1));one(f::Fixture(pitch,2,5,7));one(f::Fixture(pitch,3,6,9));one(f::Fixture(pitch,2,64,4096));
            f::Fixture weights(pitch);weights.buffers[r::weights].weight(0,-0.f);weights.buffers[r::weights].weight(1,std::numeric_limits<float>::denorm_min());weights.buffers[r::weights].weight(5,std::numeric_limits<float>::max());one(std::move(weights));
            for(int fault=0;fault<8;++fault){f::Fixture bad(pitch);
                if(fault==0)bad.buffers[r::ids].id(5,-1);
                if(fault==1)bad.buffers[r::ids].id(5,5);
                if(fault==2)bad.buffers[r::weights].weight(5,-.25f);
                if(fault==3)bad.buffers[r::weights].weight(5,NAN);
                if(fault==4)bad.buffers[r::weights].weight(5,INFINITY);
                if(fault==5)bad.buffers[r::weights].weight(5,-INFINITY);
                if(fault>=6){bad.buffers[r::ids].id(5,bad.buffers[r::ids].id(4));if(fault==7)bad.buffers[r::weights].weight(5,0.f);}
                one(std::move(bad));
            }
        }
        one(f::Fixture(0,1,4096,4096));for(int fault=0;fault<5;++fault)one(f::Fixture(),fault);
        check(hipStreamSynchronize(stream));check(hipStreamDestroy(stream));stream=nullptr;
        std::printf("PASS: %zu tiny route-job GPU cases; exact IDs/weights, token+identity mappings, histogram and transactional errors\n",cases);return 0;
    }catch(const std::exception& error){if(stream){static_cast<void>(hipStreamSynchronize(stream));static_cast<void>(hipStreamDestroy(stream));}
        std::fprintf(stderr,"route-job checker: %s\n",error.what());return 1;}
}
