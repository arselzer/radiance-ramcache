/* Explicit native gate: 432 tiny payload cases +16 metadata refusals. No model or service. */
#include <hip/hip_runtime.h>
#include "ds41_latent_emit_fixture.h"
#include "ds41_test_scratch.h"
#include <cstdio>
#include <memory>
#include <span>
#include <string>
namespace e=rad::ds41::latent_emit;
namespace f=e::testing;
namespace {
constexpr size_t expected_cases=432,expected_metadata=16;
void require(bool value,const char* why){if(!value)throw std::runtime_error(why);}
void check(hipError_t rc){if(rc!=hipSuccess)throw std::runtime_error(hipGetErrorString(rc));}
template<class V>std::vector<uint8_t> bytes(const V& source) {
    std::vector<uint8_t> out(source.size()*sizeof(*source.data()));std::memcpy(out.data(),source.data(),out.size());return out;
}
struct Device {
    void* base=nullptr;size_t size;static constexpr size_t guard=64;
    explicit Device(const std::vector<uint8_t>& source):size(source.size()) {
        std::vector<uint8_t> host(size+2*guard,0xC7);std::memcpy(host.data()+guard,source.data(),size);check(hipMalloc(&base,host.size()));
        const auto rc=hipMemcpy(base,host.data(),host.size(),hipMemcpyHostToDevice);
        if(rc!=hipSuccess){static_cast<void>(hipFree(base));base=nullptr;check(rc);}
    }
    Device(const Device&)=delete;Device& operator=(const Device&)=delete;
    ~Device(){if(base)static_cast<void>(hipFree(base));}
    void* data()const{return static_cast<uint8_t*>(base)+guard;}
    void* tensor_data()const{return static_cast<uint8_t*>(data())+16;}
    std::vector<uint8_t> read()const {
        std::vector<uint8_t> host(size+2*guard);check(hipMemcpy(host.data(),base,host.size(),hipMemcpyDeviceToHost));
        for(size_t i=0;i<guard;++i)require(host[i]==0xC7&&host[guard+size+i]==0xC7,"emitted latent allocation red zone changed");
        return {host.begin()+guard,host.begin()+guard+size};
    }
};
enum class Scenario {normal,zero,mask,position,slot,phase,duplicate,ignored_coordinates,boundary};
void arrange(f::Fixture& c,Scenario scenario) {
    const int64_t last=c.ratio==2?5:6,first=c.ratio==2?1:0;
    if(scenario==Scenario::zero)for(int64_t row=0;row<c.rows;++row)c.buffers[e::emit].integer(row,0);
    if(scenario==Scenario::mask)c.buffers[e::emit].integer(c.rows-1,2);
    if(scenario==Scenario::position)c.buffers[e::positions].integer(last,-1);
    if(scenario==Scenario::slot)c.buffers[e::slot_mapping].integer(last,c.blocks*c.block);
    if(scenario==Scenario::phase)c.buffers[e::slot_mapping].integer(last,0);
    if(scenario==Scenario::duplicate){c.buffers[e::positions].integer(last,c.buffers[e::positions].integer(first));c.buffers[e::slot_mapping].integer(last,c.buffers[e::slot_mapping].integer(first));}
    if(scenario==Scenario::ignored_coordinates)for(int64_t row=0;row<c.rows;++row)if(c.buffers[e::emit].integer(row)==0){c.buffers[e::positions].integer(row,INT64_MIN);c.buffers[e::slot_mapping].integer(row,INT64_MAX);}
    if(scenario==Scenario::boundary){c.buffers[e::positions].integer(last,c.position_dtype==RAD_I32?INT32_MAX:INT64_MAX);c.buffers[e::slot_mapping].integer(last,3);}
}
void run(f::Fixture c,Scenario scenario,hipStream_t stream) {
    arrange(c,scenario);auto oracle=c;const int expected=oracle.run();std::array<std::unique_ptr<Device>,e::operand_count> device;
    for(int i=0;i<e::operand_count;++i)device[size_t(i)]=std::make_unique<Device>(bytes(c.buffers[size_t(i)].words));
    const std::array<uint32_t,4> sentinels{911,912,913,914};const auto scratch_before=rad::ds41::testing::make_gpu_scratch<e::Plan>(bytes(sentinels));Device scratch(scratch_before);
    auto a=c.args();for(int i=0;i<e::operand_count;++i)c.tensors[size_t(i)].data=device[size_t(i)]->tensor_data();
    a.scratch=scratch.data();a.scratch_bytes=rad::ds41::gpu_scratch_bytes<e::Plan>();
    require(ds41_latent_store_emitted(&a,stream)==RAD_OK,"emitted latent valid descriptors failed to enqueue");check(hipStreamSynchronize(stream));
    const auto after=scratch.read();int32_t flag=0;std::memcpy(&flag,after.data(),4);require(flag==expected,"emitted latent device flag differs from CPU status");
    require(rad::ds41::testing::gpu_scratch_guards<e::Plan>(after,scratch_before),"emitted latent status guards/Plan tail changed");
    for(int i=0;i<e::operand_count;++i)require(device[size_t(i)]->read()==bytes(oracle.buffers[size_t(i)].words),"emitted latent raw cache/status/source/guard bytes differ from CPU oracle");
}
void reject(int fault,hipStream_t stream) {
    f::Fixture c;std::array<std::unique_ptr<Device>,e::operand_count> device;
    for(int i=0;i<e::operand_count;++i)device[size_t(i)]=std::make_unique<Device>(bytes(c.buffers[size_t(i)].words));
    const std::array<uint32_t,4> sentinels{911,912,913,914};const auto scratch_before=rad::ds41::testing::make_gpu_scratch<e::Plan>(bytes(sentinels));Device scratch(scratch_before);
    auto a=c.args();for(int i=0;i<e::operand_count;++i)c.tensors[size_t(i)].data=device[size_t(i)]->tensor_data();
    a.scratch=scratch.data();a.scratch_bytes=rad::ds41::gpu_scratch_bytes<e::Plan>();
    if(fault==0)c.tensors[e::positions].dtype=RAD_F32;
    if(fault==1)c.tensors[e::slot_mapping].dtype=RAD_U32;
    if(fault==2)c.tensors[e::cache].dtype=RAD_I32;
    if(fault==3)++c.tensors[e::cache].stride[0];
    if(fault==4)c.tensors[e::status].stride[0]=0;
    if(fault==5)c.integer("block_size",8);
    if(fault==6)c.integer("ratio",3);
    if(fault==7)c.integer("n_blocks",2);
    if(fault==8)a.scratch_bytes=3;
    if(fault==9)a.scratch=c.tensors[e::cache].data;
    if(fault==10)c.tensors[e::cache].data=c.tensors[e::x].data;
    if(fault==11)c.tensors[e::status].data=c.tensors[e::emit].data;
    if(fault==12)a.scratch=reinterpret_cast<void*>(UINTPTR_MAX-3);
    if(fault==13)--a.scratch_bytes;
    if(fault==14)c.tensors[e::x].data=static_cast<uint8_t*>(scratch.data())+rad::ds41::device_plan_offset;
    if(fault==15)a.scratch=static_cast<uint8_t*>(scratch.data())+4;
    require(ds41_latent_store_emitted(&a,stream)<0,"invalid emitted latent metadata accepted");check(hipStreamSynchronize(stream));
    require(scratch.read()==scratch_before,"emitted latent metadata failure changed scratch");
    for(int i=0;i<e::operand_count;++i)require(device[size_t(i)]->read()==bytes(c.buffers[size_t(i)].words),"emitted latent metadata failure changed operand");
}
}
int main(int argc,char** argv) {
    if((argc!=2&&argc!=3)||std::strcmp(argv[1],"--run-gpu")){
        std::fprintf(stderr,"usage: ds41-latent-emit-selftest --run-gpu [device]\n432 tiny payload cases plus16 metadata refusals; never run automatically.\n");return 2;}
    hipStream_t stream=nullptr;
    try {
        int device=0;if(argc==3){size_t n=0;device=std::stoi(argv[2],&n);require(n==std::strlen(argv[2])&&device>=0,"invalid device");}
        check(hipSetDevice(device));hipDeviceProp_t props{};check(hipGetDeviceProperties(&props,device));require(!std::strncmp(props.gcnArchName,"gfx1201",7)&&(!props.gcnArchName[7]||props.gcnArchName[7]==':'),"checker requires gfx1201");
        check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));size_t cases=0,metadata=0;
        auto one=[&](f::Fixture c,Scenario scenario){try{run(c,scenario,stream);++cases;}catch(const std::exception& error){std::fprintf(stderr,"case=%zu dtype=%s ratio=%lld D=%lld pitch=%d pos=%s slot=%s scenario=%d: %s\n",cases+1,rad_dtype_name(c.dtype),(long long)c.ratio,(long long)c.dimension,c.pitch,rad_dtype_name(c.position_dtype),rad_dtype_name(c.slot_dtype),int(scenario),error.what());throw;}};
        for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(uint32_t pd:{RAD_I32,RAD_I64})for(uint32_t sd:{RAD_I32,RAD_I64})for(int ratio:{1,2})for(int dim:{128,512})for(int pitch=0;pitch<3;++pitch)one(f::Fixture(dtype,dim,ratio,pitch,pd,sd),Scenario::normal); //144
        for(auto scenario:{Scenario::zero,Scenario::mask,Scenario::position,Scenario::slot,Scenario::phase,Scenario::duplicate,Scenario::ignored_coordinates,Scenario::boundary})for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(uint32_t coordinate:{RAD_I32,RAD_I64})for(int ratio:{1,2})for(int pitch=0;pitch<3;++pitch)one(f::Fixture(dtype,128,ratio,pitch,coordinate,coordinate),scenario); //288
        for(int fault=0;fault<int(expected_metadata);++fault){reject(fault,stream);++metadata;}
        require(cases==expected_cases&&metadata==expected_metadata,"latent emit checker case count drifted");check(hipStreamSynchronize(stream));check(hipStreamDestroy(stream));stream=nullptr;
        std::printf("PASS: %zu emitted-latent payload cases plus %zu metadata refusals on %s; exact bits, masks, I32/I64 coordinates, atomic cache and all-row status/guards\n",cases,metadata,props.gcnArchName);return 0;
    } catch(const std::exception& error){if(stream){static_cast<void>(hipStreamSynchronize(stream));static_cast<void>(hipStreamDestroy(stream));}std::fprintf(stderr,"ds41-latent-emit-selftest: %s\n",error.what());return 1;}
}
