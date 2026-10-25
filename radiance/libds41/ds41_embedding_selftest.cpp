/* Direct native checker: 234 tiny payload cases +25 metadata failures. Explicit opt-in only. */
#include <hip/hip_runtime.h>
#include "ds41_embedding_fixture.h"
#include "ds41_test_scratch.h"
#include <cstdio>
#include <memory>
#include <span>
#include <string>
namespace e=rad::ds41::embedding;
namespace f=e::testing;
namespace pk=rad::ds41::packed;
namespace {
constexpr pk::Type formats[]{pk::Type::iq2_xxs,pk::Type::iq2_xs,pk::Type::iq2_s,pk::Type::iq3_xxs,pk::Type::q5_k,pk::Type::q8_0};
constexpr size_t expected_cases=234,expected_metadata=25;
void require(bool condition,const char* why){if(!condition)throw std::runtime_error(why);}
void check(hipError_t rc){if(rc!=hipSuccess)throw std::runtime_error(hipGetErrorString(rc));}
template<class V>std::vector<uint8_t> bytes(const V& value) {
    std::vector<uint8_t> out(value.size()*sizeof(*value.data()));std::memcpy(out.data(),value.data(),out.size());return out;
}
std::vector<uint32_t> words(const std::vector<uint8_t>& value) {
    require(value.size()%4==0,"scratch must have whole words");std::vector<uint32_t> out(value.size()/4);std::memcpy(out.data(),value.data(),value.size());return out;
}
struct Device {
    void* base=nullptr;size_t size;static constexpr size_t guard=64;
    explicit Device(std::span<const uint32_t> source):size(source.size_bytes()) {
        std::vector<uint8_t> host(size+2*guard,0xC7);std::memcpy(host.data()+guard,source.data(),size);check(hipMalloc(&base,host.size()));
        const auto rc=hipMemcpy(base,host.data(),host.size(),hipMemcpyHostToDevice);
        if(rc!=hipSuccess){static_cast<void>(hipFree(base));base=nullptr;check(rc);}
    }
    Device(const Device&)=delete;Device& operator=(const Device&)=delete;
    ~Device(){if(base)static_cast<void>(hipFree(base));}
    void* data()const{return static_cast<uint8_t*>(base)+guard;}
    void* tensor_data()const{return static_cast<uint8_t*>(data())+16;}
    std::vector<uint32_t> read()const {
        std::vector<uint8_t> host(size+2*guard);check(hipMemcpy(host.data(),base,host.size(),hipMemcpyDeviceToHost));
        for(size_t i=0;i<guard;++i)require(host[i]==0xC7&&host[guard+size+i]==0xC7,"embedding allocation red zone changed");
        std::vector<uint32_t> out(size/4);std::memcpy(out.data(),host.data()+guard,size);return out;
    }
};
void run(f::Fixture c,int invalid,hipStream_t stream) {
    if(invalid)c.ids.id(c.rows-1,invalid==1?-1:int32_t(c.n));
    auto oracle=c;const int expected=oracle.run();
    Device ids(c.ids.words),native(c.native),y(c.y.words);
    const std::array<uint32_t,4> sentinels{911,912,913,914};
    const auto scratch_before=rad::ds41::testing::make_gpu_scratch<e::Plan>(bytes(sentinels));Device scratch(words(scratch_before));
    auto a=c.args();c.tensors[e::token_ids].data=ids.tensor_data();c.tensors[e::weight].data=native.tensor_data();c.tensors[e::output].data=y.tensor_data();
    a.scratch=scratch.data();a.scratch_bytes=rad::ds41::gpu_scratch_bytes<e::Plan>();
    require(ds41_embedding(&a,stream)==RAD_OK,"embedding valid descriptor failed to enqueue");check(hipStreamSynchronize(stream));
    const auto flag=scratch.read();require(int32_t(flag[0])==expected,"embedding device payload flag differs from CPU oracle");
    require(rad::ds41::testing::gpu_scratch_guards<e::Plan>(bytes(flag),scratch_before),"embedding status guards/staged Plan tail changed");
    require(ids.read()==c.ids.words,"embedding modified token IDs or their padding");require(native.read()==c.native,"embedding modified packed weight or guards");
    require(y.read()==(expected==RAD_OK?oracle.y.words:c.y.words),"embedding output/cast/padding differs or failed validation wrote output");
}
void reject(int fault,hipStream_t stream) {
    f::Fixture c;Device ids(c.ids.words),native(c.native),y(c.y.words);
    const std::array<uint32_t,4> sentinels{911,912,913,914};
    const auto scratch_before=rad::ds41::testing::make_gpu_scratch<e::Plan>(bytes(sentinels));Device scratch(words(scratch_before));
    auto a=c.args();c.tensors[e::token_ids].data=ids.tensor_data();c.tensors[e::weight].data=native.tensor_data();c.tensors[e::output].data=y.tensor_data();
    a.scratch=scratch.data();a.scratch_bytes=rad::ds41::gpu_scratch_bytes<e::Plan>();
    if(fault==0)c.tensors[e::weight].dtype=RAD_U8;
    if(fault==1)c.tensors[e::weight].shape[1]=c.row_bytes;
    if(fault==2)c.tensors[e::weight].stride[0]=c.k+1;
    if(fault==3)c.tensors[e::weight].stride[1]=2;
    if(fault==4)c.tensors[e::output].dtype=RAD_I32;
    if(fault==5)c.tensors[e::output].stride[1]=0;
    if(fault==6)c.tensors[e::token_ids].dtype=RAD_F32;
    if(fault==7)c.tensors[e::output].data=c.tensors[e::weight].data;
    if(fault==8)c.tensors[e::token_ids].data=c.tensors[e::weight].data;
    if(fault==9)c.tensors[e::output].data=c.tensors[e::token_ids].data;
    if(fault==10)a.scratch=c.tensors[e::weight].data;
    if(fault==11)a.scratch_bytes=3;
    if(fault==12)a.scratch=reinterpret_cast<void*>(UINTPTR_MAX-3);
    if(fault==13)c.integer("M",e::max_rows+1);
    if(fault==14)c.integer("N",INT64_MAX);
    if(fault==15)c.integer("K",33);
    if(fault==16)c.text("format","q5_k");
    if(fault==17)c.text("y_dtype","i32");
    if(fault==18)c.tensors[e::weight].data=nullptr;
    if(fault==19)c.tensors[e::weight].data=reinterpret_cast<void*>(UINTPTR_MAX-31);
    if(fault==20)c.tensors[e::token_ids].data=static_cast<uint8_t*>(ids.tensor_data())+1;
    if(fault==21)c.tensors[e::output].stride[0]=INT64_MAX;
    if(fault==22)--a.scratch_bytes;
    if(fault==23)c.tensors[e::weight].data=static_cast<uint8_t*>(scratch.data())+rad::ds41::device_plan_offset;
    if(fault==24)a.scratch=static_cast<uint8_t*>(scratch.data())+4;
    require(ds41_embedding(&a,stream)<0,"invalid embedding descriptor accepted");check(hipStreamSynchronize(stream));
    require(bytes(scratch.read())==scratch_before,"embedding descriptor refusal changed scratch");
    require(ids.read()==c.ids.words&&native.read()==c.native&&y.read()==c.y.words,"embedding descriptor refusal changed operand");
}
}
int main(int argc,char** argv) {
    if((argc!=2&&argc!=3)||std::strcmp(argv[1],"--run-gpu")) {
        std::fprintf(stderr,"usage: ds41-embedding-selftest --run-gpu [device]\n234 tiny payload cases plus25 metadata failures; never run automatically.\n");return 2;
    }
    hipStream_t stream=nullptr;
    try {
        int device=0;
        if(argc==3){size_t n=0;device=std::stoi(argv[2],&n);require(n==std::strlen(argv[2])&&device>=0,"invalid device");}
        check(hipSetDevice(device));hipDeviceProp_t props{};check(hipGetDeviceProperties(&props,device));
        require(!std::strncmp(props.gcnArchName,"gfx1201",7)&&(!props.gcnArchName[7]||props.gcnArchName[7]==':'),"checker requires gfx1201");
        check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));size_t cases=0,metadata=0;
        auto one=[&](f::Fixture c,int invalid=0) {
            try{run(c,invalid,stream);++cases;}
            catch(const std::exception& error){std::fprintf(stderr,"case=%zu format=%s dtype=%s pitch=%d K=%lld invalid=%d: %s\n",cases+1,pk::type_name(c.type),rad_dtype_name(c.dtype),c.pitch,(long long)c.k,invalid,error.what());throw;}
        };
        for(auto type:formats)for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(int pitch=0;pitch<3;++pitch)for(int k:{256,2304})one(f::Fixture(type,dtype,pitch,k)); //108
        for(auto type:formats)for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(int pitch=0;pitch<3;++pitch)for(int invalid:{1,2})one(f::Fixture(type,dtype,pitch),invalid); //108
        for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(int pitch=0;pitch<3;++pitch)for(int k:{32,7168})one(f::Fixture(pk::Type::q8_0,dtype,pitch,k)); //18
        for(int fault=0;fault<int(expected_metadata);++fault){reject(fault,stream);++metadata;}
        require(cases==expected_cases&&metadata==expected_metadata,"embedding checker case count drifted");
        check(hipStreamSynchronize(stream));check(hipStreamDestroy(stream));stream=nullptr;
        std::printf("PASS: %zu tiny embedding GPU cases plus %zu descriptor failures on %s; native decode, output casts, whole-batch ID validation and guards\n",cases,metadata,props.gcnArchName);return 0;
    } catch(const std::exception& error) {
        if(stream){static_cast<void>(hipStreamSynchronize(stream));static_cast<void>(hipStreamDestroy(stream));}
        std::fprintf(stderr,"ds41-embedding-selftest: %s\n",error.what());return 1;
    }
}
