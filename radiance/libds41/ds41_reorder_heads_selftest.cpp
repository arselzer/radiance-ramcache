/* Explicit native checker. 502 tiny payload cases +26 metadata refusals, no communication. */
#include <cstring>
#include <hip/hip_runtime.h>
#include "ds41_reorder_heads_fixture.h"
#include <cstdio>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
namespace r=rad::ds41::reorder_heads;
namespace f=r::testing;
namespace {
constexpr size_t expected_cases=502,expected_metadata=26;
void require(bool condition,const char* why){if(!condition)throw std::runtime_error(why);}
void check(hipError_t rc){if(rc!=hipSuccess)throw std::runtime_error(hipGetErrorString(rc));}
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
        for(size_t i=0;i<guard;++i)require(host[i]==0xC7&&host[guard+size+i]==0xC7,"reorder allocation red zone changed");
        std::vector<uint32_t> out(size/4);std::memcpy(out.data(),host.data()+guard,size);return out;
    }
};
void mutate(f::Fixture& c,int fault) {
    if(fault==0)c.integer("tp",0);
    if(fault==1)c.integer("tp",65537);
    if(fault==2)c.integer("heads",INT64_MAX);
    if(fault==3)c.integer("head_dim",0);
    if(fault==4)c.integer("M",c.rows-1);
    if(fault==5)c.text("dtype","i32");
    if(fault==6)c.tensors[r::query].rank=3;
    if(fault==7)++c.tensors[r::query].shape[0];
    if(fault==8)c.tensors[r::weights].dtype=RAD_F16;
    if(fault==9)c.tensors[r::query_out].dtype=RAD_BF16;
    if(fault==10)++c.tensors[r::query_out].shape[1];
    if(fault==11)++c.tensors[r::query].stride[0];
    if(fault==12)c.tensors[r::query].stride[3]=0;
    if(fault==13)c.tensors[r::weights_out].stride[1]=-1;
    if(fault==14)c.tensors[r::query_out].data=nullptr;
    if(fault==15)c.tensors[r::query].data=static_cast<uint8_t*>(c.tensors[r::query].data)+1;
    if(fault==16)c.tensors[r::query].data=reinterpret_cast<void*>(UINTPTR_MAX-3);
    if(fault==17){c.integer("M",INT_MAX);c.integer("tp",65536);c.integer("heads",INT_MAX);c.integer("head_dim",INT_MAX);}
    if(fault==18||fault==19) {
        auto& t=c.tensors[r::query];t.rank=3;t.shape[0]=c.tp*c.rows;t.shape[1]=c.heads;t.shape[2]=c.dim;
        t.stride[0]=c.heads*c.dim;t.stride[1]=c.dim;t.stride[2]=1;
        if(fault==18)++t.shape[0]; // fails divisibility before any payload read
        // fault19 intentionally retains rank3 weights beside a flattened query.
    }
    int alias=20;
    for(int a=0;a<r::operand_count;++a)for(int b=0;b<a;++b,++alias)
        if(fault==alias)c.tensors[size_t(a)].data=c.tensors[size_t(b)].data;
}
void run(f::Fixture c,hipStream_t stream,int fault=-1) {
    const auto expected=c.expected();std::array<std::unique_ptr<Device>,r::operand_count> device;
    for(int i=0;i<r::operand_count;++i)device[size_t(i)]=std::make_unique<Device>(c.buffers[size_t(i)].words);
    const std::array<uint32_t,4> scratch_words{0x7F800001u,0x80000000u,0xFFFFFFFFu,0x13579BDFu};Device scratch(scratch_words);
    auto args=c.args();for(int i=0;i<r::operand_count;++i)c.tensors[size_t(i)].data=device[size_t(i)]->tensor_data();
    args.scratch=scratch.data();args.scratch_bytes=sizeof scratch_words;
    if(fault>=0)mutate(c,fault);
    const int rc=ds41_reorder_heads(&args,stream);
    require(fault<0?rc==RAD_OK:rc<0,"reorder enqueue result disagrees with metadata validity");check(hipStreamSynchronize(stream));
    require(scratch.read()==std::vector<uint32_t>(scratch_words.begin(),scratch_words.end()),"scratchless reorder touched caller scratch");
    for(int i=0;i<r::operand_count;++i)
        require(device[size_t(i)]->read()==(fault<0?expected[size_t(i)].words:c.buffers[size_t(i)].words),"reorder bits/padding differ or metadata refusal wrote an operand");
}
}
int main(int argc,char** argv) {
    if((argc!=2&&argc!=3)||std::strcmp(argv[1],"--run-gpu")) {
        std::fprintf(stderr,"usage: ds41-reorder-heads-selftest --run-gpu [device]\n502 tiny payload cases plus26 metadata failures; never run automatically.\n");return 2;
    }
    hipStream_t stream=nullptr;
    try {
        int device=0;
        if(argc==3){size_t n=0;device=std::stoi(argv[2],&n);require(n==std::strlen(argv[2])&&device>=0,"invalid device");}
        check(hipSetDevice(device));hipDeviceProp_t props{};check(hipGetDeviceProperties(&props,device));
        require(!std::strncmp(props.gcnArchName,"gfx1201",7)&&(!props.gcnArchName[7]||props.gcnArchName[7]==':'),"checker requires gfx1201");
        check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));size_t cases=0,metadata=0;
        auto one=[&](f::Fixture c) {
            try{run(c,stream);++cases;}
            catch(const std::exception& error){std::fprintf(stderr,"case=%zu dtype=%s tp=%lld M=%lld H=%lld D=%lld flat=%d: %s\n",cases+1,rad_dtype_name(c.dtype),(long long)c.tp,(long long)c.rows,(long long)c.heads,(long long)c.dim,int(c.flat),error.what());throw;}
        };
        for(bool flat:{false,true}) {
        for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(int tp:{1,2,4,8})for(int rows:{1,3,17})for(int heads:{1,7})for(int dim:{1,7,129})one(f::Fixture(dtype,tp,rows,heads,dim,flat)); //216
        for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})one(f::Fixture(dtype,8,17,7,1025,flat)); //3 grid-stride cases
        std::mt19937 random(411017);const uint32_t dtypes[]{RAD_F32,RAD_F16,RAD_BF16};
        for(int i=0;i<32;++i)one(f::Fixture(dtypes[random()%3],int64_t(1)<<(random()%4),1+random()%19,1+random()%11,1+random()%257,flat));
        }
        for(int fault=0;fault<int(expected_metadata);++fault){run(f::Fixture{},stream,fault);++metadata;}
        require(cases==expected_cases&&metadata==expected_metadata,"reorder checker case count drifted");
        check(hipStreamSynchronize(stream));check(hipStreamDestroy(stream));stream=nullptr;
        std::printf("PASS: %zu local head reorder cases plus %zu descriptor refusals on %s; exact raw bits and guards, no scratch/communication\n",cases,metadata,props.gcnArchName);return 0;
    } catch(const std::exception& error) {
        if(stream){static_cast<void>(hipStreamSynchronize(stream));static_cast<void>(hipStreamDestroy(stream));}
        std::fprintf(stderr,"ds41-reorder-heads-selftest: %s\n",error.what());return 1;
    }
}
