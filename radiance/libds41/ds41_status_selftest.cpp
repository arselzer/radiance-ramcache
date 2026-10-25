/* Tiny native-HIP collector check; core Ctx/Engine ordering has separate host/AQL checks. */
#include <hip/hip_runtime.h>
#include "rad_deferred_status.h"
#include <array>
#include <cstring>
#include <cstdio>
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
void require(bool value,const char* why){if(!value)throw std::runtime_error(why);}
void check(hipError_t rc){if(rc!=hipSuccess)throw std::runtime_error(hipGetErrorString(rc));}
struct Device {
    void* base=nullptr;size_t size;static constexpr size_t guard=64;
    explicit Device(size_t bytes):size(bytes){check(hipMalloc(&base,size+2*guard));check(hipMemset(base,0xA7,size+2*guard));}
    ~Device(){if(base)static_cast<void>(hipFree(base));}
    void* data()const{return static_cast<uint8_t*>(base)+guard;}
    void upload(const void* p)const{check(hipMemcpy(data(),p,size,hipMemcpyHostToDevice));}
    std::vector<uint8_t> read()const{
        std::vector<uint8_t> bytes(size+2*guard);check(hipMemcpy(bytes.data(),base,bytes.size(),hipMemcpyDeviceToHost));
        for(size_t i=0;i<guard;++i)require(bytes[i]==0xA7&&bytes[guard+size+i]==0xA7,"status collector allocation redzone changed");
        return {bytes.begin()+guard,bytes.begin()+guard+size};
    }
    RadDeviceStatus status()const{const auto b=read();RadDeviceStatus s{};std::memcpy(&s,b.data(),sizeof s);return s;}
};
template<class T>T symbol(void* p,const char* name){auto f=reinterpret_cast<T>(dlsym(p,name));require(bool(f),name);return f;}
}
int main(int argc,char** argv){
    if(argc!=4||std::strcmp(argv[1],"--run-gpu")){
        std::fprintf(stderr,"usage: ds41-status-selftest --run-gpu libds41.so device\nApproved idle-window tiny GPU work only.\n");return 2;
    }
    void* plugin=nullptr;hipStream_t first=nullptr,second=nullptr;hipEvent_t reset=nullptr;
    try{
        size_t used=0;const int device=std::stoi(argv[3],&used);require(used==std::strlen(argv[3])&&device>=0,"invalid device");
        check(hipSetDevice(device));hipDeviceProp_t props{};check(hipGetDeviceProperties(&props,device));
        require(!std::strncmp(props.gcnArchName,"gfx1201",7)&&(!props.gcnArchName[7]||props.gcnArchName[7]==':'),"checker requires gfx1201");
        plugin=dlopen(argv[2],RTLD_NOW|RTLD_LOCAL);if(!plugin)throw std::runtime_error(dlerror());
        // This executable deliberately does not link the core. Direct checker
        // launches own their flags; ordinary Registry loading must refuse it.
        const auto open=symbol<int(*)()>(plugin,"rad_plugin_open");
        require(open()==RAD_E_ABI,"experimental GPU plugin did not refuse a core without status capability");
        const auto version=symbol<RadPluginDeferredStatusVersionFn>(plugin,"rad_plugin_deferred_status_version");
        const auto query=symbol<RadKernelDeferredStatusFn>(plugin,"rad_kernel_deferred_status");
        const auto count=symbol<int(*)()>(plugin,"rad_kernel_count");
        const auto at=symbol<const RadKernelInfo*(*)(int)>(plugin,"rad_kernel_at");
        require(version()==RAD_DEFERRED_STATUS_VERSION,"collector version mismatch");
        auto find=[&](const char* op){for(int i=0;i<count();++i)if(!std::strcmp(at(i)->op,op))return query(i);return RadDeferredStatusLaunchFn(nullptr);};
        const auto scalar=find("ds41_hc_prepare"),routed=find("ds41_routed_packed"),store=find("ds41_latent_store"),gather=find("ds41_latent_gather");
        require(scalar&&routed&&store&&gather,"required collector absent");
        require(!query(-1)&&!query(count())&&!find("ds41_gemm_packed"),"collector attached to unsupported row");
        check(hipStreamCreateWithFlags(&first,hipStreamNonBlocking));check(hipStreamCreateWithFlags(&second,hipStreamNonBlocking));check(hipEventCreateWithFlags(&reset,hipEventDisableTiming));
        Device latch(sizeof(RadDeviceStatus)),scratch(4),vector(9*4);auto* word=static_cast<RadDeviceStatus*>(latch.data());
        const RadDeviceStatus zero{0,0};const int32_t good=0,bad=RAD_E_INVAL,next=RAD_E_STATE;
        latch.upload(&zero);scratch.upload(&good);RadArgs a{};a.scratch=scratch.data();a.scratch_bytes=4;
        require(scalar(&a,word,17,first)==RAD_OK,"success collector enqueue");check(hipStreamSynchronize(first));
        auto s=latch.status();require(!s.code&&!s.op,"success changed persistent status");
        scratch.upload(&bad);require(scalar(&a,word,17,first)==RAD_OK,"error collector enqueue");
        check(hipMemsetAsync(scratch.data(),0,4,first));require(scalar(&a,word,99,first)==RAD_OK,"scratch reuse collector enqueue");check(hipStreamSynchronize(first));
        s=latch.status();require(s.code==bad&&s.op==17,"scratch reuse lost first deferred error");
        scratch.upload(&next);require(scalar(&a,word,29,first)==RAD_OK,"later error enqueue");check(hipStreamSynchronize(first));
        s=latch.status();require(s.code==bad&&s.op==17,"later error replaced persistent first error");
        const std::array<int32_t,9> flags{0,555,0,555,RAD_E_SHAPE,555,RAD_E_INVAL,555,0};vector.upload(flags.data());
        std::array<RadTensor,6> tensors{};RadTensor t{};t.data=vector.data();t.dtype=RAD_I32;t.rank=1;t.shape[0]=5;t.stride[0]=2;
        for(const auto& pair:std::array<std::pair<RadDeferredStatusLaunchFn,int>,3>{{{routed,5},{store,4},{gather,3}}}){
            latch.upload(&zero);tensors={};tensors[size_t(pair.second)]=t;a={};a.t=tensors.data();a.n_t=int(tensors.size());
            require(pair.first(&a,word,42,first)==RAD_OK,"pitched vector collector enqueue");check(hipStreamSynchronize(first));
            s=latch.status();require((s.code==RAD_E_SHAPE||s.code==RAD_E_INVAL)&&s.op==42,"pitched vector status not latched");
        }
        require(vector.read()==std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(flags.data()),reinterpret_cast<const uint8_t*>(flags.data())+sizeof flags),"collector modified status source");
        // Reset before either lane; race two valid collectors and accept only the winning pair.
        latch.upload(&zero);scratch.upload(&bad);check(hipEventRecord(reset,first));check(hipStreamWaitEvent(second,reset,0));
        a={};a.scratch=scratch.data();a.scratch_bytes=4;RadArgs b{};b.t=tensors.data();b.n_t=int(tensors.size());tensors[5]=t;
        require(scalar(&a,word,101,first)==RAD_OK,"lane0 collector enqueue");require(routed(&b,word,202,second)==RAD_OK,"lane1 collector enqueue");
        check(hipStreamSynchronize(first));check(hipStreamSynchronize(second));s=latch.status();
        require((s.op==101&&s.code==bad)||(s.op==202&&(s.code==RAD_E_SHAPE||s.code==RAD_E_INVAL)),"cross-lane latch code/op pair inconsistent");
        const auto saved=latch.read();a.scratch=latch.data();require(scalar(&a,word,1,first)==RAD_E_INVAL,"source/latch alias accepted");
        a.scratch=reinterpret_cast<void*>(UINTPTR_MAX-3);require(scalar(&a,word,1,first)==RAD_E_SHAPE,"scratch pointer wrap accepted");
        a.scratch=scratch.data();a.scratch_bytes=3;require(scalar(&a,word,1,first)==RAD_E_INVAL,"short source flag accepted");
        require(latch.read()==saved,"metadata refusal changed latch");
        check(hipEventDestroy(reset));reset=nullptr;check(hipStreamDestroy(first));first=nullptr;check(hipStreamDestroy(second));second=nullptr;
        dlclose(plugin);plugin=nullptr;std::puts("PASS tiny GPU deferred-status collectors: stale-core refusal, scratch reuse, pitched arrays, first-error CAS, two lanes, source immutability and guards");return 0;
    }catch(const std::exception& e){
        if(first)static_cast<void>(hipStreamSynchronize(first));
        if(second)static_cast<void>(hipStreamSynchronize(second));
        if(reset)static_cast<void>(hipEventDestroy(reset));
        if(first)static_cast<void>(hipStreamDestroy(first));
        if(second)static_cast<void>(hipStreamDestroy(second));
        if(plugin)dlclose(plugin);
        std::fprintf(stderr,"FAIL status checker: %s\n",e.what());return 1;
    }
}
