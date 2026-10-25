/* Explicit-gated tiny GPU check. No service API, full weights or automatic CTest execution. */
#include <hip/hip_runtime.h>
#include "ds41_test_scratch.h"
#include "ds41_index_fixture.h"
#include "ds41_index_paged_fixture.h"
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <memory>
#include <span>
#include <stdexcept>
namespace x=rad::ds41::index;
namespace f=x::testing;
namespace pf=rad::ds41::paged_index::testing;
namespace {
template<class V>std::vector<uint8_t> bytes(const V& value){std::vector<uint8_t> out(value.size()*sizeof(*value.data()));std::memcpy(out.data(),value.data(),out.size());return out;}
std::vector<uint32_t> words(const std::vector<uint8_t>& value){std::vector<uint32_t> out(value.size()/4);std::memcpy(out.data(),value.data(),value.size());return out;}
void require(bool yes,const char* message){if(!yes)throw std::runtime_error(message);}
void check(hipError_t rc){if(rc!=hipSuccess)throw std::runtime_error(hipGetErrorString(rc));}
struct Device {
    static constexpr size_t guard=64;void* base=nullptr;size_t size;
    explicit Device(std::span<const uint32_t> words):size(words.size_bytes()){
        check(hipMalloc(&base,size+2*guard));std::vector<uint8_t> b(size+2*guard,0xC7);
        std::memcpy(b.data()+guard,words.data(),size);check(hipMemcpy(base,b.data(),b.size(),hipMemcpyHostToDevice));
    }
    ~Device(){if(base)static_cast<void>(hipFree(base));}
    void* data()const{return static_cast<uint8_t*>(base)+guard;}
    std::vector<uint32_t> read()const {
        std::vector<uint8_t> b(size+2*guard);check(hipMemcpy(b.data(),base,b.size(),hipMemcpyDeviceToHost));
        for(size_t i=0;i<guard;++i)require(b[i]==0xC7&&b[guard+size+i]==0xC7,"GPU index allocation redzone changed");
        std::vector<uint32_t> out(size/4);std::memcpy(out.data(),b.data()+guard,size);return out;
    }
};
template<class Fixture> void run(RadLaunchFn launch,Fixture test,hipStream_t stream) {
    using Plan=std::conditional_t<std::is_same_v<Fixture,pf::Fixture>,rad::ds41::paged_index::Plan,x::Plan>;
    auto oracle=test;const int expected=oracle.run();auto args=test.args();
    std::vector<std::unique_ptr<Device>> dev(size_t(test.nt()));
    for(int i=0;i<test.nt();++i)if(test.buffers[i]){dev[i]=std::make_unique<Device>(test.buffer(i).words);test.tensors[i].data=dev[i]->data();}
    const std::array<uint32_t,4> flag_words{911,912,913,914};const auto flag_before=rad::ds41::testing::make_gpu_scratch<Plan>(bytes(flag_words));Device flag(words(flag_before));args.scratch=flag.data();args.scratch_bytes=flag_before.size();
    require(launch(&args,stream)==RAD_OK,"GPU index launch refused valid metadata");check(hipStreamSynchronize(stream));
    const auto actual_flag=flag.read();require(actual_flag.size()>=4,"bad flag extent");require(rad::ds41::testing::gpu_scratch_guards<Plan>(bytes(actual_flag),flag_before),"GPU index plan staging changed scratch guards");
    if(expected==RAD_OK)require(int32_t(actual_flag[0])==RAD_OK,"GPU index validation failed for valid input");
    else require(int32_t(actual_flag[0])<0,"GPU index failed to report invalid payload");
    for(int i=1;i<4;++i)require(actual_flag[size_t(i)]==uint32_t(test.status[size_t(i)]),"GPU index scratch exceeded4bytes");
    for(int i=0;i<test.nt();++i)if(dev[i]){
        const auto actual=dev[i]->read();
        if(i!=test.output()||expected!=RAD_OK){require(actual==test.buffer(i).words,"index input or failed-batch output changed");continue;}
        auto got=test.buffer(i);got.words=actual;const auto& want=oracle.buffer(i);
        if(got.t.dtype!=RAD_F32){require(actual==want.words,"index selection/padding differs from CPU oracle");continue;}
        std::vector<bool> touched(actual.size()*4,false);
        for(int64_t j=0;j<got.count();++j){
            const float v=got.get(j),w=want.get(j);
            if(std::isinf(w))require(v==w,"invisible index score was not -Inf");
            else require(std::isfinite(v)&&std::fabs(v-w)<=3e-6f+3e-5f*std::fabs(w),"floating index score differs from CPU oracle");
            for(int b=0;b<4;++b)touched[size_t(got.at(j)*4+b)]=true;
        }
        const auto* a=reinterpret_cast<const uint8_t*>(actual.data());const auto* b=reinterpret_cast<const uint8_t*>(want.words.data());
        for(size_t j=0;j<touched.size();++j)if(!touched[j])require(a[j]==b[j],"index output pitch padding changed");
    }
}
RadLaunchFn load(void* plugin,const char* symbol){auto fn=reinterpret_cast<RadLaunchFn>(dlsym(plugin,symbol));require(bool(fn),symbol);return fn;}
}
int main(int argc,char** argv){
    if(argc!=4||std::strcmp(argv[1],"--run-gpu")){
        std::fprintf(stderr,"usage: ds41-index-selftest --run-gpu libds41.so device\nTiny synthetic GPU work: approved idle window only.\n");return 2;
    }
    void* plugin=nullptr;hipStream_t stream=nullptr;
    try{
        size_t used=0;const int device=std::stoi(argv[3],&used);require(used==std::strlen(argv[3])&&device>=0,"invalid device");
        check(hipSetDevice(device));hipDeviceProp_t props{};check(hipGetDeviceProperties(&props,device));
        require(!std::strncmp(props.gcnArchName,"gfx1201",7)&&(!props.gcnArchName[7]||props.gcnArchName[7]==':'),"checker requires gfx1201");
        plugin=dlopen(argv[2],RTLD_NOW|RTLD_LOCAL);if(!plugin)throw std::runtime_error(dlerror());
        const RadLaunchFn launch[]{load(plugin,"ds41_index_scores"),load(plugin,"ds41_candidate_blocks"),load(plugin,"ds41_index_topk")};
        check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));size_t count=0;
        auto one=[&](f::Fixture test){run(launch[int(test.op)],std::move(test),stream);++count;};
        for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(int pitch=0;pitch<3;++pitch)
            for(int dim:{6,128})for(int heads:{3,8})for(int keys:{11,257})one(f::Fixture(x::Op::scores,dtype,pitch,3,keys,4,true,dim,heads));
        for(int pitch=0;pitch<3;++pitch)for(int keys:{1,11,257})for(int top:{1,4,64}){
            one(f::Fixture(x::Op::candidates,RAD_F32,pitch,3,keys,top));
            for(bool mask:{false,true})one(f::Fixture(x::Op::topk,RAD_F32,pitch,3,keys,top,mask));
        }
        for(auto op:{x::Op::candidates,x::Op::topk}){
            f::Fixture maximum(op,RAD_F32,0,1,4097,4096,false);maximum.visible().id(0,4097);one(std::move(maximum));
        }
        for(auto op:{x::Op::scores,x::Op::candidates,x::Op::topk}){
            f::Fixture low(op);low.visible().id(2,-1);one(std::move(low));
            f::Fixture high(op);high.visible().id(2,12);one(std::move(high));
            f::Fixture bad(op);bad.buffer(0).set(bad.buffer(0).count()-1,NAN);one(std::move(bad));
        }
        f::Fixture bad_seq(x::Op::scores);bad_seq.buffer(3).id(2,3);one(std::move(bad_seq));
        f::Fixture bad_key(x::Op::scores);bad_key.buffer(1).set((2*11+10)*6+5,NAN);one(std::move(bad_key));
        f::Fixture ignored(x::Op::scores);for(int i=0;i<11*6;++i)ignored.buffer(1).set(11*6+i,NAN);one(std::move(ignored));
        f::Fixture newest(x::Op::candidates,RAD_F32,2,1,11,1);newest.visible().id(0,9);newest.buffer(0).set(10,NAN);one(std::move(newest));
        f::Fixture invisible(x::Op::topk,RAD_F32,2,1,11,4,false);invisible.visible().id(0,10);invisible.buffer(0).set(10,NAN);one(std::move(invisible));
        require(count==169,"index checker case count drifted");
        const auto paged=load(plugin,"ds41_index_scores_paged");size_t paged_count=0;
        auto page=[&](pf::Fixture test){run(paged,std::move(test),stream);++paged_count;};
        for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(int pitch=0;pitch<3;++pitch)
            for(int ratio:{1,2})for(int block:{ratio,4,8})for(int dim:{8,128})for(int heads:{3,8})
                page(pf::Fixture(dtype,pitch,ratio,block,dim,heads,pitch!=1));
        pf::Fixture missing;missing.buffer(5).id(2*missing.blocks_per_sequence+missing.blocks_per_sequence-1,-1);page(std::move(missing));
        pf::Fixture beyond;beyond.buffer(5).id(2*beyond.blocks_per_sequence,int32_t(beyond.physical_blocks));page(std::move(beyond));
        pf::Fixture origin;origin.buffer(6).id(2,4);page(std::move(origin));
        pf::Fixture seq;seq.buffer(3).id(2,3);page(std::move(seq));
        pf::Fixture vis;vis.buffer(4).id(2,12);page(std::move(vis));
        pf::Fixture value;value.buffer(1).set(value.physical_scalar(2,10,7),NAN);page(std::move(value));
        pf::Fixture negative;negative.buffer(5).id(2*negative.blocks_per_sequence,-2);page(std::move(negative));
        pf::Fixture short_table;short_table.buffer(5).t.shape[1]--;page(std::move(short_table));
        pf::Fixture unused;unused.buffer(6).id(1,123);
        for(int64_t i=0;i<unused.blocks_per_sequence;++i)unused.buffer(5).id(unused.blocks_per_sequence+i,-1);
        page(std::move(unused));require(paged_count==225,"paged index case count drifted");
        check(hipStreamDestroy(stream));stream=nullptr;dlclose(plugin);plugin=nullptr;
        std::printf("PASS: %zu dense/selection + %zu paged index GPU cases on %s; guarded scoring/heaps/causal visibility/status; no model-quality or speed claim\n",count,paged_count,props.gcnArchName);return 0;
    }catch(const std::exception& e){
        if(stream){static_cast<void>(hipStreamSynchronize(stream));static_cast<void>(hipStreamDestroy(stream));}if(plugin)dlclose(plugin);
        std::fprintf(stderr,"FAIL index checker: %s\n",e.what());return 1;
    }
}
