/* Tiny direct native-HIP checker, never automatic: 334 payload cases +15 descriptor failures. */
#include <hip/hip_runtime.h>
#include "ds41_index_stream_fixture.h"
#include "ds41_test_scratch.h"
#include <cstdio>
#include <memory>
#include <span>
#include <string>
namespace s=rad::ds41::index_stream;
namespace f=s::testing;
namespace {
constexpr size_t expected_cases=334,expected_metadata=15;
void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
void check(hipError_t rc){if(rc!=hipSuccess)throw std::runtime_error(hipGetErrorString(rc));}
template<class V>std::vector<uint8_t> bytes(const V& value){std::vector<uint8_t> out(value.size()*sizeof(*value.data()));std::memcpy(out.data(),value.data(),out.size());return out;}
std::vector<uint32_t> words(const std::vector<uint8_t>& value){require(value.size()%4==0,"scratch is not whole words");std::vector<uint32_t> out(value.size()/4);std::memcpy(out.data(),value.data(),value.size());return out;}
struct Device {
    void* base=nullptr;size_t size;static constexpr size_t guard=64;
    explicit Device(std::span<const uint32_t> words):size(words.size_bytes()){
        std::vector<uint8_t> host(size+2*guard,0xC7);std::memcpy(host.data()+guard,words.data(),size);check(hipMalloc(&base,host.size()));
        const auto rc=hipMemcpy(base,host.data(),host.size(),hipMemcpyHostToDevice);if(rc!=hipSuccess){static_cast<void>(hipFree(base));base=nullptr;check(rc);}
    }
    Device(const Device&)=delete;Device& operator=(const Device&)=delete;
    ~Device(){if(base)static_cast<void>(hipFree(base));}
    void* data()const{return static_cast<uint8_t*>(base)+guard;}
    std::vector<uint32_t> read()const{
        std::vector<uint8_t> host(size+2*guard);check(hipMemcpy(host.data(),base,host.size(),hipMemcpyDeviceToHost));
        for(size_t i=0;i<guard;++i)require(host[i]==0xC7&&host[guard+size+i]==0xC7,"stream index allocation red zone changed");
        std::vector<uint32_t> out(size/4);std::memcpy(out.data(),host.data()+guard,size);return out;
    }
};
enum class Scenario{normal,ties,empty,query_nan,weight_inf,cache_nan,page,sequence,visibility,origin,overflow,
                    duplicate,descending,hole,negative,future,empty_with_ids,ignore_unselected,ignore_tail};
void arrange(f::Fixture& c,Scenario scenario){
    if(scenario==Scenario::ties)for(int64_t i=0;i<c.buffer(s::query).count();++i)c.buffer(s::query).set(i,0);
    if(scenario==Scenario::empty){for(int64_t i=0;i<c.rows;++i)c.buffer(s::visible).id(i,0);if(c.mode==s::Mode::consumer)c.default_candidates();}
    if(scenario==Scenario::query_nan)c.buffer(s::query).set(c.buffer(s::query).count()-1,NAN);
    if(scenario==Scenario::weight_inf)c.buffer(s::head_weights).set(c.buffer(s::head_weights).count()-1,INFINITY);
    if(scenario==Scenario::cache_nan)c.buffer(s::cache).set(c.scalar_index(2,0,c.dim-1),NAN);
    if(scenario==Scenario::page)c.buffer(s::block_table).id(2*c.columns,INT32_MAX);
    if(scenario==Scenario::sequence)c.buffer(s::sequence_ids).id(2,3);
    if(scenario==Scenario::visibility)c.buffer(s::visible).id(2,int32_t(c.keys+1));
    if(scenario==Scenario::origin)c.buffer(s::origins).id(2,4);
    if(scenario==Scenario::overflow){for(int64_t i=0;i<c.buffer(s::query).count();++i)c.buffer(s::query).set(i,1e30f);for(int64_t i=0;i<c.buffer(s::cache).count();++i)c.buffer(s::cache).set(i,1e30f);}
    if(scenario>=Scenario::duplicate&&scenario<=Scenario::empty_with_ids){const int64_t first=2*c.top_blocks;
        for(int64_t i=0;i<3;++i)c.buffer(s::candidates_in).id(first+i,int32_t(i));
        if(scenario==Scenario::duplicate)c.buffer(s::candidates_in).id(first+1,0);
        if(scenario==Scenario::descending){c.buffer(s::candidates_in).id(first,1);c.buffer(s::candidates_in).id(first+1,0);}
        if(scenario==Scenario::hole)c.buffer(s::candidates_in).id(first+1,-1);
        if(scenario==Scenario::negative)c.buffer(s::candidates_in).id(first+1,-2);
        if(scenario==Scenario::future)c.buffer(s::candidates_in).id(first+2,4);
        if(scenario==Scenario::empty_with_ids)c.buffer(s::visible).id(2,0);
    }
    if(scenario==Scenario::ignore_unselected||scenario==Scenario::ignore_tail){
        for(int64_t i=0;i<c.buffer(s::candidates_in).count();++i)c.buffer(s::candidates_in).id(i,-1);
        c.buffer(s::visible).id(2,9);c.buffer(s::candidates_in).id(2*c.top_blocks,scenario==Scenario::ignore_tail?1:0);
        c.buffer(s::cache).set(c.scalar_index(2,scenario==Scenario::ignore_tail?9:8,c.dim-1),NAN);c.buffer(s::block_table).id(2*c.columns+5,INT32_MAX);
    }
}
void run(f::Fixture c,Scenario scenario,hipStream_t stream){
    arrange(c,scenario);auto oracle=c;const int expected=oracle.run();std::array<std::unique_ptr<Device>,s::operand_count> device;
    for(int i=0;i<s::operand_count;++i)if(c.buffers[size_t(i)])device[size_t(i)]=std::make_unique<Device>(c.buffer(i).words);
    const std::array<uint32_t,4> sentinels{911,912,913,914};const auto scratch_before=rad::ds41::testing::make_gpu_scratch<s::Plan>(bytes(sentinels));Device scratch(words(scratch_before));auto args=c.args();
    for(int i=0;i<s::operand_count;++i)if(device[size_t(i)])c.tensors[size_t(i)].data=device[size_t(i)]->data();
    args.scratch=scratch.data();args.scratch_bytes=rad::ds41::gpu_scratch_bytes<s::Plan>();
    require(ds41_index_stream(&args,stream)==RAD_OK,"stream index enqueue failed for valid descriptors");check(hipStreamSynchronize(stream));
    const auto flag=scratch.read();require(int32_t(flag[0])==expected,"stream index device status differs from CPU preflight");
    require(rad::ds41::testing::gpu_scratch_guards<s::Plan>(bytes(flag),scratch_before),"stream index status guard or staged-Plan tail changed");
    for(int i=0;i<s::operand_count;++i)if(device[size_t(i)]){
        const auto& wanted=i<s::selected_out||expected!=RAD_OK?c.buffer(i).words:oracle.buffer(i).words;
        require(device[size_t(i)]->read()==wanted,"stream index IDs, source bytes or output padding differ from CPU oracle");
    }
}
enum class Metadata{output_input,outputs_alias,candidate_cache,output_dtype,output_stride,cache_stride,short_scratch,scratch_alias,scratch_wrap,partial_heads,mode,unexpected_candidates,plan_short,plan_alias,plan_alignment};
void reject(Metadata invalid,hipStream_t stream){
    f::Fixture c;std::array<std::unique_ptr<Device>,s::operand_count> device;for(int i=0;i<s::operand_count;++i)if(c.buffers[size_t(i)])device[size_t(i)]=std::make_unique<Device>(c.buffer(i).words);
    const std::array<uint32_t,4> sentinels{911,912,913,914};const auto scratch_before=rad::ds41::testing::make_gpu_scratch<s::Plan>(bytes(sentinels));Device scratch(words(scratch_before));auto args=c.args();
    for(int i=0;i<s::operand_count;++i)if(device[size_t(i)])c.tensors[size_t(i)].data=device[size_t(i)]->data();
    args.scratch=scratch.data();args.scratch_bytes=rad::ds41::gpu_scratch_bytes<s::Plan>();
    if(invalid==Metadata::output_input)c.tensors[s::selected_out].data=c.tensors[s::query].data;
    if(invalid==Metadata::outputs_alias)c.tensors[s::candidates_out].data=c.tensors[s::selected_out].data;
    if(invalid==Metadata::candidate_cache)c.tensors[s::candidates_out].data=c.tensors[s::cache].data;
    if(invalid==Metadata::output_dtype)c.tensors[s::selected_out].dtype=RAD_F32;
    if(invalid==Metadata::output_stride)c.tensors[s::selected_out].stride[1]=0;
    if(invalid==Metadata::cache_stride)c.tensors[s::cache].stride[0]=INT64_MAX;
    if(invalid==Metadata::short_scratch)args.scratch_bytes=3;
    if(invalid==Metadata::scratch_alias)args.scratch=c.tensors[s::cache].data;
    if(invalid==Metadata::scratch_wrap)args.scratch=reinterpret_cast<void*>(UINTPTR_MAX-3);
    if(invalid==Metadata::partial_heads)c.integer("global_heads",c.heads*2);
    if(invalid==Metadata::mode)for(auto& p:c.params)if(!std::strcmp(p.key,"mode"))p.sval="bad";
    if(invalid==Metadata::unexpected_candidates)c.tensors[s::candidates_in]=c.tensors[s::candidates_out];
    if(invalid==Metadata::plan_short)--args.scratch_bytes;
    if(invalid==Metadata::plan_alias)c.tensors[s::query].data=static_cast<uint8_t*>(scratch.data())+rad::ds41::device_plan_offset;
    if(invalid==Metadata::plan_alignment)args.scratch=static_cast<uint8_t*>(scratch.data())+4;
    require(ds41_index_stream(&args,stream)<0,"invalid stream descriptor accepted");check(hipStreamSynchronize(stream));
    require(bytes(scratch.read())==scratch_before,"stream metadata/staging failure changed any scratch byte");
    for(int i=0;i<s::operand_count;++i)if(device[size_t(i)])require(device[size_t(i)]->read()==c.buffer(i).words,"stream metadata failure changed operand");
}
}
int main(int argc,char** argv){
    if((argc!=2&&argc!=3)||std::strcmp(argv[1],"--run-gpu")){
        std::fprintf(stderr,"usage: ds41-index-stream-selftest --run-gpu [device]\n334 tiny payload cases plus15 descriptor failures; never run automatically.\n");return 2;
    }
    hipStream_t stream=nullptr;
    try{
        int device=0;if(argc==3){size_t n=0;device=std::stoi(argv[2],&n);require(n==std::strlen(argv[2])&&device>=0,"invalid device");}
        check(hipSetDevice(device));hipDeviceProp_t props{};check(hipGetDeviceProperties(&props,device));
        require(!std::strncmp(props.gcnArchName,"gfx1201",7)&&(!props.gcnArchName[7]||props.gcnArchName[7]==':'),"checker requires gfx1201");
        check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));size_t cases=0,metadata=0;
        auto one=[&](f::Fixture c,Scenario scenario=Scenario::normal){try{run(c,scenario,stream);++cases;}catch(const std::exception& e){std::fprintf(stderr,"case=%zu mode=%s dtype=%s pitch=%d ratio=%lld K=%lld D=%lld scenario=%u: %s\n",cases+1,f::name(c.mode),rad_dtype_name(c.dtype),c.pitch,(long long)c.ratio,(long long)c.keys,(long long)c.dim,unsigned(scenario),e.what());throw;}};
        for(int keys:{11,129})for(auto mode:{s::Mode::source,s::Mode::consumer,s::Mode::plain})for(uint32_t d:{RAD_F32,RAD_F16,RAD_BF16})for(int p=0;p<3;++p)for(int ratio:{1,2})one(f::Fixture(mode,d,p,ratio,keys)); //108
        for(auto mode:{s::Mode::source,s::Mode::consumer,s::Mode::plain})for(uint32_t d:{RAD_F32,RAD_F16,RAD_BF16})for(int p=0;p<3;++p)one(f::Fixture(mode,d,p,2,33,128,4)); //27
        for(auto mode:{s::Mode::source,s::Mode::consumer,s::Mode::plain})for(uint32_t d:{RAD_F32,RAD_F16,RAD_BF16})for(int p=0;p<3;++p)one(f::Fixture(mode,d,p,2,131,8,2,4,3,3)); //27
        for(auto mode:{s::Mode::source,s::Mode::consumer,s::Mode::plain})for(int p=0;p<3;++p)one(f::Fixture(mode,RAD_F32,p,2,19,8,2,4,3,8,7)); //9
        for(auto scenario:{Scenario::ties,Scenario::empty})for(auto mode:{s::Mode::source,s::Mode::consumer,s::Mode::plain})for(uint32_t d:{RAD_F32,RAD_F16,RAD_BF16})for(int p=0;p<3;++p)one(f::Fixture(mode,d,p),scenario); //54
        for(auto scenario:{Scenario::query_nan,Scenario::weight_inf,Scenario::cache_nan,Scenario::page,Scenario::sequence,Scenario::visibility,Scenario::origin,Scenario::overflow})for(auto mode:{s::Mode::source,s::Mode::consumer,s::Mode::plain})for(int p=0;p<3;++p)one(f::Fixture(mode,RAD_F32,p),scenario); //72
        for(auto scenario:{Scenario::duplicate,Scenario::descending,Scenario::hole,Scenario::negative,Scenario::future,Scenario::empty_with_ids})for(int p=0;p<3;++p)one(f::Fixture(s::Mode::consumer,RAD_F32,p,2,25),scenario); //18
        for(auto scenario:{Scenario::ignore_unselected,Scenario::ignore_tail})for(uint32_t d:{RAD_F32,RAD_F16,RAD_BF16})for(int p=0;p<3;++p)one(f::Fixture(s::Mode::consumer,d,p),scenario); //18
        f::Fixture maximum(s::Mode::source,RAD_F32,0,2,16389,4,1,1024,2048,8,1);maximum.buffer(s::visible).id(0,int32_t(maximum.keys));one(std::move(maximum)); //1
        for(auto invalid:{Metadata::output_input,Metadata::outputs_alias,Metadata::candidate_cache,Metadata::output_dtype,Metadata::output_stride,Metadata::cache_stride,Metadata::short_scratch,Metadata::scratch_alias,Metadata::scratch_wrap,Metadata::partial_heads,Metadata::mode,Metadata::unexpected_candidates,Metadata::plan_short,Metadata::plan_alias,Metadata::plan_alignment}){reject(invalid,stream);++metadata;}
        require(cases==expected_cases&&metadata==expected_metadata,"stream checker case count drifted");check(hipStreamSynchronize(stream));check(hipStreamDestroy(stream));stream=nullptr;
        std::printf("PASS: %zu tiny streaming index GPU cases plus%zu descriptor failures on %s; exact IDs, compact lists, global atomic validation and guards\n",cases,metadata,props.gcnArchName);return 0;
    }catch(const std::exception& e){if(stream){static_cast<void>(hipStreamSynchronize(stream));static_cast<void>(hipStreamDestroy(stream));}std::fprintf(stderr,"ds41-index-stream-selftest: %s\n",e.what());return 1;}
}
