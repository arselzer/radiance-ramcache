/* Staged synthetic checker, explicitly gated: 352 device cases plus 8 descriptor failures. */
#include <cstring>
#include <hip/hip_runtime.h>
#include "ds41_test_scratch.h"
#include "ds41_attention_paged_fixture.h"
#include <dlfcn.h>
#include <cstdio>
#include <memory>
#include <string>
namespace a=rad::ds41::attention_paged;
namespace f=a::testing;
namespace {
constexpr size_t expected_cases=352,expected_metadata=8;
void check(hipError_t rc){if(rc!=hipSuccess)throw std::runtime_error(hipGetErrorString(rc));}
void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
template<class V>std::vector<uint8_t> bytes(const V& value){std::vector<uint8_t> out(value.size()*sizeof(*value.data()));std::memcpy(out.data(),value.data(),out.size());return out;}
struct Device {
    void* base=nullptr;size_t size=0;static constexpr size_t guard=64;
    explicit Device(const std::vector<uint8_t>& value):size(value.size()) {
        check(hipMalloc(&base,size+2*guard));auto rc=hipMemset(base,0xA7,size+2*guard);
        if(rc==hipSuccess)rc=hipMemcpy(data(),value.data(),size,hipMemcpyHostToDevice);
        if(rc!=hipSuccess){static_cast<void>(hipFree(base));base=nullptr;check(rc);}
    }
    Device(const Device&)=delete;Device& operator=(const Device&)=delete;
    ~Device(){if(base)static_cast<void>(hipFree(base));}
    void* data()const{return static_cast<uint8_t*>(base)+guard;}
    std::vector<uint8_t> read()const {
        std::vector<uint8_t> all(size+2*guard);check(hipMemcpy(all.data(),base,all.size(),hipMemcpyDeviceToHost));
        require(std::all_of(all.begin(),all.begin()+guard,[](uint8_t b){return b==0xA7;})&&std::all_of(all.end()-guard,all.end(),[](uint8_t b){return b==0xA7;}),"GPU allocation red zone changed");
        return {all.begin()+guard,all.end()-guard};
    }
};
enum class Scenario {normal,duplicates,disabled_sink,dominant_sink,unused_sequence,all_padding,
                     future_id,negative_id,window_page,compressed_page,window_nan,compressed_inf,query_nan,sink_inf,
                     context,position,cu,used,compressed_used};
void arrange(f::Fixture& c,Scenario s) {
    if(s==Scenario::duplicates)for(int64_t row=0;row<c.rows;++row)for(int64_t k=0;k<c.topk;++k)c.buffer(a::selected).id(row*c.topk+k,0);
    if(s==Scenario::disabled_sink||s==Scenario::dominant_sink)for(int64_t h=0;h<c.heads;++h)c.buffer(a::sinks).set(h,s==Scenario::disabled_sink?-INFINITY:10000.f);
    if(s==Scenario::unused_sequence) {
        c.buffer(a::ctx_lens).id(1,-123);c.buffer(a::window_used).id(1,-456);c.buffer(a::compressed_used).id(1,-789);
        for(int table:{a::window_table,a::compressed_table})for(int64_t j=0;j<c.buffer(table).t.shape[1];++j)c.buffer(table).id(c.buffer(table).t.shape[1]+j,INT32_MAX);
    }
    if(s==Scenario::all_padding) {
        for(int64_t j=0;j<c.buffer(a::selected).count();++j)c.buffer(a::selected).id(j,-1);
        for(int64_t j=0;j<c.buffer(a::compressed_table).count();++j)c.buffer(a::compressed_table).id(j,INT32_MAX);
        for(int64_t j=0;j<c.buffer(a::compressed_cache).count();++j)c.buffer(a::compressed_cache).set(j,NAN);
    }
    if(s==Scenario::future_id)c.buffer(a::selected).id((c.rows-1)*c.topk,int32_t((c.buffer(a::positions).id(c.rows-1)+1)/c.ratio));
    if(s==Scenario::negative_id)c.buffer(a::selected).id((c.rows-1)*c.topk,-2);
    if(s==Scenario::window_page)c.buffer(a::window_table).id(c.buffer(a::window_table).count()-1,-1);
    if(s==Scenario::compressed_page)c.buffer(a::compressed_table).id(2*c.buffer(a::compressed_table).t.shape[1],INT32_MAX);
    if(s==Scenario::window_nan)c.buffer(a::window_cache).set(c.scalar_index(false,2,13,c.dim-1),NAN);
    if(s==Scenario::compressed_inf) {
        c.buffer(a::selected).id((c.rows-1)*c.topk,0);c.buffer(a::compressed_cache).set(c.scalar_index(true,2,0,c.dim-1),INFINITY);
    }
    if(s==Scenario::query_nan)c.buffer(a::query).set(c.buffer(a::query).count()-1,NAN);
    if(s==Scenario::sink_inf)c.buffer(a::sinks).set(c.heads-1,INFINITY);
    if(s==Scenario::context)c.buffer(a::ctx_lens).id(2,INT32_MAX);
    if(s==Scenario::position)c.buffer(a::positions).id(c.rows-1,12);
    if(s==Scenario::cu)c.buffer(a::cu).id(c.sequences,int32_t(c.rows+1));
    if(s==Scenario::used)c.buffer(a::window_used).id(2,13);
    if(s==Scenario::compressed_used)c.buffer(a::compressed_used).id(2,13);
}
void promote(f::Fixture& c) {
    for(auto& b:c.buffers)if(b&&(b->t.dtype==RAD_F16||b->t.dtype==RAD_BF16)) {
        const auto values=b->values();b->t.dtype=RAD_F32;for(int64_t j=0;j<b->count();++j)b->set(j,values[size_t(j)]);
    }
    for(auto& p:c.params)if(!std::strcmp(p.key,"dtype"))p.sval="f32";c.refresh();
}
uint16_t halfword(const std::vector<uint8_t>& value,size_t at){uint16_t out;std::memcpy(&out,value.data()+at,2);return out;}
int ordered(uint16_t h){return h&0x8000u?32768-int(h&0x7FFFu):32768+int(h);}
float scalar(const std::vector<uint8_t>& value,uint32_t dtype,size_t at) {
    if(dtype==RAD_F32){float out;std::memcpy(&out,value.data()+at*4,4);return out;}
    const uint16_t h=halfword(value,at*2);return dtype==RAD_F16?rad_f16_to_f32(h):rad_bf16_to_f32(h);
}
void run(RadLaunchFn launch,f::Fixture c,Scenario scenario,hipStream_t stream) {
    arrange(c,scenario);auto oracle=c;const int expected=oracle.run();auto raw=c;promote(raw);
    require((raw.run()==RAD_OK)==(expected==RAD_OK),"FP32 shadow preflight disagrees");
    std::array<std::unique_ptr<Device>,a::tensors_count> device;
    for(int i=0;i<a::tensors_count;++i)if(c.buffers[size_t(i)])device[size_t(i)]=std::make_unique<Device>(bytes(c.buffer(i).words));
    const auto flag_before=rad::ds41::testing::make_gpu_scratch<a::Plan>(bytes(c.flag));Device scratch(flag_before);auto args=c.args();
    for(int i=0;i<a::tensors_count;++i)if(device[size_t(i)])c.tensors[size_t(i)].data=device[size_t(i)]->data();args.scratch=scratch.data();args.scratch_bytes=scratch.size;
    require(launch(&args,stream)==RAD_OK,"GPU enqueue failed for valid descriptors");check(hipStreamSynchronize(stream));
    const auto flag=scratch.read();int32_t status;std::memcpy(&status,flag.data()+a::status_offset,4);
    require(status==expected,"GPU validation flag differs from CPU preflight");
    require(rad::ds41::testing::gpu_scratch_guards<a::Plan>(flag,flag_before),"GPU status wrote outside its four-byte reservation");
    for(int i=0;i<a::tensors_count;++i)if(device[size_t(i)]) {
        const auto actual=device[size_t(i)]->read(),initial=bytes(c.buffer(i).words);
        if(i!=a::output||expected!=RAD_OK){require(actual==initial,"GPU changed readonly input or invalid-batch output");continue;}
        const auto& b=c.buffer(i);const int width=rad_dtype_bits(b.t.dtype)/8;std::vector<bool> touched(actual.size(),false);
        for(int64_t j=0;j<b.count();++j) {
            const size_t at=size_t(b.at(j));const float got=scalar(actual,b.t.dtype,at),reference=oracle.buffer(i).get(j),unrounded=raw.buffer(i).get(j);
            require(std::isfinite(got)&&std::isfinite(reference)&&std::isfinite(unrounded),"nonfinite attention output");
            const float tolerance=4e-6f+4e-5f*std::fabs(unrounded);
            if(b.t.dtype==RAD_F32)require(std::fabs(got-reference)<=tolerance,"F32 direct attention differs from CPU oracle");
            else {
                const uint16_t encoded=b.t.dtype==RAD_F16?rad_f32_to_f16(unrounded):rad_f32_to_bf16(unrounded),received=halfword(actual,at*2);
                if(encoded!=received){require(std::abs(ordered(encoded)-ordered(received))<=1,"attention differs by more than one storage step");require(std::fabs(unrounded-.5f*(got+reference))<=tolerance,"storage discrepancy away from rounding midpoint");}
            }
            if(got==0&&reference==0)require(std::signbit(got)==std::signbit(reference),"attention zero sign differs");
            if(scenario==Scenario::dominant_sink)require(got==0&&!std::signbit(got),"dominant zero-value sink did not produce positive zero");
            for(int byte=0;byte<width;++byte)touched[at*size_t(width)+size_t(byte)]=true;
        }
        for(size_t j=0;j<actual.size();++j)if(!touched[j])require(actual[j]==initial[j],"GPU output padding/canary changed");
    }
}
enum class Metadata {alias,stride,dtype,short_scratch,scratch_alias,scratch_wrap,speculation,compressed_shape};
void reject_metadata(RadLaunchFn launch,Metadata invalid,hipStream_t stream) {
    f::Fixture c;std::array<std::unique_ptr<Device>,a::tensors_count> device;
    for(int i=0;i<a::tensors_count;++i)if(c.buffers[size_t(i)])device[size_t(i)]=std::make_unique<Device>(bytes(c.buffer(i).words));
    const auto before=rad::ds41::testing::make_gpu_scratch<a::Plan>(bytes(c.flag));Device scratch(before);auto args=c.args();
    for(int i=0;i<a::tensors_count;++i)if(device[size_t(i)])c.tensors[size_t(i)].data=device[size_t(i)]->data();args.scratch=scratch.data();args.scratch_bytes=scratch.size;
    if(invalid==Metadata::alias)c.tensors[a::output].data=c.tensors[a::window_cache].data;
    if(invalid==Metadata::stride)c.tensors[a::window_table].stride[0]=INT64_MAX;
    if(invalid==Metadata::dtype)c.tensors[a::output].dtype=RAD_U32;
    if(invalid==Metadata::short_scratch)args.scratch_bytes=3;
    if(invalid==Metadata::scratch_alias)args.scratch=c.tensors[a::window_cache].data;
    if(invalid==Metadata::scratch_wrap)args.scratch=reinterpret_cast<void*>(UINTPTR_MAX-3);
    if(invalid==Metadata::speculation)c.integer("n_spec",1);
    if(invalid==Metadata::compressed_shape)c.tensors[a::compressed_cache].shape[3]++;
    require(launch(&args,stream)<0,"invalid descriptor was accepted");check(hipStreamSynchronize(stream));require(scratch.read()==before,"descriptor failure changed device status");
    for(int i=0;i<a::tensors_count;++i)if(device[size_t(i)])require(device[size_t(i)]->read()==bytes(c.buffer(i).words),"descriptor failure changed an operand");
}
}
int main(int argc,char** argv) {
    if((argc!=3&&argc!=4)||std::strcmp(argv[1],"--run-gpu")) {
        std::fprintf(stderr,"usage: ds41-attention-paged-selftest --run-gpu /path/libds41.so [device]\n"
                            "Runs 352 tiny synthetic device cases plus 8 descriptor failures; never run automatically.\n");return 2;
    }
    void* plugin=nullptr;hipStream_t stream=nullptr;
    try {
        int device=0;if(argc==4){size_t n=0;device=std::stoi(argv[3],&n);require(n==std::strlen(argv[3])&&device>=0,"invalid device");}
        check(hipSetDevice(device));hipDeviceProp_t props{};check(hipGetDeviceProperties(&props,device));
        require(!std::strncmp(props.gcnArchName,"gfx1201",7)&&(!props.gcnArchName[7]||props.gcnArchName[7]==':'),"checker requires gfx1201");
        plugin=dlopen(argv[2],RTLD_NOW|RTLD_LOCAL);if(!plugin)throw std::runtime_error(dlerror());
        const auto abi=reinterpret_cast<uint32_t(*)()>(dlsym(plugin,"rad_plugin_abi_version"));require(abi&&abi()==RAD_ABI_VERSION,"plugin ABI mismatch");
        const auto launch=reinterpret_cast<RadLaunchFn>(dlsym(plugin,"ds41_sparse_attn_paged"));require(launch,"missing paged attention launch symbol");
        check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));size_t cases=0,metadata=0;
        auto one=[&](f::Fixture c,Scenario scenario=Scenario::normal){
            try{run(launch,c,scenario,stream);++cases;}catch(const std::exception& e){std::fprintf(stderr,"case=%zu dtype=%s pitch=%d ratio=%lld D=%lld rows=%lld scenario=%u: %s\n",cases+1,rad_dtype_name(c.dtype),c.pitch,(long long)c.ratio,(long long)c.dim,(long long)c.rows,unsigned(scenario),e.what());throw;}
        };
        for(uint32_t d:{RAD_F32,RAD_F16,RAD_BF16})for(int p=0;p<3;++p)for(int ratio:{0,1,2})for(int b:{2,4,8})one(f::Fixture(d,p,ratio,8,2,5,4,b)); //81
        for(int dim:{128,512})for(uint32_t d:{RAD_F32,RAD_F16,RAD_BF16})for(int p=0;p<3;++p)for(int ratio:{0,1,2})one(f::Fixture(d,p,ratio,dim)); //54
        for(int ctx:{0,1,13})for(uint32_t d:{RAD_F32,RAD_F16,RAD_BF16})for(int ratio:{0,1,2})one(f::Fixture(d,1,ratio,8,2,5,4,4,{1},{ctx})); //27
        for(const auto geometry:{std::array<int,2>{7,0},{1,1},{3,3}})for(uint32_t d:{RAD_F32,RAD_F16,RAD_BF16})for(int p=0;p<3;++p)one(f::Fixture(d,p,2,8,2,5,4,4,{geometry[0]},{geometry[1]})); //27
        for(auto scenario:{Scenario::duplicates,Scenario::disabled_sink,Scenario::dominant_sink,Scenario::unused_sequence,Scenario::all_padding})for(uint32_t d:{RAD_F32,RAD_F16,RAD_BF16})for(int p=0;p<3;++p)one(f::Fixture(d,p),scenario); //45
        for(auto scenario:{Scenario::future_id,Scenario::negative_id,Scenario::window_page,Scenario::compressed_page,Scenario::window_nan,Scenario::compressed_inf,Scenario::query_nan,Scenario::sink_inf,Scenario::context,Scenario::position,Scenario::cu,Scenario::used,Scenario::compressed_used})for(uint32_t d:{RAD_F32,RAD_F16,RAD_BF16})for(int p=0;p<3;++p)one(f::Fixture(d,p),scenario); //117
        one(f::Fixture(RAD_F32,1,1,4,1,4096,4096,4,{1},{4099})); //1
        for(auto invalid:{Metadata::alias,Metadata::stride,Metadata::dtype,Metadata::short_scratch,Metadata::scratch_alias,Metadata::scratch_wrap,Metadata::speculation,Metadata::compressed_shape}){reject_metadata(launch,invalid,stream);++metadata;}
        require(cases==expected_cases&&metadata==expected_metadata,"case count drifted");check(hipStreamSynchronize(stream));check(hipStreamDestroy(stream));stream=nullptr;dlclose(plugin);plugin=nullptr;
        std::printf("PASS: %zu tiny direct paged DS41 attention GPU cases plus %zu descriptor failures on %s; guarded operands, joint sink and validation flags\n",cases,metadata,props.gcnArchName);return 0;
    } catch(const std::exception& e) {
        if(stream){static_cast<void>(hipStreamSynchronize(stream));static_cast<void>(hipStreamDestroy(stream));}if(plugin)dlclose(plugin);
        std::fprintf(stderr,"ds41-attention-paged-selftest: %s\n",e.what());return 1;
    }
}
