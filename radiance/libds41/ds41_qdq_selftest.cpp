/* Manual native mathematical-QDQ gate. Independent Fixture.reference, exact BF16 bits;
 * 792 payload passes (78 reuse triplets +126 compressed-scale tie fixtures),29 metadata refusals.
 * No TileLang/model-quality claim, models, server operations or automatic GPU execution. */
#include <hip/hip_runtime.h>
#include "ds41_qdq_fixture.h"
#include "ds41_test_scratch.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <memory>
#include <span>
#include <string>
namespace q=rad::ds41::qdq;
namespace m=rad::ds41::qdq_math;
namespace f=q::testing;
namespace {
constexpr size_t allocation_limit=1u<<20,outer_guard=64,fixture_prefix=16;
constexpr size_t expected_passes=792,expected_good=390,expected_bad=402,expected_reuse=78,expected_metadata=29,expected_underflow=72,expected_scale_ties=126;
constexpr m::Profile profiles[]{m::Profile::window_fp8,m::Profile::compressed_fp4,m::Profile::index_fp4};
size_t live_bytes=0,peak_bytes=0,passes=0,good_passes=0,bad_passes=0,reuse_groups=0,metadata_passes=0,underflow_proofs=0,scale_tie_proofs=0;
void require(bool okay,const std::string& why){if(!okay)throw std::runtime_error(why);}
void check(hipError_t rc){if(rc!=hipSuccess)throw std::runtime_error(hipGetErrorString(rc));}
template<class T>std::vector<uint8_t> bytes(const T& source){
    std::vector<uint8_t> out(source.size()*sizeof(*source.data()));
    if(!out.empty())std::memcpy(out.data(),source.data(),out.size());
    return out;
}
struct Device {
    void* allocation=nullptr;hipStream_t stream=nullptr;size_t payload=0,total=0;
    Device(hipStream_t s,const std::vector<uint8_t>& initial):stream(s),payload(initial.size()),total(payload+2*outer_guard){
        require(payload>0&&total<=allocation_limit&&live_bytes<=allocation_limit-total,"aggregate guarded device allocation cap exceeded");
        live_bytes+=total;peak_bytes=std::max(peak_bytes,live_bytes);
        const auto allocated=hipMalloc(&allocation,total);
        if(allocated!=hipSuccess){live_bytes-=total;allocation=nullptr;check(allocated);}
        std::vector<uint8_t> host(total,0xC7);std::memcpy(host.data()+outer_guard,initial.data(),payload);
        const auto copied=hipMemcpy(allocation,host.data(),total,hipMemcpyHostToDevice);
        if(copied!=hipSuccess){static_cast<void>(hipFree(allocation));allocation=nullptr;live_bytes-=total;check(copied);}
    }
    Device(const Device&)=delete;Device& operator=(const Device&)=delete;
    ~Device(){
        if(!allocation)return;
        // Do not free queued storage after a failed drain; process exit owns it. This tiny
        // checker does not attempt recovery from a device fault or claim driver robustness.
        if(hipStreamSynchronize(stream)!=hipSuccess)return;
        if(hipFree(allocation)==hipSuccess)live_bytes-=total;
    }
    void* data()const{return static_cast<uint8_t*>(allocation)+outer_guard;}
    void* tensor_data()const{return static_cast<uint8_t*>(data())+fixture_prefix;}
    void reset(const std::vector<uint8_t>& source){
        require(source.size()==payload,"reuse changed a guarded allocation extent");
        check(hipMemcpy(data(),source.data(),payload,hipMemcpyHostToDevice));
    }
    std::vector<uint8_t> read()const{
        std::vector<uint8_t> host(total);check(hipMemcpy(host.data(),allocation,total,hipMemcpyDeviceToHost));
        for(size_t i=0;i<outer_guard;++i)require(host[i]==0xC7&&host[outer_guard+payload+i]==0xC7,"QDQ allocation outer red zone changed");
        return {host.begin()+outer_guard,host.begin()+outer_guard+payload};
    }
};
enum class Scenario {normal,all_zero_nan,ignored_nan,all_active,positive_inf,negative_inf,nan,overflow,bad_mask};
const char* scenario_name(Scenario s){
    switch(s){case Scenario::normal:return "normal";case Scenario::all_zero_nan:return "all_zero_nan";
    case Scenario::ignored_nan:return "ignored_nan";case Scenario::all_active:return "all_active";
    case Scenario::positive_inf:return "positive_inf";case Scenario::negative_inf:return "negative_inf";
    case Scenario::nan:return "nan";case Scenario::overflow:return "overflow";case Scenario::bad_mask:return "bad_mask";}
    return "unknown";
}
void arrange(f::Fixture& fixture,Scenario scenario){
    if(scenario==Scenario::all_zero_nan){
        require(fixture.masked,"all-zero scenario requires mask");
        for(int64_t row=0;row<fixture.rows;++row){fixture.buffers[q::emit].id(row,0);for(int64_t d=0;d<fixture.dim;++d)fixture.buffers[q::input].raw(row*fixture.dim+d,uint16_t(d%2?0xFFC1:0x7FC1));}
    }
    if(scenario==Scenario::ignored_nan){
        require(fixture.masked,"ignored-NaN scenario requires mask");
        for(int64_t row=0;row<fixture.rows;++row)if(!fixture.buffers[q::emit].id(row))
            for(int64_t d=0;d<fixture.dim;++d)fixture.buffers[q::input].raw(row*fixture.dim+d,uint16_t(d%2?0x7F80:0x7FC1));
    }
    if(scenario==Scenario::all_active){require(fixture.masked,"all-active scenario requires mask");for(int64_t row=0;row<fixture.rows;++row)fixture.buffers[q::emit].id(row,1);}
    const int64_t last=fixture.rows*fixture.dim-1;
    if(scenario==Scenario::positive_inf)fixture.buffers[q::input].raw(last,0x7F80);
    if(scenario==Scenario::negative_inf)fixture.buffers[q::input].raw(last,0xFF80);
    if(scenario==Scenario::nan)fixture.buffers[q::input].raw(last,0x7FC1);
    if(scenario==Scenario::overflow)fixture.buffers[q::input].raw(last,0x7F7F);
    if(scenario==Scenario::bad_mask){require(fixture.masked,"bad-mask scenario requires mask");fixture.buffers[q::emit].id(fixture.rows-1,2);}
}
uint16_t raw_output(const std::vector<uint8_t>& data,const f::Buffer& shape,int64_t flat){
    const size_t offset=fixture_prefix+size_t(shape.at(flat))*2;require(offset+2<=data.size(),"raw output inspection exceeds fixture");
    uint16_t code=0;std::memcpy(&code,data.data()+offset,2);return code;
}
struct Runner {
    hipStream_t stream;f::Fixture base;std::array<std::unique_ptr<Device>,q::operand_count> device;
    std::vector<uint8_t> scratch_before;std::unique_ptr<Device> scratch;
    Runner(hipStream_t s,f::Fixture fixture):stream(s),base(std::move(fixture)){
        base.refresh();for(int i=0;i<q::operand_count;++i)if(base.buffers[size_t(i)].t.rank)device[size_t(i)]=std::make_unique<Device>(stream,bytes(base.buffers[size_t(i)].words));
        const std::array<int32_t,4> prefix{911,912,913,914};scratch_before=rad::ds41::testing::make_gpu_scratch<q::Plan>(bytes(prefix));
        scratch=std::make_unique<Device>(stream,scratch_before);
        require(ds41_qdq_scratch(nullptr)==rad::ds41::gpu_scratch_bytes<q::Plan>(),"QDQ callback reserved the wrong Plan size");
    }
    RadArgs bind(f::Fixture& fixture){
        auto args=fixture.args();for(int i=0;i<q::operand_count;++i)if(device[size_t(i)])fixture.tensors[size_t(i)].data=device[size_t(i)]->tensor_data();
        args.scratch=scratch->data();args.scratch_bytes=rad::ds41::gpu_scratch_bytes<q::Plan>();return args;
    }
    void reset(const f::Fixture& fixture){
        check(hipStreamSynchronize(stream));for(int i=0;i<q::operand_count;++i)if(device[size_t(i)])device[size_t(i)]->reset(bytes(fixture.buffers[size_t(i)].words));
        scratch->reset(scratch_before);
    }
    void payload(Scenario scenario){
        auto fixture=base;arrange(fixture,scenario);std::array<f::Buffer,q::operand_count> reference;
        const int expected=fixture.reference(&reference);const bool should_fail=scenario==Scenario::positive_inf||scenario==Scenario::negative_inf||scenario==Scenario::nan||scenario==Scenario::overflow||scenario==Scenario::bad_mask;
        require(should_fail?expected<0:expected==RAD_OK,"independent reference disagrees with scenario validity");
        reset(fixture);auto args=bind(fixture);const int submitted=ds41_qdq_candidate(&args,stream);
        require(submitted==RAD_OK,"valid QDQ descriptors failed to enqueue");check(hipStreamSynchronize(stream));
        const auto after=scratch->read();int32_t flag=0;std::memcpy(&flag,after.data(),4);
        require(flag==expected,"QDQ device scalar status differs from independent reference");
        require(rad::ds41::testing::gpu_scratch_guards<q::Plan>(after,scratch_before),"QDQ scalar guard/reserved prefix/Plan tail changed");
        std::vector<uint8_t> actual_output;
        for(int i=0;i<q::operand_count;++i)if(device[size_t(i)]){
            auto actual=device[size_t(i)]->read();const auto wanted=bytes(expected==RAD_OK&&i==q::output?reference[size_t(i)].words:fixture.buffers[size_t(i)].words);
            require(actual==wanted,"QDQ exact BF16 output, readonly input/mask or pitched/canary bytes differ");
            if(i==q::output)actual_output=std::move(actual);
        }
        if(expected==RAD_OK&&fixture.profile==m::Profile::index_fp4&&scenario!=Scenario::all_zero_nan){
            require(fixture.buffers[q::input].raw(0)==0x0040&&fixture.buffers[q::input].raw(1)==0x8040,"default tiny group was not retained");
            require(reference[q::output].raw(0)==0x0040&&reference[q::output].raw(1)==0x8040,"independent reference lost gradual-underflow group");
            require(raw_output(actual_output,fixture.buffers[q::output],0)==0x0040&&raw_output(actual_output,fixture.buffers[q::output],1)==0x8040,"GPU lost positive/negative2^-127 gradual FP32 underflow");++underflow_proofs;
        }
        ++passes;if(expected==RAD_OK)++good_passes;else ++bad_passes;
    }
    void refuse(int fault){
        auto fixture=base;reset(fixture);auto args=bind(fixture);
        if(fault==0)fixture.tensors[q::input].dtype=RAD_F32;
        if(fault==1)fixture.tensors[q::output].dtype=RAD_F16;
        if(fault==2)fixture.tensors[q::emit].dtype=RAD_I64;
        if(fault==3)fixture.tensors[q::output].stride[0]=0;
        if(fault==4)fixture.tensors[q::input].stride[1]=0;
        if(fault==5)fixture.integer("n",15);
        if(fault==6)fixture.integer("masked",2);
        if(fault==7)fixture.tensors[q::output].data=fixture.tensors[q::input].data;
        if(fault==8)fixture.tensors[q::output].data=fixture.tensors[q::emit].data;
        if(fault==9)args.scratch=fixture.tensors[q::output].data;
        if(fault==10)args.scratch_bytes=3;
        if(fault==11)fixture.tensors[q::input].data=reinterpret_cast<void*>(UINTPTR_MAX-1);
        if(fault==12)args.n_t=2;
        if(fault==13)fixture.integer("M",q::max_rows+1);
        if(fault==14)fixture.tensors[q::input].shape[0]=fixture.rows+2;
        if(fault==15)fixture.integer("masked",0); // fixed optional slot must now be absent
        if(fault==16)for(auto& param:fixture.params)if(!std::strcmp(param.key,"profile"))param.sval="unknown";
        if(fault==17)--args.scratch_bytes;
        if(fault==18)fixture.tensors[q::input].data=static_cast<uint8_t*>(scratch->data())+rad::ds41::device_plan_offset;
        if(fault==19)args.scratch=static_cast<uint8_t*>(scratch->data())+4;
        if(fault==20)args.scratch=reinterpret_cast<void*>(UINTPTR_MAX-3);
        if(fault==21)fixture.tensors[q::output].data=nullptr;
        if(fault==22)fixture.tensors[q::emit].data=nullptr;
        if(fault==23)fixture.tensors[q::input].data=nullptr;
        if(fault==24)fixture.integer("M",0);
        if(fault==25)fixture.integer("n",1025);
        if(fault==26)fixture.tensors[q::emit].rank=2;
        if(fault==27)fixture.tensors[q::output].data=reinterpret_cast<void*>(UINTPTR_MAX-1);
        if(fault==28)args.scratch_bytes=INT64_MIN;
        require(ds41_qdq_candidate(&args,stream)<0,"invalid QDQ metadata/alias/Plan reservation was accepted");check(hipStreamSynchronize(stream));
        require(scratch->read()==scratch_before,"metadata refusal changed ANY scalar/Plan/reserved scratch byte");
        for(int i=0;i<q::operand_count;++i)if(device[size_t(i)])require(device[size_t(i)]->read()==bytes(fixture.buffers[size_t(i)].words),"metadata refusal changed input/mask/output bytes");
        ++metadata_passes;
    }
};
void run_case(hipStream_t stream,m::Profile profile,int dim,int pitch,bool masked,const std::vector<Scenario>& sequence,bool reuse=false){
    try{
        Runner runner(stream,f::Fixture(profile,dim,pitch,masked));
        for(auto scenario:sequence){try{runner.payload(scenario);}catch(const std::exception& error){throw std::runtime_error(std::string("scenario=")+scenario_name(scenario)+": "+error.what());}}
        if(reuse){require(sequence.size()==3,"bad reuse sequence");++reuse_groups;}
    }catch(const std::exception& error){throw std::runtime_error(std::string("profile=")+f::name(profile)+" D="+std::to_string(dim)+" pitch="+std::to_string(pitch)+" masked="+std::to_string(masked)+" after_pass="+std::to_string(passes)+": "+error.what());}
}
/* E4M3 scales are short dyadic values:6*s and every midpoint product below are
 * exact BF16. Each of the THREE16D rows retains6*s as an absmax, so the actual
 * device quotient must exercise this scale rather than a smaller subgroup max.
 * Adjacent BF16 inputs on both signs bracket every E2M1 midpoint. */
void compressed_scale_ties(hipStream_t stream,uint8_t scale_code){
    namespace reference=rad::deepseek41::qdq;
    try{
        float scale=0;require(reference::e4m3_value(scale_code,&scale)==RAD_OK&&scale>0,"invalid positive finite E4M3 scale");
        uint16_t amax=0;require(reference::bf16_rne(6.f*scale,&amax)==RAD_OK,"6*s BF16 conversion failed");
        require(reference::bf16_value(amax)==6.f*scale,"6*s must be exactly representable in BF16");
        std::vector<uint16_t> values;values.reserve(45);
        for(float cut:{.25f,.75f,1.25f,1.75f,2.5f,3.5f,5.f}){
            uint16_t middle=0;require(reference::bf16_rne(scale*cut,&middle)==RAD_OK,"midpoint BF16 conversion failed");
            require(reference::bf16_value(middle)==scale*cut,"s*midpoint must be exact BF16");
            for(int delta:{-1,0,1}){
                const auto positive=uint16_t(int(middle)+delta);
                require(positive>0&&positive<amax,"midpoint neighbor changed the intended group maximum");
                values.push_back(positive);values.push_back(uint16_t(positive|0x8000u));
            }
        }
        require(values.size()==42,"compressed midpoint fixture count drifted");
        values.push_back(uint16_t(amax|0x8000u));values.push_back(0);values.push_back(0x8000u);
        f::Fixture fixture(m::Profile::compressed_fp4,16,int(scale_code%3),false,3);
        for(int row=0;row<3;++row){
            for(int col=0;col<15;++col)fixture.buffers[q::input].raw(row*16+col,values[size_t(row*15+col)]);
            fixture.buffers[q::input].raw(row*16+15,amax);
            std::array<uint16_t,16> source{};for(int col=0;col<16;++col)source[size_t(col)]=fixture.buffers[q::input].raw(row*16+col);
            reference::GroupResult result;
            require(reference::group(reference::Profile::compressed_fp4,source,&result)==RAD_OK&&result.scale_code==scale_code,
                    "independent row did not retain the intended E4M3 scale");
        }
        Runner runner(stream,std::move(fixture));runner.payload(Scenario::normal);++scale_tie_proofs;
    }catch(const std::exception& error){throw std::runtime_error("compressed E4M3 scale_code="+std::to_string(unsigned(scale_code))+": "+error.what());}
}
void usage(){std::puts("ds41-qdq-selftest --run-gpu [device]\nRequires exactly one visible gfx1201 (optional device must be0).792 tiny payload passes (including126 quotient-tie scales) +29 metadata refusals; never automatic. Mathematical BF16 QDQ only, no TileLang/model-quality claim.");}
}
int main(int argc,char** argv){
    if((argc!=2&&argc!=3)||std::strcmp(argv[1],"--run-gpu")){usage();return 2;}
    // Main gate precedes explicit query/allocation/launch. Linked HIP registration may run
    // before main; only parent should invoke this manual checker in an approved idle window.
    hipStream_t stream=nullptr;
    try{
        int device=0;if(argc==3){size_t n=0;device=std::stoi(argv[2],&n);require(n==std::strlen(argv[2])&&device==0,"with one visible card device must be0");}
        require(rad::deepseek41::qdq::supported_f32_environment(),"independent CPU oracle requires round-nearest and gradual FP32 underflow");
        int count=0;check(hipGetDeviceCount(&count));require(count==1,"expose exactly one GPU via ROCR_VISIBLE_DEVICES");check(hipSetDevice(device));
        hipDeviceProp_t props{};check(hipGetDeviceProperties(&props,device));require(!std::strncmp(props.gcnArchName,"gfx1201",7)&&(!props.gcnArchName[7]||props.gcnArchName[7]==':'),"checker requires exactly one visible gfx1201");
        check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
        const std::vector<Scenario> reuse{Scenario::normal,Scenario::nan,Scenario::normal};
        for(auto profile:profiles)for(int dim:{32,128,512,1024})for(int pitch:{0,1,2})for(bool masked:{false,true})run_case(stream,profile,dim,pitch,masked,reuse,true);
        for(int pitch:{0,1,2})for(bool masked:{false,true})run_case(stream,m::Profile::compressed_fp4,16,pitch,masked,reuse,true);
        for(auto profile:profiles)for(int dim:{32,128,512,1024})for(int pitch:{0,1,2})for(auto scenario:{Scenario::all_zero_nan,Scenario::ignored_nan,Scenario::all_active})run_case(stream,profile,dim,pitch,true,{scenario});
        for(auto profile:profiles)for(int dim:{32,128,512,1024})for(int pitch:{0,1,2}){
            for(bool masked:{false,true})for(auto scenario:{Scenario::positive_inf,Scenario::negative_inf,Scenario::nan,Scenario::overflow})run_case(stream,profile,dim,pitch,masked,{scenario});
            run_case(stream,profile,dim,pitch,true,{Scenario::bad_mask});
        }
        for(unsigned code=1;code<=126;++code)compressed_scale_ties(stream,uint8_t(code));
        {Runner metadata(stream,f::Fixture(m::Profile::index_fp4,128,1,true));for(int fault=0;fault<int(expected_metadata);++fault)metadata.refuse(fault);}
        check(hipStreamSynchronize(stream));require(live_bytes==0,"guarded device allocations were not released");
        require(passes==expected_passes&&good_passes==expected_good&&bad_passes==expected_bad&&reuse_groups==expected_reuse&&metadata_passes==expected_metadata&&underflow_proofs==expected_underflow&&scale_tie_proofs==expected_scale_ties,"QDQ checker coverage count drifted");
        check(hipStreamDestroy(stream));stream=nullptr;
        std::printf("PASS: %zu exact-bit BF16 QDQ payload passes (%zu good/%zu rejected),%zu same-allocation good-bad-good groups,%zu metadata refusals,%zu signed2^-127 underflow proofs,%zu compressed E4M3 quotient-tie scales on %s; peak guarded device allocation=%zu B (cap=%zu B).\n",passes,good_passes,bad_passes,reuse_groups,metadata_passes,underflow_proofs,scale_tie_proofs,props.gcnArchName,peak_bytes,allocation_limit);
        std::puts("Scope: independent CPU/Torch-backed mathematical profiles, raw input/mask/padding guards and full Plan status/tail checks. No official TileLang/model-quality, service, weights or throughput claim. Driver/module/runtime metadata and host reference vectors are separate from device-allocation receipt.");return 0;
    }catch(const std::exception& error){if(stream){static_cast<void>(hipStreamSynchronize(stream));static_cast<void>(hipStreamDestroy(stream));}std::fprintf(stderr,"FAIL: ds41-qdq-selftest: %s\n",error.what());return 1;}
}
