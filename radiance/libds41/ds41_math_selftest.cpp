/* Explicitly gated standalone checker; never run by CTest or by a build command. Compares the
 * GPU's four existing ds41_* math operations with CPU emulation already checked against the
 * independent pinned semantic oracle. Synthetic tiny tensors only: no weights/server access. */
#include <hip/hip_runtime.h>
#include "ds41_test_scratch.h"
#include "ds41_math_emulate.h"
#include <dlfcn.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace m=rad::ds41::math;
namespace {
constexpr size_t derived_fault_cycles=11,expected_cases=141+3*derived_fault_cycles;
void check(hipError_t status) {if(status!=hipSuccess)throw std::runtime_error(hipGetErrorString(status));}
void require(bool value,const char* message) {if(!value)throw std::runtime_error(message);}
struct Device {
    void* base=nullptr;size_t size=0;static constexpr size_t guard=64;
    explicit Device(const std::vector<uint8_t>& bytes):size(bytes.size()) {
        check(hipMalloc(&base,size+2*guard));
        auto status=hipMemset(base,0xA7,size+2*guard);
        if(status==hipSuccess)status=hipMemcpy(data(),bytes.data(),size,hipMemcpyHostToDevice);
        if(status!=hipSuccess) {static_cast<void>(hipFree(base));base=nullptr;check(status);}
    }
    Device(const Device&)=delete;Device& operator=(const Device&)=delete;
    ~Device() {if(base)static_cast<void>(hipFree(base));}
    void* data() const {return static_cast<uint8_t*>(base)+guard;}
    std::vector<uint8_t> read() const {
        std::vector<uint8_t> all(size+2*guard);
        check(hipMemcpy(all.data(),base,all.size(),hipMemcpyDeviceToHost));
        require(std::all_of(all.begin(),all.begin()+guard,[](uint8_t b){return b==0xA7;}),"GPU prefix red zone changed");
        require(std::all_of(all.end()-guard,all.end(),[](uint8_t b){return b==0xA7;}),"GPU suffix red zone changed");
        return {all.begin()+guard,all.end()-guard};
    }
};
template<class T> std::vector<uint8_t> bytes(const std::vector<T>& values) {
    std::vector<uint8_t> result(values.size()*sizeof(T));std::memcpy(result.data(),values.data(),result.size());return result;
}
struct Buffer {
    RadTensor t{};std::vector<uint32_t> words;
    Buffer(uint32_t dtype,std::initializer_list<int64_t> dims,int pitch) {
        t.dtype=dtype;t.rank=dims.size();unsigned axis=0;for(auto d:dims)t.shape[axis++]=d;
        int64_t stride=pitch==2?2:1;
        for(int d=int(t.rank)-1;d>=0;--d) {t.stride[d]=stride;stride=stride*t.shape[d]+(pitch?3:0);}
        words.resize(size_t(stride+8),0xD5D5D5D5u);t.data=words.data();
    }
    int64_t count() const {int64_t n=1;for(unsigned d=0;d<t.rank;++d)n*=t.shape[d];return n;}
    int64_t offset(int64_t flat) const {
        int64_t at=0;for(int d=int(t.rank)-1;d>=0;--d) {at+=(flat%t.shape[d])*t.stride[d];flat/=t.shape[d];}return at;
    }
    float get(int64_t i) const {return rad_load_f32(words.data(),t.dtype,offset(i));}
    void set(int64_t i,float value) {rad_store_f32(words.data(),t.dtype,offset(i),value);}
    void set_id(int64_t i,int32_t value) {reinterpret_cast<int32_t*>(words.data())[offset(i)]=value;}
};
enum class Scenario {normal,hc_zero,router_tie,router_bias,router_top1,router_no_norm,router_underflow,
    sparse_disabled,sparse_empty,sparse_sink,sparse_repeat,sparse_unused_nan,input_nan,
    fn_nan,scale_inf,base_nan,previous_nan,correction_nan,sink_inf,sink_nan,id_negative,id_past,id_max,kv_nan,kv_inf,
    prepare_derived,collapse_derived,collapse_cast,router_quotient,router_scaled,sparse_numerator};
struct Fixture {
    m::Op op;Scenario scenario;std::vector<Buffer> buffers;std::vector<RadParam> params;
    std::vector<RadTensor> tensors;std::vector<int32_t> status{911,912,913,914};
    Fixture(m::Op kind,uint32_t dtype,int pitch,Scenario test=Scenario::normal):op(kind),scenario(test) {
        const auto* dt=rad_dtype_name(dtype);
        if(op==m::Op::hc_prepare||op==m::Op::hc_collapse) {
            const int hc=pitch==0?2:pitch==1?4:8,n=5+pitch;
            params={RAD_INT("M",4),RAD_INT("n",n),RAD_INT("hc",hc),RAD_STR("dtype",dt)};
            add(dtype,{2,hc*n},pitch);
            if(op==m::Op::hc_prepare) {
                params.push_back(RAD_INT("iterations",pitch==0?1:pitch==1?20:256));
                params.push_back(RAD_F64("eps",1e-6));params.push_back(RAD_F64("hc_eps",test==Scenario::hc_zero?0:1e-6));
                const int mixes=hc*(hc+2);add(RAD_F32,{mixes,hc*n},pitch);add(RAD_F32,{3},pitch);add(RAD_F32,{mixes},pitch);
                add(RAD_F32,{2,hc},pitch);add(RAD_F32,{2,hc},pitch);add(RAD_F32,{2,hc*hc},pitch);
            } else {add(RAD_F32,{2,hc},pitch);add(dtype,{2,n},pitch);}
        } else if(op==m::Op::router_topk) {
            const int k=test==Scenario::router_top1||test==Scenario::router_scaled?1:6;
            params={RAD_INT("M",4),RAD_INT("experts",12),RAD_INT("topk",k),RAD_F64("temperature",.75),
                    RAD_F64("route_scale",1.5),RAD_INT("normalize",test==Scenario::router_no_norm?0:1)};
            add(RAD_F32,{3,12},pitch);add(RAD_F32,{12},pitch);add(RAD_I32,{3,k},pitch);add(RAD_F32,{3,k},pitch);
        } else {
            params={RAD_INT("M",4),RAD_INT("heads",3),RAD_INT("head_dim",8),RAD_INT("kv_rows",7),
                    RAD_INT("topk",4),RAD_F64("scale",.35),RAD_STR("dtype",dt)};
            add(dtype,{2,3,8},pitch);add(dtype,{7,8},pitch);add(RAD_F32,{3},pitch);add(RAD_I32,{2,4},pitch);add(dtype,{2,3,8},pitch);
        }
        std::mt19937 rng(411017+unsigned(op)*17+unsigned(test)*3+unsigned(pitch));
        for(int b=0;b<first_output();++b)for(int64_t i=0;i<buffers[size_t(b)].count();++i) {
            if(buffers[size_t(b)].t.dtype==RAD_I32)buffers[size_t(b)].set_id(i,int32_t(i%5)-1);
            else buffers[size_t(b)].set(i,float(int(rng()%1001)-500)/512.f);
        }
        if(op==m::Op::hc_prepare) {
            buffers[2].set(0,.6f);buffers[2].set(1,.9f);buffers[2].set(2,1.2f);
            if(test==Scenario::hc_zero)for(int64_t i=0;i<2*integer("hc");++i)buffers[3].set(i,-1000.f);
        }
        if(op==m::Op::router_topk&&test!=Scenario::normal&&test!=Scenario::input_nan&&test!=Scenario::correction_nan) {
            for(int64_t i=0;i<buffers[0].count();++i)buffers[0].set(i,test==Scenario::router_underflow?-1000.f:0.f);
            for(int i=0;i<12;++i)buffers[1].set(i,0.f);
            if(test==Scenario::router_bias)buffers[1].set(11,100.f);
        }
        if(op==m::Op::sparse_attn) {
            buffers[2].set(2,-INFINITY);
            if(test==Scenario::sparse_disabled||test==Scenario::sparse_empty) {
                for(int64_t i=0;i<buffers[3].count();++i)buffers[3].set_id(i,-1);
                for(int i=0;i<3;++i)buffers[2].set(i,test==Scenario::sparse_disabled?-INFINITY:0.f);
            }
            if(test==Scenario::sparse_sink) {
                for(int64_t i=0;i<buffers[0].count();++i)buffers[0].set(i,0.f);
                for(int64_t i=0;i<buffers[1].count();++i)buffers[1].set(i,2.f);
                for(int i=0;i<3;++i)buffers[2].set(i,std::log(2.f));
                for(int64_t i=0;i<buffers[3].count();++i)buffers[3].set_id(i,i%4<2?0:-1);
            }
            if(test==Scenario::sparse_repeat)for(int64_t i=0;i<buffers[3].count();++i)buffers[3].set_id(i,i%4==3?-1:2);
            if(test==Scenario::sparse_unused_nan)buffers[1].set(6*8,NAN);
        }
        if(test==Scenario::input_nan)buffers[0].set(buffers[0].count()-1,NAN);
        if(test==Scenario::fn_nan)buffers[1].set(buffers[1].count()-1,NAN);
        if(test==Scenario::scale_inf)buffers[2].set(2,INFINITY);
        if(test==Scenario::base_nan)buffers[3].set(buffers[3].count()-1,NAN);
        if(test==Scenario::previous_nan||test==Scenario::correction_nan)buffers[1].set(buffers[1].count()-1,NAN);
        if(test==Scenario::sink_inf||test==Scenario::sink_nan)buffers[2].set(2,test==Scenario::sink_inf?INFINITY:NAN);
        if(test==Scenario::id_negative||test==Scenario::id_past||test==Scenario::id_max)
            buffers[3].set_id(buffers[3].count()-1,test==Scenario::id_negative?-2:test==Scenario::id_past?7:INT32_MAX);
        if(test==Scenario::kv_nan||test==Scenario::kv_inf) {
            buffers[3].set_id(buffers[3].count()-1,6);buffers[1].set(6*8+7,test==Scenario::kv_nan?NAN:INFINITY);
        }
        if(test==Scenario::prepare_derived){
            // All source values remain finite; the first row is safe and the final row
            // overflows an RMS square/dot (F32/BF16), or only the dot (F16 input).
            const int64_t width=integer("hc")*integer("n");
            for(int64_t i=0;i<buffers[0].count();++i)buffers[0].set(i,0.f);
            for(int64_t i=0;i<buffers[1].count();++i)buffers[1].set(i,std::ldexp(1.f,dtype==RAD_F16?127:100));
            for(int64_t i=width;i<2*width;++i)buffers[0].set(i,dtype==RAD_F16?2.f:std::ldexp(1.f,100));
        }
        if(test==Scenario::collapse_derived||test==Scenario::collapse_cast){
            const int64_t hc=integer("hc"),n=integer("n");
            for(int64_t i=0;i<buffers[0].count();++i)buffers[0].set(i,.25f);
            for(int64_t i=0;i<buffers[1].count();++i)buffers[1].set(i,0.f);
            buffers[1].set(2*hc-1,test==Scenario::collapse_cast&&dtype==RAD_BF16?257.f/256.f:2.f);
            const float large=test==Scenario::collapse_cast?(dtype==RAD_F16?65504.f:rad_bf16_to_f32(0x7F7F)):std::ldexp(1.f,127);
            buffers[0].set((2*hc-1)*n+n-1,large);
        }
        if(test==Scenario::router_quotient||test==Scenario::router_scaled){
            for(int64_t i=0;i<buffers[0].count();++i)buffers[0].set(i,0.f);
            for(int64_t i=0;i<buffers[1].count();++i)buffers[1].set(i,0.f);
            buffers[0].set(buffers[0].count()-1,test==Scenario::router_quotient?4.f:std::ldexp(1.f,126));
            for(auto& param:params){
                if(!std::strcmp(param.key,"temperature"))param.dval=test==Scenario::router_quotient?std::ldexp(1.,-126):1.;
                if(!std::strcmp(param.key,"route_scale")&&test==Scenario::router_scaled)param.dval=std::ldexp(1.,66);
            }
        }
        if(test==Scenario::sparse_numerator){
            for(int64_t i=0;i<buffers[0].count();++i)buffers[0].set(i,0.f);
            for(int64_t i=0;i<buffers[1].count();++i)buffers[1].set(i,.125f);
            for(int head=0;head<3;++head)buffers[2].set(head,head==2?-INFINITY:std::numeric_limits<float>::max());
            for(int64_t i=0;i<buffers[3].count();++i)buffers[3].set_id(i,i<4?0:-1);
            buffers[3].set_id(4,5);buffers[3].set_id(5,6);
            for(int64_t i=5*8;i<7*8;++i)buffers[1].set(i,std::ldexp(1.f,127));
        }
        refresh();
    }
    void add(uint32_t dtype,std::initializer_list<int64_t> dims,int pitch) {buffers.emplace_back(dtype,dims,pitch);}
    int first_output() const {return op==m::Op::hc_prepare?4:op==m::Op::hc_collapse?2:op==m::Op::router_topk?2:4;}
    int64_t integer(const char* key) const {return rad_param_geti(params.data(),params.size(),key,0);}
    void refresh() {tensors.clear();for(auto& b:buffers) {b.t.data=b.words.data();tensors.push_back(b.t);}}
    RadArgs args() {
        refresh();RadArgs a{};a.t=tensors.data();a.n_t=int(tensors.size());a.p=params.data();a.n_p=int(params.size());
        a.scratch=status.data();a.scratch_bytes=m::scratch_bytes;return a;
    }
    void promote() {
        for(auto& b:buffers)if(b.t.dtype==RAD_F16||b.t.dtype==RAD_BF16) {
            std::vector<float> values;for(int64_t i=0;i<b.count();++i)values.push_back(b.get(i));
            b.t.dtype=RAD_F32;for(int64_t i=0;i<b.count();++i)b.set(i,values[size_t(i)]);
        }
        for(auto& p:params)if(!std::strcmp(p.key,"dtype"))p.sval="f32";
        refresh();
    }
};
uint16_t word16(const std::vector<uint8_t>& v,size_t byte) {uint16_t h;std::memcpy(&h,v.data()+byte,2);return h;}
float scalar(const std::vector<uint8_t>& v,size_t element,uint32_t dtype) {
    if(dtype==RAD_F32) {float f;std::memcpy(&f,v.data()+4*element,4);return f;}
    const auto h=word16(v,2*element);return dtype==RAD_F16?rad_f16_to_f32(h):rad_bf16_to_f32(h);
}
int ordered(uint16_t h) {return h&0x8000u?32768-int(h&0x7FFFu):32768+int(h);}
void compare_output(const Fixture& input,const Fixture& expected,const Fixture& fp32,size_t operand,
                    const std::vector<uint8_t>& actual) {
    const auto& b=input.buffers[operand];const auto& want=expected.buffers[operand];const auto& raw=fp32.buffers[operand];
    const auto initial=bytes(b.words);std::vector<bool> touched(actual.size(),false);
    const int width=rad_dtype_bits(b.t.dtype)/8;
    for(int64_t i=0;i<b.count();++i) {
        const size_t at=size_t(b.offset(i));
        for(int byte=0;byte<width;++byte)touched[at*size_t(width)+size_t(byte)]=true;
        if(b.t.dtype==RAD_I32) {
            int32_t got;std::memcpy(&got,actual.data()+at*4,4);
            const auto* e=reinterpret_cast<const int32_t*>(want.words.data());require(got==e[at],"GPU expert ID/tie-order mismatch");
            continue;
        }
        const float got=scalar(actual,at,b.t.dtype),reference=want.get(i),unrounded=raw.get(i);
        require(std::isfinite(got)&&std::isfinite(reference)&&std::isfinite(unrounded),"nonfinite GPU/CPU numerical result");
        const float tolerance=4e-6f+4e-5f*std::fabs(unrounded);
        if(b.t.dtype==RAD_F32)require(std::fabs(got-reference)<=tolerance,"GPU/CPU FP32 math mismatch");
        else {
            const uint16_t received=word16(actual,at*2);
            const uint16_t encoded=b.t.dtype==RAD_F16?rad_f32_to_f16(unrounded):rad_f32_to_bf16(unrounded);
            if(received!=encoded) {
                require(std::abs(ordered(received)-ordered(encoded))<=1,"GPU storage differs by more than one representable step");
                const float midpoint=.5f*(got+reference);
                require(std::fabs(unrounded-midpoint)<=tolerance,"GPU storage mismatch away from a rounding boundary");
            }
        }
        const bool zero_gate=input.scenario==Scenario::hc_zero&&(operand==4||operand==5);
        const bool zero_sink=input.scenario==Scenario::sparse_disabled||input.scenario==Scenario::sparse_empty;
        if(zero_gate||zero_sink)require(got==0.f&&!std::signbit(got),"positive-zero gate/padding convention changed");
        if(input.scenario==Scenario::sparse_sink)require(std::fabs(got-1.f)<=tolerance,"sink contributed a value or duplicate key lost weight");
    }
    for(size_t i=0;i<actual.size();++i)if(!touched[i])require(actual[i]==initial[i],"GPU output stride hole or tail canary changed");
}
void run(RadLaunchFn launch,Fixture input,hipStream_t stream) {
    Fixture expected=input;auto oracle_args=expected.args();const int oracle=m::emulate(input.op,&oracle_args);
    Fixture fp32=input;fp32.promote();auto fp_args=fp32.args();
    const int fp_status=m::emulate(input.op,&fp_args);require((fp_status==RAD_OK)==(oracle==RAD_OK),"CPU rounding shadow validation disagrees");
    std::vector<std::unique_ptr<Device>> device;
    for(const auto& b:input.buffers)device.push_back(std::make_unique<Device>(bytes(b.words)));
    const auto initial_flag=rad::ds41::testing::make_gpu_scratch<m::Plan>(bytes(input.status));Device scratch(initial_flag);
    auto args=input.args();
    for(size_t i=0;i<device.size();++i)input.tensors[i].data=device[i]->data();
    args.scratch=scratch.data();args.scratch_bytes=rad::ds41::gpu_scratch_bytes<m::Plan>();
    require(launch(&args,stream)==RAD_OK,"GPU enqueue refused valid metadata");
    check(hipStreamSynchronize(stream));
    const auto flag=scratch.read();int32_t device_status=0;std::memcpy(&device_status,flag.data()+m::status_offset,4);
    require(device_status==oracle,"GPU validation flag disagrees with CPU payload validation");
    require(rad::ds41::testing::gpu_scratch_guards<m::Plan>(flag,initial_flag),"GPU changed status sentinels or staged Plan tail guards");
    for(size_t i=0;i<device.size();++i) {
        const auto received=device[i]->read();
        if(int(i)<input.first_output()||oracle!=RAD_OK)require(received==bytes(input.buffers[i].words),"GPU input changed or invalid batch wrote output");
        else compare_output(input,expected,fp32,i,received);
    }
}
void derived_reset_cycle(RadLaunchFn launch,Fixture bad,hipStream_t stream){
    Fixture good=bad;
    for(int i=0;i<good.first_output();++i)for(int64_t j=0;j<good.buffers[size_t(i)].count();++j){
        if(good.buffers[size_t(i)].t.dtype==RAD_I32)good.buffers[size_t(i)].set_id(j,0);
        else good.buffers[size_t(i)].set(j,.125f);
    }
    // Input-only validation must accept the finite derived-fault witness. The checked
    // arithmetic, not an accidentally injected nonfinite payload, must reject it.
    auto validation=bad.args();m::Plan plan;require(m::validate(bad.op,&validation,&plan)==RAD_OK,"derived witness metadata invalid");
    for(int64_t i=0;i<m::preflight_items(plan);++i)require(m::check_item(plan,i)==RAD_OK,"derived witness has an input payload fault");
    auto oracle_args=bad.args();require(m::emulate(bad.op,&oracle_args)==RAD_E_INVAL,"derived witness must be refused by checked CPU arithmetic");
    std::vector<std::unique_ptr<Device>> device;
    for(const auto& b:good.buffers)device.push_back(std::make_unique<Device>(bytes(b.words)));
    const auto initial_flag=rad::ds41::testing::make_gpu_scratch<m::Plan>(bytes(good.status));Device scratch(initial_flag);
    std::vector<std::vector<uint8_t>> prior;
    for(const auto& d:device)prior.push_back(d->read());
    for(int pass=0;pass<3;++pass){
        Fixture input=pass==1?bad:good,expected=input,fp32=input;
        auto ea=expected.args();const int want=m::emulate(input.op,&ea);require(want==(pass==1?RAD_E_INVAL:RAD_OK),"derived cycle CPU status drifted");
        if(pass!=1){fp32.promote();auto fa=fp32.args();require(m::emulate(input.op,&fa)==RAD_OK,"derived-cycle F32 rounding shadow failed");}
        for(int i=0;i<input.first_output();++i){const auto data=bytes(input.buffers[size_t(i)].words);check(hipMemcpyAsync(device[size_t(i)]->data(),data.data(),data.size(),hipMemcpyHostToDevice,stream));check(hipStreamSynchronize(stream));}
        auto args=input.args();for(size_t i=0;i<device.size();++i)input.tensors[i].data=device[i]->data();
        args.scratch=scratch.data();args.scratch_bytes=rad::ds41::gpu_scratch_bytes<m::Plan>();
        require(launch(&args,stream)==RAD_OK,"derived cycle enqueue refused valid metadata");check(hipStreamSynchronize(stream));
        const auto flag=scratch.read();int32_t status=0;std::memcpy(&status,flag.data()+m::status_offset,4);
        require(status==want,"derived cycle status/reset failed");
        require(rad::ds41::testing::gpu_scratch_guards<m::Plan>(flag,initial_flag),"derived cycle scratch guard changed");
        for(size_t i=0;i<device.size();++i){const auto received=device[i]->read();
            if(int(i)<input.first_output())require(received==bytes(input.buffers[i].words),"derived cycle changed readonly input");
            else if(pass==1)require(received==prior[i],"late derived fault changed an output");
            else compare_output(input,expected,fp32,i,received);
            prior[i]=received;
        }
    }
}
}
int main(int argc,char** argv) {
    // This gate precedes every HIP call, dlopen, tensor allocation and test execution.
    if((argc!=3&&argc!=4)||std::strcmp(argv[1],"--run-gpu")) {
        std::fprintf(stderr,"usage: ds41-math-selftest --run-gpu /path/libds41.so [device]\n"
                            "Runs exactly 174 tiny synthetic GPU cases, including11 derived-fault reset cycles; use only in an approved idle window.\n"
                            "Never run automatically by a build or CTest.\n");return 2;
    }
    void* plugin=nullptr;hipStream_t stream=nullptr;
    try {
        int device=0;
        if(argc==4) {size_t used=0;device=std::stoi(argv[3],&used);require(used==std::strlen(argv[3])&&device>=0,"invalid device index");}
        check(hipSetDevice(device));hipDeviceProp_t properties{};check(hipGetDeviceProperties(&properties,device));
        require(!std::strncmp(properties.gcnArchName,"gfx1201",7)&&(!properties.gcnArchName[7]||properties.gcnArchName[7]==':'),"checker requires gfx1201");
        plugin=dlopen(argv[2],RTLD_NOW|RTLD_LOCAL);if(!plugin)throw std::runtime_error(dlerror());
        auto abi=reinterpret_cast<uint32_t(*)()>(dlsym(plugin,"rad_plugin_abi_version"));
        require(abi&&abi()==RAD_ABI_VERSION,"experimental plugin ABI mismatch");
        const char* names[]{"ds41_hc_prepare","ds41_hc_collapse","ds41_router_topk","ds41_sparse_attn"};
        std::array<RadLaunchFn,4> launch{};
        for(size_t i=0;i<launch.size();++i) {launch[i]=reinterpret_cast<RadLaunchFn>(dlsym(plugin,names[i]));require(launch[i],"missing GPU math launch symbol");}
        check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));size_t cases=0;
        auto one=[&](m::Op op,uint32_t dtype,int pitch,Scenario scenario=Scenario::normal) {
            try {run(launch[size_t(op)],Fixture(op,dtype,pitch,scenario),stream);++cases;}
            catch(const std::exception& e) {
                std::fprintf(stderr,"case=%zu op=%s dtype=%s pitch=%d scenario=%u: %s\n",cases+1,names[size_t(op)],rad_dtype_name(dtype),pitch,unsigned(scenario),e.what());throw;
            }
        };
        for(auto op:{m::Op::hc_prepare,m::Op::hc_collapse,m::Op::router_topk,m::Op::sparse_attn})
            for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})if(op!=m::Op::router_topk||dtype==RAD_F32)
                for(int pitch=0;pitch<3;++pitch)one(op,dtype,pitch); // 30 normal cases
        for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(int pitch=0;pitch<3;++pitch)one(m::Op::hc_prepare,dtype,pitch,Scenario::hc_zero); // 9
        for(auto scenario:{Scenario::router_tie,Scenario::router_bias,Scenario::router_top1,Scenario::router_no_norm,Scenario::router_underflow})
            for(int pitch=0;pitch<3;++pitch)one(m::Op::router_topk,RAD_F32,pitch,scenario); // 15
        for(auto scenario:{Scenario::sparse_disabled,Scenario::sparse_empty,Scenario::sparse_sink,Scenario::sparse_repeat,Scenario::sparse_unused_nan})
            for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(int pitch=0;pitch<3;++pitch)one(m::Op::sparse_attn,dtype,pitch,scenario); // 45
        for(auto op:{m::Op::hc_prepare,m::Op::hc_collapse,m::Op::router_topk,m::Op::sparse_attn})
            for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})if(op!=m::Op::router_topk||dtype==RAD_F32)
                for(int pitch=0;pitch<3;++pitch)one(op,dtype,pitch,Scenario::input_nan); // 30
        for(auto scenario:{Scenario::fn_nan,Scenario::scale_inf,Scenario::base_nan})one(m::Op::hc_prepare,RAD_F32,2,scenario);
        one(m::Op::hc_collapse,RAD_F32,2,Scenario::previous_nan);one(m::Op::router_topk,RAD_F32,2,Scenario::correction_nan);
        for(auto scenario:{Scenario::sink_inf,Scenario::sink_nan,Scenario::id_negative,Scenario::id_past,Scenario::id_max,Scenario::kv_nan,Scenario::kv_inf})
            one(m::Op::sparse_attn,RAD_F32,2,scenario); // 12 additional payload errors
        size_t reset_cycles=0;
        auto cycle=[&](m::Op op,uint32_t dtype,Scenario scenario){derived_reset_cycle(launch[size_t(op)],Fixture(op,dtype,2,scenario),stream);cases+=3;++reset_cycles;};
        for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})cycle(m::Op::hc_prepare,dtype,Scenario::prepare_derived);
        for(uint32_t dtype:{RAD_F32,RAD_BF16})cycle(m::Op::hc_collapse,dtype,Scenario::collapse_derived);
        for(uint32_t dtype:{RAD_F16,RAD_BF16})cycle(m::Op::hc_collapse,dtype,Scenario::collapse_cast);
        cycle(m::Op::router_topk,RAD_F32,Scenario::router_quotient);cycle(m::Op::router_topk,RAD_F32,Scenario::router_scaled);
        for(uint32_t dtype:{RAD_F32,RAD_BF16})cycle(m::Op::sparse_attn,dtype,Scenario::sparse_numerator);
        require(reset_cycles==derived_fault_cycles,"derived reset-cycle count drifted");
        require(cases==expected_cases,"checker case count drifted");check(hipStreamSynchronize(stream));check(hipStreamDestroy(stream));stream=nullptr;
        dlclose(plugin);plugin=nullptr;
        std::printf("PASS: %zu tiny DS41 math GPU cases on %s, including %zu good-bad-good derived-fault cycles; guarded operands/scratch, explicit validation flags; not a model-quality or speed test\n",cases,properties.gcnArchName,reset_cycles);
        return 0;
    } catch(const std::exception& e) {
        if(stream) {static_cast<void>(hipStreamSynchronize(stream));static_cast<void>(hipStreamDestroy(stream));}
        if(plugin)dlclose(plugin);
        std::fprintf(stderr,"ds41-math-selftest: %s\n",e.what());return 1;
    }
}
