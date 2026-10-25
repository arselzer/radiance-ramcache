/* Explicit standalone native HIP gate, not automatic and not proof of actual Core AQL.
 * Compiler/device execution belongs to the parent. No models, Pools or service API. */
#include <hip/hip_runtime.h>
#include "ds41_compress_finish_fixture.h"
#include "ds41_test_scratch.h"
#include <cstdio>
#include <memory>
#include <string>
namespace f=rad::ds41::compress_finish;
namespace fix=f::testing;
namespace {
void need(bool condition,const char* why){if(!condition)throw std::runtime_error(why);}
void check(hipError_t error){if(error!=hipSuccess)throw std::runtime_error(hipGetErrorString(error));}
template<class T>std::vector<uint8_t> bytes(const T& values){std::vector<uint8_t> result(values.size()*sizeof(*values.data()));std::memcpy(result.data(),values.data(),result.size());return result;}
struct Device {
    static constexpr size_t guard=64;void* base=nullptr;size_t size;
    explicit Device(const std::vector<uint8_t>& source):size(source.size()){
        need(size>0&&size<=2*1024*1024,"native checker tensor cap");check(hipMalloc(&base,size+2*guard));
        auto rc=hipMemset(base,0xC7,size+2*guard);if(rc==hipSuccess)rc=hipMemcpy(data(),source.data(),size,hipMemcpyHostToDevice);
        if(rc!=hipSuccess){static_cast<void>(hipFree(base));base=nullptr;check(rc);}
    }
    Device(const Device&)=delete;Device& operator=(const Device&)=delete;
    ~Device(){if(base)static_cast<void>(hipFree(base));}
    void* data()const{return static_cast<uint8_t*>(base)+guard;}
    void put(const std::vector<uint8_t>& source){need(source.size()==size,"reused native tensor extent changed");check(hipMemcpy(data(),source.data(),size,hipMemcpyHostToDevice));}
    std::vector<uint8_t> read()const {
        std::vector<uint8_t> result(size+2*guard);check(hipMemcpy(result.data(),base,result.size(),hipMemcpyDeviceToHost));
        for(size_t i=0;i<guard;++i)need(result[i]==0xC7&&result[guard+size+i]==0xC7,"native allocation red zone changed");
        return {result.begin()+guard,result.end()-guard};
    }
};
int order(uint16_t bits){return bits&0x8000u?32768-int(bits&0x7FFFu):32768+int(bits);}
struct Run {
    std::array<std::unique_ptr<Device>,f::operand_count> device;std::unique_ptr<Device> scratch;std::vector<uint8_t> flag_before;hipStream_t stream;
    Run(fix::Fixture& c,hipStream_t s):stream(s){size_t total=0;for(int i=0;i<f::operand_count;++i){const auto source=bytes(c.buffers[size_t(i)].words);total+=source.size()+2*Device::guard;need(total<=1024*1024,"aggregate tiny checker allocation cap");device[size_t(i)]=std::make_unique<Device>(source);}
        flag_before=rad::ds41::testing::make_gpu_scratch<f::Plan>(bytes(c.flag));
        // Extra caller-owned canary backing also keeps the deliberate native-reserve
        // alias descriptor within one allocation, distinct from all actual operands.
        flag_before.resize(flag_before.size()+c.buffers[f::rotated].words.size()*4,0xAB);total+=flag_before.size()+2*Device::guard;need(total<=1024*1024,"aggregate scratch allocation cap");scratch=std::make_unique<Device>(flag_before);}
    void pass(fix::Fixture c){
        const auto original=c.buffers;auto oracle=c;const int expected_status=oracle.run();std::vector<fix::Buffer> expected;
        if(expected_status==RAD_OK)expected=c.expected();
        for(int i=0;i<f::operand_count;++i)device[size_t(i)]->put(bytes(c.buffers[size_t(i)].words));
        auto args=c.args();for(int i=0;i<f::operand_count;++i)c.tensors[size_t(i)].data=device[size_t(i)]->data();args.scratch=scratch->data();args.scratch_bytes=rad::ds41::gpu_scratch_bytes<f::Plan>();
        need(ds41_compress_finish(&args,stream)==RAD_OK,"valid descriptor enqueue failed");check(hipStreamSynchronize(stream));
        const auto after_flag=scratch->read();int32_t status=0;std::memcpy(&status,after_flag.data(),4);need(status==expected_status,"native payload status differs from HOST preflight");
        need(rad::ds41::testing::gpu_scratch_guards<f::Plan>(after_flag,flag_before),"staged Plan prefix/tail guard changed");
        std::array<std::vector<uint8_t>,2> received;
        for(int i=0;i<f::operand_count;++i){const auto got=device[size_t(i)]->read();const auto before=bytes(original[size_t(i)].words);
            if(i<f::unrotated||status!=RAD_OK){need(got==before,"native source changed or failed batch wrote output");continue;}
            received[size_t(i-f::unrotated)]=got;const auto& output=expected[size_t(i)];std::vector<bool> touched(got.size(),false);const size_t width=size_t(rad_dtype_bits(c.dtype)/8);
            for(int64_t at=0;at<c.rows*c.dim;++at){const size_t off=size_t(output.offset(at))*width;
                if(!c.buffers[f::emit].id(at/c.dim)){for(size_t b=0;b<width;++b)need(got[off+b]==0,"inactive native result was not exact positive zero");}
                else if(c.dtype==RAD_F32){float value;std::memcpy(&value,got.data()+off,4);const float want=output.get(at);need(std::isfinite(value)&&std::fabs(value-want)<=4e-6f+4e-5f*std::fabs(want),"native F32 finish value differs from independent composition");
                    if(want==0.f&&at%c.dim<c.dim-c.rotary)need(std::signbit(value)==std::signbit(want),"native active prefix lost expected signed zero");}
                else {uint16_t value,want;std::memcpy(&value,got.data()+off,2);std::memcpy(&want,reinterpret_cast<const uint8_t*>(output.words.data())+off,2);const uint32_t storage=value;
                    need(std::isfinite(rad_load_f32(&storage,c.dtype,0))&&std::abs(order(value)-order(want))<=1,"native rounded finish value differs by more than one finite stored ULP");
                    if((want&0x7FFFu)==0&&at%c.dim<c.dim-c.rotary)need(value==want,"native active prefix lost expected signed zero");}
                for(size_t b=0;b<width;++b)touched[off+b]=true;
            }
            for(size_t b=0;b<got.size();++b)if(!touched[b])need(got[b]==before[b],"native finish wrote view padding/canary");
        }
        if(status==RAD_OK){const size_t width=size_t(rad_dtype_bits(c.dtype)/8);for(int64_t row=0;row<c.rows;++row)for(int64_t d=0;d<c.dim-c.rotary;++d){const auto a=size_t(c.buffers[f::unrotated].offset(row*c.dim+d))*width,b=size_t(c.buffers[f::rotated].offset(row*c.dim+d))*width;need(std::memcmp(received[0].data()+a,received[1].data()+b,width)==0,"native prefix copy changed rounded raw bits");}}
    }
};
void mutate(fix::Fixture& c,int fault){const int64_t row=c.rows-1;
    if(fault==0)c.buffers[f::emit].id(row,2);if(fault==1)c.buffers[f::emit].id(row,-1);
    if(fault==2)c.buffers[f::first_positions].id(row,-1);if(fault==3)c.buffers[f::first_positions].id(row,int32_t(c.positions));if(fault==4)c.buffers[f::first_positions].id(row,INT32_MAX);
    if(fault==5)c.buffers[f::pooled].set(row*c.dim+c.dim-1,NAN);if(fault==6)c.buffers[f::pooled].set(row*c.dim+c.dim-1,INFINITY);if(fault==7)c.buffers[f::gain].set(c.dim-1,NAN);
    if(fault==8)c.buffers[f::pairs].set(c.buffers[f::first_positions].id(row)*c.rotary,2.f);if(fault==9)c.buffers[f::pairs].set(c.buffers[f::first_positions].id(row)*c.rotary+1,NAN);
    if(fault==10)c.buffers[f::pooled].set(row*c.dim,1e10f);if(fault==11)c.buffers[f::gain].set(c.dim-1,1e10f);if(fault==12)c.buffers[f::pooled].set(row*c.dim,1e30f);if(fault==13)c.buffers[f::gain].set(c.dim-1,INFINITY);
}
void metadata(hipStream_t stream,int fault){
    fix::Fixture c;Run run(c,stream);auto args=c.args();for(int i=0;i<f::operand_count;++i)c.tensors[size_t(i)].data=run.device[size_t(i)]->data();args.scratch=run.scratch->data();args.scratch_bytes=rad::ds41::gpu_scratch_bytes<f::Plan>();
    if(fault==0)c.integer("M",0);if(fault==1)c.integer("n",1025);if(fault==2)c.integer("rotary_dim",3);if(fault==3)c.integer("rotary_dim",c.dim+2);if(fault==4)c.integer("ratio",3);
    if(fault==5)c.floating("eps",0);if(fault==6)c.floating("eps",NAN);if(fault==7)c.text("a_dtype","i32");if(fault==8)c.tensors[f::unrotated].dtype=RAD_U32;if(fault==9)c.tensors[f::pooled].rank=1;if(fault==10)c.tensors[f::rotated].data=nullptr;
    if(fault==11)c.tensors[f::pooled].stride[0]=INT64_MAX;if(fault==12)c.tensors[f::gain].stride[0]=-1;if(fault==13)c.tensors[f::pooled].data=reinterpret_cast<void*>(UINTPTR_MAX-3);if(fault==14)c.tensors[f::pooled].data=static_cast<uint8_t*>(c.tensors[f::pooled].data)+1;
    if(fault==15)args.scratch_bytes=3;if(fault==16)args.scratch=nullptr;if(fault==17)args.scratch=reinterpret_cast<void*>(UINTPTR_MAX-3);if(fault==18)args.scratch=c.tensors[f::pooled].data;if(fault==19)c.tensors[f::pairs].shape[0]=f::max_positions+1;
    int alias=20;for(int output:{f::unrotated,f::rotated})for(int other=0;other<f::operand_count;++other)if(other!=output){if(fault==alias)c.tensors[size_t(output)].data=c.tensors[size_t(other)].data;++alias;}
    if(fault==32)c.tensors[f::rotated].data=static_cast<uint8_t*>(args.scratch)+rad::ds41::device_plan_offset; // semantic prefix is disjoint; complete native reserve is not
    need(ds41_compress_finish(&args,stream)<0,"malformed metadata was enqueued");check(hipStreamSynchronize(stream));need(run.scratch->read()==run.flag_before,"metadata refusal changed native scratch");
    for(int i=0;i<f::operand_count;++i)need(run.device[size_t(i)]->read()==bytes(c.buffers[size_t(i)].words),"metadata refusal changed operand");
}
}
int main(int argc,char** argv){
    if((argc!=2&&argc!=3)||std::strcmp(argv[1],"--run-gpu")){std::fprintf(stderr,"usage: ds41-compress-finish-selftest --run-gpu [device]\nExplicit native HIP only; no Core AQL/full model proof.\n");return 2;}
    hipStream_t stream=nullptr;
    try {int device=0;if(argc==3){size_t n=0;device=std::stoi(argv[2],&n);need(n==std::strlen(argv[2])&&device>=0,"invalid device");}
        check(hipSetDevice(device));hipDeviceProp_t props{};check(hipGetDeviceProperties(&props,device));need(!std::strncmp(props.gcnArchName,"gfx1201",7)&&(!props.gcnArchName[7]||props.gcnArchName[7]==':'),"requires gfx1201");check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
        size_t good=0,bad=0,refused=0;auto one=[&](fix::Fixture c,bool valid=true){Run r(c,stream);r.pass(c);if(valid)++good;else ++bad;};
        for(uint32_t ad:{RAD_F32,RAD_F16,RAD_BF16})for(uint32_t yd:{RAD_F32,RAD_F16,RAD_BF16})for(int64_t dim:{2,7,128,512})for(int ratio:{1,2})for(int pitch:{0,1,2})one(fix::Fixture(ad,yd,dim,ratio,pitch));
        for(uint32_t yd:{RAD_F32,RAD_F16,RAD_BF16})one(fix::Fixture(RAD_F32,yd,1024,2,1));
        for(uint32_t ad:{RAD_F32,RAD_F16,RAD_BF16})for(uint32_t yd:{RAD_F32,RAD_F16,RAD_BF16}){fix::Fixture c(ad,yd);for(int64_t row=0;row<c.rows;++row)c.buffers[f::emit].id(row,0);one(c);}
        for(uint32_t yd:{RAD_F32,RAD_F16,RAD_BF16}){fix::Fixture c(RAD_F32,yd);c.buffers[f::first_positions].id(5,62);one(c);}
        for(uint32_t yd:{RAD_F32,RAD_F16,RAD_BF16}){fix::Fixture c(yd,yd,7,1);for(int64_t d=0;d<c.dim;++d)c.buffers[f::gain].set(d,1.f);for(int64_t at=0;at<c.rows*c.dim;++at)c.buffers[f::pooled].set(at,-0.f);one(c);}
        for(int fault=0;fault<14;++fault){fix::Fixture c(RAD_F32,fault==10||fault==11?RAD_F16:RAD_F32,7,1);mutate(c,fault);one(c,false);}
        {fix::Fixture c;c.buffers[f::first_positions].id(5,3);one(c,false);}
        {fix::Fixture c(RAD_F32,RAD_F16,2,1,0,1);for(int d=0;d<2;++d){c.buffers[f::pooled].set(d,1.f);c.buffers[f::gain].set(d,60000.f);}c.buffers[f::pairs].set(0,std::sqrt(0.5f));c.buffers[f::pairs].set(1,std::sqrt(0.5f));one(c,false);}
        {fix::Fixture c;for(int64_t row=0;row<c.rows;++row)c.buffers[f::emit].id(row,0);c.buffers[f::gain].set(0,NAN);one(c,false);}
        {fix::Fixture c(RAD_F32,RAD_BF16,128,1);Run reuse(c,stream);reuse.pass(c);++good;auto malformed=c;malformed.buffers[f::first_positions].id(c.rows-1,-1);reuse.pass(malformed);++bad;reuse.pass(c);++good;}
        for(int fault=0;fault<33;++fault){metadata(stream,fault);++refused;}
        need(good==236&&bad==18&&refused==33,"compress finish checker coverage drifted");check(hipStreamSynchronize(stream));check(hipStreamDestroy(stream));stream=nullptr;
        std::printf("PASS: compress finish native %zu good, %zu atomic payload failures, %zu metadata refusals on %s; full Plan guards, all dtypes/pitches, positive inactive zeros, reset/reuse.\n",good,bad,refused,props.gcnArchName);return 0;
    }catch(const std::exception& e){if(stream){static_cast<void>(hipStreamSynchronize(stream));static_cast<void>(hipStreamDestroy(stream));}std::fprintf(stderr,"compress finish native checker: %s\n",e.what());return 1;}
}
