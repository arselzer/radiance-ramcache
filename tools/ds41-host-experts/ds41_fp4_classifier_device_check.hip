/* SOURCE oracle. Execute ONLY in an ordinary stopped/restored-QFN GPU window
 * holding both research locks and build.lock. Usage: CHECK LIBR4D.so
 * No timings, model, feature flag, engine ABI or serving authority. Raw u32
 * comparisons plus full BF16/guard comparisons to the ACTUAL original producer.
 */
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "rad_plugin.h"
#include "ds41_fp4_classifier_oracle.h"
namespace fp=rad_ds41_fp4_classifier;
namespace ref=ds41_fp4_oracle;
#define HC(x) do {const hipError_t e=(x);if(e!=hipSuccess){std::fprintf(stderr,"FAIL %s: %s\n",#x,hipGetErrorString(e));std::exit(2);}}while(0)
namespace {
struct Counts { unsigned fast=0,fallback=0,bad=0; };
__global__ void raw_check(const uint32_t* in,uint32_t* old,uint32_t* now,size_t n,Counts* count) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
    fp::Result result;const bool fast=fp::classify_bits(in[i],&result);
    old[i]=ref::bits(ref::old_nearest(ref::value(in[i])));
    now[i]=fast?result.value_bits:ref::bits(ref::old_nearest(ref::value(in[i])));
    atomicAdd(fast?&count->fast:&count->fallback,1u);
    if(old[i]!=now[i])atomicAdd(&count->bad,1u);
}
// Independent original E4M3 scale helpers; checked below against the loaded producer.
__device__ float old_e4(uint32_t b) {
    const uint32_t s=(b>>7)<<31,e=(b>>3)&15;uint32_t m=b&7,u;
    if(!e){if(!m)u=s;else{int sh=0;while(!(m&8)){m<<=1;++sh;}u=s|(uint32_t(127-6-sh)<<23)|((m&7)<<20);}}
    else if(e==15&&m==7)u=s|0x7FC00000u;
    else u=s|((e+120)<<23)|(m<<20);
    return __uint_as_float(u);
}
__device__ uint32_t old_to_e4(float f) {
    const uint32_t u=__float_as_uint(f),s=(u>>24)&128,a=u&0x7FFFFFFF;
    if(a>0x7F800000)return s|127;if(a>=0x43E00000)return s|126;
    const int e=int(a>>23)-127;uint32_t m=a&0x7FFFFF;
    if(e< -10)return s;
    if(e< -6){const int sh=-6-e+20;m|=0x800000;const uint32_t lo=m&((1u<<sh)-1),half=1u<<(sh-1);uint32_t q=m>>sh;if(lo>half||(lo==half&&(q&1)))++q;return s|q;}
    uint32_t q=m>>20;const uint32_t lo=m&0xFFFFF;int ee=e+7;
    if(lo>0x80000||(lo==0x80000&&(q&1)))++q;if(q==8){q=0;++ee;}
    return s|((ee>15||(ee==15&&q>=7))?126:(uint32_t(ee)<<3)|q);
}
__device__ float old_scale(float a,int block) {
    if(block==16){a=a<6.0f*0x1p-9f?6.0f*0x1p-9f:a;return old_e4(old_to_e4(a/6.0f));}
    a=a<6.0f*0x1p-126f?6.0f*0x1p-126f:a;
    const float inv=1.0f/6.0f;const uint32_t u=__float_as_uint(a*inv);
    const int e=int((u>>23)&255)-127+((u&0x7FFFFF)?1:0);
    return __uint_as_float(uint32_t(e+127)<<23);
}
__global__ void producer_candidate(const uint16_t* input,uint16_t* now,int M,int n,int pitch,int block,Counts* count) {
    const int group=int(blockIdx.x*blockDim.x+threadIdx.x),ng=n/block;if(group>=M*ng)return;
    const int r=group/ng,first=(group%ng)*block;float amax=0;
    for(int c=0;c<block;++c){float a=fabsf(ref::value(uint32_t(input[size_t(r)*pitch+first+c])<<16));a=a==a?a:0;amax=amax<a?a:amax;}
    const float scale=old_scale(amax,block);
    for(int c=0;c<block;++c){const float v=ref::value(uint32_t(input[size_t(r)*pitch+first+c])<<16)/scale;
        const float clamp=v< -6.0f?-6.0f:6.0f<v?6.0f:v;fp::Result result;const bool fast=fp::classify_bits(ref::bits(clamp),&result);
        const float q=fast?ref::value(result.value_bits):ref::old_nearest(clamp);
        now[size_t(r)*pitch+first+c]=ref::bf16(q*scale);atomicAdd(fast?&count->fast:&count->fallback,1u);
    }
}
using Launch=int(*)(const RadArgs*,RadStream);
RadParam pi(const char* name,int64_t v){RadParam p{};p.key=name;p.kind=RAD_P_INT;p.ival=v;return p;}
RadParam ps(const char* name,const char* v){RadParam p{};p.key=name;p.kind=RAD_P_STR;p.sval=v;return p;}
bool producers(Launch quant,Counts* count) {
    bool okay=true;constexpr int guard=32;
    // Exactly65536 groups each: every BF16 seed is seen as a group's first input.
    for(auto shape : {std::array<int,3>{2048,512,16},{16384,128,32},{512,4096,32}}) {
        const int M=shape[0],n=shape[1],block=shape[2],pitch=n+16,groups=n/block;
        const size_t total=size_t(M)*pitch+2*guard;
        std::vector<uint16_t> input(total,0x5AA5),old(total),now(total);
        uint16_t *pre=nullptr,*actual=nullptr,*candidate=nullptr;
        for(auto p:{&pre,&actual,&candidate})HC(hipMalloc((void**)p,total*2));
        const uint16_t edge[]={0,0x8000,1,0x8001,0x007F,0x807F,0x7F7F,0xFF7F,0x7F80,0xFF80,0x7FC1,0xFFFF};
        const uint16_t ties[]={0,0x3E80,0x3F40,0x3FA0,0x3FE0,0x4020,0x4060,0x40A0,0x40C0};
        for(int profile=0;profile<4;++profile){
            for(int r=0;r<M;++r)for(int g=0;g<groups;++g)for(int c=0;c<block;++c){
                const uint16_t seed=uint16_t(r*groups+g);uint16_t v=c?0:seed;
                if(c&&profile==1)v=edge[(c+g)%12];
                if(c&&profile==2)v=uint16_t(seed^uint16_t(c*0x631));
                if(profile==3){v=ties[c%9];if(!v)v=seed%3==0?0x8000:seed%3==1?0:1;else v=uint16_t(int(v)+int(seed%3)-1);if(c==block-1)v=0x40C0;}
                input[guard+size_t(r)*pitch+g*block+c]=v;
            }
            for(auto p:{pre,actual,candidate})HC(hipMemcpy(p,input.data(),total*2,hipMemcpyHostToDevice));
            HC(hipMemset(count,0,sizeof(Counts)));
            producer_candidate<<<(M*groups+127)/128,128>>>(pre+guard,candidate+guard,M,n,pitch,block,count);
            RadTensor t{};t.data=actual+guard;t.dtype=RAD_BF16;t.rank=2;t.shape[0]=M;t.shape[1]=n;t.stride[0]=pitch;t.stride[1]=1;
            std::array<RadParam,6> p={pi("M",M),pi("n",n),pi("block",block),ps("format","fp4_e2m1"),ps("scale",block==16?"e4m3":"ue8m0"),ps("dtype","bf16")};
            RadArgs a{};a.t=&t;a.n_t=1;a.p=p.data();a.n_p=int(p.size());
            const int rc=quant(&a,nullptr);if(rc<0){std::fprintf(stderr,"FAIL original producer rc=%d\n",rc);okay=false;break;}
            HC(hipDeviceSynchronize());HC(hipMemcpy(old.data(),actual,total*2,hipMemcpyDeviceToHost));HC(hipMemcpy(now.data(),candidate,total*2,hipMemcpyDeviceToHost));
            Counts got{};HC(hipMemcpy(&got,count,sizeof(got),hipMemcpyDeviceToHost));size_t bad=0,guards=0;
            for(size_t i=0;i<total;++i){bad+=old[i]!=now[i];if(i<guard||i>=total-guard||(i-guard)%pitch>=size_t(n))guards+=old[i]!=input[i]||now[i]!=input[i];}
            const bool pass=!bad&&!guards&&uint64_t(got.fast)+got.fallback==uint64_t(M)*n&&got.fast>0;okay&=pass;
            std::printf("%s fp4-original-producer M=%d n=%d block=%d profile=%d fast=%u fallback=%u bad=%zu guards=%zu\n",pass?"PASS":"FAIL",M,n,block,profile,got.fast,got.fallback,bad,guards);
        }
        for(void* p:{(void*)pre,(void*)actual,(void*)candidate})HC(hipFree(p));
    }
    return okay;
}
}
int main(int argc,char** argv){
    if(argc!=2){std::fprintf(stderr,"usage: %s LIBR4D.so\n",argv[0]);return 2;}
    int device=0;HC(hipGetDevice(&device));char pci[64]{};HC(hipDeviceGetPCIBusId(pci,sizeof pci,device));std::printf("FP4_CLASSIFIER_CARD device=%d pci=%s\n",device,pci);
    void* lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!lib){std::fprintf(stderr,"FAIL dlopen: %s\n",dlerror());return 2;}
    const auto quant=(Launch)dlsym(lib,"r4d_rad_quant_roundtrip");if(!quant){std::fprintf(stderr,"FAIL missing original quant_roundtrip\n");return 2;}
    const auto input=ref::corpus();std::vector<uint32_t> old(input.size()+2,0xA55AA55A),now(old);
    uint32_t *in=nullptr,*a=nullptr,*b=nullptr;Counts* count=nullptr;
    HC(hipMalloc((void**)&in,input.size()*4));for(auto p:{&a,&b})HC(hipMalloc((void**)p,old.size()*4));HC(hipMalloc((void**)&count,sizeof(Counts)));
    HC(hipMemcpy(in,input.data(),input.size()*4,hipMemcpyHostToDevice));HC(hipMemcpy(a,old.data(),old.size()*4,hipMemcpyHostToDevice));HC(hipMemcpy(b,now.data(),now.size()*4,hipMemcpyHostToDevice));HC(hipMemset(count,0,sizeof(Counts)));
    raw_check<<<(input.size()+255)/256,256>>>(in,a+1,b+1,input.size(),count);HC(hipDeviceSynchronize());Counts got{};HC(hipMemcpy(&got,count,sizeof(got),hipMemcpyDeviceToHost));
    HC(hipMemcpy(old.data(),a,old.size()*4,hipMemcpyDeviceToHost));HC(hipMemcpy(now.data(),b,now.size()*4,hipMemcpyDeviceToHost));
    bool okay=old==now&&old.front()==0xA55AA55A&&old.back()==0xA55AA55A&&got.bad==0&&got.fast>0&&got.fallback>0&&uint64_t(got.fast)+got.fallback==input.size();
    std::printf("%s fp4-raw-u32 values=%zu fast=%u fallback=%u bad=%u\n",okay?"PASS":"FAIL",input.size(),got.fast,got.fallback,got.bad);
    okay=producers(quant,count)&&okay;
    for(void* p:{(void*)in,(void*)a,(void*)b,(void*)count})HC(hipFree(p));dlclose(lib);
    std::printf("%s ds41_fp4_classifier_standalone_exact raw_u32=literal actual_original_producer=12 guards=full serving_authority=0\n",okay?"PASS":"FAIL");return okay?0:1;
}
