/* SOURCE ONLY. Run CHECK LIB DEVICE only inside the ordinary two-lock stopped/restored-QFN
 * window. Independent old/native exports + exact alias + transient full-product probe. Never
 * accept NaN numeric equality. This is a component byte gate, not timing/model authority. */
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "rad_plugin.h"
#include "../../arch/common/rad_ds41_fp8_runtime.h"
namespace f=rad_ds41_fp8_runtime;
#define HC(x) do{const auto e=(x);if(e!=hipSuccess){std::fprintf(stderr,"FAIL %s: %s\n",#x,hipGetErrorString(e));std::exit(2);}}while(0)
using Launch=int(*)(const RadArgs*,RadStream);
using Probe=int(*)(const RadArgs*,RadStream,int,int);
RadParam pi(const char*k,int64_t v){RadParam p{};p.key=k;p.kind=RAD_P_INT;p.ival=v;return p;}
RadParam ps(const char*k,const char*v){RadParam p{};p.key=k;p.kind=RAD_P_STR;p.sval=v;return p;}
RadTensor tensor(void*p,uint32_t dt,std::initializer_list<int64_t>shape){RadTensor t{};t.data=p;t.dtype=dt;t.rank=shape.size();
    int i=0;for(auto n:shape)t.shape[i++]=n;rad_tensor_pack(&t);return t;}
struct Pair {
    std::vector<uint8_t>seed,old,now;void*a=nullptr,*b=nullptr;
    explicit Pair(size_t n):seed(n,0xA5),old(n),now(n){HC(hipMalloc(&a,n));HC(hipMalloc(&b,n));}
    ~Pair(){HC(hipFree(a));HC(hipFree(b));}
    void reset(hipStream_t s){HC(hipMemcpyAsync(a,seed.data(),seed.size(),hipMemcpyHostToDevice,s));HC(hipMemcpyAsync(b,seed.data(),seed.size(),hipMemcpyHostToDevice,s));}
    void read(hipStream_t s){HC(hipMemcpyAsync(old.data(),a,old.size(),hipMemcpyDeviceToHost,s));HC(hipMemcpyAsync(now.data(),b,now.size(),hipMemcpyDeviceToHost,s));}
};
int call(Launch fn,std::vector<RadTensor>&t,std::vector<RadParam>&p,hipStream_t s,int rank){
    RadArgs a{};a.t=t.data();a.n_t=t.size();a.p=p.data();a.n_p=p.size();a.rank=rank;a.world_size=4;return fn(&a,(RadStream)s);}
int probe(Probe fn,std::vector<RadTensor>&t,std::vector<RadParam>&p,hipStream_t s,int op,int candidate){
    RadArgs a{};a.t=t.data();a.n_t=t.size();a.p=p.data();a.n_p=p.size();a.world_size=4;return fn(&a,(RadStream)s,op,candidate);}
struct Exports {
    Launch old[2]{},now[2]{};Probe side=nullptr;
    explicit Exports(void*lib){const char*original[]={"quant_roundtrip","rope_rows_roundtrip"};
        for(int k=0;k<2;++k){old[k]=(Launch)dlsym(lib,(std::string("r4d_rad_")+original[k]).c_str());
            now[k]=(Launch)dlsym(lib,(std::string("r4d_rad_")+f::kNames[k]).c_str());
            if(!old[k]||!now[k]){std::fprintf(stderr,"FAIL missing original/alias export %d\n",k);std::exit(2);}}
        side=(Probe)dlsym(lib,"r4d_rad_ds41_fp8_convert_probe");if(!side){std::fprintf(stderr,"FAIL missing full-product probe\n");std::exit(2);}}
};
uint16_t input(int row,int col,int n,int profile,int replay){
    const uint16_t edge[]={0,0x8000,1,0x8001,0x7F,0x807F,0x7F7F,0xFF7F,0x7F80,0xFF80,0x7F81,0xFFFF,0x7FC1,0xFFC2};
    if(profile==0)return uint16_t(row*n+col+replay*257);
    if(profile==1)return edge[(row+col+replay)%14];
    if(profile==2)return uint16_t(0x3B00+((row*73+col*17+replay*31)%2048))^(col&1?0x8000:0);
    // Finite data with an exceptional group leader; NaN/Inf groups are not silently excluded.
    return col%32==0?edge[(row+col/32+replay)%14]:uint16_t(0x3F00+col%256);
}
struct Totals {uint64_t cases=0,replays=0,products=0,codes=0,scales=0,bad=0,guards=0,refusals=0,fallbacks=0;
    std::array<uint64_t,f::kSlotCount>fast{},slow{};};
bool guards(const Pair&x,int M,int n,int pitch,int element){
    constexpr int G=32;for(size_t z=0;z<x.seed.size()/element;++z)
        if(z<G||z>=size_t(G+M*pitch)||(z-G)%pitch>=size_t(n))
            for(int b=0;b<element;++b)if(x.old[z*element+b]!=x.seed[z*element+b]||x.now[z*element+b]!=x.seed[z*element+b])return false;
    return true;
}
bool one(const Exports&e,int op,int n,int M,int profile,bool audit,int rank,Totals&sum){
    constexpr int G=32;const int pitch=n+16,sp=n+7,csrows=19,csp=67;
    Pair actual(size_t(2*G+M*pitch)*2),side(size_t(2*G+M*pitch)*2),codes(size_t(2*G+M*sp)),
         scales(size_t(2*G+M*sp)*4),products(size_t(2*G+M*sp)*4);
    std::vector<uint32_t>cs(csrows*csp,0x3E800000u);std::vector<int32_t>pos(M*2,-77);
    const uint32_t ce[]={0x3F800000u,0xBF000000u,0u,0x80000000u,0x7F800001u,0xFFC00002u,0x7F800000u};
    for(int r=0;r<csrows;++r)for(int c=0;c<64;++c)cs[r*csp+c]=profile==1?ce[(r+c)%7]:ce[(r+c)%4];
    for(int r=0;r<M;++r){const int which=(r+2*profile)%5;pos[r*2]=which==0?-1:which==1?csrows:r%csrows;}
    void*dcs=nullptr,*dp=nullptr,*counter=nullptr;HC(hipMalloc(&dcs,cs.size()*4));HC(hipMalloc(&dp,pos.size()*4));HC(hipMalloc(&counter,(f::kWords+2)*8));
    hipStream_t stream;HC(hipStreamCreate(&stream));HC(hipMemcpyAsync(dcs,cs.data(),cs.size()*4,hipMemcpyHostToDevice,stream));
    HC(hipMemcpyAsync(dp,pos.data(),pos.size()*4,hipMemcpyHostToDevice,stream));
    auto matrix=[&](void*p){auto t=tensor((uint16_t*)p+G,RAD_BF16,{M,n});t.stride[0]=pitch;return t;};
    auto probe_tensor=[&](void*p,uint32_t type,int bytes){auto t=tensor((uint8_t*)p+G*bytes,type,{M,n});t.stride[0]=sp;return t;};
    auto ct=tensor(dcs,RAD_F32,{csrows,64});ct.stride[0]=csp;auto pt=tensor(dp,RAD_I32,{M});pt.stride[0]=2;
    std::vector<RadTensor>a={matrix(actual.a)},b={matrix(actual.b)},pa={matrix(side.a)},pb={matrix(side.b)};
    if(op==f::Rope){for(auto*t:{&a,&b,&pa,&pb}){t->push_back(ct);t->push_back(pt);}}
    for(int k=0;k<3;++k){Pair*p=k==0?&codes:k==1?&scales:&products;const int bytes=k?4:1;const uint32_t dt=k?RAD_F32:RAD_U8;
        pa.push_back(probe_tensor(p->a,dt,bytes));pb.push_back(probe_tensor(p->b,dt,bytes));}
    if(audit)b.push_back(tensor((uint64_t*)counter+1,RAD_I64,{f::kWords}));
    std::vector<RadParam>p={pi("M",M),pi("n",n),pi("block",32),ps("format","fp8_e4m3"),ps("scale","ue8m0"),ps("dtype","bf16"),
        pi("n_head",16),pi("head_dim",512),pi("rot",64),pi("off",448),ps("mode","gptj"),pi("inverse",1)};
    auto invoke=[&](){int rc=call(e.old[op],a,p,stream,rank);if(rc<0)return rc;rc=call(e.now[op],b,p,stream,rank);
        if(rc<0)return rc;rc=probe(e.side,pa,p,stream,op,0);return rc<0?rc:probe(e.side,pb,p,stream,op,1);};
    HC(hipStreamSynchronize(stream));hipGraph_t graph=nullptr;hipGraphExec_t exec=nullptr;
    HC(hipStreamBeginCapture(stream,hipStreamCaptureModeGlobal));const int rc=invoke();HC(hipStreamEndCapture(stream,&graph));
    bool okay=rc>=0;if(okay)HC(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    for(int replay=0;replay<2&&okay;++replay){
        for(int r=0;r<M;++r)for(int c=0;c<n;++c){const uint16_t v=input(r,c,n,profile,replay);
            std::memcpy(actual.seed.data()+size_t(G+r*pitch+c)*2,&v,2);}
        side.seed=actual.seed;for(Pair*x:{&actual,&side,&codes,&scales,&products})x->reset(stream);
        std::array<uint64_t,f::kWords+2>counts{};counts.front()=counts.back()=0xA55AA55AA55AA55Aull;
        if(replay==0)HC(hipMemcpyAsync(counter,counts.data(),sizeof counts,hipMemcpyHostToDevice,stream));
        HC(hipGraphLaunch(exec,stream));for(Pair*x:{&actual,&side,&codes,&scales,&products})x->read(stream);
        HC(hipMemcpyAsync(counts.data(),counter,sizeof counts,hipMemcpyDeviceToHost,stream));HC(hipStreamSynchronize(stream));
        uint64_t bad=actual.old!=actual.now;bad+=side.old!=side.now;bad+=side.old!=actual.old;bad+=side.now!=actual.old;
        bad+=codes.old!=codes.now;bad+=scales.old!=scales.now;bad+=products.old!=products.now;
        const bool g=guards(actual,M,n,pitch,2)&&guards(side,M,n,pitch,2)&&guards(codes,M,n,sp,1)&&guards(scales,M,n,sp,4)&&guards(products,M,n,sp,4);
        bad+=counts.front()!=0xA55AA55AA55AA55Aull||counts.back()!=counts.front();
        const int slot=f::slot((f::Op)op,n);
        for(int k=0;k<f::kSlotCount;++k){const auto*q=counts.data()+1+k*f::FieldCount;
            bad+=!f::valid_row(q);if(!audit||k!=slot)for(int j=0;j<f::FieldCount;++j)bad+=q[j]!=0;
            else bad+=q[f::Fast]+q[f::Fallback]!=uint64_t(replay+1)*M*n;}
        if(replay==1&&audit){sum.fast[slot]+=counts[1+slot*f::FieldCount+f::Fast];sum.slow[slot]+=counts[1+slot*f::FieldCount+f::Fallback];}
        sum.bad+=bad;sum.guards+=!g;sum.products+=uint64_t(M)*n;sum.codes+=uint64_t(M)*n;sum.scales+=uint64_t(M)*n;++sum.replays;
        okay=!bad&&g;
    }
    std::vector<uint32_t>csback(cs.size());std::vector<int32_t>posback(pos.size());
    HC(hipMemcpyAsync(csback.data(),dcs,cs.size()*4,hipMemcpyDeviceToHost,stream));HC(hipMemcpyAsync(posback.data(),dp,pos.size()*4,hipMemcpyDeviceToHost,stream));
    HC(hipStreamSynchronize(stream));okay=okay&&cs==csback&&pos==posback;
    if(exec)HC(hipGraphExecDestroy(exec));if(graph)HC(hipGraphDestroy(graph));HC(hipStreamDestroy(stream));
    HC(hipFree(counter));HC(hipFree(dp));HC(hipFree(dcs));++sum.cases;
    std::printf("fp8-runtime case op=%s n=%d M=%d profile=%d audit=%d replays=2 bad=%llu guards=%llu pass=%d\n",
        f::kNames[op],n,M,profile,int(audit),(unsigned long long)sum.bad,(unsigned long long)sum.guards,int(okay));
    return okay;
}
bool fallback_and_refusals(const Exports&e,int rank,Totals&sum){
    hipStream_t stream;HC(hipStreamCreate(&stream));Pair x(2*(4096+64));x.reset(stream);
    auto t=tensor((uint16_t*)x.a+32,RAD_BF16,{1,4096}),u=t;u.data=(uint16_t*)x.b+32;
    std::vector<RadTensor>a={t},b={u};std::vector<RadParam>p={pi("M",1),pi("n",4096),pi("block",32),ps("format","fp8_e4m3"),ps("scale","ue8m0"),ps("dtype","bf16")};
    int rc=call(e.old[f::Quant],a,p,stream,rank);if(rc>=0)rc=call(e.now[f::Quant],b,p,stream,rank);
    x.read(stream);HC(hipStreamSynchronize(stream));bool okay=rc>=0&&x.old==x.now;sum.fallbacks+=okay;
    // The native alias must refuse corrupt audit metadata before any candidate issue.
    b[0].shape[1]=512;p[1]=pi("n",512);b.push_back(tensor(x.b,RAD_I64,{f::kWords-1}));
    x.reset(stream);HC(hipStreamSynchronize(stream));const int refused=call(e.now[f::Quant],b,p,stream,rank);
    x.read(stream);HC(hipStreamSynchronize(stream));okay=okay&&refused<0&&x.now==x.seed;sum.refusals+=refused<0;
    b[0].stride[1]=2;const int stride=call(e.now[f::Quant],b,p,stream,rank);okay=okay&&stride<0;sum.refusals+=stride<0;
    HC(hipStreamDestroy(stream));return okay;
}
int main(int argc,char**argv){
    if(argc!=3){std::fprintf(stderr,"usage: %s LIBR4D.so DEVICE\n",argv[0]);return 2;}
    char*end=nullptr;long dev=std::strtol(argv[2],&end,10);if(!end||end==argv[2]||*end||dev<0||dev>63)return 2;
    HC(hipSetDevice((int)dev));hipDeviceProp_t prop{};HC(hipGetDeviceProperties(&prop,(int)dev));
    char bdf[64]{};HC(hipDeviceGetPCIBusId(bdf,sizeof bdf,(int)dev));
    if(std::strcmp(prop.gcnArchName,"gfx1201")&&std::strncmp(prop.gcnArchName,"gfx1201:",8)){std::fprintf(stderr,"FAIL gfx1201 required\n");return 2;}
    std::printf("fp8-runtime physical device=%ld bdf=%s arch=%s\n",dev,bdf,prop.gcnArchName);
    void*lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!lib){std::fprintf(stderr,"FAIL dlopen %s\n",dlerror());return 2;}
    Exports e(lib);Totals sum;bool okay=true;
    for(int op=0;op<2&&okay;++op)for(int n:{512,8192}){
        if(op==f::Rope&&n!=8192)continue;
        for(int M:{1,2,8,17,128,2048,4096})for(int profile=0;profile<4&&okay;++profile)for(bool audit:{false,true})
            if(okay)okay=one(e,op,n,M,profile,audit,(int)dev,sum);
    }
    if(okay)okay=fallback_and_refusals(e,(int)dev,sum);
    for(int k=0;k<f::kSlotCount;++k){std::printf("fp8-runtime coverage op=%s n=%d fast=%llu fallback=%llu\n",f::kNames[f::kSlots[k].op],f::kSlots[k].width,
        (unsigned long long)sum.fast[k],(unsigned long long)sum.slow[k]);okay=okay&&sum.fast[k]>0&&sum.slow[k]>0;}
    okay=okay&&sum.cases==168&&sum.replays==336&&sum.refusals==2&&sum.fallbacks==1&&!sum.bad&&!sum.guards;
    if(dlclose(lib))okay=false;
    std::printf("%s ds41_fp8_runtime_exact cases=%llu replays=%llu products=%llu codes=%llu scales=%llu bad=%llu guards=%llu refusals=%llu whole_old=%llu scope=component serving_proof=0 integrated=0 performance_eligible=0\n",
        okay?"PASS":"FAIL",(unsigned long long)sum.cases,(unsigned long long)sum.replays,(unsigned long long)sum.products,
        (unsigned long long)sum.codes,(unsigned long long)sum.scales,(unsigned long long)sum.bad,(unsigned long long)sum.guards,
        (unsigned long long)sum.refusals,(unsigned long long)sum.fallbacks);
    return okay?0:1;
}
