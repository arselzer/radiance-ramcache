/* Source-only: ONLY launch in the two-lock stopped/restored production window.
 * Actual original scorer+scatter exports are the independent byte oracle. */
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include "rad_abi.h"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
using Launch=int(*)(const RadArgs*,RadStream);
void need(bool b,const char*s){if(!b)throw std::runtime_error(s);}
void hc(hipError_t e){if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
RadParam pi(const char*k,long long v){RadParam p{};p.key=k;p.kind=RAD_P_INT;p.ival=v;return p;}
RadParam pf(const char*k,double v){RadParam p{};p.key=k;p.kind=RAD_P_F64;p.dval=v;return p;}
RadParam ps(const char*k,const char*v){RadParam p{};p.key=k;p.kind=RAD_P_STR;p.sval=v;return p;}
RadTensor tensor(void*p,uint32_t dt,std::initializer_list<int64_t>s){RadTensor t{};t.data=p;t.dtype=dt;t.rank=s.size();int i=0;
    for(auto n:s)t.shape[i++]=n;int64_t stride=1;for(int d=(int)t.rank-1;d>=0;--d){t.stride[d]=stride;stride*=t.shape[d];}return t;}
struct Buffer {
    static constexpr size_t guard=64;uint8_t*p=nullptr;size_t n;std::vector<uint8_t>before;
    explicit Buffer(size_t bytes):n(bytes),before(bytes+2*guard,0xa5){hc(hipMalloc((void**)&p,before.size()));reset();}
    ~Buffer(){if(p)(void)hipFree(p);}Buffer(const Buffer&)=delete;Buffer&operator=(const Buffer&)=delete;
    void*data(){return p+guard;}void reset(){hc(hipMemcpy(p,before.data(),before.size(),hipMemcpyHostToDevice));}
    void set(const std::vector<uint32_t>&v){need(v.size()*4==n,"input length");std::memcpy(before.data()+guard,v.data(),n);reset();}
    std::vector<uint8_t>read(bool unchanged=false){std::vector<uint8_t>x(before.size());hc(hipMemcpy(x.data(),p,x.size(),hipMemcpyDeviceToHost));
        need(!std::memcmp(x.data(),before.data(),guard)&&!std::memcmp(x.data()+guard+n,before.data()+guard+n,guard),"outer guard changed");
        if(unchanged)need(x==before,"input/refused output changed");return x;}
};
struct API {
    void*h;Launch score,scatter[2],fused;uint32_t(*paths)();
    explicit API(const char*lib){h=dlopen(lib,RTLD_NOW|RTLD_LOCAL);need(h,"dlopen failed");
        score=get<Launch>("r4d_rad_router_scored_topk");scatter[0]=get<Launch>("r4d_rad_moe_scatter");
        scatter[1]=get<Launch>("r4d_rad_moe_scatter_inv");fused=get<Launch>("r4d_rad_ds41_scored_topk_scatter");
        paths=get<uint32_t(*)()>("r4d_rad_ds41_scored_scatter_paths");}
    template<class F>F get(const char*s){void*p=dlsym(h,s);need(p,"missing original/new export");Dl_info i{};
        need(dladdr(p,&i)&&i.dli_sname&&i.dli_saddr==p&&!std::strcmp(i.dli_sname,s),"miswired actual export");return(F)p;}
};
RadArgs args(std::vector<RadTensor>&t,std::vector<RadParam>&p,int r){RadArgs a{};a.t=t.data();a.n_t=t.size();a.p=p.data();a.n_p=p.size();a.rank=r;a.world_size=4;return a;}
void seti(std::vector<RadParam>&p,const char*k,long long v){for(auto&x:p)if(!std::strcmp(x.key,k)){x.ival=v;return;}throw std::runtime_error("missing int param");}
uint32_t fbits(float v){uint32_t x;std::memcpy(&x,&v,4);return x;}
void inputs(int mode,int serial,std::vector<uint32_t>&l,std::vector<uint32_t>&b,int pitch){
    for(int i=0;i<384;++i){l[i]=fbits(float((i*17+serial*7)%97-48)*.25f);b[(size_t)i*pitch]=fbits(float((i*13)%19-9)*.125f);}
    if(mode==0){for(int i=0;i<384;++i){l[i]=0;b[(size_t)i*pitch]=0;}} // complete tie
    if(mode==1){for(int i=0;i<384;++i)l[i]=(i&1)?0x80000000u:0u;}
    if(mode==2){for(int i=0;i<384;++i)l[i]=((i&1)?0xffc00100u:0x7fc00001u)+(uint32_t)i;}
    if(mode==3){for(int i=0;i<384;++i)l[i]=((i&1)?0xff800001u:0x7f800001u)+(uint32_t)i;}
    if(mode==4){for(int i=0;i<384;++i)l[i]=0xff800000u;}
    if(mode==5){for(int i=0;i<384;++i)l[i]=0x7f800000u;}
    if(mode==6){for(int i=0;i<384;++i){l[i]=(i%3)?0x7f800000u:0xff800000u;b[(size_t)i*pitch]=i%7?0:0xff800000u;}}
    if(mode==7){for(int i=0;i<384;++i){l[i]=0;b[(size_t)i*pitch]=0x7fc01234u;}}
    if(mode==8){for(int i=0;i<384;++i){l[i]=0;b[(size_t)i*pitch]=0xff800000u;}}
    if(mode==9){for(int i=0;i<384;++i)l[i]=0x41a00000u+(uint32_t)(i%3)-1;} //20 +/-1ulp
    if(mode==10){for(int i=0;i<384;++i)l[i]=(i&1)?0x80000001u:0x00000001u;}
    if(mode==11){for(int i=0;i<384;++i)l[i]=(i&1)?0xff7fffffu:0x7f7fffffu;}
    if(mode==12){for(int i=0;i<384;++i){l[i]=0;b[(size_t)i*pitch]=(i&1)?0x80000000u:0u;}}
    if(mode==13){for(int i=0;i<384;++i){l[i]=0;b[(size_t)i*pitch]=0x7f800000u;}}
    if(mode==14){ // six high equal ranks straddling every native expert partition boundary
        static const int start[]={0,1,90,95,96,97,186,191,192,193,282,287,288,289,378,383};
        for(int i=0;i<384;++i){l[i]=0;b[(size_t)i*pitch]=0;}
        for(int j=0;j<6;++j)b[(size_t)((start[serial]+j)%384)*pitch]=fbits(100.f);
    }
    if(mode==15){for(int i=0;i<384;++i){const uint32_t u=(uint32_t)(serial*384+i)&65535u;
        l[i]=u<<16;b[(size_t)i*pitch]=((u*73u+17u)&65535u)<<16;}}
}
void one(API&api,hipStream_t stream,int rank,int mode,int serial,int norm,bool inverse,uint64_t&bytes){
    const int bp=(serial+mode)&1?3:1;
    std::vector<uint32_t>lg(401,0xa5a5a5a5u),bi(384*bp,0xa5a5a5a5u);inputs(mode,serial,lg,bi,bp);
    Buffer L(lg.size()*4),B(bi.size()*4);L.set(lg);B.set(bi);
    const std::array<int,6>sizes={13,13,13,105,103,13};
    std::array<Buffer*,6>old{},now{};
    for(int i=0;i<6;++i){old[i]=new Buffer(sizes[i]*4);now[i]=new Buffer(sizes[i]*4);}
    std::vector<RadParam>p={pi("M",1),pi("n_expert",384),pi("n_local",96),pi("top_k",6),pi("norm",norm),
        pf("route_scale",mode==10?0x1p-149f:mode==11?0x1.fffffep127f:mode==9?0x1p-126f:1.5f),
        ps("score","sqrtsoftplus"),ps("dtype","f32"),pi("expert_base",rank*96)};
    auto tensors=[&](const std::array<Buffer*,6>&o){std::vector<RadTensor>t={tensor(L.data(),RAD_F32,{1,384}),tensor(B.data(),RAD_F32,{384}),
        tensor(o[0]->data(),RAD_I32,{1,6}),tensor(o[1]->data(),RAD_F32,{1,6}),tensor(o[2]->data(),RAD_I32,{6}),
        tensor(o[3]->data(),RAD_I32,{97}),tensor(o[4]->data(),RAD_I32,{96}),inverse?tensor(o[5]->data(),RAD_I32,{6}):RadTensor{}};
        t[0].stride[0]=401;t[1].stride[0]=bp;t[2].stride[0]=13;t[3].stride[0]=13;return t;};
    auto ot=tensors(old),nt=tensors(now);std::vector<RadTensor>st(ot.begin(),ot.begin()+4);auto a=args(st,p,rank);
    need(api.score(&a,(RadStream)stream)==RAD_OK,"original scorer refused");
    std::vector<RadParam>sp={pi("M",1),pi("n_expert",96),pi("top_k",6),pi("expert_base",rank*96)};
    std::vector<RadTensor>sort={tensor(old[0]->data(),RAD_I32,{6}),ot[4],ot[5],ot[6]};if(inverse)sort.push_back(ot[7]);
    auto sa=args(sort,sp,rank);need(api.scatter[inverse](&sa,(RadStream)stream)==RAD_OK,"original scatter refused");
    auto na=args(nt,p,rank);need(api.fused(&na,(RadStream)stream)==RAD_OK,"fused normal launch refused");hc(hipStreamSynchronize(stream));
    for(int i=0;i<6;++i){auto x=old[i]->read(),y=now[i]->read();need(x==y,"ID/weight/sort/offset/count/inverse BYTE mismatch");bytes+=x.size();}
    L.read(true);B.read(true);
    for(int i=0;i<6;++i){delete old[i];delete now[i];}
}
void refusals(API&api,hipStream_t stream,int rank){
    Buffer L(384*4),B(384*4),I(6*4),W(6*4),S(6*4),O(97*4),C(96*4),V(6*4);
    std::vector<RadTensor>t={tensor(L.data(),RAD_F32,{1,384}),tensor(B.data(),RAD_F32,{384}),tensor(I.data(),RAD_I32,{1,6}),
        tensor(W.data(),RAD_F32,{1,6}),tensor(S.data(),RAD_I32,{6}),tensor(O.data(),RAD_I32,{97}),
        tensor(C.data(),RAD_I32,{96}),tensor(V.data(),RAD_I32,{6})};
    std::vector<RadParam>p={pi("M",1),pi("n_expert",384),pi("n_local",96),pi("top_k",6),pi("norm",1),
        pf("route_scale",1.5),ps("score","sqrtsoftplus"),ps("dtype","f32"),pi("expert_base",rank*96)};
    for(int mode=0;mode<12;++mode){auto tt=t;auto pp=p;auto a=args(tt,pp,rank);
        if(mode==0)seti(pp,"M",2);if(mode==1)seti(pp,"n_expert",383);if(mode==2)seti(pp,"top_k",5);
        if(mode==3)seti(pp,"n_local",95);if(mode==4)seti(pp,"expert_base",(rank*96+1)%384);
        if(mode==5)pp[6]=ps("score","sigmoid");if(mode==6)a.world_size=2;
        if(mode==7)tt[2].data=tt[0].data;if(mode==8)tt[6].data=tt[5].data;
        if(mode==9)tt[7].shape[0]=5;if(mode==10)tt[0].dtype=RAD_BF16;
        if(mode==11)pp[5]=pf("route_scale",0);
        need(api.fused(&a,(RadStream)stream)<0,"invalid/alias call accepted");hc(hipStreamSynchronize(stream));
        L.read(true);B.read(true);I.read(true);W.read(true);S.read(true);O.read(true);C.read(true);V.read(true);
    }
}
}
int main(int argc,char**argv){try{
    need(argc==3&&!std::strcmp(argv[1],"--inside-guarded-window"),"usage: CHECK --inside-guarded-window LIB");
    for(const char*k:{"HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES","CUDA_VISIBLE_DEVICES"}){
        const char*v=std::getenv(k);need(!v||!std::strcmp(v,"0,1,2,3"),"all four cards must be visible");}
    int n=0;hc(hipGetDeviceCount(&n));need(n==4,"exactly four cards required");API api(argv[2]);uint64_t bytes=0,cases=0;
    for(int rank=0;rank<4;++rank){hc(hipSetDevice(rank));hipStream_t stream;hc(hipStreamCreate(&stream));uint64_t count=0;
        for(int norm=0;norm<2;++norm)for(int inv=0;inv<2;++inv){
            for(int mode=0;mode<14;++mode){one(api,stream,rank,mode,0,norm,inv,bytes);++count;}
            for(int s=0;s<16;++s){one(api,stream,rank,14,s,norm,inv,bytes);++count;}
            for(int s=0;s<172;++s){one(api,stream,rank,15,s,norm,inv,bytes);++count;}
        }
        refusals(api,stream,rank);hc(hipStreamDestroy(stream));need(count==808,"case enumeration incomplete");cases+=count;
        std::printf("PASS ds41_scored_scatter rank=%d cases=%llu refusals=12 all_bf16_logits=65536 all_bf16_bias=65536 bad=0\n",rank,(unsigned long long)count);
    }
    need(cases==3232&&api.paths()==15,"missing actual normal dispatch rank coverage");
    std::printf("PASS ds41_scored_scatter_exact ranks=4 cases=3232 refusals=48 paths=15 bytes=%llu bad=0\n",(unsigned long long)bytes);return 0;
}catch(const std::exception&e){std::fprintf(stderr,"FAIL ds41_scored_scatter: %s\n",e.what());return 1;}}
