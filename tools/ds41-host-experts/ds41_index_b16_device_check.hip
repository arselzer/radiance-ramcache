/* Source-only oracle: run ONLY inside the stopped/restored production window.
 * Independent old f32 exports are the reference, including NaNs and WMMA.
 * Three FRESH processes: unpinned, exact, lcp4096 (env checked, not changed). */
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include "rad_abi.h"
#include "../../arch/common/rad_ds41_score_b16.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Launch=int(*)(const RadArgs*,RadStream);
using Scratch=int64_t(*)(const RadArgs*);
void hc(hipError_t e){if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
void need(bool b,const char*m){if(!b)throw std::runtime_error(m);}
RadParam pi(const char*k,long long v){RadParam p{};p.key=k;p.kind=RAD_P_INT;p.ival=v;return p;}
RadParam pf(const char*k,double v){RadParam p{};p.key=k;p.kind=RAD_P_F64;p.dval=v;return p;}
RadParam ps(const char*k,const char*v){RadParam p{};p.key=k;p.kind=RAD_P_STR;p.sval=v;return p;}
RadTensor tensor(void*p,uint32_t dt,std::initializer_list<int64_t> shape){
    RadTensor t{};t.data=p;t.dtype=dt;t.rank=(uint32_t)shape.size();int i=0;
    for(auto x:shape)t.shape[i++]=x;int64_t stride=1;
    for(int k=(int)t.rank-1;k>=0;--k){t.stride[k]=stride;stride*=t.shape[k];}return t;
}
struct Buffer {
    static constexpr size_t guard=64;
    uint8_t*base=nullptr;size_t bytes=0;std::vector<uint8_t> original;
    explicit Buffer(size_t n):bytes(n),original(n+2*guard,0xa5){hc(hipMalloc((void**)&base,original.size()));upload();}
    ~Buffer(){if(base)(void)hipFree(base);}
    Buffer(const Buffer&)=delete;Buffer&operator=(const Buffer&)=delete;
    void*data(){return base+guard;}
    void upload(){hc(hipMemcpy(base,original.data(),original.size(),hipMemcpyHostToDevice));}
    template<class T>void set(const std::vector<T>&v){need(v.size()*sizeof(T)==bytes,"input length");std::memcpy(original.data()+guard,v.data(),bytes);upload();}
    std::vector<uint8_t> get(bool unchanged=false){std::vector<uint8_t>x(original.size());hc(hipMemcpy(x.data(),base,x.size(),hipMemcpyDeviceToHost));
        need(!std::memcmp(x.data(),original.data(),guard)&&!std::memcmp(x.data()+guard+bytes,original.data()+guard+bytes,guard),"outer guard changed");
        if(unchanged)need(x==original,"input changed");return {x.begin()+guard,x.begin()+guard+bytes};}
};
struct API {
    void*h;Launch score[2],cand[2],top[2];Scratch scratch;uint32_t(*paths)();
    explicit API(const char*path){h=dlopen(path,RTLD_NOW|RTLD_LOCAL);need(h,"dlopen failed");
        score[0]=get<Launch>("r4d_rad_ds41_index_score");score[1]=get<Launch>("r4d_rad_ds41_index_score_b16");
        cand[0]=get<Launch>("r4d_rad_ds41_index_candidates");cand[1]=get<Launch>("r4d_rad_ds41_index_candidates_b16");
        top[0]=get<Launch>("r4d_rad_ds41_index_topk");top[1]=get<Launch>("r4d_rad_ds41_index_topk_b16");
        scratch=get<Scratch>("r4d_rad_ds41_index_topk_scratch");paths=get<uint32_t(*)()>("r4d_rad_ds41_index_b16_paths");}
    template<class F>F get(const char*n){auto f=(F)dlsym(h,n);need(f,"missing old/new export");return f;}
};
RadArgs args(std::vector<RadTensor>&t,std::vector<RadParam>&p,int rank){RadArgs a{};a.t=t.data();a.n_t=(int)t.size();a.p=p.data();a.n_p=(int)p.size();a.rank=rank;a.world_size=4;return a;}
uint16_t finite_value(size_t i){return uint16_t(0x3c00+(i*17)%1024)|((i%3)?0u:0x8000u);}
struct ScoreCase {int M,H,C,ratio,part,pad;bool all_bits;};
void score_case(API&api,hipStream_t stream,int rank,ScoreCase c,uint64_t&values){
    const int mq=c.part?c.part*c.M+3:c.M,qr=c.H*128+c.pad,nblk=c.C+3,S=2;
    const int pitch=c.C+7;
    std::vector<uint16_t>q((size_t)mq*qr,0x3f80),w((size_t)mq*c.H,0x3f80),k((size_t)nblk*128);
    for(size_t i=0;i<q.size();++i)q[i]=c.all_bits?uint16_t(i):finite_value(i);
    for(size_t i=0;i<k.size();++i)k[i]=c.all_bits?uint16_t(i):finite_value(i+37);
    if(!c.all_bits&&mq>7){q[0]=0x8000;q[(size_t)qr]=0x7f80;q[(size_t)qr*2]=0xff80;q[(size_t)qr*3]=0x7fc1;q[(size_t)qr*4]=0xffc3;}
    std::vector<int32_t>bt((size_t)S*(c.C+1)),pos(mq),cu={0,mq/2,mq};
    for(int s=0;s<S;++s)for(int j=0;j<=c.C;++j)bt[(size_t)s*(c.C+1)+j]=!c.all_bits&&j%13==4?-1:!c.all_bits&&j%17==6?nblk:j%c.C;
    for(int m=0;m<mq;++m)pos[m]=!c.all_bits&&mq>1&&m%11==0?-1:c.C*c.ratio-1-(!c.all_bits?m%3:0);
    Buffer Q(q.size()*2),W(w.size()*2),K(k.size()*2),B(bt.size()*4),P(pos.size()*4),U(cu.size()*4);
    Q.set(q);W.set(w);K.set(k);B.set(bt);P.set(pos);U.set(cu);
    Buffer F((size_t)c.M*pitch*4),V((size_t)c.M*pitch*2);
    std::vector<RadParam>p={pi("M",c.M),pi("heads",c.H),pi("dim",128),pi("ratio",c.ratio),pi("width",c.C),pf("wscale",.015625),ps("dtype","bf16"),pi("row_part",c.part)};
    std::vector<RadTensor>t={tensor(Q.data(),RAD_BF16,{mq,c.H*128}),tensor(W.data(),RAD_BF16,{mq,c.H}),
        tensor(K.data(),RAD_BF16,{nblk,1,c.ratio,128/c.ratio}),tensor(B.data(),RAD_I32,{S,c.C+1}),
        tensor(P.data(),RAD_I32,{mq}),tensor(U.data(),RAD_I32,{3}),tensor(F.data(),RAD_F32,{c.M,c.C})};
    t[0].stride[0]=qr;t[6].stride[0]=pitch;auto a=args(t,p,rank);
    need(api.score[0](&a,(RadStream)stream)==RAD_OK,"old score launch");
    t[6]=tensor(V.data(),RAD_BF16,{c.M,c.C});t[6].stride[0]=pitch;
    need(api.score[1](&a,(RadStream)stream)==RAD_OK,"b16 score launch");hc(hipStreamSynchronize(stream));
    auto f=F.get(),v=V.get();
    for(int m=0;m<c.M;++m)for(int j=0;j<pitch;++j){
        uint32_t old;uint16_t now;std::memcpy(&old,f.data()+((size_t)m*pitch+j)*4,4);std::memcpy(&now,v.data()+((size_t)m*pitch+j)*2,2);
        if(j<c.C){need(rad_ds41_score_b16::representable(old),"old score has nonzero low bits");need(old==rad_ds41_score_b16::expand_bits(now),"score BYTE mismatch");++values;}
        else need(old==0xa5a5a5a5u&&now==0xa5a5u,"pitched score padding changed");
    }
    // The aliases reject a cross-format call before writing any destination.
    need(api.score[0](&a,(RadStream)stream)==RAD_E_DTYPE,"old writer accepted shortened b16 plane");
    t[6]=tensor(F.data(),RAD_F32,{c.M,c.C});t[6].stride[0]=pitch;
    need(api.score[1](&a,(RadStream)stream)==RAD_E_DTYPE,"b16 writer accepted f32 plane");
    Q.get(true);W.get(true);K.get(true);B.get(true);P.get(true);U.get(true);
}
struct SelectCase {int M,C,mode;};
void select_case(API&api,hipStream_t stream,int rank,SelectCase c,uint64_t&ids){
    const int K=512,block=8,kb=2048,pitch=c.C+5,kw=((c.C+block-1)/block+31)/32;
    std::vector<uint16_t>b((size_t)c.M*pitch,0xa5a5);std::vector<uint32_t>f((size_t)c.M*pitch,0xa5a5a5a5);
    std::vector<int32_t>pos(c.M);
    for(int m=0;m<c.M;++m){pos[m]=c.C-1-(m%3);for(int j=0;j<c.C;++j){uint16_t x;
        if(c.mode==0)x=uint16_t(j);else if(c.mode==1)x=uint16_t((j%7==0)?0x7f80:(j%5==0)?0x8000:(j%3==0)?0x0000:0x3f80);
        else x=c.mode==2?uint16_t(0x7fc1):uint16_t(0xff80);
        b[(size_t)m*pitch+j]=x;f[(size_t)m*pitch+j]=uint32_t(x)<<16;}}
    Buffer F(f.size()*4),V(b.size()*2),P(pos.size()*4);F.set(f);V.set(b);P.set(pos);
    Buffer C0((size_t)c.M*kw*4),C1((size_t)c.M*kw*4);
    std::vector<RadParam>p={pi("M",c.M),pi("width",c.C),pi("block",block),pi("topk_blocks",kb),pi("ratio",1),pi("topk",K)};
    for(int v=0;v<2;++v){
        std::vector<RadTensor>t={tensor(v?V.data():F.data(),v?RAD_BF16:RAD_F32,{c.M,c.C}),tensor(P.data(),RAD_I32,{c.M}),tensor(v?C1.data():C0.data(),RAD_I32,{c.M,kw})};
        t[0].stride[0]=pitch;auto a=args(t,p,rank);need(api.cand[v](&a,(RadStream)stream)==RAD_OK,"candidate launch");
        t[0].data=F.data();t[0].dtype=v?RAD_F32:RAD_BF16;
        need(api.cand[v](&a,(RadStream)stream)==RAD_E_DTYPE,"candidate reader accepted wrong score format");
    }
    hc(hipStreamSynchronize(stream));need(C0.get()==C1.get(),"candidate keep BYTE mismatch");
    for(int use_keep=0;use_keep<2;++use_keep){
        Buffer O0((size_t)c.M*(K+3)*4),O1((size_t)c.M*(K+3)*4);
        for(int v=0;v<2;++v){std::vector<RadTensor>t={tensor(v?V.data():F.data(),v?RAD_BF16:RAD_F32,{c.M,c.C}),
            use_keep?tensor(v?C1.data():C0.data(),RAD_I32,{c.M,kw}):RadTensor{},tensor(P.data(),RAD_I32,{c.M}),tensor(v?O1.data():O0.data(),RAD_I32,{c.M,K})};
            t[0].stride[0]=pitch;t[3].stride[0]=K+3;auto a=args(t,p,rank);const int64_t need_scratch=api.scratch(&a);
            Buffer scratch((size_t)std::max<int64_t>(need_scratch,16));a.scratch=scratch.data();a.scratch_bytes=need_scratch;
            need(api.top[v](&a,(RadStream)stream)==RAD_OK,"topk launch");hc(hipStreamSynchronize(stream));scratch.get();
            t[0].data=F.data();t[0].dtype=v?RAD_F32:RAD_BF16;
            need(api.top[v](&a,(RadStream)stream)==RAD_E_DTYPE,"topk reader accepted wrong score format");
        }
        const auto o0=O0.get(),o1=O1.get();need(o0==o1,"selected IDs/padding BYTE mismatch");
        for(int m=0;m<c.M;++m)for(int j=K;j<K+3;++j){uint32_t u;std::memcpy(&u,o0.data()+((size_t)m*(K+3)+j)*4,4);need(u==0xa5a5a5a5u,"topk pitched padding changed");}
        ids+=(uint64_t)c.M*K;
    }
    F.get(true);V.get(true);P.get(true);
}
bool off(const char*n){const char*v=std::getenv(n);return !v||!*v||!std::strcmp(v,"0");}
} // namespace

int main(int argc,char**argv){try{
    if(argc!=5||std::strcmp(argv[1],"--inside-guarded-window"))throw std::runtime_error("usage: CHECK --inside-guarded-window LIB profile required-card-count");
    const std::string profile=argv[3];need(std::strcmp(argv[4],"4")==0,"exactly4 cards required");
    const char*lc="RADIANCE_EXPERIMENTAL_DS41_LCP",*xt="RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL";
    if(profile=="unpinned")need(off(lc)&&off(xt),"unpinned env mismatch");
    else if(profile=="exact")need(off(lc)&&std::getenv(xt)&&!std::strcmp(std::getenv(xt),"1"),"exact env mismatch");
    else if(profile=="lcp4096")need(std::getenv(lc)&&!std::strcmp(std::getenv(lc),"4096")&&off(xt),"LCP env mismatch");
    else throw std::runtime_error("unknown profile");
    need(off("RADIANCE_EXPERIMENTAL_DS41_KV_PACKED"),"packed writer not admitted by this oracle");
    int devices=0;hc(hipGetDeviceCount(&devices));need(devices==4,"exactly4 visible devices required");
    API api(argv[2]);
    const ScoreCase scores[]={{1,1,65,1,0,0,false},{1,2,65,2,0,0,false},{1,4,65,1,0,0,false},{1,8,65,2,0,0,false},
        {49,16,65,1,0,0,false},{49,17,65,2,0,0,false},{49,33,65,1,0,0,false},{49,16,65,1,0,2,false},
        {7,32,257,2,1,0,false},{512,1,512,1,0,0,true},{2048,32,17,1,0,0,false},{4096,32,9,2,0,0,false}};
    const SelectCase picks[]={{1,65536,0},{4,8191,1},{4,8192,1},{4,8193,1},{1,24576,1},{1,24577,1},
        {16,32769,1},{17,32769,1},{49,257,1},{1,513,2},{1,513,3},{1,32769,1}};
    uint64_t values=0,ids=0;int score_cases=0,select_cases=0;
    for(int rank=0;rank<4;++rank){hc(hipSetDevice(rank));hipStream_t stream;hc(hipStreamCreate(&stream));
        for(auto c:scores){score_case(api,stream,rank,c,values);++score_cases;}
        for(auto c:picks){select_case(api,stream,rank,c,ids);++select_cases;}
        hc(hipStreamDestroy(stream));std::printf("PASS index_b16 rank=%d profile=%s score_cases=12 select_cases=12\n",rank,profile.c_str());
    }
    const uint32_t required=profile=="unpinned"?1023u:975u; // HS1/2/4/8, selection; WMMA1/2 only unpinned
    const uint32_t paths=api.paths();need((paths&required)==required,"required actual dispatch coverage missing");
    need(profile=="unpinned"||(paths&48u)==0,"WMMA ran under scalar pins");
    std::printf("PASS index_b16 coverage: profile=%s ranks=4 score_cases=%d select_cases=%d all_score_bits=65536 all_input_bits=65536 max_M=4096 max_C=65536 score_values=%llu selected_ids=%llu paths=%u bad=0\n",
        profile.c_str(),score_cases,select_cases,(unsigned long long)values,(unsigned long long)ids,paths);return 0;
}catch(const std::exception&e){std::fprintf(stderr,"FAIL index_b16: %s\n",e.what());return 1;}}
