/* SOURCE ONLY: new aliases are NOT qualified by the old pointwise component receipt.
 * Execute only in the ordinary two-lock stopped/restored-QFN window. CHECK LIBR4D.so.
 * No timing/model/adoption authority. Compare every uint16/U8 byte including padding/guards.
 */
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "rad_plugin.h"
#include "../../arch/common/rad_ds41_fp4_runtime.h"
#include "../../arch/common/rad_ds41_kv_codec.h"
namespace f=rad_ds41_fp4_runtime;
#define HC(x) do{hipError_t e=(x);if(e!=hipSuccess){std::fprintf(stderr,"FAIL %s: %s\n",#x,hipGetErrorString(e));std::exit(2);}}while(0)
using Launch=int(*)(const RadArgs*,RadStream);
using NormCheck=int(*)(const RadArgs*,RadStream,int);
using NewNormCheck=int(*)(const RadArgs*,RadStream,int,int);
RadParam pi(const char*k,int64_t v){RadParam p{};p.key=k;p.kind=RAD_P_INT;p.ival=v;return p;}
RadParam ps(const char*k,const char*v){RadParam p{};p.key=k;p.kind=RAD_P_STR;p.sval=v;return p;}
RadParam pf(const char*k,double v){RadParam p{};p.key=k;p.kind=RAD_P_F64;p.dval=v;return p;}
RadTensor tensor(void*p,uint32_t dt,std::initializer_list<int64_t> shape){RadTensor t{};t.data=p;t.dtype=dt;t.rank=shape.size();
    int i=0;for(auto n:shape)t.shape[i++]=n;rad_tensor_pack(&t);return t;}
int call(Launch fn,std::vector<RadTensor>&t,std::vector<RadParam>&p,hipStream_t stream){RadArgs a{};a.t=t.data();a.n_t=t.size();a.p=p.data();a.n_p=p.size();a.rank=0;a.world_size=4;return fn(&a,(RadStream)stream);}
struct Bytes {
    std::vector<uint8_t> seed,old,now;void* a=nullptr;void*b=nullptr;
    explicit Bytes(size_t n):seed(n,0xA5),old(n),now(n){HC(hipMalloc(&a,n));HC(hipMalloc(&b,n));}
    ~Bytes(){HC(hipFree(a));HC(hipFree(b));}
    void reset(hipStream_t s){HC(hipMemcpyAsync(a,seed.data(),seed.size(),hipMemcpyHostToDevice,s));HC(hipMemcpyAsync(b,seed.data(),seed.size(),hipMemcpyHostToDevice,s));}
    void read(hipStream_t s){HC(hipMemcpyAsync(old.data(),a,old.size(),hipMemcpyDeviceToHost,s));HC(hipMemcpyAsync(now.data(),b,now.size(),hipMemcpyDeviceToHost,s));}
};
uint16_t input(int r,int c,int n,int profile,int replay){
    const uint16_t edge[]={0,0x8000,1,0x8001,0x7F,0x807F,0x7F7F,0xFF7F,0x7F80,0xFF80,0x7F81,0xFFFF};
    const uint16_t tie[]={0x3E80,0x3F40,0x3FA0,0x3FE0,0x4020,0x4060,0x40A0,0x40C0};
    if(profile==0)return uint16_t((r*n+c)+replay*193);
    if(profile==1)return edge[(r+c+replay)%12];
    if(profile==2)return uint16_t(tie[(c+r)%8]+((c/8+replay)%3)-1)^(c&1?0x8000:0);
    return uint16_t(0x3F00+((r*73+c*17+replay*31)%512))^(c&1?0x8000:0);
}
struct Exports {
    Launch old[6]{},now[6]{},store=nullptr;NormCheck norm[2]{};NewNormCheck newer=nullptr;
    explicit Exports(void*lib){
        const char* names[]={"quant_roundtrip","quant_roundtrip_kvpack","rmsnorm_rope_roundtrip","rmsnorm_rope_roundtrip_kvpack","rope_rows_roundtrip","rope_rows_roundtrip_kvpack"};
        for(int k=0;k<6;++k){const std::string oldname=std::string("r4d_rad_")+names[k],newname=std::string("r4d_rad_")+f::kNames[k];
            old[k]=(Launch)dlsym(lib,oldname.c_str());now[k]=(Launch)dlsym(lib,newname.c_str());
            if(!old[k]||!now[k]){std::fprintf(stderr,"FAIL missing exact original/new alias %d\n",k);std::exit(2);}}
        norm[0]=(NormCheck)dlsym(lib,"r4d_rad_rmsnorm_rope_roundtrip_n128_check");
        norm[1]=(NormCheck)dlsym(lib,"r4d_rad_rmsnorm_rope_roundtrip_kvpack_n128_check");
        newer=(NewNormCheck)dlsym(lib,"r4d_rad_rmsnorm_rope_roundtrip_fp4_classify_n128_check");
        store=(Launch)dlsym(lib,"r4d_rad_entry_store_kvpack");
        if(!norm[0]||!norm[1]||!newer||!store){std::fprintf(stderr,"FAIL missing forced norm/store oracle exports\n");std::exit(2);}
    }
};
struct Totals {uint64_t cases=0,fast=0,fallback=0,capture=0,guards=0,bad=0,stores=0,replays=0,refusals=0;
    std::array<uint64_t,f::kSlotCount> op_fast{},op_fallback{},op_capture{};};
bool one(const Exports&e,int family,int n,int M,int profile,bool capture,bool audit,int gain,int form,Totals&sum,bool native=true){
    constexpr int guard=32;const int pitch=n+16,B=n==512?16:32,kind=n==512?1:2;
    const int op=2*family+int(capture);const int64_t rb=capture?rad_ds41_kv_codec::row_bytes(kind,n):0,cp=rb+7;
    Bytes x(size_t(2*guard+M*pitch)*2),y(x.seed.size()),cap(size_t(2*guard+M*cp));
    Bytes cache(size_t(2*guard+(M+3)*rb));
    const int csp=67,ntab=19;std::vector<float> cs(ntab*csp,0.25f),wf(n,1.0f);
    std::vector<uint16_t> wh(n,0x3F80);std::vector<int32_t> pos(M*2,-77),page(M);
    for(int r=0;r<ntab;++r)for(int c=0;c<64;++c)cs[r*csp+c]=(c%4==0?1.0f:c%4==1?-0.5f:c%4==2?0.0f:-0.0f);
    for(int r=0;r<M;++r){pos[r*2]=(r%5==0?-1:r%5==1?ntab:r%ntab);page[r]=M-r+1;}
    if(M)page[0]=-1;
    void *dcs=nullptr,*dw=nullptr,*dp=nullptr,*pg=nullptr,*counter=nullptr;
    HC(hipMalloc(&dcs,cs.size()*4));HC(hipMalloc(&dw,n*4));HC(hipMalloc(&dp,pos.size()*4));
    HC(hipMalloc(&pg,page.size()*4));HC(hipMalloc(&counter,(f::kWords+2)*8));
    hipStream_t stream;HC(hipStreamCreate(&stream));
    HC(hipMemcpyAsync(dcs,cs.data(),cs.size()*4,hipMemcpyHostToDevice,stream));
    HC(hipMemcpyAsync(dw,gain?(const void*)wh.data():(const void*)wf.data(),n*(gain?2:4),hipMemcpyHostToDevice,stream));
    HC(hipMemcpyAsync(dp,pos.data(),pos.size()*4,hipMemcpyHostToDevice,stream));
    HC(hipMemcpyAsync(pg,page.data(),page.size()*4,hipMemcpyHostToDevice,stream));
    auto matrix=[&](void*p){auto t=tensor((uint16_t*)p+guard,RAD_BF16,{M,n});t.stride[0]=pitch;return t;};
    auto packed=[&](void*p){auto t=tensor((uint8_t*)p+guard,RAD_U8,{M,rb});t.stride[0]=cp;return t;};
    auto cst=tensor(dcs,RAD_F32,{ntab,64});cst.stride[0]=csp;
    auto pt=tensor(dp,RAD_I32,{M});pt.stride[0]=2;
    std::vector<RadTensor> old,newer;
    if(family==0){old={matrix(x.a)};newer={matrix(x.b)};}
    if(family==1){old={matrix(x.a),tensor(dw,gain?RAD_BF16:RAD_F32,{n}),matrix(y.a),cst,pt};
                 newer={matrix(x.b),old[1],matrix(y.b),cst,pt};}
    if(family==2){old={matrix(x.a),cst,pt};newer={matrix(x.b),cst,pt};}
    if(capture){old.push_back(packed(cap.a));newer.push_back(packed(cap.b));}
    newer.push_back(audit?tensor((uint64_t*)counter+1,RAD_I64,{f::kWords}):RadTensor{});
    std::vector<RadParam> p={pi("M",M),pi("n",n),pi("block",B),ps("format","fp4_e2m1"),
        ps("scale",B==16?"e4m3":"ue8m0"),ps("dtype","bf16"),pi("kv_format",kind),pf("eps",1e-6),
        pi("n_head",n==4096?32:1),pi("head_dim",n==4096?128:n),pi("rot",64),pi("off",(n==4096?128:n)-64),
        ps("mode","gptj"),pi("inverse",0),pf("wadd",0),pf("mult",1),pi("block_size",1)};
    auto invoke=[&]() -> int {
        if(family==1&&form>=0){RadArgs a{};a.t=old.data();a.n_t=old.size();a.p=p.data();a.n_p=p.size();
            int rc=e.norm[capture](&a,(RadStream)stream,form);if(rc<0)return rc;a.t=newer.data();a.n_t=newer.size();
            return e.newer(&a,(RadStream)stream,form,capture);}
        const int rc=call(e.old[op],old,p,stream);return rc<0?rc:call(e.now[op],newer,p,stream);
    };
    bool okay=true;hipGraph_t graph=nullptr;hipGraphExec_t exec=nullptr;
    // Capture ON/OFF kernels with changing input payload; persistent counters are not graph-local.
    HC(hipStreamSynchronize(stream));HC(hipStreamBeginCapture(stream,hipStreamCaptureModeGlobal));
    const int captured=invoke();HC(hipStreamEndCapture(stream,&graph));
    if(captured<0){std::fprintf(stderr,"FAIL capture alias rc=%d op=%s\n",captured,f::kNames[op]);okay=false;}
    if(okay)HC(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    uint64_t previous=0;
    for(int replay=0;replay<2&&okay;++replay){
        for(int r=0;r<M;++r)for(int c=0;c<n;++c){const uint16_t v=input(r,c,n,profile,replay);
            std::memcpy(x.seed.data()+size_t(guard+r*pitch+c)*2,&v,2);}
        x.reset(stream);y.reset(stream);cap.reset(stream);cache.reset(stream);
        std::array<uint64_t,f::kWords+2> counts{};counts.front()=counts.back()=0xA55AA55AA55AA55Aull;
        if(replay==0)HC(hipMemcpyAsync(counter,counts.data(),sizeof counts,hipMemcpyHostToDevice,stream));
        HC(hipGraphLaunch(exec,stream));x.read(stream);y.read(stream);cap.read(stream);
        HC(hipMemcpyAsync(counts.data(),counter,sizeof counts,hipMemcpyDeviceToHost,stream));HC(hipStreamSynchronize(stream));
        size_t bad=x.old!=x.now,guards=0;
        if(family==1)bad+=y.old!=y.now;
        if(capture)bad+=cap.old!=cap.now;
        const auto&out=family==1?y:x;
        for(size_t z=0;z<out.old.size()/2;++z)if(z<size_t(guard)||z>=out.old.size()/2-guard||(z-guard)%pitch>=size_t(n))
            guards+=out.old[2*z]!=out.seed[2*z]||out.old[2*z+1]!=out.seed[2*z+1]||out.now[2*z]!=out.seed[2*z]||out.now[2*z+1]!=out.seed[2*z+1];
        if(family==1)bad+=x.old!=x.seed||x.now!=x.seed;
        if(capture)for(size_t z=0;z<cap.old.size();++z)if(z<size_t(guard)||z>=cap.old.size()-guard||(z-guard)%cp>=size_t(rb))
            guards+=cap.old[z]!=cap.seed[z]||cap.now[z]!=cap.seed[z];
        guards+=counts.front()!=0xA55AA55AA55AA55Aull||counts.back()!=counts.front();
        const int at=f::slot((f::Op)op,n);
        const uint64_t* row=counts.data()+1+(at>=0?at*f::FieldCount:0);
        if(audit&&native){
            for(int z=0;z<f::kSlotCount;++z){const auto* q=counts.data()+1+z*f::FieldCount;
                bad+=!f::valid_row(q,(f::kSlots[z].op&1)!=0);
                if(z!=at)for(int j=0;j<f::FieldCount;++j)bad+=q[j]!=0;}
            bad+=!f::valid_row(row,capture);bad+=row[f::Fast]+row[f::Fallback]!=uint64_t(replay+1)*M*n;
            bad+=row[f::Fast]+row[f::Fallback]<=previous;previous=row[f::Fast]+row[f::Fallback];
            if(replay==1){sum.fast+=row[f::Fast];sum.fallback+=row[f::Fallback];sum.capture+=row[f::CapturedFast];
                sum.op_fast[at]+=row[f::Fast];sum.op_fallback[at]+=row[f::Fallback];sum.op_capture[at]+=row[f::CapturedFast];}}
        else for(int z=0;z<f::kWords;++z)bad+=counts[1+z]!=0;
        if(capture){auto aout=family==1?matrix(y.a):matrix(x.a),bout=family==1?matrix(y.b):matrix(x.b);
            std::vector<RadTensor> a={aout,packed(cap.a),tensor(pg,RAD_I32,{M}),tensor((uint8_t*)cache.a+guard,RAD_U8,{M+3,rb})};
            std::vector<RadTensor> b={bout,packed(cap.b),a[2],tensor((uint8_t*)cache.b+guard,RAD_U8,{M+3,rb})};
            int rc=call(e.store,a,p,stream);if(rc>=0)rc=call(e.store,b,p,stream);cache.read(stream);HC(hipStreamSynchronize(stream));
            bad+=rc<0||cache.old!=cache.now;for(int z=0;z<guard;++z)guards+=cache.old[z]!=0xA5||cache.now[z]!=0xA5||cache.old[cache.old.size()-1-z]!=0xA5||cache.now[cache.now.size()-1-z]!=0xA5;
            ++sum.stores;
        }
        sum.bad+=bad;sum.guards+=guards;++sum.replays;okay=bad==0&&guards==0;
    }
    if(exec)HC(hipGraphExecDestroy(exec));if(graph)HC(hipGraphDestroy(graph));HC(hipStreamDestroy(stream));
    for(void*q:{dcs,dw,dp,pg,counter})HC(hipFree(q));
    ++sum.cases;std::printf("%s fp4-runtime-alias op=%s M=%d n=%d profile=%d audit=%d gain=%d form=%d graph_replays=2 bad=%llu guards=%llu\n",
        okay?"PASS":"FAIL",f::kNames[op],M,n,profile,int(audit),gain,form,(unsigned long long)sum.bad,(unsigned long long)sum.guards);
    return okay;
}
bool negatives(const Exports& e,Totals& sum) {
    constexpr int n=512,guard=32;Bytes x(size_t(n+2*guard)*2);
    for(int i=0;i<n;++i){const uint16_t z=0;std::memcpy(x.seed.data()+2*(guard+i),&z,2);}
    void* counter=nullptr;HC(hipMalloc(&counter,(f::kWords+2)*8));hipStream_t s;HC(hipStreamCreate(&s));
    auto t=tensor((uint16_t*)x.b+guard,RAD_BF16,{1,n});
    std::vector<RadTensor> ts={t,tensor((uint64_t*)counter+1,RAD_I64,{f::kWords})};
    std::vector<RadParam> p={pi("M",1),pi("n",n),pi("block",16),ps("format","fp4_e2m1"),ps("scale","e4m3"),ps("dtype","bf16")};
    bool okay=true;
    for(int mode=0;mode<4;++mode){
        x.reset(s);HC(hipStreamSynchronize(s));const auto saved=ts[1];
        if(mode==0)ts[1].dtype=RAD_I32;
        if(mode==1)ts[1].shape[0]--;
        if(mode==2)ts[1].stride[0]=2;
        if(mode==3)p[4]=ps("scale","unknown");
        const int rc=call(e.now[f::Quant],ts,p,s);x.read(s);HC(hipStreamSynchronize(s));
        okay=okay&&rc<0&&x.old==x.seed&&x.now==x.seed;
        ts[1]=saved;p[4]=ps("scale","e4m3");++sum.refusals;
    }
    x.reset(s);std::array<uint64_t,f::kWords+2> counts{};
    counts.front()=counts.back()=0xA55AA55AA55AA55Aull;const int offset=1+f::slot(f::Quant,512)*f::FieldCount;counts[offset+f::Fast]=UINT64_MAX;
    HC(hipMemcpyAsync(counter,counts.data(),sizeof counts,hipMemcpyHostToDevice,s));
    const int rc=call(e.now[f::Quant],ts,p,s);HC(hipMemcpyAsync(counts.data(),counter,sizeof counts,hipMemcpyDeviceToHost,s));
    x.read(s);HC(hipStreamSynchronize(s));
    okay=okay&&rc>=0&&counts[offset+f::Overflow]==1&&counts[offset+f::Fast]==UINT64_MAX&&
        !f::valid_row(counts.data()+offset,false)&&counts.front()==0xA55AA55AA55AA55Aull&&counts.back()==counts.front()&&x.now==x.seed;
    HC(hipStreamDestroy(s));HC(hipFree(counter));++sum.refusals;
    std::printf("%s fp4-runtime-refusals malformed_audit=3 unknown_scale=1 checked_counter_overflow=1 unchanged_bytes=1\n",okay?"PASS":"FAIL");
    return okay;
}
int main(int argc,char**argv){
    if(argc!=2){std::fprintf(stderr,"usage: %s LIBR4D.so\n",argv[0]);return 2;}
    int device=0;HC(hipGetDevice(&device));char pci[64]{};HC(hipDeviceGetPCIBusId(pci,sizeof pci,device));
    std::printf("FP4_RUNTIME_CARD device=%d pci=%s\n",device,pci);
    void*lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!lib){std::fprintf(stderr,"FAIL %s\n",dlerror());return 2;}
    Exports e(lib);Totals sum;bool okay=true;
    const int heights[]={1,7,8,9,17,33,127,128,129,257,512,1024,2048,4096};
    for(int M:heights)for(int profile=0;profile<4;++profile)for(bool audit:{false,true}){
        for(int n:{128,512,4096})for(int family:{0,2})for(bool cap:{false,true}){
            if(cap&&n==4096)continue;
            if(!one(e,family,n,M,profile,cap,audit,0,-1,sum)){okay=false;goto done;}
        }
        for(bool cap:{false,true})for(int gain:{0,1})for(int form:{-1,0,1}){
            if(gain&&form==1)continue;
            if(!one(e,1,128,M,profile,cap,audit,gain,form,sum)){okay=false;goto done;}
        }
    }
    // Direct alias API unknown geometry must use the whole original body; architecture keeps old
    // op names for this case. Zero audit work is deliberately NOT model-use authority.
    okay=one(e,0,256,7,1,false,true,0,-1,sum,false)&&okay;
    okay=negatives(e,sum)&&okay;
done:
    dlclose(lib);okay=okay&&sum.cases==2241&&sum.replays==4482&&sum.stores==2016&&sum.refusals==5&&sum.fast>0&&sum.fallback>0&&sum.capture>0&&!sum.bad&&!sum.guards;
    for(int k=0;k<f::kSlotCount;++k){const bool pass=sum.op_fast[k]>0&&sum.op_fallback[k]>0&&sum.op_capture[k]==((f::kSlots[k].op&1)?sum.op_fast[k]:0);
        okay=pass&&okay;std::printf("%s fp4-runtime-use op=%s n=%d fast=%llu fallback=%llu capture_fast=%llu scope=standalone_card\n",pass?"PASS":"FAIL",f::kNames[f::kSlots[k].op],f::kSlots[k].width,
            (unsigned long long)sum.op_fast[k],(unsigned long long)sum.op_fallback[k],(unsigned long long)sum.op_capture[k]);}
    std::printf("%s ds41_fp4_runtime_alias_exact cases=%llu replay_calls=%llu stores=%llu fast=%llu fallback=%llu capture_fast=%llu bad=%llu guards=%llu serving_authority=0\n",
        okay?"PASS":"FAIL",(unsigned long long)sum.cases,(unsigned long long)sum.replays,(unsigned long long)sum.stores,
        (unsigned long long)sum.fast,(unsigned long long)sum.fallback,(unsigned long long)sum.capture,(unsigned long long)sum.bad,(unsigned long long)sum.guards);
    return okay?0:1;
}
