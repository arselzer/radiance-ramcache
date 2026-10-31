/* SOURCE ONLY. Execute inside a stopped-QFN/two-lock window, separately for unpinned,
 * EXACT_TALL=1 and LCP2048+N128=1. Always run the independent original-producer/domain
 * oracle first. Never treat float ==, tolerated error, a trap or a partial run as PASS. */
#define main ds41_window_domain_main
#include "ds41_window_codec_device_check.hip"
#undef main
#include <algorithm>

namespace {
using Check=int(*)(const RadArgs*,RadStream,int);
struct WDev {
    void* p=nullptr;
    template<class T> explicit WDev(const std::vector<T>& v) {
        HC(hipMalloc(&p,v.size()*sizeof(T)));HC(hipMemcpy(p,v.data(),v.size()*sizeof(T),hipMemcpyHostToDevice));
    }
    ~WDev(){if(p)HC(hipFree(p));}
    WDev(const WDev&)=delete;
    template<class T> std::vector<T> get(size_t n) const {
        std::vector<T> v(n);HC(hipMemcpy(v.data(),p,n*sizeof(T),hipMemcpyDeviceToHost));return v;
    }
};
RadTensor wt(void* p,uint32_t dtype,std::initializer_list<int64_t> dims) {
    RadTensor t{};t.data=p;t.dtype=dtype;t.rank=(uint32_t)dims.size();
    int i=0;for(auto d:dims)t.shape[i++]=d;
    int64_t s=1;for(i=(int)t.rank-1;i>=0;--i){t.stride[i]=s;s*=t.shape[i];}return t;
}
RadParam wf(const char* k,double v){RadParam p{};p.key=k;p.kind=RAD_P_F64;p.dval=v;return p;}
int wcall(Launch f,std::vector<RadTensor>&t,std::vector<RadParam>&p) {
    RadArgs a{};a.t=t.data();a.n_t=(int)t.size();a.p=p.data();a.n_p=(int)p.size();a.world_size=1;
    return f(&a,nullptr);
}
int wcheck(Check f,std::vector<RadTensor>&t,std::vector<RadParam>&p,int form) {
    RadArgs a{};a.t=t.data();a.n_t=(int)t.size();a.p=p.data();a.n_p=(int)p.size();a.world_size=1;
    return f(&a,nullptr,form);
}
/* All original BF16 input bit patterns occur at M4096; four independently mixed profiles
 * exercise signedzero/subnormals/NaNs/Inf alongside finite values in the same group. */
uint16_t win_input(int r,int c,int profile) {
    const uint16_t x=uint16_t(r*16+c/32);
    const uint16_t edge[]={0,0x8000,1,0x8001,0x007f,0x807f,0x7f7f,0xff7f,0x7f80,0xff80,0x7fc1,0xffff};
    if(!(c%32)||!profile)return x;
    if(profile==1)return edge[c%12];
    if(profile==2)return uint16_t(x^uint16_t(c*0x631));
    return edge[(c+c/32)%6+6];
}
bool window_producers(void* lib) {
    const char* names0[]={"r4d_rad_quant_roundtrip","r4d_rad_rmsnorm_rope_roundtrip","r4d_rad_rope_rows_roundtrip"};
    const char* names1[]={"r4d_rad_quant_roundtrip_windowpack","r4d_rad_rmsnorm_rope_roundtrip_windowpack","r4d_rad_rope_rows_roundtrip_windowpack"};
    Launch f0[3]{},f1[3]{};
    for(int i=0;i<3;++i){f0[i]=(Launch)dlsym(lib,names0[i]);f1[i]=(Launch)dlsym(lib,names1[i]);if(!f0[i]||!f1[i])return false;}
    const auto st0=(Launch)dlsym(lib,"r4d_rad_kv_store");
    const auto st1=(Launch)dlsym(lib,"r4d_rad_kv_store_windowpack");if(!st0||!st1)return false;
    size_t bad=0,raw=0,cases=0;unsigned forms=0;int maxm=0;
    for(int form=0;form<3;++form)for(int M:{1,7,33,49,257,2048,4096})for(int profile=0;profile<4;++profile) {
        constexpr int n=512,xp=528,cp=544,kp=8464;const int blocks=(M+15)/16+2;
        std::vector<uint16_t>x(size_t(M)*xp+16,0x5aa5),y=x;
        for(int r=0;r<M;++r)for(int c=0;c<n;++c)x[size_t(r)*xp+c]=win_input(r,c,profile);
        std::vector<float>w(n),cs(size_t(M)*64);std::vector<int32_t>pos(M),slots(M);
        for(int c=0;c<n;++c)w[c]=c%7?1.f:-1.f;
        for(int r=0;r<M;++r){pos[r]=r%5?-1:r;slots[r]=r%19?r+16:-1;
            for(int c=0;c<32;++c){cs[size_t(r)*64+c]=.75f;cs[size_t(r)*64+32+c]=.5f;}}
        std::vector<uint8_t>capture(size_t(M)*cp+16,0xa5),cache(size_t(blocks)*kp+16,0);
        for(int b=0;b<blocks;++b)std::fill(cache.begin()+size_t(b)*kp+8448,cache.begin()+size_t(b+1)*kp,0xa5);
        std::fill(cache.end()-16,cache.end(),0xa5);
        std::vector<uint16_t>plain(size_t(blocks)*16*512,0);
        WDev dx0(x),dx1(x),dy0(y),dy1(y),dw(w),dt(cs),dp(pos),ds(slots),dc(capture),dk(cache),db(plain);
        auto xt=[&](void*p){auto t=wt(p,RAD_BF16,{M,n});t.stride[0]=xp;return t;};
        auto ct=wt(dc.p,RAD_U8,{M,528});ct.stride[0]=cp;
        std::vector<RadParam>p={pi("M",M),pi("n",n),pi("n_head",1),pi("head_dim",n),pi("rot",64),
            pi("off",448),pi("inverse",0),ps("mode","gptj"),pi("block",32),ps("format","fp8_e4m3"),
            ps("scale","ue8m0"),ps("dtype","bf16"),wf("eps",1e-6)};
        std::vector<RadTensor>a,b;
        if(form==0){a={xt(dx0.p)};b={xt(dx1.p),ct};}
        else if(form==1){a={xt(dx0.p),wt(dw.p,RAD_F32,{n}),xt(dy0.p),wt(dt.p,RAD_F32,{M,64}),wt(dp.p,RAD_I32,{M})};
            b=a;b[0]=xt(dx1.p);b[2]=xt(dy1.p);b.push_back(ct);}
        else {a={xt(dx0.p),wt(dt.p,RAD_F32,{M,64}),wt(dp.p,RAD_I32,{M})};b=a;b[0]=xt(dx1.p);b.push_back(ct);}
        int rc=wcall(f0[form],a,p);p.push_back(pi("kv_format",3));if(rc>=0)rc=wcall(f1[form],b,p);
        HC(hipDeviceSynchronize());const auto oldbits=form==1?dy0.get<uint16_t>(y.size()):dx0.get<uint16_t>(x.size());
        const auto newbits=form==1?dy1.get<uint16_t>(y.size()):dx1.get<uint16_t>(x.size());
        const auto cap=dc.get<uint8_t>(capture.size());bad+=rc<0||oldbits!=newbits;
        size_t localraw=0;uint8_t payload[32],scale;
        for(int r=0;r<M;++r)for(int g=0;g<16;++g)
            localraw+=wc::pack_captured_group(cap[size_t(r)*cp+512+g],cap.data()+size_t(r)*cp+g*32,
                newbits.data()+size_t(r)*xp+g*32,payload,&scale)!=wc::PackResult::Exact;
        raw+=localraw;
        // Prove capture bytes equal an independent transcription from the PRE-quant producer row
        // for standalone quant. Fused closure also checks unchanged output and store decode.
        if(form==0){WDev pre(x),expected(x),refcap(capture);
            capture_source<<<(M*16+127)/128,128>>>((const uint16_t*)pre.p,xp,(uint8_t*)refcap.p,(uint16_t*)expected.p,M,n,cp);
            HC(hipDeviceSynchronize());bad+=cap!=refcap.get<uint8_t>(capture.size());bad+=oldbits!=expected.get<uint16_t>(x.size());}
        for(size_t i=0;i<cap.size();++i)if(i>=size_t(M)*cp||i%cp>=528)bad+=cap[i]!=0xa5;
        if(!localraw&&rc>=0) {
            void* actual=form==1?dy1.p:dx1.p;auto kc=wt(dk.p,RAD_U8,{blocks,8448});kc.stride[0]=kp;
            std::vector<RadParam>sp={pi("M",M),pi("n",512),pi("block_size",16),pi("kv_format",3),ps("dtype","bf16")};
            std::vector<RadTensor>s1={xt(actual),ct,wt(ds.p,RAD_I32,{M}),kc};rc=wcall(st1,s1,sp);
            auto k=wt(actual,RAD_BF16,{M,1,256});k.stride[0]=xp;k.stride[1]=256;
            auto v=k;v.data=(uint16_t*)actual+256;
            std::vector<RadTensor>s0={k,v,wt(ds.p,RAD_I32,{M}),wt(db.p,RAD_BF16,{blocks,1,16,512})};
            std::vector<RadParam>bp={pi("M",M),pi("head_dim",256),pi("n_head_kv",1),pi("block_size",16),ps("kv_dtype","bf16")};
            if(rc>=0)rc=wcall(st0,s0,bp);HC(hipDeviceSynchronize());bad+=rc<0;
            const auto pk=dk.get<uint8_t>(cache.size());const auto bf=db.get<uint16_t>(plain.size());uint16_t decoded[512];
            for(int page=0;page<blocks;++page){for(int row=0;row<16;++row){
                bad+=!wc::decode_row(pk.data()+size_t(page)*kp+row*528,512,decoded);
                for(int c=0;c<512;++c)bad+=decoded[c]!=bf[(size_t(page)*16+row)*512+c];}
                for(int i=8448;i<kp;++i)bad+=pk[size_t(page)*kp+i]!=0xa5;}
            for(size_t i=pk.size()-16;i<pk.size();++i)bad+=pk[i]!=0xa5;
        }
        ++cases;forms|=1u<<form;maxm=std::max(maxm,M);
    }
    const bool pass=!bad&&!raw&&cases==84&&forms==7&&maxm==4096;
    std::printf("%s window packed producers cases=%zu raw=%zu byte_errors=%zu forms=%u max_M=%d\n",pass?"PASS":"FAIL",cases,raw,bad,forms,maxm);
    return pass;
}

/* Self-contained packed fixtures include all E4M3 codes, every finite scale, zeroed
 * pages, literal signed-zero/NaN pairs, missing/remapped pages and mixed window/compressed tiles.
 * Their host-decoded BF16 bytes are the independent input to the OLD reader form. */
struct WindowRows {
    int blocks;std::vector<uint8_t> packed;std::vector<uint16_t> plain;
    WindowRows(int nb,int profile):blocks(nb),packed(size_t(nb)*8448),plain(size_t(nb)*16*512) {
        for(int r=1;r<nb*16;++r){auto*row=packed.data()+size_t(r)*528;
            for(int g=0;g<16;++g){uint8_t* p=row+g*32;row[512+g]=uint8_t(105+(r*11+g*17)%143);
                for(int i=0;i<32;++i)p[i]=uint8_t(r*13+g*7+i);
                if(profile==0)for(int i=0;i<32;++i)if((p[i]&127)==127)p[i]&=0xfe;
                if(profile==1&&(r+g)%7==0){std::fill(p,p+32,0);const uint16_t x=(r&1)?0x8000:0x7fc1,y=(r&1)?0:0xffc2;
                    p[0]=uint8_t(x);p[1]=uint8_t(x>>8);p[2]=uint8_t(y);p[3]=uint8_t(y>>8);
                    p[4]=0xa5;p[5]=0x5a;p[6]=0x99;p[7]=0x66;row[512+g]=0;}}
            if(!wc::decode_row(row,512,plain.data()+size_t(r)*512))std::exit(2);}
    }
};
bool window_readers(void*lib) {
    const auto ref=(Check)dlsym(lib,"r4d_rad_ds41_attn_sparse_sink_kvpack_check");
    const auto alt=(Check)dlsym(lib,"r4d_rad_ds41_attn_sparse_sink_windowpack_check");
    const auto normal=(Launch)dlsym(lib,"r4d_rad_ds41_attn_sparse_sink_windowpack");
    const auto oldplain=(Launch)dlsym(lib,"r4d_rad_ds41_attn_sparse_sink");
    const auto oldpacked=(Launch)dlsym(lib,"r4d_rad_ds41_attn_sparse_sink_kvpack");
    if(!ref||!alt||!normal||!oldplain||!oldpacked)return false;
    size_t bad=0,cases=0;unsigned mask=0;
    constexpr int H=16,K=40,nb=20,blocks=80;
    for(int profile:{0,1})for(int ratio:{0,1,2})for(int M:{1,4,16,49})for(int win:{3,16,128}) {
        WindowRows rows(nb,profile);WDev w0(rows.plain),w1(rows.packed);
        // Compressed entries deliberately use zeroed self-contained FP4 pages here; the existing
        // 504-case FP4 oracle independently covers their normal/escape/fast decoding.
        std::vector<uint8_t>cp(size_t(blocks)*288,0);WDev comp(cp);
        std::vector<uint16_t>q(size_t(M)*H*512),init(q.size()+16,0x5aa5);
        for(size_t i=0;i<q.size();++i)q[i]=uint16_t(0x3b00+i*31%256);
        std::vector<int32_t>wb(nb),cb(blocks),sel(size_t(M)*K),pos(M),used={320},cu={0,M};std::vector<float>sink(H,-1.25f);
        for(int i=0;i<nb;++i)wb[i]=i%13==0?-1:(i*3)%nb;
        for(int i=0;i<blocks;++i)cb[i]=i%11==0?-1:(i*7)%blocks;
        for(int r=0;r<M;++r){pos[r]=272+r;for(int i=0;i<K;++i)sel[r*K+i]=i%9==0?-1:(i*3+r)%blocks;}
        WDev dq(q),db(wb),dt(cb),ds(sel),dp(pos),du(used),dc(cu),dh(sink),o0(init),o1(init);
        for(int form=-1;form<=14;++form){
            std::vector<RadParam>p={pi("M",M),pi("heads",H),pi("dim",512),pi("window",win),pi("ratio",ratio),
                pi("topk",ratio?K:0),wf("scale",.03125),ps("dtype","bf16"),pi("kv_format",ratio?1:0)};
            std::vector<RadTensor>a={wt(dq.p,RAD_BF16,{M,H*512}),wt(w0.p,RAD_BF16,{nb,1,16,512}),
                wt(db.p,RAD_I32,{1,nb}),wt(du.p,RAD_I32,{1}),ratio?wt(comp.p,RAD_U8,{blocks,288}):RadTensor{},
                ratio?wt(dt.p,RAD_I32,{1,blocks}):RadTensor{},ratio?wt(ds.p,RAD_I32,{M,K}):RadTensor{},
                wt(dp.p,RAD_I32,{M}),wt(dc.p,RAD_I32,{2}),wt(dh.p,RAD_F32,{H}),wt(o0.p,RAD_BF16,{M,H*512})};
            auto b=a;b[1]=wt(w1.p,RAD_U8,{nb,8448});b[10]=wt(o1.p,RAD_BF16,{M,H*512});
            int rc=form<0?wcall(ratio?oldpacked:oldplain,a,p):wcheck(ref,a,p,form);p.push_back(pi("window_format",3));
            if(rc>=0)rc=form<0?wcall(normal,b,p):wcheck(alt,b,p,form);
            HC(hipDeviceSynchronize());bad+=rc<0||o0.get<uint16_t>(init.size())!=o1.get<uint16_t>(init.size());
            ++cases;if(form>=0)mask|=1u<<form;
        }
    }
    const bool pass=!bad&&cases==1152&&mask==32767;
    std::printf("%s window packed readers cases=%zu byte_errors=%zu forms=%u\n",pass?"PASS":"FAIL",cases,bad,mask);
    return pass;
}
} // namespace
int main(int argc,char**argv) {
    if(ds41_window_domain_main(argc,argv))return 1;
    void*lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!lib)return 2;
    const bool p=window_producers(lib),r=window_readers(lib);dlclose(lib);
    if(p&&r)std::puts("PASS ds41_window_packed_paths_exact producers=84 readers=1152 max_M=4096 raw=0 byte_errors=0");
    return p&&r?0:1;
}
