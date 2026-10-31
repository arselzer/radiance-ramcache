/* Packed DS41 reader/fused-producer BYTE oracle. Source-only until root's serialized GPU window.
 * Extends the standalone codec/producer oracle's allocation/ABI helpers; no engine source included.
 * Run with the SAME libr4d cohort, twice: unpinned and EXACT_TALL=1 (separate processes; cached flag).
 * Every comparison is bytes, including NaN payloads, signed zero and guard regions. */
#define main ds41_codec_oracle_main
#include "ds41_kv_codec_device_check.hip"
#undef main

namespace {
using Check = int (*)(const RadArgs*, RadStream, int);
struct Dev {
    void* p = nullptr;
    template<class T> explicit Dev(const std::vector<T>& v) {
        HC(hipMalloc(&p, v.size() * sizeof(T)));
        HC(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
    }
    ~Dev() { if (p) HC(hipFree(p)); }
    Dev(const Dev&) = delete;
    template<class T> std::vector<T> get(size_t n) const {
        std::vector<T> v(n); HC(hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost)); return v;
    }
};
RadParam pf(const char* key, double v) { RadParam p{}; p.key=key; p.kind=RAD_P_F64; p.dval=v; return p; }
int checked(Check fn, std::vector<RadTensor>& t, std::vector<RadParam>& p, int form) {
    RadArgs a{}; a.t=t.data(); a.n_t=(int)t.size(); a.p=p.data(); a.n_p=(int)p.size(); a.world_size=1;
    return fn(&a, nullptr, form);
}
struct Rows {
    int kind, n, count; int64_t rb;
    std::vector<uint8_t> packed;
    std::vector<uint16_t> plain;
    Rows(int k, int rows): kind(k),n(k==1?512:128),count(rows),rb(kc::row_bytes((uint32_t)k,n)),
                          packed((size_t)(rows*rb)),plain((size_t)rows*n) {
        const int G=kc::group_size((uint32_t)k);
        for (int r=0;r<rows;++r) {
            uint8_t* row=packed.data()+r*rb;
            for(int g=0;g<n/G;++g) {
                if (!r) continue; // Untouched zero-initialized page.
                uint8_t* payload=row+g*(G/2);
                const uint8_t scale=(uint8_t)(k==1 ? 1+(r*7+g*13)%126 : 1+(r*17+g*11)%200);
                row[n/2+g]=scale;
                for(int i=0;i<G;i+=2) payload[i/2]=(uint8_t)(((r+g+i)&15)|(((r+g+i+1)&15)<<4));
                if (k==2&&(r+g)%13==0) row[n/2+g]=254; // Explicit source1 half-product FTZ tag.
                if ((r+g)%7==0) { // Inline captured-bit escape, including signed zeros and NaNs.
                    std::fill(payload,payload+G/2,0);
                    const uint16_t a=(r&1)?0x8000:0x7fc1,b=(r&1)?0:0xffc2;
                    payload[0]=(uint8_t)a;payload[1]=(uint8_t)(a>>8);
                    payload[2]=(uint8_t)b;payload[3]=(uint8_t)(b>>8);
                    for(int i=0;i<G/8;++i)payload[4+i]=(uint8_t)(0xa5u^r);
                    row[n/2+g]=0;
                }
            }
            if (!kc::decode_row((uint32_t)k,row,n,plain.data()+(size_t)r*n)) std::exit(2);
        }
    }
};

bool readers(void* lib) {
    const auto ai=(Check)dlsym(lib,"r4d_rad_ds41_index_score_kvpack_check");
    const auto aa=(Check)dlsym(lib,"r4d_rad_ds41_attn_sparse_sink_kvpack_check");
    const auto ni=(Launch)dlsym(lib,"r4d_rad_ds41_index_score");
    const auto np=(Launch)dlsym(lib,"r4d_rad_ds41_index_score_kvpack");
    const auto na=(Launch)dlsym(lib,"r4d_rad_ds41_attn_sparse_sink");
    const auto nk=(Launch)dlsym(lib,"r4d_rad_ds41_attn_sparse_sink_kvpack");
    if(!ai||!aa||!ni||!np||!na||!nk)return false;
    size_t bad=0,cases=0;
    constexpr int blocks=80,C=65;
    for(int ratio:{1,2}) for(int M:{1,4,16,49}) for(int H:{1,2,4,16,32}) {
        Rows rows(2,blocks); Dev kc0(rows.plain),kc1(rows.packed);
        std::vector<uint16_t> q((size_t)M*H*128),w((size_t)M*H);
        for(size_t i=0;i<q.size();++i)q[i]=(uint16_t)(0x3c00+(i*31%1024));
        for(size_t i=0;i<w.size();++i)w[i]=(uint16_t)(0x3e00+(i*17%128));
        std::vector<int32_t> pos(M),cu={0,M/2,M};
        // Two sequences, remapped and absent pages, partial/empty reach and an output run.
        std::vector<int32_t> both(2*C);
        for(int s=0;s<2;++s)for(int i=0;i<C;++i)both[s*C+i]=i%11==0?-1:(i*7+s*13)%blocks;
        for(int i=0;i<M;++i)pos[i]=i==0?0:ratio*(C-3)+i;
        std::vector<uint32_t> init((size_t)M*(C+4),0x7fa5a5a5u);
        Dev dq(q),dw(w),db(both),dp(pos),dc(cu),o0(init),o1(init);
        for(int form:{-1,0,1}) {
            std::vector<RadParam> p={pi("M",M),pi("heads",H),pi("dim",128),pi("ratio",ratio),
                pi("width",C),pf("wscale",.015625),ps("dtype","bf16"),pi("row_part",0)};
            auto out=[&](void* d){auto t=tensor(d,RAD_F32,{M,C});t.stride[0]=C+4;return t;};
            std::vector<RadTensor> t0={tensor(dq.p,RAD_BF16,{M,H*128}),tensor(dw.p,RAD_BF16,{M,H}),
                tensor(kc0.p,RAD_BF16,{blocks,1,ratio,128/ratio}),tensor(db.p,RAD_I32,{2,C}),
                tensor(dp.p,RAD_I32,{M}),tensor(dc.p,RAD_I32,{3}),out(o0.p)};
            auto t1=t0;t1[2]=tensor(kc1.p,RAD_U8,{blocks,rows.rb});t1[6]=out(o1.p);
            int rc=form<0?call(ni,t0,p):checked(ai,t0,p,form);
            p.push_back(pi("kv_format",2));
            if(rc>=0)rc=form<0?call(np,t1,p):checked(ai,t1,p,form);
            HC(hipDeviceSynchronize()); ++cases;
            bad+=rc<0 || o0.get<uint32_t>(init.size())!=o1.get<uint32_t>(init.size());
        }
    }
    for(int ratio:{1,2}) for(int M:{1,4,16,49}) for(int win:{3,16,128}) {
        constexpr int H=16,K=40,bs=4,nb=64;
        Rows rows(1,blocks);Dev cc0(rows.plain),cc1(rows.packed);
        std::vector<uint16_t> q((size_t)M*H*512),wc((size_t)nb*bs*512);
        for(size_t i=0;i<q.size();++i)q[i]=(uint16_t)(0x3b00+(i*31%256));
        for(size_t i=0;i<wc.size();++i)wc[i]=(uint16_t)(0xbb00+(i*17%256));
        std::vector<int32_t> wb(nb),cb(blocks),sel((size_t)M*K),pos(M),used={256},cu={0,M};
        std::vector<float> sink(H,-1.25f);
        for(int i=0;i<nb;++i)wb[i]=i%13==0?-1:(i*3)%nb;
        for(int i=0;i<blocks;++i)cb[i]=i%11==0?-1:(i*7)%blocks;
        for(int r=0;r<M;++r){pos[r]=192+r;for(int i=0;i<K;++i)sel[r*K+i]=i%9==0?-1:(i*3+r)%blocks;}
        std::vector<uint16_t> init((size_t)M*H*512+16,0x5aa5);
        Dev dq(q),dw(wc),db(wb),dt(cb),ds(sel),dp(pos),du(used),dc(cu),dh(sink),o0(init),o1(init);
        for(int form=-1;form<=14;++form) {
            std::vector<RadParam> p={pi("M",M),pi("heads",H),pi("dim",512),pi("window",win),
                pi("ratio",ratio),pi("topk",K),pf("scale",.03125),ps("dtype","bf16")};
            auto out=[&](void* d){return tensor(d,RAD_BF16,{M,H*512});};
            std::vector<RadTensor> t0={tensor(dq.p,RAD_BF16,{M,H*512}),tensor(dw.p,RAD_BF16,{nb,1,bs,512}),
                tensor(db.p,RAD_I32,{1,nb}),tensor(du.p,RAD_I32,{1}),
                tensor(cc0.p,RAD_BF16,{blocks,1,ratio,512/ratio}),tensor(dt.p,RAD_I32,{1,blocks}),
                tensor(ds.p,RAD_I32,{M,K}),tensor(dp.p,RAD_I32,{M}),tensor(dc.p,RAD_I32,{2}),
                tensor(dh.p,RAD_F32,{H}),out(o0.p)};
            auto t1=t0;t1[4]=tensor(cc1.p,RAD_U8,{blocks,rows.rb});t1[10]=out(o1.p);
            int rc=form<0?call(na,t0,p):checked(aa,t0,p,form);
            p.push_back(pi("kv_format",1));
            if(rc>=0)rc=form<0?call(nk,t1,p):checked(aa,t1,p,form);
            HC(hipDeviceSynchronize());++cases;
            bad+=rc<0 || o0.get<uint16_t>(init.size())!=o1.get<uint16_t>(init.size());
        }
    }
    std::printf("%s packed reader normal/scalar6/decode7/WMMA: %zu cases %zu byte failures\n",bad?"FAIL":"PASS",cases,bad);
    return bad==0;
}

bool fused_producers(void* lib) {
    const auto norm0=(Check)dlsym(lib,"r4d_rad_rmsnorm_rope_roundtrip_n128_check");
    const auto norm1=(Check)dlsym(lib,"r4d_rad_rmsnorm_rope_roundtrip_kvpack_n128_check");
    const auto normal0=(Launch)dlsym(lib,"r4d_rad_rmsnorm_rope_roundtrip");
    const auto normal1=(Launch)dlsym(lib,"r4d_rad_rmsnorm_rope_roundtrip_kvpack");
    const auto rope0=(Launch)dlsym(lib,"r4d_rad_rope_rows_roundtrip");
    const auto rope1=(Launch)dlsym(lib,"r4d_rad_rope_rows_roundtrip_kvpack");
    const auto st=(Launch)dlsym(lib,"r4d_rad_entry_store_kvpack");
    if(!norm0||!norm1||!normal0||!normal1||!rope0||!rope1||!st)return false;
    size_t bad=0,cases=0; int max_m=0; unsigned norm_forms=0,rope_forms=0;
    std::vector<int> heights;
    for(int kind:{1,2}) for(int M:{1,7,33,49,257,2048,4096}) for(int form:{-1,0,1}) {
        if(kind==1&&form!=-1)continue;
        const int n=kind==1?512:128,G=kc::group_size((uint32_t)kind),pitch=n+16,blocks=M+2;
        const int64_t rb=kc::row_bytes((uint32_t)kind,n);
        std::vector<uint16_t> x((size_t)M*pitch,0x5aa5),init=x;
        for(int r=0;r<M;++r)for(int c=0;c<n;++c)x[r*pitch+c]=kind==2&&r%4<2
            ? (uint16_t)(0x3c00+((r*37+c*13)%1024)) : input_bits(r,c,G);
        init=x;std::vector<float>w(n),cs((size_t)M*64);std::vector<int32_t>pos(M),pg(M);
        for(int c=0;c<n;++c)w[c]=(c%7==0)?-1.f:1.f;
        for(int r=0;r<M;++r){pos[r]=r%5==0?-1:r;pg[r]=r+1;for(int c=0;c<32;++c){cs[r*64+c]=.75f;cs[r*64+32+c]=.5f;}}
        std::vector<uint8_t> cap((size_t)M*rb+16,0xa5),cache((size_t)blocks*rb,0);
        Dev dx0(x),dx1(x),y0(init),y1(init),dw(w),dt(cs),dp(pos),dg(pg),dc(cap),dk(cache);
        auto xt=[&](void* d){auto t=tensor(d,RAD_BF16,{M,n});t.stride[0]=pitch;return t;};
        std::vector<RadParam> p={pi("M",M),pi("n",n),pi("n_head",1),pi("head_dim",n),pi("rot",64),
            pi("off",n-64),pi("inverse",0),ps("mode","gptj"),pi("block",G),ps("format","fp4_e2m1"),
            ps("scale",kind==1?"e4m3":"ue8m0"),ps("dtype","bf16"),pf("eps",1e-6),pi("block_size",1)};
        std::vector<RadTensor> a,b;
        int rc;
        if(kind==2){a={xt(dx0.p),tensor(dw.p,RAD_F32,{n}),xt(y0.p),tensor(dt.p,RAD_F32,{M,64}),tensor(dp.p,RAD_I32,{M})};b=a;b[0]=xt(dx1.p);b[2]=xt(y1.p);b.push_back(tensor(dc.p,RAD_U8,{M,rb}));rc=form<0?call(normal0,a,p):checked(norm0,a,p,form);p.push_back(pi("kv_format",2));if(rc>=0)rc=form<0?call(normal1,b,p):checked(norm1,b,p,form);}
        else {a={xt(dx0.p),tensor(dt.p,RAD_F32,{M,64}),tensor(dp.p,RAD_I32,{M})};b=a;b[0]=xt(dx1.p);b.push_back(tensor(dc.p,RAD_U8,{M,rb}));rc=call(rope0,a,p);p.push_back(pi("kv_format",1));if(rc>=0)rc=call(rope1,b,p);}
        void* actual=kind==2?y1.p:dx1.p;
        std::vector<RadTensor> store={xt(actual),tensor(dc.p,RAD_U8,{M,rb}),tensor(dg.p,RAD_I32,{M}),tensor(dk.p,RAD_U8,{blocks,rb})};
        if(rc>=0)rc=call(st,store,p);HC(hipDeviceSynchronize());++cases;
        max_m=std::max(max_m,M);
        if(std::find(heights.begin(),heights.end(),M)==heights.end())heights.push_back(M);
        if(kind==2)norm_forms|=1u<<(form+1);else rope_forms|=1u<<(form+1);
        const auto oldbits=kind==2?y0.get<uint16_t>(init.size()):dx0.get<uint16_t>(init.size());
        const auto newbits=kind==2?y1.get<uint16_t>(init.size()):dx1.get<uint16_t>(init.size());
        bad+=rc<0||oldbits!=newbits;const auto packed=dk.get<uint8_t>(cache.size());std::vector<uint16_t>decoded(n);
        for(int r=0;r<M;++r){bad+=!kc::decode_row((uint32_t)kind,packed.data()+(r+1)*rb,n,decoded.data());
            for(int c=0;c<n;++c){uint16_t v=oldbits[r*pitch+c];if((v&0x7fff)>0x7f80)v|=0x40;bad+=decoded[c]!=v;}}
        const auto captured=dc.get<uint8_t>(cap.size());for(size_t i=(size_t)M*rb;i<cap.size();++i)bad+=captured[i]!=0xa5;
    }
    const bool pass=bad==0&&cases==28&&max_m==4096&&heights.size()==7&&norm_forms==7&&rope_forms==1;
    std::printf("%s packed fused rope/norm old+n128: %zu cases %zu byte failures\n",pass?"PASS":"FAIL",cases,bad);
    if(pass){
        std::sort(heights.begin(),heights.end());
        std::printf("PASS packed fused coverage: cases=%zu max_M=%d heights=",cases,max_m);
        for(size_t i=0;i<heights.size();++i)std::printf("%s%d",i?",":"",heights[i]);
        std::printf(" norm_forms=normal,old,n128 rope=normal\n");
    }
    return pass;
}
}
int main(int argc,char**argv){
    if(argc!=2){std::fprintf(stderr,"usage: %s LIBR4D.so\n",argv[0]);return 2;}
    if(ds41_codec_oracle_main(argc,argv))return 1;
    void*lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!lib)return 2;
    const bool a=fused_producers(lib),b=readers(lib);dlclose(lib);return a&&b?0:1;
}
