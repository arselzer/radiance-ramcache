/* Future guarded GPU window ONLY. Reuse unchanged allocation/ABI/reference fixtures; no timing.
 * Existing all-form/producer checker remains a separate required gate, with unchanged output. */
#define main ds41_kv_pair_unused_full_oracle_main
#include "ds41_kv_packed_paths_check.hip"
#undef main
#include "../../arch/common/rad_ds41_kv_pair.h"

namespace {
bool pair_readers(void* lib) {
    const auto check=(Check)dlsym(lib,"r4d_rad_ds41_attn_sparse_sink_kvpair_check");
    const auto plain=(Launch)dlsym(lib,"r4d_rad_ds41_attn_sparse_sink");
    const auto normal=(Launch)dlsym(lib,"r4d_rad_ds41_attn_sparse_sink_kvpack");
    if(!check||!plain||!normal) {
        std::fprintf(stderr,"FAIL pair checker needs the new coherent plugin export\n");return false;
    }
    size_t cases=0,bad=0,normal_ok=0,forced_ok=0,old_ok=0,refused=0;
    for(int ratio:{1,2})for(int win:{3,16,128})for(int K:{40,512}) {
        constexpr int M=1,H=16,blocks=640,bs=4,nb=64;
        Rows rows(1,blocks);Dev cc0(rows.plain),cc1(rows.packed);
        std::vector<uint16_t> q(H*512),wc(nb*bs*512);
        for(size_t i=0;i<q.size();++i)q[i]=(uint16_t)(0x3b00+(i*31%256));
        for(size_t i=0;i<wc.size();++i)wc[i]=(uint16_t)(0xbb00+(i*17%256));
        std::vector<int32_t> wb(nb),cb(blocks),sel(K),pos={192},used={256},cu={0,1};
        std::vector<float> sink(H,-1.25f);
        for(int i=0;i<nb;++i)wb[i]=i%13==0?-1:(i*3)%nb;
        for(int i=0;i<blocks;++i)cb[i]=i%11==0?-1:(i*7)%blocks;
        cb[1]=0; // A real selected, zero-initialized compressed page, not only sentinel padding.
        for(int i=0;i<K;++i)sel[i]=i%9==0?-1:(i*3)%blocks;
        sel[1]=1;sel[2]=blocks+1; // Real zero page plus out-of-table/null-key guard.
        std::vector<uint16_t> init(H*512+16,0x5aa5);
        Dev dq(q),dw(wc),db(wb),dt(cb),ds(sel),dp(pos),du(used),dc(cu),dh(sink);
        Dev refout(init),oldout(init),newout(init),normalout(init);
        std::vector<RadParam> p={pi("M",M),pi("heads",H),pi("dim",512),pi("window",win),
            pi("ratio",ratio),pi("topk",K),pf("scale",.03125),ps("dtype","bf16")};
        auto output=[&](void* d){return tensor(d,RAD_BF16,{M,H*512});};
        std::vector<RadTensor> t={tensor(dq.p,RAD_BF16,{M,H*512}),tensor(dw.p,RAD_BF16,{nb,1,bs,512}),
            tensor(db.p,RAD_I32,{1,nb}),tensor(du.p,RAD_I32,{1}),
            tensor(cc0.p,RAD_BF16,{blocks,1,ratio,512/ratio}),tensor(dt.p,RAD_I32,{1,blocks}),
            tensor(ds.p,RAD_I32,{M,K}),tensor(dp.p,RAD_I32,{M}),tensor(dc.p,RAD_I32,{2}),
            tensor(dh.p,RAD_F32,{H}),output(refout.p)};
        int rc=call(plain,t,p);
        p.push_back(pi("kv_format",1));t[4]=tensor(cc1.p,RAD_U8,{blocks,rows.rb});
        t[10]=output(oldout.p);const int old=checked(check,t,p,0);
        t[10]=output(newout.p);const int forced=checked(check,t,p,1);
        t[10]=output(normalout.p);const int dispatched=call(normal,t,p);
        HC(hipDeviceSynchronize());++cases;
        normal_ok+=dispatched==RAD_OK;forced_ok+=forced==RAD_OK;old_ok+=old==RAD_OK;
        const auto expected=refout.get<uint16_t>(init.size());
        bad+=rc!=RAD_OK||old!=RAD_OK||forced!=RAD_OK||dispatched!=RAD_OK;
        bad+=expected!=oldout.get<uint16_t>(init.size());
        bad+=expected!=newout.get<uint16_t>(init.size());
        bad+=expected!=normalout.get<uint16_t>(init.size());
        // The new forced entry refuses unsupported shapes instead of pretending it ran a pair.
        auto wrong=p;for(auto& v:wrong)if(!std::strcmp(v.key,"heads"))v.ival=8;
        const int wrong_shape=checked(check,t,wrong,1);
        auto plain_p=p;plain_p.back()=pi("kv_format",0);
        const int wrong_format=checked(check,t,plain_p,1);
        const int wrong_form=checked(check,t,p,2);
        refused+=(wrong_shape<0)+(wrong_format<0)+(wrong_form<0);
        bad+=wrong_shape>=0||wrong_format>=0||wrong_form>=0;
        HC(hipDeviceSynchronize());
        bad+=normalout.get<uint16_t>(init.size())!=expected;
    }
    const bool ok=!bad&&cases==12&&normal_ok==12&&forced_ok==12&&old_ok==12&&refused==36;
    std::printf("%s ds41 pair directPV reader: cases=%zu normal=%zu forced=%zu old=%zu refused=%zu bad=%zu M=1 H=16 HG=1 NT=1024 DPT=2 PVD=1\n",
        ok?"PASS":"FAIL",cases,normal_ok,forced_ok,old_ok,refused,bad);
    return ok;
}
}
int main(int argc,char** argv) {
    if(argc!=2){std::fprintf(stderr,"usage: %s LIBR4D.so (PACKED_FAST=1 PACKED_PAIR=1)\n",argv[0]);return 2;}
    if(rad_ds41_kv_pair::parse_mode(std::getenv("RADIANCE_EXPERIMENTAL_DS41_KV_PACKED_PAIR"))!=1||
       rad_ds41_kv_pair::parse_mode(std::getenv("RADIANCE_EXPERIMENTAL_DS41_KV_PACKED_FAST"))!=1) {
        std::fprintf(stderr,"FAIL pair path oracle requires explicit PACKED_PAIR=1 PACKED_FAST=1\n");return 2;
    }
    void* lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!lib)return 2;
    const bool ok=pair_readers(lib);dlclose(lib);return ok?0:1;
}
