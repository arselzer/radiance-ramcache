#pragma once
#include "../rad_core.h"
#include "../../arch/common/rad_ds41_fp4_runtime.h"
#include <dlfcn.h>
#include <climits>

namespace rad {
using Ds41Fp4Origin = const void* (*)(RadLaunchFn);
using Ds41Fp4Identity = bool (*)(RadLaunchFn,const char*);
inline const void* ds41_fp4_origin(RadLaunchFn f) { Dl_info d{};
    return f&&dladdr(reinterpret_cast<void*>(f),&d)?d.dli_fbase:nullptr; }
inline bool ds41_fp4_identity(RadLaunchFn f,const char* s) { Dl_info d{};void* p=reinterpret_cast<void*>(f);
    return f&&s&&dladdr(p,&d)&&d.dli_saddr==p&&d.dli_sname&&!std::strcmp(d.dli_sname,s); }
inline bool ds41_fp4_classify_admit(const Program& p,int* count,std::string* why,
        Ds41Fp4Origin origin=ds41_fp4_origin,Ds41Fp4Identity identity=ds41_fp4_identity) {
    namespace f=rad_ds41_fp4_runtime;
    auto no=[&](const char* s){if(why)*why=s;return false;};
    if(!count||!origin||!identity)return no("missing classifier proof lookup");
    *count=0;const void* module=nullptr;int order=-1;
    for(const auto&o:p.ops) {
        int kind=-1;for(int j=0;j<f::OpCount;++j)if(o.op==f::kNames[j])kind=j;
        const char* fmt=o.base.get_s("format");
        if(kind<0) {
            if(o.op.find("fp4_classify")!=std::string::npos)return no("unknown classifier alias");
            if(fmt&&!std::strcmp(fmt,"fp4_e2m1")&&(o.op=="quant_roundtrip"||o.op=="quant_roundtrip_kvpack"||
               o.op=="rmsnorm_rope_roundtrip"||o.op=="rmsnorm_rope_roundtrip_kvpack"||
               o.op=="rope_rows_roundtrip"||o.op=="rope_rows_roundtrip_kvpack"))return no("unconverted FP4 producer");
            continue;
        }
        const bool capture=(kind&1)!=0,quant=kind<2,norm=kind>=2&&kind<4;
        long long n=0,block=0,nh=0,hd=0,rot=0,off=0,inv=-1;
        const char* sc=o.base.get_s("scale"),*dt=o.base.get_s("dtype");
        if(!fmt||std::strcmp(fmt,"fp4_e2m1")||!sc||!dt||std::strcmp(dt,"bf16")||
           !o.base.get_i("block",&block)||o.ranged_key!="M"||o.range_lo!=1||o.range_hi<1||o.range_hi>INT_MAX)
            return no("wrong classifier numerical/row contract");
        const int scale=!std::strcmp(sc,"ue8m0")?0:!std::strcmp(sc,"e4m3")?1:-1;
        if(quant) { if(!o.base.get_i("n",&n))return no("missing classifier width"); }
        else {
            const char* mode=o.base.get_s("mode");
            if(!o.base.get_i("n_head",&nh)||!o.base.get_i("head_dim",&hd)||
               !o.base.get_i("rot",&rot)||!o.base.get_i("off",&off)||!o.base.get_i("inverse",&inv)||
               !mode||std::strcmp(mode,"gptj")||rot!=64||inv!=0||nh<1||hd<1||nh>4096/hd)
                return no("unproven classifier rope geometry");
            n=nh*hd;
            if(!f::rope_scope(n,hd,32,off,nh*32,true,1.0f)||(norm&&n!=128))
                return no("classifier fused site outside closed native geometry");
            if(norm){long long declared=0;if(!o.base.get_i("n",&declared)||declared!=n)return no("wrong norm width");}
        }
        if(!f::quant_scope(n,block,1,scale,capture))return no("unproven classifier quant geometry");
        if(capture){long long format=0;if(!o.base.get_i("kv_format",&format)||format!=(n==512?1:2))
            return no("wrong classifier original-code format");}
        const int base=quant?(capture?2:1):norm?(capture?6:5):(capture?4:3);
        if(!o.schema||!o.schema->operands||o.schema->n_operands!=base+1||o.schema->operands[base].role!=RAD_OPD_INOUT||
           !o.schema->operands[base].optional||!o.schema->operands[base].name||
           std::strcmp(o.schema->operands[base].name,"classifier_audit"))return no("missing alias-only optional audit operand");
        for(int i=0;i<base;++i){const int want=quant?(i?RAD_OPD_OUT:RAD_OPD_INOUT):norm?
            (i==1?RAD_OPD_WEIGHT:(i==2||i==5)?RAD_OPD_OUT:RAD_OPD_IN):
            (i==0?RAD_OPD_INOUT:i==3?RAD_OPD_OUT:RAD_OPD_IN);
            if(o.schema->operands[i].role!=want||o.schema->operands[i].optional)
                return no("wrong classifier producer operand roles");}
        if(o.bands.empty())return no("classifier alias has no actual resolved band");
        const std::string symbol="r4d_rad_"+o.op;
        for(const auto&b:o.bands){const auto&r=b.dom[RAD_DOMAIN_DEVICE];const KernelRow*k=r.row;
            if(!k||!k->info||!k->info->name||!k->info->op||k->plugin!="libr4d"||k->plugin_order<0||
               k->info->domain!=RAD_DOMAIN_DEVICE||std::strcmp(k->info->name,f::kRows[kind])||
               std::strcmp(k->info->op,o.op.c_str())||!identity(k->info->launch,symbol.c_str()))
                return no("unknown classifier row/export");
            const void* here=origin(k->info->launch);
            if(!here||(module&&module!=here)||(order>=0&&order!=k->plugin_order))return no("mixed classifier modules/order");
            module=here;order=k->plugin_order;
            if(k->info->tunables||k->info->n_tunables||k->info->tune_valid||!r.choice.empty()||
               (!r.tune_source.empty()&&r.tune_source!="axis defaults")||b.dom[RAD_DOMAIN_HOST].row)
                return no("classifier tuning/host fallback is unproven");
        }
        ++*count;
    }
    if(!*count)return no("no actual classifier aliases");
    if(why)why->clear();return true;
}
}
