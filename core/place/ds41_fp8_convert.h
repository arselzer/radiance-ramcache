#pragma once
#include "../rad_core.h"
#include "../../arch/common/rad_ds41_fp8_runtime.h"
#include <climits>
#include <dlfcn.h>

namespace rad {
using Ds41Fp8Origin=const void*(*)(RadLaunchFn);
using Ds41Fp8Identity=bool(*)(RadLaunchFn,const char*);
inline const void* ds41_fp8_origin(RadLaunchFn f){Dl_info d{};
    return f&&dladdr(reinterpret_cast<void*>(f),&d)?d.dli_fbase:nullptr;}
inline bool ds41_fp8_identity(RadLaunchFn f,const char*s){Dl_info d{};void*p=reinterpret_cast<void*>(f);
    return f&&s&&dladdr(p,&d)&&d.dli_saddr==p&&d.dli_sname&&!std::strcmp(d.dli_sname,s);}
inline bool ds41_fp8_convert_admit(const Program&p,int*count,std::string*why,
        Ds41Fp8Origin origin=ds41_fp8_origin,Ds41Fp8Identity identity=ds41_fp8_identity) {
    namespace f=rad_ds41_fp8_runtime;
    auto no=[&](const char*s){if(why)*why=s;return false;};
    if(!count||!origin||!identity)return no("missing conversion proof lookup");
    *count=0;int ropes=0,order=-1;const void*module=nullptr;
    for(const auto&o:p.ops){
        int kind=-1;for(int j=0;j<f::OpCount;++j)if(o.op==f::kNames[j])kind=j;
        if(kind<0){if(o.op.find("fp8_convert")!=std::string::npos)return no("unknown conversion alias");continue;}
        long long n=0,b=0,nh=0,hd=0,rot=0,off=0,inv=0;
        const char*fmt=o.base.get_s("format"),*sc=o.base.get_s("scale"),*dt=o.base.get_s("dtype");
        if(!fmt||std::strcmp(fmt,"fp8_e4m3")||!sc||std::strcmp(sc,"ue8m0")||!dt||std::strcmp(dt,"bf16")||
           !o.base.get_i("block",&b)||o.ranged_key!="M"||o.range_lo!=1||o.range_hi<1||o.range_hi>INT_MAX)
            return no("wrong conversion numerical/row contract");
        if(kind==f::Quant){if(!o.base.get_i("n",&n))return no("missing conversion width");}
        else {
            const char*mode=o.base.get_s("mode");
            if(!o.base.get_i("n_head",&nh)||!o.base.get_i("head_dim",&hd)||!o.base.get_i("rot",&rot)||
               !o.base.get_i("off",&off)||!o.base.get_i("inverse",&inv)||!mode||std::strcmp(mode,"gptj")||
               nh!=16||hd!=512||rot!=64||off!=448||inv!=1)return no("unproven conversion fused geometry");
            n=nh*hd;
            if(!f::rope_scope(n,hd,rot/2,off,nh*rot/2,true,-1.0f))return no("wrong output rope site");
            ++ropes;
        }
        if(!f::quant_scope(n,b,0,0))return no("unproven conversion quant geometry");
        const int base=kind==f::Quant?1:3;
        if(!o.schema||!o.schema->operands||o.schema->n_operands!=base+1)return no("missing conversion schema");
        for(int i=0;i<=base;++i){const auto&d=o.schema->operands[i];
            const int role=i==0||i==base?RAD_OPD_INOUT:RAD_OPD_IN;
            if(d.role!=role||bool(d.optional)!=(i==base))return no("wrong conversion operand roles");}
        const auto&extra=o.schema->operands[base];
        if(!extra.name||std::strcmp(extra.name,"convert_audit"))return no("wrong optional audit operand");
        if(o.bands.empty())return no("conversion has no actual resolved band");
        const std::string symbol="r4d_rad_"+o.op;
        for(const auto&band:o.bands){const auto&r=band.dom[RAD_DOMAIN_DEVICE];const KernelRow*k=r.row;
            if(!k||!k->info||!k->info->name||!k->info->op||k->plugin!="libr4d"||k->plugin_order<0||
               k->info->domain!=RAD_DOMAIN_DEVICE||std::strcmp(k->info->name,f::kRows[kind])||
               std::strcmp(k->info->op,o.op.c_str())||!identity(k->info->launch,symbol.c_str()))
                return no("unknown conversion row/export");
            const void*here=origin(k->info->launch);
            if(!here||(module&&here!=module)||(order>=0&&order!=k->plugin_order))return no("mixed conversion modules/order");
            module=here;order=k->plugin_order;
            if(k->info->init||k->info->fini||k->info->scratch||k->info->layout||k->info->relayout||k->info->unrelayout||k->info->replaces||
               k->info->tunables||k->info->n_tunables||k->info->tune_valid||!r.choice.empty()||
               (!r.tune_source.empty()&&r.tune_source!="axis defaults")||band.dom[RAD_DOMAIN_HOST].row)
                return no("unproven conversion tuning/host fallback");
        }
        ++*count;
    }
    /* Current native trunk has 40 output sites, independent of drafter enablement. A zero-use
     * plain-only graph does not qualify this native runtime experiment. */
    if(ropes!=40)return no("missing native trunk conversion aliases");
    if(why)why->clear();return true;
}
}
