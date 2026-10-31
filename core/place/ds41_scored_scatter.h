/* Startup-only actual row/export proof, before weight allocation/Ctx/warmup. */
#pragma once
#include "../rad_core.h"
#include "../../arch/common/rad_ds41_scored_scatter.h"
#include <dlfcn.h>
#include <set>
namespace rad {
using Ds41ScoredOrigin=const void*(*)(RadLaunchFn);
using Ds41ScoredIdentity=bool(*)(RadLaunchFn,const char*);
inline const void* ds41_scored_origin(RadLaunchFn f){Dl_info i{};return f&&dladdr(reinterpret_cast<void*>(f),&i)?i.dli_fbase:nullptr;}
inline bool ds41_scored_identity(RadLaunchFn f,const char*s){Dl_info i{};const auto at=reinterpret_cast<void*>(f);return f&&s&&dladdr(at,&i)&&
    i.dli_sname&&i.dli_saddr==at&&!std::strcmp(i.dli_sname,s);}
inline bool ds41_scored_scatter_admit(const Program&p,int rank,int*ops,std::string*why,
    Ds41ScoredOrigin origin=ds41_scored_origin,Ds41ScoredIdentity identity=ds41_scored_identity) {
    using namespace rad_ds41_scored_scatter;
    auto no=[&](const char*m){if(why)*why=m;return false;};
    if(!ops||!origin||!identity||rank<0||rank>=4)return no("missing native rank/proof lookup");
    *ops=0;const void*module=nullptr;int order=-1;std::set<rad_weight>bias;
    for(const OpInfo&o:p.ops){
        if(o.op!=kOp){if(o.op.rfind("ds41_scored_topk_scatter",0)==0)return no("unknown scored-scatter alias");continue;}
        long long ne=0,nl=0,k=0,base=-1;
        if(o.ranged_key!="M"||o.range_lo!=1||o.range_hi!=1||!o.base.get_i("n_expert",&ne)||
           !o.base.get_i("n_local",&nl)||!o.base.get_i("top_k",&k)||!o.base.get_i("expert_base",&base)||
           !scope(4,rank,1,ne,nl,k,base,true,true))return no("non-native scored-scatter geometry");
        const char*s=o.base.get_s("score"),*dt=o.base.get_s("dtype");
        if(!s||std::strcmp(s,"sqrtsoftplus")||!dt||std::strcmp(dt,"f32"))return no("wrong scored-scatter arithmetic/dtype");
        if(!o.schema||o.schema->n_operands!=8||o.weights.size()!=1)return no("wrong scored-scatter operand contract");
        if(!o.weights[0]||!bias.insert(o.weights[0]).second)return no("duplicate/missing native correction binding");
        const char*names[]={"logits","bias","expert_ids","weights","sorted_tok","expert_offset","expert_count","token2slot"};
        for(int i=0;i<8;++i){const auto&sp=o.schema->operands[i];
            if(!sp.name||std::strcmp(sp.name,names[i])||sp.role!=(i==0?RAD_OPD_IN:i==1?RAD_OPD_WEIGHT:RAD_OPD_OUT))
                return no("wrong scored-scatter operand role/index");}
        if(o.bands.empty())return no("scored-scatter has no actual band");
        for(const Band&b:o.bands){const Resolved&r=b.dom[RAD_DOMAIN_DEVICE];const KernelRow*row=r.row;
            if(b.hi!=1||!row||!row->info||!row->info->name||!row->info->op||
               std::strcmp(row->info->name,kRow)||std::strcmp(row->info->op,kOp)||
               row->plugin!="libr4d"||row->info->domain!=RAD_DOMAIN_DEVICE||
               !identity(row->info->launch,kSymbol))return no("unknown/missing/miswired actual scored-scatter export");
            const void*here=origin(row->info->launch);
            if(!here||(module&&module!=here)||(order>=0&&order!=row->plugin_order))return no("mixed scored-scatter libraries");
            module=here;order=row->plugin_order;
            if(row->info->tunables||row->info->n_tunables||row->info->tune_valid||!r.choice.empty()||
               (!r.tune_source.empty()&&r.tune_source!="axis defaults")||b.dom[RAD_DOMAIN_HOST].row)
                return no("tuned or host-fallback scored-scatter is unproven");
        }
        ++*ops;
    }
    if(*ops!=40)return no("expected one native M1 scored-scatter per40 trunk layers");
    if(why)why->clear();return true;
}
}
