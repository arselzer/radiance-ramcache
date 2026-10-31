/* Admission BEFORE Ctx/warmup: every selected score writer/reader must be an
 * explicitly known16-bit alias from one actual libr4d DSO. No packed fallback. */
#pragma once
#include "ds41_index_memory.h"
#include "../../arch/deepseek41/ds41_index_memory.h"
namespace rad {
struct Ds41IndexB16Proof { int writers=0,candidates=0,topk=0;int64_t rows=0,columns=0; };
inline bool ds41_index_b16_admit(const Program& p,Ds41IndexB16Proof*out,std::string*why,
                               Ds41IndexOrigin origin=ds41_index_module) {
    auto no=[&](const char*m){if(why)*why=m;return false;};
    if(!out||!origin)return no("missing proof/origin lookup");
    *out={};const void*module=nullptr;int order=-1;
    for(const auto&b:p.buffers)if(b.name=="index_score") {
        if(out->rows||b.decl.dtype!=RAD_BF16||b.decl.rank!=2||b.decl.shape[0]<=0||b.decl.shape[1]<=0)
            return no("missing/duplicate/non-bf16 score-plane declaration");
        out->rows=b.decl.shape[0];out->columns=b.decl.shape[1];
    }
    for(const OpInfo&op:p.ops) {
        const bool score=op.op=="index_score_b16",cand=op.op=="index_candidates_b16",top=op.op=="index_topk_b16";
        if(!score&&!cand&&!top) {
            if(op.op.rfind("index_score",0)==0||op.op.rfind("index_candidates",0)==0||op.op.rfind("index_topk",0)==0)
                return no("old/unknown/packed score writer or reader in16-bit program");
            continue;
        }
        if(op.bands.empty())return no("16-bit index op has no bands");
        const char* expected=ds41_index_known_row(op.op);
        for(const Band&b:op.bands) {
            const Resolved&r=b.dom[RAD_DOMAIN_DEVICE];const KernelRow*k=r.row;
            if(!k||!k->info||!k->info->name||!k->info->op||std::strcmp(k->info->name,expected)||
               std::strcmp(k->info->op,op.op.c_str())||k->info->domain!=RAD_DOMAIN_DEVICE||k->plugin!="libr4d")
                return no("16-bit index op resolves an unknown/missing/incompatible device row");
            const void*here=origin(k->info->launch);
            if(!here||(module&&here!=module)||(order>=0&&k->plugin_order!=order))
                return no("16-bit index rows do not share one proven loaded libr4d");
            module=here;order=k->plugin_order;
            if(k->info->tunables||k->info->n_tunables||k->info->tune_valid||!r.choice.empty()||
               (!r.tune_source.empty()&&r.tune_source!="axis defaults"))return no("unproven tuned16-bit index row");
            if(b.dom[RAD_DOMAIN_HOST].row)return no("host fallback is not a proven16-bit index implementation");
        }
        out->writers+=score;out->candidates+=cand;out->topk+=top;
    }
    if(!out->rows||!out->writers||!out->candidates||!out->topk)return no("incomplete16-bit score/candidate/topk chain");
    if(why)why->clear();return true;
}
} // namespace rad
