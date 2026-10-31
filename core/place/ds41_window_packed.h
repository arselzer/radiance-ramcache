/* Native FP8 WINDOW admission. A declaration/flag is not loaded-code authority.
 * This runs before planning/allocation; issue.cpp separately validates every actual KV operand.
 * The independent card receipt must still prove the original producer's zero-Raw domain. */
#pragma once
#include "../rad_core.h"
#include "../mem/kv_storage.h"
#include <cstring>
#include <dlfcn.h>
#include <string>

namespace rad {
inline bool ds41_window_packed_parse(const char* e, bool* out) {
    if (!out) return false;
    *out = false;
    if (!e || !*e || !std::strcmp(e, "0")) return true;
    if (std::strcmp(e, "1")) return false;
    *out = true; return true;
}
struct Ds41WindowPackedProof { int groups=0, layers=0, producers=0, stores=0, readers=0; };
using Ds41WindowOrigin = const void* (*)(RadLaunchFn);
using Ds41WindowIdentity = bool (*)(RadLaunchFn, const char*);
inline const void* ds41_window_origin(RadLaunchFn f) {
    Dl_info d{};
    return f && dladdr(reinterpret_cast<void*>(f), &d) ? d.dli_fbase : nullptr;
}
inline bool ds41_window_identity(RadLaunchFn f, const char* symbol) {
    Dl_info d{}; void* p=reinterpret_cast<void*>(f);
    return f && symbol && dladdr(p,&d) && d.dli_saddr==p && d.dli_sname && !std::strcmp(d.dli_sname,symbol);
}
inline bool ds41_window_packed_admit(const Program& p, Ds41WindowPackedProof* out, std::string* why,
        Ds41WindowOrigin origin=ds41_window_origin, Ds41WindowIdentity identity=ds41_window_identity) {
    auto no=[&](const char* m){if(why)*why=m;return false;};
    if(!out||!origin||!identity)return no("missing proof or loaded-code lookup");
    *out={};
    for(const auto& g:p.kv_groups) if(g.decl.storage_format==RAD_KV_STORAGE_DS41_FP8_WINDOW_G32) {
        if(kv_storage_page_error(g.decl,g.block_size)||g.layers.empty()||
           g.bytes_per_block!=int64_t(8448)*int64_t(g.layers.size()))return no("invalid physical native WINDOW declaration");
        ++out->groups;out->layers+=(int)g.layers.size();
    }
    if(out->groups!=1||out->layers<1)return no("expected exactly one native packed main WINDOW group");
    const void* module=nullptr;int order=-1;
    for(const auto& op:p.ops) {
        const bool quant=op.op=="quant_roundtrip_windowpack", norm=op.op=="rmsnorm_rope_roundtrip_windowpack";
        const bool rope=op.op=="rope_rows_roundtrip_windowpack", store=op.op=="kv_store_windowpack";
        const bool read=op.op=="attn_sparse_sink_windowpack";
        if(!quant&&!norm&&!rope&&!store&&!read) {
            if(op.op.find("windowpack")!=std::string::npos)return no("unknown packed-window op");
            continue;
        }
        const char* name=quant?"quant_roundtrip_fp8_windowpack":norm?"rmsnorm_rope_roundtrip_fp8_windowpack":
            rope?"rope_rows_roundtrip_fp8_windowpack":store?"ds41_kv_store_windowpack":"ds41_attn_sparse_sink_d512_windowpack";
        const char* symbol=quant?"r4d_rad_quant_roundtrip_windowpack":norm?"r4d_rad_rmsnorm_rope_roundtrip_windowpack":
            rope?"r4d_rad_rope_rows_roundtrip_windowpack":store?"r4d_rad_kv_store_windowpack":"r4d_rad_ds41_attn_sparse_sink_windowpack";
        long long format=-1;
        if(!op.base.get_i(read?"window_format":"kv_format",&format)||format!=3||op.bands.empty())
            return no("missing explicit format or resolved bands");
        for(const auto& b:op.bands) {
            const Resolved& r=b.dom[RAD_DOMAIN_DEVICE];const KernelRow* k=r.row;
            if(!k||!k->info||!k->info->name||!k->info->op||k->plugin!="libr4d"||
               k->info->domain!=RAD_DOMAIN_DEVICE||std::strcmp(k->info->name,name)||
               std::strcmp(k->info->op,op.op.c_str())||!identity(k->info->launch,symbol))
                return no("unknown device row/export for packed WINDOW");
            const void* here=origin(k->info->launch);
            if(!here||(module&&module!=here)||(order>=0&&order!=k->plugin_order))
                return no("packed WINDOW kernels are not one coherent loaded libr4d");
            module=here;order=k->plugin_order;
            if(k->info->tunables||k->info->n_tunables||k->info->tune_valid||!r.choice.empty()||
               (!r.tune_source.empty()&&r.tune_source!="axis defaults")||b.dom[RAD_DOMAIN_HOST].row)
                return no("unproven tuning or host fallback in packed WINDOW chain");
        }
        out->producers+=quant||norm||rope;out->stores+=store;out->readers+=read;
    }
    if(out->stores!=out->layers||out->readers!=out->layers||out->producers!=out->layers)
        return no("incomplete per-layer producer/store/reader packed WINDOW chain");
    if(why)why->clear();return true;
}
} // namespace rad
