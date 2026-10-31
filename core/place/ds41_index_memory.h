/* Startup proof for a lower index row floor. Environment alone is not dispatch authority. */
#pragma once
#include "../rad_core.h"
#include <cstring>
#include <dlfcn.h>
#include <string>
namespace rad {
using Ds41IndexOrigin = const void* (*)(RadLaunchFn);
inline const void* ds41_index_module(RadLaunchFn launch) {
    Dl_info info{};
    return launch && dladdr(reinterpret_cast<void*>(launch), &info) && info.dli_fbase ? info.dli_fbase : nullptr;
}
inline const char* ds41_index_known_row(const std::string& op) {
    if (op == "index_score") return "ds41_index_score_d128_bf16";
    if (op == "index_score_kvpack") return "ds41_index_score_d128_kvpack";
    if (op == "index_candidates") return "ds41_index_candidates";
    if (op == "index_topk") return "ds41_index_topk";
    return nullptr;
}
/* Called ONLY after ordinary LCP admission proved a routed twin reaches the real chunk. The
 * registry records origin as plugin name/order, not a Plugin pointer. dladdr adds actual DSO-base
 * equality so a sparse override cannot inherit another library's LCP pin claim by spelling.
 * The injectable origin lookup is only for metadata-only host refusal cases. */
inline bool ds41_index_memory_admit(const Program& prog, std::string* why,
                                   Ds41IndexOrigin origin = ds41_index_module) {
    auto refuse = [&](const char* message) { if (why) *why = message; return false; };
    if (!origin) return refuse("no loaded-library origin lookup");
    const KernelRow* twin = nullptr;
    const void* module = nullptr;
    for (const auto& op : prog.ops) for (const auto& b : op.bands) {
        const KernelRow* row = b.dom[RAD_DOMAIN_DEVICE].row;
        if (op.op != "moe_gemm_ggml" || !row || !row->info || !row->info->name) continue;
        const char* name = row->info->name;
        if (std::strcmp(name, "moe_gemm_ggml_mxfp4_gemv_tall") &&
            std::strcmp(name, "moe_gemm_ggml_mxfp4_gemv_tall_p1")) continue;
        const void* here = origin(row->info->launch);
        if (row->plugin != "libr4d" || !here)
            return refuse("admitted LCP twin has no proven libr4d loaded-library origin");
        if (twin && (module != here || row->plugin != twin->plugin || row->plugin_order != twin->plugin_order))
            return refuse("admitted LCP twins come from different loaded libraries");
        twin = row; module = here;
    }
    if (!twin) return refuse("no admitted routed LCP twin establishes the scalar pin library");
    int scores = 0, selections = 0;
    for (const auto& op : prog.ops) {
        const char* expected = ds41_index_known_row(op.op);
        if (!expected) continue;
        if (op.bands.empty()) return refuse("index op has no resolved bands");
        for (const auto& b : op.bands) {
            const Resolved& r = b.dom[RAD_DOMAIN_DEVICE];
            const KernelRow* row = r.row;
            if (!row || !row->info || !row->info->name || std::strcmp(row->info->name, expected))
                return refuse("index op resolves an unknown or missing device kernel");
            if (row->plugin != twin->plugin || row->plugin_order != twin->plugin_order ||
                !row->info->launch || origin(row->info->launch) != module)
                return refuse("index op is not from the admitted LCP twin's loaded library");
            if (row->info->tunables || row->info->n_tunables || row->info->tune_valid ||
                !r.choice.empty() || (!r.tune_source.empty() && r.tune_source != "axis defaults"))
                return refuse("index op has unproven tuned axes or a nondefault point");
        }
        scores += op.op == "index_score" || op.op == "index_score_kvpack";
        selections += op.op == "index_topk";
    }
    if (!scores || !selections) return refuse("program has no complete index-score/topk chain");
    if (why) why->clear();
    return true;
}
} // namespace rad
