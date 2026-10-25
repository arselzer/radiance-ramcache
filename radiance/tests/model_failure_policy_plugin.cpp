/* Architecture-style loader fixture. No serving graph or fake kernel rows. */
#include "rad_model_failure_policy.h"
extern "C" uint32_t rad_core_model_failure_policy_version(void) __attribute__((weak));
extern "C" int rad_require_permanent_model_failure(RadBuilder*) __attribute__((weak));
extern "C" {
uint32_t rad_plugin_abi_version() { return RAD_ABI_VERSION; }
const RadPluginInfo* rad_plugin_info() {
    static const RadPluginInfo p{RAD_PLUGIN_ARCH,"model_policy_fixture","1",
                                "permanent model failure capability fixture","host"};
    return &p;
}
int rad_plugin_open() {
    if (!rad_core_model_failure_policy_version || !rad_require_permanent_model_failure ||
        rad_core_model_failure_policy_version() < RAD_MODEL_FAILURE_POLICY_VERSION)
        return RAD_E_ABI; // old Core must refuse, not silently ignore a new optional export
    return RAD_OK;
}
const char* rad_arch_id() { return "model_policy_fixture"; }
int rad_arch_declare(RadBuilder* b,const RadModelMeta*,const RadBuildCtx*) {
    if (!rad_require_permanent_model_failure) return RAD_E_ABI;
    return rad_require_permanent_model_failure(b);
}
void rad_arch_step(RadCtx*,const RadBatch*) { } // loader-only; not advertised as serving
}
