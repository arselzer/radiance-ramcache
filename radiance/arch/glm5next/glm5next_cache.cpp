// Explicit research module, separate target/home; ordinary glm5next is unchanged.
#include "rad_glm_main_tp4_cache.h"
extern "C" uint32_t rad_core_glm_main_tp4_cache_profile_version(void) __attribute__((weak));
extern "C" int rad_require_glm_main_tp4_cache_fail_stop(RadBuilder*) __attribute__((weak));
#define RAD_GLM_MAIN_TP4_CACHE_EXPERIMENTAL 1
#define RAD_ARCH_NO_EXPORTS 1
#include "glm5next.cpp"
namespace glm5next_cache {
static int declare(RadBuilder* b,const RadModelMeta* m,const RadBuildCtx* c){return glm5next::declare(b,m,c);}
static void step(RadCtx* c,const RadBatch* b){glm5next::step(c,b);}
static int probe(const RadModelMeta* m,RadArchProbe* out){const int status=glm5next::probe(m,out);if(status>=0){out->draft_depth=0;out->enc_merge=0;}return status;}
}
RAD_ARCH_PROBE(glm5next_cache)
RAD_ARCH_PLUGIN(glm5next_cache,"glm5next","","experimental-cache-failstop-v1",
                "EXPERIMENTAL source-gated GLM main TP4 cache; process-fail-stop; no MTP/vision")
RAD_ARCH_SERVES_IMPORTS(1)
extern "C" int rad_plugin_open(){return rad_core_glm_main_tp4_cache_profile_version&&rad_require_glm_main_tp4_cache_fail_stop&&
    rad_core_glm_main_tp4_cache_profile_version()>=RAD_GLM_MAIN_TP4_CACHE_PROFILE_VERSION?RAD_OK:RAD_E_ABI;}
