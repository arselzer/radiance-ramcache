/* Deliberately does not link Core. The real architecture module must refuse
 * both an absent capability and a stale exported capability at dlopen/open. */
#include "rad_model_failure_policy.h"
#include <dlfcn.h>
#include <cstdio>
#ifdef RAD_POLICY_STALE_CORE
extern "C" uint32_t rad_core_model_failure_policy_version(void) { return 0; }
extern "C" int rad_require_permanent_model_failure(RadBuilder*) { return RAD_OK; }
#endif
int main(int argc,char** argv) {
    if (argc != 2) return 2;
    void* module=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
    if (!module) { std::fprintf(stderr,"dlopen: %s\n",dlerror());return 3; }
    auto open=reinterpret_cast<int(*)()>(dlsym(module,"rad_plugin_open"));
    const int status=open ? open() : RAD_E_STATE;
    dlclose(module);
    if (status != RAD_E_ABI) { std::fprintf(stderr,"expected old-Core ABI refusal, got %d\n",status);return 4; }
    return 0;
}
