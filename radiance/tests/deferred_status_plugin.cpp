/* Loader-only optional-contract fixtures. DS_STATUS_VARIANT:
 * 0 absent,1 valid/device,2 version-only,3 query-only,4 bad-version,5 host-row,
 * 6 collector plus self-releasing collective (refused transactionally). */
#include "rad_deferred_status.h"
#ifndef DS_STATUS_VARIANT
#define DS_STATUS_VARIANT 0
#endif
namespace {
int launch(const RadArgs*,RadStream){return RAD_OK;}
[[maybe_unused]] int collect(const RadArgs*,RadDeviceStatus*,uint32_t,RadStream){return RAD_OK;}
const RadKernelInfo row=[] {RadKernelInfo r{};r.name="status_fixture";r.op="status_fixture";r.family="test";
    r.domain=DS_STATUS_VARIANT==5?RAD_DOMAIN_HOST:RAD_DOMAIN_DEVICE;r.launch=launch;return r;}();
const RadOpSchema schema{"status_fixture",nullptr,0,nullptr,0,"loader-only status fixture"};
}
extern "C" {
uint32_t rad_plugin_abi_version(){return RAD_ABI_VERSION;}
const RadPluginInfo* rad_plugin_info(){static const RadPluginInfo p{RAD_PLUGIN_KERNEL,"status_fixture","1","optional deferred status fixture","host"};return &p;}
int rad_kernel_count(){return 1;}
const RadKernelInfo* rad_kernel_at(int i){return i==0?&row:nullptr;}
int rad_kernel_schema_count(){return 1;}
const RadOpSchema* rad_kernel_schema_at(int i){return i==0?&schema:nullptr;}
#if DS_STATUS_VARIANT!=0 && DS_STATUS_VARIANT!=3
uint32_t rad_plugin_deferred_status_version(){return DS_STATUS_VARIANT==4?999:RAD_DEFERRED_STATUS_VERSION;}
#endif
#if DS_STATUS_VARIANT!=0 && DS_STATUS_VARIANT!=2
RadDeferredStatusLaunchFn rad_kernel_deferred_status(int i){return i==0?collect:nullptr;}
#endif
#if DS_STATUS_VARIANT==6
int rad_kernel_collective(int i){return i==0?2:0;}
#endif
}
