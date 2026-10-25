// Test-only DEVICE metadata row. Collector+Native Owner must be refused at Ctx
// admission, so neither launch below may execute in this HOST checker.
#include "rad_deferred_status.h"
#include <atomic>
namespace {
std::atomic<uint64_t> bodies{0},collectors{0};
int body(const RadArgs*,RadStream){++bodies;return RAD_OK;}
int collect(const RadArgs*,RadDeviceStatus*,uint32_t,RadStream){++collectors;return RAD_OK;}
const RadKernelInfo row=[] {RadKernelInfo r{};r.name="unified_status_fixture";r.op="unified_status_fixture";
    r.family="test";r.domain=RAD_DOMAIN_DEVICE;r.launch=body;return r;}();
const RadOpSchema schema{"unified_status_fixture",nullptr,0,nullptr,0,"never-launched admission-only status fixture"};
}
extern "C" {
uint32_t rad_plugin_abi_version(){return RAD_ABI_VERSION;}
const RadPluginInfo* rad_plugin_info(){static const RadPluginInfo p{RAD_PLUGIN_KERNEL,"unified_status_fixture","1","admission-only collector fixture","host"};return &p;}
int rad_kernel_count(){return 1;}
const RadKernelInfo* rad_kernel_at(int i){return i==0?&row:nullptr;}
int rad_kernel_schema_count(){return 1;}
const RadOpSchema* rad_kernel_schema_at(int i){return i==0?&schema:nullptr;}
uint32_t rad_plugin_deferred_status_version(){return RAD_DEFERRED_STATUS_VERSION;}
RadDeferredStatusLaunchFn rad_kernel_deferred_status(int i){return i==0?collect:nullptr;}
uint64_t unified_status_bodies(){return bodies.load();}
uint64_t unified_status_collectors(){return collectors.load();}
}
