/* A conversion fixture that describes canonical multi-plane storage. It cannot run inference:
 * launch refuses, and the plugin is kept in testkernels/import, outside the serving scan. */
#include "rad_planes.h"
namespace {
int layout(const RadParam*, int, int, const RadEncoding* enc, const int* sel,
            const RadTensor* planes, int n, RadLayout* out) {
    return rad_planes_layout(enc, sel, planes, n, out);
}
int launch(const RadArgs*, RadStream) { return RAD_E_STATE; }
const RadOperandSpec operands[]{{"weight", RAD_OPD_WEIGHT, 0}};
const RadOpSchema schema{"test_import_read", nullptr, 0, operands, 1, "converter layout fixture"};
const RadKernelInfo kernel{
    .name="test_import_planes", .op="test_import_read", .family="misc",
    .computes="nothing; conversion fixture", .shape="any", .dtypes="canonical planes",
    .domain=RAD_DOMAIN_HOST, .priority=1, .launch=launch, .layout=layout
};
const RadPluginInfo info{RAD_PLUGIN_KERNEL,"radtest_importkernel","0.1","converter layout fixture","host"};
}
extern "C" {
uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
const RadPluginInfo* rad_plugin_info(void) { return &info; }
int rad_kernel_schema_count(void) { return 1; }
const RadOpSchema* rad_kernel_schema_at(int i) { return i == 0 ? &schema : nullptr; }
int rad_kernel_count(void) { return 1; }
const RadKernelInfo* rad_kernel_at(int i) { return i == 0 ? &kernel : nullptr; }
}
