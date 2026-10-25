/* Deliberately malformed optional importer interfaces, loaded only by path in CPU tests. */
#include "rad_import.h"
#include "rad_quant.h"
#ifndef IMPORT_CASE
#define IMPORT_CASE 1
#endif
namespace {
int encoding(const RadCheckpointImportWeight*, RadEncoding*) { return RAD_E_UNSUPPORTED; }
int rows(const RadCheckpointImportWeight*, const void*, int64_t, int64_t, void* const*) { return RAD_E_UNSUPPORTED; }
const RadCheckpointImporterInfo info{
#if IMPORT_CASE == 3
    "gguf", "iq2_xs",
#else
    "test-import", "test-type",
#endif
    "malformed importer fixture", 256, 74,
#if IMPORT_CASE == 2
    nullptr,
#else
    encoding,
#endif
    rows
};
const RadPluginInfo plugin{RAD_PLUGIN_QUANT,"radtest_importplugin","0.1","optional importer refusal fixture","host"};
}
extern "C" {
uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
const RadPluginInfo* rad_plugin_info(void) { return &plugin; }
int rad_quant_count(void) { return 0; }
const RadQuantizerInfo* rad_quant_at(int) { return nullptr; }
int rad_checkpoint_import_count(void) {
#if IMPORT_CASE == 4
    return 2;
#elif IMPORT_CASE == 5
    return -1;
#else
    return 1;
#endif
}
#if IMPORT_CASE != 1
const RadCheckpointImporterInfo* rad_checkpoint_import_at(int i) { return i >= 0 && i < rad_checkpoint_import_count() ? &info : nullptr; }
#endif
}
