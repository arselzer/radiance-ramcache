#pragma once
#include "rad_deferred_status.h"
#include "rad_deferred_status_capability.h"
extern "C" {
#if defined(__GNUC__)
uint32_t rad_core_deferred_status_version(void) __attribute__((weak));
#endif
int ds41_collect_scratch_status(const RadArgs*,RadDeviceStatus*,uint32_t,RadStream);
int ds41_collect_routed_status(const RadArgs*,RadDeviceStatus*,uint32_t,RadStream);
int ds41_collect_store_status(const RadArgs*,RadDeviceStatus*,uint32_t,RadStream);
int ds41_collect_gather_status(const RadArgs*,RadDeviceStatus*,uint32_t,RadStream);
}
