#pragma once
// Internal experimental candidate only. No Engine/CLI opt-in and no ABI27 change.
#include <cstdint>
namespace rad {
enum class NativeBackendProfile:uint32_t {
    HostOnly=0, TinyAqlV1=1,
    // Separate closed actual-GLM capability. It never downgrades PermanentV1.
    GlmMainTp4AqlFailStopV1=2
};
inline constexpr uint32_t kNativeDeviceOwnerProfileVersion=1;
uint32_t native_device_owner_profile_version()noexcept;
}
