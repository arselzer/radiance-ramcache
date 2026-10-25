#pragma once
// Experimental internal capability. No CLI/Engine Loader activation or real checkpoint claim.
#include "record_owner.h"
#include "native_weight_authority.h"
#include "native_device_profile.h"
#include <memory>
#include <span>
#include <string>
#include <vector>
namespace rad {
struct NativeInventoryLimits {
    uint32_t max_rows=1,top_k=8;
    size_t max_owned_units=16384,max_classes=64,max_regions=512;
    uint64_t max_device_bytes=256u<<20,max_landing_bytes=256u<<20,max_weights_bytes=512u<<20;
    uint64_t max_host_bytes=512u<<20; // HOT+COLD+landing+this inventory's metadata reservation
    uint64_t max_metadata_bytes=32u<<20,hot_bytes=0,cold_bytes=64u<<20;
    NativeBackendProfile backend_profile=NativeBackendProfile::HostOnly;
};
struct NativeClassReservation {
    int32_t exemplar=-1;
    int32_t first_region=0,slots=0;
    int64_t stride=0,region_bytes=0,allocation_bytes=0;
};
struct NativeOwnedUnit {
    int32_t unit=-1,klass=-1;
    uint32_t record=0;
    NativeRecordDescriptor descriptor;
};
struct NativeBackingFootprint {
    uint64_t original_static_bytes=0,resident_expert_bytes=0;
    uint64_t native_device_bytes=0,landing_bytes=0,native_payload_bytes=0;
    uint64_t cold_union_bytes=0,hot_bytes=0,cold_bytes=0,metadata_reservation=0;
    size_t owned_units=0,regions=0,consumer_slots=0;
    uint64_t weights_minimum()const{return (original_static_bytes+4095)/4096*4096+native_device_bytes;}
};
class NativeInventory {
public:
    // metadata_plan is a pure truthful checkpoint-style all_vram layout, no allocation.
    // HOST is the default; TinyAqlV1 consumes its separately verified closed capability.
    // Every actual expert unit must be EITHER owned OR explicitly listed as fixed resident.
    // Program identity is pointer identity plus a cloned exact native metadata snapshot.
    static int bind(const Program&,const Plan& metadata_plan,
                    std::span<const NativeRecordDescriptor> catalog,
                    std::span<const RecordOwnerBinding> owned,
                    std::span<const int32_t> resident_units,
                    const NativeInventoryLimits&,std::shared_ptr<const NativeInventory>* out,
                    std::string* why=nullptr);
    ~NativeInventory();
    NativeInventory(const NativeInventory&)=delete;
    int validate(const Program&,const Plan& unit_layout,std::string* why=nullptr)const;
    int validate_runtime_header(const Program&,const Plan&)const;
    int validate_unit(const Program&,const Plan&,int32_t unit)const;
    int32_t unit_for_weight(rad_weight)const noexcept;
    int configure_pools(const Config& base,Config* out)const;
    const Program* program()const noexcept;
    std::span<const NativeOwnedUnit> owned()const noexcept;
    std::span<const NativeClassReservation> classes()const noexcept;
    const NativeOwnedUnit* find(int32_t unit)const noexcept;
    const NativeBackingFootprint& footprint()const noexcept;
    const NativeAuthorityStats& authority_stats()const noexcept;
    const NativeInventoryLimits& limits()const noexcept;
private:
    struct Impl;
    explicit NativeInventory(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> p_;
};
} // namespace rad
