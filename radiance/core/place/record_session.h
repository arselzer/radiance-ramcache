#pragma once
#include "rad_device.h"
#include <cstdint>
#include <span>
#include <vector>
namespace rad {
// Fixed consumer allocations registered once, outside any MoveUnit-owned allocation.
// exemplar identifies the actual Program/Plan native member layout; no byte-size-only class cast.
struct RecordRegion {int32_t exemplar=-1;void* base=nullptr;int64_t bytes=0;RadEvent ready=nullptr;bool sentinel=false;};
struct RecordClaim {int32_t unit=-1, sentinel_region=-1;};
struct RecordPublication {int32_t unit=-1,region=-1;};
struct RecordSession {
    uint64_t token=0;
    std::vector<RecordRegion> regions;
    std::vector<RecordClaim> claims;
    std::vector<RecordPublication> published; // bounded live native slice only
    bool native_only=false,quiesced=true,upload_failed=false;
};
} // namespace rad
