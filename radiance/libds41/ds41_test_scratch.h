/* Checker-only scratch bytes. Native payload flag/guards occupy the first16B;
 * only actual Plan bytes may change after that. The rounded tail stays a canary. */
#pragma once
#include "ds41_device_plan.h"
#include <vector>
#include <stdexcept>
namespace rad::ds41::testing {
template<class Plan>std::vector<uint8_t> make_gpu_scratch(const std::vector<uint8_t>& prefix) {
    std::vector<uint8_t> out(size_t(gpu_scratch_bytes<Plan>())+16,0xAB);
    if(prefix.size()>device_plan_offset)throw std::runtime_error("status fixture exceeds reserved prefix");
    std::memcpy(out.data(),prefix.data(),prefix.size());return out;
}
template<class Plan>bool gpu_scratch_guards(const std::vector<uint8_t>& after,const std::vector<uint8_t>& before) {
    if(after.size()!=before.size()||after.size()<size_t(gpu_scratch_bytes<Plan>()))return false;
    for(size_t i=4;i<device_plan_offset;++i)if(after[i]!=before[i])return false;
    for(size_t i=device_plan_offset+sizeof(Plan);i<after.size();++i)if(after[i]!=before[i])return false;
    return true;
}
} // namespace rad::ds41::testing
