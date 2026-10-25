#pragma once
// Internal exact immutable authority. No Loader/CLI/backend activation.
#include "rad_core.h"
#include <cstdint>
#include <memory>
#include <span>
#include <string>
namespace rad {
struct MoveUnit;
struct NativeAuthorityStats {
    uint64_t weights=0,geometries=0,strings=0;
    uint64_t name_source_bytes=0,layout_tag_bytes=0,arena_bytes=0;
    uint64_t reference_bytes=0,geometry_bytes=0,slice_bytes=0;
    uint64_t geometry_bucket_count=0,string_bucket_count=0;
    uint64_t scratch_bytes=0,immutable_bytes=0,construction_peak_bytes=0;
    uint64_t metadata_budget=0;
    uint64_t template_string_capacity_excess=0;
    uint32_t reference_size=0,slice_size=0,bucket_size=0,geometry_size=0;
    bool admissible=false;
    bool template_strings_empty=false;
};
// Exactly the OLD native metadata predicate, exposed only in this internal header
// for differential tests. Runtime pointers/site/lifetime-op fields remain ignored.
bool native_weight_metadata_equal(const WeightInfo&,const WeightInfo&,bool names) noexcept;

class NativeWeightAuthority {
public:
    // inspect allocates bounded borrowed-index scratch only. A successful quote
    // may have admissible=false; it never publishes an authority or allocates
    // templates/string storage/pools/payloads. Failure preserves the caller quote.
    static int inspect(std::span<const WeightInfo>,std::span<const int32_t>,
                       std::span<const MoveUnit>,uint64_t metadata_budget,
                       NativeAuthorityStats* out,std::string* why=nullptr);
    // All bulk immutable storage is charged before allocation. Publication is
    // the final noexcept unique_ptr move; failure preserves an existing owner.
    static int capture(std::span<const WeightInfo>,std::span<const int32_t>,
                       std::span<const MoveUnit>,uint64_t metadata_budget,
                       std::unique_ptr<NativeWeightAuthority>* out,std::string* why=nullptr);
    ~NativeWeightAuthority();
    NativeWeightAuthority(const NativeWeightAuthority&)=delete;
    size_t size()const noexcept;
    bool matches(rad_weight,const WeightInfo&)const noexcept;
    int32_t unit_for_weight(rad_weight)const noexcept;
    int32_t member_index(rad_weight)const noexcept;
    int64_t stored_bytes(rad_weight)const noexcept;
    const NativeAuthorityStats& stats()const noexcept;
private:
    struct Impl;
    explicit NativeWeightAuthority(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> p_;
};
} // namespace rad
