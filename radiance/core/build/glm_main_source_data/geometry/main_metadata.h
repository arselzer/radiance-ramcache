#pragma once
// Research metadata only; deliberately no NativeRecordSource/Ready/checksum API.
// Use the SAME Core headers/archive as the caller's actual finished stock graph.
#include "build/startup.h"
#include "place/planner.h"
#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>
namespace rad::glm_main_metadata {
enum class Projection:uint32_t {Gate=0,Up=1,Down=2};
// Rectangle in an actual separately named raw U8 checkpoint tensor. The names
// are checked against the stock layer/expert/projection contract, not inferred
// from contiguous offsets. No borrowed mapped pointer escapes this quote.
struct PhysicalSlice {
    Projection projection=Projection::Gate;
    uint32_t plane=0,raw_dtype=0;
    std::array<int64_t,2> raw_shape{};
    int64_t raw_bytes=0,row0=0,rows=0,byte0=0,row_bytes=0,bytes_per_row=0;
};
struct MemberLayout {
    rad_weight weight=0;
    uint32_t plane=0,slices=0;
    std::array<PhysicalSlice,2> physical{};
};
// Geometry only. This intentionally lacks generation, payload checksum, lease,
// source/catalog capability, native inventory binding or a readiness boolean.
struct RecordLayout {
    uint32_t world=0,rank=0;
    int32_t unit=-1,layer=-1,expert=-1;
    int64_t stride=0;
    std::array<int64_t,4> offsets{},bytes{};
    std::array<uint32_t,4> dtype{};
    std::array<std::array<int64_t,2>,4> shape{};
    std::array<MemberLayout,4> members{};
};
struct OwnedLayout {int32_t unit=-1;uint32_t record=0;};
struct Limits {
    uint64_t max_metadata_bytes=64u<<20;
    uint64_t max_program_weights=100000,max_program_units=65536,max_name_maps=100000;
};
struct Quote {
    uint32_t rank=0,world=0;
    uint64_t input_weights=0,input_units=0,input_name_maps=0,main_units=0;
    uint64_t stored_bytes=0,source_selected_bytes=0,physical_rectangles=0;
    uint64_t immutable_requested=0,scratch_requested=0,construction_requested=0;
    uint64_t sizeof_record=0,sizeof_member=0,sizeof_slice=0,sizeof_owned=0;
    std::vector<RecordLayout> records; // canonical (layer,expert) order, all main experts
    std::vector<OwnedLayout> owned;   // records excluding the explicit resident mask
    std::vector<int32_t> resident_main_units;
};
// Closed first profile: actual stock GLM main MXFP4/E8M0, TP4 ranks0..3,
// max_spec0, main layers3..44/E288/top8, canonical identity rank2 layouts.
// Program/Plan/source cache must remain immutable throughout this call. The
// actual opened Checkpoint's imported objects and raw mapped spans bind cache
// provenance; no payload is read, copied, hashed, converted or made acquirable.
// Output is assigned only after every check succeeds. This quotes adapter-owned
// metadata storage and bounded scratch, not caller Checkpoint/Program/RSS/driver.
int quote(rad::Checkpoint&,const rad::CheckpointSources&,const rad::Program&,
          const rad::Plan&,std::span<const int32_t> resident_main_units,
          const Limits&,Quote* out,std::string* why=nullptr) noexcept;
} // namespace rad::glm_main_metadata
