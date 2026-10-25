#pragma once
// Offline synthetic record-store prototype. No runtime/provider/GPU/serving integration.
#include "build/rad_build.h"
#include "place/planner.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>
namespace rad::composite_export {
enum class Status {ok,invalid,budget,io,checksum,identity,exists,no_memory,unsupported};
const char* name(Status) noexcept;
enum class RecordClass {main_mxfp4_e8m0,mtp_fp8_f32};
enum class Projection {gate_up,down};
enum class Part {gate,up,down};
// A complete original physical plane, never a saved partial row masquerading as one.
struct PhysicalPlane {
    std::string tensor,path;Part part=Part::gate;
    int32_t layer=-1,expert=-1;uint32_t dtype=RAD_DT_INVALID;
    uint64_t rows=0,row_bytes=0,offset=0,bytes=0,checksum=0;
    bool complete=false;
};
struct CompositePlane {uint64_t bytes=0,checksum=0;std::vector<PhysicalPlane> pieces;};
struct SourceTensor {
    std::string logical;Projection projection=Projection::gate_up;RadEncoding encoding{};
    int32_t layer=-1,expert=-1;
    std::array<int64_t,2> shape{};std::array<CompositePlane,2> planes{};
};
struct SourceCatalog {
    // Only synthetic provenance is supported. These are NOT claims about checkpoint headers.
    std::string source_id,revision;std::vector<SourceTensor> tensors;
};
struct Limits {
    uint64_t max_total_bytes=8u<<20,max_segment_bytes=256u<<10,max_record_bytes=1u<<20;
    uint64_t max_source_bytes=2u<<20,max_read_bytes=1u<<20;
    size_t max_manifest_bytes=1u<<20,max_records=128,max_sources=64,max_segments=4;
    size_t max_string_bytes=512,io_chunk_bytes=4096,max_manifest_nodes=32768;
    size_t max_input_fds=2,max_physical_planes=64;
    uint64_t max_metadata_bytes=16u<<20;
    size_t max_open_fds=10;
};
struct Key {uint32_t world=0,rank=0;int32_t layer=-1,expert=-1;RecordClass record_class=RecordClass::main_mxfp4_e8m0;bool operator==(const Key&)const=default;};
struct SpecData;
class RecordSpec {
public:
    RecordSpec()=default;
    Key key()const;uint64_t stride()const;
    std::array<uint64_t,4> member_offsets()const;
    std::array<uint64_t,4> member_bytes()const;
    std::string metadata_json()const;
    // Immutable internal snapshot: caller cannot mutate a captured layout/rectangle list.
    const SpecData* data()const noexcept{return data_.get();}
private:
    std::shared_ptr<const SpecData> data_;
    friend Status capture_unit(const Program&,const Plan&,size_t,const SourceCatalog&,RecordSpec*,std::string*);
};
struct Receipt {uint64_t manifest_checksum=0,total_bytes=0,source_checksum=0;size_t records=0,segments=0,peak_io_buffer=0;uint64_t metadata_reserved=0;size_t peak_input_fds=0;bool operator==(const Receipt&)const=default;};
// Source metadata/products/caps are checked before calling actual weight_share. The result
// snapshots actual WeightInfo/MoveUnit layouts, but does not assert native R4D row resolution.
Status capture_unit(const Program&,const Plan&,size_t unit,const SourceCatalog&,RecordSpec*,std::string* why=nullptr);
// Real selected source bytes are verified and copied losslessly before atomic NOREPLACE
// publication. Only a private task-owned sibling temporary directory is cleaned on failure.
Status pack(std::span<const RecordSpec>,const SourceCatalog&,const std::string& destination,const Limits&,Receipt*,std::string* why=nullptr);
class Reader {
public:
    ~Reader();Reader(const Reader&)=delete;Reader& operator=(const Reader&)=delete;
    // Requires fresh expected Spec/catalog identity and the independently retained publication
    // checksum. Verifies segments, every record checksum and ALL padding by bounded streaming.
    static Status open(const std::string&,std::span<const RecordSpec>,const SourceCatalog&,uint64_t expected_manifest_checksum,const Limits&,std::unique_ptr<Reader>*,std::string* why=nullptr);
    // Allocates at most one record under max_read_bytes, leaves *out unchanged on all failures.
    Status read(const Key&,std::vector<std::byte>* out,std::string* why=nullptr)const;
    const Receipt& receipt()const noexcept;
private:
    struct Impl;explicit Reader(std::unique_ptr<Impl>);std::unique_ptr<Impl> impl_;
};
uint64_t checksum(std::span<const std::byte>)noexcept; // FNV1a64 integrity, NOT authentication.
} // namespace rad::composite_export
