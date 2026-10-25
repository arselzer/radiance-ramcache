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
namespace rad::expert_store_candidate {
enum class Status {ok,invalid,budget,io,checksum,identity,exists,no_memory,unsupported};
const char* name(Status) noexcept;
enum class RecordClass {main_mxfp4_e8m0,mtp_fp8_f32};
enum class Projection {gate_up,down};
struct PlaneFile {std::string path;uint64_t offset=0,bytes=0,checksum=0;};
struct SourceTensor {
    std::string logical;Projection projection=Projection::gate_up;RadEncoding encoding{};
    std::array<int64_t,2> shape{};std::array<PlaneFile,2> planes{};
};
struct SourceCatalog {
    // Only synthetic provenance is supported. These are NOT claims about checkpoint headers.
    std::string source_id,revision;std::vector<SourceTensor> tensors;
};
struct Limits {
    uint64_t max_total_bytes=8u<<20,max_segment_bytes=96u<<10,max_record_bytes=1u<<20;
    uint64_t max_source_bytes=2u<<20,max_read_bytes=1u<<20;
    size_t max_manifest_bytes=1u<<20,max_records=128,max_sources=64,max_segments=128;
    size_t max_string_bytes=512,io_chunk_bytes=65536,max_manifest_nodes=32768;
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
    friend Status capture_unit(const Program&,const Plan&,size_t,const SourceCatalog&,const Limits&,RecordSpec*,std::string*);
};
struct Receipt {uint64_t manifest_checksum=0,total_bytes=0,source_checksum=0;size_t records=0,segments=0,peak_io_buffer=0;};
// Source metadata/products/caps are checked before calling actual weight_share. The result
// snapshots actual WeightInfo/MoveUnit layouts, but does not assert native R4D row resolution.
Status capture_unit(const Program&,const Plan&,size_t unit,const SourceCatalog&,RecordSpec*,std::string* why=nullptr);
Status capture_unit(const Program&,const Plan&,size_t unit,const SourceCatalog&,const Limits&,RecordSpec*,std::string* why=nullptr);
// Real selected source bytes are verified and copied losslessly before atomic NOREPLACE
// publication. Only a private task-owned sibling temporary directory is cleaned on failure.
Status pack(std::span<const RecordSpec>,const SourceCatalog&,const std::string& destination,const Limits&,Receipt*,std::string* why=nullptr);
// Metadata exported ONLY after the unchanged S1 parser, checksum and padding checks.
// No payload/manifest/directory FD is retained by this immutable normalized object.
struct FileStamp {
    uint64_t device=0,inode=0,bytes=0;
    int64_t mtime_sec=0,mtime_nsec=0,ctime_sec=0,ctime_nsec=0;
    bool operator==(const FileStamp&)const=default;
};
struct VerifiedSegment {std::string name;FileStamp stamp;uint32_t rank=0;};
struct VerifiedRecord {
    Key key;uint32_t segment=0;uint64_t offset=0,length=0,checksum=0;
    std::array<uint64_t,4> member_offsets{},member_bytes{};
};
class VerifiedCatalog {
public:
    FileStamp directory_stamp()const noexcept{return directory_;}
    FileStamp manifest_stamp()const noexcept{return manifest_;}
    std::span<const VerifiedSegment> segments()const noexcept{return segments_;}
    std::span<const VerifiedRecord> records()const noexcept{return records_;}
    const Receipt& receipt()const noexcept{return receipt_;}
private:
    friend class Reader;
    VerifiedCatalog()=default;
    FileStamp directory_,manifest_;Receipt receipt_;
    std::vector<VerifiedSegment> segments_;std::vector<VerifiedRecord> records_;
};
class Reader {
public:
    ~Reader();Reader(const Reader&)=delete;Reader& operator=(const Reader&)=delete;
    // Requires fresh expected Spec/catalog identity and the independently retained publication
    // checksum. Verifies segments, every record checksum and ALL padding by bounded streaming.
    static Status open(const std::string&,std::span<const RecordSpec>,const SourceCatalog&,uint64_t expected_manifest_checksum,const Limits&,std::unique_ptr<Reader>*,std::string* why=nullptr);
    // Allocates at most one record under max_read_bytes, leaves *out unchanged on all failures.
    Status read(const Key&,std::vector<std::byte>* out,std::string* why=nullptr)const;
    const Receipt& receipt()const noexcept;
    // Rechecks every pinned descriptor and the directory/manifest after the complete startup
    // scan. On error *out is unchanged; no independently mutable/ready catalog is returned.
    Status export_catalog(std::shared_ptr<const VerifiedCatalog>* out,std::string* why=nullptr)const;
private:
    struct Impl;explicit Reader(std::unique_ptr<Impl>);std::unique_ptr<Impl> impl_;
};
uint64_t checksum(std::span<const std::byte>)noexcept; // FNV1a64 integrity, NOT authentication.
} // namespace rad::expert_store_candidate
