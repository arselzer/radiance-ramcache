#pragma once
// Offline, synchronous actual-source packing. No NativeInventory/source/Ready API.
#include "geometry/main_metadata.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>
namespace glm53::main_records {
enum class Status {ok,invalid,budget,identity,io,checksum,exists,unsupported,no_memory};
const char* name(Status) noexcept;
struct Limits {
    // First actual-payload gate: ONE complete expert for ONE logical rank.
    // Four calls can independently prove all TP4 cuts without a full store.
    size_t max_records=1,io_chunk_bytes=65536,max_input_fds=2;
    size_t max_index_bytes=16u<<20,max_header_bytes=4u<<20,max_manifest_bytes=1u<<20;
    size_t max_config_bytes=2u<<20; // actual pinned config1716060B; explicit bounded Quark override metadata
    uint64_t max_source_file_bytes=8ull<<30,max_selected_tensor_bytes=16u<<20;
    uint64_t max_record_bytes=4u<<20,max_read_bytes=4u<<20,max_metadata_bytes=128u<<20;
};
struct Key {uint32_t world=0,rank=0;int32_t layer=-1,expert=-1;bool operator==(const Key&)const=default;};
struct FileStamp {
    uint64_t device=0,inode=0,bytes=0;
    int64_t mtime_sec=0,mtime_nsec=0,ctime_sec=0,ctime_nsec=0;
    bool operator==(const FileStamp&)const=default;
};
struct RecordInfo {
    Key key;uint64_t offset=0,length=0,checksum=0;
    std::array<uint64_t,4> member_offsets{},member_bytes{};
};
struct Receipt {
    uint64_t manifest_checksum=0,source_generation=0,total_bytes=0;
    uint64_t complete_tensor_read_bytes=0,selected_copy_bytes=0,metadata_reserved=0;
    uint64_t source_config_bytes=0,source_index_bytes=0;
    size_t records=0,physical_tensors=0,peak_io_buffer=0,peak_input_fds=0;
    // Named simultaneous descriptor upper bound: 2 payload +6 fixed metadata/
    // output handles. Caller Checkpoint mapping descriptors are separate.
    size_t owned_fd_bound=8;
};
struct Capture;
class VerifiedCatalog {
public:
    FileStamp directory_stamp()const noexcept{return directory_;}
    FileStamp manifest_stamp()const noexcept{return manifest_;}
    FileStamp segment_stamp()const noexcept{return segment_;}
    const RecordInfo& record()const noexcept{return record_;}
    const Receipt& receipt()const noexcept{return receipt_;}
private:
    friend class MainReader;VerifiedCatalog()=default;
    FileStamp directory_,manifest_,segment_;RecordInfo record_;Receipt receipt_;
};
class PublishedRecords {
public:
    PublishedRecords()=default;
    // source_generation is the COMMON selected_raw_bundle identity across
    // rank cuts of this expert, NOT a full-model/NativeInventory generation.
    // manifest_checksum and Reader record checksum bind rank-local geometry.
    Receipt receipt()const noexcept;
    // Intentionally not an acquirable source, native descriptor catalog or
    // claim of complete domain coverage. Private captured recipes own all
    // strings/arrays and retain NO borrowed Program/Plan/Checkpoint pointers.
private:
    friend Status pack_main_records(rad::Checkpoint&,const rad::CheckpointSources&,const rad::Program&,const rad::Plan&,
        std::span<const int32_t>,const std::string&,const Limits&,PublishedRecords*,std::string*)noexcept;
    friend class MainReader;
    std::shared_ptr<const Capture> capture_;
};
// Requotes REAL selected geometry against the exact still-live source/cache/
// graph/plan, resolves physical header ranges, compares all selected COMPLETE
// tensor bytes to those mapped sources, then packs only those rank-local cuts.
// All input borrows end before return. Atomic NOREPLACE; caller unchanged on
// refusal. No full checkpoint/model/store payload operation is implicit.
Status pack_main_records(rad::Checkpoint&,const rad::CheckpointSources&,const rad::Program&,const rad::Plan&,
    std::span<const int32_t> selected_units,const std::string& new_destination,const Limits&,
    PublishedRecords*,std::string* why=nullptr)noexcept;
class MainReader {
public:
    ~MainReader();MainReader(const MainReader&)=delete;MainReader& operator=(const MainReader&)=delete;
    // Uses independently retained private publication capture, not trusted
    // mutable JSON/geometry. Old synthetic Reader schemas remain closed.
    static Status open(const std::string&,const PublishedRecords&,const Limits&,
                       std::unique_ptr<MainReader>*,std::string* why=nullptr)noexcept;
    Status read(const Key&,std::vector<std::byte>*,std::string* why=nullptr)const noexcept;
    const Receipt& receipt()const noexcept;
    // Normalized verified DATA records, only after full checksum/padding scan.
    // These values alone are not a runtime source/Ready/Native capability.
    std::span<const RecordInfo> records()const noexcept;
    // Copies normalized verified stamps/record only; retains NO FDs, recipe
    // borrows or source mappings. No Native/runtime capability is implied.
    Status export_catalog(std::shared_ptr<const VerifiedCatalog>*,std::string* why=nullptr)const noexcept;
private:
    struct Impl;explicit MainReader(std::unique_ptr<Impl>);std::unique_ptr<Impl> impl_;
};
uint64_t checksum(std::span<const std::byte>)noexcept; // FNV1a integrity, not authentication.
} // namespace glm53::main_records
