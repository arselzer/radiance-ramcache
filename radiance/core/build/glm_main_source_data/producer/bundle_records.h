#pragma once
// Offline DATA assembly only. No NativeRecordSource/Engine/Ready capability.
#include "main_records.h"
namespace rad::glm_main_bundle {
using Status=glm_main_records::Status;
using Key=glm_main_records::Key;
enum class Mode {SubsetComponent,CompleteMainData};
struct Limits {
    uint64_t max_total_bytes=32u<<20; // canonical segment payload only; disk metadata is separate
    uint64_t max_metadata_bytes=256u<<20; // requested RAM planning allowance, not RSS quota
    size_t max_records=8,io_chunk_bytes=65536,max_input_fds=2;
    size_t max_index_bytes=16u<<20,max_header_bytes=4u<<20,max_config_bytes=2u<<20;
    size_t max_manifest_bytes=4u<<20,max_proof_bytes=131072;
};
struct Record {
    glm_main_records::RecordInfo native;
    uint32_t shard=0;
    uint64_t component_source_id=0,spec_checksum=0;
};
struct Shard {
    std::string name;
    uint32_t rank=0;int32_t layer=-1;
    uint64_t bytes=0,manifest_checksum=0;
    size_t first=0,count=0;
};
struct RankLayout {
    uint32_t world=0,rank=0;
    uint64_t stride=0;
    std::array<int64_t,4> offsets{},bytes{};
    std::array<uint32_t,4> dtype{};
    std::array<std::array<int64_t,2>,4> shape{},span{};
    bool operator==(const RankLayout&)const=default;
};
struct Receipt {
    uint64_t bundle_id=0,total_bytes=0,selected_copy_bytes=0,complete_tensor_read_bytes=0;
    uint64_t source_band_read_bytes=0;
    uint64_t metadata_reserved=0,source_proof_hashed_bytes=0;
    uint64_t metadata_written_bytes=0;
    uint64_t metadata_output_upper_bytes=0;
    size_t records=0,shards=0,graph_quotes=0,peak_input_fds=0,io_window_bytes=0;
    bool complete_main_domain=false,source_proof_complete=false;
};
struct Capture;
class PublishedBundle {
public:
    PublishedBundle()=default;
    Receipt receipt()const noexcept;
    std::span<const Record> records()const noexcept;
    std::span<const Shard> shards()const noexcept;
    std::span<const RankLayout> layouts()const noexcept;
    // complete_main_domain is a DATA-coverage fact only. Runtime admission also
    // needs the real nonexpert model recipe, loader, fit and TP lifecycle gates.
private:
    friend class Assembler;
    friend Status verify(const std::string&,const PublishedBundle&,const Limits&,std::string*)noexcept;
    std::shared_ptr<const Capture> capture_;
};
class Assembler {
public:
    ~Assembler();Assembler(const Assembler&)=delete;
    // Full mode requires a trusted fresh parent-run complete source_proof.py
    // transcript matching the pinned72-file inventory and current file stamps.
    // C++ does NOT claim it computed those full payload cryptographic hashes.
    static Status begin(rad::Checkpoint&,const std::string& source_proof,Mode,
                        const std::string& NEW_destination,const Limits&,
                        std::unique_ptr<Assembler>*,std::string* why=nullptr)noexcept;
    // Exactly once per logical rank, in0..3 order. The SAME Checkpoint object
    // passed to begin must remain alive/immutable through finish; its address
    // is checked, and every copied band is compared with the current raw map.
    // No Program/Plan pointer survives append_rank. Graph/Plan borrows end before return;
    // the actual full graph/source/layout quote runs ONCE here, never per expert.
    Status append_rank(rad::Checkpoint&,const rad::CheckpointSources&,const rad::Program&,
                       const rad::Plan&,std::string* why=nullptr)noexcept;
    // Atomic NOREPLACE. Refuses anything short of all four expected catalogs.
    // Subset mode is exactly(layer3,expert0)+(layer44,expert1) on each rank and
    // always emits non-complete/nonReady provenance, regardless source proof.
    Status finish(PublishedBundle*,std::string* why=nullptr)noexcept;
private:
    struct Impl;explicit Assembler(std::unique_ptr<Impl>);std::unique_ptr<Impl> p_;
};
// Strict postpublication DATA check against the independently retained capture:
// exact manifest hashes/specs/keys/offsets, record checksums and ALL padding.
// Bounded sequential scan, no cache/provider/RuntimeReady conversion.
Status verify(const std::string&,const PublishedBundle&,const Limits&,std::string* why=nullptr)noexcept;
} // namespace rad::glm_main_bundle
