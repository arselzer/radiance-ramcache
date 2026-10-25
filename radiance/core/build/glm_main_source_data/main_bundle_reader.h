#pragma once
// Strict persisted REAL main-expert DATA authority. No Session/Engine admission.
#include "geometry/main_metadata.h"
#include "storage/candidate_native_record_store.h"
#include "place/record_owner.h"
#include "place/native_weight_authority.h"
#include <memory>
#include <span>
namespace rad::glm_main_data {
namespace native=rad::glm_main_native_store;
enum class Error {ok,invalid,budget,identity,io,checksum,no_memory,unsupported,closed};
const char* name(Error)noexcept;
struct ReaderLimits {
    uint64_t metadata_bytes=256u<<20,rank_metadata_bytes=128u<<20,authority_bytes=64u<<20;
    size_t manifest_bytes=4u<<20,proof_bytes=131072,index_bytes=16u<<20;
    size_t header_bytes=4u<<20,window_bytes=65536,max_json_nodes=300000;
};
struct Request {std::string bundle_root,source_proof;uint64_t expected_bundle_id=0;};
struct ReaderReceipt {
    uint64_t bundle_id=0,source_content_id=0,metadata_reservation=0,immutable_metadata_reservation=0;
    uint64_t native_bytes_verified=0,mapped_source_bytes_compared=0;
    uint32_t records=0,shards=0,raw_tensors=0;
    uint32_t owned_fd_upper=8; // simultaneous package bound; caller Checkpoint FDs separate
    bool complete_main_domain=false; // DATA fact only, never RuntimeReady
};
struct RankReceipt {
    uint64_t bundle_id=0,metadata_reservation=0,geometry_peak=0,source_cut_bytes_hashed=0;
    uint64_t immutable_metadata_reservation=0,retained_bundle_reservation=0;
    uint32_t rank=0,records=0;
    rad::NativeAuthorityStats compact_authority;
};
struct VerifiedShard {
    std::string name;
    std::shared_ptr<const native::VerifiedCatalog> catalog;
};
class VerifiedBundle;
class VerifiedRankMain final {
public:
    ~VerifiedRankMain();VerifiedRankMain(const VerifiedRankMain&)=delete;
    const rad::Program* program()const noexcept;
    const rad::Plan* metadata_plan()const noexcept;
    const std::string& root()const noexcept;
    native::FileStamp root_stamp()const noexcept;
    native::FileStamp bundle_stamp()const noexcept;
    native::FileStamp proof_stamp()const noexcept;
    uint64_t generation()const noexcept;
    uint32_t rank()const noexcept;
    std::span<const VerifiedShard> shards()const noexcept;
    std::span<const rad::NativeRecordDescriptor> descriptors()const noexcept;
    std::span<const rad::RecordOwnerBinding> owned()const noexcept;
    // Fixed canonical ordinal->actual unit; no repeated linear key search.
    std::span<const int32_t> units()const noexcept;
    const RankReceipt& receipt()const noexcept;
    Error audit()const noexcept;
    Error validate_program()const noexcept;
private:
    friend class VerifiedBundle;
    struct Impl;explicit VerifiedRankMain(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> p_;
};
class VerifiedBundle final:public std::enable_shared_from_this<VerifiedBundle> {
public:
    ~VerifiedBundle();VerifiedBundle(const VerifiedBundle&)=delete;
    // Complete-only: exact trusted expected ID plus persisted producer proof.
    // Current same-root physical sources are compared with the SAME opened
    // Checkpoint's raw maps once. Checkpoint+cache+Program+metadata Plan must
    // remain immutable/alive until all open sources are closed. This scan is
    // actual I/O when explicitly invoked by parent, not a metadata-only call.
    static Error open(const Request&,rad::Checkpoint&,const ReaderLimits&,
                      std::shared_ptr<const VerifiedBundle>*,std::string* why=nullptr)noexcept;
    // Successful binding is once/rank, serialized, with one identical source cache.
    Error bind_rank(rad::Checkpoint&,const rad::CheckpointSources&,const rad::Program&,
                    const rad::Plan&,std::shared_ptr<const VerifiedRankMain>*,
                    std::string* why=nullptr)const noexcept;
    const ReaderReceipt& receipt()const noexcept;
    Error audit()const noexcept;
private:
    friend class VerifiedRankMain;
    struct Impl;explicit VerifiedBundle(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> p_;
};
} // namespace rad::glm_main_data
