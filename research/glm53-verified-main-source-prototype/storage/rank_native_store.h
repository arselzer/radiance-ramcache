#pragma once
// CPU-only storage/cache mechanics with separate synthetic and opaque verified
// actual-main admission. No Engine/CLI/GPU/TP execution integration.
#include "candidate_native_record_store.h"
#include <memory>
#include <span>
#include <stop_token>
#include <string>
#include <vector>
namespace glm53::verified_main {class VerifiedRankMain;}
namespace glm53::sharded {
namespace native=rad::expert_store_candidate;
enum class Error {ok,invalid,budget,identity,io,checksum,cancelled,closed,reentrant,no_memory,unsupported,exhausted};
const char* name(Error)noexcept;
enum class IntegrityPolicy {ImmutableGenerationSelectedAudit};
struct ProcessLimits {
    size_t max_rank_owners=4,max_open_fds=32;
    uint64_t max_metadata_bytes=256u<<20,max_cache_reservations=64u<<20;
};
struct ProcessUsage {
    size_t rank_owners=0,open_fd_credits=0,peak_fd_credits=0;
    uint64_t metadata_reserved=0,peak_metadata_reserved=0,cache_reserved=0;
};
struct Limits {
    uint64_t hot_bytes=128u<<10,cold_bytes=512u<<10,max_record_bytes=1u<<20;
    uint64_t max_metadata_bytes=128u<<20,max_transactions=UINT64_MAX-1;
    size_t max_records=16384,max_batch=1024,max_shards=64,max_segments=256;
    size_t max_payload_fds=2,max_rank_fd_credits=12,max_live_records=1024,max_live_leases=8192;
    size_t io_chunk_bytes=65536;
    IntegrityPolicy integrity=IntegrityPolicy::ImmutableGenerationSelectedAudit;
    // This initial profile is buffered POSIX I/O; external OS page-cache memory is excluded.
    bool buffered_io=true;
};
namespace detail {struct Process;struct Generation;struct Ledger;struct Record;struct StoreImpl;struct BatchCredit;}
class ProcessBudget {
public:
    static Error create(const ProcessLimits&,std::shared_ptr<ProcessBudget>* out)noexcept;
    ProcessUsage usage()const noexcept;
private:
    friend struct detail::StoreImpl;
    friend class RankNativeStore;
    explicit ProcessBudget(std::shared_ptr<detail::Process>);
    std::shared_ptr<detail::Process> state_;
};
struct ShardInput {
    // One safe relative leaf under the trusted generation root, not an arbitrary path.
    std::string name;
    std::span<const native::RecordSpec> specs;
    const native::SourceCatalog* sources=nullptr;
    uint64_t manifest_checksum=0;
    native::Limits limits;
};
class BoundRecord {
public:
    BoundRecord()=default;
    bool valid()const noexcept{return bool(generation_);}
private:
    friend struct detail::StoreImpl;
    friend class RankNativeStore;
    std::shared_ptr<const detail::Generation> generation_;
    uint32_t ordinal_=0;
};
class Lease {
public:
    Lease()=default;~Lease();
    Lease(Lease&&)noexcept;Lease& operator=(Lease&&)noexcept;
    Lease(const Lease&)=delete;Lease& operator=(const Lease&)=delete;
    std::span<const std::byte> bytes()const noexcept;
    native::Key key()const noexcept;
private:
    friend struct detail::StoreImpl;
    explicit Lease(std::shared_ptr<detail::Record>);
    std::shared_ptr<detail::Record> record_;
};
struct Batch {
    Batch()=default;~Batch();
    Batch(Batch&&)noexcept;Batch& operator=(Batch&&)noexcept;
    Batch(const Batch&)=delete;Batch& operator=(const Batch&)=delete;
    std::vector<Lease> leases;
private:
    friend struct detail::StoreImpl;
    // Allocated provider vector slots stay charged even after leases.clear(). Assign an
    // empty Batch to release retained vector storage and its credits, or destroy it.
    std::shared_ptr<detail::BatchCredit> storage_;
};
struct Usage {
    uint64_t hot_bytes=0,cold_bytes=0,peak_hot_bytes=0,peak_cold_bytes=0,metadata_reserved=0;
    size_t live_records=0,live_leases=0,peak_live_leases=0,lease_vector_slots=0;
};
class Accounting {
public:
    Usage usage()const noexcept;
private:
    friend class RankNativeStore;
    explicit Accounting(std::shared_ptr<detail::Ledger> p):ledger_(std::move(p)){}
    std::shared_ptr<detail::Ledger> ledger_;
};
struct Stats {
    uint64_t acquires=0,refused=0,hot_hits=0,cold_hits=0,misses=0,duplicates=0;
    uint64_t payload_opens=0,payload_reads=0,payload_read_bytes=0,fd_evictions=0;
    uint64_t cell_lookups=0,dedup_touches=0,metadata_opens=0,stat_probes=0,global_audits=0;
    uint64_t evictions=0,admission_bypasses=0,setup_verified_bytes=0,setup_manifest_bytes=0,
             setup_payload_io_bytes=0;
    // Actual-profile authority was verified BEFORE provider open. These are
    // retained planning reservations/verification facts, not provider I/O.
    uint64_t authority_verified_bytes=0,authority_metadata_reservation=0,
             facade_metadata_reservation=0;
    size_t catalog_records=0,segments=0,shards=0,cached_payload_fds=0;
    size_t startup_reserved_fd_peak=0,startup_actual_fd_peak=0;
    Usage usage;
};
class RankNativeStore {
public:
    static Error open(std::shared_ptr<ProcessBudget>,const rad::Program&,const rad::Plan&,
                      const std::string& generation_root,uint64_t epoch,
                      std::span<const ShardInput>,const Limits&,
                      std::unique_ptr<RankNativeStore>* out,std::string* why=nullptr)noexcept;
    // Separate complete-real-main admission; never routes actual manifests into
    // the closed synthetic Reader. Retains immutable authority and its credits
    // through every handle/lease, including after close. Full audits are setup
    // only; acquire audits the generation header and selected files.
    static Error open_verified_main(std::shared_ptr<ProcessBudget>,
                      std::shared_ptr<const glm53::verified_main::VerifiedRankMain>,
                      const Limits&,uint64_t facade_metadata_bytes,
                      std::unique_ptr<RankNativeStore>* out,std::string* why=nullptr)noexcept;
    ~RankNativeStore();
    RankNativeStore(const RankNativeStore&)=delete;
    // Key lookup is a setup API. Requests use opaque immutable handles, never mutable keys.
    Error bind(const native::Key&,BoundRecord* out)const noexcept;
    // Actual profile only. Exact canonical ordinal order, O(records), no key
    // searches. Entire span is preserved if validation fails.
    Error bind_main_handles(std::span<BoundRecord> out)const noexcept;
    Error acquire(std::span<const BoundRecord>,Batch* out,std::stop_token={})noexcept;
    Error audit_all()noexcept; // Global ordinary file-stamp audit; no full payload scrub.
    Error trim()noexcept;
    Error reload()noexcept{return Error::unsupported;} // Explicitly not implemented.
    Error close()noexcept;
    bool is_closing()const noexcept;
    Stats stats()const noexcept;
    Accounting accounting()const noexcept;
private:
    explicit RankNativeStore(std::unique_ptr<detail::StoreImpl>);
    std::unique_ptr<detail::StoreImpl> impl_;
};
} // namespace glm53::sharded
