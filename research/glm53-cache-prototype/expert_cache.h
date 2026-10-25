#pragma once
// Standalone CPU research prototype. No Radiance ABI, GPU pinning or serving integration.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stop_token>
#include <string>
#include <vector>

namespace glm53::cache {
enum class Error {
    ok, invalid, too_large, budget, metadata_limit, identity_mismatch,
    io, checksum, cancelled, reentrant, no_memory, closed
};
const char* to_string(Error) noexcept;
struct Key {
    std::string source_id, model_id, layout_id, record_class;
    uint32_t tp=4,rank=0;
    int32_t layer=0,expert=0;
    bool operator==(const Key&) const = default;
};
struct RecordRef {
    Key key;
    uint64_t offset=0,length=0,checksum=0;
    bool operator==(const RecordRef&) const = default;
};
uint64_t fnv1a64(std::span<const std::byte>) noexcept; // noncryptographic integrity only
struct ReadResult {Error error=Error::ok;size_t bytes_read=0;};
class Provider {
public:
    virtual ~Provider()=default;
    // Fill exactly dst.size()==ref.length immutable bytes. Called outside the metadata mutex,
    // but inside one serialized fill transaction. Must not initiate/wait for cache mutations
    // from this callback (including via another thread). Same-thread reentry is refused.
    // Return exact bytes_read, even on short EOF. May throw: Cache maps exceptions to errors.
    virtual ReadResult read(const RecordRef&,std::span<std::byte> dst,std::stop_token)=0;
};
struct Limits {
    uint64_t hot_bytes=0,cold_bytes=0,max_record_bytes=16u*1024u*1024u;
    size_t max_records=1024,max_batch=288,max_identity_bytes=256;
    size_t max_catalog_records=16384; // immutable ref history; includes evicted records
};
struct Stats {
    uint64_t hot_bytes=0,cold_bytes=0,peak_hot_bytes=0,peak_cold_bytes=0;
    size_t live_records=0,index_entries=0,hot_records=0,inflight_records=0,catalog_entries=0;
    uint64_t requests=0,unique_demands=0,duplicate_demands=0,hot_hits=0,cold_hits=0,
             misses=0,provider_reads=0,bytes_read=0,evictions=0,admissions=0,
             admission_bypasses=0,failed_batches=0,budget_refusals=0;
};
namespace detail {struct Record;struct State;struct Ledger;}
struct Usage {uint64_t hot_bytes=0,cold_bytes=0,peak_hot_bytes=0,peak_cold_bytes=0;size_t live_records=0;};
class Accounting {
public:
    Accounting()=default;
    Usage usage() const noexcept;
private:
    explicit Accounting(std::shared_ptr<detail::Ledger> ledger):ledger_(std::move(ledger)){}
    std::shared_ptr<detail::Ledger> ledger_;
    friend class Cache;
};
class Lease {
public:
    Lease()=default;
    explicit operator bool() const noexcept {return bool(record_);}
    const RecordRef& ref() const; // throws std::logic_error only for an empty lease
    std::span<const std::byte> bytes() const noexcept;
private:
    explicit Lease(std::shared_ptr<detail::Record> record):record_(std::move(record)){}
    std::shared_ptr<detail::Record> record_;
    friend class Cache;
};
struct Batch {
    // Unique keys in first-occurrence request order; duplicates consume one lease/demand.
    std::vector<Lease> leases;
};
class Cache {
public:
    // Invalid limits throw std::invalid_argument; allocation failure may throw here.
    // Provider remains caller-owned and must outlive this Cache and active acquire calls.
    Cache(Limits,Provider&);
    ~Cache();
    Cache(const Cache&)=delete;Cache& operator=(const Cache&)=delete;
    Cache(Cache&&)=delete;Cache& operator=(Cache&&)=delete;
    // Entire miss union is reserved before first read. All bytes/checksums must validate
    // before any new ready batch is returned. On every refusal *out is unchanged.
    // Acquires are serialized (including I/O): overlapping keys reuse a ready live record.
    // Cancellation is checked before/after waiting and each provider call; it does not
    // forcibly interrupt provider I/O. No partial batch or expert dropping is supported.
    Error acquire(std::span<const RecordRef>,Batch* out,std::stop_token={}) noexcept;
    // Evicts every currently unleased hot entry. Leased records remain hot and charged.
    Error trim() noexcept;
    Stats stats() const noexcept;
    Accounting accounting() const noexcept;
    // Immutable leases may outlive Cache. Their own shared ledger keeps bytes charged until
    // the payload is actually freed. Cache destruction must not race calls on the object.
private:
    std::unique_ptr<detail::State> state_;
};
} // namespace glm53::cache
