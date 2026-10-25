#pragma once
// Isolated synthetic CPU bridge. No Engine/GPU/checkpoint/provider production API.
#include "native_record_store.h"
#include "expert_cache.h"
#include <memory>
#include <span>
#include <stop_token>
namespace glm53::record_bridge {
namespace store=rad::expert_store;
namespace cache=glm53::cache;
struct Fresh {
    std::span<const store::RecordSpec> specs;
    const store::SourceCatalog& catalog;
    uint64_t manifest_checksum=0;
};
struct Limits {
    uint32_t world=4,rank=0;
    store::Limits store;
    cache::Limits cache{90112,229376,1u<<20,128,128,256,256};
    uint64_t staging_bytes=65536,max_payload_bytes=16u<<20;
    size_t max_snapshot_bytes=1u<<20,max_watch_files=129,max_metadata_bytes=8u<<20;
};
struct Stats {
    uint64_t setup_reads=0,setup_success_bytes=0,provider_reads=0,provider_failures=0,
             provider_success_bytes=0,copy_bytes=0,staging_bytes=0,peak_staging_bytes=0,
             acquire_calls=0,refused_calls=0,stat_checks=0;
    size_t snapshot_bytes=0,watched_files=0,metadata_reservation=0;
    cache::Stats cache;
};
class RecordCache {
public:
    ~RecordCache();RecordCache(const RecordCache&)=delete;RecordCache& operator=(const RecordCache&)=delete;
    // Full S1 validation, then one sequential startup checksum read per store record.
    // On failure *out is unchanged. Only fresh canonical source/layout metadata is accepted.
    static cache::Error open(const std::string& directory,const Fresh&,const Limits&,
                             std::unique_ptr<RecordCache>* out,std::string* why=nullptr) noexcept;
    std::span<const cache::RecordRef> records() const noexcept;
    // Fresh snapshot + ordinary file stat checks run even on HOT hits. Acquires are serialized.
    // Result publication is transactional; internal verified cache fills need not roll back.
    cache::Error acquire(std::span<const cache::RecordRef>,const Fresh&,cache::Batch* out,
                         std::stop_token={}) noexcept;
    cache::Error trim() noexcept;
    Stats stats() const noexcept;
    cache::Accounting accounting() const noexcept;
private:
    struct Impl;explicit RecordCache(std::unique_ptr<Impl>);std::unique_ptr<Impl> impl_;
};
} // namespace glm53::record_bridge
