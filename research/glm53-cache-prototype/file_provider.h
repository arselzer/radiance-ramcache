#pragma once
// Standalone POSIX CPU fixture provider. No Radiance/GPU/RDMA integration.
#include "expert_cache.h"
#include <map>
#include <string>
#include <vector>
namespace glm53::cache {
inline constexpr const char* file_schema="glm53-opaque-prepacked-file-v1";
struct FileIdentity {
    std::string schema=file_schema,source_id,model_id,layout_id;
    uint32_t tp=4,rank=0;
    std::map<std::string,uint64_t> class_lengths;
    uint64_t max_file_bytes=4u*1024u*1024u,max_record_bytes=16u*1024u;
    size_t max_records=16384,max_identity_bytes=256,max_total_identity_bytes=4u*1024u*1024u;
};
struct FileStats {uint64_t attempts=0,completed=0,failed=0,pread_calls=0,bytes_read=0;};
class FileProvider final:public Provider {
public:
    ~FileProvider() override;
    FileProvider(const FileProvider&)=delete;FileProvider& operator=(const FileProvider&)=delete;
    // Exact immutable manifest, identity/classes, regular-file extent and offsets are checked
    // before publication. On failure *out remains unchanged. FNV1a64 is not authentication.
    static Error open(const std::string& path,const FileIdentity&,std::span<const RecordRef>,
                      std::unique_ptr<FileProvider>* out,std::string* why=nullptr) noexcept;
    ReadResult read(const RecordRef&,std::span<std::byte>,std::stop_token={}) override;
    FileStats stats() const noexcept;
private:
    struct Impl;explicit FileProvider(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> impl_;
};
struct CatalogLayer {int32_t layer=0,experts=0;uint64_t record_bytes=0;std::string record_class;};
struct Rank0Catalog {
    std::string source_revision,source;
    std::vector<CatalogLayer> layers;
    const CatalogLayer* find(int32_t layer) const noexcept;
};
// Deliberately pinned executed rank0 metadata, not a payload/checkpoint verifier.
Error load_rank0_catalog(const std::string&,Rank0Catalog*,std::string* why=nullptr) noexcept;
struct RouteEvent {uint64_t step=0;std::string phase;int32_t layer=0;std::vector<int32_t> experts;};
struct RouteTrace {bool synthetic=false;std::string provenance;std::vector<RouteEvent> events;};
struct TraceLimits {size_t max_bytes=1024u*1024u,max_events=10000,max_routes_per_event=4096,max_line_bytes=65536;};
// JSON object {schema:"glm53-route-unions-v1",synthetic:bool,events:[...]}, JSON array,
// JSONL objects, or strict CSV step,phase,layer,experts with semicolon-separated IDs.
// Validate the entire file, catalog keys and order before replacing *out. No production reads.
Error load_route_trace(const std::string&,const Rank0Catalog&,const TraceLimits&,RouteTrace*,std::string* why=nullptr) noexcept;
} // namespace glm53::cache
