#pragma once
// Closed synthetic HOST-only source factory. No Engine/architecture registration.
#include "build/startup.h"
#include "build/tiny_startup.h"
#include "place/record_owner.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace glm53::engine_source {
struct BindingReceipt {
    uint64_t source_fingerprint=0, file_bytes=0, payload_bytes=0;
    uint32_t physical_tensors=0, artifact_files=0;
};
struct PreparedReceipt {
    uint64_t physical_fingerprint=0, physical_source_bytes=0, native_output_bytes=0;
    uint32_t captured_records=0, owned_records=0, resident_records=0, shards=0;
};

// Budget must outlive this object AND every PreparedRecords/source/record lease
// derived from it. Its Source+Factory tickets precede opening a Checkpoint/cache.
class CheckpointBinding final : public std::enable_shared_from_this<CheckpointBinding> {
public:
    static int open(const std::string& path, rad::TinyBudget&,
                    std::shared_ptr<CheckpointBinding>* out, std::string* why=nullptr);
    ~CheckpointBinding();
    CheckpointBinding(const CheckpointBinding&)=delete;
    rad::Checkpoint& checkpoint() const noexcept;
    rad::WeightSourceFn source_fn();
    const rad::CkptWeight* find(const std::string&) const;
    std::string missing(const std::string&) const;
    std::vector<std::string> refused() const;
    int audit(std::string* why=nullptr) const;
    bool permits_zero_head(const RadModelMeta&,const RadBuildCtx&) const noexcept;
    const BindingReceipt& receipt() const noexcept;
private:
    struct Impl;
    explicit CheckpointBinding(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> impl_;
    friend class PreparedRecords;
};

// Program and Plan must remain immutable and alive while this object/source is
// open. Destruction closes future acquisition; already returned leases retain
// the same checkpoint, work generation, provider ownership and budget tickets.
class PreparedRecords final {
public:
    static int open(std::shared_ptr<CheckpointBinding>,const rad::Program&,
                    const rad::Plan& metadata_plan,const std::string& work_parent,
                    rad::TinyBudget&,std::unique_ptr<PreparedRecords>* out,
                    std::string* why=nullptr);
    ~PreparedRecords();
    PreparedRecords(const PreparedRecords&)=delete;
    std::shared_ptr<rad::NativeRecordSource> source() const noexcept;
    std::span<const rad::NativeRecordDescriptor> catalog() const noexcept;
    std::span<const rad::RecordOwnerBinding> owned() const noexcept;
    std::span<const int32_t> resident_units() const noexcept;
    // Copies a resident record into caller-owned HOST verification scratch.
    // It does NOT publish ordinary weights or prove Loader readiness.
    int bootstrap(int32_t unit,std::span<std::byte> target);
    // Engine must first compare all eight actual loaded ordinary four-member
    // spans against bootstrap copies after a successful real Loader drain.
    int finish_bootstrap();
    int audit(std::string* why=nullptr) const;
    int close();
    const PreparedReceipt& receipt() const noexcept;
private:
    struct Impl;
    explicit PreparedRecords(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> impl_;
};
} // namespace glm53::engine_source
