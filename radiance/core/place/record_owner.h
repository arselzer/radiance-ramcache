#pragma once
#include "mover.h"
#include "runtime/expert_prepare.h"
#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <vector>
namespace rad {
struct NativeRecordDescriptor {
    uint64_t source_generation=0,checksum=0;
    uint32_t world=0,rank=0;
    int32_t layer=-1,expert=-1;
    int64_t stride=0;
    std::array<int64_t,4> offsets{},bytes{};
    std::array<uint32_t,4> dtype{};
    std::array<std::array<int64_t,2>,4> shape{};
    bool operator==(const NativeRecordDescriptor&)const=default;
};
struct NativeRecordLease {
    uint32_t record=0;
    std::span<const std::byte> bytes;
    std::shared_ptr<const void> lifetime; // immutable cache lease owner, not a raw borrowed buffer
};
struct NativeRecordBatch {std::vector<NativeRecordLease> leases;};
class NativeRecordSource {
public:
    virtual ~NativeRecordSource()=default;
    virtual std::span<const NativeRecordDescriptor> catalog()const noexcept=0;
    // Exact requested immutable union, first occurrence order; *out unchanged on refusal.
    // Source owns cache/Reader memory accounting. Cooperative synchronous initial profile.
    virtual int acquire(std::span<const uint32_t> records,NativeRecordBatch* out)=0;
};
struct RecordOwnerBinding {int32_t unit=-1;uint32_t record=0;};
struct RecordOwnerLimits {
    size_t max_units=512,max_regions=1024;
    uint64_t max_landing_bytes=8u<<20,max_device_bytes=8u<<20,max_metadata_bytes=4u<<20;
};
struct RecordOwnerStats {
    uint64_t landing_bytes=0,device_bytes=0,metadata_bytes=0,original_mover_base_bytes=0,retained_cache_unit_bytes=0;
    uint64_t prepares=0,source_batches=0,source_bytes=0,copy_bytes=0,published_units=0,retirements=0;
    size_t held_leases=0;
};
class ExpertRecordOwner final:public ExpertPrepareOwner {
public:
    // Does not create a loader/planner capability. Original Mover bases remain allocated and
    // charged; this is only a bounded component proof until DISTINCT native backing is added.
    static int open(Program&,const Plan&,Mover&,std::shared_ptr<NativeRecordSource>,
                    std::span<const RecordOwnerBinding>,const RecordOwnerLimits&,
                    std::unique_ptr<ExpertRecordOwner>* out);
    ~ExpertRecordOwner();
    int prepare(const ExpertPrepareView&,bool*)override;
    int check_program(const Program&) const override;
    int end_pass(RadStream,int status)override;
    int quiesce(); // MUST succeed while Ctx/compute stream still lives; backing kept through Ctx shutdown
    RecordOwnerStats stats()const noexcept;
    int check_guards_host()const; // test-only HOST inspection, refuses GPU backend
    ExpertRecordOwner(const ExpertRecordOwner&)=delete;
private:
    struct Impl;explicit ExpertRecordOwner(std::unique_ptr<Impl>);std::unique_ptr<Impl> impl_;
};
} // namespace rad
