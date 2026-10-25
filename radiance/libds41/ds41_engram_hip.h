/* Opt-in native HIP bridge for the bounded Engram staging adapter.
 * Not registered in the serving graph. Creation allocates a stream/events and
 * pinned/device allocations are explicit; importing this header does no work. */
#pragma once
#include "../arch/deepseek41/engram_staging.h"
#include <hip/hip_runtime_api.h>

namespace rad::deepseek41::engram {
struct HipEngramOptions {
    int device=0;
    size_t max_operations=16,max_host_allocations=64,max_device_allocations=64;
    // Explicit conservative per-allocation allowances. Not a driver-RSS claim.
    uint64_t host_driver_allowance=65536,device_driver_allowance=65536;
};
struct HipEngramStats {
    size_t active_operations=0,host_allocations=0,device_allocations=0;
    size_t released_host_pending_free=0,released_device_pending_free=0;
    uint64_t pinned_bytes=0,owned_device_bytes=0,native_fixed_allowance=0;
    uint64_t queued=0,upload_fences=0,reader_fences=0,quiescent=0,errors=0;
    bool closed=false;
};
class HipEngramBackend;
struct HipEngramImpl;
class HipEngramDeviceBuffer final:public DeviceRowBuffer {
public:
    ~HipEngramDeviceBuffer()override;
    std::string_view model_identity()const noexcept override;
    uint64_t device_identity()const noexcept override;
    uintptr_t address()const noexcept override;
    size_t capacity_bytes()const noexcept override;
private:
    friend class HipEngramBackend;friend struct HipEngramImpl;
    struct State;
    explicit HipEngramDeviceBuffer(std::shared_ptr<State>);
    std::shared_ptr<State> state_;
};
class HipEngramBackend final:public TrackedRowTransferBackend {
public:
    using Consumer=int(*)(hipStream_t,void*)noexcept;
    using SealedConsumer=int(*)(hipStream_t,const DeviceRowTarget&,void*);
    static int create(std::string model_identity,HipEngramOptions,
                      std::shared_ptr<HipEngramBackend>* out,std::string* why=nullptr);
    ~HipEngramBackend()override;
    int allocation_charge(size_t,uint64_t*)const noexcept override;
    int allocate_host(size_t,uint64_t,std::shared_ptr<HostRowAllocation>*)noexcept override;
    // Separately charged device allocator. An allocation keeps its charge until
    // hipFree succeeds, even when an earlier teardown attempt failed.
    int allocate_device(size_t bytes,std::shared_ptr<StagingBudget> device_budget,
                        std::shared_ptr<HipEngramDeviceBuffer>* out,std::string* why=nullptr);
    // For externally budgeted arena suballocations. HIP validates type/device and
    // the containing allocation bounds; owner must forbid recycle until released.
    int wrap_device(void* address,size_t bytes,std::shared_ptr<void> owner,
                    std::shared_ptr<HipEngramDeviceBuffer>* out,std::string* why=nullptr);
    // All consumer work is queued through this fixed-stream callback, under a
    // device TLS guard. The callback must not retain stream or reenter backend.
    // It must not throw; failure can still mean partial work and requires sealing.
    int enqueue_consumer(const RowTransferTicket&,Consumer,void* user)noexcept;
    // Preferred single-callback scope: always seals after return/exception,
    // fencing partial work. Callback may queue only on this stream and may not
    // retain it or queue future readers after returning. Other referenced device
    // allocations must remain caller-owned until ticket quiescence.
    int consume(const RowTransferTicket&,SealedConsumer,void* user)noexcept;
    int order_upload(const RowTransferTicket&)noexcept;
    int wait_upload(const RowTransferTicket&)noexcept;
    // Explicit caller proof: after this call no future device reader may be queued
    // for this ticket. finish_no_readers additionally rejects prior consumers.
    int finish_readers(const RowTransferTicket&)noexcept;
    int finish_no_readers(const RowTransferTicket&)noexcept;
    HipEngramStats native_stats()const noexcept;
    // Call after pools/tickets/owned buffers are dropped and consumers are sealed.
    // A failed free/drain retains resources and charge; retry is allowed. Does not
    // treat stream/event errors as quiescence. Never resets the device.
    RowTransferState close()noexcept;
private:
    explicit HipEngramBackend(std::shared_ptr<HipEngramImpl>);
    NativeRowEnqueue enqueue_native(const std::shared_ptr<RowTransferRecord>&)noexcept override;
    RowTransferState drain_native(const std::shared_ptr<RowTransferRecord>&)noexcept override;
    int seal(const RowTransferTicket&,bool no_readers)noexcept;
    std::shared_ptr<HipEngramImpl> impl_;
};
} // namespace rad::deepseek41::engram
