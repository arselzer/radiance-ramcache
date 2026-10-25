#include "ds41_engram_hip.h"
#include <limits>

namespace rad::deepseek41::engram {
namespace {
class DeviceGuard {
public:
    explicit DeviceGuard(int device)noexcept{
        if(hipGetDevice(&previous_)!=hipSuccess)return;
        if(previous_!=device){if(hipSetDevice(device)!=hipSuccess)return;changed_=true;}
        valid_=true;
    }
    ~DeviceGuard(){if(changed_)static_cast<void>(hipSetDevice(previous_));}
    bool valid()const noexcept{return valid_;}
    int finish(int status)noexcept{
        if(!valid_)return RAD_E_DEVICE;
        if(changed_){changed_=false;if(hipSetDevice(previous_)!=hipSuccess)return RAD_E_DEVICE;}
        return status;
    }
private:
    int previous_=0;bool valid_=false,changed_=false;
};
bool allocation_size(size_t bytes,uint64_t allowance,uint64_t* charge,size_t* physical)noexcept{
    constexpr uint64_t page=4096;
    if(!bytes||uint64_t(bytes)>UINT64_MAX-(page-1))return false;
    const uint64_t rounded=(uint64_t(bytes)+page-1)&~(page-1);
    if(rounded>SIZE_MAX||allowance>UINT64_MAX-rounded-512)return false;
    *physical=size_t(rounded);*charge=rounded+allowance+512;return true;
}
struct NativeCompletion;
struct NativeHostState {
    StagingCharge charge;void* pointer=nullptr;size_t usable=0,physical=0;uint64_t accounted=0;
    bool reserved=false,released=false;
};
}
struct HipEngramDeviceBuffer::State {
    StagingCharge charge;std::shared_ptr<void> external_owner;std::shared_ptr<HipEngramImpl> impl;
    void* pointer=nullptr;size_t bytes=0,physical=0;bool reserved=false,released=false,owned=false;
};
struct HipEngramImpl:std::enable_shared_from_this<HipEngramImpl> {
    using ReleasedOwners=std::array<std::shared_ptr<void>,256>;
    struct Operation {
        hipEvent_t upload=nullptr,reader=nullptr;uint64_t generation=0;
        bool busy=false,upload_recorded=false,reader_recorded=false,sealed=false,consumers=false,unknown_work=false;
        int error=RAD_OK;std::weak_ptr<RowTransferRecord> record;std::weak_ptr<NativeCompletion> completion;
    };
    std::string model;HipEngramOptions options;mutable std::mutex mutex;hipStream_t stream=nullptr;
    std::vector<Operation> operations;std::vector<std::shared_ptr<NativeHostState>> hosts;
    std::vector<std::shared_ptr<HipEngramDeviceBuffer::State>> devices;
    std::shared_ptr<HipEngramImpl> retained;HipEngramStats stats;
    int initialize()noexcept;
    void release_host(const std::shared_ptr<NativeHostState>&)noexcept;
    void release_device(const std::shared_ptr<HipEngramDeviceBuffer::State>&)noexcept;
    int retry_frees_locked(ReleasedOwners&,size_t*)noexcept;
    bool close_resources_locked(int* status)noexcept;
    Operation* find_locked(const std::shared_ptr<RowTransferRecord>& record)noexcept{
        for(auto& op:operations)if(op.busy&&op.record.lock()==record)return &op;
        return nullptr;
    }
    void latch_locked(Operation&,const std::shared_ptr<NativeCompletion>&,int)noexcept;
};
namespace {
struct NativeHost final:HostRowAllocation {
    std::shared_ptr<HipEngramImpl> impl;std::shared_ptr<NativeHostState> state;
    NativeHost(std::shared_ptr<HipEngramImpl> i,std::shared_ptr<NativeHostState> s):impl(std::move(i)),state(std::move(s)){}
    ~NativeHost()override{impl->release_host(state);}
    std::span<std::byte> memory()noexcept override{return {static_cast<std::byte*>(state->pointer),state->usable};}
    uint64_t accounted_bytes()const noexcept override{return state->accounted;}
    void adopt_budget_charge(StagingCharge& charge)noexcept override{
        std::lock_guard lock(impl->mutex);assert(!state->charge.bytes());state->charge=std::move(charge);
    }
};
struct NativeCompletion final:RowTransferCompletion, std::enable_shared_from_this<NativeCompletion> {
    std::shared_ptr<HipEngramImpl> impl;size_t slot=0;uint64_t generation=0;
    bool latched=false;int result=RAD_OK;std::atomic<bool> cancelled{false};
    NativeCompletion(std::shared_ptr<HipEngramImpl> i,size_t s,uint64_t g):impl(std::move(i)),slot(s),generation(g){}
    RowTransferState poll()noexcept override{
        std::lock_guard lock(impl->mutex);
        if(latched)return {true,result};
        auto& op=impl->operations[slot];
        if(!op.busy||op.generation!=generation)return {false,RAD_E_STATE};
        if(!op.sealed||op.unknown_work||!op.upload_recorded||!op.reader_recorded)return {false,op.error};
        DeviceGuard guard(impl->options.device);if(!guard.valid())return {false,RAD_E_DEVICE};
        const auto upload=hipEventQuery(op.upload),reader=hipEventQuery(op.reader);
        if((upload!=hipSuccess&&upload!=hipErrorNotReady)||(reader!=hipSuccess&&reader!=hipErrorNotReady)){
            op.unknown_work=true;op.error=RAD_E_DEVICE;++impl->stats.errors;return {false,guard.finish(RAD_E_DEVICE)};
        }
        const int restored=guard.finish(RAD_OK);
        if(restored!=RAD_OK){op.unknown_work=true;op.error=restored;return {false,restored};}
        if(upload!=hipSuccess||reader!=hipSuccess)return {false,op.error};
        impl->latch_locked(op,shared_from_this(),op.error);return {true,result};
    }
    void request_cancel()noexcept override{cancelled.store(true,std::memory_order_relaxed);}
};
}
void HipEngramImpl::latch_locked(Operation& op,const std::shared_ptr<NativeCompletion>& completion,int status)noexcept{
    completion->latched=true;completion->result=status;op.busy=false;op.record.reset();op.completion.reset();
    assert(stats.active_operations);--stats.active_operations;++stats.quiescent;
}
int HipEngramImpl::initialize()noexcept{
    DeviceGuard guard(options.device);if(!guard.valid())return RAD_E_DEVICE;
    if(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking)!=hipSuccess)return guard.finish(RAD_E_DEVICE);
    for(auto& op:operations){
        if(hipEventCreateWithFlags(&op.upload,hipEventDisableTiming)!=hipSuccess||
           hipEventCreateWithFlags(&op.reader,hipEventDisableTiming)!=hipSuccess)return guard.finish(RAD_E_DEVICE);
    }
    return guard.finish(RAD_OK);
}
int HipEngramImpl::retry_frees_locked(ReleasedOwners& owners,size_t* count)noexcept{
    DeviceGuard guard(options.device);if(!guard.valid())return RAD_E_DEVICE;
    int result=RAD_OK;
    for(auto& host:hosts)if(host->reserved&&host->released){
        if(host->pointer&&hipHostFree(host->pointer)!=hipSuccess){result=RAD_E_DEVICE;continue;}
        assert(stats.host_allocations);--stats.host_allocations;stats.pinned_bytes-=host->physical;
        host->pointer=nullptr;host->usable=host->physical=0;host->reserved=false;host->charge.reset();
    }
    for(auto& device:devices)if(device->reserved&&device->released){
        if(device->owned&&device->pointer&&hipFree(device->pointer)!=hipSuccess){result=RAD_E_DEVICE;continue;}
        assert(stats.device_allocations);--stats.device_allocations;
        if(device->owned)stats.owned_device_bytes-=device->physical;
        device->pointer=nullptr;device->bytes=device->physical=0;device->reserved=false;
        if(device->external_owner)owners[(*count)++]=std::move(device->external_owner);
        device->charge.reset();device->impl.reset();
    }
    return guard.finish(result);
}
bool HipEngramImpl::close_resources_locked(int* result)noexcept{
    if(!stats.closed||stats.active_operations||stats.host_allocations||stats.device_allocations)return false;
    DeviceGuard guard(options.device);if(!guard.valid()){*result=RAD_E_DEVICE;return false;}
    bool success=true;
    for(auto& op:operations){
        if(op.upload){if(hipEventDestroy(op.upload)==hipSuccess)op.upload=nullptr;else success=false;}
        if(op.reader){if(hipEventDestroy(op.reader)==hipSuccess)op.reader=nullptr;else success=false;}
    }
    if(success&&stream){if(hipStreamDestroy(stream)==hipSuccess)stream=nullptr;else success=false;}
    if(guard.finish(RAD_OK)!=RAD_OK)success=false;
    if(success)retained.reset();else *result=RAD_E_DEVICE;
    return success;
}
void HipEngramImpl::release_host(const std::shared_ptr<NativeHostState>& host)noexcept{
    ReleasedOwners owners;size_t count=0;
    {std::lock_guard lock(mutex);host->released=true;int result=retry_frees_locked(owners,&count);close_resources_locked(&result);}
}
void HipEngramImpl::release_device(const std::shared_ptr<HipEngramDeviceBuffer::State>& device)noexcept{
    ReleasedOwners owners;size_t count=0;
    {std::lock_guard lock(mutex);device->released=true;int result=retry_frees_locked(owners,&count);close_resources_locked(&result);}
}
HipEngramDeviceBuffer::HipEngramDeviceBuffer(std::shared_ptr<State> state):state_(std::move(state)){}
HipEngramDeviceBuffer::~HipEngramDeviceBuffer(){if(auto impl=state_->impl)impl->release_device(state_);}
std::string_view HipEngramDeviceBuffer::model_identity()const noexcept{return state_->impl->model;}
uint64_t HipEngramDeviceBuffer::device_identity()const noexcept{return uint64_t(state_->impl->options.device);}
uintptr_t HipEngramDeviceBuffer::address()const noexcept{return reinterpret_cast<uintptr_t>(state_->pointer);}
size_t HipEngramDeviceBuffer::capacity_bytes()const noexcept{return state_->bytes;}
HipEngramBackend::HipEngramBackend(std::shared_ptr<HipEngramImpl> impl):impl_(std::move(impl)){}
HipEngramBackend::~HipEngramBackend(){
    // Pending tracker records own the backend, so reaching the final-owner
    // destructor means no such record remains. Never call shared_from_this()
    // through close()/drain_all() here: the shared control block is expiring.
    HipEngramImpl::ReleasedOwners owners;size_t count=0;
    {std::lock_guard lock(impl_->mutex);impl_->stats.closed=true;
        int status=impl_->retry_frees_locked(owners,&count);impl_->close_resources_locked(&status);
    }
}
int HipEngramBackend::create(std::string model,HipEngramOptions options,std::shared_ptr<HipEngramBackend>* out,std::string* why){
    if(!out||model.empty()||options.device<0||!options.max_operations||options.max_operations>256||
       !options.max_host_allocations||options.max_host_allocations>256||!options.max_device_allocations||options.max_device_allocations>256)
        return refuse(why,RAD_E_SHAPE,"invalid native Engram model/device/bounds");
    try{
        auto impl=std::make_shared<HipEngramImpl>();impl->model=std::move(model);impl->options=options;
        impl->operations.resize(options.max_operations);
        for(size_t i=0;i<options.max_host_allocations;++i)impl->hosts.push_back(std::make_shared<NativeHostState>());
        for(size_t i=0;i<options.max_device_allocations;++i)impl->devices.push_back(std::make_shared<HipEngramDeviceBuffer::State>());
        impl->stats.native_fixed_allowance=sizeof(HipEngramImpl)+4096+options.max_operations*(sizeof(HipEngramImpl::Operation)+8192)+
            options.max_host_allocations*(sizeof(NativeHostState)+64)+options.max_device_allocations*(sizeof(HipEngramDeviceBuffer::State)+64);
        auto backend=std::shared_ptr<HipEngramBackend>(new HipEngramBackend(impl));impl->retained=impl;
        const int status=impl->initialize();
        if(status!=RAD_OK){
            impl->stats.closed=true;int cleanup=RAD_OK;const bool released=impl->close_resources_locked(&cleanup);
            if(!released)*out=std::move(backend);
            return refuse(why,status,released?"cannot initialize native Engram stream/events":"native initialization failed; retained backend must be closed");
        }
        *out=std::move(backend);
        if(why)why->clear();
        return RAD_OK;
    }catch(const std::bad_alloc&){return refuse(why,RAD_E_NOMEM,"cannot allocate bounded native Engram controls");}
}
int HipEngramBackend::allocation_charge(size_t bytes,uint64_t* charge)const noexcept{
    size_t physical=0;if(!charge||!allocation_size(bytes,impl_->options.host_driver_allowance,charge,&physical))return RAD_E_SHAPE;
    return RAD_OK;
}
int HipEngramBackend::allocate_host(size_t bytes,uint64_t maximum,std::shared_ptr<HostRowAllocation>* out)noexcept{
    uint64_t charge=0;size_t physical=0;
    if(!out||!allocation_size(bytes,impl_->options.host_driver_allowance,&charge,&physical)||charge>maximum)return RAD_E_SHAPE;
    out->reset();
    try{
        std::lock_guard lock(impl_->mutex);if(impl_->stats.closed)return RAD_E_CANCELLED;
        std::shared_ptr<NativeHostState> slot;for(const auto& candidate:impl_->hosts)if(!candidate->reserved){slot=candidate;break;}
        if(!slot)return RAD_E_FULL;
        // Allocate the wrapper BEFORE HIP can return any physical allocation.
        DeviceGuard guard(impl_->options.device);
        if(!guard.valid())return RAD_E_DEVICE;
        auto wrapper=std::make_shared<NativeHost>(impl_,slot);
        slot->reserved=true;slot->released=false;slot->usable=bytes;slot->physical=physical;slot->accounted=charge;
        ++impl_->stats.host_allocations;impl_->stats.pinned_bytes+=physical;
        const auto allocated=hipHostMalloc(&slot->pointer,physical,hipHostMallocPortable);
        const int status=guard.finish(allocated==hipSuccess&&slot->pointer?RAD_OK:RAD_E_DEVICE);
        // Even an error can retain a nonnull native pointer. Always return its
        // wrapper so the pool adopts the charge before any cleanup/free attempt.
        *out=std::move(wrapper);return status;
    }catch(const std::bad_alloc&){return RAD_E_NOMEM;}
}
int HipEngramBackend::allocate_device(size_t bytes,std::shared_ptr<StagingBudget> budget,
    std::shared_ptr<HipEngramDeviceBuffer>* out,std::string* why){
    uint64_t accounted=0;size_t physical=0;
    if(!out||!budget||!allocation_size(bytes,impl_->options.device_driver_allowance,&accounted,&physical))return refuse(why,RAD_E_SHAPE,"invalid native device allocation/budget");
    out->reset();
    StagingCharge charge;int status=budget->reserve(accounted,&charge);if(status!=RAD_OK)return refuse(why,status,"native Engram device budget exhausted");
    try{
        std::lock_guard lock(impl_->mutex);if(impl_->stats.closed)return refuse(why,RAD_E_CANCELLED,"native Engram backend is closed");
        std::shared_ptr<HipEngramDeviceBuffer::State> slot;for(const auto& candidate:impl_->devices)if(!candidate->reserved){slot=candidate;break;}
        if(!slot)return refuse(why,RAD_E_FULL,"native Engram device-owner slots exhausted");
        auto wrapper=std::shared_ptr<HipEngramDeviceBuffer>(new HipEngramDeviceBuffer(slot));DeviceGuard guard(impl_->options.device);
        if(!guard.valid())return refuse(why,RAD_E_DEVICE,"cannot select native Engram device");
        slot->reserved=true;slot->released=false;slot->owned=true;slot->impl=impl_;slot->bytes=bytes;slot->physical=physical;slot->charge=std::move(charge);
        ++impl_->stats.device_allocations;impl_->stats.owned_device_bytes+=physical;
        const auto allocated=hipMalloc(&slot->pointer,physical);status=guard.finish(allocated==hipSuccess&&slot->pointer?RAD_OK:RAD_E_DEVICE);
        *out=std::move(wrapper);if(status!=RAD_OK)return refuse(why,status,"native Engram device allocation failed; owner retains cleanup");
        if(why)why->clear();
        return RAD_OK;
    }catch(const std::bad_alloc&){return refuse(why,RAD_E_NOMEM,"cannot allocate native Engram device owner");}
}
int HipEngramBackend::wrap_device(void* pointer,size_t bytes,std::shared_ptr<void> owner,
    std::shared_ptr<HipEngramDeviceBuffer>* out,std::string* why){
    if(!out||!pointer||!bytes||!owner||reinterpret_cast<uintptr_t>(pointer)%alignof(float))return refuse(why,RAD_E_SHAPE,"invalid native Engram arena owner/range");
    out->reset();
    try{
        std::lock_guard lock(impl_->mutex);if(impl_->stats.closed)return refuse(why,RAD_E_CANCELLED,"native Engram backend is closed");
        DeviceGuard guard(impl_->options.device);if(!guard.valid())return refuse(why,RAD_E_DEVICE,"cannot select native Engram device");
        hipPointerAttribute_t attributes{};hipDeviceptr_t base=nullptr;size_t extent=0;
        if(hipPointerGetAttributes(&attributes,pointer)!=hipSuccess||attributes.type!=hipMemoryTypeDevice||attributes.isManaged||
           attributes.device!=impl_->options.device||hipMemGetAddressRange(&base,&extent,static_cast<hipDeviceptr_t>(pointer))!=hipSuccess){
            return refuse(why,guard.finish(RAD_E_SHAPE),"arena pointer is not a device-local HIP allocation");
        }
        const uintptr_t start=reinterpret_cast<uintptr_t>(pointer),first=reinterpret_cast<uintptr_t>(base);
        if(start<first||start-first>extent||bytes>extent-size_t(start-first))return refuse(why,guard.finish(RAD_E_SHAPE),"arena suballocation exceeds native allocation bounds");
        const int restored=guard.finish(RAD_OK);if(restored!=RAD_OK)return refuse(why,restored,"cannot restore caller device");
        std::shared_ptr<HipEngramDeviceBuffer::State> slot;for(const auto& candidate:impl_->devices)if(!candidate->reserved){slot=candidate;break;}
        if(!slot)return refuse(why,RAD_E_FULL,"native Engram device-owner slots exhausted");
        auto wrapper=std::shared_ptr<HipEngramDeviceBuffer>(new HipEngramDeviceBuffer(slot));
        slot->reserved=true;slot->released=false;slot->owned=false;slot->impl=impl_;slot->pointer=pointer;slot->bytes=bytes;slot->physical=0;
        slot->external_owner=std::move(owner);++impl_->stats.device_allocations;*out=std::move(wrapper);
        if(why)why->clear();
        return RAD_OK;
    }catch(const std::bad_alloc&){return refuse(why,RAD_E_NOMEM,"cannot allocate native Engram arena owner");}
}
NativeRowEnqueue HipEngramBackend::enqueue_native(const std::shared_ptr<RowTransferRecord>& record)noexcept{
    const auto& submission=record->submission();
    auto* destination=dynamic_cast<HipEngramDeviceBuffer*>(submission.destination.owner.get());
    if(!destination||destination->state_->impl!=impl_||submission.source->model_identity!=impl_->model)return {RAD_E_SHAPE,{},true};
    try{
        std::lock_guard lock(impl_->mutex);if(impl_->stats.closed)return {RAD_E_CANCELLED,{},true};
        size_t index=impl_->operations.size();for(size_t i=0;i<impl_->operations.size();++i)if(!impl_->operations[i].busy){index=i;break;}
        if(index==impl_->operations.size())return {RAD_E_FULL,{},true};
        auto& op=impl_->operations[index];const uint64_t generation=op.generation+1;
        auto completion=std::make_shared<NativeCompletion>(impl_,index,generation);DeviceGuard guard(impl_->options.device);
        if(!guard.valid())return {RAD_E_DEVICE,{},true};
        op.generation=generation;op.busy=true;op.upload_recorded=op.reader_recorded=op.sealed=op.consumers=op.unknown_work=false;
        op.error=RAD_OK;op.record=record;op.completion=completion;++impl_->stats.active_operations;++impl_->stats.queued;
        const auto copied=hipMemcpyAsync(reinterpret_cast<void*>(destination->address()+submission.destination.offset),
            submission.host->bytes().data(),submission.bytes,hipMemcpyHostToDevice,impl_->stream);
        if(copied==hipSuccess&&hipEventRecord(op.upload,impl_->stream)==hipSuccess){op.upload_recorded=true;++impl_->stats.upload_fences;}
        else {op.error=RAD_E_DEVICE;op.unknown_work=true;++impl_->stats.errors;}
        const int status=guard.finish(op.error);if(status!=RAD_OK){op.error=status;op.unknown_work=true;}
        return {status,completion,false};
    }catch(const std::bad_alloc&){return {RAD_E_NOMEM,{},true};}
}
RowTransferState HipEngramBackend::drain_native(const std::shared_ptr<RowTransferRecord>& record)noexcept{
    std::lock_guard lock(impl_->mutex);auto* op=impl_->find_locked(record);if(!op)return {true,RAD_OK};
    if(!op->sealed)return {false,op->error!=RAD_OK?op->error:RAD_E_STATE};
    auto completion=op->completion.lock();if(!completion)return {false,RAD_E_STATE};
    DeviceGuard guard(impl_->options.device);if(!guard.valid())return {false,RAD_E_DEVICE};
    if(hipStreamSynchronize(impl_->stream)!=hipSuccess){op->unknown_work=true;op->error=RAD_E_DEVICE;++impl_->stats.errors;return {false,guard.finish(RAD_E_DEVICE)};}
    const int restored=guard.finish(RAD_OK);if(restored!=RAD_OK){op->unknown_work=true;return {false,restored};}
    const int result=op->error;impl_->latch_locked(*op,completion,result);return {true,result};
}
int HipEngramBackend::order_upload(const RowTransferTicket& ticket)noexcept{
    auto record=ticket.backend_record(*this);if(!record)return RAD_E_STATE;
    std::lock_guard lock(impl_->mutex);auto* op=impl_->find_locked(record);if(!op||op->sealed||!op->upload_recorded)return RAD_E_STATE;
    DeviceGuard guard(impl_->options.device);if(!guard.valid())return RAD_E_DEVICE;
    const auto waited=hipStreamWaitEvent(impl_->stream,op->upload,0);const int status=guard.finish(waited==hipSuccess?RAD_OK:RAD_E_DEVICE);
    if(status!=RAD_OK){op->unknown_work=true;op->error=status;}return status;
}
int HipEngramBackend::wait_upload(const RowTransferTicket& ticket)noexcept{
    auto record=ticket.backend_record(*this);if(!record)return RAD_E_STATE;
    std::lock_guard lock(impl_->mutex);auto* op=impl_->find_locked(record);if(!op||!op->upload_recorded)return RAD_E_STATE;
    DeviceGuard guard(impl_->options.device);if(!guard.valid())return RAD_E_DEVICE;
    const auto waited=hipEventSynchronize(op->upload);const int status=guard.finish(waited==hipSuccess?RAD_OK:RAD_E_DEVICE);
    if(status!=RAD_OK){op->unknown_work=true;op->error=status;}return status;
}
int HipEngramBackend::enqueue_consumer(const RowTransferTicket& ticket,Consumer consumer,void* user)noexcept{
    auto record=ticket.backend_record(*this);if(!record||!consumer)return RAD_E_STATE;
    std::lock_guard lock(impl_->mutex);auto* op=impl_->find_locked(record);
    if(!op||op->sealed||!op->upload_recorded||op->error!=RAD_OK)return RAD_E_STATE;
    DeviceGuard guard(impl_->options.device);if(!guard.valid())return RAD_E_DEVICE;
    op->consumers=true;int status=RAD_OK;
    if(hipStreamWaitEvent(impl_->stream,op->upload,0)!=hipSuccess)status=RAD_E_DEVICE;
    else status=consumer(impl_->stream,user);
    status=guard.finish(status);if(status!=RAD_OK){op->unknown_work=true;op->error=status;++impl_->stats.errors;}return status;
}
int HipEngramBackend::consume(const RowTransferTicket& ticket,SealedConsumer consumer,void* user)noexcept{
    auto record=ticket.backend_record(*this);if(!record||!consumer)return RAD_E_STATE;
    std::lock_guard lock(impl_->mutex);auto* op=impl_->find_locked(record);
    if(!op||op->sealed)return RAD_E_STATE;
    DeviceGuard guard(impl_->options.device);
    // This helper is an explicit no-future-reader promise even if device
    // selection, upload ordering, callback or cancellation fails.
    op->sealed=true;
    if(!guard.valid()){op->unknown_work=true;op->error=RAD_E_DEVICE;return RAD_E_DEVICE;}
    int status=op->error;
    if(status==RAD_OK){
        const auto completion=op->completion.lock();
        if(record->submission().stop_requested()||(completion&&completion->cancelled.load(std::memory_order_relaxed)))status=RAD_E_CANCELLED;
        else if(!op->upload_recorded||hipStreamWaitEvent(impl_->stream,op->upload,0)!=hipSuccess)status=RAD_E_DEVICE;
        else {
            op->consumers=true;
            try{status=consumer(impl_->stream,record->submission().destination,user);}
            catch(const std::bad_alloc&){status=RAD_E_NOMEM;}
            catch(...){status=RAD_E_STATE;}
        }
    }
    // Even a throwing callback may have queued valid readers. Always attempt a
    // final stream fence; a failure remains quarantined until safe stream drain.
    if(hipEventRecord(op->reader,impl_->stream)==hipSuccess){op->reader_recorded=true;++impl_->stats.reader_fences;}
    else {op->unknown_work=true;status=RAD_E_DEVICE;}
    status=guard.finish(status);
    if(status!=RAD_OK){op->error=status;++impl_->stats.errors;}
    return status;
}
int HipEngramBackend::seal(const RowTransferTicket& ticket,bool no_readers)noexcept{
    auto record=ticket.backend_record(*this);if(!record)return RAD_E_STATE;
    std::lock_guard lock(impl_->mutex);auto* op=impl_->find_locked(record);if(!op||op->sealed||(no_readers&&op->consumers))return RAD_E_STATE;
    op->sealed=true;DeviceGuard guard(impl_->options.device);if(!guard.valid()){op->unknown_work=true;op->error=RAD_E_DEVICE;return RAD_E_DEVICE;}
    const auto recorded=hipEventRecord(op->reader,impl_->stream);if(recorded==hipSuccess){op->reader_recorded=true;++impl_->stats.reader_fences;}
    else {op->unknown_work=true;op->error=RAD_E_DEVICE;++impl_->stats.errors;}
    const int status=guard.finish(op->error);if(status!=RAD_OK){op->unknown_work=true;op->error=status;}return status;
}
int HipEngramBackend::finish_readers(const RowTransferTicket& ticket)noexcept{return seal(ticket,false);}
int HipEngramBackend::finish_no_readers(const RowTransferTicket& ticket)noexcept{return seal(ticket,true);}
HipEngramStats HipEngramBackend::native_stats()const noexcept{
    std::lock_guard lock(impl_->mutex);auto stats=impl_->stats;
    for(const auto& host:impl_->hosts)if(host->reserved&&host->released)++stats.released_host_pending_free;
    for(const auto& device:impl_->devices)if(device->reserved&&device->released)++stats.released_device_pending_free;
    return stats;
}
RowTransferState HipEngramBackend::close()noexcept{
    {std::lock_guard lock(impl_->mutex);impl_->stats.closed=true;}
    auto state=drain_all(impl_->model);
    HipEngramImpl::ReleasedOwners owners;size_t count=0;
    {std::lock_guard lock(impl_->mutex);const int freed=impl_->retry_frees_locked(owners,&count);if(freed!=RAD_OK)state.status=freed;
        if(impl_->stats.active_operations||impl_->stats.host_allocations||impl_->stats.device_allocations)state.quiescent=false;
        if(state.quiescent&&!impl_->close_resources_locked(&state.status))state.quiescent=false;
    }
    return state;
}
} // namespace rad::deepseek41::engram
