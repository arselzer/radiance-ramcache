/* Bounded decoded Engram rows -> asynchronous device use. No HIP dependency.
 * Host memory is supplied by a caller-owned backend (pinned for a real async GPU
 * backend; ordinary CPU memory in tests). The shared staging budget is separate
 * from the packed SSD row-cache budget and device-allocation budget.
 *
 * Completion is deliberately conservative: quiescent means BOTH upload and the
 * last device reader are done. Error/cancellation never implies quiescence.
 * The tracked backend retains submitted records, host buffers, device owners and
 * their budget charges until that proof exists, including failed pool teardown.
 */
#pragma once
#include "engram_store.h"
#include <atomic>

namespace rad::deepseek41::engram {
class StagingBudget;
class StagingCharge {
public:
    StagingCharge()=default;~StagingCharge(){reset();}
    StagingCharge(const StagingCharge&)=delete;StagingCharge& operator=(const StagingCharge&)=delete;
    StagingCharge(StagingCharge&& other)noexcept{swap(other);}
    StagingCharge& operator=(StagingCharge&& other)noexcept{if(this!=&other){reset();swap(other);}return *this;}
    void reset()noexcept;
    uint64_t bytes()const{return bytes_;}
private:
    friend class StagingBudget;
    StagingCharge(std::shared_ptr<StagingBudget> owner,uint64_t bytes):owner_(std::move(owner)),bytes_(bytes){}
    void swap(StagingCharge& other)noexcept{owner_.swap(other.owner_);std::swap(bytes_,other.bytes_);}
    std::shared_ptr<StagingBudget> owner_;uint64_t bytes_=0;
};
class StagingBudget:public std::enable_shared_from_this<StagingBudget> {
public:
    struct Stats{uint64_t limit=0,used=0,peak=0;};
    explicit StagingBudget(uint64_t limit):limit_(limit){}
    int reserve(uint64_t bytes,StagingCharge* out){
        if(!out||!bytes)return RAD_E_INVAL;
        auto owner=shared_from_this();std::unique_lock lock(mutex_);
        if(bytes>limit_-used_)return RAD_E_FULL;
        used_+=bytes;peak_=std::max(peak_,used_);lock.unlock();
        *out=StagingCharge(std::move(owner),bytes);return RAD_OK;
    }
    Stats stats()const{std::lock_guard lock(mutex_);return {limit_,used_,peak_};}
private:
    friend class StagingCharge;
    void release(uint64_t bytes)noexcept{std::lock_guard lock(mutex_);assert(bytes<=used_);used_-=bytes;}
    const uint64_t limit_;mutable std::mutex mutex_;uint64_t used_=0,peak_=0;
};
inline void StagingCharge::reset()noexcept{if(owner_){owner_->release(bytes_);owner_.reset();bytes_=0;}}

// Backend allocation must be stable, suitably aligned and DMA-safe for that
// backend. accounted_bytes includes its host allocation/allocator control cost.
class HostRowAllocation {
public:
    virtual ~HostRowAllocation()=default;
    virtual std::span<std::byte> memory()noexcept=0;
    virtual uint64_t accounted_bytes()const noexcept=0;
    // Optional: a native allocation registry can retain this charge across a
    // failed physical free. Called immediately after allocate_host returns any
    // owner, before validation or another potentially allocating operation.
    virtual void adopt_budget_charge(StagingCharge&)noexcept{}
};
class StagingBuffer {
public:
    std::span<const std::byte> bytes()const{return {data_,size_};}
    uint64_t reserved_bytes()const{return reserved_;}
private:
    friend class EngramStagingPool;
    StagingBuffer(std::shared_ptr<HostRowAllocation> memory,StagingCharge charge,size_t usable,uint64_t reserved)
        :charge_(std::move(charge)),allocation_(std::move(memory)),data_(allocation_->memory().data()),size_(usable),reserved_(reserved){}
    std::span<float> writable(){return {reinterpret_cast<float*>(data_),size_/sizeof(float)};}
    // Charge must outlive physical deallocation, including an allocator's teardown.
    StagingCharge charge_;std::shared_ptr<HostRowAllocation> allocation_;std::byte* data_;size_t size_;uint64_t reserved_=0;
};
class DeviceRowBuffer {
public:
    virtual ~DeviceRowBuffer()=default;
    // Identity, address and extent must remain immutable while any operation owns
    // this object. A caller may not recycle the allocation behind that owner.
    virtual std::string_view model_identity()const noexcept=0;
    virtual uint64_t device_identity()const noexcept=0;
    virtual uintptr_t address()const noexcept=0;
    virtual size_t capacity_bytes()const noexcept=0;
};
struct DeviceRowTarget {
    std::shared_ptr<DeviceRowBuffer> owner;
    size_t offset=0;
    std::string source_sha256; // exact intended table asset, checked before enqueue
};
struct StagingSource {
private:
    friend class EngramStagingPool;StagingCharge charge;
public:
    std::string model_identity,sha256;int layer_id=-1;int64_t rows=0,width=0,data_offset=0,file_bytes=0;
};
struct RowTransferSubmission {
    std::shared_ptr<const StagingSource> source;
    std::shared_ptr<const StagingBuffer> host;
    DeviceRowTarget destination;
    size_t rows=0,width=0,bytes=0;
    std::stop_token request_stop,shutdown_stop;
    bool stop_requested()const{return request_stop.stop_requested()||shutdown_stop.stop_requested();}
};
struct RowTransferState {bool quiescent=false;int status=RAD_OK;};
class RowTransferCompletion {
public:
    virtual ~RowTransferCompletion()=default;
    // Nonblocking. quiescent certifies no remaining host DMA OR device reader.
    virtual RowTransferState poll()noexcept=0;
    virtual void request_cancel()noexcept=0; // advisory; must never fabricate quiescence
};
struct NativeRowEnqueue {
    int status=RAD_E_STATE;
    std::shared_ptr<RowTransferCompletion> completion;
    // Must be explicitly certified when native enqueue fails without a fence.
    // Default conservatively retains ownership until a later drain proves safe.
    bool no_work_queued=false;
};
class TrackedRowTransferBackend;
class RowTransferRecord {
public:
    const RowTransferSubmission& submission()const{return submission_;}
    RowTransferState state()noexcept{
        std::shared_ptr<RowTransferCompletion> completion;bool finished=false,forced=false,unsafe=false;int submitted=0,retired=0;
        {std::lock_guard lock(mutex_);completion=completion_;finished=finished_;forced=forced_safe_;unsafe=unsafe_;submitted=submitted_;retired=retired_;}
        if(forced)return {true,retired!=RAD_OK?retired:submitted};
        if(!finished)return {false,RAD_OK};
        if(!completion)return {!unsafe,submitted};
        if(submission_.stop_requested())completion->request_cancel();
        auto state=completion->poll();if(state.status==RAD_OK&&submitted!=RAD_OK)state.status=submitted;return state;
    }
    void request_cancel()noexcept{
        std::shared_ptr<RowTransferCompletion> completion;{std::lock_guard lock(mutex_);completion=completion_;}
        if(completion)completion->request_cancel();
    }
private:
    friend class EngramStagingPool;friend class TrackedRowTransferBackend;friend class RowTransferTicket;
    RowTransferRecord(RowTransferSubmission submission,std::shared_ptr<TrackedRowTransferBackend> backend)
        :submission_(std::move(submission)),backend_(std::move(backend)){}
    void finish(int status,std::shared_ptr<RowTransferCompletion> completion,bool unsafe=false)noexcept{
        std::lock_guard lock(mutex_);submitted_=status;completion_=std::move(completion);unsafe_=unsafe;finished_=true;
    }
    void safe(int status)noexcept{std::lock_guard lock(mutex_);forced_safe_=true;retired_=status;}
    bool finished()noexcept{std::lock_guard lock(mutex_);return finished_;}
    RowTransferSubmission submission_;std::shared_ptr<TrackedRowTransferBackend> backend_;
    std::mutex native_mutex_,mutex_;std::shared_ptr<RowTransferCompletion> completion_;bool finished_=false,forced_safe_=false,unsafe_=false;int submitted_=0,retired_=0;
};

// The common tracker installs ownership BEFORE invoking the native enqueue. Its
// pending registry retains itself through records until completion/drain is safe.
// The engine must keep polling/draining its backend; unresolved device failures
// intentionally retain memory AND budget rather than free submitted resources.
class TrackedRowTransferBackend:public std::enable_shared_from_this<TrackedRowTransferBackend> {
public:
    virtual ~TrackedRowTransferBackend()=default;
    virtual int allocation_charge(size_t usable_bytes,uint64_t* upper_bound)const noexcept=0;
    virtual int allocate_host(size_t usable_bytes,uint64_t maximum_charge,std::shared_ptr<HostRowAllocation>* out)noexcept=0;
    int enqueue(const std::shared_ptr<RowTransferRecord>& record)noexcept{
        auto self=shared_from_this();
        if(!record||record->backend_.get()!=this)return RAD_E_STATE;
        std::lock_guard native_lock(record->native_mutex_);
        if(record->finished())return RAD_E_STATE; // never replace a submitted fence
        if(record->submission().stop_requested()){record->finish(RAD_E_CANCELLED,{});return RAD_E_CANCELLED;}
        try {
            std::lock_guard lock(mutex_);
            for(const auto& pending:pending_)if(!pending->state().quiescent&&overlap(pending->submission(),record->submission())){
                record->finish(RAD_E_FULL,{});return RAD_E_FULL;
            }
            pending_.push_back(record); // can fail only before native work is queued
        }catch(...){record->finish(RAD_E_NOMEM,{});return RAD_E_NOMEM;}
        auto submitted=enqueue_native(record);
        int status=submitted.status;
        // Success without a fence is a backend protocol error. Keep ownership
        // until native drain explicitly proves safety; do not release optimistically.
        const bool unsafe=!submitted.completion&&(status==RAD_OK||!submitted.no_work_queued);
        if(status==RAD_OK&&!submitted.completion)status=RAD_E_ABI;
        record->finish(status,std::move(submitted.completion),unsafe);
        progress();return status;
    }
    size_t progress()noexcept{
        auto self=shared_from_this();std::lock_guard lock(mutex_);size_t kept=0;
        for(size_t i=0;i<pending_.size();++i)if(!pending_[i]->state().quiescent){
            if(kept!=i)pending_[kept]=std::move(pending_[i]);
            ++kept;
        }
        pending_.resize(kept);return kept;
    }
    size_t pending_count()const{std::lock_guard lock(mutex_);return pending_.size();}
    RowTransferState drain(const std::shared_ptr<RowTransferRecord>& record)noexcept{
        auto self=shared_from_this();
        if(!record||record->backend_.get()!=this)return {false,RAD_E_STATE};
        std::lock_guard native_lock(record->native_mutex_);
        // A pool can publish an active slot immediately before enqueue enters.
        // Never invoke native drain before that enqueue's no-work/fence result.
        if(!record->finished())return {false,RAD_E_CANCELLED};
        auto state=record->state();
        if(!state.quiescent){record->request_cancel();state=drain_native(record);if(state.quiescent){record->safe(state.status);state=record->state();}}
        progress();return state;
    }
    // Explicit backend/model teardown recovery, including records whose pool and
    // ticket were dropped after an unsafe drain. No request should call this for
    // unrelated work. Rotation makes this bounded without allocating a snapshot.
    RowTransferState drain_all(std::string_view model_identity={})noexcept{
        auto self=shared_from_this();const size_t attempts=pending_count();RowTransferState result{true,RAD_OK};
        for(size_t attempt=0;attempt<attempts;++attempt){
            std::shared_ptr<RowTransferRecord> record;
            {std::lock_guard lock(mutex_);for(size_t i=0;i<pending_.size();++i)
                if(model_identity.empty()||pending_[i]->submission().source->model_identity==model_identity){
                    record=pending_[i];std::rotate(pending_.begin()+i,pending_.begin()+i+1,pending_.end());break;
                }}
            if(!record)break;
            const auto state=drain(record);if(state.status!=RAD_OK)result.status=state.status;
        }
        progress();{std::lock_guard lock(mutex_);for(const auto& record:pending_)
            if(model_identity.empty()||record->submission().source->model_identity==model_identity)result.quiescent=false;}
        return result;
    }
protected:
    // Must not throw. Return a completion even on partial-submission error. A
    // missing completion remains unsafe unless no_work_queued is explicit.
    // Completion methods must not reenter this tracker's methods while polling.
    virtual NativeRowEnqueue enqueue_native(const std::shared_ptr<RowTransferRecord>& record)noexcept=0;
    // Error with quiescent=false keeps the record quarantined/charged. Do not
    // equate hipEventQuery/Sync failure with proof that DMA/readers stopped.
    virtual RowTransferState drain_native(const std::shared_ptr<RowTransferRecord>& record)noexcept=0;
private:
    static bool overlap(const RowTransferSubmission& a,const RowTransferSubmission& b)noexcept{
        if(a.destination.owner->device_identity()!=b.destination.owner->device_identity())return false;
        const uintptr_t first=a.destination.owner->address()+a.destination.offset;
        const uintptr_t second=b.destination.owner->address()+b.destination.offset;
        return first<second+b.bytes&&second<first+a.bytes;
    }
    mutable std::mutex mutex_;std::vector<std::shared_ptr<RowTransferRecord>> pending_;
};
class RowTransferTicket {
public:
    explicit operator bool()const{return bool(record_);}
    RowTransferState state()const noexcept{return record_?record_->state():RowTransferState{true,RAD_E_STATE};}
    void cancel()const noexcept{if(record_)record_->request_cancel();}
    const DeviceRowTarget* destination()const{return record_?&record_->submission().destination:nullptr;}
    std::shared_ptr<RowTransferRecord> backend_record(const TrackedRowTransferBackend& backend)const noexcept{
        return record_&&record_->backend_.get()==&backend?record_:std::shared_ptr<RowTransferRecord>{};
    }
    void reset(){record_.reset();}
private:
    friend class EngramStagingPool;std::shared_ptr<RowTransferRecord> record_;
};

struct EngramStagingOptions {size_t slots=0,max_rows=0,width=0;};
class EngramStagingPool:public std::enable_shared_from_this<EngramStagingPool> {
    enum class SlotState:uint8_t{free,filling,active};
    struct Slot{std::shared_ptr<StagingBuffer> buffer;std::shared_ptr<RowTransferRecord> operation;uint64_t generation=0;SlotState state=SlotState::free;};
public:
    struct Stats{size_t slots=0,free=0,filling=0,active=0;uint64_t submissions=0,reclaimed=0,failures=0;};
    static int create(std::string model_identity,std::shared_ptr<Iq4NlLayerStores> source,
                        std::shared_ptr<StagingBudget> budget,std::shared_ptr<TrackedRowTransferBackend> backend,
                        EngramStagingOptions options,std::shared_ptr<EngramStagingPool>* out,std::string* why=nullptr){
        size_t elements=0,bytes=0;
        if(!out||!source||!budget||!backend||model_identity.empty()||source->identity()!=model_identity||
           !options.slots||options.slots>64||!options.max_rows||!options.width||
           options.max_rows>SIZE_MAX/options.width||(elements=options.max_rows*options.width)>SIZE_MAX/sizeof(float))
            return refuse(why,RAD_E_SHAPE,"invalid Engram staging identity, geometry or owner");
        bytes=elements*sizeof(float);
        for(const auto& store:source->stores())if(uint64_t(store->asset().width)!=options.width)
            return refuse(why,RAD_E_SHAPE,"Engram staging width differs from its asset");
        try {
            const uint64_t controls=sizeof(EngramStagingPool)+4096+options.slots*2*sizeof(Slot)+source->stores().size()*32;
            StagingCharge control_charge;int status=budget->reserve(controls,&control_charge);if(status!=RAD_OK)return refuse(why,status,"Engram staging control budget exhausted");
            auto pool=std::shared_ptr<EngramStagingPool>(new EngramStagingPool);
            pool->source_=std::move(source);pool->budget_=std::move(budget);pool->backend_=std::move(backend);pool->options_=options;pool->controls_=std::move(control_charge);
            pool->slots_.resize(options.slots);
            for(const auto& store:pool->source_->stores()){
                const auto& asset=store->asset();StagingCharge charge;
                const uint64_t size=sizeof(StagingSource)+256+2*(model_identity.size()+asset.sha256.size());
                status=pool->budget_->reserve(size,&charge);if(status!=RAD_OK)return refuse(why,status,"Engram staging source budget exhausted");
                auto snapshot=std::make_shared<StagingSource>();snapshot->model_identity=model_identity;snapshot->sha256=asset.sha256;snapshot->layer_id=asset.layer_id;
                snapshot->rows=asset.rows;snapshot->width=asset.width;snapshot->data_offset=asset.data_offset;snapshot->file_bytes=asset.file_bytes;snapshot->charge=std::move(charge);
                pool->sources_.push_back(std::move(snapshot));
            }
            uint64_t allocation=0;status=pool->backend_->allocation_charge(bytes,&allocation);
            if(status!=RAD_OK||allocation<bytes||allocation>UINT64_MAX-sizeof(StagingBuffer)-128)
                return refuse(why,status!=RAD_OK?status:RAD_E_ABI,"invalid backend staging allocation charge");
            for(auto& slot:pool->slots_){
                StagingCharge charge;status=pool->budget_->reserve(allocation+sizeof(StagingBuffer)+128,&charge);
                if(status!=RAD_OK)return refuse(why,status,"Engram pinned staging budget exhausted");
                const uint64_t reserved=charge.bytes();
                std::shared_ptr<HostRowAllocation> memory;status=pool->backend_->allocate_host(bytes,allocation,&memory);
                if(memory)memory->adopt_budget_charge(charge);
                if(status!=RAD_OK)return refuse(why,status,"backend could not allocate pinned staging");
                if(!memory||memory->accounted_bytes()<bytes||memory->accounted_bytes()>allocation||memory->memory().size()<bytes||!memory->memory().data()||
                   reinterpret_cast<uintptr_t>(memory->memory().data())%alignof(float))
                    return refuse(why,RAD_E_ABI,"backend returned invalid/alignment-incompatible staging allocation");
                slot.buffer=std::shared_ptr<StagingBuffer>(new StagingBuffer(std::move(memory),std::move(charge),bytes,reserved));
            }
            *out=std::move(pool);if(why)why->clear();return RAD_OK;
        }catch(const std::bad_alloc&){return refuse(why,RAD_E_NOMEM,"cannot allocate Engram staging controls");}
    }
    ~EngramStagingPool(){shutdown();}
    Stats stats(){reap();std::lock_guard lock(mutex_);Stats stats=stats_;stats.slots=slots_.size();for(const auto& slot:slots_){if(slot.state==SlotState::free)++stats.free;else if(slot.state==SlotState::filling)++stats.filling;else ++stats.active;}return stats;}
    size_t reap()noexcept{
        if(!backend_)return 0;
        backend_->progress();std::lock_guard lock(mutex_);size_t active=0;
        for(auto& slot:slots_)if(slot.state==SlotState::active){if(slot.operation->state().quiescent){slot.operation.reset();slot.state=SlotState::free;++stats_.reclaimed;}else ++active;}
        return active;
    }
    int submit(int layer,std::span<const int64_t> rows,DeviceRowTarget destination,RowTransferTicket* out,
                 std::stop_token stop={},std::string* why=nullptr){
        if(out)out->reset();
        auto owner=shared_from_this();std::shared_ptr<const StagingSource> source;
        for(const auto& candidate:sources_)if(candidate->layer_id==layer)source=candidate;
        if(!out||!source||rows.empty()||rows.size()>options_.max_rows||!destination.owner||
           destination.owner->model_identity()!=source->model_identity||destination.source_sha256!=source->sha256||
           !destination.owner->address())return refuse(why,RAD_E_SHAPE,"invalid Engram staging layer, destination or asset identity");
        const size_t bytes=rows.size()*options_.width*sizeof(float),capacity=destination.owner->capacity_bytes();
        if(destination.offset>capacity||bytes>capacity-destination.offset||destination.offset>UINTPTR_MAX-destination.owner->address()||
           bytes>UINTPTR_MAX-(destination.owner->address()+destination.offset)||
           (destination.owner->address()+destination.offset)%alignof(float))
            return refuse(why,RAD_E_SHAPE,"Engram device destination range overflows");
        for(auto row:rows)if(row<0||row>=source->rows)return refuse(why,RAD_E_SHAPE,"Engram staging row is outside source asset");
        reap();size_t index=SIZE_MAX;uint64_t generation=0;
        {std::lock_guard lock(mutex_);if(closed_||stop.stop_requested())return refuse(why,RAD_E_CANCELLED,"Engram staging request or pool is cancelled");
            for(size_t i=0;i<slots_.size();++i)if(slots_[i].state==SlotState::free){index=i;slots_[i].state=SlotState::filling;generation=++slots_[i].generation;break;}}
        if(index==SIZE_MAX)return refuse(why,RAD_E_FULL,"all Engram staging slots are in use");
        struct FillGuard{EngramStagingPool* pool;size_t index;uint64_t generation;bool active=true;~FillGuard(){if(active){std::lock_guard lock(pool->mutex_);auto& slot=pool->slots_[index];assert(slot.generation==generation&&slot.state==SlotState::filling);slot.state=SlotState::free;++pool->stats_.failures;}}} guard{this,index,generation};
        try {
            std::stop_source combined;auto request_cancel=[&]{combined.request_stop();};
            std::stop_callback request_wake(stop,request_cancel),shutdown_wake(shutdown_stop_.get_token(),request_cancel);
            const auto memory=slots_[index].buffer;
            auto lookup=source_->row_lookup(combined.get_token());
            int status=lookup(layer,rows,int64_t(options_.width),memory->writable().first(bytes/sizeof(float)),why);
            if(status!=RAD_OK)return status;
            if(combined.stop_requested())return refuse(why,RAD_E_CANCELLED,"Engram staging was cancelled before enqueue");
            RowTransferSubmission submission{source,memory,std::move(destination),rows.size(),options_.width,bytes,stop,shutdown_stop_.get_token()};
            auto record=std::shared_ptr<RowTransferRecord>(new RowTransferRecord(std::move(submission),backend_));
            {std::lock_guard lock(mutex_);slots_[index].operation=record;slots_[index].state=SlotState::active;++stats_.submissions;guard.active=false;}
            status=backend_->enqueue(record);out->record_=std::move(record);
            if(status!=RAD_OK){std::lock_guard lock(mutex_);++stats_.failures;return refuse(why,status,"Engram transfer enqueue failed; ticket may still own in-flight work");}
            if(why)why->clear();
            return RAD_OK;
        }catch(const std::bad_alloc&){return refuse(why,RAD_E_NOMEM,"cannot allocate Engram request controls");}
    }
    // Model shutdown, never individual request cancellation. Failed/nonquiescent
    // drain leaves the backend registry holding all submitted ownership/charges.
    RowTransferState shutdown()noexcept{
        {std::lock_guard lock(mutex_);closed_=true;}
        shutdown_stop_.request_stop();std::array<std::shared_ptr<RowTransferRecord>,64> active;size_t count=0;bool filling=false;
        {std::lock_guard lock(mutex_);for(const auto& slot:slots_){if(slot.state==SlotState::active)active[count++]=slot.operation;else if(slot.state==SlotState::filling)filling=true;}}
        RowTransferState result{!filling,RAD_OK};
        if(backend_)for(size_t i=0;i<count;++i){const auto state=backend_->drain(active[i]);result.quiescent=result.quiescent&&state.quiescent;if(state.status!=RAD_OK)result.status=state.status;}
        reap();return result;
    }
private:
    EngramStagingPool()=default;
    std::shared_ptr<Iq4NlLayerStores> source_;std::shared_ptr<StagingBudget> budget_;std::shared_ptr<TrackedRowTransferBackend> backend_;
    EngramStagingOptions options_;StagingCharge controls_;std::vector<std::shared_ptr<const StagingSource>> sources_;std::vector<Slot> slots_;
    std::mutex mutex_;std::stop_source shutdown_stop_;bool closed_=false;Stats stats_;
};
} // namespace rad::deepseek41::engram
