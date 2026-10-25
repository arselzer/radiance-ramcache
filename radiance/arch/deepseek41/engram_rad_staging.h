/* Core-compatible Engram staging. Uses actual RadStream/RadEvent, never HIP casts.
 * Caller owns stream/arena lifetimes and binds the rank device before creation.
 * This is a reusable adapter, not OpStager/graph/serving integration. */
#pragma once
#include "engram_staging.h"
#include "rad_device.h"
#include <thread>

namespace rad::deepseek41::engram {
class RadStagingApi {
public:
    virtual ~RadStagingApi()=default;
    virtual int create_event(RadEvent* out)noexcept{return rad_event_create(out);}
    virtual void destroy_event(RadEvent event)noexcept{rad_event_destroy(event);}
    virtual int copy(void* out,const void* in,int64_t bytes,RadStream stream)noexcept{return rad_memcpy_async(out,in,bytes,stream);}
    virtual int record(RadEvent event,RadStream stream)noexcept{return rad_event_record(event,stream);}
    virtual int wait(RadStream stream,RadEvent event)noexcept{return rad_event_wait(stream,event);}
    virtual int query(RadEvent event)noexcept{return rad_event_query(event);}
    virtual int synchronize(RadStream stream)noexcept{return rad_stream_sync(stream);}
};
class CheckedRadHostAllocator {
public:
    virtual ~CheckedRadHostAllocator()=default;
    virtual int allocation_charge(size_t usable,uint64_t* upper_bound)const noexcept=0;
    // Precreate ownership before physical allocation. Return an owner even on an
    // error with a retained pointer. Its optional adopt_budget_charge must retain
    // physical memory AND quota on failed free. rad_dev_free(void) is not proof.
    virtual int allocate(size_t usable,uint64_t maximum_charge,std::shared_ptr<HostRowAllocation>* out)noexcept=0;
};
struct RadStagingBinding {
    RadStream stream=nullptr;
    std::shared_ptr<void> lifetime_owner; // actual stream lifetime, not a noop Ctx pointer
    uint64_t device_identity=0,arena_epoch=0;
};
struct RadStagingOptions {size_t max_operations=16;};
struct RadStagingStats {size_t active=0,events=0;uint64_t queued=0,upload_fences=0,reader_fences=0,retired=0,errors=0;bool closed=false;};
class RadEngramBackend;
class RadArenaRowBuffer final:public DeviceRowBuffer {
public:
    std::string_view model_identity()const noexcept override{return model_;}
    uint64_t device_identity()const noexcept override{return device_;}
    uintptr_t address()const noexcept override{return address_;}
    size_t capacity_bytes()const noexcept override{return bytes_;}
    uint64_t arena_epoch()const noexcept{return epoch_;}
private:
    friend class RadEngramBackend;
    RadArenaRowBuffer(std::string model,uint64_t device,uint64_t epoch,uintptr_t address,size_t bytes,
                      std::shared_ptr<void> owner,std::shared_ptr<const void> issuer)
        :model_(std::move(model)),device_(device),epoch_(epoch),address_(address),bytes_(bytes),owner_(std::move(owner)),issuer_(std::move(issuer)){}
    std::string model_;uint64_t device_,epoch_;uintptr_t address_;size_t bytes_;std::shared_ptr<void> owner_;std::shared_ptr<const void> issuer_;
};
class RadEngramBackend final:public TrackedRowTransferBackend {
    struct Completion;
    struct Operation {
        RadEvent upload=nullptr,reader=nullptr;uint64_t generation=0;
        bool active=false,upload_recorded=false,reader_recorded=false,sealed=false,consumers=false,unknown=false;
        int status=RAD_OK;std::weak_ptr<RowTransferRecord> record;std::weak_ptr<Completion> completion;
    };
    struct State {
        std::string model;RadStagingBinding binding;std::thread::id rank_thread;
        std::shared_ptr<CheckedRadHostAllocator> allocator;std::shared_ptr<RadStagingApi> api;
        std::vector<Operation> operations;std::shared_ptr<State> retained;
        mutable std::mutex mutex;RadStagingStats stats;
        bool on_rank()const noexcept{return std::this_thread::get_id()==rank_thread;}
        Operation* find(const std::shared_ptr<RowTransferRecord>& record)noexcept{
            for(auto& op:operations)if(op.active&&op.record.lock()==record)return &op;
            return nullptr;
        }
        void retire(Operation& op,const std::shared_ptr<Completion>& completion,int status)noexcept;
        bool release_events()noexcept{
            if(!on_rank()||!stats.closed||stats.active)return false;
            for(auto& op:operations){if(op.upload){api->destroy_event(op.upload);op.upload=nullptr;}if(op.reader){api->destroy_event(op.reader);op.reader=nullptr;}}
            stats.events=0;retained.reset();return true;
        }
    };
    struct Completion final:RowTransferCompletion,std::enable_shared_from_this<Completion> {
        std::shared_ptr<State> state;size_t index;uint64_t generation;bool latched=false;int result=RAD_OK;std::atomic<bool> cancelled{false};
        Completion(std::shared_ptr<State> s,size_t i,uint64_t g):state(std::move(s)),index(i),generation(g){}
        RowTransferState poll()noexcept override{
            std::lock_guard lock(state->mutex);if(latched)return {true,result};
            if(!state->on_rank())return {false,RAD_E_STATE};
            auto& op=state->operations[index];if(!op.active||op.generation!=generation)return {false,RAD_E_STATE};
            if(!op.sealed||op.unknown||!op.upload_recorded||!op.reader_recorded)return {false,op.status};
            const int upload=state->api->query(op.upload),reader=state->api->query(op.reader);
            if(upload<0||reader<0){op.unknown=true;op.status=upload<0?upload:reader;++state->stats.errors;return {false,op.status};}
            if(!upload||!reader)return {false,op.status};
            state->retire(op,shared_from_this(),op.status);return {true,result};
        }
        void request_cancel()noexcept override{cancelled.store(true,std::memory_order_relaxed);}
    };
public:
    using Consumer=int(*)(RadStream,const DeviceRowTarget&,void*);
    static int create_on_rank(std::string model,RadStagingBinding binding,std::shared_ptr<CheckedRadHostAllocator> allocator,
                              RadStagingOptions options,std::shared_ptr<RadEngramBackend>* out,std::string* why=nullptr,
                              std::shared_ptr<RadStagingApi> api={}){
        if(!out||model.empty()||!binding.stream||!binding.lifetime_owner||!binding.arena_epoch||!allocator||
           !options.max_operations||options.max_operations>256)return refuse(why,RAD_E_SHAPE,"invalid Rad Engram model/stream/owner/bounds");
        try{
            auto state=std::make_shared<State>();state->model=std::move(model);state->binding=std::move(binding);
            state->rank_thread=std::this_thread::get_id();state->allocator=std::move(allocator);state->api=api?std::move(api):std::make_shared<RadStagingApi>();
            state->operations.resize(options.max_operations);auto backend=std::shared_ptr<RadEngramBackend>(new RadEngramBackend(state));state->retained=state;
            for(auto& op:state->operations){
                int status=state->api->create_event(&op.upload);if(op.upload)++state->stats.events;
                if(status==RAD_OK){status=state->api->create_event(&op.reader);if(op.reader)++state->stats.events;}
                if(status!=RAD_OK||!op.upload||!op.reader){state->stats.closed=true;state->release_events();return refuse(why,status!=RAD_OK?status:RAD_E_ABI,"cannot create bounded Rad Engram events");}
            }
            *out=std::move(backend);if(why)why->clear();return RAD_OK;
        }catch(const std::bad_alloc&){return refuse(why,RAD_E_NOMEM,"cannot allocate Rad Engram controls");}
    }
    ~RadEngramBackend(){
        // Final owner: base records can no longer exist. Do not shared_from_this.
        // Wrong-thread destruction intentionally retains native state/resources.
        std::lock_guard lock(state_->mutex);state_->stats.closed=true;state_->release_events();
    }
    int allocation_charge(size_t usable,uint64_t* charge)const noexcept override{
        if(!state_->on_rank())return RAD_E_STATE;
        return state_->allocator->allocation_charge(usable,charge);
    }
    int allocate_host(size_t usable,uint64_t maximum,std::shared_ptr<HostRowAllocation>* out)noexcept override{
        if(!state_->on_rank())return RAD_E_STATE;
        {std::lock_guard lock(state_->mutex);if(state_->stats.closed)return RAD_E_CANCELLED;}
        return state_->allocator->allocate(usable,maximum,out);
    }
    int bind_arena(void* address,size_t bytes,std::shared_ptr<void> owner,uint64_t epoch,
                   std::shared_ptr<RadArenaRowBuffer>* out,std::string* why=nullptr){
        if(!state_->on_rank())return refuse(why,RAD_E_STATE,"Rad Engram method called off rank thread");
        const uintptr_t first=reinterpret_cast<uintptr_t>(address);
        if(!out||!address||!bytes||!owner||epoch!=state_->binding.arena_epoch||first%alignof(float)||bytes>UINTPTR_MAX-first)
            return refuse(why,RAD_E_SHAPE,"invalid caller-owned Rad Engram arena span/epoch");
        {std::lock_guard lock(state_->mutex);if(state_->stats.closed)return refuse(why,RAD_E_CANCELLED,"Rad Engram backend is closed");}
        try{auto span=std::shared_ptr<RadArenaRowBuffer>(new RadArenaRowBuffer(state_->model,state_->binding.device_identity,epoch,first,bytes,std::move(owner),state_));
            *out=std::move(span);if(why)why->clear();return RAD_OK;
        }catch(const std::bad_alloc&){return refuse(why,RAD_E_NOMEM,"cannot allocate Rad Engram arena owner");}
    }
    // Preferred scope: callback may issue consumers only on supplied actual stream
    // and may not enqueue future readers after return. Always seals partial work.
    int consume(const RowTransferTicket& ticket,Consumer callback,void* user)noexcept{
        if(!state_->on_rank())return RAD_E_STATE;
        auto record=ticket.backend_record(*this);if(!record||!callback)return RAD_E_STATE;
        std::lock_guard lock(state_->mutex);auto* op=state_->find(record);if(!op||op->sealed)return RAD_E_STATE;
        op->sealed=true;int status=op->status;
        const auto completion=op->completion.lock();
        if(status==RAD_OK){
            if(state_->stats.closed||record->submission().stop_requested()||(completion&&completion->cancelled.load(std::memory_order_relaxed)))status=RAD_E_CANCELLED;
            else if(!op->upload_recorded)status=RAD_E_STATE;
            else {
                status=state_->api->wait(state_->binding.stream,op->upload);
                if(status==RAD_OK){op->consumers=true;try{status=callback(state_->binding.stream,record->submission().destination,user);}
                    catch(const std::bad_alloc&){status=RAD_E_NOMEM;}catch(...){status=RAD_E_STATE;}}
            }
        }
        const int sealed=state_->api->record(op->reader,state_->binding.stream);
        if(sealed==RAD_OK){op->reader_recorded=true;++state_->stats.reader_fences;}else{op->unknown=true;status=sealed;}
        if(status!=RAD_OK){op->status=status;++state_->stats.errors;}return status;
    }
    int order_upload(const RowTransferTicket& ticket)noexcept{
        if(!state_->on_rank())return RAD_E_STATE;
        auto record=ticket.backend_record(*this);if(!record)return RAD_E_STATE;
        std::lock_guard lock(state_->mutex);auto* op=state_->find(record);if(!op||op->sealed||!op->upload_recorded)return RAD_E_STATE;
        op->consumers=true; // manual handoff permits subsequent caller rad_issue readers
        const int status=state_->api->wait(state_->binding.stream,op->upload);if(status!=RAD_OK){op->unknown=true;op->status=status;}return status;
    }
    // Advanced scope for rad_issue calls outside consume. The caller must seal on
    // every exit and must ensure no later lane/graph replay can read the target.
    int finish_readers(const RowTransferTicket& ticket)noexcept{return seal(ticket,false);}
    int finish_no_readers(const RowTransferTicket& ticket)noexcept{return seal(ticket,true);}
    RadStagingStats native_stats()const noexcept{std::lock_guard lock(state_->mutex);return state_->stats;}
    RowTransferState close()noexcept{
        if(!state_->on_rank())return {false,RAD_E_STATE};
        {std::lock_guard lock(state_->mutex);state_->stats.closed=true;}
        auto result=drain_all(state_->model);
        {std::lock_guard lock(state_->mutex);if(!state_->release_events())result.quiescent=false;}
        return result;
    }
private:
    explicit RadEngramBackend(std::shared_ptr<State> state):state_(std::move(state)){}
    NativeRowEnqueue enqueue_native(const std::shared_ptr<RowTransferRecord>& record)noexcept override{
        if(!state_->on_rank())return {RAD_E_STATE,{},true};
        const auto& submission=record->submission();const auto* arena=dynamic_cast<const RadArenaRowBuffer*>(submission.destination.owner.get());
        if(!arena||arena->issuer_.get()!=state_.get()||arena->epoch_!=state_->binding.arena_epoch||submission.source->model_identity!=state_->model)return {RAD_E_SHAPE,{},true};
        try{
            std::lock_guard lock(state_->mutex);if(state_->stats.closed)return {RAD_E_CANCELLED,{},true};
            size_t index=state_->operations.size();for(size_t i=0;i<state_->operations.size();++i)if(!state_->operations[i].active){index=i;break;}
            if(index==state_->operations.size())return {RAD_E_FULL,{},true};
            auto& op=state_->operations[index];auto completion=std::make_shared<Completion>(state_,index,op.generation+1);
            ++op.generation;op.active=true;op.upload_recorded=op.reader_recorded=op.sealed=op.consumers=op.unknown=false;
            op.status=RAD_OK;op.record=record;op.completion=completion;++state_->stats.active;++state_->stats.queued;
            if(submission.bytes>INT64_MAX){state_->retire(op,completion,RAD_E_SHAPE);return {RAD_E_SHAPE,completion,true};}
            int status=state_->api->copy(reinterpret_cast<void*>(arena->address()+submission.destination.offset),submission.host->bytes().data(),int64_t(submission.bytes),state_->binding.stream);
            if(status==RAD_OK){status=state_->api->record(op.upload,state_->binding.stream);if(status==RAD_OK){op.upload_recorded=true;++state_->stats.upload_fences;}}
            if(status!=RAD_OK){op.unknown=true;op.status=status;++state_->stats.errors;}return {status,completion,false};
        }catch(const std::bad_alloc&){return {RAD_E_NOMEM,{},true};}
    }
    RowTransferState drain_native(const std::shared_ptr<RowTransferRecord>& record)noexcept override{
        if(!state_->on_rank())return {false,RAD_E_STATE};
        std::lock_guard lock(state_->mutex);auto* op=state_->find(record);if(!op)return {true,RAD_OK};
        if(!op->sealed)return {false,op->status!=RAD_OK?op->status:RAD_E_STATE};
        auto completion=op->completion.lock();if(!completion)return {false,RAD_E_STATE};
        const int synchronized=state_->api->synchronize(state_->binding.stream);
        if(synchronized!=RAD_OK){op->unknown=true;op->status=synchronized;++state_->stats.errors;return {false,synchronized};}
        const int status=op->status;state_->retire(*op,completion,status);return {true,status};
    }
    int seal(const RowTransferTicket& ticket,bool no_readers)noexcept{
        if(!state_->on_rank())return RAD_E_STATE;
        auto record=ticket.backend_record(*this);if(!record)return RAD_E_STATE;
        std::lock_guard lock(state_->mutex);auto* op=state_->find(record);if(!op||op->sealed||(no_readers&&op->consumers))return RAD_E_STATE;
        op->sealed=true;const int status=state_->api->record(op->reader,state_->binding.stream);
        if(status==RAD_OK){op->reader_recorded=true;++state_->stats.reader_fences;}else{op->unknown=true;op->status=status;++state_->stats.errors;}
        return status!=RAD_OK?status:op->status;
    }
    std::shared_ptr<State> state_;
};
inline void RadEngramBackend::State::retire(Operation& op,const std::shared_ptr<Completion>& completion,int status)noexcept{
    completion->latched=true;completion->result=status;op.active=false;op.record.reset();op.completion.reset();assert(stats.active);--stats.active;++stats.retired;
}
} // namespace rad::deepseek41::engram
