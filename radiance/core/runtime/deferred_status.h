/* Opt-in asynchronous device-validation latch lifecycle. No state/commands for
 * legacy rows. Payload storage is fixed per Ctx; one outstanding pass at a time. */
#pragma once
#include "rad_deferred_status.h"
#include "rad_device.h"
#include <cstring>

namespace rad {
class DeferredStatus;
struct DeferredStatusLaunch {
    RadLaunchFn body=nullptr;RadDeferredStatusLaunchFn collect=nullptr;void* instance=nullptr;
    RadDeviceStatus* latch=nullptr;uint32_t op=0;
    const DeferredStatus* owner=nullptr;
    static int launch(const RadArgs* args,RadStream stream);
};
class DeferredStatus {
public:
    DeferredStatus()=default;
    DeferredStatus(const DeferredStatus&)=delete;DeferredStatus& operator=(const DeferredStatus&)=delete;
    ~DeferredStatus(){shutdown();}
    bool enabled()const{return device_!=nullptr;}
    RadDeviceStatus* device()const{return device_;}
    bool pending()const{return pending_;}
    bool collecting()const{return enabled()&&pending_&&!sealed_;}
    bool checkpoint_failed()const{return checkpoint_failed_;}
    bool accepts_issue()const{return enabled()&&pending_&&!sealed_&&enqueue_status_==RAD_OK;}
    RadDeviceStatus last()const{return last_;}
    int init(){
        if(enabled())return RAD_E_STATE;
        device_=static_cast<RadDeviceStatus*>(rad_dev_alloc(sizeof(RadDeviceStatus),RAD_MEM_DEVICE));
        host_allocation_=rad_dev_alloc(sizeof(RadDeviceStatus),RAD_MEM_HOST_PINNED);
        host_=host_allocation_?static_cast<RadDeviceStatus*>(rad_dev_host_ptr(host_allocation_)):nullptr;
        int status=RAD_OK;
        if(!device_||!host_)status=RAD_E_NOMEM;
        if(status==RAD_OK)status=rad_event_create(&reset_);
        if(status==RAD_OK)status=rad_event_create(&done_);
        if(status==RAD_OK)status=rad_event_create(&checkpoint_done_);
        if(status<0){shutdown();return status;}
        return RAD_OK;
    }
    // Ctx has drained both lanes before this; a queued copy must never target a
    // freed readback. Frees use the existing engine teardown policy/void ABI.
    void shutdown(){
        if(reset_)rad_event_destroy(reset_);
        if(done_)rad_event_destroy(done_);
        if(checkpoint_done_)rad_event_destroy(checkpoint_done_);
        reset_=done_=checkpoint_done_=nullptr;
        if(host_allocation_)rad_dev_free(host_allocation_,RAD_MEM_HOST_PINNED);
        if(device_)rad_dev_free(device_,RAD_MEM_DEVICE);
        host_allocation_=nullptr;host_=nullptr;device_=nullptr;pending_=false;sealed_=false;event_recorded_=false;enqueue_status_=RAD_OK;stream_=nullptr;checkpoint_failed_=false;
    }
    // Fatal queue failure: retain allocations/events until process exit rather
    // than free a readback or latch an unproven reader may still use. No retry or
    // in-process recovery is promised by the engine's device-fault policy.
    void quarantine_after_failed_drain(){
        host_allocation_=nullptr;host_=nullptr;device_=nullptr;reset_=done_=checkpoint_done_=nullptr;
        pending_=false;sealed_=false;event_recorded_=false;stream_=nullptr;checkpoint_failed_=true;
    }
    int begin(RadStream stream,RadStream existing_lane1){
        if(!enabled()||!stream||pending_||checkpoint_failed_)return RAD_E_STATE;
        stream_=stream;pending_=true;sealed_=false;event_recorded_=false;enqueue_status_=RAD_OK;last_={RAD_OK,0};
        int status=rad_memset_async(device_,0,sizeof(RadDeviceStatus),stream);
        // An existing lane1 otherwise has no ordering behind this pass's reset.
        if(status==RAD_OK&&existing_lane1){status=rad_event_record(reset_,stream);if(status==RAD_OK)status=rad_event_wait(existing_lane1,reset_);}
        if(status<0)enqueue_status_=status;
        return status;
    }
    int seal(){
        if(!pending_||sealed_)return RAD_E_STATE;
        sealed_=true;
        return queue_readback(done_,&event_recorded_);
    }
    // No reset, no seal and no new allocation. The synchronous successful
    // checkpoint completes this readback/event generation before reuse. The
    // caller must have all producers on stream_; Ctx enforces lane0 only.
    int checkpoint(uint32_t op_count){
        if(!collecting())return RAD_E_STATE;
        if(enqueue_status_<0)return reject_checkpoint(enqueue_status_);
        // Final seal uses a DIFFERENT precreated event, so a failed checkpoint
        // record/sync cannot cause reuse of an unproven event generation.
        bool recorded=false;
        const int queued=queue_readback(checkpoint_done_,&recorded);
        if(queued<0)return reject_checkpoint(queued);
        if(!recorded)return reject_checkpoint(RAD_E_STATE);
        const int completed=rad_event_sync(checkpoint_done_);
        if(completed<0)return reject_checkpoint(completed);
        const int observed=observe_readback(op_count);
        return observed<0?reject_checkpoint(observed):observed;
    }
    // A returned in-pass error is fatal until the owning Ctx is recreated.
    // Final seal/await still owns retirement; failure is never quiescence.
    int reject_checkpoint(int status){
        if(!collecting())return RAD_E_STATE;
        checkpoint_failed_=true;
        if(enqueue_status_==RAD_OK)enqueue_status_=status<0?status:RAD_E_STATE;
        return enqueue_status_;
    }
    int await(uint32_t op_count){
        if(!pending_||!sealed_)return RAD_E_STATE;
        // A failed record cannot be queried as if it were a real fence. Fall
        // back to successful stream drain solely to retire owned storage safely.
        const int synchronized=event_recorded_?rad_event_sync(done_):rad_stream_sync(stream_);
        if(synchronized<0)return synchronized;
        pending_=false;
        if(enqueue_status_<0)return enqueue_status_;
        return observe_readback(op_count);
    }
private:
    RadDeviceStatus* device_=nullptr,*host_=nullptr;void* host_allocation_=nullptr;
    RadEvent reset_=nullptr,done_=nullptr,checkpoint_done_=nullptr;RadStream stream_=nullptr;
    bool pending_=false,sealed_=false,event_recorded_=false;int enqueue_status_=RAD_OK;RadDeviceStatus last_{};
    bool checkpoint_failed_=false;
    int queue_readback(RadEvent fence,bool* recorded){
        // A previous successful checkpoint's event is not a fence for a failed
        // NEW record. Only this generation may authorize a read of host_.
        *recorded=false;
        int status=rad_memcpy_async(host_,device_,sizeof(RadDeviceStatus),stream_);
        if(status==RAD_OK){status=rad_event_record(fence,stream_);*recorded=status==RAD_OK;}
        if(status<0&&enqueue_status_==RAD_OK)enqueue_status_=status;
        return status;
    }
    int observe_readback(uint32_t op_count){
        RadDeviceStatus observed{};std::memcpy(&observed,host_,sizeof observed);
        if(observed.code>0||(observed.code<0&&(observed.op==0||observed.op>=op_count))||
           (!observed.code&&observed.op))return RAD_E_ABI;
        last_=observed;
        return observed.code;
    }
};
inline int DeferredStatusLaunch::launch(const RadArgs* args,RadStream stream){
    if(!args||!args->instance)return RAD_E_INVAL;
    auto* wrapper=static_cast<const DeferredStatusLaunch*>(args->instance);
    // The owner is a non-movable Ctx member, alive for every bound issue plan.
    // Check it before touching body/output or the possibly retired latch pointer.
    if(!wrapper->owner||!wrapper->owner->accepts_issue())return RAD_E_STATE;
    RadArgs original=*args;original.instance=wrapper->instance;
    const int status=wrapper->body(&original,stream);if(status<0)return status;
    const int collected=wrapper->collect(&original,wrapper->latch,wrapper->op,stream);
    return collected<0?collected:status;
}
// Legacy two-phase scaffolding: the supplied model phase already includes a
// joined enqueue result AND completed validation. Never wait inside an enqueue
// rank phase before synchronous failures from its peers have been joined.
template<class ModelPhase,class SamplerPhase>
int status_checked_model_sampler(ModelPhase&& model,SamplerPhase&& sampler){
    const int status=model();return status<0?status:sampler();
}
// Actual experimental Engine path. Joining enqueue errors before a blocking
// completion wait is necessary, but does NOT cancel unmatched device work or
// establish bounded teardown. Engine therefore still refuses opted TP>1.
template<class EnqueuePhase,class CheckPhase,class SamplerPhase>
int status_checked_model_sampler(EnqueuePhase&& enqueue,CheckPhase&& check,SamplerPhase&& sampler){
    const int issued=enqueue();if(issued<0)return issued;
    const int completed=check();return completed<0?completed:sampler();
}
} // namespace rad
