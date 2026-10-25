#include "build/tiny_startup.h"
#include "build/tiny_loader.h"
#include "build/tiny_component_quotes.h"
#include "device/device.h"
#include "rad_core.h"
#include "sample/sampler.h"
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>
namespace {
std::atomic<long> fail_new{-1},new_attempts{0};
std::atomic<bool> forbid_release{false};
std::atomic<int> fail_pinned{-1},pinned_attempts{0},pinned_live{0};
std::atomic<bool> trace_pinned{false},trace_heap{false};
std::atomic<uint64_t> heap_requested{0};
std::atomic<int> backend_fault{0},backend_fault_seen{0};
std::atomic<uint64_t> backend_real_success{0};
bool fault_backend(int which,int status){
    if(status<0)return false;
    int expected=which;
    if(!backend_fault.compare_exchange_strong(expected,0))return false;
    ++backend_fault_seen;++backend_real_success;return true;
}
size_t checks=0,groups=0;
void need(bool value,const char* why){++checks;if(!value)throw std::runtime_error(why);}
void good(int st,const char* why){need(st==RAD_OK,why);}
void* allocate(size_t n){
    const long at=new_attempts.fetch_add(1);
    if(fail_new.load()==at)throw std::bad_alloc();
    if(trace_heap.load())heap_requested.fetch_add(n);
    void* p=std::malloc(n?n:1);if(!p)throw std::bad_alloc();return p;
}
void free_memory(void* p) noexcept {if(forbid_release.load())::_exit(89);std::free(p);}
void no_live(const rad::TinyBudget& b) {
    const auto s=b.snapshot();for(size_t i=0;i<s.live.size();++i) {
        const uint64_t owned=i==(size_t)rad::TinyCategory::Controls?s.admission_controls:0;
        need(s.live[i]==owned,"live ticket after cleanup beyond admission-owned credit");
    }
    need(!s.streams&&!s.aux_threads,"live stream/thread role after cleanup");
}
void quote_and_ticket_tests(rad::TinyBudget& b){++groups;
    const auto q=b.backend();need(q.host&&q.task_size&&q.queue_depth==1024,"actual HOST private quote");
    need(q.stream_heap_bytes==q.task_size*q.queue_depth,"actual Task ring quote");
    need(q.stack_bytes<=8u*1024u*1024u&&q.guard_bytes<=4096,"actual pthread defaults");
    const auto s=b.snapshot();need(s.whole_reserved==135244800,"fixed category reservation");
    std::cout<<"ACTUAL_BACKEND host="<<q.host<<" streamSize="<<q.stream_size<<" eventSize="<<q.event_size
        <<" taskSize="<<q.task_size<<" queueDepth="<<q.queue_depth<<" ringRequested="<<q.stream_heap_bytes
        <<" perStreamControlAllowance="<<q.stream_control_allowance<<" stack="<<q.stack_bytes<<" guard="<<q.guard_bytes
        <<" globalControlAllowance="<<q.global_control_allowance<<" wholeReserved="<<s.whole_reserved<<" cap="<<rad::TinyBudget::kWholeCap<<"\n";
    const size_t ci=(size_t)rad::TinyCategory::Controls;
    need(s.admission_controls==q.global_control_allowance&&s.live[ci]==q.global_control_allowance&&s.peak[ci]==q.global_control_allowance,"global Controls credit has explicit admission owner");
    rad::TinyTicket controls_full,controls_remainder,controls_extra;
    need(b.take(rad::TinyCategory::Controls,rad::TinyBudget::caps[ci],&controls_full)==RAD_E_NOMEM&&!controls_full.live(),"admission-owned global credit cannot be spent by generic full-cap ticket");
    good(b.take(rad::TinyCategory::Controls,rad::TinyBudget::caps[ci]-s.admission_controls,&controls_remainder),"only unreserved Controls remainder is available");
    need(b.take(rad::TinyCategory::Controls,1,&controls_extra)==RAD_E_NOMEM&&!controls_extra.live(),"remainder plus global credit exactly fills category");
    controls_remainder.reset();need(b.snapshot().live[ci]==s.admission_controls,"global credit survives generic ticket reset");
    rad::TinyTicket full,extra;
    good(b.take(rad::TinyCategory::Loader,2097152,&full),"full category ticket");
    need(b.take(rad::TinyCategory::Loader,1,&extra)==RAD_E_NOMEM&&!extra.live(),"refuse over cap before allocation");
    full.reset();
    std::array<rad::TinyTicket,3> threads;
    good(b.take_aux_thread(&threads[0]),"first auxiliary thread");good(b.take_aux_thread(&threads[1]),"second auxiliary thread");
    need(b.take_aux_thread(&threads[2])<0&&!threads[2].live(),"third auxiliary thread refused");
    for(auto& t:threads)t.reset();
    std::array<rad::TinyTicket,4> backing,controls;
    for(size_t i=0;i<3;++i)good(b.take_stream(&backing[i],&controls[i]),"three stream admission");
    need(b.take_stream(&backing[3],&controls[3])<0&&!backing[3].live()&&!controls[3].live(),"fourth stream refused atomically");
    for(auto& t:backing)t.reset();
    for(auto& t:controls)t.reset();
    need(b.snapshot().live[ci]==s.admission_controls,"global credit survives all stream-control ticket resets");
    rad::TinyTicket occupied;good(b.take(rad::TinyCategory::Source,17,&occupied),"occupied output setup");
    need(b.take(rad::TinyCategory::Source,19,&occupied)<0&&occupied.live(),"occupied ticket preserved");occupied.reset();
    rad::SamplerConfig c;c.n_vocab=c.n_vocab_local=320;c.max_seqs=1;c.max_hist=2048;c.max_candidates=1024;
    rad::TinySamplerQuote sampler{};good(rad::Sampler::tiny_startup_quote(c,&sampler),"actual Sampler private quote");
    need(sampler.storage.pinned_requests==11&&sampler.storage.pinned_bytes==18984&&sampler.storage.pinned_page_bytes==49152,"actual eleven Sampler requests");
    sampler.rows=777;c.max_spec=1;need(rad::Sampler::tiny_startup_quote(c,&sampler)<0&&sampler.rows==777,"unsupported Sampler quote preserves caller");
    rad::TinyStepQuote step{};const int32_t eog[]={256,258,256};good(rad::tiny_step_state_quote(eog,&step),"actual StepSlot quote");
    need(step.unique_eog==2&&step.sizeof_step_slot,"actual StepSlot and unique EOG");
    no_live(b);
}
void host_constructor_tests(){++groups;
    // New failure ordinals target actual HostStream, vector<Task>, and std::thread
    // construction. No backend/private struct is mirrored or substituted.
    for(long fail=0;fail<3;++fail) {
        RadStream sentinel=(RadStream)(uintptr_t)1;
        new_attempts=0;fail_new=fail;const int st=rad_stream_create(&sentinel,0);fail_new=-1;
        need(st<0&&sentinel==(RadStream)(uintptr_t)1,"HOST partial constructor atomic refusal");
        RadStream s=nullptr;good(rad_stream_create(&s,0),"HOST constructor recovery");good(rad_stream_sync(s),"HOST real worker retirement");rad_stream_destroy(s);
    }
}
struct Fill {uint8_t value=0;std::atomic<uint32_t>* calls=nullptr;};
int fill(void* context,uint8_t* dst,int64_t bytes){auto& f=*(Fill*)context;std::memset(dst,f.value,(size_t)bytes);++*f.calls;return RAD_OK;}
void loader_tests(rad::TinyBudget& b){++groups;
    // Program metadata and caller-owned destination are separate named owners.
    // The loader's ticket covers only its own staging/controls/worker Bounces.
    rad::TinyTicket metadata,destination;
    good(b.take(rad::TinyCategory::EngineMetadata,8u<<20,&metadata),"fixture Program metadata admission");
    good(b.take(rad::TinyCategory::Pools,12u*65536u,&destination),"caller destination admission");
    {
        rad::Program p;p.weights.resize(13);
        void* backing=rad_dev_alloc(12*65536,RAD_MEM_DEVICE);need(backing,"real backend destination allocation");
        struct Owner {void* p;~Owner(){rad_dev_free(p,RAD_MEM_DEVICE);}} owner{backing};
        std::array<rad::TinyLoadJob,12> jobs{};std::array<Fill,12> fills{};std::atomic<uint32_t> calls{0};
        for(size_t i=0;i<jobs.size();++i) {
            auto& w=p.weights[i+1];w.name="tiny.foundation."+std::to_string(i);w.stored_bytes=65536;
            w.identity=true;w.n_sel=1;w.enc.n_planes=1;w.enc.plane[0].dtype=RAD_BF16;
            w.sel_rows[0]=128;w.sel_cols[0]=256;
            auto& j=jobs[i];j.program=&p;j.weight=i+1;j.destination=(uint8_t*)backing+i*65536;
            j.device_destination=true;j.input_bytes=65536;j.fill=fill;j.fill_context=&fills[i];
            fills[i].value=(uint8_t)(i+1);fills[i].calls=&calls;
        }
        rad::TinyLoaderQuote q{};good(rad::tiny_loader_quote(jobs.data(),jobs.size(),&q),"actual RelayoutPool private quote");
        need(q.jobs==12&&q.input_pages==12*65536&&q.page_upper<=2097152&&q.staged_size&&q.bounce_size&&q.pool_size,"actual bounded loader quote");
        std::cout<<"ACTUAL_LOADER jobs="<<q.jobs<<" inputPages="<<q.input_pages<<" inputArena="<<q.input_arena
            <<" pinnedSlots="<<q.pinned_slots<<" sizeofStaged="<<q.staged_size<<" sizeofPool="<<q.pool_size
            <<" sizeofBounce="<<q.bounce_size<<" sizeofUplink="<<q.uplink_size<<" sizeofSlot="<<q.slot_size
            <<" eventSize="<<q.event_size<<" controlAllowance="<<q.control_allowance<<" requested="<<q.requested_bytes<<" pageUpper="<<q.page_upper<<"\n";
        auto bad=jobs;bad[0].input_bytes=131073;rad::TinyLoaderQuote keep;keep.jobs=777;
        need(rad::tiny_loader_quote(bad.data(),bad.size(),&keep)<0&&keep.jobs==777,"loader metadata refusal preserves quote");
        bad=jobs;bad[0].plane_offset[0]=65535;need(rad::tiny_loader_quote(bad.data(),bad.size(),&keep)<0&&keep.jobs==777,"actual selected plane span refused");
        bad=jobs;bad[0].destination=bad[1].destination;need(rad::tiny_loader_quote(bad.data(),bad.size(),&keep)<0,"overlapping exact destination refused");
        const auto baseline=b.snapshot();
        auto retired=[&](const rad::TinyLoaderReceipt& r){
            need(r.all_retired&&r.workers_started==r.workers_joined&&r.submissions_stopped,"retirement after every worker join");
            need(r.pinned_allocations==r.pinned_releases&&pinned_live.load()==0,"actual pinned backing released after retirement");
            auto s=b.snapshot();need(s.live==baseline.live&&!s.streams&&!s.aux_threads,"loader/category tickets fully retired");
        };
        trace_pinned=true;pinned_attempts=0;fail_pinned=-1;
        heap_requested=0;trace_heap=true;
        rad::TinyLoaderReceipt r{};good(rad::tiny_loader_run(jobs.data(),jobs.size(),b,rad::TinyLoaderFault::None,&r),"real bounded loader baseline");
        trace_heap=false;
        need(heap_requested.load()<=q.requested_bytes-q.input_arena-q.pinned_slots+
            b.backend().stream_heap_bytes+b.backend().stream_size+b.backend().stream_control_allowance,
            "actual requested C++ controls within separately named allowances");
        retired(r);need(r.workers_started==2&&r.live_bounces_at_final_drain==2&&r.real_uplink_drains==1&&r.copies_enqueued==12&&r.records_enqueued==12,"two retained Bounces through actual shared drain");
        const uint8_t* data=(const uint8_t*)rad_dev_host_ptr(backing);need(data,"HOST actual mapped destination alias");
        for(size_t i=0;i<jobs.size();++i)for(size_t k=0;k<65536;++k)need(data[i*65536+k]==fills[i].value,"actual loader byte result");
        for(auto fault:{rad::TinyLoaderFault::BeforeSecondWorker,rad::TinyLoaderFault::BeforePinnedAllocation,
                       rad::TinyLoaderFault::AfterCopyBeforeRecord,rad::TinyLoaderFault::AfterRecord,rad::TinyLoaderFault::BeforeSlotWait}) {
            r={};const int st=rad::tiny_loader_run(jobs.data(),jobs.size(),b,fault,&r);
            need(st<0&&r.faults_triggered==1,"intended semantic loader fault actually triggered");retired(r);
            if(fault==rad::TinyLoaderFault::BeforeSecondWorker)need(r.workers_started==1&&r.pinned_releases==2,"partial worker startup retained both owners");
            if(fault==rad::TinyLoaderFault::AfterCopyBeforeRecord)need(r.copies_enqueued>r.records_enqueued&&r.real_uplink_drains==1&&r.live_bounces_at_final_drain==2,"unrecorded actual copy drained before free");
            if(fault==rad::TinyLoaderFault::AfterRecord||fault==rad::TinyLoaderFault::BeforeSlotWait)need(r.records_enqueued&&r.real_uplink_drains==1,"record/wait refusal followed actual successful shared drain");
        }
        for(int at=0;at<2;++at) {
            pinned_attempts=0;fail_pinned=at;r={};const int st=rad::tiny_loader_run(jobs.data(),jobs.size(),b,rad::TinyLoaderFault::None,&r);fail_pinned=-1;
            need(st==RAD_E_NOMEM&&pinned_attempts.load()==at+1,"actual pinned allocator OOM witness");retired(r);
        }
        for(int operation=1;operation<=3;++operation) {
            backend_fault=operation;backend_fault_seen=0;backend_real_success=0;r={};
            const int st=rad::tiny_loader_run(jobs.data(),jobs.size(),b,rad::TinyLoaderFault::None,&r);
            backend_fault=0;need(st<0&&backend_fault_seen==1&&backend_real_success==1,"intended actual copy/record/wait result fault witnessed");
            retired(r);need(r.real_uplink_drains==1&&r.live_bounces_at_final_drain==2,"backend return refusal retains backing through real drain");
        }
        for(long at=0;at<2;++at) {
            r={};new_attempts=0;fail_new=at;
            const int st=rad::tiny_loader_run(jobs.data(),jobs.size(),b,rad::TinyLoaderFault::None,&r);
            fail_new=-1;need(st==RAD_E_NOMEM&&new_attempts.load()>at&&!r.workers_started,"actual early staging/control OOM witness");
            need(!pinned_live.load()&&b.snapshot().live==baseline.live,"pre-worker OOM releases tickets/partial controls");
        }
        trace_pinned=false;
        // Hold full category: no stage or pinned request may occur on refusal.
        rad::TinyTicket blocked;good(b.take(rad::TinyCategory::Loader,2097152,&blocked),"precharge refusal setup");
        const int attempts=pinned_attempts;r={};r.category_bytes=999;
        need(rad::tiny_loader_run(jobs.data(),jobs.size(),b,rad::TinyLoaderFault::None,&r)==RAD_E_NOMEM&&r.category_bytes==999&&pinned_attempts.load()==attempts,"budget refusal precedes payload allocation/publication");
    }
    destination.reset();metadata.reset();no_live(b);
}
void actual_negative_drain_test(){++groups;
    const pid_t pid=::fork();need(pid>=0,"fail-stop subprocess fork");
    if(pid==0){::alarm(10);rad::TinyBudget b;if(b.admit()<0)::_exit(80);rad::TinyStream stream;if(stream.open(b,0)<0)::_exit(81);
        void* gate=nullptr;if(rad_stream_gate(stream.get(),&gate)<0||!gate)::_exit(82);
        if(rad_stream_sync(stream.get())!=RAD_E_STATE)::_exit(83); // genuine backend negative
        forbid_release=true;stream.close();::_exit(84);
    }
    int status=0;need(::waitpid(pid,&status,0)==pid,"fail-stop subprocess reap");
    need(WIFEXITED(status)&&WEXITSTATUS(status)==86,"genuine unretired drain fails stop before destruction/free");
}
}
void* operator new(size_t n){return allocate(n);}void* operator new[](size_t n){return allocate(n);}
void operator delete(void* p) noexcept{free_memory(p);}void operator delete[](void* p) noexcept{free_memory(p);}
void operator delete(void* p,size_t) noexcept{free_memory(p);}void operator delete[](void* p,size_t) noexcept{free_memory(p);}
extern "C" void* __real_rad_dev_alloc(int64_t,int);
extern "C" void __real_rad_dev_free(void*,int);
extern "C" void* __wrap_rad_dev_alloc(int64_t bytes,int kind){
    if(trace_pinned.load()&&kind==RAD_MEM_HOST_PINNED){const int at=pinned_attempts.fetch_add(1);if(fail_pinned.load()==at)return nullptr;
        void* p=__real_rad_dev_alloc(bytes,kind);if(p)++pinned_live;return p;}
    return __real_rad_dev_alloc(bytes,kind);
}
extern "C" void __wrap_rad_dev_free(void* p,int kind){
    if(forbid_release.load())::_exit(88);
    if(trace_pinned.load()&&kind==RAD_MEM_HOST_PINNED&&p)--pinned_live;
    __real_rad_dev_free(p,kind);
}
extern "C" int __real_rad_memcpy_async(void*,const void*,int64_t,RadStream);
extern "C" int __real_rad_event_record(RadEvent,RadStream);
extern "C" int __real_rad_event_sync(RadEvent);
extern "C" int __wrap_rad_memcpy_async(void* d,const void* s,int64_t n,RadStream stream){
    const int st=__real_rad_memcpy_async(d,s,n,stream);return fault_backend(1,st)?RAD_E_STATE:st;
}
extern "C" int __wrap_rad_event_record(RadEvent event,RadStream stream){
    const int st=__real_rad_event_record(event,stream);return fault_backend(2,st)?RAD_E_STATE:st;
}
extern "C" int __wrap_rad_event_sync(RadEvent event){
    const int st=__real_rad_event_sync(event);return fault_backend(3,st)?RAD_E_STATE:st;
}
int main(){try {
    need(rad::device_is_host(),"manual checker requires matching HOST Core");rad::TinyBudget budget;good(budget.admit(),"actual cold backend/thread admission");
    quote_and_ticket_tests(budget);host_constructor_tests();loader_tests(budget);actual_negative_drain_test();no_live(budget);
    std::cout<<"PASS foundation groups="<<groups<<" checks="<<checks<<" no Engine/model execution\n";return 0;
} catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<" checks="<<checks<<"\n";return 1;}}
