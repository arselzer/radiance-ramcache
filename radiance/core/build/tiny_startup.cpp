#include "tiny_startup.h"
#include "../device/device.h"
#include <pthread.h>
#include <cstdlib>
#include <utility>
#include <unistd.h>
namespace rad {
int tiny_thread_defaults(uint64_t* stack, uint64_t* guard) {
    if (!stack || !guard) return RAD_E_INVAL;
#if defined(__linux__)
    pthread_attr_t attr;
    if (pthread_getattr_default_np(&attr) != 0) return RAD_E_UNSUPPORTED;
    size_t s=0,g=0;
    const int a=pthread_attr_getstacksize(&attr,&s), b=pthread_attr_getguardsize(&attr,&g);
    pthread_attr_destroy(&attr);
    if (a || b || !s || s > 8u*1024u*1024u || g > 4096u) return RAD_E_UNSUPPORTED;
    *stack=s; *guard=g; return RAD_OK;
#else
    return RAD_E_UNSUPPORTED;
#endif
}
TinyTicket::~TinyTicket() { reset(); }
TinyTicket::TinyTicket(TinyTicket&& x) noexcept { *this=std::move(x); }
TinyTicket& TinyTicket::operator=(TinyTicket&& x) noexcept {
    if (this!=&x) { reset(); budget_=x.budget_; category_=x.category_; bytes_=x.bytes_;
        role_=x.role_; x.budget_=nullptr; }
    return *this;
}
void TinyTicket::reset() noexcept {
    if (budget_) { auto* b=budget_; budget_=nullptr; b->release(category_,bytes_,role_); }
}
[[noreturn]] void tiny_unretired_fail_stop() noexcept { ::_exit(86); }
void tiny_drain_or_fail(RadStream stream,int device) noexcept {
    try {if(rad_dev_set(device)<0 || rad_stream_sync(stream)<0)tiny_unretired_fail_stop();}
    catch(...) {tiny_unretired_fail_stop();}
}
TinyBudget::~TinyBudget() {
    for(size_t i=0;i<state_.live.size();++i) {
        const uint64_t owned=i==(size_t)TinyCategory::Controls?state_.admission_controls:0;
        if(state_.live[i]!=owned)tiny_unretired_fail_stop();
    }
    if(state_.streams || state_.aux_threads)tiny_unretired_fail_stop();
    // Only the logical admission ends here. Backend-global caches may outlive
    // handles/Budget; no driver or process-cache release is asserted.
}
int TinyBudget::admit() {
    TinyBackendQuote q{};
    const int st=tiny_backend_quote(&q); if (st<0) return st;
    uint64_t s=0,g=0; const int th=tiny_thread_defaults(&s,&g); if(th<0)return th;
    if (q.version!=1 || !q.stream_size || !q.event_size || q.stack_bytes!= (q.host?s:0) ||
        q.guard_bytes!=(q.host?g:0) || q.backend_threads_per_stream!=(q.host?1u:0u)) return RAD_E_UNSUPPORTED;
    const uint64_t stream=q.stream_heap_bytes+q.stream_packet_bytes+q.stream_karg_bytes+q.stream_bounce_bytes+
        q.backend_threads_per_stream*(s+g)+(q.host?q.stream_size+q.stream_control_allowance:0);
    if (stream>caps[(size_t)TinyCategory::BackendStreams]/3 ||
        q.global_control_allowance>caps[(size_t)TinyCategory::Controls] ||
        (!q.host && q.stream_control_allowance+q.stream_size>
         (caps[(size_t)TinyCategory::Controls]-q.global_control_allowance)/3) ||
        s+g>caps[(size_t)TinyCategory::AuxStacks]/2) return RAD_E_NOMEM;
    uint64_t sum=0; for(auto c:caps)sum+=c;
    if(sum!=kCategorySum || sum>kWholeCap)return RAD_E_STATE;
    std::lock_guard<std::mutex> lk(mu_);
    if(admitted_)return RAD_E_STATE;
    backend_=q; aux_bytes_=s+g; state_.whole_reserved=sum;
    // Reserve BEFORE exposing admitted_ or permitting any generic ticket. This
    // named global slice has one admission owner, no externally resettable
    // ticket, and remains unavailable after every stream/control ticket reset.
    const size_t controls=(size_t)TinyCategory::Controls;
    state_.admission_controls=q.global_control_allowance;
    state_.live[controls]=state_.peak[controls]=q.global_control_allowance;
    state_.requests[controls]=q.global_control_allowance?1:0;
    admitted_=true; return RAD_OK;
}
int TinyBudget::take_locked(TinyCategory c,uint64_t n,uint32_t role,TinyTicket* out) {
    const size_t i=(size_t)c;
    if(!admitted_ || !out || out->live() || !n || i>=caps.size() || n>caps[i]-state_.live[i]) return RAD_E_NOMEM;
    state_.live[i]+=n; if(state_.live[i]>state_.peak[i])state_.peak[i]=state_.live[i]; ++state_.requests[i];
    out->budget_=this;out->category_=c;out->bytes_=n;out->role_=role;return RAD_OK;
}
int TinyBudget::take(TinyCategory c,uint64_t n,TinyTicket* out) {
    std::lock_guard<std::mutex> lk(mu_); return take_locked(c,n,0,out);
}
int TinyBudget::take_aux_thread(TinyTicket* out) {
    uint64_t s=0,g=0;const int query=tiny_thread_defaults(&s,&g);if(query<0)return query;
    std::lock_guard<std::mutex> lk(mu_);
    if(s+g!=aux_bytes_)return RAD_E_UNSUPPORTED;
    if(state_.aux_threads>=2)return RAD_E_NOMEM;
    const int st=take_locked(TinyCategory::AuxStacks,aux_bytes_,2,out);
    if(st>=0)++state_.aux_threads;return st;
}
int TinyBudget::take_stream(TinyTicket* backing,TinyTicket* control) {
    TinyBackendQuote actual{};const int query=tiny_backend_quote(&actual);if(query<0)return query;
    std::lock_guard<std::mutex> lk(mu_);
    if(actual.host!=backend_.host || actual.stack_bytes!=backend_.stack_bytes ||
       actual.guard_bytes!=backend_.guard_bytes || actual.karg_host!=backend_.karg_host)
        return RAD_E_UNSUPPORTED;
    if(!admitted_ || !backing || !control || backing==control || backing->live() || control->live() || state_.streams>=3) return RAD_E_STATE;
    const auto& q=backend_;
    const uint64_t n=q.stream_heap_bytes+q.stream_packet_bytes+q.stream_karg_bytes+q.stream_bounce_bytes+
        q.backend_threads_per_stream*(q.stack_bytes+q.guard_bytes)+
        (q.host?q.stream_size+q.stream_control_allowance:0);
    const uint64_t c=q.host?0:q.stream_size+q.stream_control_allowance;
    const size_t bi=(size_t)TinyCategory::BackendStreams,ci=(size_t)TinyCategory::Controls;
    if(n>caps[bi]-state_.live[bi] || c>caps[ci]-state_.live[ci])return RAD_E_NOMEM;
    take_locked(TinyCategory::BackendStreams,n,1,backing);
    if(c)take_locked(TinyCategory::Controls,c,0,control);
    ++state_.streams;return RAD_OK;
}
void TinyBudget::release(TinyCategory c,uint64_t n,uint32_t r) noexcept {
    std::lock_guard<std::mutex> lk(mu_); const size_t i=(size_t)c;
    if(i>=caps.size() || state_.live[i]<n || (r==1&&!state_.streams) || (r==2&&!state_.aux_threads)) tiny_unretired_fail_stop();
    if(i==(size_t)TinyCategory::Controls && state_.live[i]-n<state_.admission_controls)
        tiny_unretired_fail_stop();
    state_.live[i]-=n;if(r==1)--state_.streams;if(r==2)--state_.aux_threads;
}
TinyBudgetSnapshot TinyBudget::snapshot() const { std::lock_guard<std::mutex>lk(mu_);return state_; }
TinyBackendQuote TinyBudget::backend() const { std::lock_guard<std::mutex>lk(mu_);return backend_; }
bool TinyBudget::admitted() const { std::lock_guard<std::mutex>lk(mu_);return admitted_; }
TinyStream::~TinyStream() { close(); }
int TinyStream::open(TinyBudget& b,int device) {
    if(stream_ || device!=0)return RAD_E_STATE;
    TinyTicket bt,ct; int st=b.take_stream(&bt,&ct);if(st<0)return st;
    st=rad_dev_set(device);if(st<0)return st;
    RadStream s=nullptr;st=tiny_backend_stream_create(&s);if(st<0)return st;
    if(!s) return RAD_E_STATE;
    stream_=s;device_=device;backing_=std::move(bt);control_=std::move(ct);return RAD_OK;
}
void TinyStream::close() {
    if(stream_) { tiny_drain_or_fail(stream_,device_);
        rad_stream_destroy(stream_);stream_=nullptr; }
    control_.reset();backing_.reset();
}
}
