/* Standalone HOST prototype. No Core, device API, GPU event or production integration.
 * The allocator/context must outlive the cache AND every borrowed lease/cleanup handle.
 * Allocator callbacks are noexcept, non-reentrant, never wait for GPU consumers, and own independent opaque tokens.
 * quote() reserves an upper bound BEFORE allocation; allocation must honor that bound.
 * A lease release is the CALLER's assertion that every consumer has completed. */
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <utility>

namespace ds41_mapped {
struct CacheHostTest;
class Cache {
    friend struct CacheHostTest;
public:
    enum class Status { Ok, Hit, Miss, Existing, Bypass, Busy, Invalid, Namespace,
                        Layout, Allocation, Metadata, FreeFailed, AllocatorContract, Empty };
    struct Allocation {
        void* token=nullptr;             // independent allocator ownership, NOT a borrowed alias
        uint8_t* host=nullptr;
        const void* device=nullptr;      // opaque DEVICE address; never dereferenced on HOST
        size_t bytes=0,charged_bytes=0;  // usable span and ACTUAL retained backing/allocator charge
    };
    struct Allocator {
        void* context=nullptr;
        size_t (*quote)(void*,size_t) noexcept=nullptr; //0 means unavailable; includes all retained allocation costs
        bool (*allocate)(void*,size_t,size_t,Allocation*) noexcept=nullptr;
        bool (*free)(void*,const Allocation&) noexcept=nullptr; //false retains the SAME independent owner
    };
    struct Stats {
        uint64_t namespace_id=0,hits=0,misses=0,inserts=0,evictions=0,bypasses=0;
        uint64_t allocation_failures=0,free_failures=0,contract_refusals=0,leased_tail_bypasses=0;
        size_t budget=0,metadata=0,payload_charge=0,idle_charge=0,reserved_charge=0,peak_charge=0;
        size_t resident_entries=0,lease_refs=0,leased_entries=0,leased_charge=0,peak_leased_charge=0;
        bool allocator_contract_broken=false,actual_charge_known=true,budget_ever_exceeded=false;
        size_t charge() const noexcept {
            if(payload_charge>SIZE_MAX-metadata||reserved_charge>SIZE_MAX-metadata-payload_charge)return SIZE_MAX;
            return metadata+payload_charge+reserved_charge;
        }
    };

private:
    static constexpr size_t none=SIZE_MAX;
    struct Entry {
        Allocation allocation{};
        size_t bytes=0,charge=0,leases=0,previous=none,next=none;
        uint64_t generation=0;
        bool evicting=false;
    };
    struct State {
        std::mutex mutex;
        std::unique_ptr<Entry[]> entries;
        Allocator allocator{};
        Stats stats{};
        size_t count=0,first=none,last=none;
        Allocation idle{}; //only a failed/invalid allocation awaiting release, NEVER uncharged
        size_t idle_ledger_charge=0;
        bool owner_alive=true,idle_present=false,idle_free_safe=true;
        void (*test_before_copy)(void*)noexcept=nullptr; //private standalone HOST seam only
        void* test_copy_context=nullptr;
    };
    std::mutex writer_mutex_; //insert/configure/clear/destructor; never held by borrow/Lease
    mutable std::mutex owner_mutex_;
    State* state_=nullptr;

    static bool span(const void* p,size_t n,uintptr_t* end) noexcept {
        if(!p||!n||n>UINTPTR_MAX-reinterpret_cast<uintptr_t>(p))return false;
        *end=reinterpret_cast<uintptr_t>(p)+n;return true;
    }
    static bool overlap(const void* a,size_t na,const void* b,size_t nb) noexcept {
        uintptr_t ae=0,be=0;
        return span(a,na,&ae)&&span(b,nb,&be)&&reinterpret_cast<uintptr_t>(a)<be&&reinterpret_cast<uintptr_t>(b)<ae;
    }
    static void peak(State& s) noexcept {
        //The trusted allocator contract keeps these additions bounded; malformed results are
        //recorded as a protocol error and never silently reported as still within budget.
        if(s.stats.payload_charge>SIZE_MAX-s.stats.metadata||s.stats.reserved_charge>SIZE_MAX-s.stats.metadata-s.stats.payload_charge)s.stats.actual_charge_known=false;
        const size_t total=s.stats.charge();s.stats.peak_charge=std::max(s.stats.peak_charge,total);
        s.stats.budget_ever_exceeded|=total>s.stats.budget;
    }
    static void unlink(State& s,size_t key) noexcept {
        Entry&e=s.entries[key];
        if(e.previous!=none)s.entries[e.previous].next=e.next;else s.first=e.next;
        if(e.next!=none)s.entries[e.next].previous=e.previous;else s.last=e.previous;
        e.previous=e.next=none;
    }
    static void touch(State& s,size_t key) noexcept {
        Entry&e=s.entries[key];if(s.first==key)return;
        if(e.previous!=none||e.next!=none||s.last==key)unlink(s,key);
        e.next=s.first;e.previous=none;
        if(s.first!=none)s.entries[s.first].previous=key;else s.last=key;s.first=key;
    }
    static Status evict(State& s,size_t key,std::unique_lock<std::mutex>& lock) noexcept {
        Entry&e=s.entries[key];if(e.leases)return Status::Busy;
        e.evicting=true;const Allocation a=e.allocation;
        lock.unlock();const bool freed=s.allocator.free(s.allocator.context,a);lock.lock();
        if(!freed){e.evicting=false;++s.stats.free_failures;return Status::FreeFailed;}
        unlink(s,key);s.stats.payload_charge-=e.charge;--s.stats.resident_entries;++s.stats.evictions;
        e.allocation={};e.bytes=e.charge=0;e.evicting=false;return Status::Ok;
    }
    static Status free_idle(State& s,std::unique_lock<std::mutex>& lock) noexcept {
        if(!s.idle_present)return Status::Ok;
        if(!s.idle_free_safe)return Status::AllocatorContract; //duplicate opaque owner must never free a live entry
        const Allocation a=s.idle;
        lock.unlock();const bool freed=s.allocator.free(s.allocator.context,a);lock.lock();
        if(!freed){++s.stats.free_failures;return Status::FreeFailed;}
        s.stats.payload_charge-=s.idle_ledger_charge;s.stats.idle_charge=0;
        s.idle={};s.idle_ledger_charge=0;s.idle_present=false;return Status::Ok;
    }
    static Status retire_all(State& s,std::unique_lock<std::mutex>& lock) noexcept {
        if(s.stats.lease_refs)return Status::Busy; //no partial clear when any consumer is live
        Status rc=free_idle(s,lock);if(rc!=Status::Ok)return rc;
        while(s.last!=none){rc=evict(s,s.last,lock);if(rc!=Status::Ok)return rc;}
        return Status::Ok;
    }
    static void warn_retained() noexcept {
        std::fputs("mapped cache: final allocator free failed; owner/backing deliberately retained, NOT confirmed freed\n",stderr);
    }

public:
    class Lease {
        friend class Cache;
        State* state_=nullptr;
        size_t key_=none,bytes_=0;
        uint64_t generation_=0;
        const uint8_t* host_=nullptr;
        const void* device_=nullptr;
        bool active_=false;
    public:
        Lease()=default;
        Lease(const Lease&)=delete;Lease&operator=(const Lease&)=delete;
        Lease(Lease&&other) noexcept :state_(other.state_),key_(other.key_),bytes_(other.bytes_),generation_(other.generation_),
            host_(other.host_),device_(other.device_),active_(other.active_){other.state_=nullptr;other.host_=nullptr;other.device_=nullptr;other.bytes_=0;other.active_=false;}
        //Explicit release is fallible. Do not hide a failed cleanup in move assignment.
        Lease&operator=(Lease&&)=delete;
        ~Lease(){const Status rc=release();if(rc!=Status::Ok&&rc!=Status::Empty)warn_retained();}
        bool active()const noexcept{return active_;}
        bool retains_cleanup()const noexcept{return state_&&!active_;}
        bool empty()const noexcept{return state_==nullptr;}
        const uint8_t* host()const noexcept{return active_?host_:nullptr;}
        const void* device()const noexcept{return active_?device_:nullptr;}
        size_t bytes()const noexcept{return active_?bytes_:0;}
        Stats stats()const {if(!state_)return{};std::lock_guard<std::mutex> lock(state_->mutex);return state_->stats;}
        Status release() noexcept {
            if(!state_)return Status::Empty;
            State*s=state_;std::unique_lock<std::mutex> lock(s->mutex);
            if(active_){
                Entry&e=s->entries[key_];
                if(e.generation!=generation_||!e.leases||!s->stats.lease_refs)return Status::Invalid;
                --e.leases;--s->stats.lease_refs;
                if(!e.leases){--s->stats.leased_entries;s->stats.leased_charge-=e.charge;}
                active_=false;host_=nullptr;device_=nullptr;bytes_=0;
            }
            if(s->owner_alive||s->stats.lease_refs){state_=nullptr;return Status::Ok;}
            //Last lease keeps the state alive after Cache destruction. A failed final free
            //retains a pointer-free cleanup handle so explicit release() can retry safely.
            const Status rc=retire_all(*s,lock);
            if(rc!=Status::Ok)return rc;
            state_=nullptr;lock.unlock();delete s;return Status::Ok;
        }
    };

    Cache()=default;Cache(const Cache&)=delete;Cache&operator=(const Cache&)=delete;
    ~Cache(){
        std::lock_guard<std::mutex> writer(writer_mutex_);
        std::lock_guard<std::mutex> owner(owner_mutex_);State*s=state_;if(!s)return;
        std::unique_lock<std::mutex> lock(s->mutex);state_=nullptr;s->owner_alive=false;
        if(s->stats.lease_refs)return; //last Lease, not this destructor, owns final retirement
        const Status rc=retire_all(*s,lock);lock.unlock();
        if(rc==Status::Ok)delete s;else warn_retained();
    }
    static size_t metadata_charge(size_t entries) noexcept {
        constexpr size_t fixed=sizeof(Cache)+sizeof(State)+128; //both fixed heap allocations' overhead reserve
        return entries&&entries<=(SIZE_MAX-fixed)/sizeof(Entry)?fixed+entries*sizeof(Entry):0;
    }
    Status configure(uint64_t namespace_id,size_t budget,size_t count,Allocator allocator){
        const size_t metadata=metadata_charge(count);
        if(!namespace_id||!metadata||metadata>=budget||!allocator.quote||!allocator.allocate||!allocator.free)return Status::Invalid;
        std::lock_guard<std::mutex> writer(writer_mutex_);
        std::lock_guard<std::mutex> owner(owner_mutex_);
        if(state_){
            std::unique_lock<std::mutex> lock(state_->mutex);const Status rc=retire_all(*state_,lock);
            if(rc!=Status::Ok)return rc;State*old=state_;state_=nullptr;lock.unlock();delete old;
        }
        State*s=new(std::nothrow)State;if(!s)return Status::Metadata;
        s->entries.reset(new(std::nothrow)Entry[count]);if(!s->entries){delete s;return Status::Metadata;}
        s->allocator=allocator;s->count=count;s->stats.namespace_id=namespace_id;
        s->stats.budget=budget;s->stats.metadata=metadata;s->stats.peak_charge=metadata;state_=s;return Status::Ok;
    }
    Status clear() noexcept {
        std::lock_guard<std::mutex> writer(writer_mutex_);
        std::lock_guard<std::mutex> owner(owner_mutex_);if(!state_)return Status::Ok;
        std::unique_lock<std::mutex> lock(state_->mutex);const Status rc=retire_all(*state_,lock);
        if(rc!=Status::Ok)return rc;State*old=state_;state_=nullptr;lock.unlock();delete old;return Status::Ok;
    }
    Stats stats()const {
        std::lock_guard<std::mutex> owner(owner_mutex_);if(!state_)return{};
        std::lock_guard<std::mutex> lock(state_->mutex);return state_->stats;
    }
    Status borrow(uint64_t namespace_id,size_t key,size_t bytes,Lease*out){
        if(!out||!out->empty()||!bytes)return Status::Invalid;
        std::lock_guard<std::mutex> owner(owner_mutex_);if(!state_)return Status::Miss;
        State&s=*state_;std::lock_guard<std::mutex> lock(s.mutex);
        if(namespace_id!=s.stats.namespace_id)return Status::Namespace;
        if(key>=s.count)return Status::Invalid;
        if(s.stats.allocator_contract_broken)return Status::AllocatorContract;
        Entry&e=s.entries[key];if(!e.allocation.token||e.evicting){++s.stats.misses;return Status::Miss;}
        if(e.bytes!=bytes)return Status::Layout;
        if(e.leases==SIZE_MAX||s.stats.lease_refs==SIZE_MAX)return Status::Invalid;
        if(!e.leases){++s.stats.leased_entries;s.stats.leased_charge+=e.charge;s.stats.peak_leased_charge=std::max(s.stats.peak_leased_charge,s.stats.leased_charge);}
        ++e.leases;++s.stats.lease_refs;++s.stats.hits;touch(s,key);
        out->state_=&s;out->key_=key;out->bytes_=bytes;out->generation_=e.generation;
        out->host_=e.allocation.host;out->device_=e.allocation.device;out->active_=true;return Status::Hit;
    }
    Status insert(uint64_t namespace_id,size_t key,const void*src,size_t bytes){
        uintptr_t end=0;if(!span(src,bytes,&end))return Status::Invalid;
        std::lock_guard<std::mutex> writer(writer_mutex_);
        std::unique_lock<std::mutex> owner(owner_mutex_);if(!state_)return Status::Bypass;
        State&s=*state_;std::unique_lock<std::mutex> lock(s.mutex);
        owner.unlock(); //writer_mutex_ keeps State/configuration stable; hits no longer wait for miss work
        if(namespace_id!=s.stats.namespace_id)return Status::Namespace;
        if(key>=s.count)return Status::Invalid;
        if(s.stats.allocator_contract_broken)return Status::AllocatorContract;
        Entry&e=s.entries[key];
        if(e.allocation.token){if(e.bytes!=bytes)return Status::Layout;touch(s,key);return Status::Existing;}
        for(size_t k=0;k<s.count;++k)if(s.entries[k].allocation.token&&overlap(src,bytes,s.entries[k].allocation.host,s.entries[k].allocation.bytes))return Status::Invalid;
        if(s.idle_present)return Status::FreeFailed; //no new allocation while failed cleanup owns backing
        if(e.generation==UINT64_MAX)return Status::Invalid;
        const size_t quoted=s.allocator.quote(s.allocator.context,bytes);
        if(!quoted||quoted<bytes||quoted>s.stats.budget-s.stats.metadata){++s.stats.bypasses;return Status::Bypass;}
        while(s.stats.payload_charge>s.stats.budget-s.stats.metadata-quoted){
            if(s.last==none)return Status::Invalid;
            if(s.entries[s.last].leases){++s.stats.bypasses;++s.stats.leased_tail_bypasses;return Status::Bypass;}
            const Status rc=evict(s,s.last,lock);if(rc!=Status::Ok)return rc;
        }
        s.stats.reserved_charge=quoted;peak(s);Allocation a{};
        lock.unlock();const bool allocated=s.allocator.allocate(s.allocator.context,bytes,quoted,&a);lock.lock();
        s.stats.reserved_charge=0;
        if(!allocated&&!a.token&&!a.host&&!a.device&&!a.bytes&&!a.charged_bytes){++s.stats.allocation_failures;return Status::Allocation;}
        //Own even malformed output before inspecting it. The independent token is sufficient
        //for allocator.free; a failed release keeps this one idle owner and its honest charge.
        s.idle=a;s.idle_present=true;s.idle_free_safe=true;
        s.idle_ledger_charge=a.charged_bytes?std::max(a.charged_bytes,a.bytes):std::max(quoted,a.bytes);
        if(a.charged_bytes<a.bytes)s.stats.actual_charge_known=false;
        if(s.idle_ledger_charge>s.stats.budget-s.stats.metadata-s.stats.payload_charge)s.stats.budget_ever_exceeded=true;
        if(s.idle_ledger_charge>SIZE_MAX-s.stats.payload_charge){s.idle_ledger_charge=quoted;s.stats.actual_charge_known=false;s.stats.peak_charge=SIZE_MAX;}
        s.stats.payload_charge+=s.idle_ledger_charge;s.stats.idle_charge=s.idle_ledger_charge;peak(s);
        uintptr_t he=0,de=0;
        bool valid=allocated&&a.token&&a.bytes>=bytes&&a.charged_bytes>=a.bytes&&a.charged_bytes<=quoted&&
                   span(a.host,a.bytes,&he)&&span(a.device,a.bytes,&de)&&!overlap(a.host,a.bytes,src,bytes);
        for(size_t k=0;k<s.count;++k)if(s.entries[k].allocation.token){
            const Allocation&old=s.entries[k].allocation;
            if(a.token==old.token){s.idle_free_safe=false;s.stats.actual_charge_known=false;}
            valid=valid&&a.token!=old.token&&!overlap(a.host,a.bytes,old.host,old.bytes)&&!overlap(a.device,a.bytes,old.device,old.bytes);
        }
        if(!valid){
            s.stats.allocator_contract_broken=true;++s.stats.contract_refusals;
            if(!a.charged_bytes||!a.token)s.stats.actual_charge_known=false;
            (void)free_idle(s,lock);return Status::AllocatorContract;
        }
        const auto hook=s.test_before_copy;void*hook_context=s.test_copy_context;
        lock.unlock();if(hook)hook(hook_context);std::memcpy(a.host,src,bytes);lock.lock();
        //No writer can replace State while writer_mutex_ is held. Borrow metadata and
        //Lease releases may proceed during allocation/copy; unpublished idle bytes never escape.
        e.allocation=a;e.bytes=bytes;e.charge=a.charged_bytes;
        ++e.generation;s.idle={};s.idle_ledger_charge=0;s.idle_present=false;s.stats.idle_charge=0;
        ++s.stats.inserts;++s.stats.resident_entries;touch(s,key);return Status::Ok;
    }
};
} //namespace ds41_mapped
