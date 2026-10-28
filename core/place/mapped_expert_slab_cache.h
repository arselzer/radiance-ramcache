/* Two fixed-size mapped expert classes in paired retained slabs. Private HOST helper.
 * Allocator callbacks/context must outlive Cache AND every Lease/cleanup handle.
 * Callbacks are noexcept/non-reentrant; quote is deterministic, nondecreasing
 * with requested bytes, and conservative before allocate.
 * Lease release asserts CALLER-proven consumer completion; it proves no GPU event.
 * free(true) retires the opaque handle, not observed physical unpin. Clear is terminal
 * once any backing was obtained: its charge is never spent again in this instance. */
#pragma once
#include "mapped_expert_cache.h"
#include <array>

namespace ds41_mapped {
struct SlabCacheHostTest;
class SlabCache {
    friend struct SlabCacheHostTest;
public:
    using Status=Cache::Status;
    using Allocation=Cache::Allocation;
    using Allocator=Cache::Allocator;
    static constexpr size_t GU=12533760,DN=6266880,guard=64,max_pairs=32;
    struct ClassStats {
        size_t capacity_slots=0,resident_slots=0,idle_slots=0,busy_slots=0,leased_slots=0;
        size_t capacity_logical_bytes=0,resident_logical_bytes=0,idle_logical_bytes=0,leased_logical_bytes=0;
        uint64_t hits=0,misses=0,bypasses=0,leased_tail_bypasses=0;
    };
    struct Stats {
        Cache::Stats cache{};
        std::array<ClassStats,2> classes{};
        size_t retained_slab_owners=0,terminal_free_invoked_owners=0;
        size_t retained_backing_charge=0,terminal_backing_charge=0;
        uint64_t growth_attempts=0,growth_failures=0,hit_bytes=0,miss_bytes=0;
        bool terminal=false;
        size_t charge()const noexcept{return cache.charge();}
    };
private:
    static constexpr size_t none=SIZE_MAX,busy=SIZE_MAX-1;
    struct Entry {
        size_t slab=none,slot=none,previous=none,next=none,leases=0;
        uint64_t generation=0;
        int klass=-1;
        bool present=false;
    };
    struct Slab {
        Allocation allocation{};
        std::array<size_t,2*max_pairs> keys{};
        size_t pairs=0,request=0,ledger_charge=0;
        bool present=false,ready=false,free_safe=true,free_invoked=false;
    };
    struct State {
        std::mutex mutex;
        std::unique_ptr<Entry[]> entries;
        std::unique_ptr<Slab[]> slabs;
        Allocator allocator{};
        Stats stats{};
        size_t count=0,slab_records=0,used_slabs=0,next_pairs=4,minimum_quote=0;
        std::array<size_t,2> first{none,none},last{none,none};
        bool owner_alive=true,retiring=false;
        void(*test_before_copy)(void*)noexcept=nullptr;
        void*test_copy_context=nullptr;
    };
    std::mutex writer_mutex_;
    mutable std::mutex owner_mutex_;
    State*state_=nullptr;
    static size_t bytes_for(int c)noexcept{return c==0?GU:DN;}
    static int klass_for(size_t n)noexcept{return n==GU?0:n==DN?1:-1;}
    static size_t stride_for(int c)noexcept{return bytes_for(c)+3*guard;}
    static size_t request_for(size_t p)noexcept{return p&&p<=max_pairs?p*(GU+DN+6*guard):0;}
    static size_t offset_for(const Slab&s,size_t slot)noexcept{
        return slot<s.pairs?slot*stride_for(0):s.pairs*stride_for(0)+(slot-s.pairs)*stride_for(1);
    }
    static int slot_class(const Slab&s,size_t slot)noexcept{return slot<s.pairs?0:1;}
    static uint8_t*host(const Slab&s,size_t slot)noexcept{return s.allocation.host+offset_for(s,slot)+guard;}
    static const void*device(const Slab&s,size_t slot)noexcept{
        return reinterpret_cast<const void*>(reinterpret_cast<uintptr_t>(s.allocation.device)+offset_for(s,slot)+guard);
    }
    static bool span(const void*p,size_t n,uintptr_t*end)noexcept{
        if(!p||!n||n>UINTPTR_MAX-reinterpret_cast<uintptr_t>(p))return false;
        *end=reinterpret_cast<uintptr_t>(p)+n;return true;
    }
    static bool overlap(const void*a,size_t an,const void*b,size_t bn)noexcept{
        uintptr_t ae=0,be=0;return span(a,an,&ae)&&span(b,bn,&be)&&reinterpret_cast<uintptr_t>(a)<be&&reinterpret_cast<uintptr_t>(b)<ae;
    }
    static size_t sat_add(size_t a,size_t b)noexcept{return b>SIZE_MAX-a?SIZE_MAX:a+b;}
    static void peak(State&s)noexcept{
        auto&c=s.stats.cache;c.peak_charge=std::max(c.peak_charge,c.charge());
        c.budget_ever_exceeded|=c.charge()>c.budget;
        if(c.payload_charge==SIZE_MAX)c.actual_charge_known=false;
    }
    static void unlink(State&s,size_t key)noexcept{
        Entry&e=s.entries[key];const int c=e.klass;
        if(e.previous!=none)s.entries[e.previous].next=e.next;else s.first[c]=e.next;
        if(e.next!=none)s.entries[e.next].previous=e.previous;else s.last[c]=e.previous;
        e.previous=e.next=none;
    }
    static void touch(State&s,size_t key)noexcept{
        Entry&e=s.entries[key];const int c=e.klass;if(s.first[c]==key)return;
        if(e.previous!=none||e.next!=none||s.last[c]==key)unlink(s,key);
        e.next=s.first[c];e.previous=none;
        if(s.first[c]!=none)s.entries[s.first[c]].previous=key;else s.last[c]=key;s.first[c]=key;
    }
    static void initialize_slot(Slab&s,size_t slot)noexcept{
        auto*h=host(s,slot);const size_t n=bytes_for(slot_class(s,slot));
        std::memset(h-guard,0xbd,guard);std::memset(h+n,0,guard);std::memset(h+n+guard,0xbd,guard);
    }
    static bool guards_ok(const Slab&s,size_t slot)noexcept{
        const auto*h=host(s,slot);const size_t n=bytes_for(slot_class(s,slot));
        for(size_t j=0;j<guard;++j)if(h[-ptrdiff_t(guard)+ptrdiff_t(j)]!=0xbd||h[n+j]!=0||h[n+guard+j]!=0xbd)return false;
        return true;
    }
    static Stats snapshot(const State&s){
        Stats out=s.stats;
        for(auto&c:out.classes){c.capacity_slots=c.resident_slots=c.idle_slots=c.busy_slots=c.leased_slots=0;
            c.capacity_logical_bytes=c.resident_logical_bytes=c.idle_logical_bytes=c.leased_logical_bytes=0;}
        for(size_t i=0;i<s.used_slabs;++i){const Slab&b=s.slabs[i];if(!b.ready||b.free_invoked)continue;
            for(size_t j=0;j<2*b.pairs;++j){auto&c=out.classes[slot_class(b,j)];const size_t n=bytes_for(slot_class(b,j));
                ++c.capacity_slots;c.capacity_logical_bytes+=n;
                if(b.keys[j]==none){++c.idle_slots;c.idle_logical_bytes+=n;}
                else if(b.keys[j]==busy)++c.busy_slots;
                else{++c.resident_slots;c.resident_logical_bytes+=n;
                    if(s.entries[b.keys[j]].leases){++c.leased_slots;c.leased_logical_bytes+=n;}}
            }
        }
        out.cache.idle_charge=0;
        for(int c=0;c<2;++c)out.cache.idle_charge+=out.classes[c].idle_slots*stride_for(c);
        return out;
    }
    static Status retire(State&s,std::unique_lock<std::mutex>&lock)noexcept{
        if(s.stats.cache.lease_refs)return Status::Busy;
        //Validate all owned live slots before the first terminal callback.
        for(size_t i=0;i<s.used_slabs;++i){const Slab&b=s.slabs[i];if(!b.present||b.free_invoked)continue;
            if(!b.free_safe)return Status::AllocatorContract;
            if(b.ready)for(size_t j=0;j<2*b.pairs;++j)if(!guards_ok(b,j))return Status::Layout;
        }
        s.retiring=true;s.stats.terminal=true; //no new borrow/insert, including a partial cleanup failure
        for(size_t i=0;i<s.used_slabs;++i){Slab&b=s.slabs[i];if(!b.present||b.free_invoked)continue;
            const Allocation a=b.allocation;
            lock.unlock();const bool invoked=s.allocator.free(s.allocator.context,a);lock.lock();
            if(!invoked){++s.stats.cache.free_failures;return Status::FreeFailed;}
            //The API consumed the owner; never retry it or spend its charge again.
            b.free_invoked=true;b.ready=false;b.allocation.host=nullptr;b.allocation.device=nullptr;
            for(size_t j=0;j<2*b.pairs;++j){const size_t key=b.keys[j];
                if(key!=none&&key!=busy&&s.entries[key].present){s.entries[key].present=false;--s.stats.cache.resident_entries;}
                b.keys[j]=none;
            }
            --s.stats.retained_slab_owners;++s.stats.terminal_free_invoked_owners;
            s.stats.retained_backing_charge-=b.ledger_charge;s.stats.terminal_backing_charge+=b.ledger_charge;
        }
        for(size_t k=0;k<s.count;++k)s.entries[k].present=false;
        s.first={none,none};s.last={none,none};s.stats.cache.resident_entries=0;
        return Status::Ok;
    }
    static void warn_retained()noexcept{
        std::fputs("mapped slab cache: final owner retirement failed; backing retained, NOT confirmed freed\n",stderr);
    }
public:
    class Lease {
        friend class SlabCache;
        State*state_=nullptr;size_t key_=none,bytes_=0;uint64_t generation_=0;
        const uint8_t*host_=nullptr;const void*device_=nullptr;bool active_=false;
    public:
        Lease()=default;Lease(const Lease&)=delete;Lease&operator=(const Lease&)=delete;
        Lease(Lease&&o)noexcept:state_(o.state_),key_(o.key_),bytes_(o.bytes_),generation_(o.generation_),host_(o.host_),device_(o.device_),active_(o.active_){
            o.state_=nullptr;o.host_=nullptr;o.device_=nullptr;o.bytes_=0;o.active_=false;}
        Lease&operator=(Lease&&)=delete;
        ~Lease(){const auto rc=release();if(rc!=Status::Ok&&rc!=Status::Empty)warn_retained();}
        bool active()const noexcept{return active_;}bool empty()const noexcept{return !state_;}
        bool retains_cleanup()const noexcept{return state_&&!active_;}
        const uint8_t*host()const noexcept{return active_?host_:nullptr;}
        const void*device()const noexcept{return active_?device_:nullptr;}
        size_t bytes()const noexcept{return active_?bytes_:0;}
        Stats stats()const{if(!state_)return{};std::lock_guard<std::mutex>l(state_->mutex);return snapshot(*state_);}
        Status release()noexcept{
            if(!state_)return Status::Empty;State*s=state_;std::unique_lock<std::mutex>l(s->mutex);
            if(active_){Entry&e=s->entries[key_];if(!e.present||e.generation!=generation_||!e.leases||!s->stats.cache.lease_refs)return Status::Invalid;
                --e.leases;--s->stats.cache.lease_refs;
                if(!e.leases){--s->stats.cache.leased_entries;s->stats.cache.leased_charge-=stride_for(e.klass);}
                active_=false;host_=nullptr;device_=nullptr;bytes_=0;}
            if(s->owner_alive||s->stats.cache.lease_refs){state_=nullptr;return Status::Ok;}
            const auto rc=retire(*s,l);if(rc!=Status::Ok)return rc;
            state_=nullptr;l.unlock();delete s;return Status::Ok;
        }
    };
    SlabCache()=default;SlabCache(const SlabCache&)=delete;SlabCache&operator=(const SlabCache&)=delete;
    ~SlabCache(){
        std::lock_guard<std::mutex>w(writer_mutex_);std::lock_guard<std::mutex>o(owner_mutex_);State*s=state_;if(!s)return;
        std::unique_lock<std::mutex>l(s->mutex);state_=nullptr;s->owner_alive=false;
        if(s->stats.cache.lease_refs)return;
        const auto rc=retire(*s,l);l.unlock();if(rc==Status::Ok)delete s;else warn_retained();
    }
    static size_t maximum_slab_records(size_t count,size_t budget,Allocator a)noexcept{
        if(!count||!a.quote)return 0;const size_t q=a.quote(a.context,request_for(1));
        return q>=request_for(1)?std::min(count,budget/q):0;
    }
    static size_t metadata_charge(size_t count,size_t slabs)noexcept{
        constexpr size_t fixed=sizeof(SlabCache)+sizeof(State)+192;
        if(!count||!slabs||slabs>(SIZE_MAX-fixed)/sizeof(Slab))return 0;
        const size_t x=fixed+slabs*sizeof(Slab);
        return count<=(SIZE_MAX-x)/sizeof(Entry)?x+count*sizeof(Entry):0;
    }
    Status configure(uint64_t ns,size_t budget,size_t count,Allocator a){
        if(!ns||!a.quote||!a.allocate||!a.free)return Status::Invalid;
        const size_t slabs=maximum_slab_records(count,budget,a),m=metadata_charge(count,slabs);
        if(!m||m>=budget)return Status::Metadata;
        std::lock_guard<std::mutex>w(writer_mutex_);std::lock_guard<std::mutex>o(owner_mutex_);
        if(state_){std::unique_lock<std::mutex>l(state_->mutex);
            if(state_->stats.cache.lease_refs)return Status::Busy;
            if(state_->used_slabs)return Status::Namespace; //terminal/free-unknown budget cannot be spent again
            State*old=state_;state_=nullptr;l.unlock();delete old;}
        auto*s=new(std::nothrow)State;if(!s)return Status::Metadata;
        s->entries.reset(new(std::nothrow)Entry[count]);s->slabs.reset(new(std::nothrow)Slab[slabs]);
        if(!s->entries||!s->slabs){delete s;return Status::Metadata;}
        s->allocator=a;s->count=count;s->slab_records=slabs;s->minimum_quote=a.quote(a.context,request_for(1));s->stats.cache.namespace_id=ns;
        if(!s->minimum_quote){delete s;return Status::Invalid;}
        s->stats.cache.budget=budget;s->stats.cache.metadata=m;s->stats.cache.peak_charge=m;state_=s;return Status::Ok;
    }
    Status clear()noexcept{
        std::lock_guard<std::mutex>w(writer_mutex_);std::lock_guard<std::mutex>o(owner_mutex_);if(!state_)return Status::Ok;
        std::unique_lock<std::mutex>l(state_->mutex);return retire(*state_,l);
    }
    Stats stats()const{std::lock_guard<std::mutex>o(owner_mutex_);if(!state_)return{};std::lock_guard<std::mutex>l(state_->mutex);return snapshot(*state_);}
    Status borrow(uint64_t ns,size_t key,size_t n,Lease*out){
        if(!out||!out->empty()||klass_for(n)<0)return Status::Invalid;
        std::lock_guard<std::mutex>o(owner_mutex_);if(!state_)return Status::Miss;State&s=*state_;std::lock_guard<std::mutex>l(s.mutex);
        if(ns!=s.stats.cache.namespace_id||s.retiring)return Status::Namespace;
        if(key>=s.count)return Status::Invalid;if(s.stats.cache.allocator_contract_broken)return Status::AllocatorContract;
        Entry&e=s.entries[key];const int c=klass_for(n);
        if(!e.present){++s.stats.cache.misses;++s.stats.classes[c].misses;s.stats.miss_bytes+=n;return Status::Miss;}
        if(e.klass!=c)return Status::Layout;
        if(e.leases==SIZE_MAX||s.stats.cache.lease_refs==SIZE_MAX)return Status::Invalid;
        if(!e.leases){++s.stats.cache.leased_entries;s.stats.cache.leased_charge+=stride_for(c);
            s.stats.cache.peak_leased_charge=std::max(s.stats.cache.peak_leased_charge,s.stats.cache.leased_charge);}
        ++e.leases;++s.stats.cache.lease_refs;++s.stats.cache.hits;++s.stats.classes[c].hits;s.stats.hit_bytes+=n;touch(s,key);
        const Slab&b=s.slabs[e.slab];out->state_=&s;out->key_=key;out->bytes_=n;out->generation_=e.generation;
        out->host_=host(b,e.slot);out->device_=device(b,e.slot);out->active_=true;return Status::Hit;
    }
    Status insert(uint64_t ns,size_t key,const void*src,size_t n){
        const int c=klass_for(n);uintptr_t end=0;if(c<0||!span(src,n,&end))return Status::Invalid;
        std::lock_guard<std::mutex>w(writer_mutex_);std::unique_lock<std::mutex>o(owner_mutex_);if(!state_)return Status::Bypass;
        State&s=*state_;std::unique_lock<std::mutex>l(s.mutex);o.unlock();
        if(ns!=s.stats.cache.namespace_id||s.retiring)return Status::Namespace;
        if(key>=s.count)return Status::Invalid;if(s.stats.cache.allocator_contract_broken)return Status::AllocatorContract;
        Entry&e=s.entries[key];if(e.present){if(e.klass!=c)return Status::Layout;touch(s,key);return Status::Existing;}
        if(e.generation==UINT64_MAX)return Status::Invalid;
        for(size_t i=0;i<s.used_slabs;++i){const Slab&b=s.slabs[i];if(!b.free_invoked&&overlap(src,n,b.allocation.host,b.allocation.bytes))return Status::Invalid;}
        size_t bi=none,si=none;
        for(size_t i=0;i<s.used_slabs&&bi==none;++i){const Slab&b=s.slabs[i];if(!b.ready||b.free_invoked)continue;
            for(size_t j=size_t(c)*b.pairs;j<size_t(c+1)*b.pairs;++j)if(b.keys[j]==none){bi=i;si=j;break;}}
        if(bi==none&&s.used_slabs<s.slab_records){
            const size_t room=s.stats.cache.budget-s.stats.cache.charge();size_t p=std::min(s.next_pairs,max_pairs),quote=0;
            for(;p;--p){quote=s.allocator.quote(s.allocator.context,request_for(p));
                if(quote&&(quote<s.minimum_quote||quote<request_for(p))){s.stats.cache.allocator_contract_broken=true;++s.stats.cache.contract_refusals;return Status::AllocatorContract;}
                if(quote&&quote<=room)break;}
            if(p){
                const size_t request=request_for(p);s.stats.cache.reserved_charge=quote;++s.stats.growth_attempts;peak(s);
                Allocation a{};l.unlock();const bool got=s.allocator.allocate(s.allocator.context,request,quote,&a);l.lock();
                s.stats.cache.reserved_charge=0;
                const bool empty=!a.token&&!a.host&&!a.device&&!a.bytes&&!a.charged_bytes;
                if(!got&&empty){++s.stats.cache.allocation_failures;++s.stats.growth_failures;return Status::Allocation;}
                bi=s.used_slabs++;Slab&b=s.slabs[bi];b.present=true;b.pairs=p;b.request=request;b.allocation=a;b.keys.fill(none);b.free_safe=a.token!=nullptr;
                //Only a credible actual charge can refund a conservative quote.
                //An undercharge/missing charge keeps the reservation as a lower bound.
                b.ledger_charge=a.charged_bytes&&a.charged_bytes>=a.bytes?
                    std::max(a.bytes,a.charged_bytes):std::max(quote,a.bytes);
                s.stats.cache.payload_charge=sat_add(s.stats.cache.payload_charge,b.ledger_charge);
                s.stats.retained_backing_charge=sat_add(s.stats.retained_backing_charge,b.ledger_charge);++s.stats.retained_slab_owners;
                uintptr_t he=0,de=0;bool valid=got&&a.token&&span(a.host,a.bytes,&he)&&span(a.device,a.bytes,&de)&&a.bytes>=request&&
                    a.charged_bytes>=a.bytes&&a.charged_bytes<=quote&&uintptr_t(a.host)%64==0&&uintptr_t(a.device)%64==0;
                if(!a.charged_bytes||a.charged_bytes<a.bytes||a.charged_bytes>quote||!a.token)s.stats.cache.actual_charge_known=false;
                for(size_t i=0;i<bi;++i){const Slab&old=s.slabs[i];
                    if(a.token==old.allocation.token||overlap(a.host,a.bytes,old.allocation.host,old.allocation.bytes)||
                       overlap(a.device,a.bytes,old.allocation.device,old.allocation.bytes)){valid=false;b.free_safe=false;s.stats.cache.actual_charge_known=false;}}
                if(overlap(src,n,a.host,a.bytes)){valid=false;b.free_safe=false;s.stats.cache.actual_charge_known=false;}
                peak(s);
                if(!valid){s.stats.cache.allocator_contract_broken=true;++s.stats.cache.contract_refusals;return Status::AllocatorContract;}
                l.unlock();for(size_t j=0;j<2*p;++j)initialize_slot(b,j);l.lock();b.ready=true;
                s.next_pairs=std::min(size_t(32),s.next_pairs*2);si=size_t(c)*p;
            }
        }
        if(bi==none){
            const size_t victim=s.last[c];
            if(victim==none||s.entries[victim].leases){++s.stats.cache.bypasses;++s.stats.classes[c].bypasses;
                if(victim!=none){++s.stats.cache.leased_tail_bypasses;++s.stats.classes[c].leased_tail_bypasses;}return Status::Bypass;}
            Entry&v=s.entries[victim];bi=v.slab;si=v.slot;
            if(!guards_ok(s.slabs[bi],si))return Status::Layout;
            unlink(s,victim);v.present=false;v.slab=v.slot=none;v.klass=-1;
            --s.stats.cache.resident_entries;++s.stats.cache.evictions;
        }
        Slab&b=s.slabs[bi];if(!guards_ok(b,si))return Status::Layout;b.keys[si]=busy;
        const auto hook=s.test_before_copy;void*context=s.test_copy_context;
        l.unlock();initialize_slot(b,si);if(hook)hook(context);std::memcpy(host(b,si),src,n);l.lock();
        e.slab=bi;e.slot=si;e.klass=c;e.present=true;e.leases=0;++e.generation;b.keys[si]=key;
        ++s.stats.cache.resident_entries;++s.stats.cache.inserts;touch(s,key);return Status::Ok;
    }
};
} // namespace ds41_mapped
