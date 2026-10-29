/* Private copy-source cache adapter. Immutable leases use the unchanged HOST/GPU-proven
 * helper. This adapter does NOT prove consumer completion; Mover owns that obligation.
 * V6 adds opt-in paired retained slabs; V5 per-entry owners remain default.
 * Layout::Packed (slabs only) holds the CPU-only packed form (tools/ds41-host-experts
 * ds41_host_experts.h) in slots of the configured sizes. Only borrow_packed and reserve_packed
 * reach it, and their leases say packed(); every block_mxfp4 call (borrow, borrow_readonly,
 * reserve_fill, insert: the card's copy path) is turned away as a miss or bypass and counted, so
 * no copy to the card ever reads a packed slot and no block bytes ever land in one. */
#pragma once
#include "mapped_expert_cache.h"
#include "mapped_expert_slab_cache.h"
#include "../rad_internal.h"
#include "../device/device.h"
#include <atomic>
#include <cstdlib>
#include <chrono>
#include <memory>
#include <mutex>
#include <new>

namespace rad {
struct NativeMappedFileCacheHostTest;
class NativeMappedFileCache {
    friend struct NativeMappedFileCacheHostTest; //private HOST fixture seam, never a runtime option
public:
    using Cache=ds41_mapped::Cache;
    using SlabCache=ds41_mapped::SlabCache;
    class Write;
    class Lease {
        friend class NativeMappedFileCache;
        friend class Write;
        Cache::Lease regular_;
        SlabCache::Lease slab_;
        bool packed_=false;
    public:
        Lease()=default;Lease(const Lease&)=delete;Lease&operator=(const Lease&)=delete;
        Lease(Lease&&o)noexcept:regular_(std::move(o.regular_)),slab_(std::move(o.slab_)),packed_(o.packed_){o.packed_=false;}
        Lease&operator=(Lease&&)=delete;
        bool active()const noexcept{return regular_.active()||slab_.active();}
        bool empty()const noexcept{return regular_.empty()&&slab_.empty();}
        //The slot holds the packed form, which only the host expert path may read.
        bool packed()const noexcept{return packed_&&slab_.active();}
        const uint8_t*host()const noexcept{return slab_.active()?slab_.host():regular_.host();}
        const void*device()const noexcept{return slab_.active()?slab_.device():regular_.device();}
        size_t bytes()const noexcept{return slab_.active()?slab_.bytes():regular_.bytes();}
        Cache::Status release()noexcept{
            const auto a=regular_.release();if(a!=Cache::Status::Ok&&a!=Cache::Status::Empty)return a;
            const auto b=slab_.release();if(!slab_.active())packed_=false;return b==Cache::Status::Empty?a:b;
        }
        Cache::Stats stats()const{return slab_.empty()?regular_.stats():slab_.stats().cache;}
    };
    class Write {
        friend class NativeMappedFileCache;
        SlabCache::Write slab_;
        bool packed_=false;
    public:
        Write()=default;Write(const Write&)=delete;Write&operator=(const Write&)=delete;
        Write(Write&&)=default;Write&operator=(Write&&)=delete;
        bool active()const noexcept{return slab_.active();}
        bool empty()const noexcept{return slab_.empty();}
        bool packed()const noexcept{return packed_&&slab_.active();}
        uint8_t*host()const noexcept{return slab_.host();}
        size_t bytes()const noexcept{return slab_.bytes();}
        Cache::Status abort()noexcept{return slab_.abort();}
        Cache::Status publish(Lease*out)noexcept {
            if(!out||!out->empty())return Cache::Status::Invalid;
            const bool packed=packed_;const auto rc=slab_.publish(&out->slab_);
            if(rc==Cache::Status::Ok)out->packed_=packed;
            return rc;
        }
    };
    using Status=Cache::Status;
    static constexpr size_t GU=12533760,DN=6266880;
    enum class Layout { Block, Packed };
    struct ApiTiming {
        uint64_t allocate_attempts=0,allocate_null_returns=0,alias_rejections=0,device_bind_failures=0;
        uint64_t allocate_ns=0,allocate_max_ns=0,free_invocations=0,free_ns=0,free_max_ns=0;
    };
    struct Snapshot {
        Cache::Stats cache{};
        SlabCache::Stats slabs{};
        bool slab_policy=false,packed=false;
        SlabCache::Sizes slot_bytes=SlabCache::block_sizes;
        uint64_t layout_refusals=0;   //block_mxfp4 calls a packed cache turned away
        ApiTiming api{};
        size_t external_metadata=0,logical_payload=0,mapped_policy_charge=0;
        uint64_t allocations=0,public_free_invocations=0;
    };
private:
    struct Record {
        void* base=nullptr;
        uint8_t* host=nullptr;
        const uint8_t* device=nullptr;
        size_t payload=0,usable=0,charge=0,next=SIZE_MAX;
    };
    //The allocator context is stable until clear succeeds; it outlives every Lease.
    mutable std::mutex mutex_;
    std::unique_ptr<Record[]> records_;
    size_t count_=0,first_free_=SIZE_MAX,metadata_=0,logical_=0,charged_=0;
    uint64_t allocations_=0,frees_=0;
    ApiTiming api_{}; //guarded by mutex_; no timing on borrow/hit/release
    bool test_reject_alias_=false; //only friend HOST fixture may inject; default false
    Cache cache_;
    SlabCache slab_cache_;
    bool slab_policy_=false;
    Layout layout_=Layout::Block;
    SlabCache::Sizes slab_sizes_=SlabCache::block_sizes; //read by quoted_slab; set before slab_cache_ is configured
    std::atomic<uint64_t> layout_refusals_{0};
    int allocator_device_=0;
    static constexpr size_t guard=64,MiB=1u<<20;
    static size_t quoted(void*,size_t bytes) noexcept {
        if(bytes!=GU&&bytes!=DN)return 0;
        const size_t requested=(bytes+63)/64*64+3*guard;
        const size_t pages=(requested+4095)/4096*4096;
        //Conservative no-SVM retained bound (not measured RSS or observed unpin).
        return (pages+2*MiB-1)/(2*MiB)*(2*MiB)+2*MiB;
    }
    using Clock=std::chrono::steady_clock;
    static uint64_t elapsed_ns(Clock::time_point start) noexcept {
        return uint64_t(std::max<int64_t>(0,std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count()));
    }
    static void free_api_locked(NativeMappedFileCache& self,void* base) noexcept {
        const auto start=Clock::now();rad_dev_free(base,RAD_MEM_HOST_MAPPED);
        const uint64_t ns=elapsed_ns(start);
        ++self.frees_;++self.api_.free_invocations;self.api_.free_ns+=ns;
        self.api_.free_max_ns=std::max(self.api_.free_max_ns,ns);
    }
    static bool allocate(void* context,size_t bytes,size_t cap,Cache::Allocation* out) noexcept {
        auto& self=*static_cast<NativeMappedFileCache*>(context);
        const size_t charge=quoted(context,bytes);
        if(!out||!charge||charge>cap)return false;
        std::lock_guard<std::mutex> lock(self.mutex_);
        if(self.first_free_==SIZE_MAX)return false;
        const size_t index=self.first_free_;
        Record& r=self.records_[index];
        const size_t usable=(bytes+63)/64*64+guard,requested=usable+2*guard;
        const auto start=Clock::now();
        void* base=rad_dev_alloc(int64_t(requested),RAD_MEM_HOST_MAPPED);
        const uint64_t ns=elapsed_ns(start);
        ++self.api_.allocate_attempts;self.api_.allocate_ns+=ns;
        self.api_.allocate_max_ns=std::max(self.api_.allocate_max_ns,ns);
        if(!base){++self.api_.allocate_null_returns;return false;}
        ++self.allocations_; //nonnull public allocation, including a later rejected alias
        auto* h=static_cast<uint8_t*>(rad_dev_host_ptr(base));
        auto* d=static_cast<uint8_t*>(rad_dev_device_ptr(base));
        if(self.test_reject_alias_||!h||!d||reinterpret_cast<uintptr_t>(h)%64||reinterpret_cast<uintptr_t>(d)%64){
            ++self.api_.alias_rejections;free_api_locked(self,base);return false;
        }
        //Payload stays unpublished until the unchanged helper overwrites every byte.
        //Initialize only the prefix guard, zero reader tail/pad and final guard.
        std::memset(h,0xbd,guard);
        std::memset(h+guard+bytes,0,usable-bytes);
        std::memset(h+guard+usable,0xbd,guard);
        self.first_free_=r.next;r={base,h+guard,d+guard,bytes,usable,charge,SIZE_MAX};
        self.logical_+=bytes;self.charged_+=charge;
        *out={&r,r.host,r.device,usable,charge};return true;
    }
    static bool free(void* context,const Cache::Allocation& allocation) noexcept {
        auto& self=*static_cast<NativeMappedFileCache*>(context);
        std::lock_guard<std::mutex> lock(self.mutex_);
        const uintptr_t begin=reinterpret_cast<uintptr_t>(self.records_.get());
        const uintptr_t token=reinterpret_cast<uintptr_t>(allocation.token);
        if(!begin||token<begin||token-begin>=self.count_*sizeof(Record)||
           (token-begin)%sizeof(Record))return false;
        const size_t index=(token-begin)/sizeof(Record);
        Record& r=self.records_[index];
        if(!r.base||allocation.host!=r.host||allocation.device!=r.device||
           allocation.bytes!=r.usable||allocation.charged_bytes!=r.charge)return false;
        for(size_t k=0;k<guard;++k)
            if(r.host[-ptrdiff_t(guard)+ptrdiff_t(k)]!=0xbd||r.host[r.usable+k]!=0xbd)return false;
        for(size_t k=r.payload;k<r.usable;++k)if(r.host[k]!=0)return false;
        //This is an API invocation; the public free API does not return an unpin status.
        free_api_locked(self,r.base);
        self.logical_-=r.payload;self.charged_-=r.charge;
        r={};r.next=self.first_free_;self.first_free_=index;return true;
    }
    static size_t quoted_slab(void* context,size_t bytes) noexcept {
        const auto& sizes=static_cast<NativeMappedFileCache*>(context)->slab_sizes_;
        const size_t pair=sizes[0]+sizes[1]+6*guard;
        if(!bytes||bytes%pair||bytes>32*pair)return 0;
        const size_t pages=(bytes+4095)/4096*4096;
        return (pages+2*MiB-1)/(2*MiB)*(2*MiB)+2*MiB;
    }
    static bool allocate_slab(void* context,size_t bytes,size_t cap,Cache::Allocation* out) noexcept {
        auto& self=*static_cast<NativeMappedFileCache*>(context);
        const size_t charge=quoted_slab(context,bytes);
        if(!out||!charge||charge>cap)return false;
        //New reader threads do not inherit Engine's device TLS. Use the actual device,
        //not the EP rank index. HOST's device0 path uses the same public contract.
        if(rad_dev_set(self.allocator_device_)<0){
            std::lock_guard<std::mutex> lock(self.mutex_);++self.api_.device_bind_failures;return false;
        }
        std::lock_guard<std::mutex> lock(self.mutex_);
        if(self.first_free_==SIZE_MAX)return false;
        const size_t index=self.first_free_;Record& r=self.records_[index];
        const auto start=Clock::now();void* base=rad_dev_alloc(int64_t(bytes),RAD_MEM_HOST_MAPPED);
        const uint64_t ns=elapsed_ns(start);++self.api_.allocate_attempts;self.api_.allocate_ns+=ns;
        self.api_.allocate_max_ns=std::max(self.api_.allocate_max_ns,ns);
        if(!base){++self.api_.allocate_null_returns;return false;}
        ++self.allocations_;
        auto* h=static_cast<uint8_t*>(rad_dev_host_ptr(base));
        auto* d=static_cast<uint8_t*>(rad_dev_device_ptr(base));
        if(self.test_reject_alias_||!h||!d||uintptr_t(h)%64||uintptr_t(d)%64){
            ++self.api_.alias_rejections;
            if(self.test_reject_alias_)h=nullptr;
        }
        //Adopt EVERY returned owner, even malformed aliases. The slab helper poisons
        //that result and retires it terminally; no false zero-byte cleanup refund.
        self.first_free_=r.next;r={base,h,d,bytes,bytes,charge,SIZE_MAX};
        self.logical_+=bytes;self.charged_+=charge;
        *out={&r,h,d,bytes,charge};return true;
    }
    static bool free_slab(void* context,const Cache::Allocation& allocation) noexcept {
        auto& self=*static_cast<NativeMappedFileCache*>(context);
        if(rad_dev_set(self.allocator_device_)<0){
            std::lock_guard<std::mutex> lock(self.mutex_);++self.api_.device_bind_failures;return false;
        }
        std::lock_guard<std::mutex> lock(self.mutex_);
        const uintptr_t begin=uintptr_t(self.records_.get()),token=uintptr_t(allocation.token);
        if(!begin||token<begin||token-begin>=self.count_*sizeof(Record)||(token-begin)%sizeof(Record))return false;
        const size_t index=(token-begin)/sizeof(Record);Record& r=self.records_[index];
        if(!r.base||allocation.host!=r.host||allocation.device!=r.device||allocation.bytes!=r.usable||allocation.charged_bytes!=r.charge)return false;
        //SlabCache validates every guard before any cleanup. This consumes a public
        //owner handle only; no successful physical-unpin claim/refund/regrowth.
        free_api_locked(self,r.base);self.logical_-=r.payload;self.charged_-=r.charge;
        r={};r.next=self.first_free_;self.first_free_=index;return true;
    }

public:
    NativeMappedFileCache()=default;
    NativeMappedFileCache(const NativeMappedFileCache&)=delete;
    ~NativeMappedFileCache(){
        //An unknown consumer/free state must not destroy the allocator context or aliases.
        if(clear()!=RAD_OK){RAD_ERR("mapped cache: unretired/invalid owner at teardown");std::_Exit(88);}
    }
    int clear() noexcept {
        const auto rc=cache_.clear();
        if(rc!=Status::Ok)return RAD_E_STATE;
        if(slab_cache_.clear()!=Status::Ok)return RAD_E_STATE;
        std::lock_guard<std::mutex> lock(mutex_);
        if(logical_||charged_)return RAD_E_STATE;
        records_.reset();count_=metadata_=0;first_free_=SIZE_MAX;return RAD_OK;
    }
    //layout Packed: slabs only, with slots of packed_sizes (gate_up, down) bytes.
    int configure(size_t budget,size_t count,size_t window_metadata,int actual_device=0,
                  Layout layout=Layout::Block,SlabCache::Sizes packed_sizes=SlabCache::block_sizes) noexcept {
        const char* env=std::getenv("RADIANCE_EXPERIMENTAL_NATIVE_CACHE_MAPPED_SLABS");
        const bool want_slabs=env&&std::strcmp(env,"1")==0;
        if(actual_device<0)return RAD_E_INVAL;
        if(layout==Layout::Packed&&(!want_slabs||!SlabCache::sizes_valid(packed_sizes)))return RAD_E_INVAL;
        RAD_TRY(clear());
        //A previous terminal void-free cannot establish reclaimed physical capacity.
        //Start a new owning Engine/process rather than spend that backing again.
        if(slab_cache_.stats().cache.payload_charge)return RAD_E_STATE;
        if(!count||window_metadata>SIZE_MAX-sizeof(*this)-128||
           count>(SIZE_MAX-sizeof(*this)-128-window_metadata)/sizeof(Record))return RAD_E_INVAL;
        const size_t external=sizeof(*this)-(want_slabs?sizeof(SlabCache):sizeof(Cache))+count*sizeof(Record)+128+window_metadata;
        if(external>=budget||(!want_slabs&&Cache::metadata_charge(count)>budget-external))return RAD_E_NOMEM;
        auto records=std::unique_ptr<Record[]>(new(std::nothrow) Record[count]);
        if(!records)return RAD_E_NOMEM;
        for(size_t i=0;i<count;++i)records[i].next=i+1<count?i+1:SIZE_MAX;
        records_=std::move(records);count_=count;first_free_=0;metadata_=external;
        allocations_=frees_=0;api_={};allocator_device_=actual_device;
        slab_sizes_=layout==Layout::Packed?packed_sizes:SlabCache::block_sizes;layout_=Layout::Block;layout_refusals_=0;
        const auto rc=want_slabs?slab_cache_.configure(1,budget-external,count,{this,quoted_slab,allocate_slab,free_slab},slab_sizes_):
                                  cache_.configure(1,budget-external,count,{this,quoted,allocate,free});
        if(rc!=Status::Ok){records_.reset();count_=metadata_=0;first_free_=SIZE_MAX;slab_sizes_=SlabCache::block_sizes;return RAD_E_NOMEM;}
        slab_policy_=want_slabs;layout_=layout;return RAD_OK;
    }
    bool packed() const noexcept { return layout_==Layout::Packed; }
    //The slots' payload bytes: gate_up's and down's (block_mxfp4's GU and DN unless packed).
    SlabCache::Sizes slot_bytes() const noexcept { return slab_sizes_; }
    // Resident without a lease, an LRU touch or a statistic; slab policy only (false otherwise).
    bool present(size_t key) noexcept { return slab_policy_&&slab_cache_.present(1,key); }
    // present() for n keys under one lock of the slab cache (out[i] 0/1); slab policy only.
    void present_many(const size_t* keys,size_t n,uint8_t* out) noexcept {
        if(!slab_policy_){for(size_t i=0;i<n;++i)out[i]=0;return;}
        slab_cache_.present_many(1,keys,n,out);
    }
    // Slab policy only: pins the budget's slabs now (MappedExpertSlabCache::pregrow). A refused
    // allocation stops the growth and is not an error; the cache grows on demand from there.
    int pregrow(size_t* slabs) noexcept {
        if(slabs)*slabs=0;
        if(!slab_policy_)return RAD_OK;
        const auto rc=slab_cache_.pregrow(1,slabs);
        return rc==Status::Ok||rc==Status::Allocation?RAD_OK:RAD_E_STATE;
    }
    Status borrow(size_t key,size_t bytes,Lease* lease) noexcept {
        if(!lease||!lease->empty())return Status::Invalid;
        if(packed()){layout_refusals_.fetch_add(1,std::memory_order_relaxed);return Status::Miss;}
        return slab_policy_?slab_cache_.borrow(1,key,bytes,&lease->slab_):cache_.borrow(1,key,bytes,&lease->regular_);
    }
    //The packed layout's own calls, for the host expert path: klass 0 (gate_up) or 1 (down).
    Status borrow_packed(size_t key,int klass,Lease* lease) noexcept {
        if(!lease||!lease->empty()||klass<0||klass>1)return Status::Invalid;
        if(!packed())return Status::Layout;
        const auto rc=slab_cache_.borrow(1,key,slab_sizes_[size_t(klass)],&lease->slab_);
        if(rc==Status::Hit)lease->packed_=true;
        return rc;
    }
    Status reserve_packed(size_t key,int klass,Write* out) noexcept {
        if(!out||!out->empty()||klass<0||klass>1)return Status::Invalid;
        if(!packed())return Status::Layout;
        const auto rc=slab_cache_.reserve(1,key,slab_sizes_[size_t(klass)],&out->slab_);
        if(rc==Status::Ok)out->packed_=true;
        return rc;
    }
    //Configuration is quiescent; legacy per-entry ReadOnly remains the existing bypass.
    bool supports_readonly_borrow()const noexcept{return slab_policy_;}
    Status borrow_readonly(size_t key,size_t bytes,Lease*lease)noexcept{
        if(!lease||!lease->empty())return Status::Invalid;
        if(packed()){layout_refusals_.fetch_add(1,std::memory_order_relaxed);return Status::Miss;}
        return slab_policy_?slab_cache_.borrow_no_touch(1,key,bytes,&lease->slab_):Status::Miss;
    }
    Status reserve_fill(size_t key,size_t bytes,Write*out)noexcept {
        if(!out||!out->empty())return Status::Invalid;
        if(packed()){layout_refusals_.fetch_add(1,std::memory_order_relaxed);return Status::Bypass;}
        return slab_policy_?slab_cache_.reserve(1,key,bytes,&out->slab_):Status::Bypass;
    }
    /* Second chances for profile-hot keys (SlabCache::victim_for); retained slabs only. */
    Status set_second_chances(int n)noexcept{return slab_policy_?slab_cache_.set_second_chances(n):Status::Bypass;}
    Status protect(size_t key)noexcept{return slab_policy_?slab_cache_.protect(1,key):Status::Bypass;}
    Status insert(size_t key,const void* source,size_t bytes) noexcept {
        if(packed()){layout_refusals_.fetch_add(1,std::memory_order_relaxed);return Status::Bypass;}
        return slab_policy_?slab_cache_.insert(1,key,source,bytes):cache_.insert(1,key,source,bytes);
    }
    Snapshot stats() const noexcept {
        Snapshot result;result.slab_policy=slab_policy_;result.packed=packed();result.slot_bytes=slab_sizes_;
        result.layout_refusals=layout_refusals_.load(std::memory_order_relaxed);
        if(slab_policy_){result.slabs=slab_cache_.stats();result.cache=result.slabs.cache;}
        else result.cache=cache_.stats();
        std::lock_guard<std::mutex> lock(mutex_);
        result.external_metadata=metadata_;result.logical_payload=slab_policy_?
            result.slabs.classes[0].capacity_logical_bytes+result.slabs.classes[1].capacity_logical_bytes:logical_;
        result.mapped_policy_charge=charged_;
        result.allocations=allocations_;result.public_free_invocations=frees_;result.api=api_;return result;
    }
};
} //namespace rad
