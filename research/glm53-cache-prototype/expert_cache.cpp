#include "expert_cache.h"
#include <algorithm>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace glm53::cache {
const char* to_string(Error error) noexcept {
    switch(error){
    case Error::ok:return "ok";case Error::invalid:return "invalid";case Error::too_large:return "too_large";
    case Error::budget:return "budget";case Error::metadata_limit:return "metadata_limit";
    case Error::identity_mismatch:return "identity_mismatch";case Error::io:return "io";
    case Error::checksum:return "checksum";case Error::cancelled:return "cancelled";
    case Error::reentrant:return "reentrant";case Error::no_memory:return "no_memory";case Error::closed:return "closed";
    }return "unknown";
}
uint64_t fnv1a64(std::span<const std::byte> bytes) noexcept {
    uint64_t hash=14695981039346656037ull;
    for(const auto byte:bytes){hash^=std::to_integer<uint8_t>(byte);hash*=1099511628211ull;}
    return hash;
}
namespace detail {
void increment(uint64_t& value,uint64_t amount=1) noexcept {
    value+=std::min(amount,std::numeric_limits<uint64_t>::max()-value);
}
enum class Tier {cold,hot};
struct Ledger {
    Limits limits;mutable std::mutex mutex;Usage used;
    explicit Ledger(Limits value):limits(value){}
    Error reserve(uint64_t bytes,size_t records){
        std::lock_guard lock(mutex);
        if(bytes>limits.cold_bytes-used.cold_bytes)return Error::budget;
        if(records>limits.max_records-used.live_records)return Error::metadata_limit;
        used.cold_bytes+=bytes;used.live_records+=records;
        used.peak_cold_bytes=std::max(used.peak_cold_bytes,used.cold_bytes);return Error::ok;
    }
    void release(Tier tier,uint64_t bytes,size_t records) noexcept {
        std::lock_guard lock(mutex);auto& count=tier==Tier::hot?used.hot_bytes:used.cold_bytes;
        count-=bytes;used.live_records-=records;
    }
    bool promote(uint64_t bytes){
        std::lock_guard lock(mutex);
        if(bytes>limits.hot_bytes-used.hot_bytes)return false;
        used.cold_bytes-=bytes;used.hot_bytes+=bytes;
        used.peak_hot_bytes=std::max(used.peak_hot_bytes,used.hot_bytes);return true;
    }
    Usage usage() const noexcept {std::lock_guard lock(mutex);return used;}
};
struct Reservation {
    std::shared_ptr<Ledger> ledger;uint64_t bytes=0;size_t records=0;
    ~Reservation(){if(ledger&&(bytes||records))ledger->release(Tier::cold,bytes,records);}
    void consume(uint64_t count) noexcept {bytes-=count;--records;}
};
struct Record {
    const RecordRef ref;std::vector<std::byte> data;std::shared_ptr<Ledger> ledger;
    Tier tier=Tier::cold;bool charged=false;
    Record(const RecordRef& source,std::shared_ptr<Ledger> account)
        :ref(source),data(size_t(source.length)),ledger(std::move(account)){}
    ~Record(){
        // Free actual payload BEFORE returning its byte/record charge. The ledger outlives
        // Cache when a user retains a lease; it owns no record, so there is no cycle.
        std::vector<std::byte>().swap(data);
        if(charged)ledger->release(tier,ref.length,1);
    }
};
struct KeyHash {
    size_t operator()(const Key& key) const noexcept {
        uint64_t hash=14695981039346656037ull;
        auto text=[&](const std::string& value){for(unsigned char byte:value){hash^=byte;hash*=1099511628211ull;}hash^=0xff;hash*=1099511628211ull;};
        text(key.source_id);text(key.model_id);text(key.layout_id);text(key.record_class);
        for(uint64_t value:{uint64_t(key.tp),uint64_t(key.rank),uint64_t(uint32_t(key.layer)),uint64_t(uint32_t(key.expert))}){
            for(unsigned i=0;i<8;++i){hash^=uint8_t(value>>(8*i));hash*=1099511628211ull;}
        }return size_t(hash);
    }
};
struct Entry {
    RecordRef ref;std::weak_ptr<Record> weak;std::shared_ptr<Record> hot;
    Entry* previous=nullptr;Entry* next=nullptr;bool pending=true;
    explicit Entry(const RecordRef& value):ref(value){}
};
struct State {
    Limits limits;Provider& provider;std::shared_ptr<Ledger> ledger;
    mutable std::mutex metadata;std::mutex transaction;
    std::unordered_map<Key,std::unique_ptr<Entry>,KeyHash> entries;
    std::unordered_map<Key,RecordRef,KeyHash> catalog;
    Entry* newest=nullptr;Entry* oldest=nullptr;Stats counters;
    State(Limits value,Provider& source):limits(value),provider(source),ledger(std::make_shared<Ledger>(value)){}
    void unlink(Entry& entry) noexcept {
        if(entry.previous)entry.previous->next=entry.next;else newest=entry.next;
        if(entry.next)entry.next->previous=entry.previous;else oldest=entry.previous;
        entry.previous=entry.next=nullptr;
    }
    void front(Entry& entry) noexcept {
        entry.previous=nullptr;entry.next=newest;if(newest)newest->previous=&entry;else oldest=&entry;newest=&entry;
    }
    void touch(Entry& entry) noexcept {if(newest!=&entry){unlink(entry);front(entry);}}
    void prune(){
        for(auto it=entries.begin();it!=entries.end();)if(!it->second->pending&&!it->second->hot&&it->second->weak.expired())it=entries.erase(it);else ++it;
    }
    bool evict_one(){
        Entry* victim=oldest;
        while(victim&&victim->hot.use_count()!=1)victim=victim->previous;
        if(!victim)return false;
        const auto found=entries.find(victim->ref.key);unlink(*victim);
        // No user can promote a weak_ptr: the only weak index is private, and all acquires
        // share transaction. use_count()==1 therefore proves there is no outstanding lease.
        entries.erase(found);increment(counters.evictions);return true;
    }
    void admit(Entry& entry,const std::shared_ptr<Record>& record){
        if(entry.hot){touch(entry);return;}
        if(record->ref.length>limits.hot_bytes){increment(counters.admission_bypasses);return;}
        while(!ledger->promote(record->ref.length))if(!evict_one()){increment(counters.admission_bypasses);return;}
        record->tier=Tier::hot;entry.hot=record;front(entry);increment(counters.admissions);
    }
};
// A provider callback may inspect stats, but cannot recursively wait on any cache mutation.
// Cross-thread callback-induced cycles remain forbidden by the Provider contract.
struct Active {Active* previous;Active();~Active();};
thread_local Active* active=nullptr;
Active::Active():previous(active){active=this;}
Active::~Active(){active=previous;}
struct Pending {
    State& state;std::vector<Entry*> entries;std::vector<const Key*> catalog_keys;bool armed=true;
    ~Pending(){if(armed){std::lock_guard lock(state.metadata);
        for(auto* entry:entries){const auto found=state.entries.find(entry->ref.key);state.entries.erase(found);}
        for(const auto* key:catalog_keys){const auto found=state.catalog.find(*key);state.catalog.erase(found);}
        state.counters.inflight_records=0;
    }}
};
Error validate(const Limits& limits,const RecordRef& ref){
    const auto& key=ref.key;
    for(const auto* value:{&key.source_id,&key.model_id,&key.layout_id,&key.record_class})
        if(value->empty()||value->size()>limits.max_identity_bytes)return Error::invalid;
    if(!key.tp||key.tp>256||key.rank>=key.tp||key.layer<0||key.expert<0)return Error::invalid;
    if(!ref.length||ref.length>limits.max_record_bytes||ref.length>SIZE_MAX)return Error::too_large;
    if(ref.offset>uint64_t(INT64_MAX)||ref.length>uint64_t(INT64_MAX)-ref.offset)return Error::invalid;
    return Error::ok;
}
} // namespace detail

Usage Accounting::usage() const noexcept {return ledger_?ledger_->usage():Usage{};}
const RecordRef& Lease::ref() const {if(!record_)throw std::logic_error("empty expert record lease");return record_->ref;}
std::span<const std::byte> Lease::bytes() const noexcept {return record_?std::span<const std::byte>(record_->data):std::span<const std::byte>{};}
Cache::Cache(Limits limits,Provider& provider){
    if(!limits.max_records||limits.max_records>65536||!limits.max_batch||limits.max_batch>4096||limits.max_batch>limits.max_records||
       !limits.max_identity_bytes||limits.max_identity_bytes>1024||!limits.max_catalog_records||limits.max_catalog_records>65536||
       !limits.max_record_bytes||limits.max_record_bytes>SIZE_MAX||
       limits.hot_bytes>UINT64_MAX-limits.cold_bytes)throw std::invalid_argument("invalid bounded cache limits");
    state_=std::make_unique<detail::State>(limits,provider);
}
Cache::~Cache()=default;
Accounting Cache::accounting() const noexcept {return Accounting(state_->ledger);}
Stats Cache::stats() const noexcept {
    std::lock_guard lock(state_->metadata);Stats result=state_->counters;const auto used=state_->ledger->usage();
    result.hot_bytes=used.hot_bytes;result.cold_bytes=used.cold_bytes;result.peak_hot_bytes=used.peak_hot_bytes;result.peak_cold_bytes=used.peak_cold_bytes;result.live_records=used.live_records;
    result.index_entries=state_->entries.size();result.catalog_entries=state_->catalog.size();result.hot_records=0;for(const auto& [key,entry]:state_->entries){(void)key;if(entry->hot)++result.hot_records;}return result;
}
Error Cache::trim() noexcept {
    if(detail::active)return Error::reentrant;
    try{std::lock_guard transaction(state_->transaction);std::lock_guard metadata(state_->metadata);while(state_->evict_one()){}state_->prune();return Error::ok;}
    catch(...){return Error::io;}
}
Error Cache::acquire(std::span<const RecordRef> request,Batch* out,std::stop_token stop) noexcept {
    if(!out)return Error::invalid;
    if(detail::active)return Error::reentrant;
    if(stop.stop_requested())return Error::cancelled;
    auto& state=*state_;
    auto fail=[&](Error error){std::lock_guard lock(state.metadata);detail::increment(state.counters.failed_batches);if(error==Error::budget||error==Error::metadata_limit)detail::increment(state.counters.budget_refusals);return error;};
    try{
        detail::Active active;std::lock_guard transaction(state.transaction);
        if(stop.stop_requested())return fail(Error::cancelled);
        if(request.size()>state.limits.max_batch)return fail(Error::metadata_limit);
        std::vector<const RecordRef*> unique;unique.reserve(request.size());
        for(const auto& ref:request){
            const auto status=detail::validate(state.limits,ref);if(status!=Error::ok)return fail(status);
            const auto existing=std::find_if(unique.begin(),unique.end(),[&](const RecordRef* prior){return prior->key==ref.key;});
            if(existing!=unique.end()){if(**existing!=ref)return fail(Error::identity_mismatch);}
            else unique.push_back(&ref);
        }
        Batch result;result.leases.resize(unique.size());
        std::vector<std::shared_ptr<detail::Record>> records(unique.size());
        std::vector<detail::Entry*> selected(unique.size());
        detail::Pending pending{state,{},{}};pending.entries.reserve(unique.size());pending.catalog_keys.reserve(unique.size());
        detail::Reservation reserved;uint64_t missing_bytes=0;size_t misses=0;
        Error prepared=Error::ok;
        {
            std::lock_guard lock(state.metadata);state.prune();
            detail::increment(state.counters.requests);detail::increment(state.counters.unique_demands,unique.size());detail::increment(state.counters.duplicate_demands,request.size()-unique.size());
            size_t new_catalog=0;
            for(size_t i=0;i<unique.size();++i){
                const auto known=state.catalog.find(unique[i]->key);
                if(known!=state.catalog.end()&&known->second!=*unique[i]){prepared=Error::identity_mismatch;break;}
                if(known==state.catalog.end())++new_catalog;
                const auto found=state.entries.find(unique[i]->key);
                if(found!=state.entries.end()){
                    auto& entry=*found->second;
                    if(entry.ref!=*unique[i]){prepared=Error::identity_mismatch;break;}
                    records[i]=entry.weak.lock();
                    if(records[i]){
                        selected[i]=&entry;
                        detail::increment(entry.hot?state.counters.hot_hits:state.counters.cold_hits);
                    }else{
                        // A final external cold lease can disappear after prune inspected
                        // this entry. It is an ordinary fresh miss, not an I/O failure.
                        // Preserve the separately bounded immutable-ref catalog.
                        if(entry.hot){prepared=Error::io;break;}
                        state.entries.erase(found);
                    }
                }
                if(!records[i]){
                    if(unique[i]->length>UINT64_MAX-missing_bytes){prepared=Error::too_large;break;}
                    missing_bytes+=unique[i]->length;++misses;
                }
            }
            if(prepared==Error::ok&&new_catalog>state.limits.max_catalog_records-state.catalog.size())prepared=Error::metadata_limit;
            if(prepared==Error::ok&&misses>state.limits.max_records-state.entries.size())prepared=Error::metadata_limit;
            if(prepared==Error::ok)prepared=state.ledger->reserve(missing_bytes,misses);
            if(prepared==Error::ok){
                reserved.ledger=state.ledger;reserved.bytes=missing_bytes;reserved.records=misses;
                state.entries.reserve(state.entries.size()+misses);
                state.catalog.reserve(state.catalog.size()+new_catalog);
                for(size_t i=0;i<unique.size();++i){
                    if(state.catalog.find(unique[i]->key)==state.catalog.end()){
                        const auto inserted=state.catalog.emplace(unique[i]->key,*unique[i]);pending.catalog_keys.push_back(&inserted.first->first);
                    }
                    if(!records[i]){
                        auto entry=std::make_unique<detail::Entry>(*unique[i]);auto* pointer=entry.get();
                        state.entries.emplace(unique[i]->key,std::move(entry));pending.entries.push_back(pointer);selected[i]=pointer;
                    }
                }
                state.counters.inflight_records=misses;detail::increment(state.counters.misses,misses);
            }
        }
        if(prepared!=Error::ok)return fail(prepared);
        for(size_t i=0;i<unique.size();++i)if(!records[i]){
            if(stop.stop_requested())return fail(Error::cancelled);
            auto record=std::shared_ptr<detail::Record>(new detail::Record(*unique[i],state.ledger));
            record->charged=true;reserved.consume(record->ref.length);records[i]=record;
            {std::lock_guard lock(state.metadata);detail::increment(state.counters.provider_reads);}
            ReadResult loaded;
            try{loaded=state.provider.read(record->ref,std::span<std::byte>(record->data),stop);}
            catch(const std::bad_alloc&){return fail(Error::no_memory);}catch(...){return fail(Error::io);}
            {
                std::lock_guard lock(state.metadata);detail::increment(state.counters.bytes_read,loaded.bytes_read);
            }
            if(stop.stop_requested())return fail(Error::cancelled);
            if(loaded.error!=Error::ok)return fail(loaded.error);
            if(loaded.bytes_read!=record->data.size())return fail(Error::io);
            if(fnv1a64(record->data)!=record->ref.checksum)return fail(Error::checksum);
        }
        if(stop.stop_requested())return fail(Error::cancelled);
        // All data, checksums, map nodes and output-vector capacity are prepared before commit.
        // This section performs no payload allocation and publishes no partial result.
        for(size_t i=0;i<records.size();++i)result.leases[i]=Lease(records[i]);
        {
            std::lock_guard lock(state.metadata);
            for(size_t i=0;i<selected.size();++i){selected[i]->weak=records[i];selected[i]->pending=false;state.admit(*selected[i],records[i]);}
            state.counters.inflight_records=0;pending.armed=false;
        }
        *out=std::move(result);return Error::ok;
    }catch(const std::bad_alloc&){return fail(Error::no_memory);}catch(...){return fail(Error::io);}
}
} // namespace glm53::cache
