#include "expert_cache.h"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>

using namespace glm53::cache;
namespace {
void require(bool value,const char* why){if(!value)throw std::runtime_error(why);}
void okay(Error error){if(error!=Error::ok)throw std::runtime_error(to_string(error));}
Limits limits(uint64_t hot,uint64_t cold,size_t records=32){
    Limits value;value.hot_bytes=hot;value.cold_bytes=cold;value.max_record_bytes=16u*1024u*1024u;
    value.max_records=records;value.max_batch=std::min(size_t(16),records);value.max_catalog_records=64;return value;
}
Key key(int expert,const char* type="main-mxfp4"){return {"pinned-source","GLM53-synthetic","tp4-rank0-layout-v1",type,4,0,3,expert};}
class MemoryProvider final:public Provider {
public:
    struct Item{RecordRef ref;std::vector<std::byte> bytes;};
    std::vector<Item> items;std::atomic<size_t> calls=0;
    size_t fail_at=0,short_at=0,corrupt_at=0,throw_at=0,bad_alloc_at=0;
    std::function<void(const RecordRef&,std::stop_token)> hook;
    RecordRef add(int expert,size_t bytes,const char* type="main-mxfp4"){
        Item item;item.ref.key=key(expert,type);item.ref.offset=items.size()*4096;item.ref.length=bytes;
        item.bytes.resize(bytes);for(size_t i=0;i<bytes;++i)item.bytes[i]=std::byte((unsigned(expert)*29u+unsigned(i)*7u)&255u);
        item.ref.checksum=fnv1a64(item.bytes);const auto ref=item.ref;items.push_back(std::move(item));return ref;
    }
    ReadResult read(const RecordRef& ref,std::span<std::byte> output,std::stop_token stop) override {
        const auto call=++calls;if(hook)hook(ref,stop);
        if(call==bad_alloc_at)throw std::bad_alloc();
        if(call==throw_at)throw std::runtime_error("injected provider exception");
        if(call==fail_at)return {Error::io,0};
        if(stop.stop_requested())return {Error::cancelled,0};
        const auto found=std::find_if(items.begin(),items.end(),[&](const Item& item){return item.ref==ref;});
        if(found==items.end()||output.size()!=found->bytes.size())return {Error::identity_mismatch,0};
        std::copy(found->bytes.begin(),found->bytes.end(),output.begin());
        if(call==corrupt_at)output.back()^=std::byte{1};
        return {Error::ok,call==short_at?output.size()-1:output.size()};
    }
};
void exact(const Lease& lease,const MemoryProvider& provider){
    require(bool(lease),"empty ready lease");const auto found=std::find_if(provider.items.begin(),provider.items.end(),[&](const auto& item){return item.ref==lease.ref();});
    require(found!=provider.items.end(),"lease identity absent from provider");
    require(std::equal(lease.bytes().begin(),lease.bytes().end(),found->bytes.begin(),found->bytes.end()),"lease bytes changed");
}
void duplicate_union_and_mixed_classes(){
    MemoryProvider provider;const auto a=provider.add(0,128),b=provider.add(1,256,"mtp-fp8");Cache cache(limits(384,512),provider);
    const RecordRef demand[]{a,a,b,a,b};Batch batch;okay(cache.acquire(demand,&batch));
    require(batch.leases.size()==2&&provider.calls==2,"duplicate routes fetched more than one record");
    exact(batch.leases[0],provider);exact(batch.leases[1],provider);
    require(batch.leases[0].ref()==a&&batch.leases[1].ref()==b,"demand order changed");
    const auto stats=cache.stats();require(stats.hot_bytes==384&&stats.cold_bytes==0&&stats.duplicate_demands==3,"mixed class byte accounting wrong");
    require(stats.peak_cold_bytes==384,"whole miss union was not cold-reserved together");
}
void whole_union_refusal_before_io(){
    MemoryProvider provider;const auto a=provider.add(0,128),b=provider.add(1,256);Cache cache(limits(0,256),provider);
    const RecordRef demand[]{a,b};Batch batch;require(cache.acquire(demand,&batch)==Error::budget,"oversized demand was not refused");
    require(provider.calls==0&&batch.leases.empty(),"refused union performed IO or published a partial batch");
    const auto used=cache.accounting().usage();require(!used.hot_bytes&&!used.cold_bytes&&!used.live_records,"failed reserve leaked charges");
    require(cache.stats().catalog_entries==0&&cache.stats().index_entries==0,"failed reserve published metadata");
}
void late_io_failure_is_atomic(){
    MemoryProvider provider;const auto old=provider.add(0,128),a=provider.add(1,128),b=provider.add(2,256);Cache cache(limits(128,1024),provider);
    Batch output;okay(cache.acquire(std::span(&old,1),&output));const auto* pointer=output.leases[0].bytes().data();
    provider.fail_at=3;size_t observations=0;
    provider.hook=[&](const RecordRef&,std::stop_token){require(cache.stats().cold_bytes==384,"miss reservation did not precede first read");++observations;};
    const RecordRef demand[]{a,b};require(cache.acquire(demand,&output)==Error::io,"late read failure accepted");
    require(observations==2&&output.leases.size()==1&&output.leases[0].bytes().data()==pointer&&output.leases[0].ref()==old,"failed transaction changed caller output");
    exact(output.leases[0],provider);const auto stats=cache.stats();
    require(stats.hot_bytes==128&&stats.cold_bytes==0&&stats.live_records==1&&stats.index_entries==1&&stats.catalog_entries==1,"late failure leaked ready records or charge");
    provider.hook={};provider.fail_at=0;Batch retry;okay(cache.acquire(demand,&retry));require(retry.leases.size()==2,"retry did not serve every demanded expert");
}
void short_checksum_and_exception_cleanup(){
    for(int mode=0;mode<4;++mode){MemoryProvider provider;const auto a=provider.add(0,128);Cache cache(limits(128,256),provider);
        if(mode==0)provider.short_at=1;else if(mode==1)provider.corrupt_at=1;else if(mode==2)provider.throw_at=1;else provider.bad_alloc_at=1;
        Batch batch;const auto result=cache.acquire(std::span(&a,1),&batch);
        require(result==(mode==1?Error::checksum:mode==3?Error::no_memory:Error::io),"short/corrupt/throwing read accepted");
        require(batch.leases.empty()&&cache.stats().live_records==0&&cache.stats().cold_bytes==0&&cache.stats().catalog_entries==0,"read failure leaked or published");
        provider.short_at=provider.corrupt_at=provider.throw_at=provider.bad_alloc_at=0;okay(cache.acquire(std::span(&a,1),&batch));exact(batch.leases[0],provider);
    }
}
void pinned_hot_bypasses_admission_but_serves_cold(){
    MemoryProvider provider;const auto a=provider.add(0,128),b=provider.add(1,128);Cache cache(limits(128,256),provider);
    Batch pinned,cold;okay(cache.acquire(std::span(&a,1),&pinned));const auto* ptr=pinned.leases[0].bytes().data();
    okay(cache.acquire(std::span(&b,1),&cold));auto stats=cache.stats();
    require(stats.hot_bytes==128&&stats.cold_bytes==128&&stats.admission_bypasses==1&&stats.evictions==0,"pinned hot record was evicted");
    require(pinned.leases[0].bytes().data()==ptr,"leased pointer was reused");exact(cold.leases[0],provider);
    Batch shared;okay(cache.acquire(std::span(&b,1),&shared));require(provider.calls==2&&shared.leases[0].bytes().data()==cold.leases[0].bytes().data(),"live cold demand was not deduplicated");
    cold.leases.clear();shared.leases.clear();require(cache.stats().cold_bytes==0,"cold bytes survived final demand lease");
    pinned.leases.clear();okay(cache.trim());require(cache.stats().hot_bytes==0,"unleased hot trim failed");
}
void lru_evicts_only_unleased_victim(){
    MemoryProvider provider;const auto a=provider.add(0,128),b=provider.add(1,128),c=provider.add(2,128);Cache cache(limits(256,512),provider);
    Batch temporary;const RecordRef first[]{a,b};okay(cache.acquire(first,&temporary));temporary.leases.clear();
    okay(cache.acquire(std::span(&a,1),&temporary));temporary.leases.clear(); // a is most recent.
    okay(cache.acquire(std::span(&c,1),&temporary));temporary.leases.clear();
    const auto count=provider.calls.load();okay(cache.acquire(std::span(&a,1),&temporary));require(provider.calls==count,"LRU evicted recent a");temporary.leases.clear();
    okay(cache.acquire(std::span(&b,1),&temporary));require(provider.calls==count+1,"LRU failed to evict old b");
    require(cache.stats().evictions==2,"unexpected LRU eviction count");
}
void leases_keep_ledger_alive_after_store_destroy(){
    MemoryProvider provider;const auto a=provider.add(0,128),b=provider.add(1,256,"mtp-fp8");Batch hot,cold;Accounting account;
    {Cache cache(limits(128,512),provider);account=cache.accounting();okay(cache.acquire(std::span(&a,1),&hot));okay(cache.acquire(std::span(&b,1),&cold));}
    auto usage=account.usage();require(usage.hot_bytes==128&&usage.cold_bytes==256&&usage.live_records==2,"store destruction refunded live leases early");
    exact(hot.leases[0],provider);exact(cold.leases[0],provider);hot.leases.clear();usage=account.usage();
    require(usage.hot_bytes==0&&usage.cold_bytes==256&&usage.live_records==1,"hot refund did not follow actual final owner");
    cold.leases.clear();usage=account.usage();require(!usage.hot_bytes&&!usage.cold_bytes&&!usage.live_records,"final payload release leaked charge");
}
void immutable_identity_survives_eviction(){
    MemoryProvider provider;const auto a=provider.add(0,128);Cache cache(limits(128,256),provider);Batch batch;
    okay(cache.acquire(std::span(&a,1),&batch));batch.leases.clear();okay(cache.trim());
    auto changed=a;++changed.checksum;const auto calls=provider.calls.load();
    require(cache.acquire(std::span(&changed,1),&batch)==Error::identity_mismatch&&provider.calls==calls,"evicted key silently changed immutable identity");
    okay(cache.acquire(std::span(&a,1),&batch));
}
void malformed_union_and_metadata_caps(){
    MemoryProvider provider;auto a=provider.add(0,128),b=provider.add(1,128);Cache cache(limits(0,256),provider);Batch batch;
    auto wrong=a;++wrong.offset;const RecordRef conflict[]{a,wrong};require(cache.acquire(conflict,&batch)==Error::identity_mismatch,"conflicting duplicate key accepted");
    wrong=a;wrong.offset=uint64_t(INT64_MAX)-1;require(cache.acquire(std::span(&wrong,1),&batch)==Error::invalid,"offset overflow accepted");
    wrong=a;wrong.key.rank=wrong.key.tp;require(cache.acquire(std::span(&wrong,1),&batch)==Error::invalid,"invalid TP rank accepted");
    wrong=a;wrong.key.layout_id.assign(257,'x');require(cache.acquire(std::span(&wrong,1),&batch)==Error::invalid,"unbounded identity accepted");
    require(provider.calls==0,"malformed demand reached IO");
    auto bounded=limits(0,256);bounded.max_catalog_records=1;Cache catalog(bounded,provider);
    okay(catalog.acquire(std::span(&a,1),&batch));batch.leases.clear();
    require(catalog.acquire(std::span(&b,1),&batch)==Error::metadata_limit,"immutable history exceeded metadata cap");
    require(catalog.acquire({},&batch)==Error::ok&&batch.leases.empty(),"empty demand union rejected");
}
void cancelled_fill_rolls_back_without_publishing(){
    MemoryProvider provider;const auto a=provider.add(0,128),b=provider.add(1,128);Cache cache(limits(128,512),provider);Batch batch;
    std::stop_source cancelled;cancelled.request_stop();require(cache.acquire(std::span(&a,1),&batch,cancelled.get_token())==Error::cancelled&&provider.calls==0,"pre-cancel did IO");
    std::stop_source during;provider.hook=[&](const RecordRef&,std::stop_token){during.request_stop();};
    const RecordRef demand[]{a,b};require(cache.acquire(demand,&batch,during.get_token())==Error::cancelled,"in-fill cancellation published a batch");
    require(provider.calls==1&&batch.leases.empty()&&cache.stats().live_records==0&&cache.stats().catalog_entries==0,"cancelled fill leaked memory/metadata");
    provider.hook={};okay(cache.acquire(demand,&batch));require(batch.leases.size()==2,"cancelled cache failed to recover");
}
void provider_can_query_stats_but_reentry_is_refused(){
    MemoryProvider provider;const auto a=provider.add(0,128);Cache cache(limits(128,256),provider);Batch batch;
    provider.hook=[&](const RecordRef&,std::stop_token){
        require(cache.stats().cold_bytes==128,"stats unavailable outside metadata lock");Batch nested;
        require(cache.acquire(std::span(&a,1),&nested)==Error::reentrant,"recursive acquire did not refuse");
        require(cache.trim()==Error::reentrant,"recursive trim did not refuse");
    };
    okay(cache.acquire(std::span(&a,1),&batch));exact(batch.leases[0],provider);
}
void concurrent_same_key_uses_one_fill(){
    MemoryProvider provider;const auto a=provider.add(0,128);Cache cache(limits(0,256),provider);
    std::mutex mutex;std::condition_variable changed;bool entered=false,released=false,first_done=false;std::atomic<bool> second_entered=false;
    provider.hook=[&](const RecordRef&,std::stop_token){std::unique_lock lock(mutex);entered=true;changed.notify_all();changed.wait(lock,[&]{return released;});};
    // Release any blocked callback before jthread destruction, including construction failure.
    struct Release{std::mutex& mutex;std::condition_variable& changed;bool& released;~Release(){std::lock_guard lock(mutex);released=true;changed.notify_all();}};
    Batch first,second;Error e1=Error::io,e2=Error::io;
    std::jthread t1([&]{e1=cache.acquire(std::span(&a,1),&first);{std::lock_guard lock(mutex);first_done=true;changed.notify_all();}});
    Release release{mutex,changed,released};
    {std::unique_lock lock(mutex);changed.wait(lock,[&]{return entered||first_done;});require(entered,"first concurrent acquire failed before provider entry");}
    std::jthread t2([&]{second_entered=true;e2=cache.acquire(std::span(&a,1),&second);});
    while(!second_entered.load())std::this_thread::yield();
    {std::lock_guard lock(mutex);released=true;changed.notify_all();}
    t1.join();t2.join();okay(e1);okay(e2);
    require(provider.calls==1&&first.leases[0].bytes().data()==second.leases[0].bytes().data(),"concurrent cold requests duplicated a fill");
    require(cache.stats().cold_hits==1&&cache.stats().cold_bytes==128,"shared cold lease was charged twice");
}
void actual_glm_strides_are_metadata_only_and_refuse_before_allocation(){
    // Executed rank0 catalog: main3342336 B and MTP6299648 B, not tiny test payloads.
    constexpr uint64_t main_stride=3342336,mtp_stride=6299648;
    static_assert(3276800+65536==main_stride);
    static_assert(6295552+512<=mtp_stride&&mtp_stride%4096==0);
    MemoryProvider provider;Cache cache(limits(0,0),provider);Batch batch;
    RecordRef main{key(0),0,main_stride,0},mtp{key(0,"mtp-fp8"),main_stride,mtp_stride,0};
    mtp.key.layer=45;const RecordRef demand[]{main,mtp};
    require(cache.acquire(demand,&batch)==Error::budget&&provider.calls==0,"real-size metadata allocated or read fake model bytes");
    require(cache.stats().live_records==0&&cache.stats().cold_bytes==0,"refused actual-stride metadata charged payload");
}
void cold_last_lease_release_racing_lookup_never_reports_false_io(){
    MemoryProvider provider;std::vector<RecordRef> background;
    for(int i=0;i<128;++i)background.push_back(provider.add(i,128));
    const auto target=provider.add(128,128);
    auto budget=limits(0,128u*132u,512);budget.max_batch=256;budget.max_catalog_records=512;
    Cache cache(budget,provider);Batch keep_background,current;
    okay(cache.acquire(background,&keep_background));okay(cache.acquire(std::span(&target,1),&current));
    std::mutex mutex;std::condition_variable changed;std::atomic<bool> go=false,closing=false;
    Batch transfer;size_t requested=0,ready=0,completed=0;
    std::jthread releaser([&]{
        for(;;){Batch owned;size_t serial=0;
            {std::unique_lock lock(mutex);changed.wait(lock,[&]{return closing.load()||requested>completed;});
                if(closing.load())return;
                serial=requested;owned=std::move(transfer);ready=serial;changed.notify_all();}
            while(!go.load(std::memory_order_acquire)&&!closing.load())std::this_thread::yield();
            for(size_t i=0;i<serial%4;++i)std::this_thread::yield();
            owned.leases.clear(); // The last cold owner races cache prune()/weak.lock().
            {std::lock_guard lock(mutex);completed=serial;changed.notify_all();}
        }
    });
    struct Release{
        std::mutex& mutex;std::condition_variable& changed;std::atomic<bool>& go;std::atomic<bool>& closing;
        ~Release(){std::lock_guard lock(mutex);closing=true;go=true;changed.notify_all();}
    } release{mutex,changed,go,closing};
    // Hook-free bounded stress, not a deterministic claim about scheduler interleaving.
    // Live background weak entries extend prune without retaining the target record.
    for(size_t serial=1;serial<=2000;++serial){
        {std::unique_lock lock(mutex);go=false;transfer=std::move(current);requested=serial;changed.notify_all();changed.wait(lock,[&]{return ready==serial;});}
        go.store(true,std::memory_order_release);Batch next;
        const auto result=cache.acquire(std::span(&target,1),&next);
        {std::unique_lock lock(mutex);changed.wait(lock,[&]{return completed==serial;});}
        okay(result);require(next.leases.size()==1,"racing cold reload lost a selected expert");exact(next.leases[0],provider);current=std::move(next);
    }
    {std::lock_guard lock(mutex);closing=true;go=true;changed.notify_all();}releaser.join();
    current.leases.clear();keep_background.leases.clear();okay(cache.trim());const auto used=cache.accounting().usage();
    require(!used.hot_bytes&&!used.cold_bytes&&!used.live_records,"racing lease-release stress leaked record charges");
}
} // namespace
int main(){
    const std::pair<const char*,void(*)()> tests[]{
        {"duplicate union/mixed classes",duplicate_union_and_mixed_classes},{"whole-union refusal",whole_union_refusal_before_io},
        {"late IO atomicity",late_io_failure_is_atomic},{"read corruption/short/throw cleanup",short_checksum_and_exception_cleanup},
        {"pinned admission bypass",pinned_hot_bypasses_admission_but_serves_cold},{"unleased LRU",lru_evicts_only_unleased_victim},
        {"lease/ledger lifetime",leases_keep_ledger_alive_after_store_destroy},{"immutable identity after eviction",immutable_identity_survives_eviction},
        {"metadata/bounds",malformed_union_and_metadata_caps},{"cancellation",cancelled_fill_rolls_back_without_publishing},
        {"provider reentry",provider_can_query_stats_but_reentry_is_refused},{"concurrent dedup",concurrent_same_key_uses_one_fill},
        {"actual GLM metadata refusal",actual_glm_strides_are_metadata_only_and_refuse_before_allocation},
        {"last cold lease racing lookup",cold_last_lease_release_racing_lookup_never_reports_false_io}};
    size_t passed=0;for(const auto& [name,test]:tests){try{test();++passed;}catch(const std::exception& error){std::cerr<<"FAIL "<<name<<": "<<error.what()<<'\n';return 1;}}
    std::cout<<"PASS "<<passed<<" standalone CPU expert-cache cases; no model payload/GPU/Radiance integration\n";
}
