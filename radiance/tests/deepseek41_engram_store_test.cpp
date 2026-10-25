/* Synthetic SSD row/cache lifecycle tests. No original model payload or device work. */
#include "rad_test.h"
#include "../arch/deepseek41/engram_store.h"
#include <nlohmann/json.hpp>
#include <atomic>
#include <chrono>
#include <fstream>
#include <future>
#include <thread>
namespace eg=rad::deepseek41::engram;
namespace {
using Json=nlohmann::json;
const Json& fixture(){static const auto j=[] {std::ifstream f(std::string(RAD_GOLDEN_FIXTURES)+"/deepseek41-engram-store.json");if(!f)throw std::runtime_error("missing store fixture");return Json::parse(f);}();return j;}
std::vector<uint8_t> unhex(const std::string& s){std::vector<uint8_t> out;for(size_t i=0;i<s.size();i+=2)out.push_back(uint8_t(std::stoul(s.substr(i,2),nullptr,16)));return out;}
std::vector<float> expected(){const auto bytes=unhex(fixture().at("f32_hex"));std::vector<float> out(bytes.size()/4);std::memcpy(out.data(),bytes.data(),bytes.size());return out;}
struct File {
    std::string path;bool other=false;size_t offset=0,size=0;
    explicit File(bool variant=false):other(variant){
        path=radtest::tmp_path("engram-store.XXXXXX");const int fd=mkstemp(path.data());if(fd<0)throw std::runtime_error("mkstemp");
        auto bytes=unhex(fixture().at("prefix_hex"));offset=bytes.size();const auto packed=unhex(fixture().at(other?"other_packed_hex":"packed_hex"));bytes.insert(bytes.end(),packed.begin(),packed.end());size=bytes.size();
        size_t done=0;while(done<bytes.size()){const ssize_t n=::write(fd,bytes.data()+done,bytes.size()-done);if(n<=0){::close(fd);throw std::runtime_error("fixture write");}done+=size_t(n);}::close(fd);
    }
    ~File(){::unlink(path.c_str());}
    eg::Iq4NlAsset asset(int layer=1)const{return {path,fixture().at(other?"other_sha256":"sha256"),"gguf-iq4_nl-block-v1",layer,int64_t(size),int64_t(offset),fixture().at("rows"),fixture().at("width")};}
};
eg::Iq4NlCacheOptions options(size_t capacity=0){eg::Iq4NlCacheOptions out;out.budget_bytes=16384;out.max_batch_rows=8;out.max_io_rows=4;out.max_cached_rows=capacity;return out;}
std::shared_ptr<eg::Iq4NlStore> open(const File& file,eg::Iq4NlCacheOptions config=options()){
    std::shared_ptr<eg::Iq4NlStore> out;std::string why;if(eg::Iq4NlStore::open(file.asset(),std::move(config),&out,&why)!=RAD_OK)throw std::runtime_error(why);return out;
}
void compare(const eg::Iq4NlRowLease& row,int64_t id){
    std::vector<float> decoded(256);REQUIRE_EQ(row.decode_into(decoded),RAD_OK);CHECK_EQ(row.row(),id);const auto wanted=expected();
    for(size_t d=0;d<256;++d)CHECK_EQ(decoded[d],wanted[size_t(id)*256+d]);
}
struct ProbeIo:eg::RowReadBackend {
    std::atomic<int> calls{0},fail_at{0};std::atomic<bool> interrupt{false},eof{false};size_t short_bytes=0;
    std::mutex mutex;std::condition_variable condition;bool block=false,entered=false,released=false;
    ssize_t read(int fd,void* out,size_t bytes,off_t offset)noexcept override{
        const int call=++calls;
        {std::unique_lock lock(mutex);if(block){entered=true;condition.notify_all();condition.wait(lock,[&]{return released;});}}
        if(interrupt.exchange(false)){errno=EINTR;return -1;}
        if(fail_at.load()==call){errno=EIO;return -1;}
        if(eof.exchange(false))return 0;
        return ::pread(fd,out,short_bytes?std::min(short_bytes,bytes):bytes,offset);
    }
    bool wait_entered(){std::unique_lock lock(mutex);return condition.wait_for(lock,std::chrono::seconds(2),[&]{return entered;});}
    void release(){std::lock_guard lock(mutex);released=true;condition.notify_all();}
};
}
TEST(deepseek41_iq4_store_decodes_independent_blocks_at_explicit_unaligned_offset){
    File file;auto store=open(file);const auto stats=store->stats();CHECK(stats.allocation_bytes<=options().budget_bytes);CHECK_EQ(stats.allocation_bytes,stats.slab_bytes+stats.fixed_bytes);
    CHECK_EQ(stats.payload_bytes,stats.capacity_rows*144);CHECK(stats.metadata_bytes>0);CHECK_EQ(stats.io_scratch_bytes,576u);
    for(int64_t i=0;i<4;++i){eg::Iq4NlRowLease row;REQUIRE_EQ(store->lease(i,&row),RAD_OK);compare(row,i);}
    CHECK_EQ(store->stats().active_references,0u);CHECK_EQ(store->stats().loading_rows,0u);
}
TEST(deepseek41_iq4_store_batches_deduplicates_and_coalesces_only_requested_adjacent_misses){
    File file;auto store=open(file);const std::vector<int64_t> ids{3,1,1,2,0};std::vector<eg::Iq4NlRowLease> leases(ids.size());
    REQUIRE_EQ(store->lease_many(ids,leases),RAD_OK);for(size_t i=0;i<ids.size();++i)compare(leases[i],ids[i]);
    CHECK_EQ(leases[1].packed().data(),leases[2].packed().data());auto stats=store->stats();CHECK_EQ(stats.loaded_rows,4u);CHECK_EQ(stats.pread_calls,1u);CHECK_EQ(stats.bytes_read,576u);CHECK_EQ(stats.duplicate_rows,1u);
    CHECK_EQ(stats.pinned_rows,4u);CHECK_EQ(stats.active_references,5u);leases.clear();CHECK_EQ(store->stats().pinned_rows,0u);
    std::vector<float> out(3*256);const std::vector<int64_t> repeated{1,1,3};REQUIRE_EQ(store->copy_rows(repeated,out),RAD_OK);CHECK_EQ(store->stats().pread_calls,1u);
    const auto wanted=expected();for(size_t i=0;i<repeated.size();++i)for(size_t d=0;d<256;++d)CHECK_EQ(out[i*256+d],wanted[size_t(repeated[i])*256+d]);
}
TEST(deepseek41_iq4_store_pinned_leases_prevent_eviction_and_remain_owned_after_store_reset){
    File file;auto store=open(file,options(2));eg::Iq4NlRowLease first,second,next;
    REQUIRE_EQ(store->lease(0,&first),RAD_OK);REQUIRE_EQ(store->lease(1,&second),RAD_OK);
    CHECK_EQ(store->lease(2,&next),RAD_E_FULL);CHECK(!next);compare(first,0);second.reset();
    REQUIRE_EQ(store->lease(2,&next),RAD_OK);compare(first,0);CHECK(store->stats().evictions>=1);
    next.reset();std::weak_ptr<eg::Iq4NlStore> weak=store;store.reset();CHECK(!weak.expired());compare(first,0);first.reset();CHECK(weak.expired());
}
TEST(deepseek41_iq4_store_copy_rows_streams_over_a_smaller_cache_and_adapter_checks_geometry){
    File file;auto store=open(file,options(1));const std::vector<int64_t> ids{0,1,2,3};std::vector<float> out(1024);
    auto lookup=store->row_lookup();REQUIRE_EQ(lookup(1,ids,256,out,nullptr),RAD_OK);CHECK(out==expected());
    CHECK_EQ(lookup(14,ids,256,out,nullptr),RAD_E_SHAPE);CHECK_EQ(lookup(1,ids,128,out,nullptr),RAD_E_SHAPE);
    CHECK_EQ(lookup(1,ids,256,std::span(out).first(4),nullptr),RAD_E_SHAPE);CHECK(store->stats().allocation_bytes<=16384);
}
TEST(deepseek41_iq4_store_short_reads_and_eintr_preserve_exact_values){
    File file;auto io=std::make_shared<ProbeIo>();io->interrupt=true;io->short_bytes=7;auto config=options();config.backend=io;auto store=open(file,config);
    const std::vector<int64_t> ids{0,1,2,3};std::vector<float> out(1024);REQUIRE_EQ(store->copy_rows(ids,out),RAD_OK);CHECK(out==expected());
    CHECK_EQ(store->stats().bytes_read,576u);CHECK(store->stats().pread_calls>80);CHECK_EQ(store->stats().loading_rows,0u);
}
TEST(deepseek41_iq4_store_failed_batch_never_publishes_partial_rows_and_retry_cleans_reservations){
    File file;auto io=std::make_shared<ProbeIo>();io->fail_at=2;auto config=options();config.backend=io;config.max_io_rows=1;auto store=open(file,config);
    const std::vector<int64_t> ids{0,1,2};std::vector<eg::Iq4NlRowLease> leases(3);
    CHECK_EQ(store->lease_many(ids,leases),RAD_E_IO);for(const auto& lease:leases)CHECK(!lease);
    auto stats=store->stats();CHECK_EQ(stats.loading_rows,0u);CHECK_EQ(stats.resident_rows,0u);CHECK_EQ(stats.active_references,0u);CHECK_EQ(stats.loaded_rows,0u);
    io->fail_at=0;REQUIRE_EQ(store->lease_many(ids,leases),RAD_OK);for(size_t i=0;i<ids.size();++i)compare(leases[i],ids[i]);
}
TEST(deepseek41_iq4_store_eof_failure_clears_loading_state){
    File file;auto io=std::make_shared<ProbeIo>();io->eof=true;auto config=options();config.backend=io;auto store=open(file,config);eg::Iq4NlRowLease row;
    CHECK_EQ(store->lease(0,&row),RAD_E_IO);CHECK_EQ(store->stats().loading_rows,0u);CHECK_EQ(store->stats().pinned_rows,0u);
    REQUIRE_EQ(store->lease(0,&row),RAD_OK);compare(row,0);
}
TEST(deepseek41_iq4_store_concurrent_same_row_misses_share_one_read){
    File file;auto io=std::make_shared<ProbeIo>();io->block=true;auto config=options();config.backend=io;auto store=open(file,config);
    std::promise<int> first;auto result=first.get_future();std::thread loader([&]{eg::Iq4NlRowLease lease;first.set_value(store->lease(0,&lease));});
    const bool entered=io->wait_entered();CHECK(entered);std::array<int,4> status{};std::vector<std::thread> readers;
    for(size_t i=0;i<4;++i)readers.emplace_back([&,i]{eg::Iq4NlRowLease row;status[i]=store->lease(0,&row);});
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(1);
    while(store->stats().waits<4&&std::chrono::steady_clock::now()<deadline)std::this_thread::yield();
    const auto in_flight=store->stats();CHECK_EQ(in_flight.loading_rows,1u);CHECK_EQ(in_flight.in_flight_io_bytes,144u);CHECK(in_flight.allocation_bytes<=config.budget_bytes);
    io->release();loader.join();for(auto& reader:readers)reader.join();CHECK_EQ(result.get(),RAD_OK);for(auto s:status)CHECK_EQ(s,RAD_OK);
    CHECK_EQ(store->stats().loaded_rows,1u);CHECK_EQ(store->stats().pread_calls,1u);CHECK_EQ(store->stats().active_references,0u);
}
TEST(deepseek41_iq4_store_cold_io_does_not_hold_ready_hit_mutex){
    File file;auto io=std::make_shared<ProbeIo>();auto config=options();config.backend=io;auto store=open(file,config);eg::Iq4NlRowLease warm;
    REQUIRE_EQ(store->lease(0,&warm),RAD_OK);warm.reset();{std::lock_guard lock(io->mutex);io->block=true;}
    std::thread loader([&]{eg::Iq4NlRowLease row;CHECK_EQ(store->lease(1,&row),RAD_OK);});CHECK(io->wait_entered());
    std::promise<int> hit;auto ready=hit.get_future();std::thread reader([&]{eg::Iq4NlRowLease row;hit.set_value(store->lease(0,&row));});
    const bool completed=ready.wait_for(std::chrono::seconds(1))==std::future_status::ready;CHECK(completed);
    io->release();reader.join();loader.join();CHECK_EQ(ready.get(),RAD_OK);
}
TEST(deepseek41_iq4_store_hot_rowlookup_batch_bypasses_unrelated_cold_io){
    File file;auto io=std::make_shared<ProbeIo>();auto config=options();config.backend=io;auto store=open(file,config);
    eg::Iq4NlRowLease warm;REQUIRE_EQ(store->lease(0,&warm),RAD_OK);warm.reset();
    {std::lock_guard lock(io->mutex);io->block=true;}
    std::thread loader([&]{eg::Iq4NlRowLease row;CHECK_EQ(store->lease(1,&row),RAD_OK);});CHECK(io->wait_entered());
    auto lookup=store->row_lookup();std::promise<int> hit;auto ready=hit.get_future();
    std::thread reader([&]{const std::vector<int64_t> ids{0,0};std::vector<float> decoded(512);hit.set_value(lookup(1,ids,256,decoded,nullptr));});
    const bool completed=ready.wait_for(std::chrono::seconds(1))==std::future_status::ready;CHECK(completed);
    io->release();reader.join();loader.join();CHECK_EQ(ready.get(),RAD_OK);
}
TEST(deepseek41_iq4_store_cancel_wakes_loading_and_workspace_waiters_and_reclaims_failed_load){
    File file;auto io=std::make_shared<ProbeIo>();io->block=true;auto config=options();config.backend=io;auto store=open(file,config);
    int loader_status=0;std::thread loader([&]{eg::Iq4NlRowLease row;loader_status=store->lease(0,&row);});CHECK(io->wait_entered());
    std::promise<int> waiting,batched;auto done=waiting.get_future(),batch_done=batched.get_future();
    std::thread waiter([&]{eg::Iq4NlRowLease row;waiting.set_value(store->lease(0,&row));});
    std::thread batch_waiter([&]{std::vector<int64_t> ids{1};std::vector<eg::Iq4NlRowLease> leases(1);batched.set_value(store->lease_many(ids,leases));});
    store->cancel();CHECK(done.wait_for(std::chrono::seconds(1))==std::future_status::ready);CHECK(batch_done.wait_for(std::chrono::seconds(1))==std::future_status::ready);
    io->release();waiter.join();batch_waiter.join();loader.join();CHECK_EQ(done.get(),RAD_E_CANCELLED);CHECK_EQ(batch_done.get(),RAD_E_CANCELLED);CHECK_EQ(loader_status,RAD_E_CANCELLED);
    CHECK_EQ(store->stats().loading_rows,0u);CHECK_EQ(store->stats().active_references,0u);
}
TEST(deepseek41_iq4_store_request_stop_wakes_waiting_batch_without_shutting_down_shared_cache){
    File file;auto io=std::make_shared<ProbeIo>();auto config=options();config.backend=io;auto store=open(file,config);
    eg::Iq4NlRowLease warm;REQUIRE_EQ(store->lease(1,&warm),RAD_OK);warm.reset();
    {std::lock_guard lock(io->mutex);io->block=true;}
    int loader_status=0;std::thread loader([&]{eg::Iq4NlRowLease row;loader_status=store->lease(0,&row);});CHECK(io->wait_entered());
    std::stop_source stop;std::promise<int> waiting;auto done=waiting.get_future();auto stopped_lookup=store->row_lookup(stop.get_token());
    const auto waits_before=store->stats().waits;
    std::thread waiter([&]{const std::vector<int64_t> ids{2};std::vector<float> decoded(256);waiting.set_value(stopped_lookup(1,ids,256,decoded,nullptr));});
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(1);
    while(store->stats().waits==waits_before&&std::chrono::steady_clock::now()<deadline)std::this_thread::yield();
    CHECK(store->stats().waits>waits_before);stop.request_stop();
    CHECK(done.wait_for(std::chrono::seconds(1))==std::future_status::ready);
    const std::vector<int64_t> hot{1};std::vector<float> decoded(256);CHECK_EQ(store->row_lookup()(1,hot,256,decoded,nullptr),RAD_OK);
    io->release();waiter.join();loader.join();CHECK_EQ(done.get(),RAD_E_CANCELLED);CHECK_EQ(loader_status,RAD_OK);
    const std::vector<int64_t> cold{2};REQUIRE_EQ(store->copy_rows(cold,decoded),RAD_OK);CHECK_EQ(store->stats().loading_rows,0u);
}
TEST(deepseek41_iq4_store_request_stop_discards_its_cold_load_and_later_requests_still_work){
    File file;auto io=std::make_shared<ProbeIo>();auto config=options();config.backend=io;auto store=open(file,config);
    eg::Iq4NlRowLease warm;REQUIRE_EQ(store->lease(1,&warm),RAD_OK);warm.reset();
    {std::lock_guard lock(io->mutex);io->block=true;}
    std::stop_source stop;auto stopped_lookup=store->row_lookup(stop.get_token());int status=0;
    std::thread loader([&]{const std::vector<int64_t> ids{0};std::vector<float> decoded(256);status=stopped_lookup(1,ids,256,decoded,nullptr);});
    CHECK(io->wait_entered());stop.request_stop();
    const std::vector<int64_t> hot{1};std::vector<float> decoded(256);CHECK_EQ(store->copy_rows(hot,decoded),RAD_OK);
    io->release();loader.join();CHECK_EQ(status,RAD_E_CANCELLED);CHECK_EQ(store->stats().loading_rows,0u);CHECK_EQ(store->stats().active_references,0u);
    const std::vector<int64_t> retry{0};REQUIRE_EQ(store->copy_rows(retry,decoded),RAD_OK);
    CHECK_EQ(stopped_lookup(1,retry,256,decoded,nullptr),RAD_E_CANCELLED);
    REQUIRE_EQ(store->row_lookup()(1,retry,256,decoded,nullptr),RAD_OK);
}
TEST(deepseek41_iq4_store_refuses_changed_asset_even_for_a_cached_hit_but_existing_lease_stays_valid){
    File file;auto store=open(file);eg::Iq4NlRowLease row,next;REQUIRE_EQ(store->lease(0,&row),RAD_OK);
    REQUIRE_EQ(::truncate(file.path.c_str(),off_t(file.offset+144)),0);CHECK_EQ(store->lease(0,&next),RAD_E_FORMAT);CHECK(!next);compare(row,0);
}
TEST(deepseek41_iq4_store_layer_group_enforces_one_total_budget_and_asset_namespaces){
    File first,second(true);auto config=options();config.budget_bytes=UINT64_MAX;
    std::shared_ptr<eg::Iq4NlLayerStores> group;REQUIRE_EQ(eg::Iq4NlLayerStores::open("test-model",{first.asset(1),second.asset(14)},16384,config,&group),RAD_OK);
    CHECK_EQ(group->budget().limit,16384u);CHECK(group->budget().allocation<=16384);CHECK_EQ(group->stores().size(),2u);
    uint64_t total=group->budget().fixed;for(const auto& store:group->stores())total+=store->stats().allocation_bytes;CHECK_EQ(total,group->budget().allocation);
    auto lookup=group->row_lookup();const std::vector<int64_t> ids{0};std::vector<float> a(256),b(256);
    REQUIRE_EQ(lookup(1,ids,256,a,nullptr),RAD_OK);REQUIRE_EQ(lookup(14,ids,256,b,nullptr),RAD_OK);CHECK(a[0]!=b[0]);
    const auto wanted=expected();CHECK_EQ(a[0],wanted[0]);CHECK_EQ(lookup(3,ids,256,a,nullptr),RAD_E_NOTFOUND);
    CHECK_EQ(eg::Iq4NlLayerStores::open("test",{first.asset(1),second.asset(1)},16384,options(),&group),RAD_E_DUPLICATE);
}
TEST(deepseek41_iq4_store_invalid_manifest_ranges_formats_and_budgets_fail_without_opening_large_assets){
    File file;std::shared_ptr<eg::Iq4NlStore> store;auto spec=file.asset();auto config=options();
    spec.format="43";CHECK_EQ(eg::Iq4NlStore::open(spec,config,&store),RAD_E_UNSUPPORTED);
    spec=file.asset();spec.sha256="bad";CHECK_EQ(eg::Iq4NlStore::open(spec,config,&store),RAD_E_SHAPE);
    spec=file.asset();spec.width=255;CHECK_EQ(eg::Iq4NlStore::open(spec,config,&store),RAD_E_SHAPE);
    spec=file.asset();spec.rows=INT64_MAX;CHECK_EQ(eg::Iq4NlStore::open(spec,config,&store),RAD_E_SHAPE);
    spec=file.asset();spec.data_offset=INT64_MAX;CHECK_EQ(eg::Iq4NlStore::open(spec,config,&store),RAD_E_SHAPE);
    spec=file.asset();spec.file_bytes++;CHECK_EQ(eg::Iq4NlStore::open(spec,config,&store),RAD_E_FORMAT);
    spec=file.asset();spec.path+=".missing";CHECK_EQ(eg::Iq4NlStore::open(spec,config,&store),RAD_E_IO);
    config=options();config.budget_bytes=1;CHECK_EQ(eg::Iq4NlStore::open(file.asset(),config,&store),RAD_E_NOMEM);CHECK(!store);
    eg::Iq4NlRowLease empty;std::vector<float> out(256);CHECK_EQ(empty.decode_into(out),RAD_E_STATE);
}
RAD_TEST_MAIN()
