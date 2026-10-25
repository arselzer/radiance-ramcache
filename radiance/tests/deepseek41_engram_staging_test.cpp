/* CPU fake backend: manual DMA and last-reader phases expose lifetime mistakes.
 * No HIP calls, physical pinning, device execution or model payloads. */
#include "rad_test.h"
#include "../arch/deepseek41/engram_staging.h"
#include <nlohmann/json.hpp>
#include <atomic>
#include <chrono>
#include <fstream>
#include <future>
#include <thread>
namespace eg=rad::deepseek41::engram;
namespace fixtures {
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
namespace {
struct FakeHost:eg::HostRowAllocation {
    std::vector<std::byte> data;uint64_t charge;std::shared_ptr<std::atomic<int>> live;
    FakeHost(size_t bytes,uint64_t n,std::shared_ptr<std::atomic<int>> count):data(bytes),charge(n),live(std::move(count)){++*live;}
    ~FakeHost(){--*live;}
    std::span<std::byte> memory()noexcept override{return data;}
    uint64_t accounted_bytes()const noexcept override{return charge;}
};
struct RetainedHostState {
    eg::StagingCharge charge;std::shared_ptr<FakeHost> physical;
};
struct RetainedHost:eg::HostRowAllocation {
    std::shared_ptr<RetainedHostState> state;
    explicit RetainedHost(std::shared_ptr<RetainedHostState> s):state(std::move(s)){}
    std::span<std::byte> memory()noexcept override{return state->physical->memory();}
    uint64_t accounted_bytes()const noexcept override{return state->physical->accounted_bytes();}
    void adopt_budget_charge(eg::StagingCharge& charge)noexcept override{state->charge=std::move(charge);}
};
struct FakeDevice:eg::DeviceRowBuffer {
    std::string model;uint64_t device=0;std::vector<std::byte> data;
    explicit FakeDevice(std::string id,size_t bytes=4096):model(std::move(id)),data(bytes){}
    std::string_view model_identity()const noexcept override{return model;}
    uint64_t device_identity()const noexcept override{return device;}
    uintptr_t address()const noexcept override{return reinterpret_cast<uintptr_t>(data.data());}
    size_t capacity_bytes()const noexcept override{return data.size();}
};
struct ManualCompletion:eg::RowTransferCompletion {
    std::weak_ptr<eg::RowTransferRecord> record;std::mutex mutex;bool dma=false,readers=false,cancelled=false;int error=0;
    eg::RowTransferState poll()noexcept override{std::lock_guard lock(mutex);return {dma&&readers,error};}
    void request_cancel()noexcept override{std::lock_guard lock(mutex);cancelled=true;}
    void dma_done(){
        auto owner=record.lock();if(!owner)throw std::runtime_error("submitted record died before simulated DMA");
        const auto& s=owner->submission();std::memcpy(reinterpret_cast<void*>(s.destination.owner->address()+s.destination.offset),s.host->bytes().data(),s.bytes);
        std::lock_guard lock(mutex);dma=true;
    }
    void readers_done(){std::lock_guard lock(mutex);assert(dma);readers=true;}
    bool cancellation(){std::lock_guard lock(mutex);return cancelled;}
};
struct FakeBackend:eg::TrackedRowTransferBackend {
    enum Mode{normal,fail_before,fail_after,unknown_failure,bad_success};Mode mode=normal;
    int alloc_calls=0,enqueue_calls=0,fail_alloc_at=0;bool overcharge=false,force_safe_drain=false;
    bool retain_host=false,error_after_host_alloc=false;std::vector<std::shared_ptr<RetainedHostState>> retained_hosts;
    std::shared_ptr<std::atomic<int>> live=std::make_shared<std::atomic<int>>(0);
    std::vector<std::pair<std::weak_ptr<eg::RowTransferRecord>,std::weak_ptr<ManualCompletion>>> entries;
    std::weak_ptr<ManualCompletion> last;
    std::mutex enqueue_mutex;std::condition_variable enqueue_condition;bool block_enqueue=false,enqueue_entered=false,enqueue_released=false;
    std::atomic<int> drain_calls{0};
    int allocation_charge(size_t bytes,uint64_t* charge)const noexcept override{*charge=uint64_t(bytes)+128;return RAD_OK;}
    int allocate_host(size_t bytes,uint64_t cap,std::shared_ptr<eg::HostRowAllocation>* out)noexcept override{
        if(++alloc_calls==fail_alloc_at)return RAD_E_NOMEM;
        try{
            auto host=std::make_shared<FakeHost>(bytes,cap+(overcharge?1:0),live);
            if(retain_host){auto state=std::make_shared<RetainedHostState>();state->physical=std::move(host);*out=std::make_shared<RetainedHost>(state);retained_hosts.push_back(std::move(state));}
            else *out=std::move(host);
            return error_after_host_alloc?RAD_E_IO:RAD_OK;
        }catch(...){return RAD_E_NOMEM;}
    }
    eg::NativeRowEnqueue enqueue_native(const std::shared_ptr<eg::RowTransferRecord>& record)noexcept override{
        ++enqueue_calls;
        {std::unique_lock lock(enqueue_mutex);if(block_enqueue){enqueue_entered=true;enqueue_condition.notify_all();enqueue_condition.wait(lock,[&]{return enqueue_released;});}}
        if(mode==fail_before)return {RAD_E_IO,{},true};
        try{
            if(mode==unknown_failure||mode==bad_success){entries.push_back({record,{}});return {mode==unknown_failure?RAD_E_IO:RAD_OK,{},false};}
            auto completion=std::make_shared<ManualCompletion>();completion->record=record;
            entries.push_back({record,completion});last=completion;return {mode==fail_after?RAD_E_IO:RAD_OK,completion,false};
        }catch(...){return {RAD_E_NOMEM,{},true};}
    }
    eg::RowTransferState drain_native(const std::shared_ptr<eg::RowTransferRecord>& record)noexcept override{
        ++drain_calls;
        for(const auto& entry:entries)if(entry.first.lock()==record){
            if(auto completion=entry.second.lock()){const auto state=completion->poll();if(state.quiescent)return state;}
        }
        return force_safe_drain?eg::RowTransferState{true,RAD_OK}:eg::RowTransferState{false,RAD_E_IO};
    }
    void finish_all(){
        for(auto& entry:entries)if(auto completion=entry.second.lock())if(!completion->poll().quiescent){completion->dma_done();completion->readers_done();}
        force_safe_drain=true;drain_all();
    }
    bool wait_enqueue(){std::unique_lock lock(enqueue_mutex);return enqueue_condition.wait_for(lock,std::chrono::seconds(2),[&]{return enqueue_entered;});}
    void release_enqueue(){std::lock_guard lock(enqueue_mutex);enqueue_released=true;enqueue_condition.notify_all();}
};
struct World {
    fixtures::File first,second{true};std::shared_ptr<eg::Iq4NlLayerStores> source;
    std::shared_ptr<FakeBackend> backend=std::make_shared<FakeBackend>();
    std::shared_ptr<eg::StagingBudget> budget=std::make_shared<eg::StagingBudget>(65536);
    explicit World(std::shared_ptr<eg::RowReadBackend> io={}){
        auto config=fixtures::options();config.backend=std::move(io);std::string why;
        if(eg::Iq4NlLayerStores::open("stage-model",{first.asset(1),second.asset(14)},16384,config,&source,&why)!=RAD_OK)throw std::runtime_error(why);
    }
    ~World(){backend->finish_all();}
    std::shared_ptr<eg::EngramStagingPool> pool(size_t slots=1){
        std::shared_ptr<eg::EngramStagingPool> out;std::string why;
        if(eg::EngramStagingPool::create("stage-model",source,budget,backend,{slots,2,256},&out,&why)!=RAD_OK)throw std::runtime_error(why);
        return out;
    }
    eg::DeviceRowTarget target(int layer=1){return {std::make_shared<FakeDevice>("stage-model"),16,layer==1?first.asset().sha256:second.asset().sha256};}
};
std::vector<float> copied(const eg::DeviceRowTarget& target,size_t rows){std::vector<float> out(rows*256);std::memcpy(out.data(),reinterpret_cast<const void*>(target.owner->address()+target.offset),out.size()*sizeof(float));return out;}
}
TEST(deepseek41_staging_never_reuses_host_slot_before_dma_and_last_reader_completion){
    World w;auto pool=w.pool();auto target=w.target();const std::vector<int64_t> first{0},next{1};eg::RowTransferTicket ticket,blocked;
    REQUIRE_EQ(pool->submit(1,first,target,&ticket),RAD_OK);auto completion=w.backend->last.lock();REQUIRE(completion);
    CHECK_EQ(pool->submit(1,next,w.target(),&blocked),RAD_E_FULL);completion->dma_done();
    CHECK_EQ(pool->submit(1,next,w.target(),&blocked),RAD_E_FULL);CHECK(!ticket.state().quiescent);
    const auto actual=copied(target,1),expected=fixtures::expected();for(size_t i=0;i<256;++i)CHECK_EQ(actual[i],expected[i]);
    completion->readers_done();CHECK(ticket.state().quiescent);CHECK_EQ(pool->reap(),0u);
    REQUIRE_EQ(pool->submit(1,next,w.target(),&blocked),RAD_OK);CHECK_EQ(w.backend->live->load(),1);w.backend->finish_all();
}
TEST(deepseek41_staging_request_cancellation_is_advisory_and_does_not_cancel_global_cache){
    World w;auto pool=w.pool();std::stop_source stop;eg::RowTransferTicket ticket,other;const std::vector<int64_t> ids{0};
    REQUIRE_EQ(pool->submit(1,ids,w.target(),&ticket,stop.get_token()),RAD_OK);auto completion=w.backend->last.lock();REQUIRE(completion);
    stop.request_stop();ticket.cancel();CHECK(completion->cancellation());CHECK(!ticket.state().quiescent);CHECK_EQ(pool->submit(1,ids,w.target(),&other),RAD_E_FULL);
    std::vector<float> cpu(256);REQUIRE_EQ(w.source->row_lookup()(1,ids,256,cpu,nullptr),RAD_OK);
    completion->dma_done();completion->readers_done();pool->reap();REQUIRE_EQ(pool->submit(1,ids,w.target(),&other),RAD_OK);w.backend->finish_all();
}
TEST(deepseek41_staging_pending_error_does_not_mean_resources_are_quiescent){
    World w;auto pool=w.pool();eg::RowTransferTicket ticket,next;const std::vector<int64_t> ids{0};
    REQUIRE_EQ(pool->submit(1,ids,w.target(),&ticket),RAD_OK);auto completion=w.backend->last.lock();REQUIRE(completion);
    {std::lock_guard lock(completion->mutex);completion->error=RAD_E_IO;}
    CHECK_EQ(ticket.state().status,RAD_E_IO);CHECK(!ticket.state().quiescent);CHECK_EQ(pool->reap(),1u);CHECK_EQ(pool->submit(1,ids,w.target(),&next),RAD_E_FULL);
    completion->dma_done();completion->readers_done();CHECK(ticket.state().quiescent);CHECK_EQ(pool->reap(),0u);
}
TEST(deepseek41_staging_partial_submission_error_retains_memory_and_destination_past_pool_destruction){
    World w;auto probe=w.pool();const auto needed=w.budget->stats().used;probe.reset();CHECK_EQ(w.budget->stats().used,0u);
    w.budget=std::make_shared<eg::StagingBudget>(needed);auto pool=w.pool();w.backend->mode=FakeBackend::fail_after;
    auto target=w.target();std::weak_ptr<eg::DeviceRowBuffer> device=target.owner;eg::RowTransferTicket ticket;const std::vector<int64_t> ids{0};
    CHECK_EQ(pool->submit(1,ids,target,&ticket),RAD_E_IO);REQUIRE(ticket);auto completion=w.backend->last.lock();REQUIRE(completion);
    CHECK(!ticket.state().quiescent);ticket.reset();target.owner.reset();pool.reset();
    CHECK(!device.expired());const auto retained=w.budget->stats().used;CHECK(retained>0);CHECK_EQ(w.backend->pending_count(),1u);
    std::shared_ptr<eg::EngramStagingPool> another;CHECK_EQ(eg::EngramStagingPool::create("stage-model",w.source,w.budget,w.backend,{1,2,256},&another),RAD_E_FULL);
    CHECK_EQ(w.budget->stats().used,retained);completion->dma_done();completion->readers_done();w.backend->progress();
    CHECK_EQ(w.budget->stats().used,0u);CHECK(device.expired());CHECK_EQ(w.backend->live->load(),0);
}
TEST(deepseek41_staging_tracker_rejects_overlapping_device_ranges_before_second_enqueue){
    World w;auto pool=w.pool(2);auto target=w.target();const std::vector<int64_t> a{0},b{1};eg::RowTransferTicket first,overlap,next;
    REQUIRE_EQ(pool->submit(1,a,target,&first),RAD_OK);CHECK_EQ(pool->submit(1,b,target,&overlap),RAD_E_FULL);CHECK_EQ(w.backend->enqueue_calls,1);
    REQUIRE(overlap);CHECK(overlap.state().quiescent);REQUIRE_EQ(pool->submit(1,b,w.target(),&next),RAD_OK);CHECK_EQ(w.backend->enqueue_calls,2);w.backend->finish_all();
}
TEST(deepseek41_staging_allocation_and_no_work_submission_errors_release_unused_slots){
    World w;w.backend->fail_alloc_at=1;std::shared_ptr<eg::EngramStagingPool> pool;
    CHECK_EQ(eg::EngramStagingPool::create("stage-model",w.source,w.budget,w.backend,{1,2,256},&pool),RAD_E_NOMEM);CHECK_EQ(w.budget->stats().used,0u);
    w.backend->fail_alloc_at=0;w.backend->overcharge=true;
    CHECK_EQ(eg::EngramStagingPool::create("stage-model",w.source,w.budget,w.backend,{1,2,256},&pool),RAD_E_ABI);CHECK_EQ(w.budget->stats().used,0u);CHECK_EQ(w.backend->live->load(),0);
    w.backend->overcharge=false;pool=w.pool();w.backend->mode=FakeBackend::fail_before;const std::vector<int64_t> ids{0};eg::RowTransferTicket failed;
    CHECK_EQ(pool->submit(1,ids,w.target(),&failed),RAD_E_IO);CHECK(failed.state().quiescent);CHECK_EQ(w.backend->pending_count(),0u);CHECK_EQ(pool->reap(),0u);
    w.backend->mode=FakeBackend::normal;REQUIRE_EQ(pool->submit(1,ids,w.target(),&failed),RAD_OK);w.backend->finish_all();
}
TEST(deepseek41_staging_missing_completion_is_quarantined_until_explicit_safe_drain){
    World w;auto pool=w.pool();w.backend->mode=FakeBackend::bad_success;const std::vector<int64_t> ids{0};eg::RowTransferTicket ticket;
    CHECK_EQ(pool->submit(1,ids,w.target(),&ticket),RAD_E_ABI);CHECK(!ticket.state().quiescent);ticket.reset();pool.reset();CHECK(w.budget->stats().used>0);
    CHECK(!w.backend->drain_all("stage-model").quiescent);CHECK_EQ(w.backend->pending_count(),1u);
    w.backend->force_safe_drain=true;CHECK(w.backend->drain_all("stage-model").quiescent);CHECK_EQ(w.budget->stats().used,0u);
}
TEST(deepseek41_staging_unknown_failure_without_fence_is_not_treated_as_no_work){
    World w;auto pool=w.pool();w.backend->mode=FakeBackend::unknown_failure;const std::vector<int64_t> ids{0};eg::RowTransferTicket ticket;
    CHECK_EQ(pool->submit(1,ids,w.target(),&ticket),RAD_E_IO);CHECK(!ticket.state().quiescent);ticket.reset();pool.reset();CHECK_EQ(w.backend->pending_count(),1u);
    w.backend->force_safe_drain=true;const auto safe=w.backend->drain_all("stage-model");CHECK(safe.quiescent);CHECK_EQ(safe.status,RAD_E_IO);CHECK_EQ(w.budget->stats().used,0u);
}
TEST(deepseek41_staging_identity_geometry_and_cancelled_decode_fail_before_enqueue){
    World w;auto pool=w.pool();const std::vector<int64_t> ids{0},bad{99};eg::RowTransferTicket ticket;auto target=w.target();
    target.source_sha256=w.second.asset().sha256;CHECK_EQ(pool->submit(1,ids,target,&ticket),RAD_E_SHAPE);
    target=w.target();target.owner=std::make_shared<FakeDevice>("another-model");CHECK_EQ(pool->submit(1,ids,target,&ticket),RAD_E_SHAPE);
    target=w.target();target.offset=SIZE_MAX;CHECK_EQ(pool->submit(1,ids,target,&ticket),RAD_E_SHAPE);
    CHECK_EQ(pool->submit(1,bad,w.target(),&ticket),RAD_E_SHAPE);CHECK_EQ(pool->submit(9,ids,w.target(),&ticket),RAD_E_SHAPE);
    std::stop_source stop;stop.request_stop();CHECK_EQ(pool->submit(1,ids,w.target(),&ticket,stop.get_token()),RAD_E_CANCELLED);CHECK_EQ(w.backend->enqueue_calls,0);
}
TEST(deepseek41_staging_cancellation_while_source_read_fills_slot_never_enqueues_device_work){
    auto io=std::make_shared<fixtures::ProbeIo>();io->block=true;World w(io);auto pool=w.pool();std::stop_source stop;int status=0;const std::vector<int64_t> ids{0};
    std::thread submitter([&]{eg::RowTransferTicket ticket;status=pool->submit(1,ids,w.target(),&ticket,stop.get_token());});CHECK(io->wait_entered());
    CHECK_EQ(pool->stats().filling,1u);stop.request_stop();io->release();submitter.join();CHECK_EQ(status,RAD_E_CANCELLED);CHECK_EQ(w.backend->enqueue_calls,0);CHECK_EQ(pool->stats().free,1u);
    std::vector<float> out(256);CHECK_EQ(w.source->row_lookup()(1,ids,256,out,nullptr),RAD_OK);
}
TEST(deepseek41_staging_shutdown_serializes_with_native_enqueue_and_retains_pending_work){
    World w;auto pool=w.pool();w.backend->block_enqueue=true;const std::vector<int64_t> ids{0};eg::RowTransferTicket ticket;int submitted=RAD_E_STATE;
    std::thread submitter([&]{submitted=pool->submit(1,ids,w.target(),&ticket);});CHECK(w.backend->wait_enqueue());
    auto shutdown=std::async(std::launch::async,[&]{return pool->shutdown();});
    CHECK_EQ(shutdown.wait_for(std::chrono::milliseconds(20)),std::future_status::timeout);
    CHECK_EQ(w.backend->drain_calls.load(),0);w.backend->release_enqueue();submitter.join();
    const auto state=shutdown.get();CHECK_EQ(submitted,RAD_OK);CHECK(!state.quiescent);CHECK_EQ(w.backend->pending_count(),1u);
    auto completion=w.backend->last.lock();REQUIRE(completion);CHECK(completion->cancellation());
    completion->dma_done();completion->readers_done();CHECK_EQ(pool->reap(),0u);CHECK(pool->shutdown().quiescent);
    std::vector<float> out(256);CHECK_EQ(w.source->row_lookup()(1,ids,256,out,nullptr),RAD_OK);
}
TEST(deepseek41_staging_two_layer_pools_share_one_cap_and_keep_asset_identity){
    World w;auto probe=w.pool();const auto one_pool=w.budget->stats().used;probe.reset();
    w.budget=std::make_shared<eg::StagingBudget>(2*one_pool);auto first=w.pool(),second=w.pool();
    CHECK_EQ(w.budget->stats().used,2*one_pool);std::shared_ptr<eg::EngramStagingPool> excess;
    CHECK_EQ(eg::EngramStagingPool::create("stage-model",w.source,w.budget,w.backend,{1,2,256},&excess),RAD_E_FULL);
    const std::vector<int64_t> ids{0};auto target1=w.target(1),target14=w.target(14);eg::RowTransferTicket ticket1,ticket14;
    REQUIRE_EQ(first->submit(1,ids,target1,&ticket1),RAD_OK);auto completion1=w.backend->last.lock();REQUIRE(completion1);
    REQUIRE_EQ(second->submit(14,ids,target14,&ticket14),RAD_OK);auto completion14=w.backend->last.lock();REQUIRE(completion14);
    const auto record=w.backend->entries.front().first.lock();REQUIRE(record);
    CHECK_EQ(w.backend->enqueue(record),RAD_E_STATE);CHECK_EQ(w.backend->enqueue_calls,2);CHECK(!ticket1.state().quiescent);
    auto foreign=std::make_shared<FakeBackend>();CHECK_EQ(foreign->enqueue(record),RAD_E_STATE);CHECK(!foreign->drain(record).quiescent);
    completion1->dma_done();completion1->readers_done();completion14->dma_done();completion14->readers_done();
    std::vector<float> expected1(256),expected14(256);REQUIRE_EQ(w.source->row_lookup()(1,ids,256,expected1,nullptr),RAD_OK);
    REQUIRE_EQ(w.source->row_lookup()(14,ids,256,expected14,nullptr),RAD_OK);
    const auto actual1=copied(target1,1),actual14=copied(target14,1);bool different=false;
    for(size_t i=0;i<256;++i){CHECK_EQ(actual1[i],expected1[i]);CHECK_EQ(actual14[i],expected14[i]);different=different||actual1[i]!=actual14[i];}
    CHECK(different);CHECK(w.budget->stats().peak<=w.budget->stats().limit);w.backend->progress();
}
TEST(deepseek41_staging_native_allocation_hook_retains_cap_through_failed_physical_free){
    World w;w.backend->retain_host=true;auto pool=w.pool();pool.reset();
    REQUIRE_EQ(w.backend->retained_hosts.size(),1u);CHECK(w.backend->retained_hosts.front()->charge.bytes()>0);
    CHECK(w.budget->stats().used>0);CHECK_EQ(w.backend->live->load(),1);
    w.backend->retained_hosts.clear();CHECK_EQ(w.budget->stats().used,0u);CHECK_EQ(w.backend->live->load(),0);
    w.backend->error_after_host_alloc=true;
    CHECK_EQ(eg::EngramStagingPool::create("stage-model",w.source,w.budget,w.backend,{1,2,256},&pool),RAD_E_IO);
    REQUIRE_EQ(w.backend->retained_hosts.size(),1u);CHECK(w.backend->retained_hosts.front()->charge.bytes()>0);
    CHECK(w.budget->stats().used>0);w.backend->retained_hosts.clear();CHECK_EQ(w.budget->stats().used,0u);CHECK_EQ(w.backend->live->load(),0);
}
TEST(deepseek41_staging_final_backend_owner_destruction_has_no_pending_tracker_records){
    struct FinalOwner:FakeBackend {
        std::shared_ptr<std::atomic<int>> destructions;
        explicit FinalOwner(std::shared_ptr<std::atomic<int>> n):destructions(std::move(n)){}
        ~FinalOwner(){CHECK_EQ(pending_count(),0u);++*destructions;}
    };
    World w;auto destroyed=std::make_shared<std::atomic<int>>(0);auto backend=std::make_shared<FinalOwner>(destroyed);
    std::weak_ptr<FinalOwner> weak=backend;std::shared_ptr<eg::EngramStagingPool> pool;
    REQUIRE_EQ(eg::EngramStagingPool::create("stage-model",w.source,w.budget,backend,{1,2,256},&pool),RAD_OK);
    const std::vector<int64_t> ids{0};eg::RowTransferTicket ticket;REQUIRE_EQ(pool->submit(1,ids,w.target(),&ticket),RAD_OK);
    auto completion=backend->last.lock();REQUIRE(completion);ticket.reset();backend.reset();pool.reset();
    CHECK(!weak.expired());CHECK_EQ(destroyed->load(),0);completion->dma_done();completion->readers_done();
    auto retained=weak.lock();REQUIRE(retained);CHECK_EQ(retained->progress(),0u);retained.reset();
    CHECK(weak.expired());CHECK_EQ(destroyed->load(),1);CHECK_EQ(w.budget->stats().used,0u);
}
RAD_TEST_MAIN()
