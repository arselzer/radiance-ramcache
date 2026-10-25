/* Core-compatible adapter exercised against real asynchronous HOST Rad streams.
 * No HIP includes, device enumeration or GPU work. */
#include "rad_test.h"
#include "../arch/deepseek41/engram_rad_staging.h"
#include "device/device.h"
#include "runtime/ctx.h"
#include <nlohmann/json.hpp>
#include <atomic>
#include <fstream>
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
}namespace {
struct StreamOwner {
    RadStream stream=nullptr;
    StreamOwner(){if(!rad::device_is_host())throw std::runtime_error("requires hidden GPUs / HOST backend");if(rad_stream_create(&stream,0)!=RAD_OK)throw std::runtime_error("host stream create");}
    ~StreamOwner(){if(stream)rad_stream_destroy(stream);}
};
struct HeapHost:eg::HostRowAllocation {
    std::vector<std::byte> bytes;uint64_t charge;
    HeapHost(size_t n,uint64_t cost):bytes(n),charge(cost){}
    std::span<std::byte> memory()noexcept override{return bytes;}
    uint64_t accounted_bytes()const noexcept override{return charge;}
};
struct Allocator:eg::CheckedRadHostAllocator {
    mutable int charged=0;int allocated=0;
    int allocation_charge(size_t bytes,uint64_t* upper)const noexcept override{++charged;*upper=bytes+128;return RAD_OK;}
    int allocate(size_t bytes,uint64_t maximum,std::shared_ptr<eg::HostRowAllocation>* out)noexcept override{
        ++allocated;try{*out=std::make_shared<HeapHost>(bytes,maximum);return RAD_OK;}catch(...){return RAD_E_NOMEM;}
    }
};
struct Api:eg::RadStagingApi {
    int created=0,destroyed=0,copies=0,records=0,waits=0,queries=0,syncs=0;
    int fail_create_at=0,fail_record_at=0;bool fail_copy=false,fail_query=false,fail_sync=false;
    int create_event(RadEvent* out)noexcept override{++created;const int status=RadStagingApi::create_event(out);return created==fail_create_at?RAD_E_IO:status;}
    void destroy_event(RadEvent event)noexcept override{++destroyed;RadStagingApi::destroy_event(event);}
    int copy(void* out,const void* in,int64_t bytes,RadStream stream)noexcept override{++copies;const int status=RadStagingApi::copy(out,in,bytes,stream);return fail_copy?RAD_E_IO:status;}
    int record(RadEvent event,RadStream stream)noexcept override{++records;const int status=RadStagingApi::record(event,stream);return records==fail_record_at?RAD_E_IO:status;}
    int wait(RadStream stream,RadEvent event)noexcept override{++waits;return RadStagingApi::wait(stream,event);}
    int query(RadEvent event)noexcept override{++queries;return fail_query?RAD_E_DEVICE:RadStagingApi::query(event);}
    int synchronize(RadStream stream)noexcept override{++syncs;return fail_sync?RAD_E_IO:RadStagingApi::synchronize(stream);}
};
struct Arena {std::vector<std::byte> bytes;explicit Arena(size_t n=4096):bytes(n){}};
struct World {
    fixtures::File first,second{true};std::shared_ptr<eg::Iq4NlLayerStores> source;
    std::shared_ptr<StreamOwner> stream=std::make_shared<StreamOwner>();
    std::shared_ptr<Api> api=std::make_shared<Api>();std::shared_ptr<Allocator> allocator=std::make_shared<Allocator>();
    std::shared_ptr<eg::StagingBudget> budget=std::make_shared<eg::StagingBudget>(65536);
    std::shared_ptr<eg::RadEngramBackend> backend;
    explicit World(size_t ops=2){
        if(!rad::device_is_host())throw std::runtime_error("this test requires hidden GPUs / HOST backend");
        auto options=fixtures::options();std::string why;
        if(eg::Iq4NlLayerStores::open("rad-stage",{first.asset(1),second.asset(14)},16384,options,&source,&why)!=RAD_OK)throw std::runtime_error(why);
        if(eg::RadEngramBackend::create_on_rank("rad-stage",{stream->stream,stream,0,101},allocator,{ops},&backend,&why,api)!=RAD_OK)throw std::runtime_error(why);
    }
    ~World(){if(backend)backend->close();}
    std::shared_ptr<eg::EngramStagingPool> pool(size_t slots=1){
        std::shared_ptr<eg::EngramStagingPool> out;std::string why;
        if(eg::EngramStagingPool::create("rad-stage",source,budget,backend,{slots,2,256},&out,&why)!=RAD_OK)throw std::runtime_error(why);
        return out;
    }
    eg::DeviceRowTarget target(int layer=1){
        auto arena=std::make_shared<Arena>();std::shared_ptr<eg::RadArenaRowBuffer> view;
        if(backend->bind_arena(arena->bytes.data(),arena->bytes.size(),arena,101,&view)!=RAD_OK)throw std::runtime_error("arena bind");
        return {view,16,layer==1?first.asset().sha256:second.asset().sha256};
    }
};
struct Gate {
    void* value=nullptr;
    explicit Gate(RadStream stream){if(rad_stream_gate(stream,&value)!=RAD_OK)throw std::runtime_error("host stream gate");}
    void open(){if(value){const int status=rad_gate_open(value);value=nullptr;if(status!=RAD_OK)throw std::runtime_error("gate open");}}
    ~Gate(){if(value)static_cast<void>(rad_gate_open(value));}
};
struct Copy {void* out;size_t bytes;RadStream expected;bool throw_after=false;int calls=0;};
int copy_reader(RadStream stream,const eg::DeviceRowTarget& input,void* user){
    auto& copy=*static_cast<Copy*>(user);++copy.calls;if(stream!=copy.expected)return RAD_E_STATE;
    const int status=rad_memcpy_async(copy.out,reinterpret_cast<const void*>(input.owner->address()+input.offset),int64_t(copy.bytes),stream);
    if(copy.throw_after)throw std::runtime_error("reader failed after submission");
    return status;
}
void compare(const eg::DeviceRowTarget& target,int64_t id){const auto want=fixtures::expected();const auto* got=reinterpret_cast<const float*>(target.owner->address()+target.offset);for(size_t i=0;i<256;++i)CHECK_EQ(got[i],want[size_t(id)*256+i]);}

int issued_copy(const RadArgs* args,RadStream stream){
    if(!args||args->n_t!=2||args->t[0].dtype!=RAD_F32||args->t[1].dtype!=RAD_F32)return RAD_E_SHAPE;
    return rad_memcpy_async(args->t[1].data,args->t[0].data,1024,stream);
}
struct ContextCall {eg::RadEngramBackend* backend;eg::RowTransferTicket* ticket;rad_op op;rad_buf input,output;int calls=0;};
ContextCall* context_call=nullptr;
void context_step(RadCtx* ctx,const RadBatch*){
    auto& call=*context_call;++call.calls;
    int status=call.backend->order_upload(*call.ticket);
    const RadOperand operands[]{RAD_B(call.input),RAD_B(call.output)};
    if(status==RAD_OK)status=rad_issue(ctx,call.op,operands,2,1);
    const int sealed=call.backend->finish_readers(*call.ticket);
    if(status!=RAD_OK||sealed!=RAD_OK)rad_step_fail(ctx,"test Engram handoff refused");
}
struct ContextOwner {
    // The test-only copy kernel is issued by the real Ctx; it uses the actual Rad
    // async copy API and no fake stream or fake rad_issue implementation.
    std::shared_ptr<StreamOwner> stream;rad::Program program;RadKernelInfo info{};rad::KernelRow row;rad::Ctx ctx;
    explicit ContextOwner(std::shared_ptr<StreamOwner> s):stream(std::move(s)){
        program.arena_bytes=4096;program.weights.resize(1);program.buffers.resize(1);program.ops.resize(1);
        for(int i=0;i<2;++i){rad::BufferInfo b;b.name=i?"row_output":"row_input";b.decl.dtype=RAD_F32;b.decl.rank=2;
            b.decl.shape[0]=1;b.decl.shape[1]=256;b.decl.kind=RAD_BUF_PERSIST;b.decl.domain=RAD_DOMAIN_DEVICE;b.bytes=1024;b.arena_offset=i*2048;program.buffers.push_back(b);}
        info.name="test_engram_row_copy";info.op="test_engram_row_copy";info.family="test";info.domain=RAD_DOMAIN_DEVICE;info.launch=issued_copy;
        row.info=&info;row.plugin="test_actual_rad_issue";
        rad::OpInfo op;op.op=info.op;op.index=1;op.ranged_key="M";op.range_lo=op.range_hi=1;op.base.set_i("M",1);
        rad::Band band;band.hi=1;band.dom[RAD_DOMAIN_DEVICE].row=&row;band.dom[RAD_DOMAIN_DEVICE].geom.set_i("M",1);op.bands.push_back(band);program.ops.push_back(op);
        rad::CtxDesc descriptor;descriptor.program=&program;descriptor.device=0;descriptor.stream=stream->stream;descriptor.arch_step=context_step;descriptor.record_passes=false;
        if(ctx.init(descriptor)!=RAD_OK)throw std::runtime_error("actual Ctx init");
        if(ctx.bind_thread()!=RAD_OK)throw std::runtime_error("actual Ctx rank bind");
    }
};
}
TEST(deepseek41_rad_staging_orders_real_host_upload_and_final_reader_before_reuse){
    World w;auto pool=w.pool();auto target=w.target();const std::vector<int64_t> ids{0},next{1};eg::RowTransferTicket ticket,blocked;
    Gate upload(w.stream->stream);REQUIRE_EQ(pool->submit(1,ids,target,&ticket),RAD_OK);CHECK(!ticket.state().quiescent);CHECK_EQ(pool->submit(1,next,w.target(),&blocked),RAD_E_FULL);
    upload.open();REQUIRE_EQ(rad_stream_sync(w.stream->stream),RAD_OK);CHECK(!ticket.state().quiescent);compare(target,0);
    std::vector<float> output(256);Copy copy{output.data(),1024,w.stream->stream};Gate reader(w.stream->stream);
    REQUIRE_EQ(w.backend->consume(ticket,copy_reader,&copy),RAD_OK);CHECK_EQ(copy.calls,1);CHECK(!ticket.state().quiescent);CHECK_EQ(pool->submit(1,next,w.target(),&blocked),RAD_E_FULL);
    reader.open();REQUIRE(w.backend->drain(ticket.backend_record(*w.backend)).quiescent);CHECK_EQ(pool->reap(),0u);
    const auto expected=fixtures::expected();for(size_t i=0;i<256;++i)CHECK_EQ(output[i],expected[i]);
    REQUIRE_EQ(pool->submit(1,next,w.target(),&blocked),RAD_OK);REQUIRE_EQ(w.backend->finish_no_readers(blocked),RAD_OK);CHECK(w.backend->drain(blocked.backend_record(*w.backend)).quiescent);
    CHECK(ticket.state().quiescent);CHECK_EQ(w.backend->native_stats().retired,2u);
}
TEST(deepseek41_rad_staging_callback_exception_seals_partial_reader_and_preserves_error){
    World w;auto pool=w.pool();auto target=w.target();const std::vector<int64_t> ids{1};eg::RowTransferTicket ticket;
    REQUIRE_EQ(pool->submit(1,ids,target,&ticket),RAD_OK);std::vector<float> out(256);Copy copy{out.data(),1024,w.stream->stream,true};
    CHECK_EQ(w.backend->consume(ticket,copy_reader,&copy),RAD_E_STATE);const auto state=w.backend->drain(ticket.backend_record(*w.backend));CHECK(state.quiescent);CHECK_EQ(state.status,RAD_E_STATE);
    const auto expected=fixtures::expected();for(size_t i=0;i<256;++i)CHECK_EQ(out[i],expected[256+i]);
}
TEST(deepseek41_rad_staging_request_cancellation_seals_without_global_cache_shutdown){
    World w;auto pool=w.pool();const std::vector<int64_t> ids{0};eg::RowTransferTicket ticket;std::stop_source stop;
    REQUIRE_EQ(pool->submit(1,ids,w.target(),&ticket,stop.get_token()),RAD_OK);stop.request_stop();ticket.cancel();std::vector<float> out(256);Copy copy{out.data(),1024,w.stream->stream};
    CHECK(!ticket.state().quiescent);CHECK_EQ(w.backend->consume(ticket,copy_reader,&copy),RAD_E_CANCELLED);CHECK_EQ(copy.calls,0);
    const auto drained=w.backend->drain(ticket.backend_record(*w.backend));CHECK(drained.quiescent);CHECK_EQ(drained.status,RAD_E_CANCELLED);
    CHECK_EQ(w.source->row_lookup()(1,ids,256,out,nullptr),RAD_OK);
}
TEST(deepseek41_rad_staging_partial_copy_and_failed_upload_fence_require_sealed_safe_drain){
    for(int failure=0;failure<2;++failure){World w;auto pool=w.pool();auto target=w.target();const std::vector<int64_t> ids{2};eg::RowTransferTicket ticket;
        if(failure==0)w.api->fail_copy=true;else w.api->fail_record_at=1;
        CHECK_EQ(pool->submit(1,ids,target,&ticket),RAD_E_IO);REQUIRE(ticket);CHECK(!ticket.state().quiescent);
        CHECK(!w.backend->drain(ticket.backend_record(*w.backend)).quiescent);CHECK_EQ(w.backend->finish_no_readers(ticket),RAD_E_IO);
        const auto drained=w.backend->drain(ticket.backend_record(*w.backend));CHECK(drained.quiescent);CHECK_EQ(drained.status,RAD_E_IO);compare(target,2);
    }
}
TEST(deepseek41_rad_staging_reader_fence_and_query_sync_errors_retain_until_real_safe_drain){
    for(int failure=0;failure<2;++failure){World w;auto pool=w.pool();const std::vector<int64_t> ids{0};eg::RowTransferTicket ticket;
        REQUIRE_EQ(pool->submit(1,ids,w.target(),&ticket),RAD_OK);if(failure==0)w.api->fail_record_at=2;
        CHECK_EQ(w.backend->finish_no_readers(ticket),failure==0?RAD_E_IO:RAD_OK);
        if(failure==1){w.api->fail_query=true;CHECK(!ticket.state().quiescent);CHECK_EQ(ticket.state().status,RAD_E_DEVICE);}
        w.api->fail_sync=true;CHECK(!w.backend->drain(ticket.backend_record(*w.backend)).quiescent);CHECK_EQ(pool->reap(),1u);
        w.api->fail_sync=false;CHECK(w.backend->drain(ticket.backend_record(*w.backend)).quiescent);CHECK_EQ(pool->reap(),0u);
    }
}
TEST(deepseek41_rad_staging_thread_confinement_and_epoch_issuer_validation){
    World w;auto pool=w.pool();auto target=w.target();const std::vector<int64_t> ids{0};eg::RowTransferTicket ticket;
    REQUIRE_EQ(pool->submit(1,ids,target,&ticket),RAD_OK);const int records=w.api->records,copies=w.api->copies;int consumed=0,closed=0,charged=0;bool observed_safe=true;
    std::thread worker([&]{uint64_t n=0;charged=w.backend->allocation_charge(1,&n);consumed=w.backend->finish_no_readers(ticket);closed=w.backend->close().status;observed_safe=ticket.state().quiescent;});worker.join();
    CHECK_EQ(charged,RAD_E_STATE);CHECK_EQ(consumed,RAD_E_STATE);CHECK_EQ(closed,RAD_E_STATE);CHECK(!observed_safe);CHECK_EQ(w.api->records,records);CHECK_EQ(w.api->copies,copies);
    auto arena=std::make_shared<Arena>();std::shared_ptr<eg::RadArenaRowBuffer> invalid;CHECK_EQ(w.backend->bind_arena(arena->bytes.data(),4096,arena,102,&invalid),RAD_E_SHAPE);
    REQUIRE_EQ(w.backend->order_upload(ticket),RAD_OK);CHECK_EQ(w.backend->finish_no_readers(ticket),RAD_E_STATE);REQUIRE_EQ(w.backend->finish_readers(ticket),RAD_OK);CHECK(w.backend->drain(ticket.backend_record(*w.backend)).quiescent);
    std::shared_ptr<eg::RadEngramBackend> foreign;REQUIRE_EQ(eg::RadEngramBackend::create_on_rank("rad-stage",{w.stream->stream,w.stream,0,101},w.allocator,{1},&foreign,nullptr,w.api),RAD_OK);
    std::shared_ptr<eg::EngramStagingPool> other;REQUIRE_EQ(eg::EngramStagingPool::create("rad-stage",w.source,w.budget,foreign,{1,2,256},&other),RAD_OK);
    eg::RowTransferTicket refused;CHECK_EQ(other->submit(1,ids,target,&refused),RAD_E_SHAPE);CHECK(refused.state().quiescent);CHECK(foreign->close().quiescent);
}
TEST(deepseek41_rad_staging_bounded_event_slots_reuse_latched_generations_across_two_assets){
    World w(1);auto pool=w.pool(2);const std::vector<int64_t> ids{0};eg::RowTransferTicket first,second;auto target=w.target(),other=w.target(14);
    REQUIRE_EQ(pool->submit(1,ids,target,&first),RAD_OK);CHECK_EQ(pool->submit(14,ids,other,&second),RAD_E_FULL);CHECK(second.state().quiescent);CHECK_EQ(w.api->created,2);
    REQUIRE_EQ(w.backend->finish_no_readers(first),RAD_OK);CHECK(w.backend->drain(first.backend_record(*w.backend)).quiescent);pool->reap();
    REQUIRE_EQ(pool->submit(14,ids,other,&second),RAD_OK);CHECK(first.state().quiescent);CHECK(!second.state().quiescent);REQUIRE_EQ(w.backend->finish_no_readers(second),RAD_OK);CHECK(w.backend->drain(second.backend_record(*w.backend)).quiescent);
    std::vector<float> expected(256);REQUIRE_EQ(w.source->row_lookup()(14,ids,256,expected,nullptr),RAD_OK);const auto* got=reinterpret_cast<const float*>(other.owner->address()+other.offset);for(size_t i=0;i<256;++i)CHECK_EQ(got[i],expected[i]);CHECK_EQ(w.api->created,2);
}
TEST(deepseek41_rad_staging_retains_borrowed_stream_and_arena_until_final_reader){
    World w;auto pool=w.pool();auto target=w.target();std::weak_ptr<eg::DeviceRowBuffer> arena=target.owner;std::weak_ptr<StreamOwner> stream=w.stream;
    const std::vector<int64_t> ids{0};eg::RowTransferTicket ticket;REQUIRE_EQ(pool->submit(1,ids,target,&ticket),RAD_OK);
    REQUIRE_EQ(w.backend->finish_no_readers(ticket),RAD_OK);w.api->fail_query=true;w.api->fail_sync=true;target.owner.reset();pool.reset();w.stream.reset();
    CHECK(!arena.expired());CHECK(!stream.expired());CHECK(w.budget->stats().used>0);CHECK(!w.backend->close().quiescent);
    w.api->fail_sync=false;CHECK(w.backend->close().quiescent);ticket.reset();CHECK(arena.expired());CHECK_EQ(w.budget->stats().used,0u);
    w.backend.reset();CHECK(stream.expired());CHECK_EQ(w.api->created,w.api->destroyed);
}
TEST(deepseek41_rad_staging_event_initialization_failure_releases_unqueued_resources){
    if(!rad::device_is_host())SKIP("requires host backend");
    auto stream=std::make_shared<StreamOwner>();auto allocator=std::make_shared<Allocator>();auto api=std::make_shared<Api>();api->fail_create_at=3;
    std::shared_ptr<eg::RadEngramBackend> backend;CHECK_EQ(eg::RadEngramBackend::create_on_rank("rad-stage",{stream->stream,stream,0,101},allocator,{2},&backend,nullptr,api),RAD_E_IO);
    CHECK(!backend);CHECK_EQ(api->created,3);CHECK_EQ(api->destroyed,3);CHECK_EQ(allocator->allocated,0);
}
TEST(deepseek41_rad_staging_real_ctx_rad_issue_reads_upload_and_arena_owner_outlives_pass){
    World w;auto pool=w.pool();auto owner=std::make_shared<ContextOwner>(w.stream);std::weak_ptr<ContextOwner> alive=owner;
    std::shared_ptr<eg::RadArenaRowBuffer> span;void* input=rad_buf_ptr(&owner->ctx,1);
    REQUIRE_EQ(w.backend->bind_arena(input,1024,owner,101,&span),RAD_OK);eg::DeviceRowTarget target{span,0,w.first.asset().sha256};span.reset();
    const std::vector<int64_t> ids{3};eg::RowTransferTicket ticket;Gate gate(w.stream->stream);
    REQUIRE_EQ(pool->submit(1,ids,target,&ticket),RAD_OK);ContextCall call{w.backend.get(),&ticket,1,1,2};context_call=&call;
    RadBatch batch{};batch.phase=RAD_PHASE_DECODE;batch.n_tok=batch.n_seq=1;
    const int status=owner->ctx.run_step(&batch);context_call=nullptr;REQUIRE_EQ(status,RAD_OK);CHECK_EQ(call.calls,1);
    CHECK(!ticket.state().quiescent);owner.reset();CHECK(!alive.expired());gate.open();CHECK(w.backend->drain(ticket.backend_record(*w.backend)).quiescent);
    auto retained=alive.lock();REQUIRE(retained);const auto* result=static_cast<const float*>(rad_buf_ptr(&retained->ctx,2));const auto expected=fixtures::expected();
    for(size_t i=0;i<256;++i)CHECK_EQ(result[i],expected[3*256+i]);
    retained.reset();
    target.owner.reset();pool.reset();ticket.reset();CHECK(alive.expired());CHECK_EQ(w.budget->stats().used,0u);
}
RAD_TEST_MAIN()
