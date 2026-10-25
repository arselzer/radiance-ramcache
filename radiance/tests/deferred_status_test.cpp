/* Deferred validation exercised through real Ctx scratch, Rad streams and the
 * same joined model/sampler phase gate used by the opt-in Engine path. HOST only. */
#include "rad_test.h"
#include "runtime/ctx.h"
#include "runtime/rank_barrier.h"
#include "plugin/registry.h"
#include "device/device.h"
#include "rad_deferred_status_capability.h"
#include <atomic>
#include <deque>
#include <functional>
#include <thread>

namespace {
struct Producer {
    RadDeviceStatus source{RAD_OK,0};void* scratch=nullptr;bool original_instance=false,collector_instance=false;int calls=0,collections=0;uint32_t output_marker=0;
};
thread_local Producer* expected_producer=nullptr;
int produce(const RadArgs* args,RadStream stream){
    if(args->instance!=expected_producer)return RAD_E_INVAL;
    auto* state=static_cast<Producer*>(args->instance);if(!state||!args->scratch||args->scratch_bytes<8)return RAD_E_INVAL;
    state->original_instance=true;state->scratch=args->scratch;++state->calls;++state->output_marker;
    return rad_memcpy_async(args->scratch,&state->source,sizeof state->source,stream);
}
int overwrite(const RadArgs* args,RadStream stream){return rad_memset_async(args->scratch,0,8,stream);}
// Single reporter per test pass: copy the ACTUAL queued scratch result before it
// is reused. This intentionally tests core capture/order, not the GPU collector's
// atomic first-error reduction (the actual DS41 collector has its own checker).
int capture(const RadArgs* args,RadDeviceStatus* latch,uint32_t op,RadStream stream){
    if(args->instance!=expected_producer||op!=1)return RAD_E_INVAL;
    auto* state=static_cast<Producer*>(args->instance);state->collector_instance=true;++state->collections;
    return rad_memcpy_async(latch,args->scratch,sizeof(*latch),stream);
}
int sample_copy(const RadArgs* args,RadStream stream){
    if(args->n_t!=2)return RAD_E_SHAPE;
    return rad_memcpy_async(args->t[1].data,args->t[0].data,4,stream);
}
thread_local std::function<void(rad::Ctx&)>* body=nullptr;
void arch_step(RadCtx* ctx,const RadBatch*){if(body)(*body)(*static_cast<rad::Ctx*>(ctx));}
struct Fixture {
    rad::Program program;std::deque<RadKernelInfo> infos;std::deque<rad::KernelRow> rows;
    Producer producer;rad::Ctx ctx;RadBatch batch{};std::function<void(rad::Ctx&)> step;
    explicit Fixture(bool opted=true,int rank=0,int world=1){
        if(!rad::device_is_host())throw std::runtime_error("requires hidden GPUs / HOST backend");
        program.arena_bytes=256;program.scratch_bytes=256;program.weights.resize(1);program.buffers.resize(1);program.ops.resize(1);
        add("produce",produce,&producer,opted?capture:nullptr);add("overwrite",overwrite,nullptr,nullptr);add("sample",sample_copy,nullptr,nullptr);
        rad::CtxDesc d;d.program=&program;d.device=0;d.rank=rank;d.world_size=world;d.arch_step=arch_step;d.record_passes=true;
        if(ctx.init(d)!=RAD_OK)throw std::runtime_error("Ctx status fixture init");
        if(ctx.bind_thread()!=RAD_OK)throw std::runtime_error("Ctx status rank bind");
        batch.phase=RAD_PHASE_DECODE;batch.n_tok=batch.n_seq=1;
        step=[](rad::Ctx& c){c.issue(1,nullptr,0,1);c.issue(2,nullptr,0,1);};
    }
    void add(const char* name,RadLaunchFn launch,void* instance,RadDeferredStatusLaunchFn collector){
        infos.push_back(RadKernelInfo{});auto& info=infos.back();info.name=name;info.op=name;info.family="test";info.domain=RAD_DOMAIN_DEVICE;info.launch=launch;
        rows.push_back(rad::KernelRow{});auto& row=rows.back();row.info=&info;row.plugin="status_capture_test";row.deferred_status=collector;
        rad::OpInfo op;op.op=name;op.index=int32_t(program.ops.size());op.ranged_key="M";op.range_lo=op.range_hi=1;op.base.set_i("M",1);
        rad::Band band;band.hi=1;auto& resolved=band.dom[RAD_DOMAIN_DEVICE];resolved.row=&row;resolved.instance=instance;resolved.geom.set_i("M",1);resolved.scratch_bytes=8;
        op.bands.push_back(band);program.ops.push_back(op);
    }
    int run(){body=&step;expected_producer=&producer;const int status=ctx.run_step(&batch);body=nullptr;expected_producer=nullptr;return status;}
};
struct Gate {
    void* gate=nullptr;
    explicit Gate(RadStream stream){if(rad_stream_gate(stream,&gate)!=RAD_OK)throw std::runtime_error("gate create");}
    void open(){if(gate){const int status=rad_gate_open(gate);gate=nullptr;if(status!=RAD_OK)throw std::runtime_error("gate open");}}
    ~Gate(){if(gate)static_cast<void>(rad_gate_open(gate));}
};
struct TwoRanks {
    enum Phase{model,check,sampler,stop};rad::RankBarrier barrier{3};std::atomic<Phase> phase{model};
    std::atomic<int> combined{RAD_OK},model_enqueued{0},model_done{0},sampler_calls{0};std::atomic<int> published[2]{{-1},{-1}};
    std::thread threads[2];int failing_rank,host_failing_rank;
    explicit TwoRanks(int fail,int host_fail=-1):failing_rank(fail),host_failing_rank(host_fail){
        for(int rank=0;rank<2;++rank)threads[rank]=std::thread([this,rank]{
            std::unique_ptr<Fixture> fixture;std::unique_ptr<Gate> held_work;int initialized=RAD_OK;
            try{fixture=std::make_unique<Fixture>(true,rank,2);fixture->producer.source=rank==failing_rank?RadDeviceStatus{RAD_E_SHAPE,1}:RadDeviceStatus{RAD_OK,0};
                if(rank==host_failing_rank)fixture->step=[](rad::Ctx& c){c.step_fail("synthetic synchronous rank refusal");};
                if(host_failing_rank>=0&&rank!=host_failing_rank)held_work=std::make_unique<Gate>(fixture->ctx.stream());}
            catch(...){initialized=RAD_E_NOMEM;}
            int token=100+rank,published_token=-1;
            for(;;){barrier.arrive_and_wait();const auto selected=phase.load(std::memory_order_acquire);if(selected==stop)break;
                int status=initialized;
                if(selected==model){
                    if(status==RAD_OK)status=fixture->run();
                    ++model_enqueued;
                }else if(selected==check){
                    if(status==RAD_OK)status=fixture->ctx.await_device_status();
                    ++model_done;
                }else{
                    ++sampler_calls;
                    if(status==RAD_OK){const RadOperand operands[]{RAD_P_T(&token,RAD_I32),RAD_P_T(&published_token,RAD_I32)};
                        status=rad_issue(&fixture->ctx,3,operands,2,1);if(status==RAD_OK)status=rad_stream_sync(fixture->ctx.stream());
                        if(status==RAD_OK)published[rank].store(published_token,std::memory_order_release);
                    }
                }
                if(status<0){int expected=RAD_OK;combined.compare_exchange_strong(expected,status);}
                barrier.arrive_and_wait();
            }
        });
    }
    ~TwoRanks(){phase.store(stop,std::memory_order_release);barrier.arrive_and_wait();for(auto& thread:threads)thread.join();}
    int run(){
        return rad::status_checked_model_sampler([&]{phase.store(model,std::memory_order_release);barrier.arrive_and_wait();barrier.arrive_and_wait();return combined.load();},
            [&]{CHECK_EQ(model_enqueued.load(),2);phase.store(check,std::memory_order_release);barrier.arrive_and_wait();barrier.arrive_and_wait();return combined.load();},
            [&]{CHECK_EQ(model_done.load(),2);phase.store(sampler,std::memory_order_release);barrier.arrive_and_wait();barrier.arrive_and_wait();return combined.load();});
    }
};
}
TEST(deferred_status_captures_actual_flag_before_same_scratch_is_overwritten){
    Fixture f;f.producer.source={RAD_E_SHAPE,1};Gate gate(f.ctx.stream());
    REQUIRE_EQ(f.run(),RAD_OK);CHECK(f.ctx.checks_device_status());CHECK(!f.ctx.records_passes());CHECK(f.producer.original_instance);CHECK(f.producer.collector_instance);CHECK_EQ(f.producer.calls,1);CHECK_EQ(f.producer.collections,1);
    gate.open();CHECK_EQ(f.ctx.await_device_status(),RAD_E_SHAPE);CHECK(std::strstr(f.ctx.step_error(),"op 1 (produce)")!=nullptr);
    RadDeviceStatus overwritten{};REQUIRE_EQ(rad_memcpy_async(&overwritten,f.producer.scratch,8,f.ctx.stream()),RAD_OK);REQUIRE_EQ(rad_stream_sync(f.ctx.stream()),RAD_OK);
    CHECK_EQ(overwritten.code,RAD_OK);CHECK_EQ(overwritten.op,0u);
}
TEST(deferred_status_lane1_join_and_existing_lane_reset_preserve_actual_failure){
    Fixture f;f.step=[](rad::Ctx& c){CHECK_EQ(c.set_lane(1),RAD_OK);c.issue(1,nullptr,0,1);c.issue(2,nullptr,0,1);CHECK_EQ(c.set_lane(0),RAD_OK);};
    REQUIRE_EQ(f.run(),RAD_OK);REQUIRE_EQ(f.ctx.await_device_status(),RAD_OK);
    f.producer.source={RAD_E_INVAL,1};++f.batch.step;Gate reset(f.ctx.stream());
    REQUIRE_EQ(f.run(),RAD_OK);reset.open();CHECK_EQ(f.ctx.await_device_status(),RAD_E_INVAL);CHECK(std::strstr(f.ctx.step_error(),"rank 0")!=nullptr);
}
TEST(deferred_status_requires_completed_readback_before_next_pass_and_refuses_speculation){
    Fixture f;REQUIRE_EQ(f.run(),RAD_OK);CHECK_EQ(f.run(),RAD_E_STATE);REQUIRE_EQ(f.ctx.await_device_status(),RAD_OK);
    f.batch.n_spec=1;CHECK_EQ(f.run(),RAD_E_UNSUPPORTED);f.batch.n_spec=0;f.batch.draft_pass=2;CHECK_EQ(f.run(),RAD_E_UNSUPPORTED);f.batch.draft_pass=0;
    REQUIRE_EQ(f.run(),RAD_OK);CHECK_EQ(f.ctx.await_device_status(),RAD_OK);
}
TEST(deferred_status_lifecycle_rejects_opted_issues_before_body_or_collector_can_write){
    for(int phase=0;phase<3;++phase){
        Fixture f;
        if(phase){REQUIRE_EQ(f.run(),RAD_OK);REQUIRE_EQ(rad_stream_sync(f.ctx.stream()),RAD_OK);if(phase==2)REQUIRE_EQ(f.ctx.await_device_status(),RAD_OK);}
        const int calls=f.producer.calls,collections=f.producer.collections;const auto marker=f.producer.output_marker;
        f.producer.source={RAD_E_SHAPE,1};
        CHECK_EQ(f.ctx.issue(1,nullptr,0,1),RAD_E_STATE);
        CHECK_EQ(f.producer.calls,calls);CHECK_EQ(f.producer.collections,collections);CHECK_EQ(f.producer.output_marker,marker);
        if(phase){
            RadDeviceStatus output{999,999};REQUIRE_EQ(rad_memcpy_async(&output,f.producer.scratch,sizeof output,f.ctx.stream()),RAD_OK);REQUIRE_EQ(rad_stream_sync(f.ctx.stream()),RAD_OK);
            CHECK_EQ(output.code,RAD_OK);CHECK_EQ(output.op,0u);if(phase==1)CHECK_EQ(f.ctx.await_device_status(),RAD_OK);
        }else CHECK(f.producer.scratch==nullptr);
    }
}
TEST(deferred_status_rejects_malformed_positive_or_unattributed_payload){
    for(const RadDeviceStatus bad: {RadDeviceStatus{1,1},RadDeviceStatus{RAD_E_SHAPE,0},RadDeviceStatus{RAD_E_SHAPE,999},RadDeviceStatus{RAD_OK,1}}){
        Fixture f;f.producer.source=bad;REQUIRE_EQ(f.run(),RAD_OK);CHECK_EQ(f.ctx.await_device_status(),RAD_E_ABI);
    }
}
TEST(deferred_status_absent_metadata_keeps_original_dispatch_and_needs_no_readback){
    CHECK_EQ(rad_core_deferred_status_version(),RAD_DEFERRED_STATUS_VERSION);
    Fixture f(false);f.producer.source={RAD_E_SHAPE,1};CHECK(!f.ctx.checks_device_status());
    REQUIRE_EQ(f.run(),RAD_OK);CHECK_EQ(f.ctx.await_device_status(),RAD_OK);CHECK(f.producer.original_instance);CHECK_EQ(f.producer.calls,1);
}
TEST(deferred_status_all_rank_model_phase_blocks_every_sampler_when_one_rank_fails){
    TwoRanks ranks(1);CHECK_EQ(ranks.run(),RAD_E_SHAPE);CHECK_EQ(ranks.model_done.load(),2);CHECK_EQ(ranks.sampler_calls.load(),0);
    CHECK_EQ(ranks.published[0].load(),-1);CHECK_EQ(ranks.published[1].load(),-1);
}
TEST(deferred_status_all_rank_model_phase_releases_samplers_only_after_success){
    TwoRanks ranks(-1);CHECK_EQ(ranks.run(),RAD_OK);CHECK_EQ(ranks.model_done.load(),2);CHECK_EQ(ranks.sampler_calls.load(),2);
    CHECK_EQ(ranks.published[0].load(),100);CHECK_EQ(ranks.published[1].load(),101);
}
TEST(deferred_status_joins_host_refusal_before_any_peer_waits_on_held_device_work){
    TwoRanks ranks(-1,0);CHECK_EQ(ranks.run(),RAD_E_SHAPE);
    CHECK_EQ(ranks.model_enqueued.load(),2);CHECK_EQ(ranks.model_done.load(),0);
    CHECK_EQ(ranks.sampler_calls.load(),0);
    CHECK_EQ(ranks.published[0].load(),-1);CHECK_EQ(ranks.published[1].load(),-1);
    // The fixture opens held work before Ctx teardown. Real unmatched
    // collectives have no such cancellation proof: Engine refuses opted TP>1.
}
TEST(deferred_status_loader_requires_complete_supported_pair_and_device_rows){
    for(int variant=0;variant<7;++variant){rad::Registry registry;const auto path=::home()+"/teststatus/status"+std::to_string(variant)+".so";
        const int status=registry.load_plugin(path,0);const bool valid=variant<2;
        CHECK_EQ(status,valid?RAD_OK:RAD_E_ABI);
        if(valid){REQUIRE_EQ(registry.rows().size(),1u);CHECK_EQ(bool(registry.rows()[0].deferred_status),variant==1);}
        else{CHECK(registry.rows().empty());CHECK(registry.plugins().empty());CHECK(registry.schema("status_fixture")==nullptr);}
    }
}
RAD_TEST_MAIN()
