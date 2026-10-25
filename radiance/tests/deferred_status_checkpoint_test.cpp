/* Actual HOST Ctx/streams/events/readback. Not a GPU or collective proof. */
#include "rad_test.h"
#include "runtime/ctx.h"
#include "plugin/registry.h"
#include "device/device.h"
#include <deque>

namespace {
struct Producer {
    RadDeviceStatus source{RAD_OK,0};void* scratch=nullptr;
    int calls=0,collections=0;
};
thread_local std::function<void(rad::Ctx&)>* active=nullptr;
int produce(const RadArgs* a,RadStream stream){
    auto* p=static_cast<Producer*>(a->instance);
    if(!p||!a->scratch||a->scratch_bytes<8)return RAD_E_INVAL;
    ++p->calls;p->scratch=a->scratch;
    return rad_memcpy_async(a->scratch,&p->source,sizeof p->source,stream);
}
int collect(const RadArgs* a,RadDeviceStatus* latch,uint32_t op,RadStream stream){
    auto* p=static_cast<Producer*>(a->instance);
    if(!p||op!=1||!a->scratch||a->scratch_bytes<8)return RAD_E_INVAL;
    ++p->collections;
    // One reporter per negative test: capture real queued scratch before reuse.
    // The production GPU CAS collector is tested by the explicit AQL checker.
    return rad_memcpy_async(latch,a->scratch,sizeof(*latch),stream);
}
int overwrite(const RadArgs* a,RadStream stream){return rad_memset_async(a->scratch,0,8,stream);}
void step(RadCtx* c,const RadBatch*){if(active)(*active)(*static_cast<rad::Ctx*>(c));}
struct Fixture {
    rad::Program program;std::deque<RadKernelInfo> infos;std::deque<rad::KernelRow> rows;
    Producer producer;RadBatch batch{};std::function<void(rad::Ctx&)> body;
    rad::Ctx ctx; // drain before producer/Program/resolved rows disappear
    explicit Fixture(bool opted=true,int world=1){
        if(!rad::device_is_host())throw std::runtime_error("HOST backend required; hide GPUs");
        program.arena_bytes=256;program.scratch_bytes=256;
        program.weights.resize(1);program.buffers.resize(1);program.ops.resize(1);
        add("checkpoint_produce",produce,&producer,opted?collect:nullptr);
        add("checkpoint_overwrite",overwrite,nullptr,nullptr);
        rad::CtxDesc d;d.program=&program;d.device=0;d.rank=0;d.world_size=world;
        d.arch_step=step;d.record_passes=true;
        if(ctx.init(d)!=RAD_OK||ctx.bind_thread()!=RAD_OK)throw std::runtime_error("Ctx init");
        batch.phase=RAD_PHASE_DECODE;batch.n_tok=batch.n_seq=1;
        body=[](rad::Ctx&c){c.issue(1,nullptr,0,1);};
    }
    void add(const char* name,RadLaunchFn launch,void* instance,RadDeferredStatusLaunchFn collector){
        infos.push_back(RadKernelInfo{});auto& info=infos.back();
        info.name=info.op=name;info.family="checkpoint-test";info.domain=RAD_DOMAIN_DEVICE;info.launch=launch;
        rows.push_back(rad::KernelRow{});auto& row=rows.back();row.info=&info;
        row.plugin="checkpoint-host-fixture";row.deferred_status=collector;
        rad::OpInfo op;op.op=name;op.index=int32_t(program.ops.size());op.ranged_key="M";
        op.range_lo=op.range_hi=1;op.base.set_i("M",1);
        rad::Band band;band.hi=1;auto&r=band.dom[RAD_DOMAIN_DEVICE];
        r.row=&row;r.instance=instance;r.geom.set_i("M",1);r.scratch_bytes=8;
        op.bands.push_back(band);program.ops.push_back(op);
    }
    int run(){active=&body;const int status=ctx.run_step(&batch);active=nullptr;return status;}
};
}
TEST(checkpoint_good_repeated_readbacks_preserve_active_pass_and_allow_later_issues){
    Fixture f;int checkpoints=0;
    f.body=[&](rad::Ctx&c){
        for(int i=0;i<6;++i){
            CHECK_EQ(c.issue(1,nullptr,0,1),RAD_OK);
            CHECK_EQ(c.issue(2,nullptr,0,1),RAD_OK); // scratch reuse before the checkpoint
            CHECK_EQ(c.checkpoint_device_status(),RAD_OK);++checkpoints;
        }
        CHECK_EQ(c.issue(1,nullptr,0,1),RAD_OK);
    };
    REQUIRE_EQ(f.run(),RAD_OK);CHECK_EQ(checkpoints,6);CHECK_EQ(f.producer.calls,7);
    CHECK_EQ(f.producer.collections,7);CHECK(!f.ctx.records_passes());
    CHECK_EQ(f.ctx.await_device_status(),RAD_OK);
    ++f.batch.step;REQUIRE_EQ(f.run(),RAD_OK);CHECK_EQ(f.ctx.await_device_status(),RAD_OK);
}
TEST(checkpoint_bad_after_good_preserves_first_attribution_and_never_publishes_or_reuses_ctx){
    Fixture f;int checkpoint=RAD_OK,later=RAD_OK,published=0;std::string first;
    f.body=[&](rad::Ctx&c){
        CHECK_EQ(c.issue(1,nullptr,0,1),RAD_OK);CHECK_EQ(c.checkpoint_device_status(),RAD_OK);
        f.producer.source={RAD_E_SHAPE,1};
        CHECK_EQ(c.issue(1,nullptr,0,1),RAD_OK);CHECK_EQ(c.issue(2,nullptr,0,1),RAD_OK);
        checkpoint=c.checkpoint_device_status();first=c.step_error();
        f.producer.source={RAD_OK,0}; // would reset a wrongly dispatched fixture collector
        later=c.issue(1,nullptr,0,1);CHECK_EQ(c.checkpoint_device_status(),RAD_E_SHAPE);
        CHECK_EQ(std::string(c.step_error()),first);
    };
    const int enqueued=f.run();const int validated=f.ctx.await_device_status();
    if(enqueued>=0&&validated>=0)++published;
    CHECK_EQ(enqueued,RAD_E_SHAPE);CHECK_EQ(checkpoint,RAD_E_SHAPE);CHECK_EQ(later,RAD_E_SHAPE);
    CHECK_EQ(validated,RAD_E_SHAPE);CHECK_EQ(published,0);CHECK_EQ(f.producer.calls,2);
    CHECK_EQ(f.producer.collections,2);CHECK(std::strstr(first.c_str(),"op 1 (checkpoint_produce)")!=nullptr);
    CHECK_EQ(std::string(f.ctx.step_error()),first);
    RadDeviceStatus scratch{99,99};
    REQUIRE_EQ(rad_memcpy_async(&scratch,f.producer.scratch,sizeof scratch,f.ctx.stream()),RAD_OK);
    REQUIRE_EQ(rad_stream_sync(f.ctx.stream()),RAD_OK);CHECK_EQ(scratch.code,0);CHECK_EQ(scratch.op,0u);
    // The negative latch survived actual scratch overwrite and final retirement.
    ++f.batch.step;CHECK_EQ(f.run(),RAD_E_SHAPE);CHECK_EQ(f.producer.calls,2);
    CHECK_EQ(std::string(f.ctx.step_error()),first);
}
TEST(checkpoint_misuse_outside_unsealed_pass_refuses_without_poisoning_next_pass){
    Fixture f;CHECK_EQ(f.ctx.checkpoint_device_status(),RAD_E_STATE);CHECK_EQ(f.producer.calls,0);
    f.body=[](rad::Ctx&c){CHECK_EQ(c.issue(1,nullptr,0,1),RAD_OK);CHECK_EQ(c.checkpoint_device_status(),RAD_OK);};
    REQUIRE_EQ(f.run(),RAD_OK);CHECK_EQ(f.ctx.checkpoint_device_status(),RAD_E_STATE);
    REQUIRE_EQ(f.ctx.await_device_status(),RAD_OK);CHECK_EQ(f.ctx.checkpoint_device_status(),RAD_E_STATE);
    ++f.batch.step;REQUIRE_EQ(f.run(),RAD_OK);CHECK_EQ(f.ctx.await_device_status(),RAD_OK);
}
TEST(checkpoint_refuses_active_or_created_lane1_instead_of_ignoring_its_collectors){
    for(bool joined:{false,true}){
        Fixture f;f.body=[&](rad::Ctx&c){
            CHECK_EQ(c.set_lane(1),RAD_OK);CHECK_EQ(c.issue(1,nullptr,0,1),RAD_OK);
            if(joined){CHECK_EQ(c.lane_join(1,0),RAD_OK);CHECK_EQ(c.set_lane(0),RAD_OK);}
            CHECK_EQ(c.checkpoint_device_status(),RAD_E_UNSUPPORTED);
            CHECK_EQ(c.issue(1,nullptr,0,1),RAD_E_UNSUPPORTED);
        };
        CHECK_EQ(f.run(),RAD_E_UNSUPPORTED);CHECK_EQ(f.ctx.await_device_status(),RAD_E_UNSUPPORTED);
        CHECK_EQ(f.producer.calls,1);CHECK(std::strstr(f.ctx.step_error(),"no lane1")!=nullptr);
    }
}
TEST(checkpoint_logical_tp2_is_explicitly_unsupported_without_any_collective){
    Fixture f(true,2);f.body=[](rad::Ctx&c){CHECK_EQ(c.checkpoint_device_status(),RAD_E_UNSUPPORTED);};
    CHECK_EQ(f.run(),RAD_E_UNSUPPORTED);CHECK_EQ(f.ctx.await_device_status(),RAD_E_UNSUPPORTED);
    CHECK_EQ(f.producer.calls,0);
}
TEST(checkpoint_preserves_an_earlier_synchronous_failure_including_its_diagnostic){
    Fixture f;f.body=[](rad::Ctx&c){
        CHECK_EQ(c.step_fail("earlier synchronous fixture error"),RAD_E_SHAPE);
        CHECK_EQ(c.checkpoint_device_status(),RAD_E_SHAPE);CHECK_EQ(c.issue(1,nullptr,0,1),RAD_E_SHAPE);
    };
    CHECK_EQ(f.run(),RAD_E_SHAPE);CHECK_EQ(f.ctx.await_device_status(),RAD_E_SHAPE);
    CHECK_EQ(std::string(f.ctx.step_error()),"earlier synchronous fixture error");CHECK_EQ(f.producer.calls,0);
}
TEST(checkpoint_absent_metadata_refuses_without_adding_a_validation_claim_or_default_behavior){
    Fixture f(false);CHECK_EQ(f.ctx.checkpoint_device_status(),RAD_E_UNSUPPORTED);
    f.body=[](rad::Ctx&c){CHECK_EQ(c.issue(1,nullptr,0,1),RAD_OK);CHECK_EQ(c.checkpoint_device_status(),RAD_E_UNSUPPORTED);};
    CHECK(!f.ctx.checks_device_status());REQUIRE_EQ(f.run(),RAD_OK);CHECK_EQ(f.ctx.await_device_status(),RAD_OK);
    CHECK_EQ(f.producer.calls,1);
}
TEST(checkpoint_malformed_payload_is_fatal_without_fabricating_an_op_attribution){
    for(const RadDeviceStatus bad:{RadDeviceStatus{1,1},RadDeviceStatus{RAD_E_SHAPE,0},
                                  RadDeviceStatus{RAD_E_SHAPE,999},RadDeviceStatus{RAD_OK,1}}){
        Fixture f;f.producer.source=bad;f.body=[](rad::Ctx&c){
            CHECK_EQ(c.issue(1,nullptr,0,1),RAD_OK);CHECK_EQ(c.checkpoint_device_status(),RAD_E_ABI);
            CHECK_EQ(c.issue(1,nullptr,0,1),RAD_E_ABI);
        };
        CHECK_EQ(f.run(),RAD_E_ABI);CHECK_EQ(f.ctx.await_device_status(),RAD_E_ABI);
        CHECK(std::strstr(f.ctx.step_error(),"op 0 (?)")!=nullptr);CHECK_EQ(f.producer.calls,1);
    }
}
TEST(checkpoint_primitive_physically_preserves_sticky_latch_after_negative_readback){
    REQUIRE(rad::device_is_host());RadStream stream=nullptr;REQUIRE_EQ(rad_stream_create(&stream,0),RAD_OK);
    {
        rad::DeferredStatus status;REQUIRE_EQ(status.init(),RAD_OK);REQUIRE_EQ(status.begin(stream,nullptr),RAD_OK);
        const RadDeviceStatus source{RAD_E_INVAL,1};
        REQUIRE_EQ(rad_memcpy_async(status.device(),&source,sizeof source,stream),RAD_OK);
        CHECK_EQ(status.checkpoint(2),RAD_E_INVAL);CHECK(status.pending());CHECK(status.collecting());
        CHECK(status.checkpoint_failed());CHECK(!status.accepts_issue());
        RadDeviceStatus physical{};REQUIRE_EQ(rad_memcpy_async(&physical,status.device(),sizeof physical,stream),RAD_OK);
        REQUIRE_EQ(rad_stream_sync(stream),RAD_OK);CHECK_EQ(physical.code,source.code);CHECK_EQ(physical.op,source.op);
        CHECK_EQ(status.checkpoint(2),RAD_E_INVAL);REQUIRE_EQ(status.seal(),RAD_OK);CHECK_EQ(status.await(2),RAD_E_INVAL);
        CHECK(!status.pending());CHECK_EQ(status.last().op,1u);CHECK_EQ(status.begin(stream,nullptr),RAD_E_STATE);
    }
    rad_stream_destroy(stream);
}
RAD_TEST_MAIN()
