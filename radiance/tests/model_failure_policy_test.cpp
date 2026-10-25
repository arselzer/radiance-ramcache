/* Real HOST Ctx queues/events/scratch + actual Engine admission/prepare methods.
 * This is not a GPU execution, cancellation or failed-drain lifetime proof. */
#include "rad_test.h"
#include "rad_model_failure_policy.h"
#include "engine_priv.h"
#include "runtime/model_failure_policy.h"
#include "device/device.h"
#include "util/stallwatch.h"
#include <deque>
#include <functional>
#include <sys/wait.h>
#include <unistd.h>

namespace rad {
struct ModelFailurePolicyTestAccess {
    static Ctx& add(Engine& engine,Program program,ArchStepFn step) {
        auto rank=std::make_unique<Engine::Rank>();
        rank->program=std::move(program);
        CtxDesc d;d.program=&rank->program;d.world_size=1;d.arch_step=step;
        if (rank->ctx.init(d) != RAD_OK) throw std::runtime_error("Engine fixture Ctx init");
        engine.ranks_.push_back(std::move(rank));
        return engine.ranks_.back()->ctx;
    }
    static int admission(const Engine& e) { return e.model_policy_admission(); }
    static bool profile(const Engine& e) { return e.requires_checked_model_profile(); }
    static int prepare(Engine& e,const RadBatch* b) { return e.prepare_release(b); }
    static int encodes(Engine& e) { return e.run_encodes(); }
    static int gen(Engine& e) { return e.gen_loop(); }
};
}
namespace {
rad::Program program(bool strict) {
    rad::Registry r;RadModelMeta m{};RadBuildCtx b{};b.world_size=1;
    rad::Builder declare(r,m,b);
    if (strict && rad_require_permanent_model_failure(&declare) != RAD_OK)
        throw std::runtime_error("policy declaration");
    if (declare.finish() != RAD_OK) throw std::runtime_error("finish empty fixture");
    return std::move(declare.program());
}
struct Producer {
    RadDeviceStatus value{RAD_OK,0};RadDeviceStatus* latch=nullptr;void* scratch=nullptr;
    int calls=0,collections=0,overwrites=0,fail_calls=0,host_calls=0;
    uint32_t persistent=0;
};
int produce(const RadArgs* a,RadStream s) {
    auto* p=static_cast<Producer*>(a->instance);
    if (!p || !a->scratch || a->scratch_bytes<8) return RAD_E_INVAL;
    ++p->calls;p->scratch=a->scratch;
    return rad_memcpy_async(a->scratch,&p->value,8,s);
}
int capture(const RadArgs* a,RadDeviceStatus* latch,uint32_t op,RadStream s) {
    auto* p=static_cast<Producer*>(a->instance);
    if (!p || op!=1) return RAD_E_INVAL;
    ++p->collections;p->latch=latch;
    return rad_memcpy_async(latch,a->scratch,8,s);
}
int overwrite(const RadArgs* a,RadStream s) {
    auto* p=static_cast<Producer*>(a->instance);++p->overwrites;
    return rad_memset_async(a->scratch,0,8,s);
}
int refuse(const RadArgs* a,RadStream) {
    ++static_cast<Producer*>(a->instance)->fail_calls;return RAD_E_INVAL;
}
int host_mutate(const RadArgs* a,RadStream) {
    auto* p=static_cast<Producer*>(a->instance);++p->host_calls;++p->persistent;return RAD_OK;
}
thread_local std::function<void(rad::Ctx&)>* body=nullptr;
void step(RadCtx* c,const RadBatch*) { if (body) (*body)(*static_cast<rad::Ctx*>(c)); }
struct CountingResidency : rad::ResidencyTable {
    mutable int lookups=0,epochs=0,reads=0;int clears=0;std::vector<rad_weight> changed;
    const rad::WeightSlot* lookup(rad_weight) const override {++lookups;return nullptr;}
    uint64_t epoch() const override {++epochs;return 1;}
    const std::vector<rad_weight>& changes() const override {++reads;return changed;}
    void clear_changes() override {++clears;changed.clear();}
};
struct CountingStager : rad::OpStager {
    int begins=0,befores=0,ends=0;
    bool begin_pass(const RadBatch*) override {++begins;return true;}
    int before_op(rad_op,RadStream,bool* resync) override {++befores;*resync=false;return RAD_OK;}
    int end_pass(RadStream) override {++ends;return RAD_OK;}
};
struct Fixture {
    rad::Program p;std::deque<RadKernelInfo> infos;std::deque<rad::KernelRow> rows;
    Producer producer;CountingResidency residency;CountingStager stager;
    RadBatch batch{};std::function<void(rad::Ctx&)> callback;
    rad::Ctx ctx; // resources drain while source/instances/metadata are still alive
    Fixture(bool strict=true,bool collector=true,bool stager_on=false) : p(program(strict)) {
        if (!rad::device_is_host()) throw std::runtime_error("HOST backend required; hide GPUs");
        p.arena_bytes=p.scratch_bytes=256;
        add("policy_produce",produce,collector?capture:nullptr,RAD_DOMAIN_DEVICE);
        add("policy_overwrite",overwrite,nullptr,RAD_DOMAIN_DEVICE);
        add("policy_refuse",refuse,nullptr,RAD_DOMAIN_HOST);
        add("policy_host_mutate",host_mutate,nullptr,RAD_DOMAIN_HOST);
        rad::CtxDesc d;d.program=&p;d.arch_step=step;d.world_size=1;d.residency=&residency;
        if (ctx.init(d)!=RAD_OK) throw std::runtime_error("fixture Ctx init");
        if (stager_on) ctx.set_stager(&stager);
        batch.phase=RAD_PHASE_DECODE;batch.n_seq=batch.n_tok=1;
        callback=[](rad::Ctx& c){c.issue(1,nullptr,0,1);c.issue(2,nullptr,0,1);};
    }
    void add(const char* name,RadLaunchFn launch,RadDeferredStatusLaunchFn collector,int domain) {
        infos.emplace_back();auto& info=infos.back();info.name=info.op=name;
        info.family="policy-fixture";info.domain=domain;info.launch=launch;
        rows.emplace_back();auto& row=rows.back();row.info=&info;row.plugin="policy-fixture";
        row.deferred_status=collector;
        rad::OpInfo op;op.op=name;op.index=(int32_t)p.ops.size();op.ranged_key="M";
        op.range_lo=op.range_hi=1;op.base.set_i("M",1);rad::Band band;band.hi=1;
        auto& selected=band.dom[domain];selected.row=&row;selected.instance=&producer;
        selected.geom.set_i("M",1);selected.scratch_bytes=8;op.bands.push_back(band);p.ops.push_back(op);
    }
    int run() {body=&callback;const int s=ctx.run_step(&batch);body=nullptr;return s;}
};
RadDeviceStatus read_status(Fixture& f,const void* src) {
    RadDeviceStatus value{99,99};
    if (rad_memcpy_async(&value,src,8,f.ctx.stream())!=RAD_OK || rad_stream_sync(f.ctx.stream())!=RAD_OK)
        throw std::runtime_error("HOST status readback");
    return value;
}
int run_probe(const char* executable,const std::string& module) {
    const pid_t child=fork();if(child<0)return -1;
    if(child==0){execl(executable,executable,module.c_str(),(char*)nullptr);_exit(127);}
    int status=0;if(waitpid(child,&status,0)!=child)return -1;
    return WIFEXITED(status)?WEXITSTATUS(status):-1;
}
}
// Controlled return injection around actual HOST API calls. Every noninjected
// call reaches the unchanged Core dispatcher/backend; row launch is untouched.
namespace policy_fault {
enum class Mode {none,copy,record,sync};
thread_local Mode armed=Mode::none;
thread_local int hits=0;
bool hit(Mode m) {if(armed!=m)return false;armed=Mode::none;++hits;return true;}
void arm(Mode m) {armed=m;hits=0;}
}
extern "C" {
int __real_rad_memcpy_async(void*,const void*,int64_t,RadStream);
int __real_rad_event_record(RadEvent,RadStream);
int __real_rad_event_sync(RadEvent);
int __wrap_rad_memcpy_async(void* d,const void* s,int64_t n,RadStream stream) {
    return policy_fault::hit(policy_fault::Mode::copy)?RAD_E_DEVICE:__real_rad_memcpy_async(d,s,n,stream);
}
int __wrap_rad_event_record(RadEvent e,RadStream stream) {
    return policy_fault::hit(policy_fault::Mode::record)?RAD_E_DEVICE:__real_rad_event_record(e,stream);
}
int __wrap_rad_event_sync(RadEvent e) {
    return policy_fault::hit(policy_fault::Mode::sync)?RAD_E_DEVICE:__real_rad_event_sync(e);
}
}
TEST(policy_default_legacy_final_and_sync_failures_can_still_reset) {
    Fixture f(false);f.producer.value={RAD_E_SHAPE,1};
    REQUIRE_EQ(f.run(),RAD_OK);CHECK_EQ(f.ctx.await_device_status(),RAD_E_SHAPE);
    CHECK_EQ(f.ctx.terminal_status(),RAD_OK);f.producer.value={RAD_OK,0};
    REQUIRE_EQ(f.run(),RAD_OK);CHECK_EQ(f.ctx.await_device_status(),RAD_OK);
    f.callback=[](rad::Ctx& c){c.issue(3,nullptr,0,1);};
    CHECK_EQ(f.run(),RAD_E_INVAL);CHECK_EQ(f.ctx.await_device_status(),RAD_OK);
    f.callback=[](rad::Ctx& c){c.issue(1,nullptr,0,1);};
    REQUIRE_EQ(f.run(),RAD_OK);CHECK_EQ(f.ctx.await_device_status(),RAD_OK);
}
TEST(policy_good_status_passes_and_host_only_passes_remain_reusable) {
    for (bool collector:{false,true}) {
        Fixture f(true,collector);CHECK(!f.ctx.records_passes());
        f.callback=[](rad::Ctx& c){c.issue(4,nullptr,0,1);};
        for (int i=0;i<3;++i) {++f.batch.step;REQUIRE_EQ(f.run(),RAD_OK);CHECK_EQ(f.ctx.await_device_status(),RAD_OK);}
        CHECK_EQ(f.producer.host_calls,3);CHECK_EQ(f.ctx.terminal_status(),RAD_OK);
    }
}
TEST(policy_final_flag_survives_scratch_overwrite_and_denies_reset_before_any_preparation) {
    Fixture f(true,true,true);f.producer.value={RAD_E_SHAPE,1};
    REQUIRE_EQ(f.run(),RAD_OK);REQUIRE_EQ(f.ctx.await_device_status(),RAD_E_SHAPE);
    CHECK_EQ(read_status(f,f.producer.scratch).code,0); // actual scratch has already been reused
    REQUIRE(f.producer.latch!=nullptr);CHECK_EQ(read_status(f,f.producer.latch).code,RAD_E_SHAPE);
    const std::string diagnostic=f.ctx.step_error();
    const int begins=f.stager.begins,ends=f.stager.ends,reads=f.residency.reads,clears=f.residency.clears;
    const int calls=f.producer.calls,collections=f.producer.collections;
    f.producer.value={RAD_OK,0};++f.batch.step;
    CHECK_EQ(f.run(),RAD_E_SHAPE);CHECK_EQ(f.ctx.issue(1,nullptr,0,1),RAD_E_SHAPE);
    CHECK_EQ(f.ctx.step_fail("must not replace the first error"),RAD_E_SHAPE);
    CHECK_EQ(f.ctx.run_step(nullptr),RAD_E_SHAPE);CHECK_EQ(f.ctx.set_lane(1),RAD_E_SHAPE);
    CHECK_EQ(f.ctx.await_device_status(),RAD_E_SHAPE);CHECK_EQ(std::string(f.ctx.step_error()),diagnostic);
    CHECK_EQ(f.stager.begins,begins);CHECK_EQ(f.stager.ends,ends);
    CHECK_EQ(f.residency.reads,reads);CHECK_EQ(f.residency.clears,clears);
    CHECK_EQ(f.producer.calls,calls);CHECK_EQ(f.producer.collections,collections);
    CHECK_EQ(read_status(f,f.producer.latch).code,RAD_E_SHAPE); // denied begin did not reset latch
}
TEST(policy_synchronous_late_failure_is_terminal_without_rolling_back_prior_host_state) {
    Fixture f(true,false,true);
    f.callback=[](rad::Ctx& c){c.issue(4,nullptr,0,1);c.issue(3,nullptr,0,1);c.issue(4,nullptr,0,1);};
    CHECK_EQ(f.run(),RAD_E_INVAL);CHECK_EQ(f.ctx.await_device_status(),RAD_E_INVAL);
    CHECK_EQ(f.producer.persistent,1u);CHECK_EQ(f.producer.host_calls,1);CHECK_EQ(f.stager.ends,1);
    CHECK_EQ(f.run(),RAD_E_INVAL);CHECK_EQ(f.producer.persistent,1u);
}
TEST(policy_checkpoint_failure_and_later_final_failure_both_become_permanent) {
    for (bool fail_at_checkpoint:{false,true}) {
        Fixture f;f.callback=[&](rad::Ctx& c){
            CHECK_EQ(c.issue(1,nullptr,0,1),RAD_OK);CHECK_EQ(c.checkpoint_device_status(),RAD_OK);
            f.producer.value={RAD_E_SHAPE,1};
            CHECK_EQ(c.issue(1,nullptr,0,1),RAD_OK);CHECK_EQ(c.issue(2,nullptr,0,1),RAD_OK);
            if (fail_at_checkpoint) {
                CHECK_EQ(c.checkpoint_device_status(),RAD_E_SHAPE);
                CHECK_EQ(c.issue(1,nullptr,0,1),RAD_E_SHAPE);
            }
        };
        CHECK_EQ(f.run(),fail_at_checkpoint?RAD_E_SHAPE:RAD_OK);
        REQUIRE_EQ(f.ctx.await_device_status(),RAD_E_SHAPE);
        const auto diagnostic=std::string(f.ctx.step_error());CHECK_EQ(f.run(),RAD_E_SHAPE);
        CHECK_EQ(std::string(f.ctx.step_error()),diagnostic);CHECK_EQ(f.producer.calls,2);
    }
}
TEST(policy_malformed_readback_refuses_without_invented_attribution) {
    for (auto invalid:{RadDeviceStatus{1,1},RadDeviceStatus{RAD_E_SHAPE,0},RadDeviceStatus{RAD_OK,1}}) {
        Fixture f;f.producer.value=invalid;REQUIRE_EQ(f.run(),RAD_OK);
        CHECK_EQ(f.ctx.await_device_status(),RAD_E_ABI);CHECK_EQ(f.run(),RAD_E_ABI);
        CHECK(std::strstr(f.ctx.step_error(),"op 0 (?)")!=nullptr);
    }
}
TEST(policy_failed_ctx_shutdown_does_not_allow_reinit_or_lose_owned_diagnostic) {
    Fixture f(true,false);f.callback=[](rad::Ctx& c){c.step_fail("first owned failure");};
    REQUIRE_EQ(f.run(),RAD_E_SHAPE);f.ctx.shutdown();
    auto fresh=program(false);rad::CtxDesc d;d.program=&fresh;d.world_size=1;d.arch_step=step;
    CHECK_EQ(f.ctx.init(d),RAD_E_SHAPE);CHECK(f.ctx.stream()==nullptr);CHECK(f.ctx.program()==nullptr);
    CHECK_EQ(std::string(f.ctx.step_error()),"first owned failure");CHECK_EQ(f.ctx.await_device_status(),RAD_E_SHAPE);
    rad::Ctx new_owner;CHECK_EQ(new_owner.init(d),RAD_OK); // no claim this resets external old state
}
TEST(policy_trivial_null_admission_does_not_claim_a_model_pass_was_mutated) {
    Fixture f;CHECK_EQ(f.ctx.run_step(nullptr),RAD_E_INVAL);CHECK_EQ(f.ctx.terminal_status(),RAD_OK);
    CHECK_EQ(f.ctx.checkpoint_device_status(),RAD_E_STATE);REQUIRE_EQ(f.run(),RAD_OK);CHECK_EQ(f.ctx.await_device_status(),RAD_OK);
}
TEST(policy_checkpoint_transport_return_refusal_allows_retirement_but_never_reuse) {
    for (auto mode:{policy_fault::Mode::copy,policy_fault::Mode::record,policy_fault::Mode::sync}) {
        Fixture f;f.callback=[&](rad::Ctx& c){
            CHECK_EQ(c.issue(1,nullptr,0,1),RAD_OK);
            CHECK_EQ(c.checkpoint_device_status(),RAD_OK);
            policy_fault::arm(mode);
            CHECK_EQ(c.checkpoint_device_status(),RAD_E_DEVICE);
            CHECK_EQ(policy_fault::hits,1);
            CHECK_EQ(c.issue(1,nullptr,0,1),RAD_E_DEVICE);
        };
        REQUIRE_EQ(f.run(),RAD_E_DEVICE);const std::string diagnostic=f.ctx.step_error();
        CHECK_EQ(f.ctx.await_device_status(),RAD_E_DEVICE); // final real event/stream drain succeeds
        CHECK_EQ(f.ctx.await_device_status(),RAD_E_DEVICE);CHECK_EQ(f.run(),RAD_E_DEVICE);
        CHECK_EQ(f.producer.calls,1);CHECK_EQ(std::string(f.ctx.step_error()),diagnostic);
    }
}
TEST(policy_final_await_return_refusal_remains_first_error_after_later_successful_retirement) {
    Fixture f;REQUIRE_EQ(f.run(),RAD_OK);policy_fault::arm(policy_fault::Mode::sync);
    REQUIRE_EQ(f.ctx.await_device_status(),RAD_E_DEVICE);CHECK_EQ(policy_fault::hits,1);
    const std::string diagnostic=f.ctx.step_error();
    // The first failed sync did not prove quiescence; retry only the owned final
    // retirement. The injected HOST return is not an actual failed GPU queue.
    CHECK_EQ(f.ctx.await_device_status(),RAD_E_DEVICE);CHECK_EQ(f.run(),RAD_E_DEVICE);
    CHECK_EQ(std::string(f.ctx.step_error()),diagnostic);CHECK_EQ(f.producer.calls,1);
}
TEST(policy_final_seal_copy_or_record_refusal_does_not_authorize_a_new_pass) {
    for (auto mode:{policy_fault::Mode::copy,policy_fault::Mode::record}) {
        Fixture f;f.callback=[&](rad::Ctx& c){
            CHECK_EQ(c.issue(1,nullptr,0,1),RAD_OK);policy_fault::arm(mode);
        };
        CHECK_EQ(f.run(),RAD_E_DEVICE);CHECK_EQ(policy_fault::hits,1);
        CHECK_EQ(f.ctx.await_device_status(),RAD_E_DEVICE);CHECK_EQ(f.run(),RAD_E_DEVICE);
        CHECK_EQ(f.producer.calls,1);
    }
}
TEST(policy_builder_requirement_and_sizing_cannot_be_downgraded) {
    rad::Registry registry;RadModelMeta meta{};RadBuildCtx context{};context.world_size=1;
    rad::Builder full(registry,meta,context);CHECK_EQ(rad_require_permanent_model_failure(nullptr),RAD_E_INVAL);
    REQUIRE_EQ(rad_require_permanent_model_failure(&full),RAD_OK);CHECK_EQ(rad_require_permanent_model_failure(&full),RAD_OK);
    REQUIRE_EQ(full.finish(),RAD_OK);CHECK(full.program().requires_permanent_model_failure());
    CHECK_EQ(rad_require_permanent_model_failure(&full),RAD_E_STATE);
    rad::Builder bad(registry,meta,context);bad.set_reference(&full.program());CHECK_EQ(bad.finish(),RAD_E_STATE);
    rad::Builder good(registry,meta,context);good.set_reference(&full.program());
    REQUIRE_EQ(rad_require_permanent_model_failure(&good),RAD_OK);CHECK_EQ(good.finish(),RAD_OK);
    rad::ArenaLevel output;output.rows=123;std::string why;auto legacy=program(false);
    CHECK_EQ(rad::plan_level(full.program(),legacy,1,&output,&why),RAD_E_STATE);CHECK_EQ(output.rows,123);
    CHECK(why.find("policy")!=std::string::npos);
}
TEST(policy_actual_engine_zero_collector_host_profile_and_failure_admission) {
    rad::Config config;config.tp=1;config.n_spec=0;rad::Engine engine(config);
    auto& ctx=rad::ModelFailurePolicyTestAccess::add(engine,program(true),step);
    CHECK(!ctx.checks_device_status());CHECK(rad::ModelFailurePolicyTestAccess::profile(engine));
    CHECK_EQ(rad::ModelFailurePolicyTestAccess::admission(engine),RAD_OK);
    RadBatch batch{};batch.phase=RAD_PHASE_DECODE;batch.n_tok=batch.n_seq=1;
    std::function<void(rad::Ctx&)> fail=[](rad::Ctx& c){c.step_fail("Engine owned HOST failure");};
    body=&fail;CHECK_EQ(ctx.run_step(&batch),RAD_E_SHAPE);body=nullptr;
    CHECK_EQ(rad::ModelFailurePolicyTestAccess::admission(engine),RAD_E_SHAPE);
    // Real Engine methods must return before touching unconfigured scheduler,
    // arenas, Gen driver or any sampler. No launch pointer has been replaced.
    CHECK_EQ(rad::ModelFailurePolicyTestAccess::prepare(engine,&batch),RAD_E_SHAPE);
    CHECK_EQ(rad::ModelFailurePolicyTestAccess::encodes(engine),RAD_E_SHAPE);
    CHECK_EQ(rad::ModelFailurePolicyTestAccess::gen(engine),RAD_E_SHAPE);
    CHECK_EQ(engine.serve(nullptr),RAD_E_SHAPE);CHECK_EQ(std::string(ctx.step_error()),"Engine owned HOST failure");
}
TEST(policy_actual_engine_rejects_unintegrated_profiles_before_serving) {
    for (int mode=0;mode<4;++mode) {
        rad::Config config;config.tp=1;config.n_spec=mode==0?1:0;rad::Engine engine(config);
        auto p=program(true);
        if (mode==1) p.encoder.modalities=1;
        if (mode==2) p.audio_gen.max_lanes=1;
        if (mode==3) p.generator.layout=1;
        rad::ModelFailurePolicyTestAccess::add(engine,std::move(p),step);
        CHECK_EQ(rad::ModelFailurePolicyTestAccess::admission(engine),RAD_E_UNSUPPORTED);
        CHECK_EQ(engine.serve(nullptr),RAD_E_UNSUPPORTED);
    }
}
TEST(policy_engine_protocol_no_sampler_or_publication_after_host_or_async_failure) {
    for (bool asynchronous:{false,true}) {
        Fixture f(true,asynchronous);int sampled=0,published=0;
        if (asynchronous) f.producer.value={RAD_E_SHAPE,1};
        else f.callback=[](rad::Ctx& c){c.issue(4,nullptr,0,1);c.step_fail("HOST synchronous refusal");};
        CHECK(rad::checked_model_profile(f.p,f.ctx.checks_device_status()));
        const int status=rad::status_checked_model_sampler([&]{return f.run();},
            [&]{return f.ctx.await_device_status();},[&]{++sampled;return RAD_OK;});
        if (status>=0) ++published;
        CHECK_EQ(status,RAD_E_SHAPE);CHECK_EQ(sampled,0);CHECK_EQ(published,0);
        CHECK_EQ(f.ctx.await_device_status(),RAD_E_SHAPE); // retirement is still allowed
    }
}
TEST(policy_profile_tp1_and_spec_guard_is_not_a_collective_cancellation_claim) {
    auto strict=program(true),legacy=program(false);
    CHECK_EQ(rad::model_profile_admission(strict,false,2,0),RAD_E_UNSUPPORTED);
    CHECK_EQ(rad::model_profile_admission(strict,false,1,1),RAD_E_UNSUPPORTED);
    CHECK_EQ(rad::model_profile_admission(legacy,false,4,3),RAD_OK);
    CHECK_EQ(rad::model_profile_admission(legacy,true,2,0),RAD_E_UNSUPPORTED);
    Fixture f(true,false);f.batch.n_spec=1;CHECK_EQ(f.run(),RAD_E_UNSUPPORTED);
    f.batch.n_spec=0;CHECK_EQ(f.run(),RAD_E_UNSUPPORTED);CHECK_EQ(f.producer.calls,0);
}
TEST(policy_loader_new_core_and_real_decl_requirement_old_cores_fail_closed) {
    const auto module=::home()+"/testpolicy/policy.so";
    CHECK_EQ(rad_core_model_failure_policy_version(),RAD_MODEL_FAILURE_POLICY_VERSION);
    CHECK_EQ(run_probe(RAD_POLICY_MISSING_PROBE,module),0);
    CHECK_EQ(run_probe(RAD_POLICY_STALE_PROBE,module),0);
    rad::Registry registry;REQUIRE_EQ(registry.load_plugin(module,0),RAD_OK);
    REQUIRE_EQ(registry.plugins().size(),1u);const auto& plugin=registry.plugins()[0];
    RadModelMeta meta{};RadBuildCtx context{};context.world_size=1;
    rad::Builder b(registry,meta,context);REQUIRE(plugin.arch_declare!=nullptr);
    REQUIRE_EQ(plugin.arch_declare(&b,&meta,&context),RAD_OK);REQUIRE_EQ(b.finish(),RAD_OK);
    CHECK(b.program().requires_permanent_model_failure());
}
RAD_TEST_MAIN()
