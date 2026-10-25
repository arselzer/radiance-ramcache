/* EXPLICIT MANUAL one-GPU checker for the checkpoint overlay. No automatic CTest. */
#include "build/rad_build.h"
#include "runtime/ctx.h"
#include "device/device.h"
#include "../libds41/ds41_route_jobs_fixture.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace rad;
namespace route=rad::ds41::route_jobs;
namespace {
constexpr int64_t guard=64,limit=1<<20;
int64_t live=0,peak=0;
void require(bool b,const std::string& why){if(!b)throw std::runtime_error(why);}
void check(int s,const char* why){if(s<0)throw std::runtime_error(std::string(why)+": "+rad_strerror(s));}
void charge(int64_t n){require(n>0&&n<=limit&&live<=limit-n,"explicit tiny byte cap");live+=n;peak=std::max(peak,live);}
struct Stream {
    RadStream s=nullptr;Stream(){check(rad_stream_create(&s,0),"create actual core stream");}
    ~Stream(){if(s)rad_stream_destroy(s);}
};
struct Owned {
    RadStream stream;int64_t bytes;int kind;void* allocation=nullptr;void* device=nullptr;void* host=nullptr;
    Owned(RadStream s,int64_t n,int k):stream(s),bytes(n),kind(k){
        charge(n);allocation=rad_dev_alloc(n,k);
        if(!allocation){live-=n;throw std::runtime_error("allocation");}
        device=k==RAD_MEM_DEVICE?allocation:rad_dev_device_ptr(allocation);
        host=k==RAD_MEM_DEVICE?nullptr:rad_dev_host_ptr(allocation);
        if(!device||(k!=RAD_MEM_DEVICE&&!host)){
            rad_dev_free(allocation,k);allocation=nullptr;live-=n;
            throw std::runtime_error("owned mapped staging views");
        }
    }
    ~Owned(){
        if(!allocation)return;
        if(rad_stream_sync(stream)<0){std::fputs("failed drain: retaining checker allocation until process exit\n",stderr);return;}
        rad_dev_free(allocation,kind);live-=bytes;
    }
};
struct Receipt {int64_t bytes=0;~Receipt(){live-=bytes;}void take(int64_t n){charge(n);bytes=n;}};
struct Harness;Harness* active=nullptr;
void step(RadCtx*,const RadBatch*);
struct Harness {
    Registry& registry;RadStream stream;bool bad_pass;int expected_error=RAD_OK;
    route::testing::Fixture initial{},bad{},reference{};
    std::unique_ptr<Builder> builder;rad_op op=0;
    std::array<std::unique_ptr<Owned>,route::operand_count> device,staging;
    std::vector<RadOperand> operands;
    Receipt arena;
    int steps=0,good_checkpoints=0,negative_checkpoints=0,blocked_issues=0,publications=0;
    Ctx ctx; // actual Ctx drains before all operand/staging and Builder owners
    Harness(Registry& r,RadStream s,bool failing):registry(r),stream(s),bad_pass(failing){
        bad.buffers[route::ids].id(bad.jobs-1,99);
        expected_error=bad.run();require(expected_error<0,"bad oracle needs error");
        require(reference.run()==RAD_OK,"good route CPU component oracle");
        RadModelMeta meta{};meta.arch_id="checkpoint-aql-tiny";meta.name="synthetic";meta.quant="native";
        RadBuildCtx bc{};bc.rank=0;bc.world_size=1;bc.max_tok=4;bc.max_seqs=1;bc.max_ctx=8;bc.scope="";
        builder=std::make_unique<Builder>(registry,meta,bc);builder->set_instances(false);
        RadParam params[]{RAD_RANGE("M",1,4),RAD_INT("topk",2),RAD_INT("n_expert",5)};
        op=builder->decl_op("ds41_route_jobs",params,3,nullptr,0);
        require(op!=0,"actual Builder route op");check(builder->finish(),"actual Builder finish");
        auto& program=builder->program();
        for(const auto&band:program.ops[op].bands){
            const auto&resolved=band.dom[RAD_DOMAIN_DEVICE];
            require(resolved.row&&resolved.row->plugin=="libds41"&&resolved.row->info&&
                    resolved.row->info->launch&&resolved.row->deferred_status,"actual DS41 device row/collector");
        }
        const auto reserved=arena_plan_bytes(program,nullptr,nullptr);
        require(reserved>0&&reserved<=128*1024,"tiny actual Ctx arena/scratch");arena.take(reserved);
        for(int i=0;i<route::operand_count;++i){
            const auto& b=initial.buffers[size_t(i)];const auto bytes=int64_t(b.words.size()*4)+2*guard;
            require(bytes<=4096,"individual route fixture too large");
            device[size_t(i)]=std::make_unique<Owned>(stream,bytes,RAD_MEM_DEVICE);
            staging[size_t(i)]=std::make_unique<Owned>(stream,bytes,RAD_MEM_HOST_MAPPED);
            void* ptr=static_cast<uint8_t*>(device[size_t(i)]->device)+guard;
            if(b.t.rank==2)operands.push_back(RAD_P_T2(ptr,b.t.dtype,b.t.shape[0],b.t.shape[1]));
            else {auto operand=RAD_P_T(ptr,b.t.dtype);operand.rows=b.t.shape[0];operands.push_back(operand);}
        }
        CtxDesc cd{};cd.program=&program;cd.device=0;cd.rank=0;cd.world_size=1;
        cd.stream=stream;cd.arch_step=step;cd.record_passes=true;
        check(ctx.init(cd),"actual Ctx init");check(ctx.bind_thread(),"actual Ctx bind");
        require(ctx.stream()==stream&&ctx.checks_device_status()&&!ctx.records_passes(),"real stream/deferred/eager binding");
        require(ctx.checkpoint_device_status()==RAD_E_STATE,"outside-pass checkpoint refused");
        for(int i=0;i<route::operand_count;++i)upload(i,initial.buffers[size_t(i)].words);
        check(rad_stream_sync(stream),"finish owned initial payload uploads");
    }
    void upload(int i,const std::vector<uint32_t>& words){
        auto& b=*staging[size_t(i)];
        require(int64_t(words.size()*4)+2*guard==b.bytes,"staging geometry");
        std::memset(b.host,0xBD,size_t(b.bytes));std::memcpy(static_cast<uint8_t*>(b.host)+guard,words.data(),words.size()*4);
        rad_dev_host_wrote();check(rad_memcpy_async(device[size_t(i)]->device,b.device,b.bytes,stream),"owned mapped upload");
    }
    void inputs(bool invalid){
        for(int i=0;i<route::expert_ids;++i)upload(i,(invalid?bad:initial).buffers[size_t(i)].words);
    }
    void compare(int i,const std::vector<uint32_t>& words){
        auto& b=*staging[size_t(i)];
        check(rad_memcpy_async(b.device,device[size_t(i)]->device,b.bytes,stream),"owned diagnostic readback");
        check(rad_stream_sync(stream),"complete diagnostic readback");
        const auto* p=static_cast<const uint8_t*>(b.host);
        for(int64_t j=0;j<guard;++j)require(p[j]==0xBD&&p[b.bytes-guard+j]==0xBD,"device allocation guard");
        require(!std::memcmp(p+guard,words.data(),words.size()*4),"route payload/source/trailing canary differs");
    }
    void good_outputs(){for(int i=route::expert_ids;i<route::operand_count;++i)compare(i,reference.buffers[size_t(i)].words);}
    int issue(){return rad_issue(&ctx,op,operands.data(),int(operands.size()),initial.rows);}
    void run_arch(){
        ++steps;check(issue(),"first actual registered route launch");
        check(ctx.checkpoint_device_status(),"first completed checkpoint");++good_checkpoints;
        good_outputs(); // diagnostic only; caller still does not publish logits/state
        if(!bad_pass){
            for(int j=0;j<3;++j){check(issue(),"later registered launch");
                check(ctx.checkpoint_device_status(),"repeated completed checkpoint");++good_checkpoints;}
            check(issue(),"registered launch after last successful checkpoint");
            return;
        }
        inputs(true);check(issue(),"bad payload still enqueues actual row");
        require(ctx.checkpoint_device_status()==expected_error,"negative checkpoint must observe actual collector");++negative_checkpoints;
        const std::string first=ctx.step_error();
        require(first.find("op "+std::to_string(op)+" (ds41_route_jobs)")!=std::string::npos,"actual op attribution");
        // GOOD source plus poison output would visibly change if a later body
        // ran. Diagnostic copies do not pass through Ctx issue dispatch.
        inputs(false);
        for(int i=route::expert_ids;i<route::operand_count;++i)upload(i,initial.buffers[size_t(i)].words);
        require(issue()==expected_error,"later good issue was not blocked");++blocked_issues;
        require(ctx.checkpoint_device_status()==expected_error,"first checkpoint failure replaced/reset");
        require(ctx.step_error()==first,"first attribution changed");
    }
    void run(){
        RadBatch batch{};batch.phase=RAD_PHASE_PREFILL;batch.n_tok=initial.rows;batch.n_seq=1;
        active=this;const int queued=ctx.run_step(&batch);active=nullptr;
        const std::string first=ctx.step_error();const int completed=ctx.await_device_status();
        if(queued>=0&&completed>=0)++publications;
        if(bad_pass){
            require(queued==expected_error&&completed==expected_error&&publications==0,"bad pass published or lost first failure");
            require(ctx.step_error()==first,"final drain replaced attribution");
            require(good_checkpoints==1&&negative_checkpoints==1&&blocked_issues==1,"bad checkpoint counts");
            for(int i=0;i<route::operand_count;++i)compare(i,initial.buffers[size_t(i)].words);
            active=this;require(ctx.run_step(&batch)==expected_error,"fatal checkpoint Ctx reused");active=nullptr;
            require(steps==1,"fatal context ran architecture again");
        }else{
            require(queued==RAD_OK&&completed==RAD_OK&&publications==1&&good_checkpoints==4,"good checkpoint lifecycle");
            good_outputs();for(int i=0;i<route::expert_ids;++i)compare(i,initial.buffers[size_t(i)].words);
        }
        require(ctx.checkpoint_device_status()==RAD_E_STATE,"sealed/awaited misuse not refused");
        require(ctx.pass_counts().played==0&&!ctx.records_passes(),"checkpoint profile entered replay");
        check(rad_stream_sync(stream),"finish tiny checker stream");
    }
};
void step(RadCtx* context,const RadBatch*){
    if(!active){rad_step_fail(context,"missing checkpoint checker owner");return;}
    try{active->run_arch();}catch(...){rad_step_fail(context,"checkpoint checker assertion/operation failed");throw;}
}
void usage(){std::puts("ds41-aql-checkpoint-check --run-gpu --home ISOLATED_HOME [--host-kargs]\nExplicit one-card gfx1201 actual Builder/Ctx/registered route row/AQL checkpoint check; no model, Pools or service.");}
} // namespace
int main(int argc,char**argv){
    bool approved=false,host_kargs=false;std::string home;
    for(int i=1;i<argc;++i){const std::string a=argv[i];
        if(a=="--help"||a=="-h"){usage();return 0;}
        if(a=="--run-gpu")approved=true;
        else if(a=="--host-kargs")host_kargs=true;
        else if(a=="--home"&&i+1<argc)home=argv[++i];
        else{std::fprintf(stderr,"unknown/incomplete argument: %s\n",a.c_str());return 2;}
    }
    if(!approved||home.empty()){usage();return 2;}
    try{
#if defined(RAD_HAVE_HIP) && RAD_HAVE_HIP
        if(host_kargs)check(aql_force_host_kargs(),"select host argument rings");
#else
        (void)host_kargs;throw std::runtime_error("HIP-enabled Core required");
#endif
        require(rad_dev_count()==1,"exactly one visible GPU required");
        require(!device_is_host()&&!std::strcmp(device_backend_name(),"aql"),"actual AQL required, HOST fallback is not a pass");
        RadDeviceProps p{};check(rad_dev_props(0,&p),"one visible card properties");
        require(!p.is_host_backend&&!std::strcmp(p.arch,"gfx1201"),"requires visible gfx1201");check(rad_dev_set(0),"select visible card");
        Registry registry;check(registry.load_plugin(home+"/quantizers/libquant.so",0),"load isolated quantizer");
        check(registry.load_plugin(home+"/experimental/kernels/libds41.so",0),"load isolated actual DS41 device module");
        Stream stream;{Harness good(registry,stream.s,false);good.run();}{Harness bad(registry,stream.s,true);bad.run();}
        check(rad_stream_sync(stream.s),"final actual AQL drain");require(live==0,"explicit fixture receipt leak");
        std::printf("PASS: checkpoint overlay actual AQL TP1; five successful checkpoints, one negative checkpoint, one blocked later issue; peak requested bytes=%lld.\n",(long long)peak);
        std::puts("No Core launch substitution, TP collective, full model, rollback, hardware-failure recovery or performance proof.");return 0;
    }catch(const std::exception&e){active=nullptr;std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
