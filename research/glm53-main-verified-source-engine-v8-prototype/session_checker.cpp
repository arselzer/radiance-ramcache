// Actual Core Session/Builder/Ctx admission on HOST. No verified-source capability,
// Native TP4 payload, architecture step, sampler step, or GPU is created here.
#include "runtime/cache_prepare4_session.h"
#include "runtime/ctx.h"
#include "engine_priv.h"
#include "device/device.h"
#include "rad_glm_main_tp4_cache.h"
#include "rad_model_failure_policy.h"
#include "regressions/compact/allocation_hooks.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <csignal>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
namespace rad {
// Existing private HOST fixture friend; no production setter/capability added.
struct ModelFailurePolicyTestAccess {
    static int gate(Engine& e,const char* name){e.plugins_.arch_plugin=name;return e.glm_main_tp4_cold_source_gate();}
    static bool no_ranks(const Engine& e){return e.ranks_.empty()&&!e.cache_prepare4_;}
};
}
namespace {
namespace cp=glm53::cache_prepare4;
size_t checks=0,groups=0,refusals=0;
void need(bool b,const char* why){++checks;if(!b)throw std::runtime_error(why);}
void good(int rc,const char* why){need(rc==RAD_OK,why);}
std::unique_ptr<rad::CachePrepare4Session> session(){std::unique_ptr<rad::CachePrepare4Session> s;
    good(rad::CachePrepare4ProtocolHarness::create(11,29,&s),"protocol harness construction");return s;}
RadBatch batch(){RadBatch b{};b.phase=RAD_PHASE_PREFILL;b.n_tok=3;b.n_seq=1;b.n_out=1;b.step=17;return b;}
std::array<int32_t,128*8> routes(const RadBatch& b,int layer){std::array<int32_t,128*8> a{};
    for(int i=0;i<b.n_tok*8;++i)a[size_t(i)]=(i*13+layer*7)%288;return a;}
template<class Fn>std::array<int,4> four(rad::CachePrepare4Session& s,const RadBatch& b,Fn fn){
    // A failed authored gate must fail the checker, not strand its test process.
    std::atomic<bool> timeout_fired{false};
    std::jthread timeout([&](std::stop_token stop){const auto limit=std::chrono::steady_clock::now()+std::chrono::seconds(10);
        while(!stop.stop_requested()&&std::chrono::steady_clock::now()<limit)std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if(!stop.stop_requested()){timeout_fired.store(true,std::memory_order_release);(void)s.binding(0)->finish(b,RAD_E_IO);}});
    std::array<int,4> result{};std::array<std::jthread,4> workers;
    for(int rank=0;rank<4;++rank)workers[size_t(rank)]=std::jthread([&,rank]{try{result[size_t(rank)]=fn(rank);}catch(...){result[size_t(rank)]=s.binding(rank)->finish(b,RAD_E_STATE);}});
    for(auto& worker:workers)worker.join();timeout.request_stop();timeout.join();
    need(!timeout_fired.load(std::memory_order_acquire),"protocol required watchdog rescue; intended gate notification did not complete");return result;
}
void generations_and_sampler_frame(){++groups;auto s=session();
    need(!s->runtime_admitted()&&s->model_generation()==11&&s->source_generation()==29,"CPU protocol cannot grant source admission");
    need(!s->binding(-1)&&!s->binding(4),"rank boundary");
    for(int round=0;round<2;++round){auto b=batch();b.step+=round;
        if(!round){b.n_tok=128;b.n_ahead=37;}else{b.phase=RAD_PHASE_DECODE;b.n_tok=1;b.n_ahead=0;}
        good(s->begin(b),"begin truthful exact batch");need(b.n_ahead==(!round?37:0),"Batch lookahead preserved");
        std::atomic<unsigned> gu{0},owner_work{0};
        const auto result=four(*s,b,[&](int rank){auto* binding=s->binding(rank);
            for(int layer=3;layer<=44;++layer){auto ids=routes(b,layer);int rc=binding->enter(layer,0,b.n_tok,{ids.data(),size_t(b.n_tok*8)});if(rc<0)return rc;
                if(rank!=2)++owner_work; // resident-only rank still votes at EVERY boundary
                rc=binding->ready(RAD_OK);if(rc<0)return rc;++gu;}
            return binding->finish(b,RAD_OK);});
        for(int rc:result)good(rc,"four actual Session rank completions");
        const auto state=s->observe();need(!state.active&&state.finished==15&&state.ordinal==42,"MODEL coordinator complete");
        need(s->active()&&gu==168&&owner_work==126,"frame stays active through sampler/collect, resident-only votes");
        for(int rank=0;rank<4;++rank)good(s->binding(rank)->check_batch(b),"same exact pass accepted before sampler");
        auto stale=b;++stale.step;need(s->binding(0)->check_batch(stale)<0,"stale sampler pass refused");++refusals;
        good(s->complete_frame(b),"exactly one successful collect completes frame");need(!s->active(),"frame inactive after collect");
        if(round){need(s->complete_frame(b)<0&&!s->active(),"duplicate collect completion refused");++refusals;}
    }
    s->stop_admission();need(s->admission_closed(),"idle stop permanently closes begin admission");
    allocation_fault::begin();const int rc=s->begin(batch());const auto calls=allocation_fault::calls;allocation_fault::end();
    need(rc==RAD_E_STATE&&!calls&&!s->active(),"closed idle Session refuses begin without work");++refusals;
    s->stop_admission();need(s->admission_closed(),"idle stop idempotent");
}
void malformed_and_atomic_admission(){++groups;
    std::unique_ptr<rad::CachePrepare4Session> s;
    need(rad::CachePrepare4ProtocolHarness::create(0,29,&s)==RAD_E_INVAL&&!s,"zero model generation refused");
    need(rad::CachePrepare4ProtocolHarness::create(11,0,&s)==RAD_E_INVAL&&!s,"zero source generation refused");
    allocation_fault::begin();allocation_fault::remaining=0;const int oom=rad::CachePrepare4ProtocolHarness::create(11,29,&s);allocation_fault::end();
    need(oom==RAD_E_NOMEM&&!s&&!allocation_fault::live,"Session OOM publication atomic");
    s=session();auto* original=s.get();need(rad::CachePrepare4ProtocolHarness::create(11,29,&s)==RAD_E_INVAL&&s.get()==original,"occupied caller preserved");
    for(int mutation=0;mutation<15;++mutation){auto test=session();auto b=batch();
        switch(mutation){case 0:b.step=-1;break;case 1:b.n_tok=0;break;case 2:b.n_tok=129;break;case 3:b.n_seq=2;break;case 4:b.n_out=-1;break;
            case 5:b.n_out=2;break;case 6:b.draft_pass=1;break;case 7:b.n_spec=1;break;case 8:b.n_ahead=-1;break;case 9:b.n_ahead=129;break;
            case 10:b.phase=RAD_PHASE_DECODE;b.n_tok=2;break;case 11:b.phase=RAD_PHASE_DECODE;b.n_tok=1;b.n_ahead=1;break;
            case 12:b.phase=0x7fff;break;case 13:b.n_seq=0;break;case 14:b.n_tok=-1;break;}
        allocation_fault::begin();const int rc=test->begin(b);const auto calls=allocation_fault::calls;allocation_fault::end();
        need(rc<0&&!calls&&!test->active(),"unsupported batch refuses before allocation/activation");++refusals;}
    for(int layer:{INT_MIN,-1,2,45,INT_MAX}){auto test=session();auto b=batch();good(test->begin(b),"malformed-layer pass");auto ids=routes(b,3);
        need(test->binding(0)->enter(layer,0,b.n_tok,{ids.data(),size_t(b.n_tok*8)})<0,"bounded layer refusal before signed subtraction");++refusals;}
    {auto test=session();auto b=batch();good(test->begin(b),"premature collect pass");need(test->complete_frame(b)<0&&test->active(),"cannot complete frame before model gates");++refusals;}
    {auto test=session();auto b=batch();good(test->begin(b),"second begin pass");need(test->begin(b)<0,"cannot replace in-flight frame");++refusals;}
}
void coordinated_failures(){++groups;
    for(int mutation=0;mutation<4;++mutation){auto s=session();auto b=batch();b.n_ahead=7;good(s->begin(b),"failure pass begin");std::atomic<unsigned> owners{0},gu{0},late_source_witness{0};
        const auto result=four(*s,b,[&](int rank){auto* binding=s->binding(rank);auto ids=routes(b,3);
            if(mutation==0&&rank==3)ids[0]=(ids[0]+1)%288;
            if(mutation==1&&rank==3)return binding->finish(b,RAD_OK); // omitted first prepare boundary
            if(mutation==2&&rank==3)return binding->enter(INT_MIN,0,b.n_tok,{ids.data(),size_t(b.n_tok*8)});
            int rc=binding->enter(3,0,b.n_tok,{ids.data(),size_t(b.n_tok*8)});if(rc<0)return rc;++owners;
            if(mutation==3){
                // Test-only late-fault gate: all four workers must have actually
                // returned from enter and reached owner work before the injected
                // READY error. A faster peer may otherwise fail enter waiters
                // correctly, before they can increment owners.
                const auto limit=std::chrono::steady_clock::now()+std::chrono::seconds(10);
                while(owners.load(std::memory_order_acquire)!=4){
                    const int failed=s->observe().first.status;if(failed<0)return failed;
                    if(std::chrono::steady_clock::now()>=limit)return binding->finish(b,RAD_E_STATE);
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                if(rank==3)++late_source_witness;
            }
            rc=binding->ready(mutation==3&&rank==3?RAD_E_IO:RAD_OK);if(rc==RAD_OK)++gu;return rc;});
        for(int rc:result)need(rc<0,"all rank waiters wake on first refusal");
        const auto first=s->observe().first;need(first.status<0&&gu==0,"no GU after boundary refusal");
        if(mutation<3)need(owners==0,"route/early-exit refusal before owner work");else need(owners==4&&late_source_witness==1&&first.status==RAD_E_IO&&first.reason==cp::Reason::source_failure&&first.rank==3,"late source error after all four actual owner-work arrivals");
        need(s->begin(b)<0&&s->observe().first==first,"first failure sticky/no pass retry");++refusals;}
}
void builder_and_closed_runtime(){++groups;rad::Registry registry;RadModelMeta meta{};RadBuildCtx bc{};bc.world_size=4;bc.max_tok=3;bc.max_seqs=1;
    need(rad_core_glm_main_tp4_cache_profile_version()==1,"optional additive probe version");
    need(rad_require_glm_main_tp4_cache_fail_stop(nullptr)==RAD_E_INVAL,"null builder refused");
    {rad::Builder b(registry,meta,bc);need(rad_require_glm_main_tp4_cache_fail_stop(&b)==RAD_E_UNSUPPORTED,"opaque-instance declaration not admitted");}
    {rad::Builder b(registry,meta,bc);b.set_instances(false);good(rad_require_glm_main_tp4_cache_fail_stop(&b),"metadata-only flag");
        need(b.program().requires_glm_main_tp4_cache_fail_stop()&&!b.program().requires_permanent_model_failure(),"separate immutable policies");
        good(b.finish(),"finish metadata-only empty graph");need(rad_require_glm_main_tp4_cache_fail_stop(&b)==RAD_E_STATE,"finished declaration refuses new policy");
        auto s=session();rad::Ctx ctx;rad::CtxDesc d;d.program=&b.program();d.rank=0;d.world_size=4;d.record_passes=false;d.cache_prepare4=s->binding(0);
        allocation_fault::begin();const int rc=ctx.init(d);const auto calls=allocation_fault::calls;allocation_fault::end();
        need(rc==RAD_E_UNSUPPORTED&&!calls&&!ctx.program(),"protocol binding cannot grant real Ctx source/backend capability");}
    {rad::Builder b(registry,meta,bc);b.set_instances(false);good(b.require_permanent_model_failure(),"original PermanentV1 declaration");
        need(b.require_glm_main_tp4_cache_fail_stop()==RAD_E_UNSUPPORTED&&!b.program().requires_glm_main_tp4_cache_fail_stop(),"TP4 cannot replace PermanentV1");}
    {rad::Builder b(registry,meta,bc);b.set_instances(false);good(b.require_glm_main_tp4_cache_fail_stop(),"new profile declaration");
        need(b.require_permanent_model_failure()==RAD_E_UNSUPPORTED&&!b.program().requires_permanent_model_failure(),"PermanentV1 cannot replace TP4");}
    rad::Config cfg;rad::Engine e(cfg);
    good(rad::ModelFailurePolicyTestAccess::gate(e,"glm5next"),"ordinary plugin cold path unchanged");
    const auto committed=rad::rad_dev_vmem_committed();
    need(rad::ModelFailurePolicyTestAccess::gate(e,"glm5next_cache")==RAD_E_UNSUPPORTED,"actual experimental Engine gate remains hard closed");
    need(rad::rad_dev_vmem_committed()==committed&&rad::ModelFailurePolicyTestAccess::no_ranks(e),"gate creates no rank/Pools/model payload or Session");
}
int fatal_case(int which,int fd){need(::fcntl(fd,F_GETFD)>=0,"live inherited unwind witness");
    struct Marker{int fd;~Marker(){const char c='D';(void)::write(fd,&c,1);}} marker{fd};
    auto s=session();auto b=batch();good(s->begin(b),"fatal child pass begin");
    if(which==0)s->binding(2)->fail_stop(RAD_E_IO);
    if(which==1)s->stop_admission();
    if(which==2)s->stop_if_active();
    return 9;
}
void fail_stop_before_unwind(const char* executable){++groups;
    for(int which=0;which<3;++which){int fds[2];need(::pipe2(fds,O_CLOEXEC)==0,"death witness pipe");const pid_t pid=::fork();need(pid>=0,"death child fork");
        if(!pid){::close(fds[0]);
            // Only the owned child's writer deliberately survives exec.
            if(::fcntl(fds[1],F_SETFD,0)<0)_exit(126);
            const std::string a=std::to_string(which),b=std::to_string(fds[1]);::execl(executable,executable,"--fatal-case",a.c_str(),b.c_str(),(char*)nullptr);_exit(127);}
        ::close(fds[1]);
        // Even an unexpected external SIGCHLD reaper cannot turn the witness
        // read into another unbounded wait.
        if(::fcntl(fds[0],F_SETFL,O_NONBLOCK)<0){(void)::kill(pid,SIGKILL);while(::waitpid(pid,nullptr,0)<0&&errno==EINTR){}::close(fds[0]);need(false,"nonblocking witness read");}
        int status=0;bool reaped=false,owned=true;
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
        while(std::chrono::steady_clock::now()<deadline){const pid_t got=::waitpid(pid,&status,WNOHANG);
            if(got==pid){reaped=true;break;}if(got<0&&errno!=EINTR){owned=false;break;}
            std::this_thread::sleep_for(std::chrono::milliseconds(2));}
        if(!reaped&&owned){(void)::kill(pid,SIGKILL);while(::waitpid(pid,&status,0)<0&&errno==EINTR){}}
        char marker=0;const auto count=::read(fds[0],&marker,1);::close(fds[0]);
        need(reaped,"death child exceeded monotonic deadline; owned child killed and reaped");
        need(WIFEXITED(status)&&WEXITSTATUS(status)==88&&count==0,"fixed process-stop 88 before stack cleanup marker");++refusals;}
}
}
int main(int argc,char** argv){try{
    if(argc==4&&!std::strcmp(argv[1],"--fatal-case"))return fatal_case(std::atoi(argv[2]),std::atoi(argv[3]));
    need(argc==2&&!std::strcmp(argv[1],"--run-host-protocol"),"explicit --run-host-protocol required");
    need(rad::device_is_host(),"HOST only; no DEVICE/TP4 payload execution");
    generations_and_sampler_frame();malformed_and_atomic_admission();coordinated_failures();builder_and_closed_runtime();fail_stop_before_unwind(argv[0]);
    std::cout<<"PASS actual Core TP4 protocol groups="<<groups<<" checks="<<checks<<" refusals="<<refusals<<"; runtime source gate CLOSED, no model/GPU inference claim\n";return 0;
}catch(const std::exception& e){allocation_fault::end();std::cerr<<"FAIL actual Core TP4 protocol: "<<e.what()<<'\n';return 1;}}
