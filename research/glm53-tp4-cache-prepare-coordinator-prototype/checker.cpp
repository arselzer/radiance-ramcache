#include "cache_prepare4.h"
#include <array>
#include <atomic>
#include <barrier>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <sys/wait.h>
#include <unistd.h>
namespace gate=glm53::cache_prepare4;
namespace {
size_t checks=0,cases=0,refusals=0;uint64_t successful_gu=0;
void need(bool b,const char* why){++checks;if(!b)throw std::runtime_error(why);}
gate::Pass pass(uint64_t epoch=1){gate::Pass p;p.model_generation=11;p.source_generation=29;p.epoch=epoch;p.step=17;p.total_rows=3;return p;}
gate::Key key(gate::Pass p,uint32_t ordinal){return {p,ordinal,int32_t(gate::first_layer+ordinal),0,p.total_rows};}
std::array<int32_t,gate::max_rows*gate::top_k> ids(const gate::Pass& p,uint32_t layer){
    std::array<int32_t,gate::max_rows*gate::top_k> a{};for(int64_t i=0;i<p.total_rows*gate::top_k;++i)a[size_t(i)]=int32_t((i*13+layer*7)%gate::experts);return a;}
template<class Predicate>bool await(gate::Coordinator& c,Predicate pred){
    const auto limit=std::chrono::steady_clock::now()+std::chrono::seconds(10);
    while(std::chrono::steady_clock::now()<limit){const auto snapshot=c.observe();if(pred(snapshot))return true;
        if(snapshot.first.status<0)return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));}
    c.fail(-1,gate::protocol_error);return false;
}
template<class Fn>std::array<int,4> four(gate::Coordinator& c,Fn fn){
    std::array<int,4> result{};std::array<std::jthread,4> threads;
    for(int r=0;r<4;++r)threads[size_t(r)]=std::jthread([&,r]{try{result[size_t(r)]=fn(r);}catch(...){result[size_t(r)]=c.fail(r,gate::protocol_error);}});
    for(auto& t:threads)t.join();
    return result;
}
void same_failure(const gate::Coordinator& c,const std::array<int,4>& result,int expected){
    const auto first=c.observe().first;need(first.status==expected,"sticky first status");for(const int status:result)need(status==expected,"same all-rank failure");}
void success_and_generations(){++cases;gate::Coordinator c(11,29);
    for(int round=0;round<2;++round){auto p=pass(uint64_t(round+1));p.total_rows=round?1:gate::max_rows;p.phase=round?gate::Phase::decode:gate::Phase::prefill;
        need(c.begin(p)==0,"begin exact4rank profile");std::atomic<uint32_t> gu{0},fetch{0};
        const auto result=four(c,[&](int rank){for(uint32_t ord=0;ord<gate::layers;++ord){auto routes=ids(p,ord);const auto k=key(p,ord);
                int rc=c.enter(rank,k,{routes.data(),size_t(p.total_rows*gate::top_k)});if(rc<0)return rc;
                // Rank2 models an all-resident/no-miss local union. It still
                // participates in EVERY enter/ready vote, with no fetch work.
                if(rank!=2)++fetch;
                rc=c.ready(rank,k,0);if(rc<0)return rc;++gu;}
            return c.finish(rank,p,0);});
        for(const int status:result)need(status==0,"successful worker");
        const auto state=c.observe();need(!state.active&&state.ordinal==42&&state.finished==15&&state.first.status==0,"completed42layers all-rank finish");
        need(gu==168&&fetch==126,"exact4GU votes/layer including resident-only rank");successful_gu+=gu;
    }
}
struct Files {
    std::string root;std::array<int,4> fd{{-1,-1,-1,-1}};std::array<std::string,4> paths;
    Files(){char p[]="/tmp/glm53-cache-gate-XXXXXX";char* made=::mkdtemp(p);if(!made)throw std::runtime_error("files scratch");
        try{root=made;for(int r=0;r<4;++r){paths[size_t(r)]=root+"/rank"+std::to_string(r);fd[size_t(r)]=::open(paths[size_t(r)].c_str(),O_RDWR|O_CREAT|O_EXCL|O_CLOEXEC,0600);
                if(fd[size_t(r)]<0)throw std::runtime_error("rank physical source");
                std::array<uint8_t,64> bytes{};for(size_t i=0;i<bytes.size();++i)bytes[i]=uint8_t(r*17+int(i));
                if(::write(fd[size_t(r)],bytes.data(),bytes.size())!=ssize_t(bytes.size()))throw std::runtime_error("rank source bytes");}}
        catch(...){cleanup();throw;}}
    void cleanup()noexcept{for(int r=0;r<4;++r){if(fd[size_t(r)]>=0){::close(fd[size_t(r)]);fd[size_t(r)]=-1;}if(!paths[size_t(r)].empty())::unlink(paths[size_t(r)].c_str());}if(!root.empty())::rmdir(root.c_str());}
    ~Files(){cleanup();}
    int read(int rank){std::array<uint8_t,64> out{};const auto n=::pread(fd[size_t(rank)],out.data(),out.size(),0);if(n!=ssize_t(out.size()))return -13;
        for(size_t i=0;i<out.size();++i)if(out[i]!=uint8_t(rank*17+int(i)))return -14;
        return 0;}
};
void late_real_io_failure(){++cases;Files source;gate::Coordinator c(11,29);const auto p=pass();need(c.begin(p)==0,"I/O pass begin");std::atomic<uint32_t> gu{0},reads{0};
    const auto result=four(c,[&](int rank){auto routes=ids(p,0);const auto k=key(p,0);int rc=c.enter(rank,k,{routes.data(),size_t(p.total_rows*8)});if(rc<0)return rc;
        if(rank==3){if(!await(c,[](const auto& s){return s.ready==7;}))return c.fail(rank,gate::protocol_error);
            if(::ftruncate(source.fd[3],0)!=0)return c.fail(rank,gate::protocol_error);}
        ++reads;rc=source.read(rank);const int decision=c.ready(rank,k,rc);if(decision==0)++gu;return decision;});
    same_failure(c,result,-13);need(reads==4&&gu==0,"late actual short-read after3ready refuses ALL GU");
    need(c.observe().first.reason==gate::Reason::source_failure&&c.observe().first.rank==3,"source error attribution");
    const auto first=c.observe().first;need(c.begin(pass(2))==-13&&c.fail(0,-6)==-13&&c.observe().first==first,"no restart/first diagnostic replacement");++refusals;
}
void missing_rank(){++cases;gate::Coordinator c(11,29);const auto p=pass();need(c.begin(p)==0,"missing rank pass begin");std::atomic<uint32_t> callbacks{0},gu{0};
    const auto result=four(c,[&](int rank){if(rank==3){if(!await(c,[](const auto& s){return s.entered==7;}))return c.fail(rank,gate::protocol_error);
            // Real early-return notifier; rank3 NEVER enters the prepare gate.
            return c.finish(rank,p,0);}
        auto routes=ids(p,0);const auto k=key(p,0);int rc=c.enter(rank,k,{routes.data(),size_t(p.total_rows*8)});if(rc<0)return rc;
        ++callbacks;rc=c.ready(rank,k,0);if(rc==0)++gu;return rc;});
    same_failure(c,result,gate::protocol_error);need(callbacks==0&&gu==0&&c.observe().first.reason==gate::Reason::early_exit,"early exit wakes already-waiting peers before cache/GU");++refusals;
}
void route_mismatch(){++cases;gate::Coordinator c(11,29);const auto p=pass();need(c.begin(p)==0,"mismatched route pass begin");std::atomic<uint32_t> callbacks{0},gu{0};
    const auto result=four(c,[&](int rank){auto routes=ids(p,0);if(rank==3)routes[0]=(routes[0]+1)%288;
        const auto k=key(p,0);int rc=c.enter(rank,k,{routes.data(),size_t(p.total_rows*8)});if(rc<0)return rc;
        ++callbacks;rc=c.ready(rank,k,0);if(rc==0)++gu;return rc;});
    same_failure(c,result,gate::protocol_error);need(callbacks==0&&gu==0&&c.observe().first.reason==gate::Reason::ids_mismatch,"exact canonical route mismatch before owner work");++refusals;
}
void metadata_refusals(){++cases;
    for(int mutation=0;mutation<11;++mutation){gate::Coordinator c(11,29);auto p=pass();need(c.begin(p)==0,"metadata mutation fresh pass");auto k=key(p,0);auto routes=ids(p,0);int rank=0;size_t size=size_t(p.total_rows*8);
        if(mutation==0)k.pass.source_generation=30;
        if(mutation==1)k.pass.epoch=2;
        if(mutation==2)k.pass.step=18;
        if(mutation==3)k.pass.draft_pass=-1;
        if(mutation==4)k.layer=14;
        if(mutation==5)k.ordinal=1;
        if(mutation==6)k.first_row=1;
        if(mutation==7)--k.rows;
        if(mutation==8)routes[0]=288;
        if(mutation==9)--size;
        if(mutation==10)rank=4;
        need(c.enter(rank,k,{routes.data(),size})==gate::protocol_error,"metadata/source/slice/ID bound refusal");++refusals;
    }
    for(int mutation=0;mutation<6;++mutation){gate::Coordinator c(11,29);auto p=pass();
        if(mutation==0)p.total_rows=129;
        if(mutation==1)p.n_spec=1;
        if(mutation==2)p.n_ahead=1;
        if(mutation==3)p.sequences=2;
        if(mutation==4)p.epoch=0;
        if(mutation==5){p.phase=gate::Phase::decode;p.total_rows=3;}
        need(c.begin(p)==gate::protocol_error,"unsupported initial profile refused");++refusals;
    }
}
void abort_after_prior_go(){++cases;gate::Coordinator c(11,29);const auto p=pass();need(c.begin(p)==0,"late abort pass begin");std::barrier after_gu(4);std::atomic<uint32_t> prior_work{0},later_work{0};
    const auto result=four(c,[&](int rank){auto routes=ids(p,0);auto k=key(p,0);int rc=c.enter(rank,k,{routes.data(),size_t(p.total_rows*8)});if(rc<0)return rc;
        rc=c.ready(rank,k,0);if(rc<0)return rc;++prior_work;after_gu.arrive_and_wait();
        if(rank==0)return c.fail(rank,-13);
        k=key(p,1);routes=ids(p,1);rc=c.enter(rank,k,{routes.data(),size_t(p.total_rows*8)});if(rc<0)return rc;
        rc=c.ready(rank,k,0);if(rc==0)++later_work;return rc;});
    same_failure(c,result,-13);need(prior_work==4&&later_work==0,"earlier private work remains; no later GO after failure");++refusals;
}
void close_waiters(){++cases;gate::Coordinator c(11,29);const auto p=pass();need(c.begin(p)==0,"close pass begin");
    const auto result=four(c,[&](int rank){if(rank==3){if(!await(c,[](const auto& s){return s.entered==7;}))return c.fail(rank,gate::protocol_error);
            c.close();return c.observe().first.status;}
        auto routes=ids(p,0);return c.enter(rank,key(p,0),{routes.data(),size_t(p.total_rows*8)});});
    same_failure(c,result,gate::closed_error);need(c.observe().closed&&c.begin(pass(2))==gate::closed_error,"shutdown closes/wakes without reuse");++refusals;
}
void epoch_overflow(){++cases;gate::Coordinator c(11,29);auto p=pass(std::numeric_limits<uint64_t>::max());need(c.begin(p)==0,"last valid epoch");
    const auto result=four(c,[&](int rank){for(uint32_t i=0;i<42;++i){auto routes=ids(p,i);auto k=key(p,i);int rc=c.enter(rank,k,{routes.data(),size_t(p.total_rows*8)});if(rc<0)return rc;
            rc=c.ready(rank,k,0);if(rc<0)return rc;}return c.finish(rank,p,0);});
    for(const int status:result)need(status==0,"last epoch completes");
    need(c.begin(pass(1))==gate::protocol_error&&c.observe().first.reason==gate::Reason::epoch_overflow,"epoch wrap refuses permanently");++refusals;
}
int fatal_child(int marker){struct Marker {int fd;~Marker(){const char x='D';(void)::write(fd,&x,1);}} destruction{marker};
    gate::Coordinator c(11,29);const auto p=pass();if(c.begin(p)!=0)return 91;
    const auto result=four(c,[&](int rank){if(rank==3){if(!await(c,[](const auto& s){return s.entered==7;}))gate::process_fail_stop();
            const int status=c.fail(rank,-13);gate::require_success_or_process_stop(status);return status;}
        auto routes=ids(p,0);const int status=c.enter(rank,key(p,0),{routes.data(),size_t(p.total_rows*8)});gate::require_success_or_process_stop(status);return status;});
    (void)result;return 92;
}
void process_fatal_proof(){++cases;int pipefd[2]{};need(::pipe(pipefd)==0,"destructor witness pipe");const pid_t pid=::fork();need(pid>=0,"fatal-profile subprocess");
    if(pid==0){::close(pipefd[0]);::alarm(15);const int status=fatal_child(pipefd[1]);std::_Exit(status);}
    ::close(pipefd[1]);int status=0;need(::waitpid(pid,&status,0)==pid,"fatal subprocess wait");char byte=0;const auto read=::read(pipefd[0],&byte,1);::close(pipefd[0]);
    need(WIFEXITED(status)&&WEXITSTATUS(status)==88&&read==0,"negative profile exits process before C++ destructor, not graceful GPU retirement");++refusals;
}
} // namespace
int main(){try{success_and_generations();late_real_io_failure();missing_rank();route_mismatch();metadata_refusals();abort_after_prior_go();close_waiters();epoch_overflow();process_fatal_proof();
    need(cases==9&&refusals==24,"complete closed-profile matrix");
    std::cout<<"PASS TP4 cache HOST coordinator cases="<<cases<<" checks="<<checks<<" refusalGroups="<<refusals<<" twoSuccessPassGUCount="<<successful_gu<<" sizeofCoordinator="<<sizeof(gate::Coordinator)<<"; no Core, GPU, model, collectives or recovery proof\n";return 0;
}catch(const std::exception& e){std::cerr<<"FAIL coordinator: "<<e.what()<<"\n";return 1;}}
