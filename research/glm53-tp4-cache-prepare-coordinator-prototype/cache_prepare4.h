#pragma once
// Closed CPU executable specification. No Core/GPU/collective recovery support.
#include <array>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <span>

namespace glm53::cache_prepare4 {
inline constexpr int ranks=4,experts=288,top_k=8,max_rows=128;
inline constexpr uint32_t layers=42,first_layer=3;
inline constexpr uint8_t all=15;
inline constexpr int protocol_error=-10001,closed_error=-10002,cancelled_error=-10003;
// Private specification statuses above are not proposed public RAD ABI numbers.
// A future actual Core adapter maps them to RAD_E_STATE/explicit diagnostics.
enum class Phase:uint8_t {prefill,decode};
struct Pass {
    uint64_t model_generation=0,source_generation=0,epoch=0;
    int32_t step=0,draft_pass=0,n_spec=0,n_ahead=0;
    Phase phase=Phase::prefill;
    int64_t total_rows=0,sequences=1,n_out=1;
    bool operator==(const Pass&)const=default;
};
struct Key {
    Pass pass;
    uint32_t ordinal=0;
    int32_t layer=first_layer;
    int64_t first_row=0,rows=0;
    bool operator==(const Key&)const=default;
};
enum class Reason:uint8_t {
    none,bad_profile,bad_pass,bad_rank,bad_key,bad_ids,ids_mismatch,
    duplicate,source_failure,early_exit,explicit_abort,closed,epoch_overflow
};
struct Failure {
    int status=0,rank=-1;
    Reason reason=Reason::none;
    uint64_t epoch=0;
    uint32_t ordinal=0;
    bool operator==(const Failure&)const=default;
};
struct Snapshot {
    Failure first;
    uint64_t epoch=0;
    uint32_t ordinal=0;
    uint8_t entered=0,ready=0,consumed=0,finished=0;
    bool active=false,closed=false;
};

class Coordinator final {
public:
    Coordinator(uint64_t model_generation,uint64_t source_generation)noexcept
        :model_(model_generation),source_(source_generation){}
    Coordinator(const Coordinator&)=delete;
    // Startup/scheduler-only before releasing workers. No overlapping pass.
    int begin(const Pass& p){std::lock_guard lock(mu_);
        if(first_.status<0)return first_.status;
        if(closed_)return fail_locked(-1,closed_error,Reason::closed);
        if(!model_||!source_)return fail_locked(-1,protocol_error,Reason::bad_profile);
        if(active_||p.model_generation!=model_||p.source_generation!=source_||!p.epoch||p.step<0||
           p.draft_pass||p.n_spec||p.n_ahead||p.total_rows<1||p.total_rows>max_rows||p.sequences!=1||p.n_out<0||p.n_out>1||
           (p.phase==Phase::decode&&p.total_rows!=1)||
           (p.phase!=Phase::prefill&&p.phase!=Phase::decode))return fail_locked(-1,protocol_error,Reason::bad_pass);
        if(previous_epoch_==std::numeric_limits<uint64_t>::max())return fail_locked(-1,protocol_error,Reason::epoch_overflow);
        if(previous_epoch_&&p.epoch!=previous_epoch_+1)return fail_locked(-1,protocol_error,Reason::bad_pass);
        pass_=p;previous_epoch_=p.epoch;active_=true;ordinal_=0;entered_=ready_=consumed_=finished_=0;
        completed_.fill(0);phase_=State::entering;return 0;
    }
    // Local routing has ALREADY completed and passed its real local coverage
    // validation. This exact canonical-ID gate precedes owner/cache work.
    int enter(int rank,const Key& key,std::span<const int32_t> ids){std::unique_lock lock(mu_);
        if(first_.status<0)return first_.status;
        if(!member(rank))return fail_locked(rank,protocol_error,Reason::bad_rank);
        if(!active_||key.pass!=pass_)return fail_locked(rank,protocol_error,Reason::bad_pass);
        // A fast rank may have consumed GO and reached the next slice while a
        // peer is still returning from ready. Wait for that SAME prior gate.
        if(phase_==State::going&&key.ordinal==ordinal_+1&&(consumed_&bit(rank)))
            cv_.wait(lock,[&]{return first_.status<0||phase_!=State::going;});
        if(first_.status<0)return first_.status;
        if(!valid_key(key)||phase_!=State::entering)return fail_locked(rank,protocol_error,Reason::bad_key);
        if(entered_&bit(rank))return fail_locked(rank,protocol_error,Reason::duplicate);
        const auto count=size_t(pass_.total_rows*top_k);
        if(ids.size()!=count||!ids.data())return fail_locked(rank,protocol_error,Reason::bad_ids);
        for(size_t i=0;i<count;++i)if(ids[i]<-1||ids[i]>=experts)return fail_locked(rank,protocol_error,Reason::bad_ids);
        for(size_t i=0;i<count;++i)ids_[size_t(rank)][i]=ids[i];
        entered_|=bit(rank);
        if(entered_==all){
            for(int r=1;r<ranks;++r)for(size_t i=0;i<count;++i)
                if(ids_[size_t(r)][i]!=ids_[0][i])return fail_locked(r,protocol_error,Reason::ids_mismatch);
            phase_=State::fetching;cv_.notify_all();
        }else cv_.notify_all();
        cv_.wait(lock,[&]{return first_.status<0||phase_==State::fetching||phase_==State::going;});
        return first_.status<0?first_.status:0;
    }
    // Cache acquire/validation/upload preparation happens OUTSIDE this lock.
    // status0 means the integration established its stated LOCAL readiness
    // contract, not that this class inspected any stream/event/lease.
    int ready(int rank,const Key& key,int status){std::unique_lock lock(mu_);
        if(first_.status<0)return first_.status;
        if(!member(rank))return fail_locked(rank,protocol_error,Reason::bad_rank);
        if(status<0)return fail_locked(rank,status,Reason::source_failure);
        if(!active_||key.pass!=pass_)return fail_locked(rank,protocol_error,Reason::bad_pass);
        if(status!=0||!valid_key(key)||!(entered_&bit(rank))||
           (phase_!=State::fetching&&phase_!=State::going))return fail_locked(rank,protocol_error,Reason::bad_key);
        if(ready_&bit(rank))return fail_locked(rank,protocol_error,Reason::duplicate);
        ready_|=bit(rank);if(ready_==all)phase_=State::going;cv_.notify_all();
        cv_.wait(lock,[&]{return first_.status<0||phase_==State::going;});
        if(first_.status<0)return first_.status;
        consumed_|=bit(rank);++completed_[size_t(rank)];
        if(consumed_==all){++ordinal_;entered_=ready_=consumed_=0;
            phase_=ordinal_==layers?State::complete:State::entering;cv_.notify_all();}
        return 0; // Only this success authorizes LOCAL GU work for this key.
    }
    // Rank guards MUST call this even if no enter/ready was reached. A normal
    // early return is not allowed to strand an expected routed-layer gate.
    int finish(int rank,const Pass& pass,int status){std::lock_guard lock(mu_);
        if(first_.status<0)return first_.status;
        if(!member(rank))return fail_locked(rank,protocol_error,Reason::bad_rank);
        if(status<0)return fail_locked(rank,status,Reason::early_exit);
        if(!active_||pass!=pass_||status!=0)return fail_locked(rank,protocol_error,Reason::bad_pass);
        if(finished_&bit(rank))return fail_locked(rank,protocol_error,Reason::duplicate);
        if(completed_[size_t(rank)]!=layers)return fail_locked(rank,protocol_error,Reason::early_exit);
        finished_|=bit(rank);if(finished_==all){active_=false;cv_.notify_all();}return 0;
    }
    // Immediate notification before diagnostics, drains, destructor cleanup or
    // a returned architecture error. First status remains sticky forever.
    int fail(int rank,int status,Reason reason=Reason::explicit_abort)noexcept{
        std::lock_guard lock(mu_);return fail_locked(rank,status<0?status:protocol_error,reason);
    }
    void close()noexcept{std::lock_guard lock(mu_);closed_=true;
        if(active_&&first_.status>=0)(void)fail_locked(-1,closed_error,Reason::closed);
        cv_.notify_all();}
    Snapshot observe()const{std::lock_guard lock(mu_);return {first_,pass_.epoch,ordinal_,entered_,ready_,consumed_,finished_,active_,closed_};}
private:
    enum class State:uint8_t {entering,fetching,going,complete,failed};
    static bool member(int r)noexcept{return r>=0&&r<ranks;}
    static uint8_t bit(int r)noexcept{return uint8_t(1u<<r);}
    bool valid_key(const Key& k)const noexcept{return k.ordinal==ordinal_&&k.ordinal<layers&&
        k.layer==int32_t(first_layer+k.ordinal)&&k.first_row==0&&k.rows==pass_.total_rows;}
    int fail_locked(int rank,int status,Reason reason)noexcept{
        if(first_.status>=0){first_={status,rank,reason,pass_.epoch,ordinal_};phase_=State::failed;}
        cv_.notify_all();return first_.status;
    }
    const uint64_t model_,source_;
    mutable std::mutex mu_;std::condition_variable cv_;
    Pass pass_{};Failure first_{};uint64_t previous_epoch_=0;
    uint32_t ordinal_=0;uint8_t entered_=0,ready_=0,consumed_=0,finished_=0;
    bool active_=false,closed_=false;State phase_=State::entering;
    std::array<uint32_t,ranks> completed_{};
    std::array<std::array<int32_t,max_rows*top_k>,ranks> ids_{};
};
static_assert(sizeof(Coordinator)<24576,"closed4rank/128row metadata reservation must remain bounded");

// The FIRST actual experimental TP4 cache integration must take this path on a
// negative coordinator/owner/cache result. No stream drain or C++ destructor is
// safe to infer from an aborted CPU gate. Returning errors below is for HOST
// executable-spec tests, NOT recoverable whole-Engine handling.
[[noreturn]] inline void process_fail_stop()noexcept{std::_Exit(88);}
inline void require_success_or_process_stop(int status)noexcept{if(status<0)process_fail_stop();}
} // namespace glm53::cache_prepare4
