/* Default-off DSpark bank reuse. Pure transition state; no allocator, clock or GPU dependency.
 * The owner is the sole step thread under Scheduler's block-frame lock. */
#pragma once
#include "rad_types.h"
#include <cstdint>
#include <cstring>
#include <limits>
namespace rad { namespace draft_bank {
inline constexpr char kLendEnv[]="RADIANCE_EXPERIMENTAL_DS41_DSPARK_BANK_LEND";
inline constexpr char kBackupEnv[]="RADIANCE_EXPERIMENTAL_DS41_DSPARK_BANK_LEND_BACKUP_MIB";
inline constexpr char kGraceEnv[]="RADIANCE_EXPERIMENTAL_DS41_DSPARK_BANK_LEND_GRACE_MS";
inline constexpr char kCooldownEnv[]="RADIANCE_EXPERIMENTAL_DS41_DSPARK_BANK_LEND_COOLDOWN_MS";
inline constexpr char kPackedFillEnv[]="RADIANCE_EXPERIMENTAL_DS41_DSPARK_BANK_LEND_PACKED_FILL";
inline constexpr char kTimingsEnv[]="RADIANCE_EXPERIMENTAL_DS41_DSPARK_BANK_LEND_TIMINGS";
inline constexpr char kComplementWordsEnv[]="RADIANCE_EXPERIMENTAL_DS41_DSPARK_BANK_LEND_COMPLEMENT_WORDS";
struct Config {
    bool on=false;
    bool packed_fill=false;
    bool timings=false;
    bool complement_words=false;
    int64_t backup_cap=0,grace_ms=1000,cooldown_ms=1000;
    static bool number(const char* s,int64_t fallback,int64_t maximum,int64_t* out) {
        *out=fallback;if(!s)return true;if(!*s)return false;
        int64_t v=0;for(const char* p=s;*p;++p){if(*p<'0'||*p>'9'||v>(maximum-(*p-'0'))/10)return false;v=v*10+*p-'0';}
        *out=v;return true;
    }
    static int parse(const char* flag,const char* budget,const char* grace,const char* cooldown,Config* out,
                     const char* packed_fill=nullptr,const char* timings=nullptr,const char* complement_words=nullptr){
        if(!out)return RAD_E_INVAL;*out=Config{};
        if(packed_fill&&std::strcmp(packed_fill,"0")){
            if(std::strcmp(packed_fill,"1")||!flag||std::strcmp(flag,"1"))return RAD_E_INVAL;
            out->packed_fill=true;
        }
        if(timings&&std::strcmp(timings,"0")){
            if(std::strcmp(timings,"1")||!flag||std::strcmp(flag,"1"))return RAD_E_INVAL;
            out->timings=true;
        }
        if(complement_words&&std::strcmp(complement_words,"0")){
            if(std::strcmp(complement_words,"1")||!flag||std::strcmp(flag,"1"))return RAD_E_INVAL;
            out->complement_words=true;
        }
        if(!flag||!std::strcmp(flag,"0"))return RAD_OK;
        if(std::strcmp(flag,"1"))return RAD_E_INVAL;
        out->on=true;int64_t mib=0;
        if(!number(budget,0,std::numeric_limits<int64_t>::max()/(1ll<<20),&mib)||mib<=0||
           !number(grace,1000,600000,&out->grace_ms)||!number(cooldown,1000,600000,&out->cooldown_ms))return RAD_E_INVAL;
        out->backup_cap=mib*(1ll<<20);return RAD_OK;
    }
};
struct Work {
    int64_t step=-1;
    uint64_t epoch=0;
    uint64_t query_seq=0; // actual Scheduler request id, never a device seq_ids dereference
    bool deferred=false,solo_held=false;
    int64_t context_rows=0,query_rows=0;
    bool valid()const{return step>=0&&epoch>0&&context_rows>=0&&query_rows>=0&&((query_rows==0)==(query_seq==0));}
    bool lendable()const{return valid()&&deferred&&solo_held&&context_rows==0&&query_rows==0;}
};
enum class Action:uint8_t{None,Lend,Restore};
enum class Phase:uint8_t{Ready,Filling,Lent,Restoring,Failed};
class Controller {
public:
    explicit Controller(Config c={}):cfg_(c){}
    Phase phase()const{return phase_;}
    bool ready()const{return phase_==Phase::Ready;}
    bool lent()const{return phase_==Phase::Lent;}
    uint64_t epoch()const{return ticket_;}
    /* Duplicate frames are idempotent; a restore is never delayed by time hysteresis or an old
     * lend ticket. Only a fully inspected no-context/no-query solo-held frame may begin reuse. */
    int decide(const Work&w,int64_t now,Action*out){
        if(!out||!w.valid()||now<0)return RAD_E_INVAL;*out=Action::None;
        if(phase_==Phase::Failed)return RAD_E_STATE;
        if(phase_==Phase::Filling||phase_==Phase::Restoring)return RAD_E_STATE;
        if(w.epoch<seen_)return RAD_E_STATE;
        if(phase_==Phase::Lent&&!w.lendable()){ticket_=w.epoch;phase_=Phase::Restoring;*out=Action::Restore;seen_=w.epoch;return RAD_OK;}
        if(w.epoch==seen_)return RAD_OK;seen_=w.epoch;
        if(!cfg_.on||phase_==Phase::Lent)return RAD_OK;
        if(!w.lendable()){since_=-1;return RAD_OK;}
        if(since_<0)since_=now;
        if(now<since_||now<restored_)return RAD_E_STATE;
        if(now-since_<cfg_.grace_ms||now-restored_<cfg_.cooldown_ms)return RAD_OK;
        ticket_=w.epoch;phase_=Phase::Filling;*out=Action::Lend;return RAD_OK;
    }
    int finish(Action action,uint64_t ticket,bool success,int64_t now){
        if(ticket!=ticket_||now<0)return RAD_E_STATE;
        if(action==Action::Lend&&phase_==Phase::Filling){
            // A failed fill may report success=false ONLY AFTER a complete verified rollback.
            phase_=success?Phase::Lent:Phase::Ready;if(!success){since_=-1;restored_=now;}return RAD_OK;
        }
        if(action==Action::Restore&&phase_==Phase::Restoring){
            phase_=success?Phase::Ready:Phase::Failed;since_=-1;restored_=now;return success?RAD_OK:RAD_E_STATE;
        }
        return RAD_E_STATE;
    }
    void fatal(){phase_=Phase::Failed;}
    /* Idle/teardown still return bytes before a future query; same checked completion protocol. */
    int restore_idle(uint64_t epoch,Action*out){
        if(!out)return RAD_E_INVAL;*out=Action::None;
        if(phase_==Phase::Failed)return RAD_E_STATE;
        if(phase_==Phase::Ready){since_=-1;return RAD_OK;}
        if(phase_!=Phase::Lent)return RAD_E_STATE;
        ticket_=epoch?epoch:seen_;phase_=Phase::Restoring;*out=Action::Restore;return RAD_OK;
    }
private:
    Config cfg_;Phase phase_=Phase::Ready;int64_t since_=-1,restored_=0;
    uint64_t seen_=0,ticket_=0;
};
}} // namespace rad::draft_bank
