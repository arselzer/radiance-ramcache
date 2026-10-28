// Fixed96 immutable pinned DSpark pairs in two existing file-buffer tails.
// Private owner state only. No allocation, aliases, heat policy or device API here.
#pragma once
#include "rad_runtime.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rad {
inline constexpr char kDsparkHeadTailEnv[]="RADIANCE_EXPERIMENTAL_DS41_DSPARK_HEAD_FILE_TAIL";
enum class DsparkHeadTailPass { Unknown, Query, Verify, Context };
inline DsparkHeadTailPass dspark_head_tail_pass(const RadBatch* b) {
    if(!b||b->phase!=RAD_PHASE_DECODE||b->n_seq!=1)return DsparkHeadTailPass::Unknown;
    if(b->draft_pass==1&&b->n_tok==5&&b->n_spec==0&&b->n_seq_decode==0&&b->n_tok_decode==0)
        return DsparkHeadTailPass::Query;
    if(b->draft_pass==0&&b->n_tok>0&&b->n_tok<=6&&b->n_seq_decode==1&&
       b->n_tok_decode==b->n_tok&&b->n_spec>=0&&b->n_spec<b->n_tok)return DsparkHeadTailPass::Verify;
    if(b->draft_pass==-1&&b->n_tok>0&&b->n_tok<=6&&b->n_seq_decode==0&&
       b->n_tok_decode==0&&b->n_spec==0)return DsparkHeadTailPass::Context;
    return DsparkHeadTailPass::Unknown;
}
class DsparkHeadTailOwner {
public:
    static constexpr int64_t pair_bytes=18800640, pairs=96, max_verify_rows=6;
    static constexpr int64_t alignment=4096, guard=alignment;
    struct Budget {
        std::array<uintptr_t,2> base{},tail{};
        std::array<int64_t,2> capacity{},tail_capacity{};
        int64_t serving_bytes=0,head_bytes=0,tail_bytes=0;
    };
    struct Stats {uint64_t copies=0,revokes=0,copy_generations=0,keeps=0;};
    bool configure(const std::array<uintptr_t,2>&base,const std::array<int64_t,2>&cap,
                   int64_t required,int64_t largest,int count) {
        if(configured_||pending_||count!=pairs||largest!=pair_bytes||required!=pairs*pair_bytes)return false;
        Budget b;b.base=base;b.capacity=cap;b.head_bytes=required;
        // Guard on BOTH sides of each tail. Prefix includes valid zero-local dummy backing.
        b.serving_bytes=max_verify_rows*6*pair_bytes+guard;
        for(std::size_t i=0;i<2;++i){
            if(!base[i]||base[i]%alignment||cap[i]<=b.serving_bytes+guard||cap[i]>pairs*pair_bytes||cap[i]%alignment||
               uint64_t(cap[i])>std::numeric_limits<uintptr_t>::max()-base[i])return false;
            b.tail[i]=base[i]+uintptr_t(b.serving_bytes);
            b.tail_capacity[i]=(cap[i]-b.serving_bytes-guard)/pair_bytes*pair_bytes;
            b.tail_bytes+=b.tail_capacity[i];
        }
        if(!(base[0]+uintptr_t(cap[0])<=base[1]||base[1]+uintptr_t(cap[1])<=base[0])||
           b.tail_bytes<required)return false;
        budget_=b;configured_=true;return true;
    }
    bool configured()const{return configured_;}
    bool pending()const{return pending_;} // includes failed/partial enqueue; not completion evidence
    bool published()const{return published_;}
    const Budget& budget()const{return budget_;}
    const Stats& stats()const{return stats_;}
    bool selected_fits(int64_t selected,int64_t bytes)const{
        return configured_&&selected>=0&&selected<=36&&bytes>=0&&bytes<=budget_.serving_bytes-guard;
    }
    // Backend is a fixed private FileStager adapter; tests supply a fake event/transport owner.
    // drain_* must be checked completion proofs, not enqueue/status counters.
    template<class Backend> int revoke(Backend& b){
        if(!pending_)return RAD_OK;
        int rc=b.drain_copy();if(rc<0)return rc;
        rc=b.drain_compute();if(rc<0)return rc;
        b.withdraw(); // only now may residency point back to retained pinned copies
        pending_=published_=ordered_=false;stream_=0;++stats_.revokes;return RAD_OK;
    }
    template<class Backend> int ensure(Backend& b,uintptr_t stream){
        if(!configured_||!stream)return RAD_E_STATE;
        if(!b.epochs_current()){
            const int rc=revoke(b);return rc<0?rc:RAD_E_STATE;
        }
        if(pending_){
            if(!published_||!ordered_||stream_!=stream)return RAD_E_STATE;
            ++stats_.keeps;return RAD_OK;
        }
        // Previous file writes/readers in BOTH windows finish before either tail is overwritten.
        int rc=b.drain_copy();if(rc<0)return rc;
        rc=b.drain_compute();if(rc<0)return rc;
        pending_=true;stream_=stream;++stats_.copy_generations; // BEFORE a possibly partial enqueue
        rc=b.stage(budget_);if(rc<0)return rc;
        published_=true;
        if(!b.epochs_current())return RAD_E_STATE;
        rc=b.wait_done();if(rc<0)return rc;
        ordered_=true;++stats_.copies;return RAD_OK;
    }
private:
    Budget budget_{};Stats stats_{};
    bool configured_=false,pending_=false,published_=false,ordered_=false;
    uintptr_t stream_=0;
};
} //namespace rad
