#pragma once
// Observations only: no prediction, reads, cache admission or pointer publication.
#include "ep_route_counts.h"
#include <chrono>
#include <cstdint>

namespace rad {
inline constexpr const char* kM1PrefetchTraceEnv = "RADIANCE_EXPERIMENTAL_DS41_M1_PREFETCH_TRACE";
inline constexpr uint64_t kM1PrefetchTraceMaxLines = 16384;
inline bool m1_trace_opt_in(const char* p) noexcept {
    return p && p[0]=='1' && p[1]=='\0';
}
inline bool m1_trace_target(bool decode,bool draft,int64_t rows,int64_t sequences,
                            int64_t speculative,int64_t decode_rows,int64_t decode_sequences) noexcept {
    return decode && !draft && rows==1 && sequences==1 && speculative==0 &&
           decode_rows==1 && decode_sequences==1;
}
inline uint64_t m1_trace_now_ns() noexcept {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
// Copied into each reader job. No borrowed witness, count array or mutable
// caller-context pointer can outlive the existing HOST-readback scope.
struct M1PrefetchTrace {
    uint64_t generation=0, producer=0, ready_ns=0;
    int64_t rank=-1, layer=-1, expert_base=-1;
    bool enabled=false;
};
inline bool m1_trace_tag_valid(const M1PrefetchTrace& t) noexcept {
    return t.enabled && t.generation && t.producer && t.ready_ns &&
        t.rank>=0 && t.rank<4 && t.layer>=0 && t.layer<40 && t.expert_base==96*t.rank;
}
inline M1PrefetchTrace m1_trace_scope(bool requested, bool target_m1,
                                     const EpRouteCounts& v) noexcept {
    if(!requested || !target_m1 || !v.program || !v.stream || !v.row ||
       !v.successfully_issued || !v.generation || !v.producer_serial ||
       v.rows!=1 || v.world!=4 || v.rank<0 || v.rank>=4 ||
       v.layer<0 || v.layer>=40 || v.local_experts!=96 ||
       v.global_experts!=384 || v.top_k!=6 || v.expert_base!=v.rank*96) return {};
    return {v.generation,v.producer_serial,m1_trace_now_ns(),v.rank,v.layer,v.expert_base,true};
}
inline bool m1_trace_weight(const M1PrefetchTrace& t,int64_t layer,int64_t expert,
                            int64_t slot,int64_t bytes) noexcept {
    return m1_trace_tag_valid(t) && t.layer==layer && expert>=0 && expert<96 &&
           ((slot==0 && bytes==12533760) || (slot==1 && bytes==6266880));
}
} // namespace rad
