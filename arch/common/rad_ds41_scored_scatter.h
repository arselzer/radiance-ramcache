/* Each Model parses once at declare; no environment lookup in issue/kernel code. */
#pragma once
#include <cstring>
namespace rad_ds41_scored_scatter {
inline constexpr const char* kEnv="RADIANCE_EXPERIMENTAL_DS41_SCORED_SCATTER";
inline constexpr const char* kOp="ds41_scored_topk_scatter";
inline constexpr const char* kRow="ds41_scored_topk_scatter_m1";
inline constexpr const char* kSymbol="r4d_rad_ds41_scored_topk_scatter";
inline bool parse(const char*s,bool*out) {
    if(!out)return false;
    if(!s||!*s||!std::strcmp(s,"0")){*out=false;return true;}
    if(!std::strcmp(s,"1")){*out=true;return true;}
    return false;
}
inline bool scope(int world,int rank,long long rows,long long experts,long long local,
                  long long topk,long long base,bool scored,bool ungated) {
    return world==4&&rank>=0&&rank<4&&rows>=1&&experts==384&&local==96&&topk==6&&
           base==rank*96&&scored&&ungated;
}
}
