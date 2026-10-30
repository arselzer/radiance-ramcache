#pragma once
#include <cstdint>
#if defined(__HIPCC__) || defined(__HIP__) || defined(__CUDACC__)
#define DS41_TAIL_HD __host__ __device__
#else
#define DS41_TAIL_HD
#endif
namespace ds41_live_tail {
constexpr bool eligible(long long m,long long n,long long k,long long ne,long long top){
 return m>=1&&m<=256&&ne==96&&top==6&&((n==64&&k==256)||(n==128&&k==512)||(n==4608&&k==5120)||(n==5120&&k==2304));
}
// The rows are one whole routing as moe_scatter writes it: total = m * top slots, the activation
// rows the tokens (token order) or the slots (sorted order). Only then is [off[ne], total) the
// slots routed to other ranks, which moe_scatter fills with -1.
constexpr bool whole(long long m,long long top,long long total,long long ar,bool sorted){
 return total==m*top&&ar==(sorted?total:m);
}
constexpr bool geometry(long long m,long long n,long long k,long long ne,long long top,long long total,long long ar,bool sorted){
 return eligible(m,n,k,ne,top)&&whole(m,top,total,ar,sorted);
}
inline bool opt_in(const char*p){return p&&p[0]=='1'&&p[1]=='\0';}
// The DSpark draft bank's grouped GEMMs (RADIANCE_EXPERIMENTAL_DS41_DRAFT_GEMV): its 128 experts
// are 32 a rank at EP4, top-3, with the trunk's gate_up and down shapes; a query pass is 5 rows a
// sequence. A shape of its own, so the switch never reaches the trunk's 96 / top-6 GEMMs, whose
// live tail stays ..._LIVE_TAIL's.
constexpr bool draft_eligible(long long m,long long n,long long k,long long ne,long long top){
 return m>=1&&m<=256&&ne==32&&top==3&&((n==4608&&k==5120)||(n==5120&&k==2304));
}
constexpr bool draft_geometry(long long m,long long n,long long k,long long ne,long long top,long long total,long long ar,bool sorted){
 return draft_eligible(m,n,k,ne,top)&&whole(m,top,total,ar,sorted);
}
// A call's modes as bits: kTail the live tail, kOnePass all of an expert's rows in one pass over
// its weights. The draft switch's "1" is both (the bank read in place from pinned host memory, the
// only placement DSpark serves today: dspark_draft_control_scope), "2" the live tail alone (a bank
// in VRAM, where re-reading a weight costs less than restaging the activations); else off.
enum : int { kTail = 1, kOnePass = 2 };
inline int draft_mode(const char*p){
 if(!p||!p[0]||p[1])return 0;
 return p[0]=='1'?(kTail|kOnePass):p[0]=='2'?kTail:0;
}
DS41_TAIL_HD inline int clamp(int v,int total){return v<0?0:(v>total?total:v);}
// Only valid current monotonic prefixes are producer authority; clamps are defensive.
DS41_TAIL_HD inline void range(const int32_t*off,int e,int total,int*first,int*last){
 int a=e==0?0:clamp(off[e],total),b=clamp(off[e+1],total);b=b<a?a:b;*first=a;*last=b;
}
DS41_TAIL_HD inline int end(const int32_t*off,int ne,int total){return clamp(off[ne],total);}
}
#undef DS41_TAIL_HD
