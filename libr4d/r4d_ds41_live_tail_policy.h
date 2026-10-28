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
constexpr bool geometry(long long m,long long n,long long k,long long ne,long long top,long long total,long long ar,bool sorted){
 return eligible(m,n,k,ne,top)&&total==m*top&&ar==(sorted?total:m);
}
inline bool opt_in(const char*p){return p&&p[0]=='1'&&p[1]=='\0';}
DS41_TAIL_HD inline int clamp(int v,int total){return v<0?0:(v>total?total:v);}
// Only valid current monotonic prefixes are producer authority; clamps are defensive.
DS41_TAIL_HD inline void range(const int32_t*off,int e,int total,int*first,int*last){
 int a=e==0?0:clamp(off[e],total),b=clamp(off[e+1],total);b=b<a?a:b;*first=a;*last=b;
}
DS41_TAIL_HD inline int end(const int32_t*off,int ne,int total){return clamp(off[ne],total);}
}
#undef DS41_TAIL_HD
