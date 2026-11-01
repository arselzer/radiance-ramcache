#ifndef RAD_DS41_FP8_RUNTIME_H
#define RAD_DS41_FP8_RUNTIME_H
#include <cstdint>
#include <cstring>
#if defined(__HIPCC__) || defined(__CUDACC__)
#define RAD_DS41_FP8_HD __host__ __device__
#else
#define RAD_DS41_FP8_HD
#endif

/* New aliases only. No old operation/schema gains a parameter, operand or buffer. */
namespace rad_ds41_fp8_runtime {
inline constexpr const char* kEnv="RADIANCE_EXPERIMENTAL_DS41_FP8_CONVERT";
inline constexpr const char* kAuditEnv="RADIANCE_EXPERIMENTAL_DS41_FP8_CONVERT_AUDIT";
inline constexpr const char* kDevice="ds41.fp8_convert.audit.device";
inline constexpr const char* kHost="ds41.fp8_convert.audit.host";
enum Op:int { Quant=0,Rope,OpCount };
enum Field:int { Fast=0,Fallback,Blocks,Overflow,FieldCount };
struct Slot { Op op; int width; };
inline constexpr Slot kSlots[]={{Quant,512},{Quant,8192},{Rope,8192}};
inline constexpr int kSlotCount=3,kWords=kSlotCount*FieldCount,kBytes=kWords*int(sizeof(uint64_t));
inline constexpr const char* kNames[OpCount]={"quant_roundtrip_fp8_convert","rope_rows_roundtrip_fp8_convert"};
inline constexpr const char* kRows[OpCount]={"ds41_quant_fp8_convert","ds41_rope_fp8_convert"};
inline bool parse(const char*s,bool*out) {
    if(!out)return false;
    if(!s||!std::strcmp(s,"0")){*out=false;return true;}
    if(!std::strcmp(s,"1")){*out=true;return true;}
    return false;
}
inline constexpr int slot(Op op,int64_t n) {
    for(int i=0;i<kSlotCount;++i)if(kSlots[i].op==op&&kSlots[i].width==n)return i;
    return -1;
}
inline constexpr bool quant_scope(int64_t n,int64_t block,int fmt,int scale) {
    return (n==512||n==8192)&&block==32&&fmt==0&&scale==0;
}
inline bool declare_quant(bool on,int64_t n,int64_t block,const char*fmt,const char*scale,bool capture) {
    return on&&!capture&&fmt&&scale&&!std::strcmp(fmt,"fp8_e4m3")&&
           !std::strcmp(scale,"ue8m0")&&quant_scope(n,block,0,0);
}
inline constexpr bool rope_scope(int64_t n,int64_t hd,int64_t half,int64_t off,int64_t pairs,
                                 bool gptj,float sign) {
    return n==8192&&hd==512&&half==32&&off==448&&pairs==512&&gptj&&sign==-1.0f;
}
/* Conversion classification is exactly the standalone d89 component's closed finite domain.
 * A NaN/Inf value, zero/subnormal/negative/nonfinite scale uses literal software conversion. */
RAD_DS41_FP8_HD inline constexpr bool hardware_domain(uint32_t value,uint32_t scale) {
    return (value&0x7fffffffu)<0x7f800000u&&scale>=0x00800000u&&scale<0x7f800000u;
}
inline constexpr bool valid_row(const uint64_t*p) {
    return p&&!p[Overflow]&&p[Fast]<=UINT64_MAX-p[Fallback]&&
           ((p[Fast]||p[Fallback])?p[Blocks]>0:p[Blocks]==0);
}
}
#undef RAD_DS41_FP8_HD
#endif
