/* Independent-reference byte cases, shared by host/device. No performance measurement. */
#pragma once
#include "../../arch/common/rad_ds41_kv_pair.h"
#if defined(__HIPCC__) || defined(__CUDACC__)
#define DS41_KVP_ORACLE __host__ __device__
#else
#define DS41_KVP_ORACLE
#endif
namespace ds41_kv_pair_oracle {
namespace ref = rad_ds41_kv_codec;
namespace pair = rad_ds41_kv_pair;
struct Counts { unsigned bad=0, even=0, odd=0, rejected=0; };
DS41_KVP_ORACLE inline unsigned next(unsigned& s) { s^=s<<13; s^=s>>17; s^=s<<5; return s; }
template<ref::Kind K,int W>
DS41_KVP_ORACLE inline void compare(const uint8_t* row,int col,Counts& c) {
    uint32_t out=0xA55ABEEFu;
    const int G=ref::group_size(K);
    uint16_t lo=0xA55A,hi=0xA55A;
    const bool a=ref::decode_group_value(K,row[W/2+col/G],row+(col/G)*(G/2),col%G,&lo);
    const bool b=ref::decode_group_value(K,row[W/2+(col+1)/G],row+((col+1)/G)*(G/2),(col+1)%G,&hi);
    const bool ok=pair::decode_row2<K,W>(row,col,&out);
    c.bad+=ok!=(a&&b);
    if(ok) {
        c.bad+=out!=((uint32_t)lo|((uint32_t)hi<<16));
        c.bad+=pair::decode_row2_bits<K,W>(row,col)!=out;
        if(col&1)++c.odd;else ++c.even;
    } else { ++c.rejected; c.bad+=out!=0xA55ABEEFu; }
}
template<ref::Kind K>
DS41_KVP_ORACLE inline Counts normal(unsigned at) {
    constexpr int G=K==ref::Kind::Fp4E4M3G16?16:32;
    uint8_t guarded[G/2+3]={};uint8_t* row=guarded+1;
    guarded[0]=guarded[G/2+2]=0xD5;
    row[0]=(uint8_t)at;row[G/2]=(uint8_t)(at>>8);
    Counts c;compare<K,G>(row,0,c);
    c.bad+=guarded[0]!=0xD5||guarded[G/2+2]!=0xD5;
    return c;
}
template<ref::Kind K>
DS41_KVP_ORACLE inline Counts literal(unsigned at) {
    constexpr int G=K==ref::Kind::Fp4E4M3G16?16:32;
    uint8_t guarded[G/2+3];uint8_t* row=guarded+1;unsigned seed=0xFE39A831u^at;
    for(int i=0;i<G/2+3;++i)guarded[i]=(uint8_t)next(seed);
    guarded[0]=guarded[G/2+2]=0xD5;
    const uint16_t a=(uint16_t)at,b=(uint16_t)(at^0xA55Au);
    row[0]=(uint8_t)a;row[1]=(uint8_t)(a>>8);row[2]=(uint8_t)b;row[3]=(uint8_t)(b>>8);
    row[4]=(uint8_t)((row[4]&~3u)|2u);row[G/2]=0;
    Counts c;compare<K,G>(row,0,c); // Every uint16 pattern is actually selected as both literals.
    compare<K,G>(row,1+2*(int)(at%(G/2-1)),c);
    c.bad+=guarded[0]!=0xD5||guarded[G/2+2]!=0xD5;
    return c;
}
template<ref::Kind K,int W>
DS41_KVP_ORACLE inline Counts rows(unsigned at) {
    constexpr int G=K==ref::Kind::Fp4E4M3G16?16:32,B=W/2+W/G;
    uint8_t guarded[B+2];uint8_t* row=guarded+1;unsigned seed=0x987123ABu^(at*0x9E3779B9u);
    for(int i=0;i<B+2;++i)guarded[i]=(uint8_t)next(seed);
    guarded[0]=guarded[B+1]=0xD5;
    for(int g=0;g<W/G;++g) {
        uint8_t s=(uint8_t)(1+next(seed)%(K==ref::Kind::Fp4E4M3G16?126:253));
        if((g+at)%3==0)s=0;
        if(K==ref::Kind::Fp4Pow2G32&&(g+at)%7==0)s=254;
        row[W/2+g]=s;
    }
    if(at%11==0)row[B-1]=K==ref::Kind::Fp4E4M3G16?127:255;
    if(at==1)for(int i=0;i<B;++i)row[i]=0;
    Counts c;
    // Includes all selector boundaries, every group crossing, and the final legal pair.
    for(int col=0;col<W-1;++col)compare<K,W>(row,col,c);
    c.bad+=guarded[0]!=0xD5||guarded[B+1]!=0xD5;
    return c;
}
template<ref::Kind K,int W>
DS41_KVP_ORACLE inline Counts invalid() {
    constexpr int G=K==ref::Kind::Fp4E4M3G16?16:32,B=W/2+W/G;
    uint8_t row[B]={};uint32_t out=0xA55ABEEFu;Counts c;
    const int cols[4]={-1,W-1,W,W+1};
    for(int j=0;j<4;++j) {
        const int col=cols[j];
        c.bad+=pair::decode_row2<K,W>(row,col,&out)||out!=0xA55ABEEFu;++c.rejected;
    }
    c.bad+=pair::decode_row2<K,W>(nullptr,0,&out)||out!=0xA55ABEEFu;++c.rejected;
    c.bad+=pair::decode_row2<K,W>(row,0,nullptr);++c.rejected;
    row[W/2+1]=K==ref::Kind::Fp4E4M3G16?127:255;
    c.bad+=pair::decode_row2<K,W>(row,G-1,&out)||out!=0xA55ABEEFu;++c.rejected;
    row[W/2]=row[W/2+1];
    c.bad+=pair::decode_row2<K,W>(row,0,&out)||out!=0xA55ABEEFu;++c.rejected;
    return c;
}
} // namespace ds41_kv_pair_oracle
#undef DS41_KVP_ORACLE
