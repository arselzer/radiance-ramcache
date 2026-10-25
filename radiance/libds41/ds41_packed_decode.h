/* Scalar native GGUF decoders shared by the experimental GPU baseline and CPU tests.
 * Block definitions: GGML 06938ac12, corresponding to libquant/lq_ggml.cpp.
 * No float weight expansion is stored. No activation or cache quantization is introduced.
 * Tables must address the standard grids/signs for the requested format. */
#pragma once
#include <cstdint>
#if defined(__HIPCC__)
#define DS41_HD __attribute__((host,device,always_inline)) inline
#else
#define DS41_HD inline
#endif

namespace rad::ds41::packed {
enum class Type {iq2_xxs,iq2_xs,iq2_s,iq3_xxs,q5_k,q8_0};
struct Tables {
    const uint64_t* iq2_xxs;
    const uint64_t* iq2_xs;
    const uint64_t* iq2_s;
    const uint32_t* iq3_xxs;
    const uint8_t* signs;
};
DS41_HD constexpr int block_values(Type t) {
    return static_cast<unsigned>(t)<6?(t==Type::q8_0?32:256):0;
}
DS41_HD constexpr int block_bytes(Type t) {
    switch(t) {
    case Type::iq2_xxs:return 66;case Type::iq2_xs:return 74;case Type::iq2_s:return 82;
    case Type::iq3_xxs:return 98;case Type::q5_k:return 176;case Type::q8_0:return 34;
    default:return 0;
    }
}
DS41_HD uint32_t load16(const uint8_t* p) {return uint32_t(p[0])|(uint32_t(p[1])<<8);}
DS41_HD uint32_t load32(const uint8_t* p) {return load16(p)|(load16(p+2)<<16);}
DS41_HD float from_bits(uint32_t bits) {
    float out;__builtin_memcpy(&out,&bits,sizeof out);return out;
}
DS41_HD uint32_t float_bits(float value) {
    uint32_t out;__builtin_memcpy(&out,&value,sizeof out);return out;
}
// Match abi/rad_plugin.h's deterministic round-to-nearest-even storage contract.
DS41_HD uint16_t to_bf16(float value) {
    const uint32_t u=float_bits(value);
    if((u&0x7fffffffu)>0x7f800000u)return uint16_t((u>>16)|0x40u);
    return uint16_t((u+0x7fffu+((u>>16)&1u))>>16);
}
DS41_HD uint16_t to_half(float value) {
    const uint32_t u=float_bits(value),s=(u>>16)&0x8000u,a=u&0x7fffffffu;
    if(a>=0x7f800000u)return uint16_t(s|0x7c00u|(a>0x7f800000u?((a>>13)&1023u)|0x200u:0u));
    if(a>=0x47800000u)return uint16_t(s|0x7c00u);
    if(a<0x33000000u)return uint16_t(s);
    const int e=int(a>>23)-127;uint32_t m=a&0x7fffffu;
    if(e<-14) {
        const int shift=-14-e+13;m|=0x800000u;
        uint32_t q=m>>shift;const uint32_t lo=m&((1u<<shift)-1),tie=1u<<(shift-1);
        if(lo>tie||(lo==tie&&(q&1u)))++q;
        return uint16_t(s|q);
    }
    uint32_t q=m>>13,he=uint32_t(e+15);const uint32_t lo=m&8191u;
    if(lo>4096u||(lo==4096u&&(q&1u)))++q;
    if(q==1024u) {q=0;++he;}
    return uint16_t(s|(he>=31?0x7c00u:(he<<10)|q));
}
DS41_HD float half(uint32_t h) {
    const uint32_t sign=(h&0x8000u)<<16,exponent=(h>>10)&31u;
    uint32_t fraction=h&1023u,bits=sign;
    if(exponent==31)bits|=0x7f800000u|(fraction<<13);
    else if(exponent)bits|=((exponent+112u)<<23)|(fraction<<13);
    else if(fraction) {
        int e=-14;
        while(!(fraction&1024u)) {fraction<<=1;--e;}
        bits|=(uint32_t(e+127)<<23)|((fraction&1023u)<<13);
    }
    return from_bits(bits);
}
DS41_HD float decode(Type type,const uint8_t* block,int element,const Tables& table) {
    if(!block||element<0||element>=block_values(type))return from_bits(0x7fc00000u);
    const float d=half(load16(block));
    if(type==Type::q8_0) {
        const int value=block[2+element]<128?int(block[2+element]):int(block[2+element])-256;
        return d*float(value);
    }
    if(type==Type::q5_k) {
        const int group=element/32,local=group%4;
        const uint8_t* s=block+4;
        const unsigned scale=group<4?s[group]&63u:(s[8+local]&15u)|((s[local]>>6)<<4);
        const unsigned minimum=group<4?s[4+group]&63u:(s[8+local]>>4)|((s[4+local]>>6)<<4);
        const unsigned low=(block[48+32*(element/64)+element%32]>>(element%64<32?0:4))&15u;
        const unsigned high=(block[16+element%32]>>(element/32))&1u;
        return (d*float(scale))*float(low|(high<<4))-half(load16(block+2))*float(minimum);
    }
    unsigned magnitude=0,negative=0,odd_scale=0;
    if(type==Type::iq2_xxs) {
        if(!table.iq2_xxs||!table.signs)return from_bits(0x7fc00000u);
        const int sub=element/32,group=(element%32)/8;
        const uint32_t ss=load32(block+6+8*sub);
        const uint64_t grid=table.iq2_xxs[block[2+8*sub+group]];
        magnitude=unsigned((grid>>(8*(element%8)))&255u);
        negative=(table.signs[(ss>>(7*group))&127u]>>(element%8))&1u;
        odd_scale=2*(ss>>28)+1;
    } else if(type==Type::iq2_xs) {
        if(!table.iq2_xs||!table.signs)return from_bits(0x7fc00000u);
        const int group=element/8;const uint32_t qs=load16(block+2+2*group);
        const uint64_t grid=table.iq2_xs[qs&511u];
        magnitude=unsigned((grid>>(8*(element%8)))&255u);
        negative=(table.signs[qs>>9]>>(element%8))&1u;
        const int sub=element/16;odd_scale=2*((block[66+sub/2]>>(4*(sub%2)))&15u)+1;
    } else if(type==Type::iq2_s) {
        if(!table.iq2_s)return from_bits(0x7fc00000u);
        const int group=element/8;
        const unsigned high=(block[66+group/4]>>(2*(group%4)))&3u;
        const uint64_t grid=table.iq2_s[unsigned(block[2+group])|(high<<8)];
        magnitude=unsigned((grid>>(8*(element%8)))&255u);
        negative=(block[34+group]>>(element%8))&1u;
        const int sub=element/16;odd_scale=2*((block[74+sub/2]>>(4*(sub%2)))&15u)+1;
    } else { // IQ3_XXS
        if(!table.iq3_xxs||!table.signs)return from_bits(0x7fc00000u);
        const int sub=element/32,group=(element%32)/8;
        const uint32_t ss=load32(block+66+4*sub);
        const uint32_t grid=table.iq3_xxs[block[2+element/4]];
        magnitude=(grid>>(8*(element%4)))&255u;
        negative=(table.signs[(ss>>(7*group))&127u]>>(element%8))&1u;
        odd_scale=2*(ss>>28)+1;
    }
    const float scale=d*(float(odd_scale)*(type==Type::iq3_xxs?0.25f:0.125f));
    return scale*float(negative?-int(magnitude):int(magnitude));
}
} // namespace rad::ds41::packed
#undef DS41_HD
