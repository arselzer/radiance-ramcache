/* Exact inverses of standard GGML IQ block-to-plane import. No floating arithmetic.
 * Grid/sign constants retain their upstream notice in rad_gguf_iq_tables.h.
 * Raw byte definitions: GGML 06938ac12, as libquant/lq_ggml.cpp unpack_iq* reads them. */
#pragma once
#include "rad_gguf_repack_base.h"
#include "rad_gguf_iq_tables.h"
namespace rad::gguf::detail {
inline int parity_index(uint8_t signs) {
    const unsigned index=signs&127u;
    return tables::ksigns_iq2xs[index]==signs?int(index):-1;
}
inline uint8_t grid_byte(Format format,int64_t row,int64_t col) {
    switch(format) {
    case Format::iq2_xxs:return uint8_t(tables::iq2xxs_grid[row]>>(8*col));
    case Format::iq2_xs:return uint8_t(tables::iq2xs_grid[row]>>(8*col));
    case Format::iq2_s:return uint8_t(tables::iq2s_grid[row]>>(8*col));
    case Format::iq3_xxs:return uint8_t(tables::iq3xxs_grid[row]>>(8*col));
    default:return 0;
    }
}
inline int iq_constants(Format format,const Reader& input,std::string* why) {
    const int64_t width=format==Format::iq3_xxs?4:8;
    const int64_t rows=format==Format::iq2_xs?512:format==Format::iq2_s?1024:256;
    if(input.rows(1)!=rows||input.cols(1)!=width)
        return refuse(RAD_E_DTYPE,"IQ grid geometry is not the source codebook",why);
    for(int64_t r=0;r<rows;++r)for(int64_t c=0;c<width;++c)
        if(input.bits(1,r,c)!=grid_byte(format,r,c))
            return refuse(RAD_E_DTYPE,"IQ grid contents differ from the standard source codebook",why);
    const uint32_t factor=format==Format::iq3_xxs?0x3e800000u:0x3e000000u;
    if(input.bits(5,0,0)!=factor)
        return refuse(RAD_E_DTYPE,"IQ fixed scale factor is not exactly 1/8 or 1/4",why);
    return RAD_OK;
}
inline int iq_validate(Format format,const Reader& input,const Region& region,std::string* why) {
    const int status=iq_constants(format,input,why);
    if(status!=RAD_OK)return status;
    const int64_t scale_group=(format==Format::iq2_xs||format==Format::iq2_s)?16:32;
    for(int64_t row=region.row;row<region.row+region.rows;++row) {
        for(int64_t group=region.col/scale_group;group<(region.col+region.cols)/scale_group;++group) {
            const auto scale=input.bits(3,row,group);
            if(scale<1||scale>31||!(scale&1u))
                return refuse(RAD_E_FORMAT,"IQ local scale is not an odd source code in 1..31",why);
        }
        if(format!=Format::iq2_s)
            for(int64_t group=region.col/8;group<(region.col+region.cols)/8;++group)
                if(parity_index(input.byte(2,row,size_t(group)))<0)
                    return refuse(RAD_E_FORMAT,"IQ sign byte is outside the source parity table",why);
    }
    return RAD_OK;
}
inline void iq_write(Format format,const Reader& input,const Region& region,uint8_t* dst) {
    const auto bytes=geometry(format)->bytes;
    for(int64_t row=region.row;row<region.row+region.rows;++row)
        for(int64_t col=region.col;col<region.col+region.cols;col+=256,dst+=bytes) {
            put16(dst,input.bits(4,row,col/256)); // retain every F16 bit, including signed zero
            if(format==Format::iq2_xxs) {
                for(int sub=0;sub<8;++sub) {
                    uint32_t codes=0,sign_scale=0;
                    for(int group=0;group<4;++group) {
                        const int64_t index=col/8+4*sub+group;
                        codes|=input.bits(0,row,index)<<(8*group);
                        sign_scale|=uint32_t(parity_index(input.byte(2,row,size_t(index))))<<(7*group);
                    }
                    sign_scale|=((input.bits(3,row,col/32+sub)-1)/2)<<28;
                    put32(dst+2+sub*8,codes);put32(dst+6+sub*8,sign_scale);
                }
            } else if(format==Format::iq2_xs) {
                for(int group=0;group<32;++group) {
                    const int64_t index=col/8+group;
                    const uint32_t packed=input.bits(0,row,index)|
                        (uint32_t(parity_index(input.byte(2,row,size_t(index))))<<9);
                    put16(dst+2+group*2,packed);
                }
                for(int sub=0;sub<8;++sub) {
                    const uint32_t lo=(input.bits(3,row,col/16+sub*2)-1)/2;
                    const uint32_t hi=(input.bits(3,row,col/16+sub*2+1)-1)/2;
                    dst[66+sub]=uint8_t(lo|(hi<<4));
                }
            } else if(format==Format::iq2_s) {
                std::memset(dst+2,0,80);
                for(int sub=0;sub<8;++sub) {
                    for(int group=0;group<4;++group) {
                        const int slot=4*sub+group;const int64_t index=col/8+slot;
                        const uint32_t code=input.bits(0,row,index);
                        dst[2+slot]=uint8_t(code);dst[34+slot]=input.byte(2,row,size_t(index));
                        dst[66+sub]|=uint8_t((code>>8)<<(2*group));
                    }
                    const uint32_t lo=(input.bits(3,row,col/16+sub*2)-1)/2;
                    const uint32_t hi=(input.bits(3,row,col/16+sub*2+1)-1)/2;
                    dst[74+sub]=uint8_t(lo|(hi<<4));
                }
            } else { // IQ3_XXS: two 4-value grid indices per sign byte
                for(int group=0;group<64;++group)dst[2+group]=uint8_t(input.bits(0,row,col/4+group));
                for(int sub=0;sub<8;++sub) {
                    uint32_t sign_scale=0;
                    for(int group=0;group<4;++group)
                        sign_scale|=uint32_t(parity_index(input.byte(2,row,size_t(col/8+4*sub+group))))<<(7*group);
                    sign_scale|=((input.bits(3,row,col/32+sub)-1)/2)<<28;
                    put32(dst+66+4*sub,sign_scale);
                }
            }
        }
}
} // namespace rad::gguf::detail
