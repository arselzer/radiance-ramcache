/* Native GGUF blocks -> canonical planes for host oracle checks. This inverse shares no
 * quantizer/plugin state and never converts scale bits through floating point. Every block
 * bit pattern is representable. All metadata, capacities and aliases are checked before any
 * writes. Active row bytes are initialized; caller-owned row padding/tails are preserved. */
#pragma once
#include "rad_gguf_repack.h"

namespace rad::gguf {
struct MutablePlane {
    std::span<uint8_t> bytes;
    size_t row_stride = 0;
};
struct MutableView {
    RadEncoding encoding{};
    int64_t rows = 0, cols = 0;
    std::array<MutablePlane,RAD_ENC_MAX_PLANES> planes{};
    View read_view() const {
        View out;out.encoding=encoding;out.rows=rows;out.cols=cols;
        for(int p=0;p<RAD_ENC_MAX_PLANES;++p)
            out.planes[p]={planes[p].bytes,planes[p].row_stride};
        return out;
    }
};
namespace detail {
inline uint32_t get16(const uint8_t* p) {return uint32_t(p[0])|(uint32_t(p[1])<<8);}
inline uint32_t get32(const uint8_t* p) {
    return get16(p)|(get16(p+2)<<16);
}
class Writer {
public:
    explicit Writer(const MutableView& view):view_(view) {}
    int init(std::string* why) {
        const auto read_view=view_.read_view();Reader reader(read_view);
        const int status=reader.init(why);if(status!=RAD_OK)return status;
        for(int p=0;p<view_.encoding.n_planes;++p) {
            rows_[p]=reader.rows(p);cols_[p]=reader.cols(p);
            widths_[p]=unsigned(rad_dtype_bits(view_.encoding.plane[p].dtype));
            row_bytes_[p]=size_t((uint64_t(cols_[p])*widths_[p]+7)/8);
            strides_[p]=view_.planes[p].row_stride?view_.planes[p].row_stride:row_bytes_[p];
        }
        return RAD_OK;
    }
    void clear() {
        for(int p=0;p<view_.encoding.n_planes;++p)
            for(int64_t r=0;r<rows_[p];++r)
                std::memset(view_.planes[p].bytes.data()+size_t(r)*strides_[p],0,row_bytes_[p]);
    }
    void bits(int p,int64_t row,int64_t element,uint32_t value) {
        const uint64_t bit=uint64_t(element)*widths_[p];
        auto* dst=view_.planes[p].bytes.data()+size_t(row)*strides_[p]+size_t(bit/8);
        const unsigned shift=unsigned(bit%8),count=(shift+widths_[p]+7)/8;
        const uint64_t mask=((uint64_t(1)<<widths_[p])-1)<<shift;
        uint64_t current=0;for(unsigned i=0;i<count;++i)current|=uint64_t(dst[i])<<(8*i);
        current=(current&~mask)|((uint64_t(value)<<shift)&mask);
        for(unsigned i=0;i<count;++i)dst[i]=uint8_t(current>>(8*i));
    }
    void byte(int p,int64_t row,size_t offset,uint8_t value) {
        view_.planes[p].bytes[size_t(row)*strides_[p]+offset]=value;
    }
private:
    const MutableView& view_;
    std::array<int64_t,RAD_ENC_MAX_PLANES> rows_{},cols_{};
    std::array<size_t,RAD_ENC_MAX_PLANES> row_bytes_{},strides_{};
    std::array<unsigned,RAD_ENC_MAX_PLANES> widths_{};
};
inline void unrepack_iq(Format format,const uint8_t* src,const MutableView& view,Writer& out) {
    const int grid_rows=format==Format::iq2_xs?512:format==Format::iq2_s?1024:256;
    const int grid_cols=format==Format::iq3_xxs?4:8;
    for(int r=0;r<grid_rows;++r)for(int c=0;c<grid_cols;++c)
        out.bits(1,r,c,grid_byte(format,r,c));
    out.bits(5,0,0,format==Format::iq3_xxs?0x3e800000u:0x3e000000u);
    for(int64_t r=0;r<view.rows;++r)for(int64_t c=0;c<view.cols;c+=256) {
        out.bits(4,r,c/256,get16(src));
        if(format==Format::iq2_xxs) {
            for(int sub=0;sub<8;++sub) {
                const uint32_t signs_scale=get32(src+6+sub*8);
                out.bits(3,r,c/32+sub,2*(signs_scale>>28)+1);
                for(int group=0;group<4;++group) {
                    const int64_t index=c/8+4*sub+group;
                    out.bits(0,r,index,src[2+sub*8+group]);
                    out.byte(2,r,size_t(index),tables::ksigns_iq2xs[(signs_scale>>(7*group))&127u]);
                }
            }
        } else if(format==Format::iq2_xs) {
            for(int group=0;group<32;++group) {
                const uint32_t value=get16(src+2+2*group);const int64_t index=c/8+group;
                out.bits(0,r,index,value&511u);
                out.byte(2,r,size_t(index),tables::ksigns_iq2xs[value>>9]);
            }
            for(int sub=0;sub<16;++sub)
                out.bits(3,r,c/16+sub,2*((src[66+sub/2]>>(4*(sub%2)))&15u)+1);
        } else if(format==Format::iq2_s) {
            for(int group=0;group<32;++group) {
                const int64_t index=c/8+group;
                const uint32_t high=(src[66+group/4]>>(2*(group%4)))&3u;
                out.bits(0,r,index,uint32_t(src[2+group])|(high<<8));
                out.byte(2,r,size_t(index),src[34+group]);
            }
            for(int sub=0;sub<16;++sub)
                out.bits(3,r,c/16+sub,2*((src[74+sub/2]>>(4*(sub%2)))&15u)+1);
        } else { // IQ3_XXS
            for(int group=0;group<64;++group)out.bits(0,r,c/4+group,src[2+group]);
            for(int sub=0;sub<8;++sub) {
                const uint32_t signs_scale=get32(src+66+4*sub);
                out.bits(3,r,c/32+sub,2*(signs_scale>>28)+1);
                for(int group=0;group<4;++group)
                    out.byte(2,r,size_t(c/8+4*sub+group),tables::ksigns_iq2xs[(signs_scale>>(7*group))&127u]);
            }
        }
        src+=geometry(format)->bytes;
    }
}
inline void unrepack_dense(Format format,const uint8_t* src,const MutableView& view,Writer& out) {
    const auto* g=geometry(format);
    for(int64_t r=0;r<view.rows;++r)for(int64_t c=0;c<view.cols;c+=g->values) {
        if(format==Format::q8_0) {
            out.bits(1,r,c/32,get16(src));
            for(int i=0;i<32;++i)out.bits(0,r,c+i,src[2+i]);
        } else { // Q5_K
            out.bits(2,r,c/256,get16(src));out.bits(4,r,c/256,get16(src+2));
            const auto* scales=src+4;
            for(int i=0;i<4;++i) {
                out.bits(1,r,c/32+i,scales[i]&63u);
                out.bits(3,r,c/32+i,scales[4+i]&63u);
                out.bits(1,r,c/32+i+4,(scales[8+i]&15u)|((scales[i]>>6)<<4));
                out.bits(3,r,c/32+i+4,(scales[8+i]>>4)|((scales[4+i]>>6)<<4));
            }
            for(int i=0;i<256;++i) {
                const int low_index=32*(i/64)+i%32;
                const uint32_t low=(src[48+low_index]>>(i%64<32?0:4))&15u;
                const uint32_t high=(src[16+i%32]>>(i/32))&1u;
                out.bits(0,r,c+i,low|(high<<4));
            }
        }
        src+=g->bytes;
    }
}
} // namespace detail
inline int unrepack(Format format,std::span<const uint8_t> input,const MutableView& output,
                      std::string* why=nullptr) {
    Layout layout;int status=plan(format,output.encoding,output.rows,output.cols,
                                  {0,output.rows,0,output.cols},&layout,why);
    if(status!=RAD_OK)return status;
    if(uint64_t(layout.bytes)>SIZE_MAX||uint64_t(layout.bytes)>input.size()||!input.data())
        return refuse(RAD_E_SHAPE,"packed input is missing or truncated",why);
    detail::Writer writer(output);status=writer.init(why);if(status!=RAD_OK)return status;
    for(int p=0;p<output.encoding.n_planes;++p) {
        if(detail::overlaps(output.planes[p].bytes,input))
            return refuse(RAD_E_INVAL,"canonical output overlaps packed input",why);
        for(int q=0;q<p;++q)
            if(detail::overlaps(output.planes[p].bytes,output.planes[q].bytes))
                return refuse(RAD_E_INVAL,"canonical output planes overlap each other",why);
    }
    writer.clear();
    if(format==Format::q5_k||format==Format::q8_0)
        detail::unrepack_dense(format,input.data(),output,writer);
    else detail::unrepack_iq(format,input.data(),output,writer);
    if(why)why->clear();
    return RAD_OK;
}
} // namespace rad::gguf
