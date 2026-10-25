/* Host-only lossless canonical-plane -> standard GGUF block bridge.
 * No quantization, float-value expansion, plugin linkage, or GPU execution.
 * This is shared host code for future kernel relayouts, not a registered GPU layout. */
#pragma once
#include "rad_encoding.h"
#include "rad_abi.h"
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <string>

namespace rad::gguf {
enum class Format { iq2_xxs, iq2_xs, iq2_s, iq3_xxs, q5_k, q8_0 };
struct Plane {
    std::span<const uint8_t> bytes;
    size_t row_stride = 0; // zero selects tightly packed canonical rows
};
struct View {
    RadEncoding encoding{};
    int64_t rows = 0, cols = 0;
    std::array<Plane,RAD_ENC_MAX_PLANES> planes{};
};
struct Region { int64_t row = 0, rows = 0, col = 0, cols = 0; };
struct Layout {
    Format format{};
    int64_t rows = 0, cols = 0, row_bytes = 0, bytes = 0;
    const char* tag = nullptr;
};
struct Geometry { const char* name; int64_t values, bytes; const char* tag; };
inline const Geometry* geometry(Format format) {
    static constexpr Geometry types[]{
        {"iq2_xxs",256,66,"gguf.iq2_xxs.v1"}, {"iq2_xs",256,74,"gguf.iq2_xs.v1"},
        {"iq2_s",256,82,"gguf.iq2_s.v1"}, {"iq3_xxs",256,98,"gguf.iq3_xxs.v1"},
        {"q5_K",256,176,"gguf.q5_K.v1"}, {"q8_0",32,34,"gguf.q8_0.v1"}};
    const auto i=static_cast<unsigned>(format);
    return i<std::size(types)?&types[i]:nullptr;
}
inline int refuse(int code,const char* message,std::string* why) {
    if(why)*why=message;
    return code;
}
inline RadEncoding canonical_encoding(Format format) {
    RadEncoding e{};rad_enc_clear(&e);rad_enc_copy_str(e.scheme,"affine");
    auto p=[&](const char* role,uint32_t dt,int64_t bc) {rad_enc_add_plane(&e,role,dt,1,bc);};
    switch(format) {
    case Format::iq2_xxs:case Format::iq2_xs:case Format::iq2_s:case Format::iq3_xxs: {
        const bool iq3=format==Format::iq3_xxs;
        const int n=format==Format::iq2_xs?512:format==Format::iq2_s?1024:256;
        p("codes",n==512?RAD_U9:n==1024?RAD_U10:RAD_U8,iq3?4:8);
        rad_enc_add_table(&e,"grid",RAD_U8,n,iq3?4:8);
        p("signs",RAD_U1,1);p("scale",RAD_U5,(format==Format::iq2_xs||format==Format::iq2_s)?16:32);
        p("scale.1",RAD_F16,256);rad_enc_add_plane(&e,"scale.2",RAD_F32,0,0);break;
    }
    case Format::q5_k:
        p("codes",RAD_U5,1);p("scale",RAD_U6,32);p("scale.1",RAD_F16,256);
        p("min",RAD_U6,32);p("min.1",RAD_F16,256);break;
    case Format::q8_0:p("codes",RAD_I8,1);p("scale",RAD_F16,32);break;
    default:rad_enc_clear(&e);break;
    }
    return e;
}
inline int plan(Format format,const RadEncoding& encoding,int64_t source_rows,int64_t source_cols,
                  const Region& region,Layout* out,std::string* why=nullptr) {
    const auto* g=geometry(format);
    if(!g||!out)return refuse(RAD_E_INVAL,"unknown block format or missing layout destination",why);
    const auto expected=canonical_encoding(format);
    // Exact identity first: generic validation also handles unrelated transform metadata whose
    // arithmetic is not this adapter's contract. Reject it without interpreting those fields.
    if(!rad_enc_equal(&encoding,&expected)||!rad_enc_valid(&encoding))
        return refuse(RAD_E_DTYPE,"canonical encoding is not the exact requested GGUF form",why);
    if(source_rows<=0||source_cols<=0||source_cols%g->values||
       source_rows>INT64_MAX/source_cols || region.row<0 || region.rows<=0 ||
       region.row>source_rows || region.rows>source_rows-region.row || region.col<0 || region.cols<=0 ||
       region.col>source_cols || region.cols>source_cols-region.col ||
       region.col%g->values || region.cols%g->values)
        return refuse(RAD_E_SHAPE,"block region is out of range, overflows or cuts a source block",why);
    const int64_t blocks=region.cols/g->values;
    if(blocks>INT64_MAX/g->bytes)return refuse(RAD_E_SHAPE,"packed row extent overflows",why);
    const int64_t rb=blocks*g->bytes;
    if(region.rows>INT64_MAX/rb)return refuse(RAD_E_SHAPE,"packed matrix extent overflows",why);
    *out={format,region.rows,region.cols,rb,region.rows*rb,g->tag};
    if(why)why->clear();
    return RAD_OK;
}

/* Reader is initialized only after metadata planning succeeds. Every plane is checked against
 * its logical extent and byte pitch before bits() may read it. Tables/constants use full views. */
class Reader {
public:
    explicit Reader(const View& view):view_(view) {}
    int init(std::string* why=nullptr) {
        ready_=false;
        if(!rad_enc_valid(&view_.encoding)||view_.rows<=0||view_.cols<=0||view_.rows>INT64_MAX/view_.cols)
            return refuse(RAD_E_SHAPE,"invalid canonical matrix extent",why);
        for(int p=0;p<view_.encoding.n_planes;++p) {
            const auto& e=view_.encoding.plane[p];
            int64_t rows=0,cols=0;rad_enc_plane_dims(&e,view_.rows,view_.cols,&rows,&cols);
            const int width=rad_dtype_bits(e.dtype);
            if(rows<=0||cols<=0||width<=0||width>32||cols>(INT64_MAX-7)/width)
                return refuse(RAD_E_SHAPE,"canonical plane extent overflows",why);
            const uint64_t rb=(uint64_t(cols)*unsigned(width)+7)/8;
            const auto& plane=view_.planes[p];const uint64_t stride=plane.row_stride?plane.row_stride:rb;
            if(stride<rb||uint64_t(rows-1)>(UINT64_MAX-rb)/stride)
                return refuse(RAD_E_SHAPE,"canonical plane pitch or extent overflows",why);
            const uint64_t required=uint64_t(rows-1)*stride+rb;
            if(required>SIZE_MAX || required>plane.bytes.size() || !plane.bytes.data())
                return refuse(RAD_E_SHAPE,"canonical plane is missing or truncated",why);
            rows_[p]=rows;cols_[p]=cols;strides_[p]=size_t(stride);widths_[p]=unsigned(width);
        }
        ready_=true;return RAD_OK;
    }
    uint32_t bits(int plane,int64_t row,int64_t element) const {
        const uint64_t bit=uint64_t(element)*widths_[plane];
        const auto* src=view_.planes[plane].bytes.data()+size_t(row)*strides_[plane]+size_t(bit/8);
        const unsigned shift=unsigned(bit%8),count=(shift+widths_[plane]+7)/8;
        uint64_t value=0;for(unsigned i=0;i<count;++i)value|=uint64_t(src[i])<<(8*i);
        const uint64_t mask=(uint64_t(1)<<widths_[plane])-1;
        return uint32_t((value>>shift)&mask);
    }
    uint8_t byte(int plane,int64_t row,size_t offset) const {
        return view_.planes[plane].bytes[size_t(row)*strides_[plane]+offset];
    }
    int64_t rows(int plane) const {return rows_[plane];}
    int64_t cols(int plane) const {return cols_[plane];}
    const View& view() const {return view_;}
    bool ready() const {return ready_;}
private:
    const View& view_;
    std::array<int64_t,RAD_ENC_MAX_PLANES> rows_{},cols_{};
    std::array<size_t,RAD_ENC_MAX_PLANES> strides_{};
    std::array<unsigned,RAD_ENC_MAX_PLANES> widths_{};
    bool ready_=false;
};
namespace detail {
inline void put16(uint8_t* dst,uint32_t value) {dst[0]=uint8_t(value);dst[1]=uint8_t(value>>8);}
inline void put32(uint8_t* dst,uint32_t value) {for(int i=0;i<4;++i)dst[i]=uint8_t(value>>(8*i));}
inline bool overlaps(std::span<uint8_t> output,std::span<const uint8_t> input) {
    if(output.empty()||input.empty())return false;
    const auto a=reinterpret_cast<uintptr_t>(output.data()),b=reinterpret_cast<uintptr_t>(input.data());
    // Subtraction avoids overflowing either end address; no unrelated-pointer comparison.
    return a<=b?b-a<output.size():a-b<input.size();
}
} // namespace detail
} // namespace rad::gguf
