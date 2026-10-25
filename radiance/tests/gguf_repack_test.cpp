/* Import real packed GGUF blocks through libquant, then reconstruct their bytes through the
 * host-only bridge. Golden numeric values come from an independent GGML decoder; arbitrary
 * block tests prove bit preservation, not validity/quality of arbitrary model values. */
#include "quant_fixture.h"
#include "../arch/common/rad_gguf_repack.h"
#include <nlohmann/json.hpp>
#include <array>
#include <bit>
#include <fstream>
#include <random>

namespace gg = rad::gguf;
namespace {
using Json = nlohmann::json;
struct Type { gg::Format format; const char* name; int64_t block_bytes; };
constexpr Type types[]{
    {gg::Format::iq2_xxs,"iq2_xxs",66}, {gg::Format::iq2_xs,"iq2_xs",74},
    {gg::Format::iq2_s,"iq2_s",82}, {gg::Format::iq3_xxs,"iq3_xxs",98}};
const Json& fixtures() {
    static const Json data=[] {
        std::ifstream file(std::string(RAD_GOLDEN_FIXTURES)+"/gguf-import.json");
        if(!file)throw std::runtime_error("missing GGUF oracle fixture");
        return Json::parse(file);
    }();
    return data;
}
const Json& fixture(const char* name) {
    for(const auto& f:fixtures().at("formats"))if(f.at("name")==name)return f;
    throw std::runtime_error(std::string("missing format ")+name);
}
std::vector<uint8_t> unhex(const std::string& text) {
    std::vector<uint8_t> out;out.reserve(text.size()/2);
    for(size_t i=0;i<text.size();i+=2)out.push_back(uint8_t(std::stoul(text.substr(i,2),nullptr,16)));
    return out;
}
void put16(uint8_t* p,uint16_t value) {p[0]=uint8_t(value);p[1]=uint8_t(value>>8);}
void bits(std::vector<uint8_t>& bytes,size_t at,unsigned width,uint32_t value) {
    for(unsigned bit=0;bit<width;++bit) {
        const size_t index=(at+bit)/8;const unsigned shift=unsigned((at+bit)%8);
        bytes.at(index)=uint8_t((bytes[index]&~(uint8_t(1)<<shift))|(((value>>bit)&1)<<shift));
    }
}
struct Imported {
    Type type;
    int64_t rows,cols;
    RadEncoding encoding{};
    std::vector<uint8_t> original;
    std::array<std::vector<uint8_t>,RAD_ENC_MAX_PLANES> storage;
    std::array<size_t,RAD_ENC_MAX_PLANES> strides{};
    Imported(Type t,int64_t r,int64_t c,std::vector<uint8_t> packed)
        :type(t),rows(r),cols(c),original(std::move(packed)) {
        const auto* row=qfix::reg().checkpoint_importer("gguf",type.name);
        if(!row)throw std::runtime_error("required GGUF importer is missing");
        RadCheckpointImportWeight weight{"repack-test",type.name,rows,cols};
        if(row->info->encoding(&weight,&encoding)!=RAD_OK)throw std::runtime_error("import encoding declined");
        void* data[RAD_ENC_MAX_PLANES]{};
        for(int p=0;p<encoding.n_planes;++p) {
            const auto size=rad_enc_plane_bytes(&encoding.plane[p],rows,cols);
            if(size<=0)throw std::runtime_error("invalid test plane size");
            storage[p].resize(size_t(size),0);data[p]=storage[p].data();
        }
        const int status=row->info->import_rows(&weight,original.data(),int64_t(original.size()),rows,data);
        if(status!=RAD_OK)throw std::runtime_error("packed test import failed");
    }
    gg::View view() const {
        gg::View result;result.encoding=encoding;result.rows=rows;result.cols=cols;
        for(int p=0;p<encoding.n_planes;++p)result.planes[p]={storage[p],strides[p]};
        return result;
    }
    int plane(const char* role) const {
        const int index=rad_enc_find(&encoding,role);
        if(index<0)throw std::runtime_error(std::string("missing plane ")+role);
        return index;
    }
    size_t row_bytes(int p) const {
        int64_t pr=0,pc=0;rad_enc_plane_dims(&encoding.plane[p],rows,cols,&pr,&pc);
        return size_t(rad_enc_row_bytes(encoding.plane[p].dtype,pc));
    }
    void set(int p,int64_t row,int64_t element,uint32_t value) {
        const size_t stride=strides[p]?strides[p]:row_bytes(p);
        const unsigned width=unsigned(rad_dtype_bits(encoding.plane[p].dtype));
        bits(storage[p],size_t(row)*stride*8+size_t(element)*width,width,value);
    }
    void pad_rows(size_t extra) {
        for(int p=0;p<encoding.n_planes;++p) {
            int64_t pr=0,pc=0;rad_enc_plane_dims(&encoding.plane[p],rows,cols,&pr,&pc);
            const size_t rb=row_bytes(p),stride=rb+extra;
            std::vector<uint8_t> next(size_t(pr)*stride,0xCD);
            for(int64_t row=0;row<pr;++row)std::memcpy(next.data()+size_t(row)*stride,storage[p].data()+size_t(row)*rb,rb);
            storage[p]=std::move(next);strides[p]=stride;
        }
    }
};
Imported golden(Type type) {
    const auto& f=fixture(type.name);
    return Imported(type,f.at("rows"),f.at("cols"),unhex(f.at("packed_hex")));
}
Imported random_blocks(Type type,int64_t rows,int64_t cols,uint32_t seed) {
    std::mt19937 rng(seed);std::vector<uint8_t> raw(size_t(rows*(cols/256)*type.block_bytes));
    for(auto& byte:raw)byte=uint8_t(rng());
    return Imported(type,rows,cols,std::move(raw));
}
std::vector<uint8_t> selected_raw(const Imported& data,const gg::Region& region) {
    std::vector<uint8_t> result;
    const int64_t row_bytes=(data.cols/256)*data.type.block_bytes;
    const int64_t take=(region.cols/256)*data.type.block_bytes;
    for(int64_t row=region.row;row<region.row+region.rows;++row) {
        const auto first=data.original.begin()+row*row_bytes+(region.col/256)*data.type.block_bytes;
        result.insert(result.end(),first,first+take);
    }
    return result;
}
void roundtrip(const Imported& data,const gg::Region& region) {
    gg::Layout layout;std::string why;
    REQUIRE_EQ(gg::plan(data.type.format,data.encoding,data.rows,data.cols,region,&layout,&why),RAD_OK);
    CHECK_EQ(layout.rows,region.rows);CHECK_EQ(layout.cols,region.cols);
    CHECK_EQ(layout.row_bytes,(region.cols/256)*data.type.block_bytes);
    CHECK_EQ(layout.bytes,layout.row_bytes*region.rows);REQUIRE(layout.tag);
    std::vector<uint8_t> output(size_t(layout.bytes)+48,0xA5);
    const auto view=data.view();
    REQUIRE_EQ(gg::repack(data.type.format,view,region,std::span(output).subspan(16,size_t(layout.bytes)+16),&why),RAD_OK);
    const auto expected=selected_raw(data,region);
    CHECK(std::equal(expected.begin(),expected.end(),output.begin()+16));
    CHECK(std::all_of(output.begin(),output.begin()+16,[](uint8_t b){return b==0xA5;}));
    CHECK(std::all_of(output.begin()+16+layout.bytes,output.end(),[](uint8_t b){return b==0xA5;}));
}
void rejected(gg::Format format,const gg::View& view,const gg::Region& region,size_t capacity) {
    std::vector<uint8_t> out(capacity+32,0x96);std::string why;
    CHECK(gg::repack(format,view,region,std::span(out).subspan(16,capacity),&why)<0);
    CHECK(std::all_of(out.begin(),out.end(),[](uint8_t b){return b==0x96;}));
}
void decode_golden(const Imported& data) {
    const auto& f=fixture(data.type.name);const auto expected_bytes=unhex(f.at("f32_hex"));
    std::vector<float> expected(expected_bytes.size()/4),decoded(expected.size());
    std::memcpy(expected.data(),expected_bytes.data(),expected_bytes.size());
    const void* ptr[RAD_ENC_MAX_PLANES]{};
    for(int p=0;p<data.encoding.n_planes;++p)ptr[p]=data.storage[p].data();
    rad::EncodedView encoded;rad::enc_view(data.encoding,data.rows,data.cols,ptr,&encoded);
    REQUIRE_EQ(rad::enc_decode_rows(encoded,0,data.rows,decoded.data()),RAD_OK);
    for(size_t i=0;i<expected.size();++i)CHECK_EQ(decoded[i],expected[i]);
}
}

TEST(gguf_iq_repack_golden_bytes_and_independent_numeric_values_agree) {
    for(const auto type:types) {
        const auto data=golden(type);roundtrip(data,{0,data.rows,0,data.cols});decode_golden(data);
        std::vector<uint8_t> raw(data.original.size());
        REQUIRE_EQ(gg::repack(type.format,data.view(),{0,data.rows,0,data.cols},raw),RAD_OK);
        const Imported again(type,data.rows,data.cols,raw);decode_golden(again);
        for(int p=0;p<data.encoding.n_planes;++p)CHECK(data.storage[p]==again.storage[p]);
    }
}
TEST(gguf_iq_repack_preserves_arbitrary_original_codes_and_half_scale_bits) {
    constexpr uint16_t scales[]{0,0x8000,1,0x8001,0x0400,0x8400,0x2800,0xA800,0x7BFF,0xFBFF,0x7C00,0xFC00,0x7E03,0xFE55};
    constexpr int64_t blocks[]{1,2,9,20};
    for(const auto type:types)for(uint32_t trial=0;trial<32;++trial) {
        const int64_t rows=1+trial%4,cols=256*blocks[trial%4];
        auto data=random_blocks(type,rows,cols,1017+trial);
        auto raw=data.original;
        for(int64_t block=0;block<rows*(cols/256);++block)put16(raw.data()+block*type.block_bytes,scales[(block+trial)%std::size(scales)]);
        const Imported varied(type,rows,cols,std::move(raw));roundtrip(varied,{0,rows,0,cols});
    }
}
TEST(gguf_iq_repack_exhaustively_inverts_the_parity_sign_indices) {
    for(const auto type:types) {
        if(type.format==gg::Format::iq2_s)continue;
        const auto base=random_blocks(type,1,256,109).original;
        for(unsigned sign=0;sign<128;++sign) {
            auto raw=base;
            if(type.format==gg::Format::iq2_xs)bits(raw,2*8+9,7,sign);
            else if(type.format==gg::Format::iq2_xxs)bits(raw,(2+4)*8,7,sign);
            else bits(raw,(2+64)*8,7,sign);
            const Imported data(type,1,256,std::move(raw));roundtrip(data,{0,1,0,256});
        }
    }
}
TEST(gguf_iq2_s_repack_preserves_all_independent_eight_bit_sign_patterns) {
    const auto type=types[2];const auto base=random_blocks(type,1,256,340).original;
    for(unsigned sign=0;sign<256;++sign) {
        auto raw=base;raw[2+32]=uint8_t(sign);
        const Imported data(type,1,256,std::move(raw));roundtrip(data,{0,1,0,256});
    }
}
TEST(gguf_iq_repack_crosses_u9_u10_bit_boundaries_and_scale_extrema) {
    for(const auto type:{types[1],types[2]}) {
        auto raw=random_blocks(type,1,256,983).original;
        for(unsigned entry=0;entry<32;++entry) {
            if(type.format==gg::Format::iq2_xs) {
                constexpr unsigned codes[]{0,1,255,256,511};
                const auto word=uint16_t(codes[entry%std::size(codes)]|((entry*37%128)<<9));
                put16(raw.data()+2+entry*2,word);
            } else {
                constexpr unsigned codes[]{0,1,255,256,511,512,767,768,1023};const auto code=codes[entry%std::size(codes)];
                raw[2+entry]=uint8_t(code);bits(raw,(2+64+entry/4)*8+(entry%4)*2,2,code>>8);
            }
        }
        const size_t scale=type.format==gg::Format::iq2_xs?2+64:2+64+8;
        for(size_t i=0;i<8;++i)raw[scale+i]=std::array<uint8_t,4>{0,15,240,255}[i%4];
        const Imported data(type,1,256,std::move(raw));roundtrip(data,{0,1,0,256});
    }
}
TEST(gguf_iq_repack_accepts_row_bands_expert_views_and_pitched_planes) {
    for(const auto type:types) {
        auto data=random_blocks(type,6,5120,718);roundtrip(data,{2,2,256,768});
        auto expert=data.view();expert.rows=2;
        for(int p=0;p<data.encoding.n_planes;++p)if(data.encoding.plane[p].kind==RAD_PLANE_TILED&&data.encoding.plane[p].block[0]==1)
            expert.planes[p].bytes=expert.planes[p].bytes.subspan(2*data.row_bytes(p),2*data.row_bytes(p));
        std::vector<uint8_t> output(size_t(2*20*type.block_bytes));
        REQUIRE_EQ(gg::repack(type.format,expert,{0,2,0,5120},output),RAD_OK);
        CHECK(output==selected_raw(data,{2,2,0,5120}));
        data.pad_rows(13);roundtrip(data,{1,4,512,1280});
    }
}
TEST(gguf_iq_repack_all_rotating_tp4_ranges_recover_original_blocks) {
    for(const auto type:types) {
        const auto data=random_blocks(type,2,2304,687);
        for(int layer=0;layer<40;++layer) {
            int64_t col=0;
            for(int rank=0;rank<4;++rank) {
                const int64_t width=rank==layer%4?768:512;roundtrip(data,{0,2,col,width});col+=width;
            }
            CHECK_EQ(col,2304);
        }
    }
}
TEST(gguf_iq_repack_rejects_partial_blocks_out_of_range_and_overflow_geometry) {
    const auto data=random_blocks(types[1],4,2304,211);const auto view=data.view();gg::Layout out;
    for(const auto region:{gg::Region{0,1,0,576},gg::Region{0,1,1,256},gg::Region{-1,1,0,256},
                          gg::Region{4,1,0,256},gg::Region{0,0,0,256},gg::Region{0,1,0,0},
                          gg::Region{0,1,0,2560},gg::Region{INT64_MAX,1,0,256},gg::Region{0,1,INT64_MAX,256}}) {
        CHECK(gg::plan(data.type.format,data.encoding,data.rows,data.cols,region,&out)<0);
        rejected(data.type.format,view,region,size_t(data.type.block_bytes*4));
    }
    CHECK(gg::plan(data.type.format,data.encoding,INT64_MAX,256,{0,INT64_MAX,0,256},&out)<0);
    CHECK(gg::plan(data.type.format,data.encoding,4,257,{0,1,0,256},&out)<0);
    CHECK(gg::plan(static_cast<gg::Format>(999),data.encoding,4,256,{0,1,0,256},&out)<0);
    CHECK(gg::plan(data.type.format,data.encoding,4,256,{0,1,0,256},nullptr)<0);
}
TEST(gguf_iq_repack_rejects_wrong_encoding_roles_types_transforms_and_kind) {
    const auto data=golden(types[1]);const auto correct=data.view();const gg::Region region{0,4,0,256};
    auto view=correct;rad_enc_copy_str(view.encoding.transform,"fwht256");rejected(data.type.format,view,region,data.original.size());
    view=correct;rad_enc_copy_str(view.encoding.scheme,"plain");rejected(data.type.format,view,region,data.original.size());
    view=correct;view.encoding.plane[0].dtype=RAD_U10;rejected(data.type.format,view,region,data.original.size());
    view=correct;view.encoding.plane[3].block[1]=32;rejected(data.type.format,view,region,data.original.size());
    view=correct;std::swap(view.encoding.plane[3],view.encoding.plane[4]);rejected(data.type.format,view,region,data.original.size());
    view=correct;--view.encoding.n_planes;rejected(data.type.format,view,region,data.original.size());
    rejected(gg::Format::iq2_s,correct,region,data.original.size());
}
TEST(gguf_iq_repack_validates_exact_grid_bytes_and_fixed_factor) {
    for(const auto type:types) {
        auto data=golden(type);data.storage[data.plane("grid")].back()^=1;
        rejected(type.format,data.view(),{0,data.rows,0,data.cols},data.original.size());
        data=golden(type);data.storage[data.plane("scale.2")][0]^=1; // one ULP is not the exact fixed scale
        rejected(type.format,data.view(),{0,data.rows,0,data.cols},data.original.size());
    }
}
TEST(gguf_iq_repack_rejects_even_zero_and_late_invalid_scale_codes_before_writing) {
    for(const auto type:types)for(uint32_t invalid:{0u,2u,30u}) {
        auto data=random_blocks(type,3,512,89);const int p=data.plane("scale");
        int64_t pr=0,pc=0;rad_enc_plane_dims(&data.encoding.plane[p],data.rows,data.cols,&pr,&pc);
        data.set(p,pr-1,pc-1,invalid);
        rejected(type.format,data.view(),{0,data.rows,0,data.cols},data.original.size());
    }
}
TEST(gguf_iq_repack_rejects_nonrepresentable_sign_parity_before_writing) {
    for(const auto type:types) {
        if(type.format==gg::Format::iq2_s)continue;
        auto data=random_blocks(type,3,512,112);data.storage[data.plane("signs")].back()^=1;
        rejected(type.format,data.view(),{0,data.rows,0,data.cols},data.original.size());
    }
}
TEST(gguf_iq_repack_validates_every_plane_capacity_and_pitch) {
    for(const auto type:types) {
        const auto data=golden(type);const auto base=data.view();const gg::Region all{0,data.rows,0,data.cols};
        for(int p=0;p<data.encoding.n_planes;++p) {
            auto view=base;view.planes[p].bytes={};rejected(type.format,view,all,data.original.size());
            view=base;view.planes[p].bytes=view.planes[p].bytes.first(view.planes[p].bytes.size()-1);
            rejected(type.format,view,all,data.original.size());
            view=base;view.planes[p].row_stride=data.row_bytes(p)-1;rejected(type.format,view,all,data.original.size());
        }
        auto view=base;view.planes[0].row_stride=SIZE_MAX;rejected(type.format,view,all,data.original.size());
        rejected(type.format,base,all,data.original.size()-1);
    }
}
TEST(gguf_iq_repack_refuses_output_overlapping_any_source_plane) {
    for(const auto type:types) {
        auto data=random_blocks(type,4,5120,67);const auto view=data.view();
        for(int p=0;p<data.encoding.n_planes;++p) {
            if(data.storage[p].size()<size_t(type.block_bytes))continue;
            const auto saved=data.storage[p];std::string why;
            CHECK(gg::repack(type.format,view,{0,1,0,256},std::span(data.storage[p]).first(size_t(type.block_bytes)),&why)<0);
            CHECK(data.storage[p]==saved);
        }
    }
}
RAD_TEST_MAIN()
