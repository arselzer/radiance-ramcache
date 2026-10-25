/* CPU oracle for the GPU-shared native block decoder. Expected values come from independent
 * GGML golden blocks, or libquant's importer and the core's generic canonical-plane decoder.
 * No new packed decoder is used to generate expectations; no model payloads or GPU work. */
#include "quant_fixture.h"
#include "../libds41/ds41_packed_decode.h"
#include "../arch/common/rad_gguf_iq_tables.h"
#include <nlohmann/json.hpp>
#include <array>
#include <fstream>
#include <random>

namespace packed=rad::ds41::packed;
namespace {
struct Format {packed::Type type;const char* name;int values,bytes;};
constexpr Format formats[]{
    {packed::Type::iq2_xxs,"iq2_xxs",256,66},{packed::Type::iq2_xs,"iq2_xs",256,74},
    {packed::Type::iq2_s,"iq2_s",256,82},{packed::Type::iq3_xxs,"iq3_xxs",256,98},
    {packed::Type::q5_k,"q5_K",256,176},{packed::Type::q8_0,"q8_0",32,34}};
const packed::Tables tables{
    rad::gguf::tables::iq2xxs_grid,rad::gguf::tables::iq2xs_grid,
    rad::gguf::tables::iq2s_grid,rad::gguf::tables::iq3xxs_grid,
    rad::gguf::tables::ksigns_iq2xs};
using Json=nlohmann::json;
const Json& fixture(const char* name) {
    static const Json data=[] {
        std::ifstream file(std::string(RAD_GOLDEN_FIXTURES)+"/gguf-import.json");
        if(!file)throw std::runtime_error("missing independent GGML decoder fixture");
        return Json::parse(file).at("formats");
    }();
    for(const auto& value:data)if(value.at("name")==name)return value;
    throw std::runtime_error("missing independent GGML format");
}
std::vector<uint8_t> unhex(const std::string& text) {
    std::vector<uint8_t> out;out.reserve(text.size()/2);
    for(size_t i=0;i<text.size();i+=2)out.push_back(uint8_t(std::stoul(text.substr(i,2),nullptr,16)));
    return out;
}
void put16(uint8_t* out,uint16_t value) {out[0]=uint8_t(value);out[1]=uint8_t(value>>8);}
void put_bits(uint8_t* out,size_t at,unsigned width,uint32_t value) {
    for(unsigned b=0;b<width;++b) {
        const size_t byte=(at+b)/8;const unsigned bit=unsigned((at+b)%8);
        out[byte]=uint8_t((out[byte]&~(1u<<bit))|(((value>>b)&1u)<<bit));
    }
}
std::vector<uint8_t> random_blocks(Format f,int64_t rows,int64_t cols,uint32_t seed) {
    std::mt19937 rng(seed);std::vector<uint8_t> out(size_t(rows*(cols/f.values)*f.bytes));
    for(auto& b:out)b=uint8_t(rng());
    // Small exactly representable half scales keep matvecs finite and prevent NaNs from
    // concealing sign/index errors. A separate case exercises finite half exponent extremes.
    for(size_t at=0;at<out.size();at+=size_t(f.bytes)) {
        put16(out.data()+at,uint16_t(0x2400+(rng()%16)*0x20));
        if(f.type==packed::Type::q5_k)put16(out.data()+at+2,uint16_t(0x2000+(rng()%16)*0x20));
    }
    return out;
}
std::vector<float> oracle(Format f,int64_t rows,int64_t cols,const std::vector<uint8_t>& raw) {
    const auto* entry=qfix::reg().checkpoint_importer("gguf",f.name);
    if(!entry)throw std::runtime_error(std::string("missing real GGUF importer ")+f.name);
    RadCheckpointImportWeight weight{"packed-decode-oracle",f.name,rows,cols};
    RadEncoding encoding{};
    if(entry->info->encoding(&weight,&encoding)!=RAD_OK)throw std::runtime_error("import encoding failed");
    std::array<std::vector<uint8_t>,RAD_ENC_MAX_PLANES> planes;
    void* output[RAD_ENC_MAX_PLANES]{};const void* input[RAD_ENC_MAX_PLANES]{};
    for(int p=0;p<encoding.n_planes;++p) {
        planes[p].resize(size_t(rad_enc_plane_bytes(&encoding.plane[p],rows,cols)));
        output[p]=planes[p].data();input[p]=planes[p].data();
    }
    if(entry->info->import_rows(&weight,raw.data(),raw.size(),rows,output)!=RAD_OK)
        throw std::runtime_error("GGUF import failed");
    rad::EncodedView view;rad::enc_view(encoding,rows,cols,input,&view);
    std::vector<float> result(size_t(rows*cols));std::string why;
    if(rad::enc_decode_rows(view,0,rows,result.data(),&why)!=RAD_OK)
        throw std::runtime_error("generic canonical decode failed: "+why);
    return result;
}
void compare(Format f,int64_t rows,int64_t cols,const uint8_t* raw,const std::vector<float>& expected) {
    REQUIRE_EQ(expected.size(),size_t(rows*cols));
    for(int64_t r=0;r<rows;++r)for(int64_t c=0;c<cols;++c) {
        const auto* block=raw+(r*(cols/f.values)+c/f.values)*f.bytes;
        const float got=packed::decode(f.type,block,int(c%f.values),tables);
        const float want=expected[size_t(r*cols+c)];
        if(got!=want) {
            std::fprintf(stderr,"    native decode: %s row=%lld column=%lld offset=%lld\n",
                         f.name,(long long)r,(long long)c,(long long)(c%f.values));
            CHECK_EQ(got,want);return;
        }
        CHECK_EQ(got,want);
    }
}
}

TEST(ds41_packed_decode_geometry_matches_standard_native_block_sizes) {
    for(const auto f:formats) {
        CHECK_EQ(packed::block_values(f.type),f.values);CHECK_EQ(packed::block_bytes(f.type),f.bytes);
    }
    static_assert(packed::block_values(packed::Type::iq2_xs)==256);
    static_assert(packed::block_bytes(packed::Type::iq2_xs)==74);
    static_assert(packed::block_values(packed::Type::q8_0)==32);
    static_assert(packed::block_bytes(packed::Type::q8_0)==34);
}
TEST(ds41_packed_decode_matches_independent_ggml_goldens_at_every_offset) {
    for(const auto f:formats) {
        const auto& g=fixture(f.name);const auto raw=unhex(g.at("packed_hex"));
        const auto bytes=unhex(g.at("f32_hex"));std::vector<float> expected(bytes.size()/4);
        std::memcpy(expected.data(),bytes.data(),bytes.size());
        compare(f,g.at("rows"),g.at("cols"),raw.data(),expected);
    }
}
TEST(ds41_packed_decode_random_model_widths_and_unaligned_blocks_match_canonical_decoder) {
    for(const auto f:formats)for(int64_t cols:{256,512,768,2304,4096,7168})for(uint32_t seed=0;seed<3;++seed) {
        const auto raw=random_blocks(f,2,cols,411017+seed+uint32_t(cols));
        const auto expected=oracle(f,2,cols,raw);
        std::vector<uint8_t> guarded(raw.size()+16,0xA3);const size_t offset=1+seed;
        std::copy(raw.begin(),raw.end(),guarded.begin()+offset);const auto before=guarded;
        compare(f,2,cols,guarded.data()+offset,expected);CHECK(guarded==before);
    }
}
TEST(ds41_packed_decode_finite_half_scale_edges_include_subnormals_negative_and_maximum) {
    constexpr uint16_t scales[]{0,0x8000,1,0x8001,0x03FF,0x83FF,0x0400,0x8400,
                                0x0401,0x8401,0x2800,0xA800,0x3C00,0xBC00,0x7BFF,0xFBFF};
    for(const auto f:formats) {
        constexpr int64_t rows=std::size(scales),cols=256;
        auto raw=random_blocks(f,rows,cols,9308);
        for(int64_t r=0;r<rows;++r)for(int b=0;b<cols/f.values;++b) {
            auto* block=raw.data()+(r*(cols/f.values)+b)*f.bytes;
            put16(block,scales[(size_t(r)+size_t(b))%std::size(scales)]);
            if(f.type==packed::Type::q5_k)put16(block+2,scales[std::size(scales)-1-size_t(r)]);
        }
        compare(f,rows,cols,raw.data(),oracle(f,rows,cols,raw));
    }
}
TEST(ds41_packed_decode_exhausts_iq_grids_parity_signs_and_local_scale_nibbles) {
    for(int format=0;format<4;++format) {
        const auto f=formats[format];const int64_t rows=f.type==packed::Type::iq2_s?256:128;
        auto raw=random_blocks(f,rows,256,9217+format);
        for(int64_t r=0;r<rows;++r) {
            auto* block=raw.data()+r*f.bytes;
            if(f.type==packed::Type::iq2_xxs||f.type==packed::Type::iq3_xxs) {
                const int grid_groups=f.type==packed::Type::iq2_xxs?32:64;
                for(int group=0;group<grid_groups;++group) {
                    const int offset=f.type==packed::Type::iq2_xxs?2+(group/4)*8+group%4:2+group;
                    block[offset]=uint8_t(r*grid_groups+group);
                }
                for(int group=0;group<32;++group) {
                    const int offset=f.type==packed::Type::iq2_xxs?2+(group/4)*8+4:2+64+(group/4)*4;
                    put_bits(block,size_t(offset)*8+unsigned(group%4)*7,7,unsigned(r+group)%128);
                }
                for(int scale=0;scale<8;++scale) {
                    const int offset=f.type==packed::Type::iq2_xxs?2+scale*8+4:2+64+scale*4;
                    put_bits(block,size_t(offset)*8+28,4,unsigned(r+scale)%16);
                }
            } else if(f.type==packed::Type::iq2_xs) {
                for(int group=0;group<32;++group)
                    put16(block+2+group*2,uint16_t(((r*32+group)%512)|(((r+group)%128)<<9)));
                for(int scale=0;scale<16;++scale)
                    put_bits(block,(2+64)*8+unsigned(scale)*4,4,unsigned(r+scale)%16);
            } else {
                for(int group=0;group<32;++group) {
                    const unsigned code=unsigned(r*32+group)%1024;
                    block[2+group]=uint8_t(code);block[2+32+group]=uint8_t(r+group);
                    put_bits(block,(2+64)*8+unsigned(group)*2,2,code>>8);
                }
                for(int scale=0;scale<16;++scale)
                    put_bits(block,(2+64+8)*8+unsigned(scale)*4,4,unsigned(r+scale)%16);
            }
        }
        compare(f,rows,256,raw.data(),oracle(f,rows,256,raw));
    }
}
TEST(ds41_packed_decode_dense_formats_cover_all_raw_code_and_metadata_byte_values) {
    for(int format=4;format<6;++format) {
        const auto f=formats[format];constexpr int64_t rows=256,cols=256;
        auto raw=random_blocks(f,rows,cols,51080+format);
        for(int64_t r=0;r<rows;++r)for(int b=0;b<cols/f.values;++b) {
            auto* block=raw.data()+(r*(cols/f.values)+b)*f.bytes;
            const int first=f.type==packed::Type::q5_k?4:2;
            for(int byte=first;byte<f.bytes;++byte)block[byte]=uint8_t(r+byte*17+b*29);
        }
        compare(f,rows,cols,raw.data(),oracle(f,rows,cols,raw));
    }
}
TEST(ds41_packed_decode_top_two_routed_fp32_matvec_matches_generic_decoded_experts) {
    constexpr int experts=3,tokens=4,rows=6,cols=2304;
    constexpr int routes[tokens][2]{{2,0},{1,2},{0,2},{2,1}};
    constexpr float probabilities[tokens][2]{{0.625f,0.375f},{0.25f,0.75f},{0.875f,0.125f},{0.5f,0.5f}};
    std::array<std::vector<float>,tokens> activation;
    for(int token=0;token<tokens;++token)for(int k=0;k<cols;++k)
        activation[token].push_back(float((k*37+token*53)%257-128)/256.0f);
    for(const auto f:formats) {
        const size_t expert_bytes=size_t(rows*(cols/f.values)*f.bytes);
        std::vector<uint8_t> weights;std::array<std::vector<float>,experts> canonical;
        for(int expert=0;expert<experts;++expert) {
            const auto raw=random_blocks(f,rows,cols,9031+expert);
            canonical[expert]=oracle(f,rows,cols,raw);weights.insert(weights.end(),raw.begin(),raw.end());
        }
        bool nonzero=false;
        for(int token=0;token<tokens;++token)for(int row=0;row<rows;++row) {
            float actual=0,expected=0;
            for(int slot=0;slot<2;++slot) {
                const int expert=routes[token][slot];float got=0,want=0;
                for(int k=0;k<cols;++k) {
                    const auto* block=weights.data()+size_t(expert)*expert_bytes+size_t(row*(cols/f.values)+k/f.values)*f.bytes;
                    got=std::fma(packed::decode(f.type,block,k%f.values,tables),activation[token][k],got);
                    want=std::fma(canonical[expert][size_t(row*cols+k)],activation[token][k],want);
                }
                actual=std::fma(probabilities[token][slot],got,actual);
                expected=std::fma(probabilities[token][slot],want,expected);
            }
            CHECK(std::isfinite(actual));CHECK_EQ(actual,expected);nonzero|=expected!=0;
        }
        CHECK(nonzero);
    }
}
RAD_TEST_MAIN()
