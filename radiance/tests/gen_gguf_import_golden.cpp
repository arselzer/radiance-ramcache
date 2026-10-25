// Generate synthetic packed GGUF block fixtures using an independent GGML decoder.
// This generator never calls the importer or Radiance's decoder; port headers supply only the
// standard block structures. See data/golden/gguf-import.README.txt for source pin/build commands.
#include "../libquant/lq_ggml_port.h"
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <nlohmann/json.hpp>
#include <cstdarg>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <random>

extern "C" {
void dequantize_row_iq2_xxs(const void*,float*,int64_t);
void dequantize_row_iq2_xs(const void*,float*,int64_t);
void dequantize_row_iq2_s(const void*,float*,int64_t);
void dequantize_row_iq3_xxs(const void*,float*,int64_t);
void dequantize_row_iq4_nl(const void*,float*,int64_t);
void dequantize_row_q3_K(const void*,float*,int64_t);
void dequantize_row_q4_K(const void*,float*,int64_t);
void dequantize_row_q5_K(const void*,float*,int64_t);
void dequantize_row_q6_K(const void*,float*,int64_t);
void dequantize_row_q8_0(const void*,float*,int64_t);
void dequantize_row_q4_0(const void*,float*,int64_t);
void dequantize_row_q4_1(const void*,float*,int64_t);
void dequantize_row_q5_0(const void*,float*,int64_t);
void dequantize_row_q5_1(const void*,float*,int64_t);
void dequantize_row_q2_K(const void*,float*,int64_t);
void dequantize_row_iq4_xs(const void*,float*,int64_t);
void dequantize_row_iq3_s(const void*,float*,int64_t);
void dequantize_row_iq1_s(const void*,float*,int64_t);
void dequantize_row_iq1_m(const void*,float*,int64_t);
[[noreturn]] void ggml_abort(const char* file,int line,const char* format,...) {
    std::fprintf(stderr,"GGML reference abort at %s:%d: ",file,line);
    va_list args;va_start(args,format);std::vfprintf(stderr,format,args);va_end(args);
    std::abort();
}
}

struct Format {
    const char* name;
    lq::ggml::ggml_type type;
    int block,bytes,d_offset,min_offset;
    void (*reference)(const void*,float*,int64_t);
};

#define FMT(NAME,ENUM,BLOCK,REF) {#NAME,lq::ggml::ENUM,BLOCK,sizeof(lq::ggml::block_##NAME),offsetof(lq::ggml::block_##NAME,d),-1,REF}
#define FMT_MIN(NAME,ENUM,BLOCK,REF) {#NAME,lq::ggml::ENUM,BLOCK,sizeof(lq::ggml::block_##NAME),offsetof(lq::ggml::block_##NAME,d),offsetof(lq::ggml::block_##NAME,dmin),REF}


int main(int argc,char** argv) {
    if (argc!=2) return 2;
    const Format formats[]={
        {"q4_0",lq::ggml::GGML_TYPE_Q4_0,32,sizeof(lq::ggml::block_q4_0),offsetof(lq::ggml::block_q4_0,d),-1,dequantize_row_q4_0},
        {"q4_1",lq::ggml::GGML_TYPE_Q4_1,32,sizeof(lq::ggml::block_q4_1),offsetof(lq::ggml::block_q4_1,d),offsetof(lq::ggml::block_q4_1,m),dequantize_row_q4_1},
        {"q5_0",lq::ggml::GGML_TYPE_Q5_0,32,sizeof(lq::ggml::block_q5_0),offsetof(lq::ggml::block_q5_0,d),-1,dequantize_row_q5_0},
        {"q5_1",lq::ggml::GGML_TYPE_Q5_1,32,sizeof(lq::ggml::block_q5_1),offsetof(lq::ggml::block_q5_1,d),offsetof(lq::ggml::block_q5_1,m),dequantize_row_q5_1},
        {"q2_K",lq::ggml::GGML_TYPE_Q2_K,256,sizeof(lq::ggml::block_q2_K),offsetof(lq::ggml::block_q2_K,d),offsetof(lq::ggml::block_q2_K,dmin),dequantize_row_q2_K},
        {"iq4_xs",lq::ggml::GGML_TYPE_IQ4_XS,256,sizeof(lq::ggml::block_iq4_xs),offsetof(lq::ggml::block_iq4_xs,d),-1,dequantize_row_iq4_xs},
        {"iq3_s",lq::ggml::GGML_TYPE_IQ3_S,256,sizeof(lq::ggml::block_iq3_s),offsetof(lq::ggml::block_iq3_s,d),-1,dequantize_row_iq3_s},
        {"iq1_s",lq::ggml::GGML_TYPE_IQ1_S,256,sizeof(lq::ggml::block_iq1_s),offsetof(lq::ggml::block_iq1_s,d),-1,dequantize_row_iq1_s},
        {"iq1_m",lq::ggml::GGML_TYPE_IQ1_M,256,sizeof(lq::ggml::block_iq1_m),-1,-1,dequantize_row_iq1_m},
        FMT(iq2_xxs,GGML_TYPE_IQ2_XXS,256,dequantize_row_iq2_xxs),
        FMT(iq2_xs,GGML_TYPE_IQ2_XS,256,dequantize_row_iq2_xs),
        FMT(iq2_s,GGML_TYPE_IQ2_S,256,dequantize_row_iq2_s),
        FMT(iq3_xxs,GGML_TYPE_IQ3_XXS,256,dequantize_row_iq3_xxs),
        FMT(iq4_nl,GGML_TYPE_IQ4_NL,32,dequantize_row_iq4_nl),
        FMT(q3_K,GGML_TYPE_Q3_K,256,dequantize_row_q3_K),
        FMT_MIN(q4_K,GGML_TYPE_Q4_K,256,dequantize_row_q4_K),
        FMT_MIN(q5_K,GGML_TYPE_Q5_K,256,dequantize_row_q5_K),
        FMT(q6_K,GGML_TYPE_Q6_K,256,dequantize_row_q6_K),
        FMT(q8_0,GGML_TYPE_Q8_0,32,dequantize_row_q8_0),
    };

    std::mt19937 random(1017);
    nlohmann::json result;
    result["reference"]="DS41Engine 122884239009d8220d6521f4278aacfaf1d9ad92, vendored ggml-quants.c";
    result["description"]="Synthetic packed blocks, no model weights. f32 words are little endian IEEE-754. Signed zero compares numerically.";
    result["seed"]=1017;
    auto hex=[](const uint8_t* p,size_t n) {
        static const char digits[]="0123456789abcdef";
        std::string out;out.reserve(n*2);
        for(size_t i=0;i<n;++i){out+=digits[p[i]>>4];out+=digits[p[i]&15];}
        return out;
    };
    for(const auto& f:formats) {
        constexpr int64_t rows=4,cols=256;
        const int64_t nb=cols/f.block;
        std::vector<uint8_t> packed(rows*nb*f.bytes);
        for(auto& b:packed)b=(uint8_t)random();
        for(int64_t i=0;i<rows*nb;++i){
            const uint16_t scales[]={0x2800,0xa800,0x0001,0x3400,0x0400,0};
            const uint16_t scale=scales[i%6];
            if (f.d_offset>=0) std::memcpy(packed.data()+i*f.bytes+f.d_offset,&scale,2);
            else for(int j=0;j<4;++j) {
                auto* at=packed.data()+i*f.bytes+offsetof(lq::ggml::block_iq1_m,scales)+j*2;
                uint16_t word;std::memcpy(&word,at,2);word=(word&0x0fff)|(((scale>>(j*4))&15)<<12);
                std::memcpy(at,&word,2);
            }
            if(f.min_offset>=0)std::memcpy(packed.data()+i*f.bytes+f.min_offset,&scale,2);
        }
        std::vector<float> expected(rows*cols);
        for(int64_t r=0;r<rows;++r)f.reference(packed.data()+r*nb*f.bytes,expected.data()+r*cols,cols);
        result["formats"].push_back({{"name",f.name},{"type",(int)f.type},{"rows",rows},{"cols",cols},
            {"block_values",f.block},{"block_bytes",f.bytes},{"packed_hex",hex(packed.data(),packed.size())},
            {"f32_hex",hex((const uint8_t*)expected.data(),expected.size()*4)}});
    }
    std::ofstream file(argv[1]);file<<result.dump(2)<<'\n';
}
