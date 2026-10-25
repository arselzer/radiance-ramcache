/* Independent native -> canonical comparisons with libquant's importer, including arbitrary
 * blocks, exact F16 payloads and the actual RadUnrelayoutFn ABI. No device execution. */
#include "quant_fixture.h"
#include "../arch/common/rad_gguf_relayout.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <random>

namespace gg=rad::gguf;
namespace {
struct Type {gg::Format format;RadUnrelayoutFn inverse;};
#define TYPE(f) {gg::Format::f,&gg::abi_unrelayout<gg::Format::f>}
constexpr Type types[]{TYPE(iq2_xxs),TYPE(iq2_xs),TYPE(iq2_s),TYPE(iq3_xxs),TYPE(q5_k),TYPE(q8_0)};
#undef TYPE
struct Imported {
    Type type;int64_t rows,cols;RadEncoding enc{};
    std::vector<uint8_t> raw;
    std::array<std::vector<uint8_t>,RAD_ENC_MAX_PLANES> expected;
    Imported(Type t,int64_t r,int64_t c,std::vector<uint8_t> bytes):type(t),rows(r),cols(c),raw(std::move(bytes)) {
        const auto* entry=qfix::reg().checkpoint_importer("gguf",gg::geometry(t.format)->name);
        if(!entry)throw std::runtime_error("missing GGUF importer");
        RadCheckpointImportWeight weight{"inverse-layout-fixture",gg::geometry(t.format)->name,rows,cols};
        if(entry->info->encoding(&weight,&enc)!=RAD_OK)throw std::runtime_error("cannot describe fixture");
        void* ptr[RAD_ENC_MAX_PLANES]{};
        for(int p=0;p<enc.n_planes;++p) {
            expected[p].resize(size_t(rad_enc_plane_bytes(&enc.plane[p],rows,cols)),0);
            ptr[p]=expected[p].data();
        }
        if(entry->info->import_rows(&weight,raw.data(),raw.size(),rows,ptr)!=RAD_OK)
            throw std::runtime_error("cannot import fixture");
    }
};
struct Buffers {
    gg::MutableView view;
    std::array<std::vector<uint8_t>,RAD_ENC_MAX_PLANES> storage;
    std::array<size_t,RAD_ENC_MAX_PLANES> row_bytes{},plane_rows{};
    explicit Buffers(const Imported& data,size_t padding=0) {
        view.encoding=data.enc;view.rows=data.rows;view.cols=data.cols;
        for(int p=0;p<data.enc.n_planes;++p) {
            int64_t r=0,c=0;rad_enc_plane_dims(&data.enc.plane[p],data.rows,data.cols,&r,&c);
            const size_t rb=size_t(rad_enc_row_bytes(data.enc.plane[p].dtype,c)),stride=rb+padding;
            row_bytes[p]=rb;plane_rows[p]=size_t(r);
            storage[p].resize(size_t(r)*stride+32,0xC7);
            view.planes[p]={std::span(storage[p]).subspan(16,size_t(r)*stride),stride};
        }
    }
    void matches(const Imported& data) const {
        for(int p=0;p<data.enc.n_planes;++p) {
            const auto& bytes=storage[p];const size_t rb=row_bytes[p],stride=view.planes[p].row_stride;
            CHECK(std::all_of(bytes.begin(),bytes.begin()+16,[](uint8_t b){return b==0xC7;}));
            for(size_t r=0;r<plane_rows[p];++r) {
                CHECK(std::equal(data.expected[p].begin()+r*rb,data.expected[p].begin()+(r+1)*rb,
                                  bytes.begin()+16+r*stride));
                CHECK(std::all_of(bytes.begin()+16+r*stride+rb,bytes.begin()+16+(r+1)*stride,
                                   [](uint8_t b){return b==0xC7;}));
            }
            CHECK(std::all_of(bytes.end()-16,bytes.end(),[](uint8_t b){return b==0xC7;}));
        }
    }
    std::vector<RadTensor> tensors() const {
        std::vector<RadTensor> result;
        for(int p=0;p<view.encoding.n_planes;++p) {
            int64_t r=0,c=0;rad_enc_plane_dims(&view.encoding.plane[p],view.rows,view.cols,&r,&c);
            RadTensor t{};t.data=view.planes[p].bytes.data();t.dtype=view.encoding.plane[p].dtype;
            t.rank=2;t.shape[0]=r;t.shape[1]=c;t.stride[0]=c;t.stride[1]=1;result.push_back(t);
        }
        return result;
    }
};
Imported random(Type type,int64_t rows,int64_t cols,uint32_t seed) {
    std::mt19937 rng(seed);const auto* g=gg::geometry(type.format);
    std::vector<uint8_t> raw(size_t(rows*(cols/g->values)*g->bytes));
    for(auto& byte:raw)byte=uint8_t(rng());
    return Imported(type,rows,cols,std::move(raw));
}
void check(const Imported& data,size_t padding=0) {
    Buffers buffers(data,padding);std::string why;
    REQUIRE_EQ(gg::unrepack(data.type.format,data.raw,buffers.view,&why),RAD_OK);CHECK(why.empty());
    buffers.matches(data);
    std::vector<uint8_t> again(data.raw.size());
    REQUIRE_EQ(gg::repack(data.type.format,buffers.view.read_view(),{0,data.rows,0,data.cols},again),RAD_OK);
    CHECK(again==data.raw);
}
void rejected(const Imported& data,Buffers& buffers,const gg::MutableView& bad,
                std::span<const uint8_t> input) {
    const auto before=buffers.storage;std::string why;
    CHECK(gg::unrepack(data.type.format,input,bad,&why)<0);CHECK(!why.empty());
    CHECK(buffers.storage==before);
}
}
TEST(gguf_unrepack_matches_importer_on_independent_ggml_goldens) {
    std::ifstream file(std::string(RAD_GOLDEN_FIXTURES)+"/gguf-import.json");REQUIRE(file.good());
    const auto gold=nlohmann::json::parse(file);
    for(const auto type:types) {
        bool found=false;
        for(const auto& f:gold.at("formats"))if(f.at("name")==gg::geometry(type.format)->name) {
            const std::string hex=f.at("packed_hex");std::vector<uint8_t> bytes;
            for(size_t i=0;i<hex.size();i+=2)bytes.push_back(uint8_t(std::stoul(hex.substr(i,2),nullptr,16)));
            check(Imported(type,f.at("rows"),f.at("cols"),std::move(bytes)));found=true;
        }
        CHECK(found);
    }
}
TEST(gguf_unrepack_preserves_random_model_width_blocks_and_row_pitches) {
    for(const auto type:types)for(uint32_t seed=0;seed<16;++seed)
        check(random(type,3,5120,0x64CC+seed),seed%2?13:0);
}
TEST(gguf_unrepack_preserves_every_f16_payload_without_float_arithmetic) {
    for(const auto type:types) {
        const auto* g=gg::geometry(type.format);std::vector<uint8_t> raw(size_t(65536*g->bytes),0xA9);
        for(unsigned bits=0;bits<65536;++bits) {
            auto* p=raw.data()+size_t(bits)*size_t(g->bytes);p[0]=uint8_t(bits);p[1]=uint8_t(bits>>8);
            if(type.format==gg::Format::q5_k) {p[2]=uint8_t(~bits);p[3]=uint8_t((~bits)>>8);}
        }
        // Compare all plane bytes without per-row assertion overhead for this exhaustive case.
        Imported data(type,65536,g->values,std::move(raw));Buffers buffers(data);
        REQUIRE_EQ(gg::unrepack(type.format,data.raw,buffers.view),RAD_OK);
        for(int p=0;p<data.enc.n_planes;++p)
            CHECK(std::equal(data.expected[p].begin(),data.expected[p].end(),buffers.storage[p].begin()+16));
    }
}
TEST(gguf_unrepack_rejects_bad_metadata_extents_and_late_plane_caps_atomically) {
    for(const auto type:types) {
        const auto data=random(type,2,512,0x17);Buffers buffers(data,7);
        rejected(data,buffers,buffers.view,std::span(data.raw).first(data.raw.size()-1));
        rejected(data,buffers,buffers.view,{});
        auto bad=buffers.view;bad.rows=0;rejected(data,buffers,bad,data.raw);
        bad=buffers.view;bad.cols=511;rejected(data,buffers,bad,data.raw);
        bad=buffers.view;bad.rows=INT64_MAX;rejected(data,buffers,bad,data.raw);
        bad=buffers.view;rad_enc_copy_str(bad.encoding.transform,"fwht256");rejected(data,buffers,bad,data.raw);
        for(int p=0;p<data.enc.n_planes;++p) {
            bad=buffers.view;bad.planes[p].bytes={};rejected(data,buffers,bad,data.raw);
            bad=buffers.view;bad.planes[p].row_stride=buffers.row_bytes[p]-1;
            if(bad.planes[p].row_stride)rejected(data,buffers,bad,data.raw);
            bad=buffers.view;bad.planes[p].row_stride=SIZE_MAX;
            if(buffers.plane_rows[p]>1)rejected(data,buffers,bad,data.raw);
            bad=buffers.view;
            const size_t required=(buffers.plane_rows[p]-1)*bad.planes[p].row_stride+buffers.row_bytes[p];
            bad.planes[p].bytes=bad.planes[p].bytes.first(required-1);
            rejected(data,buffers,bad,data.raw);
        }
        const auto before=buffers.storage;
        CHECK(gg::unrepack(static_cast<gg::Format>(999),data.raw,buffers.view)<0);
        CHECK(before==buffers.storage);
    }
}
TEST(gguf_unrepack_rejects_source_and_interplane_aliases_before_writes) {
    for(const auto type:types) {
        const auto data=random(type,2,512,0x18);Buffers buffers(data);
        const size_t cap=std::max(buffers.view.planes[0].bytes.size(),buffers.view.planes[1].bytes.size());
        std::vector<uint8_t> shared(cap+data.raw.size()+64,0xC7);
        std::copy(data.raw.begin(),data.raw.end(),shared.begin());const auto before=shared;
        auto bad=buffers.view;bad.planes[0].bytes={shared.data()+16,buffers.view.planes[0].bytes.size()};
        rejected(data,buffers,bad,{shared.data(),data.raw.size()});CHECK(shared==before);
        bad=buffers.view;
        bad.planes[0].bytes={shared.data(),buffers.view.planes[0].bytes.size()};
        bad.planes[1].bytes={shared.data()+1,buffers.view.planes[1].bytes.size()};
        rejected(data,buffers,bad,data.raw);CHECK(shared==before);
    }
}
TEST(gguf_unrelayout_abi_recovers_all_planes_and_checks_destination_geometry) {
    for(const auto type:types) {
        const auto data=random(type,3,512,0x19);Buffers buffers(data);auto planes=buffers.tensors();
        int selection[RAD_ENC_MAX_PLANES]{};for(int p=0;p<data.enc.n_planes;++p)selection[p]=p;
        REQUIRE_EQ(type.inverse(nullptr,0,0,&data.enc,selection,data.raw.data(),data.raw.size(),
                                planes.data(),planes.size()),RAD_OK);
        buffers.matches(data);const auto before=buffers.storage;
        auto rejected_call=[&](const RadTensor* output,int n,const int* sel,const RadEncoding* enc,
                               const void* src,int64_t cap) {
            CHECK(type.inverse(nullptr,0,0,enc,sel,src,cap,output,n)<0);CHECK(buffers.storage==before);
        };
        for(int p=0;p<data.enc.n_planes;++p) {
            auto bad=planes;bad[p].data=nullptr;
            rejected_call(bad.data(),bad.size(),selection,&data.enc,data.raw.data(),data.raw.size());
            bad=planes;++bad[p].shape[1];
            rejected_call(bad.data(),bad.size(),selection,&data.enc,data.raw.data(),data.raw.size());
        }
        rejected_call(planes.data(),planes.size(),selection,&data.enc,data.raw.data(),data.raw.size()-1);
        rejected_call(planes.data(),planes.size(),selection,&data.enc,data.raw.data(),-1);
        rejected_call(planes.data(),planes.size(),selection,&data.enc,nullptr,data.raw.size());
        rejected_call(planes.data(),planes.size(),nullptr,&data.enc,data.raw.data(),data.raw.size());
        rejected_call(planes.data(),planes.size(),selection,nullptr,data.raw.data(),data.raw.size());
        selection[1]=0;
        rejected_call(planes.data(),planes.size(),selection,&data.enc,data.raw.data(),data.raw.size());
    }
}
RAD_TEST_MAIN()
