/* Native embedding outputs use independent GGML goldens or the real importer's generic
 * canonical decoder as oracle. The new native scalar decoder never generates expectations. */
#include "quant_fixture.h"
#include "../libds41/ds41_embedding_fixture.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <sys/mman.h>
namespace e=rad::ds41::embedding;
namespace f=e::testing;
namespace pk=rad::ds41::packed;
namespace {
constexpr pk::Type formats[]{pk::Type::iq2_xxs,pk::Type::iq2_xs,pk::Type::iq2_s,pk::Type::iq3_xxs,pk::Type::q5_k,pk::Type::q8_0};
struct Imported {
    RadEncoding encoding{};std::vector<std::vector<uint8_t>> storage;std::vector<int> selection;
    std::vector<float> decoded;int64_t n,k;
    explicit Imported(const f::Fixture& c):n(c.n),k(c.k) {
        const char* name=pk::type_name(c.type);const auto* entry=qfix::reg().checkpoint_importer("gguf",name);
        if(!entry)throw std::runtime_error("missing real GGUF importer");
        RadCheckpointImportWeight w{"embedding-oracle",name,n,k};
        if(entry->info->encoding(&w,&encoding)!=RAD_OK)throw std::runtime_error("embedding oracle encoding");
        void* out[RAD_ENC_MAX_PLANES]{};const void* in[RAD_ENC_MAX_PLANES]{};
        for(int i=0;i<encoding.n_planes;++i){storage.emplace_back(size_t(rad_enc_plane_bytes(&encoding.plane[i],n,k)));
            out[i]=storage.back().data();in[i]=out[i];selection.push_back(i);}
        if(entry->info->import_rows(&w,c.raw(),size_t(n*c.row_bytes),n,out)!=RAD_OK)throw std::runtime_error("embedding oracle import");
        rad::EncodedView view;rad::enc_view(encoding,n,k,in,&view);decoded.resize(size_t(n*k));
        if(rad::enc_decode_rows(view,0,n,decoded.data())!=RAD_OK)throw std::runtime_error("embedding canonical decode");
    }
    std::vector<RadTensor> tensors(bool data=true)const {
        std::vector<RadTensor> out;
        for(int i=0;i<encoding.n_planes;++i){int64_t rows=0,cols=0;rad_enc_plane_dims(&encoding.plane[i],n,k,&rows,&cols);
            RadTensor t{};t.data=data?const_cast<uint8_t*>(storage[size_t(i)].data()):nullptr;t.dtype=encoding.plane[i].dtype;t.rank=2;
            t.shape[0]=rows;t.shape[1]=cols;t.stride[0]=cols;t.stride[1]=1;out.push_back(t);}
        return out;
    }
};
std::vector<uint8_t> unhex(const std::string& value) {
    std::vector<uint8_t> out;
    for(size_t i=0;i<value.size();i+=2)out.push_back(uint8_t(std::stoul(value.substr(i,2),nullptr,16)));
    return out;
}
void compare(f::Fixture& c,const std::vector<float>& canonical) {
    auto expected=c.y;for(int64_t r=0;r<c.rows;++r)for(int64_t col=0;col<c.k;++col)
        expected.set(r*c.k+col,canonical[size_t(int64_t(c.ids.id(r))*c.k+col)]);
    const auto before_ids=c.ids.words,before_native=c.native;const auto before_flag=c.flag;
    REQUIRE_EQ(c.run(),RAD_OK);CHECK(c.y.words==expected.words);CHECK(c.ids.words==before_ids);CHECK(c.native==before_native);
    CHECK_EQ(c.flag[0],RAD_OK);for(int i=1;i<4;++i)CHECK_EQ(c.flag[size_t(i)],before_flag[size_t(i)]);
}
struct Inaccessible {
    void* data=nullptr;size_t bytes;
    explicit Inaccessible(size_t size):bytes((size+4095)/4096*4096){data=mmap(nullptr,bytes,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);if(data==MAP_FAILED)throw std::runtime_error("mmap");}
    ~Inaccessible(){munmap(data,bytes);}
};
}
TEST(ds41_embedding_all_formats_model_widths_pitches_and_dtypes_match_canonical_decoder) {
    for(auto type:formats)for(int64_t k:{256,2304,7168})for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(int pitch=0;pitch<3;++pitch){
        f::Fixture c(type,dtype,pitch,k);const Imported imported(c);compare(c,imported.decoded);
    }
}
TEST(ds41_embedding_independent_ggml_goldens_match_at_every_selected_element) {
    std::ifstream file(std::string(RAD_GOLDEN_FIXTURES)+"/gguf-import.json");REQUIRE(file.good());
    const auto fixtures=nlohmann::json::parse(file).at("formats");size_t found=0;
    for(auto type:formats)for(const auto& golden:fixtures)if(golden.at("name")==pk::type_name(type)) {
        ++found;
        const auto raw=unhex(golden.at("packed_hex")),bytes=unhex(golden.at("f32_hex"));std::vector<float> values(bytes.size()/4);std::memcpy(values.data(),bytes.data(),bytes.size());
        for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(int pitch=0;pitch<3;++pitch){
            f::Fixture c(type,dtype,pitch,golden.at("cols"),golden.at("rows"),7);
            REQUIRE_EQ(raw.size(),size_t(c.n*c.row_bytes));std::memcpy(c.raw(),raw.data(),raw.size());compare(c,values);
        }
    }
    CHECK_EQ(found,std::size(formats));
}
TEST(ds41_embedding_finite_half_scale_edges_and_unaligned_native_rows_preserve_cast_boundary) {
    constexpr uint16_t scales[]{0,0x8000,1,0x8001,0x3ff,0x83ff,0x400,0x8400,0x3c00,0xbc00,0x7bff,0xfbff};
    for(auto type:formats)for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16}) {
        f::Fixture c(type,dtype,2,256,std::size(scales),std::size(scales));
        for(int64_t row=0;row<c.n;++row) {
            c.ids.id(row,int32_t(row));
            for(int64_t b=0;b<c.k/pk::block_values(type);++b) {
                auto* raw=c.raw()+row*c.row_bytes+b*pk::block_bytes(type);const auto scale=scales[size_t(row)];
                raw[0]=uint8_t(scale);raw[1]=uint8_t(scale>>8);
                if(type==pk::Type::q5_k){raw[2]=uint8_t(scale);raw[3]=uint8_t(scale>>8);}
            }
        }
        const Imported imported(c);compare(c,imported.decoded);
        const auto expected=c.y.words;auto a=c.args();std::vector<uint8_t> unaligned(size_t(c.n*c.row_bytes)+2,0xA3);
        std::copy(c.raw(),c.raw()+c.n*c.row_bytes,unaligned.begin()+1);const auto before=unaligned;
        c.tensors[e::weight].data=unaligned.data()+1;
        CHECK_EQ(e::emulate(&a),RAD_OK);CHECK(c.y.words==expected);CHECK(unaligned==before);
    }
}
TEST(ds41_embedding_layout_hooks_reuse_exact_packed_bridge_all_planes_and_private_storage) {
    RadLayoutFn layout=&e::layout;RadRelayoutFn relayout=&e::relayout;RadUnrelayoutFn unrelayout=&e::unrelayout;
    for(auto type:formats){f::Fixture c(type,RAD_BF16,0,256);Imported imported(c);auto a=c.args();const auto empty=imported.tensors(false),full=imported.tensors();RadLayout l{};
        REQUIRE_EQ(layout(a.p,a.n_p,e::weight,&imported.encoding,imported.selection.data(),empty.data(),empty.size(),&l),RAD_OK);
        CHECK_EQ(l.dtype,pk::storage_dtype(type));CHECK_EQ(std::string(l.tag),std::string(pk::runtime_tag(type)));CHECK_EQ(l.align,256);
        CHECK_EQ(l.rank,2u);CHECK_EQ(l.shape[0],c.n);CHECK_EQ(l.shape[1],c.row_bytes);CHECK_EQ(l.bytes,c.n*c.row_bytes);
        std::vector<uint8_t> raw(size_t(l.bytes)+32,0xA3);
        REQUIRE_EQ(relayout(a.p,a.n_p,e::weight,&imported.encoding,imported.selection.data(),full.data(),full.size(),raw.data()+8,l.bytes+16),RAD_OK);
        CHECK(std::equal(raw.begin()+8,raw.begin()+8+l.bytes,c.raw()));
        CHECK(std::all_of(raw.begin(),raw.begin()+8,[](uint8_t x){return x==0xA3;}));
        CHECK(std::all_of(raw.begin()+8+l.bytes,raw.end(),[](uint8_t x){return x==0xA3;}));
        auto restore=imported;
        for(auto& p:restore.storage)std::fill(p.begin(),p.end(),0xA5);
        const auto destination=restore.tensors();
        REQUIRE_EQ(unrelayout(a.p,a.n_p,e::weight,&imported.encoding,imported.selection.data(),raw.data()+8,l.bytes,destination.data(),destination.size()),RAD_OK);
        CHECK(restore.storage==imported.storage);
        RadLayout bad{};CHECK(layout(a.p,a.n_p,e::token_ids,&imported.encoding,imported.selection.data(),empty.data(),empty.size(),&bad)<0);
        CHECK(layout(a.p,a.n_p,e::weight,&imported.encoding,imported.selection.data(),empty.data(),empty.size()-1,&bad)<0);
        auto wrong=imported.selection;std::swap(wrong[0],wrong[1]);
        CHECK(relayout(a.p,a.n_p,e::weight,&imported.encoding,wrong.data(),full.data(),full.size(),raw.data()+8,l.bytes)<0);
        CHECK(std::equal(raw.begin()+8,raw.begin()+8+l.bytes,c.raw()));
        RadOpdDesc d{};CHECK_EQ(e::shape(a.p,a.n_p,e::token_ids,&d),RAD_OK);CHECK_EQ(d.dtype,uint32_t(RAD_I32));CHECK_EQ(d.shape[0],c.rows+2);CHECK_EQ(d.idx_max,c.n);
        CHECK_EQ(e::shape(a.p,a.n_p,e::weight,&d),RAD_E_UNSUPPORTED);CHECK_EQ(e::shape(a.p,a.n_p,e::output,&d),RAD_OK);CHECK_EQ(d.shape[1],c.k);
    }
}
TEST(ds41_embedding_invalid_final_id_is_atomic_before_reading_weights) {
    for(auto type:formats)for(int32_t bad:{-1,5,INT32_MAX}){f::Fixture c(type,RAD_BF16,2);auto a=c.args();c.ids.id(c.rows-1,bad);
        const auto before_ids=c.ids.words,before_native=c.native,before_y=c.y.words;Inaccessible weights(size_t(c.n*c.row_bytes));c.tensors[e::weight].data=weights.data;
        CHECK_EQ(e::emulate(&a),RAD_E_SHAPE);CHECK_EQ(c.flag[0],RAD_E_SHAPE);CHECK_EQ(c.flag[1],912);
        CHECK(c.ids.words==before_ids);CHECK(c.native==before_native);CHECK(c.y.words==before_y);
    }
}
TEST(ds41_embedding_metadata_failures_preserve_all_operands_and_status) {
    for(int fault=0;fault<22;++fault){f::Fixture c;auto a=c.args();const auto before_ids=c.ids.words,before_native=c.native,before_y=c.y.words;const auto before_flag=c.flag;
        if(fault==0)c.tensors[e::weight].dtype=RAD_U8;
        if(fault==1)c.tensors[e::weight].shape[1]=c.row_bytes;
        if(fault==2)c.tensors[e::weight].stride[0]=c.k+1;
        if(fault==3)c.tensors[e::weight].stride[1]=2;
        if(fault==4)c.tensors[e::output].dtype=RAD_I32;
        if(fault==5)c.tensors[e::output].stride[1]=0;
        if(fault==6)c.tensors[e::token_ids].dtype=RAD_F32;
        if(fault==7)c.tensors[e::output].data=c.tensors[e::weight].data;
        if(fault==8)c.tensors[e::token_ids].data=c.tensors[e::weight].data;
        if(fault==9)c.tensors[e::output].data=c.tensors[e::token_ids].data;
        if(fault==10)a.scratch=c.tensors[e::weight].data;
        if(fault==11)a.scratch_bytes=3;
        if(fault==12)a.scratch=reinterpret_cast<void*>(UINTPTR_MAX-3);
        if(fault==13)c.integer("M",e::max_rows+1);
        if(fault==14)c.integer("N",INT64_MAX);
        if(fault==15)c.integer("K",33);
        if(fault==16)c.text("format","q5_k");
        if(fault==17)c.text("y_dtype","i32");
        if(fault==18)c.tensors[e::weight].data=nullptr;
        if(fault==19)c.tensors[e::weight].data=reinterpret_cast<void*>(UINTPTR_MAX-31);
        if(fault==20)c.tensors[e::token_ids].data=reinterpret_cast<uint8_t*>(c.tensors[e::token_ids].data)+1;
        if(fault==21)c.tensors[e::output].stride[0]=INT64_MAX;
        e::Plan p;CHECK(e::validate(&a,&p)<0);CHECK(c.flag==before_flag);CHECK(c.ids.words==before_ids);CHECK(c.native==before_native);CHECK(c.y.words==before_y);
    }
}
TEST(ds41_embedding_gpu_plan_storage_metadata_checks_full_reserved_span_without_writes) {
    f::Fixture c;auto a=c.args();std::vector<uint32_t> scratch(size_t(rad::ds41::gpu_scratch_bytes<e::Plan>())/4+8,0x12345678u);
    a.scratch=scratch.data();a.scratch_bytes=rad::ds41::gpu_scratch_bytes<e::Plan>();e::Plan p;
    REQUIRE_EQ(e::validate(&a,&p),RAD_OK);const auto before=scratch;
    CHECK_EQ(rad::ds41::validate_plan_storage(&a,p),RAD_OK);
    --a.scratch_bytes;CHECK(rad::ds41::validate_plan_storage(&a,p)<0);++a.scratch_bytes;
    a.scratch=static_cast<uint8_t*>(static_cast<void*>(scratch.data()))+4;CHECK_EQ(rad::ds41::validate_plan_storage(&a,p),RAD_E_ALIGN);
    a.scratch=scratch.data();p.t[e::weight].data=static_cast<uint8_t*>(a.scratch)+16;CHECK_EQ(rad::ds41::validate_plan_storage(&a,p),RAD_E_INVAL);
    CHECK(scratch==before);
}
TEST(ds41_embedding_host_wrapper_returns_payload_errors_without_touching_device_scratch) {
    f::Fixture c;const Imported imported(c);auto a=c.args();Inaccessible scratch(4096);a.scratch=scratch.data;a.scratch_bytes=4096;
    auto expected=c.y;for(int64_t r=0;r<c.rows;++r)for(int64_t col=0;col<c.k;++col)expected.set(r*c.k+col,imported.decoded[size_t(c.ids.id(r)*c.k+col)]);
    CHECK_EQ(ds41_ref_embedding(&a,nullptr),RAD_OK);CHECK(c.y.words==expected.words);CHECK_EQ(c.flag[0],911);
    c.ids.id(c.rows-1,-1);const auto before=c.y.words;CHECK_EQ(ds41_ref_embedding(&a,nullptr),RAD_E_SHAPE);CHECK(c.y.words==before);CHECK_EQ(c.flag[0],911);
}
RAD_TEST_MAIN()
