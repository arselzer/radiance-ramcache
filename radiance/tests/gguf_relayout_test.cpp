/* The actual RadLayoutFn/RadRelayoutFn boundary, not a fake device kernel. A layout call
 * receives metadata-only canonical tensors; relayout receives trusted allocated tensor
 * payloads. Raw byte capacities are tested separately through the bounded repack API. */
#include "quant_fixture.h"
#include "../arch/common/rad_gguf_relayout.h"
#include <nlohmann/json.hpp>
#include <array>
#include <fstream>

namespace gg = rad::gguf;
namespace {
using Json = nlohmann::json;
struct Hooks { gg::Format format; RadLayoutFn layout; RadRelayoutFn relayout; };
constexpr Hooks hooks[]{
    {gg::Format::iq2_xxs,&gg::abi_layout<gg::Format::iq2_xxs>,&gg::abi_relayout<gg::Format::iq2_xxs>},
    {gg::Format::iq2_xs,&gg::abi_layout<gg::Format::iq2_xs>,&gg::abi_relayout<gg::Format::iq2_xs>},
    {gg::Format::iq2_s,&gg::abi_layout<gg::Format::iq2_s>,&gg::abi_relayout<gg::Format::iq2_s>},
    {gg::Format::iq3_xxs,&gg::abi_layout<gg::Format::iq3_xxs>,&gg::abi_relayout<gg::Format::iq3_xxs>},
    {gg::Format::q5_k,&gg::abi_layout<gg::Format::q5_k>,&gg::abi_relayout<gg::Format::q5_k>},
    {gg::Format::q8_0,&gg::abi_layout<gg::Format::q8_0>,&gg::abi_relayout<gg::Format::q8_0>}};
const Json& fixtures() {
    static const Json value=[] {
        std::ifstream file(std::string(RAD_GOLDEN_FIXTURES)+"/gguf-import.json");
        if(!file)throw std::runtime_error("missing independent GGUF fixture");
        return Json::parse(file).at("formats");
    }();
    return value;
}
const Json& fixture(gg::Format format) {
    const auto* g=gg::geometry(format);
    if(g)for(const auto& f:fixtures())if(f.at("name")==g->name)return f;
    throw std::runtime_error("missing GGUF format");
}
std::vector<uint8_t> unhex(const std::string& value) {
    std::vector<uint8_t> result;result.reserve(value.size()/2);
    for(size_t i=0;i<value.size();i+=2)result.push_back(uint8_t(std::stoul(value.substr(i,2),nullptr,16)));
    return result;
}
struct Imported {
    Hooks api;
    int64_t rows,cols;
    RadEncoding encoding{};
    std::vector<uint8_t> raw;
    std::vector<std::vector<uint8_t>> storage;
    std::vector<int> selection;
    explicit Imported(Hooks hook):api(hook) {
        const auto& f=fixture(api.format);rows=f.at("rows");cols=f.at("cols");raw=unhex(f.at("packed_hex"));
        const auto& g=*gg::geometry(api.format);
        const auto* entry=qfix::reg().checkpoint_importer("gguf",g.name);
        if(!entry)throw std::runtime_error("missing GGUF importer");
        RadCheckpointImportWeight weight{"ABI-relayout-fixture",g.name,rows,cols};
        if(entry->info->encoding(&weight,&encoding)!=RAD_OK)throw std::runtime_error("cannot describe canonical fixture");
        void* data[RAD_ENC_MAX_PLANES]{};
        for(int p=0;p<encoding.n_planes;++p) {
            storage.emplace_back(size_t(rad_enc_plane_bytes(&encoding.plane[p],rows,cols)),0);
            data[p]=storage.back().data();selection.push_back(p);
        }
        if(entry->info->import_rows(&weight,raw.data(),raw.size(),rows,data)!=RAD_OK)
            throw std::runtime_error("cannot import canonical fixture");
    }
    std::vector<RadTensor> tensors(bool payload=true) const {
        std::vector<RadTensor> result;
        for(int p=0;p<encoding.n_planes;++p) {
            int64_t pr=0,pc=0;rad_enc_plane_dims(&encoding.plane[p],rows,cols,&pr,&pc);
            RadTensor t{};t.rank=2;t.dtype=encoding.plane[p].dtype;t.shape[0]=pr;t.shape[1]=pc;
            t.stride[0]=pc;t.stride[1]=1;t.data=payload?const_cast<uint8_t*>(storage[p].data()):nullptr;
            result.push_back(t);
        }
        return result;
    }
};
void rejected(const Imported& data,const RadEncoding* encoding,const int* selection,
                const RadTensor* planes,int count) {
    RadLayout layout{};
    CHECK(data.api.layout(nullptr,0,0,encoding,selection,planes,count,&layout)<0);
    std::vector<uint8_t> output(data.raw.size()+32,0xB7);
    CHECK(data.api.relayout(nullptr,0,0,encoding,selection,planes,count,output.data()+16,data.raw.size())<0);
    CHECK(std::all_of(output.begin(),output.end(),[](uint8_t b){return b==0xB7;}));
}
}

TEST(gguf_relayout_declare_time_hooks_work_with_no_payload_pointers) {
    for(const auto api:hooks) {
        const Imported data(api);const auto planes=data.tensors(false);RadLayout layout{};
        REQUIRE_EQ(api.layout(nullptr,0,3,&data.encoding,data.selection.data(),planes.data(),planes.size(),&layout),RAD_OK);
        const auto& g=*gg::geometry(api.format);const int64_t rb=data.cols/g.values*g.bytes;
        REQUIRE(layout.tag);CHECK_EQ(std::string(layout.tag),std::string(g.tag));
        CHECK_EQ(layout.dtype,uint32_t(RAD_U8));CHECK_EQ(layout.rank,2u);
        CHECK_EQ(layout.shape[0],data.rows);CHECK_EQ(layout.shape[1],rb);
        CHECK_EQ(layout.bytes,data.rows*rb);CHECK_EQ(layout.align,256);
    }
}
TEST(gguf_relayout_actual_abi_function_pointers_preserve_all_six_native_formats) {
    for(const auto api:hooks) {
        const Imported data(api);const auto planes=data.tensors();std::vector<uint8_t> output(data.raw.size()+48,0xE9);
        REQUIRE_EQ(api.relayout(nullptr,0,2,&data.encoding,data.selection.data(),planes.data(),planes.size(),output.data()+16,data.raw.size()+16),RAD_OK);
        CHECK(std::equal(data.raw.begin(),data.raw.end(),output.begin()+16));
        CHECK(std::all_of(output.begin(),output.begin()+16,[](uint8_t b){return b==0xE9;}));
        CHECK(std::all_of(output.begin()+16+data.raw.size(),output.end(),[](uint8_t b){return b==0xE9;}));
    }
}
TEST(gguf_relayout_rejects_missing_partial_reordered_and_duplicate_selections) {
    for(const auto api:hooks) {
        const Imported data(api);const auto planes=data.tensors();const auto n=data.encoding.n_planes;
        auto selection=data.selection;selection[1]=selection[0];rejected(data,&data.encoding,selection.data(),planes.data(),n);
        selection=data.selection;std::swap(selection[0],selection[1]);rejected(data,&data.encoding,selection.data(),planes.data(),n);
        selection=data.selection;selection.back()=n;rejected(data,&data.encoding,selection.data(),planes.data(),n);
        selection=data.selection;selection.front()=-1;rejected(data,&data.encoding,selection.data(),planes.data(),n);
        rejected(data,&data.encoding,data.selection.data(),planes.data(),n-1);
        rejected(data,&data.encoding,data.selection.data(),planes.data(),0);
        rejected(data,&data.encoding,data.selection.data(),planes.data(),-1);
        auto extra_selection=data.selection;auto extra_planes=planes;extra_selection.push_back(n);extra_planes.emplace_back();
        rejected(data,&data.encoding,extra_selection.data(),extra_planes.data(),n+1);
        rejected(data,&data.encoding,nullptr,planes.data(),n);
        rejected(data,&data.encoding,data.selection.data(),nullptr,n);
        rejected(data,nullptr,data.selection.data(),planes.data(),n);
        CHECK(api.layout(nullptr,0,0,&data.encoding,data.selection.data(),planes.data(),n,nullptr)<0);
    }
}
TEST(gguf_relayout_rejects_bad_plane_shapes_dtypes_and_noncontiguous_pitches) {
    for(const auto api:hooks) {
        const Imported data(api);const auto good=data.tensors();const int n=data.encoding.n_planes;
        for(int p=0;p<n;++p) {
            auto planes=good;planes[p].rank=1;rejected(data,&data.encoding,data.selection.data(),planes.data(),n);
            planes=good;++planes[p].shape[0];rejected(data,&data.encoding,data.selection.data(),planes.data(),n);
            planes=good;++planes[p].shape[1];rejected(data,&data.encoding,data.selection.data(),planes.data(),n);
            planes=good;planes[p].dtype=planes[p].dtype==RAD_F32?RAD_U8:RAD_F32;
            rejected(data,&data.encoding,data.selection.data(),planes.data(),n);
        }
        auto planes=good;++planes[0].stride[0];rejected(data,&data.encoding,data.selection.data(),planes.data(),n);
        planes=good;planes[0].stride[0]=0;rejected(data,&data.encoding,data.selection.data(),planes.data(),n);
        planes=good;planes[0].stride[1]=2;rejected(data,&data.encoding,data.selection.data(),planes.data(),n);
        planes=good;planes[0].stride[0]=-1;rejected(data,&data.encoding,data.selection.data(),planes.data(),n);
        // The public bounded API accepts byte pitches. The ABI adapter deliberately requires
        // canonical element-contiguous planes, never fractional sub-byte row addresses.
        if(api.format!=gg::Format::q8_0) {
            planes=good;const int p=rad_enc_find(&data.encoding,"scale");++planes[p].stride[0];
            rejected(data,&data.encoding,data.selection.data(),planes.data(),n);
        }
    }
}
TEST(gguf_relayout_requires_payloads_only_when_copying_and_validates_destination_capacity) {
    for(const auto api:hooks) {
        const Imported data(api);const auto good=data.tensors();const int n=data.encoding.n_planes;
        std::vector<uint8_t> out(data.raw.size()+32,0x53);
        for(int p=0;p<n;++p) {
            auto planes=good;planes[p].data=nullptr;RadLayout layout{};
            CHECK_EQ(api.layout(nullptr,0,0,&data.encoding,data.selection.data(),planes.data(),n,&layout),RAD_OK);
            CHECK(api.relayout(nullptr,0,0,&data.encoding,data.selection.data(),planes.data(),n,out.data()+16,data.raw.size())<0);
            CHECK(std::all_of(out.begin(),out.end(),[](uint8_t b){return b==0x53;}));
        }
        for(int64_t capacity:{int64_t(-1),int64_t(0),int64_t(data.raw.size()-1)}) {
            CHECK(api.relayout(nullptr,0,0,&data.encoding,data.selection.data(),good.data(),n,out.data()+16,capacity)<0);
            CHECK(std::all_of(out.begin(),out.end(),[](uint8_t b){return b==0x53;}));
        }
        CHECK(api.relayout(nullptr,0,0,&data.encoding,data.selection.data(),good.data(),n,nullptr,data.raw.size())<0);
    }
}
TEST(gguf_relayout_keeps_declare_time_metadata_separate_from_payload_validation) {
    for(size_t index=0;index<4;++index) {
        const auto api=hooks[index];Imported data(api);const int n=data.encoding.n_planes;
        const int grid=rad_enc_find(&data.encoding,"grid"),factor=rad_enc_find(&data.encoding,"scale.2");
        for(int altered:{grid,factor}) {
            Imported changed=data;changed.storage[altered].back()^=1;const auto planes=changed.tensors();RadLayout layout{};
            CHECK_EQ(api.layout(nullptr,0,0,&changed.encoding,changed.selection.data(),planes.data(),n,&layout),RAD_OK);
            std::vector<uint8_t> out(changed.raw.size(),0x35);
            CHECK(api.relayout(nullptr,0,0,&changed.encoding,changed.selection.data(),planes.data(),n,out.data(),out.size())<0);
            CHECK(std::all_of(out.begin(),out.end(),[](uint8_t b){return b==0x35;}));
        }
        Imported late=data;const int scale=rad_enc_find(&late.encoding,"scale");
        const auto& plane=late.encoding.plane[scale];int64_t pr=0,pc=0;rad_enc_plane_dims(&plane,late.rows,late.cols,&pr,&pc);
        const size_t rb=size_t(rad_enc_row_bytes(plane.dtype,pc));const size_t bit=size_t(pr-1)*rb*8+size_t(pc-1)*5;
        for(unsigned b=0;b<5;++b)late.storage[scale][(bit+b)/8]&=uint8_t(~(1u<<((bit+b)%8)));
        const auto planes=late.tensors();std::vector<uint8_t> out(late.raw.size(),0x35);
        CHECK(api.relayout(nullptr,0,0,&late.encoding,late.selection.data(),planes.data(),n,out.data(),out.size())<0);
        CHECK(std::all_of(out.begin(),out.end(),[](uint8_t b){return b==0x35;}));
    }
}
TEST(gguf_relayout_refuses_transforms_unknown_formats_and_source_destination_overlap) {
    constexpr auto unknown=static_cast<gg::Format>(999);
    const RadLayoutFn unknown_layout=&gg::abi_layout<unknown>;
    const RadRelayoutFn unknown_relayout=&gg::abi_relayout<unknown>;
    for(const auto api:hooks) {
        const Imported data(api);const auto good=data.tensors();const int n=data.encoding.n_planes;
        auto encoding=data.encoding;rad_enc_copy_str(encoding.transform,"fwht256");
        rejected(data,&encoding,data.selection.data(),good.data(),n);
        std::vector<uint8_t> out(data.raw.size(),0x68);RadLayout layout{};
        CHECK(unknown_layout(nullptr,0,0,&data.encoding,data.selection.data(),good.data(),n,&layout)<0);
        CHECK(unknown_relayout(nullptr,0,0,&data.encoding,data.selection.data(),good.data(),n,out.data(),out.size())<0);
        CHECK(std::all_of(out.begin(),out.end(),[](uint8_t b){return b==0x68;}));
        // Allocate both ranges for real: the ABI has no payload-capacity field, so fabricating
        // a larger destination span around a small source vector would be an invalid test.
        std::vector<uint8_t> shared(data.raw.size()+data.storage[0].size()+64,0x68);
        std::copy(data.storage[0].begin(),data.storage[0].end(),shared.begin()+32);
        auto planes=good;planes[0].data=shared.data()+32;const auto before=shared;
        CHECK(api.relayout(nullptr,0,0,&data.encoding,data.selection.data(),planes.data(),n,shared.data()+8,data.raw.size())<0);
        CHECK(shared==before);
    }
}
RAD_TEST_MAIN()
