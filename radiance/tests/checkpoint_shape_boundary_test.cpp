/* Adversarial source-shape checks: all GGUF fixtures are tiny, zero-payload files. */
#include "rad_test.h"
#include "format/checkpoint.h"
#include <fstream>
#include <stdexcept>
#include <unistd.h>

using namespace rad;
namespace {
struct Tensor { std::string name; std::vector<int64_t> shape; };
std::string zero_gguf(const std::vector<Tensor>& tensors) {
    std::string path=radtest::tmp_path("gguf_shape.XXXXXX.gguf");
    const int fd=mkstemps(path.data(),5);
    if(fd<0)throw std::runtime_error("mkstemps");
    close(fd);
    std::vector<uint8_t> bytes;
    auto put=[&]<class T>(T value) {
        const auto* p=reinterpret_cast<const uint8_t*>(&value);
        bytes.insert(bytes.end(),p,p+sizeof(value));
    };
    auto text=[&](const std::string& value) {
        put(uint64_t(value.size()));bytes.insert(bytes.end(),value.begin(),value.end());
    };
    put(uint32_t(0x46554747));put(uint32_t(3));put(uint64_t(tensors.size()));put(uint64_t(1));
    text("general.architecture");put(uint32_t(GGUF_STR));text("shape_fixture");
    for(const auto& t:tensors) {
        text(t.name);put(uint32_t(t.shape.size()));
        for(auto i=t.shape.rbegin();i!=t.shape.rend();++i)put(*i);
        put(uint32_t(0));put(uint64_t(0)); // F32, empty payload at the data section
    }
    bytes.resize((bytes.size()+31)/32*32);
    std::ofstream file(path,std::ios::binary);
    file.write(reinterpret_cast<const char*>(bytes.data()),(std::streamsize)bytes.size());
    if(!file)throw std::runtime_error("fixture write failed");
    return path;
}
int resolve(Checkpoint& ck,std::initializer_list<const char*> names,CkptWeight* out) {
    Program prog;RadNameMap nm{};nm.declared="logical";
    nm.mode=names.size()>1?RAD_MAP_CONCAT:RAD_MAP_COPY;nm.concat_dim=0;
    for(int& index:nm.src_index)index=-1;
    for(const auto* name:names)nm.src[nm.n_src++]=name;
    prog.name_map.push_back(nm);std::string why;
    return ckpt_resolve(ck,prog,"logical",false,out,&why);
}
}

TEST(checkpoint_source_numel_preserves_empty_shapes_without_multiplication_overflow) {
    SrcTensor t;
    CHECK_EQ(t.numel(),0);
    t.shape={INT64_MAX,INT64_MAX,0};CHECK_EQ(t.numel(),0);
    t.shape={0,INT64_MAX,INT64_MAX};CHECK_EQ(t.numel(),0);
    t.shape={INT64_MAX,1};CHECK_EQ(t.numel(),INT64_MAX);
    t.shape={INT64_MAX,2};CHECK_EQ(t.numel(),-1);
    t.shape={-1,0};CHECK_EQ(t.numel(),-1);
    t.shape={0,-1};CHECK_EQ(t.numel(),-1);
}
TEST(checkpoint_tiny_zero_payload_gguf_refuses_unrepresentable_logical_row_extent) {
    Checkpoint ck;
    REQUIRE_EQ(ck.open(zero_gguf({{"empty",{INT64_MAX,INT64_MAX,0}}})),RAD_OK);
    const auto* tensor=ck.find("empty");REQUIRE(tensor);
    CHECK_EQ(tensor->bytes,0);CHECK_EQ(tensor->numel(),0);
    CkptWeight out;CHECK_EQ(resolve(ck,{"empty"},&out),RAD_E_SHAPE);
}
TEST(checkpoint_tiny_zero_payload_gguf_refuses_overflowing_concat_extent) {
    Checkpoint ck;
    REQUIRE_EQ(ck.open(zero_gguf({{"a",{INT64_MAX,0}},{"b",{1,0}}})),RAD_OK);
    CkptWeight out;CHECK_EQ(resolve(ck,{"a","b"},&out),RAD_E_SHAPE);
}
TEST(checkpoint_representable_empty_source_and_join_keep_their_existing_shape) {
    Checkpoint ck;
    REQUIRE_EQ(ck.open(zero_gguf({{"a",{2,0}},{"b",{3,0}},{"leading_zero",{0,INT64_MAX,INT64_MAX,0}}})),RAD_OK);
    CkptWeight out;
    REQUIRE_EQ(resolve(ck,{"a"},&out),RAD_OK);CHECK_EQ(out.rows,2);CHECK_EQ(out.cols,0);
    REQUIRE_EQ(resolve(ck,{"a","b"},&out),RAD_OK);CHECK_EQ(out.rows,5);CHECK_EQ(out.cols,0);
    REQUIRE_EQ(resolve(ck,{"leading_zero"},&out),RAD_OK);CHECK_EQ(out.rows,0);CHECK_EQ(out.cols,0);
}

RAD_TEST_MAIN()
