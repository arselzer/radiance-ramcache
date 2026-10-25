/* Split GGUF checkpoints and sidecars, with real files and mapped payloads. */
#include "rad_test.h"
#include "format/checkpoint.h"
#include <filesystem>
#include <stdexcept>
#include <cstring>
#include <unistd.h>

using namespace rad;

namespace {
struct Bytes {
    std::vector<uint8_t> data;
    template<class T> void put(T value) {
        const auto* p = reinterpret_cast<const uint8_t*>(&value);
        data.insert(data.end(),p,p+sizeof(value));
    }
    void string(const std::string& s) {
        put<uint64_t>(s.size());data.insert(data.end(),s.begin(),s.end());
    }
    void pad(size_t n) { data.resize((data.size()+n-1)/n*n); }
};
struct Kv { std::string key; uint32_t type; Bytes value; };
Kv integer(const char* key,int64_t value) {
    Kv kv{key,GGUF_I64,{}};kv.value.put<int64_t>(value);return kv;
}
Kv text(const char* key,const std::string& value) {
    Kv kv{key,GGUF_STR,{}};kv.value.string(value);return kv;
}
Kv integers(const char* key,const std::vector<int64_t>& values) {
    Kv kv{key,GGUF_ARR,{}};kv.value.put<uint32_t>(GGUF_I64);kv.value.put<uint64_t>(values.size());
    for (auto value:values) kv.value.put<int64_t>(value);
    return kv;
}
Kv floats(const char* key,const std::vector<double>& values) {
    Kv kv{key,GGUF_ARR,{}};kv.value.put<uint32_t>(GGUF_F64);kv.value.put<uint64_t>(values.size());
    for (auto value:values) kv.value.put<double>(value);
    return kv;
}
std::string directory() {
    std::string path=radtest::tmp_path("gguf_ckpt.XXXXXX");
    if (!mkdtemp(path.data())) throw std::runtime_error("mkdtemp failed");
    return path;
}
struct Tensor { std::string name; float value; };
void write(const std::string& path,const std::vector<Kv>& kvs,const std::vector<Tensor>& tensors) {
    Bytes bytes;bytes.put<uint32_t>(0x46554747);bytes.put<uint32_t>(3);
    bytes.put<uint64_t>(tensors.size());bytes.put<uint64_t>(kvs.size());
    for (const Kv& kv:kvs) {
        bytes.string(kv.key);bytes.put<uint32_t>(kv.type);
        bytes.data.insert(bytes.data.end(),kv.value.data.begin(),kv.value.data.end());
    }
    for (size_t i=0;i<tensors.size();++i) {
        bytes.string(tensors[i].name);bytes.put<uint32_t>(1);bytes.put<uint64_t>(1);
        bytes.put<uint32_t>(0);bytes.put<uint64_t>(32*i);
    }
    bytes.pad(32);
    for (const Tensor& tensor:tensors) { bytes.put<float>(tensor.value);bytes.pad(32); }
    FILE* file=std::fopen(path.c_str(),"wb");
    if (!file) throw std::runtime_error("cannot write fixture");
    std::fwrite(bytes.data.data(),1,bytes.data.size(),file);std::fclose(file);
}
std::vector<Kv> split(int64_t number,int64_t count=2,int64_t total=2) {
    auto kvs=std::vector<Kv>{integer("split.no",number),integer("split.count",count),
                             integer("split.tensors.count",total)};
    if (number==0) kvs.push_back(text("general.architecture","deepseek41"));
    return kvs;
}
float value(const SrcTensor* tensor) {
    float out=0;if (tensor) std::memcpy(&out,tensor->data,4);return out;
}
}

TEST(split_gguf_reads_payloads_from_every_mapping) {
    const auto dir=directory();
    write(dir+"/model-00001-of-00002.gguf",split(0),{{"first",1.25f}});
    write(dir+"/model-00002-of-00002.gguf",split(1),{{"last",7.5f}});
    Checkpoint ck;CHECK_OK(ck.open(dir+"/model-00001-of-00002.gguf"));
    CHECK_EQ(ck.tensors().size(),2u);CHECK_EQ(value(ck.find("first")),1.25f);
    CHECK_EQ(value(ck.find("last")),7.5f);CHECK_EQ(ck.meta().arch,std::string("deepseek41"));
    CHECK_EQ(std::string(ck.meta().get("source.gguf.shard_count")),std::string("2"));
}
TEST(split_gguf_accepts_a_metadata_only_first_shard) {
    const auto dir=directory();
    write(dir+"/model-00001-of-00002.gguf",split(0),{});
    write(dir+"/model-00002-of-00002.gguf",split(1),{{"one",1},{"two",2}});
    Checkpoint ck;CHECK_OK(ck.open(dir+"/model-00001-of-00002.gguf"));
    CHECK_EQ(ck.tensors().size(),2u);CHECK_EQ(value(ck.find("two")),2);
}
TEST(split_gguf_requires_the_first_shard) {
    const auto dir=directory();write(dir+"/model-00002-of-00002.gguf",split(1),{{"last",2}});
    Checkpoint ck;CHECK(ck.open(dir+"/model-00002-of-00002.gguf")<0);
}
TEST(split_gguf_rejects_missing_part) {
    const auto dir=directory();write(dir+"/model-00001-of-00002.gguf",split(0),{{"first",1}});
    Checkpoint ck;CHECK(ck.open(dir+"/model-00001-of-00002.gguf")<0);
}
TEST(split_gguf_rejects_a_misnumbered_part) {
    const auto dir=directory();
    write(dir+"/model-00001-of-00002.gguf",split(0),{{"first",1}});
    write(dir+"/model-00002-of-00002.gguf",split(0),{{"last",2}});
    Checkpoint ck;CHECK(ck.open(dir+"/model-00001-of-00002.gguf")<0);
}
TEST(split_gguf_rejects_duplicate_tensors_across_parts) {
    const auto dir=directory();
    write(dir+"/model-00001-of-00002.gguf",split(0),{{"same",1}});
    write(dir+"/model-00002-of-00002.gguf",split(1),{{"same",2}});
    Checkpoint ck;CHECK(ck.open(dir+"/model-00001-of-00002.gguf")<0);
}
TEST(split_gguf_rejects_incorrect_total_count) {
    const auto dir=directory();
    write(dir+"/model-00001-of-00002.gguf",split(0,2,3),{{"first",1}});
    write(dir+"/model-00002-of-00002.gguf",split(1,2,3),{{"last",2}});
    Checkpoint ck;CHECK(ck.open(dir+"/model-00001-of-00002.gguf")<0);
}
TEST(split_gguf_rejects_conflicting_architecture_metadata) {
    const auto dir=directory();auto last=split(1);last.push_back(text("general.architecture","other"));
    write(dir+"/model-00001-of-00002.gguf",split(0),{{"first",1}});
    write(dir+"/model-00002-of-00002.gguf",last,{{"last",2}});
    Checkpoint ck;CHECK(ck.open(dir+"/model-00001-of-00002.gguf")<0);
}
TEST(split_gguf_rejects_filename_metadata_disagreement) {
    const auto dir=directory();write(dir+"/model-00001-of-00003.gguf",split(0),{{"first",1}});
    Checkpoint ck;CHECK(ck.open(dir+"/model-00001-of-00003.gguf")<0);
}
TEST(gguf_preserves_complete_architecture_metadata) {
    const auto dir=directory();std::vector<int64_t> schedule(43,2);schedule[42]=0;
    std::string expected;for (size_t i=0;i<schedule.size();++i)expected+=(i?",":"")+std::to_string(schedule[i]);
    write(dir+"/model.gguf",{text("general.architecture","deepseek41"),
          integers("deepseek41.attention.compress_ratios",schedule),
          integer("deepseek41.expert_feed_forward_length",2304),
          floats("deepseek41.clamps",{.25,.5,.75,1,2,3,4,5,6,7}),text("long_config",std::string(4096,'x'))},{});
    Checkpoint ck;CHECK_OK(ck.open(dir+"/model.gguf"));
    CHECK_EQ(std::string(ck.meta().get("deepseek41.attention.compress_ratios")),expected);
    CHECK_EQ(std::string(ck.meta().get("deepseek41.clamps")),std::string("0.25,0.5,0.75,1,2,3,4,5,6,7"));
    CHECK_EQ(std::string(ck.meta().get("long_config")).size(),4096u);
    CHECK_EQ(ck.meta().m.n_ff,2304);
}
TEST(gguf_preserves_a_full_engram_token_map) {
    const auto dir=directory();std::vector<int64_t> map(129280);
    for(size_t i=0;i<map.size();++i)map[i]=(int64_t)i;
    write(dir+"/model.gguf",{text("general.architecture","deepseek41"),
          integers("deepseek41.ple.token_map",map)},{});
    Checkpoint ck;CHECK_OK(ck.open(dir+"/model.gguf"));
    const auto* raw=ck.meta().get("deepseek41.ple.token_map");REQUIRE(raw);
    const std::string stored(raw);CHECK_EQ(std::count(stored.begin(),stored.end(),','),129279);
    CHECK(stored.starts_with("0,1,2,"));CHECK(stored.ends_with(",129279"));
}
TEST(gguf_sidecar_keeps_core_metadata_and_mapping_alive) {
    const auto dir=directory();
    write(dir+"/model.gguf",{text("general.architecture","deepseek41")},{{"core",2}});
    write(dir+"/engram.gguf",{text("general.architecture","deepseek41-engram")},{{"lookup",3}});
    Checkpoint ck;CHECK_OK(ck.open(dir+"/model.gguf"));CHECK_OK(ck.add_file(dir+"/engram.gguf"));
    CHECK_EQ(ck.meta().arch,std::string("deepseek41"));CHECK_EQ(ck.tensors().size(),2u);
    CHECK_EQ(value(ck.find("core")),2);CHECK_EQ(value(ck.find("lookup")),3);
}
TEST(gguf_sidecar_duplicate_is_rejected_without_partial_merge) {
    const auto dir=directory();
    write(dir+"/model.gguf",{text("general.architecture","deepseek41")},{{"core",2}});
    write(dir+"/engram.gguf",{},{{"new",1},{"core",3}});
    Checkpoint ck;CHECK_OK(ck.open(dir+"/model.gguf"));CHECK(ck.add_file(dir+"/engram.gguf")<0);
    CHECK_EQ(ck.tensors().size(),1u);CHECK(ck.find("new")==nullptr);CHECK_EQ(value(ck.find("core")),2);
}
TEST(gguf_sidecar_cannot_silently_add_one_part_of_a_split) {
    const auto dir=directory();
    write(dir+"/model.gguf",{text("general.architecture","deepseek41")},{{"core",2}});
    write(dir+"/engram.gguf",split(0),{{"lookup",3}});
    Checkpoint ck;CHECK_OK(ck.open(dir+"/model.gguf"));CHECK(ck.add_file(dir+"/engram.gguf")<0);
    CHECK_EQ(ck.tensors().size(),1u);
}

TEST(split_gguf_rejects_array_valued_split_fields) {
    const auto dir=directory();auto kvs=split(0);kvs[1]=integers("split.count",{2});
    write(dir+"/model-00001-of-00002.gguf",kvs,{{"first",1}});
    Checkpoint ck;CHECK(ck.open(dir+"/model-00001-of-00002.gguf")<0);
}
TEST(gguf_sidecar_rejects_partial_or_inconsistent_split_metadata) {
    const auto dir=directory();write(dir+"/model.gguf",{text("general.architecture","deepseek41")},{{"core",2}});
    Checkpoint ck;CHECK_OK(ck.open(dir+"/model.gguf"));
    write(dir+"/partial.gguf",{integer("split.no",1)},{{"lookup",3}});
    CHECK(ck.add_file(dir+"/partial.gguf")<0);
    write(dir+"/wrong-total.gguf",split(0,1,2),{{"lookup",3}});
    CHECK(ck.add_file(dir+"/wrong-total.gguf")<0);CHECK_EQ(ck.tensors().size(),1u);
}

RAD_TEST_MAIN()
