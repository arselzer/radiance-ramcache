#include "rad_test.h"
#include "plugin/registry.h"
#include "../libds41/ds41_transform_fixture.h"
#include "../libds41/ds41_compress_fixture.h"
#include "../arch/common/rad_ds41_ref.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <limits>

namespace t=rad::ds41::transform;
namespace fix=t::testing;
namespace ref=rad::deepseek41::ref;
namespace {
const RadKernelInfo* actual_norm() {
    static rad::Registry registry;static bool loaded=false;
    if(!loaded) {loaded=true;if(registry.load_plugin(home()+"/kernels/libref.so",0)!=RAD_OK)throw std::runtime_error("missing libref");}
    for(const auto& row:registry.rows())if(!std::strcmp(row.info->op,"rmsnorm"))return row.info;
    throw std::runtime_error("missing actual RMSNorm");
}
void flatten(const nlohmann::json& j,std::vector<float>& out) {if(j.is_array())for(const auto& v:j)flatten(v,out);else out.push_back(j.get<float>());}
std::vector<float> values(const nlohmann::json& j) {std::vector<float> out;flatten(j,out);return out;}
const nlohmann::json& golden() {
    static const auto data=[] {std::ifstream file(std::string(RAD_GOLDEN_FIXTURES)+"/deepseek41-reference.json");
        if(!file)throw std::runtime_error("missing independent DS41 goldens");
        return nlohmann::json::parse(file);}();return data;
}
void expected(fix::Fixture& c) {
    const auto input=c.buffers[0].values();
    if(c.op==t::Op::rmsnorm) {
        const auto gain=c.buffers[1].values();std::vector<float> row(size_t(c.dim),0);
        for(int64_t r=0;r<c.rows;++r) {
            REQUIRE_EQ(ref::rms_norm(ref::Floats(input).subspan(size_t(r*c.dim),size_t(c.dim)),gain,1e-6f,row),RAD_OK);
            for(int64_t d=0;d<c.dim;++d)c.buffers[2].set(r*c.dim+d,row[size_t(d)]);
        }
    } else {
        const bool inverse=rad_param_geti(c.params.data(),c.params.size(),"inverse",0)!=0;
        const ref::Rope spec{size_t(c.rotary),c.spec.theta,c.spec.factor,c.spec.beta_fast,c.spec.beta_slow,c.spec.original_context};
        for(int64_t r=0;r<c.rows;++r)for(int64_t h=0;h<c.heads;++h) {
            std::vector<float> head(input.begin()+(r*c.heads+h)*c.dim,input.begin()+(r*c.heads+h+1)*c.dim);
            REQUIRE_EQ(ref::rotary(head,c.buffers[1].id(r),spec,inverse),RAD_OK);
            for(int64_t d=0;d<c.dim;++d)c.buffers[3].set((r*c.heads+h)*c.dim+d,head[size_t(d)]);
        }
    }
}
void compare(fix::Fixture& c) {
    auto oracle=c;expected(oracle);const auto before=c.buffers;REQUIRE_EQ(c.run(),RAD_OK);CHECK_EQ(c.flag[0],RAD_OK);
    for(size_t i=0;i+1<c.buffers.size();++i)CHECK(c.buffers[i].words==before[i].words);
    CHECK(c.buffers.back().words==oracle.buffers.back().words);CHECK_EQ(c.flag[1],912);CHECK_EQ(c.flag[2],913);CHECK_EQ(c.flag[3],914);
}
void reject(fix::Fixture& c,bool payload=true) {
    const auto before=c.buffers;const auto old=c.flag;CHECK(c.run()<0);
    for(size_t i=0;i<before.size();++i)CHECK(c.buffers[i].words==before[i].words);
    if(payload)CHECK(c.flag[0]<0);else CHECK(c.flag==old);
    CHECK_EQ(c.flag[1],912);CHECK_EQ(c.flag[2],913);CHECK_EQ(c.flag[3],914);
}
}

TEST(ds41_transform_norm_has_explicit_storage_boundaries_and_matches_actual_generic_norm) {
    for(uint32_t ad:{RAD_F32,RAD_F16,RAD_BF16})for(uint32_t yd:{RAD_F32,RAD_F16,RAD_BF16})for(int pitch=0;pitch<3;++pitch) {
        fix::Fixture c(t::Op::rmsnorm,ad,yd,pitch);compare(c);const auto reference=c.buffers.back().words;
        auto args=c.args();const RadParam params[]{RAD_INT("M",5),RAD_INT("n",c.dim),RAD_F64("eps",1e-6),RAD_STR("dtype",rad_dtype_name(yd)),RAD_F64("wadd",0),RAD_F64("mult",1)};
        args.p=params;args.n_p=6;REQUIRE_EQ(actual_norm()->launch(&args,nullptr),RAD_OK);CHECK(c.buffers.back().words==reference);
    }
}
TEST(ds41_transform_tail_pairs_inverse_yarn_and_pitches_match_independent_rotary) {
    for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(int pitch=0;pitch<3;++pitch)for(bool inverse:{false,true})for(bool yarn:{false,true}) {
        fix::Fixture c(t::Op::rope_tail,dtype,dtype,pitch,inverse);if(!yarn) {c.spec.original_context=0;c.rebuild();}compare(c);
    }
    fix::Fixture full(t::Op::rope_tail,RAD_F32,RAD_F32,2,false,4,4,1);compare(full);
}
TEST(ds41_transform_host_pair_builder_matches_independent_high_position_goldens) {
    for(const auto& g:golden()["rope"]) {
        const int64_t position=g["position"],rotary=g["dim"];
        const t::RopeSpec spec{rotary,g["original"],g["theta"],g["factor"],g["fast"],g["slow"]};
        std::vector<float> table;t::TableReceipt receipt;
        REQUIRE_EQ(t::build_pairs(spec,position+1,uint64_t((position+1)*rotary*4),&table,&receipt),RAD_OK);
        const auto x=values(g["x"]);const auto start=x.size()-size_t(rotary);
        for(bool inverse:{false,true}) {
            auto actual=x;const auto expected=values(g[inverse?"inverse":"forward"]);
            for(int64_t pair=0;pair<rotary/2;++pair) {
                const float cosine=table[size_t(position*rotary+2*pair)],sine=table[size_t(position*rotary+2*pair+1)]*(inverse?-1.f:1.f);
                const float a=x[start+size_t(2*pair)],b=x[start+size_t(2*pair+1)];
                actual[start+size_t(2*pair)]=a*cosine-b*sine;actual[start+size_t(2*pair+1)]=a*sine+b*cosine;
            }
            for(size_t i=0;i<x.size();++i)CHECK_NEAR(actual[i],expected[i],8e-4+4e-5*std::fabs(expected[i]));
        }
        CHECK_EQ(receipt.bytes,(position+1)*rotary*4);CHECK_EQ(receipt.positions,position+1);
    }
}
TEST(ds41_transform_table_budget_overflow_and_config_selection_are_checked_before_mutation) {
    t::RopeSpec spec{4,4096,40000,40,32,1};std::vector<float> out{11,12,13};const auto before=out;t::TableReceipt receipt{7,8,9};
    CHECK_EQ(t::build_pairs(spec,64,1023,&out,&receipt),RAD_E_NOMEM);CHECK(out==before);CHECK_EQ(receipt.bytes,9);
    CHECK(t::build_pairs(spec,INT64_MAX,UINT64_MAX,&out,&receipt)<0);CHECK(out==before);
    auto bad=spec;bad.rotary=INT64_MAX;CHECK(t::build_pairs(bad,1,UINT64_MAX,&out)<0);CHECK(out==before);
    bad=spec;bad.factor=std::numeric_limits<float>::denorm_min();CHECK(t::build_pairs(bad,64,1024,&out)<0);CHECK(out==before);
    CHECK(t::build_pairs(spec,0,UINT64_MAX,&out)<0);CHECK(t::build_pairs(spec,64,1024,nullptr)<0);
    rad::deepseek41::Config c;c.n_layers=2;c.layers.resize(2);c.layers[0].compress_ratio=0;c.layers[1].compress_ratio=2;
    c.rope_type="yarn";c.rope_head_dim=4;c.rope_theta=10000;c.compress_rope_theta=40000;c.original_seq_len=65536;c.rope_factor=40;c.beta_fast=32;c.beta_slow=1;
    t::RopeSpec selected;REQUIRE_EQ(t::select_spec(c,0,&selected),RAD_OK);CHECK_EQ(selected.theta,10000);CHECK_EQ(selected.original_context,0);
    REQUIRE_EQ(t::select_spec(c,1,&selected),RAD_OK);CHECK_EQ(selected.theta,40000);CHECK_EQ(selected.original_context,65536);
    CHECK(t::select_spec(c,2,&selected)<0);c.rope_type="other";CHECK(t::select_spec(c,0,&selected)<0);
}
TEST(ds41_transform_compressor_cast_before_norm_and_first_original_position_are_preserved) {
    namespace comp=rad::ds41::compress;comp::testing::Fixture compressed(5,2,RAD_F32,1,{7},{0},{2});REQUIRE_EQ(compressed.run(),RAD_OK);
    fix::Fixture norm(t::Op::rmsnorm,RAD_BF16,RAD_BF16,1,false,5);for(int d=0;d<5;++d)norm.buffers[1].set(d,1.f);
    std::vector<int32_t> firsts;int row=0;
    for(int input=0;input<7;++input)if(compressed.buffer(comp::emit).geti(input)) {
        firsts.push_back(compressed.buffer(comp::first_position).geti(input));
        for(int d=0;d<5;++d)norm.buffers[0].set(row*5+d,compressed.buffer(comp::pooled).getf(input*5+d));
        ++row;
    }
    REQUIRE_EQ(row,3);REQUIRE_EQ(norm.run(),RAD_OK);
    ref::Compressor oracle(2,std::vector<float>(5,1),1e-6f,"cast-boundary",ref::Precision::bf16);row=0;
    for(int input=0;input<7;++input) {
        std::vector<float> kv,score,latent;int64_t first=-1;
        for(int d=0;d<5;++d) {kv.push_back(compressed.buffer(comp::projected_kv).getf(input*5+d));score.push_back(compressed.buffer(comp::gate_scores).getf(input*5+d));}
        REQUIRE_EQ(oracle.push(kv,score,&latent,&first),RAD_OK);
        if(!latent.empty()) {CHECK_EQ(first,firsts[size_t(row)]);for(int d=0;d<5;++d)CHECK_EQ(norm.buffers[2].get(row*5+d),latent[size_t(d)]);++row;}
    }
    fix::Fixture rope(t::Op::rope_tail,RAD_BF16,RAD_BF16,1,false,5,4,1);
    for(int r=0;r<3;++r) {rope.buffers[1].id(r,firsts[size_t(r)]);for(int d=0;d<5;++d)rope.buffers[0].set(r*5+d,norm.buffers[2].get(r*5+d));}
    const auto unrotated=rope.buffers[0].words;compare(rope);CHECK(rope.buffers[0].words==unrotated);
}
TEST(ds41_transform_payload_errors_are_batch_atomic_and_unused_pairs_are_not_read) {
    for(auto op:{t::Op::rmsnorm,t::Op::rope_tail}) {fix::Fixture c(op);c.buffers[0].set(c.buffers[0].count()-1,NAN);reject(c);}
    fix::Fixture gain;gain.buffers[1].set(gain.dim-1,INFINITY);reject(gain);
    for(int pos:{-1,64,INT32_MAX}) {fix::Fixture c(t::Op::rope_tail);c.buffers[1].id(2,pos);reject(c);}
    fix::Fixture pair(t::Op::rope_tail);pair.buffers[2].set(37*4+3,NAN);reject(pair);
    fix::Fixture unit(t::Op::rope_tail);unit.buffers[2].set(37*4,2.f);reject(unit);
    fix::Fixture unseen(t::Op::rope_tail);unseen.buffers[2].set(63*4,NAN);compare(unseen);
}
TEST(ds41_transform_descriptor_failures_preserve_flag_and_output) {
    for(auto op:{t::Op::rmsnorm,t::Op::rope_tail}) {
        fix::Fixture c(op);auto args=c.args();t::Plan p;const auto tensors=c.tensors;
        c.tensors.back().data=c.tensors[0].data;CHECK(t::validate(op,&args,&p)<0);c.tensors=tensors;
        c.tensors[0].stride[0]=INT64_MAX;CHECK(t::validate(op,&args,&p)<0);c.tensors=tensors;
        c.tensors.back().dtype=RAD_U32;CHECK(t::validate(op,&args,&p)<0);c.tensors=tensors;
        args.scratch_bytes=3;CHECK(t::validate(op,&args,&p)<0);args=c.args();args.scratch=reinterpret_cast<void*>(UINTPTR_MAX-3);CHECK(t::validate(op,&args,&p)<0);
        CHECK_EQ(c.flag[0],911);CHECK_EQ(ds41_transform_scratch(nullptr),rad::ds41::gpu_scratch_bytes<t::Plan>());
    }
}
RAD_TEST_MAIN()
