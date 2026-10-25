#include "rad_test.h"
#include "../libds41/ds41_compress_fixture.h"
#include "../arch/common/rad_ds41_ref.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <map>
#include <memory>

namespace c=rad::ds41::compress;
namespace fix=c::testing;
namespace ref=rad::deepseek41::ref;
namespace {
void flatten(const nlohmann::json& j,std::vector<float>& out) {if(j.is_array())for(const auto& v:j)flatten(v,out);else out.push_back(j.get<float>());}
std::vector<float> floats(const nlohmann::json& j) {std::vector<float> out;flatten(j,out);return out;}
const nlohmann::json& golden() {
    static const auto value=[] {std::ifstream f(std::string(RAD_GOLDEN_FIXTURES)+"/deepseek41-reference.json");
        if(!f)throw std::runtime_error("missing independent compressor goldens");
        return nlohmann::json::parse(f);}();return value;
}
void unchanged(const fix::Fixture& a,const fix::Fixture& b) {
    for(int i=0;i<c::tensors_count;++i) {
        REQUIRE_EQ(a.buffers[size_t(i)].has_value(),b.buffers[size_t(i)].has_value());
        if(a.buffers[size_t(i)])CHECK(a.buffer(i).words==b.buffer(i).words);
    }
}
void fail(fix::Fixture& f,bool payload=true) {
    const auto before=f;CHECK(f.run()<0);unchanged(f,before);
    if(payload)CHECK(f.flag[0]<0);else CHECK(f.flag==before.flag);
    CHECK_EQ(f.flag[1],912);CHECK_EQ(f.flag[2],913);CHECK_EQ(f.flag[3],914);
}
void check_chunk(fix::Fixture& f,std::map<int,std::unique_ptr<ref::Compressor>>& refs) {
    const auto before=f;REQUIRE_EQ(f.run(),RAD_OK);CHECK_EQ(f.flag[0],RAD_OK);CHECK_EQ(f.flag[1],912);
    for(int i=0;i<c::tensors_count;++i)if(i!=c::pending_state&&i<c::pooled&&f.buffers[size_t(i)])CHECK(f.buffer(i).words==before.buffer(i).words);
    int64_t row=0;
    for(int64_t s=0;s<f.sequences;++s) {
        const int slot=f.slots[size_t(s)],length=f.lengths[size_t(s)];if(!length)continue;
        if(!refs.count(slot))refs.emplace(slot,std::make_unique<ref::Compressor>(size_t(f.ratio),std::vector<float>(size_t(f.dimension),1.f),1e-6f,"slot"+std::to_string(slot)));
        auto& reference=*refs.at(slot);
        for(int token=0;token<length;++token,++row) {
            std::vector<float> kv,gates;
            for(int64_t d=0;d<f.dimension;++d) {kv.push_back(f.buffer(c::projected_kv).getf(row*f.dimension+d));if(f.ratio==2)gates.push_back(f.buffer(c::gate_scores).getf(row*f.dimension+d));}
            const auto prior=reference.snapshot();std::vector<float> latent;int64_t position=-1;
            REQUIRE_EQ(reference.push(kv,gates,&latent,&position),RAD_OK);
            CHECK_EQ(f.buffer(c::emit).geti(row),latent.empty()?0:1);CHECK_EQ(f.buffer(c::first_position).geti(row),position);
            std::vector<float> actual;
            for(int64_t d=0;d<f.dimension;++d) {
                const float got=f.buffer(c::pooled).getf(row*f.dimension+d);actual.push_back(got);
                if(latent.empty()) {CHECK_EQ(got,0.f);CHECK(!std::signbit(got));}
                else if(f.ratio==1)CHECK_EQ(got,kv[size_t(d)]);
                else {
                    const float maximum=std::max(prior.scores[size_t(d)],gates[size_t(d)]);
                    const float a=std::exp(prior.scores[size_t(d)]-maximum),b=std::exp(gates[size_t(d)]-maximum);
                    CHECK_EQ(got,prior.kv[size_t(d)]*(a/(a+b))+kv[size_t(d)]*(b/(a+b)));
                }
            }
            if(!latent.empty()) {
                std::vector<float> normalized(actual.size());
                REQUIRE_EQ(ref::rms_norm(actual,std::vector<float>(actual.size(),1.f),1e-6f,normalized),RAD_OK);
                for(size_t d=0;d<latent.size();++d)CHECK_NEAR(normalized[d],latent[d],2e-6);
            }
        }
        const auto snapshot=reference.snapshot();CHECK_EQ(f.state_get(slot,0,f.dimension),uint32_t(reference.next_position()));
        CHECK_EQ(f.state_get(slot,1,f.dimension),c::magic(f.ratio));
        if(snapshot.pending)for(int64_t d=0;d<f.dimension;++d) {
            CHECK_EQ(f.state_get(slot,0,d),std::bit_cast<uint32_t>(snapshot.kv[size_t(d)]));
            CHECK_EQ(f.state_get(slot,1,d),std::bit_cast<uint32_t>(snapshot.scores[size_t(d)]));
        }
    }
    CHECK_EQ(row,f.tokens);
}
}

TEST(ds41_compress_pool_then_norm_matches_independent_numpy_goldens) {
    for(const auto& g:golden()["compression"]) {
        const int ratio=g.at("ratio");fix::Fixture f(5,ratio,RAD_F32,2,{7},{0},{2});
        const auto kv=floats(g["kv"]),scores=floats(g["scores"]),norm=floats(g["norm"]),expected=floats(g["latents"]);
        for(size_t i=0;i<kv.size();++i)f.buffer(c::projected_kv).setf(i,kv[i]);
        if(ratio==2)for(size_t i=0;i<scores.size();++i)f.buffer(c::gate_scores).setf(i,scores[i]);
        REQUIRE_EQ(f.run(),RAD_OK);std::vector<float> all;
        for(int64_t row=0;row<7;++row)if(f.buffer(c::emit).geti(row)) {
            std::vector<float> pool(5),normalized(5);for(int d=0;d<5;++d)pool[d]=f.buffer(c::pooled).getf(row*5+d);
            REQUIRE_EQ(ref::rms_norm(pool,norm,1e-6f,normalized),RAD_OK);all.insert(all.end(),normalized.begin(),normalized.end());
        }
        REQUIRE_EQ(all.size(),expected.size());for(size_t i=0;i<all.size();++i)CHECK_NEAR(all[i],expected[i],3e-6+2e-5*std::fabs(expected[i]));
    }
}
TEST(ds41_compress_packed_sequences_odd_chunks_reordering_and_state_slots_match_compressor) {
    for(int ratio:{1,2})for(int64_t dim:{1,128,512})for(int pitch=0;pitch<3;++pitch) {
        std::map<int,std::unique_ptr<ref::Compressor>> refs;
        fix::Fixture a(dim,ratio,RAD_F32,pitch,{1,3,0},{0,0,0},{2,0,-1});check_chunk(a,refs);
        fix::Fixture b(dim,ratio,RAD_F32,pitch,{2,3,0},{3,1,0},{0,2,-1});b.buffer(c::pending_state).words=a.buffer(c::pending_state).words;check_chunk(b,refs);
        fix::Fixture d(dim,ratio,RAD_F32,pitch,{1,1},{4,5},{2,0});d.buffer(c::pending_state).words=b.buffer(c::pending_state).words;check_chunk(d,refs);
        fix::Fixture e(dim,ratio,RAD_F32,pitch,{2,3},{6,5},{0,2});e.buffer(c::pending_state).words=d.buffer(c::pending_state).words;check_chunk(e,refs);
    }
}
TEST(ds41_compress_ratio_one_widens_supplied_projection_without_hidden_rounding) {
    for(uint32_t dtype:{RAD_F32,RAD_F16,RAD_BF16})for(int pitch=0;pitch<3;++pitch) {
        fix::Fixture f(128,1,dtype,pitch);std::map<int,std::unique_ptr<ref::Compressor>> refs;check_chunk(f,refs);
    }
}
TEST(ds41_compress_equal_and_extreme_column_gates_keep_exact_scalar_pooling) {
    fix::Fixture f(5,2,RAD_F32,2,{2},{0},{2});
    for(int d=0;d<5;++d) {
        f.buffer(c::projected_kv).setf(d,float(d));f.buffer(c::projected_kv).setf(5+d,float(8-d));
        f.buffer(c::gate_scores).setf(d,d==0?0:d&1?1000.f:-1000.f);
        f.buffer(c::gate_scores).setf(5+d,d==0?0:d&1?-1000.f:1000.f);
    }
    REQUIRE_EQ(f.run(),RAD_OK);
    CHECK_EQ(f.buffer(c::pooled).getf(5),4.f);
    for(int d=1;d<5;++d)CHECK_EQ(f.buffer(c::pooled).getf(5+d),d&1?float(d):float(8-d));
}
TEST(ds41_compress_warm_cursor_preserves_nan_shaped_u32_bits_and_ignores_even_pending) {
    for(int32_t start:{16777217,2139095041,INT32_MAX-1}) {
        fix::Fixture f(5,2,RAD_F32,2,{1},{start},{2});f.seed(2,start,(start&1)!=0);
        REQUIRE_EQ(f.run(),RAD_OK);CHECK_EQ(f.state_get(2,0,5),uint32_t(start)+1);
        CHECK_EQ(f.buffer(c::emit).geti(0),start&1);CHECK_EQ(f.buffer(c::first_position).geti(0),(start&1)?start-1:-1);
    }
    fix::Fixture f(128,2,RAD_F32,1,{3},{2},{2});f.seed(2,2,false);REQUIRE_EQ(f.run(),RAD_OK);
}
TEST(ds41_compress_whole_batch_preflight_keeps_all_state_and_outputs_atomic) {
    for(int fault=0;fault<16;++fault) {
        fix::Fixture f;const int64_t last=f.tokens-1;
        if(fault==0)f.buffer(c::projected_kv).setf(f.tokens*f.dimension-1,NAN);
        if(fault==1)f.buffer(c::gate_scores).setf(f.tokens*f.dimension-1,INFINITY);
        if(fault==2)f.buffer(c::state_ids).seti(1,4);
        if(fault==3)f.buffer(c::state_ids).seti(1,-1);
        if(fault==4)f.buffer(c::state_ids).seti(1,2);
        if(fault==5)f.buffer(c::cu).seti(0,1);
        if(fault==6)f.buffer(c::cu).seti(f.sequences,f.tokens+1);
        if(fault==7)f.buffer(c::cu).seti(1,f.tokens+1);
        if(fault==8)f.buffer(c::positions).seti(last,7);
        if(fault==9)f.buffer(c::positions).seti(last,INT32_MAX);
        if(fault==10)f.buffer(c::positions).seti(0,-1);
        if(fault==11)f.buffer(c::has_init).seti(1,-1);
        if(fault==12)f.buffer(c::num_accepted).seti(1,2);
        if(fault==13)f.buffer(c::has_init).seti(1,1); // no initialized cursor/tag
        if(fault==14) {f.buffer(c::has_init).seti(1,3);f.seed(0,0);f.state_set(0,0,5,1);}
        if(fault==15) {f.buffer(c::has_init).seti(1,3);f.seed(0,0);f.state_set(0,1,5,c::magic(1));}
        fail(f);
    }
    fix::Fixture odd(5,2,RAD_F32,2,{1},{1},{2});odd.seed(2,1,false);fail(odd);
}
TEST(ds41_compress_snapshot_restore_and_stale_cold_reset_use_raw_u32_state) {
    fix::Fixture a(128,2,RAD_F32,2,{1},{0},{2});REQUIRE_EQ(a.run(),RAD_OK);
    const auto snapshot=a.buffer(c::pending_state).words;
    fix::Fixture b(128,2,RAD_F32,2,{3},{1},{2});b.buffer(c::pending_state).words=snapshot;REQUIRE_EQ(b.run(),RAD_OK);
    fix::Fixture replay(128,2,RAD_F32,2,{3},{1},{2});replay.buffer(c::pending_state).words=snapshot;REQUIRE_EQ(replay.run(),RAD_OK);
    CHECK(b.buffer(c::pending_state).words==replay.buffer(c::pending_state).words);
    CHECK(b.buffer(c::pooled).words==replay.buffer(c::pooled).words);
    fix::Fixture reset(128,2,RAD_F32,2,{1},{0},{2});reset.buffer(c::pending_state).words=b.buffer(c::pending_state).words;REQUIRE_EQ(reset.run(),RAD_OK);
    CHECK_EQ(reset.state_get(2,0,128),1u);
}
TEST(ds41_compress_schema_metadata_and_profile_reject_speculation_and_numeric_state_casts) {
    RadKVGroupDecl d{};REQUIRE_EQ(c::state_decl(512,0,0,&d),RAD_OK);CHECK_EQ(d.dtype,uint32_t(RAD_U32));
    CHECK_EQ(d.kind,RAD_KV_LINEAR);CHECK_EQ(d.state_dim[0],2);CHECK_EQ(d.state_dim[1],513);
    CHECK_EQ(2*513*rad_dtype_bytes(d.dtype,1),4104);CHECK(c::state_decl(512,RAD_F16,0,&d)<0);CHECK(c::state_decl(512,RAD_F32,0,&d)<0);
    CHECK(c::state_decl(512,0,1,&d)<0);CHECK(c::state_decl(0,0,0,&d)<0);
    CHECK_EQ(c::schema.n_operands,11);CHECK_EQ(c::operands[c::pending_state].role,RAD_OPD_INOUT);
    fix::Fixture f;f.params.back().ival=1;fail(f,false);
    fix::Fixture no_gate(5,2);no_gate.buffers[c::gate_scores].reset();fail(no_gate,false);
    fix::Fixture typed(5,2,RAD_BF16);fail(typed,false);
    fix::Fixture opaque;opaque.buffer(c::pending_state).t.dtype=RAD_F32;fail(opaque,false);
    fix::Fixture optional(5,1,RAD_F16,1,{3},{0},{2},false);REQUIRE_EQ(optional.run(),RAD_OK);
    fix::Fixture meta;auto a=meta.args();c::Plan p;
    a.scratch_bytes=3;CHECK(c::validate(&a,&p)<0);a=meta.args();a.scratch=meta.tensors[0].data;CHECK(c::validate(&a,&p)<0);
    a=meta.args();meta.tensors[c::pooled].data=meta.tensors[c::projected_kv].data;CHECK(c::validate(&a,&p)<0);
    CHECK_EQ(meta.flag[0],911);CHECK_EQ(ds41_compress_scratch(nullptr),rad::ds41::gpu_scratch_bytes<c::Plan>());
}
RAD_TEST_MAIN()
