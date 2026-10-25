/* End-to-end serial FP32 attention vs an independent full-history NumPy oracle. Tiny weights
 * and projections are synthetic; this tests semantics/state, not packed precision or quality. */
#include "rad_test.h"
#include "../arch/deepseek41/attention_ref.h"
#include <nlohmann/json.hpp>
#include <fstream>

namespace attn=rad::deepseek41::attention;
namespace ds=rad::deepseek41;
namespace ref=rad::deepseek41::ref;
namespace {
using Json=nlohmann::json;
const Json& fixture() {
    static const Json data=[] {
        std::ifstream file(std::string(RAD_GOLDEN_FIXTURES)+"/deepseek41-attention.json");
        if(!file)throw std::runtime_error("missing attention oracle fixture");
        return Json::parse(file);
    }();return data;
}
void flatten(const Json& value,std::vector<float>* out) {
    if(value.is_array())for(const auto& part:value)flatten(part,out);
    else out->push_back(value.get<float>());
}
std::vector<float> values(const Json& value) {std::vector<float> out;flatten(value,&out);return out;}
ds::Config config() {
    ds::Config c;const auto& j=fixture().at("config");
#define FIELD(name) c.name=j.at(#name).get<decltype(c.name)>()
    FIELD(n_layers);FIELD(n_embd);FIELD(n_head);FIELD(n_head_kv);FIELD(head_dim);FIELD(rope_head_dim);
    FIELD(q_lora_rank);FIELD(o_lora_rank);FIELD(o_groups);FIELD(n_ff);FIELD(n_vocab);FIELD(context_length);
    FIELD(n_expert);FIELD(n_expert_used);FIELD(n_expert_shared);FIELD(window_size);FIELD(index_n_heads);
    FIELD(index_head_dim);FIELD(index_topk);FIELD(norm_eps);FIELD(route_scale);FIELD(gate_temp);
    FIELD(norm_topk_prob);FIELD(score_func);FIELD(routed_clamp);FIELD(shared_clamp);FIELD(rope_type);
    FIELD(rope_theta);FIELD(compress_rope_theta);FIELD(rope_factor);FIELD(original_seq_len);
    FIELD(beta_fast);FIELD(beta_slow);FIELD(hc_mult);FIELD(hc_sinkhorn_iters);FIELD(hc_eps);FIELD(hc_shifted);
    FIELD(candidate_source);FIELD(candidate_topk_blocks);FIELD(candidate_block_size);FIELD(compress_ratios);
    FIELD(kv_source_layers);FIELD(index_k_source_layers);FIELD(index_source_layers);FIELD(draft_layers);FIELD(engram_present);
#undef FIELD
    return c;
}
std::vector<attn::LayerWeights> weights() {
    std::vector<attn::LayerWeights> result;
    for(const auto& j:fixture().at("weights")) {
        attn::LayerWeights w;
#define WEIGHT(name) w.name=values(j.at(#name))
        WEIGHT(q_a);WEIGHT(q_norm);WEIGHT(q_b);WEIGHT(kv);WEIGHT(kv_norm);WEIGHT(out_a);WEIGHT(out_b);WEIGHT(sinks);
        WEIGHT(compress_kv);WEIGHT(compress_gate);WEIGHT(compress_norm);WEIGHT(index_q);WEIGHT(index_head_weights);
        WEIGHT(index_k);WEIGHT(index_k_norm);
#undef WEIGHT
        result.push_back(std::move(w));
    }return result;
}
std::shared_ptr<const attn::Model> model() {
    static const auto result=[] {
        std::shared_ptr<const attn::Model> m;std::string why;
        if(attn::Model::create(config(),weights(),"independent-attention-fixture",&m,&why)!=RAD_OK)
            throw std::runtime_error(why);
        return m;
    }();return result;
}
void compare(ref::Floats actual,const Json& expected) {
    const auto want=values(expected);REQUIRE_EQ(actual.size(),want.size());
    for(size_t i=0;i<want.size();++i)CHECK_NEAR(actual[i],want[i],2e-5+2e-5*std::fabs(want[i]));
}
float difference(ref::Floats a,ref::Floats b) {
    if(a.size()!=b.size())return INFINITY;
    float largest=0;
    for(size_t i=0;i<a.size();++i)largest=std::max(largest,std::fabs(a[i]-b[i]));
    return largest;
}
std::vector<float> input(int token,int layer,float multiplier=1) {
    auto x=values(fixture().at("inputs").at(token).at(layer));for(float& v:x)v*=multiplier;return x;
}
int token(attn::Sequence* seq,int position,std::vector<std::vector<float>>* outputs=nullptr,float multiplier=1) {
    if(outputs)outputs->clear();
    for(int layer=0;layer<seq->model()->config().n_layers;++layer) {
        auto x=input(position,layer,multiplier);std::vector<float> out(x.size());
        const int status=seq->step(layer,position,x,out);if(status!=RAD_OK)return status;
        if(outputs)outputs->push_back(std::move(out));
    }return RAD_OK;
}
}
TEST(deepseek41_attention_serial_intermediates_match_independent_causal_prefill_oracle) {
    const auto m=model();attn::Sequence seq(m,"golden",24);
    for(int pos=0;pos<(int)fixture()["inputs"].size();++pos)for(int layer=0;layer<m->config().n_layers;++layer) {
        auto x=input(pos,layer);std::vector<float> out(x.size());attn::Trace tr;std::string why;
        REQUIRE_EQ(seq.step(layer,pos,x,out,&tr,&why),RAD_OK);
        const auto& want=fixture()["traces"][pos][layer];
#define CHECK_TRACE(name) compare(tr.name,want.at(#name))
        CHECK_TRACE(q_latent);CHECK_TRACE(query);CHECK_TRACE(window_row);CHECK_TRACE(compressed_latent);
        CHECK_TRACE(compressed_row);CHECK_TRACE(index_query);CHECK_TRACE(index_key);CHECK_TRACE(scores);
        CHECK_TRACE(attended);CHECK_TRACE(unrotated_heads);CHECK_TRACE(output);
#undef CHECK_TRACE
        CHECK(tr.compressed_indices==want.at("compressed_indices").get<std::vector<int32_t>>());
        CHECK(tr.window_indices==want.at("window_indices").get<std::vector<int32_t>>());
        CHECK(tr.combined_indices==want.at("combined_indices").get<std::vector<int32_t>>());
        CHECK(tr.candidates==want.at("candidates").get<std::vector<uint8_t>>());
        CHECK_EQ(tr.compressed_first_position,want.at("compressed_first_position").get<int64_t>());
        compare(out,want.at("output"));
    }
    CHECK_EQ(seq.next_position(),18);CHECK_EQ(seq.next_layer(),0);
}
TEST(deepseek41_attention_uses_prerope_latents_and_removes_query_rotation_before_output_projection) {
    const auto m=model();attn::Sequence seq(m,"ordering",24);bool tested_key=false,tested_output=false;
    for(int pos=0;pos<5;++pos)for(int layer=0;layer<6;++layer) {
        auto x=input(pos,layer);std::vector<float> out(4);attn::Trace tr;
        REQUIRE_EQ(seq.step(layer,pos,x,out,&tr),RAD_OK);
        if(layer==2 && tr.compressed_first_position>0) {
            const auto& w=m->layer(layer);std::vector<float> wrong(4);
            REQUIRE_EQ(ref::linear(tr.compressed_row,w.index_k,4,wrong),RAD_OK);
            REQUIRE_EQ(ref::rms_norm(wrong,w.index_k_norm,1e-6f,wrong),RAD_OK);
            const ref::Rope rope{4,100,2,32,1,4};
            REQUIRE_EQ(ref::rotary(wrong,tr.compressed_first_position,rope),RAD_OK);
            CHECK(difference(wrong,tr.index_key)>1e-3f);tested_key=true;
        }
        if(pos==3 && layer==2) {
            const auto& w=m->layer(layer);std::vector<float> wrong(4);
            REQUIRE_EQ(ref::grouped_output(tr.attended,2,4,2,3,w.out_a,w.out_b,wrong),RAD_OK);
            CHECK(difference(wrong,out)>1e-3f);tested_output=true;
        }
    }
    CHECK(tested_key);CHECK(tested_output);
}
TEST(deepseek41_attention_consumers_reuse_the_correct_current_token_source_and_candidate_mask) {
    const auto m=model();attn::Sequence seq(m,"sharing",24);int filtered=0,reselected=0;
    for(int pos=0;pos<18;++pos) {
        attn::Trace source2,source4;
        for(int layer=0;layer<6;++layer) {
            auto x=input(pos,layer);std::vector<float> out(4);attn::Trace tr;
            REQUIRE_EQ(seq.step(layer,pos,x,out,&tr),RAD_OK);
            if(layer==2)source2=tr;
            if(layer==3) {
                CHECK(tr.index_query.empty());CHECK(tr.scores.empty());CHECK(tr.index_key.empty());
                CHECK(tr.compressed_indices==source2.compressed_indices);
            }
            if(layer==4)source4=tr;
            if(layer==5) {
                CHECK(tr.compressed_latent.empty());CHECK(tr.index_key.empty());CHECK(!tr.index_query.empty());
                CHECK(tr.candidates==source4.candidates);
                if(std::count(tr.candidates.begin(),tr.candidates.end(),uint8_t(1))<(int)tr.candidates.size())++filtered;
                if(tr.compressed_indices!=source4.compressed_indices)++reselected;
                for(int index:tr.compressed_indices)if(index>=0)CHECK(tr.candidates[size_t(index)]);
                CHECK(tr.candidates.back()); // newest partially filled block is pinned
            }
        }
    }
    CHECK(filtered>0);CHECK(reselected>0);
}
TEST(deepseek41_attention_can_reproject_index_keys_at_a_distinct_finer_source) {
    auto c=config();c.index_k_source_layers={2,3,4};c.index_source_layers={2,3,4,5};auto w=weights();
    w[3].index_k=w[2].index_k;w[3].index_k_norm=w[2].index_k_norm;
    w[3].index_q=w[2].index_q;w[3].index_head_weights=w[2].index_head_weights;
    std::shared_ptr<const attn::Model> m;REQUIRE_EQ(attn::Model::create(c,w,"finer-keys",&m),RAD_OK);
    attn::Sequence seq(m,"finer",24);
    for(int pos=0;pos<4;++pos) {
        attn::Trace source;
        for(int layer=0;layer<6;++layer) {
            auto x=input(pos,layer);std::vector<float> out(4);attn::Trace tr;
            REQUIRE_EQ(seq.step(layer,pos,x,out,&tr),RAD_OK);
            if(layer==2)source=tr;
            if(layer==3)CHECK(tr.index_key==source.index_key);
        }
    }
}
TEST(deepseek41_attention_prefix_restore_preserves_pending_compression_window_wrap_and_shared_state) {
    const auto m=model();attn::Sequence seq(m,"prefix",24);attn::Sequence::Snapshot saved;
    REQUIRE_EQ(token(&seq,0),RAD_OK);REQUIRE_EQ(seq.snapshot(&saved),RAD_OK);
    REQUIRE(saved.compressors[2]);CHECK_EQ(saved.compressors[2]->pending,1u);
    std::vector<std::vector<std::vector<float>>> expected;
    for(int pos=1;pos<12;++pos) {expected.emplace_back();REQUIRE_EQ(token(&seq,pos,&expected.back()),RAD_OK);}
    REQUIRE_EQ(seq.restore(saved),RAD_OK);
    for(int pos=1;pos<12;++pos) {
        std::vector<std::vector<float>> got;REQUIRE_EQ(token(&seq,pos,&got),RAD_OK);
        CHECK(got==expected[size_t(pos-1)]);
    }
    REQUIRE_EQ(seq.snapshot(&saved),RAD_OK);CHECK_EQ(saved.position,12);
    for(const auto& window:saved.windows)CHECK_EQ(window.next_position,12);
    CHECK_EQ(saved.compressed_kv[2].size(),6u*4);CHECK_EQ(saved.compressed_kv[4].size(),12u*4);
    REQUIRE_EQ(token(&seq,12),RAD_OK);REQUIRE_EQ(seq.restore(saved),RAD_OK);
    std::vector<std::vector<float>> replay;REQUIRE_EQ(token(&seq,12,&replay),RAD_OK);
    for(int l=0;l<6;++l)compare(replay[size_t(l)],fixture()["traces"][12][l]["output"]);
}
TEST(deepseek41_attention_restore_rejects_other_owners_and_malformed_snapshots_transactionally) {
    const auto m=model();attn::Sequence seq(m,"prefix",24);REQUIRE_EQ(token(&seq,0),RAD_OK);
    attn::Sequence::Snapshot saved;REQUIRE_EQ(seq.snapshot(&saved),RAD_OK);
    attn::Sequence other(m,"other",24);CHECK(other.restore(saved)<0);CHECK_EQ(other.next_position(),0);
    std::shared_ptr<const attn::Model> replacement;
    REQUIRE_EQ(attn::Model::create(config(),weights(),m->identity(),&replacement),RAD_OK);
    attn::Sequence new_model(replacement,"prefix",24);CHECK(new_model.restore(saved)<0);
    for(int which=0;which<4;++which) {
        auto bad=saved;
        if(which==0)bad.windows[0].next_position=2;
        if(which==1)bad.compressors[2]->pending=0;
        if(which==2)bad.compressed_kv[4].clear();
        if(which==3)bad.index_keys[4][0]=NAN;
        CHECK(seq.restore(bad)<0);CHECK_EQ(seq.next_position(),1);CHECK_EQ(seq.next_layer(),0);
    }
    REQUIRE_EQ(token(&seq,1),RAD_OK);
}
TEST(deepseek41_attention_failed_steps_do_not_advance_state_or_overwrite_outputs) {
    const auto m=model();attn::Sequence seq(m,"retry",24);auto x=input(0,0);std::vector<float> out(4,99);
    CHECK(seq.step(1,0,x,out)<0);CHECK_EQ(seq.next_layer(),0);CHECK(out==std::vector<float>(4,99));
    x[0]=NAN;CHECK(seq.step(0,0,x,out)<0);CHECK_EQ(seq.next_position(),0);CHECK(out==std::vector<float>(4,99));
    x=input(0,0);attn::Sequence::Snapshot initial;REQUIRE_EQ(seq.snapshot(&initial),RAD_OK);
    REQUIRE_EQ(seq.step(0,0,x,out),RAD_OK);CHECK_EQ(seq.next_layer(),1);
    attn::Sequence::Snapshot unavailable;CHECK(seq.snapshot(&unavailable)<0);
    REQUIRE_EQ(seq.restore(initial),RAD_OK);CHECK_EQ(seq.next_layer(),0);REQUIRE_EQ(token(&seq,0),RAD_OK);
}
TEST(deepseek41_attention_interleaved_sequences_have_no_global_cache_or_selection_state) {
    const auto m=model();attn::Sequence isolated(m,"isolated",24);std::vector<std::vector<std::vector<float>>> baseline;
    for(int pos=0;pos<10;++pos) {baseline.emplace_back();REQUIRE_EQ(token(&isolated,pos,&baseline.back(),-.7f),RAD_OK);}
    attn::Sequence a(m,"a",24),b(m,"b",24);
    for(int pos=0;pos<10;++pos)for(int layer=0;layer<6;++layer) {
        auto xa=input(pos,layer),xb=input(pos,layer,-.7f);std::vector<float> ya(4),yb(4);
        REQUIRE_EQ(a.step(layer,pos,xa,ya),RAD_OK);REQUIRE_EQ(b.step(layer,pos,xb,yb),RAD_OK);
        compare(ya,fixture()["traces"][pos][layer]["output"]);CHECK(yb==baseline[size_t(pos)][size_t(layer)]);
    }
}
TEST(deepseek41_attention_model_validates_source_weight_ownership_and_context_bounds) {
    const auto original=model();
    for(int which=0;which<5;++which) {
        auto w=weights();
        if(which==0)w[2].index_k_norm.clear();
        if(which==1)w[3].compress_kv=w[2].compress_kv;
        if(which==2)w[0].q_norm.push_back(1);
        if(which==3)w[5].index_q[0]=NAN;
        if(which==4)w[0].sinks[0]=INFINITY;
        std::shared_ptr<const attn::Model> unchanged=original;
        CHECK(attn::Model::create(config(),std::move(w),"bad",&unchanged)<0);CHECK(unchanged==original);
    }
    attn::Sequence seq(original,"limit",5);
    for(int pos=0;pos<5;++pos)REQUIRE_EQ(token(&seq,pos),RAD_OK);
    auto x=input(5,0);std::vector<float> out(4,11);CHECK(seq.step(0,5,x,out)<0);
    CHECK_EQ(seq.next_position(),5);CHECK(out==std::vector<float>(4,11));
}
RAD_TEST_MAIN()
