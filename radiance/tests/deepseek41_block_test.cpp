/* Complete tiny serial block and head fixtures. Attention is a supplied linear test callback;
 * full attention/cache math has separate tests. No serving/GPU/model-size path is implied. */
#include "rad_test.h"
#include "../arch/deepseek41/block_ref.h"
#include <nlohmann/json.hpp>
#include <fstream>
namespace r=rad::deepseek41::ref;
namespace eg=rad::deepseek41::engram;
namespace {
using Json=nlohmann::json;
const Json& fixture(){static const auto data=[] {std::ifstream f(std::string(RAD_GOLDEN_FIXTURES)+"/deepseek41-block.json");if(!f)throw std::runtime_error("missing DeepSeek block fixture");return Json::parse(f);}();return data;}
void append(const Json& j,std::vector<float>& out){if(j.is_array())for(const auto& e:j)append(e,out);else out.push_back(j.get<float>());}
std::vector<float> values(const Json& j){std::vector<float> out;append(j,out);return out;}
r::ExpertWeights expert(const Json& j){return {values(j.at("gate")),values(j.at("up")),values(j.at("down"))};}
r::MhcWeights hc(const Json& j){return {values(j.at("fn")),values(j.at("scale")),values(j.at("base"))};}
r::BlockWeights block(const Json& j){
    r::BlockWeights w;w.layer_id=j.at("layer_id");w.dim=j.at("dim");w.copies=j.at("copies");w.sinkhorn_iterations=j.at("sinkhorn_iterations");
    w.norm_eps=j.at("norm_eps");w.hc_eps=j.at("hc_eps");w.attention_hc=hc(j.at("attention_hc"));w.ffn_hc=hc(j.at("ffn_hc"));
    w.attention_norm=values(j.at("attention_norm"));w.ffn_norm=values(j.at("ffn_norm"));const auto m=j.at("moe");
    auto& e=w.moe;e.dim=m.at("dim");e.intermediate=m.at("intermediate");e.topk=m.at("topk");e.temperature=m.at("temperature");
    e.route_scale=m.at("route_scale");e.routed_clamp=m.at("routed_clamp");e.shared_clamp=m.at("shared_clamp");e.normalize=m.at("normalize");
    e.router=values(m.at("router"));e.correction=values(m.at("correction"));for(const auto& a:m.at("experts"))e.experts.push_back(expert(a));e.shared=expert(m.at("shared"));
    if(j.contains("engram")){const auto g=j.at("engram");w.engram=r::EngramWeights{g.at("control_layer"),values(g.at("projection")),values(g.at("q_weight")),values(g.at("k_weight"))};}
    return w;
}
r::HeadWeights head(const Json& j){return {j.at("dim"),j.at("vocab"),j.at("norm_eps"),values(j.at("norm")),values(j.at("projection"))};}
r::EngramToken engram_token(){
    const auto j=fixture();const auto a=j.at("engram_controls");eg::HashControls c;
    c.asset_identity=a.at("asset_identity");c.compressed_vocab_size=a.at("compressed_vocab_size");c.raw_pad_id=a.at("raw_pad_id");
    c.max_ngram_size=a.at("max_ngram_size");c.n_heads=a.at("n_heads");c.head_dim=a.at("head_dim");c.token_map=a.at("token_map").get<std::vector<int32_t>>();
    for(const auto& l:a.at("layers"))c.layers.push_back({l.at("layer_id"),l.at("table_rows"),l.at("primes").get<std::vector<int64_t>>(),l.at("offsets").get<std::vector<int64_t>>(),l.at("multipliers").get<std::vector<int64_t>>()});
    r::EngramToken token;std::string why;if(eg::ModelControls::create(std::move(c),&token.model,&why)!=RAD_OK)throw std::runtime_error(why);
    static const std::vector<int64_t> ids=fixture().at("engram_hashes").get<std::vector<int64_t>>();token.hashes=ids;
    token.lookup=[](int layer,std::span<const int64_t> rows,int64_t width,std::span<float> out,std::string*) {
        CHECK_EQ(layer,1);CHECK_EQ(width,2);CHECK_EQ(rows.size(),2u);
        for(size_t i=0;i<rows.size();++i)for(int64_t d=0;d<width;++d)out[i*size_t(width)+size_t(d)]=float(std::sin(double(rows[i]*width+d)*.37)*.2);
        return RAD_OK;
    };
    return token;
}
void near(r::Floats a,r::Floats b,float eps=8e-6f){REQUIRE_EQ(a.size(),b.size());for(size_t i=0;i<a.size();++i)CHECK(std::abs(a[i]-b[i])<=eps+2e-5f*std::abs(b[i]));}
r::AttentionCallback attention(const Json& j,r::Precision precision,std::vector<int>* calls=nullptr){
    std::vector<std::vector<float>> matrices;for(const auto& layer:j.at("weights"))matrices.push_back(values(layer.at("attention_matrix")));
    return [matrices=std::move(matrices),precision,calls](int layer,int64_t position,r::Floats x,r::Mutable out,std::string* why){
        if(calls)calls->push_back(layer);
        CHECK_EQ(position,0);
        return r::linear(x,matrices.at(size_t(layer)),out.size(),out,precision,why);
    };
}
}
TEST(deepseek41_complete_block_matches_official_shifted_math_with_engram_first){
    REQUIRE_EQ(fixture().at("cases").size(),2u);
    for(const auto& f:fixture().at("cases")){
        const auto p=f.at("precision")=="bf16"?r::Precision::bf16:r::Precision::f32;const auto w=block(f.at("weights").at(1));
        const auto e=engram_token();const auto expected=f.at("standalone");r::SerialState input{values(expected.at("h")),values(expected.at("pre"))},out;
        r::BlockTrace trace;std::string why;REQUIRE_EQ(r::block_step(w,0,input,attention(f,p),&out,&e,p,&trace,&why),RAD_OK);
        near(trace.after_engram,values(expected.at("after_engram")));near(out.h,values(expected.at("output")));near(out.previous_pre,values(expected.at("next_pre")));
        CHECK(trace.moe.routing.ids==expected.at("route_ids").get<std::vector<int32_t>>());near(trace.moe.routing.weights,values(expected.at("route_weights")));
        std::vector<float> correct(4),wrong(4);
        REQUIRE_EQ(r::mhc_collapse(trace.after_engram,input.previous_pre,correct,p),RAD_OK);
        REQUIRE_EQ(r::mhc_collapse(trace.after_engram,trace.attention_mix.pre,wrong,p),RAD_OK);
        near(trace.attention_input,correct);CHECK(!std::equal(correct.begin(),correct.end(),wrong.begin()));
    }
}
TEST(deepseek41_two_blocks_and_final_head_use_carried_last_pre_mix){
    REQUIRE_EQ(fixture().at("cases").size(),2u);
    for(const auto& f:fixture().at("cases")){
        const auto p=f.at("precision")=="bf16"?r::Precision::bf16:r::Precision::f32;std::vector<r::BlockWeights> weights;
        for(const auto& j:f.at("weights"))weights.push_back(block(j));
        const auto hw=head(f.at("head"));const auto e=engram_token();
        std::vector<float> logits(hw.vocab);r::SerialTrace trace;std::vector<int> calls;
        REQUIRE_EQ(r::serial_token(values(f.at("embedding")),weights,0,hw,attention(f,p,&calls),logits,&e,p,&trace),RAD_OK);
        CHECK(calls==std::vector<int>({0,1}));near(logits,values(f.at("logits")));CHECK_EQ(trace.layers.size(),2u);
        r::SerialState state;REQUIRE_EQ(r::init_token(values(f.at("embedding")),4,&state,p),RAD_OK);
        for(size_t i=0;i<weights.size();++i){
            REQUIRE_EQ(r::block_step(weights[i],0,state,attention(f,p),&state,&e,p),RAD_OK);
            near(state.h,values(f.at("stages").at(i).at("h")));near(state.previous_pre,values(f.at("stages").at(i).at("pre")));
        }
        std::vector<float> collapsed(4),normalized(4);REQUIRE_EQ(r::mhc_collapse(state.h,state.previous_pre,collapsed,p),RAD_OK);
        REQUIRE_EQ(r::rms_norm(collapsed,hw.norm,hw.norm_eps,normalized,p),RAD_OK);
        near(collapsed,values(f.at("head_collapsed")));near(normalized,values(f.at("head_normalized")));
        r::SerialState wrong=state;wrong.previous_pre={1,0,0,0};std::vector<float> wrong_logits(hw.vocab);
        REQUIRE_EQ(r::final_head(wrong,hw,wrong_logits,p),RAD_OK);CHECK(!std::equal(logits.begin(),logits.end(),wrong_logits.begin()));
    }
}
TEST(deepseek41_expert_weights_activation_before_down_projection_rounding){
    const r::ExpertWeights w{{1},{1},{1.25f}};const std::vector<float> x{1};std::vector<float> y(1);r::ExpertTrace trace;
    REQUIRE_EQ(r::expert_forward(x,w,1,0,.1f,y,r::Precision::bf16,&trace),RAD_OK);
    CHECK_EQ(trace.down_input[0],.0732421875f);CHECK_EQ(y[0],.091796875f);
    // Weighting the already projected result takes a different BF16 rounding path.
    std::vector<float> unweighted(1);REQUIRE_EQ(r::expert_forward(x,w,1,0,1,unweighted,r::Precision::bf16),RAD_OK);
    CHECK(y[0]!=r::rounded(unweighted[0]*.1f,r::Precision::bf16));
}
TEST(deepseek41_moe_can_route_every_expert_and_always_adds_shared_expert){
    auto w=block(fixture().at("cases").at(0).at("weights").at(0)).moe;w.topk=w.experts.size();
    const std::vector<float> x{.6f,-.8f,.9f,-.2f};std::vector<float> y(4);r::MoeTrace trace;
    REQUIRE_EQ(r::moe_forward(x,w,y,r::Precision::f32,&trace),RAD_OK);
    CHECK_EQ(trace.routing.ids.size(),3u);CHECK_EQ(trace.experts.size(),3u);
    float shared_size=0;
    for(size_t d=0;d<4;++d){float expected=0;for(size_t i=0;i<3;++i){REQUIRE_EQ(trace.experts[i].output.size(),4u);expected+=trace.experts[i].output[d];}
        expected+=trace.shared.output[d];CHECK_EQ(y[d],expected);shared_size+=std::abs(trace.shared.output[d]);}
    CHECK(shared_size>0);
}
TEST(deepseek41_block_failure_keeps_local_output_and_missing_engram_fails_before_attention){
    const auto f=fixture().at("cases").at(0);auto w=block(f.at("weights").at(1));const auto data=f.at("standalone");
    r::SerialState input{values(data.at("h")),values(data.at("pre"))},out{{999},{888}};int calls=0;
    auto cb=[&](int,int64_t,r::Floats,r::Mutable,std::string*){++calls;return RAD_E_IO;};
    CHECK_EQ(r::block_step(w,0,input,cb,&out),RAD_E_STATE);CHECK_EQ(calls,0);CHECK_EQ(out.h[0],999);
    const auto e=engram_token();CHECK_EQ(r::block_step(w,0,input,cb,&out,&e),RAD_E_IO);CHECK_EQ(out.h[0],999);CHECK_EQ(out.previous_pre[0],888);
    w.moe.experts[0].down.pop_back();calls=0;CHECK(r::block_step(w,0,input,cb,&out,&e)<0);CHECK_EQ(calls,0);
}
RAD_TEST_MAIN()
