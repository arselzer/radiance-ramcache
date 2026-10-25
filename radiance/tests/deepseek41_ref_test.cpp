/* FP32 semantic oracles for DeepSeek1017, with independent NumPy golden data.
 * Production activation/cache quantization and GPU arithmetic are separate gates. */
#include "rad_test.h"
#include "../arch/common/rad_ds41_ref.h"
#include <nlohmann/json.hpp>
#include <fstream>
using namespace rad::deepseek41::ref;
using Json = nlohmann::json;
namespace {
const Json& fixture() {
    static const auto data=[] { std::ifstream file(std::string(RAD_GOLDEN_FIXTURES)+"/deepseek41-reference.json");
        if(!file)throw std::runtime_error("missing DS41 reference fixture");
        return Json::parse(file); }();
    return data;
}
void flatten(const Json& x,std::vector<float>& out) {
    if(x.is_array())for(const auto& v:x)flatten(v,out);else out.push_back(x.get<float>());
}
std::vector<float> floats(const Json& x) {std::vector<float> out;flatten(x,out);return out;}
void near(Floats a,Floats b,float absolute=3e-6f,float relative=2e-5f) {
    REQUIRE_EQ(a.size(),b.size());
    for(size_t i=0;i<a.size();++i)CHECK(std::fabs(a[i]-b[i])<=absolute+relative*std::fabs(b[i]));
}
MhcMix prepare(const Json& f,Floats input={}) {
    MhcMix mix;const auto h=input.empty()?floats(f["h"]):std::vector<float>(input.begin(),input.end());
    std::string why;const int status=mhc_prepare(h,6,4,floats(f["fn"]),floats(f["scale"]),floats(f["base"]),20,1e-6f,1e-6f,&mix,&why);
    if(status!=RAD_OK)throw std::runtime_error(why);
    return mix;
}
}
TEST(deepseek_shifted_mhc_coefficients_match_independent_reference) {
    for(const auto& f:fixture()["mhc"]) {
        const auto mix=prepare(f);near(mix.pre,floats(f["pre"]));near(mix.post,floats(f["post"]));near(mix.comb,floats(f["comb"]));
        std::vector<float> x(6);CHECK_OK(mhc_collapse(floats(f["h"]),floats(f["previous_pre"]),x));near(x,floats(f["collapsed"]));
    }
}
TEST(deepseek_shifted_sublayers_and_final_head_use_carried_pre_mix) {
    auto h=floats(fixture()["mhc"][0]["h"]);std::vector<float> previous{1,0,0,0};
    for(const auto& f:fixture()["mhc"]) {
        auto mix=prepare(f,h);std::vector<float> out(h.size());
        auto sublayer=[&](Floats x,Mutable y) {
            std::vector<float> normed(6);int status=rms_norm(x,floats(f["norm"]),1e-6f,normed);
            if(status==RAD_OK)status=linear(normed,floats(f["subweight"]),6,y);
            for(auto& v:y)v=std::tanh(v);
            return status;
        };
        CHECK_OK(shifted_sublayer(h,previous,mix,sublayer,out));near(out,floats(f["output"]));
        std::vector<float> wrong(h.size());CHECK_OK(shifted_sublayer(h,mix.pre,mix,sublayer,wrong));
        CHECK(!std::equal(out.begin(),out.end(),wrong.begin(),[](float a,float b){return std::fabs(a-b)<1e-3f;}));
        h=out;previous=mix.pre;
    }
    std::vector<float> head(6);CHECK_OK(mhc_collapse(h,previous,head));near(head,floats(fixture()["final_head_collapse"]));
}
TEST(deepseek_mhc_comb_is_source_destination_and_allows_in_place_residual) {
    MhcMix mix{2,{1,0},{0,0},{.1f,.2f,.7f,.8f}};std::vector<float> h{1,10},y{0},out(2);
    CHECK_OK(mhc_expand(y,h,mix,out));near(out,std::vector<float>{7.1f,8.2f});
    CHECK_OK(mhc_expand(y,h,mix,h));near(h,out);
}
TEST(deepseek_routing_uses_correction_for_ids_and_unbiased_weights) {
    for(const auto& f:fixture()["routing"]) {
        Routing route_out;CHECK_OK(route(floats(f["logits"]),floats(f["bias"]),f["topk"],f["temperature"],f["route_scale"],true,&route_out));
        CHECK(route_out.ids==f["ids"].get<std::vector<int32_t>>());near(route_out.weights,floats(f["weights"]));
        if(f["topk"]==6)CHECK_NEAR(std::accumulate(route_out.weights.begin(),route_out.weights.end(),0.f),1.5f,2e-6);
    }
    Routing one;CHECK_OK(route(std::vector<float>{0,-10},std::vector<float>{0,20},1,1,1.5f,true,&one));
    CHECK_EQ(one.ids[0],1);CHECK_NEAR(one.weights[0],1.5*std::sqrt(std::log1p(std::exp(-10.f))),1e-7);
    CHECK(route(std::vector<float>{0},std::vector<float>{0},1,0,1,true,&one)<0);
    CHECK_OK(route(std::vector<float>{-1000,1000},std::vector<float>{0,0},2,1,1,true,&one));
    CHECK(finite(one.weights));
}
TEST(deepseek_swiglu_clamps_gate_only_above_and_up_on_both_sides) {
    for(const auto& f:fixture()["swiglu"]) {
        std::vector<float> out(f["gate"].size());CHECK_OK(swiglu(floats(f["gate"]),floats(f["up"]),f["limit"],out));near(out,floats(f["out"]));
    }
    std::vector<float> out(1);CHECK_OK(swiglu(std::vector<float>{-20},std::vector<float>{1},10,out));
    CHECK_NEAR(out[0],-20.f*sigmoid(-20),1e-11);
}
TEST(deepseek_rotary_yarn_and_inverse_output_rotation_match_reference) {
    for(const auto& f:fixture()["rope"]) {
        const auto input=floats(f["x"]);auto forward=input,inverse=input;
        Rope cfg{f["dim"],f["theta"],f["factor"],f["fast"],f["slow"],f["original"]};
        CHECK_OK(rotary(forward,f["position"],cfg));CHECK_OK(rotary(inverse,f["position"],cfg,true));
        near(forward,floats(f["forward"]),8e-4f,4e-5f);near(inverse,floats(f["inverse"]),8e-4f,4e-5f);
        CHECK_OK(rotary(forward,f["position"],cfg,true));near(forward,input);
        for(size_t i=0;i<input.size()-cfg.rotary_dim;++i)CHECK_EQ(inverse[i],input[i]);
    }
    std::vector<float> x{1,2,3};const auto before=x;CHECK(rotary(x,0,Rope{3})<0);CHECK(x==before);
}
TEST(deepseek_sparse_attention_includes_sink_and_padding_conventions) {
    for(const auto& f:fixture()["attention"]) {
        auto q=floats(f["q"]),kv=floats(f["kv"]),sinks=floats(f["sinks"]);std::vector<float> out(q.size());
        CHECK_OK(sparse_attention(q,kv,3,8,sinks,f["ids"].get<std::vector<int32_t>>(),f["scale"],out));near(out,floats(f["out"]));
    }
    const std::vector<float> q{0},kv{2,4};std::vector<float> out(1);
    CHECK_OK(sparse_attention(q,kv,1,1,std::vector<float>{0},std::vector<int32_t>{0,1},1,out));CHECK_NEAR(out[0],2.f,1e-6);
    CHECK_OK(sparse_attention(q,kv,1,1,std::vector<float>{std::log(2.f)},std::vector<int32_t>{0,1},1,out));CHECK_NEAR(out[0],1.5f,1e-6);
    out[0]=7;CHECK(sparse_attention(q,kv,1,1,std::vector<float>{0},std::vector<int32_t>{2},1,out)<0);CHECK_EQ(out[0],7);
    CHECK_OK(sparse_attention(q,{},1,1,std::vector<float>{-INFINITY},{},1,out));CHECK_EQ(out[0],0);
    CHECK_OK(sparse_attention(q,std::vector<float>{2,NAN},1,1,std::vector<float>{0},std::vector<int32_t>{0},1,out));
    CHECK_EQ(out[0],1); // unreachable cache storage never participates in attention
}
TEST(deepseek_index_scores_use_global_head_scaling) {
    const auto& f=fixture()["index"];const auto q=floats(f["q"]),keys=floats(f["keys"]),w=floats(f["weights"]);
    std::vector<float> scores(11),first(11),rest(11);CHECK_OK(index_scores(q,keys,w,4,6,scores));near(scores,floats(f["scores"]));
    CHECK_OK(index_scores(Floats(q).first(4),keys,Floats(w).first(1),4,6,first));
    CHECK_OK(index_scores(Floats(q).subspan(4),keys,Floats(w).subspan(1),4,6,rest));
    for(size_t i=0;i<scores.size();++i)CHECK_NEAR(scores[i],first[i]+rest[i],1e-6);
}
TEST(deepseek_candidate_blocks_pin_newest_partial_block_and_keep_causal_visibility_separate) {
    const auto scores=floats(fixture()["index"]["scores"]);
    for(const auto& f:fixture()["candidate"]) {
        std::vector<uint8_t> mask(11);CHECK_OK(candidate_blocks(scores,f["visible"],f["block_size"],f["top_blocks"],mask));
        CHECK(mask==f["keep"].get<std::vector<uint8_t>>());std::vector<int32_t> ids;
        CHECK_OK(index_topk(scores,f["visible"],4,mask,&ids));CHECK(ids==f["ids"].get<std::vector<int32_t>>());
        for(auto id:ids)CHECK(id==-1 || size_t(id)<f["visible"].get<size_t>());
    }
    std::vector<uint8_t> keep(8);CHECK_OK(candidate_blocks(std::vector<float>{9,8,7,6,5,4,-100,-99},7,3,1,keep));
    CHECK(keep==std::vector<uint8_t>({0,0,0,0,0,0,1,1}));
}
TEST(deepseek_compressor_matches_independent_pooling_and_preserves_partial_prefix_groups) {
    for(const auto& f:fixture()["compression"]) {
        const size_t ratio=f["ratio"];const auto kv=floats(f["kv"]),scores=floats(f["scores"]),norm=floats(f["norm"]);
        Compressor state(ratio,norm,1e-6f,"asset/source"),replay(ratio,norm,1e-6f,"asset/source");
        const auto expected=floats(f["latents"]);std::vector<float> all,latent;int64_t position=-1;
        for(size_t token=0;token<7;++token) {
            CHECK_OK(state.push(Floats(kv).subspan(token*5,5),ratio==2?Floats(scores).subspan(token*5,5):Floats{},&latent,&position));
            if(!latent.empty()){CHECK_EQ(position,int64_t(token+1-ratio));all.insert(all.end(),latent.begin(),latent.end());}
            if(token==2)CHECK_OK(replay.restore(state.snapshot()));
            if(token>2) {
                std::vector<float> again;int64_t repeated=-1;
                CHECK_OK(replay.push(Floats(kv).subspan(token*5,5),ratio==2?Floats(scores).subspan(token*5,5):Floats{},&again,&repeated));
                CHECK(again==latent);CHECK_EQ(repeated,position);
            }
        }
        near(all,expected);CHECK_EQ(state.next_position(),7);
        auto broken=state.snapshot();broken.pending=ratio;CHECK(state.restore(broken)<0);CHECK_EQ(state.next_position(),7);
        Compressor other(ratio,norm,1e-6f,"other/source");CHECK(other.restore(state.snapshot())<0);
    }
}
TEST(deepseek_window_ring_handles_wrap_restore_and_alternating_sequences) {
    WindowCache state(3,2,"asset/seq1/layer0"),other(3,2,"asset/seq2/layer0");
    for(int token=0;token<5;++token)CHECK_OK(state.append(token,std::vector<float>{float(token),-float(token)}));
    CHECK(state.selected()==std::vector<int32_t>({2,0,1}));const auto prefix=state.snapshot();
    CHECK_OK(state.append(5,std::vector<float>{5,-5}));const auto expected=state.snapshot();
    CHECK_OK(state.restore(prefix));CHECK_OK(state.append(5,std::vector<float>{5,-5}));CHECK(state.snapshot().slots==expected.slots);
    CHECK(other.restore(prefix)<0);CHECK_EQ(other.next_position(),0);CHECK(state.append(7,std::vector<float>{7,-7})<0);CHECK_EQ(state.next_position(),6);
}
TEST(deepseek_window_attention_prefill_domain_matches_ring_decode_domain) {
    WindowCache ring(4,2,"asset/seq/layer");std::vector<float> full;
    for(int64_t token=0;token<11;++token) {
        const std::vector<float> row{std::sin(float(token)),std::cos(float(token))};full.insert(full.end(),row.begin(),row.end());CHECK_OK(ring.append(token,row));
        std::vector<int32_t> logical;for(int64_t p=std::max<int64_t>(0,token-3);p<=token;++p)logical.push_back(int32_t(p));
        std::vector<float> q{.25f,-.5f},sinks{.2f},a(2),b(2);
        CHECK_OK(sparse_attention(q,full,1,2,sinks,logical,.7f,a));CHECK_OK(sparse_attention(q,ring.values(),1,2,sinks,ring.selected(),.7f,b));near(a,b);
    }
}
TEST(deepseek_output_projection_is_grouped_before_the_final_dense_projection) {
    const auto& f=fixture()["grouped_output"];std::vector<float> out(5);
    CHECK_OK(grouped_output(floats(f["heads"]),4,6,2,3,floats(f["wa"]),floats(f["wb"]),out));near(out,floats(f["out"]));
}
TEST(deepseek_reference_rejects_bad_shapes_without_mutating_state) {
    std::vector<float> out(2,13);CHECK(linear(std::vector<float>{1,2},std::vector<float>{1},2,out)<0);CHECK_EQ(out[0],13);
    CHECK(rms_norm(std::vector<float>{1,2},std::vector<float>{1,1},0,out)<0);
    Compressor state(2,std::vector<float>{1,1},1e-6f,"asset");std::vector<float> latent;int64_t first;
    CHECK(state.push(std::vector<float>{1},std::vector<float>{1,2},&latent,&first)<0);CHECK_EQ(state.next_position(),0);
    std::vector<int32_t> ids;CHECK(index_topk(std::vector<float>{1},2,1,{},&ids)<0);
}
TEST(deepseek_bf16_rounding_preserves_ties_and_explicit_boundaries) {
    CHECK_EQ(rounded(1.f+std::ldexp(1.f,-8),Precision::bf16),1.f);
    CHECK_EQ(rounded(1.f+3*std::ldexp(1.f,-8),Precision::bf16),1.f+std::ldexp(1.f,-6));
    CHECK(std::isnan(rounded(NAN,Precision::bf16)));CHECK_EQ(rounded(INFINITY,Precision::bf16),INFINITY);
}
RAD_TEST_MAIN()
