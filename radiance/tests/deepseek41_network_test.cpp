/* Integrated tiny serial reference. Independent component goldens live in the block,
 * attention and Engram tests; this checks callback ordering and joint state lifecycle.
 * Dense/logical weights and FP32 activations only. It is not a serving architecture. */
#include "rad_test.h"
#include "../arch/deepseek41/block_ref.h"
#include "../arch/deepseek41/attention_ref.h"
#include <nlohmann/json.hpp>
#include <fstream>
namespace r=rad::deepseek41::ref;
namespace eg=rad::deepseek41::engram;
namespace attn=rad::deepseek41::attention;
namespace ds=rad::deepseek41;
namespace ref=rad::deepseek41::ref;
namespace blockfix {
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
namespace attnfix {
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
}

namespace {
struct Network {
    attn::Sequence attention;
    eg::HashState history;
    std::vector<r::BlockWeights> blocks;
    r::HeadWeights head;
    r::EngramToken engram;
    struct Snapshot {attn::Sequence::Snapshot attention;eg::HistorySnapshot history;};
    explicit Network(const std::string& name)
        :attention(attnfix::model(),name,24),history(hash_model()) {
        const auto& data=blockfix::fixture().at("cases").at(0);
        head=blockfix::head(data.at("head"));
        for(int layer=0;layer<6;++layer) {
            auto weights=blockfix::block(data.at("weights").at(layer%2));
            weights.layer_id=layer;
            if(layer!=1)weights.engram.reset();
            blocks.push_back(std::move(weights));
        }
        engram=blockfix::engram_token();engram.model=hash_model();
    }
    static std::shared_ptr<const eg::ModelControls> hash_model() {
        static const auto model=blockfix::engram_token().model;return model;
    }
    int snapshot(Snapshot* out) const {
        const int status=attention.snapshot(&out->attention);
        if(status==RAD_OK)out->history=history.snapshot();
        return status;
    }
    int restore(const Snapshot& saved) {
        auto next_attention=attention;auto next_history=history;
        int status=next_attention.restore(saved.attention);
        if(status==RAD_OK)status=next_history.restore(saved.history);
        if(status==RAD_OK){attention=std::move(next_attention);history=std::move(next_history);}
        return status;
    }
    int token(int32_t raw,std::vector<float>* logits,bool fail_engram=false,std::vector<int>* called=nullptr) {
        Snapshot before;int status=snapshot(&before);if(status!=RAD_OK)return status;
        const int64_t position=attention.next_position();std::vector<int64_t> hashes;
        status=history.consume(std::span(&raw,1),{},&hashes);if(status!=RAD_OK)return status;
        auto context=engram;context.hashes=hashes;
        if(fail_engram)context.lookup=[](int,std::span<const int64_t>,int64_t,std::span<float>,std::string*){return RAD_E_IO;};
        auto embedding=blockfix::values(blockfix::fixture().at("cases").at(0).at("embedding"));
        for(size_t d=0;d<embedding.size();++d)embedding[d]+=float(raw)*(.07f+float(d)*.02f);
        r::AttentionCallback callback=[&](int layer,int64_t pos,r::Floats input,r::Mutable out,std::string* why) {
            if(called)called->push_back(layer);
            return attention.step(layer,pos,input,out,nullptr,why);
        };
        std::vector<float> staged(head.vocab);
        status=r::serial_token(embedding,blocks,position,head,callback,staged,&context);
        if(status!=RAD_OK){const int rollback=restore(before);if(rollback!=RAD_OK)return rollback;return status;}
        *logits=std::move(staged);return RAD_OK;
    }
};
const std::vector<int32_t> tokens{3,1,0,2,3,2,1,0,1,3,2,0};
std::vector<std::vector<float>> baseline(const std::string& id,const std::vector<int32_t>& input) {
    Network n(id);std::vector<std::vector<float>> out;
    for(auto raw:input){std::vector<float> row;if(n.token(raw,&row)!=RAD_OK)throw std::runtime_error("network baseline failed");out.push_back(std::move(row));}
    return out;
}
}
TEST(deepseek41_network_orders_all_attention_layers_and_consumes_hashes_once_per_token) {
    Network n("ordered");std::vector<float> logits;std::vector<int> layers;
    REQUIRE_EQ(n.token(3,&logits,false,&layers),RAD_OK);
    CHECK(layers==std::vector<int>({0,1,2,3,4,5}));CHECK_EQ(n.history.position(),1);
    CHECK_EQ(n.attention.next_position(),1);CHECK_EQ(n.attention.next_layer(),0);CHECK_EQ(logits.size(),5u);
}
TEST(deepseek41_network_prefix_replay_restores_hash_history_partial_compressor_and_ring_windows) {
    const auto expected=baseline("baseline",tokens);Network n("replay");std::vector<float> logits;
    for(size_t i=0;i<5;++i){REQUIRE_EQ(n.token(tokens[i],&logits),RAD_OK);CHECK(logits==expected[i]);}
    Network::Snapshot prefix;REQUIRE_EQ(n.snapshot(&prefix),RAD_OK);
    CHECK_EQ(prefix.history.position,5);CHECK_EQ(prefix.attention.position,5);
    // Five tokens wrap the three-row SWA window and leave the ratio-2 source half full.
    REQUIRE(prefix.attention.compressors.at(2).has_value());
    CHECK_EQ(prefix.attention.compressors.at(2)->pending,1u);
    for(size_t i=5;i<tokens.size();++i)REQUIRE_EQ(n.token(tokens[i],&logits),RAD_OK);
    REQUIRE_EQ(n.restore(prefix),RAD_OK);
    for(size_t i=5;i<tokens.size();++i){REQUIRE_EQ(n.token(tokens[i],&logits),RAD_OK);CHECK(logits==expected[i]);}
}
TEST(deepseek41_network_failed_engram_fetch_rolls_back_attention_and_hash_progress) {
    const auto expected=baseline("reference",tokens);Network n("cancel");std::vector<float> logits{999};
    for(size_t i=0;i<5;++i)REQUIRE_EQ(n.token(tokens[i],&logits),RAD_OK);
    logits={999};CHECK_EQ(n.token(tokens[5],&logits,true),RAD_E_IO);
    CHECK_EQ(n.attention.next_position(),5);CHECK_EQ(n.attention.next_layer(),0);CHECK_EQ(n.history.position(),5);CHECK(logits==std::vector<float>({999}));
    REQUIRE_EQ(n.token(tokens[5],&logits),RAD_OK);CHECK(logits==expected[5]);
}
TEST(deepseek41_network_interleaved_sequences_do_not_share_history_or_attention_state) {
    auto other=tokens;std::reverse(other.begin(),other.end());const auto expected_a=baseline("a-base",tokens),expected_b=baseline("b-base",other);
    Network a("a"),b("b");std::vector<float> logits;
    for(size_t i=0;i<tokens.size();++i){REQUIRE_EQ(a.token(tokens[i],&logits),RAD_OK);CHECK(logits==expected_a[i]);REQUIRE_EQ(b.token(other[i],&logits),RAD_OK);CHECK(logits==expected_b[i]);}
}
RAD_TEST_MAIN()
