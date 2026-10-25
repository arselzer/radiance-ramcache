/* Engram math/history against actual official forward bodies run on tiny NumPy CPU fixtures.
 * This validates operator semantics, not checkpoint quality, SSD performance or a serving graph. */
#include "rad_test.h"
#include "../arch/deepseek41/engram_ref.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <unordered_set>

namespace eg = rad::deepseek41::engram;
namespace {
using Json = nlohmann::json;
const Json& golden() {
    static const auto data=[] {
        std::ifstream file(std::string(RAD_GOLDEN_FIXTURES)+"/deepseek41-engram.json");
        if(!file)throw std::runtime_error("missing DeepSeek Engram golden fixture");
        return Json::parse(file);
    }();
    return data;
}
eg::HashControls raw_controls() {
    const auto j=golden().at("controls");eg::HashControls c;
    c.asset_identity=j.at("asset_identity");c.compressed_vocab_size=j.at("compressed_vocab_size");
    c.raw_pad_id=j.at("raw_pad_id");c.max_ngram_size=j.at("max_ngram_size");
    c.n_heads=j.at("n_heads");c.head_dim=j.at("head_dim");c.token_map=j.at("token_map").get<std::vector<int32_t>>();
    for(const auto& l:j.at("layers"))c.layers.push_back({l.at("layer_id"),l.at("table_rows"),
        l.at("primes").get<std::vector<int64_t>>(),l.at("offsets").get<std::vector<int64_t>>(),
        l.at("multipliers").get<std::vector<int64_t>>()});
    return c;
}
std::shared_ptr<const eg::ModelControls> controls() {
    std::shared_ptr<const eg::ModelControls> out;std::string why;
    if(eg::ModelControls::create(raw_controls(),&out,&why)!=RAD_OK)throw std::runtime_error(why);
    return out;
}
std::vector<int32_t> ids() {return golden().at("raw_ids").get<std::vector<int32_t>>();}
std::vector<uint8_t> mask() {return golden().at("token_mask").get<std::vector<uint8_t>>();}
std::vector<int64_t> hashes() {return golden().at("hashes").get<std::vector<int64_t>>();}
}

TEST(deepseek41_engram_hash_matches_official_order_and_compressed_token_map) {
    const auto model=controls();eg::HashState state(model);std::vector<int64_t> got;std::string why;
    REQUIRE_EQ(state.consume(ids(),mask(),&got,&why),RAD_OK);
    CHECK(got==hashes());CHECK_EQ(got[0],10); // (0*3 XOR pad(3)*7) % prime(11)
    CHECK_EQ(state.position(),12);CHECK_EQ(state.history_size(),3u);
    CHECK_EQ(model->get().token_map[1],model->get().token_map[3]);
    CHECK_EQ(model->columns(),6);CHECK_EQ(model->get().layers.size(),2u);
}
TEST(deepseek41_engram_prefill_chunk_and_single_token_decode_match_full_forward) {
    const auto model=controls();eg::HashState state(model);const auto input=ids();const auto alive=mask();
    const size_t cuts[]={0,1,4,6,7,8,9,10,11,12};std::vector<int64_t> all;
    for(size_t k=1;k<std::size(cuts);++k) {
        std::vector<int64_t> block;const size_t start=cuts[k-1],count=cuts[k]-start;
        REQUIRE_EQ(state.consume(std::span(input).subspan(start,count),std::span(alive).subspan(start,count),&block),RAD_OK);
        all.insert(all.end(),block.begin(),block.end());
        CHECK(state.history_size()<=3);
    }
    CHECK(all==hashes());
}
TEST(deepseek41_engram_dead_token_blocks_all_older_lookbacks) {
    const auto model=controls();eg::HashState dead(model),padding(model);
    std::vector<int64_t> with_dead,with_pad;
    const std::vector<int32_t> a{5},b{2};const std::vector<uint8_t> no{0};
    REQUIRE_EQ(dead.consume(a,no,&with_dead),RAD_OK);
    REQUIRE_EQ(padding.consume(b,{},&with_pad),RAD_OK);
    CHECK(with_dead==with_pad);CHECK_EQ(dead.snapshot().history[0],eg::HashState::DEAD);
    const std::vector<int32_t> next{6};std::vector<int64_t> after_dead;
    REQUIRE_EQ(dead.consume(next,{},&after_dead),RAD_OK);
    // Newest=4; prior DEAD and every older position become compressed pad=3.
    CHECK_EQ(after_dead[0],((int64_t(4)*3)^(int64_t(3)*7))%11);
    CHECK_EQ(after_dead[2],(((int64_t(4)*3)^(int64_t(3)*7))^(int64_t(3)*11))%17+24);
    CHECK_EQ(after_dead[4],((((int64_t(4)*3)^(int64_t(3)*7))^(int64_t(3)*11))^(int64_t(3)*13))%23+60);
}
TEST(deepseek41_engram_prefix_restore_preserves_dead_history_and_rejected_suffixes) {
    const auto model=controls();eg::HashState state(model),restored(model);const auto input=ids();const auto alive=mask();
    std::vector<int64_t> ignored;REQUIRE_EQ(state.consume(std::span(input).first(5),std::span(alive).first(5),&ignored),RAD_OK);
    const auto snapshot=state.snapshot();CHECK_EQ(snapshot.history[1],eg::HashState::DEAD);
    REQUIRE_EQ(restored.restore(snapshot),RAD_OK);
    const std::vector<int32_t> rejected{9,9,9};REQUIRE_EQ(state.consume(rejected,{},&ignored),RAD_OK);
    REQUIRE_EQ(state.restore(snapshot),RAD_OK);
    std::vector<int64_t> left,right;
    REQUIRE_EQ(state.consume(std::span(input).subspan(5),std::span(alive).subspan(5),&left),RAD_OK);
    REQUIRE_EQ(restored.consume(std::span(input).subspan(5),std::span(alive).subspan(5),&right),RAD_OK);
    CHECK(left==right);const auto expected=hashes();CHECK(std::equal(left.begin(),left.end(),expected.begin()+5*12));
    state.reset();CHECK_EQ(state.position(),0);CHECK_EQ(state.history_size(),0u);
}
TEST(deepseek41_engram_snapshot_rejects_wrong_model_and_incomplete_state) {
    const auto model=controls();eg::HashState first(model),other(controls());std::vector<int64_t> output;
    REQUIRE_EQ(first.consume(ids(),mask(),&output),RAD_OK);const auto snapshot=first.snapshot();
    CHECK_EQ(other.restore(snapshot),RAD_E_STATE);CHECK_EQ(other.position(),0);
    eg::HashState same(model);auto broken=snapshot;broken.history.pop_back();
    CHECK_EQ(same.restore(broken),RAD_E_SHAPE);CHECK_EQ(same.position(),0);
    broken=snapshot;broken.history[0]=99;CHECK_EQ(same.restore(broken),RAD_E_FORMAT);
    REQUIRE_EQ(same.restore(snapshot),RAD_OK);CHECK_EQ(same.position(),12);
}
TEST(deepseek41_engram_bad_input_is_atomic_and_history_stays_bounded) {
    eg::HashState state(controls());std::vector<int64_t> out{888};const std::vector<int32_t> bad{-1},high{999};
    CHECK_EQ(state.consume(bad,{},&out),RAD_E_FORMAT);CHECK_EQ(state.consume(high,{},&out),RAD_E_FORMAT);
    const std::vector<uint8_t> wrong_mask{1,1};const std::vector<int32_t> good{1};
    CHECK_EQ(state.consume(good,wrong_mask,&out),RAD_E_SHAPE);CHECK_EQ(state.position(),0);CHECK(out==std::vector<int64_t>{888});
    std::vector<int32_t> many(4096,1);REQUIRE_EQ(state.consume(many,{},&out),RAD_OK);
    CHECK_EQ(state.position(),4096);CHECK_EQ(state.history_size(),3u);
    const auto before=state.snapshot();CHECK_EQ(state.consume(bad,{},&out),RAD_E_FORMAT);
    CHECK_EQ(state.position(),before.position);CHECK(state.snapshot().history==before.history);
}
TEST(deepseek41_engram_controls_refuse_malformed_shapes_and_overflow) {
    const auto good=raw_controls();std::shared_ptr<const eg::ModelControls> out;
    auto invalid=[&](eg::HashControls c){return eg::ModelControls::create(std::move(c),&out)<0;};
    auto c=good;c.token_map[0]=-1;CHECK(invalid(c));
    c=good;c.compressed_vocab_size=7;CHECK(invalid(c));
    c=good;c.token_map[0]=6;CHECK(invalid(c));
    c=good;c.layers[0].offsets[1]--;CHECK(invalid(c));
    c=good;c.layers[0].table_rows++;CHECK(invalid(c));
    c=good;c.layers[0].multipliers[0]=INT64_MAX;CHECK(invalid(c));
    c=good;c.layers[0].multipliers[0]=2;CHECK(invalid(c));
    c=good;c.layers[0].multipliers.pop_back();CHECK(invalid(c));
    c=good;c.layers[1].layer_id=c.layers[0].layer_id;CHECK(invalid(c));
    c=good;c.asset_identity.clear();CHECK(invalid(c));
    c=good;c.n_heads=INT64_MAX;CHECK(invalid(c));
    c=good;c.max_ngram_size=65;CHECK(invalid(c));
    REQUIRE_EQ(eg::ModelControls::create(good,&out),RAD_OK);
}
TEST(deepseek41_engram_row_lookup_requests_only_valid_layer_local_rows) {
    const auto model=controls();const auto packed=hashes();std::vector<int64_t> rows;
    for(int token=0;token<12;++token)for(int col=0;col<6;++col)rows.push_back(packed[size_t(token*12+6+col)]);
    std::vector<float> out(rows.size()*3,-999);int calls=0;
    eg::RowLookup lookup=[&](int layer,std::span<const int64_t> requested,int64_t width,std::span<float> values,std::string*) {
        ++calls;CHECK_EQ(layer,14);CHECK_EQ(width,3);CHECK_EQ(requested.size(),72u);CHECK_EQ(values.size(),216u);
        for(size_t i=0;i<requested.size();++i)for(int d=0;d<3;++d)values[i*3+d]=float(requested[i]*3+d);
        return RAD_OK;
    };
    REQUIRE_EQ(eg::gather_rows(*model,1,12,rows,lookup,out),RAD_OK);CHECK_EQ(calls,1);
    for(size_t i=0;i<rows.size();++i)for(int d=0;d<3;++d)CHECK_EQ(out[i*3+d],float(rows[i]*3+d));
    CHECK(std::unordered_set<int64_t>(rows.begin(),rows.end()).size()<rows.size());
    rows[0]=model->get().layers[1].table_rows;
    CHECK_EQ(eg::gather_rows(*model,1,12,rows,lookup,out),RAD_E_FORMAT);CHECK_EQ(calls,1);
    CHECK_EQ(eg::gather_rows(*model,2,12,rows,lookup,out),RAD_E_SHAPE);
}
TEST(deepseek41_engram_gate_matches_official_per_copy_norm_signed_sqrt_and_bf16_output) {
    REQUIRE_EQ(golden().at("gate_cases").size(),4u);
    for(const auto& f:golden().at("gate_cases")) {
        const auto x=f.at("x").get<std::vector<float>>(),kv=f.at("projected_kv").get<std::vector<float>>();
        const auto q=f.at("q_weight").get<std::vector<float>>(),k=f.at("k_weight").get<std::vector<float>>();
        const auto mask=f.at("token_mask").get<std::vector<uint8_t>>();
        const auto want=f.at("output").get<std::vector<float>>(),want_gate=f.at("gates").get<std::vector<float>>();
        std::vector<float> out(x.size()),gates(6);const bool bf16=f.at("dtype")=="bf16";
        REQUIRE_EQ(eg::gate_residual(3,2,4,x,kv,q,k,mask,1e-6f,bf16,out,gates),RAD_OK);
        for(size_t i=0;i<out.size();++i) {
            if(bf16)CHECK_EQ(out[i],want[i]);else CHECK_NEAR(out[i],want[i],3e-6);
        }
        for(size_t i=0;i<gates.size();++i)CHECK_NEAR(gates[i],want_gate[i],2e-6);
        CHECK_EQ(gates[2],0.0f);CHECK_EQ(gates[3],0.0f);
        for(size_t i=8;i<16;++i)CHECK_EQ(out[i],x[i]);
    }
}
TEST(deepseek41_engram_gate_refuses_bad_shapes_before_writing_and_supports_inplace) {
    const auto f=golden().at("gate_cases").at(0);
    auto x=f.at("x").get<std::vector<float>>();const auto kv=f.at("projected_kv").get<std::vector<float>>();
    const auto q=f.at("q_weight").get<std::vector<float>>(),k=f.at("k_weight").get<std::vector<float>>();
    std::vector<float> out(x.size(),999);
    CHECK_EQ(eg::gate_residual(3,2,4,x,kv,q,k,{},0,false,out),RAD_E_SHAPE);CHECK_EQ(out[0],999);
    CHECK_EQ(eg::gate_residual(3,INT64_MAX,4,x,kv,q,k,{},1e-6f,false,out),RAD_E_SHAPE);
    x[0]=std::numeric_limits<float>::infinity();
    CHECK_EQ(eg::gate_residual(3,2,4,x,kv,q,k,{},1e-6f,false,out),RAD_E_FORMAT);CHECK_EQ(out[0],999);
    x=f.at("x").get<std::vector<float>>();const auto mask=f.at("token_mask").get<std::vector<uint8_t>>();
    REQUIRE_EQ(eg::gate_residual(3,2,4,x,kv,q,k,mask,1e-6f,false,x),RAD_OK);
    const auto want=f.at("output").get<std::vector<float>>();for(size_t i=0;i<x.size();++i)CHECK_NEAR(x[i],want[i],3e-6);
}
TEST(deepseek41_engram_gate_clamps_signed_zero_and_bf16_rounds_ties_to_even) {
    CHECK(eg::gate_from_dot(0.0f)>.5f);CHECK(eg::gate_from_dot(-0.0f)<.5f);
    CHECK_NEAR(eg::gate_from_dot(4)+eg::gate_from_dot(-4),1,1e-7);
    CHECK_EQ(std::bit_cast<uint32_t>(eg::bf16(std::bit_cast<float>(0x3f808000u))),0x3f800000u);
    CHECK_EQ(std::bit_cast<uint32_t>(eg::bf16(std::bit_cast<float>(0x3f818000u))),0x3f820000u);
    CHECK(std::isinf(eg::bf16(std::numeric_limits<float>::infinity())));
    CHECK(std::isnan(eg::bf16(std::numeric_limits<float>::quiet_NaN())));
}
RAD_TEST_MAIN()
