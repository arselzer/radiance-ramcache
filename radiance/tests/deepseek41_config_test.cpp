/* DeepSeek V4.1 normalization/ownership tests. The two embedded metadata-only fixtures are
 * copied from pinned ISTA cb02422c900df489cea5833e8a92171224059b3c and official HF
 * 1bae0e1088993b9b82de08f59de5be5a34e42170; no model payload or tokenizer vocab is included. */
#include "rad_test.h"
#include "format/checkpoint.h"
#include "../arch/deepseek41/config.h"
#include <nlohmann/json.hpp>
#include <numeric>

using namespace rad;
using namespace rad::deepseek41;
using DsConfig = rad::deepseek41::Config;
namespace {
const char kGgufMetadata[] = R"META({
  "general.architecture": "deepseek41",
  "deepseek41.block_count": 40,
  "deepseek41.context_length": 1048576,
  "deepseek41.embedding_length": 5120,
  "deepseek41.attention.head_count": 64,
  "deepseek41.attention.head_count_kv": 1,
  "deepseek41.rope.scaling.type": "yarn",
  "deepseek41.rope.scaling.factor": 16.0,
  "deepseek41.rope.scaling.original_context_length": 65536,
  "deepseek41.rope.scaling.yarn_beta_fast": 32.0,
  "deepseek41.rope.scaling.yarn_beta_slow": 1.0,
  "deepseek41.rope.freq_base": 10000.0,
  "deepseek41.attention.layer_norm_rms_epsilon": 9.999999682655225e-21,
  "deepseek41.expert_count": 384,
  "deepseek41.expert_used_count": 6,
  "deepseek41.expert_gating_func": 4,
  "deepseek41.attention.key_length": 512,
  "deepseek41.attention.value_length": 512,
  "deepseek41.rope.dimension_count": 64,
  "deepseek41.attention.q_lora_rank": 1280,
  "deepseek41.attention.sliding_window": 128,
  "deepseek41.expert_feed_forward_length": 2304,
  "deepseek41.expert_shared_count": 1,
  "deepseek41.expert_weights_scale": 1.5,
  "deepseek41.expert_weights_norm": true,
  "deepseek41.swiglu_clamp_exp": [
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0
  ],
  "deepseek41.swiglu_clamp_shexp": [
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0
  ],
  "deepseek41.attention.indexer.head_count": 32,
  "deepseek41.attention.indexer.key_length": 128,
  "deepseek41.attention.indexer.top_k": 512,
  "deepseek41.attention.output_group_count": 8,
  "deepseek41.attention.output_lora_rank": 1024,
  "deepseek41.attention.compress_ratios": [
    0,
    0,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    0,
    0,
    0
  ],
  "deepseek41.attention.compress_rope_freq_base": 160000.0,
  "deepseek41.hyper_connection.count": 4,
  "deepseek41.hyper_connection.sinkhorn_iterations": 20,
  "deepseek41.hyper_connection.epsilon": 9.999999974752427e-07,
  "deepseek41.hash_layer_count": 0,
  "deepseek41.attention.candidate_block_size": 8,
  "deepseek41.attention.candidate_topk_blocks": 2048,
  "deepseek41.attention.candidate_source_layer_id": 20,
  "deepseek41.kv_source_layer_ids": [
    2,
    8,
    14,
    20
  ],
  "deepseek41.index_source_layer_ids": [
    2,
    8,
    14,
    20,
    24,
    28,
    32,
    36
  ],
  "deepseek41.dspark_target_layer_ids": [
    37,
    38,
    39
  ],
  "deepseek41.expert_scoring_func": "sqrtsoftplus",
  "deepseek41.engram.present": true,
  "deepseek41.engram.layer_ids": [
    1,
    14
  ],
  "deepseek41.engram.num_embeddings": [
    384006168,
    384016682
  ],
  "deepseek41.engram_head_dim": 256,
  "deepseek41.engram_n_heads": 8,
  "deepseek41.engram_max_ngram_size": 4,
  "deepseek41.engram_vocab_size": 16000000,
  "deepseek41.engram_compressed_vocab_size": 99092,
  "deepseek41.engram_pad_token_id": 2,
  "deepseek41.engram.source": "1bae0e1088993b9b82de08f59de5be5a34e42170",
  "deepseek41.engram.file": "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf"
})META";
const char kHfMetadata[] = R"META({
  "architectures": [
    "DeepseekV41ForCausalLM"
  ],
  "model_type": "deepseek_v41",
  "dtype": "bfloat16",
  "transformers_version": "5.6.0",
  "bos_token_id": 0,
  "eos_token_id": 1,
  "pad_token_id": 2,
  "image_token_id": 129264,
  "quantization_config": {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [
      32,
      32
    ],
    "scale_fmt": "ue8m0",
    "expert_dtype": "fp4"
  },
  "text_config": {
    "model_type": "deepseek_v41_text",
    "vocab_size": 129280,
    "hidden_size": 5120,
    "moe_intermediate_size": 2304,
    "num_hidden_layers": 40,
    "num_attention_heads": 64,
    "num_key_value_heads": 1,
    "head_dim": 512,
    "qk_rope_head_dim": 64,
    "q_lora_rank": 1280,
    "o_lora_rank": 1024,
    "o_groups": 8,
    "hidden_act": "silu",
    "swiglu_limit": 10.0,
    "rms_norm_eps": 1e-20,
    "attention_bias": false,
    "attention_dropout": 0.0,
    "initializer_range": 0.02,
    "use_cache": true,
    "tie_word_embeddings": false,
    "max_position_embeddings": 1048576,
    "rope_theta": 10000,
    "rope_scaling": {
      "rope_type": "yarn",
      "factor": 16,
      "beta_fast": 32,
      "beta_slow": 1,
      "original_max_position_embeddings": 65536
    },
    "n_routed_experts": 384,
    "n_shared_experts": 1,
    "num_experts_per_tok": 6,
    "scoring_func": "sqrtsoftplus",
    "topk_method": "noaux_tc",
    "norm_topk_prob": true,
    "routed_scaling_factor": 1.5,
    "sliding_window": 128,
    "compress_ratios": [
      0,
      0,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0
    ],
    "compress_rope_theta": 160000,
    "kv_source_layer_ids": [
      2,
      8,
      14,
      20
    ],
    "index_source_layer_ids": [
      2,
      8,
      14,
      20,
      24,
      28,
      32,
      36
    ],
    "index_n_heads": 32,
    "index_head_dim": 128,
    "index_topk": 512,
    "candidate_source_layer_id": 20,
    "candidate_topk_blocks": 2048,
    "candidate_block_size": 8,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-06,
    "engram_layer_ids": [
      1,
      14
    ],
    "engram_num_embeddings": [
      384006168,
      384016682
    ],
    "engram_max_ngram_size": 4,
    "engram_vocab_size": 16000000,
    "engram_n_heads": 8,
    "engram_head_dim": 256,
    "engram_pad_token_id": 2,
    "engram_compressed_vocab_size": 99092,
    "num_nextn_predict_layers": 3,
    "dspark_block_size": 5,
    "dspark_noise_token_id": 128799,
    "dspark_target_layer_ids": [
      37,
      38,
      39
    ],
    "dspark_markov_rank": 256,
    "dspark_n_routed_experts": 128,
    "dspark_num_experts_per_tok": 3
  },
  "vision_config": {
    "model_type": "deepseek_v41_vision",
    "num_hidden_layers": 32,
    "hidden_size": 1024,
    "num_attention_heads": 16,
    "intermediate_size": 2816,
    "patch_size": 14,
    "rope_theta": 10000,
    "downsample_ratio": 3,
    "max_image_tokens": 1024,
    "min_pixels": 295936,
    "max_wh_ratio": null
  }
})META";

void flatten(const nlohmann::json& value,const std::string& prefix,MetaOwn* out) {
    for (auto it=value.begin();it!=value.end();++it) {
        const std::string key=prefix.empty()?it.key():prefix+"."+it.key();
        if (it->is_object()) { flatten(*it,key,out); continue; }
        if (it->is_null()) continue;
        if (it->is_array()) {
            std::string list;
            for(const auto& item:*it)list+=(list.empty()?"":" ")+(item.is_string()?item.get<std::string>():item.dump());
            out->kv(key,list);
        } else out->kv(key,it->is_string()?it->get<std::string>():it->dump());
    }
}
MetaOwn gguf_meta() {
    MetaOwn out;out.arch="deepseek41";
    flatten(nlohmann::json::parse(kGgufMetadata),"",&out);
    // Actual GGUF numeric arrays use commas, unlike HF's flattened spaces.
    const auto values=nlohmann::json::parse(kGgufMetadata);
    for (auto it=values.begin();it!=values.end();++it)if(it->is_array()) {
        std::string list;for(const auto& item:*it)list+=(list.empty()?"":",")+item.dump();
        out.set(it.key(),list);
    }
    out.m.n_vocab=129280; // token count from tokenizer.ggml.tokens in the same shard
    return out;
}
MetaOwn hf_meta() {
    MetaOwn out;out.arch="deepseekv41";flatten(nlohmann::json::parse(kHfMetadata),"",&out);return out;
}
int read(MetaOwn& meta,DsConfig* out,std::string* why=nullptr) {
    meta.finish();return normalize_config(meta.m,out,why);
}
DsConfig baseline() {
    auto meta=gguf_meta();DsConfig c;std::string why;
    if(read(meta,&c,&why)!=RAD_OK)throw std::runtime_error(why);
    return c;
}
void erase(MetaOwn* meta,const char* key) {
    for(size_t i=0;i<meta->keys.size();++i)if(meta->keys[i]==key) {
        meta->keys.erase(meta->keys.begin()+i);meta->vals.erase(meta->vals.begin()+i);return;
    }
}
}

TEST(deepseek41_pinned_gguf_and_hf_metadata_normalize_to_the_same_architecture) {
    auto gm=gguf_meta(),hm=hf_meta();DsConfig g,h;std::string why;
    REQUIRE_EQ(read(gm,&g,&why),RAD_OK);REQUIRE_EQ(read(hm,&h,&why),RAD_OK);
    CHECK_EQ(g.n_layers,40);CHECK_EQ(g.n_embd,5120);CHECK_EQ(g.n_head,64);CHECK_EQ(g.n_head_kv,1);
    CHECK_EQ(g.head_dim,512);CHECK_EQ(g.rope_head_dim,64);CHECK_EQ(g.q_lora_rank,1280);
    CHECK_EQ(g.o_groups,8);CHECK_EQ(g.o_lora_rank,1024);CHECK_EQ(g.n_ff,2304);CHECK_EQ(g.n_vocab,129280);
    CHECK_EQ(g.n_expert,384);CHECK_EQ(g.n_expert_used,6);CHECK_EQ(g.n_expert_shared,1);
    CHECK_EQ(g.window_size,128);CHECK_EQ(g.index_topk,512);CHECK_EQ(g.index_n_heads,32);CHECK_EQ(g.index_head_dim,128);
    CHECK_EQ(g.route_scale,1.5);CHECK_EQ(g.gate_temp,1.0);CHECK(g.norm_topk_prob);CHECK_EQ(g.score_func,std::string("sqrtsoftplus"));
    CHECK(g.hc_shifted);CHECK_EQ(g.hc_mult,4);CHECK_EQ(g.hc_sinkhorn_iters,20);
    CHECK_EQ(g.context_length,1048576);CHECK_EQ(g.rope_type,std::string("yarn"));
    CHECK_EQ(g.rope_theta,10000);CHECK_EQ(g.compress_rope_theta,160000);CHECK_EQ(g.rope_factor,16);
    CHECK_EQ(g.original_seq_len,65536);CHECK_EQ(g.beta_fast,32);CHECK_EQ(g.beta_slow,1);
    CHECK_EQ(g.compress_ratios.size(),43u);CHECK_EQ(g.layers.size(),40u);CHECK_EQ(g.draft_layers,3);
    CHECK(g.compress_ratios==h.compress_ratios);CHECK(g.kv_source_layers==h.kv_source_layers);
    CHECK(g.index_k_source_layers==h.index_k_source_layers);CHECK(g.index_source_layers==h.index_source_layers);
    CHECK(g.routed_clamp==h.routed_clamp);CHECK(g.shared_clamp==h.shared_clamp);
    CHECK(g.engram_layers==h.engram_layers);CHECK(g.engram_rows==h.engram_rows);
    CHECK(g.engram_layers==std::vector<int64_t>({1,14}));CHECK(g.engram_rows==std::vector<int64_t>({384006168,384016682}));
    CHECK_EQ(g.engram_compressed_vocab_size,99092);CHECK_EQ(g.engram_pad_id,2);
    CHECK_EQ(g.n_layers,h.n_layers);CHECK_EQ(g.n_vocab,h.n_vocab);CHECK_EQ(g.hc_shifted,h.hc_shifted);
    CHECK_NEAR(g.norm_eps/h.norm_eps,1.0,1e-6);CHECK_NEAR(g.hc_eps/h.hc_eps,1.0,1e-6);
}
TEST(deepseek41_accepts_only_explicit_architecture_aliases) {
    for(const char* id:{"deepseek41","deepseek_v41","deepseekv41"}) {
        auto meta=gguf_meta();meta.arch=id;DsConfig c;CHECK_EQ(read(meta,&c),RAD_OK);
    }
    for(const char* id:{"deepseek4","deepseekv3","deepseek_v41_text",""}) {
        auto meta=gguf_meta();meta.arch=id;DsConfig c;CHECK(read(meta,&c)<0);
    }
}
TEST(deepseek41_source_maps_match_the_recorded_1017_layer_schedule) {
    const auto c=baseline();
    for(int layer=0;layer<40;++layer) {
        const auto& s=c.layers[layer];
        if(layer<2) {CHECK_EQ(s.compress_ratio,0);CHECK_EQ(s.kv,-1);CHECK_EQ(s.index_k,-1);CHECK_EQ(s.index,-1);}
        else {
            const int kv=layer<8?2:layer<14?8:layer<20?14:20;
            const int idx=layer<20?kv:20+4*((layer-20)/4);
            CHECK_EQ(s.compress_ratio,layer<20?2:1);CHECK_EQ(s.kv,kv);CHECK_EQ(s.index_k,kv);CHECK_EQ(s.index,idx);
            CHECK_EQ(s.writes_kv(layer),layer==kv);CHECK_EQ(s.writes_index_k(layer),layer==kv);
            CHECK_EQ(s.writes_index(layer),layer==idx);
        }
        CHECK_EQ(s.candidate,layer<20?-1:20);
        CHECK_EQ(s.writes_candidates(layer),layer==20);CHECK_EQ(s.uses_candidates(layer),layer>20);
    }
}
TEST(deepseek41_keeps_index_key_ownership_distinct_from_topk_and_kv_ownership) {
    auto c=baseline();c.index_k_source_layers={2,5,8,11,14,17,20,24,28,32,36};
    c.index_source_layers={2,4,5,8,10,11,14,16,17,20,22,24,28,32,36};
    std::string why;REQUIRE_EQ(validate_config(&c,&why),RAD_OK);
    CHECK_EQ(c.layers[4].index,4);CHECK_EQ(c.layers[4].index_k,2);CHECK_EQ(c.layers[4].kv,2);
    CHECK_EQ(c.layers[5].index,5);CHECK_EQ(c.layers[5].index_k,5);CHECK_EQ(c.layers[5].kv,2);
    CHECK_EQ(c.layers[11].index_k,11);CHECK_EQ(c.layers[11].kv,8);
    c.index_k_source_layers={2,5,14,20};CHECK(validate_config(&c,&why)<0);
}
TEST(deepseek41_rejects_source_gaps_unsorted_ids_and_incompatible_segments) {
    for(const auto& bad:std::vector<std::vector<int64_t>>{{8,14,20},{2,8,8,14,20},{2,14,8,20},{0,2,8,14,20},{2,8,14,40}}) {
        auto c=baseline();c.kv_source_layers=bad;CHECK(validate_config(&c)<0);
    }
    auto c=baseline();c.kv_source_layers={2,8,14};CHECK(validate_config(&c)<0);
    c=baseline();c.compress_ratios[5]=0;CHECK(validate_config(&c)<0);
    c=baseline();c.index_source_layers={2,8,20};CHECK(validate_config(&c)<0);
}
TEST(deepseek41_candidate_filtering_requires_a_real_producer_and_sufficient_capacity) {
    auto c=baseline();c.candidate_source=21;CHECK(validate_config(&c)<0);
    c=baseline();c.candidate_topk_blocks=8;c.candidate_block_size=8;CHECK(validate_config(&c)<0);
    c=baseline();c.candidate_topk_blocks=7;CHECK(validate_config(&c)<0);
    c=baseline();c.candidate_block_size=0;CHECK(validate_config(&c)<0);
    c=baseline();c.candidate_topk_blocks=INT64_MAX-7;CHECK(validate_config(&c)<0);
    c=baseline();c.candidate_source=-1;CHECK(validate_config(&c)<0);
    c.candidate_topk_blocks=c.candidate_block_size=0;REQUIRE_EQ(validate_config(&c),RAD_OK);
    for(const auto& layer:c.layers)CHECK_EQ(layer.candidate,-1);
}
TEST(deepseek41_semantic_candidate_geometry_has_no_hidden_gpu_tile_restrictions) {
    auto c=baseline();c.candidate_topk_blocks=65;c.candidate_block_size=8;
    CHECK_EQ(validate_config(&c),RAD_OK);
    c.candidate_topk_blocks=16;c.candidate_block_size=32;
    CHECK_EQ(validate_config(&c),RAD_OK);
}
TEST(deepseek41_preserves_draft_metadata_without_sharing_or_serving_draft_layers) {
    auto c=baseline();CHECK_EQ(c.draft_layers,3);CHECK_EQ(c.layers.size(),40u);
    c.compress_ratios[40]=1;CHECK(validate_config(&c)<0);
    c=baseline();c.compress_ratios.pop_back();CHECK(validate_config(&c)<0);
    c=baseline();c.dspark_target_layers={37,38,40};CHECK(validate_config(&c)<0);
}
TEST(deepseek41_rejects_conflicting_aliases_and_malformed_numbers_without_changing_output) {
    for(const auto& key:std::vector<std::string>{"text_config.hidden_size","hidden_size"}) {
        auto meta=gguf_meta();meta.set(key,"5121");DsConfig c;c.n_embd=777;
        std::string why;CHECK(read(meta,&c,&why)<0);CHECK_EQ(c.n_embd,777);CHECK(why.find("conflicting")!=std::string::npos);
    }
    for(const char* value:{"40oops","9223372036854775808","1e3",""}) {
        auto meta=gguf_meta();meta.set("deepseek41.block_count",value);DsConfig c;CHECK(read(meta,&c)<0);
    }
    for(const char* value:{"nan","inf","1e9999","1.5x",""}) {
        auto meta=gguf_meta();meta.set("deepseek41.expert_weights_scale",value);DsConfig c;CHECK(read(meta,&c)<0);
    }
    auto meta=gguf_meta();meta.kv("deepseek41.embedding_length","1");DsConfig c;CHECK(read(meta,&c)<0);
}
TEST(deepseek41_requires_full_valid_metadata_and_refuses_empty_array_elements) {
    for(const char* value:{"2,,8,14,20","2,8,14,20,",",2,8,14,20","2 8a 14 20","[2,8,14,20"}) {
        auto meta=gguf_meta();meta.set("deepseek41.kv_source_layer_ids",value);DsConfig c;CHECK(read(meta,&c)<0);
    }
    for(const char* key:{"deepseek41.expert_scoring_func","deepseek41.kv_source_layer_ids", "deepseek41.expert_weights_norm"}) {
        auto meta=gguf_meta();erase(&meta,key);DsConfig c;CHECK(read(meta,&c)<0);
    }
    auto meta=gguf_meta();meta.set("deepseek41.kv_source_layer_ids","[2, 8, 14, 20]");DsConfig c;
    CHECK_EQ(read(meta,&c),RAD_OK);
}
TEST(deepseek41_preserves_per_layer_clamps_and_rejects_disagreeing_uniform_aliases) {
    auto meta=gguf_meta();meta.set("text_config.swiglu_limit","11");DsConfig c;CHECK(read(meta,&c)<0);
    c=baseline();c.shared_clamp[14]=9;c.routed_clamp[14]=11;REQUIRE_EQ(validate_config(&c),RAD_OK);
    CHECK_EQ(c.shared_clamp[14],9);CHECK_EQ(c.routed_clamp[14],11);
    c.routed_clamp[0]=-1;CHECK(validate_config(&c)<0);
    c=baseline();c.shared_clamp.pop_back();CHECK(validate_config(&c)<0);
}
TEST(deepseek41_rejects_unimplemented_semantics_and_inconsistent_geometry) {
    auto c=baseline();c.hc_shifted=false;CHECK(validate_config(&c)<0);
    c=baseline();c.hc_mult=2;CHECK(validate_config(&c)<0);
    c=baseline();c.hc_sinkhorn_iters=257;CHECK(validate_config(&c)<0);
    c=baseline();c.n_head_kv=2;CHECK(validate_config(&c)<0);
    c=baseline();c.rope_head_dim=63;CHECK(validate_config(&c)<0);
    c=baseline();c.o_groups=7;CHECK(validate_config(&c)<0);
    c=baseline();c.score_func="softmax";CHECK(validate_config(&c)<0);
    c=baseline();c.norm_topk_prob=false;CHECK(validate_config(&c)<0);
    c=baseline();c.gate_temp=0;CHECK(validate_config(&c)<0);
    c=baseline();c.n_expert_used=385;CHECK(validate_config(&c)<0);
    c=baseline();c.rope_type="linear";CHECK(validate_config(&c)<0);
    c=baseline();c.beta_fast=c.beta_slow;CHECK(validate_config(&c)<0);
    c=baseline();c.n_embd=INT64_MAX;CHECK(validate_config(&c)<0);
}
TEST(deepseek41_engram_metadata_must_cover_matching_layers_and_table_geometry) {
    auto c=baseline();c.engram_rows.pop_back();CHECK(validate_config(&c)<0);
    c=baseline();c.engram_present=false;CHECK(validate_config(&c)<0);
    c=baseline();c.engram_layers={14,1};CHECK(validate_config(&c)<0);
    c=baseline();c.engram_rows[0]=23;CHECK(validate_config(&c)<0);
    c=baseline();c.engram_compressed_vocab_size=c.n_vocab+1;CHECK(validate_config(&c)<0);
    c=baseline();c.engram_pad_id=c.n_vocab;CHECK(validate_config(&c)<0);
    c=baseline();c.engram_n_heads=65537;CHECK(validate_config(&c)<0);
}
TEST(deepseek41_rotating_tp4_covers_every_original_ffn_dimension_once) {
    int64_t aggregate[4]{};std::string why;
    for(int layer=0;layer<40;++layer) {
        int64_t end=0;
        for(int rank=0;rank<4;++rank) {
            Partition p;REQUIRE_EQ(ffn_partition(2304,4,rank,layer,256,&p,&why),RAD_OK);
            CHECK_EQ(p.offset,end);CHECK_EQ(p.width,rank==layer%4?768:512);
            CHECK_EQ(p.offset%256,0);CHECK_EQ(p.width%256,0);aggregate[rank]+=p.width;end+=p.width;
        }
        CHECK_EQ(end,2304);
    }
    for(auto width:aggregate)CHECK_EQ(width,23040);
}
TEST(deepseek41_group_partitions_handle_other_world_sizes_and_refuse_invalid_cuts) {
    for(int world=1;world<=9;++world)for(int layer=0;layer<world;++layer) {
        int64_t end=0;
        for(int rank=0;rank<world;++rank) {
            Partition p;REQUIRE_EQ(ffn_partition(2304,world,rank,layer,256,&p),RAD_OK);
            CHECK_EQ(p.offset,end);CHECK_EQ(p.width%256,0);CHECK(p.width>0);end+=p.width;
        }
        CHECK_EQ(end,2304);
    }
    Partition p;
    CHECK(ffn_partition(2305,4,0,0,256,&p)<0);CHECK(ffn_partition(2304,10,0,0,256,&p)<0);
    CHECK(ffn_partition(2304,4,4,0,256,&p)<0);CHECK(ffn_partition(2304,4,0,-1,256,&p)<0);
    CHECK(ffn_partition(2304,0,0,0,256,&p)<0);CHECK(ffn_partition(2304,4,0,0,0,&p)<0);
    const auto c=baseline();CHECK_EQ(validate_tp_geometry(c,4),RAD_OK);CHECK_EQ(validate_tp_geometry(c,8),RAD_OK);
    CHECK(validate_tp_geometry(c,3)<0);CHECK(validate_tp_geometry(c,16)<0);CHECK(validate_tp_geometry(c,0)<0);
}
TEST(deepseek41_rejects_hf_activation_bias_dropout_and_architecture_mismatches) {
    for (const auto& [key,value] : std::vector<std::pair<std::string,std::string>>{
        {"text_config.hidden_act","gelu"},{"text_config.attention_bias","true"},
        {"text_config.attention_dropout","0.1"},{"model_type","deepseek_v3"},
        {"text_config.model_type","llama"}}) {
        auto meta=hf_meta();meta.set(key,value);DsConfig c;CHECK(read(meta,&c)<0);
    }
    auto meta=gguf_meta();meta.set("deepseek41.attention.value_length","256");DsConfig c;CHECK(read(meta,&c)<0);
    c=baseline();c.index_head_dim=32;CHECK(validate_config(&c)<0);
    c=baseline();c.n_expert=INT64_MAX/2;CHECK(validate_config(&c)<0);
}
RAD_TEST_MAIN()
