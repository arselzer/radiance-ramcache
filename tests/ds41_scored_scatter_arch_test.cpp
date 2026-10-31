#define RAD_ARCH_NO_EXPORTS 1
#include "arch_fakes.h"
#include "../arch/deepseek41/deepseek41.cpp"
namespace {
using namespace rad_ds41_scored_scatter;
struct Env {
    explicit Env(const char*v){unsetenv(kEnv);if(v)setenv(kEnv,v,1);}
    ~Env(){unsetenv(kEnv);unsetenv("RADIANCE_EXPERIMENTAL_DS41_ROUTE_TRACE");unsetenv("RADIANCE_EXPERIMENTAL_DS41_P2_AUDIT");}
};
MetaKV meta(){RadModelMeta b{};b.arch_id="deepseek41";b.name="ds41-native-scored-scatter";b.n_layers=40;
    b.n_embd=5120;b.n_head=64;b.n_head_kv=1;b.head_dim=512;b.n_vocab=129280;b.n_expert=384;b.n_expert_used=6;
    b.n_expert_shared=1;b.rms_eps=1e-6f;b.rope_theta=10000.f;b.rope_scale=1.f;MetaKV m(b);
    std::string ratios;for(int l=0;l<40;++l)ratios+=(l?",":"")+std::to_string(l<2?0:l<20?2:1);
    m.set("deepseek41.block_count","40").set("deepseek41.rope.dimension_count","64")
      .set("deepseek41.attention.q_lora_rank","1280").set("deepseek41.attention.output_group_count","8")
      .set("deepseek41.attention.output_lora_rank","1024").set("deepseek41.attention.sliding_window","128")
      .set("deepseek41.expert_feed_forward_length","2304").set("deepseek41.expert_weights_scale","1.5")
      .set("deepseek41.expert_weights_norm","1").set("deepseek41.swiglu_clamp_exp","10")
      .set("deepseek41.attention.indexer.head_count","32").set("deepseek41.attention.indexer.key_length","128")
      .set("deepseek41.attention.indexer.top_k","512").set("deepseek41.attention.indexer.candidate_source_layer","20")
      .set("deepseek41.attention.indexer.candidate_top_k_blocks","2048").set("deepseek41.attention.indexer.candidate_block_size","8")
      .set("deepseek41.hyper_connection.count","4").set("deepseek41.hyper_connection.sinkhorn_iterations","20")
      .set("deepseek41.hyper_connection.epsilon","1e-6").set("deepseek41.attention.compress_rope_freq_base","160000")
      .set("deepseek41.attention.compress_ratios",ratios.c_str()).set("deepseek41.attention.kv_source_layers","2,8,14,20")
      .set("deepseek41.attention.indexer.source_layers","2,8,14,20,24,28,32,36").set("deepseek41.ple.layers","1,14")
      .set("deepseek41.ple.ngram_size","4").set("deepseek41.ple.heads_per_ngram","8")
      .set("deepseek41.embedding_length_per_layer_input","256").set("deepseek41.engram.pad_token_id","2")
      .set("deepseek41.ple.head_vocab_sizes","1000,1000");return m;
}
int declare(RadBuilder&b,int rank=0,int world=4,bool unavailable=false){
    b.kv_block_of=[](const std::string&n){for(const char*s:{"kv_compressed_r","kv_index_r"})
        if(n.rfind(s,0)==0)return(int64_t)std::atoll(n.c_str()+std::strlen(s));return(int64_t)0;};
    if(unavailable)b.undeclared.insert(kOp);auto m=meta();auto c=spec_ctx(rank,world,0);c.max_tok=32;c.max_seqs=8;
    return stderr_of_rc([&]{return deepseek41::declare(&b,m.meta(),&c);});
}
uint64_t step_hash(RadBuilder&b,int rows,int rank){auto k=kit(b,{rows},{});
    auto v=run_passes(b,deepseek41::step,{k.get()},rank,4);CHECK(steps_ok(v));CHECK_EQ(report_lint(lint_step(b,view(v))),(size_t)0);
    return issue_hash(b,v,{k.get()});}
}
TEST(scored_scatter_off_empty_and_zero_preserve_the_complete_graph_and_issue_hash){
    uint64_t graph=0,step=0;{Env e(nullptr);RadBuilder b;REQUIRE_EQ(declare(b),RAD_OK);graph=graph_hash(b);step=step_hash(b,1,0);}
    for(const char*s:{"","0"}){Env e(s);RadBuilder b;REQUIRE_EQ(declare(b),RAD_OK);CHECK_EQ(graph_hash(b),graph);CHECK_EQ(step_hash(b,1,0),step);}
}
TEST(scored_scatter_m1_replaces_only_select_and_sort_preserving_projection_and_other_heights){
    for(int rank=0;rank<4;++rank){uint64_t old2=0,old17=0;
        {Env e(nullptr);RadBuilder b;REQUIRE_EQ(declare(b,rank),RAD_OK);old2=step_hash(b,2,rank);old17=step_hash(b,17,rank);}
        Env e("1");RadBuilder b;REQUIRE_EQ(declare(b,rank),RAD_OK);auto*m=deepseek41::models[rank].get();
        REQUIRE(m);CHECK(m->scored_scatter);CHECK_EQ(step_hash(b,2,rank),old2);CHECK_EQ(step_hash(b,17,rank),old17);
        auto k=kit(b,{1},{});auto v=run_passes(b,deepseek41::step,{k.get()},rank,4);REQUIRE(steps_ok(v));
        CHECK_EQ(report_lint(lint_step(b,view(v))),(size_t)0);const auto issued=issued_ops(b,*v[0]);
        CHECK_EQ(count_issued(issued,kOp),40);CHECK_EQ(count_issued(issued,"router_scored_topk"),0);
        CHECK_EQ(count_issued(issued,"moe_scatter"),0);CHECK_EQ(count_issued(issued,"moe_scatter_inv"),0);
        for(const auto&L:m->layers){int projects=0;for(const auto&i:v[0]->issues)projects+=i.op==L.moe.op_router;CHECK_EQ(projects,1);}
    }
}
TEST(scored_scatter_missing_capability_wrong_world_and_unproven_diagnostics_refuse){
    {Env e("1");RadBuilder b;CHECK(declare(b,0,4,true)<0);}
    {Env e("1");RadBuilder b;CHECK(declare(b,0,2)<0);}
    for(const char*s:{"2","01","true"}){Env e(s);RadBuilder b;CHECK(declare(b)<0);}
    for(const char*key:{"RADIANCE_EXPERIMENTAL_DS41_ROUTE_TRACE","RADIANCE_EXPERIMENTAL_DS41_P2_AUDIT"}){
        Env e("1");setenv(key,"1",1);RadBuilder b;CHECK(declare(b)<0);unsetenv(key);}
}
RAD_TEST_MAIN()
