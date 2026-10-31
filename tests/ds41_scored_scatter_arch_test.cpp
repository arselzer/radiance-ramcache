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
    /* An encoding-free fake means block-FP8 experts, but ds41's input wires are BF16. Model a
     * supported BF16 expert container explicitly instead of issuing null quantized wires. */
    b.encs={{"ffn_gate_up_exps",rad_enc_plain(RAD_BF16)},
            {"ffn_down_exps",rad_enc_plain(RAD_BF16)}};
    if(unavailable)b.undeclared.insert(kOp);auto m=meta();auto c=spec_ctx(rank,world,0);c.max_tok=32;c.max_seqs=8;
    return stderr_of_rc([&]{return deepseek41::declare(&b,m.meta(),&c);});
}
/* The generic fake lint predates libr4d's pitched_rows()/gemm_ops(pitched=1)
 * (r4d_gemm_args.h) and BF16 m64/tiled row-pitch support. Admit only the two exact native
 * output-A views per layer here; keep the generic lint's bounds, liveness and every other finding.
 * This is fixture-local: unknown GEMMs and malformed group views never gain an exemption. */
bool native_output_view(const RadBuilder&b,const deepseek41::Model&m,
                        const deepseek41::Layer&L,int j,const RecIssue&i,int64_t rows){
    if(m.gr!=2||m.g.n_head!=16||m.c.head_dim!=512||m.c.o_rank!=1024||
       i.op!=L.o_a[(size_t)j]||i.n!=rows||i.opd.size()!=3)return false;
    const auto*op=b.op_of(i.op);if(!op||op->op!="gemm_nt"||pint(op,"K")!=4096||pint(op,"N")!=1024)return false;
    const auto&a=i.opd[0];const auto&w=i.opd[1];const auto&y=i.opd[2];
    if(a.kind!=RAD_OPK_BUF||a.handle!=m.o||a.offset!=j*4096||a.rows!=rows||a.cols!=4096||
       w.kind!=RAD_OPK_WEIGHT||w.handle!=L.wo_a[(size_t)j]||w.offset||w.rows||w.cols||
       y.kind!=RAD_OPK_BUF||y.handle!=m.oa||y.offset!=j*1024||y.rows!=rows||y.cols!=1024)return false;
    const auto&ad=b.bufs[m.o-1].d;const auto&yd=b.bufs[m.oa-1].d;
    return ad.dtype==RAD_BF16&&yd.dtype==RAD_BF16&&ad.rank==2&&yd.rank==2&&
           ad.shape[0]>=rows&&yd.shape[0]>=rows&&ad.shape[1]==8192&&yd.shape[1]==2048;
}
std::vector<std::string> strict_lint(const RadBuilder&b,const deepseek41::Model&m,
                                   const std::vector<std::unique_ptr<RadCtx>>&v,int64_t rows){
    auto bad=lint_step(b,view(v));std::set<std::string> supported;size_t groups=0;
    std::map<rad_op,size_t> group_counts;
    for(const auto&L:m.layers)for(rad_op h:L.o_a)group_counts.emplace(h,0);
    for(size_t p=0;p<v.size();++p)for(size_t n=0;n<v[p]->issues.size();++n){
        const auto&i=v[p]->issues[n];
        for(const auto&L:m.layers)for(int j=0;j<m.gr;++j)if(i.op==L.o_a[(size_t)j]){
            ++groups;++group_counts[i.op];
            if(!native_output_view(b,m,L,j,i,rows)){
                bad.push_back("invalid native grouped-attention output-A view");continue;
            }
            char at[128];std::snprintf(at,sizeof at,"pass %zu issue %zu (gemm_nt, op %u) operand 0: ",p,n,(unsigned)i.op);
            supported.insert(std::string(at)+"gemm_nt reads 4096 of 8192 columns of 'attn_o' as its input, which is not contiguous");
        }
    }
    if(groups!=m.layers.size()*2||supported.size()!=m.layers.size()*2)
        bad.push_back("missing or duplicate native grouped-attention output-A views");
    for(const auto&[op,count]:group_counts)if(count!=1)
        bad.push_back("native grouped-attention output-A op does not occur exactly once");
    bad.erase(std::remove_if(bad.begin(),bad.end(),[&](const std::string&s){return supported.count(s)!=0;}),bad.end());
    return bad;
}
std::unique_ptr<Kit> batch(const RadBuilder&b,int rows){
    /* Native N1/N2/N4/N8 decode, then a non-speculative M17 prefill. kit({17},{}) would
     * instead invent 16 speculative rows although this declaration has max_spec=0. */
    return rows<=8?kit(b,std::vector<int>((size_t)rows,1),{}):kit(b,{}, {rows});
}
uint64_t graph_without_scored_op(RadBuilder b){
    b.ops.erase(std::remove_if(b.ops.begin(),b.ops.end(),[](const RecOp&o){return o.op==kOp;}),b.ops.end());
    return graph_hash(b);
}
uint64_t unchanged_m1_issues(const RadBuilder&b,const std::vector<std::unique_ptr<RadCtx>>&v,const Kit&k){
    std::vector<std::unique_ptr<RadCtx>> rest;
    for(const auto&c:v){auto out=std::make_unique<RadCtx>(*c);out->issues.clear();std::vector<size_t>prefix{0};
        for(const auto&i:c->issues){const auto&name=b.op_of(i.op)->op;
            if(name!=kOp&&name!="router_scored_topk"&&name!="moe_scatter"&&name!="moe_scatter_inv")out->issues.push_back(i);
            prefix.push_back(out->issues.size());
        }
        for(auto&join:out->joins)join.at=prefix.at(join.at);
        rest.push_back(std::move(out));
    }
    return issue_hash(b,rest,{&k});
}
uint64_t step_hash(RadBuilder&b,int rows,int rank){auto k=batch(b,rows);
    CHECK_EQ(k->b.n_spec,int64_t(0));
    auto v=run_passes(b,deepseek41::step,{k.get()},rank,4);CHECK(steps_ok(v));
    CHECK_EQ(report_lint(strict_lint(b,*deepseek41::models[rank],v,rows)),(size_t)0);
    return issue_hash(b,v,{k.get()});}
}
TEST(scored_scatter_off_empty_and_zero_preserve_the_complete_graph_and_issue_hash){
    uint64_t graph=0,step=0;{Env e(nullptr);RadBuilder b;REQUIRE_EQ(declare(b),RAD_OK);graph=graph_hash(b);step=step_hash(b,1,0);}
    for(const char*s:{"","0"}){Env e(s);RadBuilder b;REQUIRE_EQ(declare(b),RAD_OK);CHECK_EQ(graph_hash(b),graph);CHECK_EQ(step_hash(b,1,0),step);}
}
TEST(scored_scatter_m1_replaces_only_select_and_sort_preserving_projection_and_other_heights){
    for(int rank=0;rank<4;++rank){std::map<int,uint64_t> old;uint64_t old_graph=0,old_m1_rest=0;
        {Env e(nullptr);RadBuilder b;REQUIRE_EQ(declare(b,rank),RAD_OK);old_graph=graph_hash(b);
            for(int rows:{2,4,8,17})old[rows]=step_hash(b,rows,rank);
            auto k=batch(b,1);auto v=run_passes(b,deepseek41::step,{k.get()},rank,4);REQUIRE(steps_ok(v));
            CHECK_EQ(report_lint(strict_lint(b,*deepseek41::models[rank],v,1)),size_t(0));
            const auto issued=issued_ops(b,*v[0]);CHECK_EQ(count_issued(issued,"router_scored_topk"),40);
            CHECK_EQ(count_issued(issued,"moe_scatter")+count_issued(issued,"moe_scatter_inv"),40);
            old_m1_rest=unchanged_m1_issues(b,v,*k);
        }
        Env e("1");RadBuilder b;REQUIRE_EQ(declare(b,rank),RAD_OK);auto*m=deepseek41::models[rank].get();
        REQUIRE(m);CHECK(m->scored_scatter);for(const auto&[rows,h]:old)CHECK_EQ(step_hash(b,rows,rank),h);
        CHECK_EQ(graph_without_scored_op(b),old_graph);CHECK_EQ(count_op(b,kOp),40);
        auto k=kit(b,{1},{});auto v=run_passes(b,deepseek41::step,{k.get()},rank,4);REQUIRE(steps_ok(v));
        CHECK_EQ(report_lint(strict_lint(b,*m,v,1)),(size_t)0);const auto issued=issued_ops(b,*v[0]);
        CHECK_EQ(unchanged_m1_issues(b,v,*k),old_m1_rest);
        CHECK_EQ(count_issued(issued,kOp),40);CHECK_EQ(count_issued(issued,"router_scored_topk"),0);
        CHECK_EQ(count_issued(issued,"moe_scatter"),0);CHECK_EQ(count_issued(issued,"moe_scatter_inv"),0);
        for(const auto&L:m->layers){int projects=0;for(const auto&i:v[0]->issues)projects+=i.op==L.moe.op_router;CHECK_EQ(projects,1);}
    }
}
TEST(scored_scatter_fixture_refuses_malformed_supported_views_and_keeps_other_lint_fatal){
    Env e("1");RadBuilder b;REQUIRE_EQ(declare(b),RAD_OK);const auto&m=*deepseek41::models[0];
    auto k=batch(b,1);auto v=run_passes(b,deepseek41::step,{k.get()},0,4);REQUIRE(steps_ok(v));
    REQUIRE(strict_lint(b,m,v,1).empty());
    auto&issues=v[0]->issues;size_t at=0;while(at<issues.size()&&issues[at].op!=m.layers[0].o_a[0])++at;
    REQUIRE(at<issues.size());const RecIssue original=issues[at];
    for(int field=0;field<4;++field){issues[at]=original;
        if(field==0)++issues[at].opd[0].offset;
        if(field==1)--issues[at].opd[0].cols;
        if(field==2)++issues[at].opd[2].offset;
        if(field==3)issues[at].opd[0].rows=33;
        CHECK(!strict_lint(b,m,v,1).empty());
    }
    issues[at]=original;
    /* The fused op's own buffer bounds are never one of the supported old GEMM findings. */
    size_t fused=0;while(fused<issues.size()&&b.op_of(issues[fused].op)->op!=kOp)++fused;
    REQUIRE(fused<issues.size());issues[fused].opd[0].handle=0;
    CHECK(!strict_lint(b,m,v,1).empty());
}
TEST(scored_scatter_missing_capability_wrong_world_and_unproven_diagnostics_refuse){
    {Env e("1");RadBuilder b;CHECK(declare(b,0,4,true)<0);}
    {Env e("1");RadBuilder b;CHECK(declare(b,0,2)<0);}
    for(const char*s:{"2","01","true"}){Env e(s);RadBuilder b;CHECK(declare(b)<0);}
    for(const char*key:{"RADIANCE_EXPERIMENTAL_DS41_ROUTE_TRACE","RADIANCE_EXPERIMENTAL_DS41_P2_AUDIT"}){
        Env e("1");setenv(key,"1",1);RadBuilder b;CHECK(declare(b)<0);unsetenv(key);}
}
RAD_TEST_MAIN()
