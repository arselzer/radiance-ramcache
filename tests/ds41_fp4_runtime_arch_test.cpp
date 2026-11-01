#define RAD_ARCH_NO_EXPORTS 1
#include "arch_fakes.h"
#include "../arch/deepseek41/deepseek41.cpp"
namespace f=rad_ds41_fp4_runtime;
namespace {
struct Env {
    Env(const char* value,const char* audit=nullptr,bool packed=false) {
        unsetenv(f::kEnv);unsetenv(f::kAuditEnv);unsetenv("RADIANCE_EXPERIMENTAL_DS41_KV_PACKED");
        if(value)setenv(f::kEnv,value,1);if(audit)setenv(f::kAuditEnv,audit,1);
        if(packed)setenv("RADIANCE_EXPERIMENTAL_DS41_KV_PACKED","1",1);
    }
    ~Env(){unsetenv(f::kEnv);unsetenv(f::kAuditEnv);unsetenv("RADIANCE_EXPERIMENTAL_DS41_KV_PACKED");}
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
int declare(RadBuilder&b,int rank=0,bool absent=false) {
    b.kv_block_of=[](const std::string&n){for(const char*s:{"kv_compressed_r","kv_index_r"})
        if(n.rfind(s,0)==0)return(int64_t)std::atoll(n.c_str()+std::strlen(s));return(int64_t)0;};
    b.encs={{"ffn_gate_up_exps",rad_enc_plain(RAD_BF16)},{"ffn_down_exps",rad_enc_plain(RAD_BF16)}};
    if(absent)b.undeclared.insert(f::kNames[f::Quant]);
    auto m=meta();auto c=spec_ctx(rank,4,0);c.max_tok=32;c.max_seqs=8;
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
int alias_kind(const std::string& s){for(int k=0;k<f::OpCount;++k)if(s==f::kNames[k])return k;return -1;}
void normalize(RadBuilder& b,std::vector<std::unique_ptr<RadCtx>>& v) {
    for(auto&ctx:v)for(auto&i:ctx->issues)if(alias_kind(b.op_of(i.op)->op)>=0) {
        REQUIRE(!i.opd.empty());CHECK_EQ(i.opd.back().kind,RAD_OPK_NONE);i.opd.pop_back();
    }
    for(auto&o:b.ops)if(alias_kind(o.op)>=0)o.op.erase(o.op.find("_fp4_classify"),std::strlen("_fp4_classify"));
}
}

TEST(ds41_fp4_runtime_arch_flag0_has_identical_complete_graph_and_issue_hash) {
    for(int rank:{0,1,2,3}) {
        uint64_t graph=0;std::map<int,uint64_t> steps;
        {Env e(nullptr);RadBuilder b;REQUIRE_EQ(declare(b,rank),RAD_OK);graph=graph_hash(b);
            for(int rows:{1,8,17}){auto k=batch(b,rows);auto v=run_passes(b,deepseek41::step,{k.get()},rank,4);
                REQUIRE(steps_ok(v));steps[rows]=issue_hash(b,v,{k.get()});}}
        {Env e("0","0");RadBuilder b;REQUIRE_EQ(declare(b,rank),RAD_OK);CHECK_EQ(graph_hash(b),graph);
            for(const auto&o:b.ops)CHECK(alias_kind(o.op)<0);
            for(const auto&x:b.bufs)CHECK(x.name!=f::kDevice&&x.name!=f::kHost);
            for(int rows:{1,8,17}){auto k=batch(b,rows);auto v=run_passes(b,deepseek41::step,{k.get()},rank,4);
                REQUIRE(steps_ok(v));CHECK_EQ(issue_hash(b,v,{k.get()}),steps[rows]);}}
    }
}
TEST(ds41_fp4_runtime_arch_aliases_only_replace_fp4_and_preserve_all_other_issues) {
    for(bool packed:{false,true})for(int rank:{0,3})for(int rows:{1,8,17}) {
        uint64_t graph=0,issued=0;
        {Env e("0","0",packed);RadBuilder b;REQUIRE_EQ(declare(b,rank),RAD_OK);graph=graph_hash(b);
            auto k=batch(b,rows);auto v=run_passes(b,deepseek41::step,{k.get()},rank,4);REQUIRE(steps_ok(v));
            CHECK_EQ(report_lint(strict_lint(b,*deepseek41::models[rank],v,rows)),size_t(0));
            issued=issue_hash(b,v,{k.get()});}
        {Env e("1","0",packed);RadBuilder b;REQUIRE_EQ(declare(b,rank),RAD_OK);
            for(const auto&x:b.bufs)CHECK(x.name!=f::kDevice&&x.name!=f::kHost);
            auto k=batch(b,rows);auto v=run_passes(b,deepseek41::step,{k.get()},rank,4);REQUIRE(steps_ok(v));
            size_t aliases=0;for(const auto&ctx:v)for(const auto&i:ctx->issues)if(alias_kind(b.op_of(i.op)->op)>=0)++aliases;
            CHECK(aliases>0);normalize(b,v);CHECK_EQ(graph_hash(b),graph);CHECK_EQ(issue_hash(b,v,{k.get()}),issued);
            CHECK_EQ(report_lint(strict_lint(b,*deepseek41::models[rank],v,rows)),size_t(0));}
    }
}
TEST(ds41_fp4_runtime_arch_audit_is_fixed_persistent_and_every_alias_binds_the_owned_plane) {
    for(bool packed:{false,true}){Env e("1","1",packed);RadBuilder b;REQUIRE_EQ(declare(b),RAD_OK);
        const auto&m=*deepseek41::models[0];REQUIRE(m.fp4_audit);REQUIRE(m.fp4_audit_host);
        const auto&d=b.bufs[m.fp4_audit-1];const auto&h=b.bufs[m.fp4_audit_host-1];
        CHECK_EQ(d.name,std::string(f::kDevice));CHECK_EQ(h.name,std::string(f::kHost));
        for(const auto*x:{&d,&h}){CHECK_EQ(x->d.kind,RAD_BUF_PERSIST);CHECK_EQ(x->d.dtype,RAD_I64);
            CHECK_EQ(x->d.rank,1u);CHECK_EQ(x->d.shape[0],int64_t(f::kWords));}
        CHECK_EQ(d.d.domain,RAD_DOMAIN_DEVICE);CHECK_EQ(h.d.domain,RAD_DOMAIN_HOST);
        auto k=batch(b,17);auto v=run_passes(b,deepseek41::step,{k.get()},0,4);REQUIRE(steps_ok(v));
        size_t got=0;for(const auto&ctx:v)for(const auto&i:ctx->issues){const int kind=alias_kind(b.op_of(i.op)->op);if(kind<0)continue;
            const auto&o=i.opd.back();CHECK_EQ(o.kind,RAD_OPK_BUF);CHECK_EQ(o.handle,m.fp4_audit);
            CHECK_EQ(o.offset,int64_t(0));CHECK_EQ(o.rows,int64_t(f::kWords));CHECK_EQ(o.cols,int64_t(0));++got;}
        CHECK(got>0);
    }
}
TEST(ds41_fp4_runtime_arch_bad_configuration_and_missing_alias_refuse) {
    for(const char*s:{"","2","01","true","1 "}){Env e(s);RadBuilder b;CHECK_EQ(declare(b),RAD_E_INVAL);}
    {Env e("0","1");RadBuilder b;CHECK_EQ(declare(b),RAD_E_INVAL);}
    {Env e("1","0");RadBuilder b;CHECK_EQ(declare(b,0,true),RAD_E_UNSUPPORTED);}
}
TEST(ds41_fp4_runtime_arch_unknown_static_sites_keep_original_operation_names) {
    Env e("0","0");RadBuilder b;deepseek41::Model m;m.fp4_classify=true;m.c.rope=64;m.c.eps=1e-6;
    auto x=rad::arch::decl_b(&b,"x",RAD_BF16,{32,512}),y=rad::arch::decl_b(&b,"y",RAD_BF16,{32,512}),cs=rad::arch::decl_b(&b,"cs",RAD_F32,{32,64});
    auto w=rad::arch::decl_w(&b,"gain",RAD_F32,{512},RAD_ACCESS_PER_TOKEN,RAD_SHARD_NONE,rad::arch::grp_model());
    auto q=deepseek41::roundtrip(&b,32,256,32,"fp4_e2m1","ue8m0",x,0,0,true);
    auto r=deepseek41::rope_rt(&b,m,32,2,128,0,32,"fp4_e2m1","ue8m0",x,cs);
    auto n=deepseek41::norm_rope_rt(&b,m,32,w,512,16,"fp4_e2m1","e4m3",x,y,cs);
    REQUIRE(q);REQUIRE(r);REQUIRE(n);
    CHECK_EQ(b.op_of(q)->op,std::string("quant_roundtrip"));
    CHECK_EQ(b.op_of(r)->op,std::string("rope_rows_roundtrip"));
    CHECK_EQ(b.op_of(n)->op,std::string("rmsnorm_rope_roundtrip"));
}
RAD_TEST_MAIN()
