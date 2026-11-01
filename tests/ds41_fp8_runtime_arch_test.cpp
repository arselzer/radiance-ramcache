#define RAD_ARCH_NO_EXPORTS 1
#include "arch_fakes.h"
#include "../arch/deepseek41/deepseek41.cpp"
namespace f=rad_ds41_fp8_runtime;
namespace {
struct Env {
    Env(const char*on,const char*audit=nullptr){unsetenv(f::kEnv);unsetenv(f::kAuditEnv);
        if(on)setenv(f::kEnv,on,1);if(audit)setenv(f::kAuditEnv,audit,1);}
    ~Env(){unsetenv(f::kEnv);unsetenv(f::kAuditEnv);}
};
MetaKV native_meta(){RadModelMeta b{};b.arch_id="deepseek41";b.name="ds41-fp8-native";b.n_layers=40;
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
int declare(RadBuilder&b,int rank=0,bool missing=false){
    b.kv_block_of=[](const std::string&n){for(const char*s:{"kv_compressed_r","kv_index_r"})
        if(n.rfind(s,0)==0)return(int64_t)std::atoll(n.c_str()+std::strlen(s));return(int64_t)0;};
    b.encs={{"ffn_gate_up_exps",rad_enc_plain(RAD_BF16)},{"ffn_down_exps",rad_enc_plain(RAD_BF16)}};
    if(missing)b.undeclared.insert(f::kNames[f::Rope]);
    auto meta=native_meta();auto c=spec_ctx(rank,4,0);c.max_tok=32;c.max_seqs=8;
    return stderr_of_rc([&]{return deepseek41::declare(&b,meta.meta(),&c);});
}
int kind(const std::string&s){for(int k=0;k<f::OpCount;++k)if(s==f::kNames[k])return k;return -1;}
std::unique_ptr<Kit> batch(const RadBuilder&b,int rows){return rows<=8?kit(b,std::vector<int>((size_t)rows,1),{}):kit(b,{}, {rows});}
/* Fixture-local handling of the exact native grouped output-A row-pitched views. The real
 * gemm backend accepts these via pitched_rows()/gemm_ops(pitched=1); all other lint is fatal. */
std::vector<std::string> strict_lint(const RadBuilder&b,const deepseek41::Model&m,
                                    const std::vector<std::unique_ptr<RadCtx>>&v,int64_t rows){
    auto bad=lint_step(b,view(v));std::set<std::string> supported;std::map<rad_op,size_t> found;
    for(const auto&L:m.layers)for(auto h:L.o_a)found.emplace(h,0);
    for(size_t p=0;p<v.size();++p)for(size_t z=0;z<v[p]->issues.size();++z){const auto&i=v[p]->issues[z];
        for(const auto&L:m.layers)for(int j=0;j<m.gr;++j)if(i.op==L.o_a[(size_t)j]){
            ++found[i.op];bool valid=m.gr==2&&m.g.n_head==16&&m.c.head_dim==512&&m.c.o_rank==1024&&i.n==rows&&i.opd.size()==3;
            const auto*op=b.op_of(i.op);valid=valid&&op&&op->op=="gemm_nt"&&pint(op,"K")==4096&&pint(op,"N")==1024;
            if(valid){const auto&a=i.opd[0];const auto&w=i.opd[1];const auto&y=i.opd[2];
                const auto&ad=b.bufs[m.o-1].d;const auto&yd=b.bufs[m.oa-1].d;
                valid=a.kind==RAD_OPK_BUF&&a.handle==m.o&&a.offset==j*4096&&a.rows==rows&&a.cols==4096&&
                  w.kind==RAD_OPK_WEIGHT&&w.handle==L.wo_a[(size_t)j]&&!w.offset&&!w.rows&&!w.cols&&
                  y.kind==RAD_OPK_BUF&&y.handle==m.oa&&y.offset==j*1024&&y.rows==rows&&y.cols==1024&&
                  ad.dtype==RAD_BF16&&yd.dtype==RAD_BF16&&ad.rank==2&&yd.rank==2&&ad.shape[0]>=rows&&yd.shape[0]>=rows&&
                  ad.shape[1]==8192&&yd.shape[1]==2048;}
            if(!valid){bad.push_back("invalid native grouped output view");continue;}
            char at[160];std::snprintf(at,sizeof at,"pass %zu issue %zu (gemm_nt, op %u) operand 0: ",p,z,(unsigned)i.op);
            supported.insert(std::string(at)+"gemm_nt reads 4096 of 8192 columns of 'attn_o' as its input, which is not contiguous");
        }}
    if(supported.size()!=m.layers.size()*2)bad.push_back("missing native grouped views");
    for(const auto&[op,n]:found)if(n!=1)bad.push_back("missing/duplicate native grouped output op");
    bad.erase(std::remove_if(bad.begin(),bad.end(),[&](const std::string&s){return supported.count(s)!=0;}),bad.end());return bad;
}
}
TEST(ds41_fp8_runtime_arch_unset_zero_graph_and_issue_identity) {
    for(int rank:{0,1,2,3}){uint64_t graph;std::map<int,uint64_t>steps;
        {Env e(nullptr);RadBuilder b;REQUIRE_EQ(declare(b,rank),RAD_OK);graph=graph_hash(b);
            for(int rows:{1,4,8,17}){auto k=batch(b,rows);auto v=run_passes(b,deepseek41::step,{k.get()},rank,4);
                REQUIRE(steps_ok(v));steps[rows]=issue_hash(b,v,{k.get()});}}
        {Env e("0","0");RadBuilder b;REQUIRE_EQ(declare(b,rank),RAD_OK);CHECK_EQ(graph_hash(b),graph);
            for(const auto&o:b.ops)CHECK(kind(o.op)<0);
            for(const auto&x:b.bufs)CHECK(x.name!=f::kDevice&&x.name!=f::kHost);
            for(int rows:{1,4,8,17}){auto k=batch(b,rows);auto v=run_passes(b,deepseek41::step,{k.get()},rank,4);
                REQUIRE(steps_ok(v));CHECK_EQ(issue_hash(b,v,{k.get()}),steps[rows]);}}
    }
}
TEST(ds41_fp8_runtime_arch_native_trunk_only_conversion_alias_changes) {
    for(int rank:{0,3})for(int rows:{1,4,8,17}){uint64_t graph,issued;
        {Env e("0","0");RadBuilder b;REQUIRE_EQ(declare(b,rank),RAD_OK);graph=graph_hash(b);
            auto k=batch(b,rows);auto v=run_passes(b,deepseek41::step,{k.get()},rank,4);REQUIRE(steps_ok(v));
            CHECK_EQ(report_lint(strict_lint(b,*deepseek41::models[rank],v,rows)),size_t(0));issued=issue_hash(b,v,{k.get()});}
        {Env e("1","0");RadBuilder b;REQUIRE_EQ(declare(b,rank),RAD_OK);
            auto k=batch(b,rows);auto v=run_passes(b,deepseek41::step,{k.get()},rank,4);REQUIRE(steps_ok(v));
            size_t rope=0;for(const auto&ctx:v)for(const auto&i:ctx->issues)if(kind(b.op_of(i.op)->op)==f::Rope)++rope;
            CHECK_EQ(rope,size_t(40));for(auto&o:b.ops)if(kind(o.op)>=0)o.op.erase(o.op.find("_fp8_convert"));
            CHECK_EQ(graph_hash(b),graph);CHECK_EQ(issue_hash(b,v,{k.get()}),issued);
            CHECK_EQ(report_lint(strict_lint(b,*deepseek41::models[rank],v,rows)),size_t(0));}
    }
}
TEST(ds41_fp8_runtime_arch_audit_is_owned_fixed_persistent_and_alias_only) {
    Env e("1","1");RadBuilder b;REQUIRE_EQ(declare(b),RAD_OK);const auto&m=*deepseek41::models[0];
    REQUIRE(m.fp8_audit);REQUIRE(m.fp8_audit_host);
    for(auto h:{m.fp8_audit,m.fp8_audit_host}){const auto&d=b.bufs[h-1].d;
        CHECK_EQ(d.kind,RAD_BUF_PERSIST);CHECK_EQ(d.dtype,RAD_I64);CHECK_EQ(d.rank,1u);CHECK_EQ(d.shape[0],int64_t(f::kWords));}
    CHECK_EQ(b.bufs[m.fp8_audit-1].d.domain,RAD_DOMAIN_DEVICE);CHECK_EQ(b.bufs[m.fp8_audit_host-1].d.domain,RAD_DOMAIN_HOST);
    auto k=batch(b,17);auto v=run_passes(b,deepseek41::step,{k.get()},0,4);REQUIRE(steps_ok(v));size_t got=0;
    for(const auto&ctx:v)for(const auto&i:ctx->issues)if(kind(b.op_of(i.op)->op)>=0){const auto&o=i.opd.back();
        CHECK_EQ(o.kind,RAD_OPK_BUF);CHECK_EQ(o.handle,m.fp8_audit);CHECK_EQ(o.rows,int64_t(f::kWords));CHECK_EQ(o.offset,int64_t(0));++got;}
    CHECK_EQ(got,size_t(40));
}
TEST(ds41_fp8_runtime_arch_refuses_bad_flag_or_missing_alias_and_preserves_unknown_whole_ops) {
    for(const char*s:{"","2","01","true"}){Env e(s);RadBuilder b;CHECK_EQ(declare(b),RAD_E_INVAL);}
    {Env e("0","1");RadBuilder b;CHECK_EQ(declare(b),RAD_E_INVAL);}
    {Env e("1","0");RadBuilder b;CHECK_EQ(declare(b,0,true),RAD_E_UNSUPPORTED);}
    Env e("0","0");RadBuilder b;deepseek41::Model m;m.fp8_convert=true;m.c.rope=64;
    auto x=rad::arch::decl_b(&b,"x",RAD_BF16,{32,8192}),cs=rad::arch::decl_b(&b,"cs",RAD_F32,{32,64});
    auto q=deepseek41::fp8_roundtrip(&b,m,32,4096,32,"fp8_e4m3","ue8m0",x);
    auto r=deepseek41::fp8_rope_rt(&b,m,32,8,512,1,32,"fp8_e4m3","ue8m0",x,cs);
    REQUIRE(q);REQUIRE(r);CHECK_EQ(b.op_of(q)->op,std::string("quant_roundtrip"));CHECK_EQ(b.op_of(r)->op,std::string("rope_rows_roundtrip"));
}
RAD_TEST_MAIN()
