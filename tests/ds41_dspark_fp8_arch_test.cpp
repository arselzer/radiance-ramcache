/* ds41_dspark_fp8_arch_test.cpp -- RADIANCE_EXPERIMENTAL_DS41_DSPARK_FP8: the DSpark drafter's
 * dense linears read from their E4M3 codes (gemm_nt_bf16f8) as ..._DENSE_FP8 reads the trunk's,
 * declared for DeepSeek-V4.1-Flash's geometry at TP4 with its three DSpark stages (128 experts top-3,
 * a block of 5) on the recording builder (arch_fakes.h), no card or model file:
 *   - with both switches every stage's q_a, q_b, kv (query and context pass), wo_a groups and wo_b,
 *     the shared expert's two linears and the context pass's fc go through gemm_nt_bf16f8, counted
 *     on a startup line of the drafter's own while the trunk's line counts what it counted before;
 *   - without ..._DSPARK_FP8, or without ..._DENSE_FP8 (a note), the drafter stays on gemm_nt and
 *     prints no line;
 *   - ..._DENSE_FP8_EXCLUDE keeps a named drafter weight on gemm_nt, counted as left.
 * The load-time exactness of the bank's weights is ds41_dense_fp8_scan's (it reads draft.rad). */
#define RAD_ARCH_NO_EXPORTS 1

#include "arch_fakes.h"

#include "../arch/deepseek41/deepseek41.cpp"

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr int kWorld = 4;

const char* const kVars[] = { "RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8",
                              "RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8_EXCLUDE",
                              "RADIANCE_EXPERIMENTAL_DS41_DSPARK_FP8",
                              "RADIANCE_EXPERIMENTAL_DS41_DSPARK_SHORT_BLOCK",
                              "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN",
                              "RADIANCE_EXPERIMENTAL_DS41_ENGRAM_SPLIT",
                              "RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16" };
struct Env {
    Env(std::initializer_list<std::pair<const char*, const char*>> kv) {
        for (const char* n : kVars) unsetenv(n);
        for (const auto& p : kv) setenv(p.first, p.second, 1);
    }
    ~Env() {
        for (const char* n : kVars) unsetenv(n);
    }
};

/* ds41_prefetch_arch_test's release geometry (GGUF keys), plus the three DSpark stages and the
 * drafter's keys as the 1017 states them, and the release's FP8 statement. */
MetaKV dspark_meta() {
    RadModelMeta base{};
    base.arch_id = "deepseek41";
    base.name = "ds41-flash-dspark-geometry";
    base.n_layers = 40;
    base.n_embd = 5120;
    base.n_head = 64;
    base.n_head_kv = 1;
    base.head_dim = 512;
    base.n_vocab = 129280;
    base.n_expert = 384;
    base.n_expert_used = 6;
    base.n_expert_shared = 1;
    base.rms_eps = 1e-6f;
    base.rope_theta = 10000.f;
    base.rope_scale = 1.f;
    MetaKV k(base);
    std::string ratios;
    for (int l = 0; l < 40; ++l) ratios += (l ? "," : "") + std::to_string(l < 2 ? 0 : l < 20 ? 2 : 1);
    k.set("deepseek41.block_count", "40")
        .set("deepseek41.rope.dimension_count", "64")
        .set("deepseek41.attention.q_lora_rank", "1280")
        .set("deepseek41.attention.output_group_count", "8")
        .set("deepseek41.attention.output_lora_rank", "1024")
        .set("deepseek41.attention.sliding_window", "128")
        .set("deepseek41.expert_feed_forward_length", "2304")
        .set("deepseek41.expert_weights_scale", "1.5")
        .set("deepseek41.expert_weights_norm", "1")
        .set("deepseek41.swiglu_clamp_exp", "10")
        .set("deepseek41.attention.indexer.head_count", "32")
        .set("deepseek41.attention.indexer.key_length", "128")
        .set("deepseek41.attention.indexer.top_k", "512")
        .set("deepseek41.attention.indexer.candidate_source_layer", "20")
        .set("deepseek41.attention.indexer.candidate_top_k_blocks", "2048")
        .set("deepseek41.attention.indexer.candidate_block_size", "8")
        .set("deepseek41.hyper_connection.count", "4")
        .set("deepseek41.hyper_connection.sinkhorn_iterations", "20")
        .set("deepseek41.hyper_connection.epsilon", "1e-6")
        .set("deepseek41.attention.compress_rope_freq_base", "160000")
        .set("deepseek41.attention.compress_ratios", ratios.c_str())
        .set("deepseek41.attention.kv_source_layers", "2,8,14,20")
        .set("deepseek41.attention.indexer.source_layers", "2,8,14,20,24,28,32,36")
        .set("deepseek41.ple.layers", "1,14")
        .set("deepseek41.ple.ngram_size", "4")
        .set("deepseek41.ple.heads_per_ngram", "8")
        .set("deepseek41.embedding_length_per_layer_input", "256")
        .set("deepseek41.engram.pad_token_id", "2")
        .set("deepseek41.ple.head_vocab_sizes", "1000,1000")
        .set("deepseek41.nextn_predict_layers", "3")
        .set("deepseek41.dspark.block_size", "5")
        .set("deepseek41.dspark.expert_count", "128")
        .set("deepseek41.dspark.expert_used_count", "3")
        .set("deepseek41.dspark.markov_rank", "256")
        .set("deepseek41.dspark.markov_top_k", "16")
        .set("deepseek41.dspark.noise_token_id", "128799")
        .set("deepseek41.dspark.target_layers", "37,38,39")
        .set("imported_quantization_config.quant_method", "fp8")
        .set("imported_quantization_config.scale_fmt", "ue8m0")
        .set("imported_quantization_config.weight_block_size", "32,32");
    return k;
}

int64_t block_of(const std::string& n) {
    for (const char* p : { "kv_compressed_r", "kv_index_r" })
        if (n.rfind(p, 0) == 0) return std::atoll(n.c_str() + std::strlen(p));
    return 0;
}

struct Decl {
    RadBuilder b;
    int rc = RAD_E_INVAL;
    std::string err;
    deepseek41::Model* m = nullptr;
};

/* deepseek41's declare for `rank` of four with the drafter (a speculative serve of 5, one
 * sequence, 16 rows a step), the dense linears stored as plain bf16 as the container and the bank
 * store them. */
std::unique_ptr<Decl> declare(int rank) {
    auto d = std::make_unique<Decl>();
    d->b.kv_block_of = block_of;
    MetaKV meta = dspark_meta();
    const RadEncoding plain = rad_enc_plain(RAD_BF16);
    for (const char* n : { "attn_q_a.weight", "attn_kv.weight", "attn_q_b.weight", "attn_output_a.",
                           "attn_output_b.weight", "engram_kv.weight", "ffn_gate_up_shexp.weight",
                           "ffn_down_shexp.weight", "dspark.fc.weight" })
        d->b.encs.push_back({ n, plain });
    RadBuildCtx ctx = spec_ctx(rank, kWorld, 5);
    ctx.max_tok = 16;
    ctx.max_seqs = 1;
    d->err = stderr_of([&] { d->rc = deepseek41::declare(&d->b, meta.meta(), &ctx); });
    d->m = deepseek41::models[rank].get();
    return d;
}

const std::string& opname(const RadBuilder& b, rad_op h) {
    static const std::string none = "<none>";
    const RecOp* o = b.op_of(h);
    return o ? o->op : none;
}

/* Every dense linear of the drafter's, query and context pass: their ops' names. */
std::vector<std::string> drafter_linears(const Decl& d) {
    std::vector<std::string> v;
    const auto& ds = d.m->dsp;
    v.push_back(opname(d.b, ds.c_fc));
    for (const auto& L : ds.st) {
        for (rad_op h : { L.q_a, L.q_b, L.kv_g, L.d_kv, L.o_b }) v.push_back(opname(d.b, h));
        for (rad_op h : L.o_a) v.push_back(opname(d.b, h));
    }
    return v;
}

/* The trunk's startup count on this rank's line, -1 without one. */
int trunk_count(const std::string& err, int rank) {
    const std::string k = "deepseek41: rank " + std::to_string(rank) + ": ";
    size_t at = 0;
    while ((at = err.find(k, at)) != std::string::npos) {
        at += k.size();
        if (err.compare(at, 14, "DSpark drafter") == 0) continue;
        const int n = std::atoi(err.c_str() + at);
        if (err.find(" dense weights read by gemm_nt_bf16f8 (RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8)", at) !=
            std::string::npos)
            return n;
    }
    return -1;
}

}  // namespace

TEST(ds41_dspark_fp8_reads_the_drafters_dense_linears_from_codes) {
    int trunk_alone[kWorld] = {};
    {
        Env env({ { "RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8", "1" } });
        for (int rank = 0; rank < kWorld; ++rank) {
            auto d = declare(rank);
            REQUIRE_EQ(d->rc, (int)RAD_OK);
            REQUIRE(d->m->dsp.on);
            REQUIRE(d->m->f8_on);
            CHECK(!d->m->dsp_f8);
            REQUIRE_EQ(d->m->dsp.st.size(), (size_t)3);
            for (const std::string& op : drafter_linears(*d)) CHECK_EQ(op, std::string("gemm_nt"));
            for (const auto& L : d->m->dsp.st) {
                CHECK(!L.moe.sh_gate_up.bf16_alt);
                CHECK(!L.moe.sh_down.bf16_alt);
            }
            CHECK(d->err.find("DSpark drafter:") == std::string::npos);
            trunk_alone[rank] = trunk_count(d->err, rank);
            CHECK(trunk_alone[rank] > 0);
        }
    }
    Env env({ { "RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8", "1" },
              { "RADIANCE_EXPERIMENTAL_DS41_DSPARK_FP8", "1" } });
    for (int rank = 0; rank < kWorld; ++rank) {
        auto d = declare(rank);
        REQUIRE_EQ(d->rc, (int)RAD_OK);
        REQUIRE(d->m->dsp_f8);
        const auto ops = drafter_linears(*d);
        /* fc, and per stage q_a, q_b, kv twice, wo_b and the rank's wo_a groups. */
        CHECK_EQ(ops.size(), (size_t)(1 + 3 * (5 + d->m->gr)));
        for (const std::string& op : ops) CHECK_EQ(op, std::string(deepseek41::kDenseF8));
        for (const auto& L : d->m->dsp.st) {
            CHECK(L.moe.sh_gate_up.bf16_alt);
            CHECK(L.moe.sh_down.bf16_alt);
        }
        /* Weights, not ops: fc, and per stage q_a, q_b, kv, wo_b, the groups and the shared pair. */
        CHECK_EQ(d->m->dsp_f8_n, (int)(1 + 3 * (4 + d->m->gr + 2)));
        CHECK_EQ(d->m->dsp_f8_kept, 0);
        CHECK(d->m->dsp_f8_codes > 0);
        const std::string line = "deepseek41: rank " + std::to_string(rank) + ": DSpark drafter: " +
                                 std::to_string(d->m->dsp_f8_n) + " dense weights read by gemm_nt_bf16f8";
        CHECK(d->err.find(line) != std::string::npos);
        /* The trunk's line counts the trunk's alone. */
        CHECK_EQ(trunk_count(d->err, rank), trunk_alone[rank]);
        std::fprintf(stderr, "    rank %d: %d drafter weights, %.1f MiB of codes (bf16 %.1f MiB)\n", rank,
                     d->m->dsp_f8_n, (double)d->m->dsp_f8_codes / 1048576.0,
                     2.0 * (double)d->m->dsp_f8_codes / 1048576.0);
    }
}

TEST(ds41_dspark_fp8_needs_the_trunks_switch) {
    Env env({ { "RADIANCE_EXPERIMENTAL_DS41_DSPARK_FP8", "1" } });
    auto d = declare(0);
    REQUIRE_EQ(d->rc, (int)RAD_OK);
    CHECK(!d->m->f8_on);
    CHECK(!d->m->dsp_f8);
    for (const std::string& op : drafter_linears(*d)) CHECK_EQ(op, std::string("gemm_nt"));
    CHECK(d->err.find("RADIANCE_EXPERIMENTAL_DS41_DSPARK_FP8 is set, but the trunk's dense FP8 is off") !=
          std::string::npos);
    CHECK(d->err.find("DSpark drafter:") == std::string::npos);
}

TEST(ds41_dspark_fp8_honours_the_exclusions) {
    Env env({ { "RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8", "1" },
              { "RADIANCE_EXPERIMENTAL_DS41_DSPARK_FP8", "1" },
              { "RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8_EXCLUDE", "dspark.fc.weight,blk.41.attn_kv.weight" } });
    auto d = declare(1);
    REQUIRE_EQ(d->rc, (int)RAD_OK);
    REQUIRE(d->m->dsp_f8);
    const auto& ds = d->m->dsp;
    CHECK_EQ(opname(d->b, ds.c_fc), std::string("gemm_nt"));
    CHECK_EQ(opname(d->b, ds.st[1].kv_g), std::string("gemm_nt"));
    CHECK_EQ(opname(d->b, ds.st[1].d_kv), std::string("gemm_nt"));
    CHECK_EQ(opname(d->b, ds.st[0].kv_g), std::string(deepseek41::kDenseF8));
    CHECK_EQ(opname(d->b, ds.st[2].q_b), std::string(deepseek41::kDenseF8));
    CHECK_EQ(d->m->dsp_f8_kept, 2);
    CHECK_EQ(d->m->dsp_f8_n, (int)(1 + 3 * (4 + d->m->gr + 2) - 2));
}

/* With K1's short block declared too the drafter declares the same FP8 reads (the short block's
 * ops are its attention, offsets and walk, no linear of its own). */
TEST(ds41_dspark_fp8_with_the_short_block) {
    Env env({ { "RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8", "1" },
              { "RADIANCE_EXPERIMENTAL_DS41_DSPARK_FP8", "1" },
              { "RADIANCE_EXPERIMENTAL_DS41_DSPARK_SHORT_BLOCK", "2" } });
    auto d = declare(2);
    REQUIRE_EQ(d->rc, (int)RAD_OK);
    CHECK_EQ(d->m->dsp.block_lo, (int64_t)2);
    CHECK_EQ(d->b.drafter_short, (int64_t)2);
    for (const std::string& op : drafter_linears(*d)) CHECK_EQ(op, std::string(deepseek41::kDenseF8));
    CHECK_EQ(d->m->dsp_f8_n, (int)(1 + 3 * (4 + d->m->gr + 2)));
}

RAD_TEST_MAIN()
