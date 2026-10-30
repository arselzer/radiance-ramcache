/* ds41_prefetch_arch_test.cpp -- deepseek41's on-die reads (RADIANCE_EXPERIMENTAL_DS41_PREFETCH_*,
 * Layer::pf) declared and issued for DeepSeek-V4.1-Flash's own geometry at TP4 (40 layers, d 5120,
 * 64 heads of 512, q_lora 1280, eight output groups of 1024, the 2304-wide shared expert split
 * 768/512/512/512, compressors at ratio 2 then 1, Engram at layers 1 and 14), on the recording
 * builder (arch_fakes.h), no card or model file:
 *   - the switch off declares and issues no read;
 *   - the switch alone declares 16be66c5's reads: q_a, kv, q_b and wo_a's two groups whole, every
 *     layer but 0 and the two Engram layers, at the same bytes;
 *   - a budget fills the use-order list per layer: the compressor and the indexer where the layer
 *     has them, wo_b, the router and the shared expert's slice, the read that crosses it cut;
 *   - ..._ENGRAM reads an Engram layer's engram_kv from the layer before and the rest behind its
 *     GEMM;
 *   - a decode step issues each layer's reads on lane 1 behind a join, after the shared expert's
 *     down and before the gather (or on lane 0 where launches chain), an Engram layer's between
 *     its GEMM and the all_gather; a prompt step issues none;
 *   - with ..._ROUTER_BF16 the router and the ratio-2 compressors' kv|gate go through
 *     gemm_nt_f32bf16 at their planes' dtypes, are counted on rank 0's startup line, and are read
 *     at their bf16 bytes; refused where no library declares the op.
 * The pure plan (parse, fit) is ds41_prefetch_plan_test's. */
#define RAD_ARCH_NO_EXPORTS 1

#include "arch_fakes.h"

#include "../arch/deepseek41/deepseek41.cpp"

#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

namespace {

constexpr int64_t MiB = 1ll << 20;
constexpr int kWorld = 4;

const char* const kVars[] = { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN",
                              "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MIB",
                              "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_SET",
                              "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ENGRAM",
                              "RADIANCE_EXPERIMENTAL_DS41_ENGRAM_SPLIT",
                              "RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8",
                              "RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16" };
/* The variables a case sets, every other one unset, for the case's declares. */
struct Env {
    Env(std::initializer_list<std::pair<const char*, const char*>> kv) {
        for (const char* n : kVars) unsetenv(n);
        for (const auto& p : kv) setenv(p.first, p.second, 1);
    }
    ~Env() {
        for (const char* n : kVars) unsetenv(n);
    }
};

/* The release's geometry as llama.cpp's converter states it (GGUF keys), with a token Engram
 * table: its size is no part of what is checked here. */
MetaKV flash_meta() {
    RadModelMeta base{};
    base.arch_id = "deepseek41";
    base.name = "ds41-flash-geometry";
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
        .set("deepseek41.ple.head_vocab_sizes", "1000,1000");
    return k;
}

/* The compressed groups' blocks are their ratio (entry_store fixes them; deepseek41's declare
 * checks it), every other group's the fake's 16. */
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
/* deepseek41's declare for `rank` of four, decode-sized (16 rows a step), under the current
 * environment; its stderr (the plan line) kept in `err`. `fp8`: the checkpoint states its dense
 * linears are E4M3 with an E8M0 scale a 32x32 block and stores them as plain bf16, as the release's
 * container does, so ..._DENSE_FP8=1 takes them (dense_f8_gate, lw8). `setup`: anything else the
 * builder should say (more plane encodings, ops no library declares), before the declare. */
std::unique_ptr<Decl> declare(int rank, bool fp8 = false,
                              const std::function<void(RadBuilder&)>& setup = {}) {
    auto d = std::make_unique<Decl>();
    d->b.kv_block_of = block_of;
    MetaKV meta = flash_meta();
    if (fp8) {
        meta.set("imported_quantization_config.quant_method", "fp8")
            .set("imported_quantization_config.scale_fmt", "ue8m0")
            .set("imported_quantization_config.weight_block_size", "32,32");
        const RadEncoding plain = rad_enc_plain(RAD_BF16);
        for (const char* n : { "attn_q_a.weight", "attn_kv.weight", "attn_q_b.weight", "attn_output_a.",
                               "attn_output_b.weight", "engram_kv.weight", "ffn_gate_up_shexp.weight",
                               "ffn_down_shexp.weight" })
            d->b.encs.push_back({ n, plain });
    }
    if (setup) setup(d->b);
    RadBuildCtx ctx = build_ctx(rank, kWorld);
    ctx.max_tok = 16;
    d->err = stderr_of([&] { d->rc = deepseek41::declare(&d->b, meta.meta(), &ctx); });
    d->m = deepseek41::models[rank].get();
    return d;
}

std::string wname(const RadBuilder& b, rad_weight w) {
    return w && w <= b.weights.size() ? b.weights[w - 1].name : std::string("<none>");
}
std::vector<int> kinds(const std::vector<deepseek41::PfRead>& v) {
    std::vector<int> k;
    for (const auto& r : v) k.push_back(r.kind);
    return k;
}
int64_t total(const std::vector<deepseek41::PfRead>& v) {
    int64_t s = 0;
    for (const auto& r : v) s += r.bytes;
    return s;
}
/* Each read is a declared `prefetch` of its weight at its bytes. */
bool reads_declared(const RadBuilder& b, const std::vector<deepseek41::PfRead>& v) {
    for (const auto& r : v) {
        const RecOp* o = b.op_of(r.op);
        const RecParam* p = param(o, "bytes");
        if (!o || o->op != "prefetch" || o->w.size() != 1 || o->w[0] != r.w || !p || p->ival != r.bytes)
            return false;
    }
    return true;
}

/* Whether `v` reads `want`'s kinds in order; on a mismatch the reads are printed. */
bool reads_are(const std::vector<deepseek41::PfRead>& v, const std::vector<int>& want) {
    const std::vector<int> k = kinds(v);
    if (k == want) return true;
    std::string s;
    for (const int x : k) s += std::string(s.empty() ? "" : ",") + deepseek41::pf::name(x);
    fprintf(stderr, "    reads: %s\n", s.empty() ? "(none)" : s.c_str());
    return false;
}

using pf_kinds = std::vector<int>;
namespace pk = deepseek41::pf;

}  // namespace

TEST(ds41_prefetch_off_declares_and_issues_nothing) {
    Env env({});
    /* The geometry carries no DSpark stages, so no drafter and no draft depth. */
    MetaKV meta = flash_meta();
    RadArchProbe pb{};
    CHECK_EQ(deepseek41::probe(meta.meta(), &pb), (int)RAD_OK);
    CHECK_EQ(pb.draft_depth, 0);
    auto d = declare(0);
    REQUIRE_EQ(d->rc, (int)RAD_OK);
    REQUIRE(d->m);
    CHECK(!d->m->dsp.on);
    CHECK_EQ(count_op(d->b, "prefetch"), 0);
    for (const auto& L : d->m->layers) CHECK(L.pf.empty() && L.pf_post.empty());
    CHECK(d->err.find("on-die reads") == std::string::npos);
    auto k = kit(d->b, { 1 }, {});
    auto run = run_passes(d->b, deepseek41::step, { k.get() }, 0, kWorld);
    CHECK(steps_ok(run));
    CHECK_EQ(count_issued(issued_ops(d->b, *run[0]), "prefetch"), 0);
    CHECK_EQ(d->m->pfs.reads, (uint64_t)0);
}

TEST(ds41_prefetch_switch_alone_declares_the_measured_reads) {
    Env env({ { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN", "1" } });
    for (int rank = 0; rank < kWorld; ++rank) {
        auto d = declare(rank);
        REQUIRE_EQ(d->rc, (int)RAD_OK);
        const auto& L = d->m->layers;
        REQUIRE_EQ(L.size(), (size_t)40);
        int n = 0;
        for (int l = 0; l < 40; ++l) {
            CHECK(L[l].pf_post.empty());
            if (l == 0 || l == 1 || l == 14) {
                CHECK(L[l].pf.empty());
                continue;
            }
            /* 16be66c5's four, each whole: q_a [1280, 5120], kv [512, 5120], this rank's q_b
             * [16 x 512, 1280], its two wo_a groups [1024, 4096]. */
            REQUIRE(reads_are(L[l].pf, pf_kinds{ pk::kQa, pk::kKv, pk::kQb, pk::kOa, pk::kOa }));
            const auto& r = L[l].pf;
            CHECK_EQ(r[0].bytes, (int64_t)1280 * 5120 * 2);
            CHECK_EQ(r[1].bytes, (int64_t)512 * 5120 * 2);
            CHECK_EQ(r[2].bytes, (int64_t)8192 * 1280 * 2);
            CHECK_EQ(r[3].bytes, (int64_t)1024 * 4096 * 2);
            CHECK_EQ(r[4].bytes, (int64_t)1024 * 4096 * 2);
            for (const auto& x : r) CHECK(!x.cut);
            const std::string p = "blk." + std::to_string(l) + ".";
            CHECK_EQ(wname(d->b, r[0].w), p + "attn_q_a.weight");
            CHECK_EQ(wname(d->b, r[1].w), p + "attn_kv.weight");
            CHECK_EQ(wname(d->b, r[2].w), p + "attn_q_b.weight");
            CHECK_EQ(wname(d->b, r[3].w), p + "attn_output_a.g" + std::to_string(2 * rank) + ".weight");
            CHECK_EQ(wname(d->b, r[4].w), p + "attn_output_a.g" + std::to_string(2 * rank + 1) + ".weight");
            CHECK(reads_declared(d->b, r));
            n += (int)r.size();
        }
        CHECK_EQ(n, 37 * 5);
        CHECK_EQ(count_op(d->b, "prefetch"), n);
        CHECK(d->err.find("rank " + std::to_string(rank) + ": on-die reads") != std::string::npos);
        CHECK(d->err.find("qa,kv,qb,oa whole (the default); 37 layers read ahead, 5-5 reads, 53.5 MiB") !=
              std::string::npos);
    }
}

TEST(ds41_prefetch_budget_fills_the_use_order) {
    Env env({ { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN", "1" },
              { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MIB", "56" } });
    auto d = declare(0);
    REQUIRE_EQ(d->rc, (int)RAD_OK);
    const auto& L = d->m->layers;
    /* Layer 2, a kv and index source at ratio 2: the f32 compressor [1024, 5120] first, then q_a,
     * kv, the indexer's q_b [4096, 1280], and q_b cut to the 8.5 MiB left. */
    REQUIRE(reads_are(L[2].pf, pf_kinds{ pk::kComp, pk::kQa, pk::kKv, pk::kIq, pk::kQb }));
    CHECK_EQ(L[2].pf[0].bytes, (int64_t)1024 * 5120 * 4);
    CHECK_EQ(L[2].pf[3].bytes, (int64_t)4096 * 1280 * 2);
    CHECK(L[2].pf[4].cut);
    CHECK_EQ(L[2].pf[4].bytes, 17 * MiB / 2);
    CHECK_EQ(total(L[2].pf), 56 * MiB);
    CHECK_EQ(wname(d->b, L[2].pf[0].w), std::string("blk.2.attn_compressor_kvg.weight"));
    CHECK_EQ(wname(d->b, L[2].pf[3].w), std::string("blk.2.indexer.attn_q_b.weight"));
    /* Layer 3, no source: the four whole and wo_b's first 2.5 MiB. */
    REQUIRE(reads_are(L[3].pf, pf_kinds{ pk::kQa, pk::kKv, pk::kQb, pk::kOa, pk::kOa, pk::kOb }));
    CHECK(L[3].pf[5].cut);
    CHECK_EQ(L[3].pf[5].bytes, 5 * MiB / 2);
    CHECK_EQ(wname(d->b, L[3].pf[5].w), std::string("blk.3.attn_output_b.weight"));
    /* Layer 20, the ratio-1 kv source: its compressor is bf16 [512, 5120]; wo_a's first group cut. */
    REQUIRE(reads_are(L[20].pf, pf_kinds{ pk::kComp, pk::kQa, pk::kKv, pk::kIq, pk::kQb, pk::kOa }));
    CHECK_EQ(L[20].pf[0].bytes, (int64_t)512 * 5120 * 2);
    CHECK_EQ(L[20].pf[5].bytes, 7 * MiB / 2);
    /* Layer 24, an index source only: 55.5 MiB whole leaves 0.5 MiB, under the cut floor. */
    REQUIRE(reads_are(L[24].pf, pf_kinds{ pk::kQa, pk::kKv, pk::kIq, pk::kQb, pk::kOa }));
    CHECK_EQ(total(L[24].pf), 111 * MiB / 2);
    for (const auto& x : L[24].pf) CHECK(!x.cut);
    /* Nothing for layer 0 or the Engram layers without ..._ENGRAM. */
    CHECK(L[0].pf.empty() && L[1].pf.empty() && L[14].pf.empty());
    CHECK(L[1].pf_post.empty() && L[14].pf_post.empty());
    for (const auto& x : L) CHECK(reads_declared(d->b, x.pf));
    CHECK(d->err.find("a budget of 56.0 MiB an issue point over "
                      "ekv,comp,qa,kv,iq,qb,oa,ob,rt,sgu,sdn") != std::string::npos);
}

TEST(ds41_prefetch_set_reaches_the_router_and_the_shared_slice) {
    Env env({ { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN", "1" },
              { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_SET", "rt,sgu,sdn,ma,mf,iw,ik" } });
    for (int rank = 0; rank < kWorld; ++rank) {
        auto d = declare(rank);
        REQUIRE_EQ(d->rc, (int)RAD_OK);
        const auto& L = d->m->layers;
        /* The router [384, 5120] f32; this rank's shared slice, 768 columns on rank 0 and 512 on
         * the others (shared_slice: nine 256-column blocks); the two f32 mixes [24, 4 x 5120]. */
        const int64_t w = rank == 0 ? 768 : 512;
        REQUIRE(reads_are(L[3].pf, pf_kinds{ pk::kRt, pk::kSgu, pk::kSdn, pk::kMa, pk::kMf }));
        CHECK_EQ(L[3].pf[0].bytes, (int64_t)384 * 5120 * 4);
        CHECK_EQ(L[3].pf[1].bytes, 2 * w * 5120 * 2);
        CHECK_EQ(L[3].pf[2].bytes, 5120 * w * 2);
        CHECK_EQ(L[3].pf[3].bytes, (int64_t)24 * 20480 * 4);
        CHECK_EQ(wname(d->b, L[3].pf[0].w), std::string("blk.3.glm_router.weight"));
        CHECK_EQ(wname(d->b, L[3].pf[1].w), std::string("blk.3.ffn_gate_up_shexp.weight"));
        CHECK_EQ(wname(d->b, L[3].pf[2].w), std::string("blk.3.ffn_down_shexp.weight"));
        CHECK_EQ(wname(d->b, L[3].pf[3].w), std::string("blk.3.hc_attn_fn.weight"));
        CHECK_EQ(wname(d->b, L[3].pf[4].w), std::string("blk.3.hc_ffn_fn.weight"));
        /* Layer 2 adds the indexer's two small projections, in the order named. */
        REQUIRE(reads_are(L[2].pf,
                          pf_kinds{ pk::kRt, pk::kSgu, pk::kSdn, pk::kMa, pk::kMf, pk::kIw, pk::kIk }));
        CHECK_EQ(L[2].pf[5].bytes, (int64_t)32 * 5120 * 2);
        CHECK_EQ(L[2].pf[6].bytes, (int64_t)128 * 512 * 2);
        for (const auto& x : L) CHECK(reads_declared(d->b, x.pf));
    }
}

TEST(ds41_prefetch_engram_layers_read_around_their_gemm) {
    {
        Env env({ { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN", "1" },
                  { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MIB", "56" },
                  { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ENGRAM", "1" },
                  { "RADIANCE_EXPERIMENTAL_DS41_ENGRAM_SPLIT", "1" } });
        auto d = declare(1);
        REQUIRE_EQ(d->rc, (int)RAD_OK);
        const auto& L = d->m->layers;
        /* engram_kv split: this rank's 6400 rows of 6144, 75 MiB, read by the layer before up to
         * the budget. */
        for (int l : { 1, 14 }) {
            REQUIRE(reads_are(L[l].pf, pf_kinds{ pk::kEkv }));
            CHECK(L[l].pf[0].cut);
            CHECK_EQ(L[l].pf[0].bytes, 56 * MiB);
            CHECK_EQ(wname(d->b, L[l].pf[0].w), "blk." + std::to_string(l) + ".engram_kv.weight");
        }
        REQUIRE(reads_are(L[1].pf_post,
                          pf_kinds{ pk::kQa, pk::kKv, pk::kQb, pk::kOa, pk::kOa, pk::kOb }));
        CHECK_EQ(total(L[1].pf_post), 56 * MiB);
        REQUIRE(reads_are(L[14].pf_post, pf_kinds{ pk::kComp, pk::kQa, pk::kKv, pk::kIq, pk::kQb }));
        CHECK_EQ(total(L[14].pf_post), 56 * MiB);
        CHECK(reads_declared(d->b, L[1].pf) && reads_declared(d->b, L[1].pf_post));
        CHECK(reads_declared(d->b, L[14].pf) && reads_declared(d->b, L[14].pf_post));
        CHECK(d->err.find("Engram layers (MiB ahead of the GEMM + behind it) 1 56.0 + 56.0, "
                          "14 56.0 + 56.0") != std::string::npos);
    }
    {
        /* The default list has no ekv: nothing ahead of the GEMM, the four behind it. Replicated
         * engram_kv here ([25600, 6144]), no all_gather. */
        Env env({ { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN", "1" },
                  { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ENGRAM", "1" } });
        auto d = declare(0);
        REQUIRE_EQ(d->rc, (int)RAD_OK);
        const auto& L = d->m->layers;
        CHECK(L[1].pf.empty() && L[14].pf.empty());
        CHECK(reads_are(L[1].pf_post, pf_kinds{ pk::kQa, pk::kKv, pk::kQb, pk::kOa, pk::kOa }));
        CHECK(reads_are(L[14].pf_post, pf_kinds{ pk::kQa, pk::kKv, pk::kQb, pk::kOa, pk::kOa }));
    }
}

/* With the dense linears read from their E4M3 codes (..._DENSE_FP8=1) every read but the f32
 * compressor's and router's is half the bytes, so the same 56 MiB reach the shared expert's down. */
TEST(ds41_prefetch_fp8_codes_fit_the_whole_list) {
    {
        Env env({ { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN", "1" },
                  { "RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8", "1" } });
        auto d = declare(0, true);
        REQUIRE_EQ(d->rc, (int)RAD_OK);
        REQUIRE(d->m->f8_on);
        /* The switch alone: the same four reads at their codes' bytes, 26.75 MiB. */
        const auto& L = d->m->layers;
        REQUIRE(reads_are(L[3].pf, pf_kinds{ pk::kQa, pk::kKv, pk::kQb, pk::kOa, pk::kOa }));
        CHECK_EQ(L[3].pf[0].bytes, (int64_t)1280 * 5120);
        CHECK_EQ(L[3].pf[2].bytes, (int64_t)8192 * 1280);
        CHECK_EQ(total(L[3].pf), 107 * MiB / 4);
    }
    Env env({ { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN", "1" },
              { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MIB", "56" },
              { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ENGRAM", "1" },
              { "RADIANCE_EXPERIMENTAL_DS41_ENGRAM_SPLIT", "1" },
              { "RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8", "1" } });
    for (int rank = 0; rank < kWorld; ++rank) {
        auto d = declare(rank, true);
        REQUIRE_EQ(d->rc, (int)RAD_OK);
        REQUIRE(d->m->f8_on);
        const auto& L = d->m->layers;
        const int64_t w = rank == 0 ? 768 : 512;
        const pf_kinds all{ pk::kQa, pk::kKv, pk::kQb, pk::kOa, pk::kOa, pk::kOb, pk::kRt, pk::kSgu, pk::kSdn };
        /* Layer 3: the five attention weights (36.75 MiB), the f32 router (7.5) and this rank's
         * shared slice as codes: 55.5 MiB on rank 0, 51.25 on the others, every read whole. */
        REQUIRE(reads_are(L[3].pf, all));
        for (const auto& x : L[3].pf) CHECK(!x.cut);
        CHECK_EQ(L[3].pf[7].bytes, 2 * w * 5120);
        CHECK_EQ(L[3].pf[8].bytes, 5120 * w);
        CHECK_EQ(total(L[3].pf), 147 * MiB / 4 + 15 * MiB / 2 + 3 * w * 5120);
        /* Layer 2 starts with the f32 compressor (20 MiB) and the indexer's q_b, and cuts wo_b. */
        REQUIRE(reads_are(L[2].pf, pf_kinds{ pk::kComp, pk::kQa, pk::kKv, pk::kIq, pk::kQb, pk::kOa,
                                             pk::kOa, pk::kOb }));
        CHECK_EQ(L[2].pf[0].bytes, (int64_t)1024 * 5120 * 4);
        CHECK(L[2].pf[7].cut);
        CHECK_EQ(total(L[2].pf), 56 * MiB);
        /* Layer 1: all 37.5 MiB of its split engram_kv's codes ahead of the GEMM, the whole list
         * behind it. */
        REQUIRE(reads_are(L[1].pf, pf_kinds{ pk::kEkv }));
        CHECK(!L[1].pf[0].cut);
        CHECK_EQ(L[1].pf[0].bytes, (int64_t)6400 * 6144);
        CHECK(reads_are(L[1].pf_post, all));
        for (const auto& x : L) CHECK(reads_declared(d->b, x.pf) && reads_declared(d->b, x.pf_post));
    }
}

/* Where a decode step issues the reads: every declared read once, each layer's after its shared
 * expert's down and before its gather, on lane 1 behind a join from lane 0; an Engram layer's after
 * its GEMM and before its all_gather. A prompt step issues none. */
TEST(ds41_prefetch_decode_step_issues_each_read_beside_the_wait) {
    Env env({ { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN", "1" },
              { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MIB", "56" },
              { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ENGRAM", "1" },
              { "RADIANCE_EXPERIMENTAL_DS41_ENGRAM_SPLIT", "1" } });
    auto d = declare(0);
    REQUIRE_EQ(d->rc, (int)RAD_OK);
    const RadBuilder& b = d->b;
    const auto& L = d->m->layers;
    size_t want = 0;
    for (const auto& x : L) want += x.pf.size() + x.pf_post.size() - (x.l == 0 ? x.pf.size() : 0);
    auto k = kit(d->b, { 1 }, {});
    auto run = run_passes(d->b, deepseek41::step, { k.get() }, 0, kWorld);
    REQUIRE(steps_ok(run));
    const RadCtx& c = *run[0];
    const std::vector<std::string> ops = issued_ops(b, c);
    REQUIRE_EQ((size_t)count_issued(ops, "prefetch"), want);
    CHECK_EQ(d->m->pfs.steps, (uint64_t)1);
    CHECK_EQ(d->m->pfs.reads, (uint64_t)want);
    CHECK_EQ(d->m->pfs.epoints, (uint64_t)2);
    /* Each prefetch on lane 1, and a join 0 -> 1 right before each run of them. */
    for (size_t i = 0; i < c.issues.size(); ++i) {
        if (ops[i] != "prefetch") continue;
        CHECK_EQ(c.issues[i].lane, 1);
        if (i == 0 || ops[i - 1] != "prefetch") {
            bool joined = false;
            for (const auto& j : c.joins) joined = joined || (j.at == i && j.from == 0 && j.to == 1);
            CHECK(joined);
        }
    }
    /* The issue index of the first issue of `op` reading weight `wn`, from `from` on. */
    auto at_w = [&](const std::string& op, const std::string& wn, size_t from = 0) -> size_t {
        for (size_t i = from; i < c.issues.size(); ++i) {
            const RecOp* o = b.op_of(c.issues[i].op);
            if (o && o->op == op && !o->w.empty() && wname(b, o->w[0]) == wn) return i;
        }
        return c.issues.size();
    };
    auto at_op = [&](const std::string& op, size_t from) -> size_t {
        for (size_t i = from; i < c.issues.size(); ++i)
            if (ops[i] == op) return i;
        return c.issues.size();
    };
    /* Layer l's tail: its shared down, then layer l+1's reads in order, then the gather. */
    for (int l : { 0, 2, 13, 20, 38 }) {
        const std::string p = "blk." + std::to_string(l) + ".";
        const size_t dn = at_w("gemm_nt", p + "ffn_down_shexp.weight");
        REQUIRE(dn < c.issues.size());
        size_t i = dn + 1;
        for (const auto& r : L[(size_t)l + 1].pf) {
            i = at_op("prefetch", i);
            REQUIRE(i < c.issues.size());
            CHECK_EQ(c.issues[i].op, r.op);
            ++i;
        }
        const size_t g = std::min(at_op("moe_gather", dn), at_op("moe_gather_inv", dn));
        CHECK(i <= g);
        CHECK(g < c.issues.size());
    }
    /* Layer 1's own reads: after its Engram GEMM, before its all_gather. */
    const size_t eg = at_w("gemm_nt", "blk.1.engram_kv.weight");
    REQUIRE(eg < c.issues.size());
    const size_t ag = at_op("all_gather", eg);
    size_t i = eg + 1;
    for (const auto& r : L[1].pf_post) {
        i = at_op("prefetch", i);
        REQUIRE(i < c.issues.size());
        CHECK_EQ(c.issues[i].op, r.op);
        ++i;
    }
    CHECK(i <= ag);
    /* (No step lint here: on this fake the experts read as block fp8, whose quantised activations
     * a bf16 model declares none of, and wo_a's groups read column slices of attn_o; neither is
     * the reads' doing, which name no buffer.) */

    /* A prompt step issues none. */
    auto pk_ = kit(d->b, {}, { 16 });
    auto pre = run_passes(d->b, deepseek41::step, { pk_.get() }, 0, kWorld);
    REQUIRE(steps_ok(pre));
    CHECK_EQ(count_issued(issued_ops(b, *pre[0]), "prefetch"), 0);
    CHECK_EQ(d->m->pfs.steps, (uint64_t)1);
    /* The shutdown line (rad_plugin_close): every layer but 0 read at the tail before it, the two
     * Engram layers behind their GEMMs too. */
    const size_t epost = L[1].pf_post.size() + L[14].pf_post.size();
    const std::string rep = stderr_of([] { deepseek41::pf_report(); });
    CHECK(rep.find("rank 0: on-die reads issued live in 1 decode steps: " +
                   std::to_string(want - epost) + " reads at 39 layer tails") != std::string::npos);
    CHECK(rep.find("behind the Engram GEMMs " + std::to_string(epost) + " reads at 2 layers") !=
          std::string::npos);
}

/* Where launches chain, the reads go on lane 0 itself, with no join (prefetch_beside's rule). */
TEST(ds41_prefetch_reads_stay_on_lane0_where_launches_chain) {
    Env env({ { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN", "1" } });
    auto d = declare(0);
    REQUIRE_EQ(d->rc, (int)RAD_OK);
    ChainOn chain(1);
    auto k = kit(d->b, { 1 }, {});
    auto run = run_passes(d->b, deepseek41::step, { k.get() }, 0, kWorld);
    REQUIRE(steps_ok(run));
    const RadCtx& c = *run[0];
    const std::vector<std::string> ops = issued_ops(d->b, c);
    CHECK_EQ(count_issued(ops, "prefetch"), 37 * 5);
    for (size_t i = 0; i < ops.size(); ++i) {
        if (ops[i] != "prefetch") continue;
        CHECK_EQ(c.issues[i].lane, 0);
        for (const auto& j : c.joins) CHECK(j.at != i);
    }
}

/* RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16=1 over the planes the release's container holds: every
 * router f32 (its low halves zero, which gemm_nt_f32bf16's layout keeps as the bf16 halves) and the
 * ratio-2 compressors' kv|gate bf16 (read as they are). */
namespace {
void release_planes(RadBuilder& b) {
    b.encs.push_back({ "glm_router.weight", rad_enc_plain(RAD_F32) });
    b.encs.push_back({ "attn_compressor_kvg.weight", rad_enc_plain(RAD_BF16) });
}
uint32_t decl_dtype(const RadBuilder& b, const std::string& name) {
    for (const auto& w : b.weights)
        if (w.name == name) return w.d.dtype;
    return RAD_DT_INVALID;
}
}  // namespace

/* Off, the routers (op and split part, 40 layers) and the three ratio-2 compressors are 83
 * gemm_nt_f32 over f32, read ahead at four bytes an element. On, the same 83 go through
 * gemm_nt_f32bf16 -- the router still declared f32, the compressor bf16 -- rank 0 counts them
 * (150 + 30 MiB), and the reads take their bf16 bytes. A decode step issues the bf16 form only. */
TEST(ds41_router_bf16_declares_counts_and_reads_half) {
    {
        Env env({ { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN", "1" },
                  { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_SET", "comp,rt" } });
        auto d = declare(0, false, release_planes);
        REQUIRE_EQ(d->rc, (int)RAD_OK);
        CHECK_EQ(count_op(d->b, "gemm_nt_f32"), 83);
        CHECK_EQ(count_op(d->b, "gemm_nt_f32bf16"), 0);
        CHECK_EQ(decl_dtype(d->b, "blk.3.glm_router.weight"), (uint32_t)RAD_F32);
        CHECK_EQ(decl_dtype(d->b, "blk.2.attn_compressor_kvg.weight"), (uint32_t)RAD_F32);
        const auto& L = d->m->layers;
        for (const auto& x : L) CHECK(!x.rt_bf16 && !x.comp_bf16);
        REQUIRE(reads_are(L[2].pf, pf_kinds{ pk::kComp, pk::kRt }));
        CHECK_EQ(L[2].pf[0].bytes, (int64_t)1024 * 5120 * 4);
        CHECK_EQ(L[2].pf[1].bytes, (int64_t)384 * 5120 * 4);
        CHECK(d->err.find("ROUTER_BF16") == std::string::npos);
    }
    Env env({ { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN", "1" },
              { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_SET", "comp,rt" },
              { "RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16", "1" } });
    for (int rank = 0; rank < kWorld; ++rank) {
        auto d = declare(rank, false, release_planes);
        REQUIRE_EQ(d->rc, (int)RAD_OK);
        CHECK_EQ(count_op(d->b, "gemm_nt_f32bf16"), 83);
        CHECK_EQ(count_op(d->b, "gemm_nt_f32"), 0);
        CHECK_EQ(decl_dtype(d->b, "blk.3.glm_router.weight"), (uint32_t)RAD_F32);
        CHECK_EQ(decl_dtype(d->b, "blk.2.attn_compressor_kvg.weight"), (uint32_t)RAD_BF16);
        const auto& L = d->m->layers;
        REQUIRE_EQ(L.size(), (size_t)40);
        for (const auto& x : L) {
            CHECK(x.rt_bf16);
            CHECK_EQ(x.comp_bf16, x.l == 2 || x.l == 8 || x.l == 14);
        }
        /* Layer 2: the kv|gate [1024, 5120] and the router [384, 5120] at two bytes; layer 20's
         * ratio-1 compressor is lw()'s bf16 [512, 5120] either way; layer 3 has the router only. */
        REQUIRE(reads_are(L[2].pf, pf_kinds{ pk::kComp, pk::kRt }));
        CHECK_EQ(L[2].pf[0].bytes, (int64_t)1024 * 5120 * 2);
        CHECK_EQ(L[2].pf[1].bytes, (int64_t)384 * 5120 * 2);
        REQUIRE(reads_are(L[20].pf, pf_kinds{ pk::kComp, pk::kRt }));
        CHECK_EQ(L[20].pf[0].bytes, (int64_t)512 * 5120 * 2);
        REQUIRE(reads_are(L[3].pf, pf_kinds{ pk::kRt }));
        CHECK_EQ(L[3].pf[0].bytes, (int64_t)384 * 5120 * 2);
        CHECK_EQ(wname(d->b, L[3].pf[0].w), std::string("blk.3.glm_router.weight"));
        for (const auto& x : L) CHECK(reads_declared(d->b, x.pf));
        CHECK_EQ(d->m->n_router_bf16, 40);
        CHECK_EQ(d->m->n_comp_bf16, 3);
        const std::string want = "RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16: 40 routers and 3 "
                                 "compressor kv|gate projections read as bf16 (gemm_nt_f32bf16), "
                                 "180.0 MiB less a card";
        const bool line = d->err.find(want) != std::string::npos;
        CHECK_EQ(line, rank == 0);
        if (rank) continue;
        auto k = kit(d->b, { 1 }, {});
        auto run = run_passes(d->b, deepseek41::step, { k.get() }, 0, kWorld);
        REQUIRE(steps_ok(run));
        const std::vector<std::string> ops = issued_ops(d->b, *run[0]);
        CHECK(count_issued(ops, "gemm_nt_f32bf16") >= 40);
        CHECK_EQ(count_issued(ops, "gemm_nt_f32"), 0);
    }
}

/* With the dense linears as E4M3 codes and a 56 MiB budget, the halved compressor and router let
 * layer 2 read wo_b whole and its router (55.5 MiB; f32 they cut wo_b), and layer 3 read its whole
 * list in 3.75 MiB less. */
TEST(ds41_router_bf16_with_fp8_codes_reaches_further) {
    Env env({ { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN", "1" },
              { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MIB", "56" },
              { "RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8", "1" },
              { "RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16", "1" } });
    for (int rank = 0; rank < kWorld; ++rank) {
        auto d = declare(rank, true, release_planes);
        REQUIRE_EQ(d->rc, (int)RAD_OK);
        REQUIRE(d->m->f8_on);
        const auto& L = d->m->layers;
        const int64_t w = rank == 0 ? 768 : 512;
        REQUIRE(reads_are(L[2].pf, pf_kinds{ pk::kComp, pk::kQa, pk::kKv, pk::kIq, pk::kQb, pk::kOa,
                                             pk::kOa, pk::kOb, pk::kRt }));
        for (const auto& x : L[2].pf) CHECK(!x.cut);
        CHECK_EQ(L[2].pf[0].bytes, (int64_t)1024 * 5120 * 2);
        CHECK_EQ(L[2].pf[8].bytes, (int64_t)384 * 5120 * 2);
        CHECK_EQ(total(L[2].pf), 111 * MiB / 2);
        REQUIRE(reads_are(L[3].pf, pf_kinds{ pk::kQa, pk::kKv, pk::kQb, pk::kOa, pk::kOa, pk::kOb,
                                             pk::kRt, pk::kSgu, pk::kSdn }));
        CHECK_EQ(total(L[3].pf), 147 * MiB / 4 + 15 * MiB / 4 + 3 * w * 5120);
        for (const auto& x : L) CHECK(reads_declared(d->b, x.pf));
    }
}

/* A plane whose encoding the builder does not know is still read through the op, but not counted
 * and read ahead at four bytes; no library declaring the op refuses the declare rather than serve
 * the routers as f32. */
TEST(ds41_router_bf16_unknown_planes_and_refusal) {
    {
        Env env({ { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN", "1" },
                  { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_SET", "rt" },
                  { "RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16", "1" } });
        auto d = declare(0);
        REQUIRE_EQ(d->rc, (int)RAD_OK);
        CHECK_EQ(count_op(d->b, "gemm_nt_f32bf16"), 83);
        CHECK(!d->m->layers[3].rt_bf16);
        REQUIRE(reads_are(d->m->layers[3].pf, pf_kinds{ pk::kRt }));
        CHECK_EQ(d->m->layers[3].pf[0].bytes, (int64_t)384 * 5120 * 4);
        CHECK(d->err.find("ROUTER_BF16: 0 routers and 0 compressor") != std::string::npos);
    }
    Env env({ { "RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16", "1" } });
    auto d = declare(0, false, [](RadBuilder& b) {
        release_planes(b);
        b.undeclared.insert("gemm_nt_f32bf16");
    });
    CHECK_EQ(d->rc, (int)RAD_E_UNSUPPORTED);
    CHECK(d->err.find("ROUTER_BF16=1 needs gemm_nt_f32bf16") != std::string::npos);
    CHECK_EQ(count_op(d->b, "gemm_nt_f32bf16"), 0);
}

RAD_TEST_MAIN()
