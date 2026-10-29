/* ds41_consumer_skip_arch_test.cpp -- deepseek41's half of the consumer skip
 * (RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP; core/sched/consumer_skip.h), on the recording builder
 * (arch_fakes.h) with DeepSeek-V4.1-Flash's own geometry at TP4, no card or model file:
 *   - off, nothing is declared and a batch's consumer_full is never read;
 *   - on, every rank declares boundary 20 and window 128 (the last kv source and the window), and
 *     a graph off that shape is refused with its reason (speculation, an Engram or candidate layer
 *     past the boundary); a later kv source moves the boundary;
 *   - a cut pass issues exactly the full pass's ops up to layer F's input (the previous layer's
 *     fused post + collapse + norm), then layer F's key ops -- the same handles, in the full pass's
 *     order -- and nothing else: no query, indexer, attention, all-reduce or FFN of layer F, no
 *     layer after it, no final norm or head; at the boundary that includes its compressor and its
 *     compressed and index key stores;
 *   - a cut that samples, or names a layer outside the suffix, fails the step;
 *   - over a whole prompt's passes (the schedule's F for each), every layer's stores are issued on
 *     exactly the passes the schedule says. */
#define RAD_ARCH_NO_EXPORTS 1

#include "arch_fakes.h"

#include "../arch/deepseek41/deepseek41.cpp"
#include "../core/sched/consumer_skip.h"

#include <cstdlib>
#include <set>
#include <string>
#include <vector>

namespace {

constexpr int kWorld = 4;

const char* const kVars[] = { "RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP",
                              "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN",
                              "RADIANCE_EXPERIMENTAL_DS41_ENGRAM_SPLIT",
                              "RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8",
                              "RADIANCE_EXPERIMENTAL_DS41_KV_REPLICATED" };
struct Env {
    Env(std::initializer_list<std::pair<const char*, const char*>> kv) {
        for (const char* n : kVars) unsetenv(n);
        for (const auto& p : kv) setenv(p.first, p.second, 1);
    }
    ~Env() {
        for (const char* n : kVars) unsetenv(n);
    }
};

/* The release's geometry as llama.cpp's converter states it (ds41_prefetch_arch_test's). */
MetaKV flash_meta(const char* kv_src = "2,8,14,20", const char* ple = "1,14", const char* cand = "20") {
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
        .set("deepseek41.attention.indexer.candidate_source_layer", cand)
        .set("deepseek41.attention.indexer.candidate_top_k_blocks", "2048")
        .set("deepseek41.attention.indexer.candidate_block_size", "8")
        .set("deepseek41.hyper_connection.count", "4")
        .set("deepseek41.hyper_connection.sinkhorn_iterations", "20")
        .set("deepseek41.hyper_connection.epsilon", "1e-6")
        .set("deepseek41.attention.compress_rope_freq_base", "160000")
        .set("deepseek41.attention.compress_ratios", ratios.c_str())
        .set("deepseek41.attention.kv_source_layers", kv_src)
        .set("deepseek41.attention.indexer.source_layers", "2,8,14,20,24,28,32,36")
        .set("deepseek41.ple.layers", ple)
        .set("deepseek41.ple.ngram_size", "4")
        .set("deepseek41.ple.heads_per_ngram", "8")
        .set("deepseek41.embedding_length_per_layer_input", "256")
        .set("deepseek41.engram.pad_token_id", "2")
        .set("deepseek41.ple.head_vocab_sizes", "1000,1000");
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
/* deepseek41's declare for `rank` of four at 16 rows a step, under the current environment. */
std::unique_ptr<Decl> declare(int rank, MetaKV meta, int max_spec = 0, int world = kWorld) {
    auto d = std::make_unique<Decl>();
    d->b.kv_block_of = block_of;
    RadBuildCtx ctx = build_ctx(rank, world);
    ctx.max_tok = 16;
    ctx.max_ctx = 8192;
    ctx.max_spec = max_spec;
    d->err = stderr_of([&] { d->rc = deepseek41::declare(&d->b, meta.meta(), &ctx); });
    d->m = deepseek41::models[rank].get();
    return d;
}
std::unique_ptr<Decl> declare(int rank) { return declare(rank, flash_meta()); }

/* A 16-row prefill chunk at `at`, sampling nothing (a chunk in the middle of a prompt), cut to
 * `full` leading layers (0: every layer). */
std::unique_ptr<Kit> chunk(const RadBuilder& b, int32_t at, int32_t full, bool sample = false) {
    auto k = kit(b, {}, { 16 }, at);
    if (!sample) {
        k->out.clear();
        k->b.n_out = 0;
        k->b.out_ids = nullptr;
    }
    k->b.consumer_full = full;
    k->b.consumer_end = full ? at + 400 : 0;
    return k;
}

std::vector<rad_op> issued(const RadCtx& c) {
    std::vector<rad_op> v;
    for (const auto& i : c.issues) v.push_back(i.op);
    return v;
}

/* Layer L's own op handles (pcn_a, issued for the layer before, excluded), and of them its key
 * ops: what the cut keeps of it. */
void layer_ops(const deepseek41::Layer& L, std::set<rad_op>* own, std::set<rad_op>* keys) {
    const rad_op key_ops[] = { L.col_a, L.norm_a, L.cn_a, L.kv_g, L.kv_n, L.kv_rope, L.kv_q, L.kv_st,
                               L.kv_f, L.c_gemm, L.c_work, L.c_pool, L.c_norm, L.c_tab, L.k_gemm,
                               L.k_norm, L.k_rope, L.k_q, L.k_st, L.k_f, L.l_rope, L.l_q, L.l_st,
                               L.l_f, L.c_tail };
    const rad_op rest[] = { L.e_ids, L.e_emb, L.e_copy, L.e_gemm, L.e_gather, L.e_gate_op, L.e_back,
                            L.mix_a, L.q_a, L.q_an, L.q_b, L.q_rope, L.i_q, L.i_rope, L.i_qq, L.i_w,
                            L.i_score, L.i_cand, L.i_topk, L.i_f, L.attn, L.o_rope, L.o_q, L.o_b,
                            L.o_f, L.post_a, L.attn_fill, L.ar_attn, L.mix_f, L.col_f, L.norm_f,
                            L.post_f, L.pcn_f, L.cn_f, L.tap };
    for (rad_op h : key_ops) if (h) { own->insert(h); keys->insert(h); }
    for (rad_op h : rest) if (h) own->insert(h);
    for (rad_op h : L.o_a) if (h) own->insert(h);
    for (rad_op h : L.i_runs) if (h) own->insert(h);
}

/* The full pass's issues a pass cut at F must be: everything before layer F's first own op, then
 * layer F's key ops in the full pass's order, then nothing. Layer F-1's MoE ops (MoeFP8, its
 * handles not in Layer) and its post (or the fused post + layer F's collapse + norm) come before
 * layer F's first own op. */
std::vector<rad_op> expected_cut(const deepseek41::Model& m, const std::vector<rad_op>& full, int F) {
    std::set<rad_op> own, keys;
    layer_ops(m.layers[(size_t)F], &own, &keys);
    const rad_op pcn = m.layers[(size_t)F].pcn_a;
    size_t first = full.size();
    for (size_t i = 0; i < full.size(); ++i)
        if (own.count(full[i])) { first = i; break; }
    std::vector<rad_op> want(full.begin(), full.begin() + (std::ptrdiff_t)first);
    /* With the collapse fused into the layer before's post, the prefix already ends with it. */
    if (pcn) {
        bool seen = false;
        for (rad_op h : want) seen = seen || h == pcn;
        if (!seen) want.push_back(0);   /* fails the comparison: the fused op must precede */
    }
    for (size_t i = first; i < full.size(); ++i)
        if (keys.count(full[i])) want.push_back(full[i]);
    return want;
}

std::string op_name(const RadBuilder& b, rad_op h) {
    const RecOp* o = b.op_of(h);
    return o ? o->op : std::string("<none>");
}
/* On a mismatch, the first difference. */
bool same_issues(const RadBuilder& b, const std::vector<rad_op>& got, const std::vector<rad_op>& want) {
    if (got == want) return true;
    size_t i = 0;
    while (i < got.size() && i < want.size() && got[i] == want[i]) ++i;
    fprintf(stderr, "    %zu issues against %zu expected; first difference at %zu: %s against %s\n",
            got.size(), want.size(), i, i < got.size() ? op_name(b, got[i]).c_str() : "(end)",
            i < want.size() ? op_name(b, want[i]).c_str() : "(end)");
    return false;
}

}  // namespace

TEST(ds41_consumer_skip_off_declares_nothing_and_reads_no_cut) {
    Env env({});
    auto d = declare(0);
    REQUIRE_EQ(d->rc, (int)RAD_OK);
    CHECK_EQ(d->b.consumer_calls, 0);
    CHECK(!d->m->cut_on);
    CHECK(d->err.find("consumer") == std::string::npos);
    /* A batch's consumer_full is not read without the declaration: every layer runs. */
    auto full = chunk(d->b, 1024, 0);
    auto cut = chunk(d->b, 1024, 25);
    auto rf = run_passes(d->b, deepseek41::step, { full.get() }, 0, kWorld);
    auto rc = run_passes(d->b, deepseek41::step, { cut.get() }, 0, kWorld);
    REQUIRE(steps_ok(rf));
    REQUIRE(steps_ok(rc));
    CHECK(issued(*rf[0]) == issued(*rc[0]));
}

TEST(ds41_consumer_skip_declares_the_flash_suffix_on_every_rank) {
    Env env({ { "RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP", "1" } });
    for (int rank = 0; rank < kWorld; ++rank) {
        auto d = declare(rank);
        REQUIRE_EQ(d->rc, (int)RAD_OK);
        CHECK_EQ(d->b.consumer_calls, 1);
        CHECK_EQ(d->b.consumer.n_layer, 40);
        CHECK_EQ(d->b.consumer.boundary, 20);
        CHECK_EQ(d->b.consumer.window, 128);
        CHECK_EQ(d->b.consumer.struct_size, (uint32_t)sizeof(RadConsumerSuffix));
        CHECK(d->m->cut_on);
        CHECK_EQ(d->m->cut_b, 20);
        /* One line, from rank 0. */
        const bool line = d->err.find("consumer suffix declared: layers 21..39 see earlier rows only "
                                      "through their own 128-row windows and layer 20's") != std::string::npos;
        CHECK_EQ(line, rank == 0);
    }
}

TEST(ds41_consumer_skip_refuses_a_graph_off_the_contract) {
    Env env({ { "RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP", "1" } });
    struct Case { MetaKV meta; int spec; const char* why; };
    Case cases[] = {
        { flash_meta(), 2, "speculation is on" },
        { flash_meta("2,8,14,20", "1,24"), 0, "layer 24 has an Engram block" },
        { flash_meta("2,8,14,20", "1,14", "24"), 0, "the candidate layer 24 is past the boundary" },
    };
    for (Case& k : cases) {
        auto d = declare(0, k.meta, k.spec);
        REQUIRE_EQ(d->rc, (int)RAD_OK);
        CHECK_EQ(d->b.consumer_calls, 0);
        CHECK(!d->m->cut_on);
        const bool said = d->err.find(std::string("consumer skip refused: ") + k.why) != std::string::npos;
        CHECK(said);
        if (!said) fprintf(stderr, "    stderr: %s\n", d->err.c_str());
    }
    /* A later kv source moves the boundary: layer 30's keys, read by the layers after it. */
    auto d = declare(0, flash_meta("2,8,14,20,30"));
    REQUIRE_EQ(d->rc, (int)RAD_OK);
    CHECK_EQ(d->b.consumer.boundary, 30);
    CHECK_EQ(d->m->cut_b, 30);
}

/* The cut pass against the full pass, at the boundary (its keys alone: window, compressor, index
 * and compressed stores), one past it, in the middle and at the last cut layer, on every rank. */
TEST(ds41_cut_pass_issues_the_full_prefix_then_the_key_ops_alone) {
    Env env({ { "RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP", "1" } });
    for (int rank = 0; rank < kWorld; ++rank) {
        auto d = declare(rank);
        REQUIRE_EQ(d->rc, (int)RAD_OK);
        REQUIRE(d->m->cut_on);
        const deepseek41::Model& m = *d->m;
        auto fk = chunk(d->b, 1024, 0);
        auto rf = run_passes(d->b, deepseek41::step, { fk.get() }, rank, kWorld);
        REQUIRE(steps_ok(rf));
        const std::vector<rad_op> full = issued(*rf[0]);
        for (int F : { 20, 21, 33, 39 }) {
            auto ck = chunk(d->b, 1024, F);
            auto rc = run_passes(d->b, deepseek41::step, { ck.get() }, rank, kWorld);
            REQUIRE(steps_ok(rc));
            const std::vector<rad_op> got = issued(*rc[0]);
            CHECK(same_issues(d->b, got, expected_cut(m, full, F)));
            /* Nothing of the head, of the layer's attention or FFN, of any later layer. */
            const std::vector<std::string> names = issued_ops(d->b, *rc[0]);
            CHECK_EQ(count_issued(names, "logits_gemm") + count_issued(names, "gather_rows"), 0);
            const deepseek41::Layer& L = m.layers[(size_t)F];
            for (rad_op h : got) {
                CHECK(h != L.attn && h != L.q_a && h != L.mix_a && h != L.ar_attn);
                CHECK(h != L.post_f && h != L.mix_f);
            }
            for (int l = F + 1; l < 40; ++l) {
                std::set<rad_op> own, keys;
                layer_ops(m.layers[(size_t)l], &own, &keys);
                int n = 0;
                for (rad_op h : got) n += own.count(h) != 0 && h != m.layers[(size_t)l].pcn_a;
                CHECK_EQ(n, 0);
            }
            /* The key layer's stores: its window key, and at the boundary the two key stores. */
            int kv_st = 0, k_st = 0, l_st = 0;
            for (rad_op h : got) {
                kv_st += h == L.kv_st;
                k_st += L.k_st && h == L.k_st;
                l_st += L.l_st && h == L.l_st;
            }
            CHECK_EQ(kv_st, 1);
            CHECK_EQ(k_st, F == 20 ? 1 : 0);
            CHECK_EQ(l_st, F == 20 ? 1 : 0);
        }
        CHECK_EQ(m.cuts.passes, (uint64_t)4);
        CHECK_EQ(m.cuts.key, (uint64_t)4);
        CHECK_EQ(m.cuts.full, (uint64_t)(20 + 21 + 33 + 39));
    }
}

/* Three ranks attend on two (Geom::attn_world): the attention-zero rank's key layer is its collapse
 * alone (its caches are zeroed and never written), and the attention ranks' as at four. */
TEST(ds41_cut_pass_on_an_attention_zero_rank) {
    Env env({ { "RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP", "1" } });
    for (int rank : { 0, 2 }) {
        auto d = declare(rank, flash_meta(), 0, 3);
        REQUIRE_EQ(d->rc, (int)RAD_OK);
        REQUIRE(d->m->cut_on);
        CHECK_EQ(d->m->zero, rank == 2);
        auto fk = chunk(d->b, 1024, 0);
        auto rf = run_passes(d->b, deepseek41::step, { fk.get() }, rank, 3);
        REQUIRE(steps_ok(rf));
        for (int F : { 20, 25 }) {
            auto ck = chunk(d->b, 1024, F);
            auto rc = run_passes(d->b, deepseek41::step, { ck.get() }, rank, 3);
            REQUIRE(steps_ok(rc));
            CHECK(same_issues(d->b, issued(*rc[0]), expected_cut(*d->m, issued(*rf[0]), F)));
        }
    }
}

TEST(ds41_cut_pass_outside_the_suffix_fails_the_step) {
    Env env({ { "RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP", "1" } });
    auto d = declare(0);
    REQUIRE_EQ(d->rc, (int)RAD_OK);
    for (int F : { 19, 40 }) {
        auto k = chunk(d->b, 1024, F);
        auto r = run_passes(d->b, deepseek41::step, { k.get() }, 0, kWorld);
        CHECK(!r[0]->step_fail.empty());
    }
    auto s = chunk(d->b, 1024, 30, /*sample=*/true);
    auto r = run_passes(d->b, deepseek41::step, { s.get() }, 0, kWorld);
    CHECK(r[0]->step_fail.find("consumer suffix") != std::string::npos);
}

/* A whole 400-token prompt at 16-row passes under the schedule (target 400): every layer stores
 * its window keys on exactly the passes the schedule runs it on (full or keys alone), the boundary
 * its compressed and index keys on every pass, and the last pass runs every layer and the head. */
TEST(ds41_a_prompt_under_the_schedule_stores_what_the_schedule_says) {
    Env env({ { "RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP", "1" } });
    auto d = declare(1);
    REQUIRE_EQ(d->rc, (int)RAD_OK);
    const deepseek41::Model& m = *d->m;
    const rad::ConsumerShape shape{ 40, 20, 128, 16, 16 };
    const int64_t n = 400;
    const std::vector<int64_t> f = rad::consumer_thresholds(shape, n);
    REQUIRE_EQ(f.size(), (size_t)40);
    int cut = 0;
    for (int64_t s = 0; s < n; s += 16) {
        const int64_t e = std::min<int64_t>(n, s + 16);
        const int32_t F = rad::consumer_full_layers(shape, f, e - 1);
        auto k = kit(d->b, {}, { (int)(e - s) }, (int32_t)s);
        if (e < n) {
            k->out.clear();
            k->b.n_out = 0;
            k->b.out_ids = nullptr;
        }
        k->b.consumer_full = F < 40 ? F : 0;
        cut += F < 40;
        auto r = run_passes(d->b, deepseek41::step, { k.get() }, 1, kWorld);
        REQUIRE(steps_ok(r));
        const std::vector<rad_op> got = issued(*r[0]);
        for (int l = 0; l < 40; ++l) {
            const deepseek41::Layer& L = m.layers[(size_t)l];
            int stores = 0;
            for (rad_op h : got) stores += h == L.kv_st;
            const bool runs = l < F || l == F;   /* in full, or its keys alone */
            CHECK_EQ(stores, runs ? 1 : 0);
            if (L.k_st) {
                int ks = 0;
                for (rad_op h : got) ks += h == L.k_st;
                CHECK_EQ(ks, l <= 20 ? 1 : 0);   /* the sources, the boundary on every pass */
            }
        }
        const std::vector<std::string> names = issued_ops(d->b, *r[0]);
        CHECK_EQ(count_issued(names, "logits_gemm") + count_issued(names, "logits_gemm_ggml"),
                 e == n ? 1 : 0);
    }
    CHECK(cut > 0);
    CHECK_EQ(m.cuts.passes, (uint64_t)cut);
}

/* RADIANCE_EXPERIMENTAL_DS41_KV_REPLICATED (kv_replicate): the compressed and index caches are
 * declared replicated on every rank of four, so a host-tier entry holds one copy; the window, the
 * compressor tails and the Engram history are left alone, and without the switch nothing is
 * declared. At three ranks one keeps no attention cache (attention-zero), so it is refused there
 * on every rank, with its line from rank 0. */
TEST(ds41_kv_replicated_declares_the_full_caches_on_every_rank) {
    for (int on = 0; on < 2; ++on) {
        Env env({});
        if (on) setenv("RADIANCE_EXPERIMENTAL_DS41_KV_REPLICATED", "1", 1);
        for (int rank = 0; rank < kWorld; ++rank) {
            auto d = declare(rank);
            REQUIRE_EQ(d->rc, (int)RAD_OK);
            int full = 0, rep = 0;
            for (const RecKV& k : d->b.kv_groups) {
                const bool f = k.name.rfind("kv_index_r", 0) == 0 ||
                               k.name.rfind("kv_compressed_r", 0) == 0;
                full += f;
                if (k.replicated) {
                    ++rep;
                    CHECK(f);
                }
            }
            CHECK_EQ(full, 4);                    /* index and compressed, at ratios 2 and 1 */
            CHECK_EQ(rep, on ? 4 : 0);
            const bool line = d->err.find("KV_REPLICATED: 4 paged group(s) declared replicated "
                                          "(kv_index_r1,kv_index_r2,kv_compressed_r1,"
                                          "kv_compressed_r2)") != std::string::npos;
            CHECK_EQ(line, on && rank == 0);
        }
    }
    Env env({ { "RADIANCE_EXPERIMENTAL_DS41_KV_REPLICATED", "1" } });
    for (int rank = 0; rank < 3; ++rank) {
        auto d = declare(rank, flash_meta(), 0, 3);
        REQUIRE_EQ(d->rc, (int)RAD_OK);
        CHECK_EQ(d->m->zero, rank == 2);
        for (const RecKV& k : d->b.kv_groups) CHECK(!k.replicated);
        const bool refused = d->err.find("KV_REPLICATED refused: a rank keeps no attention cache") !=
                             std::string::npos;
        CHECK_EQ(refused, rank == 0);
    }
}

RAD_TEST_MAIN()
