/* ds41_shape_probe_arch_test.cpp -- deepseek41's sizing declare (RadBuildCtx::shape_probe), which the
 * core's arena levels need (Engine::probe_arena_levels, plan_level) and prefill Phase 1's lent arena
 * uses (RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA, core/place/card_expert_stager.h), on the recording
 * builder (arch_fakes.h) with DeepSeek-V4.1-Flash's own geometry at TP4, no card or model file:
 *   - probe() says shape_probe_ok only beside the lent arena (P1 on and a count above 0), so every
 *     other start declares and lays out its arena as before;
 *   - a sizing declare at 256 or 1024 rows under the window's switches (the consumer skip, the on-die
 *     reads, the Engram split) declares the real one's graph op for op, which the core's builder
 *     requires of it (rad_builder.cpp, set_reference): the same op at every position with the same
 *     fixed parameters (only qsa_work's `work`, a capacity in both kernel libraries, shrinks), and the
 *     same buffers with only their rows smaller (plan_level);
 *   - it leaves the real declare's model alone (models[] still the real one, at the real step) and
 *     prints nothing (the window counts the real declare's lines);
 *   - at a context long enough for the indexer to score a step in several runs, it keeps the real
 *     declare's run count;
 *   - a step of at most the level's rows, issued by the real model, stays inside the level's buffers
 *     (the core's operand guard checks the active level's extents): the compressor tail's work-sized
 *     buffers, which step() issues at the real bound whatever the step, keep their real rows; a
 *     2,048-row step does not fit the level (the guard's control);
 *   - beside LCP, as P1 runs, step() issues those at a step's own bound (work_rows()), so the level
 *     declares them for its own rows, and steps of its rows still fit;
 *   - laid out as the core does (persistent buffers at the bottom, transients packed largest first
 *     by lifetime), the 256-row level ends far below the full plan: the lent arena's room, printed. */
#define RAD_ARCH_NO_EXPORTS 1

#include "arch_fakes.h"

#include "../arch/deepseek41/deepseek41.cpp"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

constexpr int kWorld = 4;

const char* const kVars[] = { "RADIANCE_EXPERIMENTAL_DS41_P1",
                              deepseek41::kIndexCompactEnv,
                              "RADIANCE_EXPERIMENTAL_DS41_P1_LEND_ARENA",
                              "RADIANCE_EXPERIMENTAL_DS41_LCP",
                              "RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL",
                              "RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP",
                              "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN",
                              "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MIB",
                              "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ENGRAM",
                              "RADIANCE_EXPERIMENTAL_DS41_ENGRAM_SPLIT",
                              "RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8",
                              "RADIANCE_EXPERIMENTAL_DS41_ROUTER_BF16",
                              "RADIANCE_EXPERIMENTAL_DS41_KV_REPLICATED",
                              "RADIANCE_EXPERIMENTAL_SCHED_TBO" };
struct Env {
    Env(std::initializer_list<std::pair<const char*, const char*>> kv) {
        for (const char* n : kVars) unsetenv(n);
        for (const auto& p : kv) setenv(p.first, p.second, 1);
    }
    ~Env() {
        for (const char* n : kVars) unsetenv(n);
    }
};
/* The window's switches the plugin reads (the adopted P1 arm's Cs, Pa / Pm56 / Pe, Es and F8). */
#define WINDOW_SWITCHES                                                                             \
    { "RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP", "1" }, { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ATTN", "1" }, \
        { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_MIB", "56" }, { "RADIANCE_EXPERIMENTAL_DS41_PREFETCH_ENGRAM", "1" }, \
        { "RADIANCE_EXPERIMENTAL_DS41_ENGRAM_SPLIT", "1" }, { "RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8", "1" }

/* The release's geometry as llama.cpp's converter states it (ds41_consumer_skip_arch_test's). */
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
        .set("deepseek41.ple.head_vocab_sizes", "1000,1000")
        /* The release's statement of its dense linears (ds41_dspark_fp8_arch_test's), for F8. */
        .set("imported_quantization_config.quant_method", "fp8")
        .set("imported_quantization_config.scale_fmt", "ue8m0")
        .set("imported_quantization_config.weight_block_size", "32,32");
    return k;
}

std::string say(const char* f, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
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
    deepseek41::Model* m = nullptr;   /* models[rank] after the call */
};
/* deepseek41's declare for `rank` of four at `max_tok` rows a step (one sequence, as the P1 arms
 * serve), a sizing declare when `probe`, under the current environment. */
std::unique_ptr<Decl> declare(int rank, int64_t max_tok, bool probe, int64_t max_ctx = 4096) {
    auto d = std::make_unique<Decl>();
    d->b.kv_block_of = block_of;
    RadBuildCtx ctx = build_ctx(rank, kWorld);
    ctx.max_tok = max_tok;
    ctx.max_ctx = max_ctx;
    ctx.max_seqs = 1;
    ctx.shape_probe = probe ? 1 : 0;
    MetaKV meta = flash_meta();
    d->err = stderr_of([&] { d->rc = deepseek41::declare(&d->b, meta.meta(), &ctx); });
    d->m = deepseek41::models[rank].get();
    return d;
}

/* The core's rules for laying a sizing declare over the real one (rad_builder.cpp's reference check,
 * rad_bufplan.cpp's plan_level): the first difference, or empty. The one fixed integer a kernel declares
 * RAD_PROLE_CAPACITY in this graph, qsa_work's `work` (both libraries), may shrink. */
std::string differs(const RadBuilder& real, const RadBuilder& probe) {
    if (real.ops.size() != probe.ops.size())
        return say("%zu ops in the real declare, %zu in the sizing one", real.ops.size(), probe.ops.size());
    for (size_t i = 0; i < real.ops.size(); ++i) {
        const RecOp& a = real.ops[i];
        const RecOp& q = probe.ops[i];
        if (a.op != q.op) return say("op %zu is '%s', and '%s' in the sizing declare", i, a.op.c_str(), q.op.c_str());
        size_t fixed_a = 0, fixed_q = 0;
        for (const RecParam& p : a.p) fixed_a += p.kind != RAD_P_RANGE;
        for (const RecParam& p : q.p) fixed_q += p.kind != RAD_P_RANGE;
        if (fixed_a != fixed_q) return say("op %zu ('%s'): the parameter list", i, a.op.c_str());
        for (const RecParam& p : a.p) {
            if (p.kind == RAD_P_RANGE) continue;
            const RecParam* y = nullptr;
            for (const RecParam& x : q.p)
                if (x.key == p.key) y = &x;
            bool same = y && y->kind == p.kind;
            if (same && p.kind == RAD_P_INT)
                same = y->ival == p.ival || (a.op == "qsa_work" && p.key == "work" && y->ival <= p.ival);
            else if (same && p.kind == RAD_P_STR) same = y->sval == p.sval;
            else if (same) same = y->dval == p.dval;
            if (!same)
                return say("op %zu ('%s') changes '%s' (%lld to %lld)", i, a.op.c_str(), p.key.c_str(), (long long)p.ival,
                           y ? (long long)y->ival : -1ll);
        }
    }
    if (real.bufs.size() != probe.bufs.size())
        return say("%zu buffers in the real declare, %zu in the sizing one", real.bufs.size(), probe.bufs.size());
    for (size_t i = 0; i < real.bufs.size(); ++i) {
        const RecBuf& f = real.bufs[i];
        const RecBuf& q = probe.bufs[i];
        if (f.name != q.name || f.d.kind != q.d.kind || f.d.domain != q.d.domain || f.d.dtype != q.d.dtype ||
            f.d.rank != q.d.rank)
            return say("buffer %zu is '%s', and '%s' in the sizing declare", i, f.name.c_str(), q.name.c_str());
        for (uint32_t d = 1; d < f.d.rank; ++d)
            if (f.d.shape[d] != q.d.shape[d])
                return say("buffer '%s' changes extent %u", f.name.c_str(), d);
        if (f.d.rank > 0 && q.d.shape[0] > f.d.shape[0])
            return say("buffer '%s' grows (%lld rows to %lld)", f.name.c_str(), (long long)f.d.shape[0],
                       (long long)q.d.shape[0]);
    }
    return "";
}

/* ---- the core's buffer plan, for the arena's sizes (rad_bufplan.cpp: plan_buffers, plan_level) ---- */
int64_t buf_bytes(const RadBufDecl& d) {
    int64_t n = 1;
    for (uint32_t k = 0; k < d.rank; ++k) n *= d.shape[k];
    return (rad_dtype_bytes(d.dtype, n) + 255) / 256 * 256;
}
struct PItem { size_t idx; int64_t bytes; int32_t lo, hi; int64_t off = -1; };
int64_t pack(std::vector<PItem>& items) {
    std::sort(items.begin(), items.end(), [](const PItem& a, const PItem& b) {
        if (a.bytes != b.bytes) return a.bytes > b.bytes;
        if (a.lo != b.lo) return a.lo < b.lo;
        return a.idx < b.idx;
    });
    int64_t end = 0;
    for (PItem& it : items) {
        std::vector<std::pair<int64_t, int64_t>> occ;
        for (const PItem& p : items)
            if (p.off >= 0 && p.lo <= it.hi && it.lo <= p.hi) occ.emplace_back(p.off, p.off + p.bytes);
        std::sort(occ.begin(), occ.end());
        int64_t off = 0;
        for (auto& [s, e] : occ) {
            if (s >= off + it.bytes) break;
            off = std::max(off, (e + 255) / 256 * 256);
        }
        it.off = off;
        end = std::max(end, it.off + it.bytes);
    }
    return end;
}
/* The device arena the real declare's plan takes, and where a level of the sizing declare's rows ends:
 * persistent buffers at the bottom at their real size, transients packed above at the real
 * lifetimes. */
void arena(const RadBuilder& real, const RadBuilder& probe, int64_t* full, int64_t* level) {
    const std::vector<Live> L = liveness(real);
    int64_t fixed = 0;
    std::vector<PItem> a, q;
    for (size_t i = 0; i < real.bufs.size(); ++i) {
        const RadBufDecl& d = real.bufs[i].d;
        if (d.domain == RAD_DOMAIN_HOST) continue;
        if (d.kind != RAD_BUF_TRANSIENT) {
            fixed += buf_bytes(d);
            continue;
        }
        const Live& l = L[i + 1];
        a.push_back({ i, buf_bytes(d), l.lo, l.hi });
        q.push_back({ i, buf_bytes(probe.bufs[i].d), l.lo, l.hi });
    }
    *full = fixed + pack(a);
    *level = fixed + pack(q);
}

/* The real model's step over `batch` against a level's buffers: the real builder with each transient
 * device buffer's declaration taken from the sizing declare (plan_level keeps the rest), and the step
 * lint's findings of an operand past its buffer. */
std::vector<std::string> overruns(const RadBuilder& real, const RadBuilder& probe, const RadBatch& batch, int rank,
                                  std::string* step_fail) {
    RadBuilder lv = real;
    for (size_t i = 0; i < lv.bufs.size(); ++i)
        if (lv.bufs[i].d.kind == RAD_BUF_TRANSIENT && lv.bufs[i].d.domain != RAD_DOMAIN_HOST) lv.bufs[i].d = probe.bufs[i].d;
    RadCtx c;
    c.b = &lv;
    c.batch = &batch;
    c.rank = rank;
    c.world = kWorld;
    deepseek41::step(&c, &batch);
    if (step_fail) *step_fail = c.step_fail;
    std::vector<std::string> past;
    for (const std::string& f : lint_step(lv, { &c }))
        if (f.find("span past") != std::string::npos) past.push_back(f);
    return past;
}

}  // namespace

TEST(shape_probe_is_said_only_beside_the_lent_arena) {
    struct Case { const char* p1; const char* lend; bool want; } cases[] = {
        { nullptr, nullptr, false }, { "1", nullptr, false }, { nullptr, "40", false }, { "1", "40", true },
        { "1", "1", true }, { "1", "64", true }, { "1", "0", false }, { "1", "65", false }, { "1", "4x", false },
        { "1", "", false }, { "0", "40", false }, { "2", "40", false },
    };
    for (const Case& c : cases) CHECK_EQ(ds41p1::lend_arena_asked(c.p1, c.lend), c.want);
    /* probe() says what the environment says (P1 is read once a process: unset here, so never). */
    Env off{};
    MetaKV meta = flash_meta();
    RadArchProbe pb{};
    CHECK_EQ(deepseek41::probe(meta.meta(), &pb), RAD_OK);
    CHECK_EQ(pb.shape_probe_ok, ds41p1::lend_arena_asked() ? 1 : 0);
    CHECK_EQ(pb.shape_probe_ok, 0);
}

TEST(a_sizing_declare_lays_the_real_graph_over_smaller_rows) {
    Env e{ WINDOW_SWITCHES };
    for (int rank = 0; rank < kWorld; ++rank) {
        auto real = declare(rank, 2048, false);
        REQUIRE_EQ(real->rc, RAD_OK);
        REQUIRE(real->m != nullptr);
        CHECK(real->m->cut_on);
        CHECK(real->m->pfc.on);
        CHECK(real->m->esplit > 0);
        CHECK(real->m->f8_on);
        CHECK(real->err.find("dense weights read by gemm_nt_bf16f8") != std::string::npos);
        if (rank == 0) CHECK(real->err.find("consumer suffix declared") != std::string::npos);
        for (int64_t rows : { (int64_t)256, (int64_t)1024 }) {
            auto probe = declare(rank, rows, true);
            REQUIRE_EQ(probe->rc, RAD_OK);
            const std::string why = differs(real->b, probe->b);
            CHECK(why.empty());
            if (!why.empty()) std::fprintf(stderr, "  rank %d, %lld rows: %s\n", rank, (long long)rows, why.c_str());
            /* The real declare's model, untouched; the sizing one silent. */
            CHECK(probe->m == real->m);
            CHECK_EQ(real->m->g.max_tok, (int64_t)2048);
            CHECK(!real->m->probe);
            CHECK(probe->err.empty());
            if (!probe->err.empty()) std::fprintf(stderr, "  rank %d printed: %s\n", rank, probe->err.c_str());
            /* Some buffer is smaller: the step-tall ones. */
            const RecBuf* ff = find_b(probe->b, "expert_gate_up");
            REQUIRE(ff != nullptr);
            CHECK_EQ(ff->d.shape[0], rows * 6);
        }
    }
}

TEST(the_sizing_declare_keeps_the_real_index_runs_at_a_long_context) {
    Env e{ WINDOW_SWITCHES };
    /* A million positions: the score plane takes the floor of 256 rows a run, so a 2,048-row step is
     * eight runs, and a sizing declare at 256 rows would be one of its own. */
    auto real = declare(0, 2048, false, 1 << 20);
    REQUIRE_EQ(real->rc, RAD_OK);
    CHECK(real->m->iparts > 1);
    auto probe = declare(0, 256, true, 1 << 20);
    REQUIRE_EQ(probe->rc, RAD_OK);
    const std::string why = differs(real->b, probe->b);
    CHECK(why.empty());
    if (!why.empty()) std::fprintf(stderr, "  %s\n", why.c_str());
}

TEST(compact_index_score_metadata_keeps_32k_and_bounds_512k_1m_without_allocating_them) {
    Env e{ WINDOW_SWITCHES };
    for (int64_t ctx : { int64_t(32768), int64_t(524288), int64_t(1048576) }) {
        unsetenv(deepseek41::kIndexCompactEnv);
        setenv("RADIANCE_EXPERIMENTAL_DS41_LCP", "2048", 1);
        auto base = declare(0, 2048, false, ctx); REQUIRE_EQ(base->rc, RAD_OK);
        const int64_t old_rows = base->m->irows, old_parts = base->m->iparts;
        setenv(deepseek41::kIndexCompactEnv, "1", 1);
        auto compact = declare(0, 2048, false, ctx); REQUIRE_EQ(compact->rc, RAD_OK);
        REQUIRE(compact->m->index_compact);
        const int64_t rows = compact->m->irows;
        const RecBuf* plane = find_b(compact->b, "index_score"); REQUIRE(plane);
        CHECK_EQ(plane->d.shape[0], rows); CHECK_EQ(plane->d.shape[1], ctx + 1);
        CHECK_EQ(rows, ctx == 32768 ? old_rows : ctx == 524288 ? int64_t(127) : int64_t(64));
        CHECK_EQ(compact->m->iparts, (int64_t(2048) + rows - 1) / rows);
        if (ctx == 32768) {
            CHECK_EQ(compact->m->iparts, old_parts);
            CHECK(differs(base->b, compact->b).empty());
            CHECK(differs(compact->b, base->b).empty());
        }
        /* Both large tables still exist at their ORIGINAL full-context extents. */
        for (const char* name : { "rope_window", "rope_compressed" }) {
            const RecBuf* rope = find_b(compact->b, name); REQUIRE(rope);
            CHECK_EQ(rope->d.shape[0], ctx); CHECK_EQ(rope->d.shape[1], int64_t(64));
        }
    }
}

TEST(compact_index_score_requires_lcp_and_sizing_levels_keep_the_real_run_graph) {
    Env e{ WINDOW_SWITCHES };
    setenv(deepseek41::kIndexCompactEnv, "1", 1);
    for (const char* lcp : { (const char*)nullptr, "0", "48", "2049", "4097", "bad" }) {
        if (lcp) setenv("RADIANCE_EXPERIMENTAL_DS41_LCP", lcp, 1);
        else unsetenv("RADIANCE_EXPERIMENTAL_DS41_LCP");
        setenv("RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL", "1", 1);
        auto d = declare(0, 2048, false, 1 << 20); REQUIRE_EQ(d->rc, RAD_OK);
        CHECK(!d->m->index_compact); CHECK_EQ(d->m->irows, int64_t(256));
    }
    setenv("RADIANCE_EXPERIMENTAL_DS41_LCP", "2048", 1);
    auto real = declare(0, 2048, false, 1 << 20); REQUIRE_EQ(real->rc, RAD_OK);
    CHECK_EQ(real->m->irows, int64_t(64)); CHECK_EQ(real->m->iparts, int64_t(32));
    for (int64_t rows : { int64_t(32), int64_t(64), int64_t(256) }) {
        auto probe = declare(0, rows, true, 1 << 20); REQUIRE_EQ(probe->rc, RAD_OK);
        CHECK(differs(real->b, probe->b).empty());
        const RecBuf* plane = find_b(probe->b, "index_score"); REQUIRE(plane);
        CHECK_EQ(plane->d.shape[0], std::min(rows, int64_t(64)));
        CHECK(probe->err.empty());
    }
    setenv(deepseek41::kIndexCompactEnv, "2", 1);
    CHECK_EQ(declare(0, 2048, false, 1 << 20)->rc, RAD_E_INVAL);
}

TEST(compact_index_score_recorded_runs_keep_full_inputs_causal_offsets_and_tail_extents) {
    Env e{ WINDOW_SWITCHES };
    setenv(deepseek41::kIndexCompactEnv, "1", 1);
    setenv("RADIANCE_EXPERIMENTAL_DS41_LCP", "2048", 1);
    unsetenv("RADIANCE_EXPERIMENTAL_DS41_CONSUMER_SKIP");
    for (int64_t ctx : { int64_t(32768), int64_t(524288), int64_t(1048576) }) {
        auto d = declare(0, 2048, false, ctx); REQUIRE_EQ(d->rc, RAD_OK);
        const auto* m = d->m;
        for (int T : { 49, 65, 129, 259 }) {
            auto batch = kit(d->b, {}, {T}, 16);
            RadCtx c; c.b = &d->b; c.batch = &batch->b; c.rank = 0; c.world = kWorld;
            deepseek41::step(&c, &batch->b); CHECK(c.step_fail.empty());
            const int64_t n = (T + m->irows - 1) / m->irows, R = (T + n - 1) / n;
            int64_t scores = 0, topks = 0;
            const auto& L = m->layers[2];
            for (const RecIssue& issue : c.issues) {
                if (std::find(L.i_runs.begin(), L.i_runs.end(), issue.op) != L.i_runs.end()) {
                    REQUIRE_EQ(issue.opd.size(), size_t(7));
                    REQUIRE((size_t)scores < L.i_runs.size());
                    CHECK_EQ(issue.op, L.i_runs[(size_t)scores]); CHECK_EQ(issue.opd[0].rows, int64_t(T));
                    CHECK_EQ(issue.opd[6].rows, R); ++scores;
                }
                if (issue.op == L.i_topk) {
                    REQUIRE_EQ(issue.opd.size(), size_t(4));
                    const int64_t rows = std::min(R, int64_t(T) - topks * R);
                    CHECK_EQ(issue.opd[0].rows, rows); CHECK_EQ(issue.opd[2].rows, rows);
                    CHECK(issue.opd[2].raw == (void*)(batch->b.positions + topks * R));
                    CHECK_EQ(issue.opd[3].rows, rows); CHECK_EQ(issue.opd[3].offset, topks * R * m->c.itopk);
                    ++topks;
                }
            }
            CHECK_EQ(scores, n); CHECK_EQ(topks, n);
            CHECK(overruns(d->b, d->b, batch->b, 0, nullptr).empty());
        }
    }
}

TEST(a_step_of_the_levels_rows_stays_inside_its_buffers) {
    Env e{ WINDOW_SWITCHES };
    for (int rank = 0; rank < kWorld; ++rank) {
        auto real = declare(rank, 2048, false);
        REQUIRE_EQ(real->rc, RAD_OK);
        auto probe = declare(rank, 256, true);
        REQUIRE_EQ(probe->rc, RAD_OK);
        struct Step { std::vector<int> dq, pq; int32_t at; } steps[] = {
            { { 1 }, {}, 0 }, { {}, { 16 }, 0 }, { {}, { 256 }, 0 }, { {}, { 256 }, 1040 }, { {}, { 37 }, 300 },
        };
        for (const Step& st : steps) {
            auto k = kit(real->b, st.dq, st.pq, st.at);
            std::string fail;
            const std::vector<std::string> past = overruns(real->b, probe->b, k->b, rank, &fail);
            CHECK(fail.empty());
            CHECK(past.empty());
            if (!fail.empty() || !past.empty())
                std::fprintf(stderr, "  rank %d, %lld rows: %s%s\n", rank, (long long)k->b.n_tok, fail.c_str(),
                             past.empty() ? "" : past[0].c_str());
        }
        /* The control: a chunk of the real step's rows reaches past the level's buffers. */
        auto big = kit(real->b, {}, { 2048 }, 1040);
        CHECK(!overruns(real->b, probe->b, big->b, rank, nullptr).empty());
        CHECK(overruns(real->b, real->b, big->b, rank, nullptr).empty());
    }
}

TEST(beside_lcp_a_level_sizes_the_compressors_work_rows_for_itself) {
    int64_t level_at[2] = { 0, 0 };
    for (int lcp = 0; lcp < 2; ++lcp) {
        Env e{ WINDOW_SWITCHES };
        if (lcp) setenv("RADIANCE_EXPERIMENTAL_DS41_LCP", "2048", 1);
        for (int rank = 0; rank < kWorld; ++rank) {
            auto real = declare(rank, 2048, false);
            REQUIRE_EQ(real->rc, RAD_OK);
            CHECK_EQ(real->m->work_fit, lcp == 1);
            auto probe = declare(rank, 256, true);
            REQUIRE_EQ(probe->rc, RAD_OK);
            const std::string why = differs(real->b, probe->b);
            CHECK(why.empty());
            if (!why.empty()) std::fprintf(stderr, "  rank %d: %s\n", rank, why.c_str());
            /* The work bound, max_tok + max_seqs: the real one's without the pins, the level's beside them. */
            const RecBuf* st = find_b(probe->b, "comp_stage");
            const RecBuf* pg = find_b(probe->b, "comp_page");
            REQUIRE(st != nullptr && pg != nullptr);
            CHECK_EQ(pg->d.shape[0], lcp ? (int64_t)257 : (int64_t)2049);
            CHECK_EQ(st->d.shape[0], 2 * pg->d.shape[0]);
            struct Step { std::vector<int> dq, pq; int32_t at; } steps[] = {
                { { 1 }, {}, 0 }, { {}, { 16 }, 0 }, { {}, { 256 }, 0 }, { {}, { 256 }, 1041 }, { {}, { 37 }, 300 },
            };
            for (const Step& s : steps) {
                auto k = kit(real->b, s.dq, s.pq, s.at);
                std::string fail;
                const std::vector<std::string> past = overruns(real->b, probe->b, k->b, rank, &fail);
                CHECK(fail.empty());
                CHECK(past.empty());
                if (!fail.empty() || !past.empty())
                    std::fprintf(stderr, "  rank %d, %lld rows: %s%s\n", rank, (long long)k->b.n_tok, fail.c_str(),
                                 past.empty() ? "" : past[0].c_str());
            }
            auto big = kit(real->b, {}, { 2048 }, 1040);
            CHECK(!overruns(real->b, probe->b, big->b, rank, nullptr).empty());
            if (rank) continue;
            int64_t full = 0;
            arena(real->b, probe->b, &full, &level_at[lcp]);
        }
    }
    std::fprintf(stderr, "  T 2048: the 256-row level ends at %.1f MiB, %.1f beside LCP\n", level_at[0] / 1048576.0,
                 level_at[1] / 1048576.0);
    CHECK(level_at[1] < level_at[0]);
}

TEST(lcp4096_keeps_the_small_arena_level_and_full_step_bounds) {
    Env e{ WINDOW_SWITCHES };
    setenv("RADIANCE_EXPERIMENTAL_DS41_LCP", "4096", 1);
    for (int rank = 0; rank < kWorld; ++rank) {
        auto real = declare(rank, 4096, false);
        auto probe = declare(rank, 256, true);
        REQUIRE_EQ(real->rc, RAD_OK);
        REQUIRE_EQ(probe->rc, RAD_OK);
        CHECK(real->m->work_fit);
        CHECK(differs(real->b, probe->b).empty());
        const RecBuf* pg = find_b(probe->b, "comp_page");
        REQUIRE(pg != nullptr);
        CHECK_EQ(pg->d.shape[0], (int64_t)257);
        for (const std::vector<int>& rows : { std::vector<int>{ 1 }, std::vector<int>{ 256 } }) {
            auto small = kit(real->b, {}, rows, 4095);
            std::string fail;
            CHECK(overruns(real->b, probe->b, small->b, rank, &fail).empty());
            CHECK(fail.empty());
        }
        auto large = kit(real->b, {}, { 4096 }, 1024);
        CHECK(!overruns(real->b, probe->b, large->b, rank, nullptr).empty());
        CHECK(overruns(real->b, real->b, large->b, rank, nullptr).empty());
        if (rank == 0) {
            int64_t full = 0, level = 0;
            arena(real->b, probe->b, &full, &level);
            const int64_t room = full - level - (1 << 20);
            CHECK(room / ds41p1::kSlotBytes >= ds41p1::kMaxLend);
            std::fprintf(stderr, "  LCP4096 arena %.1f MiB, level256 %.1f MiB, %lld physical slots; "
                                 "lending cap remains %d\n", full / 1048576.0, level / 1048576.0,
                         (long long)(room / ds41p1::kSlotBytes), ds41p1::kMaxLend);
        }
    }
}

TEST(the_256_row_level_ends_far_below_the_full_plan) {
    Env e{ WINDOW_SWITCHES };
    for (int64_t T : { (int64_t)1024, (int64_t)2048, (int64_t)4096 }) {
        auto real = declare(0, T, false);
        REQUIRE_EQ(real->rc, RAD_OK);
        auto probe = declare(0, 256, true);
        REQUIRE_EQ(probe->rc, RAD_OK);
        int64_t full = 0, level = 0;
        arena(real->b, probe->b, &full, &level);
        const int64_t lent = full - level - (1 << 20);
        std::fprintf(stderr, "  T %lld: arena %.1f MiB, the 256-row level %.1f MiB; %lld slots of %.2f MiB above it and "
                             "the guard (%.1f MiB)\n", (long long)T, full / 1048576.0, level / 1048576.0,
                     (long long)(lent / ds41p1::kSlotBytes), ds41p1::kSlotBytes / 1048576.0, lent / 1048576.0);
        CHECK(level * 3 < full);
        CHECK(lent / ds41p1::kSlotBytes >= T / 64);
        if (T != 2048) continue;
        /* What the rows buy: the largest transients, a row's bytes each. */
        std::vector<std::pair<int64_t, std::string>> big;
        int64_t fixed = 0;
        for (const RecBuf& x : real->b.bufs) {
            if (x.d.domain == RAD_DOMAIN_HOST) continue;
            if (x.d.kind != RAD_BUF_TRANSIENT) fixed += buf_bytes(x.d);
            else big.push_back({ buf_bytes(x.d), x.name });
        }
        std::sort(big.rbegin(), big.rend());
        std::string top;
        for (size_t i = 0; i < big.size() && i < 10; ++i)
            top += say("%s%s %.1f", i ? ", " : "", big[i].second.c_str(), big[i].first / 1024.0 / T);
        std::fprintf(stderr, "  T 2048: KiB a row: %s; persistent %.1f MiB\n", top.c_str(), fixed / 1048576.0);
    }
}

RAD_TEST_MAIN()
