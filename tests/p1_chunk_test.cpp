/* p1_chunk_test.cpp -- prefill Phase 1's end-aligned chunks in Scheduler::plan_chunk (sched/p1_chunk.h;
 * RADIANCE_EXPERIMENTAL_DS41_P1 with ..._P1_ALIGN=end), on the scheduler tests' fakes: every prompt
 * length from 17 to 20,000 at a 2,048-token step, with a checkpoint interval (a linear-state group)
 * and without, the chunks' ends on the quantum's grid, the last chunk full, the interval's snapshots
 * where a chunk ends on it and the prompt's tail snapshot kept. */
#include "sched_fakes.h"

#include <cstdio>
#include <vector>

using namespace rad;
using namespace sched_fake;

namespace {

static const RadParamSpec g_ps[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE },
    { "chunk", RAD_P_INT, RAD_DERIVED,  RAD_PROLE_SEQ_CHUNK },
};
static const RadOpSchema g_schema = { "gdn_chunk_scan", g_ps, 2, nullptr, 0, "test" };
static RadConstraint g_c[] = { RAD_CDIV("chunk", 64) };
static RadKernelInfo g_ki;
static KernelRow     g_row;

struct Fixture {
    Program prog;
    Config  cfg;
    FakeKV  kv;
    FakePC  pc;
    Scheduler sch;
    std::vector<void*> owned;
    ~Fixture() { sch.fini(); for (void* p : owned) std::free(p); }

    void add_derived(const char* name, const char* derive, int64_t n) {
        BufferInfo b;
        b.name = name;
        b.decl.dtype = RAD_I32;
        b.decl.rank = 1;
        b.decl.shape[0] = n;
        b.decl.kind = RAD_BUF_DERIVED;
        b.decl.derive = derive;
        b.bytes = n * 4;
        void* p = std::calloc(1, (size_t)b.bytes);
        owned.push_back(p);
        b.ptr = p;
        prog.buffers.push_back(b);
    }
    /* A paged group, and with `linear` a gated-delta-net group (quantum 64, an interval). */
    int start(bool linear, int64_t max_tok = 2048) {
        g_ki = RadKernelInfo{};
        g_ki.name = "gdn_fake";
        g_ki.op = "gdn_chunked";
        g_ki.constraints = g_c;
        g_ki.n_constraints = 1;
        g_row = KernelRow{ &g_ki, "fake", 0, 0 };
        prog.meta.n_ctx_train = 32768;
        prog.meta.n_vocab = 512;
        prog.ops.emplace_back();
        prog.buffers.emplace_back();
        KVGroupInfo attn;
        attn.name = "attn";
        attn.decl.kind = RAD_KV_FULL;
        attn.block_size = 16;
        prog.kv_groups.push_back(attn);
        if (linear) {
            KVGroupInfo lin;
            lin.name = "gdn";
            lin.decl.kind = RAD_KV_LINEAR;
            lin.decl.state_dim[0] = 128;
            lin.decl.state_dim[1] = 128;
            prog.kv_groups.push_back(lin);
            KVGroupInfo conv;
            conv.name = "conv";
            conv.decl.kind = RAD_KV_CONV;
            conv.decl.conv_width = 4;
            prog.kv_groups.push_back(conv);
            OpInfo op;
            op.op = "gdn_chunk_scan";
            op.schema = &g_schema;
            Band b;
            b.hi = 32768;
            b.dom[RAD_DOMAIN_DEVICE].row = &g_row;
            b.dom[RAD_DOMAIN_DEVICE].geom.set_i("chunk", 64);
            op.bands.push_back(b);
            prog.ops.push_back(op);
            add_derived("seq_start", "seq_start_index", 257);
            add_derived("chunk_bounds", "state_chunk_bounds", 32769);
            add_derived("conv_idx", "conv_state_index", 256);
        }
        cfg.max_tok = max_tok;
        cfg.max_seqs = 1;
        cfg.max_ctx = 24576;
        cfg.checkpoint_interval = 2048;
        cfg.prefix_cache = true;
        cfg.prefix_exact = false;
        cfg.n_spec = 0;
        kv.setup(prog.kv_groups, cfg.max_ctx, 8192, 64);
        pc.kv = &kv;
        pc.groups.clear();
        for (size_t i = 0; i < prog.kv_groups.size(); ++i)
            if (kv.geom.g[i].paged) pc.groups.push_back((int32_t)i);
        std::vector<BatchBuilder::RankIO> rio;
        rio.push_back({ 0, nullptr, &prog });
        return sch.init(prog, cfg, &kv, &pc, rio, nullptr, nullptr);
    }
};

struct Chunks { std::vector<int64_t> n, ckpt; };
Chunks cut(Scheduler& s, int64_t P, int64_t budget) {
    Chunks c;
    for (int64_t at = 0; at < P && c.n.size() < 64;) {
        int64_t ck = -1;
        const int64_t n = s.plan_chunk(at, P, budget, &ck);
        if (n <= 0) break;
        c.n.push_back(n);
        c.ckpt.push_back(ck);
        at += n;
    }
    return c;
}

}  // namespace

TEST(p1_plan_chunk_end_aligns_every_prompt_without_an_interval) {
    Fixture f;
    REQUIRE_EQ(f.start(false), RAD_OK);
    const int64_t C = 2048, g = f.sch.p1_grid();
    CHECK_EQ(g, (int64_t)16);
    f.sch.set_p1_end_align(true, C);
    {
        const Chunks c = cut(f.sch, 3086, C);
        REQUIRE(c.n.size() == (size_t)2);
        CHECK_EQ(c.n[0], (int64_t)1040);
        CHECK_EQ(c.n[1], (int64_t)2046);
    }
    {
        const Chunks c = cut(f.sch, 6817, C);
        REQUIRE(c.n.size() == (size_t)4);
        CHECK_EQ(c.n[0], (int64_t)688);
        CHECK_EQ(c.n[3], (int64_t)2033);
    }
    for (int64_t P = 17; P <= 20000; ++P) {
        const Chunks c = cut(f.sch, P, C);
        int64_t sum = 0;
        bool ok = !c.n.empty() && (int64_t)c.n.size() == (P + C - 1) / C;
        for (size_t i = 0; i < c.n.size(); ++i) {
            sum += c.n[i];
            ok = ok && c.n[i] <= C;
            if (i + 1 < c.n.size()) ok = ok && sum % g == 0;
        }
        ok = ok && sum == P && (P <= C || c.n.back() > C - g);
        if (!ok) {
            CHECK(ok);
            std::fprintf(stderr, "P %lld: %zu chunks\n", (long long)P, c.n.size());
            return;
        }
    }
    /* A shared step (a budget below the chunk) is cut as before. */
    { int64_t ck = -1; CHECK_EQ(f.sch.plan_chunk(0, 3086, 512, &ck), (int64_t)512); }
    /* Off: start-aligned, as without the switch. */
    f.sch.set_p1_end_align(false, C);
    { int64_t ck = -1; CHECK_EQ(f.sch.plan_chunk(0, 3086, C, &ck), C); }
}

TEST(p1_plan_chunk_keeps_the_interval_snapshots_and_the_tail_with_a_linear_group) {
    Fixture f;
    REQUIRE_EQ(f.start(true), RAD_OK);
    const int64_t C = 2048, g = f.sch.p1_grid();
    REQUIRE_EQ(f.sch.geometry().checkpoint_interval, (int64_t)2048);
    CHECK_EQ(g, (int64_t)64);
    f.sch.set_p1_end_align(true, C);
    for (int64_t P = 17; P <= 20000; ++P) {
        const Chunks c = cut(f.sch, P, C);
        int64_t sum = 0;
        bool ok = !c.n.empty() && (int64_t)c.n.size() <= (P + C - 1) / C + 1;   /* + the tail's remainder */
        for (size_t i = 0; i < c.n.size(); ++i) {
            sum += c.n[i];
            ok = ok && c.n[i] >= 1 && c.n[i] <= C;
            if (i + 1 < c.n.size()) ok = ok && sum % g == 0;
            if (c.ckpt[i] >= 0) ok = ok && c.ckpt[i] == sum;                    /* a snapshot where its chunk ends */
            if (i + 1 < c.n.size() && sum % 2048 == 0) ok = ok && c.ckpt[i] == sum;
        }
        ok = ok && sum == P;
        /* The prompt's tail snapshot, where the start-aligned cut writes one, is written here too. */
        Chunks s;
        {
            f.sch.set_p1_end_align(false, C);
            s = cut(f.sch, P, C);
            f.sch.set_p1_end_align(true, C);
        }
        int64_t tail_s = -1, tail_e = -1;
        for (size_t i = 0; i < s.ckpt.size(); ++i) tail_s = std::max(tail_s, s.ckpt[i]);
        for (size_t i = 0; i < c.ckpt.size(); ++i) tail_e = std::max(tail_e, c.ckpt[i]);
        if (tail_s > 0 && tail_s % 2048 != 0) ok = ok && tail_e == tail_s;
        if (!ok) {
            CHECK(ok);
            std::fprintf(stderr, "P %lld: %zu chunks, deepest snapshot %lld (start-aligned %lld)\n", (long long)P,
                         c.n.size(), (long long)tail_e, (long long)tail_s);
            return;
        }
    }
    CHECK(true);
}

RAD_TEST_MAIN()
